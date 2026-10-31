"""Lower a v2 :class:`ExecutionPlan` to an ordered list of PL/host dispatches.

This is a **pure planner**: ``ExecutionPlan`` + block metadata in → an ordered
list of typed :class:`PLOp` records + a residency manifest out. It opens no
sockets and imports no ``pyxrt`` — so it is unit-testable on a laptop and lets
``runner.py`` / ``agent.py`` own transport and execution.

What it decides, once, at plan time:

* **Free monomials** (``permute`` / ``gather``) stay host-side (the corner-turn
  is PS index manipulation, per the locked PL/host split).
* **Elementwise products** (``diagonal_scale`` / ``elementwise_mul``) map to the
  resident ``mul`` CU.
* **ramp** maps to the resident ``ramp`` CU (linear ``ramp_R``/``ramp_b`` only by
  default; ``nonlinear`` folds quad/cubic — the kernel's stubbed extension).
* **fft / ifft**: a 64-point transform maps to the resident ``fft64`` CU;
  ``ifft`` is expanded to the ``conj → fft64 → conj → (1/N) scale`` macro **here,
  in one place**, so the 1/N fold cannot drift (the scale is a host SPF multiply
  by a unit-phase constant — no extra CU). Larger N is **factored onto the 64-pt
  macro** by a Cooley–Tukey split (``_factor_fft64``): an ``N = a·64`` transform
  lowers to ``a``/``64`` resident ``fft64`` leaves + a host reshape/permute/twiddle
  corner-turn — the only path to ``N > 1024`` (which the 64..1024 ``spf32_fft``
  kernel refuses), and ``N = 4096 = 64²`` lowers fully resident (two ``fft64``). A
  non-multiple-of-64 ``N`` still falls back to the single host PS transform.
* **reduce**: STATIC routing runs ``route()`` *here* and **pins** the domain.
  ``TROPICAL`` / ``LOGPOLAR`` → the resident ``reduce`` CU (with the guard
  thresholds pre-converted to Q7.8 for the kernel); ``CARTESIAN`` (pivoted FP64
  Kahan) is structurally impossible on PL → host PS fallback, recorded in
  ``ps_fallback_steps``. The PL kernel never re-routes and never computes R̄.

The PHASE_UNRELIABLE advisory is produced at execution time and rides the
``ReductionResult`` out-of-band — it is never encoded into an SPF word and so
never appears on a :class:`PLOp`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import reducers
from .core import (
    SPF32_L_SCALE,
    AccumulationDomain,
    PrecisionMode,
    ReducerPolicy,
    ReducerSpec,
    RoutingPolicy,
    SPFError,
    encode_words,
)
from .op_registry import lifted_ops
from .v2 import BlockMetadata, ExecutionPlan, ExecutionStep

# Lowering profile -> the bake-ready overlay that carries that profile's CUs.
PROFILE_OVERLAY = {"radar": "spf_tensor_radar_pl", "ofdm": "spf_tensor_ofdm_pl"}

PL_TARGET = "pl"
HOST_TARGET = "host"

# Resident CU op-sets for the two bake profiles of the spf_tensor_pl overlay
# (kv260_bist 5-CU ceiling). ``select_profile`` checks a lowered plan against
# these; a plan needing ops outside either set cannot run on a single profile.
# Radar set carries `mul` (spectral conjugate-multiply for pulse compression —
# the validated PR #287 matched filter), not `reduce`. `spf_reduce` stays a CU
# kernel for tensor reductions + a future phase-slope overlay, but no shipped
# 5-CU overlay resides it (a reduce-only plan -> select_profile() returns None).
PL_OPS_RADAR = frozenset({"fft64", "ramp", "mul", "abs2", "conj"})
PL_OPS_OFDM = frozenset({"fft64", "ramp", "mul", "div", "conj"})

_FFT64_N = 64


@dataclass(frozen=True)
class PLOp:
    """One lowered dispatch.

    ``target`` is ``"pl"`` (resident CU) or ``"host"`` (free numpy / PS
    fallback). ``op`` is the canonical op name. ``reduce_domain`` carries the
    pinned :class:`AccumulationDomain` (int) for reduce ops, else ``None``.
    """

    target: str
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    reduce_domain: int | None = None


@dataclass(frozen=True)
class LoweredPlan:
    ops: tuple[PLOp, ...]
    required_pl_ops: frozenset[str]
    ps_fallback_steps: tuple[int, ...]

    def select_profile(self) -> str | None:
        """First bake profile whose resident CUs cover ``required_pl_ops``.

        Returns ``"radar"`` then ``"ofdm"`` (in that order), or ``None`` if no
        shipped 5-CU overlay covers the plan (e.g. a plan needing ``reduce`` —
        ``spf_reduce`` is a built kernel but not in either shipped overlay).

        Note: radar and ofdm overlap on {fft64, ramp, mul, conj}; the two
        distinguishing CUs (``abs2`` radar-only, ``div`` ofdm-only) are not
        ExecutionPlan-expressible ops, so an EP-lowered plan is covered by radar
        first. A workload that wants on-PL ZF (``div``) must select ofdm
        explicitly; radar runs every other EP-expressible plan identically.
        """
        if self.required_pl_ops <= PL_OPS_RADAR:
            return "radar"
        if self.required_pl_ops <= PL_OPS_OFDM:
            return "ofdm"
        return None



# spf_ramp CU kernel versions. Version 1 folds ramp_R/ramp_b + quad/cubic;
# version 2 (hw/hls/spf_ramp, SPF_RAMP_KERNEL_VERSION 2, 2026-09-17) adds the
# phase_gradient/mixed_ramp s_axilite arguments (ABI v23 full-Hessian jet).
SPF_RAMP_KERNEL_VERSION_HESSIAN = 2
SPF_RAMP_HESSIAN_CAPABILITY = "spf_ramp_hessian"

# Capabilities the *shipped* overlays declare. Both bake-ready overlays were
# linked against spf_ramp kernel version 1, so neither carries the Hessian
# datapath; a re-bake with the version-2 kernel flips its entry here (or the
# caller passes ``capabilities={SPF_RAMP_HESSIAN_CAPABILITY}`` to ``lower``).
OVERLAY_CAPABILITIES: dict[str, frozenset[str]] = {
    "spf_tensor_radar_pl": frozenset(),
    "spf_tensor_ofdm_pl": frozenset(),
}


def overlay_capabilities(overlay: str) -> frozenset[str]:
    """Declared capabilities of a shipped overlay (empty for unknown names)."""

    return OVERLAY_CAPABILITIES.get(str(overlay), frozenset())


def _reject_unlowerable_ramp_terms(params: Any, *, where: str, capable: bool) -> None:
    """Fail closed for ramp terms the target ``spf_ramp`` CU does not implement.

    ``phase_gradient`` and ``mixed_ramp`` (ABI v22 full-Hessian phase jet) are
    executed by the host ``ExecutionPlan``, the C TensorBlock reconstruct and,
    since ABI v23, the ``spf32_phase_ramp_nd_apply_array`` kernel. The PL
    ``spf_ramp`` CU folds them only from kernel version
    ``SPF_RAMP_KERNEL_VERSION_HESSIAN``; an overlay baked earlier must not have
    a plan carrying the new terms silently lowered without them, so the caller
    states whether the target declares ``SPF_RAMP_HESSIAN_CAPABILITY``.

    ``cubic_mixed`` (ABI v24, AlgRef v0.2 §13.6's two-dimensional mixed cubics
    ``x^2 y`` / ``x y^2``) is rejected UNCONDITIONALLY: no baked ``spf_ramp``
    kernel version folds it, and re-synthesising the CU is a separate decision
    (design §12.6 prefers "full quadratic + adaptive subdivision + residual"
    over an always-on incomplete cubic header, so the fabric deliberately has no
    cubic-mixed datapath yet). A plan carrying it runs on the host.
    """

    for key in ("cubic_mixed",):
        values = tuple(int(v) for v in (params.get(key) or ()))
        if any(values):
            raise SPFError(
                f"{where} does not implement ramp {key!r} (AlgRef v0.2 §13.6 mixed "
                "cubics, ABI v24; no spf_ramp kernel version folds them); "
                "execute on the host plan instead"
            )
    if capable:
        return
    for key in ("phase_gradient", "mixed_ramp"):
        values = tuple(int(v) for v in (params.get(key) or ()))
        if any(values):
            raise SPFError(
                f"{where} does not implement ramp {key!r} (needs spf_ramp kernel version "
                f">= {SPF_RAMP_KERNEL_VERSION_HESSIAN}); execute on the host plan instead"
            )

def _guard_params_q78(mode: PrecisionMode) -> dict[str, int]:
    """Guard-layer thresholds in Q7.8 L-code units (kernel s_axilite args)."""
    g = reducers.guard_cfg(mode)
    return {
        "cancel_bits_q78": round(g.cancel_bits * SPF32_L_SCALE),
        "phase_unreliable_bits_q78": round(g.phase_unreliable_bits * SPF32_L_SCALE),
        "dom_bypass_gap_q78": round(g.dom_bypass_gap * SPF32_L_SCALE),
    }


def ifft_scale_word(n: int) -> np.uint32:
    """Unit-phase SPF-32 constant ``(L = -log2(N)·L_SCALE, Θ = 0)`` for the 1/N fold.

    Multiplying by this word scales magnitude by 1/N and leaves phase untouched
    — identical to the scaling inside :func:`spf.fft.spf32_fft.ifft`.
    """
    log2n = round(math.log2(n))
    l_code = np.int16(-log2n * SPF32_L_SCALE)
    return np.uint32(np.uint32(np.uint16(l_code)) << np.uint32(16))


def _resolve_reduce_domain(
    spec: ReducerSpec,
    mode: PrecisionMode,
    stats: reducers.ReductionStats | None,
) -> AccumulationDomain:
    if spec.routing == RoutingPolicy.PINNED:
        if spec.domain is None:
            raise SPFError("PINNED routing requires an explicit ReducerSpec.domain")
        return AccumulationDomain(spec.domain)
    if spec.routing == RoutingPolicy.ADAPTIVE:
        raise SPFError("ADAPTIVE routing is deferred; lower with STATIC or PINNED")
    # STATIC: route from stats at plan time, or fall back to the exact PS path
    # when no stats are available (never silently pick a PL domain blind).
    if stats is None:
        return AccumulationDomain.CARTESIAN
    return reducers.route(stats, mode)


def _twiddle_fusion(
    plan: ExecutionPlan, twiddle_coeffs: frozenset[str]
) -> tuple[dict[int, str], set[int]]:
    """Identify ``diagonal_scale(x, twiddle) -> fft`` pairs to fuse on lowering.

    Returns ``(fft_index -> twiddle name, dropped diagonal_scale indices)``. A
    twiddle multiply immediately before a forward transform folds into the FFT's
    first butterfly stage, so it needs no separate ``mul`` CU dispatch — the same
    "stubbed kernel extension point" the nonlinear ramp uses. Only single-use
    diagonals whose coeff is vouched a twiddle (``twiddle_coeffs``) are fused.
    """
    if not twiddle_coeffs:
        return {}, set()
    prod = {o: i for i, s in enumerate(plan.steps) for o in s.outputs}
    uses: dict[str, int] = {}
    for s in plan.steps:
        for nm in s.inputs:
            uses[nm] = uses.get(nm, 0) + 1
    fused: dict[int, str] = {}
    dropped: set[int] = set()
    for j, s in enumerate(plan.steps):
        if s.op != "fft":
            continue
        di = prod.get(s.inputs[0])
        if di is None or plan.steps[di].op != "diagonal_scale":
            continue
        diag = plan.steps[di]
        if diag.inputs[1] in twiddle_coeffs and uses.get(diag.outputs[0], 0) == 1:
            fused[j] = diag.inputs[1]
            dropped.add(di)
    return fused, dropped


def _ct_can_factor(n: int) -> bool:
    """A length the Cooley–Tukey lowering can split onto 64-pt leaves.

    Requires ``n`` a multiple of 64 and the outer factor ``a = n/64`` itself a
    legal single SPF transform — 64 (a resident ``fft64`` leaf) or a power of two
    in ``[2, 1024]`` (one host ``spf32_fft``). Everything else (non-multiple of 64,
    or a non-power-of-two outer factor the kernel can't transform) stays a single
    host FFT, exactly as before.
    """
    if n <= _FFT64_N or n % _FFT64_N != 0:
        return False
    a = n // _FFT64_N
    return a == _FFT64_N or (a & (a - 1) == 0 and 2 <= a <= 1024)


def _ct_twiddle_words(a: int, b: int) -> np.ndarray:
    """SPF words for the Cooley–Tukey inter-stage twiddle ``exp(-2πi·k1·n2/N)``.

    Shape ``(a, b)`` — the phase ramp applied between the two sub-transforms of an
    ``N = a·b`` split. Encoded once at plan time; multiplied in by a host
    ``mul_words`` op (the kernel folds it into the following ``fft64`` butterfly on
    hardware, the same extension point as :func:`_twiddle_fusion`).
    """
    grid = np.outer(np.arange(a), np.arange(b))
    return encode_words(np.exp(-2j * np.pi * grid / (a * b)))


def _factor_fft64(src: str, dst: str, n: int) -> list[PLOp]:
    """Cooley–Tukey schedule for a forward ``fft`` of size ``n = a·64`` onto fft64.

    Mirrors the T8.94 convention (``x.reshape(a,64)`` → axis-0 FFT → twiddle →
    axis-1 FFT → ``F2ᵀ`` flatten): the inner 64-point transforms map to the
    resident ``fft64`` CU; the size-``a`` axis is another ``fft64`` when ``a == 64``
    (the fully-resident ``N = 4096`` case) or a single host ``fft`` otherwise.
    Reshape / permute / twiddle are host index work. This is the only path to
    ``N > 1024``, which the bare 64..1024 ``spf32_fft`` kernel refuses.
    """
    a, b = n // _FFT64_N, _FFT64_N
    p = f"_{dst}__ct"
    r0, p0, f0, p1, t1, f2, p2 = (f"{p}_r0", f"{p}_p0", f"{p}_f0", f"{p}_p1",
                                  f"{p}_t1", f"{p}_f2", f"{p}_p2")
    tw = _ct_twiddle_words(a, b)
    inner: PLOp = (
        PLOp(PL_TARGET, "fft64", (p0,), (f0,))
        if a == _FFT64_N
        else PLOp(HOST_TARGET, "fft", (p0,), (f0,), {"n": a})
    )
    return [
        PLOp(HOST_TARGET, "reshape", (src,), (r0,), {"shape": (a, b)}),
        PLOp(HOST_TARGET, "permute", (r0,), (p0,), {"axes": (1, 0)}),   # last axis = a
        inner,                                                          # size-a transforms
        PLOp(HOST_TARGET, "permute", (f0,), (p1,), {"axes": (1, 0)}),   # back to (a, b)
        PLOp(HOST_TARGET, "mul_words", (p1,), (t1,), {"words": tw}),    # inter-stage twiddle
        PLOp(PL_TARGET, "fft64", (t1,), (f2,)),                         # size-64 leaves
        PLOp(HOST_TARGET, "permute", (f2,), (p2,), {"axes": (1, 0)}),   # F2ᵀ → (b, a)
        PLOp(HOST_TARGET, "reshape", (p2,), (dst,), {"shape": (n,)}),   # flatten X[k2·a+k1]
    ]


def lower(
    plan: ExecutionPlan,
    *,
    block_meta: BlockMetadata | None = None,
    mode: PrecisionMode | None = None,
    stats_by_step: Mapping[int, reducers.ReductionStats] | None = None,
    twiddle_coeffs: frozenset[str] = frozenset(),
    capabilities: frozenset[str] = frozenset(),
) -> LoweredPlan:
    """Lower ``plan`` to PL/host dispatches.

    ``block_meta`` supplies the O(1) header stats used to pin reduce domains
    (overridable per step via ``stats_by_step``). ``mode`` overrides each
    reducer's ``precision_mode`` when given. ``twiddle_coeffs`` names the
    plan-time coefficients vouched as pure twiddles: a ``diagonal_scale`` by one
    immediately before an ``fft`` is folded into the ``fft64`` dispatch (no
    separate ``mul`` CU) — the lowering-side twiddle pre-multiply fusion.
    ``capabilities`` names what the target overlay declares (see
    :func:`overlay_capabilities`); a ramp carrying ``phase_gradient`` /
    ``mixed_ramp`` lowers only when ``SPF_RAMP_HESSIAN_CAPABILITY`` is present,
    otherwise lowering fails closed.
    """
    ramp_hessian = SPF_RAMP_HESSIAN_CAPABILITY in capabilities
    ops: list[PLOp] = []
    ps_fallback: list[int] = []
    default_stats = reducers.from_block_metadata(block_meta) if block_meta is not None else None
    fused_twiddle, dropped_twiddle = _twiddle_fusion(plan, twiddle_coeffs)
    lifted = lifted_ops()

    for i, step in enumerate(plan.steps):
        if i in dropped_twiddle:
            continue  # folded into the following fft64 (no mul dispatch)
        out0 = step.outputs[0]
        op = step.op
        if op == "permute":
            ops.append(
                PLOp(HOST_TARGET, "permute", step.inputs, step.outputs,
                     {"axes": tuple(step.params["axes"])})
            )
        elif op == "gather":
            # A gather is a host index op regardless of how its indices arise.
            # Static gathers carry a concrete ``indices`` array; index-op gathers
            # (``indices_op`` = ``affine_mod`` / ``sum_mod``, as the character
            # circuit and the CharacterPhaseBind fuse emit) compute their indices
            # from data inputs at run time. Both lower to one host ``gather`` — so
            # carry whichever the step provides instead of demanding ``indices``.
            gparams: dict[str, Any] = {"axis": int(step.params.get("axis", 0))}
            if "indices" in step.params:
                gparams["indices"] = step.params["indices"]
            # Carry the runtime-index recipe through verbatim: the index op and the
            # operands it needs (modulus, affine coeffs, single-axis scale).
            for key in (
                "indices_op", "modulus", "coeffs", "index_coeff",
                "coeffs_a", "coeffs_b", "index_coeff_bank", "moduli",
            ):
                if key in step.params:
                    gparams[key] = step.params[key]
            ops.append(PLOp(HOST_TARGET, "gather", step.inputs, step.outputs, gparams))
        elif op in {"diagonal_scale", "elementwise_mul"}:
            ops.append(PLOp(PL_TARGET, "mul", step.inputs, step.outputs))
        elif op == "ramp":
            # The spf_ramp CU now folds quadratic/cubic phase (ABI v21,
            # spf32_phase_ramp_nd_apply_array), so a nonlinear ramp — the LFM
            # chirp deramp/MF — lowers to the ramp CU instead of falling back to
            # the host. Carry the higher-order coefficients + the nonlinear flag
            # so the kernel evaluates them; linear ramps leave them absent.
            nonlinear = bool(step.params.get("nonlinear", False))
            _reject_unlowerable_ramp_terms(step.params, where="PL spf_ramp CU", capable=ramp_hessian)
            ramp_params: dict[str, Any] = {
                "ramp_R": [list(r) for r in step.params["ramp_R"]],
                "ramp_b": list(step.params["ramp_b"]),
            }
            grad = step.params.get("phase_gradient")
            if grad is not None and any(int(v) for v in grad):
                ramp_params["phase_gradient"] = list(grad)
            if nonlinear:
                mixed = step.params.get("mixed_ramp")
                if mixed is not None and any(int(v) for v in mixed):
                    ramp_params["mixed_ramp"] = list(mixed)
                ramp_params["nonlinear"] = True
                quad = step.params.get("quadratic_ramp")
                cube = step.params.get("cubic_ramp")
                if quad is not None:
                    ramp_params["quadratic_ramp"] = list(quad)
                if cube is not None:
                    ramp_params["cubic_ramp"] = list(cube)
            ops.append(
                PLOp(PL_TARGET, "ramp", step.inputs, step.outputs, ramp_params)
            )
        elif op == "fft":
            n = int(step.params.get("n", _FFT64_N))
            tw = fused_twiddle.get(i)
            if tw is not None:
                # Twiddle folded into the transform: consume the diagonal's data
                # input directly and carry the twiddle as a kernel-side param.
                di = next(k for k, s in enumerate(plan.steps) if step.inputs[0] in s.outputs)
                x_name = plan.steps[di].inputs[0]
                params = {"pre_twiddle": tw, **({"n": n} if n != _FFT64_N else {})}
                fft_op = "fft64" if n == _FFT64_N else "fft"
                target = PL_TARGET if n == _FFT64_N else HOST_TARGET
                ops.append(PLOp(target, fft_op, (x_name,), step.outputs, params))
            elif n == _FFT64_N:
                ops.append(PLOp(PL_TARGET, "fft64", step.inputs, step.outputs))
            elif _ct_can_factor(n):
                # Cooley–Tukey onto the 64-pt macro (and the only path to N>1024).
                ops.extend(_factor_fft64(step.inputs[0], out0, n))
            else:
                ops.append(PLOp(HOST_TARGET, "fft", step.inputs, step.outputs, {"n": n}))
        elif op == "ifft":
            n = int(step.params.get("n", _FFT64_N))
            if n == _FFT64_N:
                ops.extend(_expand_ifft64(step.inputs[0], out0, n))
            elif _ct_can_factor(n):
                # ifft = (1/N)·conj(fft(conj(X))); reuse the factored forward fft.
                c0, fwd = f"_{out0}__ic", f"_{out0}__if"
                ops.append(PLOp(PL_TARGET, "conj", (step.inputs[0],), (c0,)))
                ops.extend(_factor_fft64(c0, fwd, n))
                c1 = f"_{out0}__ic2"
                ops.append(PLOp(PL_TARGET, "conj", (fwd,), (c1,)))
                ops.append(PLOp(HOST_TARGET, "mul_const", (c1,), (out0,),
                                {"const_word": int(ifft_scale_word(n))}))
            else:
                ops.append(PLOp(HOST_TARGET, "ifft", step.inputs, step.outputs, {"n": n}))
        elif op == "reduce":
            assert step.reducer is not None
            spec = step.reducer
            m = mode if mode is not None else spec.precision_mode
            stats = (stats_by_step or {}).get(i, default_stats)
            domain = _resolve_reduce_domain(spec, m, stats)
            params: dict[str, Any] = {
                "axis": step.params.get("axis"),
                "mode": int(m),
                "domain": int(domain),
                **_guard_params_q78(m),
            }
            if domain == AccumulationDomain.CARTESIAN:
                params["reducer_spec"] = spec
                ops.append(
                    PLOp(HOST_TARGET, "reduce", step.inputs, step.outputs, params,
                         reduce_domain=int(domain))
                )
                ps_fallback.append(len(ops) - 1)
            else:
                ops.append(
                    PLOp(PL_TARGET, "reduce", step.inputs, step.outputs, params,
                         reduce_domain=int(domain))
                )
        elif op in {
            "phase_character_features", "phase_shear", "phase_neuron", "monomial_scan", "routed_buckets", "packed_sum"
        }:
            ops.append(PLOp(HOST_TARGET, op, step.inputs, step.outputs, dict(step.params)))
        elif op in lifted:
            # Lifted / oracle contractions stay on the host (PS) for now. Carry the
            # ReducerSpec through (lowering otherwise drops step.reducer) so the
            # reference executor can replay the contraction faithfully.
            params = dict(step.params)
            if step.reducer is not None:
                params.setdefault("reducer_spec", step.reducer)
            ops.append(PLOp(HOST_TARGET, op, step.inputs, step.outputs, params))
        else:  # pragma: no cover - guarded by ExecutionStep validation
            raise SPFError(f"cannot lower execution op {op!r}")

    required = frozenset(o.op for o in ops if o.target == PL_TARGET)
    return LoweredPlan(tuple(ops), required, tuple(ps_fallback))


def _expand_ifft64(src: str, dst: str, n: int) -> list[PLOp]:
    """IFFT-by-composition macro: ``(1/N)·conj(fft64(conj(X)))``.

    Built in one place so the 1/N fold can't drift. conj + fft64 run on resident
    CUs (present in both profiles); the 1/N scale is a host SPF multiply by a
    unit-phase constant, so it needs no extra CU.
    """
    t_conj = f"_{dst}__ic"
    t_fft = f"_{dst}__if"
    t_conj2 = f"_{dst}__ic2"
    return [
        PLOp(PL_TARGET, "conj", (src,), (t_conj,)),
        PLOp(PL_TARGET, "fft64", (t_conj,), (t_fft,)),
        PLOp(PL_TARGET, "conj", (t_fft,), (t_conj2,)),
        PLOp(HOST_TARGET, "mul_const", (t_conj2,), (dst,),
             {"const_word": int(ifft_scale_word(n))}),
    ]


def _reduce_along_axis(arr: np.ndarray, op: PLOp) -> np.ndarray | np.uint32:
    spec = op.params.get("reducer_spec")
    if spec is None:
        spec = ReducerSpec(
            policy=ReducerPolicy.AUTO,
            routing=RoutingPolicy.PINNED,
            domain=AccumulationDomain(op.reduce_domain),
            precision_mode=PrecisionMode(op.params["mode"]),
        )
    axis = op.params.get("axis")
    return reducers.reduce_axis(arr, spec, None if axis is None else int(axis))


def _product_strides(moduli: Sequence[int]) -> list[int]:
    """Positional strides for flat mixed-radix indexing of ``∏ Z/nᵢ`` (last factor fastest)."""
    strides = [1] * len(moduli)
    for i in range(len(moduli) - 2, -1, -1):
        strides[i] = strides[i + 1] * int(moduli[i + 1])
    return strides


def _product_sum_mod(idx_a: np.ndarray, idx_b: np.ndarray, moduli: Sequence[int]) -> np.ndarray:
    """Componentwise ``(a ⊕ b) mod (n₁,…,n_d)`` on flat mixed-radix indices.

    Decode each flat index to its per-factor components, add modulo each factor (no
    carry between factors), re-encode. This is the product-group character bind that
    :class:`~spf.tensor.rewrite.ProductGroupCharacterBind` fuses to.
    """
    strides = _product_strides(moduli)
    out = np.zeros(np.broadcast(idx_a, idx_b).shape, dtype=np.int64)
    for i, n in enumerate(moduli):
        ca = (idx_a // strides[i]) % int(n)
        cb = (idx_b // strides[i]) % int(n)
        out = out + ((ca + cb) % int(n)) * strides[i]
    return out


def _bank_coeffs(coeffs: Any, idx_ndim: int) -> np.ndarray:
    """A ``(k,)`` coefficient vector reshaped to ``(k,) + (1,) * idx_ndim``.

    Lets a per-atom bank coefficient broadcast against an index array of *any*
    rank: the bank axis is prepended and the original index shape is kept intact,
    so a flat ``(M,)`` vector yields ``(k, M)`` and a shaped ``(N, N)`` grid yields
    ``(k, N, N)`` — not just the ``k == N`` accident the old ``[:, None]`` allowed.
    """
    return np.asarray(coeffs, dtype=np.int64).reshape((-1,) + (1,) * idx_ndim)


def _lowered_gather_indices(op: PLOp, env: Mapping[str, Any]) -> np.ndarray:
    """Resolve a host ``gather``'s index vector from one of three sources.

    A lowered ``gather`` reads ``env[inputs[0]]`` (the table); ``lower`` carries
    through whichever index form the plan step had:

    * **static** — ``params["indices"]`` is a plan-time constant vector (the
      ordinary reindex / DFT-reversal gather);
    * **``indices_op="sum_mod"``** — the :class:`~spf.tensor.rewrite.CharacterPhaseBind`
      fused sum bind ``χ_f((a+b) mod N)``: combine the two *runtime* index inputs as
      ``(idx_a + idx_b) mod modulus`` (``inputs[1]``, ``inputs[2]``);
    * **``indices_op="affine_mod"``** — the affine/difference fused bind
      ``χ_g((αa+βb) mod N)``: ``(α·idx_a + β·idx_b) mod modulus`` with
      ``params["coeffs"] = [α, β]`` (the ``a-b``, ``a+2b``, ``7a-7b`` family);
    * **``indices_op="affine_bank_mod"``** — the :class:`~spf.tensor.rewrite.CharacterBankBind`
      fused *bank* bind: a sparse 2-D character surface's k atoms at once. With
      coefficient vectors ``coeffs_a=[α_j]``, ``coeffs_b=[β_j]`` the index is the
      ``(k, M)`` array ``(α_j·a_m + β_j·b_m) mod N`` so the gather yields the whole
      ``χ_{(u_j,v_j)}`` bank stacked on axis 0;
    * **plain runtime input** — no index op: the un-fused per-axis character gather
      ``table[(c·idx) mod N]`` reads its indices from ``inputs[1]`` and applies the
      optional per-axis scale ``params["index_coeff"] = c`` (``c=1`` / absent for
      the bare reindex gather), or a per-atom scale vector ``index_coeff_bank``
      that broadcasts to a ``(k, M)`` bank.

    Refuses an unrecognised form loudly rather than silently mis-gathering — the
    documented reference path stays faithful instead of ``KeyError``-ing on a
    fused/character plan.
    """
    if "indices" in op.params:
        return np.asarray(op.params["indices"], dtype=np.int64)
    indices_op = op.params.get("indices_op")
    if indices_op == "sum_mod":
        modulus = int(op.params["modulus"])
        idx_a = np.asarray(env[op.inputs[1]], dtype=np.int64)
        idx_b = np.asarray(env[op.inputs[2]], dtype=np.int64)
        return (idx_a + idx_b) % modulus
    if indices_op == "affine_mod":
        modulus = int(op.params["modulus"])
        alpha, beta = (int(c) for c in op.params["coeffs"])
        idx_a = np.asarray(env[op.inputs[1]], dtype=np.int64)
        idx_b = np.asarray(env[op.inputs[2]], dtype=np.int64)
        return (alpha * idx_a + beta * idx_b) % modulus
    if indices_op == "product_sum_mod":
        # Product group ∏ Z/nᵢ: flat mixed-radix indices add COMPONENTWISE (no carry
        # across factors), unlike sum_mod's single-modulus add.
        moduli = [int(m) for m in op.params["moduli"]]
        idx_a = np.asarray(env[op.inputs[1]], dtype=np.int64)
        idx_b = np.asarray(env[op.inputs[2]], dtype=np.int64)
        return _product_sum_mod(idx_a, idx_b, moduli)
    if indices_op == "affine_bank_mod":
        modulus = int(op.params["modulus"])
        idx_a = np.asarray(env[op.inputs[1]], dtype=np.int64)
        idx_b = np.asarray(env[op.inputs[2]], dtype=np.int64)
        # Prepend a bank axis to the *original* index shape so a shaped grid
        # (e.g. N x N) still broadcasts: u is (k, 1, ..., 1), idx is (1, *S).
        u = _bank_coeffs(op.params["coeffs_a"], idx_a.ndim)
        v = _bank_coeffs(op.params["coeffs_b"], idx_b.ndim)
        return (u * idx_a[None, ...] + v * idx_b[None, ...]) % modulus  # (k, *S)
    if indices_op is None and len(op.inputs) >= 2:
        idx = np.asarray(env[op.inputs[1]], dtype=np.int64)
        bank = op.params.get("index_coeff_bank")
        if bank is not None:
            u = _bank_coeffs(bank, idx.ndim)  # (k, 1, ..., 1)
            return (u * idx[None, ...]) % int(op.params["modulus"])  # (k, *S)
        coeff = op.params.get("index_coeff")
        if coeff is not None:
            idx = (int(coeff) * idx) % int(op.params["modulus"])
        return idx
    raise NotImplementedError(
        f"execute_lowered: gather has no resolvable indices "
        f"(indices_op={indices_op!r}, inputs={op.inputs!r})"
    )


def execute_lowered(plan: LoweredPlan, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Reference executor: run a :class:`LoweredPlan` via SPF Python equivalents.

    This is the no-hardware path — PL ops run through their bit-exact (mul/conj/
    ramp) or envelope-exact (fft64/reduce) Python references, host ops via numpy.
    It proves the lowering is semantically faithful end to end (``execute_lowered
    (lower(plan)) == plan.execute(plan_inputs)`` for a PINNED-reducer plan) and is
    the structural template the board backend follows, swapping PL ops for real
    dispatches. ``runner.py``'s ``kria_pl_plan`` backend uses it in reference mode.
    """
    from spf.fft import spf32_fft

    from .core import spf_conj_words, spf_mul_words

    env: dict[str, Any] = dict(inputs)
    for op in plan.ops:
        o0 = op.outputs[0]
        if op.op == "permute":
            env[o0] = np.transpose(env[op.inputs[0]], axes=tuple(op.params["axes"]))
        elif op.op == "gather":
            env[o0] = np.take(
                env[op.inputs[0]],
                _lowered_gather_indices(op, env),
                axis=int(op.params.get("axis", 0)),
            )
        elif op.op == "mul":
            env[o0] = spf_mul_words(env[op.inputs[0]], env[op.inputs[1]])
        elif op.op == "mul_const":
            env[o0] = spf_mul_words(env[op.inputs[0]], np.uint32(op.params["const_word"]))
        elif op.op == "mul_words":
            env[o0] = spf_mul_words(
                env[op.inputs[0]], np.asarray(op.params["words"], dtype=np.uint32)
            )
        elif op.op == "reshape":
            env[o0] = np.asarray(env[op.inputs[0]], dtype=np.uint32).reshape(
                tuple(op.params["shape"])
            )
        elif op.op == "conj":
            env[o0] = spf_conj_words(env[op.inputs[0]])
        elif op.op == "ramp":
            ramp_params: dict[str, Any] = {
                "ramp_R": op.params["ramp_R"],
                "ramp_b": op.params["ramp_b"],
                "nonlinear": op.params.get("nonlinear", False),
            }
            # Forward the quad/cubic folds so the lowered nonlinear ramp matches
            # the host executor (spf_ramp CU evaluates them on fabric, ABI v21).
            if "quadratic_ramp" in op.params:
                ramp_params["quadratic_ramp"] = op.params["quadratic_ramp"]
            if "cubic_ramp" in op.params:
                ramp_params["cubic_ramp"] = op.params["cubic_ramp"]
            # The lowered op only carries the Hessian terms when ``lower`` was
            # told the target overlay implements them; the host reference
            # (ExecutionPlan) folds them like the version-2 CU.
            if "phase_gradient" in op.params:
                ramp_params["phase_gradient"] = op.params["phase_gradient"]
            if "mixed_ramp" in op.params:
                ramp_params["mixed_ramp"] = op.params["mixed_ramp"]
            ramp_step = ExecutionStep("ramp", ("x",), ("y",), params=ramp_params)
            env[o0] = ExecutionPlan(steps=(ramp_step,)).execute({"x": env[op.inputs[0]]})["y"]
        elif op.op in {"fft64", "fft"}:
            x = np.asarray(env[op.inputs[0]], dtype=np.uint32)
            # A fused pre-twiddle (see _twiddle_fusion) carries the coeff env-var
            # name: the diagonal_scale was dropped on lowering, so the multiply
            # the kernel folds into the first butterfly must be reapplied here, or
            # the reference would compute fft(x) instead of fft(twiddle ⊙ x).
            tw = op.params.get("pre_twiddle")
            if tw is not None:
                x = spf_mul_words(x, np.asarray(env[tw], dtype=np.uint32))
            env[o0] = spf32_fft.fft(x)
        elif op.op == "ifft":
            env[o0] = spf32_fft.ifft(np.asarray(env[op.inputs[0]], dtype=np.uint32))
        elif op.op == "reduce":
            env[o0] = _reduce_along_axis(np.asarray(env[op.inputs[0]], dtype=np.uint32), op)
        elif op.op == "packed_sum":
            from .phase_clock import packed_pairwise
            env[op.outputs[0]] = packed_pairwise(np.stack([env[name] for name in op.inputs], axis=-1))
        elif op.op in {"phase_character_features", "phase_shear", "phase_neuron"}:
            step = ExecutionStep(op.op, op.inputs, op.outputs, params=dict(op.params))
            env[o0] = ExecutionPlan(steps=(step,)).execute({name: env[name] for name in op.inputs})[o0]
        elif op.op == "monomial_scan":
            from .monomial_scan import execute_monomial_scan_step

            env.update(execute_monomial_scan_step(env, op.inputs, op.outputs, op.params))
        elif op.op == "routed_buckets":
            from .routed_buckets import execute_routed_buckets_step

            env.update(execute_routed_buckets_step(env, op.inputs, op.outputs, op.params))
        elif op.op in {"low_rank_apply", "decoded_contract_reference", "contract"}:
            # Replay the lifted contraction through the v2 reference executor: a
            # one-step ExecutionPlan keeps the SPF-word low-rank/contraction math in
            # one place rather than duplicating it here (mirrors the ramp branch).
            params = {k: v for k, v in op.params.items() if k != "reducer_spec"}
            reducer = op.params.get("reducer_spec") or ReducerSpec(policy=ReducerPolicy.CARTESIAN_KAHAN)
            step = ExecutionStep(op.op, op.inputs, op.outputs, reducer=reducer, params=params)
            sub = {name: env[name] for name in op.inputs}
            env[o0] = ExecutionPlan(steps=(step,)).execute(sub)[o0]
        else:  # pragma: no cover
            raise SPFError(f"execute_lowered: unhandled lowered op {op.op!r}")
    return env


def block_meta_from_words(words: Sequence[int] | np.ndarray) -> BlockMetadata:
    """Convenience: derive :class:`BlockMetadata` from a packed word stream.

    For callers/tests that hold the actual words at plan time rather than a
    precomputed header. Reuses the v2 plane-based metadata builder.
    """
    from .core import unpack_L, unpack_theta
    from .v2 import block_metadata_from_planes

    w = np.asarray(words, dtype=np.uint32).reshape(-1)
    return block_metadata_from_planes(unpack_L(w), unpack_theta(w))
