"""SPFTensor v3 ITERATED_OPERATOR block (Axis 2) — a compositional generator.

This is the primitive the rest of the stack was missing: closure under
**composition in the depth/time direction**. Where ``SEPARABLE_PHASE`` stores a
*scalar* phase generator and accumulates it along one index
(``Θ[p] = Θ[p−1] + ω``), ``ITERATED_OPERATOR`` stores an **operator** generator
``A`` and materialises its **orbit** ``{A^k · x₀}`` along a depth axis. It is
``SEPARABLE_PHASE`` lifted one order: in the eigenbasis the two coincide — ``A``'s
eigenphases *are* the per-mode ``ω``, and the per-mode log-magnitude rate is
``log₂|λ|`` — so a productively-iterated linear map is stored once (its spectrum,
``O(rank)``) and replayed to any depth ``k`` in closed form, independent of ``k``.

Two reconstruct paths:

* **closed form (diagonal)** — store the generator's spectrum
  (``eigenphase``, ``logmag_rate``) plus the seed's modal coordinates
  (``seed_L``, ``seed_T``). ``Θ[k, c] = Θ₀[c] + k·ωc`` is the same in-ring phase
  accumulator as ``SEPARABLE_PHASE`` (PAN's pure-add primitive, now over layers),
  and ``L[k, c] = L₀[c] + k·gc`` is the dual log-magnitude ramp. ``O(rank)`` bits
  for any depth — this is the *productive* store.
* **explicit iteration** — store a reference to a generator **block** (a
  ``JACOBIAN_OPERATOR``) plus a complex seed, and reconstruct ``x_k = A·x_{k−1}``
  materially. The honest fallback when ``A`` is not cleanly diagonalisable or the
  productivity gate declines closed-form storage.

The **productivity gate** (:func:`productivity_gate`) is the catalogued-vs-computed
decision made mechanical: a generator whose eigenphases are *stable across depth*
(low continuity CV) and *actually turn* is productively iterating → store as a
generator; a static direction that does not propagate through depth declines to a
flat per-layer store. See ``docs/composition-closure.md``.

Registration happens at import; :mod:`spf.tensor.v3` triggers it lazily.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from spf.scalar._constants import SPF32 as _SPF32  # scalar Format (l_scale, l_code_zero)

from .base_models import accumulate_phase_ring
from .core import (
    SPFError,
    decode_complex,
    encode_complex,
)
from .v3 import (
    BaseModelKind,
    BaseModelRef,
    Representation,
    RepresentationRef,
    ResidualCodecKind,
    ResidualRef,
    SPFBlockV3,
    block_to_complex,
    register_base_model,
)

_TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# Spectrum / generator helpers.
# --------------------------------------------------------------------------- #


def operator_spectrum(
    A: np.ndarray, *, phase_bits: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigendecompose a step operator ``A`` into a stored phase spectrum.

    Returns ``(eigenphase, logmag_rate, eigvecs)``:

    * ``eigenphase[c]`` — ``∠λ_c`` as a ``2^B`` phase-ring increment (the per-step
      rotation of mode ``c``; this is the operator generalisation of a
      ``SEPARABLE_PHASE`` ``ω``).
    * ``logmag_rate[c]`` — ``log₂|λ_c|`` scaled to the SPF-32 L fixed-point (the
      per-step magnitude ramp; ``0`` ⇔ unit modulus, a pure rotation).
    * ``eigvecs`` — the modal basis ``V`` (columns are eigenvectors), needed to map
      a seed into / out of modal coordinates.
    """
    A = np.asarray(A, dtype=np.complex128)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise SPFError(f"operator_spectrum requires a square matrix, got shape {A.shape}")
    lam, V = np.linalg.eig(A)
    mod = 1 << int(phase_bits)
    eigenphase = np.rint((np.angle(lam) % _TWO_PI) * (mod / _TWO_PI)).astype(np.int64) % mod
    mag = np.abs(lam)
    # log₂|λ| · l_scale (SPF-32 L fixed-point, l_scale = 256). Unit modulus → 0.
    rate = np.zeros_like(mag)
    nz = mag > 0.0
    rate[nz] = np.log2(mag[nz])
    logmag_rate = np.rint(rate * _SPF32.l_scale).astype(np.int64)
    return eigenphase, logmag_rate, V


# --------------------------------------------------------------------------- #
# The base model.
# --------------------------------------------------------------------------- #


class IteratedOperatorModel:
    """Axis-2 base model for the orbit ``{A^k · x₀}`` of a referenced operator.

    ``params``:

    Closed-form (diagonal) store — the productive generator:
      * ``closed_form``   — ``True``.
      * ``eigenphase``    — int64[r] per-mode per-step phase increment (ring code).
      * ``logmag_rate``   — int64[r] per-mode per-step L ramp (``log₂|λ|``-scaled).
      * ``seed_L`` / ``seed_T`` — int64[r] the seed's modal L / Θ codes at ``k=0``.
      * ``phase_bits``    — ring width (default 16, SPF-32).
      * ``depth_axis`` / ``mode_axis`` — which block dim is depth / mode (default 0/1).

    Explicit-iteration store — the honest fallback:
      * ``closed_form``     — ``False``.
      * ``generator_block`` — a nested :class:`SPFBlockV3` (a ``JACOBIAN_OPERATOR``)
        whose reconstruction is the step matrix ``A``.
      * ``seed_re`` / ``seed_im`` — float64[n] the complex seed ``x₀``.

    The base contributes the orbit directly as the ``(L, Θ)`` planes (residual is
    ``NONE``), exactly the ``SEPARABLE_PHASE`` "law generates the planes" pattern.
    """

    kind = int(BaseModelKind.ITERATED_OPERATOR)

    def reconstruct_planes(
        self,
        params: Mapping[str, Any],
        dims: tuple[int, ...],
        origin: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(dims) != 2:
            raise SPFError(
                f"ITERATED_OPERATOR block requires a 2-D (depth, mode) grid, got dims={tuple(dims)!r}"
            )
        depth_axis = int(params.get("depth_axis", 0))
        mode_axis = int(params.get("mode_axis", 1))
        if {depth_axis, mode_axis} != {0, 1}:
            raise SPFError("ITERATED_OPERATOR depth_axis/mode_axis must be a permutation of (0, 1)")
        steps = int(dims[depth_axis])
        n_mode = int(dims[mode_axis])

        if params.get("closed_form", True):
            gridL, gridT = self._closed_form_grid(params, steps, n_mode)
        else:
            gridL, gridT = self._explicit_grid(params, steps, n_mode)

        # gridL/gridT are (steps, n_mode). Lay them out in the block's dim order.
        if depth_axis == 0:
            outL, outT = gridL, gridT
        else:
            outL, outT = gridL.T, gridT.T
        n = steps * n_mode
        active = np.arange(n, dtype=np.int64)
        return active, outL.reshape(-1).astype(np.int64), outT.reshape(-1).astype(np.int64)

    def _closed_form_grid(
        self, params: Mapping[str, Any], steps: int, n_mode: int
    ) -> tuple[np.ndarray, np.ndarray]:
        phase_bits = int(params.get("phase_bits", 16))
        omega = np.asarray(params["eigenphase"], dtype=np.int64).reshape(-1)
        rate = np.asarray(params["logmag_rate"], dtype=np.int64).reshape(-1)
        seed_L = np.asarray(params["seed_L"], dtype=np.int64).reshape(-1)
        seed_T = np.asarray(params["seed_T"], dtype=np.int64).reshape(-1)
        for name, arr in (("eigenphase", omega), ("logmag_rate", rate), ("seed_L", seed_L)):
            if arr.size != n_mode:
                raise SPFError(
                    f"ITERATED_OPERATOR {name} length {arr.size} != mode count {n_mode}"
                )
        # Θ[k, c] = (seed_T[c] + k·ωc) mod 2^B — the in-ring phase accumulator, but
        # ω is now the operator's eigenphase spectrum (SEPARABLE_PHASE lifted).
        gridT = accumulate_phase_ring(omega, steps, bias=seed_T, phase_bits=phase_bits)
        # L[k, c] = seed_L[c] + k·gc — the dual log-magnitude ramp. A zero-magnitude
        # seed mode (L == zero sentinel) stays exactly zero for the whole orbit;
        # ramping the sentinel would corrupt it into a finite magnitude.
        k = np.arange(steps, dtype=np.int64).reshape(-1, 1)
        gridL = seed_L.reshape(1, -1) + k * rate.reshape(1, -1)
        zero = seed_L.reshape(1, -1) == int(_SPF32.l_code_zero)
        gridL = np.where(zero, int(_SPF32.l_code_zero), gridL)
        gridT = np.where(zero, 0, gridT)
        return gridL, gridT

    def _explicit_grid(
        self, params: Mapping[str, Any], steps: int, n_mode: int
    ) -> tuple[np.ndarray, np.ndarray]:
        A = self._resolve_generator(params)
        if A.shape != (n_mode, n_mode):
            raise SPFError(
                f"ITERATED_OPERATOR generator is {A.shape}, expected ({n_mode}, {n_mode})"
            )
        x = np.asarray(params["seed_re"], dtype=np.float64) + 1j * np.asarray(
            params["seed_im"], dtype=np.float64
        )
        if x.shape != (n_mode,):
            raise SPFError(f"ITERATED_OPERATOR seed is {x.shape}, expected ({n_mode},)")
        orbit = np.empty((steps, n_mode), dtype=np.complex128)
        cur = x.astype(np.complex128)
        for kk in range(steps):
            orbit[kk] = cur
            cur = A @ cur  # x_{k+1} = A · x_k — the iterated composition
        gridL, gridT = encode_complex(orbit)
        return gridL.astype(np.int64), gridT.astype(np.int64)

    @staticmethod
    def _resolve_generator(params: Mapping[str, Any]) -> np.ndarray:
        gen_block = params.get("generator_block")
        if isinstance(gen_block, SPFBlockV3):
            return block_to_complex(gen_block)
        gen = params.get("generator")
        if gen is not None:
            return np.asarray(gen, dtype=np.complex128)
        raise SPFError(
            "ITERATED_OPERATOR explicit store needs a 'generator_block' (a JACOBIAN_OPERATOR "
            "block) or an inline 'generator' matrix"
        )

    def storage_bits(self, params: Mapping[str, Any]) -> int:
        if params.get("closed_form", True):
            # The whole point: O(rank) and INDEPENDENT of depth k. One spectrum
            # (eigenphase + logmag_rate) + the seed's modal coords.
            r = int(np.asarray(params.get("eigenphase", ())).size)
            return 4 * r * 16 + 16  # 4 int16-ish vectors of length r + axes/flags
        # Explicit store: the generator block is charged by its own storage_bits
        # (the flat per-layer matrix), plus the complex seed.
        bits = 16
        gen_block = params.get("generator_block")
        if isinstance(gen_block, SPFBlockV3):
            bits += gen_block.storage_bits()
        elif params.get("generator") is not None:
            bits += int(np.asarray(params["generator"]).size) * 64
        for key in ("seed_re", "seed_im"):
            if params.get(key) is not None:
                bits += int(np.asarray(params[key]).size) * 32
        return bits


# --------------------------------------------------------------------------- #
# Productivity gate — catalogued-vs-computed made mechanical.
# --------------------------------------------------------------------------- #


def productivity_gate(
    trajectory: np.ndarray,
    *,
    resultant_min: float = 0.85,
    min_turn: float = 0.02,
    cv_threshold: float = 0.30,
) -> dict[str, Any]:
    """Decide whether a depth-trajectory iterates a stable generator.

    ``trajectory`` is a complex array ``(steps, r)`` — the modal coordinates of a
    structure read out at each depth. Per mode, the per-step rotation is estimated
    modular-safely from the first difference ``Δθ_k = ∠(z_{k+1} · z̄_k)``. The primary
    stability statistic is the **circular concentration** ``R = |mean unit-increment|``
    ∈ [0, 1]: ``R ≈ 1`` ⇔ a steady rotation (clean generator) at *any* speed, ``R ≈ 0``
    ⇔ no consistent rotation. (The continuity ``CV = circ_std/|turn|`` is also reported
    but is speed-sensitive — a clean *slow* clock has large CV despite ``R ≈ 1`` — so
    the gate keys on ``R``, not ``CV``.)

    * **passes** (``store closed-form as a generator``) iff the rotation is stable
      (median ``R > resultant_min``) **and** actually turns (median ``|Δθ| > min_turn``).
      The *computed* class: position, modular arithmetic, productive morphology — a
      generator the computation iterates.
    * **fails** (``decline to a flat per-layer store``) when the direction is static
      (``|Δθ| ≈ 0``) or unstable (``R`` low). The *catalogued/lexical* class: a clock in
      storage that never composes through the computation (calendar), or no clock.

    Returns ``{passed, resultant_median, turn_median, cv_median, per_mode_resultant,
    per_mode_turn, per_mode_cv, n_steps}``.
    """
    z = np.asarray(trajectory, dtype=np.complex128)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.ndim != 2:
        raise SPFError("productivity_gate trajectory must be (steps, r)")
    steps, r = z.shape
    if steps < 3:
        # Two states give exactly one increment — no spread to measure, so the
        # continuity test is vacuous. Refuse rather than pass on no evidence.
        return {
            "passed": False,
            "reason": "need ≥3 depth steps to measure eigenphase continuity",
            "resultant_median": float("nan"),
            "turn_median": float("nan"),
            "cv_median": float("nan"),
            "per_mode_resultant": np.full(r, np.nan),
            "per_mode_turn": np.full(r, np.nan),
            "per_mode_cv": np.full(r, np.nan),
            "n_steps": int(steps),
        }
    # Per-step increment (unit complex): z_{k+1} · conj(z_k), normalised. A mode
    # that is ~zero at some depth has no defined rotation there — guard the norm.
    prod = z[1:] * np.conj(z[:-1])
    mag = np.abs(prod)
    safe = mag > 0
    incr = np.where(safe, prod / np.where(safe, mag, 1.0), 0.0)  # (steps-1, r) unit phasors
    # Mean resultant per mode (circular). |mean| near 1 ⇒ consistent rotation.
    mean_vec = incr.mean(axis=0)
    turn = np.angle(mean_vec)  # mean per-step rotation, rad
    resultant = np.abs(mean_vec)  # circular concentration in [0, 1] — speed-independent
    resultant_c = np.clip(resultant, 1e-12, 1.0)
    circ_std = np.sqrt(-2.0 * np.log(resultant_c))
    turn_mag = np.abs(turn)
    per_mode_cv = circ_std / np.where(turn_mag > 1e-9, turn_mag, np.nan)
    resultant_median = float(np.median(resultant))
    cv_median = float(np.nanmedian(per_mode_cv))
    turn_median = float(np.nanmedian(turn_mag))
    passed = bool(turn_median > min_turn and resultant_median > resultant_min)
    return {
        "passed": passed,
        "resultant_median": resultant_median,
        "turn_median": turn_median,
        "cv_median": cv_median,
        "per_mode_resultant": resultant,
        "per_mode_turn": turn_mag,
        "per_mode_cv": per_mode_cv,
        "n_steps": int(steps),
    }


# --------------------------------------------------------------------------- #
# Constructors + readers.
# --------------------------------------------------------------------------- #


def build_iterated_operator_block(
    *,
    generator: np.ndarray | None = None,
    generator_block: SPFBlockV3 | None = None,
    seed: np.ndarray,
    steps: int,
    closed_form: bool = True,
    gate_trajectory: np.ndarray | None = None,
    phase_bits: int = 16,
    representation: int = int(Representation.POLAR_SPF32),
    generator_ref: Mapping[str, Any] | None = None,
) -> SPFBlockV3:
    """Build an ``ITERATED_OPERATOR`` block from a step operator and a seed.

    Provide the generator as a dense matrix ``generator`` **or** as a nested
    ``generator_block`` (a ``JACOBIAN_OPERATOR`` — this is the structural change:
    a block whose payload references another block). ``seed`` is the initial state
    ``x₀`` (complex, length ``n``); the block reconstructs ``{x₀, A·x₀, …, A^{steps-1}·x₀}``.

    With ``closed_form=True`` (default) the generator is eigendecomposed and only
    its **spectrum** is stored — ``O(n)`` bits for any ``steps``. If
    ``gate_trajectory`` is supplied, :func:`productivity_gate` is consulted first
    and a **declining** gate forces the explicit (flat) store, so a non-productive
    generator is never silently compressed into a closed form it does not earn.
    """
    if (generator is None) == (generator_block is None):
        raise SPFError("provide exactly one of generator / generator_block")
    A = block_to_complex(generator_block) if generator_block is not None else np.asarray(
        generator, dtype=np.complex128
    )
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise SPFError(f"generator must be a square matrix, got shape {A.shape}")
    n = A.shape[0]
    x0 = np.asarray(seed, dtype=np.complex128).reshape(-1)
    if x0.size != n:
        raise SPFError(f"seed length {x0.size} != operator dim {n}")
    steps = int(steps)
    if steps < 1:
        raise SPFError("steps must be ≥ 1")

    if closed_form and gate_trajectory is not None:
        gate = productivity_gate(gate_trajectory)
        if not gate["passed"]:
            closed_form = False  # gate declined: a static/unstable generator → flat store

    meta: dict[str, Any] = {
        "closed_form": bool(closed_form),
        "depth_axis": 0,
        "mode_axis": 1,
        "steps": steps,
    }
    if generator_ref is not None:
        meta["generator_ref"] = dict(generator_ref)

    if closed_form:
        eigenphase, logmag_rate, V = operator_spectrum(A, phase_bits=phase_bits)
        # Seed in modal coordinates: s = V⁻¹ x₀ so x_k = V diag(λ^k) s, and the
        # block's "modes" are the eigenmodes (the clock is A's eigenphases).
        s = np.linalg.solve(V, x0)
        seed_L, seed_T = encode_complex(s)
        meta.update(
            eigenphase=eigenphase,
            logmag_rate=logmag_rate,
            seed_L=np.asarray(seed_L, dtype=np.int64),
            seed_T=np.asarray(seed_T, dtype=np.int64),
            phase_bits=int(phase_bits),
        )
    else:
        meta["closed_form"] = False
        if generator_block is not None:
            meta["generator_block"] = generator_block
        else:
            meta["generator"] = np.asarray(A, dtype=np.complex128)
        meta["seed_re"] = np.ascontiguousarray(x0.real)
        meta["seed_im"] = np.ascontiguousarray(x0.imag)

    base = BaseModelRef(int(BaseModelKind.ITERATED_OPERATOR), meta)
    residual = ResidualRef(int(ResidualCodecKind.NONE), None)
    return SPFBlockV3(
        dims=(steps, n),
        origin=(0, 0),
        representation=RepresentationRef(int(representation)),
        base=base,
        residual=residual,
    )


def iterated_operator_orbit(block: SPFBlockV3) -> np.ndarray:
    """Reconstruct the ``(steps, n_mode)`` complex orbit a block encodes.

    For a closed-form block this is the **modal** orbit ``x_k[c] = s_c·λ_c^k`` (the
    block's coordinates are the eigenmodes; the per-mode rotation is ``A``'s
    eigenphase). For an explicit block it is ``A^k·x₀`` in the original basis.
    """
    if int(block.base.profile_id) != int(BaseModelKind.ITERATED_OPERATOR):
        raise SPFError("block is not an ITERATED_OPERATOR block")
    L, T = block.reconstruct()
    return decode_complex(L, T)


def iterated_operator_meta(block: SPFBlockV3) -> dict[str, Any]:
    """Return the parameter/metadata dict carried by an ITERATED_OPERATOR block."""
    if int(block.base.profile_id) != int(BaseModelKind.ITERATED_OPERATOR):
        raise SPFError("block is not an ITERATED_OPERATOR block")
    return dict(block.base.params)


register_base_model(IteratedOperatorModel())


__all__ = [
    "IteratedOperatorModel",
    "build_iterated_operator_block",
    "iterated_operator_meta",
    "iterated_operator_orbit",
    "operator_spectrum",
    "productivity_gate",
]
