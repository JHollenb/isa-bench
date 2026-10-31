"""SPFTensor v3 base models (Axis 2) — the generative wavefield laws.

The ``StbAffinePolyModel`` ramp reconstruction below preserves the v2
serialization ABI.  New v0.2 field code should use ``spf.tensor.v02.Jet2D``
for explicit gradient/Hessian coefficients instead of inferring a gradient
from a legacy ``ramp_R`` matrix.

Two families are implemented:

* :class:`StbAffinePolyModel` (``STB_AFFINE_POLY``) — the v2 affine/quadratic/
  cubic ramp + band envelope. Its ``reconstruct_planes`` reproduces the exact
  pre-residual ``(L, Θ)`` contribution of :meth:`spf.tensor.v2.TensorBlock.reconstruct`
  so a relabelled v2 block is bit-identical.

* :class:`SeparableModel` (``SEPARABLE_PHASE`` / ``SEPARABLE_LOGMAG``) — the new
  exponent-domain low-rank primitive (design §6). A rank-``r`` outer product in
  the phase ring ``ℤ_{2^B}`` (``Θ_sep[idx] = Σ_r a_r[i]·ω_r[j] mod 2^B``) and/or
  the log-magnitude plane (``L_sep[idx] = Σ_r b_r[i]·g_r[j]``), plus optional
  per-coordinate phase/log-magnitude bias. This captures a *per-channel
  frequency* (RoPE, per-channel FMCW chirp) that no coordinate polynomial can.

Registration happens at import; :mod:`spf.tensor.v3` triggers it lazily.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .core import (
    SPF32_PHASE_MASK,
    SPF32_PHASE_MOD,
    SPFError,
    _coord_matrix,
)
from .v3 import (
    BaseModelKind,
    BaseModelRef,
    register_base_model,
)

_TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# STB_AFFINE_POLY — the v2 relabel base.
# --------------------------------------------------------------------------- #


class StbAffinePolyModel:
    """v2 affine/quad/cubic ramp + band envelope (profile ``STB_AFFINE_POLY``).

    ``params``: ``{"header": TensorBlockHeaderV2, "bands": tuple[BandRange],
    "active_indices": int64[k]}``. Returns the raw int64 base contribution over
    the active indices — identical to the pre-residual ``l_val`` / ``t_val`` of
    :meth:`spf.tensor.v2.TensorBlock.reconstruct`.
    """

    kind = int(BaseModelKind.STB_AFFINE_POLY)

    def reconstruct_planes(
        self,
        params: Mapping[str, Any],
        dims: tuple[int, ...],
        origin: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # A header-only dense law needs no stored address plane. Match the
        # sample_words consumer's existing implicit-dense convention.
        active = (
            np.asarray(params["active_indices"], dtype=np.int64)
            if "active_indices" in params
            else np.arange(int(np.prod(dims, dtype=np.int64)) if dims else 1, dtype=np.int64)
        )
        L, T = self.sample_planes(params, dims, origin, active)
        return active, L, T

    def sample_planes(
        self,
        params: Mapping[str, Any],
        dims: tuple[int, ...],
        origin: tuple[int, ...],
        active: np.ndarray,
        *,
        phase: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        header = params["header"]
        bands = params.get("bands", ())
        order = int(header.order or 0)
        hdims = tuple(header.dims)
        # Use the header's own origin so the coordinate lattice matches the v2
        # block exactly (the relabel carries the header verbatim).
        Q = _coord_matrix(active, hdims, header.origin, order)
        # local imports: v2 imports this module
        from .v2 import _cubic_mixed_phase, _mixed_ramp_phase

        band_exp = np.asarray(header.band_exp, dtype=np.int64)
        log_quad = np.asarray(getattr(header, "log_quadratic", ()) or (0,) * order, dtype=np.int64)
        l_val = (
            np.int64(header.global_L_bias)
            + np.int64(header.coarse_exp)
            + Q @ band_exp
            + (Q * Q) @ log_quad
            + _mixed_ramp_phase(Q, getattr(header, "log_mixed", ()) or (), order)
        )
        t_val = np.zeros(active.size, dtype=np.int64)
        if phase:
            R = np.asarray(header.ramp_R, dtype=np.int64).reshape(order, order)
            ramp_b = np.asarray(header.ramp_b, dtype=np.int64)
            grad = np.asarray(getattr(header, "phase_gradient", ()) or (0,) * order, dtype=np.int64)
            quad = np.asarray(header.quadratic_ramp, dtype=np.int64)
            cubic = np.asarray(header.cubic_ramp, dtype=np.int64)
            t_val = (
                np.int64(header.global_T_bias)
                + ramp_b.sum()
                + Q @ (R.sum(axis=0) + grad)
                + (Q * Q) @ quad
                + (Q * Q * Q) @ cubic
                + _mixed_ramp_phase(Q, getattr(header, "mixed_ramp", ()) or (), order)
                # AlgRef v0.2 §13.6 mixed cubics, 2-D only (ABI v24).
                + _cubic_mixed_phase(Q, getattr(header, "cubic_mixed", ()) or (), order)
            )
        for band in bands:
            covered = (Q[:, band.dim] >= band.start) & (Q[:, band.dim] < band.start + band.length)
            if band.effective_flags & 0x01:
                l_val[covered] += int(band.L_delta)
            if phase and band.effective_flags & 0x02:
                t_val[covered] += int(band.T_delta)
        return np.asarray(l_val, dtype=np.int64), np.asarray(t_val, dtype=np.int64)

    def storage_bits(self, params: Mapping[str, Any]) -> int:
        header = params["header"]
        order = int(header.order or 0)
        # Header scalars (~16b each) + ramp_R (order² × 16b) + ramp_b/quad/cubic
        # (order × 16b each) + band_exp + active-index list (u32 each, sparse).
        scalars = 8
        ramp = order * order + 3 * order + order
        active = np.asarray(params.get("active_indices", ()), dtype=np.int64)
        sparse = 0 if order == 0 else int(active.size) * 32
        return (scalars + ramp) * 16 + sparse


# --------------------------------------------------------------------------- #
# SEPARABLE_PHASE / SEPARABLE_LOGMAG — the new exponent-domain primitive.
# --------------------------------------------------------------------------- #


class SeparableModel:
    """Rank-``r`` exponent-domain outer product on the phase and/or log-mag plane.

    ``params`` (all keys optional; absent → zero contribution):

    * ``"phase_terms"``: list of ``{"axis_i", "axis_j", "a", "omega"}`` — each a
      rank-1 outer product added to ``Θ`` (``a[i]·ω[j]``). RoPE is one such term
      with ``a = positions``, ``ω = RoPE frequency schedule (u16)``.
    * ``"logmag_terms"``: list of ``{"axis_i", "axis_j", "b", "g"}`` — added to ``L``.
    * ``"phase_bias"``: ``{"axis", "vec"}`` — per-coordinate additive ``Θ`` bias.
    * ``"logmag_bias"``: ``{"axis", "vec"}`` — per-coordinate additive ``L`` bias.

    Dense by construction (``active = arange(N)``): a separable law is defined
    over the whole index grid. Promotes to dense under contraction over a ramp
    axis (``Σ_p e^{i p ω_c}`` is off-lattice); it never falsely flags closure.
    """

    promotes_to_dense_under_contraction = True

    def __init__(self, kind: int) -> None:
        self.kind = int(kind)

    def _coords(self, dims: tuple[int, ...], n: int) -> list[np.ndarray]:
        idx = np.arange(n, dtype=np.int64)
        if len(dims) <= 1:
            return [idx]
        return [c.astype(np.int64) for c in np.unravel_index(idx, dims)]

    def reconstruct_planes(
        self,
        params: Mapping[str, Any],
        dims: tuple[int, ...],
        origin: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(np.prod(dims, dtype=np.int64)) if dims else 1
        active = np.arange(n, dtype=np.int64)
        L, T = self.sample_planes(params, dims, origin, active)
        return active, L, T

    def sample_planes(
        self,
        params: Mapping[str, Any],
        dims: tuple[int, ...],
        origin: tuple[int, ...],
        active: np.ndarray,
        *,
        phase: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        coords = list(np.unravel_index(active, dims)) if dims else [np.zeros(active.size, dtype=np.int64)]
        ndim = len(coords)
        baseL = np.zeros(active.size, dtype=np.int64)
        baseT = np.zeros(active.size, dtype=np.int64)
        org = tuple(int(o) for o in origin) if origin else (0,) * ndim

        def abs_coord(ax: int) -> np.ndarray:
            # Absolute (origin-shifted) coordinate. An axis must index an
            # existing dimension — a stray axis is a malformed/hostile term, not
            # a silently zero-filled coordinate (which would yield a plausible
            # but wrong plane).
            if ax < 0 or ax >= ndim:
                raise SPFError(
                    f"SEPARABLE base: term axis {ax} out of range for a "
                    f"{ndim}-D block (must index an existing dimension)"
                )
            shift = org[ax] if ax < len(org) else 0
            return coords[ax] + np.int64(shift)

        def factor(vec: Any, idx: np.ndarray, *, what: str, axis: int) -> np.ndarray:
            # Index a stored factor/bias vector by an absolute coordinate, with a
            # bounds guard: an origin that runs past a pre-sized vector is a clear
            # error, not an IndexError leaking out of reconstruct.
            v = np.asarray(vec, dtype=np.int64)
            if idx.size and (int(idx.max()) >= v.size or int(idx.min()) < 0):
                raise SPFError(
                    f"SEPARABLE base: {what} index out of range on axis {axis} "
                    f"(coord {int(idx.min())}..{int(idx.max())}, vector length "
                    f"{v.size}); use a closed-form ramp or pre-size the vector"
                )
            return v[idx]

        def ramp(term: Mapping[str, Any], idx: np.ndarray, key: str, axis: int) -> np.ndarray:
            # A closed-form ramp (RoPE's a[p] = p) IS the absolute coordinate plus
            # an optional ``a_origin`` offset — never a stored array. It is free
            # in bits AND extends without bound under origin advance, so a
            # streaming block genuinely reuses its base and only shifts origin
            # (the append-only KV property, design §4.3). Otherwise index the
            # stored ramp with the same bounds guard as any factor vector.
            if term.get("closed_form"):
                return idx + np.int64(int(term.get("a_origin", 0)))
            return factor(term[key], idx, what=key, axis=axis)

        for term in params.get("phase_terms", ()) if phase else ():
            ai, aj = int(term["axis_i"]), int(term["axis_j"])
            baseT = baseT + ramp(term, abs_coord(ai), "a", ai) * factor(
                term["omega"], abs_coord(aj), what="omega", axis=aj
            )
        for term in params.get("logmag_terms", ()):
            ai, aj = int(term["axis_i"]), int(term["axis_j"])
            baseL = baseL + ramp(term, abs_coord(ai), "b", ai) * factor(term["g"], abs_coord(aj), what="g", axis=aj)
        pbias = params.get("phase_bias")
        if phase and pbias is not None:
            ax = int(pbias["axis"])
            baseT = baseT + factor(pbias["vec"], abs_coord(ax), what="phase_bias", axis=ax)
        lbias = params.get("logmag_bias")
        if lbias is not None:
            ax = int(lbias["axis"])
            baseL = baseL + factor(lbias["vec"], abs_coord(ax), what="logmag_bias", axis=ax)
        return baseL, baseT

    def storage_bits(self, params: Mapping[str, Any]) -> int:
        bits = 0
        for term in params.get("phase_terms", ()):
            # A ``closed_form`` ramp (e.g. RoPE's ``a[p] = p``) is not stored — it
            # is the position index itself (design §9.1: RoPE is metadata, zero
            # payload). Only the per-channel ``ω`` vector costs bits.
            a_bits = 0 if term.get("closed_form") else int(np.asarray(term["a"]).size) * 16
            bits += a_bits + int(np.asarray(term["omega"]).size) * 16
        for term in params.get("logmag_terms", ()):
            bits += int(np.asarray(term["b"]).size + np.asarray(term["g"]).size) * 16
        for key in ("phase_bias", "logmag_bias"):
            b = params.get(key)
            if b is not None:
                bits += int(np.asarray(b["vec"]).size) * 16
        return bits


# Shared instance registered under both ids — the implementation handles phase
# and log-magnitude terms uniformly; the id is the routing/intent tag.
_SEPARABLE_PHASE = SeparableModel(int(BaseModelKind.SEPARABLE_PHASE))
_SEPARABLE_LOGMAG = SeparableModel(int(BaseModelKind.SEPARABLE_LOGMAG))


# --------------------------------------------------------------------------- #
# RoPE / separable-phase construction + fit helpers (design §6.4, §9).
# --------------------------------------------------------------------------- #


def rope_frequencies_u16(head_dim: int, base: float = 10000.0, *, phase_bits: int = 16) -> np.ndarray:
    """RoPE per-channel angular frequency (rad/step) as a phase-ring increment.

    ``θ_j = base^(-2⌊j/2⌋ / head_dim)`` rad/step → ``ω = round(θ_j · 2^B/2π)`` in a
    ``2^B`` ring (``B = phase_bits``). Paired channels (``2k``, ``2k+1``) share a
    frequency, matching the standard rotary embedding layout. The default
    ``phase_bits=16`` is the SPF-32 ring (bit-identical to the original ``u16``
    helper); pass 8/32/64 to build a law for a ``POLAR_SPF16/64/128`` block so the
    per-position rotation lands in the representation's wider ring. Returned as
    int64 ω increments (the SeparableModel multiplies in int64; the v3 finaliser
    applies ``mod 2^B``).
    """
    j = np.arange(head_dim, dtype=np.float64)
    theta = base ** (-2.0 * np.floor(j / 2.0) / head_dim)  # rad/step, in (0, 1]
    phase_mod = 1 << int(phase_bits)
    scaled = np.rint(theta * (phase_mod / _TWO_PI))
    if phase_bits > 32:
        # The 2^B mask overruns int64 for B > 63; reduce in Python int. RoPE ω are
        # small (θ ≤ 1 rad/step) so the residues still fit int64.
        return np.array([int(x) % phase_mod for x in scaled.tolist()], dtype=np.int64)
    return scaled.astype(np.int64) & (phase_mod - 1)


def separable_phase_params(
    *,
    axis_i: int,
    axis_j: int,
    a: Sequence[int] | np.ndarray,
    omega: Sequence[int] | np.ndarray,
    closed_form: bool = False,
    a_origin: int = 0,
) -> dict[str, Any]:
    """Build a one-term ``SEPARABLE_PHASE`` base params dict (``Θ = a[i]·ω[j]``).

    ``closed_form=True`` marks the ``a`` ramp as the position index itself (not
    stored data) — used for RoPE, where the position multiplier is free. Under a
    closed-form ramp the reconstruction uses the *absolute* coordinate (block
    origin + local index) ``+ a_origin``, so the ramp extends freely as a
    streaming block advances its origin; the stored ``a`` is informational only.
    """
    return {
        "phase_terms": [
            {
                "axis_i": int(axis_i),
                "axis_j": int(axis_j),
                "a": np.asarray(a, dtype=np.int64),
                "omega": np.asarray(omega, dtype=np.int64),
                "closed_form": bool(closed_form),
                "a_origin": int(a_origin),
            }
        ]
    }


def accumulate_phase_ring(
    omega: Sequence[int] | np.ndarray,
    n_index: int,
    *,
    bias: Sequence[int] | np.ndarray | int = 0,
    a_origin: int = 0,
    phase_bits: int = 16,
) -> np.ndarray:
    """The ``SEPARABLE_PHASE`` accumulator reconstruct — Θ from its Jacobian generator.

    The eager form ``Θ[p, c] = (a_origin + p) · ω_c`` materialises a large int64
    absolute product (measured peak ~10.6M for a 1024-position ramp = 162 wraps of
    2¹⁶) that is then masked away — wasted compute, and not what a phase
    accumulator does. Store and reconstruct the **derivative** instead:

        ∂Θ/∂p = ω_c                                  # constant per channel, native u16
        Θ[0]  = (bias + (a_origin mod 2^B)·ω) mod 2^B
        Θ[p]  = (Θ[p−1] + ω) mod 2^B                  # PAN's pure-add primitive, in-ring

    Each per-channel value stays in ``[0, 2^B)`` and every step is a single ``2^B``
    adder (carry-out discarded), so the eager int64 product is never formed. The
    result is **bit-identical** to ``((a_origin + p)·ω + bias) mod 2^B`` — the
    derivative is small and native; only the integral blows up. Returns the
    ``(n_index, n_channel)`` Θ grid as int64 codes already reduced mod ``2^B``
    for B < 64, and uint64 for B = 64. The 64-bit path retains the complete
    fractional phase state; consumers may round their output without modifying it.
    """
    if not 1 <= int(phase_bits) <= 64 or int(phase_bits) != phase_bits:
        raise SPFError("phase_bits must be an integer in [1,64]")
    if int(n_index) != n_index or n_index < 0:
        raise SPFError("n_index must be a nonnegative integer")
    mod = 1 << int(phase_bits)
    if phase_bits >= 63:
        om, seed = _phase_ring_u64_seed(omega, bias, a_origin, int(phase_bits))
        dtype = np.uint64 if phase_bits == 64 else np.int64
        out = np.empty((int(n_index), om.size), dtype=dtype)
        cur = seed
        for p in range(int(n_index)):
            out[p] = cur
            cur = cur + om  # uint64 vector addition discards carry, exactly mod 2^64
            if phase_bits == 63:
                cur = cur & np.uint64(2**63 - 1)
        return out
    om = np.asarray(omega, dtype=np.int64).reshape(-1) % mod
    n_chan = int(om.size)
    bias_arr = (
        (np.asarray(bias, dtype=np.int64).reshape(-1) % mod)
        if np.ndim(bias)
        else np.full(n_chan, int(bias) % mod, dtype=np.int64)
    )
    if bias_arr.size != n_chan:
        raise SPFError(f"bias length {bias_arr.size} != n_channel {n_chan}")
    out = np.empty((int(n_index), n_chan), dtype=np.int64)
    # Seed at the absolute origin with a single reduced product (a_origin pre-reduced
    # mod 2^B, so the seed product is < 2^B · 2^B, never the eager full-range product).
    cur = (bias_arr + (np.int64(int(a_origin) % mod) * om)) % mod
    for p in range(int(n_index)):
        out[p] = cur
        cur = (cur + om) % mod  # one in-ring add per step; carry discarded by the mod
    return out


def separable_phase_accumulator_reconstruct(
    omega: Sequence[int] | np.ndarray,
    n_index: int,
    *,
    bias: Sequence[int] | np.ndarray | int = 0,
    a_origin: int = 0,
    phase_bits: int = 16,
) -> np.ndarray:
    """Eager-equivalent check for :func:`accumulate_phase_ring` (the reference it matches).

    Returns ``((a_origin + p)·ω + bias) mod 2^B`` computed the eager way — the
    value the accumulator reproduces bit-for-bit. Kept as the validation oracle
    (and a streaming convenience): the two must always agree.
    """
    if not 1 <= int(phase_bits) <= 64 or int(phase_bits) != phase_bits:
        raise SPFError("phase_bits must be an integer in [1,64]")
    if int(n_index) != n_index or n_index < 0:
        raise SPFError("n_index must be a nonnegative integer")
    if phase_bits >= 63:
        om, seed = _phase_ring_u64_seed(omega, bias, a_origin, int(phase_bits))
        # Unsigned products/sums wrap in the requested ring, with no float path.
        p = np.arange(int(n_index), dtype=np.uint64).reshape(-1, 1)
        result = p * om.reshape(1, -1) + seed.reshape(1, -1)
        return result if phase_bits == 64 else (result & np.uint64(2**63 - 1)).astype(np.int64)
    mod = 1 << int(phase_bits)
    om = np.asarray(omega, dtype=np.int64).reshape(-1)
    n_chan = int(om.size)
    bias_arr = (
        np.asarray(bias, dtype=np.int64).reshape(-1) if np.ndim(bias) else np.full(n_chan, int(bias), dtype=np.int64)
    )
    p = np.arange(int(n_index), dtype=np.int64).reshape(-1, 1) + np.int64(int(a_origin))
    return ((p * om.reshape(1, -1)) + bias_arr.reshape(1, -1)) % mod


def _phase_ring_u64_seed(
    omega: Sequence[int] | np.ndarray,
    bias: Sequence[int] | np.ndarray | int,
    a_origin: int,
    phase_bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact one-time integer seeding, including signed/large origins and wraps."""
    mod = 1 << phase_bits
    increments = [int(v) % mod for v in np.asarray(omega, dtype=object).reshape(-1)]
    biases = (
        [int(v) % mod for v in np.asarray(bias, dtype=object).reshape(-1)]
        if np.ndim(bias)
        else [int(bias) % mod] * len(increments)
    )
    if len(biases) != len(increments):
        raise SPFError(f"bias length {len(biases)} != n_channel {len(increments)}")
    origin = int(a_origin) % mod
    seed = [(b + origin * w) % mod for b, w in zip(biases, increments, strict=True)]
    return np.asarray(increments, dtype=np.uint64), np.asarray(seed, dtype=np.uint64)


def rope_base_ref(
    *,
    n_positions: int,
    head_dim: int,
    position_axis: int = 0,
    channel_axis: int = 1,
    base: float = 10000.0,
    position_origin: int = 0,
) -> BaseModelRef:
    """A closed-form RoPE ``SEPARABLE_PHASE`` base ref: ``Θ[p,c] = p · ω_c``.

    The position ramp is ``a[p] = position_origin + p`` and ``ω_c`` is the RoPE
    schedule. Zero residual bits — RoPE is metadata, not payload (design §9.1).
    ``position_origin`` flows through ``a_origin`` (the closed-form offset), never
    baked into the stored ramp, so the block ``origin`` is the single source of
    the position shift — no double count, and the ramp extends under streaming.
    """
    a = np.arange(n_positions, dtype=np.int64)
    omega = rope_frequencies_u16(head_dim, base=base)
    return BaseModelRef(
        int(BaseModelKind.SEPARABLE_PHASE),
        params=separable_phase_params(
            axis_i=position_axis,
            axis_j=channel_axis,
            a=a,
            omega=omega,
            closed_form=True,
            a_origin=position_origin,
        ),
    )


def fit_separable_phase_rank1(
    theta_u16: np.ndarray,
    *,
    axis_i: int = 0,
    axis_j: int = 1,
) -> dict[str, Any]:
    """Recover a rank-1 per-channel frequency ``ω`` and phase bias from a 2-D Θ field.

    Modular-safe first-difference estimator (Viterbi–Viterbi style): per channel
    ``c``, ``ω_c = ∠(mean_p z_{p+1} z_p*)`` where ``z = e^{i2πΘ/2¹⁶}``. Robust
    under many phase wraps as long as the per-step increment ``|ω_c| < π`` (true
    for RoPE, whose top channel is ≈1 rad/step). Returns a ``SEPARABLE_PHASE``
    params dict with ``a[p] = p`` and the recovered ``ω``/bias.
    """
    field = np.asarray(theta_u16, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("fit_separable_phase_rank1 expects a 2-D (position, channel) field")
    if axis_i == 1 and axis_j == 0:
        field = field.T
    n_pos, n_chan = field.shape
    z = np.exp(1j * field * (_TWO_PI / SPF32_PHASE_MOD))
    if n_pos >= 2:
        steps = z[1:, :] * np.conj(z[:-1, :])
        incr = np.angle(steps.mean(axis=0))  # rad/step per channel
    else:
        incr = np.zeros(n_chan)
    omega = np.rint(incr * (SPF32_PHASE_MOD / _TWO_PI)).astype(np.int64) & SPF32_PHASE_MASK
    bias = np.rint(np.angle(z[0, :]) % _TWO_PI * (SPF32_PHASE_MOD / _TWO_PI)).astype(np.int64)
    bias &= SPF32_PHASE_MASK
    a = np.arange(n_pos, dtype=np.int64)
    return {
        "phase_terms": [{"axis_i": int(axis_i), "axis_j": int(axis_j), "a": a, "omega": omega}],
        "phase_bias": {"axis": int(axis_j), "vec": bias},
    }


def compose_separable_phase(
    params_a: Mapping[str, Any],
    params_b: Mapping[str, Any],
) -> dict[str, Any]:
    """Hadamard (free SPF multiply) of two ``SEPARABLE_PHASE`` laws.

    SPF elementwise multiply adds phase, so two separable phase laws with the
    same axis structure and matching ``a`` ramp compose exactly to
    ``a ⊗ (ω₁ + ω₂)`` (design §6.3 — closed under the Hadamard product). Falls
    back to concatenating the term lists when the ramps differ (still exact, but
    rank grows). A per-coordinate ``phase_bias`` adds through the product (the
    biases sum); the ``closed_form`` ramp is preserved so a composed RoPE law
    stays free of stored-ramp charge.

    The Hadamard product is a phase-plane operation: an input carrying a
    log-magnitude term/bias cannot be represented and is rejected, rather than
    silently dropped.
    """
    for src in (params_a, params_b):
        if src.get("logmag_terms") or src.get("logmag_bias"):
            raise SPFError(
                "compose_separable_phase: cannot compose a law carrying a "
                "log-magnitude term/bias (phase Hadamard product is phase-only)"
            )

    terms_a = list(params_a.get("phase_terms", ()))
    terms_b = list(params_b.get("phase_terms", ()))
    result: dict[str, Any] = {}
    # Merge two rank-1 terms into one omega-sum ONLY when their ramps are
    # *semantically* identical, not merely when their stored ``a`` arrays match.
    # A closed-form ramp reconstructs as ``coord + a_origin`` (the stored ``a`` is
    # ignored); a stored ramp reconstructs as ``a[coord]``. So two terms that
    # share ``a = arange`` but differ in ``closed_form``/``a_origin`` are NOT the
    # same ramp — array-merging them would silently drop the ``a_origin·ω``
    # position offset. When they differ, fall through to concatenation (still an
    # exact product, rank just grows by one).
    if (
        len(terms_a) == 1
        and len(terms_b) == 1
        and terms_a[0]["axis_i"] == terms_b[0]["axis_i"]
        and terms_a[0]["axis_j"] == terms_b[0]["axis_j"]
        and np.array_equal(np.asarray(terms_a[0]["a"]), np.asarray(terms_b[0]["a"]))
        and bool(terms_a[0].get("closed_form")) == bool(terms_b[0].get("closed_form"))
        and int(terms_a[0].get("a_origin", 0)) == int(terms_b[0].get("a_origin", 0))
    ):
        merged = {
            "axis_i": terms_a[0]["axis_i"],
            "axis_j": terms_a[0]["axis_j"],
            "a": np.asarray(terms_a[0]["a"], dtype=np.int64),
            "omega": (
                np.asarray(terms_a[0]["omega"], dtype=np.int64) + np.asarray(terms_b[0]["omega"], dtype=np.int64)
            ),
        }
        # Preserve the closed-form ramp (and its origin offset) only when both
        # inputs agree — otherwise the merged term keeps the stored ramp.
        a_org = int(terms_a[0].get("a_origin", 0))
        if (
            terms_a[0].get("closed_form")
            and terms_b[0].get("closed_form")
            and a_org == int(terms_b[0].get("a_origin", 0))
        ):
            merged["closed_form"] = True
            merged["a_origin"] = a_org
        result["phase_terms"] = [merged]
    else:
        result["phase_terms"] = terms_a + terms_b

    bias_a = params_a.get("phase_bias")
    bias_b = params_b.get("phase_bias")
    if bias_a is not None and bias_b is not None:
        va = np.asarray(bias_a["vec"], dtype=np.int64)
        vb = np.asarray(bias_b["vec"], dtype=np.int64)
        if int(bias_a["axis"]) != int(bias_b["axis"]) or va.shape != vb.shape:
            raise SPFError("compose_separable_phase: phase_bias axis/length mismatch — cannot Hadamard-compose")
        result["phase_bias"] = {"axis": int(bias_a["axis"]), "vec": va + vb}
    elif bias_a is not None:
        result["phase_bias"] = {"axis": int(bias_a["axis"]), "vec": np.asarray(bias_a["vec"], dtype=np.int64)}
    elif bias_b is not None:
        result["phase_bias"] = {"axis": int(bias_b["axis"]), "vec": np.asarray(bias_b["vec"], dtype=np.int64)}
    return result


def doppler_replica_dictionary(
    *,
    n_pulse: int,
    doppler_bins: Sequence[int],
    hypothesis_axis: int = 0,
    pulse_axis: int = 1,
) -> dict[str, Any]:
    """A Doppler-replica bank as one ``SEPARABLE_PHASE`` law (zero stored bits).

    Each Doppler hypothesis ``h`` is a slow-time phase ramp ``Θ[h,p] = p · ω_h``
    with ``ω_h = round(bin_h · 2¹⁶ / n_pulse)`` (the per-pulse phase increment of
    a Doppler bin). The whole bank is the separable product ``a[h]=h`` (a closed-
    form index ramp, free) × ``ω`` (one frequency per hypothesis) — the radar
    analogue of the RoPE construction, so the bank costs a handful of ``ω`` ints
    instead of a dense ``len(bins)·n_pulse`` replica matrix.

    Use :func:`compose_separable_phase` to fold this with another separable law
    (e.g. a steering ramp) for free.
    """
    bins = np.asarray(list(doppler_bins), dtype=np.int64)
    omega = np.rint(bins.astype(np.float64) * (SPF32_PHASE_MOD / float(n_pulse)))
    omega = omega.astype(np.int64) & SPF32_PHASE_MASK
    a = np.arange(bins.size, dtype=np.int64)
    return separable_phase_params(
        axis_i=hypothesis_axis,
        axis_j=pulse_axis,
        a=a,
        omega=omega,
    )


def register_defaults() -> None:
    register_base_model(StbAffinePolyModel())
    register_base_model(_SEPARABLE_PHASE)
    register_base_model(_SEPARABLE_LOGMAG)


register_defaults()


__all__ = [
    "SeparableModel",
    "StbAffinePolyModel",
    "accumulate_phase_ring",
    "compose_separable_phase",
    "fit_separable_phase_rank1",
    "rope_base_ref",
    "rope_frequencies_u16",
    "separable_phase_accumulator_reconstruct",
    "separable_phase_params",
]
