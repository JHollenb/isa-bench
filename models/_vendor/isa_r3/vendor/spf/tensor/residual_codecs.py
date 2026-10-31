"""SPFTensor v3 residual codecs (Axis 3) — how post-base deviations are stored.

A codec turns the post-base ``(dL, dΘ)`` deviation streams into a compact
payload and back. The family:

* ``NONE`` — zero residual (the base is the whole story).
* ``RAW_INT16`` — exact int16 ``dL``/``dΘ`` (the v2 behaviour; lossless).
* ``UNIFORM`` — uniform-bin requant of each plane to ``b`` bits (lossy, budgeted).
* ``CODEBOOK_CIRCULAR`` — circular Lloyd–Max codebook on the phase residual
  (lossy; **wins only on concentrated** ``dΘ``) + uniform ``dL``.
* ``PALETTE`` — exact ``(dL, dΘ)`` palette + indices; **lossless** for a true
  finite constellation (M-PSK), useless on continuous residuals.

Selection is **metadata-gated and byte-accounted** (design §7.2 / §3.3, the
PR#302 invariant): a structured codec MUST NOT be applied to a white residual
(it cannot reduce size and only adds table overhead), and the chosen codec must
win on ``metadata + codebook + index + outlier`` bytes subject to an error
budget. ``select_codec`` enforces both.

**Scalar width.** ``NONE`` / ``RAW_INT16`` / ``UNIFORM`` / ``PALETTE`` carry int64
residues and are width-generic — the v3 finaliser applies the representation's
``mod 2^theta_bits`` ring, so they round-trip at SPF-16/32/64/128.
``CODEBOOK_CIRCULAR`` is now **width-generic** via the ``theta_mod`` parameter
(default ``2^16`` = SPF-32, byte-identical to the previous implementation).
Pass ``theta_mod=2^theta_bits`` to encode/select at other widths. The Lloyd–Max
quantiser runs in **float64**, so centre precision saturates at ~2^-53 of a
turn — SPF-16/32/64 centres (values ≤ 2^31) are float64-exact, but SPF-128's
full 2^64 phase is only exploitable to ~2^-53 turn (still far finer than
SPF-64's 2^-32 ring). Decode is width-free: ``centres[idx]`` are int64 and the
v3 ``reconstruct()`` finaliser applies ``mod 2^theta_bits``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import SPF32_PHASE_MOD
from .v3 import (
    ResidualCodecKind,
    ResidualRef,
    register_residual_codec,
)

_TWO_PI = 2.0 * np.pi
_PHASE_TO_RAD = _TWO_PI / SPF32_PHASE_MOD  # back-compat module constant


# --------------------------------------------------------------------------- #
# Uniform per-plane quantiser (shared by UNIFORM and the dL side of CODEBOOK).
# --------------------------------------------------------------------------- #


def _uniform_encode(x: np.ndarray, bits: int) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.int64)
    lo = int(x.min()) if x.size else 0
    hi = int(x.max()) if x.size else 0
    span = hi - lo
    if span == 0 or bits <= 0:
        return {"bits": int(bits), "lo": lo, "span": 0, "q": np.zeros(x.size, dtype=np.int64)}
    levels = (1 << bits) - 1
    q = np.clip(np.rint((x - lo) / span * levels), 0, levels).astype(np.int64)
    return {"bits": int(bits), "lo": lo, "span": int(span), "q": q}


def _uniform_decode(p: Mapping[str, Any], n: int) -> np.ndarray:
    bits = int(p["bits"])
    lo = int(p["lo"])
    span = int(p["span"])
    if span == 0 or bits <= 0:
        return np.full(n, lo, dtype=np.int64)
    levels = (1 << bits) - 1
    if "packed" in p:
        q = _unpack_indices(p["packed"], bits, n)
        return _uniform_integer_decode(p, q)
    q = np.asarray(p["q"], dtype=np.int64)
    return lo + np.rint(q.astype(np.float64) / levels * span).astype(np.int64)


def _uniform_integer_decode(p: Mapping[str, Any], q: np.ndarray) -> np.ndarray:
    levels = (1 << int(p["bits"])) - 1
    span = int(p["span"])
    if span > ((1 << 62) - 1) // levels:
        raise ValueError("integer uniform decoder span exceeds int64 product headroom")
    quotient, remainder = np.divmod(q * span, levels)
    rounded = quotient + ((2 * remainder > levels) | ((2 * remainder == levels) & (quotient & 1))).astype(np.int64)
    return int(p["lo"]) + rounded


def _uniform_bits(p: Mapping[str, Any], n: int) -> int:
    if "packed" in p:
        return int(np.asarray(p["packed"]).nbytes) * 8 + 64
    return n * int(p["bits"]) + 64  # indices + (lo, span) header


def _pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    """Physical little-endian bitstream; this packs codec indices, not SPF scalars."""
    if not 0 <= bits <= 32:
        raise ValueError("packed index width must be in [0, 32]")
    q = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(q < 0) or (bits and np.any(q >= (1 << bits))):
        raise ValueError("index outside bit width")
    if not bits:
        return np.empty(0, dtype=np.uint8)
    planes = ((q[:, None] >> np.arange(bits, dtype=np.int64)) & 1).astype(np.uint8)
    return np.packbits(planes.reshape(-1), bitorder="little")


def _unpack_indices(payload: np.ndarray, bits: int, count: int, positions=None) -> np.ndarray:
    """Decode consumed indices directly, including codes crossing byte boundaries."""
    data = np.asarray(payload, dtype=np.uint8)
    if data.size != (count * bits + 7) // 8:
        raise ValueError("packed residual length mismatch")
    pos = np.arange(count, dtype=np.int64) if positions is None else np.asarray(positions, dtype=np.int64)
    if np.any(pos < 0) or np.any(pos >= count):
        raise ValueError("packed residual position outside extent")
    out = np.zeros(pos.shape, dtype=np.int64)
    for bit in range(bits):
        offset = pos * bits + bit
        out |= ((data[offset >> 3].astype(np.int64) >> (offset & 7)) & 1) << bit
    return out


def pack_residual(ref: ResidualRef) -> ResidualRef:
    """Pack existing uniform/palette indices while retaining their codec IDs."""
    payload = dict(ref.payload)
    if ref.codec_id == int(ResidualCodecKind.UNIFORM):
        for key in ("L", "T"):
            plane = dict(payload[key])
            q = np.asarray(plane.pop("q"), dtype=np.int64)
            plane.update(packed=_pack_indices(q, int(plane["bits"])), count=q.size)
            payload[key] = plane
    elif ref.codec_id == int(ResidualCodecKind.PALETTE):
        idx = np.asarray(payload.pop("idx"), dtype=np.int64)
        bits = max(1, (len(payload["palette"]) - 1).bit_length())
        palette = np.asarray(payload["palette"])
        if palette.size and int(palette.min()) >= -32768 and int(palette.max()) <= 32767:
            payload["palette"] = palette.astype(np.int16)
        payload.update(idx_packed=_pack_indices(idx, bits), index_bits=bits, count=idx.size)
    else:
        raise ValueError("physical packing supports UNIFORM and PALETTE")
    return ResidualRef(ref.codec_id, payload)


# --------------------------------------------------------------------------- #
# Circular Lloyd–Max codebook on a phase residual.
# --------------------------------------------------------------------------- #


def _circular_lloyd_max(
    values: np.ndarray,
    k: int,
    *,
    iters: int = 30,
    theta_mod: int = SPF32_PHASE_MOD,
) -> tuple[np.ndarray, np.ndarray]:
    """K circular centres + per-sample assignment for ``values`` (u-units).

    Lloyd–Max on the circle: distance and the centroid update both wrap at
    ``theta_mod``, so a residual that straddles the ±π seam clusters correctly.
    On concentrated data the centres crowd the dense region — the win over
    uniform bins. Returns ``(centres int64 in u-units, idx)``; ``idx`` indexes
    ``centres``.

    For ``theta_mod <= 2**62`` the rounding uses the standard numpy
    ``.astype(np.int64)`` path (max ``|centre| = theta_mod/2 <= 2**61``, safe).
    For ``theta_mod > 2**62`` (SPF-128's 2^64) a cluster whose circular mean
    lands on the ±π seam rounds to ``+2^63`` — one past int64-max — which
    ``np.array(..., dtype=np.int64)`` rejects with ``OverflowError``. The ring is
    signed (``+2^63 ≡ -2^63``), so each rounded centre is folded into the signed
    int64 range; the seam maps to int64-min instead of overflowing. Decode and
    the v3 finaliser re-apply ``mod 2^64``, so ``-2^63`` recovers the π code
    exactly. This is required (not merely defensive): the codebook is circular
    precisely so a residual that *straddles the seam* clusters there.
    """
    phase_to_rad = _TWO_PI / theta_mod
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    k = max(1, min(int(k), n if n else 1))
    ang = v * phase_to_rad
    z = np.exp(1j * ang)
    # Seed centres at empirical circular quantiles (sorted by angle from mean).
    order = np.argsort(np.angle(z * np.conj(z.mean()) if z.size else z))
    seeds = order[np.linspace(0, n - 1, k).astype(np.int64)] if n else np.zeros(k, dtype=np.int64)
    centres = ang[seeds].copy() if n else np.zeros(k)
    idx = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        # circular distance to each centre
        diff = np.angle(np.exp(1j * (ang[:, None] - centres[None, :])))
        idx = np.argmin(np.abs(diff), axis=1)
        counts = np.bincount(idx, minlength=k)
        z_re = np.bincount(idx, weights=z.real, minlength=k)
        z_im = np.bincount(idx, weights=z.imag, minlength=k)
        nonempty = counts > 0
        new_centres = centres.copy()
        new_centres[nonempty] = np.angle(z_re[nonempty] + 1j * z_im[nonempty])
        moved = bool(np.any(~np.isclose(new_centres[nonempty], centres[nonempty])))
        centres = new_centres
        if not moved:
            break
    if theta_mod > 2**62:
        # SPF-128: theta_mod = 2^64. A cluster mean on the ±π seam rounds to
        # +2^63 (one past int64-max); the ring is signed (+2^63 ≡ -2^63), so fold
        # each rounded centre into [-2^63, 2^63) per-element in Python before the
        # int64 cast. The seam maps to int64-min, not an OverflowError. Decode +
        # the v3 finaliser re-apply mod 2^64, recovering the exact phase code.
        half = theta_mod // 2  # 2^63
        centres_list = [((round(float(c) / phase_to_rad) + half) % theta_mod) - half for c in centres.tolist()]
        centres_u = np.array(centres_list, dtype=np.int64)
    else:
        centres_u = np.rint(centres / phase_to_rad).astype(np.int64)
    return centres_u, idx


# --------------------------------------------------------------------------- #
# Codecs.
# --------------------------------------------------------------------------- #


@dataclass
class NoneCodec:
    kind = int(ResidualCodecKind.NONE)

    def decode(self, payload: Any, n_active: int) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(n_active, dtype=np.int64), np.zeros(n_active, dtype=np.int64)

    def storage_bits(self, payload: Any) -> int:
        return 0

    @staticmethod
    def encode() -> ResidualRef:
        return ResidualRef(int(ResidualCodecKind.NONE), None)


@dataclass
class RawInt16Codec:
    kind = int(ResidualCodecKind.RAW_INT16)

    def decode(self, payload: Mapping[str, Any], n_active: int) -> tuple[np.ndarray, np.ndarray]:
        dL = np.asarray(payload["dL"], dtype=np.int64)
        dT = np.asarray(payload["dT"], dtype=np.int64)
        dL = dL << int(payload.get("log_shift_bits", 0))
        dT = dT << int(payload.get("phase_shift_bits", 0))
        return dL, dT

    def storage_bits(self, payload: Mapping[str, Any]) -> int:
        return int(np.asarray(payload["dL"]).size) * 32  # 16 bits per plane

    @staticmethod
    def encode(dL: np.ndarray, dT: np.ndarray) -> ResidualRef:
        return ResidualRef(
            int(ResidualCodecKind.RAW_INT16),
            {"dL": np.asarray(dL, dtype=np.int64), "dT": np.asarray(dT, dtype=np.int64)},
        )


@dataclass
class UniformCodec:
    kind = int(ResidualCodecKind.UNIFORM)

    def decode(self, payload: Mapping[str, Any], n_active: int) -> tuple[np.ndarray, np.ndarray]:
        dL = _uniform_decode(payload["L"], n_active)
        dT = _uniform_decode(payload["T"], n_active)
        return dL, dT

    def storage_bits(self, payload: Mapping[str, Any]) -> int:
        n = int(payload["L"]["count"]) if "packed" in payload["L"] else int(np.asarray(payload["L"]["q"]).size)
        return _uniform_bits(payload["L"], n) + _uniform_bits(payload["T"], n)

    @staticmethod
    def encode(dL: np.ndarray, dT: np.ndarray, *, bL: int = 5, bT: int = 5) -> ResidualRef:
        return ResidualRef(
            int(ResidualCodecKind.UNIFORM),
            {"L": _uniform_encode(dL, bL), "T": _uniform_encode(dT, bT)},
        )


@dataclass
class CodebookCircularCodec:
    kind = int(ResidualCodecKind.CODEBOOK_CIRCULAR)

    def decode(self, payload: Mapping[str, Any], n_active: int) -> tuple[np.ndarray, np.ndarray]:
        centres = np.asarray(payload["T_centres"], dtype=np.int64)
        idx = np.asarray(payload["T_idx"], dtype=np.int64)
        dT = centres[idx]
        dL = _uniform_decode(payload["L"], n_active)
        return dL, dT

    def storage_bits(self, payload: Mapping[str, Any]) -> int:
        centres = np.asarray(payload["T_centres"])
        idx = np.asarray(payload["T_idx"])
        k = int(centres.size)
        idx_bits = int(np.ceil(np.log2(max(k, 2))))
        n = int(idx.size)
        theta_bits = int(payload.get("theta_bits", 16))
        return k * theta_bits + n * idx_bits + _uniform_bits(payload["L"], n)

    @staticmethod
    def encode(
        dL: np.ndarray,
        dT: np.ndarray,
        *,
        k: int = 16,
        bL: int = 5,
        theta_mod: int = SPF32_PHASE_MOD,
    ) -> ResidualRef:
        centres, idx = _circular_lloyd_max(dT, k, theta_mod=theta_mod)
        theta_bits = int(theta_mod).bit_length() - 1
        return ResidualRef(
            int(ResidualCodecKind.CODEBOOK_CIRCULAR),
            {
                "T_centres": centres,
                "T_idx": idx,
                "L": _uniform_encode(dL, bL),
                "theta_bits": theta_bits,
            },
        )


@dataclass
class PaletteCodec:
    kind = int(ResidualCodecKind.PALETTE)

    def decode(self, payload: Mapping[str, Any], n_active: int) -> tuple[np.ndarray, np.ndarray]:
        palette = np.asarray(payload["palette"], dtype=np.int64)  # (K, 2): (dL, dT)
        idx = (
            _unpack_indices(payload["idx_packed"], int(payload["index_bits"]), n_active)
            if "idx_packed" in payload
            else np.asarray(payload["idx"], dtype=np.int64)
        )
        return palette[idx, 0], palette[idx, 1]

    def storage_bits(self, payload: Mapping[str, Any]) -> int:
        palette = np.asarray(payload["palette"])
        if "idx_packed" in payload:
            return int(palette.nbytes + np.asarray(payload["idx_packed"]).nbytes) * 8
        idx = np.asarray(payload["idx"])
        k = int(palette.shape[0])
        idx_bits = int(np.ceil(np.log2(max(k, 2))))
        return k * 32 + int(idx.size) * idx_bits

    @staticmethod
    def encode(dL: np.ndarray, dT: np.ndarray) -> ResidualRef:
        pairs = np.stack([np.asarray(dL, dtype=np.int64), np.asarray(dT, dtype=np.int64)], axis=1)
        palette, idx = np.unique(pairs, axis=0, return_inverse=True)
        return ResidualRef(
            int(ResidualCodecKind.PALETTE),
            {"palette": palette, "idx": np.asarray(idx, dtype=np.int64).reshape(-1)},
        )


def decode_residual_at(
    ref: ResidualRef, positions: np.ndarray, n_active: int, *, phase: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Decode only consumed addresses and planes; codebooks stay in code space."""
    kind, payload = ResidualCodecKind(ref.codec_id), ref.payload
    from .v3 import get_residual_codec

    registered = get_residual_codec(ref.codec_id)
    if type(registered) is not type(_CODEC_BY_ID.get(ref.codec_id)):
        L, T = registered.decode(payload, n_active)
        return np.asarray(L)[positions], np.asarray(T)[positions]
    zero = np.zeros(positions.size, dtype=np.int64)

    def uniform(p: Mapping[str, Any]) -> np.ndarray:
        if not int(p["span"]) or int(p["bits"]) <= 0:
            return np.full(positions.size, int(p["lo"]), dtype=np.int64)
        if "packed" in p:
            q = _unpack_indices(p["packed"], int(p["bits"]), n_active, positions)
            return _uniform_integer_decode(p, q)
        return _uniform_decode({**p, "q": np.asarray(p["q"])[positions]}, positions.size)

    if kind is ResidualCodecKind.NONE:
        return zero, zero
    if kind is ResidualCodecKind.RAW_INT16:
        return np.asarray(payload["dL"], dtype=np.int64)[positions] << int(payload.get("log_shift_bits", 0)), (
            np.asarray(payload["dT"], dtype=np.int64)[positions] << int(payload.get("phase_shift_bits", 0))
            if phase
            else zero
        )
    if kind is ResidualCodecKind.UNIFORM:
        return uniform(payload["L"]), uniform(payload["T"]) if phase else zero
    if kind is ResidualCodecKind.CODEBOOK_CIRCULAR:
        T = np.asarray(payload["T_centres"], dtype=np.int64)[np.asarray(payload["T_idx"])[positions]] if phase else zero
        return uniform(payload["L"]), T
    if kind is ResidualCodecKind.PALETTE:
        indices = (
            _unpack_indices(payload["idx_packed"], int(payload["index_bits"]), n_active, positions)
            if "idx_packed" in payload
            else np.asarray(payload["idx"])[positions]
        )
        palette = np.asarray(payload["palette"], dtype=np.int64)
        return palette[indices, 0], palette[indices, 1] if phase else zero
    from .v3 import get_residual_codec

    L, T = get_residual_codec(ref.codec_id).decode(payload, n_active)
    return np.asarray(L)[positions], np.asarray(T)[positions] if phase else zero


# --------------------------------------------------------------------------- #
# Residual statistics + the metadata-gated, byte-accounted selector.
# --------------------------------------------------------------------------- #


# Mean-resultant whiteness threshold. A genuinely uniform phase has R̄ near 0
# up to the O(1/√n) sampling floor; concentrated phase (a RoPE residual, κ≈6)
# sits near 1. 0.30 separates the two with margin for small-block variance.
_WHITE_RBAR = 0.30


@dataclass(frozen=True, slots=True)
class ResidualStats:
    n: int
    entropy: float  # phase-residual histogram entropy (bits)
    max_entropy: float  # log2(n_bins)
    mean_resultant: float  # R̄ = |mean e^{iθ}| ∈ [0, 1]
    kappa: float  # von Mises concentration estimate from R̄
    is_finite_constellation: bool

    @property
    def is_white(self) -> bool:
        # Whiteness gate (design §7.3 / PR#302): a white residual admits only
        # NONE/RAW — a structured codec cannot reduce its entropy. Use the
        # circular mean resultant R̄, NOT histogram entropy: a fixed-bin entropy
        # under-reads on small blocks (a 64-bin histogram of n<512 samples is
        # structurally sparse) and would misflag white data as structured,
        # charging a codebook that can only hurt. R̄ is n-robust; subtract the
        # uniform-phase sampling floor so a small white block is not mistaken
        # for a faint concentration. A finite constellation is structured
        # (palette-codable) even when its symmetric phases cancel to R̄ ≈ 0, so
        # it is never white.
        if self.is_finite_constellation:
            return False
        if self.n <= 1:
            return True
        floor = float(np.sqrt(np.pi / (4.0 * self.n)))
        return (self.mean_resultant - floor) < _WHITE_RBAR


def _kappa_from_rbar(r: float) -> float:
    """Banerjee et al. closed-form von Mises κ estimate from the mean resultant R̄."""
    r = float(min(max(r, 0.0), 0.999999))
    if r < 1e-9:
        return 0.0
    return (r * (2.0 - r * r)) / (1.0 - r * r)


def residual_stats(
    dL: np.ndarray,
    dT: np.ndarray,
    *,
    n_bins: int = 64,
    constellation_max: int = 8,
    theta_mod: int = SPF32_PHASE_MOD,
) -> ResidualStats:
    """Cheap features for codec selection over the residual streams."""
    phase_to_rad = _TWO_PI / theta_mod
    dT = np.asarray(dT, dtype=np.int64)
    n = int(dT.size)
    ang = (dT.astype(np.float64) * phase_to_rad) % _TWO_PI
    if n:
        rbar = float(np.abs(np.exp(1j * ang).mean()))
        hist, _ = np.histogram(ang, bins=n_bins, range=(0.0, _TWO_PI))
        p = hist.astype(np.float64) / max(hist.sum(), 1)
        nz = p > 0
        entropy = float(-(p[nz] * np.log2(p[nz])).sum())
    else:
        rbar = 1.0
        entropy = 0.0
    pairs = np.stack([np.asarray(dL, dtype=np.int64), dT], axis=1) if n else np.zeros((0, 2))
    n_unique = int(np.unique(pairs, axis=0).shape[0]) if n else 0
    return ResidualStats(
        n=n,
        entropy=entropy,
        max_entropy=float(np.log2(n_bins)),
        mean_resultant=rbar,
        kappa=_kappa_from_rbar(rbar),
        is_finite_constellation=0 < n_unique <= constellation_max,
    )


def _resid_rms(
    dL: np.ndarray,
    dT: np.ndarray,
    dLr: np.ndarray,
    dTr: np.ndarray,
    *,
    theta_mod: int = SPF32_PHASE_MOD,
) -> float:
    """Combined RMS reconstruction error of a codec over both residual planes.

    ``dΘ`` error is measured on the circle (wrap-aware) in u-units so a center
    that straddles the seam is not penalised for the wrap.

    For ``theta_mod <= 2**32`` (SPF-16/32/64) the existing integer-modulo path
    is used unchanged (fits int64). For ``theta_mod > 2**32`` (SPF-128) integer
    modulo by 2^64 overflows int64, so the wrapped phase error is computed via
    the float-angle form.
    """
    dL = np.asarray(dL, dtype=np.int64)
    dT = np.asarray(dT, dtype=np.int64)
    eL = (dLr - dL).astype(np.float64)
    if theta_mod <= 2**32:
        # SPF-16/32/64: integer-modulo path, fits int64.
        half = theta_mod // 2
        dphi = (dTr - dT + half) % theta_mod - half
        e = np.concatenate([eL, dphi.astype(np.float64)])
    else:
        # SPF-128: theta_mod = 2^64 overflows int64; use the float-angle form.
        phase_to_rad = _TWO_PI / theta_mod
        dphi_float = np.angle(np.exp(1j * ((dTr - dT).astype(np.float64) * phase_to_rad)))
        # dphi_float is in radians ∈ (-π, π]; convert back to u-units (float64).
        dphi_u = dphi_float / phase_to_rad
        e = np.concatenate([eL, dphi_u])
    return float(np.sqrt(np.mean(e * e))) if e.size else 0.0


def select_codec(
    dL: np.ndarray,
    dT: np.ndarray,
    *,
    stats: ResidualStats | None = None,
    budget_rms: float | None = None,
    kappa_min: float = 4.0,
    k: int = 16,
    bits: int = 5,
    theta_mod: int = SPF32_PHASE_MOD,
) -> ResidualRef:
    """Pick the smallest-payload codec that meets ``budget_rms`` (design §7.2).

    White residuals (``entropy ≥ 0.98·max``) admit only ``NONE``/``RAW_INT16`` —
    a structured codec is never applied to white data. ``CODEBOOK_CIRCULAR`` is
    a candidate only when concentrated (``κ ≥ kappa_min``); ``PALETTE`` only on a
    finite constellation. Among admitted candidates the winner is the one with
    the fewest payload bytes whose reconstruction RMS ≤ budget (``RAW_INT16`` is
    the lossless backstop, error 0).

    Pass ``theta_mod=2^theta_bits`` to operate at a non-SPF-32 width; default
    ``SPF32_PHASE_MOD`` (2^16) keeps the SPF-32 path 100% unchanged.
    """
    if stats is None:
        stats = residual_stats(dL, dT, theta_mod=theta_mod)
    dL = np.asarray(dL, dtype=np.int64)
    dT = np.asarray(dT, dtype=np.int64)
    n = int(dT.size)
    if budget_rms is None:
        budget_rms = float("inf")

    # A genuinely zero residual is "the base is the whole story": NONE ships 0
    # payload bits and reconstructs exactly. This must be decided BEFORE the
    # white/structured split — an all-zero residual is a 1-symbol finite
    # constellation, so ``is_white`` is False and the white branch (the only
    # place NONE was a candidate) is never reached, leaving PALETTE to win at
    # 32 + n bits over a strictly-cheaper, equally-lossless NONE.
    if n and not np.any(dL) and not np.any(dT):
        return NoneCodec.encode()

    candidates: list[ResidualRef] = []
    if stats.is_white:
        # White: only the lossless / empty options. A structured table cannot
        # reduce index entropy here — it would only add overhead (PR#302 rule).
        candidates.append(RawInt16Codec.encode(dL, dT))
    else:
        if stats.kappa >= kappa_min:
            candidates.append(CodebookCircularCodec.encode(dL, dT, k=k, bL=bits, theta_mod=theta_mod))
        if stats.is_finite_constellation:
            candidates.append(PaletteCodec.encode(dL, dT))
        candidates.append(UniformCodec.encode(dL, dT, bL=bits, bT=bits))
        candidates.append(RawInt16Codec.encode(dL, dT))

    best: tuple[int, ResidualRef] | None = None
    for ref in candidates:
        codec = _CODEC_BY_ID[ref.codec_id]
        dLr, dTr = codec.decode(ref.payload, n)
        if _resid_rms(dL, dT, dLr, dTr, theta_mod=theta_mod) > budget_rms:
            continue
        bits_cost = codec.storage_bits(ref.payload)
        if best is None or bits_cost < best[0]:
            best = (bits_cost, ref)
    if best is None:  # nothing met the budget → lossless backstop
        return RawInt16Codec.encode(dL, dT)
    return best[1]


def register_defaults() -> None:
    register_residual_codec(NoneCodec())
    register_residual_codec(RawInt16Codec())
    register_residual_codec(UniformCodec())
    register_residual_codec(CodebookCircularCodec())
    register_residual_codec(PaletteCodec())


_CODEC_BY_ID: dict[int, Any] = {
    int(ResidualCodecKind.NONE): NoneCodec(),
    int(ResidualCodecKind.RAW_INT16): RawInt16Codec(),
    int(ResidualCodecKind.UNIFORM): UniformCodec(),
    int(ResidualCodecKind.CODEBOOK_CIRCULAR): CodebookCircularCodec(),
    int(ResidualCodecKind.PALETTE): PaletteCodec(),
}

register_defaults()


__all__ = [
    "CodebookCircularCodec",
    "NoneCodec",
    "PaletteCodec",
    "RawInt16Codec",
    "ResidualStats",
    "UniformCodec",
    "residual_stats",
    "select_codec",
]
