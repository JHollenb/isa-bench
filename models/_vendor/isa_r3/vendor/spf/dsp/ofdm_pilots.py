"""Pilot patterns and pilot-based channel estimation for sOFDM RX.

This module supplies the data structures and compatibility-mode estimator
used by ``receive_sofdm_frame`` when ``OFDMRXPlan.channel_policy`` is
:data:`ChannelPolicy.PILOT_ESTIMATED`.

Compatibility-mode interpolation decodes the received pilot bins to Cartesian
floats, divides by the known transmitted pilot values to obtain ``H_hat`` at
the pilot subcarriers, linearly interpolates across the active grid, then
re-encodes to SPF words. The SPF-native log-phase interpolation mode runs the
native RX-6b log-linear pilot interpolator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from spf.dsp.modulation import psk_modulate_spf, qam16_modulate_spf
from spf.dsp.ofdm_compat import Modulation
from spf.dsp.ofdm_grid import OFDMBinClass, OFDMResourceGrid
from spf.dsp.ofdm_rx_plan import ChannelPolicy, OFDMRXPlan
from spf.tensor import SPF32_WORD_ZERO, decode_words, encode_words

InterpMode = Literal["compat", "spf_native_logphase_interp", "pinv"]

DEFAULT_COMB_STRIDE: int = 4
DEFAULT_PILOT_BOOST_DB: float = 3.0
DEFAULT_PILOT_SEED: int = 20260521
# Default CIR-basis order for the pinv channel estimator. A frequency-selective
# channel with ``L`` significant taps has an exactly ``L``-dimensional
# frequency response, so an ``L``-tap least-squares fit recovers the smooth
# trend with far less noise than per-pilot linear interpolation (earlier sim:
# ~3x lower channel MSE on a smooth multipath channel). Capped to the pilot
# count so the system stays over-determined.
DEFAULT_PINV_CIR_TAPS: int = 8


@dataclass(frozen=True, slots=True)
class PilotPattern:
    """Pilot reference symbols anchored to (symbol, subcarrier) positions.

    ``values`` carries the transmitted pilot complex amplitude per pilot
    subcarrier — assumed identical across the listed ``symbol_indices`` so a
    single sequence covers comb and block patterns alike.
    """

    pattern_id: str
    symbol_indices: npt.NDArray[np.uint32]
    subcarrier_indices: npt.NDArray[np.uint32]
    values: npt.NDArray[np.complex128]
    boost_db: float = 0.0

    def __post_init__(self) -> None:
        symbol_indices = np.ascontiguousarray(
            np.asarray(self.symbol_indices), dtype=np.uint32
        ).ravel()
        subcarrier_indices = np.ascontiguousarray(
            np.asarray(self.subcarrier_indices), dtype=np.uint32
        ).ravel()
        values = np.ascontiguousarray(np.asarray(self.values), dtype=np.complex128).ravel()
        if subcarrier_indices.size == 0:
            raise ValueError("PilotPattern requires at least one pilot subcarrier")
        if symbol_indices.size == 0:
            raise ValueError("PilotPattern requires at least one pilot symbol index")
        if subcarrier_indices.size != values.size:
            raise ValueError(
                "PilotPattern values must match the number of pilot subcarriers "
                f"(got {values.size} values for {subcarrier_indices.size} subcarriers)"
            )
        if np.any(values == 0.0):
            raise ValueError("PilotPattern values must be nonzero")
        if not np.all(subcarrier_indices[1:] > subcarrier_indices[:-1]):
            raise ValueError("PilotPattern subcarrier_indices must be strictly increasing")
        if not np.all(symbol_indices[1:] > symbol_indices[:-1]):
            raise ValueError("PilotPattern symbol_indices must be strictly increasing")
        if subcarrier_indices.size and int(subcarrier_indices[0]) == 0:
            raise ValueError("DC subcarrier 0 cannot carry a pilot (reserved DC_NULL)")
        object.__setattr__(self, "symbol_indices", symbol_indices)
        object.__setattr__(self, "subcarrier_indices", subcarrier_indices)
        object.__setattr__(self, "values", values)

    @property
    def n_pilots(self) -> int:
        return int(self.subcarrier_indices.size)

    @property
    def pilot_spacing(self) -> int:
        if self.n_pilots <= 1:
            return 0
        deltas = np.diff(self.subcarrier_indices.astype(np.int64))
        return int(deltas[0]) if np.all(deltas == deltas[0]) else -1


@dataclass(frozen=True, slots=True)
class ChannelEstimate:
    """Per-active-subcarrier channel estimate built from pilots."""

    h_active: npt.NDArray[np.complex128]
    h_words: npt.NDArray[np.uint32]
    h_inv_words: npt.NDArray[np.uint32]
    policy: ChannelPolicy
    source: Literal[
        "compat_decode_interp_encode",
        "spf_native_logphase_interp",
        "pinv_cir_lsq",
    ]
    mse_db: float | None
    rms_pilot_mag: float
    rmse_data_interp: float | None
    n_pilots: int
    pilot_spacing: int
    stages: tuple[str, ...]


def comb_pilot_pattern(
    n_fft: int,
    active: npt.ArrayLike,
    *,
    stride: int = DEFAULT_COMB_STRIDE,
    seed: int = DEFAULT_PILOT_SEED,
    boost_db: float = DEFAULT_PILOT_BOOST_DB,
    symbol_indices: npt.ArrayLike | None = None,
) -> PilotPattern:
    """Comb pilot pattern: every ``stride``-th active subcarrier carries a pilot."""
    active_arr = np.ascontiguousarray(np.asarray(active), dtype=np.uint32).ravel()
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if active_arr.size == 0:
        raise ValueError("active subcarrier set is empty")
    pilot_idx = active_arr[::stride]
    if pilot_idx.size == 0:
        raise ValueError("comb stride larger than active grid; no pilots produced")
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=pilot_idx.size)
    boost = float(10.0 ** (boost_db / 20.0))
    values = boost * np.exp(1j * angles)
    if symbol_indices is None:
        sym_idx = np.asarray([0], dtype=np.uint32)
    else:
        sym_idx = np.ascontiguousarray(np.asarray(symbol_indices), dtype=np.uint32).ravel()
    if int(pilot_idx[0]) == 0:
        # First pilot is reserved DC. Drop it and shift the values.
        pilot_idx = pilot_idx[1:]
        values = values[1:]
    return PilotPattern(
        pattern_id=f"comb_every_{stride}",
        symbol_indices=sym_idx,
        subcarrier_indices=np.ascontiguousarray(pilot_idx, dtype=np.uint32),
        values=np.ascontiguousarray(values, dtype=np.complex128),
        boost_db=boost_db,
    )


def block_pilot_pattern(
    n_fft: int,
    active: npt.ArrayLike,
    *,
    symbol_idx: int,
    seed: int = DEFAULT_PILOT_SEED,
    boost_db: float = 0.0,
) -> PilotPattern:
    """Block pilot pattern: every active subcarrier of one symbol carries a pilot."""
    active_arr = np.ascontiguousarray(np.asarray(active), dtype=np.uint32).ravel()
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=active_arr.size)
    boost = float(10.0 ** (boost_db / 20.0))
    values = boost * np.exp(1j * angles)
    return PilotPattern(
        pattern_id="block_one_symbol",
        symbol_indices=np.asarray([symbol_idx], dtype=np.uint32),
        subcarrier_indices=active_arr,
        values=np.ascontiguousarray(values, dtype=np.complex128),
        boost_db=boost_db,
    )


def data_subcarriers(
    active_subcarriers: npt.ArrayLike,
    pilots: PilotPattern,
) -> npt.NDArray[np.uint32]:
    """Return the active subcarrier indices that are not pilot positions."""
    active = np.ascontiguousarray(np.asarray(active_subcarriers), dtype=np.uint32).ravel()
    pilot_set = set(int(x) for x in pilots.subcarrier_indices.tolist())
    return np.asarray([int(k) for k in active if int(k) not in pilot_set], dtype=np.uint32)


def tx_frame_with_pilots(
    bits: npt.ArrayLike,
    pilots: PilotPattern,
    *,
    modulation: Modulation,
    n_fft: int,
    active_subcarriers: npt.ArrayLike,
) -> OFDMResourceGrid:
    """Build a TX grid with data bits and pilot SPF words.

    Bits map in symbol-major order to active subcarriers that are not pilot
    positions for that symbol. Pilot subcarriers receive ``pilots.values``
    (encoded to SPF words). The grid is sized through the largest pilot symbol
    index and marks pilot positions as :class:`OFDMBinClass.PILOT`.
    """
    active = np.ascontiguousarray(np.asarray(active_subcarriers), dtype=np.uint32).ravel()
    if np.any(active == 0) or np.any(active >= n_fft):
        raise ValueError("active_subcarriers must be non-DC bins inside n_fft")
    if not np.all(active[1:] > active[:-1]):
        raise ValueError("active_subcarriers must be strictly increasing")
    pilot_set = set(int(x) for x in pilots.subcarrier_indices.tolist())
    if not pilot_set.issubset(set(int(k) for k in active.tolist())):
        raise ValueError("pilot subcarriers must be a subset of active_subcarriers")
    pilot_symbols = sorted({int(x) for x in pilots.symbol_indices.tolist()})
    n_symbols = pilot_symbols[-1] + 1
    data_positions: list[tuple[int, int]] = []
    active_ints = [int(k) for k in active.tolist()]
    for sym_idx in range(n_symbols):
        symbol_has_pilots = sym_idx in pilot_symbols
        for sc_idx in active_ints:
            if symbol_has_pilots and sc_idx in pilot_set:
                continue
            data_positions.append((sym_idx, sc_idx))
    n_data = len(data_positions)

    bits_arr = np.ascontiguousarray(np.asarray(bits), dtype=np.uint8).ravel()
    if modulation == "qpsk":
        if bits_arr.size != n_data * 2:
            raise ValueError(
                f"qpsk requires {n_data * 2} bits for {n_data} data bins, "
                f"got {bits_arr.size}"
            )
        data_symbols = psk_modulate_spf(bits_arr, 4)
    elif modulation == "qam16":
        if bits_arr.size != n_data * 4:
            raise ValueError(
                f"qam16 requires {n_data * 4} bits for {n_data} data bins, "
                f"got {bits_arr.size}"
            )
        data_symbols = qam16_modulate_spf(bits_arr)
    else:
        raise ValueError(f"unsupported modulation {modulation!r}")

    shape = (1, n_symbols, n_fft, 1)
    words = np.full(shape, SPF32_WORD_ZERO, dtype=np.uint32)
    classes = np.full(shape, int(OFDMBinClass.GUARD), dtype=np.int16)
    classes[:, :, 0, :] = int(OFDMBinClass.DC_NULL)
    classes[0, :, active, 0] = int(OFDMBinClass.DATA)
    for word, (sym_idx, sc_idx) in zip(data_symbols, data_positions, strict=True):
        words[0, sym_idx, sc_idx, 0] = word

    pilot_words = encode_words(pilots.values)
    for sym_idx in pilot_symbols:
        words[0, sym_idx, pilots.subcarrier_indices, 0] = pilot_words
        classes[0, sym_idx, pilots.subcarrier_indices, 0] = int(OFDMBinClass.PILOT)
    return OFDMResourceGrid(words, classes)


def estimate_channel_from_pilots(
    rx_grid: OFDMResourceGrid,
    pilots: PilotPattern,
    *,
    plan: OFDMRXPlan,
    mode: InterpMode = "compat",
    cir_taps: int = DEFAULT_PINV_CIR_TAPS,
) -> ChannelEstimate:
    """Estimate the per-active-subcarrier channel response from pilot bins.

    ``mode='compat'`` decodes Y[pilot] to Cartesian floats, divides by the
    known pilot values, linearly interpolates across the active grid, and
    re-encodes. ``mode='spf_native_logphase_interp'`` runs the same H_hat
    division at pilot positions but interpolates entirely in SPF word space
    via ``spf32_pilot_interp_loglinear_array`` so no Cartesian decode lands
    at data positions. ``mode='pinv'`` fits a low-order CIR (channel impulse
    response) model to the pilot H_hat by least squares (Moore-Penrose
    pseudoinverse) and projects it onto the active grid — a small add-domain
    boundary solve at RX egress that yields ~3x lower channel error than
    linear interpolation on a smooth multipath channel.

    **Honest scoping:** the pinv fit is an FP64 add-domain boundary solve run
    once per estimate at RX egress; it is *not* on the multiplier-free SPF hot
    path and is *not* counted as an SPF-native win.
    """
    if mode not in ("compat", "spf_native_logphase_interp", "pinv"):
        raise ValueError(f"unsupported pilot interpolation mode {mode!r}")
    if plan.channel_policy is not ChannelPolicy.PILOT_ESTIMATED:
        raise ValueError(
            "estimate_channel_from_pilots called with channel_policy="
            f"{plan.channel_policy.name}; expected PILOT_ESTIMATED"
        )
    if mode == "spf_native_logphase_interp":
        return _estimate_channel_spf_native(rx_grid, pilots, plan=plan)
    if mode == "pinv":
        return _estimate_channel_pinv(rx_grid, pilots, plan=plan, cir_taps=cir_taps)

    active = np.ascontiguousarray(plan.active_subcarriers, dtype=np.uint32).ravel()
    pilot_vals = pilots.values
    n_symbols = int(rx_grid.shape[1])
    sym_idx = int(pilots.symbol_indices[0])
    if sym_idx >= n_symbols:
        raise ValueError(
            f"pilot symbol_idx={sym_idx} out of range for rx_grid with {n_symbols} symbols"
        )

    y_pilot_words = rx_grid.words[0, sym_idx, pilots.subcarrier_indices, 0]
    y_pilot = decode_words(y_pilot_words)
    h_at_pilots = y_pilot / pilot_vals
    # ``rms_pilot_mag`` reports RMS |H_hat| at pilot subcarriers (no true-H
    # reference is available at runtime; the field name is retained for
    # ChannelEstimate dataclass / evidence-doc back-compat).
    rms_pilot_mag = float(np.sqrt(np.mean(np.abs(h_at_pilots) ** 2)))

    # Linear interp (real + imag) across the active grid using pilot anchors.
    # PilotPattern.__post_init__ enforces strictly ascending subcarrier_indices,
    # so the inputs are already sorted; pass straight to np.interp.
    pilot_sc_f = pilots.subcarrier_indices.astype(np.float64)
    active_f = active.astype(np.float64)
    h_real = np.interp(active_f, pilot_sc_f, h_at_pilots.real)
    h_imag = np.interp(active_f, pilot_sc_f, h_at_pilots.imag)
    h_active = (h_real + 1j * h_imag).astype(np.complex128)
    if np.any(h_active == 0.0):
        raise ValueError("pilot interpolation produced a zero channel response at an active bin")

    h_words = encode_words(h_active)
    h_inv_words = encode_words(1.0 / h_active)

    return ChannelEstimate(
        h_active=h_active,
        h_words=h_words,
        h_inv_words=h_inv_words,
        policy=plan.channel_policy,
        source="compat_decode_interp_encode",
        mse_db=None,
        rms_pilot_mag=rms_pilot_mag,
        rmse_data_interp=None,
        n_pilots=pilots.n_pilots,
        pilot_spacing=pilots.pilot_spacing,
        stages=("pilot_decode", "fp_linear_interp", "pilot_encode"),
    )


def _estimate_channel_spf_native(
    rx_grid: OFDMResourceGrid,
    pilots: PilotPattern,
    *,
    plan: OFDMRXPlan,
) -> ChannelEstimate:
    """RX-6b: SPF-native log-linear pilot interp.

    The H_hat division at pilot positions runs in Cartesian (decode Y, divide
    by known P, re-encode) — that's setup-side and the caller is expected to
    route the count into ``pilot_cartesian_decode_calls``. The interpolation
    across data subcarriers runs through the native
    ``spf32_pilot_interp_loglinear_array`` kernel without decoding payload
    bins.
    """
    from spf.native import spf32 as c_spf32

    active = np.ascontiguousarray(plan.active_subcarriers, dtype=np.uint32).ravel()
    n_symbols = int(rx_grid.shape[1])
    sym_idx = int(pilots.symbol_indices[0])
    if sym_idx >= n_symbols:
        raise ValueError(
            f"pilot symbol_idx={sym_idx} out of range for rx_grid with {n_symbols} symbols"
        )
    # Pilot H_hat via Cartesian decode of Y[pilot] / P. This is the only
    # decode in the path and is intentionally counted at the caller's pilot
    # counter.
    y_pilot_words = rx_grid.words[0, sym_idx, pilots.subcarrier_indices, 0]
    y_pilot = decode_words(y_pilot_words)
    h_at_pilots = y_pilot / pilots.values
    rms_pilot_mag = float(np.sqrt(np.mean(np.abs(h_at_pilots) ** 2)))
    h_pilot_words = encode_words(h_at_pilots)
    # Native interp across active grid.
    h_words = c_spf32.pilot_interp_loglinear(
        h_pilot_words,
        pilots.subcarrier_indices,
        plan.n_fft,
        active,
    )
    # Decode once to derive h_active (used by AGC), then encode h_inv.
    h_active = decode_words(h_words)
    if np.any(h_active == 0.0):
        raise ValueError(
            "native pilot interpolation produced a zero channel response "
            "at an active bin"
        )
    h_inv_words = encode_words(1.0 / h_active)
    return ChannelEstimate(
        h_active=h_active,
        h_words=h_words,
        h_inv_words=h_inv_words,
        policy=plan.channel_policy,
        source="spf_native_logphase_interp",
        mse_db=None,
        rms_pilot_mag=rms_pilot_mag,
        rmse_data_interp=None,
        n_pilots=pilots.n_pilots,
        pilot_spacing=pilots.pilot_spacing,
        stages=("pilot_decode", "spf_native_interp", "pilot_encode"),
    )


def _estimate_channel_pinv(
    rx_grid: OFDMResourceGrid,
    pilots: PilotPattern,
    *,
    plan: OFDMRXPlan,
    cir_taps: int,
) -> ChannelEstimate:
    """CIR-basis least-squares (pseudoinverse) channel estimate.

    The pilot H_hat = Y[pilot] / P is a noisy sample of the channel frequency
    response ``H[k] = sum_t h[t] * exp(-j 2 pi k t / N)``. A frequency-selective
    channel has only a few significant taps, so we fit an ``M``-tap CIR ``h`` by
    least squares to the pilot observations and synthesise ``H`` on the active
    grid from the fitted taps. The pseudoinverse of the (P x M) DFT submatrix is
    the minimum-norm least-squares solver; over-determined (P > M) it averages
    pilot noise across taps instead of carrying it through per-pilot, which is
    why it beats linear interpolation on a smooth channel.

    Add-domain boundary solve at RX egress — NOT an SPF-native-win-counted op.
    """
    active = np.ascontiguousarray(plan.active_subcarriers, dtype=np.uint32).ravel()
    n_fft = int(plan.n_fft)
    n_symbols = int(rx_grid.shape[1])
    sym_idx = int(pilots.symbol_indices[0])
    if sym_idx >= n_symbols:
        raise ValueError(
            f"pilot symbol_idx={sym_idx} out of range for rx_grid with {n_symbols} symbols"
        )
    # Pilot H_hat via Cartesian decode of Y[pilot] / P (the only decode; counted
    # at the caller's pilot counter, as in the other estimators).
    y_pilot_words = rx_grid.words[0, sym_idx, pilots.subcarrier_indices, 0]
    y_pilot = decode_words(y_pilot_words)
    h_at_pilots = y_pilot / pilots.values
    rms_pilot_mag = float(np.sqrt(np.mean(np.abs(h_at_pilots) ** 2)))

    pilot_sc = pilots.subcarrier_indices.astype(np.int64)
    # Keep the system over-determined: never fit more taps than pilots.
    n_basis = int(max(1, min(cir_taps, pilot_sc.size)))
    taps = np.arange(n_basis, dtype=np.int64)
    twopi_over_n = 2.0 * np.pi / float(n_fft)
    # DFT submatrix A[p, t] = exp(-j 2 pi k_p t / N); H_pilot ≈ A @ h.
    a_pilot = np.exp(-1j * twopi_over_n * np.outer(pilot_sc, taps))
    h_cir = np.linalg.pinv(a_pilot) @ h_at_pilots  # least-squares CIR taps.
    # Synthesise H on the active grid from the fitted taps.
    a_active = np.exp(-1j * twopi_over_n * np.outer(active.astype(np.int64), taps))
    h_active = (a_active @ h_cir).astype(np.complex128)
    # Residual of the fit at pilot positions (RMSE of A @ h vs measured H_hat).
    fit_residual = float(np.sqrt(np.mean(np.abs(a_pilot @ h_cir - h_at_pilots) ** 2)))
    if np.any(h_active == 0.0):
        raise ValueError(
            "pinv channel estimate produced a zero channel response at an active bin"
        )
    h_words = encode_words(h_active)
    h_inv_words = encode_words(1.0 / h_active)
    return ChannelEstimate(
        h_active=h_active,
        h_words=h_words,
        h_inv_words=h_inv_words,
        policy=plan.channel_policy,
        source="pinv_cir_lsq",
        mse_db=None,
        rms_pilot_mag=rms_pilot_mag,
        rmse_data_interp=fit_residual,
        n_pilots=pilots.n_pilots,
        pilot_spacing=pilots.pilot_spacing,
        stages=("pilot_decode", "cir_lsq_pinv", "active_synthesis", "pilot_encode"),
    )


__all__ = [
    "DEFAULT_COMB_STRIDE",
    "DEFAULT_PILOT_BOOST_DB",
    "DEFAULT_PINV_CIR_TAPS",
    "ChannelEstimate",
    "InterpMode",
    "PilotPattern",
    "block_pilot_pattern",
    "comb_pilot_pattern",
    "data_subcarriers",
    "estimate_channel_from_pilots",
    "tx_frame_with_pilots",
]
