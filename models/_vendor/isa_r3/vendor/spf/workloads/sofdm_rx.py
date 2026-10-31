"""Reference implementations for sOFDM RX workload contracts.

The four workloads (``sofdm_rx_qpsk_flat``, ``sofdm_rx_qam16_flat``,
``sofdm_rx_qam16_multipath``, ``sofdm_rx_qam16_pilot_estimated``) share the
TX → ADC → RX pipeline from :mod:`spf.dsp.ofdm_receiver`. These helpers
unpack the case parameters, build the plan + pilot pattern, drive
``run_sofdm_adc_roundtrip``, and return the contract-declared output
arrays — pure math, no I/O, no print.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.dsp.ofdm_pilots import DEFAULT_PILOT_SEED, comb_pilot_pattern, data_subcarriers
from spf.dsp.ofdm_receiver import ADCConfig, run_sofdm_adc_roundtrip
from spf.dsp.ofdm_rx_plan import ChannelPolicy, OFDMRXPlan

# QAM16 bits per subcarrier — used to mask payload bits into per-SC groups.
_QAM16_BPS: int = 4


def _active_subcarriers(n_fft: int, n_active: int) -> npt.NDArray[np.uint32]:
    """Return ``n_active`` non-DC subcarriers on a fixed stride starting at bin 1.

    Stride is ``max(1, (n_fft - 1) // n_active)``. When ``n_active * stride <
    n_fft - 1`` the chosen bins cluster in the low band rather than spanning
    the full active grid — adequate for the workload contracts in this module
    (multipath/pilot fixtures don't depend on high-band coverage), but not a
    DC-symmetric, full-band layout.
    """
    if n_active <= 0 or n_active > n_fft - 1:
        raise ValueError(f"invalid n_active={n_active} for n_fft={n_fft}")
    stride = max(1, (n_fft - 1) // n_active)
    indices = np.arange(1, 1 + stride * n_active, stride, dtype=np.uint32)
    if indices.size > n_active:
        indices = indices[:n_active]
    return indices


def _multipath_channel(n_fft: int) -> npt.NDArray[np.complex128]:
    taps = np.asarray(
        [1.0 + 0.0j, 0.08 - 0.03j, -0.05 + 0.02j, 0.04 + 0.01j, -0.02 - 0.01j],
        dtype=np.complex128,
    )
    impulse = np.zeros(n_fft, dtype=np.complex128)
    impulse[: taps.size] = taps
    return np.fft.fft(impulse)


def _rand_bits(seed: int, n: int) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def _known_h_reference(
    *,
    seed: int,
    n_fft: int,
    ncp: int,
    n_active: int,
    modulation: str,
    channel_h: npt.NDArray[np.complex128],
    bit_depth: int = 14,
    snr_db: float | None = None,
) -> dict[str, np.ndarray]:
    active = _active_subcarriers(n_fft, n_active)
    plan = OFDMRXPlan(
        n_fft=n_fft,
        ncp=ncp,
        modulation=modulation,  # type: ignore[arg-type]
        active_subcarriers=active,
        channel_policy=ChannelPolicy.KNOWN,
        channel_h=channel_h,
    )
    bps = 2 if modulation == "qpsk" else 4
    bits = _rand_bits(seed, n_active * bps)
    rt = run_sofdm_adc_roundtrip(
        bits,
        receiver=plan,
        adc=ADCConfig(bit_depth=bit_depth, full_scale=1.0, snr_db=snr_db),
        channel_h=channel_h,
    )
    return {
        "rx_bits": np.ascontiguousarray(rt.receiver.rx_bits, dtype=np.uint8),
        "equalized_words": np.ascontiguousarray(
            rt.receiver.equalized_words, dtype=np.uint32
        ),
        "agc_gain": np.asarray([rt.receiver.agc_gain], dtype=np.float64),
    }


def generate_qpsk_flat_reference(
    seed: int,
    n_fft: int,
    ncp: int,
    n_active: int,
    bit_depth: int = 14,
    snr_db: float | None = None,
) -> dict[str, np.ndarray]:
    return _known_h_reference(
        seed=seed,
        n_fft=n_fft,
        ncp=ncp,
        n_active=n_active,
        modulation="qpsk",
        channel_h=np.ones(n_fft, dtype=np.complex128),
        bit_depth=bit_depth,
        snr_db=snr_db,
    )


def generate_qam16_flat_reference(
    seed: int,
    n_fft: int,
    ncp: int,
    n_active: int,
    bit_depth: int = 14,
    snr_db: float | None = None,
) -> dict[str, np.ndarray]:
    return _known_h_reference(
        seed=seed,
        n_fft=n_fft,
        ncp=ncp,
        n_active=n_active,
        modulation="qam16",
        channel_h=np.ones(n_fft, dtype=np.complex128),
        bit_depth=bit_depth,
        snr_db=snr_db,
    )


def generate_qam16_multipath_reference(
    seed: int,
    n_fft: int,
    ncp: int,
    n_active: int,
    bit_depth: int = 14,
    snr_db: float | None = None,
) -> dict[str, np.ndarray]:
    return _known_h_reference(
        seed=seed,
        n_fft=n_fft,
        ncp=ncp,
        n_active=n_active,
        modulation="qam16",
        channel_h=_multipath_channel(n_fft),
        bit_depth=bit_depth,
        snr_db=snr_db,
    )


def generate_qam16_pilot_estimated_reference(
    seed: int,
    n_fft: int,
    ncp: int,
    n_active: int,
    pilot_stride: int,
    pilot_boost_db: float = 3.0,
    pilot_seed: int = DEFAULT_PILOT_SEED,
    bit_depth: int = 14,
) -> dict[str, np.ndarray]:
    active = _active_subcarriers(n_fft, n_active)
    pilots = comb_pilot_pattern(
        n_fft=n_fft,
        active=active,
        stride=pilot_stride,
        seed=pilot_seed,
        boost_db=pilot_boost_db,
    )
    plan = OFDMRXPlan(
        n_fft=n_fft,
        ncp=ncp,
        modulation="qam16",
        active_subcarriers=active,
        channel_policy=ChannelPolicy.PILOT_ESTIMATED,
    )
    channel = _multipath_channel(n_fft)
    data_idx = data_subcarriers(active, pilots)
    bits = _rand_bits(seed, data_idx.size * _QAM16_BPS)
    rt = run_sofdm_adc_roundtrip(
        bits,
        receiver=plan,
        adc=ADCConfig(bit_depth=bit_depth, full_scale=1.0),
        channel_h=channel,
        pilots=pilots,
    )
    # rx_bits from receive_sofdm_frame covers every active subcarrier (data
    # + pilot). The contract surface is payload-only, so we mask to data
    # positions matching the bit ordering tx_bits used.
    pilot_set = {int(x) for x in pilots.subcarrier_indices.tolist()}
    is_data_sc = np.asarray(
        [int(k) not in pilot_set for k in active.tolist()], dtype=bool
    )
    is_data_bit = np.repeat(is_data_sc, _QAM16_BPS)
    payload_bits = rt.receiver.rx_bits[is_data_bit]
    return {
        "rx_bits": np.ascontiguousarray(payload_bits, dtype=np.uint8),
        "equalized_words": np.ascontiguousarray(
            rt.receiver.equalized_words, dtype=np.uint32
        ),
    }


__all__ = [
    "generate_qam16_flat_reference",
    "generate_qam16_multipath_reference",
    "generate_qam16_pilot_estimated_reference",
    "generate_qpsk_flat_reference",
]
