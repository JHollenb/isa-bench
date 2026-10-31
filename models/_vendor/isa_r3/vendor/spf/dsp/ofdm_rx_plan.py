"""sOFDM receive plan, policy enums, and structural counters.

``OFDMRXPlan`` is the forward-compatible configuration object for the spectral
OFDM receive surface. It carries channel-source, equalizer, CPE, timing, and
ADC policies so RX behaviour is fully captured by a single immutable value
that flows through both the compatibility (NumPy-FFT) path and the resident
spectral-bin path.

``OFDMRXCounters`` records the structural hot-path counts that every receive
call must report so claim-ledger evidence docs can cite boundary imports,
Cartesian decodes, SPF multiplies, log-projection demaps, etc., without having
to instrument each test individually.

This module deliberately does not import from :mod:`spf.dsp.ofdm_receiver` to
avoid a cycle; the receiver imports from here.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from spf.dsp.ofdm_compat import Modulation

if TYPE_CHECKING:
    from spf.dsp.ofdm_receiver import OFDMReceiverConfig


class ChannelPolicy(Enum):
    """How the receiver obtains the per-active-subcarrier channel response."""

    KNOWN = "known"
    PILOT_ESTIMATED = "pilot_estimated"
    PRECOMPUTED_H_INV = "precomputed_h_inv"


class EqualizerKind(Enum):
    """Equalizer family applied to the diagonal channel inverse."""

    ZF = "zf"
    MMSE = "mmse"


class CPEPolicy(Enum):
    """Common-phase-error correction policy."""

    NONE = "none"
    PILOT_COMMON_PHASE = "pilot_common_phase"


class TimingPolicy(Enum):
    """Per-symbol timing residual correction policy."""

    NONE = "none"
    PILOT_PHASE_SLOPE = "pilot_phase_slope"


class ADCPolicy(Enum):
    """Declared ADC behaviour. Resampled inputs require explicit timing sync."""

    FIXED_RATE_IQ = "fixed_rate_iq"
    RESAMPLED_REQUIRES_SYNC = "resampled_requires_sync"


@dataclass(frozen=True, slots=True)
class OFDMRXPlan:
    """Forward-compatible RX configuration.

    The legacy :class:`spf.dsp.ofdm_receiver.OFDMReceiverConfig` stays as a
    compatibility alias; :func:`from_receiver_config` converts it into an
    ``OFDMRXPlan`` with sensible policy defaults.
    """

    n_fft: int
    ncp: int
    modulation: Modulation
    active_subcarriers: npt.NDArray[np.uint32]
    channel_policy: ChannelPolicy = ChannelPolicy.KNOWN
    equalizer: EqualizerKind = EqualizerKind.ZF
    cpe_policy: CPEPolicy = CPEPolicy.NONE
    timing_policy: TimingPolicy = TimingPolicy.NONE
    adc_policy: ADCPolicy = ADCPolicy.FIXED_RATE_IQ
    channel_h: npt.NDArray[np.complex128] | None = None
    agc: bool = True
    qam16_threshold_l: int = 256
    cpe_theta: int = 0
    phase_bits: int = 10
    agc_l_offset: int = 0

    def __post_init__(self) -> None:
        if self.equalizer is EqualizerKind.MMSE:
            raise NotImplementedError("EqualizerKind.MMSE is not implemented yet")
        if self.adc_policy is ADCPolicy.RESAMPLED_REQUIRES_SYNC:
            raise NotImplementedError("ADCPolicy.RESAMPLED_REQUIRES_SYNC requires explicit timing recovery")
        if self.channel_policy is ChannelPolicy.KNOWN and self.channel_h is None:
            object.__setattr__(self, "channel_h", None)
        active = np.ascontiguousarray(np.asarray(self.active_subcarriers), dtype=np.uint32)
        object.__setattr__(self, "active_subcarriers", active)
        if self.channel_h is not None:
            object.__setattr__(
                self,
                "channel_h",
                np.ascontiguousarray(np.asarray(self.channel_h), dtype=np.complex128),
            )

    @property
    def n_active(self) -> int:
        return int(self.active_subcarriers.size)


@dataclass(frozen=True, slots=True)
class OFDMRXCounters:
    """Structural hot-path counters reported by every receive call.

    Each field counts a discrete claim-relevant event. ``boundary_imports`` and
    ``reducer_calls`` are reserved for future MIMO / coherent-sum paths and
    stay zero for SISO diagonal RX. ``pilot_cartesian_decode_calls`` is kept
    separate from ``cartesian_decode_calls`` so the resident-bins path can
    prove its hot-path stays Cartesian-free even when the pilot setup path
    runs a compat-mode decode at pilot positions. ``spf_fft_calls`` counts
    word-domain SPF transforms (``profile="fb7"`` / ``profile="v02"`` on
    :func:`spf.dsp.ofdm_receiver.receive_sofdm_frame`) and stays 0 on the
    NumPy-FFT profile, where ``numpy_fft_calls`` carries the count instead.
    """

    numpy_fft_calls: int = 0
    cartesian_decode_calls: int = 0
    spf_mul_calls: int = 0
    spf_mul_calls_operational: int = 0
    spf_mul_calls_diagnostic: int = 0
    qam_logproj_calls: int = 0
    qpsk_phase_demap_calls: int = 0
    spf_encode_calls: int = 0
    spf_decode_calls: int = 0
    equalizer_inverts: int = 0
    boundary_imports: int = 0
    reducer_calls: int = 0
    fb7_butterfly_calls: int = 0
    spf_fft_calls: int = 0
    agc_applied: bool = False
    pilot_cartesian_decode_calls: int = 0

    def increment(self, **deltas: int | bool) -> OFDMRXCounters:
        """Return a copy with the given counter fields incremented or replaced."""
        kwargs: dict[str, object] = {}
        for name, delta in deltas.items():
            current = getattr(self, name)
            if isinstance(current, bool):
                kwargs[name] = bool(delta)
            else:
                kwargs[name] = int(current) + int(delta)
        return replace(self, **kwargs)  # type: ignore[arg-type]


@dataclass
class _MutableCounters:
    """Mutable accumulator used while a receive call executes.

    The receive path bumps fields in place during stages and freezes the
    result into :class:`OFDMRXCounters` before constructing the receiver
    result. Keeping the mutable form private avoids leaking a mutable counter
    type through the public surface.
    """

    numpy_fft_calls: int = 0
    cartesian_decode_calls: int = 0
    spf_mul_calls: int = 0
    spf_mul_calls_operational: int = 0
    spf_mul_calls_diagnostic: int = 0
    qam_logproj_calls: int = 0
    qpsk_phase_demap_calls: int = 0
    spf_encode_calls: int = 0
    spf_decode_calls: int = 0
    equalizer_inverts: int = 0
    boundary_imports: int = 0
    reducer_calls: int = 0
    fb7_butterfly_calls: int = 0
    spf_fft_calls: int = 0
    agc_applied: bool = False
    pilot_cartesian_decode_calls: int = 0

    def bump_spf_mul_operational(self, n: int = 1) -> None:
        """Increment the operational spf-mul counter and the total."""
        self.spf_mul_calls += n
        self.spf_mul_calls_operational += n

    def bump_spf_mul_diagnostic(self, n: int = 1) -> None:
        """Increment the diagnostic spf-mul counter and the total."""
        self.spf_mul_calls += n
        self.spf_mul_calls_diagnostic += n

    def freeze(self) -> OFDMRXCounters:
        return OFDMRXCounters(
            numpy_fft_calls=self.numpy_fft_calls,
            cartesian_decode_calls=self.cartesian_decode_calls,
            spf_mul_calls=self.spf_mul_calls,
            spf_mul_calls_operational=self.spf_mul_calls_operational,
            spf_mul_calls_diagnostic=self.spf_mul_calls_diagnostic,
            qam_logproj_calls=self.qam_logproj_calls,
            qpsk_phase_demap_calls=self.qpsk_phase_demap_calls,
            spf_encode_calls=self.spf_encode_calls,
            spf_decode_calls=self.spf_decode_calls,
            equalizer_inverts=self.equalizer_inverts,
            boundary_imports=self.boundary_imports,
            reducer_calls=self.reducer_calls,
            fb7_butterfly_calls=self.fb7_butterfly_calls,
            spf_fft_calls=self.spf_fft_calls,
            agc_applied=self.agc_applied,
            pilot_cartesian_decode_calls=self.pilot_cartesian_decode_calls,
        )


@functools.lru_cache(maxsize=1)
def _emit_legacy_config_deprecation() -> None:
    warnings.warn(
        "OFDMReceiverConfig is deprecated; use spf.dsp.OFDMRXPlan instead. "
        "Use spf.dsp.from_receiver_config(cfg) for an automatic conversion.",
        DeprecationWarning,
        stacklevel=3,
    )


def from_receiver_config(cfg: OFDMReceiverConfig) -> OFDMRXPlan:
    """Convert a legacy :class:`OFDMReceiverConfig` into an :class:`OFDMRXPlan`.

    Emits :class:`DeprecationWarning` exactly once per process so callers can
    migrate incrementally without spamming logs. Internal receiver code paths
    that already operate in ``OFDMRXPlan`` skip this function entirely.
    """
    _emit_legacy_config_deprecation()
    return OFDMRXPlan(
        n_fft=cfg.n_fft,
        ncp=cfg.ncp,
        modulation=cfg.modulation,
        active_subcarriers=np.ascontiguousarray(np.asarray(cfg.active_subcarriers), dtype=np.uint32),
        channel_policy=ChannelPolicy.KNOWN,
        channel_h=None
        if cfg.channel_h is None
        else np.ascontiguousarray(np.asarray(cfg.channel_h), dtype=np.complex128),
        agc=cfg.agc,
        qam16_threshold_l=cfg.qam16_threshold_l,
    )


def to_receiver_config(plan: OFDMRXPlan) -> OFDMReceiverConfig:
    """Down-convert an :class:`OFDMRXPlan` into a legacy :class:`OFDMReceiverConfig`.

    Useful for code that still consumes the legacy type internally during the
    migration window. Channel policies other than ``KNOWN`` are rejected
    because the legacy type can't express them.
    """
    if plan.channel_policy is not ChannelPolicy.KNOWN:
        raise ValueError(f"to_receiver_config requires channel_policy=KNOWN, got {plan.channel_policy.name}")
    from spf.dsp.ofdm_receiver import OFDMReceiverConfig as _OFDMReceiverConfig

    return _OFDMReceiverConfig(
        n_fft=plan.n_fft,
        ncp=plan.ncp,
        modulation=plan.modulation,
        active_subcarriers=plan.active_subcarriers,
        channel_h=plan.channel_h,
        qam16_threshold_l=plan.qam16_threshold_l,
        agc=plan.agc,
    )


__all__ = [
    "ADCPolicy",
    "CPEPolicy",
    "ChannelPolicy",
    "EqualizerKind",
    "OFDMRXCounters",
    "OFDMRXPlan",
    "TimingPolicy",
    "_MutableCounters",
    "from_receiver_config",
    "to_receiver_config",
]
