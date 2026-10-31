"""``spf.dsp`` — SPF-32 DSP product APIs.

Stable OFDM / QAM / channel / equalization helpers promoted from
``benchmarks/dsp/_helpers.py`` into the package surface.

Submodules
----------
``spf.dsp.modulation``   — PSK / QAM symbol mapping (Gray-coded).
``spf.dsp.ofdm``         — OFDM modulate / demodulate / FIR filter.
``spf.dsp.ofdm_beamspace`` — Rank-1 beamspace sOFDM helpers.
``spf.dsp.ofdm_compat``  — SISO compatibility OFDM over resource grids.
``spf.dsp.ofdm_grid``    — Level-4 spectral OFDM resource grid.
``spf.dsp.channels``     — AWGN and Rayleigh multipath channel models.
``spf.dsp.equalization`` — ZF and LS-pilot equalization.
``spf.dsp.hdc_fhrr``     — Phase-only HDC/FHRR side-path helpers.
``spf.dsp.metrics``      — BER and EVM measurement.
``spf.dsp.qam_logproj``  — Operator wrapper for QAM16 log-projection demap.
``spf.dsp.ofdm_bench``   — repeated-run timing harness (4 explicit modes: diagnostic/hot-path × cross-link/link-load).
``spf.dsp.ofdm_fb7``     — FB7-fused OFDM RX pipeline (Phase 2 promotion).
``spf.dsp.ofdm_link``    — sOFDM TX/DAC/ADC/RX load harness.
``spf.dsp.ofdm_runtime`` — SPF-resident OFDM product receive paths.
``spf.dsp.ofdm_pilots``  — pilot patterns and pilot-based channel estimation.
``spf.dsp.ofdm_receiver`` — sOFDM receiver with a simulated IQ ADC.
``spf.dsp.ofdm_v02``     — Algorithm Reference v0.2 §12 OFDM validation profile (SPF-native FB3 RX).
``spf.dsp.ofdm_resident`` — SPF-resident spectral OFDM operator path.
``spf.dsp.ofdm_residual`` — CPE and timing-ramp residual correction helpers.
``spf.dsp.ofdm_rx_plan`` — OFDMRXPlan, policy enums, and structural counters.
``spf.dsp.ofdm_synth``   — OFDM resource-grid IQ16 export helpers.
``spf.dsp.radar_metrics`` — radar pulse-compression metrics (PSLR/ISLR/peak-SNR).
``spf.dsp.radar_mf_sim``  — LFM chirp / multi-target echo signal model.
``spf.dsp.radar_mf``      — SPF-hotpath matched filter + fp64 baseline arms.
``spf.dsp.radar_mf_bench`` — repeated-run matched-filter bench (4 modes).
``spf.dsp.radar_mf_bank`` — Doppler matched-filter bank (naive/flat/SPFTensor engines).
``spf.dsp.radar_runtime`` — SPF-resident sparse radar product paths.
``spf.dsp.aoa_runtime``   — SPF-resident AoA/localization helpers.
``spf.dsp.beam_runtime``  — SPF-resident beamforming/MIMO composition.
``spf.dsp.rf_native``     — RF-native polar source simulation adapters.

Quick imports::

    from spf.dsp.modulation import qam_modulate, qam_demodulate
    from spf.dsp.ofdm import modulate, demodulate
    from spf.dsp.equalization import zf_equalize
    from spf.dsp.metrics import ber, evm
    from spf.dsp.ofdm_fb7 import ofdm_rx_fused_poly7
    from spf.dsp.ofdm_runtime import receive_qpsk
"""

from spf.dsp import (
    aoa_runtime,
    beam_runtime,
    channels,
    equalization,
    hdc_fhrr,
    metrics,
    modulation,
    ofdm,
    ofdm_beamspace,
    ofdm_bench,
    ofdm_compat,
    ofdm_fb7,
    ofdm_grid,
    ofdm_link,
    ofdm_pilots,
    ofdm_receiver,
    ofdm_resident,
    ofdm_residual,
    ofdm_runtime,
    ofdm_rx_plan,
    ofdm_synth,
    ofdm_v02,
    qam_logproj,
    radar_metrics,
    radar_mf,
    radar_mf_bank,
    radar_mf_bench,
    radar_mf_sim,
    radar_runtime,
    rf_native,
)
from spf.dsp.aoa_runtime import AoARuntimeResult, estimate_pair
from spf.dsp.beam_runtime import BeamRuntimeResult, beamform_reduce
from spf.dsp.hdc_fhrr import (
    bundle_circular_mean,
    bundle_pairwise_phase_average,
    cosine_similarity_i64,
    fhrr_bind,
    fhrr_conjugate,
    fhrr_unbind,
)
from spf.dsp.ofdm_beamspace import (
    BeamCodebook,
    dft_beam_codebook,
    effective_channel,
    rank1_beamform_grid,
)
from spf.dsp.ofdm_bench import (
    CrossLinkCellStats,
    CrossLinkMatrixStats,
    SOFDMLinkStats,
    StageStats,
    run_cross_matrix_diagnostic,
    run_cross_matrix_hot_path,
    run_link_load_diagnostic,
    run_link_load_hot_path,
)
from spf.dsp.ofdm_compat import SISOCompatOFDMResult, run_siso_compat_ofdm
from spf.dsp.ofdm_fb7 import OfdmFB7Result, ofdm_rx_fused_poly7
from spf.dsp.ofdm_grid import OFDMBinClass, OFDMResourceGrid
from spf.dsp.ofdm_link import (
    BaselineOFDMResult,
    CrossLinkCell,
    CrossLinkIdentity,
    CrossLinkMatrix,
    DACConfig,
    DACResult,
    OFDMTXConfig,
    SOFDMLinkMetrics,
    SOFDMLinkResult,
    SOFDMLoadConfig,
    SOFDMTXFrame,
    apply_known_channel,
    build_sofdm_tx_frame,
    load_sofdm_link_config,
    run_basic_ofdm_baseline,
    run_cross_link_matrix,
    run_sofdm_link_load,
    simulate_iq_dac,
    sofdm_link_config_from_mapping,
    validate_sofdm_link_load,
    write_cross_link_metrics,
    write_sofdm_link_metrics,
)
from spf.dsp.ofdm_pilots import (
    ChannelEstimate,
    PilotPattern,
    block_pilot_pattern,
    comb_pilot_pattern,
    data_subcarriers,
    estimate_channel_from_pilots,
    tx_frame_with_pilots,
)
from spf.dsp.ofdm_receiver import (
    ADCConfig,
    ADCResult,
    OFDMReceiverConfig,
    SOFDMADCRoundTripResult,
    SOFDMReceiverResult,
    receive_sofdm_frame,
    run_sofdm_adc_roundtrip,
    simulate_iq_adc,
)
from spf.dsp.ofdm_resident import (
    ResidentOFDMCounters,
    ResidentRXResult,
    SISOResidentOFDMResult,
    receive_sofdm_resident_bins,
    run_siso_resident_ofdm,
)
from spf.dsp.ofdm_residual import (
    CPEEstimate,
    TimingEstimate,
    apply_cpe_correction,
    apply_timing_correction,
    estimate_cpe_from_pilots,
    estimate_timing_ramp,
)
from spf.dsp.ofdm_runtime import (
    OfdmRuntimeResult,
    buffer_from_words,
    receive_qam_llr,
    receive_qam_lut,
    receive_qpsk,
)
from spf.dsp.ofdm_rx_plan import (
    ADCPolicy,
    ChannelPolicy,
    CPEPolicy,
    EqualizerKind,
    OFDMRXCounters,
    OFDMRXPlan,
    TimingPolicy,
    from_receiver_config,
    to_receiver_config,
)
from spf.dsp.ofdm_synth import (
    export_iq16_ifft,
    export_iq16_ifft_trusted,
    export_iq16_nco,
)
from spf.dsp.ofdm_v02 import (
    OfdmV02Metrics,
    OfdmV02Result,
    RXProfile,
    ofdm_rx_spf_v02,
    ofdm_v02_metrics,
    spf_word_equalize,
    spf_word_fft,
)
from spf.dsp.qam_logproj import QAM16LogProjOperator, QAM16LogProjResult
from spf.dsp.radar_runtime import RadarRuntimeResult, sparse_doppler
from spf.dsp.rf_native import PolarRFBuffer

__all__ = [
    "ADCConfig",
    "ADCPolicy",
    "ADCResult",
    "AoARuntimeResult",
    "BaselineOFDMResult",
    "BeamCodebook",
    "BeamRuntimeResult",
    "CPEEstimate",
    "CPEPolicy",
    "ChannelEstimate",
    "ChannelPolicy",
    "CrossLinkCell",
    "CrossLinkCellStats",
    "CrossLinkIdentity",
    "CrossLinkMatrix",
    "CrossLinkMatrixStats",
    "DACConfig",
    "DACResult",
    "EqualizerKind",
    "OFDMBinClass",
    "OFDMRXCounters",
    "OFDMRXPlan",
    "OFDMReceiverConfig",
    "OFDMResourceGrid",
    "OFDMTXConfig",
    "OfdmFB7Result",
    "OfdmRuntimeResult",
    "OfdmV02Metrics",
    "OfdmV02Result",
    "PilotPattern",
    "PolarRFBuffer",
    "QAM16LogProjOperator",
    "QAM16LogProjResult",
    "RXProfile",
    "RadarRuntimeResult",
    "ResidentOFDMCounters",
    "ResidentRXResult",
    "SISOCompatOFDMResult",
    "SISOResidentOFDMResult",
    "SOFDMADCRoundTripResult",
    "SOFDMLinkMetrics",
    "SOFDMLinkResult",
    "SOFDMLinkStats",
    "SOFDMLoadConfig",
    "SOFDMReceiverResult",
    "SOFDMTXFrame",
    "StageStats",
    "TimingEstimate",
    "TimingPolicy",
    "aoa_runtime",
    "apply_cpe_correction",
    "apply_known_channel",
    "apply_timing_correction",
    "beam_runtime",
    "beamform_reduce",
    "block_pilot_pattern",
    "buffer_from_words",
    "build_sofdm_tx_frame",
    "bundle_circular_mean",
    "bundle_pairwise_phase_average",
    "channels",
    "comb_pilot_pattern",
    "cosine_similarity_i64",
    "data_subcarriers",
    "dft_beam_codebook",
    "effective_channel",
    "equalization",
    "estimate_channel_from_pilots",
    "estimate_cpe_from_pilots",
    "estimate_pair",
    "estimate_timing_ramp",
    "export_iq16_ifft",
    "export_iq16_ifft_trusted",
    "export_iq16_nco",
    "fhrr_bind",
    "fhrr_conjugate",
    "fhrr_unbind",
    "from_receiver_config",
    "hdc_fhrr",
    "load_sofdm_link_config",
    "metrics",
    "modulation",
    "ofdm",
    "ofdm_beamspace",
    "ofdm_bench",
    "ofdm_compat",
    "ofdm_fb7",
    "ofdm_grid",
    "ofdm_link",
    "ofdm_pilots",
    "ofdm_receiver",
    "ofdm_resident",
    "ofdm_residual",
    "ofdm_runtime",
    "ofdm_rx_fused_poly7",
    "ofdm_rx_plan",
    "ofdm_rx_spf_v02",
    "ofdm_synth",
    "ofdm_v02",
    "ofdm_v02_metrics",
    "qam_logproj",
    "radar_metrics",
    "radar_mf",
    "radar_mf_bank",
    "radar_mf_bench",
    "radar_mf_sim",
    "radar_runtime",
    "rank1_beamform_grid",
    "receive_qam_llr",
    "receive_qam_lut",
    "receive_qpsk",
    "receive_sofdm_frame",
    "receive_sofdm_resident_bins",
    "rf_native",
    "run_basic_ofdm_baseline",
    "run_cross_link_matrix",
    "run_cross_matrix_diagnostic",
    "run_cross_matrix_hot_path",
    "run_link_load_diagnostic",
    "run_link_load_hot_path",
    "run_siso_compat_ofdm",
    "run_siso_resident_ofdm",
    "run_sofdm_adc_roundtrip",
    "run_sofdm_link_load",
    "simulate_iq_adc",
    "simulate_iq_dac",
    "sofdm_link_config_from_mapping",
    "sparse_doppler",
    "spf_word_equalize",
    "spf_word_fft",
    "to_receiver_config",
    "tx_frame_with_pilots",
    "validate_sofdm_link_load",
    "write_cross_link_metrics",
    "write_sofdm_link_metrics",
]
