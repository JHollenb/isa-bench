"""ctypes bridge for SPF Level-4 OperatorBlock kernels."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spf.native._libspf32 import LIBSPF_AVAILABLE, LIBSPF_PATH, load_library

from .core import (
    ActiveEncoding,
    BandRange,
    ClipPolicy,
    OperatorHeader,
    OutputFormat,
    Profile,
    QAM16LogProjMetadata,
    ReducerPolicy,
    ScalarFormat,
    ScalePolicy,
    SPFError,
    SynthMode,
    SynthPlan,
    _active_indices_for_symbols,
    _reshape_symbols,
    pack_words,
    unpack_L,
    unpack_theta,
)
from .v2 import ProfileTerm, TensorBlock, TensorBlockHeaderV2


def to_c_byte_mass_header(header: object, context_count: int) -> object:
    """Build the ctypes mirror for a registered byte-mass block header.

    The import is lazy so pure-Python tensor users do not require a native
    ``libspftensor`` build.  The returned object is a layout mirror only; the
    host integer consumer remains the authoritative byte generator.
    """

    from spf.native._libspftensor import SPFT_BASE_BYTE_MASS_PLANE, CByteMassHeader

    q = int(header.q)  # type: ignore[attr-defined]
    mass_bits = q + 1
    mass_bytes = int(header.mass_dtype.itemsize)  # type: ignore[attr-defined]
    rom = str(header.rom_id).encode("utf-8")  # type: ignore[attr-defined]
    if not rom or len(rom) >= 64:
        raise SPFError("byte mass rom_id must fit the native 63-byte identity field")
    c = CByteMassHeader()
    c.profile_id = SPFT_BASE_BYTE_MASS_PLANE
    c.q = q
    c.bit_order = 0 if header.bit_order == "msb0" else 1  # type: ignore[attr-defined]
    c.stop_law = 0 if header.stop_law == "max-length" else 1  # type: ignore[attr-defined]
    c.mass_bits = mass_bits
    c.mass_bytes = mass_bytes
    c.reserved0 = 0
    c.node_count = int(header.node_count)  # type: ignore[attr-defined]
    c.context_count = int(context_count)
    c.context_limit = int(header.context_limit)  # type: ignore[attr-defined]
    c.max_output_length = int(header.max_output_length)  # type: ignore[attr-defined]
    c.score_min = int(header.score_min)  # type: ignore[attr-defined]
    c.score_max = int(header.score_max)  # type: ignore[attr-defined]
    c.full_scale = int(header.full_scale)  # type: ignore[attr-defined]
    c.missing_sentinel = int(header.missing_sentinel)  # type: ignore[attr-defined]
    c.invalid_sentinel = int(header.invalid_sentinel)  # type: ignore[attr-defined]
    c.stop_min_mass = int(header.stop_min_mass)  # type: ignore[attr-defined]
    c.context_ids_offset = 0
    c.valid_offset = 0
    c.missing_offset = 0
    c.masses_offset = 0
    c.stop_masses_offset = 0
    c.payload_bytes = 0
    c.rom_id = rom
    return c


def validate_byte_mass_plane_native(plane: object) -> int:
    """Run the native layout/invariant guard for one registered mass view.

    A native library is required explicitly.  Return ``0`` on success or the
    native negative error code; no native byte generation is implied.
    """

    from spf.native._libspftensor import LIBSPFTENSOR_AVAILABLE, load_library

    if not LIBSPFTENSOR_AVAILABLE:
        raise RuntimeError("libspftensor is not available for the byte-mass native guard")
    header = plane.header  # type: ignore[attr-defined]
    context_count = int(plane.context_count)  # type: ignore[attr-defined]
    c_header = to_c_byte_mass_header(header, context_count)
    context_ids = np.ascontiguousarray(plane.context_ids, dtype=np.uint32)  # type: ignore[attr-defined]
    masses = np.ascontiguousarray(plane.masses)  # type: ignore[attr-defined]
    valid = np.ascontiguousarray(plane.valid, dtype=np.uint8)  # type: ignore[attr-defined]
    missing = np.ascontiguousarray(plane.missing, dtype=np.uint8)  # type: ignore[attr-defined]
    stop = getattr(plane, "stop_masses", None)
    stop_arr = None if stop is None else np.ascontiguousarray(stop)
    lib = load_library()
    rc = lib.spft_byte_mass_header_validate(ctypes.byref(c_header), 0)
    if rc != 0:
        return int(rc)
    return int(
        lib.spft_byte_mass_plane_validate(
            ctypes.byref(c_header),
            context_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_void_p(masses.ctypes.data),
            valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            missing.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            None if stop_arr is None else ctypes.c_void_p(stop_arr.ctypes.data),
        )
    )


class COperatorHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version_major", ctypes.c_uint8),
        ("version_minor", ctypes.c_uint8),
        ("scalar_format_id", ctypes.c_uint8),
        ("profile_id", ctypes.c_uint8),
        ("reducer_id", ctypes.c_uint8),
        ("ndims", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("phase_bits", ctypes.c_uint8),
        ("log_frac_bits", ctypes.c_uint16),
        ("header_bytes", ctypes.c_uint16),
        ("payload_bytes", ctypes.c_uint32),
        ("payload_offset", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 4),
        ("strides", ctypes.c_int32 * 4),
        ("origin", ctypes.c_uint32 * 4),
        ("global_L_bias", ctypes.c_int16),
        ("reserved_0", ctypes.c_uint16),
        ("global_T_bias", ctypes.c_uint32),
        ("ramp_alpha", ctypes.c_int32 * 4),
    ]


class CBandRange(ctypes.Structure):
    _fields_ = [
        ("dim", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint16),
        ("start", ctypes.c_uint32),
        ("length", ctypes.c_uint32),
        ("L_delta", ctypes.c_int16),
        ("T_delta", ctypes.c_int16),
    ]


class CSynthPlan(ctypes.Structure):
    _fields_ = [
        ("nfft", ctypes.c_uint32),
        ("ncp", ctypes.c_uint32),
        ("n_symbols", ctypes.c_uint32),
        ("sample_rate_hz", ctypes.c_uint32),
        ("phase_acc_bits", ctypes.c_uint32),
        ("subcarrier_spacing_hz_q", ctypes.c_uint32),
        ("center_frequency_hz_q", ctypes.c_uint32),
        ("sin_lut_bits", ctypes.c_uint16),
        ("amp_lut_bits", ctypes.c_uint16),
        ("output_format", ctypes.c_uint16),
        ("scale_policy", ctypes.c_uint16),
        ("clip_policy", ctypes.c_uint16),
        ("synth_mode", ctypes.c_uint16),
        ("active_mask_offset", ctypes.c_uint32),
        ("tile_map_offset", ctypes.c_uint32),
        ("window_profile_id", ctypes.c_uint32),
        ("sparse_threshold", ctypes.c_uint32),
    ]


class CSynthStats(ctypes.Structure):
    _fields_ = [
        ("synth_mode", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16),
        ("active_tones", ctypes.c_uint32),
        ("ifft_calls", ctypes.c_uint32),
        ("nco_tones", ctypes.c_uint32),
        ("template_hits", ctypes.c_uint32),
        ("estimated_adds", ctypes.c_uint32),
        ("estimated_lut_reads", ctypes.c_uint32),
        ("boundary_crossings", ctypes.c_uint32),
        ("clip_count", ctypes.c_uint32),
        ("papr", ctypes.c_float),
        ("evm_vs_reference", ctypes.c_float),
    ]


class CQAM16LogProjMetadata(ctypes.Structure):
    _fields_ = [
        ("qam_projection_phase_bits", ctypes.c_uint8),
        ("confidence_output", ctypes.c_uint8),
        ("reserved", ctypes.c_uint16),
        ("qam_projection_threshold_L", ctypes.c_int16),
        ("agc_L_offset", ctypes.c_int16),
        ("cpe_theta", ctypes.c_uint16),
        ("erasure_policy", ctypes.c_uint16),
    ]


class CProfileTermV2(ctypes.Structure):
    _fields_ = [
        ("profile_id", ctypes.c_uint16),
        ("flags", ctypes.c_uint16),
        ("rank", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16),
        ("coefficient_q16", ctypes.c_int32),
    ]


class CTensorBlockHeaderV2(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("header_bytes", ctypes.c_uint16),
        ("version", ctypes.c_uint16),
        ("order", ctypes.c_uint8),
        ("scalar_format_id", ctypes.c_uint8),
        ("active_encoding", ctypes.c_uint8),
        ("precision_mode", ctypes.c_uint8),
        ("global_L_bias", ctypes.c_int16),
        ("coarse_exp", ctypes.c_int16),
        ("q_dL", ctypes.c_uint8),
        ("q_dTheta", ctypes.c_uint8),
        ("profile_term_count", ctypes.c_uint16),
        ("payload_bytes", ctypes.c_uint32),
        ("payload_offset", ctypes.c_uint32),
        ("outlier_count", ctypes.c_uint32),
        ("global_T_bias", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 4),
        ("origin", ctypes.c_uint32 * 4),
        ("band_exp", ctypes.c_int16 * 4),
        ("ramp_alpha_v1", ctypes.c_int32 * 4),
        ("ramp_beta_v1", ctypes.c_int32 * 4),
        ("ramp_R", ctypes.c_uint16 * 16),
        ("ramp_b", ctypes.c_uint16 * 4),
        ("quadratic_ramp", ctypes.c_uint16 * 4),
        ("cubic_ramp", ctypes.c_uint16 * 4),
        ("active_offset", ctypes.c_uint32),
        ("dL_offset", ctypes.c_uint32),
        ("dT_offset", ctypes.c_uint32),
        ("outlier_offset", ctypes.c_uint32),
        ("packed_residual_offset", ctypes.c_uint32),
        ("phase_gradient", ctypes.c_uint16 * 4),  # ABI v22: carved from reserved
        ("mixed_ramp", ctypes.c_uint16 * 6),  # ABI v22: pairs per mixed_ramp_pairs(order)
        ("log_quadratic", ctypes.c_int16 * 4),  # ABI v23: signed L ticks per q_d^2
        ("log_mixed", ctypes.c_int16 * 6),  # ABI v23: signed L ticks, mixed_ramp_pairs order
        # ABI v24: AlgRef v0.2 §13.6 mixed cubics, order 2 only.
        # [0] -> q0^2 q1, [1] -> q0 q1^2.
        ("cubic_mixed", ctypes.c_uint16 * 2),
        ("reserved", ctypes.c_uint32 * 7),
    ]


def _as_u32_array(values: npt.ArrayLike) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.uint32)


def _as_i16_array(values: npt.ArrayLike) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int16)


def _as_u16_array(values: npt.ArrayLike) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.uint16)


def _ptr(arr: np.ndarray, ctype: type[ctypes._SimpleCData]) -> ctypes.POINTER[ctypes._SimpleCData]:
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


def _validate_active_indices(idx: np.ndarray, total: int) -> None:
    if np.any((idx < 0) | (idx >= total)):
        raise SPFError("active index out of bounds")
    if idx.size and not np.all(idx[1:] > idx[:-1]):
        raise SPFError("active indices must be strictly increasing")


def to_c_header(header: OperatorHeader) -> COperatorHeader:
    header.validate()
    c = COperatorHeader()
    c.magic = header.magic
    c.version_major = header.version_major
    c.version_minor = header.version_minor
    c.scalar_format_id = int(ScalarFormat(header.scalar_format_id))
    c.profile_id = int(Profile(header.profile_id))
    c.reducer_id = int(ReducerPolicy(header.reducer_id))
    c.ndims = header.ndims
    c.flags = header.flags
    c.phase_bits = header.phase_bits
    c.log_frac_bits = header.log_frac_bits
    c.header_bytes = header.header_bytes
    c.payload_bytes = header.payload_bytes
    c.payload_offset = header.payload_offset
    for i, value in enumerate(header.padded_dims):
        c.dims[i] = value
    for i, value in enumerate(header.strides or (0, 0, 0, 0)):
        c.strides[i] = value
    for i, value in enumerate(header.origin or (0, 0, 0, 0)):
        c.origin[i] = value
    c.global_L_bias = header.global_L_bias
    c.reserved_0 = 0
    c.global_T_bias = header.global_T_bias
    for i, value in enumerate(header.ramp_alpha or (0, 0, 0, 0)):
        c.ramp_alpha[i] = value
    return c


def to_c_bands(bands: Sequence[BandRange]) -> ctypes.Array[CBandRange]:
    arr_t = CBandRange * len(bands)
    arr = arr_t()
    for i, band in enumerate(bands):
        arr[i].dim = int(band.dim)
        arr[i].flags = int(band.effective_flags)
        arr[i].reserved = 0
        arr[i].start = int(band.start)
        arr[i].length = int(band.length)
        arr[i].L_delta = int(band.L_delta)
        arr[i].T_delta = int(band.T_delta)
    return arr


def to_c_synth_plan(plan: SynthPlan) -> CSynthPlan:
    plan.validate()
    c = CSynthPlan()
    c.nfft = plan.nfft
    c.ncp = plan.ncp
    c.n_symbols = plan.n_symbols
    c.sample_rate_hz = plan.sample_rate
    c.phase_acc_bits = plan.phase_acc_bits
    c.subcarrier_spacing_hz_q = plan.subcarrier_spacing_hz_q
    c.center_frequency_hz_q = plan.center_frequency_hz_q
    c.sin_lut_bits = plan.sin_lut_bits
    c.amp_lut_bits = plan.amp_lut_bits
    c.output_format = int(OutputFormat(plan.output_format))
    c.scale_policy = int(ScalePolicy(plan.scale_policy))
    c.clip_policy = int(ClipPolicy(plan.clip_policy))
    c.synth_mode = int(SynthMode(plan.synth_mode))
    c.active_mask_offset = plan.active_mask_offset
    c.tile_map_offset = plan.tile_map_offset
    c.window_profile_id = plan.window_profile_id
    c.sparse_threshold = plan.sparse_threshold
    return c


def to_c_qam16_logproj_metadata(meta: QAM16LogProjMetadata) -> CQAM16LogProjMetadata:
    meta.validate()
    c = CQAM16LogProjMetadata()
    c.qam_projection_phase_bits = int(meta.qam_projection_phase_bits)
    c.confidence_output = 1 if meta.confidence_output else 0
    c.reserved = 0
    c.qam_projection_threshold_L = int(meta.qam_projection_threshold_L)
    c.agc_L_offset = int(meta.agc_L_offset)
    c.cpe_theta = int(meta.cpe_theta)
    c.erasure_policy = 1 if meta.erasure_policy == "mask_zeroes_bits_and_confidence" else 0
    return c


def from_c_qam16_logproj_metadata(c: CQAM16LogProjMetadata) -> QAM16LogProjMetadata:
    policy = "mask_zeroes_bits_and_confidence" if int(c.erasure_policy) == 1 else "none"
    return QAM16LogProjMetadata(
        qam_projection_phase_bits=int(c.qam_projection_phase_bits),
        qam_projection_threshold_L=int(c.qam_projection_threshold_L),
        agc_L_offset=int(c.agc_L_offset),
        cpe_theta=int(c.cpe_theta),
        erasure_policy=policy,
        confidence_output=bool(c.confidence_output),
    )


def to_c_tensorblock_header_v2(header: TensorBlockHeaderV2) -> CTensorBlockHeaderV2:
    header.validate()
    c = CTensorBlockHeaderV2()
    c.magic = 0x53505442
    c.header_bytes = ctypes.sizeof(CTensorBlockHeaderV2)
    c.version = int(header.version)
    c.order = int(header.order)
    c.scalar_format_id = int(ScalarFormat(header.scalar_format_id))
    c.active_encoding = int(ActiveEncoding(header.active_encoding))
    c.precision_mode = int(header.precision_mode)
    c.global_L_bias = int(header.global_L_bias)
    c.coarse_exp = int(header.coarse_exp)
    c.q_dL = int(header.q_dL)
    c.q_dTheta = int(header.q_dTheta)
    c.profile_term_count = len(header.profile_terms or ())
    c.payload_bytes = int(header.payload_bytes)
    c.payload_offset = int(header.payload_offsets.get("payload", 0))
    c.outlier_count = int(header.outlier_count)
    c.global_T_bias = int(header.global_T_bias) & 0xFFFFFFFF
    for i in range(4):
        c.dims[i] = int(header.dims[i]) if i < header.order else 0
        c.origin[i] = int(header.origin[i]) if i < header.order else 0
        c.band_exp[i] = int(header.band_exp[i]) if i < header.order else 0
        c.ramp_alpha_v1[i] = int(header.ramp_alpha_v1[i]) if i < header.order else 0
        c.ramp_beta_v1[i] = int(header.ramp_beta_v1[i]) if i < header.order else 0
        c.ramp_b[i] = int(header.ramp_b[i]) if i < header.order else 0
        c.quadratic_ramp[i] = int(header.quadratic_ramp[i]) if i < header.order else 0
        c.cubic_ramp[i] = int(header.cubic_ramp[i]) if i < header.order else 0
        c.phase_gradient[i] = int(header.phase_gradient[i]) if i < header.order else 0
        c.log_quadratic[i] = int(header.log_quadratic[i]) if i < header.order else 0
    for k in range(6):
        c.mixed_ramp[k] = int(header.mixed_ramp[k]) if k < len(header.mixed_ramp) else 0
        c.log_mixed[k] = int(header.log_mixed[k]) if k < len(header.log_mixed) else 0
    cubic_mixed = tuple(getattr(header, "cubic_mixed", ()) or ())
    for k in range(2):
        c.cubic_mixed[k] = int(cubic_mixed[k]) if k < len(cubic_mixed) else 0
    for r in range(4):
        for col in range(4):
            c.ramp_R[r * 4 + col] = int(header.ramp_R[r][col]) if r < header.order and col < header.order else 0
    c.active_offset = int(header.payload_offsets.get("active_indices", 0))
    c.dL_offset = int(header.payload_offsets.get("dL", 0))
    c.dT_offset = int(header.payload_offsets.get("dT", 0))
    c.outlier_offset = int(header.payload_offsets.get("outliers", 0))
    c.packed_residual_offset = int(header.payload_offsets.get("packed_residuals", 0))
    for i, value in enumerate(header.reserved or ()):
        if i >= 7:
            break
        c.reserved[i] = int(value)
    return c


def to_c_profile_term_v2(term: ProfileTerm) -> CProfileTermV2:
    c = CProfileTermV2()
    c.profile_id = int(term.profile_id)
    c.flags = int(term.flags)
    c.rank = int(term.rank)
    c.reserved = 0
    c.coefficient_q16 = int(term.coefficient_q16)
    return c


def from_c_tensorblock_header_v2(c: CTensorBlockHeaderV2) -> TensorBlockHeaderV2:
    order = int(c.order)
    ramp_R = tuple(tuple(int(c.ramp_R[r * 4 + col]) for col in range(order)) for r in range(order))
    return TensorBlockHeaderV2(
        dims=tuple(int(c.dims[i]) for i in range(order)),
        version=int(c.version),
        order=order,
        scalar_format_id=ScalarFormat(int(c.scalar_format_id)),
        global_L_bias=int(c.global_L_bias),
        global_T_bias=int(c.global_T_bias),
        coarse_exp=int(c.coarse_exp),
        band_exp=tuple(int(c.band_exp[i]) for i in range(order)),
        origin=tuple(int(c.origin[i]) for i in range(order)),
        ramp_alpha_v1=tuple(int(c.ramp_alpha_v1[i]) for i in range(order)),
        ramp_beta_v1=tuple(int(c.ramp_beta_v1[i]) for i in range(order)),
        ramp_R=ramp_R,
        ramp_b=tuple(int(c.ramp_b[i]) for i in range(order)),
        quadratic_ramp=tuple(int(c.quadratic_ramp[i]) for i in range(order)),
        cubic_ramp=tuple(int(c.cubic_ramp[i]) for i in range(order)),
        phase_gradient=tuple(int(c.phase_gradient[i]) for i in range(order)),
        mixed_ramp=tuple(int(c.mixed_ramp[k]) for k in range(order * (order - 1) // 2)),
        log_quadratic=tuple(int(c.log_quadratic[i]) for i in range(order)),
        log_mixed=tuple(int(c.log_mixed[k]) for k in range(order * (order - 1) // 2)),
        cubic_mixed=tuple(int(c.cubic_mixed[k]) for k in range(2)),
        q_dL=int(c.q_dL),
        q_dTheta=int(c.q_dTheta),
        precision_mode=int(c.precision_mode),
        active_encoding=ActiveEncoding(int(c.active_encoding)),
        payload_offsets={
            "payload": int(c.payload_offset),
            "active_indices": int(c.active_offset),
            "dL": int(c.dL_offset),
            "dT": int(c.dT_offset),
            "outliers": int(c.outlier_offset),
            "packed_residuals": int(c.packed_residual_offset),
        },
        payload_bytes=int(c.payload_bytes),
        outlier_count=int(c.outlier_count),
        reserved=tuple(int(c.reserved[i]) for i in range(7)),
    )


@dataclass(slots=True)
class NativeSynthResult:
    samples: np.ndarray
    stats: CSynthStats


class NativeOperatorBlock:
    """Thin native wrapper; pure-Python paths remain authoritative references."""

    def __init__(self) -> None:
        if not LIBSPF_AVAILABLE:
            raise RuntimeError("libspf32 is not available. Run `make -C native` or set SPF_LIBRARY.")
        self.lib = load_library()

    def circular_diff_u16(self, a: int, b: int) -> int:
        return int(self.lib.spf32_op_circular_diff_u16(ctypes.c_uint16(a), ctypes.c_uint16(b)))

    def validate_header(self, header: OperatorHeader, *, containing_bytes: int = 0) -> int:
        c_header = to_c_header(header)
        return int(self.lib.spf32_op_header_validate(ctypes.byref(c_header), containing_bytes))

    def validate_tensorblock_v2(
        self,
        header: TensorBlockHeaderV2,
        *,
        containing_bytes: int = 0,
    ) -> int:
        c_header = to_c_tensorblock_header_v2(header)
        return int(self.lib.spf32_tb_v2_header_validate(ctypes.byref(c_header), containing_bytes))

    def validate_profile_term_v2(self, term: ProfileTerm) -> int:
        c_term = to_c_profile_term_v2(term)
        return int(self.lib.spf32_tb_v2_profile_term_validate(ctypes.byref(c_term)))

    def migrate_header_v2(self, header: OperatorHeader) -> TensorBlockHeaderV2:
        c_v1 = to_c_header(header)
        c_v2 = CTensorBlockHeaderV2()
        rc = self.lib.spf32_tb_v2_from_operator_header(ctypes.byref(c_v1), ctypes.byref(c_v2))
        if rc != 0:
            raise SPFError(f"native v1-to-v2 migration failed rc={rc}")
        return from_c_tensorblock_header_v2(c_v2)

    def tensorblock_v2_affine_reconstruct(self, block: TensorBlock) -> tuple[np.ndarray, np.ndarray]:
        block.validate()
        c_header = to_c_tensorblock_header_v2(block.header)
        total = block.header.total_elements
        outL = np.empty(total, dtype=np.int16)
        outT = np.empty(total, dtype=np.uint16)
        dL_arr = _as_i16_array(block.dL)
        dT_arr = _as_i16_array(block.dT)
        if block.header.active_encoding == ActiveEncoding.DENSE_ALL:
            active_count = total
            idx_ptr = None
        else:
            if block.active_indices is None:
                raise SPFError("COO_U32 requires active_indices")
            idx_i64 = np.ascontiguousarray(block.active_indices, dtype=np.int64)
            _validate_active_indices(idx_i64, total)
            idx = _as_u32_array(idx_i64)
            active_count = int(idx.size)
            idx_ptr = _ptr(idx, ctypes.c_uint32)
        if dL_arr.size != active_count or dT_arr.size != active_count:
            # Mirrors affine_reconstruct; guards the raw C call independently of
            # block.validate() so a direct caller can't trigger a native over-read.
            raise SPFError("residual stream count mismatch")
        c_bands = to_c_bands(block.bands)
        band_ptr = ctypes.cast(c_bands, ctypes.POINTER(CBandRange)) if block.bands else None
        rc = self.lib.spf32_tb_v2_affine_reconstruct(
            ctypes.byref(c_header),
            idx_ptr,
            active_count,
            band_ptr,
            len(block.bands),
            _ptr(dL_arr, ctypes.c_int16),
            _ptr(dT_arr, ctypes.c_int16),
            _ptr(outL, ctypes.c_int16),
            _ptr(outT, ctypes.c_uint16),
            total,
        )
        if rc != 0:
            raise SPFError(f"native v2 affine reconstruct failed rc={rc}")
        if block.outliers:
            words = pack_words(outL, outT).reshape(-1)
            for linear, word in block.outliers.items():
                words[int(linear)] = np.uint32(int(word) & 0xFFFFFFFF)
            outL = unpack_L(words)
            outT = unpack_theta(words)
        return outL.reshape(block.header.dims), outT.reshape(block.header.dims)

    def affine_reconstruct(
        self,
        header: OperatorHeader,
        *,
        active_encoding: ActiveEncoding,
        dL: npt.ArrayLike,
        dT: npt.ArrayLike,
        active_indices: npt.ArrayLike | None = None,
        bands: Sequence[BandRange] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        c_header = to_c_header(header)
        total = header.total_elements
        outL = np.empty(total, dtype=np.int16)
        outT = np.empty(total, dtype=np.uint16)
        dL_arr = _as_i16_array(dL)
        dT_arr = _as_i16_array(dT)
        if active_encoding == ActiveEncoding.DENSE_ALL:
            active_count = total
            idx_ptr = None
        else:
            if active_indices is None:
                raise SPFError("COO_U32 requires active_indices")
            idx_i64 = np.ascontiguousarray(active_indices, dtype=np.int64)
            _validate_active_indices(idx_i64, total)
            idx = _as_u32_array(idx_i64)
            active_count = int(idx.size)
            idx_ptr = _ptr(idx, ctypes.c_uint32)
        if dL_arr.size != active_count or dT_arr.size != active_count:
            raise SPFError("residual stream count mismatch")
        c_bands = to_c_bands(bands)
        band_ptr = ctypes.cast(c_bands, ctypes.POINTER(CBandRange)) if bands else None
        rc = self.lib.spf32_op_affine_reconstruct(
            ctypes.byref(c_header),
            int(active_encoding),
            idx_ptr,
            active_count,
            band_ptr,
            len(bands),
            _ptr(dL_arr, ctypes.c_int16),
            _ptr(dT_arr, ctypes.c_int16),
            _ptr(outL, ctypes.c_int16),
            _ptr(outT, ctypes.c_uint16),
            total,
        )
        if rc != 0:
            raise SPFError(f"native affine reconstruct failed rc={rc}")
        return outL.reshape(header.dims), outT.reshape(header.dims)

    def elementwise_mul_words(self, a: npt.ArrayLike, x: npt.ArrayLike) -> np.ndarray:
        aa = _as_u32_array(a)
        xx = _as_u32_array(x)
        if aa.shape != xx.shape:
            raise SPFError("native elementwise multiply shape mismatch")
        out = np.empty_like(aa)
        self.lib.spf32_op_elementwise_mul_words(
            _ptr(aa, ctypes.c_uint32),
            _ptr(xx, ctypes.c_uint32),
            _ptr(out, ctypes.c_uint32),
            aa.size,
        )
        return out

    def hdc_bind(self, a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
        aa = _as_u16_array(a)
        bb = _as_u16_array(b)
        if aa.shape != bb.shape:
            raise SPFError("native HDC bind shape mismatch")
        out = np.empty_like(aa)
        self.lib.spf32_op_hdc_bind_phase_u16(
            _ptr(aa, ctypes.c_uint16),
            _ptr(bb, ctypes.c_uint16),
            _ptr(out, ctypes.c_uint16),
            aa.size,
        )
        return out

    def synth_auto_iq16(
        self,
        bins: npt.ArrayLike,
        plan: SynthPlan,
        active_indices: npt.ArrayLike,
    ) -> NativeSynthResult:
        words_by_symbol = _reshape_symbols(_as_u32_array(bins), plan)
        active_i64 = _active_indices_for_symbols(words_by_symbol, active_indices)
        active = _as_u32_array(active_i64)
        words = np.ascontiguousarray(words_by_symbol.reshape(-1), dtype=np.uint32)
        c_plan = to_c_synth_plan(plan)
        n_out = plan.n_symbols * (plan.nfft + plan.ncp) * 2
        out = np.empty(n_out, dtype=np.int16)
        stats = CSynthStats()
        rc = self.lib.spf32_synth_auto_iq16(
            _ptr(words, ctypes.c_uint32),
            _ptr(active, ctypes.c_uint32),
            int(active.size),
            ctypes.byref(c_plan),
            _ptr(out, ctypes.c_int16),
            int(out.size),
            ctypes.byref(stats),
        )
        if rc != 0:
            raise SPFError(f"native synth failed rc={rc}")
        return NativeSynthResult(samples=out, stats=stats)


__all__ = [
    "LIBSPF_AVAILABLE",
    "LIBSPF_PATH",
    "CBandRange",
    "COperatorHeader",
    "CProfileTermV2",
    "CQAM16LogProjMetadata",
    "CSynthPlan",
    "CSynthStats",
    "CTensorBlockHeaderV2",
    "NativeOperatorBlock",
    "NativeSynthResult",
    "from_c_qam16_logproj_metadata",
    "from_c_tensorblock_header_v2",
    "to_c_bands",
    "to_c_byte_mass_header",
    "to_c_header",
    "to_c_profile_term_v2",
    "to_c_qam16_logproj_metadata",
    "to_c_synth_plan",
    "to_c_tensorblock_header_v2",
    "validate_byte_mass_plane_native",
]
