"""SPFTensor v3 — three-axis registered block (additive, experimental).

v3 is **v2 factored into three orthogonal axes** plus one new base-model term.
It does not change the SPF-32 scalar contract, the FB7/poly7 add path, or the
v2 reducer; it relabels every v2 ``TensorBlock`` as a registered combination of:

    Representation × BaseModel × ResidualCodec

* **Representation** — how a scalar is stored: the scalar word width, one of
  ``POLAR_SPF16`` / ``POLAR_SPF32`` (default) / ``POLAR_SPF64`` / ``POLAR_SPF128``.
  ``reconstruct`` / ``to_words`` / ``from_dense`` finalise to that ``Format``;
  SPF-128 words are arbitrary-precision Python ints in ``object`` arrays.
* **BaseModel** — the generative wavefield: the v2 affine/poly ramp
  (``STB_AFFINE_POLY``) plus the new exponent-domain ``SEPARABLE_PHASE`` /
  ``SEPARABLE_LOGMAG`` low-rank terms (see :mod:`spf.tensor.base_models`).
* **ResidualCodec** — how post-base deviations ``(dL, dΘ)`` are stored
  (``RAW_INT16`` today; ``UNIFORM`` / ``CODEBOOK_CIRCULAR`` / ``PALETTE`` in
  :mod:`spf.tensor.residual_codecs`).

The container does no heavy math: reconstruction is one integer pass over
``base.reconstruct_planes`` + ``residual.decode`` finalised by the v2
zero-sentinel / saturate / mask / outlier rules — **bit-identical to
``TensorBlock.reconstruct`` for any relabelled v2 block** (see the T15/T17
parity suites). Everything heavy lives behind the three registries.

Design: ``docs/tensor/spftensor-v3-design.md``. Public/IP name stays
``SPFTensor`` / ``SPF_TB``; the v3 unit is :class:`SPFBlockV3` (the existing
:class:`spf.tensor.v2.SPFBlock` multi-block container is unchanged).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from spf.scalar._constants import SPF16 as _SPF16
from spf.scalar._constants import SPF32 as _SPF32
from spf.scalar._constants import SPF64 as _SPF64
from spf.scalar._constants import SPF128 as _SPF128
from spf.scalar._constants import Format as _ScalarFmt

from .core import (
    SPFError,
    _sat_l,
    decode_complex,
    encode_complex,
    pack_words,
    unpack_L,
    unpack_theta,
)
from .experimental_formats import M18_P14, M19_P13, M20_P12, Q3_12_P16

SPFTENSOR_V3_MAJOR = 3
SPFTENSOR_V3_MINOR = 2
SPFTENSOR_DTYPE_V3 = "SPF_TB_v3"

# The byte-mass output head is a registered v3 structural profile.  Its
# integer planes are owned by ``SPFBlockV3.base.params`` and are serialized by
# the ordinary v3 envelope; there is no second carrier type.
BYTE_MASS_SCHEMA = "spf-byte-mass-v1"
BYTE_MASS_NODE_COUNT = 255
BYTE_MASS_MAX_Q = 63
BYTE_MASS_MAX_CONTEXT_LIMIT = (1 << 32) - 1
BYTE_MASS_MAX_OUTPUT_LENGTH = (1 << 32) - 1
BYTE_MASS_SCORE_MIN = -(1 << 31)
BYTE_MASS_SCORE_MAX = (1 << 31) - 1

# Reserved profile-id discipline (extends v2; see design §5.1 / §13). CP is
# banned at construction; the separable exponent-domain family, the spectral
# family, the TT family, and the non-abelian Fourier range are reserved.
PROFILE_SEPARABLE_MIN = 0x0100
PROFILE_SEPARABLE_MAX = 0x01FF
PROFILE_SPECTRAL_MIN = 0x0200
PROFILE_SPECTRAL_MAX = 0x0FFF
PROFILE_TT_MIN = 0x1000
PROFILE_TT_MAX = 0x3FFF
PROFILE_CP_BANNED = 0x4000
PROFILE_NONABELIAN_MIN = 0x8000
PROFILE_NONABELIAN_MAX = 0xBFFF


class Representation(IntEnum):
    """Axis 1 — how a scalar is stored (design §4).

    The ``POLAR_SPF*`` members select the scalar bit-width of the ``(L, Θ)`` word
    (see :func:`format_for_representation`). ``POLAR_SPF32 = 0`` is the historical
    default and stays at id 0 so existing serialised blocks are unchanged; the
    other widths take fresh ids (the envelope stores the rep id as a u8).
    """

    POLAR_SPF32 = 0  # current (L, Θ) SPF-32 words (16-bit L, 16-bit Θ)
    CARTESIAN_BFP = 1  # reserved/optional shared-exp re/im (T20, reducer-only today)
    STRUCTURAL_VIEW = 2  # reserved: sparse / Hermitian / low-rank / circulant / TT
    POLAR_SPF16 = 3  # (L, Θ) SPF-16 words (8-bit L, 8-bit Θ)
    POLAR_SPF64 = 4  # (L, Θ) SPF-64 words (32-bit L, 32-bit Θ)
    POLAR_SPF128 = 5  # (L, Θ) SPF-128 words (64-bit L, 64-bit Θ; Python-int/object)
    EXPERIMENTAL_M18_P14 = 6
    EXPERIMENTAL_M19_P13 = 7
    EXPERIMENTAL_M20_P12 = 8
    EXPERIMENTAL_Q3_12_P16 = 9


# Map each POLAR_SPF* representation to its scalar Format (bit widths, scales,
# sentinels, dtypes). Non-polar reps (CARTESIAN_BFP / STRUCTURAL_VIEW) have no
# single (L, Θ) word and are absent.
_REPRESENTATION_TO_FMT: dict[int, _ScalarFmt] = {
    int(Representation.POLAR_SPF16): _SPF16,
    int(Representation.POLAR_SPF32): _SPF32,
    int(Representation.POLAR_SPF64): _SPF64,
    int(Representation.POLAR_SPF128): _SPF128,
}


_REPRESENTATION_TO_FMT.update(
    {
        int(Representation.EXPERIMENTAL_M18_P14): M18_P14,
        int(Representation.EXPERIMENTAL_M19_P13): M19_P13,
        int(Representation.EXPERIMENTAL_M20_P12): M20_P12,
        int(Representation.EXPERIMENTAL_Q3_12_P16): Q3_12_P16,
    }
)


def format_for_representation(rep_id: int) -> _ScalarFmt:
    """Resolve a ``POLAR_SPF*`` representation id to its scalar :class:`Format`."""
    try:
        return _REPRESENTATION_TO_FMT[int(rep_id)]
    except KeyError:
        raise SPFError(
            f"representation {int(rep_id)} is not a POLAR_SPF* scalar word "
            "(no (L, Θ) Format); reconstruct() needs a polar representation"
        ) from None


class BaseModelKind(IntEnum):
    """Axis 2 — the generative wavefield law (design §5.1)."""

    STB_AFFINE_POLY = 0x0000  # v2 TensorBlockHeader affine/quad/cubic ramp + bands
    SEPARABLE_PHASE = 0x0100  # exponent-domain rank-r phase outer product (RoPE/chirp)
    SEPARABLE_LOGMAG = 0x0101  # exponent-domain rank-r log-magnitude outer product
    FOURIER_ROW_COL = 0x0102  # reserved: row/col Fourier laws
    PSK_GAUGE_FIT = 0x0103  # reserved: analysis/fitter, not magic storage
    WAVE_PACKET = 0x0104  # reserved: experimental wave law
    LATENT_PHASE_PROJECTION = 0x0105  # experimental: continuous-input phase basis (PAN handoff)
    PHASE_CLOCK_RECORDS = 0x0106  # packed records; phase-carried exposure and rational transport
    BYTE_MASS_PLANE = 0x0107  # typed Q+1 integer conditional-byte mass planes
    CIRCULANT_SPECTRUM = 0x0200  # spectral operator/view: diagonal-in-Fourier (circulant/GFNet)
    JACOBIAN_OPERATOR = 0x0201  # linear-operator view: Jacobian / matrix + metadata
    ITERATED_OPERATOR = 0x0202  # compositional generator: orbit {A^k x0} of a referenced operator
    TT_STRUCTURED = 0x1000  # reserved: TT / H-Tucker family
    NONABELIAN_OPERATOR = 0x8000  # non-abelian generator: element of a small matrix subalgebra/group


def _byte_mass_dtype(q: int) -> np.dtype:
    """Return the canonical unsigned storage dtype for a Q+1 mass field."""

    bits = int(q) + 1
    for dtype, width in (
        (np.uint8, 8),
        (np.uint16, 16),
        (np.uint32, 32),
        (np.uint64, 64),
    ):
        if bits <= width:
            # The native validator loads multi-byte masses explicitly in
            # little-endian order.  Make the serialized NumPy dtype carry the
            # same contract on every host architecture.
            return np.dtype(dtype).newbyteorder("<")
    raise SPFError("byte mass Q+1 width exceeds exact uint64 storage")


def _byte_mass_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SPFError(f"byte mass {name} must be an integer")
    return int(value)


def _byte_mass_bool_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.bool_ or raw.shape != shape:
        raise SPFError(f"byte mass {name} must be a boolean vector of shape {shape}")
    return raw.astype(np.bool_, copy=True)


def _immutable_bytes_array(value: Any, dtype: np.dtype) -> np.ndarray:
    """Own an array through an immutable bytes buffer.

    ``flags.writeable = False`` alone is reversible when an ndarray owns its
    allocation.  Rebuilding from ``bytes`` makes the read-only property a
    storage invariant: callers cannot re-enable writes through the public
    array view.
    """

    arr = np.ascontiguousarray(value, dtype=dtype)
    raw = bytes(arr.tobytes(order="C"))
    return np.frombuffer(raw, dtype=dtype, count=arr.size).reshape(arr.shape)


def validate_byte_mass_params(
    dims: tuple[int, ...] | list[int],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize the registered BYTE_MASS_PLANE header.

    The returned mapping owns immutable byte-backed copies of every typed plane.  This is
    the profile boundary used by :class:`SPFBlockV3` and by the v3 envelope;
    callers must not create a parallel ``ByteMassPlane`` carrier.  The shape is
    ``(contexts, 255, 2)`` with one ``uint{8,16,32,64}`` mass dtype selected by
    ``Q+1`` bits.
    """

    p = dict(params)
    if p.get("schema") != BYTE_MASS_SCHEMA:
        raise SPFError(f"byte mass schema must be {BYTE_MASS_SCHEMA!r}")
    if p.get("abi", BYTE_MASS_SCHEMA) != BYTE_MASS_SCHEMA:
        raise SPFError(f"byte mass abi must be {BYTE_MASS_SCHEMA!r}")
    q = _byte_mass_int(p.get("q"), "Q")
    if q < 0 or q > BYTE_MASS_MAX_Q:
        raise SPFError("byte mass Q must be in [0, 63]")
    full = 1 << q
    dtype = _byte_mass_dtype(q)
    if p.get("mass_bits") != q + 1:
        raise SPFError("byte mass mass_bits must equal Q+1")
    if p.get("mass_dtype") != dtype.name:
        raise SPFError(f"byte mass mass_dtype must be {dtype.name!r}")
    if p.get("full_scale") != full:
        raise SPFError("byte mass full_scale must equal 2**Q")
    if p.get("missing_sentinel") != full + 1 or p.get("invalid_sentinel") != full + 2:
        raise SPFError("byte mass sentinels must be full_scale+1 and full_scale+2")
    if p.get("node_count") != BYTE_MASS_NODE_COUNT:
        raise SPFError("byte mass node_count must be 255")
    if p.get("bit_order") not in ("msb0", "lsb0"):
        raise SPFError("byte mass bit_order must be 'msb0' or 'lsb0'")
    if not isinstance(p.get("rom_id"), str) or not p["rom_id"]:
        raise SPFError("byte mass rom_id must be a non-empty string")
    try:
        rom_bytes = p["rom_id"].encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SPFError("byte mass rom_id must be valid UTF-8") from exc
    if len(rom_bytes) >= 64 or b"\x00" in rom_bytes:
        raise SPFError("byte mass rom_id must fit a NUL-free native 63-byte identity field")
    score_min = _byte_mass_int(p.get("score_min"), "score_min")
    score_max = _byte_mass_int(p.get("score_max"), "score_max")
    if score_min > score_max or score_min < BYTE_MASS_SCORE_MIN or score_max > BYTE_MASS_SCORE_MAX:
        raise SPFError("byte mass score bounds are reversed")
    context_limit = _byte_mass_int(p.get("context_limit"), "context_limit")
    max_output_length = _byte_mass_int(p.get("max_output_length"), "max_output_length")
    if (
        context_limit <= 0
        or context_limit > BYTE_MASS_MAX_CONTEXT_LIMIT
        or max_output_length <= 0
        or max_output_length > BYTE_MASS_MAX_OUTPUT_LENGTH
    ):
        raise SPFError("byte mass context_limit and max_output_length must be positive")
    stop_law = p.get("stop_law")
    if stop_law not in ("max-length", "two-mass"):
        raise SPFError("byte mass stop_law must be 'max-length' or 'two-mass'")
    stop_min_mass = _byte_mass_int(p.get("stop_min_mass"), "stop_min_mass")
    if not 0 <= stop_min_mass <= full:
        raise SPFError("byte mass stop_min_mass is outside [0, full_scale]")
    backend = p.get("consumer_backend", "host-python-integer")
    if not isinstance(backend, str) or not backend:
        raise SPFError("byte mass consumer_backend must be a non-empty string")

    if len(tuple(dims)) != 3:
        raise SPFError("BYTE_MASS_PLANE dims must be (contexts, 255, 2)")
    context_ids = np.asarray(p.get("context_ids"))
    if context_ids.ndim != 1 or not np.issubdtype(context_ids.dtype, np.integer):
        raise SPFError("byte mass context_ids must be a one-dimensional integer array")
    n_context = int(context_ids.size)
    if tuple(int(d) for d in dims) != (n_context, BYTE_MASS_NODE_COUNT, 2):
        raise SPFError("BYTE_MASS_PLANE dims do not match its typed planes")
    if n_context < 1 or n_context > context_limit:
        raise SPFError("byte mass context count exceeds context_limit")
    if np.any(context_ids.astype(object) < 0) or np.any(context_ids.astype(object) >= context_limit):
        raise SPFError("byte mass context ids are out of bounds")
    ids = context_ids.astype(np.uint32, copy=True)
    if np.unique(ids).size != ids.size:
        raise SPFError("byte mass context ids are out of bounds or duplicated")

    masses = np.asarray(p.get("masses"))
    if masses.dtype != dtype or masses.shape != (n_context, BYTE_MASS_NODE_COUNT, 2):
        expected_shape = (n_context, BYTE_MASS_NODE_COUNT, 2)
        raise SPFError(f"byte mass masses must have shape {expected_shape} and dtype {dtype.name}")
    valid = _byte_mass_bool_vector(p.get("valid"), (n_context,), "valid")
    missing = _byte_mass_bool_vector(p.get("missing"), (n_context,), "missing")
    if np.any(valid & missing):
        raise SPFError("byte mass valid and missing masks overlap")
    mass_copy = masses.astype(dtype, copy=True)
    mass_obj = mass_copy.astype(object)
    if np.any(mass_obj > full):
        raise SPFError("byte mass value exceeds full_scale")
    if np.any(valid) and np.any(mass_obj[valid].sum(axis=-1) != full):
        raise SPFError("valid byte mass sibling pairs must sum exactly to full_scale")

    stop_raw = p.get("stop_masses")
    stop_copy: np.ndarray | None
    if stop_raw is None:
        stop_copy = None
        if stop_law == "two-mass":
            raise SPFError("two-mass byte stop law requires stop_masses")
    else:
        stop = np.asarray(stop_raw)
        expected = (n_context, max_output_length, 2)
        if stop.dtype != dtype or stop.shape != expected:
            raise SPFError(f"byte mass stop_masses must have shape {expected} and dtype {dtype.name}")
        stop_copy = stop.astype(dtype, copy=True)
        stop_obj = stop_copy.astype(object)
        if np.any(stop_obj > full):
            raise SPFError("byte stop mass exceeds full_scale")
        if np.any(valid):
            if np.any(stop_obj[valid].sum(axis=-1) != full):
                raise SPFError("valid byte stop sibling pairs must sum exactly to full_scale")
            if stop_min_mass and np.any(stop_obj[valid, ..., 1] < stop_min_mass):
                raise SPFError("valid byte stop mass is below stop_min_mass")

    # Every plane is rebuilt from immutable bytes after it enters the block
    # header.  A write-protected owning ndarray can otherwise be made writable
    # again by a caller, changing a supposedly stable block in place.
    ids = _immutable_bytes_array(ids, np.dtype("<u4"))
    mass_copy = _immutable_bytes_array(mass_copy, dtype)
    valid = _immutable_bytes_array(valid, np.dtype(np.bool_))
    missing = _immutable_bytes_array(missing, np.dtype(np.bool_))
    if stop_copy is not None:
        stop_copy = _immutable_bytes_array(stop_copy, dtype)
    p.update(
        {
            "schema": BYTE_MASS_SCHEMA,
            "q": q,
            "mass_bits": q + 1,
            "mass_dtype": dtype.name,
            "full_scale": full,
            "missing_sentinel": full + 1,
            "invalid_sentinel": full + 2,
            "node_count": BYTE_MASS_NODE_COUNT,
            "score_min": score_min,
            "score_max": score_max,
            "context_limit": context_limit,
            "max_output_length": max_output_length,
            "stop_min_mass": stop_min_mass,
            "consumer_backend": backend,
            "context_ids": ids,
            "masses": mass_copy,
            "valid": valid,
            "missing": missing,
            "stop_masses": stop_copy,
        }
    )
    return p


class ResidualCodecKind(IntEnum):
    """Axis 3 — how post-base deviations are stored (design §7)."""

    NONE = 0
    RAW_INT16 = 1
    UNIFORM = 2
    CODEBOOK_CIRCULAR = 3
    PALETTE = 4
    ENTROPY_PACKED_INDICES = 5  # reserved/future


def validate_profile_id(profile_id: int) -> int:
    """Reject CP at construction; accept known/reserved profile ids (design rule 6)."""
    pid = int(profile_id)
    if pid == PROFILE_CP_BANNED:
        raise SPFError("CP (0x4000) is not a closed contraction profile in SPFTensor v3")
    return pid


# --------------------------------------------------------------------------- #
# Axis refs — the small, serialisable handles the block carries.
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RepresentationRef:
    """How scalars in this block are stored."""

    rep_id: int = int(Representation.POLAR_SPF32)

    def __post_init__(self) -> None:
        self.rep_id = int(Representation(self.rep_id))

    @property
    def format(self) -> _ScalarFmt:
        """The scalar :class:`Format` for this (polar) representation."""
        return format_for_representation(self.rep_id)


@dataclass(slots=True)
class BaseModelRef:
    """A base-model profile id plus its model-specific parameters."""

    profile_id: int
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.profile_id = validate_profile_id(self.profile_id)
        self.params = dict(self.params)


@dataclass(slots=True)
class ResidualRef:
    """A residual-codec id plus its codec-specific payload."""

    codec_id: int = int(ResidualCodecKind.NONE)
    payload: Any = None

    def __post_init__(self) -> None:
        self.codec_id = int(ResidualCodecKind(self.codec_id))


# --------------------------------------------------------------------------- #
# Registry spine (design §3.2).
# --------------------------------------------------------------------------- #


@runtime_checkable
class BaseModel(Protocol):
    """Generates the predictable part of the ``(L, Θ)`` planes for active coords.

    ``reconstruct_planes`` returns ``(active_indices, baseL, baseT)`` over the
    block's active linear indices: ``baseL`` / ``baseT`` are **raw int64**
    pre-residual contributions (no saturation/mask yet) so the container can
    finalise base+residual uniformly, exactly as ``TensorBlock.reconstruct``.
    """

    kind: int

    def reconstruct_planes(
        self,
        params: Mapping[str, Any],
        dims: tuple[int, ...],
        origin: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def storage_bits(self, params: Mapping[str, Any]) -> int: ...


@runtime_checkable
class ResidualCodec(Protocol):
    """Encodes/decodes the post-base ``(dL, dΘ)`` deviation streams."""

    kind: int

    def decode(self, payload: Any, n_active: int) -> tuple[np.ndarray, np.ndarray]: ...

    def storage_bits(self, payload: Any) -> int: ...


@runtime_checkable
class RepresentationOps(Protocol):
    kind: int


BASE_MODEL_REGISTRY: dict[int, BaseModel] = {}
RESIDUAL_CODEC_REGISTRY: dict[int, ResidualCodec] = {}
REPRESENTATION_REGISTRY: dict[int, RepresentationOps] = {}


def register_base_model(model: BaseModel) -> BaseModel:
    BASE_MODEL_REGISTRY[validate_profile_id(model.kind)] = model
    return model


def register_residual_codec(codec: ResidualCodec) -> ResidualCodec:
    RESIDUAL_CODEC_REGISTRY[int(codec.kind)] = codec
    return codec


def register_representation(rep: RepresentationOps) -> RepresentationOps:
    REPRESENTATION_REGISTRY[int(rep.kind)] = rep
    return rep


def _ensure_registries() -> None:
    """Import the default registry contents on first use (avoids import cycles)."""
    if BASE_MODEL_REGISTRY and RESIDUAL_CODEC_REGISTRY and int(BaseModelKind.BYTE_MASS_PLANE) in BASE_MODEL_REGISTRY:
        return
    from . import (  # noqa: F401  (registration side effect)
        base_models,
        byte_output,
        iterated_operator,
        jacobian,
        nonabelian,
        phase_clock,
        residual_codecs,
        spectral_surface,
    )


def get_base_model(profile_id: int) -> BaseModel:
    _ensure_registries()
    try:
        return BASE_MODEL_REGISTRY[int(profile_id)]
    except KeyError:
        raise SPFError(f"no base model registered for profile 0x{int(profile_id):04x}") from None


def get_residual_codec(codec_id: int) -> ResidualCodec:
    _ensure_registries()
    try:
        return RESIDUAL_CODEC_REGISTRY[int(codec_id)]
    except KeyError:
        raise SPFError(f"no residual codec registered for id {int(codec_id)}") from None


# --------------------------------------------------------------------------- #
# The v3 block container (design §3.1).
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SPFBlockV3:
    """SPFTensor v3 block: ``Representation × BaseModel × ResidualCodec`` + outliers.

    The container holds only refs and the exact-override outlier table; all
    arithmetic lives behind the registries. ``reconstruct`` is one integer pass.
    """

    dims: tuple[int, ...]
    origin: tuple[int, ...] = (0,)
    representation: RepresentationRef = field(default_factory=RepresentationRef)
    base: BaseModelRef = field(default_factory=lambda: BaseModelRef(int(BaseModelKind.STB_AFFINE_POLY)))
    residual: ResidualRef = field(default_factory=ResidualRef)
    outliers: Mapping[int, int] = field(default_factory=dict)
    meta: Any = None  # cached BlockMetadata (optional)

    def __post_init__(self) -> None:
        self.dims = tuple(int(d) for d in self.dims)
        # An order-0 (scalar) block keeps dims == () so reconstruct() returns a
        # 0-d array, bit-identical to v2 TensorBlock.reconstruct (which reshapes
        # to header.dims == ()). Coercing () → (1,) would diverge in shape.
        order = len(self.dims)
        self.origin = tuple(int(o) for o in self.origin)[:order] or (0,) * order
        if len(self.origin) < order:
            self.origin = self.origin + (0,) * (order - len(self.origin))
        if not isinstance(self.representation, RepresentationRef):
            self.representation = RepresentationRef(int(self.representation))
        if not isinstance(self.base, BaseModelRef):
            raise SPFError("SPFBlockV3.base must be a BaseModelRef")
        if not isinstance(self.residual, ResidualRef):
            raise SPFError("SPFBlockV3.residual must be a ResidualRef")
        if self.base.profile_id == int(BaseModelKind.BYTE_MASS_PLANE):
            if self.representation.rep_id != int(Representation.STRUCTURAL_VIEW):
                raise SPFError("BYTE_MASS_PLANE requires the STRUCTURAL_VIEW representation")
            if self.residual.codec_id != int(ResidualCodecKind.NONE) or self.residual.payload is not None:
                raise SPFError("BYTE_MASS_PLANE cannot carry a scalar residual codec")
            if self.outliers:
                raise SPFError("BYTE_MASS_PLANE cannot carry scalar outlier overrides")
            self.base = BaseModelRef(
                self.base.profile_id,
                validate_byte_mass_params(self.dims, self.base.params),
            )
            self.base.params = MappingProxyType(dict(self.base.params))
        # Outlier override words are masked to this representation's word width
        # (SPF-128 words are wide Python ints). Non-polar reps fall back to the
        # historical 32-bit mask.
        try:
            word_mask = (1 << self.representation.format.total_bits) - 1
        except SPFError:
            word_mask = 0xFFFFFFFF
        self.outliers = {int(k): int(v) & word_mask for k, v in dict(self.outliers).items()}
        total = self.total_elements
        for linear in self.outliers:
            if linear < 0 or linear >= total:
                raise SPFError("outlier index out of bounds")

    @property
    def total_elements(self) -> int:
        return int(np.prod(self.dims, dtype=np.int64)) if self.dims else 1

    def reconstruct(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialise ``(L, Θ)`` planes shaped to ``dims`` in one integer pass.

        ``L = base_L + dL`` (saturated to the representation's L range,
        zero-sentinel preserved), ``Θ = base_T + dΘ`` (mod ``2^theta_bits``);
        outliers then override individual linear indices with raw words — the
        exact v2 finalisation, generalised across base models, residual codecs,
        **and scalar widths** (SPF-16/32/64/128 via the representation's Format).
        """
        if self.base.profile_id == int(BaseModelKind.BYTE_MASS_PLANE):
            raise SPFError("BYTE_MASS_PLANE is a typed integer mass carrier; use the byte output consumer")
        fmt = self.representation.format
        wide = fmt.total_bits > 64
        total = self.total_elements
        model = get_base_model(self.base.profile_id)
        active, baseL, baseT = model.reconstruct_planes(self.base.params, self.dims, self.origin)
        active = np.asarray(active, dtype=np.int64)
        baseL = np.asarray(baseL, dtype=np.int64)
        baseT = np.asarray(baseT, dtype=np.int64)
        codec = get_residual_codec(self.residual.codec_id)
        dL, dT = codec.decode(self.residual.payload, int(active.size))
        l_val = baseL + np.asarray(dL, dtype=np.int64)
        t_val = baseT + np.asarray(dT, dtype=np.int64)
        L = np.full(total, fmt.l_code_zero, dtype=fmt.l_signed_dtype)
        T = np.zeros(total, dtype=fmt.theta_dtype)
        # Mirror TensorBlock.reconstruct exactly: a resolved L of the zero
        # sentinel is zero, not a tiny finite magnitude — preserve it (Θ forced
        # 0) rather than clipping it to a finite L.
        zero = l_val == int(fmt.l_code_zero)
        L[active] = np.where(zero, fmt.l_code_zero, _sat_l(l_val, fmt)).astype(fmt.l_signed_dtype)
        # Θ wraps to [0, 2^theta_bits). For SPF-128 the ring (2⁶⁴) overruns int64,
        # so reduce in Python int (object) before casting to the unsigned code.
        t_masked = (t_val.astype(object) & fmt.word_mask_theta) if wide else (t_val & fmt.word_mask_theta)
        T[active] = np.where(zero, 0, t_masked).astype(fmt.theta_dtype)
        if self.outliers:
            words = pack_words(L, T, fmt)
            word_mask = (1 << fmt.total_bits) - 1
            for linear, word in self.outliers.items():
                words[int(linear)] = (int(word) & word_mask) if wide else fmt.word_dtype(int(word) & word_mask)
            L = unpack_L(words, fmt)
            T = unpack_theta(words, fmt)
        return L.reshape(self.dims), T.reshape(self.dims)

    def to_words(self) -> np.ndarray:
        if self.base.profile_id == int(BaseModelKind.BYTE_MASS_PLANE):
            raise SPFError("BYTE_MASS_PLANE has typed integer planes, not scalar SPF words")
        L, T = self.reconstruct()
        return pack_words(L, T, self.representation.format)

    def active_addresses(self) -> np.ndarray:
        """Stored support plus absolute overrides, without reconstructing a plane."""
        if self.base.profile_id == int(BaseModelKind.STB_AFFINE_POLY) and "active_indices" in self.base.params:
            active = np.asarray(self.base.params["active_indices"], dtype=np.int64)
        else:
            active = np.arange(self.total_elements, dtype=np.int64)
        return np.union1d(active, np.fromiter(self.outliers, dtype=np.int64)) if self.outliers else active

    def sample_words(self, linear: Any, *, phase: bool = True) -> np.ndarray:
        """Consume addressed base/residual codes before scalar finalization.

        Registered models may provide ``sample_planes``. Models without it use
        their declared reconstruction consumer; their storage is never mistaken
        for a separable amplitude factorization.
        """
        from .residual_codecs import decode_residual_at
        from .v2 import apply_outliers

        if self.base.profile_id == int(BaseModelKind.BYTE_MASS_PLANE):
            raise SPFError("BYTE_MASS_PLANE must be consumed through its typed mass-plane view")

        query = np.asarray(linear, dtype=np.int64)
        flat = query.reshape(-1)
        if np.any(flat < 0) or np.any(flat >= self.total_elements):
            raise SPFError("SPFBlockV3 sample outside extent")
        fmt = self.representation.format
        model = get_base_model(self.base.profile_id)
        sample = getattr(model, "sample_planes", None)
        if sample is None:
            return self.to_words().reshape(-1)[flat].reshape(query.shape)
        active = (
            self.base.params.get("active_indices")
            if self.base.profile_id == int(BaseModelKind.STB_AFFINE_POLY)
            else None
        )
        if active is None:
            live = np.ones(flat.size, dtype=bool)
            positions, count = flat, self.total_elements
        else:
            active = np.asarray(active, dtype=np.int64)
            if active.size > 1 and np.any(active[1:] <= active[:-1]):
                raise SPFError("addressed base requires strictly increasing active indices")
            positions = np.searchsorted(active, flat)
            live = positions < active.size
            live[live] &= active[positions[live]] == flat[live]
            count = active.size
        L, T = sample(self.base.params, self.dims, self.origin, flat[live], phase=phase)
        dL, dT = decode_residual_at(self.residual, positions[live], count, phase=phase)
        L, T = L + dL, T + dT
        zero = fmt.l_code_zero == L
        L = np.where(zero, fmt.l_code_zero, _sat_l(L, fmt)).astype(fmt.l_signed_dtype)
        T = np.where(zero, 0, (T.astype(object) if fmt.total_bits > 64 else T) & fmt.word_mask_theta).astype(
            fmt.theta_dtype
        )
        words = np.full(flat.size, fmt.word_zero, dtype=object if fmt.total_bits > 64 else fmt.word_dtype)
        words[live] = pack_words(L, T, fmt)
        apply_outliers(words, flat, self.outliers)
        return words.reshape(query.shape)

    def storage_bits(self) -> int:
        """Charged payload cost: base + residual + outliers (design rule 5)."""
        model = get_base_model(self.base.profile_id)
        codec = get_residual_codec(self.residual.codec_id)
        return (
            model.storage_bits(self.base.params)
            + codec.storage_bits(self.residual.payload)
            + ((32 + self.representation.format.total_bits) * len(self.outliers) if self.outliers else 0)
        )


# --------------------------------------------------------------------------- #
# v2 → v3 relabel (design §14 / §8). A relabel, not a port.
# --------------------------------------------------------------------------- #


def tensorblock_to_v3(block: Any) -> SPFBlockV3:
    """Relabel a v2 :class:`spf.tensor.v2.TensorBlock` as a v3 :class:`SPFBlockV3`.

    ``representation = POLAR_SPF32``, ``base = STB_AFFINE_POLY(header, bands)``,
    ``residual = RAW_INT16(dL, dT)``, ``outliers`` preserved. Reconstruction is
    bit-identical to the v2 block (proved by the T17 migration suite).
    """
    from .core import ActiveEncoding

    block.validate()
    header = block.header
    total = header.total_elements
    if header.active_encoding == ActiveEncoding.DENSE_ALL:
        active = np.arange(total, dtype=np.int64)
    else:
        active = np.asarray(block.active_indices, dtype=np.int64)
    base = BaseModelRef(
        int(BaseModelKind.STB_AFFINE_POLY),
        params={
            "header": header,
            "bands": tuple(block.bands),
            "active_indices": active,
        },
    )
    residual = ResidualRef(
        int(ResidualCodecKind.RAW_INT16),
        payload={
            # Mirror v2's int16 truncation-before-add (TensorBlock.reconstruct
            # casts dL/dT to int16 first): wrap to int16 then widen so the
            # relabel is bit-identical even for an in-memory block whose residual
            # arrays carry out-of-int16 values (v2.validate does not constrain
            # residual range). Production residuals are already int16 → no-op.
            "dL": np.asarray(block.dL, dtype=np.int16).astype(np.int64),
            "dT": np.asarray(block.dT, dtype=np.int16).astype(np.int64),
        },
    )
    return SPFBlockV3(
        dims=tuple(header.dims),
        origin=tuple(header.origin or (0,) * (header.order or 1)),
        representation=RepresentationRef(int(Representation.POLAR_SPF32)),
        base=base,
        residual=residual,
        outliers=dict(block.outliers),
    )


def from_dense(
    z: Any,
    representation: int = int(Representation.POLAR_SPF32),
) -> SPFBlockV3:
    """Encode a dense complex array as a v3 block at any ``POLAR_SPF*`` width.

    Width-generic entry point: encodes ``z`` to ``(L, Θ)`` planes for the
    representation's :class:`Format`, then carries them as an exact ``RAW_INT16``
    residual over a zero ``SEPARABLE_PHASE`` base (``base = 0`` everywhere). The
    Θ code is stored as a **signed int64 residue** — for SPF-128 its 64-bit code
    overruns int64, but ``reconstruct`` re-applies ``mod 2^theta_bits`` so the
    exact code is recovered. Round-trips through :func:`block_to_complex` to the
    representation's quantisation floor.
    """
    rep = RepresentationRef(int(representation))
    fmt = rep.format
    zz = np.asarray(z, dtype=np.complex128)
    dims = tuple(int(d) for d in zz.shape)
    plane_L, plane_T = encode_complex(zz, fmt)
    dL = np.asarray(plane_L, dtype=np.int64).reshape(-1)
    dT = np.asarray(plane_T, dtype=fmt.theta_dtype).reshape(-1).astype(np.int64)
    base = BaseModelRef(int(BaseModelKind.SEPARABLE_PHASE), {})
    residual = ResidualRef(int(ResidualCodecKind.RAW_INT16), {"dL": dL, "dT": dT})
    return SPFBlockV3(
        dims=dims,
        origin=(0,) * len(dims),
        representation=rep,
        base=base,
        residual=residual,
    )


def block_to_complex(block: SPFBlockV3) -> np.ndarray:
    """Reconstruct a v3 block and decode to complex128 at its representation width."""
    fmt = block.representation.format
    plane_L, plane_T = block.reconstruct()
    return decode_complex(plane_L, plane_T, fmt)


__all__ = [
    "BASE_MODEL_REGISTRY",
    "BYTE_MASS_MAX_CONTEXT_LIMIT",
    "BYTE_MASS_MAX_OUTPUT_LENGTH",
    "BYTE_MASS_MAX_Q",
    "BYTE_MASS_NODE_COUNT",
    "BYTE_MASS_SCHEMA",
    "BYTE_MASS_SCORE_MAX",
    "BYTE_MASS_SCORE_MIN",
    "PROFILE_CP_BANNED",
    "PROFILE_NONABELIAN_MAX",
    "PROFILE_NONABELIAN_MIN",
    "PROFILE_SEPARABLE_MAX",
    "PROFILE_SEPARABLE_MIN",
    "PROFILE_SPECTRAL_MAX",
    "PROFILE_SPECTRAL_MIN",
    "PROFILE_TT_MAX",
    "PROFILE_TT_MIN",
    "REPRESENTATION_REGISTRY",
    "RESIDUAL_CODEC_REGISTRY",
    "SPFTENSOR_DTYPE_V3",
    "SPFTENSOR_V3_MAJOR",
    "SPFTENSOR_V3_MINOR",
    "BaseModel",
    "BaseModelKind",
    "BaseModelRef",
    "Representation",
    "RepresentationOps",
    "RepresentationRef",
    "ResidualCodec",
    "ResidualCodecKind",
    "ResidualRef",
    "SPFBlockV3",
    "block_to_complex",
    "format_for_representation",
    "from_dense",
    "get_base_model",
    "get_residual_codec",
    "register_base_model",
    "register_representation",
    "register_residual_codec",
    "tensorblock_to_v3",
    "validate_byte_mass_params",
    "validate_profile_id",
]
