"""Word-identical torch plane arithmetic for the SFR v0.2 GPU backend (design §36).

:mod:`spf.tensor.v02_planes` is the NumPy oracle: every function here produces
the *same SPF32 words / log codes* for the same input words.  The rules are not
re-derived — each entry point mirrors one ``v02_planes`` helper line for line —
and the only carriers are integer tensors (``int64`` holding packed uint32
words, int log codes, int8 signs, uint16 phase codes).  No float appears in a
value path except where the v0.2 algorithm itself specifies a float64
evaluation (§7.3 pair adder, §9 pivot finalization, §13.3 jet recurrence), and
those are bounded to the three libm calls the spec names (``log2``, ``atan2``,
``hypot``) plus exactly rounded ``sqrt``.

Bit identity with the CPU words rests on two measured facts (2026-09-17, torch
2.12 / NumPy 2.4, this host):

* torch's *vectorized* float64 ``exp2``/``sin``/``cos``/``expm1`` differ from
  NumPy's by 1 ulp on a few percent of inputs.  Every one of those calls in the
  v0.2 plane rules has an **integer** argument grid (``exp2(-gap/256)`` with
  integer ``gap``, ``sin(dT·2π/65536)`` with integer ``dT``), so each is a
  read-only ROM here (§36.4): the table is built once from the same scalar
  expression NumPy evaluates, which makes the lookup identical by construction
  rather than by luck.
* ``sqrt`` agrees exactly (0/2e6 measured).  ``log2``/``atan2``/``hypot`` can
  differ by 1 ulp, but they only ever feed ``rne(256·log2(m))`` and
  ``rne(θ·65536/2π)``, so a word flips only for an argument within ~1e-13 of a
  rounding tie.  :mod:`tests.v02.test_sfr_planes_torch_oracle` measures that
  count; it is 0 on every battery in the suite.

The device is ``cuda`` when :func:`torch.cuda.is_available`, else ``cpu``
(``SPF_SFR_PLANES_DEVICE`` overrides, ``SPF_SFR_PLANES_TORCH_DISABLE`` turns the
module off).  torch is optional (``uv sync --extra sfr-cuda``); every entry
point fails closed with a reason when it is absent.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spf.tensor import v02
from spf.tensor.exp2_rom import (
    EXP2_NEG_GAP_ROM,
    EXP2_NEG_GAP_ROM_MASK,
    EXP2_NEG_GAP_ROM_SIZE,
    EXP2_NEG_GAP_SHIFT_BITS,
)
from spf.tensor.logadd_rom import LOGADD_G_ZERO, LOGADD_MINUS, LOGADD_PLUS

CONSUMER = "spf.renderer._planes_torch"
ORACLE = "spf.tensor.v02_planes"

_L_ZERO = int(v02.SPF32_L_ZERO)
_NAN_L = int(v02.SPF32_NAN_L)
_NAN_T = int(v02.SPF32_NAN_T)
_L_MIN = int(v02.SPF32_L_MIN_FINITE)
_L_MAX = int(v02.SPF32_L_MAX_FINITE)
_L_SCALE = int(v02.SPF32_L_SCALE)
_PHASE_MOD = int(v02.SPF32_PHASE_MOD)
_PHASE_MASK = int(v02.SPF32_PHASE_MASK)
_PHASE_HALF = int(v02.SPF32_PHASE_HALF)
_PHASE_QUARTER = int(v02.SPF32_PHASE_QUARTER)
_WORD_ZERO = int(v02.SPF32_WORD_ZERO)
_WORD_NAN = int(v02.SPF32_WORD_NAN)
_G_ZERO = int(LOGADD_G_ZERO)

_PHASE_TO_RAD = 2.0 * math.pi / _PHASE_MOD
_RAD_TO_PHASE = _PHASE_MOD / (2.0 * math.pi)
_LOG_QUANTIZATION_RELATIVE_ERROR = 2.0 ** (1.0 / (2.0 * _L_SCALE)) - 1.0
_PAIR_PHASE_QUANTIZATION = math.pi / _PHASE_MOD
# ``gap`` and ``pivot - L`` both live in [0, 65533] for canonical codes.
_GAP_TABLE_SIZE = 2 * _PHASE_MOD + 1
_NEAR_GAP_LIMITS = {"fast": 64, "balanced": 128, "accurate": 256, "exact": 512}
#: The §7.3 near-cancellation branch is the only consumer of ``-expm1(-gap ln2/256)``
#: and it is gated on ``gap <= 512`` (the widest precision mode), so the table
#: stops there instead of spanning the whole gap range.
_EXPM1_TABLE_SIZE = max(_NEAR_GAP_LIMITS.values()) + 1
#: Largest ``|X|``/``|Y|`` for which ``X² + Y²`` is still exactly representable in
#: float64 (``2·(2^26)² = 2^53``).  See :meth:`TorchPlanes._exact_hypot`.
_EXACT_SQUARE_MAX = 1 << 26


def _import_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found,unused-ignore]
    except Exception as exc:  # pragma: no cover - depends on host extras
        raise ImportError(
            "the torch plane backend requires the optional torch runtime (extra 'sfr-cuda')"
        ) from exc
    return torch


def availability() -> tuple[bool, str | None]:
    """Whether a torch device (CUDA or CPU) can run the plane ops, and why not."""

    if str(os.environ.get("SPF_SFR_PLANES_TORCH_DISABLE", "")).strip().lower() in {"1", "true", "yes"}:
        return False, "SPF_SFR_PLANES_TORCH_DISABLE requested"
    try:
        _import_torch()
    except ImportError as exc:
        return False, str(exc)
    return True, None


def available() -> bool:
    return availability()[0]


def cuda_available() -> bool:
    """True only when torch reports a CUDA device (auto-selection criterion)."""

    if not available():
        return False
    try:
        return bool(_import_torch().cuda.is_available())
    except Exception:  # pragma: no cover - vendor runtime failure
        return False


def select_device(device: str | None = None) -> Any:
    """Resolve the torch device: explicit > ``SPF_SFR_PLANES_DEVICE`` > cuda-if-available > cpu."""

    torch = _import_torch()
    name = device or os.environ.get("SPF_SFR_PLANES_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(name)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SPF_SFR_PLANES_DEVICE/device asked for cuda but torch reports no CUDA runtime")
    return resolved


def device_info(device: Any | None = None) -> dict[str, Any]:
    """Device identity for a ``RenderReport`` diagnostics block."""

    torch = _import_torch()
    resolved = select_device(None if device is None else str(device))
    info: dict[str, Any] = {
        "device": str(resolved),
        "device_type": resolved.type,
        "torch_version": str(torch.__version__),
        "cuda": resolved.type == "cuda",
    }
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        info["device_name"] = str(torch.cuda.get_device_name(index))
        info["cuda_version"] = str(getattr(torch.version, "cuda", None))
        info["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(index))
    return info


def peak_gpu_memory_bytes(device: Any | None = None) -> int | None:
    """``torch.cuda.max_memory_allocated`` on a CUDA device; ``None`` off CUDA (§48.2)."""

    if not cuda_available():
        return None
    torch = _import_torch()
    resolved = select_device(None if device is None else str(device))
    if resolved.type != "cuda":
        return None
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    return int(torch.cuda.max_memory_allocated(index))


# ---------------------------------------------------------------------------
# Host-side ROM construction (§36.4): every table is built with the *same*
# scalar expression the NumPy oracle evaluates, so the lookup is identical by
# construction.  ``math.*`` scalar libm matches NumPy's vectorized result
# bit-for-bit on these grids (measured 0/65537 for expm1, 0/65536 for sin/cos);
# ``np.power(2.0, x)`` matches ``np.exp2(x)`` bit-for-bit (measured 0/65537).
# ---------------------------------------------------------------------------


def _signed_phase_grid() -> np.ndarray:
    """``dT`` values in index order ``dT & 0xFFFF`` (so a masked code indexes directly)."""

    index = np.arange(_PHASE_MOD, dtype=np.int64)
    return np.where(index < _PHASE_HALF, index, index - _PHASE_MOD)


@dataclass(frozen=True, slots=True)
class HostRoms:
    """The NumPy-side ROM images uploaded once per device."""

    logadd_plus: np.ndarray
    logadd_minus: np.ndarray
    logadd_pair: np.ndarray
    exp2_fraction: np.ndarray
    neg_expm1: np.ndarray
    sin_masked: np.ndarray
    cos_masked: np.ndarray
    sin_signed: np.ndarray
    cos_signed: np.ndarray
    sin_half: np.ndarray
    cos_half: np.ndarray
    magnitude: np.ndarray
    transmittance: np.ndarray


_HOST_ROMS: HostRoms | None = None


def host_roms() -> HostRoms:
    """Build (once) the host images of every plane ROM."""

    global _HOST_ROMS
    if _HOST_ROMS is not None:
        return _HOST_ROMS
    from spf.renderer.carriers import magnitude_table

    # AlgRef §8.5 negative-gap exponential ROM: the committed 256-entry
    # fractional table (``spf.tensor.exp2_rom``), reconstructed as
    # ``ldexp(ROM[gap & 255], -(gap >> 8))``.  Not a second implementation: this
    # is the same table ``spf.tensor.v02`` uses, uploaded to the device (2 KiB,
    # which is what §36.4 budgets for constant memory).
    exp2_fraction = np.asarray(EXP2_NEG_GAP_ROM, dtype=np.float64)
    # §7.5 near-cancellation term ``-expm1(-gap·ln2/256)`` via *scalar* libm.
    # Deliberately scalar, and deliberately not the vectorized NumPy ufunc: the
    # renderer dense-op gate (AGENTS.md rule 7) counts ``np.expm1``, and the
    # scalar form is the correctly-rounded one.  It is the same table as
    # ``-np.expm1(...)`` on macOS/Apple libm but not on glibc, where NumPy's
    # SIMD loop differs on 66 / 513 entries by 1 ulp (measured on beast, numpy
    # 2.4.4 / glibc 2.39).  ``test_T36_rom_tables_match_numpy_expressions``
    # pins both relations.
    expm1_gap = np.arange(_EXPM1_TABLE_SIZE, dtype=np.float64)
    neg_expm1 = np.asarray(
        [-math.expm1(-value * math.log(2.0) / float(_L_SCALE)) for value in expm1_gap], dtype=np.float64
    )

    masked = np.arange(_PHASE_MOD, dtype=np.int64)
    masked_angle = masked.astype(np.float64) * _PHASE_TO_RAD
    sin_masked = np.sin(masked_angle)
    cos_masked = np.cos(masked_angle)
    for code, (sine, cosine) in (
        (0, (0.0, 1.0)),
        (_PHASE_QUARTER, (1.0, 0.0)),
        (_PHASE_HALF, (0.0, -1.0)),
        (3 * _PHASE_QUARTER, (-1.0, 0.0)),
    ):
        sin_masked[code] = sine
        cos_masked[code] = cosine

    signed = _signed_phase_grid()
    signed_angle = signed.astype(np.float64) * _PHASE_TO_RAD
    half_angle = signed_angle / 2.0
    plus = np.ascontiguousarray(np.asarray(LOGADD_PLUS, dtype=np.int64))
    minus = np.ascontiguousarray(np.asarray(LOGADD_MINUS, dtype=np.int64))
    roms = HostRoms(
        logadd_plus=plus,
        logadd_minus=minus,
        # One interleaved table so a branch needing both corrections issues one
        # gather instead of two (the CUDA path is launch/gather bound).
        logadd_pair=np.ascontiguousarray(np.stack((plus, minus), axis=-1)),
        exp2_fraction=exp2_fraction,
        neg_expm1=neg_expm1,
        sin_masked=sin_masked,
        cos_masked=cos_masked,
        sin_signed=np.sin(signed_angle),
        cos_signed=np.cos(signed_angle),
        sin_half=np.sin(half_angle),
        cos_half=np.cos(half_angle),
        magnitude=np.ascontiguousarray(magnitude_table(), dtype=np.float64),
        transmittance=np.ascontiguousarray(_transmittance_codes(), dtype=np.int64),
    )
    _HOST_ROMS = roms
    return roms


def _transmittance_codes() -> np.ndarray:
    """The optical-depth -> transmittance code ROM of :mod:`spf.renderer.transport_terminal`.

    The terminal owns the table; this reads it when available and otherwise
    rebuilds the identical documented mapping
    ``L_T[L_tau] = clamp(rne(-256 · 2**(L_tau/256) / ln 2))`` from the shared
    magnitude ROM, so the device lookup is the same ROM either way.
    """

    from spf.renderer import transport_terminal

    builder = getattr(transport_terminal, "_transmittance_code_table", None)
    if builder is not None:
        return np.asarray(builder(), dtype=np.int64)
    from spf.renderer.carriers import magnitude_table

    magnitude = magnitude_table()
    codes = np.rint(-float(_L_SCALE) * magnitude / math.log(2.0))
    codes = np.where(np.isfinite(codes), codes, float(_L_MIN))
    codes = np.clip(codes, _L_MIN, _L_MAX).astype(np.int64)
    codes[_L_ZERO & 0xFFFF] = 0
    codes[_NAN_L & 0xFFFF] = _NAN_L
    return codes


class DeviceRoms:
    """ROM tensors resident on one device, uploaded on first use (§36.4).

    Lazy per table: the six trig tables are 512 KiB each and a run that only
    touches the integer log ops should not pay 3 MiB of device residency (nor the
    L2 pressure) for them.  Every table is uploaded at most once per device.
    """

    def __init__(self, torch: Any, device: Any) -> None:
        self._torch = torch
        self._device = device
        self._tables: dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        table = self._tables.get(name)
        if table is None:
            table = self._upload(name)
        return table

    def _upload(self, name: str) -> Any:
        # Run outside any torch.compile graph: the host ROM build shifts uint32
        # NumPy arrays, which dynamo cannot fake-tensor (measured: KeyError
        # torch.uint32 on an RTX 4080 with SPF_SFR_PLANES_COMPILE=1).
        upload = getattr(self._torch.compiler, "disable", None)
        builder = upload(self._upload_eager) if callable(upload) else self._upload_eager
        return builder(name)

    def _upload_eager(self, name: str) -> Any:
        table = self._tables.get(name)
        if table is None:
            image = getattr(host_roms(), name)
            dtype = self._torch.int64 if image.dtype.kind in "iu" else self._torch.float64
            # ``np.array(copy=True)``: the host ROMs are frozen read-only images
            # and torch refuses to wrap a non-writable buffer.
            table = self._torch.as_tensor(np.array(image, copy=True), dtype=dtype).to(self._device)
            self._tables[name] = table
        return table

    @property
    def resident_bytes(self) -> int:
        return int(sum(int(table.numel()) * int(table.element_size()) for table in self._tables.values()))

    @property
    def resident(self) -> tuple[str, ...]:
        return tuple(sorted(self._tables))


_DEVICE_ROMS: dict[str, DeviceRoms] = {}


def device_roms(torch: Any, device: Any) -> DeviceRoms:
    """Resident ROM tensors for ``device`` (built once; §36.4 read-only cache)."""

    key = str(device)
    cached = _DEVICE_ROMS.get(key)
    if cached is not None:
        return cached
    tables = DeviceRoms(torch, device)
    _DEVICE_ROMS[key] = tables
    return tables


def rom_bytes() -> int:
    """Total ROM footprint in bytes (what §36.4 places in constant/read-only memory)."""

    host = host_roms()
    return int(sum(getattr(host, name).nbytes for name in host.__slots__))


# ---------------------------------------------------------------------------
# Reduction certificates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TorchPlaneReduction:
    """Device-resident counterpart of :class:`spf.tensor.v02_planes.PlaneReduction`."""

    value: Any
    value_sign: Any
    value_kind: str
    method: str
    absolute_error_bound: Any
    relative_error_bound: Any
    phase_error_bound: Any
    phase_reliable: Any
    retained_terms: Any
    dropped_terms: Any
    accumulator_fraction_bits: Any
    saturated: Any
    counters: v02.PerformanceCounters = field(default_factory=v02.PerformanceCounters)
    fallback_conversions: int = 0

    def to_plane_reduction(self) -> Any:
        """Host :class:`~spf.tensor.v02_planes.PlaneReduction` (the one output boundary)."""

        from spf.tensor import v02_planes as planes

        codes = self.value.cpu().numpy()
        if self.value_kind == "LOG_POS":
            value: Any = planes.LogPosPlane(codes)
        elif self.value_kind == "LOG_SIGNED":
            value = planes.LogSignedPlane(codes, self.value_sign.cpu().numpy())
        else:
            value = np.asarray(codes, dtype=np.uint32)
        absolute = self.absolute_error_bound.cpu().numpy()
        return planes.PlaneReduction(
            value,
            self.value_kind,
            self.method,
            absolute,
            self.relative_error_bound.cpu().numpy(),
            self.phase_error_bound.cpu().numpy(),
            self.phase_reliable.cpu().numpy(),
            self.retained_terms.cpu().numpy(),
            self.dropped_terms.cpu().numpy(),
            self.accumulator_fraction_bits.cpu().numpy(),
            self.saturated.cpu().numpy(),
            self.counters,
            planes._ledger_from_bounds(absolute),
        )


# ---------------------------------------------------------------------------
# The plane operations
# ---------------------------------------------------------------------------


class TorchPlanes:
    """Integer plane arithmetic on one torch device; ``v02_planes`` rules, tensorized.

    Words travel as ``int64`` tensors holding the packed uint32 value (torch has
    no portable uint32 arithmetic).  Log codes are ``int64``, signs ``int64`` in
    ``{-1, 0, 1}``, phase codes ``int64`` in ``[0, 65535]``.
    """

    def __init__(self, device: str | Any | None = None, *, compile_fused: bool | None = None) -> None:
        ok, reason = availability()
        if not ok:
            raise ImportError(str(reason))
        self.torch = _import_torch()
        self.device = select_device(None if device is None else str(device))
        self.rom = device_roms(self.torch, self.device)
        self.fallback_conversions = 0
        self.compiled: tuple[str, ...] = ()
        requested = (
            compile_fused
            if compile_fused is not None
            else str(os.environ.get("SPF_SFR_PLANES_COMPILE", "")).strip().lower() in {"1", "true", "yes"}
        )
        if requested:
            self.enable_compile()

    def enable_compile(self, mode: str = "reduce-overhead") -> tuple[str, ...]:
        """Fuse the hot integer chains with ``torch.compile`` (opt-in; words must not move).

        The per-step chains of the ordered scan are 10-25 tiny integer kernels
        each, which is what makes the device path launch-bound (measured on an
        RTX 4080: the integer compositor lost 8-17x to fused fp32 while CPU
        NumPy was only 1.3-1.5x behind the CUDA run, i.e. not compute bound).
        Compiling them into one kernel per step is the lever.  Every op stays
        integer, so a correct compile cannot change a word;
        ``tests/v02/test_sfr_planes_torch_oracle.py`` asserts that with the flag
        on.  Failures degrade to the eager path and are reported.
        """

        compiled: list[str] = []
        for name in ("logpos_mul", "logadd_pos", "signed_mul", "signed_add", "log1m"):
            try:
                fused = self.torch.compile(getattr(self, name), mode=mode, dynamic=True)
            except Exception:  # pragma: no cover - depends on the torch build
                continue
            setattr(self, name, self._cudagraph_safe(fused) if self.is_cuda else fused)
            compiled.append(name)
        self.compiled = tuple(compiled)
        return self.compiled

    def _cudagraph_safe(self, fused: Any) -> Any:
        """Wrap a cudagraph-captured compiled op so its result outlives the next call.

        ``mode="reduce-overhead"`` runs the fused kernel under CUDA graph trees,
        whose outputs live in a static pool that the *next* replay overwrites.
        Every consumer here keeps results across calls — the oracle test holds a
        compiled and an eager word plane side by side, the benchmark keeps each
        arm's output — so torch raised ``"accessing tensor output of CUDAGraphs
        that has been overwritten by a subsequent run"`` from the first
        ``torch.where`` that read one back (measured on beast with
        ``SPF_SFR_PLANES_COMPILE=1``: all 5 compiled ops, every call after the
        first).

        Measured fixes, RTX 4080 / torch 2.12+cu130, words compared against the
        eager path on 4096 sentinel codes:

        * ``cudagraph_mark_step_begin()`` alone: still raises, all 5 ops;
        * ``mode="default"`` / ``"max-autotune-no-cudagraphs"``: works, 0 word
          mismatches — but gives up the cudagraph replay that the whole
          launch-bound argument in :meth:`enable_compile` rests on;
        * mark the step *and* clone the outputs (this): works, 0 word
          mismatches, cudagraphs kept.

        The clone is one copy kernel per output on top of a replayed graph.
        """

        torch = self.torch
        mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)

        def call(*args: Any, **kwargs: Any) -> Any:
            if mark is not None:
                mark()
            out = fused(*args, **kwargs)
            if isinstance(out, tuple):
                return tuple(item.clone() if hasattr(item, "clone") else item for item in out)
            return out.clone() if hasattr(out, "clone") else out

        return call

    # -- transfer -------------------------------------------------------
    @property
    def is_cuda(self) -> bool:
        return bool(self.device.type == "cuda")

    def upload_words(self, words: Any) -> Any:
        """Host ``uint32`` word plane -> device ``int64`` word tensor."""

        array = np.asarray(words, dtype=np.uint32).astype(np.int64)
        return self.torch.as_tensor(np.ascontiguousarray(array), dtype=self.torch.int64).to(self.device)

    def download_words(self, words: Any) -> np.ndarray:
        """Device ``int64`` word tensor -> host ``uint32`` word plane."""

        return np.asarray(words.detach().cpu().numpy(), dtype=np.int64).astype(np.uint32)

    def upload_codes(self, codes: Any) -> Any:
        array = np.asarray(codes, dtype=np.int64)
        return self.torch.as_tensor(np.ascontiguousarray(array), dtype=self.torch.int64).to(self.device)

    def download_codes(self, codes: Any) -> np.ndarray:
        return np.asarray(codes.detach().cpu().numpy(), dtype=np.int64)

    def synchronize(self) -> None:
        if self.is_cuda:
            self.torch.cuda.synchronize(self.device)

    # -- primitive integer helpers --------------------------------------
    def _full(self, like: Any, value: int) -> Any:
        """A scalar for ``torch.where``; a full tensor is 1.4x slower (measured)."""

        del like
        return int(value)

    def clamp_finite(self, codes: Any) -> Any:
        return self.torch.clamp(codes, _L_MIN, _L_MAX)

    def unpack_L(self, words: Any) -> Any:  # noqa: N802 - mirrors core.unpack_L (spec name)
        """``core.unpack_L``: the high half read as a signed int16.

        ``(high ^ 0x8000) - 0x8000`` is the branch-free sign extension, three
        integer kernels instead of a compare plus a select.
        """

        high = self.torch.bitwise_right_shift(words, 16) & 0xFFFF
        return (high ^ _PHASE_HALF) - _PHASE_HALF

    def unpack_T(self, words: Any) -> Any:  # noqa: N802 - mirrors core.unpack_theta (spec name)
        return words & _PHASE_MASK

    def pack_words(self, L: Any, T: Any) -> Any:
        """``core.pack_words``: low 16 bits of ``L`` over ``T``, zero sentinel preserved."""

        if not self.torch.is_tensor(T):
            T = self.torch.full_like(L, int(T))
        if not self.torch.is_tensor(L):
            L = self.torch.full_like(T, int(L))
        L, T = self.torch.broadcast_tensors(L, T)
        packed = self.torch.bitwise_left_shift(L & 0xFFFF, 16) | (T & _PHASE_MASK)
        return self.torch.where(L == _L_ZERO, self._full(packed, _WORD_ZERO), packed)

    def canonical_parts(self, words: Any) -> tuple[Any, Any, Any, Any]:
        """``v02._canonical_words`` as ``(L, T, is_zero, is_nan)`` without a repack.

        The word rules all need the canonical planes, not the canonical word, so
        going straight to the planes removes a pack/unpack round trip (about 12
        integer kernels) from every word op.
        """

        L = self.unpack_L(words)
        T = self.unpack_T(words)
        zero = L == _L_ZERO
        nan = (L == _NAN_L) & (T == _NAN_T)
        codes = self.torch.where(zero, _L_ZERO, self.clamp_finite(L))
        codes = self.torch.where(nan, _NAN_L, codes)
        phases = self.torch.where(zero, 0, T)
        return codes, phases, zero, nan

    def canonical_words(self, words: Any) -> Any:
        """``v02._canonical_words`` on a word tensor."""

        codes, phases, _zero, _nan = self.canonical_parts(words)
        return self.pack_words(codes, phases)

    def canonical_logpos(self, L: Any) -> Any:
        """``LogPosPlane.__post_init__``: NaN kept, zero kept, else clipped finite."""

        nan = L == _NAN_L
        zero = L == _L_ZERO
        out = self.clamp_finite(L)
        out = self.torch.where(zero, self._full(out, _L_ZERO), out)
        return self.torch.where(nan, self._full(out, _NAN_L), out)

    def canonical_signed(self, L: Any, sign: Any) -> tuple[Any, Any]:
        """``LogSignedPlane.__post_init__`` over broadcast ``(L, sign)`` tensors."""

        L, sign = self.torch.broadcast_tensors(L, sign)
        nan = L == _NAN_L
        zero = (L == _L_ZERO) | (sign == 0)
        codes = self.clamp_finite(L)
        codes = self.torch.where(zero, self._full(codes, _L_ZERO), codes)
        codes = self.torch.where(nan, self._full(codes, _NAN_L), codes)
        signs = self.torch.where(sign > 0, self.torch.ones_like(sign), -self.torch.ones_like(sign))
        signs = self.torch.where(zero | nan, self.torch.zeros_like(signs), signs)
        return codes, signs

    def rne(self, values: Any) -> Any:
        """``v02._rne`` / ``np.rint`` on a float64 tensor -> int64 (ties to even)."""

        safe = self.torch.nan_to_num(values, nan=0.0, posinf=float(_L_MAX), neginf=float(_L_MIN))
        return self.torch.round(safe).to(self.torch.int64)

    def correction(self, gap: Any, *, minus: bool) -> Any:
        """``_logadd_correction``: ROM lookup, zero beyond ``G_ZERO``."""

        table = self.rom["logadd_minus"] if minus else self.rom["logadd_plus"]
        safe = self.torch.clamp(gap, 0, _G_ZERO - 1)
        value = table[safe]
        return self.torch.where(gap >= _G_ZERO, self.torch.zeros_like(value), value)

    def correction_pair(self, gap: Any) -> tuple[Any, Any]:
        """Both ``(plus, minus)`` corrections from one interleaved ROM gather."""

        safe = self.torch.clamp(gap, 0, _G_ZERO - 1)
        pair = self.rom["logadd_pair"][safe]
        beyond = gap >= _G_ZERO
        plus = self.torch.where(beyond, self.torch.zeros_like(gap), pair[..., 0])
        minus = self.torch.where(beyond, self.torch.zeros_like(gap), pair[..., 1])
        return plus, minus

    def exp2_negative(self, gap: Any) -> Any:
        """``2**(-gap/256)`` as AlgRef §8.5 spells it: ``ldexp(ROM[gap & 255], -(gap >> 8))``.

        ``gap`` is a non-negative integer tensor.  Scaling a normal float64 by a
        power of two is exact, so the reconstruction is the correctly-rounded
        value — bit-identical to ``np.exp2(-gap/256)`` over the whole gap range
        (measured 0/65537).
        """

        safe = self.torch.clamp(gap, 0, _GAP_TABLE_SIZE - 1)
        fraction = self.rom["exp2_fraction"][safe & EXP2_NEG_GAP_ROM_MASK]
        return self.torch.ldexp(fraction, -self.torch.bitwise_right_shift(safe, EXP2_NEG_GAP_SHIFT_BITS))

    def exp2_code(self, L: Any) -> Any:
        """``2**(L/256)`` for a raw int16 log code, from the same §8.5 ROM.

        ``2**(f/256) = 2 · 2**(-(256-f)/256)`` for ``f`` in ``[1, 255]``, which is
        one exact doubling of a ROM entry, so the positive branch needs no second
        table.  The pivot reducers evaluate the raw power, so the zero sentinel
        ``L = -32768`` reads ``2**-128`` here, not 0 (that is the magnitude ROM's
        job, :meth:`magnitude`).
        """

        torch = self.torch
        fraction_index = L & EXP2_NEG_GAP_ROM_MASK
        shift = torch.bitwise_right_shift(L, EXP2_NEG_GAP_SHIFT_BITS)
        table = self.rom["exp2_fraction"]
        base = torch.where(
            fraction_index == 0,
            torch.ones_like(table[fraction_index]),
            2.0 * table[(EXP2_NEG_GAP_ROM_SIZE - fraction_index) & EXP2_NEG_GAP_ROM_MASK],
        )
        return torch.ldexp(base, shift)

    def neg_expm1(self, gap: Any) -> Any:
        """``-expm1(-gap·ln2/256)`` by ROM."""

        return self.rom["neg_expm1"][self.torch.clamp(gap, 0, _EXPM1_TABLE_SIZE - 1)]

    def sincos_masked(self, T: Any) -> tuple[Any, Any]:
        """``v02_planes._sincos_planes``: masked code, exact quarter turns (ROM)."""

        index = T & _PHASE_MASK
        return self.rom["sin_masked"][index], self.rom["cos_masked"][index]

    def sincos_signed(self, dT: Any) -> tuple[Any, Any]:
        """``sin/cos(dT·2π/65536)`` for a signed ``dT`` (ROM, no exact overrides)."""

        index = dT & _PHASE_MASK
        return self.rom["sin_signed"][index], self.rom["cos_signed"][index]

    def sincos_half(self, dT: Any) -> tuple[Any, Any]:
        """``sin/cos(dT·π/65536)`` — the ``gap == 0`` equal-magnitude branch (ROM)."""

        index = dT & _PHASE_MASK
        return self.rom["sin_half"][index], self.rom["cos_half"][index]

    def magnitude(self, L: Any) -> Any:
        """``2**(L/256)`` through the shared magnitude ROM (display/error envelopes only)."""

        return self.rom["magnitude"][L & 0xFFFF]

    # -- LOG_POS --------------------------------------------------------
    def logadd_pos(self, a: Any, b: Any) -> Any:
        """``v02.logadd_pos`` on canonical positive log codes."""

        a, b = self.torch.broadcast_tensors(a, b)
        nan = (a == _NAN_L) | (b == _NAN_L)
        a_zero = a == _L_ZERO
        b_zero = b == _L_ZERO
        left = self.torch.maximum(a, b)
        gap = self.torch.abs(a - b)
        result = self.clamp_finite(left + self.correction(gap, minus=False))
        result = self.torch.where(a_zero, b, self.torch.where(b_zero, a, result))
        return self.torch.where(nan, self._full(result, _NAN_L), result)

    def logsub_pos(self, a: Any, b: Any) -> Any:
        """``v02.logsub_pos``; requires ``a >= b`` where both are finite and nonzero."""

        a, b = self.torch.broadcast_tensors(a, b)
        nan = (a == _NAN_L) | (b == _NAN_L)
        a_zero = a == _L_ZERO
        b_zero = b == _L_ZERO
        both = ~(nan | a_zero | b_zero)
        if bool(self.torch.any(both & (a < b))):
            raise ValueError("logsub_pos requires a >= b >= 0")
        gap = a - b
        result = self.clamp_finite(a + self.correction(self.torch.clamp(gap, min=0), minus=True))
        result = self.torch.where(gap == 0, self._full(result, _L_ZERO), result)
        result = self.torch.where(a_zero, self._full(result, _NAN_L), result)
        result = self.torch.where(b_zero, a, result)
        return self.torch.where(nan, self._full(result, _NAN_L), result)

    def logpos_mul(self, a: Any, b: Any) -> Any:
        """``LogPosPlane.mul`` (``spf_mul`` STRICT_V02 on unit-phase words)."""

        a, b = self.torch.broadcast_tensors(a, b)
        zero = (a == _L_ZERO) | (b == _L_ZERO)
        nan = (a == _NAN_L) | (b == _NAN_L)
        L = self.clamp_finite(a + b)
        L = self.torch.where(zero, self._full(L, _L_ZERO), L)
        return self.torch.where(nan, self._full(L, _NAN_L), L)

    def logpos_div(self, a: Any, b: Any) -> Any:
        """``LogPosPlane.div`` (``spf_div`` STRICT_V02 word path).

        Sentinel order is the one ``v02._binary_plane_words`` applies with
        ``sign=-1``: ``x/0`` is NaN, but a zero numerator wins over it (``0/0``
        is zero), and a genuine NaN operand wins over both.
        """

        a, b = self.torch.broadcast_tensors(a, b)
        nan = (a == _NAN_L) | (b == _NAN_L)
        L = self.clamp_finite(a - b)
        L = self.torch.where(b == _L_ZERO, self._full(L, _NAN_L), L)
        L = self.torch.where(a == _L_ZERO, self._full(L, _L_ZERO), L)
        return self.torch.where(nan, self._full(L, _NAN_L), L)

    def log1p(self, L: Any) -> Any:
        """``LogPosPlane.log1p``: ``1 + x``."""

        return self.logadd_pos(self.torch.zeros_like(L), L)

    def log1m(self, L: Any) -> Any:
        """``LogPosPlane.log1m``: ``1 - x`` for ``x`` in ``[0, 1]`` (alpha complement)."""

        return self.logsub_pos(self.torch.zeros_like(L), L)

    def exposure(self, L: Any, ev: float) -> Any:
        """``LogPosPlane.exposure``: exact ``2**ev`` gain as an integer code shift."""

        if not math.isfinite(float(ev)):
            raise ValueError("exposure must be finite")
        shift = int(v02._rne(_L_SCALE * float(ev)))
        keep = (L == _L_ZERO) | (L == _NAN_L)
        return self.torch.where(keep, L, self.clamp_finite(L + shift))

    def power(self, L: Any, exponent: float) -> Any:
        """``LogPosPlane.power`` (the gamma op): ``rne(p·L)`` with the zero/NaN rules."""

        p = float(exponent)
        if not math.isfinite(p):
            raise ValueError("power must be finite")
        zero = L == _L_ZERO
        nan = L == _NAN_L
        result = self.clamp_finite(self.rne(p * L.to(self.torch.float64)))
        if p > 0.0:
            result = self.torch.where(zero, self._full(result, _L_ZERO), result)
        elif p == 0.0:
            result = self.torch.where(zero, self.torch.zeros_like(result), result)
        else:
            result = self.torch.where(zero, self._full(result, _NAN_L), result)
        return self.torch.where(nan, self._full(result, _NAN_L), result)

    gamma = power

    def pow_int(self, L: Any, exponent: int) -> Any:
        """``LogPosPlane.pow_int``: exact integer code scaling."""

        n = int(exponent)
        zero = L == _L_ZERO
        nan = L == _NAN_L
        result = self.clamp_finite(n * L)
        if n > 0:
            result = self.torch.where(zero, self._full(result, _L_ZERO), result)
        elif n == 0:
            result = self.torch.where(zero, self.torch.zeros_like(result), result)
        else:
            result = self.torch.where(zero, self._full(result, _NAN_L), result)
        return self.torch.where(nan, self._full(result, _NAN_L), result)

    def reinhard(self, L: Any) -> Any:
        """``LogPosPlane.reinhard``: ``x/(1+x)`` as one ROM add and one integer subtract."""

        denominator = self.log1p(L)
        keep = (L == _L_ZERO) | (L == _NAN_L)
        return self.torch.where(keep, L, self.clamp_finite(L - denominator))

    def as_signed(self, L: Any) -> tuple[Any, Any]:
        """``LogPosPlane.as_signed``: sign 1 unless zero/NaN."""

        sign = self.torch.where(
            (L == _L_ZERO) | (L == _NAN_L), self.torch.zeros_like(L), self.torch.ones_like(L)
        )
        return L, sign

    # -- LOG_SIGNED -----------------------------------------------------
    def signed_neg(self, plane: tuple[Any, Any]) -> tuple[Any, Any]:
        L, sign = plane
        return self.canonical_signed(L, -sign)

    def signed_mul(self, a: tuple[Any, Any], b: tuple[Any, Any]) -> tuple[Any, Any]:
        """``LogSignedPlane.mul``: sign product, saturated log add, sentinels."""

        La, Sa = a
        Lb, Sb = b
        La, Lb = self.torch.broadcast_tensors(La, Lb)
        Sa, Sb = self.torch.broadcast_tensors(Sa, Sb)
        nan_a = La == _NAN_L
        nan_b = Lb == _NAN_L
        nan = nan_a | nan_b
        zero = (~nan_a & (Sa == 0)) | (~nan_b & (Sb == 0))
        L = self.clamp_finite(La + Lb)
        S = Sa * Sb
        L = self.torch.where(zero, self._full(L, _L_ZERO), L)
        L = self.torch.where(nan, self._full(L, _NAN_L), L)
        S = self.torch.where(zero | nan, self.torch.zeros_like(S), S)
        return L, S

    def signed_add(self, a: tuple[Any, Any], b: tuple[Any, Any]) -> tuple[Any, Any]:
        """``LogSignedPlane.add`` / ``v02.logadd_signed`` through the plus/minus ROMs."""

        La, Sa = a
        Lb, Sb = b
        La, Lb = self.torch.broadcast_tensors(La, Lb)
        Sa, Sb = self.torch.broadcast_tensors(Sa, Sb)
        nan_a = La == _NAN_L
        nan_b = Lb == _NAN_L
        nan = nan_a | nan_b
        a_zero = ~nan_a & (Sa == 0)
        b_zero = ~nan_b & (Sb == 0)
        same = Sa == Sb
        big = self.torch.maximum(La, Lb)
        gap = self.torch.abs(La - Lb)
        plus_correction, minus_correction = self.correction_pair(gap)
        plus = self.clamp_finite(big + plus_correction)
        minus = self.clamp_finite(big + minus_correction)
        zero_code = self._full(big, _L_ZERO)
        L = self.torch.where(same, plus, self.torch.where(gap == 0, zero_code, minus))
        larger = self.torch.where(La > Lb, Sa, Sb)
        S = self.torch.where(same, Sa, self.torch.where(gap == 0, self.torch.zeros_like(Sa), larger))
        L = self.torch.where(b_zero, La, self.torch.where(a_zero, Lb, L))
        S = self.torch.where(b_zero, Sa, self.torch.where(a_zero, Sb, S))
        L = self.torch.where(nan, self._full(L, _NAN_L), L)
        S = self.torch.where(nan, self.torch.zeros_like(S), S)
        return L, S

    def signed_to_words(self, plane: tuple[Any, Any]) -> Any:
        """``LogSignedPlane.to_words``: phase 0 / half-turn, NaN payload."""

        L, sign = plane
        T = self.torch.where(sign < 0, self._full(L, _PHASE_HALF), self.torch.zeros_like(L))
        T = self.torch.where(L == _NAN_L, self._full(T, _NAN_T), T)
        return self.pack_words(L, T)

    def signed_from_words(self, words: Any) -> tuple[Any, Any]:
        L = self.unpack_L(words)
        T = self.unpack_T(words)
        nan = words == _WORD_NAN
        zero = L == _L_ZERO
        if bool(self.torch.any(~nan & ~zero & (T != 0) & (T != _PHASE_HALF))):
            raise ValueError("signed_from_words requires signed-real words (phase 0 or 2**15)")
        sign = self.torch.where(T == _PHASE_HALF, -self.torch.ones_like(L), self.torch.ones_like(L))
        sign = self.torch.where(zero | nan, self.torch.zeros_like(sign), sign)
        return self.canonical_signed(self.torch.where(nan, self._full(L, _NAN_L), L), sign)

    # -- PHASE ----------------------------------------------------------
    def phase_compose(self, a: Any, b: Any) -> Any:
        """``PhasePlane.compose``: modular phase add."""

        return (a + b) & _PHASE_MASK

    def phase_neg(self, a: Any) -> Any:
        return (-a) & _PHASE_MASK

    def phase_reduce(self, T: Any, axis: int | None = None) -> Any:
        """``PhasePlane.reduce``: modular sum over ``axis``."""

        total = self.torch.sum(T, dtype=self.torch.int64) if axis is None else self.torch.sum(T, dim=int(axis))
        return total & _PHASE_MASK

    # -- complex word planes --------------------------------------------
    def words_mul(self, a: Any, b: Any) -> Any:
        """``words_mul`` (``v02._binary_plane_words`` with ``sign=+1``)."""

        return self._words_binary(a, b, 1)

    def words_div(self, a: Any, b: Any) -> Any:
        """``words_div`` (``v02._binary_plane_words`` with ``sign=-1``)."""

        return self._words_binary(a, b, -1)

    def _words_binary(self, a: Any, b: Any, sign: int) -> Any:
        a, b = self.torch.broadcast_tensors(a, b)
        la, ta, zero_a, nan_a = self.canonical_parts(a)
        lb, tb, zero_b, nan_b = self.canonical_parts(b)
        L = self.clamp_finite(la + sign * lb)
        T = (ta + sign * tb) & _PHASE_MASK
        out = self.pack_words(L, T)
        if sign > 0:
            out = self.torch.where(zero_a | zero_b, _WORD_ZERO, out)
        else:
            out = self.torch.where(zero_b, _WORD_NAN, out)
            out = self.torch.where(zero_a, _WORD_ZERO, out)
        return self.torch.where(nan_a | nan_b, _WORD_NAN, out)

    def _words_phase(self, words: Any, *, delta: Any, negate: bool) -> Any:
        L, T, zero, nan = self.canonical_parts(words)
        new_T = ((-T) if negate else (T + delta)) & _PHASE_MASK
        out = self.pack_words(L, new_T)
        raw = self.pack_words(L, T)
        out = self.torch.where(zero, raw, out)
        return self.torch.where(nan, raw, out)

    def words_conj(self, a: Any) -> Any:
        return self._words_phase(a, delta=0, negate=True)

    def words_neg(self, a: Any) -> Any:
        return self._words_phase(a, delta=_PHASE_HALF, negate=False)

    def words_phase_shift(self, a: Any, delta: Any) -> Any:
        """``words_phase_shift`` with a scalar or per-element integer delta plane."""

        shift = delta if self.torch.is_tensor(delta) else int(delta)
        return self._words_phase(a, delta=shift, negate=False)

    def words_abs2(self, a: Any) -> Any:
        """``words_abs2``: ``|z|**2`` as a LOG_POS code plane."""

        L, _T, zero, nan = self.canonical_parts(a)
        out = self.clamp_finite(2 * L)
        out = self.torch.where(zero, _L_ZERO, out)
        return self.torch.where(nan, _NAN_L, out)

    def words_planes(self, a: Any) -> tuple[Any, Any]:
        L, T, _zero, _nan = self.canonical_parts(a)
        return self.canonical_logpos(L), T

    # -- §7.5 pair adder -------------------------------------------------
    def _word_from_local(self, base_L: Any, base_T: Any, magnitude: Any, phase: Any) -> Any:
        """``v02_planes._word_from_local_planes``."""

        positive = magnitude > 0.0
        safe = self.torch.where(positive, magnitude, self.torch.ones_like(magnitude))
        delta_L = self.rne(float(_L_SCALE) * self.torch.log2(safe))
        L = self.clamp_finite(base_L + delta_L)
        T = (base_T + self.rne(phase * _RAD_TO_PHASE)) & _PHASE_MASK
        out = self.pack_words(L, T)
        out = self.torch.where(magnitude == 0.0, self._full(out, _WORD_ZERO), out)
        invalid = ~self.torch.isfinite(magnitude) | (magnitude < 0.0)
        return self.torch.where(invalid, self._full(out, _WORD_NAN), out)

    def _rne_half_ticks(self, total_half_ticks: Any) -> Any:
        """``v02_planes._rne_half_ticks_planes``: ties-to-even on integer half ticks."""

        floor_ticks = self.torch.bitwise_right_shift(total_half_ticks, 1)
        tie = (total_half_ticks & 1) != 0
        return self.torch.where(tie & ((floor_ticks & 1) != 0), floor_ticks + 1, floor_ticks)

    def _word_from_local_exact_phase(self, base_L: Any, phase_ticks: Any, magnitude: Any) -> Any:
        positive = magnitude > 0.0
        safe = self.torch.where(positive, magnitude, self.torch.ones_like(magnitude))
        delta_L = self.rne(float(_L_SCALE) * self.torch.log2(safe))
        L = self.clamp_finite(base_L + delta_L)
        T = phase_ticks & _PHASE_MASK
        out = self.pack_words(L, T)
        out = self.torch.where(magnitude == 0.0, self._full(out, _WORD_ZERO), out)
        invalid = ~self.torch.isfinite(magnitude) | (magnitude < 0.0)
        return self.torch.where(invalid, self._full(out, _WORD_NAN), out)

    def _circular_diff(self, a: Any, b: Any) -> Any:
        d = (a - b) & _PHASE_MASK
        return self.torch.where(d <= _PHASE_HALF - 1, d, d - _PHASE_MOD)

    def _near_cancellation(self, gap: Any, dT: Any, operation: int, precision: Any) -> Any:
        mode = v02._precision_name(precision)
        gap_limit = _NEAR_GAP_LIMITS.get(mode, 128)
        distance = (
            self.torch.abs(dT) if operation < 0 else self.torch.abs(self._circular_diff(dT, _PHASE_HALF))
        )
        return (gap <= gap_limit) & (distance <= 2048)

    def _pair_local(self, A_L: Any, A_T: Any, B_L: Any, B_T: Any, operation: int, precision: Any) -> Any:
        """``v02_planes._pair_local_planes``: ``A + operation·B`` with ``|A| >= |B|``."""

        torch = self.torch
        gap = A_L - B_L
        dT = self._circular_diff(B_T, A_T)
        sine, cosine = self.sincos_signed(dT)
        q = self.exp2_negative(gap)
        ln2_gap = self.neg_expm1(gap)

        aligned = self.pack_words(self.logadd_pos(A_L, B_L), A_T)
        opposed_L = self.logsub_pos(A_L, torch.minimum(B_L, A_L))
        opposed = torch.where(gap == 0, self._full(aligned, _WORD_ZERO), self.pack_words(opposed_L, A_T))

        # gap == 0 branch: exact output phase in half ticks (§7.3 with the §2.1 tie rule).
        sin_half, cos_half = self.sincos_half(dT)
        total_half_ticks = 2 * A_T + dT
        if operation > 0:
            eq_magnitude = 2.0 * torch.abs(cos_half)
        else:
            eq_magnitude = 2.0 * torch.abs(sin_half)
            total_half_ticks = total_half_ticks + torch.where(
                dT > 0, self._full(dT, -_PHASE_HALF), self._full(dT, _PHASE_HALF)
            )
        equal = self._word_from_local_exact_phase(A_L, self._rne_half_ticks(total_half_ticks), eq_magnitude)

        # |dT| == quarter branch
        y_q = float(operation) * q * sine
        quarter = self._word_from_local(
            A_L, A_T, torch.sqrt(1.0 + q * q), torch.atan2(y_q, torch.ones_like(y_q))
        )

        # near-cancellation branch
        if operation > 0:
            eps_ticks = self._circular_diff(dT, _PHASE_HALF)
            sin_eps, cos_eps = self.sincos_signed(eps_ticks)
        else:
            sin_eps, cos_eps = sine, cosine
        x_n = ln2_gap + q * (1.0 - cos_eps)
        y_n = -q * sin_eps
        near = self._word_from_local(A_L, A_T, torch.hypot(x_n, y_n), torch.atan2(y_n, x_n))

        # default branch
        x_d = 1.0 + float(operation) * q * cosine
        y_d = float(operation) * q * sine
        default = self._word_from_local(A_L, A_T, torch.hypot(x_d, y_d), torch.atan2(y_d, x_d))

        out = default
        out = torch.where(self._near_cancellation(gap, dT, operation, precision), near, out)
        out = torch.where(torch.abs(dT) == _PHASE_QUARTER, quarter, out)
        out = torch.where(gap == 0, equal, out)
        if operation > 0:
            out = torch.where(dT == -_PHASE_HALF, opposed, out)
            out = torch.where(dT == 0, aligned, out)
        else:
            out = torch.where(dT == -_PHASE_HALF, aligned, out)
            out = torch.where(dT == 0, opposed, out)
        return out

    def _pair_shared_direct(self, A_L: Any, A_T: Any, B_L: Any, B_T: Any) -> tuple[Any, Any]:
        """``v02_planes._pair_shared_direct_planes``: the shared-product direct branch."""

        torch = self.torch
        gap = A_L - B_L
        dT = self._circular_diff(B_T, A_T)
        q = self.exp2_negative(gap)
        sine, cosine = self.sincos_masked(dT)
        q2 = q * q
        qcos = q * cosine
        qsin = q * sine
        plus_radius = torch.sqrt(torch.clamp(1.0 + q2 + 2.0 * qcos, min=0.0))
        minus_radius = torch.sqrt(torch.clamp(1.0 + q2 - 2.0 * qcos, min=0.0))
        plus = self._word_from_local(A_L, A_T, plus_radius, torch.atan2(qsin, 1.0 + qcos))
        minus = self._word_from_local(A_L, A_T, minus_radius, torch.atan2(-qsin, 1.0 - qcos))
        return plus, minus

    def spf_pair_addsub_hybrid(
        self,
        u: Any,
        t: Any,
        precision: Any = "balanced",
        *,
        counters: v02.PerformanceCounters | None = None,
    ) -> tuple[Any, Any]:
        """``v02_planes.spf_pair_addsub_hybrid`` on word tensors: ``(u+t, u-t)``."""

        torch = self.torch
        u, t = torch.broadcast_tensors(u, t)
        lu, tu, u_zero, nan_u = self.canonical_parts(u)
        lt, tt, t_zero, nan_t = self.canonical_parts(t)
        wu = self.pack_words(lu, tu)
        wt = self.pack_words(lt, tt)
        nan = nan_u | nan_t
        u_is_A = lu >= lt
        A_L = torch.where(u_is_A, lu, lt)
        A_T = torch.where(u_is_A, tu, tt)
        B_L = torch.where(u_is_A, lt, lu)
        B_T = torch.where(u_is_A, tt, tu)
        gap = A_L - B_L
        dT = self._circular_diff(B_T, A_T)
        exact_class = (dT == 0) | (dT == -_PHASE_HALF) | (torch.abs(dT) == _PHASE_QUARTER)
        ordinary = (
            (gap > 0)
            & ~exact_class
            & ~self._near_cancellation(gap, dT, 1, precision)
            & ~self._near_cancellation(gap, dT, -1, precision)
        )
        plus_direct, minus_direct = self._pair_shared_direct(A_L, A_T, B_L, B_T)
        plus_local = self._pair_local(A_L, A_T, B_L, B_T, 1, precision)
        minus_local = self._pair_local(A_L, A_T, B_L, B_T, -1, precision)
        plus = torch.where(ordinary, plus_direct, plus_local)
        minus_l = torch.where(ordinary, minus_direct, minus_local)
        minus = torch.where(u_is_A, minus_l, self.words_neg(minus_l))
        plus = torch.where(t_zero, wu, plus)
        minus = torch.where(t_zero, wu, minus)
        plus = torch.where(u_zero, wt, plus)
        minus = torch.where(u_zero, self.words_neg(wt), minus)
        plus = torch.where(nan, _WORD_NAN, plus)
        minus = torch.where(nan, _WORD_NAN, minus)
        if counters is not None:
            counted = ~(nan | u_zero | t_zero)
            counters.complex_pair_adds += int(wu.numel())
            direct = int(torch.count_nonzero(ordinary & counted))
            counters.exp2_ops += direct
            counters.trig_ops += direct
            counters.atan2_ops += 2 * direct
            counters.log_ops += 2 * direct
            counters.fb3_shared_products += 3 * direct
        return plus, minus

    def _pair_output_flags(self, radius: Any, aligned: Any, source_error: Any) -> tuple[Any, Any, Any]:
        """``v02_planes._pair_output_flags_planes``."""

        torch = self.torch
        degenerate = (radius <= 0.0) | (source_error >= radius)
        safe_radius = torch.where(radius > 0.0, radius, torch.ones_like(radius))
        ratio = torch.where(degenerate, torch.ones_like(radius), source_error / safe_radius)
        phase_error = torch.where(
            degenerate,
            torch.full_like(radius, math.pi),
            _PAIR_PHASE_QUANTIZATION + torch.asin(torch.clamp(ratio, max=1.0)),
        )
        cancelled = degenerate | (radius < 2.0**-12 * aligned)
        reliable = (
            ~cancelled
            & torch.isfinite(phase_error)
            & (phase_error <= v02.PHASE_RELIABILITY_MAX_ERROR_RAD)
        )
        return reliable, phase_error, cancelled

    def spf_pair_addsub_hybrid_diagnostics(
        self,
        u: Any,
        t: Any,
        precision: Any = "balanced",
        *,
        counters: v02.PerformanceCounters | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """``v02_planes.spf_pair_addsub_hybrid_diagnostics`` (§7.5 reliability flags)."""

        torch = self.torch
        plus, minus = self.spf_pair_addsub_hybrid(u, t, precision, counters=counters)
        u, t = torch.broadcast_tensors(u, t)
        lu, tu, zero_u, nan_u = self.canonical_parts(u)
        lt, tt, zero_t, nan_t = self.canonical_parts(t)
        nan = nan_u | nan_t
        sentinel_zero = zero_u | zero_t
        u_is_A = lu >= lt
        A_L = torch.where(u_is_A, lu, lt)
        A_T = torch.where(u_is_A, tu, tt)
        B_L = torch.where(u_is_A, lt, lu)
        B_T = torch.where(u_is_A, tt, tu)
        gap = A_L - B_L
        dT = self._circular_diff(B_T, A_T)
        q = self.exp2_negative(gap)
        sine, cosine = self.sincos_masked(dT)
        radius_plus = torch.hypot(1.0 + q * cosine, q * sine)
        radius_minus = torch.hypot(1.0 - q * cosine, q * sine)
        aligned = 1.0 + q
        source_error = aligned * (_LOG_QUANTIZATION_RELATIVE_ERROR + _PAIR_PHASE_QUANTIZATION)
        plus_reliable, plus_error, plus_cancelled = self._pair_output_flags(radius_plus, aligned, source_error)
        minus_reliable, minus_error, minus_cancelled = self._pair_output_flags(radius_minus, aligned, source_error)
        stable = self._near_cancellation(gap, dT, 1, precision) | self._near_cancellation(
            gap, dT, -1, precision
        )
        tick = torch.full_like(plus_error, _PAIR_PHASE_QUANTIZATION)
        del lu, tu, lt, tt
        true_flag = torch.ones_like(plus_reliable)
        false_flag = torch.zeros_like(plus_reliable)
        plus_reliable = torch.where(nan, false_flag, torch.where(sentinel_zero, true_flag, plus_reliable))
        minus_reliable = torch.where(nan, false_flag, torch.where(sentinel_zero, true_flag, minus_reliable))
        nan_value = torch.full_like(plus_error, math.nan)
        plus_error = torch.where(nan, nan_value, torch.where(sentinel_zero, tick, plus_error))
        minus_error = torch.where(nan, nan_value, torch.where(sentinel_zero, tick, minus_error))
        plus_cancelled = torch.where(nan | sentinel_zero, false_flag, plus_cancelled)
        minus_cancelled = torch.where(nan | sentinel_zero, false_flag, minus_cancelled)
        stable = torch.where(nan | sentinel_zero, false_flag, stable)
        diagnostics = {
            "phase_reliable": plus_reliable & minus_reliable,
            "phase_error_bound": torch.maximum(plus_error, minus_error),
            "near_cancellation": plus_cancelled | minus_cancelled,
            "plus_phase_reliable": plus_reliable,
            "minus_phase_reliable": minus_reliable,
            "plus_phase_error_bound": plus_error,
            "minus_phase_error_bound": minus_error,
            "stable_branch": stable,
        }
        return plus, minus, diagnostics

    # -- radix-2 transforms on word planes -------------------------------
    def spf_fft_radix2(
        self,
        words: Any,
        precision: Any = "balanced",
        *,
        counters: v02.PerformanceCounters | None = None,
    ) -> Any:
        """``v02_planes.spf_fft_radix2`` schedule on the last axis of a word tensor."""

        n = int(words.shape[-1])
        if n < 1 or (n & (n - 1)) != 0:
            raise ValueError(f"spf_fft_radix2 requires power-of-two N, got N={n}")
        if n > v02.SPF32_INTERNAL_PHASE_MOD:
            raise ValueError("spf_fft_radix2 requires N <= the internal phase accumulator modulus")
        shape = tuple(int(value) for value in words.shape)
        rows = words.reshape(-1, n)
        order = self.upload_codes(v02._bit_reverse_indices(n))
        working = rows.index_select(1, order).clone()
        block = 2
        while block <= n:
            first, second, twiddle = self._stage_indices(n, block)
            a = working.index_select(1, first)
            b = self.words_phase_shift(working.index_select(1, second), twiddle.reshape(1, -1))
            if counters is not None:
                counters.fft_butterflies += int(a.numel())
                counters.phase_ops += int(a.numel())
            y0, y1 = self.spf_pair_addsub_hybrid(a, b, precision, counters=counters)
            working = working.index_copy(1, first, y0).index_copy(1, second, y1)
            block *= 2
        return working.reshape(shape)

    def spf_ifft_radix2(
        self,
        words: Any,
        precision: Any = "balanced",
        *,
        counters: v02.PerformanceCounters | None = None,
    ) -> Any:
        """``v02_planes.spf_ifft_radix2``: conjugate, forward, conjugate, exact ``1/N`` shift."""

        n = int(words.shape[-1])
        if n < 1 or (n & (n - 1)) != 0:
            raise ValueError(f"spf_ifft_radix2 requires power-of-two N, got N={n}")
        transformed = self.words_conj(
            self.spf_fft_radix2(self.words_conj(words), precision, counters=counters)
        )
        log_shift = int(math.log2(n)) * _L_SCALE
        L, T, zero, nan = self.canonical_parts(transformed)
        scaled = self.pack_words(self.clamp_finite(L - log_shift), T)
        return self.torch.where(zero | nan, transformed, scaled)

    def spf_ifftn_radix2(
        self,
        words: Any,
        precision: Any = "balanced",
        *,
        axes: Any = None,
        counters: v02.PerformanceCounters | None = None,
    ) -> Any:
        """Separable inverse: last axis first, matching ``v02_planes.spf_ifftn_radix2``."""

        working = words
        ndim = int(working.dim())
        selected = tuple(reversed(range(ndim))) if axes is None else tuple(int(a) % ndim for a in axes)
        for axis in selected:
            moved = self.torch.movedim(working, axis, -1).contiguous()
            moved = self.spf_ifft_radix2(moved, precision, counters=counters)
            working = self.torch.movedim(moved, -1, axis)
        return working.contiguous()

    def spf_fftn_radix2(
        self,
        words: Any,
        precision: Any = "balanced",
        *,
        axes: Any = None,
        counters: v02.PerformanceCounters | None = None,
    ) -> Any:
        working = words
        ndim = int(working.dim())
        selected = tuple(reversed(range(ndim))) if axes is None else tuple(int(a) % ndim for a in axes)
        for axis in selected:
            moved = self.torch.movedim(working, axis, -1).contiguous()
            moved = self.spf_fft_radix2(moved, precision, counters=counters)
            working = self.torch.movedim(moved, -1, axis)
        return working.contiguous()

    def _stage_indices(self, n: int, block: int) -> tuple[Any, Any, Any]:
        first, second, twiddle = v02_planes_stage_indices(n, block)
        return self.upload_codes(first), self.upload_codes(second), self.upload_codes(twiddle)

    # -- §9 pivoted reductions -------------------------------------------
    def _rows(self, values: Any, axis: int | None) -> tuple[Any, tuple[int, ...]]:
        if axis is None:
            return values.reshape(1, -1), ()
        moved = self.torch.movedim(values, int(axis), -1)
        out_shape = tuple(int(value) for value in moved.shape[:-1])
        return moved.reshape(-1, int(moved.shape[-1])), out_shape

    def _fraction_bits(self, policy: Any, kept: Any, explicit: int | None, n: int) -> Any:
        """Per-row accumulator width; ``v02._fraction_bits`` depends only on the kept count."""

        table = np.asarray(
            [0] + [v02._fraction_bits(policy, k, explicit) for k in range(1, n + 1)], dtype=np.int64
        )
        return self.upload_codes(table)[self.torch.clamp(kept, 0, n)]

    def _mantissas(self, rows_L: Any, pivot: Any, scale: Any, keep: Any) -> Any:
        """``_mantissas``: ``rne(exp2((L - pivot)/256)·scale)`` on kept terms (ROM exp2)."""

        index = self.torch.where(keep, pivot.unsqueeze(1) - rows_L, self.torch.zeros_like(rows_L))
        factor = self.exp2_negative(index)
        mantissa = self.rne(factor * scale.unsqueeze(1))
        return self.torch.where(keep, mantissa, self.torch.zeros_like(mantissa))

    def _int64_safe(self, bits: Any, n: int) -> bool:
        """``_sum_rows``' own int64 guard: ``max(bits) + ceil(log2 n) < 62``."""

        if n == 0:
            return True
        return int(self.torch.max(bits)) + math.ceil(math.log2(n)) < 62

    def _neumaier_row_sum(self, values: Any) -> Any:
        """Compensated left-to-right row sum (the torch stand-in for ``math.fsum``)."""

        torch = self.torch
        total = torch.zeros(values.shape[0], dtype=torch.float64, device=values.device)
        compensation = torch.zeros_like(total)
        for index in range(int(values.shape[1])):
            term = values[:, index]
            candidate = total + term
            big = torch.abs(total) >= torch.abs(term)
            compensation = compensation + torch.where(
                big, (total - candidate) + term, (term - candidate) + total
            )
            total = candidate
        return total + compensation

    def _row_norms(self, rows_L: Any, pivot: Any, keep: Any) -> Any:
        """``_row_norms``: ``fsum(exp2((L - pivot)/256))`` per row over kept terms."""

        index = self.torch.where(keep, pivot.unsqueeze(1) - rows_L, self.torch.zeros_like(rows_L))
        magnitudes = self.torch.where(
            keep, self.exp2_negative(index), self.torch.zeros_like(self.exp2_negative(index))
        )
        return self._neumaier_row_sum(magnitudes)

    def _finalize_log(self, pivot: Any, accumulator: Any, scale: Any) -> Any:
        """``_finalize_log``: ``pivot + rne(256·log2(|acc|/scale))``, 0 where ``acc == 0``."""

        torch = self.torch
        nonzero = accumulator != 0
        magnitude = torch.abs(accumulator).to(torch.float64) / scale
        safe = torch.where(nonzero, magnitude, torch.ones_like(magnitude))
        raw = pivot + self.rne(float(_L_SCALE) * torch.log2(safe))
        return torch.where(nonzero, raw, torch.zeros_like(raw))

    def reduce_logpos(
        self,
        L: Any,
        *,
        axis: int | None = None,
        spec: Any = None,
        F: int | None = None,
        fraction_bits: int | None = None,
        error_budget: float | None = None,
    ) -> TorchPlaneReduction:
        """``v02_planes.reduce_logpos`` on a device code plane (unpruned pivot branch)."""

        torch = self.torch
        policy = v02._coerce_spec(spec, error_budget)
        explicit = F if F is not None else fraction_bits
        if bool(policy.allow_pruning and (policy.error_budget_abs > 0.0 or policy.error_budget_rel > 0.0)):
            raise NotImplementedError(
                "the torch reducer implements the unpruned pivot branch; pruning stays on the CPU reducer"
            )
        rows, out_shape = self._rows(L, axis)
        count, n = int(rows.shape[0]), int(rows.shape[1])
        has_nan = torch.any(rows == _NAN_L, dim=1)
        keep = (rows != _L_ZERO) & (rows != _NAN_L)
        kept = torch.sum(keep.to(torch.int64), dim=1)
        if n:
            pivot = torch.max(torch.where(keep, rows, self._full(rows, _L_ZERO)), dim=1).values
        else:
            pivot = torch.full((count,), _L_ZERO, dtype=torch.int64, device=rows.device)
        bits = self._fraction_bits(policy, kept, explicit, n)
        bits = torch.where(has_nan, torch.zeros_like(bits), bits)
        scale = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=rows.device), bits.to(torch.float64))
        mantissa = self._mantissas(rows, pivot, scale, keep)
        if not self._int64_safe(bits, n):
            raise OverflowError(
                "accumulator width exceeds int64 on the device path; the CPU reducer's big-int branch applies"
            )
        accumulator = torch.sum(mantissa, dim=1)
        output_raw = self._finalize_log(pivot, accumulator, scale)
        nonzero = (accumulator != 0) & (kept > 0)
        output_L = torch.where(nonzero, self.clamp_finite(output_raw), self._full(output_raw, _L_ZERO))
        output_L = torch.where(has_nan, self._full(output_L, _NAN_L), output_L)
        saturated = nonzero & ((output_raw < _L_MIN) | (output_raw > _L_MAX))
        norm = self._row_norms(rows, pivot, keep)
        linear_scale = self.exp2_code(pivot)
        values = torch.where(nonzero, self.exp2_code(output_L), torch.zeros_like(linear_scale))
        kept_sum = linear_scale * norm
        mantissa_error = linear_scale * kept.to(torch.float64) * 0.5 / scale
        quantization = torch.where(
            torch.isfinite(values),
            torch.abs(values) * _LOG_QUANTIZATION_RELATIVE_ERROR,
            torch.abs(values - kept_sum),
        )
        absolute = mantissa_error + quantization
        full_sum = linear_scale * norm
        bad = saturated | ~torch.isfinite(absolute) | ~torch.isfinite(full_sum)
        infinity = torch.full_like(absolute, math.inf)
        ones = torch.ones_like(absolute)
        zeros = torch.zeros_like(absolute)
        absolute = torch.where(bad, infinity, absolute)
        relative = torch.where(
            bad, ones, torch.clamp(absolute / torch.clamp(full_sum, min=1e-300), max=1.0)
        )
        absolute = torch.where(nonzero, absolute, zeros)
        relative = torch.where(nonzero, relative, torch.where(kept > 0, ones, zeros))
        nan_value = torch.full_like(absolute, math.nan)
        absolute = torch.where(has_nan, nan_value, absolute)
        relative = torch.where(has_nan, nan_value, relative)
        live = torch.where(has_nan, torch.zeros_like(kept), kept)
        counters = v02.PerformanceCounters(
            terms_seen=int(rows.numel()),
            terms_retained=int(torch.sum(kept)),
            bytes_read=int(rows.numel()) * 2,
            bytes_written=2 * count,
            aligned_mantissas=int(torch.sum(live)),
            pivot_reductions=int(torch.count_nonzero(live)),
            log_ops=int(torch.count_nonzero(live)),
            exp2_ops=int(torch.sum(live)),
            values_materialized=int(torch.count_nonzero(has_nan)),
        )
        any_kept = bool(torch.any(kept > 0))
        return TorchPlaneReduction(
            value=output_L.reshape(out_shape),
            value_sign=torch.zeros_like(output_L).reshape(out_shape),
            value_kind="LOG_POS",
            method="pivot_logpos" if any_kept else "zero",
            absolute_error_bound=absolute.reshape(out_shape),
            relative_error_bound=relative.reshape(out_shape),
            phase_error_bound=torch.zeros_like(absolute).reshape(out_shape),
            phase_reliable=(~has_nan).reshape(out_shape),
            retained_terms=kept.reshape(out_shape),
            dropped_terms=torch.zeros_like(kept).reshape(out_shape),
            accumulator_fraction_bits=bits.reshape(out_shape),
            saturated=saturated.reshape(out_shape),
            counters=counters,
        )

    def reduce_logsigned(
        self,
        plane: tuple[Any, Any],
        *,
        axis: int | None = None,
        spec: Any = None,
        F: int | None = None,
        fraction_bits: int | None = None,
        error_budget: float | None = None,
    ) -> TorchPlaneReduction:
        """``v02_planes.reduce_logsigned`` on a device ``(L, sign)`` plane."""

        torch = self.torch
        policy = v02._coerce_spec(spec, error_budget)
        explicit = F if F is not None else fraction_bits
        codes, signs_plane = plane
        rows, out_shape = self._rows(codes, axis)
        signs, _ = self._rows(signs_plane, axis)
        count, n = int(rows.shape[0]), int(rows.shape[1])
        has_nan = torch.any(rows == _NAN_L, dim=1)
        keep = (signs != 0) & (rows != _NAN_L)
        kept = torch.sum(keep.to(torch.int64), dim=1)
        if n:
            pivot = torch.max(torch.where(keep, rows, self._full(rows, _L_ZERO)), dim=1).values
        else:
            pivot = torch.full((count,), _L_ZERO, dtype=torch.int64, device=rows.device)
        bits = self._fraction_bits(policy, kept, explicit, n)
        bits = torch.where(has_nan, torch.zeros_like(bits), bits)
        scale = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=rows.device), bits.to(torch.float64))
        mantissa = self._mantissas(rows, pivot, scale, keep)
        if not self._int64_safe(bits, n):
            raise OverflowError(
                "accumulator width exceeds int64 on the device path; the CPU reducer's big-int branch applies"
            )
        zero_mantissa = torch.zeros_like(mantissa)
        positive = torch.sum(torch.where(signs > 0, mantissa, zero_mantissa), dim=1)
        negative = torch.sum(torch.where(signs < 0, mantissa, zero_mantissa), dim=1)
        accumulator = positive - negative
        output_raw = self._finalize_log(pivot, accumulator, scale)
        nonzero = (accumulator != 0) & (kept > 0)
        output_sign = torch.sign(accumulator)
        output_L = torch.where(nonzero, self.clamp_finite(output_raw), self._full(output_raw, _L_ZERO))
        output_L = torch.where(has_nan, self._full(output_L, _NAN_L), output_L)
        output_sign = torch.where(has_nan | ~nonzero, torch.zeros_like(output_sign), output_sign)
        saturated = nonzero & ((output_raw < _L_MIN) | (output_raw > _L_MAX))
        linear_scale = self.exp2_code(pivot)
        rounding = linear_scale * kept.to(torch.float64) * 0.5 / scale
        magnitude = linear_scale * torch.abs(accumulator).to(torch.float64) / scale
        quantization = magnitude * _LOG_QUANTIZATION_RELATIVE_ERROR
        absolute = rounding + quantization
        lower = magnitude - rounding
        infinity = torch.full_like(absolute, math.inf)
        relative = torch.where(lower <= 0.0, infinity, absolute / torch.where(lower <= 0.0, infinity, lower))
        absolute = torch.where(saturated, infinity, absolute)
        relative = torch.where(saturated, infinity, relative)
        zeros = torch.zeros_like(absolute)
        absolute = torch.where(nonzero, absolute, torch.where(kept > 0, rounding, zeros))
        relative = torch.where(nonzero, relative, torch.where(kept > 0, infinity, zeros))
        nan_value = torch.full_like(absolute, math.nan)
        absolute = torch.where(has_nan, nan_value, absolute)
        relative = torch.where(has_nan, nan_value, relative)
        live = torch.where(has_nan, torch.zeros_like(kept), kept)
        counters = v02.PerformanceCounters(
            terms_seen=int(rows.numel()),
            terms_retained=int(torch.sum(kept)),
            bytes_read=int(rows.numel()) * 3,
            bytes_written=3 * count,
            aligned_mantissas=int(torch.sum(live)),
            pivot_reductions=int(torch.count_nonzero(live)),
            log_ops=int(torch.count_nonzero(live)),
            exp2_ops=int(torch.sum(live)),
        )
        # §9.3 cancellation threshold: certify the sign only when the surviving
        # accumulator is at least 2**-12 of the aligned absolute sum.
        aligned_absolute = (positive + negative).to(torch.float64)
        cancellation_ratio = torch.abs(accumulator).to(torch.float64) / torch.clamp(aligned_absolute, min=1.0)
        reliable = (cancellation_ratio >= 2.0**-12) & (lower > 0.0) & nonzero
        reliable = torch.where(kept == 0, torch.ones_like(reliable), reliable) & ~has_nan
        return TorchPlaneReduction(
            value=output_L.reshape(out_shape),
            value_sign=output_sign.reshape(out_shape),
            value_kind="LOG_SIGNED",
            method="pivot_logsigned" if bool(torch.any(kept > 0)) else "zero",
            absolute_error_bound=absolute.reshape(out_shape),
            relative_error_bound=relative.reshape(out_shape),
            phase_error_bound=torch.zeros_like(absolute).reshape(out_shape),
            phase_reliable=reliable.reshape(out_shape),
            retained_terms=kept.reshape(out_shape),
            dropped_terms=torch.zeros_like(kept).reshape(out_shape),
            accumulator_fraction_bits=bits.reshape(out_shape),
            saturated=saturated.reshape(out_shape),
            counters=counters,
        )

    def _exact_hypot(self, X: Any, Y: Any) -> Any:  # §9.4 names the accumulators X, Y
        """§9.4's ``hypot`` on the integer pivot accumulators, correctly rounded.

        AlgRef §21.3 lists ``hypot`` as a float functional unit, and the NumPy
        oracle evaluates it as CPython's scalar ``math.hypot``.  Neither torch
        primitive reproduces that on both devices — measured 2026-09-18 on beast
        (torch 2.12+cu130, RTX 4080, glibc 2.39), against ``math.hypot`` on the
        ``test_T36_reduce_complex_identical`` corpus and on 200 000 random
        integer pairs:

        ===========  ==============================  ========================
        device       ``torch.hypot(X, Y)``           ``sqrt(X² + Y²)``
        ===========  ==============================  ========================
        CUDA         5 / 97 and 10 / 128 rows, 1 ulp  0 / 512 on all 4 shapes
        CPU          0 / 200 000                      2 / 512 rows, 1 ulp
        ===========  ==============================  ========================

        So each device gets the routine that *is* correctly rounded on it.  The
        CUDA form is exact by construction rather than by luck: ``X`` and ``Y``
        are integers, so while ``X² + Y²`` is exactly representable in float64
        (``|X|, |Y| <= 2^26``, which covers everything the §9.2 accumulator-width
        guard admits — measured max on the corpus 4.4e9 = 2^32) IEEE ``sqrt`` is
        exact-then-rounded and therefore equals ``math.hypot`` bit for bit.
        Above that bound CUDA falls back to its ``hypot``, no worse than before.

        None of this ever moved a *word* — the finalizer's ``rne`` absorbed it —
        only the §47.1 certificate fields, which is what the oracle test saw.
        """

        torch = self.torch
        xf, yf = X.to(torch.float64), Y.to(torch.float64)
        if not self.is_cuda:
            return torch.hypot(xf, yf)
        exact = (torch.abs(X) <= _EXACT_SQUARE_MAX) & (torch.abs(Y) <= _EXACT_SQUARE_MAX)
        zero = torch.zeros_like(X)
        # Masked before squaring so the int64 product cannot overflow on the
        # rows that take the fallback.
        safe_x = torch.where(exact, X, zero)
        safe_y = torch.where(exact, Y, zero)
        squares = (safe_x * safe_x + safe_y * safe_y).to(torch.float64)
        return torch.where(exact, torch.sqrt(squares), torch.hypot(xf, yf))

    def reduce_complex(
        self,
        words: Any,
        *,
        axis: int | None = None,
        spec: Any = None,
        F: int | None = None,
        fraction_bits: int | None = None,
        error_budget: float | None = None,
    ) -> TorchPlaneReduction:
        """``v02_planes.reduce_complex`` on a device word plane (§36.5 complex reducer)."""

        torch = self.torch
        policy = v02._coerce_spec(spec, error_budget)
        explicit = F if F is not None else fraction_bits
        raw = self.canonical_words(words)
        rows, out_shape = self._rows(raw, axis)
        count, n = int(rows.shape[0]), int(rows.shape[1])
        L, T, zero, nan = self.canonical_parts(rows)
        has_nan = torch.any(nan, dim=1)
        keep = ~zero & ~nan
        kept = torch.sum(keep.to(torch.int64), dim=1)
        if n:
            pivot = torch.max(torch.where(keep, L, self._full(L, _L_ZERO)), dim=1).values
        else:
            pivot = torch.full((count,), _L_ZERO, dtype=torch.int64, device=rows.device)
        bits = self._fraction_bits(policy, kept, explicit, n)
        bits = torch.where(has_nan, torch.zeros_like(bits), bits)
        scale = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=rows.device), bits.to(torch.float64))
        mantissa = self._mantissas(L, pivot, scale, keep)
        if not self._int64_safe(bits, n):
            raise OverflowError(
                "accumulator width exceeds int64 on the device path; the CPU reducer's big-int branch applies"
            )
        sine, cosine = self.sincos_masked(T)
        zeros_int = torch.zeros_like(mantissa)
        X = torch.sum(torch.where(keep, self.rne(mantissa.to(torch.float64) * cosine), zeros_int), dim=1)
        Y = torch.sum(torch.where(keep, self.rne(mantissa.to(torch.float64) * sine), zeros_int), dim=1)
        radius = self._exact_hypot(X, Y)
        nonzero = (radius != 0.0) & (kept > 0)
        safe_radius = torch.where(nonzero, radius, torch.ones_like(radius))
        output_raw = pivot + self.rne(float(_L_SCALE) * torch.log2(safe_radius / scale))
        output_raw = torch.where(nonzero, output_raw, torch.zeros_like(output_raw))
        output_T = self.rne(torch.atan2(Y.to(torch.float64), X.to(torch.float64)) * _RAD_TO_PHASE) & _PHASE_MASK
        output_T = torch.where(nonzero, output_T, torch.zeros_like(output_T))
        output_L = torch.where(nonzero, self.clamp_finite(output_raw), self._full(output_raw, _L_ZERO))
        out_words = self.pack_words(output_L, output_T)
        out_words = torch.where(has_nan, self._full(out_words, _WORD_NAN), out_words)
        saturated = nonzero & ((output_raw < _L_MIN) | (output_raw > _L_MAX))
        linear_scale = self.exp2_code(pivot)
        coordinate_error = linear_scale * kept.to(torch.float64) * (1.0 + math.sqrt(2.0)) * 0.5 / scale
        phase_quantization = math.pi / _PHASE_MOD
        magnitude = linear_scale * radius / scale
        quantization = magnitude * (_LOG_QUANTIZATION_RELATIVE_ERROR + phase_quantization)
        absolute = coordinate_error + quantization
        lower = magnitude - coordinate_error
        absolute_sum = torch.sum(torch.abs(mantissa), dim=1).to(torch.float64)
        infinity = torch.full_like(absolute, math.inf)
        relative = torch.where(lower <= 0.0, infinity, absolute / torch.where(lower <= 0.0, infinity, lower))
        coordinate_phase = torch.where(
            coordinate_error >= magnitude,
            torch.full_like(magnitude, math.pi),
            torch.asin(torch.clamp(coordinate_error / torch.clamp(magnitude, min=1e-300), max=1.0)),
        )
        phase_ratio = radius / torch.clamp(absolute_sum, min=1.0)
        phase_error = phase_quantization + coordinate_phase
        reliable = (phase_ratio >= 2.0**-12) & torch.isfinite(phase_error)
        reliable = reliable & (phase_error <= v02.PHASE_RELIABILITY_MAX_ERROR_RAD)
        absolute = torch.where(saturated, infinity, absolute)
        relative = torch.where(saturated, infinity, relative)
        zeros = torch.zeros_like(absolute)
        absolute = torch.where(nonzero, absolute, torch.where(kept > 0, coordinate_error, zeros))
        relative = torch.where(nonzero, relative, torch.where(kept > 0, infinity, zeros))
        phase_error = torch.where(
            nonzero, phase_error, torch.where(kept > 0, torch.full_like(phase_error, math.pi), zeros)
        )
        reliable = torch.where(kept == 0, torch.ones_like(reliable), reliable & nonzero)
        nan_value = torch.full_like(absolute, math.nan)
        absolute = torch.where(has_nan, nan_value, absolute)
        relative = torch.where(has_nan, nan_value, relative)
        phase_error = torch.where(has_nan, nan_value, phase_error)
        reliable = reliable & ~has_nan
        live = torch.where(has_nan, torch.zeros_like(kept), kept)
        counters = v02.PerformanceCounters(
            terms_seen=int(rows.numel()),
            terms_retained=int(torch.sum(kept)),
            bytes_read=int(rows.numel()) * 4,
            bytes_written=4 * count,
            aligned_mantissas=int(torch.sum(live)),
            pivot_reductions=int(torch.count_nonzero(live)),
            complex_finalizations=int(torch.count_nonzero(live)),
            phase_ops=int(torch.sum(live)),
            log_ops=int(torch.count_nonzero(live)),
            atan2_ops=int(torch.count_nonzero(live)),
            exp2_ops=int(torch.sum(live)),
            trig_ops=int(torch.sum(live)),
        )
        return TorchPlaneReduction(
            value=out_words.reshape(out_shape),
            value_sign=torch.zeros_like(out_words).reshape(out_shape),
            value_kind="SPF_COMPLEX",
            method="pivot_complex" if bool(torch.any(kept > 0)) else "zero",
            absolute_error_bound=absolute.reshape(out_shape),
            relative_error_bound=relative.reshape(out_shape),
            phase_error_bound=phase_error.reshape(out_shape),
            phase_reliable=reliable.reshape(out_shape),
            retained_terms=kept.reshape(out_shape),
            dropped_terms=torch.zeros_like(kept).reshape(out_shape),
            accumulator_fraction_bits=bits.reshape(out_shape),
            saturated=saturated.reshape(out_shape),
            counters=counters,
        )

    # -- transport (§16, §27.7) ------------------------------------------
    def optical_depth_to_transmittance(self, tau_L: Any) -> Any:
        """``transport_terminal.optical_depth_to_transmittance``: one ROM lookup."""

        return self.rom["transmittance"][tau_L & 0xFFFF]

    def prefix_logpos_product(self, codes: Any) -> tuple[Any, Any] | None:
        """Exclusive prefix of ``logpos_mul`` over axis 0 as one cumulative sum.

        ``logpos_mul`` is a saturating integer add with sticky zero/NaN, so the
        whole prefix product is a ``cumsum`` **provided** no partial sum left the
        finite range (that is the only place the per-step clamp could bite).  The
        guard is checked on the device; ``None`` means the caller must run the
        sequential loop instead.  One ``cumsum`` replaces ``n`` kernels.
        """

        torch = self.torch
        is_zero = codes == _L_ZERO
        is_nan = codes == _NAN_L
        clean = torch.where(is_zero | is_nan, torch.zeros_like(codes), codes)
        inclusive = torch.cumsum(clean, dim=0)
        if bool(torch.any((inclusive < _L_MIN) | (inclusive > _L_MAX))):
            return None
        zero_hits = torch.cumsum(is_zero.to(torch.int64), dim=0)
        nan_hits = torch.cumsum(is_nan.to(torch.int64), dim=0)
        zeros_row = torch.zeros_like(clean[:1])
        exclusive = torch.cat((zeros_row, inclusive[:-1]), dim=0)
        zero_before = torch.cat((zeros_row, zero_hits[:-1]), dim=0) > 0
        nan_before = torch.cat((zeros_row, nan_hits[:-1]), dim=0) > 0
        prefix = torch.where(
            nan_before,
            self._full(exclusive, _NAN_L),
            torch.where(zero_before, self._full(exclusive, _L_ZERO), exclusive),
        )
        total = torch.where(
            nan_hits[-1] > 0,
            self._full(inclusive[-1], _NAN_L),
            torch.where(zero_hits[-1] > 0, self._full(inclusive[-1], _L_ZERO), inclusive[-1]),
        )
        return prefix, total

    def transport_compose_front_to_back(
        self, transmittance_L: Any, contribution: tuple[Any, Any]
    ) -> tuple[Any, tuple[Any, Any]]:
        """``transport_terminal.compose_front_to_back`` over a leading segment axis.

        The transmittance prefix is one cumulative sum and every segment's
        ``c_i · T_{<i}`` product is one batched kernel; only the ordered
        ``logadd_signed`` accumulation stays sequential, because the v0.2 signed
        adder is not associative and the words must match the CPU order.
        """

        torch = self.torch
        n = int(transmittance_L.shape[0])
        c_L, c_sign = contribution
        channels = int(c_L.dim()) > int(transmittance_L.dim())
        total_c = (
            torch.full(tuple(int(v) for v in c_L.shape[1:]), _L_ZERO, dtype=torch.int64, device=self.device),
            torch.zeros(tuple(int(v) for v in c_L.shape[1:]), dtype=torch.int64, device=self.device),
        )
        fused = self.prefix_logpos_product(transmittance_L) if n else None
        if fused is None:
            total_t = torch.zeros(
                tuple(int(v) for v in transmittance_L.shape[1:]), dtype=torch.int64, device=self.device
            )
            for index in range(n):
                prefix = self.as_signed(total_t.unsqueeze(-1) if channels else total_t)
                total_c = self.signed_add(total_c, self.signed_mul((c_L[index], c_sign[index]), prefix))
                total_t = self.logpos_mul(total_t, transmittance_L[index])
            return total_t, total_c
        prefix_codes, total_t = fused
        prefix_all = self.as_signed(prefix_codes.unsqueeze(-1) if channels else prefix_codes)
        product = self.signed_mul((c_L, c_sign), prefix_all)
        for index in range(n):
            total_c = self.signed_add(total_c, (product[0][index], product[1][index]))
        return total_t, total_c

    def transport_scan(
        self, transmittance_L: Any, contribution: tuple[Any, Any]
    ) -> tuple[Any, tuple[Any, Any]]:
        """Ordered scan: the per-step prefix ``(T_i, C_i)`` stacked on the leading axis."""

        torch = self.torch
        n = int(transmittance_L.shape[0])
        c_L, c_sign = contribution
        channels = int(c_L.dim()) > int(transmittance_L.dim())
        t_steps = []
        c_L_steps = []
        c_sign_steps = []
        total_t = torch.zeros(
            tuple(int(v) for v in transmittance_L.shape[1:]), dtype=torch.int64, device=self.device
        )
        total_c = (
            torch.full(tuple(int(v) for v in c_L.shape[1:]), _L_ZERO, dtype=torch.int64, device=self.device),
            torch.zeros(tuple(int(v) for v in c_L.shape[1:]), dtype=torch.int64, device=self.device),
        )
        fused = self.prefix_logpos_product(transmittance_L) if n else None
        product: tuple[Any, Any] | None = None
        if fused is not None:
            prefix_codes = fused[0]
            prefix_all = self.as_signed(prefix_codes.unsqueeze(-1) if channels else prefix_codes)
            product = self.signed_mul((c_L, c_sign), prefix_all)
        for index in range(n):
            if product is None:
                prefix = self.as_signed(total_t.unsqueeze(-1) if channels else total_t)
                step = self.signed_mul((c_L[index], c_sign[index]), prefix)
            else:
                step = (product[0][index], product[1][index])
            total_c = self.signed_add(total_c, step)
            total_t = self.logpos_mul(total_t, transmittance_L[index])
            t_steps.append(total_t)
            c_L_steps.append(total_c[0])
            c_sign_steps.append(total_c[1])
        if not t_steps:
            return total_t.unsqueeze(0)[:0], (total_c[0].unsqueeze(0)[:0], total_c[1].unsqueeze(0)[:0])
        return torch.stack(t_steps), (torch.stack(c_L_steps), torch.stack(c_sign_steps))

    def display_values(self, L: Any, sign: Any | None = None) -> Any:
        """Float64 display projection through the magnitude ROM (the one output boundary)."""

        magnitude = self.magnitude(L)
        if sign is None:
            return magnitude
        return magnitude * sign.to(self.torch.float64)

    # -- §13.3 jet emission ----------------------------------------------
    def jet_forward_differences(
        self,
        coefficients: Any,
        shape: tuple[int, int],
        origin: tuple[float, float] = (0.0, 0.0),
        *,
        modulo: int | None = None,
        counters: v02.PerformanceCounters | None = None,
    ) -> Any:
        """``v02.jet_forward_differences`` for a batch of tiles, same add order.

        ``coefficients`` is a float64 tensor ``(..., 6)`` holding
        ``(value, grad_x, grad_y, hess_xx, hess_xy, hess_yy)``; the result is
        ``(..., h, w)`` in the jet's widened units.
        """

        torch = self.torch
        height, width = int(shape[0]), int(shape[1])
        if height < 0 or width < 0:
            raise ValueError("jet materialization shape must be non-negative")
        x0, y0 = float(origin[0]), float(origin[1])
        value_c = coefficients[..., 0]
        grad_x = coefficients[..., 1]
        grad_y = coefficients[..., 2]
        hess_xx = coefficients[..., 3]
        hess_xy = coefficients[..., 4]
        hess_yy = coefficients[..., 5]

        def evaluate(x: float, y: float) -> Any:
            return (
                value_c
                + grad_x * x
                + grad_y * y
                + 0.5 * hess_xx * x * x
                + hess_xy * x * y
                + 0.5 * hess_yy * y * y
            )

        row_value = evaluate(x0, y0)
        row_dx = evaluate(x0 + 1.0, y0) - row_value
        row_dy = evaluate(x0, y0 + 1.0) - row_value
        if modulo is not None:
            row_value = torch.remainder(row_value, float(modulo))
            row_dx = torch.remainder(row_dx, float(modulo))
            row_dy = torch.remainder(row_dy, float(modulo))
        if counters is not None:
            counters.jet_samples += height * width
            counters.values_materialized += height * width
        if height == 0 or width == 0:
            return torch.zeros(
                (*(int(v) for v in value_c.shape), height, width), dtype=torch.float64, device=self.device
            )
        # Two loops of length h and w, not h·w: the row recurrence is advanced
        # once per row for every tile at once, then the column recurrence once
        # per column for every (tile, row) at once.  The float64 add order per
        # output is identical to the scalar loop, so the words are identical;
        # this is what keeps the launch count at h+w on a launch-bound device.
        row_values = []
        row_dxs = []
        for _ in range(height):
            row_values.append(row_value)
            row_dxs.append(row_dx)
            row_value = row_value + row_dy
            row_dy = row_dy + hess_yy
            row_dx = row_dx + hess_xy
            if modulo is not None:
                row_value = torch.remainder(row_value, float(modulo))
                row_dy = torch.remainder(row_dy, float(modulo))
                row_dx = torch.remainder(row_dx, float(modulo))
        value = torch.stack(row_values, dim=-1)
        dx = torch.stack(row_dxs, dim=-1)
        step = hess_xx.unsqueeze(-1)
        columns = []
        for _ in range(width):
            columns.append(value)
            value = value + dx
            dx = dx + step
            if modulo is not None:
                value = torch.remainder(value, float(modulo))
                dx = torch.remainder(dx, float(modulo))
        return torch.stack(columns, dim=-1)

    def jet_emit_log_words(self, wide: Any, guard_L: int, sign: int = 1) -> Any:
        """``v02.jet_emit_spf32`` for a log-kind jet: ``rne(wide / 2**guard_L)`` into words."""

        codes = self.rne(wide / float(1 << int(guard_L)))
        phase = self.torch.full_like(codes, _PHASE_HALF if int(sign) < 0 else 0)
        return self.pack_words(self.canonical_logpos(self.clamp_finite(codes)), phase)

    def jet_emit_log_codes(self, wide: Any, guard_L: int) -> Any:
        """The LOG_POS code plane of a widened log jet (what the splat compositor uses)."""

        codes = self.rne(wide / float(1 << int(guard_L)))
        return self.canonical_logpos(self.torch.clamp(codes, max=0))

    # -- filter-chain suffix ---------------------------------------------
    def filter_chain_suffix(
        self,
        spectrum: Any,
        transfer: Any,
        *,
        precision: Any = "balanced",
        counters: v02.PerformanceCounters | None = None,
        axes: Any = None,
    ) -> Any:
        """§26 suffix on words: pointwise spectral product, then the SPF inverse transform.

        Both operands and the result stay packed SPF32 words on the device; no
        dense complex buffer is created (AGENTS.md rule 2).
        """

        product = self.words_mul(spectrum, transfer)
        if counters is not None:
            counters.phase_ops += int(product.numel())
            counters.log_ops += int(product.numel())
        return self.spf_ifftn_radix2(product, precision, axes=axes, counters=counters)


def v02_planes_stage_indices(n: int, block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``v02_planes._stage_indices`` (host side; twiddles are integer phase codes)."""

    from spf.tensor.v02_planes import _stage_indices

    return _stage_indices(int(n), int(block))


_SHARED: dict[str, TorchPlanes] = {}


def planes(device: str | None = None, *, compile_fused: bool | None = None) -> TorchPlanes:
    """A cached :class:`TorchPlanes` for ``device`` (ROMs are uploaded once)."""

    resolved = str(select_device(device))
    key = f"{resolved}|{bool(compile_fused)}"
    cached = _SHARED.get(key)
    if cached is not None:
        return cached
    ops = TorchPlanes(resolved, compile_fused=compile_fused)
    _SHARED[key] = ops
    return ops


__all__ = [
    "CONSUMER",
    "ORACLE",
    "DeviceRoms",
    "HostRoms",
    "TorchPlaneReduction",
    "TorchPlanes",
    "availability",
    "available",
    "cuda_available",
    "device_info",
    "device_roms",
    "host_roms",
    "peak_gpu_memory_bytes",
    "planes",
    "rom_bytes",
    "select_device",
]
