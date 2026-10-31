"""Pluggable row-wise FFT backend for the OFDM/sOFDM receive path.

By default the receivers call :func:`fft_rows`, which is plain ``np.fft.fft``
over ``axis=1`` — identical to the historical hardcoded call. Wrapping a block
of work in :func:`overlay_fft_active` swaps that backend for the KV260
``spf_hotpath_fft32_bittrick_pl`` FFT64 compute-unit (dispatched over the
persistent ``pl_server`` TCP session), so the *same* OFDM code runs the FFT
stage either in software (numpy) or on the PL overlay. This is the
with/without-overlay toggle for the board eval; nothing else in the datapath
changes.

The overlay FFT is an SPF-domain transform — approximate vs the FP64
``np.fft.fft`` (the accuracy delta is exactly what the eval measures). The
transform size is fixed by the FFT compute-unit resident in the loaded overlay
(64-point for the ``*_fft*_pl`` overlays, 256-point for
``spf32_fft256_fb7_bittrick``); :func:`overlay_fft_active` validates that the
requested ``n_fft`` matches the loaded overlay's CU before any dispatch.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from spf.backends.kria.pl_client import OP_FFT64, PersistentPLSession
from spf.tensor import decode_words, encode_words

Complex = npt.NDArray[np.complex128]
FFTRows = Callable[[Complex], Complex]

# FFT64 compute-unit size — the default (and historical) overlay CU is 64 points.
FFT64_N = 64

# Host-side FFT-overlay capability map: the point-count of the FFT compute-unit
# resident in each composite overlay this client can drive. This client encodes
# SPF-32 (one uint32 per complex value) and sends exactly ``n_fft`` words per
# transform, so it can only feed 1-lane (SPF-32) FFT CUs whose point-count it
# knows. The board (``pl_server._FB7_OVERLAY_FFT64_ALIASES`` /
# ``_fft64_words_for_kernel``) is the source of truth for word packing; this map
# is the host's matching declaration. There is no wire handshake exposing the
# resident CU size, so without this guard a size mismatch (e.g. a 256-subcarrier
# fixture run against the 64-pt overlay) would pass the server's ``n % words``
# divisibility check and silently run several wrong-size sub-FFTs. Adding a
# metadata handshake so the client can *query* the resident size is the deeper
# fix; until then, refuse loudly here. Keep in sync with the alias table.
_OVERLAY_FFT_POINTS: dict[str, int] = {
    "spf_hotpath_fft32_bittrick_pl": 64,   # SPF-32 64-pt bit-trick (default)
    "spf32_fb7_core_pl":             64,   # SPF-32 64-pt FB7 composite
    "spf32_fb7_fields_core_pl":      64,   # SPF-32 64-pt FB7 (field mode)
    "spf32_fft256_fb7_bittrick":     256,  # SPF-32 256-pt bit-trick
}

# Overlays whose FFT CU this client cannot pack for, mapped to the reason. Listed
# explicitly so the failure names the cause rather than "unknown overlay".
_OVERLAY_FFT_UNSUPPORTED: dict[str, str] = {
    "spf_hotpath_fft_pl": (
        "SPF-16 lane-packed FFT64 CU (2 values/uint32); this host client only "
        "emits SPF-32 words (1 value/uint32). Drive it with an SPF-16 packing path."
    ),
}


def _validate_overlay_fft(overlay: str | None, n_fft: int) -> None:
    """Refuse an overlay/``n_fft`` pairing this client cannot dispatch correctly.

    Raises ``NotImplementedError`` for a known-but-unsupported (non-SPF-32)
    overlay, and ``ValueError`` for an unknown overlay or a point-count that does
    not match the resident CU. This closes the silent-wrong-result path where a
    mismatched size slips through the server's divisibility-only guard.
    """
    if overlay in _OVERLAY_FFT_UNSUPPORTED:
        raise NotImplementedError(
            f"overlay {overlay!r}: {_OVERLAY_FFT_UNSUPPORTED[overlay]}"
        )
    points = _OVERLAY_FFT_POINTS.get(overlay) if overlay is not None else None
    if points is None:
        raise ValueError(
            f"overlay {overlay!r} is not a known host-encodable FFT overlay; "
            f"cannot verify n_fft={n_fft} matches the resident CU. "
            f"Known SPF-32 FFT overlays: {sorted(_OVERLAY_FFT_POINTS)}"
        )
    if points != n_fft:
        raise ValueError(
            f"n_fft={n_fft} does not match the {points}-point FFT CU of overlay "
            f"{overlay!r}; load the matching overlay (e.g. spf32_fft256_fb7_bittrick "
            f"for 256-subcarrier fixtures)."
        )


# Thread-local override so a context manager on one thread cannot leak the
# overlay backend into another (tests, parallel evals).
_state = threading.local()


def _numpy_fft_rows(useful: Complex) -> Complex:
    return np.fft.fft(np.asarray(useful, dtype=np.complex128), axis=1)


def fft_rows(useful: Complex) -> Complex:
    """Row-wise FFT (``axis=1``). Uses the active backend, defaulting to numpy.

    The OFDM/sOFDM receivers call this in place of ``np.fft.fft(useful, axis=1)``
    so the FFT stage is swappable to the PL overlay without touching the datapath.
    """
    backend: FFTRows = getattr(_state, "fft_rows", _numpy_fft_rows)
    return backend(useful)


@dataclass(slots=True)
class OverlayFFTStats:
    """Telemetry accumulated while the overlay FFT backend is active."""

    fft64_dispatches: int = 0  # wire round-trips (one per block, not per transform)
    rows: int = 0              # FFT transforms performed (one per symbol row)
    kernel_wait_ms_total: float = 0.0
    overlay: str | None = None
    xclbin_sha256: str | None = None
    extra: dict[str, float] = field(default_factory=dict)


class OverlayFFT64:
    """Row-wise FFT backend that runs each N-point transform on the KV260 overlay.

    Encodes the whole ``(n_symbols, n_fft)`` block to SPF-32 words and dispatches
    it in a *single* ``OP_FFT64`` request: the resident CU transforms one N-point
    batch per ``n_fft`` words, and ``pl_server`` reshapes the flat payload into
    ``n_symbols`` independent transforms server-side, so one round-trip covers the
    whole block instead of one per row. ``n_fft`` selects the transform size and
    must match the FFT compute unit resident in the loaded overlay — validated
    against :data:`_OVERLAY_FFT_POINTS` at construction. (Class name retained for
    back-compat; it is size-parametric.)
    """

    def __init__(
        self,
        session: PersistentPLSession,
        stats: OverlayFFTStats | None = None,
        n_fft: int = FFT64_N,
    ) -> None:
        self._session = session
        self._n_fft = int(n_fft)
        _validate_overlay_fft(session.overlay, self._n_fft)
        self.stats = stats if stats is not None else OverlayFFTStats()
        self.stats.overlay = session.overlay
        self.stats.xclbin_sha256 = session.xclbin_sha256

    def __call__(self, useful: Complex) -> Complex:
        arr = np.asarray(useful, dtype=np.complex128)
        if arr.ndim != 2 or arr.shape[1] != self._n_fft:
            raise ValueError(
                f"overlay FFT requires (n_symbols, {self._n_fft}) complex rows, got {arr.shape}"
            )
        if arr.shape[0] == 0:
            return arr.copy()
        # One dispatch for the whole block: flat n_symbols*n_fft words → the
        # server reshapes into n_symbols transforms of n_fft words each.
        words = encode_words(arr.reshape(-1))
        t0 = time.perf_counter_ns()
        res_words = self._session.dispatch_unary(OP_FFT64, words)
        self.stats.kernel_wait_ms_total += (time.perf_counter_ns() - t0) / 1e6
        self.stats.fft64_dispatches += 1  # wire round-trips (rows tracks transforms)
        self.stats.rows += arr.shape[0]
        decoded = decode_words(np.ascontiguousarray(res_words, dtype=np.uint32))
        return decoded.reshape(arr.shape)


@dataclass(slots=True)
class CartesianLogchartFFTAdapterStats:
    """Timing telemetry for :class:`CartesianLogchartFFTAdapter`.

    Stage clocks are taken around the actual codec and native batch call.  The
    values accumulate across calls made while one backend is installed, which
    makes the object suitable for a receiver frame or a small benchmark.
    ``encode_ms`` and ``decode_ms`` are aliases retained for callers that use
    the shorter stage names.
    """

    batches: int = 0
    rows: int = 0
    encoding_ms: float = 0.0
    fft_ms: float = 0.0
    decoding_ms: float = 0.0
    total_ms: float = 0.0
    codec: str = ""
    mode: int = 2
    chart: int = 0

    @property
    def encode_ms(self) -> float:
        return self.encoding_ms

    @property
    def decode_ms(self) -> float:
        return self.decoding_ms

    @property
    def fft_calls(self) -> int:
        return self.batches


class CartesianLogchartFFTAdapter:
    """Explicit Cartesian-to-SPF-to-Cartesian interoperability adapter.

    The receiver works in complex Cartesian values, while the native API works
    in packed SPF-32 words.  Each call therefore performs one block encode,
    one native ``spf32_fft_logchart_rows`` dispatch, and one block decode.  The
    backend is never installed globally; use :func:`native_logchart_fft_active`
    or pass it to :func:`use_fft_backend` explicitly.
    """

    def __init__(
        self,
        mode: str | int = "accurate",
        chart: str | int = "r1",
        stats: CartesianLogchartFFTAdapterStats | None = None,
        codec: tuple[Callable[[Complex], np.ndarray], Callable[[np.ndarray], Complex], str]
        | None = None,
    ) -> None:
        from spf.native import logchart

        self._mode = logchart.mode_from_name(mode)
        self._chart = logchart.chart_from_name(chart)
        self.stats = stats if stats is not None else CartesianLogchartFFTAdapterStats()
        self.stats.mode = self._mode
        self.stats.chart = self._chart
        self._codec_override = codec

    @property
    def mode(self) -> int:
        return self._mode

    @property
    def chart(self) -> int:
        return self._chart

    @staticmethod
    def _codec() -> tuple[Callable[[Complex], np.ndarray], Callable[[np.ndarray], Complex], str]:
        """Return the canonical native fast codec when its library is loaded."""

        try:
            from spf.native._libspf32 import LIBSPF_AVAILABLE

            if LIBSPF_AVAILABLE:
                from spf.native import spf32 as native_spf32

                return native_spf32.encode_fast, native_spf32.decode_fast, "native_fast"
        except (ImportError, RuntimeError, AttributeError):
            # The pure Python codec remains a valid Cartesian boundary when the
            # optional native library is absent or an older build lacks the
            # fast codec symbols.
            pass

        from spf.tensor import decode_words, encode_words

        return encode_words, decode_words, "python_reference"

    @staticmethod
    def _validate_rows(arr: Complex) -> None:
        if arr.ndim != 2:
            raise ValueError(
                "native logchart FFT backend expects complex rows with shape "
                f"(rows, n); got {arr.shape}"
            )
        n = int(arr.shape[1])
        if n < 2 or n > (1 << 16) or (n & (n - 1)) != 0:
            raise ValueError(
                "native logchart FFT backend requires a power-of-two row length "
                f"in [2, 65536]; got n={n}"
            )

    def __call__(self, useful: Complex) -> Complex:
        from spf.native import logchart

        t_total = time.perf_counter_ns()
        arr = np.ascontiguousarray(np.asarray(useful, dtype=np.complex128))
        self._validate_rows(arr)
        encoder, decoder, codec_name = self._codec_override or self._codec()

        t0 = time.perf_counter_ns()
        words = np.ascontiguousarray(encoder(arr), dtype=np.uint32)
        self.stats.encoding_ms += (time.perf_counter_ns() - t0) / 1e6
        if words.shape != arr.shape:
            raise ValueError(
                "native logchart encoder returned shape "
                f"{words.shape}, expected {arr.shape}"
            )

        t0 = time.perf_counter_ns()
        transformed = logchart.fft_rows(
            words,
            mode=self._mode,
            chart=self._chart,
        )
        self.stats.fft_ms += (time.perf_counter_ns() - t0) / 1e6

        t0 = time.perf_counter_ns()
        decoded = np.asarray(decoder(np.ascontiguousarray(transformed, dtype=np.uint32)))
        self.stats.decoding_ms += (time.perf_counter_ns() - t0) / 1e6
        if decoded.shape != arr.shape:
            raise ValueError(
                "native logchart decoder returned shape "
                f"{decoded.shape}, expected {arr.shape}"
            )

        result = np.ascontiguousarray(decoded, dtype=np.complex128)
        self.stats.batches += 1
        self.stats.rows += int(arr.shape[0])
        self.stats.codec = codec_name
        self.stats.total_ms += (time.perf_counter_ns() - t_total) / 1e6
        return result


# A descriptive alias for callers that use "backend" in configuration code.
# Compatibility aliases preserve existing DSP callers while making the actual
# boundary visible to new code and documentation.
NativeLogchartFFTStats = CartesianLogchartFFTAdapterStats
NativeLogchartFFT = CartesianLogchartFFTAdapter
NativeLogchartFFTBackend = CartesianLogchartFFTAdapter


@contextmanager
def use_fft_backend(backend: FFTRows | None) -> Iterator[None]:
    """Temporarily install ``backend`` as the active row-wise FFT (thread-local).

    ``None`` resets to numpy. Restores the prior backend on exit.
    """
    prev = getattr(_state, "fft_rows", None)
    _state.fft_rows = _numpy_fft_rows if backend is None else backend
    try:
        yield
    finally:
        # Restore the thread-local exactly: if nothing was installed before,
        # clear the attribute (don't leave an explicit numpy backend behind, so
        # "unset == default" stays true and the thread carries no residual state).
        if prev is None:
            if hasattr(_state, "fft_rows"):
                del _state.fft_rows
        else:
            _state.fft_rows = prev


@contextmanager
def native_logchart_fft_active(
    mode: str | int = "accurate",
    chart: str | int = "r1",
    stats: CartesianLogchartFFTAdapterStats | None = None,
) -> Iterator[CartesianLogchartFFTAdapterStats]:
    """Install the explicit Cartesian logchart adapter for one thread/block."""

    backend = CartesianLogchartFFTAdapter(mode=mode, chart=chart, stats=stats)
    with use_fft_backend(backend):
        yield backend.stats


# Short alias matching the existing ``overlay_fft_active`` context name.
logchart_fft_active = native_logchart_fft_active


@contextmanager
def overlay_fft_active(
    session: PersistentPLSession,
    stats: OverlayFFTStats | None = None,
    n_fft: int = FFT64_N,
) -> Iterator[OverlayFFTStats]:
    """Run the enclosed block with the FFT stage on the KV260 overlay.

    ``n_fft`` must match the FFT compute unit resident in the loaded overlay
    (64 for the fft64 overlays, 256 for ``spf32_fft256_fb7_bittrick``). Yields the
    :class:`OverlayFFTStats` accumulated over the block (dispatch count + kernel
    wait), so the caller can record PL telemetry alongside the link-quality metrics.
    """
    backend = OverlayFFT64(session, stats, n_fft=n_fft)
    with use_fft_backend(backend):
        yield backend.stats
