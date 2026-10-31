"""Host-side TCP client for the KV260 PL dispatch daemon (pl_server.py).


Eliminates the ~200–400 ms SSH + Python startup cost incurred by the
per-call SSH dispatch in pl_dispatch.py.  The XRT session is kept alive
on the board between calls, so consecutive ops of the same type also
skip the ~5 s xmutil overlay switch.

Recommended setup (SSH tunnel default)
--------------------------------------
``pl_server.py`` binds to ``127.0.0.1`` on the board by default — the
unauthenticated dispatch socket is **not** exposed on the LAN.  Set up
an SSH tunnel from the host once per shell::

    ssh -fNT -L 9999:127.0.0.1:9999 kv260

then connect as ``localhost``::

    with PersistentPLSession("127.0.0.1") as sess:
        x = spf.asarray(z, backend="kv260_pl", session=sess)
        y = x * x          # dispatched via session, no SSH overhead
        out = y.to_numpy()

Or explicitly::

    sess = PersistentPLSession("127.0.0.1")
    sess.connect()
    try:
        result = sess.dispatch_binary(OP_MUL, a_words, b_words)
    finally:
        sess.close()

For trusted-network / dev workflows that bind the daemon directly,
launch ``pl_server.py --allow-unsafe-bind --bind 0.0.0.0`` on the board
and pass the board hostname/IP to ``PersistentPLSession`` instead.

Op codes (match pl_server.py)::

    OP_MUL=0  OP_DIV=1  OP_ADD=2  OP_NEG=3
    OP_ABS2=4  OP_CONJ=5  OP_FFT64=6  OP_RAMP=7

``OP_RAMP`` (single-overlay daemon only) carries a param trailer after the
input words — see :meth:`PersistentPLSession.dispatch_ramp`.

Wire protocol (matches pl_server.py)
------------------------------------
Request frame:
    [magic: 4B b"SPFP"] [version: 1B uint8 == 1]
    [op: 1B uint8] [binary: 1B uint8] [n: 4B uint32 LE]
    [a_data: n*4B uint32 LE]
    [b_data: n*4B uint32 LE, only when binary==1]

Response frame:
    [status: 1B]   0=ok, 1=err
    ok:  [n_out: 4B uint32 LE] [out_data: n_out*4B uint32 LE]
    err: [msg_len: 4B uint32 LE] [msg: msg_len bytes UTF-8]
"""
from __future__ import annotations

import contextlib
import socket
import struct

import numpy as np

# Op codes + wire header live in protocol.py (the single source of truth shared
# with pl_dispatch.py). Re-exported here because external callers historically
# import ``OP_*`` and ``PersistentPLSession`` from this module.
from spf.backends.kria.protocol import (
    OP_ABS2,
    OP_ADD,
    OP_CONJ,
    OP_DIV,
    OP_FFT64,
    OP_MUL,
    OP_NEG,
    OP_RAMP,
)
from spf.backends.kria.protocol import (
    OP_BINARY as _OP_BINARY,
)
from spf.backends.kria.protocol import (
    WIRE_MAGIC as _WIRE_MAGIC,
)
from spf.backends.kria.protocol import (
    WIRE_VERSION as _WIRE_VERSION,
)

__all__ = [
    "OP_ABS2",
    "OP_ADD",
    "OP_CONJ",
    "OP_DIV",
    "OP_FFT64",
    "OP_MUL",
    "OP_NEG",
    "OP_RAMP",
    "PersistentPLSession",
]

# spf_ramp on-fabric dispatch is order==1 only (single coefficient per array
# rides the kernel's s_axilite control register). Mirrors
# pl_server.SPF_RAMP_WIRE_ORDER.
RAMP_WIRE_ORDER: int = 1


def _ramp_trailer_pack(
    q: object, order: int, nonlinear: int,
    ramp_R: object, ramp_b: object, quad: object, cubic: object,
) -> bytes:
    """Serialize the ramp param trailer (mirror of pl_server._ramp_trailer_pack).

    Layout (little-endian): ``order:u32 nonlinear:u32 q:[n*order]i32
    ramp_R:[order*order]u32 ramp_b:[order]u32 quad:[order]u32 cubic:[order]u32``.
    Coefficients are stored as their mod-2^16 phase residue; q keeps signed bits.
    """
    order = int(order)
    parts = [struct.pack("<II", order, int(nonlinear))]
    parts.append(np.ascontiguousarray(np.asarray(q, dtype=np.int32).ravel()).tobytes())
    for arr, cnt in ((ramp_R, order * order), (ramp_b, order),
                     (quad, order), (cubic, order)):
        v = np.zeros(cnt, dtype=np.uint32)
        src = (np.asarray(arr, dtype=np.int64).ravel() & 0xFFFF).astype(np.uint32)
        v[: min(src.size, cnt)] = src[:cnt]
        parts.append(v.tobytes())
    return b"".join(parts)


def collapse_separable_ramp(
    ramp_R: object,
    ramp_b: object,
    quad: object,
    cubic: object,
    mixed: object = None,
    cubic_mixed: object = None,
) -> tuple[int, int, int, int, int]:
    """Collapse a separable N-axis phase ramp to its order-1 fabric equivalent.

    The host executor's increment is
    ``inc = Σ_row b[row] + Σ_col (Σ_row R[row][col])·q[col]
           + (nonlinear ? Σ_dim quad[dim]·q[dim]² + cubic[dim]·q[dim]³)``.

    The ``spf_ramp`` CU carries one coefficient per array in its s_axilite
    control register, so it evaluates only a single axis. A ramp is reducible to
    that order-1 form when it is **separable** (zero off-diagonal ``R`` — no
    cross-axis coupling, and no ``mixed_ramp`` / §13.6 ``cubic_mixed`` term,
    which are cross-axis by construction) and at most one axis carries a
    q-dependent term
    (diagonal ``R``, ``quad`` or ``cubic``). Both converted pipelines satisfy
    this: the radar deramp acts on the range axis only (``R=[[0,0],[0,r]]``) and
    the OFDM CFO on the symbol axis only — they are order-2 purely as an encoding
    of a 2-D cube, not because the phase couples the axes.

    Returns ``(axis, R1, b1, quad1, cubic1)`` — the active axis index and the
    collapsed order-1 coefficients. ``b1`` is the full ``Σ b`` constant offset
    (axis-independent). Raises :class:`NotImplementedError` when the ramp couples
    axes (nonzero off-diagonal ``R``) or spreads a q-dependent term across more
    than one axis (would need sequential dispatch — not wired yet).
    """
    # Cross-axis terms are never separable: fail closed before collapsing, the
    # same way a nonzero off-diagonal R does. mixed_ramp is the ABI v22
    # off-diagonal phase Hessian; cubic_mixed is the ABI v24 pair of AlgRef v0.2
    # §13.6 mixed cubics (q0^2 q1, q0 q1^2). Neither is carried on the wire to
    # the spf_ramp CU, so silently dropping them would produce wrong words.
    for name, values in (("mixed_ramp", mixed), ("cubic_mixed", cubic_mixed)):
        if values is None:
            continue
        if np.any(np.asarray(values, dtype=np.int64).ravel()):
            raise NotImplementedError(
                f"on-fabric ramp cannot lower a cross-axis {name} term; got "
                f"{name}={np.asarray(values, dtype=np.int64).ravel().tolist()} — "
                "run on the host executor."
            )
    R = np.atleast_2d(np.asarray(ramp_R, dtype=np.int64))
    order = int(R.shape[0])
    b = np.asarray(ramp_b, dtype=np.int64).ravel()
    qd = np.zeros(order, np.int64) if quad is None else np.asarray(quad, np.int64).ravel()
    cb = np.zeros(order, np.int64) if cubic is None else np.asarray(cubic, np.int64).ravel()
    b1 = int(b.sum())  # constant offset is axis-independent (Σ over all rows)
    if order == 1:
        return 0, int(R[0, 0]), b1, int(qd[0]), int(cb[0])
    # Separable iff no off-diagonal coupling.
    if np.any(R - np.diag(np.diag(R))):
        raise NotImplementedError(
            "on-fabric ramp cannot lower a coupled (nonzero off-diagonal R) "
            f"phase ramp; got R={R.tolist()} — run on the host executor."
        )
    diag = np.diag(R)
    qdep = [a for a in range(order) if diag[a] or qd[a] or cb[a]]
    if len(qdep) > 1:
        raise NotImplementedError(
            "on-fabric ramp supports one q-dependent axis per dispatch; ramp is "
            f"separable but spans axes {qdep} (sequential per-axis dispatch is "
            "not wired yet) — run on the host executor."
        )
    a = qdep[0] if qdep else 0
    return a, int(diag[a]), b1, int(qd[a]), int(cb[a])


class PersistentPLSession:
    """Persistent TCP session to a running ``pl_server.py`` daemon.

    Parameters
    ----------
    host:
        Hostname or IP of the board (or ``"localhost"`` when an SSH tunnel
        is active).
    port:
        TCP port the server is listening on (default 9999).
    timeout:
        Socket timeout in seconds for individual send/recv calls.
    """

    def __init__(
        self,
        host: str,
        port: int = 9999,
        timeout: float = 120.0,
        *,
        overlay: str | None = None,
        xclbin_sha256: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        overlay
            When set, the daemon at ``host:port`` was launched with
            ``--single-overlay --overlay <name>`` (issue #147). Passing
            this here is purely a host-side provenance assertion: it
            does NOT change the wire protocol or alter the server's
            mode. The session reports the value via :meth:`provenance`
            so artifacts written by callers can label rows
            ``kv260_pl_single_overlay`` instead of ``kv260_pl_sess``.
            Leaving this ``None`` records the legacy multi-overlay
            session label.
        xclbin_sha256
            sha256 of the composite xclbin loaded on the board, when
            known. Carried verbatim into provenance dicts.
        """
        self._host = host
        self._port = port
        self._timeout = timeout
        self._overlay = overlay
        self._xclbin_sha256 = xclbin_sha256
        self._sock: socket.socket | None = None

    # ── provenance ────────────────────────────────────────────────────────

    @property
    def overlay(self) -> str | None:
        """Composite overlay name when running in single-overlay mode, else
        ``None`` (legacy multi-overlay ``kv260_pl_sess`` behaviour)."""
        return self._overlay

    @property
    def xclbin_sha256(self) -> str | None:
        """sha256 of the loaded composite xclbin, when known."""
        return self._xclbin_sha256

    @property
    def is_single_overlay(self) -> bool:
        return self._overlay is not None

    @property
    def backend_label(self) -> str:
        """Schema-valid backend label.

        ``kv260_pl_single_overlay`` when an ``overlay`` was supplied (issue
        #147), else ``kv260_pl_sess``. The two labels are intentionally
        distinct — never collapse them in artifact rows.
        """
        return "kv260_pl_single_overlay" if self._overlay is not None else "kv260_pl_sess"

    @property
    def kernel_granularity(self) -> str:
        """Schema-valid ``kernel_granularity`` for this session."""
        if self._overlay is not None:
            return "xrt_persistent_session_single_overlay"
        return "xrt_persistent_session"

    def provenance(self) -> dict[str, str | None]:
        """Return a provenance dict suitable for inclusion in benchmark
        artifacts under the row's ``backend_info`` / top-level keys.

        Keys mirror the shape used by ``benchmarks/spfarray/backend_bench.py``
        rows so a caller can ``row.update(session.provenance())`` directly:

        * ``backend``               — schema-valid backend label
        * ``transport``             — always ``"persistent_tcp_session"``
        * ``overlay``               — composite name or ``None``
        * ``xclbin_sha256``         — when known
        * ``kernel_granularity``    — schema-valid granularity value
        """
        return {
            "backend": self.backend_label,
            "transport": "persistent_tcp_session",
            "overlay": self._overlay,
            "xclbin_sha256": self._xclbin_sha256,
            "kernel_granularity": self.kernel_granularity,
        }

    # ── connection lifecycle ──────────────────────────────────────────────

    def connect(self) -> None:
        """Open the TCP connection to the board daemon.

        Raises ``RuntimeError`` if the session is already connected — silently
        replacing the socket would leak the previous connection and mask
        caller-side lifecycle bugs.  Call :meth:`close` first to reconnect.
        """
        if self._sock is not None:
            raise RuntimeError(
                "PersistentPLSession is already connected; call close() first to reconnect."
            )
        self._sock = socket.create_connection(
            (self._host, self._port), timeout=self._timeout
        )
        self._sock.settimeout(self._timeout)
        # Disable Nagle: each dispatch is a small (~1 KB) latency-sensitive
        # request/response, and Nagle + delayed-ACK adds ~0.5 ms per dispatch
        # (measured board-local on KV260, 2026-05-29). Must be set on both
        # ends; the pl_server accept loop sets it server-side too.
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self) -> None:
        """Close the connection (idempotent)."""
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None

    def __enter__(self) -> PersistentPLSession:
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── wire protocol helpers ─────────────────────────────────────────────

    def _recvall(self, n: int) -> bytes:
        assert self._sock is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("pl_server closed connection unexpectedly")
            buf.extend(chunk)
        return bytes(buf)

    def _send_request(
        self,
        op: int,
        a: np.ndarray,
        b: np.ndarray | None = None,
    ) -> None:
        assert self._sock is not None
        binary = int(b is not None)
        # Catch wrong-arity calls on the host before they hit the wire so the
        # error blames the caller, not the server's protocol guardrail.
        if binary == 1 and op not in _OP_BINARY:
            raise ValueError(f"op {op} is unary; pass b=None")
        if binary == 0 and op in _OP_BINARY:
            raise ValueError(f"op {op} is binary; b is required")
        n = a.size
        hdr = _WIRE_MAGIC + struct.pack("<BBBI", _WIRE_VERSION, op, binary, n)
        payload = bytearray(hdr)
        payload += np.ascontiguousarray(a.ravel(), dtype=np.uint32).tobytes()
        if b is not None:
            payload += np.ascontiguousarray(b.ravel(), dtype=np.uint32).tobytes()
        self._sock.sendall(bytes(payload))

    def _recv_response(self) -> np.ndarray:
        hdr = self._recvall(5)   # status(1) + n_or_len(4)
        status, n = struct.unpack("<BI", hdr)
        if status == 0:
            data = self._recvall(n * 4)
            return np.frombuffer(data, dtype=np.uint32).copy()
        else:
            msg_bytes = self._recvall(n)
            raise RuntimeError(
                f"pl_server error: {msg_bytes.decode('utf-8', errors='replace')}"
            )

    # ── dispatch API ──────────────────────────────────────────────────────

    def dispatch_unary(self, op: int, x: np.ndarray) -> np.ndarray:
        """Send a unary op request and return the result as a flat uint32 array.

        The caller is responsible for reshaping the result to match ``x.shape``.
        """
        if self._sock is None:
            raise RuntimeError(
                "PersistentPLSession: not connected. "
                "Use 'with session:' or call session.connect() first."
            )
        arr = np.ascontiguousarray(x.ravel(), dtype=np.uint32)
        self._send_request(op, arr)
        return self._recv_response().reshape(x.shape)

    def dispatch_binary(
        self,
        op: int,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """Send a binary op request and return the result as a uint32 array.

        ``a`` and ``b`` must have compatible shapes; broadcasting is resolved
        on the caller's side before calling this method.
        The result has the same shape as the broadcast-resolved ``a``.
        """
        if self._sock is None:
            raise RuntimeError(
                "PersistentPLSession: not connected. "
                "Use 'with session:' or call session.connect() first."
            )
        shape = np.broadcast_shapes(a.shape, b.shape)
        a_c = np.ascontiguousarray(np.broadcast_to(a, shape).ravel(), dtype=np.uint32)
        b_c = np.ascontiguousarray(np.broadcast_to(b, shape).ravel(), dtype=np.uint32)
        self._send_request(op, a_c, b_c)
        return self._recv_response().reshape(shape)

    def dispatch_ramp(
        self,
        x: np.ndarray,
        q: np.ndarray,
        ramp_R: np.ndarray,
        ramp_b: np.ndarray,
        *,
        quadratic_ramp: np.ndarray | None = None,
        cubic_ramp: np.ndarray | None = None,
        mixed_ramp: np.ndarray | None = None,
        cubic_mixed: np.ndarray | None = None,
        nonlinear: bool = False,
    ) -> np.ndarray:
        """Dispatch a phase ramp on the resident spf_ramp CU.

        ``x`` is the input SPF-32 word array; ``q`` is the per-element logical
        index grid — shape ``(n,)`` for an order-1 ramp, or ``(n, order)`` for a
        higher-order one. ``ramp_R`` (order×order), ``ramp_b`` (len order), and
        the optional ``quadratic_ramp`` / ``cubic_ramp`` (len order) are the
        phase-ramp coefficients; ``nonlinear`` enables the quad/cubic fold.

        The CU itself is order-1 (one coefficient per s_axilite register), but a
        **separable** higher-order ramp — zero off-diagonal ``R``, a single
        q-dependent axis — is collapsed to that order-1 form here
        (:func:`collapse_separable_ramp`) and dispatched on the resident CU. Both
        converted pipelines (radar range deramp, OFDM CFO) are separable, so they
        run on fabric with no re-bake. A coupled or multi-q-axis ramp — or one
        carrying a cross-axis ``mixed_ramp`` / §13.6 ``cubic_mixed`` term —
        raises :class:`NotImplementedError`. Result has the same shape as ``x``;
        bit-identical to the host ExecutionPlan ramp op (single-overlay only).
        """
        if self._sock is None:
            raise RuntimeError(
                "PersistentPLSession: not connected. "
                "Use 'with session:' or call session.connect() first."
            )
        R2 = np.atleast_2d(np.asarray(ramp_R))
        order = int(R2.shape[0])
        axis, R1, b1, q1, c1 = collapse_separable_ramp(
            ramp_R, ramp_b, quadratic_ramp, cubic_ramp, mixed_ramp, cubic_mixed
        )
        qarr = np.asarray(q, dtype=np.int64)
        if order == 1:
            q1d = qarr.ravel()
        elif qarr.ndim == 2 and qarr.shape[1] == order:
            q1d = qarr[:, axis]  # project to the single active axis
        else:
            raise ValueError(
                f"dispatch_ramp: order-{order} ramp needs q of shape (n, {order}); "
                f"got q.shape={qarr.shape}"
            )
        n = x.size
        arr = np.ascontiguousarray(x.ravel(), dtype=np.uint32)
        trailer = _ramp_trailer_pack(
            q1d, RAMP_WIRE_ORDER, int(nonlinear), [[R1]], [b1], [q1], [c1]
        )
        hdr = _WIRE_MAGIC + struct.pack("<BBBI", _WIRE_VERSION, OP_RAMP, 0, n)
        assert self._sock is not None
        self._sock.sendall(bytes(hdr) + arr.tobytes() + trailer)
        return self._recv_response().reshape(x.shape)
