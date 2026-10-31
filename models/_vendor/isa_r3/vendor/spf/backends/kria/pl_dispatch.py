"""KV260 PL fabric dispatch — SSH + SCP + pyxrt.

Sends SPF-32 encoded arrays to the KV260 PL fabric over SSH + SCP.
One SSH session per dispatch call; ``xmutil loadapp`` is called on the
board before opening the pyxrt device so the correct firmware overlay
is active.

**Per-kernel dispatch, not the batched HLS kernel from kria-pl-bench.**
The board scripts call ``kernel(...); run.wait()`` once per invocation.
Throughput scales with the N passed to each call (larger N = more work
per SSH round-trip), but the SSH + xmutil overhead (~5 s/call) dominates
for any N reachable in one dispatch.

Not a contract runner — no golden comparison, no acceptance gate.
For production evidence collection use hw/platforms/kria/host/runner.py.

SSH failure raises and loses the entire batch. Callers are responsible
for retry logic in production deployments.

The SSH + SCP choreography (push inputs, run a board script over stdin,
pull outputs, clean up) lives in :mod:`spf.backends.kria.transport`; this
module only carries the board-side pyxrt scripts and the named op wrappers.

Persistent session
------------------
Pass ``session=<PersistentPLSession>`` to any named wrapper to route
the call through the board daemon instead of a fresh SSH session.
This eliminates the ~200–400 ms SSH + Python startup overhead and
amortises xmutil (~5 s) over consecutive calls of the same op.
See :mod:`spf.backends.kria.pl_client` for daemon setup instructions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# Op codes live in protocol.py (shared with pl_client.py).
from spf.backends.kria.protocol import (
    OP_ABS2,
    OP_ADD,
    OP_CONJ,
    OP_DIV,
    OP_FFT64,
    OP_MUL,
    OP_NEG,
)
from spf.backends.kria.transport import resolve_host, resolve_workdir, run_inline_script

if TYPE_CHECKING:
    from spf.backends.kria.pl_client import PersistentPLSession

# Per-call SSH timeout (seconds) for the mkdir/scp steps; the board-side run
# step scales this with batch size where relevant.
_SSH_TIMEOUT = 120


# ── Session-aware dispatch (bypasses SSH/SCP when session is set) ─────────────


def unary_via_session(
    session: PersistentPLSession,
    op: int,
    x: np.ndarray,
) -> np.ndarray:
    """Dispatch a unary op through a persistent board session."""
    return session.dispatch_unary(op, x)


def binary_via_session(
    session: PersistentPLSession,
    op: int,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Dispatch a binary op through a persistent board session."""
    return session.dispatch_binary(op, a, b)

_BOARD_SCRIPT = """\
import sys, os, subprocess, numpy as np, pyxrt

tmp = sys.argv[1]
workdir = sys.argv[2] if len(sys.argv) > 2 else '/home/ubuntu/spfr'
x = np.load(os.path.join(tmp, 'fft_in.npy'))
batch = x.shape[0]
xclbin = os.path.join(workdir, 'xclbins', 'spf32_fft64.xclbin')

subprocess.run(['sudo', 'xmutil', 'unloadapp'], capture_output=True)
subprocess.run(['sudo', 'xmutil', 'loadapp', 'spf32_fft64'], check=True, capture_output=True)
device = pyxrt.device(0)
uid = device.load_xclbin(xclbin)
kernel = pyxrt.kernel(device, uid, 'spf32_fft64_kernel')

nbytes = 64 * 4
bo_in  = pyxrt.bo(device, nbytes, pyxrt.bo.cacheable, kernel.group_id(0))
bo_out = pyxrt.bo(device, nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
view_in  = np.frombuffer(bo_in.map(),  dtype=np.uint32)
view_out = np.frombuffer(bo_out.map(), dtype=np.uint32)

out = np.empty_like(x)
for i in range(batch):
    view_in[:] = x[i]
    bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes, 0)
    run = kernel(bo_in, bo_out)
    run.wait()
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, 0)
    out[i] = view_out[:]

np.save(os.path.join(tmp, 'fft_out.npy'), out)
"""


def fft64_batch(
    x: np.ndarray,
    *,
    host: str | None = None,
    workdir: str | None = None,
) -> np.ndarray:
    """Dispatch one or a batch of N=64 SPF-32 FFTs to the KV260 PL fabric.

    Parameters
    ----------
    x
        uint32 array, shape (64,) or (batch, 64).
    host
        SSH target (e.g. ``kv260`` or ``ubuntu@192.168.1.x``).
        Defaults to ``$SPF_KRIA_HOST``.
    workdir
        Remote working dir. Defaults to ``$SPF_KRIA_WORKDIR`` or
        ``/home/ubuntu/spfr``.

    Returns
    -------
    uint32 array, same shape as *x*.

    Notes
    -----
    Latency includes one SSH round-trip + SCP overhead regardless of batch
    size. At N=64 the per-symbol PL compute is ~1044 µs (from kria-pl-bench);
    the SSH round-trip adds ~50–200 ms amortised over the full batch.
    ``xmutil loadapp`` adds ~4–5 s per call (firmware overlay switch).
    """
    host = resolve_host(host)
    workdir = resolve_workdir(workdir)

    scalar = x.ndim == 1
    if scalar:
        x = x[np.newaxis, :]
    if x.shape[1] != 64:
        raise ValueError(f"kv260_pl fft64 kernel is fixed N=64; got N={x.shape[1]}")

    x_c = np.ascontiguousarray(x, dtype=np.uint32)
    out = run_inline_script(
        host,
        _BOARD_SCRIPT.encode(),
        infiles={"fft_in.npy": x_c},
        outfiles=["fft_out.npy"],
        extra_args=[workdir],
        # Board-side kernel loop can be slow at large batch; scale the run
        # timeout with batch size as the original per-call path did.
        timeout=_SSH_TIMEOUT * x_c.shape[0],
        prefix="spf-pl-fft-",
    )["fft_out.npy"]

    result: np.ndarray = out[0] if scalar else out
    return result


# ── Generic arithmetic PL dispatch ───────────────────────────────────────────
#
# Board has xclbins for: mul, div, add, abs2, conj, neg.
# All use the same SSH+SCP pattern as fft64_batch but with variable N.
# sub is NOT on the board; _KV260PLBackend implements it as neg(b)+add(a,neg_b).

_UNARY_SCRIPT = """\
import sys, os, subprocess, numpy as np, pyxrt
tmp, workdir, kname, xname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
a = np.load(os.path.join(tmp, 'in_a.npy'))
n = a.size
app = xname.replace('.xclbin', '')
subprocess.run(['sudo', 'xmutil', 'unloadapp'], capture_output=True)
subprocess.run(['sudo', 'xmutil', 'loadapp', app], check=True, capture_output=True)
device = pyxrt.device(0)
uid = device.load_xclbin(os.path.join(workdir, 'xclbins', xname))
kernel = pyxrt.kernel(device, uid, kname)
nbytes = n * 4
bo_a   = pyxrt.bo(device, nbytes, pyxrt.bo.cacheable, kernel.group_id(0))
bo_out = pyxrt.bo(device, nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
view_a   = np.frombuffer(bo_a.map(),   dtype=np.uint32)
view_out = np.frombuffer(bo_out.map(), dtype=np.uint32)
view_a[:] = a.ravel()
bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes, 0)
run = kernel(bo_a, bo_out, n); run.wait()
bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, 0)
np.save(os.path.join(tmp, 'out.npy'), view_out[:n].reshape(a.shape).copy())
"""

_BINARY_SCRIPT = """\
import sys, os, subprocess, numpy as np, pyxrt
tmp, workdir, kname, xname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
a = np.load(os.path.join(tmp, 'in_a.npy'))
b = np.load(os.path.join(tmp, 'in_b.npy'))
n = a.size
app = xname.replace('.xclbin', '')
subprocess.run(['sudo', 'xmutil', 'unloadapp'], capture_output=True)
subprocess.run(['sudo', 'xmutil', 'loadapp', app], check=True, capture_output=True)
device = pyxrt.device(0)
uid = device.load_xclbin(os.path.join(workdir, 'xclbins', xname))
kernel = pyxrt.kernel(device, uid, kname)
nbytes = n * 4
bo_a   = pyxrt.bo(device, nbytes, pyxrt.bo.cacheable, kernel.group_id(0))
bo_b   = pyxrt.bo(device, nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
bo_out = pyxrt.bo(device, nbytes, pyxrt.bo.cacheable, kernel.group_id(2))
view_a   = np.frombuffer(bo_a.map(),   dtype=np.uint32)
view_b   = np.frombuffer(bo_b.map(),   dtype=np.uint32)
view_out = np.frombuffer(bo_out.map(), dtype=np.uint32)
view_a[:] = a.ravel()
view_b[:] = b.ravel()
bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes, 0)
bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes, 0)
run = kernel(bo_a, bo_b, bo_out, n); run.wait()
bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, 0)
np.save(os.path.join(tmp, 'out.npy'), view_out[:n].reshape(a.shape).copy())
"""


def _run_pl_unary(
    kernel_name: str,
    xclbin_name: str,
    x: np.ndarray,
    *,
    host: str | None = None,
    workdir: str | None = None,
) -> np.ndarray:
    host = resolve_host(host)
    workdir = resolve_workdir(workdir)
    arr = np.ascontiguousarray(x, dtype=np.uint32)
    return run_inline_script(
        host,
        _UNARY_SCRIPT.encode(),
        infiles={"in_a.npy": arr},
        outfiles=["out.npy"],
        extra_args=[workdir, kernel_name, xclbin_name],
        timeout=_SSH_TIMEOUT,
        prefix="spf-pl-op-",
    )["out.npy"]


def _run_pl_binary(
    kernel_name: str,
    xclbin_name: str,
    a: np.ndarray,
    b: np.ndarray,
    *,
    host: str | None = None,
    workdir: str | None = None,
) -> np.ndarray:
    host = resolve_host(host)
    workdir = resolve_workdir(workdir)
    shape = np.broadcast_shapes(a.shape, b.shape)
    a_c = np.ascontiguousarray(np.broadcast_to(a, shape), dtype=np.uint32)
    b_c = np.ascontiguousarray(np.broadcast_to(b, shape), dtype=np.uint32)
    return run_inline_script(
        host,
        _BINARY_SCRIPT.encode(),
        infiles={"in_a.npy": a_c, "in_b.npy": b_c},
        outfiles=["out.npy"],
        extra_args=[workdir, kernel_name, xclbin_name],
        timeout=_SSH_TIMEOUT,
        prefix="spf-pl-op-",
    )["out.npy"]


# ── Named wrappers (mirror spf.native.spf32 array API) ───────────────────────────
#
# All wrappers accept an optional ``session=`` kwarg.  When set, the call is
# routed through the persistent board daemon (no SSH/SCP overhead).  When
# omitted, the legacy SSH+SCP path is used.

def mul_batch(
    a: np.ndarray, b: np.ndarray,
    *, host: str | None = None, workdir: str | None = None,
    session: PersistentPLSession | None = None,
) -> np.ndarray:
    if session is not None:
        return binary_via_session(session, OP_MUL, a, b)
    return _run_pl_binary("spf32_mul_kernel", "spf32_mul.xclbin", a, b,
                          host=host, workdir=workdir)


def div_batch(
    a: np.ndarray, b: np.ndarray,
    *, host: str | None = None, workdir: str | None = None,
    session: PersistentPLSession | None = None,
) -> np.ndarray:
    if session is not None:
        return binary_via_session(session, OP_DIV, a, b)
    return _run_pl_binary("spf32_div_kernel", "spf32_div.xclbin", a, b,
                          host=host, workdir=workdir)


def add_batch(
    a: np.ndarray, b: np.ndarray,
    *, host: str | None = None, workdir: str | None = None,
    session: PersistentPLSession | None = None,
) -> np.ndarray:
    if session is not None:
        return binary_via_session(session, OP_ADD, a, b)
    return _run_pl_binary("spf32_add_kernel", "spf32_add.xclbin", a, b,
                          host=host, workdir=workdir)


def neg_batch(
    x: np.ndarray,
    *, host: str | None = None, workdir: str | None = None,
    session: PersistentPLSession | None = None,
) -> np.ndarray:
    if session is not None:
        return unary_via_session(session, OP_NEG, x)
    return _run_pl_unary("spf32_neg_kernel", "spf32_neg.xclbin", x,
                         host=host, workdir=workdir)


def abs2_batch(
    x: np.ndarray,
    *, host: str | None = None, workdir: str | None = None,
    session: PersistentPLSession | None = None,
) -> np.ndarray:
    if session is not None:
        return unary_via_session(session, OP_ABS2, x)
    return _run_pl_unary("spf32_abs2_kernel", "spf32_abs2.xclbin", x,
                         host=host, workdir=workdir)


def conj_batch(
    x: np.ndarray,
    *, host: str | None = None, workdir: str | None = None,
    session: PersistentPLSession | None = None,
) -> np.ndarray:
    if session is not None:
        return unary_via_session(session, OP_CONJ, x)
    return _run_pl_unary("spf32_conj_kernel", "spf32_conj.xclbin", x,
                         host=host, workdir=workdir)


def fft64_via_session(session: PersistentPLSession, x: np.ndarray) -> np.ndarray:
    """FFT64 through a persistent session. x: uint32 (64,) or (batch, 64)."""
    scalar = x.ndim == 1
    if scalar:
        x = x[np.newaxis, :]
    if x.shape[1] != 64:
        raise ValueError(f"fft64 kernel is fixed N=64; got N={x.shape[1]}")
    x_c = np.ascontiguousarray(x, dtype=np.uint32)
    # Server enforces n % 64 == 0 too, but reject client-side so a
    # malformed batch never hits the wire.
    if x_c.size % 64 != 0:
        raise ValueError(
            f"fft64 batched payload must be a multiple of 64 words, got size={x_c.size}"
        )
    result = session.dispatch_unary(OP_FFT64, x_c)
    return result[0] if scalar else result
