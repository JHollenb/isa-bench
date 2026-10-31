"""SPFP wire protocol constants — single source of truth for the host clients.

Op codes and the framed-socket header used to talk to the KV260 PL dispatch
daemon. Both :mod:`spf.backends.kria.pl_client` (the persistent TCP session)
and :mod:`spf.backends.kria.pl_dispatch` (the per-call SSH path) import these
so they cannot drift from each other.

The board daemon ``hw/platforms/kria/board/pl_server.py`` keeps its *own*
copy of these constants on purpose: only ``agent.py`` + the shared objects
are shipped to the board (see ``hw/platforms/kria/scripts/install.sh``), so
the board never imports ``spf``. ``tests/backends/test_kria_protocol_drift.py``
parses ``pl_server.py`` and asserts the two copies stay in lock-step.
"""

from __future__ import annotations

# Op codes — must match pl_server.py's ``OP_*`` block.
OP_MUL = 0
OP_DIV = 1
OP_ADD = 2
OP_NEG = 3
OP_ABS2 = 4
OP_CONJ = 5
OP_FFT64 = 6
OP_RAMP = 7  # spf.tensor phase ramp (chirp / CFO). Unary-with-params.
OP_REDUCE = 8  # spf.tensor guard+TROPICAL+LOGPOLAR reducer. Length-reducing.

# Ops that consume two input arrays; everything else is unary.
OP_BINARY: frozenset[int] = frozenset({OP_MUL, OP_DIV, OP_ADD})

# Framed-socket header (pl_client.py speaks it; pl_server.py mirrors it).
WIRE_MAGIC: bytes = b"SPFP"
WIRE_VERSION: int = 1
