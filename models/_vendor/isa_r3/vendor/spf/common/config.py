"""SPF_DATA_DIR resolver and per-run directory helpers.

Convention (mirrors `.claude/skills/data-runs-layout.md`):

    $SPF_DATA_DIR/<YYYY-MM-DD>-<meaningful-name>/

`SPF_DATA_DIR` defaults to `~/domains/spf-data` when unset. The data dir is
created lazily on first use of :func:`get_data_dir(create=True)` or
:func:`get_run_dir`.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

DEFAULT_DATA_DIR = "~/domains/spf-data"
ENV_VAR = "SPF_DATA_DIR"


def get_data_dir(create: bool = False) -> Path:
    """Return the resolved SPF data dir.

    Resolves `$SPF_DATA_DIR` (or the default) and expands `~`. When
    `create=True`, makes the directory if missing.
    """
    raw = os.environ.get(ENV_VAR)
    if not raw:
        raw = DEFAULT_DATA_DIR
    path = Path(os.path.expanduser(raw)).resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_dir(name: str, *, date: _dt.date | None = None, create: bool = True) -> Path:
    """Return `$SPF_DATA_DIR/<YYYY-MM-DD>-<name>/`, creating it by default.

    `date` defaults to today's UTC date.
    """
    if not name or "/" in name:
        raise ValueError(f"invalid run name: {name!r}")
    when = date or _dt.datetime.now(_dt.UTC).date()
    run_dir = get_data_dir(create=create) / f"{when.isoformat()}-{name}"
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_variant_run_dir(family: str, *, date: _dt.date | None = None) -> Path:
    """Return the per-variant-family run dir for unit-test output."""
    return get_run_dir(f"variant-{family}", date=date, create=True)
