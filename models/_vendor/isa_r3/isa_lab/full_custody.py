"""Write-as-you-go mrun custody, plus a disk-streamed final archive."""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path


def publish(paths, key):
    if not os.environ.get("ISA_MRUN_WORKER"):
        return
    try:
        from mrun.outbox import default
        box = default()
        blobs = [box.put_file(path, str(Path(path).relative_to("result"))) for path in paths if Path(path).is_file()]
        box.record(key, {"files": blobs})
        box.flush()
    except Exception as exc:  # noqa: BLE001 - local immutable evidence survives publication failure
        print("ISA_PUBLICATION_DIAGNOSTIC:" + json.dumps({"key": key, "error": type(exc).__name__}), flush=True)


def pack(out):
    target = Path("result.tar.gz")
    with tarfile.open(target, "w:gz", compresslevel=3) as archive:
        for path in sorted(Path(out).rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(out)), recursive=False)
    h = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    try:
        from mrun.outbox import default
        box = default()
        blob = box.put_file(target, "result.tar.gz")
        box.record("isa-full-final-archive", {"archive": blob, "sha256": h.hexdigest()})
        box.flush()
    except Exception as exc:  # noqa: BLE001 - manual exact-job collection remains possible
        print("ISA_PUBLICATION_DIAGNOSTIC:" + json.dumps({"error": type(exc).__name__}), flush=True)
    print("ISA_ARTIFACT:" + json.dumps({"sha256": h.hexdigest(), "name": "result.tar.gz",
                                       "bytes": target.stat().st_size}), flush=True)
