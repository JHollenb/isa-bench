"""Directory-format ``.spfmodel`` archives.

The archive keeps SPFTensor payload bytes separate from model/export metadata:

* ``manifest.json`` names blocks and records interpretation/reconstruction info.
* ``payloads.spftensor`` is a v3 multi-block envelope.
* ``calibration.json``, ``rewrite_report.json``, and ``oracle_stats.json`` are
  stable sidecar report files used by model-specific exporters.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..v3 import SPFBlockV3
from ..v3_envelope import from_spftensor_v3, to_spftensor_v3


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


@dataclass(frozen=True)
class SPFModelArchive:
    path: Path
    manifest: Mapping[str, Any]
    blocks: tuple[SPFBlockV3, ...]
    calibration: Mapping[str, Any]
    rewrite_report: Mapping[str, Any]
    oracle_stats: Mapping[str, Any]

    def block_for(self, name: str) -> SPFBlockV3:
        entry = self.manifest["entries"][name]
        return self.blocks[int(entry["block_index"])]

    def entry_for(self, name: str) -> Mapping[str, Any]:
        return self.manifest["entries"][name]


def _normalise_entry(entry: Mapping[str, Any], index: int) -> tuple[str, SPFBlockV3, dict[str, Any]]:
    if "block" not in entry:
        raise ValueError("each SPF model entry must contain a 'block'")
    block = entry["block"]
    if not isinstance(block, SPFBlockV3):
        raise TypeError(f"entry block must be SPFBlockV3, got {type(block)!r}")
    name = str(entry.get("name") or entry.get("parameter") or f"block_{index}")
    meta = dict(entry.get("manifest") or entry.get("meta") or {})
    meta.setdefault("block_index", index)
    meta.setdefault("dims", list(block.dims))
    meta.setdefault("base_model", int(block.base.profile_id))
    meta.setdefault("residual_codec", int(block.residual.codec_id))
    meta.setdefault("storage_bits", int(block.storage_bits()))
    return name, block, meta


def write_spfmodel(
    path: str | Path,
    entries: Sequence[Mapping[str, Any]],
    manifest_meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write a directory-format ``.spfmodel`` archive and return its path."""
    out = Path(path)
    if out.exists():
        if not out.is_dir():
            raise ValueError(f"{out} exists and is not a directory")
        # Only clobber a directory we recognise as ours: an existing .spfmodel
        # archive (carries manifest.json) or an empty dir. Refuse to rmtree an
        # unrelated populated directory — a mistyped path would otherwise
        # silently destroy data.
        if not (out / "manifest.json").exists() and any(out.iterdir()):
            raise ValueError(
                f"{out} exists, is non-empty, and is not a .spfmodel archive "
                "(no manifest.json); refusing to overwrite"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    meta = dict(manifest_meta or {})
    blocks: list[SPFBlockV3] = []
    manifest_entries: dict[str, Any] = {}
    for i, entry in enumerate(entries):
        name, block, entry_meta = _normalise_entry(entry, i)
        blocks.append(block)
        manifest_entries[name] = _jsonable(entry_meta)

    manifest = {
        "format": "spfmodel",
        "version": 1,
        "payload": "payloads.spftensor",
        "entries": manifest_entries,
        "meta": _jsonable(meta.get("meta", {})),
    }
    for key in ("model", "task", "producer", "config"):
        if key in meta:
            manifest[key] = _jsonable(meta[key])

    (out / "payloads.spftensor").write_bytes(to_spftensor_v3(tuple(blocks)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    for name in ("calibration", "rewrite_report", "oracle_stats"):
        data = _jsonable(meta.get(name, {}))
        (out / f"{name}.json").write_text(json.dumps(data, indent=2, sort_keys=True))
    return out


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def read_spfmodel(path: str | Path) -> SPFModelArchive:
    """Read a directory-format ``.spfmodel`` archive."""
    root = Path(path)
    manifest = _read_json(root / "manifest.json")
    payload_name = str(manifest.get("payload", "payloads.spftensor"))
    decoded = from_spftensor_v3((root / payload_name).read_bytes())
    blocks = decoded if isinstance(decoded, tuple) else (decoded,)
    return SPFModelArchive(
        path=root,
        manifest=manifest,
        blocks=tuple(blocks),
        calibration=_read_json(root / "calibration.json"),
        rewrite_report=_read_json(root / "rewrite_report.json"),
        oracle_stats=_read_json(root / "oracle_stats.json"),
    )


__all__ = ["SPFModelArchive", "read_spfmodel", "write_spfmodel"]
