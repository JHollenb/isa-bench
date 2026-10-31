from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .fabric_executor import _PSOpEngine, _numpy


PREFLIGHT_SCHEMA = "pan-fabric-oracle-preflight-v1"


def run_oracle_preflight(
    ops: Sequence[str],
    *,
    elements: int = 64,
    workdir: str = "/home/ubuntu/spfr",
    libspf32_path: str | None = None,
    libspf16_path: str | None = None,
) -> dict[str, Any]:
    engine = _PSOpEngine(workdir=workdir, libspf32_path=libspf32_path, libspf16_path=libspf16_path)
    rows = []
    for op in ops:
        rows.append(_preflight_one(engine, op, int(elements)))
    return {
        "schema": PREFLIGHT_SCHEMA,
        "ok": all(row["ok"] for row in rows),
        "workdir": workdir,
        "elements": int(elements),
        "ops": rows,
        "failed_ops": [row for row in rows if not row["ok"]],
    }


def _preflight_one(engine: _PSOpEngine, op: str, elements: int) -> dict[str, Any]:
    arity = _op_arity(op)
    try:
        inputs = _inputs_for_op(op, elements, arity)
        output = engine.compute(op, inputs, elements)
        size = int(getattr(output, "size", len(output)))
        return {
            "op": op,
            "ok": True,
            "arity": arity,
            "elements": elements,
            "output_elements": size,
            "error": None,
        }
    except Exception as exc:
        return {
            "op": op,
            "ok": False,
            "arity": arity,
            "elements": elements,
            "output_elements": 0,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def _op_arity(op: str) -> int:
    name = op.split(".", 1)[1] if "." in op else op
    if name in {"abs2", "conj", "neg", "fft64", "fft64_fb7", "fft64_fb7_fields", "fft64_fb7_sdf"}:
        return 1
    return 2


def _inputs_for_op(op: str, elements: int, arity: int) -> list[Any]:
    np = _numpy()
    base = np.arange(int(elements), dtype=np.uint32) + np.uint32(1)
    if op.startswith("spf16."):
        base = ((base & np.uint32(0x0000FFFF)) | ((base + np.uint32(3)) << np.uint32(16))).astype(np.uint32)
    inputs = [np.ascontiguousarray(base + np.uint32(index), dtype=np.uint32) for index in range(arity)]
    if op.endswith(".div") and len(inputs) > 1:
        inputs[1] = np.ascontiguousarray(inputs[1] | np.uint32(1), dtype=np.uint32)
    return inputs


def write_report(report: dict[str, Any], path: str | Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight PS/rule fabric oracles before promotion evidence runs.")
    parser.add_argument("--op", action="append", required=True, help="Typed SPF op to check; may be repeated.")
    parser.add_argument("--elements", type=int, default=64)
    parser.add_argument("--workdir", default="/home/ubuntu/spfr")
    parser.add_argument("--libspf32")
    parser.add_argument("--libspf16")
    parser.add_argument("--write", help="Write JSON report; stdout when omitted.")
    args = parser.parse_args(argv)
    report = run_oracle_preflight(
        args.op,
        elements=args.elements,
        workdir=args.workdir,
        libspf32_path=args.libspf32,
        libspf16_path=args.libspf16,
    )
    write_report(report, args.write)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
