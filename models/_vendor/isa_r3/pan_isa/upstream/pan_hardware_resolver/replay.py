from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

from .ledger import short_hash, validate_ledger_row


def read_ledger_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: ledger row must be a JSON object")
        rows.append(row)
    return rows


def replay_ledger_jsonl(path: str | Path, *, require_contiguous: bool = False) -> dict[str, Any]:
    return replay_ledger_rows(read_ledger_jsonl(path), require_contiguous=require_contiguous, source=str(path))


def replay_ledger_rows(
    rows: Iterable[dict[str, Any]],
    *,
    require_contiguous: bool = False,
    source: str | None = None,
) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []
    families: dict[str, int] = {}
    previous_after_state_hash: str | None = None
    final_state_hash: str | None = None
    row_count = 0
    for row_count, row in enumerate(rows, start=1):
        family = str(row.get("action_family", "unknown"))
        families[family] = families.get(family, 0) + 1
        row_problems = _replay_row_problems(row)
        envelope = row.get("envelope", {}) if isinstance(row.get("envelope"), dict) else {}
        input_payload = envelope.get("input", {}) if isinstance(envelope.get("input"), dict) else {}
        commit_payload = envelope.get("commit", {}) if isinstance(envelope.get("commit"), dict) else {}
        input_state_hash = input_payload.get("state_hash")
        after_state_hash = commit_payload.get("after_state_hash")
        if require_contiguous and previous_after_state_hash is not None and input_state_hash != previous_after_state_hash:
            row_problems.append(
                f"input state_hash {input_state_hash!r} does not match previous after_state_hash {previous_after_state_hash!r}"
            )
        if row_problems:
            problems.append({"row": row_count, "ledger_id": row.get("ledger_id"), "problems": row_problems})
        previous_after_state_hash = after_state_hash if isinstance(after_state_hash, str) else None
        final_state_hash = previous_after_state_hash
    return {
        "ok": not problems,
        "schema": "pan-action-ledger-replay-v1",
        "source": source,
        "row_count": row_count,
        "families": dict(sorted(families.items())),
        "require_contiguous": bool(require_contiguous),
        "final_state_hash": final_state_hash,
        "problems": problems,
    }


def _replay_row_problems(row: dict[str, Any]) -> list[str]:
    problems = validate_ledger_row(row)
    envelope = row.get("envelope", {}) if isinstance(row.get("envelope"), dict) else {}
    if isinstance(envelope, dict):
        expected_action_id = _expected_action_id(envelope)
        if envelope.get("action_id") != expected_action_id:
            problems.append("action_id mismatch")
        input_payload = envelope.get("input", {}) if isinstance(envelope.get("input"), dict) else {}
        proposal_payload = envelope.get("proposal", {}) if isinstance(envelope.get("proposal"), dict) else {}
        verification_payload = envelope.get("verification", {}) if isinstance(envelope.get("verification"), dict) else {}
        commit_payload = envelope.get("commit", {}) if isinstance(envelope.get("commit"), dict) else {}
        if commit_payload.get("before_state_hash") != input_payload.get("state_hash"):
            problems.append("commit before_state_hash must match input state_hash")
        if bool(verification_payload.get("ok")) != bool(commit_payload.get("ok")):
            problems.append("verification ok must match commit ok")
        expected_after = proposal_payload.get("expected_after", {})
        after_state = commit_payload.get("after_state", {})
        if commit_payload.get("ok") and isinstance(expected_after, dict) and isinstance(after_state, dict):
            if not _contains_subset(after_state, expected_after):
                problems.append("proposal expected_after is not contained in commit after_state")
    if row.get("ledger_id") != _expected_ledger_id(row):
        problems.append("ledger_id mismatch")
    return problems


def _expected_action_id(envelope: dict[str, Any]) -> str:
    body = deepcopy(envelope)
    body.pop("action_id", None)
    return "act_" + short_hash(body)


def _expected_ledger_id(row: dict[str, Any]) -> str:
    body = deepcopy(row)
    body.pop("ledger_id", None)
    return "ledger_" + short_hash(body)


def _contains_subset(payload: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(payload, dict):
            return False
        for key, value in expected.items():
            if key not in payload or not _contains_subset(payload[key], value):
                return False
        return True
    if isinstance(expected, list):
        return isinstance(payload, list) and payload == expected
    return payload == expected


def write_replay_report(report: dict[str, Any], path: str | Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay and validate PAN action ledger JSONL rows.")
    parser.add_argument("ledger_jsonl", help="Ledger JSONL file to replay.")
    parser.add_argument("--require-contiguous", action="store_true", help="Require each row input state to match the previous after-state.")
    parser.add_argument("--write", help="Write the replay report as JSON; stdout when omitted.")
    args = parser.parse_args(argv)
    report = replay_ledger_jsonl(args.ledger_jsonl, require_contiguous=bool(args.require_contiguous))
    write_replay_report(report, args.write)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
