"""Encoding adapter contracts for full-ISA trace generation."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from pan_x86_automation import full_isa_user64


HELPER_SOURCE = Path(__file__).resolve().parent / "tools" / "xed_json_helper.c"
HELPER_REL = Path("full_isa") / "tools" / "xed_json_helper"
HELPER_MANIFEST_REL = Path("full_isa") / "tools" / "xed_json_helper_manifest.json"
HELPER_OPERATIONS = ("decode", "encode", "roundtrip", "iform-info")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def helper_paths(root: Path) -> Dict[str, Path]:
    return {
        "source": HELPER_SOURCE,
        "binary": root / HELPER_REL,
        "manifest": root / HELPER_MANIFEST_REL,
    }


def load_helper_manifest(root: Path) -> Dict[str, Any]:
    path = root / HELPER_MANIFEST_REL
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_xed_helper(root: Path, matrix: Dict[str, Any], *, write: bool) -> Dict[str, Any]:
    paths = helper_paths(root)
    cc = shutil.which("cc")
    xed_source_deps = str(matrix.get("xed_source_deps", "source_deps/intel-xed-main-2026-06-16"))
    xed_install = root / xed_source_deps / "install"
    manifest: Dict[str, Any] = {
        "schema_version": "pan-x86-xed-json-helper-manifest-v1",
        "created_unix": int(time.time()),
        "source": str(paths["source"]),
        "source_sha256": sha256_file(paths["source"]) if paths["source"].exists() else None,
        "binary": str(paths["binary"]),
        "operations": list(HELPER_OPERATIONS),
        "jsonl": True,
        "xed_install": str(xed_install),
        "xed_include_available": (xed_install / "include" / "xed").exists(),
        "xed_lib_available": (xed_install / "lib").exists(),
        "xed_linked": False,
        "promotion_eligible": False,
        "implementation": "contract_builtin_table",
        "ok": False,
        "gate_failures": [],
        "compile_command": None,
    }
    if cc is None:
        manifest["gate_failures"].append("missing_cc")
        if write:
            paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
            paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest
    command = [
        cc,
        "-std=c99",
        "-O2",
        "-Wall",
        "-Wextra",
        "-o",
        str(paths["binary"]),
        str(paths["source"]),
    ]
    manifest["compile_command"] = command
    if not write:
        manifest["ok"] = paths["source"].exists()
        return manifest
    paths["binary"].parent.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(command, text=True, capture_output=True, check=False)
    manifest["compile_returncode"] = run.returncode
    manifest["compile_stdout"] = run.stdout.strip().splitlines()[:20]
    manifest["compile_stderr"] = run.stderr.strip().splitlines()[:20]
    if run.returncode != 0:
        manifest["gate_failures"].append("compile_failed")
    elif not paths["binary"].exists():
        manifest["gate_failures"].append("missing_binary_after_compile")
    else:
        manifest["binary_sha256"] = sha256_file(paths["binary"])
        manifest["ok"] = True
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_encoder_probe(root: Path, matrix: Dict[str, Any]) -> Dict[str, Any]:
    xed_source_deps = str(matrix.get("xed_source_deps", "source_deps/intel-xed-main-2026-06-16"))
    xed_install = root / xed_source_deps / "install"
    include_dir = xed_install / "include" / "xed"
    lib_dir = xed_install / "lib"
    helper = root / HELPER_REL
    manifest = load_helper_manifest(root)
    xed_cli = shutil.which("xed")
    return {
        "schema_version": "pan-x86-encoder-probe-v1",
        "xed_install": str(xed_install),
        "xed_include_available": include_dir.exists(),
        "xed_lib_available": lib_dir.exists(),
        "xed_json_helper": str(helper),
        "xed_json_helper_available": helper.exists(),
        "xed_json_helper_manifest": str(root / HELPER_MANIFEST_REL),
        "xed_json_helper_manifest_available": bool(manifest),
        "xed_json_helper_xed_linked": bool(manifest.get("xed_linked")),
        "xed_json_helper_promotion_eligible": bool(manifest.get("promotion_eligible")),
        "xed_cli": xed_cli,
        "ok": helper.exists() or bool(xed_cli),
    }


def _helper_payload(row: Dict[str, Any], operand_policy: Dict[str, Any], operand_case: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    operands = dict((operand_case or {}).get("operand_values") or {})
    payload: Dict[str, Any] = {
        "operation": "roundtrip",
        "target_profile": operand_policy.get("target_profile"),
        "iform": str(row.get("iform") or ""),
        "operand_policy_id": operand_policy.get("policy_id"),
        "operand_case_id": (operand_case or {}).get("operand_case_id"),
        "split": (operand_case or {}).get("split"),
        "operand_values": operands,
        "mode_bits": (operand_case or {}).get("mode_bits"),
        "cpuid_profile": (operand_case or {}).get("cpuid_profile"),
    }
    for key in ("dst", "src", "base", "index"):
        if key in operands:
            payload[key] = operands[key]
    return payload


def _invoke_helper(root: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    helper = root / HELPER_REL
    run = subprocess.run(
        [str(helper)],
        input=json.dumps(payload, sort_keys=True) + "\n",
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if run.returncode != 0:
        return {
            "ok": False,
            "exact_failure_reason": "helper_process_failed",
            "returncode": run.returncode,
            "stderr": run.stderr.strip().splitlines()[:20],
        }
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "exact_failure_reason": "helper_no_output"}
    try:
        return json.loads(lines[0])
    except json.JSONDecodeError as exc:
        return {"ok": False, "exact_failure_reason": f"helper_invalid_json:{exc}"}


def encode_case(
    row: Dict[str, Any],
    operand_policy: Dict[str, Any],
    root: Path,
    matrix: Dict[str, Any],
    operand_case: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    probe = build_encoder_probe(root, matrix)
    iform = str(row.get("iform") or "")
    if probe["xed_json_helper_available"]:
        manifest = load_helper_manifest(root)
        payload = _helper_payload(row, operand_policy, operand_case)
        helper_result = _invoke_helper(root, payload)
        helper_ok = bool(helper_result.get("ok"))
        if helper_ok:
            promotion_eligible = bool(manifest.get("promotion_eligible") and helper_result.get("promotion_eligible"))
            return {
                "status": "xed_json_helper",
                "xed_roundtrip_ok": True,
                "bytes_hex": str(helper_result.get("bytes_hex") or ""),
                "asm": str(helper_result.get("asm") or ""),
                "iform": iform,
                "decoded_iform": helper_result.get("decoded_iform"),
                "decoded_operands": helper_result.get("decoded_operands") or {},
                "attributes": helper_result.get("attributes") or [],
                "extension": helper_result.get("extension") or row.get("extension"),
                "category": helper_result.get("category") or row.get("category"),
                "isa_set": helper_result.get("isa_set") or row.get("isa_set"),
                "operand_policy_id": operand_policy.get("policy_id"),
                "operand_case_id": (operand_case or {}).get("operand_case_id"),
                "promotion_eligible": promotion_eligible,
                "probe": probe,
                "helper_manifest": {
                    key: manifest.get(key)
                    for key in ("schema_version", "binary_sha256", "source_sha256", "xed_linked", "promotion_eligible", "implementation")
                },
                "helper_result": helper_result,
                "note": "Helper result is promotion evidence only when the helper manifest is XED-linked and promotion eligible.",
            }
        return {
            "status": "xed_helper_failed",
            "xed_roundtrip_ok": False,
            "iform": iform,
            "operand_policy_id": operand_policy.get("policy_id"),
            "operand_case_id": (operand_case or {}).get("operand_case_id"),
            "probe": probe,
            "exact_failure_reason": helper_result.get("exact_failure_reason") or "helper_failed",
            "helper_result": helper_result,
            "promotion_eligible": False,
        }
    bytes_hex, asm = full_isa_user64.KNOWN_ENCODINGS.get(iform, ("", ""))
    if bytes_hex:
        return {
            "status": "static_fixture",
            "xed_roundtrip_ok": True,
            "bytes_hex": bytes_hex,
            "asm": asm,
            "iform": iform,
            "operand_policy_id": operand_policy.get("policy_id"),
            "operand_case_id": (operand_case or {}).get("operand_case_id"),
            "promotion_eligible": False,
            "probe": probe,
            "note": "Fixture bytes are smoke-only and never satisfy promotion-grade XED evidence.",
        }
    if probe["xed_cli"]:
        return {
            "status": "xed_cli_available_not_invoked",
            "xed_roundtrip_ok": False,
            "iform": iform,
            "operand_policy_id": operand_policy.get("policy_id"),
            "operand_case_id": (operand_case or {}).get("operand_case_id"),
            "probe": probe,
            "note": "Local xed CLI was found, but iform-directed encode requires the JSON helper.",
        }
    return {
        "status": "missing_encoder",
        "xed_roundtrip_ok": False,
        "bytes_hex": "",
        "asm": "",
        "iform": iform,
        "operand_policy_id": operand_policy.get("policy_id"),
        "operand_case_id": (operand_case or {}).get("operand_case_id"),
        "promotion_eligible": False,
        "probe": probe,
        "exact_failure_reason": "missing_xed_json_helper",
        "note": "Fixture bytes are smoke-only and never satisfy promotion-grade XED evidence.",
    }
