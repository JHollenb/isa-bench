#!/usr/bin/env python3
"""Validate Stage 2 traces with a Bochs long-mode boot-image oracle."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pan_x86_stages.stage0_x86_pan import ResourceMonitor, write_json
from pan_x86_stages.validate_stage0_qemu import text_tail
from pan_x86_stages.validate_stage2_qemu import DEFAULT_RUN_DIR, build_image, collect_steps, parse_qemu_output


RESULT_LINE_RE = re.compile(r"(?m)^RESULT[ \t]+\d+[ \t]+\d+[ \t]+\d+[ \t]*(?:\r?\n|\x00|$)")


def bochs_command(
    bochs: str,
    image_path: Path,
    serial_path: Path,
    log_path: Path,
    cpu_model: str,
    ips: int,
    memory_mib: int,
) -> List[str]:
    return [
        bochs,
        "-n",
        "-log",
        str(log_path),
        "display_library: nogui",
        f"cpu: model={cpu_model}, count=1, ips={ips}, reset_on_triple_fault=1, ignore_bad_msrs=1",
        f"memory: guest={memory_mib}, host={memory_mib}",
        "romimage: file=$BXSHARE/BIOS-bochs-latest, options=fastboot",
        "vgaromimage: file=$BXSHARE/VGABIOS-lgpl-latest.bin",
        "boot: floppy",
        f"floppya: 1_44={image_path}, status=inserted",
        f"com1: enabled=1, mode=file, dev={serial_path}",
        "panic: action=fatal",
        "error: action=report",
        "info: action=ignore",
        "debug: action=ignore",
    ]


def run_bochs_until_result(cmd: Sequence[str], serial_path: Path, timeout_s: float) -> Dict[str, Any]:
    start = time.time()
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write("6\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        pass

    serial = ""
    stopped_after_result = False
    timed_out = False
    while proc.poll() is None:
        if serial_path.exists():
            serial = serial_path.read_text(encoding="utf-8", errors="replace")
            if RESULT_LINE_RE.search(serial):
                stopped_after_result = True
                proc.terminate()
                break
        if time.time() - start > timeout_s:
            timed_out = True
            proc.terminate()
            break
        time.sleep(0.05)

    try:
        stdout, stderr = proc.communicate(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True

    if serial_path.exists():
        serial = serial_path.read_text(encoding="utf-8", errors="replace")
    return {
        "cmd": list(cmd),
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - start, 4),
        "timed_out": timed_out,
        "stopped_after_result": stopped_after_result,
        "serial": serial,
        "stdout": text_tail(stdout),
        "stderr": text_tail(stderr),
    }


def run_bochs_oracle(args: argparse.Namespace) -> Dict[str, Any]:
    report_dir = args.report_dir or (args.run_dir / "bochs_stage2_oracle")
    report_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(report_dir, interval_s=args.monitor_interval)
    monitor.start()
    try:
        monitor.set_phase("collect")
        selected, source_counts, skipped_counts = collect_steps(args.run_dir, args.splits, args.max_steps, args.per_split_limit)
        if not selected:
            raise ValueError("no Stage 2 steps selected")

        bochs = shutil.which(args.bochs_binary)
        availability = {
            "bochs_binary": args.bochs_binary,
            "bochs_path": bochs,
            "bochs_available": bochs is not None,
            "cpu_model": args.cpu_model,
        }
        if bochs is None:
            report = {
                "enabled": True,
                "oracle": "bochs_x86_64_long_mode",
                "ok": False,
                "error": f"{args.bochs_binary} not found",
                "availability": availability,
            }
            write_json(report_dir / "bochs_stage2_oracle_report.json", report)
            write_json(args.run_dir / "bochs_stage2_oracle_report.json", report)
            return report

        chunks = []
        checked_steps = 0
        exact_steps = 0
        failure_count = 0
        failure_examples: List[Dict[str, Any]] = []
        all_stopped_after_result = True
        any_timed_out = False
        last_run: Dict[str, Any] = {}
        for chunk_index, start in enumerate(range(0, len(selected), args.chunk_size)):
            chunk_items = [dict(item, global_index=start + local_index) for local_index, item in enumerate(selected[start:start + args.chunk_size])]
            monitor.set_phase(f"build_image_{chunk_index}")
            image_info = build_image(chunk_items, report_dir / f"stage2_bochs_oracle_{chunk_index:03d}.img", args.progress_interval)
            serial_path = report_dir / f"bochs_serial_{chunk_index:03d}.log"
            log_path = report_dir / f"bochs_{chunk_index:03d}.log"
            serial_path.write_text("", encoding="utf-8")

            cmd = bochs_command(
                bochs,
                Path(image_info["image_path"]),
                serial_path,
                log_path,
                args.cpu_model,
                args.ips,
                args.memory_mib,
            )
            monitor.set_phase(f"execute_{chunk_index}")
            run = run_bochs_until_result(cmd, serial_path, args.timeout_s)
            parsed = parse_qemu_output(run["serial"], chunk_items)
            chunk_ok = (
                parsed["checked_steps"] == len(chunk_items)
                and parsed["exact_steps"] == len(chunk_items)
                and parsed["failure_count"] == 0
                and run["stopped_after_result"]
            )
            checked_steps += parsed["checked_steps"]
            exact_steps += parsed["exact_steps"]
            failure_count += parsed["failure_count"]
            failure_examples.extend(parsed["failure_examples"])
            all_stopped_after_result = all_stopped_after_result and run["stopped_after_result"]
            any_timed_out = any_timed_out or run["timed_out"]
            last_run = run
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "start_index": start,
                    "steps": len(chunk_items),
                    "ok": chunk_ok,
                    "checked_steps": parsed["checked_steps"],
                    "exact_steps": parsed["exact_steps"],
                    "exact": parsed["exact"],
                    "failure_count": parsed["failure_count"],
                    "failure_examples": parsed["failure_examples"],
                    "elapsed_s": run["elapsed_s"],
                    "returncode": run["returncode"],
                    "timed_out": run["timed_out"],
                    "stopped_after_result": run["stopped_after_result"],
                    "serial_path": str(serial_path),
                    "log_path": str(log_path),
                    "serial_tail": run["serial"][-2000:],
                    "stdout_tail": run["stdout"][-2000:],
                    "stderr_tail": run["stderr"][-2000:],
                    "image": image_info,
                }
            )
            if not chunk_ok:
                break

        ok = (
            checked_steps == len(selected)
            and exact_steps == len(selected)
            and failure_count == 0
            and all_stopped_after_result
        )
        split_counts = {
            split: {"checked_steps": sum(1 for item in selected if item["split"] == split)}
            for split in args.splits
        }
        report = {
            "enabled": True,
            "oracle": "bochs_x86_64_long_mode",
            "ok": ok,
            "stage": "stage2_control_flow",
            "run_dir": str(args.run_dir),
            "report_dir": str(report_dir),
            "checked_steps": checked_steps,
            "exact_steps": exact_steps,
            "exact": round(exact_steps / max(1, checked_steps), 6),
            "failure_count": failure_count,
            "failure_examples": failure_examples,
            "splits": split_counts,
            "requested_splits": list(args.splits),
            "source_trace_counts": source_counts,
            "skipped_unsafe_steps": skipped_counts,
            "max_steps": args.max_steps,
            "per_split_limit": args.per_split_limit,
            "chunk_size": args.chunk_size,
            "elapsed_s": round(sum(float(chunk["elapsed_s"]) for chunk in chunks), 4),
            "availability": availability,
            "chunks": chunks,
            "image": {
                "chunk_count": len(chunks),
                "max_used_image_bytes": max((chunk["image"]["used_image_bytes"] for chunk in chunks), default=0),
                "max_second_stage_bytes": max((chunk["image"]["second_stage_bytes"] for chunk in chunks), default=0),
                "max_second_stage_sectors": max((chunk["image"]["second_stage_sectors"] for chunk in chunks), default=0),
            },
            "cmd": last_run.get("cmd", []),
            "returncode": last_run.get("returncode"),
            "timed_out": any_timed_out,
            "stopped_after_result": all_stopped_after_result,
            "serial_tail": str(last_run.get("serial", ""))[-4000:],
            "stdout_tail": str(last_run.get("stdout", ""))[-4000:],
            "stderr_tail": str(last_run.get("stderr", ""))[-4000:],
            "normalization": (
                "Stage 2 uses the same generated long-mode floppy images as the QEMU oracle. "
                "Bochs does not use QEMU's isa-debug-exit device here; the runner starts simulation "
                "through the text config menu, watches COM1 for RESULT, and terminates Bochs after "
                "serial completion."
            ),
        }
        write_json(report_dir / "bochs_stage2_oracle_report.json", report)
        write_json(args.run_dir / "bochs_stage2_oracle_report.json", report)
        return report
    finally:
        monitor.set_phase("finalize")
        monitor.stop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=["train", "val", "long"], default=["val", "long"])
    p.add_argument("--max-steps", type=int, default=768)
    p.add_argument("--per-split-limit", type=int, default=384)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--timeout-s", type=float, default=48.0)
    p.add_argument("--bochs-binary", default="bochs")
    p.add_argument("--cpu-model", default="corei7_haswell_4770")
    p.add_argument("--ips", type=int, default=50_000_000)
    p.add_argument("--memory-mib", type=int, default=64)
    p.add_argument("--progress-interval", type=int, default=96)
    p.add_argument("--chunk-size", type=int, default=384)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_bochs_oracle(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
