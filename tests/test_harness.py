"""Unit checks for the comparison, Linux boundary, memory image and metrics (no models, no compute)."""
import json
import struct

import numpy as np
import pytest

from isa_bench.adapters import finish, linux_boundary
from isa_bench.metrics import flat_metrics, summarize
from isa_bench.state import TF, ArchState, Comparator, PageMemory, Trace, eflags_of, flags_of


def make_program(tmp_path, rows, stdout=b"hi", exit_code=0):
    d = tmp_path / "prog-Os"
    d.mkdir()
    np.savez_compressed(d / "trace.npz", steps=np.array(rows, dtype=np.uint64))
    (d / "native.json").write_text(json.dumps({"stdout_hex": stdout.hex(), "exit_code": exit_code,
                                               "steps": len(rows), "complete": True}))
    body = b"\x90" * 4096
    (d / "mem.bin").write_bytes(struct.pack("<QQ", 0x401000, len(body)) + b"r-xp" + body)
    return d


def row(rip, rax=0, r11=0, eflags=0x202):
    regs = [0] * 16
    regs[0], regs[11] = rax, r11
    return [rip, *regs, eflags]


def test_flags_roundtrip():
    f = flags_of(0x8C5)  # CF PF AF ZF SF OF set? 0x8C5 = OF|SF|ZF|PF|CF
    assert f == {"CF": 1, "PF": 1, "AF": 0, "ZF": 1, "SF": 1, "OF": 1}
    assert flags_of(eflags_of(f)) == f


def test_comparator_prefix_and_first_divergence(tmp_path):
    rows = [row(0x401000), row(0x401001, rax=1), row(0x401002, rax=2), row(0x401003, rax=3)]
    trace = Trace(make_program(tmp_path, rows))
    cmp = Comparator(trace, gprs=16)
    for i, rax in enumerate((1, 2, 9)):
        got = trace.state(i + 1)
        got.regs = list(got.regs)
        got.regs[0] = rax
        cmp.observe(i, got, window=b"\x90", before=trace.state(i))
    s = cmp.summary()
    assert s["exact_prefix_steps"] == 2 and s["exact_steps"] == 2
    assert s["first_divergence"]["step"] == 2
    assert s["first_divergence"]["diff"]["RAX"] == ["0x9", "0x3"]
    assert s["divergence_inputs"]["step"] == 2 and s["divergence_inputs"]["window"] == "90"


def test_comparator_masks_tf_in_r11_and_skips_uncompared_regs(tmp_path):
    rows = [row(0x401000), row(0x401002, r11=0x202 | TF)]
    trace = Trace(make_program(tmp_path, rows))
    cmp = Comparator(trace, gprs=16)
    got = trace.state(1)
    got.regs = list(got.regs)
    got.regs[11] = 0x202
    assert cmp.observe(0, got, window=b"\x0f\x05", before=trace.state(0))
    narrow = Comparator(trace, gprs=8)  # an 8-register model: R8..R15 are not its state
    got.regs[12] = None
    assert narrow.observe(0, got, window=b"\x0f\x05")
    assert not cmp.observe(0, got, window=b"\x0f\x05")  # None inside the compared range is a divergence


def test_linux_boundary_write_and_exit(tmp_path):
    mem = PageMemory(make_program(tmp_path, [row(0x401000)]) / "mem.bin")
    out = bytearray()
    regs = [1, 0, 3, 0, 0, 0, 0x401000, 1] + [0] * 8
    stop, after = linux_boundary(regs, 0x401010, mem, out, rflags=0x246)
    assert stop is None and bytes(out) == b"\x90" * 3
    assert after[0] == 3 and after[1] == 0x401010 and after[11] == 0x246
    stop, _ = linux_boundary([60, 0, 0, 0, 0, 0, 0, 7] + [0] * 8, 0, mem, out)
    assert stop == {"kind": "exit", "exit_code": 7}
    stop, _ = linux_boundary([2] + [0] * 15, 0, mem, out)
    assert stop["kind"] == "unsupported_syscall"


def test_page_memory_fault_and_fetch(tmp_path):
    mem = PageMemory(make_program(tmp_path, [row(0x401000)]) / "mem.bin")
    assert mem.fetch(0x401FF8) == b"\x90" * 8
    assert not mem.write(0x500000, b"\x00")
    assert mem.fault == {"address": "0x500000"}


@pytest.mark.parametrize("stop,stdout,status", [
    ({"kind": "exit", "exit_code": 0}, b"hi", "success"),
    ({"kind": "exit", "exit_code": 1}, b"hi", "wrong_output"),
    ({"kind": "abstain"}, b"", "abstain"),
    ({"kind": "unsupported_syscall"}, b"", "refused"),
    ({"kind": "store_fault"}, b"", "fault"),
    ({"kind": "step_limit"}, b"", "step_limit"),
    ({"kind": "wall_limit"}, b"", "wall_limit"),
])
def test_finish_status(tmp_path, stop, stdout, status):
    trace = Trace(make_program(tmp_path, [row(0x401000), row(0x401001)]))
    o = finish({"stop": stop, "stdout_hex": stdout.hex()}, Comparator(trace), trace)
    assert o["status"] == status


def test_summarize_and_flat_metrics(tmp_path):
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "reference.json").write_text(json.dumps({"programs": {
        "a-Os": {"steps": 11, "native_timing": [{"wall_s": 0.001}]},
        "b-Os": {"steps": 101, "native_timing": [{"wall_s": 0.002}]}}}))
    ev = tmp_path / "eval"
    (ev / "cells").mkdir(parents=True)
    usage = {"wall_s": 0.01, "us_per_step": 1000.0, "peak_rss_mb": 50}
    (ev / "cells/a-Os.json").write_text(json.dumps({"outcome": {"status": "success", "exact_prefix_steps": 10,
                                                                "trace_exact": True, "endpoint_ok": True},
                                                    "samples": [{"usage": usage}, {"usage": usage}],
                                                    "deterministic": True, "metadata": {"parameter_count": 5}}))
    (ev / "cells/b-Os.json").write_text(json.dumps({"outcome": {"status": "fault", "exact_prefix_steps": 40},
                                                    "samples": [{"usage": usage}], "deterministic": True}))
    s = summarize("m", "suite", ev, ref)
    m = s["metrics"]
    assert m["success"] == 1 and m["programs"] == 2 and m["trace_exact"] == 1
    assert m["step_coverage"] == pytest.approx(50 / 110)
    assert m["status_counts"]["fault"] == 1 and m["parameter_count"] == 5
    flat = flat_metrics(s)
    assert flat["suite.success"] == 1.0 and flat["suite.status.fault"] == 1.0
    assert all(isinstance(v, float) for v in flat.values())


def test_undefined_flags_masked_until_redefined(tmp_path):
    zf = 0x202 | (1 << 6)
    rows = [row(0x401000), row(0x401004, eflags=zf), row(0x401007, eflags=zf), row(0x40100a, eflags=zf)]
    d = make_program(tmp_path, rows)
    (d / "disasm.txt").write_text("  401000:\timul   $0x7,%rax,%rax\n  401004:\tmov    %rax,%rbx\n"
                                  "  401007:\ttest   %rax,%rax\n")
    trace = Trace(d)
    cmp = Comparator(trace, gprs=16, flags=("CF", "ZF", "SF", "OF"))
    for i in range(3):
        got = trace.state(i + 1)
        got.flags = dict(got.flags, ZF=0)  # model's ZF is 0 throughout
        cmp.observe(i, got)
    s = cmp.summary()
    # imul leaves ZF undefined, mov keeps it undefined, test defines it again: divergence at step 2 only
    assert s["exact_prefix_steps"] == 2 and s["masked_undefined_flag_diffs"] == 2
    assert s["first_divergence"]["step"] == 2 and "ZF" in s["first_divergence"]["diff"]
