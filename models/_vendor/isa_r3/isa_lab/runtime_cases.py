"""Executable runtime continuations and every currently supported Linux service."""


def cases(seed):
    syscall_args = {
        0: {"RDI": 0, "RSI": 0x500000, "RDX": 8},
        1: {"RDI": 1, "RSI": 0x500000, "RDX": 8},
        9: {"RDI": 0, "RSI": 4096, "RDX": 3, "R10": 0x22, "R8": (1 << 64) - 1, "R9": 0},
        10: {"RDI": 0x500000, "RSI": 8192, "RDX": 1},
        11: {"RDI": 0x500000, "RSI": 8192},
        12: {"RDI": 0}, 39: {}, 60: {"RDI": seed & 255},
        158: {"RDI": 0x1002, "RSI": 0x12345678}, 231: {"RDI": seed & 255},
    }
    for number, args in syscall_args.items():
        yield {"code": "0f05", "seed": seed, "options": {"gpr": {"RAX": number, **args}},
               "case": f"syscall-{number}"}
    exit_code = "b83c000000bf" + int(seed & 255).to_bytes(4, "little").hex() + "0f05"
    programs = {
        "scalar-stack": "48b8" + int(seed & 255).to_bytes(8, "little").hex() + "555d4883c0054883e8014889c131c0" + exit_code,
        "call-return-branch": "e802000000eb01c3" + exit_code,
        "simd-memory": "f30f6f03660fd4c1660fefc2f30f7f03" + exit_code,
        "x87-stack": "d9e8d9eed9d0ddd8" + exit_code,
    }
    expected_steps = {"scalar-stack": 10, "call-return-branch": 6, "simd-memory": 7, "x87-stack": 7}
    for name, code in programs.items():
        yield {"code": code, "seed": seed, "options": {}, "case": f"program-{name}",
               "max_steps": expected_steps[name]}
