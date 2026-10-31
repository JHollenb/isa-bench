"""Unicorn 2.1.4 emulator from the native initial state: the harness's known-good control (no learning)."""
from __future__ import annotations

from isa_bench.adapters import Base, finish, linux_boundary
from isa_bench.state import GPRS, MASK64, TF, ArchState, PageMemory, flags_of


class Adapter(Base):
    family = "unicorn"

    def load(self):
        import unicorn
        from unicorn import x86_const as X

        self.U, self.X = unicorn, X
        self.metadata = {**self.base_metadata(), "parameter_count": 0, "learned_records": 0,
                         "implementation": f"unicorn {unicorn.__version__}"}

    def run(self, program_dir, trace, max_steps):
        U, X = self.U, self.X
        image = PageMemory(program_dir / "mem.bin")
        uc = U.Uc(U.UC_ARCH_X86, U.UC_MODE_64)
        for page, data in image.pages.items():
            uc.mem_map(page, 0x1000, U.UC_PROT_ALL)
            uc.mem_write(page, bytes(data))
        ids = [getattr(X, f"UC_X86_REG_{name}") for name in GPRS]
        first = trace.state(0)
        for rid, value in zip(ids, first.regs):
            uc.reg_write(rid, value)
        uc.reg_write(X.UC_X86_REG_EFLAGS, trace.initial_eflags)
        trapped = []
        uc.hook_add(U.UC_HOOK_INSN, lambda u, _d: (trapped.append(1), u.emu_stop()), None, 1, 0, X.UC_X86_INS_SYSCALL)

        class Mem:
            def read(self, addr, n):
                try:
                    return bytes(uc.mem_read(addr, n))
                except U.UcError:
                    return b""

        cmp, stdout, stop, rip = self.comparator(trace), bytearray(), None, first.rip
        i = 0
        for i in range(max_steps):
            if self.time_up(i):
                stop = {"kind": "wall_limit"}
                break
            window = image.fetch(rip)
            trapped.clear()
            try:
                uc.emu_start(rip, MASK64, count=1)
            except U.UcError as error:
                stop = {"kind": "fault", "detail": str(error)}
                break
            regs = [uc.reg_read(r) for r in ids]
            eflags = uc.reg_read(X.UC_X86_REG_EFLAGS)
            if trapped:
                rip_after = rip + 2
                stop, regs = linux_boundary(regs, rip_after, Mem(), stdout, rflags=(eflags | TF))
                if stop is not None:
                    break
                for rid, value in zip(ids, regs):
                    uc.reg_write(rid, value)
                uc.reg_write(X.UC_X86_REG_RIP, rip_after)
            rip = uc.reg_read(X.UC_X86_REG_RIP)
            cmp.observe(i, ArchState(rip, regs, flags_of(eflags)), window=window)
        else:
            stop = {"kind": "step_limit"}
        return finish({"stop": stop, "stdout_hex": bytes(stdout).hex(), "steps_run": i + 1,
                       "exit_code": stop.get("exit_code")}, cmp, trace)
