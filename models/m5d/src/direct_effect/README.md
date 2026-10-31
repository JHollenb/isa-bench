# Direct-effect learners (x86-de64-v1)

The candidate-selection track learns which of 33 families an instruction belongs to while the shared PAN runtime parses operands, computes results and flags, and stores registers. This track removes all three. A model receives the 15-byte instruction window at its own RIP and the architectural state, and emits the complete next state: all eight legacy GPRs (64 bits each), CF/ZF/SF/OF, the instruction length (hence next RIP) and a SYSCALL terminal decode. It also emits decode fields (destination, source, width, immediate value) that are checked but never consumed by any runtime.

**Authorities.** Labels come from Unicorn 2.1.4 single-step emulation. Counter programs and a consumer-bug ELF are traced on the real CPU with [`native_asm/trace_steps.c`](../native_asm/trace_steps.c) (ptrace single-step, all GPRs and EFLAGS). The PAN typed adapter is a shadow checker only; it never repairs a state. It has two hardware-confirmed discrepancies ([flags](../docs/runtime-add-flag-discrepancy.md), [C7 sign extension](../docs/runtime-mov-c7-sign-extension.md)).

**Supplied (reported, not semantics).** Bytes and register values are presented as bits; the register file is laid out as 64 bit-position tokens (LSB first), each holding bit i of all eight GPRs; instruction fetch reads memory at the model-emitted RIP; after the model emits SYSCALL, the exit service reads the model's own RDI. There is no parser, ALU, flag routine, write mux or candidate vocabulary.

**Forms.** x86-al8-v1 AL forms (MOV/ADD/SUB/XOR/AND/OR/CMP/TEST imm8, INC/DEC AL) plus `MOV r32,imm32`, `MOV r64,imm64`, `MOV r/m64,imm32` (C7), `ADD/SUB r64,imm8`, `ADD/SUB r32,imm8`, `ADD/SUB r64,imm32`, `XOR r32,r32`, NOP, SYSCALL. Discovery and feedback immediates have low byte ≢ 0 mod 4; the heldout-literal panel uses ≡ 0. Five (form, register) pairs never appear in discovery or feedback (heldout-register panel).

**Architectures** differ only in the mixer along the bit axis: `nn` (MLP-Mixer, no recurrence), `mamba` (causal selective scans, LSB→MSB), `hydra` (bidirectional scans). PAN's original networks consume supplied decoded opcode IDs and are not used here.

**C3a vs C3b.** In C3a ([`model.Executor`](model.py)) the model reads its own previously emitted register bits; every bit of every register, including untouched ones, is a model output. In C3b ([`model.LatentExecutor`](model.py)) no register value is ever shown to the model after program start: a learned encoder writes the initial registers into a ±1 latent code (16 channels per bit position, straight-through sign), each instruction maps code → code, and architectural registers are decoded from the code only for checking.

**Training** runs through Saturn's `DeterministicCheckpointController` (lexicographic: discovery loss, feedback rollout exactness, feedback state exactness), with every future and rollback receipt retained. Launch with `scripts/direct_effect.py launch [--smoke]`.

**Rosetta probes.** Linear probes (standardized logistic regression) on each mixer layer for the true carry/borrow into bit i and the running OR of result bits, against the pre-mixer input layer as control; for C3b, register bits from the latent code.

## x86-de64-v2 ([`v2.py`](v2.py), [`scripts/direct_effect_v2.py`](../scripts/direct_effect_v2.py))

Literal-diverse long training. Immediates cover every residue; the sealed heldout is 24 seeded low-byte VALUES (`v2.HELD_BYTES`, never 0/1/8/60, which the counter and stdout programs use) plus seven (form, register) pairs (`v2.HELDOUT_PAIRS`). Added encodings: no-REX `C7 /0` (`mov_r32_c7`), register-to-register MOV (`89`, `48 89` incl. `48 89 e6` = MOV RSI,RSP, `48 8B`), `48 31` XOR, and `MOV [RSP], r64` (`48 89 /r` + SIB 0x24). The executor gains store outputs (enable, 64-bit address, 64-bit value); in rollouts the harness writes those bytes to a memory dict at the model-emitted address, and after the model emits SYSCALL a Linux boundary performs write(1, buf, n) from that memory (RAX = n, RCX = return RIP) or exit(RDI & 0xff). That makes the stdout ELFs runnable with model-computed state.

One mrun lease per architecture; Saturn `DeterministicCheckpointController` with 5,000-update futures, warmup + cosine learning rate computed inside each update from the restorable update counter (no drift on rollback), batches keyed by interval attempt, plateau stop (no ≥0.3-point feedback gain for four intervals) or an update cap. Every future's weights are retained and replayed afterwards for exactness and probes (carry-in, immediate bit, destination bit, per layer). Native ptrace traces: counter 16/128/1024/8192 (8192 is run only if 1024 is exact) and stdout 16/128/1024. Aggregate with `scripts/direct_effect_v2_report.py`.

## Round 3 options (`scripts/direct_effect_v2.py`)

- `--aligned`: register-bit token *i* also receives window bit 8·*o* + *i* for byte offsets *o* = 0..7, plus a soft offset choice from the decoded instruction (`Executor(aligned=True)`). Layout only; the offset, sign extension and everything downstream are learned.
- `--relptr`: an extra read whose logits depend only on (source bit − output bit) through a profile chosen from the decoded instruction (`Executor(relptr=True)`); no aligned slots.
- `--wide-large F --wide-boost K`: training-data mix only (panels keep the v2 distribution): fraction of full-width random immediates in ≥4-byte forms, and K× sampling of wide-immediate forms.
- `--margin-se K`: Saturn noise-aware promotion (k standard errors of the paired per-row difference on the feedback panel, rollout programs and flag rows; within-band candidates promoted, reference anchored to the last strict improvement). Needs the controller from branch `feat/noise-aware-promotion`, sealed into the payload via `--controller` (default: `~/domains/.worktrees/saturn-noise-promotion/...`). Locally, `ISA_SATURN_CONTROLLER=<path>` selects it.
- `--schedule-on attempts`: the cosine follows attempted updates, so it always reaches its floor; `--patience 99` disables the plateau stop.
- `--eval-from DIR`: evaluate a retained `<arch>/{model.pt,training.json,futures/}` without training (salvage after a lease is killed during evaluation).

Every evaluated future is kept; the child scores the accepted state, the last future and the top three by feedback on the sealed panels (`result["candidates"]`), plus a carry-run panel (`ADD RDI,1` with L trailing ones). Diagnostics: `scripts/direct_effect_r3_wide_diag.py` (offset choice, per-byte destination accuracy and read hits) and `scripts/direct_effect_r3_program_diag.py` (form and state field at each random program's first divergence). Results: `outputs/direct-effect-r3-report/`.
