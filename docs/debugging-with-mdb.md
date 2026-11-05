# Debugging learned x86 executors with MDB

isa-bench tells you *where* a learned CPU goes wrong: which step, which register, which bit. MDB (the model debugger in `../saturn-pub/packages/mdb`) tells you *why*. MDB turns that one wrong forward pass into a program you can step through. You can set breakpoints at its MLP boundaries, fork it, edit any intermediate value and replay the rest. You can also check the result against a symbol table that was recovered for the same checkpoint.

Here is what that looks like on real failures. Every result in this list was produced on the laptop for this guide, one forward at a time:

- **One untrained opcode, three different guesses.** The second instruction of every compiled `dev` program is `and $-16,%rsp`, an encoding none of the direct-effect models was trained on. It is the first divergence on all 10 of them. MDB shows each model decodes the operands perfectly (destination RSP, immediate `0xfffffffffffffff0`, length 4) and only guesses the operation wrong. The Mamba runs it as `add`; the NN and Hydra run it as `sub`.
- **Two numbers out of 256.** Fork the Mamba's forward at the decoder's exit and paste just 2 of the 256 decoder coordinates from a `sub` encoding. The prediction flips from `RSP-16` to `RSP+16`. The same 2 coordinates do *not* turn a real `sub` into `add`, so they are a tipping point, not an opcode switch. MDB shows you both.
- **A symbol table that predicts compiled C.** The R3 Mamba's recovered symbol table describes `add $1,%reg` as a 3-bit raw-byte route plus a bit-serial XOR/AND carry. It predicts the model's output exactly on 5/5 `add $1` instructions taken from real isa-bench programs with their native register state. The model also reproduces all 3 of the table's recorded exceptions bit for bit. One of its flag rules fails on a fresh input where the model is actually right.
- **Finding the bit that's wrong.** One recorded failure (RDI bit 32 set, logit +0.07) is fixed by scaling a single site, `mixers.2.out_proj`, by 0.9. The fix reproduces on 2 of 3 failing inputs and breaks none of 5 correct ones.
- **Predicting the wrong answer before the model runs.** MDB's independent FP64 equations for the whole forward forecast `RSP − 16, CF = 1` for `and $-16,%rsp`. That is the model's wrong answer, not x86's. The forecast is sealed before the model is allowed to run, and the model then matches every head decision on all 10 inputs tried (real `and` and `add $1` steps plus the table's exceptions). The worst logit error was 4e-5. A forecast that tries to call the model before the seal is refused.
- **Selectors, too.** The six learned selectors pick an instruction family instead of computing state. `just debug` now records the neural ones' forward at the refusing instruction. On `xor %rbp,%rbp`, the first instruction of every `dev` program, all four pick the wrong family; the right one is ranked 6th to 12th. Forking the readout shows the wrong pick lives in the byte-scan half, not the numeric features. The structural selectors get one instruction further and stop at `and $-16,%rsp`. That is a gap in the PAN consumer, not in the selector.
- **Save today, resume next week.** A session with an edited branch saves to a content-addressed frontier (15.5 MB). It resumes in a fresh process at the same cursor, with the same outputs and the full work history, without re-running the prefix.

**Labels.** *Measured here* means run while writing this guide on 2026-10-30 or 2026-10-31: laptop CPU, FP32, one forward at a time, outputs pasted and trimmed. *Reported* means taken from the cited doc or post. Nothing below is labelled measured unless it was run.

---

## Contents

0. [What MDB covers per model](#0-what-mdb-covers-per-model)
1. [Quickstart](#1-quickstart)
2. [The mental model in one screen](#2-the-mental-model-in-one-screen)
3. [Use case: why did the model mispredict step N?](#3-use-case-why-did-the-model-mispredict-step-n)
4. [Use case: which feature drove it? Fork and edit](#4-use-case-which-feature-drove-it-fork-and-edit)
5. [Use case: what does the symbol table predict, and was it right?](#5-use-case-what-does-the-symbol-table-predict-and-was-it-right)
6. [Use case: run the model as a program, instruction by instruction](#6-use-case-run-the-model-as-a-program-instruction-by-instruction)
7. [Use case: compare checkpoints](#7-use-case-compare-checkpoints)
8. [Use case: save a frontier, resume later, audit the journal](#8-use-case-save-a-frontier-resume-later-audit-the-journal)
9. [Use case: seal a forecast of the model's answer, then let it run](#9-use-case-seal-a-forecast-of-the-models-answer-then-let-it-run)
10. [Use case: why did a selector refuse?](#10-use-case-why-did-a-selector-refuse)
11. [Cheat sheet](#11-cheat-sheet)
12. [Limits, gotchas, next steps](#12-limits-gotchas-next-steps)
- [Appendix: the scripts behind the measured examples](#appendix-the-scripts-behind-the-measured-examples)

---

## 0. What MDB covers per model

| model | what it is | MDB today | what you get |
|---|---|---|---|
| `de_mamba` | R3 direct-effect Mamba (3 scan mixers over the 64 bit positions), checkpoint `9399e921…` | **Full.** `just debug` writes a recipe (571 ATen ops); console, Python API, instruction driver. | Recovered **symbol table** (`../isa/outputs/moonshot-r3-mamba/`). The debug report links it automatically when the hash matches. |
| `de_nn` | R3 direct-effect MLP-mixer, checkpoint `30caebe4…` | **Full** (286 ops, 11 MLP sites). | No symbol table. The same checkpoint carries the reported NN value-edit results (§9, "Elsewhere in MDB"). |
| `de_hydra` | R3 direct-effect bidirectional scans, checkpoint `0dc718f4…` | **Full** (655 ops). | No symbol table. |
| `m5d` | M5d-final v3: a byte-trie of induced records, not a neural forward | **No.** There is no tensor program to replay. | `just debug` report: the abstain point and the native disassembly. |
| `sel_nn`, `sel_mamba`, `sel_hydra`, `sel_pan_isa` | Neural selectors over the frozen PAN runtime: pick 1 of 33 families; the consumer computes the effect | **Full** for the runtime head's forward at the refusing instruction (8 / 106 / 134 / 11 ops). | Report: top-5 families vs the family the typed adapter executes, and its rank (§10). |
| `sel_fieldlm`, `sel_spf` | Structural selectors: route tables (FieldLM records, SPF phase planes) | **No.** There is no tensor forward. | Report: the route key looked up, and the learned routes for that opcode (§10). |
| `unicorn` | Emulator control | n/a | Always 100% trace-exact; if it isn't, the harness is wrong. |

The three direct-effect checkpoints share one ABI: a 15-byte window, 8 GPRs as 8×64 bits and CF/ZF/SF/OF in; 11 heads out. So every recipe, script and decoder in this guide works on all three. Model-specific site names come from the recipe's own catalog.

## 1. Quickstart

The commands below run from the repo root. Single forwards and console replays are fine on the laptop. Anything bigger (sweeps, evals, training) goes to Beast through mrun (`CLAUDE.md` rule 1). Keep at most one heavy local Python process at a time.

```sh
# convenience wrapper (the examples below were run in this expanded form)
mdb() { PYTHONPATH=../saturn-pub/packages/mdb/src ../common/bin/common exec python -m mdb "$@"; }
```

**Step 1. Get a first-divergence recipe.** This needs eval cells under `outputs/eval/<model>/<suite>/cells/`. It records one forward locally. *Measured here:* 18.5 s wall, mostly environment start-up.

```text
$ just debug de_mamba dev fib-O2
{"debug_dir": "debug/de_mamba/dev/fib-O2", "status": "wrong_output", "exact_prefix_steps": 1,
 "mdb": {"root_recipe": "debug/de_mamba/dev/fib-O2/root-recipe",
         "checkpoint_sha256": "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e",
         "symbol_table": "../isa/outputs/moonshot-r3-mamba/catalog.json", "operations": 571}}
```

`debug/de_mamba/dev/fib-O2/report.md` (trimmed):

```text
Status **wrong_output**; exact prefix 1 of 150753 native steps.
## First divergence
Step 1 at 0x401003, window `4883e4f0e8a403000089c7b83c0000`.
| field | model | native |
| CF | 1 | 0 |
| RSP | 0x7fffffffe350 | 0x7fffffffe360 |
{"predicted_length": 4, "store": false, "syscall": false}
## Native code around RIP
     401000:	xor    %rbp,%rbp
=>   401003:	and    $0xfffffffffffffff0,%rsp
     401007:	call   4013b0 <cmain>
```

**Step 2. Open it.** The interactive console reads one command per line. The prompt shows the focused lane and the default unit:

```text
$ printf 'break module:post.1 exit --id p1\ncontinue\ncall {"operation":"capabilities"}\nquit\n' \
    | just mdb debug/de_mamba/dev/fib-O2/root-recipe
(discovery-v1 | lane_00 | unit=law) { ... "installed_id": "p1", ... }
(discovery-v1 | lane_00 | unit=law) { ... "replay_native_operations": 369 ... }
(discovery-v1 | lane_00 | unit=law) {
  "act_value_specs": ["zeros_like", "scale"],
  "checkpoint_support": ["live", "native-durable"],
  "default_unit": "law",
  "direct_eager_pause": false,
  "instruction_drivers": ["r3-predicted-length"],
  "motion_units": ["op", "math", "law", "update", "forward", "instruction"],
  ...
{"commands_executed": 3, "handler_hits": 0, "ok": true, "output": "debug/de_mamba/dev/fib-O2/root-recipe/../console"}
```

> `just mdb` used to write to a fixed `<recipe>/../console` folder, so a second run on the same recipe failed with `FileExistsError` (*measured here*). It now writes a timestamped `console-<stamp>` folder each time. MDB itself still refuses to overwrite an output folder, so a hand-run `mdb discovery replay` needs a new `--output` every time.

**Step 3. Script it.** Anything you type interactively also works as `--command` (repeatable) or as a `--script` file. Every run writes `result.json` plus a hash-chained journal:

```text
$ mdb discovery replay debug/de_mamba/c_apps/medium_sort-Os/root-recipe \
      --command status --command catalog --output debug/scratch/ex01
{"commands_executed": 2, "handler_hits": 0, "ok": true, "output": "debug/scratch/ex01"}
```

`catalog` lists 59 structural sites with their ATen entry/exit cursors. These are the lines that matter, summarized from `result.json`:

```text
program:8303c261…  (program binding; site ids are bound to this recording)
mlp_001 mlp decoder 2 8          (window bits -> 256-d instruction vector)
mlp_002 mlp post.0 228 238
mlp_003 mlp post.1 359 369
mlp_004 mlp post.2 490 500
mlp_005 mlp flags 541 546
affine_016 affine mixers.0.out_proj 223 227      affine_022 … mixers.1.out_proj 354 358
affine_028 affine mixers.2.out_proj 485 489
affine_038 affine length 552 554                 affine_042 affine store_flag 566 568
update_001/002/003 residual 238/369/500 -> 239/370/501
```

**Step 4. Read the outputs.** `inspect outputs` returns the 11 raw heads (about 17 KB of logits). A ~35-line decoder turns them into x86 state; the [appendix](#appendix-the-scripts-behind-the-measured-examples) has it, and it uses no torch:

```text
$ python3 debug/scratch/decode.py debug/scratch/tour/result.json
{"length": 4, "length_logits_top3": [[29.25, 4], [5.23, 7], [0.2, 5]],
 "regs": {"RSP": "0x7fffffffe350", ...}, "flags": {"CF": 1, "ZF": 0, "SF": 0, "OF": 0},
 "store": false, "store_logit": -17.54, "syscall": false,
 "dest": "RSP", "src": "RSP", "width": 64, "imm": "0xfffffffffffffff0"}
```

The replay reproduces the eval cell exactly (RSP `…e350`, CF 1, length 4). The auxiliary heads (`dest`, `src`, `width`, `imm`) tell you what the model *decoded*, which the eval cell never shows.

## 2. The mental model in one screen

- **One forward = one instruction.** Inputs: `window_bits [1,120]` (15 bytes, LSB first), `reg_bits [1,8,64]` (RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI) and `flags [1,4]` (CF, ZF, SF, OF). Outputs: `regs` and `flags` logits (bit = logit > 0), `length` (argmax + 1), `store`/`store_addr`/`store_value`, `syscall`, and the aux heads `dest`, `src`, `width`, `imm`. There is no memory-read input and no state carried between calls. Source: `saturn-pub/packages/mdb/docs/ISA-DISCOVERY-DEBUGGER.md`.
- **A recipe is a sealed recording plus its roots, not a paused process.** `root-recipe/` holds `trace.json` (the ATen program: 571 events and 676 values for the Mamba) and `roots.pt` (inputs and weights), hashed in `capture.json`. Every replay starts at event 0 and re-executes the original primitives (`native_frontier_restorable: false`). Source: `src/mdb/discovery_cli.py`.
- **Sites and selectors.** The catalog names sites generically by kind: `mlp_00N`, `affine_0NN`, `gelu_…`, `normalize_…`, and `update_…` (residual). Address them as `mlp_003`, `module:post.1`, `mlp_003:output` / `:input`, or a raw SSA id `v000430`. `alias name = selector` adds your own names. Site and value ids are bound to one recording's `program:<hash>`; use the names, not the numbers, across recordings.
- **Clocks.** `op` (one ATen primitive), `math` (a stage such as affine or GELU), `law` (a whole MLP, before the residual add), `update` (the residual add), `forward` (all heads) and `instruction` (decode and commit, which needs a driver; §6). Motions: `step`, `next`, `into`, `finish`, `until`, `continue`. Breakpoints are checked between primitives.
  Each unit moves the cursor a different distance (*measured here*, `debug/scratch/scripts/clocks.mdb` on the `and` recipe; ops = ATen primitives executed):

  ```text
  step op          -> aten_next 1,   ops run 1    [mul]
  step math        -> aten_next 4,   ops run 3    [sub, t, addmm]         decoder.0 affine
  step math        -> aten_next 5,   ops run 1    [gelu]
  step law         -> aten_next 8,   ops run 3    (rest of the decoder MLP)
  into law         -> aten_next 228, ops run 220  (head prep + mixer 0, to post.0's entry)
  step math        -> aten_next 229, ops run 1    [native_layer_norm]
  finish law       -> aten_next 238, ops run 9    (post.0 MLP output, residual not yet added)
  step update      -> aten_next 239, ops run 1    [add]                   the residual
  until mlp_004    -> aten_next 500, ops run 261
  step forward     -> aten_next 571, ops run 71   stop [motion_target, forward_complete]
  ```
- **History.** `capture NAME` makes a cut. `restore NAME` returns to it. `fork NAME` + `select NAME` branches from the current head (or from any cut via `call`). Branches are independent replays that share weights.
- **Edits (Acts).** `act` replaces or adds to a *produced, live, not-yet-consumed* tensor. Use `value_spec` `zeros_like` or `scale`, or a literal `value`. Roots (inputs and **weights**) are read-only, so the discovery console edits activations, not parameters. Source: `src/mdb/discovery.py`, `src/mdb/discovery_runtime.py`.
- **Not interpretation.** Names like `mlp_003` say where a computation happens, not what it means. Semantic claims come from the symbol table's panels or from your own interventions (§5).

## 3. Use case: why did the model mispredict step N?

**The symptom (read from the eval cells `outputs/eval/*/dev/cells/*.json`, produced on Beast by mrun; not re-run here).** All three direct-effect models diverge at step 1, `0x401003`, on all 10 `dev` programs. The instruction is `and $-16,%rsp` (`48 83 e4 f0`). RSP is already 16-aligned (`…e360`), so native leaves it unchanged with CF = 0.

| model | predicted RSP | predicted CF | length |
|---|---|---|---|
| de_mamba | `0x7fffffffe350` (−16) | 1 | 4 |
| de_nn | `0x7fffffffe370` (+16) | 1 | 4 |
| de_hydra | `0x7fffffffe370` (+16) | 1 | 4 |
| native | `0x7fffffffe360` | 0 | 4 |

**Step 1. Did the model decode the instruction?** Yes. The decoded outputs above show `dest RSP`, `width 64`, `imm 0xfffffffffffffff0` and length 4, with a large margin (length logit 29.25 against 5.23 for the runner-up). The operands are right; only the operation is wrong. `RSP + (-16)` with CF = 1 is exactly what `add $-16,%rsp` produces.

**Step 2. Is `and r64, imm8` something the model ever saw?** No. The R3 training forms are in `models/direct_effect/src/direct_effect/profile.py` (`FORMS`) and `v2.py`: `add`/`sub` `r64, imm8` (`83 /0`, `83 /5`), AL8 logic ops, `mov` forms, `xor r32,r32` and `mov [rsp],r64`. There is no `83 /4`. So the question is not "what bug?" but "what does the model map an unseen ModRM `/4` to?"

**Step 3. Ask all eight `/r` values on all three checkpoints** (*measured here*, 24 single forwards, `lab.py sweep_ext`). This keeps the real window and the native before-state and changes only the ModRM reg field:

```text
{"window": "4883e4f0e8a403000089c7b83c0000", "RSP_before": "0x7fffffffe360", ...}
{"model": "de_mamba", "/0 add": "RSP-16 CF=1 len=4", "/1 or": "RSP-16 CF=1 len=4", "/2 adc": "RSP+0 CF=1 len=4",
  "/3 sbb": "RSP+0 CF=1 len=4", "/4 and": "RSP-16 CF=1 len=4", "/5 sub": "RSP+16 CF=1 len=4",
  "/6 xor": "RSP+0 CF=1 len=4", "/7 cmp": "RSP+0 CF=1 len=4"}
{"model": "de_nn",    ... "/4 and": "RSP+16 CF=1 len=4", "/5 sub": "RSP+16 CF=1 len=4", ...}
{"model": "de_hydra", ... "/4 and": "RSP+16 CF=1 len=4", "/5 sub": "RSP+16 CF=1 len=4", ...}
```

The two trained encodings (`/0` add, `/5` sub) are right on every checkpoint. On unseen encodings the models extrapolate: `/1` behaves like add, and `/2 /3 /6 /7` leave RSP alone. `/4` is where the checkpoints disagree. `/4` (`100`) is one bit from `/0` (`000`) and one bit from `/5` (`101`); the Mamba lands on the add side and the NN and Hydra on the sub side.

**Step 4. Which inputs can a head see at all?** `mdb forward-symbols` reads `trace.json` without torch and lists each root's `output_ancestry`, i.e. which heads it can reach (*measured here*):

```text
$ mdb forward-symbols debug/de_mamba/dev/fib-O2/root-recipe/trace.json --readable
forward readable program: 676 values, 571 operations
roots:
  v000000 = input ; tensor shape=[1, 120] ... output_ancestry=/dest,/flags,/imm,/length,/regs,/src,/store,/store_addr,/store_value,/syscall,/width
  v000025 = input ; tensor shape=[1, 8, 64] ... output_ancestry=/flags,/imm,/regs,/store_addr,/store_value
$ mdb forward-symbols debug/de_mamba/dev/fib-O2/root-recipe/trace.json --value v000012
v000012: operation-output  shape=[1,256]  dtype=torch.float32
producer: event 7 aten.gelu.default  @ decoder.3
uses: [9, 11, 37, 80, 85, 510, 540, 551, 553, 555, 567]
```

The register input `v000025` never reaches `length`, `store`, `syscall`, `width`, `dest` or `src`. Those decisions are functions of the window bytes alone, routed through the decoder's 256-d output `v000012`, which feeds `dest`, `src`, `length`, `store_flag`, `syscall`, `width_head`, `offset_head`, the flag path, and two query paths into the register computation (its `uses` list). That is a structural fact read from the recording, not an inference. It also tells you where to intervene next: the decoder vector.

**The same recipe for `push %rax`.** `medium_sort-Os` dies at step 0 on `push %rax` (`50`). The model predicts length 10 and no store, then store-faults later. Decoding its recipe shows a very different picture from the confident `and` (*measured here*):

```text
$ mdb discovery replay debug/de_mamba/c_apps/medium_sort-Os/root-recipe \
      --command "step forward" --command "inspect outputs" --output debug/scratch/push
{"length": 10, "length_logits_top3": [[6.87, 10], [6.02, 3], [5.52, 6]], ... "RSP": "0x7fffffffe350",
 "flag_logits": [-11.16, -12.24, -10.78, -10.72], "store": false, "store_logit": -8.58,
 "dest": "RAX", "src": "RSI", "width": 64, "imm": "0x0"}
```

The length head is nearly flat (6.87 / 6.02 / 5.52) instead of 29 against 5. *Interpretation:* that is the signature of an input far outside training. Push is not one of the R3 forms at all, while `and` is one bit away from two trained forms.

## 4. Use case: which feature drove it? Fork and edit

Since the operation decision lives in the decoder vector (§3 step 4), test that directly. Pause the `and` forward at the decoder's exit, fork, replace the vector, and replay the rest.

**From the console (*measured here*).** This script uses only built-in commands. It breaks at the decoder's exit, captures there, finishes the original forward, then forks a branch with the decoder zeroed and saves that branch durably:

```text
# debug/scratch/scripts/and_tour.mdb
interface mdb-discovery-v1
alias dec = module:decoder
break dec exit --id dec_exit
continue
capture at_decoder
inspect dec:output
call {"operation":"break","payload":{"action":"disable","id":"dec_exit"}}
step forward
inspect outputs
restore at_decoder
fork no_decoder
select no_decoder
call {"operation":"act","payload":{"selector":"dec:output","value_spec":{"kind":"zeros_like"}}}
step forward
inspect outputs
save-frontier debug/scratch/frontier-store
```

```text
$ mdb discovery replay debug/de_mamba/dev/fib-O2/root-recipe --script debug/scratch/scripts/and_tour.mdb --output debug/scratch/and_tour
{"commands_executed": 15, "handler_hits": 0, "ok": true, "output": "debug/scratch/and_tour"}
```

Per-command trace (summarized from `result.json`: line, command, stop causes, ATen cursor, branch, receipt):

```text
4  continue                 stop [breakpoint:dec_exit]   aten_next 8    main
5  capture at_decoder       cut_000003                   aten_next 8
8  step forward             [motion_target, forward_complete]  aten_next 571
10 restore at_decoder                                    aten_next 8
11 fork no_decoder / 12 select no_decoder                aten_next 8    no_decoder
13 act dec:output zeros_like   receipt value_id v000012
14 step forward             [motion_target, forward_complete]  aten_next 571  no_decoder
16 save-frontier            durable cut edfbec91…, admitted_bytes 15529316
```

Decoded heads of the two branches:

```text
main:       length 4, RSP 0x7fffffffe350, CF 1, ZF 0, store false, syscall false, dest RSP, imm 0xfffffffffffffff0
no_decoder: length 1, RSP 0x7fffffffe360, CF 0, ZF 1, store false, syscall true,  dest RSP, src RAX, imm 0x0
```

With no instruction vector, the model leaves the registers alone, predicts length 1 and raises `syscall`. That is a different wrong instruction, not a fix.

**Breakpoint handlers.** The example script shipped with MDB, `../saturn-pub/packages/mdb/examples/discovery/post1.mdb`, runs unchanged on any of these recipes. It installs an `on-hit` handler that captures and inspects at `post.1`'s exit, then forks a branch with that MLP's output zeroed (*measured here*):

```text
$ mdb discovery replay debug/de_mamba/dev/fib-O2/root-recipe \
      --script ../saturn-pub/packages/mdb/examples/discovery/post1.mdb --output debug/scratch/post1
{"commands_executed": 14, "handler_hits": 1, "ok": true, "output": "debug/scratch/post1"}
zero_post1 branch: length 4, RSP 0x7fffffffe371, CF 1   (baseline RSP 0x7fffffffe350)
```

Removing post.1's law corrupts several low RSP bits but leaves the length alone. That fits §3: the length head reads the decoder vector directly and never passes through the post-MLPs.

**Transplants and coordinate pastes (Python API, *measured here*, `lab.py edit_and`).** The console has no "take this value from another recording" command yet (Saturn request `mdb-discovery-donor-lane-acts`), so this part uses the same `DiscoverySession.call()` API the console sits on. Donor decoder vectors come from MDB recordings of the same window with ModRM `/0` and `/5`, stepped to `mlp_001`'s exit and read. Each arm is a fork from the cut at the decoder's exit, then an Act, then `step forward`:

```python
s = session_for(recipe)                          # DiscoverySession + load_capture(root-recipe)
prepare_cuts(s, ["mlp_001"])                     # break at exit, capture cut "mlp_001", disarm
arm(s, "sub_transplant", "mlp_001",              # fork + select + act + step forward + decode
    {"selector": "mlp_001:output", "value": donors[5].tolist()})
```

```text
{"cuts": [["mlp_001", 8]]}
{"recipe_decoder_equals_/4_donor": true}
{"decoder_dim": 256, "l2(/5-/4)": 9.073, "l2(/0-/4)": 9.499,
 "top8_coords_/5-/4": [[220, 2.97], [152, 2.72], [20, 2.68], [89, -2.38], [85, -2.05], [73, 1.95], [176, -1.86], [158, -1.66]]}
{"arm": "baseline (no edit)",        "RSP": "0x7fffffffe350", "RSP_delta": -16, "CF": 1, "len": 4, "dest": "RSP"}
{"arm": "decoder := /5 (sub) vector", "RSP": "0x7fffffffe370", "RSP_delta": 16,  "CF": 1, "len": 4, "act": "v000012"}
{"arm": "decoder := /0 (add) vector", "RSP": "0x7fffffffe350", "RSP_delta": -16, "CF": 1, "len": 4, "act": "v000012"}
{"arm": "top-1 coords of /5 pasted",  "RSP_delta": -16}
{"arm": "top-2 coords of /5 pasted",  "RSP_delta": 16}
{"arm": "top-4 … top-128 coords of /5 pasted", "RSP_delta": 16}      (each k in 4, 8, 16, 32, 64, 128)
{"arm": "decoder zeroed", "RSP": "0x7fffffffe360", "RSP_delta": 0, "CF": 0, "len": 1, "imm": "0x0"}
```

The operation choice follows the transplanted vector: the `/5` vector gives sub and the `/0` vector gives add, with the window, registers and flags unchanged. Pasting just coordinates 220 and 152 from the `sub` vector is enough to tip `and` from add to sub.

**Does that make 220/152 an "add vs sub" switch? Test it before naming it** (*measured here*, `lab.py selector_check`). This takes the real `sub $0x18,%rsp` at `medium_sort-Os` step 3 and pastes in the `add` encoding's values:

```text
{"instr": "sub $0x18,%rsp @0x401004", "step": 3, "window": "4883ec1889c28b1495002040008954",
 "RSP_before": "0x7fffffffe350", "native_RSP_after": "0x7fffffffe338"}
{"arm": "baseline",                 "RSP": "0x7fffffffe338", "RSP_delta": -24, "len": 4}
{"arm": "paste add coord 220",      "RSP_delta": -24}
{"arm": "paste add coord 152",      "RSP_delta": -24}
{"arm": "paste add coords 220+152", "RSP_delta": -24}
{"arm": "paste whole add vector",   "RSP": "0x7fffffffe368", "RSP_delta": 24, "len": 4}
```

No. On a trained `sub`, those two coordinates move nothing, though the whole `add` vector flips it. The honest reading is that the `/4` input sits near the add/sub decision boundary, and two coordinates were enough to push it across. This is one input, one checkpoint; a coordinate sweep over many windows is a Beast job, not a laptop loop.

## 5. Use case: what does the symbol table predict, and was it right?

**What the table says (reported).** `../isa/outputs/moonshot-r3-mamba/symbol-table.md` and `catalog.json` hold 14 entries for checkpoint `9399e921…`. They join source equations (the five MLPs and the scan) to finite measured response programs. The part that predicts outputs is a small program measured on windows like `48 83 c7 01` (x86 `add $1,%rdi`; the table itself assigns no mnemonic):

| piece | rule | evidence (reported) |
|---|---|---|
| route | the destination slot = raw window bits 16–18 (byte 2, ModRM rm field) | 32/32 held-out route bits; destination head agrees 32/32. `../isa/outputs/response-route-r3-mamba/SUMMARY.md` |
| word response | bit-serial over the 64 bit positions: `y = q ⊕ x`, `q_next = q·x`, `q0 = 1` (an incrementer) | 93/96 held-out target words: 32/32, 32/32 and 29/32 at prefix cuts 11, 17 and 31. 3 exceptions retained. `../isa/outputs/response-quotient-r3-mamba/SUMMARY.md` |
| flags, length | CF = 0, ZF = 0, SF = output bit 63, OF = 0; length 4 | 8/8 four-step own-state steps (registers, flags, length). Alternatives retained. `../isa/outputs/symbolic-loop-r3-mamba/SUMMARY.md` |

The table names no instruction semantics and claims nothing outside its panels. That makes it a falsifiable program: run it next to the model on inputs it never saw.

**The test (*measured here*, `lab.py symtab`).** `symtab_program()` in the appendix is the three rows above as 15 lines of Python. It runs before the model. The test takes real `add $1` instructions from compiled isa-bench programs: the 15-byte fetch window from `mem.bin` (real trailing code bytes, not the panel's zero or repeated tails) and the native before-state from `trace.npz`. Then it adds the table's own three exceptions and five fresh long-carry inputs:

```text
case                               slot in                   symtab               model                exact  flags model vs symtab
fib-O2@0x40121a step 236           RDI  0x1                  0x2                  0x2                  yes    = (native 0x2, flags =)
checksum-O2@0x401183 step 309023   RDI  0x403000             0x403001             0x403001             yes    = (native 0x403001)
fib-O2@0x401236 step 5255          RAX  0xe8                 0xe9                 0xe9                 yes    = (native 0xe9)
checksum-O2@0x4010f3 step 39016    RDX  0x403000             0x403001             0x403001             yes    = (native 0x403001)
checksum-O2@0x4010ba step 11       RCX  0x403000             0x403001             0x403001             yes    = (native 0x403001)
sealed panel held-081              RDI  0x6e1f6958ffff7fff   0x6e1f6958ffff8000   0x6e1f6959ffff8000   bit 32
sealed panel held-085              RDI  0x6e1f695affff7fff   0x6e1f695affff8000   0x6e1f695bffff8000   bit 32
sealed panel held-087              RDI  0x6e1f695bffff7fff   0x6e1f695bffff8000   0x6e1f695fffff8000   bit 34
fresh RDI=2^31-1 / 2^32-1 / 2^33-1 / 2^47-1                                                          yes ×4
fresh RDI=2^63-1                   RDI  0x7fffffffffffffff   0x8000000000000000   0x8000000000000000   yes    model OF=1, symtab OF=0
```

(The table is condensed from `lab.py symtab` JSON lines. In every row the seven other registers were exactly equal and the model's length was 4.)

What this shows:

1. **The table predicts compiled C.** 5/5 real-program `add $1` instructions match the table, the model and native: target word, other registers and flags. Three of them (RAX, RDX, RCX) go through the same 3-bit route, though the q-law panel varied only RDI.
2. **It knows its own failures.** All 3 sealed exceptions reproduce bit for bit (bits 32, 32, 34). The table recorded them rather than refitting them away, so they are known model errors, not table errors.
3. **One rule is wrong outside its panel.** At RDI = 2^63−1, x86 sets OF (signed overflow). The model predicts OF = 1, which is correct; the table's "OF is constant 0" rule says 0. The symbol-loop summary kept alternatives for exactly this reason; this input separates them. Saturn request `mdb-computation-hypotheses-and-causal-naming` (upvoted while writing this guide) would keep counterexamples like this attached to the rule.

**Drill down on an exception: where does the stray bit 32 come from?** `lab.py bit32` records held-081's exact inputs as a new recipe (`debug/scratch/held-081/root-recipe`). It sets breakpoints at the exits of the three mixer `out_proj` affines and the three post-MLPs, captures a cut at each, then forks two arms per site: zero the output, and scale it by 0.9. The bit-32 logit in each arm (*measured here*):

```text
{"cuts": [["affine_016", 227], ["mlp_002", 238], ["affine_022", 358], ["mlp_003", 369], ["affine_028", 489], ["mlp_004", 500]]}
{"arm": "baseline",                       "RDI": "0x6e1f6959ffff8000", "bit32_logit": 0.0716}
{"arm": "zero mixers.0.out_proj output",  "RDI": "0x35a2b15ac782c800", "bit32_logit": -7.6849}
{"arm": "x0.9 mixers.0.out_proj output",  "RDI": "0x6e1f6959ffff8000", "bit32_logit": 0.4905}
{"arm": "zero mlp_002 output",            "RDI": "0x6e1f6958ffff7ffe", "bit32_logit": -4.6893}
{"arm": "x0.9 mlp_002 output",            "RDI": "0x6e1f6959ffff8000", "bit32_logit": 0.0954}
{"arm": "zero mixers.1.out_proj output",  "RDI": "0x6e1f6958ffff7ffe", "bit32_logit": -5.5151}
{"arm": "x0.9 mixers.1.out_proj output",  "RDI": "0x6e1f6959ffff8000", "bit32_logit": 0.1045}
{"arm": "zero mlp_003 output",            "RDI": "0x6e1f6959ffff8000", "bit32_logit": 1.7049}
{"arm": "x0.9 mlp_003 output",            "RDI": "0x6e1f6959ffff8000", "bit32_logit": 0.268}
{"arm": "zero mixers.2.out_proj output",  "RDI": "0x6e1f6958ffff7fff", "bit32_logit": -3.7722}
{"arm": "x0.9 mixers.2.out_proj output",  "RDI": "0x6e1f6958ffff8000", "bit32_logit": -0.8373}
{"arm": "zero mlp_004 output",            "RDI": "0x6e1f6959ffffc800", "bit32_logit": 3.3167}
{"arm": "x0.9 mlp_004 output",            "RDI": "0x6e1f6959ffff8000", "bit32_logit": 0.3961}
```

Zeroing a site mostly wrecks other bits (zeroing `mlp_003` leaves the word as it was but raises bit 32's logit to 1.70). Of the gentle 0.9 scalings, only the last mixer's write (`mixers.2.out_proj`) pushes bit 32 negative, and that arm yields exactly the table's answer `0x6e1f6958ffff8000`. Every other 0.9 arm makes bit 32 *more* positive.

**Is that a repair or a coincidence? Same edit on every case** (*measured here*, `lab.py bit32_check`). Each case is recorded in memory, forked at `affine_028`'s exit, and compared with no edit and with 0.9:

```text
case                    symtab               baseline exact   x0.9 exact   other regs (x0.9)
held-081                0x6e1f6958ffff8000   no               yes          exact
held-085                0x6e1f695affff8000   no               yes          exact
held-087                0x6e1f695bffff8000   no               no (still bit 34)  exact
fib-O2@0x40121a         0x2                  yes              yes          exact
checksum-O2@0x401183    0x403001             yes              yes          exact
RDI=2^32-1, 2^47-1, 2^63-1                   yes ×3           yes ×3       exact
```

It fixes 2 of 3 known failures and breaks none of 5 correct cases. That is a lead, not a fix: 8 inputs, one factor, one checkpoint. It is consistent with what the computation-atlas census reports for this checkpoint, that mixer 2's write is where carries are committed and beyond depth 33 only layer 2 matters (`obsidian/blog/2026-10-30-000214-the-computation-atlas-prediction-programs-and-the-carry.md`). The next step is a factor × input sweep, which belongs on Beast through mrun.

## 6. Use case: run the model as a program, instruction by instruction

The discovery session can own an architectural driver (`mdb.discovery_isa.ISADriver`). It fetches 15 bytes at the model's own predicted offset, runs a forward, commits the predicted registers and flags, and advances by the predicted length. Breakpoints keep firing inside every forward. This is Python-only today (*measured here*, `lab.py driver`): it runs the first 24 bytes of `fib-O2`'s `_start` from the native initial state, with a breakpoint at `post.2`'s exit:

```python
s = DiscoverySession(budgets={"max_model_forwards": 8, "max_native_operations": 100_000})
s.add_lane("fib", factory, driver=ISADriver(code.hex(), regs, flags, max_instructions=3),
           reference_forwards=1, reference_operations=8192)
s.call("break", {"selector": "module:post.2", "edge": "exit", "id": "post2"})
s.call("motion", {"action": "step", "unit": "instruction"})   # repeated
```

```text
{"program_hex": "4831ed4883e4f0e8a403000089c7b83c0000000f05f4662e", "rip": "0x401000", "RSP": "0x7fffffffe360"}
{"call": 0, "stop": [["breakpoint", "post2"]], "clock": {"forward": 0, "aten_next": 500, "instruction": 0}}
{"call": 1, "status": "completed", "clock": {"forward": 0, "aten_next": 571, "instruction": 1},
 "commits": [{"offset": 0, "->": 3, "RSP": "00007fffffffe360", "flags": [0, 1, 0, 0]}]}
{"call": 2, "stop": [["breakpoint", "post2"]], "clock": {"forward": 1, "aten_next": 500, "instruction": 1}}
{"call": 3, "commits": [{"offset": 3, "->": 7, "RSP": "00007fffffffe350", "flags": [1, 0, 0, 0]}]}
{"call": 4, "stop": [["breakpoint", "post2"]], "clock": {"forward": 2, "aten_next": 500, "instruction": 2}}
{"call": 5, "stop": [["instruction_limit", null], ["motion_target", null]],
 "commits": [{"offset": 7, "->": 9, "RSP": "00007fffffffe350", "flags": [0, 1, 0, 0]}]}
native after step 0: rip 0x401003 RSP 0x7fffffffe360 flags [0,1,0,0]
native after step 1: rip 0x401007 RSP 0x7fffffffe360 flags [0,0,0,0]
native after step 2: rip 0x4013b0 RSP 0x7fffffffe358 flags [0,0,0,0]
```

Instruction 1 (`xor %rbp,%rbp`) matches native, including ZF. Instruction 2 is the `and` from §3. Instruction 3 is `call` (`e8`): the model predicts length 2 and commits without a store (the driver stops before a predicted store), while native jumps to `0x4013b0` and pushes. The driver has no branch-target output and no RAM, and it stops before committing predicted stores or syscalls by default (`DISCOVERY-DEBUGGER.md`). So `call`/`push` sequences can't be stepped faithfully here; that is what the harness's own run loop is for.

## 7. Use case: compare checkpoints

**Structure** (*measured here*). Same window, three recipes, from `catalog` (`debug/scratch/cat_de_nn`, `cat_de_hydra`, `tour`):

| | ATen events | MLP sites | affine | normalize | residual updates |
|---|---|---|---|---|---|
| de_mamba | 571 | 5 (decoder, post.0–2, flags) | 42 | 3 | 3 |
| de_nn | 286 | 11 (decoder, mixers.k.cross/token ×3, post.0–2, flags) | 42 | 9 | 3 |
| de_hydra | 655 | 5 (decoder, post.0–2, flags) | 39 | 3 | 3 |

All three share the decoder → three-block → heads layout, so `mlp_001` (the decoder), `module:post.N`, and the output heads mean the same thing in each. Scripts that address by module name port across checkpoints; scripts that use raw SSA ids or cursor numbers don't.

**Behaviour.** Use the same decoded-heads view on each recipe (`step forward`, `inspect outputs`, decode):

```text
de_mamba  outputs: RSP 0x7fffffffe350 CF 1 len 4 dest RSP imm 0xfffffffffffffff0 width 64
de_nn     outputs: RSP 0x7fffffffe370 CF 1 len 4 dest RSP imm 0xfffffffffffffff0 width 64
de_hydra  outputs: RSP 0x7fffffffe370 CF 1 len 4 dest RSP imm 0xfffffffffffffff0 width 64
```

Identical decoding, opposite operation choice. Together with the `/r` sweep in §3, this is a compact fingerprint of how each checkpoint generalizes. It costs 24 forwards, which you can't get from aggregate eval coverage (all three are at 0.0% step coverage on `dev`).

**Not covered by MDB.** For `m5d`, `just debug` still gives you the stop and the native context, but no recipe (*measured here*):

```text
$ just debug m5d dev fib-O2
{"debug_dir": "debug/m5d/dev/fib-O2", "status": "abstain", "exact_prefix_steps": 86, "mdb": null}
## Stop  {"kind": "abstain", "reason": "no_record", "rip": "0x401143", "window": "48894c24204883fa010f84aa000000"}
=>   401143:	mov    %rcx,0x20(%rsp)
```

M5d runs 86 steps exactly and then abstains on a store form its trie has no record for. By design it declines rather than guessing (reported: 0 wrong steps on fresh unseen gcc programs; `obsidian/blog/2026-10-30-170130-isa-survey-every-learned-x86-system-june-to-october.md`). The selectors have their own section, §10.

## 8. Use case: save a frontier, resume later, audit the journal

The `save-frontier` at the end of `and_tour.mdb` (§4) saved the edited `no_decoder` branch: 15,529,316 logical bytes, 33 MB on disk in `debug/scratch/frontier-store`. A fresh process resumes it at the saved cursor without replaying the prefix (*measured here*):

```text
$ mdb discovery resume debug/scratch/frontier-store edfbec91f251e57ad6f130bb2c0b7bf07fc6a68c11d0a40b7be183dafaeac9e8 \
      --command status --command "inspect outputs" --output debug/scratch/resumed
{"commands_executed": 2, "handler_hits": 0, "ok": true, "output": "debug/scratch/resumed"}
status: clock {"aten_next": 571, "forward": 0, "instruction": null}, branch no_decoder,
        work {"act_commits": 1, "branches": 2, "commands": 16, "native_operations": 1134, ...}
outputs: length 1, RSP 0x7fffffffe360, CF 0, ... (identical to the no_decoder branch in §4)
```

The work counters carry the original session's history (1,134 replayed ops, 1 Act); resuming adds none. Both the journal and the frontier can be inspected offline. The journal check needs no torch:

```text
$ mdb discovery inspect debug/scratch/and_tour/journal --last 1
mdb-discovery-offline-inspection-v1 16 metadata-only
{"command": "save-frontier debug/scratch/frontier-store", "line": 16, "operation": "checkpoint", ...}
$ mdb discovery inspect debug/scratch/frontier-store --cut edfbec91…
{"cut_id": "edfbec91…", "native_frontier": {"manifest": {"boundary": "mdb-discovery-durable-frontier-v1",
  "execution": {"backend": "cpu-fp32", "kind": "isa-bench-first-divergence", "program_step": 1},
  "model_identity": "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e", ... "acts": [...] ...
```

Every recorded command is a content-addressed object chained to the previous one (`journal/objects/<sha256>.json`, `head.json`), so a session is an auditable artifact you can attach to a result. Resume requires the same CPU/Torch/platform contract and `safetensors`. A *new* forward after resume needs an explicitly supplied model factory; none is deserialized (`DISCOVERY-DEBUGGER.md`).

## 9. Use case: seal a forecast of the model's answer, then let it run

§5 asked a 15-line symbol table what the model would do. A sealed forecast asks a harder question: can an independent account of the *whole* forward compute every output of the model, wrong answers included, before the model is allowed to run? MDB enforces the order for you.

**The parts.**

- **The equations.** `saturn-pub/packages/mdb/qualification/raw_isa_equations.py` (with `explicit_suffix_equations.py`) is a NumPy FP64 re-derivation of the R3 Mamba forward, written for checkpoint `9399e921…`. It covers the instruction decoder, the soft register reads, the two immediate-attention reads, the aligned slots, the input projection, all three mixer/MLP pairs and all eleven heads. It starts from the raw window, register and flag bits, uses a sequential scan where Torch uses a parallel one, and imports no Torch and no model code. It is complete but uninterpreted: a match tests that MDB's account of the forward's arithmetic is exact, not that anyone understands *why* the model computes `and` as `add`. That second question is what §3–§5 are for.
- **The transaction.** `mdb.forecast_transaction.ForecastTransaction` runs four phases in order: publish the inputs (`frontier`), compute the forecasts (`predict`), hash and seal them (`seal`), and only then release the model (`execute`). Module guards on the bound model refuse any call before the seal. The forecast arrays, the seal, the observations and a hash-chained event log are written to disk, where `mdb forecast verify` re-checks them without loading a model.
- **The comparison.** `mdb.equation_comparison.compare_equations` compares head by head under declared decision rules (sign for bit heads, argmax for `dest`/`src`/`width`/`length`). It reports signed residuals, observed minus forecast. `EquationCatalog` keeps one immutable, searchable record per case.

**The run** (*measured here*, 2026-10-31, laptop CPU; `debug/scratch/forecast.py`, appendix). There are ten inputs:

- `and $-16,%rsp` at step 1 of four `dev` programs, each with its own trailing window bytes and the native before-state from `trace.npz`;
- two compiled `add $1` instructions from §5;
- the symbol table's three sealed exceptions (§5);
- the fresh `RDI = 2^63−1` input, where §5 found the table's OF rule wrong.

The script prints each sealed forecast before the model has run once:

```text
$ PYTHONPATH=src:../saturn-pub/packages/mdb/src ../common/bin/common exec python debug/scratch/forecast.py debug/scratch/forecast-run
{"phase": "sealed", "seal_sha256": "0a5eef028747d8aa151f03adbb5a4277549a1ff417c12eda6ab0bc49971081ad", "model_calls_so_far": 0, "native_calls": 0, "wall_s": 2.23}
{"sealed": "and_rsp.fib-O2", "regs": {"RSP": "0x7fffffffe350"}, "flags": {"CF": 1, "ZF": 0, "SF": 0, "OF": 0}, "length": 4}
{"sealed": "and_rsp.fib-O0", "regs": {"RSP": "0x7fffffffe350"}, "flags": {"CF": 1, "ZF": 0, "SF": 0, "OF": 0}, "length": 4}
...
{"sealed": "add1.fib-O2.40121a", "regs": {"RDI": "0x2"}, "flags": {"CF": 0, "ZF": 0, "SF": 0, "OF": 0}, "length": 4}
{"sealed": "panel.held-081", "regs": {"RDI": "0x6e1f6959ffff8000"}, "flags": {"CF": 0, "ZF": 0, "SF": 0, "OF": 0}, "length": 4}
{"sealed": "fresh.rdi_2p63m1", "regs": {"RDI": "0x8000000000000000"}, "flags": {"CF": 0, "ZF": 0, "SF": 1, "OF": 1}, "length": 4}
{"phase": "released", "model_calls": 10, "native_calls": 10, "event_head": "4a6df2cf…", "wall_s": 4.03}
```

The forecast says the model will compute `RSP − 16` with CF = 1 for `and $-16,%rsp`. That is the wrong answer §3 found by running the model, now predicted with the model never called. After release, each case is compared against the model and against native x86 (the next state in the reference trace). The panel and fresh inputs have no native trace. Here is the condensed per-case output:

| case | input | forecast | model | native x86 | head decisions differing (of 11 fields) | max \|logit error\| | model's smallest \|logit\| |
|---|---|---|---|---|---|---|---|
| `fib-O2@0x401003` step 1, `and $-16,%rsp` | RSP `…e360`, ZF 1 | RSP `…e350`, CF 1 | RSP `…e350`, CF 1 | RSP `…e360`, CF 0 | 0 | 1.8e-5 | 0.204 |
| `fib-O0@0x401003` step 1 | same | same | same | same | 0 | 1.8e-5 | 0.114 |
| `sort-O2@0x401003` step 1 | same | same | same | same | 0 | 1.7e-5 | 0.041 |
| `strings-O0@0x401003` step 1 | same | same | same | same | 0 | 1.9e-5 | **0.007** |
| `fib-O2@0x40121a` step 236, `add $1,%rdi` | RDI `0x1` | `0x2` | `0x2` | `0x2` | 0 | 2.1e-5 | 7.46 |
| `checksum-O2@0x401183` step 309023 | RDI `0x403000` | `0x403001` | `0x403001` | `0x403001` | 0 | 2.8e-5 | 0.077 |
| panel held-081 | RDI `…58ffff7fff` | `…59ffff8000` | `…59ffff8000` | (correct: `…58ffff8000`) | 0 | 4.0e-5 | 0.072 |
| panel held-085 | RDI `…5affff7fff` | `…5bffff8000` | `…5bffff8000` | (correct: `…5affff8000`) | 0 | 3.2e-5 | 0.072 |
| panel held-087 | RDI `…5bffff7fff` | `…5fffff8000` | `…5fffff8000` | (correct: `…5bffff8000`) | 0 | 3.2e-5 | 0.160 |
| fresh RDI = 2^63−1 | `0x7fff…ffff` | `0x8000…0000`, SF 1, OF 1 | same | (correct: same, OF 1) | 0 | 3.2e-5 | 0.160 |

"Head decisions differing" counts every head: 512 register bits, 4 flags, the 64-bit immediate, the store address and value, length, `dest`, `src`, `width`, store and syscall. The two right-hand columns are the largest FP64-vs-FP32 logit difference anywhere, and the model's smallest |logit| on any sign-decided head. The script took 61 s wall, almost all of it environment start-up and model load. Sealing came 2.2 s after `t0` and release at 4.0 s.

**Audit it from the command line** (*measured here*; each call took 1–8 s with no model loaded):

```text
$ mdb forecast verify debug/scratch/forecast-run/transaction \
      --expected-seal 0a5eef028747d8aa151f03adbb5a4277549a1ff417c12eda6ab0bc49971081ad
{"event_head": "c03ecb7a…", "events": 7, "forecasts": ["equations"], "frontiers": ["cases"],
 "identity_scope": "externally-pinned", "phase": "closed", "verified": true, ...}

$ mdb equations compare forecast-run/and_rsp.predicted.json forecast-run/and_rsp.observed.json --rules forecast-run/rules.json
{"decision_fields": 11, "decision_mismatches": 0, "max_abs": 1.8093583737055496e-05,
 "residual_direction": "observed-minus-predicted", "fields": {
   "dest": {"decisions": {"rule": "argmax", "predicted": [4], "observed": [4], "different": 0}, ...},   # 4 = RSP
   "flags": {"decisions": {"predicted": [[1,0,0,0]], "observed": [[1,0,0,0]], "different": 0}, ...},   # CF wrongly set
   "length": {"decisions": {"predicted": [3], "observed": [3]}, ...},                                   # index 3 = 4 bytes
   "regs": {"decisions": {"decisions": 512, "different": 0}, "residual": {"max_abs": 1.81e-05}}, ...}}

$ mdb equations find --catalog forecast-run/equations --alias "isa-bench and_rsp.fib-O2"
[{"law_id": "r3.full-raw-state-transition", "route_id": "native", "alias": "isa-bench and_rsp.fib-O2",
  "evidence": {"forecast_seal": "0a5eef02…"}, "readers": ["regs", "flags", "length"],
  "tested_context": {"case": "and_rsp.fib-O2", "checkpoint": "9399e921…", "window_hex": "4883e4f0e8a403000089c7b83c0000"},
  "fingerprint": "7f5bd6b7…", "report": {...}}]

$ mdb ports forecast-run/ports/trace.json --select 'module:mixers.2.out_proj:return' --first-consumers
[{"value_id": "v000600", "producer_pc": 497, "port_pc": 498,
  "first_consumer": {"module_path": "mixers.2", "operator": "aten.add.Tensor", "pc": 498}, "already_consumed": false, ...}]
```

(Paths shortened from `debug/scratch/forecast-run/…`; the JSON is trimmed. `verify` prints a later event head than the run did because closing the transaction appends a `closed` event.) `mdb ports` needs a recording made with `record_forward(..., capture_ports=True)`. `just debug` records without it by default, because port capture renumbers every `vNNNNNN` and would break the ids quoted in §3–§5. Pass `debug.capture_ports=true` to get a port-ready recipe; the script records its own. That port, the final mixer's write, is the site of the §5 `×0.9` repair. Its value is SSA `v000600`, produced at PC 497 and consumed by the residual add at PC 498. The port-ready `just debug de_mamba dev fib-O2 debug.capture_ports=true` recipe puts the same write at `v000591`, PC 488 → 489 (*measured here*), because ids and PCs belong to one recording. To act on that write in a live session, use the id from your own recording, not an inferred affine name.

**A forecast that peeks is refused** (*measured here*, `forecast.py guard`). Its predict callback calls the model directly:

```text
{"error": "ForecastTransactionError: native module entry forbidden"}
{"phase": "closed", "seal_sha256": null, "work": {"predictions": 1, "native_calls": 0, "denied_attempts": 1, ...},
 "violations": [{"kind": "native module entry forbidden", "module": "<root>", "phase": "predict"}]}
```

What this shows:

1. **The forecast predicts the bug, not the spec.** On all four `and $-16,%rsp` inputs, the sealed forecast gave the model's wrong state (RSP − 16, CF = 1), not x86's (RSP unchanged, CF = 0). It matched every one of the model's head decisions, and it did so before the model ran. The same holds for the table's three exceptions: the forecast sealed bits 32, 32 and 34 wrong exactly as the model computes them. For these inputs, MDB's equations describe the checkpoint, not what the checkpoint should have learned.
2. **It is right when the model is right.** Both compiled `add $1` instructions (forecast = model = native) and the 2^63−1 case match. That includes OF = 1, which the symbol table's flag rule got wrong in §5. The full forward has no such blind spot, because it doesn't summarise.
3. **No forecast–model disagreement was found.** That is not because the decisions were easy. `strings-O0`'s `and` forward has a head only 0.007 from flipping, and the worst FP64-vs-FP32 error anywhere was 4.0e-5, about 175× smaller. A disagreement needs an input whose margin falls inside that error. To look for one, sweep many windows on Beast and rank them by the model's smallest |logit|. The guide did not run that sweep.
4. **Custody, not just numbers.** The forecast's seal is pinned, checkable offline, and tied to the catalog record. The equations declare the source hashes they were written against. isa-bench vendors `isa_lab/ssm.py` with the same hash (`319f9429…`). Its `direct_effect/model.py` hashes `0cecd869…`, which differs from the equations' declared `20231797…`. The forecast imports neither file. The 10/10 match is evidence that the forward used by this checkpoint is unchanged; it is not proof.

**Where this came from (reported).** The same equations and transaction have been run as larger campaigns on Beast:

- 32/32 fresh and 16/16 anchor decisions for suffix forecasts (`job-7a4c750a929f`, `saturn-pub/packages/mdb/docs/EXPLICIT-COMPUTATION-FORECASTS.md`);
- 18/18 raw-input steps over three successive instructions, with each forecast's predicted state feeding the next instruction's forecast and no native prefix (`job-62804158d4e4`, `RAW-INPUT-SEQUENCE-FORECASTS.md`);
- 42/42 executed rows matching every head across two later campaigns, including `add rax,rbx` (`4801d8` followed by XOR bytes), for which the forecast sealed RAX = 13,120,839 instead of 1030 (`obsidian/blog/2026-10-30-200139-predicting-how-an-isa-model-computes.md`, `INPUT-GENERALIZATION.md`).

The API reference is `COMPUTATION-FORECAST-WORKFLOW.md`. Forecasts give no speed-up: they recompute the model in FP64.

### Elsewhere in MDB (reported)

None of this was run for this guide; each item says where it comes from.

- **Value edits on the R3 NN.** A recipient-only edit installs a register value as an arithmetic operand. It produced the intended state in 24/24 new contexts, with 15/16 complete register states correct; one edit clears an unrelated bit (`obsidian/blog/2026-10-29-115335-what-our-neural-symbol-table-can-now-do.md`). ISA *weight* patches ("neural recompilation") live in `saturn/experiments/2026-10-29-neural-recompilation/`, not in MDB.
- **Batched screening and remote control.** `mdb.discovery_batch.FrozenNativeBatch` screens up to four independent full forwards per native call, with per-row output Acts. `mdb discovery attach JOB_ID` drives a resident worker inside an mrun lease over the authenticated debugger transport (`DISCOVERY-DEBUGGER.md`). These are the routes for the sweeps this guide deferred: the `/r` × checkpoint × window grid, decoder-coordinate sweeps, and scale × input grids.

## 10. Use case: why did a selector refuse?

The six selectors (§0) don't compute machine state. Each one picks one of 33 instruction families, and the frozen PAN consumer executes that family. The runtime only takes the step if two things hold. The pick's probability must be at least 0.5. It must also equal the family the consumer's own typed adapter would run. Otherwise the runtime refuses. So for a selector the question is never "which bit is wrong". It is: what did it pick, and how far away was the right family?

**What `just debug` gives you.** The selector adapter records its last call in the cell's `debug_bundle`: the instruction bytes, plus the 15 raw-byte tokens and 42 numeric features the selector was fed. `just debug` replays that one call locally:

- **Neural selectors** (`sel_nn`, `sel_mamba`, `sel_hydra`, `sel_pan_isa`): you get an MDB root recipe of the runtime head's forward (8, 106, 134 and 11 ATen ops respectively). You also get the top-5 families and the typed family's rank.
- **Structural selectors** (`sel_fieldlm`, `sel_spf`): these are route tables with no forward. The report shows the route key that was looked up and the learned routes for the same opcode.

Bundles only exist in cells evaluated with this adapter. This guide's cells come from `job-88365acf0ff3`: all six selectors × `c_apps`/`dev`/`heldout_m4`, 156 cells, with status counts identical to the earlier evals.

```text
$ just debug sel_mamba c_apps tiny_hello-Os                       # 81 s wall, mostly environment start-up
{"debug_dir": "debug/sel_mamba/c_apps/tiny_hello-Os", "status": "refused", "exact_prefix_steps": 0,
 "mdb": {"root_recipe": "debug/sel_mamba/c_apps/tiny_hello-Os/root-recipe", "checkpoint_sha256": "f29a132a…", "operations": 106},
 "explain": {"top": [["JMP_REL8", 0.998052], ["CALL_REL32", 0.001832], ["PUSH_RBP", 0.000108], ["MOV_IMM_REG", 4e-06], …],
             "typed": {"label": "MOV_IMM_REG", "rank": 4, "prob": 4e-06}}}

$ sed -n '/## Selector/,/## Native/p' debug/sel_mamba/c_apps/tiny_hello-Os/report.md
## Selector
Step 0 at 0x401000, fetch window `b801000000be00204000ba0e000000`.

The typed adapter executes **MOV_IMM_REG**; the selector has to name the same family.

| rank | family | probability |
|---:|---|---:|
| 1 | JMP_REL8 | 0.998 |
| 2 | CALL_REL32 | 0.00183 |
| 3 | PUSH_RBP | 0.000108 |
| 4 | MOV_IMM_REG | 4.04e-06 |
| 5 | XOR_RM_REG | 3e-06 |
```

The recipe opens in the same console as the direct-effect ones: `just mdb debug/sel_mamba/c_apps/tiny_hello-Os/root-recipe`. For the SSM selectors the catalog names `mlp_001` as the readout: `cat(scan first, scan mean, scan last, numeric features)` → Linear → GELU → Linear. `affine_004` is the mixer's `out_proj`. `inspect outputs` returns one `[1, 33]` logit tensor, not the direct-effect head dict, so `decode.py` does not apply. The label order is the PAN runtime's `RUNTIME_NEURAL_LABELS`.

**Every selector at its first refusal** (*measured here*, 2026-10-31; one process via `debug/scratch/debug_many.py`, 12 cells in 183 s):

| selector | program | step, rip | instruction | consumer runs | selector's pick | consumer family's rank |
|---|---|---|---|---|---|---|
| `sel_nn` | `dev/copy-O2` | 0, `0x401000` | `xor %rbp,%rbp` | XOR_RM_REG | LEA 0.335 | 12 |
| `sel_mamba` | `dev/copy-O2` | 0, `0x401000` | `xor %rbp,%rbp` | XOR_RM_REG | MOV_REG_RM 0.967 | 6 |
| `sel_hydra` | `dev/copy-O2` | 0, `0x401000` | `xor %rbp,%rbp` | XOR_RM_REG | MOV_RM_REG 0.999 | 6 |
| `sel_pan_isa` | `dev/copy-O2` | 0, `0x401000` | `xor %rbp,%rbp` | XOR_RM_REG | SUB_IMM_RM 0.590 | 11 |
| `sel_fieldlm`, `sel_spf` | `dev/copy-O2` | 1, `0x401003` | `and $-16,%rsp` | **none** (consumer: unsupported) | route `83 /4` missing; learned `83` routes: `/0` ADD_IMM_RM, `/5` SUB_IMM_RM | — |
| `sel_mamba` | `c_apps/tiny_hello-Os` | 0, `0x401000` | `mov $1,%eax` | MOV_IMM_REG | JMP_REL8 0.998 | 4 |
| `sel_hydra` | `c_apps/tiny_hello-Os` | 2, `0x40100a` | `mov $0xe,%edx` | MOV_IMM_REG | XOR_RM_REG 0.765 | 7 |
| all six | `c_apps/large_records-Os` | 0, `0x401026` | `push %rax` | **none** (consumer: unsupported) | e.g. `sel_mamba` MOV_IMM_REG 1.000; structural: route `50` missing | — |

What this shows:

1. **The neural selectors fail on the first instruction of every `dev` program.** That instruction is `xor %rbp,%rbp`. The consumer can run it (XOR_RM_REG), and the structural selectors route it correctly. All four neural selectors pick something else, and two of them are almost certain (0.967 and 0.999). The right family is ranked 6th to 12th.
2. **The structural selectors stop on the instruction where the direct-effect models first diverge.** That is `and $-16,%rsp` (§3). It is not a selection error: the PAN consumer has no family for `83 /4`, so no selector could pass it. The route table shows the same gap: for opcode `0x83` it knows only `/0` (add) and `/5` (sub).
3. **`push %rax` is a consumer gap too.** It stops all six selectors on `large_records`. The neural selectors still pick a family for it, `sel_mamba` with probability 1.000. The runtime refuses because the typed adapter refuses, not because of the selector.
4. **A family can be right for one register and wrong for another.** `sel_hydra` runs `mov $1,%eax` and `mov $0x402000,%esi` as MOV_IMM_REG. It then calls `mov $0xe,%edx` XOR_RM_REG at 0.765. Same opcode row (`b8+r`), different register.

**Fork and edit: which half of the readout picks the wrong family?** The SSM selectors' readout input concatenates 48 byte-scan features with the 42 numeric features. `debug/scratch/sel_lab.py` (appendix) breaks at `mlp_001`'s entry and captures a cut there. It then forks three arms from that one cut: no edit, numeric half zeroed, and scan half zeroed. In each arm it reads the top family and the rank of the consumer's family (*measured here*, 4 cases, 216 s including one `just debug` recording):

```text
{"case": "sel_mamba/c_apps/tiny_hello-Os", "arm": "no edit",                  "top": ["JMP_REL8", 0.9981],        "typed": ["MOV_IMM_REG", 4, 4.04e-06]}
{"case": "sel_mamba/c_apps/tiny_hello-Os", "arm": "scan only (numeric := 0)", "top": ["JMP_REL8", 0.9938],        "typed": ["MOV_IMM_REG", 5, 8.91e-07]}
{"case": "sel_mamba/c_apps/tiny_hello-Os", "arm": "numeric only (scan := 0)", "top": ["MOV_IMM_REG", 0.204],       "typed": ["MOV_IMM_REG", 1, 0.204]}
{"case": "sel_hydra/c_apps/tiny_hello-Os", "arm": "no edit",                  "top": ["XOR_RM_REG", 0.7655],      "typed": ["MOV_IMM_REG", 7, 0.0083]}
{"case": "sel_hydra/c_apps/tiny_hello-Os", "arm": "scan only (numeric := 0)", "top": ["XOR_RM_REG", 0.6683],      "typed": ["MOV_IMM_REG", 11, 0.00127]}
{"case": "sel_hydra/c_apps/tiny_hello-Os", "arm": "numeric only (scan := 0)", "top": ["MOVDQA_XMM_M128", 0.4513], "typed": ["MOV_IMM_REG", 3, 0.0423]}
{"case": "sel_mamba/dev/copy-O2",          "arm": "no edit",                  "top": ["MOV_REG_RM", 0.967],       "typed": ["XOR_RM_REG", 6, 1.65e-10]}
{"case": "sel_mamba/dev/copy-O2",          "arm": "scan only (numeric := 0)", "top": ["MOV_REG_RM", 0.8354],      "typed": ["XOR_RM_REG", 8, 9e-10]}
{"case": "sel_mamba/dev/copy-O2",          "arm": "numeric only (scan := 0)", "top": ["MOVDQU_XMM_XMM", 0.4485],  "typed": ["XOR_RM_REG", 16, 0.00298]}
{"case": "sel_hydra/dev/copy-O2",          "arm": "no edit",                  "top": ["MOV_RM_REG", 0.9986],      "typed": ["XOR_RM_REG", 6, 1.31e-06]}
{"case": "sel_hydra/dev/copy-O2",          "arm": "scan only (numeric := 0)", "top": ["MOV_RM_REG", 0.9996],      "typed": ["XOR_RM_REG", 6, 4.74e-06]}
{"case": "sel_hydra/dev/copy-O2",          "arm": "numeric only (scan := 0)", "top": ["MOVDQU_XMM_XMM", 0.2992],  "typed": ["XOR_RM_REG", 15, 0.0157]}
```

In each case the baseline arm reproduces the debug report's top-1 probability, which checks the replay. The **byte-scan half carries the wrong pick in all 4 cases.** With the numeric features zeroed, each selector keeps the same wrong family at 0.67–1.00. The numeric half alone puts the consumer's family first in only 1 of 4 cases (`sel_mamba` on `mov`, at 0.20). In the rest it drifts to SIMD moves. So the handcrafted numeric features (opcode, ModRM and prefix norms) would not fix these selectors by being weighted up. The scan path reads these byte patterns into the wrong family. This is a lead from 4 inputs and 2 checkpoints. The next probe would transplant the scan features from an instruction the selector gets right, in the style of §4.

## 11. Cheat sheet

Setup and entry points:

```sh
mdb() { PYTHONPATH=../saturn-pub/packages/mdb/src ../common/bin/common exec python -m mdb "$@"; }
just debug <model> <suite> [program] [overrides]    # report.md + bundle.json (+ root-recipe/ for de_* and neural sel_*)
just debug de_mamba dev fib-O2 debug.capture_ports=true   # port-ready recipe for `mdb ports` (renumbers vNNNNNN)
just mdb debug/<model>/<suite>/<prog>/root-recipe   # interactive; rm -rf …/console before a 2nd run
mdb discovery replay <recipe> --interactive --output <new dir>
mdb discovery replay <recipe> --command "<cmd>" [--command …] --output <new dir>
mdb discovery replay <recipe> --script file.mdb --output <new dir>     # file starts: interface mdb-discovery-v1
mdb discovery resume <store> <cut_id> --command … --output <new dir>
mdb discovery inspect <out>/journal --last N        # offline, verifies the hash chain
mdb discovery inspect <store> --cut <cut_id>        # offline frontier manifest
mdb forward-symbols <recipe>/trace.json --readable | --value vNNNNNN | --step N | --list
mdb forecast verify|show <transaction> [--expected-seal SHA]   # offline seal/event-chain check, no model
mdb equations compare predicted.json observed.json --rules rules.json [--residuals]
mdb equations find|list|show --catalog <dir> [--alias …|--law-id …|--reader …]
mdb ports <trace.json> [--select 'module:<path>:return' --first-consumers | --kind module]   # needs capture_ports=True (just debug … debug.capture_ports=true)
```

Console commands (`src/mdb/discovery_script.py`). Entries marked † come from the parser and were not exercised in this guide:

| command | does |
|---|---|
| `status`, `capabilities`, `catalog`, `events`† | lanes/clocks/work; feature list; structural sites; event log |
| `alias NAME = SELECTOR` | session alias (`module:post.1`, `mlp_003`, `affine_028`, `vNNNNNN`) |
| `break SELECTOR enter\|exit [--id ID]` | breakpoint at a site boundary |
| `step\|next\|into\|finish [UNIT]` | UNIT ∈ `op math law update forward instruction`; default `law` |
| `until SELECTOR`, `continue` / `run`† | run to a site / to the next stop; `--all --stop-policy lane`† for many lanes |
| `capture NAME`, `restore NAME` | make / return to a cut |
| `fork NAME`, `select NAME` | branch from the current head / focus a branch |
| `inspect state\|outputs\|live\|SELECTOR[:input\|:output]` | read lane frame, heads, live frontier, or a value |
| `save-frontier STORE` | durable snapshot of the selected branch |
| `lane focus ID`† | switch lanes |
| `on-hit ID … end`, `repeat N … end`† | handler block (motion only as `continue --defer`); finite loop |
| `call {"operation": …, "payload": {…}}` | any typed operation, e.g. below |

Typed operations you will want via `call`:

```text
call {"operation":"act","payload":{"selector":"mlp_003:output","value_spec":{"kind":"zeros_like"}}}
call {"operation":"act","payload":{"selector":"affine_028:output","value_spec":{"kind":"scale","factor":0.9}}}
call {"operation":"act","payload":{"selector":"v000430","value":[[...]],"sign":"add"}}     # literal; replace|add († "add" not run here)
call {"operation":"break","payload":{"action":"disable","id":"p1"}}                      # enable|remove|list
call {"operation":"watch","payload":{"selector":"mlp_004:output","predicate":"change"}}  # change watches only († not run here)
call {"operation":"fork","payload":{"name":"arm2","cut":"at_decoder"}}                  # fork from a named cut
call {"operation":"capabilities"}
```

Sealed forecast skeleton (§9; full version in the appendix's `forecast.py`):

```python
from mdb.forecast_transaction import ForecastTransaction
from mdb.equation_comparison import compare_equations
t = ForecastTransaction(out, model=model, model_identity=ckpt_sha, input_identity=inputs_sha,
                        source_identity=equation_shas, frontier={"kind": "raw-input"})
with t:
    t.frontier("cases", inputs)            # publish raw inputs; no model call
    t.predict("equations", predict)        # predict(ctx) -> {"predictions": ...}; ctx.weights = actual state_dict
    t.seal()                               # model calls before this point raise
    t.execute("native", execute)           # execute(ctx) calls ctx.model(...)
compare_equations(t.prediction("equations")[k]["heads"], t.observation("native")[k]["heads"],
                  decision_rules={"regs": "positive", "length": "argmax", ...})
```

Python (same semantics, `src/mdb/discovery.py`):

```python
from mdb.discovery import DiscoverySession
from mdb.discovery_cli import load_capture, save_capture
from mdb.forward_recording import record_forward      # record any frozen eval-mode torch model
from mdb.discovery_runtime import ReplayForward
from mdb.discovery_isa import ISADriver                # instruction stepping (R3 ABI)
s = DiscoverySession(budgets={"max_native_operations": 4_000_000})
s.add_lane("lane_00", lambda: load_capture("debug/de_mamba/dev/fib-O2/root-recipe"))
s.call("break", {"selector": "mlp_001", "edge": "exit", "id": "dec"}); s.call("motion", {"action": "continue"})
```

## 12. Limits, gotchas, next steps

Gotchas (all *measured here* unless marked):

- **`--output` must be a new directory.** Every `replay`/`resume` refuses an existing one, including `just mdb`'s fixed `<recipe>/../console`.
- **`inspect outputs` returns raw logits.** Decode them yourself (appendix `decode.py`). Saturn request `mdb-discovery-isa-output-decode-profile` asks for this in the console.
- **Ids are per recording.** Program ids differ per input (`program:8303…` for the push recipe, `program:4012…` for the `and` recipe). Use site names (`mlp_001`, `module:post.2`); `vNNNNNN` and cursor numbers only hold within one recording, or the same model and recording mode (reported: freezing parameters with `requires_grad_(False)` shifts the post.1 MLP from 359/369 to 365/375).
- **No weight edits in the discovery console.** Roots are read-only (`edits require a produced, live, unconsumed tensor; roots are read-only`). Edit a produced value instead: a module output, or the transposed weight view an `affine` produces just before its `addmm`, for a one-forward weight experiment. That last idea is *not run here*.
- **A recipe replays from its roots.** It does not resume the original eager run (`direct_eager_pause: false`); use `save-frontier` for real pause/resume.
- **The driver has no RAM, no branch targets, and stops before stores/syscalls** (§6). It is for stepping straight-line ALU code, not whole programs.
- **`mdb ports` on a default `just debug` recipe fails** with `NativePortError: trace has no native ports; record with capture_ports=True` (*measured here*). Re-run `just debug … debug.capture_ports=true`. Expect new `vNNNNNN` ids and PCs; the 571 events are the same.
- **Forecast scope.** §9's equations are written for checkpoint `9399e921…` (`de_mamba`) only. `de_nn` and `de_hydra` have no independent equations, so a sealed forecast for them would need new ones. A forecast is not an oracle either: it predicts the model, and correctness still comes from the native trace.
- **`mdb forward-symbols --readable --scope decoder` printed "no operations in this scope"** on these recipes; use `--value` / `--step` or the unscoped listing.
- **Cost.** Each CLI call took about 20–60 s wall, almost all of it environment start-up. Inside one Python process, 8 recordings plus 16 replayed suffixes took 12.2 s. Keep loops in one process, and send anything wider than a handful of inputs to Beast.

Scope of the claims:

- Symbol-table statements are about checkpoint `9399e921…` and the inputs listed. "Predicts compiled C" means 5/5 `add $1` instructions, not general decoding.
- The decoder 2-coordinate flip and the `mixers.2` 0.9 repair are single-checkpoint observations on 1 and 8 inputs respectively. Both came with a measured counter-test (the `sub` transfer failed; held-087 not fixed). Treat them as leads.

Next steps that would make this better:

1. Selectors (§10): transplant the byte-scan readout features from an instruction each selector gets right. Then sweep the scan/numeric fork over every refusal in one mrun lease.
2. Saturn requests filed today: `mdb-discovery-donor-lane-acts` (console transplants with donor custody) and `mdb-discovery-isa-output-decode-profile` (decoded heads in the console). Upvoted: `mdb-computation-hypotheses-and-causal-naming` (attach the OF counterexample to the table's flag rule).
3. Move the `/r` sweep, decoder-coordinate sweep and `mixers.2` scale × input grid onto Beast as one mrun lease using `FrozenNativeBatch`, and seal the full-forward forecast for every swept input before release. §9 did this for 10 inputs locally; the sweep would also hunt for forecast-model disagreements at the smallest margins.
4. Give `just mdb` a fresh output directory per session.

---

## Appendix: the scripts behind the measured examples

The measured Python examples were run as:

```sh
PYTHONPATH=src:../saturn-pub/packages/mdb/src ../common/bin/common exec python debug/scratch/lab.py <section>
```

The sections are `sweep_ext`, `symtab`, `edit_and`, `selector_check`, `bit32`, `bit32_check` and `driver`, run one process at a time. §9 used `debug/scratch/forecast.py OUT_DIR` and `forecast.py guard OUT_DIR`. §10 used `debug/scratch/debug_many.py MODEL:SUITE[:PROGRAM] …` (stage=debug for many cells in one process) and `sel_lab.py DEBUG_DIR …`. `debug/` is gitignored, so all of these files are reproduced verbatim below. `lab.py` imports harness code (`load_adapter`) the same way `src/isa_bench/debug.py` does.

### `debug/scratch/decode.py`

```python
"""Decode `inspect outputs` records of an MDB discovery result.json into x86 state (no torch)."""
import json
import sys

REGS = ("RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI")
FLAGS = ("CF", "ZF", "SF", "OF")


def word(bits):
    return sum(1 << i for i, b in enumerate(bits) if b > 0)


def decode(heads):
    v = {k: h["value"] for k, h in heads.items()}
    argmax = lambda xs: max(range(len(xs)), key=xs.__getitem__)
    out = {"length": argmax(v["length"][0]) + 1,
           "length_logits_top3": sorted(((round(x, 2), i + 1) for i, x in enumerate(v["length"][0])), reverse=True)[:3],
           "regs": {n: hex(word(v["regs"][0][i])) for i, n in enumerate(REGS)},
           "flags": {n: int(v["flags"][0][i] > 0) for i, n in enumerate(FLAGS)},
           "flag_logits": [round(x, 2) for x in v["flags"][0]],
           "store": v["store"][0] > 0, "store_logit": round(v["store"][0], 2), "syscall": v["syscall"][0] > 0}
    if "dest" in v:
        out.update({"dest": REGS[argmax(v["dest"][0])], "src": REGS[argmax(v["src"][0])],
                    "width": (8, 32, 64)[argmax(v["width"][0])], "imm": hex(word(v["imm"][0]))})
    return out


if __name__ == "__main__":
    result = json.load(open(sys.argv[1]))
    for rec in result["records"]:
        if rec.get("command") == "inspect outputs":
            branch = rec["response"]["lane"]["branch_id"]
            print(json.dumps({"line": rec["line"], "branch": branch, **decode(rec["response"]["value"])}))
```

### `debug/scratch/forecast.py`

```python
"""Sealed computation forecasts for de_mamba on real isa-bench inputs (docs/debugging-with-mdb.md section 9).

    PYTHONPATH=src:../saturn-pub/packages/mdb/src ../common/bin/common exec python debug/scratch/forecast.py OUT_DIR

For each case, MDB's independent NumPy FP64 equations (saturn-pub/packages/mdb/qualification/raw_isa_equations.py,
written for checkpoint 9399e921) compute every head of the forward from the raw window, registers and flags. The
forecasts are sealed in a ForecastTransaction, and only then is the Torch FP32 model allowed to run. The
transaction's module guards deny any model call before the seal.

Writes OUT_DIR/{transaction/, equations/, ports/trace.json, and_rsp.predicted.json, and_rsp.observed.json, rules.json}.
"""
from __future__ import annotations

import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import FLAGS, M64, REGS, REPO, adapter, native_before, roots  # noqa: E402

QUAL = REPO.parent / "saturn-pub/packages/mdb/qualification"
RULES = {name: "argmax" if name in ("dest", "src", "width", "length") else "positive"
         for name in ("regs", "flags", "dest", "src", "width", "length", "syscall", "imm",
                      "store", "store_addr", "store_value")}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def cases():
    """Real program inputs (with their native after-state) first, then the symbol table's panel inputs."""
    out = []
    for prog in ("fib-O2", "fib-O0", "sort-O2", "strings-O0"):  # `and $-16,%rsp` is instruction 2 of each
        step, window, before, after = native_before("dev", prog, 0x401003)
        out.append((f"and_rsp.{prog}", f"{prog}@0x401003 step {step}", window, before, after))
    for suite, prog, rip in (("dev", "fib-O2", 0x40121A), ("heldout_m4", "checksum-O2", 0x401183)):
        step, window, before, after = native_before(suite, prog, rip)
        out.append((f"add1.{prog}.{rip:x}", f"{prog}@{rip:#x} step {step}", window, before, after))
    f = json.loads((REPO.parent / "isa/outputs/response-quotient-r3-mamba/findings.json").read_text())
    for ex in f["heldout_metrics"]["target_bit_exceptions"]:
        out.append((f"panel.{ex['query_id']}", f"sealed panel {ex['query_id']}", bytes.fromhex(ex["raw_window_hex"]),
                    (ex["input_register_words_before"], dict.fromkeys(FLAGS, 0)), None))
    regs = list(f["heldout_metrics"]["target_bit_exceptions"][0]["input_register_words_before"])
    regs[7] = (1 << 63) - 1
    out.append(("fresh.rdi_2p63m1", "fresh RDI=2^63-1", bytes.fromhex("4883c7014883c7014883c7014883c7"),
                (regs, dict.fromkeys(FLAGS, 0)), None))
    rows = {}
    for key, label, window, before, after in out:
        if isinstance(before, tuple):
            words, flags = list(before[0]), before[1]
        else:
            words, flags = list(before.regs[:8]), {n: before.flags[n] for n in FLAGS}
        rows[key] = {"label": label, "window_hex": bytes(window).ljust(15, b"\0")[:15].hex(),
                     "register_words": [int(w) for w in words], "flags": [int(flags[n]) for n in FLAGS],
                     "native_after": None if after is None else {
                         "register_words": [int(w) for w in after.regs[:8]],
                         "flags": [int(after.flags[n]) for n in FLAGS], "rip": int(after.rip)}}
    return rows


def margin(heads):
    """Smallest |logit| over the sign-decided heads: how close the forward came to flipping a bit."""
    import numpy as np

    return float(min(np.abs(np.asarray(heads[k], dtype=np.float64)).min()
                     for k, rule in RULES.items() if rule == "positive" and k in heads))


def main(out: Path):
    import numpy as np

    from mdb.equation_comparison import EquationCatalog, compare_equations
    from mdb.forecast_transaction import ForecastTransaction
    from mdb.forward_recording import record_forward

    formulas = runpy.run_path(str(QUAL / "raw_isa_equations.py"))
    model = adapter("de_mamba").model.requires_grad_(False)
    checkpoint = sha((REPO / "artifacts/frozen/de_mamba/model.pt").read_bytes())
    plan = cases()
    plan_raw = json.dumps(plan, sort_keys=True).encode()
    sources = {name: sha((QUAL / name).read_bytes()) for name in ("raw_isa_equations.py", "explicit_suffix_equations.py")}
    sources["forecast.py"] = sha(Path(__file__).read_bytes())
    out.mkdir(parents=True, exist_ok=False)
    calls = {"model": 0}
    model.register_forward_pre_hook(lambda module, args: calls.__setitem__("model", calls["model"] + 1))
    t0 = time.monotonic()

    def predict(context):
        weights = formulas["prepare_weights"](context.weights)
        result = {}
        for key, case in context.frontiers["cases"].items():
            raw = formulas["roots"](case["window_hex"], case["register_words"], case["flags"])
            p = formulas["predict_raw"](*raw, weights)
            result[key] = {"heads": {k: np.asarray(v, dtype=np.float64) for k, v in p["heads"].items()},
                           "decoded": p["decoded"]}
        return {"predictions": result}

    def execute(context):
        import torch

        result = {}
        with torch.no_grad():
            for key, case in context.frontiers["cases"].items():
                words, flags = case["register_words"], dict(zip(FLAGS, case["flags"]))
                heads = context.model(*roots(bytes.fromhex(case["window_hex"]), words, flags))
                result[key] = {"heads": {k: v.detach().numpy().astype(np.float64) for k, v in heads.items()}}
        return result

    transaction = ForecastTransaction(out / "transaction", model=model, model_identity=checkpoint,
                                      input_identity=sha(plan_raw), source_identity=sources,
                                      frontier={"kind": "raw-input"})
    with transaction:
        transaction.frontier("cases", plan)
        transaction.predict("equations", predict)
        transaction.seal()
        sealed = transaction.report()
        print(json.dumps({"phase": "sealed", "seal_sha256": sealed["seal_sha256"], "model_calls_so_far": calls["model"],
                          "native_calls": sealed["work"]["native_calls"], "wall_s": round(time.monotonic() - t0, 2)}))
        forecasts = transaction.prediction("equations")
        for key in plan:
            d = forecasts[key]["decoded"]
            print(json.dumps({"sealed": key, "regs": {REGS[i]: hex(w) for i, w in enumerate(d["register_words"])
                                                      if w != plan[key]["register_words"][i]},
                              "flags": dict(zip(FLAGS, d["flags"])), "length": d["length"]}))
        transaction.execute("native", execute)
        report = transaction.report()
    observed = transaction.observation("native")
    print(json.dumps({"phase": report["phase"], "model_calls": calls["model"], "native_calls": report["work"]["native_calls"],
                      "event_head": report["event_head"], "wall_s": round(time.monotonic() - t0, 2)}))

    catalog = EquationCatalog(out / "equations")
    for key, case in plan.items():
        pred, obs = forecasts[key]["heads"], observed[key]["heads"]
        cmp = compare_equations(pred, obs, decision_rules=RULES)
        model_words = formulas["decode"](obs)
        native = case["native_after"]
        slot = next((i for i in range(8) if forecasts[key]["decoded"]["register_words"][i] != case["register_words"][i]), None)
        row = {"case": key, "label": case["label"], "decision_mismatches": cmp["decision_mismatches"],
               "max_abs_logit_error": round(cmp["max_abs"], 7), "model_margin": round(margin(obs), 4),
               "forecast_margin": round(margin(pred), 4)}
        if slot is not None:
            row.update({"slot": REGS[slot], "in": hex(case["register_words"][slot]),
                        "forecast": hex(forecasts[key]["decoded"]["register_words"][slot]),
                        "model": hex(model_words["register_words"][slot]),
                        "native": None if native is None else hex(native["register_words"][slot])})
        row["flags_forecast_model_native"] = [forecasts[key]["decoded"]["flags"], model_words["flags"],
                                              None if native is None else native["flags"]]
        if native is not None:
            row["model_correct"] = (model_words["register_words"] == native["register_words"]
                                    and model_words["flags"] == native["flags"])
        print(json.dumps(row))
        catalog.record(law_id="r3.full-raw-state-transition", route_id="native", alias="isa-bench " + key,
                       binding={"model_identity": checkpoint, "execution": {"backend": "cpu", "dtype": "float32"},
                                "input_identity": sha(json.dumps(case, sort_keys=True).encode()),
                                "frontier": "raw-input", "horizon": 1, "source_identity": sources},
                       report=cmp, conditions=[{"field": "checkpoint", "op": "eq", "value": checkpoint}],
                       tested_context={"checkpoint": checkpoint, "case": key, "window_hex": case["window_hex"]},
                       readers=["regs", "flags", "length"],
                       evidence={"forecast_seal": report["seal_sha256"]})

    # files for `mdb equations compare` and `mdb ports`
    key = "and_rsp.fib-O2"
    for name, heads in (("predicted", forecasts[key]["heads"]), ("observed", observed[key]["heads"])):
        (out / f"and_rsp.{name}.json").write_text(json.dumps({k: np.asarray(v).tolist() for k, v in heads.items()}))
    (out / "rules.json").write_text(json.dumps(RULES))
    case = plan[key]
    capture = record_forward(model, roots(bytes.fromhex(case["window_hex"]), case["register_words"],
                                          dict(zip(FLAGS, case["flags"]))),
                             model_identity=checkpoint, execution={"backend": "cpu", "dtype": "float32"},
                             input_identity=sha(json.dumps(case, sort_keys=True).encode()), capture_ports=True)
    (out / "ports").mkdir()
    (out / "ports/trace.json").write_text(json.dumps(capture.trace, sort_keys=True))
    print(json.dumps({"done": str(out), "wall_s": round(time.monotonic() - t0, 2)}))


def guard(out: Path):
    """A forecast that peeks: the predict callback calls the model before the seal."""
    from mdb.forecast_transaction import ForecastTransaction

    model = adapter("de_mamba").model.requires_grad_(False)
    case = cases()["and_rsp.fib-O2"]
    args = roots(bytes.fromhex(case["window_hex"]), case["register_words"], dict(zip(FLAGS, case["flags"])))

    def cheat(context):
        return {"predictions": {"heads": {k: v.detach().numpy() for k, v in model(*args).items()}}}

    transaction = ForecastTransaction(out, model=model, model_identity="9399e921", input_identity="peek",
                                      source_identity={"cheat": "1"}, frontier={"kind": "raw-input"})
    try:
        with transaction:
            transaction.frontier("cases", case)
            transaction.predict("peek", cheat)
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"error": f"{type(error).__name__}: {error}"[:300]}))
    r = transaction.report()
    print(json.dumps({"phase": r["phase"], "seal_sha256": r["seal_sha256"], "work": r["work"],
                      "violations": r["violations"]})[:800])


if __name__ == "__main__":
    (guard if sys.argv[1:2] == ["guard"] else main)(Path(sys.argv[-1]))
```

### `debug/scratch/lab.py`

```python
"""Scratch lab for docs/debugging-with-mdb.md. One section per invocation, one process at a time.

    PYTHONPATH=src:../saturn-pub/packages/mdb/src ../common/bin/common exec python debug/scratch/lab.py <section>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGS = ("RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI")
FLAGS = ("CF", "ZF", "SF", "OF")
M64 = (1 << 64) - 1


def adapter(name):
    from isa_bench.adapters import load_adapter
    from isa_bench.cli import compose

    a = load_adapter(compose("model", name), REPO)
    a.load()
    return a


def native_before(suite, program, rip):
    """Native state before the first execution of `rip`, plus the 15-byte fetch window and the native after-state."""
    from isa_bench.state import PageMemory, Trace

    pdir = REPO / "outputs/reference" / suite / "programs" / program
    tr = Trace(pdir)
    for i in range(len(tr)):
        if int(tr.steps[i][0]) == rip:
            before, after = tr.state(i), tr.state(i + 1)
            return i, PageMemory(pdir / "mem.bin").fetch(rip), before, after
    raise KeyError(hex(rip))


def roots(window, regs, flags):
    import torch

    window = bytes(window).ljust(15, b"\0")[:15]
    win = torch.tensor([[(v >> i) & 1 for v in window for i in range(8)]], dtype=torch.float32)
    reg = torch.tensor([[[(v >> i) & 1 for i in range(64)] for v in regs]], dtype=torch.float32)
    flg = torch.tensor([[flags[n] for n in FLAGS]], dtype=torch.float32)
    return win, reg, flg


def symtab_program(window, regs):
    """The R3 Mamba symbol table as a program: route(raw bits 16-18) -> q-law y=q^x, q'=q&x (q0=1); flags rule; length 4.

    Sources: isa/outputs/response-route-r3-mamba (route), response-quotient-r3-mamba (q-law, q0=1),
    symbolic-loop-r3-mamba (flags CF=0, ZF=0, SF=output bit 63, OF=0; length 4)."""
    slot = window[2] & 7
    x, q, y = regs[slot], 1, 0
    for i in range(64):
        xi = (x >> i) & 1
        y |= (q ^ xi) << i
        q = q & xi
    out = list(regs)
    out[slot] = y
    return {"slot": REGS[slot], "regs": out, "flags": {"CF": 0, "ZF": 0, "SF": (y >> 63) & 1, "OF": 0}, "length": 4}


def sweep_ext():
    """`48 83 /r f0` on RSP for every ModRM reg field, across the three direct-effect checkpoints."""
    _, window, before, after = native_before("dev", "fib-O2", 0x401003)
    regs, flags = list(before.regs[:8]), {n: before.flags[n] for n in FLAGS}
    print(json.dumps({"window": window.hex(), "RSP_before": hex(regs[4]), "flags_before": flags,
                      "native_after": {"RSP": hex(after.regs[4]), "CF": after.flags["CF"]}}))
    names = ("add", "or", "adc", "sbb", "and", "sub", "xor", "cmp")
    for model in ("de_mamba", "de_nn", "de_hydra"):
        a = adapter(model)
        row = {}
        for r in range(8):
            w = bytearray(window)
            w[2] = 0xC4 | (r << 3)
            out = a.step(bytes(w), regs, flags, 0x401003)
            d = (out["regs"][4] - regs[4]) & M64
            delta = d - (1 << 64) if d >> 63 else d
            row[f"/{r} {names[r]}"] = f"RSP{delta:+d} CF={out['flags']['CF']} len={out['length']}"
        print(json.dumps({"model": model, **row}))


def symtab():
    """Ask the symbol table first, then run de_mamba, on real program instructions and on the sealed panel's inputs."""
    a = adapter("de_mamba")
    cases = []
    for suite, prog, rip in (("dev", "fib-O2", 0x40121A), ("heldout_m4", "checksum-O2", 0x401183),
                             ("dev", "fib-O2", 0x401236), ("heldout_m4", "checksum-O2", 0x4010f3),
                             ("heldout_m4", "checksum-O2", 0x4010ba)):
        try:
            step, window, before, after = native_before(suite, prog, rip)
        except KeyError:
            print(json.dumps({"case": f"{prog}@{rip:#x}", "skipped": "not executed natively"}))
            continue
        cases.append((f"{prog}@{rip:#x} step {step}", window, list(before.regs[:8]),
                      {n: before.flags[n] for n in FLAGS}, after))
    f = json.loads((REPO.parent / "isa/outputs/response-quotient-r3-mamba/findings.json").read_text())
    for ex in f["heldout_metrics"]["target_bit_exceptions"]:
        cases.append((f"sealed panel {ex['query_id']}", bytes.fromhex(ex["raw_window_hex"]),
                      ex["input_register_words_before"], dict.fromkeys(FLAGS, 0), None))
    base = f["heldout_metrics"]["target_bit_exceptions"][0]["input_register_words_before"]
    for k in (31, 32, 33, 47, 63):
        regs = list(base)
        regs[7] = (1 << k) - 1
        cases.append((f"fresh RDI=2^{k}-1", bytes.fromhex("4883c7014883c7014883c7014883c7"), regs,
                      dict.fromkeys(FLAGS, 0), None))
    for label, window, regs, flags, native in cases:
        pred = symtab_program(window, regs)
        out = a.step(window, regs, flags, 0)
        slot = REGS.index(pred["slot"])
        wrong_bits = [i for i in range(64) if ((out["regs"][slot] ^ pred["regs"][slot]) >> i) & 1]
        other_regs_equal = all(out["regs"][i] == pred["regs"][i] for i in range(8) if i != slot)
        row = {"case": label, "window": window[:4].hex(), "slot": pred["slot"], "in": hex(regs[slot]),
               "symtab": hex(pred["regs"][slot]), "model": hex(out["regs"][slot]),
               "target_word_exact": not wrong_bits, "wrong_bits": wrong_bits, "other_regs_equal": other_regs_equal,
               "flags_symtab": pred["flags"], "flags_model": out["flags"], "len_model": out["length"]}
        if native is not None:
            row["native"] = hex(native.regs[slot])
            row["flags_native"] = {n: native.flags[n] for n in FLAGS}
        print(json.dumps(row))


# ---------------------------------------------------------------- MDB discovery sessions (Python API)

def ok(r):
    if not r.get("ok"):
        raise SystemExit(json.dumps(r, default=str)[:3000])
    return r


def session_for(recipe):
    from mdb.discovery import DiscoverySession
    from mdb.discovery_cli import load_capture

    s = DiscoverySession(budgets={"max_native_operations": 4_000_000})
    s.add_lane("lane_00", lambda: load_capture(recipe))
    return s


def outputs(s):
    sys.path.insert(0, str(Path(__file__).parent))
    from decode import decode

    raw = ok(s.call("inspect", {"selector": "outputs"}))["value"]
    return decode(raw), raw


def prepare_cuts(s, sites):
    """Break at every site's exit, capture a named cut at each stop, then disarm."""
    for site in sites:
        ok(s.call("break", {"selector": site, "edge": "exit", "id": "b_" + site}))
    seen = []
    while len(seen) < len(sites):
        r = ok(s.call("motion", {"action": "continue"}))
        hits = [c["id"] for c in r["stop_causes"] if c.get("kind") == "breakpoint"]
        if not hits:
            break
        for h in hits:
            ok(s.call("capture", {"name": h[2:]}))
            seen.append((h[2:], r["lane"]["clock"]["aten_next"]))
    for site in sites:
        ok(s.call("break", {"action": "disable", "id": "b_" + site}))
    return seen


def arm(s, name, cut, act=None):
    ok(s.call("fork", {"name": name, "cut": cut}))
    ok(s.call("select", {"name": name}))
    receipt = ok(s.call("act", act))["receipt"] if act else None
    ok(s.call("motion", {"action": "step", "unit": "forward"}))
    dec, raw = outputs(s)
    return dec, raw, receipt


def donor_vector(a, window, regs, flags, site="mlp_001"):
    """Record a donor forward with MDB and read one site's output at its exit."""
    from mdb.discovery_runtime import ReplayForward
    from mdb.forward_recording import record_forward

    cap = record_forward(a.model, roots(window, regs, flags), model_identity="de_mamba-donor",
                         execution={"backend": "cpu-fp32"}, input_identity={"window_hex": bytes(window).hex()})
    rf = ReplayForward(cap)
    exit_pc = rf.catalog.resolve(site)["exit"]
    while rf.pc < exit_pc:
        rf.advance_one()
    return rf.read(rf.catalog.resolve(site + ":output")["value_id"])


def edit_and():
    """Why does de_mamba run `and $-16,%rsp` as add? Transplant the decoder vector from other /r encodings."""
    recipe = REPO / "debug/de_mamba/dev/fib-O2/root-recipe"
    a = adapter("de_mamba")
    _, window, before, _ = native_before("dev", "fib-O2", 0x401003)
    regs, flags = list(before.regs[:8]), {n: before.flags[n] for n in FLAGS}
    donors = {}
    for r in (0, 4, 5):
        w = bytearray(window)
        w[2] = 0xC4 | (r << 3)
        donors[r] = donor_vector(a, bytes(w), regs, flags)
    s = session_for(recipe)
    print(json.dumps({"cuts": prepare_cuts(s, ["mlp_001"])}))
    own = ok(s.call("inspect", {"selector": "mlp_001:output"}))["value"]["value"]
    import torch

    print(json.dumps({"recipe_decoder_equals_/4_donor": bool(torch.equal(torch.tensor(own), donors[4]))}))
    d = (donors[5] - donors[4])[0]
    order = d.abs().argsort(descending=True).tolist()
    print(json.dumps({"decoder_dim": d.numel(), "l2(/5-/4)": round(float(d.norm()), 3),
                      "l2(/0-/4)": round(float((donors[0] - donors[4]).norm()), 3),
                      "top8_coords_/5-/4": [(i, round(float(d[i]), 2)) for i in order[:8]]}))

    def show(name, dec, receipt=None):
        delta = ((int(dec["regs"]["RSP"], 16) - regs[4] + (1 << 63)) & M64) - (1 << 63)
        print(json.dumps({"arm": name, "RSP": dec["regs"]["RSP"], "RSP_delta": delta, "CF": dec["flags"]["CF"],
                          "len": dec["length"], "dest": dec["dest"], "imm": dec["imm"], "store": dec["store"],
                          **({"act": receipt["value_id"]} if receipt else {})}))

    show("baseline (no edit)", *arm(s, "baseline", "mlp_001")[::2])
    for name, vec in (("decoder := /5 (sub) vector", donors[5]), ("decoder := /0 (add) vector", donors[0])):
        dec, _, rc = arm(s, name.split()[2].strip("()") + "_transplant", "mlp_001",
                         {"selector": "mlp_001:output", "value": vec.tolist()})
        show(name, dec, rc)
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        v = donors[4].clone()
        v[0, order[:k]] = donors[5][0, order[:k]]
        dec, _, rc = arm(s, f"top{k}", "mlp_001", {"selector": "mlp_001:output", "value": v.tolist()})
        show(f"top-{k} coords of /5 pasted", dec)
    dec, _, _ = arm(s, "zero_decoder", "mlp_001", {"selector": "mlp_001:output", "value_spec": {"kind": "zeros_like"}})
    show("decoder zeroed", dec)


def bit32():
    """Sealed-panel exception held-081: where does the wrong RDI bit 32 come from?"""
    from mdb.discovery_cli import save_capture
    from mdb.forward_recording import record_forward

    f = json.loads((REPO.parent / "isa/outputs/response-quotient-r3-mamba/findings.json").read_text())
    ex = f["heldout_metrics"]["target_bit_exceptions"][0]
    window, regs = bytes.fromhex(ex["raw_window_hex"]), ex["input_register_words_before"]
    a = adapter("de_mamba")
    from isa_bench.adapters import sha256_file

    ck = sha256_file(a.artifact / "model.pt")
    cap = record_forward(a.model, roots(window, regs, dict.fromkeys(FLAGS, 0)), model_identity=ck,
                         execution={"backend": "cpu-fp32", "kind": "symbol-table-exception"},
                         input_identity={"window_hex": window.hex(), "register_words": regs, "flags_CF_ZF_SF_OF": [0] * 4},
                         metadata={"case_id": ex["query_id"], "checkpoint_sha256": ck})
    root = REPO / "debug/scratch/held-081/root-recipe"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    save_capture(cap, root)
    s = session_for(root)
    sites = ["mixers.0.out_proj", "mlp_002", "mixers.1.out_proj", "mlp_003", "mixers.2.out_proj", "mlp_004"]
    sel = {"mixers.0.out_proj": "affine_016", "mixers.1.out_proj": "affine_022", "mixers.2.out_proj": "affine_028"}
    print(json.dumps({"recipe": str(root.relative_to(REPO)), "cuts": prepare_cuts(s, [sel.get(x, x) for x in sites])}))

    def show(name, dec, raw):
        logit = raw["regs"]["value"][0][7]
        print(json.dumps({"arm": name, "RDI": dec["regs"]["RDI"], "bit32_logit": round(logit[32], 4),
                          "bit34_logit": round(logit[34], 4), "bit15_logit": round(logit[15], 3),
                          "len": dec["length"]}))

    dec, raw, _ = arm(s, "baseline", sel.get("mlp_004", "mlp_004"))
    show("baseline", dec, raw)
    for site in sites:
        cut = sel.get(site, site)
        for kind, spec in (("zero", {"kind": "zeros_like"}), ("x0.9", {"kind": "scale", "factor": 0.9})):
            dec, raw, _ = arm(s, f"{kind}_{site}", cut, {"selector": cut + ":output", "value_spec": spec})
            show(f"{kind} {site} output", dec, raw)


def live_session(a, window, regs, flags):
    """Record one forward in memory and open it as a discovery lane (no recipe on disk)."""
    from mdb.discovery import DiscoverySession
    from mdb.discovery_runtime import ReplayForward
    from mdb.forward_recording import record_forward

    cap = record_forward(a.model, roots(window, regs, flags), model_identity="de_mamba",
                         execution={"backend": "cpu-fp32"}, input_identity={"window_hex": bytes(window).hex(),
                                                                            "register_words": list(regs)})
    s = DiscoverySession(budgets={"max_native_operations": 4_000_000})
    s.add_lane("lane_00", lambda: ReplayForward(cap))
    return s


def bit32_check():
    """Does `scale mixers.2.out_proj x0.9` generalize, or did it only fix held-081? Same edit on every symtab case."""
    a = adapter("de_mamba")
    f = json.loads((REPO.parent / "isa/outputs/response-quotient-r3-mamba/findings.json").read_text())
    cases = [(e["query_id"], bytes.fromhex(e["raw_window_hex"]), e["input_register_words_before"])
             for e in f["heldout_metrics"]["target_bit_exceptions"]]
    for suite, prog, rip in (("dev", "fib-O2", 0x40121A), ("heldout_m4", "checksum-O2", 0x401183)):
        _, window, before, _ = native_before(suite, prog, rip)
        cases.append((f"{prog}@{rip:#x}", window, list(before.regs[:8])))
    base = cases[0][2]
    for k in (32, 47, 63):
        regs = list(base)
        regs[7] = (1 << k) - 1
        cases.append((f"RDI=2^{k}-1", bytes.fromhex("4883c7014883c7014883c7014883c7"), regs))
    for label, window, regs in cases:
        flags = dict.fromkeys(FLAGS, 0)
        want = symtab_program(window, regs)
        slot = REGS.index(want["slot"])
        s = live_session(a, window, regs, flags)
        prepare_cuts(s, ["affine_028"])
        res = {"case": label, "symtab": hex(want["regs"][slot])}
        for name, act in (("baseline", None),
                          ("x0.9", {"selector": "affine_028:output", "value_spec": {"kind": "scale", "factor": 0.9}})):
            dec, _, _ = arm(s, name, "affine_028", act)
            got = [int(dec["regs"][n], 16) for n in REGS]
            res[name] = {"target": dec["regs"][want["slot"]],
                         "target_exact": got[slot] == want["regs"][slot],
                         "other_regs_exact": all(got[i] == want["regs"][i] for i in range(8) if i != slot)}
        print(json.dumps(res))


def selector_check():
    """Do decoder coords 220/152 act as an add/sub selector on a real `sub $0x18,%rsp`?"""
    a = adapter("de_mamba")
    step, window, before, after = native_before("c_apps", "medium_sort-Os", 0x401004)
    regs, flags = list(before.regs[:8]), {n: before.flags[n] for n in FLAGS}
    print(json.dumps({"instr": "sub $0x18,%rsp @0x401004", "step": step, "window": window.hex(),
                      "RSP_before": hex(regs[4]), "native_RSP_after": hex(after.regs[4])}))
    add_w = bytearray(window)
    add_w[2] = 0xC4  # 48 83 c4 18 = add $0x18,%rsp
    add_vec = donor_vector(a, bytes(add_w), regs, flags)
    s = live_session(a, window, regs, flags)
    prepare_cuts(s, ["mlp_001"])
    own = __import__("torch").tensor(ok(s.call("inspect", {"selector": "mlp_001:output"}))["value"]["value"])
    for name, coords in (("baseline", []), ("paste add coord 220", [220]), ("paste add coord 152", [152]),
                         ("paste add coords 220+152", [220, 152]), ("paste whole add vector", list(range(256)))):
        v = own.clone()
        v[0, coords] = add_vec[0, coords]
        act = {"selector": "mlp_001:output", "value": v.tolist()} if coords else None
        dec, _, _ = arm(s, name.replace(" ", "_"), "mlp_001", act)
        delta = ((int(dec["regs"]["RSP"], 16) - regs[4] + (1 << 63)) & M64) - (1 << 63)
        print(json.dumps({"arm": name, "RSP": dec["regs"]["RSP"], "RSP_delta": delta, "len": dec["length"],
                          "dest": dec["dest"], "imm": dec["imm"]}))


def driver():
    """Run de_mamba as a program under the debugger: fib-O2 `_start` bytes, native initial state, ISADriver."""
    from mdb.discovery import DiscoverySession
    from mdb.discovery_isa import ISADriver
    from mdb.discovery_runtime import ReplayForward
    from mdb.forward_recording import record_forward
    from isa_bench.state import PageMemory, Trace

    a = adapter("de_mamba")
    pdir = REPO / "outputs/reference/dev/programs/fib-O2"
    tr = Trace(pdir)
    s0 = tr.state(0)
    code = PageMemory(pdir / "mem.bin").read(s0.rip, 24)
    regs = list(s0.regs[:8])
    flags = [s0.flags[n] for n in FLAGS]
    print(json.dumps({"program_hex": code.hex(), "rip": hex(s0.rip), "RSP": hex(regs[4]), "flags": flags}))

    def factory(*r):
        cap = record_forward(a.model, r, model_identity="de_mamba", execution={"backend": "cpu-fp32"},
                             input_identity={"window_bits": [int(v) for v in r[0][0].tolist()]}, max_operations=8192)
        return ReplayForward(cap)

    s = DiscoverySession(budgets={"max_model_forwards": 8, "max_native_operations": 100_000})
    s.add_lane("fib", factory, driver=ISADriver(code.hex(), regs, flags, max_instructions=3),
               reference_forwards=1, reference_operations=8192)
    ok(s.call("break", {"selector": "module:post.2", "edge": "exit", "id": "post2"}))
    for i in range(8):
        r = s.call("motion", {"action": "step", "unit": "instruction"})
        lane = r.get("lane", {})
        commits = [{"offset": c["before"]["byte_offset"], "->": c["after"]["byte_offset"],
                    "RBP": c["after"]["register_words"][5], "RSP": c["after"]["register_words"][4],
                    "flags": c["after"]["flags_CF_ZF_SF_OF"]} for c in r.get("commits") or []]
        print(json.dumps({"call": i, "ok": r.get("ok"), "status": r.get("status"),
                          "stop": [(c.get("kind"), c.get("id")) for c in r.get("stop_causes", [])],
                          "clock": lane.get("clock"), "commits": commits}, default=str)[:900])
        kinds = {c.get("kind") for c in r.get("stop_causes", [])}
        if not r.get("ok") or kinds & {"program_end", "instruction_limit", "effect_blocked", "unhandled_effect"}:
            print(json.dumps({"final_stop": r.get("stop_causes"), "error": r.get("error")}, default=str)[:900])
            break
    for i in range(3):
        rip = int(tr.steps[i + 1][0])
        st = tr.state(i + 1)
        print(json.dumps({"native_after_step": i, "rip": hex(rip), "RBP": hex(st.regs[5]), "RSP": hex(st.regs[4]),
                          "flags": [st.flags[n] for n in FLAGS]}))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
    raise SystemExit(0)
    {"sweep_ext": sweep_ext, "symtab": symtab, "edit_and": edit_and, "bit32": bit32,
     "bit32_check": bit32_check, "selector_check": selector_check}[sys.argv[1]]()
```

`debug/scratch/scripts/and_tour.mdb` is printed in full in §4. `debug/scratch/scripts/clocks.mdb` (§2):

```text
interface mdb-discovery-v1
step op
step math
step math
step law
into law
step math
finish law
step update
until mlp_004
step forward
```

### `debug/scratch/debug_many.py`

```python
"""stage=debug for many (model, suite, program) cells in one process: python debug/scratch/debug_many.py m:suite[:program] ..."""
import json
import sys

from hydra import compose, initialize_config_dir

from isa_bench.cli import CONF, _plain
from isa_bench.debug import run_debug

with initialize_config_dir(config_dir=str(CONF), version_base=None):
    for spec in sys.argv[1:]:
        model, suite, *program = spec.split(":")
        over = [f"model={model}", f"suite={suite}", "stage=debug"] + ([f"debug.program={program[0]}"] if program else [])
        info = run_debug(_plain(compose(config_name="config", overrides=over)))
```

### `debug/scratch/sel_lab.py`

```python
"""Selector lab for docs/debugging-with-mdb.md §9: which half of the readout input names the wrong family?

    PYTHONPATH=src:../saturn-pub/packages/mdb/src ../common/bin/common exec python debug/scratch/sel_lab.py \
        debug/sel_mamba/c_apps/tiny_hello-Os [more debug dirs ...]

The SSM selectors' runtime head reads cat(scan[first], scan[mean], scan[last], numeric) -> MLP -> 33 logits.
At the readout's entry this forks three arms from one cut: no edit, numeric features zeroed (scan only),
scan zeroed (numeric only), and prints the top family and the typed family's rank in each.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def ok(r):
    if not r.get("ok"):
        raise SystemExit(json.dumps(r, default=str)[:3000])
    return r


def main(dirs):
    import torch
    from mdb.discovery import DiscoverySession
    from mdb.discovery_cli import load_capture

    sys.path.insert(0, str(REPO / "models/_vendor/isa_r3"))
    from pan_isa.native import mount, runtime_module

    mount()
    labels = runtime_module().RUNTIME_NEURAL_LABELS
    for d in (Path(x).resolve() for x in dirs):
        bundle = json.loads((d / "bundle.json").read_text())
        typed = bundle["typed_family"]
        s = DiscoverySession(budgets={"max_native_operations": 4_000_000})
        s.add_lane("lane_00", lambda: load_capture(d / "root-recipe"))
        ok(s.call("break", {"selector": "mlp_001", "edge": "enter", "id": "readout"}))
        ok(s.call("motion", {"action": "continue"}))
        ok(s.call("capture", {"name": "readout"}))
        ok(s.call("break", {"action": "disable", "id": "readout"}))
        pooled = torch.tensor(ok(s.call("inspect", {"selector": "mlp_001:input"}))["value"]["value"])
        scan = pooled.shape[-1] - len(bundle["numeric"])
        assert scan == 48, scan  # SSMRuntimeHead: 3 x 16 pooled scan features
        arms = {"no edit": None}
        for name, keep in (("scan only (numeric := 0)", slice(0, scan)), ("numeric only (scan := 0)", slice(scan, None))):
            v = torch.zeros_like(pooled)
            v[..., keep] = pooled[..., keep]
            arms[name] = v
        for i, (name, value) in enumerate(arms.items()):
            ok(s.call("fork", {"name": f"arm{i}", "cut": "readout"}))
            ok(s.call("select", {"name": f"arm{i}"}))
            if value is not None:
                ok(s.call("act", {"selector": "mlp_001:input", "value": value.tolist()}))
            ok(s.call("motion", {"action": "step", "unit": "forward"}))
            probs = torch.tensor(ok(s.call("inspect", {"selector": "outputs"}))["value"]["value"])[0].softmax(-1)
            order = probs.argsort(descending=True).tolist()
            t = labels.index(typed)
            print(json.dumps({"case": str(d.relative_to(REPO / "debug")), "arm": name,
                              "top": [labels[order[0]], round(float(probs[order[0]]), 4)],
                              "typed": [typed, order.index(t) + 1, float(f"{float(probs[t]):.3g}")]}))


if __name__ == "__main__":
    main(sys.argv[1:])
```
