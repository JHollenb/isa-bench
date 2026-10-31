# Original PAN-ISA models

`upstream/` is a frozen, byte-for-byte copy of the original ISA, stage, runtime and action-ledger packages from `~/domains/pan-isa`. `provenance.json` records every source SHA-256. `scripts/sync_pan_isa.py` explicitly refreshes that snapshot; campaign sealing preserves the exact snapshot used by each run.

`full_model.py` instantiates the original `FactorizedStage0PAN` through `FactorizedStage4PAN` networks with their original default dimensions. `native.py` also extracts and executes the exact original `TorchRuntimeNeuralHead` class AST, which the upstream source defines inside its training function. This avoids invoking its classical training loop. The runtime head has 15 byte positions, 8-dimensional byte embeddings, the original numeric context, two 96-unit ReLU layers and 33 outputs.

Training uses the common discovery data and Saturn checkpoint controller. It starts from fresh weights rather than borrowing an existing PAN checkpoint. Serving uses Torch CPU for the original neural head; a regression checks numerical agreement with PAN's pure-Python exported head. These results describe the new trained original architectures on the shared corpus, rather than reproducing an old PAN experiment's data, optimizer or backend.

`runtime_adapter.py` installs any of the four learners into the original `SPFPrimaryPanRuntime`. The same original typed proposal, confidence, profile, write-mask, memory and commit checks apply to every model. Static ELF loading, GDB state access, Linux service behavior and PS/PL routing remain shared infrastructure. Hardware XRT/FPGA execution requires the same external artifacts and devices as upstream.

The current source inventory is 25 Stage 0 forms, 16 Stage 1 forms, 18 Stage 2 forms, 125 Stage 3 forms and 268 Stage 4 forms: **452 staged forms**, plus **33 runtime selector families**. Stage 5 composes the staged state surfaces; it does not add a new trained neural head. This scope is a constrained x86 implementation, with upstream operand, mode, memory and floating-point limitations.
