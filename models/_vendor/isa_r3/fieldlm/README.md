# FieldLM ISA models

`full_model.py` adds multi-axis learned records for all PAN stage forms and runtime families, including conditional-branch flag charts. Its native FieldFabric reader shares an immutable indexed view while keeping query frames and materializations disposable. The full lane measures this reader and feeds its selected effects through the same original PAN consumer as the neural models.

`model.py` induces opcode-to-action records from discovery transitions and queries them through the existing `RoleChartField`/`FieldFabric` engine. Integer role coordinates bind byte identity and selected action. The supplied candidate operator then computes the exact transition. Fitting is structural, with no gradient optimizer; Saturn captures the records, fabric state, process RNG and cursor.

This machine-byte adapter uses FieldLM's existing addressed executor. It does not use the public default's sparse natural-language encoder or claim learned parsing/role induction. The next step is learned effect programs with shared register/operand roles, followed by editable instruction schemas and memory/control-flow programs.

The starter detaches/reconstructs its consumer for each proposal to keep evaluation from mutating training caches. That overhead is counted. A resident read-only query path is a measured optimization target, not a current performance advantage.
