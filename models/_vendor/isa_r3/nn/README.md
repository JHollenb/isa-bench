# Neural network ISA models

`full_model.py` expands the independent NN control to the complete PAN native scope. It uses small MLP heads over the same supplied decoded opcode/flag inputs as PAN's staged networks, plus a 15-byte runtime classifier. Its shared native consumer now executes memory, branches, stack operations, complex scalar forms, SIMD and exceptions. It remains a separate architecture from the original model in `pan_isa/`.

`model.py` provides the initial byte-embedding/MLP selector: two observed byte positions, seven state/length features, a 64-unit hidden layer and eleven candidate logits. It learns discovery-derived action assignments with supervised gradient updates. Saturn evaluates short futures through the same isolated and autonomous consumers and captures parameters, optimizer, gradients, mode, RNG and cursor for exact restore.

The supplied candidate library computes architectural values. This is the NN control for candidate-selection-v1, not direct arithmetic learning. The next model will predict typed write masks and effects directly. Keep that model's result track separate.
