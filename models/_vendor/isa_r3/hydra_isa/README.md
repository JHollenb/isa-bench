# Hydra-style ISA selectors

An independent, tiny bidirectional state-space learner for the shared PAN-ISA selector contract. Hydra here means the sequence architecture, separately from Hydra's configuration library.

The mixer has shared noncausal depthwise convolution, forward and reverse selective scans, a scalar transition per head, shifted directional outputs, an input-dependent diagonal and gated normalization. The two scans see only the current instruction's padded 15 byte slots (or the three stage-input tokens). There is no access to later program instructions or expected answers. State resets for each instruction; the native consumer carries architectural state.

Width 16, expanded width 32, four heads and state size 4 use a portable CPU implementation. This is a Hydra-style reference model, not a reproduction of the paper's benchmark or fused Triton/CUDA performance. See [the paper](https://arxiv.org/abs/2407.09941), [official implementation](https://github.com/goombalab/hydra) and [shared source](../isa_lab/ssm.py).

The same six-head vocabulary, supplied semantics, original discovery/feedback/heldout splits, optimizer budget and Saturn controller apply. This package deliberately uses `hydra_isa` so it does not shadow the Hydra configuration dependency.
