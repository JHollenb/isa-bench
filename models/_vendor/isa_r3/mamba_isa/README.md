# Mamba-style ISA selectors

An independent, tiny PyTorch selective state-space learner for the shared PAN-ISA selector contract. Each of five stage heads scans three tokens (BOS, operation, flags); the runtime head scans the current instruction's padded 15 byte slots and combines its output with the original 42 numeric features. State resets for each instruction. Architectural registers, flags and memory carry between instructions through the shared native consumer.

The mixer uses input-dependent delta/B/C, a stable diagonal transition, causal depthwise convolution, a learned skip and gated readout. The CPU scan uses an associative prefix implementation with a recurrence/gradient parity test. Width 16, expanded width 32 and state size 4 keep the experiment small.

This is a Mamba-style reference implementation, not the official fused CUDA implementation or a pretrained Mamba checkpoint. See [the paper](https://arxiv.org/abs/2312.00752), [official implementation](https://github.com/state-spaces/mamba), [shared mixer source](../isa_lab/ssm.py) and [campaign config](../conf/study/ssm_native.yaml).

The six selectors learn supplied operation vocabularies; parsing, arithmetic and state commit are supplied by PAN-ISA. Training uses the same discovery suite and AdamW update budget as NN/PAN-ISA, with Saturn checkpoint futures, consumer feedback and exact rollback. Heldout labels do not select training branches.
