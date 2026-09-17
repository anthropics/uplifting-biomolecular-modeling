# CUDA-graph capture / replay of the trunk fwd+bwd under grad (`ef2_stepgraph.enable(model, n_slots=…)`)

**Applies to:** the pinned Biohub transformers fork (`models/esmfold2`) and the esm cookbook's `binder_design` (STOCK.md §Pin). Applied at run
time by patching bound methods of the loaded model; no source edits.

## What
At design sizes the autograd trunk is launch-bound. One `GraphedGradSegment` per trunk pass of a step: captured once per shape with static
input buffers (the pair mask a live static input), the forward and backward graphs replayed; no retained autograd graph. Pool memory = one
pass's activations per slot, held permanently (the memory plan budgets for it). Installed last, over whatever trunk forward is installed.

## Numerics class
exact: bitwise vs eager (the same kernels replayed).
