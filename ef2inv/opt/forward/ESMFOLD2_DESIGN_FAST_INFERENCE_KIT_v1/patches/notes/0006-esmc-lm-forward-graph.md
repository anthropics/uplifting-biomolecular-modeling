# CUDA-graph replay of the ESMC-6B language-model forward per padded length (`ef2_esmc_graph.enable(model)`)

**Applies to:** the pinned Biohub transformers fork (`models/esmfold2`) and the esm cookbook's `binder_design` (STOCK.md §Pin). Applied at run
time by patching bound methods of the loaded model; no source edits.

## What
The 80-layer LM forward runs under inference_mode once per design step and is launch-bound; the input length is fixed for a design trajectory,
so the forward is captured once per (batch, padded length, dtype, flags) and replayed.

## Numerics class
exact: bitwise vs eager as long as the same SDPA backend is selected (the capture records whatever eager dispatches to).
