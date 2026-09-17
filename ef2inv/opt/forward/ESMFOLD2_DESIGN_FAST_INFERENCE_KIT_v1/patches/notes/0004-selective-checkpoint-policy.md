# Activation-checkpoint policy for the 24-block trunk (`checkpoint='none'|'ckpt:m'|'keep:m'|'every:k'|'block'`)

**Applies to:** the pinned Biohub transformers fork (`models/esmfold2`) and the esm cookbook's `binder_design` (STOCK.md §Pin). Applied at run
time by patching bound methods of the loaded model; no source edits.

## What
Stock checkpoints every PairUpdateBlock under grad and therefore recomputes the whole trunk forward inside backward. `none` keeps all
activations (fastest, most memory), `ckpt:m` checkpoints the first m blocks and keeps the last 24-m, `block` is stock. The kit does not pick
m by hand: `ef2_bwd_ckpt` chooses the policy that fits the free device memory (`k/README.md`).

## Numerics class
exact: bitwise (identical kernels; only the redundant recomputation is removed). Unit test: `k/test_ef2_bwd_ckpt.py`.
