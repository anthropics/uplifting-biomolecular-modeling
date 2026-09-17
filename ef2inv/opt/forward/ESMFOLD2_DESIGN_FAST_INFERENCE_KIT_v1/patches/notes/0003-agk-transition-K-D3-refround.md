# K-D3 — pair transition fwd+bwd with stock rounding points (`transition='refround'` | `'refround_lean'`)

**Applies to:** the pinned Biohub transformers fork (`models/esmfold2`) and the esm cookbook's `binder_design` (STOCK.md §Pin). Applied at run
time by patching bound methods of the loaded model; no source edits.

## What
Replaces LayerNorm -> Linear(w12) -> SiLU*gate -> Linear(w3) -> +residual by a Triton LayerNorm (fp32 statistics, bf16 out), the same cuBLAS
GEMMs nn.Linear issues and two Triton elementwise SwiGLU kernels that round exactly where eager PyTorch rounds; no dW / dgamma / dbeta.
`refround_lean` additionally drops the saved [M, 2h] activation and recomputes LN-out + the W12 GEMM in backward (bitwise-equal to `refround`;
less activation memory per kept block, one extra GEMM per block in backward). fast / big use `refround_lean`; on compute capability 9.0 its
forward runs on the t16 kernel (`k/ef2_t16_transition.py`).

## Numerics class
fast: reordered fp32 accumulation, same bf16 rounding points (fp32-internal differences: LN reduction order, 1/sqrt vs rsqrtf, libdevice expf,
IEEE division where tl.math.div_rn exists, GEMM M-shape N² vs 64N). K-D2 (the vendored fused LN+SwiGLU: rounding points changed) is selectable
by argument and used by no mode.
