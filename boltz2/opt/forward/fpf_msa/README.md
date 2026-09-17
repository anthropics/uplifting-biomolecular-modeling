# fpf_msa — the Boltz-2 entry points of the shared core's fused MSA-module cells

`fpf_msa.boltz2:outer_product_mean` and `fpf_msa.boltz2:msa_pair_weighted_avg` bind the shared core's fused MSA-module Triton cells
(`opt_core.ops.msa_opm`, form `mask_norm`; `opt_core.ops.msa_pwa`, form `masked`) to Boltz-2 2.2.1's `OuterProductMean.forward(m, mask, chunk_size)` /
`PairWeightedAveraging.forward(m, z, mask, chunk_heads)` signatures: the outer-product mean with the `[N, N, c_hidden²]` outer product never
materialized (LayerNorm + `proj_a` / `proj_b` + mask in a prologue kernel, then the outer-product GEMM over the MSA depth with the `c_hidden² → c_z`
projection fused into its epilogue) and the pair-weighted averaging (the LayerNorm / `proj_m` / `proj_g` / sigmoid prologue, the pair-bias softmax per
head, the weighted average over tokens with the gate and `proj_o` epilogue). Installed by `opt/boltz2_opt/msa_kernels.py` under `BOLTZ_FPF_MSA=opm,pwa`
(`fast`, `big`) in place of `OuterProductMean.forward` / `PairWeightedAveraging.forward`; unsupported dims raise `FPFFallback` and the stock module
serves, counted. The kernels, their pinned tile tables (per card) and `CFG_VARIANTS` live in the core packages; this directory holds the Boltz-2
signatures only (`boltz2.py`) and the fallback class import (`_compat.py`).

Runtime monkeypatches only; no boltz file is modified. Scope limits: `diffusion_samples` ≥ 1 (the kernels see the trunk only); the affinity module is untouched.
