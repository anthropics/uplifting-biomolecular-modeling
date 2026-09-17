# PR: Memoize the SDPA sliding-window mask in SWA3DRoPEAttention

The boolean window mask (allowed, valid) depends only on (indices, B, N, half_window) and is identical for every SWA block and every diffusion step of one sample() call, but is rebuilt on every forward (n_blocks x n_steps times per fold; ~15% of sampler wall at 700 tokens on H100 in the SDPA fallback path). This patch memoizes it in a small module-level LRU keyed on the indices tensor object (id) + shape. Outputs are bit-identical (unit test: 48/48 cases incl. cold/cached, with/without indices). Only the SDPA fallback path is touched (flash-attn path unchanged).

Diff: `U1_swa_sdpa_mask_memoize.diff` (against Biohub/transformers @ ef32577f55da19a4989cd7b22e004dc43a4998cb).
