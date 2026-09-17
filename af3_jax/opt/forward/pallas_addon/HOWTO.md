# pallas_addon — the `exact` mode's Pallas levers (GLUT, ATTNCFG)

`patches/af3_pallas_levers.py` installs two bitwise levers into the model process before the kit script runs: `af3_jax_opt/levers_launch.py` does it
for `--mode exact`, and `big_launch.py` installs the same two at the sites the disengaged FlashPairformer kernels vacate under `--mode big`
(GLUT on one device only: under `--n_gpu P > 1` the row-sharded pair stack owns the triangle-multiplication site). Nothing here changes the
model, weights, inputs, recycles, samples or output files: both levers are bit-identical to stock inside one compilation-cache class
(`fast_inference/HOWTO.md`).

| lever | switch (set by the mode table, `af3_jax_opt/modes.py PALLAS_LEVERS`; never read from the caller's shell) | numerics |
|---|---|---|
| GLUT — TriangleMultiplication's gated linear unit as one fused kernel with a transposed, masked store | `AF3P_GLU_T=1` | bitwise |
| ATTNCFG — the pair attention's flash-attention tile configuration pinned | `AF3P_ATTN_CFG=64,64,4,3` (block_q, block_k, warps, stages) | bitwise |

## Lever mechanics

* GLUT: stock `TriangleMultiplication` does `tokamax.gated_linear_unit(act, W)` → `jnp.transpose(., (2,0,1))` → `*= mask[None]` — three ops, two
  extra HBM passes over a `[2·C=256, N, N]` bf16 tensor. The lever runs one Pallas-Triton kernel (the same tile configuration stock resolves through
  the same tokamax policy) whose epilogue stores each tile transposed and multiplied by the mask. Rounding points are unchanged: f32 accumulation
  over K → bf16 store of the two projections → activation in f32 → product → bf16 → mask multiply in bf16, exactly the stock dtype chain. When the
  mask dtype differs from the activation dtype the kernel stores unmasked and the stock `*=` promotes outside it (in AF3 inference both are bf16, so
  the fused path is taken). The kernel is the shared core's `opt_core.kernels.pallas_glut`, reached through the provider face
  `serve.glu_transposed_masked`; this module subclasses the stock Haiku module onto it (`make_patched_trimul_class`).
* ATTNCFG: pins tokamax's Pallas-Triton flash-attention tile configuration for the pair attention (`GridSelfAttention`) only. Stock resolves the
  heuristic `64,64,4,2` (tokamax's packaged H100 table for this shape does not parse on this stack). The bitwise class of configurations is
  `block_k = 64` with `num_warps ∈ {4, 8}`; `block_q` and `num_stages` never change bits; `block_k` 32 / 128 or 2 warps change the softmax / PV
  reduction order and are not bitwise (not used).

The caller's environment never selects these levers: every `AF3P_*` variable of the calling shell is stripped from the model process
(`stock/PINS.json stock_proof.must_be_absent_prefixes`), and the mode table alone sets the two switches above. Third-party attributions: `NOTICE`.
