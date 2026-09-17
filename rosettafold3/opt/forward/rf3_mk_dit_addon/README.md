# MK-DiT — single-launch megakernel for the RF3 token diffusion transformer (foundry rf3 @ 4010e3e2e)

`mkdit/mk2.py::MK2TokenTransformer` runs the 24-block token-level `DiffusionTransformer` of one denoiser call as ONE persistent Triton kernel
(grid = number of SMs, inter-block dependency counters; the S-side projections precomputed with cuBLAS before the launch) with rf3's
structure kept as is (kq-norm, the transition on the block input) and stock's bf16-mixed precision policy; outputs differ from stock's at
bf16 rounding level (a different summation order), identically from run to run. `mkdit/mkrf3.py` holds `MK1`, the row-persistent variant the
kit's `mkdit` lever names as its fallback structure. The kit installs the lever (`rosettafold3_opt/mkdit.py`, mode `fast`); it needs the
patched-file flags on (`RF3_CUDAGRAPH=1 RF3_HOIST=1`: the pair bias it reads is the hoisted one).

Use in code (any `RF3InferenceEngine` process with `RF3_CUDAGRAPH=1 RF3_HOIST=1`):

    sys.path[:0] = ['<this>/mkdit']
    import mk2; mk = mk2.MK2TokenTransformer(model.diffusion_module.diffusion_transformer); mk.hoist(Z_II)  # once per roll-out
    A_out = mk.forward(A_I, S_I)

One tile configuration per compute capability (`mkdit.CONFIGS`: 9.0 and 8.0 rows; numerics are configuration-specific; a card takes the row of
the highest key at or below its compute capability). Caveats: (1) the persistent kernel assumes all of its CTAs are co-resident (true on an otherwise idle GPU and inside the kit's single-stream
graph); (2) calls with `Beta_II` (the atom transformers' windowed attention) and token counts below the lever's size gate keep the class's own
forward, counted in the lever's line; a leading diffusion batch D runs D launches sharing the hoisted pair bias; (3) the hoisted pair bias costs
24·16·I²·2 bytes (0.76 GB at 705 tokens).
