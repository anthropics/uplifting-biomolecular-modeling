# flashpairformer — fused Pallas kernels for AlphaFold 3's TriangleMultiplication and triangle attention, and the diffusion-conditioning hoist (`fast`, `big`)

The `fast` mode's levers FPF_TRIMUL, FPF_TRIATT and FPF_HOIST. `af3_jax_opt/fpf_launch.py` runs `run_alphafold_flashpairformer.py`, which imports
`af3_flashpairformer` before the kit script's first forward; the import rebinds two Haiku classes of `alphafold3.model.network.modules` and, with
the hoist switch on, the diffusion sampler's per-step conditioning. Inference only (no custom VJP), zero weight changes, automatic fall-back to the
stock code outside the served shapes. The two switches are set by the mode table (`af3_jax_opt/modes.py` reads their names and on-values out of
`af3_flashpairformer/__init__.py`): `AF3_FLASHPAIRFORMER=both` (`off | trimul | triatt | both`) and `AF3_DIFFUSION_HOIST=1`; `--mode big` sets
them per its composition (its memory levers own some of these sites) and `--mode exact` never imports the package. A caller's own copies of these
variables are stripped from the model process (`stock/PINS.json stock_proof.must_be_absent_prefixes`). When the tree's TRIATT_XLA lever is on,
`patch.fused_with_core` runs the fused triangle-attention block with the attention between the Pallas prologue and epilogue handed to the support
library's provider; with it off the block is `grid_self_attention_fused`.

## What it does

- **TriangleMultiplication** → `FlashTriangleMultiplication`: a Pallas-Triton prologue kernel (input LayerNorm with f32 statistics + GLU
  projection/gate + mask, written directly as channel-major bf16 planes; the 'incoming' equation is handled by transposed writes), the cubic
  contraction on cuBLAS (`einsum('cik,cjk->cij')`), and a Pallas epilogue kernel (centre LayerNorm + output projection + sigmoid gating, the input
  LayerNorm recomputed instead of stored).
- **GridSelfAttention** (triangle attention, starting and ending node) → `FlashGridSelfAttention`: a Pallas prologue (LayerNorm + q/k/v + pair-bias
  projections written in the attention layout; the ending-node variant reads the pair activation transposed through the BlockSpec instead of
  materialising `swapaxes`), a Pallas flash-attention forward for `[B=N, S=N, H=4, D=32]` with pair bias and key mask (online softmax, f32
  accumulation, finite masking: a fully-masked row gives the same uniform average as stock, no NaN), and a Pallas epilogue (gating + output
  projection with transposed writes).
- **Diffusion-conditioning hoist** (`diffusion_hoist.py`, FPF_HOIST): the diffusion head's step-invariant pair conditioning
  (pair_cond_initial_norm/projection + 2 pair transitions, f32 N×N×128) and the 24 per-block pair-logit projections are computed once per sample
  call instead of at every one of the 200 denoising steps (stock recomputes them inside the 4×-unrolled `hk.scan`). Same parameters, same Haiku
  scopes (`hk.experimental.layer_stack(..., with_per_layer_inputs=True, name='__layer_stack_no_per_layer')` addresses the stock stacked weights;
  the sampler's rng key is drawn at the stock position). Implemented for the OF3-weights layout (`of3_weights=True`, per-block pair LayerNorm);
  with other weights the stock transformer path runs. It holds 24 × [16, N, N] pair logits for the whole sample call — the memory `--mode big`
  gives back above its size boundary.
- Parameters are fetched under the stock names (`projection`, `gate`, `center_norm`, `output_projection`, `gating_linear`, `left_norm_input`;
  `q_projection`, `k_projection`, `v_projection`, `pair_bias_projection`, `gating_query`, `output_projection`, `act_norm`) with the stock shapes and
  dtypes, so any checkpoint that loads into stock loads unchanged; the fork's `GlobalConfig.of3_weights` pair-bias transpose in
  `GridSelfAttention.__call__` is honoured via `getattr(global_config, 'of3_weights', False)`.
- **Served shapes** (else the stock path, decided per call site at trace time; the launcher's `SERVED` line reports what was fused): pair
  activation `[N, N, C]` in bfloat16 (AF3 inference's default `bfloat16='all'`), N a multiple of the kernel tile (all of the kit's buckets), C a
  multiple of 16 and ≤ 256 (pair stack C=128, template stack C=64), heads × head_dim ≤ 256 with head_dim ∈ {16, 32, 64}. float32 runs
  (`bfloat16='none'`) fall back to stock entirely. The kernels and the per-GPU tile tables are the shared core's (`opt_core/kernels/fpf_pallas/`,
  `opt_core/kernels/fpf_pallas_serve.py`): a GPU with its own table is served `tiles=own:<cc>` (9.0 = H100, 8.0 = A100), another 8.x / 9.x / 10.x
  part the generation's safe tile settings (`tiles=safe:<gen>`), and below compute capability 8.0 the kernels are held by name and the mode refuses
  before launch.

## Numerics class

Not bitwise: the kernels re-associate bf16 arithmetic (fewer bf16 rounding points than stock; run-to-run deterministic), and 48 pairformer layers ×
recycles carry the differences into confidence values and coordinates, at a fixed seed, by less than stock's own outputs differ across seeds. Rank
with several seeds and do not mix `fast` and `off` outputs in one ranking.
