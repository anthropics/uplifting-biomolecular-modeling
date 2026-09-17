# kernels.pallas — design note (the JAX-family provider and its Pallas / XLA members; portable FlashPairformer members for jax programs)

## The provider
One face over every carried Pallas (Triton lowering) and pure-XLA implementation ("row") of seven sub-layer ops and the measured cell table
`PALLAS_CELLS.json`; call faces in `serve.py` (jax imported when called, never at import): `attention` (core: softmax_k(scale q.k + bias[h]
shared over the batch rows + key mask) v; q/k/v `[B,S,H,D]` or `[B,H,S,D]`), `triangle_attention_block` (LayerNorm -> q/k/v/bias/gate
projections -> core -> gate -> output projection; act `[N,N,C]`), `triangle_multiplication` (module: LN -> gated projections * mask ->
contraction -> LN -> gated output projection), `glu_transposed_masked` (the AF3-form contraction prologue), `transition` (LN -> W1 (+b1) ->
relu | swiglu -> W2 (+b2)), `layer_norm`, `outer_product_mean`. Each face takes `word=` (row | arm `<row>[:<f32 word>][@<setting>]` | tier
word), resolves it with `select` (pure: table + words; jax line = major.minor and the device's cc key the cells), serves the first candidate
that engages and records a refusing row BY NAME (`Refusal(kind, row, fallback)`; `strict=True` raises instead of walking on); the decision is
recorded once per call class in `opt_core.cell_census`. `reference_*` in `serve.py` are the XLA statements (row `xla`), the numerics reference
of every row. Forward-only rows are wrapped (`forward_only`, `jax.custom_vjp`) so that differentiating THROUGH them raises `no_backward` by
name. f32 inputs: rows whose products are not full fp32 require the caller's precision word (`tf32` | `tf32x3` | `ieee`) in the arm.
Source-compiled Pallas rows and the XLA statements travel across compute capabilities (`portable_arm`); the prebuilt bridge rows
(`triattn_xla`, `native_xla` = `kernels.trimul_xla`) are keyed by architecture and refuse elsewhere by name.

## Members (what each kernel computes; all Pallas with the Triton lowering unless said)
* `cd_trimul/trimul_pallas.py` (row `cd_trimul`) -- the AF2-multimer fused-projection TriangleMultiplication, FORWARD and BACKWARD
  (`jax.custom_vjp`), any N, pair / intermediate channels powers of two: prologue kernel (LayerNorm + [left | right] gated projections * mask
  written directly as zero-padded channel-major planes) -> one batched GEMM (XLA / cuBLAS) -> epilogue kernel (center LayerNorm + output
  projection * gate). tanh-form sigmoid, fp32 LayerNorm statistics in registers; bf16 products for bf16, default-precision products for f32.
* `fpf_pallas/trimul_pallas.py` (row `fpf_trimul`, through `kernels.fpf_pallas_serve`) -- the AF3-family pairformer TriangleMultiplication,
  inference: K1 prologue (LN_in + GLU + mask -> channel-major planes; incoming planes written transposed) -> one XLA batched GEMM -> epilogue;
  bf16 (f32 hand-off of the contraction as setting `hi`; f32 / bias variants beside it).
* `fpf_pallas/triattn_pallas.py` (rows `fpf_core`, `fpf_block`) -- flash-attention FORWARD for triangle attention in the module layout
  `[B,S,H,D]`, bias `[1,H,S,S]` shared over B, key mask; online softmax in the exp2 domain, fp32 accumulation, finite negative for masked keys
  (fully masked rows -> uniform average, as the stock softmax); S a multiple of the q / k block.
* `fpf_pallas/transition_pallas.py` (row `fpf_transition`) -- LayerNorm -> SwiGLU -> W2 as ONE kernel per row tile (the 4C intermediate never
  reaches HBM), bias-free, bf16, rows a multiple of the tile; `mlp_transition/` (row `mlp_transition`) generalises it: relu | swiglu, optional
  Linear biases, the engine's LayerNorm variance form, inference forward (`make_transition_op(bwd='reference')` = custom_vjp with the XLA
  reference backward).
* `pallas_attn/af2_flash_pallas.py` (row `pallas_attn`, through `kernels.pallas_attn_serve`) -- flash attention with per-head pair bias
  `[H,Sq,Sk]` shared over B and key mask `[B,Sk]`, heads-major `[B,H,S,D]`, FORWARD + BACKWARD (`jax.custom_vjp`): exact online softmax forward,
  FlashAttention-2 recomputation backward, dQ in its own kernel (no atomics, run-to-run bitwise), d bias = per-batch dS partials reduced by XLA
  or summed over B inside a third kernel (`dbias='kernel'`). bf16 / f16 / f32 inputs, fp32 accumulation; head dims >= 16 powers of two (8 is
  zero-padded to 16 by the face; 24 / 48 refused by name). Shims select `pl.load / pl.store` or `plgpu.load / plgpu.store` per jax line.
* `rowshared_flash_pallas.py` (row `rowshared`) -- the same forward with one program serving R consecutive batch rows of one (q block, head):
  the bias tile is loaded and converted once per key block and applied to the R rows, each row keeping its own online-softmax state and the
  carried kernel's per-row arithmetic (bitwise per row); launch order `legacy` (q block fastest) or `grouped` (rows of a group fastest, so a
  bias row-block is read from HBM once per group); backward = `pallas_attn`'s. Refusals `index_ge_2p31` (an operand of one pallas_call past
  2^31 elements: the face chunks rows below that), `bad_rows`, `bad_order`, `bad_tiles`, `head_dim_lt_16`.
* `pallas_triatt/triatt_attn.py` + `attbwd_dkdv.py` (row `cd_triatt`) -- flash attention FORWARD + BACKWARD for few heads / small head dims /
  many batch rows sharing one bias: one XLA prep pass pads the bias key stride and writes the key mask as an fp32 additive row + a per-row dead
  flag; in-kernel the bias tile enters through the tensor core (`dot(I, bias_tile)`, exact, already in the MMA register layout); backward =
  K1' key-major dK / dV (no dS partials) and K2' dQ + the G-summed d-bias partials, no atomics; P and dS rounded to the input dtype before
  their products (the module's rounding points), fp32 statistics from the forward's lse. Head dims 16..128 powers of two.
* `cd_layers/` -- `layers_ln.py` (row `cd_ln`: LayerNorm over the channel axis, one row kernel forward and one backward for d_x; d_scale /
  d_offset are XLA reductions), `layers_transition.py` (row `cd_transition`: the AF2 ReLU transition, one kernel forward and one backward),
  `layers_opm.py` (the outer-product mean with its two contractions RE-ASSOCIATED in XLA so the `[N, c, c, chunk]` intermediate is never
  formed); `opm_pallas/opm_pallas.py` -- outer-product mean GEMM1 -> GEMM2 without the transposed intermediate, inference forward.
* Rows served from modules beside this package: `glut` (`kernels/pallas_glut`), `triattn_xla` (the XLA-FFI bridge over pre-compiled
  triangle-attention kernels, its own CELLS.json), `native_xla` (`kernels/trimul_xla`), and the stock rows `xla`, `xla_subbatch4`, `xla_sdpa`,
  `cudnn`, `tokamax*`.

## Numerics
Every row is a tolerance-class statement of its XLA reference unless its `rows` entry says bitwise (e.g. `glut` at the stock kernel's own
rounding points, `rowshared` per row vs `pallas_attn`): fp32 softmax / LayerNorm statistics and accumulators, 16-bit products for 16-bit
inputs, the arm's precision word for f32 inputs. The differentiable rows are `cd_trimul`, `pallas_attn`, `rowshared`, `cd_triatt`, `cd_ln`,
`cd_transition` (`has_backward`); the others refuse differentiation by name.

## Cells, loading, limits
`PALLAS_CELLS.json`: cells per (op, family word, jax line, cc, dtype, size) with the measured arms, refusal words and peak memory; `cell_for`
takes the smallest measured size at or above the call's; a cc without cells reads the nearest arch-compatible measured column
(`inherit_cc`), said in the census line; `tuned_for` names a launch setting only where the table records it; tier word big ranks arms by
measured peak memory (`memory_cost_arms`). Pallas rows compile through jax at first trace per shape / dtype (no run-time tuning); bridge rows
load prebuilt binaries through their own digest checks. `MODEL_OPT_LEVERS_OFF` words `pallas` / `pallas:<row>` switch rows off by name.
