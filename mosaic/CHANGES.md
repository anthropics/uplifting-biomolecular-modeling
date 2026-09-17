# mosaic kit — what changes vs stock

Stock = `mosaic` / `joltz` / `boltz` at the pins (STOCK.md). Each lever is a module the kit installs over one upstream function or class at
start-up under a kit mode (the per-step levers ship as the `mosaic.fast` sub-package `run.sh install` places inside the installed `mosaic`), or a
start-up replacement the driver makes; `off` loads none. A mode is all of its levers; `fast` includes `exact`'s start-up levers except P3, and
`big` includes every `fast` lever. Lever ids are the ones printed on the run's `LEVER name=…` lines; the module name follows each id below.

## exact — outputs identical to stock

Stock's arithmetic unchanged (start-up levers only); with the pinned autotune results every process of one GPU type and stack reproduces the
warm-up's design bitwise at the same seed, which stock alone does not do across fresh processes.
- `P1` repro_cache — a persistent XLA compilation cache under `$MOSAIC_OPT_CACHE_ROOT/<shape>/` through the shared core's
  `opt_core.jax_design.pcc`; in `exact` the autotune results are also dumped once by `warm` and loaded by every later process, which is what
  makes the design bitwise across processes of one GPU type and stack. Numerics: bitwise (caching only). Steps aside: on a shape not yet
  warmed (`p1=none(…)` naming the shape, together with P3); in `fast` / `big` when `MOSAIC_OPT_CACHE_ROOT` is unset (`p1=none(…)`).
- `P2` fastload — the torch `Boltz2` module is constructed with its random initialisers disabled, since the strict checkpoint load
  overwrites every parameter; driver flag `--weights fastinit`. Numerics: bitwise (identical parameters; an all-leaves digest is recorded on
  every run). Steps aside: on the in-process route (`driver_only`: a call-site replacement only the driver makes).
- `P3` frozen — the shape's featurized inputs written once by `warm` (`features_<shape>.npz` + sha256) and loaded instead of re-featurizing
  (boltz's featurizer re-draws `ref_pos` in every process). `exact` only. Numerics: bitwise. Steps aside: shape not warmed; in-process route.

## fast — within stock's seed-to-seed variation

This mode binds FlashPairformer kernels via the shared core (opt_core). Numerics class: TF32 sampler matmuls, a bfloat16 pair track and
re-associated reductions — not bitwise with stock; deterministic run-to-run in one process.
Per-step levers, installed in this order after the upstream import and before any trace (P1 in its cache-only form and P2 ride along):
- `E1` dead_template — when the featurized template mask is all zero (no `TargetChain(template_chain=…)`), the template module's
  identically-zero addend is skipped forward and backward. Numerics: exact algebra; XLA's kernel choices for the changed program can differ,
  hence not bitwise. Steps aside: a real template, or a loss built outside `Boltz2.build_loss` / `build_multisample_loss` / `model_output`
  (`aside=<reason>` on its line; the template module runs as stock).
- `P6` precision — the structure sampler's matmuls at TF32 (`jax.default_matmul_precision("high")`) instead of stock's IEEE float32
  (`highest`); trunk and confidence unchanged. Setting `diffusion=high`. Numerics: TF32 matmuls. Steps aside: never.
- `K1` flashattn — the attention core of `joltz.TriangleAttention` (logits, key mask and pair bias, softmax, value contraction) served
  forward and backward by the shared core's attention provider `opt_core.kernels.pallas` (`serve.attention`, kind `tri`) at the mode's tier
  word; the module keeps its LayerNorm, projections and gating. Kernel: FlashPairformer triangle attention via the shared core — CUDA-native
  (sm_90a / sm_80) through the XLA binding `triattn_xla` (forward; Pallas backward) or Pallas, chosen per call from compute capability, dtype,
  head geometry and token count. Numerics: online-softmax re-association, TF32 / bf16 operands. Steps aside: a call the provider does not serve runs
  joltz's statement under a named, counted reason; a run with zero served calls fails by name (`nothing_served`).
- `E10` layer_unroll — `joltz.DiffusionTransformer2`'s scanned layer stacks (token transformer, atom encoder / decoder) run as a static
  loop over upstream's layer function, so weights are read at static offsets. Numerics: XLA fuses across the exposed layer boundaries. The
  sampler's executables take longer to compile, once per process (P1 caches them). Steps aside: never.
- `F6` trimul_layout — `TriangleMultiplicationOutgoing` / `Incoming` kept in one channel-major layout: input projections contracted to
  `[2C, N, N]`, gating and masking there, the triangle product as a channel-batched matmul, `norm_out` along the leading axis, the output
  projection contracting back; none of stock's layout transposes. Setting `cmajor`. Numerics: same contractions, other dimension numbers.
- `F8` trimul_fused (needs F6) — F6's call served by the shared core's triangle-multiplication provider (`opt_core.kernels.pallas`,
  `serve.triangle_multiplication`) at the mode's tier word. Kernel: FlashPairformer triangle multiplication via the shared core — Pallas or XLA
  binding (fused LayerNorm + projections + gates + mask as channel-major planes, one batched GEMM, LayerNorm + output projection + gate; a fused
  backward where provided), chosen per call from compute capability, dtype, token count and whether the call is differentiated. Numerics: fused-kernel rounding. Steps aside: calls the provider
  does not cover run F6's XLA body, counted by name on its line.
- `F9` transition_fused — the trunk's bfloat16 pair transitions (`joltz.Transition`: LayerNorm → fc1|fc2 → silu(a)·b → fc3, hidden 4c,
  bias-free) as one fused forward and one fused backward kernel under `jax.custom_vjp`, with P7's casts and the block's residual add inside
  (setting `zres`; `sub` leaves them to XLA); the `[rows, 4c]` intermediates never round-trip HBM; float32 transitions stay joltz's. Kernel:
  the kit's own Pallas kernels (Triton lowering, compute capability 8.0 or newer; `mosaic_fast/transition_fused.py`, tile settings `CFG`).
  Numerics: bf16 re-association. Steps aside: P7 off (`p7_tz_off`) or a GPU whose probe fails (`probe_failed:<kind>`); an unexpected layout
  is counted unserved and fails the run.
- `P7` halfpair — the trunk pairformer's pair track (triangle multiplications, triangle attentions, pair transition; setting `pf+msa` adds
  the MSA module's pair blocks) on bfloat16 operands: parameters and the pair activation cast at each sub-layer boundary, float32
  accumulation, the output cast back before the residual add; LayerNorm statistics, the residual stream, the sequence track and everything
  outside the blocks stay float32 — the policy upstream Boltz-2 applies to its torch trunk (`bf16-mixed`). Installed last: it wraps the
  pairformer call F6, F8, K1 (and P5) have bound. Numerics: bf16 operand rounding. Steps aside: never (configured on with no pairformer block executed fails
  the run by name, `nothing_served`).

## big — lowest peak GPU memory

- `P5` memlevers — grouped two-level rematerialisation of the trunk's pairformer blocks (8 per group) with sub-block remat of the triangle
  multiplications and transition, and query-row-chunked triangle attention (un-chunked calls go to K1), at every input size. Setting
  `pf8+sub`; the optional word `fit` applies the schedule only where the un-rescheduled step would not fit the memory pool
  (`--levers big+P5=pf8+sub+fit`). Numerics: as `fast`, plus re-associated accumulation under chunking. Steps aside: never at `pf8+sub`.
- `K1` and `F8` run at the provider's tier word `big` (the served kernel of lowest peak memory per call); every other `fast` lever at the
  same setting.

## Every mode

- `off` runs the kit driver with stock's weight load and no lever (`opt/mosaic_opt/stock_design.py`); two observers ride every mode, `off`
  included, and change nothing: `numstate` (JAX / torch numeric switches recorded before the design and compared after) and
  `fastload.fingerprint` (the parameter digest).
- Stock exceptions: none (STOCK.md). No opt-in upstream fix ships; `upstream_issues/` is text only.
- Cards: `configs/h100.env`, `configs/h200.env` and `configs/a100.env` differ only in the card name, compute capability and stack-key example;
  modes and lever sets are the same on all three, and the card-dependent kernel choices are made inside the shared core's providers (K1, F8) by compute
  capability. A lever that cannot engage on the card seen is a refusal naming the lever (`lever_refused: <id>`), never a substitute.
- The shared core (`common/opt_core`, minimum version in `opt/pyproject.toml` `[tool.opt_core]`) provides P1's cache primitive, the K1 / F8
  kernel providers, the ACTIVE / LEVER line grammar and the out-of-memory classifier; `mosaic_opt.core_gate()` is the first statement of
  every entry point and refuses by name (exit 3) on an absent or older core.

## Switches

- `MODEL_OPT_LEVERS_OFF=<id>[,<id>…]` — the mode without the named levers: `levers_off=<ids>` on the ACTIVE line, `-off[<ids>]` on its row
  token, `LEVER name=<id> state=off reason=levers_off:MODEL_OPT_LEVERS_OFF` per lever; a lever another one needs is refused rather than
  dropped (`lever_needs`: F8 needs F6), an unknown id is refused (`levers_off_unknown`).
- `--det 1` — `PYTHONUNBUFFERED=1` and the driver's asserted numeric settings in every mode; `--det 0` (default) exports nothing.
- `--allow-partial` — `design` / `warm`: a lever the driver's manifest does not show applied is recorded (`partial=<ids>` on the ACTIVE /
  DONE lines) instead of exit 3; `check`: a plan that would be partial passes instead of exit 3.
- Kit variables that change lever behaviour: `MOSAIC_OPT_CACHE_ROOT`, `MOSAIC_OPT_CACHE_DIR`, `MOSAIC_OPT_FORCE` — values and defaults: STOCK.md.
