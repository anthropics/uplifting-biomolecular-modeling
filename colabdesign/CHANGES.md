# ColabDesign kit — what changes vs stock

Stock = BindCraft's design step @ `efb5bfeb` on ColabDesign 1.1.3 at the pin (STOCK.md). Each lever is a module the kit installs over one
ColabDesign function or haiku class at start-up under a kit mode, in the process that builds the design model; `off` loads none. A mode
is all of its levers (`opt/colabdesign_opt/registry.py` is the lever registry, `modes.py` the mode table); `fast` includes `exact`'s
levers. Lever names are the registry ids; each prints one `[colabdesign-opt] LEVER name=… state=<on|skipped|off> …` line per process
(`trimul` prints as `name=trimul_pallas`, `pallas` as `name=F1.pallas_attn`).

Speed, stated without figures (as the README's At a glance): the comparison statistic is equal-work time per design step (arms compared only inside matching (stage, recycle-count) cells of BindCraft's schedule; headline = the 1-recycle soft-stage cell every trajectory runs): `exact` steps at stock's pace or slightly faster and starts each process and trajectory sooner; `fast` steps faster than stock at every size, modestly at the smallest token counts and by a growing factor above 384 tokens, where both design executables run unchunked.

## exact — outputs identical to stock

- `lowercache` — the design step's two programs (`fn`, `grad_fn`) keep their COMPILED executables per call signature beside the
  compile cache (`…/pcc/<stack key>/lowered/<key>/`) and every later model build — a new process, or the next trajectory of the same process —
  calls them directly (`jax.experimental.serialize_executable`; an in-process memo serves trajectories 2..N without touching the disk).
  Why it exists (the cold start, in one paragraph): the compile cache already makes XLA compilation a one-time cost per machine, but ColabDesign
  builds a fresh `mk_afdesign_model` for every trajectory (BindCraft does the same), and jax then re-TRACES the haiku multimer model, re-LOWERS it
  to StableHLO — the kit's Pallas kernels lower with the graph — and re-hashes the module for the cache key before it can even look the
  executable up — a fixed cost per trajectory, larger under `fast` (its kernels lower with the graph), paid on every trajectory, in-process or not.
  That fixed cost is why, without this lever, `fast`'s per-step advantage barely shows per trajectory at small token counts. With `lowercache` a
  configuration met before costs one executable load plus one trace for the levers' census (no lowering, no compile) once per process and nothing on
  later trajectories of that process: one trajectory per process gains mostly the compile and trace removed, and several trajectories in one process
  (the BindCraft loop) gain the per-step speed as well. Numerics: exact — the loaded executable is the compiled executable;
  design.pdb / design.fasta / trajectory.jsonl are byte-identical with and without the lever (exact ≡ stock ≡ exact-no-lowercache; fast ≡
  fast-no-lowercache) and `COLABDESIGN_OPT_LOWERCACHE=relower` (re-lower on load, compare the StableHLO sha256) reads `relower=same` on the design
  programs. Key = stack (jax / jaxlib / PJRT plugin / GPU, XLA_FLAGS and every JAX_* / XLA_* / MODEL_OPT_* / AF_PALLAS_* variable that is not a path,
  matmul precision) + code content hash (colabdesign, colabdesign_opt, opt_core: an edited tree is a new key) + lever set + model configuration
  (args, protocol, lengths, haiku config, loss callbacks) + the call's abstract signature. XLA-FFI kernel targets (lever `txla`) are registered
  before the first load. Evidence: `LEVER name=lowercache state=on … calls=<n> memo_hits=<n> loads=<n> stores=<n> traced=<n> fallbacks=<n>
  load_s=<s> traced_s=<s> … relower=<off|same:n|DIFFERENT:n> why=<none|…>` (calls>0 = applied). A/B word: `<mode>-no-lowercache`.

- `compilecache` — jax's persistent compilation cache switched on before the first compile, in a directory keyed by jax, jaxlib, the CUDA
  plugin and the GPU product (or `JAX_COMPILATION_CACHE_DIR`), with XLA's per-fusion autotune results persisted beside it: a process whose
  design executables were compiled before on this machine loads them. Numerics: bitwise (caching only). Steps aside: `reason=cannot_run`
  on a cache directory it cannot create or write. Shared core: `opt_core.jax_design.pcc`.
- `parcompile` — appends XLA's LLVM module-parallel compilation flags (`--xla_gpu_enable_llvm_module_compilation_parallelism=true
  --xla_gpu_force_compilation_parallelism=<min(16, usable CPUs)>`) to `XLA_FLAGS` after ColabDesign's own assignment and before jax
  creates a backend. Numerics: bitwise (compilation scheduling only). Steps aside: `reason=cannot_run` if a jax backend already exists.
- `hoist_prev` — ColabDesign's `_af_design._recycle` and `run` re-stated so the recycle-0 `prev` zeros are built on the device once per
  shape and, with one model per step, `aux["prev"]` stays the device arrays the model returned instead of a host mean of one array (zeros
  canonicalised as stock's host path does); several models per step take stock's host path, counted `multi_model_stock_path=`.
  Numerics: bitwise (same arithmetic, same order). Steps aside: `reason=cannot_run` when either method body is not the pinned one (sha256).

## fast — within stock's seed-to-seed variation

This mode binds FlashPairformer kernels via the shared core (opt_core).

- `nosub`, `nosub_fn` — ColabDesign's `_prep_model` replaced so the forward+backward (`grad_fn`) and the forward-only (`fn`) design
  executables are traced without sub-batch chunking; stock chunks both by 4 above 384 tokens. Numerics: XLA fuses and accumulates the
  unchunked program in another order. Steps aside: `reason=gated` at or below 384 tokens (the programs are then stock's).
  Shared core: `opt_core.jax_design.subbatch_policy`.
- `trimul` — `modules.TriangleMultiplication` rebound to a same-named subclass with stock's parameter tree: input LayerNorm + projections +
  gates as one prologue kernel, the triangle product as one batched GEMM, centre LayerNorm + output projection + gate as one epilogue
  kernel, a custom_vjp backward of the same shape. Numerics: bf16 products, fp32 accumulation, fewer bf16 rounding points than stock's
  op-by-op program. Kernel: FlashPairformer triangle multiplication via the shared core — Pallas (`opt_core.kernels.pallas`, sm_90 / sm_80). Steps aside:
  a module without fused projection weights runs stock's class (`not_fused`); no GPU lowering → `reason=cannot_run`.
- `triatt` — every `modules.Attention` call (triangle start/end, MSA row with pair bias, template; extra-MSA row attention's head dim 8
  padded to 16 in-kernel) on flash-attention kernels written for AF2's shapes: pair bias and mask read in-kernel, several (batch, head)
  pairs per program, exact online softmax with fp32 statistics, a fused backward. Numerics: bf16 rounding points move against stock's
  op-by-op attention. Kernel: FlashPairformer triangle attention via the shared core — Pallas (`opt_core.kernels.pallas`, sm_90 / sm_80; the same kernel serves the MSA-row and template calls). Steps aside: calls
  with fewer than 16 keys (MSA column attention over 2 sequences) run stock's method (`below_keys_rule`). Supersedes `pallas` — the shared
  core's generic Pallas flash-attention kernel (`opt_core.kernels.pallas_attn`) over the same calls — which stays registered, prints
  `state=off reason=replaced`, and is restored by `--mode fast-no-triatt`.
- `opm_fold` — `modules.OuterProductMean` re-associated: the right-projection × output-weight contraction formed first in f32, then one f32
  GEMM instead of the [N,N,c,c] outer product, chunk scan and output transpose; masks, normalisation and bias as stock. Numerics: f32
  re-association (TF32-class products). Kernel: plain JAX through the shared core's provider. Steps aside: above 32 MSA sequences the
  stock module runs (`gated_s_gt_32`; not reached at BindCraft's settings).
- `ln` — every `common_modules.LayerNorm` over a power-of-two channel count in [32, 1024] on one row kernel forward and one backward (f32
  statistics; each instance's eps, axes and variance form honoured). Numerics: another reduction order. Kernel: the shared core's
  LayerNorm kernel (Pallas, `opt_core.kernels.pallas`). Steps aside: the 384-channel single-representation norms run stock's method
  (`channels_not_pow2`). Installed after `trimul`, whose fused module absorbs its own norms.
- `proj` — the q, k, v and gating input projections of every kernel-served attention call as one GEMM over the normalised activation
  instead of four contractions (one dX and one dW GEMM in the backward). Numerics: forward terms as stock, gradients re-associated.
  Steps aside: head dim < 16 goes to the class below (`head_dim_lt_16`); without an attention-kernel lever in the run (`triatt`, or
  `pallas` when restored) → `reason=no_attention_kernel`.
- `transition` — every `modules.Transition` (LayerNorm → Linear(C→4C) → relu → Linear(4C→C)) on a fused ReLU-transition kernel, one
  kernel per direction with the [M, 4C] intermediate kept on-chip per row tile; stock's parameter names, dtypes and bf16 rounding points.
  Numerics: the 4C contraction accumulated in f32 over on-chip chunks (another association than cuBLAS's). Kernel: FlashPairformer
  transition via the shared core — Pallas (`opt_core.kernels.pallas`). Steps aside: `dtype_not_bf16`, `channels`, `intermediate_width`, `rank`,
  `below_size_rule`, or a provider cell naming stock's op (`cell_xla`) run the stock class; no GPU lowering → `reason=cannot_run`.
- `txla` — the forward-only executable's attention calls (recycle 0; BindCraft's forward-only predicts) through the shared core's
  triangle-attention bridge, an XLA custom call whose per-card table picks the kernel per call (printed as `rows=`). The op is a
  `jax.custom_vjp` whose rules are `triatt`'s, so the graded pass (`grad_fn`, `hk.remat` included) is `triatt`'s bit for bit and only
  recycle 0's outputs move within bf16 rounding. Kernel: FlashPairformer triangle attention via the shared core — CUDA-native (sm_90a / sm_80) and Triton
  ahead-of-time binding (`opt_core.kernels.triattn_xla`); forward only. Steps aside: calls below 16 queries / keys or head dim (`small_call`) and calls the bridge declines (`refused`)
  stay on `triatt`; without `triatt` in the run → `reason=no_attention_kernel`.

`trimul`, `triatt`, `ln`, `opm_fold` and `transition` bind their kernels through one module, `kernels/provider.py`, by the tier word `fast`:
per cell (jax line, card, layer family, size bucket, direction) the shared core's provider names the kernel among those the kit's adapter
binds, else stock's operation by name (`fallback_by=cell_xla:n`); the lever's line prints `provider= word= row= tier=` and its census
(`served= fallback= fallback_by=`). An undeclared fallback word, or a lever that installed and served no call, ends the run
`NOT ACTIVE: partial activation` (exit 3); a declared structural word runs stock's op by name and the run exits 0.

## Every mode

- No stock exception and no upstream fix exists (STOCK.md): every mode runs BindCraft's design step and ColabDesign as shipped.
  Configuration alone is refused (exit 3): an unknown mode word, ColabDesign off its pinned commit, the shared core missing or older than
  the kit's pin (`opt/pyproject.toml` `[tool.opt_core] version`); `--mode` disagreeing with `COLABDESIGN_OPT` is a usage error (exit 2).
  The card and the stack are named on the `ACTIVE` line (`gpu=`, `stack_drift=`), never refused.

## Switches

- `--mode <mode>-no-<lever>[-no-<lever>…]` (or the same word in `COLABDESIGN_OPT`) — a kit mode without the named levers, for A/B
  attribution: `ACTIVE mode=fast-no-triatt base=fast ablated=triatt restored=pallas … class=ablation`, each dropped lever
  `state=off reason=ablated`; a base other than `exact|fast`, a lever outside the base's set, or a word dropping every lever is an unknown
  mode. No variable switches a single lever; `MODEL_OPT_LEVERS_OFF` is the shared core's switch for its kernel-provider rows (a row switched
  off leaves its calls on stock's operation by name, printed on the lever's line).
