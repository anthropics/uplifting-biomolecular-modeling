# OpenFold3 (OpenBind-0) kit — what changes vs stock

Stock = OpenFold3 0.5.0 at the pin (STOCK.md). Each lever is a module the kit installs over one stock function or class at start-up under a kit mode (an import hook;
no upstream file is edited); `off` loads none. A mode is all of its levers, listed below from the one mode table `opt/openfold3_ob0_opt/modes.py`; the names are the
ones on the run's `LEVER name=… state=… served=… fallback:<reason>=…` lines — a call a lever cannot serve (shape, dtype, layout) goes to the stock statement for that
call and is counted there, never refused. Kernels come from the shared core (`common/opt_core`, `opt_core.*` below).

## exact — outputs identical to stock
Bitwise to `off` under `--det 1`: placement, scheduling and caching only, or the same arithmetic proven equal in-process before a kernel serves.
- `fast_init` — skips the CPU-side weight initialisation that the checkpoint load overwrites.
- `paircache` — the diffusion transformer's step-invariant conditioned pair representation and 24 per-block pair biases computed once per roll-out instead of per
  step, held head-major so the attention levers read them in place (a 4 GB cap; blocks over it run the stock computation, counted).
- `templ_distinct` — an untemplated chain's four identical dummy template slots pass the template pair stack once and are re-expanded before the stock reduction.
- `cuda_graphs`, `graphs_strict` — one denoising step of the diffusion sampler captured per input shape and replayed for the other steps, samples and later items
  of equal shapes; a capture failure fails the run rather than running eager. `exact` captures up to 512 polymer tokens, eager above (`--graphs-max-tokens`).
- `atom_hoist`, `castcache` — the atom-attention encoder/decoder's roll-out-invariant features computed once per roll-out into address-stable buffers
  (`opt_core.of3_sampler`); the `Linear` primitive's per-call fp32→bf16 weight casts memoised per module, keyed on storage + version (`opt_core.of3_trunk.castcache`).
- `exactln` — `LayerNorm` bound to `opt_core.kernels.ln` on word `exact`: an NVRTC replica of ATen's kernel serves a call class only after `torch.equal` in-process.
- `apb_hoist` — `AttentionPairBias`'s key-mask bias memoised per stack call instead of rebuilt in every block (`opt_core.of3_trunk.apb_hoist`).
- `triatt_exact` — the pair stacks' triangle-attention calls go to `opt_core.kernels.triattn` with word `exact`: on an H100 stack the core's table vouches for, its exact-class row serves them
  with a fused kernel whose outputs are bit-identical to the engine's cuEquivariance call (the stock op); everywhere else — other cards and stacks, shapes outside the row's cells — that stock call
  answers (on sm_80 above 1,536 tokens as per-head calls of the same library kernel, admitted per call class only after an in-process `torch.equal` proof against the full call).
- `trimul_exact` — the pair stacks' triangle multiplicative updates served by `opt_core.kernels.trimul` on word `exact` (row `tmk3_exact` at c_z 128 up to 400
  tokens), admitted per call class after the same proof; the cuEquivariance call by name elsewhere.
- `trimul_form` — the module-statement classes of that binding (form `of3_module`); ships off (`OPENFOLD3_OB0_OPT_TRIMUL_EXACT_FORM=1` arms it; `line=triton_flag` aside).
- `transition_exact` — the SwiGLU transitions bound to `opt_core.kernels.transition` (tier `exact`, form `swiglu`), fed the module's own LayerNorm output; a
  kernel row serves where the provider's table names one bitwise for the cell on the running stack, the module's statement by name otherwise.
- `post_release`, `postfwd_mem` — at ≥ 2400 tokens or < 16 GB free, graph pools, cached buffers and allocator segments are released after the sampler and the
  item; the post-forward confidence scoring runs per (sample, row block) under a 256 MiB budget into the engine's tensors — the same bytes at the forward's peak.
- `loader_workers`, `hostfeat`, `ckpt_mmap`, `fastjson`, `writer_overlap` — host side, same bytes: max(1, min(items, configured)) DataLoader workers; `parse_a3m`'s
  deletion-count loop with `bytes.translate` and uint8 views; the checkpoint read with `torch.load(mmap=True)`; the confidence-JSON encoder renders the same
  characters at C level; item k's files written by one forked writer worker while item k+1 runs (`OPENFOLD3_OB0_OPT_WRITER_OVERLAP_ROUTE=process|thread|sync`).

## fast — within stock's seed-to-seed variation
`exact`'s levers except `exactln`, `triatt_exact`, `trimul_exact`, `trimul_form`, `transition_exact`, plus the levers below; numerics: bf16 re-association and
fused-kernel rounding. The pair-stack levers bind FlashPairformer kernels via opt_core.
- `trimul_v4`, `trimul_provider` — `PairBlock`'s outgoing + incoming triangle multiplication as one fused Triton cell per direction (`opt_core.kernels.fpf_trimul_v4`,
  pair widths 128 / 256), the row per call class chosen by `opt_core.kernels.trimul` on word `fast` (also serving the c 64 template stack); a refused row is counted.
- `triatt_block`, `triatt_provider` — starting + ending triangle attention as prologue kernel → attention core → epilogue kernel adding into z in place
  (`opt_core.attn.pair_fused`), the core per call class from `opt_core.kernels.triattn` on word `fast` (sm_90a and sm_80 members; a row without a prebuilt for the
  running torch ABI is refused by name and the named fallback serves); an fp32 pair stream runs the line's own attention by name.
- `pair_transition` — the pair SwiGLU transition fused so the 4·c_z hidden never reaches HBM (`opt_core.attn.pair_fused`; the `v2_fold` kernel on cc 9.0, else `v1`).
- `rollout_bf16` — `DiffusionModule` conditioning + diffusion transformer under bf16 autocast; the atom-attention encoder/decoder, the position update and the
  sampler's arithmetic stay fp32 (`OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS` gates smaller inputs back to the fp32 roll-out, counted `gated=`).
- `dit_attn`, `dit_glue` — the diffusion transformer's pair-bias attention on a flash kernel and each token block as one schedule of 11 fused kernels
  (`opt_core.of3_sampler.dit_glue`, `opt_core.kernels.apb` on tier word `fast`), bf16 operands only (`use_high_precision_attention` overridden there, counted).
- `apb_trunk`, `templ_embed` — the pairformer single track's attention with pair bias as one fused LayerNorm+projection kernel for the bias (`opt_core.kernels.ln_proj`)
  and one launch of `opt_core.attn.apb_core`; the template embedder's eight feature Linears + `linear_z(LayerNorm(z))` as one kernel and mean → relu → `linear_t`
  as one kernel (`opt_core.of3_trunk.templ_embed`; upstream's offloaded template path keeps upstream's statements). Refusals run the stock forward, counted.
- `token_agg`, `atom_window` — the atom→token mean as one deterministic segment-reduce kernel (`opt_core.of3_sampler.token_agg`; other layouts → stock, named);
  the sampler's sequence-local atom attention on two fused kernels (`opt_core.kernels.atom_window`: TF32 dots, fp32 in/out; launch tile per card, `ln_tile=`).
- `ln_provider` — the LayerNorm binding on tier word `fast`: the provider's fastest row inside each cell's tolerance, witnessed per signature or retired by name.
- `trunk_graph`, `tuner_guard`, `alloc_expandable` (exact class) — `PairFormerStack.forward` in a CUDA graph per (instance, signature) from its second call (above
  1024 tokens or after a failed capture the stack's own forward, named); a raising chunk-tuner comparison answers "changed"; `expandable_segments:True`.
- `z_dtype`, `conf_dtype` — the trunk hands `s_input, s, z` on in bf16; the confidence phase runs under bf16 autocast (`--z-dtype fp32` / `--conf-dtype fp32` restore fp32).

## big — lowest peak GPU memory
Numerics as `fast`; runner YAML `big_bf16_c16_predict.yml` (chunk 16, tuner off, upstream's third-party pair kernels off, `bf16-mixed`). Below the item gate (`OF3O_MIN_TOKENS` polymer tokens, per card in `modes.OF3O_GATE_BY_CARD`) a query runs the `fast` line as composed and the units below step aside by
name (`reason=of3o_gate`). At or above it: `fast`'s levers except `cuda_graphs`, `graphs_strict`, `paircache`, `atom_hoist`, `dit_glue`, `castcache`, `trunk_graph`, plus:
- `trimul_hostsnap`, `triatt_lean`, `trans_inplace` — triangle multiplication streams its update in row blocks from a pinned host snapshot of z; the ending
  triangle attention runs on one lean transposed buffer; the pair transition is applied in place.
- `input_rows`, `recycle_rows`, `cond_once`, `templ_host` — relative-position features, the recycling add and the template embedding built per row block with
  template pair features on the host; the diffusion conditioning's pair path computed once per item.
- `bigln_guard`, `host_pool` — LayerNorm over ≥ 2³¹ elements in row blocks; one pinned host buffer pool (`OF3O_PIN_BUDGET_GB`; a buffer over budget is pageable, counted).
- `conf_chunked`, `confhead` — from `modes.CONF_MIN_TOKENS` (2048) tokens the PAE/PDE heads are evaluated per scorer row block and the scores reduced over row
  blocks (the [S,N,N,64] logits never exist); below it upstream's confidence path runs, named `conf=gated:lt_min`.
- From `OPENFOLD3_OB0_OPT_REACH_GATE` polymer tokens (`modes.REACH_GATE_BY_CARD`) the full-N² pair cells `trimul_v4`, `trimul_provider`, `triatt_block`,
  `triatt_provider`, `pair_transition` step aside by name (`reason=reach_gate`) and the row-block statements serve the trunk.

### `--n_gpu P`
P ∈ {2, 4, 8}; another P, fewer visible GPUs than P, or `--n_gpu` > 1 under another mode is refused by name. Numerics as `fast`. A pocket-constrained query, or
one too small for a row block per rank, is predicted by the stock route on one GPU and named (`fallbacks=tp:small_n_unsharded=1` / `tp:pocket_unsharded=1`).
- `tp_shard_s` — each of the P rank processes holds a row shard of z from the input embedder on; the pair stack, MSA module, template embedder, diffusion
  conditioning and confidence heads run on each rank's rows with ring-streamed peers (`opt_core.mem.rowpair`, bound by `opt/openfold3_ob0_opt/tp_rowpair/`);
  rank 0 featurises and broadcasts the batch, and the ranks' feature digests must agree (`feats_ranks_differ` refuses otherwise).
- `tp_triatt`, `tp_trimul` — triangle attention and multiplication per row block on the flash / fused kernels (bf16 operands, fp32 accumulation); the
  multiplication kernel from 2048 tokens; the c 64 template stack and smaller pairs use OpenFold3's torch statements, counted on the `TRIATT` / `TRIMUL` lines.
- `sample_loop` — the diffusion roll-out and confidence heads one sample per pass, random draws in stock order; device memory independent of `--num-diffusion-samples`.
- Structure first: each sample's model file is written when the sampler returns and overwritten by upstream's writer after the confidence pass
  (`OF3TP_STRUCTURE_FIRST=0` restores stock's order). Also on this line: `fast_init`, `conf_dtype`, `z_dtype`, `loader_workers`, `fastjson`, `hostfeat`, `ckpt_mmap`.

## Every mode
- No stock exception and no opt-in upstream fix ships (STOCK.md §Stock exceptions): upstream's code runs unmodified; `--det 1` applies one deterministic recipe, `off` included.

## Switches
- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — a mode without the named levers (`levers_off=<names>` on the ACTIVE line); `--no-compile` — a named no-op (`compile=none`); `--z-dtype`,
  `--conf-dtype`, `--graphs-max-tokens`, `OF3O_MIN_TOKENS`, `OPENFOLD3_OB0_OPT_REACH_GATE`, `OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS`, `OF3TP_*`: values and defaults in STOCK.md.
