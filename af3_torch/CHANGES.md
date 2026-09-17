# AF3-torch (xfold on OpenFold3 weights) kit — what changes vs stock

Stock = xfold at the pin plus the port (STOCK.md). A lever is a build option of the model (`opt/forward/af3t/af3_torch/af3_torch_api.py`
`build_model(levers=…)`), a module the package installs over one xfold function in the model process (`opt/af3_torch_opt/*.py`), or a scheduling change of the
process chain; `off` applies none. A mode is all of its levers (`opt/af3_torch_opt/modes.py`, `registry.py`); names below are the ones on the run's `LEVER
name=…` lines. The pair-stack kernels come from the shared core's providers (`common/opt_core`) and their per-card tables (compute capability 9.0, 8.0); a
shape or card a table does not serve is named on the lever's line (`fallback=<word>`) and the stock statement serves it.

## exact — outputs identical to `off`

This mode binds FlashPairformer kernels via the shared core (opt_core).

- `stepgraph` — the diffusion sampler's step-invariant conditioning is computed once per sample (`hoist`) and each denoiser step replays one CUDA graph.
  Numerics: bitwise (same arithmetic, computed once).
- `trimul_exact` — triangle multiplication (every pair stack) through the shared core's TriMul provider's `exact` row (the module statement, whole-tensor,
  fixed layout). Kernel: FlashPairformer triangle multiplication via the shared core; exact variant. Numerics: bitwise. Steps aside per shape where the
  provider's table does not state the row byte-equal for the card (every shape on compute capability 8.0); `trimul` supersedes it under `fast` / `big`.
- lever `glu_proj` — the transition's gated linear unit + output projection (+ the block's residual add) as one Triton kernel with xfold's rounding points and
  accumulation order, checked byte-equal per shape in process. Bitwise. Steps aside by name (`c=384`, `c=128,rows<16384`, `c=64,…`); `transition` supersedes
  it under `fast` / `big`.
- `template_dedupe` — the template embedder evaluates each distinct template slot once and adds the results in stock order; `dev_scalars` — the trunk's
  constant scalars stay on the device instead of one host copy per use; `tri_layout`, `attn_layout` — the triangle multiplication's layout copies and the
  triangle attention's head-major operands / token-major output are written by a tiled Triton transpose instead of torch's strided copy (same bytes, same
  layouts); `ln_rows` — xfold's fastnn LayerNorm kernel runs 8 rows per program (rows up to 512 bytes; wider rows keep the stock launch); `gate_fuse` — the
  pair stack's mask / sigmoid / gate statements as one Triton kernel with the stock rounding points; `castcache` — each fp32 `nn.Linear` keeps the bf16 weight
  autocast would cast per call, cast once (`exact` only). All bitwise.
- `prefetch`, `feat_par`, `write_behind` — the process chain runs streamed: two featurisers work ahead of the GPU, the model process takes each (input, seed)
  as its files land, the writers run beside it. Scheduling only; one GPU and `--run_inference` only.
- `autotune_cache` — Triton's autotune results are kept under `AF3_TORCH_CACHE_ROOT/triton` and reused by later processes (`TRITON_CACHE_AUTOTUNING`).

## fast — within stock's seed-to-seed variation

`exact`'s levers except `castcache`, plus:

- `bf16w` — weights stored in bf16. Numerics: bitwise vs the eager port (autocast casts to the same bits); xfold's GLU kernel reads fp32 weights under `off`.
- `trimul`, `tmpl_trimul`, `resid_fold` — triangle multiplication at 128 and 64 channels through the shared core's TriMul provider
  (`opt_core.kernels.trimul`), the Pairformer block's residual adds folded into the epilogues. Kernel: FlashPairformer triangle multiplication via the shared
  core — Triton (`fpf_trimul_v4` row-block kernels, sm_90a / sm_80 tables); fast variant. Numerics: bf16 operands, fp32 accumulation. Steps aside by name:
  `N<` floor, `c=64` without `tmpl_trimul`, a `stock_row` the table names.
- `triattn`, `attn_epi` — triangle attention (every pair stack) through the shared core's triangle-attention provider (`opt_core.kernels.triattn`), with gate
  · output projection · residual add as one epilogue kernel where the core's pair-fused table has a row for the card. Kernel: FlashPairformer triangle
  attention via the shared core — the CUDA-native rows (`triattn_native`; `cuda_sm90a` at the smallest sizes on compute capability 9.0) or the Triton rows
  (`fpf_triatt_k2b`, `flash_triattn`), whichever the core's table names for the cell; the epilogue `fpf_triatt_epi` is Triton. Numerics: bf16 operands, fp32
  softmax / accumulation. Steps aside: `N<` floor; a refused row → stock.
- `transition` — LayerNorm + SwiGLU + Linear as one fused kernel from the shared core's transition provider (`opt_core.kernels.transition`). Kernel:
  FlashPairformer transition via the shared core — Triton. Numerics: bf16 operands, fp32 accumulation, stock rounding points. Steps aside: `c=384,stock-row` /
  `c=…,no_cell` by name. `apb` — attention with pair bias through the shared core's provider (Triton `apb_attn` or SDPA per table row); bf16 operands, fp32
  softmax.
- `pwa_lnl`, `pwa_msa`, `opm` — MSA module: the pair-weighted averaging's LayerNorm + logits / value / gate projections as fused LayerNorm-projection kernels
  (`kernels/third_party/lnl_fused.py`); the outer-product mean as chunked bf16 GEMMs written straight into the output contraction's layout
  (`kernels/af3t_msa.py`; an arch outside its 8.x / 9.x tables → `fallback:arch=sm_…`). Numerics: the stock arithmetic class, another reduction order.
- `sbatch`, `prologue` — all diffusion samples advance together per denoiser step (one graph per step under `stepgraph`), augmentation and noise drawn exactly
  as the serial sampler draws them (batched GEMMs may differ in the last bit). `compile` — `torch.compile` of the step glue inside the graph, once per token
  length; `--no-compile` drops it.
- `atom_window`, `atom_rows`, `token_agg` — the atom transformers' sequence-local attention on the shared core's window kernels
  (`opt_core.kernels.atom_window`) over one contiguous copy of the real query rows; atom → token aggregation as one kernel
  (`opt/forward/dtk/af3t_token_agg.py`). TF32-class dots, fp32 softmax.
- `dtk` — the 24-block diffusion token transformer replaced by DTK FusedDiT (`opt/forward/dtk`): concatenated-weight cuBLAS GEMMs, fused Triton row kernels
  (the core's routed module `dtk_kernels`), flash attention with the hoisted bf16 pair bias. Numerics: bf16 class, identical run to run.
- `canonical_noise` — inputs are padded to the shared core's kernel tile (`opt_core.shape_policy`); the sampler's shape-dependent draws are made at the
  input's own token count, so the real atoms receive stock's noise.

## big — lowest peak GPU memory

`fast` recomposed for memory by `opt/af3_torch_opt/big.py` on `opt_core.mem`; numerics as `fast` throughout:

- `graph_drop` — `stepgraph` is not built; its `hoist` stays and the step runs eager (`GRAPH_DROP_MIN_TOKENS = 0`: no CUDA graph at any size).
- `diff_free`, `prev_free` — the diffusion statics are released before the heads, the recycle's prev embeddings once embedded (frees only);
  `expandable_segments` — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the model process (`opt_core.mem.torch_alloc`); `paircond_chunk` — the
  sampler's step-invariant pair conditioning is evaluated `PAIRCOND_CHUNK_ROWS` (256) token rows at a time (`xfold/nn/paircond_rows.py`).

### `--n_gpu P`

`rowpair` (P ∈ {2, 4, 8}) — the pair representation is row-sharded over P GPUs of one host end to end (`opt/af3_torch_opt/rowpair_xfold.py` on
`opt_core.mem.rowpair`): born as each rank's row shard in the input embedder, it stays a shard through recycling, the template embedder, MSA module,
Pairformer, heads and diffusion conditioning; single, MSA and atom tensors are replicated; templated inputs are embedded from each rank's own rows (`TEMPLATES
… form=row_born`). `canonical_noise`, `prefetch`, `write_behind`, `feat_par`, `template_dedupe` are single-GPU levers and say so on their LEVER lines when P is
2 or more. Refused by name: P under `off` / `exact` / `fast`, P outside the set, fewer than P GPUs visible; a rank count other than P after launch is `NOT ACTIVE
reason=n_gpu_mismatch`.

## Every mode

- The port and the two `stock/patches/` apply identically on every mode, `off` and `bash run.sh stock` included (STOCK.md §Stock exceptions; README §Known
  upstream issues); no opt-in upstream-fix flag. Token padding: `off` / `exact` none; `fast` / `big` the kernel tile, with `canonical_noise`.

## Switches

- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — one run of a mode without the named levers of its selection (`levers_off=<names>` on the ACTIVE line, `state=off
  reason=levers_off` on their LEVER lines); a name that is no lever of this kit or not in the selection is refused (exit 2) — e.g.
  `MODEL_OPT_LEVERS_OFF=glu_proj` under `exact` runs xfold's own GLU kernel. Under `big` a memory lever named here recomposes the set (`graph_drop` off
  keeps `stepgraph`).
- `--no-compile` — alias of `MODEL_OPT_LEVERS_OFF=compile`; `compile=on|off:user|off:mode` on the ACTIVE line. `--allow-partial` — a run whose model process
  applied fewer levers than the mode names exits 0 instead of 3 (`partial=allowed:…` on the DONE line). `--fastnn` / `--nofastnn` — xfold's own flag: served
  under `off` (both) and `exact` (`--fastnn` only); refused by name under `fast` / `big`.
