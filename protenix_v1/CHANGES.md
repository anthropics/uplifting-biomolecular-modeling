# Protenix v1 kit — what changes vs stock

Stock = Protenix 1.1.0 at the pin (STOCK.md). Each lever is a module the kit installs over one stock function or class at start-up
under a kit mode (`opt/forward/v05_addon/ptxfpf/levers_ptx1.py` and the adapters beside it; the memory levers in
`opt/protenix_v1_opt/big.py`); `off` loads none. A mode is all of its levers (`opt/protenix_v1_opt/modes.py`); every lever engages or
steps aside by name on its `LEVER name=…` line. Kernels are the shared core's (`common/opt_core`), bound per mode by its tier word.

## exact — outputs identical to stock

This mode binds FlashPairformer kernels via the shared core (opt_core). Outputs identical to `off` under `--det 1`; every lever bitwise class.
- `exact` — TriangleMultiplication (c 128) through the core's TriMul provider (`opt_core.kernels.trimul`) `exact` word. Kernel: FlashPairformer
  triangle multiplication via the shared core — CUDA-native (sm_90 / sm_80), exact variant, at the bf16 trunk keys | cuEquivariance 0.11.1 at
  the fp32 confidence-head keys. Steps aside: a key the provider refuses keeps upstream's statement (`stock:trimul:<refusal>`, `reason=aside`).
- `gblock` — TriangleAttention as the core's fused block (`opt_core.attn.pair_fused`: stock LayerNorm → one-kernel q|k|v|g|bias prologue → the
  stock attention call (cuEquivariance 0.11.1, as upstream configures it) → one-kernel gate·o @ W_o epilogue). Kernel: FlashPairformer fused
  projection kernels via the shared core; the attention itself is the stock call unless `triexact` (below) serves it. Steps aside: `reason=above_ceiling` above
  2,048 tokens (upstream row-chunks there).
- `triexact` — rides `gblock`: its attention call by the core's triangle-attention provider `exact` word (`opt_core.kernels.triattn`): the
  bit-identical row `triattn_exact` where the core's table admits the running stack and the call's shape lies inside the row's proven cells (H100 on
  this kit's stack: `row=triattn_exact member=<served>/<calls>`), the stock call by name for any other shape, stack or card (`row=cueq`). Numerics: bitwise.
- `xtr` — pair Transition (128 → 512 → 128) on the stock LayerNorm output. Kernel: FlashPairformer transition via the shared core
  (`opt_core.kernels.transition`), exact variant. Steps aside: a stock answer keeps the module (`reason=aside aside=stock:xtr:row=<row>`).
- `sg` + `sampler_prep`, `hoist` — the diffusion denoiser step captured and replayed as a CUDA graph (`lib/kit112_src/infopt_graphs`), its
  per-step host work (pinned rotation upload, shape-keyed graph cache, one warm-up pass) off the critical path; the 24 DiT blocks' pair-bias
  projections computed once per sample outside the 200-step loop (`lib/kit112_src/dit_hoist.py`). Numerics: bitwise (scheduling only).
  Steps aside (the three together): `reason=above_cap` when the run's largest input exceeds the mode's sampler-graph token cap (exact
  1,536 / fast 1,999 on cards ≥ 64 GiB, 768 / 999 below; `PTX_SAMPLER_GRAPH_MAXTOK` overrides, `0` = none).
- `keep_pool`, `summary_hostidx`, `lazy_init` — stock's in-forward `torch.cuda.empty_cache()` calls become counted no-ops
  (`lib/ptx1_keep_pool.py`); summary-confidence bookkeeping from one host copy of the chain / atom ids instead of per-chain device→host
  reads, the same GPU ops in the same order (`lib/ptx1_summary_host.py`); random parameter initialisation skipped, the strict checkpoint
  load sets every tensor (`lib/ptx1_lazy_init.py`; `PTX_LAZY_INIT=recheck` builds both ways and compares). Numerics: bitwise.
- `dit_attn_exact`, `atom_attn_exact` — the DiT fp32 pair-bias attention (16 × 48) on the core's bit-exact CUDA kernel (`opt_core.kernels.apb`
  row `dit_exact`, sm_90) and the atom transformers' 32 × 128 local-window attention on one Triton launch reproducing the stock arithmetic
  (row `atom_exact`), each behind a load-time bitwise check. Steps aside: `dit_attn_exact` `reason=card_off` on a card without the kernel (sm_80);
  `atom_attn_exact` `reason=aside` wherever the core's exact word for the atom cell names the stock kernel (`sdpa_gather`: this stack's sm_90 and sm_80 tables).
- `template_dedupe`, `tmpl_triatt`, `tmpl_xtr`, `tmpl_trimul_exact` (`ptxfpf/ptx1_templ.py`) — the template embedder's pair stack: each
  distinct template slot evaluated once per recycle; its triangle attention through the same fused block around the stock call, its pair
  transition (c 64) and triangle multiplication (64 → 128) through the providers' `exact` words (the multiplication: the stock torch path by
  name today).
- sm_80 (`--config a100`): `gblock`, `xtr`, `tmpl_xtr` serve only the pair-row counts of the core's exact band (`attn/pair_fused_cells.json`
  "exact_rows"; `range=` / `gated_by=stock:below_min_rows|above_max_rows` on the line): cuBLAS's summation order there changes with the row count.

## fast — within stock's seed-to-seed variation

`sg`, `hoist`, `keep_pool`, `summary_hostidx`, `lazy_init`, `sampler_prep`, `template_dedupe`, `tmpl_xtr` as under exact.
- `fast` — TriangleMultiplication through the TriMul provider's `fast` word. Kernel: FlashPairformer triangle multiplication via the shared
  core — Triton, fast variant. Numerics: bf16 MMA / fp32 accumulate. Steps aside: as `exact`.
- `gflash` + `tricuda` — the fused triangle-attention block with the LayerNorm inside the prologue (`opt_core.kernels.lnl_fused`) around the
  provider's `fast` word. Kernel: FlashPairformer triangle attention via the shared core — CUDA-native (sm_90a) | Triton, by row length;
  fast variant. Numerics: exp2 online softmax, bf16 products, fp32 accumulation. Steps aside: as `gblock`.
- `ttr` — pair (128 × 512) and MSA (64 × 256) Transitions as one fused LayerNorm + SwiGLU kernel. Kernel: FlashPairformer transition via
  the shared core — Triton (provider `fast` word).
- `ditattn` + `ditattnfp16`, `atomattn` (`ptxfpf/apb_ptx1.py`; `opt_core.kernels.apb` rows `fpf_apb`, `fpf_atom`) — DiT token attention with
  pair bias on one fused flash launch per call (fp16 operands, fp32 softmax / accumulation, gate fused); atom encoder / decoder
  local-window attention on one launch (TF32 operands). Any card ≥ sm_80 runs the sm_90 cell, named on the line.
- `cond_dedupe`, `dit_fused` + `dit_lowp`, `atom_fused` (`lib/protenix_fpf_ditfast`, `ptxfpf/ditfast_ptx1.py`) — diffusion conditioning once
  per step instead of once per sample; the 24-block DiT as one fused forward (AdaLN row kernel, merged GEMMs, `ditattn`'s kernel, fused
  gate + residual; `dit_lowp`: fp16 operands, fp32 residual and statistics); both atom transformers as fused stacks around `atomattn`. Steps
  aside: `reason=card_off` on sm_80; `cond_dedupe` `aside=n_sample:1` at `--sample 1`; `dit_fused` / `atom_fused` need `ditattn` / `atomattn` (named).
- `pfattn`, `opm_fused`, `pwa_fused` (`ptxfpf/trunk2_ptx1.py`, `lib/protenix_fpf_msa`, `opt_core.ops.msa_fused`) — Pairformer attention with pair
  bias on the core's `opt_core.kernels.apb` kernels (one Triton pass for LayerNorm + bias projection, flash attention with in-kernel bias and gate);
  MSA outer-product mean and pair-weighted averaging on fused Triton kernels around stock's own einsum. Numerics: bf16, as stock's autocast.
  Steps aside: the fp32 confidence pairformer keeps stock (`stock:no_autocast`).
- `tmpl_triatt` + `tmpl_pairfused`, `tmpl_trimul` — the template pair stack's triangle attention through the fused block with the LayerNorm
  fused; its triangle multiplication through the TriMul `fast` word.

## big — lowest peak GPU memory

`fast` without `sg`, `hoist`, `keep_pool`, `sampler_prep`, `dit_fused`, `dit_lowp` (`big.BASE_LEVERS_OFF`), kernels by the providers'
`big` word, plus the memory levers over `opt_core.mem`, sized once per process on the run's largest input (`--input` estimate or
`PROTENIX_V1_BIG_SIZE_N_TOKEN`; unsized: the size-gated ones read `reason=below_gate`). Numerics: as `fast` unless noted. Each memory
lever refuses by name if the stock file it patches differs from the pin (`stock/PINS.json` "model_files").
- `expandable_segments`, `cache_release`, `chunk_pair` — the expandable-segments allocator (read back), one allocator release per input,
  upstream's own chunked pair path (`chunk_size` 128, dynamic chunking off).
- `drop_bond_mask`, `relp_lean`, `msa_zfree`, `diffcache_free`, `recycle_carry` — the unused [N, N] `bond_mask` dropped before the device
  copy; the relative-position plane written block by block; each MSA block's pair input released after its last read; the diffusion
  caches released at the confidence head; `z_init` parked on pinned host between recycles. Numerics: bitwise (placement only).
- `diffusion_cond_chunk`, `conf_head_chunk` — diffusion conditioning and the confidence head's PAE / PDE tail in row blocks
  (`opt_core.mem.chunk`). Numerics: fp32 reduction order follows the block extent.
- `trimul_torch` — upstream's row-chunked torch TriangleMultiplication above `PROTENIX_V1_BIG_TRIMUL_TORCH_TOKENS` (2,048) tokens only.

### `--n_gpu P`

P ∈ {2, 4, 8} (`ngpu.py`, `rowpair.py`, `tp.py` over `opt_core.mem.rowpair`): one process per GPU under one `PYTHONHASHSEED` (`RANKENV hashseed=…`).
Rank 0 featurises each item and broadcasts the features (`data_form=rank0_bcast`; differing per-rank digests stop the run, `feats_ranks_differ`).
The pair representation is row-sharded: every pair block (trunk, MSA module, template embedder, confidence head) runs the core's row statements with
the row TriMul and the triangle-attention `big` word per query-row block; the diffusion sampler caches its pair-bias rows in fp16 when they fit
and runs query-row attention on the core's rows face (`ROWPAIR_DIT_ATTN=rows|sdpa|stock`); the trunk shard is parked on pinned host through the
roll-out and confidence passes; a templated input's template pair features are formed per rank as that rank's rows (`TEMPLATE … form=row_born`).
When the stock's MC dropout applies to an item, every rank draws the stock keep-mask over the whole pair plane from the global generator and applies
its own rows (`tp_mc_dropout=stock_mask`), so the process's later draws stay the stock's; above `PROTENIX_V1_TP_MC_DROPOUT_FULL_MAX` tokens (default
2,048; `0` = at every size) the rows draw from rank-forked generators instead (`tp_mc_dropout=rank_streams`, named by a NOTE: later draws become an
independent realisation of the seed). The launch exports the core's `ROWPAIR_*` settings (`ngpu.TP_EXPORTS`) unless the caller set them. Levers
whose sites the row statements own read `replaced_by_rowpair` (`summary_hostidx`, `pfattn`, `opm_fused`, `pwa_fused`, the sampler words,
`msa_zfree`, `diffcache_free`, `relp_lean`, `recycle_carry`, the chunk levers). Refused by name: P ∉ {1, 2, 4, 8}, fewer GPUs visible than P,
`--n_gpu` > 1 under another mode.

## Every mode

- No stock exceptions (STOCK.md); no opt-in upstream fixes. Upstream's `--triatt_kernel` / `--trimul_kernel` / `--dtype` set away from
  their defaults route the affected levers off by name (`reason=stock_knob:<flag>:<value>`, exit 0).

## Switches

- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — a mode without the named levers (`ablated=<names>` on the ACTIVE line). `--det 1` — the
  deterministic recipe on any mode. `--allow-partial` — accept a `PARTIAL` activation. Variables that move lever gates
  (`PTX_SAMPLER_GRAPH_MAXTOK`, `PROTENIX_V1_BIG_*`, `ROWPAIR_DIT_ATTN`): names here, values in STOCK.md.
