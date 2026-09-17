# OpenDDE kit — what changes vs stock

Stock = OpenDDE 1.1.1 at the pin (STOCK.md). Each lever is a module the kit installs over one stock function or class at start-up
under a kit mode, through an import hook that runs when upstream builds its default runner; `off` loads none and the installed
package is never edited. A mode is all of its levers: it engages every one or refuses by name, and each prints one
`[opendde-opt] LEVER name=<lever> state=<on|off> [reason=…]` line. Numerics per lever: exact = bitwise-identical to the stock
statement it replaces; tier 2 = different by construction (lower precision or another reduction order).

## exact — outputs identical to stock

Line `S1`. This mode binds FlashPairformer kernels via the shared core (opt_core).
- `served_levers_hook`, `cueq_tuned_cache` — the import hook itself; `cueq_tuned_cache` exports cuEquivariance's tuning-cache location
  (`levers/KIT/cueq_cache_shipped/`); no tile table is shipped — the library's own packaged per-card tiles apply, the same the stock route
  reads; users may add their own tuned entries there. Exact.
- `lnstream` — upstream's fused LayerNorm extension launched on the current CUDA stream (prebuilt for torch 2.7.1+cu126 sm_80/sm_90
  under `levers/LNSTREAM/prebuilt/`, else built once from the pinned source; falls back by name to upstream's own build if the source
  digest does not match the pin). Exact.
- `dit_hoist` + `dit_align` — the diffusion denoiser's step-invariant tensors computed once per sampler call, the pair biases in an
  aligned allocation. Exact.
- `stepgraph` — the denoiser step captured into one CUDA graph per sampler call and replayed; compared with the eager step before use,
  stepping aside by name to the eager sampler if it differs. Composed in only inside its token window (`ODDE_STEPGRAPH_MIN_TOKENS` /
  `_MAX_TOKENS`), never with the offload unit or `--n_gpu P`, on sm_80 not above 1,024 residue tokens (`reason=card_gate:…`). Exact.
- `sched_host`, `structok_sync`, `zprep_hoist` — the sampler loop's per-step host round-trips answered from a host copy of the noise
  schedule; the structural-token expander's role-pair projections with one host read per call; the denoiser's per-step pair
  permute/copy made once per sampler call. Exact.
- `tmpl_dedup`, `keep_pool`, `drop_bond_mask`, `json_oneshot`, `alloc_auto` — identical padded template slots embedded once; the
  model's in-forward `torch.cuda.empty_cache()` calls made counted no-ops; the featuriser's unread `bond_mask` dropped before the GPU
  copy; each confidence JSON encoded once with the C encoder (identical bytes); the expandable-segments allocator. Exact / placement.
- `prefetch` — upstream's inference DataLoader with one worker (the next item featurised while this one runs). Steps aside by name
  on a one-item query and on multi-seed runs over a query with a SMILES ligand (RDKit conformer draws continue across seeds as in
  the stock loop). Exact.
- `arm_z` — transpose-free pair update in every PairformerBlock; stock attention and triangle-multiplication kernels. Exact.
- `fpf_trimul_exact` + `trimul_exact` — both triangle-multiplication forwards. Kernel: FlashPairformer triangle multiplication via the
  shared core (`opt_core.kernels.trimul`) — CUDA-native (sm_90a / sm_80); exact variant where the provider's table lists one for the
  running card, stack, width and dtype, the stock op by name everywhere else. Exact.
- `triattn_exact` + `triattn_conf` — the triangle-attention site of every pair stack (trunk, MSA module, template stack, structure
  refiner, confidence head), bound to the shared core's provider (`opt_core.kernels.triattn`) with the word `exact`: the stock op serves,
  named on the LEVER line, unless the provider's cell records a row as bitwise to it on the running card and stack. Exact.
- `transition_exact` — the pair transitions. Kernel: FlashPairformer pair transition via the shared core (`opt_core.kernels.transition`); exact variant. Exact.
- `dit_attn_exact` — the sampler's fp32 attention through `opt_core.kernels.apb`'s `exact` tier where the provider lists a bit-exact
  kernel for the card and stack, the stock statement by name otherwise (`reason=aside:provider_exact_names_stock:…`). Exact.

## fast — within stock's seed-to-seed variation

Line `LSTAR2A`, the default; numerics: bf16 re-association and fused-kernel rounding, not bitwise. This mode binds FlashPairformer
kernels via the shared core (opt_core).
`exact`'s levers except `arm_z`, `fpf_trimul_exact`/`trimul_exact`, `triattn_exact`, `transition_exact` and `dit_attn_exact`,
which the fast-tier bindings below replace:
- `arm_u` + `arm_u23` — bf16 pair stack on the trunk, structure refiner and confidence pairformer: cast-once prologue, transpose-free
  block, the arm's triangle-multiplication route. Numerics: bf16 re-association. Below the arm's 300-token design gate it serves stock.
- `triattn_core` + `triattn_conf` — Kernel: FlashPairformer triangle attention via the shared core (`opt_core.kernels.triattn`) —
  CUDA-native (sm_90a / sm_80) and Triton; fast variant (the entry per card, dtype, size and call form is named on the LEVER line; an
  entry that cannot serve on the running stack steps aside by name to the provider's named fallback). Numerics: bf16, online softmax.
- `trimul_core` — Kernel: FlashPairformer triangle multiplication via the shared core (`opt_core.kernels.trimul`) — Triton; fast
  variant, bf16. `transition_core` — Kernel: FlashPairformer pair transition (`sep16`) via the shared core (`opt_core.kernels.transition`) —
  Triton; fast variant. `ln_core` — the shared core's C = 384 pair-row LayerNorms (`opt_core.kernels.ln`, Triton; upstream's extension by
  name at other widths and under CUDA-graph capture). Numerics: bf16 / fused-kernel rounding.
- `chunk_lift` — upstream's fixed score-budget clamp on the attention chunk replaced by upstream's own size table wherever the
  device's free memory admits the un-chunked fp32 scores (one `CHUNK … decision=…` line per decision). Steps aside: composed in only
  when every item is ≤ 1,024 residue tokens (`reason=above_gate:…`). Numerics: chunk boundaries move.
- Sampler kernels through `opt_core.kernels.apb`'s `fast` tier: `dit_attn_apb` (the 24 token-attention modules), `atom_attn_apb`
  (the atom transformers), `cond_dedupe` (conditioning computed once), `dit_fused` + `dit_lowp` (the token DiffusionTransformer as one
  fused schedule, fp16 operands), `atom_fused` (both atom transformers fused). Numerics: reduced-precision operands, fp32 accumulate.

## big — lowest peak GPU memory

Line `BIG_F` (`BIG_TP` under `--n_gpu P`). This mode binds FlashPairformer kernels via the shared core (opt_core).
`fast`'s levers with the providers bound under their `big` tier word, without `alloc_auto`, `zprep_hoist`, `dit_fused`, `dit_lowp`
(resident-memory costs), on the expandable-segments allocator, plus memory levers decided from the query's residue-token count;
below the offload gate `big` is `fast`'s resident lever set under the `big` words. Numerics: as `fast`.
- `pair_offload_trunk`, `pair_offload_struct`, `pair_offload_conf` (from `MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS`) — the trunk, structural
  and confidence pair tensors live in pinned host RAM and stream through the GPU in row/column blocks, their triangle attention per row block through
  the core provider's `big` tier. With them: `diffz` (no per-step pair clone in diffusion, the pair LayerNorm memoised across
  steps), `bigln_guard` (LayerNorm calls kept below torch 2.7.1's 2^31-element limit), `free_templ` (template pair features dropped
  after the trunk), `sample_chunk` (diffusion samples one at a time); `chunk_lift`, `stepgraph`, `keep_pool` and `tmpl_dedup` are
  off while the unit is on.
- `no_dit_hoist` (from `MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS`) — the conditioning hoist left out; the stock per-step statement runs.

### `--n_gpu P`

- `rowpair_tp` — every pair track (pair init, trunk, template, MSA module, structural stage, diffusion conditioning, confidence
  head) row-sharded over P rank processes through `opt_core.mem.rowpair`; no host offload; each rank holds its row block of the pair
  tensor. Rank 0 featurises each item and broadcasts the features; every rank's feature digest must agree (`refused:
  feats_ranks_differ`). Refused by name: P > 1 under `off`, `exact` or `fast`; fewer than P GPUs visible. Cannot be left out alone.
- `tp_triatt`, `struct_pair_bf16` — each rank's triangle-attention row windows on the core's tier dispatch (`ROWPAIR_TRIATT_CORE=torch`
  opts out), the sharded triangle multiplication on the core's fused row-block kernels with the torch statements as the counted
  fallback; the structural pair shard and refiner pair stack in bf16 (`ODDE_TP_STRUCT_PAIR_DTYPE=fp32` keeps fp32). Off this line:
  `chunk_lift`, `stepgraph`, `keep_pool`, `sched_host`, `structok_sync`, `tmpl_dedup`, `zprep_hoist`, `prefetch`, the offload levers.
  Numerics: tier 2 vs P = 1 (tiled contractions).

## Every mode

- STOCK.md's two speed settings and the stdin detach apply identically on every mode, `off` included. No upstream fix is shipped.
- Small-input floor: when every item of a `pred` call is below `MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS` residue tokens, the trunk levers
  (`arm_z` / `arm_u`+`arm_u23`, `fpf_trimul_exact` and the provider bindings riding them) are composed out (`reason=below_gate:…`).
- Upstream's `--triatt_kernel` / `--trimul_kernel` stated with a value other than `auto`: the kit's levers at that site step aside
  (`reason=aside:stock_knob:<flag>=<value>`) and upstream's selection runs.
- Registered but on no shipped mode: `sampler_amp` (sampler under bf16 autocast), `writer_overlap` (result dump on a background
  thread), `xl_tri_ln` (LayerNorm'd pair copy freed before chunked triangle attention), `dit_attn_bf16` (bf16 SDPA in DiT attention).

## Switches

- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — a mode without the named levers (`reason=ablated`; `ablated=<names>` on the ACTIVE line); a
  lever that carries others takes them with it; `rowpair_tp` and `bigln_guard` cannot be left out alone.
- `--det 1` — upstream's deterministic recipe on every mode; `--allow-partial` — accept a run whose activation is `PARTIAL`.
- Kit variables that move lever gates — `MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS`, `MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS`,
  `MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS`; values and defaults: STOCK.md.
