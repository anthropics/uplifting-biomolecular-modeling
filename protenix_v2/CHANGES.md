# Protenix v2 kit — what changes vs stock

Stock = Protenix 2.0.0 at the pin (STOCK.md). Each lever is a module the kit installs over one stock function or class at start-up under a
kit mode; `off` loads none and the installed `protenix` package is never edited. A mode is all of its levers; `fast` includes `exact`'s levers
and `big` includes `fast`'s unless a line says otherwise. Lever names are the ones printed on the run's `LEVER name=…` lines.
`opt/forward/flashpairformer` carries the trunk and sampler levers (`env.sh` is its switch table: `ARM=E` for `exact`, `ARM=T` for `fast`),
`opt/forward/PTX_TP` the multi-GPU line, `opt/forward/DIT_FUSE` the diffusion transformer's fused elementwise kernels; `opt/protenix_opt`
resolves a mode into that environment (`modes.py`, lever table `registry.py`), applies it in-process, gates it per card and prints the
activation lines. Numerics class per lever: exact = the stock statement's arithmetic, bitwise; tolerance = bf16 / fp16 rounding-class
differences.
Kernels come from the shared core (`../common/opt_core`) — its Triton packages `fpf_triatt_pro`, `fpf_triatt_epi`, `fpf_glue_v2`,
`fpf_mkpf`, `fpf_triatt_k2b`, `fpf_transition` and its providers `opt_core.kernels.{trimul, triattn, transition, apb, ln}`, which pick a
kernel per (compute capability, Triton version, shape) cell by the mode's tier word; a provider without a cell for the running card says
so by name and the lever either runs the stock statement by name or the mode refuses (README 'Notes'). Inputs at `--dtype fp32|fp16`
(stock's option) run every mode: kernels that serve bf16 only step aside per call to the stock computation, counted on that lever's
`LEVER` line (`aside=…`).

## exact — outputs identical to stock

This mode binds FlashPairformer kernels via the shared core (opt_core).

- Pairformer block path (`blk2_block_path`, `glue_v2`, `mk_pf`, `nomask`, `blk2_chunked_exact`, `triatt_prologue_cuda`): each of the 48
  trunk blocks (and the MSA-module / confidence-head pair stacks) runs LayerNorm + q/k/v/g/bias projections as one fused prologue kernel
  (Triton; on sm_90 a hand-written TMA + wgmma kernel, `triatt_prologue_cuda`), the stock cuEquivariance triangle-attention kernel without
  the all-ones mask, and output gating + projection as one fused epilogue kernel; row-chunked above the memory policy's token threshold.
  On cc 8.0 the fused statement starts at 400 tokens; below it the stock statement runs by name. exact.
- Triangle attention through the shared core by the word `exact` (`triatt_exact`): its bit-identical row where vouched, else the library op. exact.
- Padded-to-8 layout for the stock triangle-attention kernel above 512 tokens on cc 9.0 (`pad8`). exact.
- Triangle multiplication, c_z 256, both directions, through `opt_core.kernels.trimul` by the word `exact` (`trimul_core_exact`); the
  template embedder's c 64 pair stack likewise (`templ_trimul_tmk3`); tuned cuEquivariance tile choices for the calls that stay on the
  library, served from the kit's autotune cache (`cueq_tuned_tiles`; `cueq_cache/*.json`: a cc 9.0 and a cc 10.0 table, four and six large-M entries, produced by running this kit's workloads
  through the library's autotuner on cards of those compute capabilities and stored in the library's cache-file format (every other shape runs the kernel's default tiles); kit-generated data, no entry of the library's packaged tables, no other NVIDIA SDK data; the LEVER line's `tiles=` pair says whether a table for the running card is present). exact; a shape the provider has no bitwise cell for takes the library op by name.
  Kernel: FlashPairformer triangle multiplication and transition via the shared core — CUDA-native (sm_90a) and Triton members by cell; exact variant.
- Transitions through `opt_core.kernels.transition` by the word `exact` (`transition_core_exact`), the two-GEMM body with a fused
  silu×mul kernel answering the calls the provider hands back (`t1_fused_transition`). exact.
- Diffusion and atom attention as one bit-exact Triton launch per call reproducing the stock statement's order of operations
  (`dit_attn_exact`, `atom_attn_exact`). exact.
- Template embedder evaluates each distinct template once (`template_dedupe`); the 41 modules whose checkpoint weights are ~1e-37 are
  skipped (`deadskip`, outputs exactly 0); zero-template pair-weighted-average cache (`pwa_zcache`); random init skipped at model
  construction (`lazy_init`). exact.
- CUDA graphs: the 48-block stack for inputs up to 448 tokens (`stackgraph`, replay checked once against the eager call per signature) and
  the diffusion sampler's denoiser step with the step-invariant tensors hoisted (`sampler_graph`, `sampler_prep`, `sampler_reach`,
  `sampler_graph_cache_policy`; up to 1536 tokens on a device with at least 64 GiB, 995 below); `sampler_admit` decides per item from the
  projected sampler peak whether the graphed or the eager sampler runs and names the route. exact.
- Sampler elementwise chains (AdaLN tail, output gate, gated residual, SwiGLU) as one fused Triton kernel each (`sampler_fuse`). exact.
- Runtime policy, no numeric effect: prebuilt stream-correct build of stock's fast-LayerNorm extension for the installed torch
  (`fastln_prebuilt`; source-built by name when none matches), stock fast LayerNorm selected (`layernorm_fast`), memory policy for large
  inputs (`xl_policy`: row-chunk threshold sized to the device, expandable-segments allocator), allocator pool kept across the confidence
  head (`keep_pool`), host-side chain bookkeeping for the summary confidences (`summary_hostidx`), previous item's prediction released at
  the next item's start (`pred_release`), per-process lever records (`lever_report`).

## fast — within stock's seed-to-seed variation

`exact`'s set, with these in place of their exact counterparts:
- Flash triangle attention in the block core through `opt_core.kernels.triattn` by the word `fast` (`triattn_native`; `k2b_flash_triattention`
  / `blk2_chunked_k2b` where the provider has no cell); below 300 tokens the block core steps down to the exact statement (`smalln_size_gate`).
  tolerance.
  Kernel: FlashPairformer triangle attention via the shared core — CUDA-native (sm_90a / sm_80), Triton where the provider holds no cell; fast variant.
- Triangle multiplication and transitions by the word `fast` (`trimul_core`, `transition_core`, template stack `templ_trimul_esm`,
  `templ_triatt_core`); standalone LayerNorms through `opt_core.kernels.ln` (`ln_core`). tolerance.
  Kernel: FlashPairformer triangle multiplication and transition via the shared core — CUDA-native (sm_90a) and Triton members by cell; fast variant.
- MSA module: fused OuterProductMean and MSAPairWeightedAveraging (`opm_fused`, `pwa_fused`, `opt_core.ops.msa_fused`); Pairformer
  AttentionPairBias with a fused bias producer and attention core (`pf_attn`, `opt_core.kernels.apb`). tolerance.
- Diffusion sampler: DiT pair-bias attention and atom local attention as fused Triton flash-attention launches (`dit_attn`, `atom_attn`),
  fp16 tensor-core operands with fp32 statistics and accumulation (`dit_attn_fp16`, `dit_lowp`), the 24 token blocks and the atom
  transformers as fused stacks with merged projections (`dit_fused`, `atom_fused`), diffusion conditioning evaluated once per step and
  broadcast over the samples (`cond_dedupe`). tolerance. `pad8` is not part of `fast`.

## big — lowest peak GPU memory

`fast`'s levers without `stackgraph` and `keep_pool`, with the sampler CUDA graph's token cap set below every input (the DiT hoist
runs eagerly per item), plus:
- `guard_lift`: the stock runner's refusal above 2560 tokens lifted; the item runs at protenix-v2's own precision settings. exact.
- Memory levers on the shared core's registry (`opt_core.mem`): `drop_bond_mask` (the unread int64 atom×atom bond mask not built),
  `cond_chunk` and `apb_bias_chunk` (diffusion conditioning cache and attention pair bias computed 256 token rows at a time above 1023
  tokens; tolerance), `relp_lazy` (relative-position one-hot never materialised, the encoding linear run per 128-row block; tolerance),
  `msa_zfree` and `diffcache_free` (pair tensors released as soon as their last reader has run), `cache_release` (allocator emptied at the
  phase seams). exact unless marked.

### `--n_gpu P`

- P > 1: `torchrun` starts P ranks on one machine (`opt/forward/PTX_TP`, `opt/protenix_opt/tp.py`); the pair representation
  is sharded by rows across the ranks through the shared core's pair-block driver and row kernels (row triangle multiplication, triangle
  attention per query-row block by the word `big`, DiT query-block attention on `opt_core.kernels.apb`), in the trunk, MSA module,
  template embedder (each rank builds only its rows of the template pair features) and confidence head; the sampler caches the DiT pair
  biases of the local rows when they fit the free memory at sampler entry. The launcher appends `--trimul_kernel torch`; the ACTIVE /
  FINAL / EXIT lines carry `n_gpu=P sharding=rowpair`, the `EXECUTION` line the ranks' completion and row-shard map; a run short of P
  ranks exits 1. tolerance (fast's class).

## Every mode

- No stock exception: stock runs as released on every mode and `off` is exactly stock (STOCK.md 'How stock is run').

## Switches

- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — a mode without the named levers (`ablated=<names>` on the ACTIVE / FINAL lines; the levers'
  LEVER lines read `state=off reason=ablated`; a name outside the mode is refused, exit 3).
- `--det 1` — the deterministic recipe on any mode, `off` included: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PTX_DET=1` (torch's
  deterministic algorithms switched on at interpreter start), and the deterministic scatter copy over the installed
  `protenix/utils/scatter_utils.py`, restored at exit (STOCK.md 'How stock is run'); `exact --det 1` equals `off --det 1` bit for bit.
- Kit variables a caller may pre-set to change lever behaviour (names only; the mode's own value applies when unset):
  `PTX_SAMPLER_GRAPH_MAXTOK`, `PTX_FPF_CHUNK_TOK`, `PTX_LAZY_INIT`, `PTX_SAMPLER_FUSE`, `INFOPT_FASTLN_PREBUILT`; values and the
  deployment variables: STOCK.md.
