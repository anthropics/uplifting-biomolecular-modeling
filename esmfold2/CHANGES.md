# ESMFold2 kit — what changes vs stock

Stock = Biohub `esm` 3.3.0 + the `transformers` 4.57.6 fork at the pin (STOCK.md). Each lever is a module the kit installs over one stock
function or class of the loaded model before its first fold, under a kit mode, in the fixed order of the driver's mode table
(`opt/forward/fast_inference/driver/ef2_server.py` `MODES`, entries `opt7x` / `opt14_msa`; modules in `opt/forward/fast_inference/driver/`
and `opt/forward/EF2_XL_ADDON_v1/`); `off` loads none. A mode is all of its levers on a GPU class: each engages, steps aside by name
(`state=skipped reason=<word>`), or the mode refuses before folding (`NOT ACTIVE: partial activation …`, exit 3). Lever names are the ones
printed on the run's `LEVER name=…` lines (`opt/esmfold2_opt/registry.py`). (Full) = acts on ESMFold2 only (the model with the MSA module),
(Fast) = on ESMFold2-Fast only; on the other model the line reads `not_for_variant`.

## exact — outputs identical to stock

Identical to stock's `off --backend fused` configuration, bit for bit under `--det 1` (`server_mode=opt7x`).

- `fused` — the base of every kit mode: `set_kernel_backend('fused')` + `set_chunk_size(None)`. Numerics: this IS the stock configuration compared against.
- `tg`, `eg`, `sg` — CUDA-graph capture and replay of the pair trunk / coda / confidence trunk, of the LM and MSA encoders, and of the
  diffusion sampler with its per-step host syncs removed (random draws in stock order); one graph generation per input shape
  (`GRAPHS … released=`). Numerics: bitwise (scheduling only). Steps aside: per-site token budgets (`EF2_GRAPH_BUDGET_TOKENS`,
  `…_TRUNK` / `_ENCODER` / `_SAMPLER`) run larger shapes with the same kernels uncaptured (`graph budget … -> eager`).
- `ls`, `rg` (Fast) — the recycle loop with loop-resident static I/O; one CUDA graph per recycle, replayed for the fold's remaining
  recycles. Numerics: bitwise. Steps aside: `rg` above `EF2_RECYCLE_GRAPH_MAX_TOKENS` (512) pair tokens, named per fold.
- `ec`, `fc`, `pb` — in-process caches, emptied on a new input: language-model hidden states across the seeds of one input; the builder's
  `prepare_input` features (a SMILES-ligand input bypasses it per seed); the diffusion transformer's step-invariant pair bias per fold. Numerics: bitwise (caching only).
- `t3`, `t5`, `t6` — the fused backend's pair-trunk kernels with a re-tuned Triton tile table per compute capability, bf16 weight casts
  cached once, the pair transition as a row-block kernel. Numerics: bitwise (same arithmetic in the same order). Steps aside: on classes
  with less than 160 KB of shared memory per block `t1` substitutes for `t6` and `t3` leaves the set (`substituted=` on the `APPLIED` line).
- `xte`, `xtr` (Full) — the trunk's d=256 `Transition`, and the MSA module's `PairTransition` / `msa_transition`, through the shared core's
  `opt_core.kernels.transition` under the tier word `exact` (rows `esm_fused_exact`; `flash_sm90a` / `v1` with the module's LayerNorm output
  given). Numerics: bitwise. Steps aside: a card, stack or size the provider holds no bitwise row for keeps the module's statement (`word=` `rows=` `refused=`).
- `xln` — pair-sized `nn.LayerNorm` sites (≥ 4,096 rows) through `opt_core.kernels.ln` row `exactln` (ATen's forward, bit for bit, in one
  pass). Numerics: bitwise. Steps aside: by name on a class or stack the row does not serve.
- `ax`, `ro` — the atom transformer's per-fold constants hoisted (adaLN factors, var-len attention layout, RoPE tables); the diffusion
  sampler as a device roll-out (per-step scalars from device tables folded as the stock loop folds them, conditioning once per fold, steps
  replayed from a captured graph). Numerics: bitwise. Steps aside: `ro` outside one diffusion sample at batch 1 (`sg` runs; `scope=`).
- `fz` — MSA featurisation vectorised on the host. Numerics: identical arrays (checked against upstream's function on first use; steps aside on a difference).
- `mh` (`full_msa`), `trimul` / `glue` (Full), `disto` — the MSA encoder's output reused across a fold's recycles when no row subsample is
  drawn; the MSA triangle multiplication without permute copies, its sigmoid-gate glue as one checked kernel; distogram logits copied to
  pinned host memory asynchronously. Numerics: bitwise.

## fast — within stock's seed-to-seed variation

This mode binds FlashPairformer kernels via the shared core (opt_core). The default mode (`server_mode=opt14_msa`).

`exact`'s levers except `t6`, `xte`, `xtr`, `trimul`, `glue` (the fused kernels below take those sites), plus:

- `tx` — the whole pair TriMul of every `PairUpdateBlock` (both directions) through `opt_core.kernels.trimul` under the mode's tier word:
  the provider's cell table picks the row per compute capability, stack, width, size class and direction. Numerics: bf16 re-association.
  Kernel: FlashPairformer triangle multiplication via the shared core — CUDA-native (sm_90a / sm_80) | Triton; fast variant. Steps aside: a class whose cell names the stock op, or a
  refused call, is served by the upstream fused TriMul by name (`tx:` line per class; `refused=` `stock_classes=`).
- `t10`, `t15`, `t15msa` (Full), `t16` — the pair transition (trunk `Transition`, MSA-block `PairTransition`) as one fused kernel through
  `opt_core.kernels.transition` under the tier word. Kernel: on compute capability 9.0 row `esm_t16` (warp-specialised CuTe C++, sm_90a:
  TMA-fed, in-register LayerNorm, wgmma dual GEMM → SiLU·mul → out-GEMM → residual, fp32 accumulate) and `t10` / `t15` / `t15msa` read
  `not_for_class:sm90:superseded_by_t16`; on other classes `t16` is off by class and `t15` / `t15msa` (row `esm_t15`, Triton) with `t10`
  (Triton) serve. Numerics: fused-kernel rounding. Steps aside: a refused call keeps the module's statement by name.
- `msa` (Full), `af` — the fused TriMul inside the MSA encoder; the atom-transformer block as three fused Triton kernels and the
  atom-to-token mean as a segmented reduction, GEMMs bf16-in / fp32-accumulate (`gemm=bf16`). Numerics: bf16 re-association.
- `kd`, `dit` — the per-step rigid alignment on the device (3×3 SVD kernel); the diffusion-transformer step fused (Triton elementwise
  chains, pair-biased attention through flash-attention 2.8.3, token GEMMs and conditioning bf16-in / fp32-accumulate:
  `gemm=bf16 cond=bf16 attn=flash attn_precision=bf16`). Numerics: bf16 re-association. Steps aside: a batch of several inputs
  (`scope=ndsN_batchN` names where it ran: inside `ro` at one sample, inside `sg` at more).
- `m15`, `m16`, `m17` (Full) — MSA transition + residual as one row-block kernel; pair-weighted averaging; outer-product mean with its
  LayerNorm, contraction and projection fused (Triton). Numerics: fused-kernel rounding.
- `x2b`, `x3`, `x6`, `x7`, `x8`, `x10` — storage only, each bitwise against the set without it: pair storage freed after its last consumer
  (single sample); a diffusion-conditioning pair path with one `[L,L,512]` hidden alive; the LM→pair projection and the relative-position
  encoding row-chunked; pair-state init with the same RNG draws selected in place; distogram logits moved to pinned host memory (so
  `disto` reads `state=off reason=dropped_by_line:fast`).

No TF32, no 8-bit format, no model setting changed; each bf16 choice has an fp32 spelling through the ablation variable (§Switches).

## big — lowest peak GPU memory

`fast`'s levers less `mh` (it would pin a pair-sized tensor), with no CUDA graph captured at any site (`EF2_GRAPH_CAPTURE=0`: `tg` / `eg` /
`sg` / `rg` / `ro` run their eager forwards — same kernels, same order) and the `t*` first-call probe off; the `ACTIVE` line reads
`server_mode=opt14_msa+EF2_GRAPH_CAPTURE=0,EF2_W4_IDENTITY_PROBE=0`. Requires `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (exported
when unset; another value refused by name). Numerics: as `fast`. Plus:

- `x4` — from `X4_MIN_TOKENS` (1,500) tokens the language model's blocks are streamed through the LM pass from pinned host memory, one
  block resident at a time. Numerics: as `fast`. Steps aside: below the threshold (nothing to save there).
- With more than one diffusion sample the confidence head runs once per sample (`EF2_CONF_PER_SAMPLE=1`, exported by the mode); `tx` asks
  the provider for row `tx_sm90a` first on eager calls up to 512 tokens (`prefer=`) and `t16` resolves each call class once (`memo=per_class`).

### `--n_gpu P`

P = 2, 4 or 8. `tp.py` launches P rank processes (rank r on visible GPU r, NCCL, one shared `PYTHONHASHSEED`: `RANKENV hashseed=…`) over the shared
core's `opt_core.mem.rowpair`; `rowpair.py`, `rowpair_feats.py`, `rowpair_msa.py`, `rowpair_heads.py` bind ESMFold2's modules onto its
row-sharded statements. Sharded: rows of the pair representation from the pair-state init on — triangle multiplication (row blocks
through `opt_core.mem.rowpair.trimul_fused` on sm_90a from the sharding floor up, on other classes above `EF2_ROWPAIR_TRIMUL_MIN_TOKENS`;
torch statements wherever the core declines a unit, counted by reason on its `F2.trimul_rows` line; `ROWPAIR_TRIMUL_KERNELS=torch`
declines all), pair transition in blocks of `EF2_ROWPAIR_TRANSITION_ROWS` (256), the MSA module's pair paths, diffusion conditioning, the
confidence trunk (once per sample). Each rank holds its row block plus streamed peer blocks; token-level, LM and atom tensors are
replicated. Rank 0 featurises and broadcasts; every rank digests its tensors and a mismatch refuses on every rank (`feats_ranks_equal=`).
The sampler runs the one-GPU fused step on every rank over the conditioned pair gathered whole in bf16 when it fits inside memory the trunk
already peaked at, else row-sharded (flash attention per rank on `opt_core.kernels.apb`, pair-bias rows by `opt_core.kernels.ln_proj`;
`EF2_ROWPAIR_SAMPLER=auto|whole|rows`; the fold line's `sampler_route=` `dit_rows=` `dit_bias=` say what ran). Levers reading
`state=off reason=not_for_route:n_gpu=<P>:…`: `ls rg m16 m17 trimul glue mh disto`, and `dit` on the row-sharded sampler. Refused by name:
P ∉ {2,4,8} or fewer than P GPUs visible (exit 3), `--n_gpu` > 1 under another mode (exit 2); below the sharding floor the input is folded unsharded and says so.

At `--n_gpu` > 1 the set also carries the seven row-chunking levers below (`opt/esmfold2_opt/rowchunk/`; bound on every rank after the pair
stack is sharded, named in `levers=` like the rest). Each re-issues, per row block of the rank's shard, a statement the sharded stack
otherwise runs on the whole shard. They engage from `EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS` (1,024) tokens (`biasfree`, which changes no statement,
at every size): a fold below that runs the unchunked statements exactly, in every output, and says `rowchunk_rows=whole:below_floor` on its
fold line (a declaration, not a refusal). Each is removable by name through the ablation variable; `pdeskip`, `confmem` and `zbf16` ride on
`confrows`, and `confbf16` on both `confrows` and `confmem` (removing `confrows` or `confmem` without the levers riding on it is refused by
name); the two bf16 members' fp32 spelling is their own name in that variable. At `--n_gpu 1` the seven are in no mode's set, so naming one
there is refused by name (exit 3) like any other lever of this kit outside the set. A member that cannot bind, or that stays unused through a
fold at or above the floor, is a partial activation like any other lever (refused by name, exit 3).

- `injrows` — the pair loop's per-recycle inject `z ← a·z + Linear(LayerNorm(z_in))` in row blocks of `EF2_ROWPAIR_INJECT_MB` MiB, residual
  written in place. Numerics: row-local statements with the GEMM split on rows — the whole-shard values up to re-association, not bitwise.
- `confrows` — the sharded confidence statement's `LayerNorm → head → expectation` pairs (PAE, PDE), its `softmax → TM` reduction and the
  residual add of the confidence trunk's output in row blocks of `EF2_ROWPAIR_CONF_MB` MiB, into one preallocated logits tensor. Numerics: as `injrows`.
- `confbf16` — the confidence head's pair prologue (`z_norm` + relative position + bonds + the `s_to_z` terms) built per row block in fp32
  and stored bf16: the confidence trunk's residual accumulates in bf16 and its heads read bf16 activations (each row block upcast before its
  LayerNorm; logits stay fp32). Numerics: pLDDT, PAE, pTM, ipTM computed from bf16 pair activations. Rides on `confrows` and `confmem`.
- `pdeskip` — the PDE head is not evaluated: the fold result's `pde` / `pde_logits` are `None`. Nothing reads or writes PDE at `--n_gpu` > 1
  (`…_pae.npz` and `pred_rows.jsonl` carry pLDDT, PAE, pTM, ipTM), so no written value changes. Rides on `confrows`.
- `confmem` — the pLDDT / resolved heads' weight gathers in atom blocks, the row-attention pooling per row block, the distance-bin tensor
  freed at its last use. Numerics: row-local statements re-blocked; no reduction is split. Rides on `confrows`.
- `biasfree` — the sampler's per-block pair-bias buffers released through the kit's `clear_graphs` once `structure_head.sample()` returns
  (no graph is captured on this route); the next fold allocates them again. Numerics: none — same values, shorter lifetime.
- `zbf16` — the confidence head reads a bf16 copy of the rank's pair rows, built in row bands of `EF2_ROWPAIR_ZBF16_MB` MiB in place of the
  whole fp32 cast at that call (the distogram and the sampler read fp32). Numerics: pLDDT, PAE, pTM, ipTM computed from bf16 pair rows.
  Rides on `confrows`.

The rank's `rowpair fold` line carries `rowchunk=<levers bound>` `rowchunk_floor=<tokens>` `rowchunk_rows=blocked|whole:below_floor` and each
member's counters so far (`rc_<lever>_<counter>=`, e.g. `rc_injrows_blocks=`, `rc_confrows_row_blocks=`, `rc_zbf16_conf_calls=`, `…_floor_skips=`);
the `EXIT` line repeats them as `rowchunk={levers=…,…}`; each member's `LEVER` line names `min_tokens=` and its `block_mb=`. The sharded
pair-state init draws with torch's rejection-form `trunc_normal_`; on a torch whose `nn.init.trunc_normal_` is the inverse-CDF form the route
installs a rejection-form sampler process-wide before the init instead of refusing by name, and says `trunc_normal=shim` on the `rowpair`
LEVER line (the init's random stream is then that sampler's); on the pinned torch nothing is replaced (`trunc_normal=native`).

## Every mode

- Stock exceptions: none (STOCK.md). Upstream's fused pair-bias attention kernel is launched in per-sample row blocks that keep its 32-bit
  offsets inside 2³¹−1 (same kernel, byte-identical to one launch; `pair_bias_row_launches` on the `EXIT` line); proposed upstream patches,
  none applied, are notes under `opt/forward/fast_inference/upstream/`.
- Levers that re-issue an upstream function statement by statement (`ls rg mh m15 m16 m17 t15 t15msa t16 xtr trimul glue ax af ro kd dit fz`)
  are checked at install against a digest of that function's pinned source (`ef2_srcguard.py`); on another upstream the lever refuses by name.
- The registry also names `t1` (the substitute above) and `mk`, `t11`, `t12`, `t13`, `t14` (a sampler-conditioning hoist and
  first-generation MSA kernels), which no shipped mode carries.

## Switches

- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` (alias `ESMFOLD2_OPT_ABLATE`) — a mode without the named levers (`ablate=<names>` on the ACTIVE line,
  `state=off reason=ablated` on theirs); `<lever>.<knob>=fp32` sets a precision knob instead (`af.gemm`, `dit.gemm`, `dit.cond`,
  `dit.attn`, `dit.attn_precision`); a lever another one rides on cannot be removed alone (refused by name); words naming no lever of this
  kit pass to the shared core's kernel packages, which read the same variable.
- `--det 0|1|2` — STOCK.md §How stock is run; the same recipe on every mode. Kit variables that change lever behaviour: `EF2_GRAPH_BUDGET_TOKENS`
  (+ per-site), `EF2_GRAPH_CAPTURE`, `EF2_CONF_PER_SAMPLE`, `EF2_ROWPAIR_SAMPLER`, `EF2_ROWPAIR_TRANSITION_ROWS`, `EF2_ROWPAIR_TRIMUL_MIN_TOKENS`,
  `ROWPAIR_TRIMUL_KERNELS`, `EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS`, `EF2_ROWPAIR_INJECT_MB`, `EF2_ROWPAIR_CONF_MB`, `EF2_ROWPAIR_ZBF16_MB` — values and
  defaults: STOCK.md §Variables.
