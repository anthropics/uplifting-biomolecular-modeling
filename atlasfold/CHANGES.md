# AtlasFold kit — what changes vs stock

Stock = AtlasFold v1.0.0 at the pin (STOCK.md). Each lever is a module the kit installs over one stock function or class at
start-up under a kit mode; `off` loads none; no file under `stock/src` is modified. `fast` = `exact`'s levers with `trimul_v4` in
place of `trimul_exact`, plus its own; `big` = `fast` minus `denoiser_graph` and `graph_reuse`. Lever names are the ones
`bash run.sh check` prints and `MODEL_OPT_LEVERS_OFF` takes. Kernel levers hand the mode's tier word to the shared core's provider
for the op, which picks the row per call; a call a lever does not serve runs the stock statement, counted on its `LEVER` line
(`fallback_by=<reason>`). Switches and thresholds (`AFO_*`): STOCK.md §Variables.

## exact — outputs identical to stock (bitwise under `--det 1` on the same card)
This mode binds FlashPairformer kernels via the shared core (opt_core).
- `trimul_exact` — triangle multiplication, word `exact`. Numerics: bitwise. Kernel: FlashPairformer triangle multiplication via
  the shared core; exact variant — on the pinned stack every call is answered with the stock cuEquivariance op.
- `triattn_exact` — the triangle-attention call, word `exact`: on a card and stack the shared core holds a bitwise record for (kept per compute capability, torch and
  cuEquivariance version; the `LEVER` line names the row served), its exact-class row (a fused kernel bit-identical to the stock cuEquivariance call); elsewhere that stock
  call by name (`stock_row:cueq`), or on cc 8.0 at large N the same call one head at a time (`exact_headsplit`, less memory). Numerics: bitwise. `fast` / `big`: `flash_triattn` serves.
- `triatt_block_exact` — projections around the stock attention call as prologue + epilogue kernels; bf16, c_z 128, size band per
  card. Numerics: bitwise. Kernel: FlashPairformer prologue/epilogue kernels via the shared core (attention core: the stock call).
- `transition_exact` — SwiGLU `Transition.forward`, word `exact`, for shapes recorded bitwise (else `stock_row:`); idle in `fast`
  / `big`. Numerics: bitwise. Kernel: FlashPairformer transition via the shared core — Triton; exact variant.
- `ln_bf16` — bf16 `LayerNorm.forward` as one `F.layer_norm` call (fp32 statistics in-kernel, one rounding) instead of upcast →
  LayerNorm → downcast. Numerics: bitwise on the pinned torch. Steps aside: `fp32_input`, `fp32_params`, `cpu`.
- `exactln` — every `LayerNorm.forward` form, word `exact`. Numerics: bitwise. Kernel: the shared core's LayerNorm kernel (CUDA
  compiled at first use), a bit-for-bit ATen replica where recorded so; else ATen by name (`stock_row:`, `no_affine:`).
- `dit_apb` — the diffusion transformer's pair-biased attention (rank-4 calls). Kernel: the shared core's pair-biased attention
  kernels (Triton); under `exact` no bitwise row exists on the pinned stack, so every call is the stock statement (`stock_row:`).
- `pae_stream` · `pae_stream_m` — monomer · AtlasFold-M confidence head: each sample's PAE / PDE logits reduced with the stock
  functions as produced instead of stacked first (`n/a` for the head not loaded). Numerics: bitwise.
- `conf_transition_chunk` · `pair_transition_chunk` — pair `Transition` inside · outside the confidence heads applied in row
  blocks (intermediate ≤ `AFO_TRANSITION_CHUNK_MIB`) into one output. Numerics: bitwise (row-wise map).
- `distogram_offload` · `relpos_lazy` — fp32 distogram logits moved to pinned host memory when produced · relative-position
  one-hots recomputed inside their three consumers instead of held in the batch. Numerics: bitwise (placement / recompute only).
- `sampler_hoist` · `sampler_hostsync` — the roll-out's step-invariant statements evaluated once per `DiffusionHead.sample` and
  reused every step · its per-step host synchronisations made once per roll-out. Numerics: bitwise. Steps aside: `source:`.
- `atom_kdedup` — the windowed atom cross-attention's key side (LayerNorm, AdaLN, k/v projections) evaluated once per atom row
  instead of per overlapping window, returned as the same strided views. Numerics: bitwise. Steps aside: `kv_layout`.
- `output_overlap` — multimer runner: a finished batch's host post-processing (SASA, structure objects, mmCIF / PDB text) runs in
  one CPU worker while the next batch's forward runs; files and order identical. Steps aside: `runner:monomer`, `source:`.
- `alloc_expandable` — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the kit process (allocator policy only). Steps
  aside: an operator's own setting (`user_conf:`); beside `denoiser_graph` in `fast` unless `AFO_ALLOC_EXPANDABLE=graphs`.
- `lever_report` — the `LEVER` lines and the exit tally.

## fast — within stock's seed-to-seed variation
- `trimul_v4` — triangle multiplication, word `fast`: the fastest admitted row per card and size. Numerics: bf16 re-association.
  Kernel: FlashPairformer triangle multiplication via the shared core — CUDA-native (sm_90a / sm_80), Triton; fast variant.
- `flash_triattn` — the triangle-attention call, word `fast`, below `triatt_block`'s floor. Numerics: softmax re-association.
  Kernel: FlashPairformer triangle attention via the shared core — CUDA-native (sm_90a / sm_80), Triton; fast variant.
- `triatt_block` · `triattn_core` — all of `TriangleAttention*Node.forward`: prologue → core → epilogue kernels; bf16, c_z 128, ≥
  512 tokens. Numerics: bf16 re-association. Kernel: FlashPairformer triangle attention via the shared core; fast variant.
- `pair_transition` · `pair_block_residual` — bf16 pair transitions as one fused LayerNorm → SwiGLU → projection kernel, residuals
  and mask folded in. Numerics: fused rounding. Kernel: FlashPairformer transition via the shared core — Triton; fast variant.
- `lm_sdpa` — AtlasLM attention: the exported q·kᵀ logits exactly as stock, softmax·V through `F.scaled_dot_product_attention`.
  Numerics: SDPA-kernel rounding. Inputs below `AFO_LM_SDPA_MIN_TOKENS` run stock.
- `diffusion_bf16` — bf16 autocast over `DiffusionHead.sample` (stock runs the diffusion transformer in fp32); the atom encoder /
  decoder keep their fp32 islands. Numerics: bf16 matmuls.
- `atom_tf32` · `atom_bf16` — inside those fp32 islands: TF32 GEMMs · bf16 autocast, coordinate-facing linears and the
  augmentation kept fp32. Numerics: TF32 / bf16 matmuls. Steps aside: `cc<8`, `cpu`.
- `atom_sdpa` — the windowed atom attention through torch's memory-efficient SDPA kernel. Numerics: SDPA-kernel rounding. Steps
  aside: `rank`, `high_precision`, `cpu`.
- `atom_rows` — the atom blocks' conditioned transition on fused LayerNorm-modulate / SwiGLU / gate row kernels. Numerics:
  fused-kernel rounding. Kernel: the shared core's row kernels (Triton). Steps aside: `source:`, `cpu`.
- `denoiser_graph` — `DiffusionModule.forward` captured into a CUDA graph at each roll-out's first step, replayed every step
  (static buffers, RNG outside). Above `AFO_DENOISER_GRAPH_MAX_TOKENS`: eager (`above_gate`); a failed capture: exit 3 by name.
- `graph_reuse` — keeps the captured graph and every buffer it reads for the next input of the same shape and replays instead of
  recapturing (one input's graphs alive at a time).
- `exactln`, `dit_apb`, `triattn_exact` receive the word `fast`: the provider's tolerance-class row where it has one, else as
  under `exact` (`triattn_exact` hands its calls to `flash_triattn`, `tier_beneath:`); `AFO_*_WORD` overrides.

## big — lowest peak GPU memory
- The `fast` row without `denoiser_graph` and `graph_reuse` (the CUDA-graph pool is `fast`'s one memory cost); providers receive
  the word `big`. Numerics: as `fast`. `--n_gpu` accepts 1 in every mode (P > 1: `rowpair_tp_not_in_<version>`).

## Every mode
- No stock exception and no upstream fix exists (STOCK.md); the stock command line and run settings are untouched. Per-input
  `PHASE` / `PEAK` timing lines print on every route, `off` included (timing only).

## Switches
- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — a mode without the named levers and those requiring them (`ablated=<names>` on the ACTIVE
  line). `--det 0|1` — deterministic recipe. `--allow-partial` — proceed when a lever cannot install.
