# Chai-1 kit — what changes vs stock

Stock = `chai_lab` 0.6.1 at the pin (STOCK.md); `off` loads nothing of the kit. A kit mode runs upstream's `run_folding_on_context` once per
(input, seed) from the kit driver (`opt/forward/errata_02/chai_worker.py` over `opt/forward/fast_inference/kit/`) with the exported modules
re-expressed as eager PyTorch (`opt/forward/eager_trunk`, `chai1_eager`) and the denoiser-step add-on (`opt/forward/dstep_megakernel`,
`chai1_fastln`) installed over them. A mode is all of its levers (`opt/chai1_opt/modes.py`; numerics class per lever in `registry.py`); each
mode below includes the previous mode's levers unless a line says otherwise. Lever names are the ones printed on the run's `LEVER name=…`
lines; a lever that cannot serve a call steps aside by name to the module's own statement, counted on the `EXIT` line, never silently.

## exact — outputs identical to stock

Bit for bit under `--det 1` on the pinned stack; on compute capability 8.0 through the 1024-token crop (STOCK.md §Stock exceptions).

- `W1` — the six exported modules and the ESM model stay resident across inputs. Numerics: bitwise (caching only). Steps aside: never.
- `W2` — the feature context is built once per input and reused across its seeds. Bitwise (caching only).
- `W5` — ESM embeddings memoised across inputs (`CHAI1_OPT_ESM_MEMO_SCOPE=global`; `input` recomputes per input as stock does). Bitwise.
- `tier1` — the exported model as eager PyTorch: copy-free trunk layouts, flat embedders and confidence head, the denoiser step's
  step-invariant statements hoisted out of the step loop and the step replayed from a CUDA graph (un-graphed under `--det 1`). Bitwise.
- `hoist2` — the hoist partitioned by value dependence: work that depends on neither the noised coordinates nor sigma (atom-pair update
  block and its LayerNorms, blocked pair-bias projections, AdaLN conditioning) runs once per sample, not per step; same ops and operands; the
  cache is released when the input's sampling ends. Bitwise. Steps aside: per input under `big` past the card's gate (`CHAI1_BIG_HOIST2_MAX_N`).
- `templ_empty` — the template embedder's empty-template short cut (the module's own algebra when no template slot carries a mask). Bitwise.
- `exactln` — the trunk's and confidence head's LayerNorms through the shared core's LayerNorm provider (`opt_core.kernels.ln`) by the mode's
  tier word per statement class (`opt/forward/chai1_exactln/serve.py`); under `exact` the ATen-replica row, bit-compared against torch at each
  class's first call (a differing class is refused by name); `statement:<row>` where the provider names the stock op. Bitwise on `exact`.
- `msa_pad` — the MSA module's row work stops at the last MSA row carrying a mask, on the export's own slice grid (dropped slices are
  all-masked: exact zeros); first call per shape class bit-compared whole-vs-cut (a class that differs serves the whole statement
  and books `fallback:not_bitwise`, which the run's exit gate refuses by name). Bitwise.
- `transition` — the trunk's four transition classes by the tier word per (card, class, crop) (`chai1_eager/transition_core.py`); on `exact`
  fed the statement's own LayerNorm output and bit-compared per class at first call. Kernel: FlashPairformer transition via the shared core
  (`opt_core.kernels.transition`, Triton); exact | fast variant. Steps aside: `fallback:stock | no_cell | refused | class_differs`, counted.
- `rankcc` — upstream's ranker with the clash census counted on existing atoms (first sample bit-compared with upstream's statement; a
  mismatch refuses the lever). Bitwise. `tailasync` — CIF writing and the MSA plot on one writer thread, joined before the fold returns.
- `confmemo` — one conformer-library tokenizer per process instead of per input. `prefetch` — the next input's feature context built on a
  helper thread; steps aside under the MSA / template server options and with ESM embeddings on. Both bitwise (`postproc.py`, `featfast.py`).
- `alloc` (implied) — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in the driver process before CUDA initialises. Placement only.
- memory gate — at the 2048-token crop `msa_chunk` and `nograph` (§big) apply to that input (`[chai1-opt] MEMORY …` per input;
  `CHAI1_BIG_MSA_CHUNK_MIN_N` / `CHAI1_BIG_NOGRAPH_MIN_N` move the gate); smaller crops run the mode unchanged.

## fast — within stock's seed-to-seed variation

This mode binds FlashPairformer kernels via the shared core (opt_core).

- `tf32` (implied) — TF32 tensor-core products for the process's fp32 GEMMs (`precision.py`). Numerics: TF32 matmuls.
- `v4trimul` — the pairformer's and MSA module's triangle multiplication by the mode's tier word (`opt_core.trimul.by_word`). Numerics: bf16
  re-association. Kernel: FlashPairformer triangle multiplication via the shared core — CUDA-native (sm_90a / sm_80) or Triton member per the
  provider's table (card, pair width, crop, direction); fast variant. Steps aside: a refused call (`fallback:<row>:<reason>`) runs the
  statement, counted. Not in `exact`: no member is bitwise to Chai-1's bf16 statement.
- `triattn` — the trunk's triangle attention (both directions, pair bias with the pair mask folded in, no key mask) by the tier word,
  `form="bias_only"`, at every crop and card (`triattn_core.py`). Numerics: bf16 products, fp32 softmax (cuDNN SDPA's class). Kernel:
  FlashPairformer triangle attention via the shared core (`opt_core.kernels.triattn`) — CUDA-native (sm_90a / sm_80) or Triton member per the
  provider's row; fast variant. Steps aside: `stock_statement` (counted); `refused:<kind>` refuses the exit gate; `fallback:row_error`.
- `trunk_n` — each trunk call runs on the leading ceil64(live tokens) positions of its token-indexed inputs instead of the padded crop, outputs
  written back zero-padded. Numerics: reduction lengths change (never in `exact`). Steps aside: `fallback:no_pad` when the input fills its crop.
- `msa_rows` — the MSA module runs on the MSA rows that carry a mask instead of all 16,384 padded rows. Numerics: reduction extent on deep MSAs.
- `dit_attn` — the diffusion transformer's 16 pair-biased token attentions per step through the shared core's pair-biased attention provider
  (`opt_core.kernels.apb`) by the tier word, on a 4-D view instead of SDPA's math path over 5-D fp32 operands (`chai1_fastln/dit_attn.py`; the
  row per token count is on the EXIT line). Numerics: memory-efficient-attention order. Steps aside: `stepped_aside:no_apb_row_<kind>`.
- `compiled` — the hoisted per-step function through torch.compile / Inductor (pointwise and layout glue between the step's GEMMs fused into
  Triton kernels), replayed inside the step's CUDA graph. `run.sh warm` exports the step ahead of time (torch.export + AOTInductor,
  `chai1_fastln/aoti.py`) into `$MODEL_OPT_JIT_ROOT/<key>/aoti/`, one package per crop and `--num-diffn-samples` count, weights bound from
  the live module at load; a fold loads a matching package (`compile=on:aoti`) or compiles through Dynamo at the first input of each crop
  (`compile=on:dynamo`). Numerics: Inductor lowering of TF32 arithmetic. Steps aside (the eager hoisted step serves): a package for other
  code or another sample count (absent), a launcher for a CPU ISA the host lacks (`cpu_isa_mismatch`), `no_layout_meta`, cc 8.0 without
  packages (`compile=off:card:no_aoti_packages_cc80`), a compile or launch error (`compiled=stepped_aside:<reason>`); `--no-compile` → `compile=off:user`.

## big — lowest peak GPU memory

Numerics as `fast`; its providers are asked the tier word `big`; `msa_rows` as in `fast`. The memory line (`big.py` over `opt_core.mem`)
is fixed: no switch removes a lever. One GPU per fold: `--n_gpu 1`, a larger value is refused by name (`ngpu.py`).

- `msa_chunk` — the MSA pair-weighted averaging on 1,024-row blocks (`opt_core.mem.chunk.chunk_rows`; row-local: the same statement per
  row) at every crop. Numerics: as `fast`. Steps aside: never (a site an input never enters is a named skip on the EXIT census).
- `trunk_chunk` — the pairformer's triangle multiplication on output-row blocks and its triangle attention on query-row blocks (each block
  through the module's own SDPA statement), from the card's gate: the 2048-token crop on a card with 60 GiB or more, the 1536-token crop
  below that, read once from device memory at model build (`CHAI1_BIG_TRUNK_CHUNK_MIN_N` overrides; `MEMORY memclass=80g|40g …` names
  it). Numerics: as `fast`. Steps aside: below the gate, or the trunk narrowed below it by `trunk_n` (a named skip).
- `opm_chunk` — the outer-product mean on pair-row blocks, from the same card gate (`CHAI1_BIG_OPM_CHUNK_MIN_N`). Numerics: as `fast`.
- `nograph` — the hoisted denoiser step un-graphed at every crop (no private graph pool); `alloc` — expandable allocator segments.
- A memory lever whose gate engaged and that left no mark makes the input partial (exit 3 unless `--allow-partial`).

## Every mode

- Stock exceptions (STOCK.md §Stock exceptions) apply identically on every mode, `off` included. The kit ships no opt-in upstream fixes.
- Printed lines: `ACTIVE` (mode, levers requested / applied, `compile=`, `big=`, `n_gpu=`), `SETTINGS`, one `LEVER` line per lever,
  `MEMORY` and `FORWARD` per input, `SEED` per seed, and the `EXIT` census of what served and what stepped aside.

## Switches

- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — a mode without the named levers (`PARTIAL off=<names> (MODEL_OPT_LEVERS_OFF)` on the ACTIVE line);
  `tier1`, `W1` and `big`'s memory line cannot be switched off alone and say so; a name the mode does not compose is refused (exit 3).
- `--no-compile` — `MODEL_OPT_LEVERS_OFF=compiled`. `--det 1` — deterministic torch / cuDNN algorithms and `CUBLAS_WORKSPACE_CONFIG` on any
  mode; the denoiser step runs un-graphed under it. `--allow-partial` (`CHAI1_OPT_ALLOW_PARTIAL=1` on the `CHAI1_OPT` route) — a partial
  activation proceeds, named, instead of being refused. The memory gates' variables: STOCK.md §Variables.
