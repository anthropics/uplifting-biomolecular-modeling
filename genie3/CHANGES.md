# Genie 3 kit — what changes vs stock

Stock = Genie 3 at the pin (STOCK.md). Under a kit mode the request runs through one persistent driver process
(`opt/genie3_opt/g3batch.py`, over `opt/forward/fast_inference/driver/g3fast.py` and `opt/forward/g3cap/`) instead of a fresh
`genie3 generate`; each lever is a switch of that driver, declared once in `opt/genie3_opt/registry.py` and composed into modes in
`opt/genie3_opt/modes.py`. `off` loads none. A mode is all of its levers: a planned lever that cannot run makes the mode refuse by name
after the pass (`NOT ACTIVE: mode=<m> refused — lever(s) <ids> could not run …`, exit 3), never a subset. Lever ids are the ones printed
on the run's `ACTIVE … levers=…` and `LEVER lever=…` lines. Numerics classes: 1 = byte-identical to eager stock at the same batch size
and seed · 2 = the same arithmetic through another kernel selection · 3 = reduced-precision operands or re-associated reductions.

## exact — outputs identical to stock

- `L1` persistent process — model, sampler and featuriser built once per process; many designs per process. Numerics: none (orchestration).
- `L2` sync-free DDIM step — device-side index tensors, vectorised Frenet frames and a static cond-group count replace host-synchronising
  calls in the step and the embedders (`g3fast_patches.py`). Numerics: class 1 (same values, same kernels). Steps aside: never.
- `L4` CUDA-graph replay (`--cuda-graphs`) — the denoiser core (pair-transform, sequence and structure nets) captured once per batch shape and
  replayed for every step; the embedders stay eager. Numerics: class 1. Steps aside: never.
- `L8` featuriser hoist (`--hoist`) — the pair featuriser's step-invariant terms (relative-position encoding, the four pair masks, the
  conditional-template term) computed once per batch instead of once per step. Numerics: class 1. Steps aside: never.
- `L9` graph reuse (`--reuse-graphs 16`) — captured graphs kept by feature-shape signature, up to 16 per process, oldest freed when a later
  batch would not fit beside it; a later batch of the same shapes replays the kept graph. Numerics: class 1. Steps aside: never.
- `L11` upstream's batch semantics (`--batch-size B`) — B designs per denoiser call in dataset order, one noise draw over the batch tensor
  per step, PDB files written per batch: what `genie3 generate` computes with `generation.dataset.batch_size: B`. B = `design
  --batch_size B`, else the request's key, else upstream's 1. Numerics: class 1 at every B against stock at the same B.
- `L17` lean pair stack (`--lean-pair`) — `LatentTransformer` blocks pass ownership of the pair tensor so a block's input is freed as soon
  as the block rebinds it; on the stock triangle-multiplication path, in-place gating with the bmm operands materialised before the
  un-permuted projections are freed (`opt/forward/g3cap/g3lean.py`). Numerics: class 1 (same ops, values and order; lifetimes only).
- `L18` wide capture (`--wide-capture`) — the hoisted featuriser's per-step tail computed inside the captured region one batch element at
  a time straight into the pre-padded pair tensor; the hoisted terms live in graph-owned buffers refilled in place when a kept graph is
  re-pointed at a new batch. Numerics: class 1 (row-independent ops per element).
- `L19` allocator (`--alloc expandable`) — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in the driver before CUDA initialises,
  and the capacity probe run once per feature-shape signature per process. Numerics: none.

## fast — within stock's seed-to-seed variation

The default mode: `exact`'s levers plus:

- `L12` pair transition per design (`--pt-chunk design`) — `chunk_size` = the pair tensor's row count on the five `PairTransition`
  modules, so each call runs B chunks of one design's N×N×c slab with one design's hidden intermediate live (stock chunks 4 rows at a
  time). Numerics: class 2 (cuBLAS selects another GEMM kernel for the larger row count). Steps aside: never.
- `L13` TF32 matmuls (`--tf32`) — TF32 tensor-core matmuls for the whole process through the shared core's `opt_core.precision.policy`
  (FP32_TF32). Numerics: class 3 (TF32 operand rounding).
- `L7` fused triangle multiplication (`--trimul fpf`) — the ten `TriangleMultiplicativeUpdate` modules served by the shared core's
  `fpf_trimul_v4` through `opt_core.trimul` (Triton; bf16 GEMM operands, fp32 accumulate / LayerNorm / gating; settings per compute
  capability in `opt/genie3_opt/fpf_cells.json`: 9.0 and 8.0). Numerics: class 3 (bf16 re-association). Steps aside: pair extents
  under the kernel's floor of 101 tokens run the module's own forward, counted on the `KERNELS` line; a request whose every call is
  under the floor prints `LEVER lever=L7 state=skipped reason=below_min_tokens mode=fast served_by=module_forward` and the rest of
  `fast` runs. On a card without an entry the shared core's defaults for the capability serve when it has them; where the kernel
  cannot serve the card every call is a counted `no_cell:…` fallback and `fast` refuses by name after the pass (exit 3).
- `L16` compiled core (`--compile`) — `torch.compile` (inductor, static shapes) of the pair-transform / sequence / structure nets, the
  compiled callable recorded by the kit's CUDA graph once per batch shape (traced during the eager warm-up, captured under the
  `fail_on_recompile` stance); L7's module forward stays out of dynamo and inside the graph; cache under `$MODEL_OPT_JIT_ROOT/inductor`.
  One compile per distinct complex length and batch size: fixed-length problems compile once; a binder problem whose binder length
  is drawn per design compiles once per length encountered and reuses the cache thereafter.
  Numerics: class 3 (inductor re-associates fp32 reductions and fuses producer chains).

## Every mode

- The request is upstream's experiment YAML, read as upstream reads it; `design --out_dir/--n/--selections/--seed/--batch_size` are
  composed into the copy `request.yaml` that every mode, `off` included, runs. Dataset shards (`--num-shards M --shard-id K`) and the
  sequence stage (`predict_sequence`) run on every mode with upstream's slicing, shard markers and decode; a shard whose marker exists
  is complete and nothing runs (exit 0). Beam search, `predict_sidechain` and `--num-devices` > 1 are refused by name on `exact`/`fast`
  before anything runs (exit 3); `--mode off` runs them.
- Capacity guard: a batch whose projected resident memory exceeds the card it runs on is refused before it runs (`CAPACITY …
  verdict=refused … suggest_batch=<b>`, exit 3), judged after earlier batches' kept graphs have yielded.
- Environment uncertainty (another GPU, another torch / triton patch level, a card without a kernel-settings entry) is named on the ACTIVE /
  NOTE / LEVER lines and every lever engages. Stock exceptions: none (STOCK.md).

## Switches

- `--det 1` (`opt/genie3_opt/det.py`) — `experiment.seed` (the request's, else 0; `--seed` overrides) through
  `lightning.seed_everything(seed, workers=True)` on every mode in the same order, plus the driver's graph-vs-eager check.
- Shared-core components bound: the kit binds FlashPairformer kernels via `opt_core` — `opt_core.trimul` / `fpf_trimul_v4` (L7); also `opt_core.precision.policy` (L13) and the core pin gate
  (`opt/pyproject.toml [tool.opt_core]`, first on every route: an absent or mismatched core is one `NOT ACTIVE` line, exit 3).
