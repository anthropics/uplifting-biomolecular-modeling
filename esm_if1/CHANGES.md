# ESM-IF1 kit — what changes vs stock

Stock = fair-esm 2.0.1 at the pin (STOCK.md): `examples/inverse_folding/sample_sequences.py`, one `model.sample()` call per requested sequence.
The kit's one lever is a driver around the unmodified `esm.inverse_folding` modules and checkpoint — nothing is patched over a stock function,
no upstream file is modified, and `off` loads none of it. Modes are `off | fast` (`opt/esm_if1_opt/modes.py`); the lever name is the one printed
on the run's `LEVER name=…` line. The kit binds no shared-core kernel: it uses the shared core (`common/opt_core`) for the mode table, the
report lines, the run record, the stock-subprocess check and the install gate.

## fast — within stock's seed-to-seed variation

Numerics class here: same distribution as stock, different draws — stock is unseeded, the kit seeds and batches the draws. `fast` is the default mode.

- `batched_sampling` (`opt/esm_if1_opt/batched.py`, run in one child process `esm_if1_opt.kit_design` for the whole run) — replaces the
  script's per-sequence loop with batched forward passes. Every input is parsed once up front (`PREPASS`); the model is loaded once by the
  script's own two statements and moved to the GPU unless `--nogpu` or no GPU is visible (`STARTUP … device=cuda|cpu`);
  `torch.manual_seed(--seed)` is applied once. Rows are (backbone, sample) pairs in input order, each backbone's samples adjacent, cut into
  batches of at most `--batch_size` rows that share one backbone length. Per batch, upstream's `GVPTransformerModel.sample()` statements
  run over B rows instead of one: `CoordBatchConverter` over the rows, the `<cath>`-prefixed all-`<mask>` token matrix, one encoder call,
  L incremental decoder steps, `softmax(logits / T)` and `torch.multinomial` per row, under `torch.inference_mode`. `--multichain-backbone`
  is batched the same way (`sample_complex_batch`): per row the chains are concatenated with the designed chain first and upstream's
  10-residue pads (`_concatenate_coords`), upstream's `<mask>`/`<pad>` partial sequence is laid on every row, rows share one (concatenated
  length, designed length), and the decode loop stops at the designed chain's last position — where upstream's returned sequence ends —
  so the random stream is consumed as upstream consumes it. A structure's FASTA (upstream's record format) is written as soon as all its
  rows are sampled; one `CALL` and one `ITEM` line per batch, then `OUTPUTS_WRITTEN`, `PEAK`, `KERNELS`. The parent composes
  `LEVER name=batched_sampling state=on … rows_per_forward=B batches=n rows=n` from the child's `ITEM` records (`state=off
  reason=no_batch_ran` when the child ended before its first forward pass).
  Numerics: upstream's fp32 arithmetic, statement for statement; at `--batch_size 1` and the same generator state each record equals what
  `model.sample()` (or `sample_sequence_in_complex`) returns; at B > 1 the multinomial draws consume the random stream B rows per step, so
  the sequences are a different draw from the same distribution. At a fixed seed, batch size and input order the sequences repeat run to
  run; floating-point values do not (the graph network aggregates neighbours with float atomics). Kernel: none of the kit's own —
  upstream's modules on the pinned torch. Steps aside: never — no card, stack or shape condition; a batch that exceeds GPU memory ends the
  pass with torch's error (exit 1, `incomplete=k/n`), with no fallback to a smaller batch.

## Every mode

- Stock exceptions: none (STOCK.md). Opt-in upstream fixes: README, Known upstream issues; each is a module `upstream_issues/<ID>_<slug>.py`
  loaded by path in the process that runs the model — after the environment check under `off` — and announced with
  `[esm_if1-opt] UPSTREAM-FIX <ID> applied`. `ESMIF1-001` rebinds `GVPTransformerModel.sample` to sample on the model's own device; no
  effect unless `--nogpu`.
- Weights: when `ESM_IF1_WEIGHTS` names a file, `design` makes torch.hub's cache entry a symlink to it so upstream's loader reads it without
  a download (`WEIGHTS … hub_entry=linked`).
- Run record: `opt_manifest.json` (mode, lever, device, settings, seed, batch size, inputs, outputs, each child's command and status, the
  exit rule) and `timing.jsonl` (the child's lines as JSON) in the output directory; under `off` also `stock_env_proof.json`.

## Switches

- `--seed N` (default 37) and `--batch_size B` (`--batch`; default 64) — inputs of every `fast` pass; reported `NOT APPLIED` under `off`.
- `--det [0|1]` (default 0) — the deterministic-recipe switch; nothing to switch on this model (no deterministic-algorithm setting is
  needed for the sequences to repeat, and none exists for the graph network's scatter aggregation): `--det 1` is reported `NOT APPLIED`
  under either mode.
- `--upstream-fix <ID>[,<ID>…]` (default none) — above. `ESM_IF1_OPT=<mode>` — the mode when `--mode` is absent (STOCK.md, Variables).
