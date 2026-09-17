# OB0-002 — the chunk-size tuner's argument comparison raises when one stack's call form changes rank (openfold3 0.5.x (OpenBind-0) as pinned in STOCK.md; its record is `(shape, itemsize)` per tensor, same comparison; always-on on `fast` as cell `tuner_guard`)

Not an opt-in flag: the `fast` line is the configuration that reaches the defect, so its cell `tuner_guard`
(`opt/openfold3_ob0_opt/cells/tuner_guard.py` → `opt_core.of3_trunk.tuner_guard`) carries the correction; `off` and `exact` run the stock configuration, which
never reaches it, and are byte-unchanged.

WHAT UPSTREAM DOES. `core/utils/chunk_utils.py ChunkSizeTuner` (one per `PairFormerStack`) tunes a chunk size on the first chunked call and
caches the call's argument record (`tune_chunk_size`: the shapes of `args=(s.clone(), z.clone())`); every later chunked call is compared with
that record (`_compare_arg_caches`: `for a1, a2 in zip(ac1, ac2, strict=True)`, recursing into tuples — `torch.Size` is one — with
`assert type(a1) is type(a2)`) and a difference re-tunes ("If args have changed shape/value, we need to re-tune"). The confidence head
(`core/model/heads/prediction_heads.py PairformerEmbedding`) calls its stack in two FORMS: at or below `per_sample_token_cutoff` (750) tokens
`pairformer_emb` folds the samples into the leading dimension (`s` rank 3, `z` rank 4) — and keeps the caller's `chunk_size` unless one of
the alternative-kernel flags is on (`if use_kernels and si.shape[0] > 1: chunk_size = None`); above the cutoff `per_sample_pairformer_emb`
loops over samples with the batch dimensions kept (`z` rank 5). Comparing a rank-4 record with a rank-5 one makes the strict `zip` raise
`ValueError: zip() argument 2 is longer than argument 1` instead of answering "changed".

SYMPTOM. One `run_openfold predict` process on a runner configuration whose kernel flags are off (the kit's `fast` line) with a query set that
contains a query of at most 750 tokens BEFORE a larger one: every larger query fails in the confidence head with the ValueError above (the
traceback ends in `chunk_utils.py _compare_arg_caches`); each query alone, or the larger ones first and only, predicts fine. The stock
configuration (cuEquivariance / DeepSpeed flags on) runs the batched form unchunked, caches nothing there, and is not affected.

HOW WE KNOW. H100, four ladder queries of 400 / 400 / 800 / 800 tokens in one process: stock configuration (`--mode off`) exit 0, 20/20
structures; the kernels-off configuration without the cell exit 1, both 800-token queries failed (10/20); with the cell exit 0, 20/20, the
cell's census `resets=ValueError:1` (the one form switch); the order 800 / 400 / 800 / 400 likewise.

WHAT THE FIX CHANGES. `opt_core/of3_trunk/tuner_guard.py` wraps `ChunkSizeTuner._compare_arg_caches`: a comparison that raises (`ValueError`
from the strict zip, `AssertionError` from the type check — records of different structure) returns `False`, the tuner's own "changed"
answer, so it re-tunes for the new form and caches the new record; comparable records go through upstream's comparison untouched. Chunk size
does not enter the arithmetic of a chunked layer (upstream tunes it per process already): outputs unchanged.

UPSTREAM'S OWN PATH. A one-line upstream change would do the same (`_compare_arg_caches` returning `False` on records of different structure,
or `tune_chunk_size` treating them as changed); until the pinned release carries it, the cell does.
