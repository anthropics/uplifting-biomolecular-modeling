# PR (esm 3.3.0, models/esmfold2/paired_msa.py): vectorise `msa_to_res_type_and_deletions`

`msa_to_res_type_and_deletions(msa, letter_to_res_type)` converts an MSA into `(res_type[M, L] int64, deletions[M, L] float32)` with a
pure-Python double loop over every row and every character of the a3m text: one `is_a3m_insertion()` call, one `dict.get` and one
`str.upper` per character. `ESMFold2InputBuilder.fold()` runs it inside `prepare_input` on every fold (once per protein chain), so it sits
inside any per-prediction timing window: on an H100 box, for a 6-chain 1400-token complex with full-depth MSAs it is ~13.6 M calls and
~3 s of a 3.3 s `prepare_input` (85-90 %), i.e. ~8 % of a 40 s Full-model fold and ~20-25 % of a 2.5 s 200-token fold; the Fast model
(no MSAs) is unaffected. The same module tree already ships the vectorised idea for one row (`esm.utils.msa.msa.a3m_deletion_counts`),
and `MSA.from_a3m` already stores per-row deletion counts that this function then recomputes character by character and discards.

This patch computes the same two arrays over one bytes view of the whole MSA: an insertion mask (lowercase a-z or '.', exactly
`is_a3m_insertion`), per-row match counts (`np.add.reduceat`), and — when every row holds exactly `L` match columns, which
`MSA.from_a3m` guarantees — `bytes.translate` to drop the insertion characters and a 256-entry lookup table built from the caller's own
`letter_to_res_type` ('-' -> gap, unknown -> `PROTEIN_UNK_RES_TYPE`, as the loop's branches) applied to the `[M, L]` byte grid.
Deletions: the stored `msa.deletions.astype(float32)` as before, else insertion-run lengths by cumulative-sum differences. Non-ASCII
rows and ragged rows keep the per-character loop (renamed `_msa_to_res_type_and_deletions_loop`). Outputs are integer ids and float32
counts of small integers, so the result is identical arrays, not approximately equal ones: checked with `np.array_equal` + dtype + shape on
every chain MSA of 122 test inputs (540 chain MSAs, 4.57 M rows,
1.30 G characters, read at full depth with insertions kept; each also re-checked with the stored deletions removed so the insertion-run
path is exercised) and on 206 synthetic ragged / dotted / unknown-letter / empty / non-ASCII cases (identical arrays or the identical
exception); end-to-end mmCIF + PAE sha256 identical under the kit's deterministic recipe (Full model, exact and fast lever sets). Function
time over that sweep 252.0 s -> 12.7 s (18-23x per input); in-window `prepare_input` of a Full-model fold (H100 box, 6 protein chains):
0.53 -> 0.19 s at 200 tokens, 1.00 -> 0.26 s at 400, 3.0 -> 0.62 s at 1400 (the rest is taxonomy pairing, tokenisation, tensor build).

Diff: `U6_msa_features_vectorised.diff` (against Biohub esm 3.3.0 @ 26b0bc2b771e3e419ea74f445a5f35cc094a1509). The kit obtains the
same effect in-process (`driver/ef2_feats.py`, lever `fz`) without editing the vendored tree.
