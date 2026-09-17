# OF3 → AF3 conversion verification

Manual verification harness for `src/alphafold3/model/of3_weight_converter.py`.
These are **not** unit tests and do not run in CI — they need a real OpenFold3
checkpoint and, for the module comparisons, an importable `openfold3`. The fast
unit tests that do run unattended live in
`src/alphafold3/model/of3_weight_converter_test.py`.

Run them after any change to the converter or to an `of3_weights`-gated branch in
the model. Between them they caught four real porting bugs; see
`OF3_AF3_PORTING_NOTES.md` for the findings.

## Prerequisites

| variable | meaning | fallback |
|---|---|---|
| `OF3_CHECKPOINT` | OpenFold3 `.pt` checkpoint | `../af3_of3/weights/of3-p2-155k.pt` |
| `OPENFOLD3_PATH` | checkout of `aqlaboratory/openfold3` | `../openfold3` |
| `AF3_NATIVE_DIR` | directory with native AF3 `af3.bin.zst` | `../af3_of3/weights/af3_native` |

Fallbacks assume the repos are siblings. `torch` is required throughout; `jax` +
`haiku` for the `compare_*` scripts; `AF3_NATIVE_DIR` is optional (only
`audit_structure.py` uses it, and it degrades gracefully).

The public checkpoint is at `s3://openfold/staging/of3-p2-155k.pt` (no sign-in).

## Structural audits

Cheap, and they catch dropped or misnamed weights.

```
python3 audit_structure.py     # unread tensors; diff vs native AF3 names/shapes
python3 audit_probed.py        # as above, but distinguishes _has() probes from _get() reads;
                               # also lists square matrices (shape-invisible transposes)
python3 audit_unconsumed.py    # proves the unread tensors are the sample_diffusion duplicates
python3 dump_param_digest.py out.json   # {param: [shape, md5]}, to diff two conversions
```

Expected: every tensor read except the `sample_diffusion.*` duplicates and
`version_tensor`; no shape mismatches against native AF3 apart from the
documented `of3_weights` deviations (the diffusion transformer layer-stack
naming, and `single_cond_initial_*` at 833 rather than 831 rows).

`dump_param_digest.py` is the tool for "did my converter edit change anything I
did not intend": dump before and after, then diff the JSON.

## Module comparisons

These run the *real* OF3 PyTorch module and the *real* AF3 Haiku module on
identical inputs, with converted weights, and compare outputs. This is the only
check that catches square-matrix transposes, SwiGLU gate-vs-value order, and
block ordering inside a layer stack — none of which change any shape.

```
python3 compare_pair_features.py            # trunk/MSA input embeddings, confidence pair embed, template aatype
python3 compare_pairformer_block.py [i]     # trunk PairFormer block i (default 0; try 0 and 47)
python3 compare_msa_block.py [i]            # MSA module block i (try 0 and 3 — the last drops the MSA update)
python3 compare_template_block.py [i]       # template pair-stack block i
python3 compare_confidence_pairformer.py [i] # confidence-head pairformer block i (the PAE path)
python3 compare_diffusion_transformer.py    # all 24 diffusion transformer blocks
python3 compare_diffusion_conditioning.py   # Algorithm 21, incl. the relpos feature layout
```

Expect relative errors around 1e-5 or below (float32 accumulation noise) and
Pearson r of 1.00000000. Absolute magnitudes look large because the random
inputs are not in-distribution; judge by the *relative* error.

Each `compare_*` script honours `SABOTAGE=1`, which perturbs one weight so you
can confirm the harness actually discriminates before trusting a pass:

```
SABOTAGE=1 python3 compare_pairformer_block.py     # expect MISMATCH, rel ~0.4
```

Always sanity-check with `SABOTAGE=1` when adapting a script to a new module —
it is easy to write a comparison that passes because it is testing nothing.

## Coverage

The comparisons cover ~99% of the 368M converted parameters. Not covered: the
atom cross-attention transformers and atom feature embedders (~3.4M), because
AF3's `CrossAttTransformer` needs a constructed `queries_to_keys` gather layout —
a hand-built stand-in risks a false result more than it buys confidence. Those
parameters are covered by the structural audits only.
