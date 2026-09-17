# Measuring motif effects (marginalize / ablate / space)

These are tangermeme's flagship "what has the model learned" tools. All three
apply a function before and after a sequence edit and return both, so you compute
the effect as the delta. (Start with the vignette "Inspecting what Cis-Regulatory
Features a Model has Learned" — it is the canonical use case.)

## marginalize — add a motif into backgrounds

```python
from tangermeme.marginalize import marginalize

y_before, y_after = marginalize(model, X, "CTCAGTGATG")   # PerturbationResult
effect = y_after - y_before
```

- `motif` may be a string or a one-hot tensor; it is **substituted** (length
  preserved). `start=` is the **first position** of the substitution (not the
  center); the default centers the motif.
- `X` should be background sequences so the delta reflects the motif "in isolation".

### Choosing backgrounds — the GC mirage (important)

Uniformly random sequences are **much higher in GC content** than real genomic DNA,
and models are sensitive to local GC — so substituting a motif into the wrong
background can produce a "mirage" effect driven entirely by the GC change, not the
motif. In order of preference:

1. Real inactive regions matched on GC/dinucleotide content (best).
2. `random_one_hot(..., probs=...)` with genomic base frequencies (e.g. hg38
   ≈ `[0.291, 0.209, 0.209, 0.291]`), or `dinucleotide_shuffle` of real sequences.
   `probs` must sum to exactly 1 — numpy raises otherwise, so don't round the
   frequencies independently.
3. Uniform random — prototyping only.

Returns diminish past ~100 backgrounds.

## ablate — shuffle out a region

```python
from tangermeme.ablate import ablate

y_before, y_after = ablate(model, X, start=990, end=1010, n=20, random_state=0)
```

- Replaces `[start:end]` with `n` shuffles and reports before/after.
- The internal shuffle is **non-deterministic without `random_state=`** — always
  pass a seed when you need reproducible / regression values.
- Conceptual opposite of marginalize: ablate removes signal from real sequences;
  marginalize adds signal into backgrounds.

## space — distance dependence between motifs

```python
from tangermeme.space import space

# motifs are SEQUENCES, not TF names; spacing is nested even for one gap
y_before, y_afters = space(model, X, motifs=["AGATAA", "CAGCTG"],
                           spacing=[[5], [10], [20], [40]])   # SpaceResult
```

Returns `SpaceResult(y_before, y_afters)` where `y_afters` covers each spacing —
use it to find cooperative/competitive distance preferences (output axis is
`(n_examples, n_spacings, ...)`).

- **`spacing` shape is `(n_spacings, n_motifs-1)`** — entry `[i, j]` is the gap in
  experiment `i` between motif `j` and `j+1`. Even a single two-motif experiment
  needs the nested form `[[10]]`; a flat list raises `ValueError: spacing has shape
  ... but must have shape (-1, 1)`. Sweep one gap with `torch.arange(50)[:, None]`.
- Characters **inside the gaps are left as background** (not overwritten).

## Multi-task and multi-input models

- **Multi-task:** when the model returns multiple outputs, `y_before`/`y_after`
  come back as a **list of tensors**, one per output. Index before subtracting.
- **Multi-input:** pass extra model inputs through `args=(tensor, ...)` (ablate)
  or kwargs forwarded to `func`; each arg's batch dim aligns with `X`.

## Return type

`marginalize` and `ablate` return `PerturbationResult(y_before, y_after)`;
`space` returns `SpaceResult(y_before, y_afters)`. Both are NamedTuples — unpack
positionally (`y_before, y_after = ...`) or by attribute (`r.y_before`), and
`isinstance(r, tuple)` is True.

### Annotation variants (per-region, in batch)

`marginalize_annotations(model, X, X0, annotations)` (X0 = backgrounds) and
`ablate_annotations(model, X, annotations)` apply the effect to each annotated
region individually and return `PerturbationAnnotationsResult(y_befores, y_afters)`.
`annotations` is a `(n_annotations, 3)` tensor `(example_idx, start, end)`. Both
loop one annotation at a time and stack, so the output carries an **extra leading
axis** and the second axis differs between them:

- `ablate_annotations` → `(n_annotations, 1, ...)` and
  `(n_annotations, 1, n_shuffles, ...)`. The `1` is real, not `n_examples`: each
  annotation is ablated on its own single example `X[idx:idx+1]`, so you almost
  always want to squeeze it.
- `marginalize_annotations` → `(n_annotations, len(X0), ...)` for both, since each
  annotation's span is substituted into every background in `X0`.

Index carefully before taking deltas. See [annotate.md](annotate.md) for building
the annotation tensor.

## Attributions instead of predictions

All three take `func=`. Swap in `deep_lift_shap` to see the effect on
attributions rather than predictions; route attribution kwargs through
`additional_func_kwargs=` (see [func-pattern.md](func-pattern.md)):

```python
from tangermeme.deep_lift_shap import deep_lift_shap
attr_before, attr_after = marginalize(model, X, "CTCAGTGATG", func=deep_lift_shap,
                                      additional_func_kwargs={'target': 0})
```

- **Set `target` explicitly** — `deep_lift_shap`'s default is output 0, so on a
  multi-task model you silently attribute the wrong head.
- **Cost blows up:** attribution runs before *and* after, each building `n_shuffles`
  (default 20) references — 5 examples ≈ 200 forward/backward passes. Shrink the
  attributed window via `additional_func_kwargs={'start': ..., 'end': ...}`.
- **Backgrounds-as-references trick:** `func=deep_lift_shap` with `n_shuffles=1` over
  many background sequences trades per-example reference averaging for more examples,
  giving cleaner aggregate attributions cheaply.

## Related references

[func-pattern.md](func-pattern.md), [deep_lift_shap.md](deep_lift_shap.md),
[saturation_mutagenesis.md](saturation_mutagenesis.md),
[model-wrapping.md](model-wrapping.md).
