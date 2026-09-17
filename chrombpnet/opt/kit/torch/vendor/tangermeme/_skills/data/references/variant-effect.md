# Variant effect scoring (substitution / deletion / insertion)

`tangermeme.variant_effect` scores what a sequence edit does to a model's output.
All three functions apply `func` (default `predict`) before and after the edit and
return `PerturbationResult(y_before, y_after)` — there is **no built-in distance**,
you compute your own (Euclidean, log-fold, etc.). They ride on `predict`, so
multi-output models return lists and `args=` carries extra inputs; swap
`func=deep_lift_shap` to get attributions before/after instead.

Alphabet indices are positions in `alphabet` (default A=0, C=1, G=2, T=3).

## substitution_effect — change characters in place

```python
from tangermeme.variant_effect import substitution_effect

# substitutions: COO tensor, shape (-1, 3) = (example_idx, position, new_char_idx)
subs = torch.tensor([[0, 1050, 2]])          # example 0, pos 1050 -> 'G'
y_before, y_after = substitution_effect(model, X, subs)
```

**`position` is relative to the extracted window `X`, not a genomic coordinate.** A
variant at genome position `g` in a window starting at `w` goes at `position = g - w`
(account for any `max_jitter` expansion). Off-by-window errors silently mis-score.

### Footgun #1 — edits are per-example, not per-variant

Rows are grouped by `example_idx` and **all applied to that one sequence at once**;
they do not each yield an independent result. Two rows on example 0 give you the
*combined* effect, not two scores. To score N variants **independently**, replicate
the example N times and put each variant on its own copy:

```python
Xr = X[0:1].repeat(N, 1, 1)
subs = torch.stack([torch.tensor([i, pos[i], alt[i]]) for i in range(N)])  # one per copy
y_before, y_after = substitution_effect(model, Xr, subs)
```

### Footgun #2 — one substitution per `(example, position)`

**Duplicate `(example_idx, position)` coordinates raise a `ValueError`**, because the
edit is a vectorized assignment that would set every listed alt base to 1 and leave a
multi-hot column. There is no last-write-wins. To score several alternate alleles at
one position, replicate the example so each allele gets its own row of `X` (as
above). The same position in *different* examples is fine.

`deletion_effect` and `insertion_effect` are unconstrained here: duplicate deletion
rows collapse to a single deletion, and insertions are applied one at a time in a
loop.

## deletion_effect — remove characters (needs over-length input)

```python
from tangermeme.variant_effect import deletion_effect

# deletions: shape (-1, 2) = (example_idx, position)  — no char needed
dels = torch.tensor([[0, 1050], [0, 1051]])
y_before, y_after = deletion_effect(model, X_long, dels, left=False)
```

- Deleting shortens the sequence, but the model needs a fixed window, so **`X` must
  be `model_length + max_deletions_per_example`** wide. Every sequence is trimmed to
  the same post-deletion length even if it has fewer deletions.
- **tangermeme cannot check this for you** — it has no idea what length your model
  expects, and only raises when the deletions would consume the entire sequence
  (`max_deletions >= X.shape[-1]`). Pass a model-length `X` and nothing is caught: a
  fixed-length model fails with its own shape error, and a length-flexible model
  silently scores a shortened window.
- `left=` chooses which side to trim back to `model_length` (`False`/right is
  default). Use whichever side matches how the model was trained.
- **`y_before` is computed on the *trimmed* slice of `X`, not the raw input** — so
  the before/after delta isolates the deletion. Don't expect `y_before` to equal a
  prediction on full-length `X`.

## insertion_effect — add characters (trim after)

```python
from tangermeme.variant_effect import insertion_effect

# insertions: shape (-1, 3) = (example_idx, position, char_idx)
ins = torch.tensor([[0, 1050, 0]])           # insert 'A' at pos 1050
y_before, y_after = insertion_effect(model, X, ins, left=False)
```

- Mirror image of deletions: pass **model-length** `X`; the insertion lengthens it
  and the result is trimmed back to `model_length`, with `left=` choosing the side.

## From a VCF

Read variants with `io.read_vcf` (see [io-loci.md](io-loci.md)) and build the COO
`substitutions` tensor (`example_idx`, `position` relative to your extracted window,
`alt` allele index). Remember one row per variant **and one example copy per variant**
if you want per-variant scores.

## Related references

[func-pattern.md](func-pattern.md) (swapping `func`/`additional_func_kwargs`),
[io-loci.md](io-loci.md) (`read_vcf`, extracting the windows),
[model-wrapping.md](model-wrapping.md) (multi-output / multi-input models).
