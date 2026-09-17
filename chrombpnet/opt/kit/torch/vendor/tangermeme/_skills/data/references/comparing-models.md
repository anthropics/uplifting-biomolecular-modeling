# Comparing predictions and attributions across N models

Running tangermeme over several models (replicates, architectures, an ensemble, a
fine-tuning sweep) is mostly "loop the single-model calls" — `predict` and
`deep_lift_shap` move each model to `device` and restore its state per call, so
iterating is safe. The traps are not in the looping; they are in making the
results **comparable**. This file covers the four that bite.

Start from the single-model flow in [notebook-walkthrough.md](notebook-walkthrough.md);
everything here assumes you already know how to run one model.

## 1. Fair attribution comparison needs shared references (the big one)

`deep_lift_shap` attributes relative to *random* shuffled backgrounds (default
`references=dinucleotide_shuffle`, `n_shuffles=20`). If you let each model draw its
own backgrounds, you are comparing models **and** reference noise. Hold the
references fixed across models.

Best: precompute references once and pass them to every model.

```python
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.deep_lift_shap import deep_lift_shap

refs = dinucleotide_shuffle(X, n=20, random_state=0)   # (n, 20, 4, length)
attrs = {
    name: deep_lift_shap(model, X, target=0, references=refs, device=device)
    for name, model in models.items()
}
```

A fixed `random_state=0` on each call is the lighter-weight alternative, but
explicit shared `references=` is unambiguous and also lets `only_warn=True` paths
work (see [deep_lift_shap.md](deep_lift_shap.md)).

## 2. Harmonize outputs before correlating

Models rarely agree on shape. Before any concordance metric:

- **Output window:** different `out_window` / cropping — slice all predictions to a
  common genomic window centered the same way.
- **Task identity:** `target`/output index `k` may mean different assays in
  different models — map by name, don't assume aligned columns.
- **Input length:** if input windows differ, re-`extract_loci` per model at its own
  `in_window` from the *same loci* rather than reusing one `X`
  (see [io-loci.md](io-loci.md)).

```python
preds = {name: predict(m, X, batch_size=64, device=device)
         for name, m in models.items()}
totals = {name: y_hat.sum(dim=-1) for name, y_hat in preds.items()}   # (n, n_tasks)
```

## 3. Normalize attribution scale before comparing logos/heatmaps

Attribution magnitudes are not comparable across models (different output scales,
different reference gaps). For visual or quantitative comparison, normalize per
model — e.g. divide each example's attributions by its per-example L2 norm, or
z-score — before stacking into a logo grid or computing cross-model similarity.

```python
def _norm(a):                      # a: (n, 4, length)
    return a / a.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-8)

sim = {                            # cosine similarity of projected attributions
    (i, j): torch.nn.functional.cosine_similarity(
        _norm(attrs[i]).flatten(1), _norm(attrs[j]).flatten(1)).mean()
    for i in attrs for j in attrs if i < j
}
```

## 4. Keep N models on CPU; let `device=` shuttle them

Because every `predict` / `deep_lift_shap` call moves the model to `device` for the
duration and restores its original device afterward, you do **not** need all N
models resident on the GPU. Build/load them on CPU, keep them in a dict, and pass
`device=device` to each call — only one model occupies GPU memory at a time. This
is what makes a large sweep fit.

```python
models = {name: torch.load(p, weights_only=False) for name, p in paths.items()}
# all on CPU; each call below moves one over and back
```

This is the *across-models* half of memory management. The *within-call* half —
lowering `batch_size` (which counts example×reference pairs) or `n_shuffles` when a
single `deep_lift_shap` call OOMs — is in
[deep_lift_shap.md](deep_lift_shap.md), under "`batch_size` counts example-reference
pairs".

## Concordance / ensembling (generic — brief)

Once predictions are harmonized, the comparison itself is ordinary analysis:
per-task Spearman/Pearson across loci, prediction deltas per locus to find where
models disagree, or a mean/median ensemble. These are not tangermeme-specific, so
use whatever you normally would (`scipy.stats`, `numpy`).

## Related references

[notebook-walkthrough.md](notebook-walkthrough.md) (the single-model spine),
[deep_lift_shap.md](deep_lift_shap.md) (references / `random_state`),
[io-loci.md](io-loci.md) (per-model windows),
[model-wrapping.md](model-wrapping.md) (aligning heterogeneous model outputs).
