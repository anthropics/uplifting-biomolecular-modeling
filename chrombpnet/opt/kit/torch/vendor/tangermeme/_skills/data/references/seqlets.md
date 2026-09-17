# Seqlet calling (recursive_seqlets / tfmodisco_seqlets)

Seqlets are contiguous spans of high attribution — the "where are the motifs"
step. `tangermeme.seqlet` has two callers with genuinely different behavior.

## The #1 footgun: input is a 2D projected track

Both callers take **per-position projected attribution sums**, shape `(n, length)`
— i.e. the default (`hypothetical=False`) `deep_lift_shap` output collapsed over the
channel axis:

```python
attr_sums = X_attr.sum(dim=1)        # (n, 4, length) -> (n, length)
```

Do **not** pass the `(n, 4, length)` attributions or hypothetical attributions.
(See [deep_lift_shap.md](deep_lift_shap.md): hypothetical is for building CWMs, not
for seqlet calling.)

## recursive_seqlets — sharp, p-value based

```python
from tangermeme.seqlet import recursive_seqlets

seqlets = recursive_seqlets(attr_sums, threshold=0.01, additional_flanks=0)
# DataFrame columns: example_idx, start, end, attribution, p-value (sorted by p-value)
```

- Returns **positive seqlets only**. For negative ones, run on `attr_sums.abs()` and
  re-extract signed attribution from the returned boundaries.
- `threshold` is a p-value; it effectively requires ~`1/threshold` candidate spans,
  so the right value **depends on how many examples/positions you pass**. Large runs
  often use `0.001`.
- `min_seqlet_len` (default 4) is also the smallest span the recursive property must
  hold over — **don't set it to 1**. `max_seqlet_len` (default 25) raises cost.
- `additional_flanks` pads the reported `start`/`end` **and** the reported
  `attribution` is re-summed over the padded span, so it grows with the flanks. The
  `p-value` is unchanged — it is computed on the un-padded core. Flanks may overlap
  neighbors even though the cores cannot, so flanked spans are not disjoint, their
  attributions can double-count the same positions, and
  `pairwise_annotations_spacing` rejects them (see [annotate.md](annotate.md)).
- **Over-calls on a single example** (it can't fit a null from one sequence) — pass
  many examples, or threshold by magnitude.

## tfmodisco_seqlets — longer, discovery-oriented

```python
from tangermeme.seqlet import tfmodisco_seqlets

seqlets = tfmodisco_seqlets(attr_sums, window_size=21, flank=10)
# DataFrame columns: example_idx, start, end, attribution (positive + negative sets)
```

- Replicates TF-MoDISco's caller: seqlets are **longer and less sensitive** by
  design (local context helps motif discovery; flanks get trimmed downstream).
- **Adaptively lowers its threshold to hit a minimum count → it over-calls**; expect
  to filter the results by attribution afterward.
- Boundaries are **left-offset/asymmetric** because each position is replaced by the
  sum of the window to its right (the chosen position is the left edge).

## Which to use

`recursive_seqlets` for sharp, minimal spans and per-seqlet p-values;
`tfmodisco_seqlets` when feeding a motif-discovery pipeline that wants generous
spans. Both flow straight into annotation and plotting.

## Related references

- **Label / count seqlets** → [annotate.md](annotate.md) (`annotate_seqlets`,
  `count_annotations`, `pairwise_annotations`).
- **Plot seqlets** → [plot.md](plot.md). Three things to get right: filter to one
  example, pass `score_key='attribution'`, and move a name column to the front —
  `plot_logo` reads the label from column 0, which in a seqlet frame is
  `example_idx`.
- **Where the attributions come from** → [deep_lift_shap.md](deep_lift_shap.md)
  (projected vs hypothetical).
