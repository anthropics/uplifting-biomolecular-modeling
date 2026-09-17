# Annotating and counting motifs (tangermeme.annotate)

Once you have spans of interest (seqlets, or FIMO hits), `tangermeme.annotate`
labels them with motif identities and counts how they occur and co-occur.

## The annotation tensor format

Two conventions show up:

- **Region annotations** `(n, 3)` = `(example_idx, start, end)` (0-indexed, end
  exclusive). This is what `ablate_annotations` / `marginalize_annotations` consume
  (see [motif-effects.md](motif-effects.md)).
- **Counting input** `(n, 2)` = `(example_idx, annotation_idx)` for `count_annotations`
  / `pairwise_annotations`, or `(n, 4)` = `(example_idx, annotation_idx, start, end)`
  for spacing.

## annotate_seqlets — assign a motif to each seqlet (TOMTOM)

```python
from tangermeme.annotate import annotate_seqlets

# returns TWO tensors, each shape (len(seqlets), n_nearest):
motif_idxs, pvals = annotate_seqlets(X, seqlets, "motifs.meme", n_nearest=1)
```

- Returns `(idxs, p_values)` — **motif indices and their TOMTOM p-values**, not
  names (the function's own type hint is wrong about this). A `-1` index means no
  match passed the threshold. Map indices to names yourself, e.g.
  `names = list(read_meme(path).keys()); [names[i] for i in motif_idxs[:, 0]]`.
- `motifs` is a MEME-file path **or** a `{name: PWM}` dict. Matching uses TOMTOM
  (from the external `memelite`), which handles overhangs — so it is the right tool
  for **seqlets** (short spans).
- Returns the `n_nearest` best matches per seqlet. **This only *appears* to dedupe a
  redundant motif DB** — with a non-deduplicated database, near-identical motifs are
  silently split/dropped, so don't trust raw match counts from a redundant DB. Use a
  merged/deduped motif set if counts matter.

### Scanning long sequences instead of seqlets → FIMO (external)

FIMO is **not** in tangermeme; it lives in `memelite` (the memesuite-lite package).
Use it to scan whole sequences (it does not handle overhangs well, so it's for
scanning, not seqlets), then attach attribution by summing `X_attr` over each hit's
span and thresholding. Beware: a redundant DB yields many overlapping hits over the
same span, and attribution-thresholded hits can **double-count** the same span under
several similar motifs — unlike seqlet calling, which assigns one label per seqlet.

## count_annotations — per-example motif counts

```python
from tangermeme.annotate import count_annotations

counts = count_annotations((seqlets['example_idx'], motif_idxs[:, 0]))  # (n_examples, n_motifs)
```

- Accepts a `(n, 2)` tensor or a tuple of two vectors.
- **Default dtype is `uint8`** — counts saturate at 255; pass `dtype=torch.int32` if
  a motif can occur more often.
- `shape=(n_examples, n_motifs)` pins the output dims (needed when stacking matrices
  across models/runs that don't all see every motif). `dim=0`/`dim=1` reduces without
  materializing the full matrix.

## pairwise_annotations — co-occurrence matrix

```python
from tangermeme.annotate import pairwise_annotations

co = pairwise_annotations(df_example_motif, shape=n_motifs)   # (n_motifs, n_motifs)
```

- Default `symmetric=True`: entry counts pairs in both orders (sum = 2×pairs −
  diagonal; take a triangle for the pair count). `symmetric=False` gives an
  **ordered** matrix (row = the motif that came first) — input row order is then
  meaningful.
- **`unique=True` (the default) breaks that ordering.** It collapses repeated motifs
  within an example — 3 KLF4 hits give one KLF4–KLF4 pair, not many — by sorting
  them, which also discards the input row order. So with the default, `symmetric=False`
  orders by *annotation index*, not by position. Pass `unique=False` for a genuinely
  sequence-ordered matrix, at the cost of counting repeats.
- Default `dtype` is `torch.int64` here, unlike `count_annotations`.

## pairwise_annotations_spacing — co-occurrence by distance

```python
from tangermeme.annotate import pairwise_annotations_spacing

sp = pairwise_annotations_spacing(df4, max_distance=100)   # (n_motifs, n_motifs, max_distance)
```

- Input is the `(n, 4)` form `(example_idx, annotation_idx, start, end)`. Row order
  doesn't matter — the function groups by `example_idx` and reads pair direction off
  `start` — but pass it as a **single DataFrame** so each motif index stays glued to
  its own coordinates; sorting the columns separately scrambles the pairing.
- Distance is `start_of_right − end_of_left`, capped at `max_distance` (default
  100 — raise for long-range). Memory grows as `n_motifs² × max_distance`.
- **Annotations that overlap within an example raise a `ValueError`** — the gap
  would be negative and there is no meaningful bin for it. Abutting spans
  (`end0 == start1`, distance 0) are fine. Drop or merge overlaps first: FIMO hits
  from a redundant motif DB routinely overlap, and so do `recursive_seqlets` spans
  once you set `additional_flanks` (the un-flanked cores are disjoint).
- **Default `dtype` is `uint8`** — the same 255 saturation as `count_annotations`.
  Pass `dtype=torch.int32` for anything but small runs.

## The discovery pipeline

attributions → [seqlets.md](seqlets.md) (`recursive_seqlets`) →
`annotate_seqlets` → `count_annotations` / `pairwise_annotations` /
`pairwise_annotations_spacing`. For a method-selection view (TF-MoDISco vs
marginalize vs seqlet+TOMTOM vs FIMO), see the "Inspecting what cis-regulatory
features a model has learned" vignette.

## Related references

[seqlets.md](seqlets.md) (producing the spans), [io-loci.md](io-loci.md)
(`read_meme`), [motif-effects.md](motif-effects.md) (the `*_annotations`
perturbation variants), [plot.md](plot.md) (drawing annotations on logos).
