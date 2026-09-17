# Plotting logos and annotations (tangermeme.plot)

`plot_logo` draws attribution/PWM logos (it replaces logomaker). The function call
itself is simple; the **annotation subsystem** is where the non-obvious behavior
lives, and it is the connective tissue between seqlets/FIMO and the figure.

```python
from tangermeme.plot import plot_logo

ax = plot_logo(X_attr[i], ax=ax, start=900, end=1200)
```

- Pass `X_attr` of shape `(len(alphabet), length)` (one example). Positive
  characters stack up, negative stack down, ordered by magnitude.
- For a **probability PWM** (columns sum to 1), use `plot_pwm` instead — it draws
  the information-content-weighted logo. `plot_logo` is for attributions/weights.
- It draws **only the logo panel** — add titles/labels/limits with normal matplotlib
  afterward. Without `ax=` it falls back to `plt.gca()`, i.e. the *current* axes, not
  a fresh one — so pass `ax=` explicitly when looping over examples or you will stack
  every logo onto one panel.
- `start`/`end` subset the plotted window **and** are required for annotation
  coordinates to line up, because annotations use absolute coordinates.

## Annotations contract

`annotations=` is a pandas DataFrame with columns `motif_name`, `start`, `end`,
`strand` (currently unused), `score`. Coordinates are 0-indexed. Extra columns are
ignored. Three non-obvious rules:

- **The label is read positionally, as the first column** (`row.values[0]`), not from
  a column named `motif_name`. FIMO frames already lead with `motif_name`; seqlet
  frames lead with `example_idx`, so reorder those (below).
- **`score` is only rendered into the label** (`show_score=False` hides it).
  Annotations are packed into rows in **`start` order**, not score order, so a
  high-scoring hit gets no priority for the top track.
- **Window filtering is strict**: a row is drawn only if `row.start > start` **and**
  `row.end < end`. A hit flush with either edge is silently dropped — including one
  at position 0 when you didn't pass `start` (which defaults to 0). Widen the window
  by a base if you need the edges.

```python
plot_logo(X_attr[i], ax=ax, annotations=hits, start=900, end=1200)
```

### Seqlet footgun: set score_key, filter per example, put a name column first

Seqlet DataFrames use the column `attribution`, not `score`, so pass
`score_key='attribution'`. They hold **all examples in one frame** — filter to the
example you're plotting first, or you'll draw another example's coordinates. And
their first column is `example_idx`, so passing one straight through labels every
annotation with the example index; insert the motif name (or any label) as the
first column:

```python
# names / motif_idxs from annotate_seqlets — see annotate.md. recursive_seqlets
# returns a 0..n-1 index aligned with the motif_idxs rows, so s.index maps across.
s = seqlets[seqlets['example_idx'] == i].copy()
s.insert(0, 'motif_name', [names[j] for j in motif_idxs[s.index, 0]])
plot_logo(X_attr[i], ax=ax, annotations=s, score_key='attribution',
          start=900, end=1200)
```

## Three interoperable annotation sources

The same `annotations=` arg accepts: a hand-built DataFrame, FIMO hits (from the
external `memelite` package — `fimo(motifs, X, dim=1)[i]` gives the i-th sequence's
hits, already in `motif_name`/`start`/`end`/`score` form), and seqlets from
`recursive_seqlets` / `tfmodisco_seqlets`.

## Track packing

Overlapping annotations are greedily packed into rows; past `n_tracks` (default 4)
they collapse into a compact gray name-only strip. `show_extra=False` hides the
overflow; raise `n_tracks` to show more. `ylim=` fixes the y-axis for comparison.

`color=` controls glyph coloring and accepts several forms: a single color string
(uniform, handy for cross-model overlays), a `{char: color}` dict (per character),
or an **array-like the length of the sequence** (per position). A per-position
array can hold color specs (names/hex/RGB(A)) used verbatim, or numeric values
mapped through `color_cmap` with optional `color_vmin`/`color_vmax`. It is sliced
alongside `X_attr`; a length mismatch warns and falls back to per-character color.

## The other entry points

- `plot_attributions(models, X, func=deep_lift_shap, attribute_kwargs=,
  plot_kwargs=, layout=)` — attribute *and* plot in one call, over one or more
  models × sequences. Its `func` is the attribution function, not the `func(model,
  X)` plug-point (see [func-pattern.md](func-pattern.md)).
- `interactive_logo` — the `plot_logo` counterpart with hover tooltips listing every
  column of the annotation. Needs the optional `interactive` extra (`mpld3`). Worth
  it only when each annotation carries several useful fields.
- `plot_pwm` for probability PWMs, as above.

## Related references

[seqlets.md](seqlets.md) and [annotate.md](annotate.md) (producing annotations),
[deep_lift_shap.md](deep_lift_shap.md) (producing `X_attr`).
