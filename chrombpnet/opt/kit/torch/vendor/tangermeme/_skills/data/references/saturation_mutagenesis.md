# Saturation mutagenesis (ISM) in tangermeme

`tangermeme.saturation_mutagenesis.saturation_mutagenesis` is an attribution
method, alongside [deep_lift_shap.md](deep_lift_shap.md). In-silico saturation
mutagenesis (ISM) mutates every position to every base and measures the change in
the model's output. It is **purely forward-pass**, so it sidesteps the gradient /
custom-backward machinery entirely.

Every forward pass goes through `predict`, so inference runs under `model.eval()`
and `torch.no_grad()` — no autograd graph, and dropout/batchnorm in eval mode. Your
model's original training mode and device are restored afterward, so you never need
to set (or reset) `.eval()` yourself.

## Signature (defaults that matter)

```python
saturation_mutagenesis(
    model, X,
    args=None,
    start=0, end=-1,                # restrict to a window — see cost note
    batch_size=32,
    target=None,                    # int | slice | None; None = average ALL tasks
    hypothetical=False,
    raw_outputs=False,
    dtype=None, device=None,        # device None -> CUDA if available else CPU
    verbose=False,
    func=None,                      # POST-PROCESSING hook, not the func= plug-point
)
```

`func=` here has the same meaning as `predict(..., func=None)`: a function applied
to each batch of predictions after the forward pass (pick a head, apply a final
non-linearity). It is applied identically to the reference prediction and to every
perturbation, so attributions are computed on the post-processed values. This is
**not** the `func(model, X)` plug-point — see [func-pattern.md](func-pattern.md).

## When to use ISM instead of DeepLIFT/SHAP

- **DeepLIFT/SHAP convergence deltas are too high** and can't be fixed with
  `additional_nonlinear_ops` — ISM has no additivity assumption to violate, so it
  is trustworthy where DLS is not.
- **the model uses an op that can't be registered** for the custom backward pass.
- **the model is massively multi-task** — ISM runs the perturbed sequences through
  the model once and reads off all outputs, so N outputs cost ~the same as 1.
  DeepLIFT/SHAP attributes one `target` at a time and scales with the number of
  outputs you want.

## The trade-off: cost scales with sequence length

ISM does one forward pass per single-base edit, so cost grows with the length of
the region being mutated (≈ `(end - start) * (len(alphabet) - 1)` perturbed
sequences per example). Restrict it to the region of interest with `start`/`end`
rather than scanning a whole long input.

Windowed ISM is **bit-identical** to running the full sequence and slicing (edits
are independent), so there is no accuracy cost — only speed (~15× faster for ~5% of
the sequence). Triage pattern: loop windowed ISM over a list of motif-hit
coordinates and sum `(X_ism * X)` over each window to rank which hits actually drive
the prediction.

## Footgun: `X` is cast to int8, so soft one-hots are truncated

The edit-distance-one kernel works on int8, so a non-integer `X` — a scaled,
softened or probability-valued one-hot — is truncated toward zero. Anything with
magnitude below 1 becomes 0, which can zero out the whole input and hand back an
all-zero attribution. It does emit a `TangermemeWarning`, but that is easy to miss
in a notebook. Pass a hard one-hot (int8, or float-valued 0/1).

## Single-tensor requirement

Like DeepLIFT/SHAP and design, ISM needs a model whose forward returns a **single
tensor**. The tensor may have multiple outputs (`(batch, n_targets)`), but the
model must not return a *list* of tensors. `target=None` (the default) averages the
attribution across **all** tasks; for a model with thousands of heterogeneous heads
(binding + expression + histone) that average is noisy and uninterpretable, so pass
an int or slice to subset to the output(s) you care about first, or wrap the model —
see [model-wrapping.md](model-wrapping.md). `target=N` is exactly equivalent to a
single-task slice wrapper.

## Return type

- Default (`raw_outputs=False`, `hypothetical=False`): an attribution tensor shaped
  like `X` (`(batch, len(alphabet), end-start)`), **projected** onto the observed
  bases. It is built from the per-edit prediction differences, mean-centered across
  the bases at each position and averaged over the selected task(s) — not a raw
  Euclidean distance. Collapse over the channel axis (`.sum(dim=1)`) for a
  per-position track. It deliberately does **not** Z-score-normalize (unlike
  TF-MoDISco-style scores): Z-scoring would flatten magnitude differences between
  regions, so you could no longer compare a strong region head-to-head with a weak
  one.
- `hypothetical=True`: the same scores across all four bases, *before* projecting
  onto the observed sequence (feeds CWM construction).
- `raw_outputs=True`: a `SaturationMutagenesisRawResult(y0, y_hat)` NamedTuple —
  `y0` the original predictions and `y_hat` the per-edit predictions — when you
  want to aggregate the effect yourself.

## Composing with perturbations

`saturation_mutagenesis` satisfies the `func=` contract, so it drops into
`marginalize`, `ablate`, `variant_effect.*`, etc. to get ISM scores before/after
an edit — route ISM kwargs via `additional_func_kwargs` (see
[func-pattern.md](func-pattern.md)).

## Plotting

```python
from tangermeme.plot import plot_logo
plot_logo(X_ism[0], ax=ax)
```

## Related references

[deep_lift_shap.md](deep_lift_shap.md) (the gradient-based attribution method),
[model-wrapping.md](model-wrapping.md), [func-pattern.md](func-pattern.md).
