# DeepLIFT/SHAP attributions in tangermeme

`tangermeme.deep_lift_shap.deep_lift_shap` runs a model "backwards" to score the
contribution of each input base to an output. It corrects several issues found in
other DeepLIFT/SHAP implementations and supports custom non-linearities.

## Signature (defaults that matter)

```python
deep_lift_shap(
    model, X,
    args=None,
    target=0,                       # WHICH output — see below
    batch_size=32,                  # example-REFERENCE pairs, not examples — see below
    references=dinucleotide_shuffle,# callable OR a precomputed tensor
    n_shuffles=20,                  # references per example when references is a fn
    return_references=False,
    hypothetical=False,
    warning_threshold=0.001,        # additivity (convergence-delta) check
    additional_nonlinear_ops=None,
    print_convergence_deltas=False,
    raw_outputs=False,
    only_warn=False,
    dtype=None, device=None,        # device None -> CUDA if available else CPU
    random_state=None,
    verbose=False,
)
```

## Footgun #1 — `target=` is mandatory for multi-task models

`target` defaults to `0`. If your model emits multiple outputs in one tensor and
you forget `target=`, you will silently get attributions for output 0, not an
error. Always set it explicitly for multi-task models:

```python
X_attr = deep_lift_shap(model, X, target=267, random_state=0)
```

If the model returns *multiple tensors* (a list), `target` cannot select among
them — wrap the model so its forward returns a single tensor first (see
[model-wrapping.md](model-wrapping.md)).

## Footgun #2 — reproducibility needs `random_state=`

The default `references=dinucleotide_shuffle` draws `n_shuffles` random
backgrounds per example, so attributions vary run to run. Pass `random_state=` to
make them deterministic (required when capturing regression values). For comparing
attributions **across tasks or models**, holding `random_state` *and* `n_shuffles`
fixed is mandatory — otherwise each call attributes against different backgrounds
and the differences you see are noise, not signal (see
[comparing-models.md](comparing-models.md)). A custom `references=` callable has the
signature `f(X, n, random_state) -> (n_examples, n, len(alphabet), length)`; if you
only shuffle a sub-span, attributions are exactly `0` outside it — don't interpret
those positions.

## Footgun #3 — `only_warn=True` still raises inside shuffling

`only_warn=True` downgrades validation errors on `X`, but the internal
`dinucleotide_shuffle` re-validates and will still raise on a malformed `X`.
Workaround: precompute references from a valid one-hot input and pass them in. Note
the standalone `dinucleotide_shuffle` parameter is `n=` (the `n_shuffles` name is
`deep_lift_shap`'s own), and a precomputed `references=` tensor must have shape
`(batch, n_shuffles, len(alphabet), length)`:

```python
refs = dinucleotide_shuffle(X_valid, n=20, random_state=0)  # (batch, 20, 4, length)
X_attr = deep_lift_shap(model, X, references=refs, only_warn=True)
```

## `batch_size` counts example-reference pairs

`batch_size` is the number of example×reference pairs run at once, not the number
of examples. With the default `n_shuffles=20`, a single example already expands to
20 forward/backward passes, so `batch_size=32` does **not** mean 32 examples in
flight. If you hit OOM, lower `batch_size` (or `n_shuffles`); the attributions are
unchanged, only the per-step memory differs.

## Convergence deltas — the correctness signal (read this)

DeepLIFT/SHAP has an additive property: per-example attributions should sum to the
prediction difference between the sequence and its references. The convergence
delta measures the violation, and its **magnitude tells you the cause**:

- **delta > ~0.01**: a non-linearity the hooks don't handle is unregistered — a
  real **correctness bug**. Fix it (below), don't ship the attributions.
- **delta ~1e-3 to 1e-5 (often 1e-7 on CPU)**: benign floating-point error.
  Architecture-dependent; safe to ignore.

tangermeme auto-warns above `warning_threshold`; use `print_convergence_deltas=True`
to see per example-reference-pair values.

**High deltas do NOT produce garbage logos — this is the dangerous part.** A model
with an unregistered op can still highlight motif-shaped patterns that look real but
*vanish* once the op is registered. Motif-shaped ≠ correct; only low deltas are.
Cross-check a suspicious logo against the actual prediction.

### Registering a custom non-linearity

```python
from tangermeme.deep_lift_shap import _nonlinear
X_attr = deep_lift_shap(model, X, additional_nonlinear_ops={MyActivation: _nonlinear})
```

The built-in hooks already cover essentially every stock `torch.nn` elementwise
activation — the ReLU family (`ReLU`, `ReLU6`, `LeakyReLU`, `RReLU`, `PReLU`),
`GELU`/`SiLU`/`Mish`/`ELU`/`CELU`/`SELU`, `Sigmoid`/`LogSigmoid`/`Tanh`,
`Softplus`/`Softshrink`, `GLU` — plus `Softmax` and `MaxPool1d`/`MaxPool2d`
(`pisa` registers the same set). Check that list before assuming you need
`additional_nonlinear_ops`; it is for **custom** modules. The catch:
`_nonlinear` divides `delta_out / delta_in`, so it **must be registered on a layer
with equal input and output shape**. If your op also reduces (e.g. a profile head
that does `logits * softmax(logits)` then `.sum()`), split it: put the elementwise,
shape-preserving part in its own `nn.Module`, register *that*, and do the reduction
in the parent wrapper. Registering the reducing layer raises a size-mismatch error.

### Precision: CPU vs CUDA, and the fp64 escape hatch

The same model gives **higher deltas on CUDA than CPU** (parallel reductions reorder
float sums) — thresholds tuned on CPU may need raising on GPU. To disambiguate
"unregistered op" from "precision noise," re-run a few examples on CPU. For
genuinely precision-driven deltas, `deep_lift_shap(model.double(), X.double(),
references=refs.double(), ...)` drops them to ~1e-16 (slower; fp64).

When you can't get deltas down (an op that can't be expressed as a shape-preserving
hook), switch to ISM — see [saturation_mutagenesis.md](saturation_mutagenesis.md).

## hypothetical vs projected attributions

- Default (`hypothetical=False`): attributions are projected onto the observed
  bases — what you plot as a logo of the actual sequence.
- `hypothetical=True`: per-base hypothetical contributions across all four bases.
  Use these to build motif patterns / contribution-weight matrices (CWMs)
  downstream — **not** as the seqlet-caller input.

The seqlet callers take the projected output collapsed over the channel axis
(`X_attr.sum(dim=1)`) — see [seqlets.md](seqlets.md).

## Return type

Returns a tensor shaped like `X` (e.g. `(batch, 4, length)`). With
`return_references=True` it returns an `AttributionReferencesResult(attributions,
references)` NamedTuple — unpack positionally or by attribute.

## Plotting

```python
from tangermeme.plot import plot_logo
plot_logo(X_attr[0], ax=ax)   # tangermeme has its own logo plotting (no logomaker)
```

## Composing with perturbations

`deep_lift_shap` satisfies the `func=` contract, so it drops into `marginalize`,
`ablate`, `variant_effect.*`, etc. to get attributions before/after an edit —
route attribution kwargs via `additional_func_kwargs` (see
[func-pattern.md](func-pattern.md)).

## Related references

[saturation_mutagenesis.md](saturation_mutagenesis.md) (the forward-pass
alternative), [seqlets.md](seqlets.md) (consuming projected attributions),
[model-wrapping.md](model-wrapping.md) (single-tensor requirement),
[comparing-models.md](comparing-models.md) (shared references across models).
