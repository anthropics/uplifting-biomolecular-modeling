# The `func=` plug-point in tangermeme

tangermeme's perturbation functions are built around one contract so any analysis
can be swapped for any other. Master this and the rest of the library composes.

## The contract

Any function passed as `func=` must satisfy:

```python
func(model, X, args=None, **kwargs) -> torch.Tensor | list[torch.Tensor]
```

Three library functions already satisfy it and can be passed directly:

- `tangermeme.predict.predict` — the default; returns model predictions.
- `tangermeme.deep_lift_shap.deep_lift_shap` — returns attributions.
- `tangermeme.saturation_mutagenesis.saturation_mutagenesis` — returns ISM scores.

## Where `func=` is accepted

```python
ablate(model, X, start, end, func=...)
marginalize(model, X, motif, func=...)
space(model, X, motifs, spacing, func=...)
variant_effect.substitution_effect(model, X, subs, func=...)
variant_effect.deletion_effect(...)   ;  variant_effect.insertion_effect(...)
product.apply_pairwise(func, model, X, ...)   # func is FIRST positional here
product.apply_product(func, model, X, ...)
```

`func` defaults to `predict` on `ablate`, `marginalize`, `space` and the three
`variant_effect` functions. On `product.apply_pairwise` / `apply_product` it is a
**required first positional argument** with no default — and because of that
position they do **not** themselves satisfy the contract; see
[product.md](product.md).

**Elsewhere in the library `func=` is a different contract entirely.** None of these
take the `func(model, X)` predictor contract above:

- `design.screen(func=random_one_hot)` — a *sequence generator*
  `func(shape_tuple, random_state=, **kwargs)` that produces candidates. Don't pass
  `predict`/`deep_lift_shap` here.
- `predict(func=None)` and `saturation_mutagenesis(func=None)` — a *post-processing*
  hook applied to each batch of predictions, e.g. to pick a head or apply a final
  non-linearity. In `saturation_mutagenesis` it is applied identically to the
  reference and to every perturbation, so attributions are computed on the
  post-processed values.
- `plot.plot_attributions(models, X, func=deep_lift_shap)` — the *attribution*
  function to run before plotting.

## The key idea: the return type follows `func`

The outer function returns whatever `func` returns, before and after the edit.
`marginalize` with the default `predict` gives predictions before/after; swap in
`deep_lift_shap` and you get *attributions* before/after — same call, same
`PerturbationResult(y_before, y_after)` container.

```python
from tangermeme.marginalize import marginalize
from tangermeme.deep_lift_shap import deep_lift_shap

# Effect of a motif on PREDICTIONS
y_before, y_after = marginalize(model, X, "CTCAGTGATG")

# Effect of the same motif on ATTRIBUTIONS
attr_before, attr_after = marginalize(model, X, "CTCAGTGATG", func=deep_lift_shap)
```

## The collision footgun: `additional_func_kwargs`

Extra `**kwargs` on the outer function are forwarded to `func`. This breaks when
a kwarg name means different things to the outer function and to `func` (e.g.
`random_state`, `batch_size`, `target`). Disambiguate by routing kwargs meant for
`func` through `additional_func_kwargs=`:

```python
# Tell deep_lift_shap to attribute output #3, not the marginalize-level kwargs
marginalize(model, X, motif, func=deep_lift_shap,
            additional_func_kwargs={'target': 3, 'n_shuffles': 50})
```

- Default is `None`, **not** `{}`.
- The dict is copied defensively in `ablate`, `marginalize`, `space`,
  `product.*`, `variant_effect.*`, and `design.screen` — passing it will not
  mutate your caller-side dict.
- Anything in `additional_func_kwargs` goes only to `func`; bare `**kwargs` may
  be consumed by the outer function first.

Rule of thumb: if a name could plausibly belong to either layer, put it in
`additional_func_kwargs` to be explicit.

## Multi-input models

`args=(tensor, ...)` carries extra model inputs through the whole stack; the i-th
example is paired with the i-th row of each arg. It flows from the outer function
into `func` into `model(X, *args)`.

`product.apply_pairwise` / `apply_product` are the exception: there `args` defines
the axes to sweep over rather than per-example inputs, so its rows are *not* aligned
to `X` (see [product.md](product.md)).

## Related references

[motif-effects.md](motif-effects.md) for marginalize/ablate/space specifics,
[deep_lift_shap.md](deep_lift_shap.md) and
[saturation_mutagenesis.md](saturation_mutagenesis.md) for the attribution
methods, [model-wrapping.md](model-wrapping.md) for adapting multi-output models.
