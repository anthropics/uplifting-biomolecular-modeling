# Cartesian product over inputs (apply_pairwise / apply_product)

`tangermeme.product` applies any `func` across combinations of `X` and additional
inputs — e.g. running a single-cell model over (sequence × cell-state × read-depth).
Both take `func` as a **required first positional argument**, and both need `args`
(`apply_pairwise` nominally defaults it to `None`, but `None` raises).

```python
from tangermeme.product import apply_pairwise, apply_product
from tangermeme.predict import predict

apply_pairwise(predict, model, X, args=(cell_states, read_depths))
apply_product(predict, model, X, args=(cell_states, read_depths))
```

## pairwise vs product is a CORRECTNESS choice, not a perf choice

- **`apply_pairwise`** zips the args *with each other* — row `j` of every arg belongs
  together (e.g. the cell state and read depth measured from the *same* cell) — and
  crosses that paired list with **every** example: `(X_i, (a_j, b_j))` for all `i, j`.
  `X` is not zipped in, so `len(X)` and `len(args[0])` are independent; all args must
  be the same length as each other. Output gains **one** axis, length =
  `len(args[0])`, giving `(len(X), len(args[0]), ...)`.
- **`apply_product`** takes the full Cartesian product: every `(X_i, a_j, b_k)`.
  Output gains **one axis per arg**, giving `(len(X), len(a), len(b), ...)`.

Using `apply_product` on metadata that is actually paired is a **silent scientific
bug** — it fabricates examples where the read depth and cell state come from
different cells. Reach for product only when the axes are genuinely independent.

## Output indexing trap

The extra axes mean a batch-of-one arg still adds a length-1 axis. To recover plain
`predict`-shaped output you must index it away:

```python
y = apply_pairwise(predict, model, X, args=(a,))[:, 0]        # one paired arg
y = apply_product(predict, model, X, args=(a, b))[:, 0, 0]    # two product args
```

## Composing and nesting

**Product on the outside** — any `func` works: `predict`, `deep_lift_shap`, even
`marginalize` (which returns its `(y_before, y_after)` pair through the product).
Route the *inner* function through `additional_func_kwargs` so the outer product
doesn't intercept it:

```python
apply_pairwise(marginalize, model, X, args=(cell_states,),
               additional_func_kwargs={'func': deep_lift_shap}, motif="TGACGTCA")
```

**Product on the inside** — because `func` is positional here, these two do **not**
satisfy the `func=` contract themselves. Passing `apply_pairwise` straight to
`marginalize(..., func=...)` misaligns every positional argument and raises
`TypeError: apply_pairwise() missing 1 required positional argument: 'X'`. Bind the
inner `func` first:

```python
from functools import partial
marginalize(model, X, motif, func=partial(apply_pairwise, predict),
            args=(cell_states,))
```

Batches are built iteratively, so the full product is never materialized in memory.

## Related references

[func-pattern.md](func-pattern.md) (the `func`/`additional_func_kwargs` contract),
[model-wrapping.md](model-wrapping.md) (passing per-example extra inputs).
