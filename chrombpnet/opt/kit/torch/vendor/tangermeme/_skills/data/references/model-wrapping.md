# Wrapping models for tangermeme

tangermeme is deliberately assumption-free, which pushes one requirement onto
you: most analysis functions assume a model whose forward returns a **single
tensor**. Wrapping is the productivity hack that makes everything else work.

## The contracts

- **Input layout:** `X` is `(batch, len(alphabet), length)` — default
  `(batch, 4, length)` for DNA. Channels are the one-hot alphabet axis.
- **Forward:** `y = model(X)` must work; broadcasting is supported where possible.
- **Single tensor vs list:** `predict` tolerates a model that returns a *list* of
  tensors (multi-output) and returns a list back. But `deep_lift_shap`,
  `saturation_mutagenesis`, and `design` need a **single tensor** — wrap to pick
  one head before calling them.
- **target= vs wrapping:** if multiple outputs live in *one tensor*
  `(batch, n_tasks)`, select with `target=`. If they are *separate tensors* in a
  list, you must wrap to return just one.

## Recipe: select one output head

```python
import torch

class TaskWrapper(torch.nn.Module):
    def __init__(self, model, task):
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, X, *args):
        y = self.model(X, *args)          # may be a list or a multi-task tensor
        return y[self.task]               # -> single tensor

attr = deep_lift_shap(TaskWrapper(model, 0), X, target=0)
```

## Recipe: profile head -> scalar (e.g. BPNet counts)

A common pattern is summing a profile to a total-count scalar, or cropping to a
center window, so attribution/design has a single target to optimize:

```python
class CountWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, *args):
        profile, counts = self.model(X, *args)   # BPNet-style two heads
        return profile.sum(dim=-1, keepdim=True) # or return counts
```

## More recipes (from the "Wrappers are Productivity Hacks" vignette)

- **Reverse-complement averaging** — average predictions on `X` and its RC
  (`torch.flip(X, dims=(-1, -2))`, flipping both length and the ACGT axis).
  **Footgun:** this silently mishandles **stranded** outputs — those must also be
  flipped along the output axis. It's only correct as-is for strand-agnostic outputs
  (e.g. counts).
- **Bake in a control track** — wrap a model that needs a control input so it
  supplies an all-zeros control automatically, removing the need to thread `args=`.
- **Concatenate inputs the function won't split** — when a function accepts only one
  tensor (no `args=`), wrap the model to take a single concatenated tensor (e.g.
  4-channel sequence + 2-channel control = 6 channels) and split it inside `forward`.
- **Fix a non-standard layout** — some models (e.g. `enformer_pytorch`) expect
  length-first input; wrap with `X.permute(0, 2, 1)`. Same place to do dict-indexed
  outputs (`y['human']`) or bin-summing (`y.sum(dim=-2)`).
- **Harmonize length across models** — pad or trim wrappers let you compare models
  with different input windows on the same loci (also see
  [comparing-models.md](comparing-models.md)). Padding with N pushes sequences
  out-of-distribution if the model wasn't trained on N; trimming discards flanks.
- **Squish N models into one** — a wrapper that runs several models and concatenates
  their outputs feeds straight into `marginalize` / `saturation_mutagenesis` as a
  single model. Wrappers stack: `CountWrapper(ControlWrapper(model))`.

## Multi-input models: use `args=`, don't bake inputs into the wrapper

Extra model inputs (cell-state vectors, control tracks, etc.) flow through every
tangermeme function via `args=(tensor, ...)`. The i-th example is paired with the
i-th row of each arg, so each arg's batch dimension must match `X`. The wrapper's
forward should accept `*args` and forward them:

```python
y = predict(model, X, args=(controls,), batch_size=64)
```

## What you do NOT need to handle

`predict`, `deep_lift_shap`, `pisa`, `saturation_mutagenesis`, `design.*`, and
`product.*` already move the model to `device`, switch to eval mode, run without
gradients where appropriate, and restore the model's original device + training
mode afterward. Your wrapper only needs a correct `forward`.

## Related references

[func-pattern.md](func-pattern.md) (how wrapped models flow through
perturbations), [deep_lift_shap.md](deep_lift_shap.md) and
[saturation_mutagenesis.md](saturation_mutagenesis.md) (why a single-tensor
forward is required there).
