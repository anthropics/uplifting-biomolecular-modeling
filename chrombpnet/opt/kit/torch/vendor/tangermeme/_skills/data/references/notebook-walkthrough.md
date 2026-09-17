# A starter notebook: model → predictions → attributions → seqlets

This is a runnable scaffold for the most common tangermeme workflow: take a
trained sequence-to-function model and a set of peaks, sanity-check its
predictions, attribute, call seqlets, and test motif hypotheses. Paste the cells
into a notebook and edit the paths / model loading for your setup.

It is a **spine**, not a tutorial — each step links to the reference file with the
exact signatures and footguns. Read those before relying on a step.

Assumed layout: a one-hot input of shape `(batch, 4, length)`, a model where
`y = model(X)` returns a single tensor (wrap yours first if not — see
[model-wrapping.md](model-wrapping.md)).

```python
# === Cell 1: setup ===
import torch
import numpy
import matplotlib.pyplot as plt

from tangermeme.utils import set_seed

set_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# === Cell 2: load the model ===
# Load your own trained model however you normally do; it just needs eval-able
# forward where y = model(X). tangermeme moves it to `device` and restores its
# original device + train/eval state after each call, so no .to()/.eval() needed.
model = torch.load("model.pt", weights_only=False)

# If the model is multi-output (returns a list) or multi-input, wrap it so that
# forward returns a single tensor before attribution/design. See model-wrapping.md.
# class Wrapper(torch.nn.Module):
#     def __init__(self, model): super().__init__(); self.model = model
#     def forward(self, X, *args): return self.model(X, *args)[0]
# model = Wrapper(model)
```

```python
# === Cell 3: load peaks + sequence (and observed signal) ===
# extract_loci returns a VARIABLE number of objects depending on which kwargs are
# set — here (sequences + signals) it returns two. See io-loci.md for the full
# return-order rules and the counts/exclusion filters.
from tangermeme.io import extract_loci

X, y = extract_loci(
    "peaks.narrowPeak",          # BED / narrowPeak path or a pandas DataFrame
    "genome.fa",                 # FASTA path or pyfaidx.Fasta
    signals=["signal.bw"],       # observed bigWig(s) -> y
    in_window=2114,              # match your model's input length
    out_window=1000,             # match your model's output length
)
# X comes back as int8. Leave it — predict / deep_lift_shap /
# saturation_mutagenesis upcast each batch to the model's dtype, so keeping the
# full tensor as int8 is the memory win. (pisa is the exception: pass X.float().)
print(X.shape, y.shape)
```

```python
# === Cell 4: predictions on the peaks, vs observed ===
# predict batches to `device` and back, runs under eval()+no_grad, and returns the
# model's parameter dtype (float32 for a float32 model; override with dtype=).
# Multi-output models return a list. See the predict bullet in SKILL.md.
from tangermeme.predict import predict

y_hat = predict(model, X, batch_size=64, device=device)

plt.scatter(y.sum(dim=-1).log1p(), y_hat.sum(dim=-1).log1p(), s=2, alpha=0.3)
plt.xlabel("observed (log total counts)")
plt.ylabel("predicted (log total counts)")
plt.gca().spines[["top", "right"]].set_visible(False)
plt.show()
```

```python
# === Cell 5: attributions at loci of interest ===
# Pick a few loci to inspect (here the highest-signal peaks). For a multi-task
# model you MUST pass target= or you silently attribute output 0. random_state=
# makes the shuffled references reproducible. See deep_lift_shap.md.
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.plot import plot_logo

idx = y.sum(dim=-1).argsort(descending=True)[:4]
X_attr = deep_lift_shap(model, X[idx], target=0, random_state=0, device=device)

fig, axes = plt.subplots(len(idx), 1, figsize=(12, 2 * len(idx)))
for ax, attr in zip(axes, X_attr):
    plot_logo(attr[:, 900:1200], ax=ax)   # zoom into the center
plt.tight_layout()
plt.show()
```

```python
# === Cell 6 (optional): ISM as a cross-check / fallback ===
# Use ISM if DeepLIFT/SHAP convergence deltas are high, an op can't be registered,
# or the model is massively multi-task. Restrict to a window — cost scales with
# length. See saturation_mutagenesis.md.
from tangermeme.saturation_mutagenesis import saturation_mutagenesis

X_ism = saturation_mutagenesis(model, X[idx], start=900, end=1200, target=0,
    device=device)
plot_logo(X_ism[0], ax=plt.gca())
plt.show()
```

```python
# === Cell 7: call seqlets from the attributions ===
# recursive_seqlets takes per-position PROJECTED attribution sums, shape
# (n, length) — collapse the channel axis. Returns a DataFrame sorted by p-value
# with columns example_idx / start / end / attribution / p-value. See seqlets.md;
# label/count the seqlets with annotate.md, draw them with plot.md.
from tangermeme.seqlet import recursive_seqlets

# Attribute the whole batch first (here just the few loci; scale up as needed).
X_attr_all = deep_lift_shap(model, X[idx], target=0, random_state=0, device=device)
attr_sums = X_attr_all.sum(dim=1)            # (n, length)

seqlets = recursive_seqlets(attr_sums, threshold=0.01)
print(seqlets.head())
```

```python
# === Cell 8: test a motif hypothesis with marginalization ===
# Did a motif you found in the seqlets actually drive the model? Insert it into
# shuffled backgrounds and measure the delta. See motif-effects.md.
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.marginalize import marginalize

X_bg = dinucleotide_shuffle(X[:64], random_state=0)[:, 0]   # one shuffle each
y_before, y_after = marginalize(model, X_bg, "CTCAGTGATG", device=device)

delta = (y_after.sum(dim=-1) - y_before.sum(dim=-1))
print("mean marginal effect:", delta.mean().item())
```

## Where to go next

- **Multiple models / comparisons** — loop the same `predict` / `deep_lift_shap`
  calls over a list of models; the device/state handling is per-call and safe. To
  keep the results *comparable* (shared references, output harmonization, scale
  normalization, memory), see [comparing-models.md](comparing-models.md).
- **Variant effects** — `tangermeme.variant_effect.*` with variants from
  `io.read_vcf` (see [variant-effect.md](variant-effect.md)).
- **Spacing / cooperativity** — `tangermeme.space.space` (see motif-effects.md).
- **Sequence design** — [design.md](design.md).
- **Any perturbation as attributions** — pass `func=deep_lift_shap` into
  marginalize/ablate/etc. ([func-pattern.md](func-pattern.md)).

Each step above links to its deep-dive file inline; see those for the footguns.
