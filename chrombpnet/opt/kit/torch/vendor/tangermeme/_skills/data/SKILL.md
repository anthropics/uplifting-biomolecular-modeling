---
name: tangermeme
description: Use for any task involving the tangermeme library — post-training analysis of genomic sequence-to-function (S2F) deep learning models. Triggers on predictions, DeepLIFT/SHAP attributions, marginalization/ablation/spacing of motifs, saturation mutagenesis (ISM), variant effect scoring, sequence design, seqlet calling, loading loci/FASTA/bigWig/MEME/VCF, or wrapping a PyTorch genomic model for these analyses. This is a router skill — read the relevant file under references/ for details and footguns before writing tangermeme code.
---

# tangermeme

`tangermeme` answers the "what did my genomic model learn, and what do I do with
it after training" question. It provides atomic sequence operations, batched
prediction, attribution, perturbation experiments, and sequence design — all
deliberately assumption-free (any PyTorch model, any alphabet, raw outputs
returned rather than distances).

This skill is a **router**. Each topic below has a detailed reference file with
the exact signatures and footguns. **Read the relevant reference file before 
writing code** — do not rely on memory of the API, because several functions 
have non-obvious defaults (silent wrong-output selection, variable-length returns, 
reproducibility traps).

## Two cross-cutting concepts (read these first if unsure)

- **[The `func=` plug-point](references/func-pattern.md)** — `ablate`,
  `marginalize`, `space`, `variant_effect.*`, and `product.*` (where it is the
  first positional argument) all accept `func(model, X, args=, **kwargs)`.
  Swapping `predict` for
  `deep_lift_shap` turns a "predictions before/after" experiment into an
  "attributions before/after" one. Covers the `additional_func_kwargs` collision
  trap. **This is what makes the library compose.**

- **[Wrapping models](references/model-wrapping.md)** — tangermeme assumes
  `y = model(X)` returns a single tensor with layout `(batch, channels, length)`.
  Real multi-input / multi-output models must be wrapped first. Read this before
  attribution or design on any non-trivial model. Data preprocessing or output
  post-processing should be handled in custom wrappers rather than in custom
  functions.

## Task → reference file

| If the task is… | Read |
|---|---|
| **starting from scratch** — set up a notebook to load a model and run predictions → attributions → seqlets → motif tests, end to end | [references/notebook-walkthrough.md](references/notebook-walkthrough.md) |
| attribution via **DeepLIFT/SHAP** — "which bases drive this prediction", attribution logos, hypothetical contributions for CWMs | [references/deep_lift_shap.md](references/deep_lift_shap.md) |
| attribution via **ISM / saturation mutagenesis** — the forward-pass alternative; use it when DeepLIFT/SHAP convergence deltas are too high, an op can't be registered, or the model is massively multi-task | [references/saturation_mutagenesis.md](references/saturation_mutagenesis.md) |
| comparing predictions/attributions **across N models** (replicates, architectures, ensembles) | [references/comparing-models.md](references/comparing-models.md) |
| effect of a motif / region: marginalize, ablate, spacing between motifs | [references/motif-effects.md](references/motif-effects.md) |
| scoring **variant effects** (substitution / deletion / insertion, from a VCF) | [references/variant-effect.md](references/variant-effect.md) |
| **calling seqlets** from attributions (recursive / TF-MoDISco) | [references/seqlets.md](references/seqlets.md) |
| **annotating / counting motifs** — TOMTOM/FIMO labels, co-occurrence, spacing | [references/annotate.md](references/annotate.md) |
| running a function over a **product of inputs** (sequence × cell-state × …) | [references/product.md](references/product.md) |
| **plotting logos** and drawing seqlet/motif annotations on them | [references/plot.md](references/plot.md) |
| composing predict / deep_lift_shap / saturation_mutagenesis through a perturbation fn | [references/func-pattern.md](references/func-pattern.md) |
| adapting a multi-input/output PyTorch model to the tangermeme contract | [references/model-wrapping.md](references/model-wrapping.md) |
| loading sequences/signals at loci, reading FASTA/bigWig/BED/MEME/VCF | [references/io-loci.md](references/io-loci.md) |
| designing sequences to hit a target output (screen / greedy / beam substitution) | [references/design.md](references/design.md) |

## Quick orientation (the rest of the library)

These are well-covered by the official tutorials and are mostly single-call ops —
no dedicated reference file, but here is where to look:

- `tangermeme.predict.predict` — batched, memory-efficient inference. Returns the
  model's parameter dtype (override with `dtype=`) and upcasts each batch from `X`,
  so int8 sequences go straight in; multi-output models return a list. Satisfies the
  `func=` contract.
- `tangermeme.ersatz` — atomic sequence ops: `insert`, `substitute`,
  `multisubstitute`, `delete`, `shuffle`, `dinucleotide_shuffle` (most motif-add
  ops *substitute*, preserving length; `start`/`end` confine shuffles to a region).
- `tangermeme.utils` — `one_hot_encode` (returns int8), `characters`,
  `random_one_hot`, `reverse_complement` (single sequence), `pwm_consensus`,
  `set_seed`, `gc_content`, etc.
- `tangermeme.pisa.pisa` — per-position (PISA) attribution reusing the DLS hooks.
  Footguns: some paths return tensors on the **input device**, not CPU; and it does
  **not** upcast `X`, so pass float — it is the one entry point int8 fails on.
- `tangermeme.kmers` — k-mer counts, batched k-mers, gapped k-mers.

(Motif *scanning* — FIMO/TOMTOM — is not in tangermeme; it lives in the external
`memelite` / memesuite-lite package. `annotate_seqlets` wraps TOMTOM internally.)

## Conventions used throughout

- Tensor layout: `(batch, channels, length)`; channels = one-hot alphabet axis
  (default `['A','C','G','T']`).
- Variable naming: `X`/`y` observed, `X_bar`/`y_bar` designed/target,
  `y_hat` predictions, `X_attr` attributions.
- Device defaults: `predict`, `deep_lift_shap`, `pisa`, `design.*`,
  `saturation_mutagenesis`, `product.*` take `device=None` → CUDA if available
  else CPU; the model's original device + training mode are restored afterward.
- Perturbation functions return NamedTuples (`PerturbationResult`, etc.) — unpack
  positionally or by attribute; `isinstance(result, tuple)` is True.
