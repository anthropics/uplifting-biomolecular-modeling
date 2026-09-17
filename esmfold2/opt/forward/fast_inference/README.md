# fast_inference — the ESMFold2 kit's lever modules

Vendored in-process levers for the Biohub ESMFold2 implementation (`transformers` fork 4.57.6 @ `ef32577f`, `esm` 3.3.0 @ `26b0bc2b`, torch 2.13 /
Triton 3.7). Nothing in the installed packages, the weights or the fold call is modified: every lever is a method wrapper or a Triton kernel installed
on a loaded model by `driver/ef2_server.py: configure(model, mode, builder)`, which the package `esmfold2_opt` calls before the model's first fold
(`esmfold2/CHANGES.md` maps the kit's modes `exact` / `fast` / `big` onto the two lever sets below).

| file | what it holds |
|---|---|
| `driver/ef2_server.py` | `MODES` (the two lever sets: `opt7x` = bitwise, `opt14_msa` = tolerance class), `configure()` (install order ef2_mk_sampler → ef2_msa → ef2_w4 → ef2_opt.install: graph capture last), `write_outputs()` (one prediction's mmCIF / PAE-pLDDT npz / score row — upstream's own outputs, nothing derived) |
| `driver/ef2_opt.py` | CUDA-graph capture and replay of the pair trunk, coda, confidence trunk, LM and MSA encoders (`tg`, `eg`), the sync-free graphed diffusion sampler (`sg`), the per-input ESMC / feature / pair-bias caches (`ec`, `fc`, `pb`), the fused TriMul inside the MSA encoder (`msa`); the per-shape graph budget `EF2_GRAPH_BUDGET_TOKENS` (larger inputs run the same kernels eagerly) |
| `driver/ef2_w4.py` | pair-trunk kernels: `t3` tile table, `t5` cached bf16 weight casts, `t6` / `t10` fused pair transition (`t1` the small-shared-memory substitute), `tx` the pair TriMul bound to the shared core's TriMul provider (`opt_core.kernels.trimul`) under the mode's tier word, the provider's cell table naming the row per card, stack and size; a first-call probe prints whether each kernel is bit-identical or tolerance-class on the running stack (`EF2_W4_IDENTITY_PROBE=0` turns it off) |
| `driver/ef2_mk_sampler.py` | `mk`: the diffusion sampler's step-invariant conditioning hoisted out of the step loop (same ops; one diffusion sample per call) |
| `driver/ef2_msa.py` | `t11`–`t14`: MSA-module kernels (transition, outer-product mean epilogue, pair-weighted averaging, bf16-input LayerNorm) for the Full model |
| `driver/run_ef2_om.py` | constants and the score-row field reader `configure()` / `write_outputs()` use |
| `tests/w4_public_slice.json` | public example inputs (PDB 1BRS barnase / barstar, upstream's prediction-input schema) — the kit README's example and `run.sh warm`'s fold |
| `upstream/` | execution-only patches proposed to the upstream fork (U1–U6, `.diff` + note); not applied by the kit |
| `HAZARDS.md` | what can silently change a kernel's numerics class or engagement (cells, probes, shared-memory limits, graph order) |
| `NOTICE` | third-party attributions: the modified Biohub kernels, the transition kernel adapted from the support library's kernels, the public test input |

The `EF2_*` environment names read inside these modules are their internal switches; the package sets the ones a mode needs (`EF2_MK`, `EF2_MSA`,
`EF2_GRAPH_BUDGET_TOKENS`, `EF2_W4_IDENTITY_PROBE`) and drops caller-set lever switches so the mode table stays the source of truth. Every lever
prints one line when it engages, substitutes or stays off, and `configure()` returns the description string the package records on its APPLIED line.
