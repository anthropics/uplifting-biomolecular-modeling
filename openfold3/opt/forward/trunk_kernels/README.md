# OpenFold3 trunk-kernels add-on — `of3t_hook`

An import hook (`of3t_hook/` on `PYTHONPATH`) that changes **how** OpenFold3 0.4.1's Pairformer trunk and diffusion sampler execute: same
weights, inputs, MSAs, recycles, diffusion steps, seeds and outputs. Every lever is OFF unless its environment variable is set; in this
tree the mode table (`opt/openfold3_opt/modes.py`) sets them — `exact` takes the two exact levers, `fast` the default set below — and
chains the fast-inference kit's hook behind this one (`OF3T_KIT_LEVERS=<dir of of3_levers>`).

## Levers

| env | what changes | numerics class |
|---|---|---|
| `OF3T_PAIRCACHE=1` | inside one diffusion rollout the conditioned pair representation (DiffusionConditioning over `cat[zij_trunk, relpos]` + 2 pair transitions) and the 24 per-block pair biases of the token DiffusionTransformer do not depend on the step, the noisy coordinates or the RNG; stock recomputes them at every step; the lever computes them at the first step of each rollout and reuses them (per rollout = per query × seed; never shared across seeds) | **exact**: same kernels on the same inputs, once instead of once per step; `OF3T_PAIRCACHE_MAX_GB` (default 4) caps the cache — above it the tensors are recomputed as in stock |
| `OF3T_TEMPL_DISTINCT=1` | identical template rows are embedded once | **exact** |
| `OF3T_TRIMUL=cueq` | triangle multiplicative update (trunk + MSA module + template stack) through NVIDIA cuEquivariance's fused kernel, in the fp32/TF32 precision OpenFold3 already runs | **near-exact** (kernel swap: rounding points change, not bitwise; bitwise run-to-run) |

Default set = all three (`OF3T_TEMPL_DISTINCT=1 OF3T_PAIRCACHE=1 OF3T_TRIMUL=cueq`); with `OF3T_TRIMUL` unset the numerics are stock's.
The triangle attention of every pair stack is the kit's pair cells' (the tree's core provider by the line's tier word,
`opt/openfold3_opt/of3_triattn.py`); `OF3T_TRIATT=ds|cueq|triton` only names upstream's own kernel flags for the stock statement beneath them and
no kit line sets it. Not offered: cuEquivariance's fp32 triangle *attention* / OpenFold3's own `use_cueq_triangle_kernels: true` runner option on
the fast line (its rounding on trunk tensors falls outside stock's seed-to-seed class).

Memory: the pair cache holds `zij` [S,N,N,128] fp32 + 24 × [S,16,N,N] fp32 per rollout (scales with N²), freed at the next
rollout; under CUDA graphs one static set per captured shape, refreshed eagerly at the first step of every sampling call.

## Stock configuration, composition, requirements

`config/stock_predict.yml` is the same kernels-off runner configuration the fast-inference kit ships (see its README): the PyTorch
attention path selected explicitly, so a stock comparison run neither uses a different attention kernel silently nor writes zero structures.
Composition with the fast-inference kit: this hook first on `PYTHONPATH`, `OF3T_KIT_LEVERS` naming the kit's `of3_levers`, the kit's
`OF3_FAST_INIT` / `OF3_CUDA_GRAPHS` switches as that kit documents; the pair cache finds rollout boundaries by wrapping `SampleDiffusion.forward` /
`_sample_rollout` at import and again at the first `OpenFold3.forward`. Requires Python 3.10–3.12, a C compiler for Triton's JIT, an
NVIDIA GPU of compute capability 8.0 or newer, and the `cuequivariance` packages of the kit's lock for `OF3T_TRIMUL=cueq`.

## Files
`of3t_hook/`: `sitecustomize.py` (import hook; chains the kit hook via `OF3T_KIT_LEVERS`; installs the per-item model timer), `of3t_levers.py` (template / trimul levers, the triangle-attention flag words),
`of3t_paircache.py` (the pair cache). `tests/inputs/`: small public query files (`warm`, `check`).
Third-party attributions: `NOTICE`; OpenFold3's licence text: `OPENFOLD3_LICENSE`.
