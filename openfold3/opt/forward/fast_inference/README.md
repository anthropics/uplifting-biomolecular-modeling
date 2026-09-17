# OpenFold3 fast-inference kit — `of3_levers`

Inference speed-ups for **OpenFold3 0.4.1** (preview-2 weights, `of3-p2-155k.pt`) that change only *how* the computation is scheduled —
same weights, same inputs and MSAs, same recycles, diffusion steps, seeds and samples, same output files. No source edits: the levers load
through an import hook (`of3_levers/sitecustomize.py` on `PYTHONPATH`) and are switched on by environment variables. In this tree the
hook directory and its switches are set by the mode table (`opt/openfold3_opt/modes.py`); nothing here is called directly.

| lever | env | what it does | numerics |
|---|---|---|---|
| fast-init | `OF3_FAST_INIT=1` | skips the CPU-side random weight initialisation that the checkpoint load overwrites anyway (model construction in seconds instead of about a minute) | exact (bitwise) |
| CUDA-graph diffusion step | `OF3_CUDA_GRAPHS=1` | captures one denoising step of the diffusion sampler per input shape into one shared memory pool and replays it for every step of every sample; two data-dependent host syncs are replaced by memoised gather tables that a new item of equal shapes refreshes in place (checked against the stock broadcast; the resident graph is kept); the capture key and the static input dict cover the feature entries the step READS (recorded by name at the eager warm-up), so a later item of the process whose read entries have the captured shapes replays the resident generation after the per-item checks (parameter storage unmoved, gather tables refreshed, static buffers at their captured addresses) although entries it never reads (MSA / template stacks) changed shape — census `kept=<n>`; `OPENFOLD3_OPT_GRAPHS_KEEP=0` re-captures at any difference; the sampler loop iterates a host copy of the noise schedule (no per-step synchronisation), one warm-up step before every capture after the first; a rollout that requests DeepSpeed's DS4Sci attention runs it off inside the sampler, named (`ds4sci_off_rollouts`: that kernel launches on the legacy default stream and cannot be captured — the cuEquivariance / Triton attention paths capture); `release()` drops every generation for the package's item-boundary memory release | exact (bitwise vs eager under the deterministic reference configuration) |
| strict capture | `OF3_GRAPHS_STRICT=1` | a capture failure fails the run instead of falling back to the eager sampler | — |

`of3_levers/`: `sitecustomize.py` (the import hook: installs a meta-path finder when a lever switch is set; with `OF3_DETERMINISTIC=1` it
also sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)` — the deterministic reference configuration, a
testing aid, not a speed lever), `of3_fastinit.py` (fast init), `of3_graphs.py` (CUDA graphs). `config/stock_predict.yml` is the kernels-off runner configuration described next.

## The stock configuration (`config/stock_predict.yml`)

Public openfold3 0.4.1 defaults `settings.memory.eval.use_deepspeed_evo_attention` to `True` on CUDA. Unless the DeepSpeed DS4Sci
evoformer-attention kernel can be JIT-built in the environment (CUTLASS + nvcc), every prediction then fails inside `predict_step`, the
exception is logged, and `run_openfold predict` exits 0 having written no structures. `config/stock_predict.yml` selects the stock PyTorch
attention path explicitly (the path openfold3 itself uses on ROCm) — a kernel-backend selection, not a numerics or memory change. It is
the base of the `fast` and `big` runner yamls; `off` and `exact` run the kit's stock configuration (`opt/openfold3_opt/stock_cueq_on_predict.yml`).

## Limits

The graphed sampler's share of model time shrinks with input size, so the speed-up does too (on the fp32 `exact` line the step is
GPU-bound above ~512 tokens and the eager DS4Sci sampler is faster there: that line caps its graphs at 512 tokens, `modes.Line.graphs_max_tokens`);
DeepSpeed's DS4Sci attention cannot run inside a captured step (it launches on the legacy default stream) — a graphed rollout that requests it
runs the diffusion module's attention on the cuEquivariance / torch path instead, named once, while the trunk keeps it; the kernels-off graphed
line (`fast`) pins a caller's yaml kernel flags off (`runner_yaml.line_pins`); per-seed bitwise equality holds for identical call composition
(openfold3's own seed stream depends on query order within a call). Requires an NVIDIA GPU of compute capability 8.0 or newer for the
Triton kernels the other add-ons bring; this kit itself needs no compiler. Third-party attributions: `NOTICE`; OpenFold3's licence text: `OPENFOLD3_LICENSE`.
