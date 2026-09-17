# trunk_kernels — template distinct-evaluation and the diffusion pair cache

The add-on directory behind the kit's `templ_distinct` and `paircache` levers. The kit's modes export the switches below and put `of3t_hook/`
on their hook chain ahead of `fast_inference/of3_levers` (`OF3T_KIT_LEVERS=<that directory>` makes this hook execute that one); nothing here
is called directly when you use `run.sh` or `openfold3_ob0-opt`. On its own the directory is an import hook: `of3t_hook/` first on
`PYTHONPATH`, each lever off unless its variable is set. The hook always wraps `OpenFold3.forward` in a timer that prints
`[of3t] model forward <s>s (N tokens)` per predicted item (one device synchronisation per item; numerically inert).

## Levers

- `OF3T_TEMPL_DISTINCT=1` (`templ_distinct`, `of3t_levers.py`) — for a chain without templates the featuriser emits `n_templ=4` identical dummy
  template slots. `TemplatePairStack.forward` detects slots identical by value (`torch.equal` on its input and mask), runs the 2-block template
  pair stack once and expands the result back to `n_templ` copies, so the caller's own `torch.sum(t, dim=-4) / n_templ` reduces exactly the
  values stock reduces. Templated input (slots differ) takes the stock path; under `settings.memory.eval.offload_inference.template_module`
  every call carries one template and the lever is inactive for that item. One `[of3t_levers] templ distinct census: …` line per item names
  the case. Exact. `OF3T_TRIATT` is not a switch of this add-on: setting it raises at install, by name.
- `OF3T_PAIRCACHE=1` (`paircache`, `of3t_paircache.py`) — within one diffusion roll-out the step-invariant pair inputs — `DiffusionConditioning`'s
  conditioned `zij`, the diffusion transformer's shared `layer_norm_z(zij)` and its 24 per-block `linear_z` pair biases — are computed by the
  stock statements at the roll-out's first step and served to the remaining steps; the single (`si`) half runs every step through the module's
  own `_forward` / `_chunk_forward`. Roll-outs are delimited by the sampling call (`SampleDiffusion._sample_rollout`, `OpenFold3._rollout`), never
  by tensor identity. The biases are held head-major with 16-byte-aligned rows so the kit's diffusion-attention levers read them in place;
  stock consumers add them by value. `MAX_GB` (4) caps the cache: blocks over it run the stock computation, whole. With `OF3_CUDA_GRAPHS=1`
  the cached tensors are static graph inputs, allocated once per shape and refreshed in place at each roll-out start. Exact.

The levers do not depend on the runner YAML's kernel switches (`settings.memory.eval.use_*`); those remain runner settings, not switches of
this add-on.

## Standalone use

```bash
export PYTHONPATH=<this dir>/of3t_hook:<fast_inference>/of3_levers      # this hook first
export OF3T_KIT_LEVERS=<fast_inference>/of3_levers                       # optional: chain that add-on's hook and levers
OF3T_TEMPL_DISTINCT=1 OF3T_PAIRCACHE=1 run_openfold predict --query_json q.json --inference-ckpt-path <checkpoint> --output-dir out
```

Files: `of3t_hook/sitecustomize.py` (the hook and the model timer), `of3t_levers.py` (template distinct-evaluation and its per-item census),
`of3t_paircache.py` (the pair cache, incl. its CUDA-graph cooperation); `NOTICE`, `OPENFOLD3_LICENSE` carry the third-party attributions.
