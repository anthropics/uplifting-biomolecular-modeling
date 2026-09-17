# fast_inference — fast model construction and the CUDA-graph diffusion step

The add-on directory behind the kit's `fast_init`, `cuda_graphs` and `graphs_strict` levers. The kit's modes export the switches below and
put `of3_levers/` on their hook chain (`fast_init` on every kit mode; `cuda_graphs` + `graphs_strict` on `exact` and `fast`); nothing here is
called directly when you use `run.sh` or `openfold3_ob0-opt`. On its own the directory is an import hook: `of3_levers/` first on `PYTHONPATH`,
each lever off unless its variable is set.

## Levers

- `OF3_FAST_INIT=1` (`fast_init`, `of3_fastinit.py`) — the per-Linear weight-initialisation functions and OpenFold3's `Linear.reset_parameters`
  become no-ops before the model is built; `load_state_dict(strict=True)` then overwrites every tensor they would have initialised. Exact.
- `OF3_CUDA_GRAPHS=1` (`cuda_graphs`, `of3_graphs.py`) — one denoising step of `SampleDiffusion._sample_rollout` (the `diffusion_module` call) is
  captured into a CUDA graph per input shape after eager warm-up steps and replayed for the remaining steps, the other samples and later items
  whose read feature entries have the captured shapes (`OPENFOLD3_OB0_OPT_GRAPHS_KEEP=0`: re-capture on any differing entry). The sampler's
  random draws stay outside the graph in stock order; the loop iterates a host copy of the noise schedule, so no step synchronises the device.
  Two data-dependent host syncs (`broadcast_token_feat_to_atoms`, `apply_cyclic_offsets`) are served from gather tables / predicates recorded
  with the stock functions during the eager steps, checked per item and refreshed in place. One capture generation lives in one shared memory
  pool; a new shape drops the old generation whole. A step whose warm-up calls a DeepSpeed DS4Sci or cuEquivariance kernel is not captured
  (named; that shape runs eager), and a roll-out that requests DS4Sci runs with it off inside the sampler (`ds4sci_off_rollouts`); `release()`
  drops every generation, table and the pool. Exact under the deterministic recipe; otherwise the run-to-run variation stock itself has.
- `OF3_GRAPHS_STRICT=1` (`graphs_strict`) — a capture refusal or failure raises instead of running that shape eager.
- `OF3_DETERMINISTIC=1` — the deterministic recipe (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; `=warn`:
  warn-only), which the kit's `--det 1` sets. Not a speed lever.

The levers do not depend on the runner YAML's kernel switches (`settings.memory.eval.use_*`). `of3_graphs.py` imports the shared core
(`common/opt_core`) for one rule: an out-of-memory error inside a served path propagates; no fallback is applied to it.

## Standalone use

```bash
export PYTHONPATH=<this dir>/of3_levers${PYTHONPATH:+:$PYTHONPATH}
OF3_FAST_INIT=1 OF3_CUDA_GRAPHS=1 OF3_GRAPHS_STRICT=1 run_openfold predict --query_json q.json --output-dir out --inference-ckpt-path <checkpoint>
```

stderr carries `[of3_levers] …` at install and `[of3_graphs] …` per capture, per refusal (by name) and once at exit (captures, replays, eager steps).

Files: `of3_levers/sitecustomize.py` (the hook), `of3_fastinit.py`, `of3_graphs.py` (GraphedStep, the memoised host-sync replacements,
`release()`); `tests/public_inputs.py` is the kit's `warm` input builder (a barnase–barstar query, PDB 1BRS, single-sequence MSAs, tiled
`1to1` … `6to6`); `NOTICE`, `OPENFOLD3_LICENSE` carry the third-party attributions.
