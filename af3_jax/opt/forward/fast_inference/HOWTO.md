# fast_inference — the kit script `run_alphafold_fast.py` (patch 01) and the compilation-cache class

This directory holds `patches/01_run_alphafold_fast_inference.diff`, a patch to the fork's `run_alphafold.py`, and the patched file itself,
`patches/patched_files/run_alphafold.py` (byte for byte = `stock/src/run_alphafold.py` + the patch; the package copies it next to the stock
script as `run_alphafold_fast.py` — the stock file is never edited). `rows.sh` holds the flag rows the mode table reads (`COMMON` =
`--norun_data_pipeline …`, `FAST` = `--featurisation_workers=3 --featurisation_prefetch=4 --output_writer`); `tests/inputs/` holds the example
fold input (PDB 1BRS, barnase–barstar, 5 seeds). `exact`, `fast` and `big` run this script; `off` runs the stock one. Nothing here touches
the network, the weights, the inputs, recycles, samples or seeds.

## Levers in the script

| lever | flag | what it does | numerics |
|---|---|---|---|
| L1 featurisation prefetch | `--featurisation_workers W --featurisation_prefetch K` (both default 0 = upstream's in-line featurisation) | featurises the upcoming (input, seed) items in W forked worker processes with the identical `featurise_input` call while the GPU runs the current item; needs `--norun_data_pipeline` (every work item known up front), otherwise the script featurises in line exactly as upstream | bitwise: each item is featurised with its own `RandomState(seed)` as upstream, `jax.random.PRNGKey(seed)` untouched |
| WRITER output writer | `--output_writer` (default off = upstream's in-line extraction and writing) | after each seed's inference returns, the seed's result extraction (`extract_inference_results`, embeddings, distogram) is submitted to one writer thread, and after the job's last seed its `write_outputs`; tasks run in submission order, so a job is written while the next job's batch is on the GPU; `Fold job … done` prints when the files are written and `main` drains the thread before `Done running N fold jobs.` (a failure on the thread re-raises in the main thread at the next submission or at the drain) | bitwise (the same calls on the same arrays; only when they run changes) |
| FIX1 one-shot autotune attempt | none (always on in this script) | the fork calls `tokamax.autotune()` before the first inference of every new bucket size; on this stack the call raises inside tokamax (jaxlib 0.10.2 changed the MLIR location API tokamax 0.0.12 walks: `'NameLoc' object has no attribute 'is_a_file'`), stock swallows the exception and repeats the attempt per bucket. This script makes the attempt once per process and prints the cause once (`Tokamax autotune unavailable (…)`) | none: no process on this stack obtains an autotune result, patched or not (the wrapper's `TOKAMAX` line says so per pass) |
| `_jitted_apply` seam | none | the jitted forward is exposed as `ModelRunner._jitted_apply` (the same `hk.transform` + `jax.jit` as upstream's `_model`, which becomes `functools.partial(_jitted_apply, params)`): the attach point of `--mode big --n_gpu P`'s row-sharded pair stack | none |

## The compilation cache is the numerics class

Two processes produce bit-identical outputs only when both load the same XLA autotune results, and those travel in the `--cache_dir`
compilation cache: two stock processes that autotune independently can differ in the last bits (small per-seed confidence differences; a
borderline ranking can flip). Hence the kit's rule: `run.sh warm --mode <m>` builds the mode's class once per (GPU model, jax build) under
`$AF3_JAX_CACHE_ROOT`, every `pred` of that mode reads it, and `exact` is bit-identical to the stock script run on the same class
(`--cache_dir <exact's class>`). A class is per (jax / jaxlib build, GPU model, model configuration, bucket list); a longer bucket list extends
it. tokamax's own table (`<class>/tokamax_autotune.json`) is never written on this stack (FIX1 above): every tokamax kernel runs at its packaged
per-GPU table entry or, where the table has none, at tokamax's `heuristics` cache-miss policy — a fixed function of the shape, the same on every
host. Host-to-host differences are XLA's compile-time autotuning, carried by the class, not tokamax's.

## Limits

* `--featurisation_workers` needs `--norun_data_pipeline`; with the data pipeline on, featurisation is upstream's in-line path.
* Host RAM: about 3 GB per featurisation worker on top of the process's own; `--output_writer` holds one finished job's results (host NumPy) while the next runs.
* AF3 writes a wall-clock timestamp into every mmCIF (`_ma_model_list.model_group_name`); compare structures without that line.
* Outputs differ across GPU models (an H100 class and an A100 class are two classes): compare runs within one GPU model and one class.
