# PXDesign hoist add-on — exact step-invariant hoisting for the PXDesign diffusion sampler

An add-on for [PXDesign](https://github.com/bytedance/PXDesign) (commit `f788441`, which uses Protenix `v0.5.0+pxd`) that makes
`pxdesign infer` faster **without changing what it computes**: same model, same weights, same inputs, same 400 diffusion steps, same
number of samples, same random draws in the same order. In this tree the `pxdesign_opt` package drives it (`run.sh design --mode
exact|fast|big`: the package installs the lever after checkpoint load and prints one `[pxd_hoist v1.2.2]` line per prepare); this file
documents the lever itself.

## What it does (one paragraph)
Inside one `sample_diffusion` call the conditioning tensors (`input_feature_dict`, `s_inputs`, `s_trunk`, `z_trunk`) are constant across all
400 steps and all sample chunks, yet the stock denoiser recomputes, at every step, several sub-graphs that depend only on them: the pair
conditioning `pair_z`, the 16 token-block pair biases `Linear(LayerNorm(z))` (evaluated on an `N_sample`-expanded copy of `z`), the atom
encoder's `c_l`/`p_lm` (including the small MLP), the 8 atom-block pair biases and the atom blocks' AdaLN conditioning terms (their
conditioning input `c_l` is step-invariant). The add-on computes these **once per `sample_diffusion` call by calling the stock sub-modules
on the stock inputs in the stock shapes**, caches them, and replays them at every step. Everything that depends on the noisy coordinates or
the noise level runs through the untouched stock code. Default numerics throughout (no TF32, no precision change).

## Requirements
* The stock PXDesign environment as PXDesign's own Dockerfile/`install.sh` builds it: CUDA 12.1 devel toolkit (`nvcc` — Protenix
  JIT-compiles its fast LayerNorm CUDA extension on first use), gcc/g++, ninja, Python 3.11, `torch==2.3.1+cu121`, `deepspeed==0.15.4`,
  `numpy==1.26.3`, PXDesign `f788441` (which pulls Protenix `v0.5.0+pxd` and imports PXDesignBench `v0.1.2`). The add-on adds no
  further dependency (pure Python).
* The PXDesign checkpoint `pxdesign_v0.1.0.pt` in a directory `$CKPT`; PXDesign's CCD cache directory exported as
  `PROTENIX_DATA_ROOT_DIR`; one CUDA GPU.

## Use without the package
```bash
export PYTHONPATH=/path/to/opt/forward/hoist:$PYTHONPATH          # provides the `pxd_xattempt` package
# from Python, on an existing runner:  import pxd_xattempt.hoist as H; H.install(runner.model)   (H.uninstall(runner.model) restores stock)
# this tree's package: run.sh design --mode exact ... (README.md at the tree root)
```

## Switches
| env | default | meaning |
|---|---|---|
| `PXD_HOIST` | 1 | 0 disables `install()` entirely (stock code path) |
| `PXD_HOIST_MODE` | `shape` | `shape`: hoisted ops are evaluated once on `N_sample`-expanded rows exactly like stock and the first slice is cached — the same values per element as stock; `rows`: evaluate on un-expanded rows (cheaper one-time prepare; last-ulp differences from cuBLAS choosing kernels by M) |
| `PXD_HOIST_MASK` | 1 | 1: also cache the `_local_attention` padding-mask bias (H5; depends only on N_atom and the 32/128 window); 0 leaves that term to stock |

## What is NOT in this add-on (and why)
* No TF32 / bf16 / precision changes (numerics-changing; out of scope for an exact lever).
* No change of `diffusion_chunk_size` / sample batching: merging chunks changes which random numbers each design receives (the CUDA
  Philox stream and numpy's `Rotation.random` are consumed per chunk per step), so per-seed outputs would differ like a reseed.
* No target-feature cache across seeds (featurisation consumes the RNG streams the sampler uses).
* Multi-GPU / DDP is not exercised (PXDesign inference is single-GPU per process).

## Files
`pxd_xattempt/hoist.py` (the lever) ·
`inputs/` (the warm-up's inputs: `tasks_3targets.json`, three design tasks on the public wwPDB entries 5O45, 1TNF, 3DI3, and the vendored
copies of those entries under `inputs/targets/`) · `LICENSE` · `NOTICE` · `LICENSE_NOTES.md`.

Version 1.2.2. Third-party notices: `NOTICE`, `LICENSE_NOTES.md`.
