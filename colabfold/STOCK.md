# ColabFold — stock, as pinned

## Pin

Upstream: `colabfold` 1.6.1 (https://github.com/sokrypton/ColabFold) with `alphafold-colabfold` 2.3.13 (https://github.com/sokrypton/alphafold),
shipped under `stock/` as the two PyPI wheels (`colabfold-1.6.1-py3-none-any.whl`, `alphafold_colabfold-2.3.13-py3-none-any.whl`; versions and
the package freeze in `stock/PINS.json`); entry point `colabfold_batch` (`colabfold.batch:main`). Weights: AlphaFold2-Multimer v3,
`params/params_model_{1..5}_multimer_v3.npz` (DeepMind's 2022-12-06 parameter release, CC BY 4.0; per-file sha256 in `stock/PINS.json`),
fetched by `bash run.sh install --weights DIR` through `colabfold.download`, which pulls `https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-12-06.tar`
(the same five files as `alphafold_params_2022-12-06.tar`; `stock/PINS.json` `weights.source` records both), and digest-checked;
colabfold's marker `params/download_complexes_multimer_v3_finished.txt` must exist under the root or `colabfold_batch` downloads at run
time. A single-chain query resolves to `alphafold2_ptm` under `--model-type auto`; its `params_model_{1..5}_ptm.npz` are read from the
same root when present and are neither pinned nor fetched by the kit (the parameter lines print them `NOT PINNED`). `stock/src/` holds
unmodified copies of the upstream files the kit rebinds or reads, for reference. Both wheels also carry OpenStructure's
`stereo_chemical_props.txt` (LGPL-3.0): `colabfold/openstructure/` inside the colabfold wheel, with the licence text beside it, and
`alphafold/common/stereo_chemical_props.txt` inside the alphafold-colabfold wheel (THIRD_PARTY_NOTICES.md section 5). `stock/` is never edited.

## Stack

Ubuntu 22.04, CUDA 12 (runtime 12.9 from the `nvidia-*-cu12` wheels; host driver 550 or newer), Python 3.11 (the image uses 3.11.5; the lock's
cp311 wheels serve any 3.11), jax / jaxlib 0.5.3 + jax-cuda12-plugin 0.5.3, dm-haiku 0.0.16, tensorflow-cpu as locked — the full list is
`environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it, with `hhsuite` and `kalign` (run by `colabfold_batch` for
`--templates`; the apt line below installs them on a bare host). Nothing is compiled at install. From inside `colabfold/` — the pinned recipe for
your own environment (README route C), about 6.4 GB installed:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl hhsuite kalign   # bare Ubuntu host (root: no sudo); skip what you have — hhsuite/kalign only for --templates
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); the installer runs only when its sha256 matches
uv venv --seed --managed-python --python 3.11 venv && . venv/bin/activate   # any released CPython 3.11 works (the image uses 3.11.5); --managed-python makes uv use its own CPython build even where a system python3.11 exists (Ubuntu 22.04's apt package is 3.11.0rc1); alternatives: an existing 3.11 environment, python.org, conda, or apt/deadsnakes python3.11 + python3.11-venv (no python3.11-dev needed: nothing compiles at install)
pip install --no-deps -r environment/requirements.lock                # every pin, ColabFold 1.6.1 / alphafold-colabfold 2.3.13 included (the same files as stock/*.whl)
export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   # the memory environment ('Stock exceptions'); the image also sets JAX_PLATFORMS=cuda PYTHONHASHSEED=0
bash run.sh install --weights /weights/af2_params                        # = README step 2's install line — type it once; step 2 continues at its export line
```

An environment already at the pin works the same way from `bash run.sh install` on: the pin check names any package off its version or any of the three
rebound files off the release bytes — or outside site-packages, as in an editable checkout — and refuses (exit 3). Its summary line `PINS ok
packages=… files=… weights=… env_differs=…` reports each check; `env_differs` lists the image-environment variables above whose value differs in this
process (`none`, or `skipped` when that check is not run, as under `run.sh install`) — informational, never a refusal. Other cards: one stack for
H100, H200 and A100; `--config <card>` selects `configs/<card>.env`, and the compile cache is kept per stack key (`…-sm90` for H100 and H200, `…-sm80`
for A100); within a tree JAX keys each entry by device name, so an H200 compiles its own entries on its first run at a length, also from an H100-cut
cache, and loads them afterwards. The image build also takes an optional pre-filled compile cache, `_jitcache/colabfold-<stack key>-jit.tar` in the
build context, unpacked under `/opt/jit_cache/<stack key>/…` (the build is identical without one): `run.sh` then points `JAX_COMPILATION_CACHE_DIR`
(the image's `/root/.cache/jax`, read by every mode) at `<root>/<stack key of the chosen config>/default/jax` (the kit's cache layout, as `warm`
writes it) at run time — the image tree itself where it is writable, so the shipped executables are read in place and extended, nothing linked at
build time; `run.sh` prints `[colabfold-kit] jit cache: <dir> (<how>)` — `/opt/jit_cache (in-image)` when that directory is writable, otherwise a
one-time copy at `${TMPDIR:-/tmp}/model_opt_jit-uid<uid> (seeded from image)` exported as `MODEL_OPT_JIT_ROOT` and used the same way (a `MODEL_OPT_JIT_ROOT`
you set is seeded from the image when empty and left alone when not; `COLABFOLD_OPT_JIT_ROOT` follows it when no config file set it; a preset
`MODEL_OPT_JIT_ROOT` the process cannot write is used read-only — its `<stack key>` subtree is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (up to
`MODEL_OPT_JIT_SEED_MAX_FILES` files, default 5000; printed `seeded from read-only root`), otherwise that root is left as it is, nothing of this kit
compiled into it). Without a shipped cache or a seedable preset root nothing of this prints or applies.

## How stock is run

`--mode off` executes `colabfold_batch <input> <results> [options]` — the caller's command line verbatim, plus `--data
$COLABFOLD_OPT_DATA_DIR` when `--data` is not given — in a clean subprocess whose environment carries no `COLABFOLD_OPT*` / `AF_PALLAS_ATTN*`
variable and where nothing of the kit is importable, shown on the `[colabfold-opt stock] STOCK cli=… env_prefixes_absent=… kit_dirs=…
proof=ok` line before the launch. An option not given is upstream's default in every mode (`--model-type auto` → `alphafold2_multimer_v3`
for a complex, 5 models, up to 20 recycles with early stop at tolerance 0.5, one seed, `--random-seed 0`, no templates, no relax). `--det 1`
is refused under every mode: no deterministic recipe is set by the kit.

## Stock exceptions

- Memory environment: `TF_FORCE_UNIFIED_MEMORY=0` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` are preset in the kit's environment (the image's
  `ENV`; export them in your own environment). Left unset, `colabfold_batch` sets `1` / `4.0` at import — CUDA unified memory with a 4×
  oversubscribed pool — and on this stack that setting does not make progress on an H100 80 GB; upstream keeps preset values by its own
  rule. Applied identically on every mode, `off` included; changes no model arithmetic. Upstream's `--disable-unified-memory` is not used
  because it also drops the pool to JAX's default fraction.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `COLABFOLD_OPT_DATA_DIR` | yes | — | the parameters root, passed as `--data` (`<root>/params/…`, e.g. `/weights/af2_params`); unset: the config refuses by name (exit 2) and `pred` refuses unless `--data <dir>` is given |
| `COLABFOLD_OPT` | no | `off` for a bare `colabfold_batch`; `fast` through run.sh / `colabfold-opt` | the mode when `--mode` is absent; must agree with `--mode` when both are set |
| `COLABFOLD_OPT_JIT_ROOT` | no | `~/.cache/colabfold_opt/jit` (config) | compile-cache root, which must be writable: `<root>/<stack key>/<recipe>/jax` is created on the first run of a kit mode, recipe `det` when `XLA_FLAGS` select deterministic ops or autotune level 0 (XLA's autotune results then neither loaded nor stored), else `default`; unset with no config file sourced, no cache directory is used (`XLA_CACHE state=skipped reason=no_cache_root`); a set `JAX_COMPILATION_CACHE_DIR` is used instead, as given, by every mode |
| `MODEL_OPT` | no | this `colabfold/` directory (config) | where the package finds the tree and the pins |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` (config) | the card the configuration targets; another visible card is a note on the ACTIVE line, never a refusal |
| `MODEL_OPT_STACK_KEY` | no | read from the package (config) | `jax<version>-cu<runtime>-sm<compute capability>`, e.g. `jax0.5.3-cu12.9-sm90`; the cache key above |
| `MODEL_OPT_LEVERS_OFF` | no | unset | `<LEVER>[,…]`: the mode without the named levers (`ablated=…` on the ACTIVE line); `pallas:<row>` / `triattn_xla:<row>` switch one provider row off; a name outside the mode, `ROWPAIR`, a list that empties the mode, or any name under `off` is refused by name (exit 3) |
| `TF_FORCE_UNIFIED_MEMORY`, `XLA_PYTHON_CLIENT_MEM_FRACTION` | preset | `0`, `0.95` (image) | 'Stock exceptions'; at `--n_gpu 8` `pred` starts the model process with fraction `0.90` unless another value is exported |
| `XLA_FLAGS`, `JAX_COMPILATION_CACHE_DIR` | no | unset | read, never set by the kit: the cache recipe and a caller's cache directory (above) |

Kit-declared names: `COLABFOLD_OPT`, `COLABFOLD_OPT_DATA_DIR`, `_JIT_ROOT`, `_HOME` / `_KIT` (tree location overrides), and `_N_GPU`,
`_LAUNCH_ID`, `_WORK_DIR`, which `pred` sets for the model process; any other `COLABFOLD_OPT*` name is refused by name. `AF_PALLAS_ATTN` is
exported by the activation itself; `AF_PALLAS_ATTN_ALL` set by hand is refused by name.
