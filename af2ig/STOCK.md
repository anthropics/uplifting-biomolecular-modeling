# AF2 initial guess (dl_binder_design) — stock, as pinned

## Pin

Upstream: https://github.com/nrbennet/dl_binder_design at commit `cafa3853ac94dceb1b908c8d9e6954d71749871a`, shipped under
`stock/` as the source archive `dl_binder_design-cafa3853.tar.gz` (that commit's files, unmodified; digests in `stock/PINS.json`);
weights: `params/params_model_1_ptm.npz` (373 MB, sha256 in PINS.json), the AlphaFold-2 `model_1_ptm` parameters of DeepMind's
`alphafold_params_2022-12-06.tar` (CC BY 4.0) — the archive upstream's README has users download; `bash run.sh install --weights DIR`
streams that archive, keeps the one member and checks its digest. The upstream README inside the carried archive
(`af2_initial_guess/README.md`) describes the 2021 AlphaFold parameter archive as CC BY-NC 4.0; this kit fetches
`alphafold_params_2022-12-06.tar`, which DeepMind publishes under CC BY 4.0 (https://github.com/google-deepmind/alphafold#model-parameters-license).
Upstream's licences as found: `opt/forward/af2ig_kit/upstream/` (dl_binder_design MIT; the vendored AlphaFold Apache-2.0); third-party notices
and verbatim licence texts for everything the kit adapts or ships: `THIRD_PARTY_NOTICES.md`, `third_party_licenses/`. `stock/` is never edited;
`run.sh install` unpacks the archive to `./dl_binder_design` and applies the patch series `opt/forward/af2ig_kit/patches/[0-9][0-9]_*.diff`
(`stock/PINS.json` `checkout.patches`) to that checkout, and `opt/forward/af2ig_kit/patches/patched_files/` carries the five files the series
creates or modifies as they read after it (`THIRD_PARTY_NOTICES.md` names them and the upstream files among them).

## Stack

Ubuntu 22.04 (`nvidia/cuda:12.4.1-runtime-ubuntu22.04`), the CUDA 12 runtime from the pinned NVIDIA wheels (a driver for CUDA 12),
Python 3.11.5, jax / jaxlib 0.5.3 with jax-cuda12-plugin 0.5.3, dm-haiku 0.0.16, dm-tree 0.1.9, ml-collections 1.1.0,
tensorflow-cpu 2.21.0, numpy 1.26.4, biopython 1.85 — the full list is `environment/requirements.lock`; `environment/Dockerfile`
and `apptainer.def` build it. Your own environment, the pinned recipe (Linux x86_64, a driver for CUDA 12, GNU `tar` and `patch` on
PATH), from inside `af2ig/` (the image runs
CPython 3.11.5; instead of uv any released CPython 3.11 works — python.org, conda, deadsnakes `python3.11` + `python3.11-venv` — or an
existing 3.11 environment at exactly these pins, but not Ubuntu 22.04's `python3.11` apt package, 3.11.0rc1; nothing is built from source,
so no headers are needed):

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl build-essential patch   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); the installer runs only when its sha256 matches
uv venv --seed --managed-python --python 3.11 ~/venv-af2ig && . ~/venv-af2ig/bin/activate   # uv's own released CPython 3.11
grep -v '^#' environment/requirements.lock | python -m pip install --no-deps -r /dev/stdin   # one pass: the lock is complete, CUDA wheels included
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 CFLAGS=-g0           # the image's run-time environment (environment/Dockerfile ENV), for parity
bash run.sh install --weights /weights/af2ig                             # = README step 2's install line — type it once
export AF2IG_DIR=$PWD/dl_binder_design/af2_initial_guess              # the patched checkout install just made
export AF2_PARAMS=/weights/af2ig                                      # the directory that holds params/
bash run.sh check --config h100 --mode fast                              # prints DRY-RUN … (exit 0); runs nothing
```

`run.sh install` installs the shared core and the kit editable, unpacks and patches the pinned tree at `./dl_binder_design`, then runs
`stock/check_pins.py`: each named pin of `stock/PINS.json` at its version (another version or an absent one stops the step, named,
exit 3), the checkout file for file, and with `--weights DIR` the parameter file's sha256 (a file already in `DIR/params/` is kept and
only checked; nothing is fetched at run time).
The image build takes an optional pre-filled compile cache, `_jitcache/af2ig-*-jit.tar` in the build context (one per stack key; the
build is identical without one), unpacked at `/opt/jit_cache`; in the image `run.sh` seeds the JIT root from it once — with `--config <card>`
that root is the config's `AF2IG_OPT_JIT_ROOT` (default `~/.cache/af2ig_opt/jit`; printed `[af2ig-kit] jit cache: <dir> (seeded from image)`),
with no config and no root named it is `/opt/jit_cache` itself when writable (`(in-image)`), else `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`; a named root
that already holds files is left alone (`(user)`). Layout `<root>/<stack key>/<recipe>/{jax,programs}`; which recipes a given image's cache holds
is decided at its build (the default recipe for the example's lengths today; `ls <root>/<stack key>/` lists them). A preset `MODEL_OPT_JIT_ROOT` the process cannot write is used as it
is (read: programs and cache entries already there load; nothing is compiled into it — `ccache` steps aside by name and new programs go to
the default root); only when the caller also names `MODEL_OPT_STACK_KEY` and `<root>/<that key>/` holds at most `MODEL_OPT_JIT_SEED_MAX_FILES`
files (default 5000) is that subtree seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` and used from there (`… (seeded from read-only root)`).
Other cards: `configs/a100.env` (A100) and `configs/h200.env` (H200); the stack is the same.

## How stock is run

`--mode off` executes `python -I -u $AF2IG_DIR/predict_pdb.py -pdbdir <in> -outpdbdir <out>/pdbs -scorefilename <out>/out.sc
-checkpoint_name <out>/check.point -af2_dir $AF2_PARAMS -timers <tmp>/timers.jsonl [options after --]` in a clean subprocess:
no `AF2IG_OPT*`, `JAX_COMPILATION_CACHE_DIR`, `JAX_PERSISTENT_CACHE_*`, `AF2_SUBBATCH_SIZE` or `CUDA_MPS_*` variable; `AF2_PARAMS`,
`AF2IG_DIR` and any XLA / JAX variable you set pass through (`[af2ig-opt stock] ENV-CLEAN ok: …` says so before the driver starts).
Model settings are upstream's defaults: `model_1_ptm`, `-recycle 3`, initial guess on, `-force_monomer` off, `-max_amide_dist 3.0`,
feature pipeline `random_seed=0`, `PRNGKey(0)`. `--det 1` under `off`: `XLA_FLAGS=--xla_gpu_autotune_level=0`, as under every mode.

## Stock exceptions

- PDB front end: upstream's `af2_initial_guess/predict.py` reads and writes Rosetta silent files through PyRosetta, a separately
  licensed dependency the pinned stack does not include. Patch `opt/forward/af2ig_kit/patches/00_pdb_frontend_envport.diff` adds
  `predict_pdb.py` beside it: the same AlphaFold-2 computation as `predict.py` states it (features, initial guess, chain-break
  offset, model, scores), with PDB files read by a small numpy parser (chain 1 = binder, the rest = target) and written by a plain
  PDB writer (per-residue pLDDT in the B-factor column), the score file and checkpoint in `predict.py`'s format, plus per-design
  timings (`-timers`). Applied identically on every mode, `off` included; changes no model arithmetic.
- Import shims in the same patch: `af2_util.py` imports PyRosetta only if present; `alphafold/data/mmcif_parsing.py` accepts
  Biopython ≥ 1.80's `PDBData` under the old `SCOPData` name. The later patches of the series add the levers' code behind opt-in
  flags of `predict_pdb.py` that the stock line never passes (the stock caller refuses them by name).

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `AF2IG_DIR` | yes | — (the image sets it) | the patched tree's `af2_initial_guess` directory |
| `AF2_PARAMS` | yes | — | the directory holding `params/params_model_1_ptm.npz` |
| `AF2IG_OPT` | no | `fast` | the mode when `--mode` is not given; a `--mode` that disagrees is refused (exit 2) |
| `AF2IG_OPT_JIT_ROOT` | no | `~/.cache/af2ig_opt/jit` (`MODEL_OPT_JIT_ROOT` when set) | root of the kit modes' compile cache and program store, created under `<root>/<stack key>/` (per stack and GPU) on the first run, so a root you set (either variable) should be writable — one that is not makes `ccache` step aside by name (`reason=cache_dir_unwritable:<dir>`) and sends the program store to the default root; a `JAX_COMPILATION_CACHE_DIR` you preset is kept as your cache directory, unless you also name a root before `run.sh` (`MODEL_OPT_JIT_ROOT` or `AF2IG_OPT_JIT_ROOT` exported, or an image's shipped cache) — then `run.sh` drops it and the cache and the program store go under the root; the config's default root alone never overrides it |
| `AF2IG_OPT_HOME`, `AF2IG_OPT_KIT` | no | found from the editable install | this `af2ig/` directory and its `opt/forward/af2ig_kit/` |
| `AF2IG_OPT_FORCE` | no | unset | `1` overrides a refusal of the pins, checkout or weights check; the printed line says so |
| `AF2IG_OPT_CACHE_DIR` | no | `$XDG_CACHE_HOME/af2ig_opt`, else `~/.cache/af2ig_opt` | the memo of the weights file's digest |
| `AF2IG_OPT_TRIMUL_CHUNK` | no | unset | `on` or `<rows>:<min residues>` opts `big` into the row-chunked TriangleMultiplication (CHANGES.md) |
| `AF2IG_OPT_TMPL_POINTWISE_SUB` | no | unset (the mode's value) | `<rows>` or `off` for the template point-wise attention batch (CHANGES.md `L18`) |
| `AF2IG_OPT_TRIATTN_CORE_DTYPE` | no | `bf16` under `fast` and `big` | `fp32` keeps float32 TriangleAttention operands (CHANGES.md `L19`) |
| `MODEL_OPT_LEVERS_OFF` | no | unset | lever ids dropped from the mode (README.md Notes) |
| `MODEL_OPT_TARGET_GPU` | no | `H100` (`A100` / `H200` from `configs/a100.env` / `configs/h200.env`) | the card the configuration targets; another card present is reported, never refused |

An `AF2IG_OPT*` name the kit does not declare is refused by name as a mistyped switch.
