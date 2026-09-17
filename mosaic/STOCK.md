# mosaic — stock, as pinned

## Pin

Upstream is three repositories, each pinned by commit (none is tagged or published as a wheel) and shipped under `stock/` as a source
archive of the importable package plus its packaging files, licence and README (`git archive` at the pin; recipe and digests in
`stock/PINS.json`): `mosaic` 0.1.0, https://github.com/escalante-bio/mosaic @ `70fec525423f5f87156a1a957b4a4048f9f8e676` (the design engine:
`Boltz2` wrapper, losses, `simplex_APGM`, ProteinMPNN with its weights committed upstream); `joltz` 0.1.0, https://github.com/nboyd/joltz @
`ed0f04257dac85bd4b7bf45521cc280f86d38ded` (the JAX translation of the Boltz-2 torch module); `boltz` 2.2.1, the escalante-bio fork,
https://github.com/escalante-bio/boltz @ `1acc397b6e81f30dc07a80d17a26c38f9c6f6942` (the torch module the checkpoint loads into, the featurizer,
the weights downloader). `stock/src/` unpacks the upstream modules this kit's documents cite, plus the recipe notebook
`examples/boltz_notebook.py`. `stock/check_pins.py` (run by `run.sh` before every command, and by the package's activation gate) accepts an
install from the pinned commit or from these archives and refuses anything else. `stock/` is never edited and no installed upstream file is
modified: the kit is add-only (its lever files install as a new `mosaic.fast` sub-package that stock never imports).

Weights: the Boltz-2 checkpoint `boltz2_conf.ckpt` and the CCD molecule library, fetched into `$MOSAIC_CACHE_DIR/boltz/` by
`bash run.sh install --weights DIR` (or by `warm` when absent) through boltz's own `download_boltz2()` and checked against the sha256 in
`stock/PINS.json` "weights" (a checkpoint already in `DIR` is kept, nothing is downloaded, and only the digest is checked); never redistributed;
terms: MIT per the Boltz README (see `THIRD_PARTY_NOTICES.md` §2). The ProteinMPNN (MIT) and AbMPNN (CC BY 4.0) weight files and the UniRef50-derived trigram table (CC BY 4.0)
committed upstream ship inside the `mosaic` package and, verbatim, inside `stock/mosaic-70fec525.tar.gz` (their terms: `THIRD_PARTY_NOTICES.md`
and `opt/mosaic_opt/NOTICE.md`). The example target (barstar, PDB
1BRS chain D, 89 residues) is inlined in `opt/mosaic_opt/tools/fetch_public_inputs.py`.

## Stack

Ubuntu 22.04, CUDA 12.8 runtime (driver 550 or newer), Python 3.12; `jax` = `jaxlib` = `jax-cuda12-plugin` = `jax-cuda12-pjrt` = 0.10.2,
`equinox` 0.13.8, `torch` 2.7.1+cpu (checkpoint read, featurizer and ProteinMPNN only), `numpy` 2.5.2 — the full list is
`environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it. Upstream's own `pyproject.toml` is not an
installable set at the pin (`upstream_issues/03_mosaic_pyproject_unsatisfiable_jax_cuda_group.md`), so the lock is the install. From inside
`mosaic/`, into a fresh or existing Python 3.12 environment:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git build-essential   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; } # uv itself, once per user (skipped when present)
uv venv --seed --managed-python --python 3.12.1 ~/venvs/mosaic && . ~/venvs/mosaic/bin/activate   # uv's own released CPython build (the python-build-standalone 3.12.1 the image itself uses), headers included; if you use apt/deadsnakes python instead of uv: python3.12 + python3.12-venv + python3.12-dev and `python3.12 -m venv`; skip the line for an environment already at the pins
export PYTHONHASHSEED=0                                              # the image's process environment (environment/Dockerfile ENV also sets CFLAGS=-g0 and LD_LIBRARY_PATH=/usr/local/cuda/lib64, the base image's CUDA libraries)
python -m pip install $(grep -E '^(pip|setuptools|wheel)==' environment/requirements.lock)
python -m pip install --no-deps -r environment/requirements.lock     # every distribution pinned, the three upstream packages from their commits
bash run.sh install [--weights DIR]                                     # = README step 2's install line — type it once; step 2 continues at its export line (../common/opt_core + opt/ editable, stock/check_pins.py, the kit's lever files into the installed mosaic)
```

`stock/mosaic-70fec525.tar.gz`, `stock/joltz-ed0f0425.tar.gz` and `stock/boltz-1acc397b.tar.gz` are reference copies, for reading: pip installs `mosaic`,
`joltz` and `boltz` from their git URLs at the pinned commits (the lock's `name @ git+https://…@<commit>` lines), building each from source in an
isolated build environment whose backend it fetches from the index at install time (hatchling for `mosaic`, setuptools for `boltz` and `joltz`;
network and `git` on PATH needed) — the lock pins runtime packages only. The pin check also accepts an install made from one of these archives.
The image build takes an optional pre-filled compile cache `_jitcache/mosaic-<stack key>-jit.tar` from the build context (the build is identical without one;
the source tree and route C carry none); in the image `run.sh` uses `/opt/jit_cache` in place when it is writable, else seeds a JIT root from it once (`MODEL_OPT_JIT_ROOT`
when set, writable and empty, otherwise `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`; a populated writable `MODEL_OPT_JIT_ROOT` is left as it is; a read-only one whose
`<stack key>` subtree holds up to `MODEL_OPT_JIT_SEED_MAX_FILES` files (default 5000) is copied once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` and that copy becomes the root,
otherwise the read-only root is used as is — nothing is compiled into it, caches already there are read, and no line is printed), points `MOSAIC_OPT_CACHE_ROOT` at
`<root>/<MODEL_OPT_STACK_KEY>/mosaic` under `--config` unless you set it yourself, and prints `[mosaic-kit] jit cache: <dir> (<how>)`.
The kit names `jax`, `jaxlib`, the CUDA plugin / pjrt and `equinox` on every activation line and notes a stack that differs from the pins
there, without refusing; upstream packages off their pinned commits are refused (`MOSAIC_OPT_FORCE=1` runs anyway and records the override).
Cards: `configs/h100.env` (compute capability 9.0), `configs/h200.env` (compute capability 9.0; the H100's entries with the H200 as target) and
`configs/a100.env` (compute capability 8.0, 80GB or 40GB) set `MODEL_OPT_TARGET_GPU` and derive `MODEL_OPT_STACK_KEY`; the compilation-cache key carries the GPU product name, so each card compiles and keeps its own cache.

## How stock is run

Upstream has no command line: the model is a Python API and the recipe is the notebook. `--mode off` runs the recipe through the kit
driver (`opt/mosaic_opt/tools/public_design_run.py`; every recipe call lives in `opt/mosaic_opt/tools/recipe.py`) with every lever off,
launched by `opt/mosaic_opt/stock_design.py` in a subprocess whose environment is stripped of every kit and JAX-cache variable
(`MOSAIC_CACHE_DIR` excepted); the subprocess prints an `ENV-CLEAN` line stating what it checked and which install layout it runs on before
`jax` is imported. On stderr the route reads `[mosaic-opt] NOT ACTIVE: mode off: stock mosaic (the kit driver with every lever off in a clean
subprocess; no environment set, no lever applied) (mode=off route=driver)` — the stock proof line, not a refusal — then `[mosaic-opt] ENV-CLEAN ok:
absent=… mosaic.fast=… stock_install=…` and, when the design completes, `[mosaic-opt] DONE mode=off rc=0 … out=<dir>` (exit 0; a failed design is rc 1). The recipe's calls and upstream defaults:

| step | recipe call | upstream default |
|---|---|---|
| model | `Boltz2()`: torch checkpoint → `joltz.from_torch` | weights from `MOSAIC_CACHE_DIR` |
| features | `model.binder_features(binder_length=…, chains=[TargetChain(seq)])` | `binder_length = 75`; `TargetChain.use_msa = True` (MSA from upstream's server) |
| loss | `model.build_loss(loss=2·BinderTargetContact() + WithinBinderContact() + 5·InverseFoldingSequenceRecovery(mpnn, temp=0.01), features=…)` | `recycling_steps=1`, `sampling_steps=25`, `deterministic=True`; ProteinMPNN `v_48_020` |
| design | `simplex_APGM(n_steps=75, stepsize=0.1, momentum=0)` then `simplex_APGM(n_steps=50, stepsize=0.5, scale=1.5, momentum=0)` | x0 seeded with `np.random.randint(100000)` |
| refold | `model.predict(PSSM=…, features=…, key=key(0))` | Boltz-2 refold of the final PSSM |

Precision is JAX's default (float32, default matmul precision, x64 off); `JAX_ENABLE_X64` and `JAX_DEFAULT_MATMUL_PRECISION` are refused when set, in every mode. `--det 1` under `off`
sets `PYTHONUNBUFFERED=1` only; the JAX allocator is the library default in every mode. Stock is deterministic within one process and not
bitwise across fresh processes (each fresh compile re-autotunes).

## Stock exceptions

None at the numerics level. The driver departs from upstream's `examples/boltz_notebook.py` in four places, identically in every mode
(`off` included): an explicit seed (`--seed S`: x0 from `key(seed)`, stage keys `fold_in(key(seed), 1|2)`, refold `key(0)`; the notebook
draws `np.random.randint`), binder length 80 by default (`--binder-length 75` is the notebook's), the target's MSA from a staged file
(`--msa`; without it the target runs single-sequence and no route contacts a server), and the refold written as arrays.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `MOSAIC_CACHE_DIR` | yes | — | mosaic's own weights cache root (`<root>/boltz/`); every verb refuses by name while it is unset |
| `MOSAIC_OPT_CACHE_ROOT` | `warm`, `exact`; optional for `fast` / `big` | — | root of the per-shape files: `<root>/<shape>/xla_cache*/` (compilation cache, autotune results) and `features_<shape>.npz`; one root per stack key. On the ACTIVE / DRY-RUN line `p1=<autotune phase>:<dir>`: `load` / `dump` under `exact` (`<shape>/xla_cache`, the pinned autotune results loaded or, by `warm`, written), `off` = compilation cache only (`fast`: `<shape>/xla_cache_fast`, `big`: `<shape>/xla_cache_big`, no autotune pin); `features=present|absent` refers to `exact`'s frozen features (`fast` / `big` featurize in every run). A cache seeded from an image (§Stack) therefore serves `exact`'s warm start (compilation cache, autotune results, frozen features) and `fast` / `big`'s compilation cache for the shapes it carries |
| `MODEL_OPT_STACK_KEY` | no | derived by `configs/<card>.env` | `jax<v>-jaxlib<v>-cuda12plugin<v>-<gpu product>`: the stack and GPU type a cache belongs to; set by hand only where no GPU is visible when the config is sourced |
| `MODEL_OPT_JIT_ROOT` | no | — (`run.sh` supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`, when none is set: made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) | when set and `MOSAIC_OPT_CACHE_ROOT` is not, the config derives `<MODEL_OPT_JIT_ROOT>/<MODEL_OPT_STACK_KEY>/mosaic`, so it must be writable: the per-shape compilation caches and frozen features are created under it, per GPU / stack key, on first use (`warm`, or the first design of a shape); unset, nothing is derived and `MOSAIC_OPT_CACHE_ROOT` is given directly (`fast` / `big` run without a persistent cache when neither is set) |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` from the config | a mismatch with the GPU seen is a note on the ACTIVE line |
| `MOSAIC_OPT` | no | `fast` | the mode when `--mode` is not given (`run.sh` refuses a `--mode` that disagrees with it; the `mosaic-opt` command lets `--mode` take precedence) |
| `MODEL_OPT_LEVERS_OFF` | no | — | `<id>[,<id>…]`: the mode without the named levers (CHANGES.md "Switches") |
| `MOSAIC_OPT_CACHE_DIR` | no | — | in-process route only: that process's compilation-cache directory in place of the root |
| `MOSAIC_OPT_ALLOW_PARTIAL` | no | unset | in-process route only: `1` records a partial activation instead of exiting 3 (the verbs take `--allow-partial`) |
| `MOSAIC_OPT_FORCE` | no | unset | `1` runs with upstream packages off their pinned commits; recorded on the ACTIVE line |

`JAX_ENABLE_X64` and `JAX_DEFAULT_MATMUL_PRECISION` must stay unset (refused by name). The kit sets no `XLA_PYTHON_CLIENT_*` variable.
