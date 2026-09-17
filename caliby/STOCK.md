# Caliby — stock, as pinned

## Pin

Upstream: `caliby` 0.1 = https://github.com/ProteinDesignLab/caliby @ `41d31560c3c73d7980d94f40f3c852b90bfab5c0` (Apache-2.0) and the two packages
its own `pyproject.toml` pins — `atomworks-caliby` 1.0.0 = https://github.com/richardshuai/atomworks-caliby @ `ea2c998a593af05a47dd972ad6e72b89f87d4e14`
(BSD-3-Clause; structure parsing and featurisation, touched by no lever) and `protpardelle` 1.3.2 = https://github.com/ProteinDesignLab/protpardelle-1c @
`7962da091a335251fa8e5ddef5d2c937fbd9d9ae` (MIT; conformer generation, needed by `ensemble32` only) — shipped under `stock/` as source archives
(`stock/caliby-41d31560.tar.gz`, `stock/atomworks-caliby-ea2c998a.tar.gz`, `stock/protpardelle-1c-7962da09.tar.gz`; commits and install lines in
`stock/PINS.json`); `stock/src/` holds, for reading, the upstream files the levers replace and upstream's `inference.yaml`, `pyproject.toml`,
`env_setup.sh` and `download_model_params.sh`. Weights: Hugging Face `ProteinDesignLab/caliby-weights` @ `a51f011f4ec7ffe2daad3ab9b2bfb67ff628a096`
(sha256 per file in `stock/PINS.json` "weights"), fetched by `bash run.sh install --weights DIR` through upstream's own downloader into upstream's layout:
`$MODEL_PARAMS_DIR/caliby/caliby.ckpt` (the default checkpoint), `$MODEL_PARAMS_DIR/caliby/soluble_caliby.ckpt` and `soluble_caliby_v1.ckpt`
(`--model_name`), `$MODEL_PARAMS_DIR/protpardelle-1c/weights/cc95_epoch3490.pth` and `$MODEL_PARAMS_DIR/protpardelle-1c/configs/cc95.yaml` (`ensemble32`), plus, for `ensemble32`, upstream's `proteinmpnn/` directory, which
`generate_ensembles` fetches beside `protpardelle-1c/` (no pinned file in it). A route needs its variant's
files and, of the checkpoints, only the run's own; a missing file is refused by name. Upstream's licences and notices as found in the archives.
`stock/` is never edited.

## Stack

Debian 12 (the `python:3.12.10-slim-bookworm` image), CUDA 12.4 user-space from the `nvidia-*-cu12` wheels (driver ≥ 550), Python 3.12.10,
torch 2.6.0+cu124, triton 3.2.0, numpy 2.1.3, lightning 2.6.5, biotite 1.6.0, gemmi 0.7.1 — the full list is `environment/requirements.lock`;
`environment/Dockerfile` and `environment/apptainer.def` build it. Upstream's `pyproject.toml` ranges are all met, torch at its floor; flash-attn,
xformers, cuEquivariance and PyRosetta are not installed and not needed. From inside `caliby/`, into a fresh or existing Python 3.12 environment with a C
compiler on `PATH` (several GB of CUDA and torch wheels):

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential curl ca-certificates   # bare Ubuntu host (root: no sudo); skip what you have — the C compiler the kit modes need, curl for the uv line
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); the installer runs only when its sha256 matches
uv venv --seed --managed-python --python 3.12.10 ~/venvs/caliby && . ~/venvs/caliby/bin/activate   # uv brings its own released CPython 3.12 with headers; any other released 3.12 works too (python.org, conda; apt/deadsnakes instead of uv: python3.12 + python3.12-venv + python3.12-dev, not python3-dev); or activate the environment you have
grep -v -E '^#| @ git\+' environment/requirements.lock > /tmp/stack.txt && python -m pip install --no-deps -r /tmp/stack.txt
python -m pip install --no-deps stock/atomworks-caliby-ea2c998a.tar.gz stock/protpardelle-1c-7962da09.tar.gz stock/caliby-41d31560.tar.gz
bash run.sh install            # = README step 2's install line less `--weights`: type ONE of the two (step 2's form also fetches/checks the weights),
                            # then continue at step 2's export line. Installs the kit + shared core editable, then `python -I stock/check_pins.py`: exit 3
                            # unless caliby and atomworks-caliby are at their pins; protpardelle is checked when installed, optional for `single`-only use
```

Installing the `stock/` archives runs pip's isolated builds, which fetch each build backend from the package index at install time (`hatchling` and
`hatch-vcs` for atomworks-caliby, `setuptools` and `wheel` for caliby and protpardelle): that step needs network access; `environment/requirements.lock`
pins the runtime packages only. The image's process environment (`environment/Dockerfile` `ENV`) is `PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 CFLAGS=-g0
HF_HUB_OFFLINE=1 WANDB_MODE=disabled PDB_MIRROR_PATH= CCD_MIRROR_PATH=`; export the same in route C to match it (the configs export `HF_HUB_OFFLINE`, the mirror
variables, `TQDM_DISABLE` and `PYTHONDONTWRITEBYTECODE` on every `--config` run). How the routes engage (nothing from here on is a step to type): a SIF built from `environment/apptainer.def` links `/opt/culib/libcuda.so` to
the driver `--nv` binds and sets `TRITON_LIBCUDA_PATH` itself, so route B needs nothing extra; the manual form below is only for an image converted
straight from the Docker image (`apptainer build … docker-daemon://…` or `docker-archive://…`) or for route C run under Apptainer. Triton links the
kit kernel's launcher with `-lcuda`: on a host or container that exposes only `libcuda.so.1` (Apptainer `--nv` binds what the host's `ldconfig` lists) that link fails with `ld: cannot find -lcuda` and the kit
mode ends NOT ACTIVE — point `TRITON_LIBCUDA_PATH` at a directory, visible inside the container, holding an unversioned `libcuda.so` whose target
exists inside it: once on the host `mkdir -p ~/.libcuda && ln -sfn /.singularity.d/libs/libcuda.so.1 ~/.libcuda/libcuda.so` (`$HOME` is bound), then
`apptainer run --nv --env TRITON_LIBCUDA_PATH=$HOME/.libcuda …`; the image carries no CUDA toolkit stubs. The image build takes an optional pre-filled compile cache `_jitcache/caliby-*-jit.tar` from the build context, unpacked
under `/opt/jit_cache/<stack key>/…` (the build is identical without one; a seed serves only its own stack key — a card with another key compiles under its own `<key>/` directory beside it); in the image `run.sh` uses `/opt/jit_cache` in place when it is writable and
`MODEL_OPT_JIT_ROOT` is unset, otherwise seeds the JIT root from it once (`MODEL_OPT_JIT_ROOT` when set and empty; `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` when unset
and the image is read-only; a populated root is left alone), keys `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` under `<root>/$MODEL_OPT_STACK_KEY/`, and
prints `[caliby-kit] jit cache: <dir> (<how>)`: `in-image` | `seeded from image` | `user` (route C with no `MODEL_OPT_JIT_ROOT` prints no such line: `run.sh` silently supplies the private per-user root below and Triton compiles under it); a preset `MODEL_OPT_JIT_ROOT` the process cannot write is
used read-only — its `$MODEL_OPT_STACK_KEY` subtree is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (up to `MODEL_OPT_JIT_SEED_MAX_FILES` files,
default 5000; `seeded from read-only root`), otherwise the read-only root is used as is (nothing is compiled into it; caches already there are read); when the root
moves, the cache directories `configs/<card>.env` derived from the preset root follow it. The editable install of `opt` ships `caliby_opt_autoload.pth` (what makes `CALIBY_OPT=<mode>` reach your own script); a copied, non-editable install
refuses (exit 3). Other cards: `configs/a100.env` and `configs/h200.env` are `configs/h100.env` with `MODEL_OPT_TARGET_GPU=A100` / `H200`; nothing is built per card.

## How stock is run

Upstream declares no console script. `--mode off` runs `python -s -m caliby_opt.stock_design` in a clean subprocess — every `CALIBY_*` switch
stripped and proven absent, the installed tree's digest checked against `stock/PINS.json` "tree_digest_upstream", no import hook — which calls
upstream's Python API through the one design writer every mode shares (`opt/forward/xattempt_addon/tests/xcaliby_design.py`):
`caliby.clean_pdbs(paths, out_dir, num_workers=1)` → `caliby.load_model(name)` → `model.sample(...)`, or `caliby.generate_ensembles(...)` →
`model.ensemble_sample(...)` on `ensemble32`. The sampler runs as upstream's `inference.yaml` configures it on every mode: 500 DLMC sweeps annealed
to temperature 0.01, low-complexity regularisation, no rejection step, float32 with PyTorch's precision defaults (no route sets a TF32 switch);
inputs are designed in sorted order. Upstream's own start-up messages print in every mode, `off` included: caliby's `Using torch.compile to optimize
model performance...` (it wraps its denoiser in `torch.compile` when the checkpoint's config says so) and, on `ensemble32`, protpardelle's
LigandMPNN / Foldseek path warnings; the kit's `[caliby-opt] STACK … compile=never` token speaks for the kit's levers only. `--det 1 --seed S` under every mode: `lightning.seed_everything(S)` plus `torch.backends.cudnn.deterministic=True`
/ `benchmark=False`, upstream's scripts' own recipe. Upstream's Hydra scripts (`seq_des.py`, `generate_ensembles.py`, `seq_des_ensemble.py`) are not
wrapped: they run as shipped on the installed tree, stock unless `CALIBY_OPT` is set.

`design` forwards upstream's own keywords under their own names, only when given (absent → upstream's default; `opt/caliby_opt/settings.py`);
the `ensemble32` ones are refused by name on `single`:

| flag | upstream call | upstream's default |
|---|---|---|
| `--model_name C` | `load_model(model_name=C)`: a name in upstream's `MODEL_REGISTRY` or a `.ckpt` path. The kit pins and presence-checks `caliby`, `soluble_caliby` and `soluble_caliby_v1` (`stock/PINS.json` "weights"); any other registry name is passed through unchecked and its file must already be under `$MODEL_PARAMS_DIR` (nothing is fetched at run time) | `caliby` |
| `--device D`, `--sampling_cfg_path Y` | `load_model(device=, sampling_cfg_path=)` | `cuda`, upstream's `inference.yaml` |
| `--num_seqs_per_pdb N`, `--batch_size B`, `--temperature T`, `--num_workers W`, `--verbose true\|false`, `--omit_aas A,B` | `sample(...)` | 1, 4, 0.01, 2, true, none |
| `--sampling_overrides K=V …` | `sample(sampling_overrides={…})`, Hydra-style keys of the sampling config (e.g. `potts_sampling.n_sweeps=200`) | none |
| `--pos_constraint_csv F` | `sample(pos_constraint_df=pd.read_csv(F))` — upstream's constraint CSV (`pdb_key`, `fixed_pos_seq`, `fixed_pos_scn`, `fixed_pos_override_seq`, `pos_restrict_aatype`, `symmetry_pos`, …); on `ensemble32` expanded to the input's conformers (`get_ensemble_constraint_df`) | none: every position designed |
| `--clean_workers N` | `clean_pdbs(num_workers=)`; N > 1 is upstream's joblib-parallel clean, and under `fast` on `single` selects the kit's parallel cleaning instead | 1 |
| `--num_samples_per_pdb N`, `--pp_batch_size P` (`ensemble32`) | `generate_ensembles(num_samples_per_pdb=, batch_size=)`: conformers per input, conformers per Protpardelle-1c batch | 32, 8 |
| `--sampling_yaml_path Y`, `--max_num_conformers M`, `--include_primary_conformer`, `--use_primary_res_type` (`ensemble32`) | `generate_ensembles(sampling_yaml_path=)`; the members `ensemble_sample` is given; `ensemble_sample(use_primary_res_type=)` | upstream's partial-diffusion YAML, 32, true, true |
| `--seed S` | `lightning.seed_everything(S)` before `sample`, `S+i` per input for conformer generation | none: upstream's unseeded call |

## Stock exceptions

None. Every mode, `off` included, runs the three upstream packages as shipped.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `MODEL_PARAMS_DIR` | yes | — | the weights root above; unset, every verb refuses by name |
| `CALIBY_OPT`, `CALIBY_VARIANT` | no | unset | the mode / variant when `--mode` / `--variant` are absent; a flag that disagrees with its variable is refused (exit 2); an unknown value is refused by name |
| `MODEL_OPT` | no | this `caliby/` directory | where the package finds `opt/forward/` and `stock/PINS.json`; set by `run.sh` and the configs |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` / `H200` per config | the GPU class the configuration targets; a mismatch is reported, never refused |
| `MODEL_OPT_STATE` | no | `${XDG_CACHE_HOME:-$HOME/.cache}/caliby_opt` | out-of-tree state: `warm`'s scratch outputs |
| `MODEL_OPT_STACK_KEY` | no | computed, e.g. `torch2.6.0-cu124-sm90` | keys the JIT cache directory |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused | shared JIT-cache root: `TRITON_CACHE_DIR` becomes `$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton` unless already set to an existing location (the key names the torch / CUDA versions and the GPU's compute capability); a writable root gets that directory created at the first kit-mode compile; a root the process cannot write is only read — its `<stack key>` subtree is copied once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` and the run compiles and reads there, so one warmed root can be shared read-only between users; left unset, Triton's own `~/.triton/cache` is used |
| `HF_HUB_OFFLINE`, `TQDM_DISABLE`, `PYTHONDONTWRITEBYTECODE` | no | `1` | nothing fetched at run time; line-oriented logs; no `.pyc` in the tree |
| `PDB_MIRROR_PATH`, `CCD_MIRROR_PATH` | no | `""` | read by atomworks; empty for this use |
| `CALIBY_FAST_*`, `CALIBY_X_*` | no | unset | the lever switches; a kit mode exports its own set and refuses (exit 3) when any is already set differently, `off` when any is set at all |
