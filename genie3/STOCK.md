# Genie 3 — stock, as pinned

## Pin

Upstream: https://github.com/aqlaboratory/genie3 at commit `d77ae5ac04212ff1e8b29b585859a3244c614804` (version 0.0.1, no tag), shipped
under `stock/` as the source archive `genie3-d77ae5ac.tar.gz` (recipe and member list in `stock/PINS.json`); weights:
`pretrained/v1/checkpoints/step=600000.ckpt` and `pretrained/v1/config.yaml` from https://huggingface.co/yeqinglin/genie3 (0.35 GB,
sha256 in PINS.json), fetched by `bash run.sh install --weights DIR` through upstream's own downloader (`scripts/setup/download.sh`:
`hf download yeqinglin/genie3 --include 'pretrained/**'`); upstream's licence and notices as found in the archive. `stock/` is never
edited. `stock/PINS.json` is the machine-readable form of this page (read by `run.sh`, `stock/check_pins.py` and the package).
Besides the source tree the archive carries upstream's `LICENSE`, `README.md`, `setup.py`, `scripts/setup/`, the example request files
(`examples/`) and upstream's pre-processed binder-design example set `data/design/binder_design/binderbench/` — ten problem definitions and
the target coordinate files they name, taken from the wwPDB entries listed in each problem's `pdb_id` (wwPDB archive data, CC0 1.0; the
member-by-member account is in `THIRD_PARTY_NOTICES.md`); upstream's motif-scaffolding problem sets, the targets' FASTA / MSA files,
`scripts/problem` and `assets` are not carried (PINS.json `archive_recipe` `excluded`).

## Stack

Ubuntu 22.04, CUDA 12.6 runtime with cuDNN 9.5 (driver ≥ 525.60), Python 3.10.13, torch 2.7.1+cu126, triton 3.3.1, lightning 2.6.5,
numpy 2.2.6, scipy 1.15.3 — the full list is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it
(stock as the editable checkout `/opt/genie3`, this kit under `/kit`). Upstream declares torch==2.7.1, numpy >=2.0.2,<3, biopython <1.86,
lightning unversioned. Route C takes one of two recipes.

**Existing environment** — a Python 3.10 environment that already runs Genie 3 at the pin (a source checkout installed with upstream's
`pip install -e .` — e.g. `git clone https://github.com/aqlaboratory/genie3 genie3-src && git -C genie3-src checkout
d77ae5ac04212ff1e8b29b585859a3244c614804 && pip install -e ./genie3-src`): `export GENIE3_ROOT=<checkout>` if pip's editable-install record does not name it, then `bash run.sh install`
(kit + shared core editable — `pip install -e ../common/opt_core -e opt` — then `stock/check_pins.py`, which confirms the checkout
matches the pin file for file; a torch / lightning / numpy off this stack is reported, not refused).

**Fresh venv** — the pinned recipe, as `environment/Dockerfile` builds it. From the directory that holds `genie3/` (where README's first
Setup block leaves you; the paths below start with `genie3/`); any Python 3.10.x, the patch level is not checked; the venv occupies ≈5.6 GB; the lock header's `pip install -r … --src <dir>` is the one-pass form of the same lock:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git build-essential   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; } # uv 0.12.15 itself, once per user; installer digest-checked
uv venv --seed --managed-python --python 3.10 venv_g3 && . venv_g3/bin/activate   # uv's own released CPython 3.10 build, headers included;
                                                                  # the image runs 3.10.13. If you use apt/deadsnakes python instead of uv: python3.10 + python3.10-venv + python3.10-dev
pip install "$(grep '^pip==' genie3/environment/requirements.lock)"
grep -v -E '^(#|-e )' genie3/environment/requirements.lock > /tmp/g3-stack.txt && pip install --no-deps -r /tmp/g3-stack.txt
pip install --no-deps --no-build-isolation --src "$PWD/src" $(grep '^-e ' genie3/environment/requirements.lock)   # clones and installs stock at the pin: src/genie3
export GENIE3_ROOT=$PWD/src/genie3                                # the image sets it too (ENV); the kit and the README examples read it
export PYTHONHASHSEED=0 CFLAGS=-g0                                # the image's values (environment/Dockerfile ENV), exported to match it
```

Cards: `configs/h100.env`, `configs/a100.env` and `configs/h200.env` set the same variables
(the A100 and H200 files differ from the H100 one only in `MODEL_OPT_TARGET_GPU`; `h200.env` sources `h100.env`).
Route C also needs git (pip clones the stock checkout from its URL) plus a C compiler and the Python development headers, which `fast`'s first-run Triton compile uses (the shared
core's fused triangle-multiplication kernel and the Triton kernels `torch.compile` emits for the core) — git and build-essential on
Debian-family hosts, the headers come with the uv-managed interpreter (python3.10-dev only for an apt/deadsnakes python); `off` and `exact` compile
nothing. The image installs both.

## Route B (Apptainer) and the image's compile cache

Route B only. The image is read-only and the stock child runs with cwd = the checkout (`/opt/genie3`), where stock's Lightning
`Trainer` (built without `logger` or `default_root_dir`) creates `lightning_logs/` — run `--mode off` as `apptainer run --nv --writable-tmpfs …`
(or bind a host directory there: `-B $PWD/lightning_logs:/opt/genie3/lightning_logs`); the kit modes write nothing under the checkout.
A SIF built from `environment/apptainer.def` links `/opt/culib/libcuda.so` to the driver `--nv` binds and sets `TRITON_LIBCUDA_PATH` itself, so route B
needs nothing extra; the manual form below is only for an image converted straight from the Docker image (`apptainer build … docker-daemon://…` or
`docker-archive://…`) or for route C run under Apptainer. Triton links `fast`'s kernels with `-lcuda`: on a host whose linker cache lists only `libcuda.so.1`, make a directory holding a `libcuda.so` → `libcuda.so.1`
symlink (inside the container the driver library is `/.singularity.d/libs/libcuda.so.1`) and pass `--env TRITON_LIBCUDA_PATH=<that directory>`;
the image ships no CUDA stub libraries.
The image's compile cache (routes A and B): the image build takes an optional pre-filled compile cache `_jitcache/genie3-<stack key>-<card>-jit.tar` from the build context (the build is
identical without one); in the image `run.sh` uses `/opt/jit_cache` in place when it is writable, else seeds the JIT root from it once
(`MODEL_OPT_JIT_ROOT` when it is set and empty, otherwise `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`; layout `<root>/<stack key>/triton|inductor`; a non-empty
`MODEL_OPT_JIT_ROOT` is left as it is; a preset `MODEL_OPT_JIT_ROOT` the process cannot write is used read-only — with `--config <card>` its
`<stack key>` subtree is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (up to `MODEL_OPT_JIT_SEED_MAX_FILES` files, default 5000), otherwise the
read-only root is used as it is — nothing is compiled into it, caches already there are read, and no line is printed) and otherwise prints `[genie3-kit] jit cache: <dir> (<how>)`, which says which case applied.

## How stock is run

`--mode off` executes `genie3 generate -c <out_dir>/request.yaml --log-dir <out_dir>/logs [upstream's generate flags]` from the checkout
in a clean subprocess with nothing of the kit importable (`opt/genie3_opt/stock_cli.py`); `request.yaml` is the input request with
this pass's `--out_dir` / `--n` / `--selections` / `--seed` / `--batch_size` composed in, so every mode reads one request. Upstream
defaults the modes rely on: `experiment.seed` null (unseeded), `generation.dataset.batch_size` 1, sampler `ddim` with
`n_sample_step` 100. `--det 1` under `off`: `experiment.seed` (the request's, else 0) through `lightning.seed_everything(seed,
workers=True)`, as on the kit modes.

## Stock exceptions

None.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `GENIE3_ROOT` | no | pip's editable-install record of the `genie3` distribution | the checkout at the pin — the working directory of every pass; refused by name when neither resolves |
| `GENIE3_WEIGHTS` | no | `<checkout>/pretrained/v1` | the directory holding `checkpoints/step=600000.ckpt` + `config.yaml`; a missing file is refused by name |
| `GENIE3_OPT` | no | unset (= `fast` through run.sh / `genie3-opt`; inert at the stock command line) | the mode, in place of `--mode`; a `--mode` that disagrees with it is a usage error in run.sh (exit 2) |
| `GENIE3_OPT_AUTOLOAD` | no | on | `0` disables the `GENIE3_OPT=<mode> genie3 generate …` route |
| `MODEL_OPT_TARGET_GPU` | no | `H100` (`configs/h100.env`), `A100` (`configs/a100.env`), `H200` (`configs/h200.env`) | the card the configuration targets; a differing card is noted on the activation lines, not refused |
| `MODEL_OPT_STACK_KEY` | no | derived when a config is sourced (`torch<ver>-cu<ver>-sm<cc>`) | names the running stack in the pass's records; a stack it cannot read (no visible GPU) is refused when the config is sourced, exit 3; preset it (e.g. `MODEL_OPT_STACK_KEY=torch2.7.1-cu126-sm90`, the H100/H200 key on the pinned stack) to source a config on a host without a GPU, e.g. an image build step — the probe of the CUDA device is then skipped |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` silently supplies the private per-user root `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made 0700, owned by this user alone, no group/other write bit, not a symbolic link — otherwise refused by name on stderr) and keys both caches under it; without `run.sh`, Triton `~/.triton` and inductor `<that root>/inductor` (refused: a fresh private directory serves that process) | root of the compile caches `fast` fills on its first run: through `run.sh`, `<root>/<stack key>/triton` and `<root>/<stack key>/inductor` (the stack key of `MODEL_OPT_STACK_KEY`); through `genie3-opt` / the stock command line alone, `<root>/inductor`; must be writable when set; an already-set `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` takes precedence |
| `MODEL_OPT` | no | this `genie3/` directory (set by `run.sh` and the configs) | how the package locates the kit and the pins |
