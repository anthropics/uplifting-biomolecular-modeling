# Boltz-2 — stock, as pinned

## Pin

Upstream: `boltz` 2.2.1 = github.com/jwohlwend/boltz tag `v2.2.1` (commit `cb04aecc`), shipped under `stock/` as the PyPI wheel
`boltz-2.2.1-py3-none-any.whl` and the tag's `git archive` `boltz-cb04aecc.tar.gz` (digests in `stock/PINS.json`; `stock/src/` holds
excerpts for reading); entry point `boltz predict` (`boltz.main:cli`). Weights: the Boltz-2 cache — `boltz2_conf.ckpt` (sha256 in
`stock/PINS.json`, checked on every route's `WEIGHTS` line), `boltz2_aff.ckpt`, `ccd.pkl`, `mols.tar` unpacked to `mols/` — fetched by
`bash run.sh install --weights DIR` through upstream's own downloader (entries already there are kept and checked); terms: upstream releases code and weights under the MIT licence (its README, in the `git archive` under `stock/`). The cache directory must be
writable while `install --weights` completes it (upstream's downloader writes the missing entries and unpacks `mols/` in place); complete, it may be mounted
read-only (the kit's digest memo is then computed afresh and the `WEIGHTS` line says so); a read-only, partial pre-downloaded set: copy it to a
writable directory and run `install --weights` there. Upstream's licence and notices as found under `stock/`; the kit's third-party notices: `THIRD_PARTY_NOTICES.md`, licence texts in `third_party_licenses/`. `stock/` is never edited.

## Stack

Debian 12 (`python:3.11.12-slim-bookworm`), CUDA 13.0 from the lock's nvidia-* wheels (host driver 580 or newer), Python 3.11.12, torch
2.12.0+cu130, triton 3.7.0, cuequivariance / cuequivariance-ops-torch-cu13 0.10.0, pytorch-lightning 2.5.0, numpy 1.26.4 — the full list
is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it, and a C compiler must be on PATH at run time
(Triton). The image build takes optional pre-filled compile caches `_jitcache/boltz2-<stack key>-jit.tar` from the build context (made in a warmed
container: `tar -C /opt/jit_cache -cf … <stack key>`); the build is identical without one, and a builder that refuses an unmatched COPY glob needs an
empty archive there first (`tar -cf _jitcache/boltz2-empty-jit.tar -T /dev/null`). In the image `run.sh` uses `/opt/jit_cache` in place when it is
writable, else seeds `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` from it once (`[boltz2-kit] jit cache: <dir> (<how>)`). Into your own Python 3.11 environment instead — typed from the directory
holding `boltz2/` and `common/` (where the README's Setup block leaves you):

```bash
cd boltz2
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential gfortran curl ca-certificates   # bare Ubuntu host (root: no sudo); skip what you have — the Dockerfile's apt packages (C compiler for Triton, gfortran for source builds) plus curl for the next line
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); the installer runs only when its sha256 matches
uv venv --seed --managed-python --python 3.11.12 ~/boltz2-venv && . ~/boltz2-venv/bin/activate   # uv's own released CPython build, headers included; any released CPython 3.11 works instead (python.org, conda, or apt/deadsnakes python3.11 + python3.11-venv + python3.11-dev — not Ubuntu 22.04's python3.11 package, a 3.11.0rc1)
grep -E '^(pip|wheel)==' environment/requirements.lock > /tmp/pip.txt && python -m pip install --no-deps -r /tmp/pip.txt
printf 'setuptools==82.0.1\n' > /tmp/bc.txt && grep -v -E '^(#|pip==|wheel==|boltz==)' environment/requirements.lock > /tmp/stack.txt
# ^ build-time setuptools for the packages pip builds from source; the installed setuptools stays the lock's; --build-constraint needs the lock's pip (installed two lines up)
python -m pip install --no-deps --build-constraint /tmp/bc.txt -r /tmp/stack.txt
python -m pip install --no-deps stock/boltz-2.2.1-py3-none-any.whl
bash run.sh install --weights DIR   # = the README's next-block install line (DIR: your BOLTZ_CACHE directory — a writable copy of the weights, /weights/boltz2 in the images): kit + shared core editable (places boltz2_opt_autoload.pth), stock/check_pins.py (boltz at the pin, file for file), then the cache filled and checked
export LD_LIBRARY_PATH=$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cu13/lib && echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/boltz2-venv/bin/activate   # the image's ENV (the lock's CUDA 13 libraries for compiled extensions without their own search path); the echo makes every later `activate` re-export it
```

A first-time setup needs Python 3.11, an NVIDIA driver of the 580 series or newer and a C compiler on PATH; no CUDA toolkit (the CUDA 13.0
libraries are the lock's wheels). The image additionally sets `PYTHONHASHSEED=0` and `CFLAGS=-g0` (optional in your own environment). The shared core (`common/opt_core`) must be at or above the version in `opt/pyproject.toml` `[tool.opt_core]`; an older core is refused by
name at every entry. Cards: compute capability 8.0 and up (`configs/h100.env` and `configs/h200.env` target 9.0, `configs/a100.env` 8.0; the three files
differ only in `MODEL_OPT_TARGET_GPU` and their own name in messages); one stack serves both, the kit's Triton kernels compile per card into the stack-keyed JIT cache. On the A100 (8.0) `exact`'s fused pair transition has no per-call fallback: an input whose pair-row count (N² for N tokens) the shared core has not established as bitwise with cuBLAS here — N below 64, 1,024–1,099, 1,446–1,502, 1,772–1,818, 2,046–2,087, 2,288–2,324, 2,507–2,540, 2,708–2,738, 2,895–2,924 — completes but ends `NOT ACTIVE: fused transition gate refused: … exact_rows_…_on_cc_8.0`, exit 3; `fast` and `big` serve every size.

## How stock is run

`--mode off` executes `boltz predict <input> --out_dir <out_dir> [<the boltz predict options given, in the order given>]` in a clean
subprocess with nothing of the kit importable: upstream's CLI exactly as shipped, so the weights and CCD come from `$BOLTZ_CACHE` by
upstream's own `--cache` default, the checkpoint is upstream's own `<cache>/boltz2_conf.ckpt` resolution, `--model` defaults to `boltz2`,
and every setting not given keeps upstream's default (recycling_steps 3, sampling_steps 200, diffusion_samples 1, the cuEquivariance
triangle kernels on, bf16-mixed precision); an option not given is not passed. `pred` accepts `boltz predict`'s options by name on every mode with upstream's spelling and defaults
(`python -m boltz2_opt pred --help`); `--no_kernels` belongs to `off` only, and `--det 1` under `off` adds `--num_workers 1 --no_kernels`.

## Stock exceptions

None.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `BOLTZ_CACHE` | yes | — | the Boltz-2 cache directory; a missing file refuses every route by name; nothing is downloaded at run time |
| `BOLTZ2_OPT` | no | unset (`fast` when `--mode` is absent too) | the mode when `--mode` is not given; a `--mode` that disagrees is refused; any other `BOLTZ2_OPT*` name is refused by name |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` / `H200` from the config | the card class the configuration targets; `check` reports a mismatch as a note |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused | a persistent JIT-cache root, normally writable (a read-only one: see the end of this row): the compile caches are created under it per GPU / stack key on first run (`$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton` becomes `TRITON_CACHE_DIR` when that is not set itself); unset, the configs use the tree's own `.jit/` when it exists and is writable, and in the image `run.sh` takes the caches shipped under `/opt/jit_cache` (in place when writable, else a writable copy seeded once); else Triton's default cache; a preset root the process cannot write is used read-only: its `<stack key>` subtree is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (up to `MODEL_OPT_JIT_SEED_MAX_FILES` files, default 5000), otherwise the read-only root is used as is (nothing is compiled into it; caches already there are read); the printed `[boltz2-kit] jit cache: <dir> (<how>)` line says when a copy was seeded |
| `TRITON_CACHE_DIR` | no | Triton's own | Triton's JIT cache set directly; a pre-set value always wins |
| `MODEL_OPT_STACK_KEY` | no | derived | `torch<version>-cu<cuda>-sm<cc>` of the running machine (`boltz2_opt.modes.jit_cache_key`), e.g. `torch2.12.0-cu130-sm90`; a preset value is taken as is and the probe skipped — preset it (e.g. `MODEL_OPT_STACK_KEY=torch2.12.0-cu130-sm90` for the H100) to source a config on a host without a GPU, such as an image build step, where the probe would yield `unknown` |
| `MODEL_OPT_WEIGHTS_DIGEST_DIR` | no | parent of `TRITON_CACHE_DIR`, else `BOLTZ_CACHE` | where the weights sha256 memo `weights_digests.json` is written; a read-only directory is named on the `WEIGHTS` line |
| `MODEL_OPT_LEVERS_OFF` | no | unset | comma-separated words that take a lever off by name; `compile` is the word every verb's `--no-compile` flag adds — inert in this kit, which uses no `torch.compile` (`compile=off:none_in_kit`, or `compile=off:user` with the flag, on the ACTIVE line) |
| `BOLTZ_OPT_XFER` | no | `auto` | how the persistent featurizer hands a batch's CPU storages to the predicting process: `auto` = torch's shared memory below `BOLTZ_OPT_XFER_MIN_MIB`, a file under `BOLTZ_OPT_XFER_DIR` at and above it; `shm` = torch's transport for every storage; `file` = every storage by file; another word is refused by name |
| `BOLTZ_OPT_XFER_MIN_MIB` | no | `4096` | the storage size (MiB) from which `auto` uses a file |
| `BOLTZ_OPT_XFER_DIR` | no | `<out_dir>/_kit/_xfer` | where those files are written (created on first use, each unlinked once read, the directory removed at exit); without the worker's launch directory, `<tempdir>/bz2xfer_<uid>`; a directory without the room is refused by name |
| `BOLTZ_OPT_HELPER_TIMEOUT_S`, `BOLTZ_OPT_LOADER_TIMEOUT_S` | no | `3600`, `1800` | deadlines (s) on the featurizer helper's reply and on the stock loaders the kit builds; an expired wait fails by name (`fallback_by=helper_timeout…` on the LEVER line, or `[boltz2-opt prefetch] FAILED by name: …` and a non-zero exit), never hangs |
| `MODEL_OPT` | no | derived | this `boltz2/` directory; `run.sh` and the configs export it |
| `PYTHONHASHSEED` | no | `0` for the ranks when unset | `--n_gpu P` starts every rank under one value (`RANKENV hashseed=<v> source=default|inherited`) |
