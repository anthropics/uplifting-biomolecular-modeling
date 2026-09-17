# AF3-torch (xfold on OpenFold3 weights) — stock, as pinned

## Pin

Upstream: xfold, https://github.com/Shenggan/xfold at commit `22bdeedfa309ef4ff6f9199910d8403915de69d6`, shipped under `stock/` as GitHub's
source archive of that commit (`stock/xfold-22bdeed.tar.gz`; `bash run.sh install` fetches it from GitHub when the tree arrives without it —
`stock/fetch_upstream.py`); upstream's Apache-2.0 licence and the per-file notices of its AlphaFold 3-derived files as found. Featurisation, the
output writers and the weight converter come from the AlphaFold 3 open-code fork https://github.com/sokrypton/alphafold3 at commit
`bc32b22ff5902e3daffd5d1f7203d7f2ab6cb997` (the JAX environment; CPU only). Weights: the public OpenFold3-preview2 checkpoint
(`of3-p2-155k.pt`, URL and sha256 in `stock/PINS.json`; published by the OpenFold3 authors, https://github.com/aqlaboratory/openfold-3, under the
Apache License 2.0 — `stock/PINS.json` `variants.p2.checkpoint.license`) converted to AlphaFold 3's parameter-record layout by the fork's own
`convert_of3_weights.py` → `of3_ported_weights.bin.zst` (sha256 in PINS.json), with the optional `of3_conventions.json` beside it;
`bash run.sh install --weights DIR --fetch` downloads, converts and digest-checks it, `bash run.sh install --weights DIR` checks a file already there.
One checkpoint; no variant switch. `stock/PINS.json` is the machine-readable pin; `stock/` is never edited.

## Stack

Host: Linux x86-64 with an NVIDIA driver that runs CUDA 13.0 (580 or newer). No CUDA toolkit is needed — torch's wheels carry the CUDA 13.0 runtime (the image
itself builds on `nvidia/cuda:12.6.3-base-ubuntu24.04`). Route C also needs git, gcc / g++ and make, `zlib1g-dev`, `zstd`, `patch`, and network to GitHub,
PyPI and the wwPDB (the fork's build fetches the chemical-components dictionary). Python 3.12 for the two environments: the recipe's
`uv venv --managed-python` lines fetch uv's own released CPython 3.12 build, headers included (the fork's extension build needs them; the image uses CPython
3.12.1; uv itself: the block's first line after the apt line installs it once per user). An apt / deadsnakes Python used instead of uv's needs `python3.12` +
`python3.12-venv` + `python3.12-dev` (Ubuntu 22.04's `python3.11` / `python3-dev` packages are a release candidate / 3.10's headers — not usable); python.org
and conda builds also work. `run.sh` itself runs on any Python ≥ 3.10 that has pip (a `uv venv` needs `--seed`) and imports neither torch nor jax.

The torch environment (`AF3_TORCH_PY`: torch 2.13.0 with CUDA 13.0, triton 3.7.1, xfold's requirements — `environment/requirements-torch.lock`) runs the model
process; the JAX environment (`AF3_TORCH_JAX_PY`: jax / jaxlib 0.10.2, alphafold3-open 3.1.4 built from the fork checkout at `AF3_TORCH_JAX_REPO` with
`stock/patches/04_of3_empty_template_restype_gap.diff` applied — `environment/requirements-jax.lock`) featurises and writes on CPU. `environment/Dockerfile`
is this recipe (Route A; `environment/apptainer.def` converts its image, Route B; `--build-arg WHEELS_FROM=prebuilt` installs the wheel and data files a
previous build left in `stock/wheels/`, sha256-checked against `stock/PINS.json` `reference.artefacts`, instead of compiling — they have no public URL). By
hand (Route C), from the directory holding `af3_torch/` and `common/`, with `uv` (or a released `python3.12 -m venv` and `pip` with the same arguments):

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates wget git build-essential zlib1g-dev zstd patch   # bare Ubuntu host (root: no sudo); skip what you have
command -v uv >/dev/null || { f=$(mktemp) && wget -qO "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); wget from the apt line; the installer runs only when its sha256 matches
uv venv --managed-python --python 3.12 /torch_venv                         # 1. the torch environment: uv's own released CPython, the lock, no resolver
grep -v '^#' af3_torch/environment/requirements-torch.lock > /tmp/torch.txt
uv pip install --python /torch_venv/bin/python --no-deps -r /tmp/torch.txt
uv venv --managed-python --python 3.12 /alphafold3_venv                    # 2. the JAX environment: the lock minus the fork's own wheel
grep -v -E '^(#|alphafold3-open==)' af3_torch/environment/requirements-jax.lock > /tmp/jax.txt
uv pip install --python /alphafold3_venv/bin/python --no-deps -r /tmp/jax.txt
git clone https://github.com/sokrypton/alphafold3 /app/alphafold            # 3. the fork at the pin, patched, built (minutes on all cores), installed
git -C /app/alphafold checkout bc32b22ff5902e3daffd5d1f7203d7f2ab6cb997 && rm -rf /app/alphafold/.git   # without .git the wheel builds as 3.1.4
patch -d /app/alphafold -p1 < af3_torch/stock/patches/04_of3_empty_template_restype_gap.diff
grep -E '^numpy==' /tmp/jax.txt > /tmp/constraints.txt
CC=gcc CXX=g++ uv build --wheel --python 3.12 --build-constraint /tmp/constraints.txt --out-dir /tmp/wheels /app/alphafold
uv pip install --python /alphafold3_venv/bin/python --no-deps /tmp/wheels/alphafold3_open-3.1.4-cp312-cp312-linux_x86_64.whl
/alphafold3_venv/bin/build_data                                            #    required: AlphaFold 3's chemical-components data step inside the installed package
export AF3_TORCH_PY=/torch_venv/bin/python AF3_TORCH_JAX_PY=/alphafold3_venv/bin/python AF3_TORCH_JAX_REPO=/app/alphafold PYTHONHASHSEED=0 CFLAGS=-g0
uv venv --seed --managed-python --python 3.12 /kit/venv && . /kit/venv/bin/activate   # 4. the python run.sh installs into and runs on (any released CPython >= 3.10 with pip)
cd af3_torch && bash run.sh install [--weights DIR [--fetch]]   # 5. = README step 2's install line — type it once; step 2 continues at its export line
```

The paths are the image's; any three of your own do. `PYTHONHASHSEED=0` and `CFLAGS=-g0` are the image's run-time environment (the kit sets the fork's XLA
variables and `JAX_PLATFORMS=cpu` on the JAX processes itself). The pin check (`stock/check_pins.py`, run by `install` and quietly before every other verb)
asks both interpreters for torch / triton and jax / jaxlib and exits 3 naming the package or the unset variable when one is off the pin. `bash run.sh stock`
additionally needs the stock venv: `python -m af3_torch_opt.stock_venv --dest DIR` composes xfold's own CLI onto a venv built from the two environments;
`AF3_TORCH_STOCK_PY=DIR/bin/python`. Cards: `configs/h100.env`, `configs/h200.env` (compute capability 9.0) and `configs/a100.env` (8.0, the 80 GB and 40 GB
parts) differ only in the `AF3_TORCH_GPU` word; one image serves all three, nothing is compiled for a GPU at build time. The image build takes optional
pre-filled compile caches `_jitcache/af3_torch-<stack key>[-<card>]-jit.tar` from the build context, unpacked under `/opt/jit_cache/<stack key>/` (the build
is identical without one). `run.sh` settles the JIT root before it sources the config: in the image it uses `/opt/jit_cache` in place when it is writable,
else copies it once to `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (or into an empty `MODEL_OPT_JIT_ROOT`); a preset `MODEL_OPT_JIT_ROOT` the process cannot write is
never compiled into — its `<stack key>` subtree is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` when `MODEL_OPT_STACK_KEY` names it and it holds at most
`MODEL_OPT_JIT_SEED_MAX_FILES` files (default 5000), otherwise the read-only root is used as it is (nothing is compiled into it; caches already there are
read); the printed `[af3_torch-kit] jit cache: <dir> (<how>)` line says which. The config then points `AF3_TORCH_CACHE_ROOT` at `<that root>/<stack key>`
(`torch2.13.0-cu130-sm90` on compute capability 9.0, `torch2.13.0-cu130-sm80` on 8.0), so a shipped cache is used where it lies; a root that still holds no
compiled kernel is seeded by the kit from the image's tree on the first `pred` / `warm`, which says so on its `CACHE cache_seed=…` line.

## How stock is run

There is no upstream program for these weights: stock on this engine is xfold plus the port below. `--mode off` runs it through the kit's
process chain (featurise → forward → postprocess) with no lever: fp32 weights under bf16 autocast, xfold's fastnn kernels on (the stock CLI's
`--fastnn` default), xfold's 5 diffusion samples, each input at its own token count, in a model process whose environment carries no
`AF3_TORCH_OPT*` variable — the run's `STOCK` line states it. `bash run.sh stock` runs xfold's own `run_alphafold.py` (the archive's bytes) at its
shipped defaults on the stock venv; it takes no mode.

## Stock exceptions

- The port (`opt/forward/af3t/af3_torch/xfold/`; `stock/PINS.json` `upstream.port` lists the changed and added files, every other file is the
  archive's byte for byte): the OpenFold3 parameter layout and loader (`of3.OF3 = True`: the diffusion transformer's per-block pair LayerNorm /
  Linear, the symmetric bond contact matrix, 0-indexed reference elements, padded key atoms masked out of the atom cross-attention offsets, the
  transposed pair bias of the column-wise attention; `params.py` reads the converted records and raises on a missing key); two statements of the
  fork's atom cross-attention encoder mirrored (query single conditioning masked by the query mask; the keys' `ref_space_uid` gathered from the
  queries layout); the four corrections of README §Known upstream issues; and the entry points the `hoist` / `stepgraph` levers call, inert unless a
  mode enables them. Applied identically on every mode, `off` and `bash run.sh stock` included.
- `stock/patches/04_equivalent_template_gap.diff` — absent template slots are featurised as GAP (restype 21, the OpenFold3 convention the weights
  were trained with) instead of ALA in `xfold/nn/template.py`; the fork-side twin `04_of3_empty_template_restype_gap.diff` does the same in the
  fork's template network. Every mode.
- `stock/patches/05_oom_propagates_through_kernel_adapters.diff` — the kit's kernel adapters (`opt/forward/af3t/kernels/af3_kernels.py`) re-raise a
  CUDA out-of-memory error instead of taking their kernel-error route to the stock statement. Every mode; changes no arithmetic.
- The fork's XLA variables (`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, `XLA_PYTHON_CLIENT_PREALLOCATE=true`, `XLA_CLIENT_MEM_FRACTION=0.95`)
  are set on the JAX-environment processes, which run on CPU (`JAX_PLATFORMS=cpu`); the torch process owns the GPU.

## Variables

| variable | required | default (as `configs/<card>.env` sets it when unset) | effect |
|---|---|---|---|
| `AF3_TORCH_PY` | yes | — (image: `/torch_venv/bin/python`) | the torch environment's interpreter: the model process |
| `AF3_TORCH_JAX_PY` | yes | — (image: `/alphafold3_venv/bin/python`) | the JAX environment's interpreter: featurisation, the writers, the weight converter |
| `AF3_TORCH_JAX_REPO` | yes | — (image: `/app/alphafold`) | the fork checkout holding `run_alphafold.py` and `convert_of3_weights.py` |
| `AF3_TORCH_PARAMS_DIR` | yes (pred, check, warm) | — | the directory holding `of3_ported_weights.bin.zst` (else its one `*.bin.zst` / `*.bin`) |
| `AF3_TORCH_CACHE_ROOT` | no | `<MODEL_OPT_JIT_ROOT>/<stack key>` (run.sh exports that root); the config sourced by hand without it: `${TMPDIR:-/tmp}/af3_torch_cache-uid<uid>/<stack key>`, the per-user directory made owner-only and used only when this user owns it and neither a symbolic link nor group- or other-writable stands there — otherwise refused by name and a fresh private directory serves that shell (an explicit value wins; no config sourced: `~/.cache/af3_torch_opt`) | Triton / Inductor / JAX cache root (`<root>/triton`, `/inductor`, `/jax`); must be writable: the caches are created under it on a mode's first run per GPU and stack; the shared core's native triangle-attention row (lever `triattn`) keeps its gate memo beside them, or under `MODEL_OPT_JIT_ROOT` when that is set |
| `AF3_TORCH_STOCK_PY` | `stock` only | — (unset: `stock` prints `NOT ACTIVE reason=…` and exits 3) | the stock venv's interpreter (`opt/af3_torch_opt/stock_cli.py`): `DIR/bin/python` of the venv `python -m af3_torch_opt.stock_venv --dest DIR` built; an interpreter, or a directory holding it, that belongs to another account (neither you nor root) or that others can write is refused by name (exit 3) |
| `AF3_TORCH_GPU` | no | `H100` (`configs/h100.env`), `H200` (`configs/h200.env`), `A100` (`configs/a100.env`) | the target card word, reported as `gpu_target` on the ACTIVE line; read for the compute capability only when `nvidia-smi` cannot name the card |
| `AF3_TORCH_OPT` | no | `fast` | the mode when `--mode` is not given; a `--mode` that disagrees is refused (exit 2) |
| `AF3_TORCH_OPT_HOME` | no | the package's own tree | the kit directory (`stock/`, `opt/`) for a non-editable install used without run.sh (run.sh sets `MODEL_OPT`) |
| `AF3_TORCH_OPT_KIT`, `AF3_TORCH_OPT_DTK` | no | `opt/forward/af3t`, `opt/forward/dtk` | another model / DTK tree |
| `MODEL_OPT_LEVERS_OFF` | no | unset | comma list of levers dropped from the mode for one run (README §Notes) |

Any other `AF3_TORCH_OPT*` name, and any `AF3_TORCH_BIG_*` name, is refused by name (exit 3). The kit sets `TRITON_CACHE_DIR`,
`TRITON_CACHE_AUTOTUNING` and, under `big`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the model process itself.
