# Flashzoi — stock, as pinned

## Pin

Upstream: `borzoi-pytorch` 0.5.1 (https://github.com/johahi/borzoi-pytorch, tag `v0.5.1`, commit `8a05eb15dd56871806771b2870c98a0c1dd9e0c8`),
shipped under `stock/` as the PyPI wheel `borzoi_pytorch-0.5.1-py3-none-any.whl` with its source unpacked beside it (`stock/src/`); every
pin below is in `stock/PINS.json`, and `stock/check_pins.py` compares the installed package file for file with the wheel's own RECORD.
Weights: the four replicate checkpoints `johahi/flashzoi-replicate-{0,1,2,3}` (`model.safetensors` at the revision and sha256 listed per
repository under PINS.json `"weights"`, `config.json` at the top-level `"config_json_sha256"`),
fetched by `bash run.sh install --weights DIR` through `huggingface_hub` into the hub cache `DIR`
(`models--johahi--flashzoi-replicate-<k>/snapshots/<revision>/`) and digest-checked — files already in `DIR` are kept and only hashed (nothing is written, so a
read-only hub cache of that layout works with or without `--weights`; `FLASHZOI_WEIGHTS` then points at it); both routes load them offline through one loader
(`opt/flashzoi_opt/weights.py`) that hashes each file at load and records the digest in `opt_manifest.json` (a digest off its pin is
named on the line, not refused; a repository without a pin or a missing file refuses by name). The weights are published by their
author on Hugging Face; the terms that apply to them are those stated on the four model cards (`license: mit` at the pinned
revisions). Upstream's licence and notices as found in the wheel. `stock/` is never edited.

## Stack

Debian 12, CUDA 12.4 user space from the `nvidia-*-cu12` wheels (host driver 550 or newer), Python 3.11, torch 2.5.1+cu124, triton
3.1.0, flash-attn 2.7.0.post2, transformers 4.57.6, cuDNN 9.1.0.70 — the full list is `environment/requirements.lock`;
`environment/Dockerfile` and `apptainer.def` build it (a C compiler is part of it: Triton builds the kernels' launchers at first use).
One build serves H100, H200 and A100; nothing is compiled for a GPU at build time. The README's route C is this block in a fresh Python 3.11 environment
(line 1; an existing one at the pin also works), in this order — the stack pinned and `--no-deps` first, then the stock wheel, then the kit.
From inside `flashzoi/` (README fence 1 leaves you in its parent directory):

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y curl build-essential   # bare Ubuntu/Debian host (as root: no sudo); skip what you have — what environment/Dockerfile adds to its base image
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user; installer digest-checked
uv venv --seed --managed-python --python 3.11 fz-env && . fz-env/bin/activate   # uv installs and uses its own released CPython 3.11 with its headers, ignoring system pythons; any other released CPython 3.11 also works (python.org, conda; the image uses 3.11.12) — if you use apt/deadsnakes instead of uv: python3.11 + python3.11-venv + python3.11-dev (Triton compiles its launchers against Python.h), and not Ubuntu 22.04's own python3.11 package (3.11.0rc1)
grep -v -E '^(#|borzoi-pytorch==)' environment/requirements.lock > /tmp/stack.txt && python -m pip install --no-deps -r /tmp/stack.txt
python -m pip install --no-deps stock/borzoi_pytorch-0.5.1-py3-none-any.whl
bash run.sh install            # = README step 2's install line — type it once; step 2 continues at its export line (pip install -e opt: package flashzoi_opt, console script flashzoi-opt, the FLASHZOI_OPT autoload .pth; then stock/check_pins.py)
export PYTHONHASHSEED=0 CFLAGS=-g0   # the two variables the image sets in its environment (Dockerfile ENV); export them in yours for parity
```

The image builds `FROM python:3.11.12-slim-bookworm` with `build-essential` added — the reference for a host install is therefore Debian 12 with
gcc, Python 3.11 and an NVIDIA driver 550 or newer; the flash-attn wheel in the lock is the prebuilt cp311 / torch 2.5 / cxx11abi-FALSE build.
The image build also takes an optional pre-filled compile cache, `_jitcache/flashzoi-<stack key>-jit.tar` in the build context (any
`flashzoi-*-jit.tar` there; the build is identical without one), unpacked to `/opt/jit_cache/<stack key>/…`. In an image built with one, `run.sh` uses
`/opt/jit_cache` in place when it is writable and `MODEL_OPT_JIT_ROOT` is unset, else seeds the JIT root from it once (`MODEL_OPT_JIT_ROOT` when
set and empty, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` when the image directory is read-only, as under Apptainer; a populated root is left as it is),
points `TRITON_CACHE_DIR` at `<root>/<stack key>/triton` unless it is already set, and prints `[flashzoi-kit] jit cache: <dir> (<how>)`; an image built
without one prints no such line and Triton compiles into `TRITON_CACHE_DIR` — `<root>/<stack key>/triton`, the root being `MODEL_OPT_JIT_ROOT` or, when that is
unset, the private per-user root `run.sh` silently supplies (Triton's own `~/.triton/cache` only when that root is refused). A preset
`MODEL_OPT_JIT_ROOT` the process cannot write is used read-only: with `--config` (which names the stack key) its `<stack key>` subtree is seeded
once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (up to `MODEL_OPT_JIT_SEED_MAX_FILES` files, default 5000), otherwise the read-only root is used as it is (nothing is compiled into it; caches already there are read) and no line is printed;
when the root moves, `TRITON_CACHE_DIR` follows it unless it was set to some other path, and the printed line says which case applied.

`python stock/check_pins.py [--package-only] [--weights] [--hf-home <dir>] [--quiet]` confirms the installed `borzoi-pytorch` matches
the wheel file for file and the stack matches the table (a torch local tag such as `+cu124` is ignored); `--weights` also hashes the
weight files. `run.sh` runs it on every route, `off` included: `borzoi-pytorch` off its pin refuses (exit 3 — another upstream release
is another stock); any other stack difference is printed as a `NOTE`, the run goes on, and a kit mode repeats it on its ACTIVE line as
`drift=[stack …]`.

A SIF built from `environment/apptainer.def` links `/opt/culib/libcuda.so` to the driver `--nv` binds and sets `TRITON_LIBCUDA_PATH` itself, so
route B needs nothing extra; the manual form below is only for an image converted straight from the Docker image
(`apptainer build … docker-daemon://…` or `docker-archive://…`) or for route C run under Apptainer.
Under Apptainer (README route B, `apptainer run --nv`), the host's NVIDIA driver libraries are bound into the container as the host's
`ldconfig` lists them (at `/.singularity.d/libs/`). Triton 3.1.0 links each kernel launcher it compiles with `-lcuda`, which needs an
unversioned `libcuda.so` at link time, and the image carries no CUDA toolkit stub of it; on a host that lists only `libcuda.so.1` the first
kernel launch of either mode therefore fails with `/usr/bin/ld: cannot find -lcuda`. Point Triton at a directory holding a `libcuda.so` —
a symlink to the bound driver library is enough, it is used at link time only and the launcher loads `libcuda.so.1` at run time:

```bash
mkdir -p $HOME/.libcuda && ln -sf /.singularity.d/libs/libcuda.so.1 $HOME/.libcuda/libcuda.so     # once, on the host ($HOME is bound; the target resolves inside the container)
export FLASHZOI_WEIGHTS=/weights/flashzoi                                                              # on the host: Apptainer passes the variable in; the bind below makes the directory visible
mkdir -p out && apptainer run --nv --bind /weights/flashzoi:/weights/flashzoi --bind "$PWD/out":/kit/flashzoi/out --env TRITON_LIBCUDA_PATH=$HOME/.libcuda flashzoi-kit.sif pred --config h100 --mode exact --input $PWD/in --out out/exact   # the README's route-B form plus the --env
```

A host whose `ldconfig -p` lists `libcuda.so` needs none of this; Docker (route A, `--gpus all`) mounts both names.

## How stock is run

`--mode off` executes upstream's documented route in a clean subprocess with nothing of the kit importable and `FLASHZOI_OPT`
stripped (`opt/flashzoi_opt/stock_pred.py`, lines prefixed `[flashzoi-stock]`):

    models = [Borzoi.from_pretrained(f"johahi/flashzoi-replicate-{k}", revision=<pin>).to("cuda").eval() for k in range(4)]
    with torch.autocast("cuda"):
        y = predict_tracks(models, sequence_one_hot, slices)     # borzoi_pytorch.pytorch_borzoi_helpers; (1, 4, 6144, 7611) float32

One window per call, replicates 0..3 in order, fp16 autocast on trunk and transformer with the model's own fp32 head, FlashAttention-2,
PyTorch-default numerics (cuDNN TF32 on, matmul TF32 off). It writes the same `<item>.npy`, `rows.jsonl` and `opt_manifest.json`
(mode `off`, with the subprocess's environment recorded under `stock_env_proof`) as the kit route. `--det` on either route:
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialises (`:16:8` also accepted), `torch.manual_seed(0)`,
`torch.use_deterministic_algorithms(True)`, cuDNN deterministic on and its autotuner off; TF32 switches stay at PyTorch's defaults;
recorded in `opt_manifest.json` under `det`.

## Stock exceptions

None. No setting or patch is applied to upstream on any route.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `FLASHZOI_WEIGHTS` | yes | — | the hub cache `install --weights` filled; `configs/<card>.env` exports `HF_HUB_CACHE=$FLASHZOI_WEIGHTS` and `HF_HOME=$(dirname …)` from it (either may be exported directly instead); none named → `NOT ACTIVE: FLASHZOI_WEIGHTS is not set …`, exit 2 |
| `FLASHZOI_OPT` | no | unset | the mode for unchanged scripts (`exact` or `off`); unset or empty selects the package default, `exact`; on `run.sh` a `--mode` that disagrees with it exits 2; stripped from the stock subprocess |
| `MODEL_OPT` | no | this `flashzoi/` directory | where the package finds the kit tree and `stock/PINS.json`; `run.sh` and the configs export it |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused | persistent cache root: `TRITON_CACHE_DIR=<root>/$MODEL_OPT_STACK_KEY/triton` when `TRITON_CACHE_DIR` is unset; must be writable when set — the compile cache is created under it per stack key (torch, CUDA, GPU capability) on first run; left unset, the kernels compile into `TRITON_CACHE_DIR` if set, else Triton's default below |
| `TRITON_CACHE_DIR` | no | `<root>/<stack key>/triton` through `run.sh` (see `MODEL_OPT_JIT_ROOT`); Triton's own (`~/.triton/cache`) otherwise | where the kernels compile once and load thereafter; a pre-set value is kept |
| `MODEL_OPT_STACK_KEY` | no | derived, e.g. `torch2.5.1-cu124-sm90` | `torch<version>-cu<cuda>-sm<cc>`, the key of the persistent kernel cache; the configs derive it through torch from the running stack and the CUDA device unless it is preset — preset it (e.g. `MODEL_OPT_STACK_KEY=torch2.5.1-cu124-sm90` for an H100 or H200, `torch2.5.1-cu124-sm80` for an A100) to source a config on a host without a GPU, e.g. an image build step; neither preset nor derivable → `NOT ACTIVE: MODEL_OPT_STACK_KEY could not be derived …`, exit 2 |
| `HF_HUB_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`, `PYTHONDONTWRITEBYTECODE` | no | `1`, `1`, `1` (set by the configs) | nothing fetched at run time; no `.pyc` in the tree |

The kit itself reads no other switch. TF32 override variables (`NVIDIA_TF32_OVERRIDE`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`) are never
refused: both routes run under them as the libraries define, and a kit mode names them on its ACTIVE line (`drift=[NVIDIA_TF32_OVERRIDE=0]`)
and in `opt_manifest.json`.
