# E1 — stock, as pinned

## Pin

Upstream: Profluent-AI E1 1.0.0, github.com/Profluent-AI/E1 at commit `bfd2620a602248499f3d2583d85a7ecddf0b6e02` (no tags),
installed unmodified as `E1 @ git+https://github.com/Profluent-AI/E1.git@bfd2620a602248499f3d2583d85a7ecddf0b6e02`; shipped under
`stock/` as the `git archive` of that commit (`E1-bfd2620a.tar.gz`) plus a readable copy of its scoring modules, `LICENSE`, `NOTICE`
and `ATTRIBUTION` under `stock/src/`. Every pin value is in `stock/PINS.json` (the kit reads the same values from
`opt/forward/engines/e1/kits/pins.py`; the two must agree or every route refuses). Weights: one `model.safetensors` per size —
`Profluent-Bio/E1-150m`, `E1-300m`, `E1-600m` at the revisions and sha256 digests in PINS.json — fetched by `bash run.sh install
--weights DIR` through upstream's own downloader (`huggingface_hub.snapshot_download`) and digest-checked, plus the Hugging Face hub
RMSNorm kernel `kernels-community/triton-layer-norm` at its pinned revision, whose package files ship in the tree (`stock/hub_kernel/`,
BSD-3-Clause, `SOURCE.md` and `LICENSE` beside them; per-file digests in PINS.json) and are copied by the same step into the cache
layout the `kernels` package reads — nothing is fetched for the kernel; files already present are kept and checked. The container images carry no weights: DIR is a directory you mount (`docker run -v …`, `apptainer
run --bind …`) and `HF_HOME` names it at run time; the image holds only the hub RMSNorm kernel snapshot (Triton source, staged at
build from `stock/hub_kernel` by `python -m e1_opt.weights --kernels-cache /opt/kernels_cache`, the directory `KERNELS_CACHE` names there). Layout: `$HF_HOME/hub/models--Profluent-Bio--E1-<size>/snapshots/<revision>/model.safetensors`, fetched by revision, so
the cache holds no `refs/main` for the model and upstream's repo-id form `--model-name Profluent-Bio/E1-<size>` does not resolve offline against it (pass the
snapshot directory, as run.sh does; huggingface_hub writes `refs/main` only when a model is fetched by name); the
kernel snapshot under `$KERNELS_CACHE` when set, else under `$HF_HOME/hub`. A weights file with another digest runs, with its digest
printed on the weights line. `stock/` is never edited.

## Stack

Debian 12, CUDA 12.8 (driver ≥ 570), Python 3.12, torch 2.8.0+cu128, triton 3.4.0, transformers 4.56.2, tokenizers 0.22.1,
kernels 0.11.0, flash-attn 2.8.3.post1 (the Dao-AILab release wheel for cp312 / torch 2.8 / cu12) — upstream's `pixi.lock`; the
full list is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it. One build serves every card:
nothing is compiled for a GPU at build time; Triton and `torch.compile` build kernels at first use, so a C compiler stays on PATH.
Outside the images (README route C), from the directory holding `e1/`, on a host with git, a C compiler and an NVIDIA driver ≥ 570
(no CUDA toolkit: the CUDA libraries come as wheels; Triton needs the Python headers at run time — uv's interpreter carries them):

    sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git ca-certificates curl  # bare Ubuntu host (root: no sudo); skip what you have
    command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; } # uv 0.12.15 itself, once per user; installer digest-checked
    uv venv --seed --managed-python --python 3.12 ~/e1-venv && . ~/e1-venv/bin/activate  # uv's own released CPython 3.12 build, never a system python (the image uses 3.12.10); if you use apt/deadsnakes python instead of uv: python3.12 + python3.12-venv + python3.12-dev; or an existing Python 3.12 environment already at the pin
    python -m pip install --no-deps -r e1/environment/requirements.lock  # every pin in one pass: torch 2.8.0 (cu128 wheels), the flash-attn release wheel, stock E1 from its git URL
    # next: README Setup, second block — cd e1, bash run.sh install --weights /weights/e1/hf_home, export HF_HOME=…, bash run.sh check …

The image additionally exports `PYTHONHASHSEED=0`, `CFLAGS=-g0`, `USE_FLASH_ATTN=1` (upstream's default attention route),
`HF_HUB_DISABLE_TELEMETRY=1`, `DISABLE_TELEMETRY=1` and `KERNELS_CACHE=/opt/kernels_cache`; none of them is required outside it.
The image build also takes an optional pre-filled compile cache `_jitcache/e1-<stack key>-jit.tar` from the build context and unpacks it
into `/opt/jit_cache/<stack key>/…` (the build is identical without one: no cache line is printed and kernels compile at first use);
in an image that ships one, `run.sh` uses it in place as
`MODEL_OPT_JIT_ROOT` if the variable is unset and the directory writable, seeds `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` from it once if it is
read-only (an Apptainer image), seeds a set-but-empty `MODEL_OPT_JIT_ROOT` from it once and leaves a populated one untouched, printing
`[e1-kit] jit cache: <dir> (in-image|seeded from image|user)` before the config keys the Triton and Inductor caches under `<root>/<stack key>/`.
A preset `MODEL_OPT_JIT_ROOT` the process cannot write is never compiled into: when `MODEL_OPT_STACK_KEY` is already exported and
`<preset root>/<stack key>` holds at most `MODEL_OPT_JIT_SEED_MAX_FILES` files (default 5000), `run.sh` seeds that subtree once into
`${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` and uses it (`(seeded from read-only root)`); otherwise the read-only root is used as is — nothing is
compiled into it, caches already there are read — and no line is printed. A SIF built from `environment/apptainer.def` links `/opt/culib/libcuda.so` to
the driver `--nv` binds and sets `TRITON_LIBCUDA_PATH` itself, so route B needs nothing extra; the manual form below is only for an image
converted straight from the Docker image (`apptainer build … docker-daemon://…` or `docker-archive://…`) or for route C run under
Apptainer. There, Triton links each newly
compiled kernel's launcher against an unversioned `libcuda.so`, which the image does not carry: on a host that lists only `libcuda.so.1`
a kit mode stops at its first compile with `/usr/bin/ld: cannot find -lcuda` (stock's hub RMSNorm kernel is Triton-compiled too).
Once on the host, `mkdir -p ~/.libcuda && ln -sf /.singularity.d/libs/libcuda.so.1 ~/.libcuda/libcuda.so` (`$HOME` is bound; the
target resolves inside the container), then add `--env TRITON_LIBCUDA_PATH=$HOME/.libcuda` to every `apptainer run` line; a host whose
`ldconfig -p` lists `libcuda.so` needs none of this, and Docker (`--gpus all`) mounts both names. pip builds stock E1 from its git URL with hatchling and the kit package with
setuptools in isolated build environments, fetching those build backends from the package index at install time (network
needed; the Dockerfile constrains hatchling to 1.32.0); the lock pins runtime packages only. `python stock/check_pins.py
--variant all --with-torch` is the full check of an environment once the weights are staged: package, stack, weights and kernel
snapshot against the pins; exit 3 on any drift.

## How stock is run

`--mode off` executes `python -m E1.tools.score --model-name <snapshot dir> --parent-path P --mutants-path M --output-path O`
(plus upstream's other options as given) in a clean subprocess with nothing of the kit importable: every `E1_OPT*`, `E1_KIT*`,
`E1_VARIANT`, `MODEL_OPT*` variable stripped and no kit directory on `sys.path`, stated on the subprocess's own
`[e1-opt stock] ENV-CLEAN ok …` line. The model loads as `E1ForMaskedLM.from_pretrained(<snapshot>, dtype=torch.float)` and
forwards under bf16 autocast; upstream engages flash-attn on the within-sequence layers (`USE_FLASH_ATTN` unset = on), the hub
Triton RMSNorm and compiled flex-attention on the global layers whenever they import — the pinned stack carries all three, and each
model process's `KERNELS` line says what it bound. `--det 1` under `off`: the recipe's environment (`CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`OPENBLAS_CORETYPE=Haswell`, `OPENBLAS_NUM_THREADS=1`, `NPY_DISABLE_CPU_FEATURES=…`) is set before the interpreter starts, then
seed 0 for `random` / `numpy` / `torch`, `torch.use_deterministic_algorithms(True)`, TF32 off for matmul and cuDNN, cuDNN autotuning
off, and the hub RMSNorm kernel's Triton autotune fixed to the (card class, size) pin at `E1Scorer` construction — applied to
stock's own objects, nothing else of the kit loaded. Values: `kits/pins.py` `DET_RECIPE` and `TRITON_AUTOTUNE_PIN`; the switch:
`opt/e1_opt/det.py`. `exact --det 1` runs the same recipe.

## Stock exceptions

None. Both modes run E1 1.0.0 exactly as released.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `HF_HOME` | yes, unless the weights sit in the default cache | `~/.cache/huggingface` | the Hugging Face cache holding the weights (layout above); nothing is fetched at run time, a missing weights file is refused |
| `KERNELS_CACHE` | no | unset (`$HF_HOME/hub`) | where the hub RMSNorm kernel snapshot lives; searched first when set |
| `E1_OPT` | no | unset (`run.sh` / `e1-opt` then use `exact`) | the mode; set it to engage a mode from the unchanged stock command or a program; a `--mode` that disagrees exits 2 |
| `E1_VARIANT` | no | unset | the model size, as `--variant`; a `--variant` / `--model-name` that disagrees exits 2 |
| `E1_OPT_DET` | no | `0` | as `--det`; `0` or `1` — any other value is rejected with its name: exit 2 (usage) from `run.sh` / `e1-opt`, the NOT ACTIVE line and exit 3 when the autoload or a program reads it from the environment |
| `MODEL_OPT` | no | the directory above `configs/` | this `e1/` directory; the package locates the kit tree (`opt/forward`) through it |
| `MODEL_OPT_TARGET_GPU` | no | `H100` (`h100.env`), `H200` (`h200.env`), `A100` (`a100.env`) | the card class the config targets (`H100`, `H200`, `A100`); a card of another class is named on the ACTIVE line, never refused |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | no | `1`, `1` | no fetch at run time |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root); without `run.sh`, or when that root is refused, Triton and Inductor keep their own default cache locations | a persistent root for the JIT caches; it must be writable when set — the caches are created under it per stack key (`torch<ver>-cu<ver>-sm<cc>`: GPU architecture and stack; the `cu` part is read from torch's `+cu…` version tag and reads `NA` where the installed torch carries none, as in the images — e.g. `torch2.8.0-cuNA-sm80`, while the ACTIVE line's `stack=` word, taken with torch loaded, says `cu128`; the directory is reused all the same) on first run: the config exports `MODEL_OPT_STACK_KEY` (from `e1_opt.stack.jit_cache_key`), `TRITON_CACHE_DIR=<root>/<key>/triton` and `TORCHINDUCTOR_CACHE_DIR=<root>/<key>/inductor`, each unless already set |
| `PYTHONDONTWRITEBYTECODE` | no | `1` | no `.pyc` written into the kit tree |

The values in the default column other than the library defaults are what `configs/<card>.env` exports; the config fills only what is
unset and exits 2 when `e1_opt` is not importable. `E1_OPT`, `E1_VARIANT` and `E1_OPT_DET` are never set by a config. The kit's
`e1_opt_autoload.pth` is what makes `E1_OPT=<mode> E1_VARIANT=<size>` act on the unchanged stock command; with `E1_OPT` unset or
`off` nothing of the kit is imported.
