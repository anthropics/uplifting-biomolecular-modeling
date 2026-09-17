# OpenFold3 — stock, as pinned

## Pin

Upstream: `openfold3` 0.4.1 = github.com/aqlaboratory/openfold-3 tag `0.4.1` (`d12f5955`), shipped under `stock/` as the PyPI wheel
`openfold3-0.4.1-py3-none-any.whl` plus the tagged source tree `stock/src/` (each file of the wheel's `openfold3/` package equals its counterpart there byte for byte; the
tree's 23 reference-molecule data files under `openfold3/core/data/resources/` are not packaged in the wheel — `stock/PINS.json` records both);
`stock/PINS.json` is the machine-readable pin; `stock/check_pins.py` refuses an install that is off it, file for file (exit 3) — every
`run.sh` verb runs it before the package is called, once `configs/<card>.env` is sourced and the interpreter found (`install`: after its pip step). Entry point `run_openfold predict` (`openfold3.run_openfold:cli`). Weights: `of3-p2-155k.pt` (OpenFold3
preview-2; sha256 and URL in `stock/PINS.json`), fetched by `bash run.sh install --weights DIR` through upstream's own S3 client and
digest-checked; every route hashes the file it is given and labels an unknown checkpoint on its `WEIGHTS` line (it still runs).
Upstream's licence and notices as found under `stock/`; the wheel and `stock/src/` also carry upstream's test data (`openfold3/tests/test_data/`:
PDB entries, sequence-search hits from UniRef90 / UniProtKB / MGnify / BFD, generated fixtures) and example inputs, whose sources and licences
`THIRD_PARTY_NOTICES.md` lists group by group. `stock/` is never edited.

## Stack

Ubuntu 22.04, CUDA 12.8 (driver ≥ 570), Python 3.11 (3.11.5 in the image), torch 2.10.0+cu128, triton 3.6.0, cuDNN 9.10.2.21, deepspeed 0.19.2 with the
DS4Sci evoformer-attention op prebuilt, pytorch_lightning 2.6.5, cuequivariance_torch 0.10.0, numpy 2.4.6 — the full list is
`environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it and end with `run.sh install`. Other cards: the
Dockerfile's `--build-arg DS4SCI_ARCHS="8.0;9.0"` adds A100 device code to the DS4Sci op (append `10.0` for B200, `10.3` for B300: one entry per compute capability the image must serve); `configs/<card>.env`
sets `MODEL_OPT_TARGET_GPU`; on such a card the DRY-RUN / ACTIVE line may end with `card_support=uncertified:<sm>(<levers>)` — those levers run the same code path as on H100 and the outputs are produced the same way; the word only records that their timing and identity runs were made on H100-class cards, not on this one (a statement, never a switch; absent when every requested lever has run on the card's class; `--det 1` with `exact` against `off` is the check to run where it matters). Into a released CPython 3.11 environment (the block's `-m venv` line makes one) — route C also needs the CUDA 12.8 toolkit (`nvcc`), a C compiler (`build-essential`
on Debian-family hosts: Triton and the DeepSpeed op compile on the machine; that line's uv-managed interpreter carries its own Python headers — with apt/deadsnakes Python instead of uv,
install python3.11 + python3.11-venv + python3.11-dev, not python3-dev), and on a headless host the system libraries
`libxrender1 libxext6` (`libsm6 libgl1`) that RDKit imports — the Dockerfile's `apt-get` line. From inside `openfold3/`:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl build-essential libxrender1 libxext6 libsm6 libgl1   # bare Ubuntu host (root: no sudo); skip what you have; the CUDA 12.8 toolkit comes from NVIDIA's repository (above)
command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; }   # uv itself, once per user (skipped when present)
uv python install 3.11 && "$(uv python find --managed-python 3.11)" -m venv ~/of3 && . ~/of3/bin/activate   # uv's own released CPython 3.11 (headers included; Ubuntu 22.04's apt
        # python3.11 is 3.11.0rc1) and the stdlib venv module — not `uv venv`, whose _virtualenv.pth import hook the kit takes for an installed kit hook, so every mode
        # refuses (NOT ACTIVE: a kit hook is already installed); run it with no venv active; the image runs 3.11.5, any released 3.11 works
grep -v -E '^(#|openfold3==|deepspeed==)' environment/requirements.lock > /tmp/stack.txt && python -m pip install --no-deps -r /tmp/stack.txt   # ≈5 GB of wheels, about 2 min
python -m pip install --no-deps stock/openfold3-0.4.1-py3-none-any.whl        # or: python -m pip install --no-deps openfold3==0.4.1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" PYTHONHASHSEED=0   # as the image's ENV: nvcc for the two source builds (this op, the FlashPairformer extension at first use); the hash seed A and B run with
ARCH="${TORCH_CUDA_ARCH_LIST:-$(cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null) && echo "$cc" | head -1)}"; ARCH="${ARCH:-9.0}"   # the visible card's compute capability (9.0 H100/H200 · 8.0 A100 · 10.0 B200 · 10.3 B300; 9.0 when no driver answers; "8.0;9.0" = one build for A100 and H100)
CC=gcc CXX=g++ DS_ACCELERATOR=cuda DS_BUILD_EVOFORMER_ATTN=1 TORCH_CUDA_ARCH_LIST="$ARCH" \
  python -m pip install --no-deps --no-build-isolation --no-binary deepspeed "$(grep -E '^deepspeed==' environment/requirements.lock)"   # compiles the op: about 8 min on 8 cores, only pip's 'still running...'
bash run.sh install [--weights DIR]                                               # kit + core editable, pin check; = README Setup step 2's install line — type it once; step 2 continues at its export line
```

Compile caches in the image (routes A/B; nothing of this applies to route C): the image build takes an optional pre-filled cache `_jitcache/openfold3-*-jit.tar` from the
build context (the build is identical without one) and unpacks it to `/opt/jit_cache`. `run.sh` uses that directory in place when it is writable; in a read-only Apptainer
image, or with `MODEL_OPT_JIT_ROOT` set to an empty directory, it seeds the JIT root from it once (`MODEL_OPT_JIT_ROOT`, default `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`; layout
`<root>/<stack key>/triton` and `…/torch_extensions`), leaves a populated root untouched, and prints `[openfold3-kit] jit cache: <dir> (in-image | seeded from image | user)`.
A preset `MODEL_OPT_JIT_ROOT` the process cannot write is used read-only: its `<stack key>` subtree is copied once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` when it holds at
most `MODEL_OPT_JIT_SEED_MAX_FILES` files (default 5000; printed `(seeded from read-only root)`; `TRITON_CACHE_DIR` / `TORCH_EXTENSIONS_DIR` follow the new root), otherwise
it stays in use as is (nothing is compiled into it, its caches are read, no line is printed). With neither an image cache nor a preset root nothing is printed.

## How stock is run

`--mode off` executes `run_openfold predict` through `opt/openfold3_opt/stock_pred.py` in a clean `python -I` subprocess with nothing of
the kit importable (confirmed before and after the call, else `NOT STOCK`, exit 3), under the stock configuration: upstream's predict
preset plus its two documented runner-YAML speed settings, `use_cueq_triangle_kernels: true` and `pl_trainer_args.precision: bf16-mixed`
(`opt/openfold3_opt/stock_cueq_on_predict.yml`). The one addition is the `Model forward time:` timer around `OpenFold3.forward`. Seeds,
samples, `--use-msa-server`, `--use-templates` as given; a `--runner-yaml` is laid under the stock configuration's kernel and precision
keys, each override printed (naming `opt/openfold3_opt/shipped_predict.yml` — upstream's preset with no runner YAML — selects it as the
base instead). `--det 1` under `off`: torch deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before torch is imported,
DS4Sci attention off (`stock_cueq_on_det_predict.yml`) — the same recipe every mode applies, and the setting under which `exact` equals `off`.

## Stock exceptions

None. `off` is upstream 0.4.1 as released; the opt-in upstream fixes (README §Known upstream issues) apply identically on every mode.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `OPENFOLD3_CKPT` | yes | — | the weights file every verb reads (`--ckpt FILE` per call instead) |
| `OPENFOLD_CACHE` | no | directory of `OPENFOLD3_CKPT` | upstream's checkpoint cache root |
| `MODEL_OPT_TARGET_GPU` | no | `H100` (per `configs/<card>.env`) | the GPU class the configuration targets; `check` / `pred` report a mismatch with the visible GPU and key the per-card gates below |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused | persistent root for compiled kernels; must be writable when set — `configs/<card>.env` exports `TRITON_CACHE_DIR=<root>/<stack-key>/triton` (one directory per GPU/stack key, created on first run) unless that is already set; unset, nothing is exported and Triton uses its own cache directory |
| `TRITON_CACHE_DIR` | no | Triton's own | set directly to place the Triton cache; a pre-set value wins |
| `MODEL_OPT_STACK_KEY` | no | probed (`torch<version>-cu<CUDA>-sm<cc>`) | the kernel-cache key, read from the running stack and GPU when `configs/<card>.env` is sourced; preset it (e.g. `MODEL_OPT_STACK_KEY=torch2.10.0-cu128-sm90`, the pinned stack on H100/H200) to source the config on a host without a GPU, e.g. an image build step |
| `OPENFOLD3_OPT` | no | `fast` | the mode when `--mode` is not given (a `--mode` that disagrees with it is refused) |
| `OPENFOLD3_OPT_N_GPU` | no | `1` | the GPU count when `--n_gpu` is not given (`--n_gpu` wins) |
| `OF3O_MIN_TOKENS` | no | `1401` on H100/A100 (`modes.OF3O_GATE_BY_CARD`) | `big`: polymer tokens from which the row-block units engage; `<int>` or `none` |
| `OPENFOLD3_OPT_REACH_GATE` | no | `4500` (`modes.REACH_GATE_BY_CARD`) | `big`: polymer tokens from which the trunk's full-size pair kernels step aside; `<int>` or `none` |
| `OPENFOLD3_OPT_GRAPHS_MAX_TOKENS` | no | the mode's own cap | `--graphs-max-tokens`: capture CUDA graphs only up to N polymer tokens (`0` never) |
| `MODEL_OPT_LEVERS_OFF` | no | unset | levers of the selected mode to leave off for one run (README Notes) |

Lever switches (`OPENFOLD3_OPT_*`, `OF3*`) are set by the mode, never by hand; a switch preset to a value that contradicts the selected
mode is refused by name. `configs/<card>.env` fills only what is unset.
