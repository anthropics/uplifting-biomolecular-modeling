# OpenFold3 (OpenBind-0) — stock, as pinned

## Pin
Upstream: `openfold3` 0.5.0 = https://github.com/aqlaboratory/openfold-3 tag `v0.5.0` (`c4771653`), shipped under `stock/` as the PyPI wheel
`openfold3-0.5.0-py3-none-any.whl` plus the tagged source tree `stock/src/` (sha256 and sizes in `stock/PINS.json`; the wheel's `openfold3/`
package is byte-identical to `stock/src/openfold3/`). `run.sh install` places whichever is missing — the wheel from PyPI or `--wheel FILE`, the
tree as the repository archive of the pinned commit or `--src TARBALL` — each checked against `stock/PINS.json` first. Weights: OpenBind-0
`of3-ob-2025-06-30-174k.pt` (upstream's default checkpoint at 0.5.0; sha256 in PINS.json), fetched by `bash run.sh install --weights DIR` through
upstream's `download_model_parameters`; any other checkpoint runs with `WARNING: WEIGHTS unknown … proceeding` on the run's `WEIGHTS` line.
CCD: upstream's full `components.bcif` (sha256 in PINS.json), written over biotite's bundled subset by upstream's own `setup_biotite_ccd`
during install (`--ccd FILE` hands over a copy fetched beforehand); nothing fetches it at inference. Upstream's licence and notices as
found under `stock/`. `stock/` is never edited; `pred`, `check` and `warm` run `stock/check_pins.py` after the environment probes (package and core importable;
the core pin gate when a config is sourced) and before a mode is resolved or anything is predicted, `install` runs it right after placing the packages,
and a tree not at the pin exits 3.

## Stack
Ubuntu 22.04, CUDA 12.8 (driver ≥ 570), Python 3.11 (tested: 3.11.5), torch 2.10.0+cu128, triton 3.6.0, cuDNN 9.10, cuequivariance-torch 0.10.0,
pytorch-lightning 2.6.5, numpy 2.4.6, rdkit 2025.9.3, biotite 1.6.0, deepspeed 0.19.2 (installed with its DS4Sci evoformer-attention op built; the stock
configuration and every mode keep that op off) — the full list is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it. Into
an existing environment that already runs OpenFold3 at the pin (a released CPython 3.11 and a C compiler — build-essential on Debian-family hosts; uv's managed
interpreter carries its own headers, an apt/deadsnakes one needs python3.11-dev, not python3-dev; RDKit's X11 client libraries, libxrender1 libxext6 libsm6, on
a headless machine; beside the `openfold3` wheel at least cuequivariance-torch and cuequivariance-ops-torch-cu12 0.10.0 — the stock configuration here (`off`,
and the base of `exact` and `fast`) runs upstream's cuEquivariance triangle kernels and the wheel does not depend on them, so a run without them ends inside
upstream with no structures written; `environment/requirements.lock` is the whole pinned set; the CUDA 12.8 toolkit with `nvcc`, `CUDA_HOME` and CUTLASS as well
when a runner YAML turns DeepSpeed's DS4Sci evoformer attention on, because DeepSpeed then builds that op at its first call — the images carry all of this):
`bash run.sh install [--weights DIR]` — in this order: `pip install -e ../common/opt_core -e opt`; the stock wheel and source tree when absent
(`stock/install_upstream.py`); `stock/check_pins.py`, which confirms the installed `openfold3` matches the pin file for file; upstream's CCD step — the 63 MB
dictionary goes into this environment's own `biotite` package (fetched unless the copy there is already at the pin, so once per environment; a copy anywhere
else, e.g. beside the weights, is not consulted; offline: `--ccd FILE`); with `--weights DIR`, the checkpoint (2.3 GB, fetched when absent, only sha256-checked
when present). Other cards: `configs/<card>.env` sets `MODEL_OPT_TARGET_GPU`; the Dockerfile builds DeepSpeed's op for compute capability 9.0 (its
`TORCH_CUDA_ARCH_LIST` line), the route-C block below for the card it sees (`torch.cuda.get_device_capability()`: 8.0 on an A100, 9.0 on an H100 / H200).

```bash
# Route C — the pinned stack into a fresh Python 3.11 venv, the steps environment/Dockerfile runs; typed from the directory holding openfold3_ob0/ (cd below).
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl build-essential libxrender1 libxext6 libsm6 libgl1   # bare Ubuntu host; skip what you have
cd openfold3_ob0                                                  # README fence 1 leaves you in the directory holding openfold3_ob0/ and common/
# uv's own released CPython 3.11 (--managed-python: never a system interpreter; its headers come with it; the image runs 3.11.5).
# If you use apt/deadsnakes python instead of uv: python3.11 + python3.11-venv + python3.11-dev (not python3-dev), and not Ubuntu 22.04's own python3.11
# package (3.11.0rc1).
command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; } # uv itself, once per user (skipped when present)
uv venv --seed --managed-python --python 3.11 ~/of3-ob0 && . ~/of3-ob0/bin/activate
export CUDA_HOME=/usr/local/cuda PYTHONHASHSEED=0 CFLAGS=-g0       # the image's process environment; the DeepSpeed build reads CUDA_HOME
grep -v -E '^(#|openfold3==|deepspeed==)' environment/requirements.lock > /tmp/stack.txt
python -m pip install --no-deps -r /tmp/stack.txt                  # every pin (≈8.8 GB, minutes): torch 2.10.0+cu128, triton, cuDNN, cuEquivariance, CUTLASS …
python -I stock/install_upstream.py --wheel-only                  # the pinned openfold3 wheel (stock/ or PyPI, sha256-checked first), installed --no-deps
ARCH="${TORCH_CUDA_ARCH_LIST:-$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())' 2>/dev/null)}"; ARCH="${ARCH:-9.0}"
# ^ this card (A100 8.0, H100 / H200 9.0); a preset TORCH_CUDA_ARCH_LIST wins; 9.0 when no GPU answers; the image builds 9.0 only
CC=gcc CXX=g++ DS_ACCELERATOR=cuda DS_BUILD_EVOFORMER_ATTN=1 TORCH_CUDA_ARCH_LIST="$ARCH" \
  python -m pip install --no-deps --no-build-isolation --no-binary deepspeed deepspeed==0.19.2 # DeepSpeed + DS4Sci op from source: ≈13 min on 8 cores, quiet
python -m pip check
bash run.sh install --weights /weights                               # = README step 2's install line — type it once; step 2 continues at its export line
```

The image build takes optional pre-filled compile caches `_jitcache/openfold3_ob0-<stack key>-jit.tar` from the build context (`environment/Dockerfile` unpacks
any that are present into `/opt/jit_cache/<stack key>/…`; the build is identical without one). In the image `run.sh` uses `/opt/jit_cache` in place as the JIT
root when `MODEL_OPT_JIT_ROOT` is unset and the directory is writable, copies it once to `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` when it is read-only (an Apptainer
image), seeds a set but empty `MODEL_OPT_JIT_ROOT` from it once and leaves a non-empty one alone, then prints `[openfold3_ob0-kit] jit cache: <dir> (in-image |
seeded from image | user | unseeded)`; `configs/<card>.env` key `TRITON_CACHE_DIR` and `TORCH_EXTENSIONS_DIR` under `<root>/<stack key>/`. Outside an image, or
in one built without a tar, nothing is printed and the kernels compile on first use as before. A preset `MODEL_OPT_JIT_ROOT` this process cannot write is used
as it is — nothing is compiled into it, caches already there are read, no line is printed — unless `MODEL_OPT_STACK_KEY` is preset too and that key's subtree
holds at most `MODEL_OPT_JIT_SEED_MAX_FILES` files (default 5000): then the subtree is copied once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`, which becomes the root,
and the line reads `(seeded from read-only root)`.

## How stock is run
`--mode off` executes `run_openfold predict --runner-yaml <stock configuration> --inference-ckpt-path $OPENFOLD3_OB0_CKPT <your flags>` in a
clean subprocess with nothing of the kit importable (`opt/openfold3_ob0_opt/stock_pred.py`; the child proves no hook directory or lever
switch is on its path before it imports `openfold3`). The only addition is a timing wrap around `OpenFold3.forward` that prints
`Model forward time` (two device synchronisations per item, arithmetic untouched). `--det 1` under `off`: the same recipe as the kit modes,
applied inside the child through one start-up site directory — `torch.use_deterministic_algorithms(True)`, cuDNN deterministic,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, upstream's seed as given, runner YAML unchanged.

## Stock exceptions
None in code: upstream runs unmodified on every mode. `--upstream-fix ID[,ID]` is parsed on every arm, but this kit registers no fix (`upstream_issues/`
holds notes only), so any ID is refused by name as unknown before anything launches.
Configuration: the stock configuration is `opt/openfold3_ob0_opt/stock_cueq_bf16_notune_predict.yml` — upstream's `predict` preset with
`use_cueq_triangle_kernels: true`, `use_triton_triangle_kernels: true`, `use_deepspeed_evo_attention: false`, `tune_chunk_size: false`,
`chunk_size: 1024`, `pl_trainer_args.precision: bf16-mixed` (upstream's own options; `offload_inference` as the preset ships it).
`shipped_predict.yml` beside it is upstream's configuration exactly as shipped (Triton kernels only, `32-true`, chunk tuner on); named as
`--runner-yaml` under `off` it replaces the stock configuration as the base. A caller's `--runner-yaml` files are laid over the stock
configuration under `off` (your keys win) and under the mode's execution keys on the kit modes; the note line names what changed.
Templates: upstream consumes templates at inference on every mode and every route prints one `TEMPLATES DECLARED …` line; `--use-templates
false` on a query JSON that still carries template keys fails inside upstream (`UnpicklingError`) — remove the keys from an input meant to
run untemplated. `upstream_issues/OB0-001_templates_dropped_on_fetch_failure.md` describes the silent template drop on an offline host.

## Variables
| variable | required | default | effect |
|---|---|---|---|
| `OPENFOLD3_OB0_CKPT` | yes | — | the weights file (`--ckpt FILE` per call wins); `pred` / `warm` refuse by name without one |
| `OPENFOLD_CACHE` | no | the checkpoint's directory | upstream's cache root; upstream merges a `runner.yml` found there under any runner YAML — every route names such a file on stderr and passes the weights path explicitly |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused | writable persistent root for compiled-kernel caches, created under it per GPU / stack key on first run |
| `TRITON_CACHE_DIR` | no | Triton's own | `$MODEL_OPT_JIT_ROOT/<stack-key>/triton` when the root is set and this is not; pre-set wins; else Triton's default |
| `MODEL_OPT_STACK_KEY` | no | probed | `torch<ver>-cu<CUDA>-sm<cc>`; preset it (e.g. `torch2.10.0-cu128-sm90`) to source a config without a GPU (image build) |
| `MODEL_OPT_TARGET_GPU` | no | per config | the GPU class the configuration targets; a mismatch with the card is reported |
| `OPENFOLD3_OB0_OPT` | no | `fast` | the mode when `--mode` is not given; a `--mode` that disagrees with it is refused |
| `OPENFOLD3_OB0_OPT_N_GPU` | no | `1` | `--n_gpu` when the flag is not given |
| `OPENFOLD3_OB0_OPT_GRAPHS_MAX_TOKENS` | no | per line | the CUDA-graph size cap (`--graphs-max-tokens` wins): `<int>` polymer tokens, `always` (no cap) or `never` / `off` (no capture) |
| `OPENFOLD3_OB0_OPT_Z_DTYPE`, `OPENFOLD3_OB0_OPT_CONF_DTYPE` | no | per line | `bf16` \| `fp32`: `--z-dtype` / `--conf-dtype` when the flags are not given (the `z_dtype` / `conf_dtype` levers; `fp32` = upstream's) |
| `OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS` | no | `600` | `fast`: the smallest input (all tokens) whose diffusion roll-out runs in bf16; smaller inputs keep fp32, counted `gated=` |
| `OF3O_MIN_TOKENS`, `OPENFOLD3_OB0_OPT_REACH_GATE` | no | per card (`modes.py`) | `big`'s item gate and reach gate in polymer tokens (`<int>` or `none`) |
| `MODEL_OPT_LEVERS_OFF` | no | unset | comma-separated lever names left off for one run (README Notes) |
| `OF3TP_ADDR`, `OF3TP_PORT` | no | loopback, a free port | `big --n_gpu P`: the rank group's rendezvous; the other `OF3TP_*` run variables (row-block sizes, host budgets, `OF3TP_STRUCTURE_FIRST`) are declared in `opt/openfold3_ob0_opt/tp_rowpair/env.py` and printed on the ACTIVE line |

The caller-facing kit variables, plus the stock variables the kit reads (the per-lever `OPENFOLD3_OB0_OPT_*` / `OF3*_` switches the LEVER lines name are
declared in `opt/openfold3_ob0_opt/modes.py` and the add-ons); an `OPENFOLD3_OB0_OPT*` name the package does not declare, or a value it does not run, is refused by
name at interpreter start. Values in `configs/<card>.env` are deployment settings, never lever switches.
