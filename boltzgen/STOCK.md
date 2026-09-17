# BoltzGen — stock, as pinned

## Pin

Upstream: `boltzgen` 0.3.2 = github.com/HannesStark/boltzgen tag `v0.3.2` (commit `31d9d9b9`), shipped under `stock/` as the package-index wheel
`stock/boltzgen-0.3.2-py3-none-any.whl` (sha256 in `stock/PINS.json`) plus, for reference, the twelve upstream modules the levers patch at run time
(`stock/src/`, equal to the tag, never imported); weights: the checkpoints `boltzgen1_diverse`, `boltzgen1_adherence`, `boltzgen1_ifold`,
`boltz2_conf_final`, `boltz2_aff` and `mols.zip` in Hugging Face cache layout (`models--boltzgen--boltzgen-1/snapshots/<snapshot>/*.ckpt`,
`datasets--boltzgen--inference-data/snapshots/<snapshot>/mols.zip`; snapshot ids and per-file sha256 in `stock/PINS.json` `weights`), fetched by
`bash run.sh install --weights DIR` through upstream's own downloader; upstream's MIT licence and notices as found in the wheel. `stock/` is never edited.
`stock/PINS.json` is the machine-readable pin the package reads (`opt/boltzgen_opt/stack.py`): a kit mode refuses an installed `boltzgen` off the pin (exit 3).

Upstream's entry point: `boltzgen run <spec.yaml> --output <dir> …` = `configure` (writes `<dir>/steps.yaml` and `<dir>/config/<step>.yaml`), then one
subprocess `python <site-packages>/boltzgen/resources/main.py <dir>/config/<step>.yaml` per step with `BOLTZGEN_PIPELINE_STEP=<step>`; unseeded.
Configure defaults the kit relies on: `--protocol protein-anything`; six steps (`design, inverse_folding, design_folding, folding, analysis, filtering`);
`--num_workers 1`; `--diffusion_batch_size` 1 below 100 designs and 10 from 100; design step `matmul_precision: high`, 3 recycles, 500 sampling steps.
Accelerators: `--use_kernels auto|true|false` (default `auto` = on at compute capability ≥ 8) selects cuEquivariance 0.11.1 triangle attention and
triangle multiplication; the library routes calls of ≤ 100 tokens to its PyTorch path (`CUEQ_TRIATTN_FALLBACK_THRESHOLD`, `CUEQ_TRIMUL_FALLBACK_THRESHOLD`);
`--config design compile_pairformer=true` / `compile_structure=true` switch `torch.compile` on (off by default).

## Stack

Ubuntu 22.04, CUDA 13.0 (driver ≥ 580), Python 3.11, torch 2.13.0+cu130 (cuDNN 9.20), triton 3.7.1, cuequivariance / cuequivariance-torch 0.11.1 with the
CUDA-13 ops pair (`cuequivariance-ops-cu13`, `cuequivariance-ops-torch-cu13`), pytorch-lightning 2.6.5, numpy 2.0.2, numba 0.61.0, `gcc` on `PATH` (Triton
builds the cuEquivariance kernels' launchers at first use) — the full list is `environment/requirements.lock` (104 pins); `environment/Dockerfile` and
`environment/apptainer.def` build it, one image for every card (nothing is compiled per card at build time). Outside the image THE pinned recipe is a uv-managed CPython
3.11.5 venv and the lock, installed without the resolver — run from inside `boltzgen/` (README route C; an environment that already has BoltzGen 0.3.2 at the pin
skips to `bash run.sh install`, whose pin check refuses a differing install by name):

    sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl build-essential   # bare Ubuntu host (root: no sudo); skip what you have — the packages environment/Dockerfile installs
    command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); the installer runs only when its sha256 matches
    uv venv --seed --managed-python --python 3.11.5 ~/venvs/boltzgen-kit && . ~/venvs/boltzgen-kit/bin/activate   # uv's own released CPython, the image's 3.11.5; apt/deadsnakes python instead of uv: python3.11 + python3.11-venv + python3.11-dev, never Ubuntu 22.04's python3.11 package (3.11.0rc1)
    grep -E '^(pip|setuptools|wheel)==' environment/requirements.lock | xargs python -m pip install --no-deps
    grep -v -E '^(#|boltzgen==)' environment/requirements.lock > /tmp/stack.txt && python -m pip install --no-deps -r /tmp/stack.txt
    python -m pip install --no-deps stock/boltzgen-0.3.2-py3-none-any.whl
    export PYTHONHASHSEED=0 CFLAGS=-g0   # the two variables the image sets (environment/Dockerfile ENV); export them in this shell to match it
    bash run.sh install [--weights DIR]     # pip install -e ../common/opt_core -e opt, then stock/check_pins.py (--weights: fetch what is absent into DIR, digest-check what is there) = README step 2's install line — type it once; step 2 continues at its export line

The install must stay editable: the package finds the lever directories under `opt/` and the pin under `stock/` through its own location. `install` without
`--weights` never touches the weights; on a DIR that already holds the six files `--weights` downloads nothing and only checks their sha256. `check` reads
package metadata (the `boltzgen` and shared-core pins), the lever directories and the GPU through `nvidia-smi`; it imports neither torch nor `boltzgen` and does
not read the weights. Other cards: `configs/a100.env` / `configs/h200.env` are `configs/h100.env` with `MODEL_OPT_TARGET_GPU=A100` / `H200`. After `install`,
the checkout, the venv and uv's interpreter directory (`~/.local/share/uv/python`) may be read-only for the users who run the kit: a run writes only under
`--output` and into the compile caches ("Variables": `$HOME` defaults, or `MODEL_OPT_JIT_ROOT`), never into the tree (`PYTHONDONTWRITEBYTECODE=1`); each user
exports `BOLTZGEN_CACHE` (read-only is fine once filled) and, on a shared install, their own `MODEL_OPT_JIT_ROOT`.
The image build takes an optional pre-filled compile cache `_jitcache/boltzgen-*-jit.tar` from the build context and unpacks it to `/opt/jit_cache` (the build
is identical without one); in the image `run.sh` uses `/opt/jit_cache` in place when `MODEL_OPT_JIT_ROOT` is unset and it is writable, copies it once to
`${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` when it is not, seeds an empty `MODEL_OPT_JIT_ROOT` from it once and leaves a non-empty one alone, uses a preset
`MODEL_OPT_JIT_ROOT` the process cannot write read-only — its `<stack key>` subtree is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (up to
`MODEL_OPT_JIT_SEED_MAX_FILES` files, default 5000), otherwise the read-only root is used as it is (nothing is compiled into it; caches already there are read)
— exports the cache variables under `<root>/<stack key>/` (`triton`, `torch_extensions`, `inductor`, `numba`; ones the config derived from a replaced root
follow the new one) and prints `[boltzgen-kit] jit cache: <dir> (in-image|seeded from image|user|unseeded|seeded from read-only root)`, which says which (silent
when a read-only root is left as it is).

## How stock is run

`--mode off` executes `boltzgen configure <spec> --output <out> [--cache $BOLTZGEN_CACHE] <the caller's boltzgen options>` in a clean environment, then one
clean subprocess per step of `<out>/steps.yaml` (`python -S -s -c <preamble>`: this tree's `opt/` and shared-core directories go first on `sys.path` and the package
is imported from them before the interpreter's site runs, then `boltzgen_opt.stock_design` runs as `__main__` and calls upstream's step `main(<config>)`) with every kit
variable stripped (`BG_*`, `XA_*`, `SZ_*`, `FL_*`, `HL_*`, `BOLTZGEN_OPT`, `BOLTZGEN_OPT_KERNELS`, `PYTHONPATH`) and no kit directory importable — checked
before any upstream import and written to `<out>/stock_env_proof_<step>.json` (any finding: exit 3, nothing runs). `--seed N` under `off`: step *i* runs
`pl.seed_everything(N + i, workers=True)` with `random`, NumPy and torch seeded, and upstream's featurizer generator (`np.random.default_rng(None)`) seeded
from that state (MT19937 key + position + the call's ordinal under that key in the calling process, so a DataLoader worker counts from its own first
call) — the derivation the kit modes' seed hook uses, so `exact` equals `off` on structure-conditioned specs too; without `--seed` each step runs exactly as `boltzgen run` leaves it. `warm --mode off` configures
`--num_designs 1 --steps design inverse_folding` on the bundled example. Tokens after a bare `--` on the `design` line are forwarded to `configure` verbatim.

## Stock exceptions

- cuEquivariance CUDA-13 ops: the `boltzgen` 0.3.2 wheel's metadata names the CUDA-12 ops distributions (`cuequivariance-ops-cu12`,
  `cuequivariance-ops-torch-cu12`); this stack installs NVIDIA's CUDA-13 build of the same 0.11.1 release instead — the same modules, Python API and dispatch
  rules — so that the ops library links the CUDA 13 libraries torch 2.13.0+cu130 ships rather than a second CUDA runtime. Applied identically on every mode,
  `off` included; changes no model arithmetic. `pip check` reports those two names and nothing else. The library's import-time line `FALLING BACK to the slow
  PyTorch reference path instead of the fast SM100f kernel` on H100 / H200 / A100 is its notice that one Blackwell-only (cc 10.x) kernel is absent on the
  card; the sm90 / sm80 triangle kernels still serve every call (the `KERNELS … cueq_triatt=engaged … cueq_trimul=engaged` line), in `off` exactly as in the
  kit modes.

## Variables

| variable | required | default / set by the config | effect |
|---|---|---|---|
| `BOLTZGEN_CACHE` | yes | — | the weights directory (`configs/*.env` exit 3 when it is unset; upstream's `--cache` defaults to it) |
| `BOLTZGEN_OPT` | no | unset | the mode when `--mode` is absent; through `boltzgen_opt_autoload.pth` it activates the package in any Python process that imports `boltzgen`; a `--mode` that disagrees is refused (exit 2) |
| `BOLTZGEN_OPT_HOME` | no | derived | the `opt/` directory |
| `MODEL_OPT` | no | derived | this `boltzgen/` directory (how the package locates `opt/` and `stock/PINS.json`) |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` / `H200`, filled when unset | the GPU class the sourced config targets |
| `MODEL_OPT_STACK_KEY` | no | derived | the JIT-cache key `torch<version>-cu<CUDA>-sm<cc>` (e.g. `torch2.13.0-cu130-sm90`); one `NOTE` line when it cannot be derived |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused | persistent JIT-cache root: sets `TRITON_CACHE_DIR` / `TORCH_EXTENSIONS_DIR` / `TORCHINDUCTOR_CACHE_DIR` / `NUMBA_CACHE_DIR` under `$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/` unless already set — must be writable when set (the compile caches are created there, per stack key = torch + CUDA + compute capability, on the first run); unset: nothing is exported and each cache stays at its library's per-user default — Triton `~/.triton/cache`, torch extensions `~/.cache/torch_extensions`, TorchInductor `/tmp/torchinductor_<user>` (only under upstream's opt-in compile flags), numba `__pycache__` beside upstream's featurizer inside site-packages when that is writable, else `~/.cache/numba` (the CUDA driver's own `~/.nv/ComputeCache` is outside the kit's control); on a shared or read-only install each user exports `MODEL_OPT_JIT_ROOT=<own writable dir>`, which routes all four caches under it |
| `HF_HOME`, `HF_HUB_OFFLINE` | no | `HF_HOME=$BOLTZGEN_CACHE` filled when unset; `HF_HUB_OFFLINE=1` set by the config (unconditionally) | the same cache for anything that reads `HF_HOME`; nothing is fetched at run time |
| `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED` | no | `1`, both set by the config (unconditionally) | no `.pyc` in the tree; stderr lines in order |

Kit-declared variables plus the stock variables the kit sets. Sourcing `configs/h100.env` / `configs/a100.env` / `configs/h200.env` assigns the rows marked 'set by the config' on every source and fills the others only when they are unset. `BOLTZGEN_OPT_KERNELS`,
`BOLTZGEN_OPT_HANDOVER` and `BG_TIMING_FILE` are set by the package itself per run. The lever switches (`BG_*`, `XA_*`, `SZ_*`, `FL_*`, `HL_*`) are exported
per mode by the package; a caller-set value is dropped and named (`dropped=…` on the ACTIVE line). Upstream's own `CUEQ_*` variables, `--use_kernels` and
compile switches reach the stock processes and the kit modes' model processes alike.
