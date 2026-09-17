# ProGen2 — stock, as pinned

## Pin

Upstream: `https://github.com/salesforce/progen` at commit `c27a419c234a0997923761e1fe7daffcebf0eaf5` (subdirectory `progen2/`; no tag
and no package metadata, so the commit is the pin), shipped under `stock/` as the unpacked tree `stock/src/` that every route runs (sha256 of every stock file in `stock/PINS.json`)
plus `stock/progen-c27a419c.tar.gz`, a reference copy of the same checkout — nothing is installed from it; it is the byte source the
pin check names for restoring `stock/src/progen2` when it refuses a modified file; weights: one archive per size from upstream's release URLs
(`pytorch_model.bin` + `config.json`, URLs and sha256 in PINS.json `weights.*`), fetched by `bash run.sh install --weights DIR [--model
progen2-<size>]` (every size when `--model` is omitted; when DIR already holds the files, omit `--weights` and `--model` — `install`
then installs the kit and runs the pin check only, and `check` digest-checks the weights it finds under `PROGEN2_WEIGHTS`) and laid out as `DIR/<progen2-size>/{pytorch_model.bin, config.json}` with a `SHA256SUMS` beside them; upstream's
licence and README as found under `stock/src/`. `stock/` is never edited. Sizes: `progen2-small` (151M) · `progen2-medium`,
`progen2-oas`, `progen2-base` (764M) · `progen2-large`, `progen2-BFD90` (2.7B) · `progen2-xlarge` (6.4B); the kit's stderr lines carry
the short word (`small` … `bfd90` … `xlarge`) as `variant=`. Context window `n_positions`: 1024 tokens, 2048 for `base`.

## Stack

Ubuntu 22.04, CUDA 12.8 (driver ≥ 570), Python 3.9 (any 3.9.x passes the pin check as `pinned (3.9.x)`; the image builds 3.9.23), torch 2.8.0+cu128, transformers 4.16.2, tokenizers 0.10.3, numpy 2.0.2 —
the full list is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it (`docker run --rm --gpus all
-v /path/to/progen2_weights:/weights -e PROGEN2_WEIGHTS=/weights progen2-kit:dev bash run.sh <verb> …`; the Apptainer image is read-only
and starts in `/kit/progen2`, so inputs, `--out_dir` and the weights go on absolute host paths — `$HOME` and `$PWD` are bound by
default, `apptainer run --nv --bind DIR progen2-kit.sif …` binds another). Outside the images (README route C) the pinned recipe is, from the
directory holding `progen2/`:

    sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl build-essential   # bare Ubuntu host (root: no sudo); skip what you have — the packages environment/Dockerfile installs
    command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; } # uv itself, once per user (skipped when present)
    uv venv --seed --managed-python --python 3.9 ~/venvs/progen2 && . ~/venvs/progen2/bin/activate   # uv's own released CPython 3.9 (the image builds 3.9.23 the same way); alternatives: python.org, conda, or deadsnakes python3.9 + python3.9-venv (Ubuntu apt itself ships no python3.9; no -dev headers needed, nothing compiles); no host CUDA toolkit: torch's wheel brings the CUDA 12.8 runtime (driver ≥ 570)
    python -m pip install --no-deps -r progen2/environment/requirements.lock   # the whole pinned stack in one pass (torch and triton by PyTorch-index URL + sha256, the rest PyPI)
    cd progen2 && bash run.sh install                                             # = README step 2's install line — type it once; step 2 continues at its export line

An existing Python 3.9 environment at the pin works the same (at minimum torch 2.8.0+cu128, transformers 4.16.2, tokenizers 0.10.3;
`install` names any version drift). `bash run.sh install` is (the
kit, editable — `pip install -e opt`, the only pip install besides the stack: nothing of upstream is pip-installed, the reference archive
`stock/progen-c27a419c.tar.gz` included — then `stock/check_pins.py`, which confirms the stock files match the pin file for file and
reports the interpreter and library versions against the pins as `noted:` lines). `PROGEN2_PYTHON` names the interpreter when it is
not `python` on PATH. A stack off these pins is named on the kit's `stack` line at activation and engaged on, not refused. The scoring
route's fused kernels ship prebuilt as `opt/forward/engines/progen2/kits/v0_ew/libew_progen2.so` (CUDA 12.8; SASS sm_80 / sm_90 /
sm_100, PTX compute_90; `BUILD.json` beside it); `python -m engines.progen2.kits.v0_ew.build` from `opt/forward`, with `nvcc` on PATH,
rebuilds it. No other card set-up exists: the card class is read from the device.

## How stock is run

`--mode off` executes `python sample.py <sample.py's own flags>` or `python likelihood.py <likelihood.py's own flags>` under
`$PROGEN2_PYTHON` in a clean subprocess with nothing of the kit importable, from a run directory of symlinks to the pinned checkout
with `checkpoints/<progen2-size>` → the weights (`opt/progen2_opt/stock_cli.py`; the kit's `PROGEN2_*` variables other than the four
below are stripped from the child's environment). One process per call, or per item of an `--input` job. Stock defaults kept on every
arm: fp16 weights under `--fp16 true`, no autocast around `generate`, `autocast(fp16)` around the scoring forward,
`set_seed(<seed>, deterministic=True)`, temperature 0.2, top-p 0.95, pad id 0; PyTorch precision switches untouched.

The stock flags, as both modes take them (defaults in brackets): `sample.py --model <progen2-size> [--device cuda:0] [--rng-seed 42]
[--rng-deterministic true] [--p 0.95] [--t 0.2] [--max-length 256] [--num-samples 1] [--fp16 true] [--context 1] [--sanity true]`;
`likelihood.py --model <progen2-size> [--device cuda:0] [--rng-seed 42] [--rng-deterministic true] [--fp16 true] [--context <its own
422-character default>] [--sanity false]`.

## Stock exceptions

- STACK: upstream's `requirements.txt` pins `torch==1.9.0+cu111`, a CUDA 11.1 build with no kernel image for compute capability 9.0;
  the stack re-pins torch to 2.8.0+cu128 and the interpreter to Python 3.9, keeping transformers and tokenizers at upstream's pins and
  the stock code unchanged — applied identically on every mode, `off` included; changes no model arithmetic.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `PROGEN2_WEIGHTS` | yes | `weights/` under the kit tree, if present | weights root: `<dir>/<progen2-size>/{pytorch_model.bin, config.json}` |
| `PROGEN2_PYTHON` | no | `python` on PATH (`run.sh`); the running interpreter (package) | the stack's interpreter, for `run.sh` and every stock process the kit spawns |
| `PROGEN2_STOCK_DIR` | no | the tree's `stock/src/progen2` | the pinned checkout's `progen2/` directory |
| `PROGEN2_RUN_DIR` | no | a per-user temp dir | the stock run directory the kit builds |

Kit-declared variables only; no variable selects a mode or a size (`--mode`, `--model` do) and nothing is configured per card. `run.sh`
exports `PYTHONDONTWRITEBYTECODE=1`; the stock scripts' own `set_env()` sets `TOKENIZERS_PARALLELISM=false`.
