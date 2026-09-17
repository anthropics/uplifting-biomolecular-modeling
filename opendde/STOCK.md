# OpenDDE — stock, as pinned

## Pin

Upstream: `opendde` 1.1.1 (PyPI; github.com/aurekaresearch/OpenDDE tag v1.1.1), shipped under `stock/` as the wheel
`opendde-1.1.1-py3-none-any.whl`, the sdist and the unpacked source `stock/src/` with upstream's README and `examples/` (sha256 in
`stock/PINS.json`); model `opendde_v1`. Weights: `checkpoint/opendde.pt` plus the `common/` assets (`components.cif`,
`components.cif.rdkit_mol.pkl`, and `release_date_cache.json` / `obsolete_to_successor.json` for `--use_template true`), sha256 in
PINS.json, fetched into `DIR` by `bash run.sh install --weights DIR` through upstream's own downloader and digest-checked; a run never
downloads. The weights are published by the upstream authors (https://huggingface.co/aurekaresearch/OpenDDE, revision in `stock/PINS.json`);
the terms that apply to them are the ones published there. Upstream's licence and notices as found under `stock/`; `THIRD_PARTY_NOTICES.md` lists the add-on files that restate OpenDDE code and the one
prebuilt binary, with the licence text under `third_party_licenses/`. `stock/` is never edited.

## Stack

Ubuntu 24.04, CUDA 12.6 with `nvcc` and gcc at run time (driver ≥ 560; upstream's fused LayerNorm extension and the Triton kernels
compile on first use), Python 3.11.5, torch 2.7.1+cu126, triton 3.3.1, cuequivariance 0.10.0, numpy 2.4.1, ninja 1.13.2 — the full
Python list is `environment/requirements.lock` (the resolution of upstream's own `uv pip install --torch-backend cu126
"opendde[gpu]==1.1.1"`, pinned); system packages come from apt in `environment/Dockerfile` (build-essential, `kalign` pinned at 3.3.5 for
template alignment, `hmmer` as Ubuntu 24.04 ships it); `environment/Dockerfile` and `apptainer.def` build the whole stack. Into an
existing environment that already runs OpenDDE at the pin: `pip install --no-deps stock/opendde-1.1.1-py3-none-any.whl` if the wheel
is not the installed one, then `bash run.sh install` (kit + shared core editable, then `stock/check_pins.py`, which confirms the installed
`opendde` matches the pin). A fresh environment (README route C) is this block, run from the directory holding `opendde/` and `common/` on a host
with the CUDA 12.6 compiler, gcc and a libstdc++ from GCC 13 or newer:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential ca-certificates curl git hmmer kalign libglib2.0-0 libgl1 libxrender1 pkg-config tzdata   # bare Ubuntu host (root: no sudo); skip what you have; the image pins kalign 1:3.3.5-1
command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; } # uv itself, once per user (skipped when present)
uv venv --seed --managed-python --python 3.11 ~/odde && . ~/odde/bin/activate   # uv's own released CPython 3.11 build, headers included; a python.org or conda 3.11 works too; if you use apt/deadsnakes python instead of uv: python3.11 + python3.11-venv + python3.11-dev, and not Ubuntu 22.04's python3.11 apt package (3.11.0rc1)
grep -v -E '^(#|opendde==)' opendde/environment/requirements.lock > /tmp/stack.txt
pip install --no-deps -r /tmp/stack.txt                                # the whole pinned stack, torch 2.7.1+cu126 by URL and sha256 (≈6 GB)
pip install --no-deps opendde/stock/opendde-1.1.1-py3-none-any.whl     # stock from the bundled wheel
export PYTHONHASHSEED=0                                                # the images' process environment (Dockerfile ENV; with PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CFLAGS=-g0)
cd opendde && bash run.sh install [--weights /weights/opendde]            # = README step 2's install line — type it once; step 2 continues at its export line (--weights fetches what is absent, digest-checks what is present; omit once downloaded)
```

The image build takes an optional pre-filled compile cache, `_jitcache/opendde-<stack key>-jit.tar` in the build context, unpacked under `/opt/jit_cache/<stack key>/…` (the
build is identical without one); in the image `run.sh` prints `[opendde-kit] jit cache: <dir> (<how>)` when one is present: with `--config <card>` the JIT root the config
names (`MODEL_OPT_JIT_ROOT`, by default `~/.cache/opendde_opt/jit`, layout `<root>/<stack key>/{triton,torch_ext}`) is seeded from `/opt/jit_cache` once while it is empty
(`seeded from image`) and left as it is once it holds anything (`user`); without a config `/opt/jit_cache` is used in place when it is writable (`in-image`), else seeded
into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`. In or out of an image, a preset `MODEL_OPT_JIT_ROOT` the process cannot write is used read-only: its `<stack key>` subtree is seeded
once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (up to `MODEL_OPT_JIT_SEED_MAX_FILES` files, default 5000; printed `seeded from read-only root`), otherwise that read-only root is used as it
is (nothing is compiled into it; caches already there are read; no line printed); the cache directories the config derived from a moved root follow the move. Route B (read-only
image, `$HOME` and `$PWD` bound) therefore writes only under that per-user root and `${TMPDIR:-/tmp}` (`model_opt_jit-uid<uid>` above; the shared core may also create `opt_core-uid<uid>/` there
for kernel byte-gate stamps); with the default flags the weights directory is only read, so a read-only bind works.

On another torch / cuEquivariance version every `pred` prints `STACK MISMATCH stack_mismatch:<pkg>=…` and
proceeds; only the upstream package's version and files are refused on. Floor for `exact` outside the images: the shared core's prebuilt
bit-exact sampler-attention kernel (lever `dit_attn_exact`) is linked against GCC 13's libstdc++ (`GLIBCXX_3.4.32`, as on Ubuntu 24.04 or with
conda's `libstdcxx-ng` ≥ 13; the images carry it) — on an older libstdc++ (Ubuntu 22.04) `exact` stops with that ImportError: upgrade it — `sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y libstdc++6`, or in a conda environment `conda install -c conda-forge 'libstdcxx-ng>=13'` — or run
`MODEL_OPT_LEVERS_OFF=dit_attn_exact bash run.sh pred --mode exact …` (upstream's own attention statement serves there; outputs stay identical
to stock). Other cards: `--config a100` (compute capability 8.0, the
80 GB and 40 GB card), `--config h200` (the H100's settings; compute capability 9.0); kernel variants follow the running device, never the config label, and no cuEquivariance tile table is shipped: the library's own packaged per-card table
applies on every card (`opt/forward/fast_inference/levers/KIT/cueq_cache_shipped/` is only the optional user-cache location the kit modes export; users may add their own tuned entries there, see its README). Under `exact` a kernel is bit-exact only on the cards
and stacks its record was taken on; elsewhere it steps aside to the stock op with a `LEVER … reason=aside:…` or `NAMED_FALLBACK:<op> … exact_vouch_not_recorded… served=<stock op>` line
naming the running stack and, marked `(ref)`, the reference stack the shared core's table holds for that card (on A100 with this stack: triangle multiplication, sampler attention; on H200: triangle multiplication, `NAMED_FALLBACK:trimul … served=cueq`); under `exact` triangle attention is a fused kernel bit-identical to the stock op on H100 stacks the core vouches for, the stock op elsewhere.

## How stock is run

`--mode off` executes `opendde pred -i <query> -o <out> --dtype bf16 <stated upstream flags>` with `LAYERNORM_TYPE=fast_layernorm`
in a clean subprocess with nothing of the kit importable. Stock = upstream's CLI defaults (`-c 10 -p 200 -e 5 -n opendde_v1
--use_msa true --use_template false --use_rna_msa false --need_atom_confidence true`, triangle kernels `auto`) plus the two documented
speed settings upstream exposes: `--dtype bf16` (passed unless the caller states `--dtype`) and `LAYERNORM_TYPE=fast_layernorm`
(exported unless already set). `--det 1` under `off`: upstream's `--deterministic true` plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Stock exceptions

None: no stock file is patched and the installed package runs unmodified on every route. Two launcher behaviours apply identically
on every mode, `off` included, and change no model arithmetic: the two speed settings above, and `pred` detaching its process tree
from a non-terminal stdin (`kalign`, which upstream runs per template hit, otherwise blocks on an open pipe).

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `OPENDDE_ROOT_DIR` | yes | — | weights root: `checkpoint/opendde.pt` and `common/`; `pred` and `warm` refuse by name when it is unset or a file is absent, `check` reports the weights state without refusing |
| `OPENDDE_OPT` | no | unset | the mode for the bare `opendde pred` command through the autoload hook (unset: the hook engages nothing and stock runs); `run.sh` / `opendde-opt` use it when no `--mode` is given (neither given: the package default, `fast`); a `--mode` that disagrees with it is refused |
| `MODEL_OPT_JIT_ROOT` | no | `~/.cache/opendde_opt/jit` (the config's default; without a config `run.sh` supplies the private per-user root `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`, made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) | compile-cache root, layout `<root>/$MODEL_OPT_STACK_KEY/<cache class>/` created on first use per GPU/stack key: the config derives `TRITON_CACHE_DIR` and `TORCH_EXTENSIONS_DIR` (classes `triton`, `torch_ext`) and `MODEL_OPT_WEIGHTS_DIGEST_DIR` (`<root>/weights`) unless already set, and the shared core's kernels keep their own classes beside them (e.g. `triattn`, `trimul_native`); if set it must be writable (only the digest memo tolerates a read-only directory: the checkpoint is hashed afresh) |
| `MODEL_OPT_STACK_KEY` | no | computed | cache key `torch<ver>-cu<CUDA>-sm<cc>` (e.g. `torch2.7.1-cu126-sm90`), read from torch's metadata and `nvidia-smi` when the config is sourced; on a host without a GPU the `sm` part reads `unknown` (named on stderr), so preset it (e.g. `MODEL_OPT_STACK_KEY=torch2.7.1-cu126-sm90`) to source the config there, e.g. in an image build step, and keep the cache paths those of the GPU host |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` / `H200` (per config) | label of the card the config targets; kernel variants follow the running device |
| `MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS` | no | `1400`; `1160` on a card under 64 GiB | `big`: pair tensors host-resident and one diffusion sample at a time from this many residue tokens; below it `big` runs `fast`'s resident lever set |
| `MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS` | no | `2565`; `1856` on a card under 64 GiB | `big`: the diffusion-conditioning hoist left out from this many tokens |
| `MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS` | no | `300` | `exact`, `fast`, resident `big`: when every item of a call is below this many residue tokens the trunk levers are left out of the line; `0` = bound at every size; a malformed value is refused by name |
| `MODEL_OPT_LEVERS_OFF` | no | unset | comma-separated lever names left out of any mode (CHANGES.md); an unknown name is refused |
| `LAYERNORM_TYPE` | no | exported as `fast_layernorm` | upstream's LayerNorm selector; a caller's own value runs as stated |

`configs/<card>.env` fills only what is unset; a value already in the environment wins.
