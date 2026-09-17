# ESMFold2 — stock, as pinned

## Pin

Upstream: Biohub `esm` 3.3.0 at commit `26b0bc2b` and the Biohub `transformers` fork 4.57.6 at commit `ef32577f`, shipped under `stock/`
as two source archives plus `stock/PINS.json` (commits, install lines, sha256 digests, stack, GPU classes); upstream's licences and notices
as found inside the archives. `stock/esm-26b0bc2b.tar.gz` is a `git archive` snapshot of the `esm` commit from https://github.com/Biohub/esm
(the `esm` package, `pyproject.toml`, the licence file `LICENSE.md` — MIT License, `Copyright 2026 Chan Zuckerberg Biohub, Inc.` — and
upstream's `THIRD_PARTY_NOTICE.md`). `stock/transformers-ef32577f.tar.gz` is a full source snapshot of the fork at its commit: the complete
`src/transformers/` package, `setup.py`, `pyproject.toml`, `README.md` and the fork's licence file `LICENSE` (Apache License 2.0, `Copyright
2018- The Hugging Face team. All rights reserved.`; no `NOTICE` file). The fork's repository, github.com/Biohub/transformers, is no longer
publicly available; the same commit is served unchanged by https://github.com/huggingface/transformers, which is where the lock installs it
from. As recorded in the archive, the fork is huggingface/transformers 4.57.6 plus the ESMFold2 and ESMC model directories
(`src/transformers/models/esmfold2/`, `models/esmc/`) and their four model-registry entries, every other file identical to that upstream
release; `THIRD_PARTY_NOTICES.md` gives the file-level account, the licences and the data files the `esm` archive carries. `stock/src/` holds
byte-identical reading copies of the ESMFold2-related files of both archives. `stock/` is never edited. Weights: the three published checkpoints `biohub/ESMFold2`, `biohub/ESMFold2-Fast`
and `biohub/ESMC-6B` on huggingface.co (their model cards state the terms that apply to them) at the snapshot commits in `PINS.json`
"weights", fetched by `bash run.sh install --weights DIR` through `huggingface_hub` and checked file by file against the pinned sha256 (exit 1
naming a file that differs). `DIR` is then the `HF_HOME` the other verbs read (`hub/models--biohub--ESMFold2`,
`hub/models--biohub--ESMFold2-Fast`, `hub/models--biohub--ESMC-6B`, each `refs/main` naming its commit); `run.sh` and `esmfold2-opt` set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` unless the caller already set them (`configs/<card>.env` set them to `1` outright; `install --weights`
lifts them for the download only), so an absent snapshot is a refusal by name and a checkpoint with another digest runs with
`WEIGHTS unknown … proceeding` in place of `WEIGHTS pinned`.

Variants (`--variant` or `ESMFOLD2_VARIANT`, required on `pred`, `warm` and `check`; one per process): `fast` = ESMFold2-Fast (24-layer
trunk, no MSA module); `full_msa` / `full_nomsa` = ESMFold2 (48-layer trunk + MSA module) with / without MSAs. All three load the 6B
language model.

## Stack

Linux x86-64, CUDA 13.0 (driver ≥ 580), Python 3.12 (3.12.10 in the image), torch 2.13.0+cu130, triton 3.7.1, flash-attn 2.8.3.post1,
transformer_engine 2.15.0, xformers 0.0.35 — the full list is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def`
build it. The three extensions are the library's own accelerated paths and part of the stock configuration; no index carries them for this
torch, so the image compiles them (`environment/build_wheels.sh` is that step for an environment of your own) or takes wheels placed in
`stock/wheels/` (`--build-arg WHEELS_FROM=prebuilt`). The lock installs `esm` and the `transformers` fork from their git URLs at the pinned
commits; `stock/esm-26b0bc2b.tar.gz` and `stock/transformers-ef32577f.tar.gz` are reference copies of those commits (for reading; the pin check
also accepts a `pip install` of either archive, and refuses any other source by name). pip builds the three git-sourced packages (`esm`, the
fork, `DockQ`) in isolated build environments and fetches their build backends (setuptools, wheel; Cython and numpy for DockQ) from the index at
install time — network needed; the lock pins runtime packages only.

Route C of the README — a fresh environment on a GPU host; needs `uv` (the block's second line installs it once per user; its managed CPython brings its own headers), git, a C++ compiler (Triton compiles kernel launchers at run time,
DockQ a C extension at install) and, for the three extensions, the CUDA 13.0 toolkit (`nvcc`). From the directory holding `esmfold2/` and `common/`:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y gcc gfortran build-essential git ca-certificates curl    # bare Ubuntu host (root: no sudo); skip what you have; the CUDA 13.0 toolkit comes from NVIDIA's repository
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; } # uv 0.12.15 itself, once per user; installer digest-checked
uv venv --seed --managed-python --python 3.12.10 ef2-venv && . ef2-venv/bin/activate       # uv's own released CPython 3.12.10 (the image's), never a system python; without uv: apt/deadsnakes python3.12 + python3.12-venv + python3.12-dev, then `python3.12 -m venv`
grep -E '^pip==' esmfold2/environment/requirements.lock | xargs pip install --no-deps            # pass 1: the lock's own pip
grep -v -E '^(#|(flash_attn|transformer_engine|xformers)==)' esmfold2/environment/requirements.lock > /tmp/ef2-stack.txt
pip install --no-deps -r /tmp/ef2-stack.txt                                                     # pass 2: every other pin, the two Biohub packages included
bash esmfold2/environment/build_wheels.sh                                                       # A100 host: TORCH_CUDA_ARCH_LIST=8.0 bash esmfold2/environment/build_wheels.sh --stack img_esmfold2_a100 (see below). The three CUDA extensions from source: ≈16 min with 24 build jobs on a large host, plausibly 30–45 min on 8–16 cores (scales with cores; --jobs N, default all)
export PYTHONHASHSEED=0 NVTE_FRAMEWORK=pytorch XFORMERS_IGNORE_FLASH_VERSION_CHECK=1            # the image's ENV; export them in every shell that runs the kit
cd esmfold2 && bash run.sh install --weights /weights/esmfold2                                     # then HF_HOME and check as in the README
```

`build_wheels.sh [--stack img_ef2_fa|img_esmfold2_a100] [--jobs N]` checks its prerequisites by name (torch 2.13.0+cu130, nvcc 13.x — `CUDA_HOME`
honoured —, git, a C++ compiler, the lock's NCCL and cuDNN wheels; exit 3 otherwise), fetches xformers and TransformerEngine at their release tags
and the flash-attn sdist (sha256-checked), builds with the image's settings, writes the wheels to `stock/wheels/` and installs them `--no-index
--no-deps`; until they are installed every verb refuses at the pin check (`image pins NOT MET …`, exit 3). An environment that already runs
ESMFold2 at the pin takes `bash run.sh install` directly (kit + shared core editable, then the pin check). Other cards: `docker build --build-arg
STACK=img_esmfold2_a100 …` compiles the extensions for compute capability 8.0 as well (route C on an A100 host: `TORCH_CUDA_ARCH_LIST=8.0
bash esmfold2/environment/build_wheels.sh --stack img_esmfold2_a100` builds for the A100 alone, about the single-card times above; without the variable both 8.0 and 9.0, twice the compile), and
`--config a100` targets that card; `--config h200` (`configs/h200.env`) uses the H100 settings on H200. The image build takes an optional pre-filled compile cache `_jitcache/esmfold2-*-jit.tar` from the build
context into `/opt/jit_cache` (the build is identical without one); in the image `run.sh` uses `/opt/jit_cache` in place when it is writable,
else — a read-only image, or `MODEL_OPT_JIT_ROOT` naming an empty directory — seeds the JIT root from it once (`MODEL_OPT_JIT_ROOT`, layout
`<root>/<stack key>/…`; `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` when the variable is unset), leaves a populated root untouched, and prints
`[esmfold2-kit] jit cache: <dir> (<how>)` before the config is read; a preset `MODEL_OPT_JIT_ROOT` the process cannot write is used read-only —
its `<stack key>` subtree (`MODEL_OPT_STACK_KEY` as preset; the config derives the key only afterwards) is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`
(up to `MODEL_OPT_JIT_SEED_MAX_FILES` files, default 5000), otherwise the read-only root is used as is (nothing is compiled into it; caches
already there are read) and no line is printed.

## How stock is run

Upstream is a Python API with no command line, so `--mode off` executes the kit's stock caller `opt/esmfold2_opt/stock_fold.py`
(`from_pretrained` at its defaults, then `ESMFold2InputBuilder().fold` per input and seed) in a clean subprocess with every kit variable
stripped and nothing of the kit importable. `--backend fused` adds the library's two documented speed calls `set_kernel_backend('fused')`
+ `set_chunk_size(None)` — the configuration `exact` reproduces bit for bit; `--backend shipped` (the flag's default) makes neither call
(`model_calls=none` on the `SETTINGS` line). Fold settings pass through by name on every mode, each at the library's default when absent:
`--num_loops` (20), `--num_sampling_steps` (200), `--num_diffusion_samples` (1), `--msa_max_depth` (1024), `--lm_dropout`,
`--msa_column_mask_rate`, `--remove_insertions`, `--max_sequences`, and `fold()`'s optional overrides `--noise_scale`, `--step_scale`,
`--max_inference_sigma`, `--lm_mask_pct`, `--early_exit` (`opt/esmfold2_opt/settings.py`); the resolved values print on one `settings …`
line, identical on `off` and the kit modes. One diffusion-sample count per run (the flag, else the inputs' own key, else 1; inputs that
disagree are refused, exit 2); `--seeds` wins over an input's `seeds` key, and without either a fold runs unseeded. Two settings are
stochastic at inference as shipped: `lm_dropout` (0.3, drawn inside every recycle) and the MSA row subsample to `msa_max_depth`.
`--det 1` under every mode, `off` included (`opt/esmfold2_opt/det.py`): `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before torch is imported,
`torch.use_deterministic_algorithms(True, warn_only=True)`, `msa_column_mask_rate=0`, and a seed is required (exit 2 without one); the
LM dropout stays on with its draw seeded. `--det 2` additionally zeroes `lm_encoder.lm_dropout` at load.

## Stock exceptions

None. No route patches the installed packages; `configs/<card>.env` hold deployment parameters only.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `HF_HOME` | yes | none | the weights root (above); `pred` / `warm` refuse by name (exit 3) while it is unset or not a directory |
| `ESMFOLD2_OPT` | no | unset | the mode on the environment route; a `--mode` that disagrees is refused (exit 2) |
| `ESMFOLD2_VARIANT` | no | unset | the variant on the environment route; same rule against `--variant` |
| `ESMFOLD2_OPT_REQUIRE_FAST_ENV` | no | `1` in `configs/` | `1`: every route refuses (exit 3) unless the flash-attention and TransformerEngine paths import; `0`: report them on the `ACTIVE` line and proceed — the three extension packages must be installed at their pins either way (`run.sh` runs the pin check, which reads their metadata, before every verb) |
| `ESMFOLD2_OPT_ABLATE` | no | unset | alias of `MODEL_OPT_LEVERS_OFF` (CHANGES.md §Switches) |
| `ESMCFOLD_CCD_PATH` | no | derived from `HF_HOME` | upstream's CCD pickle inside the ESMFold2 snapshot |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | no | `1` (set by `run.sh` / `esmfold2-opt` when unset; `configs/` set `1`) | the libraries' offline switches, read by upstream; `install --weights` unsets them for its download |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused | persistent cache root, which must be writable when set: the config exports `TRITON_CACHE_DIR=<root>/<stack key>/triton` (compile caches are created there per GPU / stack key on first run) and `ESMFOLD2_OPT_WEIGHTS_MEMO_DIR=<root>/weights` (the weights-digest memo; `check` digests afresh) unless pre-set; left unset, nothing is exported — Triton compiles into its own default cache directory and the memo uses the package default |
| `MODEL_OPT_STACK_KEY` | no | derived | the cache key, e.g. `torch2.13.0-cu130-sm90`; `configs/<card>.env` derives it through `esmfold2_opt.modes.jit_cache_key` (torch version, CUDA version, the device's compute capability) and refuses by name when a part cannot be established — preset it (e.g. `MODEL_OPT_STACK_KEY=torch2.13.0-cu130-sm90`) to source the config on a host without a GPU, e.g. an image build step |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `H200` / `A100` | the class the config targets; another card is a printed `NOTE`, never a refusal |
| `EF2_GRAPH_BUDGET_TOKENS` | no | package default | per-shape CUDA-graph budget under `exact` / `fast` (`…_TRUNK` / `_ENCODER` / `_SAMPLER` per site): larger inputs run the same kernels without capture; `0` = capture every shape |
| `EF2_ROWPAIR_SAMPLER`, `EF2_ROWPAIR_TRANSITION_ROWS`, `EF2_ROWPAIR_TRIMUL_MIN_TOKENS`, `EF2_CONF_PER_SAMPLE` | no | package defaults | `big` / `--n_gpu` route choices (CHANGES.md) |
| `EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS` | no | `1024` | `--n_gpu` > 1: the token count from which the row-chunking levers act (`injrows`, `confrows`, `confbf16`, `pdeskip`, `confmem`, `zbf16`; the seventh, `biasfree`, releases the sampler's pair-bias buffers at every size — CHANGES.md §`--n_gpu P`); a fold below it runs the unchunked statements exactly, in every output, and says `rowchunk_rows=whole:below_floor` on its fold line |
| `EF2_ROWPAIR_INJECT_MB`, `EF2_ROWPAIR_CONF_MB`, `EF2_ROWPAIR_ZBF16_MB` | no | `512`, `512`, `1024` | `--n_gpu` > 1: MiB per row block of the recycle inject (`injrows`) and of the confidence statement (`confrows`, `confbf16`, `confmem`), and per row band of the bf16 pair copy (`zbf16`); read once per rank when the levers bind with the floor above — a value that is not a positive number is refused by name (exit 3) |
| `PYTORCH_CUDA_ALLOC_CONF` | no | exported by `big` | `big` requires `expandable_segments:True` and refuses another value by name |
