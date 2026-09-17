# PXDesign — stock, as pinned

"Stock" is upstream exactly as a user installs and runs it, with nothing from this tree on the path. `stock/PINS.json` is the
machine-readable form of this page (read by `configs/h100.env`, `stock/check_pins.py` and the package); `stock/` is never edited.

## Pin

| component | repository | commit / tag | version | reference copy in `stock/` (extracted under `stock/src/`; nothing installs from it) |
|---|---|---|---|---|
| `pxdesign` | https://github.com/bytedance/PXDesign | `f788441313c84c3074fe9596ac2433f96b15c763` (no tag) | 0.1.0 | `pxdesign-f7884413.tar.gz` (the commit minus two font files, below) |
| `protenix` | https://github.com/bytedance/Protenix | `d18aa1daadd02a001b32bc7fa2278fc8a2f8f025` (tag `v0.5.0+pxd`) | 0.5.0+pxd | `protenix-d18aa1da.tar.gz` |
| `pxdbench` | https://github.com/bytedance/PXDesignBench | `f6d0d72496c23ca1374df5943754f3454ae8c549` (tag `v0.1.2`) | 0.1.2 | none — not distributed in this tree; pip fetches it from the repository at install, at this commit |

The two archives are reference copies (`git archive` of the pinned PXDesign and Protenix commits): the pin check's byte source and for reading.
The PXDesign archive and `stock/src/PXDesign/` are upstream at `f7884413` minus two font files under `pxdesign/pxd_server/` (`Helvetica-Regular.ttf`,
`TimesNewRoman.ttf`; `stock/PINS.json` `upstream.pxdesign.omitted`), which the kit does not use: upstream reads them only for the `pxdesign pipeline`
verb's summary figure and its web UI; obtain them from the upstream repository if you run those. `environment/Dockerfile` deletes them from the
checkout it installs, too. The pin check compares source files, so it is unaffected.
pip installs the three packages from the repository URLs above at those commits (the three stock lines of `environment/requirements.lock`;
§Stack); PXDesignBench arrives only that way, under its own licence, and the pin check verifies its installed version and checked-out commit.
Upstream's licences and notices as found inside the archives and the fetched checkout. Weights and data are never downloaded at run time (an absent file is
refused before upstream's download would start); `bash run.sh install --weights DIR` fetches them once with upstream's own downloader
into `DIR/checkpoint` and `DIR/ccd_cache` and checks each file's sha256 against `stock/PINS.json`. `$PXDESIGN_CKPT_DIR` (passed as
`--load_checkpoint_dir`) holds `pxdesign_v0.1.0.pt`, the only variant, plus the three Protenix files every `pxdesign infer` run loads
from the same directory; `$PROTENIX_DATA_ROOT_DIR` is upstream's CCD cache (read by `protenix/data/ccd.py`).

| directory | file | sha256 |
|---|---|---|
| `$PXDESIGN_CKPT_DIR` | `pxdesign_v0.1.0.pt` | `b075867bae942dc0c6487173736922b0e2913308c1ba542d227418b6e176478d` |
| `$PXDESIGN_CKPT_DIR` | `protenix_base_default_v0.5.0.pt` | `9ea20b0aba42f2256711da1d0cd081510a4b291e64375bff6b70ced70b87a5f1` |
| `$PXDESIGN_CKPT_DIR` | `protenix_mini_default_v0.5.0.pt` | `3803340c5d9958c038e799ddd2b53b532db21855f261592ad455a5f003791f81` |
| `$PXDESIGN_CKPT_DIR` | `protenix_mini_tmpl_v0.5.0.pt` | `221ca4da769e36ea0e2fa1fa82c46f6ca3a00bc7a1eff8ec0e3ddc02ed830474` |
| `$PROTENIX_DATA_ROOT_DIR` | `components.v20240608.cif` | `7240b17369ccfbbcc86e2d02dc8c9db59f46c32e0420f58889c6c121c60bfef0` |
| `$PROTENIX_DATA_ROOT_DIR` | `components.v20240608.cif.rdkit_mol.pkl` | `2d6caced2d26c62015115a1d0a50f4106755a300e0c2a2d2b5c101c9038dcbcd` |
| `$PROTENIX_DATA_ROOT_DIR` | `clusters-by-entity-40.txt` | `1ab4af905e75b382eda8dec59917dc3608bee0729e36b9e71baf860bbe86850c` |

## Stack

`nvidia/cuda:12.1.1-devel-ubuntu22.04` (driver that runs CUDA 12.1: 525.60 or newer), Python 3.11.5, torch 2.3.1+cu121 (cuDNN 8.9.2,
triton 2.3.1), deepspeed 0.15.4, numpy 1.26.3 — the full list is `environment/requirements.lock`; `environment/Dockerfile` and
`apptainer.def` build it. Upstream itself declares torch 2.3.1, numpy 1.26.3, wheel tags cu121 | cu124, Python 3.11, deepspeed >= 0.15.1.
Route C, typed from inside `pxdesign/` on a bare Ubuntu host (as root drop `sudo`; skip what you have; the CUDA 12.1 toolkit's `nvcc` on PATH as well):

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git build-essential
command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; }   # uv itself, once per user
uv venv --seed --managed-python --python 3.11 venv && . venv/bin/activate                                      # uv's own released CPython 3.11, headers included
grep -v '^#' environment/requirements.lock > /tmp/stack.txt && pip install --no-deps --src "$HOME/src" -r /tmp/stack.txt   # the pinned stack, stock's three git lines included
```

Then README step 2 from its `bash run.sh install --weights DIR` line (already inside `pxdesign/`: skip its `cd`): that install puts the kit and the shared core in
editable form and ends with `stock/check_pins.py` (the installed packages match the pins: version, and source file for file against `stock/src/`; pxdbench, which
has no copy there, by version and checked-out commit). The stack
line pulls ≈3 GB of wheels from PyPI and download.pytorch.org plus three GitHub clones (the environment takes ≈5.6 GB): protenix as a regular install from its
repository at the pinned commit, pxdesign and pxdbench as editable checkouts of theirs (PXDesign's own form, since `pxdesign/configs` has no `__init__.py`).
Interpreter alternatives: apt/deadsnakes python3.11 + python3.11-venv + python3.11-dev (never python3-dev, and not Ubuntu 22.04's own python3.11 package,
which is 3.11.0rc1), python.org or conda builds, or an existing Python 3.11 environment with git, gcc and `nvcc` on PATH; `check` reports a patch-level
difference from the image's 3.11.5 as `stack_differs=…` and proceeds. Protenix's fused-LayerNorm CUDA extension is built with `nvcc` on first use (or at
image build) inside the installed `protenix` package (`…/protenix/model/layer_norm/`, the `build_directory` its `layer_norm.py` passes; writable once) — not
under `TORCH_EXTENSIONS_DIR`; that build needs the C/C++ compiler above and the Python development headers (`Python.h`: bundled with uv-managed interpreters
and with the image's standalone Python; python3.11-dev for an apt/deadsnakes interpreter). The image also exports `PYTHONHASHSEED=0` and `CFLAGS=-g0`
(Dockerfile `ENV`); route C may export the same. One image serves H100, H200 and A100.

## How stock is run

Console script `pxdesign = pxdesign.runner.cli:cli`; the verb is `pxdesign infer -i <tasks> -o <out> [options]` (the `pipeline` verb's
filters are outside this kit). `--mode off` runs it in a clean subprocess: the caller's environment minus every variable whose name
starts with a prefix in `stock/PINS.json` `stock_environment.must_be_absent_prefixes` — `PXD_` (the kit switches), `PXDESIGN_HOIST`,
`PXDESIGN_OPT`, `CUDA_MPS_` — and no kit directory on `PYTHONPATH`: the console script itself under
`--det 0`, upstream's `main()` in-process with deterministic seeding under `--det 1` (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, then Protenix's
`seed_everything(seed, deterministic=True)`). Upstream's documented defaults are rendered explicitly unless the caller gives another
value (`stock/PINS.json` `cli_defaults`): `--N_step 400 --N_sample 5 --dtype bf16 --eta_type const --eta_min 2.5 --eta_max 2.5
--num_workers 16 --use_msa true --use_fast_ln true`; no default seed (without `--seeds` upstream derives one from `time.time_ns()`),
in every mode. Kit modes refuse to run on an installed upstream that is not the pinned one (`stock/check_pins.py`; `bash run.sh check`
prints the facts as its `PINS` line); `off` runs upstream as installed.

## Stock exceptions

- `LAYERNORM_TYPE`: Protenix chooses its LayerNorm class from this variable when its modules are imported
  (`protenix/openfold_local/model/primitives.py`), and PXDesign's runner exports it only afterwards, so a bare `pxdesign infer` runs the
  plain LayerNorm although `--use_fast_ln true` (upstream's default) asks for the fused one. `run.sh design` sets `LAYERNORM_TYPE` from
  `--use_fast_ln` before Protenix is imported (`true`: `fast_layernorm`; `false`: unset) — applied identically on every mode, `off`
  included — and `configs/h100.env` (sourced by `a100.env` and `h200.env` too) sets it for the `PXDESIGN_OPT=<mode> pxdesign infer …` route with its
  line `export LAYERNORM_TYPE=${LAYERNORM_TYPE-fast_layernorm} …` (a caller-set value, even empty, is kept). Changes no model
  arithmetic beyond selecting the kernel upstream's flag names; no upstream code is patched.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `PXDESIGN_CKPT_DIR` | yes | none (unset: `[pxdesign-opt] NOT ACTIVE: PXDESIGN_CKPT_DIR is not set …`, rc 3) | the weights directory above |
| `PROTENIX_DATA_ROOT_DIR` | yes | none (refused by name, rc 3, as above) | upstream's CCD cache |
| `PXDESIGN_OPT` | no | unset = `off` on the `pxdesign infer` route; `run.sh` verbs default to `fast` | the mode; must agree with `--mode` when both are given |
| `LAYERNORM_TYPE` | no | `fast_layernorm` (`configs/h100.env`: `export LAYERNORM_TYPE=${LAYERNORM_TYPE-fast_layernorm}`; a caller-set value, even empty, is kept) | Protenix's LayerNorm switch (§Stock exceptions) |
| `MODEL_OPT_TARGET_GPU` | no | `H100` (`h100.env`), `A100` (`a100.env`), `H200` (`h200.env`) | the GPU the configuration targets; a mismatch with the visible card is named, never refused |
| `MODEL_OPT_STACK_KEY` | no | derived by `configs/h100.env`'s line `export MODEL_OPT_STACK_KEY=${MODEL_OPT_STACK_KEY:-$(python -c "from pxdesign_opt.modes import jit_cache_key; print(jit_cache_key())" …)}`: `torch<version>-cu<cuda>-sm<cc>`; refused when a part cannot be established (set it explicitly) | the JIT-cache key |
| `MODEL_OPT_JIT_ROOT` | no | unset | shared JIT-cache root; when set it must be writable — `TORCH_EXTENSIONS_DIR` (torch's and DeepSpeed's JIT-extension root) then defaults to `$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/torch_extensions`, one directory per stack key (so per GPU class); upstream's default options build nothing there (Protenix's LayerNorm extension builds inside the installed package, §Stack); unset, `TORCH_EXTENSIONS_DIR` stays as the caller has it |
| `TORCH_EXTENSIONS_DIR` | no | `$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/torch_extensions` when the root is set, else as the caller has it (torch's own default) | torch's / DeepSpeed's JIT-extension build root; Protenix's LayerNorm extension does not build there (§Stack) |
| `PXDESIGN_OPT_HOME` / `MODEL_OPT` | no | derived from the package location | the `pxdesign/` tree (the kit locates `opt/forward/hoist` through it) |

The kit switches `PXD_HOIST`, `PXD_HOIST_MODE`, `PXD_HOIST_MASK` are set by the mode (CHANGES.md §Switches); a caller-set one is
dropped at activation and named on the activation line.
