# RFdiffusion3 — stock, as pinned

## Pin

Upstream: RFdiffusion3 = the `rfd3` model of github.com/RosettaCommons/foundry at commit `4010e3e2e7350edada3e25a45c908c6bf407df4d`
(13 commits after `v0.2.0`; distribution `rc-foundry`, version `0.2.1.dev13+g4010e3e2e`, extra `rfd3`; entry point `rfd3 design` =
`rfd3.cli:app`, engine config `rfd3/configs/inference_engine/rfdiffusion3.yaml`), shipped under `stock/` as the source archive
`stock/foundry-4010e3e.tar.gz` — a `git archive` of the paths the wheel needs plus `models/rfd3/docs/examples/` and
`models/rfd3/docs/input_pdbs/`, extracting to `foundry-4010e3e2/`, sha256 in `stock/PINS.json`; upstream's licence travels inside it
(`foundry-4010e3e2/LICENSE.md`). Weights: `rfd3_latest.ckpt` (upstream's registry name `rfd3`, source file
`rfd3_foundry_2025_12_01_remapped.ckpt`, 2,690,316,669 bytes, sha256 in `stock/PINS.json` "checkpoint" — upstream registers no checksum),
fetched by `bash run.sh install --weights DIR` through upstream's own `foundry install rfd3 --checkpoint-dir DIR` and checked against that
digest; by hand: https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt. `stock/` is never edited.

`stock/check_pins.py` (standard library only) reports an interpreter against `stock/PINS.json`: the install source (the git commit
through PEP 610 `direct_url.json`, the archive by sha256, or a local directory, always at that version string; an index install is a
finding), the eight overlaid `rfd3/model` files (`--expect pristine`: at upstream's sha256 with the kit's two new files absent;
`--expect any` in the kit interpreter) and python / torch / triton against the stack below. `run.sh check` prints its report; `design`
and `warm` print one `PINS …` line per finding and run — a checkout upstream itself runs is stock; only `run.sh install` stops on a
finding (exit 3).

Upstream's defaults, which no mode touches: `diffusion_batch_size` 8, `n_batches` 1, `inference_sampler.num_timesteps` 200,
`step_scale` 1.5, `noise_scale` 1.003, `gamma_0` 0.6, `skip_existing` True, `compile_model` False, `low_memory_mode` False,
classifier-free guidance off, `ckpt_path: rfd3`, `seed` null (unseeded; `seed=<int>` seeds torch, numpy and random through
`lightning.fabric.seed_everything`).

## Stack

Ubuntu 22.04, CUDA 12.8 runtime (host driver with CUDA 13.0 support, 580 or newer), Python 3.12.1, torch 2.13.0+cu130, triton 3.7.1,
NVIDIA apex 0.1 (github.com/NVIDIA/apex at commit `a1d527a857e8da64c4e7237ca89ec699fb4d9eaf`, its C++/CUDA extensions built against
that torch) — the full list is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it. pip fetches the build backends from the index at install time — hatchling and hatch-vcs for
`rc-foundry` (the Dockerfile holds them at fixed releases through a constraints file), setuptools for the editable installs of the kit and
the shared core — so an install needs network access; the lock pins runtime packages only. The image build takes an optional pre-filled compile cache
`_jitcache/rfdiffusion3-<stack key>-…-jit.tar` from the build context into `/opt/jit_cache/<stack key>/` (the build is identical
without one); in the image `run.sh` uses `/opt/jit_cache` in place when it is writable and `MODEL_OPT_JIT_ROOT` is unset, else seeds the
JIT root from it once (`MODEL_OPT_JIT_ROOT` when set and empty, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` when the image is read-only — a `.seeded` marker: later runs reuse the copy; a populated
root is left alone), prints `[rfdiffusion3-kit] jit cache: <dir> (<how>)` on every run (the same words once seeded), and the configuration then keys the Triton, torch-extensions
and inductor caches under `<root>/<stack key>/` (§Variables); a preset `MODEL_OPT_JIT_ROOT` the process cannot write is never compiled into —
`run.sh` seeds `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` once from `<that root>/<stack key>/` when `MODEL_OPT_STACK_KEY` is preset and the subtree holds at
most `MODEL_OPT_JIT_SEED_MAX_FILES` files (default 5000) and says so on the printed line, otherwise the root stays as set, silently (caches already
there are read), and the configuration keys the three cache directories under `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>/<stack key>/` instead. Upstream pins no
stack (`torch>=2.2.0`, Python >= 3.12). apex is not on upstream's install line, but upstream's own import guard
(`rfd3/model/layers/layer_utils.py`) binds `apex.normalization.fused_layer_norm.FusedRMSNorm` as every RMSNorm of the network when it
is importable (`torch.nn.RMSNorm` otherwise), so it is part of the pinned stack in both interpreters: without it a kit mode's `KERNELS`
census reads `rmsnorm=fallback:torch.nn.RMSNorm(…)` and refuses the pass (exit 5); `--mode off` reports it and runs, as upstream does.

Two interpreters, built alike. The kit is a file overlay on the installed `rfd3` package, so a stock run and a kit run never share a
site-packages: the PRISTINE interpreter (never touched by the overlay's `install.sh`) runs `--mode off` and is named by
`RFDIFFUSION3_STOCK_PYTHON`; the KIT interpreter (the `python` on `PATH`) carries the ten-file overlay under `site-packages/rfd3/model/`
(eight upstream files replaced, `hoist.py` and `cudagraph_sampler.py` added, originals kept as `<file>.hoist_orig`, the install recorded
in `site-packages/rfd3/.xattempt_addon_overlay.json`) and runs `exact` / `fast`. The container image holds both — `/opt/stock` pristine,
`/opt/kit` with the overlay applied and checked at build — and the kit tree at `/kit/rfdiffusion3` (from the directory holding `rfdiffusion3/`
and `common/`: `docker build -f rfdiffusion3/environment/Dockerfile -t rfdiffusion3-kit:dev .`; `apptainer build rfdiffusion3-kit.sif
rfdiffusion3/environment/apptainer.def` converts that image). Route C prerequisites: a released CPython 3.12.1 exactly (`stock/check_pins.py` compares major.minor.micro; the image takes
python-build-standalone 20240107 and `uv venv --managed-python --python 3.12.1` fetches the same family, with the headers Triton needs
bundled; pyenv's or conda's own 3.12.1 builds serve as the alternative, headers included; apt offers no 3.12.1); an NVIDIA driver that supports CUDA 13.0 — torch's wheels
carry the CUDA 13.0 libraries, so no toolkit is needed at run time; nvcc 13.0 with gcc and ninja only to build the apex wheel; a C compiler
on `PATH` at run time (Triton compiles kernels at first use); `git` and `curl` for the block's clones and the uv installer; network access for pip. The container image sets `PYTHONHASHSEED=0`,
`PYTHONUNBUFFERED=1` and `RFDIFFUSION3_STOCK_PYTHON=/opt/stock/bin/python` for every process (`environment/Dockerfile` `ENV`); the exports
at the end of the block give Route C the same.

Two environments, made by running the block below twice from the directory holding `rfdiffusion3/` and `common/` — pass 1 the pristine one,
pass 2 the kit one; about half an hour in all on 8 cores, most of it the one apex build of pass 1: the stack and `run.sh install` go into BOTH
(`--mode off` starts `python -I -m rfdiffusion3_opt.stock_design` in the pristine one), the apex wheel into both, `install apply` into the kit one only:

    sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git build-essential   # once per host (bare Ubuntu; as root drop sudo; skip what you have): nvcc and the CUDA 13.0 development packages for the apex build come from NVIDIA's CUDA apt repository (the package names `environment/Dockerfile`'s apex stage installs), ninja from pip
    command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; } # once per user: uv itself (skipped when present)
    uv venv --seed --managed-python --python 3.12.1 ~/rfd3-stock && . ~/rfd3-stock/bin/activate   # pass 1 = this line as written, the pristine environment; pass 2 = the whole block again with ~/rfd3-kit in place of ~/rfd3-stock (any two paths); --managed-python takes uv's own released build even when a system python3.12 exists; alternative: pyenv's or conda's 3.12.1 (headers included)
    [ -d foundry ] || git clone https://github.com/RosettaCommons/foundry.git; git -C foundry checkout 4010e3e2e7350edada3e25a45c908c6bf407df4d   # both passes: pass 2 finds the clone and only checks out the pin again
    pip install -c rfdiffusion3/environment/requirements.lock './foundry[rfd3]'    # both passes: torch 2.13.0+cu130, triton 3.7.1; or from the archive: tar -xzf rfdiffusion3/stock/foundry-4010e3e.tar.gz, cd foundry-4010e3e2, SETUPTOOLS_SCM_PRETEND_VERSION=0.2.1.dev13+g4010e3e2e pip install -c ../rfdiffusion3/environment/requirements.lock '.[rfd3]'
    [ -d apex ] || git clone https://github.com/NVIDIA/apex.git; git -C apex checkout a1d527a857e8da64c4e7237ca89ec699fb4d9eaf   # pass 1 only: no public apex wheel exists for this torch and none is shipped — build it once, about ten minutes on 16 cores; needs nvcc 13.0, gcc and ninja
    (cd apex && pip install ninja==1.13.0 && APEX_CPP_EXT=1 APEX_CUDA_EXT=1 TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" pip wheel --no-deps --no-build-isolation -w ../rfdiffusion3/stock/wheels .)   # pass 1 only: ninja, the build tool, from pip at the version `environment/Dockerfile` uses
    pip install --no-deps rfdiffusion3/stock/wheels/apex-0.1-cp312-cp312-linux_x86_64.whl   # both passes (pass 2 installs the wheel pass 1 built); `environment/Dockerfile` builds it the same way, or installs a wheel placed there beforehand with `--build-arg WHEELS_FROM=prebuilt` (sha256 as in stock/PINS.json)
    (cd rfdiffusion3 && bash run.sh install)                                              # both passes: README step 2 types it once more with `--weights` for the checkpoint (the pip step is then skipped); = pip install -e ../common/opt_core -e opt (ships rfdiffusion3_opt_autoload.pth; the install must stay editable), then stock/check_pins.py --expect any; `install --weights DIR` also fetches the checkpoint
    python -I rfdiffusion3/stock/check_pins.py --expect pristine                  # both passes, before `install apply`: the eight target files at upstream's sha256 and the kit's two new files absent (after `apply`, `--expect any`)
    (cd rfdiffusion3 && python -m rfdiffusion3_opt.install apply)                         # pass 2 only (the kit environment): README step 2's apply line then finds it applied and changes nothing; runs opt/forward/xattempt_addon/install.sh (also: status | classify | rebind | check-only | uninstall; --python <interpreter>); an earlier install of this overlay is re-based by `rebind`, an rfd3 file in any other state is refused (install.sh exit 4)
    export RFDIFFUSION3_STOCK_PYTHON=$HOME/rfd3-stock/bin/python RFD3_CKPT=/weights/rfd3/rfd3_latest.ckpt PYTHONHASHSEED=0 PYTHONUNBUFFERED=1   # both passes: pass 1's python; the checkpoint where `run.sh install --weights /weights/rfd3` (README step 2) put it
    source rfdiffusion3/configs/h100.env                                          # pass 2 and every later shell, kit interpreter on PATH; exits 3 when rfdiffusion3_opt, or the opt_core it pins (`opt/pyproject.toml` `[tool.opt_core]`), is not importable

One image and one recipe serve H100, H200 and A100; `configs/h200.env` and `configs/a100.env` are `configs/h100.env` with their card's
`MODEL_OPT_TARGET_GPU` word.

## How stock is run

`--mode off` executes `rfd3 design <the line's key=value tokens, verbatim and in order>` in the PRISTINE interpreter under `python -I`
(`python -I -m rfdiffusion3_opt.stock_design`, the one module of the kit that runs in a stock process): it strips the kit's names
(`stock/PINS.json` "stock_environment" "must_be_absent": the `RFD3_*` lever switches, `RFDIFFUSION3_OPT`, the recipe's two
variables) from the child and proves them absent — upstream's own switches (`RFD3_DENSE_SDPA_ATTENTION`, `RFD3_LOW_MEMORY_MODE`) and
torch's variables pass through as the caller set them and are reported on the line —, proves the interpreter's `rfd3` tree pristine
by sha256 (`[rfdiffusion3-opt stock] STOCK rc-foundry=<version> tree=pristine …`, recorded in
`opt_manifest.json` `stock_env_proof`; `NOT STOCK: …`, exit 3, otherwise), attaches logging handlers to upstream's own loggers for the
read-only `KERNELS` line, and calls upstream's command-line app in-process. No upstream function is wrapped or re-bound, no import hook
is set, no argument, tensor or random draw changes, and the run's exit is its own. There is no `default` mode: upstream's
`compile_model=true` on an `off` line is upstream's own `torch.compile` of its four diffusion submodules (route `stock`); without it the
route is `default`. `--det 1` under `off`: the child sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, torch's deterministic algorithms, cuDNN
deterministic and the fixed-order `scatter_mean` before upstream is imported — the same recipe the kit modes apply.

The kit has no preset and no name of its own for an upstream setting: every mode hands upstream the line's tokens and adds only
`ckpt_path=<--ckpt | RFD3_CKPT>` when the line names none, so the file `opt_manifest.json` hashes is the file the engine loads.

## Stock exceptions

None. The kit carries no upstream fix and no stock-side setting; `off` runs the pin as shipped.

## Variables

`configs/h100.env` (`configs/a100.env`, `configs/h200.env`: the same values with their card's target word) fills what is unset — every value
can be pre-set.
The `RFDIFFUSION3_*` and `RFD3_CKPT` rows are read by `run.sh` or the package and set by the user.

| variable | required | default | effect |
|---|---|---|---|
| `RFD3_CKPT` | yes (`design`, `warm`) | none | the weights file above; `design` / `warm` refuse without it or `--ckpt` (exit 2); `check` needs none; upstream's `FOUNDRY_CHECKPOINT_DIRS` follows its directory unless pre-set |
| `RFDIFFUSION3_STOCK_PYTHON` | for `--mode off` | the `python` on `PATH`, proved pristine or refused (exit 3) | the pristine interpreter's `python`, run under `python -I` |
| `RFDIFFUSION3_OPT` | no | unset: `fast` for the package's verbs, stock on upstream's own command line | the mode when `--mode` is absent (`--mode` wins); in the kit interpreter `RFDIFFUSION3_OPT=<mode> rfd3 design …` activates the kit at `import rfd3` through `rfdiffusion3_opt_autoload.pth`; a value that is not a mode exits 3 at interpreter start |
| `RFDIFFUSION3_OPT_DET` | no | unset (`0`) | `1` = `--det 1` on upstream's own command line under `RFDIFFUSION3_OPT=<mode>`; the stock child takes `--det` instead |
| `RFDIFFUSION3_OPT_ALLOW_PARTIAL` | no | unset | `1` = `--allow-partial` (the only form on upstream's own command line) |
| `RFDIFFUSION3_OPT_CACHE` | no | `$XDG_CACHE_HOME/rfdiffusion3_opt`, else `~/.cache/rfdiffusion3_opt` | directory of the checkpoint-digest cache `checkpoint_sha256.json` (the 2.7 GB hash is paid once per path, size and mtime) |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` / `H200` (`configs/<card>.env`) | the GPU class the configuration targets; `check` reports the machine's; the levers read the card from the device |
| `MODEL_OPT_JIT_ROOT` | no | unset (`run.sh`, and the configuration for the three caches below, then use a private per-user root `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`, made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the tools keep their own default locations) | root of persistent JIT caches; must be writable when set — the Triton, torch-extensions and inductor caches are created under it per GPU / stack key on first run: `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, `TORCHINDUCTOR_CACHE_DIR` default to `<root>/$MODEL_OPT_STACK_KEY/{triton,torch_extensions,inductor}`; a pre-set cache directory is kept when it or its parent exists, else dropped by name; `KERNELS … inductor_cache=cold\|warm` says which |
| `MODEL_OPT_STACK_KEY` | no | derived, e.g. `torch2.13.0-cu130-sm90` | `torch<version>-cu<CUDA>-sm<cc>` of the interpreter on `PATH`, read through torch from the visible GPU when unset; the configuration exits 2 rather than guess it — preset it (e.g. `MODEL_OPT_STACK_KEY=torch2.13.0-cu130-sm90`, the H100 / H200 key on the pinned stack; `…-sm80` on the A100) to source the configuration on a host without a GPU, e.g. an image build step |
| `MODEL_OPT` | no | derived | this `rfdiffusion3/` directory (how the package finds the overlay and the pins) |
| `PYTORCH_CUDA_ALLOC_CONF` | no | `expandable_segments:True` | torch's allocator setting: the captured step's private pool persists across a pass's designs and the next design's initialiser runs beside it, which the default allocator fragments at long lengths; applies to every verb's process, the `--mode off` child included; placement only, no effect on numerics; read back as `alloc_conf=` on the `ACTIVE` and `KERNELS` lines |
| `PYTHONDONTWRITEBYTECODE` | no | `1` | no `.pyc` written into the tree |

Inherited copies of the recipe's variables (`FOUNDRY_DET_SCATTER`, `CUBLAS_WORKSPACE_CONFIG`) are reported under the kit modes (`det_env`)
and stripped from the stock child; `--det 1` sets them itself on every route. Under `exact` / `fast` a kit switch (`RFD3_*`) preset at a
value other than the mode's, or one of upstream's two attention switches set at all, is refused by name (exit 3); under `off` upstream's
switches are the caller's and pass through.
