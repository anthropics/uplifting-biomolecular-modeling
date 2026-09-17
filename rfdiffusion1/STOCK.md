# RFdiffusion-1 — stock, as pinned

## Pin

Upstream: https://github.com/RosettaCommons/RFdiffusion at commit `86507b6538f51fce57b5a72477165f03999ed7ae` (`setup.py` version 1.1.0; no tag at that
commit), shipped under `stock/` as the source archive `stock/rfdiffusion-86507b65.tar.gz` — `git archive` of `LICENSE README.md setup.py rfdiffusion/ scripts/
config/ env/ docker/`, per-file digests in `stock/PINS.json` — including its vendored NVIDIA SE3Transformer 1.0.0 (`env/SE3Transformer`, MIT). `stock/src/`
holds the entry script, `config/inference/base.yaml` and the network and sampler modules the levers wrap, unpacked for reference. Weights:
`Complex_base_ckpt.pt` (upstream selects it whenever `ppi.hotspot_res` is given) and `Base_ckpt.pt` (upstream's default without hotspots), size and sha256
in `stock/PINS.json`, fetched by `bash run.sh install --weights DIR` from upstream's download URLs (the files upstream's `scripts/download_models.sh` lists)
and digest-checked; the other checkpoints that script lists serve requests that run under `--mode off` here — fetch them into the same directory with
upstream's script when needed. Upstream's licence and notices as found inside the archive. Terms: the upstream `LICENSE` (BSD-3-Clause; text at
`third_party_licenses/RFdiffusion.BSD-3-Clause.txt` and `stock/src/LICENSE`) states that it covers both the source code and the model weights referenced for
download in the upstream README. `stock/` is never edited.

## Stack

Ubuntu 22.04 with an NVIDIA driver for CUDA 12 (550 or newer; no CUDA toolkit — the wheels carry the runtime), git, CPython 3.11 (tested 3.11.5; required, the lock's dgl wheel
is cp311; any released 3.11 build — python.org, deadsnakes, conda, or uv's as below), torch 2.4.0 (PyPI's CUDA 12.1 wheel), dgl 2.4.0+cu124, triton 3.0.0,
e3nn 0.5.1, hydra-core 1.3.2 — the full list is `environment/requirements.lock`; `environment/Dockerfile` builds it into one image for every configured card (nothing is compiled
for a GPU at build time; no weights inside) and `environment/apptainer.def` converts that image. A fresh environment at the pin, from the directory holding `rfdiffusion1/` and
`common/` (about 6.0 GB on disk, mostly the torch and nvidia-* wheels):

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl build-essential git   # bare Ubuntu host (as root: no sudo); skip what you have — the image's apt packages, plus git for the clone below
command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; } # uv itself, once per user (skipped when present)
uv venv --seed --managed-python --python 3.11.5 rfd1 && . rfd1/bin/activate  # uv's own released CPython; any released 3.11 works instead (python.org, conda; with apt/deadsnakes: python3.11 + python3.11-venv + python3.11-dev) — not Ubuntu 22.04's python3.11 apt package (3.11.0rc1)
pip install --no-deps -r rfdiffusion1/environment/requirements.lock       # one pass; --no-deps installs exactly the locked builds (dgl's line names its wheel by URL and digest)
git clone https://github.com/RosettaCommons/RFdiffusion && git -C RFdiffusion checkout 86507b6538f51fce57b5a72477165f03999ed7ae   # git's detached-HEAD notice is expected
pip install --no-deps -e RFdiffusion/env/SE3Transformer -e RFdiffusion    # upstream's own two installs: the vendored SE3Transformer, then the checkout, editable
export PYTHONHASHSEED=0 CFLAGS=-g0 CC=gcc CXX=g++                        # the image's process environment (environment/Dockerfile ENV): fixed hash seed, no debug info in Triton's launchers, gcc/g++ for its compile
cd rfdiffusion1 && bash run.sh install --weights /abs/path/to/weights       # = README step 2's cd and install lines — type them once; step 2 continues at its export WEIGHTS line, then check
```

An environment that already runs RFdiffusion at the pin works the same, from inside `rfdiffusion1/`: `pip install --no-deps -r environment/requirements.lock` for any missing line, then `bash run.sh install`,
whose pin check (`stock/check_pins.py`) confirms the checkout matches the archive file for file and torch / dgl match the pin (`install` exits 3 off the pin, `check` prints the
same report without refusing). pip fetches the build backend (setuptools, and wheel for the kit package) from the index at install time — RFdiffusion's two `setup.py` installs
and the two editable kit packages build in pip's isolated environment, so the install needs network access; the lock pins runtime packages only. `fast` needs a C compiler on
PATH (`CC`, else gcc) for Triton's first-use compile; `design --pack K` needs `nvidia-cuda-mps-control`. A bare `import dgl` outside the kit's runs prints DGL's backend-selection
notice; the kit sets `DGLBACKEND=pytorch` for its own processes.

Under Apptainer (README route B) the SIF is read-only and `apptainer run` starts in `/kit/rfdiffusion1`, so three writes need a host path. Upstream writes its IGSO(3) schedule
cache into the checkout (`/opt/rfd/schedules`) on first use: bind a host directory over it (`mkdir -p $PWD/out/schedules`, then `-B $PWD/out/schedules:/opt/rfd/schedules`) or
type `inference.schedule_directory_path=<absolute host dir>` on any mode. `off` runs upstream's hydra entry point, which writes its `outputs/<date>/<time>/` log directory into
the working directory: type `hydra.run.dir=<absolute host dir>` on `off` lines (the kit modes write none and refuse typed `hydra.*` keys). `fast` links its Triton launchers with
`-lcuda` and the runtime image ships no `libcuda.so`. A SIF built from `environment/apptainer.def` links `/opt/culib/libcuda.so` to the driver `--nv` binds and sets
`TRITON_LIBCUDA_PATH` itself, so route B needs nothing extra; the manual form below is only for an image converted straight from the Docker image (`apptainer build …
docker-daemon://…` or `docker-archive://…`) or for route C run under Apptainer: once on the host, `mkdir -p ~/libcuda && ln -sf /.singularity.d/libs/libcuda.so.1
~/libcuda/libcuda.so`, then pass `--env TRITON_LIBCUDA_PATH=$HOME/libcuda`. `WEIGHTS` is exported on the host (Apptainer passes it in) and its directory bound when it lies
outside `$HOME` and `$PWD` (bound by default); each attempt takes a fresh output prefix directory (the `run.log` of a failed attempt is read again by the next). README's route B
wraps all of this in its `kit()` shell function — the weights bind, `$PWD/out` over `/kit/rfdiffusion1/out` (so the Run lines' relative `out/…` prefixes and
`hydra.run.dir=out/off/hydra` land in `$PWD/out`) and the schedule-cache bind; without the `out` bind those two must be absolute host paths under a bound directory.

The image build takes optional pre-filled compile caches, `_jitcache/rfdiffusion1-*-jit.tar` in the build context, unpacked under `/opt/jit_cache/<stack key>/…` (the build is
identical without one); in the image `run.sh` picks the JIT root before any verb — a set `MODEL_OPT_JIT_ROOT` is kept (and seeded once from `/opt/jit_cache` when empty),
otherwise `/opt/jit_cache` itself when it is writable, else a copy seeded once under `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` — prints `[rfdiffusion1-kit] jit cache: <dir> (in-image |
seeded from image | user)`, and with `--config <card>` Triton's cache is `<dir>/<stack key>/triton` as in "Variables". A preset `MODEL_OPT_JIT_ROOT` the process cannot write is
only read: with `--config <card>` its `<stack key>` subtree (up to `MODEL_OPT_JIT_SEED_MAX_FILES` files, default 5000) is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`
(printed `(seeded from read-only root)`), otherwise the read-only root is used as is (nothing is compiled into it; caches already there are read; no line is printed) — and
`TRITON_CACHE_DIR` follows the root in use. On route C there is no `/opt/jit_cache`, so `run.sh` prints no jit cache line; `off` and `exact` write no compile cache on any route,
and `fast` compiles its Triton kernels once per machine under `$MODEL_OPT_JIT_ROOT/<stack key>/triton` when `--config <card>` is given (`run.sh` always has a root: yours, else the
private per-user one it silently supplies), else into Triton's own cache (`~/.triton`).

## How stock is run

`run.sh design|check|warm [--config h100|a100|h200] [--mode M] … calls `python -m rfdiffusion1_opt <verb> …` (`rfdiffusion1-opt`; the kit's flags: CHANGES.md
"Switches"); `run.sh install [--weights DIR]` is run.sh's own arm (the pip installs, `stock/check_pins.py`, then `python -m rfdiffusion1_opt.weights DIR`). `--mode off` executes `python -s $RFD_ROOT/scripts/run_inference.py
<the typed overrides>` in a clean subprocess with nothing of the kit importable
(`opt/rfdiffusion1_opt/stock_cli.py`): the caller's environment minus the kit namespace (`RFD_*`, `RFDIFFUSION1_*`, `ALLOW_ANY_GPU`; `RFD_ROOT` and
`WEIGHTS` pass through), `DGLBACKEND=pytorch`, and `inference.model_directory_path=$WEIGHTS` added when not typed; the exit code is upstream's.
`--det 1` under `off` adds `inference.deterministic=True` to that line (upstream seeds torch, numpy and `random` with the design index before every
design); `--det 0`, the default, is upstream's unseeded default. The kit line takes the same key the same way. TF32 under `off` is torch's default
(matmul off, cuDNN on); the resident driver sets both off under `exact` and both on under `fast`, in its own process only, and drops
`NVIDIA_TF32_OVERRIDE` / `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` from that process. `exact` = `off` byte for byte holds at the same seed and design position
within one CPU host class: upstream's IGSO(3) schedule cache and per-step CPU math depend on the host's instruction set (`upstream_issues/RFD1-002_igso3_cache_write.md`), so
cross-host settings (`MKL_CBWR` and friends, a fresh `inference.schedule_directory_path=`) are the caller's to export and type; they reach both arms.

## Stock exceptions

None. No setting or patch is needed to run stock at the pin on this stack; the kit modes wrap upstream's functions at run time inside their own driver
process, and the checkout on disk is never edited.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `WEIGHTS` | yes, unless `inference.model_directory_path=` is typed | — | directory holding `Complex_base_ckpt.pt` and `Base_ckpt.pt`; `design`, `warm` and `check` refuse naming it when unset (exit 3; only the variable is checked, not the files); `install` reads it only through `--weights` |
| `RFD_ROOT` | no | pip's editable install of `rfdiffusion` | the RFdiffusion checkout at the pin; neither present → refused naming `RFD_ROOT` |
| `RFDIFFUSION1_OPT` | no | unset (= `fast`) | the mode when `--mode` is absent; a `--mode` that disagrees with it is refused (exit 2) |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `A100` from `configs/<card>.env` | the card the configuration targets; `check` reports a mismatch with the card found, never refuses on it |
| `MODEL_OPT_STACK_KEY` | no | derived `torch<version>-cu<CUDA>-sm<cc>`, e.g. `torch2.4.0-cu121-sm90` | JIT-cache key (taken from an existing `<root>/<key>/torch_extensions` layout when `TORCH_EXTENSIONS_DIR` points into one); underivable → noted, default caches |
| `MODEL_OPT_JIT_ROOT` | no | unset: `run.sh` then supplies a private per-user root, `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` (made with mode 0700; one another user owns, that group or others can write, or that is a symbolic link is refused by name and the run then has no root) — the tools' own default locations this row names apply only without `run.sh` or when that root is refused | persistent JIT cache root; if set it must be writable: `TRITON_CACHE_DIR` = `$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton` (when `TRITON_CACHE_DIR` is unset) is created under it per stack key — torch, CUDA and GPU capability — on the first `fast` run; left unset, Triton's default cache directory is used |
| `TRITON_CACHE_DIR` | no | as above, else Triton's default | Triton cache for `fast`'s kernels; a pre-set value is used as is |
| `MODEL_OPT_PACK_WORKER_GB` | no | `10` | `design --pack K`: the per-worker footprint (GB) the packing launcher's memory estimate multiplies; over the card → noted, never refused |
| `MODEL_OPT` | no | this `rfdiffusion1/` directory | where the package finds the kit code and `stock/PINS.json`; set by `run.sh` and the configs |

Kit-declared variables plus the two stock variables the kit reads (`RFD_ROOT`, `WEIGHTS`). `configs/h100.env` fills the `MODEL_OPT_*` names above when
unset and returns 3 only when `rfdiffusion1_opt` or the pinned shared core is not importable; `configs/a100.env` and `configs/h200.env` set `MODEL_OPT_TARGET_GPU`
(`A100`, `H200`) and source it. The drivers' internal `RFD_*` knobs are not user variables: they are stripped from every child process except the mode's own row.
