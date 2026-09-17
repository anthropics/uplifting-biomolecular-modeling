# Borzoi — stock, as pinned

## Pin

Upstream: calico/borzoi @ `5c9358222b5026abb733ed5fb84f3f6c77239b37` (`v1.0.0-7-g5c93582`; installs as
`borzoi 1.0.1.dev7+g5c9358222`) on calico/baskerville @ `544073b87245d9f43ba63442c75bdbf32a9f8720` (tag `v1.0.0`; installs
as `baskerville 1.0.0`), shipped under `stock/` as source archives `borzoi-5c93582.tar.gz` and `baskerville-544073b.tar.gz`
(`git archive` of each commit; sha256 in `stock/PINS.json`). Stock entry point: `$BORZOI_DIR/src/scripts/borzoi_sad.py`
(sha256 `10c7016704d9972a225311ddda5b872ee585c76e1112f067a232a40ad5957911` — the pin that `--mode off`, the `BORZOI_OPT`
hook and `opt/datapath/pipeline_tf/build.py` check). Weights: `f0/model0_best.h5`, the published human fold-0 replicate
(sha256 in PINS.json), fetched by `bash run.sh install --weights DIR` from the URL recorded in PINS.json (the address upstream's README publishes), digest-checked, to
`DIR/f0/model0_best.h5` — the `<model_file>` argument (upstream's `download_models.sh` fetches every replicate; this step
fetches the one file; `python -m borzoi_opt.weights DIR` is the same step). Upstream's licence and notices as found in the
archives. `stock/` is never edited.

## Stack

Debian 12 (any current x86_64 Linux with an NVIDIA driver that runs CUDA 12.2), Python 3.10.17, tensorflow 2.15.1, keras 2.15.0, numpy 1.24.4,
h5py 3.10.0, pandas 1.5.3, pysam 0.22.1, scipy 1.9.3, `bedtools` on `PATH` — the full list is `environment/requirements.lock`; CUDA 12.2 and
cuDNN 8.9 come as the lock's `nvidia-*-cu12` wheels, which TensorFlow loads from site-packages (no CUDA toolkit is installed). `environment/Dockerfile`
and `apptainer.def` build all of it (one image for every configured card; nothing is compiled per GPU). By hand — a fresh venv, or an existing
environment that already runs Borzoi at the pin (then only the lines it lacks) — in bash, from the directory holding `borzoi/`:

    sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y curl bedtools build-essential gfortran git libbz2-dev libcurl4-openssl-dev liblzma-dev libssl-dev zlib1g-dev   # bare Ubuntu host (root: no sudo); skip what you have
    command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); the installer runs only when its sha256 matches
    uv venv --seed --managed-python --python 3.10.17 ~/borzoi-env && . ~/borzoi-env/bin/activate   # uv's own CPython 3.10.17; or apt/deadsnakes python3.10 + python3.10-venv + python3.10-dev (no pre-release); other patch levels are reported (`STACK not pinned`), not refused
    pip install --no-deps "$(grep -E '^pip==' borzoi/environment/requirements.lock)" "$(grep -E '^setuptools==' borzoi/environment/requirements.lock)"   # the lock's pip and setuptools first: pybedtools (setup.py, no pyproject) builds with the environment's setuptools
    echo 'setuptools==84.0.0' > /tmp/build-constraints.txt                                   # build-time setuptools for pip's isolated (pyproject) source builds, the version the Dockerfile pins
    PIP_CONSTRAINT=/tmp/build-constraints.txt pip install --no-deps -r <(grep -v -E '^(#|-e |setuptools==)' borzoi/environment/requirements.lock)   # the pinned stack, nothing resolved
    mkdir -p ~/src/baskerville ~/src/borzoi                                                   # each archive holds one top directory; unpacked flat:
    tar -xzf borzoi/stock/baskerville-544073b.tar.gz -C ~/src/baskerville --strip-components=1
    tar -xzf borzoi/stock/borzoi-5c93582.tar.gz -C ~/src/borzoi --strip-components=1
    PIP_CONSTRAINT=/tmp/build-constraints.txt SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BASKERVILLE=1.0.0 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BORZOI=1.0.1.dev7+g5c9358222 \
      pip install --no-deps -e ~/src/baskerville -e ~/src/borzoi                              # git checkouts of the two commits instead: the same line without the SETUPTOOLS_SCM variables
    export BORZOI_DIR=~/src/borzoi PATH=~/src/borzoi/src/scripts:$PATH                         # upstream runs its scripts by name; the image sets both

pip builds the two checkouts in isolated environments and fetches their build backend (`setuptools`, `setuptools_scm`, per the
archives' `[build-system]` table) from the index at install time, so that step needs network; the constraints file holds the build-time `setuptools` at one version here and in `environment/Dockerfile`
alike, while the lock pins the run-time packages only. Then, from inside `borzoi/`, `bash run.sh install`
(the kit editable, built by its own in-tree backend with `--no-build-isolation` — nothing fetched — then `python -I stock/check_pins.py`, which
confirms the installed borzoi and baskerville match the archives file for file and the entry script matches its digest;
another Python or package version is reported as `STACK not pinned: …`, not refused). The image also sets `PYTHONHASHSEED=0`, `CFLAGS=-g0` and
`TF_CPP_MIN_LOG_LEVEL=1` (hash seed, source-build flags, quieter TensorFlow logs) and `BASKERVILLE_DIR` / `PYTHONPATH` naming the two
script directories for upstream's other scripts; the kit requires none of them — export them to reproduce the image's process environment
exactly. Other cards: nothing to build; `configs/a100.env` and `configs/h200.env` differ from `configs/h100.env` in `MODEL_OPT_TARGET_GPU` only.

## How stock is run

`--mode off` executes the pinned `borzoi_sad.py` with your arguments as `__main__` under `python -s`, in a clean subprocess
with nothing of the kit importable: `BORZOI_OPT` and `KIT_*` stripped, no kit module or directory on the path, TensorFlow
not yet imported, entry sha256 = the pin — stated on one `[borzoi-opt stock] ENV-CLEAN ok: …` line before the script starts.
The documented call is `borzoi_sad.py -f <hg38.ml.fa> --rc --stats SAD,logSAD,D2,logD2 -t $BORZOI_DIR/examples/targets_human.txt
-u -o <out_dir> $BORZOI_DIR/examples/params.json <model0_best.h5> <vcf>` (`-u` is a bare flag; `<out_dir>` is created
non-recursively and `sad.h5` opened `'w'`). TensorFlow 2.15 defaults throughout: float32 model, TF32 where the card enables
it, cuDNN autotuning on, eager execution, no XLA or mixed precision; one SNV per model call as the (ref, alt) batch of 2
(`train.batch_size` in `params.json`); the genome is upstream's `hg38.ml.fa` build (contigs `chr…`; pysam writes the `.fai`
beside it on first use). `--det 1` under any mode, `off` included, exports
`NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR,AVX2,FMA3,X86_V3,X86_V4`,
`OPENBLAS_CORETYPE=Haswell`, `OPENBLAS_NUM_THREADS=1` and `TF_CUDNN_USE_AUTOTUNE=0` into the job (`borzoi_opt/modes.py`
`DET_RECIPE`): numpy's CPU dispatch pinned to one instruction set, one BLAS thread, cuDNN's convolution algorithms chosen by
heuristic instead of per-process timing.

## Stock exceptions

None. (The `SETUPTOOLS_SCM_PRETEND_VERSION_*` variables only give the archive installs the version strings a git checkout
produces; they change nothing at run time.)

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `BORZOI_DIR` | no | unset → `borzoi_sad.py` on `PATH` | upstream's variable; locates the stock entry `$BORZOI_DIR/src/scripts/borzoi_sad.py` |
| `BORZOI_OPT` | no | unset | the mode (`exact` / `off`); equals `--mode`, a disagreeing pair is refused (exit 2); unset under run.sh / `borzoi-opt` means `exact`, unset under a bare stock command means stock |
| `BORZOI_OPT_ALLOW_PARTIAL` | no | unset | `1` = `--allow-partial` |
| `MODEL_OPT` | no | this directory (found from the editable install; run.sh and the configs export it) | where the kit and `stock/` are |
| `MODEL_OPT_TARGET_GPU` | no | unset | `H100` / `A100` / `H200`, set by `configs/<card>.env`; a card that is not the target is a `notes=` entry on the ACTIVE line |
| `PYTHONDONTWRITEBYTECODE` | no | `1` from the configs | no `.pyc` written into the checkouts |
| `BORZOI_HG38` | no | stock's | stock's own default for `-f`, passed through untouched |
| `KIT_*` | — | set per mode by the package (`KIT_FWD=1`, `KIT_STAMP_DIR`) | internal switches; a `KIT_*` name the kit reads, set by the user, is refused by name |

`configs/<card>.env` returns 2 when `borzoi_opt` is not importable on the `python` on `PATH`.
