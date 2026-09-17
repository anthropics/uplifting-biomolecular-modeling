# ChromBPNet — stock, as pinned

## Pin

Upstream: https://github.com/kundajelab/chrombpnet tag `v1.0.1` = commit `eaa0fe58b6a43da62ea23b75cfc2bef4ecd3550c`, shipped under
`stock/` as a source archive (`stock/chrombpnet-eaa0fe58.tar.gz`, `git archive` of the tag; sha256 in `stock/PINS.json`) with upstream's
licence and notices inside it as found; `stock/` is never edited. No wheel exists for the tag: the checkout is the pin. The kit covers
the console script's `pred_bw` subcommand; every other subcommand is stock only.

Weights: three Keras files under `$CHROMBPNET_OPT_WEIGHTS/GM12878_ATAC/fold_0/` (sha256 in `stock/PINS.json` `weights`) —
`chrombpnet_recompiled.h5`, the `-cm` model (the two component models recombined into one Keras file: `bash run.sh install --weights DIR`
builds it from them when it is absent, composing them as upstream does for training, on the CPU, the same bytes every time under the
pinned stack; ENCODE's own combined member is a Keras 2.4 file the pinned stack cannot load), and `bias_scaled.h5` / `nobias.h5`, the
component models, byte for byte the members
`model.bias_scaled.fold_0.ENCSR637XSC.h5` / `model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5` of the ENCODE archive ENCFF142IOR (the
Triton forward loads them from beside `-cm`). Upstream ships no downloader for trained models: `bash run.sh install --weights DIR` builds
the `-cm` model when absent (DIR must then be writable), checks the three digests and fetches nothing; a `chrombpnet_recompiled.h5` of
another digest is reported `MISMATCH … (left in place)` — delete it and re-run `install --weights DIR`, it is rebuilt from
`bias_scaled.h5` + `nobias.h5`. Optional: `$CHROMBPNET_OPT_WEIGHTS/cache/nv_compute_cache_<GPU>.tar`, a captured CUDA driver
ComputeCache for the TensorFlow routes (CHANGES.md `jit_cache`); results are identical with or without it.
Terms of the weights: the ENCODE file record names no licence; the ENCODE portal's data-use policy and citation guidance apply
(https://www.encodeproject.org/help/citing-encode/ — acknowledge the ENCODE Consortium and the producing laboratory, cite the accessions).

## Stack

Ubuntu 20.04, CUDA 11.2 / cuDNN 8.1 (base `tensorflow/tensorflow:2.8.2-gpu`), Python 3.8.10, tensorflow 2.8.0 (re-pinned by the tag's
`requirements.txt`), numpy 1.23.4, h5py 3.11.0, pyfaidx 0.6.1, pyBigWig 0.3.22, protobuf 3.20.0 — the full list is
`environment/requirements.lock` (stack `s2` in `stock/PINS.json`, enough for `off`). Stack `s1` adds torch 2.4.1+cu124 / triton 3.0.0
from `environment/requirements-opt-torch.lock`, installed with `--target /opt/torch` off the default `sys.path`: the kit inserts it for
its Triton forward, stock never sees it. `environment/Dockerfile` builds `s1` and installs the kit; `environment/apptainer.def` converts
that image. The image build takes an optional pre-filled compile cache `_jitcache/chrombpnet-*-jit.tar` (one per stack key) from the build
context and unpacks it to `/opt/jit_cache` (the build is identical without one); in the image `run.sh` uses `/opt/jit_cache` in place when it
is writable, else seeds a JIT root from it once — `MODEL_OPT_JIT_ROOT` when the caller sets one (seeded only if empty, left alone if not),
otherwise `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` — exports `MODEL_OPT_JIT_ROOT` and prints `[chrombpnet-kit] jit cache: <dir> (<how>)`; with a
root set, the Triton forward's cache is `TRITON_CACHE_DIR=<root>/torch2.4.1-cu124-<sm90|sm80>/triton` unless `TRITON_CACHE_DIR` is already
set; a preset `MODEL_OPT_JIT_ROOT` the process cannot write is never compiled into — its `$MODEL_OPT_STACK_KEY` subtree (when that
variable is set and holds at most `MODEL_OPT_JIT_SEED_MAX_FILES` files, default 5000) is seeded once into `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` and a
`TRITON_CACHE_DIR` the config derived from the preset root follows the move, otherwise the read-only root is used as is (nothing is
compiled into it; caches already there are read); the printed `(<how>)`, when there is one, says which (no root and no image cache, or a
read-only root left as is: nothing printed — with no preset root `run.sh` silently supplies the private per-user one; Triton's own default only when that is refused). Into an existing environment that already runs ChromBPNet 1.0.1 at the pin (Python 3.8 with the CUDA 11.2 / cuDNN 8.1
libraries), from `chrombpnet/`:

```bash
python3 -m pip install --no-deps --target /opt/torch -r environment/requirements-opt-torch.lock   # the Triton forward's stack (≈4.9 GB; /opt/torch is the one side root the kit searches: `sudo mkdir -p /opt/torch && sudo chown $USER /opt/torch` first on a host where /opt is root's)
export PYTHONHASHSEED=0 CFLAGS=-g0   # the two process variables the image sets (environment/Dockerfile ENV): hash seed fixed, no debug info in the launchers Triton compiles
bash run.sh install [--weights DIR]     # = README step 2's install line — type it once; step 2 continues at its export line (pip install -e opt unless already from this tree, stock/check_pins.py; --weights DIR builds the -cm model when absent, checks the three digests)
```

From a fresh Python 3.8 environment (README route C; the `uv venv` line below makes it), inside `chrombpnet/`: the stock stack first, then the block above. Outside pip it needs the CUDA 11 / cuDNN 8 system libraries TensorFlow 2.8.0 loads (its wheel brings none; the image has 11.2 /
8.1), an NVIDIA driver that runs CUDA 12.4 for the torch stack, and about 4.9 GB for `/opt/torch`; another Python 3.8 patch level or
CUDA 11 / cuDNN 8 minor is reported as drift and runs.

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y wget bzip2 ca-certificates curl git jq libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev   # bare Ubuntu host (root: no sudo); skip what you have — environment/Dockerfile's apt line; the CUDA 11.2 / cuDNN 8.1 libraries come from NVIDIA's repository, not this line
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user (skipped when present); the installer runs only when its sha256 matches
uv venv --seed --managed-python --python 3.8 ~/cbp-venv && . ~/cbp-venv/bin/activate   # uv's own released CPython 3.8 build, headers included; if you use apt/deadsnakes python instead of uv: python3.8 + python3.8-venv + python3.8-dev; the image's is 3.8.10, another patch level is reported as drift and runs
grep -E '^(pip|setuptools)==' environment/requirements.lock | xargs python3 -m pip install
grep -v -E '^(#|$)' environment/requirements.lock > /tmp/stack.txt && python3 -m pip install --no-deps -r /tmp/stack.txt
python3 -m pip install six==1.14.0 requests==2.22.0 urllib3==1.25.8 idna==2.8 chardet==3.0.4 certifi==2019.11.28 wheel==0.34.2   # the seven the image takes from Ubuntu packages, absent from the lock (its header): the same versions from PyPI
mkdir -p ~/src && tar -xzf stock/chrombpnet-eaa0fe58.tar.gz -C ~/src \
  && mv ~/src/chrombpnet-eaa0fe58 ~/src/chrombpnet && python3 -m pip install --no-deps ~/src/chrombpnet   # keep ~/src/chrombpnet: the pin check reads the install's recorded source directory
```

`stock/check_pins.py [--stack s1|s2]` prints one line per pin (`pinned`, or `DRIFT … (want …)`: a companion package, the Python
version, a CUDA library, `/opt/torch`) and exits 3 only when the installed `chrombpnet` is not the pin; `run.sh` stops every verb there
and proceeds on any other drift, stating it first. The kit package (`opt/`, `chrombpnet_opt`) has no dependencies of its own; its
install also lays down `chrombpnet_opt_autoload.pth`, the interpreter-start hook behind `CHROMBPNET_OPT=<mode> chrombpnet pred_bw …`.
bpnet-lite 1.0.0 and tangermeme 1.4.1 (the Triton forward's weight loader) ship vendored under `opt/kit/torch/vendor/` with their
licences. Other cards: `configs/<card>.env` changes `MODEL_OPT_TARGET_GPU` and the cache tarball's name; the stack is the same.

## How stock is run

`--mode off` executes `chrombpnet pred_bw <your arguments>` — the console script's own `main`, called through
`opt/chrombpnet_opt/stock_pred_bw.py` (stdlib only) — in a clean subprocess with nothing of the kit importable: kit- and
package-prefixed variables stripped, `PYTHONPATH` entries under this tree dropped, and the child re-checks that itself
(`stock_env_proof.json` beside the outputs; a kit directory on `sys.path`, a kit variable or a loaded kit module ends it with exit 3,
`ENV NOT CLEAN`). Stock as shipped: `load_model(compile=False)`, `model.predict(batch_size=-bs)` (default 64), TF32 convolutions, cuDNN
autotune, no XLA, no seed. `-bs 1024` is the largest power of two the model runs at (a larger batch overflows cuDNN 8.1's
tensor-descriptor element limit in the first convolution).

`--det 1` under `off`: stock has no deterministic switch of its own (`TF_DETERMINISTIC_OPS=1` alone makes TensorFlow 2.8's `load_model`
refuse for want of a seed), so the kit's `opt/kit_ho/tf/det_subprocess/` goes first on `PYTHONPATH` with `CHROMBPNET_DET_SUBPROCESS=1`
and `CHROMBPNET_DET_SEED=0`; its `sitecustomize.py` runs at interpreter start — `tf.random.set_seed(0)`, `enable_op_determinism()`,
TF32 off — and prints `[det_subprocess] applied seed=0;op_determinism=1;tf32=False;tf=2.8.0 pid=<pid>` on stderr. The environment
block `TF_DETERMINISTIC_OPS=1 TF_USE_DEFAULT_CONV_ALGO=1 TF_CUDNN_USE_AUTOTUNE=1 TF_XLA_FLAGS=--tf_xla_auto_jit=0 TF_CPP_MIN_LOG_LEVEL=0
TF_CPP_VMODULE=gpu_utils=2` completes it; `-bs` is untouched. `CHROMBPNET_OPT_DET=1` is the environment spelling. The kit modes take no
`--det`: `exact` composes the same block for its own process (plus `CUBLAS_WORKSPACE_CONFIG=:4096:8` on the Triton route).

## Stock exceptions

None. `off` adds no setting and no patch to stock; `--det 1` is opt-in and announced on stderr.

## Variables

| variable | required | default (`configs/<card>.env`) | effect |
|---|---|---|---|
| `CHROMBPNET_OPT_WEIGHTS` | yes | none — unset, the config prints `NOT ACTIVE … is not set` and returns 2 | weights root (§Pin) |
| `CHROMBPNET_OPT_DATA` | yes | none, likewise | data root the paths below derive from |
| `CHROMBPNET_OPT_MODEL` | no | `$CHROMBPNET_OPT_WEIGHTS/GM12878_ATAC/fold_0/chrombpnet_recompiled.h5` | the `-cm` model |
| `CHROMBPNET_OPT_GENOME` | no | `$CHROMBPNET_OPT_DATA/hg38.genome.fa` | reference fasta (`-g`), `.fai` beside it |
| `CHROMBPNET_OPT_CHROM_SIZES` | no | `$CHROMBPNET_OPT_DATA/hg38.chrom.sizes` | chromosome sizes (`-c`) |
| `CHROMBPNET_OPT_REGIONS` | no | `$CHROMBPNET_OPT_DATA/regions.bed` | 10-column narrowPeak regions (`-r`; also `warm`'s input) |
| `CHROMBPNET_OPT_BIGWIG` | no | `$CHROMBPNET_OPT_DATA/ENCFF180XQC.bigWig` | observed bigWig (`-bw`) |
| `CHROMBPNET_OPT_CACHE_TAR` | no | `$CHROMBPNET_OPT_WEIGHTS/cache/nv_compute_cache_<GPU>.tar` | the driver-cache tarball (TensorFlow routes) |
| `MODEL_OPT_TARGET_GPU` | no | `H100` / `H200` / `A100` | the `nvidia-smi` name the config targets; another card is noted on stderr, never refused |
| `MODEL_OPT` | no | this directory, from the config's own path | the kit root `run.sh` exports |
| `PYTHONDONTWRITEBYTECODE` | no | `1` | no `.pyc` in the tree |
| `CHROMBPNET_OPT` | no | unset (= `fast`) | the mode, for the environment route; a `--mode` that disagrees with it exits 2 |
| `CHROMBPNET_OPT_DET` | no | unset (= `0`) | `1` = `--mode off --det 1` |
| `CHROMBPNET_OPT_HOME` | no | unset | this directory, needed only when `chrombpnet_opt` is loaded from outside the tree |

The data files are upstream's tutorial inputs, not shipped here — into your data directory (the `--bind …:/data` source on route B):

```bash
curl -LO https://storage.googleapis.com/chrombpnet_data/input_files/hg38.genome.fa      # the fasta upstream's README links (~3 GB)
curl -LO https://storage.googleapis.com/chrombpnet_data/input_files/hg38.chrom.sizes
curl -LO https://www.encodeproject.org/files/ENCFF180XQC/@@download/ENCFF180XQC.bigWig   # GM12878 ATAC-seq observed signal (-bw), 1.2 GB
curl -L https://www.encodeproject.org/files/ENCFF333TAT/@@download/ENCFF333TAT.bed.gz | gunzip > regions.bed   # any BED/narrowPeak of regions works
samtools faidx hg38.genome.fa   # or: apptainer exec --bind /data:/data chrombpnet-kit.sif python -c "import pyfaidx; pyfaidx.Faidx('/data/hg38.genome.fa')"
```

`regions.bed` has no fixed count: ENCFF333TAT is the peak file upstream's tutorial downloads (K562 ATAC-seq overlap peaks); the GM12878 experiment's own
(ENCSR637XSC, e.g. `ENCFF078VWH` pseudoreplicated peaks, same URL form) or your own regions work the same way; ENCODE peak files are gzipped.

Each config line keeps a value already set in the environment (`${NAME:-default}`), except `PYTHONDONTWRITEBYTECODE`, which is always set
to `1`. The last three are read by the package, not set by the configs.
Kit-internal names found in the caller's environment are removed for the run and named as `[chrombpnet-opt] IGNORED names=<N1,N2,…> reason=<…>`. The `EXIT` line's
`stripped=` lists the `CHROMBPNET_OPT*` names the package read and then withheld from the job's own environment (the job gets them as arguments) — bookkeeping, not a warning.
