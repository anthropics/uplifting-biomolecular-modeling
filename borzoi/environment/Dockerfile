# Borzoi optimization kit — the tested stack as a container image.
# Base: the official python:3.10.17-slim-bookworm image — Debian 12 with CPython 3.10.17 at /usr/local/bin/python, the interpreter and OS the
# stack is tested on; no CUDA base image is involved because TensorFlow 2.15.1 loads its whole CUDA 12.2 / cuDNN 8.9 user-space from the
# nvidia-*-cu12 wheels of borzoi/environment/requirements.lock (the GPU driver comes from the host). On top: bedtools and a compiler toolchain from
# Debian (pybedtools calls bedtools; build-essential, gfortran and the -dev headers are the packages the tested image carries), the pinned stack
# of requirements.lock, the pinned upstream — calico/baskerville @ 544073b at /opt/baskerville and calico/borzoi @ 5c93582 at /opt/borzoi, unpacked
# from the archives in borzoi/stock/ and installed editable (their versions given to setuptools_scm as STOCK.md 'Install' shows: an archive has
# no .git; pip's isolated builds — theirs and the three source-only pins of the lock, intervaltree, pybedtools, sorted_nearest — run with
# setuptools 84.0.0, the release the tested image's install metadata is generated with) — then this kit, editable, under /kit
# (`run.sh install`, which ends with the pin check). BORZOI_DIR, PATH and PYTHONPATH name the two checkouts' script directories the way
# upstream's env_vars.sh does and the tested image sets them: upstream's training and evaluation scripts import their sibling modules (slurm,
# util, pygene) from /opt/borzoi/src/scripts through PYTHONPATH; this kit's own verbs do not depend on it. The stock entry is
# /opt/borzoi/src/scripts/borzoi_sad.py. One image serves H100 and A100: nothing is compiled for a GPU at build time
# (TensorFlow 2.15.1 ships no sm_90 kernels and JIT-compiles them from PTX at process start on an H100, in every mode alike). No weights and no
# genome are inside: `run.sh install --weights DIR` fetches the fold-0 weights into a directory you mount, the genome / targets / params are
# the command's own arguments (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 12.2 (535 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding borzoi/ and common/):
#   docker build -f borzoi/environment/Dockerfile -t borzoi-kit:dev .
# Run (README.md 'Run' has the full command; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/borzoi_data:/data -v "$PWD/out":/kit/borzoi/out borzoi-kit:dev \
#     bash run.sh sad --config h100 --mode exact -f /data/hg38.ml.fa -o out/exact --rc --stats SAD,logSAD -t /opt/borzoi/examples/targets_human.txt -u \
#     /opt/borzoi/examples/params.json /data/f0/model0_best.h5 tests/inputs/example_snvs.vcf
# To try the kit in your own environment instead: Python 3.10, the lock installed as README.md 'Setup' route C shows (at minimum tensorflow
# 2.15.1 with its CUDA 12.2 wheels, numpy 1.24.x, h5py, pandas, pysam, scipy), baskerville @ 544073b and borzoi @ 5c93582 from borzoi/stock/,
# BORZOI_DIR naming the borzoi checkout, then `bash borzoi/run.sh install`.
FROM python:3.10.17-slim-bookworm

# ---- the process environment of the tested stack: hash seed, compile flags (also in force for the source builds below), TensorFlow's log level
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    TF_CPP_MIN_LOG_LEVEL=1

# ---- system packages: bedtools (pybedtools' binary) and the toolchain / headers the tested image carries
RUN apt-get update && apt-get install -y bedtools build-essential gfortran git libbz2-dev libcurl4-openssl-dev liblzma-dev libssl-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# ---- the pinned stack (borzoi/environment/requirements.lock): the lock's pip first (it replaces the base image's bundled pip and runs every
#      later install), then the lock minus its two upstream lines, which the next step installs from borzoi/stock/. pip's isolated builds of
#      the source-only pins run with setuptools 84.0.0 (build-constraints.txt, read through PIP_CONSTRAINT); the lock's own setuptools line —
#      the run-time setuptools, another version — is installed apart from the pass that carries that constraint. The step prints the
#      build machine's core count; nothing in it runs in parallel
COPY borzoi/environment/requirements.lock /tmp/requirements.lock
RUN echo "build machine: $(nproc --all) cores" \
 && grep -E '^pip==' /tmp/requirements.lock | xargs python -m pip install --no-cache-dir --no-deps \
 && echo 'setuptools==84.0.0' > /tmp/build-constraints.txt \
 && grep -v -E '^(#|-e |setuptools==)' /tmp/requirements.lock > /tmp/stack.txt \
 && PIP_CONSTRAINT=/tmp/build-constraints.txt python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && grep -E '^setuptools==' /tmp/requirements.lock | xargs python -m pip install --no-cache-dir --no-deps \
 && rm -f /tmp/stack.txt /tmp/requirements.lock

# ---- the pinned upstream: calico/baskerville @ 544073b and calico/borzoi @ 5c93582 from the archives in borzoi/stock/, editable at /opt;
#      every stock script executable (upstream runs them by name from PATH)
COPY borzoi/stock/baskerville-544073b.tar.gz borzoi/stock/borzoi-5c93582.tar.gz /tmp/stock/
RUN mkdir -p /opt/baskerville /opt/borzoi \
 && tar -xzf /tmp/stock/baskerville-544073b.tar.gz -C /opt/baskerville --strip-components=1 \
 && tar -xzf /tmp/stock/borzoi-5c93582.tar.gz -C /opt/borzoi --strip-components=1 \
 && chmod +x /opt/borzoi/src/scripts/*.py /opt/baskerville/src/baskerville/scripts/*.py \
 && PIP_CONSTRAINT=/tmp/build-constraints.txt \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BASKERVILLE=1.0.0 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BORZOI=1.0.1.dev7+g5c9358222 \
    python -m pip install --no-cache-dir --no-deps -e /opt/baskerville -e /opt/borzoi \
 && rm -rf /tmp/stock /tmp/build-constraints.txt /root/.cache/pip
ENV BORZOI_DIR=/opt/borzoi \
    BASKERVILLE_DIR=/opt/baskerville \
    PATH=/opt/borzoi/src/scripts:/opt/baskerville/src/baskerville/scripts:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/opt/borzoi/src/scripts:/opt/baskerville/src/baskerville/scripts

# ---- this kit, editable, then the pin check (run.sh install; the kit needs no other package of the bundle)
COPY borzoi /kit/borzoi
WORKDIR /kit/borzoi
RUN bash run.sh install
