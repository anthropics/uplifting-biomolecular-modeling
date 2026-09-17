# OpenDDE optimization kit — the tested stack as a container image.
# Builds Ubuntu 24.04 + the CUDA 12.6.3 *development* base (public NVIDIA image; the development flavour because the CUDA 12.6 compiler and headers are
# needed AT RUN TIME — stock's fused LayerNorm, LAYERNORM_TYPE=fast_layernorm, the precision base of every mode, builds its torch extension with
# ninja + the CUDA compiler at first use; Triton builds the kit's kernels with the system gcc 13.2; cuDNN, cuBLAS and NCCL load from the lock's nvidia-*
# wheels, not from the base), hmmer (Ubuntu 24.04's package) and kalign 3.3.5 (pinned) from apt
# (kalign aligns template hits under `--use_template true`), Python 3.11.5 at /usr/local/bin/python, the pinned stack of
# opendde/environment/requirements.lock (torch 2.7.1+cu126 with the CUDA 12.6 / cuDNN 9.5 / NCCL 2.26 runtime wheels, triton 3.3.1, cuequivariance
# 0.10.0, numpy 2.4.1, rdkit 2025.9.3), stock opendde 1.1.1 from the wheel in opendde/stock/, then this kit and its shared core, editable, under
# /kit (`run.sh install`). One image serves every supported GPU: nothing is compiled for a GPU at build time. No weights are inside:
# `run.sh install --weights DIR` fetches the checkpoint and the four `common/` files into a directory you mount, and OPENDDE_ROOT_DIR names it at
# run time (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 12.6 (560 or newer) and the NVIDIA Container Toolkit.
# Build, from the bundle's top directory (the one holding opendde/ and common/):
#   docker build -f opendde/environment/Dockerfile -t opendde-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/opendde_root:/weights -e OPENDDE_ROOT_DIR=/weights -v "$PWD/out":/kit/opendde/out opendde-kit:dev \
#     bash run.sh warm --config h100 -o out/warm
# To try the kit in your own environment instead: Python 3.11, the CUDA 12.6 compiler + gcc on PATH, the lock installed as STOCK.md 'Stack' describes
# (at minimum torch 2.7.1 with CUDA 12.6, triton 3.3.1, cuequivariance 0.10.0, numpy 2.4.x, ninja), the opendde wheel from opendde/stock/,
# then `run.sh install`.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# The C/C++ toolchain the CUDA compiler and Triton need at run time, upstream's two search binaries (hmmer; kalign at the version this stack pins), the
# shared libraries rdkit's drawing modules link against, and the download tools of this build.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git hmmer kalign=1:3.3.5-1 libglib2.0-0 libgl1 libxrender1 pkg-config tzdata \
 && rm -rf /var/lib/apt/lists/*

# Python 3.11.5: the python-build-standalone release this stack was assembled with, unpacked into /usr/local with its links written out as files, the
# layout the tested stack has (no system Python is used).
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -aL /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete; torch's line is the URL of its CUDA 12.6 file, with
# its sha256), then stock from the bundled wheel (--no-deps: its dependencies are the lock's lines).
COPY opendde/environment/requirements.lock /tmp/requirements.lock
COPY opendde/stock/opendde-1.1.1-py3-none-any.whl /tmp/stock/opendde-1.1.1-py3-none-any.whl
RUN echo "installing the pinned stack (build host: $(nproc --all) cores; nothing is compiled)" \
 && grep -v -E '^(#|opendde==)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/opendde-1.1.1-py3-none-any.whl \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/requirements.lock

# The kit: the shared core, then opendde_opt, both editable, and the install checked against the pin — the one install step, `run.sh install` (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY opendde /kit/opendde
WORKDIR /kit/opendde
RUN bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY opendde/environment/Dockerfile _jitcache/opendde-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment of the tested stack: hash seed fixed, unbuffered output, no bytecode written into the tree, no debug info in the
# launchers built at run time (the loader path, LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64, is the base image's own). The
# weights root is never baked in: OPENDDE_ROOT_DIR is set at run time.
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CFLAGS=-g0
