# BoltzGen optimization kit — the tested stack as a container image.
# Base: NVIDIA's public CUDA 13.0.1 base image on Ubuntu 22.04 — the tested stack's own base, and the lightest CUDA flavour, because every CUDA
# library this stack loads (the CUDA 13.0 runtime, cuBLAS 13.1, cuDNN 9.20, NCCL 2.29, NVRTC) is a pinned nvidia-* wheel of the lock reached through
# torch's own library search path, and no CUDA code is compiled here at build or run time. On it: gcc 11.4 (build-essential: Triton builds the
# launchers of the cuEquivariance kernels at first use), Python 3.11.5 at
# /usr/local/bin/python (python-build-standalone 20230826), the pinned stack of boltzgen/environment/requirements.lock (torch 2.13.0+cu130 with the
# CUDA 13.0 / cuDNN 9.20 / NCCL 2.29 runtime wheels, triton 3.7.1, cuequivariance 0.11.1 with its CUDA-13 kernel pair, pytorch-lightning 2.6.5, numpy
# 2.0.2, numba 0.61.0), stock boltzgen 0.3.2 from the wheel in boltzgen/stock/, then this kit and its shared core, editable, under /kit
# (`run.sh install`). One image serves H100 and A100: nothing is compiled for a GPU at build time. No weights are inside: `run.sh install --weights DIR`
# fetches the six stock files into a directory you mount, and BOLTZGEN_CACHE names it at run time (README.md 'Setup'). Host: an NVIDIA driver that
# runs CUDA 13.0 (580 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding boltzgen/ and common/):
#   docker build -f boltzgen/environment/Dockerfile -t boltzgen-kit:dev .
# Run (README.md 'Run' has one command per mode; the run directory lands under the mounted runs/):
#   docker run --rm --gpus all -v /path/to/boltzgen-cache:/weights -e BOLTZGEN_CACHE=/weights -v "$PWD/runs":/kit/boltzgen/runs boltzgen-kit:dev \
#     bash run.sh design --config h100 --mode fast /kit/boltzgen/opt/forward/fast_inference/tests/specs/pdl1_ref.yaml --output /kit/boltzgen/runs/fast --num_designs 16 --seed 0
# To try the kit in your own environment instead: Python 3.11, the lock installed as README.md 'Setup' route C shows (at minimum torch 2.13.0 with
# CUDA 13.0, triton 3.7.1, cuequivariance 0.11.1 with the cuequivariance-ops-cu13 / cuequivariance-ops-torch-cu13 pair, pytorch-lightning 2.6.5,
# numpy 2.0.2, numba 0.61.0), the boltzgen wheel from boltzgen/stock/, a C compiler on PATH, then `run.sh install`.
FROM nvidia/cuda:13.0.1-base-ubuntu22.04

# The C toolchain Triton needs at run time (gcc, libc headers) and the download tools of this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential && rm -rf /var/lib/apt/lists/*

# Python 3.11.5: the python-build-standalone release the stack was assembled on, unpacked at /usr/local (its site-packages is the stack's).
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack, from the lock alone (--no-deps: the lock is the whole list; torch's CUDA build and triton are the two lines the lock names by URL and
# sha256): first the lock's own pip, setuptools and wheel lines by the interpreter's bundled pip, then the list minus the stock line, then stock boltzgen
# from the wheel in boltzgen/stock/.
COPY boltzgen/environment/requirements.lock /tmp/requirements.lock
COPY boltzgen/stock/boltzgen-0.3.2-py3-none-any.whl /tmp/stock/boltzgen-0.3.2-py3-none-any.whl
RUN echo "build cores: $(nproc --all)" && grep -E '^(pip|setuptools|wheel)==' /tmp/requirements.lock | xargs python -m pip install --no-cache-dir --no-deps \
 && grep -v -E '^(#|boltzgen==)' /tmp/requirements.lock > /tmp/stack.txt && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/boltzgen-0.3.2-py3-none-any.whl \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/requirements.lock

# The kit and its shared core, editable, from the bundle: `run.sh install` (pip install -e ../common/opt_core -e opt, then the pin check).
COPY common/opt_core /kit/common/opt_core
COPY boltzgen /kit/boltzgen
WORKDIR /kit/boltzgen
RUN bash run.sh install
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY boltzgen/environment/Dockerfile _jitcache/boltzgen-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment the stack was tested under: hash randomisation off and Triton's launcher builds without debug info. The loader path is the
# base image's own (LD_LIBRARY_PATH names its driver and CUDA library directories; torch resolves its CUDA libraries from the nvidia-* wheels by its own
# search path). Thread-pool sizes (OMP_NUM_THREADS and kin) are left to the libraries' defaults; BOLTZGEN_CACHE and the JIT-cache root are yours to set
# at run time (configs/h100.env).
ENV PYTHONHASHSEED=0 CFLAGS=-g0
