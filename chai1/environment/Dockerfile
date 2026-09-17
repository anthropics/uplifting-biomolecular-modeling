# Chai-1 optimization kit — the pinned stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 13.0.1 runtime (public NVIDIA base) + gcc 11.4 (build-essential: Triton builds the kit's kernels and their
# launchers at first use) + the CUDA 13.0 development headers and static runtime (a step function compiled ahead of time links against them),
# Python 3.11.5 at /usr/local/bin/python, the pinned stack of chai1/environment/requirements.lock (torch 2.13.0+cu130 with
# the CUDA 13.0 / cuDNN 9.20 / NCCL 2.29 runtime wheels, triton 3.7.1, numpy 1.26.4, rdkit 2024.9.6, gemmi 0.6.7), stock chai_lab 0.6.1 from the wheel
# in chai1/stock/, then this kit and its shared core, editable, under /kit (`run.sh install`). One image serves H100 and A100: nothing is compiled
# for a GPU at build time. No weights are inside: `run.sh install --weights DIR` fetches the eight stock files into a directory you mount, and
# CHAI_DOWNLOADS_DIR names it at run time (README.md, Setup). Host: an NVIDIA driver that runs CUDA 13.0 (580 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding chai1/ and common/):
#   docker build -f chai1/environment/Dockerfile -t chai1-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/chai_downloads:/weights -e CHAI_DOWNLOADS_DIR=/weights -v "$PWD/out":/kit/chai1/out chai1-kit:dev \
#     bash run.sh pred --config h100 --mode fast --input opt/forward/fast_inference/tests/public_inputs/1BRS_1to1.fasta --out_dir out/fast --seeds 0
# To try the kit in your own environment instead: Python 3.11, chai1/environment/requirements.lock installed (at minimum torch 2.13.0
# with CUDA 13.0, triton 3.7.1, numpy 1.26.x, rdkit, gemmi), the chai_lab wheel from chai1/stock/, a C compiler on PATH, then `run.sh install`.
FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

# The C toolchain Triton needs at run time (gcc, libc headers) and the download tools of this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential && rm -rf /var/lib/apt/lists/*

# Python 3.11.5: the python-build-standalone release this stack was assembled with, unpacked into /usr/local (no system Python is used).
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete), then stock from the bundled wheel (--no-deps: its
# declared torch<2.7 does not hold on this stack, STOCK.md — `pip check` reports that one line and nothing else).
COPY chai1/environment/requirements.lock /tmp/requirements.lock
COPY chai1/stock/chai_lab-0.6.1-py3-none-any.whl /tmp/stock/chai_lab-0.6.1-py3-none-any.whl
RUN grep -v -E '^(#|chai_lab==)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/chai_lab-0.6.1-py3-none-any.whl \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/requirements.lock

# The CUDA 13.0 development files beside the runtime (NVIDIA's apt repository, configured in the base image): cuda.h / cuda_runtime.h and the
# compiler-side headers they include, the static runtime and the driver stub — what torch's Inductor needs to compile and link, with the gcc
# above, the C++ launcher of a step function compiled ahead of time (torch.export + AOTInductor; the GPU kernels themselves come from Triton, no
# CUDA compiler is installed). CUDA_HOME names the toolkit root for torch.
RUN apt-get update && apt-get install -y --no-install-recommends cuda-cudart-dev-13-0 cuda-crt-13-0 && rm -rf /var/lib/apt/lists/* \
 && ls /usr/local/cuda-13.0/include/cuda.h /usr/local/cuda-13.0/include/cuda_runtime.h /usr/local/cuda-13.0/include/crt/host_config.h /usr/local/cuda-13.0/lib64/libcudart_static.a /usr/local/cuda-13.0/lib64/stubs/libcuda.so
ENV CUDA_HOME=/usr/local/cuda-13.0

# The kit: the shared core, then chai1_opt, both editable, and the install checked against the pin — the one install step, `run.sh install` (README.md, Setup).
COPY common/opt_core /kit/common/opt_core
COPY chai1 /kit/chai1
WORKDIR /kit/chai1
RUN bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY chai1/environment/Dockerfile _jitcache/chai1-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment of the pinned stack (STOCK.md): hash seed fixed, no debug info in the launchers Triton compiles.
# Thread-pool sizes are left to the user's machine; the weights root is never baked in: CHAI_DOWNLOADS_DIR is set at run time.
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    CFLAGS=-g0
