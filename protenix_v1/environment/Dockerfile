# Protenix v1 optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.9.1 development toolkit (public NVIDIA base, kept at development weight because this stack compiles CUDA
# with nvcc: stock protenix's fast LayerNorm extension is built into the image below; the kit's stream-correct build of it ships prebuilt for
# this stack and is rebuilt (under TORCH_EXTENSIONS_DIR, once) only if its loader's self-test rejects it — nvcc 12.9, the CUDA headers and
# gcc 11.4 are needed at build time and for that fallback),
# Python 3.11.5 at /usr/local/bin/python (python-build-standalone 20230826), the pinned stack of protenix_v1/environment/requirements.lock
# (torch 2.13.0+cu130 with the CUDA 13.0 / cuDNN 9.20 / NCCL 2.29 runtime wheels, triton 3.7.1, cuequivariance 0.11.1, deepspeed 0.17.5,
# numpy 2.4.1), stock protenix 1.1.0 from the wheel in protenix_v1/stock/ with its fast LayerNorm extension compiled in place
# (site-packages/protenix/model/layer_norm/fast_layer_norm_cuda_v2.so, sm_70…sm_100, by upstream's own first import), kalign 3.3.1 (apt)
# behind /usr/local/bin/kalign, which closes its stdin (the `--use_template true` realigner; kalign blocks on an open stdin), then this kit
# and its shared core, editable, under /kit (`run.sh install`). Nothing of the kit is compiled for a GPU at build time: its Triton kernels and
# its LayerNorm build compile at first use for the GPU present (H100, H200, B200, B300, A100: stock/PINS.json "supported_gpus"). No weights are
# inside: `run.sh install --weights DIR` fetches the checkpoint and the six data files into a directory you mount, and PROTENIX_ROOT_DIR
# names it at run time (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 13.0 (580 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding protenix_v1/ and common/):
#   docker build -f protenix_v1/environment/Dockerfile -t protenix_v1-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/protenix_root:/weights -e PROTENIX_ROOT_DIR=/weights -v "$PWD/out":/kit/protenix_v1/out protenix_v1-kit:dev \
#     bash run.sh warm --config h100 --mode fast --out_dir out/warm
# To try the kit in your own environment instead: Python 3.11, the lock installed as STOCK.md 'Stack' shows (at minimum torch 2.13.0
# with CUDA 13.0, triton 3.7.1, cuequivariance-torch + cuequivariance-ops-torch-cu12 0.11.1, deepspeed 0.17.5, numpy, rdkit, biotite), the
# protenix wheel from protenix_v1/stock/, nvcc and a C compiler on PATH, then `run.sh install`.
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# The process environment of the tested stack (set first: the source builds below run under it): hash seed fixed, no debug info in what the C
# compiler builds, stock's fast LayerNorm path selected (the upstream default, stated), the toolkit root. No TORCH_CUDA_ARCH_LIST: both fast
# LayerNorm builds (stock's below, the kit's at first use) take their architectures from upstream's torch_ext_compile, which asks the nvcc
# present; nothing else here compiles for a GPU. The weights root is never baked in: PROTENIX_ROOT_DIR is set at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    LAYERNORM_TYPE=fast_layernorm \
    CUDA_HOME=/usr/local/cuda

# The tools the tested image carries beside the base's build-essential (git, wget, clang, ninja; with their recommends, as it installed them)
# and kalign, plus the stdin-closing wrapper the stack calls it through.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git wget build-essential clang ninja-build kalign && rm -rf /var/lib/apt/lists/* \
 && printf '#!/bin/sh\nexec /usr/bin/kalign "$@" </dev/null\n' > /usr/local/bin/kalign && chmod 755 /usr/local/bin/kalign

# Python 3.11.5: the python-build-standalone release this stack was assembled with, unpacked into /usr/local (no system Python is used).
RUN wget -q -O /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete; the two +cu129 lines come from their pinned URLs,
# everything else from PyPI), in two passes read from the lock by name: everything but the CUDA 13 line, then that line — torch 2.13.0,
# triton 3.7.1 and the nvidia-*-cu13 wheels — so the CUDA 13 copies own the directories they share with the cu12 line (nvidia/cudnn,
# nvidia/nccl, nvidia/cusparselt: the ones torch 2.13.0 loads), installed as `python3.11 -m pip`, the interpreter name that line's console
# scripts (torchrun, proton) name; nvidia/ stays a namespace package as in the tested tree (the one regular __init__.py a cu12 wheel ships is
# removed). Then stock from the bundled wheel (--no-deps: its declared torch==2.7.1 / triton==3.3.1 / cuequivariance 0.8.0 pins do not hold on
# this stack, STOCK.md) and its fast LayerNorm extension compiled in place by its own first import (upstream's torch_ext_compile: nvcc for
# sm_70…sm_100, about five minutes, no GPU needed).
# BUILD_JOBS: parallel jobs for the source builds of this step (the lock's sdists, the LayerNorm extension); 0 = every core of the builder.
ARG BUILD_JOBS=0
COPY protenix_v1/environment/requirements.lock /tmp/requirements.lock
COPY protenix_v1/stock/protenix-1.1.0-py3-none-any.whl /tmp/stock/protenix-1.1.0-py3-none-any.whl
RUN J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); echo "build cores: $(nproc --all); build jobs: $J" \
 && export MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=4 \
 && grep -v -E '^(#|protenix==|torch==|triton==|nvidia-[a-z-]+-cu13==)' /tmp/requirements.lock > /tmp/pass1.txt \
 && grep -E '^(torch==|triton==|nvidia-[a-z-]+-cu13==)' /tmp/requirements.lock > /tmp/pass2.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/pass1.txt \
 && python3.11 -m pip install --no-cache-dir --no-deps -r /tmp/pass2.txt \
 && rm -f /usr/local/lib/python3.11/site-packages/nvidia/__init__.py \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/protenix-1.1.0-py3-none-any.whl \
 && python -c "import protenix.model.layer_norm.layer_norm as m; print('fast LayerNorm extension:', m.fast_layer_norm_cuda_v2.__file__)" \
 && rm -rf /tmp/stock /tmp/pass1.txt /tmp/pass2.txt /tmp/requirements.lock

# The kit: the shared core, then protenix_v1_opt, both editable (+ the start-up hook protenix_v1_opt_autoload.pth), and the install checked
# against the pin — the one install step, `run.sh install` (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY protenix_v1 /kit/protenix_v1
WORKDIR /kit/protenix_v1
RUN bash run.sh install
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY protenix_v1/environment/Dockerfile _jitcache/protenix_v1-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
