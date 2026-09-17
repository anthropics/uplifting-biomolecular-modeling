# Protenix v2 optimization kit — the tested stack as a container image.
# Builds Ubuntu 24.04 + the CUDA 13.0.1 development image with cuDNN 9.13 (public NVIDIA base; the development flavour because stock Protenix compiles
# its fast-LayerNorm CUDA extension with nvcc at first use — torch cpp_extension, `--mode off` and the kit's source fallback on a torch it ships no
# prebuilt for — and DeepSpeed builds its ops the same way; the runtime image has no nvcc or CUDA headers), gcc 13.3, git, ninja, kalign 3.3.5
# (template realignment under `--use_template true`; /usr/local/bin/kalign runs it with stdin closed — Ubuntu's kalign waits on a stdin that is open
# and not a terminal), Python 3.11.5 at /usr/local/bin/python (python-build-standalone 20230826), the pinned stack of
# protenix_v2/environment/requirements.lock (torch 2.13.0+cu130 with the CUDA 13.0 / cuDNN 9.20 / NCCL 2.29 runtime wheels, triton 3.7.1, cuEquivariance
# 0.11.1, deepspeed 0.17.5, numpy 2.4.6, rdkit 2025.9.3), stock protenix 2.0.0 from the wheel in protenix_v2/stock/, then this kit and its shared core,
# editable, under /kit (`run.sh install`). The configured card is H100 (configs/h100.env): stock's fast-LayerNorm extension is compiled at build
# (below), the kit's Triton and CUDA kernels compile at first use on the card they meet (`run.sh warm`). No weights are inside: `run.sh install --weights DIR` fetches the checkpoint and the six stock data
# caches into a directory you mount, and PROTENIX_ROOT_DIR names it at run time (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 13.0
# (580 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding protenix_v2/ and common/):
#   docker build -f protenix_v2/environment/Dockerfile -t protenix_v2-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/protenix_root:/weights -e PROTENIX_ROOT_DIR=/weights -v "$PWD/out":/kit/protenix_v2/out \
#     protenix_v2-kit:dev bash -c 'sed "s#__KIT__#$PWD/opt/forward/flashpairformer#" opt/forward/flashpairformer/public_inputs/p199.json > /tmp/p199.json \
#     && bash run.sh pred --config h100 --mode fast --input /tmp/p199.json --out_dir out/fast --model_name protenix-v2'
# To try the kit in your own environment instead: Python 3.11, the lock installed as STOCK.md 'Stack' shows (at minimum torch 2.13.0 with
# CUDA 13.0, triton 3.7.1, cuEquivariance 0.11.1 cu13, deepspeed, rdkit, biotite), the protenix wheel from protenix_v2/stock/, nvcc and a C compiler
# on PATH, then `run.sh install`.
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# The process environment of the tested stack: hash seed fixed, no debug info in what the C compiler builds (the source distributions below, the
# extensions torch and Triton compile at run time), nvcc found through CUDA_HOME by torch's extension builder, which compiles at run time for compute
# capability 9.0 — the tested card — plus PTX (newer cards JIT from it) with up to 32 parallel jobs, and the loader path every tested run had — the
# base image's driver directories plus the CUDA toolkit's lib64 (torch itself loads the CUDA libraries of the lock's nvidia-* wheels by path; the
# entry serves what resolves libraries by name).
# The weights root is never baked in: PROTENIX_ROOT_DIR is set at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    TORCH_CUDA_ARCH_LIST="9.0+PTX" \
    MAX_JOBS=32

# The C/C++ toolchain (gcc 13.3, make, ninja: torch's and Triton's run-time builds), git, the download tool of this build, and kalign with its
# stdin-closing entry ahead of /usr/bin on PATH.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget git build-essential ninja-build kalign && rm -rf /var/lib/apt/lists/* \
 && printf '#!/bin/sh\nexec /usr/bin/kalign "$@" </dev/null\n' > /usr/local/bin/kalign && chmod 755 /usr/local/bin/kalign

# Python 3.11.5: the python-build-standalone release this stack was assembled with, unpacked into /usr/local with its links made plain files (no system
# Python is used).
RUN wget -q -O /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -rL /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack: the lock's own pip first (it replaces the interpreter's bundled pip and installs everything after it), then every line of
# the lock with --no-deps (the lock is complete: PyPI's files, the three PyTorch-index files by URL and digest, the three source-only distributions
# built here), then stock from the bundled wheel (--no-deps: its declared torch / triton / cuequivariance pins do not hold on this stack, STOCK.md —
# `pip check` reports those lines and nothing else), then stock's fast-LayerNorm CUDA extension compiled the way stock compiles it at first import
# (protenix/model/layer_norm: torch's extension builder and nvcc, into the package directory; no GPU needed; that step chooses its GPU architectures
# itself from the ones this nvcc supports — 8.0, 8.6, 8.9, 9.0, 10.0 — whatever the environment names) so the image carries it built. BUILD_JOBS
# = the parallel compile jobs of this step (the three source builds and the extension's ninja build); 0 = every core of the builder.
ARG BUILD_JOBS=0
COPY protenix_v2/environment/requirements.lock /tmp/requirements.lock
COPY protenix_v2/stock/protenix-2.0.0-py3-none-any.whl /tmp/stock/protenix-2.0.0-py3-none-any.whl
RUN J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); export MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=4 \
 && echo "build jobs: $J (cores: $(nproc --all))" \
 && grep -E '^pip==' /tmp/requirements.lock | xargs python -m pip install --no-cache-dir \
 && grep -v -E '^(#|protenix==)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/protenix-2.0.0-py3-none-any.whl \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/requirements.lock \
 && python -c "import protenix.model.layer_norm"

# The kit: the shared core, then protenix_opt, both editable, and the install checked against the pins — the one install step, `run.sh install`
# (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY protenix_v2 /kit/protenix_v2
WORKDIR /kit/protenix_v2
RUN bash run.sh install
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY protenix_v2/environment/Dockerfile _jitcache/protenix_v2-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
