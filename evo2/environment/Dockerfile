# Evo 2 optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.8.1 toolkit (public NVIDIA devel base: nvcc 12.8.93 compiles Transformer Engine's torch extension below, at
# build time — the reason this is the devel flavour and not the runtime one; gcc 11.4 comes with it and Triton builds vortex's kernels with it at
# first use), Python 3.11.5 at /usr/local/bin/python, the pinned stack of ONE lock in evo2/environment/ selected by the build argument STACK:
# `img_full` (the default: NVIDIA H100 80GB, variants 7b and 40b, every mode — torch 2.7.1+cu128 with the CUDA 12.8 / cuDNN 9.7 / NCCL 2.26
# runtime wheels, triton 3.3.1, flash_attn 2.8.0.post2, Transformer Engine 2.3.0 whose torch extension is compiled here for sm_90) or `img_a100`
# (NVIDIA A100 80GB: the same list without Transformer Engine — upstream's light install: evo2_7b; the 40b needs FP8 and fails in
# upstream's own constructor there), stock evo2 0.6.0 and vtx 1.1.0 from the wheels in evo2/stock/, then this kit, editable, under
# /kit (`run.sh install`). No weights are inside: `run.sh install --weights DIR` fetches a model's checkpoint into a directory you mount, and
# EVO2_OPT_WEIGHTS names it at run time (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 12.8 (570 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding evo2/ and common/):
#   docker build -f evo2/environment/Dockerfile -t evo2-kit:dev .                              # H100
#   docker build -f evo2/environment/Dockerfile --build-arg STACK=img_a100 -t evo2-kit:dev .  # A100 80GB
# Run (README.md 'Run'; the weights directory holds evo2_7b.pt, README.md 'Setup'):
#   docker run --rm --gpus all -v /path/to/evo2_weights:/weights -e EVO2_OPT_WEIGHTS=/weights -v $PWD/out:/out evo2-kit:dev \
#     bash run.sh score --model_name evo2_7b --input /out/windows.fa --out_dir /out/exact
# To try the kit in your own environment instead: Python 3.11 and the CUDA 12.8 toolkit (nvcc) on PATH, the lock installed as STOCK.md 'Stack'
# describes (this file's own steps), the two stock wheels from evo2/stock/, then `run.sh install`.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
ARG STACK=img_full
# Parallel jobs for the one compile below (transformer-engine-torch, img_full only); 0 = every core of the build machine.
ARG BUILD_JOBS=0

# The download tools of this build (the C/C++ toolchain and make are part of the devel base).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*

# Python 3.11.5: the python-build-standalone release this stack was assembled with, unpacked into /usr/local (no system Python is used; the archive's
# few internal symlinks — bin/python3, bin/2to3 … — are laid down as the files they point to, the layout the tested stack has).
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -aL /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack of the selected lock, in the order its one source archive needs: every wheel at the lock's version (--no-deps: the lock is
# complete; the stock lines evo2 / vtx are left out so the files in evo2/stock/ are the ones installed), then — when the lock lists it —
# transformer-engine-torch, compiled against the torch just installed (--no-build-isolation) with nvcc 12.8 for sm_90, the one architecture the
# tested stack carries (NVTE_CUDA_ARCHS), the cuDNN 9.7 headers of the nvidia-cudnn-cu12 wheel (CUDNN_PATH: torch's extension builder reads it)
# and gcc / g++ as the C and C++ drivers (the interpreter records clang, which is not in this image), on BUILD_JOBS parallel jobs (MAX_JOBS is
# the extension builder's knob; about 2 minutes on 16 cores), then stock from the bundled wheels (--no-deps: vtx's declared einops==0.8.1 etc.
# are lines of the lock).
COPY evo2/environment/ /tmp/environment/
COPY evo2/stock/vtx-1.1.0-py3-none-any.whl evo2/stock/evo2-0.6.0-py3-none-any.whl /tmp/stock/
RUN test -f /tmp/environment/requirements-${STACK}.lock || { echo "no such stack: ${STACK} (evo2/environment/requirements-<stack>.lock: img_full | img_a100)" >&2; exit 2; } \
 && grep -v -E '^(#|evo2==|vtx==|transformer-engine-torch==)' /tmp/environment/requirements-${STACK}.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && { grep -E '^transformer-engine-torch==' /tmp/environment/requirements-${STACK}.lock > /tmp/te.txt || true; } \
 && if [ -s /tmp/te.txt ]; then J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); echo "transformer-engine-torch: compiling with $J parallel jobs ($(nproc --all) cores on this machine)"; \
         export MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVTE_BUILD_THREADS_PER_JOB=2; \
         CUDNN_PATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cudnn" CC=gcc CXX=g++ NVTE_CUDA_ARCHS=90 \
         python -m pip install --no-cache-dir --no-deps --no-build-isolation -r /tmp/te.txt || exit 1; fi \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/vtx-1.1.0-py3-none-any.whl /tmp/stock/evo2-0.6.0-py3-none-any.whl \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/te.txt /tmp/environment

# The kit: evo2_opt, editable, and the install checked against the pins of this stack — the one install step, `run.sh install` (README.md 'Setup').
COPY evo2 /kit/evo2
WORKDIR /kit/evo2
RUN bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY evo2/environment/Dockerfile _jitcache/evo2-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment of the tested stack (STOCK.md 'Stack'): hash seed fixed, no debug info in the launchers Triton compiles, and the
# library search path every run of the tested stack had — cuDNN 9.7 from its wheel first (Transformer Engine loads it by name), then the CUDA
# 12.8 toolkit's lib64. Thread-pool sizes are left to the user's machine; the weights root is never baked in: EVO2_OPT_WEIGHTS is set at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64
