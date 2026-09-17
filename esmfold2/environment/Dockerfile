# ESMFold2 optimization kit — the tested stack as a container image.
# Base: the official python:3.12.10-slim-bookworm image (Debian 12, CPython 3.12.10 at /usr/local built --enable-optimizations --enable-shared
# --with-lto, gcc 12.2 from build-essential) because that is the image the tested stack was assembled on; no CUDA toolkit is inside the
# finished image and none is needed at run time: every CUDA 13.0 / cuDNN 9.20 / NCCL 2.29 library torch, flash-attn, transformer_engine and
# xformers load comes from the nvidia-* wheels of esmfold2/environment/requirements.lock (Triton builds the kit's kernels and their launchers
# at first use, with the gcc installed here). On top: the pinned stack of the lock (torch 2.13.0+cu130, triton 3.7.1, numpy 2.3.5, stock esm
# 3.3.0 and the transformers 4.57.6 fork from git at their pinned commits), the three CUDA extensions the lock names but no index carries —
# flash_attn 2.8.3.post1, transformer_engine 2.15.0+42b8400, xformers 0.0.35+03b91d7.d20260904 — compiled for the GPUs of ONE stack selected by
# the build argument STACK: `img_ef2_fa` (the default: compute capability 9.0 code, NVIDIA H100 / H200) or `img_esmfold2_a100` (the same three
# packages at the same versions compiled for compute capability 8.0 and 9.0: NVIDIA A100 80GB as well, `run.sh … --config a100` there; every other
# line of the image identical) — then this kit and its shared core, editable, under /kit (`run.sh install`). The three extensions come one of two
# ways (build argument WHEELS_FROM):
#   WHEELS_FROM=build (the default) compiles them here from their public sources exactly as the tested wheels were made — flash-attn from the
#     PyPI sdist, TransformerEngine v2.15 and xformers v0.0.35 from their git tags — with the CUDA 13.0 compiler and development libraries from
#     NVIDIA's Debian 12 repository installed for that step only and removed in the same layer, using every core (BUILD_JOBS=0 = all;
#     `--build-arg BUILD_JOBS=N` caps it); most of the time is flash-attn.
#     A source build is the same code at the same versions but not byte-identical to the tested wheels, so no digest is checked on this route.
#   WHEELS_FROM=prebuilt installs the tested wheel files themselves from esmfold2/stock/wheels/ (place the three wheel files there first),
#     each checked by name and sha256 against the selected stack's wheel list in
#     stock/PINS.json ("image".wheels for img_ef2_fa, "image_variants".a100.wheels for img_esmfold2_a100) — minutes, no compiler.
# Either way the wheel files stay in the image at /kit/esmfold2/stock/wheels/ (a copy to install into another environment at the pin).
# No weights are inside: `run.sh install --weights DIR` fetches the pinned Hugging Face snapshots (biohub/ESMFold2, ESMFold2-Fast, ESMC-6B;
# ~27 GB) into a directory you mount, and HF_HOME names it at run time (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 13.0
# (580 or newer) and the NVIDIA Container Toolkit; the compiled extensions carry code for the selected stack's GPUs only (a card outside it
# imports them and fails at the first kernel launch: 'no kernel image is available').
# Build, from the bundle's top directory (the one holding esmfold2/ and common/):
#   docker build -f esmfold2/environment/Dockerfile -t esmfold2-kit:dev .                                        # H100 / H200: compiles the three extensions
#   docker build -f esmfold2/environment/Dockerfile --build-arg STACK=img_esmfold2_a100 -t esmfold2-kit:dev .     # A100 80GB as well (compute capability 8.0 and 9.0)
#   docker build -f esmfold2/environment/Dockerfile --build-arg WHEELS_FROM=prebuilt -t esmfold2-kit:dev .        # installs esmfold2/stock/wheels/*.whl
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/hf_home:/weights -e HF_HOME=/weights -v "$PWD/out":/kit/esmfold2/out esmfold2-kit:dev \
#     bash run.sh pred --config h100 --variant fast --mode fast --input opt/forward/fast_inference/tests/w4_public_slice.json --out_dir out/fast --seeds 0
# To try the kit in your own environment instead: Python 3.12, the lock installed as the header of requirements.lock describes (at minimum torch 2.13.0
# with CUDA 13.0, triton 3.7.1, numpy 2.3.x, flash-attn 2.8.3, transformer_engine 2.15, esm and the transformers fork at the pinned commits),
# a C compiler on PATH, then `run.sh install`.
FROM python:3.12.10-slim-bookworm
# The stack this image serves: img_ef2_fa (compute capability 9.0: H100 / H200) | img_esmfold2_a100 (8.0 and 9.0: A100 80GB as well). It decides the
# CUDA architectures the three extensions are compiled for and the stock/PINS.json wheel list the prebuilt route checks — nothing else.
ARG STACK=img_ef2_fa
ARG WHEELS_FROM=build
ARG BUILD_JOBS=0

# The image ENV of the tested stack that does not depend on the machine's shape: hash seed, C flags for the compiles above, transformer_engine's
# framework selector, xformers' flash-attn version waiver, and the Hugging Face transfer backend `install --weights` uses. Weights and cache
# locations are never baked in (HF_HOME is set at run time).
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    NVTE_FRAMEWORK=pytorch \
    XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# The C and Fortran toolchain layer of the tested image (gcc/g++ 12.2: Triton's launcher compiles at run time, dockq's C extension at install)
# and git (esm, transformers and dockq install from git at the commits the lock pins; the source route clones TransformerEngine and xformers).
RUN apt-get update && apt-get install -y gcc gfortran build-essential git && rm -rf /var/lib/apt/lists/* \
 && echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# The pinned stack: first the lock's own pip (the tested stack's pip is newer than the interpreter's bundled one and installs everything after it),
# then every distribution at the lock's version (--no-deps: the lock is complete), less the three lines naming the CUDA extensions, which the
# next step builds or takes from esmfold2/stock/wheels/.
COPY esmfold2/environment/requirements.lock /tmp/requirements.lock
RUN grep -E '^pip==' /tmp/requirements.lock | xargs python -m pip install --no-cache-dir --no-deps \
 && grep -v -E '^(#|(flash_attn|transformer_engine|xformers)==)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt && rm -f /tmp/stack.txt /tmp/requirements.lock

# The kit tree and its shared core under /kit, then the three CUDA extensions into /kit/esmfold2/stock/wheels/ and from there into the stack
# (--no-index --no-deps: nothing else is fetched), then `pip check` over the whole stack.
# WHEELS_FROM=build: the CUDA 13.0 compiler + development libraries (NVIDIA's Debian 12 repository; CUDA_HOME=/usr/local/cuda-13.0) and the
#   builds' Python tools (cmake, ninja, pybind11, nvidia-cudnn-frontend — TransformerEngine's build requirements) are installed, the three wheels
#   are built for the selected stack's compute capabilities (TORCH_CUDA_ARCH_LIST set from STACK in the one `case` below; TransformerEngine's
#   NVTE_CUDA_ARCHS and flash-attn's FLASH_ATTN_CUDA_ARCHS are the same list without the dots) against the torch of the lock with the settings the
#   tested wheels were built with — xformers v0.0.35 (git tag, submodules;
#   BUILD_VERSION pins its version string to the lock's), TransformerEngine v2.15 (git tag = commit 42b8400, submodules; one monolithic wheel,
#   the NCCL expert-parallel extension off as tested; cuDNN and NCCL headers from the lock's nvidia-cudnn / nvidia-nccl wheels), flash-attn 2.8.3.post1 (PyPI sdist,
#   sha256-checked, forced local build) — and then every package and file that step added is removed again: the finished layer holds the
#   wheels and their installation only. Jobs: BUILD_JOBS, else every core (`nproc --all`); flash-attn's nvcc jobs are additionally capped at
#   one per 9 GB of available memory; NVCC_THREADS=2. The step prints the core count and its wall time.
# WHEELS_FROM=prebuilt: esmfold2/stock/wheels/ must hold the selected stack's three wheel files; each is checked by name and sha256 against
#   that stack's list in stock/PINS.json ("image".wheels | "image_variants".a100.wheels) before anything is installed.
# An unknown STACK is refused by name before either route runs.
COPY common/opt_core /kit/common/opt_core
COPY esmfold2 /kit/esmfold2
WORKDIR /kit/esmfold2
RUN set -eu; W=/kit/esmfold2/stock/wheels; mkdir -p "$W"; \
 case "$STACK" in \
   img_ef2_fa)        TORCH_CUDA_ARCH_LIST="9.0";     PINS_WHEELS="image wheels" ;; \
   img_esmfold2_a100) TORCH_CUDA_ARCH_LIST="8.0;9.0"; PINS_WHEELS="image_variants a100 wheels" ;; \
   *) echo "esmfold2: no such stack: STACK=$STACK (img_ef2_fa: compute capability 9.0, H100 / H200 | img_esmfold2_a100: 8.0 and 9.0, A100 80GB as well)" >&2; exit 2 ;; \
 esac; export TORCH_CUDA_ARCH_LIST; SM_ARCHS=$(printf %s "$TORCH_CUDA_ARCH_LIST" | tr -d .); \
 if [ "$WHEELS_FROM" = build ]; then \
   T0=$(date +%s); J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); \
   MEM_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); JFA=$(( MEM_GB / 9 )); [ "$JFA" -ge 1 ] || JFA=1; [ "$JFA" -le "$J" ] || JFA=$J; \
   echo "esmfold2 wheels (STACK=$STACK): building xformers, transformer_engine, flash_attn for compute capability $TORCH_CUDA_ARCH_LIST with $J jobs (nproc --all = $(nproc --all), nproc = $(nproc), ${MEM_GB} GB available; flash-attn capped at $JFA jobs)"; \
   dpkg-query -W -f '${Package}\n' | sort > /tmp/dpkg.before; \
   apt-get update; apt-get install -y --no-install-recommends --no-upgrade ca-certificates curl; \
   curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb; \
   echo "e7f219eab6fe4819cdb5c15b98233dc3420302d9c00883219cd3d896857cf48d  /tmp/cuda-keyring.deb" | sha256sum -c -; \
   dpkg -i /tmp/cuda-keyring.deb; apt-get update; \
   apt-get install -y --no-install-recommends cuda-compiler-13-0 cuda-libraries-dev-13-0 cuda-cudart-dev-13-0 cuda-nvtx-13-0 cuda-profiler-api-13-0 cuda-nvml-dev-13-0; \
   export CUDA_HOME=/usr/local/cuda-13.0; export PATH="$CUDA_HOME/bin:$PATH" MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=2; \
   nvcc --version | tail -1; \
   python -m pip install --no-cache-dir cmake==4.4.3 ninja 'pybind11[global]==3.1.0' nvidia-cudnn-frontend==1.28.0; \
   mkdir -p /tmp/src; \
   git clone --quiet --depth 1 --branch v0.0.35 --recurse-submodules --shallow-submodules https://github.com/facebookresearch/xformers.git /tmp/src/xformers; \
   test "$(git -C /tmp/src/xformers rev-parse HEAD)" = 03b91d7d9ff295ae68a320e2e733dd6c2ef8f342; \
   T1=$(date +%s); ( cd /tmp/src/xformers && BUILD_VERSION=0.0.35+03b91d7.d20260904 FORCE_CUDA=1 XFORMERS_BUILD_TYPE=Release python -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$W" . ); echo "xformers built in $(( $(date +%s) - T1 ))s"; \
   git clone --quiet --depth 1 --branch v2.15 --recurse-submodules --shallow-submodules https://github.com/NVIDIA/TransformerEngine.git /tmp/src/TransformerEngine; \
   test "$(git -C /tmp/src/TransformerEngine rev-parse HEAD)" = 42b840051647eef89761a16dfdff87e82bb253ab; git -C /tmp/src/TransformerEngine config core.abbrev 7; test "$(git -C /tmp/src/TransformerEngine rev-parse --short HEAD)" = 42b8400; \
   NCCL_HOME=$(python -c "import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])"); CUDNN_HOME=$(python -c "import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])"); test -f "$NCCL_HOME/include/nccl.h"; test -f "$CUDNN_HOME/include/cudnn.h"; \
   T1=$(date +%s); ( cd /tmp/src/TransformerEngine && export CPATH="$NCCL_HOME/include:$CUDNN_HOME/include" LIBRARY_PATH="$NCCL_HOME/lib" && NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="$SM_ARCHS" NVTE_WITH_NCCL_EP=0 NVTE_BUILD_MAX_JOBS=$J CUDNN_PATH=$CUDNN_HOME python -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$W" . ); echo "transformer_engine built in $(( $(date +%s) - T1 ))s"; \
   python -m pip download --no-cache-dir --no-binary :all: --no-deps --no-build-isolation -d /tmp/src flash-attn==2.8.3.post1; \
   echo "55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348  /tmp/src/flash_attn-2.8.3.post1.tar.gz" | sha256sum -c -; \
   T1=$(date +%s); ( cd /tmp/src && MAX_JOBS=$JFA FLASH_ATTN_CUDA_ARCHS="$SM_ARCHS" FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$W" /tmp/src/flash_attn-2.8.3.post1.tar.gz ); echo "flash_attn built in $(( $(date +%s) - T1 ))s"; \
   python -m pip uninstall -y cmake ninja pybind11 pybind11-global nvidia-cudnn-frontend; \
   dpkg-query -W -f '${Package}\n' | sort > /tmp/dpkg.after; comm -13 /tmp/dpkg.before /tmp/dpkg.after > /tmp/dpkg.added; \
   echo "removing the $(wc -l < /tmp/dpkg.added) packages this step installed"; xargs -r apt-get purge -y -qq < /tmp/dpkg.added; \
   rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/cuda* /usr/local/cuda* /tmp/src /tmp/cuda-keyring.deb /tmp/dpkg.before /tmp/dpkg.after /tmp/dpkg.added /root/.cache; \
   ls -l "$W"; echo "esmfold2 wheels (STACK=$STACK, compute capability $TORCH_CUDA_ARCH_LIST): built in $(( $(date +%s) - T0 ))s on $(nproc --all) cores"; \
 else \
   python -I -c "import functools, hashlib, json, os, sys; keys = sys.argv[1:]; ws = functools.reduce(lambda d, k: d[k], keys, json.load(open('stock/PINS.json'))); bad = [w['name'] for w in ws if not os.path.isfile(os.path.join('stock/wheels', w['name'])) or hashlib.sha256(open(os.path.join('stock/wheels', w['name']), 'rb').read()).hexdigest() != w['sha256']]; print('stock/wheels: ' + (('MISSING OR OFF THEIR PIN (stock/PINS.json ' + '.'.join(keys) + '): ' + ', '.join(bad)) if bad else 'all %d wheels match stock/PINS.json %s' % (len(ws), '.'.join(keys)))); sys.exit(1 if bad else 0)" $PINS_WHEELS; \
 fi; \
 python -m pip install --no-cache-dir --no-index --no-deps "$W"/*.whl; python -m pip check

# The kit: `run.sh install` = the shared core and esmfold2_opt, editable, then the install checked against the pins (stock/check_pins.py).
RUN bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY esmfold2/environment/Dockerfile _jitcache/esmfold2-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
