# ESM C optimization kit — the tested stack as a container image.
# Builds Ubuntu 24.04 + the CUDA 13.0.1 runtime (public NVIDIA base; torch, flash-attn and Transformer Engine load the CUDA libraries of the
# nvidia-* wheels in the lock, Transformer Engine also NVRTC from the base's /usr/local/cuda/lib64) + gcc 13.3 (build-essential: Triton builds
# kernels and their launchers at first use) + clang 18 (the interpreter below was built with clang and names it as its C compiler, so
# setuptools compiles C extensions with it: DockQ's, a dependency esm declares, when pip installs it) + git (esm, the transformers fork and
# DockQ install from pinned git commits), Python 3.12.1 at /usr/local/bin/python (python-build-standalone 20240107, the
# x86_64_v3 build, with its bundled pip 23.3.2 and setuptools 69.0.3), the pinned stack of esmc/environment/requirements.lock (torch 2.11.0+cu130,
# triton 3.6.0, esm 3.4.0, the Biohub transformers fork 4.57.6, flash_attn 2.7.4.post1, transformer_engine 2.15.0, numpy 2.5.2), the CUDA 13.0
# compiler (nvcc) + ninja for the kit's one CUDA extension, then this kit, editable, under /kit (`run.sh install`, which compiles that extension
# for sm80 and sm90 — the image builder has no GPU, so both served capabilities are built). One image serves H100, H200 and A100 alike. The
# kit's Triton kernel compiles at the first model load of a container (~half a minute, once; mount a directory and name it in TRITON_CACHE_DIR
# to keep it across containers). No weights are inside: the weights live in a directory
# you mount, and HF_HOME names its hf/ subdirectory at run time (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 13.0 (580 or newer)
# and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding esmc/):
#   docker build -f esmc/environment/Dockerfile -t esmc-kit:dev .
# Run your own ESM C script with the kit on (README.md 'Run'; ESMC_OPT=off or unset = stock; HF_HUB_OFFLINE=1 is the image's default):
#   docker run --rm --gpus all -v /path/to/weights/hf:/weights/hf -e HF_HOME=/weights/hf -e ESMC_OPT=exact \
#     -v "$PWD":/work esmc-kit:dev python /work/your_script.py
# To try the kit in your own environment instead: Python 3.12, the lock installed as README.md 'Setup' shows (at minimum torch 2.11.0
# with CUDA 13.0, triton 3.6.0, esm 3.4.0 and the transformers fork at their pinned commits, flash_attn 2.7.4.post1 and transformer_engine 2.15.0
# for the kit modes), a C compiler, the CUDA toolkit's nvcc and ninja on PATH, then `run.sh install`.
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

# The process environment the stack was tested under, for the build's own compiles too: hash randomisation off, C builds without debug info
# (DockQ's extension here, Triton's launchers at run time), and the loader path of every tested run — the base's two driver directories plus
# the base toolkit's /usr/local/cuda/lib64 (inert for torch and flash-attn, which load the lock's nvidia-* wheels by path; Transformer Engine's
# core library resolves cuBLAS / cudart there only when it is imported before torch).
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
# Loads are pinned to the weights snapshot on disk: upstream's hub resolver never asks the Hub for a newer `main` inside a container.
# `run.sh install --weights DIR --variant V` clears the switch for its own download only; `-e HF_HUB_OFFLINE=0` opts a container out.
ENV HF_HUB_OFFLINE=1

# The C toolchains (gcc + libc headers for Triton; clang, the interpreter's own CC, for setuptools-built extensions), git for the three git-pinned
# packages, and the download tools of this build.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl git build-essential clang \
 && rm -rf /var/lib/apt/lists/*

# Python 3.12.1: the python-build-standalone release the stack was tested on, unpacked into /usr/local with its links made plain files (its own
# pip and setuptools come with it); `python` = python3.
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/astral-sh/python-build-standalone/releases/download/20240107/cpython-3.12.1+20240107-x86_64_v3-unknown-linux-gnu-install_only.tar.gz" \
 && echo "a7bcc6c9f66dbd47ea99615f30f101c5d2dd0084ca333d2f7336e64050951338  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -RL --preserve=mode,timestamps /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack, exactly the lock, in one pass (no dependency resolution: the lock is already the resolved set). The three git-pinned packages
# are built from source by pip in isolated build environments, which the constraint holds to the setuptools release their tested wheels were
# built with (setuptools 84.0.0; the lock installs that same release, and ninja, for the interpreter — torch's extension builder imports
# setuptools at `run.sh install` below); everything else is a wheel.
# BUILD_JOBS: the parallelism of whatever pip compiles from source in this step (DockQ's C extension — seconds; MAX_JOBS,
# CMAKE_BUILD_PARALLEL_LEVEL and MAKEFLAGS are what setuptools-, cmake- and make-driven builds read): all of the builder's cores unless a
# `--build-arg BUILD_JOBS=N` says otherwise. The step prints the count it ran with.
ARG BUILD_JOBS=0
COPY esmc/environment/requirements.lock /tmp/requirements.lock
RUN J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); export MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J; echo "build jobs: $J (cores: $(nproc --all))" \
 && echo "setuptools==84.0.0" > /tmp/build-constraints.txt \
 && PIP_CONSTRAINT=/tmp/build-constraints.txt python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.lock \
 && rm -f /tmp/build-constraints.txt /tmp/requirements.lock

# The CUDA 13.0 compiler and ninja: torch's extension builder compiles the kit's one CUDA extension with them at `run.sh install` below (the
# CUDA library headers it compiles against are the lock's nvidia-* wheels', already installed above).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-nvcc-13-0 ninja-build \
 && rm -rf /var/lib/apt/lists/* && /usr/local/cuda/bin/nvcc --version | tail -1

# This kit, installed the way README.md 'Setup' says (editable, the pin check, the extension built). HF_HOME etc. are the user's at run time.
COPY esmc /kit/esmc
WORKDIR /kit/esmc
RUN bash run.sh install
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY esmc/environment/Dockerfile _jitcache/esmc-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
