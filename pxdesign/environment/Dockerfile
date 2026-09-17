# PXDesign optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.1.1 development toolkit (public NVIDIA base; the development flavour because stock's LayerNorm CUDA extension —
# LAYERNORM_TYPE=fast_layernorm, `--use_fast_ln true`, stock's default and every mode's — is compiled with nvcc: here at build time, and by
# protenix itself at first use wherever the compiled module is absent), gcc 11.4 and git, Python 3.11.5 at /usr/local/bin/python, the pinned stack of
# pxdesign/environment/requirements.lock (torch 2.3.1+cu121 with the CUDA 12.1 / cuDNN 8.9.2 / NCCL 2.20.5 runtime wheels, triton 2.3.1, numpy 1.26.3,
# deepspeed 0.15.4) including stock — protenix 0.5.0+pxd from its repository at the pinned commit, pxdesign 0.1.0 and pxdbench 0.1.2 as editable
# checkouts of their pinned commits under /opt (STOCK.md) — stock's LayerNorm CUDA extension compiled into the protenix package the way protenix
# compiles it (compute capability 7.0, 8.0, 8.6 and 9.0; no GPU is needed to compile), then this kit and its shared core, editable, under /kit
# (`run.sh install`). One image serves H100 and A100; a run starts without a compiler pass. No weights are inside:
# `run.sh install --weights DIR` fetches the four checkpoint files and the three CCD-cache files into DIR/checkpoint and DIR/ccd_cache on a
# directory you mount, and PXDESIGN_CKPT_DIR / PROTENIX_DATA_ROOT_DIR name those two at run time (README.md 'Setup'). Host: an NVIDIA driver that
# runs CUDA 12.1 (525.60 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding pxdesign/ and common/):
#   docker build -f pxdesign/environment/Dockerfile -t pxdesign-kit:dev .
# Run (README.md 'Run' has the design command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/pxdesign_data:/weights -e PXDESIGN_CKPT_DIR=/weights/checkpoint -e PROTENIX_DATA_ROOT_DIR=/weights/ccd_cache \
#     -v "$PWD/out":/kit/pxdesign/out pxdesign-kit:dev bash run.sh warm --config h100 --mode fast
# To try the kit in your own environment instead: Python 3.11, git, gcc and the CUDA 12.1 toolkit (nvcc) on PATH, the lock installed with
# `pip install --no-deps -r` as below (at minimum torch 2.3.1 with CUDA 12.1, numpy 1.26.3, deepspeed 0.15.4 and the three stock lines), then `run.sh install`.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# The C toolchain (gcc, libc headers: the LayerNorm extension's host code and the posix_ipc build of the lock), git (pip clones the three stock
# repositories the lock names) and the download tools of this build.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Python 3.11.5: the python-build-standalone distribution (release 20230826, x86_64-unknown-linux-gnu, install_only), unpacked into /usr/local.
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The process environment of the tested stack: hash randomization off, and no debug info in what the C compiler builds here and at run time
# (the posix_ipc extension of the lock, the LayerNorm extension's host objects). The loader path is the base image's own.
ENV PYTHONHASHSEED=0 CFLAGS=-g0

# The pinned stack, stock included: every line of the lock exactly, nothing resolved (--no-deps). torch 2.3.1+cu121 comes from the PyTorch
# index the lock names; protenix installs from its repository at the pinned commit; pxdesign and pxdbench are cloned at their pinned commits
# under /opt (--src) and installed editable, the form PXDesign's own install uses.
# pxdesign's checkout drops its .git directory in the same layer: nothing reads it after the install, and the pin check compares the
# checked-out files with pxdesign/stock/src rather than asking git. pxdbench's checkout keeps its .git: no copy of PXDesignBench is
# carried in the tree, so the pin check reads the checked-out commit from it (stock/check_pins.py). The same layer also deletes two
# proprietary font files of PXDesign's checkout (pxdesign/pxd_server/*.ttf), used only by the upstream web UI, which this kit does not run:
# no layer of the image carries them (STOCK.md 'Pin'; the pin check compares source files only).
COPY pxdesign/environment/requirements.lock /tmp/requirements.lock
RUN grep -v '^#' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps --src /opt -r /tmp/stack.txt \
 && rm /tmp/stack.txt /tmp/requirements.lock \
 && rm -rf /opt/pxdesign/.git \
 && rm /opt/pxdesign/pxdesign/pxd_server/Helvetica-Regular.ttf /opt/pxdesign/pxdesign/pxd_server/TimesNewRoman.ttf

# Stock's LayerNorm CUDA extension, compiled now the way protenix compiles it at first use (protenix/model/layer_norm/layer_norm.py: importing the
# module when no compiled fastfold_layer_norm_cuda is present builds one with nvcc and ninja into the protenix package directory). The image then
# carries the compiled module: no run waits on a compiler, and a read-only image (Apptainer) serves LAYERNORM_TYPE=fast_layernorm. Build
# parallelism: every core unless BUILD_JOBS says otherwise (`docker build --build-arg BUILD_JOBS=8 …`); the extension is two translation units
# for the GPU architectures protenix itself lists, about three minutes of nvcc.
ARG BUILD_JOBS=0
RUN J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); echo "compiling stock's LayerNorm CUDA extension: $J jobs on $(nproc --all) cores" \
 && MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=4 \
    python -c "import os, protenix.model.layer_norm.layer_norm as m; p = os.path.join(os.path.dirname(m.__file__), 'fastfold_layer_norm_cuda.so'); assert os.path.isfile(p), p; print('compiled', p)"

# The kit and its shared core at the layout run.sh expects (pxdesign/ beside common/), installed editable by the kit's own install step, which
# ends with the pin check of the three stock packages (pxdesign and protenix against the copies in pxdesign/stock/src, pxdbench by
# version and checked-out commit).
COPY common/opt_core /kit/common/opt_core
COPY pxdesign /kit/pxdesign
WORKDIR /kit/pxdesign
RUN bash run.sh install
