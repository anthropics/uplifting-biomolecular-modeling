# ColabDesign optimization kit — the tested stack as a container image.
# Builds on mambaorg/micromamba:2.1.1 — the base the tested image itself was built on (Debian 12 slim + micromamba 2.1.1 with its /opt/conda root prefix
# and its entrypoint, which runs every command inside the activated environment); no CUDA base image is involved: the CUDA 12.9 libraries jax loads are
# packages of the environment. On it: Debian's build-essential (gcc 12.2 at /usr/bin/gcc — the C compiler the kit's Setup names; it also builds freesasa
# below), git and wget as in the tested image; then the conda environment of colabdesign/environment/conda-linux-64.lock unpacked into /opt/conda as the
# base environment (an explicit package list: nothing is solved) — CPython 3.10.18 at /opt/conda/bin/python, jax / jaxlib 0.6.0 (conda-forge's CUDA 12
# build with the jax-cuda12-plugin 0.6.0 entries), dm-haiku 0.0.17, numpy 1.26.4, scipy 1.15.2, cudart / cuBLAS 12.9, cuDNN 9.10, NCCL 2.30, cuda-nvcc
# 12.9 for ptxas, and the rest of BindCraft's package list; then the two pip distributions of colabdesign/environment/requirements.lock — stock
# ColabDesign 1.1.3 at the pinned commit e31a56fe from its git URL and freesasa 2.2.1 — then this kit and its shared core, editable,
# under /kit (`run.sh install`). One image serves H100 and A100: nothing is compiled for a GPU at build time. No AlphaFold parameters are inside; the
# upstream ColabDesign package ships its ProteinMPNN weight files (MIT) — see THIRD_PARTY_NOTICES.md. `run.sh install --weights DIR` fetches the five AlphaFold-Multimer v3 parameter files into DIR/params/, and COLABDESIGN_PARAMS_DIR (or --params-dir) names DIR at run
# time (README.md 'Setup'). Host: an NVIDIA driver that loads the environment's CUDA 12.9 libraries (tested with driver 575; CUDA 12 minor-version compatibility from R525 on, untested here) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding colabdesign/ and common/):
#   docker build -f colabdesign/environment/Dockerfile -t colabdesign-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/af2:/weights -e COLABDESIGN_PARAMS_DIR=/weights -v "$PWD/out":/kit/colabdesign/out colabdesign-kit:dev \
#     bash run.sh design --config h100 --mode fast --starting-pdb stock/src/bindcraft/example/PDL1.pdb --chains A --binder-len 65 --target-hotspot-residues 56 --seed 0 --out out/fast
# To try the kit in your own environment instead: create the conda environment from conda-linux-64.lock (micromamba, mamba and conda take that file as
# it stands: STOCK.md 'Stack'), pip install requirements.lock (--no-deps; git and a C compiler on PATH), then `run.sh install`.
FROM mambaorg/micromamba:2.1.1

# root builds and runs the image (the tested image runs as root); Debian's compiler and download tools as the tested image has them.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates git wget bzip2 && rm -rf /var/lib/apt/lists/*

# The conda environment: the locked packages unpacked into the root prefix /opt/conda as its base environment (micromamba checks each md5), the
# package cache dropped afterwards. RUN lines from here on execute inside the activated environment (MAMBA_DOCKERFILE_ACTIVATE, the base image's shell).
COPY colabdesign/environment/conda-linux-64.lock /tmp/conda-linux-64.lock
RUN micromamba install -y -n base -f /tmp/conda-linux-64.lock \
 && micromamba clean -a -y && rm -f /tmp/conda-linux-64.lock
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# The image environment of the tested stack: the environment's programs and libraries first on the search paths for commands that bypass the entrypoint
# too (python = /opt/conda/bin/python; the CUDA libraries, cuDNN and NCCL jax loads resolve in /opt/conda/lib), the hash seed fixed, unbuffered output,
# CFLAGS as the tested image sets it, and the base image's activate-on-RUN switch kept for shells in the container. jax's own variables stay at their
# defaults (stock ColabDesign assigns XLA_FLAGS itself at import); CPU thread-pool counts are left to the host; the weights root is never baked in.
ENV PATH=/opt/conda/bin:/opt/conda/condabin:${PATH} \
    LD_LIBRARY_PATH=/opt/conda/lib \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    CFLAGS=-g0 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MAMBA_DOCKERFILE_ACTIVATE=1

# The pip layer, the lock as it stands (--no-deps: the dependencies are the conda packages above): stock ColabDesign at the pinned commit from its
# git URL (pip clones it with the git installed above and records the commit, as in the tested image) and freesasa from its PyPI source release —
# the one compile step of this build (freesasa's C extension, seconds). BUILD_JOBS = the parallelism offered to anything compiled here: all cores
# unless set (`docker build --build-arg BUILD_JOBS=8 …`); the step prints the core count it ran with.
ARG BUILD_JOBS=0
COPY colabdesign/environment/requirements.lock /tmp/requirements.lock
RUN J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); export MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J; echo "build cores: $(nproc --all), jobs: $J" \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.lock \
 && rm -f /tmp/requirements.lock

# The kit: the shared core, then colabdesign_opt, both editable, BindCraft's DSSP executable fetched from the BindCraft repository at the pinned commit
# (sha256-checked against stock/PINS.json; the tree does not carry it — network needed at this step) and the install checked against the pin — the one
# install step, `run.sh install` (README.md 'Setup', STOCK.md 'Pin').
COPY common/opt_core /kit/common/opt_core
COPY colabdesign /kit/colabdesign
WORKDIR /kit/colabdesign
RUN bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY colabdesign/environment/Dockerfile _jitcache/colabdesign-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
