# RoseTTAFold3 optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.6.3 runtime (public NVIDIA base — the tested stack's own base lineage, nvidia/cuda 12.6.3 on Ubuntu 22.04, at
# runtime weight: torch loads every CUDA library it uses from the pinned nvidia-* wheels and no mode uses the CUDA compiler, so the CUDA toolkit is
# not inside) + gcc 11.4 (build-essential: Triton builds the kit's kernels and their launchers at first use) + git (stock installs from its repository at
# the pinned commit), Python 3.12.1 at /usr/local/bin/python (python-build-standalone 20240107, the x86_64_v3 build), the pinned stack of
# rosettafold3/environment/requirements.lock (torch 2.13.0+cu130 with the CUDA 13.0 / cuDNN 9.20 / NCCL 2.29 runtime wheels, triton 3.7.1,
# cuequivariance 0.11.1 with its cu13 ops pair, atomworks 2.2.1, lightning 2.6.5, numpy 2.4.4, and stock rc-foundry 0.2.1.dev13 from
# RosettaCommons/foundry at commit 4010e3e2e), then this kit and its shared core under /kit (`run.sh install --make-venv`). Two interpreters,
# as the kit requires (README.md 'Setup'): /usr/local/bin/python is the pristine stock interpreter (`--mode off` runs there), /kit/rosettafold3/opt/venv
# is the patched interpreter derived from it (the kit modes run there; the shared core and the kit are installed editable into it, and its own
# copy of the rf3 package carries the add-on's five rf3 files, applied at image build from the kit tree at /kit/rosettafold3). /usr/local stays
# PRISTINE and runs `--mode off`. `run.sh install` inside the container is then a check that changes nothing (with `--weights DIR` it also fetches
# the checkpoint); an image converted to another container format carries the applied files as they are.
# One image serves H100 and A100: nothing is compiled for a GPU at build time. No weights are inside: `run.sh install --weights DIR` fetches the one
# checkpoint (2.8 GiB) into a directory you mount, and ROSETTAFOLD3_OPT_CKPT names that file at run time (README.md 'Setup'). Host: an NVIDIA
# driver that runs CUDA 13.0 (580 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding rosettafold3/ and common/):
#   docker build -f rosettafold3/environment/Dockerfile -t rosettafold3-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/weights:/weights -e ROSETTAFOLD3_OPT_CKPT=/weights/rf3_foundry_01_24_latest_remapped.ckpt \
#     -v "$PWD/out":/kit/rosettafold3/out rosettafold3-kit:dev \
#     bash run.sh pred --config h100 --mode fast --input opt/forward/rf3_xattempt_addon/public_inputs/1brs_tiles.json --out_dir out/fast
# The same stack without Docker: Python 3.12, git and a C compiler on PATH, the two pip steps below on
# rosettafold3/environment/requirements.lock, then `run.sh install --make-venv` from rosettafold3/ (there the add-on is applied in the same step).
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

# The C toolchain Triton needs at run time (gcc, libc headers), git for the stock pin, and the download tools of this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget git build-essential && rm -rf /var/lib/apt/lists/*

# Python 3.12.1: the python-build-standalone release (x86_64_v3 build) this stack was assembled with, unpacked into /usr/local (no system Python is
# used); it carries pip 23.3.2.
RUN wget -q -O /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.12.1+20240107-x86_64_v3-unknown-linux-gnu-install_only.tar.gz" \
 && echo "a7bcc6c9f66dbd47ea99615f30f101c5d2dd0084ca333d2f7336e64050951338  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete; stock rc-foundry is the lock's `git+https` line, cloned
# at the pinned commit and built with the hatchling release that built the tested wheel), then the cu13 cuDNN / NCCL wheels laid down once more:
# they share their nvidia/cudnn and nvidia/nccl directories with the cu12 wheels of the lock, and the tested tree holds the cu13 files there.
COPY rosettafold3/environment/requirements.lock /tmp/requirements.lock
RUN echo "build cores: $(nproc --all)" && grep -v -E '^#' /tmp/requirements.lock > /tmp/stack.txt \
 && printf 'hatchling==1.31.0\nhatch-vcs==0.4.0\n' > /tmp/build-pins.txt \
 && PIP_CONSTRAINT=/tmp/build-pins.txt python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps --force-reinstall nvidia-cudnn-cu13==9.20.0.48 nvidia-nccl-cu13==2.29.7 \
 && rm -f /tmp/stack.txt /tmp/build-pins.txt /tmp/requirements.lock

# The kit: the patched interpreter /kit/rosettafold3/opt/venv derived from the stock one, the shared core and rosettafold3_opt installed editable into
# it, both interpreters' tree states read, the pin checked on both, then the add-on's five rf3 files applied into /kit/rosettafold3/opt/venv — the
# one install step, `run.sh install` (README.md 'Setup'). Only the venv's copy of the package is written: /usr/local/bin/python stays pristine for
# `--mode off`.
COPY common/opt_core /kit/common/opt_core
COPY rosettafold3 /kit/rosettafold3
WORKDIR /kit/rosettafold3
RUN bash run.sh install --make-venv
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY rosettafold3/environment/Dockerfile _jitcache/rosettafold3-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment of the tested stack (STOCK.md 'pinned stack'): hash seed fixed, no debug info in the launchers Triton compiles (the
# loader path, LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64, is the base image's own and stays as it sets it).
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0
# This image's deployment parameter (configs/h100.env fills it only when unset): the pristine stock interpreter is the image's own python. The
# checkpoint is never baked in: ROSETTAFOLD3_OPT_CKPT is set at run time.
ENV ROSETTAFOLD3_OPT_STOCK_PYTHON=/usr/local/bin/python
