# Caliby optimization kit — the tested stack as a container image.
# Builds on the public python:3.12.10-slim-bookworm image — Debian 12 with Python 3.12.10 at /usr/local/bin/python, the interpreter this stack was
# assembled with; no CUDA toolkit in the system: torch's CUDA 12.4 / cuDNN 9.1 / NCCL 2.21 user-space libraries are the nvidia-*-cu12 wheels of the
# lock — plus gcc 12.2 (build-essential: Triton builds the kit's fused LCP kernel launcher at first use, and pip builds the lock's few source
# distributions), the pinned stack of caliby/environment/requirements.lock (torch 2.6.0+cu124, triton 3.2.0, numpy 2.1.3, lightning 2.6.5), stock
# caliby 0.1, atomworks-caliby 1.0.0 and protpardelle 1.3.2 from the archives in caliby/stock/ (the pinned commits), then this kit and its shared
# core, editable, under /kit (`run.sh install`). One image serves H100 and A100: nothing is compiled for a GPU at build time. No weights are inside:
# `run.sh install --weights DIR` fetches them into a directory you mount, and MODEL_PARAMS_DIR names it at run time (README.md 'Setup'). Host: an
# NVIDIA driver that runs CUDA 12.4 (550 or newer) and the NVIDIA Container Toolkit.
# Build, from the bundle's top directory (the one holding caliby/ and common/):
#   docker build -f caliby/environment/Dockerfile -t caliby-kit:dev .
# Run (README.md 'Run' has one command per mode and variant; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/model_params:/weights -e MODEL_PARAMS_DIR=/weights -v "$PWD/out":/kit/caliby/out caliby-kit:dev \
#     bash run.sh design --config h100 --mode fast --input opt/forward/xattempt_addon/tests/public_inputs/7xhz.cif --out_dir out/fast --seed 0
# To try the kit in your own environment instead: Python 3.12, the lock installed as README.md 'Setup' route C shows (at minimum torch 2.6.0 with
# CUDA 12.4, triton 3.2.0, numpy 2.1.x, lightning, hydra-core, biotite, gemmi), the three upstream archives from caliby/stock/, a C compiler on PATH,
# then `run.sh install`.
FROM python:3.12.10-slim-bookworm

# The C toolchain Triton needs at run time (gcc, libc headers); pip uses it once here for the lock's source distributions.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# The process environment of the tested stack (STOCK.md 'pinned stack'): hash seed fixed, no debug info in what the C compiler builds (the source
# distributions here, Triton's launchers at run time), unbuffered output, the Hugging Face client offline and wandb disabled (nothing is fetched or
# reported at run time; `run.sh install --weights` lifts the offline switch for its own process), and the two atomworks mirror variables present and
# empty (no local PDB/CCD mirror). The weights root is never baked in: MODEL_PARAMS_DIR is set at run time.
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    CFLAGS=-g0 \
    HF_HUB_OFFLINE=1 \
    WANDB_MODE=disabled \
    PDB_MIRROR_PATH="" \
    CCD_MIRROR_PATH=""

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete; pip, setuptools and wheel come first so the rest
# installs and builds with the lock's own — invoked as python3.12, the name pip stamps into the pip and wheel launchers of this stack), then stock
# from the bundled archives in place of the lock's three `@ git+https` lines (--no-deps).
COPY caliby/environment/requirements.lock /tmp/requirements.lock
COPY caliby/stock/caliby-41d31560.tar.gz caliby/stock/atomworks-caliby-ea2c998a.tar.gz caliby/stock/protpardelle-1c-7962da09.tar.gz /tmp/stock/
RUN grep -E '^(pip|setuptools|wheel)==' /tmp/requirements.lock > /tmp/installers.txt \
 && grep -v -E '^#|^(pip|setuptools|wheel)==| @ git\+' /tmp/requirements.lock > /tmp/stack.txt \
 && python3.12 -m pip install --no-cache-dir --no-deps -r /tmp/installers.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/atomworks-caliby-ea2c998a.tar.gz /tmp/stock/protpardelle-1c-7962da09.tar.gz /tmp/stock/caliby-41d31560.tar.gz \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/installers.txt /tmp/requirements.lock

# The kit: the shared core, then caliby_opt, both editable, and the install checked against the pins — the one install step, `run.sh install` (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY caliby /kit/caliby
WORKDIR /kit/caliby
RUN bash run.sh install
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY caliby/environment/Dockerfile _jitcache/caliby-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
