# Flashzoi optimization kit — the tested stack as a container image.
# Base: the official python:3.11.12-slim-bookworm image (Debian 12 + CPython 3.11.12 at /usr/local/bin/python), the base the tested stack was
# assembled on — there is no NVIDIA CUDA base image because this stack loads its whole CUDA 12.4 user space (runtime, cuBLAS, cuDNN 9.1, NCCL 2.21)
# from the pinned nvidia-*-cu12 wheels of flashzoi/environment/requirements.lock; the host supplies only the driver. On top: gcc 12 (build-essential:
# Triton builds the kit's kernels and their launchers at first use), the pinned stack of the lock (torch 2.5.1+cu124, triton 3.1.0, flash-attn
# 2.7.0.post2 as upstream's prebuilt wheel, transformers 4.57.6, huggingface_hub 0.36.2, numpy 2.4.6), stock borzoi-pytorch 0.5.1 from the wheel in
# flashzoi/stock/, then this kit, editable, under /kit (`run.sh install`; flashzoi_opt imports nothing from common/opt_core, so the image
# carries flashzoi/ alone). One image serves H100/H200 and A100: nothing is compiled for a GPU at build time. No weights are inside: `run.sh install --weights DIR` fetches the four pinned replicates into a Hugging Face
# hub-cache directory you mount, and FLASHZOI_WEIGHTS names it at run time (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 12.4
# (550 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding flashzoi/):
#   docker build -f flashzoi/environment/Dockerfile -t flashzoi-kit:dev .
# Run (README.md 'Run' has one command per mode; inputs are read from the mounted in/, outputs land under the mounted out/):
#   docker run --rm --gpus all -v /data/weights/flashzoi/hf:/weights -e FLASHZOI_WEIGHTS=/weights -v "$PWD/in":/kit/flashzoi/in \
#     -v "$PWD/out":/kit/flashzoi/out flashzoi-kit:dev bash run.sh pred --config h100 --mode exact --input in --out out/exact
# To try the kit in your own environment instead: Python 3.11, the lock installed as README.md 'Setup' route C shows (at minimum torch 2.5.1
# with CUDA 12.4, triton 3.1.0, flash-attn 2.7.0.post2, transformers, huggingface_hub, safetensors, numpy), the borzoi-pytorch wheel from
# flashzoi/stock/, a C compiler on PATH, then `run.sh install`.
FROM python:3.11.12-slim-bookworm

# The C toolchain Triton needs at run time (gcc, libc headers) and the download tools of this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential && rm -rf /var/lib/apt/lists/*

# The pinned stack. The installer's own tools first, at the lock's versions and by the interpreter's full name (pip, wheel, uv — the base layer
# of the tested stack has them so, and their scripts name that interpreter), then every other line of the lock with --no-deps (the lock is
# already the complete, resolved set; borzoi-pytorch is filtered out because the stock wheel below is that distribution), then stock from the file
# in flashzoi/stock/.
COPY flashzoi/environment/requirements.lock /tmp/requirements.lock
COPY flashzoi/stock/borzoi_pytorch-0.5.1-py3-none-any.whl /tmp/stock/borzoi_pytorch-0.5.1-py3-none-any.whl
RUN grep -E '^(pip|wheel|uv)==' /tmp/requirements.lock > /tmp/tools.txt \
 && python3.11 -m pip install --no-cache-dir --no-deps -r /tmp/tools.txt \
 && grep -v -E '^(#|borzoi-pytorch==)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/borzoi_pytorch-0.5.1-py3-none-any.whl \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/tools.txt /tmp/requirements.lock

# The kit. `run.sh install` = pip install -e flashzoi/opt, then the pin check (stock/check_pins.py) — the same step a user runs.
COPY flashzoi /kit/flashzoi
WORKDIR /kit/flashzoi
RUN bash run.sh install
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY flashzoi/environment/Dockerfile _jitcache/flashzoi-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment of the tested stack (hash seed; no debug info in the launchers Triton compiles). FLASHZOI_WEIGHTS is set at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0
