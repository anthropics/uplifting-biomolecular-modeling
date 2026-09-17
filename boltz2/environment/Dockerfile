# Boltz-2 optimization kit — the tested stack as a container image.
# Base: the official Python image python:3.11.12-slim-bookworm (Debian 12, CPython 3.11.12 at /usr/local/bin/python) — the interpreter this stack
# was assembled and tested on, byte for byte, and the reason no CUDA base image is involved: torch 2.12.0 (CUDA 13.0 build) loads the CUDA 13.0
# runtime, cuBLAS, cuDNN 9.20 and NCCL 2.29 from the nvidia-* wheels of boltz2/environment/requirements.lock, so the host needs only an NVIDIA
# driver that runs CUDA 13.0 (580 or newer) and the NVIDIA container toolkit. On top: gcc and gfortran 12.2 (build-essential: Triton builds the
# kits' kernels and their launchers at first use), the pinned stack of requirements.lock (torch 2.12.0, triton 3.7.0, cuequivariance 0.10.0 with its CUDA 13
# ops, pytorch-lightning 2.5.0, numpy 1.26.4, rdkit, scipy — the lock is the full list), stock boltz 2.2.1 from the wheel in boltz2/stock/, then
# this kit and its shared core, editable, under /kit (`run.sh install`). One image serves H100 and A100: nothing is compiled for a GPU at build
# time. No weights are inside: `run.sh install --weights DIR` fills a directory you mount with the Boltz-2 cache (the two checkpoints, the CCD,
# the molecule library) and BOLTZ_CACHE names it at run time (README.md 'Setup').
# Build, from the bundle's top directory (the one holding boltz2/ and common/):
#   docker build -f boltz2/environment/Dockerfile -t boltz2-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/boltz_cache:/weights -e BOLTZ_CACHE=/weights -v "$PWD/out":/kit/boltz2/out boltz2-kit:dev \
#     bash run.sh pred --config h100 --mode fast --input inputs/1BRS_barnase_barstar.yaml --out_dir out/fast --seeds 0
# To try the kit in your own environment instead: Python 3.11, the lock installed as STOCK.md 'Stack' shows (at minimum torch 2.12.0
# with CUDA 13.0, triton 3.7.0, cuequivariance 0.10.0 with cuequivariance-ops-torch-cu13, pytorch-lightning 2.5.0, numpy 1.26.x, rdkit), the
# boltz wheel from boltz2/stock/, a C compiler on PATH, then `run.sh install`.
FROM python:3.11.12-slim-bookworm

# The C toolchain Triton needs at run time (gcc, libc headers; the Python headers ship with the base image) and gfortran, the tested stack's
# other compiler (nothing in the lock builds Fortran; it is part of the system the stack was tested on).
RUN apt-get update && apt-get install -y build-essential gfortran && rm -rf /var/lib/apt/lists/*

# The pinned stack: pip and wheel at the lock's versions first (through python3.11, the name their launchers carry in the tested stack), then
# every distribution at the lock's version (--no-deps: the lock is complete; the source-only distributions in it — antlr4-python3-runtime,
# fairscale, ihm, modelcif — build here with the setuptools release the tested stack built them with, held by --build-constraint), then stock
# from the bundled wheel (--no-deps; the lock's own boltz line names the same release and is left out so the file in boltz2/stock/ is the one installed).
COPY boltz2/environment/requirements.lock /tmp/requirements.lock
COPY boltz2/stock/boltz-2.2.1-py3-none-any.whl /tmp/stock/boltz-2.2.1-py3-none-any.whl
RUN grep -E '^(pip|wheel)==' /tmp/requirements.lock > /tmp/pip.txt && python3.11 -m pip install --no-cache-dir --no-deps -r /tmp/pip.txt \
 && printf 'setuptools==82.0.1\n' > /tmp/build-constraints.txt \
 && grep -v -E '^(#|pip==|wheel==|boltz==)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps --build-constraint /tmp/build-constraints.txt -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/boltz-2.2.1-py3-none-any.whl \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/pip.txt /tmp/build-constraints.txt /tmp/requirements.lock

# The kit: the shared core, then boltz2_opt, both editable, and the install checked against the pin — the one install step, `run.sh install` (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY boltz2 /kit/boltz2
WORKDIR /kit/boltz2
RUN bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
# One tar per stack key, made in a warmed container from the root so it unpacks as <stack key>/…, staged in the build context's _jitcache/ (never in the tree):
#   tar -C /opt/jit_cache -cf _jitcache/boltz2-<stack key>-jit.tar <stack key>
# run.sh uses /opt/jit_cache in place when it is writable, else seeds a writable copy once (its `[boltz2-kit] jit cache:` line says which).
COPY boltz2/environment/Dockerfile _jitcache/boltz2-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment of the tested stack: hash seed fixed, no debug info in the launchers Triton compiles, and the library search path of
# the CUDA 13 libraries in the nvidia wheels for compiled extensions that do not carry their own (torch itself resolves them through its wheels'
# RPATH). The cache directory is never baked in: BOLTZ_CACHE is set at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib
