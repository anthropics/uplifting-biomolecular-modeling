# mosaic optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.8.1 runtime (public NVIDIA base: the runtime flavour, since nothing in this stack compiles CUDA source —
# jax brings its own ptxas and CUDA libraries as wheels, torch is the CPU build; the image keeps the base's /usr/local/cuda/lib64 on
# LD_LIBRARY_PATH, so jax's CUDA plugin resolves cuBLAS 12.8.4, cuSPARSE and nvJitLink from the base and the other CUDA libraries from its
# wheels — the resolution this stack is tested with), gcc 11.4
# (build-essential) and git (pip fetches the three pinned upstream packages from GitHub), Python 3.12.1 at /usr/local/bin/python3
# (python-build-standalone), the pinned stack of mosaic/environment/requirements.lock in the virtual environment /kit/venv (jax / jaxlib /
# jax-cuda12-plugin 0.10.2, equinox 0.13.8, torch 2.7.1+cpu, numpy 2.5.2; mosaic @ 70fec525, joltz @ ed0f0425, boltz @ 1acc397b), then this kit
# and its shared core, editable, under /kit, with the kit's lever files inside the installed mosaic package (`run.sh install`). /kit/venv/bin
# is first on PATH. One image serves H100 and A100: nothing is compiled for a GPU at build time (XLA compiles at first use; `run.sh warm`
# keeps a shape's compiled programs under MOSAIC_OPT_CACHE_ROOT). No Boltz-2 or AlphaFold parameters are inside (the upstream mosaic package
# installed here ships its own small ProteinMPNN/AbMPNN weight files — see THIRD_PARTY_NOTICES.md): `run.sh install --weights DIR` fetches the
# Boltz-2 checkpoint and CCD molecules into a directory you mount, and MOSAIC_CACHE_DIR names it at run time (README.md 'Setup'). Host: an NVIDIA
# driver that runs CUDA 12 (550 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding mosaic/ and common/):
#   docker build -f mosaic/environment/Dockerfile -t mosaic-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/, the P1 files under the mounted cache/):
#   docker run --rm --gpus all -v /path/to/mosaic_cache:/weights -e MOSAIC_CACHE_DIR=/weights -v "$PWD/cache":/cache -e MOSAIC_OPT_CACHE_ROOT=/cache \
#     -v "$PWD/out":/kit/mosaic/out mosaic-kit:dev bash run.sh design --config h100 --mode fast --out out/fast --seed 0
# To try the kit in your own environment instead: Python 3.12 with the lock installed into a virtual environment as README.md 'Setup'
# route C shows (at minimum jax / jaxlib / jax-cuda12-plugin 0.10.2, equinox 0.13.8, torch 2.7.1 CPU, numpy 2.5.x, gemmi, and the three
# upstream packages at their pins), then `run.sh install`.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# The C toolchain and git (pip builds the three upstream packages from their pinned commits), and the download tools of this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git build-essential && rm -rf /var/lib/apt/lists/*

# Python 3.12.1: the python-build-standalone release this stack was assembled with (the x86_64_v3 build), unpacked into /usr/local (no system Python is used).
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.12.1+20240107-x86_64_v3-unknown-linux-gnu-install_only.tar.gz" \
 && echo "a7bcc6c9f66dbd47ea99615f30f101c5d2dd0084ca333d2f7336e64050951338  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack in its own virtual environment, /kit/venv: the environment's installer tools at the lock's versions first, then every
# distribution at the lock's version (--no-deps: the lock is complete; the three upstream packages come from their pinned commits on GitHub,
# the torch CPU wheels from the direct URLs the lock names).
COPY mosaic/environment/requirements.lock /tmp/requirements.lock
RUN /usr/local/bin/python3 -m venv /kit/venv \
 && /kit/venv/bin/python -m pip install --no-cache-dir $(grep -E '^(pip|setuptools|wheel)==' /tmp/requirements.lock) \
 && /kit/venv/bin/python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.lock \
 && /kit/venv/bin/python -m pip freeze > /kit/venv/pip_freeze_built.txt && rm -f /tmp/requirements.lock \
 && echo "the pinned stack is installed; nothing in this image is compiled ($(nproc --all) cores visible to the build)"
ENV PATH=/kit/venv/bin:$PATH

# The kit: the shared core, then mosaic_opt, both editable into /kit/venv, the pin check, and the kit's lever files placed inside the installed
# mosaic package — the one install step, `run.sh install` (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY mosaic /kit/mosaic
WORKDIR /kit/mosaic
RUN bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY mosaic/environment/Dockerfile _jitcache/mosaic-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment of the tested stack (STOCK.md 'pinned stack'): hash seed fixed, no debug info in anything a C compiler builds, and the
# base's CUDA libraries directory on the loader path (jax's CUDA plugin resolves cuBLAS, cuSPARSE and nvJitLink there, the rest of the CUDA
# libraries from the nvidia-* wheels of the lock). The Boltz-2 weights cache and the P1 files' root are never baked in: MOSAIC_CACHE_DIR and
# MOSAIC_OPT_CACHE_ROOT are set at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64
