# ChromBPNet optimization kit — the tested stack as a container image.
# Base: tensorflow/tensorflow:2.8.2-gpu, the image upstream chrombpnet's own dockerfile builds on and the one this stack was assembled in — Ubuntu 20.04,
# Ubuntu's Python 3.8.10 at /usr/bin/python3.8 (`python` on PATH), the CUDA 11.2.2 libraries, cuDNN 8.1.0.77 and TensorRT 7.2.2 from NVIDIA's apt packages,
# gcc 9.4 (build-essential: Triton compiles the K1 route's kernel launchers at first use). On it: pip 25.0.1 / setuptools 75.3.2, the pinned stack of
# chrombpnet/environment/requirements.lock (tensorflow 2.8.0, numpy 1.23.4, h5py 3.11.0, modisco, …), stock chrombpnet 1.0.1 from the source archive in
# chrombpnet/stock/ unpacked at /scratch/chrombpnet, the K1 route's torch 2.4.1+cu124 / triton 3.0.0 under the side root /opt/torch
# (chrombpnet/environment/requirements-opt-torch.lock; off the default sys.path, the kit inserts it), then this kit, editable, under /kit (`run.sh install`).
# One image serves H100 / H200 (K1 Triton forward) and A100 (the TensorFlow route): nothing is compiled for a GPU at build time. No weights are inside:
# the three model files and the driver-cache tarball live in a directory you mount and name with CHROMBPNET_OPT_WEIGHTS (README.md 'Setup';
# `run.sh install --weights DIR` checks them). Host: an NVIDIA driver that runs CUDA 12.4 (550 or newer; the TensorFlow route alone runs from 460)
# and the NVIDIA Container Toolkit.
# Build, from the bundle's top directory (the one holding chrombpnet/ and common/):
#   docker build -f chrombpnet/environment/Dockerfile -t chrombpnet-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/weights:/weights -v /path/to/data:/data -e CHROMBPNET_OPT_WEIGHTS=/weights -e CHROMBPNET_OPT_DATA=/data \
#     -v "$PWD/out":/kit/chrombpnet/out chrombpnet-kit:dev bash run.sh warm --config h100 --out out/warm
# To try the kit in your own environment instead: Python 3.8 with the two locks installed as STOCK.md 'Stack' shows (at minimum tensorflow
# 2.8.0 on CUDA 11.2 / cuDNN 8.1, numpy 1.23.4, h5py 3.11.0, pyBigWig, pyfaidx; torch 2.4.1 + triton 3.0.0 importable or under /opt/torch for H100/H200),
# stock chrombpnet from chrombpnet/stock/, a C compiler on PATH, then `run.sh install`.
FROM tensorflow/tensorflow:2.8.2-gpu

# Upstream's own apt line (its dockerfile: the download utilities plus the cairo / pango / gdk-pixbuf libraries its report rendering loads); the base
# already holds build-essential, pkg-config and python3-dev. The CUDA apt list the base carries is signed with a key NVIDIA has since rotated, so it is
# removed before apt runs, as upstream's dockerfile does; no CUDA package is installed or changed here.
RUN rm -f /etc/apt/sources.list.d/cuda.list \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y wget bzip2 ca-certificates curl git jq libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev \
 && rm -rf /var/lib/apt/lists/*

# The lock's own pip and setuptools lines first (the base interpreter's pip 20.2 cannot lay down this stack; setuptools 75.3.2 builds the five source
# distributions and the kit's editable install), then the pinned stack: every distribution of requirements.lock at the lock's version (--no-deps: the lock
# is complete; it moves the base's tensorflow 2.8.2 to the 2.8.0 stock pins), then stock chrombpnet 1.0.1 from the bundled source archive, unpacked where
# upstream's dockerfile keeps its checkout (--no-deps: its requirements.txt is what the lock already holds). `python3 -m pip`, as upstream installs:
# the console scripts it writes name /usr/bin/python3, the interpreter `python` also is.
COPY chrombpnet/environment/requirements.lock /tmp/requirements.lock
COPY chrombpnet/stock/chrombpnet-eaa0fe58.tar.gz /tmp/chrombpnet-eaa0fe58.tar.gz
RUN grep -E '^(pip|setuptools)==' /tmp/requirements.lock | xargs python3 -m pip install --no-cache-dir \
 && grep -v -E '^(#|$)' /tmp/requirements.lock > /tmp/stack.txt \
 && python3 -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && mkdir -p /scratch && tar -xzf /tmp/chrombpnet-eaa0fe58.tar.gz -C /scratch && mv /scratch/chrombpnet-eaa0fe58 /scratch/chrombpnet \
 && python3 -m pip install --no-cache-dir --no-deps /scratch/chrombpnet \
 && rm -f /tmp/stack.txt /tmp/requirements.lock /tmp/chrombpnet-eaa0fe58.tar.gz

# The K1 route's torch stack under the side root /opt/torch (stock/PINS.json pins_s1_only), off the interpreter's default sys.path: the kit's H100/H200
# forward inserts it at import; the TensorFlow environment above never sees it (--target: a self-contained directory; --no-deps: the lock is complete;
# `python -m pip` here: the side root's own scripts name /usr/local/bin/python, as the tested image's do).
COPY chrombpnet/environment/requirements-opt-torch.lock /tmp/requirements-opt-torch.lock
RUN python -m pip install --no-cache-dir --no-deps --target /opt/torch -r /tmp/requirements-opt-torch.lock && rm -f /tmp/requirements-opt-torch.lock

# The kit: chrombpnet_opt, editable, and the install checked against the pins — the one install step, `run.sh install` (README.md 'Setup'). The kit
# depends on nothing but the stack above (it does not use the shared opt_core).
COPY chrombpnet /kit/chrombpnet
WORKDIR /kit/chrombpnet
RUN bash run.sh install
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY chrombpnet/environment/Dockerfile _jitcache/chrombpnet-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment every run of the tested stack had: hash seed fixed, no debug info in the launchers Triton compiles. The base image's own ENV
# (CUDA_VERSION, LD_LIBRARY_PATH, NVIDIA_*, LANG=C.UTF-8, PATH with /usr/local/cuda/bin) is inherited unchanged. Weights and data roots are never baked in:
# CHROMBPNET_OPT_WEIGHTS and CHROMBPNET_OPT_DATA are set at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0
