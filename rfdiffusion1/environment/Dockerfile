# RFdiffusion-1 optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.4.1 runtime (public NVIDIA base; the runtime flavour suffices: the stack loads its CUDA 12.1 / cuDNN 9.1 /
# NCCL 2.20 libraries from the nvidia-* wheels of the lock and compiles no CUDA source at build or install time) + gcc 11.4 (build-essential:
# Triton builds the `fast` line's kernels and their launchers at first use), Python 3.11.5 at /usr/local/bin/python, the pinned stack of
# rfdiffusion1/environment/requirements.lock (torch 2.4.0+cu121, dgl 2.4.0+cu124, triton 3.0.0, e3nn 0.5.1, hydra-core 1.3.2, numpy 1.26.4),
# stock RFdiffusion 1.1.0 (RosettaCommons/RFdiffusion @ 86507b65) unpacked from the source archive in rfdiffusion1/stock/ to /opt/rfd — its
# vendored SE3Transformer installed as a package, the checkout itself editable — then this kit and its shared core, editable, under /kit
# (`run.sh install`). One image serves H100 and A100: nothing is compiled for a GPU at build time. No weights are inside: `run.sh install
# --weights DIR` fetches the two pinned checkpoints into a directory you mount, and WEIGHTS names it at run time (README.md 'Setup').
# Host: an NVIDIA driver that runs CUDA 12.4 (550 or newer) and the NVIDIA Container Toolkit.
# Build, from the bundle's top directory (the one holding rfdiffusion1/ and common/):
#   docker build -f rfdiffusion1/environment/Dockerfile -t rfdiffusion1-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   T=opt/forward/fast_inference/inputs_public/insulin_target.pdb      (a bundled example target; relative paths resolve from /kit/rfdiffusion1 in the image)
#   docker run --rm --gpus all -v /path/to/rfd_weights:/weights -e WEIGHTS=/weights -v "$PWD/out":/kit/rfdiffusion1/out rfdiffusion1-kit:dev \
#     bash run.sh design --config h100 --mode fast inference.input_pdb=$T \
#     'contigmap.contigs=[A1-115/0 80-80]' 'ppi.hotspot_res=[A59,A83,A91]' inference.num_designs=4 inference.output_prefix=out/fast/des
# To try the kit in your own environment instead: Python 3.11, the lock installed with `pip install --no-deps -r` (at minimum torch 2.4.0
# with CUDA 12.1, dgl 2.4.0, triton 3.0.0, e3nn 0.5.1, hydra-core 1.3.2, numpy 1.26.x), stock RFdiffusion at the pin installed from
# rfdiffusion1/stock/ as below, a C compiler on PATH, then `run.sh install`.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# The C toolchain Triton needs at run time (gcc, libc headers) and the download tools of this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential && rm -rf /var/lib/apt/lists/*

# Python 3.11.5: the python-build-standalone release this stack was assembled with, unpacked into /usr/local (no system Python is used).
# (The last command prints the build machine's core count beside the interpreter version, for reading build times off the log.)
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V && echo "build machine cores: $(nproc --all)"

# The pinned stack, exactly as locked: --no-deps installs the listed builds without re-resolving anything (dgl's line names its wheel by URL and digest).
COPY rfdiffusion1/environment/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.lock && rm -f /tmp/requirements.lock

# Stock and the kit are source trees pip builds in an isolated environment; that environment's setuptools is pinned (PIP_CONSTRAINT, set inside
# the two build steps below and nowhere else), so the install metadata and the editable-install finder pip writes are the same on every build.
RUN printf 'setuptools==84.0.0\n' > /usr/local/share/pip-build-constraints.txt

# Stock RFdiffusion at the pin, from the source archive shipped in rfdiffusion1/stock/ (PINS.json `archive`): the checkout at /opt/rfd, its
# vendored SE3Transformer installed as a package, the checkout installed editable (upstream's own two install steps, run as python3.11 — the
# name upstream's run_inference.py launcher in /usr/local/bin then carries), and the directory upstream writes its IGSO(3) schedule cache into
# on first use (rfdiffusion/inference/model_runners.py: <checkout>/schedules): root's like the checkout and closed to every other user —
# upstream unpickles what it finds there, so a directory every user could write would run any user's file. A process under another uid is
# handed a directory of its own by the kit (opt/rfdiffusion1_opt/schedules.py).
COPY rfdiffusion1/stock/rfdiffusion-86507b65.tar.gz /tmp/stock/rfdiffusion-86507b65.tar.gz
RUN mkdir -p /opt && tar -xzf /tmp/stock/rfdiffusion-86507b65.tar.gz -C /opt && mv /opt/rfdiffusion-86507b65 /opt/rfd && rm -rf /tmp/stock \
 && export PIP_CONSTRAINT=/usr/local/share/pip-build-constraints.txt \
 && python3.11 -m pip install --no-cache-dir --no-deps /opt/rfd/env/SE3Transformer \
 && python3.11 -m pip install --no-cache-dir --no-deps -e /opt/rfd \
 && mkdir -m 755 /opt/rfd/schedules

# The kit: the shared core and the one-GPU packing launcher (`--pack K`) from common/, then rfdiffusion1/, the core and the kit package installed
# editable and the install checked against the pin — the one install step, `run.sh install` (README.md 'Setup'); a stack off the pin stops the build here.
COPY common/opt_core /kit/common/opt_core
COPY common/mps_packing /kit/common/mps_packing
COPY rfdiffusion1 /kit/rfdiffusion1
WORKDIR /kit/rfdiffusion1
RUN PIP_CONSTRAINT=/usr/local/share/pip-build-constraints.txt bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY rfdiffusion1/environment/Dockerfile _jitcache/rfdiffusion1-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment the kit's numerics were tested under: the hash seed fixed, no debug info in the launchers Triton compiles, gcc and
# g++ as the compilers Triton and any source build call (the interpreter's own build records clang, which is not in the image). The loader path
# stays the base image's. WEIGHTS is deliberately unset: name your weights directory at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    CC=gcc \
    CXX=g++
