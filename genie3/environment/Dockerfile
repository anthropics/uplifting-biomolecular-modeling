# Genie 3 optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.6.3 runtime with cuDNN 9.5.1 (public NVIDIA base: the runtime flavour of the cudnn-devel image the stack was
# assembled and tested on — nothing in the stack compiles CUDA sources, so the compiler half is left out; the base's own libcudnn9 package stays
# because torch's cuDNN loader resolves its versioned sub-libraries from it ahead of the copies in the pip wheel) + gcc 11.4 (build-essential:
# Triton builds the `fast` line's fused kernel and its launchers at first use) + git (stock Genie 3 is a source checkout at a pinned commit),
# Python 3.10.13 at /usr/local (the python-build-standalone 20230826 release, links dereferenced), the venv /opt/venv_g3 holding the pinned stack of
# genie3/environment/requirements.lock (torch 2.7.1+cu126 with the CUDA 12.6 / cuDNN 9.5 / NCCL 2.26 runtime wheels, triton 3.3.1, lightning 2.6.5,
# numpy 2.2.6, scipy 1.15.3), stock Genie 3 at commit d77ae5ac as the editable checkout /opt/genie3 (the lock's one VCS line: pip clones it and
# runs upstream's `pip install -e .`, without build isolation as upstream's scripts/setup/setup.sh does; the checkout's .git directory is removed
# once installed — stock/check_pins.py verifies the checkout byte for byte against the pinned archive), then this kit and the shared core
# (`run.sh install`: pip, then the pin check). The image is upstream's GENERATION stack: the evaluation tools upstream's scripts/setup/setup.sh adds
# around it (the inverse-folding, refolding, structure-search and scoring packages under <checkout>/packages/ and their databases) are not
# installed — add them with that script in a derived image when a request enables sequence / side-chain prediction or you run upstream's evaluation. Weights are NOT in the image:
# `run.sh install --weights DIR` fetches upstream's pretrained/ tree into DIR with upstream's own downloader and checks the two files the kit
# reads against stock/PINS.json; GENIE3_WEIGHTS then names DIR/pretrained/v1 (README.md Setup).
# ENV: PATH puts the venv first and GENIE3_ROOT names the checkout, as on the tested image; PYTHONHASHSEED=0 and CFLAGS=-g0 are the values every
# tested run inherited from its container runtime, made explicit so other runtimes match; LD_LIBRARY_PATH stays the base image's own (torch
# resolves its CUDA libraries from its own wheels by RPATH either way). Thread-pool sizes are left to the host.
# Build context = the directory holding genie3/ and common/ (this file's grandparent: the kit imports ../common/opt_core):
#   docker build -f genie3/environment/Dockerfile -t genie3-kit:dev .    (run from the directory holding genie3/ and common/)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git build-essential && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.10.13+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "ba512bcca3ac6cb6d834f496cd0a66416f0a53ff20b05c4794fa82ece185b85a  /tmp/python.tar.gz" | sha256sum -c - \
 && mkdir /tmp/py && tar -xzf /tmp/python.tar.gz -C /tmp/py && cp -RL /tmp/py/python/. /usr/local/ && rm -rf /tmp/py /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python3 -V

RUN /usr/local/bin/python3 -m venv /opt/venv_g3
ENV PATH=/opt/venv_g3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

COPY genie3/environment/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir "$(grep '^pip==' /tmp/requirements.lock)" \
 && grep -v -E '^(#|-e )' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps --no-build-isolation --src /opt $(grep '^-e ' /tmp/requirements.lock) \
 && python -m pip check \
 && git -C /opt/genie3 log -1 --format='genie3 checkout %H' && rm -rf /opt/genie3/.git \
 && rm -f /tmp/stack.txt /tmp/requirements.lock

COPY common/opt_core /kit/common/opt_core
COPY genie3 /kit/genie3
WORKDIR /kit/genie3
ENV GENIE3_ROOT=/opt/genie3
RUN bash run.sh install
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY genie3/environment/Dockerfile _jitcache/genie3-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0
