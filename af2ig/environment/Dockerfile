# AF2 initial guess optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.4.1 runtime (public NVIDIA base) + gcc 11.4 (build-essential) and GNU patch, Python 3.11.5 at /usr/local/bin/python,
# the pinned stack of af2ig/environment/requirements.lock (jax / jaxlib 0.5.3 with the CUDA-12 plugin and the CUDA 12.9 / cuDNN 9.23 / NCCL 2.30 runtime
# wheels, dm-haiku 0.0.16, tensorflow-cpu 2.21.0, numpy 1.26.4), then this kit and its shared core, editable, under /kit, and the pinned upstream tree
# (dl_binder_design @ cafa3853 from the archive in af2ig/stock/) unpacked and patched at /kit/af2ig/dl_binder_design (`run.sh install`). The runtime
# base, not the devel one: nothing here compiles CUDA code at build or install time — XLA compiles the model's kernels at run time with the ptxas that
# jax's CUDA toolchain wheel ships (requirements.lock), and the CUDA libraries the jax plugin loads are the lock's nvidia-*-cu12 wheels, found beside it in
# site-packages. One image serves H100 and A100: nothing is compiled for a GPU at build time. No weights are inside: `run.sh install --weights DIR` fetches
# the one AlphaFold-2 parameter file into a directory you mount, and AF2_PARAMS names it at run time (README.md 'Setup'). AF2IG_DIR is set to the patched
# tree built here.
# Host: an NVIDIA driver that runs CUDA 12 (525.60.13 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding af2ig/ and common/):
#   docker build -f af2ig/environment/Dockerfile -t af2ig-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/af2_params:/weights -e AF2_PARAMS=/weights -v "$PWD/out":/kit/af2ig/out af2ig-kit:dev \
#     bash run.sh pred --config h100 --mode fast --pdbdir opt/forward/af2ig_kit/tests/inputs/pdbs --out out/fast
# To try the kit in your own environment instead: Python 3.11 on Linux x86_64, the lock installed with
#   grep -v '^#' af2ig/environment/requirements.lock | python -m pip install --no-deps -r /dev/stdin
# (at minimum jax[cuda12] 0.5.3, dm-haiku 0.0.16, tensorflow-cpu 2.21.0, numpy 1.26.4 — stock/PINS.json `pins`), GNU patch and tar on PATH, then `run.sh install`.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# The C toolchain, GNU patch (run.sh install applies the kit's patch series to the upstream tree) and the download tools of this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential patch && rm -rf /var/lib/apt/lists/*

# Python 3.11.5: the python-build-standalone release this stack was assembled with, unpacked into /usr/local (no system Python is used).
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete; `pip check` reports no broken requirement). Nothing is
# compiled anywhere in this file, so there is no build-parallelism setting; the step prints the build host's core count for the log.
COPY af2ig/environment/requirements.lock /tmp/requirements.lock
RUN echo "build host: $(nproc --all) cores" \
 && grep -v -E '^#' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && rm -f /tmp/stack.txt /tmp/requirements.lock

# The kit: the shared core, then af2ig_opt, both editable; the pinned upstream tree unpacked from af2ig/stock/ and patched; the install checked against
# the pin — the one install step, `run.sh install` (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY af2ig /kit/af2ig
WORKDIR /kit/af2ig
RUN bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY af2ig/environment/Dockerfile _jitcache/af2ig-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment of the tested stack (STOCK.md 'pinned stack'): hash seed fixed, no .pyc written into the tree, no debug info from any C build.
# AF2IG_DIR names the patched tree of this image; the weights root is never baked in: AF2_PARAMS is set at run time.
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    CFLAGS=-g0 \
    AF2IG_DIR=/kit/af2ig/dl_binder_design/af2_initial_guess
