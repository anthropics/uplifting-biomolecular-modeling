# ColabFold optimization kit — the tested stack as a container image.
# Base: NVIDIA's public CUDA 12.4.1 runtime image on Ubuntu 22.04 — the base of the stack this kit was tested on, at runtime weight: the model's CUDA
# libraries and ptxas come from the pinned nvidia-*-cu12 wheels and nothing is compiled at build or run time, so neither the devel flavour nor a C
# compiler is needed (the tested stack has none). Then Python 3.11.5 at /usr/local/bin/python, the pinned stack of colabfold/environment/requirements.lock
# (jax / jaxlib with the CUDA 12 plugin and NVIDIA's runtime, cuDNN and NCCL wheels, dm-haiku, numpy, tensorflow-cpu — every version is the lock's),
# stock colabfold + alphafold-colabfold from the two wheels in colabfold/stock/ (stock/PINS.json names them), then this kit and its shared core,
# editable, under /kit (`run.sh install`). One image serves H100 and A100: nothing is compiled for a GPU at build time. No AlphaFold parameters are inside:
# `run.sh install --weights DIR` fetches them into a directory you mount, and COLABFOLD_OPT_DATA_DIR names it at run time (README.md 'Setup').
# Host: an NVIDIA driver that runs CUDA 12 (550 or newer) and the NVIDIA Container Toolkit.
# Build, from the bundle's top directory (the one holding colabfold/ and common/):
#   docker build -f colabfold/environment/Dockerfile -t colabfold-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/af2_params:/weights -e COLABFOLD_OPT_DATA_DIR=/weights -v "$PWD/out":/kit/colabfold/out colabfold-kit:dev \
#     bash run.sh pred --config h100 --mode fast tests/inputs/1BRS_AD.a3m out/fast
# To try the kit in your own environment instead: Python 3.11, the lock installed with --no-deps as below, the two wheels from
# colabfold/stock/, then `run.sh install` (README.md 'Setup').
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# The download tools of this build, and the two alignment programs stock colabfold_batch runs when a query carries templates (`--templates`,
# `--custom-template-path`: `hhsearch` from hhsuite and `kalign`, colabfold/batch.py:143-153) — Ubuntu 22.04's packages (hhsuite 3.3.0, kalign 3.3).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl hhsuite kalign && rm -rf /var/lib/apt/lists/*

# Python 3.11.5: the python-build-standalone release this stack was assembled with (its bundled pip and setuptools are the stack's), copied into
# /usr/local with the archive's internal links (python3, libpython3.11.so, …) materialized as files, as the tested stack has them (no system Python is used).
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -RL --preserve=mode,timestamps /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete), then stock from the bundled wheels (--no-deps: their
# dependencies are the lock); `pip check` confirms every declared requirement holds on this stack.
COPY colabfold/environment/requirements.lock /tmp/requirements.lock
COPY colabfold/stock/*.whl /tmp/stock/
RUN grep -v -E '^(#|colabfold==|alphafold-colabfold==)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/*.whl \
 && python -m pip check && rm -rf /tmp/stock /tmp/stack.txt /tmp/requirements.lock

# The kit: the shared core, then colabfold_opt, both editable, and the install checked against the pin — the one install step, `run.sh install` (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY colabfold /kit/colabfold
WORKDIR /kit/colabfold
RUN bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY colabfold/environment/Dockerfile _jitcache/colabfold-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
# JAX_COMPILATION_CACHE_DIR=/root/.cache/jax below is the persistent compilation cache of the image (read by stock and by every kit
# mode). A shipped cache is never linked here at build time — an image can carry several cards' trees and the build cannot know the GPU:
# run.sh points jax at /opt/jit_cache/<stack key of the chosen config>/default/jax at run time (in place when writable, else the seeded copy).

# The process environment of the tested stack (STOCK.md 'pinned stack'): jax on the CUDA backend, 95 % of device memory preallocated with unified memory
# off (colabfold sets 1 / 4.0 only where these are unset, and that setting does not progress on an H100 80 GB), JAX's persistent compilation cache
# under /root/.cache/jax (a kit mode keeps a JAX_COMPILATION_CACHE_DIR already set; unset it and configs/<gpu>.env's COLABFOLD_OPT_JIT_ROOT names the
# root instead — mount a persistent directory at either to keep compiled executables across containers), hash seed fixed, CFLAGS as every tested
# run carried it; the loader path is the base image's own. The parameters root is never baked in: COLABFOLD_OPT_DATA_DIR is set at run time.
ENV JAX_PLATFORMS=cuda \
    JAX_COMPILATION_CACHE_DIR=/root/.cache/jax \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    TF_FORCE_UNIFIED_MEMORY=0 \
    PYTHONHASHSEED=0 \
    CFLAGS=-g0
