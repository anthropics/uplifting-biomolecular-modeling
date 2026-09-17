# AF3-torch optimization kit (xfold on the converted OpenFold3-preview2 weights) — the tested stack as one container image.
# Builds Ubuntu 24.04 on NVIDIA's CUDA 12.6.3 *base* image (the base the tested stack was assembled on: it contributes libcudart 12.6 and the
# compat directory only — torch loads its CUDA 13.0 libraries from the wheels of the lock by path, and no CUDA source is compiled at build or
# run time, so neither the runtime nor the devel flavour is needed) + gcc 13.3 (build-essential: Triton builds the kit's kernels and their
# launchers at first use), CPython 3.12.1 (python-build-standalone 20240107, the x86-64-v3 build) at /usr/local with uv 0.9.24, and the kit's TWO interpreters as
# uv virtual environments of that Python: /torch_venv = af3_torch/environment/requirements-torch.lock (torch 2.13.0+cu130, triton 3.7.1, numpy
# 2.4.1: the model process, AF3_TORCH_PY) and /alphafold3_venv = af3_torch/environment/requirements-jax.lock + alphafold3-open 3.1.4 (jax 0.10.2 on
# CPU: AlphaFold 3's featurisation and output writers, AF3_TORCH_JAX_PY) — the sokrypton/alphafold3 fork at its pinned commit, fetched from
# GitHub into /app/alphafold (AF3_TORCH_JAX_REPO: run_alphafold.py, the writers' entry) with the OpenFold3 empty-template convention applied
# (af3_torch/stock/patches/04_of3_empty_template_restype_gap.diff, STOCK.md) and BUILT HERE into a wheel (one C++ extension: scikit-build-core,
# cmake, pybind11 — minutes on all cores, ARG BUILD_JOBS caps the jobs), then AlphaFold 3's `build_data` step (the chemical-components data
# inside the installed package). `--build-arg WHEELS_FROM=prebuilt` skips the compile and the data step: it takes the tested stack's own wheel and
# data files from af3_torch/stock/wheels/ instead (two files, not in git, no public URL; stock/PINS.json `reference.artefacts` pins their sha256
# and this build checks them; STOCK.md 'Stack' says where they come from) — the extension's build fetches the wwPDB chemical-components
# dictionary current at build time and the data step's pickles are not byte-stable across runs, so only that route reproduces the tested stack's
# files byte for byte; the default route differs in that dictionary, the data files made from it and the compiled extension. Then this kit and its
# shared core, editable, in /kit/venv (first on PATH: `run.sh install`). One image serves H100 and A100: nothing is compiled for a GPU at build
# time. No weights are inside: the converted checkpoint directory is mounted and AF3_TORCH_PARAMS_DIR names it at run time (README.md
# 'Setup'); `run.sh install --weights DIR` checks it. Host: an NVIDIA driver that runs CUDA 13.0 (580 or newer) and the NVIDIA container toolkit.
# Build, from the directory that holds af3_torch/ and common/ (the bundle's top directory):
#   docker build -f af3_torch/environment/Dockerfile -t af3_torch-kit:dev .                                   # from source and public URLs only
#   docker build -f af3_torch/environment/Dockerfile -t af3_torch-kit:dev --build-arg WHEELS_FROM=prebuilt .  # with the two files in af3_torch/stock/wheels/
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/params:/params -e AF3_TORCH_PARAMS_DIR=/params -v "$PWD/out":/kit/af3_torch/out af3_torch-kit:dev \
#     bash run.sh pred --config h100 --mode fast --json_path inputs/1BRS.json --output_dir out/fast
# To try the kit in your own environment instead: two Python 3.12 environments installed from the two locks as STOCK.md 'Stack' describes
# shows (at minimum torch 2.13.0 with CUDA 13.0 + triton 3.7.1 in one, jax/jaxlib 0.10.2 + the alphafold3-open wheel in the other), the fork
# tree cloned somewhere, a C compiler on PATH, the three AF3_TORCH_* variables below pointing at them, then `run.sh install` on any Python >= 3.10.
FROM nvidia/cuda:12.6.3-base-ubuntu24.04
# WHEELS_FROM: build (default) = the fork's wheel and data are made here from source; prebuilt = taken from af3_torch/stock/wheels/ (sha256-checked).
# BUILD_JOBS: parallel compile jobs for that wheel; 0 = every core of the builder.
ARG WHEELS_FROM=build
ARG BUILD_JOBS=0

# The C toolchain Triton and Inductor need at run time and the fork's extension needs at build time (build-essential: gcc/g++ 13.3, make, libc
# headers; zlib headers for the extension), git (the fork and its extension's pinned C++ dependencies come from GitHub), zstd, patch, wget.
RUN apt-get update --quiet \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
      ca-certificates wget git build-essential zlib1g-dev zstd patch \
 && rm -rf /var/lib/apt/lists/*

# CPython 3.12.1: python-build-standalone's install_only build for x86-64-v3 (clang 17, PGO+LTO, shared libpython), unpacked at /usr/local — the
# interpreter both virtual environments below are made of; it carries pip 23.3.2 and setuptools 69.0.3. uv 0.9.24 makes and fills the environments.
RUN wget -q -O /tmp/python.tgz https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.12.1+20240107-x86_64_v3-unknown-linux-gnu-install_only.tar.gz \
 && echo "a7bcc6c9f66dbd47ea99615f30f101c5d2dd0084ca333d2f7336e64050951338  /tmp/python.tgz" | sha256sum -c - \
 && tar xzf /tmp/python.tgz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tgz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python \
 && /usr/local/bin/python3 -m pip install --no-cache-dir uv==0.9.24

# /torch_venv — the model interpreter: every distribution of requirements-torch.lock at the lock's version (--no-deps: the lock is complete).
COPY af3_torch/environment/requirements-torch.lock af3_torch/environment/requirements-jax.lock /tmp/env/
RUN uv venv --python /usr/local/bin/python3.12 /torch_venv \
 && grep -v '^#' /tmp/env/requirements-torch.lock > /tmp/env/torch.txt \
 && uv pip install --no-cache --python /torch_venv/bin/python --no-deps -r /tmp/env/torch.txt

# /alphafold3_venv — the featurisation/writer interpreter: requirements-jax.lock (torch 2.5.1+cpu by its line's URL on download.pytorch.org, the rest from
# PyPI; the alphafold3-open line is left to the steps below), then the fork: its source tree at the pinned commit from GitHub with the patch applied
# (/app/alphafold; COMMIT names the commit, .git is dropped), its wheel — compiled here from that tree with the fork's own build backend (its build
# requirements come from PyPI, numpy held at the lock's version; CC/CXX name the image's gcc — the interpreter's own build records clang), or, under WHEELS_FROM=prebuilt, the tested stack's wheel from af3_torch/stock/wheels/
# after its sha256 is checked against stock/PINS.json — installed --no-deps (its dependencies are the lock's), and the chemical-components data
# AlphaFold 3's `build_data` step writes into the installed package (alphafold3/constants/converters/ccd.pickle, ccd_codes.txt,
# chemical_component_sets.pickle): generated here by that step, or, under prebuilt, the tested stack's own files unpacked over the package.
# af3_torch/stock/wheels/ holds only .keep in git; the prebuilt route's two files are placed there first (STOCK.md 'Stack').
COPY af3_torch/stock/PINS.json af3_torch/stock/patches/04_of3_empty_template_restype_gap.diff /tmp/stock/
COPY af3_torch/stock/wheels/ /tmp/wheels/
RUN uv venv --python /usr/local/bin/python3.12 --prompt alphafold3-open /alphafold3_venv \
 && grep -v -E '^(#|alphafold3-open==)' /tmp/env/requirements-jax.lock > /tmp/env/jax.txt \
 && uv pip install --no-cache --python /alphafold3_venv/bin/python --no-deps -r /tmp/env/jax.txt \
 && git clone --quiet https://github.com/sokrypton/alphafold3 /app/alphafold && cd /app/alphafold \
 && git checkout --quiet bc32b22ff5902e3daffd5d1f7203d7f2ab6cb997 && git rev-parse HEAD > COMMIT \
 && [ "$(cat COMMIT)" = bc32b22ff5902e3daffd5d1f7203d7f2ab6cb997 ] && rm -rf .git \
 && patch -p1 < /tmp/stock/04_of3_empty_template_restype_gap.diff \
 && { J="${BUILD_JOBS:-0}"; [ "$J" -gt 0 ] || J=$(nproc --all); } \
 && echo "af3_torch environment: WHEELS_FROM=$WHEELS_FROM, $J build jobs on a $(nproc --all)-core builder" \
 && if [ "$WHEELS_FROM" = build ]; then \
      export CC=gcc CXX=g++ MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J \
      && grep -E '^numpy==' /tmp/env/jax.txt > /tmp/env/build-constraints.txt \
      && uv build --wheel --no-cache --python /usr/local/bin/python3.12 --build-constraint /tmp/env/build-constraints.txt --out-dir /tmp/wheels /app/alphafold \
      && rm -rf /app/alphafold/dist /app/alphafold/build; \
    elif [ "$WHEELS_FROM" = prebuilt ]; then \
      /usr/local/bin/python3 -c 'import hashlib, json, os, sys; arts = json.load(open("/tmp/stock/PINS.json"))["reference"]["artefacts"]["files"]; bad = [a["file"] for a in arts if not os.path.isfile("/tmp/wheels/" + a["file"]) or hashlib.sha256(open("/tmp/wheels/" + a["file"], "rb").read()).hexdigest() != a["sha256"]]; print("af3_torch/stock/wheels: " + (("MISSING OR NOT THE PINNED BYTES: " + ", ".join(bad)) if bad else "%d files match stock/PINS.json reference.artefacts" % len(arts))); sys.exit(65 if bad else 0)'; \
    else echo "WHEELS_FROM must be build or prebuilt, not '$WHEELS_FROM'"; exit 64; fi \
 && uv pip install --no-cache --python /alphafold3_venv/bin/python --no-deps /tmp/wheels/alphafold3_open-3.1.4-cp312-cp312-linux_x86_64.whl \
 && if [ "$WHEELS_FROM" = build ]; then /alphafold3_venv/bin/build_data; \
    else tar xzf /tmp/wheels/alphafold3-build_data.tar.gz -C /alphafold3_venv/lib/python3.12/site-packages; fi \
 && ls -l /alphafold3_venv/lib/python3.12/site-packages/alphafold3/constants/converters/ccd.pickle \
 && rm -rf /tmp/wheels /tmp/stock /tmp/env /root/.cache

# The process environment of the tested stack. PATH: the kit's environment first (`python` = the kit; it drives the two interpreters above by
# their variables), then the featurisation environment, as the stack had it. AF3_TORCH_PY / AF3_TORCH_JAX_PY / AF3_TORCH_JAX_REPO: where this
# image put the two interpreters and the fork tree (STOCK.md 'Variables'; AF3_TORCH_PARAMS_DIR is yours to set at run time). The three XLA
# variables are the stack's image settings (the same values the kit's JAX steps set per process, STOCK.md; those steps run on CPU). LD_LIBRARY_PATH
# stays the base image's (/usr/local/nvidia/lib:/usr/local/nvidia/lib64): torch loads its own CUDA 13 libraries by path. PYTHONHASHSEED=0 and
# CFLAGS=-g0 were set in every run of the tested stack.
ENV PATH="/kit/venv/bin:/alphafold3_venv/bin:${PATH}" \
    AF3_TORCH_PY=/torch_venv/bin/python \
    AF3_TORCH_JAX_PY=/alphafold3_venv/bin/python \
    AF3_TORCH_JAX_REPO=/app/alphafold \
    XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_CLIENT_MEM_FRACTION=0.95 \
    PYTHONHASHSEED=0 \
    CFLAGS=-g0

# This kit and its shared core under /kit (the kit tree as it is), installed editable into an environment of their own (/kit/venv, pip inside) by the
# kit's own install step (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY af3_torch /kit/af3_torch
WORKDIR /kit/af3_torch
RUN uv venv --seed --python /usr/local/bin/python3.12 /kit/venv && bash run.sh install
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY af3_torch/environment/Dockerfile _jitcache/af3_torch-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
# af3_torch mapping: the kit's own seed home (opt/af3_torch_opt/stack.py: <prefix of AF3_TORCH_PY>/share/af3_torch_cache/<stack key>/{triton,inductor,jax,verdict},
# copied into the writable AF3_TORCH_CACHE_ROOT on a first pred / warm, Triton group manifests re-rooted) is /opt/jit_cache itself
RUN mkdir -p /torch_venv/share && ln -s /opt/jit_cache /torch_venv/share/af3_torch_cache
