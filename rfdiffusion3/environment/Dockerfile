# RFdiffusion3 optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.8.1 runtime (public NVIDIA base: the stack was assembled on that image line, and nothing in it is compiled at build
# time — torch brings its own CUDA 13.0 user-space libraries, apex arrives as a built wheel — so the runtime flavour carries everything it loads) + gcc 11.4
# (build-essential: Triton and torch.compile build kernels and launchers at first use), Python 3.12.1 at /usr/local (python-build-standalone
# 20240107, the x86_64_v3 build), and the kit's two-interpreter layout (STOCK.md 'Stack'): two identical virtual environments off
# that Python, /opt/stock and /opt/kit, each holding the pinned stack of rfdiffusion3/environment/requirements.lock (torch 2.13.0+cu130 with the
# CUDA 13.0 / cuDNN 9.20 / NCCL 2.29 runtime wheels, triton 3.7.1, numpy 2.4.4, lightning 2.6.5, atomworks 2.2.1), stock RFdiffusion3 (rc-foundry
# 0.2.1.dev13+g4010e3e2e, built from the source archive in rfdiffusion3/stock/) and NVIDIA apex 0.1 (no public wheel exists for this torch: compiled during the
# build from NVIDIA's repository at the commit stock/PINS.json pins — the default — or, with --build-arg WHEELS_FROM=prebuilt, the published wheel placed in
# rfdiffusion3/stock/wheels/ first: README.md 'Setup'), then this kit and its shared core, editable, in both (`run.sh install`).
# The kit's ten-file overlay is baked into /opt/kit, the environment that runs `exact` / `fast`, from the kit tree at /kit/rfdiffusion3
# (`python -m rfdiffusion3_opt.install apply`, then `check-only`, at build); /opt/stock holds the PRISTINE upstream package and runs `--mode off`.
# `install apply` run again inside a container meets the applied files and changes nothing.
# One image serves H100 and A100: nothing is compiled for a GPU at build time. No weights are inside: `run.sh install --weights DIR` fetches upstream's
# checkpoint into a directory you mount, and RFD3_CKPT names the file at run time (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 13.0
# (580 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding rfdiffusion3/ and common/) (apex compiles from source):
#   docker build -f rfdiffusion3/environment/Dockerfile -t rfdiffusion3-kit:dev .                                   # add --build-arg BUILD_JOBS=<n> to cap the compile jobs
#   docker build -f rfdiffusion3/environment/Dockerfile -t rfdiffusion3-kit:dev --build-arg WHEELS_FROM=prebuilt .   # a wheel already in rfdiffusion3/stock/wheels/ instead (minutes; name and sha256 as in stock/PINS.json)
# Run (README.md 'Run' has one command per mode; the design spec and the outputs live in the mounted directory):
#   docker run --rm --gpus all -v /path/to/weights:/weights -e RFD3_CKPT=/weights/rfd3_latest.ckpt -v "$PWD":/work rfdiffusion3-kit:dev \
#     bash run.sh design --config h100 --mode fast inputs=/work/my_design.json out_dir=/work/out/fast n_batches=1 seed=101
# To try the kit in your own environment instead: Python 3.12, two interpreters built alike from the lock as STOCK.md 'Stack' shows (at minimum
# torch 2.13.0 with CUDA 13.0, triton 3.7.1, rc-foundry at the pin, apex 0.1 with its CUDA extensions), a C compiler on PATH, then `run.sh install` in each
# and `python -m rfdiffusion3_opt.install apply` in the one that runs the kit modes.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ARG WHEELS_FROM=build
ARG BUILD_JOBS=0

# The C toolchain Triton and torch.compile need at run time (gcc, libc headers) and the download tools of this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential && rm -rf /var/lib/apt/lists/*

# Python 3.12.1: the python-build-standalone release this stack was assembled with (the x86_64_v3 build), unpacked into /usr/local with the archive's
# internal links (python3, libpython3.12.so, …) materialised as files, `python` linked to python3 (no system Python is used).
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.12.1+20240107-x86_64_v3-unknown-linux-gnu-install_only.tar.gz" \
 && echo "a7bcc6c9f66dbd47ea99615f30f101c5d2dd0084ca333d2f7336e64050951338  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -RLp /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The two interpreters, identical (STOCK.md 'Stack'), each a virtual environment off that Python: the environment's pip at the
# lock's version first, then the pinned stack, every distribution at the lock's version (--no-deps: the lock is complete), then the two distributions
# that come from rfdiffusion3/stock/ instead of an index (--no-deps; filtered out of the lock pass by name): stock RFdiffusion3, built from the source
# archive at the version stock/PINS.json pins (the archive carries no git metadata, so the build reads the version from
# SETUPTOOLS_SCM_PRETEND_VERSION; the build backends are held at the releases that produce this stack's installed files), and the
# apex wheel, by route (build argument WHEELS_FROM) — `build` (the default; self-contained): compiled here from NVIDIA's repository at the pinned commit
# against the torch just installed, with the CUDA 13.0 compiler and the -dev headers its extensions include taken from the NVIDIA apt repository the base
# image configures and ninja as the build tool outside the environments, all removed again once the wheel exists (C++ and CUDA extensions, the A100 and
# H100 cards: TORCH_CUDA_ARCH_LIST 8.0;9.0+PTX; BUILD_JOBS compile jobs, default every core — the wheel's digest is printed: a wheel compiled here is
# this build's own, with its own digest); `prebuilt`: a wheel placed in rfdiffusion3/stock/wheels/ beforehand (none is distributed with this tree), refused unless its name and
# sha256 are stock/PINS.json's — then installed into both environments (--no-deps); last, one import of each proves the environment loads them (apex's
# extension modules resolve the CUDA runtime through torch — no GPU is needed for that).
# The apex source archive is refused unless its sha256 is the digest written below (the digest belongs to the pinned commit: another commit in
# stock/PINS.json needs its own digest here).
COPY rfdiffusion3/environment/requirements.lock /tmp/requirements.lock
COPY rfdiffusion3/stock /tmp/stock
RUN grep -v -E '^(#|apex==|rc-foundry==)' /tmp/requirements.lock > /tmp/stack.txt \
 && printf 'hatchling==1.32.0\nhatch-vcs==0.4.0\nsetuptools-scm==10.2.1\nvcs-versioning==2.3.1\n' > /tmp/build-pins.txt \
 && eval "$(python -c "import json; p = json.load(open('/tmp/stock/PINS.json')); a = p['pinned_stack']['apex']; print('FOUNDRY_VERSION=%s APEX_WHEEL=%s APEX_SHA256=%s APEX_REPO=%s APEX_COMMIT=%s' % (p['upstream']['foundry']['version'], a['wheel'], a['wheel_sha256'], a['repo'], a['commit']))")" \
 && for venv in /opt/stock /opt/kit; do \
      /usr/local/bin/python -m venv "$venv" \
   && "$venv/bin/python" -m pip install --no-cache-dir --no-deps "$(grep -E '^pip==' /tmp/stack.txt)" \
   && "$venv/bin/python" -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
   && SETUPTOOLS_SCM_PRETEND_VERSION="$FOUNDRY_VERSION" PIP_CONSTRAINT=/tmp/build-pins.txt "$venv/bin/python" -m pip install --no-cache-dir --no-deps /tmp/stock/foundry-4010e3e.tar.gz \
   || exit 1; \
    done \
 && if [ "$WHEELS_FROM" = build ]; then \
      J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); P=$(( J < 8 ? J : 8 )); MJ=$(( J / P )); [ "$MJ" -gt 0 ] || MJ=1; \
      echo "apex $APEX_COMMIT from source: $(nproc --all) cores, BUILD_JOBS=$J ($P extensions at a time, $MJ ninja jobs each, nvcc --threads 4), TORCH_CUDA_ARCH_LIST=8.0;9.0+PTX" \
   && apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-nvrtc-dev-13-0 cuda-profiler-api-13-0 cuda-nvml-dev-13-0 \
        libcublas-dev-13-0 libcurand-dev-13-0 libcusparse-dev-13-0 libcusolver-dev-13-0 \
   && /usr/local/cuda-13.0/bin/nvcc --version | tail -1 \
   && curl -fsSL -o /tmp/apex.tar.gz "$APEX_REPO/archive/$APEX_COMMIT.tar.gz" && echo "f3cc6e36d357f5c47e39fe6f7ab7282ec7fc19af0b4a8425758ec2f2c6cc038e  /tmp/apex.tar.gz" | sha256sum -c - \
   && mkdir -p /tmp/src /tmp/stock/wheels && tar -xzf /tmp/apex.tar.gz -C /tmp/src \
   && /opt/stock/bin/python -m pip install --no-cache-dir --target /tmp/buildtools ninja==1.13.0 \
   && cd "/tmp/src/apex-$APEX_COMMIT" \
   && PYTHONPATH=/tmp/buildtools PATH=/tmp/buildtools/bin:/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0 CC=gcc CXX=g++ \
      APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_PARALLEL_BUILD=$P MAX_JOBS=$MJ CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_APPEND_FLAGS="--threads 4" TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" \
      /opt/stock/bin/python -m pip wheel --no-deps --no-build-isolation --no-cache-dir -w /tmp/stock/wheels . \
   && cd / && echo "apex wheel compiled here (this build's own file, not the published one): $(sha256sum "/tmp/stock/wheels/$APEX_WHEEL")" \
   && apt-get purge -y --auto-remove cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-nvrtc-dev-13-0 cuda-profiler-api-13-0 cuda-nvml-dev-13-0 \
        libcublas-dev-13-0 libcurand-dev-13-0 libcusparse-dev-13-0 libcusolver-dev-13-0 \
   && rm -rf /var/lib/apt/lists/* /tmp/src /tmp/buildtools /tmp/apex.tar.gz || exit 1; \
    elif [ "$WHEELS_FROM" = prebuilt ]; then echo "$APEX_SHA256  /tmp/stock/wheels/$APEX_WHEEL" | sha256sum -c - || exit 1; \
    else echo "WHEELS_FROM=$WHEELS_FROM: build or prebuilt" >&2; exit 2; fi \
 && for venv in /opt/stock /opt/kit; do \
      "$venv/bin/python" -m pip install --no-cache-dir --no-deps "/tmp/stock/wheels/$APEX_WHEEL" \
   && "$venv/bin/python" -c "import rfd3, apex, importlib; importlib.import_module('apex.normalization.fused_layer_norm'); import fused_layer_norm_cuda" \
   || exit 1; \
    done \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/build-pins.txt /tmp/requirements.lock /root/.cache

# The kit: the shared core, then rfdiffusion3_opt, both editable, in BOTH interpreters, each install checked against the pin — the one install step,
# `run.sh install`, once per interpreter (README.md 'Setup') — then the ten-file overlay into /opt/kit ONLY (`install apply` copies the patched
# rfd3/model files from this kit tree, `check-only` confirms them byte for byte), so the image and any Apptainer / Singularity image converted
# from it run `exact` / `fast` as shipped; /opt/stock stays pristine for `--mode off`.
COPY common/opt_core /kit/common/opt_core
COPY rfdiffusion3 /kit/rfdiffusion3
WORKDIR /kit/rfdiffusion3
RUN PATH=/opt/stock/bin:$PATH bash run.sh install \
 && PATH=/opt/kit/bin:$PATH bash run.sh install \
 && rm -rf /root/.cache
RUN cd /kit/rfdiffusion3 && /opt/kit/bin/python -m rfdiffusion3_opt.install apply && /opt/kit/bin/python -m rfdiffusion3_opt.install check-only
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY rfdiffusion3/environment/Dockerfile _jitcache/rfdiffusion3-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment of the tested stack: hash seed fixed, unbuffered output, no debug info in the launchers Triton compiles, the loader path
# naming the base image's CUDA runtime directory (the value the stack has always run with; torch itself loads its CUDA libraries from the lock's
# nvidia wheels by path, so the entry changes nothing it maps); the kit interpreter's python first on PATH (it runs `exact` and `fast`; the rest of
# the list is the base image's own) and the pristine interpreter named for `--mode off` (STOCK.md § Variables). The weights path is never baked in:
# RFD3_CKPT is set at run time.
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    CFLAGS=-g0 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    PATH=/opt/kit/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    RFDIFFUSION3_STOCK_PYTHON=/opt/stock/bin/python
