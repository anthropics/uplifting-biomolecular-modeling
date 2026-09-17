# E1 optimization kit — the tested stack as a container image.
# Base: the official `python:3.12.10-slim-bookworm` image — Debian 12 with CPython 3.12.10 (gcc 12.2, --enable-shared --with-lto) in /usr/local —
# because that interpreter, byte for byte, is the one the stack was assembled and tested on; no CUDA base image is involved: torch 2.8.0+cu128 loads
# the CUDA 12.8 runtime, cuDNN 9.10 and NCCL 2.27 from the nvidia-*-cu12 wheels of e1/environment/requirements.lock, and the GPU driver is the host's.
# On top: build-essential (gcc 12: Triton builds the hub RMSNorm kernel, the kit's kernels and their launchers at first use; torch.compile builds the
# flex-attention code) and git (stock E1 installs from its repository at the pinned commit), the pinned stack of requirements.lock (torch 2.8.0+cu128,
# triton 3.4.0, transformers 4.56.2, kernels 0.11.0, the flash-attn 2.8.3.post1 release wheel, stock E1 1.0.0 @ bfd2620a), this kit editable under
# /kit (`run.sh install`; the package needs nothing from the bundle's shared core), and the hub RMSNorm kernel snapshot (kernels-community/triton-layer-norm at the revision pinned in
# stock/PINS.json, copied from the tree's stock/hub_kernel, no network) at /opt/kernels_cache, which KERNELS_CACHE names. One image serves H100 and H200
# (and the A100 configuration): nothing is compiled for a GPU at build time. No weights are inside: `run.sh install --weights DIR` stages the three
# checkpoints into a directory you mount, and HF_HOME names it at run time (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 12.8 (570 or
# newer) and the NVIDIA container toolkit. The build downloads wheels; nothing is compiled.
# Build, from the bundle's top directory (the one holding e1/):
#   docker build -f e1/environment/Dockerfile -t e1-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/hf_home:/weights -e HF_HOME=/weights -v "$PWD/out":/kit/e1/out e1-kit:dev \
#     bash run.sh score --config h100 --variant 600m --mode exact --parent-path opt/e1_opt/tests/fixtures/warm/parent.fasta \
#       --mutants-path opt/e1_opt/tests/fixtures/warm/mutants.fasta --output-path out/exact/scores.csv
# To try the kit in your own environment instead: Python 3.12, a C compiler and git on PATH, the lock installed as STOCK.md 'Stack' shows
# (at minimum torch 2.8.0 with CUDA 12.8, triton 3.4.0, transformers 4.56.2, kernels 0.11.0, flash-attn 2.8.3.post1 and stock E1 at commit bfd2620a),
# then `run.sh install --weights DIR`.
FROM python:3.12.10-slim-bookworm

# The C toolchain Triton and torch.compile need at run time, and git for the stock install of this build (the build host's core count is printed
# first: the build is downloads and one pure-Python wheel build, nothing is compiled, so its time is network-bound).
RUN echo "build host: $(nproc --all) cores" && apt-get update && apt-get install -y --no-install-recommends build-essential git ca-certificates && rm -rf /var/lib/apt/lists/*

# The process environment of the tested stack: hash seed fixed, launchers built without debug info, upstream's flash-attn route named explicitly
# (its default), no usage telemetry from the Hugging Face libraries, and the kernels package's cache root — the hub RMSNorm kernel is staged there below.
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    USE_FLASH_ATTN=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DISABLE_TELEMETRY=1 \
    KERNELS_CACHE=/opt/kernels_cache

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete): pip first, at the lock's own line, then the rest —
# stock E1 is built from its repository by hatchling, held at the release that built the tested stack's copy so the installed metadata is the same.
# pip runs as `python3.12 -m pip`: the console scripts it writes then name /usr/local/bin/python3.12, as the tested stack's do.
COPY e1/environment/requirements.lock /tmp/requirements.lock
RUN python3.12 -m pip install --no-cache-dir "$(grep -E '^pip==' /tmp/requirements.lock)" \
 && echo 'hatchling==1.32.0' > /tmp/build-constraints.txt \
 && PIP_CONSTRAINT=/tmp/build-constraints.txt python3.12 -m pip install --no-cache-dir --no-deps -r /tmp/requirements.lock \
 && rm -f /tmp/requirements.lock /tmp/build-constraints.txt

# The kit: e1_opt editable and the install checked against the software pins — the one install step, `run.sh install` (README.md 'Setup');
# then the hub RMSNorm kernel snapshot into KERNELS_CACHE by the same step's kernel form (the tree's copy under stock/hub_kernel staged into the
# cache layout the kernels package reads, no network; layer_norm.py checked against stock/PINS.json).
COPY e1 /kit/e1
WORKDIR /kit/e1
RUN bash run.sh install \
 && python -m e1_opt.weights --kernels-cache "$KERNELS_CACHE"

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY e1/environment/Dockerfile _jitcache/e1-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
# (the build context is the directory holding e1/: the always-present anchor is e1/environment/Dockerfile and the optional tar sits in a
#  staging directory `_jitcache/` at the context root, outside e1/, so `COPY e1 /kit/e1` above never carries it. run.sh seeds
#  MODEL_OPT_JIT_ROOT from /opt/jit_cache when the caches are present; configs/<card>.env then key the Triton and Inductor caches under it.)
