# ProteinMPNN optimization kit — the tested stack as a container image: torch 2.5.1+cu124, numpy 1.26.4, biopython 1.84
# (environment/requirements-mpnn.lock; both weight sets, --variant soluble | vanilla, run on it). Base: Ubuntu 22.04 + the CUDA 12.4.1 runtime
# (public NVIDIA image; the runtime flavour because nothing here compiles CUDA code — torch loads its CUDA libraries from the lock's own nvidia
# wheels) + gcc (build-essential: Triton builds the exact line's draw kernel and its launcher at first use) + git (upstream's own distribution
# tool, used by `run.sh install --weights`: the clone, and the .fa header's commit), Python 3.11.5
# (python-build-standalone) at /usr/local/bin/python, the pinned stack of the lock, and this kit with its shared core, editable, under /kit.
# The image holds no ProteinMPNN checkout and no weights (upstream ships code and weights as one git repository): MPNN_DIR=/opt/ProteinMPNN
# names a directory you mount, filled once by `run.sh install --weights /opt/ProteinMPNN` (the clone at the pinned commit, digest-checked).
# One image serves H100 and A100: nothing is compiled for a GPU at build time.
# Host: an NVIDIA driver that runs CUDA 12.4 (550 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding proteinmpnn/ and common/):
#   docker build -f proteinmpnn/environment/Dockerfile -t proteinmpnn-kit:dev .
# Run, with a writable host directory mounted at /opt/ProteinMPNN (README.md 'Run' has one command per mode; outputs under the mounted out/):
#   docker run --rm -v "$PWD/ProteinMPNN":/opt/ProteinMPNN proteinmpnn-kit:dev bash run.sh install --weights /opt/ProteinMPNN   # once: the clone
#   docker run --rm --gpus all -v "$PWD/ProteinMPNN":/opt/ProteinMPNN -v "$PWD/out":/kit/proteinmpnn/out proteinmpnn-kit:dev \
#     bash run.sh design --config h100 --mode exact --input /opt/ProteinMPNN/inputs/PDB_monomers/pdbs --out out/exact --seed 37
# To try the kit in your own environment instead: Python 3.11, the lock installed as README.md 'Setup' route C shows (at minimum torch
# 2.5.1 with CUDA 12.4 and numpy 1.26.x), git and a C compiler (gcc) on PATH, then `run.sh install`.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# The process environment of the tested stack that does not depend on the machine: the hash seed. The library search path stays the base image's
# (LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64; torch loads the lock's nvidia wheel libraries by their own path).
ENV PYTHONHASHSEED=0

# The C toolchain Triton needs at run time (gcc, libc headers), upstream's distribution tool (git: the ProteinMPNN clone and the .fa header's
# commit) and the download tools of this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git build-essential && rm -rf /var/lib/apt/lists/*

# Python 3.11.5: the python-build-standalone distribution (the interpreter build of the tested stack), unpacked over /usr/local with its internal
# links (bin/python3, bin/2to3, lib/libpython3.11.so, …) materialised as files, the layout of the tested stack.
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -RL /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack: every line of the lock, nothing resolved beyond it.
COPY proteinmpnn/environment /tmp/environment
RUN python -m pip install --no-cache-dir --no-deps -r /tmp/environment/requirements-mpnn.lock && rm -rf /tmp/environment

# The kit and its shared core, editable, from this build context (the release bundle's proteinmpnn/ and common/), then the core pin gate —
# the code-only part of `run.sh install`: its last step, the stock pin check, needs the ProteinMPNN clone, which is never put in the image.
COPY common/opt_core /kit/common/opt_core
COPY proteinmpnn /kit/proteinmpnn
WORKDIR /kit/proteinmpnn
RUN python -m pip install --no-cache-dir -e /kit/common/opt_core -e /kit/proteinmpnn/opt && python -c "from proteinmpnn_opt import core_gate; core_gate()"

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY proteinmpnn/environment/Dockerfile _jitcache/proteinmpnn-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# Where every verb looks for the ProteinMPNN clone (code, .git, both weight sets): the mount point of the directory `run.sh install --weights` fills.
ENV MPNN_DIR=/opt/ProteinMPNN
