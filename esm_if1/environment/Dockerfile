# ESM-IF1 optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.1.1 runtime (public NVIDIA base) + gcc 11.4 (build-essential: the C toolchain of the kit images; nothing here
# compiles CUDA code — the PyG extensions and every other wheel of the lock are prebuilt, so the CUDA *devel* toolkit is not needed), Python 3.11.5
# at /usr/local/bin/python, the pinned stack of esm_if1/environment/requirements.lock (torch 2.4.0+cu121 with the CUDA 12.1 / cuDNN 9.1 / NCCL 2.20
# runtime wheels, triton 3.0.0, numpy 1.26.4, scipy 1.11.4, biotite 0.39.0, torch-geometric 2.6.1 + torch_scatter / torch_sparse / torch_cluster
# pt24cu121), stock fair-esm 2.0.1 from the archive in esm_if1/stock/, then this kit and its shared core, editable, under /kit (`run.sh install`).
# One image serves H100 and A100: nothing is compiled for a GPU at build time. No weights are inside: `run.sh install --weights DIR` fetches the
# one checkpoint file into a directory you mount, and ESM_IF1_WEIGHTS names it at run time (README.md 'Setup'). Host: an NVIDIA driver that runs
# CUDA 12.1 (530 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding esm_if1/ and common/):
#   docker build -f esm_if1/environment/Dockerfile -t esm_if1-kit:dev .
# Run (README.md 'Run'; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/weights:/weights -e ESM_IF1_WEIGHTS=/weights/esm_if1_gvp4_t16_142M_UR50.pt -v "$PWD/out":/kit/esm_if1/out \
#     esm_if1-kit:dev bash run.sh design --config h100 --chain C --num-samples 8 --input /tmp/examples/inverse_folding/data/5YH2.pdb --out out/one
#   (the example structure is upstream's, inside the image after `tar -xzf stock/fair-esm-2b369911.tar.gz -C /tmp examples/inverse_folding/data/5YH2.pdb`)
# To try the kit in your own environment instead: Python 3.11, the lock installed as STOCK.md 'install' shows (at minimum torch 2.4.0 with
# CUDA 12.1, torch-geometric 2.6.1 with the pt24cu121 extension wheels, biotite 0.39.0, numpy 1.26.x, scipy), fair-esm from esm_if1/stock/, then `run.sh install`.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# The C toolchain (gcc, libc headers) and the download tools of this build; the builder's core count is printed for the build-time record
# (nothing in this build compiles; the count is informational).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential && rm -rf /var/lib/apt/lists/* \
 && echo "build cores: $(nproc --all)"

# Python 3.11.5: the python-build-standalone release this stack was assembled with, unpacked into /usr/local (no system Python is used).
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete; torch +cu121 and the three PyG extension wheels are
# named there by URL and digest, everything else is the PyPI release), then stock fair-esm 2.0.1 from the bundled archive (the lock's
# `fair-esm @ git+…@2b369911` line, filtered out here by name, is the same source at the same commit).
COPY esm_if1/environment/requirements.lock /tmp/requirements.lock
COPY esm_if1/stock/fair-esm-2b369911.tar.gz /tmp/stock/fair-esm-2b369911.tar.gz
RUN grep -v -E '^(#|fair-esm @)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/fair-esm-2b369911.tar.gz \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/requirements.lock

# The kit: the shared core, then esm_if1_opt, both editable, and the install checked against the pin — the one install step, `run.sh install` (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY esm_if1 /kit/esm_if1
WORKDIR /kit/esm_if1
RUN bash run.sh install

# The process environment of the tested stack (STOCK.md 'pinned stack'): hash seed fixed, no debug info in anything a C compiler builds; the loader
# path is the base image's own. Thread-pool counts are the host's to set; the weights path is never baked in: ESM_IF1_WEIGHTS (or TORCH_HOME) is
# set at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0
