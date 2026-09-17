# OpenFold3 optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.8.1 toolkit (public NVIDIA *devel* base: nvcc compiles DeepSpeed's DS4Sci evoformer-attention op in this build, and
# `fast` compiles its FlashPairformer extension with nvcc at first use — CUDA_HOME=/usr/local/cuda) + gcc 11.4, Python 3.11.5 at /usr/local/bin/python,
# the pinned stack of openfold3/environment/requirements.lock (torch 2.10.0+cu128 with the CUDA 12.8 / cuDNN 9.10 / NCCL 2.27 runtime wheels, triton 3.6.0,
# deepspeed 0.19.2, pytorch-lightning 2.6.5, cuequivariance-torch 0.10.0, numpy 2.4.6, rdkit 2026.3.3), stock openfold3 0.4.1 from the wheel in
# openfold3/stock/, then this kit and its shared core, editable, under /kit (`run.sh install`). The DS4Sci op is compiled for compute capability 9.0 (H100)
# by default; `--build-arg DS4SCI_ARCHS="8.0;9.0"` builds one image that also serves A100 (run with --config a100 there). No weights are inside:
# `run.sh install --weights DIR` fetches of3-p2-155k.pt into a directory you mount, and OPENFOLD3_CKPT names the file at run time (README.md 'Setup').
# Host: an NVIDIA driver that runs CUDA 12.8 (570 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding openfold3/ and common/):
#   docker build -f openfold3/environment/Dockerfile -t openfold3-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/weights:/weights -e OPENFOLD3_CKPT=/weights/of3-p2-155k.pt -v "$PWD/out":/kit/openfold3/out openfold3-kit:dev \
#     bash run.sh pred --config h100 --mode fast --query-json stock/src/examples/example_inference_inputs/query_ubiquitin.json --output-dir out/fast \
#       --use-msa-server false --use-templates false
# To try the kit in your own environment instead: Python 3.11, the CUDA 12.8 toolkit (nvcc) and a C compiler on PATH, the lock installed as STOCK.md
# 'Stack' shows (at minimum torch 2.10.0 with CUDA 12.8, triton 3.6.0, deepspeed 0.19.2 with its evoformer_attn op built as below,
# pytorch-lightning 2.6.5, cuequivariance-torch 0.10.0, numpy 2.4.x, rdkit), the openfold3 wheel from openfold3/stock/, then `run.sh install`.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# The C toolchain (gcc: DeepSpeed's host compiler here, Triton's and the FlashPairformer extension's at run time), rdkit's X libraries, and the
# download tools of this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential libxrender1 libxext6 libsm6 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Python 3.11.5: the python-build-standalone release this stack was assembled with, unpacked into /usr/local (no system Python is used).
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The process environment of the tested stack (STOCK.md 'Stack'), in force from here on so the compile step below runs under it too: hash seed
# fixed, no debug info in what gcc compiles (the DeepSpeed op, Triton's launchers, the FlashPairformer extension), the CUDA toolkit root for nvcc and
# torch's extension builder. PATH and LD_LIBRARY_PATH are the base image's (torch loads the CUDA runtime, cuBLAS, cuDNN and NCCL from the lock's
# nvidia-* wheels by path). CPU thread-pool sizes (OMP_NUM_THREADS and kin) are left to the host. The weights path is never baked in: OPENFOLD3_CKPT
# is set at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    CUDA_HOME=/usr/local/cuda

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete), then stock (--no-deps): the wheel in openfold3/stock/
# when the tree carries it, else the same wheel from PyPI at the lock's openfold3 line (`run.sh install` below then checks the install file for file
# against the wheel, or against the tagged source openfold3/stock/src). deepspeed is held back for the next step, which installs the lock's deepspeed
# line from source with its op compiled.
COPY openfold3/environment/requirements.lock /tmp/requirements.lock
COPY openfold3/stock /tmp/stock
RUN grep -v -E '^(#|openfold3==|deepspeed==)' /tmp/requirements.lock > /tmp/stack.txt && grep -E '^deepspeed==' /tmp/requirements.lock > /tmp/deepspeed.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && if [ -f /tmp/stock/openfold3-0.4.1-py3-none-any.whl ]; then python -m pip install --no-cache-dir --no-deps /tmp/stock/openfold3-0.4.1-py3-none-any.whl; \
    else grep -E '^openfold3==' /tmp/requirements.lock > /tmp/stock.txt && python -m pip install --no-cache-dir --no-deps -r /tmp/stock.txt; fi \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/stock.txt

# DS4Sci: the lock's deepspeed, installed from its source distribution with the evoformer_attn op compiled ahead of time for the compute capabilities in
# DS4SCI_ARCHS — 9.0 (H100) by default; "8.0;9.0" builds one image for A100 (sm_80) and H100, "8.0" an A100-only one — by nvcc against the CUTLASS
# headers of the lock's nvidia-cutlass, gcc as host compiler (the interpreter's own build names clang, which is not here), DeepSpeed's cross-compile
# form DS_ACCELERATOR=cuda (no GPU is visible during a build). Stock's shipped configuration (use_deepspeed_evo_attention: true) then runs with no
# CUTLASS_PATH, nvcc or ninja involved at run time; on a card whose capability is not in the list that configuration fails by name (CUDA: no kernel
# image is available for execution on the device). The variables on the RUN line are scoped to this step only.
# BUILD_JOBS = parallel compile jobs (0 = every core of the builder), exported as MAX_JOBS / CMAKE_BUILD_PARALLEL_LEVEL / MAKEFLAGS for anything
# a build tool parallelises; DeepSpeed's own setup compiles this op's three translation units one after the other (its default build path,
# ninja off) with nvcc --threads 4, so this step takes about the same ten minutes on any core count.
ARG DS4SCI_ARCHS=9.0
ARG BUILD_JOBS=0
RUN J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); echo "deepspeed DS4Sci evoformer_attn op: sm ${DS4SCI_ARCHS}, ${J} build jobs, $(nproc --all) cores on this builder" \
 && export MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J DS_NVCC_THREADS=4 \
 && CC=gcc CXX=g++ LDSHARED='gcc -pthread -shared' LDCXXSHARED='g++ -pthread -shared' \
    DS_ACCELERATOR=cuda DS_BUILD_EVOFORMER_ATTN=1 TORCH_CUDA_ARCH_LIST="${DS4SCI_ARCHS}" \
    python -m pip install --no-cache-dir --no-deps --no-build-isolation --no-binary deepspeed -r /tmp/deepspeed.txt \
 && python -c "from deepspeed.git_version_info import installed_ops; assert installed_ops.get('evoformer_attn'), installed_ops; print('DS4Sci evoformer_attn op: installed')" \
 && rm -f /tmp/deepspeed.txt /tmp/requirements.lock

# The kit: the shared core, then openfold3_opt, both editable, and the install checked against the pin — the one install step, `run.sh install` (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY openfold3 /kit/openfold3
WORKDIR /kit/openfold3
RUN bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY openfold3/environment/Dockerfile _jitcache/openfold3-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
