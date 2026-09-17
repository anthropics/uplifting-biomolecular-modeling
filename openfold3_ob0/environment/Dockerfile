# OpenFold3 0.5.0 (OpenBind-0) optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.8.1 toolkit (public NVIDIA base; the devel flavour, not runtime, because DeepSpeed's DS4Sci
# evoformer-attention op is compiled here by nvcc for compute capability 9.0) + gcc 11.4 (build-essential: that compile, and the Triton
# kernels stock and the kit build at first use) + the X client libraries RDKit's drawing module loads, Python 3.11.5 at /usr/local/bin/python,
# the pinned stack of openfold3_ob0/environment/requirements.lock (torch 2.10.0+cu128 with the CUDA 12.8 / cuDNN 9.10 / NCCL 2.27 runtime wheels,
# triton 3.6.0, cuequivariance-torch 0.10.0, deepspeed 0.19.2 with the DS4Sci op built in, pytorch-lightning 2.6.5, numpy 2.4.6, rdkit 2025.9.3,
# biotite 1.6.0), stock openfold3 0.5.0 from the wheel in openfold3_ob0/stock/, then this kit and its shared core, editable, under /kit, with
# upstream's full Chemical Component Dictionary written into biotite by upstream's own setup step (`run.sh install`; checked against stock/PINS.json).
# The image targets H100 (compute capability 9.0: the DS4Sci op and the kit's modes are built and tuned for it). No weights are inside:
# `run.sh install --weights DIR` fetches the OpenBind-0 checkpoint (2.3 GB) into a directory you mount, and OPENFOLD3_OB0_CKPT names the file at
# run time (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 12.8 (570 or newer) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding openfold3_ob0/ and common/):
#   docker build -f openfold3_ob0/environment/Dockerfile -t openfold3_ob0-kit:dev .
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/weights:/weights -e OPENFOLD3_OB0_CKPT=/weights/of3-ob-2025-06-30-174k.pt -v "$PWD/out":/kit/openfold3_ob0/out \
#     openfold3_ob0-kit:dev bash run.sh pred --config h100 --mode fast --query-json stock/src/examples/example_inference_inputs/query_ubiquitin.json \
#     --output-dir out/fast --use-msa-server false
# To try the kit in your own environment instead: Python 3.11 and the CUDA 12.8 toolkit (nvcc) with gcc on PATH, the lock installed as STOCK.md
# 'Stack' shows for README.md 'Setup' route C (the three pip steps below), then `run.sh install`.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# The C/C++ toolchain (the DS4Sci compile below; Triton's launcher builds at run time), the X libraries rdkit.Chem.Draw loads, the download tools of this build.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential libxrender1 libxext6 libsm6 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Python 3.11.5: the python-build-standalone distribution (release 20230826, install_only), unpacked over /usr/local; it brings pip 23.2.1.
RUN curl -fsSL -o /tmp/python.tar.gz \
      "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -RL --preserve=mode,timestamps /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V

# The process environment of the tested stack, set before the stack is built so the DeepSpeed compile below runs under it too: the CUDA toolkit
# root the extension build reads, hash randomisation off, no debug info in anything compiled here or at run time.
ENV CUDA_HOME=/usr/local/cuda \
    PYTHONHASHSEED=0 \
    CFLAGS=-g0

# The pinned stack, three steps from one lock: every line --no-deps except two — stock openfold3 comes from its pinned PyPI wheel (the file in
# stock/ when the build context carries it, else fetched from PyPI by stock/install_upstream.py --wheel-only and checked against stock/PINS.json
# sha256 + bytes before it is installed), and deepspeed is built from its PyPI source distribution with the DS4Sci evoformer-attention op compiled
# for compute capability 9.0 (CUTLASS headers from the lock's nvidia-cutlass; no GPU is needed to build; `--build-arg BUILD_JOBS=N` caps the
# compile's parallel jobs, default all cores). The build then asserts the op is installed and the environment consistent (pip check).
COPY openfold3_ob0/environment/requirements.lock /tmp/requirements.lock
COPY openfold3_ob0/stock/PINS.json openfold3_ob0/stock/check_pins.py openfold3_ob0/stock/install_upstream.py openfold3_ob0/stock/*.whl /tmp/stock/
ARG BUILD_JOBS=0
RUN grep -v -E '^(#|openfold3==|deepspeed==)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -I /tmp/stock/install_upstream.py --wheel-only \
 && J=${BUILD_JOBS:-0} && { [ "$J" -gt 0 ] || J=$(nproc --all); } && echo "deepspeed DS4Sci build: $J jobs on $(nproc --all) cores" \
 && MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=4 DS_NVCC_THREADS=4 \
    CC=gcc CXX=g++ LDSHARED='gcc -pthread -shared' LDCXXSHARED='g++ -pthread -shared' \
    DS_ACCELERATOR=cuda DS_BUILD_EVOFORMER_ATTN=1 TORCH_CUDA_ARCH_LIST="9.0" \
    python -m pip install --no-cache-dir --no-deps --no-build-isolation --no-binary deepspeed "$(grep -E '^deepspeed==' /tmp/requirements.lock)" \
 && python -c "from deepspeed.git_version_info import installed_ops; assert installed_ops.get('evoformer_attn'), installed_ops; print('DS4Sci evoformer_attn op: built')" \
 && python -m pip check \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/requirements.lock /root/.cache

# The kit and its shared core under /kit, installed editable by the kit's own install step, which also runs the pin check and places upstream's
# full CCD into biotite (fetched from upstream's bucket, then checked against stock/PINS.json).
COPY common/opt_core /kit/common/opt_core
COPY openfold3_ob0 /kit/openfold3_ob0
WORKDIR /kit/openfold3_ob0
RUN bash run.sh install

# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY openfold3_ob0/environment/Dockerfile _jitcache/openfold3_ob0-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
