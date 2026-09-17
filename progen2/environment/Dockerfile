# ProGen2 optimization kit — the tested stack as a container image. Base: the public NVIDIA CUDA 12.8.1 runtime image on Ubuntu 22.04 — the runtime
# flavour of the base this stack was assembled and tested on, since nothing in the kit compiles CUDA source at build, install or run time.
# Builds Ubuntu 22.04 + the CUDA 12.8.1 runtime (public NVIDIA base) + gcc 11.4 (build-essential), python-build-standalone CPython 3.9.23 under /opt/python
# (fetched by uv 0.8.13, the tool this stack's interpreter was installed with) and the virtual environment /opt/venv on it — first on PATH, so `python` is the
# stack's interpreter — holding the pinned stack of progen2/environment/requirements.lock, pip 26.0.1 included (torch 2.8.0+cu128 with the CUDA 12.8 / cuDNN 9.10 / NCCL 2.27 runtime
# wheels, transformers 4.16.2, tokenizers 0.10.3, numpy 2.0.2), then this kit, editable, under /kit (`run.sh install`). Stock ProGen2
# (salesforce/progen @ c27a419c, the progen2/ scripts) is not a package: it ships in the kit tree at /kit/progen2/stock/src/progen2 and runs from there.
# One image serves H100 and A100: nothing is compiled for a GPU at build time (the kit's fused kernels ship prebuilt for sm_80 / sm_90 / sm_100). No weights
# are inside: `run.sh install --weights DIR` fetches the per-size checkpoints into a directory you mount, and PROGEN2_WEIGHTS names it at run time
# (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA 12.8 (570 or newer) and the NVIDIA Container Toolkit.
# Build, from the directory that holds progen2/ (the build context; nothing outside progen2/ is copied):
#   docker build -f progen2/environment/Dockerfile -t progen2-kit:dev .
# Run (README.md 'Run' has one command per verb and mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/progen2_weights:/weights -e PROGEN2_WEIGHTS=/weights -v "$PWD/out":/kit/progen2/out progen2-kit:dev \
#     bash run.sh score --model progen2-xlarge --mode exact --input out/items.jsonl --out_dir out/score
# To try the kit in your own environment instead: Python 3.9 with pip, `python -m pip install --no-deps -r progen2/environment/requirements.lock`
# (README.md 'Setup'; at minimum torch 2.8.0+cu128, transformers 4.16.2, tokenizers 0.10.3), then `run.sh install`.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# The C toolchain of the tested image (gcc, libc headers) and the download tools of this build (the build compiles nothing: its time is download
# and install time; the host's core count is printed for the record).
RUN echo "build host: $(nproc --all) cores" && apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential && rm -rf /var/lib/apt/lists/*

# CPython 3.9.23: the python-build-standalone build (release 20250818, the plain x86_64 GNU asset) uv 0.8.13 installs for that request, checked by uv against
# its published digest, under /opt/python (readable by any user the container runs as); then the virtual environment /opt/venv on it, first on PATH. uv
# itself is a build tool and is removed.
RUN curl -fsSL -o /tmp/uv.tar.gz "https://github.com/astral-sh/uv/releases/download/0.8.13/uv-x86_64-unknown-linux-gnu.tar.gz" \
 && echo "8ca3db7b2a3199171cfc0870be1f819cb853ddcec29a5fa28dae30278922b7ba  /tmp/uv.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/uv.tar.gz -C /tmp && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv \
 && UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_BIN_DIR=/tmp/uv-bin UV_CACHE_DIR=/tmp/uv-cache uv python install cpython-3.9.23-linux-x86_64-gnu \
 && UV_CACHE_DIR=/tmp/uv-cache uv venv /opt/venv --python /opt/python/cpython-3.9.23-linux-x86_64-gnu/bin/python3.9 \
 && rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu /tmp/uv-bin /tmp/uv-cache /usr/local/bin/uv && /opt/venv/bin/python -V
ENV PATH="/opt/venv/bin:${PATH}"

# The pinned stack: the environment's pip first (the interpreter's bundled pip bootstraps it to the lock's own pip line), then every distribution at the
# lock's version in one pass (--no-deps: the lock is complete; the torch and triton lines are those wheels' own URLs and digests on the PyTorch index, every other line PyPI's).
COPY progen2/environment/requirements.lock /tmp/requirements.lock
RUN python -m ensurepip --default-pip >/dev/null && grep -E '^pip==' /tmp/requirements.lock | xargs python -m pip install --no-cache-dir \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.lock \
 && rm -f /tmp/requirements.lock

# The kit: progen2_opt, editable (it depends on nothing beyond the stack), and the install checked against the pin — the one install step,
# `run.sh install` (README.md 'Setup').
COPY progen2 /kit/progen2
WORKDIR /kit/progen2
RUN bash run.sh install

# The process environment of the tested stack (STOCK.md 'Pinned stack'): hash seed fixed, no debug info in anything a C compiler builds, no .pyc
# files written into the tree, the tokenizers library's thread pool off (the stock scripts set the same), and the loader path of the tested image
# (inert for torch, whose CUDA libraries resolve through its own RPATH to the wheels of the lock). CPU thread-pool sizes (OMP_NUM_THREADS and kin)
# are the host's to set. The weights root is never baked in: PROGEN2_WEIGHTS is set at run time.
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    CFLAGS=-g0 \
    TOKENIZERS_PARALLELISM=false \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64
