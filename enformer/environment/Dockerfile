# Enformer optimization kit — the container image of the tested stack: the interpreter and every pinned distribution of
# environment/requirements.lock, stock enformer-pytorch 0.8.12 from enformer/stock/, and this kit installed under /kit by its own install step.
# One image serves H100 / H200 and A100: the kit selects the kernel build for the card at run time (README.md 'Notes').
#
# Base: the official python:3.11.12-slim-bookworm image, because the tested stack's interpreter IS that image's CPython 3.11.12 (Debian 12, glibc
# 2.36, built --enable-optimizations --enable-shared --with-lto) — nothing else reproduces it byte for byte; build-essential adds gcc 12.2, a host C
# compiler kept in the image for torch / Triton code paths that build extensions at run time — the kit's own modes invoke no compiler (their
# kernels ship prebuilt under opt/forward/). No CUDA base image and no system-wide CUDA installation (no nvcc): torch 2.13.0+cu130 loads the CUDA 13.0
# runtime, cuBLAS, cuDNN 9.20 and NCCL from the lock's NVIDIA wheels (cuda-toolkit meta-package, nvidia-*) through its own RPATH, and the kit's kernels ship prebuilt (opt/forward/), so nothing is compiled at build
# or install time. The GPU driver is the host's: it must run CUDA 13.0 (driver 580 or newer).
#
# Build from the release tree's top directory (the one holding enformer/ and common/):
#   docker build -f enformer/environment/Dockerfile -t enformer-kit:dev .
# Run your own enformer_pytorch code with the kit engaged (weights come from your HF cache as upstream fetches them):
#   docker run --rm --gpus all -v /data/hf_home:/weights -e HF_HOME=/weights -e ENFORMER_OPT=exact enformer-kit:dev python your_script.py
FROM python:3.11.12-slim-bookworm

# gcc / g++ / make and the libc headers (Triton compiles its launchers with the system C compiler at run time).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# The installer set first, at the lock's versions, laid down by the base image's own pip: pip, wheel and uv (run as python3.11: the
# launchers pip writes into /usr/local/bin name the interpreter they were installed with, and the tested stack's pip and wheel name that one).
COPY enformer/environment/requirements.lock /tmp/requirements.lock
RUN grep -E '^(pip|wheel|uv)==' /tmp/requirements.lock > /tmp/tools.txt \
 && python3.11 -m pip install --no-cache-dir --no-deps -r /tmp/tools.txt

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete), then the two named wheels: stock from the bundled
# wheel and nvidia-cudnn-frontend from its PyPI file, checked against the sha256 the lock carries (both --no-deps).
COPY enformer/stock/enformer_pytorch-0.8.12-py3-none-any.whl /tmp/stock/enformer_pytorch-0.8.12-py3-none-any.whl
RUN grep -v -E '^(#|(pip|wheel|uv|enformer-pytorch)==|nvidia-cudnn-frontend @ )' /tmp/requirements.lock > /tmp/stack.txt \
 && grep -E '^nvidia-cudnn-frontend @ ' /tmp/requirements.lock > /tmp/named.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/enformer_pytorch-0.8.12-py3-none-any.whl -r /tmp/named.txt \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/named.txt /tmp/tools.txt /tmp/requirements.lock

# The kit: enformer_opt installed editable and the install checked against the pins — the one install step, `run.sh install` (README.md 'Setup').
# The package depends on nothing outside the lock (opt/pyproject.toml), so no shared core is installed.
COPY enformer /kit/enformer
WORKDIR /kit/enformer
RUN bash run.sh install

# The process environment of the tested stack: hash seed fixed, no debug info in the launchers Triton compiles. The weights cache is never baked
# in: HF_HOME is set at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0
