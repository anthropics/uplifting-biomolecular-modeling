# GPN-Star optimization kit — the tested stack as a container image.
# Base: the official `python:3.13.15-slim-bookworm` image — Debian 12 with CPython 3.13.15 in /usr/local, the interpreter series stock gpn requires
# (Python >=3.13,<3.14); no CUDA base image is involved: torch 2.13.0+cu130 loads the CUDA 13.0 runtime, cuBLAS 13.1, cuDNN 9.20 and NCCL 2.29 from
# the nvidia-* and cuda-toolkit wheels of gpnstar/environment/requirements.lock, and the GPU driver is the host's.
# On top: the pinned stack of requirements.lock (torch 2.13.0+cu130, triton 3.7.1, transformers 5.15.0, huggingface_hub 1.31.0, safetensors 0.8.0,
# accelerate 1.15.0, numpy 2.5.3, zarr 3.3.0), stock gpn 0.9.0 built from the archive of commit 6f28c81b in gpnstar/stock/, then this kit and the
# shared core it imports, editable, under /kit (`run.sh install`). Nothing is compiled at build time, and the kit's own path compiles nothing at
# run time: the model runs eager PyTorch in fp32 with the attention math backend, so the image carries no compile-cache step. The one toolchain
# piece inside is gcc with the C library headers, for upstream's optional `--torch-compile` flag: torch.compile's inductor backend compiles its
# kernels' host stubs with a C compiler at run time. One image serves H100 and A100: nothing in it is built for one GPU architecture. No weights
# are inside: `run.sh install --weights DIR` stages the checkpoints (the Hugging Face snapshots at the revisions pinned in stock/PINS.json, each
# file digest-checked) into a directory you mount, and HF_HOME names it at run time (README.md 'Setup'). Host: an NVIDIA driver that runs CUDA
# 13.0 (580 or newer) and the NVIDIA container toolkit. The build downloads wheels and two Debian packages; nothing is compiled.
# Build, from the bundle's top directory (the one holding gpnstar/ and common/):
#   docker build -f gpnstar/environment/Dockerfile -t gpnstar-kit:dev .
# Run (README.md 'Run': the unchanged `gpn star …` command with GPNSTAR_OPT=exact in front; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/hf_home:/weights -e HF_HOME=/weights -v "$PWD/out":/kit/gpnstar/out gpnstar-kit:dev bash run.sh check
#   docker run --rm --gpus all -v /path/to/hf_home:/weights -e HF_HOME=/weights -e HF_HUB_OFFLINE=1 -e GPNSTAR_OPT=exact -v "$PWD/out":/kit/gpnstar/out gpnstar-kit:dev \
#     gpn star vep --input-path examples/variants.parquet --msa-path examples/msa --window-size 128 --model-path /weights/hub/models--songlab--gpn-star-hg38-v100-200m/snapshots/<commit> --output-path out/scores.parquet --tf32
# To try the kit in your own environment instead: Python 3.13, the lock installed from the bundle's top directory as README.md 'Setup' shows (at
# minimum torch 2.13.0 with CUDA 13.0, transformers 5.15.0, huggingface_hub 1.31.0, safetensors 0.8.0, numpy 2.5.3 and stock gpn at commit
# 6f28c81b), then `run.sh install --weights DIR`.
# The tag is shown for readability; the digest pins the bytes (the multi-arch manifest list the tag resolves to).
FROM python:3.13.15-slim-bookworm@sha256:ed86c82274b3c69b52fb5820f358f0bd7df0b603332063cb5c6e32bd220c3e6e

# The process environment of the tested stack: unbuffered output, hash seed fixed, no usage telemetry from the Hugging Face libraries. The weights
# path is never baked in: HF_HOME is set at run time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DISABLE_TELEMETRY=1

# A C compiler and the C library headers (gcc, libc6-dev; no build-essential): torch.compile's inductor backend compiles its kernels' host stubs
# with a C compiler at run time, which upstream's `--torch-compile` flag reaches; the kit's eager path never calls it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

# The pinned stack, every distribution at the lock's version (--no-deps: the lock is complete): pip first, at the lock's own line, then the rest less
# the stock line; stock gpn 0.9.0 is then built by hatchling from the archive COPY'd from gpnstar/stock/ (the lock names this same archive by the
# path `./gpnstar/stock/gpn-6f28c81b.tar.gz`, which resolves only from the bundle's top directory, so that line is filtered out here and the copy
# is installed explicitly), hatchling held at one release so the build backend is the same on every build. pip runs
# as `python3.13 -m pip`: the console scripts it writes (`gpn` among them) then name /usr/local/bin/python3.13. The build host's core count is
# printed first: the build is downloads and one pure-Python wheel build, so its time is network-bound.
COPY gpnstar/environment/requirements.lock /tmp/requirements.lock
COPY gpnstar/stock/gpn-6f28c81b.tar.gz /tmp/stock/gpn-6f28c81b.tar.gz
RUN echo "build host: $(nproc --all) cores" \
 && python3.13 -m pip install --no-cache-dir "$(grep -E '^pip==' /tmp/requirements.lock)" \
 && grep -v -E '^(#|pip==|[.]/gpnstar/stock/)' /tmp/requirements.lock > /tmp/stack.txt \
 && python3.13 -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && echo 'hatchling==1.32.0' > /tmp/build-constraints.txt \
 && PIP_CONSTRAINT=/tmp/build-constraints.txt python3.13 -m pip install --no-cache-dir --no-deps /tmp/stock/gpn-6f28c81b.tar.gz \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/requirements.lock /tmp/build-constraints.txt

# The kit: the shared core, then gpnstar_opt, both editable, and the install checked against the pins — the one install step, `run.sh install`
# (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY gpnstar /kit/gpnstar
WORKDIR /kit/gpnstar
RUN bash run.sh install
