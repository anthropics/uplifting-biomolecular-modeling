# Proteina-Complexa optimization kit — the tested stack as a container image.
# Base: the official python:3.12.10-slim-bookworm image, because that is the tested stack's own base — Debian 12 with the docker-library build of
# CPython 3.12.10 under /usr/local (configure --enable-optimizations --with-lto --enable-shared); there is no nvidia/cuda base and no CUDA toolkit in
# the image: the CUDA 12.6 runtime, cuDNN 9.5.1 and NCCL 2.26.2 are the nvidia-*-cu12 wheels of complexa/environment/requirements.lock, which torch
# 2.7.0 loads through its own RPATH. On top: the tested image's Debian packages — build-essential (gcc 12: Triton builds its launchers at first use),
# git, wget (the transfer tool upstream's weight downloader calls), gfortran and libxrender1, installed with their recommended packages as Debian
# does by default (among them libx11-6 and libxext6: the openbabel wheel's format plugins link against libX11 / libXext / libXrender, and upstream's
# data pipeline imports openbabel) — then the pinned stack of the lock (172 distributions, `pip install --no-deps -r`), upstream Proteina-Complexa
# 1.1.0 (github.com/NVIDIA-Digital-Bio/proteina-complexa @916eaaed) unpacked from the archive in complexa/stock/ to /opt/pc and installed editable together with its
# community_models/colabdesign — the tested layout, LOCAL_CODE_PATH=/opt/pc — then this kit and its shared core, editable, under /kit
# (`run.sh install`). pip runs as `python3.12 -m pip` throughout: pip writes the interpreter it runs under into the first line of every script it
# installs in /usr/local/bin, and the tested stack's scripts name /usr/local/bin/python3.12. One image serves H100 and A100: nothing is compiled
# for a GPU at build time. No weights are inside: `run.sh install --weights DIR` fetches complexa.ckpt + complexa_ae.ckpt (NVIDIA Open Model License)
# into a directory you mount, and CKPT_PATH names it at run time (README.md 'Setup'). ENV: PYTHONHASHSEED=0 and CFLAGS=-g0 are the values every
# tested run had in its environment (hash randomisation off; no debug info in what Triton and pip compile); LOCAL_CODE_PATH is where this file puts
# upstream. Host: linux x86_64, an NVIDIA driver that runs CUDA 12.6 (560 or newer; data-centre drivers from 525 on through CUDA minor-version
# compatibility) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding complexa/ and common/):
#   docker build -f complexa/environment/Dockerfile -t complexa-kit:dev .
# Run (the weights directory mounted read-only, an output directory mounted writable; every verb of run.sh works the same way):
#   docker run --rm --gpus all -v /path/to/complexa_weights:/weights:ro -e CKPT_PATH=/weights -v "$PWD/out":/out complexa-kit:dev \
#     bash run.sh design --config h100 --mode off --out /out/pdl1 -- ++generation.dataloader.dataset.nres.nsamples=2
# To try the kit in your own environment instead: Python 3.12 on linux x86_64, the lock installed with `pip install --no-deps -r` as below (at minimum
# torch 2.7.0 with CUDA 12.6, lightning 2.5.x, hydra-core 1.3.1 and upstream's own dependencies), upstream's checkout from complexa/stock/ installed
# editable with LOCAL_CODE_PATH naming it, then `run.sh install`.
FROM python:3.12.10-slim-bookworm

# The tested image's Debian package set: the C and Fortran toolchains, git, wget, libxrender1, with Debian's recommended packages (the X11 client
# libraries the openbabel wheel's plugins load come in that way).
RUN apt-get update && apt-get install -y build-essential git wget gfortran libxrender1 && rm -rf /var/lib/apt/lists/*

# The pinned stack. First the lock's own pip (the tested stack's pip is newer than the interpreter's bundled one), read from the lock; then every
# distribution at the lock's version (--no-deps: the lock is complete). The two `-e git+…` lines of the lock are upstream itself and are installed
# from the bundled archive below, never from the network. PIP_CONSTRAINT holds the lock's `==` lines plus upstream's build backend at the version
# that built the tested tree (hatchling, fetched by pip at build time and never installed into the image): the isolated environments in which pip
# builds the lock's source distributions and the editable installs below resolve their build tools (setuptools, wheel, numpy, hatchling) to the
# tested versions instead of the newest release of the day.
COPY complexa/environment/requirements.lock /tmp/requirements.lock
RUN grep -v '^-e ' /tmp/requirements.lock > /tmp/stack.txt \
 && { grep -E '^[A-Za-z0-9._-]+==' /tmp/stack.txt; echo 'hatchling==1.31.0'; } > /tmp/build-pins.txt \
 && grep -E '^pip==' /tmp/stack.txt | xargs python3.12 -m pip install --no-cache-dir \
 && PIP_CONSTRAINT=/tmp/build-pins.txt python3.12 -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt

# Stock: upstream's tree at the pinned commit (complexa/stock/proteina-complexa-916eaaed.tar.gz, `git archive` of 916eaaed) at /opt/pc, installed
# editable with no dependency resolution (the lock already holds its dependencies at the tested versions), plus the ColabDesign copy upstream ships
# and installs from community_models/colabdesign (colabdesign 1.1.1.1) — the two editable installs of the tested stack (STOCK.md).
COPY complexa/stock/proteina-complexa-916eaaed.tar.gz /tmp/stock/proteina-complexa-916eaaed.tar.gz
RUN tar -xzf /tmp/stock/proteina-complexa-916eaaed.tar.gz -C /opt && mv /opt/proteina-complexa-916eaaed /opt/pc \
 && PIP_CONSTRAINT=/tmp/build-pins.txt python3.12 -m pip install --no-cache-dir --no-deps -e /opt/pc -e /opt/pc/community_models/colabdesign \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/build-pins.txt /tmp/requirements.lock

# The tested runs' environment values (header) and upstream's checkout inside this image: the kit reads LOCAL_CODE_PATH (configs/h100.env gives it
# no default). The weights root is never baked in: CKPT_PATH is set at run time.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    LOCAL_CODE_PATH=/opt/pc

# The kit: the shared core, then complexa_opt, both editable, and the install checked against the pin — the one install step, `run.sh install`
# (README.md 'Setup').
COPY common/opt_core /kit/common/opt_core
COPY complexa /kit/complexa
WORKDIR /kit/complexa
RUN bash run.sh install
