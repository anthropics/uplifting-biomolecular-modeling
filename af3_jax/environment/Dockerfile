# AlphaFold 3 (JAX) optimization kit — the tested stack as a container image.
# Builds Ubuntu 24.04 + the CUDA 12.6.3 base image (public NVIDIA image; the base flavour is the one this stack was assembled on: jax brings the CUDA 12.9
# runtime, cuDNN 9.17 and NCCL 2.29 as pip wheels, so no CUDA user-space library of the system is used) + gcc 13.3 and upstream's build tools, Python 3.12.1
# (python-build-standalone) at /usr/local/bin/python3 with uv 0.9.24, the stock environment /alphafold3_venv holding the pinned stack of
# af3_jax/environment/requirements.lock (jax / jaxlib / jax-cuda12-plugin 0.10.2, dm-haiku 0.0.16, tokamax 0.0.12, numpy 2.4.1, rdkit 2025.9.4, torch 2.7.1+cpu)
# and stock — the fork at the STOCK.md pin with the one stock patch — as a wheel, upstream's `build_data` step, and the fork checkout at /app/alphafold
# (run_alphafold.py and convert_of3_weights.py, which the routes call); then this kit and its shared core, editable, in an environment of their own at
# /kit/venv (`run.sh install`). The stock wheel comes one of two ways, chosen by --build-arg WHEELS_FROM: `build` (the default) compiles it during the build
# from the pinned stock source (af3_jax/stock/unpack_src.sh: the upstream archive under stock/ when the tree carries it, otherwise a clone of the
# pinned commit) with the patch applied, by upstream's own build (scikit-build-core + CMake fetch abseil, pybind11, libcifpp and
# dssp at the tags CMakeLists.txt pins, and libcifpp fetches the wwPDB Chemical Component Dictionary of the build day — network needed; such
# a wheel runs every mode, but its dictionary snapshot and compiled bytes are that build's own, never the tested wheel's, so nothing compares its
# digest); `prebuilt` installs the tested wheel af3_jax/stock/wheels/alphafold3_open-3.1.4-cp312-cp312-linux_x86_64.whl (README.md Setup says where it is
# published; its compiled extension and dictionary snapshot are the tested bytes) after comparing its sha256 with stock/PINS.json upstream.wheel.
# One image serves every card the kit runs on: nothing is compiled for a GPU at build time. No weights are inside: `run.sh install --weights DIR` fetches the
# public checkpoint the fork's --of3_weights path loads (stock/PINS.json names it) and converts it into a directory you mount, and AF3_JAX_PARAMS_ROOT names it at run time (README.md Setup). The
# genetic-search stage of upstream (HMMER, sequence databases) is not in the image: the routes run fold-input JSONs that already carry their MSAs and templates.
# Host: an NVIDIA driver that runs CUDA 12 (550 or newer; 580 tested) and the NVIDIA container toolkit.
# Build, from the bundle's top directory (the one holding af3_jax/ and common/):
#   docker build -f af3_jax/environment/Dockerfile -t af3_jax-kit:dev .                                  # stock compiled from af3_jax/stock/ during the build
#   docker build -f af3_jax/environment/Dockerfile -t af3_jax-kit:dev --build-arg WHEELS_FROM=prebuilt .  # the tested wheel, placed in af3_jax/stock/wheels/ first
# Run (README.md 'Run' has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/af3_params:/params -e AF3_JAX_PARAMS_ROOT=/params -v /path/to/af3_cache:/cache -e AF3_JAX_CACHE_ROOT=/cache \
#     -v "$PWD/inputs":/inputs -v "$PWD/out":/kit/af3_jax/out af3_jax-kit:dev \
#     bash run.sh pred --config h100 --variant p2 --mode fast --input_dir /inputs --output_dir out/fast
# To try the kit in your own environment instead: README.md Setup / STOCK.md 'Install outside the image' is these same steps without Docker (Python 3.12, the lock, the stock wheel,
# build_data, the checkout, then `run.sh install` from an environment of the kit's own).
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

# Upstream's build tools (gcc / g++ / make, zlib headers, git, zstd) — the stock wheel's compile (WHEELS_FROM=build) and a user building inside this image have
# what upstream's Dockerfile has — and the download tools of this build.
RUN apt-get update --quiet \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet git wget curl ca-certificates gcc g++ make zlib1g-dev zstd patch \
 && rm -rf /var/lib/apt/lists/*

# Python 3.12.1: the python-build-standalone build this stack was assembled with (release 20240107, the x86_64_v3 install_only archive), unpacked into /usr/local
# with the archive's few internal links written as plain files (the layout this stack was assembled on) — no system Python is used — and uv 0.9.24, the installer the stock environment was assembled with, installed into it with its bundled pip.
RUN curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.12.1+20240107-x86_64_v3-unknown-linux-gnu-install_only.tar.gz" \
 && echo "a7bcc6c9f66dbd47ea99615f30f101c5d2dd0084ca333d2f7336e64050951338  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -RL --preserve=mode,timestamps /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python3 -V \
 && python3 -m pip install --no-cache-dir uv==0.9.24

# The process environment of the tested stack (STOCK.md 'pinned stack'): hash seed fixed, unbuffered output, C builds by gcc / g++ without debug info (the
# interpreter's own build records clang, which is not in the image: the stock wheel's compile and anything pip compiles here use these) and the three XLA
# variables every route runs the model with (stock/check_pins.py refuses a process without them). The loader path stays the base image's own (LD_LIBRARY_PATH=/usr/local/nvidia/lib:
# /usr/local/nvidia/lib64): jax's cuda plugin resolves cuBLAS, cuDNN, NCCL and the CUDA runtime from the nvidia-* wheels of the lock by its own RPATH. Set before the build steps below so upstream's build_data and the kit's install step run in it too. The weights and cache roots are never
# baked in: AF3_JAX_PARAMS_ROOT / AF3_JAX_CACHE_ROOT are set at run time.
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    CFLAGS=-g0 \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
    XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_CLIENT_MEM_FRACTION=0.95

# The stock environment, upstream's layout: a uv virtual environment of that interpreter at /alphafold3_venv, on PATH, holding every distribution of the lock
# at its version (--no-deps: the lock is complete; torch 2.7.1+cpu by the URL and sha256 its line names, everything else from PyPI) in two passes read from the
# lock by name — upstream's own set through the environment's `python3`, then torch and sympy, the weights converter's additions, through its `python`, the
# order and interpreter names this stack was assembled with (an entry point's first line names the interpreter it was installed through); then the stock wheel —
# WHEELS_FROM=build: compiled now from the bundled source with the stock patch applied (`uv build --wheel`: scikit-build-core in an isolated build environment
# that brings CMake and ninja; --build-arg BUILD_JOBS=N caps the parallel compile jobs, default every core) and named with its digest; WHEELS_FROM=prebuilt: the tested wheel copied in from af3_jax/stock/wheels/, installed only after its
# sha256 equals stock/PINS.json upstream.wheel, or the digest of a wheel you built once and keep there, passed as --build-arg STOCK_WHEEL_SHA256=<sha256> —
# installed --no-deps; then upstream's `build_data` (the CCD pickles inside the installed package, from the dictionary snapshot the wheel carries); the fork
# checkout is the pinned source laid out at /app/alphafold by af3_jax/stock/unpack_src.sh (the archive under stock/, or a clone of stock/PINS.json
# upstream.repo at upstream.commit when the tree carries no archive) with the stock patch applied — the source of either wheel.
ARG WHEELS_FROM=build
ARG STOCK_WHEEL_SHA256
ARG BUILD_JOBS=0
ENV PATH=/alphafold3_venv/bin:$PATH
COPY af3_jax/environment/requirements.lock /tmp/stock/
COPY af3_jax/stock /tmp/stock/tree
RUN uv venv --python /usr/local/bin/python3.12 /alphafold3_venv \
 && grep -v -E '^(torch|sympy)([ =@]|$)' /tmp/stock/requirements.lock > /tmp/stock/upstream.lock && grep -E '^(torch|sympy)([ =@]|$)' /tmp/stock/requirements.lock > /tmp/stock/converter.lock \
 && uv pip install --python /alphafold3_venv/bin/python3 --no-cache --no-deps -r /tmp/stock/upstream.lock \
 && uv pip install --python /alphafold3_venv/bin/python --no-cache --no-deps -r /tmp/stock/converter.lock \
 && bash /tmp/stock/tree/unpack_src.sh /app/alphafold \
 && if [ "$WHEELS_FROM" = build ]; then J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); echo "stock wheel: compiling with $J parallel jobs ($(nproc --all) cores here)"; export MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J; \
      cp -a /app/alphafold /tmp/af3src && uv build --wheel --python /usr/local/bin/python3.12 --out-dir /tmp/stock/tree/wheels /tmp/af3src && rm -rf /tmp/af3src /root/.cache; \
    elif [ "$WHEELS_FROM" != prebuilt ]; then echo "WHEELS_FROM=$WHEELS_FROM: build or prebuilt" >&2; exit 2; fi \
 && python3 -c "import hashlib, json, os, sys; w = json.load(open('/tmp/stock/tree/PINS.json'))['upstream']['wheel']; p = '/tmp/stock/tree/wheels/' + w['file']; route = os.environ.get('WHEELS_FROM', 'build'); own = os.environ.get('STOCK_WHEEL_SHA256', ''); os.path.isfile(p) or sys.exit(w['file'] + ' is not in af3_jax/stock/wheels/: fetch the tested wheel there first (README.md Install), or build with WHEELS_FROM=build'); h = hashlib.sha256(open(p, 'rb').read()).hexdigest(); built = route == 'build'; ok = built or h == (own or w['sha256']); print('stock wheel ' + w['file'] + ': sha256 ' + h + (' — compiled from af3_jax/stock/ in this build (stock/PINS.json upstream.wheel, ' + w['sha256'][:16] + '..., describes the tested wheel; a source build never repeats its bytes)' if built else ' = STOCK_WHEEL_SHA256, a wheel of your own build (stock/PINS.json upstream.wheel describes the tested one)' if own else ' = stock/PINS.json upstream.wheel, the tested build') if ok else w['file'] + ': sha256 ' + h + ' is not ' + ('STOCK_WHEEL_SHA256=' + own if own else 'the pin ' + w['sha256'] + ' (stock/PINS.json upstream.wheel) — a wheel you built yourself takes --build-arg STOCK_WHEEL_SHA256=<its sha256>'), file=sys.stderr); sys.exit(0 if ok else 1)" \
 && uv pip install --python /alphafold3_venv/bin/python3 --no-cache --no-deps /tmp/stock/tree/wheels/alphafold3_open-3.1.4-cp312-cp312-linux_x86_64.whl \
 && build_data \
 && rm -rf /tmp/stock

# The kit: an environment of its own for af3_jax_opt and the shared core, made from the same interpreter and first on PATH — `python` is the kit's, the fork's
# stays AF3_JAX_PY (/alphafold3_venv/bin/python; the kit never installs into the stock environment) — then both installed editable and the install checked
# against the pin: the one install step, `run.sh install` (README.md Setup). Last, the kit's own copy of the model script is placed beside the stock one
# in the checkout, the step `pred` and `warm` otherwise do at first use, so a read-only image (Apptainer) runs every mode as built; a stock wheel that came in
# with the kit tree (the prebuilt route) is not kept a second time under /kit.
COPY common/opt_core /kit/common/opt_core
COPY af3_jax /kit/af3_jax
RUN /usr/local/bin/python3.12 -m venv /kit/venv
ENV PATH=/kit/venv/bin:$PATH
WORKDIR /kit/af3_jax
RUN bash run.sh install \
 && python -c "from af3_jax_opt.cli import ensure_fast_script; r = ensure_fast_script(); print('[af3-jax-opt install]', r['path'], r['state'])" \
 && rm -f /kit/af3_jax/stock/wheels/*.whl
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY af3_jax/environment/Dockerfile _jitcache/af3_jax-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit
