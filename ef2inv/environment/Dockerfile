# ef2inv optimization kit — the tested stack as a container image.
# Builds Ubuntu 22.04 + the CUDA 12.8.1 runtime (public NVIDIA base: the tested stack's own base line at runtime weight — nothing is compiled
# at build time, the two compiled wheels arrive as files, and transformer_engine resolves libcublas / libcudart from the base's /usr/local/cuda/lib64)
# + gcc 11.4 (build-essential: Triton and torch's inductor build kernels and their launchers at first use) + git (two distributions install from
# git at pinned commits and the stock design script is a git checkout), uv 0.12.5 (the installer of the whole stack: the interpreter and every
# distribution; the environment carries no pip), Python 3.12.14 (uv's python-build-standalone build) under /opt/uv-python, the stack's virtual
# environment /opt/ef2inv/.venv holding the pinned stack of ef2inv/environment/requirements.lock (torch 2.11.0+cu128 with the CUDA 12.8 /
# cuDNN 9.19 / NCCL 2.28 runtime wheels, triton 3.6.0, xformers 0.0.35, cuequivariance 0.10.0, numpy 2.5.2, the ESMFold2 transformers fork at
# ef32577f), stock esm 3.4.0 as the editable checkout /opt/esmfold2/esm at d0207ea3 (STOCK.md — the cookbook design script ships in the checkout),
# flash-attn 2.8.3 and transformer_engine 2.15.0+42b8400 — the two distributions with no index build for this torch — by one of two routes
# (build argument WHEELS_FROM): `build` (the default; self-contained) compiles both wheels here from their pinned sources — the flash-attn 2.8.3
# PyPI sdist and NVIDIA/TransformerEngine at commit 42b84005 (v2.15) — against this exact stack with nvcc 12.8 for the CUDA architectures of the
# build argument STACK: `img_ef2inv` (the default) = sm_90, the settings stock/PINS.json pinned_stack.image_recipe records for the tested wheels;
# `img_ef2inv_a100` = sm_80 and sm_90, the A100 80GB build of pinned_stack.arch_variants.sm80 (the same sources, flags and stack: only the two
# wheels' device code differs, so every version pin holds on it) (BUILD_JOBS
# parallel compile jobs, default 0 = every core, each wanting a few GB of RAM — lower it on a small machine with --build-arg BUILD_JOBS=N),
# then removes the compiler again; `prebuilt` installs the two published wheel files placed in ef2inv/stock/wheels/ (STOCK.md §Stack)
# after their sha256 match stock/PINS.json — minutes, and byte-identical to the tested stack, which a source build cannot
# promise (nvcc does not emit the same bytes twice; the arithmetic is the same source at the same flags). Then HMMER 3.4 — the profile-HMM
# programs ANARCI runs (`hmmscan`) when the cookbook numbers an antibody-framework binder; abnumber 0.4.4 and ANARCI 2026.2.13.2 themselves are lines
# of the lock — as the conda packages of ef2inv/environment/conda-hmmer.lock (bioconda's build and the conda-forge libraries it links, every package
# pinned by URL and md5) in their own prefix /opt/hmmer, installed by micromamba 2.9.0 (fetched for that step, sha256-checked, removed after it) and
# linked into /usr/local/bin. Then this kit and its shared core,
# editable, under /kit (`run.sh install`). No weights are inside: `run.sh install --weights DIR` fetches the six pinned Hugging Face snapshots into
# a directory you mount, and HF_HOME names it at run time (README.md §Setup). Host: an NVIDIA driver that runs CUDA 12.8 (570 or newer) and the
# NVIDIA container toolkit; STACK=img_ef2inv's compiled wheels carry sm_90 code only (H100, configs/h100.env), STACK=img_ef2inv_a100's carry sm_80 and
# sm_90 (A100 80GB under configs/a100.env, and H100) — the prebuilt route exists for img_ef2inv only (the published wheels are the sm_90 pair).
# Build, from the bundle's top directory (the one holding ef2inv/ and common/):
#   docker build -f ef2inv/environment/Dockerfile -t ef2inv-kit:dev .                                   # compiles the two wheels (default; H100)
#   docker build -f ef2inv/environment/Dockerfile -t ef2inv-kit:dev --build-arg WHEELS_FROM=prebuilt .   # the two files in ef2inv/stock/wheels/
#   docker build -f ef2inv/environment/Dockerfile -t ef2inv-kit:dev --build-arg STACK=img_ef2inv_a100 . # A100 80GB (+ H100): compiles the sm_80 + sm_90 wheels
# Run (README.md §Run has one command per mode; outputs land under the mounted out/):
#   docker run --rm --gpus all -v /path/to/hf_home:/weights/hf -e HF_HOME=/weights/hf -v "$PWD/out":/kit/ef2inv/out ef2inv-kit:dev \
#     bash run.sh design --config h100 --mode fast --target-name pd-l1 --binder-len 80 --seed 0 --out out/fast
# To try the kit in an existing environment instead: STOCK.md §Stack repeats the steps below for a Linux + CUDA 12.8 host.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ARG STACK=img_ef2inv
ARG WHEELS_FROM=build
ARG BUILD_JOBS=0

# The C toolchain Triton / inductor need at run time (gcc, libc headers), git (the pinned git installs below and the stock checkout), and curl + CA
# certificates to fetch uv:
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git build-essential && rm -rf /var/lib/apt/lists/*

# uv 0.12.5 — the one installer of this stack (it pins the python-build-standalone release it installs 3.12.14 from), at /root/.local/bin/uv:
RUN curl -LsSf -o /tmp/uv-install.sh https://astral.sh/uv/0.12.5/install.sh \
 && echo "504511fbbbd811aeaba6738abc79408956b6c7da0ca35437b3dcc24a41efc111  /tmp/uv-install.sh" | sha256sum -c - \
 && sh /tmp/uv-install.sh && rm -f /tmp/uv-install.sh && /root/.local/bin/uv --version
ENV PATH=/opt/ef2inv/.venv/bin:/root/.local/bin:${PATH}

# Python 3.12.14 exactly as the stack has it — uv's python-build-standalone build under /opt/uv-python — and the stack's virtual environment
# /opt/ef2inv/.venv (a standalone environment; no pip inside; requested as `3.12`, so its home is uv's minor-version link
# /opt/uv-python/cpython-3.12-linux-x86_64-gnu), then the stock esm checkout /opt/esmfold2/esm at the commit stock/PINS.json `upstream` names
# (d0207ea3; STOCK.md §Pin) — it installs editable right after the lock, below. The checkout's .git directory is removed in the same step: the
# editable install and stock/check_pins.py read the files and the package metadata, not git, so no repository history ships in the image:
COPY ef2inv/environment/requirements.lock ef2inv/stock/PINS.json /tmp/stack/
RUN UV_PYTHON_INSTALL_DIR=/opt/uv-python uv python install 3.12.14 \
 && UV_PYTHON_INSTALL_DIR=/opt/uv-python uv venv --python 3.12 --python-preference only-managed --prompt ef2inv /opt/ef2inv/.venv \
 && python -V \
 && ESM_REPO=$(python -I -c "import json; print(json.load(open('/tmp/stack/PINS.json'))['upstream']['repo'])") \
 && ESM_COMMIT=$(python -I -c "import json; print(json.load(open('/tmp/stack/PINS.json'))['upstream']['commit'])") \
 && git clone -q "$ESM_REPO" /opt/esmfold2/esm && git -C /opt/esmfold2/esm checkout -q "$ESM_COMMIT" && git -C /opt/esmfold2/esm log -1 --format='esm checkout %H' \
 && rm -rf /opt/esmfold2/esm/.git

# The pinned stack (ef2inv/environment/requirements.lock, read with --no-deps: the file is the complete list of index / URL distributions; torch and
# triton by direct URL on download.pytorch.org, everything else from PyPI or the git URL on its line) minus its flash-attn / transformer-engine lines,
# then the editable stock esm checkout (whose cookbook script and editable form stock/check_pins.py requires);
# `--python <the environment>` installs into it with its bin/python3 named in the entry-point scripts:
RUN grep -v -E '^(#|flash-attn==|transformer-engine==)' /tmp/stack/requirements.lock > /tmp/stack/stack.txt \
 && UV_LINK_MODE=copy uv pip install --python /opt/ef2inv/.venv --no-cache --no-deps -r /tmp/stack/stack.txt \
 && UV_LINK_MODE=copy uv pip install --python /opt/ef2inv/.venv --no-cache --no-deps -e /opt/esmfold2/esm

# The two compiled wheels, by route. The CUDA architectures they are compiled for are STACK's, set in one place (the `case` below) and read by
# both builds: FLASH_ATTN_CUDA_ARCHS (flash-attn's setup.py: one -gencode per listed arch, no PTX), NVTE_CUDA_ARCHS (transformer_engine's
# CMAKE_CUDA_ARCHITECTURES for libtransformer_engine.so) and TORCH_CUDA_ARCH_LIST (torch's extension builder) — img_ef2inv: 90 / "9.0";
# img_ef2inv_a100: 80;90 / "8.0;9.0"; any other STACK is refused by name. WHEELS_FROM=build: the CUDA 12.8 compiler and the -dev headers the two
# builds include, from the NVIDIA apt repository the base image configures, and pip / wheel / ninja / cmake / pybind11 / nvidia-cudnn-frontend as
# build tools in the environment — all removed again once the wheels exist (the image keeps runtime weight; the stack stays the lock); flash-attn
# 2.8.3 from its PyPI sdist; transformer_engine from NVIDIA/TransformerEngine at 42b84005 with its submodules, NVTE_FRAMEWORK=pytorch, its
# version's commit tag at the 7 hex digits stock/PINS.json pins (2.15.0+42b8400: core.abbrev), the NCCL expert-parallel extension off (NVTE_WITH_NCCL_EP=0: unused by single-GPU design), cuDNN and NCCL headers and
# libraries (CUDNN_PATH for its CMake, CPATH / LIBRARY_PATH for its torch extension and logging header) = the stack's nvidia-cudnn-cu12 /
# nvidia-nccl-cu12 wheels; both `pip wheel --no-deps --no-build-isolation` against the torch already installed (stock/PINS.json
# pinned_stack.image_recipe). WHEELS_FROM=prebuilt (STACK=img_ef2inv only; refused by name for img_ef2inv_a100, whose wheels are always compiled):
# the files of ef2inv/stock/wheels/, refused by name unless their names and sha256 are stock/PINS.json's. Either way the wheels then install
# --no-deps, and the cu12 pair of cuequivariance-ops / cuequivariance-ops-torch once more so it owns the package directories it shares with the
# cu13 pair (requirements.lock header). The step runs under `set -e` from its first word and its commands are joined with `;`, so any
# failing command — the clone, the checkout of the pinned commit (verified again with `git rev-parse HEAD`), a compile — stops the build:
COPY ef2inv/stock/wheels/ /tmp/wheels/
RUN set -e; case "$STACK" in \
      img_ef2inv)      FLASH_ATTN_CUDA_ARCHS=90;      NVTE_CUDA_ARCHS=90;      TORCH_CUDA_ARCH_LIST="9.0" ;; \
      img_ef2inv_a100) FLASH_ATTN_CUDA_ARCHS="80;90"; NVTE_CUDA_ARCHS="80;90"; TORCH_CUDA_ARCH_LIST="8.0;9.0" ;; \
      *) echo "ef2inv/environment/Dockerfile: STACK=$STACK — the stacks are img_ef2inv (the two wheels compiled for sm_90: H100) and img_ef2inv_a100 (sm_80 and sm_90: A100 80GB, H100)" >&2; exit 2 ;; \
    esac; export FLASH_ATTN_CUDA_ARCHS NVTE_CUDA_ARCHS TORCH_CUDA_ARCH_LIST; \
    if [ "$WHEELS_FROM" = build ]; then \
      apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-nvrtc-dev-12-8 \
        cuda-profiler-api-12-8 cuda-nvml-dev-12-8 libcublas-dev-12-8=${NV_LIBCUBLAS_VERSION} libcurand-dev-12-8 libcusparse-dev-12-8 libcusolver-dev-12-8; \
      /usr/local/cuda-12.8/bin/nvcc --version | tail -1; \
      uv pip install --python /opt/ef2inv/.venv --no-cache pip wheel ninja cmake 'pybind11[global]' nvidia-cudnn-frontend; \
      J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); echo "compile jobs: $J (cores: $(nproc --all))"; \
      export CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=4; rm -f /tmp/wheels/*.whl; \
      echo "building flash-attn 2.8.3 from the PyPI sdist: CUDA archs $FLASH_ATTN_CUDA_ARCHS (STACK=$STACK), MAX_JOBS=$J"; \
      FLASH_ATTENTION_FORCE_BUILD=TRUE \
        python -m pip wheel --no-deps --no-build-isolation --no-binary flash-attn --no-cache-dir -w /tmp/wheels "$(grep -E '^flash-attn==' /tmp/stack/requirements.lock)"; \
      git clone -q https://github.com/NVIDIA/TransformerEngine.git /tmp/src/TransformerEngine; \
      git -C /tmp/src/TransformerEngine checkout -q 42b840051647eef89761a16dfdff87e82bb253ab; \
      git -C /tmp/src/TransformerEngine submodule -q update --init --recursive; \
      test "$(git -C /tmp/src/TransformerEngine rev-parse HEAD)" = 42b840051647eef89761a16dfdff87e82bb253ab; \
      git -C /tmp/src/TransformerEngine config core.abbrev 7; \
      echo "building transformer_engine 2.15.0 at 42b84005: CUDA archs $NVTE_CUDA_ARCHS (STACK=$STACK), MAX_JOBS=$J"; \
      NVTE_FRAMEWORK=pytorch NVTE_WITH_NCCL_EP=0 NVTE_BUILD_MAX_JOBS=$J \
      CUDNN_PATH=/opt/ef2inv/.venv/lib/python3.12/site-packages/nvidia/cudnn \
      CPATH=/opt/ef2inv/.venv/lib/python3.12/site-packages/nvidia/cudnn/include:/opt/ef2inv/.venv/lib/python3.12/site-packages/nvidia/nccl/include LIBRARY_PATH=/opt/ef2inv/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/ef2inv/.venv/lib/python3.12/site-packages/nvidia/nccl/lib \
        python -m pip wheel --no-deps --no-build-isolation --no-cache-dir -w /tmp/wheels /tmp/src/TransformerEngine; \
      sha256sum /tmp/wheels/*.whl; \
      uv pip uninstall --python /opt/ef2inv/.venv pip wheel ninja cmake pybind11 pybind11-global nvidia-cudnn-frontend; \
      apt-get purge -y --auto-remove cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-nvrtc-dev-12-8 cuda-profiler-api-12-8 cuda-nvml-dev-12-8 \
        libcublas-dev-12-8 libcurand-dev-12-8 libcusparse-dev-12-8 libcusolver-dev-12-8; \
      rm -rf /var/lib/apt/lists/* /tmp/src /root/.cache; \
    elif [ "$STACK" != img_ef2inv ]; then echo "ef2inv/environment/Dockerfile: WHEELS_FROM=$WHEELS_FROM with STACK=$STACK — published wheel files exist for STACK=img_ef2inv only (stock/PINS.json pinned_stack.image_recipe: the sm_90 pair); this stack's wheels are compiled: WHEELS_FROM=build (the default)" >&2; exit 2; \
    else python -I -c "import hashlib, json, os, sys; d = '/tmp/wheels'; r = json.load(open('/tmp/stack/PINS.json'))['pinned_stack']['image_recipe']; want = {r['flash_attn_wheel']['file']: r['flash_attn_wheel']['sha256'], **r['transformer_engine_wheels']['sha256']}; have = sorted(f for f in os.listdir(d) if f.endswith('.whl')); got = {f: hashlib.sha256(open(os.path.join(d, f), 'rb').read()).hexdigest() for f in have}; bad = ['missing ef2inv/stock/wheels/' + f + ' (README.md Install)' for f in want if f not in got] + ['unexpected ef2inv/stock/wheels/' + f for f in got if f not in want] + ['ef2inv/stock/wheels/%s: sha256 %s != stock/PINS.json %s' % (f, got[f], want[f]) for f in want if f in got and got[f] != want[f]]; print('\\n'.join(bad or ['wheel OK %s sha256=%s' % (f, got[f]) for f in have])); sys.exit(3 if bad else 0)"; fi; \
    UV_LINK_MODE=copy uv pip install --python /opt/ef2inv/.venv --no-cache --no-deps /tmp/wheels/*.whl; \
    UV_LINK_MODE=copy uv pip install --python /opt/ef2inv/.venv --no-cache --no-deps --reinstall \
      $(grep -E '^cuequivariance-ops(-torch)?-cu12==' /tmp/stack/requirements.lock); \
    python -c "import torch, triton, transformers, importlib.metadata as md; print('torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__, 'esm', md.version('esm'), 'flash_attn', md.version('flash_attn'), 'transformer_engine', md.version('transformer_engine'))"; \
    rm -rf /tmp/stack /tmp/wheels

# HMMER 3.4 (hmmer.org) — the profile-HMM search programs. The cookbook scores an antibody-framework binder's CDR contacts through abnumber -> ANARCI
# -> `hmmscan` (`_is_antibody_sequence`, `_cdr_indices`, reached by `--binder-name <antibody>_framework_vhvl` or `--is-antibody 1`); abnumber 0.4.4 and
# ANARCI 2026.2.13.2 are distributions of the lock above, HMMER is C programs with no Python distribution ANARCI can call, so it comes as the conda
# packages of ef2inv/environment/conda-hmmer.lock: bioconda's hmmer 3.4 build and the conda-forge libraries it links — the set the cookbook's own
# `hmmer=3.4` conda line resolves —, every package pinned by URL and md5, installed by micromamba 2.9.0 (a static binary fetched for this step from its
# release page, sha256-checked, removed after it together with its package cache) into the prefix /opt/hmmer, outside the Python environment. HMMER's
# programs are then linked into /usr/local/bin — on PATH after the environment's own bin; nothing installed above moves. The last command numbers a
# heavy-chain variable domain through the whole route (abnumber -> ANARCI -> hmmscan) and fails the build if it cannot.
COPY ef2inv/environment/conda-hmmer.lock /tmp/hmmer/conda-hmmer.lock
RUN set -eu; \
    curl -LsSf -o /tmp/hmmer/micromamba https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64; \
    echo "366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3  /tmp/hmmer/micromamba" | sha256sum -c -; chmod +x /tmp/hmmer/micromamba; \
    MAMBA_ROOT_PREFIX=/tmp/hmmer/root /tmp/hmmer/micromamba create -y -q -p /opt/hmmer -f /tmp/hmmer/conda-hmmer.lock; \
    for p in alimask hmmalign hmmbuild hmmconvert hmmemit hmmfetch hmmlogo hmmpgmd hmmpgmd_shard hmmpress hmmscan hmmsearch hmmsim hmmstat jackhmmer makehmmerdb nhmmer nhmmscan phmmer; do \
      [ -x /opt/hmmer/bin/$p ] || { echo "ef2inv/environment/Dockerfile: /opt/hmmer/bin/$p is missing from the conda-hmmer.lock install" >&2; exit 3; }; ln -s /opt/hmmer/bin/$p /usr/local/bin/$p; \
    done; \
    rm -rf /tmp/hmmer /root/.mamba /root/.conda; \
    command -v hmmscan; hmmscan -h | sed -n 2p; \
    python -c "import abnumber, anarci; from abnumber import Chain; c = Chain('EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS', scheme='chothia'); assert (c.chain_type, c.cdr3_seq) == ('H', 'WGGDGFYAMDY'), (c.chain_type, c.cdr3_seq); print('antibody numbering OK: abnumber', abnumber.__version__, '| ANARCI', anarci.__version__, '| heavy chain, Chothia CDR3', c.cdr3_seq)"

# This kit and its shared core, editable, on top (`run.sh install`: the two installs, then the stock software pins checked):
COPY common/opt_core /kit/common/opt_core
COPY ef2inv /kit/ef2inv
WORKDIR /kit/ef2inv
RUN rm -f stock/wheels/*.whl && bash run.sh install
# [install]-jitcache: optional pre-filled compile caches (build is identical when no tar is present)
COPY ef2inv/environment/Dockerfile _jitcache/ef2inv-*-jit.ta[r] /tmp/jit/
RUN set -e; mkdir -p /opt/jit_cache; \
    for t in /tmp/jit/*-jit.tar; do [ -f "$t" ] && tar -xf "$t" -C /opt/jit_cache; done; \
    chmod -R a+rX,u+w /opt/jit_cache; rm -rf /tmp/jit

# The process environment of the tested stack: hash seed, C flags for the JIT-built launchers, the CUDA library path transformer_engine resolves
# libcublas / libcudart through, xformers accepting this flash-attn build, transformer_engine's framework binding (stock/PINS.json image_recipe
# image_env). HF_HOME is the operator's (-e at run time); the configs set the offline switches.
ENV PYTHONHASHSEED=0 \
    CFLAGS=-g0 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 \
    NVTE_FRAMEWORK=pytorch
