#!/bin/bash
# ESMFold2 binder-design kit — build the pinned stack's two CUDA-extension wheels for an environment of your own (README Setup, route C).
# No index carries flash-attn 2.8.3 or transformer_engine 2.15.0+42b8400 built for torch 2.11.0+cu128; environment/Dockerfile compiles them at
# image build (its WHEELS_FROM=build step), and this script runs that same step in the current Python environment: same sources (the flash-attn
# sdist checked by sha256 against stock/PINS.json, TransformerEngine checked out at the pinned commit with its submodules), same build settings,
# wheels written to ef2inv/stock/wheels/ and installed from there with --no-index --no-deps, then `pip check`.
#
# usage:  bash environment/build_wheels.sh [--stack img_ef2inv|img_ef2inv_a100] [--jobs N]      (inside the environment; any working directory)
#   --stack  img_ef2inv (default): compute capability 9.0 (H100 / H200) · img_ef2inv_a100: 8.0 and 9.0 (the A100 as well)
#   --jobs   parallel compile jobs (default: every core; each nvcc job wants a few GB of RAM, so flash-attn is additionally capped at one job
#            per 9 GB of available memory)
# Needs, before it starts (each is checked and refused by name, exit 3): the environment's python with torch 2.11.0+cu128 and the nvidia-nccl /
# nvidia-cudnn wheels of environment/requirements.lock installed; the CUDA 12.8 toolkit's nvcc on PATH (CUDA_HOME honoured); git; a C++ compiler.
# Network: github.com (TransformerEngine and its submodules) and the Python index (the flash-attn sdist, five build tools). The compile is long
# and memory-hungry; the Dockerfile header describes the same trade-offs. Afterwards export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 and
# NVTE_FRAMEWORK=pytorch in the shells that run the kit (STOCK.md §Stack). Exit codes: 0 built and installed · 1 a build or the install failed ·
# 2 usage · 3 a prerequisite is missing.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)          # ef2inv/
STACK=img_ef2inv; JOBS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stack) STACK=${2:-}; shift 2 ;;
    --jobs)  JOBS=${2:-0}; shift 2 ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "build_wheels.sh: unknown argument $1 (usage: bash environment/build_wheels.sh [--stack img_ef2inv|img_ef2inv_a100] [--jobs N])" >&2; exit 2 ;;
  esac
done
case "$STACK" in
  img_ef2inv)      TORCH_CUDA_ARCH_LIST="9.0" ;;
  img_ef2inv_a100) TORCH_CUDA_ARCH_LIST="8.0;9.0" ;;
  *) echo "build_wheels.sh: no such stack: $STACK (img_ef2inv: compute capability 9.0, H100 / H200 | img_ef2inv_a100: 8.0 and 9.0, the A100 as well)" >&2; exit 2 ;;
esac
export TORCH_CUDA_ARCH_LIST; SM_ARCHS=$(printf %s "$TORCH_CUDA_ARCH_LIST" | tr -d .)     # 90 | 80;90 — FLASH_ATTN_CUDA_ARCHS / NVTE_CUDA_ARCHS
# ---- prerequisites, refused by name
refuse() { echo "build_wheels.sh: NOT BUILT: $1" >&2; exit 3; }
command -v python >/dev/null || refuse "no python on PATH (activate the environment that holds the pinned stack first)"
command -v git >/dev/null    || refuse "git is not on PATH (TransformerEngine is cloned at its pinned commit)"
command -v g++ >/dev/null || command -v c++ >/dev/null || refuse "no C++ compiler on PATH"
TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null) || refuse "torch does not import in $(command -v python) — install environment/requirements.lock minus its flash-attn / transformer-engine lines first (STOCK.md §Stack)"
[ "$TORCH" = "2.11.0+cu128" ] || refuse "torch is $TORCH; the wheels are built against the pinned torch 2.11.0+cu128 (environment/requirements.lock)"
if [ -n "${CUDA_HOME:-}" ]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
command -v nvcc >/dev/null   || refuse "nvcc is not on PATH — install the CUDA 12.8 toolkit (or set CUDA_HOME to it)"
NVCC_REL=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
case "$NVCC_REL" in 12.*) ;; *) refuse "nvcc is release ${NVCC_REL:-unknown}; the pinned stack is CUDA 12.8 (torch 2.11.0+cu128)" ;; esac
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
NCCL_HOME=$(python -c "import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])" 2>/dev/null) || refuse "the nvidia-nccl wheel of environment/requirements.lock is not installed (TransformerEngine builds against its headers)"
CUDNN_HOME=$(python -c "import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])" 2>/dev/null) || refuse "the nvidia-cudnn wheel of environment/requirements.lock is not installed (TransformerEngine builds against its headers)"
[ -f "$NCCL_HOME/include/nccl.h" ]   || refuse "$NCCL_HOME/include/nccl.h is missing"
[ -f "$CUDNN_HOME/include/cudnn.h" ] || refuse "$CUDNN_HOME/include/cudnn.h is missing"
FA_SDIST_SHA=$(python -I -c "import json, re, sys; s = json.load(open(sys.argv[1]))['pinned_stack']['image_recipe']['flash_attn_wheel']['source']; m = re.search('[0-9a-f]{64}', s); print(m.group(0) if m else '')" "$HERE/stock/PINS.json")
[ -n "$FA_SDIST_SHA" ] || refuse "stock/PINS.json pinned_stack.image_recipe.flash_attn_wheel.source carries no sdist sha256"
# ---- the build: the Dockerfile's WHEELS_FROM=build step, in this environment
W="$HERE/stock/wheels"; mkdir -p "$W"
SRC=$(mktemp -d "${TMPDIR:-/tmp}/ef2inv_wheels.XXXXXX"); trap 'rm -rf "$SRC"' EXIT
T0=$(date +%s); J=$JOBS; [ "$J" -gt 0 ] || J=$(nproc --all)
MEM_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); JFA=$(( MEM_GB / 9 )); [ "$JFA" -ge 1 ] || JFA=1; [ "$JFA" -le "$J" ] || JFA=$J
echo "ef2inv wheels (STACK=$STACK): building flash_attn and transformer_engine for compute capability $TORCH_CUDA_ARCH_LIST with $J jobs (${MEM_GB} GB available; flash-attn capped at $JFA jobs); nvcc: $(nvcc --version | tail -1); torch $TORCH; wheels to $W"
export MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=4 CFLAGS=${CFLAGS:--g0} NVTE_FRAMEWORK=pytorch XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
python -m pip install --no-cache-dir -c <(grep -E '^[A-Za-z0-9_.-]+(\[[^]]*\])?==' "$HERE/environment/requirements.lock" | grep -v -E '^(flash-attn|transformer-engine)==') \
  wheel ninja cmake 'pybind11[global]' nvidia-cudnn-frontend        # build tools only; -c holds every pinned package (cuda-bindings, cuda-pathfinder, …) at the lock's version
python -m pip download --no-cache-dir --no-binary :all: --no-deps --no-build-isolation -d "$SRC" flash-attn==2.8.3
echo "$FA_SDIST_SHA  $SRC/flash_attn-2.8.3.tar.gz" | sha256sum -c -
T1=$(date +%s); ( cd "$SRC" && MAX_JOBS=$JFA FLASH_ATTN_CUDA_ARCHS="$SM_ARCHS" FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$W" "$SRC/flash_attn-2.8.3.tar.gz" ); echo "flash_attn built in $(( $(date +%s) - T1 ))s"
git clone --quiet https://github.com/NVIDIA/TransformerEngine.git "$SRC/TransformerEngine"
git -C "$SRC/TransformerEngine" checkout --quiet 42b840051647eef89761a16dfdff87e82bb253ab; git -C "$SRC/TransformerEngine" submodule --quiet update --init --recursive
git -C "$SRC/TransformerEngine" config core.abbrev 7; test "$(git -C "$SRC/TransformerEngine" rev-parse --short HEAD)" = 42b8400
T1=$(date +%s); ( cd "$SRC/TransformerEngine" && export CUDNN_PATH="$CUDNN_HOME" CPATH="$CUDNN_HOME/include:$NCCL_HOME/include" LIBRARY_PATH="$NCCL_HOME/lib:$CUDNN_HOME/lib" && NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="$SM_ARCHS" NVTE_WITH_NCCL_EP=0 NVTE_BUILD_MAX_JOBS=$J python -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$W" . ); echo "transformer_engine built in $(( $(date +%s) - T1 ))s"
python -m pip uninstall -y cmake ninja pybind11 pybind11-global nvidia-cudnn-frontend
ls -l "$W"
python -m pip install --no-cache-dir --no-index --no-deps "$W"/flash_attn-2.8.3-*.whl "$W"/transformer_engine-2.15.0+42b8400-*.whl
python -m pip check
echo "ef2inv wheels (STACK=$STACK, compute capability $TORCH_CUDA_ARCH_LIST): built and installed in $(( $(date +%s) - T0 ))s on $(nproc --all) cores; next: export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 NVTE_FRAMEWORK=pytorch; cd ef2inv && ./run.sh install [--weights DIR]"
