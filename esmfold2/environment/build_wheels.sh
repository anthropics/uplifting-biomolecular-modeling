#!/bin/bash
# ESMFold2 kit — build the pinned stack's three CUDA-extension wheels for an environment of your own (README Setup, route C).
# No index carries xformers 0.0.35+03b91d7, transformer_engine 2.15.0+42b8400 or flash-attn 2.8.3.post1 built for torch 2.13.0+cu130;
# environment/Dockerfile compiles them at image build (its WHEELS_FROM=build step), and this script runs that same step in the current
# Python environment: same sources (git tags checked to the commit, the flash-attn sdist checked by sha256), same build settings, wheels
# written to esmfold2/stock/wheels/ and installed from there with --no-index --no-deps, then `pip check`.
#
# usage:  bash environment/build_wheels.sh [--stack img_ef2_fa|img_esmfold2_a100] [--jobs N]      (from esmfold2/, inside the environment)
#   --stack  img_ef2_fa (default): compute capability 9.0 (H100 / H200) · img_esmfold2_a100: 8.0 and 9.0 (A100 80GB as well)
#   --jobs   parallel compile jobs (default: every core; flash-attn is additionally capped at one nvcc job per 9 GB of available memory)
#   TORCH_CUDA_ARCH_LIST=8.0 (exported) builds for that compute capability alone — a route-C host needs only its own card; unset builds the stack's image list
# Needs, before it starts (each is checked and refused by name, exit 3): the environment's python with torch 2.13.0+cu130 and the
# nvidia-nccl / nvidia-cudnn wheels of environment/requirements.lock installed; the CUDA 13.0 toolkit's nvcc on PATH (CUDA_HOME honoured);
# git; a C++ compiler. Network: github.com (xformers, TransformerEngine) and the Python index (flash-attn sdist, four build tools).
# The compile is long — most of it flash-attn — and memory-hungry; the Dockerfile header describes the same trade-offs. Exit codes:
# 0 built and installed · 1 a build or the install failed · 2 usage · 3 a prerequisite is missing.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)          # esmfold2/
STACK=img_ef2_fa; JOBS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stack) STACK=${2:-}; shift 2 ;;
    --jobs)  JOBS=${2:-0}; shift 2 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "build_wheels.sh: unknown argument $1 (usage: bash environment/build_wheels.sh [--stack img_ef2_fa|img_esmfold2_a100] [--jobs N])" >&2; exit 2 ;;
  esac
done
case "$STACK" in                                                  # an exported TORCH_CUDA_ARCH_LIST (e.g. 8.0 on an A100 host) narrows the build to that list; unset = the image's list for the stack
  img_ef2_fa)        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" ;;
  img_esmfold2_a100) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}" ;;
  *) echo "build_wheels.sh: no such stack: $STACK (img_ef2_fa: compute capability 9.0, H100 / H200 | img_esmfold2_a100: 8.0 and 9.0, A100 80GB as well)" >&2; exit 2 ;;
esac
printf %s "$TORCH_CUDA_ARCH_LIST" | grep -Eq '^[0-9]+\.[0-9]+a?(;[0-9]+\.[0-9]+a?)*$' || { echo "build_wheels.sh: TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST is not a list like 8.0 or 8.0;9.0 (unset it to build the stack's own list)" >&2; exit 2; }
export TORCH_CUDA_ARCH_LIST; SM_ARCHS=$(printf %s "$TORCH_CUDA_ARCH_LIST" | tr -d .)

# ---- prerequisites, refused by name
refuse() { echo "build_wheels.sh: NOT BUILT: $1" >&2; exit 3; }
command -v python >/dev/null || refuse "no python on PATH (activate the environment that holds the pinned stack first)"
command -v git >/dev/null    || refuse "git is not on PATH (xformers and TransformerEngine are cloned at their release tags)"
command -v g++ >/dev/null || command -v c++ >/dev/null || refuse "no C++ compiler on PATH"
TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null) || refuse "torch does not import in $(command -v python) — install environment/requirements.lock first (STOCK.md §Stack)"
[ "$TORCH" = "2.13.0+cu130" ] || refuse "torch is $TORCH; the wheels are built against the pinned torch 2.13.0+cu130 (environment/requirements.lock)"
if [ -n "${CUDA_HOME:-}" ]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
command -v nvcc >/dev/null   || refuse "nvcc is not on PATH — install the CUDA 13.0 toolkit (or set CUDA_HOME to it)"
NVCC_REL=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
case "$NVCC_REL" in 13.*) ;; *) refuse "nvcc is release ${NVCC_REL:-unknown}; the pinned stack is CUDA 13.0 (torch 2.13.0+cu130)" ;; esac
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
NCCL_HOME=$(python -c "import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])" 2>/dev/null) || refuse "the nvidia-nccl wheel of environment/requirements.lock is not installed (TransformerEngine builds against its headers)"
CUDNN_HOME=$(python -c "import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])" 2>/dev/null) || refuse "the nvidia-cudnn wheel of environment/requirements.lock is not installed (TransformerEngine builds against its headers)"
[ -f "$NCCL_HOME/include/nccl.h" ]   || refuse "$NCCL_HOME/include/nccl.h is missing"
[ -f "$CUDNN_HOME/include/cudnn.h" ] || refuse "$CUDNN_HOME/include/cudnn.h is missing"

# ---- the build: the Dockerfile's WHEELS_FROM=build step, in this environment
W="$HERE/stock/wheels"; mkdir -p "$W"
SRC=$(mktemp -d "${TMPDIR:-/tmp}/esmfold2_wheels.XXXXXX"); trap 'rm -rf "$SRC"' EXIT
T0=$(date +%s); J=$JOBS; [ "$J" -gt 0 ] || J=$(nproc --all)
MEM_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); JFA=$(( MEM_GB / 9 )); [ "$JFA" -ge 1 ] || JFA=1; [ "$JFA" -le "$J" ] || JFA=$J
echo "esmfold2 wheels (STACK=$STACK): building xformers, transformer_engine, flash_attn for compute capability $TORCH_CUDA_ARCH_LIST with $J jobs (${MEM_GB} GB available; flash-attn capped at $JFA jobs); nvcc: $(nvcc --version | tail -1)"
export MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=2 CFLAGS=${CFLAGS:--g0} NVTE_FRAMEWORK=pytorch XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
python -m pip install --no-cache-dir cmake==4.4.3 ninja 'pybind11[global]==3.1.0' nvidia-cudnn-frontend==1.28.0

git clone --quiet --depth 1 --branch v0.0.35 --recurse-submodules --shallow-submodules https://github.com/facebookresearch/xformers.git "$SRC/xformers"
test "$(git -C "$SRC/xformers" rev-parse HEAD)" = 03b91d7d9ff295ae68a320e2e733dd6c2ef8f342
T1=$(date +%s); ( cd "$SRC/xformers" && BUILD_VERSION=0.0.35+03b91d7.d20260904 FORCE_CUDA=1 XFORMERS_BUILD_TYPE=Release python -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$W" . ); echo "xformers built in $(( $(date +%s) - T1 ))s"

git clone --quiet --depth 1 --branch v2.15 --recurse-submodules --shallow-submodules https://github.com/NVIDIA/TransformerEngine.git "$SRC/TransformerEngine"
test "$(git -C "$SRC/TransformerEngine" rev-parse HEAD)" = 42b840051647eef89761a16dfdff87e82bb253ab; git -C "$SRC/TransformerEngine" config core.abbrev 7; test "$(git -C "$SRC/TransformerEngine" rev-parse --short HEAD)" = 42b8400
T1=$(date +%s); ( cd "$SRC/TransformerEngine" && export CPATH="$NCCL_HOME/include:$CUDNN_HOME/include" LIBRARY_PATH="$NCCL_HOME/lib" && NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="$SM_ARCHS" NVTE_WITH_NCCL_EP=0 NVTE_BUILD_MAX_JOBS=$J CUDNN_PATH=$CUDNN_HOME python -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$W" . ); echo "transformer_engine built in $(( $(date +%s) - T1 ))s"

python -m pip download --no-cache-dir --no-binary :all: --no-deps --no-build-isolation -d "$SRC" flash-attn==2.8.3.post1
echo "55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348  $SRC/flash_attn-2.8.3.post1.tar.gz" | sha256sum -c -
T1=$(date +%s); ( cd "$SRC" && MAX_JOBS=$JFA FLASH_ATTN_CUDA_ARCHS="$SM_ARCHS" FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$W" "$SRC/flash_attn-2.8.3.post1.tar.gz" ); echo "flash_attn built in $(( $(date +%s) - T1 ))s"

python -m pip uninstall -y cmake ninja pybind11 pybind11-global nvidia-cudnn-frontend
ls -l "$W"
python -m pip install --no-cache-dir --no-index --no-deps "$W"/xformers-*.whl "$W"/transformer_engine-*.whl "$W"/flash_attn-*.whl
python -m pip check
echo "esmfold2 wheels (STACK=$STACK, compute capability $TORCH_CUDA_ARCH_LIST): built and installed in $(( $(date +%s) - T0 ))s on $(nproc --all) cores; next: ./run.sh install [--weights DIR]"
