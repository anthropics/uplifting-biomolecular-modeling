#!/bin/bash
# build.sh — build edm_ops.so (this kit's TensorFlow GPU ops) from edm_ops.cc + edm_softmax.cc + edm_ops.cu.cc + edm_layernorm.cu.cc + edm_pool_logits.cu.cc + edm_softmax.cu.cc + edm_biasact.cc + edm_biasact.cu.cc against the TensorFlow installed in the
# python on PATH, and write BUILD.json (toolchain, flags, source and object digests) and SHA256SUMS (the object's line; the loader re-hashes
# the library against it) beside it.
#   bash build.sh              build ./edm_ops.so, ./BUILD.json and ./SHA256SUMS, then load the library and list the registered ops
# Environment: PYTHON=<interpreter> (default python3) · NVCC=<path> (default: the pip package nvidia-cuda-nvcc-cu12's nvcc, else nvcc on PATH,
#   else $CUDA_HOME/bin/nvcc) · CXX=<g++> (default g++) · CUDA_HOME (headers / libcudart when the nvidia-cuda-runtime-cu12 pip package is absent)
# Requires tensorflow 2.17.1 (the version the ops are written for; EDM_OPS_ALLOW_TF=<version> builds against another one at your own risk),
# nvcc from CUDA 12.x (TensorFlow 2.17.1 is built with CUDA 12.3) and g++ with C++17. No GPU is needed to build.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
CXX="${CXX:-g++}"
TAG="[enformer-deepmind-opt] build.sh:"
die() { echo "$TAG $*" >&2; exit 3; }

# ---- the stack: TensorFlow's own compile / link flags, the CUDA compiler and runtime of its pip packages -------------------------------
eval "$("$PY" - <<'EOF'
import importlib.util, os, shlex, shutil, sys
try:
    import tensorflow as tf
except Exception as e:  # noqa
    print(f'die "tensorflow is not importable by {sys.executable}: {type(e).__name__}: {e}"'); sys.exit(0)
def pkg_dir(name):
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return ""
    if spec is None or not spec.submodule_search_locations:
        return ""
    return list(spec.submodule_search_locations)[0]
nvcc_pkg, rt_pkg = pkg_dir("nvidia.cuda_nvcc"), pkg_dir("nvidia.cuda_runtime")
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
nvcc = os.environ.get("NVCC") or ""
if not nvcc:
    for cand in ([os.path.join(nvcc_pkg, "bin", "nvcc")] if nvcc_pkg else []) + [shutil.which("nvcc") or "", os.path.join(cuda_home, "bin", "nvcc")]:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            nvcc = cand; break
incs = []                                   # cuda_runtime.h + crt/host_config.h: the runtime wheel has the first, the nvcc wheel the second; a toolkit has both
for cand in ([os.path.join(rt_pkg, "include")] if rt_pkg else []) + ([os.path.join(nvcc_pkg, "include")] if nvcc_pkg else []) + [os.path.join(cuda_home, "include")]:
    have = lambda rel: any(os.path.isfile(os.path.join(d, rel)) for d in incs)  # noqa: E731
    if os.path.isdir(cand) and not (have("cuda_runtime.h") and have("crt/host_config.h")):
        if (os.path.isfile(os.path.join(cand, "cuda_runtime.h")) and not have("cuda_runtime.h")) or \
           (os.path.isfile(os.path.join(cand, "crt/host_config.h")) and not have("crt/host_config.h")):
            incs.append(cand)
have = lambda rel: any(os.path.isfile(os.path.join(d, rel)) for d in incs)  # noqa: E731
inc = " ".join(incs) if have("cuda_runtime.h") and have("crt/host_config.h") else ""
lib = ""
for cand in ([os.path.join(rt_pkg, "lib")] if rt_pkg else []) + [os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib")]:
    if os.path.isfile(os.path.join(cand, "libcudart.so.12")):
        lib = cand; break
q = shlex.quote
print(f"TF_VERSION={q(tf.__version__)}")
print("TF_CFLAGS=(" + " ".join(q(f) for f in tf.sysconfig.get_compile_flags()) + ")")
print("TF_LFLAGS=(" + " ".join(q(f) for f in tf.sysconfig.get_link_flags()) + ")")
print(f"NVCC={q(nvcc)}"); print("CUDA_INCS=(" + " ".join(q(d) for d in incs) + ")"); print(f"CUDA_INC={q(inc)}"); print(f"CUDART_LIB={q(lib)}")
print(f"PY_VERSION={q(sys.version.split()[0])}")
EOF
)"
WANT_TF="2.17.1"
if [ "$TF_VERSION" != "$WANT_TF" ] && [ "${EDM_OPS_ALLOW_TF:-}" != "$TF_VERSION" ]; then
  die "tensorflow $TF_VERSION found; these ops are written for tensorflow $WANT_TF (EDM_OPS_ALLOW_TF=$TF_VERSION overrides)"
fi
[ -n "$NVCC" ] || die "nvcc not found (pip install nvidia-cuda-nvcc-cu12==12.3.107, or set NVCC / CUDA_HOME)"
[ -n "$CUDA_INC" ] || die "cuda_runtime.h + crt/host_config.h not found (pip install nvidia-cuda-runtime-cu12==12.3.101 nvidia-cuda-nvcc-cu12==12.3.107, or set CUDA_HOME)"
[ -n "$CUDART_LIB" ] || die "libcudart.so.12 not found (pip install nvidia-cuda-runtime-cu12==12.3.101, or set CUDA_HOME)"
command -v "$CXX" >/dev/null || die "$CXX not found (a C++17 host compiler is required)"
NVCC_VERSION="$("$NVCC" --version | grep -i release | sed 's/^ *//')"
case "$NVCC_VERSION" in *"release 12."*) ;; *) die "nvcc is '$NVCC_VERSION'; CUDA 12.x is required (TensorFlow $WANT_TF is built with CUDA 12.3)";; esac
case "$NVCC_VERSION" in *"release 12.3"*) ;; *) echo "$TAG note: nvcc is '$NVCC_VERSION', not CUDA 12.3 — its libdevice provides the expf the kernels call" >&2;; esac
CXX_VERSION="$("$CXX" --version | head -1)"
ABI_FLAGS=(); for f in "${TF_CFLAGS[@]}"; do case "$f" in -D_GLIBCXX_USE_CXX11_ABI=*|-DEIGEN_MAX_ALIGN_BYTES=*) ABI_FLAGS+=("$f");; esac; done
echo "$TAG tensorflow $TF_VERSION · $NVCC_VERSION · $CXX_VERSION"
echo "$TAG nvcc=$NVCC cuda_include=${CUDA_INCS[*]} cudart=$CUDART_LIB"

# ---- compile ---------------------------------------------------------------------------------------------------------------------------
GENCODE=(-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90)
NVCC_FLAGS=(-std=c++17 -O3 --ftz=true --prec-div=true --prec-sqrt=true --fmad=true "${GENCODE[@]}" -Xcompiler -fPIC -Xcompiler -O3 "${ABI_FLAGS[@]}")
CXX_FLAGS=(-std=c++17 -O3 -fPIC -DGOOGLE_CUDA=1 -DNDEBUG "${TF_CFLAGS[@]}")
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
set -x
INC_FLAGS=(-I"$HERE"); for d in "${CUDA_INCS[@]}"; do INC_FLAGS+=(-I"$d"); done
"$NVCC" -ccbin "$CXX" -x cu -c "$HERE/edm_ops.cu.cc" -o "$TMP/edm_ops.cu.o" "${INC_FLAGS[@]}" "${NVCC_FLAGS[@]}"
"$NVCC" -ccbin "$CXX" -x cu -c "$HERE/edm_layernorm.cu.cc" -o "$TMP/edm_layernorm.cu.o" "${INC_FLAGS[@]}" "${NVCC_FLAGS[@]}"
NVCC_FLAGS_GEMM=("${NVCC_FLAGS[@]/--ftz=true/--ftz=false}")                          # the GEMM: fused multiply-adds and adds only; denormals kept as the stock float32 GEMM keeps them
"$NVCC" -ccbin "$CXX" -x cu -c "$HERE/edm_pool_logits.cu.cc" -o "$TMP/edm_pool_logits.cu.o" "${INC_FLAGS[@]}" "${NVCC_FLAGS_GEMM[@]}"
"$NVCC" -ccbin "$CXX" -x cu -c "$HERE/edm_softmax.cu.cc" -o "$TMP/edm_softmax.cu.o" "${INC_FLAGS[@]}" "${NVCC_FLAGS[@]}"
"$NVCC" -ccbin "$CXX" -x cu -c "$HERE/edm_biasact.cu.cc" -o "$TMP/edm_biasact.cu.o" "${INC_FLAGS[@]}" "${NVCC_FLAGS[@]}"
"$CXX" -c "$HERE/edm_ops.cc" -o "$TMP/edm_ops.o" "${INC_FLAGS[@]}" "${CXX_FLAGS[@]}"
"$CXX" -c "$HERE/edm_softmax.cc" -o "$TMP/edm_softmax.o" "${INC_FLAGS[@]}" "${CXX_FLAGS[@]}"
"$CXX" -c "$HERE/edm_biasact.cc" -o "$TMP/edm_biasact.o" "${INC_FLAGS[@]}" "${CXX_FLAGS[@]}"
"$CXX" -shared -o "$TMP/edm_ops.so" "$TMP/edm_ops.o" "$TMP/edm_softmax.o" "$TMP/edm_biasact.o" "$TMP/edm_ops.cu.o" "$TMP/edm_layernorm.cu.o" "$TMP/edm_pool_logits.cu.o" "$TMP/edm_softmax.cu.o" "$TMP/edm_biasact.cu.o" "${TF_LFLAGS[@]}" -L"$CUDART_LIB" -l:libcudart.so.12
{ set +x; } 2>/dev/null
mv -f "$TMP/edm_ops.so" "$HERE/edm_ops.so"

# ---- BUILD.json + SHA256SUMS ---------------------------------------------------------------------------------------------------------
export HERE TF_VERSION PY_VERSION NVCC NVCC_VERSION CXX_VERSION CUDA_INC
export NVCC_FLAGS_S="${NVCC_FLAGS[*]}" CXX_FLAGS_S="${CXX_FLAGS[*]}"
"$PY" - <<'EOF'
import datetime, hashlib, json, os
here = os.environ["HERE"]
def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
so = os.path.join(here, "edm_ops.so")
build = {
    "what": "edm_ops.so — the enformer_deepmind kit's TensorFlow GPU ops (EdmScaleShiftGelu, EdmSoftmaxPool2, EdmPoolLogits, EdmBiasResidual, EdmBiasScaleShiftGelu, EdmBiasGelu, EdmSoftmaxPool2Gelu, EdmBias2Residual, EdmLayerNorm, EdmRelShiftSoftmax, EdmBiasAct, EdmQScaleBias), built by build.sh",
    "tensorflow": os.environ["TF_VERSION"],
    "python": os.environ["PY_VERSION"],
    "nvcc": os.environ["NVCC_VERSION"],
    "cxx": os.environ["CXX_VERSION"],
    "cuda_include": os.environ["CUDA_INC"].split(),
    "archs": [[8, 0], [9, 0]],
    "ptx": [[9, 0]],
    "build": "sm_80+sm_90",
    "flags": {"nvcc": os.environ["NVCC_FLAGS_S"].split(), "cxx": os.environ["CXX_FLAGS_S"].split()},
    "ops": ["EdmScaleShiftGelu", "EdmSoftmaxPool2", "EdmPoolLogits", "EdmBiasResidual", "EdmBiasScaleShiftGelu", "EdmBiasGelu", "EdmSoftmaxPool2Gelu", "EdmBias2Residual", "EdmLayerNorm", "EdmRelShiftSoftmax", "EdmBiasAct", "EdmQScaleBias"],
    "sources": {n: digest(os.path.join(here, n)) for n in ("edm_ops.cc", "edm_softmax.cc", "edm_biasact.cc", "edm_ops.cu.cc", "edm_layernorm.cu.cc", "edm_pool_logits.cu.cc", "edm_softmax.cu.cc", "edm_biasact.cu.cc", "edm_ops.h", "edm_f32.cuh")},
    "object": {"file": "edm_ops.so", "sha256": digest(so), "bytes": os.path.getsize(so)},
    "built": datetime.date.today().isoformat(),
}
with open(os.path.join(here, "BUILD.json"), "w") as f:
    json.dump(build, f, indent=1)
    f.write("\n")
with open(os.path.join(here, "SHA256SUMS"), "w") as f:              # the line ops.load() holds the library to before TensorFlow maps it
    f.write(build["object"]["sha256"] + "  edm_ops.so\n")
print("[enformer-deepmind-opt] build.sh: wrote", so, build["object"]["sha256"], build["object"]["bytes"], "bytes; BUILD.json; SHA256SUMS")
EOF

# ---- load it: every op registers (no GPU needed) ---------------------------------------------------------------------------------------
"$PY" "$HERE/__init__.py"
