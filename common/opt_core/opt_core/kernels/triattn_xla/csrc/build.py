#!/usr/bin/env python3
"""triattn_xla build: produces everything under ../bin/ and the binary section of ../manifest.json.  Nothing here runs at import or
at prediction time; a stack without compilers uses the shipped binaries as they are.

    python build.py launcher --ffi-include <jaxlib include dir> [--tag ffi-<maj>.<min>]   # g++: bin/launcher/<tag>/libtriattn_xla_launch.so
    python build.py cuda [--nvcc nvcc]                                                    # nvcc sm_90a: bin/cuda/sm_90a/libtriattn_mw_cuda.so
    python build.py k2b --arch 90,80 [--dtypes bf16,fp32] [--dims 32,16]                  # Triton ahead-of-time: bin/k2b/sm_<arch>/*.cubin + .json
    python build.py manifest                                                              # (re)writes ../manifest.json from bin/**/*.json + sha256 of every binary

launcher: needs g++, cuda.h (CUDA toolkit or the nvidia-cuda-runtime wheel's include dir) and libcuda (or its stub).
k2bl:     as k2b, for the differentiable row's forward (csrc/k2b_lse_kernel.py: the same kernel + a log-sum-exp store), head_dim 32.
cuda:     needs nvcc >= 12.8 (sm_90a); no CUTLASS, no torch.  The device code is kernels/triattn/cuda_sm90a/csrc/triattn_mw.cu (the
          torch provider's carried sources, not copied here) up to its torch host section.
k2b:      needs the pinned Triton compiler (3.7.x) and torch importable (the carried kernel module imports torch); no GPU needed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                      # .../kernels/triattn_xla
BIN = os.path.join(PKG, "bin")
KERNELS = os.path.dirname(PKG)                   # .../kernels
TORCH_SPLIT_MARK = "#include <torch/types.h>"
CUDA_KERNEL_SRC = os.path.join(KERNELS, "triattn", "cuda_sm90a", "csrc")     # triattn_mw.cu + mw_ptx.h, carried by the torch provider


def sha256(path_or_bytes) -> str:
    h = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        h.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            for ch in iter(lambda: f.read(1 << 20), b""):
                h.update(ch)
    return h.hexdigest()


def run(cmd, **kw):
    print("[build] $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kw)


def find_cuda_include(user: str = "") -> str:
    cands = [user] if user else []
    cands += [p for sp in sys.path for p in glob.glob(os.path.join(sp, "nvidia", "cuda_runtime", "include"))]
    cands += sorted(glob.glob("/usr/local/cuda*/include")) + sorted(glob.glob("/usr/local/cuda*/targets/x86_64-linux/include"))
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "cuda.h")):
            return c
    raise SystemExit("cuda.h not found (pass --cuda-include)")


def find_libcuda_dir() -> str:
    for p in ("/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1") + tuple(sorted(glob.glob("/usr/local/cuda*/lib64/stubs/libcuda.so"))) + \
             tuple(sorted(glob.glob("/usr/local/cuda*/targets/x86_64-linux/lib/stubs/libcuda.so"))):
        if os.path.exists(p):
            return os.path.dirname(p)
    raise SystemExit("libcuda.so not found")


def ffi_api_version(inc: str) -> str:
    txt = open(os.path.join(inc, "xla", "ffi", "api", "c_api.h"), encoding="utf-8").read()
    maj = re.search(r"#define XLA_FFI_API_MAJOR (\d+)", txt).group(1)
    mnr = re.search(r"#define XLA_FFI_API_MINOR (\d+)", txt).group(1)
    return f"{maj}.{mnr}"


# ---------------------------------------------------------------------------------------------------------------- launcher (g++)
def build_launcher(a):
    inc = a.ffi_include
    ver = ffi_api_version(inc)
    tag = a.tag or f"ffi-{ver}"
    out = os.path.join(BIN, "launcher", tag); os.makedirs(out, exist_ok=True)
    cuda_inc = find_cuda_include(a.cuda_include)
    libdir = find_libcuda_dir()
    so = os.path.join(out, "libtriattn_xla_launch.so")
    stage = tempfile.mkdtemp(prefix="txla_launch_")            # compile from a neutral copy: no build-machine path in the binary
    for f in ("cubin_launch.cc", "triattn_cuda_abi.h", "triattn_m1_abi.h", "triattn_sm80_abi.h"):
        shutil.copy2(os.path.join(HERE, f), stage)
    gxx = a.gxx or "g++"
    cmd = [gxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-fvisibility-inlines-hidden", f"-ffile-prefix-map={stage}=csrc", f"-ffile-prefix-map={inc}=ffi_include",
           "-I", inc, "-I", cuda_inc, os.path.join(stage, "cubin_launch.cc"), "-o", so, "-L", libdir, "-lcuda", "-Wl,--as-needed"]
    t0 = time.time(); run(cmd); dt = time.time() - t0
    gv = subprocess.run([gxx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    ldd = subprocess.run(["ldd", so], capture_output=True, text=True).stdout
    dump = subprocess.run(["objdump", "-T", so], capture_output=True, text=True).stdout
    vkey = lambda v: [int(x) for x in v.split(".")]
    glibc_req = sorted(set(re.findall(r"GLIBC_([0-9.]+)", dump)), key=vkey); cxx_req = sorted(set(re.findall(r"GLIBCXX_([0-9.]+)", dump)), key=vkey); abi_req = sorted(set(re.findall(r"CXXABI_([0-9.]+)", dump)), key=vkey)
    meta = {"kind": "launcher", "file": os.path.relpath(so, PKG), "requires": {"GLIBC_max": glibc_req[-1] if glibc_req else None, "GLIBCXX_max": cxx_req[-1] if cxx_req else None, "CXXABI_max": abi_req[-1] if abi_req else None}, "sha256": sha256(so), "bytes": os.path.getsize(so), "ffi_api_version": ver, "tag": tag,
            "jaxlib_lines": [s for s in a.jaxlib_lines.split(",") if s], "gxx": gv, "cuda_include": os.path.basename(os.path.dirname(os.path.dirname(cuda_inc))) or cuda_inc,
            "sources_sha256": {f: sha256(os.path.join(HERE, f)) for f in ("cubin_launch.cc", "triattn_cuda_abi.h", "triattn_m1_abi.h", "triattn_sm80_abi.h")},
            "needed": sorted(set(re.findall(r"^\s*(\S+) =>", ldd, re.M))), "build_s": round(dt, 1), "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(meta, open(os.path.join(out, "libtriattn_xla_launch.json"), "w"), indent=1, sort_keys=True)
    print("[build] launcher", json.dumps(meta, indent=None)[:600])
    shutil.rmtree(stage, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------- cuda (nvcc)
NVCC_FLAGS = ["-O3", "-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr", "-DNDEBUG", "-Xptxas", "-v"]


def device_prefix(src_path: str) -> bytes:
    raw = open(src_path, "rb").read()
    i = raw.find(TORCH_SPLIT_MARK.encode())
    if i < 0:
        raise SystemExit(f"{src_path}: split mark {TORCH_SPLIT_MARK!r} not found")
    return raw[:i]


LSE_PATCH = os.path.join(HERE, "cuda_lse.patch")     # the differentiable row's forward: the device section + a per-row log-sum-exp store (unified diff)


def apply_patch_strict(text: str, patch: str) -> str:
    """Apply a unified diff to text; every hunk's context and removed lines must match exactly (no fuzz, no offsets)."""
    src = text.split("\n"); out = []; pos = 0
    lines = patch.split("\n"); i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
            start = int(m.group(1)) - 1
            out.extend(src[pos:start]); pos = start; i += 1
            while i < len(lines) and not lines[i].startswith("@@"):
                h = lines[i]
                if h.startswith(("---", "+++")) and not h.startswith(("--- ", "+++ ")) or h == "":
                    if h == "" and i == len(lines) - 1:
                        i += 1; continue
                tag, body = h[:1], h[1:]
                if tag == " ":
                    if src[pos] != body: raise SystemExit("patch context mismatch at source line %d: %r != %r" % (pos + 1, src[pos], body))
                    out.append(body); pos += 1
                elif tag == "-":
                    if src[pos] != body: raise SystemExit("patch removal mismatch at source line %d" % (pos + 1))
                    pos += 1
                elif tag == "+":
                    out.append(body)
                elif h.startswith("\\"):
                    pass
                else:
                    raise SystemExit("unexpected patch line %r" % h)
                i += 1
        else:
            i += 1
    out.extend(src[pos:])
    return "\n".join(out)


def build_cuda(a):
    """Two payloads from the carried kernel sources: libtriattn_mw_cuda.so (the device section unmodified: row cuda_sm90a) and
    libtriattn_mw_cuda_lse.so (the device section + csrc/cuda_lse.patch: the differentiable row's forward, which also stores each query
    row's log2-sum-exp; host translation unit compiled with -DTXLA_LSE, entry triattn_mw_cuda_fwd_lse, call struct revision 2)."""
    out = os.path.join(BIN, "cuda", "sm_90a"); os.makedirs(out, exist_ok=True)
    src = os.path.join(CUDA_KERNEL_SRC, "triattn_mw.cu")
    prefix = device_prefix(src)
    patch = open(LSE_PATCH, encoding="utf-8").read()
    patched = apply_patch_strict(prefix.decode("utf-8"), patch).encode("utf-8")
    assert patched != prefix and b"float* lse;" in patched
    nvcc = a.nvcc or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    nv = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    for variant in (a.variants.split(",") if a.variants else ["plain", "lse"]):
        lse = variant == "lse"
        stage = tempfile.mkdtemp(prefix="txla_cuda_")
        open(os.path.join(stage, "mw_device.inc"), "wb").write(patched if lse else prefix)
        shutil.copy2(os.path.join(CUDA_KERNEL_SRC, "mw_ptx.h"), os.path.join(stage, "mw_ptx.h"))      # mw_device.inc includes "mw_ptx.h" beside it
        for f in ("triattn_mw_cuda.cu", "triattn_cuda_abi.h"):
            shutil.copy2(os.path.join(HERE, f), stage)
        name = "libtriattn_mw_cuda_lse" if lse else "libtriattn_mw_cuda"
        so = os.path.join(out, name + ".so")
        entry = "triattn_mw_cuda_fwd_lse" if lse else "triattn_mw_cuda_fwd"
        cmd = [nvcc] + NVCC_FLAGS + (["-DTXLA_LSE=1"] if lse else []) + ["-Xcompiler", "-fPIC", "-Xcompiler", "-fvisibility=hidden", "-Xcompiler", f"-ffile-prefix-map={stage}=csrc",
               "-shared", "-cudart", "static", "-I", stage, os.path.join(stage, "triattn_mw_cuda.cu"), "-o", so]
        vis = os.path.join(stage, "exports.map")          # exported C symbols keep default visibility through the extern "C" block + an explicit export list
        open(vis, "w").write("{ global: %s; triattn_mw_cuda_fix_elems; triattn_mw_cuda_smem_bytes; triattn_mw_cuda_describe; local: *; };\n" % entry)
        cmd += ["-Xlinker", f"--version-script={vis}"]
        t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True); dt = time.time() - t0
        print("[build] $", " ".join(cmd)); print(r.stdout[-3000:]); print(r.stderr[-6000:])
        if r.returncode:
            raise SystemExit("nvcc failed (%s)" % variant)
        ldd = subprocess.run(["ldd", so], capture_output=True, text=True).stdout
        regs = re.findall(r"Function properties for (\S+)\n.*?\n.*?Used (\d+) registers", r.stderr, re.S)
        dump = subprocess.run(["objdump", "-T", so], capture_output=True, text=True).stdout
        vkey = lambda v: [int(x) for x in v.split(".")]
        glibc_req = sorted(set(re.findall(r"GLIBC_([0-9.]+)", dump)), key=vkey); cxx_req = sorted(set(re.findall(r"GLIBCXX_([0-9.]+)", dump)), key=vkey); abi_req = sorted(set(re.findall(r"CXXABI_([0-9.]+)", dump)), key=vkey)
        sources = {"kernels/triattn/cuda_sm90a/csrc/triattn_mw.cu": sha256(src), "kernels/triattn/cuda_sm90a/csrc/triattn_mw.cu[device prefix]": hashlib.sha256(prefix).hexdigest(),
                   "kernels/triattn/cuda_sm90a/csrc/mw_ptx.h": sha256(os.path.join(CUDA_KERNEL_SRC, "mw_ptx.h")),
                   "triattn_mw_cuda.cu": sha256(os.path.join(HERE, "triattn_mw_cuda.cu")), "triattn_cuda_abi.h": sha256(os.path.join(HERE, "triattn_cuda_abi.h"))}
        if lse:
            sources["csrc/cuda_lse.patch"] = sha256(LSE_PATCH); sources["triattn_mw.cu[device prefix + cuda_lse.patch]"] = hashlib.sha256(patched).hexdigest()
        meta = {"kind": "cuda", "role": "fwd_lse" if lse else "fwd", "file": os.path.relpath(so, PKG),
                "requires": {"GLIBC_max": glibc_req[-1] if glibc_req else None, "GLIBCXX_max": cxx_req[-1] if cxx_req else None, "CXXABI_max": abi_req[-1] if abi_req else None},
                "sha256": sha256(so), "bytes": os.path.getsize(so), "arch": "sm_90a", "nvcc": nv, "nvcc_flags": NVCC_FLAGS + (["-DTXLA_LSE=1"] if lse else []),
                "cudart": "static", "abi_version": 2 if lse else 1, "entry": entry, "sources_sha256": sources,
                "device_prefix_bytes": len(patched if lse else prefix), "split_mark": TORCH_SPLIT_MARK, "ptxas_registers": {k: int(v) for k, v in regs},
                "needed": sorted(set(re.findall(r"^\s*(\S+) =>", ldd, re.M))), "build_s": round(dt, 1), "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if lse:
            meta["lse"] = {"dtype": "fp32", "layout": "[B,N,H,S] dense", "units": "log2", "dead_row": "log2(key count) (unseeded offset 0)"}
        json.dump(meta, open(os.path.join(out, name + ".json"), "w"), indent=1, sort_keys=True)
        print("[build] cuda", variant, json.dumps({k: v for k, v in meta.items() if k != "ptxas_registers"})[:900])
        shutil.rmtree(stage, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------- m1 (row triattn_native)
NATIVE_GEN = "v11"
_NATIVE_CANDIDATES = [os.path.join(KERNELS, "triattn", "triattn_native", "pkg", NATIVE_GEN, "triattn_pkg")]                                          # carried by the torch provider: opt_core's own lifted package is the only source (0.5.102.0)
NATIVE_PKG = next((c for c in _NATIVE_CANDIDATES if os.path.isdir(c)), _NATIVE_CANDIDATES[0])
M1_SRC = os.path.join(NATIVE_PKG, "cuda_b", "csrc")        # m1/triattn_m1_sm90.cuh + fa3_utils.h
SM80_SRC = os.path.join(NATIVE_PKG, "cuda_80", "csrc")     # triattn_sm80.cuh, launch_sm80.cuh, inst_sm80.h, inst_d16/32/64.cu
SM80_FILES = ("triattn_sm80.cuh", "launch_sm80.cuh", "inst_sm80.h", "inst_d16.cu", "inst_d32.cu", "inst_d64.cu")
SM80_NVCC_FLAGS = ["-O3", "-std=c++17", "-gencode", "arch=compute_80,code=sm_80", "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas", "-v", "-DNDEBUG", "-DTS_HAVE_D16", "-DTS_HAVE_D32", "-DTS_HAVE_D64"]   # the kernel directory's own (triattn_sm80.py _build)
M1_NVCC_FLAGS = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math", "-gencode", "arch=compute_90a,code=sm_90a", "-DNDEBUG",
                 "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "--ftemplate-backtrace-limit=0", "-lineinfo", "-Xcompiler", "-Wno-psabi", "-diag-suppress", "177,550", "-Xptxas", "-v"]   # the kernel directory's own (cuda_b/triattn_m1.py _build)


def find_cutlass(hint: str = "") -> str:
    """A directory whose include/cute/tensor.hpp exists: --cutlass, $CUTLASS_PATH, /opt/cutlass, the nvidia-cutlass wheel (cutlass_library/source)."""
    cands = [hint, os.environ.get("CUTLASS_PATH", ""), "/opt/cutlass", "/usr/local/cutlass"]
    try:
        import cutlass_library                                  # the nvidia-cutlass wheel ships the headers under source/include
        cands.append(os.path.join(os.path.dirname(cutlass_library.__file__), "source"))
    except ImportError:
        pass
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "include", "cute", "tensor.hpp")):
            return c
    raise SystemExit("CUTLASS headers not found (need <dir>/include/cute/tensor.hpp): pass --cutlass or set CUTLASS_PATH")


def build_m1(a):
    """libtriattn_m1_xla.so: the M1 kernel header of kernels/triattn/triattn_native (payload generation 10, cuda_b kernel directory) compiled unmodified with that kernel directory's
    nvcc flags + CUTLASS, and csrc/triattn_m1_xla.cu (its host side restated over raw pointers; C entry triattn_m1_xla_fwd); static cudart."""
    out = os.path.join(BIN, "cuda", "sm_90a"); os.makedirs(out, exist_ok=True)
    cutlass = find_cutlass(a.cutlass)
    vh = os.path.join(cutlass, "include", "cutlass", "version.h"); tag = None
    if os.path.isfile(vh):
        t = open(vh).read(); tag = "v%s.%s.%s" % tuple(re.search(r"#define CUTLASS_%s (\d+)" % w, t).group(1) for w in ("MAJOR", "MINOR", "PATCH"))
    nvcc = a.nvcc or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    nv = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    ptxas = shutil.which("ptxas") or os.path.join(os.path.dirname(nvcc), "ptxas")
    pv = subprocess.run([ptxas, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1] if os.path.exists(ptxas) else "?"
    stage = tempfile.mkdtemp(prefix="txla_m1_")
    for f in ("triattn_m1_xla.cu", "triattn_m1_abi.h"):
        shutil.copy2(os.path.join(HERE, f), stage)
    so = os.path.join(out, "libtriattn_m1_xla.so")
    cmd = [nvcc] + M1_NVCC_FLAGS + ["-Xcompiler", "-fPIC", "-Xcompiler", "-fvisibility=hidden", "-Xcompiler", f"-ffile-prefix-map={stage}=csrc",
           "-shared", "-cudart", "static", "-I", stage, "-I", os.path.join(M1_SRC, "m1"), "-I", M1_SRC, "-I", os.path.join(cutlass, "include"),
           os.path.join(stage, "triattn_m1_xla.cu"), "-o", so]
    vis = os.path.join(stage, "exports.map")
    open(vis, "w").write("{ global: triattn_m1_xla_fwd; triattn_m1_xla_describe; triattn_m1_xla_smem_bytes; local: *; };\n")
    cmd += ["-Xlinker", f"--version-script={vis}"]
    t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True); dt = time.time() - t0
    print("[build] $", " ".join(cmd)); print(r.stdout[-3000:]); print(r.stderr[-8000:])
    if r.returncode:
        raise SystemExit("nvcc failed (m1)")
    ldd = subprocess.run(["ldd", so], capture_output=True, text=True).stdout
    regs = re.findall(r"Function properties for (\S+)\n.*?\n.*?Used (\d+) registers", r.stderr, re.S)
    dump = subprocess.run(["objdump", "-T", so], capture_output=True, text=True).stdout
    vkey = lambda v: [int(x) for x in v.split(".")]
    glibc_req = sorted(set(re.findall(r"GLIBC_([0-9.]+)", dump)), key=vkey); cxx_req = sorted(set(re.findall(r"GLIBCXX_([0-9.]+)", dump)), key=vkey); abi_req = sorted(set(re.findall(r"CXXABI_([0-9.]+)", dump)), key=vkey)
    sources = {"kernels/triattn/triattn_native/pkg/" + NATIVE_GEN + "/triattn_pkg/cuda_b/csrc/m1/triattn_m1_sm90.cuh": sha256(os.path.join(M1_SRC, "m1", "triattn_m1_sm90.cuh")),
               "kernels/triattn/triattn_native/pkg/" + NATIVE_GEN + "/triattn_pkg/cuda_b/csrc/fa3_utils.h": sha256(os.path.join(M1_SRC, "fa3_utils.h")),
               "triattn_m1_xla.cu": sha256(os.path.join(HERE, "triattn_m1_xla.cu")), "triattn_m1_abi.h": sha256(os.path.join(HERE, "triattn_m1_abi.h"))}
    meta = {"kind": "cuda", "role": "m1_fwd", "file": os.path.relpath(so, PKG),
            "requires": {"GLIBC_max": glibc_req[-1] if glibc_req else None, "GLIBCXX_max": cxx_req[-1] if cxx_req else None, "CXXABI_max": abi_req[-1] if abi_req else None},
            "sha256": sha256(so), "bytes": os.path.getsize(so), "arch": "sm_90a", "nvcc": nv, "ptxas": pv, "cutlass_tag": tag, "nvcc_flags": M1_NVCC_FLAGS,
            "cudart": "static", "abi_version": 1, "entry": "triattn_m1_xla_fwd", "sources_sha256": sources, "instantiations": ["Traits<0>", "Traits<1024>"], "package_generation": NATIVE_GEN,
            "ptxas_registers": {k[-60:]: int(v) for k, v in regs}, "needed": sorted(set(re.findall(r"^\s*(\S+) =>", ldd, re.M))), "build_s": round(dt, 1),
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(meta, open(os.path.join(out, "libtriattn_m1_xla.json"), "w"), indent=1, sort_keys=True)
    print("[build] m1", json.dumps({k: v for k, v in meta.items() if k != "ptxas_registers"})[:1200])
    shutil.rmtree(stage, ignore_errors=True)


def build_sm80(a):
    """libtriattn_sm80_xla.so: the sm_80 member of kernels/triattn/triattn_native's kernel family (package generation 11, kernel directory cuda_80: device header,
    launch template and per-head-dim instantiation units compiled UNMODIFIED with that directory's own nvcc flags) + csrc/triattn_sm80_xla.cu (the torch
    binding's host side restated over raw pointers; C entries triattn_sm80_xla_*); static cudart; sm_80 code only."""
    out = os.path.join(BIN, "cuda", "sm_80"); os.makedirs(out, exist_ok=True)
    nvcc = a.nvcc or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    nv = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    ptxas = shutil.which("ptxas") or os.path.join(os.path.dirname(nvcc), "ptxas")
    pv = subprocess.run([ptxas, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1] if os.path.exists(ptxas) else "?"
    for f in SM80_FILES:
        if not os.path.isfile(os.path.join(SM80_SRC, f)):
            raise SystemExit("sm80: %s not found under %s" % (f, SM80_SRC))
    stage = tempfile.mkdtemp(prefix="txla_sm80_")
    for f in ("triattn_sm80_xla.cu", "triattn_sm80_abi.h"):
        shutil.copy2(os.path.join(HERE, f), stage)
    so = os.path.join(out, "libtriattn_sm80_xla.so")
    vis = os.path.join(stage, "exports.map")
    open(vis, "w").write("{ global: triattn_sm80_xla_fwd; triattn_sm80_xla_fix_elems; triattn_sm80_xla_dims; triattn_sm80_xla_geometry; triattn_sm80_xla_describe; local: *; };\n")
    srcs = [os.path.join(stage, "triattn_sm80_xla.cu")] + [os.path.join(SM80_SRC, f) for f in ("inst_d16.cu", "inst_d32.cu", "inst_d64.cu")]
    cmd = [nvcc] + SM80_NVCC_FLAGS + ["-Xcompiler", "-fPIC", "-Xcompiler", "-fvisibility=hidden", "-Xcompiler", f"-ffile-prefix-map={stage}=csrc",
           "-shared", "-cudart", "static", "-I", stage, "-I", SM80_SRC] + srcs + ["-o", so, "-Xlinker", f"--version-script={vis}"]
    t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True); dt = time.time() - t0
    print("[build] $", " ".join(cmd)); print(r.stdout[-3000:]); print(r.stderr[-12000:])
    if r.returncode:
        raise SystemExit("nvcc failed (sm80)")
    ldd = subprocess.run(["ldd", so], capture_output=True, text=True).stdout
    regs = re.findall(r"Function properties for (\S+)\n.*?\n.*?Used (\d+) registers", r.stderr, re.S)
    spills = re.findall(r"Function properties for (\S+)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", r.stderr)
    dump = subprocess.run(["objdump", "-T", so], capture_output=True, text=True).stdout
    vkey = lambda v: [int(x) for x in v.split(".")]
    glibc_req = sorted(set(re.findall(r"GLIBC_([0-9.]+)", dump)), key=vkey); cxx_req = sorted(set(re.findall(r"GLIBCXX_([0-9.]+)", dump)), key=vkey); abi_req = sorted(set(re.findall(r"CXXABI_([0-9.]+)", dump)), key=vkey)
    sources = {"kernels/triattn/triattn_native/pkg/" + NATIVE_GEN + "/triattn_pkg/cuda_80/csrc/" + f: sha256(os.path.join(SM80_SRC, f)) for f in SM80_FILES}
    sources.update({"triattn_sm80_xla.cu": sha256(os.path.join(HERE, "triattn_sm80_xla.cu")), "triattn_sm80_abi.h": sha256(os.path.join(HERE, "triattn_sm80_abi.h"))})
    meta = {"kind": "cuda", "role": "sm80_fwd", "file": os.path.relpath(so, PKG),
            "requires": {"GLIBC_max": glibc_req[-1] if glibc_req else None, "GLIBCXX_max": cxx_req[-1] if cxx_req else None, "CXXABI_max": abi_req[-1] if abi_req else None},
            "sha256": sha256(so), "bytes": os.path.getsize(so), "arch": "sm_80", "nvcc": nv, "ptxas": pv, "nvcc_flags": SM80_NVCC_FLAGS, "cudart": "static", "abi_version": 1,
            "entry": "triattn_sm80_xla_fwd", "sources_sha256": sources, "package_generation": NATIVE_GEN, "head_dims": [16, 32, 64],
            "ptxas_registers": {k[-70:]: int(v) for k, v in regs}, "ptxas_spills": {k[-70:]: [int(s1), int(s2), int(s3)] for k, s1, s2, s3 in spills if int(s2) or int(s3)},
            "needed": sorted(set(re.findall(r"^\s*(\S+) =>", ldd, re.M))), "build_s": round(dt, 1), "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(meta, open(os.path.join(out, "libtriattn_sm80_xla.json"), "w"), indent=1, sort_keys=True)
    print("[build] sm80", json.dumps({k: v for k, v in meta.items() if k not in ("ptxas_registers", "ptxas_spills")})[:1500])
    shutil.rmtree(stage, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------- k2b (Triton AOT)
MAIN_INTS = ["sqb", "sqi", "sqh", "sqq", "skb", "ski", "skh", "skk", "svb", "svi", "svh", "svk", "sbb", "sbh", "sbq", "smb", "smi", "smk",
             "sob", "soi", "soh", "soq", "N_ROWS", "SEQ_Q", "SEQ_K", "H", "n_flags"]
PREP_INTS = ["sbb", "sbh", "sbq", "sbk", "H", "SQ", "SK", "SKp", "NQB"]
# divisibility-by-16 classes: "always" (true for every call the face makes), "s16" (true when SQ, SK, N are multiples of 16), "never"
MAIN_DIV = {"sqb": "always", "sqi": "always", "sqh": "always", "sqq": "always", "skb": "always", "ski": "always", "skh": "always", "skk": "always",
            "svb": "always", "svi": "always", "svh": "always", "svk": "always", "sbb": "always", "sbh": "always", "sbq": "always",
            "smb": "mask:s16|nomask:always", "smi": "mask:s16|nomask:always", "smk": "mask:one|nomask:always",
            "sob": "always", "soi": "always", "soh": "always", "soq": "always", "N_ROWS": "s16", "SEQ_Q": "s16", "SEQ_K": "s16", "H": "never", "n_flags": "never"}
PREP_DIV = {"sbb": "s16", "sbh": "s16", "sbq": "s16", "sbk": "one", "H": "never", "SQ": "s16", "SK": "s16", "SKp": "always", "NQB": "never"}


def _k2b_module():
    d = os.path.join(KERNELS, "fpf_triatt_k2b")
    if d not in sys.path:
        sys.path.insert(0, d)
    import triatt_k2b as M          # the carried kernel, by its own loader convention (top-level module from its directory)
    return M


def _cells_cfg(arch: int, dtype: str, D: int, M):
    """The launch cell for (arch, dtype, D): K2B_CELLS.json entry for the arch when it has one, else the package table."""
    cells = json.load(open(os.path.join(KERNELS, "fpf_triatt_k2b", "K2B_CELLS.json")))
    ent = cells.get("entries", {}).get(f"sm{arch}", {})
    rows = ent.get(dtype, {}).get(str(D))
    src = f"K2B_CELLS.json entries.sm{arch}.{dtype}.{D}"
    if not rows:
        tab = M._CONFIG_TABLE_F32 if dtype == "fp32" else M._CONFIG_TABLE
        rows = tab.get(D) or tab[32]
        src = ("_CONFIG_TABLE_F32" if dtype == "fp32" else "_CONFIG_TABLE") + f"[{D}]"
    cfg = dict(rows[-1][1])
    if dtype == "fp32":
        cfg.pop("MAXNREG", None)                     # the fp32 path launches uncapped (flash_triangle_attention: maxnreg=None for fp32)
    else:
        cfg["MAXNREG"] = 128                         # the Triton >= 3.7 register cap of the package CONFIG (this build's compiler line)
    return cfg, src


def _compile(fn, all_names, ptr_types, int_names, div16, ones, consts, num_warps, num_stages, maxnreg, arch):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    signature, cx, attrs = {}, dict(consts), {}
    for i, n in enumerate(all_names):
        if n in ptr_types:
            signature[n] = ptr_types[n]; attrs[(i,)] = [["tt.divisibility", 16]]
        elif n in int_names:
            if n in ones:
                signature[n] = "constexpr"; cx[n] = 1
            else:
                signature[n] = "i32"
                if n in div16:
                    attrs[(i,)] = [["tt.divisibility", 16]]
        elif n == "qk_scale":
            signature[n] = "fp32"
        elif n in cx:
            signature[n] = "constexpr"
        else:
            raise SystemExit(f"kernel parameter {n!r} unaccounted")
    src = ASTSource(fn=fn, signature=signature, constexprs=cx, attrs=attrs)
    opts = dict(num_warps=int(num_warps), num_stages=int(num_stages))
    if maxnreg:
        opts["maxnreg"] = int(maxnreg)
    ck = triton.compile(src, target=GPUTarget("cuda", int(arch), 32), options=opts)
    md = ck.metadata
    g = (lambda k, d=None: getattr(md, k, d)) if not isinstance(md, dict) else (lambda k, d=None: md.get(k, d))
    info = dict(name=g("name"), shared=int(g("shared") or 0), num_warps=int(g("num_warps") or num_warps), n_regs=getattr(ck, "n_regs", None), n_spills=getattr(ck, "n_spills", None),
                global_scratch_size=int(g("global_scratch_size", 0) or 0), profile_scratch_size=int(g("profile_scratch_size", 0) or 0))
    if info["global_scratch_size"] or info["profile_scratch_size"]:
        raise SystemExit(f"kernel {info['name']} needs Triton scratch memory: {info}")
    return ck.asm["cubin"], info, {"signature": signature, "constexprs": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in cx.items()},
                                   "attrs": {str(k[0]): v for k, v in attrs.items()}, "options": opts}


def _trailing_null_ptrs() -> int:
    import triton.backends.nvidia.driver as d
    src = open(d.__file__).read()
    return int("global_scratch" in src) + int("profile_scratch" in src)


def build_k2b(a):
    import triton
    M = _k2b_module()
    kpath = os.path.abspath(M.__file__)
    ksha = sha256(kpath)
    tver = triton.__version__
    ptxas = glob.glob(os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas"))
    ptxas_v = subprocess.run([ptxas[0], "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1] if ptxas else "?"
    n_null = _trailing_null_ptrs()
    archs = [int(x) for x in a.arch.split(",")]
    dtypes = a.dtypes.split(","); dims = [int(x) for x in a.dims.split(",")]
    main_names = list(M._flash_triattn_fwd.arg_names); prep_names = list(M._bias_prep.arg_names)
    built = []
    for arch in archs:
        out = os.path.join(BIN, "k2b", f"sm_{arch}"); os.makedirs(out, exist_ok=True)
        # -- bias preparation kernel: MAKE16 x div class (dtype of the 16-bit copy: bf16)
        for make16 in (0, 1):
            for div in ("s16", "any"):
                key = f"k2b_prep_mk{make16}_{div}"
                ptr = {"Bias": "*fp32", "Out32": "*fp32", "Out16": "*bf16" if make16 else "*fp32", "Flags": "*i32"}
                d16 = {n for n, c in PREP_DIV.items() if c == "always" or (c == "s16" and div == "s16")}
                ones = {n for n, c in PREP_DIV.items() if c == "one"}
                t0 = time.time()
                cub, info, spec = _compile(M._bias_prep, prep_names, ptr, set(PREP_INTS), d16, ones, dict(PB=32, PK=128, MAKE16=bool(make16)), 4, 2, None, arch)
                meta = {"kind": "k2b_cubin", "key": key, "role": "prep", "arch": f"sm_{arch}", "file": f"bin/k2b/sm_{arch}/{key}.cubin", "sha256": sha256(cub), "bytes": len(cub),
                        "kernel": info, "params": [n for n in prep_names if spec["signature"].get(n) != "constexpr"], "ints": PREP_INTS, "div16": sorted(d16), "ones": sorted(ones),
                        "constexprs": spec["constexprs"], "signature": spec["signature"], "options": spec["options"], "n_trailing_null": n_null,
                        "triton": tver, "ptxas": ptxas_v, "source": "kernels/fpf_triatt_k2b/triatt_k2b.py::_bias_prep", "source_sha256": ksha, "compile_s": round(time.time() - t0, 2)}
                open(os.path.join(out, key + ".cubin"), "wb").write(cub)
                json.dump(meta, open(os.path.join(out, key + ".json"), "w"), indent=1, sort_keys=True)
                built.append(meta); print("[build] k2b", key, f"sm_{arch}", info, flush=True)
        # -- the forward kernel
        for dtype in dtypes:
            for D in dims:
                cfg, cfg_src = _cells_cfg(arch, dtype, D, M)
                exp_mode = int(M.CONFIG["EXP_MODE_FP32"] if dtype == "fp32" else M.CONFIG["EXP_MODE_16BIT"])
                for has_mask in (0, 1):
                    for b16 in ((0,) if dtype == "fp32" else (0, 1)):
                        for div in ("s16", "any"):
                            key = f"k2b_fwd_{dtype}_d{D}_mask{has_mask}_b16{b16}_{div}"
                            el = "*bf16" if dtype == "bf16" else "*fp32"
                            ptr = {"Q": el, "K": el, "V": el, "Bias32": "*fp32", "Bias16": "*bf16" if b16 else "*fp32", "Flags": "*i32", "Mask": "*u8" if has_mask else el, "Out": el}
                            d16, ones = set(), set()
                            for n, c in MAIN_DIV.items():
                                for alt in c.split("|"):
                                    cond, _, cls = alt.rpartition(":")
                                    if cond == "mask" and not has_mask: continue
                                    if cond == "nomask" and has_mask: continue
                                    if cls == "always" or (cls == "s16" and div == "s16"): d16.add(n)
                                    if cls == "one": ones.add(n)
                            consts = dict(HEAD_DIM=D, BLOCK_M=int(cfg["BLOCK_M"]), BLOCK_N=int(cfg["BLOCK_N"]), ROWS=int(cfg["ROWS"]), HAS_MASK=bool(has_mask), EXP_MODE=exp_mode,
                                          BIAS16=int(b16), ORDER=int(cfg.get("ORDER", 0)), MSCAN=1024, NFLAG=1024, IP="tf32")
                            t0 = time.time()
                            cub, info, spec = _compile(M._flash_triattn_fwd, main_names, ptr, set(MAIN_INTS), d16, ones, consts, cfg["num_warps"], cfg["num_stages"], cfg.get("MAXNREG"), arch)
                            meta = {"kind": "k2b_cubin", "key": key, "role": "fwd", "arch": f"sm_{arch}", "dtype": dtype, "head_dim": D, "has_mask": has_mask, "bias16": b16, "div": div,
                                    "file": f"bin/k2b/sm_{arch}/{key}.cubin", "sha256": sha256(cub), "bytes": len(cub), "kernel": info, "cell": cfg, "cell_source": cfg_src,
                                    "params": [n for n in main_names if spec["signature"].get(n) != "constexpr"], "ints": MAIN_INTS, "div16": sorted(d16), "ones": sorted(ones),
                                    "constexprs": spec["constexprs"], "signature": spec["signature"], "options": spec["options"], "n_trailing_null": n_null,
                                    "triton": tver, "ptxas": ptxas_v, "source": "kernels/fpf_triatt_k2b/triatt_k2b.py::_flash_triattn_fwd", "source_sha256": ksha, "compile_s": round(time.time() - t0, 2)}
                            open(os.path.join(out, key + ".cubin"), "wb").write(cub)
                            json.dump(meta, open(os.path.join(out, key + ".json"), "w"), indent=1, sort_keys=True)
                            built.append(meta); print("[build] k2b", key, f"sm_{arch}", info, cfg_src, flush=True)
    print(f"[build] k2b: {len(built)} cubins, {sum(m['bytes'] for m in built) / 1e6:.2f} MB, triton {tver}, {ptxas_v}")



def _k2bl_module():
    _k2b_module()                    # the carried module first (k2b_lse_kernel imports its device functions)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import k2b_lse_kernel as L
    return L


def build_k2bl(a):
    """The differentiable row's forward: kernels/fpf_triatt_k2b's forward + a per-row log-sum-exp store (csrc/k2b_lse_kernel.py), compiled with the
    forward-only cubins' cells, constexprs and divisibility classes; one extra output pointer `Lse` (fp32 [B, N, H, SQ]).  Keys k2bl_fwd_<dtype>_d<D>_..."""
    import triton
    M = _k2b_module(); L = _k2bl_module()
    ksha = sha256(os.path.abspath(M.__file__)); lsha = sha256(os.path.abspath(L.__file__))
    tver = triton.__version__
    ptxas = glob.glob(os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas"))
    ptxas_v = subprocess.run([ptxas[0], "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1] if ptxas else "?"
    n_null = _trailing_null_ptrs()
    archs = [int(x) for x in a.arch.split(",")]
    dtypes = a.dtypes.split(","); dims = [int(x) for x in a.dims.split(",")]
    names = list(L._flash_triattn_fwd_lse.arg_names)
    assert names[:9] == ["Q", "K", "V", "Bias32", "Bias16", "Flags", "Mask", "Out", "Lse"], names[:9]
    built = []
    for arch in archs:
        out = os.path.join(BIN, "k2b", f"sm_{arch}"); os.makedirs(out, exist_ok=True)
        for dtype in dtypes:
            for D in dims:
                cfg, cfg_src = _cells_cfg(arch, dtype, D, M)
                exp_mode = int(M.CONFIG["EXP_MODE_FP32"] if dtype == "fp32" else M.CONFIG["EXP_MODE_16BIT"])
                for has_mask in (0, 1):
                    for b16 in ((0,) if dtype == "fp32" else (0, 1)):
                        for div in ("s16", "any"):
                            key = f"k2bl_fwd_{dtype}_d{D}_mask{has_mask}_b16{b16}_{div}"
                            el = "*bf16" if dtype == "bf16" else "*fp32"
                            ptr = {"Q": el, "K": el, "V": el, "Bias32": "*fp32", "Bias16": "*bf16" if b16 else "*fp32", "Flags": "*i32", "Mask": "*u8" if has_mask else el, "Out": el, "Lse": "*fp32"}
                            d16, ones = set(), set()
                            for n, c in MAIN_DIV.items():
                                for alt in c.split("|"):
                                    cond, _, cls = alt.rpartition(":")
                                    if cond == "mask" and not has_mask: continue
                                    if cond == "nomask" and has_mask: continue
                                    if cls == "always" or (cls == "s16" and div == "s16"): d16.add(n)
                                    if cls == "one": ones.add(n)
                            consts = dict(HEAD_DIM=D, BLOCK_M=int(cfg["BLOCK_M"]), BLOCK_N=int(cfg["BLOCK_N"]), ROWS=int(cfg["ROWS"]), HAS_MASK=bool(has_mask), EXP_MODE=exp_mode,
                                          BIAS16=int(b16), ORDER=int(cfg.get("ORDER", 0)), MSCAN=1024, NFLAG=1024, IP="tf32")
                            t0 = time.time()
                            cub, info, spec = _compile(L._flash_triattn_fwd_lse, names, ptr, set(MAIN_INTS), d16, ones, consts, cfg["num_warps"], cfg["num_stages"], cfg.get("MAXNREG"), arch)
                            meta = {"kind": "k2b_cubin", "key": key, "role": "fwd_lse", "arch": f"sm_{arch}", "dtype": dtype, "head_dim": D, "has_mask": has_mask, "bias16": b16, "div": div,
                                    "file": f"bin/k2b/sm_{arch}/{key}.cubin", "sha256": sha256(cub), "bytes": len(cub), "kernel": info, "cell": cfg, "cell_source": cfg_src,
                                    "params": [n for n in names if spec["signature"].get(n) != "constexpr"], "ints": MAIN_INTS, "div16": sorted(d16), "ones": sorted(ones),
                                    "constexprs": spec["constexprs"], "signature": spec["signature"], "options": spec["options"], "n_trailing_null": n_null,
                                    "lse": {"dtype": "fp32", "layout": "[B,N,H,SQ] dense", "units": "log2", "exp_mode": exp_mode},
                                    "triton": tver, "ptxas": ptxas_v, "source": "kernels/triattn_xla/csrc/k2b_lse_kernel.py::_flash_triattn_fwd_lse", "source_sha256": lsha,
                                    "device_functions_source": "kernels/fpf_triatt_k2b/triatt_k2b.py (_attend, _row_step)", "device_functions_source_sha256": ksha, "compile_s": round(time.time() - t0, 2)}
                            open(os.path.join(out, key + ".cubin"), "wb").write(cub)
                            json.dump(meta, open(os.path.join(out, key + ".json"), "w"), indent=1, sort_keys=True)
                            built.append(meta); print("[build] k2bl", key, f"sm_{arch}", info, cfg_src, flush=True)
    print(f"[build] k2bl: {len(built)} cubins, {sum(m['bytes'] for m in built) / 1e6:.2f} MB, triton {tver}, {ptxas_v}")


# ---------------------------------------------------------------------------------------------------------------- manifest
def build_manifest(a):
    entries = []
    for js in sorted(glob.glob(os.path.join(BIN, "**", "*.json"), recursive=True)):
        m = json.load(open(js))
        f = os.path.join(PKG, m["file"])
        if not os.path.isfile(f):
            raise SystemExit(f"{js}: binary {m['file']} missing")
        if sha256(f) != m["sha256"]:
            raise SystemExit(f"{js}: sha256 of {m['file']} differs from its sidecar")
        entries.append(m)
    man_path = os.path.join(PKG, "manifest.json")
    man = json.load(open(man_path)) if os.path.isfile(man_path) else {}
    man["schema"] = "triattn_xla/manifest/v1"
    man["sources_sha256"] = {os.path.relpath(p, PKG): sha256(p) for p in sorted(glob.glob(os.path.join(HERE, "**", "*"), recursive=True))
                            if os.path.isfile(p) and not p.endswith((".pyc",)) and "__pycache__" not in p}
    for f in ("triattn_mw.cu", "mw_ptx.h"):                      # the CUDA kernel sources live with the torch provider (kernels/triattn); recorded, not copied
        man["sources_sha256"]["kernels/triattn/cuda_sm90a/csrc/" + f] = sha256(os.path.join(CUDA_KERNEL_SRC, f))
    man["binaries"] = [{k: v for k, v in m.items() if k not in ("signature",)} for m in entries]
    man["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(man, open(man_path, "w"), indent=1, sort_keys=True)
    print(f"[build] manifest: {len(entries)} binaries -> {man_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launcher"); p.add_argument("--ffi-include", required=True); p.add_argument("--cuda-include", default=""); p.add_argument("--tag", default="")
    p.add_argument("--gxx", default=""); p.add_argument("--jaxlib-lines", default="", help="comma list of jaxlib versions these headers serve (recorded)")
    p = sub.add_parser("cuda"); p.add_argument("--nvcc", default=""); p.add_argument("--variants", default="plain,lse")
    p = sub.add_parser("m1"); p.add_argument("--nvcc", default=""); p.add_argument("--cutlass", default="")
    p = sub.add_parser("sm80"); p.add_argument("--nvcc", default="")
    p = sub.add_parser("k2b"); p.add_argument("--arch", default="90,80"); p.add_argument("--dtypes", default="bf16,fp32"); p.add_argument("--dims", default="32,16")
    p = sub.add_parser("k2bl"); p.add_argument("--arch", default="90,80"); p.add_argument("--dtypes", default="bf16,fp32"); p.add_argument("--dims", default="32")
    sub.add_parser("manifest")
    a = ap.parse_args()
    {"launcher": build_launcher, "cuda": build_cuda, "m1": build_m1, "sm80": build_sm80, "k2b": build_k2b, "k2bl": build_k2bl, "manifest": build_manifest}[a.cmd](a)


if __name__ == "__main__":
    main()
