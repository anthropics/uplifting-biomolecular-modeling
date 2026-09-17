"""Build libew_progen2.so from kernels.cu with the pinned stack's nvcc (CUDA 12.8); write BUILD.json (the pins ext.load checks) and SHA256SUMS
(the digest it holds the library to) beside it.

    python -m engines.progen2.kits.v0_ew.build [--out DIR] [--ln-variant V]        (cwd opt/forward; the pinned stack's interpreter, nvcc on PATH)

nvcc defaults (no --use_fast_math, -ftz=false, -prec-div=true, -prec-sqrt=true; fmad left ON so the CUDA math library's own
tanhf/rsqrtf compile as torch's do — every arithmetic op of OUR chains is an explicit *_rn intrinsic, which the backend never
contracts); SASS for sm_80, sm_90 and sm_100 plus compute_90 PTX. The PTX audit counts the .rn ops and any contractable mul/add left
in our kernels (must be 0 outside the library code).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "kernels.cu")
BINARY = "libew_progen2.so"
GENCODE = ["-gencode=arch=compute_80,code=sm_80", "-gencode=arch=compute_90,code=sm_90", "-gencode=arch=compute_100,code=sm_100", "-gencode=arch=compute_90,code=compute_90"]   # SASS for sm_80 (A100), sm_90 (H100/H200), sm_100 (B200) + compute_90 PTX; the same source and flags on every arch (the *_rn intrinsics pin the arithmetic; the PTX audit below holds)
FLAGS = ["-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared", "-cudart", "shared", "-lineinfo"]


def ptx_audit(so: str) -> dict:
    ptx = subprocess.run(["cuobjdump", "-ptx", so], capture_output=True, text=True).stdout
    out = {"ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(), "kernels": {}}
    # split per .entry; count the op classes inside each kernel body (library code is inlined, so tanhf's fma.rn shows up in gelu)
    for m in re.finditer(r"\.entry\s+(\w+)\s*\((.*?)\n\}\n", ptx, flags=re.S):
        name, body = m.group(1), m.group(2)
        out["kernels"][name] = {"mul.rn.f32": len(re.findall(r"\bmul\.rn\.f32\b", body)), "add.rn.f32": len(re.findall(r"\badd\.rn\.f32\b", body)),
                                "sub.rn.f32": len(re.findall(r"\bsub\.rn\.f32\b", body)), "fma.rn.f32": len(re.findall(r"\bfma\.rn\.f32\b", body)),
                                "div.rn.f32": len(re.findall(r"\bdiv\.rn\.f32\b", body)), "mul.f32": len(re.findall(r"\bmul\.f32\b", body)),
                                "add.f32": len(re.findall(r"\badd\.f32\b", body)), "sub.f32": len(re.findall(r"\bsub\.f32\b", body)),
                                "ftz": len(re.findall(r"\.ftz\b", body)), "rsqrt.approx": len(re.findall(r"\brsqrt\.approx", body)),
                                "cvt.rn.f16.f32": len(re.findall(r"\bcvt\.rn\.f16\.f32\b", body)), "tanh.approx": len(re.findall(r"\btanh\.approx", body))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--ln-variant", type=int, default=None, help="the LN contraction variant to record in BUILD.json (ln_variant)")
    ap.add_argument("--keep-build", default=None)
    ap.add_argument("--audit-only", action="store_true", help="no build: PTX-audit the committed binary of --so into --out/ptx_audit.json")
    ap.add_argument("--so", default=os.path.join(HERE, BINARY))
    a = ap.parse_args(argv)
    if a.audit_only:
        audit = {"so": a.so, **ptx_audit(a.so)}
        os.makedirs(a.out, exist_ok=True)
        json.dump(audit, open(os.path.join(a.out, "ptx_audit.json"), "w"), indent=1)
        print(json.dumps({"ptx_sha256": audit["ptx_sha256"], "n_kernels": len(audit["kernels"])}))
        print(json.dumps(audit["kernels"], indent=0))
        return 0
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-2:]
    so = os.path.join(a.out, BINARY)
    cmd = [nvcc, *FLAGS, *GENCODE, "-o", so, SOURCE]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        return 1
    build = {"binary": BINARY, "nvcc": ver, "cmd": cmd[1:-2],
             "gencode": GENCODE, "build_s": round(time.time() - t0, 1), "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "warnings": (r.stdout + r.stderr)[-2000:]}
    try:
        import torch
        build["torch"] = torch.__version__
        build["cuda"] = torch.version.cuda
    except Exception as ex:  # noqa: BLE001
        build["torch"] = None
        build["torch_err"] = repr(ex)
    build["ptx_audit"] = ptx_audit(so)
    prev = os.path.join(a.out, "BUILD.json")
    if a.ln_variant is not None:
        build["ln_variant"] = int(a.ln_variant)
    elif os.path.exists(prev):
        old = json.load(open(prev))
        if "ln_variant" in old:
            build["ln_variant"] = old["ln_variant"]
    json.dump(build, open(prev, "w"), indent=1)
    h = hashlib.sha256()
    with open(so, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    with open(os.path.join(a.out, "SHA256SUMS"), "w", encoding="utf-8") as fh:   # the line ext.load holds the library to
        fh.write(f"{h.hexdigest()}  {BINARY}\n")
    print(json.dumps({k: build[k] for k in ("binary", "nvcc", "build_s", "torch")}))
    print(json.dumps(build["ptx_audit"]["kernels"], indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
