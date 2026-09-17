#!/usr/bin/env python3
"""build_prebuilt.py -- build this package's CUDA extensions IN THIS INTERPRETER'S STACK and write them to triattn_pkg/prebuilt/<stack tag>/.

A torch C++/CUDA extension is specific to (torch version + its CUDA major, CPython minor, C++ ABI).  Run this inside the interpreter that will load
the package (nvcc + ninja on PATH; CUTLASS >= 4 headers at $CUTLASS_PATH for `cuda` / `cuda_b`; no GPU is needed to build).  Per extension it imports
the kernel directory's Python module from triattn_pkg/<dir>/, calls that module's own `_build()` (sources, flags and the extension name are the
module's, never restated here), copies `<ext>.so` next to a build record `<ext>.json`:

    stack_tag, ext, route, dir, module, torch, torch_cuda, python, soabi, cxx11abi, nvcc, ptxas, cutlass_tag, arch (the -gencode flags the module
    declares), built_utc, build_s, so_bytes, so_sha256, source_sha256 {file under <dir>/csrc: sha256}, module_sha256 {<module>.py: sha256},
    loadcheck = {test-vector id: sha256/16 of the output bytes} -- filled here when this machine has a device of the extension's cc (the byte gate
    runs in a fresh interpreter with TRIATTN_PKG_PREBUILT=always), else "pending" until `python test_pkg.py --record-loadcheck` runs on such a device.

    python tools/build_prebuilt.py --exts triattn_sm80_ext            [--out triattn_pkg/prebuilt] [--no-loadcheck]
    python tools/build_prebuilt.py --exts all                         every extension in triattn_pkg._face.CUDA_SEATS

Architectures: cuda / cuda_c / cuda_b are sm_90a code, cuda_80 is sm_80 code (-gencode arch=compute_80,code=sm_80); any CUDA >= 12 nvcc builds both.
TORCH_CUDA_ARCH_LIST is set per extension (9.0a / 8.0) for the duration of its build so the binary never depends on the build machine's GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
PAYLOAD = os.path.join(PKG, "triattn_pkg")
ARCH_LIST = {"sm_90a": "9.0a", "sm_80": "8.0"}      # TORCH_CUDA_ARCH_LIST exported for the duration of each extension's build


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def source_digests(seat_dir):
    """{relative path under <dir>/csrc: sha256} over every file of the kernel directory's csrc (sorted) -- the set a loader re-verifies."""
    d = os.path.join(PAYLOAD, seat_dir, "csrc")
    out = {}
    for r, _, fs in os.walk(d):
        if "__pycache__" in r:
            continue
        for f in sorted(fs):
            p = os.path.join(r, f)
            out[os.path.relpath(p, d)] = sha256(p)
    return dict(sorted(out.items()))


def declared_arch(module_file):
    """The -gencode / arch flags the module's _build() passes, read from its source (the module is the source of truth)."""
    txt = open(module_file).read()
    return sorted(set(re.findall(r"arch=compute_\w+,code=\w+", txt)))


def _ver(binary):
    try:
        return subprocess.run([binary, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    except Exception as e:  # noqa: BLE001
        return "unavailable: %s" % e


def build_one(name, seat, out_root, do_loadcheck):
    import torch
    from torch.utils import cpp_extension as ce
    tag = "torch%s-%s" % (torch.__version__, sysconfig.get_config_var("SOABI"))
    dst = os.path.join(out_root, tag); os.makedirs(dst, exist_ok=True)
    sys.path.insert(0, os.path.join(PAYLOAD, seat["dir"]))
    t0 = time.time()
    M = importlib.import_module(seat["module"])
    arch_prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = ARCH_LIST[seat["arch"]]        # the bytes must not depend on the build machine's device or the caller's environment (torch adds the visible device's arch otherwise)
    try:
        ext = M._build(verbose=True)
    finally:
        if arch_prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_prev
    so_src = ext.__file__; ext_name = os.path.splitext(os.path.basename(so_src))[0]
    if ext_name != seat["ext"]:
        sys.exit("build_prebuilt: %s built extension %r, the face expects %r" % (name, ext_name, seat["ext"]))
    so_dst = os.path.join(dst, ext_name + ".so"); shutil.copy2(so_src, so_dst)
    toolkit_ptxas = os.path.join(ce.CUDA_HOME or "/usr/local/cuda", "bin", "ptxas")
    p134 = getattr(M, "PTXAS_134", None)
    used_134 = bool(p134 and os.path.exists(p134) and os.environ.get("TRIATTN_PTXAS", "") != "image")
    rec = {"ext": ext_name, "route": name, "dir": seat["dir"], "module": seat["module"], "stack_tag": tag, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "python": sys.version.split()[0], "soabi": sysconfig.get_config_var("SOABI"), "cxx11abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
           "nvcc": _ver(os.path.join(ce.CUDA_HOME or "/usr/local/cuda", "bin", "nvcc")),
           "ptxas": {"path": p134, "version": _ver(p134)} if used_134 else {"path": toolkit_ptxas, "version": _ver(toolkit_ptxas)},
           "arch": declared_arch(os.path.join(PAYLOAD, seat["dir"], seat["module"] + ".py")),
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build_s": round(time.time() - t0, 1),
           "so_bytes": os.path.getsize(so_dst), "so_sha256": sha256(so_dst), "source_sha256": source_digests(seat["dir"]),
           "module_sha256": {seat["module"] + ".py": sha256(os.path.join(PAYLOAD, seat["dir"], seat["module"] + ".py"))},
           "built_by": "tools/build_prebuilt.py: the kernel directory's own _build() in the target stack", "loadcheck": "pending"}
    if seat.get("needs_cutlass"):
        tagfile = os.path.join(os.environ.get("CUTLASS_PATH", "/opt/cutlass"), "TAG")
        rec["cutlass_tag"] = open(tagfile).read().strip() if os.path.isfile(tagfile) else "unknown (no $CUTLASS_PATH/TAG)"
    rec_path = os.path.join(dst, ext_name + ".json")
    json.dump(rec, open(rec_path, "w"), indent=1, sort_keys=True); open(rec_path, "a").write("\n")
    print("[build_prebuilt] %s: %s (%d bytes, sha256 %s, %.0f s, arch %s) -> %s" % (name, ext_name, rec["so_bytes"], rec["so_sha256"][:16], rec["build_s"], rec["arch"], os.path.relpath(so_dst, PKG)), flush=True)
    if do_loadcheck and out_root == os.path.join(PAYLOAD, "prebuilt"):
        have = torch.cuda.is_available() and "%d.%d" % tuple(torch.cuda.get_device_capability(0)) == seat["cc"]
        if have:
            rc = subprocess.run([sys.executable, os.path.join(PKG, "test_pkg.py"), "--record-loadcheck", "--ext", ext_name], cwd=PKG).returncode
            rec = json.load(open(rec_path))                                # the subprocess rewrote the record
            print("[build_prebuilt] %s: loadcheck %s" % (name, ("recorded %s" % rec.get("loadcheck")) if rc == 0 else "FAILED (record keeps loadcheck: pending)"))
        else:
            print("[build_prebuilt] %s: no cc %s device here -- loadcheck stays pending; run `python test_pkg.py --record-loadcheck` on one" % (name, seat["cc"]))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exts", required=True, help="comma-separated extension names (triattn_sm80_ext, triattn_m1_ext, triattn_mw_ext_g3x4, triattn_sm90_ext) or 'all'")
    ap.add_argument("--out", default=os.path.join(PAYLOAD, "prebuilt"), help="root that receives <stack tag>/<ext>.{so,json} (default: the package's prebuilt/)")
    ap.add_argument("--no-loadcheck", action="store_true", help="do not run the byte gate even if a device of the extension's cc is present")
    a = ap.parse_args()
    sys.path.insert(0, PKG)
    from triattn_pkg._face import CUDA_SEATS
    by_ext = {s["ext"]: dict(s, name=n, cc={"sm_90a": "9.0", "sm_80": "8.0"}[s["arch"]]) for n, s in CUDA_SEATS.items()}
    names = list(by_ext) if a.exts == "all" else [x for x in a.exts.split(",") if x]
    for e in names:
        if e not in by_ext:
            sys.exit("build_prebuilt: unknown extension %r (known: %s)" % (e, ", ".join(by_ext)))
    os.environ["TRIATTN_PKG_PREBUILT"] = "never"          # the modules' _build() must compile, not pick up a shipped binary
    recs = {e: build_one(by_ext[e]["name"], by_ext[e], os.path.abspath(a.out), not a.no_loadcheck) for e in names}
    print("[build_prebuilt] BUILD_RESULTS " + json.dumps({e: {k: r[k] for k in ("stack_tag", "so_sha256", "so_bytes", "build_s", "arch", "loadcheck")} for e, r in recs.items()}))


if __name__ == "__main__":
    main()
