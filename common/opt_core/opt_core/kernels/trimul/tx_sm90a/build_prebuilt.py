#!/usr/bin/env python3
"""Build this package's sm_90a extension for THIS process's ABI key (ops.abi_tag(): torch version, CPython SOABI, arch) from the shipped csrc with
the manifest's nvcc flags, record it under prebuilt/<key>/ + prebuilt/manifest.json, and confirm that a fresh interpreter loads it through
ops.load() (binary + source digests, then the load-time kernel fingerprints, which must equal the manifest's: the new binary reproduces the
recorded kernels bit for bit).  Run inside the image whose key you are building for, on an sm_90 GPU, with that image's CUDA toolkit (nvcc):

    python build_prebuilt.py [--out <prebuilt dir, default: this package's>] [--note "<who / where>"]
"""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, sysconfig, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = "protenix_trimul_tx_sm90"
NVCC_FLAGS = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr", "-DNDEBUG", "-Xptxas=--register-usage-level=10"]


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "prebuilt"))
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    import torch
    from torch.utils import cpp_extension
    sys.path.insert(0, os.path.dirname(HERE))
    pkg = os.path.basename(HERE)
    ops = __import__(pkg + ".ops", fromlist=["ops"])
    key = ops.abi_tag()
    man_p = os.path.join(a.out, "manifest.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {"binaries": {}, "sources": {}, "loadcheck": {}, "package": pkg}
    for rel, want in man.get("sources", {}).items():                                   # build exactly the recorded sources
        got = _sha256(os.path.join(HERE, rel))
        if got != want:
            sys.exit("[build_prebuilt] source %s digest %s.. != manifest %s.." % (rel, got[:12], want[:12]))
    scratch = tempfile.mkdtemp(prefix="tx_build_")
    src = os.path.join(scratch, "csrc"); shutil.copytree(os.path.join(HERE, "csrc"), src)   # path-neutral build: the scratch prefix is mapped to csrc
    t0 = time.time()
    cpp_extension.load(name=MODULE, sources=[os.path.join(src, "trimul_tx.cu")], extra_cuda_cflags=NVCC_FLAGS + ["-Xcompiler", "-ffile-prefix-map=%s=csrc" % src],
                       extra_cflags=["-O3", "-ffile-prefix-map=%s=csrc" % src], build_directory=scratch, verbose=True)
    built = time.time() - t0
    sos = glob.glob(os.path.join(scratch, MODULE + "*.so")); assert len(sos) == 1, sos
    dst = os.path.join(a.out, key); os.makedirs(dst, exist_ok=True)
    so = os.path.join(dst, MODULE + ".so"); shutil.copy2(sos[0], so)
    nvcc = subprocess.run("nvcc --version | tail -1", shell=True, capture_output=True, text=True).stdout.strip()
    gcc = subprocess.run("gcc --version | head -1", shell=True, capture_output=True, text=True).stdout.strip()
    ent = {"file": "%s/%s.so" % (key, MODULE), "module": MODULE, "sha256": _sha256(so), "bytes": os.path.getsize(so), "python_soabi": sysconfig.get_config_var("SOABI"),
           "nvcc_flags": NVCC_FLAGS + ["-Xcompiler -ffile-prefix-map=<scratch>=csrc"], "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build_s": round(built, 1),
           "toolchain": {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0], "nvcc": nvcc, "gcc": gcc, "device_built_on": torch.cuda.get_device_name(0)}}
    if a.note:
        ent["note"] = a.note
    man.setdefault("binaries", {})[key] = ent
    json.dump(man, open(man_p, "w"), indent=1, sort_keys=True); open(man_p, "a").write("\n")
    json.dump({"key": key, **ent}, open(os.path.join(dst, "PROVENANCE.json"), "w"), indent=1, sort_keys=True)
    print("[build_prebuilt] wrote %s (%.0fs build); manifest entry %s" % (dst, built, key))
    code = ("import sys, json; sys.path.insert(0, %r); ops = __import__(%r, fromlist=['ops']); ops.load(); st = ops._STATE; "
            "print(json.dumps({'loaded': st['loaded'], 'loadcheck_equal': st['loadcheck']['got'] == {k: v for k, v in st['loadcheck']['want'].items() if k in st['loadcheck']['got']}, "
            "'got': st['loadcheck']['got'], 'want': st['loadcheck']['want']}, default=str))") % (os.path.dirname(HERE), pkg + ".ops")
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True)
    print("[build_prebuilt] fresh-process load:", r.stdout.strip()[-1500:], r.stderr.strip()[-800:] if r.returncode else "")
    if r.returncode:
        sys.exit("[build_prebuilt] FRESH-PROCESS LOAD FAILED (the binary is left in place for inspection; the manifest entry names it)")
    print("[build_prebuilt] OK", key, ent["sha256"][:16])


if __name__ == "__main__":
    main()
