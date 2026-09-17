#!/usr/bin/env python3
"""Build the sealed sm_90a transition unit (flash_sm90a/csrc/flash_transition_sm90.cu) for THIS interpreter's ABI key (kernels.transition.flash_abi_key():
torch build + CPython SOABI + arch) with the sealed build recipe (flash_sm90a/csrc/build.py: the same nvcc flags, CUTLASS headers of the recorded version, a
neutral staging path), store it under flash_prebuilt/<key>/ and record it in flash_prebuilt/manifest.json; then confirm that a FRESH interpreter loads it through
the provider face and that the kernel's SiLU table equals torch's on all 65536 bf16 patterns.  Run inside the image whose key you are building for (its torch,
its CUDA toolkit's nvcc), on an sm_90 GPU:

    CUTLASS_INC=<dir with cute/ and cutlass/>  python build_flash_prebuilt.py [--note "<free text: where / who>"]

The sealed directory itself is never written (it stays byte-identical to the unit its kit ships); binaries of other interpreters live in flash_prebuilt/ only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SEALED = os.path.join(HERE, "flash_sm90a")
STORE = os.path.join(HERE, "flash_prebuilt")


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="")
    ap.add_argument("--build-dir", default=None)
    a = ap.parse_args()
    import torch
    from torch.utils import cpp_extension
    spec = importlib.util.spec_from_file_location("sealed_build", os.path.join(SEALED, "csrc", "build.py"))
    B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)             # the sealed recipe: NAME, SRC, FLAGS_CUDA
    inc = os.environ.get("CUTLASS_INC", "")
    if not (inc and os.path.isfile(os.path.join(inc, "cute", "tensor.hpp"))):
        sys.exit("CUTLASS_INC must point at the CUTLASS include directory (cute/tensor.hpp not found)")
    sealed_key = "torch%s-cu%s" % (torch.__version__.split("+")[0], (torch.version.cuda or "none").replace(".", ""))
    sealed_man = json.load(open(os.path.join(SEALED, "prebuilt", sealed_key, "manifest.json"), encoding="utf-8"))
    src = os.path.join(SEALED, "csrc", B.SRC)
    sources = {B.SRC: _sha256(src), "build.py": _sha256(os.path.join(SEALED, "csrc", "build.py"))}
    for fn, d in sources.items():                                                        # build exactly the source the sealed binary was built from
        if sealed_man["source_sha256"].get(fn) != d:
            sys.exit("[build_flash_prebuilt] csrc/%s digest %s.. != sealed manifest %s.." % (fn, d[:12], sealed_man["source_sha256"].get(fn, "?")[:12]))
    if list(B.FLAGS_CUDA) != list(sealed_man["cuda_flags"]):
        sys.exit("[build_flash_prebuilt] the sealed recipe's nvcc flags differ from the sealed manifest's")
    soabi = sysconfig.get_config_var("SOABI") or ("cpython-%d%d" % sys.version_info[:2])
    key = "torch%s-%s-sm90" % (torch.__version__, soabi)
    out = os.path.join(STORE, key); os.makedirs(out, exist_ok=True)
    bdir = a.build_dir or os.path.join("/tmp", "build_%s_%s" % (B.NAME, soabi)); os.makedirs(bdir, exist_ok=True)
    stage = os.path.join("/tmp", "protenix_fpf_flash_transition", "csrc"); os.makedirs(stage, exist_ok=True)   # the sealed recipe's neutral staging path
    staged = os.path.join(stage, B.SRC); shutil.copy(src, staged)
    pmap = ["-Xcompiler=-ffile-prefix-map=%s=csrc" % stage, "-Xcompiler=-ffile-prefix-map=%s=cutlass" % inc]
    t0 = time.time()
    mod = cpp_extension.load(name=B.NAME, sources=[staged], extra_include_paths=[inc], extra_cuda_cflags=list(B.FLAGS_CUDA) + pmap,
                             extra_cflags=["-O3", "-std=c++17"] + [f.split("=", 1)[1] for f in pmap], extra_ldflags=["-L/usr/local/cuda/lib64/stubs", "-lcuda"],
                             build_directory=bdir, verbose=True)
    so_name = B.NAME + ".so"
    shutil.copy(mod.__file__, os.path.join(out, so_name))
    nvcc_v = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    cutlass_ver = "?"
    vf = os.path.join(inc, "cutlass", "version.h")
    if os.path.isfile(vf):
        import re
        txt = open(vf).read()
        mm = [re.search(r"#define CUTLASS_%s (\d+)" % k, txt) for k in ("MAJOR", "MINOR", "PATCH")]
        if all(mm):
            cutlass_ver = ".".join(m.group(1) for m in mm)
    if cutlass_ver != sealed_man.get("cutlass_headers"):
        sys.exit("[build_flash_prebuilt] CUTLASS headers %s != the sealed build's %s" % (cutlass_ver, sealed_man.get("cutlass_headers")))
    ent = {"so": "%s/%s" % (key, so_name), "so_sha256": _sha256(os.path.join(out, so_name)), "torch": torch.__version__, "cuda": torch.version.cuda,
           "python_tag": "cp%d%d" % sys.version_info[:2], "soabi": soabi, "arch": ["sm_90a"], "cuda_flags": list(B.FLAGS_CUDA), "cutlass_headers": cutlass_ver,
           "nvcc": nvcc_v, "gpu_at_build": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "build_s": round(time.time() - t0, 1), "smem_bytes": int(mod.smem_bytes()), "hidden_chunk": int(mod.hidden_chunk()), "note": a.note}
    if (ent["smem_bytes"], ent["hidden_chunk"]) != (sealed_man["smem_bytes"], sealed_man["hidden_chunk"]):
        sys.exit("[build_flash_prebuilt] smem_bytes / hidden_chunk differ from the sealed build's")
    man_p = os.path.join(STORE, "manifest.json")
    man = json.load(open(man_p, encoding="utf-8")) if os.path.isfile(man_p) else {"unit": "flash_sm90a", "module_name": B.NAME, "sources": sources, "binaries": {}}
    if man.get("sources") != sources:
        sys.exit("[build_flash_prebuilt] the store manifest names other source digests; refuse to mix builds of different sources")
    man["binaries"][key] = ent
    with open(man_p, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, sort_keys=True); f.write("\n")
    # a fresh interpreter loads the stored binary through the face and checks the SiLU table against torch on every bf16 pattern
    chk = r'''
import sys, json, torch
import torch.nn.functional as F
sys.path.insert(0, %r)
from opt_core.kernels import transition as T
E = T._flash_ext()
dev = torch.device("cuda")
bits = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16)
want = F.silu(bits); got = torch.empty(65536, dtype=torch.bfloat16, device=dev); E.silu_table(got)
nan2 = want.isnan() & got.isnan()
bad = int(((want.view(torch.int16) != got.view(torch.int16)) & ~nan2).sum())
print(json.dumps({"loaded_from": E.__file__, "abi_key": T.flash_abi_key(), "silu_mismatch": bad, "smem_bytes": int(E.smem_bytes()), "hidden_chunk": int(E.hidden_chunk())}))
sys.exit(1 if bad else 0)
''' % os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    r = subprocess.run([sys.executable, "-c", chk], capture_output=True, text=True)
    print(r.stdout.strip()); print(r.stderr.strip()[-2000:], file=sys.stderr)
    if r.returncode != 0:
        sys.exit("[build_flash_prebuilt] fresh-interpreter load check FAILED")
    rec = json.loads(r.stdout.strip().splitlines()[-1])
    ent["loadcheck"] = {"silu_mismatch": rec["silu_mismatch"], "fresh_interpreter": True}
    man["binaries"][key] = ent
    with open(man_p, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, sort_keys=True); f.write("\n")
    print(json.dumps({"key": key, "entry": ent}, indent=1))


if __name__ == "__main__":
    main()
