"""Build the flash_transition extension for the running (torch, CUDA) stack into ../prebuilt/<key>/ and write manifest.json.
Needs: nvcc matching torch.version.cuda, ninja, and the CUTLASS (>= 3.5; built with 4.2.0) headers: CUTLASS_INC=<dir containing cute/ and cutlass/>
(e.g. `pip download nvidia-cutlass==4.2.0.0` and unzip; the headers are a build-time dependency only).  Usage:  CUTLASS_INC=... python build.py [--out DIR]"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
import torch
from torch.utils import cpp_extension

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "flash_transition_sm90.cu"
NAME = "flash_transition_sm90a"
FLAGS_CUDA = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-Wno-psabi", "-DNDEBUG",
              "-diag-suppress=177,550,20012", "-Xptxas=--register-usage-level=10"]


def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); ap.add_argument("--build-dir", default=None); args = ap.parse_args()
    inc = os.environ.get("CUTLASS_INC", "")
    if not (inc and os.path.isfile(os.path.join(inc, "cute", "tensor.hpp"))):
        sys.exit("CUTLASS_INC must point at the CUTLASS include directory (cute/tensor.hpp not found)")
    tv = torch.__version__.split("+")[0]; key = f"torch{tv}-cu{(torch.version.cuda or 'none').replace('.', '')}"
    out = args.out or os.path.join(HERE, "..", "prebuilt", key)
    bdir = args.build_dir or os.path.join("/tmp", f"build_{NAME}")
    os.makedirs(out, exist_ok=True); os.makedirs(bdir, exist_ok=True)
    t0 = time.time()
    # compile from a neutral staging copy so the binary embeds no site-specific source path (nvcc records the source file name)
    stage = os.path.join("/tmp", "protenix_fpf_flash_transition", "csrc"); os.makedirs(stage, exist_ok=True)
    staged_src = os.path.join(stage, SRC); shutil.copy(os.path.join(HERE, SRC), staged_src)
    pmap = [f"-Xcompiler=-ffile-prefix-map={stage}=csrc", f"-Xcompiler=-ffile-prefix-map={inc}=cutlass"]
    mod = cpp_extension.load(name=NAME, sources=[staged_src], extra_include_paths=[inc], extra_cuda_cflags=FLAGS_CUDA + pmap,
                             extra_cflags=["-O3", "-std=c++17"] + [f.split("=", 1)[1] for f in pmap], extra_ldflags=["-L/usr/local/cuda/lib64/stubs", "-lcuda"], build_directory=bdir, verbose=True)
    so_built = mod.__file__
    so_name = NAME + ".so"
    shutil.copy(so_built, os.path.join(out, so_name))
    nvcc_v = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    cutlass_ver = "?"
    vf = os.path.join(inc, "cutlass", "version.h")
    if os.path.isfile(vf):
        txt = open(vf).read(); import re
        mm = [re.search(rf"#define CUTLASS_{k} (\d+)", txt) for k in ("MAJOR", "MINOR", "PATCH")]
        if all(mm): cutlass_ver = ".".join(m.group(1) for m in mm)
    man = {"so": so_name, "module_name": NAME, "so_sha256": sha(os.path.join(out, so_name)), "source_sha256": {SRC: sha(os.path.join(HERE, SRC)), "build.py": sha(os.path.join(HERE, "build.py"))},
           "torch": torch.__version__, "cuda": torch.version.cuda, "python_tag": f"cp{sys.version_info[0]}{sys.version_info[1]}", "arch": ["sm_90a"],
           "cutlass_headers": cutlass_ver, "nvcc": nvcc_v, "gpu_at_build": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "source_commit": os.environ.get("SOURCE_COMMIT", ""), "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build_s": round(time.time() - t0, 1),
           "cuda_flags": FLAGS_CUDA, "smem_bytes": int(mod.smem_bytes()), "hidden_chunk": int(mod.hidden_chunk())}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
