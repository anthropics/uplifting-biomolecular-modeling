#!/usr/bin/env python3
"""build.py — compile csrc/triatt_procuda_sm90.cu into prebuilt/libtriatt_procuda_sm90a.so and write prebuilt/manifest.json (run once on the image that
ships the kit; the runtime never compiles).  Usage: python build.py [--nvcc /usr/local/cuda/bin/nvcc]"""
import os, sys, json, subprocess, hashlib, datetime, argparse

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc")); args = ap.parse_args()
    src_dir = os.path.join(HERE, "csrc"); out_dir = os.path.join(HERE, "prebuilt"); os.makedirs(out_dir, exist_ok=True)
    so = os.path.join(out_dir, "libtriatt_procuda_sm90a.so")
    cmd = [args.nvcc, "-O3", "-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a", "--shared", "-Xcompiler", "-fPIC",
           "-Xcompiler", f"-ffile-prefix-map={src_dir}=.", "-o", so, "triatt_procuda_sm90.cu"]
    subprocess.run(cmd, cwd=src_dir, check=True)          # relative source name + prefix map: no build paths inside the binary
    ver = subprocess.run([args.nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    man = {"so": os.path.basename(so), "so_sha256": sha256(so), "src": "csrc/triatt_procuda_sm90.cu", "src_sha256": sha256(os.path.join(src_dir, "triatt_procuda_sm90.cu")),
           "arch": "sm_90a", "nvcc": ver, "built_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1); f.write("\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
