#!/usr/bin/env python3
"""Build the prebuilt extension of fpf_triattn_cuda for THIS process's stack + write the load-time load-check data, then confirm that a
fresh interpreter loads it by the prebuilt route with a bitwise load-check.  Run on the target GPU class with the target torch / CUDA
toolkit (nvcc on PATH):

    python build_prebuilt.py [--out <dir, default: this package's prebuilt/>] [--copy-to <dir>] [--note "<who / where>"]

Writes <out>/<stack key>/{triattn_cuda_sm90.so, manifest.json, loadcheck.pt}; stack key = torch version + python SOABI + sm, e.g.
torch2.13.0+cu130-cpython-311-x86_64-linux-gnu-sm90.  Another stack gets its own directory from the same command run on that stack
(the kernel source targets sm_90a; install() refuses other compute capabilities by name)."""
import argparse, glob, json, os, shutil, subprocess, sys, sysconfig, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
PKG = os.path.basename(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "prebuilt"))
    ap.add_argument("--copy-to", default="", help="also copy the stack-key directory here")
    ap.add_argument("--note", default="", help="free text recorded in manifest.json (builder, machine)")
    a = ap.parse_args()
    import torch
    import importlib
    T = importlib.import_module(PKG)          # this package, by its directory name (fpf_triattn_cuda or the integrated protenix_fpf_triattn_cuda)
    key = T.stack_key()
    dst = os.path.join(a.out, key); os.makedirs(dst, exist_ok=True)
    bdir = tempfile.mkdtemp(prefix="triattn_build_")
    t0 = time.time()
    mod = T.build_ext(build_dir=bdir, verbose=True)
    built = time.time() - t0
    sos = glob.glob(os.path.join(bdir, T.EXT_NAME + "*.so"))
    assert len(sos) == 1, sos
    so_name = T.EXT_NAME + ".so"
    shutil.copy2(sos[0], os.path.join(dst, so_name))
    # bake the load-check with the freshly built module
    T._EXT = mod
    hashes = T.bake_loadcheck(os.path.join(dst, "loadcheck.pt"))
    nvcc = subprocess.run("nvcc --version | tail -1", shell=True, capture_output=True, text=True).stdout.strip()
    man = {"so": so_name, "module_name": T.EXT_NAME, "so_sha256": T._sha(os.path.join(dst, so_name)), "stack_key": key,
           "torch": torch.__version__, "cuda": torch.version.cuda, "python_soabi": sysconfig.get_config_var("SOABI"), "python": sys.version.split()[0],
           "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability()), "nvcc": nvcc, "nvcc_flags": T.NVCC_FLAGS,
           "source_sha256": T.source_sha256(), "loadcheck_sha256_16": hashes, "loadcheck_cases": list(hashes),
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build_s": round(built, 1), "note": a.note}
    json.dump(man, open(os.path.join(dst, "manifest.json"), "w"), indent=1, sort_keys=True)
    print(f"[build_prebuilt] wrote {dst}: {os.listdir(dst)} ({built:.0f}s build)")
    # fresh interpreter: must take the prebuilt route and pass the bitwise load-check
    code = (f"import sys, json, importlib; sys.path.insert(0, {os.path.dirname(HERE)!r}); T = importlib.import_module({PKG!r}); r = T.install(prebuilt_root={os.path.abspath(a.out)!r}); "
            "print(json.dumps(r, default=str)); assert r['route'] == 'prebuilt' and r['loadcheck']['ok'] and all(c.get('bitwise') for c in r['loadcheck']['cases'].values()), r")
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=os.path.dirname(HERE), capture_output=True, text=True)
    print("[build_prebuilt] fresh-process install:", r.stdout.strip()[-1500:], r.stderr.strip()[-600:] if r.returncode else "")
    if r.returncode:
        sys.exit("[build_prebuilt] FRESH-PROCESS PREBUILT LOAD FAILED")
    if a.copy_to:
        cdst = os.path.join(a.copy_to, key); os.makedirs(a.copy_to, exist_ok=True)
        if os.path.exists(cdst): shutil.rmtree(cdst)
        shutil.copytree(dst, cdst); print(f"[build_prebuilt] copied to {cdst}")
    print("[build_prebuilt] OK", key, man["so_sha256"][:16])


if __name__ == "__main__":
    main()
