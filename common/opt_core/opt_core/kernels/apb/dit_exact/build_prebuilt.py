#!/usr/bin/env python3
"""build_prebuilt.py — build prebuilt/<torch>-cu<cuda>-sm<cc>/dit_attn_exact.so from csrc/ on the pinned image (GPU process), check it in this
process (load through the package loader path: manifest, load-time check vs torch SDPA) and write manifest.json + PROVENANCE.md.
  python3 build_prebuilt.py [--out prebuilt/<key>]        (needs nvcc from the CUDA toolkit matching torch.version.cuda, ninja)
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch
import protenix_fpf_dit_attn_exact as L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); ap.add_argument("--build-dir", default=None)
    a = ap.parse_args()
    from torch.utils.cpp_extension import load
    cc = torch.cuda.get_device_capability(); arch = f"{cc[0]}{cc[1]}"
    garch = arch + "a" if cc == (9, 0) else arch          # the kernel uses sm_90a instructions (wgmma, setmaxnreg)
    key = L.stack_key(); out = a.out or L.prebuilt_dir(key)
    modname = f"dit_attn_exact_v7_sm{arch}"
    bdir = a.build_dir or os.path.join("/tmp", f"build_{modname}")
    os.makedirs(bdir, exist_ok=True); os.makedirs(out, exist_ok=True)
    # build from a neutral staging copy of csrc/ so no build-host path is embedded in the binary (assert strings carry __FILE__)
    stage = os.path.join(bdir, "protenix_fpf_dit_attn_exact"); shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
    shutil.copy(os.path.join(HERE, "csrc", "dit_attn_exact.cu"), os.path.join(stage, "dit_attn_exact.cu"))
    src = os.path.join(stage, "dit_attn_exact.cu")
    pmap = f"-ffile-prefix-map={bdir}/=./"
    flags = ["-O3", "-std=c++17", f"-gencode=arch=compute_{garch},code=sm_{garch}", "--expt-relaxed-constexpr", "-Xptxas=-O3", "-Xcompiler", pmap]
    t0 = time.time()
    cwd = os.getcwd(); os.chdir(bdir)
    try:
        mod = load(name=modname, sources=[os.path.relpath(src, bdir)], extra_cuda_cflags=flags, extra_cflags=["-O3", "-std=c++17", pmap], build_directory=bdir, verbose=True)
    finally:
        os.chdir(cwd)
    build_s = round(time.time() - t0, 1)
    so_built = mod.__file__
    so_name = "dit_attn_exact.so"
    shutil.copy(so_built, os.path.join(out, so_name))
    blob = open(os.path.join(out, so_name), "rb").read()
    for needle in (bdir.encode(), HERE.encode(), b"/work/", os.path.expanduser("~").encode()):
        if len(needle) > 1 and needle in blob:
            sys.exit(f"build refused: the binary embeds the build path {needle!r} (prefix map ineffective)")
    mod.init()
    sc = L.run_loadcheck(mod, None)
    if not sc["bit_equal_vs_sdpa"]:
        sys.exit(f"build check FAILED: not bit-identical to SDPA in the build process: {sc}")
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    man = {"lever": "dit_attn_exact", "kernel_version": str(mod.version), "module_name": modname, "so": so_name,
           "so_sha256": L._sha_file(os.path.join(out, so_name)), "source_sha256": L.source_sha256(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "arch": [f"sm{arch}"], "python": sys.version.split()[0],
           "gpu": torch.cuda.get_device_name(0), "nvcc": nvcc, "nvcc_flags": [f for f in flags if f != "-Xcompiler" and not f.startswith("-ffile-prefix-map")] + ["-Xcompiler -ffile-prefix-map=<build dir>/=./"], "build_s": build_s,
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "loadcheck_cases": sc["cases"], "loadcheck_digests": sc["digests"], "loadcheck_bit_equal_vs_sdpa_at_build": sc["bit_equal_vs_sdpa"]}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    open(os.path.join(out, "PROVENANCE.md"), "w").write(
        f"# dit_attn_exact prebuilt ({key})\n\n"
        f"- built from `csrc/dit_attn_exact.cu` (sha256 {man['source_sha256']['dit_attn_exact.cu']}) by `build_prebuilt.py`\n"
        f"- torch {man['torch']}, CUDA {man['cuda']}, {nvcc}; flags: {' '.join(man['nvcc_flags'])}; arch sm_{arch}; python {man['python']}\n"
        f"- device at build: {man['gpu']}; built {man['built_utc']} in {build_s} s\n"
        f"- {so_name} sha256 {man['so_sha256']}\n"
        f"- load-time check: {sc['cases']} cases bit-identical to torch.nn.functional.scaled_dot_product_attention (fp32, memory-efficient kernel) at build; output digests recorded in manifest.json\n")
    # reload through the package loader (the run-time path) as the final check
    L._MOD = None
    m2 = L.load_prebuilt(key)
    print(json.dumps({"out": out, "manifest": man, "reload_loadcheck": L.STATS["loadcheck"]}, indent=1))


if __name__ == "__main__":
    main()
