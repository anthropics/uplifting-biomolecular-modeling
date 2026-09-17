#!/usr/bin/env python3
"""Build the active package's CUDA extensions for THIS interpreter's stack (torch version + python SOABI), byte-gate them on this device, and write
one prebuilt directory ``<out>/<stack key>/{<ext>.so, <ext>.json}`` ready to drop into ``pkg/<ACTIVE_PKG>/triattn_pkg/prebuilt/``.

    python build_prebuilt.py --out /tmp/prebuilt_out --cutlass /path/holding/include [--exts triattn_sm90_ext,triattn_m1_ext,triattn_mw_ext_g3x4] [--note TEXT]
    python build_prebuilt.py --out /tmp/prebuilt_out --exts triattn_sm80_ext          (generation >= 11, in an A100 image: the sm_80 member; no third-party headers)

Default ``--exts``: the extensions of THIS device's compute capability (cc 9.0: the three sm_90a extensions; cc 8.0: triattn_sm80_ext).
Needs: a device of the extensions' cc (the byte gate runs the kernels), nvcc + ninja of the image, CUTLASS >= 4 headers for triattn_sm90_ext /
triattn_m1_ext (``--cutlass``: the directory whose ``include/cute/tensor.hpp`` exists; triattn_mw_ext_g3x4 and triattn_sm80_ext need none).  The
extensions are compiled by the package's own ``_build()`` functions (their flags, unmodified); the record repeats the producer's fields and adds the
digests the face verifies (``so_sha256``, ``source_sha256``, ``loadcheck``).  Ends with a fresh-interpreter ``install()`` through the face against
the just-written directory.  ``--skip-loadcheck 1`` writes ``"loadcheck": "pending"`` (a payload whose vectors for this cc are not recorded yet).
"""
import argparse, datetime, hashlib, importlib, json, os, re, shutil, subprocess, sys, sysconfig, tempfile, time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--cutlass", default=os.environ.get("CUTLASS_PATH", "/opt/cutlass"))
ap.add_argument("--exts", default="", help="default: the extensions of this device's cc")
ap.add_argument("--note", default="")
ap.add_argument("--fresh-check", default="1")
ap.add_argument("--skip-loadcheck", default="0")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))          # .../common/opt_core
sys.path.insert(0, CORE)
if not os.environ.get("TORCH_EXTENSIONS_DIR"):
    os.environ["TORCH_EXTENSIONS_DIR"] = tempfile.mkdtemp(prefix="te_triattn_native-")     # a private build directory (0700) unless one is named
os.environ["CUTLASS_PATH"] = args.cutlass
os.environ.setdefault("MAX_JOBS", "8")
import torch
from opt_core.kernels.triattn import triattn_native as CC

key = CC.stack_key()
dev_cc = tuple(torch.cuda.get_device_capability())
if not args.exts:
    args.exts = ",".join(CC.extensions_for_cc(dev_cc))
outdir = os.path.join(args.out, key); os.makedirs(outdir, exist_ok=True)
print("[build] pkg %s stack %s device %s cc %s torch %s cuda %s" % (CC.ACTIVE_PKG, key, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.__version__, torch.version.cuda), flush=True)
M = CC._module()                                    # payload imported (sets policy 'always'); the build wants the JIT path:
os.environ["TRIATTN_PKG_PREBUILT"] = "never"
F = sys.modules[M.__name__ + "._face"]
F._router()                                         # the payload's directories are on sys.path now
have_cutlass = os.path.isfile(os.path.join(args.cutlass, "include", "cute", "tensor.hpp"))
cutlass_tag = None
vh = os.path.join(args.cutlass, "include", "cutlass", "version.h")
if os.path.isfile(vh):
    t = open(vh).read(); cutlass_tag = "v%s.%s.%s" % tuple(re.search(r"#define CUTLASS_%s (\d+)" % w, t).group(1) for w in ("MAJOR", "MINOR", "PATCH"))
nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1] if shutil.which("nvcc") else "?"
ptxas = subprocess.run(["ptxas", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1] if shutil.which("ptxas") else "?"
EXTENSIONS = {"triattn_sm90_ext": ("cuda", "triattn_cuda", "_EXT"), "triattn_mw_ext_g3x4": ("cuda_c", "triattn_mw", "_EXTS[3x4]"), "triattn_m1_ext": ("cuda_b", "triattn_m1", "_EXT"),
              "triattn_sm80_ext": ("cuda_80", "triattn_sm80", "_EXT")}
NEEDS_CUTLASS = ("triattn_sm90_ext", "triattn_m1_ext")
ARCH = {"triattn_sm90_ext": "9.0a (-gencode arch=compute_90a,code=sm_90a)", "triattn_mw_ext_g3x4": "9.0a (-gencode arch=compute_90a,code=sm_90a)",
        "triattn_m1_ext": "9.0a (-gencode arch=compute_90a,code=sm_90a)", "triattn_sm80_ext": "8.0 (-gencode arch=compute_80,code=sm_80)"}
built, skipped = {}, {}
from torch.utils.cpp_extension import _get_build_directory
for ext in args.exts.split(","):
    route_dir, modname, slot = EXTENSIONS[ext]
    if ext in NEEDS_CUTLASS and not have_cutlass:
        skipped[ext] = "no CUTLASS headers at %s" % args.cutlass; print("[build] SKIP %s: %s" % (ext, skipped[ext]), flush=True); continue
    if tuple(CC.EXTENSIONS[ext]["cc"]) != dev_cc and args.skip_loadcheck != "1":
        skipped[ext] = "device cc %s cannot byte-gate %s code (pass --skip-loadcheck 1 to build with loadcheck pending)" % (dev_cc, CC.EXTENSIONS[ext]["arch"]); print("[build] SKIP %s: %s" % (ext, skipped[ext]), flush=True); continue
    t0 = time.time()
    mod = importlib.import_module(modname)
    e = mod._build(geom="3x4") if ext == "triattn_mw_ext_g3x4" else mod._build()
    bdir = _get_build_directory(ext, False)
    so_src = os.path.join(bdir, ext + ".so")
    assert os.path.isfile(so_src), (ext, bdir, os.listdir(bdir) if os.path.isdir(bdir) else None)
    so_dst = os.path.join(outdir, ext + ".so"); shutil.copy2(so_src, so_dst)
    so_sha = hashlib.sha256(open(so_dst, "rb").read()).hexdigest()
    print("[build] %s built in %.0f s: %s (%d bytes, sha256 %s)" % (ext, time.time() - t0, so_dst, os.path.getsize(so_dst), so_sha[:16]), flush=True)
    # byte gate through the payload with the just-built extension injected (policy 'never' = the JIT object the module now caches)
    lc = CC.loadcheck(cases=list(CC.LOADCHECK_CASES[ext]), key=None) if args.skip_loadcheck != "1" else None
    rec = {"ext": ext, "route": {"cuda": "cuda", "cuda_c": "mw", "cuda_b": "m1", "cuda_80": "cuda_80"}[route_dir], "route_dir": route_dir, "module": modname, "pkg": CC.ACTIVE_PKG,
           "stack_tag": key, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "python": sys.version.split()[0], "soabi": sysconfig.get_config_var("SOABI"),
           "cxx11abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI), "nvcc": nvcc, "ptxas": {"path": shutil.which("ptxas"), "version": ptxas}, "cutlass_tag": cutlass_tag if ext != "triattn_mw_ext_g3x4" else None,
           "arch": ARCH[ext], "built_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "so_bytes": os.path.getsize(so_dst),
           "device": torch.cuda.get_device_name(), "built_by": "the kernel directory's own _build() through build_prebuilt.py", "note": args.note,
           "so_sha256": so_sha, "source_sha256": CC.source_sha256(ext),
           "module_sha256": {modname + ".py": hashlib.sha256(open(os.path.join(CC.payload_dir(), route_dir, modname + ".py"), "rb").read()).hexdigest()},
           "loadcheck": ({cid: r["sha256_16"] for cid, r in lc.items()} if lc is not None else "pending"), "loadcheck_bitwise": ({cid: r["bitwise"] for cid, r in lc.items()} if lc is not None else "pending")}
    json.dump(rec, open(os.path.join(outdir, ext + ".json"), "w"), indent=1, sort_keys=True); open(os.path.join(outdir, ext + ".json"), "a").write("\n")
    built[ext] = rec; print("[build] %s loadcheck %s" % (ext, {cid: (r["route"], r["bitwise"], r["sha256_16"]) for cid, r in lc.items()} if lc is not None else "pending"), flush=True)
print("[build] built %s skipped %s -> %s" % (sorted(built), skipped, outdir), flush=True)
if args.fresh_check == "1" and built and args.skip_loadcheck != "1":
    # the shipped path: drop the directory into the payload's prebuilt/ and install() in a fresh interpreter (policy 'always': prebuilt or refusal)
    dst = os.path.join(CC.payload_dir(), "prebuilt", key)
    if os.path.abspath(dst) != os.path.abspath(outdir):
        if os.path.isdir(dst): shutil.rmtree(dst)
        shutil.copytree(outdir, dst)
    code = ("import json, sys; sys.path.insert(0, %r); from opt_core.kernels.triattn import triattn_native as CC; rep = CC.install(check=True); "
            "print('[fresh] install ok', json.dumps({'stack_key': rep['stack_key'], 'manifests': {k: v['digests_in_record'] for k, v in rep['manifests'].items()}, "
            "'loadcheck': {k: (v['route'], v['bitwise'], v['sha256_16']) for k, v in rep['loadcheck'].items()}}, default=str))") % CORE
    env = dict(os.environ); env.pop("TRIATTN_PKG_PREBUILT", None)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    print(r.stdout.strip()[-1500:] or r.stderr.strip()[-1500:], flush=True)
    print("[build] fresh-interpreter install rc %d" % r.returncode, flush=True)
    sys.exit(r.returncode)
