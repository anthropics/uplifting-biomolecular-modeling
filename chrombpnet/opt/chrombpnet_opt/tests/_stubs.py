"""Stand-ins for the CPU tests: a stub kit tree (the documented entry script that records its argv and environment, a fastdefault with
the kit's own function names whose GPU class comes from STUB_GPU_CLASS, the seed hook reading the recipe's two names, README), a
stub stock console script `chrombpnet` that records its environment, and a stub `chrombpnet` package for the finder
tests. Nothing here imports tensorflow, torch or chrombpnet. The stand-in kit lives at <tree>/opt/kit of a stand-in tree (the package
locates it through CHROMBPNET_OPT_HOME=<tree>, as run.sh does from its own location); a stand-in opt/kit_ho beside it serves the --items
tests. The real kits, when present in this tree (opt/kit, opt/kit_ho), are used by the tests that say so."""
import os
import stat

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # opt/chrombpnet_opt
OPT_DIR = os.path.dirname(PKG_DIR)                                             # opt
TREE_DIR = os.path.dirname(OPT_DIR)                                            # chrombpnet/
ROUTES = ("k1", "tf_function", "keras_predict_fileorder", "stock_cli")

FASTDEFAULT_STUB = '''"""stub fastdefault: the kit's function names, a class from STUB_GPU_CLASS, routes by (class, mode) from a tiny table."""
import os
_TABLE = {"H100": {"prod": "k1", "det": "k1"}, "H200": {"prod": "k1", "det": "k1"}, "A100": {"prod": "k1", "det": "tf_function"}, "unknown": {"prod": "keras_predict_fileorder", "det": "tf_function"}}
IMAGE_TORCH_TREES = tuple(p for p in os.environ.get("STUB_TORCH_TREES", "/opt/torch").split(os.pathsep) if p)   # the kit's list, from the environment for the tests
def nvsmi_query():
    c = os.environ.get("STUB_GPU_CLASS")
    return {"error": "nvidia-smi unavailable"} if not c else {"name": "NVIDIA " + c + " STUB", "driver": "stub"}
def detect_gpu():
    c = os.environ.get("STUB_GPU_CLASS")
    return ("unknown", "nvidia-smi unavailable (stub)", None) if not c else (c if c in _TABLE else "unknown", "NVIDIA " + c + " STUB", "8.0" if c == "A100" else "9.0")
def route_mode():
    return "det" if any(os.environ.get(k) == "1" for k in ("TF_DETERMINISTIC_OPS", "TF_USE_DEFAULT_CONV_ALGO", "CHROMBPNET_DET_SUBPROCESS")) else "prod"
def arch_key(cap, cls=None):
    return ("cuda-" + str(cap).replace(".", "")) if cap else None
def arch_default_route(kit_root, arch, mode=None):
    return None                                                   # no arch-keyed entry in the stub: the class table decides
def class_route(cls, cap, mode="prod"):
    return _TABLE.get(cls, _TABLE["unknown"])[mode] == "k1", "stub class table"
def resolve(kit_root=None):
    import sys, types
    sys.modules["_stub_probe_marker"] = types.ModuleType("_stub_probe_marker")   # the stand-in for the kit's K1 stack probe (torch import)
    cls, name, cap = detect_gpu(); mode = route_mode()
    return {"gpu_class": cls, "gpu_name": name, "compute_cap": cap, "mode": mode, "forward": _TABLE.get(cls, _TABLE["unknown"])[mode],
            "forward_reason": "stub table", "native_dilation": False, "native_dilation_reason": "stub", "kit_version": "0.0.0-stub", "torch_stack": {"present": False}}
'''

PRED_BW_FAST_STUB = '''"""stub entry script: records argv and environment beside the -op prefix (beside the --items file under --items), writes the kit's run
record (one entry per prefix: the forward that ran by the stub table, the fastdefault record, the untar stamp, the outputs) to the file named by
CHROMBPNET_FASTKIT_RECORD — STUB_STAMP_FORWARD overrides the forward, STUB_STAMP_SKIPPED adds a fallback reason, STUB_JIT_CACHE_FALLBACK a cache
fallback, STUB_NO_RECORD=1 writes no record, STUB_NO_OUTPUTS=1 lists outputs without writing them, STUB_RECORD_SHORT=1 leaves the last item off
the record, STUB_REFUSED=<reason> refuses the mode by name (the record's `refused`, no outputs, exit 3) — and exits with STUB_RC (default 0)."""
import json, os, sys
argv = sys.argv[1:]
op = argv[argv.index("-op") + 1] if "-op" in argv else (argv[argv.index("--items") + 1] if "--items" in argv else "stub")
os.makedirs(os.path.dirname(os.path.abspath(op)) or ".", exist_ok=True)
with open(op + "_kit_call.json", "w") as fh:
    json.dump({"argv": argv, "env": {k: v for k, v in os.environ.items() if k.startswith(("CHROMBPNET", "K1_", "TF_", "CUBLAS_", "CUDA_", "MODEL_OPT", "PYTHONPATH", "HOME_X"))}, "cwd": os.getcwd(), "python": sys.executable}, fh)
_TABLE = {"H100": {"prod": "k1", "det": "k1"}, "H200": {"prod": "k1", "det": "k1"}, "A100": {"prod": "k1", "det": "tf_function"}, "unknown": {"prod": "keras_predict_fileorder", "det": "tf_function"}}
_cls = os.environ.get("STUB_GPU_CLASS") or "unknown"
if os.environ.get("STUB_REFUSED"):
    print("[pred_bw_fast] REFUSED: " + os.environ["STUB_REFUSED"], flush=True)
    if os.environ.get("CHROMBPNET_FASTKIT_RECORD"): json.dump({"kit": "chrombpnet_fastkit", "items": [], "refused": os.environ["STUB_REFUSED"]}, open(os.environ["CHROMBPNET_FASTKIT_RECORD"], "w"))
    sys.exit(3)
_mode = "det" if any(os.environ.get(k) == "1" for k in ("TF_DETERMINISTIC_OPS", "TF_USE_DEFAULT_CONV_ALGO", "CHROMBPNET_DET_SUBPROCESS")) else "prod"
_fwd = os.environ.get("STUB_STAMP_FORWARD") or _TABLE.get(_cls, _TABLE["unknown"])[_mode]
_skipped = [os.environ["STUB_STAMP_SKIPPED"]] if os.environ.get("STUB_STAMP_SKIPPED") else []
_prefixes = [op]
if "--items" in argv:
    _prefixes = [l.rstrip("\\n").split("\\t")[1] for l in open(op) if l.strip() and not l.lstrip().startswith("#")]
_recorded = _prefixes if os.environ.get("STUB_RECORD_SHORT") != "1" or len(_prefixes) < 2 else _prefixes[:-1]
_doc = {"kit": "chrombpnet_fastkit", "items": []}
for p in _prefixes:
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    outs = [os.path.basename(p) + "_chrombpnet.bw", os.path.basename(p) + "_chrombpnet_preds.bed"]
    if os.environ.get("STUB_NO_OUTPUTS") != "1":
        for o in outs:
            open(os.path.join(os.path.dirname(os.path.abspath(p)), o), "w").write("stub\\n")
    if os.environ.get("STUB_NO_RECORD") != "1" and p in _recorded:
        stamp = {"kit": "chrombpnet_fastkit", "version": "0.0.0-stub", "forward": _fwd + " (stub)", "import_witness": {"forward": _fwd, "tensorflow_loaded": _fwd != "k1"},
                 "fastdefault": {"gpu_class": _cls, "mode": _mode, "forward": _fwd, "forward_reason": "stub table", "skipped": _skipped}, "outputs": outs,
                 "jit_cache": ({"tarball": os.environ.get("CHROMBPNET_JIT_CACHE_TAR"), "untar_s": 0.01, "fallback": os.environ.get("STUB_JIT_CACHE_FALLBACK") or None} if os.environ.get("CHROMBPNET_JIT_CACHE_TAR") else {})}
        _doc["items"].append(dict(stamp, item=len(_doc["items"]) + 1, prefix=p))
if os.environ.get("CHROMBPNET_FASTKIT_RECORD"):
    json.dump(_doc, open(os.environ["CHROMBPNET_FASTKIT_RECORD"], "w"))
print("[stub kit] ran", flush=True)
sys.exit(int(os.environ.get("STUB_RC", "0")))
'''

SITECUSTOMIZE_STUB = '''"""stub seed hook: reads the recipe's two names and stamps the environment."""
import os
if os.environ.get("CHROMBPNET_DET_SUBPROCESS") == "1":
    os.environ["CHROMBPNET_DET_SUBPROCESS_APPLIED"] = "stub seed=" + os.environ.get("CHROMBPNET_DET_SEED", "unset")
'''

README_STUB = "# stub kit\n\nDocumented form (no kit env vars): `python tf/pred_bw_fast.py -cm <model.h5> -r <regions.bed> -g <genome.fa> -c <chrom.sizes> -op <prefix>`\n"


def _write(path, text, exe=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    if exe:
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def tree_of(kit):
    """The stand-in tree root of a kit at <tree>/opt/kit (what CHROMBPNET_OPT_HOME names)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(kit)))


def make_tree(root, **kw):
    """A stand-in port tree at `root`: the stub carried kit at <root>/opt/kit and the stub fast-mode kit at <root>/opt/kit_ho; returns
    the carried kit's path (the fast mode runs <root>/opt/kit_ho, stack.kit_home)."""
    kit = make_kit(os.path.join(root, "opt", "kit"), **kw)
    make_kit_ho(root, version=kw.get("version", "0.0.0-stub"))
    return kit


def fast_kit(kit):
    """The fast mode's kit directory of the stand-in tree of `kit` (<tree>/opt/kit_ho)."""
    return os.path.join(tree_of(kit), "opt", "kit_ho")


def make_kit_ho(tree, version="0.0.0-stub"):
    """A stand-in opt/kit_ho beside the stub carried kit of `tree` (the fast mode's kit directory — the one tf/ tree: the recorder script,
    README, fastdefault, the seed hook and two tf/ CPU tests); returns its path."""
    root = os.path.join(tree, "opt", "kit_ho")
    _write(os.path.join(root, "README.md"), README_STUB)
    _write(os.path.join(root, "tf", "pred_bw_fast.py"), PRED_BW_FAST_STUB)
    _write(os.path.join(root, "tf", "chrombpnet_fastkit", "__init__.py"), '"""stub kit_ho package"""\n__version__ = "%s"\n' % version)
    _write(os.path.join(root, "tf", "chrombpnet_fastkit", "fastdefault.py"), FASTDEFAULT_STUB)
    _write(os.path.join(root, "tf", "det_subprocess", "sitecustomize.py"), SITECUSTOMIZE_STUB)
    _write(os.path.join(root, "tf", "test_stub_cpu.py"), "import sys\nprint('stub cpu test')\nsys.exit(0)\n")
    _write(os.path.join(root, "tf", "test_stub2_cpu.py"), "import os, sys\nsys.exit(int(os.environ.get('STUB_TF_TEST_RC', '0')))\n")
    return root


def make_kit(root, with_cache_for=(), version="0.0.0-stub"):
    """A stand-in CARRIED kit at `root` (normally <tree>/opt/kit: make_tree) — the torch side (a stub chrombpnet_k1 dir with one torch/ CPU
    test) and the driver caches; returns root. `with_cache_for`: classes for which cache/nv_compute_cache_<class>.tar exists."""
    _write(os.path.join(root, "README.md"), README_STUB)
    _write(os.path.join(root, "torch", "chrombpnet_k1", "__init__.py"), '"""stub carried torch-side package"""\n')
    _write(os.path.join(root, "torch", "test_stub_cpu.py"), "import os, sys\nt = os.environ.get('STUB_TORCH_TREE_EXPECTED')\n"
           "sys.exit(int(os.environ.get('STUB_TORCH_TEST_RC', '0')) if not t else int(os.environ.get('PYTHONPATH', '').split(os.pathsep)[0] != t))\n")
    for cls in with_cache_for:
        _write(os.path.join(root, "cache", "nv_compute_cache_%s.tar" % cls), "stub tar\n")
    return root


def make_run_tree(root, pins_rc_env="STUB_PINS_RC"):
    """A stand-in port tree for run.sh: run.sh and configs/ copied from the tree, a stub stock/check_pins.py whose rc comes from the environment."""
    import shutil
    os.makedirs(os.path.join(root, "stock"), exist_ok=True)
    shutil.copy(os.path.join(TREE_DIR, "run.sh"), os.path.join(root, "run.sh"))
    shutil.copytree(os.path.join(TREE_DIR, "configs"), os.path.join(root, "configs"))
    _write(os.path.join(root, "stock", "check_pins.py"), "import os, sys\nsys.exit(int(os.environ.get('%s', '0')))\n" % pins_rc_env)
    return root


STOCK_MAIN_STUB = '''"""stub stock CHROMBPNET.main: records sys.argv, the environment and the loaded package modules beside the -op prefix."""
import json, os, sys
def main():
    argv = sys.argv[1:]
    if "-op" in argv:
        op = argv[argv.index("-op") + 1]
        os.makedirs(os.path.dirname(os.path.abspath(op)) or ".", exist_ok=True)
        with open(op + "_stock_call.json", "w") as fh:
            json.dump({"argv": argv, "env": {k: v for k, v in os.environ.items() if k.startswith(("CHROMBPNET", "K1_", "TF_", "CUBLAS_", "CUDA_", "MODEL_OPT", "PYTHONPATH", "HOME_X"))}, "sys_path": sys.path, "modules": sorted(m for m in sys.modules if m.startswith("chrombpnet_opt"))}, fh)
    print("[stub stock main] ran", flush=True)
    return int(os.environ.get("STUB_RC", "0"))
'''


def make_stock_package(site_dir):
    """A stub top-level `chrombpnet` package (empty __init__, as the stock's; a CHROMBPNET.main that records its state) for the finder
    tests and the stock child; returns site_dir."""
    _write(os.path.join(site_dir, "chrombpnet", "__init__.py"), "")
    _write(os.path.join(site_dir, "chrombpnet", "CHROMBPNET.py"), STOCK_MAIN_STUB)
    return site_dir


def real_kit(dirname="kit"):
    """The real carried kit (opt/kit: its torch side torch/chrombpnet_k1) when present in this tree, else None; dirname="kit_ho" for the fast
    kit (opt/kit_ho: its entry script tf/pred_bw_fast.py)."""
    cand = os.path.join(TREE_DIR, "opt", dirname)
    mark = os.path.join(cand, "torch", "chrombpnet_k1") if dirname == "kit" else os.path.join(cand, "tf", "pred_bw_fast.py")
    return cand if os.path.exists(mark) else None


def base_env(kit, extra=None, drop_prefixes=("CHROMBPNET_", "K1_", "TF_", "STUB_", "MODEL_OPT", "CUBLAS_")):
    """A subprocess environment for the package: PYTHONPATH = opt (the package importable), the stand-in tree of `kit` (<tree>/opt/kit)
    named by CHROMBPNET_OPT_HOME, no inherited switches."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop_prefixes) and k != "PYTHONPATH"}
    env["PYTHONPATH"] = OPT_DIR
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CHROMBPNET_OPT_HOME"] = tree_of(kit)
    env.update(extra or {})
    return env
