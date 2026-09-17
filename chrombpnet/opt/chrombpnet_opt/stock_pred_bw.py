"""The stock child — mode ``off``: the upstream console script's own entry, nothing from the kit on the path, proved INSIDE the child.

``pred_bw --mode off`` (cli.py) runs ``python -s <this file> --proof <outdir>/stock_env_proof.json --kit <opt/kit> --tree <port>
-- <the stock pred_bw arguments>`` — by file path, so no module of this package is loaded in the child — in the stripped environment (stack.stock_env: all kit and package variables removed,
PYTHONPATH entries under opt/kit and under the tree removed — the kit's deterministic seed-hook directory the one exception under
--det). Before the stock's ``main`` this process records, from its own state, {argv, executable, the environment names matching the
must-be-absent prefixes/names (must be empty), the PYTHONPATH entries, the sys.path entries under opt/kit (must be none; the seed-hook
directory under --det is the exception and is reported separately), the sys.path entries under the tree (this file's own directory,
python's sys.path[0] for a script, is reported separately), cwd, pid, whether the package's autoload finder is armed (must not be),
the package modules loaded (must be none)} to the proof file, then calls the stock's ``chrombpnet.CHROMBPNET:main`` (stock setup.py:27
— the function the ``chrombpnet`` console script calls) with ``sys.argv = ["chrombpnet", "pred_bw", *args]``. It imports nothing from
this package (stdlib only) and prints nothing of its own before the stock's output; the mode line is the launching process's. Exit = the
stock's; 3 when the stock package is not importable or the environment is not clean.
"""
import argparse
import json
import os
import sys

PROOF_FILENAME = "stock_env_proof.json"
FORBIDDEN_PREFIXES = ("CHROMBPNET_OPT",)                                                     # the package's own namespace: the one legitimate prefix rule
# the kit's own switch names — a copy of registry.KIT_SWITCH_NAMES (this file imports nothing from the package; a test keeps them in step)
FORBIDDEN_NAMES = (
    'CHROMBPNET_DET_SEED',
    'CHROMBPNET_DET_SUBPROCESS',
    'CHROMBPNET_DET_SUBPROCESS_APPLIED',
    'CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE',
    'CHROMBPNET_FASTKIT_RECORD',
    'CHROMBPNET_FASTKIT_TAIL',
    'CHROMBPNET_FASTKIT_TAIL_SOURCE',
    'CHROMBPNET_JIT_CACHE_TAR',
    'CHROMBPNET_K1_DIR',
    'CHROMBPNET_K1_STREAMS',
    'K1_BLOCK_K',
    'K1_BLOCK_M',
    'K1_BLOCK_N',
    'K1_EXIT_TEARDOWN',
    'K1_NUM_STAGES',
    'K1_NUM_WARPS',
    'K1_PREIMPORT',
)
DET_NAMES = ("CHROMBPNET_DET_SUBPROCESS", "CHROMBPNET_DET_SEED", "CHROMBPNET_DET_SUBPROCESS_APPLIED")   # the recipe's names: allowed under --det
DET_HOOK_RELDIR = os.path.join("tf", "det_subprocess")
STOCK_ENTRY = ("chrombpnet.CHROMBPNET", "main")                                            # stock setup.py:27


INERT_FINDER_MODULES = ("chrombpnet_opt", "chrombpnet_opt._autoload")   # what the installed chrombpnet_opt_autoload.pth imports at interpreter start: the finder is armed only by CHROMBPNET_OPT


def _under(path, root):
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)
    except ValueError:
        return False


def env_proof(kit, tree, det_on, environ=None, path=None, modules=None, argv=None):
    environ = os.environ if environ is None else environ
    path = sys.path if path is None else path
    modules = sys.modules if modules is None else modules
    hook = os.path.realpath(os.path.join(kit, DET_HOOK_RELDIR)) if kit else None
    forbidden = sorted(k for k in environ if (k.startswith(FORBIDDEN_PREFIXES) or k in FORBIDDEN_NAMES) and not (det_on and k in DET_NAMES))
    under_kit = [p for p in path if kit and p and _under(p, kit) and os.path.realpath(p) != hook]
    hook_on_path = bool(hook) and any(p and os.path.realpath(p) == hook for p in path)
    here = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.dirname(here)   # the package's OWN import root (<tree>/opt): the one directory that makes chrombpnet_opt importable is on the path of a
    # process the package itself launched by construction — a launcher that binds the package in every interpreter (a `<opt dir>` line in a site .pth)
    # puts exactly this entry on the child's sys.path; it is admitted. Kit dirs, every OTHER dir under the tree, forbidden variables, an armed
    # finder and loaded kit / package modules stay contamination.
    under_tree = [p for p in path if tree and p and _under(p, tree) and not (kit and _under(p, kit)) and os.path.realpath(p) not in (here, root)]
    launcher_dir_on_path = any(p and os.path.realpath(p) == here for p in path)
    finder_armed = any(type(f).__name__ == "Finder" and getattr(f, "armed", False) for f in sys.meta_path)
    proof = {"argv": list(sys.argv if argv is None else argv), "executable": sys.executable, "pid": os.getpid(), "cwd": os.getcwd(),
             "forbidden_prefixes": list(FORBIDDEN_PREFIXES), "forbidden_names": list(FORBIDDEN_NAMES), "forbidden_present": forbidden,
             "pythonpath": [p for p in (environ.get("PYTHONPATH") or "").split(os.pathsep) if p],
             "sys_path_under_kit": under_kit, "sys_path_under_tree": under_tree, "launcher_dir_on_path": launcher_dir_on_path, "package_root_on_path": any(p and os.path.realpath(p) == root for p in path),
             "det": bool(det_on), "det_hook_on_path": hook_on_path,
             "autoload_armed": finder_armed, "kit_modules_loaded": sorted(m for m in modules if m.startswith(("chrombpnet_fastkit", "chrombpnet_k1"))),
             "finder_modules_present": sorted(m for m in modules if m in INERT_FINDER_MODULES),     # loaded by the installed .pth line in every interpreter; inert while unarmed
             "package_modules_loaded": sorted(m for m in modules if m.startswith("chrombpnet_opt") and m not in INERT_FINDER_MODULES), "no_user_site": bool(sys.flags.no_user_site)}
    proof["ok"] = (not forbidden and not under_kit and not under_tree and not finder_armed and not proof["kit_modules_loaded"] and not proof["package_modules_loaded"]
                   and (hook_on_path if det_on else True))
    return proof


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -s stock_pred_bw.py", allow_abbrev=False)
    ap.add_argument("--proof", required=True, help="the proof file to write (beside the outputs)")
    ap.add_argument("--kit", default=None); ap.add_argument("--tree", default=None); ap.add_argument("--det", action="store_true")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="-- <the stock pred_bw arguments>")
    a = ap.parse_args(argv)
    args = a.args[1:] if a.args and a.args[0] == "--" else a.args
    proof = env_proof(a.kit, a.tree, a.det, argv=[sys.argv[0]] + args)
    os.makedirs(os.path.dirname(os.path.abspath(a.proof)) or ".", exist_ok=True)
    with open(a.proof, "w", encoding="utf-8") as fh:
        json.dump(proof, fh, indent=1, default=str)
        fh.write("\n")
    if not proof["ok"]:
        print("[chrombpnet-opt stock] ENV NOT CLEAN: forbidden={} sys_path_under_kit={} sys_path_under_tree={} autoload_armed={}".format(
            ",".join(proof["forbidden_present"]) or "none", proof["sys_path_under_kit"], proof["sys_path_under_tree"], proof["autoload_armed"]), file=sys.stderr, flush=True)
        return 3
    try:
        import importlib
        entry = getattr(importlib.import_module(STOCK_ENTRY[0]), STOCK_ENTRY[1])
    except Exception as e:  # noqa: BLE001
        print("[chrombpnet-opt stock] the stock package is not importable ({}: {})".format(type(e).__name__, e), file=sys.stderr, flush=True)
        return 3
    sys.argv = ["chrombpnet", "pred_bw"] + list(args)                                 # the console script's argv (stock parsers.py reads sys.argv)
    rc = entry()
    return int(rc) if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
