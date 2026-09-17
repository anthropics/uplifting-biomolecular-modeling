"""The stock caller — mode ``off``: upstream's command line in a clean subprocess, proven before it runs.

``design --mode off`` runs, per case, ``python -s -m rfdiffusion1_opt.stock_cli --proof-json <case dir>/stock_env_proof.json
--env-absent <names> --kit-dirs <dirs> --rfd-root <RFD_ROOT> -- <hydra overrides>`` with every environment name under
stock/PINS.json ``stock_environment.must_be_absent_prefixes`` stripped (its ``allowed_exceptions`` kept: RFD_ROOT) and every
PYTHONPATH entry inside a kit directory removed
(design.stock_command). Before anything of RFdiffusion is imported the process proves what it is (``env_proof``): no forbidden
environment name, no kit module loaded (by name or by file), no kit directory on sys.path, the package's autoload finder not
armed, torch not yet imported; the proof is written as JSON and read into ``opt_manifest.json`` by the ``design`` process. A
process that fails its proof exits 3 without running anything. Then it replaces itself (``os.execv``) with the stock command
line — ``python -s $RFD_ROOT/scripts/run_inference.py <overrides>`` (stock/PINS.json "entry_point") — so what runs is upstream's
own script, unchanged, in the proven environment. This module imports the standard library, and the shared core's proof (opt_core.stock_proof) inside
``core_proof`` — the shared core's clean-process contract for a stock arm.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterable, List, Optional

PREFIX = "[rfdiffusion1-opt stock]"
EXIT_NOT_STOCK = 3                                                       # the package's EXIT_NOT_ACTIVE: nothing ran
KIT_MODULE_PREFIXES = ("rfd_", "rfdiffusion1_opt.design", "rfdiffusion1_opt.stack")   # the kits' module names (the lever modules) and the package's driver side
STOCK_SCRIPT = ("scripts", "run_inference.py")                           # upstream's entry point under RFD_ROOT


def _forbidden(environ, names: Iterable[str]) -> List[str]:
    """Names of `environ` matching `names`: an entry ending in '_' is a prefix, any other an exact name."""
    out = []
    for k in environ:
        for n in names:
            if (n.endswith("_") and k.startswith(n)) or k == n:
                out.append(k)
                break
    return sorted(set(out))


def _under(path: str, roots: List[str]) -> bool:
    p = os.path.realpath(path)
    return any(p == r or p.startswith(r + os.sep) for r in roots)


def kit_modules_loaded(kit_dirs: Iterable[str], modules=None) -> List[str]:
    """Loaded modules that are the kits': by name (KIT_MODULE_PREFIXES) or by file (under a kit directory)."""
    modules = sys.modules if modules is None else modules
    roots = [os.path.realpath(d) for d in kit_dirs if d]
    out = []
    for name, mod in list(modules.items()):
        if name.startswith(KIT_MODULE_PREFIXES):
            out.append(name)
            continue
        f = getattr(mod, "__file__", None)
        if f and roots and _under(f, roots):
            out.append(name)
    return sorted(set(out))


def kit_dirs_on_path(kit_dirs: Iterable[str], path=None) -> List[str]:
    path = sys.path if path is None else path
    roots = [os.path.realpath(d) for d in kit_dirs if d]
    return sorted(p for p in path if p and roots and _under(p, roots))


def autoload_armed(modules=None) -> bool:
    modules = sys.modules if modules is None else modules
    m = modules.get("rfdiffusion1_opt._autoload")
    return bool(getattr(m, "FINDER", None)) if m is not None else False


def env_proof(env_absent: List[str], kit_dirs: List[str], environ=None, modules=None, path=None, allowed: Optional[List[str]] = None) -> dict:
    """The proof record: every check with its finding; ``ok`` when all pass."""
    environ = os.environ if environ is None else environ
    forb = [n for n in _forbidden(environ, env_absent) if n not in (allowed or [])]
    rec = {"forbidden_env_present": forb, "kit_modules_loaded": kit_modules_loaded(kit_dirs, modules), "kit_dirs_on_sys_path": kit_dirs_on_path(kit_dirs, path),
           "autoload_armed": autoload_armed(modules), "torch_imported": "torch" in (sys.modules if modules is None else modules),
           "rfdiffusion_imported": any(m == "rfdiffusion" or m.startswith("rfdiffusion.") for m in (sys.modules if modules is None else modules)),
           "python": sys.executable, "site_disabled": bool(sys.flags.no_site) or bool(sys.flags.no_user_site), "pid": os.getpid()}
    rec["core"] = core_proof(env_absent, kit_dirs, environ, modules, path, allowed)
    rec["ok"] = not (rec["forbidden_env_present"] or rec["kit_modules_loaded"] or rec["kit_dirs_on_sys_path"] or rec["autoload_armed"]
                     or rec["torch_imported"] or rec["rfdiffusion_imported"]) and bool(rec["core"].get("ok", True))
    return rec


def core_proof(env_absent: List[str], kit_dirs: List[str], environ=None, modules=None, path=None, allowed: Optional[List[str]] = None) -> dict:
    """The shared core's clean-process proof (``opt_core.stock_proof.env_proof``: the same names, plus armed autoload finders, a kit sitecustomize,
    torch before the proof, core modules beyond the proof itself) — ``{"ok", "violations", "proof"}``; ``{"ok": True, "unavailable": …}`` when the
    core does not resolve in this process (the proof above stands alone; the fact is recorded)."""
    try:
        from opt_core import stock_proof as _sp
    except Exception as e:  # noqa: BLE001
        return {"ok": True, "unavailable": repr(e), "violations": None, "proof": None}
    environ = os.environ if environ is None else environ
    env = {k: v for k, v in environ.items() if k not in (allowed or [])}          # the allowed exceptions (RFD_ROOT) are locations, not switches
    prf = _sp.env_proof(env_absent=list(env_absent), kit_dirs=[os.path.normpath(d) for d in kit_dirs], module_prefixes=[],
                        environ=env, modules=modules, path=list(path) if path is not None else None)   # the core reads every entry as a prefix: an exact name is its own prefix
    return {"ok": bool(prf["ok"]), "violations": None if prf["ok"] else _sp.violations_sentence(prf), "proof": prf}


def stock_argv(rfd_root: str, overrides: List[str], python: Optional[str] = None) -> List[str]:
    """``python -s <RFD_ROOT>/scripts/run_inference.py <overrides>``."""
    return [python or sys.executable, "-s", os.path.join(rfd_root, *STOCK_SCRIPT)] + list(overrides)


def parse(argv: List[str]) -> dict:
    a = {"proof_json": None, "env_absent": [], "kit_dirs": [], "rfd_root": None, "allowed": [], "overrides": []}
    i = 0
    while i < len(argv):
        t = argv[i]
        if t == "--":
            a["overrides"] = argv[i + 1:]
            break
        if t == "--proof-json":
            a["proof_json"] = argv[i + 1]; i += 2; continue
        if t == "--env-absent":
            a["env_absent"] = [x for x in argv[i + 1].split(",") if x]; i += 2; continue
        if t == "--kit-dirs":
            a["kit_dirs"] = [x for x in argv[i + 1].split(os.pathsep) if x]; i += 2; continue
        if t == "--allowed":
            a["allowed"] = [x for x in argv[i + 1].split(",") if x]; i += 2; continue
        if t == "--rfd-root":
            a["rfd_root"] = argv[i + 1]; i += 2; continue
        raise SystemExit(f"{PREFIX} unknown argument {t!r}")
    return a


def main(argv: Optional[List[str]] = None) -> int:
    a = parse(list(sys.argv[1:] if argv is None else argv))
    rfd = a["rfd_root"] or os.environ.get("RFD_ROOT")
    proof = env_proof(a["env_absent"], a["kit_dirs"], allowed=a["allowed"])
    proof["rfd_root"] = rfd
    proof["argv"] = stock_argv(rfd or "", a["overrides"]) if rfd else None
    if not rfd or not os.path.isfile(os.path.join(rfd, *STOCK_SCRIPT)):
        proof["ok"] = False
        proof["error"] = f"no stock entry point at RFD_ROOT={rfd!r}/{'/'.join(STOCK_SCRIPT)}"
    if a["proof_json"]:
        os.makedirs(os.path.dirname(os.path.abspath(a["proof_json"])), exist_ok=True)
        with open(a["proof_json"], "w", encoding="utf-8") as fh:
            json.dump(proof, fh, indent=1)
    if not proof["ok"]:
        sys.stderr.write(f"{PREFIX} NOT STOCK: {proof.get('error') or json.dumps({k: v for k, v in proof.items() if k not in ('ok', 'argv')})}\n")
        return EXIT_NOT_STOCK
    sys.stderr.write(f"{PREFIX} proof ok pid={os.getpid()}; exec {' '.join(proof['argv'][:3])} ...\n")
    sys.stderr.flush()
    os.execv(proof["argv"][0], proof["argv"])
    return 0                                                              # not reached


if __name__ == "__main__":
    sys.exit(main())
