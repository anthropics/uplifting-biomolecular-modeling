"""The stock caller: ``python -s -m genie3_opt.stock_cli --proof-json <file> --env-absent <names> --kit-dirs <dirs> --genie3-root <dir>
--pins-check <json> -- generate -c <request.yaml> --log-dir <dir>``.

Proves, in the process that becomes the stock CLI, that nothing of the kit or of this package is on the path, then ``os.execv`` into
upstream's own console script `genie3` (setup.py:14, `genie3.cli:main`) with cwd = the checkout: (1) none of the must-be-absent names
is in the environment (stock/PINS.json ``stock_environment``: the package's own switches, CUDA MPS, NVIDIA_TF32_OVERRIDE; the two
deployment paths GENIE3_ROOT / GENIE3_WEIGHTS are the allowed exceptions), (2) no module of the kit is loaded (by name: the driver's
modules; by file: anything under a kit directory), (3) no kit directory is on sys.path, (4) the autoload finder is not armed, (5) torch
and genie3 are not imported here — the proof is written before anything of upstream is imported, and (6) the checkout's pins REPORT
(``--pins-check``: stock/check_pins.py's report as design.py computed it — files checked, files differing, git HEAD, tracked files modified —
copied into the proof so the stock log carries it; ``pinned`` true or false, never a refusal: upstream runs any checkout), (7) the shared core's clean-process proof over the same
names beside the package's own (``core_proof``: opt_core.stock_proof.env_proof — armed autoload finders, a kit ``sitecustomize``, torch before
the proof, core modules beyond the proof module; recorded under ``"core"``). A violation on either side refuses: exit 3 with the proof on stderr.
The stock arguments after ``--`` are upstream's own, verbatim (``generate -c <request> --log-dir <dir>`` plus any of upstream's generate flags the
caller gave).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterable, List, Optional

from .codes import EXIT_NOT_ACTIVE, TAG

PREFIX = f"[{TAG} stock]"
EXIT_NOT_STOCK = EXIT_NOT_ACTIVE                                         # nothing ran (codes.py, a leaf: no driver-side module loads here)
KIT_MODULE_PREFIXES = ("g3fast", "genie3_opt.design", "genie3_opt.stack")   # the kit's module names (the driver and its patch module) and the package's driver side
STOCK_EXE = "genie3"                                                     # upstream's console script (setup.py:14)


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
    """Loaded modules that are the kit's: by name (KIT_MODULE_PREFIXES) or by file (under a kit directory)."""
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
    m = modules.get("genie3_opt._autoload")
    return bool(getattr(m, "FINDER", None)) if m is not None else False


def env_proof(env_absent: List[str], kit_dirs: List[str], environ=None, modules=None, path=None, allowed: Optional[List[str]] = None) -> dict:
    """The proof record: every check with its finding; ``ok`` when all pass."""
    environ = os.environ if environ is None else environ
    mods = sys.modules if modules is None else modules
    forb = [n for n in _forbidden(environ, env_absent) if n not in (allowed or [])]
    rec = {"forbidden_env_present": forb, "kit_modules_loaded": kit_modules_loaded(kit_dirs, modules), "kit_dirs_on_sys_path": kit_dirs_on_path(kit_dirs, path),
           "autoload_armed": autoload_armed(modules), "torch_imported": "torch" in mods,
           "genie3_imported": any(m == "genie3" or m.startswith("genie3.") for m in mods),
           "python": sys.executable, "site_disabled": bool(sys.flags.no_site) or bool(sys.flags.no_user_site), "pid": os.getpid()}
    rec["core"] = core_proof(list(env_absent), kit_dirs, environ=environ, modules=mods, path=path, allowed=allowed)
    rec["ok"] = not (rec["forbidden_env_present"] or rec["kit_modules_loaded"] or rec["kit_dirs_on_sys_path"] or rec["autoload_armed"]
                     or rec["torch_imported"] or rec["genie3_imported"]) and rec["core"]["ok"]
    return rec


def core_proof(env_absent: List[str], kit_dirs: List[str], environ=None, modules=None, path=None, allowed: Optional[List[str]] = None) -> dict:
    """The shared core's clean-process proof (opt_core.stock_proof.env_proof: the must-be-absent names, kit modules by prefix or file, kit
    directories on sys.path, armed autoload finders, a kit ``sitecustomize``, torch before the proof, core modules beyond the proof module) —
    ``{"ok", "violations", "allowed_excluded", "proof"}``. The core matches every must-be-absent entry as a prefix and has no exception list:
    the allowed exceptions (stock/PINS.json ``stock_environment.allowed_exceptions``: the two deployment paths) are excluded from its view and
    named in the record. A core that does not resolve in this process is itself a violation (``unavailable``): the activation that started
    this process was gated on the same core (every entry's statement one: _core.core_gate)."""
    environ = os.environ if environ is None else environ
    excluded = sorted(k for k in (allowed or []) if k in environ)
    environ = {k: v for k, v in environ.items() if k not in (allowed or [])}
    try:
        from . import _core
        _core.ensure_importable()
        from opt_core import stock_proof as _sp
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "unavailable": repr(e), "violations": f"opt_core does not resolve: {e!r}", "allowed_excluded": excluded, "proof": None}
    p = _sp.env_proof(env_absent=list(env_absent), kit_dirs=[os.path.normpath(d) for d in kit_dirs], module_prefixes=list(KIT_MODULE_PREFIXES),
                      environ=environ, modules=modules, path=None if path is None else list(path))
    return {"ok": bool(p["ok"]), "violations": None if p["ok"] else _sp.violations_sentence(p), "allowed_excluded": excluded, "proof": p}


def stock_exe(python: Optional[str] = None) -> Optional[str]:
    """The `genie3` console script beside the interpreter, else on PATH (standard library only: no package import here)."""
    import shutil
    cand = os.path.join(os.path.dirname(os.path.abspath(python or sys.executable)), STOCK_EXE)
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    return shutil.which(STOCK_EXE)


def stock_argv(exe: str, args: List[str]) -> List[str]:
    """``<genie3> generate -c <request.yaml> --log-dir <dir>`` (the arguments after ``--``, verbatim)."""
    return [exe] + list(args)


def parse(argv: List[str]) -> dict:
    a = {"proof_json": None, "env_absent": [], "kit_dirs": [], "genie3_root": None, "allowed": [], "pins_check": None, "args": []}
    i = 0
    while i < len(argv):
        t = argv[i]
        if t == "--":
            a["args"] = argv[i + 1:]
            break
        if t == "--proof-json":
            a["proof_json"] = argv[i + 1]; i += 2; continue
        if t == "--env-absent":
            a["env_absent"] = [x for x in argv[i + 1].split(",") if x]; i += 2; continue
        if t == "--kit-dirs":
            a["kit_dirs"] = [x for x in argv[i + 1].split(os.pathsep) if x]; i += 2; continue
        if t == "--allowed":
            a["allowed"] = [x for x in argv[i + 1].split(",") if x]; i += 2; continue
        if t == "--genie3-root":
            a["genie3_root"] = argv[i + 1]; i += 2; continue
        if t == "--pins-check":
            a["pins_check"] = argv[i + 1]; i += 2; continue
        raise SystemExit(f"{PREFIX} unknown argument {t!r}")
    return a


def main(argv: Optional[List[str]] = None) -> int:
    a = parse(list(sys.argv[1:] if argv is None else argv))
    root = a["genie3_root"] or os.environ.get("GENIE3_ROOT")
    proof = env_proof(a["env_absent"], a["kit_dirs"], allowed=a["allowed"])
    proof["genie3_root"] = root
    exe = stock_exe()
    proof["argv"] = stock_argv(exe, a["args"]) if exe else None
    if a["pins_check"] and os.path.isfile(a["pins_check"]):
        try:
            pc = json.load(open(a["pins_check"], encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            pc = {"error": repr(e)}
        proof["pins_check"] = pc                                             # the pins REPORT (stock/check_pins.py as design.py computed it): recorded, never gated — upstream runs any checkout
        proof["pinned"] = not pc.get("bad")
    if not root or not os.path.isdir(root):
        proof["ok"] = False
        proof["error"] = f"no checkout at GENIE3_ROOT={root!r}"
    elif not exe:
        proof["ok"] = False
        proof["error"] = f"no `{STOCK_EXE}` console script beside {sys.executable} or on PATH"
    if a["proof_json"]:
        os.makedirs(os.path.dirname(os.path.abspath(a["proof_json"])), exist_ok=True)
        with open(a["proof_json"], "w", encoding="utf-8") as fh:
            json.dump(proof, fh, indent=1)
    if not proof["ok"] and not proof.get("error") and not (proof.get("core") or {}).get("ok", True):
        proof["error"] = f"shared core proof: {proof['core'].get('violations')}"
    if not proof["ok"]:
        sys.stderr.write(f"{PREFIX} NOT STOCK: {proof.get('error') or json.dumps({k: v for k, v in proof.items() if k not in ('ok', 'argv', 'pins_check')})}\n")
        return EXIT_NOT_STOCK
    pc = proof.get("pins_check") or {}
    co = ((pc.get("detail") or {}).get("checkout") or {})
    sys.stderr.write(f"{PREFIX} proof ok pid={os.getpid()} checkout={root} pinned={co.get('pinned')} files_checked={co.get('files_checked')} "
                     f"git_head={co.get('git_head')} modified={len(co.get('git_modified') or [])}; cwd={root}; exec {' '.join(proof['argv'][:3])} ...\n")
    sys.stderr.flush()
    os.chdir(root)
    os.execv(proof["argv"][0], proof["argv"])
    return 0                                                              # not reached


if __name__ == "__main__":
    sys.exit(main())
