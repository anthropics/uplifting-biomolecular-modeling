"""The stock caller — mode ``off``: the pinned stock ``borzoi_sad.py`` in a clean subprocess, proven before it runs.

``sad --mode off`` execs this module (``python -s -m borzoi_opt.stock_sad --proof-json <private dir>/stock_env_proof.json --stock-entry
<path> --stock-sha256 <pin> --kit-dirs <dirs> -- <stock arguments>``) with every ``KIT_*`` name and ``BORZOI_OPT`` stripped from the
environment and every PYTHONPATH entry inside a kit directory removed (cli.stock_command). Before anything of the stock is imported the
process proves what it is (``env_proof``): no forbidden environment name, no kit module loaded (``kitlib`` by name, or any module whose file
is under a kit directory), no kit directory on sys.path, the package's autoload hook not armed, tensorflow not yet imported, and the
entry's sha256 equal to the pin; the proof is said on one ``ENV-CLEAN`` line and written as JSON where ``--proof-json`` points (the ``sad``
process reads it for its EXIT line and removes it). A process that fails its proof exits 3 without running anything. Then the stock script runs as ``__main__`` (``runpy.run_path``: the same as
``python borzoi_sad.py <arguments>``) with the arguments unchanged. This module imports the standard library and, through the script,
the stock — nothing of the kit and nothing else of this package.
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from typing import Iterable, List, Optional

PREFIX = "[borzoi-opt stock]"
EXIT_NOT_STOCK = 3
KIT_MODULE_NAMES = ("kitlib",)                                    # the kit's runtime package (v17/kitlib)
FORBIDDEN_PREFIXES = ("KIT_",)                                    # every switch the kit reads
FORBIDDEN_NAMES = ("BORZOI_OPT",)                                 # the package's own switch


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _forbidden(environ) -> List[str]:
    return sorted(k for k in environ if k in FORBIDDEN_NAMES or any(k.startswith(p) for p in FORBIDDEN_PREFIXES))


def _under(path: str, roots: List[str]) -> bool:
    p = os.path.realpath(path)
    return any(p == r or p.startswith(r + os.sep) for r in roots)


def kit_modules_loaded(kit_dirs: Iterable[str], modules=None) -> List[str]:
    modules = sys.modules if modules is None else modules
    roots = [os.path.realpath(d) for d in kit_dirs if d]
    return sorted(m for m, mod in list(modules.items())
                  if m.split(".")[0] in KIT_MODULE_NAMES or (roots and _under(getattr(mod, "__file__", None) or "", roots)))


def env_proof(kit_dirs: Iterable[str], stock_entry: str, stock_sha256: Optional[str], environ=None, modules=None, path=None, meta_path=None) -> dict:
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    path = sys.path if path is None else path
    meta_path = sys.meta_path if meta_path is None else meta_path
    roots = [os.path.realpath(d) for d in kit_dirs if d]
    hits = _forbidden(environ)
    kit_mods = kit_modules_loaded(kit_dirs, modules)
    kit_dirs_on_path = sorted(p for p in path if p and roots and _under(p, roots))
    armed = [type(f).__name__ for f in meta_path if type(f).__module__ == "borzoi_opt._autoload" and getattr(f, "armed", True)]
    tf_loaded = "tensorflow" in modules
    entry_ok = os.path.isfile(stock_entry)
    got = _sha256(stock_entry) if entry_ok else None
    pinned = bool(stock_sha256) and got == stock_sha256
    proof = {"forbidden_present": hits, "kit_modules_loaded": kit_mods, "kit_dirs_on_path": kit_dirs_on_path, "autoload_armed": armed,
             "tensorflow_loaded_before_proof": tf_loaded, "stock_entry": stock_entry, "stock_entry_sha256": got, "stock_pin_sha256": stock_sha256,
             "stock_entry_pinned": pinned, "no_user_site": bool(sys.flags.no_user_site), "python": sys.executable,
             "ok": not (hits or kit_mods or kit_dirs_on_path or armed or tf_loaded) and entry_ok and pinned}
    return proof


def _split(argv: List[str]):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def run(argv: Optional[List[str]] = None) -> int:
    import argparse
    own, stock = _split(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser(prog="python -s -m borzoi_opt.stock_sad", description="Borzoi stock call: the stock borzoi_sad.py, proven clean first")
    ap.add_argument("--proof-json", required=True)
    ap.add_argument("--stock-entry", required=True)
    ap.add_argument("--stock-sha256", default="")
    ap.add_argument("--kit-dirs", default="", help=f"{os.pathsep}-separated kit directories: none may hold a loaded module or a sys.path entry")
    a = ap.parse_args(own)
    kit_dirs = [d for d in a.kit_dirs.split(os.pathsep) if d]
    proof = env_proof(kit_dirs, a.stock_entry, a.stock_sha256 or None)
    proof["stock_argv"] = list(stock)
    os.makedirs(os.path.dirname(os.path.abspath(a.proof_json)), exist_ok=True)

    def write_proof() -> None:
        with open(a.proof_json, "w", encoding="utf-8") as fh:
            json.dump(proof, fh, indent=1, default=str)
            fh.write("\n")
    write_proof()
    if not proof["ok"]:
        print(f"{PREFIX} NOT STOCK: forbidden env {proof['forbidden_present']}, kit modules {proof['kit_modules_loaded']}, kit dirs "
              f"{proof['kit_dirs_on_path']}, autoload {proof['autoload_armed']}, tensorflow loaded {proof['tensorflow_loaded_before_proof']}, "
              f"entry pinned {proof['stock_entry_pinned']} ({proof['stock_entry_sha256']} vs {proof['stock_pin_sha256']})", file=sys.stderr, flush=True)
        return EXIT_NOT_STOCK
    print(f"{PREFIX} ENV-CLEAN ok: entry={a.stock_entry} sha256={str(proof['stock_entry_sha256'])[:12]}… kit_modules=none kit_dirs=none "
          f"autoload=none no_user_site={proof['no_user_site']}", file=sys.stderr, flush=True)
    rc = 0
    try:
        sys.argv = [a.stock_entry] + list(stock)
        runpy.run_path(a.stock_entry, run_name="__main__")
    except SystemExit as e:
        code = e.code
        rc = 0 if code is None else (code if isinstance(code, int) else 1)
    finally:
        proof["kit_modules_loaded_after"] = kit_modules_loaded(kit_dirs)
        write_proof()
        if proof["kit_modules_loaded_after"]:
            print(f"{PREFIX} KIT MODULES LOADED during the stock call: {proof['kit_modules_loaded_after']}", file=sys.stderr, flush=True)
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
