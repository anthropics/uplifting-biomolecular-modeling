"""The stock caller — mode ``off``: the stock ``boltz predict`` command line in a clean subprocess that first proves it is stock.

``pred --mode off`` execs this module (``python -s -m boltz2_opt.stock_pred --env-absent <prefixes> --kit-modules <prefixes> --kit-dirs
<dirs> --pins <stock/PINS.json> -- predict <stock arguments>``) with every environment name under stock/PINS.json
``stock_environment.must_be_absent_prefixes`` stripped and every PYTHONPATH entry inside a kit directory removed (cli.stock_command).
Before anything of boltz is imported the process checks what it is (``env_proof``): no forbidden environment name, no kit module loaded
(by name or by file), no kit directory on sys.path, the package's autoload finder not armed, the process's sitecustomize (if any) not a
kit's, torch not yet imported, and the installed boltz tree at the pin (stock/PINS.json installed_tree); the verdict is the printed line
(``[boltz2-opt stock] proven stock: …`` / ``[boltz2-opt stock] NOT STOCK: <problems>``) and a process that fails it exits 3 without
running anything. Then the stock click group (``boltz.main:cli``, the ``boltz`` console script's entry point) runs with the arguments
unchanged. Imports: the standard library and ``opt_core.stock_proof`` (the one core module a stock process may hold) at import; at run
time the stock package, ``boltz2_opt.phase`` (the per-item PHASE line on ``Boltz2.predict_step``: synchronize + a clock read at each
boundary, numerics unchanged, with its PEAK line read through ``opt_core.mem.allocator``) and ``boltz2_opt.kernels`` (the KERNELS census
and REQUIRE guard: standard library at import; the cuEquivariance modules and one RNG-neutral probe call at the first ``predict_step``)
— nothing of the kit and nothing else of this package.
"""
from __future__ import annotations

import glob
import hashlib
import importlib.metadata as md
import json
import os
import sys
from typing import List, Optional

PREFIX = "[boltz2-opt stock]"
EXIT_NOT_STOCK = 3                                                      # the package's EXIT_NOT_ACTIVE: nothing ran
STOCK_ENTRY = ("boltz.main", "cli")                                     # the `boltz` console script's entry point (pyproject.toml [project.scripts])


OWN_MODULES = ("boltz2_opt", "boltz2_opt.stock_pred", "boltz2_opt._autoload", "boltz2_opt.kernels")   # this module, its package and the package's .pth hook (inert with BOLTZ2_OPT unset — the armed-finder check covers it);
                                                                                    # boltz2_opt.phase and boltz2_opt.kernels are imported AFTER the proof (standard library only at import)


def _tree_digest(root: str):   # the pin's digest definition: sha256 over the installed tree's .py bytes in sorted path order (stock/PINS.json upstream.installed_tree), not a file digest
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)); h = hashlib.sha256()
    for f in files:
        with open(f, "rb") as fh:
            h.update(fh.read())
    return len(files), h.hexdigest()


def env_proof(environ, env_absent: List[str], kit_modules: List[str], kit_dirs: List[str], pins: Optional[dict] = None, modules=None, path=None) -> dict:
    """What this process is, before boltz is imported: the core's clean-process proof (``opt_core.stock_proof.env_proof``: forbidden
    environment names, kit modules by name or file, kit directories on sys.path, an armed autoload finder, a kit sitecustomize, torch
    already imported, core modules beyond the proof machinery) plus this kit's pin check of the installed boltz tree (``pins``:
    stock/PINS.json). ``ok`` is True only when every check passes; ``problems`` names what failed in the core's words."""
    from opt_core import stock_proof as core                          # the proof machinery: the one core module a stock process holds (core.CORE_ALLOWED_IN_STOCK)
    modules = sys.modules if modules is None else modules
    path = sys.path if path is None else path
    scanned = {k: v for k, v in list(modules.items()) if k not in OWN_MODULES}
    proof = core.env_proof(env_absent=list(env_absent), kit_dirs=list(kit_dirs), module_prefixes=list(kit_modules), environ=environ, modules=scanned, path=list(path))
    problems = [] if proof["ok"] else [core.violations_sentence(proof)]
    if proof.get("core_modules_loaded"):
        problems.append(f"core modules loaded beyond the proof machinery: {','.join(proof['core_modules_loaded'])}")
    boltz_ok = None; tree = None
    if pins:
        try:
            ver = md.version("boltz")
        except md.PackageNotFoundError:
            ver = None
        import importlib.util
        spec = importlib.util.find_spec("boltz")
        root = os.path.dirname(spec.origin) if spec and spec.origin else None
        want = pins["upstream"]["installed_tree"]
        if root:
            n, sha = _tree_digest(root); tree = {"root": root, "n_py_files": n, "sha256_all_py_concat": sha}
            boltz_ok = (ver == pins["boltz_version"] and n == want["n_py_files"] and sha == want["sha256_all_py_concat"])
        else:
            boltz_ok = False
        if not boltz_ok:
            problems.append(f"boltz not at the pin: version {ver}, tree {tree}")
    sc = modules.get("sitecustomize")
    proof.update({"ok": bool(proof["ok"]) and not problems, "problems": problems, "executable": sys.executable, "python": sys.version.split()[0], "cwd": os.getcwd(),
                  "environment_kept": {k: environ[k] for k in ("BOLTZ_CACHE", "TRITON_CACHE_DIR", "CUDA_VISIBLE_DEVICES") if k in environ},
                  "forbidden_prefixes": list(env_absent), "kit_dirs": list(kit_dirs), "sitecustomize": getattr(sc, "__file__", None) if sc else None,
                  "boltz_pinned": boltz_ok, "boltz_tree": tree, "modules_at_proof": len(modules)})
    return proof


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print(f"{PREFIX} usage: python -s -m boltz2_opt.stock_pred --env-absent A,B --kit-modules M,N --kit-dirs D1,D2 --pins PINS [--proof-only 1] "
              "[--kernels-route stock|default --kernels-expect SPEC --kernels-settings W --kernels-json P] -- predict ...", file=sys.stderr)
        return 2
    i = argv.index("--"); opts, stock_args = argv[:i], argv[i + 1:]
    from . import kernels                                                # the KERNELS census + REQUIRE guard: standard library at import (the proof allows it by name); torch / cuequivariance touched at the first predict_step only
    kopts, opts = kernels.parse_cli_opts(opts)
    o = {}
    it = iter(opts)
    for k in it:
        o[k] = next(it, "")
    pins = json.load(open(o["--pins"])) if o.get("--pins") else None
    proof = env_proof(os.environ, [x for x in o.get("--env-absent", "").split(",") if x], [x for x in o.get("--kit-modules", "").split(",") if x],
                      [x for x in o.get("--kit-dirs", "").split(",") if x], pins)
    proof["stock_args"] = stock_args
    if not proof["ok"]:
        print(f"{PREFIX} NOT STOCK: " + "; ".join(proof["problems"]), flush=True)
        return EXIT_NOT_STOCK
    if o.get("--proof-only") == "1":                                        # prove, run nothing (the kit's tests)
        print(f"{PREFIX} proven stock (proof only)", flush=True)
        return 0
    print(f"{PREFIX} proven stock: boltz {pins['boltz_version'] if pins else '?'} at {proof['boltz_tree']['root'] if proof['boltz_tree'] else '?'}; running: boltz {' '.join(stock_args)}", flush=True)
    import importlib
    entry = getattr(importlib.import_module(STOCK_ENTRY[0]), STOCK_ENTRY[1])
    from . import phase                                                  # the per-item PHASE timing line, placed after the proof and the stock import; numerics unchanged
    phase.install()
    if kopts:                                                            # the KERNELS census (kernels.py): one line per pass + the REQUIRE guard (exit 5) before the first item's timed work; RNG-neutral
        kernels.install(mode="stock", **kopts)
    try:
        entry.main(args=stock_args, prog_name="boltz", standalone_mode=True)
    except SystemExit as e:
        return int(e.code or 0) if not isinstance(e.code, str) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
