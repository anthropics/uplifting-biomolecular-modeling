"""The stock caller — mode ``off``: the upstream command line in the pristine interpreter, proven before it runs.

``design --mode off`` execs this module in the pristine interpreter (``<stock python> -I -m rfdiffusion3_opt.stock_design --proof-json
<temporary file> -- <the rfd3 design arguments>``; the parent folds the proof into ``opt_manifest.json`` ``stock_env_proof``; RFDIFFUSION3_STOCK_PYTHON names that interpreter, default: the
current one; ``-I`` = isolated: that interpreter's own site-packages, no PYTHONPATH, no user site) with the kit's names (stock/PINS.json ``stock_environment.must_be_absent``) stripped from the environment (cli.stock_env); upstream's
own switches (``upstream_switches_unset``: RFD3_DENSE_SDPA_ATTENTION, RFD3_LOW_MEMORY_MODE) and torch's variables pass through as the caller set
them and are reported on the STOCK line (``upstream_env=``). Before anything of the upstream stack is imported the process proves what it is
(``env_proof``): no forbidden environment name, this package's autoload finder not armed, no kit module loaded, the installed rfd3
tree PRISTINE (the eight target files at their stock sha, the kit's two new files absent — stock/check_pins.py in this interpreter),
torch not yet imported; the proof is written as
JSON and read into ``opt_manifest.json`` by the ``design`` process. A process that fails its proof exits 3 without running anything.
The upstream distribution's pin (stock/PINS.json: commit / archive, foundry/utils/torch.py) is REPORTED in the proof (``pins.notes``) and
never refuses: what refuses is a forbidden name, an armed finder, a kit module, torch imported early, or an rfd3 tree that is not pristine.
torch's own numerics variables the caller's environment carries (e.g. TORCH_ALLOW_TF32_CUBLAS_OVERRIDE) reach this child unchanged; what
torch then computed with is the KERNELS line's ``tf32_matmul=`` read-back.
Then upstream's own entry point — ``rfd3.cli:app`` (the ``rfd3`` console script, pyproject.toml:92) — runs ``design`` with the
arguments unchanged, observed by the KERNELS census (``census.observe`` after the proof, ``census.finish`` after the run: the one
``KERNELS route=stock|default …`` line — stock when the line carries upstream's ``compile_model=true``, default otherwise, routes.ROUTES;
report-only on these two routes and READ-ONLY in this process — logging handlers on upstream's own loggers, no upstream callable wrapped or re-bound,
no import hook (census.Observer.install) —: an accelerator that missed its expected kind is named on a ``KERNELS report-only …`` line and the run keeps its
own exit code; the census lands in the proof's ``after.kernels``). This
module imports the standard library and, at run time, the stock package — nothing of the kit and, from this package, only ``registry``
before the proof (the tree's paths, the pins file and the pin checker's file; standard library, nothing of the activation core) and
``kernels`` + the routes table after it. This is the only stock caller in the tree (``run.sh design --mode off`` and ``design --mode off`` both reach it),
and ``warm --mode off`` runs the same proof (``env_proof``) before it builds the engine.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterable, List, Optional

from ._autoload import TAG
from ._emit import emit
from .registry import check_pins_module as _check_pins, pins as _pins

PREFIX = f"[{TAG} stock]"                                              # "[rfdiffusion3-opt stock]"
EXIT_NOT_STOCK = 3                                                      # the package's EXIT_NOT_ACTIVE: nothing ran
KIT_MODULES = ("rfd3.model.hoist", "rfd3.model.cudagraph_sampler")     # the kit's new modules (its edits live in stock-named files: the sha proof covers them)
STOCK_ENTRY = ("rfd3.cli", "app")                                       # the `rfd3` console script's entry point


def _finder_armed() -> bool:
    from opt_core.stock_proof import armed_finders           # the core's one reader: the mode finder (opt_core.autoload) armed on the meta path
    return bool(armed_finders())


GATE_CORE_MODULES = ("opt_core.gates", "opt_core.modes", "opt_core.report")   # what the package's own `off` gate imports (stack.activate): allowed in warm's off child, which gates before it proves


def _core_extra(extra_allowed: Iterable[str] = ()) -> list:
    from opt_core.stock_proof import CORE_ALLOWED_IN_STOCK, core_modules_loaded      # core modules beyond the proof machinery itself (opt_core, opt_core.stock_proof)
    return core_modules_loaded(allowed=tuple(CORE_ALLOWED_IN_STOCK) + tuple(extra_allowed))


def env_proof(environ=None, forbidden: Optional[Iterable[str]] = None, core_allowed: Iterable[str] = ()) -> dict:
    """What this process is, measured before the upstream stack is imported. ``ok`` is True only when every check passed.
    ``core_allowed`` names core modules this process may hold beyond the proof machinery (warm's off child gates the mode through
    the package first: GATE_CORE_MODULES); the design stock child holds none. The pin checker's findings split in two: the rfd3
    TREE (``pins.pristine``: the eight target files at their stock sha, the kit's two new files absent) refuses; the distribution's
    install source / commit and foundry/utils/torch.py (``pins.notes``) are reported, never refused — a checkout upstream itself runs
    is stock."""
    environ = os.environ if environ is None else environ
    pins = _pins()
    names = list(forbidden) if forbidden is not None else list(pins["stock_environment"]["must_be_absent"])   # the KIT's names; upstream's own switches are stock's knobs, reported below, never forbidden
    present = sorted(k for k in names if k in environ)
    upstream_env = {k: environ[k] for k in pins["stock_environment"].get("upstream_switches_unset", []) if k in environ}   # upstream's own switches as the caller set them (reported on the STOCK line)
    kit_loaded = sorted(m for m in sys.modules if m in KIT_MODULES)
    cp = _check_pins()
    notes, detail = cp.check(pins, "pristine")                       # every finding of the pin checker is a NOTE here; the tree state below is the gate
    tree_ok = bool(detail.get("site")) and bool(detail.get("pristine"))
    proof = {
        "python": sys.executable,
        "python_version": ".".join(str(x) for x in sys.version_info[:3]),
        "forbidden_names_checked": names,
        "forbidden_present": present,
        "autoload_finder_armed": _finder_armed(),
        "kit_modules_loaded": kit_loaded,
        "core_modules_loaded": _core_extra(core_allowed),
        "torch_imported": "torch" in sys.modules,
        "rfd3_imported": any(m == "rfd3" or m.startswith("rfd3.") for m in sys.modules),
        "pins": {"notes": notes, "distribution": detail.get("distribution"), "site": detail.get("site"), "files": detail.get("files"), "pristine": detail.get("pristine"), "stack": detail.get("stack"), "pinned_stack": detail.get("pinned_stack")},
        "rfdiffusion3_opt_env": environ.get("RFDIFFUSION3_OPT"),
        "upstream_env": upstream_env,
    }
    proof["ok"] = (not present and not proof["autoload_finder_armed"] and not kit_loaded and not proof["core_modules_loaded"] and not proof["torch_imported"] and tree_ok)
    reasons = []
    if present:
        reasons.append(f"forbidden environment names present: {present}")
    if proof["autoload_finder_armed"]:
        reasons.append("the package's autoload finder is armed (RFDIFFUSION3_OPT is set in this process)")
    if kit_loaded:
        reasons.append(f"kit modules loaded: {kit_loaded}")
    if proof["core_modules_loaded"]:
        reasons.append(f"core modules loaded beyond the proof: {proof['core_modules_loaded']}")
    if proof["torch_imported"]:
        reasons.append("torch was imported before the proof")
    if not detail.get("site"):
        reasons.append("rfd3 is not importable in this interpreter")
    elif not detail.get("pristine"):
        files = detail.get("files") or {}
        reasons.append("rfd3 tree is not pristine: " + ", ".join(f"{k}={v}" for k, v in sorted(files.items()) if k.startswith("rfd3/") and v != "stock"
                                                               and not (v == "absent" and k in pins["kit_new_files_absent_in_stock"])))
    proof["reasons"] = reasons
    return proof


def stock_line(proof: dict) -> str:
    """The one line a proven stock process prints before it imports upstream; `upstream_env=` names upstream's own switches as the caller set them (`none` = unset: upstream's defaults)."""
    d = proof["pins"]["distribution"] or {}
    ups = ",".join(f"{k}={v}" for k, v in sorted((proof.get("upstream_env") or {}).items())) or "none"
    det = (proof.get("det") or {}).get("level", 0)
    return f"{PREFIX} STOCK rc-foundry={d.get('version')} tree=pristine python={proof['python']} forbidden_absent={len(proof['forbidden_names_checked'])} upstream_env={ups} det={det}"


def pins_note_lines(proof: dict) -> List[str]:
    """One `PINS <finding>` line per pin-checker finding (the distribution's install source / commit, foundry/utils/torch.py): reported, never a refusal."""
    return [f"{PREFIX} PINS {n}" for n in (proof.get("pins") or {}).get("notes") or []]


def not_stock_line(proof: dict) -> str:
    return f"{PREFIX} NOT STOCK: " + "; ".join(proof["reasons"])


def write_proof(path: str, proof: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(proof, fh, indent=1, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def run_stock(args: List[str]) -> int:
    """upstream's `rfd3` entry point with `args` (`design key=value ...`), in this process. Returns its exit code."""
    import importlib
    mod = importlib.import_module(STOCK_ENTRY[0])
    app = getattr(mod, STOCK_ENTRY[1])
    sys.argv = ["rfd3"] + list(args)
    try:
        app()
    except SystemExit as e:
        code = e.code
        return 0 if code is None else (code if isinstance(code, int) else 1)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    from ._autoload import core_gate
    core_gate()                                              # the core pin gate of this (pristine) interpreter, before the proof imports opt_core.stock_proof; imports nothing of the core itself
    from . import det as _det
    argv = list(sys.argv[1:] if argv is None else argv)
    proof_json, det_level = None, 0
    if "--proof-json" in argv:
        i = argv.index("--proof-json")
        proof_json = argv[i + 1]
        del argv[i:i + 2]
    if "--det" in argv[:max(argv.index("--"), 0) if "--" in argv else len(argv)]:
        i = argv.index("--det")
        det_level = _det.level(argv[i + 1])
        del argv[i:i + 2]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv or argv[0] != "design":
        emit(f"{PREFIX} usage: python -I -m rfdiffusion3_opt.stock_design [--proof-json <file>] [--det 0|1] -- design <rfd3 design arguments>")
        return 2
    proof = env_proof()
    proof["det"] = {"level": det_level}
    if proof_json:
        write_proof(proof_json, proof)
    for note in pins_note_lines(proof):
        emit(note)
    if not proof["ok"]:
        emit(not_stock_line(proof))
        return EXIT_NOT_STOCK
    if det_level:
        proof["det"] = _det.apply(det_level)                  # after the proof (the inherited environment was clean), before upstream is imported: the deterministic recipe, as the kit routes apply it
        if proof_json:
            write_proof(proof_json, proof)
    emit(stock_line(proof))
    from . import census                                    # the KERNELS census (census.py; standard library at import): route stock | default by the line's compile switch (routes.ROUTES)
    from .routes import route_of                             # the routes table, core-free (the pristine interpreter need not carry the shared core for the census)
    census.observe(route_of("off", argv[1:]).name, "off", argv[1:])   # stock | default by the line's compile switch
    rc = run_stock(argv)
    rc = census.finish(rc)                                  # the line once; report-only on the stock routes: the run's own exit code, nothing stops here
    if proof_json:
        proof["after"] = {"exit_code": rc, "kit_modules_loaded": sorted(m for m in sys.modules if m in KIT_MODULES),
                          "forbidden_present": sorted(k for k in proof["forbidden_names_checked"] if k in os.environ),
                          "kernels": census.record()}
        write_proof(proof_json, proof)
    return rc


if __name__ == "__main__":
    sys.exit(main())
