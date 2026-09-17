"""The engine-contract adapters of three carried kernel units live in a sibling ``_engine_adapter.py`` (the functions that import the
engine's upstream package lazily, inside the call); every public name stays at its original module path as a thin wrapper that imports
the adapter on FIRST CALL and delegates.  Held here, per moved name: (a) the name resolves at its original path; (b) its signature equals
the adapter function's; (c) a call forwards positional and keyword arguments unchanged and returns the adapter's result (the adapter
function is replaced by a recorder: the real bodies need the engine's package and a GPU); (d) importing the unit does not import its
``_engine_adapter`` (a single-engine install may not carry it); (e) no module of the unit other than ``_engine_adapter.py`` imports the
engine's upstream package.  ``fpf_mkpf._rebind_levers_prologue`` also runs for real on CPU (without the engine's package it records the
import error by name in ``_STATE`` — the same record through the wrapper and through the adapter)."""
import ast
import importlib
import inspect
import os
import subprocess
import sys

import pytest

import opt_core

PKG = os.path.dirname(opt_core.__file__)
MOVED = {                                   # unit -> (module holding the public names, relative to the unit package; names)
    "fpf_triatt_epi": ("epilogue", ("fn", "fn_block_residual", "_stock_prologue_attention")),
    "fpf_mkpf": ("", ("_rebind_levers_prologue",)),
    "fpf_triatt_k2b": ("", ("fn", "fn_k2")),
}
ENGINE_PACKAGE = "protenix"                 # the upstream package the adapters import lazily


def _unit_modules(unit):
    """(public module, adapter module) of a unit, imported by package path; a unit whose framework is absent here is skipped by name."""
    holder, _names = MOVED[unit]
    base = "opt_core.kernels." + unit
    try:
        pub = importlib.import_module(base + ("." + holder if holder else ""))
        ad = importlib.import_module(base + "._engine_adapter")
    except ImportError as e:                # torch / triton absent on this interpreter: named skip, never a silent pass
        pytest.skip(f"{unit}: not importable here ({type(e).__name__}: {e})")
    return pub, ad


@pytest.mark.parametrize("unit,name", [(u, n) for u, (_h, names) in MOVED.items() for n in names])
def test_public_name_resolves_with_the_adapter_signature(unit, name):
    pub, ad = _unit_modules(unit)
    w, f = getattr(pub, name), getattr(ad, name)
    assert callable(w) and callable(f) and w is not f
    assert inspect.signature(w) == inspect.signature(f), (unit, name, str(inspect.signature(w)), str(inspect.signature(f)))
    assert w.__module__ == pub.__name__ and f.__module__ == ad.__name__


@pytest.mark.parametrize("unit,name", [(u, n) for u, (_h, names) in MOVED.items() for n in names])
def test_wrapper_forwards_arguments_and_returns_the_adapter_result(unit, name, monkeypatch):
    pub, ad = _unit_modules(unit)
    sig = inspect.signature(getattr(ad, name))
    calls, sentinel = [], object()

    def recorder(*a, **k):
        calls.append((a, k)); return sentinel
    monkeypatch.setattr(ad, name, recorder)
    params = list(sig.parameters.values())
    positional = [p for p in params if p.default is inspect.Parameter.empty]
    optional = [p for p in params if p.default is not inspect.Parameter.empty]
    args = [object() for _ in positional]
    kwargs = {p.name: object() for p in optional}
    out = getattr(pub, name)(*args, **kwargs)
    assert out is sentinel and len(calls) == 1
    got_a, got_k = calls[0]
    bound = sig.bind(*got_a, **got_k); bound.apply_defaults()       # whatever mix the wrapper forwards, it binds to the same values
    want = sig.bind(*args, **kwargs); want.apply_defaults()
    assert list(bound.arguments.items()) == list(want.arguments.items())
    for k_, v in want.arguments.items():
        assert bound.arguments[k_] is v, k_


@pytest.mark.parametrize("unit", sorted(MOVED))
def test_importing_the_unit_does_not_import_its_adapter(unit):
    holder = MOVED[unit][0]
    target = "opt_core.kernels." + unit + ("." + holder if holder else "")
    code = ("import importlib, sys, json\n"
            f"try:\n    importlib.import_module({target!r})\nexcept ImportError as e:\n    print(json.dumps({{'skip': repr(e)}})); raise SystemExit(0)\n"
            f"print(json.dumps({{'adapter_loaded': {('opt_core.kernels.' + unit + '._engine_adapter')!r} in sys.modules, "
            f"'engine_loaded': any(m.split('.')[0] == {ENGINE_PACKAGE!r} for m in sys.modules)}}))\n")
    env = dict(os.environ); env["PYTHONPATH"] = os.path.dirname(PKG) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    import json
    rec = json.loads(r.stdout.strip().splitlines()[-1])
    if "skip" in rec:
        pytest.skip(f"{unit}: not importable here ({rec['skip']})")
    assert rec == {"adapter_loaded": False, "engine_loaded": False}, rec


@pytest.mark.parametrize("unit", sorted(MOVED))
def test_only_the_adapter_imports_the_engine_package(unit):
    udir = os.path.join(PKG, "kernels", unit)
    offenders = []
    for r_, dirs, files in os.walk(udir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py") or f == "_engine_adapter.py" or f.startswith(("selftest", "microbench", "verify_batch", "run_spec_script")):
                continue                    # the unit's own test programs import the engine when RUN; they are not modules of the served path
            tree = ast.parse(open(os.path.join(r_, f), encoding="utf-8").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    offenders += [f"{f}:{node.lineno}" for a in node.names if a.name.split(".")[0] == ENGINE_PACKAGE]
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == ENGINE_PACKAGE:
                    offenders.append(f"{f}:{node.lineno}")
    assert offenders == [], offenders
    ad = os.path.join(udir, "_engine_adapter.py")
    assert os.path.isfile(ad)
    tree = ast.parse(open(ad, encoding="utf-8").read())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all((n.module or "").split(".")[0] != ENGINE_PACKAGE if isinstance(n, ast.ImportFrom) else all(a.name.split(".")[0] != ENGINE_PACKAGE for a in n.names)
               for n in top), "the adapter imports the engine's package lazily (inside its functions), never at module level"


def test_rebind_levers_prologue_runs_through_the_wrapper_on_cpu():
    """Without the engine's package the body records the import error by name in _STATE['rebound_closure_cells'] and returns None: the
    same observable result through the public name and through the adapter."""
    pub, ad = _unit_modules("fpf_mkpf")
    class _Lev: _STATS = {}
    saved = dict(pub._STATE)
    try:
        pub._STATE.pop("rebound_closure_cells", None)
        assert pub._rebind_levers_prologue(_Lev, object(), object()) is None
        via_wrapper = pub._STATE.get("rebound_closure_cells")
        pub._STATE.pop("rebound_closure_cells", None)
        assert ad._rebind_levers_prologue(_Lev, object(), object()) is None
        via_adapter = pub._STATE.get("rebound_closure_cells")
        assert via_wrapper == via_adapter and via_wrapper is not None      # 0 cells rebound (engine importable) or the named import error (not)
    finally:
        pub._STATE.clear(); pub._STATE.update(saved)
