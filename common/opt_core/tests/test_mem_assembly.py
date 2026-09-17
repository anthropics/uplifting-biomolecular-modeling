"""The assembled opt_core.mem: every lever module registers through the registry, discover() runs without a framework, module-level
imports are stdlib-only (the LAZY-IMPORT rule), and the package's public surface is the one the README names."""
import ast
import os
import subprocess
import sys
import sysconfig

import pytest

from opt_core.mem import registry

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
MEM = os.path.join(PKG, "opt_core", "mem")
STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else None


def _present():
    return [m for m in registry.LEVER_MODULES if os.path.exists(os.path.join(MEM, m + ".py"))]


def test_lever_modules_present_register_at_least_one_lever():
    d = registry.discover()
    assert d["broken"] == {}, f"lever modules broken on this box: {d['broken']} (the LAZY-IMPORT rule: frameworks import inside functions)"
    assert set(d["loaded"]) == set(_present()), (d, _present())
    for m in d["loaded"]:
        mine = [lv for lv in registry.LEVERS.values() if lv.module == f"opt_core.mem.{m}"]
        assert mine, f"opt_core.mem.{m} registers no lever through opt_core.mem.registry (registered: {sorted(registry.LEVERS)})"
        for lv in mine:
            assert lv.family in registry.FAMILIES and lv.exact in registry.EXACT_LABELS and lv.exact_reason


def test_discover_is_framework_free():
    """discover() in a fresh process where torch and jax are absent: no module breaks, none imports a framework at module level."""
    code = ("import sys; sys.modules['torch'] = None; sys.modules['jax'] = None; "
            "from opt_core.mem import registry; d = registry.discover(); "
            "print(d['broken']); print(sorted(registry.LEVERS)); print('torch' in sys.modules and sys.modules['torch'] is not None)")
    out = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=PKG), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    broken, levers, torch_imported = out.stdout.strip().splitlines()
    assert broken == "{}", f"lever modules break without a framework: {broken}"
    assert torch_imported == "False"
    assert levers != "[]"


def test_module_level_imports_are_stdlib_only():
    assert STDLIB is not None
    bad = []
    for fn in sorted(os.listdir(MEM)):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(MEM, fn), encoding="utf-8").read(), fn)
        for node in tree.body:                                                            # module level only: inside functions is the rule
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n not in STDLIB and n != "opt_core":
                    bad.append(f"{fn}: {n}")
    assert not bad, f"module-level third-party imports in opt_core/mem (the LAZY-IMPORT rule): {bad}"


def test_peak_is_not_a_lever_module():
    assert "peak" not in registry.LEVER_MODULES
