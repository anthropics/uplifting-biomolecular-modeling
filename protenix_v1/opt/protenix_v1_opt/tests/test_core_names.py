"""Every shared-core attribute the kit reaches through `opt_core.gates` and `opt_core.kernels.triattn.triattn_native` exists in the
importable core (a renamed or removed core name must fail here, on the CPU, instead of stepping aside inside an except at run time)."""
import ast
import importlib
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_OPT = os.path.dirname(HERE)                                                     # opt/protenix_v1_opt
LEVERS = os.path.join(os.path.dirname(KIT_OPT), "forward", "v05_addon", "ptxfpf", "levers_ptx1.py")
MODES = os.path.join(KIT_OPT, "modes.py")


def _attr_uses(source: str, alias: str):
    """The attribute names read off `alias` (`alias.name`) anywhere in `source`."""
    return sorted(set(re.findall(r"\b%s\.([A-Za-z_][A-Za-z0-9_]*)" % re.escape(alias), source)))


def test_gates_names_used_by_modes_exist():
    gates = importlib.import_module("opt_core.gates")
    src = open(MODES).read()
    used = _attr_uses(src, "G")
    assert "nvidia_smi_probe" in used, used
    missing = [n for n in used if not hasattr(gates, n)]
    assert not missing, "opt_core.gates lacks: %s" % missing


def test_device_memory_mib_answers_without_raising():
    modes = importlib.import_module("protenix_v1_opt.modes")
    mem = modes.device_memory_mib()
    assert mem is None or (isinstance(mem, int) and mem > 0), mem


def test_triattn_native_names_used_by_levers_exist():
    cc = pytest.importorskip("opt_core.kernels.triattn.triattn_native")
    src = open(LEVERS).read()
    used = _attr_uses(src, "_CC")
    assert used, "levers_ptx1 no longer names triattn_native attributes through _CC (update this test)"
    missing = [n for n in used if not hasattr(cc, n)]
    assert not missing, "opt_core.kernels.triattn.triattn_native lacks: %s" % missing
    assert callable(cc.stack_key)


def test_no_dead_core_names_remain():
    src = open(LEVERS).read() + open(MODES).read()
    for dead in ("gpu_probe(", "_CC.available(", "_CC.dist_version("):
        assert dead not in src, dead
