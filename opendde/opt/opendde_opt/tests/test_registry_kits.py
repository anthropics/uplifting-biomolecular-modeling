"""Every registry lever cites an existing kit file, the kit versions the modes rely on are the carried files' own, and every
switch a line exports is a switch some carried kit file reads.

The CPU suite's route is one file, tests/cpu_suite.sh."""
import os
import re

import pytest

from opendde_opt import modes, registry, stack
from opendde_opt.tests._stubs import PINNED, TREE


OPT = os.path.join(TREE, "opt")
KIT = os.path.join(OPT, "forward", "fast_inference")


def test_registry_invariants():
    assert registry.validate() == []
    assert len(registry.LEVERS) >= 20


def test_every_lever_and_knob_cites_an_existing_kit_file():
    assert registry.LEVERS and registry.KNOBS
    for name, lv in registry.LEVERS.items():
        for ref in lv.file.split(";"):
            f = ref.strip().split(":")[0]
            assert os.path.exists(os.path.join(OPT, f)), (name, f)
    for knob, (_d, ref, _m) in registry.KNOBS.items():
        assert os.path.exists(os.path.join(OPT, ref.split(":")[0])), (knob, ref)


def test_every_lever_takes_effect_in_the_pred_process():
    """The registry's route column has one value: every lever the lines carry takes effect in the `pred` process (stack.ROUTE)."""
    assert {lv.routes for lv in registry.LEVERS.values()} == {stack.ROUTE} == {"cli"}


def test_kit_versions_are_the_carried_bytes_own():
    def version_of(path):
        m = re.search(r'^__version__\s*=\s*"([^"]+)"', open(path).read(), re.M)
        return m.group(1) if m else None
    assert version_of(os.path.join(KIT, "levers/ACCEL/odde_served_levers.py")) == "0.2.0"
    assert version_of(os.path.join(KIT, "levers/ACCEL/odde_accel_v2.py")) == "0.3.0"
    assert version_of(os.path.join(KIT, "levers/ARMT/odde_arm_t/__init__.py")) == "0.3"
    assert version_of(os.path.join(KIT, "src/fpf_engines/__init__.py")) == "0.1.2"
    assert os.path.isdir(os.path.join(modes.kit_layer(TREE), "cueq_cache_shipped"))          # the kit layer: the tuning-cache location (README, no table shipped)
    assert not any(f.endswith(".json") for f in os.listdir(os.path.join(modes.kit_layer(TREE), "cueq_cache_shipped")))   # no tile table is shipped
    # the kit's DITFAST tools every line runs (levers/DITFAST)
    assert os.path.isfile(os.path.join(modes.kit_dir(TREE, modes.DITFAST_TOOLS), "odde_addon.py"))


def _env_reads():
    names = set()
    pat = re.compile(r"""(?:environ(?:\.get)?\(|getenv\(|_flag\(|_env_int\(|_env_float\(|_env_bool\(|_env\()\s*['"]([A-Z][A-Z0-9_]+)['"]|\$\{?([A-Z][A-Z0-9_]+)(?::[-=]|\})""")
    for dp, _dn, fn in os.walk(OPT):
        for f in fn:
            if f.endswith((".py", ".sh")):
                for m in pat.finditer(open(os.path.join(dp, f), errors="replace").read()):
                    names.add(m.group(1) or m.group(2))
    return names


def test_every_exported_switch_is_read_by_a_carried_kit_file():
    reads = _env_reads()
    for ln in modes.LINES.values():
        for k in ln.exports:
            if k == modes.ALLOCATOR:
                continue                                                      # read by torch's allocator, not by a kit file
            assert k in reads, f"line {ln.name} exports {k}, which no carried kit file reads"
    for k in modes.FPF_SWITCHES + modes.ARM_SWITCHES:
        assert k in reads, k


def _script_commands():
    lines = open(os.path.join(TREE, "tests", "cpu_suite.sh")).read().splitlines()
    cmds = [l for l in lines if l.strip() and not l.startswith("#") and l.strip() != "set -e"]
    return [re.sub(r'\s+"\$@"$', "", c) for c in cmds]                      # the pytest line forwards extra arguments; the quote shows the bare command


def test_the_route_installs_the_documented_extras():
    cmds = _script_commands()
    assert len(cmds) == 3 and cmds[0].startswith("pip install 'torch==2.7.1'") and cmds[-1] == "python -m pytest opt/opendde_opt/tests", cmds
    assert "-e ../common/opt_core" in cmds[1] and "'opt[test]'" in cmds[1]
