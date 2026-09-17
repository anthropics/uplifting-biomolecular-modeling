"""The hook shims (levers/{ACCEL,ARMT,OFFLOAD,XL}/sitecustomize.py) chain FORWARD from their own sys.path position under one re-entry guard per
import cascade (sys._odde_shim_chain), so on every line's path each shim executes exactly once, in path order, without recursion, and a foreign
sitecustomize.py listed after the kit's directories executes once too (0.2.61). site.py imports only the FIRST `sitecustomize` on the path; the
test executes that one the same way and lets the chain do the rest. CPU only: every lever switch is unset (the shims' install bodies stay
inert) and `odde_served_levers` is a stub whose arm_hook() is counted."""
import importlib.util
import os
import sys
import types

import pytest

from opendde_opt import modes
from . import _stubs

SHIM = "sitecustomize.py"
SHIM_DIRS = {k: modes.kit_dir(_stubs.TREE, k) for k in (modes.ACCEL, modes.ARMT, modes.OFFLOAD, modes.XL)}


def _shim(d):
    return os.path.realpath(os.path.join(d, SHIM))


@pytest.fixture
def clean(monkeypatch, tmp_path):
    for k in set(modes.ALL_SWITCHES if hasattr(modes, "ALL_SWITCHES") else ()) | {sw for sws in modes.LEVER_SWITCHES.values() for sw in sws} | {
            "ODDE_SERVED_LEVERS", "ODDE_ARM_U", "ODDE_ARM_Z", "ODDE_OFFLOAD", "ODDE_TRAJ_EVERY", "ODDE_XL", "ODDE_XL_PROBE"}:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delattr(sys, "_odde_shim_chain", raising=False)
    arm = types.SimpleNamespace(calls=0)
    stub = types.ModuleType("odde_served_levers"); stub.arm_hook = lambda: setattr(arm, "calls", arm.calls + 1)
    monkeypatch.setitem(sys.modules, "odde_served_levers", stub)
    foreign = tmp_path / "foreign_site"; foreign.mkdir()
    (foreign / SHIM).write_text("import sys\nsys._foreign_site_runs = getattr(sys, '_foreign_site_runs', 0) + 1\n")
    monkeypatch.setattr(sys, "_foreign_site_runs", 0, raising=False)
    loads = []
    real = importlib.util.spec_from_file_location
    def counting(name, location=None, *a, **k):                                       # every chained load the shims make goes through here
        if location and os.path.basename(str(location)) == SHIM:
            loads.append(os.path.realpath(str(location)))
        return real(name, location, *a, **k)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", counting)
    return types.SimpleNamespace(arm=arm, foreign=str(foreign), loads=loads, monkeypatch=monkeypatch)


def _run_like_site(dirs, env, fx):
    """sys.path = dirs (+ the interpreter's own), env applied, then import the FIRST sitecustomize.py on dirs exactly as site.py does."""
    for k, v in env.items():
        fx.monkeypatch.setenv(k, v)
    fx.monkeypatch.delattr(sys, "_odde_shim_chain", raising=False)
    del fx.loads[:]; fx.arm.calls = 0; sys._foreign_site_runs = 0
    fx.monkeypatch.setattr(sys, "path", list(dirs) + [p for p in sys.path if p not in dirs])
    meta_before = list(sys.meta_path)
    first = next(d for d in dirs if os.path.isfile(os.path.join(d, SHIM)))
    spec = importlib.util.spec_from_file_location("sitecustomize", os.path.join(first, SHIM))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert sys.meta_path == meta_before, "a shim installed an import hook with every lever switch unset"
    assert sys._odde_shim_chain["live"] is False, "the shim that opened the cascade closes it"
    return list(sys._odde_shim_chain["ran"])


def _paths():
    """Every sys.path the kit composes: each line's (S1, LSTAR2A, the big rows on the offload path and resident, --n_gpu P), every big
    composition, plus the XL unit's path order and the orderings no shipped line composes (the latent hazard itself)."""
    out = {}
    for name, ln in modes.LINES.items():
        out[name] = [modes.kit_dir(_stubs.TREE, k) for k in ln.path_order]
    for n in modes.BIG_LINES:
        for mem in ((), modes.BIG_MEM_LEVERS[n]):
            out[f"{n}{list(mem)}"] = [modes.kit_dir(_stubs.TREE, k) for k in modes.big_line(n, mem).path_order]
    out["all four + reversed (no line: the ordering hazard itself)"] = [SHIM_DIRS[modes.XL], SHIM_DIRS[modes.OFFLOAD], SHIM_DIRS[modes.ACCEL], SHIM_DIRS[modes.ARMT]]
    out["XL unit listed last (XL_PATH_ORDER)"] = [modes.kit_dir(_stubs.TREE, k) for k in modes.XL_PATH_ORDER]
    out["OFFLOAD + XL together (no shipped line composes both)"] = [modes.kit_dir(_stubs.TREE, k) for k in modes.OFFLOAD_PATH_ORDER + (modes.XL,)]
    assert modes.kit_dir(_stubs.TREE, modes.OFFLOAD) in out["BIG_F"]                                 # the offload path is among them
    return out


@pytest.mark.parametrize("served", [False, True], ids=["switches_unset", "ODDE_SERVED_LEVERS=1"])
def test_every_shim_on_every_line_path_runs_once_in_order_and_a_foreign_one_after_them_too(clean, served):
    armt = _shim(SHIM_DIRS[modes.ARMT])
    for label, dirs in _paths().items():
        dirs = dirs + [clean.foreign]
        ran = _run_like_site(dirs, {"ODDE_SERVED_LEVERS": "1"} if served else {}, clean)
        shims = [_shim(d) for d in dirs if os.path.isfile(os.path.join(d, SHIM))]
        assert len(shims) >= 3 and shims[-1] == _shim(clean.foreign) and _shim(SHIM_DIRS[modes.ACCEL]) in shims, (label, shims)   # not vacuous: the kit's shims are on every path
        first_is_armt = shims[0] == armt
        expect = [s for s in shims if not (served and s == armt and not first_is_armt)]     # under the served-levers hook the ARM-T shim is passed over by name
        assert ran == expect, (label, ran, expect)
        assert len(set(ran)) == len(ran), (label, "a shim ran twice")
        assert clean.loads == expect, (label, "loads", clean.loads, expect)                   # the site import + each chained file loaded exactly once: no recursion, no re-entry
        assert sys._foreign_site_runs == 1, (label, sys._foreign_site_runs)
        assert clean.arm.calls == (1 if served and _shim(SHIM_DIRS[modes.ACCEL]) in shims else 0), (label, clean.arm.calls)   # the hook armed once (no idempotent double-arm)


def test_a_back_chaining_foreign_shim_does_not_re_run_the_kit_shims(clean, tmp_path):
    """A foreign sitecustomize that scans from the head of the path and re-imports the kit's first shim (the pre-0.2.61 pattern) gets a no-op
    mid-cascade: no shim body runs twice, the hook arms once; a later site-style import in the same interpreter starts a fresh cascade."""
    back = tmp_path / "back_chainer"; back.mkdir()
    (back / SHIM).write_text("import importlib.util, os, sys\nsys._foreign_site_runs = getattr(sys, '_foreign_site_runs', 0) + 1\n"
                            "f = os.path.join(sys.path[0], 'sitecustomize.py')\nspec = importlib.util.spec_from_file_location('sc_head', f)\n"
                            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n")
    dirs = [SHIM_DIRS[modes.ACCEL], SHIM_DIRS[modes.XL], str(back)]
    ran = _run_like_site(dirs, {"ODDE_SERVED_LEVERS": "1"}, clean)
    assert ran == [_shim(d) for d in dirs] and sys._foreign_site_runs == 1 and clean.arm.calls == 1     # ACCEL re-imported by the back-chainer: body skipped, no second arm
    assert clean.loads == [_shim(d) for d in dirs] + [_shim(dirs[0])]                                  # the re-import happened (4 loads) and was a no-op
    again = _run_like_site(dirs, {"ODDE_SERVED_LEVERS": "1"}, clean)                                    # the cascade closed: a second start-up runs afresh
    assert again == ran and clean.arm.calls == 1
