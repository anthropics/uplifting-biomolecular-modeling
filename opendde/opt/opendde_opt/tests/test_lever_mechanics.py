"""The big line's fail-loud path and its allocator: a planned offload lever the unit did not install is a fallback by name (PARTIAL,
`pred` exits 3), and the line's PYTORCH_CUDA_ALLOC_CONF is refused by name when it cannot bind.

drop_bond_mask: the featurizer's unread `bond_mask` is dropped from its return — patched at once when the featurizer is imported,
at its import otherwise (the core's per-site patch, opt_core.autoload.patch_attr_at_import); every drop counted; the gate names an unwrapped
featurizer and an absent key.

ERRATA_02 on the carried offload unit: every runtime fallback path counts itself in odde_offload.STATS["fallbacks"] (reason -> n) and the
kit turns every count into a named row-level event (`offload:<reason> x<n>` in levers_fallback -> PARTIAL, `pred` exit 3). The reasons
below are the unit's own strings at the cited lines; the real paths reachable without a model are driven by fault injection.

Out-of-memory propagates through the served path: a fallback route never absorbs it.

The per-file census of the re-raise sites.

Per-lever activation-evidence lines (report.lever_lines): one `[opendde-opt] LEVER name=<lever> state=<on|off|skipped> ...` line per
registry lever per process, rendered by the core's report.lever_line (blank-free tokens; impl and origin lead); `skipped` always carries reason=.

The EXIT tally never drops a key: every kit lever module's scalar counters print through the core's one formatter
(``opt_core.report.tally_fields``) with the equality keys first (installed / armed / n_gpu / rank / group) and no truncation marker.

The TriMul kernels are the core provider's since 0.2.57 (no kit copy; their int64 row addressing is the core's own test). OpenDDE's structural pair tensor (C = 384) passes 2**31 addressable
elements at N >= 2365 rows, where an int32 product wraps below the buffer.
"""
import importlib
import json
import os
import re
import sys
import types

import pytest

from opt_core import report as core_report

from opendde_opt import ActivationError, big, bondmask, cli, modes, registry, report, stack
from opendde_opt import report as _report
from opendde_opt.tests import _stubs
from opendde_opt.tests._stubs import TREE


def _offload(installed=True, stages=("struct", "trunk", "conf"), trunk=True, bigln=True, diffz=True, free_templ=True):
    m = types.ModuleType("odde_offload")
    m._INSTALLED = installed
    m.CFG = {"stages": set(stages), "diffz": diffz, "free_templ": free_templ, "rows": 256}
    m._BIGLN = {"installed": bigln, "limit": 1 << 31, "hits": 0, "calls": 7 if bigln else 0}
    m.STATS = {"h2d_bytes": 3 * 2 ** 30, "d2h_bytes": 2 ** 30, "strided_bytes": 0,
               "stages": [{"stage": "trunk", "wall_s": 10.0, "host_rss_gib": 12.0, "host_hwm_gib": 13.5, "peak_alloc_gib": 20.0, "peak_reserved_gib": 22.5}],
               # the unit's flat per-lever entry counters after one prediction (ran-or-refuse reads these; the stage timers above are not evidence)
               "calls_struct": int("struct" in stages), "calls_diff_cache": int("struct" in stages), "calls_trunk": int(trunk and "trunk" in stages) * 10,
               "calls_conf": int(trunk and "conf" in stages), "calls_disto": int(trunk and "conf" in stages), "calls_diffz": int(diffz) * 200,
               "calls_free_templ": int(free_templ and "struct" in stages), "fallbacks": {}}
    t = types.ModuleType("odde_offload_trunk"); t._TRUNK_INSTALLED = trunk
    return m, t


def _planned_report():
    return {"active": True, "mode": "big", "line": "BIG_F", "levers_planned": list(modes.LINES["BIG_F"].levers),
            "levers_applied": [x for x in modes.LINES["BIG_F"].levers if x in modes._OFFLOAD_LEVERS or x == "served_levers_hook"], "levers_fallback": [], "partial": False}


@pytest.fixture
def big_process(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("ODDE_", "PYTORCH_CUDA", "FPF_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(stack, "_REPORT", _planned_report())
    monkeypatch.delitem(sys.modules, "odde_served_levers", raising=False)
    for unit in ("odde_triattn_bind", "odde_trimul_bind"):                          # the provider bindings an in-process exact activation of an earlier test imported:
        monkeypatch.delitem(sys.modules, unit, raising=False)                        # this process's big line never reached them (a loaded unit at zero calls = never ran)
    def use(m, t):
        monkeypatch.setattr(stack, "_REPORT", _planned_report())                   # every fake state starts from the planned report (refresh folds into it)
        monkeypatch.setitem(sys.modules, "odde_offload", m)
        monkeypatch.setitem(sys.modules, "odde_offload_trunk", t)
    big._reset()
    yield use
    big._reset()                                                                   # an in-process `pred` leaves its size-gate plan in the adapter: one activation per kit process, several here


@pytest.mark.parametrize("state, fell", [
    (dict(), []),
    (dict(installed=False), sorted(modes._OFFLOAD_LEVERS)),                                       # the shim swallowed the install exception
    (dict(stages=("trunk", "conf")), ["free_templ", "pair_offload_struct"]),                       # a stage missing (free_templ acts in struct)
    (dict(trunk=False), ["pair_offload_conf", "pair_offload_trunk"]),                              # the trunk module failed to import: stock trunk + confidence
    (dict(bigln=False, diffz=False), ["bigln_guard", "diffz"]),                                    # diffz off = the BIGLN guard never installed
])
def test_offload_levers_the_unit_did_not_install_are_fallbacks_by_name(big_process, state, fell):
    big_process(*_offload(**state))
    rep = stack.refresh()
    assert rep["levers_fallback"] == fell and rep["partial"] is bool(fell)
    assert not (set(rep["levers_applied"]) & set(fell)) and (set(modes._OFFLOAD_LEVERS) - set(fell)) <= set(rep["levers_applied"])
    line = _report.exit_tally_line(1)
    assert "offload={" in line and "installed=" in line and "peak_reserved_gib=22.5" in line and "host_hwm_gib=13.5" in line
    assert ("PARTIAL" in line) is bool(fell) and all(f in line for f in fell)
    st = rep["kit_stats"]["offload"]
    assert st["n_stage_records"] == 1 and st["h2d_gib"] == 3.0 and st["stages_cfg"] == "+".join(sorted(state.get("stages", ("struct", "trunk", "conf"))))


def test_pred_on_big_with_a_fallen_offload_lever_exits_not_active(tmp_path, monkeypatch, capsys, big_process, weights_root):
    site = _stubs.make_site(str(tmp_path)); monkeypatch.syspath_prepend(site); monkeypatch.setenv("MODEL_OPT", _stubs.TREE)
    big_process(*_offload(trunk=False))
    q = tmp_path / "q.json"; q.write_text(json.dumps([{"name": "a", "sequences": [{"proteinChain": {"sequence": "MKT", "count": 1, "unpairedMsaPath": str(tmp_path / "a.a3m")}}]}]))   # a precomputed MSA path: the frozen-weights gate admits the chain under use_msa=true (the PARTIAL path is what this test asserts)
    active = dict(_planned_report(), line="BIG_F(x)", line_name="BIG_F", applied={"exports": {}}, notes=[])
    monkeypatch.setattr(stack, "activate", lambda *a, **k: dict(active))
    monkeypatch.setattr(cli, "run_stock_cli", _stubs.fake_stock_cli)   # writes the owed structure files, rc 0
    monkeypatch.setattr(stack, "applied", lambda: {})
    out = tmp_path / "o"
    assert cli.main(["pred", "--mode", "big", "-i", str(q), "-o", str(out)]) == cli.EXIT_NOT_ACTIVE
    assert "PRED PARTIAL refused: exit 3 fallbacks=pair_offload_conf,pair_offload_trunk" in capsys.readouterr().out
    m = json.load(open(out / "opt_manifest.json"))
    assert m["activation"]["partial"] is True and m["activation"]["levers_fallback"] == ["pair_offload_conf", "pair_offload_trunk"]


def test_the_lines_allocator_is_refused_by_name_when_it_cannot_bind(monkeypatch):
    assert all(modes.LINES[n].allocator == modes.EXPANDABLE for n in modes.BIG_LINES)   # every big line's field; the writer is one (alloc.apply)
    monkeypatch.setattr(big, "_apply_mem", lambda res: None)                                     # the memory levers are test_big_adapter.py's; this is the allocator's test
    res = modes.resolve("big", _stubs.TREE, {})
    assert res.line.name == "BIG_F" and res.exports[modes.ALLOCATOR] == modes.EXPANDABLE
    monkeypatch.setenv(modes.ALLOCATOR, "max_split_size_mb:128")
    with pytest.raises(ActivationError, match="names another allocator configuration"):          # the core's refusal (opt_core.mem.torch_alloc), re-raised with the line's name
        big.apply(res)
    monkeypatch.setenv(modes.ALLOCATOR, modes.EXPANDABLE)                                          # the same value: not a conflict
    fake = types.ModuleType("torch"); fake.cuda = types.SimpleNamespace(is_initialized=lambda: False, is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert big.apply(res)["alloc_export"] == "present"
    fake.cuda = types.SimpleNamespace(is_initialized=lambda: True, is_available=lambda: True)
    with pytest.raises(ActivationError, match="already initialised"):
        big.apply(res)
    monkeypatch.delenv(modes.ALLOCATOR)
    assert modes.resolve("exact", _stubs.TREE, {}).line.allocator is None                          # every other line: no allocator, no refusal


FEATURIZER_SRC = '''
class _T:
    def __init__(self, n): self.n = n
    def numel(self): return self.n * self.n
    def element_size(self): return 8

class Featurizer:
    calls = 0
    def __init__(self, n=10, with_key=True): self.n, self.with_key = n, with_key
    def get_mask_features(self):
        Featurizer.calls += 1
        out = {"is_protein": [1] * self.n}
        if self.with_key:
            out["bond_mask"] = _T(self.n)
        return out
'''


@pytest.fixture
def featurizer_package(tmp_path, monkeypatch):
    """A stand-in `opendde.data.core.featurizer` importable from tmp_path (the real stock package must not be loaded by the kit's tests)."""
    pkg = tmp_path / "opendde" / "data" / "core"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "opendde", tmp_path / "opendde" / "data", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "featurizer.py").write_text(FEATURIZER_SRC)
    for k in [k for k in sys.modules if k == "opendde" or k.startswith("opendde.")]:
        monkeypatch.delitem(sys.modules, k)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bondmask, "STATS", {"installed": False, "armed": False, "dropped": 0, "bytes_dropped": 0, "missing": 0})
    from opt_core import autoload                                              # one process = one stock module; a fresh stand-in per test needs a fresh per-site patch
    monkeypatch.delitem(autoload._PATCHES, (bondmask.TARGET, f"{bondmask.CLASS}.{bondmask.METHOD}"), raising=False)
    finders = list(sys.meta_path); before = set(sys.modules)
    yield tmp_path
    sys.meta_path[:] = finders                                                 # a finder left armed by a test never leaks
    for k in [k for k in sys.modules if k not in before and (k == "opendde" or k.startswith("opendde."))]:
        del sys.modules[k]                                                     # nor the stand-in featurizer (later tests read sys.modules)


def test_the_lever_is_on_every_line():
    for name, ln in modes.LINES.items():
        assert "drop_bond_mask" in ln.levers, name
    assert modes.resolve("off", "/x", {}).line is None                          # the stock route has no line: nothing patched


def test_patch_now_when_the_featurizer_is_already_imported(featurizer_package):
    m = importlib.import_module(bondmask.TARGET)
    bondmask.install()
    assert bondmask.STATS["installed"] and not bondmask.STATS["armed"]
    out = m.Featurizer(12).get_mask_features()
    assert "bond_mask" not in out and out["is_protein"] == [1] * 12
    assert {k: v for k, v in bondmask.STATS.items() if k != "patch"} == {"installed": True, "armed": False, "dropped": 1, "bytes_dropped": 12 * 12 * 8, "missing": 0}
    bondmask.install()                                                          # idempotent: one wrapper, one count per call
    m.Featurizer(3).get_mask_features()
    assert bondmask.STATS["dropped"] == 2 and m.Featurizer.calls == 2 and m.Featurizer.get_mask_features._drop_bond_mask


def test_patch_at_import_through_the_core_per_site_patch(featurizer_package):
    assert bondmask.TARGET not in sys.modules
    bondmask.install()
    assert bondmask.STATS["armed"] and not bondmask.STATS["installed"]
    p = bondmask.STATS["patch"]
    assert sys.meta_path[0] is p and p.state == "armed" and p.name == "drop_bond_mask"
    m = importlib.import_module(bondmask.TARGET)                                # the patch serves the import (the module's body first), then wraps the class
    bondmask._sync()
    assert bondmask.STATS["installed"] and not bondmask.STATS["armed"] and p not in sys.meta_path and p.state == "installed"
    assert "bond_mask" not in m.Featurizer(4).get_mask_features() and bondmask.STATS["dropped"] == 1


def test_a_featurizer_without_the_method_fails_activation_by_name(featurizer_package):
    m = importlib.import_module(bondmask.TARGET)
    del m.Featurizer.get_mask_features
    with pytest.raises(bondmask.ActivationError, match="get_mask_features"):
        bondmask.install()
    assert not bondmask.STATS["installed"]


def test_the_gate_names_an_unwrapped_featurizer_and_an_absent_key(featurizer_package):
    planned = list(modes.LINES["S1"].levers)
    assert bondmask.fallbacks(planned) == []                                     # nothing imported: no event (the served route's worker)
    m = importlib.import_module(bondmask.TARGET)
    assert bondmask.fallbacks(planned) == ["drop_bond_mask: the featurizer is imported but not wrapped (installed after its import without the patch)"]
    bondmask.install()
    assert bondmask.fallbacks(planned) == []
    m.Featurizer(5, with_key=False).get_mask_features()
    assert bondmask.fallbacks(planned) == ["drop_bond_mask: 1 featurizer calls returned no bond_mask (the stock changed under the lever)"]
    assert bondmask.fallbacks([x for x in planned if x != "drop_bond_mask"]) == []
    assert _report.kit_stats()["drop_bond_mask"]["missing"] == 1


def test_refresh_applies_or_names_the_lever(featurizer_package, monkeypatch):
    m = importlib.import_module(bondmask.TARGET)
    planned = ["dit_align", "drop_bond_mask"]                                   # no XL / offload / graph lever: their own gates stay out of this one
    monkeypatch.setattr(stack, "_REPORT", {"active": True, "levers_planned": planned, "levers_applied": [x for x in planned if x != "drop_bond_mask"], "levers_fallback": [], "partial": False})
    rep = stack.refresh()
    assert "drop_bond_mask" not in rep["levers_applied"] and rep["partial"] is True and rep["levers_fallback"][0].startswith("drop_bond_mask: the featurizer is imported but not wrapped")
    bondmask.install(); m.Featurizer(6).get_mask_features()
    monkeypatch.setattr(stack, "_REPORT", {"active": True, "levers_planned": planned, "levers_applied": [x for x in planned if x != "drop_bond_mask"], "levers_fallback": [], "partial": False})
    rep = stack.refresh()
    assert "drop_bond_mask" in rep["levers_applied"] and rep["partial"] is False and rep["kit_stats"]["drop_bond_mask"]["dropped"] == 1


def test_registry_row_and_docs():
    from opendde_opt import registry
    lv = registry.LEVERS["drop_bond_mask"]
    assert lv.kit == registry.HOUSE and lv.tier == "exact" and lv.cls == "datapath" and os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(bondmask.__file__)), lv.file))
    assert registry.validate() == []


UNIT = os.path.join(_stubs.TREE, "opt", "forward", "fast_inference", "levers", "OFFLOAD")
# (file, the recorder call as the source carries it): one row per runtime path of the CARRY FALLBACK CENSUS
RECORDERS = [
    ("odde_offload.py", 'fallback("libcudart unavailable: column blocks via pinned staging + CPU strided copy")'),
    ("odde_offload.py", 'fallback("HostPair pinned alloc refused: pageable memory")'),
    ("odde_offload.py", 'fallback("HostPair staging buffer pinned alloc failed: pageable staging buffer")'),
    ("odde_offload.py", 'fallback("pitched column copy consistency check mismatch: staging path")'),
    ("odde_offload.py", 'fallback("pitched column copy unavailable: staging path")'),
    ("odde_offload.py", 'fallback("pinned pool alloc failed: pageable buffer")'),
    ("odde_offload.py", 'fallback(f"CKPT save {name} failed")'),
    ("odde_offload.py", 'fallback("diffz: pair_z release skipped")'),
    ("odde_offload.py", 'fallback("diffz not applied: pair_z is None or not inplace_safe (stock per-step clone)")'),
    ("odde_offload.py", 'fallback("trunk/conf offload not importable: trunk + confidence run stock")'),
    ("odde_offload_trunk.py", 'OO.fallback("struct stage not offloaded: residue pair tensor materialised on the GPU")'),
]


def test_every_runtime_path_of_the_census_carries_the_units_recorder():
    src = {f: open(os.path.join(UNIT, f)).read() for f in ("odde_offload.py", "odde_offload_trunk.py")}
    for f, call in RECORDERS:
        assert src[f].count(call) == 1, (f, call)
    assert src["odde_offload.py"].count("fallback(") == len([r for r in RECORDERS if r[0] == "odde_offload.py"]) + 1      # the def + every site
    assert '"fallbacks": {}' in src["odde_offload.py"] and 'STATS["diffz_images"]' in src["odde_offload.py"]


def _with_fallbacks(fallbacks, diffz_images=1, **state):
    """The unit's state after a run with the ERRATA_02 counters (`_offload`'s fake carries the pre-errata STATS: no counters, no runtime event)."""
    m, t = _offload(**state)
    m.STATS["fallbacks"] = dict(fallbacks)
    m.STATS["diffz_images"] = diffz_images
    return m, t


def test_a_unit_without_the_counters_has_no_runtime_event(big_process):
    big_process(*_offload())
    rep = stack.refresh()
    assert rep["levers_fallback"] == [] and rep["partial"] is False and rep["kit_stats"]["offload"]["fallbacks"] == {}


@pytest.mark.parametrize("reason", ["HostPair pinned alloc refused: pageable memory", "HostPair staging buffer pinned alloc failed: pageable staging buffer",
                                    "pinned pool alloc failed: pageable buffer", "diffz not applied: pair_z is None or not inplace_safe (stock per-step clone)",
                                    "CKPT save trunk failed"])
def test_every_counted_fallback_is_a_named_event_and_partial(big_process, reason):
    big_process(*_with_fallbacks({reason: 3}))
    rep = stack.refresh()
    assert rep["levers_fallback"] == [f"offload:{reason} x3"] and rep["partial"] is True
    assert set(modes._OFFLOAD_LEVERS) <= set(rep["levers_applied"])                         # installed levers stay applied: the event is the row's
    assert rep["kit_stats"]["offload"]["fallbacks"] == {reason: 3}
    line = _report.exit_tally_line(1)
    assert "PARTIAL" in line and f"offload:{reason} x3" in line


def test_no_fallback_is_no_event(big_process):
    big_process(*_with_fallbacks({}))
    rep = stack.refresh()
    assert rep["levers_fallback"] == [] and rep["partial"] is False and rep["kit_stats"]["offload"]["diffz_images"] == 1


def test_a_planned_diffz_that_never_built_its_image_is_named(big_process):
    big_process(*_with_fallbacks({}, diffz_images=0))
    rep = stack.refresh()
    assert rep["levers_fallback"] == ["offload:diffz never engaged (no LayerNorm image of pair_z built)"] and rep["partial"] is True
    big_process(*_with_fallbacks({}, diffz_images=0, diffz=False))                        # diffz configured off: the install gate names it, not this one
    rep = stack.refresh()
    assert rep["levers_fallback"] == ["diffz"] and rep["partial"] is True


def test_pred_exits_3_on_a_runtime_fallback(tmp_path, monkeypatch, capsys, big_process, weights_root):
    site = _stubs.make_site(str(tmp_path)); monkeypatch.syspath_prepend(site); monkeypatch.setenv("MODEL_OPT", _stubs.TREE)
    big_process(*_with_fallbacks({"HostPair pinned alloc refused: pageable memory": 2}))
    q = tmp_path / "q.json"; q.write_text(json.dumps([{"name": "a", "sequences": [{"proteinChain": {"sequence": "MKT", "count": 1, "unpairedMsaPath": str(tmp_path / "a.a3m")}}]}]))   # a precomputed MSA path: the frozen-weights gate admits the chain under use_msa=true (the PARTIAL path is what this test asserts)
    active = dict(_planned_report(), line="BIG_F(x)", line_name="BIG_F", applied={"exports": {}}, notes=[])
    monkeypatch.setattr(stack, "activate", lambda *a, **k: dict(active))
    monkeypatch.setattr(cli, "run_stock_cli", _stubs.fake_stock_cli)   # writes the owed structure files, rc 0
    monkeypatch.setattr(stack, "applied", lambda: {})
    out = tmp_path / "o"
    assert cli.main(["pred", "--mode", "big", "-i", str(q), "-o", str(out)]) == cli.EXIT_NOT_ACTIVE
    assert "PRED PARTIAL refused: exit 3 fallbacks=offload:HostPair pinned alloc refused: pageable memory x2" in capsys.readouterr().out
    m = json.load(open(out / "opt_manifest.json"))
    assert m["activation"]["partial"] is True and m["activation"]["levers_fallback"] == ["offload:HostPair pinned alloc refused: pageable memory x2"]


def test_fault_injection_on_the_units_real_paths(tmp_path, monkeypatch):
    """The paths reachable without a model, driven on the carried bytes: the checkpoint save, libcudart, and (with CUDA) the pinned allocations."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("ODDE_OFFLOAD_LOG", "")
    monkeypatch.syspath_prepend(UNIT)
    for k in ("odde_offload", "odde_offload_trunk"):
        monkeypatch.delitem(sys.modules, k, raising=False)
    import odde_offload as OO
    OO.STATS["fallbacks"].clear()
    # 1) the cycle checkpoint save fails (an unwritable directory) -> counted, the run continues
    monkeypatch.setitem(OO.CFG, "ckpt_dir", str(tmp_path / "no" / "such")) if "ckpt_dir" in OO.CFG else None
    monkeypatch.setattr(torch, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    OO.ckpt_save("trunk", {"x": 1})
    assert OO.STATS["fallbacks"].get("CKPT save trunk failed") == 1
    # 2) libcudart cannot be loaded -> the staging route, counted
    import ctypes
    monkeypatch.setattr(OO, "_CUDART", None)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: (_ for _ in ()).throw(OSError("no libcudart")))
    assert OO._cudart() is False
    assert OO.STATS["fallbacks"].get("libcudart unavailable: column blocks via pinned staging + CPU strided copy") == 1
    if torch.cuda.is_available():
        # 3) the pinned host allocations fail -> pageable memory, counted (HostPair, its staging buffer, the pinned pool)
        real_empty = torch.empty
        def no_pin(*a, **k):
            if k.get("pin_memory"):
                raise RuntimeError("pinned memory exhausted")
            return real_empty(*a, **k)
        monkeypatch.setattr(torch, "empty", no_pin)
        hp = OO.HostPair(8, 4, device="cuda", pin=True)
        assert hp.pinned is False and OO.STATS["fallbacks"].get("HostPair pinned alloc refused: pageable memory") == 1
        hp._stage(2)
        assert OO.STATS["fallbacks"].get("HostPair staging buffer pinned alloc failed: pageable staging buffer") == 1
        OO._PIN_POOL.clear(); OO.pooled_pinned((16,))
        assert OO.STATS["fallbacks"].get("pinned pool alloc failed: pageable buffer") == 1
    # every count is a named event through the kit's gate
    m = types.ModuleType("odde_offload"); m._INSTALLED = True; m.CFG = {"stages": {"struct", "trunk", "conf"}, "diffz": True, "free_templ": True}
    m.STATS = dict(OO.STATS); m._BIGLN = {"installed": True, "limit": 1 << 31, "hits": 0}
    monkeypatch.setitem(sys.modules, "odde_offload", m)
    t = types.ModuleType("odde_offload_trunk"); t._TRUNK_INSTALLED = True; monkeypatch.setitem(sys.modules, "odde_offload_trunk", t)
    named = big.offload_runtime_fallbacks(list(modes._OFFLOAD_LEVERS))
    assert all(f"offload:{r} x{n}" in named for r, n in OO.STATS["fallbacks"].items()) and len(named) >= 2


OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # opendde/opt
SITES = {                                                                                  # served file (opt-relative) -> `if is_oom(...): raise` sites
    "opendde_opt/stack.py": 1,
    "forward/fast_inference/levers/ACCEL/odde_served_levers.py": 4,
    "forward/fast_inference/levers/ACCEL/odde_accel_v2.py": 1,
    "forward/fast_inference/levers/ACCEL/sitecustomize.py": 2,
    "forward/fast_inference/levers/ARMT/odde_arm_t/__init__.py": 3,
    "forward/fast_inference/levers/ARMT/sitecustomize.py": 2,
    "forward/fast_inference/levers/OFFLOAD/odde_offload.py": 6,
    "forward/fast_inference/levers/OFFLOAD/sitecustomize.py": 3,
    "forward/fast_inference/levers/XL/sitecustomize.py": 1,
    "forward/fast_inference/levers/DITFAST/tools/odde_addon.py": 1,
    "forward/fast_inference/src/fpf_engines/__init__.py": 2,
}


def _oom_class():
    try:
        import torch
        return torch.cuda.OutOfMemoryError
    except Exception:  # noqa: BLE001 — a torch without the CUDA OOM class: the classifier matches the class name
        return type("OutOfMemoryError", (RuntimeError,), {})


def test_every_served_reroute_site_reraises_oom_first_with_the_cores_one_classifier():
    for rel, n in SITES.items():
        src = open(os.path.join(OPT, rel)).read()
        assert len(re.findall(r"^\s*if is_oom\(\w+\): raise\b.*$", src, re.M)) == n, rel
        assert "from opt_core.oom import is_oom" in src, rel
    assert sum(SITES.values()) == 26
    needle = "def " + "is_oom"                                                          # one spelling: the core's (no kit-local classifier anywhere under opt/)
    for dp, _dn, fn in os.walk(OPT):
        for f in fn:
            if f.endswith(".py"):
                assert needle not in open(os.path.join(dp, f), errors="replace").read(), os.path.join(dp, f)


# ---- the offload unit's pinned host buffers: a refused pinned allocation is an error naming ODDE_OFFLOAD_PIN; =0 selects pageable buffers with no exception route
def _offload_pinned_env(monkeypatch, pin):
    import importlib
    import sys
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("ODDE_OFFLOAD_PIN", pin)
    monkeypatch.delenv("ODDE_OFFLOAD", raising=False)
    monkeypatch.syspath_prepend(os.path.join(OPT, "forward/fast_inference/levers/OFFLOAD"))
    sys.modules.pop("odde_offload", None)
    m = importlib.import_module("odde_offload")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real_empty = torch.empty
    calls = {"pinned": 0}

    def empty(*a, **kw):
        if kw.get("pin_memory"):
            calls["pinned"] += 1
            raise RuntimeError("CUDA error: out of memory (cudaHostAlloc)")
        kw.pop("pin_memory", None)
        return real_empty(*a, **kw)

    monkeypatch.setattr(m.torch, "empty", empty)
    return m, torch, calls


def test_a_refused_pinned_host_allocation_is_an_oom_error_naming_the_switch(monkeypatch):
    m, torch, calls = _offload_pinned_env(monkeypatch, "1")
    from opt_core.oom import is_oom
    m._PIN_POOL.clear(); before = dict(m.STATS["fallbacks"])
    with pytest.raises(m.PinnedHostMemoryRefused) as ei:
        m.pooled_pinned((4, 4))
    assert is_oom(ei.value) and "ODDE_OFFLOAD_PIN=0" in str(ei.value) and calls["pinned"] == 1 and m.STATS["fallbacks"] == before
    with pytest.raises(m.PinnedHostMemoryRefused):
        m.HostPair(4, 2, dtype=torch.float32, device="cuda")


def test_pin_off_takes_the_pageable_path_without_an_exception_route(monkeypatch):
    m, torch, calls = _offload_pinned_env(monkeypatch, "0")
    assert m.CFG["pin"] is False
    m._PIN_POOL.clear(); before = dict(m.STATS["fallbacks"])
    t = m.pooled_pinned((4, 4)); hp = m.HostPair(4, 2, dtype=torch.float32, device="cuda"); hp._stage(2)
    assert tuple(t.shape) == (4, 4) and hp.pinned is False and calls["pinned"] == 0 and m.STATS["fallbacks"] == before


def test_the_big_pin_setting_is_the_rows_and_no_variable_sets_it():
    from opendde_opt import big, modes
    F = modes.MODE_LINES["big"]
    assert big.compose(F, {}).exports["ODDE_OFFLOAD_PIN"] == "1"                                   # the row's setting (pin=1) is the unit's switch
    with pytest.raises(big.BigRefusal, match="OPENDDE_BIG_PAIR_OFFLOAD_PIN"):                  # no caller variable sets it
        big.compose(F, {"OPENDDE_BIG_PAIR_OFFLOAD_PIN": "0"})
    with pytest.raises(ValueError, match="pin="):                                                     # the setting's domain, checked where a line is built
        modes.big_line(F, modes.BIG_MEM_LEVERS[F], {("pair_offload", "pin"): "2"})


def test_one_line_per_registry_lever_with_the_core_grammar():
    rep = {"mode": "fast", "line_name": "LSTAR", "route": "cli", "active": True,
           "levers_planned": ["dit_align", "arm_u", "dit_attn_bf16"], "levers": ["dit_align", "arm_u", "dit_attn_bf16"],
           "levers_applied": ["dit_align", "arm_u"],
           "levers_fallback": ["dit_attn_bf16: 10 calls reached the site, 0 served in bf16"]}
    stats = {}
    lines = report.lever_lines(rep, stats)
    assert len(lines) == len(registry.LEVERS)
    by = {ln.split("name=")[1].split(" ")[0]: ln for ln in lines}
    assert set(by) == set(registry.LEVERS)
    assert by["dit_align"].startswith("[opendde-opt] LEVER name=dit_align state=on ") and "mode=fast line=LSTAR" in by["dit_align"]
    assert " origin=kit " in by["dit_align"]
    assert "state=skipped reason=fell_back:_dit_attn_bf16:_10_calls" in by["dit_attn_bf16"]
    assert "state=off" in by["diffz"] and "reason=" not in by["diffz"]
    for ln in lines:
        off_by_pin = any(f" name={n} " in ln for n in registry.OFF_LINE)
        assert (" state=skipped " in ln or (off_by_pin and " state=off " in ln)) == (" reason=" in ln), ln


def test_off_mode_has_every_lever_off():
    lines = report.lever_lines({"mode": "off", "active": True, "route": "cli"}, {})
    assert lines and all(" state=off " in ln + " " for ln in lines)


def test_on_lever_carries_its_counters():
    rep = {"mode": "exact", "line_name": "S1", "active": True, "levers_planned": ["drop_bond_mask"], "levers_applied": ["drop_bond_mask"]}
    ln = [x for x in report.lever_lines(rep, {"drop_bond_mask": {"dropped": 3, "missing": 0, "bytes_dropped": 4096}}) if "name=drop_bond_mask " in x][0]   # the key report.kit_stats writes
    assert "state=on" in ln and "dropped=3" in ln and "bytes_dropped=4096" in ln and "missing=0" in ln


def _rowpair_like_stats(n_scalar=35):
    d = {"installed": True, "armed": True, "n_gpu": 2, "rank": 0, "group": True}
    i = 0
    while len(d) < n_scalar:
        d[f"counter_{i:02d}"] = i
        i += 1
    d["schedule"] = {"layout_B": 128, "conf_rows": 128}          # nested: printed as a count, never expanded, never dropped
    return {"rowpair_tp": d, "served_levers": {"active": True, "n_installs": 1}}


def test_exit_tally_prints_id_keys_first_and_every_key():
    if not hasattr(core_report, "tally_fields"):
        pytest.skip("the pinned opt_core has no report.tally_fields (opt_core >= 0.4.4): this kit line rides the 0.4.4 re-pin")
    stats = _rowpair_like_stats(35)
    fields = report.tally_fields(stats)
    assert isinstance(fields, list) and len(fields) == 2, fields
    tp_block = fields[0]
    assert tp_block.startswith("rowpair_tp={installed=True,armed=True,n_gpu=2,rank=0,group=True,"), tp_block
    for k in stats["rowpair_tp"]:
        assert f"{k}=" in tp_block, (k, tp_block)                # all 35 scalars + the nested count present
    assert "…+" not in tp_block and "...+" not in tp_block, tp_block
    assert fields[1].startswith("served_levers={active=True,"), fields[1]
    assert not hasattr(report, "TALLY_MAX_FIELDS")                # the kit's truncating copy is gone
    print("EXIT-TALLY-EXAMPLE " + f"{report.PREFIX} EXIT pid=0 n_gpu=2 sharding=rowpair " + " ".join(fields))


KIT = os.path.join(TREE, "opt", modes.KIT_DIRS["fast_inference"])


def test_the_tree_carries_no_trimul_kernel_copy():
    """0.2.57: the TriMul kernels are the core provider's (opt_core.kernels.trimul, bound by tier word in levers/ARMT/odde_trimul_bind.py); the kit's
    src/fpf_trimul copy, its arch tile table and the arm's bf16-planes construction on it are retired -- src/ holds the fpf_engines adapter only."""
    assert not os.path.isfile(os.path.join(KIT, "src", "fpf_trimul", "kernels.py")) and os.path.isfile(os.path.join(KIT, "src", "fpf_engines", "__init__.py"))
    assert not os.path.exists(os.path.join(KIT, "levers", "ARMT", "third_party", "fpf_trimul_v32"))
    arm_src = open(os.path.join(KIT, "levers", "ARMT", "odde_arm_t", "__init__.py")).read()
    for gone in ("load_trimul", "U2_CELLS", "u2_admission", "fpf_trimul", "_u2_trimul"):
        assert gone not in arm_src, gone
    assert "TB.serve(self, z, mask" in arm_src                                          # the arm's one TriMul route: the provider binding
