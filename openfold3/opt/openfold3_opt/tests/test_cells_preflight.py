"""The pair-track pre-flight (cells/pairfused.preflight) on the core's ONE safe-settings mechanism: per requested lever, for the TRUNK pair stack on a
named (cc, triton) part — a table row serves (the pinned parts: nothing new prints), the core's SAFE cell serves (a part whose table has no row: the
lever is INSTALLED, its census line ends ` cells=safe settings=safe:no_cell:<shape>`, the core prints its ONE line at the first call), or nothing serves
(the lever is installed all the same: those calls run the line's own statement, counted `fallback:no-cell:…=<n>`; activation does not refuse over an untested part).  CPU only: stacks are named, tables are the shipped ones or copies."""
import copy
import re
import json
import os

import pytest

from openfold3_opt import report, stack
from openfold3_opt.cells import pairfused
from openfold3_opt.tests import _stubs
from opt_core.attn import pair_fused as PF

H100, A100, NOROWS = ("9.0", "3.3"), ("8.0", "3.6"), ("7.0", "3.3")


@pytest.fixture
def clean_state():
    import importlib
    pf = importlib.import_module("openfold3_opt.cells.pairfused")            # the live module (an earlier test may have re-imported the package)
    saved = {k: (copy.copy(v) if isinstance(v, (dict, list)) else v) for k, v in pf.STATE.items()}
    yield pf.STATE
    pf.STATE.clear(); pf.STATE.update(saved)


@pytest.fixture
def no_80_rows(monkeypatch, tmp_path):
    """The core tables minus their cc 8.0 rows: a part the tables do not know, WITH safe cells (SAFE_ROWS 8.0)."""
    tab = copy.deepcopy(PF.cells()); tab["rows"] = [r for r in tab["rows"] if r["cc"] != "8.0"]
    monkeypatch.setattr(PF, "cells", lambda path=None: tab)
    from opt_core.kernels.fpf_trimul_v4 import table as VT
    core = {k: v for k, v in VT.load_table().items() if not k.startswith("8.0|")}
    monkeypatch.setattr(VT, "load_table", lambda path=None: core)


def test_pinned_part_every_lever_served_by_a_row():
    v = pairfused.preflight(list(pairfused.LEVER_NAMES), "fpf", stack=H100)
    assert set(v) == set(pairfused.LEVER_NAMES) and all(x["served"] for x in v.values())
    assert v["trimul_v4"] == {"served": True, "cells": "9.0|3.3", "settings": None, "reason": ""}          # the core's fpf_trimul_v4 table: the exact row of the pinned stack
    CELL_KEY = re.compile(r"9\.0\|(?:\*|\d+\.\d+)$")                        # the core table's (capability | triton) key grammar for this part — which of its rows serves is the core's measurement
    assert CELL_KEY.match(v["pair_transition"]["cells"]) and v["pair_transition"]["settings"] is None, v["pair_transition"]
    assert v["triatt_block"]["settings"] is None and all(CELL_KEY.match(c) for c in v["triatt_block"]["cells"].split("+")), v["triatt_block"]
    assert {l for l, x in v.items() if not x['served']} <= {"triatt_block"}
    lnl = pairfused.preflight(["triatt_block", "pair_transition"], "lnl", stack=H100)
    assert all(x["served"] for x in lnl.values())
    assert pairfused.preflight(["triatt_block"], "torch", stack=H100) == {"triatt_block": {"served": True, "cells": None, "settings": None, "reason": ""}}   # impl torch pins no pair_fused row


def test_a_part_without_rows_serves_the_any_capability_safe_cells_for_the_trunk_shape():
    """A capability no table has a row for (cc 7.0): every lever is SERVED by the core's any-capability SAFE settings, named — never `no-cell`."""
    v = pairfused.preflight(list(pairfused.LEVER_NAMES), "fpf", stack=NOROWS)
    assert v == {"trimul_v4": {"served": True, "cells": "safe", "settings": "safe:no_cell:cc70", "reason": ""},
                 "triatt_block": {"served": True, "cells": "safe", "settings": "safe:no_cell:c_z=128,H=4", "reason": ""},
                 "pair_transition": {"served": True, "cells": "safe", "settings": "safe:no_cell:c_z=128,n=4", "reason": ""}}
    assert pairfused.preflight(["triatt_block", "pair_transition"], "lnl", stack=NOROWS) == {
        "triatt_block": {"served": True, "cells": "default", "settings": None, "reason": ""}, "pair_transition": {"served": True, "cells": "default", "settings": None, "reason": ""}}


def test_a_part_without_rows_serves_the_safe_cells_it_has(no_80_rows):
    v = pairfused.preflight(list(pairfused.LEVER_NAMES), "fpf", stack=A100)
    assert v["pair_transition"] == {"served": True, "cells": "safe", "settings": "safe:no_cell:c_z=128,n=4", "reason": ""}   # SAFE_ROWS pair_fused:transition 8.0
    assert v["trimul_v4"] == {"served": True, "cells": "safe", "settings": "safe:no_cell:cc80", "reason": ""}                    # SAFE_ROWS pair_fused:trimul 8.0
    tb = v["triatt_block"]                                                                                                       # SAFE_ROWS pair_fused:prologue|epilogue 8.0: served by the safe cells, or refused BY NAME where the core's table marks the piece off on that part (`+off(<why>)`)
    assert tb == {"served": True, "cells": "safe", "settings": "safe:no_cell:c_z=128,H=4", "reason": ""} or (not tb["served"] and "+off(" in tb["reason"] and tb["reason"].startswith("no-cell:fpf:")), tb
    assert {l for l, x in v.items() if not x['served']} <= {"triatt_block"}


def test_activation_does_not_refuse_on_a_part_nothing_serves(monkeypatch):
    """A trunk shape no cell serves on this GPU is not a refusal of the mode: the activation is what it is on the pinned parts (no NOT ACTIVE, no note);
    the levers install and those calls run the line's own statement per call, counted."""
    d = _stubs.stub_dist()
    try:
        pkg = _stubs.reset_package()
        from openfold3_opt import stack as st
        monkeypatch.setattr(st, "gpu_probe", lambda environ=None: {"name": "Some GPU", "cc": "7.0", "sm": "sm70", "memory_mib": 16384, "supported": True, "probe": "stub"})
        dry = st.activate("fast", dry_run=True)
        assert dry["reason"] == report.DRY_RUN_OK and not any("engage" in n for n in dry.get("notes") or []) and "cells_preflight" not in dry
    finally:
        _stubs.reset_package()
        _stubs.unstub_dist(d)


def test_a_trunk_shape_no_cell_serves_is_a_route_under_strict(clean_state):
    """OPENFOLD3_OPT_PAIR_STRICT=1 fails the run on a trunk-shape refusal that is a defect of the line (dtype / layout / mask words) — never on the
    core's no-cell word: those calls run the line's own statement, `fallback:<word>=<n>` on the census, exit 0."""
    import importlib
    pairfused = importlib.import_module("openfold3_opt.cells.pairfused")
    st = clean_state; st["strict"] = True
    word = "no-cell:fpf:epilogue:256x8x32:8.0|3.6+no_safe(c_z_above_128_slower_than_stock_on_cc8.0)"
    pairfused._strict_refusal("triatt_block", word, pairfused.TRUNK_C, 384)                      # a route: returns
    pairfused._strict_refusal("pair_transition", "no-cell:fpf:transition:128x512:7.0|3.3", pairfused.TRUNK_C, 384)
    with pytest.raises(RuntimeError):
        pairfused._strict_refusal("triatt_block", "mask-batch", pairfused.TRUNK_C, 384)          # a defect of the line: fails the run by name
    pairfused._strict_refusal("triatt_block", "mask-batch", 64, 384)                              # not the trunk: a route
    st.update(installed=True, levers=["triatt_block"], impl="fpf", core="flash_triattn", cells={})
    pairfused._count("triatt_block", False, word); pairfused._count("triatt_block", False, word); pairfused._count("triatt_block", True)
    line = [l for l in pairfused.census_lines() if "name=triatt_block " in l][0]
    assert f" served=1 fallback=2 fallback:{word}=2 " in line and "cells=" not in line and "state=on" in line


def test_pinned_part_activation_is_unchanged(monkeypatch):
    d = _stubs.stub_dist()
    try:
        pkg = _stubs.reset_package()
        from openfold3_opt import stack as st
        monkeypatch.setattr(st, "gpu_probe", lambda environ=None: {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90", "memory_mib": 81559, "supported": True, "probe": "stub"})
        dry = st.activate("fast", dry_run=True)
        assert dry["reason"] == report.DRY_RUN_OK and not any("engage" in n for n in dry.get("notes") or []) and "cells_preflight" not in dry
    finally:
        _stubs.reset_package()
        _stubs.unstub_dist(d)


def test_census_lines(clean_state):
    import importlib
    pairfused = importlib.import_module("openfold3_opt.cells.pairfused"); stack = importlib.import_module("openfold3_opt.stack")
    st = pairfused.STATE
    st.update(installed=True, levers=["trimul_v4", "triatt_block", "pair_transition"], impl="fpf", core="flash_triattn",
              cells={"trimul_v4": {"cells": "8.0|*", "settings": None}, "pair_transition": {"cells": "safe", "settings": "safe:no_cell:c_z=128,n=4"}})   # triatt_block: nothing serves its trunk shape (installed all the same)
    lines = pairfused.census_lines()
    assert lines[0].startswith("[openfold3-opt/pairfused] LEVER name=trimul_v4 state=on served=0 fallback=0 impl=fpf_trimul_v4.generic") and "cells=" not in lines[0]
    assert lines[1].startswith("[openfold3-opt/pairfused] LEVER name=triatt_block state=on served=0 fallback=0 impl=fpf core=flash_triattn") and "cells=" not in lines[1]
    assert lines[2].startswith("[openfold3-opt/pairfused] LEVER name=pair_transition state=on served=0 fallback=0 impl=fpf") and lines[2].endswith(" cells=safe settings=safe:no_cell:c_z=128,n=4")
    assert all(stack._PROBES[l](stack._env.tree_home()) is True for l in ("trimul_v4", "triatt_block", "pair_transition"))   # installed = applied


def test_pinned_part_census_lines_are_unchanged(clean_state):
    import importlib
    pairfused = importlib.import_module("openfold3_opt.cells.pairfused")
    st = pairfused.STATE
    st.update(installed=True, levers=list(pairfused.LEVER_NAMES), impl="fpf", core="flash_triattn", refused={},
              cells={l: {"cells": "9.0|3.3", "settings": None} for l in pairfused.LEVER_NAMES})
    lines = pairfused.census_lines()
    assert len(lines) == 3 and not any(("cells=" in l or "settings=" in l or "state=skipped" in l) for l in lines)
