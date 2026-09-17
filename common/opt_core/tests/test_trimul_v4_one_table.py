"""fpf_trimul_v4 serves every engine from ONE cells table (opt_core/kernels/fpf_trimul_v4/table.json).

The hard gate: every (engine, card, triton) launch-cell selection an engine's own table admitted (tests/fixtures/fpf_trimul_v4_land_selections.json:
132 selections over 10 kit tables x {A100, H100, B200, B300} x triton {3.3, 3.6, 3.7, any other}; a table's cards are the ones it has rows for) resolves, through the one table and the kit's shape class
(C, D, bias), to the IDENTICAL k1/k3 cells on every MEASURED path (the engine's pinned-stack triton on a card it has a recorded config for, and the kits'
own Blackwell sweeps) — or to a SUCCESSOR of the measured cell the row names (``successors``: a cell change that kept the output bytes, e.g. the H100
(128, 128) descriptor cells); the other selections that differ are exactly the fixture's unmeasured ones (a triton the engine's image never runs).

The loader (cells.cell_for): a capability with rows serves the table by name; the kit's own table (FPF_TRIMUL_V4_CELLS) is named as evidence and serves
its own cells only at the kit_keys parts; a capability without rows serves the SAFE cell (the capability's row or the any-capability '*' row), engaged
and NAMED once; a capability with neither answers None (generic refuses `no-cell` by name); a launch cell that fails to BUILD switches the process to the
SAFE cell (named), and the SAFE cell failing turns the lever off BY NAME (`cell=none:<why>`) and raises the cannot-run refusal (the caller's mode refuses;
no stock route). CPU-only: device queries are stubbed, nothing is launched."""
import glob
import hashlib
import importlib
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "fpf_trimul_v4_land_selections.json")
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))          # model-opt-release/: the kits' own tables live in the kit trees beside common/
T = importlib.import_module("opt_core.kernels.fpf_trimul_v4.table")
ROW_KEY = re.compile(r"^\d+\.\d+\|(\d+\.\d+|\*)$")


def _kit_tables():
    """Every fpf_trimul_v4 cell table a kit tree of this release carries, keyed by the first 16 hex digits of the sha256 of its bytes (the fixture's
    'table' id): found by CONTENT — a JSON object with at least one '<cc>|<triton>' row holding k1/k3 — never by an engine's name or path."""
    found = {}
    for opt in glob.glob(os.path.join(RELEASE, "*", "opt")):
        if os.path.dirname(opt) == os.path.dirname(os.path.dirname(HERE)):
            continue                                                                 # common/ itself
        for root, dirs, files in os.walk(opt):
            dirs[:] = [d for d in dirs if d not in ("stock", "src", "__pycache__", ".git", "build", "dist", "tests") and not d.endswith(".egg-info")]
            for f in files:
                if not f.endswith(".json"):
                    continue
                path = os.path.join(root, f)
                if os.path.getsize(path) > 512 * 1024:
                    continue
                raw = open(path, "rb").read()
                try:
                    doc = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(doc, dict) and any(ROW_KEY.match(str(k)) and isinstance(v, dict) and "k1" in v and "k3" in v for k, v in doc.items()):
                    found[hashlib.sha256(raw).hexdigest()[:16]] = doc
    return found


def _triton(word):
    return "3.9" if word == "other" else word          # 'other' = a triton with no exact row anywhere


def _succeeded(row, r, k1, k3):
    """Whether the cells (k1, k3) the table serves differ from the kit's land cells only where the row names the land cell as the one its current cell
    replaced (``successors``, per override key: a change that kept the output bytes)."""
    succ = row.get("successors") or {}
    sfx = "_bias" if r["has_bias"] else ""
    old1 = succ.get("k1_C%d_D%d%s" % (r["C"], r["D"], sfx)); old3 = succ.get("k3_C%d_D%d%s" % (r["C"], r["D"], sfx))
    return (k1 == r["land_k1"] or old1 == r["land_k1"]) and (k3 == r["land_k3"] or old3 == r["land_k3"])


def test_every_land_selection_resolves_through_the_one_table_and_measured_paths_are_identical():
    fx = json.load(open(FIXTURE))
    table, kits = T.load_table(), _kit_tables()
    rows = fx["rows"]
    assert len(rows) == 132 and len({r["table"] for r in rows}) == 10 and set(fx["tables"]) == {r["table"] for r in rows}
    changed, measured, judged = [], 0, 0
    for r in rows:
        kit = kits.get(r["table"])                                                          # the kit's own table when this tree carries it (a sparse checkout may not)
        if kit is None and r["core_source"] == "kit-table":
            continue                                                                        # a kit_keys selection cannot be judged without the kit's table
        judged += 1
        sel, k1, k3 = T.cells_for(table, r["cc"], _triton(r["triton"]), r["C"], r["D"], r["has_bias"], kit_table=kit, has_desc=r["has_desc"])
        tag = "%s %s|%s" % (r["table"], r["cc"], r["triton"])
        assert sel.key == r["core_row"] and sel.source == r["core_source"], (tag, sel)
        assert (k1, k3) == (r["core_k1"], r["core_k3"]), (tag, k1, k3)                  # the fixture records what the table serves: it is regenerated with the table, never by hand
        same = (k1, k3) == (r["land_k1"], r["land_k3"])
        assert same == r["identical"], tag
        if r["measured"]:
            measured += 1
            assert same or _succeeded(table[sel.key], r, k1, k3), "MEASURED path changed: %s land %s/%s -> %s/%s" % (tag, r["land_k1"], r["land_k3"], k1, k3)
        elif not same:
            changed.append(tag)
    assert sum(r["measured"] for r in rows) == 32 and sum(not r["identical"] for r in rows) == 34
    succeeded = [r for r in rows if r["measured"] and not r["identical"]]                 # measured selections the table serves with a SUCCESSOR cell (row 'successors': equal output bytes):
    assert len(succeeded) == 3 and all((r["cc"], r["triton"], r["C"], r["D"], r["has_bias"]) == ("9.0", "3.7", 128, 128, False) for r in succeeded)   # the H100 (128,128) descriptor cells
    if judged == len(rows):
        assert measured == 32 and len(changed) == 31, (measured, len(changed))
    by_table = {}
    for r in rows:
        by_table.setdefault(r["table"], []).append(r["identical"])
    assert sum(all(v) for v in by_table.values()) == 4                                      # four kit tables change nothing at all; the others only unmeasured tritons
    assert {(r["cc"], r["triton"]) for r in rows if r["core_source"] == "kit-table"} == {("9.0", "3.3"), ("10.0", "3.7"), ("10.3", "3.7")}     # kit tables serve only at the three kit_keys parts


def test_the_kit_keys_parts_serve_each_kits_own_cells_and_nothing_else():
    fx = json.load(open(FIXTURE)); table, kits = T.load_table(), _kit_tables()
    assert T.kit_keys(table) == ("9.0|3.3", "10.0|3.7", "10.3|3.7")
    own = [r for r in fx["rows"] if r["triton"] == "3.7" and r["cc"] in ("10.0", "10.3")]                       # the Blackwell triton-3.7 parts: the eight tables with Blackwell rows
    assert len({r["table"] for r in own}) == 8 and len(own) == 14 and all(r["identical"] and r["core_source"] == "kit-table" for r in own)   # 8 tables carry Blackwell rows (6 both parts, 2 one)
    h33 = [r for r in fx["rows"] if (r["cc"], r["triton"]) == ("9.0", "3.3")]                                  # the H100 / triton-3.3 part: every table (all ten have 9.0 rows) serves its own row
    assert len(h33) == 10 and all(r["core_source"] == "kit-table" and r["identical"] for r in h33)
    for r in own + h33:
        kit = kits.get(r["table"])
        if kit is None:
            continue
        sel, k1, k3 = T.cells_for(table, r["cc"], r["triton"], r["C"], r["D"], r["has_bias"], kit_table=kit, has_desc=r["has_desc"])
        assert sel.source == "kit-table" and (k1, k3) == (r["land_k1"], r["land_k3"]) and sel.kit_row == r["land_row"], (r["table"], r["cc"], r["triton"])
        assert T.cells_for(table, r["cc"], "3.9", r["C"], r["D"], r["has_bias"], kit_table=kit)[0].source == "core"          # other tritons: the row's cells
    row = T.cells_for(table, "10.0", "3.7", 128, 128, False)                                                                            # no kit table (a new kit): the row
    assert row[0].source == "core" and row[1] == {"BM": 128, "BN": 64, "num_warps": 4, "num_stages": 1} and row[2] == {"BM": 64, "BN": 64, "num_warps": 8, "num_stages": 1}
    assert T.cells_for(table, "10.0", "3.7", 128, 128, False, kit_table={})[0].source == "core"                                          # an empty kit table: the row
    assert any(r["measured"] and (r["cc"], r["C"], r["has_bias"]) == ("10.0", 128, False) and (row[1], row[2]) == (r["land_k1"], r["land_k3"]) for r in own)   # the row's B200 cell IS one kit's measured cell


# ------------------------------------------------------------------------------------------------------------ the device loader (CPU: stubbed queries)
torch = pytest.importorskip("torch")
pytest.importorskip("triton")                     # kernels.py decorates its kernels at import (nothing is compiled or launched here)


@pytest.fixture
def loader(monkeypatch):
    CELLS = importlib.import_module("opt_core.kernels.fpf_trimul_v4.cells")
    SAFE = importlib.import_module("opt_core.kernels.safe_settings")
    state = {"cc": (9, 0), "mm": "3.7", "desc": True}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: state["cc"])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Stub GPU")
    monkeypatch.setattr(CELLS, "triton_mm", lambda: state["mm"])
    monkeypatch.setattr(CELLS, "has_desc", lambda: state["desc"])
    monkeypatch.setattr(CELLS, "_CELL_CACHE", {}); monkeypatch.setattr(CELLS, "INFO", {}); monkeypatch.setattr(CELLS, "_OFF", {"why": None})
    monkeypatch.setattr(CELLS, "KIT_TABLE_PATH", None); monkeypatch.setattr(CELLS, "_CFG_OVERRIDE", None); monkeypatch.setattr(CELLS, "TABLE_PATH", T.CORE_TABLE_PATH)
    net = SAFE.SafeNet(CELLS.SAFE_LEVER, refused=lambda msg: importlib.import_module("opt_core.kernels.fpf_trimul_v4.generic").TrimulUnsupported("refused", msg))
    monkeypatch.setattr(type(CELLS.NET), "_n", net)
    return CELLS, state


def test_loader_serves_the_table_by_name(loader, capfd):
    CELLS, state = loader
    cfg = CELLS.cell_for("cuda:0")
    sel = T.select(T.load_table(), "9.0", "3.7")
    assert cfg == sel.cfg and CELLS.INFO["cuda:0"]["key"] == "9.0|3.7" and CELLS.INFO["cuda:0"]["served_by"] == "core" and CELLS.cell_word("cuda:0") == "cell=9.0|3.7"
    err = capfd.readouterr().err
    assert err.count("[fpf_trimul_v4] cell on cuda:0: table.json[9.0|3.7] (exact match, source=core") == 1 and "SERVE" in err
    assert CELLS.cell_for("cuda:0") is cfg and capfd.readouterr().err == ""                                   # decided once per device
    state.update(cc=(9, 0), mm="3.9"); CELLS._CELL_CACHE.clear()
    CELLS.cell_for("cuda:0")                                                                                   # a triton with no row of its own: the any-triton cells, said so
    assert CELLS.INFO["cuda:0"]["key"] == "9.0|*" and "triton 3.9 has no row of its own on cc 9.0" in capfd.readouterr().err
    state.update(mm="3.3", desc=False); CELLS._CELL_CACHE.clear()
    CELLS.cell_for("cuda:0")
    assert CELLS.INFO["cuda:0"]["key"] == "9.0|3.3" and not CELLS.NET.on


def test_loader_names_the_kit_table_and_serves_it_only_at_a_kit_keys_part(loader, tmp_path, monkeypatch, capfd):
    CELLS, state = loader
    kit = {"9.0|3.7": {"k1": {"BM": 1, "BN": 1, "num_warps": 1, "num_stages": 1}, "k3": {"BM": 2, "BN": 2, "num_warps": 2, "num_stages": 2}, "status": "CERTIFIED_OP: kit H100 row"},
           "10.0|*": {"k1": {"BM": 128, "BN": 32, "num_warps": 4, "num_stages": 2}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}, "status": "CERTIFIED_OP: kit B200 sweep"}}
    p = tmp_path / "kit" / "cells.json"; p.parent.mkdir(); p.write_text(json.dumps(kit)); monkeypatch.setattr(CELLS, "KIT_TABLE_PATH", str(p))
    cfg = CELLS.cell_for("cuda:0")                                                                             # H100 triton 3.7: the package row serves; the kit's row is evidence, named
    assert cfg == T.select(T.load_table(), "9.0", "3.7").cfg and CELLS.INFO["cuda:0"]["served_by"] == "core" and CELLS.INFO["cuda:0"]["kit_row"] == "9.0|3.7"
    err = capfd.readouterr().err
    assert "source=core" in err and "kit table kit/cells.json: certified row 9.0|3.7" in err and CELLS.cell_word("cuda:0") == "cell=9.0|3.7"
    state.update(mm="3.9"); CELLS._CELL_CACHE.clear()
    CELLS.cell_for("cuda:0")
    assert CELLS.INFO["cuda:0"]["kit_row"] is None and "kit table kit/cells.json: no row for this part (uncertified by this kit)" in capfd.readouterr().err
    state.update(cc=(10, 0), mm="3.7"); CELLS._CELL_CACHE.clear()                                             # B200 triton 3.7 is a kit_keys part: the kit's own cells serve, named
    cfg = CELLS.cell_for("cuda:0")
    assert T.resolve_cfg(cfg, 128, 128, False) == (kit["10.0|*"]["k1"], kit["10.0|*"]["k3"])
    assert CELLS.INFO["cuda:0"]["served_by"] == "kit-table" and CELLS.cell_word("cuda:0") == "cell=10.0|*:kit-table" and "source=kit-table" in capfd.readouterr().err
    monkeypatch.setattr(CELLS, "KIT_TABLE_PATH", str(tmp_path / "gone.json")); CELLS._CELL_CACHE.clear()    # an unreadable kit table is named, the package row serves
    CELLS.cell_for("cuda:0")
    assert CELLS.INFO["cuda:0"]["served_by"] == "core" and "unreadable (FileNotFoundError" in capfd.readouterr().err


def test_a_capability_without_rows_serves_the_safe_cell_named_once(loader, tmp_path, monkeypatch, capfd):
    CELLS, state = loader
    SAFE = importlib.import_module("opt_core.kernels.safe_settings")
    table = {k: v for k, v in T.load_table().items() if not k.startswith("9.0|")}                              # a table with no H100 rows: H100 is an uncertified capability here
    p = tmp_path / "table.json"; p.write_text(json.dumps(table)); monkeypatch.setattr(CELLS, "TABLE_PATH", str(p))
    cfg = CELLS.cell_for("cuda:0")
    srow = SAFE.safe_row(CELLS.SAFE_LEVER, "9.0")
    assert cfg == {"k1": srow["settings"]["k1"], "k3": srow["settings"]["k3"], "overrides": {}} and CELLS.NET.on and CELLS.NET.word() == "safe:no_cell:9.0|3.7"
    err = capfd.readouterr().err
    assert err.count("[opt_core/pair_fused:trimul] safe settings served (no_cell:9.0|3.7, cc 9.0, triton 3.7)") == 1 and "status=SAFE;" in err
    assert CELLS.cell_word("cuda:0") == "cell=safe(uncertified 9.0|3.7)" and CELLS.INFO["cuda:0"]["served_by"] == "safe"
    for cc in ((8, 6), (8, 9), (12, 0)):                                                                      # capabilities no engine certified: the any-capability SAFE row, engaged by name
        state.update(cc=cc); CELLS._CELL_CACHE.clear()
        dev = "cuda:%d" % cc[1]
        cfg = CELLS.cell_for(dev)
        star = SAFE.SAFE_ROWS[CELLS.SAFE_LEVER]["*"]
        assert SAFE.safe_row(CELLS.SAFE_LEVER, cc) is star and cfg == {"k1": star["settings"]["k1"], "k3": star["settings"]["k3"], "overrides": {}}
        assert CELLS.INFO[dev]["served_by"] == "safe" and CELLS.cell_word(dev) == "cell=safe(uncertified %d.%d|3.7)" % cc
        assert "status=SAFE:inferred; no table.json row for cc %d.%d" % cc in capfd.readouterr().err
    monkeypatch.setitem(SAFE.SAFE_ROWS, CELLS.SAFE_LEVER, {k: v for k, v in SAFE.SAFE_ROWS[CELLS.SAFE_LEVER].items() if k != "*"})
    state.update(cc=(7, 5)); CELLS._CELL_CACHE.clear()                                                       # neither a row nor a SAFE cell: None, refused by name downstream
    assert CELLS.cell_for("cuda:7") is None and CELLS.cell_word("cuda:7") == "cell=no-cell"
    assert "no table.json row and no safe settings for cc 7.5 (triton 3.7): refused by name (no-cell)" in capfd.readouterr().err


def test_every_pair_fused_lever_has_safe_settings_on_any_capability():
    SAFE = importlib.import_module("opt_core.kernels.safe_settings")
    for lever in ("pair_fused:trimul", "pair_fused:prologue", "pair_fused:epilogue", "pair_fused:transition"):
        rows = SAFE.SAFE_ROWS[lever]
        assert set(rows) == {"8.0", "9.0", "10.0", "10.3", "*"}, lever
        for cc in ("8.6", "8.9", "12.0", (8, 6), "11.0"):
            r = SAFE.safe_row(lever, cc)
            assert r is rows["*"] and r["status"] == "SAFE:inferred" and r.get("when") is None and r.get("refusal") is None, (lever, cc)
            assert SAFE.safe_settings_of(lever, cc)() == rows["*"]["settings"]
        for cc in ("8.0", "9.0", "10.0", "10.3"):
            assert SAFE.safe_row(lever, cc) is rows[cc], (lever, cc)                                          # the certified capabilities' rows are untouched by the '*' row
    star = SAFE.SAFE_ROWS["pair_fused:trimul"]["*"]["settings"]
    assert star == {"k1": {"BM": 64, "BN": 32, "num_warps": 4, "num_stages": 1}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}} and "impl" not in star["k1"]


def test_build_failure_serves_the_safe_cell_and_the_safe_cell_failing_turns_the_lever_off_by_name(loader, capfd):
    CELLS, _ = loader
    G = importlib.import_module("opt_core.kernels.fpf_trimul_v4.generic")
    SAFE = importlib.import_module("opt_core.kernels.safe_settings")
    tuned = CELLS.cell_for("cuda:0"); capfd.readouterr()
    safe = CELLS.safe_cfg("9.0")
    seen = []
    def build(cfg):
        seen.append(cfg)
        if cfg == tuned:
            raise RuntimeError("PassManager::run failed (a triton compile failure of the tuned cell)")
        return "served-with-%s" % ("safe" if cfg == safe else "other")
    assert CELLS.run(build, tuned, "cuda:0") == "served-with-safe" and seen == [tuned, safe]
    assert CELLS.NET.on and CELLS.cell_for("cuda:0") == safe and CELLS.INFO["cuda:0"]["served_by"] == "safe" and CELLS.cell_word("cuda:0") == "cell=safe(build_failed:RuntimeError)"
    assert capfd.readouterr().err.count("[opt_core/pair_fused:trimul] safe settings served (build_failed:RuntimeError, cc 9.0, triton 3.7)") == 1
    assert CELLS.run(lambda cfg: ("ok", cfg), None, "cuda:0") == ("ok", safe)                                   # the process stays on the SAFE cell
    def never(cfg):
        raise RuntimeError("PassManager::run failed (the SAFE cell cannot build either)")
    with pytest.raises(G.TrimulUnsupported) as e:
        CELLS.run(never, safe, "cuda:0")
    assert e.value.reason == "none" and e.value.cannot_run and CELLS.off_word() == "none:RuntimeError"
    assert CELLS.cell_for("cuda:0") is None and CELLS.cell_word("cuda:0") == "cell=none:RuntimeError" and CELLS.INFO["cuda:0"]["source"].startswith("none: RuntimeError")
    assert capfd.readouterr().err.count("[fpf_trimul_v4] cell=none: RuntimeError") == 1
    with pytest.raises(RuntimeError):                                                                          # not a build failure: propagates unchanged (the caller's kernel-error route)
        CELLS.run(lambda cfg: (_ for _ in ()).throw(RuntimeError("an unrelated runtime error")), safe, "cuda:0")
    assert not SAFE.is_build_failure(RuntimeError("an unrelated runtime error"))


def test_generic_refuses_by_the_levers_words(loader, monkeypatch):
    CELLS, _ = loader
    G = importlib.import_module("opt_core.kernels.fpf_trimul_v4.generic")
    z = torch.zeros(128, 128, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    assert G.supported(z) == (True, "ok")
    monkeypatch.setitem(G._PROBES, (str(z.device), 128, 128, False, False), {"ok": False, "line": "warm probe (...) -> REFUSED"})
    assert G.supported(z, d_hidden=128) == (False, "probe-failed")                                             # the class the warm probe refused: the lever cannot run, by that name
    with pytest.raises(G.TrimulUnsupported) as e:
        G.trimul(z, None, direction="outgoing", weights={"C": 128, "D": 128, "has_bias": False})
    assert e.value.reason == "probe-failed" and e.value.cannot_run and G.COUNTS["unsupported"]["probe-failed"] >= 1
    CELLS.lever_off("RuntimeError (the SAFE cell failed to build after build_failed:RuntimeError)")
    assert G.supported(z) == (False, "none:RuntimeError")
    assert not G.TrimulUnsupported("c", "").cannot_run and not G.TrimulUnsupported("no-cell", "").cannot_run and G.TrimulUnsupported("none:X", "").cannot_run
    assert (G.SAME_CLASS_RMS, G.SAME_CLASS_MAX) == (1.25, 2.5) and G.PROBE_N >= G.N_MIN
