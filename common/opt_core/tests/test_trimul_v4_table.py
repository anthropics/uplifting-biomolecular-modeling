"""The fpf_trimul_v4 cell table's admission rule is ONE pure statement (opt_core.kernels.fpf_trimul_v4.table) — the device loader
(cells.cell_for), the row-block provider (mem.rowpair.trimul_fused.pointer_row) and a kit's pre-flight read it; importing it needs neither
torch nor triton."""
import importlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(HERE)


def test_table_rule_words_and_order():
    T = importlib.import_module("opt_core.kernels.fpf_trimul_v4.table")
    assert T.candidate_keys("9.0", "3.7") == (("9.0|3.7", "exact"), ("9.0|*", "generic-triton"))
    for st in ("CERTIFIED", "certified", "CERTIFIED_OP (v4.0.4 tma cell …)", "Certified-here"):
        assert T.admitted({"k1": {}, "k3": {}, "status": st}) is True, st
    for st in ("DEFAULT", "candidate", "SAFE", "", "PROVISIONAL"):
        assert T.admitted({"k1": {}, "k3": {}, "status": st}) is False, st
        assert T.admitted({"k1": {}, "k3": {}, "status": st}, allow_default_cells=True) is True, st               # a kit pre-flight's engineering switch
    assert T.admitted({"k1": {}}) is False and T.status_word({"k1": {}}) == "DEFAULT"          # no status word = DEFAULT: not served
    assert T.allow_default({}) is False and T.allow_default({T.ALLOW_DEFAULT_ENV: "1"}) is True and T.allow_default({T.ALLOW_DEFAULT_ENV: "0"}) is False
    assert T.KIT_TABLE_ENV == "FPF_TRIMUL_V4_CELLS" and T.CFG_ENV == "FPF_TRIMUL_V4_CFG" and os.path.isfile(T.CORE_TABLE_PATH)


def test_select_is_the_one_statement():
    T = importlib.import_module("opt_core.kernels.fpf_trimul_v4.table")
    ptr = {"k1": {"BM": 128, "BN": 128, "num_warps": 8, "num_stages": 1}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}}
    tma = {"k1": {"impl": "tma", "BM": 128, "BN": 32, "num_warps": 4, "num_stages": 3}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}}
    table = {"_doc": "x", "kit_keys": ["10.0|3.7"],
             "9.0|3.7": dict(tma, status="CERTIFIED_OP", overrides={"k1_C128_D128": {"impl": "tma", "BM": 64, "BN": 32, "num_warps": 4, "num_stages": 3}, "_note": "dropped"}),
             "9.0|*": dict(ptr, status="CERTIFIED_OP"), "10.3|*": dict(ptr, status="PROVISIONAL"), "10.0|3.7": dict(ptr, status="CERTIFIED_OP"), "10.0|*": dict(ptr, status="CERTIFIED_OP")}
    assert set(T.rows(table)) == {"9.0|3.7", "9.0|*", "10.3|*", "10.0|3.7", "10.0|*"} and T.table_ccs(table) == ("9.0", "10.0", "10.3") and T.kit_keys(table) == ("10.0|3.7",)
    sel = T.select(table, "9.0", "3.7")
    assert (sel.key, sel.kind, sel.source, sel.skipped, sel.kit_row) == ("9.0|3.7", "exact", "core", (), None)
    assert sel.cfg["overrides"] == {"k1_C128_D128": {"impl": "tma", "BM": 64, "BN": 32, "num_warps": 4, "num_stages": 3}}
    assert T.resolve_cfg(sel.cfg, 256, 256, False) == (tma["k1"], tma["k3"]) and T.resolve_cfg(sel.cfg, 128, 128, False)[0]["BM"] == 64
    assert T.resolve_cfg(sel.cfg, 128, 128, True)[0]["BM"] == 64                                   # bias falls to k1_C128_D128 when no _bias entry names the class
    nod = T.select(table, "9.0", "3.7", has_desc=False)                                             # no tensor-descriptor API: the tma row is skipped by name, the any-triton row serves
    assert (nod.key, nod.kind) == ("9.0|*", "generic-triton") and nod.skipped == ("9.0|3.7 needs the triton tensor-descriptor API",)
    assert T.select(table, "9.0", "3.9").key == "9.0|*" and T.select(table, "8.0", "3.7") == T.Selection(None, None, None, None, None, (), None)
    prov = T.select(table, "10.3", "3.7")                                                           # a row that is not served is skipped by name; nothing else on 10.3
    assert prov.key is None and prov.skipped == ("10.3|* status=PROVISIONAL (not served)",)
    # a kit's own table: evidence everywhere (kit_row), the routing decision only at a kit_keys part
    kit = {"9.0|3.7": dict(ptr, status="CERTIFIED_OP: the kit's H100 row"), "10.0|*": {"k1": {"BM": 1, "BN": 2, "num_warps": 3, "num_stages": 4}, "k3": ptr["k3"], "status": "CERTIFIED_OP: the kit's own B200 sweep"},
           "10.0|3.7": dict(ptr, status="TESTED elsewhere")}
    ks = T.select(table, "9.0", "3.7", kit_table=kit)
    assert (ks.key, ks.source, ks.kit_row) == ("9.0|3.7", "core", "9.0|3.7") and ks.cfg == sel.cfg          # not a kit_keys part: the package row serves; the kit's row is named
    assert T.select(table, "9.0", "3.9", kit_table=kit).kit_row == ("9.0|3.7" if False else None) or True  # (a kit without a 9.0|* row has no evidence for other tritons)
    assert T.select(table, "9.0", "3.9", kit_table=kit).kit_row is None
    kb = T.select(table, "10.0", "3.7", kit_table=kit)                                              # a kit_keys part the kit table serves (its unserved exact row skipped, its any-triton row serves)
    assert (kb.key, kb.kind, kb.source, kb.kit_row) == ("10.0|*", "generic-triton", "kit-table", "10.0|*") and kb.cfg["k1"] == {"BM": 1, "BN": 2, "num_warps": 3, "num_stages": 4}
    assert T.select(table, "10.0", "3.7").source == "core" and T.select(table, "10.0", "3.7", kit_table={}).source == "core"     # no kit table / an empty one: the package row
    assert T.select(table, "10.0", "3.6", kit_table=kit).source == "core"                            # 10.0|3.6 is not a kit_keys part
    s2, k1, k3 = T.cells_for(table, "10.0", "3.7", 128, 128, False, kit_table=kit)
    assert s2 == kb and k1 == {"BM": 1, "BN": 2, "num_warps": 3, "num_stages": 4} and k3 == ptr["k3"]
    assert T.cells_for(table, "8.0", "3.7", 128, 128, False)[1:] == (None, None)


def test_core_table_is_served_everywhere_it_has_rows():
    """The package's one table: every row admitted, every row resolves all three engine shape classes, kit_keys are rows of the table."""
    T = importlib.import_module("opt_core.kernels.fpf_trimul_v4.table")
    table = T.load_table()
    rows = T.rows(table)
    assert T.table_ccs(table) == ("8.0", "9.0", "10.0", "10.3") and all(cc + "|*" in rows for cc in T.table_ccs(table))
    for key, row in rows.items():
        assert T.admitted(row), key
        for C, D, hb in ((256, 256, False), (128, 128, True), (128, 128, False)):
            k1, k3 = T.resolve_cfg(T.row_cfg(row), C, D, hb)
            assert {"BM", "BN", "num_warps", "num_stages"} <= set(k1) and {"BM", "BN", "num_warps", "num_stages"} <= set(k3), (key, C, D, hb)
            assert 2 * D % k1["BN"] == 0 and C % k3["BN"] == 0, (key, C, D, hb, k1, k3)
        if key.endswith("|*"):
            assert row["k1"].get("impl", "ptr") != "tma", key                                       # any-triton rows are pointer cells (no descriptor API assumed)
    assert T.kit_keys(table) == ("9.0|3.3", "10.0|3.7", "10.3|3.7") and all(k in rows for k in T.kit_keys(table))          # exactly the three parts where two sweeps measured different cells, all rows of the table


def test_table_module_is_pure():
    """No torch / triton at import: a kit's dry-run pre-flight on a GPU-less host reads it."""
    code = "import sys; import opt_core.kernels.fpf_trimul_v4.table as T; print(int('torch' in sys.modules), int('triton' in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], cwd=CORE, env={**os.environ, "PYTHONPATH": CORE}, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.split() == ["0", "0"], (out.stdout, out.stderr[-300:])


def test_loader_and_provider_read_the_table_rule():
    """cells.cell_for and trimul_fused.pointer_row spell no status word and no key order of their own: the table module's statement decides."""
    for rel, uses in (("opt_core/kernels/fpf_trimul_v4/cells.py", "T.select("), ("opt_core/mem/rowpair/trimul_fused.py", "_v4table.admitted(")):
        src = open(os.path.join(CORE, rel), encoding="utf-8").read()
        assert 'startswith("CERTIFIED")' not in src and "SERVED_STATUS_PREFIX" not in src and uses in src, rel


def test_pointer_row_admission_is_the_tables(tmp_path):
    TF = importlib.import_module("opt_core.mem.rowpair.trimul_fused")
    ptr = {"k1": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 2}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}}
    tma_cfg = {"k1": {"impl": "tma", "BM": 128}, "k3": {"BM": 64}}
    served = {"9.0|*": dict(ptr, status="CERTIFIED_OP (pointer cell)")}
    assert TF.pointer_row(served, "9.0", "9.0|3.7", tma_cfg) == ({"k1": dict(ptr["k1"]), "k3": dict(ptr["k3"]), "overrides": {}}, "9.0|*")
    unserved = {"9.0|*": dict(ptr, status="candidate")}
    assert TF.pointer_row(unserved, "9.0", "9.0|3.7", tma_cfg) == (tma_cfg, "9.0|3.7")                       # not admitted: the resolved row as is
