"""L7's cell table (`genie3_opt/fpf_cells.json`, exported as FPF_TRIMUL_V4_CELLS for the shared core's fpf_trimul_v4): one admitted row per served
compute capability, keyed as broadly as the kernel serves — `9.0|*` (H100 / H200 on any triton: the pinned stack's triton 3.3 included) and `8.0|*`
(A100) — each a base k1/k3 tile pair; the A100 row also carries speed-only `overrides` for the C128/D128 class, which is this model's pair track
(c_z = c_hidden = 128). No exact `<cc>|<triton>` row exists: such a row is only for a triton line measured to need other tiles. Resolution is the
core's: the row by exact key then `<cc>|*` (kernels/fpf_trimul_v4/cells.py cell_for), the launch tiles by kernels/fpf_trimul_v4/table.py
resolve_cfg — loaded here from that file alone (a pure module: json, os, typing), without importing the kernels package (triton, absent on the CPU stack)."""
import importlib.util
import json
import os

import opt_core

from genie3_opt import trimul

CORE_TABLE = os.path.join(os.path.dirname(os.path.abspath(opt_core.__file__)), "kernels", "fpf_trimul_v4", "table.py")
TILE_KEYS = {"BM", "BN", "num_warps", "num_stages"}


def _table():
    with open(trimul.CELLS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _core_resolve_cfg():
    """The core's resolve_cfg function object (the one grammar for `overrides`), from its own module loaded by path."""
    spec = importlib.util.spec_from_file_location("_fpf_trimul_v4_table", CORE_TABLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                             # table.py imports json, os, typing only — no triton
    return mod.resolve_cfg


def _row_for(tab, cc, triton_mm):
    """The core's key order (cells.py cell_for): exact `<cc>|<triton>` first, then `<cc>|*`; rows whose status admits them only; None = no row (the stock forward)."""
    for key in (f"{cc}|{triton_mm}", f"{cc}|*"):
        row = tab.get(key)
        if row is not None and str(row.get("status", "")).upper().startswith("CERTIFIED_OP"):   # the core's status rule (cells.py cell_for) on this table's status word
            return key, row
    return None, None


H100_TILES = ({"BM": 128, "BN": 128, "num_warps": 8, "num_stages": 1}, {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1})   # (k1, k3) served on cc 9.0 for every (C, D) and every triton


def test_the_table_has_one_admitted_row_per_served_capability():
    tab = _table()
    assert [k for k in tab if not k.startswith("_")] == ["9.0|*", "8.0|*"]                 # cc-wide rows only: no exact <cc>|<triton> row
    for key, row in tab.items():
        if key.startswith("_"):
            continue
        assert str(row["status"]).startswith("CERTIFIED_OP"), key
        for kern in ("k1", "k3"):
            assert set(row[kern]) == TILE_KEYS and all(isinstance(v, int) for v in row[kern].values()), (key, kern)
        for name, ov in (row.get("overrides") or {}).items():
            assert name.split("_")[0] in ("k1", "k3") and set(ov) == TILE_KEYS, (key, name)
    assert "overrides" not in tab["9.0|*"]                                                # the H100 row serves its base tiles for every (C, D)
    assert set(tab["8.0|*"]["overrides"]) == {"k1_C128_D128", "k3_C128_D128"}


def test_cc90_resolves_to_the_wildcard_row_with_the_same_tiles_on_every_triton():
    tab, resolve_cfg = _table(), _core_resolve_cfg()
    for triton_mm in ("3.3", "3.7", "3.4", "4.0"):                               # the pinned stack (torch 2.7.1 / triton 3.3.1), triton 3.7.1, and lines never measured: all served by 9.0|*
        key, row = _row_for(tab, "9.0", triton_mm)
        assert key == "9.0|*", triton_mm
        for C, D, has_bias in ((128, 128, True), (256, 256, False), (128, 256, True)):
            assert resolve_cfg(row, C, D, has_bias) == (row["k1"], row["k3"]) == H100_TILES, (triton_mm, C, D)   # the tiles the pinned stack is served


def test_the_a100_row_resolves_to_its_c128_d128_overrides():
    tab, resolve_cfg = _table(), _core_resolve_cfg()
    key, row = _row_for(tab, "8.0", "3.3")                                    # the pinned triton on an A100: no exact row, the generic one
    assert key == "8.0|*"
    k1, k3 = resolve_cfg(row, 128, 128, True)                                 # this model's pair track (C=128, D=128, biases passed)
    assert (k1, k3) == (row["overrides"]["k1_C128_D128"], row["overrides"]["k3_C128_D128"])
    assert (k1, k3) == ({"BM": 128, "BN": 32, "num_warps": 4, "num_stages": 1}, {"BM": 128, "BN": 64, "num_warps": 4, "num_stages": 1})
    assert resolve_cfg(row, 256, 256, False) == (row["k1"], row["k3"])       # another (C, D) class: the base tiles


def test_a_card_without_a_row_resolves_to_none():
    tab = _table()
    for cc in ("8.9", "8.6", "7.0", "10.0"):
        assert _row_for(tab, cc, "3.3") == (None, None), cc                  # L4 / A10 / V100 / B200: no row — the module's own forward, counted `no_cell:…`
