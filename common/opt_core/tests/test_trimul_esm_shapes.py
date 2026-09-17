"""kernels.trimul_esm_shapes: the face is framework-free, the table is well-formed, selection is by measurement with named fallbacks,
and everything outside the envelope is refused BY NAME before a launch (no torch here)."""
import json
import os
import sys

import pytest

from opt_core.kernels import trimul_esm_shapes as S


def test_face_imports_no_framework():
    assert "torch" not in sys.modules and "triton" not in sys.modules


def test_table_is_well_formed():
    t = S.table()
    assert t["schema"].startswith("trimul_esm_shapes_tiles/")
    env = t["envelope"]
    assert [64, 64] in env["shapes"] and [384, 384] in env["shapes"] and env["batch"] == 1
    for cc, d in t["defaults"].items():
        assert cc in ("9.0", "8.0")
        assert set(d["any"]) >= {"k1", "k3", "form"} and "no_descriptor_k1" in d and not d["no_descriptor_k1"].get("tma")
        if cc == "8.0":
            assert not d["any"]["k1"].get("tma")
    for key, c in t["cells"].items():
        cc, io, C, H, nb, d = key.split("|")
        assert cc in ("9.0", "8.0") and io in S.IO_WORDS and C[0] == "C" and H[0] == "H" and nb.startswith("N<=") and d in ("out", "in")
        Cv, Hv = int(C[1:]), int(H[1:])
        cell = c["cell"]
        assert (2 * Hv) % int(cell["k1"]["BN"]) == 0, key
        assert Cv % int(cell["k3"]["BN"]) == 0, key
        S.chunk_plan(Cv, cell.get("ck")); S.chunk_plan(Hv, cell.get("chk"))
        if cc == "8.0":
            assert not cell["k1"].get("tma"), key
        assert c.get("ms") and c.get("evidence"), key                     # a cell is a measurement, never a guess


def test_key_grammar_and_words():
    assert S.cell_key((9, 0), "bf16", 384, 384, 800, "outgoing") == "9.0|bf16|C384|H384|N<=800|out"
    assert S.cell_key("80", "fp32", 64, 128, 400, "in") == "8.0|fp32|C64|H128|N<=400|in"
    assert S.io_word("torch.float32") == "fp32" and S.io_word("tf32") == "fp32" and S.io_word("bfloat16") == "bf16"
    with pytest.raises(S.Unsupported):
        S.io_word("float16")


def test_chunk_plan():
    assert S.chunk_plan(384) == (128, 3) and S.chunk_plan(256) == (256, 1) and S.chunk_plan(256, 128) == (128, 2) and S.chunk_plan(64) == (64, 1)
    assert S.chunk_plan(512) == (128, 4)
    for bad in ((384, 96), (100, None), (1024, None), (384, 512)):
        with pytest.raises(S.Unsupported):
            S.chunk_plan(*bad)


def test_outside_the_envelope_is_refused_by_name():
    ok, why = S.supported((9, 0), "bf16", 320, 320, 400)
    assert not ok and why.startswith("shape:")
    ok, why = S.supported((9, 0), "bf16", 128, 128, 400, batch=4)
    assert not ok and why.startswith("batch:")
    ok, why = S.supported((7, 5), "bf16", 128, 128, 400)
    assert not ok and why.startswith("cc:")
    ok, why = S.supported((9, 0), "fp16", 128, 128, 400)
    assert not ok and why.startswith("io:")
    ok, why = S.supported((9, 0), "bf16", 128, 128, 4)
    assert not ok and why.startswith("n<")


def test_selection_prefers_measured_cells_and_names_defaults():
    for cc in ("9.0", "8.0"):
        sel = S.select_cell(cc, "bf16", 64, 128, 100000, "out")
        assert sel["cell"]["k1"] and sel["cell"]["k3"]
        assert sel["measured"] == (sel["size"] is not None)
        if sel["measured"]:
            assert sel["beyond_measured"]
    sel = S.select_cell("9.0", "bf16", 384, 384, 700, "out", descriptor_api=False)
    assert not sel["cell"]["k1"].get("tma") and "k1_ptr" not in sel["cell"]
    txt = S.describe_cell(sel)
    assert "k1[ptr" in txt and "measured pointer cell" in txt
    both = S.select_cell("9.0", "bf16", 384, 384, 700, "out", descriptor_api=True)
    assert both["cell"]["k1"].get("tma") and "k1_ptr" not in both["cell"]


def test_every_descriptor_cell_carries_a_measured_pointer_k1():
    for key, c in S.table()["cells"].items():
        if c["cell"]["k1"].get("tma"):
            alt = c["cell"].get("k1_ptr")
            assert alt and not alt.get("tma") and c["ms"].get("k1_ptr"), key


def test_measured_cells_cover_the_campaign_shapes():
    """Every tabled shape has out+in cells at 400/800/1200 on both cards for bf16 (the table's purpose); fp32 io at least where fp32 trunks live."""
    t = S.table()
    missing = []
    for cc in ("9.0", "8.0"):
        for C, H in t["envelope"]["shapes"]:
            for n in (400, 800, 1200):
                for d in ("out", "in"):
                    k = S.cell_key(cc, "bf16", C, H, n, d)
                    if k not in t["cells"]:
                        missing.append(k)
    allowed = set(t.get("coverage", {}).get("open", []))
    assert not (set(missing) - allowed), sorted(set(missing) - allowed)


def test_no_environment_reads_in_the_package():
    here = os.path.dirname(S.__file__)
    for fn in os.listdir(here):
        if fn.endswith(".py"):
            src = open(os.path.join(here, fn), encoding="utf-8").read()
            assert "os.environ" not in src and "getenv" not in src, fn
