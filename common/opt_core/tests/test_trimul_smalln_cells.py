"""0.5.212.2 — TriMul below the former 101-token floor.

The small-N measurement (four H100 SXM columns) moved the fast / big words of the
cc-9.0 c_z-128 classes below 101 tokens from the library op to row v4 BY DATA: cells `9.0|{f32z_bf16,bf16}|C128|H128|N<=100|{out,in}|fwd`,
new floor cells `N<=1`, and `rows.v4.admits.n_min_cells`.  Row v4's token floor is read from the table (kernels.trimul.v4_n_min); the
fpf_trimul_v4 face takes `n_min=` (4.4.5).  These tests pin: the table answer, the floor per class (unchanged word for every other class), the
words select() gives at 1 / 2 / 24 / 100 / 101 tokens, the exact word unchanged, and the byte identity of EVERY OTHER cell (inertness at and
above the old floor is proven by the select() grid artifact of the release; this pin keeps the table from drifting under it).  CPU only."""
import hashlib
import inspect
import json

import pytest

from opt_core.kernels import trimul as KT

VOUCHED = ("H100:2.13.0+cu130/3.7.1/cueq0.11.1", "H100:2.12.0+cu130/3.7.0/cueq0.10.0", "H100:2.10.0+cu128/3.6.0/cueq0.10.0", "H100:2.7.1+cu128/3.3.1/cueq0.10.0")
CLASSES = (("f32z_bf16", "fp32"), ("bf16", None))          # (precision word, residency argument for a bf16 dtype word)
WIDTHS = (128,)                                              # C128: the c_z-128 pair stacks (four columns); the c_z-256 class is not moved by this release
COL_213 = "H100:2.13.0+cu130/3.7.1/cueq0.11.1"
OTHER_CELLS_SHA256 = "4c6d6bb635cfba4a716f14214f80d81d945d50139a93d22cb071690c3dede4de"       # sha256(json.dumps({every cell NOT in smalln_cert.changed_cells}, sort_keys=True)) at 0.5.212.2 == at 0.5.212.1
SIZES = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 100]


def test_the_smalln_cells_are_the_certified_answer():
    T = KT.table()
    sc = T["smalln_cert"]
    assert sc["version"] == "0.5.212.2" and sc["sizes"] == SIZES and sorted(sc["columns"]) == sorted(VOUCHED)
    for prec, _ in CLASSES:
        for C in WIDTHS:
            cols = VOUCHED if C == 128 else (COL_213,)
            for dw in ("out", "in"):
                c = T["cells"]["9.0|%s|C%d|H%d|N<=100|%s|fwd" % (prec, C, C, dw)]
                assert (c["fast"], c["big"], c["exact"]) == ("v4", "v4", "cueq"), (prec, C, dw, c["fast"], c["big"], c["exact"])
                assert c["fast_order"][0] == "v4" and c["measured"] is True and c["measured_sizes"] == SIZES and c["ref_stack"] in cols
                assert all(c["fast_per_stack"][st] == "v4" and c["big_per_stack"][st] == "v4" and c["exact_per_stack"][st] == "cueq" for st in cols)
                v = c["smalln_cert"]["vouched_on"]
                assert sorted(v) == sorted(cols)
                for st, rec in v.items():                                        # the gates, per column: faster than the yardstick at every size, inside its error class
                    assert set(rec["per_size"]) == {str(n) for n in SIZES}, (prec, C, dw, st)
                    if C == 128 and prec == "f32z_bf16":
                        assert rec["x_min"] > 1.03 and rec["err_ratio_max"] <= 1.5, (st, rec["x_min"], rec["err_ratio_max"])
                    elif C == 128:
                        assert rec["x_vs_module_math_min"] > 1.03 and rec["x_vs_library_min"] > 1.03 and rec["acc_ratio_max"] <= 1.0, (st, rec)
                    else:
                        assert rec["x_min"] > 1.03 and rec["acc_ratio_max"] <= (1.5 if prec == "f32z_bf16" else 1.0), (st, rec)
                        if prec == "bf16": assert rec["x_vs_library_min"] > 1.03, (st, rec)
                c1 = T["cells"]["9.0|%s|C%d|H%d|N<=1|%s|fwd" % (prec, C, C, dw)]     # the floor cell: measured at N = 1 (v4 refuses N<2, native n<16): the stock op on every word
                assert (c1["fast"], c1["big"], c1["exact"], c1["measured"], c1["measured_sizes"], c1["fast_order"][0]) == ("cueq", "cueq", "cueq", True, [1], "cueq")
                assert "v4" in c1["kit_floor"]["refused"] and "v4" not in c1["ms"]
    assert T["rows"]["v4"]["admits"]["n_min"] == 101
    assert T["rows"]["v4"]["admits"]["n_min_cells"] == {"9.0|%s|C%d|H%d" % (p, C, C): 2 for p in ("f32z_bf16", "bf16") for C in WIDTHS}


def test_every_other_cell_is_byte_identical_to_0_5_212_1():
    T = KT.table()
    skip = set(T["smalln_cert"]["changed_cells"])
    assert skip == {"9.0|%s|C%d|H%d|N<=%d|%s|fwd" % (p, C, C, n, d) for p in ("f32z_bf16", "bf16") for C in WIDTHS for n in (100, 1) for d in ("out", "in")}
    h = hashlib.sha256(json.dumps({k: v for k, v in T["cells"].items() if k not in skip}, sort_keys=True).encode()).hexdigest()
    assert h == OTHER_CELLS_SHA256


@pytest.mark.parametrize("prec,residency", CLASSES)
def test_v4_floor_per_class(prec, residency):
    assert KT.V4_N_MIN_DEFAULT == 101
    assert KT.v4_n_min("9.0", prec, 128, 128) == 2
    assert KT.v4_n_min(9.0, prec, 128, 128) == 2                              # cc as a number
    for other in (("8.0", prec, 128, 128), ("9.0", prec, 256, 256), ("9.0", prec, 384, 384), ("9.0", "tf32", 128, 128), ("9.0", "fp32", 128, 128), ("10.0", prec, 128, 128), ("9.0", prec, 64, 128)):
        assert KT.v4_n_min(*other) == 101, other
    KT.admits("v4", "9.0", "bf16", 128, 128, 2, residency=residency)          # admitted (no raise)
    KT.admits("v4", "9.0", "bf16", 128, 128, 100, residency=residency)
    with pytest.raises(KT.Refusal) as e:
        KT.admits("v4", "9.0", "bf16", 128, 128, 1, residency=residency)
    assert e.value.kind == "n<2:below_n_min"
    with pytest.raises(KT.Refusal) as e:                                        # every other class: the identical word as 0.5.212.1
        KT.admits("v4", "8.0", "bf16", 128, 128, 24, residency=residency)
    assert e.value.kind == "n<101:below_n_min"
    with pytest.raises(KT.Refusal) as e:
        KT.admits("v4", "9.0", "bf16", 256, 256, 24, residency=residency)      # C256: floor unchanged in this release
    assert e.value.kind == "n<101:below_n_min"


@pytest.mark.parametrize("prec,residency", CLASSES)
@pytest.mark.parametrize("stack", VOUCHED + ("H100:2.7.1+cu126/3.3.1/cueq0.10.0", None))   # + an UNLISTED cc-9.0 stack (inherits the cell word) and no stack word
@pytest.mark.parametrize("C", WIDTHS)
def test_select_words_below_at_and_above_the_old_floor(prec, residency, stack, C):
    for direction in ("outgoing", "incoming"):
        rows = {}
        for N in (1, 2, 24, 100, 101, 256):
            for word in ("fast", "big", "exact"):
                rows[(N, word)] = KT._select("9.0", "bf16", C, C, N, direction, word=word, residency=residency, stack=stack, tf32=None, has_cueq=None).row
        for N in (2, 24, 100):
            assert rows[(N, "fast")] == "v4" and rows[(N, "big")] == "v4", (stack, direction, N, rows[(N, "fast")], rows[(N, "big")])
            assert rows[(N, "exact")] == "cueq", (stack, direction, N)             # exact: unchanged (no exact-class kernel is bitwise vs the library's small-N algorithm)
        assert rows[(1, "fast")] == rows[(1, "big")] == rows[(1, "exact")] == "cueq"   # a 1-token call: the stock op (measured: no kernel beats it; v4 refuses N<2)
        if C == 128:
            assert rows[(101, "fast")] == "v4" and rows[(256, "fast")] == "v4"           # at / above the old floor: as before
        else:
            assert rows[(101, "fast")] in KT.ROW_NAMES and rows[(101, "fast")] not in ("cueq", "torch_math")   # C256 101..256: the N<=256 cell's kernel rows, as before


def test_the_package_face_takes_n_min_and_keeps_its_floor():
    from opt_core.kernels.fpf_trimul_v4 import generic as G
    import opt_core.kernels.fpf_trimul_v4 as P
    assert G.N_MIN == 101
    assert P.__version__ == "4.4.6"
    for fn in (G.supported, G.trimul, G.trimul_packed, G._trimul_packed, G._check):
        assert "n_min" in inspect.signature(fn).parameters, fn.__name__
    assert inspect.signature(G.supported).parameters["n_min"].default is None
    src = inspect.getsource(G._check)
    assert "nmin = N_MIN if n_min is None else int(n_min)" in src and 'raise TrimulUnsupported("N<%d" % nmin' in src


def test_no_kernel_source_or_launch_cell_changed_words():
    """The release's claim in words the CHANGES carry: fpf_trimul_v4's kernel modules and launch table are not part of this change (their bytes are
    pinned by the package's own tests; here: the face names them untouched)."""
    import opt_core.kernels.fpf_trimul_v4 as P, os
    d = os.path.dirname(P.__file__)
    for name in ("kernels.py", "kdesc.py", "cells.py", "table.json"):
        assert os.path.isfile(os.path.join(d, name)), name


def test_fpf_trimul_v4_first_call_forwards_the_floor():
    """4.4.6 (opt_core 0.5.212.3): the first (stack-padded) call of a class passes n_min= too (source pin; the GPU proof is the kit re-smoke: a <=100-token
    item's census reads `LEVER name=trimul_v4 ... fallback=0`)."""
    from opt_core.kernels.fpf_trimul_v4 import generic as G
    import opt_core.kernels.fpf_trimul_v4 as P
    assert P.__version__ == "4.4.6"
    src = inspect.getsource(G.trimul_packed)
    assert "padded_call(_trimul_packed, z, mask, outgoing=outgoing, weights=weights, residual=residual, eps=eps, stock_round=stock_round, pad=pad, n_min=n_min)" in src
    assert src.count("n_min=n_min") >= 3                                           # the ImportError branch, the padded first call, every later call
