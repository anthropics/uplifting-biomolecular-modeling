"""kernels.trimul: the triangle-multiplication provider's cell table, words, refusals, default rule, META census and carried-copy byte identity --
pure checks (no GPU, no framework)."""
import ast
import filecmp
import json
import os

import pytest

from opt_core import kernels
from opt_core.kernels import trimul as T

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_TREE = os.path.dirname(os.path.dirname(CORE_DIR))
CARDS = ("9.0", "8.0", "10.0", "10.3", "12.0")      # measured capabilities: H100 / A100 columns, the Blackwell parts (B200 10.0, B300 10.3) by the parts fold
SIZES = (400, 800, 1200)
# every (precision, c_z, c_hidden) triangle-multiplication shape the release tree's torch pair stacks run today, both directions:
# trunk c_z 128 (bf16 | fp32-resident z under bf16 autocast | fp32 | tf32), c_z 256 trunks (bf16), c_z 384 (bf16 / fp32 / tf32), template c 64
SHAPES = [("bf16", 128, 128, None), ("bf16", 128, 128, "fp32"), ("fp32", 128, 128, None), ("tf32", 128, 128, None), ("bf16", 256, 256, None),
          ("fp32", 256, 256, None), ("bf16", 384, 384, None), ("fp32", 384, 384, None), ("tf32", 384, 384, None), ("bf16", 64, 64, None), ("fp32", 64, 64, None)]


def test_table_schema_rows_words():
    t = T.table()
    assert t["schema"] == "trimul_cells/v1"
    assert set(t["rows"]) == set(T.ROW_NAMES)
    assert tuple(t["weights"]) == T.WEIGHT_KEYS
    for name, row in t["rows"].items():
        assert row["class"] in ("fast", "exact", "stock"), name
        assert "backward" in row and "capture_safe" in row and "boundary" in row and "admits" in row and "fallback" in row, name
        assert (row["class"] == "exact") == (name in T.EXACT_ROWS), name
        assert bool(row["backward"]) == (name in T.BACKWARD_ROWS), name
    assert t["rows"]["tx_sm90a"]["fallback"] == "v4" and t["rows"]["tx_sm90a_exact"]["fallback"] == "tmk3_exact"
    assert t["rows"]["ef2_fused"]["fallback"] == "cueq" and t["rows"]["cueq"]["fallback"] == "torch_math"
    assert "9.0 only" in " ".join(t["rows"]["tx_sm90a"]["admits"]["cc"])
    assert set(t["tiers"]["exact"]) == set(T.EXACT_ROWS + T.STOCK_ROWS)
    ccs = set()
    for key, cell in t["cells"].items():
        cc, prec, c, h, n, d, p = key.split("|")
        ccs.add(cc)
        assert cc in CARDS and prec in T.PRECISIONS and c[0] == "C" and h[0] == "H" and n.startswith("N<=") and d in ("out", "in", "outin") and p in ("fwd", "fwdbwd"), key
        assert cell["measured"] is True and cell["ref_stack"] in t["stacks"], key
        assert cell["fast"] == cell["fast_order"][0], key
        for st, w in cell["fast_per_stack"].items():
            assert st in t["stacks"] and w in T.ROW_NAMES, (key, st, w)
        for st, w in cell["exact_per_stack"].items():
            assert w in T.EXACT_ROWS + T.STOCK_ROWS + ("torch_module",), (key, st, w)
            if w in T.EXACT_ROWS:                                                  # the exact rule: bitwise AND >= x1.00 on that stack
                assert cell["class"][w][st].startswith("bitwise"), (key, st, w)
                if st in (cell.get("byte_vouch") or {}) and st not in (cell.get("x_stock", {}).get(w) or {}):
                    continue                                                       # a byte-vouch-only column (core 0.5.109.0): bitwise established by the vouch, no timing on that column by order (parity vs the stock op rests on the cc's raced columns)
                assert cell["x_stock"][w][st] >= 1.0, (key, st, w)
    assert {"9.0", "8.0"} <= ccs <= set(CARDS), ccs                                   # both reference cards carry cells; the parts fold adds the Blackwell capabilities


def test_the_torch_211_h100_column_resolves_fast_and_big_to_esm_v61_at_c256_forward():
    """in_model.c256_fwd_h100_t211: on the torch-2.11.0+cu128 / triton-3.6.0 / cuequivariance-0.10.0 H100 column (one design engine's image; tx_sm90a had no
    binary for that ABI until 0.5.75.0) the c_z-256 bf16 FORWARD cells resolve `fast` and `big` to esm_v61 at N <= 800 in both directions (that engine's same-image
    numbers: esm_v61 x1.22 over v4 per direction at 800 tokens, x1.15-1.18 at 431); N >= 1200 keeps v4; `exact` is what it was (cueq below the byte-tested
    buckets, tmk3_exact from 400 on); the fwd+bwd form resolves to ef2_fused (esm_v61 has no backward); v4 / tx_sm90a stay selectable by row word (tx served by
    name on that ABI from 0.5.75.0: prebuilt key added); rows.esm_v61.bound_by records the tier binding."""
    t = T.table()
    S = "H100:2.11.0+cu128/3.6.0/cueq0.10.0"
    ABI = "torch2.11.0+cu128-cpython-312-x86_64-linux-gnu-sm90"
    rul = t["in_model"]["c256_fwd_h100_t211"]
    assert rul["applies_to"] and all(e[1] == S and e[2] in ("fast", "big") for e in rul["applies_to"])
    assert "bound_by" in t["rows"]["esm_v61"] and S in t["rows"]["esm_v61"]["bound_by"]
    for N in (200, 256, 400, 431, 800):
        for d in ("outgoing", "incoming"):
            for w in ("fast", "big"):
                assert T.select("9.0", "bf16", 256, 256, N, d, word=w, stack=S, abi=ABI, has_cueq=True).row in ("esm_v61", "native"), (N, d, w)   # native where the policy's same-box run on that column tied / won
            ex = T.select("9.0", "bf16", 256, 256, N, d, word="exact", stack=S, abi=ABI, has_cueq=True).row
            assert ex == ("cueq" if N <= 256 else "tmk3_exact") or ex == "native_exact", (N, d, ex)   # native_exact where byte-vouched on this very column (EXACT VOUCH C3a: bitwise to this stack's stock op)
            assert T.select("9.0", "bf16", 256, 256, N, d, word="fast", stack=S, abi=ABI, has_cueq=True, backward=True).row == "ef2_fused", (N, d)
            assert T.select("9.0", "bf16", 256, 256, N, d, word="v4", stack=S, abi=ABI).row == "v4"
            assert T.select("9.0", "bf16", 256, 256, N, d, word="tx_sm90a", stack=S, abi=ABI).row == "tx_sm90a"      # prebuilt key for this ABI from 0.5.75.0 (was refused no_prebuilt)
    for N in (1200,):
        for w in ("fast", "big"):
            assert T.select("9.0", "bf16", 256, 256, N, "outgoing", word=w, stack=S, abi=ABI, has_cueq=True).row in (("v4", "native") if w == "fast" else ("v4", "native", "esm_shapes", "esm_v5_fwd", "tmk3_fast")), (N, w)   # CORE R1: or native where the column inherited the 9.0 reference verdict; big from 0.5.102.0 = the lowest FOOTPRINT row (the payload's resident planes counted), an ESM row here
    for key, cell in t["cells"].items():                                           # the ruling touches no other column: every moved word is listed in applies_to
        cc, prec, c, h, nb, d, mode = key.split("|")
        for tier in ("fast", "big", "exact"):
            w = (cell.get(tier + "_per_stack") or {}).get(S)
            if w == "esm_v61":
                assert [key, S, tier] in [e[:3] for e in rul["applies_to"]], (key, tier)


def test_exact_class_rows_serve_under_the_exact_word_only_on_a_vouched_stack():
    """EXACT VOUCH IS STACK-SPECIFIC: cells.<key>.vouched_on.<row> lists the stacks an exact-class row's bitwise record was taken on (= the class
    block's 'bitwise' entries); every exact column word is vouched on its own stack and a cell-level exact-class word on the reference stack; under
    the tier word `exact` a stack WITHOUT the record gets the row refused by name (exact_vouch_not_recorded_on:<stack>) and the stock op of the stack;
    a vouched stack gets the same row as before; row words are unaffected."""
    t = T.table()
    n_gate = 0
    for key, cell in t["cells"].items():
        v = cell.get("vouched_on") or {}
        for row, sts in v.items():
            assert row in T.EXACT_ROWS and sts, (key, row)
            for st in sts:
                assert str(cell["class"][row][st]).startswith("bitwise"), (key, row, st)
        for st, w in cell["exact_per_stack"].items():
            if w in T.EXACT_ROWS:
                assert st in v.get(w, []), (key, st, w)
        if cell["exact"] in T.EXACT_ROWS:
            assert cell["ref_stack"] in v.get(cell["exact"], []), (key, cell["exact"])
    UNL = {"9.0": "H100:2.13.0+cu130/3.7.9/cueq0.11.1", "8.0": "A100:2.13.0+cu130/3.7.9/cueq0.11.1"}      # a stack word no column carries
    assert not any(UNL[c] in t["stacks"] for c in UNL)
    for key, cell in t["cells"].items():
        cc, prec, c, h, nb, d, mode = key.split("|")
        if mode != "fwd" or cell["exact"] not in T.EXACT_ROWS:
            continue
        dt = {"bf16": "bf16", "f32z_bf16": "bf16", "fp32": "fp32", "tf32": "fp32"}[prec]
        kw = dict(word="exact", residency=("fp32" if prec == "f32z_bf16" else None), tf32=(prec == "tf32"))
        dn = "outgoing" if d == "out" else "incoming"
        ref = T.select(cc, dt, int(c[1:]), int(h[1:]), int(nb[3:]), dn, stack=cell["ref_stack"], has_cueq=(not cell["ref_stack"].endswith("nocueq")), **kw)
        assert ref.row == cell["exact"], (key, ref.row)                                                  # vouched stack: unchanged
        g = T.select(cc, dt, int(c[1:]), int(h[1:]), int(nb[3:]), dn, stack=UNL.get(cc, "X9:9.9.9+cu999/9.9.9/cueq9.9.9"), has_cueq=True, **kw)
        assert g.row in T.STOCK_ROWS and "exact_vouch_not_recorded_on:%s" % UNL.get(cc, "X9:9.9.9+cu999/9.9.9/cueq9.9.9") in g.reason, (key, g.row, g.reason)   # unvouched: refused by name -> stock op
        g2 = T.select(cc, dt, int(c[1:]), int(h[1:]), int(nb[3:]), dn, stack=UNL.get(cc, "X9:9.9.9+cu999/9.9.9/cueq9.9.9"), has_cueq=False, **kw)
        assert g2.row == "torch_math", (key, g2.row)
        assert T.select(cc, dt, int(c[1:]), int(h[1:]), int(nb[3:]), dn, word=cell["exact"], stack=UNL.get(cc, "X9:9.9.9+cu999/9.9.9/cueq9.9.9"), residency=kw["residency"], tf32=kw["tf32"]).row == cell["exact"]   # by row word
        n_gate += 1
    assert n_gate >= 50, n_gate


def test_round5_coverage_cells_exact_records_and_the_stock_op_guard():
    """RACE round 5 (fresh z per call; the two reference images per card): the 1556-covering / 2048 buckets at c 384 are CELL_HIT on both cards, the
    N<=1024 width strip exists on cc 8.0, the f32z / tf32 template-width families have cells (no torch_math fallback), the f32z c 384 H100 columns
    resolve fast to esm_shapes, the torch-2.10 H100 column carries its own tmk3_exact exact words where byte-tested at >= parity, tmk3_exact fp32 c 128
    is recorded NOT bitwise on the A100 stacks (never the exact word there), and the stock library op raising inside its call is a refusal BY NAME."""
    t = T.table()
    H10 = "H100:2.10.0+cu128/3.6.0/cueq0.10.0"
    for cc in ("9.0", "8.0"):                                                       # the 1556-token pair width 384 key: a measured cell on both cards
        for d in ("outgoing", "incoming"):
            for n in (1556, 2048):
                s = T.select(cc, "bf16", 384, 384, n, d, word="fast")
                assert s.size_measured and s.cell.endswith("|N<=%d|%s|fwd" % (n, d[:3] if d == "outgoing" else "in")), (cc, n, d, s)
        for C, H in ((64, 64), (64, 128), (128, 128), (256, 256), (384, 384)):    # the N = 1024 width strip
            s = T.select(cc, "bf16", C, H, 1024, "outgoing", word="fast")
            assert s.size_measured and "|N<=1024|" in s.cell, (cc, C, H, s)
        for prec, kw in (("f32z_bf16", dict(residency="fp32")), ("tf32", dict(tf32=True))):   # the template widths under fp32 residency / TF32: cells, not the module math
            for C, H in ((64, 64), (64, 128)):
                s = T.select(cc, "bf16" if prec == "f32z_bf16" else "fp32", C, H, 2048, "outgoing", word="fast", **kw)
                assert s.cell is not None and s.row not in T.STOCK_ROWS and (s.x_stock is None or s.x_stock > 1.5), (cc, prec, C, H, s)   # a measured kernel row (the re-tiled rows / the ESM line / native by the policy), never the module math
    for n in (400, 800, 1200):
        for st in ("H100:2.13.0+cu130/3.7.1/cueq0.11.1", H10, "H100:2.12.0+cu130/3.7.0/cueq0.10.0"):
            s = T.select("9.0", "bf16", 384, 384, n, "outgoing", word="fast", residency="fp32", stack=st)
            assert s.row == "esm_shapes" and s.x_stock > 2.0, (n, st, s)                                # fresh-z round 5: x1.06-1.19 over tmk3_exact, x2+ over the stock op
            assert T.select("9.0", "bf16", 384, 384, n, "outgoing", word="exact", residency="fp32", stack=st).row == "tmk3_exact"
    assert T.select("8.0", "bf16", 384, 384, 800, "outgoing", word="fast", residency="fp32").row == "tmk3_exact"   # cc 8.0: parity with esm_shapes -> the word holds
    keyed = 0
    for key, cell in t["cells"].items():                                            # the 2.10 H100 column: its exact word is byte-tested ON that stack at >= parity
        w = cell["exact_per_stack"].get(H10)
        if w in T.EXACT_ROWS:
            assert T.exact_vouched(cell, w, H10) and cell["x_stock"][w][H10] >= 1.0, (key, w); keyed += 1
    assert keyed >= 60, keyed
    assert T.select("9.0", "bf16", 128, 128, 256, "outgoing", word="exact", residency="fp32", stack=H10).row == "tmk3_exact"
    neg = 0
    for key, cell in t["cells"].items():                                            # the NEGATIVE record: tmk3_exact fp32 c 128 vs the library fp32 op on cc 8.0
        if key.startswith("8.0|fp32|C128|H128|"):
            for st, c_ in (cell["class"].get("tmk3_exact") or {}).items():
                assert not str(c_).startswith("bitwise"), (key, st, c_)
                neg += "NOT bitwise" in str(c_)
            assert all(w in T.STOCK_ROWS for w in cell["exact_per_stack"].values()) and cell["exact"] in T.STOCK_ROWS, key
            assert "tmk3_exact" not in (cell.get("vouched_on") or {}), key
    assert neg >= 20, neg
    import sys, types                                                               # the stock op raising (its own assert at a width / build) -> Refusal by name -> the module math
    def raising(z, direction, mask, eps, **kw):                                     # (no torch needed: a stand-in module and a pre-built pack in the cache)
        assert False, "pair width 64 unsupported by this build"
    class _Tens:                                                                    # what weights_key reads off a tensor
        shape, dtype, device = (4, 4), "float32", "cpu"
        def data_ptr(self): return id(self)
        def stride(self): return (4, 1)
    w = {k: _Tens() for k in T.WEIGHT_KEYS}
    cache = {("cueq_w",) + T.weights_key(w): ({}, w["w_ap"], w["w_o"])}
    had = "torch" in sys.modules
    if not had: sys.modules["torch"] = types.ModuleType("torch")
    try:
        with pytest.raises(T.Refusal) as e:
            T._cueq_call(raising, object(), None, "outgoing", w, 1e-5, cache)
    finally:
        if not had: del sys.modules["torch"]
    assert e.value.kind.startswith("stock_op_raised:AssertionError(pair_width_64") and e.value.row == "cueq" and e.value.fallback == "torch_math", e.value.kind


def test_an_unlisted_stack_word_reads_its_nearest_listed_column_for_fast_and_big():
    """SIBLING-COLUMN rule: a stack word the cell does not list (here torch 2.12.0+cu130 / triton 3.7.0 / cuequivariance 0.11.1 on either card --
    a listed torch build with another library minor) reads ONE listed column for the fast / big tiers: the same torch version + cuda tag first
    (the 2.12.0+cu130 / cuequivariance 0.10.0 column), else the same torch major.minor, else the cc's reference column; the Selection names that
    column (stack=, reason column=sibling:<word>) and the words over the size buckets are THAT COLUMN's (a row refused by name on the caller's
    stack -- no prebuilt binary for its ABI -- steps aside to the next word of the same column); the exact tier inherits nothing."""
    U, L = "H100:2.12.0+cu130/3.7.0/cueq0.11.1", "H100:2.12.0+cu130/3.7.0/cueq0.10.0"
    ABI = "torch2.12.0+cu130-cpython-311-x86_64-linux-gnu-sm90"                       # that image's extension ABI: tx_sm90a ships no binary for it
    assert ABI not in T.TX_PREBUILT_ABIS
    t = T.table()
    ref9 = T.cc_reference_column("9.0")
    assert ref9 == "H100:2.13.0+cu130/3.7.1/cueq0.11.1" and T.cc_reference_column("8.0").startswith("A100:2.13.0+cu130/")
    c = t["cells"]["9.0|bf16|C256|H256|N<=800|out|fwd"]
    assert T.sibling_column(c, U) == (L, "sibling:torch")                               # rule 1: same torch + cuda tag, another library minor
    assert T.sibling_column(c, L) == (L, "listed")
    assert T.sibling_column(c, "H100:2.12.1+cu128/3.7.2/cueq0.12.0") == (L, "sibling:minor")   # rule 2: same torch major.minor
    assert T.sibling_column(c, "H100:2.14.0+cu131/3.8.0/cueq0.12.0") == (ref9, "reference")    # rule 3: the cc's reference column
    assert T.sibling_column(c, None) == (c["ref_stack"], "reference")                   # a planning call reads the cell's reference column as before
    assert T.sibling_column(t["cells"]["8.0|bf16|C256|H256|N<=800|out|fwd"], "A100:2.12.0+cu130/3.7.0/cueq0.11.1") == ("A100:2.12.0+cu130/3.7.0/cueq0.10.0", "sibling:torch")
    seen_sibling = 0
    for C in (128, 256):
        for d in ("outgoing", "incoming"):
            for word in ("fast", "big"):
                served = []
                for N in (256, 400, 512, 800, 1024, 1200, 1536, 2048, 3072, 4096):
                    s = T.select("9.0", "bf16", C, C, N, d, word=word, stack=U, abi=ABI)
                    cell = t["cells"][s.cell]
                    col, rule = T.sibling_column(cell, U)
                    assert s.stack == col and rule in ("sibling:torch", "reference"), (C, d, word, N, s, rule)   # cells listing a 2.12 column: it; the others: the reference
                    assert ("column=%s:%s" % ("sibling" if rule.startswith("sibling") else "reference", col)) in s.reason, s.reason
                    seen_sibling += rule == "sibling:torch"
                    colw = cell.get(word + "_per_stack", {}).get(col, cell[word])          # THE word of that column at this bucket ...
                    try:
                        T.admits(colw, "9.0", "bf16", C, C, N, abi=ABI, triton="3.7.0"); col_admits = True
                    except T.Refusal:
                        col_admits = False
                    if col_admits:
                        assert s.row == colw, (C, d, word, N, s.row, colw)                # ... is what the unlisted word serves (no mix of columns over the buckets)
                    else:
                        assert s.row != colw and ("skip %s(" % colw) in s.reason, s.reason   # ... unless it refuses by name on this stack: the next word of that column
                    same = T.select("9.0", "bf16", C, C, N, d, word=word, stack=col, abi=ABI)   # = exactly what the sibling column's own stack serves under the same ABI facts
                    assert s.row == same.row, (C, d, word, N, s.row, same.row)
                    served.append(s.row)
                assert "tx_sm90a" not in served and "tx_sm90a_exact" not in served, served
    assert seen_sibling >= 40, seen_sibling
    e = T.select("9.0", "bf16", 256, 256, 512, "incoming", word="exact", stack=U)      # the exact tier: stack-specific vouch only -> the stock op, refused rows named
    assert e.row in T.STOCK_ROWS and ("exact_vouch_not_recorded_on:%s" % U) in e.reason and e.stack == t["cells"][e.cell]["ref_stack"], e
    r = T.select("9.0", "bf16", 256, 256, 800, "outgoing", word="native", stack=U)     # a row word: served as asked; its numbers read the sibling column
    assert r.row == "native" and r.stack == L and r.reason.startswith("column=sibling:" + L), r


def test_the_exact_tier_refuses_by_name_beyond_the_vouched_envelope():
    """Beyond a family's largest bucket no byte test vouches any size >= N: the tier word `exact` skips every exact-class row BY NAME
    (`exact_beyond_vouched_envelope:N>{largest bucket}`) and resolves to the stock op -- on the column that vouches the row up to its largest
    vouched bucket (the torch-2.7.1+cu126 / triton-3.3.1 column at c 384: native_exact vouched <= 2048) and on the reference column alike; inside the
    vouched envelope the vouched row serves as before; between the largest VOUCHED bucket and the largest bucket (cells measured above 2048 whose byte
    test could not run in-process) the cell's own exact word is the stock op by name; fast / big resolve the covering cell (else the nearest cell,
    beyond_measured); the form-keyed exact refuses beyond its size cells."""
    t = T.table()
    ODDE, R9 = "H100:2.7.1+cu126/3.3.1/cueq0.10.0", "H100:2.13.0+cu130/3.7.1/cueq0.11.1"
    big = max(int(k.split("|")[4][3:].split("+")[0]) for k in t["cells"] if k.startswith("9.0|bf16|C384|H384|") and k.endswith("|out|fwd"))   # table-derived
    assert big >= 2048, big
    assert T.exact_envelope("9.0", "bf16", 384, 384, "outgoing", "native_exact", ODDE) == (big, 2048)
    buckets = sorted(int(k.split("|")[4][3:].split("+")[0]) for k in t["cells"] if k.startswith("9.0|bf16|C384|H384|") and k.endswith("|out|fwd"))
    for st in (ODDE, R9, None):
        for d in ("outgoing", "incoming"):
            inside = T.select("9.0", "bf16", 384, 384, 2048, d, word="exact", stack=st)
            assert inside.row == "native_exact" and inside.size_measured, (st, d, inside)
            for N in sorted({2049, 2708, big + 1, 4096, 2 * big}):
                s = T.select("9.0", "bf16", 384, 384, N, d, word="exact", stack=st)
                assert s.row in T.STOCK_ROWS, (st, d, N, s)
                for r in T.EXACT_ROWS:                                                  # no exact-class row is the answer above the vouched envelope
                    assert s.row != r
                if N > big:                                                              # beyond the largest bucket: the exact-class word (if the largest cell names one on this column) is skipped by name, else the cell's stock word extends (said so)
                    top = t["cells"]["9.0|bf16|C384|H384|N<=%d|%s|fwd" % (big, "out" if d == "outgoing" else "in")]
                    topw = top["exact_per_stack"].get(s.stack, top["exact"])
                    want = ("skip %s(exact_beyond_vouched_envelope:N>%d)" % (topw, big)) if topw in T.EXACT_ROWS else ("beyond_measured(N<=%d)" % big)
                    assert not s.size_measured and want in s.reason, (N, want, s.reason)
            cover = min((bN for bN in buckets if bN >= 2708), default=None)
            f = T.select("9.0", "bf16", 384, 384, 2708, d, word="fast", stack=st)      # fast / big: the covering cell (else the nearest cell, said so)
            near = t["cells"]["9.0|bf16|C384|H384|N<=%d|%s|fwd" % (cover or max(buckets), "out" if d == "outgoing" else "in")]
            col = f.stack
            assert f.row == near["fast_per_stack"].get(col, near["fast"]), (st, d, f)
            assert (cover is not None) or ("beyond_measured(N<=%d)" % max(buckets) in f.reason and not f.size_measured), (st, d, f)
            b = T.select("9.0", "bf16", 384, 384, 2708, d, word="big", stack=st)
            assert b.row not in T.STOCK_ROWS and (cover is not None or "beyond_measured" in b.reason), (st, d, b)
    rw = T.select("9.0", "bf16", 384, 384, 2708, "outgoing", word="native_exact", stack=ODDE)   # a ROW word is the caller's own statement: unaffected
    assert rw.row == "native_exact"
    # the form-keyed exact: beyond the form's largest size cell on a vouched stack -> Refusal by name naming the envelope; inside -> the form's row
    fc = t["form_cells"]
    fams = {}
    for k, c in fc.items():
        base, _, form = k.partition("+")
        pp = base.split("|")
        fams.setdefault((pp[0], pp[1], pp[2], pp[3], pp[5], pp[6], form), []).append((int(pp[4][3:]), k))
    tried = 0
    for (cc, prec, C, H, d, mode, form), lst in sorted(fams.items()):
        if mode != "fwd" or prec != "bf16": continue
        nmax, kmax = max(lst)
        stacks = list((fc[kmax].get("stacks") or {}))
        if not stacks: continue
        st = stacks[0]; direction = "outgoing" if d == "out" else ("incoming" if d == "in" else d)
        with pytest.raises(T.Refusal) as e:
            T.select(cc, "bf16", int(C[1:]), int(H[1:]), nmax + 500, direction, word="exact", form=form, stack=st)
        assert e.value.kind.startswith("form_vouch_not_recorded") and ("N>%d" % nmax) in e.value.kind and e.value.row == T.FORMS[form], e.value.kind
        rec = fc[kmax]["stacks"][st]
        n_in = next((n for n in range(nmax, 15, -1) if T.vouched(n, rec) is not None), None)
        if n_in is not None:
            ok = T.select(cc, "bf16", int(C[1:]), int(H[1:]), n_in, direction, word="exact", form=form, stack=st)
            assert ok.row == T.FORMS[form], ok
        tried += 1
    assert tried >= 1


def test_every_column_word_is_what_select_returns_on_that_stack():
    """The table's per-stack columns ARE the selection: for every cell x listed stack x tier, select(stack=<that stack>) returns the column's row
    unless the row's static admits() refuses the envelope there (then the reason names the skip) -- a column never sits unread behind the cell-level order."""
    t = T.table()
    n = skipped = 0
    for key, cell in t["cells"].items():
        cc, prec, c, h, nb, d, mode = key.split("|")
        if mode != "fwd" or d not in ("out", "in"):
            continue
        dt = {"bf16": "bf16", "f32z_bf16": "bf16", "fp32": "fp32", "tf32": "fp32", "fp16": "fp16", "f32z_fp16": "fp16"}.get(prec)
        if dt is None:
            continue
        for tier in ("fast", "exact", "big"):
            for st, row in (cell.get(tier + "_per_stack") or {}).items():
                sel = T.select(cc, dt, int(c[1:]), int(h[1:]), int(nb[3:]), "outgoing" if d == "out" else "incoming", word=tier, stack=st,
                               residency=("fp32" if prec.startswith("f32z") else None), tf32=(prec == "tf32"), has_cueq=(not st.endswith("nocueq")))
                if sel.row != row:
                    assert "skip %s(" % row in sel.reason, (key, tier, st, row, sel)
                    skipped += 1
                n += 1
    assert n >= 2000 and skipped <= n // 20, (n, skipped)


@pytest.mark.parametrize("cc", ("9.0", "8.0"))
def test_every_shape_the_tree_runs_lands_on_a_measured_cell_or_a_named_stock_row(cc):
    t = T.table()
    for prec, C, H, res in SHAPES:
        for N in SIZES:
            for d in ("outgoing", "incoming"):
                dt = "bf16" if prec == "bf16" else prec
                sel = T.select(cc, dt, C, H, N, d, word="fast", residency=res, tf32=(prec == "tf32"))
                assert sel.row in T.ROW_NAMES
                if sel.cell is None:                                               # an unmeasured family: the named stock row, never a guess
                    assert sel.row in T.STOCK_ROWS and sel.reason.startswith("named stock row"), (cc, prec, C, N, d, sel)
                else:
                    assert sel.cell in t["cells"] and sel.stack in t["stacks"], sel
                ex = T.select(cc, dt, C, H, N, d, word="exact", residency=res, tf32=(prec == "tf32"))
                assert ex.row in T.EXACT_ROWS + T.STOCK_ROWS, ex
    # the target coverage block names every tri_mul target of the torch engines on this card: a measured cell family or a named stock value
    cov = [k for k in t["coverage"] if k.startswith(cc + "|")]
    assert cov
    for k in cov:
        v = t["coverage"][k]
        assert v["cell_sizes"] or (v.get("value", "").startswith("stock:") and v.get("n/a")), k


def test_default_rule_a_row_word_serves_that_row():
    for cc in CARDS:
        for row in ("v4", "tmk3_exact", "tmk3_fast", "cueq", "torch_math"):
            sel = T.select(cc, "bf16", 128, 128, 800, "outgoing", word=row)
            assert sel.row == row and sel.word == row, sel
        assert T.select("9.0", "bf16", 256, 256, 800, "outgoing", word="tx_sm90a").row == "tx_sm90a"
        assert T.select("9.0", "bf16", 256, 256, 800, "outgoing", word="tx_sm90a_exact").row == "tx_sm90a_exact"
        assert T.select(cc, "bf16", 256, 256, 500, "outgoing", word="ef2_fused", backward=True).row == "ef2_fused"
        assert T.select(cc, "bf16", 256, 256, 500, "incoming", word="ef2_cueq_tiles", backward=True).row == "ef2_cueq_tiles"
    # a row word does not need a measured cell (the cell is informative): fp16 has no cells, the row still serves by name
    sel = T.select("9.0", "fp16", 128, 128, 800, "outgoing", word="cueq")
    assert sel.row == "cueq" and sel.cell is None


def test_tier_words_are_the_measured_winners():
    """The unification policy (table-level `policy`): FAST and BIG = the FASTEST measured row of the deciding same-box run on that column, TIES -> native
    (a challenger holds the word against a native arm only with a win beyond noise: median <= 0.95 x native's and the IQR intervals disjoint); big = the
    same decision; every decided column carries cells.<key>.policy.<stack> whose 'after' words ARE the column words; the cell-level word is the reference
    column's; EXACT untouched (stack-specific vouch: cc-8.0 c_z-128 exact = the stock op or native_exact where vouched)."""
    t = T.table()
    pol = t.get("policy") or {}
    assert pol.get("rule", "").startswith("FAST") and "FASTEST measured row wins" in pol["rule"] and "kbench" in pol and pol.get("version"), pol.keys()
    NATIVE_FAM = ("native", "native:f32in")
    ESM = tuple(pol["big_excluded"]["rows"])                                     # excluded from big everywhere and from fast at c_z 128 (program_evidence, kind in_model)
    TIE = 0.98
    def eligible(row, key, tier): return not (row in ESM and (tier == "big" or key.split("|")[2:4] == ["C128", "H128"]))
    def beats(a, b):
        (ma, ia), (mb, ib) = a, b
        if ma > TIE * mb: return False
        if ia is None or ib is None: return True
        return (ma + ia / 2.0) < (mb - ib / 2.0)
    n_rec = n_native = n_tie_checked = 0
    for key, cell in t["cells"].items():
        ref = cell.get("ref_stack")
        for tier in ("fast", "big"):                                              # the cell-level word = the reference column's word; fast_order headed by it
            if ref in cell.get(tier + "_per_stack", {}):
                assert cell[tier] == cell[tier + "_per_stack"][ref], (key, tier, cell[tier], ref)
        assert cell["fast_order"][0] == cell["fast"], key
        for st, rec in (cell.get("policy") or {}).items():
            n_rec += 1
            for tier in ("fast", "big"):
                assert cell[tier + "_per_stack"].get(st) == rec[tier]["after"], (key, st, tier, rec[tier], cell[tier + "_per_stack"].get(st))
            assert rec["big"]["after"] == rec["fast"]["after"] or rec["fast"]["after"] in ESM or rec["mode"] in ("oldrun", "inherited", "excluded", "in_model"), (key, st, rec)   # big = the fast decision unless fast is an (excluded) ESM row
            n_native += rec["fast"]["after"] in NATIVE_FAM
            if rec["mode"] == "run":                                                # re-derive the tie rule from the deciding run's arms, per tier with its exclusions
                run = cell[rec["run"]][st]
                for tier in ("fast", "big"):
                    arms = {r: (a["ms"], a.get("iqr")) for r, a in run["arms"].items() if isinstance(a, dict) and isinstance(a.get("ms"), (int, float)) and r in t["tiers"]["fast"] and eligible(r, key, tier)}
                    nats = [r for r in arms if r in NATIVE_FAM]
                    w = rec[tier]["after"]
                    if nats and w not in NATIVE_FAM:
                        nat = min(nats, key=lambda r: arms[r][0])
                        assert w in arms and beats(arms[w], arms[nat]), (key, st, tier, w, arms.get(w), nat, arms[nat])   # a non-native word beat native beyond noise
                        n_tie_checked += 1
                    elif nats:
                        assert not any(beats(arms[r], arms[w]) for r in arms if r not in NATIVE_FAM and T_admits(t, r, key, st)), (key, st, tier, w, arms)   # nothing eligible beat native beyond noise
                        n_tie_checked += 1
    for key, cell in t["cells"].items():                                            # the ESM family: never a big word; never a c_z-128 fast word (program_evidence, kind in_model)
        c128 = key.split("|")[2:4] == ["C128", "H128"]
        assert cell["big"] not in ESM and not [st for st, w in cell["big_per_stack"].items() if w in ESM], key
        if c128:
            assert cell["fast"] not in ESM and not [st for st, w in cell["fast_per_stack"].items() if w in ESM], key
            assert cell.get("program_evidence", {}).get("kind") == "in_model" and "c128_esm" in cell["program_evidence"]["records"], key
        assert (cell.get("big_excluded") or {}).get("rows") == "esm_family", key
    assert t["program_evidence"]["c128_esm"]["kind"] == "in_model" and t["program_evidence"]["esm_big"]["kind"] == "in_model"
    assert n_rec >= 1500 and n_native >= 700 and n_tie_checked >= 500, (n_rec, n_native, n_tie_checked)
    # cc-8.0 cells that read native: c 128 (both directions, every bucket) and c 256 N <= 256, on every cc-8.0 column
    for key, cell in t["cells"].items():
        cc, prec, c, h, nb, d, mode = key.split("|")
        if cc == "8.0" and prec == "bf16" and mode == "fwd" and ((c, h) == ("C128", "H128") or ((c, h) == ("C256", "H256") and int(nb[3:]) <= 256)) and d in ("out", "in"):
            for st, w in cell["fast_per_stack"].items():
                rec = (cell.get("policy") or {}).get(st) or {}
                measured = rec.get("mode") == "run" and rec.get("run") in ("race_c2", "race_p2", "race_p3")      # this table's own cc-8.0 measurement on that column decides over the panel
                assert (w == "native" and cell["big_per_stack"][st] == "native") or measured or cell.get("kit_floor"), (key, st, w, rec.get("run"))   # a kit-floor cell reads the stock op
    # c_z 128: the in-model v4 pin is superseded by the policy (v4 keeps a column only where it beats native beyond noise there); exact untouched
    assert "superseded_by" in t["in_model"]["c128_v4"]
    for key, cell in t["cells"].items():
        cc, prec, c, h, nb, d, mode = key.split("|")
        if (c, h, mode) != ("C128", "H128", "fwd"):
            continue
        assert str(cell.get("in_model", "")).startswith("c128_v4"), key
        if cc == "8.0":                                                            # cc 8.0: exact = the stock op (in_model.c128_exact_a100) or native_exact where VOUCHED on that stack
            assert cell["exact"] in T.STOCK_ROWS or (cell["exact"] == "native_exact" and T.exact_vouched(cell, "native_exact", cell.get("ref_stack"))), (key, cell["exact"])
            for st, row in cell["exact_per_stack"].items():
                assert row in T.STOCK_ROWS or st in ((cell.get("in_form") or {}).get("stacks") or {}) or (row == "native_exact" and st in ((cell.get("vouched_on") or {}).get("native_exact") or ())), (key, st, row)
    assert T.select("9.0", "bf16", 128, 128, 400, "outgoing", word="esm_v5_fwd").row == "esm_v5_fwd"                 # row words serve the rows
    assert T.select("9.0", "bf16", 256, 256, 800, "outgoing", word="fast", prefer=("v4", "tmk3_fast")).row == "v4"
    assert T.select("8.0", "bf16", 256, 256, 800, "outgoing", word="fast", prefer=("v4",)).row == "v4"
    s = T.select("9.0", "bf16", 256, 256, 800, "outgoing", word="exact", has_cueq=False)                             # a stack without the library: the module math is the stock row
    assert s.row == "torch_math", s
    for n in (1536, 2048):                                                                                             # cc 8.0 c 256 exact >= 1536: an exact-class row, never the module math
        for d in ("outgoing", "incoming"):
            assert T.select("8.0", "bf16", 256, 256, n, d, word="exact").row in ("tmk3_exact", "native_exact"), (n, d)
    assert T.select("9.0", "fp32", 384, 384, 800, "incoming", word="exact", tf32=True).row in ("tmk3_exact", "native_exact")
    assert T.select("8.0", "fp32", 64, 64, 800, "outgoing", word="exact").row == "cueq"
    for cc in ("9.0", "8.0"):                                                                                                   # the ESM-family pair block with backward: ef2_fused fast, ef2_cueq_tiles exact where >= x1.00
        s = T.select(cc, "bf16", 256, 256, 400, "outgoing", word="fast", backward=True)
        assert s.row == "ef2_fused" and s.backward and s.cell.endswith("|outin|fwdbwd"), s
        assert T.select(cc, "bf16", 256, 256, 400, "incoming", word="exact", backward=True).row == "ef2_cueq_tiles"
    assert T.select("9.0", "bf16", 256, 256, 800, "incoming", word="exact", backward=True).row == "cueq"             # x0.99 at 800: below the floor
    top = max(int(k.split("|")[4][3:].split("+")[0]) for k in t["cells"] if k.startswith("9.0|bf16|C128|H128|") and k.endswith("|out|fwd"))
    s = T.select("9.0", "bf16", 128, 128, top + 904, "outgoing", word="fast")                                         # beyond the largest measured size (table-derived): the largest cell, flagged
    assert not s.size_measured and "beyond_measured" in s.reason, (top, s)
    assert t["exact_rule"].startswith("an exact-class row")


def T_admits(t, row, key, st):
    cc, prec, C, H, nb, d, mode = key.split("|")
    dtype = {"bf16": "bf16", "f32z_bf16": "bf16", "fp32": "fp32", "tf32": "fp32"}[prec]
    try:
        T.admits(row, cc, dtype, int(C[1:]), int(H[1:]), int(nb.split("+")[0][3:]), residency=("fp32" if prec == "f32z_bf16" else None), backward=(mode == "fwdbwd"),
                 tf32=(True if prec == "tf32" else None), triton=(st.split("/")[1] if "/" in st else None), has_cueq=(not st.endswith("nocueq")))
        return True
    except (T.Refusal, ValueError):
        return False
def test_an_unmeasured_capability_inherits_the_nearest_measured_columns_portable_rows():
    """NOREG glue: a capability with no column in the table (10.0 / 10.3 / 12.0, an 8.x part other than 8.0) reads the nearest measured
    architecture-compatible capability's cells -- fast / big = that column's PORTABLE rows (v4, the tmk3 pair; the native payload only where its
    build record carries the architecture: today it refuses 10.x / 12.0 / 8.6 by name), never an sm_90a cubin row, never the library op while a
    portable row admits; exact = the stock op by name (no byte vouch on that capability); the reason / census carry inherited_cc:unmeasured."""
    FAM = ("bf16", 128, 128, "out", "fwd")                                                  # cell-independent: a synthetic capability (12.9) that never has cells; per-family expectations from the table
    top9 = max((c for c in T.measured_ccs() if float(c) < 12.9 and T._has_family(c, FAM)), key=float)
    assert T.inherited_cc("12.9", family=FAM) == top9 and T.inherited_cc("8.6", family=FAM) == "8.0" and T.inherited_cc("7.5") is None
    assert T.inherited_cc("9.0") is None and T.inherited_cc("8.0") is None
    B = "B300:2.9.0+cu129/3.5.0/cueq0.7.0"
    CUBIN = ("tx_sm90a", "tx_sm90a_exact", "esm_v61")
    for cc, st in (("12.9", B), ("12.0", B), ("8.6", "A10G:2.13.0+cu130/3.7.1/cueq0.11.1")):
        for (C, H) in ((128, 128), (256, 256), (384, 384), (64, 64), (64, 128)):
            for N in (400, 1536, 2048):
                for d in ("outgoing", "incoming"):
                    for word in ("fast", "big"):
                        s = T.select(cc, "bf16", C, H, N, d, word=word, stack=st, has_cueq=True)
                        assert s.row in T.INHERIT_PORTABLE_ROWS + T.STOCK_ROWS and s.row not in CUBIN, (cc, C, H, N, d, word, s)
                        if s.row in T.STOCK_ROWS and s.cell:                                              # the stock op only when the donor cell's order carries no live portable row
                            donor = T.table()["cells"][s.cell]; dead = {r for (r, c_) in (T._TABLE.get("dead_rows") or set()) if c_ == cc} if isinstance(T._TABLE.get("dead_rows"), (set, dict, list)) else set()
                            live = [r for r in [donor.get(word)] + list(donor.get("fast_order") or []) if r in T.INHERIT_PORTABLE_ROWS and r not in dead]
                            assert not live or cc == "8.6", (cc, C, H, N, d, word, s, live)
                        icc = T.inherited_cc(cc, family=("bf16", C, H, "out" if d == "outgoing" else "in", "fwd"))
                        if T.cell_family(cc, "bf16", C, H, "out" if d == "outgoing" else "in", "fwd"): continue     # a family some writer measured on this capability: its own cells (not this test's subject)
                        assert "inherited_cc:unmeasured(%s->%s)" % (cc, icc) in s.reason and s.cell.startswith(icc + "|") and not s.size_measured, s
                        if s.row.startswith("native"):                                   # only where the payload's build record carries the architecture
                            from opt_core.kernels.trimul import native as NT
                            assert NT.admits(cc, "bf16", C, H, N, d) is None, (cc, s)
                    e = T.select(cc, "bf16", C, H, N, d, word="exact", stack=st, has_cueq=True)
                    assert e.row == ("cueq" if C == H else "torch_math") and "exact_vouch_not_recorded_on_cc:%s" % cc in e.reason, (cc, C, H, N, d, e)
        with pytest.raises(T.Refusal) as ex:                                                 # an sm_90a prebuilt row BY NAME off 9.0: refused by name (unchanged)
            T.select(cc, "bf16", 256, 256, 800, "outgoing", word="tx_sm90a", stack=st)
        assert ex.value.kind.startswith("cc:")
    assert T.select("12.9", "fp32", 128, 128, 800, "outgoing", word="fast", stack=B, tf32=True).row in T.INHERIT_PORTABLE_ROWS
    # measured capabilities: untouched (a listed column reads itself; no token)
    s9 = T.select("9.0", "bf16", 256, 256, 800, "outgoing", word="fast", stack="H100:2.13.0+cu130/3.7.1/cueq0.11.1", abi="torch2.13.0+cu130-cpython-312-x86_64-linux-gnu-sm90")
    assert "inherited_cc" not in s9.reason and s9.cell.startswith("9.0|") and T.select("8.0", "bf16", 256, 256, 800, "outgoing", word="exact").cell.startswith("8.0|")


def test_cc_10_3_with_ONE_injected_cell_still_inherits_the_portable_row_for_every_OTHER_key():
    """NOREG glue, PER-KEY inheritance (RETIRE greps this name; cell-independent: the synthetic capability 12.9 stands in for 'a newer part',
    whatever real 10.x cells the table carries): a capability that owns cells for ONE family key (an injected 12.9 c_z-256 cell) serves THAT key
    from its own cell and every OTHER family key (c_z 128 / 384 / (64,64) / (64,128), the other direction) by inheritance from the nearest
    measured capability below it that has the family -- never the stock op merely because the capability has 'some' cells."""
    import copy
    t = T.table(); cells = t["cells"]
    B = "B300:2.13.0+cu130/3.7.1/cueq0.11.1"
    inj = "12.9|bf16|C256|H256|N<=800|out|fwd"
    cells.pop(inj, None); T._FAMILIES.clear()
    src = cells["9.0|bf16|C256|H256|N<=800|out|fwd"]
    fake = copy.deepcopy(src); fake["fast"] = fake["big"] = "v4"; fake["exact"] = "cueq"
    fake["fast_per_stack"] = {B: "v4"}; fake["big_per_stack"] = {B: "v4"}; fake["exact_per_stack"] = {B: "cueq"}; fake["ref_stack"] = B; fake["fast_order"] = ["v4", "cueq"]
    try:
        cells[inj] = fake; T._FAMILIES.clear()
        assert "12.9" in T.measured_ccs()
        assert T.inherited_cc("12.9") is None                                                        # capability-level: it has cells
        fam128 = ("bf16", 128, 128, "out", "fwd")
        near = T.inherited_cc("12.9", family=fam128)
        assert near is not None and float(near) < 12.9 and T._has_family(near, fam128)                 # per key: the nearest measured capability below with the c_z-128 family
        own = T.select("12.9", "bf16", 256, 256, 800, "outgoing", word="fast", stack=B)
        assert own.cell == inj and own.row == "v4" and "inherited_cc" not in own.reason, own          # the injected key: its own cell
        for (C, H) in ((128, 128), (384, 384), (64, 64), (64, 128)):
            for N in (400, 2048):
                for d in ("outgoing", "incoming"):
                    fam = ("bf16", C, H, "out" if d == "outgoing" else "in", "fwd")
                    icc = T.inherited_cc("12.9", family=fam)
                    assert icc is not None, fam
                    for word in ("fast", "big"):
                        s = T.select("12.9", "bf16", C, H, N, d, word=word, stack=B, has_cueq=True)
                        assert s.row in T.INHERIT_PORTABLE_ROWS or s.row in T.STOCK_ROWS, (C, H, N, d, word, s)     # a portable row of the donor cell (or the stock op by name when the donor cell names none)
                        assert "inherited_cc:unmeasured(12.9->%s),partial_cc" % icc in s.reason and s.cell.startswith(icc + "|"), s
                    e = T.select("12.9", "bf16", C, H, N, d, word="exact", stack=B, has_cueq=True)
                    assert e.row in T.STOCK_ROWS and "exact_vouch_not_recorded_on_cc:12.9" in e.reason, e
        inn = T.select("12.9", "bf16", 256, 256, 800, "incoming", word="fast", stack=B)               # the OTHER direction of the injected width: no 12.9 cell -> inherited
        assert not inn.cell.startswith("12.9|") and "partial_cc" in inn.reason, inn
    finally:
        cells.pop(inj, None); T._FAMILIES.clear()


def test_an_inherited_row_whose_launch_raises_steps_aside_by_name_and_is_not_retried():
    """LAUNCH GUARD (NOREG glue): on an unmeasured capability the inherited portable row is guarded at its first launch -- a build / compile / launch
    failure (any exception but out-of-memory) marks the row unavailable for the process on that capability (census / step-aside token
    stepped_aside:error:<ExcType>, said once), the call serves the NEXT portable row of the donor cell else the stock op by name, and a second call
    does not retry the dead row; OOM re-raises; a measured selection re-raises (unchanged)."""
    B = "B300:2.9.0+cu129/3.5.0/cueq0.7.0"
    T._TABLE.pop("dead_rows", None)
    try:
        s1 = T.select("12.9", "bf16", 128, 128, 800, "outgoing", word="fast", stack=B, has_cueq=True)
        assert s1.row in T.INHERIT_PORTABLE_ROWS and "inherited_cc:unmeasured" in s1.reason, s1
        tried, cache = [], {}
        r1 = s1.row
        assert r1 in T.INHERIT_PORTABLE_ROWS, s1
        assert T._inherited_step_aside(s1, RuntimeError("no kernel image is available for execution on the device"), tried, cache, "12.9") is True
        assert (r1, "12.9") in T._TABLE["dead_rows"] and tried == [r1] and cache["_stepaside"] == ["%s(stepped_aside:error:RuntimeError)" % r1], (tried, cache)
        s2 = T.select("12.9", "bf16", 128, 128, 800, "outgoing", word="fast", stack=B, has_cueq=True)          # the second call: the dead row is not retried
        assert s2.row != r1 and s2.row in T.INHERIT_PORTABLE_ROWS + T.STOCK_ROWS and "inherited_cc:unmeasured" in s2.reason, s2
        n = 0
        while s2.row not in T.STOCK_ROWS and n < 8:                                                             # every portable row dying in turn ends at the stock op BY NAME
            assert T._inherited_step_aside(s2, RuntimeError("boom"), tried, cache, "12.9") is True
            s2 = T.select("12.9", "bf16", 128, 128, 800, "outgoing", word="fast", stack=B, has_cueq=True); n += 1
        assert s2.row == "cueq" and n >= 1, (s2, n)
        assert T.select("12.9", "bf16", 256, 256, 800, "outgoing", word="fast", stack=B, has_cueq=True).row != r1     # dead per (row, capability): every family on that capability
    finally:
        T._TABLE.pop("dead_rows", None)
    # re-raise cases
    s9 = T.select("9.0", "bf16", 128, 128, 800, "outgoing", word="fast")
    with pytest.raises(RuntimeError):
        T._inherited_step_aside(s9, RuntimeError("measured selection: unchanged"), [], {}, "9.0")
    s1 = T.select("12.9", "bf16", 128, 128, 800, "outgoing", word="fast", stack=B, has_cueq=True)
    class OutOfMemoryError(RuntimeError): pass
    with pytest.raises(OutOfMemoryError):
        T._inherited_step_aside(s1, OutOfMemoryError("CUDA out of memory"), [], {}, "12.9")
    with pytest.raises(T.Refusal):
        T._inherited_step_aside(s1, T.Refusal("shape:n=3", "v4", "cueq"), [], {}, "12.9")
    assert not T._TABLE.get("dead_rows")
