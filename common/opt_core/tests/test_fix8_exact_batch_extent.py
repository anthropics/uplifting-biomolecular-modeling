"""0.5.212.5: under the EXACT word kernels.trimul.select() treats the leading batch extent as a fact of the call
class.  Every exact-class row's byte vouch (`bitwise=cueq`, vouched_on) is a BATCH-1 vouch (one pair tensor [1, N, N, c]); the library op's own
batched call ([B, N, N, c], B > 1 -- OpenFold3's confidence head over the 5 diffusion samples) is not B calls of one sample byte for byte, so a
row served at B > 1 on a batch-1 vouch broke byte identity (openfold3 identity item entity_protein_rna_1urn_u1a, cell 9.0|bf16|C128|H128|N<=256,
tmk3_exact).  With `batch=` > 1 stated the exact word is the stock op of THIS process BY NAME (note exact_batch_unvouched) unless the cell records
the row's batched vouch on the stack (cells.<key>.vouched_on_batched -- none today).  batch unstated / 1, fast / big / v4 / row words: unchanged.
CPU only: pure table resolution + the census grammar + the binding's call site."""
import copy, inspect, re
import pytest
from opt_core.kernels import trimul as T
from opt_core import cell_census as CC
import opt_core.trimul as BIND

OF3 = "H100:2.10.0+cu128/3.6.0/cueq0.10.0"           # openfold3 / openfold3_ob0 image stack word (trimul / ln pieces)
PTX2 = "H100:2.13.0+cu130/3.7.1/cueq0.11.1"          # protenix_v2 / rosettafold3 / boltzgen image stack word
B2 = "H100:2.12.0+cu130/3.7.0/cueq0.10.0"            # boltz2 image stack word (the C128 cells' reference column)
NOCUEQ = "H100:2.13.0+cu130/3.7.1/nocueq"            # a library-less stack word
NOTE = "exact_batch_unvouched"


def _sel(N, direction="outgoing", C=128, **kw):
    return T.select(9.0, "bf16", C, C, N, direction, word="exact", **kw)


def test_the_openfold3_confidence_head_class_is_the_stock_op_at_batch_5():
    """[5, 118, 118, 128] bf16 on the openfold3 stack: tmk3_exact at batch 1 (the trunk), the library op BY NAME at batch 5 (the head)."""
    for d in ("outgoing", "incoming"):
        one = _sel(118, d, stack=OF3)
        assert one.row == "tmk3_exact" and str(one.cls).startswith("bitwise"), (d, one.row, one.cls)
        assert _sel(118, d, stack=OF3, batch=1) == one                       # batch 1 == unstated, field for field
        assert _sel(118, d, stack=OF3, batch=None) == one
        five = _sel(118, d, stack=OF3, batch=5)
        assert five.row == "cueq" and str(five.cls).startswith("stock"), (d, five.row, five.cls, five.reason)
        assert "%s(tmk3_exact->cueq:B5)" % NOTE in five.reason, five.reason
        assert five.cell == one.cell                                          # the same table cell: the batch extent is a fact of the call, not a new cell


def test_every_served_exact_class_row_steps_to_this_process_stock_op_at_batch_gt_1():
    """Across the table's library stacks and every C==H bf16 size bucket: wherever the exact word SERVES an exact-class row at batch 1, the
    same call at batch 2 / 5 / 8 is `cueq` (the library op = the engines' stock op on a library stack) with the note; wherever batch 1 already
    answers a stock row (N <= 100, N >= 513 on C128, unvouched stacks ...) the batched answer is the identical Selection."""
    n_changed = n_same = 0
    for st in (OF3, PTX2, B2, "H100:2.7.1+cu128/3.3.1/cueq0.10.0", "H100:2.7.1+cu126/3.3.1/cueq0.8.0", "A100:2.7.1+cu126/3.3.1/cueq0.10.0"):
        for C in (128, 256):
            for N in (2, 24, 100, 101, 118, 128, 200, 256, 257, 400, 448, 512, 513, 700, 995, 1200, 1201, 1400, 1536, 2048, 2049, 4096):
                for d in ("outgoing", "incoming"):
                    one = _sel(N, d, C, stack=st)
                    for b in (2, 5, 8):
                        got = _sel(N, d, C, stack=st, batch=b)
                        if one.row in T.EXACT_ROWS:
                            n_changed += 1
                            assert got.row == "cueq" and str(got.cls).startswith("stock"), (st, C, N, d, b, one.row, got.row, got.reason)
                            assert "%s(%s->cueq:B%d)" % (NOTE, one.row, b) in got.reason, got.reason
                        else:
                            n_same += 1
                            assert got == one, (st, C, N, d, b, one, got)
                            assert NOTE not in got.reason
    assert n_changed > 100 and n_same > 100, (n_changed, n_same)


def test_fast_big_v4_and_row_words_ignore_the_batch_extent():
    for st in (OF3, PTX2, B2, NOCUEQ, None):
        for C in (128, 256):
            for N in (2, 100, 101, 118, 256, 400, 512, 513, 1200, 1400, 2048, 4096):
                for d in ("outgoing", "incoming"):
                    for word in ("fast", "big", "v4", "tmk3_exact", "native_exact", "cueq", "torch_math"):
                        for res, tf32 in ((None, None), ("fp32", None), (None, True)):
                            kw = dict(word=word, stack=st, residency=res, tf32=tf32)
                            try:
                                one = T.select(9.0, "bf16" if not tf32 else "fp32", C, C, N, d, **kw)
                            except T.Refusal as e1:
                                with pytest.raises(T.Refusal) as e5:
                                    T.select(9.0, "bf16" if not tf32 else "fp32", C, C, N, d, batch=5, **kw)
                                assert (e5.value.kind, e5.value.row, e5.value.fallback) == (e1.kind, e1.row, e1.fallback)
                                continue
                            assert T.select(9.0, "bf16" if not tf32 else "fp32", C, C, N, d, batch=5, **kw) == one, (st, C, N, d, word, res, tf32)


def test_unvouched_stack_beyond_envelope_library_less_and_planning_calls():
    # a stack the batch-1 vouch does not cover keeps ITS OWN refusal word (census named_fallback unchanged): no batch note
    s = _sel(118, stack=NOCUEQ, batch=5)
    assert s.row == "cueq" and "skip native_exact(exact_vouch_not_recorded_on:%s)" % NOCUEQ in s.reason and NOTE not in s.reason, s.reason
    assert _sel(118, stack=NOCUEQ, batch=5) == _sel(118, stack=NOCUEQ)
    # beyond the family's largest bucket: unchanged (the stock op already)
    assert _sel(9000, stack=OF3, batch=5) == _sel(9000, stack=OF3)
    # has_cueq False (a library-less kit statement): torch_math, unchanged
    assert _sel(300, stack=PTX2, has_cueq=False, batch=5) == _sel(300, stack=PTX2, has_cueq=False)
    assert _sel(300, stack=PTX2, has_cueq=False, batch=5).row == "torch_math"
    # c_hidden != c_z (the template width): torch_math, unchanged
    a = T.select(9.0, "bf16", 64, 128, 300, "outgoing", word="exact", stack=OF3)
    assert T.select(9.0, "bf16", 64, 128, 300, "outgoing", word="exact", stack=OF3, batch=5) == a and a.row == "torch_math"
    # a planning call (no stack) that states a batch: the stock op by name (no vouch record can match without a stack)
    p = _sel(118, batch=5)
    assert p.row == "cueq" and "%s(" % NOTE in p.reason, (p.row, p.reason)
    assert _sel(118).row in T.EXACT_ROWS                                      # ... and without the batch: the column word, as before
    # prefer= an exact row at batch 5: the stock op (the preferred row is not eligible for the batched layout)
    q = _sel(118, stack=OF3, batch=5, prefer=("tmk3_exact",))
    assert q.row == "cueq", (q.row, q.reason)
    assert _sel(118, stack=OF3, prefer=("tmk3_exact",)).row == "tmk3_exact"


def test_a_recorded_batched_vouch_admits_the_row_at_that_extent_only(monkeypatch):
    tab = T.table()
    key = "9.0|bf16|C128|H128|N<=256|out|fwd"
    cells = copy.deepcopy({key: tab["cells"][key]})
    cells[key][T.BATCH_VOUCH_KEY] = {"tmk3_exact": {OF3: [5]}}
    patched = dict(tab); patched["cells"] = dict(tab["cells"]); patched["cells"].update(cells)
    monkeypatch.setattr(T, "table", lambda: patched)
    assert _sel(118, stack=OF3, batch=5).row == "tmk3_exact"                  # the recorded extent on the recorded stack
    assert _sel(118, stack=OF3, batch=2).row == "cueq"                        # another extent: not recorded
    assert _sel(118, "incoming", stack=OF3, batch=5).row == "cueq"            # another cell (in): not recorded
    assert _sel(118, stack=PTX2, batch=5).row == "cueq"                       # another stack: not recorded (its batch-1 winner native_exact steps to the library op)
    cells[key][T.BATCH_VOUCH_KEY] = {"tmk3_exact": {OF3: "any"}}
    assert _sel(118, stack=OF3, batch=2).row == "tmk3_exact" and _sel(118, stack=OF3, batch=64).row == "tmk3_exact"


def test_batch_extent_and_exact_batch_vouched_words():
    assert T.batch_extent(None) is None and T.batch_extent(True) is None and T.batch_extent("x") is None
    assert T.batch_extent(1) == 1 and T.batch_extent(5) == 5 and T.batch_extent("5") == 5 and T.batch_extent(0) == 1 and T.batch_extent(-3) == 1
    cell = T.table()["cells"]["9.0|bf16|C128|H128|N<=256|out|fwd"]
    assert T.BATCH_VOUCH_KEY not in cell                                        # the shipped table records no batched vouch (every batched exact call is the stock op)
    assert not any(T.BATCH_VOUCH_KEY in c for c in T.table()["cells"].values())
    assert T.exact_batch_vouched(cell, "tmk3_exact", OF3, None) is True
    assert T.exact_batch_vouched(cell, "tmk3_exact", OF3, 1) is True
    assert T.exact_batch_vouched(cell, "tmk3_exact", OF3, 5) is False
    assert T.exact_batch_vouched(cell, "tmk3_exact", None, 5) is False
    assert T.exact_batch_vouched(None, "tmk3_exact", OF3, 5) is False


def test_census_a_batched_exact_class_is_its_own_key_and_names_the_row():
    CC.reset()
    _sel(118, stack=OF3); _sel(118, stack=OF3, batch=5); _sel(118, stack=OF3, batch=1)
    T.select(9.0, "bf16", 128, 128, 118, "outgoing", word="fast", stack=OF3, batch=5)
    _sel(60, stack=OF3, batch=5)                                                # N <= 100: the library op already (fix7 A2) -> the batch-1 key, no note
    _sel(118, stack=NOCUEQ, batch=5)                                            # unvouched stack: its own named_fallback token, unchanged
    toks = CC.tokens()
    b5 = [t for t in toks if ".B5 " in t]
    assert len(b5) == 1, toks
    assert re.match(r"^STOCK_CELL:trimul\|cc9\.0\|%s\|bf16\|C128H128\|N<=256\|out\.fwd\.B5 word=exact served=cueq cell=9\.0\|bf16\|C128\|H128\|N<=256\|out\|fwd note=%s\(tmk3_exact->cueq:B5\)$"
                    % (re.escape(OF3), NOTE), b5[0]), b5[0]
    assert any(t.startswith("CELL_HIT:trimul|cc9.0|%s|bf16|C128H128|N<=256|out.fwd word=exact served=tmk3_exact " % OF3) for t in toks), toks
    assert any(t.startswith("CELL_HIT:trimul|cc9.0|%s|bf16|C128H128|N<=256|out.fwd word=fast served=v4 " % OF3) for t in toks), toks   # fast at batch 5: the batch-1 key
    assert any(t.startswith("STOCK_CELL:trimul|cc9.0|%s|bf16|C128H128|N<=100|out.fwd word=exact served=cueq " % OF3) for t in toks), toks
    assert any(t.startswith("NAMED_FALLBACK:trimul|cc9.0|%s|bf16|C128H128|N<=256|out.fwd word=exact refused=native_exact:exact_vouch_not_recorded_on:" % NOCUEQ) for t in toks), toks
    assert not [t for t in CC.alerts() if ".B5" in t]                            # the batched stock answer is a STOCK_CELL record, never an alert token
    for t in toks:                                                              # every token parses under the census grammar (the harness reads them with it)
        assert CC.parse_token(t) is not None, t
    CC.reset()


def test_the_binding_and_the_serving_call_state_the_batch_extent():
    """opt_core.trimul.by_word's eligible() and kernels.trimul.triangle_multiplication's own selection state `batch=` from the tensor in hand
    (GPU-free check of the call sites; the GPU proof is the openfold3 identity twin)."""
    src = inspect.getsource(BIND.by_word)
    call = src[src.index("sel = KT.select("): src.index("KT.admits(sel.row")]      # eligible()'s one select call
    assert "batch=int(zs.shape[0])" in call, "by_word.eligible must pass batch= to KT.select: %r" % call
    tm = inspect.getsource(T.triangle_multiplication)
    assert "batch=int(B)" in tm and "exact_batch_vouched(" in tm
    assert "batch" in inspect.signature(T.select).parameters and "batch" in inspect.signature(T._select).parameters
