"""The triangle-multiplication rows of the FPF add-on (``rf3fpf/fpf_rf3_trimul_rows.py``, the shared core's provider ``opt_core.kernels.trimul``):
the arm sub-word grammar (``fast.<word>``, ``xmul[.<word>]``) leaves today's arms byte for byte; a row word outside a component's vocabulary is
refused by name; a row the provider refuses on this card / stack / shape is bound to its NAMED fallback and the refusal is printed and counted
(never a silent substitute); a fast-seam call no fused row admits declines with the adapter's own census word (``N<floor>``, ``c=<c>/d=<d>``); the
exact tier's seam nests with the triangle-attention module's proxy on the same module attribute and undoes only itself; the SELECT / CENSUS lines carry
the provider's own ``describe()``.  CPU only: selections over synthetic tensor facts.
facts = (cc, precision, c, d, n_tokens, direction)."""
import os
import re
import sys
import types

import pytest

from rosettafold3_opt import modes, stack

T = pytest.importorskip("opt_core.kernels.trimul")
RF3FPF = os.path.join(stack.fpf_home(), "rf3fpf")


@pytest.fixture
def R(monkeypatch):
    monkeypatch.syspath_prepend(RF3FPF)
    sys.modules.pop("fpf_rf3_trimul_rows", None)
    import fpf_rf3_trimul_rows as R
    R.reset_census()
    for c in R.STATE:
        R.STATE[c]["word"] = R.DEFAULT_WORD[c]; R.STATE[c]["on"] = False
    monkeypatch.setattr(R, "_stock", lambda: None)
    R._PROC.clear()
    yield R
    R.reset_census()
    sys.modules.pop("fpf_rf3_trimul_rows", None)


def test_todays_arms_are_untouched_by_the_sub_word_grammar(R):
    for arm in ("fast+gflash+ttr+apb+res@L1", "tg+sapb@L1", "fast+apb+res@L1", "fast", "stock+tg@L1", "fast+gflash.k2b+ttr@L1", "tg+sapb+xatt@L1"):
        assert R.strip_components(arm) == (arm, {"fast": None, "xmul": None}), arm


def test_sub_words_parse_and_unknown_words_are_refused_by_name(R):
    assert R.strip_components("fast.tx_sm90a+gflash+ttr+apb+res@L1") == ("fast+gflash+ttr+apb+res@L1", {"fast": "tx_sm90a", "xmul": None})
    assert R.strip_components("fast.tmk3_fast+gflash.k2b+ttr@L1") == ("fast+gflash.k2b+ttr@L1", {"fast": "tmk3_fast", "xmul": None})
    assert R.strip_components("tg+sapb+xmul@L1") == ("tg+sapb@L1", {"fast": None, "xmul": "exact"})
    assert R.strip_components("tg+sapb+xatt+xmul.tmk3_exact@L1") == ("tg+sapb+xatt@L1", {"fast": None, "xmul": "tmk3_exact"})
    assert R.strip_components("xmul@L1") == ("stock@L1", {"fast": None, "xmul": "exact"})
    assert R.parse_component("fast") == ("fast", None) and R.parse_component("fast.v4") == ("fast", "v4") and R.parse_component("ttr") == (None, None)
    assert R.parse_component("gflash.k2b") == (None, None)                                             # the triangle-attention module's word: not this module's
    for bad in ("fast.turbo", "xmul.v4", "fast.exact", "xmul.fast", "fast.", "fast.bz2", "xmul.bz2", "fast.eager", "fast.esm_v5_fwd:f32in"):   # bz2: the retired row word, refused by name
        with pytest.raises(ValueError):
            R.strip_components(bad + "+ttr@L1")
    with pytest.raises(R.WordError):
        R.set_fast_word("cueq")
    assert R.COMPONENT_WORDS["fast"][0] == "v4" and R.COMPONENT_WORDS["xmul"][0] == "exact"          # defaults: today's kernel; the cell's exact row
    assert set(R.COMPONENT_WORDS["fast"]) <= set(T.ROW_NAMES + T.TIER_WORDS) and set(R.COMPONENT_WORDS["xmul"]) <= set(T.ROW_NAMES + T.TIER_WORDS + ("eager",))   # eager: the kit's scope word


def test_the_adapter_parses_the_component_and_binds_the_provider():
    src = open(os.path.join(stack.fpf_home(), modes.FPF_ADAPTER_RELPATH), encoding="utf-8").read()
    assert re.search(r"[\"']xmul[\"']", src) and "TR.serve_fast(" in src and "TR.note_default(" in src and "TR.strip_components(" in src
    fast = src[src.index("def _fast_forward"):src.index("def enable(")]
    assert "G.trimul(" in fast and "G.supported(" in fast and 'TR.DEFAULT_WORD["fast"]' in fast        # the default word launches today's kernel, admitted by its own supported()


def test_default_fast_word_is_todays_v4_row(R, capsys):
    ent = R._select("fast", ((9, 0), "bf16", 128, 128, 400, "out"))
    assert ent["sel"].row == "v4" and ent["refusal"] is None
    out = capsys.readouterr().out
    assert "[fpf_rf3 trimul] SELECT component=fast asked=v4 cc=9.0 dtype=bf16 c=128 d=128 n=400 dir=out " + T.describe(ent["sel"]) in out
    assert "cell=9.0|bf16|C128|H128|N<=400|out|fwd" in out and "REFUSED" not in out
    ent8 = R._select("fast", ((8, 0), "bf16", 128, 128, 1200, "in"))
    assert ent8["sel"].row == "v4" and ent8["refusal"] is None and "cell=8.0|bf16|C128|H128|N<=1200|in|fwd" in capsys.readouterr().out


def test_tx_sm90a_on_sm80_is_refused_by_name_and_bound_to_v4(R, capsys):
    R.set_fast_word("tx_sm90a")                                                                          # the sm_90a package's row word on an sm_80 card: refused BY NAME, bound to its named fallback
    ent = R._select("fast", ((8, 0), "bf16", 128, 128, 800, "out"))
    assert ent["sel"].row == "v4" and ent["refusal"] is not None and ent["refusal"].row == "tx_sm90a" and ent["refusal"].kind.startswith("cc:8.0!=9.0")
    out = capsys.readouterr().out
    assert "REFUSED row=tx_sm90a kind=cc:8.0!=9.0" in out and "-> bound v4 by name" in out
    R._count("fast", ent); R._count("fast", ent)
    line = R.census_line("fast")
    assert line.startswith("[fpf_rf3 trimul] CENSUS component=fast word=tx_sm90a on=True served=2 rows=v4|bf16|C128|D128:2 refused=tx_sm90a:cc:8.0!=9.0")
    assert "->v4:2" in line and T.describe(ent["sel"]) in line


def test_the_template_track_is_served_by_the_rows_that_admit_it_and_declined_by_name_otherwise(R, capsys):
    # the contract, not the core's current per-cell winner: a row that does not admit (64,64) is REFUSED BY NAME and bound to its named fallback
    # (the stock statement when nothing admits it: the current census word c=64/d=64); a row or tier word that admits it is SERVED (no refusal).
    ent = R._select("fast", ((9, 0), "bf16", 64, 64, 400, "out"))                                       # v4 (today's kernel): d_pair 128 / 256 only
    assert ent["refusal"] is not None and ent["refusal"].row == "v4" and ent["refusal"].kind.startswith("c_z:64")
    assert f"REFUSED row=v4 kind={ent['refusal'].kind} -> bound {ent['sel'].row} by name" in capsys.readouterr().out
    if ent["sel"].row in R.STOCK_ROWS:
        assert R.decline_word(ent) == "c=64/d=64"                                                       # the stock statement under the current census word
    for word in ("tmk3_exact", "tmk3_fast", "fast", "big"):                                          # rows / tier words that admit the template track: served
        R.STATE["fast"]["word"] = word
        for cc in ((9, 0), (8, 0)):
            ent = R._select("fast", (cc, "bf16", 64, 64, 800, "in"))
            assert ent["refusal"] is None and ent["sel"].row not in R.STOCK_ROWS, (word, cc, T.describe(ent["sel"]))
            if word in ("tmk3_exact", "tmk3_fast"):
                assert ent["sel"].row == word                                                           # a ROW word serves exactly that row
            if word == "tmk3_exact":
                assert ent["sel"].cls.startswith("bitwise"), T.describe(ent["sel"])


def test_the_token_floor_declines_with_the_adapters_census_word(R):
    # row v4's floor is the provider table's per call class (TRIMUL_CELLS rows.v4.admits: n_min 101; n_min_cells 2 on the cc-9.0 c_z-128 classes the
    # provider serves at small N): below it the row is refused BY NAME, bound to the stock row, and the adapter's census
    # word is that floor, N<k> (report.TRIMUL_TOKEN_FLOOR).
    ent = R._select("fast", ((8, 0), "bf16", 128, 128, 60, "out"))                                       # an unlisted class (A100): the floor stays 101
    assert ent["sel"].row == "cueq" and ent["refusal"].kind == "n<101:below_n_min" and R.decline_word(ent) == "N<101"
    ent = R._select("fast", ((9, 0), "bf16", 128, 128, 60, "out"))                                       # a floored class: row v4 down to the table's 2 tokens
    assert ent["sel"].row == "v4" and ent["refusal"] is None and T.v4_n_min((9, 0), "bf16", 128, 128) == 2
    ent = R._select("fast", ((9, 0), "bf16", 128, 128, 1, "out"))                                        # below that floor: refused by name, the floor's word
    assert ent["sel"].row in R.STOCK_ROWS and ent["refusal"].row == "v4" and R.decline_word(ent) == "N<2"
    R.STATE["fast"]["word"] = "tx_sm90a"                                                                # c 256 only: refused by name for this model's widths, bound to v4
    ent = R._select("fast", ((9, 0), "bf16", 128, 128, 1200, "out"))
    assert ent["sel"].row == "v4" and ent["refusal"].row == "tx_sm90a" and ent["refusal"].kind == "c_z=128,c_hidden=128!=256"


def test_serve_fast_declines_by_name_before_touching_the_module(R):
    class X:                                                                                            # a tensor stand-in: facts only
        shape = (300, 300, 64); device = "cuda:0"; is_cuda = True
        def dim(self): return 3
    m = types.SimpleNamespace(d_pair=64, d_hidden=64, direction="outgoing")
    R._PROC[("cc", "cuda:0")] = (9, 0); R._PROC["stack"] = None
    import opt_core.kernels.trimul as KT
    orig = KT.call_precision
    KT.call_precision = lambda z: ("bf16", None)
    try:
        with pytest.raises(R.Declined) as e:
            R.serve_fast(m, X())
        assert e.value.word == "c=64/d=64" and R.census("fast")["declined"] == 1 and R.CENSUS["fast"]["declined:c=64/d=64"] == 1
    finally:
        KT.call_precision = orig


def test_exact_word_takes_the_cells_exact_row_per_card_and_size(R, capsys):
    # the contract: under the exact tier's seam the `exact` word binds a row whose measured class is the stock op's bits (bitwise...) or a named
    # stock row (cueq) — per card and size, whatever the core's table names today; never a tolerance-class row.
    n = 0
    for cc in ((9, 0), (8, 0)):
        for C, N in ((128, 400), (128, 800), (128, 1200), (128, 2048), (64, 400), (64, 1200)):
            ent = R._select("xmul", (cc, "bf16", C, C, N, "out")); n += 1
            sel = ent["sel"]
            assert ent["refusal"] is None, (cc, C, N, T.describe(sel))
            assert sel.row in R.STOCK_ROWS or str(sel.cls).startswith("bitwise"), (cc, C, N, T.describe(sel))
            if sel.row not in R.STOCK_ROWS:
                assert sel.x_stock is None or sel.x_stock >= 1.0, T.describe(sel)                      # an exact kernel row is adopted at >= x1.00 where the cell timed it; a byte-vouch-only cell records no timing (the class is the contract)
    out = capsys.readouterr().out
    assert out.count("SELECT component=xmul asked=exact") == n


def test_the_eager_scope_word_serves_the_exact_row_outside_graph_capture_and_the_library_op_inside_it(R, capsys):
    R.STATE["xmul"]["word"] = "eager"
    out_ = R._select("xmul", ((9, 0), "bf16", 128, 128, 400, "out"), capturing=False)
    cap = R._select("xmul", ((9, 0), "bf16", 128, 128, 400, "out"), capturing=True)
    tpl = R._select("xmul", ((9, 0), "bf16", 64, 64, 400, "out"), capturing=False)
    big = R._select("xmul", ((9, 0), "bf16", 128, 128, 1200, "out"), capturing=False)
    exactish = lambda sel: sel.row in R.STOCK_ROWS or str(sel.cls).startswith("bitwise")          # the exact tier's contract, whatever the cell names today
    assert out_ is not cap and cap["sel"].row == "cueq" and all(exactish(e["sel"]) for e in (out_, tpl, big)), [T.describe(e["sel"]) for e in (out_, cap, tpl, big)]
    assert cap["refusal"] is None and out_["key"][-1] == "eager" and cap["key"][-1] == "capture"
    o = capsys.readouterr().out
    assert "SELECT component=xmul asked=eager ctx=capture" in o and "keeps the library op by name" in o and "asked=eager ctx=eager" in o
    R._count("xmul", cap); R._count("xmul", out_)
    line = R.census_line("xmul")
    assert "c128d128n400out:capture:" in line and "cueq|bf16|C128|D128:" in line and ("%s|bf16|C128|D128:" % out_["sel"].row) in line


def test_calls_at_or_below_the_librarys_fallback_threshold_go_to_the_stock_op_by_name(R, monkeypatch):
    assert R.cueq_fallback_threshold(default=77) in (77, 100)                                          # absent module -> the default (or the library's constant)
    R._PROC["thr"] = 100
    calls = []
    lib = types.SimpleNamespace(triangle_multiplicative_update=lambda x, **k: calls.append(int(x.shape[-2])) or "lib-out")
    monkeypatch.setattr(R, "_stock", lambda: lib.triangle_multiplicative_update)
    monkeypatch.setattr(R, "serve_xmul", lambda *a, **k: "row-out")
    px = R.CuetTrimulProxy(lib)
    w = dict(norm_in_weight=1, norm_in_bias=1, p_in_weight=1, g_in_weight=1, norm_out_weight=1, norm_out_bias=1, p_out_weight=1, g_out_weight=1)
    small = types.SimpleNamespace(shape=(1, 57, 57, 128)); big = types.SimpleNamespace(shape=(1, 101, 101, 128))
    assert px.triangle_multiplicative_update(small, direction="outgoing", **w) == "lib-out" and calls == [57]
    assert px.triangle_multiplicative_update(big, direction="outgoing", **w) == "row-out" and calls == [57]
    assert R.CENSUS["xmul"] == {"passthrough:N<=100": 1} and R.census("xmul")["served"] == 0
    m = types.SimpleNamespace(d_pair=128, d_hidden=128, direction="outgoing")
    x = types.SimpleNamespace(shape=(1, 100, 100, 128), dim=lambda: 4, is_cuda=False)
    with pytest.raises(R.Declined) as e:
        R.serve_fast(m, x)
    assert e.value.word == "N<101" and R.CENSUS["fast"] == {"declined:N<101": 1}                     # the current census word for the token floor


def test_the_esm_family_rows_are_fast_tier_words_and_refused_by_class_under_xmul(R, capsys):
    # measurement / ablation words (the modes bind TriMul through the tier words only): under the fast seam an ESM row word is SERVED where the
    # row admits the call, else REFUSED BY NAME and bound to its named fallback (it steps aside by name when the row retires); under the exact
    # tier's seam (xmul) a tolerance-class row is REFUSED BY CLASS -> the stock op by name.  Asserted by class, not by the core's current per-cell winner.
    assert R.parse_component("fast.esm_v5_fwd") == ("fast", "esm_v5_fwd") and R.parse_component("xmul.esm_v61") == ("xmul", "esm_v61")
    for word in ("esm_v5_fwd", "esm_v61"):
        R.STATE["fast"]["word"] = word
        for C, N in ((128, 400), (64, 800)):
            ent = R._select("fast", ((9, 0), "bf16", C, C, N, "out"))
            r = ent["refusal"]
            assert ent["sel"].row == word or (r is not None and r.row in (word, r.row) and ent["sel"].row != word), (word, C, T.describe(ent["sel"]))
            if r is not None:
                assert f"REFUSED row={r.row} kind={r.kind} -> bound {ent['sel'].row} by name" in capsys.readouterr().out
        R.STATE["xmul"]["word"] = word                                                                   # named under the exact tier's seam
        entx = R._select("xmul", ((9, 0), "bf16", 128, 128, 400, "out"))
        assert entx["refusal"] is not None and (entx["sel"].row in R.STOCK_ROWS or str(entx["sel"].cls).startswith("bitwise")), T.describe(entx["sel"])
        assert "-> bound %s by name" % entx["sel"].row in capsys.readouterr().out
    R.STATE["xmul"]["word"] = "tmk3_exact"                                                              # a bitwise row passes the class guard
    ent = R._select("xmul", ((9, 0), "bf16", 128, 128, 400, "in"))
    assert ent["refusal"] is None and ent["sel"].row == "tmk3_exact" and ent["sel"].cls.startswith("bitwise")


def test_a_row_word_refused_on_the_card_binds_the_stock_op_by_name(R, capsys):
    R.STATE["xmul"]["word"] = "tmk3_exact"
    ent = R._select("xmul", ((7, 5), "bf16", 128, 128, 400, "out"))
    assert ent["sel"].row == "cueq" and ent["refusal"].kind == "cc:7.5<8.0" and ent["refusal"].row == "tmk3_exact"
    assert "REFUSED row=tmk3_exact kind=cc:7.5<8.0 -> bound cueq by name" in capsys.readouterr().out
    R._count("xmul", ent)
    R.CENSUS["xmul"]["error:tmk3_exact"] = 3
    line = R.census_line("xmul")
    assert " served=1 rows=cueq|bf16|C128|D128:1 refused=tmk3_exact:cc:7.5<8.0->cueq:1 errors=3(tmk3_exact:3) selections=" in line
    assert R.census("xmul")["errors"] == 3


def test_selection_is_memoised_per_shape_and_word(R, capsys):
    a = R._select("fast", ((9, 0), "bf16", 128, 128, 400, "out"))
    b = R._select("fast", ((9, 0), "bf16", 128, 128, 400, "out"))
    c = R._select("fast", ((9, 0), "bf16", 128, 128, 400, "in"))
    d = R._select("fast", ((9, 0), "bf16", 128, 128, 1200, "out"))
    assert a is b and a is not c and c is not d
    assert capsys.readouterr().out.count("SELECT component=fast") == 3


def test_the_exact_seam_nests_with_another_components_proxy_and_undoes_only_itself(R, monkeypatch):
    calls = []
    lib = types.SimpleNamespace(triangle_multiplicative_update=lambda x, **k: calls.append("lib") or "lib-out", triangle_attention=lambda *a, **k: "lib-att")

    class Foreign(object):                                                                              # the triangle-attention module's proxy shape: _real + __getattr__
        def __init__(self, real): self._real = real
        def triangle_attention(self, *a, **k): return "routed-att"
        def __getattr__(self, n): return getattr(self._real, n)
    A = types.ModuleType("rf3.model.layers.attention"); A.cuet = lib
    for name, mod in (("rf3", types.ModuleType("rf3")), ("rf3.model", types.ModuleType("rf3.model")), ("rf3.model.layers", types.ModuleType("rf3.model.layers")),
                      ("rf3.model.layers.attention", A)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(R, "_stock", lambda: lib.triangle_multiplicative_update)
    # (a) this seam outermost over the foreign proxy
    A.cuet = Foreign(lib)
    assert R.enable_xmul() == "exact" and isinstance(A.cuet, R.CuetTrimulProxy) and isinstance(A.cuet._real, Foreign)
    assert A.cuet.triangle_attention() == "routed-att" and A.cuet.triangle_multiplicative_update.__self__ is A.cuet
    R.enable_xmul("cueq")                                                                               # idempotent: one proxy, the word updated
    assert isinstance(A.cuet, R.CuetTrimulProxy) and not isinstance(A.cuet._real, R.CuetTrimulProxy) and R.STATE["xmul"]["word"] == "cueq"
    R.disable_xmul()
    assert isinstance(A.cuet, Foreign) and A.cuet._real is lib and R.STATE["xmul"]["on"] is False
    # (b) this seam beneath the foreign proxy (the other component wrapped after us): disable unlinks ours, theirs stays
    A.cuet = lib
    R.enable_xmul()
    A.cuet = Foreign(A.cuet)
    R.disable_xmul()
    assert isinstance(A.cuet, Foreign) and A.cuet._real is lib
    # (c) no cuet at all: refused by name
    A.cuet = None
    with pytest.raises(RuntimeError):
        R.enable_xmul()


def test_a_tier_word_serves_the_pair_track_only_the_template_track_is_declined_by_todays_word(R):
    """Under a TIER word (fast / big / exact) the fast head word's site is the pair track (PAIR_SITE = c_z = d_hidden = 128); a template-track
    module (64, 64) is declined before any selection with the current census word c=64/d=64 (the stock statement runs it: xmul's site).  A ROW word
    is a measurement word and keeps serving whatever its row admits (selection-level tests above)."""
    import types
    torch = pytest.importorskip("torch")
    R.set_fast_word("fast")
    m = types.SimpleNamespace(d_pair=64, d_hidden=64, direction="outgoing")
    x = torch.zeros(1, 400, 400, 64)
    with pytest.raises(R.Declined) as e:
        R.serve_fast(m, x)
    assert str(e.value) == "c=64/d=64" and R.CENSUS["fast"]["declined:c=64/d=64"] == 1
    assert "declined=c=64/d=64:1" in R.census_line("fast")


# ---------------------------------------------------------------------------------------------- the fast seam's token floor is the provider table's
RF3_H100_STACK = "H100:2.13.0+cu130/3.7.1/cueq0.11.1"                                                   # this kit's H100 stack column of the core's cell table
FLOOR_SIZES = (1, 2, 12, 64, 100, 101, 128, 512)


def _fast_seam_over_sizes(R, monkeypatch, word, cc=(9, 0), site=(128, 128), direction="outgoing", old_gate=False, stack=RF3_H100_STACK):
    """{N: ("declined", <census word>) | ("served", <row>, <cell>, <the serving call's arguments less tensors>)} for ONE fast-seam call per size under
    the fast head word `word` -- CPU: the live-call facts are (cc, bf16, `stack`) and the provider's serving call is a recorder (no launch).
    old_gate=True runs the seam as it stood through rosettafold3_opt 0.2.11.6 (heads <= #5): the library threshold + 1 = 101 as the one floor for
    every word and class (fast_floor's answer replaced by that constant; every other statement is the same object code)."""
    import opt_core.kernels.trimul as KT
    R.reset_census(); R._PROC.clear()
    R.STATE["fast"]["word"] = word
    R._PROC[("cc", "cuda:0")] = tuple(cc); R._PROC["stack"] = stack; R._PROC["thr"] = 100
    monkeypatch.setattr(KT, "call_precision", lambda z: ("bf16", None))
    calls = []

    def recorder(z, mask=None, *, direction, weights, selection=None, residual=False, stock=None, cache=None, pad=None, **kw):
        calls.append(dict(n=int(z.shape[-2]), row=selection.row, word=selection.word, cell=selection.cell, config=repr(getattr(selection, "config", None)),
                          direction=direction, residual=bool(residual), pad=pad, stock=stock, mask=mask, weights=sorted(weights), extra=sorted(kw)))
        return "served"
    monkeypatch.setattr(KT, "triangle_multiplication", recorder)
    real_floor = R.__dict__.setdefault("_TEST_REAL_FAST_FLOOR", R.fast_floor)                            # the module's own, captured before any stand-in
    monkeypatch.setattr(R, "fast_floor", (lambda m, x, thr=None: (R.cueq_fallback_threshold() if thr is None else int(thr)) + 1) if old_gate else real_floor)
    torch = pytest.importorskip("torch")
    C, D = site
    t = lambda *shape: types.SimpleNamespace(weight=torch.zeros(*shape), bias=torch.zeros(shape[0]))
    m = types.SimpleNamespace(d_pair=C, d_hidden=D, direction=direction, norm_in=t(C), p_in=t(2 * D, C), g_in=t(2 * D, C), norm_out=t(D), p_out=t(C, D), g_out=t(C, C))

    class X:                                                                                             # a CUDA pair tensor stand-in: facts only, nothing allocated
        device = "cuda:0"; is_cuda = True; dtype = torch.bfloat16
        def __init__(self, n): self.shape = (1, n, n, C)
        def dim(self): return 4
    out = {}
    for n in FLOOR_SIZES:
        before = len(calls)
        try:
            r = R.serve_fast(m, X(n), residual=True, pad=16)
        except R.Declined as d:
            assert len(calls) == before, (word, n)                                                       # a declined call never reached the serving call
            out[n] = ("declined", d.word)
            continue
        assert r == "served" and len(calls) == before + 1, (word, n)
        c = calls[-1]; assert c["n"] == n
        out[n] = ("served", c["row"], c["cell"], tuple(sorted((k, repr(v)) for k, v in c.items() if k != "n")))
    return out


def test_head6_the_tier_words_serve_the_pair_track_from_the_tables_floor_and_nothing_at_or_above_101_changes(R, monkeypatch):
    """Under the modes' TIER words (arm fast.fast / fast.big) the fast seam's token floor
    is row v4's per-class floor from the core's ONE table (kernels.trimul.v4_n_min: 2 on 9.0|bf16|C128|H128 and 9.0|f32z_bf16|C128|H128, the classes
    the provider serves at small N; 101 everywhere else) instead of the library threshold + 1 for every class: 2..100-token pair-track
    calls on cc 9.0 are SERVED by the cell's row (v4) where heads <= #5 declined them (N<101 -> the stock statement -> the library's torch path);
    N = 1 declines as N<2.  EVERYTHING AT OR ABOVE 101 TOKENS IS THE SAME CALL AS BEFORE -- same row, same cell, same serving arguments -- for
    every word, card and track; so is every A100 call, every template-track call and every ROW-word call at every size."""
    from rosettafold3_opt import report
    tier_row = lambda wd, n, d: (T.table()["cells"]["9.0|bf16|C128|H128|N<=100|%s|fwd" % d].get(wd + "_per_stack") or {}).get(RF3_H100_STACK) or T.table()["cells"]["9.0|bf16|C128|H128|N<=100|%s|fwd" % d][wd]
    seen_served_small = 0
    for word in ("fast", "big"):
        for direction, dw in (("outgoing", "out"), ("incoming", "in")):
            old = _fast_seam_over_sizes(R, monkeypatch, word, direction=direction, old_gate=True)
            new = _fast_seam_over_sizes(R, monkeypatch, word, direction=direction)
            for n in FLOOR_SIZES:
                if n >= 101:
                    assert new[n] == old[n] and new[n][0] == "served", (word, direction, n, old[n], new[n])   # byte for byte the call it has always been
                elif n >= 2:
                    assert old[n] == ("declined", "N<101"), (word, direction, n, old[n])
                    assert new[n][0] == "served" and new[n][1] == tier_row(word, n, dw) == "v4", (word, direction, n, new[n])
                    assert new[n][2] == "9.0|bf16|C128|H128|N<=100|%s|fwd" % dw, new[n]
                    seen_served_small += 1
                else:
                    assert old[n] == ("declined", "N<101") and new[n] == ("declined", "N<2"), (word, direction, n, old[n], new[n])
            assert R.CENSUS["fast"].get("declined:N<2") == 1 and "declined:N<101" not in R.CENSUS["fast"], R.CENSUS["fast"]
            # the same tier word on an A100 (an unlisted class: floor 101): every size unchanged, word for word
            old = _fast_seam_over_sizes(R, monkeypatch, word, direction=direction, old_gate=True, cc=(8, 0), stack=None)
            new = _fast_seam_over_sizes(R, monkeypatch, word, direction=direction, cc=(8, 0), stack=None)
            assert new == old and all(new[n] == ("declined", "N<101") for n in FLOOR_SIZES if n <= 100), (word, direction, old, new)
            # the template track (64, 64): DECLINED at every size before and after (the stock statement); from 101 tokens word for word (c=64/d=64), below
            # 101 the word is now the track's too (0.2.11.6 filed those calls under N<101: the threshold check ran first)
            new = _fast_seam_over_sizes(R, monkeypatch, word, direction=direction, site=(64, 64))
            assert all(v == ("declined", "c=64/d=64") for v in new.values()), (word, direction, new)
            R.STATE["fast"]["word"] = word; R._PROC["thr"] = 100
            monkeypatch.setattr(R, "fast_floor", lambda m, x, thr=None: 101)                            # 0.2.11.6's serve_fast head, verbatim in effect: N <= thr -> N<101 first
            m64 = types.SimpleNamespace(d_pair=64, d_hidden=64, direction=direction)
            for n in FLOOR_SIZES:
                x = types.SimpleNamespace(shape=(1, n, n, 64), device="cuda:0", is_cuda=True, dim=lambda: 4)
                with pytest.raises(R.Declined) as e:
                    R.serve_fast(m64, x)
                assert e.value.word == "c=64/d=64", (n, e.value.word)                                # never reaches the floor (fast_floor's stand-in would say N<101)
            for n, v in new.items():                                                                     # every decline word stays a NAMED size route for the EXIT tally
                if v[0] == "declined":
                    assert report.TRIMUL_TOKEN_FLOOR.fullmatch(v[1]) or report.TRIMUL_TRACK_CLASS.fullmatch(v[1]), v
    assert seen_served_small == 4 * 4                                                                    # (fast, big) x (out, in) x N in (2, 12, 64, 100)
    # a ROW word is a measurement word: the library threshold + 1 stays its floor at every size (the default word v4 included)
    for word in ("v4", "tmk3_fast"):
        old = _fast_seam_over_sizes(R, monkeypatch, word, old_gate=True)
        new = _fast_seam_over_sizes(R, monkeypatch, word)
        assert new == old and all(new[n] == ("declined", "N<101") for n in FLOOR_SIZES if n <= 100), (word, new)


def test_head6_fast_floor_is_the_providers_v4_floor_capped_at_the_library_threshold(R, monkeypatch):
    import opt_core.kernels.trimul as KT
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(KT, "call_precision", lambda z: ("bf16", None))
    R._PROC[("cc", "cuda:0")] = (9, 0); R._PROC["stack"] = RF3_H100_STACK; R._PROC["thr"] = 100
    x = types.SimpleNamespace(shape=(1, 7, 7, 128), device="cuda:0", is_cuda=True, dtype=torch.bfloat16, dim=lambda: 4)
    pair = types.SimpleNamespace(d_pair=128, d_hidden=128, direction="outgoing"); templ = types.SimpleNamespace(d_pair=64, d_hidden=64, direction="incoming")
    x64 = types.SimpleNamespace(shape=(1, 7, 7, 64), device="cuda:0", is_cuda=True, dtype=torch.bfloat16, dim=lambda: 4)
    assert R.fast_floor(pair, x) == 101                                                                  # the default ROW word v4: the library threshold + 1
    for word in ("fast", "big"):
        R.STATE["fast"]["word"] = word
        assert R.fast_floor(pair, x) == T.v4_n_min("9.0", "bf16", 128, 128) == 2                        # the table's floor for the pair track's class on cc 9.0
        assert R.fast_floor(templ, x64) == 101                                                           # the template track: no table floor -> 101
        assert R.fast_floor(pair, x, thr=0) == 1                                                         # never above the library threshold + 1
        R._PROC[("cc", "cuda:0")] = (8, 0)
        assert R.fast_floor(pair, x) == 101                                                              # A100: unlisted class -> 101
        R._PROC[("cc", "cuda:0")] = (9, 0)
    monkeypatch.delattr(KT, "v4_n_min")                                                                  # a core older than the table floor: 101, as before
    assert R.fast_floor(pair, x) == 101
