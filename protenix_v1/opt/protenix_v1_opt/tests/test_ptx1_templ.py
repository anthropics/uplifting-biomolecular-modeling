"""CPU tests for the template-embedder levers (ptxfpf/ptx1_templ.py): slot grouping, the distinct-template forward's bitwise equality to the
stock loop on a small CPU TemplateEmbedder (fp32, torch paths), the tmpl_triatt route words against a stubbed cell table, and the accounts.
No GPU."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))                       # protenix_v1/
PTXFPF = os.path.join(KIT, "opt", "forward", "v05_addon", "ptxfpf")
if PTXFPF not in sys.path:
    sys.path.insert(0, PTXFPF)
import ptx1_templ as TP  # noqa: E402


def _feats(n_tok=12, slots=("gap", "zero", "zero", "zero"), seed=0):
    """Template features in the featuriser's inference form: slot kinds gap (restype 31, zero planes), zero (all zero), or rand:<k> (random,
    equal for equal k)."""
    g = torch.Generator().manual_seed(seed)
    T = len(slots)
    f = {"template_aatype": torch.zeros(T, n_tok, dtype=torch.int64), "template_distogram": torch.zeros(T, n_tok, n_tok, 39),
         "template_pseudo_beta_mask": torch.zeros(T, n_tok, n_tok), "template_unit_vector": torch.zeros(T, n_tok, n_tok, 3),
         "template_backbone_frame_mask": torch.zeros(T, n_tok, n_tok), "asym_id": torch.tensor([0] * (n_tok // 2) + [1] * (n_tok - n_tok // 2))}
    made = {}
    for t, kind in enumerate(slots):
        if kind == "gap":
            f["template_aatype"][t] = 31
        elif kind.startswith("rand"):
            if kind not in made:
                made[kind] = {"aa": torch.randint(0, 20, (n_tok,), generator=g), "dg": torch.rand(n_tok, n_tok, 39, generator=g),
                              "pb": (torch.rand(n_tok, n_tok, generator=g) > 0.3).float(), "uv": torch.randn(n_tok, n_tok, 3, generator=g),
                              "bb": (torch.rand(n_tok, n_tok, generator=g) > 0.3).float()}
            m = made[kind]
            f["template_aatype"][t] = m["aa"]; f["template_distogram"][t] = m["dg"]; f["template_pseudo_beta_mask"][t] = m["pb"]
            f["template_unit_vector"][t] = m["uv"]; f["template_backbone_frame_mask"][t] = m["bb"]
    return f


def test_groups_settings_of_record():
    assert TP.template_groups(_feats()) == [0, 1, 1, 1]
    assert TP.groups_word([0, 1, 1, 1]) == "0|1,2,3"


def test_groups_general():
    assert TP.template_groups(_feats(slots=("rand:a", "rand:b", "rand:a", "rand:b"))) == [0, 1, 0, 1]
    assert TP.template_groups(_feats(slots=("rand:a", "rand:b", "rand:c", "gap"))) == [0, 1, 2, 3]
    assert TP.template_groups(_feats(slots=("zero", "zero", "zero", "zero"))) == [0, 0, 0, 0]
    assert TP.groups_word([0, 1, 0, 1]) == "0,2|1,3"


def test_groups_unkeyed_forms_are_none():
    f = _feats(); del f["template_distogram"]
    assert TP.template_groups(f) is None                                           # a precursor / partial form: the stock statements run
    f = _feats(); f["template_unit_vector"] = f["template_unit_vector"][:3]
    assert TP.template_groups(f) is None                                           # slot counts disagree
    assert TP.template_groups("not a dict") is None


def _cpu_layernorms(module):
    """The stack's LayerNorm may be the fast_layernorm CUDA extension (LAYERNORM_TYPE): swap every LayerNorm-like submodule for torch.nn.LayerNorm
    with the same affine parameters so the module runs on CPU (both arms of the comparison run the same swapped module)."""
    for name, m in list(module.named_modules()):
        if name and "LayerNorm" in type(m).__name__ and not isinstance(m, torch.nn.LayerNorm) and getattr(m, "weight", None) is not None:
            ln = torch.nn.LayerNorm(tuple(m.weight.shape), eps=float(getattr(m, "eps", 1e-5)))
            with torch.no_grad():
                ln.weight.copy_(m.weight)
                if getattr(m, "bias", None) is not None: ln.bias.copy_(m.bias)
            parent = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
            setattr(parent, name.rsplit(".", 1)[-1], ln)
    return module


@pytest.fixture
def embedder():
    pf = pytest.importorskip("protenix.model.modules.pairformer")
    torch.manual_seed(0)
    te = pf.TemplateEmbedder(n_blocks=1, c=16, c_z=32).eval()
    for p in te.parameters():                                                       # the checkpoint-free module initialises some projections to zero: make every path carry signal
        torch.nn.init.normal_(p, std=0.1)
    te = _cpu_layernorms(te)
    yield te
    TP.set_dedupe(False)
    assert pf.TemplateEmbedder.forward is TP._ORIG["forward"]


@pytest.mark.parametrize("slots", [("gap", "zero", "zero", "zero"), ("rand:a", "rand:b", "rand:a", "rand:b"), ("rand:a", "rand:b", "rand:c", "rand:d"), ("zero",)])
def test_dedupe_forward_is_bitwise_the_stock_loop(embedder, slots):
    n = 10
    f = _feats(n_tok=n, slots=slots, seed=1)
    z = torch.randn(n, n, 32)
    TP.set_dedupe(False)
    TP.COUNTS["dedupe"].clear(); TP._STATE["group_key"] = None
    with torch.no_grad():
        ref = embedder(f, z.clone(), triangle_attention="torch", triangle_multiplicative="torch")
        assert TP.set_dedupe(True) is True
        out = embedder(f, z.clone(), triangle_attention="torch", triangle_multiplicative="torch")
    assert torch.equal(out, ref)
    d = TP.describe_dedupe()
    distinct = len(set(TP.template_groups(f)))
    assert (d["installed"], d["calls"], d["evaluated"], d["reused"], d["slots"]) == (True, 1, distinct, len(slots) - distinct, len(slots))
    assert d["groups"] == TP.groups_word(TP.template_groups(f))


def test_dedupe_untemplated_dict_takes_the_stock_early_return(embedder):
    f = _feats(); del f["template_aatype"]
    TP.COUNTS["dedupe"].clear()
    assert TP.set_dedupe(True)
    with torch.no_grad():
        assert embedder(f, torch.randn(12, 12, 32)) == 0                              # pairformer.py:1009-1011: the stock returns 0
    assert TP.describe_dedupe()["stock"] == {"untemplated": 1}


class _Dec:
    def __init__(self, served, word): self.served, self._w, self.reason = served, word, word
    def word(self): return self._w


class _M:
    class mha: no_heads = 4; c_hidden = 32


def test_triatt_gate_words(monkeypatch):
    x = torch.zeros(8, 8, 64)
    TP._DECISIONS.clear(); TP.COUNTS["triatt"].clear()
    TP.set_triatt(False)
    assert TP.triatt_gate("gflash", _M(), x) == "stock:c=64"                         # lever off: today's census word, unchanged
    TP.set_triatt(True)
    table = {("lnl", "ln_linear", (64,)): True, ("lnl", "gate_transpose", (128,)): True, ("fpf", "prologue", (64, 4, 32)): False, ("fpf", "epilogue", (64, 4, 32)): False}
    monkeypatch.setattr(TP, "_preflight", lambda wanted, device: [_Dec(table[w], "served" if table[w] else "off:not-measured:64x4x32") for w in wanted])
    assert TP.triatt_gate("gflash", _M(), x) is None                                 # cells serve: the caller's block answers the call
    assert TP.triatt_gate("gblock", _M(), x) == "stock:tmpl_triatt:off:not-measured:64x4x32"   # no served exact-class cell at (64,4,32): the stock statement, by name
    d = TP.describe_triatt()
    assert d["routed"] == {"gflash:c=64": 1} and d["stock"] == {"tmpl_triatt:off:not-measured:64x4x32": 1}
    assert d["decisions"]["gflash:c64h4d32"]["served"] is True and d["decisions"]["gblock:c64h4d32"]["served"] is False
    assert TP.triatt_cells("gflash", 64, 4, 32) == [("lnl", "ln_linear", (64,)), ("lnl", "gate_transpose", (128,))]
    assert TP.triatt_cells("gblock", 64, 4, 32) == [("fpf", "prologue", (64, 4, 32)), ("fpf", "epilogue", (64, 4, 32))]
    facts = TP.lever_facts("tmpl_triatt")
    assert facts["served"] == 1 and facts["cls"] == "tolerance"
    TP.set_triatt(False); TP._DECISIONS.clear(); TP.COUNTS["triatt"].clear()


def test_triatt_preflight_error_is_unserved_by_name(monkeypatch):
    TP._DECISIONS.clear(); TP.COUNTS["triatt"].clear(); TP.set_triatt(True)
    def boom(wanted, device): raise ImportError("no core")
    monkeypatch.setattr(TP, "_preflight", boom)
    assert TP.triatt_gate("gflash", _M(), torch.zeros(4, 4, 64)) == "stock:tmpl_triatt:preflight:ImportError"
    TP.set_triatt(False); TP._DECISIONS.clear(); TP.COUNTS["triatt"].clear()


def test_lever_names_and_classes():
    pytest.importorskip("protenix.model.modules.pairformer")
    assert TP.LEVER_NAMES == ('template_dedupe', 'tmpl_triatt', 'tmpl_trimul', 'tmpl_xtr', 'tmpl_pairfused', 'tmpl_trimul_exact')
    assert TP.CLASS == {'template_dedupe': 'exact', 'tmpl_triatt': 'tolerance', 'tmpl_trimul': 'tolerance', 'tmpl_xtr': 'exact', 'tmpl_pairfused': 'tolerance', 'tmpl_trimul_exact': 'exact'}
    assert set(TP.apply({}).keys()) == set(TP.LEVER_NAMES)


class _FakeTrimulProvider:
    """opt_core.trimul.by_word's Provider, reduced to what ptx1_templ calls (eligible / fn / kernel / version); eligible records the selection."""
    def __init__(self, refuse=None, row="tmk3_exact", cls="fast", reason=""):
        self.refuse, self.kernel, self.version, self.calls = refuse, "trimul:%s" % row, "%s/9.0|bf16|C64|H128" % row, []
        self.sel = type("Sel", (), {"row": row, "cls": cls, "reason": reason})()
    def eligible(self, call):
        from opt_core.trimul import Refused
        if self.refuse: raise Refused(self.refuse)
        call.extra["trimul_selection"] = self.sel
    def fn(self, call):
        self.calls.append((call.direction, call.residual, tuple(call.z.shape))); return call.z + 2.0


class _TM_Module:
    _outgoing = False


def test_stock_forward_hands_the_statement_back():
    seen = []
    orig = lambda module, x, mask, chunk, ta, inplace: seen.append((mask, chunk, ta, inplace)) or "y"
    assert TP.stock_forward(orig, None, torch.zeros(2, 2, 64)) == "y" and seen == [(None, None, "torch", False)]


def _tm_reset():
    TP.COUNTS["trimul"].clear(); TP._TM.update(provider=None, error=None, first=None)


def test_trimul_route_off_served_refused_idle_no_floor(monkeypatch):
    pytest.importorskip("opt_core.trimul")
    _tm_reset(); z = torch.zeros(120, 120, 64); mask = torch.ones(120, 120)
    TP.set_trimul(False)
    assert TP.trimul_route(_TM_Module(), z, mask, True, True) is None and not TP.COUNTS["trimul"]          # off: the caller's stock statement, nothing counted here
    TP.set_trimul(True)                                                  # no kit-side size floor: even a 64-token call reaches the provider (its cells decide; here a fake)
    small = _FakeTrimulProvider(row="row_at_small_n", cls="fast"); monkeypatch.setattr(TP, "_trimul_provider", lambda word=None: small)
    assert torch.equal(TP.trimul_route(_TM_Module(), torch.zeros(64, 64, 64), None, True, True), torch.zeros(64, 64, 64) + 2.0) and small.calls == [("incoming", True, (64, 64, 64))]
    _tm_reset()
    # CLASS CONTRACT (never the core's current cell winner by name): whatever row the tier word resolves to, a selection that is NOT the
    # stock's class of computation serves and is accounted under that row's own name; a stock-class selection idles BY NAME; a refusal is BY NAME.
    ROW = "some_measured_row"                                             # a stand-in name: the kit must not care which kernel row wins the cell
    fake = _FakeTrimulProvider(row=ROW, cls="fast"); monkeypatch.setattr(TP, "_trimul_provider", lambda word=None: fake)      # a measured (tolerance-class) row serves
    out = TP.trimul_route(_TM_Module(), z, mask, True, True)
    assert torch.equal(out, z + 2.0) and fake.calls == [("incoming", True, (120, 120, 64))]
    d = TP.describe_trimul()
    assert d["routed"] == {ROW: 1} and d["first"]["row"] == ROW and d["first"]["N"] == 120 and TP.lever_facts("tmpl_trimul")["served"] == 1
    _tm_reset(); monkeypatch.setattr(TP, "_trimul_provider", lambda word=None: _FakeTrimulProvider(refuse=ROW + ":dtype:fp16"))   # a refusal: by name (<row>:<kind>...)
    assert TP.trimul_route(_TM_Module(), z, mask, False, False) is None and TP.describe_trimul()["stock"] == {"tmpl_trimul:" + ROW + ":dtype:fp16": 1}
    for row, cls, why in (("cueq", "fast", "a STOCK_ROWS name"), ("torch_math", "exact", "a STOCK_ROWS name"), ("any_stock_class_row", "stock", "cls == stock")):
        _tm_reset(); idle = _FakeTrimulProvider(row=row, cls=cls, reason="named stock row: no_cell:9.0|bf16|C64|H128|out|fwd")   # the word names the stock for
        monkeypatch.setattr(TP, "_trimul_provider", lambda word=None: idle)                                                              # this cell -> idle by name, nothing launched
        assert TP.trimul_route(_TM_Module(), z, mask, True, True) is None and idle.calls == [], why
        assert TP.describe_trimul()["stock"] == {"tmpl_trimul:%s:no_cell" % row: 1}, why
    _tm_reset(); TP.set_trimul(False)


def test_trimul_live_selection_class_contract():
    """Against the shared core in the tree (skipped without it): the kit's tier word for the template TriMul resolves to SOME row per card — the
    route's only demands are the selection's shape (row / cls) and the tier semantics (word exact never names a tolerance-class row); which
    kernel row wins the (bf16, 64, 128) cell is the core's business and is not asserted here."""
    KT = pytest.importorskip("opt_core.kernels.trimul")
    for cc in ("9.0", "8.0"):
        try:
            fast = KT.select(cc, "bf16", 64, 128, 800, "outgoing", word=TP.TRIMUL_WORDS["fast"])
            exact = KT.select(cc, "bf16", 64, 128, 800, "outgoing", word="exact")
        except Exception as e:                                              # no stack / table on this host: nothing to assert here
            pytest.skip("no provider selection on this host (%s): %r" % (cc, e))
        for sel in (fast, exact):
            assert isinstance(getattr(sel, "row", None), str) and sel.row and hasattr(sel, "cls")
        assert TP.TRIMUL_WORDS == {"exact": "exact", "fast": "fast", "big": "big"} and TP.CLASS["tmpl_trimul"] == "tolerance" and TP.CLASS["tmpl_trimul_exact"] == "exact"
        klass = lambda sel: str(getattr(sel, "cls", "") or "").split("(")[0].strip()   # the core spells classes 'stock(<row>)' / 'tol(<numbers>)' / 'exact(...)': the class is the head word
        assert klass(exact) in ("stock", "exact", "bitwise"), "word exact resolved to a %s-class row %s on cc %s" % (exact.cls, exact.row, cc)
        assert klass(fast) != "", "word fast resolved to a row without a class on cc %s: %s" % (cc, fast.row)


def test_trimul_weights_keys():
    tri = pytest.importorskip("protenix.model.triangular.triangular")
    m = tri.TriangleMultiplicationOutgoing(c_z=64, c_hidden=128)
    w = TP.trimul_weights(m)
    assert tuple(sorted(w)) == tuple(sorted(("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")))
    assert tuple(w["w_ap"].shape) == (128, 64) and tuple(w["w_o"].shape) == (64, 128) and tuple(w["w_og"].shape) == (64, 64) and TP.trimul_weights(m) is w


class _XD:
    """pair_fused.CellDecision, reduced to what ptx1_templ reads (served / word() / reason)."""
    def __init__(self, served, word): self.served, self._w, self.reason = served, word, word
    def word(self): return self._w


class _XtrModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm1 = torch.nn.LayerNorm(64); self.linear_no_bias_a = torch.nn.Linear(64, 128, bias=False)
        self.linear_no_bias_b = torch.nn.Linear(64, 128, bias=False); self.linear_no_bias = torch.nn.Linear(128, 64, bias=False)


def test_xtr_route_off_unserved_served_refused(monkeypatch):
    PF = pytest.importorskip("opt_core.attn.pair_fused")
    m = _XtrModule(); x = torch.zeros(70, 70, 64)
    TP.COUNTS["xtr"].clear(); TP.set_xtr(False)
    assert TP.transition_route(m, x) is None and not TP.COUNTS["xtr"]                                     # off: nothing counted
    TP.set_xtr(True); monkeypatch.setattr(TP, "_preflight", lambda wanted, device: [_XD(False, "off:not-measured:64x128")])
    assert TP.transition_route(m, x) is None and TP.describe_xtr()["stock"] == {"tmpl_xtr:off:not-measured:64x128": 1}   # unserved cell: stock by name
    TP.set_xtr(True); TP.COUNTS["xtr"].clear(); monkeypatch.setattr(TP, "_preflight", lambda wanted, device: [_XD(True, "row")])
    seen = {}
    monkeypatch.setattr(PF, "pack_transition_weights", lambda **kw: ("packed", tuple(sorted(kw))), raising=False)
    def fake_transition(xs, T, residual, ln, x_ln, impl):
        seen.update(T=T, residual=residual, ln=ln, impl=impl, x_ln_shape=tuple(x_ln.shape)); return xs + 1.0
    monkeypatch.setattr(PF, "transition", fake_transition, raising=False)
    y = TP.transition_route(m, x)
    assert torch.equal(y, x + 1.0) and seen["ln"] == "stock" and seen["impl"] == "fpf" and seen["residual"] is False and seen["x_ln_shape"] == (70, 70, 64)
    assert TP.describe_xtr()["routed_total"] == 1 and TP.lever_facts("tmpl_xtr")["served"] == 1
    class _U(Exception):
        def __init__(self, reason): self.reason = reason
    monkeypatch.setattr(PF, "Unsupported", _U, raising=False)
    def refusing(xs, T, residual, ln, x_ln, impl): raise _U("rows<4096")
    monkeypatch.setattr(PF, "transition", refusing, raising=False)
    assert TP.transition_route(m, x) is None and TP.describe_xtr()["fallback"] == {"tmpl_xtr:rows<4096": 1}
    TP.set_xtr(False); TP.COUNTS["xtr"].clear()


def test_pairfused_impl_choice():
    TP.COUNTS["pairfused"].clear(); TP.set_pairfused(False)
    assert TP.triatt_impl("gflash", "lnl", "fused", None, 64) == ("lnl", "fused", None)                    # off: the trunk lever's choice
    TP.set_pairfused(True)
    assert TP.triatt_impl("gflash", "lnl", "fused", None, 64) == ("fpf", "fused", None) and TP.describe_pairfused()["calls"] == 1
    assert TP.triatt_impl("gblock", "fpf", "stock", "LN", 64) == ("fpf", "stock", "LN")                    # exact tier (gblock): untouched
    assert TP.triatt_impl("gflash", "lnl", "fused", None, 128)[0] == "lnl"                                  # trunk width: untouched
    assert TP.triatt_cells("gflash", 64, 4, 32) == [("fpf", "prologue", (64, 4, 32)), ("fpf", "epilogue", (64, 4, 32))]
    TP.set_pairfused(False)
    assert [c[0] for c in TP.triatt_cells("gflash", 64, 4, 32)][0] == "lnl"


def test_xtr_card_row_rule_matches_xtr():
    """tmpl_xtr rides the shared core's served-row rule for the exact transition (opt_core.attn.pair_fused exact_rows, piece `transition` — the
    statement xtr's provider tier honours): on cc 8.0 the calls outside the served rows are handed to the stock template transition by the census
    word; cc 9.0 states no rule."""
    L = pytest.importorskip("levers_ptx1")
    PF = pytest.importorskip("opt_core.attn.pair_fused")
    if not hasattr(PF, "exact_rows_word"):
        pytest.skip("opt_core below the exact_rows rule")
    for n in (400, 800, 1023, 1024, 1028, 1100, 1200, 1340, 1460, 2048):
        rows = n * n
        w = PF.exact_rows_word("transition", "8.0", rows)
        assert TP.xtr_card_word(rows, "8.0") == (None if w is None else L._rows_census_word(w)), n
        assert TP.xtr_card_word(rows, "9.0") is None, n
    assert TP.xtr_card_word(1028 * 1028, "8.0") == L.ABOVE_MAX_ROWS                      # the 1,028-token remainder piece (8,208 rows) is outside the served band
    assert TP.xtr_card_word(1200 * 1200, "8.0") is None and TP.xtr_card_word(400 * 400, "8.0") is None
    assert TP.xtr_card_word(45 * 45, "8.0") == L.BELOW_MIN_ROWS                           # under the transition floor (the route's own 4096-row envelope sits at the same line)


def test_xtr_route_steps_aside_by_name_on_the_card_rule(monkeypatch):
    PF = pytest.importorskip("opt_core.attn.pair_fused")
    m = _XtrModule(); x = torch.zeros(70, 70, 64); calls = []
    monkeypatch.setattr(TP, "_preflight", lambda wanted, device: [_XD(True, "row")])
    monkeypatch.setattr(PF, "pack_transition_weights", lambda **kw: "packed", raising=False)
    monkeypatch.setattr(PF, "transition", lambda xs, T, residual, ln, x_ln, impl: (calls.append(impl), xs + 1.0)[1], raising=False)
    TP.COUNTS["xtr"].clear(); TP.set_xtr(True)
    monkeypatch.setattr(TP, "_device_cc", lambda dev: "8.0"); monkeypatch.setattr(TP, "xtr_card_word", lambda rows, cc: "stock:above_max_rows" if cc == "8.0" else None)
    assert TP.transition_route(m, x) is None and calls == [] and TP.describe_xtr()["stock"] == {"tmpl_xtr:above_max_rows": 1}   # 8.0 outside the band: stock by name, nothing launched
    monkeypatch.setattr(TP, "_device_cc", lambda dev: "9.0")
    assert torch.equal(TP.transition_route(m, x), x + 1.0) and calls == ["fpf"] and TP.describe_xtr()["routed_total"] == 1                # 9.0: served as before
    monkeypatch.setattr(TP, "_device_cc", lambda dev: "8.0"); monkeypatch.setattr(TP, "xtr_card_word", lambda rows, cc: None)
    assert torch.equal(TP.transition_route(m, x), x + 1.0) and calls == ["fpf", "fpf"]                                                    # 8.0 inside the band: served
    TP.set_xtr(False); TP.COUNTS["xtr"].clear()


def test_trimul_word_follows_the_mode(monkeypatch):
    """The provider word is the MODE's tier word (no row word, no size floor): fast -> fast, big -> big, exact -> exact; the bare module (no kit
    mode): fast, or exact when only tmpl_trimul_exact is on; TRIMUL_WORD overrides for measurement."""
    TP.set_trimul(True); TP.set_trimul_exact(False)
    for mode, word in (("fast", "fast"), ("big", "big"), ("exact", "exact"), (None, "fast")):
        monkeypatch.setattr(TP, "_kit_mode", lambda m=mode: m)
        assert TP.trimul_word() == word, mode
    monkeypatch.setattr(TP, "_kit_mode", lambda: None)
    TP.set_trimul(False); TP.set_trimul_exact(True)
    assert TP.trimul_word() == "exact" and TP.describe_trimul("tmpl_trimul_exact")["on"] is True and TP.describe_trimul()["on"] is False
    assert TP.lever_facts("tmpl_trimul_exact")["cls"] == "exact" and TP.lever_facts("tmpl_trimul")["cls"] == "tolerance"
    monkeypatch.setattr(TP, "TRIMUL_WORD", "some_row_word")
    assert TP.trimul_word() == "some_row_word"
    TP.set_trimul_exact(False)
    assert not hasattr(TP, "TRIMUL_MIN_TOKENS")


def test_trimul_exact_word_routes_and_idles_by_name(monkeypatch):
    pytest.importorskip("opt_core.trimul")
    TP.COUNTS["trimul"].clear(); TP._TM.update(provider=None, error=None, first=None); TP._TM.get("providers", {}).clear()
    monkeypatch.setattr(TP, "_kit_mode", lambda: "exact"); TP.set_trimul(False); TP.set_trimul_exact(True)
    z = torch.zeros(120, 120, 64); mask = torch.ones(120, 120); seen = []
    class _P(_FakeTrimulProvider):
        pass
    served = _P(row="an_exact_kernel_row", cls="exact")
    monkeypatch.setattr(TP, "_trimul_provider", lambda word=None: (seen.append(word), served)[1])
    out = TP.trimul_route(_TM_Module(), z, mask, True, True)
    assert seen == ["exact"] and torch.equal(out, z + 2.0) and TP.describe_trimul("tmpl_trimul_exact")["routed"] == {"an_exact_kernel_row": 1}
    TP.COUNTS["trimul"].clear()
    idle = _P(row="torch_math", cls="stock(torch_math)", reason="named stock row: no exact-class row vouched at 8.0|bf16|C64|H128")
    monkeypatch.setattr(TP, "_trimul_provider", lambda word=None: idle)
    assert TP.trimul_route(_TM_Module(), z, mask, True, True) is None and idle.calls == [] and list(TP.describe_trimul("tmpl_trimul_exact")["stock"]) == ["tmpl_trimul:torch_math:stock_row"]
    TP.set_trimul_exact(False); TP.COUNTS["trimul"].clear()


class _FakeTransitionProvider:
    class Refusal(Exception):
        def __init__(self, kind): super().__init__(kind); self.kind = kind
    def __init__(self, refuse=None): self.refuse, self.calls = refuse, []
    def pack(self, **kw): return "W"
    def transition(self, xs, W, *, word, residual, n_tokens, family, timing):
        self.calls.append((word, family, n_tokens, residual))
        if self.refuse: raise self.Refusal(self.refuse)
        class _Sel: row = "a_transition_row"; variant = None; cell_key = "rows|c64x128|N<=400"
        return xs + 3.0, _Sel()


def test_xtr_provider_by_tier_word_then_pair_fused(monkeypatch):
    PF = pytest.importorskip("opt_core.attn.pair_fused")
    m = _XtrModule(); x = torch.zeros(70, 70, 64); pf_calls = []
    monkeypatch.setattr(TP, "_preflight", lambda wanted, device: [_XD(True, "row")])
    monkeypatch.setattr(PF, "pack_transition_weights", lambda **kw: "packed", raising=False)
    monkeypatch.setattr(PF, "transition", lambda xs, T, residual, ln, x_ln, impl: (pf_calls.append(impl), xs + 1.0)[1], raising=False)
    monkeypatch.setattr(TP, "_device_cc", lambda dev: "9.0"); monkeypatch.setattr(TP, "xtr_card_word", lambda rows, cc: None)
    TP.COUNTS["xtr"].clear(); TP._XTR["provider_first"] = None; TP.set_xtr(True)
    for mode in ("fast", "big"):                                           # the MODE's tier word reaches the provider (fast / big); served -> its output, pair_fused untouched
        prov = _FakeTransitionProvider(); monkeypatch.setattr(TP, "_transition_provider", lambda p=prov: p); monkeypatch.setattr(TP, "_kit_mode", lambda m=mode: m)
        m = _XtrModule()
        assert torch.equal(TP.transition_route(m, x), x + 3.0) and prov.calls == [(mode, "rows", 70, False)] and pf_calls == []
    assert TP.describe_xtr()["routed_total"] == 2 and TP.describe_xtr()["provider_first"]["row"] == "a_transition_row"
    prov = _FakeTransitionProvider(); monkeypatch.setattr(TP, "_transition_provider", lambda p=prov: p); monkeypatch.setattr(TP, "_kit_mode", lambda: "exact")
    assert torch.equal(TP.transition_route(_XtrModule(), x), x + 1.0) and prov.calls == [] and pf_calls == ["fpf"]   # exact keeps the pair_fused EXACT construction: the provider is not asked (provider:not_in_mode)
    assert TP.COUNTS["xtr"].pop("provider:not_in_mode") == 1                # (counted; removed here so the refusal census below reads alone)
    pf_calls.clear(); monkeypatch.setattr(TP, "_kit_mode", lambda: "big")
    prov = _FakeTransitionProvider(refuse="no_cell"); monkeypatch.setattr(TP, "_transition_provider", lambda: prov)   # a refusal BY NAME -> the pair_fused construction
    assert torch.equal(TP.transition_route(_XtrModule(), x), x + 1.0) and pf_calls == ["fpf"] and TP.describe_xtr()["provider"] == {"refused:no_cell": 1}
    def boom(): raise ImportError("no kernels.transition")
    monkeypatch.setattr(TP, "_transition_provider", boom)                                                               # an older core: named, pair_fused
    assert torch.equal(TP.transition_route(_XtrModule(), x), x + 1.0) and TP.describe_xtr()["provider"].get("none:ImportError") == 1
    monkeypatch.setattr(TP, "XTR_PROVIDER_TIERS", ("fast", "big")); monkeypatch.setattr(TP, "_kit_mode", lambda: "exact")   # a mode outside the tiers keeps pair_fused by name
    prov = _FakeTransitionProvider(); monkeypatch.setattr(TP, "_transition_provider", lambda: prov)
    assert torch.equal(TP.transition_route(_XtrModule(), x), x + 1.0) and prov.calls == [] and TP.describe_xtr()["provider"].get("not_in_mode") == 1
    TP.set_xtr(False); TP.COUNTS["xtr"].clear()


def test_triatt_route_takes_the_trunk_blocks_core_word_no_template_twin(monkeypatch):
    """tmpl_triatt routes the template calls through the trunk's block; the block's attention core is levers_ptx1.gblock_core's under gblock (the
    provider's `exact` tier word when `triexact` rides it, the stock statement otherwise) — the template lever carries no core word of its own and
    its routing answer for a call does not read one."""
    assert "tmpl_triexact" not in TP.LEVER_NAMES and not any(w in TP.CFG for w in ("triexact", "tricuda", "tmpl_triexact"))
    L = pytest.importorskip("levers_ptx1")
    assert L.CORE_WORDS["triexact"]["host"] == "gblock" and "tmpl_triatt" in L.LEVER_NAMES
    TP._DECISIONS.clear(); TP.COUNTS["triatt"].clear(); TP.set_triatt(True)
    monkeypatch.setattr(TP, "_preflight", lambda wanted, device: [_Dec(True, "served") for w in wanted])   # the fpf cells at (64, 4, 32) serve
    x = torch.zeros(8, 8, 64)
    for on in (False, True):                                                         # the word off or on: the same routing answer (the core is chosen inside the block, per key)
        monkeypatch.setitem(L.CFG, "triexact", on)
        assert TP.triatt_gate("gblock", _M(), x) is None
    assert TP.describe_triatt()["routed"] == {"gblock:c=64": 2}
    TP.set_triatt(False); TP._DECISIONS.clear(); TP.COUNTS["triatt"].clear()
