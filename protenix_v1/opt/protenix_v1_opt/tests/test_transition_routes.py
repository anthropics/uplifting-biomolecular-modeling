"""The transition levers (xtr, ttr) route through the shared core's transition PROVIDER (opt_core.kernels.transition) by TIER word: xtr asks
`exact` with the module's LayerNorm output handed in, ttr asks `fast` (`big` under --mode big).  Widths the levers do not own (c_s=384,
fp32) keep the stock statement (`stock:C=<c>`); the template pairformer's c=64 x 128 transition is tmpl_xtr's route; a shape / card / stack
the provider refuses BY NAME (no cell, no vouch, the card's served rows) and a cell whose tier IS the stock statement keep the module's own
forward — counted `stock:<lever>:<refusal>` / `stock:<lever>:row=<row>` / `stock:below_min_rows` / `stock:above_max_rows`, gates by name,
never a `fallback:` word and never a kit-side kernel statement.  CPU only: the activation, the module and the provider are stand-ins; only
the routing decision of levers_ptx1._transition_forward runs."""
import importlib.util
import sys
import types

import pytest

from protenix_v1_opt import kit as K

torch = pytest.importorskip("torch")
pytest.importorskip("opt_core.oom")


def _levers():
    """The kit's lever adapter imported by path (its module-level imports are torch and opt_core.oom only)."""
    name = "levers_ptx1_routetest"
    for d in K.kit_sys_paths():
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, K.levers_file())
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class _X:
    """Stands in for a bf16 CUDA activation [rows, C]: only what the router reads (shape, is_cuda, dtype, numel, stride, dim, device)."""
    def __init__(self, rows, C):
        self.shape = (rows, C); self.is_cuda = True; self.dtype = torch.bfloat16; self._n = rows * C; self.device = "cuda:0"

    def numel(self): return self._n

    def stride(self, i): return 1

    def dim(self): return 2

    def contiguous(self): return self


def _module(C, HID):
    """Stands in for a protenix Transition(c_in=C, n=HID/C): the router reads linear_no_bias_a.weight.shape[0] and calls layernorm1 for xtr."""
    return types.SimpleNamespace(linear_no_bias_a=types.SimpleNamespace(weight=types.SimpleNamespace(shape=(HID, C))), layernorm1=lambda x: ("LN", x))


class _Refusal(Exception):
    def __init__(self, kind, word="?", fallback="torch_swiglu", detail=""):
        super().__init__(kind); self.kind = kind; self.word = word; self.fallback = fallback; self.detail = detail


class _Sel:
    def __init__(self, row, tier, cell="9.0|bf16|pair_c128_n4|N<=800|eager|fwd"):
        self.row, self.tier, self.cell_key, self.variant, self.word = row, tier, cell, None, tier

    def line(self): return "transition:%s word=%s tier=%s cell=%s" % (self.row, self.word, self.tier, self.cell_key)


class _Provider:
    """Stands in for opt_core.kernels.transition: select() plans by (word, c, hidden), transition() serves; both recorded."""
    STOCK_ROWS = ("torch_swiglu", "engine_module")
    Refusal = _Refusal

    def __init__(self, plan):
        self.plan = plan; self.selects = []; self.serves = []

    def select(self, word, **kw):
        self.selects.append(dict(word=word, **kw))
        r = self.plan.get((word, kw["c"], kw["hidden"]))
        if isinstance(r, Exception):
            raise r
        if r is None:
            raise _Refusal("no_cell:%s_c%d_h%d" % (kw.get("family"), kw["c"], kw["hidden"]), word)
        return _Sel(r, word)

    def pack(self, **kw): return object()

    def transition(self, xs, W, *, word, **kw):
        self.serves.append(dict(word=word, **kw))
        return "PROVIDED", _Sel(self.plan[(word, int(xs.shape[-1]), {128: 512, 64: 256}[int(xs.shape[-1])])], word)


def _wire(monkeypatch, LV, lever, plan, mode="fast"):
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a, **k: True)
    LV.CFG.update({"xtr": lever == "xtr", "ttr": lever == "ttr"})
    LV._ORIG["transition"] = lambda self, x: "STOCK"
    P = _Provider(plan)
    monkeypatch.setattr(LV, "_TR", lambda: P)
    monkeypatch.setattr(LV, "_provider_weights", lambda m, T: object())
    monkeypatch.setattr(LV, "kit_mode", lambda: mode)
    monkeypatch.setattr(LV._TEMPL, "transition_route", lambda m, x: None)             # tmpl_xtr off / aside: the stock template transition
    LV.COUNTS.clear(); LV.TRANSITION["selections"].clear(); LV.TRANSITION["refusals"].clear()
    return P


@pytest.mark.parametrize("lever,word", [("ttr", "fast"), ("xtr", "exact")])
def test_the_levers_ask_the_provider_by_tier_word_and_serve_its_row(monkeypatch, lever, word):
    LV = _levers()
    P = _wire(monkeypatch, LV, lever, {(word, 128, 512): "v2", (word, 64, 256): "lnl"})
    assert LV.transition_tier(lever) == word
    assert LV._transition_forward(_module(128, 512), _X(640 * 640, 128)) == "PROVIDED"   # the pair transitions (trunk, MSA-module pair stack, confidence)
    assert LV._transition_forward(_module(64, 256), _X(8 * 640 * 640, 64)) == "PROVIDED" # the MSA transition (c_m=64, n=4): `rows` cells keyed by sqrt(rows)
    assert [s["word"] for s in P.selects] == [word, word] and [s["word"] for s in P.serves] == [word, word]
    assert P.selects[0]["family"] == "pair" and P.selects[0]["n_tokens"] == 640 and P.selects[0]["rows_count"] == 640 * 640
    assert P.selects[1]["family"] == "rows" and P.selects[1]["rows_count"] == 8 * 640 * 640
    assert P.selects[0]["ln_given"] is (lever == "xtr")                                  # the exact construction: the module's LayerNorm output handed in
    assert (P.serves[0]["x_ln"] is not None) is (lever == "xtr") and P.serves[0]["residual"] is False
    assert LV.COUNTS["transition"] == {"%s:C=128" % lever: 1, "%s:C=64" % lever: 1}
    assert set(LV.TRANSITION["selections"]) == {"%s:pair:c128x512" % lever, "%s:rows:c64x256" % lever}
    acct = LV.describe.__globals__["TRANSITION"]
    assert acct is LV.TRANSITION


def test_big_mode_asks_the_big_word(monkeypatch):
    LV = _levers()
    P = _wire(monkeypatch, LV, "ttr", {("big", 128, 512): "v1:lnfused"}, mode="big")
    assert LV.transition_tier("ttr") == "big" and LV.transition_tier("xtr") == "exact"
    assert LV._transition_forward(_module(128, 512), _X(1200 * 1200, 128)) == "PROVIDED" and P.selects[0]["word"] == "big"


@pytest.mark.parametrize("lever,word", [("ttr", "fast"), ("xtr", "exact")])
def test_refusals_and_stock_rows_keep_the_module_by_name_never_a_fallback(monkeypatch, lever, word):
    LV = _levers()
    plan = {(word, 128, 512): _Refusal("exact_rows_above_max_rows_on_cc_8.0", word), (word, 64, 256): "torch_swiglu"}   # the card's served rows; a stock-row tier
    P = _wire(monkeypatch, LV, lever, plan)
    assert LV._transition_forward(_module(128, 512), _X(1028 * 1028, 128)) == "STOCK"
    assert LV._transition_forward(_module(64, 256), _X(4 * 1028 * 1028, 64)) == "STOCK"
    assert LV._transition_forward(_module(128, 256), _X(4000 * 4000, 128)) == "STOCK"       # DiffusionConditioning.transition_z1/z2 (c_z=128, n=2): no cell -> refused by name
    assert LV._transition_forward(_module(384, 1536), _X(705, 384)) == "STOCK"              # the single transition (c_s=384): not the levers' width
    assert LV._transition_forward(_module(64, 128), _X(203 * 203, 64)) == "STOCK"           # the template pairformer's transition: tmpl_xtr's route (aside here)
    assert P.serves == []                                                                    # nothing reached a kernel
    c = LV.COUNTS["transition"]
    assert c == {"stock:above_max_rows": 1, "stock:%s:row=torch_swiglu" % lever: 1, "stock:%s:no_cell:pair_c128_h256" % lever: 1, "stock:C=384": 1}, c
    assert not any(k.startswith("fallback:") for k in c)
    assert LV.TRANSITION["refusals"] == {"exact_rows_above_max_rows_on_cc_8.0": 1, "no_cell:pair_c128_h256": 1}


def test_a_provider_error_is_a_named_fallback(monkeypatch):
    LV = _levers()
    P = _wire(monkeypatch, LV, "ttr", {("fast", 128, 512): "v2"})

    def boom(xs, W, **kw): raise RuntimeError("launch failed")
    monkeypatch.setattr(P, "transition", boom)
    assert LV._transition_forward(_module(128, 512), _X(640 * 640, 128)) == "STOCK"
    assert LV.COUNTS["transition"] == {"fallback:ttr:error:RuntimeError": 1}


def test_below_the_template_routes_envelope_the_stock_statement(monkeypatch):
    LV = _levers()
    _wire(monkeypatch, LV, "xtr", {})
    called = []
    monkeypatch.setattr(LV._TEMPL, "transition_route", lambda m, x: called.append(1) or "TMPL")
    assert LV._transition_forward(_module(64, 128), _X(45 * 45, 64)) == "STOCK" and called == []     # under TRANSITION_GATE_ROWS: the route is not entered
    assert LV._transition_forward(_module(64, 128), _X(64 * 64, 64)) == "TMPL" and called == [1]
    assert LV.COUNTS["transition"] == {"stock:C=64": 1, "tmpl_xtr": 1}
