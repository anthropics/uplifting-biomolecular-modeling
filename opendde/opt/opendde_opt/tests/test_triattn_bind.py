"""Levers triattn_core / triattn_exact (levers/ARMT/odde_triattn_bind.py): the arm's triangle-attention site through the core's provider
(opt_core.kernels.triattn). CPU: the word, the tier word straight to select() with the provider's named fallbacks, the registry / modes / floor / ablation rows, the inert-by-aside
predicate, the LEVER evidence. GPU (skipped without CUDA): the provider's rows at OpenDDE's shape (H=12, head dim 32) — the exact rows bitwise to
the stock cuEquivariance op, the fast rows inside K2B's class — and the unit's serve() under both words."""
import importlib
import os
import re
import sys
import types

import pytest

from opendde_opt import modes, ran, registry, report, smalln, stack

ARMT = os.path.join(os.path.abspath(stack.tree_root()), "opt", "forward", "fast_inference", "levers", "ARMT")   # <tree>/opt/<registry.AR>


def _load(monkeypatch, word):
    """Import odde_triattn_bind fresh with ODDE_TRIATTN=<word> (None = unset)."""
    if word is None:
        monkeypatch.delenv("ODDE_TRIATTN", raising=False)
    else:
        monkeypatch.setenv("ODDE_TRIATTN", word)
    if ARMT not in sys.path:
        monkeypatch.syspath_prepend(ARMT)
    sys.modules.pop("odde_triattn_bind", None)
    return importlib.import_module("odde_triattn_bind")


@pytest.mark.parametrize("word,active", [(None, False), ("", False), ("0", False), ("off", False), ("fast", True), ("exact", True), ("big", True), ("k2b", True), ("k2b@m128r2", True)])
def test_word(monkeypatch, word, active):
    tb = _load(monkeypatch, word)
    assert tb.active() is active and tb.COUNTS["active"] is active
    assert (tb.WORD is None) == (not active)
    sys.modules.pop("odde_triattn_bind", None)


class _FakeT:
    """A stand-in for opt_core.kernels.triattn's face: select() by word from a scripted table, the provider's Refusal / rows() / STOCK_ROWS."""
    ROW_NAMES = ("k2b", "k2", "flash", "cuda_sm90a", "triattn_native", "exact_rowx", "exact_headsplit", "cueq", "ds4sci", "sdpa", "stock")
    STOCK_ROWS = ("cueq", "ds4sci", "sdpa", "stock")
    TIER_WORDS = ("fast", "exact", "big")

    class Refusal(RuntimeError):
        def __init__(self, kind, row, fallback):
            super().__init__(kind); self.kind, self.row, self.fallback = kind, row, fallback

    def __init__(self, answers, refuse=None, fallbacks=None):
        self.answers, self.refuse, self.fallbacks, self.asked = dict(answers), dict(refuse or {}), dict(fallbacks or {}), []

    def select(self, cc, dtype, D, H, S, word, stack=None, **kw):
        self.asked.append(word)
        if word in self.refuse:
            kind, fb = self.refuse[word]; raise self.Refusal(kind, word, fb)
        row = self.answers.get(word, word)
        return types.SimpleNamespace(row=row, word=word, cell=f"{cc}|{dtype}|D{D}|H{H}|N<=800|fwd", config=None, cls="stock" if row in self.STOCK_ROWS else "fast")

    def rows(self):
        return {r: {"fallback": self.fallbacks.get(r, "cueq")} for r in self.ROW_NAMES}

    def cell_for(self, cc, dtype, D, H, S):
        return f"{cc}|{dtype}|D{D}|H{H}|N<=800|fwd", {"fast": self.answers.get("fast"), "exact": self.answers.get("exact")}, True, ""

    def admits(self, row, cc, dtype, D, H, S, stack=None):
        return (True, "")


def _prime(tb, T, word):
    tb._SEL.clear(); tb._REDIRECT.clear(); tb._FACTS.clear(); tb.reset_counts(); tb.COUNTS["unavailable"] = {}; tb.COUNTS["refusals"] = {}; tb.COUNTS["excluded"] = []
    tb._FACTS.update(cc=(9, 0), cc_word="9.0", stack=None, word=word, form_kw=False, exact_stack=None)
    return {"cc": (9, 0), "cc_word": "9.0", "stack": None, "word": word, "exact_stack": None}


def test_the_tier_word_goes_straight_to_select_and_a_refusal_serves_the_named_fallback(monkeypatch):
    """No kit chain, cell table or launch setting: select(word=<tier>) names the row; a Refusal serves ITS fallback row by name; a stock row (the
    cell's or a fallback's) is the stock op by the named reason; a row that refused at call time is redirected to the provider's named fallback
    for the rest of the process (excluded=, cell_stock_after_exclusions)."""
    tb = _load(monkeypatch, "fast")
    T = _FakeT({"fast": "triattn_native", "exact": "exact_rowx"})
    f = _prime(tb, T, "fast")
    sel, why = tb._select(T, f, "bf16", 32, 12, 800)
    assert (sel.row, why, T.asked) == ("triattn_native", None, ["fast"])                      # the tier word, verbatim, once
    assert tb._select(T, f, "bf16", 32, 12, 800)[0] is sel and T.asked == ["fast"]        # memoised per cell
    T = _FakeT({"fast": "triattn_native"}, refuse={"fast": ("no_prebuilt", "k2b")}); f = _prime(tb, T, "fast")
    sel, why = tb._select(T, f, "bf16", 32, 12, 800)
    assert (sel.row, why, T.asked) == ("k2b", None, ["fast", "k2b"]) and tb.COUNTS["refusals"] == {"fast:no_prebuilt": 1}   # the provider's NAMED fallback row
    T = _FakeT({"exact": "cueq"}); f = _prime(tb, T, "exact")
    assert tb._select(T, f, "fp32", 32, 12, 800) == (None, "cell_stock")                  # the cell names the stock op for the tier: the stock op, by the cell's rule
    T = _FakeT({"exact": "exact_rowx"}, refuse={}, fallbacks={"exact_rowx": "cueq"}); f = _prime(tb, T, "exact")
    tb._exclude(T, "exact_rowx", "unavailable_here", "cueq")                            # the row refused AT CALL TIME (driver bindings absent): its named fallback = the stock op
    assert tb._select(T, f, "bf16", 32, 12, 800) == (None, "cell_stock_after_exclusions") and tb.COUNTS["excluded"] == ["exact_rowx"]
    T = _FakeT({"fast": "triattn_native", "cuda_sm90a": "cuda_sm90a"}, fallbacks={"triattn_native": "cuda_sm90a"}); f = _prime(tb, T, "fast")
    tb._exclude(T, "triattn_native", "RuntimeError")                                           # a row that failed while running: the provider's rows()[row]['fallback'] serves from here on
    sel, why = tb._select(T, f, "bf16", 32, 12, 800)
    assert (sel.row, why) == ("cuda_sm90a", None) and T.asked == ["fast", "cuda_sm90a"]
    T = _FakeT({"fast": "k2b"}); f = _prime(tb, T, "big")                               # a core older than the big word is handled in _facts (word_served=fast); with the word known it passes verbatim
    sel, _ = tb._select(T, f, "bf16", 32, 12, 800); assert T.asked == ["big"]
    src = open(tb.__file__).read()                                                         # no kit-side selection machinery survives (the provider's cell decides)
    assert not any(w in src for w in ("KIT_CELL_ROWS", "EXACT_MIN_X", "K2B_CELLS", "_chain(", "prefer=")), "kit-side row selection is the provider's job"
    sys.modules.pop("odde_triattn_bind", None)


def test_exact_stack_is_passed_and_the_providers_vouch_gate_is_real(monkeypatch):
    """The unit hands the provider this process's exact-vouch key (select(exact_stack=...)): on a stack the table records an exact row byte-vouched
    on, the exact word serves that row; on an unknown stack the tier passes the row over BY NAME and the stock op serves (Aside cell_stock), the
    LEVER line saying `refusals=<row>:exact_vouch_not_recorded` / `unavailable=<row>:exact_vouch_not_recorded:<key>` -- no kit bypass of the gate."""
    T = pytest.importorskip("opt_core.kernels.triattn")
    if not hasattr(T, "exact_stack_key"):
        pytest.skip("this core predates the exact-vouch gate")
    tb = _load(monkeypatch, "exact")
    src = open(tb.__file__).read()
    assert 'fkw["exact_stack"] = f["exact_stack"]' in src and "T.exact_stack_key(cc)" in src          # passed on every select of the unit
    for cc in ("8.0", "9.0"):
        rows = {r for r, d in T.rows().items() if (r.startswith("exact") or d.get("class") == "exact") and any(k.startswith(cc + "|") for k in d.get("vouched_on", ()))}   # the exact-class rows the core byte-checked on this card
        vouched = sorted({re.sub(r"\+cu1[23](?=\||$)", "", k) for r in rows for k in T.rows()[r]["vouched_on"] if k.startswith(cc + "|") and k.count("|") == 2})   # as a process names its stack (a row recorded per ops build lists '<lib>+<build>'; device-tagged keys are other cards)
        row = None
        for key in vouched:                                                                          # every recorded stack of this card: an exact row serves
            _prime(tb, T, "exact"); tb._FACTS.update(cc=tuple(int(x) for x in cc.split(".")), cc_word=cc, exact_stack=key)
            f = dict(tb._FACTS, word="exact")
            sel, why = tb._select(T, f, "bf16", 32, 12, 800)                                       # the cell decides: an exact row recorded for the card, or the stock op by name
            assert ((sel is None and why == "cell_stock") or (sel.row in rows and why is None)) and not tb.COUNTS["unavailable"], (cc, key, sel, why)
            row = sel.row if sel is not None else row
        bogus = cc + "|torch2.9.0+cu126|cueq0.10.0"                                                   # a stack nobody byte-checked: the gate holds (or no exact row at all: the stock op)
        _prime(tb, T, "exact"); tb._FACTS.update(cc=tuple(int(x) for x in cc.split(".")), cc_word=cc, exact_stack=bogus)
        f = dict(tb._FACTS, word="exact")
        sel, why = tb._select(T, f, "bf16", 32, 12, 800)
        assert sel is None and why == "cell_stock", (cc, sel, why)
        if row is not None:
            assert tb.COUNTS["refusals"].get(row + ":exact_vouch_not_recorded") == 1 and tb.COUNTS["unavailable"][row] == "exact_vouch_not_recorded:" + bogus, tb.COUNTS
    for key in ("9.0|torch2.7.1+cu126|cueq0.10.0", "8.0|torch2.7.1+cu126|cueq0.10.0"):                # the kit's pinned image (torch 2.7.1+cu126, cuequivariance 0.10.0)
        rows = [r for r, d in T.rows().items() if (r.startswith("exact") or d.get("class") == "exact") and any(re.sub(r"\+cu1[23](?=\||$)", "", k) == key for k in d.get("vouched_on", ()))]   # an exact-class row the core byte-checked on this stack serves; none: the stock op
        _prime(tb, T, "exact"); tb._FACTS.update(cc=tuple(int(x) for x in key[:3].split(".")), cc_word=key[:3], exact_stack=key)
        sel, why = tb._select(T, dict(tb._FACTS, word="exact"), "bf16", 32, 12, 800)
        assert (sel.row in rows) if sel is not None else (why == "cell_stock"), (key, rows, sel, why)
    sys.modules.pop("odde_triattn_bind", None)


def test_registry_modes_floor_and_ablation_rows():
    assert registry.validate() == []
    for lv, tier in (("triattn_core", "tier2"), ("triattn_exact", "exact")):
        assert registry.LEVERS[lv].tier == tier and registry.PIN_STATUS[lv][0] == "tested" and lv in ran.COUNTERS
        assert (lv in smalln.FLOOR_LEVERS) == any(lv in modes.LINES[n].levers for n in modes.SMALL_LINES)   # a floor lever once a floor line carries it
        assert lv in modes.LEVER_SWITCHES and "ODDE_TRIATTN" in modes.LEVER_SWITCHES[lv]
    assert "ODDE_TRIATTN" in modes.ARM_SWITCHES
    fast = modes.LINES["LSTAR2A"]
    assert "triattn_core" in fast.levers and fast.exports.get("ODDE_TRIATTN") == "fast" and "ODDE_TRIATTN" not in fast.unset
    for n in modes.BIG_LINES:                                                 # the big lines are fast's lever set bound by the provider's OWN memory-tier word (0.2.57)
        assert "triattn_core" in modes.LINES[n].levers and modes.LINES[n].exports.get("ODDE_TRIATTN") == modes.BIG_TRIATTN_WORD == "big"
    s1 = modes.LINES["S1"]
    assert "ODDE_TRIATTN" in s1.unset or s1.exports.get("ODDE_TRIATTN") == "exact"   # the exact line: the word unset (stock attention) unless it carries triattn_exact
    assert ("triattn_exact" in s1.levers) == (s1.exports.get("ODDE_TRIATTN") == "exact")
    out = modes.line_without(fast, ("triattn_core",))                          # its own rule: the word out, the arm kept
    assert "triattn_core" not in out.levers and "ODDE_TRIATTN" not in out.exports and "ODDE_TRIATTN" in out.unset and "arm_u" in out.levers
    gone = [x for x in fast.levers if x not in modes.line_without(fast, ("arm_u",)).levers]
    assert "triattn_core" in gone and "arm_u23" in gone                         # the arm out takes the binding along (LEVER_DEPENDENTS)


def test_inert_by_aside_predicate(monkeypatch):
    """The unit reached its site and handed every call to the stock op by a named rule -> inert (complete); a served call -> engaged; a row error -> engaged (never_ran decides)."""
    fake_arm = types.ModuleType("odde_arm_t"); fake_arm.CFG = {"MIN_TOKENS": 300}
    fake = types.ModuleType("odde_triattn_bind")
    fake.COUNTS = {"calls": 0, "stock_calls": 12, "asides": {"cell_stock": 12}, "unavailable": {"exact_rowx": "ModuleNotFoundError"}, "errors": {}}
    monkeypatch.setitem(sys.modules, "odde_arm_t", fake_arm)
    monkeypatch.setitem(sys.modules, "odde_triattn_bind", fake)
    facts = {"n_gpu": 1, "token_floors": {"q": 800}}
    ok, why = ran.engagement("triattn_exact", facts)
    assert not ok and why.startswith("aside:cell_stock:12;unavailable=exact_rowx:ModuleNotFoundError")
    fake.COUNTS["calls"] = 3
    assert ran.engagement("triattn_exact", facts) == (True, None)
    fake.COUNTS.update(calls=0, errors={"triattn_native": "RuntimeError('x')"})
    assert ran.engagement("triattn_core", facts) == (True, None)                 # a row failed: not a by-design aside (lever_never_ran names it)
    ok, why = ran.engagement("triattn_core", {"n_gpu": 2})
    assert not ok and why.startswith("replaced_by=tp_triatt")
    ok, why = ran.engagement("triattn_core", {"n_gpu": 1, "token_floors": {"q": 200}})
    assert not ok and "300" in why                                              # below ARM's design gate: every call is stock by design


def test_lever_evidence_tokens():
    stats = {"arm_t": {"triattn": {"word": "fast", "calls": 3504, "rows": {"k2b": 3400, "cuda_sm90a": 104}, "stock_calls": 2, "asides": {"bias_per_row": 2},
                                   "refusals": {"exact_rowx:ModuleNotFoundError": 1}, "excluded": ["exact_rowx"], "word_served": "fast", "cuda_prebuilt": False,
                                   "cells": {"row=k2b word=fast class=fast cell=9.0|bf16|D32|H12|N<=800|fwd measured=yes x_stock=2.03": 3400}}}}
    ev = report.lever_evidence("triattn_core", stats)
    assert ev["word"] == "fast" and ev["calls"] == 3504 and ev["rows"] == "cuda_sm90a:104,k2b:3400" and ev["excluded"] == "exact_rowx"
    assert ev["refusals"] == "exact_rowx:ModuleNotFoundError:1" and ev["asides"] == "bias_per_row:2" and ev.get("word_served") is None   # word_served printed only when it differs (big on an old core)
    stats["arm_t"]["triattn"].update(word="big", word_served="fast")
    assert report.lever_evidence("triattn_core", stats)["word_served"] == "fast"
    assert report.lever_evidence("triattn_core", {}) == {}


# ------------------------------------------------------------------------------------------------------------------------ GPU (skipped on CPU)
def _gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


needs_gpu = pytest.mark.skipif(not _gpu(), reason="CUDA device required (the provider rows' equality at OpenDDE's shape)")


def _inputs(N, H=12, D=32, seed=0):
    import torch
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    q, k, v = (torch.randn(1, N, H, N, D, device="cuda", generator=g).bfloat16() for _ in range(3))
    bias = torch.randn(1, 1, H, N, N, device="cuda", generator=g).bfloat16().float()
    mask = torch.rand(1, N, 1, 1, N, device="cuda", generator=g) > 0.03
    mask[:, 3] = False                                                          # one fully-masked row (the uniform-average convention)
    return q, k, v, bias, mask


def _stock(q, k, v, bias, mask=None, scale=None):
    from cuequivariance_torch.primitives.triangle import triangle_attention
    r = triangle_attention(q, k, v, bias, mask=mask, scale=scale)
    return r[0] if isinstance(r, tuple) else r


@needs_gpu
@pytest.mark.parametrize("N", [320, 384, 800])
def test_gpu_provider_rows_at_opendde_shape(N):
    import math, torch
    from opt_core.kernels import triattn as T
    q, k, v, bias, mask = _inputs(N)
    sc = 1.0 / math.sqrt(32)
    ref = _stock(q, k, v, bias, mask=mask, scale=sc)
    ref32 = T.reference(q, k, v, bias, mask, scale=sc)
    hs = T.triangle_attention(q, k, v, bias, mask, scale=sc, word="exact_headsplit", stock=_stock)
    assert torch.equal(hs, ref)                                                 # exact by construction: the stock op per head
    kb = T.triangle_attention(q, k, v, bias, mask, scale=sc, word="k2b")
    rel = ((kb.float() - ref32).pow(2).mean().sqrt() / ref32.pow(2).mean().sqrt()).item()
    assert torch.isfinite(kb).all() and rel < 5e-3, rel                        # K2B's class vs the fp32 reference (cuEq's own is ~2.1e-3)


@needs_gpu
@pytest.mark.parametrize("word", ["fast", "exact"])
def test_gpu_serve_under_both_words(monkeypatch, word):
    import math, torch
    tb = _load(monkeypatch, word)
    q, k, v, bias, mask = _inputs(384)
    sc = 1.0 / math.sqrt(32)
    ref = _stock(q, k, v, bias, mask=mask, scale=sc)
    try:
        out = tb.serve(q, k, v, bias, mask, sc, stock5=_stock)
    except tb.Aside as a:                                                       # the cell's winner is the stock op on this card (e.g. exact on cc 8.0 at 384 rows): named
        assert a.reason.startswith("cell_stock"), a.reason
        sys.modules.pop("odde_triattn_bind", None)
        return
    assert out.shape == ref.shape and out.dtype == ref.dtype and torch.isfinite(out).all()
    d = tb.describe()
    assert d["calls"] == 1 and sum(d["rows"].values()) == 1 and d["cells"]
    row = next(iter(d["rows"]))
    assert row in T.ROW_NAMES and row not in T.STOCK_ROWS, d                     # a kernel row of the provider served (class contract; no winner named)
    if word == "exact":
        assert torch.equal(out, ref), (row, d)                                    # the exact word: bitwise the stock op, whichever exact-class row the cell names
    else:
        assert (out.float() - ref.float()).abs().max().item() < 0.05              # the fast word: tolerance class
    b_rows = bias.expand(1, 384, 12, 384, 384)                                  # a per-row-expanded bias (upstream's chunk_layer view) is taken once, no copy ...
    out2 = tb.serve(q, k, v, b_rows, mask, sc, stock5=_stock)
    assert torch.equal(out2, out) and tb.COUNTS["bias_forms"].get("rows_expanded_view") == 1
    with pytest.raises(tb.Aside, match="bias_per_row"):                        # ... a materialised per-row bias is the stock op's, by name
        tb.serve(q[:, :4], k[:, :4], v[:, :4], bias.expand(1, 4, 12, 384, 384).contiguous(), mask[:, :4], sc, stock5=_stock)
    sys.modules.pop("odde_triattn_bind", None)


def test_refresh_marks_the_binding_applied_from_the_arms_counts(monkeypatch):
    """stack.refresh(): triattn_* is applied when the arm's lever applied, the unit's word is live and the arm's attention site reports
    att_mode core_<word>; with the arm on but the site not taken it is a fallback BY NAME (partial), never a silent 'on'."""
    import sys, types
    from opendde_opt import stack
    fake_hook = types.SimpleNamespace(STATE={"installs": [{"levers": {"odde_arm_t": {"arm": "U"}}}], "errors": {}})
    arm = types.SimpleNamespace(COUNTS={"arm": "U", "installed": True, "att_mode": "core_fast", "u2_trimul": True, "u3": True})
    bind = types.SimpleNamespace(COUNTS={"active": True, "word": "fast", "calls": 3})
    monkeypatch.setitem(sys.modules, "odde_served_levers", fake_hook)
    monkeypatch.setitem(sys.modules, "odde_arm_t", arm)
    monkeypatch.setitem(sys.modules, "odde_triattn_bind", bind)
    saved = stack._REPORT
    try:
        stack._REPORT = {"active": True, "levers_planned": ["served_levers_hook", "arm_u", "arm_u23", "triattn_core"], "levers": ["served_levers_hook", "arm_u", "arm_u23", "triattn_core"]}
        rep = stack.refresh()
        assert "triattn_core" in rep["levers_applied"] and not rep["partial"], rep
        arm.COUNTS["att_mode"] = "stock"                                             # the site kept the stock op: the binding did not take it -- named, never a silent 'on'
        rep = stack.refresh()
        assert "triattn_core" not in rep["levers_applied"] and rep["partial"] and any(f.startswith("triattn_core:att_mode=stock") for f in rep["levers_fallback"]), rep
        monkeypatch.delitem(sys.modules, "odde_triattn_bind")
        rep = stack.refresh()
        assert any(f.startswith("triattn_core:odde_triattn_bind not imported") for f in rep["levers_fallback"]), rep
    finally:
        stack._REPORT = saved


def test_confidence_stack_sublever(monkeypatch):
    """Lever triattn_conf: on wherever the word rides -- S1, LSTAR2A and the big lines alike since 0.2.57 (the confidence head binds the tier word
    like every other pair stack; no line exports ODDE_TRIATTN_CONF) -- no switch of its own, left out by a value (MODEL_OPT_LEVERS_OFF=triattn_conf
    -> ODDE_TRIATTN_CONF=stock: the unit reports conf='stock', the arm raises Aside('conf_stock') inside the confidence head's stack), leaves with
    its parents, a floor lever, counted (ran) and engaged like its parents plus the conf_cells_stock predicate."""
    for n in ("S1", "LSTAR2A", "BIG_F"):                                              # every line carries it; none exports the ablation value
        L = modes.LINES[n]
        assert "triattn_conf" in L.levers and "ODDE_TRIATTN_CONF" in L.unset and "ODDE_TRIATTN_CONF" not in L.exports, n
        w = modes.line_without(L, ("triattn_conf",))
        assert w.exports.get("ODDE_TRIATTN_CONF") == "stock" and [x for x in L.levers if x not in w.levers] == ["triattn_conf"], n
    S1 = modes.LINES["S1"]
    w = modes.line_without(S1, ("triattn_exact",)); assert "ODDE_TRIATTN_CONF" not in w.exports and "ODDE_TRIATTN" not in w.exports and "triattn_conf" not in w.levers
    w = modes.line_without(modes.LINES["LSTAR2A"], ("arm_u",)); assert "triattn_conf" not in w.levers and "triattn_core" not in w.levers
    assert registry.LEVERS["triattn_conf"].tier == "exact" and registry.PIN_STATUS["triattn_conf"][0] == "tested"
    assert "triattn_conf" in ran.COUNTERS and "triattn_conf" in smalln.FLOOR_LEVERS and registry.ENGAGEMENT["triattn_conf"].stepped_aside
    assert "MODEL_OPT_TRIATTN_CONF" not in registry.KNOBS
    monkeypatch.setenv("ODDE_TRIATTN_CONF", "stock"); m = _load(monkeypatch, "exact")
    assert m.CONF == "stock" and m.describe()["conf"] == "stock"
    monkeypatch.delenv("ODDE_TRIATTN_CONF"); m = _load(monkeypatch, "fast")
    assert m.CONF is None
