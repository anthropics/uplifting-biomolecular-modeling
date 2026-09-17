"""levers/ARMT/odde_transition_bind.py — the pair-transition sites through the core provider (opt_core.kernels.transition): the word, the
provider's cell decision for OpenDDE's shapes (a stock-row cell / a refusal by name -> the kit's single-engine composition sep16 BY NAME;
a carried row -> the provider face), the registry / modes / floor / ablation rows, the census tokens (CPU); sep16's equality with the engine
module (GPU, skipped without CUDA)."""
import importlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
ARMT = os.path.join(OPT, "forward", "fast_inference", "levers", "ARMT")
if ARMT not in sys.path:
    sys.path.insert(0, ARMT)


@pytest.fixture(autouse=True)
def _no_module_left_behind():
    """the binding module read its word at import: drop it after each test so later renders (test_ran, the pinned mode lines) see no live word"""
    yield
    sys.modules.pop("odde_transition_bind", None)
    sys.modules.pop("fpf_transition_odde", None)
    os.environ.pop("ODDE_TRANSITION", None)


def _fresh(word):
    old = os.environ.get("ODDE_TRANSITION")
    if word is None:
        os.environ.pop("ODDE_TRANSITION", None)
    else:
        os.environ["ODDE_TRANSITION"] = word
    try:
        sys.modules.pop("odde_transition_bind", None)
        return importlib.import_module("odde_transition_bind")
    finally:
        if old is None:
            os.environ.pop("ODDE_TRANSITION", None)
        else:
            os.environ["ODDE_TRANSITION"] = old


class _Lin:
    def __init__(self, w):
        self.weight = w
        self.bias = None


class _Mod:
    """the attributes decide() reads of an OpenDDE Transition(c_in=384, n=4)"""
    def __init__(self, c=384, n=4):
        import torch
        self.c_in = c
        self.linear_no_bias = _Lin(torch.zeros(c, n * c))
        self.linear_no_bias_a = _Lin(torch.zeros(n * c, c)); self.linear_no_bias_b = _Lin(torch.zeros(n * c, c))


def test_word_parsing():
    for w, exp in ((None, None), ("", None), ("0", None), ("off", None), ("stock", None), ("fast", "fast"), ("exact", "exact"), ("big", "big"), ("v2", "v2")):
        m = _fresh(w)
        assert m.WORD == exp and m.active() == (exp is not None) and m.COUNTS["active"] == (exp is not None)
    from opt_core.kernels import transition as T
    assert set(m.TIER_WORDS) <= set(T.TIER_WORDS) and m.TIER_WORDS == ("fast", "exact", "big")   # the unit's tier words are the provider's (the big lines bind by `big`)
    assert m.SINGLETON == "sep16" and m.CELL == (384, 1536)


def test_decision_for_the_kits_cells_is_named(monkeypatch):
    """Every tier word resolves for OpenDDE's pair cell (c 384, hidden 1536) at the ladder's pair sizes and the refiner's 2N on 9.0 and 8.0: a
    carried row (-> provider), a stock row (-> singleton, reason cell_stock:<row>) or a refusal by name (-> singleton, reason refused:<kind>) --
    never anything unnamed; the decision is memoized per call class and counted once."""
    torch = pytest.importorskip("torch")
    from opt_core.kernels import transition as T
    real_select = T.select
    monkeypatch.setattr(T, "stack_word", lambda device=None: None, raising=False)
    for word in ("fast", "exact", "big"):
        m = _fresh(word)
        for cc in ("9.0", "8.0"):
            monkeypatch.setattr(T, "select", lambda w, _cc=cc, **kw: real_select(w, **{**kw, "device": None, "cc": _cc}))
            m.reset_counts(); m._STACK.clear()
            mod = _Mod()
            for N in (400, 800, 1200, 2400):
                x = torch.zeros(2, N, 384, dtype=torch.bfloat16)     # [rows.., N, c]: decide reads N from shape[-2]
                d = m.decide(mod, x)
                assert d.kind in ("provider", "singleton") and d.n_tokens == N
                if d.kind == "singleton":
                    assert d.reason.startswith(("cell_stock:", "refused:", "excluded:")), d.reason
                    if d.reason.startswith("cell_stock:"):
                        assert d.row in T.STOCK_ROWS
                else:
                    assert d.row in T.ROW_NAMES and d.row not in T.STOCK_ROWS
                assert m.decide(mod, x) is d                          # memoized per (device, N, dtype)
            assert m.COUNTS["decisions"] == 4
            # the census tokens after calls noted on both paths
            d = m.decide(mod, torch.zeros(2, 800, 384, dtype=torch.bfloat16))
            m.note_call(d if d.kind == "singleton" else d.aside("cell_stock:test"), m.SINGLETON)
            desc = m.describe()
            assert desc["word"] == word and desc["calls"] == 1 and desc["singleton"] == {"sep16": 1} and desc["prov_calls"] == 0
            assert all(isinstance(k, str) and "->" in k for k in desc["cells"]) or desc["refusals"]


def test_other_cells_refuse_by_name(monkeypatch):
    torch = pytest.importorskip("torch")
    from opt_core.kernels import transition as T
    m = _fresh("fast")
    real_select = T.select
    monkeypatch.setattr(T, "select", lambda w, **kw: real_select(w, **{**kw, "device": None, "cc": "9.0"}))
    m._STACK["word"] = None
    d = m.decide(_Mod(c=128, n=4), torch.zeros(2, 64, 128, dtype=torch.bfloat16))   # a cell the kit never binds (the adapter's own gate is (384,1536)); the provider still answers by name
    assert d.kind in ("provider", "singleton")


def test_registry_modes_floor_rows():
    from opendde_opt import modes, registry, smalln, report, ran
    for lv, word, line in (("transition_core", "fast", "LSTAR2A"), ("transition_exact", "exact", "S1")):
        assert lv in registry.LEVERS and lv in registry.ENGAGEMENT and lv in ran.COUNTERS and lv in smalln.FLOOR_LEVERS
        assert registry.LEVERS[lv].switch.startswith("ODDE_TRANSITION=" + word)
        L = modes.LINES[line]
        assert lv in L.levers and L.exports.get("ODDE_TRANSITION") == word
        assert modes.LEVER_SWITCHES[lv] == ("ODDE_TRANSITION",)
        assert report.CORE_LEVERS[lv] == "opt_core.kernels.transition" and lv in report.STRATEGY
    assert "transition_core" in modes.LEVER_DEPENDENTS["arm_u"] and "transition_exact" in modes.LEVER_DEPENDENTS["arm_z"]
    for name in modes.BIG_LINES:                                                  # the memory mode binds by its own tier word
        assert modes.LINES[name].exports.get("ODDE_TRANSITION") == modes.BIG_TRANSITION_WORD == "big"
        assert "transition_core" in modes.LINES[name].levers
    assert "fpf_transition" not in modes.LEVER_NOT_SHEDDABLE                         # the transition is individually switchable now (MODEL_OPT_LEVERS_OFF=transition_core)
    # left out by name: the word leaves the exports on every line that carries it
    for mode, lv in (("fast", "transition_core"), ("exact", "transition_exact"), ("big", "transition_core")):
        base = modes.line_of(mode)
        out = modes.without(base, (lv,)) if hasattr(modes, "without") else None
        if out is not None:
            assert "ODDE_TRANSITION" not in out.exports and lv not in out.levers


def test_evidence_tokens():
    from opendde_opt import report
    stats = {"transition_prov": {"word": "fast", "calls": 640, "prov_calls": 0, "rows": {}, "singleton": {"sep16": 640},
                                 "cells": {"9.0|bf16|pair_c384_n4|N<=800|eager|fwd->engine_module": 3}, "asides": {"cell_stock:engine_module": 640},
                                 "refusals": {}, "excluded": [], "errors": {}, "stack": None, "decisions": 3}}
    ev = report.lever_evidence("transition_core", stats)
    assert ev.get("word") == "fast" and ev.get("calls") == 640 and ev.get("rows") == "none" and ev.get("singleton") == "sep16:640"
    assert ev.get("asides") == "cell_stock:engine_module:640" and "pair_c384_n4" in (ev.get("cells") or "")


def test_no_generic_copy_in_tree():
    """the tree carries no generic fused-transition package and no fused variant of the adapter (the shared core's opt_core.kernels.fpf_transition serves)"""
    tp = os.path.join(ARMT, "third_party")
    assert not os.path.exists(os.path.join(tp, "fpf_transition"))
    src = open(os.path.join(tp, "fpf_transition_odde", "__init__.py")).read()
    assert "import fpf_transition" not in src and "_fused_transition_kernel" not in src.replace("fused_transition_kernel with", "")
    assert src.count("@triton.jit") == 1                                             # the one single-engine kernel kept by name: the elementwise SiLU*gate of sep16


@pytest.mark.skipif(not __import__("importlib").util.find_spec("torch") or not __import__("torch").cuda.is_available(), reason="CUDA")
def test_sep16_bitwise_engine_module_gpu():
    import torch
    prim = pytest.importorskip("opendde.model.modules.primitives")
    _fresh("fast")
    sys.modules.pop("fpf_transition_odde", None)
    sys.path.insert(0, os.path.join(ARMT, "third_party"))
    FT = importlib.import_module("fpf_transition_odde")
    torch.manual_seed(0)
    mod = prim.Transition(c_in=384, n=4).cuda().eval()
    for N in (64, 400):
        z = torch.randn(N, N, 384, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref = prim.Transition.forward(mod, z)
            FT.apply(mod)
            out = mod(z)
        assert out.dtype == ref.dtype and torch.equal(out, ref)
