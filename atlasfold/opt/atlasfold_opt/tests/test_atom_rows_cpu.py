"""atom_rows on CPU: fast-row membership / order, registry / installer coverage, the expected words == the module's literals, install applies
on the stock texts (digests) and refuses by name over a wrapped ConditionedTransitionBlock.forward, the CPU route / the disabled word / rank-4
(token transformer) operands are the stock statement (torch.equal), the leaf factors equal AdaLN's own cond branch.  The row kernels themselves
are Triton (CUDA): their numerics are the GPU identity evidence in CHANGES, not a CPU assertion."""
import importlib

import pytest
import torch

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import installers


def test_row_registry_installer():
    row = modes.MODES[modes.FAST]
    assert "atom_rows" in row and "atom_rows" not in modes.MODES[modes.EXACT] and "atom_rows" in modes.MODES[modes.BIG]
    assert row.index("atom_kdedup") < row.index("atom_rows") < row.index("atom_bf16") < row.index("denoiser_graph")
    r = registry.LEVERS["atom_rows"]
    assert r["cls"] == "fast" and "ConditionedTransitionBlock.forward" in r["site"] and "atom_rows" in installers()


def test_expected_words_are_the_module_literals():
    from atlasfold_opt.hooks import atom_rows
    assert tuple(registry.LEVERS["atom_rows"]["expected"]) == atom_rows.EXPECTED
    src = open(atom_rows.__file__).read()
    for word in atom_rows.EXPECTED:
        assert f'ledger.fallback("{word}")' in src, word


def _ctb(c=16, cc=8, seed=0):
    DT = importlib.import_module("atlasfold.model.network.diffusion_transformer")
    torch.manual_seed(seed)
    m = DT.ConditionedTransitionBlock(c, cc).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return DT, m


def test_cpu_route_disabled_word_and_rank4_are_the_stock_statement(monkeypatch):
    from atlasfold_opt.hooks import atom_rows
    monkeypatch.delenv("AFO_ATOM_ROWS", raising=False); atom_rows._STATE["override"] = None
    DT, m = _ctb()
    stock = DT.ConditionedTransitionBlock.forward
    a5, c5 = torch.randn(2, 3, 4, 7, 16), torch.randn(2, 1, 4, 7, 8)
    a4, c4 = torch.randn(2, 3, 9, 16), torch.randn(2, 1, 9, 8)
    with torch.no_grad():
        ref5, ref4 = stock(m, a5, c5), stock(m, a4, c4)
    ins = atom_rows.install("fast", "atlasfold-opt", {})
    try:
        assert ins.applied, ins.reason
        assert DT.ConditionedTransitionBlock.forward.__wrapped_stock__ is stock and set(ins.facts["digests"]) == set(atom_rows.SOURCE_SHA256)
        with torch.no_grad():
            assert torch.equal(m(a5, c5), ref5) and torch.equal(m(a4, c4), ref4)
        led = ins.facts["ledger"]
        assert led.served == 0 and led.fallbacks == {"cpu": 2}                    # the device word is read before the rank word
        monkeypatch.setenv("AFO_ATOM_ROWS", "0")
        with torch.no_grad():
            assert torch.equal(m(a5, c5), ref5)
        assert led.fallbacks == {"cpu": 2, "disabled": 1}
        line = ins.lines[0]()
        assert "name=LOCAL.atlasfold.atom_rows state=skipped reason=all_fallback:" in line and "switch=AFO_ATOM_ROWS=0" in line, line
    finally:
        ins.facts["restore"](); atom_rows._STATE["override"] = None
    assert DT.ConditionedTransitionBlock.forward is stock


def test_leaf_factors_are_adaln_cond_branch():
    from atlasfold_opt.hooks import atom_rows
    from opt_core.counters import Ledger
    _, m = _ctb()
    cond = torch.randn(2, 1, 4, 7, 8)
    led = Ledger("t", impl="t", origin="kit", expected=atom_rows.EXPECTED); led.set("leaf_per_call", 0)
    with torch.no_grad():
        sig, bias = atom_rows._ada_leaves(m.adaln, cond, led)
        c = m.adaln.layernorm_cond(cond)
        assert torch.equal(sig, torch.sigmoid(m.adaln.linear_g(c))) and torch.equal(bias, m.adaln.linear_bias(c))
    assert led.get("leaf_per_call") == 1


def test_install_refuses_by_name_over_a_wrapped_transition():
    from atlasfold_opt.hooks import atom_rows, rebind
    DT = importlib.import_module(atom_rows.TARGET)
    stock = DT.ConditionedTransitionBlock.forward

    def foreign(self, *a, **k):
        return stock(self, *a, **k)
    foreign.__qualname__ = "ConditionedTransitionBlock.forward[someone_else]"
    rebind(DT.ConditionedTransitionBlock, "forward", foreign, stock)
    try:
        ins = atom_rows.install("fast", "atlasfold-opt", {})
        assert not ins.applied and ins.reason == "wrapped:ConditionedTransitionBlock.forward[someone_else]"
    finally:
        setattr(DT.ConditionedTransitionBlock, "forward", stock)
