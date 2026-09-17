"""ln_core (opendde_opt/lncore.py): the LayerNorm provider binding's word, its named asides (the extension serves, counted), its install on a
module that imports under the word, its census words and the lines that carry it. CPU only: the provider rows themselves are the core's tests;
a CUDA call class is exercised on the GPU boxes (CHANGES.md)."""
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from opendde_opt import lncore, modes, registry, ran, smalln  # noqa: E402


class _LN:                                                     # the attributes FusedLayerNorm.forward reads (layer_norm.py:233-290)
    def __init__(self, C=384, weight=True, bias=True):
        self.normalized_shape = torch.Size((C,))
        self.eps = 1e-5
        self.weight = torch.ones(C) if weight else None
        self.bias = torch.zeros(C) if bias else None


def _orig(self, x):
    _orig.calls += 1
    return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


def test_word_spellings(monkeypatch):
    for v, w in (("", None), ("off", None), ("0", None), ("fast", "fast"), ("big", "big"), (" exact ", "exact"), ("fastln:lp", "fastln:lp")):
        assert lncore.word({"ODDE_LN": v}) == w
    monkeypatch.delenv("ODDE_LN", raising=False)
    assert lncore.word() is None
    monkeypatch.setenv("ODDE_LN", "fast")
    assert lncore.word() == "fast"


def test_no_word_is_the_original_untouched():
    lncore._reset()
    fwd = lncore.make_forward(_orig); _orig.calls = 0
    y = fwd(_LN(), torch.randn(4, 384))
    assert _orig.calls == 1 and y.shape == (4, 384) and lncore.COUNTS["calls"] == 0 and lncore.COUNTS["stock_calls"] == 0   # not even counted: the lever is off


@pytest.mark.parametrize("ln,x,why", [
    (_LN(C=64), torch.randn(8, 64), "form:c64"),                      # the MSA width: the extension by name
    (_LN(C=128), torch.randn(8, 128), "form:c128"),                   # the atom width
    (_LN(weight=True, bias=False), torch.randn(8, 384), "form:affine_weight_only"),
    (_LN(weight=False, bias=False), torch.randn(8, 384), "form:affine_none"),
    (_LN(), torch.randn(8, 384), "cpu_tensor"),                       # a CPU tensor never reaches the provider
])
def test_named_asides_serve_the_extension_counted(ln, x, why):
    lncore._reset(); lncore.STATS["word"] = "fast"; lncore.COUNTS.update(active=True, word="fast")
    fwd = lncore.make_forward(_orig); _orig.calls = 0
    y = fwd(ln, x)
    assert _orig.calls == 1 and torch.equal(y, _orig(ln, x))
    assert lncore.COUNTS["calls"] == 0 and lncore.COUNTS["stock_calls"] == 1 and lncore.COUNTS["asides"] == {why: 1}
    assert lncore.aside_word() == f"aside:{why}:1"                   # every call an aside: inert by design (registry.ENGAGEMENT stepped_aside), never 'never ran'
    lncore._reset()


def test_a_provider_error_puts_the_class_back_on_the_extension_named(monkeypatch, capsys):
    lncore._reset(); lncore.STATS["word"] = "fast"; lncore.COUNTS.update(active=True, word="fast")
    monkeypatch.setattr(lncore, "_decide", lambda x, C, rows, aligned: (_ for _ in ()).throw(RuntimeError("boom")))
    ln = _LN(); x = torch.randn(8, 384)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))      # walk the CUDA branch on a CPU tensor
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fwd = lncore.make_forward(_orig); _orig.calls = 0
    with torch.no_grad():
        y = fwd(ln, x)
    assert _orig.calls == 1 and y.shape == x.shape
    assert lncore.COUNTS["errors"] == {"RuntimeError": 1} and lncore.COUNTS["asides"] == {"error:RuntimeError": 1} and lncore.COUNTS["calls"] == 0
    assert lncore._CACHE[(384, torch.float32, 8, True)] == ("stock", "error:RuntimeError")
    assert "ln_core: RuntimeError: boom" in capsys.readouterr().err
    assert lncore.aside_word() is None and lncore.fallbacks(["ln_core"]) == ["ln_core:served_row_errors=RuntimeError:1"]   # an error is a fallback BY NAME (PARTIAL), not an aside
    lncore._reset()


def test_a_served_call_is_counted_per_row_and_cell(monkeypatch):
    lncore._reset(); lncore.STATS["word"] = "fast"; lncore.COUNTS.update(active=True, word="fast")
    sel = types.SimpleNamespace(row="fastln", variant="lp", cell="9.0|fp32|pair_c384|N<=400|eager|fwd")
    monkeypatch.setattr(lncore, "_decide", lambda x, C, rows, aligned: ("serve", sel, "fastln:lp"))
    fake = types.SimpleNamespace(layer_norm=lambda x, ns, w, b, eps, selection=None: (torch.nn.functional.layer_norm(x, ns, w, b, eps), selection))
    monkeypatch.setitem(lncore._ST, "PL", fake)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ln = _LN(); x = torch.randn(2, 3, 384)
    fwd = lncore.make_forward(_orig); _orig.calls = 0
    with torch.no_grad():
        y = fwd(ln, x)
    assert _orig.calls == 0 and y.shape == x.shape and torch.allclose(y, _orig(ln, x))
    assert lncore.COUNTS["calls"] == 1 and lncore.COUNTS["rows"] == {"fastln:lp": 1} and lncore.COUNTS["cells"] == {sel.cell: 1} and lncore.COUNTS["dtypes"] == {"float32": 1}
    assert ran.count("ln_core") == 1 and lncore.aside_word() is None and lncore.fallbacks(["ln_core"]) == []
    lncore._reset()


def test_capture_and_autograd_are_the_extensions(monkeypatch):
    lncore._reset(); lncore.STATS["word"] = "fast"; lncore.COUNTS.update(active=True, word="fast")
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    fwd = lncore.make_forward(_orig); _orig.calls = 0
    with torch.no_grad():
        fwd(_LN(), torch.randn(8, 384))
    x = torch.randn(8, 384, requires_grad=True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fwd(_LN(), x)
    assert _orig.calls == 2 and lncore.COUNTS["asides"] == {"capturing": 1, "autograd": 1} and lncore.COUNTS["calls"] == 0
    lncore._reset()


def test_install_binds_the_class_method_under_the_word(monkeypatch):
    lncore._reset()
    lm = types.ModuleType(lncore.TARGET)
    class FusedLayerNorm:                                         # noqa: D401  the stub class: forward is what the lever wraps
        def forward(self, input):                                 # noqa: A002
            return input
    lm.FusedLayerNorm = FusedLayerNorm
    monkeypatch.setitem(sys.modules, lncore.TARGET, lm)
    lncore.install({"ODDE_LN": ""})                               # no word: nothing installed, the lever inert by name
    assert lncore.STATS["word"] is None and not lncore.bound() and not lncore.STATS["installed"]
    lncore.install({"ODDE_LN": "big"})
    assert lncore.STATS["word"] == "big" and lncore.COUNTS["active"] and lncore.STATS["installed"] and lncore.bound()
    assert FusedLayerNorm.forward.__wrapped__ is not None and lncore.fallbacks(["ln_core"]) == []
    d = lncore.describe()
    assert d["word"] == "big" and d["bound"] is True and d["widths"] == "384" and d["calls"] == 0
    lncore._reset()


def test_the_lines_carry_the_lever_by_tier_word():
    L = modes.LINES
    assert L["LSTAR2A"].exports["ODDE_LN"] == "fast" and "ln_core" in L["LSTAR2A"].levers
    for n in ("BIG_F", "BIG_TP"):
        assert L[n].exports["ODDE_LN"] == modes.BIG_LN_WORD == "big" and "ln_core" in L[n].levers   # the memory mode's own word (core >= 0.5.112.0)
    assert "ODDE_LN" not in L["S1"].exports and "ln_core" not in L["S1"].levers and "ODDE_LN" in L["S1"].unset   # exact: the extension by name
    assert modes.LEVER_SWITCHES["ln_core"] == ("ODDE_LN",) and "ln_core" in smalln.FLOOR_LEVERS
    out = modes.line_without(L["LSTAR2A"], ("ln_core",))          # MODEL_OPT_LEVERS_OFF=ln_core: the word out, everything else as shipped
    assert "ODDE_LN" not in out.exports and "ODDE_LN" in out.unset and "ln_core" not in out.levers and "trimul_core" in out.levers
    assert registry.LEVERS["ln_core"].tier == "tier2" and "ODDE_LN" in registry.KNOBS and registry.PIN_STATUS["ln_core"][0] == "tested"
    assert "ln_core" in ran.COUNTERS and "ln_core" in registry.ENGAGEMENT
