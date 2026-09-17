"""The LayerNorm rows of the FPF add-on (``rf3fpf/fpf_rf3_ln_rows.py``, the shared core's provider ``opt_core.kernels.ln``): the arm component
``xln[.<word>]`` leaves today's arms byte for byte; a word outside the vocabulary is refused by name; the default word is the provider's exact
TIER (exactln / exactln:widen where its table measured them bitwise and >= x1.00 on the card's cell, ATen BY NAME at the floor cells); a row the
provider refuses is bound to its NAMED fallback, printed and counted; only the fp32 and autocast-widen forms are served (every other form is a
named route to the module's own forward); the SELECT / CENSUS lines carry the provider's ``describe()``.  CPU only: selections over synthetic
facts + the seam on CPU tensors (a CPU call is the route ``form:cpu``, the module's own forward)."""
import os
import re
import sys

import pytest

from rosettafold3_opt import modes, stack

L = pytest.importorskip("opt_core.kernels.ln")
RF3FPF = os.path.join(stack.fpf_home(), "rf3fpf")


@pytest.fixture
def R(monkeypatch):
    monkeypatch.syspath_prepend(RF3FPF)
    sys.modules.pop("fpf_rf3_ln_rows", None)
    import fpf_rf3_ln_rows as R
    R.reset_census()
    R.STATE["xln"]["word"] = R.DEFAULT_WORD["xln"]; R.STATE["xln"]["on"] = False
    R._PROC["stack"] = None; R._PROC["has_triton"] = True
    yield R
    R.disable_xln()
    R.reset_census()
    sys.modules.pop("fpf_rf3_ln_rows", None)


def test_todays_arms_are_untouched_by_the_component_grammar(R):
    for arm in ("fast+gflash+ttr+apb+res@L1", "tg+sapb@L1", "fast+apb+res@L1", "fast", "stock+tg@L1", "tg+sapb+xmul+xatt@L1.warm"):
        assert R.strip_components(arm) == (arm, {"xln": None}), arm


def test_words_parse_and_unknown_words_are_refused_by_name(R):
    assert R.strip_components("tg+sapb+xln@L1.warm") == ("tg+sapb@L1.warm", {"xln": "exact"})
    assert R.strip_components("tg+sapb+xln.fastln@L1") == ("tg+sapb@L1", {"xln": "fastln"})
    assert R.strip_components("fast+gflash+ttr+apb+res+xln.exactln:triton@L1") == ("fast+gflash+ttr+apb+res@L1", {"xln": "exactln:triton"})
    assert R.strip_components("xln@L1") == ("stock@L1", {"xln": "exact"})
    assert R.parse_component("xln") == ("xln", None) and R.parse_component("xln.fast") == ("xln", "fast") and R.parse_component("apb") == (None, None)
    for bad in ("xln.turbo", "xln.fast_layernorm", "xln.dtk_ln", "xln."):
        with pytest.raises(ValueError):
            R.strip_components("tg+sapb+" + bad + "@L1")
    with pytest.raises(ValueError):
        R.enable_xln("cueq")
    assert R.COMPONENT_WORDS["xln"][0] == "exact"                                                  # default: the provider's exact tier per cell


def test_the_adapter_parses_the_component_and_binds_the_provider():
    src = open(os.path.join(stack.fpf_home(), modes.FPF_ADAPTER_RELPATH), encoding="utf-8").read()
    assert re.search(r"[\"']xln[\"']", src) and "LR.enable_xln" in src and "LR.strip_components" in src


def _a_vouched_stack(cc, cell_key_part, arm):
    """A stack word on which the core recorded ``arm`` as bitwise for this card's cell (the exact record is per stack)."""
    card = {(9, 0): "H100:", (8, 0): "A100:"}[cc]
    for key, cell in (getattr(L, "CELLS", None) or {}).items():
        if key.startswith("%d.%d|" % cc) and cell_key_part in key:
            for stack in ((cell.get("vouched_on") or {}).get(arm) or ()):
                if stack.startswith(card):
                    return stack
    return card + "torch2.13.0+cu130/3.7.1/cueq0.11.1"                                          # the record stack of this package (STOCK.md)


def test_exact_word_takes_the_cells_bitwise_row_on_a_vouched_stack_and_aten_by_name_elsewhere(R, monkeypatch):
    """Contract by class (the exact record is stack-specific): on a stack where the core recorded the row's bitwise identity the
    exact word serves the bitwise-class row; on an unnamed stack the cell's stock arm (ATen) serves BY NAME, exit 0 — never a tolerance row."""
    for cc in ((9, 0), (8, 0)):
        monkeypatch.setitem(R._PROC, "stack", _a_vouched_stack(cc, "|fp32|pair_c128|N<=400|eager|fwd", "exactln"))
        s, r, cell = R.select_facts(cc, "fp32", 128, 400 * 400, 400, "fp32", "exact")             # the pairformer's c=128 pair LayerNorms, fp32 residual stream
        assert cell == "pair_c128" and r is None and (s.cls == "bitwise" and float(s.x_stock) >= 1.0 or s.row == "aten"), (cc, L.describe(s))
        assert s.row == "exactln", (cc, L.describe(s))                                            # the record stack is vouched: the bitwise row serves
        monkeypatch.setitem(R._PROC, "stack", _a_vouched_stack(cc, "|bf16w|pair_c128|N<=800|eager|fwd", "exactln:widen"))
        s, r, cell = R.select_facts(cc, "bf16", 128, 800 * 800, 800, "bf16w", "exact")            # a bf16 pair operand under autocast: the widen form
        assert (s.row, s.variant) == ("exactln", "widen") and r is None, (cc, L.describe(s))
        s, r, cell = R.select_facts(cc, "fp32", 384, 400, 400, "fp32", "exact")                   # the single track (N rows) in eager: ATen by name (the floor)
        assert cell == "single_c384" and s.row == "aten" and r is None, (cc, L.describe(s))
        monkeypatch.setitem(R._PROC, "stack", None)                                               # an unnamed stack: the stock arm by name, exit 0
        s, r, cell = R.select_facts(cc, "fp32", 128, 400 * 400, 400, "fp32", "exact")
        assert r is None and (s.row == "aten" or s.cls == "bitwise"), (cc, L.describe(s))


def test_capture_selection_never_names_a_capture_unsafe_row(R):
    for cc in ((9, 0), (8, 0)):
        for form, dt in (("fp32", "fp32"), ("bf16w", "bf16")):
            for word in ("exact", "fast", "big"):
                s, r, _ = R.select_facts(cc, dt, 128, 400 * 400, 400, form, word, capture=True)
                assert s.row not in L.CAPTURE_UNSAFE_ROWS and s.capture_safe, (cc, form, word, L.describe(s))


def test_a_row_word_the_form_refuses_binds_its_named_fallback(R, capsys):
    L = R._L()
    s, r, _ = R.select_facts((9, 0), "bf16", 128, 400 * 400, 400, "bf16w", "fastln")             # fastln (default variant) outputs x dtype: refused for the widen form BY NAME ->
    assert r is not None and r.row == "fastln" and r.kind.startswith("dtype:bf16w")                # the refusal's NAMED fallback is what binds, whatever row the core names there
    assert r.fallback and L.arm_word(s.row, s.variant) == r.fallback                               # the contract (the named fallback), not which word it is
    s, r, _ = R.select_facts((9, 0), "fp32", 128, 400 * 400, 400, "fp32", "exactln", aligned=False)   # misaligned rows: exactln's named fallback
    assert r is not None and r.row == "exactln" and "misaligned" in r.kind and r.fallback and L.arm_word(s.row, s.variant) == r.fallback


def test_cpu_calls_and_unserved_forms_route_to_the_modules_own_forward_by_name(R):
    torch = pytest.importorskip("torch")
    ln = torch.nn.LayerNorm(128)
    x = torch.randn(4, 4, 128)
    ref = ln(x)
    R.enable_xln("exact")
    assert torch.nn.LayerNorm.forward is R._ln_forward
    y = ln(x)                                                                                        # CPU: route form:cpu -> the module's own forward, same bytes
    assert torch.equal(y, ref)
    assert R.CENSUS["xln"].get("route:form:cpu") == 1
    assert R.form_of(torch, x.to(torch.bfloat16), ln.weight, ln.bias) == (None, "cpu")
    R.disable_xln()
    assert torch.nn.LayerNorm.forward is not R._ln_forward
    line = R.census_line("xln")
    assert line.startswith("[fpf_rf3 ln] CENSUS component=xln word=exact on=False served=0 stock_by_name=0 rows=- routes=form:cpu:1")


def test_form_words(R, monkeypatch):
    torch = pytest.importorskip("torch")

    class FakeCuda:
        def __init__(self, t): self.t = t
        def __getattr__(self, a): return getattr(self.t, a)
        is_cuda = True
    w32 = torch.ones(8); xb = FakeCuda(torch.zeros(2, 8, dtype=torch.bfloat16)); x32 = FakeCuda(torch.zeros(2, 8))
    assert R.form_of(torch, x32, w32, w32) == ("fp32", None)
    assert R.form_of(torch, x32, None, None) == ("fp32", None)
    assert R.form_of(torch, xb, w32, w32) == (None, "bf16_no_autocast")                           # bf16 x outside autocast: stock returns bf16, the widen form fp32 -> not served
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a, **k: True)                      # the CUDA bf16 autocast state RF3 runs under (no GPU here)
    monkeypatch.setattr(R, "_autocast_dtype", lambda t: t.bfloat16)
    assert R.form_of(torch, xb, w32, None) == ("bf16w", None)
    assert R.form_of(torch, xb, None, None) == (None, "bf16_noaffine_autocast")                     # AdaLN's affine-free norm: autocast's fp32 out, no widen row -> by name
    assert R.form_of(torch, xb, w32.to(torch.bfloat16), None) == (None, "bf16_params_bfloat16_autocast")
    assert R.form_of(torch, x32, w32, w32) == ("fp32", None)                                       # fp32 x under autocast: layer_norm runs fp32 either way


def test_census_line_counts_rows_routes_and_refusals(R, capsys):
    s, r, cell = R.select_facts((9, 0), "fp32", 128, 400 * 400, 400, "fp32", "exactln", aligned=False)
    ent = {"sel": s, "refusal": r, "call_refusal": None, "calls": 0, "cell": cell, "n": 400, "key": ("exactln", (9, 0), "fp32", 128, (400, 400, 128), False, False)}
    R._SEL[ent["key"]] = ent
    R._say_select(ent)
    out = capsys.readouterr().out
    assert "[fpf_rf3 ln] SELECT component=xln asked=exactln cc=9.0 form=fp32 C=128 shape=400x400x128 aligned=False capture=False cell=pair_c128 -> " + L.describe(s) in out
    assert "REFUSED row=exactln kind=aten_rowwise_path(misaligned rows) -> bound aten by name" in out
    R._count(ent, "fp32", 128); R._count(ent, "fp32", 128); R._route("bf16_no_autocast")
    line = R.census_line("xln")
    assert " served=0 stock_by_name=2 rows=aten|fp32|C128:2 routes=form:bf16_no_autocast:1 refused=exactln:aten_rowwise_path(misaligned rows)->aten:2" in line
    assert L.describe(s) in line
