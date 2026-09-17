"""The triangle-attention rows of the FPF add-on (``rf3fpf/fpf_rf3_triattn.py``, the shared core's provider ``opt_core.kernels.triattn``):
the arm sub-word grammar (``gflash.<word>``, ``xatt[.<word>]``) leaves today's arms byte for byte; a row word outside a component's vocabulary
is refused by name; a row the provider refuses on this card / stack is bound to its NAMED fallback and the refusal is printed and counted (never a
silent substitute); the SELECT / CENSUS lines carry the provider's own ``describe()``.  CPU only: selections over synthetic tensor facts."""
import os
import re
import sys

import pytest

from rosettafold3_opt import modes, stack

T = pytest.importorskip("opt_core.kernels.triattn")
RF3FPF = os.path.join(stack.fpf_home(), "rf3fpf")


@pytest.fixture
def R(monkeypatch):
    monkeypatch.syspath_prepend(RF3FPF)
    sys.modules.pop("fpf_rf3_triattn", None)
    import fpf_rf3_triattn as R
    R.reset_census()
    for c in R.STATE:
        R.STATE[c]["word"] = R.DEFAULT_WORD[c]; R.STATE[c]["on"] = False
    monkeypatch.setattr(R, "_stock", lambda: None)
    yield R
    R.reset_census()
    sys.modules.pop("fpf_rf3_triattn", None)


def test_todays_arms_are_untouched_by_the_sub_word_grammar(R):
    for arm in ("fast+gflash+ttr+apb+res@L1", "tg+sapb@L1", "fast+apb+res@L1", "fast", "stock+tg@L1"):
        assert R.strip_components(arm) == (arm, {"gflash": None, "xatt": None}), arm


def test_sub_words_parse_and_unknown_words_are_refused_by_name(R):
    assert R.strip_components("fast+gflash.k2b+ttr+apb+res@L1") == ("fast+gflash+ttr+apb+res@L1", {"gflash": "k2b", "xatt": None})
    assert R.strip_components("fast+gflash.cuda_sm90a+ttr@L1") == ("fast+gflash+ttr@L1", {"gflash": "cuda_sm90a", "xatt": None})
    assert R.strip_components("fast+gflash.triattn_native+ttr+apb+res+tg@L1.warm") == ("fast+gflash+ttr+apb+res+tg@L1.warm", {"gflash": "triattn_native", "xatt": None})
    assert R.strip_components("tg+sapb+xatt@L1") == ("tg+sapb@L1", {"gflash": None, "xatt": "exact"})
    assert R.strip_components("tg+sapb+xatt.exact_headsplit@L1") == ("tg+sapb@L1", {"gflash": None, "xatt": "exact_headsplit"})
    assert R.strip_components("xatt@L1") == ("stock@L1", {"gflash": None, "xatt": "exact"})
    assert R.parse_component("gflash") == ("gflash", None) and R.parse_component("gflash.flash") == ("gflash", "flash") and R.parse_component("ttr") == (None, None)
    for bad in ("gflash.turbo", "xatt.k2b", "gflash.exact_headsplit", "xatt.fast", "gflash."):
        with pytest.raises(ValueError):
            R.strip_components("fast+" + bad + "@L1")
    with pytest.raises(R.WordError):
        R.set_gflash_word("cueq")
    assert R.COMPONENT_WORDS["gflash"][0] == "card" and R.COMPONENT_WORDS["xatt"][0] == "exact"       # defaults: the device's card row; the cell's exact row
    assert R.TIER_DEFAULT["gflash"] == "fast" and R.card_word("gflash", (10, 0)) == "fast" == R.card_word("gflash", (8, 0)) and R.FORM == "bias_only"   # no per-card table: the tier word on every device   # mode rows bind by TIER word on every card
    assert R.card_word("gflash", (9, 0)) == "fast" and R.card_word("gflash", (8, 0)) == "fast" and R.card_word("gflash", (12, 0)) == "fast"
    assert R.card_word("gflash", ("9", "0")) == "fast" and R.card_word("gflash", None) == "fast"      # the tier word on every device (no per-card table)


def test_the_adapter_parses_the_component_and_binds_the_provider():
    src = open(os.path.join(stack.fpf_home(), modes.FPF_ADAPTER_RELPATH), encoding="utf-8").read()
    assert re.search(r"[\"']xatt[\"']", src) and "R.gflash_kernel" in src and "strip_components" in src


def test_the_flash_word_is_the_flash_row_by_name(R, capsys):
    R.set_gflash_word("flash")
    ent = R._select("gflash", ((9, 0), "bf16", 32, 4, 400, "fwd"))
    assert ent["sel"].row == "flash" and ent["refusal"] is None
    out = capsys.readouterr().out
    assert "[fpf_rf3 triattn] SELECT component=gflash asked=flash form=bias_only cc=9.0 dtype=bf16 H=4 D=32 n=400 " + T.describe(ent["sel"]) in out
    assert "cell=9.0|bf16|D32|H4|N<=400|fwd" in out and "REFUSED" not in out


def test_a_bare_gflash_asks_the_devices_card_row(R, monkeypatch, capsys):
    """No sub-word (the word ``card``): 9.0 and 8.0 ask the tier word fast with the call-form hint (the table's row differs per card), another
    device flash; the SELECT line
    names the card word and what the provider bound (a refusal -> the cell's fallback by name); the census word reads ``card-><word>``."""
    asked = []

    class Sel:
        def __init__(self, row): self.row = row

    class FakeT:
        class Refusal(Exception):
            pass
        @staticmethod
        def select(cc, dtype, D, H, S, direction, word=None, stack=None, prefer=None, form=None):
            asked.append((tuple(cc), word, form)); return Sel({("fast", (9, 0)): "triattn_native", ("fast", (8, 0)): "k2b"}.get((word, tuple(cc)), word))
        @staticmethod
        def describe(sel): return f"row={sel.row}"
    monkeypatch.setattr(R, "_T", lambda: FakeT)
    monkeypatch.setattr(R, "_stack_key", lambda cc, word, prefer=None: None)
    assert R.set_gflash_word(None) == "card" and R.STATE["gflash"]["word"] == "card"
    assert R._select("gflash", ((9, 0), "bf16", 32, 4, 1200, "fwd"))["sel"].row == "triattn_native"
    assert R._select("gflash", ((9, 0), "bf16", 64, 4, 1200, "fwd"))["sel"].row == "triattn_native"          # the template track's head_dim asks the same row (the provider refuses by name if it cannot)
    assert R._select("gflash", ((8, 0), "bf16", 32, 4, 1200, "fwd"))["sel"].row == "k2b"                 # the tier word; the (fake, as the real) table's 8.0 fast row is k2b
    assert R._select("gflash", ((10, 0), "bf16", 32, 4, 1200, "fwd"))["sel"].row == "fast"                # (the fake table echoes the word on a device it has no row for; the real provider names a row or refuses by name)
    assert asked == [((9, 0), "fast", "bias_only"), ((9, 0), "fast", "bias_only"), ((8, 0), "fast", "bias_only"), ((10, 0), "fast", "bias_only")]   # the tier word on every device
    out = capsys.readouterr().out
    assert "SELECT component=gflash asked=card->fast form=bias_only cc=9.0 dtype=bf16 H=4 D=32 n=1200 row=triattn_native" in out
    assert "asked=card->fast form=bias_only cc=8.0" in out and "asked=card->fast form=bias_only cc=10.0" in out
    assert "asked=card->fast form=bias_only cc=8.0 dtype=bf16 H=4 D=32 n=1200 row=k2b" in out
    assert R.census("gflash")["word"] == "card->fast"


def test_cuda_row_without_a_prebuilt_for_this_abi_binds_k2b_by_name(R, monkeypatch, capsys):
    monkeypatch.setattr(R, "_stack_key", lambda cc, word, prefer=None: "torch9.9.9+cu999-cpython-399-none")     # this kit's interpreter: no prebuilt directory of that key
    R.set_gflash_word("cuda_sm90a")
    ent = R._select("gflash", ((9, 0), "bf16", 32, 4, 1200, "fwd"))
    assert ent["sel"].row == "k2b"
    assert ent["refusal"] is not None and ent["refusal"].row == "cuda_sm90a" and ent["refusal"].kind == "no_prebuilt:torch9.9.9+cu999-cpython-399-none"
    out = capsys.readouterr().out
    assert "REFUSED row=cuda_sm90a kind=no_prebuilt:torch9.9.9+cu999-cpython-399-none -> bound k2b by name" in out
    R._count("gflash", ent, "bf16", 32, 4); R._count("gflash", ent, "bf16", 32, 4)
    line = R.census_line("gflash")
    assert line.startswith("[fpf_rf3 triattn] CENSUS component=gflash word=cuda_sm90a on=True served=2 rows=k2b|bf16|D32|H4:2 refused=cuda_sm90a:no_prebuilt:torch9.9.9+cu999-cpython-399-none->k2b:2")
    assert T.describe(ent["sel"]) in line
    assert R.census("gflash")["served"] == 2 and R.census("gflash")["errors"] == 0


def test_exact_word_takes_the_cells_exact_row_per_card(R, capsys):
    R.STATE["xatt"]["word"] = "exact"
    h100 = R._select("xatt", ((9, 0), "bf16", 32, 4, 400, "fwd"))
    assert h100["sel"].row in ("triattn_exact", "cueq") and h100["sel"].cls in ("exact", "stock") and h100["refusal"] is None   # cc 9.0: the bit-identical row where this stack is vouched, else the stock op by name -- never a tolerance-class row
    a100 = R._select("xatt", ((8, 0), "bf16", 32, 4, 400, "fwd"))
    assert a100["sel"].row in ("triattn_exact", "cueq") and a100["sel"].cls in ("exact", "stock") and a100["refusal"] is None   # cc 8.0: likewise (A100 80GB vouched stacks)
    small = R._select("xatt", ((9, 0), "bf16", 32, 4, 64, "fwd"))
    assert small["sel"].row == "cueq"                                                            # below the row's smallest proven size: the stock op, by name
    out = capsys.readouterr().out
    assert "component=xatt asked=exact form=bias_only cc=8.0" in out and "row=" in out


def test_a_row_word_refused_on_the_card_binds_the_stock_op_by_name(R, capsys, monkeypatch):
    T = R._T()
    stock_select = T.select
    def select(cc, dtype, D, H, S, direction="fwd", *, word, **kw):      # a row word the provider refuses on this card (kind arch), whatever its name
        if word == "rowx":
            raise T.Refusal("arch:sm_80", "rowx", "cueq")
        return stock_select(cc, dtype, D, H, S, direction, word=word, **kw)
    monkeypatch.setattr(T, "select", select)
    R.STATE["xatt"]["word"] = "rowx"
    ent = R._select("xatt", ((8, 0), "bf16", 32, 4, 800, "fwd"))
    assert ent["sel"].row == "cueq" and ent["refusal"].kind == "arch:sm_80" and ent["refusal"].fallback == "cueq"
    assert "REFUSED row=rowx kind=arch:sm_80 -> bound cueq by name" in capsys.readouterr().out
    R._count("xatt", ent, "bf16", 32, 4)
    R.CENSUS["xatt"]["error:rowx"] = 3
    line = R.census_line("xatt")
    assert " served=1 rows=cueq|bf16|D32|H4:1 refused=rowx:arch:sm_80->cueq:1 errors=3(rowx:3) selections=" in line
    assert R.census("xatt")["errors"] == 3


def test_selection_is_memoised_per_shape_and_word(R, capsys):
    a = R._select("gflash", ((9, 0), "bf16", 32, 4, 400, "fwd"))
    b = R._select("gflash", ((9, 0), "bf16", 32, 4, 400, "fwd"))
    c = R._select("gflash", ((9, 0), "bf16", 32, 4, 1200, "fwd"))
    assert a is b and a is not c
    assert capsys.readouterr().out.count("SELECT component=gflash") == 2


def test_xatt_passes_small_sequences_to_the_stock_op_by_name(R, monkeypatch):
    """cuEquivariance serves S <= CUEQ_TRIATTN_FALLBACK_THRESHOLD through its own torch reference path, not its CUDA kernel: the seam hands
    those calls to the stock op unchanged and counts them apart (passthrough), never to the provider."""
    import types
    fake = types.ModuleType("cuequivariance_ops_torch.triangle_attention"); fake.CUEQ_TRIATTN_FALLBACK_THRESHOLD = 100
    monkeypatch.setitem(sys.modules, "cuequivariance_ops_torch.triangle_attention", fake)
    calls = []

    class Real:
        def triangle_attention(self, q, k, v, bias=None, mask=None, scale=None, **kw):
            calls.append(q.shape); return "stock"
        other = "library attribute"

    class Q:                                   # a stand-in tensor: only .shape is read on this path
        def __init__(self, S): self.shape = (1, S, 4, S, 32)

    monkeypatch.setattr(R, "serve", lambda *a, **k: "provider")
    px = R.CuetProxy(Real())
    assert px._thr == 100 and px.other == "library attribute"
    assert px.triangle_attention(Q(61), Q(61), Q(61)) == "stock" and px.triangle_attention(Q(100), Q(100), Q(100)) == "stock"
    assert px.triangle_attention(Q(101), Q(101), Q(101)) == "provider" and px.triangle_attention(Q(400), Q(400), Q(400)) == "provider"
    assert calls == [(1, 61, 4, 61, 32), (1, 100, 4, 100, 32)] and R.CENSUS["xatt"] == {"passthrough:S<=100": 2}
    assert R.census("xatt")["served"] == 0            # passthrough calls are not the provider's
    monkeypatch.delitem(sys.modules, "cuequivariance_ops_torch.triangle_attention")
    assert R.cueq_fallback_threshold(default=77) in (77, 100)   # absent module -> the default (or the real library's constant when installed here)


def test_on_the_a100_the_tier_word_resolves_by_class_never_raises_and_never_names_the_sm90_only_exact_row(R):
    """The CONTRACT by class, not the core's current per-cell winners (those change whenever the cc 8.0 rows are re-measured). cc 8.0, the bare
    component's word `fast` (form bias_only) at RF3's two head dims and the ladder sizes: the provider never raises, binds a fast-class row or
    names the stock op (class stock) per cell, and a measured fast-class cell reads x_stock >= 1.00.
    Winners at the time of writing, for orientation only: D32 -> k2b at every size; D64 -> the stock op by name at <= 400 tokens, k2b above."""
    R.set_gflash_word(None)
    for D in (32, 64):
        for n in (100, 400, 800, 1200, 2048, 3000):
            ent = R._select("gflash", ((8, 0), "bf16", D, 4, n, "fwd"))
            sel = ent["sel"]
            assert ent["refusal"] is None, (D, n, T.describe(sel))
            assert sel.cls in ("fast", "stock"), (D, n, T.describe(sel))
            assert sel.cls == "stock" or sel.x_stock is None or float(sel.x_stock) >= 1.0, (D, n, T.describe(sel))
            assert (sel.cls == "stock") == (sel.row == "cueq"), (D, n, T.describe(sel))                          # a named stock row is the stock op, by name
