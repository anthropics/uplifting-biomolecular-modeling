"""Lever transition's binding (kernels/af3_kernels.py _TRANS_PROV / _transition_decide / _transition_forward): the transition is served by the shared core's
transition provider (opt_core.kernels.transition) asked by the MODE's tier word (fast | big) per call class -- a kernel row through the provider face, a stock
row (the engine's own module measured fastest) or a refusal by name keeping the stock statement, each counted; this tree carries no copy of the kernel and no
launch table.  CPU only: a stand-in provider module records the calls the adapter makes; the REAL provider's table is asked what the tier words resolve to at
this model's four transition shapes on both cards at the ladder's token buckets (a kernel row, a stock row, or a refusal by name -- never an exception)."""
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from af3_torch_opt import registry  # noqa: E402
from af3_torch_opt.tests.test_tp_xfold_cpu import KIT  # noqa: E402

KERNELS = os.path.join(os.path.dirname(KIT), "kernels")
SHAPES = ((128, 512, "pair"), (64, 256, "rows"), (64, 128, "rows"), (384, 1536, "single"))     # pair x4, MSA c=64 x4, template pair c=64 x2, single x4
LADDER = (448, 832, 1216)


@pytest.fixture()
def K(monkeypatch):
    if KERNELS not in sys.path:
        monkeypatch.syspath_prepend(KERNELS)
    import af3_kernels as K_
    saved = dict(K_._TRANS_PROV); sel = dict(K_._TRANS_PROV["sel"]); tier = dict(K_._TIER)
    counts = {k: dict(v) for k, v in K_.COUNTS.items()}
    yield K_
    K_._TRANS_PROV.update(saved); K_._TRANS_PROV["sel"].clear(); K_._TRANS_PROV["sel"].update(sel); K_._TIER.clear(); K_._TIER.update(tier)
    for k in K_.COUNTS:
        K_.COUNTS[k].clear(); K_.COUNTS[k].update(counts.get(k, {}))


class _Mod(torch.nn.Module):            # the attributes of xfold's Transition the adapter packs (transition1 [2*HID, C], transition2 [C, HID], input_layer_norm)
    def __init__(self, C=128, factor=4):
        super().__init__()
        self.input_layer_norm = torch.nn.LayerNorm(C)
        self.transition1 = torch.nn.Linear(C, 2 * factor * C, bias=False)
        self.transition2 = torch.nn.Linear(factor * C, C, bias=False)


def _fake_provider(record, row="v2", refuse_kind=None, serve_refuse_kind=None):
    TP = types.ModuleType("fake_transition_provider")

    class Refusal(Exception):
        def __init__(self, kind, row=None, fallback="torch_swiglu"):
            super().__init__(kind); self.kind, self.row, self.fallback = kind, row, fallback

    class Sel:
        def __init__(self, row_, key): self.row, self.cell_key = row_, key

    def pack(**kw):
        record.append(("pack", sorted(kw))); return types.SimpleNamespace(**kw)

    def select(word, *, c, hidden, n_tokens=None, dtype="bf16", family="pair", residual=False, cc=None, stack=None, capture=False, rows_count=None, **_):
        record.append(("select", word, c, hidden, n_tokens, dtype, family, bool(residual), cc, bool(capture)))
        if refuse_kind:
            raise Refusal(refuse_kind, word, "torch_swiglu")
        return Sel(row, "%s|%s|c%d_h%d|N<=%s" % (cc, dtype, c, hidden, n_tokens))

    def transition(x, Wp, *, word, residual=False, n_tokens=None, family="pair", capture=False, stack=None, **_):
        record.append(("transition", word, bool(residual), tuple(x.shape), n_tokens, family))
        if serve_refuse_kind:
            raise Refusal(serve_refuse_kind, row, "torch_swiglu")
        return x * 3, Sel(row, None)

    TP.Refusal, TP.pack, TP.select, TP.transition = Refusal, pack, select, transition
    TP.STOCK_ROWS = ("torch_swiglu", "engine_module", "compile"); TP.TIER_WORDS = ("fast", "faithful", "exact", "big")
    return TP


def _bind(K, TP, word="fast", monkeypatch=None):
    K._TRANS_PROV.update(mod=TP, cc="9.0", stack="H100:test", refused=None); K._TRANS_PROV["sel"].clear(); K._TIER["word"] = word
    monkeypatch.setattr(K, "_autocast_bf16", lambda: True)
    monkeypatch.setattr(K, "_capturing", lambda: False)
    K._ORIG["transition"] = lambda self, x: x * 0 + 7                       # the stock statement's stand-in
    for k in K.COUNTS: K.COUNTS[k].clear()


def _cuda_like(t):
    """A CPU tensor the adapter treats as on-device (is_cuda is read, nothing is launched: the provider is a stand-in)."""
    class T(torch.Tensor):
        @property
        def is_cuda(self):  # noqa: D401
            return True
    return t.as_subclass(T)


def test_the_tier_word_is_asked_per_call_class_and_the_face_serves(K, monkeypatch):
    rec = []; TP = _fake_provider(rec); _bind(K, TP, "fast", monkeypatch)
    m = _Mod(); x = _cuda_like(torch.randn(2, 16, 16, 128, dtype=torch.bfloat16))
    out = K._transition_forward(m, x)
    assert torch.equal(out, x * 3)                                                                   # the face served
    sel = [r for r in rec if r[0] == "select"]; tr = [r for r in rec if r[0] == "transition"]
    assert sel == [("select", "fast", 128, 512, 16, "bf16", "pair", False, "9.0", False)]           # the MODE's tier word, the pair family, tokens = x.shape[-2]
    assert tr == [("transition", "fast", False, (2, 16, 16, 128), 16, "pair")]
    assert K.COUNTS["transition"].get("served:c=128") == 1 and any(k.startswith("word:fast=v2@") for k in K.COUNTS["transition"])
    K._transition_forward(m, x)                                                                      # the same class again: no second select (decided once per class)
    assert len([r for r in rec if r[0] == "select"]) == 1 and K.COUNTS["transition"]["served:c=128"] == 2


def test_big_asks_big_and_the_families_follow_the_width(K, monkeypatch):
    rec = []; TP = _fake_provider(rec); _bind(K, TP, "big", monkeypatch)
    K._transition_forward(_Mod(64, 4), _cuda_like(torch.randn(8, 24, 64, dtype=torch.bfloat16)))              # MSA transition: c=64 x4, [S, N, 64]
    K._transition_forward(_Mod(64, 2), _cuda_like(torch.randn(24, 24, 64, dtype=torch.bfloat16)))             # template pair stack: c=64 x2
    K._transition_forward(_Mod(384, 4), _cuda_like(torch.randn(24, 384, dtype=torch.bfloat16)))               # single: c=384, [N, 384]
    sel = [r[1:7] for r in rec if r[0] == "select"]
    assert sel == [("big", 64, 256, 24, "bf16", "rows"), ("big", 64, 128, 24, "bf16", "rows"), ("big", 384, 1536, 24, "bf16", "single")]


def test_residual_is_folded_by_the_row_and_counted_for_resid_fold(K, monkeypatch):
    rec = []; TP = _fake_provider(rec); _bind(K, TP, "fast", monkeypatch)
    x = _cuda_like(torch.randn(4, 4, 128, dtype=torch.bfloat16))
    K._transition_forward(_Mod(), x, residual=True)
    assert [r for r in rec if r[0] == "transition"][0][2] is True and K.COUNTS["resid_fold"].get("served:transition_c=128") == 1
    xf = _cuda_like(torch.randn(4, 4, 128, dtype=torch.float32))                                     # an fp32 x: the folded add is only stock's bytes on bf16 -> torch's add after the update
    expect = xf * 4                                                                                    # x + update (the stand-in's update = 3x), torch's add (in place on x, as the block's)
    out = K._transition_forward(_Mod(), xf, residual=True)
    assert K.COUNTS["transition"].get("residual_unfused:torch.float32") == 1 and torch.equal(out, expect)
    assert [r for r in rec if r[0] == "transition"][-1][2] is False                                   # the face was asked for the update alone


def test_a_stock_row_keeps_the_stock_statement_counted(K, monkeypatch):
    rec = []; TP = _fake_provider(rec, row="torch_swiglu"); _bind(K, TP, "fast", monkeypatch)
    x = _cuda_like(torch.randn(24, 384, dtype=torch.bfloat16))
    out = K._transition_forward(_Mod(384, 4), x)
    assert torch.equal(out, x * 0 + 7) and not [r for r in rec if r[0] == "transition"]               # the engine's own module served; the face was not called
    assert K.COUNTS["transition"] .get("fallback:c=384,stock-row") == 1 and registry.fallback_expected("transition", "fallback:c=384,stock-row")


def test_a_refusal_by_name_keeps_the_stock_statement_once_per_class(K, monkeypatch):
    rec = []; TP = _fake_provider(rec, refuse_kind="no_cell:8.0|bf16|rows_c64_n2|eager|fwd"); _bind(K, TP, "fast", monkeypatch)
    m = _Mod(64, 2); x = _cuda_like(torch.randn(24, 24, 64, dtype=torch.bfloat16))
    assert torch.equal(K._transition_forward(m, x), x * 0 + 7) and torch.equal(K._transition_forward(m, x), x * 0 + 7)
    assert len([r for r in rec if r[0] == "select"]) == 1                                            # asked once for the class; the refusal is remembered
    assert K.COUNTS["transition"].get("fallback:c=64,no_cell") == 2 and K.COUNTS["transition"].get("refused:fast:no_cell:8.0|bf16|rows_c64_n2|eager|fwd") == 1
    assert K._TRANS_PROV["refused"] == "fast:no_cell:8.0|bf16|rows_c64_n2|eager|fwd" and registry.fallback_expected("transition", "fallback:c=64,no_cell")


def test_a_serve_time_refusal_is_named_and_remembered(K, monkeypatch):
    rec = []; TP = _fake_provider(rec, serve_refuse_kind="v2:build_failed:smem"); _bind(K, TP, "fast", monkeypatch)
    m = _Mod(); x = _cuda_like(torch.randn(4, 4, 128, dtype=torch.bfloat16))
    assert torch.equal(K._transition_forward(m, x), x * 0 + 7) and torch.equal(K._transition_forward(m, x), x * 0 + 7)
    assert len([r for r in rec if r[0] == "transition"]) == 1                                        # the face raised once; the class keeps the stock statement from there
    assert K.COUNTS["transition"].get("fallback:c=128,v2") == 2 and "transition" not in K._DEAD       # named per call; the lever is NOT disabled (a refusal is not a kernel error)


def test_no_provider_and_foreign_dtypes_step_aside_by_name(K, monkeypatch):
    rec = []; TP = _fake_provider(rec); _bind(K, TP, "fast", monkeypatch)
    K._TRANS_PROV["mod"] = None
    x = _cuda_like(torch.randn(4, 4, 128, dtype=torch.bfloat16))
    assert torch.equal(K._transition_forward(_Mod(), x), x * 0 + 7) and K.COUNTS["transition"].get("fallback:no-provider") == 1
    K._TRANS_PROV["mod"] = TP
    xh = _cuda_like(torch.randn(4, 4, 128, dtype=torch.float16))
    assert torch.equal(K._transition_forward(_Mod(), xh), xh * 0 + 7) and K.COUNTS["transition"].get("fallback:dtype=torch.float16") == 1


def test_the_tree_carries_no_copy_and_no_launch_table():
    names = set()
    for d, _, fs in os.walk(KERNELS):
        names.update(fs)
    assert not names & {"af3_fused.py", "af3t_pins.py", "af3t_launch_pins.json"}, names & {"af3_fused.py", "af3t_pins.py", "af3t_launch_pins.json"}
    src = open(os.path.join(KERNELS, "af3_kernels.py"), encoding="utf-8").read()
    assert "import af3_fused" not in src and "af3t_pins" not in src and "_TR_MIN_ROWS" not in src and "transition_padded" not in src
    assert 'TP.transition(x, _transition_pack(self, TP), word=_TIER["word"]' in src                                 # the MODE's tier word, through the face
    assert registry.IMPL["transition"] == ("opt_core.kernels.transition:tier", "core")
    assert registry.EXPECTED_FALLBACKS["transition"] == ("fallback:c=384,stock-row", "fallback:c=128,stock-row", "fallback:c=64,stock-row", "fallback:c=64,no_cell")


def test_the_real_provider_resolves_the_tier_words_at_this_models_shapes_by_name():
    """The shared core's own table, asked on the CPU for both cards at the ladder's buckets: every (tier word, shape, size) resolves to a kernel row or a stock row,
    or refuses BY NAME (Refusal) -- nothing raises otherwise; the words the start-up line prints come from these decisions."""
    TP = pytest.importorskip("opt_core.kernels.transition")
    assert callable(TP.select) and callable(TP.transition) and callable(TP.pack) and TP.STOCK_ROWS and {"fast", "big", "exact"} <= set(TP.TIER_WORDS)
    stacks = {"9.0": "H100:torch2.13.0+cu130/3.7.1", "8.0": "A100:torch2.13.0+cu130/3.7.1"}         # the image's stack words (definitions: torch 2.13.0+cu130, triton 3.7.1)
    seen = {}
    for cc, stack in stacks.items():
        for word in ("fast", "big"):
            for (c, h, fam) in SHAPES:
                for n in LADDER:
                    try:
                        s = TP.select(word, c=c, hidden=h, n_tokens=n, dtype="bf16", family=fam, residual=(c != 384), cc=cc, stack=stack)
                        seen[(cc, word, c, h, n)] = s.row
                        assert isinstance(s.row, str) and s.row
                    except TP.Refusal as r:
                        seen[(cc, word, c, h, n)] = "REFUSED:" + str(r.kind).split(":")[0]
    kernel_rows = [v for v in seen.values() if not v.startswith("REFUSED") and v not in TP.STOCK_ROWS]
    assert kernel_rows, seen                                                                          # the table serves kernel rows at this model's shapes
    for (cc, word, c, h, n), v in seen.items():                                                       # the pair transition (the trunk's N^2 stream) is a kernel row on both cards in both tiers
        if c == 128:
            assert not v.startswith("REFUSED") and v not in TP.STOCK_ROWS, ((cc, word, c, h, n), v)
