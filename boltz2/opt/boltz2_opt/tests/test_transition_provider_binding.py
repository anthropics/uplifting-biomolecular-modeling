"""transition.apply(): every admissible call is handed to the core transition provider BY THE ROW'S WORD with this card, this stack and the
call's (C, hidden, N, rows); what comes back decides — a row serves (exact class: fed the module's own LayerNorm output), the stock arm or a
by-name refusal puts the call on the module's original forward under a declared census word. CPU is enough (a stand-in provider)."""
import sys
import types

import pytest

from .. import transition as TR


def _fake_world(monkeypatch, calls, answer, cc=(9, 0)):
    """A stand-in boltz Transition + a stand-in provider whose select() answers `answer(word, kw)` (a row name, 'torch_swiglu', or raises its
    Refusal) + a recording ledger, with the autocast / CUDA predicates forced true and the card answering `cc`; returns (torch, Transition, ledger, KT)."""
    torch = pytest.importorskip("torch")

    class Transition(torch.nn.Module):                               # boltz.model.layers.transition.Transition's surface
        def __init__(self, dim, hidden):
            super().__init__(); self.norm = torch.nn.LayerNorm(dim); self.hidden = hidden
            self.fc1 = torch.nn.Linear(dim, hidden, bias=False); self.fc2 = torch.nn.Linear(dim, hidden, bias=False); self.fc3 = torch.nn.Linear(hidden, dim, bias=False)

        def forward(self, x, chunk_size=None):
            calls["orig"].append((int(x.shape[-1]), self.hidden)); return x + 1.0

    fake_boltz = types.ModuleType("boltz.model.layers.transition"); fake_boltz.Transition = Transition
    chain = ("boltz", "boltz.model", "boltz.model.layers", "boltz.model.layers.transition")
    mods = {name: sys.modules.get(name) or types.ModuleType(name) for name in chain[:-1]}; mods[chain[-1]] = fake_boltz
    for name in chain:
        monkeypatch.setitem(sys.modules, name, mods[name])
    for parent, child in zip(chain[:-1], chain[1:]):
        monkeypatch.setattr(mods[parent], child.rsplit(".", 1)[1], mods[child], raising=False)

    class _Ledger:
        def __init__(self): self.fell = []; self.served = []; self.facts = {}
        def fallback(self, w): self.fell.append(w)
        def serve(self, key, impl=None): self.served.append((key, impl))
        def set(self, k, v): self.facts[k] = v
        def error(self, e): raise AssertionError(e)

    led = _Ledger()
    fake_pf = types.SimpleNamespace(ledger=lambda name, expected=(): led, emit_line=lambda L, tag, **kw: f"[{tag}] LEVER name=fused_transition " + " ".join(f"{k}={v}" for k, v in kw.items()))
    import opt_core.attn as _attn
    monkeypatch.setattr(_attn, "pair_fused", fake_pf, raising=False); monkeypatch.setitem(sys.modules, "opt_core.attn.pair_fused", fake_pf)

    class Refusal(Exception):
        def __init__(self, kind, word=None, fallback=None): super().__init__(kind); self.kind = kind; self.row = word; self.fallback = fallback

    class KT:                                                          # opt_core.kernels.transition's surface the adapter binds
        STOCK_ROWS = ("torch_swiglu", "engine_module", "compile")
        Refusal = None
        seen = []

        @staticmethod
        def stack_word(device=None): return "H100:torch2.12.0+cu130/3.7.0"

        @staticmethod
        def pack(**kw): return types.SimpleNamespace(hidden=int(kw["w_a"].shape[0]), c=int(kw["w_a"].shape[1]))

        @staticmethod
        def select(word, **kw):
            KT.seen.append((word, dict(kw))); row = answer(word, kw)
            return types.SimpleNamespace(row=row, variant=None, cfg=None)

        @staticmethod
        def transition(x, W, word=None, residual=False, x_ln=None, n_tokens=None, family=None, stack=None):
            calls["cell"].append((int(x.shape[-1]), W.hidden, word, x_ln is not None, stack)); row = answer(word, {})
            return x + 2.0, types.SimpleNamespace(row=row, variant=None, cfg=None)
    KT.Refusal = Refusal
    import opt_core.kernels as _kern
    monkeypatch.setattr(_kern, "transition", KT, raising=False); monkeypatch.setitem(sys.modules, "opt_core.kernels.transition", KT)
    monkeypatch.setattr(TR, "_STATE", dict(TR._STATE, ledger=None, applied=[], errors={}, served_by={}, core_rows={}))
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a: True); monkeypatch.setattr(torch, "get_autocast_dtype", lambda dev: torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True); monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cc)
    return torch, Transition, led, KT


def test_exact_hands_every_call_to_the_provider_by_the_tier_word_and_feeds_the_module_layernorm(monkeypatch):
    calls = {"orig": [], "cell": []}
    torch, Transition, led, KT = _fake_world(monkeypatch, calls, lambda w, kw: "v1", cc=(8, 0))
    assert TR.apply("exact") == ["fused_transition"]
    y = Transition(128, 512)(torch.zeros(1, 20, 20, 128))
    word, kw = KT.seen[-1]
    assert word == "exact" and kw["c"] == 128 and kw["hidden"] == 512 and kw["n_tokens"] == 20 and kw["rows_count"] == 400 and kw["family"] == "pair"
    assert kw["cc"] == "8.0" and kw["stack"] == "H100:torch2.12.0+cu130/3.7.0" and kw["ln_given"] is True, "the exact tier's vouch is checked for THIS card and stack; no kit floor decides first"
    assert calls["cell"] == [(128, 512, "exact", True, "H100:torch2.12.0+cu130/3.7.0")] and calls["orig"] == [] and float(y.flatten()[0]) == 2.0
    assert led.served == [("20x128", "unchunked")] and led.fell == [] and TR._STATE["core_rows"] == {"v1": 1}
    m64 = Transition(64, 256); m64(torch.zeros(1, 4, 20, 64))            # the MSA rows of width 64: handed over too (family rows), the provider decides
    assert KT.seen[-1][1]["family"] == "rows" and KT.seen[-1][1]["c"] == 64
    assert "core_rows=v1:2" in TR.line() and "impl=opt_core.kernels.transition" in TR.line() and "variant=exact" in TR.line()


def test_a_refusal_or_the_stock_arm_puts_the_call_on_the_module_forward_under_a_declared_word(monkeypatch):
    for kind, word in (("exact_vouch_not_recorded_on_A100:torch2.12.0+cu130/3.7.0", "exact_unvouched"), ("exact_vouch_below_57_tokens_on_A100:x", "exact_unvouched"),
                       ("no_cell:8.0|bf16|pair_c128_n2|eager|fwd", "no_cell"), ("STOCK", "core_stock"), ("dtype:fp32", "core_refused:dtype:fp32")):
        calls = {"orig": [], "cell": []}
        with monkeypatch.context() as mp:
            box = {}
            def answer(w, kw, kind=kind):
                if kind == "STOCK":
                    return "torch_swiglu"
                raise box["KT"].Refusal(kind, w, "torch_swiglu")
            torch, Transition, led, KT = _fake_world(mp, calls, answer); box["KT"] = KT
            assert TR.apply("exact") == ["fused_transition"]
            y = Transition(128, 256)(torch.zeros(1, 8, 8, 128))
            assert calls["orig"] == [(128, 256)] and calls["cell"] == [] and led.fell == [word] and float(y.flatten()[0]) == 1.0, (kind, led.fell)
            assert (word in TR.EXPECTED) == (not word.startswith("core_refused:")), "declared words pass the gate; an undeclared refusal refuses it by its word"


def test_chunked_and_autocast_rules_by_class(monkeypatch):
    """exact class: a hidden-chunked call takes the module (another rounding chain), counted `chunked`; fast / big serve it unchunked."""
    for variant, chunk_served in (("exact", False), ("core.v1", False), ("fast", True), ("big", True), ("core.v2:fast", True)):
        calls = {"orig": [], "cell": []}
        with monkeypatch.context() as mp:
            torch, Transition, led, KT = _fake_world(mp, calls, lambda w, kw: "v2")
            assert TR.apply(variant) == ["fused_transition"]
            Transition(128, 512)(torch.zeros(1, 8, 8, 128), chunk_size=64)
            if chunk_served:
                assert led.served == [("8x128", "chunked_as_one")] and led.fell == [] and KT.seen[-1][0] == TR.core_word(variant) and KT.seen[-1][1]["ln_given"] is False, variant
            else:
                assert led.served == [] and led.fell == ["chunked"], variant
