"""transition_exact + pair_transition — the SwiGLU Transition bound BY TIER WORD to opt_core.kernels.transition (CPU: rows / registry / installers /
ORDER, the word rules, the provider's cell answers for atlasfold's shapes on both cards read through the provider's own select(), the per-call
refusal words against a stand-in module of the stock geometry, no kit-side floor / ceiling / pin left in the modules, install -> the stock statement
serving a CPU call by name with the stock output unchanged, a fast tier word installing the exact lever SKIPPED by name)."""
import pytest

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import transition_exact as X
from atlasfold_opt.hooks import pair_transition_fused as HP

STACK_H100 = "H100:torch2.7.1+cu128/3.3.1"                     # the kit's pinned stack word as the provider spells it (environment/requirements.lock)


def test_rows_registry_and_order():
    ex, fa, bg = (list(modes.levers_of(m)) for m in ("exact", "fast", "big"))
    for row in (ex, fa, bg):
        assert "transition_exact" in row
    assert "pair_transition" not in ex and "pair_transition" in fa and "pair_transition" in bg
    for row in (fa, bg):                                                                     # exact ⊂ fast: the fast binding is outermost (installed later)
        assert row.index("transition_exact") < row.index("pair_transition")
    assert ex.index("conf_transition_chunk") < ex.index("transition_exact")
    r = registry.LEVERS["transition_exact"]
    assert r["cls"] == "exact" and r["module"] == "atlasfold_opt.hooks.transition_exact" and r["strategy"] == X.NAME == "LOCAL.fused_transition"
    for w in ("stock_row:", "refused:", "dtype:", "training", "rank", "cpu"):
        assert w in r["expected"]
    p = registry.LEVERS["pair_transition"]
    assert p["cls"] == "fast" and p["module"] == "atlasfold_opt.hooks.pair_transition_fused" and p["strategy"] == HP.NAME
    for w in ("stock_row:", "refused:", "dtype:", "c:", "factor:", "training", "rank", "cpu"):
        assert w in p["expected"]
    for gone in ("below_min_tokens", "no-cell:", "word:"):                                   # the kit floor and the pair_fused cell words left with the by-number binding
        assert gone not in p["expected"]


def test_no_kit_side_floor_ceiling_or_pin_is_left():
    for mod in (X, HP):
        for name in ("PIN", "MIN_TOKENS", "MIN_TOKENS_ENV", "IDENTITY_FLOOR", "MAX_ROWS", "MAX_ROWS_ENV", "PROVIDER_WORDS", "min_tokens", "max_rows", "pin_for"):
            assert not hasattr(mod, name), (mod.__name__, name)
    assert not hasattr(HP, "WORD_DEFAULT") and X.WORD_DEFAULT == "exact" == X.word()
    assert HP.word("fast") == "fast" and HP.word("big") == "big" and HP.word("exact") == "fast"


def test_word_rules(monkeypatch):
    KT = pytest.importorskip("opt_core.kernels.transition")
    assert X.word_ok("exact", KT) is None and X.word_ok("faithful", KT) is None
    assert X.word_ok("fast", KT) == "word_not_exact_class:fast" and X.word_ok("big", KT) == "word_not_exact_class:big"
    assert X.word_ok("v1", KT) is None and X.word_ok("v1:lnfused", KT) is None            # a row word: the operator's pin, admitted where the provider admits it
    assert X.word_ok("torch_swiglu", KT).startswith("word_names_no_kernel") and X.word_ok("nope", KT) == "unknown_word:nope"
    monkeypatch.setenv(X.WORD_ENV, "v1")
    assert X.word() == "v1"
    monkeypatch.setenv(HP.WORD_ENV, "v2:fast")
    assert HP.word("big") == "v2:fast"


def test_provider_answers_for_atlasfold_shapes():
    """The provider's OWN answers at the kit's shapes (bf16, c 128 x 4 pair; c 384 / 768 single) — what the two levers print as plan=: the exact word
    serves a carried row only where the table records it bitwise on the running stack, the tier words fast / big name the cells' winners, the
    single track's cells name the statement.  Read through select(); nothing here pins a row."""
    KT = pytest.importorskip("opt_core.kernels.transition")
    for cc in ("9.0", "8.0"):
        for n in (256, 512, 896, 1280):
            for wd in ("fast", "big"):
                s = KT.select(wd, c=128, hidden=512, n_tokens=n, dtype="bf16", cc=cc, family="pair")
                assert s.tier == wd and (s.row in KT.STOCK_ROWS or s.cell_key.startswith(cc + "|bf16|pair_c128_n4|")), (cc, n, wd, s.line())
            try:
                e = KT.select("exact", c=128, hidden=512, n_tokens=n, dtype="bf16", cc=cc, family="pair", ln_given=True, stack=STACK_H100 if cc == "9.0" else None)
                assert e.row in KT.STOCK_ROWS or e.cls == "bitwise", (cc, n, e.line())       # an exact answer is the statement or a bitwise-recorded row — never a tolerance row
            except KT.Refusal as r:
                assert str(r.kind), (cc, n)                                                  # refused BY NAME (no vouch on this stack / below its floor): the statement runs
        for c in (384, 768):
            try:
                s = KT.select("exact", c=c, hidden=4 * c, n_tokens=896, dtype="bf16", cc=cc, family="single", ln_given=True)
                assert s.row in KT.STOCK_ROWS, (cc, c, s.line())
            except KT.Refusal as r:
                assert "no_cell" in str(r.kind) or str(r.kind), (cc, c)
    e = KT.select("exact", c=128, hidden=512, n_tokens=1280, dtype="bf16", cc="9.0", family="pair", ln_given=True, stack=STACK_H100)
    assert (e.row, e.cls) == ("v1", "bitwise"), e.line()                                      # the kit's stack is a recorded bitwise vouch of v1 at the 1280 bucket's cell


def _standin(C=128, n=4, torch=None):
    """A module of the stock Transition's geometry: Sequential(LayerNorm(C), SwiGLU(LinearNoBias C->2nC), LinearNoBias nC->C)."""
    class SwiGLU(torch.nn.Module):
        def __init__(self, c, k):
            super().__init__(); self.linear = torch.nn.Linear(c, 2 * k * c, bias=False)

        def forward(self, x):
            a, b = self.linear(x).chunk(2, dim=-1)
            return torch.nn.functional.silu(a) * b
    return torch.nn.Sequential(torch.nn.LayerNorm(C), SwiGLU(C, n), torch.nn.Linear(n * C, C, bias=False)).eval()


def test_refusal_words_cpu():
    torch = pytest.importorskip("torch")
    m = _standin(torch=torch)
    x32 = torch.zeros(1, 16, 16, 128); xb = x32.to(torch.bfloat16)
    assert X.geometry(m) == (128, 4) == HP.geometry(m)
    assert X.refusal(m, x32) == "dtype:float32" == HP.refusal(m, x32)
    assert X.refusal(m, xb) == "cpu" == HP.refusal(m, xb)                                    # no size floor: a 16-token CPU call is `cpu`, not `below_min_tokens`
    assert X.refusal(m, xb.reshape(-1, 128)) == "rank" == HP.refusal(m, xb.reshape(-1, 128))
    assert HP.refusal(_standin(384, 4, torch), torch.zeros(1, 16, 384, dtype=torch.bfloat16)) == "c:384"
    assert X.refusal(_standin(384, 4, torch), torch.zeros(1, 16, 384, dtype=torch.bfloat16)) == "cpu"   # the exact lever asks the provider for every width (its cells name the statement for c384 / c768)
    assert HP.refusal(_standin(128, 2, torch), xb) == "factor:2"
    m.train()
    assert X.refusal(m, xb) == "training" == HP.refusal(m, xb)
    KT = pytest.importorskip("opt_core.kernels.transition")
    W = X.weights(_standin(torch=torch), KT)
    assert (W.c, W.hidden) == (128, 512)
    mb = _standin(torch=torch); mb[2] = torch.nn.Linear(512, 128, bias=True)
    with pytest.raises(KT.Refusal):
        X.weights(mb, KT)


def test_stock_word_spelling():
    class S:                                                     # a Selection stand-in
        cell_key = "9.0|bf16|pair_c128_n4|N<=256|eager|fwd"
    assert X.stock_word(S(), 128, 256) == "stock_row:n256"
    assert X.stock_word(S(), 384, 896) == "stock_row:c384" and X.stock_word(S(), 768, 896) == "stock_row:c768"
    S.cell_key = None
    assert X.stock_word(S(), 128, 300) == "stock_row:n300"


def test_install_serves_a_cpu_call_by_name_and_a_fast_word_skips(monkeypatch):
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip(X.TARGET)
    pytest.importorskip("opt_core.kernels.transition")
    orig = tr.Transition.forward
    try:
        ins = X.install("exact", "[t]", {"mode": "exact"})
        assert ins.applied, ins.reason
        assert ins.facts["word"] == "exact" and ins.facts["pinned"] is False
        m = tr.Transition(128, 4).eval().to(torch.bfloat16)                    # module and input in one dtype: the stock statement runs on any CPU torch build
        x = torch.randn(1, 8, 8, 128).to(torch.bfloat16)
        with torch.no_grad():
            ref = orig(m, x); out = m(x)
        assert torch.equal(out, ref)
        L = ins.facts["ledger"]
        assert L.served == 0 and L.fallbacks == {"cpu": 1} and ins.gates[0]().ok
        line = ins.lines[0]()
        for tok in (" word=exact", " pinned=0", " plan=none", " rows=none", " copies=0", "state=skipped reason=all_fallback:cpu"):
            assert tok in line, line
        for gone in (" min_tokens=", " max_rows=", " pin=", " table_pin="):
            assert gone not in line, line
    finally:
        tr.Transition.forward = orig
    monkeypatch.setenv(X.WORD_ENV, "fast")
    try:
        ins = X.install("exact", "[t]", {"mode": "exact"})
        assert not ins.applied and ins.reason == "word_not_exact_class:fast"
    finally:
        tr.Transition.forward = orig


def test_pair_transition_install_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip(HP.TARGET)
    pytest.importorskip("opt_core.kernels.transition")
    orig = tr.Transition.forward
    for mode, wd in (("fast", "fast"), ("big", "big")):
        try:
            ins = HP.install(mode, "[t]", {"mode": mode})
            assert ins.applied, ins.reason
            assert ins.facts["word"] == wd and ins.facts["pinned"] is False and HP.STATE["word"] == wd
            m = tr.Transition(128, 4).eval().to(torch.bfloat16)
            x = torch.randn(1, 8, 8, 128).to(torch.bfloat16)
            with torch.no_grad():
                assert torch.equal(m(x), orig(m, x))
            line = ins.lines[0]()
            assert f" word={wd}" in line and " pinned=0" in line and "all_fallback:cpu" in line and " min_tokens=" not in line and " cells=" not in line, line
            assert ins.gates[0]().ok
        finally:
            tr.Transition.forward = orig
    monkeypatch.setenv(HP.WORD_ENV, "torch_swiglu")
    try:
        ins = HP.install("fast", "[t]", {})
        assert not ins.applied and ins.reason == "word_names_no_kernel:torch_swiglu"
    finally:
        tr.Transition.forward = orig
