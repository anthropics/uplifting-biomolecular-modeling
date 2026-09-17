"""triattn_exact — the stock triangle-attention call through opt_core.kernels.triattn under an exact word (CPU: the provider's select is pure;
on a CPU tensor the cell's exact row is the stock op, so the served path is exercised as its named step-aside): rows, words, install / rebind /
unwrap, the stock_row / rank fallbacks calling the bound stock op unchanged, an unknown or fast word installing skipped by name."""
import pytest

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import triattn_exact as X


def test_rows_and_registry():
    assert "triattn_exact" in modes.MODES["exact"] and "triattn_exact" in modes.MODES["fast"]           # exact ⊂ fast
    fast = modes.MODES["fast"]
    assert fast.index("flash_triattn") < fast.index("triattn_exact") < fast.index("triatt_block")      # above the flash lever, beneath the fused block
    row = registry.LEVERS["triattn_exact"]
    assert row["cls"] == "exact" and "opt_core.kernels.triattn" in row["provider"] and {"stock_row:", "dtype_", "arch:", "rank", "no_calls"} <= set(row["expected"])
    from atlasfold_opt.hooks import installers
    assert installers()["triattn_exact"] is X.install


def test_words_and_cells(monkeypatch):
    T = pytest.importorskip("opt_core.kernels.triattn")
    monkeypatch.delenv(X.WORD_ENV, raising=False); assert X.word() == "exact"
    for name in ("MIN_TOKENS_ENV", "MIN_TOKENS", "IDENTITY_FLOOR", "min_tokens"):
        assert not hasattr(X, name), name                                                                                # no kit size floor: the provider's cells / row envelopes decide
    xs = "9.0|torch2.7.1+cu128|cueq0.10.0"
    s = T.select("9.0", "bf16", 32, 4, 32, word="exact", exact_stack=xs); assert s.cls == "stock"                       # <= 100 tokens: the library's torch path -> the stock op by name (opt_core >= 0.5.115.1)
    s = T.select("9.0", "bf16", 32, 4, 896, word="exact", exact_stack=xs); assert s.cls == "exact" and s.exact_vs.split(" ")[0] == "cueq"   # above it, on this (vouched) stack: the bit-identical row -- exact class, bitwise the library op
    s = T.select("9.0", "bf16", 32, 4, 896, word="exact", exact_stack="9.0|torch0.0.0|cueq0.0.0"); assert s.cls == "stock"   # an unvouched stack: the stock op by name
    s = T.select("9.0", "bf16", 32, 4, 896, word="exact"); assert s.cls in ("exact", "stock") and (s.cls == "stock" or s.exact_vs == "cueq")   # an exact word never selects a fast-class row
    s = T.select("8.0", "bf16", 32, 4, 896, word="exact"); assert s.cls in ("exact", "stock")
    s = T.select("9.0", "fp32", 32, 4, 896, word="exact"); assert s.cls in ("exact", "stock")                          # fp32 (confidence stacks): never a fast-class row


def test_unwrap_follows_the_rebind_chain():
    def stock(*a): return "stock"
    def w1(*a): return "w1"
    def w2(*a): return "w2"
    w1.__wrapped_stock__ = stock; w2.__wrapped_stock__ = w1
    assert X.unwrap(w2) is stock and X.unwrap(stock) is stock




def _fake_site(monkeypatch, block_installed=True):
    """A stand-in for atlasfold's triangle_update module when the stock package is not importable here: `cueq_tri_attn` + the two node classes
    (their forward carrying triatt_block's qualname marker when `block_installed`)."""
    import sys, types
    name = "atlasfold.model.network.primitives.triangle_update"
    try:
        import importlib; return importlib.import_module(name)
    except Exception:
        pass
    tu = types.ModuleType(name)
    def cueq_tri_attn(q, k, v, bias, mask, scale):
        return q.clone()
    tu.cueq_tri_attn = cueq_tri_attn
    for cls_name in ("TriangleAttentionStartingNode", "TriangleAttentionEndingNode"):
        def forward(self, z, mask=None, kernel_backend="cuequiv"):
            return z
        forward.__qualname__ = f"{cls_name}.forward[atlasfold_opt:triatt_block]" if block_installed else f"{cls_name}.forward"
        setattr(tu, cls_name, type(cls_name, (), {"forward": forward}))
    parts = name.split(".")
    for i in range(1, len(parts)):
        pkg = ".".join(parts[:i])
        if pkg not in sys.modules:
            m = types.ModuleType(pkg); m.__path__ = []; monkeypatch.setitem(sys.modules, pkg, m)
    monkeypatch.setitem(sys.modules, name, tu)
    return tu


def _site_or_fake(monkeypatch):
    torch = pytest.importorskip("torch"); pytest.importorskip("opt_core.kernels.triattn")
    return torch, _fake_site(monkeypatch, block_installed=False)


def _site():
    torch = pytest.importorskip("torch"); pytest.importorskip("opt_core.kernels.triattn")
    try:
        import atlasfold.model.network.primitives.triangle_update as tu
    except Exception:
        pytest.skip("stock atlasfold not importable here")
    return torch, tu


def test_install_serves_by_name_on_cpu_and_restores(monkeypatch):
    torch, tu = _site()
    monkeypatch.delenv(X.WORD_ENV, raising=False)
    calls = []
    def fake_stock(q, k, v, bias, mask, scale):
        calls.append(tuple(q.shape)); return q.clone()
    orig = tu.cueq_tri_attn
    monkeypatch.setattr(tu, "cueq_tri_attn", fake_stock)
    try:
        ins = X.install("exact", "[t]", {"mode": "exact", "det": 1})
        assert ins.applied, ins.reason
        assert tu.cueq_tri_attn is not fake_stock and X.unwrap(tu.cueq_tri_attn) is fake_stock
        N = 256
        q = torch.zeros(1, N, 4, N, 32, dtype=torch.bfloat16); bias = torch.zeros(1, 1, 4, N, N); mask = torch.ones(1, N, 1, 1, N, dtype=torch.bool)
        out = tu.cueq_tri_attn(q, q, q, bias, mask, 32 ** -0.5)                       # CPU tensor -> cc (0,0): the cell's exact row is the stock op -> stock_row:cueq, the bound call serves
        assert torch.equal(out, q) and calls == [tuple(q.shape)]
        out = tu.cueq_tri_attn(q[0], q[0], q[0], bias[0], None, 32 ** -0.5)            # no mask / rank: the bound call, named
        small = torch.zeros(1, 32, 4, 32, 32, dtype=torch.bfloat16)
        tu.cueq_tri_attn(small, small, small, torch.zeros(1, 1, 4, 32, 32), torch.ones(1, 32, 1, 1, 32, dtype=torch.bool), 32 ** -0.5)   # a 32-token pair stack: the cell's exact row on a CPU tensor is the stock op too, by name
        L = ins.facts["ledger"]
        assert L.served == 0 and set(L.fallbacks) == {"stock_row:cueq", "rank"} and ins.gates[0]().ok      # all-fallback for listed reasons = a legitimate stock run
        assert "min_tokens" not in ins.facts and ins.facts["beneath"] is None
        line = ins.lines[0]()
        assert "name=LOCAL.atlasfold.triattn_exact" in line and "word=exact" in line and "impl=opt_core.kernels.triattn@" in line and " min_tokens=" not in line and "exact_stack=" in line
    finally:
        tu.cueq_tri_attn = orig


def test_unknown_or_fast_word_installs_skipped_by_name(monkeypatch):
    torch, tu = _site()
    monkeypatch.setenv(X.WORD_ENV, "nonsense")
    ins = X.install("exact", "[t]", {}); assert not ins.applied and ins.reason == "unknown_word:nonsense"
    monkeypatch.setenv(X.WORD_ENV, "k2b")
    ins = X.install("exact", "[t]", {}); assert not ins.applied and ins.reason == "not_an_exact_word:k2b"
    monkeypatch.setenv(X.WORD_ENV, "fast")
    ins = X.install("exact", "[t]", {}); assert not ins.applied and ins.reason == "not_an_exact_word:fast"


def test_in_a_fast_mode_the_lever_yields_by_name_to_the_fast_tier_binding_beneath(monkeypatch):
    """exact ⊂ fast: under --mode fast / big the same call is bound beneath this lever by the fast tier word (hooks/triatt.py), the tier that
    subsumes this one — every call goes to that binding, counted `tier_beneath:<word>`; with the flash lever absent it serves as in the exact row."""
    torch, tu = _site_or_fake(monkeypatch)
    from atlasfold_opt.hooks import triatt as FL
    monkeypatch.delenv(X.WORD_ENV, raising=False)
    calls = []
    def fake_stock(q, k, v, bias, mask, scale):
        calls.append(tuple(q.shape)); return q.clone()
    orig = tu.cueq_tri_attn
    monkeypatch.setattr(tu, "cueq_tri_attn", fake_stock)
    try:
        fl = FL.install("big", "[t]", {"mode": "big"}); assert fl.applied, fl.reason
        assert getattr(tu.cueq_tri_attn, "__afo_tier_word__", None) == "big" and fl.facts["word"] == "big"
        ins = X.install("big", "[t]", {"mode": "big"}); assert ins.applied, ins.reason
        assert ins.facts["beneath"] == "big"
        N = 256
        q = torch.zeros(1, N, 4, N, 32, dtype=torch.bfloat16); bias = torch.zeros(1, 1, 4, N, N); mask = torch.ones(1, N, 1, 1, N, dtype=torch.bool)
        out = tu.cueq_tri_attn(q, q, q, bias, mask, 32 ** -0.5)
        assert torch.equal(out, q) and calls == [tuple(q.shape)]                          # CPU: the fast binding beneath names its own step-aside and the stock op serves
        assert set(ins.facts["ledger"].fallbacks) == {"tier_beneath:big"} and ins.gates[0]().ok and fl.gates[0]().ok
        assert "word=big" in fl.lines[0]() and "name=F1.flash_triattn" in fl.lines[0]() and "impl=opt_core.kernels.triattn@" in fl.lines[0]()
        tu.cueq_tri_attn = fake_stock
        ins = X.install("fast", "[t]", {"mode": "fast"}); assert ins.applied and ins.facts["beneath"] is None   # the flash lever ablated: nothing beneath but the stock op -> serves as in the exact row
    finally:
        tu.cueq_tri_attn = orig


def test_flash_lever_word_per_mode_and_registry():
    from atlasfold_opt.hooks import triatt as FL
    assert FL.word("fast") == "fast" and FL.word("big") == "big" and FL.NAME == "F1.flash_triattn"
    assert not hasattr(FL, "MIN_TOKENS")                                                   # no kit size floor: the provider's cells decide from the first token
    row = registry.LEVERS["flash_triattn"]
    assert row["cls"] == "fast" and "opt_core.kernels.triattn" in row["what"] and {"stock_row:", "rank", "no_calls", "no_cell:"} <= set(row["expected"])
    assert "flash_triattn" in modes.MODES["fast"] and "flash_triattn" in modes.MODES["big"] and "flash_triattn" not in modes.MODES["exact"]
