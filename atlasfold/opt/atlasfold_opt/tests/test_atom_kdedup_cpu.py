"""atom_kdedup on CPU (no GPU, no weights): rows / registry / installer membership and order, the expected words == the module's literals,
the unfold-geometry reader (a real LocalAttentionIndex.to_k view is recognised and its rows are the padded copy; a contiguous tensor of the
same shape is not), CrossAttention.forward with the lever == upstream's on the same weights (conditioned atom-window operands as
AtomTransformerStack builds them, B in {1, 2}), a whole DiffusionHead.sample roll-out with the lever == the stock roll-out (alone, under
sampler_hoist, under sampler_hoist + atom_sdpa's CPU route), the k / v hand-over reaches Attention.forward (hits counted, no kv_unread),
the named routes (AFO_ATOM_KDEDUP=0, a contiguous a_k), install refuses by name over an unknown Attention.forward wrapper."""
import importlib

import pytest
import torch

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import installers
from atlasfold_opt.hooks import atom_kdedup as KD

from .test_sampler_hoist_cpu import _head, _batch, _sample, _sites, _innermost   # noqa: F401  (the small stock head + batch of the sampler_hoist suite)
from .test_sampler_hoist_cpu import stock_sites, lever as hoist_lever            # noqa: F401  (fixtures: hermetic patch sites; sampler_hoist installer)


def _restore_all():
    att = importlib.import_module(KD.TARGET)
    for cls, attr in ((att.CrossAttention, "forward"), (att.Attention, "forward")):
        setattr(cls, attr, _innermost(getattr(cls, attr)))


@pytest.fixture
def kdedup(monkeypatch, stock_sites):
    installed = []

    def _install(switch=None):
        if switch is None:
            monkeypatch.delenv(KD.ENV, raising=False)
        else:
            monkeypatch.setenv(KD.ENV, switch)
        KD._STATE["override"] = None
        ins = KD.install("exact", "atlasfold-opt", {})
        installed.append(ins)
        return ins
    yield _install
    for ins in installed:
        if ins.applied:
            ins.facts["restore"]()
    _restore_all()
    KD._STATE["override"] = None


def test_rows_registry_installer_order():
    ex, fa = modes.MODES[modes.EXACT], modes.MODES[modes.FAST]
    assert "atom_kdedup" in ex and "atom_kdedup" in fa and "atom_kdedup" in modes.MODES[modes.BIG]      # no memory cost: fast and big alike
    assert ex.index("sampler_hoist") < ex.index("atom_kdedup") < ex.index("lever_report")
    assert fa.index("atom_sdpa") < fa.index("atom_kdedup") < fa.index("denoiser_graph")          # installed after the Attention.forward wrappers it names
    assert fa.index("sampler_hoist") < fa.index("atom_kdedup")
    row = registry.LEVERS["atom_kdedup"]
    assert row["cls"] == "exact" and row["module"] == "atlasfold_opt.hooks.atom_kdedup" and "CrossAttention.forward" in row["site"]
    assert tuple(row["expected"]) == KD.EXPECTED
    assert "atom_kdedup" in installers() and installers()["atom_kdedup"] is KD.install


def test_expected_words_are_the_module_literals():
    src = open(KD.__file__).read()
    for word in KD.EXPECTED:
        assert f'ledger.fallback("{word}")' in src, word
    for word in ("rank", "kv_layout", "cond_layout"):                             # named, counted, NOT expected (none occurs in this model: the gate refuses them)
        assert f'ledger.fallback("{word}")' in src and word not in KD.EXPECTED


def test_unfold_rows_reads_the_padded_copy_and_refuses_a_contiguous_tensor():
    from atlasfold.model.network.misc import LocalAttentionIndex
    import einops
    B, N, L, c = 2, 3, 16, 8
    a = torch.randn(B, N, L, 14, c)
    lai = LocalAttentionIndex(torch.arange(L).repeat(B, 1).unsqueeze(1), torch.zeros(B, 1, L, dtype=torch.long), torch.ones(B, 1, L, dtype=torch.bool))
    a_q, a_k = lai(a, dim=-3)
    a_q, a_k = (einops.rearrange(x, "... w l a c -> ... w (l a) c", a=14) for x in (a_q, a_k))
    g = KD.unfold_rows(a_k)
    assert g is not None and (g.B, g.N, g.W, g.Lk, g.c, g.step, g.Rn) == (B, N, L // 4, 168, c, 56, (L + 8) * 14)
    padded = torch.nn.functional.pad(a, (0, 0, 0, 0, 4, 4)).reshape(B, N, (L + 8) * 14, c)
    assert torch.equal(g.rows, padded)                                            # the rows ARE the zero-padded atom rows
    for w in range(g.W):                                                          # and a_k is their unfold, window by window
        assert torch.equal(a_k[:, :, w], g.rows[:, :, w * 56: w * 56 + 168])
    k_rows = torch.randn(B, N, g.Rn, 5)
    k_w = KD.fold_windows(k_rows, g)
    assert k_w.shape == (B, N, g.W, 168, 5) and all(torch.equal(k_w[:, :, w], k_rows[:, :, w * 56: w * 56 + 168]) for w in range(g.W))
    assert KD.unfold_rows(a_k.contiguous()) is not None and KD.unfold_rows(a_k.contiguous()).step == 168   # a plain reshape: step == Lk (no overlap, still rows)
    assert KD.unfold_rows(a_q) is not None and KD.unfold_rows(a_q).step == 56                              # the query view: step == Lq (a tiling)
    bad = a_k.transpose(-1, -2)
    assert KD.unfold_rows(bad) is None and KD.unfold_rows(a_k[..., :5]) is None            # transposed / channel-sliced views: rows not dense -> not this form
    assert KD.unfold_rows(torch.randn(B, N, 4, 168)) is None                        # rank 4


def _atom_operands(B=2, N=3, L=16, c=16, seed=5):
    """(block, a_q, a_k, attn_mask, cq, ck, pair_bias_i) exactly as AtomTransformerStack.forward builds them for its first block."""
    from atlasfold.model.network.diffusion_transformer import AtomTransformerBlock
    from atlasfold.model.network.misc import LocalAttentionIndex
    import einops
    torch.manual_seed(seed)
    blk = AtomTransformerBlock(c, c, num_heads=2).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    a = torch.randn(B, N, L, 14, c, generator=g)
    cond = torch.randn(B, 1, L, 14, c, generator=g)
    seq_mask = torch.ones(B, 1, L, dtype=torch.bool); seq_mask[-1, :, L - 2:] = False
    atom_mask = torch.rand(B, 1, L, 14, generator=g) > 0.25
    asym = torch.zeros(B, 1, L, dtype=torch.long); asym[..., L // 2:] = 1
    lai = LocalAttentionIndex(torch.arange(L).repeat(B, 1).unsqueeze(1), asym, seq_mask)
    attn_mask = einops.repeat(lai.attn_mask, "... q k -> ... (q 14) (k 14)")
    mask_k = lai.to_k(atom_mask, dim=-2)
    attn_mask = attn_mask & einops.rearrange(mask_k, "... w k a -> ... w 1 (k a)", a=14)
    cq, ck = (einops.rearrange(x, "... w l a c -> ... w (l a) c", a=14) for x in lai(cond, dim=-3))
    pb = torch.randn(B, 1, lai.W, 2, 56, 168, generator=g)
    a_q, a_k = (einops.rearrange(x, "... w l a c -> ... w (l a) c", a=14) for x in lai(a, dim=-3))
    return blk, a_q, a_k, attn_mask, cq, ck, pb


@pytest.mark.parametrize("B", [1, 2])
def test_cross_attention_with_the_lever_equals_upstream(B, kdedup):
    att = importlib.import_module(KD.TARGET)
    blk, a_q, a_k, m, cq, ck, pb = _atom_operands(B=B)
    upstream = _innermost(att.CrossAttention.forward)
    with torch.no_grad():
        ref = upstream(blk.attention, a_q, a_k, m, pb, cq, ck)
    ins = kdedup()
    assert ins.applied, ins.reason
    with torch.no_grad():
        out = blk.attention(a_q, a_k, m, pb, cq, ck)
    assert out.shape == ref.shape == a_q.shape
    if not torch.equal(out, ref):                                                 # a CPU BLAS may block the reduction by the row count; the GPU byte-identity vs --mode off is the class evidence (CHANGES)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), float((out - ref).abs().max())
    led = ins.facts["ledger"]
    assert led.served == 1 and not led.fallbacks and led.get("kv_unread") == 0 and led.get("cond_per_call") == 1, led.line("t")
    line = ins.lines[0]()
    assert "name=LOCAL.atlasfold.atom_kdedup state=on" in line and f"W{a_q.shape[2]}xN{a_q.shape[1]}x56x168r" in line, line
    assert not hasattr(blk.attention.attn, KD.STASH)                              # the hand-over lives for one call only
    assert KD.KV_TAG in blk.attention.attn.linear_k.__dict__["forward"].__qualname__
    with torch.no_grad():                                                         # outside a served call the instances are the class forward
        x = torch.randn(3, 16)
        assert torch.equal(blk.attention.attn.linear_k(x), torch.nn.functional.linear(x, blk.attention.attn.linear_k.weight))
    assert KD.remove_instances(blk) == 2


def _coords(out):
    return out[0] if isinstance(out, (tuple, list)) else out


@pytest.mark.parametrize("with_hoist", [False, True])
def test_rollout_with_the_lever_equals_the_stock_rollout(with_hoist, kdedup, hoist_lever):
    head = _head(); batch, s, z = _batch()
    ref = _coords(_sample(head, batch, s, z))
    if with_hoist:
        hins = hoist_lever()
        assert hins.applied, hins.reason
    ins = kdedup()
    assert ins.applied, ins.reason
    out = _coords(_sample(head, batch, s, z))
    assert torch.isfinite(out).all()
    if not torch.equal(out, ref):
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), float((out - ref).abs().max())
    led = ins.facts["ledger"]
    assert led.served > 0 and not led.fallbacks and led.get("kv_unread") == 0, led.line("t")
    if with_hoist:                                                                # inside a sampler_hoist roll-out the conditioning leaves are memoised per roll-out (freed at close)
        assert led.get("cond_memo_miss") > 0 and led.get("cond_memo_hit") > 0 and led.get("cond_per_call") == 0, led.line("t")
        from atlasfold_opt.hooks import sampler_hoist as SH
        assert SH._cur() is None
    else:
        assert led.get("cond_per_call") > 0 and led.get("cond_memo_hit") == 0
    KD.remove_instances(head)


def test_rollout_under_sampler_hoist_and_atom_sdpa_cpu_route(kdedup, hoist_lever, monkeypatch):
    """fast-row order on CPU: sampler_hoist -> atom_sdpa -> atom_kdedup; atom_sdpa's CPU calls take torch's CPU fused route or its
    statement below — either way it reads k / v through linear_k(a_k) / linear_v(a_k): the hand-over is hit, outputs match the same stack without the lever."""
    from atlasfold_opt.hooks import atom_sdpa
    att = importlib.import_module(KD.TARGET)
    monkeypatch.delenv("AFO_ATOM_SDPA", raising=False); atom_sdpa._STATE["override"] = None
    head = _head(); batch, s, z = _batch()
    hoist_lever()
    atom_sdpa.install("fast", "atlasfold-opt", {})
    try:
        ref = _coords(_sample(head, batch, s, z))
        ins = kdedup()
        assert ins.applied, ins.reason
        out = _coords(_sample(head, batch, s, z))
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), float((out - ref).abs().max())
        led = ins.facts["ledger"]
        assert led.served > 0 and not led.fallbacks and led.get("kv_unread") == 0, led.line("t")
    finally:
        KD.remove_instances(head)
        setattr(att.Attention, "forward", _innermost(att.Attention.forward))
        atom_sdpa._STATE["override"] = None


def test_disabled_switch_and_contiguous_keys_take_the_statement_below(kdedup):
    att = importlib.import_module(KD.TARGET)
    blk, a_q, a_k, m, cq, ck, pb = _atom_operands(B=1)
    upstream = _innermost(att.CrossAttention.forward)
    with torch.no_grad():
        ref = upstream(blk.attention, a_q, a_k, m, pb, cq, ck)
    ins = kdedup(switch="0")
    with torch.no_grad():
        out = blk.attention(a_q, a_k, m, pb, cq, ck)
    assert torch.equal(out, ref)
    led = ins.facts["ledger"]
    assert led.served == 0 and led.fallbacks == {"disabled": 1}
    assert "state=skipped reason=all_fallback:disabled" in ins.lines[0]() and f"switch={KD.ENV}=0" in ins.lines[0]()
    KD._STATE["override"] = True                                                  # lever on; a contiguous a_k whose cond is NOT the same unfold -> cond_layout, stock, exact
    with torch.no_grad():
        out2 = blk.attention(a_q, a_k.contiguous(), m, pb, cq, ck)
    assert torch.equal(out2, ref) and led.fallbacks.get("cond_layout") == 1
    with torch.no_grad():                                                         # rank-4 operands (a DiT-shaped call) -> rank
        blk4, *_ = _atom_operands(B=1)
    KD.remove_instances(blk)


def test_install_refuses_by_name_over_an_unknown_attention_wrapper(kdedup):
    att = importlib.import_module(KD.TARGET)
    from atlasfold_opt.hooks import rebind
    stock = att.Attention.forward

    def foreign(self, *a, **k):
        return stock(self, *a, **k)
    foreign.__qualname__ = "Attention.forward[someone_else]"
    rebind(att.Attention, "forward", foreign, stock)
    try:
        ins = kdedup()
        assert not ins.applied and ins.reason == "attention_wrapper:Attention.forward[someone_else]"
    finally:
        setattr(att.Attention, "forward", stock)
    ins2 = kdedup()
    assert ins2.applied and set(ins2.facts["digests"]) == set(KD.SOURCE_SHA256)


def test_exact_row_floor_per_card_is_a_kit_table():
    from atlasfold_opt.hooks import atom_kdedup as H
    assert H.EXACT_MIN_ROWS == {(9, 0): 0, (8, 0): 9072}                       # a kit table fact (measured): asserted by name
    assert H.exact_min_rows("exact", (8, 0)) == 9072 and H.exact_min_rows("exact", (9, 0)) == 0
    assert H.exact_min_rows("exact", (12, 0)) == H.EXACT_MIN_ROWS_DEFAULT == 9072    # an unlisted card: the conservative floor
    assert H.exact_min_rows("exact", None) == 0                                # no CUDA device: nothing to key on
    assert H.exact_min_rows("fast", (8, 0)) == 0 and H.exact_min_rows("big", (8, 0)) == 0   # tolerance-class rows: no floor
    assert "below_exact_min_rows" in H.EXPECTED


def test_below_the_exact_floor_the_stock_statement_runs_by_name(monkeypatch):
    import torch
    from opt_core.counters import Ledger
    from atlasfold_opt.hooks import atom_kdedup as H
    calls = []
    def stock(self, a_q, a_k, mask, pair_bias=None, single_cond_q=None, single_cond_k=None):
        calls.append("stock"); return a_q
    ledger = Ledger(H.NAME, impl=H.IMPL, origin="kit", expected=H.EXPECTED)
    fwd = H.make_forward(stock, ledger)
    monkeypatch.setitem(H._STATE, "exact_min_rows", 9072)
    monkeypatch.setattr(H, "enabled", lambda: True)
    B, N, Wn, Lq, Lk, c, step = 1, 1, 4, 32, 128, 8, 32
    rows = torch.randn(B, N, (Wn - 1) * step + Lk, c)
    a_k = rows.as_strided((B, N, Wn, Lk, c), (rows.stride(0), rows.stride(1), step * c, c, 1))
    a_q = torch.randn(B, N, Wn, Lq, c)
    out = fwd(object(), a_q, a_k, torch.ones(B, N, Wn, Lk, dtype=torch.bool))
    assert calls == ["stock"] and out is a_q
    assert ledger.served == 0 and dict(ledger.fallbacks) == {"below_exact_min_rows": 1}
    line = ledger.line("[t]", exact_min_rows=9072)
    assert "state=skipped reason=all_fallback:below_exact_min_rows " in line and " exact_min_rows=9072" in line
