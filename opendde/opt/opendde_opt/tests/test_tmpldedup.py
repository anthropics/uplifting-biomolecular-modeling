"""tmpl_dedup: identical template slots embedded once, the stock accumulation kept verbatim — bitwise equal to the stock forward on a stand-in
TemplateEmbedder (4 identical dummy slots / 2+2 / all distinct), the class detection, the stand-aside surface, the patch mechanics."""
import sys
import types

import pytest

from opendde_opt import tmpldedup


def test_slot_classes_pure():
    feats = {k: ["a", "a", "b", "a"] for k in tmpldedup.SLOT_KEYS}
    assert tmpldedup.slot_classes(feats, 4, lambda x, y: x == y) == [0, 0, 2, 0]
    feats = {k: ["a", "b", "c", "d"] for k in tmpldedup.SLOT_KEYS}
    assert tmpldedup.slot_classes(feats, 4, lambda x, y: x == y) == [0, 1, 2, 3]
    feats = {k: ["a", "a", "a", "a"] for k in tmpldedup.SLOT_KEYS}
    feats["template_unit_vector"] = ["a", "a", "z", "a"]          # one key differs -> its own class
    assert tmpldedup.slot_classes(feats, 4, lambda x, y: x == y) == [0, 0, 2, 0]


def _embedder(torch):
    class Stub(torch.nn.Module):                                   # the read surface of TemplateEmbedder.forward, tiny
        def __init__(self, n=6, c=8):
            super().__init__()
            torch.manual_seed(0)
            self.n_blocks = 2
            self.layernorm_z = torch.nn.LayerNorm(c)
            self.linear_no_bias_u = torch.nn.Linear(c, c, bias=False)
            self.relu = torch.nn.ReLU()
            self.w = torch.nn.Linear(3, c, bias=False)
            self.calls = 0

        def single_template_forward(self, template_id, input_feature_dict, z, pair_mask=None, multichain_mask=None,
                                    triangle_attention="torch", triangle_multiplicative="torch", inplace_safe=False, chunk_size=None):
            self.calls += 1
            d = input_feature_dict["template_distogram"][template_id]                        # [N, N, 3]
            a = input_feature_dict["template_aatype"][template_id].to(z.dtype)                 # [N]
            v = torch.tanh(self.w(d) * 0.37 + z * 0.11) * multichain_mask[..., None] * pair_mask[..., None] + a[:, None, None] * 1e-3
            return torch.sin(v * 3.3) / 3.0

        def forward(self, input_feature_dict, z, pair_mask=None, triangle_attention="torch", triangle_multiplicative="torch",
                    inplace_safe=False, chunk_size=None):                                       # the stock body (pairformer.py:2160-2193)
            if "template_aatype" not in input_feature_dict or self.n_blocks < 1:
                return 0
            asym_id = input_feature_dict["asym_id"]
            multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(z.dtype)
            num_residues = z.shape[0]
            num_templates = input_feature_dict["template_aatype"].shape[0]
            query_num_channels = z.shape[-1]
            if pair_mask is None:
                pair_mask = z.new_ones(z.shape[:-1])
            z = self.layernorm_z(z)
            u = 0
            for template_id in range(num_templates):
                u = u + self.single_template_forward(template_id=template_id, input_feature_dict=input_feature_dict, z=z, pair_mask=pair_mask,
                                                      multichain_mask=multichain_mask)
            u = u / (1e-7 + num_templates)
            u = self.linear_no_bias_u(self.relu(u))
            assert u.shape == (num_residues, num_residues, query_num_channels)
            return u
    return Stub


def _feats(torch, n, pattern):
    """pattern: list of slot 'kinds' (equal kinds -> bit-identical slot tensors)."""
    g = torch.Generator().manual_seed(1)
    base = {}
    for kind in sorted(set(pattern)):
        base[kind] = {"template_aatype": torch.randint(0, 31, (n,), generator=g), "template_distogram": torch.rand(n, n, 3, generator=g),
                      "template_pseudo_beta_mask": torch.rand(n, n, generator=g), "template_unit_vector": torch.rand(n, n, 3, generator=g),
                      "template_backbone_frame_mask": torch.rand(n, n, generator=g)}
    feats = {k: torch.stack([base[kind][k].clone() for kind in pattern]) for k in tmpldedup.SLOT_KEYS}
    feats["asym_id"] = torch.tensor([0] * (n // 2) + [1] * (n - n // 2))
    return feats


@pytest.mark.parametrize("pattern,distinct", [("aaaa", 1), ("abab", 2), ("aabc", 3), ("abcd", 4), ("a", 1)])
def test_bitwise_equal_to_stock_and_counts(pattern, distinct):
    torch = pytest.importorskip("torch")
    Stub = _embedder(torch)
    n = 6
    feats = _feats(torch, n, list(pattern))
    z = torch.rand(n, n, 8, generator=torch.Generator().manual_seed(2))
    m = Stub()
    with torch.no_grad():
        ref = Stub.forward(m, feats, z.clone())
        calls_stock = m.calls
        m.calls = 0
        for k in ("calls", "slots", "distinct", "saved", "bypass", "decisions", "memo_hits"):
            tmpldedup.STATS[k] = 0
        tmpldedup._MEMO.update(refs=None, versions=None, rep=None)
        wrapped = tmpldedup.make_wrapper(Stub.forward)
        out = wrapped(m, feats, z.clone())
    assert torch.equal(out, ref)                                   # bit for bit
    assert calls_stock == len(pattern) and m.calls == distinct     # the pair stack ran once per class
    assert (tmpldedup.STATS["calls"], tmpldedup.STATS["slots"], tmpldedup.STATS["distinct"], tmpldedup.STATS["saved"]) == (1, len(pattern), distinct, len(pattern) - distinct)
    assert tmpldedup.STATS["bypass"] == 0
    with torch.no_grad():                                          # the recycles of one item: the same tensors -> the memo serves, no device decision
        out2 = wrapped(m, feats, z.clone()); out3 = wrapped(m, feats, z.clone())
    assert torch.equal(out2, ref) and torch.equal(out3, ref)
    assert (tmpldedup.STATS["decisions"], tmpldedup.STATS["memo_hits"]) == (1, 2)
    feats2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats.items()}   # a new item (new tensors, same values) -> a fresh decision
    with torch.no_grad():
        assert torch.equal(wrapped(m, feats2, z.clone()), ref)
    assert tmpldedup.STATS["decisions"] == 2
    feats2["template_distogram"][-1].add_(1.0)                     # an in-place edit bumps _version -> the memo cannot serve a stale class
    with torch.no_grad():
        wrapped(m, feats2, z.clone())
    assert tmpldedup.STATS["decisions"] == 3 and m.calls >= 1


def test_memo_never_serves_a_freed_tensor():
    torch = pytest.importorskip("torch")
    import gc
    feats = _feats(torch, 5, list("aabb"))
    tmpldedup._MEMO.update(refs=None, versions=None, rep=None)
    rep = tmpldedup.classes_for(feats, 4, torch)
    assert rep == [0, 0, 2, 2] and tmpldedup._memo_lookup(feats) == rep
    feats3 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats.items()}
    del feats; gc.collect()                                        # the old tensors die: their weak references clear, whatever ids new tensors take
    assert tmpldedup._memo_lookup(feats3) is None


def test_stands_aside_by_name():
    torch = pytest.importorskip("torch")
    Stub = _embedder(torch)
    m = Stub()
    wrapped = tmpldedup.make_wrapper(Stub.forward)
    tmpldedup.STATS["bypass"] = 0; tmpldedup.STATS["bypass_reasons"].clear()
    feats = _feats(torch, 6, list("aaaa"))
    z = torch.rand(6, 6, 8)
    with torch.no_grad():
        assert wrapped(m, {k: v for k, v in feats.items() if k != "template_aatype"}, z) == 0          # stock's own early return, delegated
        m.n_blocks = 0
        assert wrapped(m, feats, z) == 0
        m.n_blocks = 2
        bad = dict(feats); bad["template_distogram"] = feats["template_distogram"][:2]                # leading dim != slots -> delegate (the stock indexes it per slot: IndexError is the stock's)
        with pytest.raises(IndexError):
            wrapped(m, bad, z)
    with torch.enable_grad():
        out = wrapped(m, feats, z)                                                                  # autograd on -> the stock method
    assert tmpldedup.STATS["bypass_reasons"] == {"no_templates_or_zero_blocks": 2, "slot_key_unreadable:template_distogram": 1, "grad_enabled": 1}
    assert torch.is_tensor(out)


def test_patch_installs_on_a_stand_in_module(monkeypatch):
    pytest.importorskip("opt_core")
    torch = pytest.importorskip("torch")
    mod = types.ModuleType(tmpldedup.TARGET)
    Stub = _embedder(torch)
    Stub.__name__ = "TemplateEmbedder"
    mod.TemplateEmbedder = Stub
    tmpldedup._reset()                                                                              # a patch an earlier activation armed in this process is withdrawn (the core keeps one per site)
    monkeypatch.setitem(sys.modules, tmpldedup.TARGET, mod)
    tmpldedup.install()
    try:
        assert tmpldedup.STATS["installed"] and getattr(mod.TemplateEmbedder.forward, "_tmpl_dedup", False)
        assert tmpldedup.fallbacks(["tmpl_dedup"]) == []
        mod.TemplateEmbedder.forward = lambda self, *a, **k: 0                                      # another unit re-binds the site -> named
        assert tmpldedup.fallbacks(["tmpl_dedup"]) and "re-bound" in tmpldedup.fallbacks(["tmpl_dedup"])[0]
        assert tmpldedup.fallbacks(["chunk_lift"]) == []
    finally:
        tmpldedup._reset()


def test_registry_row_and_lines():
    from opendde_opt import modes, registry
    if "tmpl_dedup" not in registry.LEVERS:
        pytest.skip("tmpl_dedup not wired in this tree")
    lv = registry.LEVERS["tmpl_dedup"]
    assert (lv.kit, lv.tier, lv.file) == (registry.HOUSE, "exact", "opendde_opt/tmpldedup.py")
    assert registry.PIN_STATUS["tmpl_dedup"][0] == "tested"
    for ln in ("S1", "LSTAR2A"):
        assert "tmpl_dedup" in modes.LINES[ln].levers
    assert "tmpl_dedup" in modes.BIG_KIT_ROWS_OFF["pair_offload"] and "tmpl_dedup" in modes.BIG_TP_DROP   # off by name with the offload unit and on the row-sharded line (TemplateEmbedder.forward is not entered there)
    assert registry.validate() == []
