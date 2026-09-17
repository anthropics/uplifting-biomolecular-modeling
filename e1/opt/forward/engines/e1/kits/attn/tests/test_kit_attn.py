"""CPU contract test for kit ``attn``: the KIT SURFACE, routing on a tiny random E1 model (the real package on CPU), counters ==
expectations, the padded-batch fallback bitwise with the stock, the cheap BlockMask's structure == the stock document mask's,
shape-derived cu_seqlens == the stock unpad data, restore after unapply, fail-closed pins. No GPU, no weights, no network."""
from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

torch = pytest.importorskip("torch")
E1 = pytest.importorskip("E1", reason="the stock E1 package (CPU install) is the fake-model source for this test")

from engines.e1 import kits  # noqa: E402
from engines.e1.kits import attn  # noqa: E402
from engines.e1.kits.attn import adapter  # noqa: E402
from engines.e1.kits.attn.adapter import Config  # noqa: E402


def _tiny_model():
    from E1.config import E1Config
    from E1.modeling import E1ForMaskedLM
    cfg = E1Config(vocab_size=34, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
                   global_attention_every_n_layers=3, clip_qkv=8, gated_mlp=True, rms_norm_eps=1e-5)
    torch.manual_seed(0)
    m = E1ForMaskedLM(cfg)
    m.eval()
    return m


def _batch(rows):
    from E1.batch_preparer import E1BatchPreparer
    return E1BatchPreparer().get_batch_kwargs(rows, device=torch.device("cpu"))


def _fwd(m, b):
    with torch.no_grad():
        o = m(input_ids=b["input_ids"], within_seq_position_ids=b["within_seq_position_ids"], global_position_ids=b["global_position_ids"],
              sequence_ids=b["sequence_ids"], past_key_values=None, use_cache=False)
    return o.logits, o.embeddings


@pytest.fixture(autouse=True)
def _clean():
    yield
    if adapter._S["cfg"] is not None:
        adapter.unapply()
    attn._state.update(applied=False, size=None, num_warps=None, model_id=None, expected=None)


def test_kit_surface():
    kits.assert_kit_surface(attn)
    assert attn.KIT == "attn" and attn.kit_files() == {"__init__.py": True, "adapter.py": True}
    assert attn.CONFIG.within == "fa2_varlen" and attn.CONFIG.global_ == "flex" and attn.CONFIG.flex_block_mask == "cheap"
    assert attn.TESTED_SHAPES == kits.pins.PROBE_SHAPES


def test_expected_per_forward_counts_layers():
    m = _tiny_model()
    e = attn.expected_per_forward(m, 4, 32)
    assert e["_mode"] == "dense" and e["within:fa2_varlen"] == 2 and e["global:flex"] == 1 and e["mask_cheap"] == 1 and e["within:stock"] == 0
    f = attn.expected_per_forward(m, 8, "32m")
    assert f["_mode"] == "fallback" and f["within:stock"] == 2 and f["global:stock"] == 1 and f["unpad_computed"] == 1 and f["unpad_reused"] == 1 and f["forwards_dense"] == 0


def test_apply_refuses_without_gpu_and_on_bad_pins():
    m = _tiny_model()
    with pytest.raises(attn.KitRefused):
        attn.apply(m, size="150m")                              # require_gpu on a CPU box
    with pytest.raises((attn.KitRefused, kits.pins.PinDrift)):      # a CPU box fails the stack pin before the size check
        attn.apply(m, size="nope", require_gpu=False)


def test_padded_batch_falls_back_to_the_stock_path_bitwise():
    """A padded two-length batch routes to the stock code path (unpad once per forward) and is bitwise with the stock."""
    m = _tiny_model()
    b = _batch(["ACDEFGHIK?MNPQRSTVWY", "ACDEF?HIKLMNPQ"])
    stock = _fwd(m, b)
    adapter.apply(m, attn.CONFIG)
    adapter.reset_counters()
    out = _fwd(m, b)
    c = adapter.counters()
    assert c["forwards"] == 1 and c.get("forwards_dense", 0) == 0 and c["within:stock"] == 2 and c["global:stock"] == 1, c
    assert torch.equal(out[0], stock[0]) and torch.equal(out[1], stock[1])
    adapter.unapply()
    assert all(adapter.is_pristine().values())
    again = _fwd(m, b)
    assert torch.equal(again[0], stock[0])


def test_cheap_block_mask_matches_stock_structure():
    """The index-only mask yields the stock document mask's BlockMask structure for a dense batch (the property the exact
    claim rests on), for L not a multiple of the block size; cached per shape."""
    from E1.model.flex_attention import create_block_causal_mask_optimized
    for B, L in ((2, 260), (1, 516), (3, 128)):
        sid = torch.zeros(B, L, dtype=torch.long)
        stock = create_block_causal_mask_optimized(sid)
        cheap = adapter.cheap_block_mask(B, L, torch.device("cpu"))
        assert torch.equal(stock.kv_num_blocks, cheap.kv_num_blocks) and torch.equal(stock.kv_indices, cheap.kv_indices)
        assert torch.equal(stock.full_kv_num_blocks, cheap.full_kv_num_blocks) and torch.equal(stock.full_kv_indices, cheap.full_kv_indices)
        assert stock.BLOCK_SIZE == cheap.BLOCK_SIZE and stock.shape == cheap.shape
        assert adapter.cheap_block_mask(B, L, torch.device("cpu")) is cheap


def test_dense_cu_seqlens_matches_unpad_data():
    from E1.model import flash_attention_utils as U
    for B, L in ((1, 260), (16, 260), (3, 7)):
        sid = torch.zeros(B, L, dtype=torch.long)
        idx, cu, mx = U._get_unpad_data(sid)
        mine = adapter.dense_cu_seqlens(B, L, torch.device("cpu"))
        assert torch.equal(cu.to(mine.dtype), mine) and mx == L and torch.equal(idx, torch.arange(B * L))


def test_unpad_once_equals_stock_unpad():
    """The fallback's unpad (index data once per forward, no-op gathers skipped) returns what the stock _unpad_input
    returns, tensor for tensor, on a padded batch and on an unpadded multi-sequence batch."""
    from E1.model import flash_attention_utils as U
    torch.manual_seed(1)
    B, L, nh, hd = 3, 7, 2, 4
    q, k, v = (torch.randn(B, L, nh, hd) for _ in range(3))
    padded = torch.tensor([[0, 0, 0, 1, 1, -1, -1], [0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 2, 2, -1]])
    multi = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 2, 2, 2]])
    for ids, ident in ((padded, False), (multi, True)):
        adapter._S["fwd"] = adapter._Forward(False)
        adapter._S["cfg"] = Config(within="stock", global_="stock")
        adapter._S["orig"].setdefault("_unpad_input", U._unpad_input)
        adapter.reset_counters()
        try:
            mine = adapter._unpad_input_once(q, k, v, ids, ids)
            mine2 = adapter._unpad_input_once(q, k, v, ids, ids)
        finally:
            adapter._S["fwd"] = None
            adapter._S["cfg"] = None
        stock = U._unpad_input(q, k, v, ids, ids)
        for a, b in zip(mine[:4], stock[:4]):
            assert torch.equal(a, b)
        assert torch.equal(mine[4][0], stock[4][0]) and torch.equal(mine[4][1], stock[4][1]) and mine[5] == stock[5]
        assert torch.equal(mine2[0], stock[0])
        c = adapter.counters()
        assert c["unpad_computed"] == 1 and c["unpad_reused"] == 1 and (c.get("unpad_identity", 0) == 2) == ident, c


def test_dense_routing_counts_on_cpu_with_stock_kernels():
    """Dense mode on CPU: the within route needs flash_attn (CUDA-only) — so the CPU check uses the stock-kernel config for
    the within-seq layers and the flex route with the cheap mask for the global layer (eager flex on CPU), bitwise with the
    stock forward (the cheap mask is the same mask for a dense batch)."""
    pytest.importorskip("torch.nn.attention.flex_attention")
    m = _tiny_model()
    b = _batch(["ACDEFGHIK?MNPQRSTVWY", "ACDEFGHIKLMNPQ?STVWY"])
    stock = _fwd(m, b)
    adapter.apply(m, Config(within="stock", global_="flex", flex_block_mask="cheap", flex_dynamic=False))
    adapter.reset_counters()
    o1 = _fwd(m, b)
    o2 = _fwd(m, b)
    c = adapter.counters()
    assert c["forwards"] == 2 and c["forwards_dense"] == 2 and c["global:flex"] == 2 and c["mask_cheap"] == 2 and c["cheap_mask_built"] == 1, c
    assert torch.equal(o1[0], o2[0])
    assert (o1[0] - stock[0]).abs().max() < 1e-4        # CPU flex (eager) vs the stock's own CPU path: same math, not the same kernel


def test_dense_decision_replays_from_graph_layout_key(monkeypatch):
    """With the graph kit present (a fake ``engines.e1.kits.graph.current_layout``), the dense decision is taken once per
    host-side content key and replayed sync-free; a different content at the same shape is a different key -> re-decided."""
    import types
    import engines.e1.kits as K
    fake = types.ModuleType("engines.e1.kits.graph")
    state = {"layout": None}
    fake.current_layout = lambda: state["layout"]
    monkeypatch.setitem(sys.modules, "engines.e1.kits.graph", fake)
    monkeypatch.setattr(K, "graph", fake, raising=False)
    adapter._S["graph_mod"] = None
    adapter._S["dense_by_key"].clear()
    m = _tiny_model()
    dense_b = _batch(["ACDEFGHIK?MNPQRSTVWY", "ACDEFGHIKLMNPQ?STVWY"])
    padded_b = _batch(["ACDEFGHIK?MNPQRSTVWY", "ACDEF?HIKLMNPQ"])
    adapter.apply(m, Config(within="stock", global_="stock", flex_block_mask="cached"))
    adapter.reset_counters()
    state["layout"] = {"key": (2, 24, "k-dense")}
    _fwd(m, dense_b)
    _fwd(m, dense_b)
    state["layout"] = {"key": (2, 24, "k-padded")}
    _fwd(m, padded_b)
    c = adapter.counters()
    assert c["dense_check_sync"] == 2 and c["dense_from_layout_key"] == 1 and c["forwards_dense"] == 2 and c["forwards_fallback"] == 1, c
    state["layout"] = None                                         # graph absent / outside a forward: the counted sync check
    _fwd(m, dense_b)
    assert adapter.counters()["dense_check_sync"] == 3
    adapter.unapply()
    adapter._S["graph_mod"] = None
