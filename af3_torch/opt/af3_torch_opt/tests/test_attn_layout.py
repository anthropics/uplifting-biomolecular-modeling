"""Lever attn_layout: _attention restates GridSelfAttention._attention but for the regroupings; regroup is a pure copy."""
import inspect, os
import pytest
from af3_torch_opt import attn_layout, stack

ATT_FILE = os.path.join("af3t", "af3_torch", "xfold", "nn", "attention.py")
WRAP_FILE = os.path.join("af3t", "af3_torch", "xfold", "fastnn", "attention.py")


def test_attention_is_the_stock_statements_but_the_regroupings():
    src = open(os.path.join(stack.forward_dir(), ATT_FILE), encoding="utf-8").read()
    ours = inspect.getsource(attn_layout._attention)
    for stmt in ("q = self.q_projection(pair)", "k = self.k_projection(pair)", "v = self.v_projection(pair)",
                 "q, k, v = map(lambda t: einops.rearrange(", "t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])",
                 "weighted_avg = fastnn.dot_product_attention(q, k, v,", "weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')",
                 "gate_values = self.gating_query(pair)", "weighted_avg *= torch.sigmoid(gate_values)", "return self.output_projection(weighted_avg)"):
        assert stmt in src and stmt in ours, stmt
    wrap = open(os.path.join(stack.forward_dir(), WRAP_FILE), encoding="utf-8").read()
    for stmt in ("q = q.contiguous()", "k = k.contiguous()", "v = v.contiguous()", "o = torch.empty_like(q)", "assert Lk in {16, 32, 64, 128}"):
        assert stmt in wrap, stmt                                     # the wrapper copies strided operands and allocates o like q: what head_major / token_major stand in for
    assert "off_hz * stride_qh" in wrap and "offs_m[:, None] * stride_qm" in wrap   # the kernel's addressing that makes the copies necessary (z and h folded on stride_h)


def test_regroup_is_a_pure_copy_on_cpu_shapes():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("regroup is a Triton kernel (CUDA)")
    pytest.importorskip("xfold.fastnn", reason="the kit's model tree on sys.path (the model process)")
    attn_layout._bind_modules()
    b, n, h, d = 3, 5, 4, 16
    t = torch.arange(b * n * h * d, dtype=torch.float32).reshape(b, n, h * d)
    t = t.cuda()
    hm = attn_layout.head_major(t, h)
    assert hm.shape == (b, h, n, d) and hm.is_contiguous() and torch.equal(hm, t.view(b, n, h, d).permute(0, 2, 1, 3).contiguous())
    tm = attn_layout.token_major(hm)
    assert tm.shape == (b, n, h * d) and tm.is_contiguous() and torch.equal(tm, t)


def test_install_steps_aside_when_the_class_is_rebound():
    torch = pytest.importorskip("torch")
    import importlib, sys, types
    kit = os.path.join(stack.forward_dir(), "af3t", "af3_torch"); sys.path.insert(0, kit)
    try:
        am = importlib.import_module("xfold.nn.attention")
    finally:
        sys.path.remove(kit)
    GA = am.GridSelfAttention
    model = torch.nn.Sequential(GA(c_pair=16, num_head=4))
    keep = GA.forward
    try:
        GA.forward = lambda self, pair, mask: pair                     # a kernel lever owns the class
        r = attn_layout.install(model)
        assert r == {"installed": False, "already": False, "modules": 1, "reason": "class_rebound_by_a_kernel_lever"}
    finally:
        GA.forward = keep
        for a in ("_attn_layout", "_attn_layout_reason"):
            if hasattr(GA, a): delattr(GA, a)
    r = attn_layout.install(model)
    try:
        assert r["installed"] and GA._attention is attn_layout._attention and GA._stock_attention.__qualname__ == "GridSelfAttention._attention"
        assert attn_layout.install(model)["already"]
    finally:
        GA._attention = GA._stock_attention
        for a in ("_attn_layout", "_attn_layout_reason", "_stock_attention"):
            if hasattr(GA, a): delattr(GA, a)


def test_short_pair_tracks_keep_the_stock_copies():
    assert attn_layout.MIN_TOKENS == 512 and set(attn_layout.COUNTS) == {"calls", "regrouped", "small", "generic"}
