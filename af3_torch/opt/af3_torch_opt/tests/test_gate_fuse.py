"""Lever gate_fuse: mask_gate_ is the stock statements when off; the fused kernels equal them (CUDA); install names its hosts."""
import inspect
import pytest
from af3_torch_opt import gate_fuse


def test_off_is_the_stock_statements():
    torch = pytest.importorskip("torch")
    gate_fuse._bind(); gate_fuse._STATE["on"] = False; gate_fuse.take()
    x = torch.randn(6, 6, 8).to(torch.bfloat16); g = torch.randn(6, 6, 8).to(torch.bfloat16); m = (torch.rand(6, 6) > 0.3).float()
    s = x.clone(); s *= m[:, :, None]; s *= torch.sigmoid(g)
    y = x.clone(); out = gate_fuse.mask_gate_(y, g, m)
    assert out is y and torch.equal(s, y) and gate_fuse.take() == {"calls": 1, "fused": 0, "generic": 1}
    s2 = x.clone(); s2 *= torch.sigmoid(g); y2 = x.clone(); gate_fuse.mask_gate_(y2, g)
    assert torch.equal(s2, y2)
    src = inspect.getsource(gate_fuse.mask_gate_)
    assert "x *= mask[..., None]" in src and "x *= torch.sigmoid(gate)" in src


def test_fused_forms_equal_the_stock_statements_on_cuda():
    torch = pytest.importorskip("torch"); pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("the fused kernels are Triton (CUDA)")
    gate_fuse._bind(); gate_fuse._STATE["on"] = True
    torch.manual_seed(0)
    for C in (128, 256, 64):
        for mdt in (torch.float32, torch.bfloat16, None):
            x = (torch.randn(48, 48, C, device="cuda") * 3).to(torch.bfloat16); g = (torch.randn(48, 48, C, device="cuda") * 4).to(torch.bfloat16)
            m = None if mdt is None else (torch.rand(48, 48, device="cuda") > 0.2).to(mdt)
            s = x.clone()
            if m is not None: s *= m[:, :, None]
            s *= torch.sigmoid(g)
            for form in ("full", "muls"):
                y = x.clone(); gate_fuse.gated_(y, g, m, form=form)
                assert torch.equal(s, y), (C, mdt, form, int((s != y).sum()))
    gate_fuse._STATE["on"] = False


def test_install_names_its_hosts():
    pytest.importorskip("torch")
    import types
    r = gate_fuse.install(types.SimpleNamespace())
    assert set(r) == {"installed", "already", "form", "hosts", "reason"} and r["form"] == gate_fuse.FORM == "full"
    if not r["hosts"]:
        assert r["installed"] is False and r["reason"] == "no_host_forward(tri_layout_and_attn_layout_absent)"


def test_hosts_resolve_the_module_forward_installed():
    """forward.py imports `gate_fuse` by its bare name; the host forwards must call THAT module object (its switch and census), not a second
    `af3_torch_opt.gate_fuse` import."""
    import importlib, sys, types
    from af3_torch_opt import tri_layout, attn_layout
    fake = types.ModuleType("gate_fuse"); keep = sys.modules.get("gate_fuse")
    sys.modules["gate_fuse"] = fake
    try:
        assert tri_layout._gate() is fake and attn_layout._gate() is fake
    finally:
        if keep is None: del sys.modules["gate_fuse"]
        else: sys.modules["gate_fuse"] = keep
    assert tri_layout._gate() is (sys.modules.get("gate_fuse") or importlib.import_module("af3_torch_opt.gate_fuse"))
