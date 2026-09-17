"""GPU (skipped without CUDA): ln_bf16's one-kernel LayerNorm is BIT-IDENTICAL to the stock fp32 round trip on bf16 inputs of a bf16-parameter
module at the trunk's shapes (the DiT attention lever's GPU contract lives in test_dit_apb_gpu.py)."""
import importlib

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


def _restore(cls, attr):
    stock = getattr(getattr(cls, attr), "__wrapped_stock__", None)
    if stock is not None:
        setattr(cls, attr, stock)


@cuda
@pytest.mark.parametrize("shape", [(1, 256, 256, 128), (1, 384, 384), (2, 64, 64, 128), (1, 512, 256)])
def test_ln_bf16_is_bitwise_to_the_stock_statement(shape):
    from atlasfold_opt.hooks import ln_bf16
    nm = importlib.import_module(ln_bf16.TARGET)
    torch.manual_seed(0)
    C = shape[-1]
    ln = nm.LayerNorm(C).cuda().to(torch.bfloat16)
    with torch.no_grad():
        ln.weight.add_(torch.randn_like(ln.weight) * 0.1); ln.bias.add_(torch.randn_like(ln.bias) * 0.1)
    x = (torch.randn(*shape, device="cuda") * 3 + 0.5).to(torch.bfloat16)
    stock = nm.LayerNorm.forward
    ins = ln_bf16.install("exact", "atlasfold-opt", {})
    try:
        with torch.no_grad():
            out = ln(x)
            with torch.autocast("cuda", torch.bfloat16):                       # the trunk runs under bf16 autocast: served all the same (autocast off at the call)
                out_ac = ln(x)
            ref = stock(ln, x)
        line = ins.lines[0]()
    finally:
        _restore(nm.LayerNorm, "forward")
    assert out.dtype == torch.bfloat16 and torch.equal(out, ref) and torch.equal(out_ac, ref)
    assert "served=2" in line and "fallback=0" in line
    ln32 = nm.LayerNorm(C).cuda()                                                   # fp32-parameter module (confidence / diffusion): the stock statement by name
    ins = ln_bf16.install("exact", "atlasfold-opt", {})
    try:
        with torch.no_grad():
            assert torch.equal(ln32(x), stock(ln32, x)) and torch.equal(ln32(x.float()), stock(ln32, x.float()))
        line = ins.lines[0]()
    finally:
        _restore(nm.LayerNorm, "forward")
    assert "fp32_params:1" in line and "fp32_input:1" in line and "served=0" in line


