"""msa2 trans2 (forward/msa2/trans2.py): the fused dim-64 Transition.
  * mode 1 (`trans2x`, lever msa_trans2_exact): BIT-IDENTICAL to the stock Boltz-2 Transition(64,256) under the trunk's regime (eval, CUDA autocast bf16),
    chunked (32) and unchunked, bf16 and fp32 inputs — rests on tl.dot == cuBLAS bf16 GEMM for K in {32,64,256} on this card/pin
    and on exhaustive silu equality; this test is the per-GPU gate for the adapter's claim.
  * mode 0 (`trans2`, lever msa_trans2): fast class — error vs an fp32 reference no larger than stock's own.
GPU test: skipped without CUDA / boltz."""
import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if importlib.util.find_spec("boltz") is None:
    pytest.skip("boltz not importable", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "msa2"))


def _t2():
    if "msa2" not in sys.modules:
        spec = importlib.util.spec_from_file_location("msa2", os.path.join(PKG, "__init__.py"), submodule_search_locations=[PKG])
        mod = importlib.util.module_from_spec(spec); sys.modules["msa2"] = mod; spec.loader.exec_module(mod)
    import importlib as il
    return il.import_module("msa2.trans2")


def _module(seed=0):
    from boltz.model.layers.transition import Transition
    torch.manual_seed(seed)
    trn = Transition(dim=64, hidden=256).cuda().eval()
    for p in trn.parameters():
        torch.nn.init.normal_(p, std=0.15)
    torch.nn.init.normal_(trn.norm.weight, 1.0, 0.2); torch.nn.init.normal_(trn.norm.bias, 0.0, 0.2)
    return trn


@pytest.mark.parametrize("dt,S,N,cs", [("bf16", 512, 64, None), ("fp32", 300, 200, None), ("bf16", 2049, 400, 32), ("fp32", 1531, 401, 32), ("fp32", 4096, 1000, 32)])
def test_mode1_bitwise_equals_stock(dt, S, N, cs):
    T2 = _t2(); trn = _module()
    from boltz.model.layers.transition import Transition
    x = torch.randn(1, S, N, 64, device="cuda") * 2.0 + 0.5
    x = x.to(torch.bfloat16) if dt == "bf16" else x
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = Transition.forward(trn, x, cs)
        for BM, nw in ((64, 4), (128, 8)):
            got = T2.transition64(trn, x, cs, mode=1, BM=BM, num_warps=nw)
            assert got.dtype == ref.dtype == torch.bfloat16 and got.shape == ref.shape
            assert torch.equal(got, ref), f"mode1 not bitwise: {(got != ref).sum().item()} mismatches, max|d|={(got.float() - ref.float()).abs().max().item()} ({dt} S={S} N={N} cs={cs} BM={BM})"


@pytest.mark.parametrize("dt,S,N,cs", [("bf16", 512, 64, None), ("fp32", 2049, 400, 32), ("bf16", 4096, 1000, 32)])
def test_mode0_error_no_worse_than_stock(dt, S, N, cs):
    T2 = _t2(); trn = _module()
    from boltz.model.layers.transition import Transition
    x = torch.randn(1, S, N, 64, device="cuda") * 2.0 + 0.5
    xin = x.to(torch.bfloat16) if dt == "bf16" else x
    with torch.no_grad():
        ref32 = Transition.forward(trn, xin.float(), None)          # the model's math in fp32 (no autocast)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            stock = Transition.forward(trn, xin, cs)
            got = T2.transition64(trn, xin, cs, mode=0)
    e_stock = ((stock.float() - ref32).norm() / ref32.norm()).item()
    e_got = ((got.float() - ref32).norm() / ref32.norm()).item()
    assert got.dtype == torch.bfloat16 and got.shape == stock.shape
    assert e_got <= e_stock * 1.05, (e_got, e_stock)
    assert ((got.float() - stock.float()).norm() / stock.float().norm()).item() < 2e-2


def test_supported_words():
    T2 = _t2(); trn = _module()
    assert T2.supported(trn, torch.zeros(4, 64, device="cuda", dtype=torch.float16), None) == "dtype"
    assert T2.supported(trn, torch.zeros(4, 64, dtype=torch.bfloat16), None) == "device"
    assert T2.supported(trn, torch.zeros(4, 64, device="cuda", dtype=torch.bfloat16), 32) is None
    from boltz.model.layers.transition import Transition
    big = Transition(dim=128, hidden=512).cuda().eval()
    assert T2.supported(big, torch.zeros(4, 128, device="cuda", dtype=torch.bfloat16), None) == "dims"
