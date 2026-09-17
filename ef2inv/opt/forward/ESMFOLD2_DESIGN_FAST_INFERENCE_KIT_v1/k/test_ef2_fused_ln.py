"""Unit tests for ef2_fused_ln (one CUDA device; the Biohub transformers fork importable).

    pytest -q k/test_ef2_fused_ln.py        (module-level test_* functions; the kit's pytest-free runner convention)

The claim under test is BITWISE identity with the kernel the lever replaces (agk._ln_bwd_dx_only -> the fork's _ln_bwd_kernel) on the
row-major layout it serves — odd row counts, bf16 and fp32 operands, with and without the residual-link gradient — plus: the
channel-major layout is routed to the vendored helper (tensor-equal by construction, counted), enable/disable swap and restore agk's
helper object exactly, a whole PairUpdateBlock backward under agk fast kernels gives tensor-equal input gradients with the lever on, and
the dx agrees with an fp64 LayerNorm backward to fp32-class accuracy.
"""
import os, sys, pytest, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers.models.esmfold2.modeling_esmfold2_common as C
import ef2_autograd_kernels as agk
import ef2_fused_ln as fl

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
AMP = torch.amp.autocast("cuda", dtype=torch.bfloat16)
FLR = agk._FLR
VENDORED = agk._ln_bwd_dx_only          # captured at import, before any enable()


def _operands(M, D, dtype, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = (torch.randn(M, D, device=dev, generator=g) * 3).to(dtype)
    gy = torch.randn(M, D, device=dev, generator=g).to(dtype)
    gr = torch.randn(M, D, device=dev, generator=g).to(dtype)
    w = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).float(); b = (0.1 * torch.randn(D, device=dev, generator=g)).float()
    _, mean, rstd = FLR._ln_fwd(x, w, b, 1e-5, FLR._LAYOUT_BND_BND, M, D)
    return x, gy, gr, w, b, mean, rstd


@cuda
@pytest.mark.parametrize("M,D,dtype,res", [(4099, 256, torch.bfloat16, True), (4099, 256, torch.bfloat16, False), (1000, 256, torch.float32, True),
                                             (777, 128, torch.bfloat16, True), (513, 64, torch.float32, False), (40401, 256, torch.bfloat16, True)])
def test_rowmajor_dx_bitwise_equal_to_vendored(M, D, dtype, res):
    x, gy, gr, w, b, mean, rstd = _operands(M, D, dtype, seed=M)
    ref = VENDORED(gy, x, w, mean, rstd, FLR._LAYOUT_BND_BND, gr if res else None, M, D)
    out = fl.ln_bwd_dx(gy, x, w, mean, rstd, FLR._LAYOUT_BND_BND, gr if res else None, M, D)
    assert out.dtype == ref.dtype and out.shape == ref.shape
    assert torch.equal(out, ref), (out.float() - ref.float()).abs().max().item()


@cuda
def test_fp64_agreement_and_determinism():
    M, D = 5000, 256
    x, gy, gr, w, b, mean, rstd = _operands(M, D, torch.bfloat16, seed=3)
    out1 = fl.ln_bwd_dx(gy, x, w, mean, rstd, FLR._LAYOUT_BND_BND, gr, M, D)
    out2 = fl.ln_bwd_dx(gy, x, w, mean, rstd, FLR._LAYOUT_BND_BND, gr, M, D)
    assert torch.equal(out1, out2)
    xd = x.double().requires_grad_(True)
    y = torch.nn.functional.layer_norm(xd, (D,), w.double(), b.double(), 1e-5)
    (gd,) = torch.autograd.grad(y, xd, gy.double())
    gold = (gd + gr.double())
    err = ((out1.double() - gold).norm() / gold.norm()).item()
    err_bf16_floor = ((gold.to(torch.bfloat16).double() - gold).norm() / gold.norm()).item()   # the output dtype's own rounding
    assert err <= 2.0 * err_bf16_floor + 1e-6, (err, err_bf16_floor)


@cuda
def test_enable_disable_swaps_agk_helper_and_routes_cmajor():
    assert agk._ln_bwd_dx_only is VENDORED and not fl.installed()
    blk = C.PairUpdateBlock(d_pair=256, expansion_ratio=4).to(dev).eval().requires_grad_(False)
    fl.enable(blk)
    try:
        assert agk._ln_bwd_dx_only is fl.ln_bwd_dx and fl.installed() and blk._fused_ln_enabled
        fl.enable(blk)                                                   # idempotent: still restorable
        M, D = 3000, 256
        x, gy, gr, w, b, mean, rstd = _operands(M, D, torch.bfloat16, seed=9)
        xT = x.t().contiguous()
        n0 = fl.stats()["routed_cmajor"]
        out = agk._ln_bwd_dx_only(gy, xT, w, mean, rstd, FLR._LAYOUT_DBN_BND, None, M, D)      # channel-major -> vendored, via the lever
        ref = VENDORED(gy, xT, w, mean, rstd, FLR._LAYOUT_DBN_BND, None, M, D)
        assert torch.equal(out, ref) and fl.stats()["routed_cmajor"] == n0 + 1, fl.describe()
    finally:
        fl.disable(blk)
    assert agk._ln_bwd_dx_only is VENDORED and not fl.installed() and not hasattr(blk, "_fused_ln_enabled")


@cuda
def test_block_backward_under_agk_fast_kernels_is_tensor_equal_with_lever():
    torch.manual_seed(0)
    blk = C.PairUpdateBlock(d_pair=256, expansion_ratio=4)
    for m in blk.modules():
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.normal_(m.weight, 1.0, 0.1); torch.nn.init.normal_(m.bias, 0.0, 0.1)
    blk = blk.to(dev).eval().requires_grad_(False); blk.set_kernel_backend(None); blk.set_chunk_size(None)
    N = 61
    z = (torch.randn(1, N, N, 256, device=dev) * 3).to(torch.bfloat16); gout = torch.randn_like(z)
    mask = torch.ones(1, N, N, device=dev); mask[:, -3:, :] = 0; mask[:, :, -3:] = 0
    agk.enable(blk, trimul="bmm2", transition="refround_lean", checkpoint="none", only_under_grad=False)

    def run():
        zz = z.detach().clone().requires_grad_(True)
        with torch.enable_grad(), AMP:
            o = blk(zz, pair_attention_mask=mask)
        o.backward(gout)
        return o.detach(), zz.grad.detach()
    try:
        o0, g0 = run()
        served0 = fl.stats()["served"]
        fl.enable(blk)
        o1, g1 = run()
        st = fl.stats()
        assert st["served"] - served0 == 3 and st["installed"], fl.describe()      # norm_in x2 (with link) + transition LN; norm_out x2 routed
        assert torch.equal(o1, o0) and torch.equal(g1, g0), ((g1.float() - g0.float()).abs().max().item())
    finally:
        fl.disable(blk); agk.disable(blk)


@cuda
def test_state_neutral():
    import ef2_state_guard as sg
    ref = sg.snapshot()
    blk = C.PairUpdateBlock(d_pair=256, expansion_ratio=4).to(dev).eval().requires_grad_(False)
    fl.enable(blk)
    x, gy, gr, w, b, mean, rstd = _operands(1000, 256, torch.bfloat16, seed=1)
    fl.ln_bwd_dx(gy, x, w, mean, rstd, FLR._LAYOUT_BND_BND, gr, 1000, 256)
    assert sg.diff(sg.snapshot(), ref) == {}
    fl.disable(blk)
    assert sg.diff(sg.snapshot(), ref) == {}


@cuda
@pytest.mark.parametrize("M,dtype,res", [(4099, torch.bfloat16, False), (40401, torch.bfloat16, False), (1000, torch.float32, True)])
def test_channel_major_kernel_within_one_ulp_of_vendored_and_fp64(M, dtype, res):
    """channel_major=True path: fp32 reassociation only — vs the vendored kernel at most one bf16 ulp of the value scale on any element,
    rel-RMS < 3e-5; vs an fp64 LayerNorm backward no worse than the vendored kernel by more than 10 %."""
    D = 256
    x, gy, gr, w, b, mean, rstd = _operands(M, D, dtype, seed=M + 1)
    xT = x.t().contiguous()
    ref = VENDORED(gy, xT, w, mean, rstd, FLR._LAYOUT_DBN_BND, gr if res else None, M, D)
    fl._SERVE_CMAJOR = True
    try:
        out = fl.ln_bwd_dx(gy, xT, w, mean, rstd, FLR._LAYOUT_DBN_BND, gr if res else None, M, D)
        out2 = fl.ln_bwd_dx(gy, xT, w, mean, rstd, FLR._LAYOUT_DBN_BND, gr if res else None, M, D)
    finally:
        fl._SERVE_CMAJOR = False
    assert out.shape == (D, M) and out.dtype == ref.dtype and torch.equal(out, out2)
    d = (out.float() - ref.float()); scale = ref.float().abs().max()
    ulp = scale * (2.0 ** -8 if dtype == torch.bfloat16 else 2.0 ** -23)
    relrms = (d.norm() / ref.float().norm()).item()
    assert d.abs().max() <= 1.001 * ulp and relrms < 3e-5, (d.abs().max().item(), ulp.item(), relrms)
    xd = x.double().requires_grad_(True)
    y = torch.nn.functional.layer_norm(xd, (D,), w.double(), b.double(), 1e-5)
    (gd,) = torch.autograd.grad(y, xd, gy.double())
    gold = (gd + (gr.double() if res else 0)).t()
    e_new = ((out.double() - gold).norm() / gold.norm()).item(); e_old = ((ref.double() - gold).norm() / gold.norm()).item()
    assert e_new <= 1.1 * e_old + 1e-9, (e_new, e_old)


@cuda
def test_channel_major_enable_flag_serves_block_backward():
    torch.manual_seed(1)
    blk = C.PairUpdateBlock(d_pair=256, expansion_ratio=4).to(dev).eval().requires_grad_(False); blk.set_kernel_backend(None); blk.set_chunk_size(None)
    N = 40
    z = (torch.randn(1, N, N, 256, device=dev) * 3).to(torch.bfloat16); gout = torch.randn_like(z); mask = torch.ones(1, N, N, device=dev)
    agk.enable(blk, trimul="bmm2", transition="refround_lean", checkpoint="none", only_under_grad=False)

    def run():
        zz = z.detach().clone().requires_grad_(True)
        with torch.enable_grad(), AMP:
            o = blk(zz, pair_attention_mask=mask)
        o.backward(gout)
        return zz.grad.detach().float()
    try:
        g0 = run()
        s0 = fl.stats()
        fl.enable(blk, channel_major=True)
        g1 = run()
        s1 = fl.stats()
        assert s1["served"] - s0["served"] == 3 and s1["served_cmajor"] - s0["served_cmajor"] == 2 and s1["channel_major"], fl.describe()
        rel = ((g1 - g0).norm() / g0.norm()).item()
        assert rel < 1e-3 and torch.nn.functional.cosine_similarity(g1.flatten().double(), g0.flatten().double(), dim=0).item() > 0.999999, rel
    finally:
        fl.disable(blk); agk.disable(blk)
    assert not fl.stats()["channel_major"] and not fl.installed()
