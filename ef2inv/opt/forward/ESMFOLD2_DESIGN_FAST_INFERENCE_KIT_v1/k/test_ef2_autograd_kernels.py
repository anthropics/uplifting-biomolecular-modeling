"""Unit tests for ef2_autograd_kernels / ef2_state_guard (run on a CUDA machine with the Biohub
transformers fork installed; weights are random-initialised modules, no checkpoint download needed).

    pytest -q k/test_ef2_autograd_kernels.py

Tolerance classes: kernel swaps must be at least as close
to the fp64 gold as the stock bf16 reference path (× 1.25 slack), frozen-weight fast backward must be BITWISE equal to
the full backward, graph replay must be BITWISE equal to eager.
"""
import os, sys, math, pytest, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import transformers.models.esmfold2.modeling_esmfold2_common as C
import ef2_autograd_kernels as agk

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
AMP = torch.amp.autocast("cuda", dtype=torch.bfloat16)


def make_block(seed=0):
    torch.manual_seed(seed)
    blk = C.PairUpdateBlock(d_pair=256, expansion_ratio=4)
    # give LayerNorms non-trivial affine params so the test exercises them
    for m in blk.modules():
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.normal_(m.weight, 1.0, 0.1); torch.nn.init.normal_(m.bias, 0.0, 0.1)
    return blk.to(dev).eval().requires_grad_(False)


def gold_block(blk, z_bf16, mask):
    z = z_bf16.detach().double().requires_grad_(True); cur = z
    for tri in (blk.tri_mul_out, blk.tri_mul_in):
        eng = tri._engine; P = {n: p.detach().double() for n, p in eng.named_parameters()}
        x = F.layer_norm(cur, (256,), P["norm_start.weight"], P["norm_start.bias"], C._EPS)
        sig, gate = (x @ P["proj_bundle.weight"].t()).split(512, dim=-1)
        l, r = (sig * torch.sigmoid(gate) * mask.double().unsqueeze(-1)).chunk(2, dim=-1)
        con = torch.einsum(eng._einsum_equation, l, r)
        cur = cur + (F.layer_norm(con, (256,), P["norm_mix.weight"], P["norm_mix.bias"], C._EPS) @ P["proj_emit.weight"].t()) * torch.sigmoid(x @ P["proj_gate.weight"].t())
    tr = blk.pair_transition; P = {n: p.detach().double() for n, p in tr.named_parameters()}
    x = F.layer_norm(cur, (256,), P["norm.weight"], P["norm.bias"], 1e-5)
    x1, x2 = (x @ P["ffn.w12.weight"].t()).split(tr.ffn.hidden_features, dim=-1)
    cur = cur + (F.silu(x1) * x2) @ P["ffn.w3.weight"].t()
    return z, cur


def run(blk, z, mask, gout, **cfg):
    agk.disable(blk)
    if cfg:
        blk.set_kernel_backend(None); blk.set_chunk_size(None); agk.enable(blk, only_under_grad=False, **cfg)
    else:
        blk.set_kernel_backend(None); blk.set_chunk_size(64)
    zz = z.detach().clone().requires_grad_(True)
    with torch.enable_grad(), AMP:
        o = blk(zz, pair_attention_mask=mask)
    o.backward(gout)
    agk.disable(blk)
    return o.detach().float(), zz.grad.detach().float()


def relrms(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@cuda
@pytest.mark.parametrize("N,B", [(48, 1), (100, 2), (150, 1)])
def test_kernel_variants_within_stock_error_class(N, B):
    blk = make_block()
    g = torch.Generator(device=dev).manual_seed(N)
    z = (torch.randn(B, N, N, 256, device=dev, generator=g) * 4).to(torch.bfloat16)
    mask = torch.ones(B, N, N, device=dev); mask[:, -3:, :] = 0; mask[:, :, -3:] = 0
    gout = torch.randn(B, N, N, 256, device=dev, generator=g).to(torch.bfloat16)
    zg, og = gold_block(blk, z, mask); og.backward(gout.double())
    o_ref, g_ref = run(blk, z, mask, gout)
    e_ref_o, e_ref_g = relrms(o_ref, og.detach()), relrms(g_ref, zg.grad)
    for cfg in (dict(trimul="bmm"), dict(trimul="bmm2"), dict(trimul="vendored"), dict(trimul="bmm", transition="fused"), dict(trimul="bmm2", transition="fused2")):
        o, gz = run(blk, z, mask, gout, **cfg)
        e_o, e_g = relrms(o, og.detach()), relrms(gz, zg.grad)
        assert e_o <= 1.25 * e_ref_o + 1e-6, (cfg, e_o, e_ref_o)
        assert e_g <= 1.25 * e_ref_g + 1e-6, (cfg, e_g, e_ref_g)
        cos = F.cosine_similarity(gz.flatten().double(), zg.grad.flatten(), dim=0).item()
        assert cos > 0.9995, (cfg, cos)


@cuda
def test_frozen_fast_backward_is_bitwise_equal_to_full_backward():
    blk = make_block(1); N, B = 96, 2
    g = torch.Generator(device=dev).manual_seed(5)
    z = (torch.randn(B, N, N, 256, device=dev, generator=g) * 4).to(torch.bfloat16)
    mask = torch.ones(B, N, N, device=dev); gout = torch.randn(B, N, N, 256, device=dev, generator=g).to(torch.bfloat16)
    o1, g1 = run(blk, z, mask, gout, trimul="bmm", transition="fused")
    o2, g2 = run(blk, z, mask, gout, trimul="bmm2", transition="fused2")
    assert torch.equal(o1, o2)
    assert torch.equal(g1, g2), (g1 - g2).abs().max()


@cuda
def test_checkpoint_policies_bitwise():
    torch.manual_seed(0)
    trunk = C.FoldingTrunk(n_layers=4).to(dev).eval().requires_grad_(False)
    for i, b in enumerate(trunk.blocks):
        b.load_state_dict(make_block(i).state_dict())
    N = 64; z = (torch.randn(1, N, N, 256, device=dev) * 4).to(torch.bfloat16); mask = torch.ones(1, N, N, device=dev); gout = torch.randn_like(z)
    outs = {}
    for mode in ("block", "none", "ckpt:2", "keep:2", "every:2"):
        agk.disable(trunk); agk.enable(trunk, trimul=None, transition=None, checkpoint=mode)
        zz = z.clone().requires_grad_(True)
        with torch.enable_grad(), AMP:
            o = trunk(zz, pair_attention_mask=mask)
        o.backward(gout); outs[mode] = (o.detach(), zz.grad.detach())
    for mode, (o, gz) in outs.items():
        assert torch.equal(o, outs["block"][0]) and torch.equal(gz, outs["block"][1]), mode


@cuda
def test_selective_ckpt_with_frozen_kernels_repeatable_and_equal_to_full_ckpt():
    """Regression: K-A2 under selective checkpointing over a 12-block trunk, repeated 4x with allocator churn between
    iterations, must give bitwise-identical input gradients to K-A2 with full checkpointing (same kernels, same order)."""
    trunk = C.FoldingTrunk(n_layers=12).to(dev).eval().requires_grad_(False)
    for i, b in enumerate(trunk.blocks):
        b.load_state_dict(make_block(100 + i).state_dict())
    N = 72; z = (torch.randn(1, N, N, 256, device=dev) * 4).to(torch.bfloat16); mask = torch.ones(1, N, N, device=dev); gout = torch.randn_like(z)
    def run_mode(mode):
        agk.disable(trunk); agk.enable(trunk, trimul="bmm2", transition="fused2", checkpoint=mode, only_under_grad=False)
        zz = z.clone().requires_grad_(True)
        with torch.enable_grad(), AMP:
            o = trunk(zz, pair_attention_mask=mask)
        o.backward(gout); return o.detach(), zz.grad.detach()
    ref = run_mode("block")
    for it in range(4):
        junk = [torch.empty(int(3e7), device=dev) for _ in range(it + 1)]; del junk; torch.cuda.empty_cache()
        for mode in ("ckpt:6", "keep:3", "none", "block"):
            o, g = run_mode(mode)
            assert torch.equal(o, ref[0]), (it, mode)
            assert torch.equal(g, ref[1]), (it, mode, (g - ref[1]).abs().max().item())


@cuda
def test_kd3_refround_transition_matches_stock_rounding():
    """K-D3: transition with stock rounding points. Versus the stock reference path (bf16 autocast) the output and dX must
    agree far more tightly than a kernel with changed rounding points: required is max|Δ| <= 2 bf16 ulp of the value scale on
    >= 99.9% of elements and cosine >= 0.999999 (fp32-ulp-level LN-stat/exp differences can flip an occasional bf16 rounding)."""
    torch.manual_seed(3)
    blk = make_block(3); tr = blk.pair_transition
    N = 48; x = (torch.randn(2, N, N, 256, device=dev) * 3).to(torch.bfloat16); gout = torch.randn_like(x)
    def run(fn):
        xx = x.clone().requires_grad_(True)
        with torch.enable_grad(), AMP:
            o = fn(xx)
        o.backward(gout); return o.detach().float(), xx.grad.detach().float()
    tr.set_chunk_size(None) if hasattr(tr, "set_chunk_size") else None
    o_ref, g_ref = run(lambda t: tr(t))
    cache = {}
    o3, g3 = run(lambda t: agk.transition_refround(t, tr, cache))
    for a, b_, name in ((o3, o_ref, "out"), (g3, g_ref, "dx")):
        scale = b_.abs().max()
        ulp2 = 2 * scale * 2.0 ** -8
        frac_off = ((a - b_).abs() > ulp2).float().mean().item()
        cos = F.cosine_similarity(a.flatten(), b_.flatten(), dim=0).item()
        assert frac_off <= 1e-3 and cos >= 0.999999, (name, frac_off, cos, (a - b_).abs().max().item(), scale.item())
    # and K-D2 (changed rounding) is measurably farther from stock than K-D3 on the same input
    o2, g2 = run(lambda t: agk.transition_fused_frozen(t, tr, agk._transition_weights(tr, {})))
    assert (g3 - g_ref).norm() < (g2 - g_ref).norm(), ((g3 - g_ref).norm().item(), (g2 - g_ref).norm().item())


@cuda
def test_kd3_lean_is_bitwise_equal_to_kd3():
    """refround_lean recomputes LN-out + W12 GEMM in backward instead of saving lin; must be bitwise-identical to refround."""
    torch.manual_seed(4)
    blk = make_block(4); tr = blk.pair_transition
    N = 40; x = (torch.randn(2, N, N, 256, device=dev) * 3).to(torch.bfloat16); gout = torch.randn_like(x)
    outs = []
    for lean in (False, True):
        xx = x.clone().requires_grad_(True); cache = {}
        with torch.enable_grad(), AMP:
            o = agk.transition_refround(xx, tr, cache, lean=lean)
        o.backward(gout); outs.append((o.detach(), xx.grad.detach()))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


@cuda
def test_ln_dx_only_backward_is_bitwise_equal_to_vendored():
    """dx-only LN backward (frozen gamma/beta) must give bitwise the same dX as the vendored 3-output kernel, for
    K-A2 (both LN layouts, with residual-link folding) and K-D3, at block level (PairUpdateBlock with bmm2 + refround)."""
    torch.manual_seed(5)
    blk = make_block(5)
    blk.set_kernel_backend(None); blk.set_chunk_size(None)
    agk.enable(blk, trimul="bmm2", transition="refround", checkpoint="none", only_under_grad=False)
    N = 44; x = (torch.randn(1, N, N, 256, device=dev) * 3).to(torch.bfloat16); gout = torch.randn_like(x)
    res = []
    for flag in (False, True):
        agk.LN_DX_ONLY = flag
        xx = x.clone().requires_grad_(True)
        with torch.enable_grad(), AMP:
            o = blk(xx, pair_attention_mask=None)
        o.backward(gout); res.append((o.detach().clone(), xx.grad.detach().clone()))
    agk.LN_DX_ONLY = True
    agk.disable(blk)
    assert torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1]), ((res[0][1] - res[1][1]).abs().max().item(),)


@cuda
def test_d59_enable_disable_and_steps_leave_process_state_unchanged():
    """Joint rule D59: enabling the kit, running fwd+bwd steps with every lever, and disabling must not change any
    process-global numeric state (matmul precision, TF32, cudnn flags, deterministic mode, default dtype, autocast, SDPA toggles)."""
    import ef2_state_guard as sg
    torch.manual_seed(11)
    blk = make_block(11); blk.set_kernel_backend(None); blk.set_chunk_size(None)
    ref = sg.snapshot()
    for kw in (dict(trimul="bmm2", transition="refround_lean", checkpoint="none", only_under_grad=False), dict(trimul=None, transition=None, checkpoint="block")):
        agk.enable(blk, **kw)
        assert sg.diff(sg.snapshot(), ref) == {}, sg.diff(sg.snapshot(), ref)
        N = 40; x = (torch.randn(1, N, N, 256, device=dev) * 3).to(torch.bfloat16).requires_grad_(True)
        with torch.enable_grad(), AMP:
            o = blk(x, pair_attention_mask=None)
        o.float().sum().backward()
        assert sg.diff(sg.snapshot(), ref) == {}, sg.diff(sg.snapshot(), ref)
        agk.disable(blk)
        assert sg.diff(sg.snapshot(), ref) == {}, sg.diff(sg.snapshot(), ref)
    g = agk.state_guard(); g.check("noop"); assert g.n_checks == 1 and not g.violations
