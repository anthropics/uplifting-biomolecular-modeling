"""The DiT cell adapters (cells/dit_glue.py, token_agg.py, atom_hoist.py over opt_core.of3_sampler): on a CUDA box with the openfold3 stack, the
dit_glue schedule against the stock block (numerics inside the stock block's own bf16 error of an fp64 evaluation; the refusal path returns the
stock output bitwise, counted by name) and seg_reduce against an fp64 reference; the adapters bind this kit's names. The GPU cases skip where
there is no CUDA device or no openfold3 package; the store / segments / row-schedule contracts are opt_core's tests (tests/test_of3_sampler_cpu.py)."""
import pytest

torch = pytest.importorskip("torch")

from opt_core.of3_sampler import dit_rows  # noqa: E402
from openfold3_ob0_opt.cells import apb_trunk, atom_hoist, dit_glue, token_agg  # noqa: E402


def test_adapters_bind_this_kits_names():
    assert atom_hoist._core.ENV == atom_hoist.ENV == "OPENFOLD3_OB0_OPT_ATOM_HOIST" and atom_hoist._core.PREFIX == "[openfold3_ob0-opt/atom_hoist]" and atom_hoist._memo.PREFIX == "[openfold3_ob0-opt/rollout_memo]"
    assert dit_glue._core.ENV_CORE == "OPENFOLD3_OB0_OPT_DIT_GLUE_CORE" and dit_glue._core.HIGH_PRECISION == "override" and token_agg._core.ENV == "OPENFOLD3_OB0_OPT_TOKEN_AGG"
    assert apb_trunk._core.ENV == apb_trunk.ENV == "OPENFOLD3_OB0_OPT_APB_TRUNK" and apb_trunk._core.HIGH_PRECISION == "override" and apb_trunk._core.PREFIX == "[openfold3_ob0-opt/apb_trunk]" and apb_trunk.STATE is apb_trunk._core.STATE
    assert atom_hoist.STATE is atom_hoist._core.STATE and dit_glue.STATE is dit_glue._core.STATE and token_agg.STATE is token_agg._core.STATE   # the stack's probes read the adapters' STATE




def _dtk():
    pytest.importorskip("triton")
    from opt_core.kernels import route                                  # the core's carried copy, as the cells route it
    route("dtk_kernels")
    import importlib
    return importlib.import_module("dtk_kernels")


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@cuda
def test_seg_reduce_matches_the_masked_mean_in_fp64():
    dk = _dtk()
    g = torch.Generator(device="cpu").manual_seed(3)
    for (N, S, C, dtype) in ((37, 3, 96, torch.float32), (5, 1, 768, torch.bfloat16), (300, 2, 33, torch.float16)):
        counts = torch.randint(1, 24, (N,), generator=g); counts[N // 2] = 0
        idx = torch.repeat_interleave(torch.arange(N), counts)
        pad = 7
        idx = torch.cat([idx, torch.zeros(pad, dtype=idx.dtype)]); A = idx.numel()
        mask = torch.ones(A); mask[-pad:] = 0; mask[torch.randint(0, A - pad, (A // 9,), generator=g)] = 0
        feat = torch.randn(S, A, C, generator=g).to("cuda", dtype); idx = idx.cuda(); mask = mask.cuda()
        st, ct, mx, ok = dk.segments(idx, mask, N)
        assert ok
        out = dk.seg_reduce(feat, st, ct, mx, atom_mask=mask, mean=True, out_dtype=torch.float32)
        f64 = feat.double() * mask.double()[None, :, None]
        ref = torch.zeros(S, N, C, dtype=torch.float64, device="cuda").index_add_(1, idx, f64)
        den = torch.zeros(N, dtype=torch.float64, device="cuda").index_add_(0, idx, mask.double())
        ref = ref / (den[None, :, None] + 1e-9)
        err = (out.double() - ref).abs().max().item(); scale = ref.abs().max().item()
        assert err <= 4 * torch.finfo(torch.float32).eps * max(scale, 1.0) * 24, (N, S, C, dtype, err)


def _block(c_a=768, c_s=384, heads=16):
    of3 = pytest.importorskip("openfold3.core.model.layers.diffusion_transformer")
    import inspect
    sig = inspect.signature(of3.DiffusionTransformerBlock.__init__).parameters
    kw = dict(c_a=c_a, c_s=c_s, c_z=128, c_hidden=c_a // heads, no_heads=heads, n_transition=2, use_ada_layer_norm=True, n_query=None, n_key=None, inf=1e9)
    kw = {k: v for k, v in kw.items() if k in sig}                     # the token (self-attention) block of the diffusion transformer: no query/key windows
    missing = [k for k, p in sig.items() if k not in kw and k != "self" and p.default is inspect.Parameter.empty]
    assert not missing, missing
    torch.manual_seed(0)
    blk = of3.DiffusionTransformerBlock(**kw).cuda().eval()
    with torch.no_grad():
        for p in blk.parameters():                                      # adaLN-zero inits gates to constants: randomise everything so every path carries signal
            p.copy_(torch.randn_like(p) * (0.05 if p.dim() > 1 else 0.5))
    return of3, blk


@cuda
def test_dit_glue_numerics_and_refusal_path():
    of3, blk = _block()
    DG = dit_glue._core
    dk = _dtk(); DG._DK["mod"] = dk; dit_rows._DK["mod"] = dk
    DG.STATE.update(state="on", core_pref="dtk"); DG._CORE.update(mod=None, tried=False)
    S, N = 3, 203                                                       # N not a multiple of any tile
    torch.manual_seed(1)
    a = torch.randn(1, S, N, 768, device="cuda"); s = torch.randn(1, 1, N, 384, device="cuda")
    z = torch.randn(1, 1, N, N, 128, device="cuda"); mask = (torch.rand(1, 1, N, device="cuda") > 0.1).float()
    with torch.no_grad():
        ref64 = blk.double()(a=a.double(), s=s.double(), z=z.double(), mask=mask.double()).float(); blk.float()
        with torch.autocast("cuda", torch.bfloat16):
            stock = blk(a=a, s=s, z=z, mask=mask)
            glue = DG.block_forward(blk, a, s, z, mask)
    assert glue.shape == stock.shape and glue.dtype == stock.dtype == a.dtype
    e_stock = (stock - ref64).norm() / ref64.norm(); e_glue = (glue - ref64).norm() / ref64.norm()
    assert float(e_glue) <= 2.0 * float(e_stock) + 1e-6, (float(e_glue), float(e_stock))
    m_stock = (stock - ref64).abs().max() / ref64.abs().max(); m_glue = (glue - ref64).abs().max() / ref64.abs().max()
    assert float(m_glue) <= 2.5 * float(m_stock) + 1e-6, (float(m_glue), float(m_stock))
    # the refusal path through the patched class forward: a Linear precision override the schedule does not read -> the stock block's output, counted
    DG._patch_block(of3)
    lin = blk.attention_pair_bias.layer_norm_a.linear_g
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        lin.precision = torch.float32
        want = of3.DiffusionTransformerBlock.forward.__wrapped__(blk, a=a, s=s, z=z, mask=mask)
        got = blk(a=a, s=s, z=z, mask=mask)
        del lin.precision
        served = DG.STATE["served"]
        again = blk(a=a, s=s, z=z, mask=mask)
    assert torch.equal(want, got) and any(k.startswith("transition_adaln:linear-precision") for k in DG.STATE["fallback"]), DG.STATE["fallback"]
    assert DG.STATE["served"] == served + 1 and torch.allclose(again, glue) and not DG.STATE["errors"]
    of3.DiffusionTransformerBlock.forward = of3.DiffusionTransformerBlock.forward.__wrapped__


@cuda
def test_conditioned_rows_serve_the_atom_transformer_flow_and_refuse_by_name():
    """dit_glue's class-wide rows: ConditionedTransitionBlock / AdaLN on the atom transformer's fp32 flow (per-atom conditioning rows periodic over
    the samples, an atom mask, the rollout's fp32 autocast island) agree with the stock modules at the stock modules' own distance from an fp64
    evaluation; under 16-bit autocast the transition is served in the compute dtype (stock's output dtype) and a standalone AdaLN call is refused
    by name (`mixed_bfloat16`) with the stock output returned bitwise; every path is counted."""
    of3 = pytest.importorskip("openfold3.core.model.layers.diffusion_transformer")
    import copy
    DG = dit_glue._core
    dk = _dtk(); DG._DK["mod"] = dk; dit_rows._DK["mod"] = dk
    CTB = of3.ConditionedTransitionBlock
    torch.manual_seed(0)
    m = CTB(c_a=128, c_s=128, n=2).cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * (0.08 if p.dim() > 1 else 0.5))
    S, A = 3, 517
    a = torch.randn(1, S, A, 128, device="cuda"); c = torch.randn(1, 1, A, 128, device="cuda"); mask = (torch.rand(1, 1, A, device="cuda") > 0.1).float()
    stock_c, stock_a = CTB.forward, type(m.layer_norm).forward
    saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DG.STATE.items() if k.startswith(("cond_", "adaln_", "rows"))}
    assert DG._patch_rows(of3) == ""
    try:
        DG.STATE["rows_state"] = "on"
        with torch.no_grad():
            ref_u = copy.deepcopy(m).double()(a.double(), c.double(), mask=mask.double())
            ref_x = stock_a(copy.deepcopy(m.layer_norm).double(), a.double(), c.double())
            with torch.autocast("cuda", dtype=torch.float32):                                   # the rollout's fp32 island around the atom transformer
                want_u = stock_c(m, a, c, mask=mask); want_x = stock_a(m.layer_norm, a, c)
                n0, n1 = DG.STATE["cond_served"], DG.STATE["adaln_served"]
                got_u = m(a, c, mask=mask); got_x = m.layer_norm(a, c)
        assert DG.STATE["cond_served"] == n0 + 1 and DG.STATE["adaln_served"] == n1 + 1, DG.census_line()
        assert got_u.dtype == want_u.dtype == torch.float32 and got_x.dtype == want_x.dtype == torch.float32 and got_u.shape == want_u.shape
        for got, want, ref in ((got_u, want_u, ref_u), (got_x, want_x, ref_x)):
            e_s = float((want.double() - ref).norm() / ref.norm()); e_g = float((got.double() - ref).norm() / ref.norm())
            assert e_g <= 2.0 * e_s + 2e-6, (e_g, e_s)                                              # TF32 GEMMs on both sides; the row kernels round once
        assert any(k.startswith("float32:periodic:mask") for k in DG.STATE["cond_modes"]), DG.STATE["cond_modes"]
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):                               # a token-side call outside the block schedule
            want_u16 = stock_c(m, a, c, mask=mask); got_u16 = m(a, c, mask=mask)
            want_x16 = stock_a(m.layer_norm, a, c); nfb = dict(DG.STATE["adaln_fallback"]); got_x16 = m.layer_norm(a, c)
        assert got_u16.dtype == want_u16.dtype and float((got_u16.double() - ref_u).norm() / ref_u.norm()) <= 2.5 * float((want_u16.double() - ref_u).norm() / ref_u.norm()) + 1e-6
        assert torch.equal(got_x16, want_x16) and DG.STATE["adaln_fallback"].get("mixed_bfloat16", 0) == nfb.get("mixed_bfloat16", 0) + 1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
            assert torch.equal(m(a, c, mask=mask, chunk_size=4), stock_c(m, a, c, mask=mask, chunk_size=4)) and DG.STATE["cond_fallback"].get("chunked") >= 1
            m.train()
            assert torch.equal(m(a, c, mask=mask), stock_c(m, a, c, mask=mask)) and DG.STATE["cond_fallback"].get("training") >= 1
            m.eval()
        assert " rows=" in DG.census_line() and " cond_served=" in DG.census_line()
    finally:
        CTB.forward = stock_c; type(m.layer_norm).forward = stock_a
        for k, v in saved.items():
            if isinstance(v, dict):
                DG.STATE[k].clear(); DG.STATE[k].update(v)
            else:
                DG.STATE[k] = v
