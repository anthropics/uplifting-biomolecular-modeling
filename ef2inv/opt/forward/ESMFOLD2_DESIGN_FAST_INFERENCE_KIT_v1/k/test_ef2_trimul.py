"""GPU unit tests for k/ef2_trimul.py (pytest, or `python test_ef2_trimul.py` without pytest).

fused (FAST class):  error vs an fp64 evaluation of the stock math within the stock kernels' own error class (<= 1.25x, the kit's
                     standard), shape sweep incl. non-multiple-of-8 token counts, batch > 1, masked rows, 700 tokens, and a
                     > 2^31-element row space (int64 indexing); enable/disable round trip, refusal of agk-trimul blocks,
                     composition with ef2_autograd_kernels (transition + checkpoint policy), process-state neutrality.
                     the per-capability gated-GEMM launch table (sm_90) tensor-equal — outputs AND input-gradients — to the
                     vendored launches at 129..800 tokens (bitwise per pin), the table itself (CPU).
cueq_tiles (EXACT):  outputs AND input-gradients tensor-equal to the untouched cuEquivariance path at 257..700 tokens (bitwise
                     per pin: torch 2.11 / triton 3.6 / cuequivariance 0.10) with the tile entry OF THE CARD THE TEST RUNS ON
                     (one entry per compute capability: sm_90, sm_80; a card without an entry skips by name — nothing is
                     installed there); the per-capability table itself (CPU); disable restores the library's table.
"""
import contextlib
import os
import sys

import torch
import torch.nn.functional as F

try:
    import pytest
except ImportError:                      # the kit images ship no pytest; the marks below then only feed the __main__ runner
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.models.esmfold2.modeling_esmfold2_common as C  # noqa: E402
import ef2_trimul as tm  # noqa: E402

dev = "cuda"
AMP = torch.amp.autocast("cuda", dtype=torch.bfloat16)
D = 256


def skipif(cond, reason=""):
    def deco(f):
        f._skip = bool(cond) or getattr(f, "_skip", False)
        return pytest.mark.skipif(cond, reason=reason)(f) if pytest else f
    return deco


def parametrize(names, values):
    def deco(f):
        f._param = (names, values)
        return pytest.mark.parametrize(names, values)(f) if pytest else f
    return deco


@contextlib.contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"{exc.__name__} not raised")


cuda = skipif(not torch.cuda.is_available(), reason="CUDA only")
CC = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (0, 0)     # the card the test runs on: its tile entry is the one proven


def make_block(seed=0):
    torch.manual_seed(seed)
    blk = C.PairUpdateBlock(d_pair=D, expansion_ratio=4)
    for m in blk.modules():
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.normal_(m.weight, 1.0, 0.1); torch.nn.init.normal_(m.bias, 0.0, 0.1)
    return blk.to(dev).eval().requires_grad_(False)


def make_inputs(N, B=1, seed=0, masked_tail=0):
    g = torch.Generator(device=dev).manual_seed(1000 + N + seed)
    z = (torch.randn(B, N, N, D, device=dev, generator=g) * 4).to(torch.bfloat16)
    mask = torch.ones(B, N, N, device=dev)
    if masked_tail:
        mask[:, -masked_tail:, :] = 0; mask[:, :, -masked_tail:] = 0
    gout = torch.randn(B, N, N, D, device=dev, generator=g).to(torch.bfloat16)
    return z, mask, gout


def gold(blk, z_bf16, mask, gout):
    """fp64 reference of pair + TriMulOut(pair); pair + TriMulIn(pair) (stock math)."""
    z = z_bf16.detach().double().requires_grad_(True); cur = z
    for tri in (blk.tri_mul_out, blk.tri_mul_in):
        eng = tri._engine; P = {n: p.detach().double() for n, p in eng.named_parameters()}
        x = F.layer_norm(cur, (D,), P["norm_start.weight"], P["norm_start.bias"], C._EPS)
        sig, gate = (x @ P["proj_bundle.weight"].t()).split(2 * D, dim=-1)
        l, r = (sig * torch.sigmoid(gate) * mask.double().unsqueeze(-1)).chunk(2, dim=-1)
        con = torch.einsum(eng._einsum_equation, l, r)
        cur = cur + (F.layer_norm(con, (D,), P["norm_mix.weight"], P["norm_mix.bias"], C._EPS) @ P["proj_emit.weight"].t()) \
            * torch.sigmoid(x @ P["proj_gate.weight"].t())
    cur.backward(gout.double())
    return cur.detach(), z.grad.detach()


def run_stock(blk, z, mask, gout, backend=None):
    blk.set_kernel_backend(backend); blk.set_chunk_size(None)
    zz = z.clone().requires_grad_(True)
    with torch.enable_grad(), AMP:
        p = zz + blk.tri_mul_out(zz, mask=mask)
        p = p + blk.tri_mul_in(p, mask=mask)
    p.backward(gout)
    blk.set_kernel_backend(None)
    return p.detach(), zz.grad.detach()


def run_fused(blk, z, mask, gout):
    zz = z.clone().requires_grad_(True)
    with torch.enable_grad(), AMP:
        p = tm.pair_trimul_residual(blk, zz, mask, variant="fused")
    p.backward(gout)
    return p.detach(), zz.grad.detach()


def relrms(a, ref):
    a = a.double().flatten(); ref = ref.double().flatten()
    return ((a - ref).norm() / ref.norm()).item()


@cuda
@parametrize("N,B,masked", [(72, 2, 0), (129, 1, 9), (200, 1, 0)])
def test_fused_within_stock_error_class(N, B, masked):
    blk = make_block(N); z, mask, gout = make_inputs(N, B, masked_tail=masked)
    o64, g64 = gold(blk, z, mask, gout)
    o_s, g_s = run_stock(blk, z, mask, gout)
    o_f, g_f = run_fused(blk, z, mask, gout)
    assert torch.isfinite(o_f).all() and torch.isfinite(g_f).all()
    for name, f, s in (("out", relrms(o_f, o64), relrms(o_s, o64)), ("dz", relrms(g_f, g64), relrms(g_s, g64))):
        assert f <= 1.25 * s, (N, B, name, f, s)
    assert F.cosine_similarity(g_f.flatten().double(), g64.flatten(), dim=0).item() > 0.99999


@cuda
def test_fused_padded_operands_are_zero_and_masked_rows_inert():
    """N=129 (pads to 136): the channel-major a|b and contraction output carry exact zeros in pad slots, and rows whose mask is
    0 produce the stock result (delta from those rows = 0)."""
    blk = make_block(3); N = 129; z, mask, gout = make_inputs(N, 1, masked_tail=17)
    cache = {}
    w = tm._weights(blk.tri_mul_out._engine, cache)
    with torch.enable_grad(), AMP:
        zz = z.clone().requires_grad_(True)
        out = tm._TriMulResidual.apply(zz, mask, w)
    ab, o = out.grad_fn.saved_tensors[1], out.grad_fn.saved_tensors[2]
    Np = tm._padded(N)
    assert ab.shape[-2:] == (Np, Np) and Np % tm.PAD == 0 and Np >= N
    assert ab[..., N:, :].abs().max().item() == 0 and ab[..., :, N:].abs().max().item() == 0
    assert o[..., N:, :].abs().max().item() == 0 and o[..., :, N:].abs().max().item() == 0
    o_s, g_s = run_stock(blk, z, mask, gout)
    o_f, g_f = run_fused(blk, z, mask, gout)
    o64, g64 = gold(blk, z, mask, gout)
    assert relrms(o_f, o64) <= 1.25 * relrms(o_s, o64) and relrms(g_f, g64) <= 1.25 * relrms(g_s, g64)


@cuda
def test_fused_700_tokens_within_stock_error_class():
    blk = make_block(7); N = 700; z, mask, gout = make_inputs(N, 1)
    o64, g64 = gold(blk, z, mask, gout)
    o_s, g_s = run_stock(blk, z, mask, gout, backend=C.BACKEND_CUEQ if C.CUE_AVAILABLE else None)
    o_f, g_f = run_fused(blk, z, mask, gout)
    assert relrms(o_f, o64) <= 1.25 * relrms(o_s, o64), (relrms(o_f, o64), relrms(o_s, o64))
    assert relrms(g_f, g64) <= 1.25 * relrms(g_s, g64), (relrms(g_f, g64), relrms(g_s, g64))
    del o64, g64
    torch.cuda.empty_cache()


@cuda
def test_fused_int64_row_space():
    """N=1470 (pads to 1472): B*Np*Np*4C > 2^31 elements in the [value|gate] gradient buffer and > 2^31-element offsets in
    the channel-major intermediates — must agree with the cuEquivariance / eager path (no fp64 gold at this size)."""
    blk = make_block(11); N = 1470; z, mask, gout = make_inputs(N, 1, masked_tail=5)
    assert tm._padded(N) ** 2 * 4 * D >= 2 ** 31
    o_s, g_s = run_stock(blk, z, mask, gout, backend=C.BACKEND_CUEQ if C.CUE_AVAILABLE else None)
    o_f, g_f = run_fused(blk, z, mask, gout)
    assert torch.isfinite(o_f).all() and torch.isfinite(g_f).all()
    assert relrms(o_f, o_s) < 1e-2 and relrms(g_f, g_s) < 1e-2, (relrms(o_f, o_s), relrms(g_f, g_s))
    del o_s, g_s, o_f, g_f, z, gout
    torch.cuda.empty_cache()


@contextlib.contextmanager
def gemm_table(use: bool):
    prev = tm.GEMM_TABLE["use"]; tm.GEMM_TABLE["use"] = use
    try:
        yield
    finally:
        tm.GEMM_TABLE["use"] = prev


def test_gemm_table_per_cc():
    """CPU: one gated-GEMM launch table per compute capability; a card without an entry runs the vendored launches by name."""
    assert tm._gg_table((9, 0)) == (tm._GG_SM90, "sm_90") and tm._gg_table((8, 0)) == (tm._GG_SM80, "sm_80")
    assert tm._gg_table((8, 6)) == (None, "vendored:cc_untuned:sm_86") and tm._gg_table((12, 0))[0] is None
    with gemm_table(False):
        assert tm._gg_table((9, 0)) == (None, "vendored:off") and tm._gg_table((8, 0)) == (None, "vendored:off")
    assert set(tm._GG_SM90) == {"dual_fwd", "dual_bwd", "og_fwd", "og_bwd"} and all(e["route"] in ("kit", "vendored_tiles") for e in tm._GG_SM90.values())
    assert set(tm._GG_SM80) <= {"dual_fwd", "dual_bwd", "og_fwd", "og_bwd"} and all(e["route"] == "vendored_tiles" and e["TILE_K"] == (32 if k == "dual_fwd" else 64) for k, e in tm._GG_SM80.items())   # sm_80: the vendored kernels retiled (no TMA); TILE_K as each kernel ships it (the K loop unchanged)
    assert all(512 % e["TILE_N"] == 0 for t in tm._GG_BY_CC.values() for k, e in t.items() if k.startswith("dual"))      # the dual kernels' column chunks tile 2C exactly


@cuda
@skipif(tm._gg_table(CC)[0] is None, reason=f"cc_untuned: ef2_trimul carries no gated-GEMM launch table for sm_{CC[0]}{CC[1]} (the vendored launches run there; nothing to prove)")
@parametrize("N,B,masked", [(257, 1, 0), (129, 2, 9), (431, 1, 3), (640, 1, 0), (800, 1, 5)])
def test_gemm_table_bitwise(N, B, masked):
    """Bitwise per pin (torch 2.11 / triton 3.6) AND per card: the fused path's outputs AND input-gradients with this card's gated-GEMM
    launch table (sm_90: the kit's persistent TMA kernels for the dual GEMM fwd / bwd partials, tuned tiles on the vendored out-gate
    kernels) are tensor-equal to the vendored launches (GEMM_TABLE use=False) — x8-padded and unpadded token counts, batch 2, masked
    rows, 800 tokens.  Re-run on any new pin or card; a new capability's entry is proven here before it is added to _GG_BY_CC."""
    blk = make_block(N % 7); z, mask, gout = make_inputs(N, B=B, masked_tail=masked)
    with gemm_table(False):
        p0, g0 = run_fused(blk, z, mask, gout)
    p1, g1 = run_fused(blk, z, mask, gout)
    assert tm.GEMM_TABLE["use"] and tm._gg_table()[1] == f"sm_{CC[0]}{CC[1]}"
    assert torch.equal(p1, p0), f"pair differs: max |d| {(p1.float() - p0.float()).abs().max().item():.3g}"
    assert torch.equal(g1, g0), f"d pair differs: max |d| {(g1.float() - g0.float()).abs().max().item():.3g}"
    trunk = _trunk(1); h = tm.enable(trunk, variant="fused")
    try:
        assert h.gemm == f"sm_{CC[0]}{CC[1]}" and tm.describe(h).endswith(f"gemm=sm_{CC[0]}{CC[1]}")
    finally:
        tm.disable(trunk)


def test_cueq_tiles_table_per_cc():
    """The per-capability table (CPU): the sm_90 entry byte for byte as the H100 has shipped it, the sm_80 entry, TILE_K = the library
    default's 32 in every entry (the K loop unchanged), and a capability without an entry named cc_untuned (nothing installed)."""
    assert tm._cueq_tiles_for((9, 0)) == (dict(TILE_M=128, TILE_N=128, TILE_K=32, num_stages=4, num_warps=8), None)
    assert tm._cueq_tiles_for((8, 0)) == (tm._CUEQ_TILES_SM80, None) and (tm._CUEQ_TILES_SM80["TILE_M"], tm._CUEQ_TILES_SM80["TILE_N"], tm._CUEQ_TILES_SM80["num_warps"]) == (64, 128, 4)
    assert all(cfg["TILE_K"] == 32 for cfg in tm._CUEQ_TILES_BY_CC.values())
    assert tm._cueq_tiles_for((7, 5)) == (None, tm.CC_UNTUNED) and tm._cueq_tiles_for((10, 0)) == (None, tm.CC_UNTUNED)


def _cueq_tiles_bitwise_case(N, expect_cfg):
    from cuequivariance_ops.triton.cache_manager import get_cache_manager
    blk = make_block(N + 1); z, mask, gout = make_inputs(N, 1, masked_tail=3)
    o0, g0 = run_stock(blk, z, mask, gout, backend=C.BACKEND_CUEQ)
    blk.set_kernel_backend(C.BACKEND_CUEQ)
    h = tm.enable(blk, variant="cueq_tiles")
    assert len(h.cueq_saved) > 0 and h.cueq_reason == "" and tm.describe(h) == f"ef2_trimul variant=cueq_tiles entries={len(h.cueq_saved)}"
    cm = get_cache_manager()
    some = next(iter(h.cueq_saved))
    assert cm.gpu_cache[some[0]][some[1]]["config"] == expect_cfg, (cm.gpu_cache[some[0]][some[1]]["config"], expect_cfg)
    try:
        o1, g1 = run_stock(blk, z, mask, gout, backend=C.BACKEND_CUEQ)
    finally:
        tm.disable(blk)
    assert torch.equal(o1, o0), (o1.float() - o0.float()).abs().max()
    assert torch.equal(g1, g0), (g1.float() - g0.float()).abs().max()
    assert cm.gpu_cache[some[0]].get(some[1]) == h.cueq_saved.get(some) or cm.gpu_cache[some[0]].get(some[1]) is None
    o2, g2 = run_stock(blk, z, mask, gout, backend=C.BACKEND_CUEQ)          # restored table -> still the same numbers
    assert torch.equal(o2, o0) and torch.equal(g2, g0)


@cuda
@skipif(not C.CUE_AVAILABLE, reason="cuequivariance_torch not importable")
@skipif(tm._cueq_tiles_for(CC)[0] is None, reason=f"cc_untuned: ef2_trimul carries no cuEquivariance tile entry for sm_{CC[0]}{CC[1]} (the library default runs there; nothing to prove)")
@parametrize("N", [257, 363, 450, 700])
def test_cueq_tiles_bitwise(N):
    """EXACT class, bitwise per pin (torch 2.11 / triton 3.6 / cuequivariance 0.10) AND per card: proves the tile entry of the card it
    runs on (sm_90 on an H100, sm_80 on an A100) against the library's untouched table. Re-run on any new pin or card."""
    _cueq_tiles_bitwise_case(N, tm._cueq_tiles_for(CC)[0])


@cuda
@skipif(not C.CUE_AVAILABLE, reason="cuequivariance_torch not importable")
@skipif(CC != (8, 0), reason="the sm_80 entry (64x128x32, 4 warps) is proven on a cc 8.0 card (A100) only")
@parametrize("N", [431, 700])
def test_cueq_tiles_bitwise_sm80(N):
    """The A100 entry by name at the A100 record's sizes (431 = a 351-residue target + 80, 700): tensor-equal out and d pair vs the library default."""
    _cueq_tiles_bitwise_case(N, tm._CUEQ_TILES_SM80)


@cuda
@skipif(not C.CUE_AVAILABLE, reason="cuequivariance_torch not importable")
def test_cueq_tiles_untuned_cc_keeps_library_default_by_name():
    """A card whose capability has no entry: enable installs nothing, names cc_untuned:sm_NN (describe / the LEVER line's reason word),
    and the block computes exactly what the untouched library computes."""
    N = 257
    blk = make_block(N + 1); z, mask, gout = make_inputs(N, 1)
    o0, g0 = run_stock(blk, z, mask, gout, backend=C.BACKEND_CUEQ)
    blk.set_kernel_backend(C.BACKEND_CUEQ)
    table = dict(tm._CUEQ_TILES_BY_CC)
    try:
        tm._CUEQ_TILES_BY_CC.pop(CC, None)                                   # this card, as if untuned
        h = tm.enable(blk, variant="cueq_tiles")
        assert h.cueq_saved == {} and h.cueq_reason == f"cc_untuned:sm_{CC[0]}{CC[1]}" and tm.describe(h) == f"ef2_trimul variant=cueq_tiles entries=0 reason=cc_untuned:sm_{CC[0]}{CC[1]}"
        o1, g1 = run_stock(blk, z, mask, gout, backend=C.BACKEND_CUEQ)
        tm.disable(blk)
    finally:
        tm._CUEQ_TILES_BY_CC.clear(); tm._CUEQ_TILES_BY_CC.update(table)
    assert torch.equal(o1, o0) and torch.equal(g1, g0)


def _trunk(n_layers, seed=0):
    trunk = C.FoldingTrunk(n_layers=n_layers).to(dev).eval().requires_grad_(False)
    for i, b in enumerate(trunk.blocks):
        b.load_state_dict(make_block(seed + i).state_dict())
    return trunk


def _trunk_step(trunk, z, mask, gout):
    zz = z.clone().requires_grad_(True)
    with torch.enable_grad(), AMP:
        o = trunk(zz, pair_attention_mask=mask)
    o.backward(gout)
    return o.detach(), zz.grad.detach()


@cuda
def test_enable_disable_roundtrip_and_state_neutral():
    import ef2_state_guard as sg
    trunk = _trunk(2); N = 72; z, mask, gout = make_inputs(N, 1)
    o0, g0 = _trunk_step(trunk, z, mask, gout)
    ref = sg.snapshot()
    h = tm.enable(trunk, variant="fused")
    assert not sg.diff(sg.snapshot(), ref)
    o1, g1 = _trunk_step(trunk, z, mask, gout)
    assert h.stats["served"] == 8 and h.stats["fallback"] == 0, h.stats   # 2 blocks x 2 tri-muls x (fwd + the block checkpoint's recompute)
    with torch.no_grad(), AMP:                                   # only_under_grad: no-grad calls take the module's own path
        trunk(z, pair_attention_mask=mask)
    assert h.stats["fallback"] == 4, h.stats
    assert F.cosine_similarity(g1.flatten().double(), g0.flatten().double(), dim=0).item() > 0.9999
    assert relrms(o1, o0) < 2e-2 and relrms(g1, g0) < 2e-2, (relrms(o1, o0), relrms(g1, g0))
    trunk.set_kernel_backend(None)                               # rebuilds every block's row_drop: the lever must re-patch it
    o1b, g1b = _trunk_step(trunk, z, mask, gout)                 # (a missed patch would add the residual twice)
    assert torch.equal(o1b, o1) and torch.equal(g1b, g1)
    tm.disable(trunk)
    assert not sg.diff(sg.snapshot(), ref)
    assert not hasattr(trunk.blocks[0].tri_mul_out, "_ef2_trimul_orig_forward") and "_ef2_trimul_handle" not in trunk.__dict__
    o2, g2 = _trunk_step(trunk, z, mask, gout)
    assert torch.equal(o2, o0) and torch.equal(g2, g0)


@cuda
def test_fused_refuses_agk_trimul_blocks_and_composes_with_agk_transition():
    import ef2_autograd_kernels as agk
    trunk = _trunk(3, seed=20); N = 64; z, mask, gout = make_inputs(N, 1)
    o0, g0 = _trunk_step(trunk, z, mask, gout)
    agk.enable(trunk, trimul="bmm2", transition="refround_lean", checkpoint="none")
    with raises(RuntimeError):
        tm.enable(trunk, variant="fused")
    agk.disable(trunk)
    agk.enable(trunk, trimul=None, transition="refround_lean", checkpoint="ckpt:1")
    h = tm.enable(trunk, variant="fused")
    o1, g1 = _trunk_step(trunk, z, mask, gout)
    assert h.stats["served"] >= 6, h.stats                        # 3 blocks x 2 (+ recompute under checkpointing)
    assert torch.isfinite(g1).all()
    assert F.cosine_similarity(g1.flatten().double(), g0.flatten().double(), dim=0).item() > 0.9999
    o1b, g1b = _trunk_step(trunk, z, mask, gout)                  # deterministic run to run
    assert torch.equal(o1b, o1) and torch.equal(g1b, g1)
    tm.disable(trunk); agk.disable(trunk)
    o2, g2 = _trunk_step(trunk, z, mask, gout)
    assert torch.equal(o2, o0) and torch.equal(g2, g0)



@cuda
def test_fused_inside_compiled_block_with_checkpoint():
    """Regression: upstream torch.compile's every PairUpdateBlock.forward and the trunk checkpoints blocks
    (use_reentrant=False).  The lever must be opaque to dynamo: same result as the eager block, twice through a checkpoint
    (first forward + recompute) without torch.utils.checkpoint's 'different number of tensors saved' error, and bitwise
    run-to-run."""
    from torch.utils.checkpoint import checkpoint
    N = 129
    blk = make_block(0); z, mask, gout = make_inputs(N, masked_tail=5)
    tm.enable(blk, variant="fused")
    try:
        def run(b):
            zz = z.detach().clone().requires_grad_(True)
            with torch.enable_grad(), AMP:
                o = checkpoint(lambda x: b(x, pair_attention_mask=mask), zz, use_reentrant=False)
                o = checkpoint(lambda x: b(x, pair_attention_mask=mask), o, use_reentrant=False)
            o.backward(gout)
            return o.detach(), zz.grad.detach()
        o_e, g_e = run(blk)                                                      # eager block, lever on
        orig_fwd = blk.forward
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        blk.forward = torch.compile(blk.forward)                                 # upstream's _apply_torch_compile, per instance
        try:
            o_c1, g_c1 = run(blk)
            o_c2, g_c2 = run(blk)
        finally:
            blk.forward = orig_fwd
        assert torch.equal(o_c1, o_c2) and torch.equal(g_c1, g_c2), "compiled block + lever not deterministic run-to-run"
        # the tri-muls run the same eager kernels in both; only the transition is compiled (Inductor may re-round it): fast-class agreement
        e_o = relrms(o_c1, o_e); e_g = relrms(g_c1, g_e)
        assert e_o < 5e-3 and e_g < 1e-2, (e_o, e_g)
        h = blk.__dict__["_ef2_trimul_handle"]
        assert h.stats["served"] >= 12 and h.stats["fallback"] == 0, h.stats     # 2 tri-muls x 2 blocks x (fwd + recompute) x 3 runs, all served
    finally:
        tm.disable(blk)
        torch._dynamo.reset()


@cuda
def test_fused_late_trajectory_statistics_within_stock_error_class():
    """Regression (cd45 @ step 90): late-trajectory pair states are large (|z| ~ 500-2000, row-std 15-150, i.e. bf16 ulp
    of the residual stream 2-16) and the loss gradient arriving at the block is tiny (~1e-7 rms, 1e-4 max).  Parity with an fp64 gold
    of the stock math must hold there exactly as at unit scale: fused's d(pair) and output errors <= 1.25x the stock module's own."""
    N = 257
    blk = make_block(3)
    g = torch.Generator(device="cuda").manual_seed(7)
    row_scale = (torch.rand(1, N, N, 1, device="cuda", generator=g) * 135 + 15)              # per-row std 15..150
    z = (torch.randn(1, N, N, D, device="cuda", generator=g) * row_scale + torch.randn(1, N, N, 1, device="cuda", generator=g) * 5).to(torch.bfloat16)
    mask = torch.ones(1, N, N, device="cuda")
    gout = (torch.randn(1, N, N, D, device="cuda", generator=g) * 3e-7 *
            torch.exp(torch.randn(1, N, N, 1, device="cuda", generator=g))).to(torch.bfloat16)  # heavy-tailed tiny gradient
    o64, g64 = gold(blk, z, mask, gout)
    o_s, g_s = run_stock(blk, z, mask, gout)
    o_f, g_f = run_fused(blk, z, mask, gout)
    es_o, es_g = relrms(o_s, o64), relrms(g_s, g64)
    ef_o, ef_g = relrms(o_f, o64), relrms(g_f, g64)
    assert ef_o <= 1.25 * es_o and ef_g <= 1.25 * es_g, (ef_o, es_o, ef_g, es_g)
    cos = torch.nn.functional.cosine_similarity(g_f.double().flatten(), g_s.double().flatten(), dim=0).item()
    assert cos > 0.9999, cos

if __name__ == "__main__":                       # pytest-free: python test_ef2_trimul.py [-k substr]
    import traceback
    sel = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    n_pass = n_fail = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn) or sel not in name:
            continue
        if getattr(fn, "_skip", False):
            print("SKIP", name); continue
        combos = [dict()]
        if hasattr(fn, "_param"):
            names = [n.strip() for n in fn._param[0].split(",")]
            combos = [dict(zip(names, v if isinstance(v, (tuple, list)) else (v,))) for v in fn._param[1]]
        for kw in combos:
            try:
                fn(**kw); n_pass += 1; print("PASS", name, kw, flush=True)
            except Exception as e:  # noqa: BLE001
                n_fail += 1; print("FAIL", name, kw, type(e).__name__, str(e)[:400], flush=True); traceback.print_exc()
            torch.cuda.empty_cache()
    print(f"RESULT passed={n_pass} failed={n_fail}")
    sys.exit(1 if n_fail else 0)
