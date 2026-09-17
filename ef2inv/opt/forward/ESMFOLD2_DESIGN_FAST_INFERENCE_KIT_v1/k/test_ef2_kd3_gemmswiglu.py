"""Unit tests for ef2_kd3_gemmswiglu (one CUDA device; the Biohub transformers fork importable).

    pytest -q k/test_ef2_kd3_gemmswiglu.py        (module-level test_* functions; the kit's pytest-free runner convention)

The claims under test: the fused W12-projection + SwiGLU kernel reproduces K-D3's chain (cuBLAS lin -> _swiglu_refround_fwd_kernel) on
pair rows of the shipped width (d = 256, h = 1024) to within one bf16 ulp under every entry of the per-capability table (bit-identical
is what the cards with an entry give; one ulp is the class the docstring claims), odd row counts included; a whole K-D3 lean
transition forward + backward with the hook bound gives K-D3's output and input gradient; engage / disable bind and restore agk's
hook exactly (idempotent) and the counters count; the non-lean row and an entry the kernel does not take fall through to K-D3's
chain by name; a capability without an entry engages nothing and says so (state stepped_aside, reason cc_untuned:sm_NN); agk's
F9-counted entry point (transition_refround) reaches the hook in grad and no-grad mode.
"""
import os, sys, pytest, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import transformers.models.esmfold2.modeling_esmfold2_common as C
import ef2_autograd_kernels as agk
import ef2_kd3_gemmswiglu as G

cuda = pytest.mark.skipif(not torch.cuda.is_available() or not G.KERNELS_OK or not agk._KD3_OK, reason="CUDA + Triton kernels required")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CC = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
ENTRY = G.cfg_for(CC)[0] if CC else None                # this card's entry (None: the tests pass sm_80's tile explicitly and say so)
CFGS = sorted({G.tile_word(c): c for c in G._CFG_BY_CC.values()}.items())
D, H = 256, 1024


def _ulps(a, b):
    m = torch.maximum(a.float().abs(), b.float().abs()).clamp(min=2.0 ** -126)
    return (a.float() - b.float()).abs() / torch.pow(2.0, torch.floor(torch.log2(m)) - 7)


def _rows(M, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(M, D, device=dev, generator=g).to(torch.bfloat16)
    w12 = (torch.randn(2 * H, D, device=dev, generator=g) / D ** 0.5).to(torch.bfloat16).contiguous()
    w3 = (torch.randn(D, H, device=dev, generator=g) / H ** 0.5).to(torch.bfloat16).contiguous()
    gam = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).float(); bet = (0.1 * torch.randn(D, device=dev, generator=g)).float()
    xhat, mean, rstd = agk._FLR._ln_fwd(x, gam, bet, C._EPS, agk._FLR._layout_to_int("bijd->bijd"), M, D)
    return x, xhat, w12, w3, gam, bet


def _kd3_hidden(xhat, w12):
    lin = F.linear(xhat, w12); M = lin.shape[0]; h = w12.shape[0] // 2
    hid = torch.empty((M, h), dtype=lin.dtype, device=dev)
    BLOCK = agk.triton.next_power_of_2(h)
    agk._swiglu_refround_fwd_kernel[(M,)](lin, hid, h, lin.stride(0), hid.stride(0), BLOCK=BLOCK, num_warps=4 if BLOCK <= 1024 else 8)
    return hid


def _weights(w12, w3, gam, bet, lean):
    return {"LN_W32": gam, "LN_B32": bet, "W12": w12, "W3": w3, "lean": lean}


@cuda
@pytest.mark.parametrize("tile,cfg", CFGS)
@pytest.mark.parametrize("M,seed", [(4099, 1), (777, 2), (128 * 37, 3), (5, 4)])
def test_fused_forward_matches_kd3_chain_to_one_ulp(tile, cfg, M, seed):
    x, xhat, w12, w3, gam, bet = _rows(M, seed)
    ref = _kd3_hidden(xhat, w12)
    out = G.w12_swiglu_fwd(xhat, w12, cfg)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    u = _ulps(out, ref)
    assert float(u.max()) <= 1.01, (tile, float(u.max()), int((u > 1.01).sum()))
    assert float((out != ref).float().mean()) < 1e-3          # 0 on the cards with an entry; a different MMA order may flip a rare rounding


@cuda
def test_transition_forward_backward_equal_kd3_with_the_hook_bound():
    M = 3001
    x, xhat, w12, w3, gam, bet = _rows(M, 7)
    w = _weights(w12, w3, gam, bet, lean=True)
    x1 = x.clone().view(1, M, 1, D).requires_grad_(True); x2 = x.clone().view(1, M, 1, D).requires_grad_(True)
    G.reset()
    assert agk._kd3_w12_swiglu is None and not G.live()
    ref = agk.TransitionRefround.apply(x1, w)                         # K-D3's own chain
    d = G.engage(cfg=ENTRY or G._CFG_SM80)
    try:
        assert d["state"] == "on" and G.live() and agk._kd3_w12_swiglu is G.hidden_for and d["tile"] == G.tile_word(ENTRY or G._CFG_SM80)
        assert G.engage(cfg=ENTRY or G._CFG_SM80)["state"] == "on" and agk._kd3_w12_swiglu is G.hidden_for      # idempotent
        out = agk.TransitionRefround.apply(x2, w)
        gr = torch.Generator(device=dev).manual_seed(8)
        go = torch.randn(ref.shape, device=dev, generator=gr).to(ref.dtype)
        ref.backward(go); out.backward(go)
        assert G.stats()["served"] == 1 and G.stats()["fallback"] == {}
    finally:
        G.disable()
    assert agk._kd3_w12_swiglu is None and not G.live() and G.describe()["state"] == "off"
    assert float(_ulps(out, ref).max()) <= 1.01 and float(_ulps(x2.grad, x1.grad).max()) <= 2.01


@cuda
def test_non_lean_row_and_an_entry_not_taken_fall_through_by_name():
    M = 555
    x, xhat, w12, w3, gam, bet = _rows(M, 9)
    G.reset(); G.engage(cfg=ENTRY or G._CFG_SM80)
    try:
        xr = x.clone().view(1, M, 1, D).requires_grad_(True)
        out = agk.TransitionRefround.apply(xr, _weights(w12, w3, gam, bet, lean=False))    # the non-lean row: the hook is not read (lean only)
        out.sum().backward()
        assert G.stats()["served"] == 0 and G.stats()["fallback"] == {}
        w_odd = _weights(w12[: 2 * (H - 8)].contiguous(), w3[:, : H - 8].contiguous(), gam, bet, lean=True)   # h = 1016: not a multiple of the tile's bn
        G.disable(); ref = agk.TransitionRefround.apply(x.clone().view(1, M, 1, D), w_odd); G.engage(cfg=ENTRY or G._CFG_SM80)
        out = agk.TransitionRefround.apply(x.clone().view(1, M, 1, D), w_odd)
        assert torch.equal(out, ref) and G.stats()["served"] == 0 and G.stats()["fallback"] == {"h": 1}   # fell through to K-D3's chain, counted by name
    finally:
        G.disable(); G.reset()


@cuda
def test_a_capability_without_an_entry_steps_aside_by_name():
    G.reset()
    d = G.engage(cc=(12, 0))
    try:
        assert d["state"] == "stepped_aside" and d["reason"] == "cc_untuned:sm_120" and d["tile"] is None and not G.live() and agk._kd3_w12_swiglu is None
        assert G.cfg_for((12, 0)) == (None, "cc_untuned:sm_120") and G.cfg_for((8, 0))[0] == G._CFG_SM80
        if ENTRY is None:
            assert G.engage()["state"] == "stepped_aside" and G.describe()["reason"] == f"cc_untuned:sm_{CC[0]}{CC[1]}"
        else:
            assert G.engage()["state"] == "on" and G.describe()["cc"] == f"sm_{CC[0]}{CC[1]}"
    finally:
        G.disable(); G.reset()


@cuda
def test_counted_entry_point_reaches_the_hook_in_grad_and_no_grad_mode():
    """agk.transition_refround (the kit's F9-counted entry point) serves through the hook in grad mode, in no-grad mode and whether the lever
    engaged before or after the weight entry was cached (the hook is read per call)."""
    M = 300
    x, xhat, w12, w3, gam, bet = _rows(M, 11)
    blk = C.PairUpdateBlock(d_pair=D, expansion_ratio=4).to(dev).eval().requires_grad_(False)     # the fork's pair block; its transition is what K-D3 serves
    tr = blk.pair_transition
    cache = {}
    xr = x.clone().view(1, M, 1, D).requires_grad_(True)
    G.reset()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y0 = agk.transition_refround(x.clone().view(1, M, 1, D).requires_grad_(True), tr, cache, lean=True)   # cached BEFORE the lever engaged: K-D3's chain
    G.engage(cfg=ENTRY or G._CFG_SM80)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y = agk.transition_refround(xr, tr, cache, lean=True)
            assert G.stats()["served"] == 1 and y.shape == xr.shape and float(_ulps(y, y0).max()) <= 1.01
            with torch.no_grad():
                agk.transition_refround(x.view(1, M, 1, D), tr, cache, lean=True)
                assert G.stats()["served"] == 2
    finally:
        G.disable(); G.reset()
