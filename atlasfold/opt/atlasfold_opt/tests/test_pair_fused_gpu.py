"""GPU tests (CUDA required; no weights) of the fused pair levers' SERVED paths against the stock statements: random bf16-representable weights in
stock TriangleAttentionStartingNode / EndingNode(128, 4) and Transition(128, 4); reference = the stock forward in fp32 (kernel_backend='torch') from
the same weights; the class's own floor = the stock forward in bf16 (what the .bfloat16()-cast trunk runs without the lever); served = the hook's
path (the installed wrapper, kernel_backend='cuequiv').  The served max-abs error to the fp32 reference must stay within 2x the stock-bf16 error
(+ 1 bf16 ulp of the output scale) — the tolerance class of a bf16 statement; a wrong q|k|v / a|b split or head order is O(1) off.
B in {1, 2}, L in {512, 576} (the levers' floor is 512 tokens; 384 is asserted to route below_min_tokens); all-ones and random pair masks."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
DEV = torch.device("cuda", torch.cuda.current_device())


def _stock():
    tu = pytest.importorskip("atlasfold.model.network.primitives.triangle_update")
    tr = pytest.importorskip("atlasfold.model.network.primitives.transition")
    pytest.importorskip("opt_core.attn.pair_fused")
    return tu, tr


def _randomize(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            base = 1.0 if (name.endswith("weight") and p.dim() == 1) else 0.0
            p.copy_((base + torch.randn(p.shape, generator=g) * 0.08).to(torch.bfloat16).float())
    return m.eval()


@pytest.fixture
def installed():
    """Install both levers for the test and restore the stock attributes after it (the wrappers are process-wide rebindings)."""
    tu, tr = _stock()
    from atlasfold_opt import TAG
    from atlasfold_opt.hooks import triatt_block as HT, pair_transition_fused as HP
    keep = {c: getattr(tu, c).forward for c, _ in HT.CLASSES}; keep_t = tr.Transition.forward
    ins_t = HT.install("fast", TAG, {}); ins_p = HP.install("fast", TAG, {})
    assert ins_t.applied and ins_p.applied, (ins_t.reason, ins_p.reason)
    try:
        yield {"triatt_block": ins_t, "pair_transition": ins_p, "tu": tu, "tr": tr}
    finally:
        for c, f in keep.items():
            setattr(getattr(tu, c), "forward", f)
        tr.Transition.forward = keep_t


def _ulp_bf16(x: float) -> float:
    import math
    return 2.0 ** (math.floor(math.log2(max(abs(x), 1e-30))) - 7)


@pytest.mark.parametrize("B,L", [(1, 512), (2, 576)])
def test_triatt_block_served_matches_stock(installed, B, L):
    tu = installed["tu"]; ledger = installed["triatt_block"].facts["ledger"]
    assert installed["triatt_block"].facts["served"] is True, installed["triatt_block"].facts          # this GPU's cells serve (H100/A100 rows, else safe settings)
    g = torch.Generator().manual_seed(100 + L)
    z32 = torch.randn(B, L, L, 128, generator=g).to(torch.bfloat16).float().to(DEV)                   # bf16-representable input: the reference sees exactly what the kernels see
    masks = {"ones": torch.ones(B, L, L, dtype=torch.bool, device=DEV), "random": (torch.rand(B, L, L, generator=g) > 0.1).to(DEV)}
    for cls_name, ending in (("TriangleAttentionStartingNode", False), ("TriangleAttentionEndingNode", True)):
        m32 = _randomize(getattr(tu, cls_name)(128, 4), seed=5 + int(ending)).to(DEV)
        m16 = getattr(tu, cls_name)(128, 4).to(DEV).eval(); m16.load_state_dict(m32.state_dict()); m16 = m16.to(torch.bfloat16)
        stock_fwd = type(m32).forward.__wrapped_stock__                                              # the stock statement (the wrapper keeps it)
        for mk_name, mk in masks.items():
            with torch.no_grad():
                ref = stock_fwd(m32, z32.clone(), mk, kernel_backend="torch").float()
                floor = stock_fwd(m16, z32.to(torch.bfloat16), mk, kernel_backend="torch").float()
                s0 = ledger.served
                got = m16(z32.to(torch.bfloat16), mk, kernel_backend="cuequiv")
                assert ledger.served == s0 + 1, (cls_name, mk_name, ledger.fields())               # served, not a counted fallback
                assert got.dtype == torch.bfloat16 and tuple(got.shape) == (B, L, L, 128)
                e_floor = float((floor - ref).abs().max()); e_got = float((got.float() - ref).abs().max()); scale = float(ref.abs().max())
            assert e_got <= 2.0 * e_floor + _ulp_bf16(scale), (cls_name, mk_name, B, L, e_got, e_floor, scale)
            assert e_got <= 0.05 * scale, (cls_name, mk_name, e_got, scale)                          # and small in absolute terms (a split/head-order defect is O(scale))
    f = ledger.fields()
    assert not f["errors"] and set(f["fallback_by"]) <= {"below_min_tokens"}, f
    assert int(ledger.get("feed_strided", 0)) >= 2 and int(ledger.get("mask_copies", 0)) >= 2 and int(ledger.get("copies", 0)) == 0, ledger.facts()


def test_triatt_block_routes_below_floor_and_3d(installed):
    tu = installed["tu"]; ledger = installed["triatt_block"].facts["ledger"]
    m16 = _randomize(tu.TriangleAttentionEndingNode(128, 4), seed=9).to(DEV).to(torch.bfloat16)
    z = torch.randn(384, 384, 128, device=DEV).to(torch.bfloat16); mk = torch.ones(384, 384, dtype=torch.bool, device=DEV)
    fb0 = dict(ledger.fallbacks); s0 = ledger.served
    with torch.no_grad():
        out = m16(z, mk, kernel_backend="cuequiv")
    assert tuple(out.shape) == (384, 384, 128) and ledger.served == s0 and ledger.fallbacks.get("below_min_tokens", 0) == fb0.get("below_min_tokens", 0) + 1
    z = torch.randn(512, 512, 128, device=DEV).to(torch.bfloat16); mk = torch.ones(512, 512, dtype=torch.bool, device=DEV)
    with torch.no_grad():
        out3 = m16(z, mk, kernel_backend="cuequiv")                                                  # 3-D z is served (u[0] of the 4-D statement)
        ref3 = type(m16).forward.__wrapped_stock__(m16.float(), z.float(), mk, kernel_backend="torch")
    assert tuple(out3.shape) == (512, 512, 128) and ledger.served == s0 + 1
    assert float((out3.float() - ref3).abs().max()) <= 0.05 * float(ref3.abs().max())
    assert installed["triatt_block"].gates[0]().ok, ledger.fields()


@pytest.mark.parametrize("B,L", [(1, 512), (2, 576)])
def test_pair_transition_served_matches_stock(installed, B, L):
    tr = installed["tr"]; ledger = installed["pair_transition"].facts["ledger"]
    assert installed["pair_transition"].facts["word"] == "fast", installed["pair_transition"].facts
    m32 = _randomize(tr.Transition(128, 4), seed=11).to(DEV)
    m16 = tr.Transition(128, 4).to(DEV).eval(); m16.load_state_dict(m32.state_dict()); m16 = m16.to(torch.bfloat16)
    x32 = torch.randn(B, L, L, 128, generator=torch.Generator().manual_seed(200 + L)).to(torch.bfloat16).float().to(DEV)
    with torch.no_grad():
        ref = torch.nn.Sequential.forward(m32, x32)
        floor = torch.nn.Sequential.forward(m16, x32.to(torch.bfloat16)).float()
        s0 = ledger.served
        got = m16(x32.to(torch.bfloat16))
        assert ledger.served == s0 + 1, ledger.fields()
        got_nc = m16(x32.to(torch.bfloat16).transpose(1, 2))                                         # a strided x: served through ONE counted copy
        assert ledger.served == s0 + 2 and int(ledger.get("copies", 0)) >= 1, ledger.facts()
        e_floor = float((floor - ref).abs().max()); e_got = float((got.float() - ref).abs().max()); scale = float(ref.abs().max())
        e_nc = float((got_nc.float() - ref.transpose(1, 2)).abs().max())
    assert got.dtype == torch.bfloat16 and tuple(got.shape) == (B, L, L, 128)
    assert e_got <= 2.0 * e_floor + _ulp_bf16(scale) and e_nc <= 2.0 * e_floor + _ulp_bf16(scale), (B, L, e_got, e_nc, e_floor, scale)
    assert e_got <= 0.05 * scale, (e_got, scale)
    small = torch.randn(1, 256, 256, 128, device=DEV).to(torch.bfloat16)
    with torch.no_grad():
        out_small = m16(small); ref_small = torch.nn.Sequential.forward(m16, small)
    assert not ledger.fields()["errors"], ledger.fields()                                             # a 256-token call: served by the cell's row or the statement by name (`stock_row:`), never an error
    assert float((out_small.float() - ref_small.float()).abs().max()) <= 0.05 * float(ref_small.float().abs().max()) + 1e-3
    assert installed["pair_transition"].gates[0]().ok, ledger.fields()
