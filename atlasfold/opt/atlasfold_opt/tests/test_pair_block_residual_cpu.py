"""pair_block_residual — the pair-to-pair residual adds folded into the fused epilogues (CPU: no kernel launches): row membership and order
(after the two fused levers it requires), the per-call refusal words on CPU tensors, install's named skip without the fused levers, the
served path's control flow on a stub core (in-place tri-attn residual, transition residual, the in-place mask multiply == the stock forward's
values), the degraded route (a declined sub-call runs its stock statement), and the fused levers' census helpers."""
import types
import pytest

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import pair_block_residual as PBR, pair_cells as PC, triatt_block as TB, pair_transition_fused as HP


def test_row_membership_order_and_registry():
    fast = modes.MODES["fast"]
    assert "pair_block_residual" in fast
    i = fast.index("pair_block_residual")
    assert i > fast.index("triatt_block") and i > fast.index("pair_transition") and i > fast.index("triattn_core")   # installed after every lever it requires
    assert "pair_block_residual" not in modes.MODES["exact"] and "pair_block_residual" not in modes.PLANNED.get("fast", [])
    row = registry.LEVERS["pair_block_residual"]
    assert row["cls"] == "fast" and row["strategy"] == PBR.NAME and row["module"] == "atlasfold_opt.hooks.pair_block_residual"
    assert "PairBlock.forward" in row["site"] and {"below_min_tokens", "dtype:", "c:", "backend_torch", "no_calls"} <= set(row["expected"])
    from atlasfold_opt.hooks import installers
    assert installers()["pair_block_residual"] is PBR.install


def _block(torch, **over):
    tu = pytest.importorskip("atlasfold.model.network.block")
    b = tu.PairBlock(channel_s=32, channel_z=128, num_heads_attn=4, num_heads_tri_attn=4).eval()
    for k, v in over.items():
        setattr(b, k, v)
    return b


def test_refusal_words_on_cpu():
    torch = pytest.importorskip("torch")
    b = _block(torch)
    N = 512
    z = torch.zeros(1, N, N, 128, dtype=torch.bfloat16); pm = torch.ones(1, N, N, dtype=torch.bool)
    assert PBR.refusal(b, z, pm, "torch") == "backend_torch"
    assert PBR.refusal(b, z.float(), pm, "cuequiv") == "dtype:float32"
    assert PBR.refusal(b, z[:, :256, :256], pm[:, :256, :256], "cuequiv") == "below_min_tokens"
    assert PBR.refusal(b, torch.zeros(1, N, N, 64, dtype=torch.bfloat16), pm, "cuequiv") == "c:64"
    assert PBR.refusal(b, z[0], pm, "cuequiv") == "rank"
    assert PBR.refusal(b, z, pm, "cuequiv") == "cpu"                                       # everything else eligible: the device is the last word on CPU
    b.train(); assert PBR.refusal(b, z, pm, "cuequiv") == "training"; b.eval()
    b2 = _block(torch); b2.pair_to_pair = False
    assert PBR.refusal(b2, z, pm, "cuequiv") == "no_pair_to_pair"


def test_install_requires_the_fused_levers():
    pytest.importorskip("torch"); pytest.importorskip("opt_core")
    try:
        import atlasfold.model.network.block  # noqa: F401
    except Exception:
        pytest.skip("stock atlasfold not importable here")
    ins = PBR.install("fast", "atlasfold-opt", {"mode": "fast", "det": 0})
    assert ins.applied is False and ins.reason in ("requires:triatt_block", "requires:pair_transition")


def test_served_path_equals_the_stock_forward_on_a_stub_core(monkeypatch):
    """The served control flow with the kernels stubbed by exact torch statements: in-place tri-attn residuals, the transition residual, the
    in-place mask — the same (s, z) values as the stock forward, z updated IN PLACE, census on the three ledgers."""
    torch = pytest.importorskip("torch"); pytest.importorskip("opt_core")
    blk = pytest.importorskip("atlasfold.model.network.block")
    from opt_core.attn import pair_fused as PF
    from opt_core.counters import Ledger
    torch.manual_seed(0)
    b = _block(torch)
    N = 16
    s0 = torch.randn(1, N, 32); z0 = torch.randn(1, N, N, 128); mask = torch.ones(1, N, dtype=torch.bool); mask[0, -3:] = False
    pm = mask[..., :, None] & mask[..., None, :]
    with torch.no_grad():
        s_ref, z_ref = blk.PairBlock.forward(b, s0.clone(), z0.clone(), mask, pm, kernel_backend="torch")
    # stub the two core entry points with the stock statements (fp32, exact) and lift every refusal
    def tri_stub(z4, W, km, *, ending, residual, impl, core, ln):
        mod = W
        u = mod(z4, tri_stub.pm, kernel_backend="torch")
        if residual:
            z4 += u; return z4
        return u
    tri_stub.pm = pm
    monkeypatch.setattr(TB, "weights", lambda m: m)                       # hand the module itself through as 'W'
    monkeypatch.setattr(TB, "key_mask", lambda m3, ending, ledger=None: m3)
    monkeypatch.setattr(PC, "core_pick", lambda m, z4, e: PC.CORE)
    monkeypatch.setattr(PF, "tri_attn_block", tri_stub)
    monkeypatch.setattr(HP, "fused", lambda module, x, residual: x + module(x) if residual else module(x))   # the transition lever's served call, stubbed above the provider (fp32 statements)
    monkeypatch.setattr(PBR, "refusal", lambda *a, **k: None)
    Lb, Lt, Lp = Ledger("b"), Ledger("t"), Ledger(PBR.NAME)
    monkeypatch.setitem(TB.STATE, "ledger", Lb); monkeypatch.setitem(HP.STATE, "ledger", Lt)
    # install the served forward on CPU (the fused levers' qualnames stubbed in, the kernels stubbed above)
    monkeypatch.setattr(blk.PairBlock, "forward", blk.PairBlock.forward)   # keep stock for the reference above
    tu = __import__("atlasfold.model.network.primitives.triangle_update", fromlist=["x"])
    trm = __import__("atlasfold.model.network.primitives.transition", fromlist=["x"])
    for cls_name, _ in TB.CLASSES:                                          # pretend the fused levers are installed (qualname check)
        f = getattr(tu, cls_name).forward
        monkeypatch.setattr(getattr(tu, cls_name), "forward", _renamed(f, f"{cls_name}.forward[atlasfold_opt:triatt_block]"))
    monkeypatch.setattr(trm.Transition, "forward", _renamed(trm.Transition.forward, "Transition.forward[atlasfold_opt:pair_transition]"))
    stock = blk.PairBlock.forward
    ins = PBR.install("fast", "atlasfold-opt", {"mode": "fast", "det": 0})
    try:
        assert ins.applied, ins.reason
        z_in = z0.clone()
        with torch.no_grad():
            s_out, z_out = b(s0.clone(), z_in, mask, pm, kernel_backend="cuequiv")
        assert torch.equal(z_out, z_ref) and torch.equal(s_out, s_ref)       # the stock forward's values exactly (fp32 stub statements)
        L = ins.facts["ledger"]
        assert L.served == 1 and L.get("triatt_res") == 2 and L.get("trans_res") == 1 and L.get("mask_inplace") == 1 and L.get("degraded") == 0
        assert Lb.served == 2 and Lb.get("feed_strided") == 1 and Lt.served == 1          # the fused levers' census counts their statements
        line = ins.lines[0]()
        assert line.startswith("[atlasfold-opt] LEVER name=LOCAL.atlasfold.pair_block_residual state=on ") and " triatt_res=2" in line and " trans_res=1" in line
        # degraded route: the tri-attn core declines by name -> the stock statement for that sub-call, same values, counted
        monkeypatch.setattr(PF, "tri_attn_block", _raiser(PF.Unsupported("no-cell:test")))
        with torch.no_grad():
            s2, z2 = b(s0.clone(), z0.clone(), mask, pm, kernel_backend="cuequiv")
        assert torch.equal(z2, z_ref) and torch.equal(s2, s_ref)
        assert L.get("degraded") == 2 and not L.errors and ins.gates[0]().ok
        # a kernel that RAISES: the stock statement serves the sub-call, the gate refuses at exit (fail-loud)
        monkeypatch.setattr(HP, "fused", _raiser(RuntimeError("boom")))          # the transition lever's served call raises
        with torch.no_grad():
            s3, z3 = b(s0.clone(), z0.clone(), mask, pm, kernel_backend="cuequiv")
        assert torch.equal(z3, z_ref) and L.errors and not ins.gates[0]().ok
    finally:
        blk.PairBlock.forward = stock


def _renamed(f, qualname):
    def g(*a, **k): return f(*a, **k)
    g.__qualname__ = qualname; g.__wrapped_stock__ = f
    return g


def _raiser(exc):
    def f(*a, **k): raise exc
    return f
