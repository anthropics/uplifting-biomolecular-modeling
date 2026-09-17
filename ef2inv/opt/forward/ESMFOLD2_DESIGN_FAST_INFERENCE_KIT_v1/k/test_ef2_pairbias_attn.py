"""Unit tests for ef2_pairbias_attn (one GPU; random-initialised modules, no weights).

    python k/test_ef2_pairbias_attn.py          # standalone: last line "RESULT passed=N failed=M"
    pytest -q k/test_ef2_pairbias_attn.py       # the same tests under pytest
Runner convention of k/run_tests_nopytest.py: module-level test_* functions; pytest.mark.skipif is the only pytest feature used.
"""
import os
import sys
import types

try:
    import pytest
except ImportError:                      # images without pytest: the same tiny stand-in k/run_tests_nopytest.py registers
    import importlib.machinery
    pytest = types.ModuleType("pytest")
    pytest.__spec__ = importlib.machinery.ModuleSpec("pytest", None)

    class _Mark:
        def skipif(self, cond, reason=""):
            def deco(f):
                f._skip = bool(cond) or getattr(f, "_skip", False)
                f._skip_reason = reason if cond else getattr(f, "_skip_reason", "")
                return f
            return deco
    pytest.mark = _Mark()
    sys.modules["pytest"] = pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ef2_pairbias_attn as pba  # noqa: E402
from transformers.models.esmfold2 import modeling_esmfold2_common as C  # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
D_MODEL, D_PAIR, HEADS = 768, 256, 16          # the structure module's token transformer (ESMFold2-Experimental-Fast: 12 such blocks)


def _tt(num_blocks=3, seed=0):
    torch.manual_seed(seed)
    tt = C.DiffusionTransformer(d_model=D_MODEL, d_pair=D_PAIR, num_heads=HEADS, num_blocks=num_blocks).cuda().float().eval()
    for blk in tt.attn_blocks:                 # trained-like scales so the pair bias and the gates matter numerically
        torch.nn.init.normal_(blk.pair_bias_proj.weight, std=0.5)
        torch.nn.init.normal_(blk.out_gate.weight, std=0.05)
    return tt.requires_grad_(False)


def _inputs(n, steps, seed=1, bsz=1, masked=()):
    g = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(bsz, n, n, D_PAIR, device="cuda", generator=g) * 2.0
    a = [torch.randn(bsz, n, D_MODEL, device="cuda", generator=g) for _ in range(steps)]
    s = [torch.randn(bsz, n, D_MODEL, device="cuda", generator=g) for _ in range(steps)]
    mask = None
    if masked:
        mask = torch.ones(bsz, n, device="cuda", dtype=torch.bool)
        mask[:, list(masked)] = False
    return z, a, s, mask


def _run(tt, z, a, s, mask):
    outs = []
    with torch.no_grad():
        for ai, si in zip(a, s):
            outs.append(tt(ai, si, z, 0.0, attention_mask=mask)[0])
    return outs


@needs_cuda
def test_hoist_is_bitwise_on_the_reference_path():
    """variant='hoist' == stock bit for bit (backend None = the eager path 'cuequivariance' takes <= 750 queries), odd length, key mask,
    6 sampling steps sharing one z; the cache serves blocks*(steps-1) calls and is dropped with the scope."""
    tt = _tt(3)
    z, a, s, mask = _inputs(137, 6, masked=(0, 5, 136))
    ref = _run(tt, z, a, s, mask)
    ref_nomask = _run(tt, z, a, s, None)
    st = pba.enable(tt, "hoist")
    with pba.bias_scope(tt):
        got = _run(tt, z, a, s, mask)
        assert all(blk._pba_cache is not None for blk in tt.attn_blocks)
    assert all(blk._pba_cache is None for blk in tt.attn_blocks), "cache must be dropped when the scope exits"
    assert all(torch.equal(r, g) for r, g in zip(ref, got)), [float((r - g).abs().max()) for r, g in zip(ref, got)]
    assert st.stats["computed"] == 3 and st.stats["served"] == 3 * 5, st.stats
    got_unscoped = _run(tt, z, a, s, None)                       # outside a scope: computed every call, still stock's arithmetic
    assert all(torch.equal(r, g) for r, g in zip(ref_nomask, got_unscoped))
    assert st.stats["unscoped"] == 3 * 6, st.stats
    pba.disable(tt)
    assert not hasattr(tt, "_pba_state") and not hasattr(tt.attn_blocks[0], "_pba_orig_forward")
    assert all(torch.equal(r, g) for r, g in zip(ref, _run(tt, z, a, s, mask)))


@needs_cuda
def test_hoist_tracks_in_place_changes_of_z():
    """A hit needs the same tensor object at the same version: an in-place update of z inside the scope recomputes (stock semantics)."""
    tt = _tt(2)
    z, a, s, mask = _inputs(64, 2)
    pba.enable(tt, "hoist")
    with pba.bias_scope(tt), torch.no_grad():
        o1 = tt(a[0], s[0], z, 0.0)[0]
        z.mul_(0.5)
        o2 = tt(a[0], s[0], z, 0.0)[0]
        assert tt._pba_state.stats["computed"] == 4, tt._pba_state.stats
    pba.disable(tt)
    with torch.no_grad():
        r2 = tt(a[0], s[0], z, 0.0)[0]
    assert torch.equal(o2, r2) and not torch.equal(o1, o2)


@needs_cuda
@pytest.mark.skipif(not C.TRITON_KERNELS_AVAILABLE, reason="the fork's fused Triton kernels are not importable")
def test_hoist_is_bitwise_under_backend_fused():
    """backend 'fused' (fast/big outside their kernels): stock = fused_pair_bias kernel per call + SDPA; hoist caches the kernel's bias."""
    tt = _tt(3)
    tt.set_kernel_backend("fused")
    z, a, s, mask = _inputs(150, 5, masked=(3, 149))
    ref = _run(tt, z, a, s, mask)
    st = pba.enable(tt, "hoist")
    with pba.bias_scope(tt):
        got = _run(tt, z, a, s, mask)
    assert all(torch.equal(r, g) for r, g in zip(ref, got)), [float((r - g).abs().max()) for r, g in zip(ref, got)]
    assert st.stats["computed"] == 3 and st.stats["served"] == 12, st.stats
    pba.disable(tt); tt.set_kernel_backend(None)


def _gold_core(q, k, v, bias_bhqk, gate_raw, key_mask, scale):
    """fp64 reference of the attention core with stock's mask convention (finfo(float32).min added at masked keys)."""
    qd, kd, vd, bd = q.double(), k.double(), v.double(), bias_bhqk.double()
    logits = torch.einsum("bihd,bjhd->bhij", qd, kd) * scale + bd
    if key_mask is not None:
        logits = logits + torch.where(key_mask.bool()[:, None, None, :], 0.0, float(torch.finfo(torch.float32).min)).double()
    p = torch.softmax(logits, dim=-1)
    ctx = torch.einsum("bhij,bjhd->bihd", p, vd)
    if gate_raw is not None:
        ctx = ctx * torch.sigmoid(gate_raw.double())
    return ctx


def _stock_core_fp32(q, k, v, bias_bqkh, g_sig, key_mask, scale):
    """Stock's eager arithmetic (AttentionPairBias reference branch) on the same operands."""
    logits = torch.einsum("... i h d, ... j h d -> ... i j h", q, k) * scale
    logits = logits + bias_bqkh
    if key_mask is not None:
        logits = logits + torch.where(key_mask.bool()[:, None, :, None], 0.0, torch.finfo(logits.dtype).min)
    attn = torch.softmax(logits, dim=-2)
    return g_sig * torch.einsum("... i j h, ... j h d -> ... i h d", attn, v)


@needs_cuda
@pytest.mark.skipif(not pba.TRITON_OK, reason="triton not importable")
def test_fused_core_against_fp64_gold_and_stock():
    """The Triton core on strided q/k/v views (k, v chunked from one kv projection as in the module), odd lengths incl. > 700 tokens,
    key masks: error vs fp64 gold within fp32 level and no worse than 1.25x stock-eager's own error vs the same gold (kit standard)."""
    torch.manual_seed(3)
    for (bsz, n, masked) in ((2, 173, (0, 7, 172)), (1, 777, (1, 500)), (1, 64, ()), (1, 1100, ())):
        H, D = HEADS, D_MODEL // HEADS
        x = torch.randn(bsz, n, 3 * D_MODEL, device="cuda")
        q = x[..., :D_MODEL].reshape(bsz, n, H, D) if False else x[..., :D_MODEL].view(bsz, n, H, D)
        kv = x[..., D_MODEL:]
        k, v = kv.chunk(2, dim=-1)
        k = k.view(bsz, n, H, D); v = v.view(bsz, n, H, D)
        bias_bqkh = torch.randn(bsz, n, n, H, device="cuda") * 3.0
        bias_bhqk = bias_bqkh.permute(0, 3, 1, 2).contiguous()
        gate_raw = torch.randn(bsz, n, D_MODEL, device="cuda").view(bsz, n, H, D)
        mask = None
        if masked:
            mask = torch.ones(bsz, n, dtype=torch.bool, device="cuda"); mask[:, list(masked)] = False
        scale = D ** -0.5
        got = pba.fused_attention_core(q, k, v, bias_bhqk, gate_raw=gate_raw, key_mask=mask, scale=scale)
        gold = _gold_core(q, k, v, bias_bhqk, gate_raw, mask, scale)
        stock = _stock_core_fp32(q, k, v, bias_bqkh, torch.sigmoid(gate_raw), mask, scale)
        den = gold.abs().max()
        e_fused = float((got.double() - gold).abs().max() / den)
        e_stock = float((stock.double() - gold).abs().max() / den)
        assert got.shape == (bsz, n, H, D) and got.is_contiguous()
        assert e_fused < 2e-5, (n, e_fused, e_stock)
        assert e_fused <= 1.25 * e_stock, (n, e_fused, e_stock)


@needs_cuda
@pytest.mark.skipif(not pba.TRITON_OK, reason="triton not importable")
def test_fused_core_fully_masked_keys_give_stock_uniform_average_not_nan():
    """R4 'no narrowing': a batch element whose keys are ALL masked yields stock's finite uniform average (finfo.min convention), no NaN."""
    torch.manual_seed(4)
    H, D = HEADS, D_MODEL // HEADS
    bsz, n = 2, 70
    q, k, v = (torch.randn(bsz, n, H, D, device="cuda") for _ in range(3))
    bias_bqkh = torch.randn(bsz, n, n, H, device="cuda")
    gate_raw = torch.randn(bsz, n, H, D, device="cuda")
    mask = torch.ones(bsz, n, dtype=torch.bool, device="cuda"); mask[1] = False; mask[0, :5] = False
    scale = D ** -0.5
    got = pba.fused_attention_core(q, k, v, bias_bqkh.permute(0, 3, 1, 2).contiguous(), gate_raw=gate_raw, key_mask=mask, scale=scale)
    stock = _stock_core_fp32(q, k, v, bias_bqkh, torch.sigmoid(gate_raw), mask, scale)
    assert torch.isfinite(got).all() and torch.isfinite(stock).all()
    rel = float((got - stock).abs().max() / stock.abs().max())
    assert rel < 1e-4, rel


@needs_cuda
@pytest.mark.skipif(not pba.TRITON_OK, reason="triton not importable")
def test_fused_variant_module_level_reference_and_fused_backends():
    """variant='fused' through the module on both stock routes: close to stock (fp32 reassociation), far from a broken kernel
    (a bias-free or gate-free core would miss by O(1)); served/computed accounting as hoist's."""
    for backend in (None, "fused"):
        if backend == "fused" and not C.TRITON_KERNELS_AVAILABLE:
            continue
        tt = _tt(3)
        tt.set_kernel_backend(backend)
        z, a, s, mask = _inputs(211, 4, masked=(2, 210))
        ref = _run(tt, z, a, s, mask)
        st = pba.enable(tt, "fused")
        with pba.bias_scope(tt):
            got = _run(tt, z, a, s, mask)
        pba.disable(tt)
        for r, gt in zip(ref, got):
            rel = float((r - gt).abs().max() / r.abs().max())
            tol = 5e-5 if backend is None else 2e-2          # backend 'fused': stock itself runs a bf16 bias + SDPA; the class is its own
            assert rel < tol, (backend, rel)
        assert st.stats["fused_calls"] == 12 and st.stats["computed"] == 3 and st.stats["served"] == 9, st.stats


@needs_cuda
def test_grad_mode_and_expanded_z_take_the_stock_code():
    tt = _tt(1)
    z, a, s, mask = _inputs(48, 1)
    st = pba.enable(tt, "hoist")
    with pba.bias_scope(tt):
        with torch.enable_grad():
            tt(a[0].requires_grad_(True), s[0], z, 0.0)
        assert st.stats["fallback_grad"] == 1, st.stats
        with torch.no_grad():
            tt(torch.cat([a[0], a[0]]), torch.cat([s[0], s[0]]), z, 0.0, num_diffusion_samples=2)
        assert st.stats["fallback_shape"] == 1, st.stats
    pba.disable(tt)


@needs_cuda
def test_enable_is_idempotent_and_refuses_a_variant_change():
    tt = _tt(1)
    s1 = pba.enable(tt, "hoist"); s2 = pba.enable(tt, "hoist")
    assert s1 is s2
    try:
        pba.enable(tt, "fused")
        raise AssertionError("variant change without disable() must raise")
    except RuntimeError:
        pass
    pba.disable(tt)
    assert pba.describe(tt) == "pairbias_attn=off"


if __name__ == "__main__":
    names = [n for n in dir(sys.modules[__name__]) if n.startswith("test_")]
    passed = failed = 0
    for n in names:
        fn = getattr(sys.modules[__name__], n)
        marks = [m.kwargs.get("reason") for m in getattr(fn, "pytestmark", []) if m.name == "skipif" and m.args and m.args[0]]
        if getattr(fn, "_skip", False):
            marks.append(getattr(fn, "_skip_reason", ""))
        if marks:
            print(f"SKIP {n}: {marks[0]}"); continue
        try:
            fn(); passed += 1; print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {n}: {type(e).__name__}: {e}")
    print(f"RESULT passed={passed} failed={failed}")
    sys.exit(1 if failed else 0)
