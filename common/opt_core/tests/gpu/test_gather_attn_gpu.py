"""GPU tests of opt_core.kernels.gather_attn (run on a CUDA box with Triton: `python -m pytest common/opt_core/tests/gpu/test_gather_attn_gpu.py -q`;
skipped by name elsewhere). The kernel vs the dense-masked fp32 reference at the tested cells and at boundary shapes (B=1 / B>1, shared vs
per-sample bias and index sets, LQ != LK, LQ not a multiple of the query tile, k not a multiple of the key chunk, -1 padding, duplicates,
unsorted rows, an empty set, fp32 vs 16-bit q/k, gate / no gate, out=), run-to-run bit-exact, no LQ x LK allocation, refusals by name."""
import math
import os

import pytest

torch = pytest.importorskip("torch", reason="needs torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
from opt_core.kernels import gather_attn as GA  # noqa: E402
if not GA.HAVE_TRITON:
    pytest.skip("needs Triton", allow_module_level=True)

DEV = torch.device("cuda")


def _case(B=2, LQ=37, LK=53, H=4, dh=32, k=128, shared_bias=False, shared_idx=False, seed=0, pad=0, dup=0, qk16=False, gate=True):
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = H * dh
    k = min(k, LK)
    q = torch.randn(B, LQ, C, generator=g).to(DEV); kk = torch.randn(B, LK, C, generator=g).to(DEV)
    if qk16:
        q, kk = q.to(torch.bfloat16), kk.to(torch.bfloat16)
    v = torch.randn(B, LK, C, generator=g).to(DEV, torch.bfloat16)
    gt = torch.rand(B, LQ, C, generator=g).to(DEV, torch.bfloat16) if gate else None
    bias = (0.5 * torch.randn(1 if shared_bias else B, LQ, LK, H, generator=g)).to(DEV, torch.bfloat16)
    ib = 1 if shared_idx else B
    idx = torch.stack([torch.stack([torch.randperm(LK, generator=g)[:k].sort().values for _ in range(LQ)]) for _ in range(ib)]).to(DEV, torch.int32)
    if pad:
        idx[:, ::3, :pad] = -1
        idx = torch.sort(idx, -1).values
    if dup:
        idx[:, 1::2, dup] = idx[:, 1::2, dup + 1]
    return q, kk, v, bias, idx, H, gt


def _check(args, allow_candidate=True, rel=6e-3, **kw):
    q, k, v, bias, idx, H, gate = args
    out = GA.gather_attn(q, k, v, bias, idx, H, gate=gate, allow_candidate=allow_candidate, **kw)
    ref = GA.reference(q, k, v, bias, idx, H, gate=gate, round_qk=True)
    assert out.shape == q.shape and out.dtype == v.dtype
    d = out.float() - ref
    rr = (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert rr < rel, rr
    assert d.abs().max().item() < 0.08, d.abs().max().item()
    out2 = GA.gather_attn(q, k, v, bias, idx, H, gate=gate, allow_candidate=allow_candidate, **kw)
    assert torch.equal(out, out2), "run-to-run bitwise"
    return out


@pytest.mark.parametrize("cell", [dict(H=4, dh=32, k=128, LQ=300, LK=300), dict(H=16, dh=48, k=32, LQ=100, LK=100)])
@pytest.mark.parametrize("B,shared_bias,shared_idx", [(1, True, True), (3, True, False), (3, False, False), (2, False, True)])
def test_certified_cells_vs_reference(cell, B, shared_bias, shared_idx):
    args = _case(B=B, shared_bias=shared_bias, shared_idx=shared_idx, **cell)
    assert GA.supported(*args[:5], args[5], gate=args[6]) is None            # tested: no opt-in needed
    _check(args, allow_candidate=False)


@pytest.mark.parametrize("kw", [dict(LQ=37, LK=53, k=13), dict(LQ=16, LK=16, k=16), dict(LQ=1, LK=9, k=9), dict(LQ=129, LK=40, k=1, H=2, dh=8),
                                dict(LQ=64, LK=200, k=129, dh=64, H=2), dict(LQ=33, LK=33, k=7, dh=100, H=1), dict(LQ=20, LK=300, k=128, qk16=True),
                                dict(LQ=20, LK=300, k=128, gate=False), dict(pad=3), dict(dup=2), dict(pad=2, dup=5)])
def test_boundary_shapes(kw):
    _check(_case(**kw))


def test_unsorted_rows_are_sorted_here_and_equal_the_sorted_call():
    q, k, v, bias, idx, H, gate = _case(LQ=25, LK=60, k=17)
    perm = torch.argsort(torch.rand(idx.shape, device=DEV), -1)
    shuffled = torch.gather(idx, -1, perm)
    n0 = GA.stats()["rows_sorted_here"]
    a = GA.gather_attn(q, k, v, bias, shuffled, H, gate=gate, allow_candidate=True)
    assert GA.stats()["rows_sorted_here"] == n0 + 1
    b = GA.gather_attn(q, k, v, bias, idx, H, gate=gate, allow_candidate=True, ensure_sorted=False)
    assert torch.equal(a, b)


def test_empty_set_row_is_nan_only_there():
    q, k, v, bias, idx, H, gate = _case(B=1, LQ=5, LK=12, k=4, H=2, dh=16)
    idx[0, 2, :] = -1
    out = GA.gather_attn(q, k, v, bias, idx, H, gate=gate, allow_candidate=True)
    assert torch.isnan(out[0, 2]).all() and not torch.isnan(out[0, [0, 1, 3, 4]]).any()


def test_out_param_and_no_lq_x_lk_allocation():
    q, k, v, bias, idx, H, gate = _case(B=2, LQ=512, LK=512, k=128)
    out = torch.empty_like(v[:, :512])
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    r = GA.gather_attn(q, k, v, bias, idx, H, gate=gate, out=out)
    torch.cuda.synchronize()
    assert r is out and torch.cuda.max_memory_allocated() - base < 4 * 2 * 512 * 128 * 4     # far below B*H*LQ*LK*2 = 4 MiB; only sort/compare temporaries
    with pytest.raises(GA.Refusal) as e:
        GA.gather_attn(q, k, v, bias, idx, H, out=torch.empty(3, device=DEV))
    assert e.value.reason == "out-layout"


def test_refusals_by_name():
    q, k, v, bias, idx, H, gate = _case(LQ=8, LK=20, k=6, dh=8, H=2)
    with pytest.raises(GA.Refusal) as e:
        GA.gather_attn(q, k, v, bias, idx, H)                                  # (8, 6, bf16) is not a tested cell
    assert e.value.reason == "cell-uncertified:dh8/k6/bfloat16+off(not-measured)"           # a (dh, k, dtype) key outside the table: the stock path BY NAME, qualified as measured-off
    os.environ[GA.ALLOW_CANDIDATE_ENV] = "1"
    try:
        GA.gather_attn(q, k, v, bias, idx, H)                                  # the env opt-in serves it
    finally:
        del os.environ[GA.ALLOW_CANDIDATE_ENV]
    assert GA.supported(q, k, v.float(), bias, idx, H) == "v-dtype-float32"
    assert GA.supported(q, k, v, bias, idx.long(), H) == "idx-not-int32"
    assert GA.supported(q.cpu(), k, v, bias, idx, H) == "not-cuda"
    assert GA.supported(q, k, v, bias.transpose(1, 2), idx, H, allow_candidate=True) == ("bias-shape" if bias.shape[1] != bias.shape[2] else None)
    with torch.enable_grad():
        assert GA.supported(q.clone().requires_grad_(True), k, v, bias, idx, H, allow_candidate=True) == "grad"


def test_scale_argument_matches_reference():
    q, k, v, bias, idx, H, gate = _case(LQ=10, LK=30, k=10)
    out = GA.gather_attn(q, k, v, bias, idx, H, gate=gate, scale=0.3, allow_candidate=True)
    ref = GA.reference(q, k, v, bias, idx, H, gate=gate, scale=0.3, round_qk=True)
    assert ((out.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item() < 6e-3
