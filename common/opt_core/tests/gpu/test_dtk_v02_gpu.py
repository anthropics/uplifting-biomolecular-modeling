"""GPU tests of opt_core.kernels.dtk_kernels' DTK v0.2 additions (run on a CUDA box with Triton:
`python -m pytest common/opt_core/tests/gpu/test_dtk_v02_gpu.py -q`; skipped by name elsewhere).
(1) `gate_residual(rowmask=m, mask_period=PM)`: rows whose mask is 0 come out as the residual EXACTLY; every row equals the fp32 reference
`res + m * sigmoid(g) * x` to fp32 rounding (the extra multiply changes the FMA contraction, so unmasked rows are within an ulp of the
mask-less call, not bitwise); a bool mask reads as 0/1; (2) `segments` +
`seg_reduce` = the per-token masked mean of an fp64 index_add reference to fp32 accumulation error on token-sorted layouts with masked and
padded atoms and an empty token; an interleaved layout is reported `sorted=False`; (3) zero-size inputs return empty outputs."""
import pytest

torch = pytest.importorskip("torch", reason="needs torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
try:
    import triton  # noqa: F401
except ImportError:
    pytest.skip("needs Triton", allow_module_level=True)
from opt_core.kernels import dtk_kernels as D  # noqa: E402

DEV = torch.device("cuda")


@pytest.mark.parametrize("S,N,C,dtype", [(5, 203, 768, torch.bfloat16), (3, 64, 384, torch.float32), (1, 7, 33, torch.float16)])
def test_rowmask_multiplies_rows_periodically(S, N, C, dtype):
    g = torch.Generator(device="cpu").manual_seed(11)
    R = S * N
    x = torch.randn(R, C, generator=g).to(DEV, dtype); res = torch.randn(R, C, generator=g).to(DEV, torch.float32)
    gate = torch.randn(N, C, generator=g).to(DEV, dtype)
    m01 = (torch.rand(N, generator=g) > 0.3).float().to(DEV)
    got = D.gate_residual(x, gate=gate, gate_period=N, rowmask=m01, mask_period=N, res=res, out_dtype=torch.float32)
    ref = D.gate_residual(x, gate=gate, gate_period=N, res=res, out_dtype=torch.float32)               # the mask-less call
    rows0 = (m01 == 0).repeat(S)                                                                       # sample-major rows: row r <- token r % N
    assert torch.equal(got[rows0], res[rows0])                                                         # masked rows: exactly the residual
    assert (got[~rows0] - ref[~rows0]).abs().max().item() <= 4 * torch.finfo(torch.float32).eps * max(1.0, ref.abs().max().item())
    mf = torch.rand(N, generator=g).to(DEV)                                                            # fractional mask: the fp32 reference
    got = D.gate_residual(x, gate=gate, gate_period=N, rowmask=mf, mask_period=N, res=res, out_dtype=torch.float32)
    ref = res + mf.view(1, N, 1).expand(S, N, 1).reshape(R, 1) * torch.sigmoid(gate.float()).repeat(S, 1) * x.float()
    assert (got - ref).abs().max().item() <= 1e-5 * max(1.0, ref.abs().max().item())
    mb = m01.bool()                                                                                    # a bool mask is taken as 0/1
    assert torch.equal(D.gate_residual(x, gate=gate, gate_period=N, rowmask=mb, mask_period=N, res=res, out_dtype=torch.float32),
                       D.gate_residual(x, gate=gate, gate_period=N, rowmask=m01, mask_period=N, res=res, out_dtype=torch.float32))


def test_zero_rows():
    x = torch.empty(0, 768, device=DEV)
    assert D.gate_residual(x).shape == (0, 768)
    st, ct, mx, ok = D.segments(torch.empty(0, dtype=torch.long, device=DEV), torch.empty(0, device=DEV), 0)
    assert ok and st.numel() == 0 and D.seg_reduce(torch.empty(2, 0, 8, device=DEV), st, ct, mx).shape == (2, 0, 8)


@pytest.mark.parametrize("N,S,C,dtype", [(37, 3, 96, torch.float32), (5, 1, 768, torch.bfloat16), (300, 2, 33, torch.float16), (2024, 5, 128, torch.bfloat16)])
def test_seg_reduce_is_the_masked_mean(N, S, C, dtype):
    g = torch.Generator(device="cpu").manual_seed(3)
    counts = torch.randint(1, 24, (N,), generator=g); counts[N // 2] = 0                              # one token without atoms
    idx = torch.repeat_interleave(torch.arange(N), counts)
    pad = 7
    idx = torch.cat([idx, torch.zeros(pad, dtype=idx.dtype)]); A = idx.numel()                       # padded tail pointing at token 0, masked
    mask = torch.ones(A); mask[-pad:] = 0; mask[torch.randint(0, A - pad, (A // 9,), generator=g)] = 0
    feat = torch.randn(S, A, C, generator=g).to(DEV, dtype); idx = idx.to(DEV); mask = mask.to(DEV)
    st, ct, mx, ok = D.segments(idx, mask, N)
    assert ok and int(ct[N // 2]) == 0
    for mean in (True, False):
        out = D.seg_reduce(feat, st, ct, mx, atom_mask=mask, mean=mean, out_dtype=torch.float32)
        f64 = feat.double() * mask.double()[None, :, None]
        ref = torch.zeros(S, N, C, dtype=torch.float64, device=DEV).index_add_(1, idx, f64)
        if mean:
            den = torch.zeros(N, dtype=torch.float64, device=DEV).index_add_(0, idx, mask.double())
            ref = ref / (den[None, :, None] + 1e-9)
        err = (out.double() - ref).abs().max().item()
        assert err <= 24 * 2 * torch.finfo(torch.float32).eps * max(ref.abs().max().item(), 1.0), (mean, err)
    fs = feat[:, ::2]                                                                                  # a strided atom axis is read in place
    st2, ct2, mx2, ok2 = D.segments(idx[::2], mask[::2], N)
    out = D.seg_reduce(fs, st2, ct2, mx2, atom_mask=mask[::2].contiguous(), mean=False, out_dtype=torch.float32)
    ref = torch.zeros(S, N, C, dtype=torch.float64, device=DEV).index_add_(1, idx[::2], fs.double() * mask[::2].double()[None, :, None])
    assert ok2 and (out.double() - ref).abs().max().item() <= 24 * 2 * torch.finfo(torch.float32).eps * max(ref.abs().max().item(), 1.0)


def test_unsorted_layout_is_reported_not_served():
    st, ct, mx, ok = D.segments(torch.tensor([0, 2, 1], device=DEV), torch.ones(3, device=DEV), 3)
    assert not ok and ct.tolist() == [0, 0, 0] and int(mx) == 0
    st, ct, mx, ok = D.segments(torch.tensor([0, 2, 1], device=DEV), torch.tensor([1.0, 0.0, 1.0], device=DEV), 3)
    assert ok and ct.tolist() == [1, 1, 0]                                                             # the out-of-order atom is masked
