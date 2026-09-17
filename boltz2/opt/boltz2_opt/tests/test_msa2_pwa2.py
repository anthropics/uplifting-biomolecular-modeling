"""msa2 pwa2x (the shared core's opt_core.ops.msa_pwa2): the fused PairWeightedAveraging, lever msa_pwa_exact (BOLTZ_MSA2=pwa2x).
BIT-IDENTICAL to the stock Boltz-2 PairWeightedAveraging(c_m=64, c_z=128, c_h=32, 8 heads) under the trunk's regime (eval, CUDA autocast bf16) for the
candidate summation structure the class selects: for each (S, N, path) case at least one of pwa2.candidates(N) must be torch.equal to the stock
module (chunked and unchunked paths, bf16 and fp32 m, N of every alignment class). GPU test on the pinned stack: skipped without CUDA / boltz."""
import importlib.util

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if importlib.util.find_spec("boltz") is None:
    pytest.skip("boltz not importable", allow_module_level=True)

def _p2():
    pytest.importorskip("triton")
    import importlib as il
    return il.import_module("opt_core.ops.msa_pwa2")


def _module(seed=0):
    from boltz.model.layers.pair_averaging import PairWeightedAveraging
    torch.manual_seed(seed)
    pwa = PairWeightedAveraging(64, 128, 32, 8).cuda().eval()
    for p in pwa.parameters():
        torch.nn.init.normal_(p, std=0.12)
    torch.nn.init.normal_(pwa.norm_m.weight, 1.0, 0.2); torch.nn.init.normal_(pwa.norm_m.bias, 0.0, 0.2)
    return pwa


def _inputs(S, N, dt, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = torch.randn(1, S, N, 64, device="cuda", generator=g) * 1.5
    m = m.to(torch.bfloat16) if dt == "bf16" else m
    z = torch.randn(1, N, N, 128, device="cuda", generator=g)
    t = (torch.rand(1, N, device="cuda", generator=g) > 0.05).float()
    return m, z, t[:, :, None] * t[:, None, :]


@pytest.mark.parametrize("S,N,dt", [(512, 400, "bf16"), (1024, 401, "fp32"), (1024, 402, "fp32"), (777, 430, "fp32"), (2048, 500, "fp32"), (600, 777, "fp32"),
                                    (256, 1292, "fp32"), (512, 256, "bf16"), (1024, 300, "fp32"), (64, 61, "bf16")])
def test_some_candidate_is_bitwise(S, N, dt):
    P = _p2(); pwa = _module()
    from boltz.model.layers.pair_averaging import PairWeightedAveraging
    m, z, mask = _inputs(S, N, dt)
    chunk = N > 384
    assert P.supported(pwa, m, z, mask, chunk) is None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = PairWeightedAveraging.forward(pwa, m, z, mask, chunk)
        results = {}
        for km, R in P.candidates(N):
            got = P.pwa_forward(pwa, m, z, mask, chunk, km, R)
            assert got.dtype == ref.dtype == torch.bfloat16 and got.shape == ref.shape
            results[(km, R)] = int((got.view(torch.int16) != ref.view(torch.int16)).sum().item())
    assert min(results.values()) == 0, f"no candidate reproduces cuBLAS at S={S} N={N} {dt}: mismatches {results}"


def test_candidates_by_alignment():
    P = _p2()
    assert P.candidates(400)[0] == (0, 0)
    assert P.candidates(402)[0] == (1, 402 % 32) and (1, 402 % 64) in P.candidates(402)
    assert P.candidates(401)[0] == (2, 401 % 32)
    assert all(len(set(P.candidates(n))) == len(P.candidates(n)) for n in (61, 384, 385, 400, 512, 995, 1292))


def test_fallback_words():
    P = _p2(); pwa = _module()
    m, z, mask = _inputs(64, 400, "bf16")
    assert P.supported(pwa, m.half(), z, mask, True) == "dtype"
    assert P.supported(pwa, m.cpu(), z.cpu(), mask.cpu(), True) == "device"
    assert P.supported(pwa, torch.cat([m, m]), torch.cat([z, z]), torch.cat([mask, mask]), True) == "batch"
    assert P.supported(pwa, m, z[:, :300, :300], mask[:, :300, :300], True) == "shape"
    from boltz.model.layers.pair_averaging import PairWeightedAveraging
    other = PairWeightedAveraging(64, 128, 16, 4).cuda().eval()
    assert P.supported(other, m, z, mask, True) == "dims"
