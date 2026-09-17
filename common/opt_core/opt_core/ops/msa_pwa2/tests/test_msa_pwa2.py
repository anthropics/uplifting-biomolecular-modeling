"""opt_core.ops.msa_pwa2: the candidate sets by alignment class and the vouch-envelope constants (CPU); the gate words and every candidate
in the tolerance class of the stock statements (GPU, bf16 autocast; skipped without CUDA / triton). The BITWISE proof of this exact cell is
the engine's: an adapter compares each candidate torch.equal against the engine's own module at run time (the lock discipline) and the engine
kit carries the unit test against its stock class; this file holds the engine-free contracts."""
import pytest


def _mod():
    pytest.importorskip("torch"); pytest.importorskip("triton")
    from opt_core.ops import msa_pwa2 as X
    return X


def test_candidates_by_alignment():
    X = _mod()
    assert X.candidates(400)[0] == (0, 0)
    assert X.candidates(402)[0] == (1, 402 % 32) and (1, 402 % 64) in X.candidates(402)
    assert X.candidates(401)[0] == (2, 401 % 32)
    assert all(len(set(X.candidates(n))) == len(X.candidates(n)) for n in (61, 384, 385, 400, 512, 995, 1292))
    assert all(set(X.distinct_candidates(n)) <= set(X.candidates(n)) for n in (61, 400, 401, 402))


def test_vouch_envelope_constants():
    X = _mod()
    assert (X.C_M, X.C_Z, X.C_H, X.HEADS, X.HC) == (64, 128, 32, 8, 256)
    assert X.VOUCHED_CC == ((9, 0),) and X.VOUCH_STACK["torch"].startswith("2.12") and X.VOUCH_STACK["cuda"] == "13.0"


def _schema(torch, c_h=32, heads=8, seed=0):
    torch.manual_seed(seed)
    ns = torch.nn.Module()
    ns.norm_m = torch.nn.LayerNorm(64); ns.norm_z = torch.nn.LayerNorm(128)
    ns.proj_m = torch.nn.Linear(64, c_h * heads, bias=False); ns.proj_g = torch.nn.Linear(64, c_h * heads, bias=False)
    ns.proj_z = torch.nn.Linear(128, heads, bias=False); ns.proj_o = torch.nn.Linear(c_h * heads, 64, bias=False)
    ns.num_heads, ns.c_h, ns.inf = heads, c_h, 1e6
    for p in ns.parameters():
        torch.nn.init.normal_(p, std=0.12)
    torch.nn.init.normal_(ns.norm_m.weight, 1.0, 0.2); torch.nn.init.normal_(ns.norm_m.bias, 0.0, 0.2)
    return ns.cuda().eval()


def _inputs(torch, S, N, dt, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = torch.randn(1, S, N, 64, device="cuda", generator=g) * 1.5
    m = m.to(torch.bfloat16) if dt == "bf16" else m
    z = torch.randn(1, N, N, 128, device="cuda", generator=g)
    t = (torch.rand(1, N, device="cuda", generator=g) > 0.05).float()
    return m, z, t[:, :, None] * t[:, None, :]


def _ref(torch, mod, m, z, mask):
    mn = mod.norm_m(m); zn = mod.norm_z(z)
    b = mod.proj_z(zn).permute(0, 3, 1, 2)
    b = b + (1 - mask[:, None]) * -mod.inf
    w = torch.softmax(b, dim=-1)
    v = mod.proj_m(mn).view(*mn.shape[:3], mod.num_heads, mod.c_h)
    g = torch.sigmoid(mod.proj_g(mn))
    o = torch.einsum("bhij,bsjhd->bsihd", w, v).reshape(*mn.shape[:3], -1)
    return mod.proj_o(g * o)


def test_gate_words():
    X = _mod()
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    mod = _schema(torch)
    m, z, mask = _inputs(torch, 64, 400, "bf16")
    assert X.supported(mod, m, z, mask, True) is None
    assert X.supported(mod, m.half(), z, mask, True) == "dtype"
    assert X.supported(mod, m.cpu(), z.cpu(), mask.cpu(), True) == "device"
    assert X.supported(mod, torch.cat([m, m]), torch.cat([z, z]), torch.cat([mask, mask]), True) == "batch"
    assert X.supported(mod, m, z[:, :300, :300], mask[:, :300, :300], True) == "shape"
    assert X.supported(_schema(torch, c_h=16, heads=4), m, z, mask, True) == "dims"


@pytest.mark.parametrize("S,N,dt", [(128, 400, "bf16"), (256, 401, "fp32"), (64, 61, "bf16")])
def test_every_candidate_is_in_the_tolerance_class(S, N, dt):
    X = _mod()
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    mod = _schema(torch)
    m, z, mask = _inputs(torch, S, N, dt)
    chunk = N > 384
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = _ref(torch, mod, m, z, mask).float()
        for km, R in X.candidates(N):
            got = X.pwa_forward(mod, m, z, mask, chunk, km, R)
            assert got.dtype == torch.bfloat16 and got.shape == ref.shape
            err = (got.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
            assert float(err) < 2e-2, (km, R, float(err))
            assert torch.equal(got, X.pwa_forward(mod, m, z, mask, chunk, km, R)), "run-to-run bitwise"
