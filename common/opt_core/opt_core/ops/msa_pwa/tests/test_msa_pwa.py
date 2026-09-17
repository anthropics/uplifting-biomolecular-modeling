"""opt_core.ops.msa_pwa: the pinned rows are named variants (CPU), and the fused pair-weighted averaging (masked form) is in the tolerance
class of the stock statements (GPU, bf16 autocast; skipped without CUDA / triton). A fast-class cell: no bitwise claim."""
import pytest


def _mod():
    pytest.importorskip("torch"); pytest.importorskip("triton")
    from opt_core.ops import msa_pwa as P
    return P


def test_pinned_rows_are_named_variants():
    P = _mod()
    names = {n for n, r in P.CFG_VARIANTS.items()}
    for c, row in P._CFG.items():
        assert any(dict(r) == dict(row) for r in P.CFG_VARIANTS.values()), (c, row)
    assert {"g_fo4p", "hp1", "v0"} <= names
    assert set(P.FPF_META) == {"msa_pair_weighted_avg"}


def _schema(torch, seed=0):
    torch.manual_seed(seed)
    ns = torch.nn.Module()
    ns.norm_m = torch.nn.LayerNorm(64); ns.norm_z = torch.nn.LayerNorm(128)
    ns.proj_m = torch.nn.Linear(64, 256, bias=False); ns.proj_g = torch.nn.Linear(64, 256, bias=False)
    ns.proj_z = torch.nn.Linear(128, 8, bias=False); ns.proj_o = torch.nn.Linear(256, 64, bias=False)
    ns.num_heads, ns.c_h, ns.inf = 8, 32, 1e6
    for p in ns.parameters():
        torch.nn.init.normal_(p, std=0.12)
    return ns.cuda().eval()


def _ref_masked(torch, mod, m, z, mask):
    """The stock statements of the masked form (unchunked path), under the caller's autocast."""
    mn = mod.norm_m(m); zn = mod.norm_z(z)
    b = mod.proj_z(zn).permute(0, 3, 1, 2)
    b = b + (1 - mask[:, None]) * -mod.inf
    w = torch.softmax(b, dim=-1)
    v = mod.proj_m(mn).view(*mn.shape[:3], mod.num_heads, mod.c_h)
    g = torch.sigmoid(mod.proj_g(mn))
    o = torch.einsum("bhij,bsjhd->bsihd", w, v).reshape(*mn.shape[:3], -1)
    return mod.proj_o(g * o)


@pytest.mark.parametrize("S,N", [(64, 48), (200, 150)])
def test_masked_form_is_in_the_tolerance_class(S, N):
    P = _mod()
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    mod = _schema(torch)
    gen = torch.Generator(device="cuda").manual_seed(1)
    m = torch.randn(1, S, N, 64, device="cuda", generator=gen)
    z = torch.randn(1, N, N, 128, device="cuda", generator=gen)
    t = (torch.rand(1, N, device="cuda", generator=gen) > 0.05).float()
    mask = t[:, :, None] * t[:, None, :]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = _ref_masked(torch, mod, m, z, mask).float()
        got = P.forward_masked(mod, m, z, mask, False).float()
        got2 = P.forward_masked(mod, m, z, mask, False).float()
    assert got.shape == ref.shape and torch.equal(got, got2)
    err = (got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert float(err) < 2e-2, float(err)
