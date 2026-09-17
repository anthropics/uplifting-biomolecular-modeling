"""Sampler levers — unit tests for bz_sampler.py / bz_sampler_dit.py.
CPU-only tests run anywhere with torch; the CUDA tests need a GPU + triton + boltz 2.2.1 importable
(`$STACK_PYTHON -m pytest opt/forward/sampler/tests/test_sampler.py`).
The exact-tier proof is NOT here (it is the per-YAML sha256 comparison of every output file against `boltz predict`, done with the dev tools);
these tests pin the pieces: scope words, the stock draw order of the pre-drawn randomness, the division-recipe probe, the device Kabsch kernel
against stock's SVD statements (incl. reflections / near-planar / rank-2 covariances), and the fused bf16 token-transformer step against the
stock module at small N (incl. exactly-zero attention weight on masked key columns)."""
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import bz_sampler  # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA device required")


def _has_boltz():
    try:
        import boltz.model.modules.utils as _  # noqa: F401
        return True
    except Exception:
        return False


needs_boltz = pytest.mark.skipif(not _has_boltz(), reason="boltz 2.2.1 not importable")


# ------------------------------------------------------------------------------------------------------------------ scope words (CPU)
def test_guidance_scope_words():
    f = {"contact_pair_index": torch.zeros(1, 2, 0, dtype=torch.long)}
    assert bz_sampler._guidance_active({"contact_guidance_update": True}, f) is None
    assert bz_sampler._guidance_active({"contact_guidance_update": False}, f) is None
    f2 = {"contact_pair_index": torch.zeros(1, 2, 3, dtype=torch.long)}
    assert bz_sampler._guidance_active({"contact_guidance_update": True}, f2) == "contact_constraints"
    f3 = dict(f, template_mask_cb=torch.ones(1, 2, 5), template_force=torch.tensor([[False, True]]))
    assert bz_sampler._guidance_active({"contact_guidance_update": True}, f3) == "template_force"
    f4 = dict(f, template_mask_cb=torch.ones(1, 2, 5), template_force=torch.tensor([[False, False]]))
    assert bz_sampler._guidance_active({"contact_guidance_update": True}, f4) is None


def test_recip32_recipes_differ_somewhere():
    """the two candidate reciprocal roundings are genuinely different functions (else the install-time probe could not decide anything)."""
    import numpy as np
    vals = [160.0 * (0.98 ** k) + 0.37 for k in range(200)]
    n_diff = sum(1 for b in vals if bz_sampler._recip32(b, "double") != bz_sampler._recip32(b, "float"))
    assert n_diff > 0
    for b in vals:
        assert abs(float(bz_sampler._recip32(b, "double")) * b - 1.0) < 1e-6 and isinstance(bz_sampler._recip32(b, "float"), np.float32)


# ------------------------------------------------------------------------------------------------------------------ CUDA pieces
@needs_cuda
def test_probe_scalar_div_decides():
    dev = torch.device("cuda")
    vals = [160.0 * (0.98 ** k) + 0.37 for k in range(200)] + [3.0, 7.0, 0.1]
    assert bz_sampler.probe_scalar_div(dev, vals) in ("double", "float")


@needs_cuda
@needs_boltz
def test_predraw_matches_stock_draw_order():
    """_predraw consumes the CUDA generator exactly like the stock loop: init randn(shape); per step random_rotations (randn((m,4)) -> R),
    randn((m,1,3)) * s_trans, randn(shape)."""
    from boltz.model.modules.utils import random_rotations
    dev = torch.device("cuda"); shape = (3, 37, 3); m, S = 3, 5
    torch.manual_seed(11)
    init, R, tr, noise = bz_sampler._predraw(shape, m, S, dev, torch.float32)
    torch.manual_seed(11)
    init_ref = torch.randn(shape, device=dev)
    assert torch.equal(init, init_ref)
    for k in range(S):
        R_ref = random_rotations(m, dtype=torch.float32, device=dev)
        tr_ref = torch.randn((m, 1, 3), dtype=torch.float32, device=dev) * 1.0
        eps_ref = torch.randn(shape, device=dev)
        assert torch.equal(R[k], R_ref) and torch.equal(tr[k], tr_ref) and torch.equal(noise[k], eps_ref), k


def _stock_R(cov):
    """weighted_rigid_align's statements from the SVD on: R = U F V^T with F = diag(1, 1, det(U V^T))."""
    U, Sv, Vh = torch.linalg.svd(cov, driver="gesvd" if cov.is_cuda else None); V = Vh.mH
    rot = torch.einsum("... i j, ... k j -> ... i k", U, V)
    F = torch.eye(3, device=cov.device)[None].repeat(cov.shape[0], 1, 1); F[:, -1, -1] = torch.det(rot)
    return torch.einsum("... i j, ... j k, ... l k -> ... i l", U, F, V), Sv


@needs_cuda
def test_kabsch_device_vs_stock_statements():
    """R from the fp64 Jacobi kernel vs stock's gesvd + det correction on random, reflected, near-planar (rank-2-ish) and exactly rank-2
    covariances: same rotation to 1e-5 where the rotation is well defined (R6(b)), proper (det +1) everywhere, smallest singular value kept."""
    pytest.importorskip("triton")
    dev = torch.device("cuda"); g = torch.Generator(device="cpu").manual_seed(5)
    rnd = torch.randn(256, 3, 3, generator=g)
    refl = torch.randn(256, 3, 3, generator=g); refl = refl * torch.sign(torch.det(refl))[:, None, None] * -1        # det < 0: the reflection branch
    planar = torch.randn(256, 3, 3, generator=g); planar[:, 2, :] *= 1e-4                                            # third singular value ~1e-4 of the others
    pts = torch.randn(256, 50, 3, generator=g); pts[:, :, 2] = 0                                                     # exactly planar point sets -> rank-2 H
    Q, _ = torch.linalg.qr(torch.randn(256, 3, 3, generator=g))
    rank2 = torch.einsum("b n i, b n j -> b i j", pts, torch.einsum("b n i, b j i -> b n j", pts, Q))
    for name, cov, tol in (("random", rnd, 1e-5), ("reflected", refl, 1e-5), ("near_planar", planar, 1e-4), ("rank2", rank2, 1e-4)):
        cov = cov.to(dev).float().contiguous()
        R = torch.empty_like(cov); det = torch.empty(cov.shape[0], device=dev); smin = torch.empty(cov.shape[0], device=dev)
        bz_sampler.kabsch_rotation_device(cov, R, det, smin)
        R_ref, Sv = _stock_R(cov)
        torch.cuda.synchronize()
        err = (R - R_ref).abs().max().item()
        assert err < tol, (name, err)
        assert (torch.det(R.double()) - 1.0).abs().max().item() < 1e-5, name
        assert ((smin - Sv[:, -1]).abs() / Sv[:, 0].abs().clamp_min(1e-30)).max().item() < 1e-4, name


@needs_cuda
@needs_boltz
def test_dit_fused_vs_stock_small():
    """the fused step vs the stock DiffusionTransformer forward (randomised weights, N=100 with 7 padded keys, 3 layers): bf16-class relative
    error bound; the tf32/ieee words tighter."""
    pytest.importorskip("triton")
    import bz_sampler_dit as D
    from boltz.model.modules.transformersv2 import DiffusionTransformer
    dev = torch.device("cuda"); torch.manual_seed(3)
    L, H, dim, N = 3, 16, 768, 100
    tt = DiffusionTransformer(depth=L, heads=H, dim=dim, dim_single_cond=dim, pair_bias_attn=True, activation_checkpointing=False).to(dev).eval()
    with torch.no_grad():
        for p in tt.parameters():                        # zero-initialised output projections would make the test vacuous: randomise everything mildly
            p.copy_(torch.randn_like(p) * (0.5 / max(p.shape[-1], 1) ** 0.5) if p.dim() > 1 else torch.randn_like(p) * 0.1)
    a = torch.randn(1, N, dim, device=dev); s = torch.randn(1, N, dim, device=dev)
    bias = torch.randn(1, N, N, L * H, device=dev) * 0.5           # the hoisted per-layer bias as boltz passes it: [B, N, N, L*H]
    mask = torch.ones(1, N, device=dev); mask[:, -7:] = 0
    with torch.no_grad():
        ref = DiffusionTransformer.forward(tt, a.clone(), s, bias=bias, mask=mask.float(), multiplicity=1)
    fake_score_model = types.SimpleNamespace(token_transformer=tt)
    for words, tol in ((["bf16"], 0.06), (["bf16", "gemm=tf32", "attn=ieee"], 0.01)):
        D._STATE.update(tt=None, st=None); D.STATS["scope"] = {}; tt.__dict__.pop("forward", None); tt.__dict__.pop("_bzs_dit", None)
        D.install(words)
        assert D.attach(fake_score_model)
        D.prepare(fake_score_model, {"token_pad_mask": mask}, {"token_trans_bias": bias}, multiplicity=1)
        with torch.no_grad():
            out = tt(a.clone(), s, bias=bias, mask=mask.float(), multiplicity=1)
        torch.cuda.synchronize()
        assert not D.STATS["scope"], D.STATS["scope"]
        err = ((out - ref).abs().max() / ref.abs().max()).item()
        assert err < tol, (words, err)
        # structural: masked KEY columns receive exactly zero attention weight — perturbing the padded tokens' a and s (the only cross-token
        # path is attention) must leave every unmasked query row bit-identical
        a2 = a.clone(); s2 = s.clone(); a2[:, -7:] += 1e3 * torch.randn_like(a2[:, -7:]); s2[:, -7:] -= 1e3
        with torch.no_grad():
            out2 = tt(a2, s2, bias=bias, mask=mask.float(), multiplicity=1)
        torch.cuda.synchronize()
        assert torch.equal(out2[:, :-7], out[:, :-7]), (words, (out2[:, :-7] - out[:, :-7]).abs().max().item())
        D.release()


@needs_cuda
def test_predraw_batched_equals_per_step_and_keeps_the_stream():
    """predraw=batched: the batched quaternion->rotation statements are bit-identical to the per-step ones on this build for the live class,
    the draws stay the stock sequence (same tensors, same generator offset afterwards), and a refused class stays looped by name."""
    dev = torch.device("cuda", 0)
    shape = (2, 50, 3); m, S = 2, 12
    gen = torch.cuda.default_generators[0]
    out = {}
    for mode in ("loop", "batched", "batched"):            # 2nd batched call exercises the proven-class fast path
        bz_sampler._CFG["predraw"] = mode
        torch.manual_seed(11); off0 = gen.get_offset()
        init, R, tr, noise = bz_sampler._predraw(shape, m, S, dev, torch.float32)
        after = torch.randn(7, device=dev)                  # the next draw of the process must be unchanged too
        out.setdefault(mode, []).append((init, R, tr, noise, after, gen.get_offset() - off0))
    a, b, c = out["loop"][0], out["batched"][0], out["batched"][1]
    for x, y, z in zip(a[:5], b[:5], c[:5]):
        assert torch.equal(x, y) and torch.equal(x, z)
    assert a[5] == b[5] == c[5]
    assert bz_sampler.STATS["predraw"] == "batched" and bz_sampler.STATS["predraw_classes"].get(f"m{m}_S{S}_torch.float32") is True
    # the per-step reference: stock's own functions step by step on the same draws
    from boltz.model.modules.utils import random_rotations
    torch.manual_seed(11); _ = torch.randn(shape, device=dev)
    for k in range(S):
        Rk = random_rotations(m, torch.float32, dev)
        assert torch.equal(Rk, a[1][k]), k
        _ = torch.randn((m, 1, 3), device=dev); _ = torch.randn(shape, device=dev)
    bz_sampler._CFG["predraw"] = "batched"
