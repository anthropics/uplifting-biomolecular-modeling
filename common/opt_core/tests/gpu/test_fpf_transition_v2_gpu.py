"""GPU numerics of the `v2` transition variant (`kernels/fpf_transition_v2` through `opt_core.attn.pair_fused.transition`): the update, the masked
update, and x + mask*update written IN PLACE agree with the torch statements of an AF3 SwiGLU transition (bf16 LN -> Linear a|b -> silu(a)*b ->
Linear -> * mask -> += ) at c=64 and c=128 (within ~1 bf16 ulp on a small fraction of the elements), at the same float64 distance as those statements; rows
that are not a tile multiple, zero rows, a strided (channel-sliced) input, bool / float / non-binary masks, run-to-run determinism, the
`transition_plan_words` variant naming and the by-name refusals; `ln_proj.pair_bias` head-major planes vs the LayerNorm + Linear statements."""
import pytest

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
V2 = ("v2_fold", "v1")          # the caller's preference that opts into the fold variant (the core serves it to no one else)
sm90 = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="v2 rows are sm_90")


def _stock(x, ln, wa, wb, wo, mask=None, residual=None):
    """The AF3 engines' statements on a bf16 stream: LayerNorm with the affine parameters cast to the stream dtype (autocast off: bf16 in/out,
    fp32 statistics), then under bf16 autocast two Linear, the fused silu(a)*b (computed in fp32 from the bf16 a, b and rounded once, as the
    engines' fused SiLU-mul kernel does), Linear, * mask, residual += ."""
    with torch.autocast("cuda", enabled=False):
        y = torch.nn.functional.layer_norm(x, (x.shape[-1],), ln.weight.to(x.dtype), ln.bias.to(x.dtype), ln.eps)
    with torch.autocast("cuda", torch.bfloat16):
        a = torch.nn.functional.linear(y, wa); b = torch.nn.functional.linear(y, wb)
        h = (a.float() * torch.sigmoid(a.float()) * b.float()).to(torch.bfloat16)
        u = torch.nn.functional.linear(h, wo)
        if mask is not None:
            u = u * mask.unsqueeze(-1)
    if residual is None:
        return u
    r = residual.clone(); r += u
    return r


def _f64(x, ln, wa, wb, wo, mask=None, residual=None):
    y = torch.nn.functional.layer_norm(x.double(), (x.shape[-1],), ln.weight.to(x.dtype).double(), ln.bias.to(x.dtype).double(), ln.eps)
    a = y @ wa.double().t(); b = y @ wb.double().t()
    u = (torch.nn.functional.silu(a) * b) @ wo.double().t()
    if mask is not None:
        u = u * mask.double().unsqueeze(-1)
    return u if residual is None else residual.double() + u


def _ulps(a, ref):
    a = a.double(); ref = ref.double()
    scale = torch.maximum(ref.abs(), ref.pow(2).mean().sqrt() * torch.ones_like(ref)) * 2.0 ** -8
    return ((a - ref).abs() / scale)


def _make(c, n, lead, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = "cuda"
    ln = torch.nn.LayerNorm(c).to(dev)
    with torch.no_grad():
        ln.weight.copy_(1 + 0.1 * torch.randn(c, generator=g)); ln.bias.copy_(0.1 * torch.randn(c, generator=g))
    wa = (torch.randn(n * c, c, generator=g) / c ** 0.5).to(dev); wb = (torch.randn(n * c, c, generator=g) / c ** 0.5).to(dev)
    wo = (torch.randn(c, n * c, generator=g) / (n * c) ** 0.5).to(dev)
    x = torch.randn(*lead, c, generator=g).to(dev, torch.bfloat16)
    return ln, wa, wb, wo, x


def _T(PF, ln, wa, wb, wo):
    return PF.pack_transition_weights(ln_w=ln.weight, ln_b=ln.bias, w_a=wa, w_b=wb, w_out=wo, eps=ln.eps)


@sm90
@pytest.mark.parametrize("c,n,lead", [(128, 4, (1, 37, 37)), (64, 2, (1, 3, 29, 29)), (64, 4, (1, 5, 61))])
def test_update_masked_update_and_inplace_fold(c, n, lead):
    from opt_core.attn import pair_fused as PF
    ln, wa, wb, wo, x = _make(c, n, lead)
    T = _T(PF, ln, wa, wb, wo)
    ok, word = PF.transition_plan_words(x, T, residual=True, mask=torch.ones(lead, device="cuda"), variants=V2)
    assert ok and word == "v2_fold"
    mask = (torch.rand(lead, device="cuda") > 0.15).float()
    with torch.no_grad():
        u = PF.transition(x, T, variants=V2)
        um = PF.transition(x, T, mask=mask, binary_mask=True, variants=V2)
        z = x.clone(); ret = PF.transition(z, T, residual=True, mask=mask, binary_mask=True, out=z, variants=V2)
        ref_u, ref_um, ref_z = _stock(x, ln, wa, wb, wo), _stock(x, ln, wa, wb, wo, mask), _stock(x, ln, wa, wb, wo, mask, residual=x)
        f_u, f_z = _f64(x, ln, wa, wb, wo), _f64(x, ln, wa, wb, wo, mask, residual=x)
    assert ret.data_ptr() == z.data_ptr() and ret.shape == z.shape                                  # in place: the caller's tensor holds x + mask*u
    for ours, ref in ((u, ref_u), (um, ref_um), (z, ref_z)):
        d = _ulps(ours, ref)
        frac = (d > 0.55).double().mean().item()
        assert d.max().item() <= 2.6 and frac <= 2e-3, (c, n, d.max().item(), frac)                  # half-ulp units: within ~1 bf16 ulp on a small fraction
    assert _ulps(z, f_z).max().item() <= _ulps(ref_z, f_z).max().item() + 1.0                         # as close to float64 as the statements
    assert _ulps(u, f_u).pow(2).mean().item() <= 1.5 * _ulps(ref_u, f_u).pow(2).mean().item() + 1e-3
    with torch.no_grad():                                                                            # deterministic run to run
        z2 = x.clone(); PF.transition(z2, T, residual=True, mask=mask, binary_mask=True, out=z2, variants=V2)
    assert torch.equal(z, z2)


@sm90
def test_rows_not_tile_multiple_zero_rows_strided_input_and_masks():
    from opt_core.attn import pair_fused as PF
    ln, wa, wb, wo, x = _make(128, 4, (1, 19, 23))                                                  # 437 rows: not a multiple of the 64-row tile
    T = _T(PF, ln, wa, wb, wo)
    with torch.no_grad():
        bmask = torch.rand(1, 19, 23, device="cuda") > 0.3                                           # bool mask: binary by construction
        assert _ulps(PF.transition(x, T, mask=bmask, variants=V2), _stock(x, ln, wa, wb, wo, bmask.float())).max().item() <= 2.6
        gmask = torch.rand(1, 19, 23, device="cuda")                                                 # general-valued mask: epilogue multiply
        d = _ulps(PF.transition(x, T, mask=gmask, binary_mask=False, variants=V2), _stock(x, ln, wa, wb, wo, gmask))
        assert d.max().item() <= 3.0
        wide = torch.randn(1, 19, 23, 256, device="cuda", dtype=torch.bfloat16)                      # a channel-sliced view: unit channel stride, row stride 256
        xs = wide[..., :128]
        assert _ulps(PF.transition(xs, T, variants=V2), _stock(xs, ln, wa, wb, wo)).max().item() <= 3.0
        z0 = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
        assert PF.transition(z0, T, variants=V2).shape == (0, 128)


@sm90
def test_refusals_by_name():
    from opt_core.attn import pair_fused as PF
    ln, wa, wb, wo, x = _make(128, 4, (1, 8, 8))
    T = _T(PF, ln, wa, wb, wo)
    assert PF.transition_plan_words(x, T) == (True, "v1")                                    # no preference named: the v1 kernel, as every other caller
    assert PF.transition_plan_words(x, T, variants=V2) == (True, "v2_fold") and PF.transition_plan_words(x, T, variant="v2_fold") == (True, "v2_fold")
    ok, word = PF.transition_plan_words(x, T, variant="v1", mask=torch.ones(1, 8, 8, device="cuda"))
    assert not ok and word.startswith("mask:variant:v1")
    ok, word = PF.transition_plan_words(x.float(), T, residual=True)
    assert not ok and word.startswith("dtype:float32")
    with pytest.raises(PF.Unsupported) as e:
        PF.transition(x, T, mask=torch.ones(1, 8, 9, device="cuda"), variants=V2)
    assert e.value.reason.startswith("mask-rows")


@cuda
def test_ln_proj_pair_bias_head_major_planes():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bf16 MMA")
    from opt_core.kernels import ln_proj as K
    g = torch.Generator(device="cpu").manual_seed(1)
    for c, H, N in ((128, 16, 37), (64, 4, 29)):
        ln = torch.nn.LayerNorm(c).cuda(); w = (torch.randn(H, c, generator=g) / c ** 0.5).cuda()
        z = torch.randn(1, N, N, c, generator=g).cuda().to(torch.bfloat16)
        P = K.pack_pair_bias_weights(ln.weight.detach(), ln.bias.detach(), w, ln.eps, z.device)
        pb = K.pair_bias(z, P, out_layout="bhij", out_dtype=torch.bfloat16)
        assert pb.shape == (1, H, N, N) and pb.stride(-1) == 1 and pb.stride(-2) == K.ld_of(N) and K.ld_of(N) % 8 == 0
        with torch.autocast("cuda", torch.bfloat16):
            ref = torch.nn.functional.linear(torch.nn.functional.layer_norm(z, (c,), ln.weight, ln.bias, ln.eps), w).permute(0, 3, 1, 2)
        assert _ulps(pb, ref).max().item() <= 2.6
        zt = z.transpose(1, 2)                                                                       # a transposed view: served in place, no copy
        pbt = K.pair_bias(zt, P, out_layout="bhij", out_dtype=torch.bfloat16)
        assert torch.equal(pbt, pb.transpose(-1, -2).contiguous()[..., :N]) or _ulps(pbt, pb.transpose(-1, -2)).max().item() <= 1.0
        with pytest.raises(K.Unsupported):
            K.pair_bias(z[..., :48], K.pack_pair_bias_weights(ln.weight[:48], ln.bias[:48], w[:, :48], ln.eps, z.device))
