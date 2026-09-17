"""GPU numerics of the carried `templ_embed` kernels (opt_core.kernels.templ_embed): `embed` within the stock bf16 chain's own error of the fp32
statements (random features, several chains, masked tokens, distinct templates, N not a tile multiple, strided feature views), `tail` bitwise
equal to the stock bf16 statements for T in {1, 2, 4} and within one bf16 rounding at T = 3 / 5, a stride-0 template axis == the materialised
expansion bitwise, c_z 64 / 128 / 256, run-to-run determinism, the high-offset region of a T = 4, N = 3400 problem (template offsets past 2^31
elements) bitwise equal to the same kernels on token crops, and the named refusals."""
import pytest

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _K():
    pytest.importorskip("triton")
    from opt_core.kernels import route
    route("templ_embed")
    import importlib
    return importlib.import_module("templ_embed")


def _problem(T, N, seed=0, chains=(0.5, 0.3, 0.2), cz=128):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = "cuda"
    bins = torch.randint(0, 39, (T, N, N), generator=g, dtype=torch.uint8).to(dev)                        # built on the device: the fp32 one-hot of a
    dg = torch.zeros(T, N, N, 39, device=dev).scatter_(-1, bins.long().unsqueeze(-1), 1.0); del bins       # T=4, N=3400 problem is 7 GB (no host copy)
    uv = torch.randn(T, N, N, 3, generator=g).to(dev)
    pbm = (torch.rand(T, N, generator=g) > 0.2).float().to(dev); bbm = (torch.rand(T, N, generator=g) > 0.2).float().to(dev)
    sizes = [int(round(f * N)) for f in chains]; sizes[-1] = N - sum(sizes[:-1])
    asym = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)]).to(dev)
    restype = torch.nn.functional.one_hot(torch.randint(0, 32, (T, N), generator=g), 32).float().to(dev)
    zp32 = torch.randn(N, N, 64, generator=g).to(dev)
    W = {k: (torch.randn(*s, generator=g) * sc).to(dev) for k, s, sc in (("w_dgram", (64, 39), 0.3), ("w_pb", (64, 1), 0.5), ("w_aa1", (64, 32), 0.3), ("w_aa2", (64, 32), 0.3),
                                                                    ("w_x", (64, 1), 0.5), ("w_y", (64, 1), 0.5), ("w_z", (64, 1), 0.5), ("w_bb", (64, 1), 0.5), ("w_t", (cz, 64), 0.2))}
    return dg, uv, pbm, bbm, asym, restype, zp32, W


def _served_embed(K, dg, uv, pbm, bbm, asym, restype, zp32, W):
    bf = torch.bfloat16
    P = K.pack_weights(**{k: v for k, v in W.items() if k not in ("w_aa1", "w_aa2")})
    ri = (restype.to(bf) @ W["w_aa1"].to(bf).t()); rj = (restype.to(bf) @ W["w_aa2"].to(bf).t())          # the module's own Linears under autocast
    return K.embed(dg, uv, pbm, bbm, asym, ri.contiguous(), rj.contiguous(), zp32.to(bf), P), P


@cuda
@pytest.mark.parametrize("T,N", [(1, 64), (4, 100), (2, 257), (5, 129)])
def test_embed_within_the_stock_bf16_chains_error(T, N):
    K = _K()
    dg, uv, pbm, bbm, asym, restype, zp32, W = _problem(T, N, seed=T * 1000 + N)
    Wf = {k: v for k, v in W.items() if k != "w_t"}
    ref32 = K.reference_embed(dg, uv, pbm, bbm, asym, restype, zp32.to(torch.bfloat16).float(), **Wf, dtype=torch.float32)
    stock = K.reference_embed(dg, uv, pbm, bbm, asym, restype, zp32.to(torch.bfloat16), **Wf, dtype=torch.bfloat16)
    got, _ = _served_embed(K, dg, uv, pbm, bbm, asym, restype, zp32, W)
    assert got.shape == (T, N, N, 64) and got.dtype == torch.bfloat16 and bool(torch.isfinite(got.float()).all())
    e_stock = (stock.float() - ref32).abs().max().item(); e_got = (got.float() - ref32).abs().max().item()
    assert e_got <= 1.05 * e_stock + 1e-6, (e_got, e_stock)                       # one rounding vs nine: at or below the stock chain's error
    got2, _ = _served_embed(K, dg, uv, pbm, bbm, asym, restype, zp32, W)
    assert torch.equal(got, got2)                                                  # deterministic


@cuda
def test_embed_serves_strided_feature_views_in_place():
    K = _K()
    T, N = 2, 96
    dg, uv, pbm, bbm, asym, restype, zp32, W = _problem(T, N, seed=7)
    base, _ = _served_embed(K, dg, uv, pbm, bbm, asym, restype, zp32, W)
    big_dg = torch.zeros(T, N + 5, N + 3, 39, device="cuda"); big_dg[:, 2:N + 2, 1:N + 1] = dg
    big_uv = torch.zeros(T, N + 1, N + 7, 3, device="cuda"); big_uv[:, :N, 4:N + 4] = uv
    got, _ = _served_embed(K, big_dg[:, 2:N + 2, 1:N + 1], big_uv[:, :N, 4:N + 4], pbm, bbm, asym, restype, zp32, W)
    assert torch.equal(base, got)


@cuda
@pytest.mark.parametrize("T", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("cz", [64, 128, 256])
def test_tail_against_the_stock_bf16_statements(T, cz):
    K = _K()
    N = 150
    g = torch.Generator(device="cpu").manual_seed(T * 10 + cz)
    s = torch.randn(T, N, N, 64, generator=g).to("cuda", torch.bfloat16)
    w_t = (torch.randn(cz, 64, generator=g) * 0.2).to("cuda")
    P = K.pack_weights(**{k: v for k, v in _problem(1, 8, cz=cz)[7].items() if k not in ("w_aa1", "w_aa2", "w_t")}, w_t=w_t)
    got = K.tail(s, P["wt"])
    stock = K.reference_tail(s, w_t.to(torch.bfloat16), dtype=torch.bfloat16)
    assert got.shape == (N, N, cz) and got.dtype == torch.bfloat16
    if T in (1, 2, 4):
        assert torch.equal(got, stock), (got.float() - stock.float()).abs().max()
    else:
        ref32 = K.reference_tail(s, w_t.to(torch.bfloat16).float(), dtype=torch.float32)
        e_stock = (stock.float() - ref32).abs().max().item(); e_got = (got.float() - ref32).abs().max().item()
        assert e_got <= 1.5 * e_stock + 1e-6, (e_got, e_stock)
    assert torch.equal(got, K.tail(s, P["wt"]))


@cuda
def test_tail_reads_a_stride0_template_axis_in_place():
    K = _K()
    T, N = 4, 200
    g = torch.Generator(device="cpu").manual_seed(11)
    one = torch.randn(1, N, N, 64, generator=g).to("cuda", torch.bfloat16)
    w_t = (torch.randn(128, 64, generator=g) * 0.2).to("cuda")
    wt = w_t.t().to(torch.bfloat16).contiguous()
    view = one.expand(T, N, N, 64)
    assert view.stride(0) == 0
    assert torch.equal(K.tail(view, wt), K.tail(view.contiguous(), wt))
    assert torch.equal(K.tail(view, wt), K.reference_tail(view, w_t.to(torch.bfloat16), dtype=torch.bfloat16))


@cuda
def test_high_offsets_past_int32_equal_token_crops_bitwise():
    """T = 4 distinct templates at N = 3400: template plane offsets reach 4 * 3400^2 * 64 = 2.96e9 > 2^31 elements. The first and the last 512-token
    diagonal blocks of the full-size outputs equal the same kernels run on those crops (needs ~14 GB free)."""
    K = _K()
    free, _total = torch.cuda.mem_get_info()
    if free < 16 * 2 ** 30:
        pytest.skip("needs 16 GB free on the device")
    T, N, C = 4, 3400, 512
    dg, uv, pbm, bbm, asym, restype, zp32, W = _problem(T, N, seed=5, chains=(0.6, 0.4))
    full, P = _served_embed(K, dg, uv, pbm, bbm, asym, restype, zp32, W)
    tail_full = K.tail(full, P["wt"])
    for sl in (slice(0, C), slice(N - C, N)):
        crop, _ = _served_embed(K, dg[:, sl, sl], uv[:, sl, sl], pbm[:, sl].contiguous(), bbm[:, sl].contiguous(), asym[sl].contiguous(), restype[:, sl], zp32[sl, sl], W)
        assert torch.equal(full[:, sl, sl], crop)
        assert torch.equal(K.tail(full[:, sl, sl], P["wt"]), K.tail(crop.contiguous(), P["wt"]))      # a strided [T, C, C, 64] view of the big tensor, template stride N*N*64
    del full, tail_full


@cuda
def test_named_refusals():
    K = _K()
    dg, uv, pbm, bbm, asym, restype, zp32, W = _problem(2, 40)
    got, P = _served_embed(K, dg, uv, pbm, bbm, asym, restype, zp32, W)
    bf = torch.bfloat16
    ri = torch.zeros(2, 40, 64, device="cuda", dtype=bf); zp = zp32.to(bf)
    with pytest.raises(K.Unsupported) as e:
        K.embed(dg[..., :38], uv, pbm, bbm, asym, ri, ri, zp, P)
    assert e.value.event == "shape:dgram"
    with pytest.raises(K.Unsupported) as e:
        K.embed(dg, uv, pbm[:, :39], bbm, asym, ri, ri, zp, P)
    assert e.value.event == "shape:pbm"
    with pytest.raises(K.Unsupported) as e:
        K.embed(dg, uv, pbm, bbm, asym, ri.float(), ri.float(), zp, P)
    assert e.value.event == "dtype:tables"
    with pytest.raises(K.Unsupported) as e:
        K.embed(dg.transpose(-1, -2).contiguous().transpose(-1, -2), uv, pbm, bbm, asym, ri, ri, zp, P)
    assert e.value.event == "stride:dgram"
    with pytest.raises(K.Unsupported) as e:
        K.tail(got.float(), P["wt"])
    assert e.value.event == "dtype:stack"
    with pytest.raises(K.Unsupported) as e:
        K.tail(got, torch.zeros(64, 512, device="cuda", dtype=bf))
    assert e.value.event == "dims:w_t"
    with pytest.raises(K.Unsupported) as e:
        K.tail(got.cpu(), P["wt"])
    assert e.value.event == "device:cpu"


@cuda
def test_lever_serves_and_finishes_a_refused_tail_on_the_stock_statements(monkeypatch):
    """opt_core.of3_trunk.templ_embed end to end on a stub TemplateEmbedderAllAtom: a served pass equals the stock statements' class; a stack that
    hands back fp32 makes the tail kernel refuse `dtype:stack` -> the pass finishes on the stock tail, counted `tail:dtype:stack`."""
    import sys
    import types
    K = _K()
    from opt_core.of3_trunk import templ_embed as TE
    saved = {k: getattr(TE, k) for k in TE.CONFIGURABLE}
    state0 = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in TE.STATE.items()}
    T, N, cz = 3, 70, 128
    dg, uv, pbm, bbm, asym, restype, zp32, W = _problem(T, N, seed=21, cz=cz)
    lin = lambda w: torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="cuda").requires_grad_(False)     # noqa: E731

    class PE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            names = dict(dgram_linear="w_dgram", pseudo_beta_mask_linear="w_pb", aatype_linear_1="w_aa1", aatype_linear_2="w_aa2", x_linear="w_x", y_linear="w_y",
                         z_linear="w_z", backbone_mask_linear="w_bb")
            for n, k in names.items():
                m = lin(W[k]); m.weight.data.copy_(W[k]); setattr(self, n, m)
            self.linear_z = lin(torch.zeros(64, cz)); torch.nn.init.normal_(self.linear_z.weight, std=0.1)
            self.layer_norm_z = torch.nn.LayerNorm(cz, device="cuda")

    class Stack(torch.nn.Module):
        dtype_out = torch.bfloat16

        def forward(self, t, mask, **kw):
            return (t * 1.5).to(self.dtype_out)

    mod_ns = types.ModuleType("y_engine.template_module")

    class TemplateEmbedderAllAtom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.template_pair_embedder = PE(); self.template_pair_stack = Stack(); self.linear_t = lin(W["w_t"]); self.linear_t.weight.data.copy_(W["w_t"])

        def forward(self, batch, z, pair_mask, chunk_size=None, _mask_trans=True, use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                    use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False):          # the stock statements (reference_embed's chain under autocast)
            pe = self.template_pair_embedder
            a = K.reference_embed(batch["template_distogram"][0], batch["template_unit_vector"][0], batch["template_pseudo_beta_mask"][0], batch["template_backbone_frame_mask"][0],
                                  batch["asym_id"][0], batch["template_restype"][0], pe.linear_z(pe.layer_norm_z(z))[0],
                                  w_dgram=pe.dgram_linear.weight, w_pb=pe.pseudo_beta_mask_linear.weight, w_aa1=pe.aatype_linear_1.weight, w_aa2=pe.aatype_linear_2.weight,
                                  w_x=pe.x_linear.weight, w_y=pe.y_linear.weight, w_z=pe.z_linear.weight, w_bb=pe.backbone_mask_linear.weight, dtype=torch.bfloat16)[None]
            t = self.template_pair_stack(a, pair_mask[..., None, :, :])
            return self.linear_t(torch.relu(torch.sum(t, dim=-4) / a.shape[-4]))
    mod_ns.TemplateEmbedderAllAtom = TemplateEmbedderAllAtom
    monkeypatch.setitem(sys.modules, "y_engine.template_module", mod_ns)
    monkeypatch.setattr(TE.atexit, "register", lambda f: None)
    try:
        TE.configure(PREFIX="[y-kit/templ_embed]", ENV="Y_TEMPL_EMBED", M_TEMPLATE="y_engine.template_module")
        m = TemplateEmbedderAllAtom().eval()
        stock_fwd = TemplateEmbedderAllAtom.forward
        TE.install({"Y_TEMPL_EMBED": "1"})
        batch = {"template_distogram": dg[None], "template_unit_vector": uv[None], "template_pseudo_beta_mask": pbm[None], "template_backbone_frame_mask": bbm[None],
                 "template_restype": restype[None], "asym_id": asym[None]}
        z = torch.randn(1, N, N, cz, device="cuda"); pm = torch.ones(1, N, N, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            served = m(batch, z, pm)
            stock = stock_fwd(m, batch, z, pm)
        assert TE.STATE["served"] == 1 and TE.STATE["fallback"] == {} and TE.STATE["first"] == "3x70"
        assert served.shape == stock.shape == (1, N, N, cz) and served.dtype == stock.dtype == torch.bfloat16
        err = (served.float() - stock.float()).abs().max().item(); scale = stock.float().abs().max().item()
        assert err <= 0.05 * scale, (err, scale)                                              # same class (the embedding rounds once, stock nine times)
        Stack.dtype_out = torch.float32                                                         # a stack handing back fp32: the tail kernel refuses by name
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            served2 = m(batch, z, pm)
        assert TE.STATE["served"] == 2 and TE.STATE["fallback"] == {"tail:dtype:stack": 1} and served2.shape == (1, N, N, cz)
        assert "fallback=tail:dtype:stack:1" in TE.census_line()
    finally:
        Stack.dtype_out = torch.bfloat16
        TE.configure(**saved); TE.STATE.clear(); TE.STATE.update(state0); TE._K["mod"] = None
