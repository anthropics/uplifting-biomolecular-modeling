"""fpf_trimul's tile-height schedule on a GPU: a stage tile scheduled to 128 / 256 / 512 rows computes the SAME BITS as the cell's own tile
(only the row blocking differs), and the TM-K3 exact line serves — bit-exact to cuEquivariance's fused TriMul — at sizes the cells' own
64 / 128-row tiles cannot launch (N >= 2048 at C = 128).  Needs CUDA (sm_80+) and triton; the cuEquivariance comparison needs cuequivariance_torch."""
import math
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
    pytest.skip("needs an sm_80+ GPU", allow_module_level=True)
os.environ.setdefault("FPF_TRIMUL_MODE", "exact")


@pytest.fixture(scope="module")
def mods():
    from opt_core import kernels as CK, trimul as T
    CK.route("fpf_trimul")
    import fpf_trimul.kernels as K, fpf_trimul.trimul as Tm
    assert Tm._MODE == "exact", Tm._MODE
    return T, K, Tm


def weights(C, dtype=torch.float32, seed=1234):
    """Engine-shaped TriMul parameters (fp32, as a bf16-autocast engine holds them): LN_in, [a|b] projection + gate [2C, C], LN_out, out projection + gate."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    tn = lambda shape, std: torch.randn(shape, generator=g) * std
    w = dict(ln_in_w=1.0 + 0.1 * tn((C,), 1.0), ln_in_b=0.05 * tn((C,), 1.0), p_in=tn((2 * C, C), 1.0 / math.sqrt(C)), g_in=tn((2 * C, C), 1.0 / math.sqrt(C)),
             ln_out_w=1.0 + 0.1 * tn((C,), 1.0), ln_out_b=0.05 * tn((C,), 1.0), p_out=tn((C, C), 0.5 / math.sqrt(C)), g_out=tn((C, C), 1.0 / math.sqrt(C)))
    return {k: v.cuda().to(dtype).contiguous() for k, v in w.items()}


def z_of(N, C, dtype=torch.float32, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed + N)
    return (torch.randn(N, N, C, generator=g) * 8.0).cuda().to(dtype).contiguous()


def packed(K, w, cdt):
    """GEMM weights in the compute dtype, LN weights native — the exact line's packing (opt_core.trimul.tmk3)."""
    return K.pack_weights(w["ln_in_w"], w["ln_in_b"], w["p_in"].to(cdt), w["g_in"].to(cdt), w["ln_out_w"], w["ln_out_b"], w["p_out"].to(cdt), w["g_out"].to(cdt))


def forward(K, Tm, z, w, cdt, cfg, direction="outgoing"):
    return K.trimul_forward(z, direction, None, packed(K, w, cdt), eps=1e-5, residual=False, cfg=cfg, contract=Tm.CONTRACT, cdt=cdt)


def wide(Tm, cfg, stage, BM, elem, dual_x):
    """cfg with `stage` forced to the schedule's BM-row tile (trimul._scheduled_tile: the accumulator rule's BN, that height's launch settings)."""
    return dict(cfg, **{stage: Tm._scheduled_tile(cfg[stage], BM, dict(Tm.GRID_SCHEDULE)[BM], elem, dual_x)})


@pytest.mark.parametrize("cdt,C", [(torch.bfloat16, 128), (torch.bfloat16, 256), (torch.float32, 384)])
def test_a_scheduled_tile_computes_the_cells_own_bits(mods, cdt, C):
    T, K, Tm = mods
    N, elem = 512, (2 if cdt == torch.bfloat16 else 4)
    w, z = weights(C), z_of(N, C, torch.float32 if cdt == torch.float32 else torch.float32)
    table = Tm._select_cfg(cdt, N, C)                                       # N = 512: the cell's own tiles (the schedule does nothing here)
    ref = forward(K, Tm, z, w, cdt, table)
    for stage, dual_x in (("A", False), ("C", True)):
        for BM in (128, 256, 512):
            if BM <= table[stage]["BM"]:
                continue
            out = forward(K, Tm, z, w, cdt, wide(Tm, table, stage, BM, elem, dual_x))
            assert torch.equal(out, ref), (str(cdt), C, stage, BM, float((out.float() - ref.float()).abs().max()))
    both = wide(Tm, wide(Tm, table, "A", 512, elem, False), "C", 512, elem, True)   # both stages at the tallest height at once
    assert torch.equal(forward(K, Tm, z, w, cdt, both), ref)


def test_the_exact_line_serves_n_2048_at_c_128_bit_exact_to_cuequivariance(mods):
    T, K, Tm = mods
    cuet = pytest.importorskip("cuequivariance_torch")
    N, C = 2048, 128
    w, z = weights(C), z_of(N, C)
    cfg = Tm._select_cfg(torch.bfloat16, N, C)
    assert (cfg["A"]["BM"], cfg["C"]["BM"]) == (128, 128) and K.launch_grid_y(N, C, cfg) <= K.CUDA_GRID_Y_MAX   # the cell's 64-row tiles cannot launch here; 128-row tiles serve
    mod = type("M", (), {})()
    prov = T.tmk3(lambda m: dict(ln_in_w=w["ln_in_w"], ln_in_b=w["ln_in_b"], w_ap=w["p_in"][:C], w_bp=w["p_in"][C:], w_ag=w["g_in"][:C], w_bg=w["g_in"][C:],
                                 ln_out_w=w["ln_out_w"], ln_out_b=w["ln_out_b"], w_o=w["p_out"], w_og=w["g_out"]), mode="exact")
    with torch.autocast("cuda", torch.bfloat16):
        call = T.Call(module=mod, z=z, mask=None, direction="outgoing", residual=False, orig=lambda: None)
        prov.eligible(call)                                                 # no Refused('launch_grid_y>65535')
        ours = prov.fn(call)
        stock = cuet.triangle_multiplicative_update(z[None], direction="outgoing", mask=None, norm_in_weight=w["ln_in_w"], norm_in_bias=w["ln_in_b"],
                                                    p_in_weight=w["p_in"], g_in_weight=w["g_in"], norm_out_weight=w["ln_out_w"], norm_out_bias=w["ln_out_b"],
                                                    p_out_weight=w["p_out"], g_out_weight=w["g_out"], eps=1e-5)[0]
    assert ours.shape == stock.shape and torch.equal(ours.to(stock.dtype), stock), float((ours.float() - stock.float()).abs().max())
