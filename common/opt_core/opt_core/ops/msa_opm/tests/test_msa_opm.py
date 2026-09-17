"""opt_core.ops.msa_opm: the launch-row tables are self-consistent on every card (CPU), and the fused outer-product mean is in the
tolerance class of the stock statements it replaces (GPU, bf16 autocast; skipped without CUDA / triton). Bitwise claims are never made
for this op: it is a fast-class cell (fp32 accumulation over the MSA depth in a tile order that differs from cuBLAS)."""
import pytest


def _mod():
    pytest.importorskip("torch"); pytest.importorskip("triton")
    from opt_core.ops import msa_opm as O
    return O


def test_rows_are_named_variants_and_fit_the_card():
    O = _mod()
    for table in (O._CFG, O._CFG_NO_TMA):
        for key, row in table.items():
            assert O.cfg_name(row) is not None, (key, row)
    for cc, rows in O._CFG_BY_CC.items():
        for (form, cz), row in rows.items():
            assert O.cfg_name(row) is not None, (cc, form, cz, row)
            if cc == (8, 0):                                   # sm_80: pointer loads only (TMA is sm_90), tiles inside the 163 KB opt-in shared-memory limit
                assert not row.get("TMA", False)
                smem = row["num_stages"] * (row["BI"] * 32 * row["BK"] + row["BK"] * row["BJ"] * 32) * 2
                assert smem <= 166912, (row, smem)
    cfg, word = O.cfg_for("mask_norm", 128)
    assert isinstance(cfg, dict) and (word in ("default", "no_tma") or word.startswith("cc") or word.startswith("override:")), word
    assert set(O.FPF_META) == {"outer_product_mean"}


def _schema(torch, form, c_in=64, ch=32, cz=128, seed=0):
    torch.manual_seed(seed)
    ns = torch.nn.Module()
    if form == "mask_norm":
        ns.norm = torch.nn.LayerNorm(c_in); ns.proj_a = torch.nn.Linear(c_in, ch, bias=False); ns.proj_b = torch.nn.Linear(c_in, ch, bias=False)
        ns.proj_o = torch.nn.Linear(ch * ch, cz); ns.c_hidden = ch
    else:
        ns.layer_norm = torch.nn.LayerNorm(c_in); ns.linear_1 = torch.nn.Linear(c_in, ch); ns.linear_2 = torch.nn.Linear(c_in, ch)
        ns.linear_out = torch.nn.Linear(ch * ch, cz); ns.eps = 1e-3
    for p in ns.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return ns.cuda().eval()


def _ref_mask_norm(torch, mod, m, mask):
    """The stock statements of the mask-norm form (unchunked path), under the caller's autocast."""
    mask_b = mask.unsqueeze(-1).to(m)
    ln = mod.norm(m)
    a = mod.proj_a(ln) * mask_b
    b = mod.proj_b(ln) * mask_b
    z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())
    z = z.reshape(*z.shape[:3], -1)
    num_mask = (mask_b[:, :, None, :] * mask_b[:, :, :, None]).sum(1).clamp(min=1)
    z = z / num_mask
    return mod.proj_o(z.to(m))


@pytest.mark.parametrize("S,N", [(64, 48), (300, 130)])
def test_mask_norm_form_is_in_the_tolerance_class(S, N):
    O = _mod()
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    mod = _schema(torch, "mask_norm")
    g = torch.Generator(device="cuda").manual_seed(1)
    m = torch.randn(1, S, N, 64, device="cuda", generator=g)
    mask = (torch.rand(1, S, N, device="cuda", generator=g) > 0.1).float()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = _ref_mask_norm(torch, mod, m, mask).float()
        got = O.forward_mask_norm(mod, m, mask, None).float()
        got2 = O.forward_mask_norm(mod, m, mask, None).float()
    assert got.shape == ref.shape
    assert torch.equal(got, got2), "run-to-run bitwise"
    err = (got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert float(err) < 2e-2, float(err)
