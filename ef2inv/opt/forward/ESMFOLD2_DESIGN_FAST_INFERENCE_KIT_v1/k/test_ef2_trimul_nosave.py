"""Unit tests for ef2_trimul_nosave (the no-save first pass of the checkpointed pair blocks on the opt_core provider's forward-only row;
numerics class fast).

    pytest -q k/test_ef2_trimul_nosave.py        (or the kit's pytest-free runner)

Random-initialised FoldingTrunk modules of the installed fork on a CUDA device (no checkpoint download). Claims:
  * words: enable() steps aside BY NAME (word:exact, cc_no_gain off sm_90, agk_absent) and installs nothing then; on sm_90 with opt_core's
    provider it is on, the sealed row refused by name on a stack without its prebuilt and the next row (v4) serving;
  * the mechanism is exact: with the provider row unreachable (every call handed inward) _NoSaveCheckpoint gives outputs AND d(pair)
    bitwise equal (torch.equal) to torch's non-reentrant checkpoint on the same kernels, and disable() restores torch's path bitwise;
  * with the row serving: the first passes are counted (first_pass == recompute == checkpointed blocks x passes, served == 2 x first_pass,
    ef2_trimul's fused counter short by exactly that), outputs / gradients inside the fast class (cosine >= 0.999, max|d| <= 5 % of scale),
    peak memory within one pair plane of torch's checkpoint.
"""
import os, sys, pytest, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers.models.esmfold2.modeling_esmfold2_common as C
import ef2_autograd_kernels as agk
import ef2_trimul as TM
import ef2_trimul_nosave as NS

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
sm90 = pytest.mark.skipif(not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (9, 0) or NS.PT is None,
                          reason="sm_90 + opt_core's trimul provider required (elsewhere the lever steps aside by name: test_words covers it)")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
AMP = torch.amp.autocast("cuda", dtype=torch.bfloat16)


class _Model(torch.nn.Module):
    def __init__(self, n_layers, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.folding_trunk = C.FoldingTrunk(n_layers=n_layers)
        for m in self.folding_trunk.modules():
            if isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.normal_(m.weight, 1.0, 0.1); torch.nn.init.normal_(m.bias, 0.0, 0.1)


def make_model(n_layers=4, seed=0, kernels="fused"):
    m = _Model(n_layers, seed).to(dev).eval().requires_grad_(False)
    m.folding_trunk.set_kernel_backend(C.BACKEND_FUSED if kernels == "fused" else None); m.folding_trunk.set_chunk_size(64)
    agk.enable(m, trimul=None, transition=("refround_lean" if kernels == "fused" else None), checkpoint="block")
    if kernels == "fused":
        TM.enable(m, variant="fused")
    return m


def inputs(n, seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    z = (torch.randn(1, n, n, 256, device=dev, generator=g) * 2.0).to(torch.bfloat16)
    mask = torch.ones(1, n, n, device=dev)
    gout = torch.randn(1, n, n, 256, device=dev, generator=g).to(torch.bfloat16)
    return z, mask, gout


def fwd_bwd(model, z, mask, gout, passes=2):
    p = z.detach().clone().requires_grad_(True)
    with torch.enable_grad(), AMP:
        cur = p
        for _ in range(passes):
            cur = model.folding_trunk(cur, pair_attention_mask=mask)
        (g,) = torch.autograd.grad((cur,), (p,), (gout,))
    return cur.detach().clone(), g.detach().clone()


@cuda
def test_words():
    m = make_model(2)
    h = NS.enable(m, word="exact")
    assert not h.on and h.reason.startswith("word:exact") and NS.describe(h).startswith("LEVER name=ef2_trimul_nosave state=stepped_aside reason=word:exact")
    assert "_agk_ckpt_fn" not in m.folding_trunk.__dict__ and not any(hasattr(t, "_ef2_nosave_inner_forward") for t in m.modules())
    NS.disable(m)
    bare = _Model(2, 0).to(dev).eval().requires_grad_(False)            # no agk patch: no checkpoint seam to hook
    hb = NS.enable(bare)
    cc = tuple(torch.cuda.get_device_capability(0))
    want = "agk_absent" if (cc == (9, 0) and NS.PT is not None and not NS._CORE_WHY) else (NS._CORE_WHY or "cc_no_gain:sm_%d%d" % cc)
    assert not hb.on and hb.reason == want, hb.reason
    assert NS.describe(None) == "LEVER name=ef2_trimul_nosave state=off" and NS.extra_bytes(bare, 800) == 0.0


@sm90
def test_on_words_and_rows():
    m = make_model(2)
    h = NS.enable(m)
    assert h.on and h.word == "fast" and h.row in NS.ROWS and h.canary.startswith("pass:") and h.n_tmu == 4 and h.ckpt, NS.describe(h)
    for row, why in h.refused.items():                                  # a row passed over is named with the provider's own word (e.g. tx_sm90a:no_prebuilt:<abi>)
        assert row in NS.ROWS and why and row != h.row
    assert NS.enable(m) is h                                            # idempotent
    assert NS.extra_bytes(m, 800) == 800 * 800 * 256 * 2
    line = NS.describe(h)
    assert line.startswith("LEVER name=ef2_trimul_nosave state=on word=fast row=%s served=0 first_pass=0 recompute=0" % h.row) and "cc=sm_90" in line
    assert h.rows == NS.ROWS == ("tx_sm90a", "esm_v61", "esm_v5_fwd", "v4")          # the preference order, asked by name; every row before the one that serves is named in refused
    assert list(h.refused) == list(NS.ROWS[:NS.ROWS.index(h.row)]), (h.row, h.refused)
    NS.disable(m)
    assert "_ef2_nosave_handle" not in m.__dict__ and "_agk_ckpt_fn" not in m.folding_trunk.__dict__ and not any(hasattr(t, "_ef2_nosave_inner_forward") for t in m.modules())
    hv = NS.enable(m, rows=("v4",))                                     # a caller's own candidate list binds that row by name; the module's order is untouched
    assert hv.on and hv.row == "v4" and hv.rows == ("v4",) and not hv.refused and NS.ROWS[0] == "tx_sm90a", NS.describe(hv)
    NS.disable(m)


@sm90
@pytest.mark.parametrize("kernels", ["fused", "stock"])
def test_mechanism_bitwise_when_the_row_is_unreachable(kernels):
    """_NoSaveCheckpoint vs torch.utils.checkpoint on the SAME kernels (MIN_TOKENS above the pair size: every call handed inward)."""
    m = make_model(4, kernels=kernels); z, mask, gout = inputs(160)
    out0, g0 = fwd_bwd(m, z, mask, gout)
    keep = NS.MIN_TOKENS; NS.MIN_TOKENS = 10 ** 9
    try:
        h = NS.enable(m); assert h.on
        out1, g1 = fwd_bwd(m, z, mask, gout)
        st = h.stats
        assert st["served"] == 0 and st["first_pass"] == 8 and st["recompute"] == 8 and st["inner"] >= 32, st
    finally:
        NS.MIN_TOKENS = keep
    assert torch.equal(out0, out1) and torch.equal(g0, g1)
    NS.disable(m)
    out2, g2 = fwd_bwd(m, z, mask, gout)
    assert torch.equal(out0, out2) and torch.equal(g0, g2)


@sm90
def test_row_serving_counts_class_and_memory():
    m = make_model(4); z, mask, gout = inputs(256)
    th = m.__dict__["_ef2_trimul_handle"]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    out0, g0 = fwd_bwd(m, z, mask, gout); peak0 = torch.cuda.max_memory_allocated()
    fused0 = int(th.stats["served"])
    h = NS.enable(m); assert h.on
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    out1, g1 = fwd_bwd(m, z, mask, gout); peak1 = torch.cuda.max_memory_allocated()
    st = h.stats
    assert st["first_pass"] == 8 == st["recompute"] and st["served"] == 16 and st["refused_calls"] == 0, st
    assert int(th.stats["served"]) - fused0 == 16                      # the recomputes only: today's 32 per fwd+bwd less the 16 first-pass tri-muls the row served
    cos = torch.nn.functional.cosine_similarity(g0.float().flatten(), g1.float().flatten(), dim=0).item()
    assert cos >= 0.999, cos
    assert (out0.float() - out1.float()).abs().max().item() <= 0.05 * out0.float().abs().max().item()
    assert (g0.float() - g1.float()).abs().max().item() <= 0.05 * g0.float().abs().max().item()
    assert peak1 <= peak0 + 2 * NS.extra_bytes(m, 256) + (64 << 20), (peak0, peak1)
    NS.disable(m)
