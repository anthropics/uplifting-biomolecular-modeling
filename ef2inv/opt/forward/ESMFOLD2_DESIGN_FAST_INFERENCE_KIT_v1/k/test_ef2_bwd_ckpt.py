"""Unit tests for ef2_bwd_ckpt (memory-planned trunk checkpoint policy; numerics class exact).

    pytest -q k/test_ef2_bwd_ckpt.py            (or the kit's pytest-free runner)

GPU tests use random-initialised FoldingTrunk modules (no checkpoint download). The exact claim: every policy the lever can install
('none' | 'ckpt:m' | 'block', planned or forced) gives outputs AND input-gradients bitwise equal (torch.equal) to stock's every-block
checkpointing, for both kernel sets the lever plans for (stock reference kernels under agk's policy-only patch; agk3 kernels).
"""
import os, sys, pytest, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers.models.esmfold2.modeling_esmfold2_common as C
import ef2_autograd_kernels as agk
import ef2_bwd_ckpt as L

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
AMP = torch.amp.autocast("cuda", dtype=torch.bfloat16)
GIB = float(2 ** 30)


def make_trunk(n_layers=5, seed=0):
    torch.manual_seed(seed)
    tr = C.FoldingTrunk(n_layers=n_layers).to(dev).eval().requires_grad_(False)
    for m in tr.modules():
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.normal_(m.weight, 1.0, 0.1); torch.nn.init.normal_(m.bias, 0.0, 0.1)
    tr.set_kernel_backend(None); tr.set_chunk_size(None)
    return tr


def fwd_bwd(trunk, z, mask, gout, passes=2):
    zz = z.detach().clone().requires_grad_(True)
    with torch.enable_grad(), AMP:
        cur = zz
        for _ in range(passes):                     # the design step's two grad-enabled trunk passes (no detach between them)
            cur = trunk(cur, pair_attention_mask=mask)
        (cur.float() * gout).sum().backward()
    return cur.detach().clone(), zz.grad.detach().clone()


def test_plan_arithmetic_monotone_and_bounded():
    big = L.plan(500, 2, "agk3", budget_bytes=1e15, total_bytes=80 * GIB)
    assert big["policy"] == "none" and big["kept_per_pass"] == 24
    tiny = L.plan(500, 2, "agk3", budget_bytes=1.0, total_bytes=80 * GIB)
    assert tiny["policy"] == "block" and tiny["kept_per_pass"] == 0
    last = -1
    for budget_gib in range(20, 90, 5):             # more budget never keeps fewer blocks; the estimate never exceeds the budget it was planned for
        r = L.plan(700, 2, "agk3", budget_bytes=budget_gib * GIB, base_bytes=19 * GIB, total_bytes=80 * GIB)
        assert r["kept_per_pass"] >= last; last = r["kept_per_pass"]
        assert r["kept_per_pass"] == 0 or r["est_peak_bytes"] <= budget_gib * GIB
        m = {"none": 0, "block": 24}.get(r["policy"])
        m = int(r["policy"].split(":")[1]) if m is None else m
        assert m == 24 - r["kept_per_pass"]
    s = L.plan(500, 2, "stock", budget_bytes=72.9 * GIB, base_bytes=18.7 * GIB, total_bytes=80 * GIB)
    a = L.plan(500, 2, "agk3", budget_bytes=72.9 * GIB, base_bytes=19.0 * GIB, total_bytes=80 * GIB)
    assert a["kept_per_pass"] > s["kept_per_pass"] > 0          # the leaner kernel set keeps more blocks in the same budget
    assert L.estimate_peak_bytes(700, 3, 2, "stock", base_bytes=18.8 * GIB) < L.estimate_peak_bytes(700, 4, 2, "stock", base_bytes=18.8 * GIB)


@cuda
@pytest.mark.parametrize("kernels", ["stock", "agk3"])
def test_every_policy_is_bitwise_equal_to_block_checkpointing(kernels):
    N, nl = 48, 5
    trunk = make_trunk(nl)
    torch.manual_seed(1)
    z = torch.randn(1, N, N, 256, device=dev).to(torch.bfloat16)
    mask = torch.ones(1, N, N, device=dev); mask[:, -5:, :] = 0; mask[:, :, -5:] = 0
    gout = torch.randn(1, N, N, 256, device=dev)
    kw = dict(trimul=None, transition=None) if kernels == "stock" else dict(trimul="bmm2", transition="refround_lean")
    agk.enable(trunk, checkpoint="block", **kw)
    want = {"stock": "refstock", "agk3": "agk3"}[kernels]                          # make_trunk's kernel backend is None: the stock blocks run the fork's REFERENCE triangle multiplication (row 'refstock'), not cuEquivariance ('stock')
    assert L.kernels_of(trunk) == want and L.n_blocks_of(trunk) == nl
    ref_out, ref_g = fwd_bwd(trunk, z, mask, gout)                              # stock behaviour: every block checkpointed
    for pol in ("none", "ckpt:2", f"ckpt:{nl - 1}", "block", None):              # None = planned from free memory (whatever it picks)
        rec = L.enable(trunk, n_tokens=N, num_passes=2, **({"policy": pol} if pol else {}))
        assert rec["applied"] and rec["kernels"] == want
        got_pol = trunk._agk_cfg.checkpoint if hasattr(trunk, "_agk_cfg") else None
        assert got_pol in (rec["policy"],)
        out, g = fwd_bwd(trunk, z, mask, gout)
        assert torch.equal(out, ref_out), f"{kernels} policy {rec['policy']}: trunk output not bitwise"
        assert torch.equal(g, ref_g), f"{kernels} policy {rec['policy']}: input gradient not bitwise"
        L.disable(trunk)
        assert trunk._agk_cfg.checkpoint == "block" and not hasattr(trunk, "_bwd_ckpt")
    agk.disable(trunk)


@cuda
def test_enable_requires_agk_and_disable_is_state_neutral():
    import ef2_state_guard as sg
    trunk = make_trunk(3)
    try:                                                                         # (pytest.raises is not in the kit's pytest-free runner)
        L.enable(trunk, n_tokens=64)                                             # agk not enabled: refused by name, nothing installed
        raise AssertionError("enable without agk must raise PlanError")
    except L.PlanError:
        pass
    assert not hasattr(trunk, "_bwd_ckpt")
    agk.enable(trunk, trimul=None, transition=None, checkpoint="ckpt:1")
    ref = sg.snapshot()
    rec = L.enable(trunk, n_tokens=64, policy="none")
    assert trunk._agk_cfg.checkpoint == "none" and "policy=none" in L.describe(trunk)
    assert not sg.diff(sg.snapshot(), ref), "enable changed process-global state"
    L.disable(trunk)
    assert trunk._agk_cfg.checkpoint == "ckpt:1" and L.describe(trunk).endswith("state=off")
    assert not sg.diff(sg.snapshot(), ref), "disable changed process-global state"
    agk.disable(trunk)
