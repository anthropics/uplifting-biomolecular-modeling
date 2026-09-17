"""The `pair_transition` cell on the core's v2 kernel: role words from module paths; on a GPU
with the engine importable — a tagged SwiGLUTransition's update path equals the stock module within 2 bf16 ulps (mask folded), an untagged
instance is untouched, and PairBlock's tail written in place (x + mask*T(x) into x) equals `add(x, stock(x, mask), inplace=True)`; the
apb_trunk producer switch parses ln_proj | mm and refuses other words."""
import pytest

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="needs an sm_90 CUDA device")


def test_role_words():
    from openfold3_opt.cells import pairfused as P
    assert P._role_from_path("pairformer_stack.blocks.3.pair_transition", "pair") == "pairformer"
    assert P._role_from_path("aux_heads.confidence.pairformer.blocks.0.pair_transition", "pair") == "confidence"
    assert P._role_from_path("msa_module.blocks.1.pair_stack.pair_transition", "pair") == "msa_pair"
    assert P._role_from_path("template_embedder.pair_stack.blocks.0.pair_transition", "pair") == "template"
    assert P._role_from_path("msa_module.blocks.1.msa_transition", "msa") == "msa"


def test_apb_producer_switch_words():
    from openfold3_opt.cells import apb_trunk as AT
    assert AT.requested({AT.ENV: "1", AT.ENV_PRODUCER: "mm"}) and AT.requested({AT.ENV: "1", AT.ENV_PRODUCER: "ln_proj"}) and AT.requested({AT.ENV: "1"})
    with pytest.raises(ValueError):
        AT.requested({AT.ENV: "1", AT.ENV_PRODUCER: "fused"})


def _ulps(a, ref):
    a = a.double(); ref = ref.double()
    scale = torch.maximum(ref.abs(), ref.pow(2).mean().sqrt() * torch.ones_like(ref)) * 2.0 ** -8
    return ((a - ref).abs() / scale).max().item()


@cuda
def test_tagged_transition_update_path_and_pairblock_tail_in_place(monkeypatch):
    tr_mod = pytest.importorskip("openfold3.core.model.layers.transition")
    bb = pytest.importorskip("openfold3.core.model.latent.base_blocks")
    from openfold3.core.utils.tensor_utils import add
    from openfold3_opt.cells import pairfused as P
    from opt_core.attn import pair_fused as PF
    monkeypatch.setitem(P.STATE, "impl", "fpf"); monkeypatch.setitem(P.STATE, "strict", False)
    stock_forward = tr_mod.SwiGLUTransition.forward
    saved = (bb.PairBlock.forward, bb.PairBlock.__init__, bb.MSABlock.__init__, getattr(bb.PairBlock, "_of3v2_tagging", None))
    P._install_pair_transition(bb.PairBlock, tr_mod.SwiGLUTransition)
    try:
        torch.manual_seed(0)
        for c, n, lead, role in ((128, 4, (1, 41, 41), "pairformer"), (64, 2, (1, 2, 23, 23), "template"), (64, 4, (1, 7, 45), "msa")):
            tr = tr_mod.SwiGLUTransition(c_in=c, n=n).cuda().eval()
            with torch.no_grad():                                                                    # trained-like weights (the stock init zeroes linear_out)
                for prm in tr.parameters():
                    prm.copy_(torch.randn_like(prm) * (0.1 if prm.dim() == 1 else prm.shape[-1] ** -0.5))
                tr.layer_norm.weight.add_(1.0)
            x = torch.randn(*lead, c, device="cuda", dtype=torch.bfloat16)
            mask = (torch.rand(lead, device="cuda") > 0.2).float()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                u_stock = stock_forward(tr, x, mask=mask)
                u_untagged = tr(x, mask=mask)                                                        # no role tag: the stock forward runs
                assert torch.equal(u_untagged, u_stock)
                setattr(tr, P.ROLE_ATTR, role)
                u_ours = tr(x, mask=mask)                                                            # update path: masked update from the v2 kernel
            assert u_ours.dtype == torch.bfloat16 and _ulps(u_ours, u_stock) <= 2.6, (c, n)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):                            # PairBlock's tail: z + mask*T(z) in place vs add(z, stock, inplace=True)
                z_ref = add(x.clone(), stock_forward(tr, x, mask=mask), inplace=True)
                z = x.clone()
                T = tr._of3v2_trans_w
                PF.transition(z, T, residual=True, mask=mask, out=z, variants=P.TRANSITION_VARIANTS)
            assert _ulps(z, z_ref) <= 2.6, (c, n)
        roles = P.STATE["roles"]
        assert roles.get("pairformer:c128", 0) >= 1 and roles.get("template:c64", 0) >= 1 and roles.get("msa:c64", 0) >= 1 and P.STATE["fold"]["update"] >= 3
    finally:
        tr_mod.SwiGLUTransition.forward = stock_forward
        bb.PairBlock.forward, bb.PairBlock.__init__, bb.MSABlock.__init__ = saved[:3]
        if saved[3] is None and hasattr(bb.PairBlock, "_of3v2_tagging"):
            delattr(bb.PairBlock, "_of3v2_tagging")
