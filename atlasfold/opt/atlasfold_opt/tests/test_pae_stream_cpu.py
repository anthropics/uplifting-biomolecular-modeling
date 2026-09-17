import torch

def test_pae_stream_equals_stock_cpu():
    from atlasfold.model.network.confidence_head import ConfidenceHead_Monomer as ConfidenceHead
    from atlasfold.model.utils import confidence_metrics as CM
    torch.manual_seed(0)
    B, N, L = 1, 3, 23
    head = ConfidenceHead(channel_s=384, channel_z=128, num_blocks=1).eval()
    for p in head.parameters():                      # heads are zero-initialised in places; randomise so the check is not trivially 0 == 0
        torch.nn.init.normal_(p, std=0.05)
    batch = {"aatype": torch.nn.functional.one_hot(torch.randint(0, 21, (B, L)), 21).float(), "seq_mask": torch.ones(B, L, dtype=torch.bool), "pseudo_beta": torch.randint(0, 5, (B, L))}
    batch["seq_mask"][:, -2:] = False
    s = torch.randn(B, L, 384); z = torch.randn(B, L, L, 128); x = torch.randn(B, N, L, 14, 3) * 6
    mask = batch["seq_mask"].unsqueeze(1)
    with torch.no_grad():
        ref = head(batch, s, z, x, "torch")
        pae0 = CM.compute_pae(**ref["pae"], mask=mask); ptm0 = CM.compute_ptm(**ref["pae"], mask=mask); pl0 = CM.compute_plddt(**ref["plddt"], mask=mask)
        from atlasfold_opt.hooks import pae as P
        ins = P.install("exact", "atlasfold-opt", {})
        try:
            out = head(batch, s, z, x, "torch")
            pae1 = CM.compute_pae(**out["pae"], mask=mask); ptm1 = CM.compute_ptm(**out["pae"], mask=mask); pl1 = CM.compute_plddt(**out["plddt"], mask=mask)
        finally:
            type(head).forward = type(head).forward.__wrapped_stock__
            import importlib; importlib.reload(CM)
    assert ins.applied and torch.equal(pae0, pae1) and torch.equal(ptm0, ptm1) and torch.equal(pl0, pl1), (float((pae0 - pae1).abs().max()), float((ptm0 - ptm1).abs().max()))
    assert torch.equal(ref["experimentally_resolved"]["logits"], out["experimentally_resolved"]["logits"])
