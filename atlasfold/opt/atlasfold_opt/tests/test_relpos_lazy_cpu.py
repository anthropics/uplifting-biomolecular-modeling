import torch

def test_relpos_lazy_replays_producers_exactly():
    from atlasfold.model.network.rel_pos_encoding import RelativePositionEncoding, AtomRelativePositionEncoding
    from atlasfold.model.network.primitives.linear import LinearNoBias
    from atlasfold.model.network.diffusion_transformer import PairConditioning
    from atlasfold.model import model as M
    torch.manual_seed(0); B, L = 1, 36
    batch = {"res_idx": torch.arange(L).unsqueeze(0), "asym_id": torch.zeros(B, L, dtype=torch.long), "entity_id": torch.zeros(B, L, dtype=torch.long),
             "sym_id": torch.zeros(B, L, dtype=torch.long), "seq_mask": torch.ones(B, L, dtype=torch.bool),
             "aatype": torch.nn.functional.one_hot(torch.randint(0, 21, (B, L)), 21).float()}
    class Fake:                                        # the two producer modules + the stock method, as AtlasFold carries them
        seq_rel_pos_encoding = RelativePositionEncoding(r_max=32, s_max=2); atom_rel_pos_encoding = AtomRelativePositionEncoding(max_r=4)
        compute_rel_pos_encoding = M.AtlasFold.compute_rel_pos_encoding
    b0 = dict(batch); Fake().compute_rel_pos_encoding(b0)            # stock: tensors
    lin = LinearNoBias(73, 128); pc = PairConditioning(128, (2, 4), 73).eval(); z = torch.randn(B, L, L, 128)
    with torch.no_grad():
        y0 = lin(b0["seq_rel_pos"]); pb0 = pc(b0, z); a0 = b0["atom_rel_pos"].unsqueeze(1)
    from atlasfold_opt.hooks import relpos as R
    ins = R.install("exact", "atlasfold-opt", {})
    try:
        Fake.compute_rel_pos_encoding = M.AtlasFold.compute_rel_pos_encoding   # the patched method
        b1 = dict(batch); Fake().compute_rel_pos_encoding(b1)
        assert isinstance(b1["seq_rel_pos"], R.LazyFeat) and isinstance(b1["atom_rel_pos"], R.LazyFeat)
        with torch.no_grad():
            y1 = lin(b1["seq_rel_pos"]); pb1 = pc(b1, z); a1 = b1["atom_rel_pos"].unsqueeze(1)
        assert torch.equal(b1["seq_rel_pos"].materialize(), b0["seq_rel_pos"]) and torch.equal(a0, a1)
        assert torch.equal(y0, y1) and torch.equal(pb0, pb1) and ins.applied
    finally:
        M.AtlasFold.compute_rel_pos_encoding = M.AtlasFold.compute_rel_pos_encoding.__wrapped_stock__
        from atlasfold.model.network.primitives import linear as LIN; from atlasfold.model.network import diffusion_transformer as DT
        if "forward" in LIN.LinearNoBias.__dict__: delattr(LIN.LinearNoBias, "forward") if getattr(LIN.LinearNoBias.forward, "__wrapped_stock__", None) else None
        DT.PairConditioning.forward = DT.PairConditioning.forward.__wrapped_stock__
