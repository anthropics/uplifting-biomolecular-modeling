"""big's reach lever paircond_chunk (big.py; xfold/nn/paircond_rows.py; the hunk in diffusion_head.py): the row-block relative
encoding is create_relative_encoding's rows exactly; the row-chunked pair conditioning equals the whole statement (CPU fp32, 1e-5);
the lever is registered, selected, carried to the model process and reported."""
import types

import pytest

torch = pytest.importorskip("torch")

from af3_torch_opt import big, registry


def _token_features(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    asym = torch.sort(torch.randint(1, 4, (n,), generator=g)).values
    ent = (asym + 1) // 2
    return types.SimpleNamespace(
        token_index=torch.arange(1, n + 1, dtype=torch.int32),
        residue_index=torch.cumsum(torch.randint(0, 2, (n,), generator=g), 0).to(torch.int32) + 1,
        asym_id=asym.to(torch.int32), entity_id=ent.to(torch.int32), sym_id=(asym - 2 * (ent - 1)).to(torch.int32))


def test_relative_encoding_rows_is_create_relative_encoding_rows_exactly():
    from xfold.nn import featurization, paircond_rows
    tf = _token_features(37)
    whole = featurization.create_relative_encoding(tf, max_relative_idx=32, max_relative_chain=2)      # int64 [37, 37, 139]
    assert whole.shape == (37, 37, 139) and whole.dtype == torch.int64
    for r0, r1 in ((0, 16), (16, 32), (32, 37), (0, 37)):
        rows = paircond_rows.relative_encoding_rows(tf, r0, r1, 32, 2, torch.float32)
        assert rows.dtype == torch.float32 and rows.shape == (r1 - r0, 37, 139)
        assert torch.equal(rows, whole[r0:r1].to(torch.float32)), (r0, r1)


class _Head(torch.nn.Module):
    """The four modules DiffusionHead._pair_conditioning uses, per token pair over channels (LayerNorm / Linear / per-pair MLPs)."""
    def __init__(self, c_pair=8, c_cond=6):
        super().__init__()
        torch.manual_seed(0)
        self.pair_cond_initial_norm = torch.nn.LayerNorm(c_pair + 139)
        self.pair_cond_initial_projection = torch.nn.Linear(c_pair + 139, c_cond, bias=False)
        self.pair_transition_0 = torch.nn.Sequential(torch.nn.LayerNorm(c_cond), torch.nn.Linear(c_cond, 2 * c_cond), torch.nn.SiLU(), torch.nn.Linear(2 * c_cond, c_cond))
        self.pair_transition_1 = torch.nn.Sequential(torch.nn.LayerNorm(c_cond), torch.nn.Linear(c_cond, 2 * c_cond), torch.nn.SiLU(), torch.nn.Linear(2 * c_cond, c_cond))

    def whole(self, batch, embeddings, use_conditioning):        # diffusion_head.py _pair_conditioning's statements (the stock path)
        from xfold.nn import featurization
        pair_embedding = use_conditioning * embeddings['pair']
        rel_features = featurization.create_relative_encoding(batch.token_features, max_relative_idx=32, max_relative_chain=2).to(dtype=pair_embedding.dtype)
        features_2d = torch.concatenate([pair_embedding, rel_features], dim=-1)
        pair_cond = self.pair_cond_initial_projection(self.pair_cond_initial_norm(features_2d))
        pair_cond += self.pair_transition_0(pair_cond)
        pair_cond += self.pair_transition_1(pair_cond)
        return pair_cond


@pytest.mark.parametrize("n,rows", [(37, 16), (37, 37), (37, 64), (5, 1)])
@pytest.mark.parametrize("use_conditioning", [True, False])
def test_pair_conditioning_rows_equals_whole(n, rows, use_conditioning):
    from xfold.nn import paircond_rows
    head = _Head()
    batch = types.SimpleNamespace(token_features=_token_features(n, seed=1))
    emb = {"pair": torch.randn(n, n, 8)}
    with torch.no_grad():
        ref = head.whole(batch, emb, use_conditioning)
        got = paircond_rows.pair_conditioning_rows(head, batch, emb, use_conditioning, rows)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), float((got - ref).abs().max())
    assert head._paircond_blocks == -(-n // rows)
    if rows >= n:                                                  # one block = the whole statement's own operations: the same bytes on CPU
        assert torch.equal(got, ref)


def test_levers_registered_selected_carried_reported():
    for lever in ("paircond_chunk",):
        assert lever in big.LEVER_ORDER
        assert lever in registry.BIG_LEVERS and lever in registry.IMPL and lever in registry.STRATEGY
    sel = big.selection()
    assert sel["settings"]["paircond_rows"] == big.PAIRCOND_CHUNK_ROWS > 0
    argv = big.forward_argv(sel)
    assert argv[argv.index("--paircond-rows") + 1] == str(big.PAIRCOND_CHUNK_ROWS)
    assert "--paircond-rows" not in big.forward_argv(dict(sel, levers=[l for l in sel["levers"] if l != "paircond_chunk"]))
    rep = {"big": sel, "big_record": {"paircond_rows": 256, "paircond_items": 2, "paircond_blocks": 9, "items": 3}}
    assert big.lever_state("paircond_chunk", rep) == ("on", None, {"rows": 256, "items_chunked": 2, "blocks": 9, "items": 3})
    assert big.lever_state("paircond_chunk", {"big": sel, "big_record": {}})[0] == "skipped"
    assert big.lever_state("paircond_chunk", {"big": dict(sel, levers=["graph_drop"])})[:2] == ("off", "not_selected")


def test_diffusion_head_hunks_are_inert_without_the_levers():
    """The stock path is untouched unless forward.py set the lever attribute: no pair_cond_rows -> the whole statement."""
    import os
    import xfold.nn as nn_pkg
    d = os.path.dirname(nn_pkg.__file__)
    src = open(os.path.join(d, "diffusion_head.py")).read()
    assert 'getattr(self, "pair_cond_rows", 0)' in src
    fwd = open(os.path.join(os.path.dirname(big.__file__), "forward.py")).read()
    assert 'big.get("paircond_rows") and RPX is None' in fwd                       # single-GPU only (ROWPAIR_GATES: skipped:rowpair_sampler)
