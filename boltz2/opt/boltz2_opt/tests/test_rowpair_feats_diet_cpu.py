"""The ×P batch diet (rowpair_msa.feats_diet_): on rank 0, before the broadcast, the distogram loss target and the two unread
atom maps leave the batch and the two {0,1} one-hot atom maps travel as uint8 (every inference reader casts or reduces them: the same values)."""
import torch

from .. import rowpair_msa as RM


def _batch(N=6, A=14):
    a2t = torch.nn.functional.one_hot(torch.arange(A) % N, N)[None]                    # [1, A, N] int64
    t2r = torch.nn.functional.one_hot(torch.arange(N) * 2 % A, A)[None]                # [1, N, A] int64
    return {"disto_target": torch.zeros(1, N, N, 1, 64), "r_set_to_rep_atom": t2r.clone(), "token_to_center_atom": t2r.clone(),
            "atom_to_token": a2t, "token_to_rep_atom": t2r, "msa": torch.randint(0, 33, (1, 5, N)), "token_pad_mask": torch.ones(1, N)}


def test_the_diet_drops_the_unread_planes_and_carries_the_onehot_maps_as_uint8(monkeypatch):
    monkeypatch.delenv(RM.FEATS_DIET_ENV, raising=False)
    b = _batch(); a2t64, t2r64, msa = b["atom_to_token"].clone(), b["token_to_rep_atom"].clone(), b["msa"]
    facts = RM.feats_diet_(b)
    assert facts["on"] and facts["dropped"] == list(RM.FEATS_DIET_DROP) and facts["uint8"] == list(RM.FEATS_DIET_UINT8) and facts["kept"] == []
    assert all(k not in b for k in RM.FEATS_DIET_DROP)
    assert b["atom_to_token"].dtype == torch.uint8 and b["token_to_rep_atom"].dtype == torch.uint8 and b["msa"] is msa
    assert torch.equal(b["atom_to_token"].float(), a2t64.float()) and torch.equal(b["token_to_rep_atom"].float(), t2r64.float())   # every reader's cast: the same values
    assert torch.equal(b["atom_to_token"].sum(1), a2t64.sum(1)) and b["atom_to_token"].sum(1).dtype == torch.int64            # confidencev2:366's reduction: the same int64 counts
    assert facts["saved_bytes"] == 6 * 6 * 64 * 4 + 2 * 6 * 14 * 8 + 2 * 6 * 14 * 7


def test_a_map_that_is_not_zero_one_travels_as_it_is_named(monkeypatch):
    monkeypatch.delenv(RM.FEATS_DIET_ENV, raising=False)
    b = _batch(); b["atom_to_token"] = b["atom_to_token"] * 3
    facts = RM.feats_diet_(b)
    assert b["atom_to_token"].dtype == torch.int64 and facts["kept"] == ["atom_to_token:int64"] and facts["uint8"] == ["token_to_rep_atom"]


def test_the_word_off_leaves_the_batch_as_featurized(monkeypatch):
    monkeypatch.setenv(RM.FEATS_DIET_ENV, "0")
    b = _batch(); keys = set(b)
    facts = RM.feats_diet_(b)
    assert not facts["on"] and set(b) == keys and b["atom_to_token"].dtype == torch.int64
