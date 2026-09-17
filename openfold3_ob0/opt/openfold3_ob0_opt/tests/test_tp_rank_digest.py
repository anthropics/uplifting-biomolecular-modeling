"""The tp line's cross-rank input digest gate (tp_rowpair/model.feature_digest, bound to opt_core.mem.rowpair.rankdata): on two ranks holding
the same batch the record names one shared digest, the leaf census and `ranks_identical=True`; a rank holding different feature bytes is refused
BY NAME on BOTH ranks (`feats_ranks_differ`) — no rank is left in a collective. Host-parked keys are out of the shared digest."""
import pytest

torch = pytest.importorskip("torch")


def _batch(shift=0.0):
    g = torch.Generator().manual_seed(0)
    return {"token_index": torch.arange(6), "ref_pos": torch.randn(6, 3, generator=g) + shift, "query_id": ["q1"], "seed": [5],
            "msa": torch.zeros(0, 6, dtype=torch.int64)}                       # a host-parked key: excluded from the shared digest by name


def _entry(rank, P, shifts):
    from opt_core.mem.rowpair import RowpairRefused
    from openfold3_ob0_opt.tp_rowpair import model as M
    lines = []
    import opt_core.mem.rowpair.dist as DD
    DD.comm().log = lines.append                                               # this rank's `[feats]` line
    try:
        rec = M.feature_digest(_batch(shifts[rank]))
        return ("ok", rec, lines)
    except RowpairRefused as e:
        return ("refused", e.reason, lines)


def test_equal_batches_share_one_digest_on_both_ranks():
    from opt_core.testing import run_ranks
    res = run_ranks(2, _entry, [0.0, 0.0])
    assert [r[0] for r in res] == ["ok", "ok"], res
    r0, r1 = res[0][1], res[1][1]
    assert r0["feats_digest_shared"] == r1["feats_digest_shared"] and r0["feats_ranks_identical"] is True and r1["feats_ranks_identical"] is True
    assert r0["feats_host_keys"] == "msa" and r0["feats_tensor_leaves"] == 2 and r0["feats_nontensor_leaves"] == "/query_id/0,/seed/0" and r0["feats_nontensor_unhashed"] == "none", r0
    assert res[1][2] and res[1][2][0].startswith(f"[feats] rank 1 shared {r1['feats_digest_shared']} keys 5 tensor_leaves=2 nontensor_leaves=2:/query_id/0,/seed/0 nontensor_unhashed=none "), res[1][2]


def test_different_feature_bytes_are_refused_by_name_on_both_ranks():
    from opt_core.testing import run_ranks
    res = run_ranks(2, _entry, [0.0, 1.0])
    assert [r[0] for r in res] == ["refused", "refused"], res
    assert res[0][1].startswith("refused: feats_ranks_differ: rank 0 digest ") and res[1][1].startswith("refused: feats_ranks_differ: rank 1 digest "), res
