"""The replicated-tensor guard sees PERMUTATIONS: ``dist.checksum`` carries a position-weighted term, so two one-hot MSA subsamples of equal
size (same multiset of values, different rows) differ, and ``dist.allreduce_checksum`` / ``trunk.guard_replicated`` refuse by name when one
rank holds a re-sampled / re-ordered copy (threaded ranks, P in {2, 3}); identical tensors pass; the digest is identical for identical bytes
on every rank and dtype route (bool / uint8 / fp16 / fp32 / int64)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
from opt_core.mem.rowpair import dist as D  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402


def _onehot(S=12, N=20, K=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, K, (S, N), generator=g)
    return torch.nn.functional.one_hot(idx, K).to(torch.float32)             # every row holds exactly N ones


def test_checksum_sees_permutation_of_onehot():
    a = _onehot()
    perm = torch.randperm(a.shape[0], generator=torch.Generator().manual_seed(1))
    assert not torch.equal(perm, torch.arange(a.shape[0]))
    b = a[perm].contiguous()                                                  # same multiset of values, rows re-ordered
    ca, cb = D.checksum(a), D.checksum(b)
    assert len(ca) == 4 and ca[:3] == cb[:3], (ca, cb)                        # numel / sum / xor-fold are permutation-invariant …
    assert ca[3] != cb[3], (ca, cb)                                           # … the position-weighted term is not
    c = _onehot(seed=5)                                                        # a different SUBSAMPLE of the same shape (still S*N ones)
    assert D.checksum(c)[:2] == ca[:2] and D.checksum(c) != ca
    assert D.checksum(a.clone()) == ca                                        # identical bytes -> identical digest
    for cast in (lambda t: t.bool(), lambda t: t.to(torch.uint8), lambda t: t.half(), lambda t: t.to(torch.int64)):
        assert D.checksum(cast(a)) == D.checksum(cast(a.clone())) and D.checksum(cast(a)) != D.checksum(cast(b))
    assert D.checksum(torch.empty(0)) == (0, 0, 0, 0)


def test_checksum_blocked_accumulation_matches_one_block(monkeypatch):
    a = torch.randn(3000, 37)
    one = D.checksum(a)
    monkeypatch.setattr(D, "CHECKSUM_BLOCK_ELEMS", 1024)                      # many blocks -> same digest
    assert D.checksum(a) == one


def _guard_worker(rank, P, a, b, bad_rank):
    from opt_core.mem.rowpair.trunk import guard_replicated
    mine = b if rank == bad_rank else a
    ok = {"same_passes": len(D.allreduce_checksum(a, "msa_same")) == P}
    try:
        guard_replicated(mine, "msa_feat")
        ok["permuted_refused"] = False
    except RowpairRefused as e:
        ok["permuted_refused"] = "msa_feat" in str(e)
    return ok


@pytest.mark.parametrize("P", [2, 3])
def test_guard_refuses_permuted_onehot_on_one_rank(P):
    a = _onehot()
    b = a[torch.randperm(a.shape[0], generator=torch.Generator().manual_seed(2))].contiguous()
    assert not torch.equal(a, b) and float(a.sum()) == float(b.sum())
    res = run_ranks(P, _guard_worker, a, b, P - 1)
    assert all(r["same_passes"] for r in res), res
    assert all(r["permuted_refused"] for r in res), res


def _params_worker(rank, P, bad_rank):
    from opt_core.mem.rowpair import trunk
    lin = torch.nn.Linear(8, 4)
    with torch.no_grad():                                                    # identical weights on every rank-thread WITHOUT the shared process RNG
        lin.weight.copy_(torch.arange(32.0).reshape(4, 8) / 7)
        lin.bias.copy_(torch.arange(4.0) / 3)
    ok = {"n": trunk.guard_replicated_params(lin.named_parameters(), "lin")}
    if rank == bad_rank:
        with torch.no_grad():
            lin.bias[0] += 1e-3                                              # a rank-local difference in ONE tensor
    try:
        trunk.guard_replicated_params(lin.state_dict(), "lin")
        ok["refused"] = False
    except RowpairRefused as e:
        ok["refused"] = "'lin'" in str(e) and "bias" in str(e) and "weight" not in str(e).split("differ across ranks:")[1]
    return ok


@pytest.mark.parametrize("P", [2, 3])
def test_guard_replicated_params_names_the_differing_tensor(P):
    res = run_ranks(P, _params_worker, P - 1)
    assert all(r["n"] == 2 for r in res), res
    assert all(r["refused"] is True for r in res), res


def test_guard_replicated_params_without_a_group_counts_only():
    from opt_core.mem.rowpair import trunk
    lin = torch.nn.Linear(8, 4)
    assert trunk.guard_replicated_params(lin.named_parameters(), "lin") == 2
