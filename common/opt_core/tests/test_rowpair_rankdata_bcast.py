"""``mem.rowpair.rankdata.broadcast_features`` — the ``rank0_bcast`` data form: rank 0 featurises, every rank returns holding the same tree.
Proven on 2 threaded ranks and on 2 gloo processes: the receiver's tree digests like the source's (tensors byte-identical, non-tensor leaves
equal), a source argument that is not a mapping is refused on every rank before anything is signalled, ``skip_keys`` do not travel and come back as meta (``msa_host.place_skipped`` builds the zero-row placeholders), ``host_keys`` land on
the host; a rank-0 failure is the SAME refusal on both ranks within seconds (no hang); a slow rank 0 is waited for at the rendezvous with no
collective pending; a control word reaches every rank with nothing broadcast; the receipt check names a malformed tree; ``carry_rng`` hands the
receivers rank 0's RNG state (python / numpy / torch)."""
import os
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused, rankdata as RD  # noqa: E402


def _feats():
    g = torch.Generator().manual_seed(0)
    return {"msa": torch.randint(0, 20, (1, 9, 6, 3), generator=g, dtype=torch.int32), "has_deletion": torch.rand(1, 9, 6, generator=g),
            "token_bonds": torch.randint(0, 2, (1, 6, 6), generator=g, dtype=torch.int32), "token_mask": torch.ones(1, 6),
            "ref_pos": torch.randn(1, 40, 3, generator=g), "flag": torch.tensor([1.0]), "is_x": torch.zeros(6, dtype=torch.bool),
            "half": torch.randn(4, generator=g).to(torch.bfloat16), "query_id": ["q1"], "seed": [5], "atom_array": [{"names": ["CA", "CB"], "n": 2}]}


SKIP = ("msa", "has_deletion")
HOST = ("token_bonds",)


def test_words_and_forms():
    assert RD.DATA_FORMS == ("rank0_bcast", "per_rank") and RD.data_form_word("per_rank") == "data_form=per_rank"
    with pytest.raises(RowpairRefused) as ei:
        RD.check_data_form("row_born")
    assert "refused: data_form='row_born' is not one of rank0_bcast|per_rank" in ei.value.reason
    assert RD.status_word() == "ok" and RD.status_word(extra="n=4960") == "ok n=4960"
    assert RD.status_word(ValueError("bad\nquery " + "x" * 300)).startswith("failed:ValueError:bad query x") and len(RD.status_word(ValueError("x" * 300))) <= 220


def test_no_group_returns_the_tree_untouched():
    f = _feats()
    fb = RD.broadcast_features(f, skip_keys=SKIP, host_keys=HOST)
    assert fb.feats is f and fb.status == "ok" and fb.skipped == {} and fb.n_tensors == 0 and fb.rng == "off"
    with pytest.raises(RowpairRefused) as ei:                                                  # a failure word refuses even without a group (one rank, one answer)
        RD.broadcast_features(None, status=RD.status_word(KeyError("tmpl")))
    assert ei.value.reason.startswith("refused: feats_rank0_failed: KeyError: ")


def test_check_received_names_a_malformed_tree():
    meta = {"keys": ["a", "b", "m"], "skip": {"m": {}}, "host": ["b"], "tensors": {"a": {"shape": (2,), "dtype": "float32", "dev": "cpu"}, "b": {"shape": (1,), "dtype": "int32", "dev": "cpu"}}}
    RD.check_received({"a": torch.zeros(2), "b": torch.zeros(1, dtype=torch.int32)}, meta, "cpu")
    for bad, word in (({"a": torch.zeros(2)}, "feats_bcast_malformed: b (announced"), ({"a": torch.zeros(3), "b": torch.zeros(1, dtype=torch.int32)}, "feats_bcast_malformed: a (announced float32[2], arrived float32[3])"),
                      ({"a": torch.zeros(2, dtype=torch.float64), "b": torch.zeros(1, dtype=torch.int32)}, "arrived float64[2]"), ([1], "feats_bcast_malformed: <tree>")):
        with pytest.raises(RowpairRefused) as ei:
            RD.check_received(bad, meta, "cpu")
        assert word in ei.value.reason, ei.value.reason


# ---------------------------------------------------------------- 2 threaded ranks (one process; the thread comm has the store verbs)
def _entry(rank, P, mode, delay=0.0):
    from opt_core.mem.rowpair import msa_host as MH
    if rank == 0:
        if delay:
            time.sleep(delay)
        if mode == "fail":
            status, feats = RD.status_word(RuntimeError("featuriser died: no MSA")), None
        elif mode == "notamapping":                                          # a bad SOURCE ARGUMENT (a list, not a mapping): refused on every rank at the rendezvous
            status, feats = None, list(_feats().values())
        elif mode == "floor":
            status, feats = "below_floor:12", _feats()
        else:
            status, feats = None, _feats()
    else:
        status, feats = None, None
    t0 = time.time()
    try:
        fb = RD.broadcast_features(feats, key=f"feats/{mode}/{delay}", status=status, skip_keys=SKIP, host_keys=HOST, device="cpu", carry_rng=False)
    except RowpairRefused as e:
        return ("refused", e.reason, round(time.time() - t0, 1))
    out = {"status": fb.status, "skipped": fb.skipped, "host": fb.host, "n_tensors": fb.n_tensors, "wait_s": fb.wait_s, "words": fb.words()}
    if fb.feats is not None:
        f = fb.feats
        if rank != 0:
            facts = MH.place_skipped(f, fb.skipped, row_dims={"msa": -3, "has_deletion": -2})
            out["facts"] = facts
        out["digest"] = RD.feature_digest(f, exclude=SKIP)
        out["keys"] = sorted(f)
        out["msa_shape"] = tuple(f["msa"].shape)
        out["bonds_dev"] = f["token_bonds"].device.type
        out["nontensor"] = (f["query_id"], f["seed"], f["atom_array"])
    return ("ok", out, round(time.time() - t0, 1))


def test_threaded_receiver_holds_the_same_tree():
    from opt_core.testing import run_ranks
    res = run_ranks(2, _entry, "ok")
    assert [r[0] for r in res] == ["ok", "ok"], res
    a, b = res[0][1], res[1][1]
    assert a["digest"] == b["digest"] and a["keys"] == b["keys"] and b["nontensor"] == (["q1"], [5], [{"names": ["CA", "CB"], "n": 2}])
    assert b["skipped"] == {"msa": {"shape": (1, 9, 6, 3), "dtype": "int32", "dev": "cpu"}, "has_deletion": {"shape": (1, 9, 6), "dtype": "float32", "dev": "cpu"}} == a["skipped"]
    assert b["msa_shape"] == (1, 0, 6, 3) and a["msa_shape"] == (1, 9, 6, 3) and b["facts"]["rows"] == {"msa": 9, "has_deletion": 9}     # the receiver's zero-row placeholders carry the row count
    assert b["host"] == ("token_bonds",) and b["bonds_dev"] == "cpu" and b["n_tensors"] == a["n_tensors"] == 6                            # 8 tensors minus the 2 skipped
    assert b["words"].startswith("feats_status=ok tensors=6 gib=0.00 skipped=has_deletion,msa host_keys=token_bonds rng=off wait_s="), b["words"]


def test_threaded_rank0_failure_refuses_on_both_ranks_without_hanging():
    from opt_core.testing import run_ranks
    res = run_ranks(2, _entry, "fail", timeout_s=60)
    assert [r[0] for r in res] == ["refused", "refused"], res
    assert res[0][1] == res[1][1] == "refused: feats_rank0_failed: RuntimeError: featuriser died: no MSA" and max(r[2] for r in res) < 30


@pytest.mark.parametrize("P", [2, 3])
def test_threaded_source_argument_that_is_not_a_mapping_refuses_on_every_rank_without_hanging(P):
    """The source's ``feats`` is validated BEFORE it signals the store: a non-mapping is the status word every rank refuses on (the receivers
    are never left in the meta broadcast while the source raises alone)."""
    from opt_core.testing import run_ranks
    res = run_ranks(P, _entry, "notamapping", timeout_s=60)
    assert [r[0] for r in res] == ["refused"] * P, res
    assert all(r[1] == "refused: feats_rank0_failed: TypeError: rank 0 must pass a mapping of features (got list)" for r in res), res
    assert max(r[2] for r in res) < 30, res


def test_threaded_slow_source_is_waited_for_and_control_words_reach_every_rank():
    from opt_core.testing import run_ranks
    res = run_ranks(2, _entry, "ok", 3.0, timeout_s=120)                                       # rank 0 "featurises" for 3 s: rank 1 waits at the rendezvous, no collective pending
    assert [r[0] for r in res] == ["ok", "ok"] and res[1][1]["wait_s"] >= 2.5 and res[0][1]["digest"] == res[1][1]["digest"], res
    res = run_ranks(2, _entry, "floor")
    assert [r[1]["status"] for r in res] == ["below_floor:12", "below_floor:12"] and "digest" in res[0][1] and "digest" not in res[1][1]   # nothing broadcast; the source keeps its tree


# ---------------------------------------------------------------- 2 gloo processes (spawned): the tree, the refusal, carry_rng
def _gloo_entry(mode):
    import random
    import numpy as np
    from opt_core.mem.rowpair import dist as DD
    P, r = DD.world()
    if r == 0:
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        _ = torch.rand(3), random.random(), np.random.rand()                                   # rank 0 consumes draws while "featurising"
        feats, status = (_feats(), None) if mode != "fail" else (None, RD.status_word(ValueError("no templates")))
    else:
        random.seed(99); np.random.seed(99); torch.manual_seed(99)
        feats, status = None, None
    try:
        fb = RD.broadcast_features(feats, key=f"feats/{mode}", status=status, skip_keys=SKIP, host_keys=HOST, carry_rng=(mode == "rng"))
        rec = ("ok", RD.feature_digest(fb.feats, exclude=SKIP), fb.rng, (torch.rand(2).tolist(), random.random(), float(np.random.rand())), sorted(fb.skipped))
    except RowpairRefused as e:
        rec = ("refused", e.reason)
    box = DD.comm().allgather_obj((r,) + rec)
    DD.barrier()
    return sorted(box)


def test_gloo_processes_tree_refusal_and_rng():
    from opt_core.mem.rowpair import launch
    got = launch.run_sharded(2, _gloo_entry, "plain", cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert [g[1] for g in got] == ["ok", "ok"] and got[0][2] == got[1][2] and got[0][3] == got[1][3] == "off" and got[0][5] == ["has_deletion", "msa"], got
    assert got[0][4] != got[1][4]                                                                # no carry: the ranks' later draws differ (seeded 1 vs 99)
    got = launch.run_sharded(2, _gloo_entry, "rng", cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert [g[3] for g in got] == ["carried", "carried"] and got[0][4] == got[1][4], got             # carry_rng: rank 1 continues from rank 0's post-featurisation state
    got = launch.run_sharded(2, _gloo_entry, "fail", cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert [g[1:] for g in got] == [("refused", "refused: feats_rank0_failed: ValueError: no templates")] * 2, got
