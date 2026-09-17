"""``mem.rowpair.rankdata``, rank side: the cross-rank feature digest (tensors are the contract; non-tensor leaves canonical, never ``repr()``;
objects without a canonical form excluded by name) and the agree gate in both modes on 2 threaded ranks and 2 gloo processes — equal digests agree
on both ranks, different digests are ``False`` on BOTH ranks, the refusal of both in ``refuse`` mode, the census line in ``census`` mode; plus
``run_sharded``'s seed export (spawn workers share the launch's seed; the parent's environment is restored)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import rankdata as RD  # noqa: E402


def _feats():
    g = torch.Generator().manual_seed(0)
    return {"msa": torch.randint(0, 20, (7, 5), generator=g), "token_index": torch.arange(5), "ref_pos": torch.randn(5, 3, generator=g),
            "name": "q1", "nested": {"a": torch.ones(2, dtype=torch.bool), "b": [torch.zeros(1), 3]}}


def test_feature_digest_definition():
    f = _feats()
    fd = RD.digest_features(f)
    assert fd.digest == RD.feature_digest(f) and len(fd.digest) == 64
    assert fd.tensor_leaves == 5 and fd.nontensor_leaves == ("/name", "/nested/b/1") and fd.unhashed == ()
    assert fd.words() == "tensor_leaves=5 nontensor_leaves=2:/name,/nested/b/1 nontensor_unhashed=none"
    assert RD.feature_digest(f, exclude=("msa",)) == RD.feature_digest({k: v for k, v in f.items() if k != "msa"}) != fd.digest
    assert RD.feature_digest(f, exclude=("absent_key",)) == fd.digest                       # excluding an absent key changes nothing
    assert RD.feature_digest(dict(reversed(list(f.items())))) == fd.digest                  # insertion order is not identity (keys sorted)
    g = {**f, "ref_pos": f["ref_pos"].clone()}
    assert RD.feature_digest(g) == fd.digest                                                 # bytes, not object identity
    g["ref_pos"][0, 0] += 1e-6
    assert RD.feature_digest(g) != fd.digest                                                 # one changed element changes the digest
    assert RD.feature_digest({**f, "ref_pos": f["ref_pos"].to(torch.float64)}) != fd.digest  # dtype is identity
    assert RD.feature_digest({**f, "name": "q2"}) != fd.digest                               # a str leaf is identity (canonical JSON)
    assert RD.feature_digest({**f, "n": 3}) != RD.feature_digest({**f, "n": 3.0})            # json 3 vs 3.0
    import numpy as np
    assert RD.digest_features({**f, "arr": np.arange(4, dtype=np.int32)}).tensor_leaves == 6  # numpy arrays are tensor leaves
    assert RD.feature_digest({**f, "arr": np.arange(4, dtype=np.int32)}) != RD.feature_digest({**f, "arr": np.arange(4, dtype=np.int64)})
    with pytest.raises(TypeError):
        RD.feature_digest([f["msa"]], exclude=("msa",))


def test_leaf_classes_without_a_byte_form_are_excluded_by_name():
    import numpy as np
    f = _feats()
    base = RD.digest_features(f)
    e = RD.digest_features({**f, "empty2d": np.zeros((0, 3), dtype=np.float32), "empty_t": torch.zeros(0, 4)})
    assert e.unhashed == () and e.tensor_leaves == base.tensor_leaves + 2 and e.digest != base.digest        # empty arrays: header only, no crash
    d = RD.digest_features({**f, "when": np.datetime64("2026-09-07"), "z": complex(1, 2), "raw": np.bytes_(b"ab"), "obj": np.array([object(), 1], dtype=object),
                            "objs": np.array(["a", 2, None], dtype=object), "u": np.uint8(3), "b": b"xy"})
    assert set(d.unhashed) == {"/when:datetime64", "/z:complex", "/obj:ndarray[object]"}, d.unhashed
    assert {"/u", "/b", "/raw"} <= set(d.nontensor_leaves) and "/objs" not in " ".join(d.unhashed + d.nontensor_leaves), d   # np.bytes_ is bytes: by content; an object array of scalars is an array leaf (canonical JSON body)
    assert d.tensor_leaves == base.tensor_leaves + 1
    assert RD.feature_digest({"m": {}}) != RD.feature_digest({"m": {"e": {}}})                                # a nested mapping's size is identity
    assert RD.feature_digest({"t": torch.ones(3, dtype=torch.bfloat16)}) != RD.feature_digest({"t": torch.ones(3, dtype=torch.float16)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_and_cpu_tensors_digest_alike():
    t = torch.randn(5, 7)
    assert RD.feature_digest({"a": t.cuda()}) == RD.feature_digest({"a": t}) == RD.feature_digest({"a": t.t().contiguous().t()})


class _Sock:                                                                                  # an object with no canonical form: excluded BY NAME
    pass


def test_unhashable_objects_are_excluded_by_name_not_silently():
    f = {**_feats(), "handle": _Sock()}
    fd = RD.digest_features(f)
    assert fd.unhashed == ("/handle:_Sock",) and "nontensor_unhashed=/handle:_Sock" in fd.words()
    assert fd.digest == RD.feature_digest(_feats())                                          # excluded: the digest is the rest's
    assert RD.digest_features({**_feats(), "handle": object()}).digest == fd.digest


class FakeAtomArray:
    """Duck-typed stand-in for a biotite AtomArray: annotations kept in ATTACHMENT order (what `repr` and `_annot` iteration follow)."""

    def __init__(self, n, order, coord=None, box=None):
        import numpy as np
        self._annot = {}
        vals = {"chain_id": np.array(["A"] * n), "res_id": np.arange(n, dtype=np.int64), "charge": np.zeros(n, dtype=np.int64),
                "entity_id": np.ones(n, dtype=np.int64), "molecule_type_id": np.zeros(n, dtype=np.int64)}
        for k in order:
            self._annot[k] = vals[k].copy()
        self.coord = np.arange(3 * n, dtype=np.float32).reshape(n, 3) if coord is None else coord
        self.box = box

    def get_annotation_categories(self):
        return list(self._annot)

    def get_annotation(self, c):
        return self._annot[c]

    def __repr__(self):                                                                       # attachment-order dependent, like biotite's
        return "Fake(" + ", ".join(f"{k}={v[0]}" for k, v in self._annot.items()) + ")"


ORDER_A = ("chain_id", "res_id", "charge", "molecule_type_id", "entity_id")
ORDER_B = ("chain_id", "res_id", "entity_id", "charge", "molecule_type_id")


def test_atom_array_summary_is_attachment_order_independent_and_content_sensitive():
    a, b = FakeAtomArray(6, ORDER_A), FakeAtomArray(6, ORDER_B)
    assert repr(a) != repr(b) and RD.is_atom_array(a)
    fa = RD.digest_features({**_feats(), "atom_array": [a]})
    fb = RD.digest_features({**_feats(), "atom_array": [b]})
    assert fa.digest == fb.digest and "/atom_array/0:atom_array" in fa.nontensor_leaves and fa.unhashed == (), fa
    c = FakeAtomArray(6, ORDER_A); c._annot["charge"][3] = 1                                 # one element of one annotation
    d = FakeAtomArray(6, ORDER_A); d.coord[2, 1] += 0.5                                      # one coordinate
    e = FakeAtomArray(6, ORDER_A, box=__import__("numpy").eye(3, dtype=__import__("numpy").float32))   # a box present
    g = FakeAtomArray(6, ORDER_A[:-1])                                                       # one category fewer
    for other in (c, d, e, g):
        assert RD.digest_features({**_feats(), "atom_array": [other]}).digest != fa.digest


def test_biotite_atom_arrays_concatenated_in_either_order_digest_alike():
    struc = pytest.importorskip("biotite.structure")
    import numpy as np

    def part(n, extra_order):
        arr = struc.AtomArray(n)
        arr.coord = np.arange(3 * n, dtype=np.float32).reshape(n, 3)
        arr.chain_id[:] = "A"; arr.res_id[:] = np.arange(n)
        for k in extra_order:                                                                # non-standard categories attached in a given order
            arr.set_annotation(k, np.full(n, {"charge": 0, "entity_id": 1, "molecule_type_id": 2}[k], dtype=np.int64))
        return arr
    x = part(4, ("charge", "entity_id", "molecule_type_id")) + part(3, ("charge", "entity_id", "molecule_type_id"))
    y = part(4, ("molecule_type_id", "charge", "entity_id")) + part(3, ("entity_id", "molecule_type_id", "charge"))
    assert sorted(x.get_annotation_categories()) == sorted(y.get_annotation_categories())
    assert RD.feature_digest({"atom_array": [x]}) == RD.feature_digest({"atom_array": [y]})
    y.charge[5] = 7
    assert RD.feature_digest({"atom_array": [x]}) != RD.feature_digest({"atom_array": [y]})


def test_rankenv_line_and_words():
    assert RD.rankenv_line("rowpair", {}, 4) == "[rowpair] RANKENV hashseed=0 source=default ranks=4"
    assert RD.rankenv_line("kit-opt", {"PYTHONHASHSEED": "7"}, 2) == "[kit-opt] RANKENV hashseed=7 source=inherited ranks=2"
    assert RD.agree_word("feats", True) == "feats_ranks_equal=yes" and RD.agree_word("feats", False) == "feats_ranks_equal=no" and RD.agree_word("x", None) == "x_ranks_equal=n/a"
    assert RD.MODES == ("refuse", "census")
    with pytest.raises(ValueError):
        RD.assert_ranks_agree("ab" * 32, mode="warn")


def test_ranks_agree_is_none_without_a_group():
    assert RD.ranks_agree("ab" * 32) is None and RD.assert_ranks_agree("ab" * 32) is None     # the solo comm (P == 1): nothing to compare, nothing raised
    lines = []
    assert RD.assert_ranks_agree("ab" * 32, mode="census", log=lines.append) is None and RD.gather_digests("ab" * 32) is None
    assert lines == ["[feats] rank 0 digest abababababababab feats_ranks_equal=n/a ranks=1"], lines


def _agree_entry(rank, P, digests):
    """Runs on rank-thread `rank`: returns (ranks_agree, refusal reason or None, census bool, census line, gathered heads) for this rank's digest."""
    from opt_core.mem.rowpair import RowpairRefused
    d = digests[rank]
    agree = RD.ranks_agree(d)
    try:
        RD.assert_ranks_agree(d, what="feats")
        reason = None
    except RowpairRefused as e:
        reason = e.reason
    lines = []
    census = RD.assert_ranks_agree(d, what="feats", mode="census", log=lines.append)      # never raises
    return agree, reason, census, lines[0], RD.gather_digests(d)


def test_ranks_agree_threaded_equal_and_differ():
    from opt_core.testing import run_ranks
    same = RD.feature_digest(_feats())
    other = RD.feature_digest({**_feats(), "name": "q2"})
    assert same != other
    res = run_ranks(2, _agree_entry, [same, same])
    assert [r[:3] for r in res] == [(True, None, True), (True, None, True)], res
    assert res[1][3] == f"[feats] rank 1 digest {same[:16]} feats_ranks_equal=yes ranks=2 digests={same[:16]},{same[:16]}" and res[0][4] == [same[:16], same[:16]], res
    res = run_ranks(2, _agree_entry, [same, other])
    assert [r[0] for r in res] == [False, False] and [r[2] for r in res] == [False, False], res    # the SAME answer on both ranks, both modes
    assert res[0][1].startswith(f"refused: feats_ranks_differ: rank 0 digest {same[:16]} ") and res[1][1].startswith(f"refused: feats_ranks_differ: rank 1 digest {other[:16]} "), res
    assert "every rank's `[feats]` line names its digest" in res[1][1]
    assert res[0][3] == f"[feats] rank 0 digest {same[:16]} feats_ranks_equal=no ranks=2 digests={same[:16]},{other[:16]}" and res[1][4] == [same[:16], other[:16]], res


def _gloo_entry(digest_by_rank):
    from opt_core.mem.rowpair import RowpairRefused, dist as DD
    P, r = DD.world()
    d = digest_by_rank[r]
    agree = RD.ranks_agree(d)
    try:
        RD.assert_ranks_agree(d)
        reason = None
    except RowpairRefused as e:
        reason = e.reason
    box = DD.comm().allgather_obj((r, agree, reason))                                          # both ranks' answers reach rank 0's return value
    DD.barrier()
    return sorted(box)


def _seed_entry():
    import os as _os
    from opt_core.mem.rowpair import dist as DD
    box = DD.comm().allgather_obj((_os.environ.get("PYTHONHASHSEED"), hash("rowpair-probe")))
    DD.barrier()
    return box


def test_run_sharded_workers_share_the_launch_seed(monkeypatch, capfd):
    """`run_sharded` (spawn workers): the seed is exported into the PARENT environment while the workers start, so both workers report the same
    PYTHONHASHSEED and the same str hash, and the parent's environment is restored; with 7 in the parent both report 7. The `RANKENV` line
    names the seed and its source once per launch."""
    from opt_core.mem.rowpair import launch
    monkeypatch.setenv("PYTHONHASHSEED", "5"); monkeypatch.delenv("PYTHONHASHSEED")   # absent now, restored to the session's value at teardown
    got = launch.run_sharded(2, _seed_entry, cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert [g[0] for g in got] == ["0", "0"] and got[0][1] == got[1][1], got
    assert "PYTHONHASHSEED" not in os.environ                                                # the parent's environment as it was
    assert "[rowpair] RANKENV hashseed=0 source=default ranks=2" in capfd.readouterr().err
    got2 = launch.run_sharded(2, _seed_entry, cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert "source=default" in capfd.readouterr().err and [g[0] for g in got2] == ["0", "0"]   # a second launch in the same process is still `default`
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    got = launch.run_sharded(2, _seed_entry, cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert [g[0] for g in got] == ["7", "7"] and got[0][1] == got[1][1] and os.environ["PYTHONHASHSEED"] == "7", got
    assert "[rowpair] RANKENV hashseed=7 source=inherited ranks=2" in capfd.readouterr().err


def test_ranks_agree_two_gloo_processes():
    from opt_core.mem.rowpair import launch
    same = RD.feature_digest(_feats())
    other = RD.feature_digest({**_feats(), "name": "q2"})
    got = launch.run_sharded(2, _gloo_entry, [same, same], cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert got == [(0, True, None), (1, True, None)], got
    got = launch.run_sharded(2, _gloo_entry, [same, other], cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert [g[1] for g in got] == [False, False] and got[0][2].startswith("refused: feats_ranks_differ: rank 0 ") and got[1][2].startswith("refused: feats_ranks_differ: rank 1 "), got
