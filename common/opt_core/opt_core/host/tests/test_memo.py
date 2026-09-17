"""CPU tests of opt_core.host.memo: canonical digests, refusals by name, LRU bound, bypass census, check fail-closed, disk mirror."""
from __future__ import annotations

import json
import os
import threading

import pytest

from opt_core.host import HostRefused
from opt_core.host.memo import Memo, MemoMismatch, digest, file_digest


def test_digest_is_canonical_and_typed():
    assert digest({"a": 1, "b": [1, 2]}) == digest({"b": [1, 2], "a": 1})          # mapping order-free
    assert digest([1, 2]) != digest((1, 2))                                        # list vs tuple
    assert digest(1) != digest("1") != digest(1.0)
    assert digest(b"x") != digest("x")
    assert digest(True) != digest(1)
    assert len(digest(None)) == 64


def test_digest_refuses_identity_objects_by_name():
    class Opaque:
        pass
    with pytest.raises(HostRefused) as ei:
        digest({"x": Opaque()})
    assert ei.value.event.reason == "undigestible:Opaque"
    assert ei.value.event.event == "REFUSED"
    with pytest.raises(HostRefused):
        digest({1.5: "float key"})
    with pytest.raises(HostRefused):
        digest({"s": {1, 2}})


def test_custom_digest_hook():
    class Named:
        def __init__(self, v):
            self.v = v

        def __opt_core_digest__(self):
            return {"v": self.v}
    assert digest(Named(3)) == digest(Named(3)) != digest(Named(4))


def test_file_digest(tmp_path):
    p = tmp_path / "in.a3m"
    p.write_bytes(b">q\nACDE\n")
    assert file_digest(str(p)) == digest.__globals__["hashlib"].sha256(b">q\nACDE\n").hexdigest()


def test_get_or_build_hits_misses_and_lru():
    calls = []
    m = Memo("feat", max_entries=2)

    def build_for(x):
        def build():
            calls.append(x)
            return {"feat": x * 2}
        return build

    assert m.get_or_build({"x": 1, "seed": 0}, build_for(1)) == {"feat": 2}
    assert m.get_or_build({"seed": 0, "x": 1}, build_for(1)) == {"feat": 2}       # same key, other order: hit
    m.get_or_build({"x": 2, "seed": 0}, build_for(2))
    m.get_or_build({"x": 3, "seed": 0}, build_for(3))                              # evicts x=1
    m.get_or_build({"x": 1, "seed": 0}, build_for(1))                              # rebuilt
    t = m.tally()
    assert calls == [1, 2, 3, 1]
    assert (t["hits"], t["misses"], t["builds"], t["evictions"], t["entries"]) == (1, 4, 4, 2, 2)
    line = m.tally_line("kit")
    assert line.startswith("[kit] HOST host.memo TALLY name=feat hits=1 misses=4 builds=4 bypass=none evictions=2")
    assert m.active_line("kit") == "[kit] LEVER name=F6.feature_cache state=on impl=host.memo origin=core memo=feat max_entries=2 disk=none verify_every=0"
    assert m.active_line("kit", "F6.plm_memo").startswith("[kit] LEVER name=F6.plm_memo state=on impl=host.memo origin=core ")


def test_empty_or_non_mapping_parts_refused():
    m = Memo("feat")
    with pytest.raises(HostRefused):
        m.get_or_build({}, lambda: 1)
    with pytest.raises(HostRefused):
        m.get_or_build(["x"], lambda: 1)                # type: ignore[arg-type]


def test_bypass_is_counted_by_reason_never_cached():
    m = Memo("feat")
    n = {"builds": 0}

    def build():
        n["builds"] += 1
        return 7

    assert m.get_or_build({"x": 1}, build, bypass="smiles_ligand") == 7
    assert m.get_or_build({"x": 1}, build, bypass="smiles_ligand") == 7
    assert n["builds"] == 2 and len(m) == 0
    assert m.tally()["bypass"] == {"smiles_ligand": 2}
    assert [e.event for e in m.events] == ["BYPASS"]                               # one event record per reason
    assert "bypass=smiles_ligand:2" in m.tally_line("kit")


def test_verify_every_fails_closed_on_mismatch():
    state = {"v": 1}
    m = Memo("feat", verify_every=2)
    build = lambda: {"value": state["v"]}                                          # noqa: E731
    m.get_or_build({"x": 1}, build)                                                # miss
    m.get_or_build({"x": 1}, build)                                                # hit 1: no check
    state["v"] = 2                                                                 # the build now disagrees with the cache: a wrong key
    with pytest.raises(MemoMismatch) as ei:
        m.get_or_build({"x": 1}, build)                                            # hit 2: check → mismatch
    assert ei.value.event.event == "MISMATCH"
    t = m.tally()
    assert t["verified"] == 1 and t["mismatches"] == 1


def test_disk_mirror_roundtrip_and_corruption_named(tmp_path):
    d = str(tmp_path / "memo")
    m1 = Memo("msa", disk_dir=d)
    assert m1.get_or_build({"f": "abc"}, lambda: [1, 2, 3]) == [1, 2, 3]
    assert m1.tally()["disk_writes"] == 1
    m2 = Memo("msa", disk_dir=d)                                                   # a new process: served from disk, no build
    built = []
    assert m2.get_or_build({"f": "abc"}, lambda: built.append(1) or [9]) == [1, 2, 3]
    assert built == [] and m2.tally()["disk_hits"] == 1
    key = m2.key({"f": "abc"})
    with open(os.path.join(d, key + ".pkl"), "ab") as fh:                        # corrupt the value file
        fh.write(b"junk")
    m3 = Memo("msa", disk_dir=d)
    assert m3.get_or_build({"f": "abc"}, lambda: [4]) == [4]                       # rebuilt, corruption counted and named
    assert m3.tally()["disk_corrupt"] == 1
    assert m3.events[0].reason == "disk_corrupt" and m3.events[0].event == "FALLBACK"
    m4 = Memo("msa", disk_dir=d, strict_disk=True)
    with open(os.path.join(d, key + ".pkl"), "ab") as fh:
        fh.write(b"junk")
    with pytest.raises(HostRefused):
        m4.get_or_build({"f": "abc"}, lambda: [5])
    with open(os.path.join(d, key + ".json")) as fh:
        assert json.load(fh)["memo"] == "msa"


def test_concurrent_callers_of_one_key_build_once():
    m = Memo("esm", max_entries=4)
    n = {"builds": 0}
    gate = threading.Event()

    def build():
        n["builds"] += 1
        gate.wait(1.0)
        return "emb"

    out = []
    ts = [threading.Thread(target=lambda: out.append(m.get_or_build({"seq": "ACD"}, build))) for _ in range(6)]
    for t in ts:
        t.start()
    gate.set()
    for t in ts:
        t.join()
    assert out == ["emb"] * 6 and n["builds"] == 1
    t = m.tally()
    assert t["hits"] + t["misses"] == 6 and t["builds"] == 1


def test_numpy_arrays_digest_by_content():
    np = pytest.importorskip("numpy")
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert digest(a) == digest(a.copy())
    assert digest(a) != digest(a.astype(np.float64))
    assert digest(a) != digest(a.reshape(3, 2))
    assert digest(a[:, ::2]) == digest(np.ascontiguousarray(a[:, ::2]))
