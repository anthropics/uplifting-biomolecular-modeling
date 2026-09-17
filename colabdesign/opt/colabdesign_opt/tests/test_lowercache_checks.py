"""Lever `lowercache`'s entries hold two pickles (`executable.bin` for `deserialize_and_load`, `trees.pkl`): both sha256 lines are checked
BEFORE a byte of the entry is unpickled, an entry another account could have written is refused, every refusal is one by-name stderr line
(what, why, the fix) followed by exactly the absent-entry behaviour, an entry whose bytes do not match is removed so the next store lands,
and what the lever itself writes — under any umask — is never refused. `serialize_executable` is a stand-in here (no jax needed)."""
import hashlib
import os
import pickle
import stat
import sys
import types

import pytest

from colabdesign_opt import lowercache as lc

TRIPPED, LOADED = [], []


def _trip(word):
    TRIPPED.append(word)
    return word


class _Trap:
    """Unpickling this calls `_trip` — the stand-in for a pickle that runs code."""

    def __reduce__(self):
        return (_trip, ("unpickled",))


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    ser = types.ModuleType("jax.experimental.serialize_executable")
    ser.serialize = lambda compiled: (b"EXE:" + compiled.encode(), "in", "out")

    def deserialize_and_load(blob, in_tree, out_tree):
        LOADED.append(blob)
        return ("loaded", blob, in_tree, out_tree)

    ser.deserialize_and_load = deserialize_and_load
    monkeypatch.setitem(sys.modules, "jax.experimental.serialize_executable", ser)
    monkeypatch.setitem(lc._STATE, "info", {"dir": str(tmp_path / "lowered")})
    monkeypatch.setitem(lc._STATE, "ffi_done", True)
    os.makedirs(str(tmp_path / "lowered"))
    TRIPPED.clear(); LOADED.clear()
    old = os.umask(0o022)
    yield str(tmp_path / "lowered")
    os.umask(old)


def test_entry_files_are_their_sha256_line_then_the_pickle_and_load(store):
    assert lc._store("k1", "compiled", {"program": "fn"}) == len(b"EXE:compiled")
    for name in ("executable.bin", "trees.pkl"):
        head, body = open(os.path.join(store, "k1", name), "rb").read().split(b"\n", 1)
        assert head == b"sha256:" + hashlib.sha256(body).hexdigest().encode()
    exe, meta = lc._load("k1")
    assert exe == ("loaded", b"EXE:compiled", "in", "out") and meta == {"program": "fn"}


@pytest.mark.parametrize("name", ("executable.bin", "trees.pkl"))
@pytest.mark.parametrize("damage", ("no_line", "wrong_line", "truncated"))
def test_entry_without_matching_lines_is_never_unpickled_and_leaves_so_the_store_lands(store, name, damage):
    lc._store("k1", "compiled", {"program": "fn"})
    p = os.path.join(store, "k1", name)
    trap, good = pickle.dumps(_Trap()), open(p, "rb").read()
    with open(p, "wb") as f:
        f.write({"no_line": trap, "wrong_line": b"sha256:" + b"0" * 64 + b"\n" + trap, "truncated": good[:-2]}[damage])
    with pytest.raises(lc.Refused, match="sha256") as e:
        lc._load("k1")
    assert e.value.rewrite and "fix:" in str(e.value) and p in str(e.value)
    assert TRIPPED == [] and LOADED == [] and not os.path.exists(os.path.join(store, "k1"))
    lc._store("k1", "compiled", {"program": "fn"})                   # rebuilt once: the traced program's store lands, the next load is served
    assert lc._load("k1")[0][0] == "loaded" and TRIPPED == []


@pytest.mark.parametrize("target", ("directory", "executable.bin", "trees.pkl"))
@pytest.mark.parametrize("bit", (stat.S_IWGRP, stat.S_IWOTH))
def test_group_or_other_writable_is_refused_with_the_fix_and_kept_then_loads_after_it(store, target, bit):
    lc._store("k1", "compiled", {"program": "fn"})
    q = os.path.join(store, "k1") if target == "directory" else os.path.join(store, "k1", target)
    os.chmod(q, stat.S_IMODE(os.stat(q).st_mode) | bit)
    with pytest.raises(lc.Refused, match="is writable by group or other") as e:
        lc._load("k1")
    assert not e.value.rewrite and f"fix: chmod go-w {q}" in str(e.value) and LOADED == [] and os.path.isdir(os.path.join(store, "k1"))
    os.chmod(q, stat.S_IMODE(os.stat(q).st_mode) & ~0o022)           # the printed fix
    assert lc._load("k1")[0][0] == "loaded"


def test_foreign_owner_is_refused_for_a_user_and_accepted_for_uid_0(store, monkeypatch):
    lc._store("k1", "compiled", {"program": "fn"})
    owner = os.stat(os.path.join(store, "k1")).st_uid
    if owner != 0:
        monkeypatch.setattr(os, "geteuid", lambda: owner + 1)        # this process is another (non-root) user
        with pytest.raises(lc.Refused, match=f"belongs to uid {owner}, not to this process"):
            lc._load("k1")
    monkeypatch.setattr(os, "geteuid", lambda: 0)                    # a container's root reading a bind-mounted host directory
    assert lc._load("k1")[0][0] == "loaded"


@pytest.mark.parametrize("umask", (0o002, 0o000, 0o077))
def test_what_the_lever_writes_is_never_refused_whatever_the_umask(store, umask):
    os.umask(umask)
    lc._store("k1", "compiled", {"program": "fn"})
    d = os.path.join(store, "k1")
    assert os.stat(d).st_mode & 0o022 == 0 and all(os.stat(os.path.join(d, n)).st_mode & 0o022 == 0 for n in ("executable.bin", "trees.pkl"))
    assert lc._load("k1")[0][0] == "loaded"


class _Jit:
    """`jit(f)` as `Persisted._first_call` uses it on the traced path: `.lower(*args).compile()`."""

    def lower(self, *args):
        return self

    def compile(self):
        return "compiled"

    def as_text(self):
        return "hlo"


def _first_call(monkeypatch):
    monkeypatch.setitem(lc._STATE, "counts", dict(lc._ZERO))
    monkeypatch.setitem(lc._STATE, "reasons", [])
    monkeypatch.setitem(lc._STATE, "relower", False)
    monkeypatch.setattr(lc, "_MEMO", {})
    monkeypatch.setattr(lc, "key_of", lambda ctx, sig: {"key": "k1", "material": {}})
    exe = lc.Persisted(_Jit(), "fn", {})._first_call("sig", ())
    return exe, dict(lc._STATE["counts"]), list(lc._STATE["reasons"])


def test_first_call_a_mismatching_entry_is_one_stderr_line_then_traced_stored_and_loaded_next_time(store, monkeypatch, capfd):
    lc._store("k1", "compiled", {"program": "fn"})
    with open(os.path.join(store, "k1", "trees.pkl"), "wb") as f:
        f.write(pickle.dumps(_Trap()))
    exe, counts, reasons = _first_call(monkeypatch)
    err = capfd.readouterr().err
    assert exe == "compiled" and TRIPPED == [] and LOADED == []
    assert (counts["traced"], counts["stores"], counts["loads"], counts["fallbacks"]) == (1, 1, 0, 0)
    assert err.count("[colabdesign-opt] lowercache: REFUSED cache entry: file " + os.path.join(store, "k1", "trees.pkl")) == 1
    assert "sha256" in err and "fix:" in err and "treated as absent" in err and len(reasons) == 1 and reasons[0].startswith("refused/fn:")
    exe, counts, reasons = _first_call(monkeypatch)                  # rebuilt once: the next process is served, silently
    assert exe[0] == "loaded" and (counts["loads"], counts["traced"]) == (1, 0) and reasons == [] and "REFUSED" not in capfd.readouterr().err


def test_first_call_an_entry_another_account_could_write_is_one_stderr_line_then_the_traced_path(store, monkeypatch, capfd):
    lc._store("k1", "compiled", {"program": "fn"})
    d = os.path.join(store, "k1")
    os.chmod(d, 0o775)
    exe, counts, reasons = _first_call(monkeypatch)
    err = capfd.readouterr().err
    assert exe == "compiled" and LOADED == [] and (counts["traced"], counts["loads"], counts["fallbacks"]) == (1, 0, 0)
    assert err.count("REFUSED cache entry") == 1 and f"directory {d} is writable by group or other (mode 0775); fix: chmod go-w {d}" in err
    os.chmod(d, 0o755)                                               # the printed fix: the entry was kept, and now loads
    assert _first_call(monkeypatch)[0][0] == "loaded"
