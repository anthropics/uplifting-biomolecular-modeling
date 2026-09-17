"""Files the core writes and later unpickles — ``capture.xla_cache`` executables, the ``host.memo`` disk mirror, ``mem.rowpair.ckpt`` tags:
the sha256 is checked BEFORE a byte is unpickled, a directory or file another account could have written is refused, every refusal is one
by-name stderr line (what, why, the fix) followed by exactly the absent-file behaviour, and what the core itself writes — under any umask
— is never refused.

Run: ``python -m pytest tests/test_cache_file_checks.py -q`` (the ckpt tests need torch; skipped without it).
"""
from __future__ import annotations

import hashlib
import os
import pickle
import stat
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from opt_core.capture import xla_cache  # noqa: E402
from opt_core.host.memo import Memo  # noqa: E402

TRIPPED = []


def _trip(word):
    TRIPPED.append(word)
    return word


class _Trap:
    """Unpickling this calls :func:`_trip` — the stand-in for a pickle that runs code."""

    def __reduce__(self):
        return (_trip, ("unpickled",))


@pytest.fixture(autouse=True)
def _clean():
    TRIPPED.clear()
    old = os.umask(0o022)
    yield
    os.umask(old)


class _Ser:
    def serialize(self, compiled):
        return b"EXE:" + compiled.encode(), "in", "out"

    def deserialize_and_load(self, serialized, in_tree, out_tree):
        return ("loaded", serialized)


class _Lowered:
    def compile(self):
        return "compiled-b64"


def _cache(tmp_path, monkeypatch, **kw):
    c = xla_cache.ExecutableCache("fwd", str(tmp_path / "xla_exec"), {"model": 3}, verbose=False, log=lambda s: None, **kw)
    monkeypatch.setattr(c, "_ser", lambda: _Ser())
    c._stack, c._hash = {"jax": "0.5.3", "jaxlib": "0.5.3"}, "cfg0123"
    return c


def _stored(tmp_path, monkeypatch):
    c = _cache(tmp_path, monkeypatch)
    c.load_or_compile("b64", _Lowered)
    return c.path("b64")


# ------------------------------------------------------------------------------------------------------------------ xla_cache
def test_xla_stored_file_is_its_sha256_line_then_the_pickle_and_loads(tmp_path, monkeypatch):
    p = _stored(tmp_path, monkeypatch)
    head, body = open(p, "rb").read().split(b"\n", 1)
    assert head == b"sha256:" + hashlib.sha256(body).hexdigest().encode()
    assert pickle.loads(body)["serialized"] == b"EXE:compiled-b64"
    c2 = _cache(tmp_path, monkeypatch)
    assert c2.load_or_compile("b64", _Lowered) == (("loaded", b"EXE:compiled-b64"), "loaded")
    assert c2.stats_["loads"] == 1 and c2.stats_["compiles"] == 0 and c2.stats_["refused"] == 0


@pytest.mark.parametrize("damage", ("no_line", "wrong_line", "truncated", "empty"))
def test_xla_file_without_a_matching_line_is_never_unpickled_and_is_rebuilt_once(tmp_path, monkeypatch, capfd, damage):
    p = _stored(tmp_path, monkeypatch)
    trap = pickle.dumps(_Trap())
    good = open(p, "rb").read()
    blob = {"no_line": trap, "wrong_line": b"sha256:" + b"0" * 64 + b"\n" + trap, "truncated": good[:-7], "empty": b""}[damage]
    with open(p, "wb") as f:
        f.write(blob)
    capfd.readouterr()
    c = _cache(tmp_path, monkeypatch, exact=True)                  # the exact tier too: a refused file is the absent file, not a failed load
    assert c.load_or_compile("b64", _Lowered) == (("loaded", b"EXE:compiled-b64"), "compiled+stored")
    err = capfd.readouterr().err
    assert TRIPPED == []
    assert err.count("REFUSED cache file " + p) == 1 and "sha256" in err and "fix:" in err and "treated as absent" in err
    assert c.stats_["refused"] == 1 and c.stats_["compiles"] == 1 and c.stats_["stores"] == 1 and c.stats_["load_failures"] == 0
    c2 = _cache(tmp_path, monkeypatch)                             # rebuilt once: the next process loads the rewritten file, silently
    assert c2.load_or_compile("b64", _Lowered) == (("loaded", b"EXE:compiled-b64"), "loaded")
    assert c2.stats_["loads"] == 1 and c2.stats_["refused"] == 0 and "REFUSED" not in capfd.readouterr().err


@pytest.mark.parametrize("target", ("directory", "file"))
@pytest.mark.parametrize("bit", (stat.S_IWGRP, stat.S_IWOTH))
def test_xla_group_or_other_writable_is_refused_with_the_fix_then_loads_after_it(tmp_path, monkeypatch, capfd, target, bit):
    p = _stored(tmp_path, monkeypatch)
    q = os.path.dirname(p) if target == "directory" else p
    os.chmod(q, stat.S_IMODE(os.stat(q).st_mode) | bit)
    monkeypatch.setattr(xla_cache.pickle, "loads", lambda b: pytest.fail("unpickled a refused file"))
    c = _cache(tmp_path, monkeypatch)
    assert c.load("b64") is None
    err = capfd.readouterr().err
    assert f"REFUSED cache file {p}: {target} {q} is writable by group or other" in err and f"fix: chmod go-w {q}" in err
    monkeypatch.undo()
    os.chmod(q, stat.S_IMODE(os.stat(q).st_mode) & ~0o022)         # the printed fix
    assert _cache(tmp_path, monkeypatch).load("b64") == ("loaded", b"EXE:compiled-b64")


def test_xla_foreign_owner_is_refused_for_a_user_and_accepted_for_uid_0(tmp_path, monkeypatch, capfd):
    p = _stored(tmp_path, monkeypatch)
    owner = os.stat(p).st_uid
    monkeypatch.setattr(os, "geteuid", lambda: owner + 1)          # this process is another (non-root) user
    if owner != 0:
        assert _cache(tmp_path, monkeypatch).load("b64") is None
        err = capfd.readouterr().err
        assert f"belongs to uid {owner}, not to this process (uid {owner + 1}) or root" in err and "fix:" in err
    monkeypatch.setattr(os, "geteuid", lambda: 0)                  # a container's root reading a bind-mounted host directory
    assert _cache(tmp_path, monkeypatch).load("b64") == ("loaded", b"EXE:compiled-b64")
    os.chmod(p, 0o666)                                             # …which still refuses what group or other can write
    assert _cache(tmp_path, monkeypatch).load("b64") is None


def test_xla_root_owned_file_is_accepted_for_a_user(tmp_path, monkeypatch):
    p = _stored(tmp_path, monkeypatch)
    real = os.stat

    def as_root_owned(path, *a, **k):
        st = real(path, *a, **k)
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, 0, 0, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))

    monkeypatch.setattr(os, "stat", as_root_owned)                 # an image's cache, read by a user
    monkeypatch.setattr(os, "fstat", lambda fd: as_root_owned(fd))
    monkeypatch.setattr(os, "geteuid", lambda: 12345)
    assert xla_cache.cache_refusal(p) is None


@pytest.mark.parametrize("umask", (0o002, 0o000, 0o077))
def test_xla_what_the_cache_writes_is_never_refused_whatever_the_umask(tmp_path, monkeypatch, capfd, umask):
    os.umask(umask)
    p = _stored(tmp_path, monkeypatch)
    assert os.stat(p).st_mode & 0o022 == 0 and os.stat(os.path.dirname(p)).st_mode & 0o022 == 0
    c = _cache(tmp_path, monkeypatch)
    assert c.load("b64") == ("loaded", b"EXE:compiled-b64") and c.stats_["refused"] == 0
    assert "REFUSED" not in capfd.readouterr().err


# ------------------------------------------------------------------------------------------------------------------ host.memo
def test_memo_entry_with_other_bytes_is_never_unpickled_and_is_rebuilt_once(tmp_path, capfd):
    d = str(tmp_path / "memo")
    m1 = Memo("msa", disk_dir=d)
    assert m1.get_or_build({"a": 1}, lambda: {"v": 7}) == {"v": 7}
    pkl = [os.path.join(d, n) for n in os.listdir(d) if n.endswith(".pkl")][0]
    with open(pkl, "wb") as f:
        f.write(pickle.dumps(_Trap()))
    m2 = Memo("msa", disk_dir=d)
    assert m2.get_or_build({"a": 1}, lambda: {"v": 7}) == {"v": 7}
    err = capfd.readouterr().err
    assert TRIPPED == [] and err.count("REFUSED cache entry " + pkl) == 1 and "sha256" in err and "fix:" in err
    assert m2.tally()["disk_corrupt"] == 1 and m2.tally()["disk_writes"] == 1
    m3 = Memo("msa", disk_dir=d)
    assert m3.get_or_build({"a": 1}, lambda: pytest.fail("rebuilt twice")) == {"v": 7} and m3.tally()["disk_hits"] == 1


@pytest.mark.parametrize("strict", (False, True))
def test_memo_group_writable_directory_is_refused_by_name_and_never_raised(tmp_path, capfd, strict):
    d = str(tmp_path / "memo")
    Memo("msa", disk_dir=d).get_or_build({"a": 1}, lambda: 1)
    os.chmod(d, 0o775)
    m = Memo("msa", disk_dir=d, strict_disk=strict)
    assert m.get_or_build({"a": 1}, lambda: 2) == 2                 # the absent entry: built
    err = capfd.readouterr().err
    assert "REFUSED cache entry" in err and f"{d} is writable by group or other (mode 0775); fix: chmod go-w {d}" in err
    assert m.tally()["disk_corrupt"] == 1 and m.tally()["disk_hits"] == 0


@pytest.mark.parametrize("umask", (0o002, 0o077))
def test_memo_what_the_mirror_writes_is_never_refused_whatever_the_umask(tmp_path, capfd, umask):
    os.umask(umask)
    d = str(tmp_path / "memo")
    Memo("msa", disk_dir=d).get_or_build({"a": 1}, lambda: 1)
    m = Memo("msa", disk_dir=d)
    assert m.get_or_build({"a": 1}, lambda: pytest.fail("rebuilt")) == 1 and m.tally()["disk_hits"] == 1
    assert "REFUSED" not in capfd.readouterr().err


# ------------------------------------------------------------------------------------------------------------------ rowpair.ckpt
def _ckpt():
    torch = pytest.importorskip("torch")
    from opt_core.mem.rowpair import ckpt as CK, shard as S
    return torch, CK, S


@pytest.mark.parametrize("umask", (0o002, 0o022))
def test_ckpt_tag_written_under_any_umask_loads_and_resumes(tmp_path, capfd, umask):
    torch, CK, S = _ckpt()
    os.umask(umask)
    N = 8
    lay = S.ctx(N, align=4)
    z, s = torch.arange(N * N * 2, dtype=torch.float32).reshape(N, N, 2), torch.ones(N, 3)
    ck = CK.TrunkCheckpointer("q", 7, N, root=str(tmp_path / "ck"), every=1, resume=True, keep=2, log=lambda m: None)
    ck.maybe_save_cycle(0, layout=lay, s=s, z_loc=z)
    ck.finalize()
    tagdir = os.path.join(ck.root, "cycle_000")
    assert all(os.stat(os.path.join(tagdir, n)).st_mode & 0o022 == 0 for n in os.listdir(tagdir)) and os.stat(tagdir).st_mode & 0o022 == 0
    st = ck.try_resume(layout=lay, device="cpu")
    assert st["tag"] == "cycle_000" and torch.equal(st["z_loc"], z)
    assert "REFUSED" not in capfd.readouterr().err


def test_ckpt_tag_another_account_could_write_is_the_absent_tag(tmp_path, capfd):
    torch, CK, S = _ckpt()
    N = 8
    lay = S.ctx(N, align=4)
    z, s = torch.zeros(N, N, 2), torch.ones(N, 3)
    ck = CK.TrunkCheckpointer("q", 7, N, root=str(tmp_path / "ck"), every=1, resume=True, keep=2, log=lambda m: None)
    ck.maybe_save_cycle(0, layout=lay, s=s, z_loc=z)
    ck.finalize()
    tagdir = os.path.join(ck.root, "cycle_000")
    os.chmod(tagdir, 0o777)
    assert ck.find_resume(lay) == (None, None) and ck.try_resume(layout=lay, device="cpu") is None
    with pytest.raises(FileNotFoundError, match="refused, treated as absent"):
        CK.load(ck.root, "cycle_000", rank=0, P=1)
    err = capfd.readouterr().err
    assert err.count(f"REFUSED checkpoint {tagdir}") == 1 and f"fix: chmod go-w {tagdir}" in err
    os.chmod(tagdir, 0o755)                                        # the printed fix
    assert ck.try_resume(layout=lay, device="cpu")["tag"] == "cycle_000"


def test_ckpt_sha256_is_checked_before_torch_load_even_with_verify_false(tmp_path, monkeypatch):
    torch, CK, S = _ckpt()
    CK.save_tag(str(tmp_path), "t", rank=0, P=1, shards={"z": torch.zeros(2)}, replicated={"s": torch.ones(2)}, rng=False)
    with open(os.path.join(str(tmp_path), "t", "rank0.pt"), "wb") as f:
        f.write(pickle.dumps(_Trap()))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("torch.load ran on a file whose sha256 differs"))
    with pytest.raises(IOError, match="sha256 mismatch"):
        CK.load(str(tmp_path), "t", rank=0, P=1, verify=False)
    assert TRIPPED == []
