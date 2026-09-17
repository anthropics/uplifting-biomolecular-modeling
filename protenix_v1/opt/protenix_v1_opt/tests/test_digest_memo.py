"""Tests of the kit-local weights digest memo (identical text in every kit except the import line)."""
import json
import os

import pytest

from protenix_v1_opt import digest_memo as dm  # KIT: protenix_v1_opt


def _write(p, data):
    p.write_bytes(data)
    return str(p)


def test_memo_is_written_only_after_a_full_hash(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)

    def interrupted(_):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        dm.digest(f, str(tmp_path / "memo"), hasher=interrupted)
    assert not (tmp_path / "memo" / dm.MEMO_NAME).exists()


def test_a_hit_skips_hashing_and_names_the_cached_time(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = str(tmp_path / "memo")
    h1, c1 = dm.digest(f, memo)
    assert c1 is None and h1 == dm.sha256_file(f)
    calls = []
    h2, c2 = dm.digest(f, memo, hasher=lambda p: calls.append(p) or "0" * 64)
    assert calls == [] and h2 == h1 and c2 is not None
    assert dm.word("pinned", c2) == "pinned (cached digest %s)" % c2
    assert dm.word("unknown", None) == "unknown"


def test_a_changed_stat_key_is_hashed_again(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = str(tmp_path / "memo")
    dm.digest(f, memo)
    _write(tmp_path / "w.bin", b"y" * 101)
    h, c = dm.digest(f, memo)
    assert c is None and h == dm.sha256_file(f)


def test_refresh_hashes_afresh_and_rewrites_the_entry(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = str(tmp_path / "memo")
    dm.digest(f, memo, hasher=lambda p: "1" * 64)  # a stale entry
    calls = []
    h, c = dm.digest(f, memo, refresh=True, hasher=lambda p: calls.append(p) or dm.sha256_file(p))
    assert calls == [os.path.realpath(f)] and c is None and h == dm.sha256_file(f)
    table = json.load(open(os.path.join(memo, dm.MEMO_NAME)))
    assert table[dm.stat_key(f)]["sha256"] == h


def test_the_digest_decides_not_the_stat(tmp_path):
    a = _write(tmp_path / "a.bin", b"x" * 100)
    b = _write(tmp_path / "b.bin", b"z" * 100)  # same size, other bytes
    memo = str(tmp_path / "memo")
    ha, _ = dm.digest(a, memo)
    hb, cb = dm.digest(b, memo)
    assert ha != hb and cb is None


def test_rank_processes_read_the_launchers_entry_or_fail_by_name(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = str(tmp_path / "memo")
    with pytest.raises(RuntimeError, match="memo miss in a rank process"):
        dm.rank_digest(f, memo)
    h, _ = dm.digest(f, memo)
    assert dm.rank_digest(f, memo)[0] == h


def test_a_corrupt_memo_file_is_ignored_not_trusted(tmp_path):
    f = _write(tmp_path / "w.bin", b"x" * 100)
    memo = tmp_path / "memo"
    memo.mkdir()
    (memo / dm.MEMO_NAME).write_text("{not json")
    h, c = dm.digest(f, str(memo))
    assert c is None and h == dm.sha256_file(f)
