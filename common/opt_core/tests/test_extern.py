"""opt_core.extern: digest_file / tree_files / digest_tree over out-of-git artifacts (never repo content -- see the module
docstring). Pure filesystem + hashlib, no framework."""
import hashlib
import os

import pytest

from opt_core import extern


def test_digest_file_matches_hashlib_and_is_stable(tmp_path):
    p = tmp_path / "weights.bin"
    p.write_bytes(b"some external artifact bytes" * 50000)                 # 28 * 50000 = 1_400_000 B ~= 1.33 MiB: genuinely over one 1 MiB chunk (1 << 20)
    want = hashlib.sha256(p.read_bytes()).hexdigest()
    assert extern.digest_file(str(p)) == want == extern.digest_file(str(p))


def test_digest_file_changes_with_the_bytes(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"v1")
    d1 = extern.digest_file(str(p))
    p.write_bytes(b"v2")
    assert extern.digest_file(str(p)) != d1


def test_tree_files_is_sorted_relative_and_prunes_caches(tmp_path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "junk.pyc").write_text("junk")
    (tmp_path / "d.pyc").write_text("junk2")
    assert extern.tree_files(str(tmp_path)) == ["a.txt", "b.txt", os.path.join("sub", "c.txt")]


def test_digest_tree_changes_with_any_file_and_with_the_file_set(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    base = extern.digest_tree(str(tmp_path))
    assert base == extern.digest_tree(str(tmp_path))                       # stable, deterministic across calls

    (tmp_path / "b.txt").write_text("b-changed")
    assert extern.digest_tree(str(tmp_path)) != base                       # a file's bytes changing moves the tree digest

    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "c.txt").write_text("c")
    assert extern.digest_tree(str(tmp_path)) != base                       # adding a file (same other bytes) also moves it


def test_digest_tree_files_argument_restricts_to_the_named_subset(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    only_a = extern.digest_tree(str(tmp_path), files=["a.txt"])
    (tmp_path / "b.txt").write_text("b-changed")                           # b is excluded from the subset: no effect on only_a
    assert extern.digest_tree(str(tmp_path), files=["a.txt"]) == only_a


def test_digest_tree_of_an_empty_directory_is_stable_and_not_a_crash(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert extern.digest_tree(str(empty)) == extern.digest_tree(str(empty))


def test_digest_file_of_a_missing_path_raises():
    with pytest.raises(OSError):
        extern.digest_file("/no/such/path/at/all")
