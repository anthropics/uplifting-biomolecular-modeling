"""The compile-cache root when the caller names none: the per-user directory ${TMPDIR:-/tmp}/model_opt_jit-uid<uid> (the one run.sh's
jit-cache block makes and README's route-B note names), this user's alone — made 0700, and used only when it is a real directory this uid
owns with no group / other write bit; anything else there is refused by name once and the process compiles into a fresh private root.
A named AF3_JAX_CACHE_ROOT, else MODEL_OPT_JIT_ROOT, is the root as given."""
import os
import re
import stat
from unittest import mock

import pytest

from af3_jax_opt import stack

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))            # .../af3_jax


@pytest.fixture
def shared_tmp(tmp_path, monkeypatch):
    """A world-writable sticky directory standing in for /tmp, the package default pointed under it, both root variables unset."""
    t = tmp_path / "tmp"; t.mkdir(); os.chmod(str(t), 0o1777)
    root = os.path.join(str(t), f"model_opt_jit-uid{os.geteuid()}")
    monkeypatch.setattr(stack, "DEFAULT_CACHE_ROOT", root)
    monkeypatch.delenv("AF3_JAX_CACHE_ROOT", raising=False); monkeypatch.delenv("MODEL_OPT_JIT_ROOT", raising=False)
    stack._CACHE.clear()
    yield root
    stack._CACHE.clear()


def test_default_root_is_the_per_user_directory_run_sh_and_readme_name():
    want = os.path.abspath(os.path.join(os.environ.get("TMPDIR") or "/tmp", f"model_opt_jit-uid{os.geteuid()}"))
    assert stack.DEFAULT_CACHE_ROOT == want
    run_sh = open(os.path.join(TREE, "run.sh"), encoding="utf-8").read()
    assert 'T="${TMPDIR:-/tmp}/model_opt_jit-uid$U"' in run_sh and "U=$(id -u)" in run_sh          # the block's per-user root: the same directory
    readme = open(os.path.join(TREE, "README.md"), encoding="utf-8").read()
    assert "/tmp/model_opt_jit-uid<uid>" in readme                                                     # the route-B note names it


def test_absent_default_is_made_private_and_used(shared_tmp, capsys):
    assert not os.path.exists(shared_tmp)
    assert stack.cache_root() == shared_tmp and stack.cache_root() == shared_tmp
    st = os.lstat(shared_tmp)
    assert stat.S_ISDIR(st.st_mode) and st.st_uid == os.geteuid() and stat.S_IMODE(st.st_mode) == 0o700, oct(st.st_mode)
    assert "REFUSED" not in capsys.readouterr().err
    stack._CACHE.clear()
    assert stack.cache_root() == shared_tmp and "REFUSED" not in capsys.readouterr().err                 # an earlier run's own directory: used again
    assert stack.cache_dir("KEY", "exact") == os.path.join(shared_tmp, "KEY")                            # the classes live under it


def _refused_once(shared_tmp, capsys, why_fragment):
    got = stack.cache_root()
    err = capsys.readouterr().err
    lines = [l for l in err.splitlines() if l.startswith("[af3-jax-opt] CACHE REFUSED ")]
    assert len(lines) == 1 and lines[0].startswith(f"[af3-jax-opt] CACHE REFUSED {shared_tmp}: {shared_tmp} {why_fragment}"), err
    assert got != shared_tmp and os.path.basename(got).startswith(stack.PRIVATE_CACHE_PREFIX) and os.path.isabs(got)
    st = os.lstat(got)
    assert stat.S_ISDIR(st.st_mode) and st.st_uid == os.geteuid() and stat.S_IMODE(st.st_mode) == 0o700
    assert stack.cache_root() == got and capsys.readouterr().err == ""                                    # resolved once per process: one line, one private root
    assert stack.cache_dir("KEY", "fast") == os.path.join(got, "KEY__fast")
    os.rmdir(got)
    return got


def test_a_symbolic_link_planted_at_the_default_is_refused_by_name(shared_tmp, capsys, tmp_path):
    theirs = tmp_path / "theirs"; theirs.mkdir(); (theirs / "KEY").mkdir()
    os.symlink(str(theirs), shared_tmp)
    _refused_once(shared_tmp, capsys, "is a symbolic link or not a directory")
    assert sorted(os.listdir(str(theirs))) == ["KEY"] and os.path.islink(shared_tmp)                     # nothing written through it, the link left as found


def test_a_group_or_other_writable_default_is_refused_by_name(shared_tmp, capsys):
    for mode in (0o777, 0o1777, 0o770, 0o702):
        if os.path.lexists(shared_tmp):
            os.rmdir(shared_tmp)
        os.mkdir(shared_tmp); os.chmod(shared_tmp, mode)
        stack._CACHE.clear()
        _refused_once(shared_tmp, capsys, f"is writable by group or other (mode {mode:04o})")


def test_a_default_another_uid_owns_is_refused_by_name(shared_tmp, capsys):
    os.mkdir(shared_tmp, 0o755)
    real = os.lstat(shared_tmp)
    other = os.stat_result((real.st_mode,) + tuple(real)[1:4] + (os.geteuid() + 1,) + tuple(real)[5:10])
    real_lstat = os.lstat
    with mock.patch.object(stack.os, "lstat", lambda p: other if p == shared_tmp else real_lstat(p)):
        _refused_once(shared_tmp, capsys, f"belongs to uid {os.geteuid() + 1}, not to this process (uid {os.geteuid()})")


def test_a_file_at_the_default_is_refused_by_name(shared_tmp, capsys):
    open(shared_tmp, "w").close()
    _refused_once(shared_tmp, capsys, "is a symbolic link or not a directory")


def test_named_roots_are_taken_as_given(shared_tmp, monkeypatch, tmp_path, capsys):
    mine = str(tmp_path / "mine")
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", mine)
    assert stack.cache_root() == mine and not os.path.exists(mine) and not os.path.exists(shared_tmp)   # run.sh's export stands in for the kit variable; nothing made here
    monkeypatch.setenv("AF3_JAX_CACHE_ROOT", "rel/cache")
    assert stack.cache_root() == os.path.abspath("rel/cache")                                              # the kit's own variable wins, absolute as before
    monkeypatch.delenv("AF3_JAX_CACHE_ROOT"); monkeypatch.delenv("MODEL_OPT_JIT_ROOT")
    assert stack.cache_root() == shared_tmp and os.path.isdir(shared_tmp)                                  # neither: the per-user default
    assert "REFUSED" not in capsys.readouterr().err


def test_refusal_wording_names_the_fix():
    src = open(stack.__file__, encoding="utf-8").read()
    assert re.search(r"fix: chmod go-w", src) and "fix: remove it, or set" in src
