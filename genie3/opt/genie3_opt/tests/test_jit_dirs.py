"""The compile caches `fast` fills (inductor: generated modules imported, graphs unpickled on a warm start) live in a directory this
user alone can have written: MODEL_OPT_JIT_ROOT as given (run.sh names one on every route), else the per-user root under the temporary
directory held to the shared core's directory rule — never a fixed name every account on the machine shares (jitdirs.py)."""
import os
import re
import stat
import tempfile

import pytest

from genie3_opt import jitdirs

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fresh(monkeypatch, tmp_path):
    """The temporary directory is tmp_path for this test; no root named; the core's per-path memo starts empty."""
    from opt_core.kernels.triattn_exact import _paths
    monkeypatch.delenv("MODEL_OPT_JIT_ROOT", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(_paths, "_CACHE_DIRS", {})
    return tmp_path / (jitdirs.PER_USER_ROOT % os.getuid())


def test_no_kit_source_names_a_fixed_directory_in_the_shared_tmp():
    """No package source falls back to a literal /tmp/<name> (the old inductor default was '/tmp' + 'inductor' for every user)."""
    hits = []
    for dirpath, dirnames, filenames in os.walk(PKG):
        dirnames[:] = [d for d in dirnames if d not in ("tests", "__pycache__")]
        for fn in filenames:
            if fn.endswith(".py"):
                for n, line in enumerate(open(os.path.join(dirpath, fn), encoding="utf-8"), 1):
                    if re.search(r"""["']/tmp\b""", line):
                        hits.append(f"{os.path.relpath(os.path.join(dirpath, fn), PKG)}:{n}: {line.strip()[:100]}")
    assert hits == [], hits


def test_a_named_root_is_used_as_given(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", "/x/y")
    assert jitdirs.jit_root() == "/x/y" and jitdirs.inductor_cache_dir() == "/x/y/inductor"
    assert not any(tmp_path.iterdir())                      # nothing made anywhere for a named root


def test_the_per_user_root_is_made_private_and_used_silently(monkeypatch, tmp_path, capsys):
    want = _fresh(monkeypatch, tmp_path)
    old = os.umask(0o002)                                   # a login umask that leaves group write on plain mkdir: the root is still 0700
    try:
        got = jitdirs.inductor_cache_dir()
    finally:
        os.umask(old)
    assert got == os.path.join(str(want), "inductor")
    st = os.lstat(want)
    assert stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and stat.S_IMODE(st.st_mode) == 0o700
    assert capsys.readouterr().err == ""
    assert jitdirs.inductor_cache_dir() == got               # a second call: same answer, still silent
    assert capsys.readouterr().err == ""


def test_an_earlier_private_root_of_this_user_is_reused_silently(monkeypatch, tmp_path, capsys):
    want = _fresh(monkeypatch, tmp_path)
    os.mkdir(want, 0o700); (want / "inductor").mkdir()
    assert jitdirs.jit_root() == str(want) and capsys.readouterr().err == ""


@pytest.mark.parametrize("mode", [0o777, 0o770, 0o702, 0o720])
def test_a_root_open_to_group_or_others_is_refused_by_name(monkeypatch, tmp_path, capsys, mode):
    planted = _fresh(monkeypatch, tmp_path)
    os.mkdir(planted); os.chmod(planted, mode); (planted / "inductor").mkdir()
    got = jitdirs.jit_root()
    err = capsys.readouterr().err
    assert got != str(planted) and os.path.isdir(got) and stat.S_IMODE(os.stat(got).st_mode) == 0o700
    assert "REFUSED cache directory %s" % planted in err and "writable by group or other" in err and "chmod go-w" in err, err
    assert err.count("\n") == 1                              # one line: what, why, the fix
    assert stat.S_IMODE(os.stat(planted).st_mode) == mode    # nothing is chmod-ed for the user
    assert jitdirs.inductor_cache_dir() == os.path.join(got, "inductor")


def test_a_symbolic_link_is_refused_by_name(monkeypatch, tmp_path, capsys):
    planted = _fresh(monkeypatch, tmp_path)
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir(mode=0o700)
    os.symlink(elsewhere, planted)
    got = jitdirs.jit_root()
    err = capsys.readouterr().err
    assert got not in (str(planted), str(elsewhere)) and "REFUSED cache directory" in err and "symbolic link" in err, err
    assert list(elsewhere.iterdir()) == []                   # nothing written through the link


def test_the_driver_defaults_the_inductor_cache_through_the_rule():
    """g3batch.py gives TORCHINDUCTOR_CACHE_DIR its default from jitdirs (setdefault: a caller's own directory is kept)."""
    src = open(os.path.join(PKG, "g3batch.py"), encoding="utf-8").read()
    assert 'os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", jitdirs.inductor_cache_dir())' in src
