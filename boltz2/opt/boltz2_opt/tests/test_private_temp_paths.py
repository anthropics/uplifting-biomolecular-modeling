"""The kit-picked places in the shared temporary directory are private to the invoking user: the featurized-storage transport's per-user
default ``<tempdir>/bz2xfer_<uid>`` (made 0700; a symbolic link, another account's directory or a group/other-writable one there is refused
by name and a fresh private directory stands in — the received files are mapped back as tensor bytes), and the development probe's row file
(beside the run in the launch's kit directory, else a private mkstemp file; never a fixed name). No torch, no GPU."""
import os
import stat
import tempfile
from unittest import mock

import pytest

from .. import msa2_probe as MP
from .. import xfer as XF


@pytest.fixture
def shared_tmp(tmp_path, monkeypatch):
    """A world-writable sticky directory standing in for the temporary directory; both transport variables unset."""
    t = tmp_path / "tmp"; t.mkdir(); os.chmod(str(t), 0o1777)
    monkeypatch.setattr(tempfile, "tempdir", str(t))
    monkeypatch.delenv(XF.ENV_DIR, raising=False); monkeypatch.delenv(XF.WORKDIR_ENV, raising=False)
    yield str(t)


def _default(shared_tmp):
    return os.path.join(shared_tmp, f"bz2xfer_{os.geteuid()}")


def test_absent_default_is_made_private_and_used(shared_tmp, capsys):
    d = _default(shared_tmp)
    assert not os.path.exists(d)
    assert XF.xfer_dir({}) == d and XF.xfer_dir({}) == d                    # made once, used again
    st = os.lstat(d)
    assert stat.S_ISDIR(st.st_mode) and st.st_uid == os.geteuid() and stat.S_IMODE(st.st_mode) == 0o700, oct(st.st_mode)
    assert "REFUSED" not in capsys.readouterr().err
    assert os.path.basename(d).startswith(XF.TMP_PREFIX)                    # cleanup's rule (the kit-owned default) still names it


def _refused(shared_tmp, capsys, why_fragment):
    d = _default(shared_tmp)
    got = XF.xfer_dir({})
    err = capsys.readouterr().err
    lines = [l for l in err.splitlines() if l.startswith("[boltz2-opt xfer] REFUSED: ")]
    assert len(lines) == 1 and lines[0].startswith(f"[boltz2-opt xfer] REFUSED: {d} {why_fragment}"), err
    assert got != d and os.path.dirname(got) == shared_tmp and os.path.basename(got).startswith(XF.TMP_PREFIX)
    st = os.lstat(got)
    assert stat.S_ISDIR(st.st_mode) and st.st_uid == os.geteuid() and stat.S_IMODE(st.st_mode) == 0o700
    os.rmdir(got)


def test_a_symbolic_link_planted_at_the_default_is_refused_by_name(shared_tmp, capsys, tmp_path):
    theirs = tmp_path / "theirs"; theirs.mkdir()
    os.symlink(str(theirs), _default(shared_tmp))
    _refused(shared_tmp, capsys, "is a symbolic link or not a directory")
    assert os.listdir(str(theirs)) == [] and os.path.islink(_default(shared_tmp))   # nothing written through it; left as found


def test_a_group_or_other_writable_default_is_refused_by_name(shared_tmp, capsys):
    for mode in (0o777, 0o1777, 0o770, 0o702):
        d = _default(shared_tmp)
        if os.path.lexists(d):
            os.rmdir(d)
        os.mkdir(d); os.chmod(d, mode)
        _refused(shared_tmp, capsys, f"is writable by group or other (mode {mode:04o})")


def test_a_default_another_uid_owns_is_refused_by_name(shared_tmp, capsys):
    d = _default(shared_tmp); os.mkdir(d, 0o755)
    real = os.lstat(d)
    other = os.stat_result((real.st_mode,) + tuple(real)[1:4] + (os.geteuid() + 1,) + tuple(real)[5:10])
    real_lstat = os.lstat
    with mock.patch.object(XF.os, "lstat", lambda p: other if p == d else real_lstat(p)):
        _refused(shared_tmp, capsys, f"belongs to uid {os.geteuid() + 1}, not to this process (uid {os.geteuid()})")


def test_named_directories_are_taken_as_given(shared_tmp, tmp_path, capsys):
    wd = tmp_path / "kit"; wd.mkdir()
    assert XF.xfer_dir({XF.ENV_DIR: str(tmp_path / "d")}) == str(tmp_path / "d") and not os.path.exists(str(tmp_path / "d"))
    assert XF.xfer_dir({XF.WORKDIR_ENV: str(wd)}) == str(wd / XF.SUBDIR) and not os.path.exists(str(wd / XF.SUBDIR))   # created on first use, as before
    assert not os.path.exists(_default(shared_tmp)) and "REFUSED" not in capsys.readouterr().err


def test_probe_rows_go_beside_the_run_or_to_a_private_file_never_a_fixed_name(shared_tmp, tmp_path, monkeypatch):
    src = open(MP.__file__, encoding="utf-8").read()
    assert '"/tmp/' not in src and "'/tmp/" not in src
    wd = tmp_path / "out" / "_kit"; wd.mkdir(parents=True)
    monkeypatch.setenv(MP.WORKDIR_ENV, str(wd)); monkeypatch.setitem(MP._STATE, "out", None)
    assert MP.out_path() == str(wd / MP.OUT_NAME) and MP.out_path() == str(wd / MP.OUT_NAME)
    monkeypatch.delenv(MP.WORKDIR_ENV); monkeypatch.setitem(MP._STATE, "out", None)
    p = MP.out_path()
    assert os.path.dirname(p) == shared_tmp and os.path.basename(p).startswith("msa2_probe_") and p.endswith(".jsonl")
    st = os.lstat(p)
    assert stat.S_ISREG(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o600 and st.st_uid == os.geteuid()
    assert MP.out_path() == p                                                # settled once per process
    os.remove(p)
