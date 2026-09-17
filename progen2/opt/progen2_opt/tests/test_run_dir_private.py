"""The default stock run directory sits under <temp dir>/progen2-opt-<uid>: a predictable name in a shared directory, and the run directory
goes first on sys.path.  That parent serves only when it is this user's own (made here 0700, or an existing real directory this user owns
that group and others cannot write); anything else is refused by name and a private directory serves instead.  CPU only."""
import os, stat, tempfile

import pytest

from progen2_opt import stack


@pytest.fixture
def temp(tmp_path, monkeypatch):
    t = tmp_path / "shared_tmp"; t.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(t)); monkeypatch.setattr(stack, "_RUN_ROOTS", {})
    return t


def _refused(capsys):
    return [l for l in capsys.readouterr().err.splitlines() if l.startswith("[progen2-opt] run directory ") and " refused: " in l]


def test_absent_parent_is_made_private_and_says_nothing(temp, capsys):
    root = str(temp / "progen2-opt-me")
    assert stack._own_run_root(root) == root and stat.S_IMODE(os.stat(root).st_mode) == 0o700 and _refused(capsys) == []


def test_own_directory_others_cannot_write_is_kept(temp, capsys):
    root = temp / "progen2-opt-me"; root.mkdir(); os.chmod(root, 0o755)                       # as earlier runs of this user made it
    assert stack._own_run_root(str(root)) == str(root) and _refused(capsys) == []


@pytest.mark.parametrize("mode", [0o775, 0o757, 0o777, 0o1777])
def test_directory_group_or_others_can_write_is_refused_by_name(temp, capsys, mode):
    root = temp / "progen2-opt-me"; root.mkdir(); os.chmod(root, mode); (root / "planted.py").write_text("X = 1\n")
    use = stack._own_run_root(str(root))
    got = _refused(capsys)
    assert len(got) == 1 and f"run directory {root} refused: group or others can write it (mode {mode:04o})" in got[0] and "chmod go-w" in got[0] and stack.ENV_RUN_DIR in got[0] and use in got[0]
    assert use != str(root) and os.path.dirname(use) == str(temp) and os.path.basename(use).startswith("progen2-opt-me-") and stat.S_IMODE(os.stat(use).st_mode) == 0o700
    assert os.listdir(use) == [] and sorted(os.listdir(str(root))) == ["planted.py"]            # as if absent: nothing read from it, nothing written to it
    assert stack._own_run_root(str(root)) == use and _refused(capsys) == []                      # one line, one private directory per process


def test_symbolic_link_is_refused_by_name(temp, capsys, tmp_path):
    mine = tmp_path / "elsewhere"; mine.mkdir(); root = temp / "progen2-opt-me"; os.symlink(str(mine), str(root))
    use = stack._own_run_root(str(root)); got = _refused(capsys)
    assert len(got) == 1 and "it is a symbolic link" in got[0] and use != str(root) and os.listdir(str(mine)) == []


def test_another_owner_is_refused_by_name(temp, capsys, monkeypatch):
    root = temp / "progen2-opt-me"; root.mkdir(); os.chmod(root, 0o700); owner = os.stat(root).st_uid
    monkeypatch.setattr(os, "getuid", lambda: owner + 1)
    use = stack._own_run_root(str(root)); got = _refused(capsys)
    assert len(got) == 1 and f"it belongs to uid {owner}, not to this user (uid {owner + 1})" in got[0] and use != str(root)


def test_a_file_at_the_name_is_refused_by_name(temp, capsys):
    root = temp / "progen2-opt-me"; root.write_text("")
    use = stack._own_run_root(str(root)); got = _refused(capsys)
    assert len(got) == 1 and "it is not a directory" in got[0] and os.path.isdir(use) and use != str(root)


def _stock(tmp_path, monkeypatch):
    sd = tmp_path / "stock"; sd.mkdir()
    for name in stack.STOCK_LINKS:
        (sd / name).write_text("")
    monkeypatch.setattr(stack, "stock_dir", lambda: str(sd)); monkeypatch.setattr(stack, "weights_dir", lambda: None)
    monkeypatch.delenv(stack.ENV_RUN_DIR, raising=False)
    return sd


def test_run_directory_is_never_under_a_refused_parent(temp, capsys, tmp_path, monkeypatch):
    _stock(tmp_path, monkeypatch)
    uid = os.getuid() if hasattr(os, "getuid") else "u"
    root = temp / f"progen2-opt-{uid}"; root.mkdir(); os.chmod(root, 0o777); (root / "torch.py").write_text("raise SystemExit('planted')\n")
    named = stack.workdir(create=False)
    assert named.startswith(str(root) + os.sep) and _refused(capsys) == []                       # naming the directory reads nothing and decides nothing
    run = stack.workdir()
    assert len(_refused(capsys)) == 1 and not run.startswith(str(root) + os.sep) and os.path.isdir(os.path.join(run, "checkpoints"))
    assert all(os.path.islink(os.path.join(run, n)) for n in stack.STOCK_LINKS) and not os.path.exists(os.path.join(run, "torch.py"))
    assert stack.workdir() == run and _refused(capsys) == []


def test_a_clean_temp_dir_runs_from_the_per_user_parent_as_before(temp, capsys, tmp_path, monkeypatch):
    _stock(tmp_path, monkeypatch)
    uid = os.getuid() if hasattr(os, "getuid") else "u"
    run = stack.workdir()
    assert os.path.dirname(run) == str(temp / f"progen2-opt-{uid}") and os.path.basename(run).startswith("run-") and _refused(capsys) == []


def test_a_named_run_directory_is_used_as_given(temp, tmp_path, monkeypatch):
    _stock(tmp_path, monkeypatch)
    mine = tmp_path / "my_run"; monkeypatch.setenv(stack.ENV_RUN_DIR, str(mine))
    monkeypatch.setattr(stack, "_own_run_root", lambda root: (_ for _ in ()).throw(AssertionError("the rule is for the default parent only")))
    assert stack.workdir() == str(mine) and os.path.isdir(str(mine / "checkpoints"))
