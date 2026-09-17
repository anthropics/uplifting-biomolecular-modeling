"""schedules.py: the schedule cache directory ``design`` / ``warm`` hand upstream when the checkout's own cannot or must not be used (upstream
reads that cache with ``pickle.load``). Nothing is passed — and nothing printed — when ``inference.schedule_directory_path`` is typed, when there
is no checkout, or when ``<checkout>/schedules`` is this user's to write (or to make) and closed to others. An unwritable one, or one every user
can write (refused by name), gets ``${TMPDIR:-/tmp}/rfdiffusion1_schedules-uid<uid>``, made with mode 0700 and reused by the next run. A
symbolic link, a file, another user's directory or one with a group / other write bit at that path is refused on ONE stderr line that names it,
the reason and the remedy; nothing in it is touched and the run gets a private temporary directory, removed at exit. Root takes a directory of
any owner but not an open one. ``cli.main`` appends the token for design and warm only, last, and a dry run makes nothing."""
import os
import stat

import pytest

from .. import cli, schedules

KEY = schedules.KEY


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "rfd"; root.mkdir()
    monkeypatch.setenv("RFD_ROOT", str(root))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp")); (tmp_path / "tmp").mkdir()
    registered = []
    monkeypatch.setattr(schedules.atexit, "register", lambda fn, *a, **k: registered.append((fn, a, k)))
    return root, registered


def _per_user(tmp_path):
    return tmp_path / "tmp" / (schedules.NAME % os.getuid())


def _lock(root, monkeypatch):
    """The checkout's schedules directory made unwritable for this user (root writes through any mode: the probe is patched instead)."""
    d = root / "schedules"; d.mkdir(exist_ok=True); os.chmod(d, 0o555)
    if os.getuid() == 0:
        monkeypatch.setattr(schedules, "writable", lambda _d: False)
    return d


def _lines(capsys):
    return [l for l in capsys.readouterr().err.splitlines() if l.strip()]


def test_nothing_when_typed_absent_or_writable(checkout, tmp_path, monkeypatch, capsys):
    root, registered = checkout
    assert schedules.override([f"{KEY}=/x", "inference.num_designs=1"]) == []
    assert schedules.override([f"++{KEY}=/x"]) == []
    assert schedules.override(["inference.num_designs=1"]) == []                       # <checkout>/schedules absent, the checkout writable: upstream makes it
    (root / "schedules").mkdir(); os.chmod(root / "schedules", 0o755)
    assert schedules.override([]) == []                                                # present, this user's to write, closed to others
    os.chmod(root / "schedules", 0o775)
    assert schedules.override([]) == []                                                # a group write bit on the caller's own checkout is the caller's affair
    monkeypatch.setenv("RFD_ROOT", str(tmp_path / "no_such_checkout"))
    assert schedules.override([]) == []                                                # no checkout: upstream's affair
    assert _lines(capsys) == [] and not _per_user(tmp_path).exists() and registered == []


def test_an_unwritable_checkout_directory_gets_the_per_user_directory(checkout, tmp_path, monkeypatch, capsys):
    root, registered = checkout
    d = _lock(root, monkeypatch); t = _per_user(tmp_path)
    try:
        assert schedules.override(["inference.num_designs=1"]) == [f"{KEY}={t}"]
        st = os.lstat(t)
        assert stat.S_ISDIR(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o700 and st.st_uid == os.getuid()
        lines = _lines(capsys)
        assert len(lines) == 1 and f"schedule cache: {KEY}={t}" in lines[0] and f"{d} is not writable by uid {os.getuid()}" in lines[0] and "REFUSED" not in lines[0]
        (t / "T_50_cached.pkl").write_bytes(b"x")                                      # what the first run leaves ...
        assert schedules.override([]) == [f"{KEY}={t}"]                               # ... the next run of this user finds: the same directory
        assert os.listdir(t) == ["T_50_cached.pkl"] and registered == []
    finally:
        os.chmod(d, 0o755)


@pytest.mark.parametrize("mode", [0o777, 0o1777, 0o757])
def test_a_checkout_directory_every_user_can_write_is_refused_by_name(checkout, tmp_path, capsys, mode):
    root, _ = checkout
    d = root / "schedules"; d.mkdir(); os.chmod(d, mode)
    (d / "T_planted.pkl").write_bytes(b"not read")
    t = _per_user(tmp_path)
    assert schedules.override(["inference.num_designs=1"]) == [f"{KEY}={t}"]
    lines = _lines(capsys)
    assert len(lines) == 1, lines
    assert f"schedule cache: REFUSED {d}: every user can write it (mode {mode & 0o7777:04o})" in lines[0]
    assert f"chmod o-w {d}" in lines[0] and f"{KEY}={t} is used instead" in lines[0]
    assert os.listdir(t) == []                                                         # nothing carried over from the open directory


@pytest.mark.parametrize("plant", ["symlink", "file", "group-writable", "other-writable", "foreign"])
def test_a_planted_per_user_path_is_refused_on_one_line_and_the_run_gets_a_private_directory(checkout, tmp_path, monkeypatch, capsys, plant):
    root, registered = checkout
    d = _lock(root, monkeypatch); t = _per_user(tmp_path); bait = tmp_path / "bait"; bait.mkdir()
    (bait / "T_planted.pkl").write_bytes(b"not read")
    if plant == "symlink":
        os.symlink(bait, t)
    elif plant == "file":
        t.write_text("")
    elif plant in ("group-writable", "other-writable"):
        t.mkdir(); os.chmod(t, 0o720 if plant == "group-writable" else 0o702)
    else:
        if os.getuid() == 0:
            pytest.skip("root takes a directory of any owner (test_root_takes_any_owner_but_not_an_open_directory)")
        t.mkdir(mode=0o700)
        real = os.lstat
        monkeypatch.setattr(schedules.os, "lstat", lambda p: _Foreign(real(p)) if str(p) == str(t) else real(p))
    try:
        got = schedules.override(["inference.num_designs=1"])
        lines = _lines(capsys)
        assert len(lines) == 1, lines                                                  # ONE line: what, why, the remedy
        assert f"schedule cache: REFUSED {t}: it " in lines[0] and "nothing in it is read" in lines[0]
        assert f"remove or repair {t}" in lines[0] and f"type {KEY}=<a directory of yours>" in lines[0] and f"{d} is not writable" in lines[0]
        assert len(got) == 1 and got[0].startswith(f"{KEY}=")
        tmp = got[0].split("=", 1)[1]
        assert tmp != str(t) and os.path.dirname(tmp) == str(tmp_path / "tmp") and tmp in lines[0]
        st = os.lstat(tmp)
        assert stat.S_ISDIR(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o700 and os.listdir(tmp) == []
        assert [(fn, a) for fn, a, _ in registered] == [(schedules.shutil.rmtree, (tmp,))]   # removed at exit: the cache behaves as absent
        assert os.listdir(bait) == ["T_planted.pkl"]                                   # nothing written through the link, nothing taken from it
    finally:
        os.chmod(d, 0o755)


class _Foreign:
    """An lstat result whose owner is another uid (chown needs root; the rule reads st_uid and st_mode only)."""
    def __init__(self, st):
        self.st_mode, self.st_uid = st.st_mode, st.st_uid + 1


def test_root_takes_any_owner_but_not_an_open_directory(checkout, tmp_path, monkeypatch):
    t = tmp_path / "tmp" / "d"; t.mkdir(mode=0o700)
    owner = os.lstat(t).st_uid
    monkeypatch.setattr(schedules.os, "getuid", lambda: 0)
    real = os.lstat
    monkeypatch.setattr(schedules.os, "lstat", lambda p: _Foreign(real(p)))            # a directory of another uid, as a bind mount shows it to a root process
    assert owner + 1 != 0 and schedules.refusal(str(t)) is None
    os.chmod(t, 0o770)
    assert "writable by group or others (mode 0770)" in schedules.refusal(str(t))
    os.chmod(t, 0o707)
    assert "writable by group or others (mode 0707)" in schedules.refusal(str(t))


def test_a_dry_run_makes_nothing(checkout, tmp_path, monkeypatch, capsys):
    root, registered = checkout
    d = _lock(root, monkeypatch); t = _per_user(tmp_path)
    try:
        assert schedules.override([], dry_run=True) == [f"{KEY}={t}"] and not t.exists()   # the token a run would get; the directory is the run's to make
        t.mkdir(); os.chmod(t, 0o777)
        assert schedules.override([], dry_run=True) == []
        lines = _lines(capsys)
        assert len(lines) == 2 and f"REFUSED {t}: it is writable by group or others (mode 0777)" in lines[1] and "a run would compute its schedule anew every time" in lines[1]
        assert sorted(os.listdir(tmp_path / "tmp")) == [t.name] and registered == []
    finally:
        os.chmod(d, 0o755)


def test_cli_appends_the_token_for_design_and_warm_only(checkout, monkeypatch):
    seen = {}
    monkeypatch.setattr(schedules, "override", lambda ov, dry_run=False: seen.__setitem__("dry_run", dry_run) or ([f"{KEY}=/per/user"] if KEY not in " ".join(ov) else []))
    monkeypatch.setattr(cli, "cmd_design", lambda a: seen.__setitem__("design", list(a.overrides)) or 0)
    monkeypatch.setattr(cli, "cmd_warm", lambda a: seen.__setitem__("warm", list(a.overrides)) or 0)
    monkeypatch.setattr(cli, "cmd_check", lambda a: seen.__setitem__("check", list(a.overrides)) or 0)
    assert cli.main(["design", "--mode", "exact", "inference.input_pdb=t.pdb", "inference.output_prefix=o/d"]) == 0
    assert seen["design"] == ["inference.input_pdb=t.pdb", "inference.output_prefix=o/d", f"{KEY}=/per/user"] and seen["dry_run"] is False
    assert cli.main(["design", "--mode", "exact", "--dry-run", "inference.input_pdb=t.pdb"]) == 0 and seen["dry_run"] is True
    assert cli.main(["design", "--mode", "off", f"{KEY}=/mine", "inference.input_pdb=t.pdb"]) == 0
    assert seen["design"] == [f"{KEY}=/mine", "inference.input_pdb=t.pdb"]           # typed: the caller's directory, verbatim, nothing appended
    assert cli.main(["warm", "--mode", "exact", "--out_dir", "w"]) == 0 and seen["warm"] == [f"{KEY}=/per/user"]
    assert cli.main(["check", "--mode", "exact"]) == 0 and seen["check"] == []
