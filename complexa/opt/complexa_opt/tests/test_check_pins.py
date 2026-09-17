"""stock/check_pins.py's gates on a CPU box: the carried configs are present (the carried stock/src tree read as a checkout); the weights
gate counts bytes before hashing; the upstream gate reports 'not installed' where proteinfoundation is absent; a checkout at the pinned
commit but locally dirty (or whose cleanliness can't be determined) is refused, never silently accepted."""
import json
import os
import subprocess

from complexa_opt import stack

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_pinned_configs_are_present_in_the_carried_checkout():
    cp = stack.check_pins()
    bad, detail = cp.check_configs(stack.pins(), checkout=os.path.join(TREE, "stock", "src"))
    assert bad == [] and detail["pinned"] is True and len(detail["files"]) == 4
    assert all(v == "present" for v in detail["files"].values()), detail["files"]


def test_weights_gate_counts_bytes_then_hashes(tmp_path):
    cp = stack.check_pins()
    d = tmp_path / "w"
    d.mkdir()
    bad, _ = cp.check_weights(stack.pins(), str(d))
    assert len(bad) == 2 and all("absent" in b for b in bad)
    (d / "complexa.ckpt").write_bytes(b"x" * 10)
    (d / "complexa_ae.ckpt").write_bytes(b"y" * 10)
    bad, detail = cp.check_weights(stack.pins(), str(d))
    assert len(bad) == 2 and all("bytes, the pin is" in b for b in bad) and detail["files"]["complexa.ckpt"]["bytes_ok"] is False
    assert "UNPINNED" in cp.weights_line(stack.pins(), str(d), pinned=False)
    assert cp.weights_line(stack.pins(), "/weights").endswith("complexa.ckpt=589db1741f29 complexa_ae.ckpt=35f8865efd26 (pinned)")


def test_upstream_gate_reports_on_this_box():
    cp = stack.check_pins()
    bad, detail = cp.check_upstream(stack.pins())
    if detail.get("version") is None:                                 # the CPU test box: proteinfoundation absent -> refused by name
        assert bad and "not installed" in bad[0] and detail["pinned"] is False
    else:                                                              # the stack image: must be the pin, clean, generate.py present
        assert bad == [] and detail["pinned"] is True and detail["generate_py_present"] is True
    rep = cp.stack_report(stack.pins(), import_torch=False)
    assert "image" not in rep and "torch" in rep["record"]   # the pinned stack is versions (pinned_stack.pins); the box the kit was tested on is a documentation line nothing compares against (tested_on)


def test_git_head_reports_zero_dirty_with_no_git_checkout(tmp_path):
    """No .git under the given path: version + presence are the gate (git_head names this, and reports 0 dirty rather than raising)."""
    cp = stack.check_pins()
    commit, note, dirty = cp.git_head(str(tmp_path))
    assert commit is None and dirty == 0 and "no .git" in note


def _git(*args, cwd):
    subprocess.run(["git", "-C", str(cwd)] + list(args), check=True, capture_output=True, text=True)


def _init_repo(root):
    """A minimal git checkout with one file under src/, committed; returns its HEAD commit."""
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.txt").write_text("a\n")
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "t@example.com", cwd=root)
    _git("config", "user.name", "t", cwd=root)
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "x", cwd=root)
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def test_git_head_detects_a_real_dirty_file_under_src(tmp_path):
    """git_head against a REAL git checkout: clean at HEAD reports dirty=0; editing a tracked file under src/ afterward reports dirty=1 —
    the actual mechanism check_upstream's 'not the pin's bytes' branch depends on."""
    cp = stack.check_pins()
    root = tmp_path / "checkout"
    head = _init_repo(root)
    commit, note, dirty = cp.git_head(str(root))
    assert commit == head and dirty == 0 and "clean" in note
    (root / "src" / "a.txt").write_text("edited\n")
    commit, note, dirty = cp.git_head(str(root))
    assert commit == head and dirty == 1 and "modified" in note


def test_check_upstream_refuses_a_pinned_commit_that_is_locally_dirty(tmp_path, monkeypatch):
    """The one branch that replaces the removed tree-hash's role: a checkout AT the pinned commit but with local edits under src/ is
    refused ('not the pin's bytes'), never silently accepted as 'pinned'. Drives check_upstream end to end against a real git checkout
    by faking only installed_upstream() (the CPU box has no proteinfoundation installed to point it at)."""
    cp = stack.check_pins()
    root = tmp_path / "checkout"
    head = _init_repo(root)
    monkeypatch.delenv(cp.ENV_UPSTREAM, raising=False)       # else checkout_dir() would read $LOCAL_CODE_PATH instead of deriving from pkg_dir
    # checkout_dir derives the checkout as two levels above pkg_dir (src layout); put pkg_dir at root/pkg/proteinfoundation so dirname(dirname(pkg_dir)) == root
    real_pkg_dir = root / "pkg" / "proteinfoundation"
    real_pkg_dir.mkdir(parents=True)
    (real_pkg_dir / "generate.py").write_text("x = 1\n")
    monkeypatch.setattr(cp, "installed_upstream", lambda: ("9.9.9", str(real_pkg_dir), {}))
    pins = json.loads(json.dumps(stack.pins()))                 # deep copy: don't mutate the real pin
    pins["upstream"]["proteinfoundation"] = dict(pins["upstream"]["proteinfoundation"], version="9.9.9", commit=head)

    bad, detail = cp.check_upstream(pins)
    assert bad == [] and detail["pinned"] is True and detail["dirty"] == 0        # clean at the pin: accepted

    (root / "src" / "a.txt").write_text("edited after the commit\n")             # dirty the checkout AFTER the pinned commit
    bad, detail = cp.check_upstream(pins)
    assert len(bad) == 1 and "not the pin's bytes" in bad[0] and detail["pinned"] is False and detail["dirty"] == 1


def test_check_upstream_refuses_when_git_status_itself_fails(tmp_path, monkeypatch):
    """A checkout AT the pinned commit whose cleanliness cannot be determined (git status errors) is refused, not assumed clean — the
    fail-closed counterpart of the dirty check above."""
    cp = stack.check_pins()
    root = tmp_path / "checkout"
    head = _init_repo(root)
    monkeypatch.delenv(cp.ENV_UPSTREAM, raising=False)
    real_pkg_dir = root / "pkg" / "proteinfoundation"
    real_pkg_dir.mkdir(parents=True)
    (real_pkg_dir / "generate.py").write_text("x = 1\n")
    monkeypatch.setattr(cp, "installed_upstream", lambda: ("9.9.9", str(real_pkg_dir), {}))
    pins = json.loads(json.dumps(stack.pins()))
    pins["upstream"]["proteinfoundation"] = dict(pins["upstream"]["proteinfoundation"], version="9.9.9", commit=head)

    real_run = subprocess.run

    def _fake_run(args, **kwargs):
        if len(args) >= 3 and args[1:3] == ["-C", str(root)] and "status" in args:
            class _R:
                returncode = 128
                stdout = ""
                stderr = "fatal: injected failure"
            return _R()
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    bad, detail = cp.check_upstream(pins)
    assert len(bad) == 1 and "refusing rather than assuming clean" in bad[0] and detail["pinned"] is False and detail["dirty"] is None
