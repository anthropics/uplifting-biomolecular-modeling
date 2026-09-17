"""0.3.22: the persistent worker runs in the INVOKING directory (worker_launch.enter_invoking_dir), as the stock CLI does — relative paths
inside an input YAML (msa: / templates / constraints files) resolve there; the kit directory it was launched in stays importable
(BOLTZ_OPT_WORKDIR on sys.path); stack.run_worker hands the invoking directory over (BOLTZ_OPT_CALLER_CWD); a parse child of the zygote
inherits it. CPU only."""
import importlib
import os
import sys

import pytest

from .. import prep, stack, worker_launch as wl

TARGET = "boltz2_opt.tests._prep_targets"


def _dirs(tmp_path):
    kit = tmp_path / "o" / "_kit"; kit.mkdir(parents=True)
    (kit / "staged_probe_mod.py").write_text("WHERE = 'kit directory'\n")
    caller = tmp_path / "run_here"; (caller / "msa").mkdir(parents=True)
    (caller / "msa" / "a.csv").write_text("key,sequence\n-1,MKT\n")
    (caller / "a.yaml").write_text("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKT\n      msa: msa/a.csv\n")   # a RELATIVE msa path, as the ladder members carry
    return kit, caller


@pytest.fixture()
def restore_path():
    before = list(sys.path); env = {k: os.environ.get(k) for k in (wl.CALLER_CWD_ENV, wl.WORKDIR_ENV)}
    yield
    sys.path[:] = before
    for k, v in env.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    sys.modules.pop("staged_probe_mod", None)


def test_the_worker_enters_the_invoking_directory_and_keeps_the_kit_directory_importable(tmp_path, monkeypatch, restore_path):
    kit, caller = _dirs(tmp_path)
    monkeypatch.chdir(kit)                                              # stack.run_worker launches the worker with cwd = the kit directory
    monkeypatch.setenv(wl.CALLER_CWD_ENV, str(caller))                 # … and names the invoking directory
    now = wl.enter_invoking_dir(os.getcwd())
    assert os.path.realpath(now) == os.path.realpath(str(caller)) == os.path.realpath(os.getcwd()), "the process moved to the invoking directory"
    assert open("msa/a.csv").read().startswith("key,sequence"), "a YAML's relative msa path resolves where the stock CLI resolves it"
    assert str(kit) in sys.path and os.environ[wl.WORKDIR_ENV] == str(kit), "the kit directory stays importable, by name"
    assert importlib.import_module("staged_probe_mod").WHERE == "kit directory", "a staged module imports from the kit directory after the move"
    assert wl.script_path("bz_worker.py", str(kit)) == os.path.join(str(kit), "bz_worker.py") and wl.script_path("/abs/w.py", str(kit)) == "/abs/w.py"


def test_without_the_variable_the_worker_stays_where_it_was_launched(tmp_path, monkeypatch, restore_path):
    kit, _ = _dirs(tmp_path)
    monkeypatch.chdir(kit); monkeypatch.delenv(wl.CALLER_CWD_ENV, raising=False)
    assert os.path.realpath(wl.enter_invoking_dir(os.getcwd())) == os.path.realpath(str(kit)) == os.path.realpath(os.getcwd())
    monkeypatch.setenv(wl.CALLER_CWD_ENV, str(tmp_path / "no_such_dir"))   # a variable naming no directory: named on stderr, the process stays
    assert os.path.realpath(wl.enter_invoking_dir(os.getcwd())) == os.path.realpath(str(kit))


def test_a_parse_child_of_the_zygote_inherits_the_invoking_directory(tmp_path, monkeypatch, restore_path):
    """The zygote forks AFTER the move (worker_launch.main: enter_invoking_dir, then prep.start): every fresh parse child sees the caller's
    relative paths — here a child opens the YAML's `msa/a.csv` by that relative name."""
    kit, caller = _dirs(tmp_path)
    monkeypatch.chdir(kit); monkeypatch.setenv(wl.CALLER_CWD_ENV, str(caller))
    wl.enter_invoking_dir(os.getcwd())
    z = prep.Zygote(preload=(TARGET,)).start(forked_at="test")
    try:
        res = z.run(TARGET + ":read_relative", path="msa/a.csv")
        assert res.rc == 0, res.err
        assert f"cwd={os.path.realpath(str(caller))}" in res.out.replace(os.getcwd(), os.path.realpath(os.getcwd())) or f"cwd={caller}" in res.out
        assert "content=key,sequence" in res.out
        monkeypatch.chdir(kit)                                          # the parent moving afterwards does not move the zygote: its children keep the directory it forked in
        assert "content=key,sequence" in z.run(TARGET + ":read_relative", path="msa/a.csv").out
    finally:
        z.close()


def test_run_worker_hands_the_invoking_directory_to_the_worker_and_launches_it_in_the_kit_directory(tmp_path, monkeypatch):
    seen = {}
    def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, **kw):
        seen.update(cmd=cmd, cwd=cwd, env=dict(env or {}))
        class R: returncode = 0
        return R()
    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)                                         # the invoking directory
    given_env = {"BOLTZ_LEVERS": "resid,mask2"}
    rc = stack.run_worker("fast", str(tmp_path / "o" / "_kit"), str(tmp_path / "o" / "_kit" / "batch_x.json"), str(tmp_path / "x_worker.log"), env=given_env)
    assert rc == 0 and seen["cwd"] == str(tmp_path / "o" / "_kit"), "the worker is launched in the kit directory (its staged files) …"
    assert seen["env"][wl.CALLER_CWD_ENV] == str(tmp_path) and seen["env"]["BOLTZ_LEVERS"] == "resid,mask2", "… with the invoking directory named for it"
    assert wl.CALLER_CWD_ENV not in given_env, "the caller's env mapping is not written to"
    assert seen["cmd"][1:3] == ["-m", "boltz2_opt.worker_launch"] and "bz_worker.py" in seen["cmd"], "under the launcher, which enters the invoking directory before the zygote forks"
