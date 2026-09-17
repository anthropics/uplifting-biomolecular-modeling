"""CPU tests for the CLI layer: argv pass-through to the stock CLI, the output directory left to stock, check (the core's dry run).

The core interface (enable/status/MODES) is replaced by a recording stub and the stock CLI by a click group with the same shape as
``runner.batch_inference.protenix_cli`` (``pred`` with ``-i/--input``, ``-o/--out_dir``, ...): no GPU, no protenix install needed.
Run: ``python -m pytest protenix_v2/opt/protenix_opt/tests`` with ``protenix_v2/opt`` importable (``pip install -e protenix_v2/opt``
or ``PYTHONPATH=protenix_v2/opt``)."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import protenix_opt
from protenix_opt import _autoload
from protenix_opt import cli, det, report
from protenix_opt.tests import _stock_stub
from protenix_opt.tests.conftest import needs_durable_install

click = pytest.importorskip("click")

TREE = Path(protenix_opt.__path__[0]).resolve().parents[1]          # protenix_v2/
HAS_BASH = shutil.which("bash") is not None
assert HAS_BASH, "bash is a precondition of the package (env.sh is sourced by bash), not a reason to skip"


# ----------------------------------------------------------------------------------------------------------------- fixtures
class CoreStub:
    """Records enable() calls; returns a contract-shaped activation report."""

    def __init__(self, active=True, dry_run_param=False, partial=False, applied=("T1:fused", "BLK:2")):
        self.calls: list[tuple] = []
        self.events: list[str] = []
        self.active = active
        self.partial = partial                                            # True: one lever fallen back (the ACTIVE partial form)
        self.applied = list(applied)
        self.last: dict | None = None
        if dry_run_param:
            def enable(mode, dry_run=False):
                return self._enable(mode, dry_run)
        else:
            def enable(mode):
                return self._enable(mode, False)
        self.enable = enable

    def _enable(self, mode, dry_run):
        self.calls.append((mode, dry_run))
        self.events.append("enable")
        active = self.active and mode != "off"
        partial = active and self.partial
        self.last = {"active": active, "mode": mode, "levers_applied": list(self.applied) if active else [],
                     "levers_fallback": ["PWAZ"] if partial else [], "partial": partial,
                     "fallback_reasons": {"PWAZ": "stub: refused(PWAZ: no cell on this GPU)"} if partial else {},
                     "gpu": {"name": "NVIDIA H100 80GB HBM3", "sm": 90},
                     "protenix_version": "2.0.0", "package_version": "0.0.test",
                     "reason": None if active else ("mode off" if mode == "off" else "stub: refused")}
        return self.last

    def status(self):
        return self.last or {"active": False, "reason": "not enabled"}


@pytest.fixture
def core(monkeypatch):
    stub = CoreStub()
    monkeypatch.setattr(protenix_opt, "enable", stub.enable, raising=False)
    monkeypatch.setattr(protenix_opt, "status", stub.status, raising=False)
    monkeypatch.setattr(protenix_opt, "MODES", {"exact": ["a"], "fast": ["b"], "off": []}, raising=False)
    monkeypatch.delenv("PROTENIX_OPT", raising=False)
    monkeypatch.delenv("PTX_LEVER_REPORT", raising=False)
    monkeypatch.delenv("PROTENIX_ROOT_DIR", raising=False)
    # the exit tally must be registered BEFORE the levers load (atexit is LIFO); record the order instead of registering for real
    monkeypatch.setattr(report, "register_exit_tally", lambda: stub.events.append("tally") or True)
    return stub


@pytest.fixture
def fake_stock(monkeypatch):
    """A stand-in for runner.batch_inference with a click group shaped like the stock one; records the argv it receives."""
    calls: dict = {"main_args": [], "pred": []}

    class Group(click.Group):
        def main(self, args=None, prog_name=None, **kw):
            calls["main_args"].append(list(args))
            return super().main(args=args, prog_name=prog_name, **kw)

    @click.group(cls=Group, context_settings=dict(help_option_names=["-h", "--help"]))
    def protenix_cli():
        pass

    @_stock_stub.pred_command                       # the stock `pred` option table (_stock_stub.PRED_OPTIONS): the full parsed set is recorded
    def predict(**params):
        calls["pred"].append(params)
        os.makedirs(params["out_dir"], exist_ok=True)
        if params["input"] == "FAIL":
            sys.exit(5)
        _stock_stub.write_outputs(params)              # the stock dumper's layout: the output census counts these

    protenix_cli.add_command(predict, name="pred")
    runner = types.ModuleType("runner")
    runner.__path__ = []  # type: ignore[attr-defined]
    bi = types.ModuleType("runner.batch_inference")
    bi.protenix_cli = protenix_cli  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runner", runner)
    monkeypatch.setitem(sys.modules, "runner.batch_inference", bi)
    return calls


# ----------------------------------------------------------------------------------------------------------------- split_mode
def test_split_mode_forms():
    assert cli.split_mode(["--mode", "fast", "--input", "x", "--out_dir", "y"]) == ("fast", ["--input", "x", "--out_dir", "y"])
    assert cli.split_mode(["--input", "x", "--mode=exact"]) == ("exact", ["--input", "x"])
    assert cli.split_mode(["--input", "x"]) == ("fast", ["--input", "x"])                     # --mode omitted = the package default (fast)
    assert cli.split_mode(["--mode", "exact", "--seeds", "0,1", "--mode", "off"]) == ("off", ["--seeds", "0,1"])
    assert cli.split_mode(["--msa_server_mode", "x"])[1] == ["--msa_server_mode", "x"]   # no abbreviation capture
    with pytest.raises(cli.CliError):
        cli.split_mode(["--input", "x", "--mode"])


def test_split_det_forms():
    """The port's own option --det (the stock parser rejects it) comes out of argv the way --mode does: both forms, last one wins, the rest
    unchanged and in order; an unknown level is a usage error."""
    assert cli.split_det(["--det", "0", "--input", "x"]) == (0, ["--input", "x"])
    assert cli.split_det(["--det=1"]) == (1, []) and cli.split_det(["--det", "1", "--det", "0"]) == (0, [])
    assert cli.split_det(["--input", "x"]) == (det.DEFAULT, ["--input", "x"]) == (0, ["--input", "x"])
    for bad in (["--det"], ["--det", "2"], ["--det", "x"], ["--det=-1"]):
        with pytest.raises(cli.CliError):
            cli.split_det(bad)


# ----------------------------------------------------------------------------------------------------------------- pred
def _stock_proofs(tmpdir) -> list:
    """The stock route's environment proofs written under a temporary-directory root (tempfile.tempdir pointed at ``tmpdir``)."""
    import glob
    return sorted(glob.glob(os.path.join(str(tmpdir), "protenix_opt_stock_*", cli.STOCK_PROOF_NAME)))


def test_pred_passthrough_lines_and_env(core, fake_stock, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    rest = ["--input", "in.json", "--out_dir", str(out), "--seeds", "0,1", "--use_msa", "false"]
    rc = cli.main(["pred", "--mode", "fast", *rest])
    assert rc == 0
    assert core.calls == [("fast", False)]
    assert core.events == ["tally", "enable"], "exit tally registered before enable() so it runs after the kit's atexit dump"
    assert fake_stock["main_args"] == [["pred", *rest]], "stock argv must pass through unchanged and in order"
    assert fake_stock["pred"] == [{**_stock_stub.STOCK_PRED_DEFAULTS, "input": "in.json", "out_dir": str(out), "seeds": "0,1", "use_msa": False}]
    assert os.environ["PROTENIX_OPT"] == "fast"
    assert os.path.basename(os.environ["PTX_LEVER_REPORT"]) == cli.LEVER_REPORT_NAME and not os.environ["PTX_LEVER_REPORT"].startswith(str(out))   # the kit's records live outside --out_dir
    err = capsys.readouterr().err
    assert "[protenix-opt] FINAL mode=fast n_gpu=1 sharding=none levers=T1:fused,BLK:2 fallbacks=none partial=false reconciled_with=none" in err
    assert "[protenix-opt] OUTPUTS complete 20/20 (entries=1 seeds=2 samples=5)" in err, "1 entry x 2 seeds x 5 samples, a CIF + a summary each"
    assert not [f for f in os.listdir(out) if f.endswith(".json")], "no kit file at the output directory's root (the stand-in's stock tree only)"


def test_pred_short_options_and_default_out_dir(core, fake_stock, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["pred", "-i", "a.json", "-o", "res", "-s", "7"])
    assert rc == 0 and fake_stock["main_args"] == [["pred", "-i", "a.json", "-o", "res", "-s", "7"]]
    assert fake_stock["pred"][-1]["out_dir"] == "res" and (tmp_path / "res").is_dir()
    rc = cli.main(["pred", "-i", "b.json"])                       # stock default --out_dir ./output, resolved by the stock parser
    assert rc == 0 and fake_stock["pred"][-1]["input"] == "b.json" and fake_stock["pred"][-1]["out_dir"] == "./output"


def test_pred_stock_exit_code_propagates(core, fake_stock, tmp_path):
    out = tmp_path / "o"
    rc = cli.main(["pred", "--input", "FAIL", "--out_dir", str(out)])
    assert rc == 5


def test_pred_refuses_when_not_active(core, fake_stock, tmp_path, capsys):
    core.active = False
    rc = cli.main(["pred", "--mode", "exact", "--input", "x", "--out_dir", str(tmp_path / "o")])
    assert rc == cli.EXIT_NOT_ACTIVE
    assert fake_stock["main_args"] == [], "the stock CLI must not run when the levers refused"
    assert "NOT ACTIVE: stub: refused" in capsys.readouterr().err
    assert not (tmp_path / "o").exists(), "nothing runs, nothing is written"


@needs_durable_install
def test_pred_mode_off_runs_stock_in_a_clean_subprocess(core, tmp_path, monkeypatch):
    """off: the stock CLI in a fresh, proven subprocess (test_stock_route.py has the proof's contents); the pred process exports nothing."""
    root = _stock_stub.write_stub_runner(str(tmp_path / "stub"))
    for m in [m for m in sys.modules if m == "runner" or m.startswith("runner.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.syspath_prepend(root)
    monkeypatch.setenv("PYTHONPATH", root)
    monkeypatch.setattr(cli.tempfile, "tempdir", str(tmp_path))
    out = tmp_path / "o"
    rc = cli.main(["pred", "--mode", "off", "--input", "x", "--out_dir", str(out)])
    assert rc == 0 and core.calls == [("off", False)]
    proof = json.load(open(_stock_proofs(tmp_path)[-1], encoding="utf-8"))
    assert proof["ok"] is True and "PROTENIX_OPT" not in os.environ
    assert (out / _stock_stub.RUN_RECORD).is_file() and not (out / cli.STOCK_PROOF_NAME).exists(), "the proof never lands in the output directory"


REACH = ["--cycle", "1", "--step", "2", "--sample", "1"]          # one trunk pass, one 2-step sample: stock knobs passed through
REACH_PARSED = {"cycle": 1, "step": 2, "sample": 1}                # REACH through the stock-shaped parser (_stock_stub.PRED_OPTIONS)


def test_pred_stock_knobs_pass_through_verbatim(core, fake_stock, tmp_path):
    """The stock knobs (--cycle/--step/--sample/--seeds …: the stock parser's names) reach the stock CLI verbatim and in order on a kit
    mode; a later occurrence wins (click: last occurrence); nothing is injected."""
    out = tmp_path / "out"
    rest = ["--input", "in.json", "--out_dir", str(out), "--seeds", "7", "--sample", "3"]
    rc = cli.main(["pred", "--mode", "fast", *REACH, *rest])
    assert rc == 0
    assert fake_stock["main_args"] == [["pred", *REACH, *rest]], "the caller's tokens reach the stock CLI unchanged and in order"
    assert fake_stock["pred"] == [{**_stock_stub.STOCK_PRED_DEFAULTS, **REACH_PARSED, "input": "in.json", "out_dir": str(out), "seeds": "7", "sample": 3}], \
        "every knob parsed by the stock-shaped parser; the explicit --sample 3 given later wins"
    for argv in (["pred", "--settings", "reach_probe", "--input", "x", "--out_dir", str(tmp_path / "s")], ["pred", "--settings=upstream", "--input", "x", "--out_dir", str(tmp_path / "s")]):
        fake_stock["main_args"].clear()
        cli.main(argv)
        assert fake_stock["main_args"] == [argv], "--settings is not a port option: it reaches the stock parser like any unknown token"


def test_pred_without_port_options_is_byte_identical_and_det_0_changes_nothing(core, fake_stock, tmp_path):
    """No port option and --det 0 (the default) leave the stock argv byte-identical to the caller's and export nothing."""
    rest = ["--input", "in.json", "--out_dir", str(tmp_path / "a"), "--seeds", "0,1", "--use_msa", "false"]
    assert cli.main(["pred", *rest]) == 0
    assert cli.main(["pred", "--det", "0", *rest]) == 0
    assert cli.main(["pred", "--det=0", *rest]) == 0
    assert fake_stock["main_args"] == [["pred", *rest]] * 3
    assert "PTX_DET" not in os.environ and "CUBLAS_WORKSPACE_CONFIG" not in os.environ, "--det 0 exports nothing"


def test_pred_det_1_with_problems_is_refused_by_name_and_bad_values_are_usage(core, fake_stock, tmp_path, capsys, monkeypatch):
    """--det 1 whose plan names a problem (here: no installed protenix package) refuses with rc 1 before the levers or the stock CLI run,
    exports nothing; bad levels are usage errors."""
    monkeypatch.setattr(det, "protenix_package_dir", lambda: None)
    monkeypatch.delenv("PTX_DET", raising=False); monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    rc = cli.main(["pred", "--det", "1", "--input", "x", "--out_dir", str(tmp_path / "o")])
    assert rc == cli.EXIT_FAIL and fake_stock["main_args"] == [] and core.calls == [], "refused before the levers or the stock CLI run"
    err = capsys.readouterr().err
    assert "--det 1 refused: protenix package not installed" in err and "PTX_DET" not in os.environ
    assert not (tmp_path / "o").exists()
    for argv in (["pred", "--det", "2", "--input", "x"], ["pred", "--det", "x", "--input", "x"]):
        assert cli.main(argv) == cli.EXIT_USAGE, argv
    assert fake_stock["main_args"] == [] and core.calls == []


def test_pred_help_works_with_det_present(core, fake_stock, capsys):
    rc = cli.main(["pred", "--det", "0", "--help"])
    assert rc == 0 and fake_stock["main_args"] == [["pred", "--help"]] and core.calls == []
    cap = capsys.readouterr()
    opts = "[--det 0|1] <stock `protenix pred` arguments>"     # the port's options, then the stock arguments, in both help texts
    assert "--out_dir" in cap.out, cap.out[:300]
    assert opts in cap.err and "--settings" not in cap.err, cap.err[:300]
    assert opts in cli.USAGE and "--settings" not in cli.USAGE


@needs_durable_install
def test_pred_mode_off_passes_the_stock_knobs_through(core, tmp_path, monkeypatch):
    """The off route (the clean subprocess) receives the caller's stock knobs verbatim and in order (a later occurrence winning)."""
    root = _stock_stub.write_stub_runner(str(tmp_path / "stub"))
    for m in [m for m in sys.modules if m == "runner" or m.startswith("runner.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.syspath_prepend(root)
    monkeypatch.setenv("PYTHONPATH", root)
    out = tmp_path / "o"
    rest = ["--input", "x", "--out_dir", str(out), "--seeds", "7", "--sample", "3"]
    monkeypatch.setattr(cli.tempfile, "tempdir", str(tmp_path))
    rc = cli.main(["pred", "--mode", "off", "--det", "0", *REACH, *rest])
    assert rc == 0 and core.calls == [("off", False)]
    rec = json.loads((out / _stock_stub.RUN_RECORD).read_text())
    assert rec["params"] == {**_stock_stub.STOCK_PRED_DEFAULTS, **REACH_PARSED, "input": "x", "out_dir": str(out), "seeds": "7", "sample": 3}, "the knobs parsed by the stock parser; the later --sample 3 wins in the subprocess too"
    proof = json.load(open(_stock_proofs(tmp_path)[-1], encoding="utf-8"))
    assert proof["det_exception"] is None and proof["stock_argv"] == ["pred", *REACH, *rest]


def test_pred_keeps_user_lever_report(core, fake_stock, tmp_path, monkeypatch):
    monkeypatch.setenv("PTX_LEVER_REPORT", str(tmp_path / "mine.jsonl"))
    assert cli.main(["pred", "--input", "x", "--out_dir", str(tmp_path / "o")]) == 0
    assert os.environ["PTX_LEVER_REPORT"] == str(tmp_path / "mine.jsonl")


def test_pred_enable_exception_is_a_named_refusal(core, fake_stock, tmp_path, monkeypatch, capsys):
    def boom(mode):
        raise RuntimeError("protenix==1.9.9; this kit targets 2.0.0 exactly")
    monkeypatch.setattr(protenix_opt, "enable", boom)
    rc = cli.main(["pred", "--input", "x", "--out_dir", str(tmp_path / "o")])
    assert rc == cli.EXIT_NOT_ACTIVE and "protenix==1.9.9" in capsys.readouterr().err and fake_stock["main_args"] == []


def test_unknown_mode_and_command(core, capsys):
    assert cli.main(["check", "--mode", "bogus"]) == cli.EXIT_USAGE
    assert "unknown --mode 'bogus'" in capsys.readouterr().err
    assert cli.main(["frobnicate"]) == cli.EXIT_USAGE
    assert cli.main([]) == cli.EXIT_USAGE
    assert cli.main(["--help"]) == 0


# ----------------------------------------------------------------------------------------------------------------- check
def test_check_is_the_cores_dry_run(core, monkeypatch, capsys):
    """check = stack.activate(mode, dry_run=True): resolve + gates + report, nothing applied; the core prints the line (one formatter)."""
    from protenix_opt import report, stack
    calls = []

    def fake_activate(mode, strict=False, trigger=None, dry_run=False, det=None):
        calls.append((mode, dry_run))
        rep = {"active": False, "dry_run": True, "mode": mode, "protenix_version": "2.0.0", "package_version": "0.1.0",
               "gpu": {"name": "NVIDIA H100 80GB HBM3", "sm": "sm90"}, "levers_applied": ["t1_fused_transition", "blk2_block_path"],
               "env": {"PTX_BLK": "2"}, "kernel_key": "9.0|3.3", "prebuilt": "/pre",
               "reason": "dry run: resolved and gated, nothing applied"}
        report.log_activation(rep); rep["logged"] = True
        return rep
    monkeypatch.setattr(stack, "activate", fake_activate)
    rc = cli.main(["check", "--mode", "fast", "--json"])
    cap = capsys.readouterr()
    assert rc == 0 and calls == [("fast", True)] and core.calls == []
    assert cap.err.count("[protenix-opt] DRY-RUN mode=fast") == 1, cap.err
    assert "levers=t1_fused_transition,blk2_block_path env_vars=1 kernel_key=9.0|3.3 prebuilt=/pre" in cap.err
    assert "served=" not in cap.err
    from protenix_opt import kits
    assert "[protenix-opt] KIT " not in cap.err and "[protenix-opt] DRY-RUN" in cap.err     # no per-unit file account: the tree's identity is its commit
    rep = json.loads(cap.out)
    assert rep["mode"] == "fast" and rep["dry_run"] is True and "kits" not in rep
    assert "PTX_BLK" not in os.environ

    def refusing(mode, strict=False, trigger=None, dry_run=False, det=None):
        rep = {"active": False, "dry_run": True, "mode": mode, "reason": "protenix==2.1.0; this kit targets 2.0.0 exactly"}
        report.log_activation(rep); rep["logged"] = True
        return rep
    monkeypatch.setattr(stack, "activate", refusing)
    assert cli.main(["check"]) == cli.EXIT_NOT_ACTIVE, "a dry run that would not activate: the code pred exits with"
    cap = capsys.readouterr()
    assert cap.err.count("[protenix-opt] NOT ACTIVE: protenix==2.1.0") == 1 and cap.out == ""
    assert cli.main(["check", "--mode", "off"]) == 0


def test_check_real_dry_run_on_this_box(core, capsys):
    """Against the real core on a CPU box: sources env.sh (or refuses on the protenix pin), applies nothing, exports nothing."""
    before = dict(os.environ)
    rc = cli.main(["check", "--mode", "exact"])
    err = capsys.readouterr().err
    assert rc in (0, cli.EXIT_FAIL, cli.EXIT_NOT_ACTIVE) and ("[protenix-opt] DRY-RUN mode=exact" in err or "[protenix-opt] NOT ACTIVE:" in err)
    assert err.count("[protenix-opt] DRY-RUN") + err.count("[protenix-opt] NOT ACTIVE") == 1, "exactly one activation line: the core's (the console-script note is separate)"
    assert {k: v for k, v in os.environ.items() if k.startswith("PTX_")} == {k: v for k, v in before.items() if k.startswith("PTX_")}


def _bin_with(tmp_path, scripts: dict, nvidia_smi_cc: str | None = None) -> str:
    b = tmp_path / "bin"; b.mkdir(parents=True, exist_ok=True)
    for name, body in scripts.items():
        p = b / name; p.write_text(body); p.chmod(0o755)
    if nvidia_smi_cc is not None:
        p = b / "nvidia-smi"; p.write_text(f"#!/bin/bash\necho {nvidia_smi_cc}\n"); p.chmod(0o755)
    return str(b)


@needs_durable_install
def test_check_fails_when_the_protenix_command_runs_another_interpreter(core, monkeypatch, tmp_path, capsys):
    """Venv hazard (25): the stock `protenix` console script is bound to its shebang's interpreter; PROTENIX_OPT=<mode> protenix pred activates
    only if THAT interpreter imports this protenix_opt. The test guards the ENV route only: with the package INSTALLED here and the script's
    interpreter not importing it, check FAILS loudly (the hazard); the same interpreter passes; the package importable but NOT installed (a
    PYTHONPATH tree) is an advisory line — the env route is unavailable, the CLI route runs in-process — rc unchanged; no command on PATH is
    an advisory too."""
    from protenix_opt import report, stack
    monkeypatch.setattr(stack, "activate", lambda mode, strict=False, trigger=None, dry_run=False, det=None:
                        {"active": False, "dry_run": True, "mode": mode, "env": {}, "logged": True, "reason": "dry run"})
    monkeypatch.setattr(report, "log_activation", lambda rep: None)
    monkeypatch.setattr(cli, "package_installed", lambda name=cli.PACKAGE_DIST: {"installed": True, "version": "0.1.0"})
    other = tmp_path / "other-python"; other.write_text("#!/bin/bash\nexit 1\n"); other.chmod(0o755)      # an interpreter that cannot import protenix_opt
    # (a) the hazard: installed here, the protenix command's shebang names another interpreter
    monkeypatch.setenv("PATH", _bin_with(tmp_path, {"protenix": f"#!{other}\n"}) + os.pathsep + os.environ["PATH"])
    rc = cli.main(["check", "--mode", "off", "--json"])
    out = capsys.readouterr()
    assert rc == cli.EXIT_FAIL and "[protenix-opt] CHECK FAIL:" in out.err and "would run STOCK silently" in out.err and f"{other} -m pip install -e" in out.err
    cs = json.loads(out.out)["console_script"]
    assert cs["status"] == "fail" and cs["ok"] is False and cs["installed"] is True and cs["version"] == "0.1.0"
    assert cs["interpreter"] == str(other) and cs["imports_this_package"] is False and cs["package"] == os.path.abspath(protenix_opt.__file__)
    # (b) the same interpreter owns the command: ok, no line
    monkeypatch.setenv("PATH", _bin_with(tmp_path, {"protenix": f"#!{sys.executable}\n"}) + os.pathsep + os.environ["PATH"])
    rc = cli.main(["check", "--mode", "off", "--json"])
    out = capsys.readouterr()
    cs = json.loads(out.out)["console_script"]
    assert rc == 0 and "CHECK" not in out.err and cs["status"] == "ok" and cs["ok"] is True
    # (c) `#!/usr/bin/env python`: PATH decides; python -> this interpreter
    monkeypatch.setenv("PATH", _bin_with(tmp_path, {"protenix": "#!/usr/bin/env python\n", "python": f"#!/bin/bash\nexec {sys.executable} \"$@\"\n"}) + os.pathsep + os.environ["PATH"])
    assert cli.main(["check", "--mode", "off"]) == 0
    # (e) importable but NOT installed (a PYTHONPATH tree) + the script's interpreter cannot import it: advisory, rc unchanged
    monkeypatch.setattr(cli, "package_installed", lambda name=cli.PACKAGE_DIST: {"installed": False, "version": None})
    monkeypatch.setenv("PATH", _bin_with(tmp_path, {"protenix": f"#!{other}\n"}) + os.pathsep + os.environ["PATH"])
    rc = cli.main(["check", "--mode", "off", "--json"])
    out = capsys.readouterr()
    cs = json.loads(out.out)["console_script"]
    assert rc == 0 and "CHECK FAIL" not in out.err
    assert "[protenix-opt] CHECK console script: env route unavailable — protenix_opt not installed (PYTHONPATH route; the CLI route runs in-process)" in out.err
    assert cs["status"] == "advisory" and cs["ok"] is None and cs["installed"] is False and cs["reason"] == cli.NOT_INSTALLED_REASON and cs["imports_this_package"] is False
    # not installed but the script's interpreter imports this package anyway (a .pth of its own): ok
    monkeypatch.setenv("PATH", _bin_with(tmp_path, {"protenix": f"#!{sys.executable}\n"}) + os.pathsep + os.environ["PATH"])
    rc = cli.main(["check", "--mode", "off", "--json"])
    out = capsys.readouterr()
    assert rc == 0 and json.loads(out.out)["console_script"]["status"] == "ok"
    # (d) no protenix command at all: an advisory, rc unchanged
    monkeypatch.setenv("PATH", _bin_with(tmp_path / "empty", {}))
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    rc = cli.main(["check", "--mode", "off", "--json"])
    out = capsys.readouterr()
    cs = json.loads(out.out)["console_script"]
    assert rc == 0 and "CHECK FAIL" not in out.err and cs["status"] == "advisory" and cs["ok"] is None and "no `protenix` command on PATH" in cs["reason"]
    assert "[protenix-opt] CHECK console script: env route unavailable — no `protenix` command on PATH" in out.err


def test_package_installed_reads_the_distribution(monkeypatch):
    """The installed/not-installed fact is importlib.metadata's: a distribution of the package in the calling interpreter."""
    import importlib.metadata
    r = cli.package_installed()
    assert set(r) == {"installed", "version"} and isinstance(r["installed"], bool)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)))
    assert cli.package_installed() == {"installed": False, "version": None}
    assert cli.package_installed("no-such-dist-xyz") == {"installed": False, "version": None}


def test_jit_cache_key():
    """(24) the cache-key rule: torch<version minus local tag>-cu<torch.version.cuda sans dot>-sm<cc digits>."""
    from protenix_opt.modes import jit_cache_key
    assert jit_cache_key("2.13.0+cu130", None, "9.0") == "torch2.13.0-cu130-sm90"
    assert jit_cache_key("2.7.1+cu126", None, "9.0") == "torch2.7.1-cu126-sm90"
    assert jit_cache_key("2.13.0", "13.0", "10.0") == "torch2.13.0-cu130-sm100"
    assert jit_cache_key("2.13.0+cu130", "13.0", "9.0") == "torch2.13.0-cu130-sm90"
    from opt_core import gates, jit_cache as J                                                     # the defaults resolve without importing torch, or raise by name
    from protenix_opt.modes import nvidia_smi_compute_cap
    if gates.dist_version("torch") or J.version_py_facts("torch").get("version"):
        assert jit_cache_key(None, "13.0", "9.0").startswith("torch")
    else:
        with pytest.raises(J.StackKeyUnknown, match="version"):
            jit_cache_key(None, "13.0", "9.0")
    if nvidia_smi_compute_cap() or gates.nvidia_smi_probe().get("cc"):
        assert jit_cache_key("2.13.0+cu130", None, None).startswith("torch2.13.0-cu130-sm")
    else:
        with pytest.raises(J.StackKeyUnknown, match="cc"):
            jit_cache_key("2.13.0+cu130", None, None)


def test_config_h100_keys_the_caches_by_the_stack_key(tmp_path, monkeypatch):
    """`source configs/h100.env` exports MODEL_OPT_STACK_KEY = jit_cache_key() (a stub torch 2.13.0+cu130 on PYTHONPATH, a fake nvidia-smi 9.0)
    and, under the optional MODEL_OPT_JIT_ROOT, the cache dirs <root>/<key>/{torch_extensions,triton} + <root>/weights (nothing exported for them
    when the root is unset); a pre-set key or cache directory is kept as given; the weights root PROTENIX_ROOT_DIR has no default and is refused
    by name (rc 3) when unset."""
    stub = tmp_path / "stub"; (stub / "torch").mkdir(parents=True)
    (stub / "torch" / "__init__.py").write_text("from .version import __version__, cuda\n")
    (stub / "torch" / "version.py").write_text("__version__ = '2.13.0+cu130'\ncuda = '13.0'\n")     # torch/version.py's own layout: the key is read from it, torch never imported
    di = stub / "torch-2.13.0+cu130.dist-info"; di.mkdir()                                                   # and its distribution metadata, first on sys.path: the key reads the metadata
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.13.0+cu130\n")             # version before torch/version.py, so an interpreter that has another torch installed still resolves the stub
    b = _bin_with(tmp_path, {"python": f"#!/bin/bash\nexec {sys.executable} \"$@\"\n"}, nvidia_smi_cc="9.0")
    env = {k: v for k, v in os.environ.items() if k not in ("MODEL_OPT_STACK_KEY", "MODEL_OPT_JIT_ROOT", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "PROTENIX_OPT_CACHE_DIR", "MODEL_OPT", "PROTENIX_ROOT_DIR")}
    env["PATH"] = b + os.pathsep + env["PATH"]; env["PYTHONPATH"] = str(stub) + os.pathsep + env.get("PYTHONPATH", "")
    weights = tmp_path / "weights"; env["PROTENIX_ROOT_DIR"] = str(weights)
    cfg = TREE / "configs" / "h100.env"
    echo = "echo key=$MODEL_OPT_STACK_KEY; echo ext=${TORCH_EXTENSIONS_DIR-unset}; echo triton=${TRITON_CACHE_DIR-unset}; echo kit=${PROTENIX_OPT_CACHE_DIR-unset}; echo root=$PROTENIX_ROOT_DIR"
    r = subprocess.run(["bash", "-c", f"source {cfg} && {echo}"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    assert r.stdout.split() == ["key=torch2.13.0-cu130-sm90", "ext=unset", "triton=unset", "kit=unset", f"root={weights}"], "no MODEL_OPT_JIT_ROOT: the key is exported, no cache directory is"
    jit = tmp_path / "jit"; env["MODEL_OPT_JIT_ROOT"] = str(jit)
    r = subprocess.run(["bash", "-c", f"source {cfg} && {echo}"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    assert r.stdout.split() == ["key=torch2.13.0-cu130-sm90", f"ext={jit}/torch2.13.0-cu130-sm90/torch_extensions", f"triton={jit}/torch2.13.0-cu130-sm90/triton", f"kit={jit}/weights", f"root={weights}"]
    env["MODEL_OPT_STACK_KEY"] = "preset-key"
    r = subprocess.run(["bash", "-c", f"source {cfg} && echo $TORCH_EXTENSIONS_DIR"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == f"{jit}/preset-key/torch_extensions"
    # a pre-set cache directory is kept as given, whether or not it exists yet
    mounted = tmp_path / "cache"; mounted.mkdir()
    env.update(TRITON_CACHE_DIR=str(mounted / "triton"), TORCH_EXTENSIONS_DIR="/elsewhere/torch_ext", PROTENIX_OPT_CACHE_DIR=str(tmp_path / "kit_cache"))
    r = subprocess.run(["bash", "-c", f"source {cfg} && echo $TRITON_CACHE_DIR && echo $TORCH_EXTENSIONS_DIR && echo $PROTENIX_OPT_CACHE_DIR"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.split() == [str(mounted / "triton"), "/elsewhere/torch_ext", str(tmp_path / "kit_cache")]
    # the weights root has no default: unset, sourcing returns 3 with one NOT ACTIVE line naming the variable, and nothing after it runs
    r = subprocess.run(["bash", "-c", f"source {cfg} && echo SOURCED"], env={k: v for k, v in env.items() if k != "PROTENIX_ROOT_DIR"}, capture_output=True, text=True)
    assert r.returncode == 3 and "SOURCED" not in r.stdout and "[protenix-opt] NOT ACTIVE: PROTENIX_ROOT_DIR is not set" in r.stderr, (r.returncode, r.stdout, r.stderr[-600:])


def _stub_torch_env(tmp_path):
    """A clean environment in which configs/*.env source fully on a CPU box: a stub torch 2.13.0+cu130 (metadata + version.py, never imported
    by the key probe), this interpreter as `python`, a fake nvidia-smi reporting cc 9.0."""
    stub = tmp_path / "stub"; (stub / "torch").mkdir(parents=True)
    (stub / "torch" / "__init__.py").write_text("from .version import __version__, cuda\n")
    (stub / "torch" / "version.py").write_text("__version__ = '2.13.0+cu130'\ncuda = '13.0'\n")
    di = stub / "torch-2.13.0+cu130.dist-info"; di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.13.0+cu130\n")
    b = _bin_with(tmp_path, {"python": f"#!/bin/bash\nexec {sys.executable} \"$@\"\n"}, nvidia_smi_cc="9.0")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PROTENIX_OPT", "MODEL_OPT", "PTX_", "FPF_")) and k not in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR")}
    env["PATH"] = b + os.pathsep + env["PATH"]; env["PYTHONPATH"] = str(stub) + os.pathsep + env.get("PYTHONPATH", "")
    env["PROTENIX_ROOT_DIR"] = str(tmp_path / "weights")            # the weights root configs/h100.env requires by name (nothing here reads weights)
    return env


def test_every_config_exports_only_declared_kit_names(tmp_path):
    """Every configs/*.env, sourced in a clean shell, leaves an environment the start-up gate accepts: every PROTENIX_OPT* name it exports is
    one the kit reads (_autoload.DECLARED) — `refuse_undeclared` passes (rc 0, no 'undeclared' word). A config exporting an undeclared
    PROTENIX_OPT* name would stop `run.sh check` and every python started after sourcing it with exit 3."""
    env = _stub_torch_env(tmp_path)
    cfgs = sorted((TREE / "configs").glob("*.env")); assert cfgs
    for cfg in cfgs:
        probe = "import os, protenix_opt._autoload as a; a.refuse_undeclared(os.environ); print('DECLARED-OK', ','.join(sorted(k for k in os.environ if k.startswith('PROTENIX_OPT'))))"
        r = subprocess.run(["bash", "-c", f'set -a; source "{cfg}" && python -c "{probe}"'], env=env, capture_output=True, text=True, cwd=str(TREE))
        assert r.returncode == 0 and "undeclared" not in (r.stdout + r.stderr) and "DECLARED-OK" in r.stdout, (cfg.name, r.returncode, (r.stdout + r.stderr)[-800:])
        exported = set(re.findall(r"(?m)^\s*export\s+(PROTENIX_OPT\w*)=", cfg.read_text()))
        assert exported <= set(_autoload.DECLARED), (cfg.name, sorted(exported - set(_autoload.DECLARED)))


def test_run_sh_check_after_the_config_never_meets_the_undeclared_gate(tmp_path):
    """`bash run.sh check --config h100 --mode <m>` on a CPU box: the config sources, the dry run resolves; the outcome is rc 0 or a NAMED
    refusal of the box (no supported GPU / kernel levers unavailable) — never the undeclared-variable gate."""
    env = _stub_torch_env(tmp_path)
    for mode in ("exact", "fast", "big"):
        r = subprocess.run(["bash", str(TREE / "run.sh"), "check", "--config", "h100", "--mode", mode], env=env, capture_output=True, text=True, cwd=str(tmp_path))
        out = r.stdout + r.stderr
        assert "undeclared" not in out, (mode, out[-800:])
        assert r.returncode in (0, 3), (mode, r.returncode, out[-800:])
        assert "[protenix-opt] ACTIVE" in out or "[protenix-opt] NOT ACTIVE" in out, (mode, out[-800:])


def test_run_sh_passes_the_interpreters_own_refusal_through(tmp_path):
    """run.sh's presence probe never masks a start-up refusal as 'not installed': with an undeclared PROTENIX_OPT_BOGUS in the environment the
    probe's python exits 3 at the kit's gate and run.sh prints that sentence and exits 3; with the package genuinely absent from the interpreter
    the word is 'not installed', rc 2."""
    env = _stub_torch_env(tmp_path)
    r = subprocess.run(["bash", str(TREE / "run.sh"), "check", "--mode", "exact"], env={**env, "PROTENIX_OPT_BOGUS": "1"}, capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and "undeclared" in r.stderr and "PROTENIX_OPT_BOGUS" in r.stderr and "not installed" not in (r.stdout + r.stderr), (r.returncode, (r.stdout + r.stderr)[-800:])
    r = subprocess.run(["bash", str(TREE / "run.sh"), "check", "--config", "h100", "--mode", "exact"], env={**env, "PROTENIX_OPT_BOGUS": "1"}, capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and "undeclared" in r.stderr and "PROTENIX_OPT_BOGUS" in r.stderr and "not installed" not in (r.stdout + r.stderr), (r.returncode, (r.stdout + r.stderr)[-800:])
    bare = _bin_with(tmp_path / "bare", {"python": f"#!/bin/bash\nexec {sys.executable} -S \"$@\"\n"})          # -S: no site directories -> the installed package is genuinely absent
    env2 = dict(env); env2["PATH"] = bare + os.pathsep + os.environ["PATH"]; env2.pop("PYTHONPATH", None)
    r = subprocess.run(["bash", str(TREE / "run.sh"), "check", "--mode", "exact"], env=env2, capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 2 and "protenix_opt is not installed" in r.stderr, (r.returncode, (r.stdout + r.stderr)[-800:])


# ---------------------------------------------------------------------------------------------------------------- warm
def test_warm_is_one_fixed_small_pred_on_the_shipped_input(monkeypatch, tmp_path):
    """warm delegates to pred: the port options given come first, then the shipped example input, the out_dir and the fixed small stock
    arguments (local constants, not knobs); --out_dir defaults to a fresh temporary directory; any stock knob is a usage error."""
    import json
    seen = []
    monkeypatch.setattr(cli, "cmd_pred", lambda argv: seen.append(list(argv)) or 0)
    monkeypatch.setitem(cli.COMMANDS, "warm", cli.cmd_warm)
    assert cli.main(["warm", "--mode", "fast", "--det", "1", "--out_dir", str(tmp_path / "w")]) == 0
    assert seen[-1] == ["--mode", "fast", "--det", "1", "--input", cli.WARM_INPUT, "--out_dir", str(tmp_path / "w"), *cli.WARM_STOCK_ARGS]
    assert cli.main(["warm", "--mode=off"]) == 0
    assert seen[-1][:2] == ["--mode", "off"] and seen[-1][2:4] == ["--input", cli.WARM_INPUT] and seen[-1][4] == "--out_dir"
    assert os.path.isdir(seen[-1][5]) and os.path.basename(seen[-1][5]).startswith("protenix_opt_warm_") and tuple(seen[-1][6:]) == cli.WARM_STOCK_ARGS
    assert cli.WARM_STOCK_ARGS == ("--seeds", "101", "--cycle", "1", "--step", "2", "--sample", "1", "--use_msa", "false")
    doc = json.load(open(cli.WARM_INPUT, encoding="utf-8"))
    assert isinstance(doc, list) and len(doc) == 1 and doc[0]["name"] and len(doc[0]["sequences"]) == 1
    assert len(doc[0]["sequences"][0]["proteinChain"]["sequence"]) == 76 and doc[0]["sequences"][0]["proteinChain"]["count"] == 1
    for bad in (["warm", "--cycle", "10"], ["warm", "--input", "x.json"], ["warm", "extra"], ["warm", "--mode"], ["warm", "--allow-partial"]):
        assert cli.main(bad) == cli.EXIT_USAGE, bad
    assert "warm" in cli.COMMANDS and "\n  warm    [--mode exact|fast|big|off] [--det 0|1] [--out_dir DIR]" in cli.USAGE


def test_warm_reaches_the_stock_cli_through_pred(core, fake_stock, tmp_path):
    """End to end through the real cmd_pred with the stock stand-in: the stock CLI receives exactly the shipped input, the out_dir and the
    fixed small arguments, parsed by the stock-shaped parser."""
    out = tmp_path / "warm"
    assert cli.main(["warm", "--mode", "fast", "--out_dir", str(out)]) == 0
    assert fake_stock["main_args"][-1] == ["pred", "--input", cli.WARM_INPUT, "--out_dir", str(out), *cli.WARM_STOCK_ARGS]
    p = fake_stock["pred"][-1]
    assert p["input"] == cli.WARM_INPUT and p["cycle"] == 1 and p["step"] == 2 and p["sample"] == 1 and p["seeds"] == "101"
