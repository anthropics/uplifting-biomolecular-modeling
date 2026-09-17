"""The activation rules: the env route refuses by name at the first import of `rfdiffusion` (one NOT ACTIVE line, exit 3, nothing of upstream's
runs under the mode's name; inert for off / unset; an undeclared word exits at interpreter start) and never reaches a child process of
the package; enable() classes the card by compute capability and memory, notes a card above the stack's capability ceiling and proceeds
(never a refusal), and exports the base driver's documented override off-H100; the stock pins are reported, never a gate; the
late-activation rule (a model instance, a kit lever module already applied); idempotency; the driver environment keeps the mode table the
only source of a switch."""
import os
import subprocess
import sys
import types

import pytest

from rfdiffusion1_opt import _autoload, modes, report, stack
from rfdiffusion1_opt.tests._stubs import H100, H100NVL, A100, B200, good_box, pkg_pythonpath


@pytest.fixture(autouse=True)
def _reset():
    stack.reset_for_tests()
    yield
    stack.reset_for_tests()


# ------------------------------------------------------------------------------------------------------------------ env route
def _drop_finder(f):
    if f is not None and f in sys.meta_path:
        sys.meta_path.remove(f)


def test_env_route_inert_for_off_and_unset():
    from opt_core.autoload import Finder
    assert _autoload.install({}) is None
    assert _autoload.install({"RFDIFFUSION1_OPT": "off"}) is None
    assert _autoload.install({"RFDIFFUSION1_OPT": "OFF"}) is None
    assert all(not (isinstance(f, Finder) and f.spec.package == "rfdiffusion1_opt") for f in sys.meta_path)


def test_autoload_spec_is_this_kits_names():
    s = _autoload.spec()
    assert (s.env, s.package, s.tag, s.triggers, s.modes, s.exit_not_active) == ("RFDIFFUSION1_OPT", "rfdiffusion1_opt", "rfdiffusion1-opt", ("rfdiffusion",), ("exact", "fast", "off"), 3)
    assert s.modes == _autoload.MODES and set(s.modes) == set(modes.MODE_NAMES)      # the hook declares every mode word of the table (an undeclared word exits at interpreter start)
    assert s.on_unknown == "exit_now"                                               # an undeclared mode refuses at interpreter start (stock never runs silently under the variable)
    assert _autoload.TRIGGERS == ("rfdiffusion",)


def test_env_route_refuses_by_name_at_the_trigger_and_exits_3(capsys):
    """The core's finder fires `enable(mode, strict=True, trigger=...)`; the package prints its ONE sentence and raises ActivationError, on which
    the finder ends the process with exit 3 — upstream's command line never runs under the mode's name (the kit line is `design --mode <m>`)."""
    from opt_core.autoload import Finder
    f = _autoload.install({"RFDIFFUSION1_OPT": "exact"})
    try:
        assert isinstance(f, Finder) and f in sys.meta_path and f.selection["mode"] == "exact"
        assert f.find_spec("numpy") is None and f.find_spec("torch") is None          # only the trigger is watched
        with pytest.raises(SystemExit) as ex:
            f._fire("rfdiffusion")                                                     # the trigger's import: refused by name, the process ends here
        assert ex.value.code == 3
        err = capsys.readouterr().err
        assert err.strip() == "[rfdiffusion1-opt] NOT ACTIVE: " + _autoload.FACT.format(mode="exact", trigger="rfdiffusion", env="RFDIFFUSION1_OPT")
        assert "mode=exact cannot serve upstream's own command line" in err and "refused by name (exit 3), nothing ran" in err
        assert "the kit line is `rfdiffusion1-opt design --mode exact <the same overrides>`" in err and "drivers/rfd_bench.py" in err
        assert "the stock line is RFDIFFUSION1_OPT=off" in err and "(this process imported rfdiffusion under RFDIFFUSION1_OPT=exact)" in err
        assert err.count("NOT ACTIVE") == 1                                            # the kit's line once; the core adds none for the package's own ActivationError
        assert f not in sys.meta_path                                                  # fired once, removed
        assert stack.status().get("active") is False and stack.status().get("reason")  # nothing was armed in this process
    finally:
        _drop_finder(f)


def test_refuse_at_trigger_is_the_autoload_sentence_and_raises_when_strict(capsys):
    """stack.activate(..., trigger=...) = refuse_at_trigger: the NOT ACTIVE line with _autoload.FACT; strict (the .pth route's call) raises
    ActivationError after printing it, strict=False (a library caller) returns the inactive report."""
    from rfdiffusion1_opt import ActivationError
    with pytest.raises(ActivationError) as ex:
        stack.activate("exact", trigger="rfdiffusion", strict=True)
    err = capsys.readouterr().err.strip()
    assert err == "[rfdiffusion1-opt] NOT ACTIVE: " + _autoload.FACT.format(mode="exact", trigger="rfdiffusion", env="RFDIFFUSION1_OPT") == "[rfdiffusion1-opt] NOT ACTIVE: " + str(ex.value)
    assert stack.status()["active"] is False and "enable() has not run" in stack.status()["reason"]   # the trigger route arms and records nothing
    rep = stack.activate("fast", trigger="rfdiffusion", strict=False)
    assert rep["active"] is False and rep["attach"] == "stock-cli" and rep["trigger"] == "rfdiffusion" and rep["mode"] == "fast" and rep["reason"]
    assert rep == stack.refuse_at_trigger("fast", "rfdiffusion", strict=False)         # one producer
    with pytest.raises(ActivationError):
        stack.refuse_at_trigger("off", "rfdiffusion", strict=True)                     # every declared word takes the same route at the trigger: named, then the raise
    assert capsys.readouterr().err.count("NOT ACTIVE") == 3


def test_env_route_ends_upstreams_command_line_with_exit_3_through_a_real_pth(tmp_path):
    """RFDIFFUSION1_OPT=<mode> python <anything importing rfdiffusion>: the .pth's finder fires at that import, the kit's NOT ACTIVE line is the
    whole of stderr, the process exits 3 and not one statement past the import runs — through a real .pth and a stand-in `rfdiffusion`."""
    site_dir = _site_dir(tmp_path)
    fake = tmp_path / "upstream" / "rfdiffusion"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("BODY_RAN = True\n")
    child = [sys.executable, "-S", "-c", f"import site, sys; site.addsitedir({str(site_dir)!r}); sys.path.insert(0, {str(fake.parent)!r}); import rfdiffusion; print('upstream ran on past the import')"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RFD_", "RFDIFFUSION1", "PYTHON"))}
    for mode in ("exact", "fast"):
        p = subprocess.run(child, env=dict(env, RFDIFFUSION1_OPT=mode), capture_output=True, text=True)
        assert p.returncode == 3 and "upstream ran" not in p.stdout, (mode, p.returncode, p.stdout, p.stderr)
        assert p.stderr.strip() == "[rfdiffusion1-opt] NOT ACTIVE: " + _autoload.FACT.format(mode=mode, trigger="rfdiffusion", env="RFDIFFUSION1_OPT"), (mode, p.stderr)
    p = subprocess.run(child, env=dict(env, RFDIFFUSION1_OPT="off"), capture_output=True, text=True)      # off: no finder, upstream's command line untouched
    assert p.returncode == 0 and "upstream ran on past the import" in p.stdout and p.stderr == "", p.stderr


def test_env_route_unknown_mode_refuses_at_interpreter_start(tmp_path):
    """A value that is not a declared mode: refused at .pth time (the core's line, os._exit 3) with the core importable — through a real .pth."""
    site_dir = _site_dir(tmp_path)
    child = [sys.executable, "-S", "-c", f"import site; site.addsitedir({str(site_dir)!r}); print('fell through to stock')"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RFD_", "RFDIFFUSION1", "PYTHON"))}
    p = subprocess.run(child, env=dict(env, RFDIFFUSION1_OPT="turbo"), capture_output=True, text=True)
    assert p.returncode == 3 and "fell through" not in p.stdout, (p.returncode, p.stdout, p.stderr)
    assert p.stderr.strip() == "[rfdiffusion1-opt] NOT ACTIVE: unknown RFDIFFUSION1_OPT='turbo' (expected exact|fast|off)"
    p = subprocess.run(child, env=dict(env, RFDIFFUSION1_OPT="exact"), capture_output=True, text=True)   # a declared mode word arms the finder and waits for the trigger (nothing imported rfdiffusion here)
    assert p.returncode == 0 and "fell through to stock" in p.stdout and p.stderr == "", p.stderr


def _site_dir(tmp_path):
    """A site directory carrying rfdiffusion1_opt_autoload.pth, the package with its pyproject.toml one above it as in the tree, and the
    core — the real .pth route of an install. The absent / stale core cases of this route live in test_core_gate_routes."""
    import shutil
    import opt_core
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    opt = os.path.join(stack.tree_root(), "opt")
    shutil.copy(os.path.join(opt, "rfdiffusion1_opt_autoload.pth"), site_dir)
    os.symlink(os.path.join(opt, "rfdiffusion1_opt"), site_dir / "rfdiffusion1_opt")
    os.symlink(os.path.join(opt, "pyproject.toml"), site_dir / "pyproject.toml")           # the gate reads the pin from the pyproject above the package
    core_pkg = os.path.dirname(os.path.abspath(opt_core.__file__))
    os.symlink(core_pkg, site_dir / "opt_core")
    return site_dir


def test_dry_run_on_a_good_h100_box(monkeypatch):
    good_box(monkeypatch)
    rep = stack.activate("exact", dry_run=True)
    assert rep["would_refuse"] == [] and rep["reason"] is None and rep["active"] is False and rep["dry_run"]
    assert rep["gpu"]["class"] == "H100" and rep["gpu"]["key"] == "9.0|3.0" and rep["driver_gate_override"] == {}
    assert rep["levers_planned"] == list(modes.resolve("exact").levers) and rep["env"] == {"RFD_PDBIO": "1"}   # the exact row's one environment switch: lever IO1
    assert rep["stack"]["pinned"] is True
    assert stack.status()["active"] is False                      # a dry run records nothing


def test_gate_off_h100_exports_the_drivers_override(monkeypatch):
    good_box(monkeypatch, gpu=A100)
    rep = stack.activate("exact", dry_run=True)
    assert rep["would_refuse"] == [] and rep["gpu"]["class"] == "A100"
    assert rep["driver_gate_override"] == {"ALLOW_ANY_GPU": "1"}
    env, _ = stack.driver_environment(modes.resolve("exact"), rep["gpu"])
    assert env["ALLOW_ANY_GPU"] == "1"
    env2, _ = stack.driver_environment(modes.resolve("exact"), H100)
    assert "ALLOW_ANY_GPU" not in env2


def test_gate_notes_above_the_stacks_capability_and_proceeds(monkeypatch, capsys):
    """A card above the pinned stack's kernel ceiling (B200, 10.0 > MAX_CC_OF_STACK) is untested hardware: ONE NOTE line, recorded in
    the report's notes, never a refusal — the dry run resolves, the real activation arms."""
    good_box(monkeypatch, gpu=B200)
    capsys.readouterr()
    rep = stack.activate("exact", dry_run=True)
    err = capsys.readouterr().err
    note = "compute capability 10.0 above the pinned stack's ceiling (torch 2.4.0 carries kernels up to sm_90) — kernels untested; proceeding"
    assert rep["would_refuse"] == [] and rep["reason"] is None and rep["notes"] == [note] and rep["gpu"]["class"] == "B200"
    assert err.count("[rfdiffusion1-opt] NOTE: ") == 1 and f"[rfdiffusion1-opt] NOTE: {note}" in err and "NOT ACTIVE" not in err
    stack.reset_for_tests()
    rep = stack.activate("exact")
    assert rep["active"] and rep["notes"] == [note]
    stack.reset_for_tests()
    good_box(monkeypatch)                                                              # the reference card: no note
    assert stack.activate("exact", dry_run=True)["notes"] == []


def test_gate_refuses_without_gpu(monkeypatch):
    good_box(monkeypatch, gpu={"name": None, "mem_gib": None, "cc": None, "sm": None, "source": None})
    rep = stack.activate("exact", dry_run=True)
    assert any("no CUDA device" in w for w in rep["would_refuse"])


def test_gate_needs_no_build_tools(monkeypatch):
    """No lever of these kits builds an extension: a box without nvcc / ninja passes the gate, and the report lists no such tool."""
    good_box(monkeypatch, tools={"python": sys.executable, "mps_control": None})
    rep = stack.activate("exact", dry_run=True)
    assert rep["would_refuse"] == [] and set(rep["tools"]) == {"python", "mps_control"}


def test_pins_are_reported_never_a_gate(monkeypatch):
    """A checkout or stack other than stock/PINS.json's is NAMED by `check` (stack.pins_report: the PINS lines), never a gate of activation:
    the kit line and the stock command line both resolve on a patched checkout (no `pins` key, no forcing switch on the report)."""
    finding = "rfdiffusion: 1 file(s) differ from the pinned commit"
    good_box(monkeypatch, pins_bad=[finding])
    for mode in ("exact", "off", "fast"):
        rep = stack.activate(mode, dry_run=True)
        assert rep["would_refuse"] == [] and rep["reason"] is None and "pins" not in rep and "forced" not in rep, (mode, rep)
    pr = stack.pins_report()
    assert pr["pinned"] is False and pr["findings"] == [finding] and set(pr) == {"pinned", "findings", "detail", "lines"}
    assert pr["lines"] == ["[rfdiffusion1-opt] PINS upstream=https://github.com/RosettaCommons/RFdiffusion@86507b65 stack=" + stack.pins()["pinned_stack"]["id"] + " pinned=False",
                           f"[rfdiffusion1-opt] PINS finding: {finding}"]
    good_box(monkeypatch)                                                            # the pinned checkout and stack: one head line, no finding
    pr = stack.pins_report()
    assert pr["pinned"] is True and pr["findings"] == [] and len(pr["lines"]) == 1 and pr["lines"][0].endswith(" pinned=True")
    assert pr["detail"]["checkout"]["files_checked"] == 75

    def unreadable(stack=True, root=None):
        raise OSError("no PINS.json")
    monkeypatch.setattr(stack, "check_pins", unreadable)                             # an unreadable pin table is a finding too, never a traceback
    pr = stack.pins_report()
    assert pr["pinned"] is False and pr["findings"][0].startswith("stock pins unreadable: OSError(") and pr["lines"][1].startswith("[rfdiffusion1-opt] PINS finding: stock pins unreadable")


def test_unknown_modes_are_not_gated_but_named(monkeypatch, capsys):
    good_box(monkeypatch)
    rep = stack.activate("fast", dry_run=True)                                                    # the fast row: K + T2, the add-on's variable in the row
    assert rep["mode"] == "fast" and rep["would_refuse"] == [] and not rep.get("mode_defaulted"), rep
    assert "[rfdiffusion1-opt] DRY-RUN mode=fast attach=driver tier=2 " in capsys.readouterr().err   # the line's head: the mode, then the attach point
    rep = stack.activate(None, dry_run=True)
    assert rep["mode"] == "fast" and rep["mode_defaulted"] and rep["would_refuse"] == []       # no mode: the default — fast, on every route
    good_box(monkeypatch, tools={"python": sys.executable, "mps_control": "/usr/bin/nvidia-cuda-mps-control"})   # the packed line gates on the MPS control binary too
    rep = stack.activate(None, dry_run=True, served=True)
    assert rep["mode"] == "fast" and rep["mode_defaulted"] and "jit_warmup" not in rep and rep["would_refuse"] == []   # the packed line defaults to fast too (it packs exact and fast)
    assert "[rfdiffusion1-opt] PLAN mode=fast route=served attach=" in report.activation_line(rep, "PLAN")   # the activation lines name the route (the head of every form: PLAN / ACTIVE / NOT ACTIVE)
    stack.reset_for_tests()
    rep = stack.activate("exact", dry_run=True, served=True)
    assert rep["mode"] == "exact" and not rep.get("mode_defaulted") and "[rfdiffusion1-opt] PLAN mode=exact route=served attach=" in report.activation_line(rep, "PLAN")
    assert "route=" not in report.activation_line(stack.activate("fast", dry_run=True), "PLAN")   # the design route's line carries no route= token
    rep = stack.activate("exact_w1", dry_run=True)                                             # a lever-set name is not a mode: unknown by name
    assert "unknown mode 'exact_w1' (expected off|exact|fast)" in rep["reason"]
    rep = stack.activate("turbo", dry_run=True)
    assert rep["reason"] == "unknown mode 'turbo' (expected off|exact|fast)"
    assert capsys.readouterr().err.count("[rfdiffusion1-opt] NOT ACTIVE: unknown mode 'turbo' (expected off|exact|fast) (mode=turbo)\n") == 1   # the refusal's tail names the mode asked for


def test_mode_from_environment(monkeypatch):
    good_box(monkeypatch)
    monkeypatch.setenv("RFDIFFUSION1_OPT", "off")
    rep = stack.activate(None, dry_run=True)
    assert rep["mode"] == "off" and rep["attach"] == "stock-cli" and rep["would_refuse"] == []


# ------------------------------------------------------------------------------------------------------------ late activation
def test_idempotent_and_refuses_a_second_mode(monkeypatch):
    good_box(monkeypatch)
    a = stack.activate("off")
    assert a["active"] and a["attach"] == "stock-cli"
    assert stack.activate("off") is a and stack.status() is a
    b = stack.activate("exact")
    assert not b["active"] and "already activated" in b["reason"]


def test_refused_once_a_model_instance_exists(monkeypatch):
    good_box(monkeypatch)
    mod = types.ModuleType("rfdiffusion.RoseTTAFoldModel")

    class RoseTTAFoldModule:                                # the upstream model class, by name
        pass
    mod.RoseTTAFoldModule = RoseTTAFoldModule
    monkeypatch.setitem(sys.modules, "rfdiffusion.RoseTTAFoldModel", mod)
    inst = RoseTTAFoldModule()
    rep = stack.activate("exact")
    assert not rep["active"] and "RoseTTAFoldModule instance(s) already exist" in rep["reason"]
    del inst


def test_refused_once_a_kit_lever_reports_itself_applied(monkeypatch):
    good_box(monkeypatch)
    fg = types.ModuleType("rfd_fullgraph")
    fg.stats = lambda: {"n_capture": 2, "n_replay": 10}
    monkeypatch.setitem(sys.modules, "rfd_fullgraph", fg)
    rep = stack.activate("exact")
    assert not rep["active"] and "rfd_fullgraph" in rep["reason"] and "rfd_fullgraph" in stack.KIT_LEVER_MODULES


def test_activation_arms_and_reports(monkeypatch):
    good_box(monkeypatch)
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    rep = stack.activate("exact")
    assert rep["active"] and rep["attach"] == "driver" and rep["tier"] == 1 and rep["env_dropped"] == []
    assert stack.status() is rep
    # the contract's keys are present once armed and filled from the evidence read-back of each driver pass
    assert rep["levers_applied"] == [] and rep["levers_fallback"] == [] and rep["levers_unavailable"] == [] and rep["partial"] is False
    assert stack.record_pass({"applied": ["C1", "P"], "missing": ["C3"], "forbidden": []}) is rep
    assert rep["levers_applied"] == ["C1", "P"] and rep["levers_unavailable"] == ["C3"] and rep["partial"] is True and rep["levers_fallback"] == []
    stack.record_pass({"applied": ["C3", "E_einsum"], "missing": [], "forbidden": []})
    assert rep["levers_applied"] == ["C1", "C3", "E_einsum", "P"] and rep["partial"] is True          # a partial pass stays recorded


def test_record_pass_before_enable_is_none():
    assert stack.record_pass({"applied": ["C1"], "missing": [], "forbidden": []}) is None


def test_strict_raises(monkeypatch):
    good_box(monkeypatch)
    from rfdiffusion1_opt import ActivationError
    with pytest.raises(ActivationError, match="unknown mode"):
        stack.activate("turbo", strict=True)                                                      # an unknown name: refused by name


# ------------------------------------------------------------------------------------------------------- driver environment
def test_driver_environment_keeps_the_table_the_source_of_truth():
    res = modes.resolve("exact")
    caller = {"PATH": "/usr/bin", "RFD_TRITON_LN": "1", "RFD_PREP": "0", "RFD_ROOT": "/opt/rfd", "RFD_ELSEWHERE": "/w", "RFDIFFUSION1_OPT": "exact",
              "RFD_FASTPATH_SEQSEP": "0", "MKL_CBWR": "AVX512", "OPENBLAS_CORETYPE": "Zen"}
    caller.update({"NVIDIA_TF32_OVERRIDE": "0", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"})
    env, dropped = stack.driver_environment(res, H100, environ=caller)
    assert dropped == ["NVIDIA_TF32_OVERRIDE", "RFDIFFUSION1_OPT", "RFD_ELSEWHERE", "RFD_FASTPATH_SEQSEP", "RFD_PREP", "RFD_TRITON_LN", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]   # the kits' switches and any other RFD_* name by prefix, the package's own, the TF32 overrides by name
    assert env["DGLBACKEND"] == "pytorch" and env["RFD_ROOT"] == "/opt/rfd" and "RFD_ELSEWHERE" not in env
    assert "RFD_PREP" not in env and env["MKL_CBWR"] == "AVX512" and env["OPENBLAS_CORETYPE"] == "Zen"   # the caller's BLAS words are the caller's: no recipe of this package sets or strips them
    assert stack.KEEP_ENV == ("RFD_ROOT",)                                                          # the only RFD_* name that reaches the driver is the checkout's location (the weights directory is WEIGHTS)
    assert stack.ENV_MODE not in env                                   # the package's own switch never reaches a process that imports rfdiffusion
    assert stack.DROP_ENV_NAMES == ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")           # the two TF32 overrides, nothing else by name
    import inspect
    assert list(inspect.signature(stack.driver_environment).parameters) == ["res", "gpu", "environ"]      # the row and the card are the only inputs: no recipe switch on the environment builder


def test_child_environment_drops_the_packages_switches_and_the_tf32_override():
    caller = {"PATH": "/usr/bin", "RFDIFFUSION1_OPT": "exact", "RFDIFFUSION1_OPT_HOME": "/tree", "RFDIFFUSION1_ANYTHING": "x",
              "NVIDIA_TF32_OVERRIDE": "1", "HOME": "/home/x", "TRITON_CACHE_DIR": "/jit", "RFD_ROOT": "/opt/rfd", "WEIGHTS": "/w"}
    env, dropped = stack.child_environment(caller)
    assert dropped == ["NVIDIA_TF32_OVERRIDE", "RFDIFFUSION1_ANYTHING", "RFDIFFUSION1_OPT", "RFDIFFUSION1_OPT_HOME"]
    assert env == {"PATH": "/usr/bin", "HOME": "/home/x", "TRITON_CACHE_DIR": "/jit", "RFD_ROOT": "/opt/rfd", "WEIGHTS": "/w"}   # the two data paths are kept (RFD_ROOT by KEEP_ENV; WEIGHTS is not a kit prefix)
    assert "NVIDIA_TF32_OVERRIDE" in stack.DROP_ENV_NAMES and stack.ENV_MODE.startswith(stack.DROP_ENV_PREFIXES)
    for name in (stack.ENV_MODE, stack.ENV_HOME):
        assert name.startswith(stack.DROP_ENV_PREFIXES), name
    assert not hasattr(stack, "ENV_FORCE")                                          # the pins are a report: no such switch exists


def test_driver_child_imports_rfdiffusion_under_the_env_route(tmp_path):
    """The .pth hook (`import rfdiffusion1_opt._autoload`) armed in a child: with RFDIFFUSION1_OPT=exact in the child's environment the
    first `import rfdiffusion` prints the ONE NOT ACTIVE line and the child exits 3 there — nothing past the import runs; under the driver
    environment the package builds from a caller that has the variable set, the same child imports it silently (the variable is dropped)."""
    (tmp_path / "rfdiffusion").mkdir()
    (tmp_path / "rfdiffusion" / "__init__.py").write_text("VERSION = 'stub'\n")
    child = [sys.executable, "-c", "import rfdiffusion1_opt._autoload; import rfdiffusion; print('imported', rfdiffusion.VERSION)"]
    base = {k: v for k, v in os.environ.items() if not k.startswith(("RFD_", "RFDIFFUSION1"))}
    base["PYTHONPATH"] = pkg_pythonpath(tmp_path)                                     # the child sees the package AND the core it stands on (an install carries both)
    p = subprocess.run(child, env=dict(base, RFDIFFUSION1_OPT="exact"), capture_output=True, text=True)
    assert p.returncode == 3 and "imported" not in p.stdout, (p.returncode, p.stdout, p.stderr)
    assert "[rfdiffusion1-opt] NOT ACTIVE: mode=exact cannot serve upstream's own command line" in p.stderr and p.stderr.count("NOT ACTIVE") == 1, p.stderr
    assert "the stock line is RFDIFFUSION1_OPT=off" in p.stderr and "Traceback" not in p.stderr
    env, dropped = stack.driver_environment(modes.resolve("exact"), H100, environ=dict(base, RFDIFFUSION1_OPT="exact"))
    assert "RFDIFFUSION1_OPT" in dropped
    p = subprocess.run(child, env=env, capture_output=True, text=True)
    assert p.returncode == 0 and "imported stub" in p.stdout and "NOT ACTIVE" not in p.stderr, p.stderr


def test_driver_command_shape(monkeypatch):
    good_box(monkeypatch)
    res = modes.resolve("exact")
    cmd = stack.driver_command(res, "/o/cases.json", "/o", "run1", "/opt/rfd", "/w", python="python")
    kit = stack.kit_dir("fast_inference")
    driver = os.path.join(kit, "drivers", "rfd_bench.py")
    # the kit line runs through the package's driver_run (upstream's defaults composed in place of the drivers' three constants, DRIVER_FIXED), then the driver
    assert cmd[:10] == ["python", "-m", "rfdiffusion1_opt.driver_run", "--fixed", "inference.write_trajectory=True", "--fixed", "inference.cautious=True",
                        "--fixed", "inference.deterministic=False", "--cases"] and cmd[10] == "/o/cases.json"
    assert [cmd[i + 1] for i, t in enumerate(cmd) if t == "--fixed"] == list(modes.settings_of(attach="driver").compose_overrides) == [f"{k}={modes.UPSTREAM_DEFAULTS[k]}" for k in modes.DRIVER_FIXED]
    assert cmd[11] == driver and "--compose" not in cmd                        # nothing typed beyond the fixed keys: nothing carried
    assert " ".join(cmd[12:]) == ("--rfd-root /opt/rfd --weights /w " + modes.K_LINE
                                  + " --no-traj 0 --cases /o/cases.json --out /o --tag run1")
    import inspect
    assert "compose_log" not in inspect.signature(stack.driver_command).parameters and "--log" not in cmd   # no composition log on the line
    # the deterministic recipe (--det 1): upstream's per-design seed composed on the driver line as the third fixed key
    cmd_d = stack.driver_command(modes.resolve("exact", det=True), "/o/cases.json", "/o", "run1", "/opt/rfd", "/w", python="python")
    assert cmd_d[7:9] == ["--fixed", "inference.deterministic=True"] and cmd_d[11] == driver
    # a typed key outside DRIVER_FIXED and TARGET_KEYS passes to the driver's configuration verbatim (--compose); the target keys are the case row's, never composed twice
    typed = ["inference.input_pdb=/in/t.pdb", "contigmap.contigs=[A1-10/0 5-5]", "inference.final_step=5", "inference.cautious=False", "denoiser.noise_scale_ca=0.5"]
    cmd_t = stack.driver_command(modes.resolve("exact", overrides=typed), "/o/cases.json", "/o", "run1", "/opt/rfd", "/w", python="python")
    assert cmd_t[3:9] == ["--fixed", "inference.write_trajectory=True", "--fixed", "inference.cautious=False", "--fixed", "inference.deterministic=False"]
    assert cmd_t[11:15] == ["--compose", "inference.final_step=5", "--compose", "denoiser.noise_scale_ca=0.5"] and cmd_t[15] == driver
    assert not any(t.startswith(("inference.input_pdb=", "contigmap.contigs=")) for t in cmd_t)
    cmd_c = stack.driver_command(res, "/o/cases.json", "/o", "run1", "/opt/rfd", "/w", python="python", compose=["inference.final_step=2"])   # `compose`: extra composed overrides, after the typed ones
    assert cmd_c[11:13] == ["--compose", "inference.final_step=2"] and cmd_c[13] == driver
    cmd2 = stack.driver_command(modes.resolve("exact", ["inference.write_trajectory=False", "inference.cautious=False"]), "/o/c.json", "/o", "t", "/opt/rfd", "/w", python="python")
    assert cmd2[3:9] == ["--fixed", "inference.write_trajectory=False", "--fixed", "inference.cautious=False", "--fixed", "inference.deterministic=False"]
    assert cmd2[11] == driver and "--no-traj 1" in " ".join(cmd2)                                      # the two knobs typed off: composed, and the driver's trajectory switch follows


def test_gpu_class_by_capability_and_memory(monkeypatch):
    assert stack.gpu_class({"cc": "9.0", "mem_gib": 79.6}) == "H100"                 # 81,559 MiB: the 80 GB card, the kits' reference class
    assert stack.gpu_class({"cc": "9.0", "mem_gib": 93.6}) == "H100NVL"              # 95,830 MiB: a class of its own, so the key names the card that ran
    assert stack.gpu_class(H100NVL) == "H100NVL"
    assert stack.gpu_class({"cc": "9.0", "mem_gib": 140.0}) == "H200"
    assert stack.gpu_class({"cc": "8.9", "mem_gib": 45.0}) == "L40S"
    assert stack.gpu_class({"cc": "10.0", "mem_gib": 180.0}) == "B200"
    assert stack.gpu_class({"cc": "9.0", "mem_gib": 20.0}) is None
    monkeypatch.setattr(stack, "dist_version", lambda n: {"torch": "2.4.0+cu121"}.get(n))
    assert stack.jit_cache_key(H100).endswith("-sm90") and stack.jit_cache_key({"cc": None}).endswith("-smNA")


def test_h100_nvl_resolves_with_its_class_named(monkeypatch):
    good_box(monkeypatch, gpu=H100NVL)
    monkeypatch.setenv("MODEL_OPT_TARGET_GPU", "H100")
    rep = stack.activate("exact", dry_run=True)
    assert rep["would_refuse"] == [] and rep["gpu"]["class"] == "H100NVL" and "H100NVL" in rep["target_gpu_mismatch"]
    assert rep["driver_gate_override"] == {}                                      # the base driver's own name assertion is met by the NVL card


def test_jit_cache_key_cuda_tag(monkeypatch):
    vers = {"torch": "2.4.0", "nvidia-cuda-runtime-cu12": "12.1.105"}
    monkeypatch.setattr(stack, "dist_version", lambda n: vers.get(n))
    assert stack.jit_cache_key(H100) == "torch2.4.0-cu121-sm90"          # a PyPI torch wheel: the CUDA tag from the runtime it pins
    vers = {"torch": "2.4.0+cu121"}
    assert stack.jit_cache_key(H100) == "torch2.4.0-cu121-sm90"
    vers = {"torch": "2.4.0"}
    with pytest.raises(RuntimeError, match="names no CUDA toolkit"):                # a CPU-only torch: refused by name, never a placeholder toolkit in the key
        stack.jit_cache_key(H100)
    from opt_core import jit_cache                                                  # the grammar is the core's: the same parts give the same string
    vers = {"torch": "2.4.0+cu121"}
    assert stack.jit_cache_key(H100) == jit_cache.key(version="2.4.0", cuda="121", cc="9.0") == "torch2.4.0-cu121-sm90"


def test_jit_cache_key_refuses_without_a_torch_distribution(monkeypatch):
    """No installed torch distribution: refuses by raising, never a silent placeholder version in the cache key."""
    monkeypatch.setattr(stack, "dist_version", lambda n: None)
    with pytest.raises(RuntimeError, match="no installed torch distribution"):
        stack.jit_cache_key(H100)


def test_fast_gates_triton_on_check_and_design_alike(monkeypatch):
    """base/fast needs triton (lever T2): without it in the stack `check --mode fast` (dry run) would refuse and `design --mode fast` refuses,
    the same fact and the same exit code; exact is unaffected; with triton installed fast resolves."""
    good_box(monkeypatch, versions={"triton": None})
    rep = stack.activate("fast", dry_run=True)
    assert any(w.startswith("triton is not installed in this stack: the fast line's lever(s) T2,K2 require it") for w in rep["would_refuse"]), rep["would_refuse"]
    rep = stack.activate("exact", dry_run=True)
    assert rep["would_refuse"] == []
    from rfdiffusion1_opt import cli
    assert cli.main(["check", "--mode", "fast"]) == 3 and cli.main(["check", "--mode", "exact"]) == 0
    stack._STATE["report"] = None
    rep = stack.activate("fast")                                                      # the design route's activation: not active, the fact named
    assert not rep["active"] and "triton is not installed" in (rep.get("reason") or "")
    stack._STATE["report"] = None
    good_box(monkeypatch)                                                             # the pinned stack carries triton 3.0.0
    assert stack.activate("fast", dry_run=True)["would_refuse"] == []
