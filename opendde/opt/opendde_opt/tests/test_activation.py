"""Activation rules, in-process with stubs of upstream: the version gate, the late-activation refusals by name
(kit shim active, runner instance), `off` refused under an active shim, idempotence, and — on the kit's own bytes — the ACCEL shim
arming the served-levers hook and wrapping `get_default_runner` at its import; a pre-set must-be-absent switch is stripped before
the shim reads it; a missing kit directory is refused by name; the route plan (worker-only levers on the CLI route, the CLI-only
lever on the served route); and the kit's own refusal (ARM-T declining without raising) is a fallback in the report, never an
applied lever."""
import importlib
import os
import sys
import types

import pytest

from opendde_opt import big, modes, stack
from opendde_opt import report as _report
from opendde_opt.tests import _stubs

KIT_MODS = ("odde_served_levers", "odde_accel_v2", "odde_arm_t", "odde_addon", "fpf_engines", "odde_trimul_bind", "fpf_trimul",
            "fpf_trimul.engines", "fpf_trimul.trimul", "fpf_trimul.kernels", "opendde.model.triangular", "opendde.model.triangular.triangular",
            "odde_triattn_bind", "odde_trimul_bind")


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A fresh activation state, the stub upstream first on sys.path, no kit module loaded, no switch in the environment."""
    big._reset()                                                                       # a census patch installed on the previous stub model module is restored while that module is still loaded
    site = _stubs.make_site(str(tmp_path))
    monkeypatch.syspath_prepend(site)
    for m in list(sys.modules):
        if m.startswith(("runner", "opendde.", "sitecustomize_opendde_opt")) or m == "opendde" or m in KIT_MODS:
            monkeypatch.delitem(sys.modules, m, raising=False)
    importlib.invalidate_caches()
    for k in list(os.environ):
        if k.startswith(("ODDE_", "OPENDDE_OPT", "DIT_", "CUEQ_", "CUBLAS_", "PYTORCH_CUDA")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_OPT", _stubs.TREE)
    for m, attrs in (("odde_offload", {"install": lambda: None, "__version__": "stub", "STATS": {}}), ("odde_traj_hook", {"apply_from_env": lambda: None})):
        monkeypatch.setitem(sys.modules, m, types.SimpleNamespace(**attrs))          # the offload unit imports torch and binds the model: on a big line the hook shim's cascade reaches
    monkeypatch.delattr(sys, "_odde_shim_chain", raising=False)                        # its shim (levers/OFFLOAD/sitecustomize.py), whose install is a no-op against this stub; a fresh cascade
    monkeypatch.setattr(stack, "_REPORT", None)
    monkeypatch.setattr(stack, "_ACTIVATING", False)
    monkeypatch.setattr(stack, "_PREDICTED", False)
    stack._APPLIED.clear()
    saved_path = list(sys.path)
    saved_meta = list(sys.meta_path)
    saved_env = dict(os.environ)
    yield site
    sys.path[:] = saved_path
    sys.meta_path[:] = saved_meta
    os.environ.clear(); os.environ.update(saved_env)
    for m in list(sys.modules):
        if m in KIT_MODS or m.startswith("sitecustomize_opendde_opt"):
            sys.modules.pop(m, None)


def test_version_gate(fresh, tmp_path, monkeypatch):
    other = _stubs.make_site(str(tmp_path / "v101"), version="1.0.1")
    monkeypatch.syspath_prepend(other)
    importlib.invalidate_caches()
    rep = stack.activate("exact", dry_run=True)
    assert not rep["active"] and "version gate" in rep["reason"] and "1.0.1" in rep["reason"]


def test_dry_run_applies_nothing(fresh):
    env_before = dict(os.environ)
    path_before = list(sys.path)
    rep = stack.activate("exact", dry_run=True)
    assert rep["dry_run"] and rep["reason"].startswith("dry run") and not rep["active"]
    assert "dit_hoist" in rep["levers_planned"]                                             # route cli: the hook installs the DiT hoist after the runner exists
    assert "levers_unavailable" not in rep and rep["route"] == "cli"                       # one route: every lever of the line is planned in this process
    assert dict(os.environ) == env_before and list(sys.path) == path_before
    assert "odde_served_levers" not in sys.modules
    assert rep["exports"]["ODDE_SERVED_LEVERS"] == "1" and "ODDE_ARM_U" in rep["unset"]


def test_overrides_are_exported_after_the_line_and_recorded(fresh):
    rep = stack.activate("exact", overrides={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})   # the deterministic recipe (det.env)
    assert rep["active"] and os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert rep["exports"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and rep["applied"]["exports"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert rep["overrides"] == rep["applied"]["overrides"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    assert "CUBLAS_WORKSPACE_CONFIG" not in modes.LINES["S1"].exports                   # the line never carries the recipe's switch: the recipe exports it at run time


def test_late_activation_refused_when_shim_active(fresh, monkeypatch):
    m = types.ModuleType("odde_served_levers")
    m.STATE = {"version": "0.2.0", "active": True, "wrapped": True, "installs": [], "errors": {}}
    monkeypatch.setitem(sys.modules, "odde_served_levers", m)
    rep = stack.activate("exact")
    assert not rep["active"] and "late activation refused" in rep["reason"] and "odde_served_levers v0.2.0" in rep["reason"]
    off = stack.activate("off")
    assert "mode off refused" in off["reason"] and "odde_served_levers" in off["reason"]


def test_late_activation_refused_when_runner_instance_exists(fresh):
    import runner.inference as ri
    inst = ri.InferenceRunner({})
    rep = stack.activate("exact")
    assert not rep["active"] and "InferenceRunner instance" in rep["reason"]
    del inst


def test_model_module_already_imported_does_not_block_the_hook(fresh):
    import opendde.model.opendde  # noqa: F401 — the hook installs after get_default_runner returns, not at this import
    rep = stack.activate("exact", dry_run=True)
    assert rep["reason"].startswith("dry run")
    rep2 = stack.activate("fast", dry_run=True)
    assert rep2["reason"].startswith("dry run") and rep2["exports"]["ODDE_ARM_U"] == "1"


def test_off_is_stock_and_never_applies(fresh):
    rep = stack.activate("off")
    assert not rep["active"] and rep["reason"].startswith("mode off: stock")
    assert "ODDE_SERVED_LEVERS" not in os.environ


def test_exact_arms_the_kits_own_shim_and_wraps_get_default_runner(fresh):
    rep = stack.activate("exact")
    assert rep["active"], rep
    assert os.environ["ODDE_SERVED_LEVERS"] == "1" and os.environ["ODDE_ADDON_LEVERS"] == "dit_hoist,dit_align" and os.environ["ODDE_ARM_Z"] == "1"
    assert os.environ["CUEQ_TRITON_CACHE_DIR"] == os.path.join(_stubs.TREE, "opt", "forward", "fast_inference", "levers", "KIT", "cueq_cache_shipped")
    assert os.environ["ODDE_SERVED_LEVERS_STRICT"] == "1"
    assert "ODDE_ARM_U" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert os.environ["FPF_OPS"] == "trimul_out,trimul_in" and os.environ["ODDE_TRIMUL"] == "exact" and "FPF_TRIMUL_MODE" not in os.environ   # S1: the FPF adapter contract + the provider's exact word (no kit TriMul mode switch since 0.2.57)
    exp = modes.resolve("exact", _stubs.TREE).sys_path
    assert sys.path[:len(exp)] == exp                                               # the line's six directories, the XL unit last
    sl = sys.modules["odde_served_levers"]
    assert sl.__version__ == "0.2.0" and sl.__file__.startswith(os.path.join(_stubs.TREE, "opt", "forward", "fast_inference", "levers", "ACCEL"))
    assert sl.requested() == {"addon_levers": ["dit_hoist", "dit_align"], "arm": "Z", "deterministic": False}
    # the kit's own FPF enable ran on the kit's bytes: fpf_engines re-bound the two TriMul forwards of the (stubbed) engine classes
    fe = sys.modules["fpf_engines"]
    assert fe.__file__.startswith(os.path.join(_stubs.TREE, "opt", "forward", "fast_inference", "src"))       # the kit's adapter, on the kit's bytes
    import opt_core.kernels as K                                                    # the TM-K3 package name is the core's copy (routed); the kit carries none (src/ holds fpf_engines only)
    assert not os.path.isfile(os.path.join(_stubs.TREE, "opt", "forward", "fast_inference", "src", "fpf_trimul", "kernels.py")) and os.path.exists(K.carried_path("fpf_trimul"))
    ck = rep["core_kernels"] if "core_kernels" in rep else stack.refresh()["core_kernels"]      # route_check's verdict is recorded (informational,
    assert "fpf_trimul" in ck and isinstance(ck["fpf_trimul"]["ok"], bool)                      # the report's own history), never a precondition of activation
    assert sorted(op for _e, op in fe._ORIG) == ["trimul_in", "trimul_out"] and "fpf_trimul_exact" in rep["levers_applied"]
    from opendde.model.triangular.triangular import TriangleMultiplicationOutgoing
    assert TriangleMultiplicationOutgoing.forward is not fe._ORIG[("opendde", "trimul_out")][1]
    import runner.batch_inference as BI                                              # the hook wraps at this import (kit behaviour)
    assert getattr(BI.get_default_runner, "_served_levers_wrapped", False)
    assert sl.STATE["wrapped"] is True
    # idempotent: a second enable returns the first report; a different line is refused by name
    again = stack.activate("exact")
    assert again["active"] and again["line"] == rep["line"]
    other = stack.activate("fast")
    assert "applied once per process" in other["reason"]
    assert stack.status()["active"]


def test_shim_is_inert_without_the_flag(fresh):
    """The kit's own contract: with ODDE_SERVED_LEVERS unset the shim arms nothing (the package never exports it for `off`)."""
    res = modes.resolve("exact", _stubs.TREE)
    for d in reversed(res.sys_path):
        sys.path.insert(0, d)
    import importlib.util
    spec = importlib.util.spec_from_file_location("sitecustomize_opendde_opt_probe", res.shim)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "odde_served_levers" not in sys.modules
    import runner.batch_inference as BI
    assert not getattr(BI.get_default_runner, "_served_levers_wrapped", False)


def test_exact_strips_preset_switches_before_the_shim_reads_them(fresh, monkeypatch):
    # a pre-set ODDE_ARM_U=1 takes precedence over Z in the hook (levers/ACCEL/odde_served_levers.py:31): the line's must-be-absent
    # switches are removed from the environment before the shim runs, so `exact` is arm Z whatever the caller had exported
    monkeypatch.setenv("ODDE_ARM_U", "1")
    monkeypatch.setenv("ODDE_ARM_T_TRIMUL", "stock")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rep = stack.activate("exact")
    assert rep["active"], rep.get("reason")
    for k in ("ODDE_ARM_U", "ODDE_ARM_T_TRIMUL", "PYTORCH_CUDA_ALLOC_CONF"):
        assert k not in os.environ, k
    assert sys.modules["odde_served_levers"].requested()["arm"] == "Z"


def test_missing_kit_directory_is_refused_by_name(fresh, tmp_path, monkeypatch):
    import shutil
    tree = tmp_path / "tree"
    shutil.copytree(os.path.join(_stubs.TREE, "opt", "opendde_opt"), str(tree / "opt" / "opendde_opt"))
    fi = modes.KIT_DIRS["fast_inference"]
    shutil.copytree(os.path.join(_stubs.TREE, "opt", fi), str(tree / "opt" / fi), symlinks=True, ignore=shutil.ignore_patterns("ARMT"))
    monkeypatch.setenv("MODEL_OPT", str(tree))
    rep = stack.activate("exact", tree=str(tree))
    assert not rep["active"] and "levers/ARMT" in rep["reason"]
    assert "odde_served_levers" not in sys.modules and "ODDE_SERVED_LEVERS" not in os.environ


def test_arm_declined_by_the_kit_is_a_fallback_not_an_applied_lever(fresh, monkeypatch):
    # ARM-T below sm80 / on CPU logs `NOT installing (stock path)` and returns without raising (odde_arm_t/__init__.py:532-535); the
    # hook then records {"arm": None, ...} with no error — the report must say arm_z fell back, partial, PARTIAL in the exit tally
    m = types.ModuleType("odde_served_levers")
    m.STATE = {"version": "0.2.0", "active": True, "wrapped": True, "errors": {}, "deterministic_forced": False,
               "installs": [{"event": "install", "requested": {"addon_levers": ["dit_hoist", "dit_align"], "arm": "Z"},
                             "levers": {"kit.cueq_cache": "/x/cueq_cache_shipped",
                                        "ditfast": {"dit_hoist": "installed", "dit_align": True},
                                        "odde_arm_t": {"arm": None, "version": "0.3", "composition": None, "min_tokens": 300}}}]}
    monkeypatch.setitem(sys.modules, "odde_served_levers", m)
    monkeypatch.setenv("ODDE_ADDON_LEVERS", "dit_hoist,dit_align")
    monkeypatch.setattr(stack, "_REPORT", {"active": True, "mode": "exact", "line": "S1", "levers_planned": list(modes._S_LEVERS),
                                           "levers_applied": ["served_levers_hook", "dit_hoist", "dit_align", "arm_z"], "levers_fallback": [], "partial": False})
    rep = stack.refresh()
    assert rep["levers_applied"] == ["cueq_tuned_cache", "dit_align", "dit_hoist", "served_levers_hook"]      # the record names no arm
    assert rep["partial"] and rep["levers_fallback"] == ["arm_z"]
    assert "PARTIAL" in _report.exit_tally_line(1) and "arm_z" in _report.exit_tally_line(1)
    # the same record with the arm installed is the applied lever
    m.STATE["installs"][0]["levers"]["odde_arm_t"]["arm"] = "Z"
    rep = stack.refresh()
    assert "arm_z" in rep["levers_applied"] and not rep["partial"] and rep["levers_fallback"] == []


# ------------------------------------------------------------------------------------------ the PENDING-REBASE gate (registry.TESTED_ON)
@pytest.mark.pending_rebase
def test_pending_rebase_gate_refuses_every_kit_mode_by_name_on_the_pin(fresh):
    """On the tree's pin the registry tests what it tests: every mode whose line holds an untested lever refuses by name with
    the levers listed, in every route; `off` is never gated by testing."""
    from opendde_opt import registry
    pin = _stubs.PINNED
    for mode, line in (("exact", "S1"), ("fast", "LSTAR2A"), ("big", "BIG_F")):
        want = registry.untested(modes.LINES[line].levers, pin)
        rep = stack.activate(mode, dry_run=True)
        if not want:                                                              # the line is fully tested on this pin: it resolves
            assert rep["reason"].startswith("dry run"), (mode, rep["reason"])
            continue
        assert not rep["active"], mode
        assert rep["reason"] == f"pending_rebase mode={mode} line={line} pin=opendde-{pin} levers_untested={','.join(want)}", rep["reason"]
        assert rep["levers_untested"] == want
        with pytest.raises(stack.ActivationError):
            stack.activate(mode, strict=True)
    off = stack.activate("off")
    assert "pending_rebase" not in off["reason"]
    for line, why in registry.LINES_ON_HOLD.items():                                  # a line held as a family refuses by name although its levers are tested
        rep = stack.activate(line, dry_run=True)
        if registry.untested(modes.LINES[line].levers, pin):
            continue
        assert not rep["active"] and f"line={line} pin=opendde-{pin} line_on_hold=" in rep["reason"] and rep["levers_untested"] == []


@pytest.mark.pending_rebase
def test_pending_rebase_state_of_the_tree(fresh):
    """The tree's own statement: which levers are tested on the pin today (the doc table renders from the same dict)."""
    from opendde_opt import registry
    assert _stubs.PINNED in registry.TESTED_ON
    assert registry.TESTED_ON["1.0.0"] == frozenset(registry.LEVERS)             # every carried lever was tested on 1.0.0
    unc = registry.untested(modes.LINES["S1"].levers, _stubs.PINNED)
    table = registry.render_status_table(_stubs.PINNED)
    for n in registry.LEVERS:
        assert f"| `{n}` |" in table
    assert all(f"| `{n}` | no" in table for n in unc)


def test_admitting_a_lever_admits_its_lines(fresh, monkeypatch):
    """Testing is data: with every lever of S1 tested on the pin the mode resolves; remove one and the line refuses naming exactly it."""
    from opendde_opt import registry
    rep = stack.activate("exact", dry_run=True)
    assert rep["reason"].startswith("dry run"), rep["reason"]
    monkeypatch.setitem(registry.TESTED_ON, _stubs.PINNED, frozenset(registry.LEVERS) - {"arm_z"})
    rep = stack.activate("exact", dry_run=True)
    assert rep["reason"] == f"pending_rebase mode=exact line=S1 pin=opendde-{_stubs.PINNED} levers_untested=arm_z"
    rep = stack.activate("fast", dry_run=True)                                    # a line without arm_z is untouched
    assert rep["reason"].startswith("dry run"), rep["reason"]
    assert registry.tested_on("arm_z") == tuple(v for v in sorted(registry.TESTED_ON) if v != _stubs.PINNED)


@pytest.mark.pending_rebase
def test_env_route_on_the_pin_exits_3_with_pending_rebase(fresh, tmp_path):
    """A fresh interpreter under OPENDDE_OPT=<mode> on this pin: the trigger fires, the gate refuses by name, exit 3 — stock never runs under a kit banner."""
    from opendde_opt import registry
    if not registry.untested(modes.LINES["S1"].levers, _stubs.PINNED):
        pytest.skip("S1 is tested on the pin")
    site = _stubs.make_site(str(tmp_path / "s2"))
    r = _stubs.run_py("import opendde_opt.stack as _S; _S.stack_mismatch = lambda tree=None: None; import runner; print('reached')", env={"OPENDDE_OPT": "exact"}, pythonpath=[site])
    assert r.returncode == 3 and "reached" not in r.stdout, r
    assert f"reason=pending_rebase mode=exact line=S1 pin=opendde-{_stubs.PINNED} levers_untested=" in r.stderr.replace("reason='", "reason="), r.stderr[-600:]


# ------------------------------------------------------------------------------------------ the box-start STACK ASSERTION (stack.stack_mismatch): named, never a refusal
@pytest.mark.stack_gate
def test_stack_assertion_names_a_wrong_stack_on_every_route_and_the_mode_applies(fresh, monkeypatch):
    import importlib.metadata as md
    want = stack.stack_pins(_stubs.TREE)
    assert set(want) == set(stack.STACK_PACKAGES) and want["torch"] == "2.7.1" and all(v == "0.10.0" for k, v in want.items() if k != "torch"), want
    have = {"torch": "2.7.1+cu126", "cuequivariance": "0.10.0", "cuequivariance-torch": "0.8.0", "cuequivariance-ops-cu12": "0.10.0", "cuequivariance-ops-torch-cu12": "0.10.0"}
    real_version = md.version
    def fake_version(name):
        if name in have:
            return have[name]
        if name in stack.STACK_PACKAGES:
            raise md.PackageNotFoundError(name)
        return real_version(name)                                                # opendde (the stub's dist) and everything else: as installed
    monkeypatch.setattr(md, "version", fake_version)
    for k in list(os.environ):                                                        # the routes below now APPLY (nothing refuses): keep this test's exports out of the next test's process
        monkeypatch.setenv(k, os.environ[k])
    monkeypatch.setattr(os, "environ", os.environ)                                   # (monkeypatch restores every variable it touched; new ones are removed below)
    env_before = set(os.environ)
    assert stack.stack_mismatch(_stubs.TREE) == "stack_mismatch:cuequivariance-torch=0.8.0!=0.10.0"
    assert stack.stack_line(_stubs.TREE).startswith("STACK MISMATCH stack_mismatch:cuequivariance-torch=0.8.0!=0.10.0 ")
    rep = stack.activate("exact", dry_run=True)                                       # the mode resolves with ALL its levers: the mismatch is a fact on its line, not a reason
    assert rep["stack_mismatch"] == "stack_mismatch:cuequivariance-torch=0.8.0!=0.10.0" and "stack_mismatch" not in (rep.get("reason") or "")
    assert rep["levers_planned"] == list(modes.resolve("exact", _stubs.TREE).line.levers) or set(rep["levers_planned"]) == set(modes.resolve("exact", _stubs.TREE).line.levers)
    assert any(n.startswith("NOTE stack_mismatch:cuequivariance-torch=0.8.0!=0.10.0 — ") for n in rep["notes"]), rep["notes"]
    assert " stack_mismatch=cuequivariance-torch=0.8.0!=0.10.0" in _report.activation_line(rep)
    off = stack.activate("off")                                                       # the stock route: runs, the mismatch named the same way
    assert "refused" not in off["reason"] and off["stack_mismatch"] == rep["stack_mismatch"]
    assert " stack_mismatch=cuequivariance-torch=0.8.0!=0.10.0" in _report.activation_line(off)
    have["cuequivariance-torch"] = "0.10.0"
    assert stack.stack_mismatch(_stubs.TREE) is None                             # the local tag +cu126 on torch matches 2.7.1
    del have["cuequivariance-ops-cu12"]
    assert stack.stack_mismatch(_stubs.TREE) == "stack_mismatch:cuequivariance-ops-cu12=absent!=0.10.0"
    for k in set(os.environ) - env_before:                                            # what activate() exported in this test (the stock base, the line's switches)
        monkeypatch.delenv(k, raising=False)
