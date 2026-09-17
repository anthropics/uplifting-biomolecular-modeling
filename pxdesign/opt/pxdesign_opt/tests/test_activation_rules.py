"""Activation refusal rules, the late-activation rule, idempotence, and the hook that applies the REAL kit install() at the kit's
activation point (InferenceRunner.load_checkpoint) — on stub upstream modules, no GPU."""
import os
import sys

import pytest

from . import _stubs


def _box(monkeypatch, stack, **kw):
    _stubs.install_torch()
    PD = _stubs.install_upstream()
    _stubs.fake_box(monkeypatch, stack, **kw)
    return PD


def test_exact_activates_exports_env_and_arms_hook(fresh_stack, monkeypatch, capsys):
    stack = fresh_stack
    _box(monkeypatch, stack)
    monkeypatch.setenv("PXD_HOIST_MODE", "rows")                                 # a caller switch: dropped
    rep = stack.activate("exact")
    assert rep["active"] and rep["mode"] == "exact" and rep["applied"] == "deferred"
    assert rep["levers_planned"] == ["h1", "h2", "h3", "h4", "h5"]
    assert os.environ["PXD_HOIST_MODE"] == "shape" and os.environ["PXD_HOIST"] == "1" and os.environ["PXD_HOIST_MASK"] == "1"
    assert rep["unset"] == ["PXD_HOIST_MODE"]
    err = capsys.readouterr().err
    assert "[pxdesign-opt] ACTIVE mode=exact" in err and "levers=h1,h2,h3,h4,h5" in err and "applied=deferred" in err
    from pxdesign.runner.inference import InferenceRunner
    assert getattr(InferenceRunner.load_checkpoint, "_pxdesign_opt_hook", False)
    # idempotent; a different mode is refused
    assert stack.activate("exact")["active"]
    r2 = stack.activate("off")
    assert not r2["active"] and r2.get("refused") and "already active" in r2["reason"]
    with pytest.raises(stack.ActivationError):
        stack.activate("off", strict=True)


def test_hook_applies_the_kits_install_after_checkpoint_load(fresh_stack, monkeypatch, capsys):
    stack = fresh_stack
    _box(monkeypatch, stack)
    stack.activate("exact")
    from pxdesign.runner.inference import InferenceRunner
    runner = InferenceRunner(configs=None)                                        # load_checkpoint() -> hook -> hoist.install(model)
    import pxd_xattempt.hoist as H
    assert H._STATE["installed"] is True
    assert runner.model.diffusion_module._pxd_hoist_ready is True
    from protenix.model.modules.diffusion import DiffusionModule
    from protenix.model.modules.transformer import AttentionPairBias, ConditionedTransitionBlock
    import protenix.model.modules.primitives as PR
    import pxdesign.model.pxdesign as P
    assert DiffusionModule.f_forward is H._f_forward_hoisted and AttentionPairBias.forward is H._apb_forward_hoisted
    assert ConditionedTransitionBlock.forward is H._ctb_forward_hoisted and PR._local_attention is H._local_attention_hoisted
    assert getattr(P.sample_diffusion, "_pxd_hoist_wrapper", False)
    rep = stack.status()
    assert rep["applied"] == "installed" and rep["levers_applied"] == ["h1", "h2", "h3", "h4", "h5"] and rep["levers_fallback"] == []
    assert rep["applications"][0]["index"] == 1
    err = capsys.readouterr().err
    assert "[pxdesign-opt] APPLIED model#1 levers=h1,h2,h3,h4,h5 fallbacks=none" in err
    # every block of the stub model was tagged by the kit (its own records)
    dm = runner.model.diffusion_module
    assert all(b.attention_pair_bias._pxd_cache_slot == ("tok", i) for i, b in enumerate(dm.diffusion_transformer.blocks))


def test_late_activation_refused_once_a_model_exists(fresh_stack, monkeypatch, capsys):
    stack = fresh_stack
    PD = _box(monkeypatch, stack)
    m = PD()                                                                      # a model built before activation
    rep = stack.activate("exact")
    assert not rep["active"] and "ProtenixDesign instance already exists" in rep["reason"]
    assert "[pxdesign-opt] NOT ACTIVE:" in capsys.readouterr().err
    with pytest.raises(stack.ActivationError):
        stack.activate("exact", strict=True)
    del m


def test_refused_when_the_kit_already_installed(fresh_stack, monkeypatch):
    stack = fresh_stack
    _box(monkeypatch, stack)
    sys.path.insert(0, stack.kit_home())
    import pxd_xattempt.hoist as H
    H._STATE["installed"] = True
    try:
        rep = stack.activate("exact")
        assert not rep["active"] and "install() already ran" in rep["reason"]
    finally:
        H._STATE["installed"] = False


def test_no_gpu_and_stock_pin_refuse_by_name(fresh_stack, monkeypatch, capsys):
    """The refusals: no visible GPU (nothing can engage) and the installed upstream not the pinned stock (another commit or a modified checkout
    changes what "stock" means) — both NOT ACTIVE by name; mode off is upstream as installed and is never gated on the pins."""
    stack = fresh_stack
    _box(monkeypatch, stack, cuda=False)
    rep = stack.activate("exact")
    assert not rep["active"] and rep["reason"].startswith("no visible GPU")
    stack.reset_for_tests()
    _box(monkeypatch, stack, pinned=False)
    rep = stack.activate("exact")
    assert not rep["active"] and rep["reason"].startswith("the installed upstream is not the pinned stock — pxdesign 0.1.0: installed from git commit deadbeef")
    assert rep["stock_pins"] == {"pinned": False, "bad": ["pxdesign 0.1.0: installed from git commit deadbeef; want f788441"]}
    assert stack.pins_line(stack.pins_report()).startswith("PINS pinned=0 stack=OPEN differs=pxdesign 0.1.0: installed from git commit deadbeef")
    err = capsys.readouterr().err
    assert "[pxdesign-opt] NOT ACTIVE: the installed upstream is not the pinned stock" in err
    stack.reset_for_tests()
    rep = stack.activate("off")                                                     # off: not gated on the pins
    assert not rep["active"] and rep["reason"] == "mode off: stock, nothing applied" and not rep.get("refused")
    stack.reset_for_tests()
    rep = stack.activate("exact", dry_run=True)                                     # check: the same facts as would_refuse
    assert rep["would_refuse"].startswith("the installed upstream is not the pinned stock")


def test_environment_uncertainty_is_named_on_the_activation_line_never_refused(fresh_stack, monkeypatch, capsys):
    """The engagement rule: a GPU class the kit is not measured on (MEASURED_CC: sm80, sm90), a torch build other than the pin, a target-GPU word
    that does not match the card, TF32 override variables — each is a note on the ACTIVE / DRY-RUN / STOCK line (`notes=`), and the mode engages.
    On a measured box with nothing to name the line has no notes= tail (its bytes are the measured ones)."""
    stack = fresh_stack
    _box(monkeypatch, stack)                                                          # H100, pinned torch, no target word, no TF32 words
    rep = stack.activate("exact")
    assert rep["active"] and rep["notes"] == []
    err = capsys.readouterr().err
    assert "[pxdesign-opt] ACTIVE mode=exact" in err and "notes=" not in err
    stack.reset_for_tests()
    _box(monkeypatch, stack, cc=(10, 0), name="NVIDIA B200", torch="2.7.0+cu128")
    monkeypatch.setenv(stack.ENV_TARGET_GPU, "H100")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    rep = stack.activate("fast")
    assert rep["active"] and rep["levers_planned"] == ["h1", "h2", "h3", "h4", "h5"] and rep["package_levers"] == ["featdiet", "padmask", "tf32", "sdedup"]
    notes = rep["notes"]
    assert notes[0] == "GPU NVIDIA B200 (sm100) is outside the measured classes (sm80, sm90): the levers engage, numerics and speed are unmeasured on it"
    assert notes[1] == "torch 2.7.0+cu128 differs from the pinned 2.3.1+cu121 (stock/PINS.json)"
    assert notes[2] == "MODEL_OPT_TARGET_GPU=H100 but the visible GPU is NVIDIA B200"
    assert notes[3].startswith("TF32 library override in the environment (NVIDIA_TF32_OVERRIDE=0)")
    err = capsys.readouterr().err
    line = next(l for l in err.splitlines() if "] ACTIVE mode=fast" in l)
    assert line.endswith(" notes=" + "; ".join(notes)) and " tier=2 notes=GPU NVIDIA B200 (sm100) is outside the measured classes" in line
    stack.reset_for_tests()
    rep = stack.activate("off")                                                     # the stock arm names the same facts
    line = next(l for l in capsys.readouterr().err.splitlines() if "] STOCK mode=off" in l)
    assert line.endswith("env=none notes=" + "; ".join(rep["notes"])) and len(rep["notes"]) == 4
    stack.reset_for_tests()
    for k in ("PXD_HOIST", "PXD_HOIST_MODE", "PXD_HOIST_MASK"):
        os.environ.pop(k, None)                                                       # the switches the activation above exported (a caller-set switch is its own note)
    rep = stack.activate("big", dry_run=True)                                     # check: DRY-RUN … notes=…, nothing to refuse
    assert rep["would_refuse"] is None and len(rep["notes"]) == 4
    assert "] DRY-RUN mode=big" in capsys.readouterr().err
    assert not hasattr(stack, "SUPPORTED_CC") and not hasattr(stack, "ENV_MIN_MEM") and stack.MEASURED_CC == ("8.0", "9.0")

def test_off_is_stock_nothing_applied(fresh_stack, monkeypatch, capsys):
    stack = fresh_stack
    _box(monkeypatch, stack)
    rep = stack.activate("off")
    assert not rep["active"] and rep["mode"] == "off" and rep["levers_planned"] == [] and rep["env"] == {}
    assert "PXD_HOIST" not in os.environ
    assert "mode=off stock" in capsys.readouterr().err
    from pxdesign.runner.inference import InferenceRunner
    assert not getattr(InferenceRunner.load_checkpoint, "_pxdesign_opt_hook", False)
    assert "pxd_xattempt.hoist" not in sys.modules


def test_dry_run_applies_nothing(fresh_stack, monkeypatch, capsys):
    stack = fresh_stack
    _box(monkeypatch, stack)
    rep = stack.activate("exact", dry_run=True)
    assert rep["dry_run"] and not rep["active"] and rep["would_refuse"] is None
    assert "PXD_HOIST" not in os.environ and stack.status()["reason"].startswith("enable() has not run")
    from pxdesign.runner.inference import InferenceRunner
    assert not getattr(InferenceRunner.load_checkpoint, "_pxdesign_opt_hook", False)
    assert "[pxdesign-opt] DRY-RUN mode=exact" in capsys.readouterr().err
    stack.reset_for_tests()
    _box(monkeypatch, stack, cuda=False)
    rep = stack.activate("exact", dry_run=True)
    assert rep["would_refuse"].startswith("no visible GPU")


def test_hook_arms_at_import_when_runner_module_not_yet_imported(fresh_stack, monkeypatch):
    stack = fresh_stack
    _box(monkeypatch, stack)
    sys.modules.pop("pxdesign.runner.inference")
    rep = stack.activate("exact")
    assert rep["hook"] == "wrap at import"
    assert any(isinstance(f, stack._ModuleFinder) and f.module_name == stack.RUNNER_MODULE for f in sys.meta_path)


def _partial_install(monkeypatch):
    """The kit's install() leaving one family without its records (the module-global `_local_attention` not rebound: h5 falls back)."""
    import pxd_xattempt.hoist as H
    import protenix.model.modules.primitives as PR
    real = H.install

    def install(model):
        out = real(model)
        PR._local_attention = lambda *a, **k: None                                  # h5's record undone after install(): a partial application
        return out
    monkeypatch.setattr(H, "install", install)


def test_a_mode_is_all_of_its_levers_partial_application_refuses_by_name(fresh_stack, monkeypatch, capsys):
    """A mode engages every lever of its set or refuses by name — it never runs under its name with a subset. A hoist family that cannot run on
    the loaded model is named (APPLIED fallbacks=) and the hook raises ActivationError after printing the NOT ACTIVE line: the CLI verbs exit 3 on
    it, a Python caller of enable() sees the error (status(): apply_failed, partial=1 on the EXIT tally)."""
    from pxdesign_opt import report
    stack = fresh_stack
    _box(monkeypatch, stack)
    stack.activate("exact")
    _partial_install(monkeypatch)
    assert not hasattr(stack, "gate_partial") and "gate" not in stack._HOOK
    from pxdesign.runner.inference import InferenceRunner
    with pytest.raises(stack.ActivationError, match="lever application incomplete"):
        InferenceRunner(configs=None)
    rep = stack.status()
    assert rep["partial"] is True and rep["apply_failed"] is True and rep["levers_fallback"] == ["h5"] and rep["levers_applied"] == ["h1", "h2", "h3", "h4"]
    assert list(rep["partial_reasons"]) == ["h5"] and rep["refusal_logged"] is True
    err = capsys.readouterr().err
    assert "APPLIED model#1 levers=h1,h2,h3,h4 fallbacks=h5" in err
    assert "[pxdesign-opt] NOT ACTIVE: lever application incomplete: {'h5': " in err and "PARTIAL" not in err and "run goes on" not in err
    assert "partial=1 fallbacks=h5" in report.exit_tally()


def test_partial_application_ends_the_process_on_the_env_route(fresh_stack, monkeypatch, capsys):
    """On the PXDESIGN_OPT route (upstream's own console script; activation trigger = the importing package) the same refusal ends the process
    with the kit's code: the NOT ACTIVE line, then SystemExit(3) out of the load_checkpoint hook — the run never goes on with a subset."""
    stack = fresh_stack
    _box(monkeypatch, stack)
    stack.activate("exact", trigger="pxdesign")
    _partial_install(monkeypatch)
    from pxdesign.runner.inference import InferenceRunner
    with pytest.raises(SystemExit) as ex:
        InferenceRunner(configs=None)
    assert ex.value.code == 3
    err = capsys.readouterr().err
    assert "APPLIED model#1 levers=h1,h2,h3,h4 fallbacks=h5" in err and "[pxdesign-opt] NOT ACTIVE: lever application incomplete" in err

def test_warm_returns_designs_not_active_code(monkeypatch, capsys):
    """warm's exit code is design's when design exited NOT ACTIVE (3): the verbs agree on the code."""
    from pxdesign_opt import cli, warm
    monkeypatch.setattr(warm, "run", lambda *a, **k: {"status": "FAIL", "exit_code": 3, "killed": False, "reason": "design exited 3 (NOT ACTIVE: the lever was not applied, or not wholly)", "mode": "exact"})
    monkeypatch.setattr(warm, "summary_line", lambda res: "WARM FAIL")
    assert cli.main(["warm", "--mode", "exact"]) == cli.EXIT_NOT_ACTIVE
    monkeypatch.setattr(warm, "run", lambda *a, **k: {"status": "FAIL", "exit_code": 1, "killed": False, "reason": "design exited 1", "mode": "exact"})
    assert cli.main(["warm", "--mode", "exact"]) == cli.EXIT_FAIL
    monkeypatch.setattr(warm, "run", lambda *a, **k: {"status": "FAIL", "exit_code": 3, "killed": True, "reason": "timeout", "mode": "exact"})
    assert cli.main(["warm", "--mode", "exact"]) == cli.EXIT_FAIL
