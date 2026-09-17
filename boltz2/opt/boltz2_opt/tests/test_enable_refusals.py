"""enable(): refusals by name (the in-process route is not a kit line), one mode per process, off, unknown, strict, status()."""
import sys
import types

import pytest

import boltz2_opt
from opt_core import instances as core_instances

from .. import modes, stack
from . import _stubs


@pytest.fixture(autouse=True)
def _fresh():
    boltz2_opt._reset_for_tests()
    core_instances._COUNTERS.pop(stack.MODEL_CLASS, None)   # the core's counter for the model class, fresh per test
    yield
    boltz2_opt._reset_for_tests()


def test_status_before_any_activation():
    s = boltz2_opt.status()
    assert s["active"] is False and s["mode"] is None


def test_off_is_inactive_and_never_strict_raises(capsys):
    r = boltz2_opt.enable("off", strict=True)
    assert r["active"] is False and r["mode"] == "off" and "mode off" in r["reason"]
    assert "[boltz2-opt] NOT ACTIVE: mode off" in capsys.readouterr().out
    assert boltz2_opt.status()["mode"] == "off"


def test_unknown_mode_refused(capsys):
    r = boltz2_opt.enable("turbo")
    assert r["active"] is False and "unknown mode" in r["reason"]
    with pytest.raises(boltz2_opt.ActivationError):
        boltz2_opt.enable("turbo", strict=True)


def test_exact_in_process_is_refused_by_the_kit_fact_when_the_box_is_fine(monkeypatch, capsys):
    _stubs.gated_ok(monkeypatch)
    r = boltz2_opt.enable("exact")
    assert r["active"] is False and r["gated"] is True and r["pinned_route"] == "worker"
    assert "in-process route of exact is not a kit line" in r["reason"] and "no stock-CLI hook" in r["reason"] and "pred --mode exact" in r["reason"]
    assert r["plan"]["env"] == modes.env_row("exact") and r["plan"]["reasons"] == []
    assert r["gpu"] == "NVIDIA H100 80GB HBM3" and r["boltz_version"] == "2.2.1"
    out = capsys.readouterr().out
    assert out.startswith("[boltz2-opt] NOT ACTIVE: in-process route of exact")
    assert boltz2_opt.status()["reason"] == r["reason"]


def test_exact_on_a_box_without_gpu_or_pin_is_refused_with_the_gate_reasons(monkeypatch):
    monkeypatch.setattr(stack, "gpu_probe", lambda: None)
    monkeypatch.setattr(stack, "pins_check", lambda ckpt=False: (["boltz: not installed"], {"version": None, "pinned": False}))
    monkeypatch.delenv("BOLTZ_CACHE", raising=False)
    r = boltz2_opt.enable("fast")
    assert r["active"] is False and r["gated"] is False
    assert "pin: boltz: not installed" in r["reason"] and "no NVIDIA GPU" in r["reason"] and "BOLTZ_CACHE" in r["reason"]


def test_strict_raises_after_printing(monkeypatch, capsys):
    _stubs.gated_ok(monkeypatch)
    with pytest.raises(boltz2_opt.ActivationError) as e:
        boltz2_opt.enable("fast", strict=True, trigger="boltz.model")
    assert "not a kit line" in str(e.value)
    assert "NOT ACTIVE" in capsys.readouterr().out
    assert boltz2_opt.status()["trigger"] == "boltz.model"


def test_one_mode_per_process_and_idempotence(monkeypatch):
    _stubs.gated_ok(monkeypatch)
    r1 = boltz2_opt.enable("exact"); r1b = boltz2_opt.enable("exact")
    assert r1b["reason"] == r1["reason"]
    r2 = boltz2_opt.enable("fast")
    assert "already requested" in r2["reason"] and r2["active"] is False
    with pytest.raises(boltz2_opt.ActivationError):
        boltz2_opt.enable("fast", strict=True)


def test_late_activation_refused_once_a_model_instance_exists(monkeypatch):
    _stubs.gated_ok(monkeypatch)
    pkg = types.ModuleType("boltz"); model = types.ModuleType("boltz.model"); models = types.ModuleType("boltz.model.models"); b2 = types.ModuleType("boltz.model.models.boltz2")

    class Boltz2:
        def __init__(self, *a, **k):
            self.ok = True
    b2.Boltz2 = Boltz2
    for name, mod in (("boltz", pkg), ("boltz.model", model), ("boltz.model.models", models), ("boltz.model.models.boltz2", b2)):
        monkeypatch.setitem(sys.modules, name, mod)
    stack.arm_instance_counter()
    assert core_instances.instance_check(*stack.MODEL_CLASS)["method"] == "counted"
    Boltz2()
    r = boltz2_opt.enable("exact")
    assert "late activation refused: a Boltz2 model instance already exists" in r["reason"]


def test_late_activation_refused_once_a_kit_lever_reports_applied(monkeypatch):
    _stubs.gated_ok(monkeypatch)
    lev = types.ModuleType("boltz_trunk_levers"); lev._STATE = {"applied": True, "levers": ("mask",)}
    monkeypatch.setitem(sys.modules, "boltz_trunk_levers", lev)
    r = boltz2_opt.enable("exact")
    assert "a kit lever is already applied in this process (boltz_trunk_levers)" in r["reason"]
    hoist = types.ModuleType("boltz_dit_hoist"); hoist.STATS = {"level": 2}
    monkeypatch.setitem(sys.modules, "boltz_dit_hoist", hoist)
    assert set(stack.kit_lever_applied()) == {"boltz_trunk_levers", "boltz_dit_hoist"}


def test_allowed_after_import_before_instance(monkeypatch):
    """The rule's permissive half: with boltz.model imported and no instance, the only refusal left is the kit fact."""
    _stubs.gated_ok(monkeypatch)
    for name in ("boltz", "boltz.model", "boltz.model.models", "boltz.model.models.boltz2"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["boltz.model.models.boltz2"].Boltz2 = type("Boltz2", (), {})
    r = boltz2_opt.enable("exact")
    assert "late activation" not in r["reason"] and "not a kit line" in r["reason"]
