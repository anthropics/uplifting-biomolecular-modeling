"""The env route: the .pth's finder installs only for a set E1_OPT (never for off / unset), fires once after the trigger module's own
body, refuses an unknown name / a missing variant / a missing weights file with NOT ACTIVE and exit 3 (an environment off the
kit's pins is NAMED on the ACTIVE line, never refused), and — armed on a green box —
wraps E1Predictor.__init__ so the kit's own apply(model, size=<variant>, …) runs at construction."""
import os
import subprocess
import sys
import types

from e1_opt import _autoload, cli, stack
from e1_opt.tests import _stubs


def test_install_only_for_a_set_switch():
    assert _autoload.install({}) is None and _autoload.install({"E1_OPT": "off"}) is None and _autoload.install({"E1_OPT": " OFF "}) is None
    f = _autoload.install({"E1_OPT": "exact"})
    try:
        assert isinstance(f, _autoload.Finder) and f.mode == "exact" and f in sys.meta_path and f.armed
        assert _autoload.install({"E1_OPT": "exact"}) is f
        assert f.find_spec("json") is None and f.find_spec("E1.other") is None
    finally:
        sys.meta_path.remove(f)


def _fire(box, mode, extra_env=None, code="import E1.predictor"):
    env = _stubs.child_env(box, E1_OPT=mode, **(extra_env or {}))
    p = subprocess.run([sys.executable, "-c", "import e1_opt._autoload\n" + code + "\nprint('REACHED')"], env=env, capture_output=True, text=True, timeout=120)
    return p


def test_trigger_refuses_unknown_and_no_variant(monkeypatch, tmp_path):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    p = _fire(box, "faster", {"E1_VARIANT": box["variant"]})
    assert p.returncode == 3 and "NOT ACTIVE: unknown mode 'faster'" in p.stdout and "REACHED" not in p.stdout
    p = _fire(box, "exact")
    assert p.returncode == 3 and "NOT ACTIVE: no variant:" in p.stdout
    p = _fire(box, "off")
    assert p.returncode == 0 and "REACHED" in p.stdout and "[e1-opt]" not in p.stdout            # off: no finder, stock untouched, silent by design
    p = _fire(box, "exact", {"E1_VARIANT": box["variant"]}, code="import json")
    assert p.returncode == 0 and "REACHED" in p.stdout and "[e1-opt]" not in p.stdout            # no trigger imported: nothing fires


def test_trigger_fires_and_names_what_is_untested(monkeypatch, tmp_path):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    p = _fire(box, "exact", {"E1_VARIANT": box["variant"]})
    # the placeholder kernel snapshot cannot hash to the pin in a subprocess: the trigger fires, ACTIVE names it (untested=), nothing is refused
    act = [s for s in p.stdout.splitlines() if s.startswith("[e1-opt] ACTIVE ")]
    assert p.returncode == 0 and len(act) == 1 and ' notes="hub kernel snapshot ' in act[0] and "REACHED" in p.stdout, p.stdout + p.stderr
    p = _fire(box, "exact", {"E1_VARIANT": box["variant"], "HF_HOME": str(tmp_path / "elsewhere")})   # no weights file at all: that IS a refusal (nothing can run), named, exit 3
    assert p.returncode == 3 and "[e1-opt] NOT ACTIVE: weights:" in p.stdout and "REACHED" not in p.stdout, p.stdout + p.stderr


def test_armed_wrap_applies_the_kit_once_with_the_variant(monkeypatch, tmp_path):
    import e1_opt
    box = _stubs.setup_box(monkeypatch, tmp_path)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])
    calls = []
    fake_kit = types.SimpleNamespace(_state={"applied": False, "record": None},
                                     apply=lambda model, **kw: (calls.append((model, kw)), fake_kit._state.update(applied=True), {"kit": "vTEST", "mode": "eager", "size": kw.get("size"), "n_on": 3, "lines": []})[-1])
    monkeypatch.setitem(sys.modules, stack.KIT_MODULE, fake_kit)
    monkeypatch.setattr(sys, "argv", ["prog"])
    rep = e1_opt.enable("exact", box["variant"])
    assert rep["active"] and rep["armed"] == ["E1Predictor"] and rep["route"] == "api"
    import E1.predictor as P
    from E1.modeling import E1ForMaskedLM
    assert getattr(P.E1Predictor.__init__, "_e1_opt_wrapped", False)
    m = E1ForMaskedLM.from_pretrained("x")
    monkeypatch.setenv(stack.ENV_DET, "1")
    from e1_opt import det as _det
    for k, v in _det.env_for(stack.load_pins(), 1).items():   # the recipe's environment in force (what the autoload exports at interpreter start)
        monkeypatch.setenv(k, v)
    P.E1Predictor(m)
    P.E1Predictor(m)
    assert len(calls) == 1 and calls[0][0] is m and calls[0][1] == {"size": box["variant"], "det": True}   # once, on the model, the size = the variant, the recipe on (E1_OPT_DET=1)
    assert e1_opt.status()["applied"] == {"kit": "vTEST", "mode": "eager", "size": box["variant"], "levers_on": 3, "lines": []} and "kit_mode_resolved" not in e1_opt.status()
    # late activation: a second enable with another variant is refused; the report is idempotent for the same one
    assert e1_opt.enable("exact", box["variant"])["active"]
    assert "already activated" in e1_opt.enable("exact", "600m")["reason"]


def test_late_activation_refused_when_an_instance_exists(monkeypatch, tmp_path):
    import e1_opt
    box = _stubs.setup_box(monkeypatch, tmp_path)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])
    fake_kit = types.SimpleNamespace(_state={"applied": False}, apply=lambda model, **kw: {"mode": "eager"})
    monkeypatch.setitem(sys.modules, stack.KIT_MODULE, fake_kit)
    import E1.predictor as P
    from E1.modeling import E1ForMaskedLM
    keep = P.E1Predictor(E1ForMaskedLM.from_pretrained("x"))
    rep = e1_opt.enable("exact", box["variant"])
    assert rep["active"] is False and "late activation refused" in rep["reason"] and "E1Predictor" in rep["reason"]
    del keep


def test_install_exports_the_recipe_env_at_interpreter_start(monkeypatch, tmp_path):
    """E1_OPT=exact E1_OPT_DET=1: install() exports the recipe's environment entries (CUBLAS_WORKSPACE_CONFIG, …) before numpy / torch import."""
    box = _stubs.setup_box(monkeypatch, tmp_path)
    env = {"E1_OPT": "exact", "E1_OPT_DET": "1"}
    for f in list(sys.meta_path):
        if isinstance(f, _autoload.Finder):
            sys.meta_path.remove(f)
    f = _autoload.install(env)
    assert f is not None
    want = stack.load_pins().DET_RECIPE["env"]
    assert want and all(env.get(k) == str(v) for k, v in want.items()), env
    sys.meta_path.remove(f)
    env2 = {"E1_OPT": "exact"}
    f2 = _autoload.install(env2)
    assert f2 is not None and not any(k in env2 for k in want)   # det 0: nothing exported
    sys.meta_path.remove(f2)


def test_det_on_without_the_recipe_environment_is_refused_by_name(monkeypatch, tmp_path, capsys):
    """E1_OPT_DET=1 at the trigger with a recipe environment entry absent: NOT ACTIVE naming the variable, exit 3 (the refusal form)."""
    import e1_opt
    box = _stubs.setup_box(monkeypatch, tmp_path)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])
    fake_kit = types.SimpleNamespace(_state={"applied": False, "record": None}, apply=lambda model, **kw: {"mode": "eager", "size": kw.get("size")})
    monkeypatch.setitem(sys.modules, stack.KIT_MODULE, fake_kit)
    monkeypatch.setattr(sys, "argv", ["prog"])
    stack._reset_for_tests()
    rep = e1_opt.enable("exact", box["variant"])
    assert rep["active"]
    monkeypatch.setenv(stack.ENV_DET, "1")
    want = stack.load_pins().DET_RECIPE["env"]
    for k in want:
        monkeypatch.delenv(k, raising=False)
    import E1.predictor as P
    from E1.modeling import E1ForMaskedLM
    m = E1ForMaskedLM.from_pretrained("x")
    try:
        P.E1Predictor(m)
        assert False, "the trigger must exit 3"
    except SystemExit as e:
        assert e.code == 3
    out = capsys.readouterr().out
    assert "[e1-opt] NOT ACTIVE: det=1 but the recipe's environment is not in force: CUBLAS_WORKSPACE_CONFIG=None" in out, out
    assert e1_opt.status()["active"] is False and "recipe's environment" in e1_opt.status()["reason"]


def test_det_value_outside_0_1_is_refused_by_name(monkeypatch, tmp_path, capsys):
    """E1_OPT_DET takes 0 or 1 only — the one reader (stack.det_level): 'yes' is refused by name at interpreter start (install) and at the CLI."""
    box = _stubs.setup_box(monkeypatch, tmp_path)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])
    stack._reset_for_tests()
    env = {"E1_OPT": "exact", "E1_OPT_DET": "yes"}
    for f in list(sys.meta_path):
        if isinstance(f, _autoload.Finder):
            sys.meta_path.remove(f)
    died = []
    monkeypatch.setattr(_autoload, "_die", lambda code: died.append(code) or (_ for _ in ()).throw(SystemExit(code)))   # os._exit stood in
    try:
        _autoload.install(env)
        assert False, "install must exit 3"
    except SystemExit as e:
        assert e.code == 3 and died == [3]
    assert "[e1-opt] NOT ACTIVE: E1_OPT_DET='yes': 0 or 1 only" in capsys.readouterr().out
    for f in list(sys.meta_path):
        if isinstance(f, _autoload.Finder):
            sys.meta_path.remove(f)
    monkeypatch.setenv(stack.ENV_DET, "on")
    stack._reset_for_tests()
    rc = cli.main(["check", "--variant", box["variant"], "--mode", "exact"])
    assert rc == 3 and "E1_OPT_DET='on': 0 or 1 only" in capsys.readouterr().out
