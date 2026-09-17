"""opt_core.mem.registry: lever declarations, the registry, the context, the flag grammar and the selection — refusals by name."""
import importlib
import sys

import pytest

from opt_core.mem import registry


def _ok(ctx):
    return None


def _apply(ctx):
    return registry.Applied(lever="fx_lever")


@pytest.fixture
def fx():
    """A fixture lever ``fx_lever`` with one declared setting ``rows``; removed after the test."""
    fn = registry.register("fx_lever", family="chunk", exact="band", exact_reason="rows re-order one reduction", applies=_ok,
                           preconditions=("hooks.module",), settings=("rows",))(_apply)
    yield fn.lever
    registry.unregister("fx_lever")


def test_lever_validation():
    with pytest.raises(registry.RegistryError):
        registry.Lever(name="Bad", family="chunk", exact="band", exact_reason="r", applies=_ok, apply=_apply)
    with pytest.raises(registry.RegistryError):
        registry.Lever(name="ok", family="magic", exact="band", exact_reason="r", applies=_ok, apply=_apply)
    with pytest.raises(registry.RegistryError):
        registry.Lever(name="ok", family="chunk", exact="close", exact_reason="r", applies=_ok, apply=_apply)
    with pytest.raises(registry.RegistryError):
        registry.Lever(name="ok", family="chunk", exact="band", exact_reason=" ", applies=_ok, apply=_apply)
    with pytest.raises(registry.RegistryError):
        registry.Lever(name="ok", family="chunk", exact="band", exact_reason="r", applies=_ok, apply=_apply, settings=("Rows",))
    with pytest.raises(registry.RegistryError):
        registry.Lever(name="ok", family="chunk", exact="band", exact_reason="r", applies=_ok, apply=_apply, frameworks=("mlx",))
    lv = registry.Lever(name="ok", family="chunk", exact="band", exact_reason="r", applies=_ok, apply=_apply, settings=["rows"])
    assert lv.settings == ("rows",) and lv.describe() == "ok [chunk; band] r"
    assert registry.setting_ref("ok", "rows") == "ctx.settings['ok']['rows']" and registry.off_ref("ok") == "switches={'ok': False}"
    assert registry.setting_word("ok", "rows") == "settings.ok.rows"


def test_register_get_names_table(fx):
    assert registry.get("fx_lever") is fx and "fx_lever" in registry.names("chunk") and "fx_lever" not in registry.names("offload")
    assert any(r["name"] == "fx_lever" and r["settings"] == ["rows"] for r in registry.table())
    with pytest.raises(registry.RegistryError):
        registry.register("fx_lever", family="chunk", exact="band", exact_reason="r", applies=_ok)(_apply)
    with pytest.raises(KeyError):
        registry.get("nope")
    assert registry.unregister("nope") is None


def test_allocator_levers_are_registered_by_import():
    importlib.import_module("opt_core.mem.allocator")
    assert registry.get("expandable_segments").family == "allocator" and registry.get("cache_release").settings == ("policy",)


def test_discover_records_absent_modules_and_propagates_broken_ones(monkeypatch):
    d = registry.discover(("allocator", "zz_not_a_module"))
    assert d["loaded"] == ["allocator"] and d["absent"] == ["zz_not_a_module"]
    assert registry.discover(("zz_not_a_module",))["absent"] == ["zz_not_a_module"]          # cached verdict

    def broken(name):
        raise ImportError("cannot import name 'x'")                                          # a defect inside the module: propagates
    monkeypatch.setattr(registry.importlib, "import_module", broken)
    with pytest.raises(ImportError):
        registry.discover(("zz_broken",))


def test_ctx_validation_hooks_and_settings(fx):
    with pytest.raises(ValueError):
        registry.Ctx(prefix="acme", tag="t")
    with pytest.raises(ValueError):
        registry.Ctx(prefix="ACME", tag="t", framework="mlx")
    ctx = registry.Ctx(prefix="ACME", tag="t", hooks={"fx_lever": {"module": object()}}, settings={"fx_lever": {"rows": "128"}},
                       environ={"ACME_BIG_FX_LEVER_ROWS": "256"})                            # the environment is never read for a setting
    assert ctx.hook("fx_lever", "module") is not None and ctx.hook("fx_lever", "dim", 7) == 7
    assert set(ctx.require("fx_lever", "module")) == {"module"}
    with pytest.raises(registry.HookMissing) as e:
        ctx.require("fx_lever", "module", "tensor")
    assert e.value.refusal.precondition == "hooks.tensor" and e.value.refusal.details["hooks_given"] == ["module"]
    assert ctx.setting("fx_lever", "rows", 32, cast=int) == 128                               # the kit's string value, cast
    ctx2 = registry.Ctx(prefix="ACME", tag="t", settings={"fx_lever": {"rows": 64}}, environ={})
    assert ctx2.setting("fx_lever", "rows", 32, cast=int) == 64                               # the kit's value as given (no cast of a non-string)
    assert registry.Ctx(prefix="ACME", tag="t", environ={"ACME_BIG_FX_LEVER_ROWS": "9"}).setting("fx_lever", "rows", 32) == 32   # unset in ctx.settings: the default; the environment has no effect
    with pytest.raises(registry.RegistryError):
        ctx.setting("fx_lever", "cols", 1)                                                    # undeclared setting
    bad = registry.Ctx(prefix="ACME", tag="t", settings={"fx_lever": {"rows": "many"}})
    with pytest.raises(registry.RefusalError) as e:
        bad.setting("fx_lever", "rows", 32, cast=int)
    assert e.value.refusal.precondition == "settings.fx_lever.rows" and "ctx.settings['fx_lever']['rows']='many' is not a valid rows" in e.value.refusal.reason


def test_selection_grammar(fx):
    importlib.import_module("opt_core.mem.allocator")
    line = ("expandable_segments", "fx_lever")
    s = registry.selection(line)
    assert s.levers == line and not s.refusals and not s.allow_partial and s.flags == {}
    s = registry.selection(line, switches={"fx_lever": False, "cache_release": True, "expandable_segments": True}, allow_partial=True)
    assert s.levers == ("expandable_segments", "cache_release") and s.off_by_flag == ("fx_lever",) and s.on_by_flag == ("cache_release",)
    assert s.allow_partial and not s.refusals and s.flags == {"fx_lever": False, "cache_release": True, "expandable_segments": True}
    s = registry.selection(line, switches={"fx_lever": "0", "cache_release": "1"})                # the kit may pass the strings it parsed
    assert s.off_by_flag == ("fx_lever",) and s.on_by_flag == ("cache_release",) and s.flags == {"fx_lever": False, "cache_release": True}
    s = registry.selection(line, switches={"fx_lever": "yes"})
    assert [(r.lever, r.precondition) for r in s.refusals] == [("fx_lever", "switch.fx_lever")] and "expected on (True/1) or off (False/0)" in s.refusals[0].reason
    s = registry.selection(line, switches={"nope": True})
    assert [(r.lever, r.precondition) for r in s.refusals] == [("big", "switch.nope")] and "names no registered lever and no line lever" in s.refusals[0].reason
    s = registry.selection(line, switches={"nope": False})
    assert [r.precondition for r in s.refusals] == ["switch.nope"]                             # an unknown name refuses whatever its value
    s = registry.selection(line, switches={"cache_release": False})                             # off for a registered lever outside the line: nothing
    assert not s.refusals and s.levers == line and s.off_by_flag == () and s.on_by_flag == ()
    s = registry.selection(("expandable_segments", "ghost"))
    assert [(r.lever, r.precondition) for r in s.refusals] == [("ghost", "registry")] and s.levers == ("expandable_segments", "ghost")
    s = registry.selection(("expandable_segments", "ghost"), switches={"ghost": False})
    assert not s.refusals and s.levers == ("expandable_segments",) and s.off_by_flag == ("ghost",)
    s2 = registry.selection(("expandable_segments",), switches={"ghost": True})
    assert [(r.lever, r.precondition) for r in s2.refusals] == [("big", "switch.ghost")]
    with pytest.raises(registry.RegistryError):
        registry.selection(("fx_lever", "fx_lever"))
    with pytest.raises(TypeError):
        registry.selection(line, {"fx_lever": False})                                            # switches are keyword-only: a positional environ mapping is an error, not a silent read
    assert s.as_dict()["off_by_flag"] == ["ghost"]


def test_selection_reads_no_environment(fx, monkeypatch):
    """The variables of the former flag grammar have no effect: set in os.environ, the selection is the line as declared, nothing is
    switched, nothing refuses, and the opt-out stays off."""
    line = ("expandable_segments", "fx_lever")
    for k, v in {"ACME_BIG_FX_LEVER": "0", "ACME_BIG_CACHE_RELEASE": "1", "ACME_BIG_FX_LEVER_ROWS": "8", "ACME_BIG_ALLOW_PARTIAL": "1",
                 "ACME_BIG_LINE": "huge", "ACME_BIG_NOPE": "1"}.items():
        monkeypatch.setenv(k, v)
    s = registry.selection(line)
    assert s.levers == line and s.off_by_flag == () and s.on_by_flag == () and not s.refusals and not s.allow_partial and s.flags == {}


def test_refusal_forms():
    r = registry.refuse("lv", "hooks.module", "no module", given=[])
    assert str(r) == "lv: hooks.module: no module" and r.as_dict()["details"] == {"given": []}
    e = registry.RefusalError(r)
    assert e.refusal is r and str(e) == str(r)


def test_framework_is_the_one_lazy_import_and_refuses_by_name(monkeypatch):
    import types
    with pytest.raises(registry.RegistryError):
        registry.framework("os", "fx_lever")                                                 # only torch / jax are frameworks
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    assert registry.framework("torch", "fx_lever").__name__ == "torch"
    monkeypatch.setitem(sys.modules, "torch", None)                                          # a process without torch
    with pytest.raises(registry.RefusalError) as e:
        registry.framework("torch", "fx_lever")
    assert e.value.refusal.precondition == "torch" and e.value.refusal.lever == "fx_lever"
