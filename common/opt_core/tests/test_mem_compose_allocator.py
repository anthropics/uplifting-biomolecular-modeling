"""opt_core.mem.compose (big on a ModeTable) and opt_core.mem.allocator (the allocator levers, the counters, the XLA rule) — CPU, no torch."""
import sys

import pytest

import importlib.util

_TORCH_IMPORTABLE = importlib.util.find_spec("torch") is not None
needs_importable_torch = pytest.mark.skipif(not _TORCH_IMPORTABLE, reason="torch is not importable here: the allocator lever refuses by name "
                                                                          "(precondition 'torch', held by test_expandable_segments_refuses_a_read_only_environ_and_a_missing_torch)")

from opt_core import mem, modes
from opt_core.mem import allocator, registry


def _cuda_present() -> bool:
    """These tests hold the no-CUDA path (refusal / named fallback); on a CUDA box they skip by name."""
    try:
        import torch                                       # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:                                      # noqa: BLE001
        return False


no_cuda_path = pytest.mark.skipif(_cuda_present(), reason="CUDA present: this test holds the no-CUDA refusal/fallback path")

FAST = modes.ModeTable(("off", "exact", "fast"), "fast")
EXACT_ONLY = modes.ModeTable(("off", "exact"), "off")


# ------------------------------------------------------------------------------------------------------------ compose


def test_compose_big_extends_the_table():
    t, line = mem.compose_big(FAST, levers=("a_lever", "b_lever"), drop=("graph_sampler",))
    assert t.modes == ("off", "exact", "fast", "big") and t.default == "fast" and t.check("BIG") == "big"
    assert modes.mode_argument(None, "big", t) == "big" and "big" in t.kit_modes
    assert line == mem.BigLine(base="fast", levers=("a_lever", "b_lever"), drop=("graph_sampler",),
                                 lines={"default": {"levers": ("a_lever", "b_lever"), "tier": None, "cost_note": "unstated"}})
    assert line.describe() == "big = fast + [a_lever,b_lever] - [graph_sampler]" and line.base_rule == mem.BASE_RULE
    assert line.as_dict() == {"name": "big", "base": "fast", "levers": ["a_lever", "b_lever"], "drop": ["graph_sampler"],
                              "base_rule": "fast-else-exact",
                              "lines": {"default": {"levers": ["a_lever", "b_lever"], "tier": None, "cost_note": "unstated"}},
                              "auto": False, "auto_default": False}
    t2, line2 = mem.compose_big(EXACT_ONLY, levers=["a_lever"])
    assert line2.base == "exact" and t2.modes == ("off", "exact", "big")
    with pytest.raises(modes.ModeError):
        mem.compose_big(t, levers=("a_lever",))                                            # composed twice
    with pytest.raises(modes.ModeError):
        mem.compose_big(modes.ModeTable(("off", "mem64"), "off"))                          # nothing to compose on
    with pytest.raises(modes.ModeError):
        mem.compose_big(FAST, base="off")
    with pytest.raises(ValueError):
        mem.compose_big(FAST, levers=("Bad",))
    with pytest.raises(ValueError):
        mem.compose_big(FAST, levers=("a", "a"))
    with pytest.raises(TypeError):
        mem.compose_big(("off", "fast"), levers=("a",))
    assert mem.base_for(FAST) == "fast" and mem.base_for(EXACT_ONLY) == "exact"


def test_compose_big_records_a_base_override():
    _, line = mem.compose_big(FAST, base="exact", levers=("a_lever",))
    assert line.base == "exact" and line.base_rule == "override" and line.describe().endswith("(base override)")


# ------------------------------------------------------------------------------------------------------------ allocator


def test_conf_parsing():
    assert allocator.parse_conf(None) == {} and allocator.parse_conf(" ") == {}
    assert allocator.parse_conf("expandable_segments:True, max_split_size_mb:128") == {"expandable_segments": "True", "max_split_size_mb": "128"}
    assert allocator.format_conf({"a": "1", "b": "2"}) == "a:1,b:2"
    with pytest.raises(ValueError):
        allocator.parse_conf("garbage")


def test_snapshot_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)                                          # torch absent, whatever ran before
    s = allocator.snapshot({"XLA_FLAGS": "--x"}, framework="jax")
    assert s["framework"] == "jax" and s["torch"]["torch"] == {"imported": False, "version": None, "cuda_initialized": None, "runtime_api": None}
    assert s["jax"] == {"XLA_PYTHON_CLIENT_PREALLOCATE": None, "XLA_PYTHON_CLIENT_MEM_FRACTION": None, "XLA_PYTHON_CLIENT_ALLOCATOR": None,
                        "XLA_FLAGS": "--x", "jax": allocator.jax_state()}                   # the interpreter's own jax state (absent on the CPU box)
    assert allocator.torch_settings({"PYTORCH_CUDA_ALLOC_CONF": "bad"})["parsed"] == {"<malformed>": "PYTORCH_CUDA_ALLOC_CONF entry 'bad' is not key:value"}
    assert allocator.counters()["max_allocated_gib"] is None and allocator.reset_peak() is False


@needs_importable_torch
def test_expandable_segments_lever():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
    ctx = mem.Ctx(prefix="ACME", tag="t", environ=env)
    rec = mem.apply(["expandable_segments"], ctx, base="fast")
    a = rec.applied[0]
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128,expandable_segments:True"        # the pre-set key is kept
    assert a.settings == {"expandable_segments": True, "conf": "max_split_size_mb:128,expandable_segments:True", "applied_via": "env",
                          "cuda_graphs": False}
    assert rec.allocator["writes"] == [{"lever": "expandable_segments", "name": "PYTORCH_CUDA_ALLOC_CONF", "before": "max_split_size_mb:128",
                                        "after": "max_split_size_mb:128,expandable_segments:True", "via": "env"}]
    assert rec.allocator["settings"]["expandable_segments"] is True and rec.exact == "bitwise"
    assert rec.active_line("t").endswith("allocator=expandable_segments:True,conf:max_split_size_mb:128,expandable_segments:True,applied_via:env,cuda_graphs:False")
    assert mem.undo(rec) == ["expandable_segments"] and env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
    env2 = {}
    rec = mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ=env2), base="fast")
    assert env2 == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} and mem.undo(rec) and env2 == {}
    # the refusals, each by name
    for env, name in (({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}, "conf"), ({"PYTORCH_CUDA_ALLOC_CONF": "junk"}, "conf")):
        with pytest.raises(mem.Refused) as e:
            mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ=env), base="fast")
        assert e.value.record.refused[0].precondition == name and env.get("PYTORCH_CUDA_ALLOC_CONF") != "expandable_segments:True"
    with pytest.raises(mem.Refused) as e:
        mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ={}, graphs=True), base="fast")
    assert e.value.record.refused[0].precondition == "graphs" and "switches={'expandable_segments': False}" in e.value.record.refused[0].reason
    rec = mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ={}), base="fast", switches={"expandable_segments": False})
    assert rec.levers == () and rec.off_by_flag == ("expandable_segments",) and rec.mode_line() == "big:"


@no_cuda_path
def test_cache_release_lever_and_release_points():
    env = {"ACME_BIG_CACHE_RELEASE_POLICY": "always"}                                   # never read: the kit's ctx.settings value below decides
    ctx = mem.Ctx(prefix="ACME", tag="t", environ=env, settings={"cache_release": {"policy": "per_stage"}})
    rec = mem.apply(["cache_release"], ctx, base="fast")
    assert rec.applied[0].settings == {"policy": "per_stage"} and rec.settings[0]["source"] == "ctx.settings"
    rec.unit_begin("u")
    assert allocator.release(ctx, "stage") is False                                          # no torch: a named fallback, never silent
    assert rec.units["u"].fallback == {"cache_release": "torch.cuda unavailable: nothing released"}
    rec.unit_end()
    with pytest.raises(mem.Refused) as e:
        mem.apply(["cache_release"], mem.Ctx(prefix="ACME", tag="t", settings={"cache_release": {"policy": "always"}}), base="fast")
    assert e.value.record.refused[0].precondition == "policy"
    ctx = mem.Ctx(prefix="ACME", tag="t", environ={})
    rec = mem.apply(["cache_release"], ctx, base="fast")
    rec.unit_begin("u")
    assert allocator.release(ctx, "stage") is False and rec.units["u"].skipped == {"cache_release": "policy per_unit: no release at stage"}
    assert allocator.release(mem.Ctx(prefix="ACME", tag="t", environ={}), "unit") is False  # lever not applied: nothing recorded


def test_jax_export_never_overrides_a_pin():
    if allocator.jax_state()["backends_initialized"]:                                     # export-before-init path: not reachable once a backend lives
        pytest.skip("jax backends are already initialised in this interpreter (jax %s; a GPU suite ran first): the export-before-init path is "
                    "exercised on the CPU box" % allocator.jax_state()["version"])
    env = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    ctx = mem.Ctx(prefix="ACME", tag="t", framework="jax", environ=env)
    ctx.record = mem.AppliedRecord(prefix="ACME", base="exact", allocator={"found": {}, "writes": [], "settings": {}})
    out = allocator.jax_export(ctx, "fx_sub", XLA_PYTHON_CLIENT_MEM_FRACTION="0.95", XLA_PYTHON_CLIENT_PREALLOCATE="FALSE")
    assert out == {"XLA_PYTHON_CLIENT_MEM_FRACTION": {"value": "0.95", "state": "exported"},
                   "XLA_PYTHON_CLIENT_PREALLOCATE": {"value": "false", "state": "kept"}}
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.95" and ctx.record.allocator["writes"][0]["name"] == "XLA_PYTHON_CLIENT_MEM_FRACTION"
    with pytest.raises(registry.RefusalError) as e:
        allocator.jax_export(ctx, "fx_sub", XLA_PYTHON_CLIENT_PREALLOCATE="true")
    assert e.value.refusal.precondition == "xla.XLA_PYTHON_CLIENT_PREALLOCATE" and env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_torch_runtime_path_when_cuda_is_initialised(monkeypatch):
    """A stand-in torch with CUDA initialised: the setting goes through the runtime API and the record says so."""
    calls = []

    class Mem:
        @staticmethod
        def _set_allocator_settings(s):
            calls.append(s)

    class Cuda:
        memory = Mem()

        @staticmethod
        def is_initialized():
            return True

    class Torch:
        __version__ = "0.0-test"
        cuda = Cuda()

    monkeypatch.setitem(sys.modules, "torch", Torch())
    try:
        env = {"PYTORCH_CUDA_ALLOC_CONF": "a:1"}
        rec = mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ=env), base="fast")
        assert rec.applied[0].settings["applied_via"] == "runtime_api" and calls == ["a:1,expandable_segments:True"]
        assert rec.applied[0].notes[0].startswith("CUDA was initialised before big applied")
        mem.undo(rec)
        assert calls[-1] == "a:1" and env == {"PYTORCH_CUDA_ALLOC_CONF": "a:1"}
        Cuda.memory = None                                                                   # no runtime API: refusal by name
        with pytest.raises(mem.Refused) as e:
            mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ={}), base="fast")
        assert e.value.record.refused[0].precondition == "cuda_state"
    finally:
        sys.modules.pop("torch", None)


def test_expandable_segments_refuses_a_read_only_environ_and_a_missing_torch(monkeypatch):
    import types
    ro = types.MappingProxyType({})
    if _TORCH_IMPORTABLE:                                   # the environ precondition is reached only past the torch one
        with pytest.raises(mem.Refused) as e:
            mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ=ro), base="fast")
        assert e.value.record.refused[0].precondition == "environ"
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(allocator.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(mem.Refused) as e:
        mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ={}), base="fast")
    assert e.value.record.refused[0].precondition == "torch"


def test_runtime_undo_without_a_prior_conf_turns_the_segments_off(monkeypatch):
    calls = []

    class Mem:
        @staticmethod
        def _set_allocator_settings(s):
            calls.append(s)

    class Cuda:
        memory = Mem()

        @staticmethod
        def is_initialized():
            return True

    class Torch:
        __version__ = "0.0-test"
        cuda = Cuda()

    monkeypatch.setitem(sys.modules, "torch", Torch())
    env = {}
    rec = mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ=env), base="fast")
    assert calls == ["expandable_segments:True"] and env == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    mem.undo(rec)
    assert calls[-1] == "expandable_segments:False" and env == {}


def test_jax_export_refuses_a_read_only_environ():
    if allocator.jax_state()["backends_initialized"]:                                     # export-before-init path: not reachable once a backend lives
        pytest.skip("jax backends are already initialised in this interpreter (jax %s; a GPU suite ran first): the export-before-init path is "
                    "exercised on the CPU box" % allocator.jax_state()["version"])
    import types
    ctx = mem.Ctx(prefix="ACME", tag="t", framework="jax", environ=types.MappingProxyType({}))
    with pytest.raises(registry.RefusalError) as e:
        allocator.jax_export(ctx, "fx_sub", XLA_PYTHON_CLIENT_MEM_FRACTION="0.9")
    assert e.value.refusal.precondition == "environ"


def test_compose_big_explicit_base_without_the_rule():
    t = modes.ModeTable(("off", "mem64", "mem128"), "off")
    t2, line = mem.compose_big(t, base="mem64", levers=("a_lever",))
    assert line.base == "mem64" and line.base_rule == "override" and "big" in t2.modes


def test_expandable_segments_is_verified_by_read_back(monkeypatch):
    """The export is checked once CUDA is up: the allocator's own settings are read back; a setting not in force is a named fallback."""
    state = {"init": False, "settings": {"expandable_segments": True}}

    class Mem:
        @staticmethod
        def _snapshot():
            return {"segments": [], "allocator_settings": dict(state["settings"])}

        @staticmethod
        def _set_allocator_settings(s):
            pass

    class Cuda:
        memory = Mem()

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return state["init"]

    class Torch:
        __version__ = "0.0-test"
        cuda = Cuda()

    monkeypatch.setitem(sys.modules, "torch", Torch())
    env = {}
    rec = mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ=env), base="fast")
    a = rec.get("expandable_segments")
    assert a.verify is allocator.expandable_in_force and a.verified is None and a.scope == "process"
    assert rec.units == {}                                                                   # not marked at apply: checked later
    rec.unit_begin("u1")                                                                     # CUDA not up yet: the check stays pending
    assert a.verified == {"ok": None, "state": "pending", "reason": "CUDA not initialised yet: the setting was never exercised"}
    state["init"] = True                                                                     # CUDA comes up INSIDE the first unit (N1)
    rec.unit_end()                                                                           # the unit boundary settles it
    assert a.verified == {"ok": True, "state": "final", "reason": None}
    c = rec.census()
    assert c["units"]["process"]["ran"] == ["expandable_segments"] and c["ok"]
    assert rec.exit_gate(0)["exit_code"] == 0
    # not in force: the export never reached the allocator
    state["settings"] = {"expandable_segments": False}
    rec2 = mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ={}), base="fast")
    v = rec2.exit_gate(0)                                                                    # the gate runs a pending check
    assert v["exit_code"] == 3 and v["partial"] == ["expandable_segments"]
    assert rec2.get("expandable_segments").verified["reason"].startswith("allocator_settings.expandable_segments is False")
    assert rec2.get("expandable_segments").verified["state"] == "final"
    assert rec2.get("expandable_segments").as_dict()["verified"]["ok"] is False
    # never exercised: CUDA never initialised
    state["init"] = False
    rec3 = mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t", environ={}), base="fast")
    rec3.unit_begin("u1"); rec3.unit_end()                                                   # still pending through the unit
    assert rec3.get("expandable_segments").verified["state"] == "pending" and rec3.census()["ok"]
    v = rec3.exit_gate(0)                                                                    # the gate finalises: never checkable
    assert v["exit_code"] == 3 and v["reasons"]["expandable_segments"] == ("process: not in force: never verifiable: CUDA not initialised yet: "
                                                                           "the setting was never exercised")


def test_jax_export_refuses_after_the_backend_is_up(monkeypatch):
    import types
    xb = types.SimpleNamespace(backends_are_initialized=lambda: True)
    monkeypatch.setitem(sys.modules, "jax", types.SimpleNamespace(__version__="0.0-test"))
    monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", xb)
    ctx = mem.Ctx(prefix="ACME", tag="t", framework="jax", environ={})
    with pytest.raises(registry.RefusalError) as e:
        allocator.jax_export(ctx, "fx_sub", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    assert e.value.refusal.precondition == "backend_initialized" and ctx.environ == {}
    assert allocator.jax_state() == {"imported": True, "version": "0.0-test", "backends_initialized": True}
    xb.backends_are_initialized = lambda: False
    assert allocator.jax_export(ctx, "fx_sub", XLA_PYTHON_CLIENT_PREALLOCATE="false") == {
        "XLA_PYTHON_CLIENT_PREALLOCATE": {"value": "false", "state": "exported"}}


@needs_importable_torch
def test_line_selector_and_auto_policy(monkeypatch):
    def policy(ctx):
        n = int(ctx.extra.get("n_tokens", 0))
        return ("xl", f"n_tokens={n} >= 4000", {"threshold": 4000, "n_tokens": n}) if n >= 4000 else ("default", f"n_tokens={n} < 4000")

    t, line = mem.compose_big(FAST, levers=("expandable_segments",), tier="bitwise", cost_note="exp7/row-3",
                                lines={"xl": {"levers": ("expandable_segments", "cache_release"), "tier": "bitwise", "cost_note": "unmeasured"}},
                                auto=policy)
    assert line.lines == {"default": {"levers": ("expandable_segments",), "tier": "bitwise", "cost_note": "exp7/row-3"},
                          "xl": {"levers": ("expandable_segments", "cache_release"), "tier": "bitwise", "cost_note": "unmeasured"}}
    assert line.as_dict()["lines"]["xl"] == {"levers": ["expandable_segments", "cache_release"], "tier": "bitwise", "cost_note": "unmeasured"}
    _, bare = mem.compose_big(FAST, levers=("expandable_segments",), lines={"xl": ("cache_release",)})
    assert bare.lines["xl"] == {"levers": ("cache_release",), "tier": None, "cost_note": "unstated"}                # never filled in silently
    with pytest.raises(ValueError):
        mem.compose_big(FAST, levers=("expandable_segments",), lines={"xl": {"levers": ("cache_release",), "tier": "close"}})
    with pytest.raises(ValueError):
        mem.compose_big(FAST, levers=("expandable_segments",), lines={"xl": {"levers": ("cache_release",), "speed": 1}})
    # unset: the default line, recorded
    levers, pol, ref = mem.select_line(line, mem.Ctx(prefix="ACME", tag="t", environ={}))
    assert levers == ("expandable_segments",) and ref is None and pol["selector"] == "default" and pol["chosen"] == "default"
    # by name
    levers, pol, ref = mem.select_line(line, mem.Ctx(prefix="ACME", tag="t"), "xl")
    assert levers == ("expandable_segments", "cache_release") and pol["reason"] == "selected by name"
    # auto through the policy, with its details
    ctx = mem.Ctx(prefix="ACME", tag="t", extra={"n_tokens": 5000})
    levers, pol, ref = mem.select_line(line, ctx, "auto")
    assert levers == ("expandable_segments", "cache_release") and pol["chosen"] == "xl" and pol["reason"] == "n_tokens=5000 >= 4000"
    assert pol["details"] == {"threshold": 4000, "n_tokens": 5000} and pol["declared"]["xl"]["cost_note"] == "unmeasured"
    # unknown name / auto without a policy / a policy naming an undeclared line: refusals naming the line selector
    levers, pol, ref = mem.select_line(line, mem.Ctx(prefix="ACME", tag="t"), "huge")
    assert levers == () and ref.precondition == "line" and "names no declared line" in ref.reason
    # the former selector VARIABLE has no effect: the default line
    levers, pol, ref = mem.select_line(line, mem.Ctx(prefix="ACME", tag="t", environ={"ACME_BIG_LINE": "xl"}))
    assert levers == ("expandable_segments",) and ref is None and pol["selector"] == "default"
    _, plain = mem.compose_big(FAST, levers=("expandable_segments",))
    _, _, ref = mem.select_line(plain, mem.Ctx(prefix="ACME", tag="t"), "auto")
    assert ref.precondition == "line" and "declares no auto policy" in ref.reason
    _, bad = mem.compose_big(FAST, levers=("expandable_segments",), auto=lambda ctx: ("ghost", "r"), auto_default=True)
    _, pol, ref = mem.select_line(bad, mem.Ctx(prefix="ACME", tag="t", environ={}))
    assert pol["selector"] == "auto" and ref.precondition == "line" and "does not declare" in ref.reason
    # through apply(): the policy rides the record, the ACTIVE line and the manifest block; a refused selector gates the exit
    rec = mem.apply(line, mem.Ctx(prefix="ACME", tag="t", extra={"n_tokens": 5000}), line="auto")
    assert rec.levers == ("expandable_segments", "cache_release") and rec.line_policy["chosen"] == "xl"
    assert " line=xl(auto: n_tokens=5000 >= 4000) base=fast " in rec.active_line("t") and rec.manifest_block()["line_policy"]["chosen"] == "xl"
    assert rec.manifest_block()["line_policy"]["declared"]["default"] == {"levers": ["expandable_segments"], "tier": "bitwise", "cost_note": "exp7/row-3"}
    assert rec.tier_check() == {"declared": "bitwise", "composed": "bitwise", "ok": True} and rec.manifest_block()["tier_check"]["ok"]
    assert rec.flags == {} and rec.line_policy["selector"] == "auto"
    with pytest.raises(mem.Refused) as e:
        mem.apply(line, mem.Ctx(prefix="ACME", tag="t"), line="huge")
    rec = e.value.record
    assert rec.levers == () and rec.refused[0].precondition == "line" and rec.exit_gate(0)["exit_code"] == 3
    with pytest.raises(TypeError):
        mem.apply(["expandable_segments"], mem.Ctx(prefix="ACME", tag="t"), line="xl")           # a plain lever sequence declares no lines
    with pytest.raises(ValueError):
        mem.compose_big(FAST, levers=("a_lever",), lines={"default": ("b_lever",)})
    with pytest.raises(ValueError):
        mem.compose_big(FAST, levers=("a_lever",), lines={"auto": ("b_lever",)})
