"""opt_core.mem: apply() end to end with synthetic levers, the applied record, the lines, the census, the exit gate, the manifest block."""
import json
import os
import subprocess
import sys

import pytest

from opt_core import manifest, mem, modes, report
from opt_core.mem import registry

TABLE = modes.ModeTable(("off", "exact", "fast"), "fast")


class Trunk:
    """A synthetic module with two sites the fixture levers patch."""

    def __init__(self):
        self.pair = "pair"
        self.chunk = None

    def transition(self, x):
        return x


def _chunk_applies(ctx):
    ctx.require("fx_chunk", "module")
    return None


@pytest.fixture
def levers():
    """Two synthetic levers: ``fx_chunk`` (needs hook ``module``; setting ``chunk``; undoable) and ``fx_park`` (refuses by name
    without hook ``tensor``; apply raises a RefusalError when ``tensor`` is ``"bad"``)."""

    @registry.register("fx_chunk", family="chunk", exact="band", exact_reason="chunked rows re-order the row reduction",
                       applies=_chunk_applies, preconditions=("hooks.module",), settings=("chunk",))
    def fx_chunk(ctx):
        mod = ctx.require("fx_chunk", "module")["module"]
        chunk = ctx.setting("fx_chunk", "chunk", 256, cast=int)
        before = mod.chunk
        mod.chunk = chunk

        def undo():
            mod.chunk = before

        rec = ctx.record
        narrow = chunk >= 4096
        return registry.Applied(lever="fx_chunk", settings={"chunk": chunk}, sites=("Trunk.transition",),
                                exact="bitwise" if narrow else None,
                                exact_reason="the chunk covers the full dimension: no re-ordered reduction" if narrow else None,
                                narrowed_by="identity:fx-row-1" if narrow else None,
                                undo=undo, notes=[] if rec is None else [f"record attached: {rec.mode}"])

    def park_applies(ctx):
        if ctx.hook("fx_park", "tensor") is None:
            return registry.refuse("fx_park", "hooks.tensor", "no tensor named")
        return None

    @registry.register("fx_park", family="offload", exact="bitwise", exact_reason="residency only", applies=park_applies,
                       preconditions=("hooks.tensor",))
    def fx_park(ctx):
        if ctx.hook("fx_park", "tensor") == "bad":
            raise registry.RefusalError(registry.refuse("fx_park", "pinned", "pinning failed by name"))
        return registry.Applied(lever="fx_park", sites=("Trunk.pair",))

    @registry.register("fx_jax", family="jax", exact="measured", exact_reason="sub-batching changes the reduction tree",
                       applies=lambda ctx: None, frameworks=("jax",))
    def fx_jax(ctx):
        return registry.Applied(lever="fx_jax")

    yield
    for n in ("fx_chunk", "fx_park", "fx_jax"):
        registry.unregister(n)


def line(*levers, **kw):
    return mem.compose_big(TABLE, levers=levers, **kw)[1]


def test_compose_exact():
    assert mem.compose_exact([]) == "bitwise" and mem.compose_exact(["bitwise", "band"]) == "band"
    assert mem.compose_exact(["band", "measured", "bitwise"]) == "measured"
    with pytest.raises(ValueError):
        mem.compose_exact(["close"])


def test_apply_strict_refuses_by_name_and_records_everything(levers):
    trunk = Trunk()
    ctx = mem.Ctx(prefix="ACME", tag="acme-opt", hooks={"fx_chunk": {"module": trunk}}, environ={})
    with pytest.raises(mem.Refused) as e:
        mem.apply(line("fx_chunk", "fx_park"), ctx)
    rec = e.value.record
    assert rec.levers == ("fx_chunk",) and rec.refused_names == ("fx_park",) and rec.base == "fast"
    assert rec.refused[0].precondition == "hooks.tensor" and "fx_park: hooks.tensor: no tensor named" in str(e.value)
    assert [p["failed"] for p in rec.preconditions] == [None, "hooks.tensor"]
    assert rec.mode_line() == "big:fx_chunk" and trunk.chunk == 256
    assert rec.not_active_line("acme-opt") == ("[acme-opt] NOT ACTIVE: big refused — fx_park: hooks.tensor: no tensor named "
                                               "(mode=big base=fast)")
    assert rec.settings == [{"lever": "fx_chunk", "key": "chunk", "value": 256, "source": "default"}]
    assert rec.applied[0].notes == ["record attached: big"] and ctx.record is rec
    assert "allocator" in rec.discovery["loaded"] and set(rec.discovery) == {"loaded", "absent", "broken"}


def test_apply_lines_flags_and_undo(levers):
    trunk = Trunk()
    env = {"ACME_BIG_FX_CHUNK_CHUNK": "1", "ACME_BIG_FX_PARK": "1", "ACME_BIG_ALLOW_PARTIAL": "1"}       # the environment is never read: no effect below
    ctx = mem.Ctx(prefix="ACME", tag="acme-opt", hooks={"fx_chunk": {"module": trunk}}, settings={"fx_chunk": {"chunk": "8192"}}, environ=env, extra={"kit": "v1"})
    rec = mem.apply(line("fx_chunk", "fx_park", drop=("graph_sampler",)), ctx, switches={"fx_park": False})
    assert rec.levers == ("fx_chunk",) and rec.off_by_flag == ("fx_park",) and rec.expected == ("fx_chunk",)
    assert rec.exact == "bitwise" and rec.exact_per_lever == {"fx_chunk": "bitwise"}          # the lever narrowed its label
    assert rec.settings[-1]["source"] == "ctx.settings" and trunk.chunk == 8192 and rec.flags == {"fx_park": False} and not rec.allow_partial
    assert rec.active_line("acme-opt", n_gpu=1) == ("[acme-opt] ACTIVE mode=big:fx_chunk base=fast drop=graph_sampler exact=bitwise "
                                                   "refused=none off=fx_park on=none allocator=none n_gpu=1")
    assert rec.drop == ("graph_sampler",) and rec.base_rule == "fast-else-exact" and rec.manifest_block()["drop"] == ["graph_sampler"]
    assert rec.extra == {"kit": "v1"} and rec.allocator["found"]["framework"] == "torch" and rec.allocator["settings"] == {}
    assert mem.undo(rec) == ["fx_chunk"] and trunk.chunk is None
    # a switch turns a registered lever on that the line does not carry; framework mismatch is a refusal by name
    ctx = mem.Ctx(prefix="ACME", tag="t", hooks={"fx_chunk": {"module": trunk}, "fx_park": {"tensor": "z"}})
    rec = mem.apply(line("fx_chunk", "fx_park"), ctx, strict=False, switches={"fx_jax": True})
    assert rec.levers == ("fx_chunk", "fx_park") and rec.on_by_flag == ("fx_jax",) and rec.refused[0].precondition == "framework"
    assert rec.exact == "band"
    # a RefusalError inside apply is a recorded refusal; a plain exception propagates
    ctx = mem.Ctx(prefix="ACME", tag="t", hooks={"fx_chunk": {"module": trunk}, "fx_park": {"tensor": "bad"}}, environ={})
    rec = mem.apply(line("fx_park"), ctx, strict=False)
    assert rec.refused[0].precondition == "pinned" and rec.preconditions[-1]["failed"] == "pinned"
    ctx = mem.Ctx(prefix="ACME", tag="t", hooks={"fx_chunk": {"module": None}}, environ={})
    rec = mem.apply(["fx_chunk"], ctx, base="exact", strict=False)
    assert rec.refused[0].precondition == "hooks.module" and rec.base == "exact"
    registry.register("fx_boom", family="setting", exact="bitwise", exact_reason="r", applies=lambda ctx: None)(
        lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    registry.register("fx_wrong", family="setting", exact="bitwise", exact_reason="r", applies=lambda ctx: None)(lambda ctx: {"x": 1})
    try:
        with pytest.raises(RuntimeError):                                                   # a broken lever is a defect: propagates
            mem.apply(["fx_boom"], mem.Ctx(prefix="ACME", tag="t", environ={}))
        with pytest.raises(TypeError):
            mem.apply(["fx_wrong"], mem.Ctx(prefix="ACME", tag="t", environ={}))
    finally:
        registry.unregister("fx_boom")
        registry.unregister("fx_wrong")


def test_census_and_exit_gate(levers):
    trunk = Trunk()
    ctx = mem.Ctx(prefix="ACME", tag="acme-opt", hooks={"fx_chunk": {"module": trunk}, "fx_park": {"tensor": "z"}}, environ={})
    rec = mem.apply(line("fx_chunk", "fx_park"), ctx)
    # no unit observed: every expected lever is partial (fail-closed) unless the kit says no units were expected
    v = rec.exit_gate(0)
    assert v["exit_code"] == report.EXIT_NOT_ACTIVE and v["partial"] == ["fx_chunk", "fx_park"] and v["reasons"]["fx_chunk"] == mem.record.NO_UNIT
    assert rec.exit_gate(0, expect_units=False)["exit_code"] == 0
    # units: ran / skipped are accounted, fallback and absent are partial
    rec.unit_begin("a"); rec.mark("fx_chunk"); rec.mark("fx_park"); rec.unit_end()
    rec.unit_begin("b"); rec.mark("fx_chunk"); rec.skip("fx_park", "nothing to park at this size"); rec.unit_end()
    rec.unit_begin("c"); rec.mark("fx_chunk"); rec.fallback("fx_park", "pinned alloc failed: pageable"); rec.unit_end()
    rec.unit_begin("d"); rec.mark("fx_park"); rec.unit_end()
    rec.unit_begin("e", expected=("fx_park",)); rec.mark("fx_park"); rec.unit_end()
    c = rec.census()
    assert c["n_units"] == 5 and c["partial_units"] == ["c", "d"] and c["partial"] == ["fx_park", "fx_chunk"]
    assert c["units"]["b"]["skipped"] == {"fx_park": "nothing to park at this size"} and c["units"]["d"]["absent"] == ["fx_chunk"]
    assert c["units"]["e"]["expected"] == ["fx_park"] and not c["units"]["e"]["partial"]
    assert c["per_lever"] == {"fx_chunk": {"ran": 3, "skipped": 0, "fallback": 0, "absent": 1, "unexpected": 0},
                              "fx_park": {"ran": 3, "skipped": 1, "fallback": 1, "absent": 0, "unexpected": 0}}
    v = rec.exit_gate(0)
    assert v["exit_code"] == 3 and v["partial"] == ["fx_park", "fx_chunk"] and v["opt_out"] == "--allow-partial" and v["allow_partial_source"] is None
    assert v["reasons"] == {"fx_park": "c: pinned alloc failed: pageable", "fx_chunk": "d: never marked"}
    assert rec.exit_line("acme-opt") == ("[acme-opt] NOT ACTIVE: big partial — levers=fx_park,fx_chunk (fx_park: c: pinned alloc "
                                         "failed: pageable; fx_chunk: d: never marked); exit 3 (--allow-partial records and proceeds)")
    assert rec.exit_gate(5)["exit_code"] == 5 and rec.exit_gate(0, incomplete="2 of 3")["exit_code"] == 1
    # the former opt-out VARIABLE has no effect: set in the environment, the gate still fails closed
    ctx = mem.Ctx(prefix="ACME", tag="acme-opt", hooks=ctx.hooks, environ={"ACME_BIG_ALLOW_PARTIAL": "1"})
    rec = mem.apply(line("fx_chunk", "fx_park"), ctx)
    rec.unit_begin("a"); rec.mark("fx_chunk"); rec.unit_end()
    v = rec.exit_gate(0)
    assert v["exit_code"] == 3 and not v["allow_partial"] and v["partial"] == ["fx_park"] and v["allow_partial_source"] is None
    assert rec.exit_line("acme-opt") == "[acme-opt] NOT ACTIVE: big partial — levers=fx_park (fx_park: a: never marked); exit 3 (--allow-partial records and proceeds)"
    # the kit's opt-out passed at selection: recorded, exit 0, the allowed line (the kit's own spelling of its flag is what the line names)
    ctx = mem.Ctx(prefix="ACME", tag="acme-opt", hooks=ctx.hooks, opt_out="--allow-partial")
    rec = mem.apply(line("fx_chunk", "fx_park"), ctx, allow_partial=True)
    rec.unit_begin("a"); rec.mark("fx_chunk"); rec.unit_end()
    v = rec.exit_gate(0)
    assert v["exit_code"] == 0 and v["allow_partial"] and v["partial"] == ["fx_park"] and v["allow_partial_source"] == "kit"
    assert rec.exit_line("acme-opt") == "[acme-opt] PARTIAL allowed: levers=fx_park (fx_park: a: never marked) (--allow-partial, recorded)"
    assert rec.manifest_block()["allow_partial"] == {"opt_out": "--allow-partial", "value": True}
    # a refused line lever (non-strict) is partial at the gate too
    ctx = mem.Ctx(prefix="ACME", tag="t", hooks={"fx_chunk": {"module": trunk}}, environ={})
    rec = mem.apply(line("fx_chunk", "fx_park"), ctx, strict=False)
    rec.unit_begin("a"); rec.mark("fx_chunk"); rec.unit_end()
    v = rec.exit_gate(0)
    assert v["exit_code"] == 3 and v["partial"] == ["fx_park"] and v["reasons"]["fx_park"].startswith("refused: hooks.tensor")
    # unit bookkeeping refuses misuse by name
    with pytest.raises(ValueError):
        rec.unit_begin("a")
    with pytest.raises(ValueError):
        rec.unit_end("zz")
    with pytest.raises(ValueError):
        rec.mark("fx_chunk")                                       # a unit-scope mark after the last unit closed
    rec.unit_begin("b")
    with pytest.raises(ValueError):
        rec.unit_begin("c")                                        # over an open unit
    with pytest.raises(ValueError):
        rec.unit_begin(mem.record.PROCESS_UNIT)
    rec.unit_end()
    assert mem.census(rec)["n_units"] == 2


def test_process_scope_levers_and_unexpected_marks(levers):
    registry.register("fx_proc", family="setting", exact="bitwise", exact_reason="r", applies=lambda ctx: None, scope="process")(
        lambda ctx: registry.Applied(lever="fx_proc"))
    try:
        trunk = Trunk()
        ctx = mem.Ctx(prefix="ACME", tag="t", hooks={"fx_chunk": {"module": trunk}})
        rec = mem.apply(line("fx_proc", "fx_chunk", "fx_park"), ctx, switches={"fx_park": False})
        assert rec.scopes == {"fx_proc": "process", "fx_chunk": "unit"}
        assert rec.units[mem.record.PROCESS_UNIT].ran == ["fx_proc"]                      # marked at apply
        rec.unit_begin("a"); rec.mark("fx_chunk"); rec.unit_end()
        rec.unit_begin("b"); rec.mark("fx_chunk"); rec.mark("fx_park"); rec.unit_end()   # fx_park is off by flag: a patch that stayed
        c = rec.census()
        assert c["units"][mem.record.PROCESS_UNIT]["expected"] == ["fx_proc"] and not c["units"][mem.record.PROCESS_UNIT]["partial"]
        assert c["units"]["a"]["expected"] == ["fx_chunk"] and not c["units"]["a"]["partial"]
        assert c["units"]["b"]["unexpected"] == ["fx_park"] and c["units"]["b"]["partial"] == ["fx_park"]
        assert c["per_lever"]["fx_park"]["unexpected"] == 1
        v = rec.exit_gate(0)
        assert v["exit_code"] == 3 and v["partial"] == ["fx_park"] and v["reasons"]["fx_park"] == "b: ran while off / refused"
        # the kit's own parsed --allow-partial passed to the gate opens it and is recorded (the record keeps the value that decided)
        v = rec.exit_gate(0, allow_partial=True)
        assert v["exit_code"] == 0 and v["allow_partial_source"] == "kit" and rec.allow_partial is True and rec.manifest_block()["allow_partial"]["value"] is True
        assert rec.exit_gate(0, allow_partial=False)["exit_code"] == 3 and rec.allow_partial is False
        # a process-scope lever alone needs no kit unit
        rec2 = mem.apply(line("fx_proc"), mem.Ctx(prefix="ACME", tag="t", environ={}))
        assert rec2.exit_gate(0)["exit_code"] == 0 and rec2.census()["units"][mem.record.PROCESS_UNIT]["expected"] == ["fx_proc"]
        # every refusal gates, deduplicated, including the selection's
        rec3 = mem.apply(line("fx_proc"), mem.Ctx(prefix="ACME", tag="t"), strict=False, switches={"nope": True})
        v = rec3.exit_gate(0)
        assert v["exit_code"] == 3 and v["partial"] == ["big"] and v["reasons"]["big"].startswith("refused: switch.nope")
    finally:
        registry.unregister("fx_proc")


def test_add_label_rule_and_guards(levers):
    rec = mem.AppliedRecord(prefix="ACME", base="fast", expected=("fx_chunk",))
    with pytest.raises(ValueError, match="wider than the declared"):
        rec.add(registry.Applied(lever="fx_chunk", exact="measured", exact_reason="r"), "band", "r")   # widening: refused outright
    with pytest.raises(ValueError, match="narrows the declared"):
        rec.add(registry.Applied(lever="fx_chunk", exact="bitwise", exact_reason="r"), "band", "r")    # narrowing without the equality id
    a = rec.add(registry.Applied(lever="fx_chunk", exact="bitwise", narrowed_by="identity:row-7"), "band", "r")
    assert a.declared_exact == "band" and a.exact == "bitwise" and rec.exact == "bitwise" and a.narrowed_by == "identity:row-7"
    assert rec.notes == ["fx_chunk: label_narrowed band -> bitwise by identity record identity:row-7"]
    with pytest.raises(ValueError):
        rec.add(registry.Applied(lever="fx_chunk"), "band", "r")                              # twice
    rec.refuse(registry.refuse("fx_park", "hooks.tensor", "none"))
    with pytest.raises(ValueError):
        rec.add(registry.Applied(lever="fx_park"), "bitwise", "r")                            # refused


def test_manifest_block_round_trip(levers, tmp_path):
    trunk = Trunk()
    ctx = mem.Ctx(prefix="ACME", tag="acme-opt", hooks={"fx_chunk": {"module": trunk}, "fx_park": {"tensor": "z"}}, environ={})
    rec = mem.apply(line("fx_chunk", "fx_park"), ctx)
    rec.unit_begin("a"); rec.mark("fx_chunk"); rec.mark("fx_park"); rec.unit_end()
    rec.exit_gate(0)
    kit = rec.attach({"levers": ["x"]})
    doc = manifest.build(package="acme_opt", package_version="1", schema="acme/1", mode="big", report={}, route="opt", kit=kit)
    path = manifest.write(str(tmp_path / manifest.FILENAME), doc)
    back = manifest.read(path)
    b = back["kit"]["big"]
    assert b["mode_line"] == "big:fx_chunk,fx_park" and b["exact"] == "band" and b["exact_per_lever"] == {"fx_chunk": "band", "fx_park": "bitwise"}
    assert b["levers"][0] == {"lever": "fx_chunk", "settings": {"chunk": 256}, "sites": ["Trunk.transition"], "exact": "band",
                              "exact_reason": "chunked rows re-order the row reduction", "declared_exact": "band", "scope": "unit",
                              "notes": ["record attached: big"], "undoable": True, "detail": {}, "verified": None, "narrowed_by": None}
    assert b["census"]["ok"] and b["exit"]["exit_code"] == 0 and b["allow_partial"] == {"opt_out": "--allow-partial", "value": False}
    assert set(b) == {"mode", "mode_line", "base", "base_rule", "drop", "line_policy", "tier_check", "line", "expected", "levers", "exact", "exact_per_lever", "refused", "preconditions",
                      "settings", "flags", "off_by_flag", "on_by_flag", "allow_partial", "allocator", "discovery", "census", "exit",
                      "notes", "extra"}
    assert json.dumps(rec.as_dict())


def test_import_is_light_and_torch_free():
    code = ("import sys; import opt_core.mem as m; assert 'torch' not in sys.modules; "
            "print(sorted(m.LEVERS), m.BIG, m.LEVER_MODULES)")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.startswith("[] big")                              # importing the package registers nothing: discover() does


def test_sink_detail_and_refusal_bridge(levers):
    trunk = Trunk()
    ctx = mem.Ctx(prefix="ACME", tag="t", hooks={"fx_chunk": {"module": trunk}, "fx_park": {"tensor": "z"}}, environ={})
    rec = mem.apply(line("fx_chunk", "fx_park"), ctx)
    sink = rec.sink("fx_chunk")                                                              # the bound per-call sink a chunked op takes
    rec.unit_begin("a")
    sink({"op": "pair_transition", "exact": "bitwise", "n_chunks": 4, "rows": 512})
    sink({"op": "pair_transition", "decision": "passthrough", "reason": "fp32 site", "n_chunks": 1})
    rec.mark("fx_park")
    rec.unit_end()
    rec.unit_begin("b")
    sink({"op": "triangle_attention", "decision": "fallback", "reason": "no mask hook"})
    rec.mark("fx_park")
    e = rec.call("fx_park", "park", "bitwise", "pinned copy", rows=4)
    assert e["detail"] == {"rows": 4} and rec.units["b"].events[-1] is e
    rec.unit_end()
    c = rec.census()
    assert c["units"]["a"]["ran"] == ["fx_chunk", "fx_park"] and c["units"]["a"]["fallback"] == {"fx_chunk": "passthrough: fp32 site"}
    assert c["units"]["a"]["partial"] == ["fx_chunk"] and c["units"]["b"]["partial"] == ["fx_chunk"]   # a passthrough is a fallback
    assert [e["kind"] for e in rec.units["a"].events] == ["ran", "call", "fallback", "call", "ran"]
    assert rec.get("fx_chunk").detail["calls"] == {"pair_transition": {"bitwise": 1}}
    with pytest.raises(ValueError, match="wider than the lever"):
        rec.call("fx_chunk", "triangle_attention", "measured", "r", unit="b")                # a call never widens the lever's label
    assert rec.detail("fx_park", {"pinned_peak_gb": 1.5})["pinned_peak_gb"] == 1.5
    assert rec.get("fx_park").as_dict()["detail"] == {"pinned_peak_gb": 1.5, "calls": {"park": {"bitwise": 1}}}
    with pytest.raises(KeyError):
        rec.sink("fx_jax")

    class NamedRefusal(RuntimeError):
        def __init__(self, name, detail, **details):
            super().__init__(f"{name}: {detail}")
            self.name, self.details = name, details

    r = mem.refusal_from(NamedRefusal("not_a_tensor", "x is a list", what="x"), "fx_chunk")
    assert (r.lever, r.precondition, r.details) == ("fx_chunk", "not_a_tensor", {"what": "x"}) and "x is a list" in r.reason
    assert mem.refusal_from(ValueError("plain"), "fx_chunk").precondition == "ValueError"


def test_discover_records_a_module_whose_dependency_is_missing(monkeypatch):
    def broken(name):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    monkeypatch.setattr(registry.importlib, "import_module", broken)
    d = registry.discover(("zz_needs_torch",))
    assert d["broken"] == {"zz_needs_torch": "ModuleNotFoundError: No module named 'torch'"} and d["loaded"] == [] and d["absent"] == []
    assert registry.discover(("zz_needs_torch",))["broken"] == d["broken"]                    # cached verdict


def test_declared_tier_is_checked_against_the_composed_label(levers):
    _, line = mem.compose_big(TABLE, levers=("fx_chunk",), tier="bitwise", cost_note="unmeasured")
    rec = mem.apply(line, mem.Ctx(prefix="ACME", tag="t", hooks={"fx_chunk": {"module": Trunk()}}, environ={}))
    assert rec.exact == "band" and rec.tier_check() == {"declared": "bitwise", "composed": "band", "ok": False}
    assert rec.notes[-1] == "line default: declared tier bitwise but the applied levers compose to band"
    assert rec.declared_tier() == "bitwise"


def test_a_kit_that_delimits_no_unit_is_judged_on_the_process_unit(levers):
    trunk = Trunk()
    rec = mem.apply(line("fx_chunk"), mem.Ctx(prefix="ACME", tag="t", hooks={"fx_chunk": {"module": trunk}}, environ={}))
    assert rec.exit_gate(0)["exit_code"] == 3                                               # nothing observed anywhere
    rec.mark("fx_chunk")                                                                     # the process unit stands in
    v = rec.exit_gate(0)
    assert v["exit_code"] == 0 and rec.census()["units"][mem.record.PROCESS_UNIT]["expected"] == ["fx_chunk"]


def test_call_compares_against_the_label_fixed_at_add(levers):
    """B20: an in-place change of the Applied's label after add() cannot widen what call() / the mode label see."""
    trunk = Trunk()
    rec = mem.apply(line("fx_chunk"), mem.Ctx(prefix="ACME", tag="t", hooks={"fx_chunk": {"module": trunk}}, environ={}))
    a = rec.get("fx_chunk")
    assert a.exact == "band" and rec.exact == "band"
    a.exact = "measured"                                                                    # a mutation behind the record's back
    assert rec.exact == "band" and rec.exact_per_lever == {"fx_chunk": "band"} and rec.manifest_block()["levers"][0]["exact"] == "band"
    rec.unit_begin("a")
    with pytest.raises(ValueError, match="changed in place"):
        rec.call("fx_chunk", "pair_transition", "measured", "r")
    a.exact = "band"
    with pytest.raises(ValueError, match="wider than the lever"):
        rec.call("fx_chunk", "pair_transition", "measured", "r")
    e = rec.call("fx_chunk", "pair_transition", None, "r", decision="fallback")              # the decision that would exceed the label: a fallback
    assert e["decision"] == "fallback" and rec.units["a"].fallback == {"fx_chunk": "fallback: r"}
    rec.unit_end()



# ------------------------------------------------------------------------------------------------ the per-lever LEVER lines (one grammar)


def _ctx(**hooks):
    return mem.Ctx(prefix="FX", tag="fx-opt", framework="torch", hooks=hooks, environ={})


def test_lever_lines_render_every_named_lever_in_report_grammar(levers):
    from opt_core.mem import record as REC
    t = Trunk()
    rec = mem.apply(line("fx_chunk", "fx_park"), _ctx(fx_chunk={"module": t}), strict=False)
    lines = rec.lever_lines("fx-opt")
    assert lines[0] == "[fx-opt] LEVER name=fx_chunk state=on impl=%s origin=kit exact=band scope=unit chunk=256 sites=Trunk.transition" % (__name__,)
    assert lines[1] == "[fx-opt] LEVER name=fx_park state=skipped reason=hooks.tensor impl=%s origin=kit" % (__name__,)
    assert len(lines) == 2
    # a Lever's own strategy id rides the line and is validated by report.lever_line (canonical or LOCAL.<kit>.<name>)
    registry.unregister("fx_jax")

    @registry.register("fx_jax", family="jax", exact="measured", exact_reason="sub-batching changes the reduction tree",
                       applies=lambda ctx: None, frameworks=("torch",), strategy="LOCAL.fx.jax_thing")
    def fx_jax(ctx):
        return registry.Applied(lever="fx_jax")

    rec2 = mem.apply(line("fx_jax"), _ctx(), strict=False)
    assert rec2.lever_lines("fx-opt") == ["[fx-opt] LEVER name=fx_jax state=on impl=%s origin=kit strategy=LOCAL.fx.jax_thing exact=measured scope=unit" % (__name__,)]
    assert REC.strategy_of("fx_jax") == "LOCAL.fx.jax_thing" and REC.strategy_of("host_park") == "F7.pair_offload" and REC.strategy_of("nobody") is None
    with pytest.raises(registry.RegistryError):
        registry.Lever(name="fx_bad", family="jax", exact="measured", exact_reason="r", applies=lambda c: None, apply=lambda c: None, strategy="two words")


def test_lever_lines_name_flag_off_levers_once(levers):
    t = Trunk()
    ctx = mem.Ctx(prefix="FX", tag="fx-opt", framework="torch", hooks={"fx_chunk": {"module": t}, "fx_park": {"tensor": "x"}})
    rec = mem.apply(line("fx_chunk", "fx_park"), ctx, strict=False, switches={"fx_park": False})
    lines = rec.lever_lines("fx-opt")
    assert [ln.split()[2:4] for ln in lines] == [["name=fx_chunk", "state=on"], ["name=fx_park", "state=off"]], lines
    assert lines[1].endswith("state=off reason=flag impl=%s origin=kit" % (__name__,))


def test_every_lever_of_this_package_has_a_canonical_strategy_id():
    """Total accounting: each lever the package's modules register maps to a canonical id of the catalogue (opt_core.strategies) and
    report.lever_line accepts it."""
    from opt_core import strategies
    from opt_core.mem import record as REC
    registry.discover()
    missing = [n for n, lv in registry.LEVERS.items() if lv.module.startswith("opt_core.") and REC.strategy_of(n) is None]
    assert missing == [], missing
    for n, lv in registry.LEVERS.items():
        if lv.module.startswith("opt_core."):
            assert strategies.check(REC.strategy_of(n)) == REC.strategy_of(n), n
            report.lever_line("t", n, "off", reason="probe", impl=lv.module, origin="core", strategy=REC.strategy_of(n))
