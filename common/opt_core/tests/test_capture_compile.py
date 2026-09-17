"""CPU tests of opt_core.capture.compile (strategy F3.torch_compile): nodynamo on/off, CompileRecord.install over (owner, attr) targets
(once; already / missing counted; the cache_size_limit floor), dynamo's census through backend='eager' (no inductor toolchain needed), the
LEVER line grammar and the gate. Needs torch with dynamo (CPU); skips by name without it."""
from __future__ import annotations

import pytest

from opt_core import report
from opt_core.capture import compile as C

torch = pytest.importorskip("torch", reason="compile tests need torch (CPU)")
pytest.importorskip("torch._dynamo", reason="compile tests need torch._dynamo")


class Owner:
    def __init__(self):
        self.f = lambda x: x * 2 + 1
        self.g = torch.nn.Linear(4, 4)
        self.absent = None


def test_nodynamo_is_identity_when_off_and_a_boundary_when_on():
    fn = lambda x: x + 1
    assert C.nodynamo(fn, active=False) is fn
    marked = C.nodynamo(fn, active=True)
    assert marked is not fn and marked(torch.ones(2)).sum().item() == 4.0


def test_install_wraps_once_counts_missing_and_already_and_sets_the_floor():
    import torch._dynamo as dyn
    dyn.config.cache_size_limit = 8
    o = Owner()
    rec = C.CompileRecord("denoiser_compile", backend="eager", cache_size_limit=64)
    assert rec.install([(o, "f"), (o, "g"), (o, "absent")]) == 2
    assert dyn.config.cache_size_limit >= 64 and rec.cache_size_limit_applied >= 64
    assert rec.install([(o, "f"), (o, "g")]) == 0                     # second call: both already compiled
    c = rec.census()
    assert (c["wrapped"], c["already"], c["missing"], c["declared"], c["installs"]) == (2, 2, 1, 5, 2)
    assert c["missing_names"] == ["Owner.absent"] and c["wrapped_names"] == ["Owner.f", "Owner.g"]
    with rec.timed():
        y = o.f(torch.ones(3))                                        # first call compiles (backend eager: dynamo traces, no codegen)
        o.g(torch.ones(2, 4))
    assert torch.equal(y, torch.ones(3) * 2 + 1)
    c = rec.census()
    assert c["graphs"] >= 2 and c["frames_ok"] >= 2 and c["compile_s"] > 0 and c["timed_calls"] == 1 and c["recompile_limit_hits"] == 0
    line = rec.line("sampler-opt")
    assert line.startswith("[sampler-opt] LEVER name=denoiser_compile state=on impl=torch_compile origin=core strategy=F3.torch_compile compiled=2 declared=5 already=2 missing=1 dynamic=0 mode=default backend=eager cache_size_limit=")
    assert " frames=" in line and " frames_skipped=0 " in line and " graph_breaks=" in line and " recompile_limit_hits=0 " in line and " cache_entries_max=" in line
    assert rec.gate(expect_wrapped=2) == ["denoiser_compile: missing targets Owner.absent"]
    assert rec.gate() and "of 5 declared targets" in rec.gate()[0]
    from opt_core import strategies
    assert strategies.check("F3.torch_compile") == "F3.torch_compile"


def test_one_call_form_and_skipped_state():
    o = Owner()
    rec = C.install([(o, "f")], name="pair_net_compile", backend="eager")
    assert rec.stats_["wrapped"] == 1 and rec.state() == "on"
    empty = C.CompileRecord("none", backend="eager")
    empty.install([(o, "absent")])
    assert empty.state() == "skipped" and " state=skipped reason=nothing_wrapped " in empty.line("kit")
    assert empty.gate(require_frames=True)[0].startswith("none: wrapped 0")
    assert empty.words() == [] and " at_limit=" not in empty.line("kit")           # no per-code cache at the budget: no word, no field


def test_dynamo_counters_shape():
    d = C.dynamo_counters()
    assert set(d) == {"graphs", "calls_captured", "frames_ok", "frames_total", "frames_skipped", "graph_breaks", "graph_break_reasons", "recompile_limit_hits"}


def test_a_real_recompile_limit_hit_is_seen_by_census_and_gate():
    """Forcing test: one compiled function, a per-code budget of 2, six distinct static shapes -> dynamo gives up on the code object
    and serves the rest EAGER. The census must see it by at least one of its three signals on the running torch, and the gate must fail closed."""
    import torch._dynamo as dyn
    dyn.reset()
    o = Owner()
    rec = C.CompileRecord("small_budget", backend="eager", cache_size_limit=2)
    rec.install([(o, "f")])
    for knob in ("recompile_limit", "cache_size_limit"):                 # force the budget DOWN to 2 after install (install only ever raises it)
        if isinstance(getattr(dyn.config, knob, None), int):
            setattr(dyn.config, knob, 2)
    rec.cache_size_limit_applied = 2
    for n in range(3, 9):
        o.f(torch.ones(n))
    c = rec.census()
    assert c["recompile_limit_hits"] >= 1 or c["frames_skipped"] >= 1, c          # dynamo's counters, any spelling / frames attempted-not-converted
    assert c["cache_entries_max"] is None or c["at_limit"] == ["Owner.f"], c       # per-code cache filled to the budget (where torch exposes it)
    problems = rec.gate()
    assert problems and any(("recompile_limit_hits" in p) or ("frames_skipped" in p) for p in problems), problems   # eager actually ran: gated on dynamo's counts
    assert not any("cache at the budget" in p for p in problems), problems          # a full per-code cache alone is evidence, never a sentence
    line = rec.line("kit")
    assert " frames_skipped=" in line and " cache_entries_max=" in line
    if c["at_limit"]:                                                               # where torch exposes the per-code cache: the word and the line field
        assert rec.words() == ["cache=at_limit(Owner.f)"] and line.endswith(" at_limit=Owner.f"), (rec.words(), line)
    import inspect
    assert "allow_at_limit" not in inspect.signature(rec.gate).parameters
    for knob in ("recompile_limit", "cache_size_limit"):
        if isinstance(getattr(dyn.config, knob, None), int):
            setattr(dyn.config, knob, 64)
    dyn.reset()


def test_dynamic_is_torchs_tristate_and_prints_auto_1_0():
    assert [C.dynamic_word(v) for v in (None, True, False)] == ["auto", "1", "0"]
    for v, word in ((None, "auto"), (True, "1"), (False, "0")):
        o = Owner()
        rec = C.CompileRecord("tri", backend="eager", dynamic=v)
        rec.install([(o, "f")])
        assert rec.dynamic is v if v is None else rec.dynamic == v
        assert rec.census()["dynamic"] == word and f" dynamic={word} " in rec.line("kit-opt")
