"""The graphed sampler's bias-cache poison self-test never raises to the run (protenix_opt 0.3.48: kit glue `protenix_opt.sampler_poison_aside` over the
vendored, byte-sealed infopt_graphs/protenix/graphed.py): a probe it cannot judge — the step output is non-finite before anything is poisoned, which is what
stock's `--dtype fp16` gives on an input whose half-precision trunk overflowed — steps THAT SIGNATURE aside BY NAME to the loop's eager step body (stock
arithmetic, stock random stream), counted `aside=nonfinite_probe:<n>`; a genuine mismatch on finite values (a captured kernel reads a stale copy) also steps
aside by name, loudly, and the process verdict keeps `poison=failed` (sampler_prep falls back by name: FINAL partial=true, exit 3).  CPU only: the graph,
the hoist, the streams and the device syncs are stand-ins; the vendored `_capture` itself runs on them; the verdict replay runs the kit's own reconcile / CLI."""
import contextlib
import os
import sys
import types

import pytest
import torch

from protenix_opt import cli, report, sampler_poison_aside as spa, stack
from protenix_opt.tests.test_cli_passthrough import CoreStub, fake_stock            # noqa: F401  (fixture: the in-process stock stand-in)

SRC = os.path.join(os.path.abspath(stack.kit_home()), "src")                           # kit_home = opt/forward/flashpairformer (the FPF unit); infopt_graphs lives in its src/


@pytest.fixture
def G(monkeypatch):
    """infopt_graphs.protenix.graphed with the kit's step-aside subclass bound as its loop class (what `stack._apply` does at activation), the device
    syncs made no-ops (no CUDA here) and sampler_prep's process verdict reset around the test.  The binding is undone after the test."""
    if SRC not in sys.path:
        monkeypatch.syspath_prepend(SRC)
    import infopt_graphs.protenix as pkg
    import infopt_graphs.protenix.graphed as graphed
    from infopt_graphs.protenix import sampler_prep as sp
    sealed = graphed.GraphedDenoiseLoop
    if getattr(sealed, spa.MARK, False):                                            # bound by an earlier activation in this process: the vendored class is its base
        sealed = sealed.__mro__[1]
    monkeypatch.setattr(graphed, "GraphedDenoiseLoop", sealed)                        # restored after the test, whatever bind() put there
    monkeypatch.setattr(pkg, "GraphedDenoiseLoop", pkg.GraphedDenoiseLoop)
    cls = spa.bind(graphed)
    assert graphed.GraphedDenoiseLoop is cls and pkg.GraphedDenoiseLoop is cls and cls.__mro__[1] is sealed and getattr(cls, spa.MARK) is True
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    saved = dict(sp.POISON_STATUS); saved_stats = dict(sp.STATS)
    sp.POISON_STATUS.update(ran=False, unlisted=[], subsumed=[], squash=[], failed=False)
    yield graphed
    sp.POISON_STATUS.clear(); sp.POISON_STATUS.update(saved); sp.STATS.clear(); sp.STATS.update(saved_stats)


def _loop(G, *, hoist=True, poison_once=True, pool="private"):
    """The bound loop class with only what the self-test and the step-aside read (no EmptyCacheGuard, no device), built the way the vendored unit's own
    hand-made stand-ins are: __new__ + attributes (the subclass arms its own state lazily)."""
    from infopt_graphs.protenix import sampler_prep as sp
    cls = G.GraphedDenoiseLoop
    lp = cls.__new__(cls)
    lp.family = "protenix_sampler"; lp.pool_mode = pool; lp.max_entries = 1
    lp.entries = {}; lp.order = []; lp._prep_chain = None; lp._prep_renewed = False
    lp.stats = {"captures": 0, "replays": 0, "warmup_steps": 0, "eager_steps": 0, "events": [], "record_steps": 0, "bc_verify_steps": 0, "poison": [],
                "bypass": 0, "evictions": 0, "evicted_bytes": 0}                     # the vendored keys only: `aside` is the subclass's to add
    lp.prep = sp.PrepConfig(True) if poison_once else sp.PrepConfig(False)
    lp.poison_check = True
    lp.biascache = _Hoist() if hoist else None
    return lp


class _Hoist:
    """The bias cache's poison protocol: poison(v) -> backup, unpoison(backup); `poisoned` is what the stand-in graph reads; bind / set_mode as the capture drives them."""
    def __init__(self):
        self.poisoned = False; self.calls = []; self.modes = []; self.mode = "hit"
    def poison(self, v):
        self.poisoned = True; self.calls.append(("poison", v)); return "backup"
    def unpoison(self, backup):
        assert backup == "backup"; self.poisoned = False; self.calls.append(("unpoison",))
    def bind(self, ent):
        self.calls.append(("bind",))
    def set_mode(self, m):
        self.mode = m; self.modes.append(m)
    def summary(self):
        return {"stand_in": True}


class _Graph:
    """replay() writes the step output the scenario dictates into the static x_l: `normal` (finite, from the snapshot), and while the hoist is
    poisoned `poisoned` ('moved' = far away, 'same' = a stale-copy graph: unaffected, 'nan' = non-finite)."""
    def __init__(self, st, hoist, normal="finite", poisoned="moved"):
        self.st, self.hoist, self.normal, self.poisoned, self.replays = st, hoist, normal, poisoned, 0
    def replay(self):
        self.replays += 1
        x = self.st["x_l"]
        if self.hoist is not None and self.hoist.poisoned:
            if self.poisoned == "nan": x.fill_(float("nan"))
            elif self.poisoned == "moved": x.copy_(torch.full_like(x, 50.0))
            else: x.copy_(torch.ones_like(x) * 2.0)
        elif self.normal == "nan": x.fill_(float("nan"))
        elif self.normal == "inf": x.fill_(float("inf"))
        else: x.copy_(torch.ones_like(x) * 2.0)


def _entry(lp, G, key=("v1", (1,), 5, 3054, 400, "torch.float32"), **graph_kw):
    st = {"x_l": torch.ones(1, 5, 16, 3), "rot": torch.zeros(1, 5, 1, 3, 3), "eps": torch.zeros(1, 5, 16, 3)}
    ent = {"key": key, "st": st, "cond": {"s_inputs": torch.zeros(1, 400, 8), "input_feature_dict": {"ref_pos": torch.zeros(16, 3)}}, "body": lambda: None}
    ent["graph"] = _Graph(st, lp.biascache, **graph_kw)
    return key, ent


# ------------------------------------------------------------------------------------------------ the binding (what stack._apply installs)
def test_install_binds_the_subclass_over_the_sealed_class_and_is_idempotent(G):
    sealed = G.GraphedDenoiseLoop.__mro__[1]
    assert sealed.__module__ == "infopt_graphs.protenix.graphed" and sealed.__name__ == "GraphedDenoiseLoop" and not getattr(sealed, spa.MARK, False)
    word = spa.install(G)                                                            # a second install over the bound module: the same class, `armed(...)`
    assert word.startswith(f"armed({spa.CLASS_NAME} over ") and word.endswith("graphed.py)") and G.GraphedDenoiseLoop.__mro__[1] is sealed
    assert spa.loop_class(G) is G.GraphedDenoiseLoop and spa.state()["installed"] is True
    src = open(G.__file__, encoding="utf-8").read()                                  # the vendored file is the published one: it still raises; the fix lives in the kit package
    assert 'raise RuntimeError(f"biascache poison test FAILED' in src and 'raise RuntimeError(f"poison test FAILED' in src and "ASIDE_NONFINITE" not in src
    assert G.GraphedDenoiseLoop.ASIDE_NONFINITE == "nonfinite_probe" and G.GraphedDenoiseLoop.ASIDE_MISMATCH == "poison_mismatch"
    lp = G.GraphedDenoiseLoop.__new__(G.GraphedDenoiseLoop)                          # the vendored switch keeps its meaning through the property
    lp.disable_capture = True
    assert lp.disable_capture is True
    lp.disable_capture = False
    assert lp.disable_capture is False


def test_install_names_an_import_failure_and_never_raises(monkeypatch, capsys):
    def refuse(name):
        raise ImportError("no infopt_graphs here")
    monkeypatch.setattr(spa, "importlib", types.SimpleNamespace(import_module=refuse))
    word = spa.install()
    assert word.startswith("unavailable(ImportError('no infopt_graphs here'") and spa.state()["installed"] is False
    assert "[protenix-opt] SAMPLER_ASIDE:unavailable(ImportError('no infopt_graphs here'" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ the magnitude probe (hoist bound)
@pytest.mark.parametrize("normal", ["nan", "inf"])
def test_nonfinite_reference_replay_is_no_verdict_and_no_raise(G, normal):
    """`--dtype fp16` on an overflowed trunk: the step output is non-finite before anything is poisoned -> verdict nonfinite, nothing judged, nothing
    poisoned, the test stays armed (poison_once), the process verdict untouched (poison=not-run)."""
    from infopt_graphs.protenix import sampler_prep as sp
    lp = _loop(G); key, ent = _entry(lp, G, normal=normal)
    rec = lp._poison_test(ent, 3054)                                         # the vendored probe alone: RuntimeError biascache poison test FAILED (poisoned_vs_normal_max: nan)
    assert rec["verdict"] == "nonfinite" and rec["reads_static_buffers"] is None and "non-finite" in rec["why"]
    assert lp.biascache.calls == [], "nothing is poisoned when there is nothing to compare against"
    assert lp.stats["poison"] == [rec] and ent["poison"] is rec and lp._poison_decided() is False, "a non-finite probe decides nothing: poison_once stays armed"
    assert sp.poison_status() == "not-run" and torch.equal(ent["st"]["x_l"], torch.ones(1, 5, 16, 3)), "x_l restored from the snapshot"
    assert ent["graph"].replays == 1, "one reference replay"


def test_finite_probe_verdicts_ok_failed_and_poisoned_nonfinite_reads(G):
    from infopt_graphs.protenix import sampler_prep as sp
    lp = _loop(G); key, ent = _entry(lp, G, poisoned="moved")
    rec = lp._poison_test(ent, 3054)
    assert rec["verdict"] == "ok" and rec["reads_static_buffers"] is True and rec["poisoned_vs_normal_max"] == 48.0 and rec["normal_vs_normal_max"] == 0.0
    assert lp._poison_decided() is True and sp.poison_status() == "not-run", "the magnitude probe's pass is not the per-class verdict's word"
    assert lp.stats["poison"] == [rec] and torch.equal(ent["st"]["x_l"], torch.ones(1, 5, 16, 3))
    lp = _loop(G); key, ent = _entry(lp, G, poisoned="nan")                  # a poisoned replay that turns x_l non-finite READS the buffers (d = inf, the per-class rule)
    rec = lp._poison_test(ent, 3054)
    assert rec["verdict"] == "ok" and rec["poisoned_vs_normal_max"] == "nonfinite" and rec["reads_static_buffers"] is True and sp.poison_status() == "not-run"
    lp = _loop(G); key, ent = _entry(lp, G, poisoned="same")                 # a stale-copy graph: poisoning moves nothing -> FAILED, never a raise; the process verdict says so
    rec = lp._poison_test(ent, 3054)
    assert rec["verdict"] == "failed" and rec["reads_static_buffers"] is False and lp.stats["poison"] == [rec]
    assert sp.poison_status() == "failed" and lp._poison_decided() is True
    lp = _loop(G, poison_once=False); key, ent = _entry(lp, G, poisoned="same")   # the per-capture form (no sampler_prep) judges the same way
    assert lp._poison_test(ent, 3054)["verdict"] == "failed"


# ------------------------------------------------------------------------------------------------ the per-class probe
def test_per_class_probe_on_a_nonfinite_reference_is_no_verdict_and_no_raise(G, capsys):
    from infopt_graphs.protenix import sampler_prep as sp
    lp = _loop(G, hoist=False); key, ent = _entry(lp, G, normal="nan")
    rec = lp._poison_test_classes(ent, 3054)
    assert rec["verdict"] == "nonfinite" and rec["reads_static_buffers"] is None and rec["classes"] == 4      # st.rot, st.eps, cond.s_inputs, cond.input_feature_dict.ref_pos
    assert ent["poison_classes"] is rec and lp._poison_decided() is False and sp.poison_status() == "not-run"
    assert ent["graph"].replays == 1, "one reference replay, no class poisoned"
    assert "poison per-class probe" not in capsys.readouterr().out


def test_per_class_probe_missing_class_is_a_failed_verdict_not_a_raise(G, capsys):
    """A graph whose output ignores every poisoned class: the per-step inputs are required -> missing -> verdict failed, POISON_STATUS failed, the probe
    line printed as before, no RuntimeError."""
    from infopt_graphs.protenix import sampler_prep as sp
    lp = _loop(G, hoist=False); key, ent = _entry(lp, G, normal="finite")   # replay always writes the same finite output: NaN-poisoning st.rot / st.eps moves nothing
    rec = lp._poison_test_classes(ent, 3054)
    assert rec["verdict"] == "failed" and rec["reads_static_buffers"] is False and set(rec["required_missing"]) == {"st.rot", "st.eps"}
    assert sp.poison_status().startswith("failed") and lp._poison_decided() is True and lp.stats["poison"] == [rec]
    assert "[infopt_graphs] poison per-class probe: failed" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ the step-aside by name
def test_step_aside_drops_the_graph_counts_and_says_so_once(G, capsys):
    lp = _loop(G); key, ent = _entry(lp, G, normal="nan")
    out = lp._step_aside(ent, key, lp.ASIDE_NONFINITE, 3054, stage="selftest", detail="the reference replay's step output is non-finite")
    assert out is ent and ent["graph"] is None and ent["aside"] == "nonfinite_probe" and ent["unsupported"].startswith("aside=nonfinite_probe (selftest): ")
    assert lp.entries[key] is ent and lp.order == [key] and lp.stats["aside"] == {"nonfinite_probe": 1} and lp._aside_token() == "nonfinite_probe:1"
    assert lp.stats["events"][-1] == {"event": "sampler_aside", "reason": "nonfinite_probe", "stage": "selftest", "N_atom": 3054, "N_token": 400,
                                      "detail": "the reference replay's step output is non-finite"}
    err = capsys.readouterr().err.splitlines()
    line = [l for l in err if l.startswith("[infopt_graphs] SAMPLER ")]
    assert len(line) == 1 and line[0].startswith("[infopt_graphs] SAMPLER route=eager reason=nonfinite_probe stage=selftest N_token=400 N_atom=3054 aside=nonfinite_probe:1 — ")
    assert "by design, not a fallback" in line[0] and "FAILED" not in line[0], "the `poison test FAILED` words are the mismatch's, never the non-finite probe's"
    key2 = ("v1", (1,), 5, 6000, 800, "torch.float32"); _, ent2 = _entry(lp, G, key=key2, normal="nan")
    lp._step_aside(ent2, key2, lp.ASIDE_NONFINITE, 6000, stage="warmup")     # a second signature: counted, its own line with the running tally
    assert lp.stats["aside"] == {"nonfinite_probe": 2} and lp.order == [key, key2]
    assert "aside=nonfinite_probe:2 " in [l for l in capsys.readouterr().err.splitlines() if l.startswith("[infopt_graphs] SAMPLER ")][0]
    _, ent3 = _entry(lp, G, key=("v1", (1,), 5, 700, 90, "torch.float32"), poisoned="same")
    lp._step_aside(ent3, ent3["key"], lp.ASIDE_MISMATCH, 700, stage="selftest", detail="{'verdict': 'failed'}")
    l3 = [l for l in capsys.readouterr().err.splitlines() if l.startswith("[infopt_graphs] SAMPLER ")][0]
    assert l3.startswith("[infopt_graphs] SAMPLER route=eager reason=poison_mismatch stage=selftest N_token=90 N_atom=700 aside=nonfinite_probe:2,poison_mismatch:1 — poison test FAILED: ")
    assert lp._aside_token() == "nonfinite_probe:2,poison_mismatch:1"


def test_shared_pool_share_is_released_like_an_eviction(G, monkeypatch):
    lp = _loop(G, pool="shared"); key, ent = _entry(lp, G, normal="nan")
    released = []
    monkeypatch.setattr(G.POOLS, "release", lambda fam: released.append(fam))
    lp._step_aside(ent, key, lp.ASIDE_NONFINITE, 3054, stage="selftest")
    assert released == ["protenix_sampler"]
    _, ent2 = _entry(lp, G, key=("k2",) * 6, normal="nan"); ent2["graph"] = None      # the warm-up aside: nothing captured, nothing to release
    lp._step_aside(ent2, ent2["key"], lp.ASIDE_NONFINITE, 3054, stage="warmup")
    assert released == ["protenix_sampler"]
    lp = _loop(G, pool="private"); key, ent = _entry(lp, G, normal="nan"); released.clear()
    lp._step_aside(ent, key, lp.ASIDE_NONFINITE, 3054, stage="selftest")
    assert released == [], "a private pool goes with its graph"


def test_summary_carries_the_asides_last(G):
    lp = _loop(G); key, ent = _entry(lp, G, normal="nan")
    lp._step_aside(ent, key, lp.ASIDE_NONFINITE, 3054, stage="warmup")
    lp.max_pool_bytes = 0; lp.max_tokens = 0; lp.teacher_forced = []; lp.ec_guard = types.SimpleNamespace(report=lambda: {})
    s = lp.summary()
    assert s["aside"] == {"nonfinite_probe": 1} and list(s)[-1] == "aside", "appended last: a run with no aside prints the SUMMARY sampler= json as before"
    assert s["captures"] == 0 and s["entries"] == 1
    lp2 = _loop(G); lp2.max_pool_bytes = 0; lp2.max_tokens = 0; lp2.teacher_forced = []; lp2.ec_guard = types.SimpleNamespace(report=lambda: {})
    assert lp2.summary()["aside"] == {}, "a bf16 run: the key is there and empty"


# ------------------------------------------------------------------------------------------------ the vendored _capture on CPU stand-ins, bracketed by the glue
class _Box:
    """CPU stand-ins for what the vendored `_capture` touches: streams, the CUDA graph, the capture context, the audits, the shared pool — and the step body
    (the vendored one imports protenix): `body_out` is what the eager step writes into x_l, `replay` what the captured graph's replay writes (a callable
    (st, hoist) -> None, or a word: 'reads' = a graph that reads every per-step buffer and the poisoned hoist, 'stale' = a constant output, 'nan')."""
    def __init__(self, G, monkeypatch, lp, *, body_out="finite", replay="reads"):
        self.G, self.lp, self.body_out, self.replay_word = G, lp, body_out, replay
        self.graphs = []; self.acquired = []; self.released = []; self.body_calls = 0
        box = self
        sealed = G.GraphedDenoiseLoop.__mro__[1]

        def body(denoise_net, st, *a, **k):                                          # the eager step body: writes the step output into x_l in place
            box.body_calls += 1
            if box.body_out == "nan": st["x_l"].fill_(float("nan"))
            else: st["x_l"].copy_(torch.full_like(st["x_l"], 2.0))
            return st["x_l"]
        monkeypatch.setattr(sealed, "_step_body", staticmethod(body))

        class Graph:                                                                  # torch.cuda.CUDAGraph(): replay() writes what the scenario says; the captured st is the one the warm-up ran on
            def __init__(self):
                self.replays = 0; box.graphs.append(self)
            def capture_begin(self, *a, **k): pass
            def capture_end(self): pass
            def replay(self):
                self.replays += 1
                st = box.lp._captured_st; hoist = box.lp.biascache; x = st["x_l"]
                if callable(box.replay_word): return box.replay_word(st, hoist)
                if box.replay_word == "nan": x.fill_(float("nan")); return
                if box.replay_word == "stale": x.copy_(torch.full_like(x, 2.0)); return
                if hoist is not None and hoist.poisoned: x.copy_(torch.full_like(x, 50.0)); return
                dep = sum(v.sum() * 0 for kk, v in st.items() if kk != "x_l")               # 0 * NaN = NaN: a NaN-poisoned per-step buffer turns the output non-finite (READ)
                x.copy_(torch.full_like(x, 2.0) + dep)

        class Stream:
            def wait_stream(self, other): pass

        class Guard:                                                                  # RNGGuard(raise_on_use=False): the body consumed no RNG
            def __init__(self, *a, **k): self.consumed = []
            def __enter__(self): return self
            def __exit__(self, *a): return False

        class Census:                                                                 # SyncCensus(mode="warn")
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def report(self): return {"total": 0}

        monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
        monkeypatch.setattr(torch.cuda, "Stream", Stream)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: Stream())
        monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(G, "capture_context", lambda g, pool=None, stream=None: contextlib.nullcontext(g))
        monkeypatch.setattr(G, "_autocast_nocache", lambda: contextlib.nullcontext())
        monkeypatch.setattr(G, "RNGGuard", Guard)
        monkeypatch.setattr(G, "SyncCensus", Census)
        monkeypatch.setattr(G.POOLS, "acquire", lambda fam: box.acquired.append(fam) or "pool")
        monkeypatch.setattr(G.POOLS, "release", lambda fam: box.released.append(fam))
        orig_step_body = G.GraphedDenoiseLoop._step_body                             # the subclass's note-taking wrapper stays in front of the stand-in body

        def noting_body(self_, denoise_net, st, *a, **k):
            self_._captured_st = st                                                   # the stand-in graph replays on the st the capture ran on
            return orig_step_body(self_, denoise_net, st, *a, **k)
        monkeypatch.setattr(G.GraphedDenoiseLoop, "_step_body", noting_body)

    def capture(self, key=("v1", (1,), 5, 3054, 400, "torch.float32"), n_atom=3054):
        cond = {"input_feature_dict": {"ref_pos": torch.zeros(16, 3)}, "s_inputs": torch.zeros(1, 400, 8), "s_trunk": torch.zeros(1, 400, 8), "z_trunk": None,
                "pair_z": None, "p_lm": None, "c_l": None}
        x0 = torch.ones(1, 5, 16, 3)
        step0 = (torch.tensor(3.0), torch.tensor(0.5), torch.tensor(-0.1)); draw0 = (torch.zeros(1, 5, 1, 3, 3), torch.zeros(1, 5, 1, 3), torch.zeros(1, 5, 16, 3))
        return self.lp._capture(key, object(), cond, x0, torch.float32, (1,), 5, n_atom, step0, draw0, True, 1.003, 1.5, None, False, True)


def _live_loop(G, *, hoist=True, pool="shared", switch_off=False):
    lp = _loop(G, hoist=hoist, pool=pool)
    lp.rng_guard = True; lp.sync_audit = True; lp.max_pool_bytes = 0; lp.max_tokens = 0; lp.evict_empty_cache = False
    lp.teacher_forced_steps = (); lp.biascache_verify_steps = (); lp.nan_check_steps = (); lp.traj = None; lp.teacher_forced = []
    lp.disable_capture = switch_off
    lp.ec_guard = types.SimpleNamespace(flush_if_safe=lambda: None, report=lambda: {})
    return lp


def _sampler_lines(capsys):
    return [l for l in capsys.readouterr().err.splitlines() if l.startswith("[infopt_graphs] SAMPLER ")]


def test_capture_of_a_nonfinite_warmup_captures_nothing_and_steps_aside_at_the_warmup(G, monkeypatch, capsys):
    """THE fp16 CASE through the vendored `_capture`: the eager warm-up step's output is already non-finite -> no graph is captured (the vendored no-capture
    route, reached through the `disable_capture` property), the signature is named `stage=warmup`, nothing is poisoned or judged (poison=not-run), the entry
    keeps its static buffers for the eager steps; the record is 0.3.47's (captures 0, one `sampler_aside` event, no `sampler_capture_disabled` event)."""
    from infopt_graphs.protenix import sampler_prep as sp
    lp = _live_loop(G); box = _Box(G, monkeypatch, lp, body_out="nan")
    ent = box.capture()
    assert ent["graph"] is None and ent["aside"] == "nonfinite_probe" and ent["unsupported"].startswith("aside=nonfinite_probe (warmup): ")
    assert "x_l non-finite after the eager warm-up step" in ent["unsupported"] and "warmup_s" in ent and "audit" in ent
    assert box.graphs == [] and box.acquired == [] and box.released == [] and box.body_calls == 1, "one eager warm-up step, no capture, no pool"
    assert lp.stats["captures"] == 0 and lp.stats["aside"] == {"nonfinite_probe": 1} and lp.stats["poison"] == [] and sp.poison_status() == "not-run"
    assert [e["event"] for e in lp.stats["events"]] == ["sampler_aside"] and lp.stats["events"][0]["stage"] == "warmup"
    assert lp.entries[ent["key"]] is ent and lp.order == [ent["key"]] and lp._probe is None and lp.disable_capture is False, "the switch itself is untouched"
    lines = _sampler_lines(capsys)
    assert lines == [lines[0]] and lines[0].startswith("[infopt_graphs] SAMPLER route=eager reason=nonfinite_probe stage=warmup N_token=400 N_atom=3054 aside=nonfinite_probe:1 — ")
    assert lines[0].endswith("[x_l non-finite after the eager warm-up step]") and "FAILED" not in lines[0]


def test_capture_whose_graph_reads_the_buffers_is_kept_as_before(G, monkeypatch, capsys):
    """The bf16 case: finite warm-up, both probes pass -> the graph is kept, counted as a capture, no aside word anywhere (the vendored record, plus aside={})."""
    from infopt_graphs.protenix import sampler_prep as sp
    lp = _live_loop(G); box = _Box(G, monkeypatch, lp, body_out="finite", replay="reads")
    ent = box.capture()
    assert ent["graph"] is box.graphs[0] and not ent.get("unsupported") and "aside" not in ent
    assert lp.stats["captures"] == 1 and lp.stats["aside"] == {} and [e["event"] for e in lp.stats["events"]] == ["sampler_captured"]
    assert [r["verdict"] for r in lp.stats["poison"]] == ["ok", "ok"] and sp.poison_status().startswith("ok") and lp._poison_decided() is True
    assert box.acquired == ["protenix_sampler"] and box.released == [] and torch.equal(ent["st"]["x_l"], torch.full((1, 5, 16, 3), 2.0)), "x_l == x_1 after the probes"
    assert _sampler_lines(capsys) == []
    lp.teacher_forced = []
    assert lp.summary()["aside"] == {}


def test_capture_whose_graph_fails_the_selftest_steps_aside_loudly_and_no_later_signature_is_captured(G, monkeypatch, capsys):
    """A stale-copy graph (finite values, poisoning moves nothing): verdict failed INSIDE the vendored frame -> the graph is dropped, its pool share released,
    the signature named `stage=selftest` with the `poison test FAILED` words, POISON_STATUS failed (prep_poison=failed -> exit 3 by the verdict), the capture
    not counted, the per-class probe not run on the failed graph; the next signature of the process is served eagerly by name without a capture
    (`stage=earlier_verdict`)."""
    from infopt_graphs.protenix import sampler_prep as sp
    lp = _live_loop(G); box = _Box(G, monkeypatch, lp, body_out="finite", replay="stale")
    ent = box.capture()
    assert ent["graph"] is None and ent["aside"] == "poison_mismatch" and ent["unsupported"].startswith("aside=poison_mismatch (selftest): ")
    assert len(box.graphs) == 1 and box.acquired == ["protenix_sampler"] and box.released == ["protenix_sampler"], "captured, judged, dropped, share released"
    assert lp.stats["captures"] == 0 and lp._poison_failed is True and sp.poison_status() == "failed" and lp.stats["aside"] == {"poison_mismatch": 1}
    assert [r["verdict"] for r in lp.stats["poison"]] == ["failed"] and "poison_classes" not in ent, "the per-class probe runs only on a graph that passed the magnitude probe"
    assert [e["event"] for e in lp.stats["events"]] == ["sampler_aside"] and torch.equal(ent["st"]["x_l"], torch.full((1, 5, 16, 3), 2.0))
    out = capsys.readouterr()
    l1 = [l for l in out.err.splitlines() if l.startswith("[infopt_graphs] SAMPLER ")]
    assert len(l1) == 1 and l1[0].startswith("[infopt_graphs] SAMPLER route=eager reason=poison_mismatch stage=selftest N_token=400 N_atom=3054 aside=poison_mismatch:1 — poison test FAILED: ")
    assert "poison per-class probe" not in out.out
    key2 = ("v1", (1,), 5, 6000, 800, "torch.float32")
    ent2 = box.capture(key=key2, n_atom=6000)                                       # a later signature: no untested graph after a failed one
    assert ent2["graph"] is None and ent2["unsupported"].startswith("aside=poison_mismatch (earlier_verdict): ") and len(box.graphs) == 1
    assert lp.stats["aside"] == {"poison_mismatch": 2} and lp.order == [ent["key"], key2] and lp.stats["captures"] == 0
    l2 = _sampler_lines(capsys)
    assert len(l2) == 1 and " stage=earlier_verdict N_token=800 N_atom=6000 aside=poison_mismatch:2 " in l2[0]


def test_capture_with_a_nonfinite_selftest_steps_aside_and_keeps_the_test_armed_for_the_next_signature(G, monkeypatch, capsys):
    """Finite warm-up but a non-finite reference replay: verdict nonfinite inside the vendored frame -> aside `stage=selftest`, nothing poisoned, poison=not-run;
    the record that decided nothing does NOT disarm poison_once — the next signature's graph is probed (not `poison_skipped`) and kept when it passes."""
    from infopt_graphs.protenix import sampler_prep as sp
    lp = _live_loop(G); box = _Box(G, monkeypatch, lp, body_out="finite", replay="nan")
    ent = box.capture()
    assert ent["graph"] is None and ent["unsupported"].startswith("aside=nonfinite_probe (selftest): ") and lp.stats["aside"] == {"nonfinite_probe": 1}
    assert [r["verdict"] for r in lp.stats["poison"]] == ["nonfinite"] and lp._poison_decided() is False and sp.poison_status() == "not-run"
    assert lp.biascache.calls.count(("poison", 8.0)) == 0 and box.released == ["protenix_sampler"] and lp.stats["captures"] == 0
    assert _sampler_lines(capsys)[0].startswith("[infopt_graphs] SAMPLER route=eager reason=nonfinite_probe stage=selftest N_token=400 N_atom=3054 aside=nonfinite_probe:1 — ")
    box.replay_word = "reads"; skipped0 = sp.STATS["poison_skipped"]
    key2 = ("v1", (1,), 5, 6000, 800, "torch.float32")
    ent2 = box.capture(key=key2, n_atom=6000)
    assert ent2["graph"] is box.graphs[-1] and not ent2.get("unsupported") and lp.stats["captures"] == 1 and sp.STATS["poison_skipped"] == skipped0
    assert [r["verdict"] for r in lp.stats["poison"]] == ["nonfinite", "ok", "ok"] and sp.poison_status().startswith("ok") and lp.stats["aside"] == {"nonfinite_probe": 1}
    assert _sampler_lines(capsys) == []


def test_the_vendored_capture_switch_is_not_an_aside(G, monkeypatch, capsys):
    """INFOPT_GRAPHS_SAMPLER_CAPTURE=0 (disable_capture set): the vendored no-capture route as before — `capture disabled`, its own event, no aside word, no line."""
    lp = _live_loop(G, switch_off=True); box = _Box(G, monkeypatch, lp, body_out="finite")
    ent = box.capture()
    assert ent["graph"] is None and ent["unsupported"] == "capture disabled (INFOPT_GRAPHS_SAMPLER_CAPTURE=0)" and "aside" not in ent
    assert lp.stats["aside"] == {} and [e["event"] for e in lp.stats["events"]] == ["sampler_capture_disabled"] and box.graphs == []
    assert _sampler_lines(capsys) == [] and lp.disable_capture is True


def test_rng_guard_refusal_comes_first_and_is_not_an_aside(G, monkeypatch, capsys):
    """A step body that consumed RNG is refused by the vendored guard before the switch is read: unchanged (no aside, no probe of x_l)."""
    lp = _live_loop(G); box = _Box(G, monkeypatch, lp, body_out="nan")

    class Consumed:
        def __init__(self, *a, **k): self.consumed = ["torch.cuda"]
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(G, "RNGGuard", Consumed)
    ent = box.capture()
    assert ent["unsupported"].startswith("RNGGuard: step body consumed RNG") and lp.stats["aside"] == {} and _sampler_lines(capsys) == []


# ------------------------------------------------------------------------------------------------ the activation hook
def test_stack_apply_arms_the_aside_with_the_sampler_hook():
    """`stack._apply` binds the subclass where it records the sampler hook (SAMPLER:hook), records the word under detail, and adds no marker of its own."""
    src = open(stack.__file__, encoding="utf-8").read()
    i = src.index('applied.append("SAMPLER:hook")')
    block = src[i:i + 400]
    assert 'detail["sampler_poison_aside"] = _poison_aside.install()' in block and "from . import sampler_poison_aside as _poison_aside" in src
    assert "SAMPLER_ASIDE" not in src, "no activation marker: the loop names an aside when one happens"


# ------------------------------------------------------------------------------------------------ the verdict
PREP = stack.SAMPLER_PREP_PARTS
def _record(aside=None, prep_poison="not-run", eager_steps=199):
    """The clisampler end-of-run record of a `--dtype fp16` run whose one signature stepped aside at the warm-up (nothing captured)."""
    sampler = {"captures": 0, "replays": 0, "warmup_steps": 1, "eager_steps": eager_steps, "entries": 1, "bypass": 0, "evictions": 0,
               "poison": [], "aside": dict(aside if aside is not None else {"nonfinite_probe": 1})}
    return {"requested": True, "installed": True, "det": False, "why": None, "graph": True, "hoist": True, "max_tokens": 1536, "sampler": sampler,
            "prep": PREP, "reach": "on(floor=1536; routes by fpf_clisampler.policy)", "prep_poison": prep_poison,
            "prep_stats": {"key_shape_miss": 1, "key_value_miss": 0, "pool_chained": 0, "pool_renewed": 0, "poison_skipped": 0, "rot_ring_waits": 0},
            "policy": {"words": "admit=memory release=reach hoist_eager=1", "verdicts": {"graph": 1, "hoist_eager": 0, "stock": 0}, "releases": 0, "admit_misses": 0}}


def test_reconcile_records_the_nonfinite_aside_by_design_and_keeps_the_failed_verdict_a_fallback():
    rep = {"mode": "fast", "active": True, "levers_applied": ["sampler_graph", "sampler_prep", "sampler_reach", "sampler_admit", "t1_fused_transition"],
           "levers_fallback": [], "fallback_reasons": {}}
    r = stack.reconcile(dict(rep), {"clisampler": _record()})
    assert r["partial"] is False and r["levers_fallback"] == [] and r["reconciled"]["moves"] == {}
    assert r["asides"] == {"sampler_graph": {"nonfinite_probe": 1}} and r["gates"]["clisampler"]["sampler"]["aside"] == {"nonfinite_probe": 1}
    assert r["gates"]["clisampler"]["prep_poison"] == "not-run"
    r = stack.reconcile(dict(rep), {"clisampler": _record(aside={})})            # a bf16 run: no aside, nothing recorded under asides
    assert "asides" not in r and r["partial"] is False
    # a genuine mismatch: the signature ran eagerly by name AND the process verdict says failed -> sampler_prep falls back by name (today's rule), partial
    r = stack.reconcile(dict(rep), {"clisampler": _record(aside={"poison_mismatch": 1}, prep_poison="failed")})
    assert r["partial"] is True and r["levers_fallback"] == ["sampler_prep"] and r["reconciled"]["moves"] == {"sampler_prep": "applied -> fallback"}
    assert r["asides"] == {"sampler_graph": {"poison_mismatch": 1}} and "prep_poison=failed" in r["fallback_reasons"]["sampler_prep"]


@pytest.mark.parametrize("mode", ["exact", "fast", "big"])
def test_verdict_replay_of_an_fp16_run_exits_0_with_partial_false(mode, monkeypatch, fake_stock, tmp_path, capsys):
    """THE REPLAY: `pred --mode <m> … --dtype fp16` on the end-of-run record of an fp16 run whose sampler signature stepped aside at the warm-up
    (non-finite step output, nothing captured, 199 eager steps): FINAL partial=false, fallbacks=none, rc 0, `--dtype fp16` reaches the stock
    parser verbatim, the sampler_prep LEVER line says poison=not-run (nothing was judged) and the record carries the aside."""
    from protenix_opt import big
    applied = ["sampler_graph", "sampler_prep", "sampler_reach", "sampler_admit", "t1_fused_transition"]
    if mode == "big":
        monkeypatch.setattr(big, "_record", lambda: None); monkeypatch.setattr(big, "_STATE", dict(big._STATE, install_error=None, unit=None))
    stub = CoreStub(applied=applied)
    monkeypatch.setattr(cli, "activate", lambda core, m, dry_run=False, det_level=None: stub.enable(m))
    monkeypatch.setattr(cli, "check_mode", lambda core, m: m)
    monkeypatch.setattr(report, "register_exit_tally", lambda: True)
    monkeypatch.delenv("PTX_LEVER_REPORT", raising=False)
    m = types.ModuleType("fpf_clisampler.clisampler"); m.report = lambda: _record()
    monkeypatch.setitem(sys.modules, "fpf_clisampler.clisampler", m)
    rc = cli.main(["pred", "--mode", mode, "--input", "x", "--out_dir", str(tmp_path / "o"), "--dtype", "fp16"])
    err = capsys.readouterr().err
    assert rc == 0, err[-3000:]
    assert fake_stock["pred"][-1]["dtype"] == "fp16", "the stock knob passes through verbatim"
    assert f"[protenix-opt] FINAL mode={mode} n_gpu=1 sharding=none levers={','.join(applied)} fallbacks=none partial=false reconciled_with=clisampler" in err
    assert "NOT ACTIVE" not in err and "[protenix-opt] OUTPUTS complete " in err
    prep_line = next(l for l in err.splitlines() if l.startswith("[protenix-opt] LEVER ") and l.endswith(" lever=sampler_prep"))
    assert f" parts={PREP} poison=not-run " in prep_line
    assert "poison=failed" not in err and "poison test FAILED" not in err and "prep_poison=failed" not in err, "none of the poison-failure words on a non-finite probe"


def test_verdict_replay_of_a_genuine_mismatch_exits_3_by_name(monkeypatch, fake_stock, tmp_path, capsys):
    """The defect signal is kept: a failed self-test ran the signature eagerly (outputs complete) and the verdict says sampler_prep fell back by name —
    FINAL partial=true, NOT ACTIVE names prep_poison=failed, exit 3."""
    stub = CoreStub(applied=["sampler_graph", "sampler_prep", "t1_fused_transition"])
    monkeypatch.setattr(cli, "activate", lambda core, m, dry_run=False, det_level=None: stub.enable(m))
    monkeypatch.setattr(cli, "check_mode", lambda core, m: m)
    monkeypatch.setattr(report, "register_exit_tally", lambda: True)
    monkeypatch.delenv("PTX_LEVER_REPORT", raising=False)
    m = types.ModuleType("fpf_clisampler.clisampler"); m.report = lambda: _record(aside={"poison_mismatch": 1}, prep_poison="failed")
    monkeypatch.setitem(sys.modules, "fpf_clisampler.clisampler", m)
    rc = cli.main(["pred", "--mode", "fast", "--input", "x", "--out_dir", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_NOT_ACTIVE == 3, err[-3000:]
    assert "[protenix-opt] FINAL mode=fast n_gpu=1 sharding=none levers=sampler_graph,t1_fused_transition fallbacks=sampler_prep partial=true" in err
    assert "prep_poison=failed" in err and "[protenix-opt] OUTPUTS complete " in err


def test_release_words():
    """The README states the rule in the release vocabulary (unchanged from 0.3.47)."""
    readme = open(os.path.join(stack.tree_home(), "README.md"), encoding="utf-8").read()
    assert "`--dtype fp16`" in readme and "whose sampler probes are non-finite run the eager sampler for that item, named" in readme
