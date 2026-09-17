"""phase.py: the per-item PHASE line — the hook wraps InferenceRunner.predict at the import of runner.inference (body first, a lever's own
import hook on the same module still fires), the model's stage methods accrue to their fields, a rebound method is re-wrapped at the next
call, a missing one prints NA + one PHASE-NOTE, and nothing returned changes."""
import importlib
import os
import sys
import textwrap
import time

import pytest

from opendde_opt import phase

FAKE_RUNNER = textwrap.dedent('''
    BODY_RAN = globals().get("BODY_RAN", 0) + 1

    def to_device(x, dev):
        return x

    class InferenceRunner:
        def __init__(self, model):
            self.model = model

        def predict(self, data):
            m = self.model
            m.get_pairformer_output(); m.expand_to_structural_tokens()
            m.prepare_diffusion_cache_for_sampling(); m.sample_diffusion()
            m.run_confidence_head()
            return {"coordinate": 42, "name": data["sample_name"]}
''')


class FakeModel:
    calls = []

    def get_pairformer_output(self):
        time.sleep(0.02); FakeModel.calls.append("trunk"); return "s", "z"

    def expand_to_structural_tokens(self):
        time.sleep(0.01); return None

    def prepare_diffusion_cache_for_sampling(self):
        time.sleep(0.01); return {}

    def sample_diffusion(self):
        time.sleep(0.03); return "xyz"

    def run_confidence_head(self):
        time.sleep(0.01); return (1, 2, 3, 4)


@pytest.fixture()
def fresh(tmp_path, monkeypatch):
    """A fresh phase module state + a fake `runner.inference` importable from tmp_path (never the real upstream)."""
    pkg = tmp_path / "runner"; pkg.mkdir(); (pkg / "__init__.py").write_text(""); (pkg / "inference.py").write_text(FAKE_RUNNER)
    for m in [k for k in sys.modules if k == "runner" or k.startswith("runner.")]:
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.reload(phase)
    monkeypatch.setenv("RANK", "0")
    yield phase
    for f in [f for f in sys.meta_path if type(f).__name__ == "_Hook" and type(f).__module__ == phase.__name__]:
        sys.meta_path.remove(f)
    for m in [k for k in sys.modules if k == "runner" or k.startswith("runner.")]:
        sys.modules.pop(m, None)


def _parse(line):
    assert line.startswith("PHASE item=")
    return dict(kv.split("=", 1) for kv in line.split()[1:])


def test_hook_wraps_predict_at_import_and_prints_one_line_per_item(fresh, capsys):
    assert fresh.install() == "armed"
    assert fresh.install() == "armed"                                       # idempotent: one hook
    assert sum(type(f).__name__ == "_Hook" for f in sys.meta_path) == 1
    ri = importlib.import_module("runner.inference")
    assert ri.BODY_RAN == 1                                                 # upstream's body ran once, by its own loader
    assert not any(type(f).__name__ == "_Hook" for f in sys.meta_path)     # the hook removed itself
    assert getattr(ri.InferenceRunner.predict, fresh.MARK) == "total"
    out = ri.InferenceRunner(FakeModel()).predict({"sample_name": "q_ab0_418"})
    assert out == {"coordinate": 42, "name": "q_ab0_418"}                  # the return value is untouched
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("PHASE")]
    assert len(lines) == 1, lines
    f = _parse(lines[0])
    assert f["item"] == "q_ab0_418" and f["lm_s"] == "-"
    trunk, sampler, conf, total = (float(f[k]) for k in ("trunk_s", "sampler_s", "conf_s", "total_s"))
    assert trunk >= 0.03 and sampler >= 0.04 and conf >= 0.01
    assert trunk + sampler + conf <= total + 2e-3                            # the guard: phases never exceed the forward they sit in (each value is rounded to the ms: three rounded terms vs one)
    # a second item resets the accumulators
    ri.InferenceRunner(FakeModel()).predict({"sample_name": "second"})
    f2 = _parse([l for l in capsys.readouterr().out.splitlines() if l.startswith("PHASE")][0])
    assert f2["item"] == "second" and abs(float(f2["trunk_s"]) - trunk) < 0.02


def test_install_after_import_wraps_now(fresh, capsys):
    ri = importlib.import_module("runner.inference")
    assert fresh.install() == "wrapped"
    ri.InferenceRunner(FakeModel()).predict({"sample_name": "x"})
    assert capsys.readouterr().out.count("PHASE item=x ") == 1


def test_a_lever_rebinding_a_stage_method_is_timed_at_its_replacement(fresh, capsys):
    fresh.install(); ri = importlib.import_module("runner.inference")

    class M(FakeModel):
        pass
    r = ri.InferenceRunner(M())
    r.predict({"sample_name": "a"}); capsys.readouterr()
    seen = []

    def lever_trunk(self):                                                  # a kit lever rebinds the class attribute between predictions
        time.sleep(0.05); seen.append(1); return "s", "z"
    M.get_pairformer_output = lever_trunk
    r.predict({"sample_name": "b"})
    f = _parse([l for l in capsys.readouterr().out.splitlines() if l.startswith("PHASE")][0])
    assert seen == [1] and float(f["trunk_s"]) >= 0.05                     # the replacement ran and was timed
    assert getattr(M.get_pairformer_output, fresh.MARK) == "trunk" and M.get_pairformer_output.__wrapped__ is lever_trunk   # functools.wraps: the lever's callable stays reachable


def test_functools_wraps_carries_the_lever_markers(fresh):
    def lever(self):
        return 1
    lever._rowpair_tp = True; lever._opt_core_original = "orig"
    w = fresh._timed(lever, "trunk")
    assert w._rowpair_tp is True and w._opt_core_original == "orig" and w.__wrapped__ is lever


def test_missing_stage_method_prints_NA_and_one_note(fresh, capsys):
    class NoConf:
        def get_pairformer_output(self): return 1
        def sample_diffusion(self): return 2

    class R:
        def __init__(self, m): self.model = m

        def predict(self, data):
            self.model.get_pairformer_output(); self.model.sample_diffusion(); return {}
    R.predict = fresh._wrap_predict(R.predict)
    R(NoConf()).predict({"sample_name": "n1"}); R(NoConf()).predict({"sample_name": "n2"})
    out = capsys.readouterr().out
    assert out.count("PHASE-NOTE conf not separable") == 1                 # once per process, not per item
    f = _parse([l for l in out.splitlines() if l.startswith("PHASE item=")][0])
    assert f["conf_s"] == "NA" and f["lm_s"] == "-" and float(f["trunk_s"]) >= 0.0 and float(f["sampler_s"]) >= 0.0


def test_only_rank0_prints(fresh, capsys, monkeypatch):
    fresh.install(); ri = importlib.import_module("runner.inference")
    monkeypatch.setenv("RANK", "3")
    ri.InferenceRunner(FakeModel()).predict({"sample_name": "r"})
    assert "PHASE" not in capsys.readouterr().out


def test_the_hook_lets_a_core_attr_patch_on_the_same_module_fire(fresh):
    """big P>1 arms opt_core's AttrPatch on runner.inference (tp.py: to_device) before phase.install(): both must install at the one import."""
    autoload = pytest.importorskip("opt_core.autoload")
    key = ("runner.inference", "to_device")
    assert key not in autoload._PATCHES
    p = autoload.patch_attr_at_import("runner.inference", "to_device", lambda orig: (lambda x, dev: ("patched", orig(x, dev))), tag="test", name="t")
    try:
        assert p.state == "armed"
        assert fresh.install() == "armed"                                   # the phase hook now sits IN FRONT of the armed AttrPatch
        ri = importlib.import_module("runner.inference")
        assert ri.BODY_RAN == 1                                             # one body run
        assert p.state == "installed" and ri.to_device(1, None) == ("patched", 1)   # the lever's patch fired
        assert getattr(ri.InferenceRunner.predict, fresh.MARK) == "total"          # and the timing wrap too
    finally:
        p._unhook(); autoload._PATCHES.pop(key, None)


def test_the_hook_behind_a_core_attr_patch_also_fires(fresh):
    """The other order (a lever armed after phase.install(), so its hook is first): still one body run, both installed."""
    autoload = pytest.importorskip("opt_core.autoload")
    key = ("runner.inference", "to_device")
    assert fresh.install() == "armed"
    p = autoload.patch_attr_at_import("runner.inference", "to_device", lambda orig: (lambda x, dev: ("patched2", orig(x, dev))), tag="test", name="t2")
    try:
        assert sys.meta_path.index(p) < [i for i, f in enumerate(sys.meta_path) if type(f).__name__ == "_Hook"][0]
        ri = importlib.import_module("runner.inference")
        assert ri.BODY_RAN == 1 and p.state == "installed" and ri.to_device(2, None) == ("patched2", 2)
        assert getattr(ri.InferenceRunner.predict, fresh.MARK) == "total"
    finally:
        p._unhook(); autoload._PATCHES.pop(key, None)


def test_stock_caller_and_kit_route_install_before_the_stock_click_group():
    """Both callers arm the same module right before upstream's entry point runs (source read: one call each)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sp = open(os.path.join(here, "stock_pred.py")).read()
    cl = open(os.path.join(here, "cli.py")).read()
    assert sp.count("_phase.install()") == 1 and sp.index("_phase.install()") < sp.index("entry.main(args=list(stock)")
    assert cl.count("_phase.install()") == 1 and cl.index("_phase.install()") < cl.index('entry.main(args=list(args), prog_name="opendde"')


def test_boundaries_resolve_on_the_pinned_upstream_source():
    """The wrapped qualnames exist in the vendored 1.1.1 source (stock/src): resolved by source read (the wheel is GPU-stack-only to import)."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    model_src = os.path.join(root, "stock", "src", "opendde", "model", "opendde.py")
    runner_src = os.path.join(root, "stock", "src", "runner", "inference.py")
    if not (os.path.isfile(model_src) and os.path.isfile(runner_src)):
        pytest.skip("stock/src not checked out beside opt/")
    ms, rs = open(model_src).read(), open(runner_src).read()
    assert re.search(r"^class OpenDDE\(", ms, re.M)
    for names in phase.PHASES.values():
        for n in names:
            assert re.search(rf"^    def {n}\(", ms, re.M), n
    assert re.search(r"^class InferenceRunner\(", rs, re.M) and re.search(r"^    def predict\(self, data", rs, re.M)
    assert re.search(r"^\s+prediction = runner\.predict\(data\)", rs, re.M)                      # the per-item call the runner's `Model forward time` brackets


# ------------------------------------------------------------------------------------------------ the per-item PEAK line
EXIT_PEAK_RE = r"PEAK item=(\S+) alloc_gib=([0-9.]+) reserved_gib=([0-9.]+)$"     # the cross-check driver reader's fixed grammar (torch kits): nothing else on the line


def test_peak_line_grammar_reset_at_item_start_and_the_parent_reduce(fresh, monkeypatch, capsys):
    """`[opendde-opt] PEAK item=<id> alloc_gib=<max_memory_allocated> reserved_gib=<max_memory_reserved>` once per item on every rank — the driver's
    exact regex, no extra token — the allocator's maxima reset at item START (mocked CUDA counters), the pre-reset maxima folded into PEAK_PROCESS,
    and the parent's rank reduce on a separate `PEAK-RANKS` line."""
    import re
    from opt_core.mem import allocator
    calls = []
    state = {"max_allocated_gib": 3.0, "max_reserved_gib": 4.0}                              # the maxima BEFORE the item (model load): folded, then reset
    monkeypatch.setattr(allocator, "counters", lambda device=None: {**state, "allocated_gib": 1.0, "reserved_gib": 2.0, "device": "0"})
    def _reset(device=None):
        calls.append("reset"); state.update(max_allocated_gib=41.257, max_reserved_gib=44.5); return True   # the item's own peak after the reset
    monkeypatch.setattr(allocator, "reset_peak", _reset)
    phase.install()
    import runner.inference as ri
    out = ri.InferenceRunner(FakeModel()).predict({"sample_name": "9c0x"})
    assert out == {"coordinate": 42, "name": "9c0x"} and calls == ["reset"]
    lines = [l for l in capsys.readouterr().out.splitlines() if " PEAK item=" in l]
    assert lines == ["[opendde-opt] PEAK item=9c0x alloc_gib=41.26 reserved_gib=44.50"], lines
    m = re.search(EXIT_PEAK_RE, lines[0]); assert m and m.groups() == ("9c0x", "41.26", "44.50")
    assert phase.PEAK_PROCESS == {"alloc_gib": 3.0, "reserved_gib": 4.0}                       # report.cuda_peak folds it: the manifest's cuda_peak stays the PROCESS high-water
    monkeypatch.setenv("WORLD_SIZE", "2"); monkeypatch.setenv("RANK", "1")
    assert phase.peak_line("x") == "[opendde-opt] PEAK item=x alloc_gib=41.26 reserved_gib=44.50"          # a rank's line carries no rank token either (the reader takes the max)
    logs = {0: "[opendde-opt] PEAK item=a alloc_gib=10.00 reserved_gib=12.00\nnoise\n", 1: "[opendde-opt] PEAK item=a alloc_gib=11.50 reserved_gib=13.25\n[opendde-opt] PEAK item=b alloc_gib=2.00 reserved_gib=3.00\n"}
    assert phase.peak_summary(logs) == ["[opendde-opt] PEAK-RANKS item=a lines=2 rank_max_alloc_gib=11.50 rank_max_reserved_gib=13.25 rank0_alloc_gib=10.00 rank0_reserved_gib=12.00",
                                        "[opendde-opt] PEAK-RANKS item=b lines=1 rank_max_alloc_gib=2.00 rank_max_reserved_gib=3.00 rank0_alloc_gib=- rank0_reserved_gib=-"]
    assert phase.peak_summary({None: logs[0] + logs[1]})[0] == "[opendde-opt] PEAK-RANKS item=a lines=2 rank_max_alloc_gib=11.50 rank_max_reserved_gib=13.25 rank0_alloc_gib=- rank0_reserved_gib=-"


def test_no_peak_line_without_cuda(fresh, monkeypatch, capsys):
    from opt_core.mem import allocator
    monkeypatch.setattr(allocator, "counters", lambda device=None: {"max_allocated_gib": None, "max_reserved_gib": None, "allocated_gib": None, "reserved_gib": None, "device": None})
    monkeypatch.setattr(allocator, "reset_peak", lambda device=None: False)
    phase.install()
    import runner.inference as ri
    ri.InferenceRunner(FakeModel()).predict({"sample_name": "cpu"})
    out = capsys.readouterr().out
    assert "PHASE item=cpu " in out and " PEAK item=" not in out
