"""phase_timing — the per-item PHASE / PHASE-NOTE / PEAK lines (CPU: host-clock path, no CUDA, no model): grammar, guard notes, the wrappers on a
dummy runner/model pair, idempotent install, and the off route's bootstrap (the module loads BY PATH, standard library only)."""
import importlib.util
import os
import subprocess
import sys

from atlasfold_opt import phase_timing as PT, stock_pred


def test_grammar_tokens_and_guard_notes():
    sec = {"lm": 1.0, "trunk": 2.0, "sampler": 3.0, "conf": 0.5, "fwd": 6.0}
    line = PT.phase_line("x+y", 101, sec, 6.5)
    assert line == "PHASE item=x+y seed=101 lm_s=1.000 trunk_s=2.000 sampler_s=3.000 conf_s=0.500 fwd_s=6.000 total_s=6.500"
    assert all("=" in t for t in line.split(" ")[1:])
    bad = PT.phase_line("x", 1, {**sec, "fwd": 5.0}, 6.5)
    assert "PHASE-NOTE guard violated on x: trunk_s+sampler_s+conf_s=5.500 > fwd_s=5.000" in bad
    assert "fwd_s=6.000 > total_s=5.000" in PT.phase_line("x", 1, sec, 5.0)
    assert "lm_s=3.000 > trunk_s=2.000" in PT.phase_line("x", 1, {**sec, "lm": 3.0}, 6.5)
    assert PT.item_token(["a b", "c=d", None]) == "a_b+c_d+unnamed" and PT.item_token([]) == "unknown"


def test_wrappers_time_a_dummy_item_on_the_host_clock(capsys):
    class Rec:
        def __init__(self, name): self.name = name

    class Model:
        def run_lm_embedder(self): return 1
        def run_trunk(self): return self.run_lm_embedder() + 1
        def inference(self, feat): return {"v": self.run_trunk()}

    class Runner:
        model = Model()
        def _iter_batch(self, items): yield 512, items
        def model_run(self, feat, seed, num_recycles=3): return self.model.inference(feat)

    Model.inference = PT.timed(Model.inference, "fwd"); Model.run_trunk = PT.timed(Model.run_trunk, "trunk"); Model.run_lm_embedder = PT.timed(Model.run_lm_embedder, "lm")
    Runner.model_run = PT.timed_total(Runner.model_run); Runner._iter_batch = PT.batch_recorder(Runner._iter_batch)
    assert PT.timed(Model.inference, "fwd") is Model.inference and PT.timed_total(Runner.model_run) is Runner.model_run   # idempotent
    r = Runner(); n0 = PT.lines_printed()
    for bucket, chunk in r._iter_batch([Rec("a"), Rec("b c")]):
        assert r.model_run({"seq_mask": [[1], [1]]}, seed=7) == {"v": 2}
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("PHASE item=a+b_c seed=7 lm_s=") and " fwd_s=" in out[0] and out[0].count("=") == 8
    assert out[1] == "PHASE-NOTE item=a+b_c seed=7 engine=atlasfold bucket=512 batch=2 lm_in=trunk_s clock=host"
    assert out[2] == "PEAK-NOTE item=a+b_c cuda=absent"
    assert PT.lines_printed() == n0 + 1 and PT.last()["item"] == "a+b_c" and PT.last()["seed"] == 7
    last = PT.last(); assert last["lm_s"] <= last["trunk_s"] <= last["fwd_s"] <= last["total_s"]


def test_static_iter_batch_keeps_its_descriptor_kind(capsys):
    """The stock monomer FoldingRunner._iter_batch is a @staticmethod called as self._iter_batch(bucketed, max_tokens): the recorder must not
    inject self (0.2.28: `run.sh warm` / every monomer in-process item raised TypeError … takes 2 positional arguments but 3 were given)."""
    class Rec:
        def __init__(self, name): self.name = name

    class Runner:
        @staticmethod
        def _iter_batch(bucketed, max_tokens_per_batch):
            for b in sorted(bucketed): yield b, bucketed[b]

    class Multi:
        def _iter_batch(self, bucketed, max_tokens_per_batch):
            for b in sorted(bucketed): yield b, bucketed[b]

    for cls in (Runner, Multi):
        setattr(cls, "_iter_batch", PT.wrap_like(cls, "_iter_batch", PT.batch_recorder(getattr(cls, "_iter_batch"))))
    assert isinstance(vars(Runner)["_iter_batch"], staticmethod) and not isinstance(vars(Multi)["_iter_batch"], staticmethod)
    for cls in (Runner, Multi):
        got = list(cls()._iter_batch({512: [Rec("a")], 64: [Rec("w")]}, 1024))
        assert [b for b, _ in got] == [64, 512] and PT._STATE["item"] == "a" and PT._STATE["bucket"] == 512
    # idempotent through install's path: an already-recording function is returned as is and re-dressed alike
    again = PT.wrap_like(Runner, "_iter_batch", PT.batch_recorder(getattr(Runner, "_iter_batch")))
    assert isinstance(again, staticmethod) and again.__func__ is vars(Runner)["_iter_batch"].__func__


def test_install_names_its_sites_or_says_why():
    r = PT.install()
    assert set(r) == {"installed", "sites", "reason"}
    if r["installed"]:
        assert set(r["sites"]) >= {"fwd", "trunk", "lm", "sampler", "conf", "total", "item"} and PT.install()["installed"]   # idempotent
    else:
        assert r["reason"]


def test_module_is_standard_library_only_and_loads_by_path():
    src = open(PT.__file__, encoding="utf-8").read()
    for forbidden in ("import torch", "from . import", "from .", "import atlasfold_opt", "opt_core"):
        assert forbidden not in src.split('"""', 2)[2], forbidden
    spec = importlib.util.spec_from_file_location("afo_phase_timing_test", PT.__file__); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.phase_line("i", 1, {"lm": 0, "trunk": 0, "sampler": 0, "conf": 0, "fwd": 0}, 0).startswith("PHASE item=i seed=1 ")


def test_off_route_bootstrap_loads_the_timer_then_the_stock_entry():
    cmd = stock_pred.command(["multimer", "--help"])
    assert cmd[0] == sys.executable and cmd[1] == "-c" and cmd[3:] == ["multimer", "--help"]
    boot = cmd[2]
    assert repr(stock_pred.PHASE_TIMING) in boot and os.path.isfile(stock_pred.PHASE_TIMING) and boot.index("_m.install()") < boot.index("from atlasfold.cli import main")
    env = stock_pred.clean_env({"AFO_X": "1"} and {})
    assert not any(k.startswith(stock_pred.ABSENT_PREFIXES) for k in env)
    # the bootstrap alone (stock entry replaced by a print) runs in a clean interpreter without the kit on sys.path
    probe = boot.replace("from atlasfold.cli import main; sys.exit(main())", "print('boot-ok', 'atlasfold_opt' in sys.modules)")
    out = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("boot-ok False") or "timing not installed" in out.stdout   # atlasfold absent under -I: named, not raised
