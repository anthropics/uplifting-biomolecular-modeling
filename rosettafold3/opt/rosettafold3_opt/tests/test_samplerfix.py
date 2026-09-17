"""The graph roll-out's fixed-cost levers: `warm` (rf3.graph_flags: the eager warm-up count keyed on the
process's capture count) and the arm grammar's lever sub-steps (`@L1.warm`, `@L1.lazy`) in the FPF adapter. CPU only: the patched
graph_flags.py is loaded from the tree by path (it needs no torch until hoist_end); the adapter's parser is exercised when torch and the
core are importable, skipped otherwise."""
import importlib.util
import os

import pytest

from .. import modes, stack

GF_PATH = os.path.join(stack.kit_home(), modes.GRAPH_FLAGS_RELPATH)


def _graph_flags(monkeypatch):
    monkeypatch.delenv("RF3_CUDAGRAPH", raising=False); monkeypatch.delenv("RF3_HOIST", raising=False)
    spec = importlib.util.spec_from_file_location("_gf_under_test", GF_PATH)
    gf = importlib.util.module_from_spec(spec); spec.loader.exec_module(gf)
    return gf


def test_warm_keys_the_warmup_count_on_the_process_capture_count(monkeypatch):
    gf = _graph_flags(monkeypatch)
    assert gf.GRAPH_WARM is False and gf.GRAPH_WARMUP == 3 and gf.GRAPH_WARMUP_REPEAT == 1 and gf.GRAPH_STATS == {"captures": 0, "warmup_steps": 0}
    assert gf.warmup_steps() == 3
    gf.note_capture(3)
    assert gf.warmup_steps() == 3 and gf.GRAPH_STATS == {"captures": 1, "warmup_steps": 3}     # off: every capture warms up GRAPH_WARMUP steps
    gf.set_levers(warm=True)
    assert gf.GRAPH_WARM is True and gf.warmup_steps() == 1                                     # on: a later capture of the process warms up once
    gf.note_capture(1)
    assert gf.GRAPH_STATS == {"captures": 2, "warmup_steps": 4}
    gf.set_levers(warm=None)
    assert gf.GRAPH_WARM is True                                                                # None leaves it
    gf.set_levers(warm=False)
    assert gf.warmup_steps() == 3


def test_warm_first_capture_of_the_process_keeps_the_full_warmup(monkeypatch):
    gf = _graph_flags(monkeypatch)
    gf.set_levers(warm=True)
    assert gf.warmup_steps() == 3                                                               # nothing captured yet: the first capture runs every kernel path 3x eagerly
    gf.set_mode("1", warmup=5)
    assert gf.warmup_steps() == 5
    gf.note_capture(5)
    assert gf.warmup_steps() == 1


def test_describe_is_unchanged_and_levers_state_names_the_lever(monkeypatch):
    gf = _graph_flags(monkeypatch)
    assert list(gf.describe()) == ["RF3_CUDAGRAPH", "RF3_GRAPH_SAFE_OPS", "RF3_CUDAGRAPH_WARMUP", "RF3_HOIST"]      # the APPLIED line's keys, as they were
    assert gf.levers_state() == {"warm": False, "warmup_repeat": 1, "captures": 0, "warmup_steps": 0}
    gf.set_levers(warm=True); gf.note_capture(3)
    assert gf.levers_state() == {"warm": True, "warmup_repeat": 1, "captures": 1, "warmup_steps": 3}
    assert set(modes.flag_table(GF_PATH)) == set(modes.SWITCHES)                                # no environment word added: the levers are runtime attributes


def test_the_sampler_reads_the_count_from_the_kit_module():
    src = open(os.path.join(stack.kit_home(), "patched", "rf3", "diffusion_samplers", "inference_sampler.py"), encoding="utf-8").read()
    assert "n_warmup = GF.warmup_steps()" in src and "for _ in range(n_warmup):" in src and "GF.note_capture(n_warmup)" in src
    assert "range(GF.GRAPH_WARMUP)" not in src and "warmup=n_warmup" in src


def test_arm_grammar_lever_substeps():
    """The adapter's parser, in a child interpreter (importing the adapter here would trip this process's activation gates): skipped where
    the adapter is not importable (no torch / no core — a CPU-only host), run where the GPU stack is installed."""
    import subprocess, sys, json
    code = (
        "import json, os, sys\n"
        f"sys.path.insert(0, {os.path.join(stack.fpf_home(), 'rf3fpf')!r})\n"
        "try:\n    import fpf_rf3_adapter as A\nexcept Exception as e:\n    print(json.dumps({'skip': repr(e)})); sys.exit(0)\n"
        "out = {'subs': A.LEVER_SUBS, 'ok': [A._lever_step('L1'), A._lever_step('L1.warm')], 'refused': []}\n"
        "for bad in ('L2', 'L1.keep', 'warm', 'L1.warm.x', 'L1.lazy'):\n"
        "    try:\n        A._lever_step(bad)\n    except ValueError:\n        out['refused'].append(bad)\n"
        "print(json.dumps(out))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("{")][-1])
    if "skip" in out:
        pytest.skip(f"adapter not importable here: {out['skip'][:160]}")
    assert tuple(out["subs"]) == ("warm",)
    assert out["ok"] == [[], ["warm"]] and out["refused"] == ["L2", "L1.keep", "warm", "L1.warm.x", "L1.lazy"]
