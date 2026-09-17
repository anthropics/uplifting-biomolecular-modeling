"""The fast-environment guard (attention.guard_ruling, behind ``EF2INV_REQUIRE_FAST_ENV=1``), every mode alike: on any sentence the mode prints
ONE info line, ``<mode>: fast environment NOT present (<sentences>) — proceeding, levers unchanged``, and runs with its own exit code — a box
that lacks an accelerated path is worded, never refused. Covered on a CPU box, where the guard's sentences are real (no flash-attn, no
transformer_engine): guard_ruling itself, the check verb per mode, and the arm's guard pass (source contract: it goes through guard_ruling on
the built models; the info line is logged and recorded, never a refusal)."""
import inspect
import json
import os
import subprocess
import sys

from .. import attention as AT, cli as C, modes as MD, report as R, stock_design as SD
from . import _paths as P


def test_the_ruling_words_every_mode_once_and_refuses_none():
    ss = ["esmc_attn=sdpa (the pinned stack's is xformers)", "flash-attn is not installed"]
    for m in ("off", "exact", "fast", "big"):
        line = AT.guard_ruling(ss, m)
        assert line == R.fast_env_line(m, ss) == f"{m}: fast environment NOT present (esmc_attn=sdpa (the pinned stack's is xformers); flash-attn is not installed) — proceeding, levers unchanged"
        assert "NOT ACTIVE" not in line and "REFUSED" not in line                                # the line trips no refusal marker a log reader keys on
        assert AT.guard_ruling([], m) is None
    assert not hasattr(AT, "REPORT_ONLY_MODES")                                                   # no per-mode split: one guard_ruling for all four


def test_check_facts_per_mode_on_a_box_without_the_fast_environment(monkeypatch):
    """This CPU box holds none of the pinned stack's accelerated paths, so with the guard on every mode carries the guard's sentences on
    `fast_env_report_only` and the one info line — never among its problems."""
    monkeypatch.setenv(AT.REQUIRE_VAR, "1")
    for m in ("off", "exact", "fast", "big"):
        rep = C._check_facts(MD.MODES[m], P.ROOT, P.PINS, cb=P.STOCK_FILE, require_gpu=False, weights=False)
        sentences = AT.require_problems(rep["attention"], rope_pin=None)
        assert sentences, "a CPU box without flash-attn / transformer_engine yields the guard's sentences"
        assert rep["attention"]["fast_env_report_only"] == sentences and rep["attention"]["fast_env_line"] == R.fast_env_line(MD.MODES[m].name, sentences), m   # the line names the resolved mode (`fast` -> big)
        assert not [p for p in rep["problems"] if p in sentences], m                             # never a problem, on any mode
    monkeypatch.setenv(AT.REQUIRE_VAR, "0")                                                     # the guard off: no sentence is read on any mode
    rep = C._check_facts(MD.MODES["exact"], P.ROOT, P.PINS, cb=P.STOCK_FILE, require_gpu=False, weights=False)
    assert rep["attention"]["fast_env_report_only"] == [] and "fast_env_line" not in rep["attention"]


def test_check_verb_prints_the_one_line_on_every_mode():
    core = os.path.normpath(os.path.join(P.ROOT, "..", "common", "opt_core"))
    env = dict(os.environ, MODEL_OPT=P.ROOT, PYTHONPATH=os.pathsep.join([P.OPT, core]), **{AT.REQUIRE_VAR: "1"})
    for m in ("off", "exact"):
        out = subprocess.run([sys.executable, "-m", "ef2inv_opt", "check", "--mode", m], capture_output=True, text=True, env=env, cwd=P.OPT)
        lines = out.stdout.splitlines()
        info = [l for l in lines if l.startswith(f"{m}: fast environment NOT present (") and l.endswith(") — proceeding, levers unchanged")]
        assert len(info) == 1, out.stdout[-600:]
        verdict = [l for l in lines if l.startswith(("ACTIVE ", "REFUSED "))][-1]
        assert "fast environment" not in verdict.split("problems=", 1)[1]                        # whatever else this CPU box lacks (a device, the stack), the guard's sentences are not among the problems
    js = json.loads(subprocess.run([sys.executable, "-m", "ef2inv_opt", "check", "--mode", "exact", "--json"], capture_output=True, text=True, env=env, cwd=P.OPT).stdout)
    assert js["attention"]["fast_env_report_only"] and js["attention"]["fast_env_line"].startswith("exact: fast environment NOT present (")


def test_the_arm_routes_its_guard_pass_through_the_ruling():
    """stock_design.main (the ONE arm process of every mode; GPU-only past its imports): one guard pass, on the built models (where esmc_mlp is
    decided), through attention.guard_ruling with the arm's mode — it logs the one info line and records the sentences under run.json
    attention.fast_env_report_only; nothing in the arm exits on the environment words."""
    src = inspect.getsource(SD.main)
    assert src.count("AT.guard_ruling(") == 1
    after = src.split("AT.guard_ruling(", 1)[1]
    assert after.startswith("sentences, a.arm)") and 'R.log(info_line); attn_rec["fast_env_report_only"] = sentences' in after[:700]
    assert "not_active_line(problem)" not in src
