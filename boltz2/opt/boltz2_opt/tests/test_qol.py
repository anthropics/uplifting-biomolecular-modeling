"""User-facing rules of this kit: (1) `--no-compile` = MODEL_OPT_LEVERS_OFF=compile, accepted on every verb and mode,
one mechanism, a `compile=` token on the ACTIVE / DRY-RUN lines; no kit-added torch.compile lever; (3) CUDA graphs in exact / fast only — big names
the trunk graphs off by rule. Class contracts only: no provider cell / row winner is asserted here."""
import ast
import os

import pytest

from boltz2_opt import cli, modes, registry, report

KIT_OPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # opt/boltz2_opt
FORWARD = os.path.join(os.path.dirname(KIT_OPT), "forward")                     # opt/forward


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(modes.LEVERS_OFF_ENV, raising=False)
    yield
    os.environ.pop(modes.LEVERS_OFF_ENV, None)                                      # main() joins the word to the real environment: leave none behind for the other modules' tests


def test_no_compile_parses_on_every_verb_and_mode():
    ap = cli.build_parser()
    for argv in (["pred", "--mode", "fast", "--input", "x.yaml", "--out_dir", "o", "--no-compile"],
                 ["pred", "--mode", "off", "--input", "x.yaml", "--out_dir", "o", "--no-compile"],
                 ["pred", "--mode", "exact", "--no-compile", "--input", "x.yaml", "--out_dir", "o", "--seeds", "0,1"],
                 ["check", "--mode", "big", "--no-compile"], ["check", "--no-compile", "--n_gpu", "2", "--mode", "big"],
                 ["warm", "--mode", "exact", "--out", "w", "--no-compile"]):
        a = ap.parse_args(argv)
        assert a.no_compile is True, argv
        if a.cmd == "pred":
            assert "--no-compile" not in (a.stock_args or []), argv                # never a `boltz predict` argument: the stock route forwards only what follows `--`
    assert ap.parse_args(["check", "--mode", "fast"]).no_compile is False


def test_no_compile_is_the_levers_off_word_one_mechanism(monkeypatch):
    assert modes.compile_word() == "off:none_in_kit" and modes.levers_off_words() == ()
    monkeypatch.setenv(modes.LEVERS_OFF_ENV, "pallas, triattn_xla:vjp")           # a caller's own words stay; `compile` joins once
    assert modes.add_levers_off_word(modes.COMPILE_WORD) == "pallas,triattn_xla:vjp,compile"
    assert modes.add_levers_off_word(modes.COMPILE_WORD) == "pallas,triattn_xla:vjp,compile"
    assert modes.levers_off_words() == ("pallas", "triattn_xla:vjp", "compile") and modes.compile_word() == "off:user"
    assert modes.compile_word({modes.LEVERS_OFF_ENV: "compile"}) == "off:user" and modes.compile_word({}) == "off:none_in_kit"
    assert modes.compile_word({modes.LEVERS_OFF_ENV: "compiler,nocompile"}) == "off:none_in_kit"   # whole words only


def test_main_puts_the_word_in_the_environment_before_anything_resolves(monkeypatch):
    seen = {}
    def fake_check(a):
        seen["env"] = os.environ.get(modes.LEVERS_OFF_ENV); seen["word"] = modes.compile_word(); return 0
    monkeypatch.setattr(cli, "cmd_check", fake_check)
    monkeypatch.delenv("BOLTZ2_OPT", raising=False)
    assert cli.main(["check", "--mode", "fast", "--no-compile"]) == 0
    assert seen == {"env": "compile", "word": "off:user"}
    seen.clear(); os.environ.pop(modes.LEVERS_OFF_ENV, None)                      # (not monkeypatch.delenv: it would record `compile` as the value to restore at teardown)
    assert cli.main(["check", "--mode", "fast"]) == 0 and seen == {"env": None, "word": "off:none_in_kit"}


def test_active_and_dry_run_lines_carry_the_compile_token(monkeypatch):
    rep = {"mode": "fast", "route": "worker", "gpu": "H", "n_gpu": 1, "levers_applied": ["resid"], "levers_fallback": [], "worker": "w.py", "kernels": "on"}
    line = report.active_line(rep)
    import boltz2_opt
    assert f" worker=w.py kit={boltz2_opt.__version__} opt_core=" in line and f" {report.version_tokens()} kernels=on compile=off:none_in_kit" in line, line   # after worker=: kit / core versions (which tree printed the line), then kernels= and compile=
    monkeypatch.setenv(modes.LEVERS_OFF_ENV, "compile")
    assert " kernels=on compile=off:user" in report.active_line(rep)
    d = report.dry_run_line({"mode": "exact", "route": "worker", "levers": ["resid", "mask2"], "env": {"A": "1"}})
    assert " levers=resid,mask2 compile=off:user env=A=1" in d, d
    assert report.active_line(dict(rep, compile="on")).count(" compile=on") == 1   # a report's own value wins (a future compile lever states itself)


def test_no_row_of_this_kit_adds_a_compile_lever():
    """QoL rule 1 by absence: no registry lever is a compile lever and modes.COMPILE_LEVERS is empty — a kit-added torch.compile lever would have
    to list itself in modes.COMPILE_LEVERS (and ship off by default unless its cache / dynamic-shape condition is met)."""
    assert modes.COMPILE_LEVERS == ()
    names = " ".join(registry.LEVERS) if isinstance(registry.LEVERS, dict) else " ".join(str(x) for x in registry.LEVERS)
    assert "compile" not in names.lower()


def test_no_torch_compile_call_in_kit_owned_sources():
    """No kit-owned module CALLS torch.compile (strings / refusal texts may name it): opt/boltz2_opt/*.py and the carried lever sources under opt/forward/."""
    offenders = []
    for root in (KIT_OPT, FORWARD):
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    tree = ast.parse(open(p, encoding="utf-8").read())
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        fn = node.func
                        dotted = []
                        while isinstance(fn, ast.Attribute):
                            dotted.append(fn.attr); fn = fn.value
                        if isinstance(fn, ast.Name):
                            dotted.append(fn.id)
                        name = ".".join(reversed(dotted))
                        if name == "torch.compile":
                            offenders.append(f"{os.path.relpath(p, os.path.dirname(KIT_OPT))}:{node.lineno} {name}")
    assert offenders == [], offenders


def test_big_names_the_trunk_graphs_off_by_rule_exact_and_fast_keep_them():
    """QoL rule 3 as a class contract: the memory mode resolves with NO CUDA-graph lever of the trunk (graph_trunk) — tabled, off by a `rule:` reason,
    its words and attachment gone — while exact / fast carry it. (The sampler's roll-out and hoist ride big up to the card's token ceiling from 0.3.15,
    modes.BIG_SAMPLER_CEILINGS: a separate, size-gated rule — the trunk graphs stay off by this one.)"""
    modes.set_n_gpu(1)
    for m in ("exact", "fast"):
        r = modes.resolve(m)
        assert "graph_trunk" in r["levers"] and "graph" in r["attach"] and r["env"].get("BOLTZ_GRAPH_TRUNK")
    rb = modes.resolve("big")
    assert "graph_trunk" in modes.MODES["big"]["levers"] and "graph_trunk" not in rb["levers"]
    assert rb["off"].get("graph_trunk", "").startswith("rule:") and "graph" not in rb["attach"]
    assert not [k for k in rb["env"] if k.startswith("BOLTZ_GRAPH_TRUNK")]
    assert "graph_sampler" not in rb["levers"] and not ({"rollout", "dit_hoist", "align_jacobi64"} & set(rb["levers"])) and rb["off"]["rollout"].startswith("rule:") \
        and modes.SAMPLER_MAX_TOKENS_ENV not in rb["env"] and rb["env"]["BOLTZ_SAMPLER_DIT"].endswith("weights=sample")   # no CUDA graph of any kind in the memory mode: the roll-out's whole-loop graph leaves by rule (its rider and the hoist with it, by name); the fused step keeps its eager host (0.3.16)
    line = report.active_line({"mode": "big", "route": "worker", "levers_applied": rb["levers"], "levers_fallback": [], "worker": "w", "kernels": "on", "off": rb["off"]})
    assert "graph_trunk:rule:" in line and " off=" in line


def test_the_worker_warms_the_stacks_imports_before_the_model_script(monkeypatch, capsys):
    """QoL rule 2 (no first-use stalls): worker_launch calls opt_core.warm_imports() once before the model script runs, prints one WARM_IMPORTS line,
    and never turns a warm-up failure into a refusal; the kit's core pin is at or above the core that ships warm_imports (0.5.66.0)."""
    import inspect
    import opt_core
    from boltz2_opt import worker_launch, stack
    from opt_core import gates as core_gates
    src = inspect.getsource(worker_launch.main)
    assert src.index("warm_imports()") < src.index("runpy.run_path("), "warmed before the model script runs"
    calls = []
    monkeypatch.setattr(opt_core, "warm_imports", lambda *a, **k: calls.append(k) or {"cuequivariance_ops_torch": "present"})
    worker_launch.warm_imports()
    libs = tuple(calls[0].get("libraries") or ())
    assert libs[:1] == ("torch",) and "cuequivariance_ops_torch" in libs and libs.index("torch") < libs.index("cuequivariance_ops_torch"), libs   # torch before cuEquivariance's ops library (its CUDA runtime resolves through torch's)
    err = capsys.readouterr().err
    assert len(calls) == 1 and "[boltz2-opt] WARM_IMPORTS cuequivariance_ops_torch=present total_s=" in err, err
    def boom(*a, **k): raise RuntimeError("x")
    monkeypatch.setattr(opt_core, "warm_imports", boom)
    worker_launch.warm_imports()                                                    # no raise
    assert "WARM_IMPORTS error=RuntimeError" in capsys.readouterr().err
    want = core_gates.core_pin_check(stack.PYPROJECT).details["pinned"]["version"] if "pinned" in (core_gates.core_pin_check(stack.PYPROJECT).details or {}) else None
    pin = want or __import__("re").search(r'\[tool\.opt_core\][^[]*?version = "([0-9.]+)"', open(stack.PYPROJECT).read(), __import__("re").S).group(1)
    assert tuple(int(x) for x in pin.split(".")) >= (0, 5, 66, 0), pin

