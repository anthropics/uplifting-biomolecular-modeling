"""`--no-compile` — the model-opt kits' one opt-out of a kit-added torch.compile lever, an alias of MODEL_OPT_LEVERS_OFF=compile (one mechanism).
This kit has no compile lever and stock `run_openfold predict` compiles nothing on its inference path, so: the flag parses on every verb, the
word `compile` is a KNOWN NO-OP of the ablation door on every line (named in a note, never 'not a lever of the line', the line byte for byte),
and every ACTIVE / DRY-RUN line carries `compile=none` right after the n_gpu fields. Two static premises are pinned: no torch.compile /
TorchScript compile call in the kit's own code, none on stock's inference path (if upstream starts compiling, the modes must inherit it and the
word changes — this test then says so)."""
import os
import re

import pytest

from openfold3_opt import cli, modes, report, stack
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()
OFF = modes.ENV_LEVERS_OFF
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # opt/<pkg>
KIT_DIR = os.path.dirname(os.path.dirname(PKG_DIR))                               # the kit directory (run.sh, opt/, stock/)
COMPILE_CALL = re.compile(r"(?<![\w.])torch\.compile\(|torch\.jit\.(?:script|trace)\(|torch\._dynamo\.optimize\(")


def _same(a, b):
    return (a.exports, a.unsets, a.hooks, a.levers, a.conflicts, a.levers_off, modes.describe_line(a)) == \
           (b.exports, b.unsets, b.hooks, b.levers, b.conflicts, b.levers_off, modes.describe_line(b))


@pytest.mark.parametrize("verb,extra", [("pred", ["--query-json", "q.json", "--output-dir", "o"]), ("check", []), ("warm", ["--out", "o"])])
def test_the_flag_parses_on_every_verb_and_merges_into_the_one_door(verb, extra, monkeypatch):
    a = cli.build_parser().parse_args([verb, "--mode", "fast", "--no-compile", *extra])
    assert a.no_compile is True and cli.build_parser().parse_args([verb, "--mode", "fast", *extra]).no_compile is False
    monkeypatch.setenv(OFF, "x"); monkeypatch.delenv(OFF)                            # setenv first: the variable's absence is what teardown restores
    cli.apply_no_compile_flag(a)
    assert os.environ[OFF] == "compile"
    cli.apply_no_compile_flag(a)                                                       # idempotent
    assert os.environ[OFF] == "compile"
    monkeypatch.setenv(OFF, "castcache, atom_window")                                  # other names kept, in order
    cli.apply_no_compile_flag(a)
    assert modes.levers_off_names() == ("castcache", "atom_window", "compile")
    monkeypatch.setenv(OFF, "x")
    cli.apply_no_compile_flag(cli.build_parser().parse_args([verb, "--mode", "fast", *extra]))   # flag absent: the variable untouched
    assert os.environ[OFF] == "x"


@pytest.mark.parametrize("mode,n_tokens", [("exact", 400), ("fast", 612), ("big", 2565)])
def test_compile_is_a_known_noop_word_on_every_line(mode, n_tokens):
    base = modes.resolve(mode, HOME, environ={}, n_tokens=n_tokens)
    r = modes.resolve(mode, HOME, environ={OFF: "compile"}, n_tokens=n_tokens)
    assert r.conflicts == [] and r.levers_off == [] and _same(base, r), (mode, r.conflicts, r.notes)
    assert len(r.notes) == len(base.notes) + 1 and r.notes[-1].startswith(f"{OFF}: compile — nothing to leave off") and "compile=none" in r.notes[-1]
    assert "compile" in modes.LEVERS_OFF_NOOP and modes.COMPILE_STATE == "none" and all("compile" not in ln.levers for ln in modes.LINES.values())


def test_the_noop_word_composes_with_a_real_ablation_and_never_masks_an_unknown_name():
    r = modes.resolve("fast", HOME, environ={OFF: "compile,castcache"}, n_tokens=612)
    assert r.conflicts == [] and r.levers_off == ["castcache"] and "castcache" not in r.levers
    assert any(n.startswith(f"{OFF}: compile — ") for n in r.notes) and any(n.startswith(f"{OFF}: castcache left off") for n in r.notes)
    r = modes.resolve("fast", HOME, environ={OFF: "compile,nope"}, n_tokens=612)
    assert len(r.conflicts) == 1 and " names nope: not a lever of the fast line" in r.conflicts[0], r.conflicts   # `compile` is not among the unknown names
    r = modes.resolve("off", HOME, environ={OFF: "compile"})                           # mode off: the door's one note, as for any name
    assert r.conflicts == [] and r.notes == [modes.levers_off_ignored({OFF: "compile"})]


def test_active_and_dry_run_lines_carry_compile_none_after_the_n_gpu_fields():
    res = modes.resolve("fast", HOME, environ={OFF: "compile,castcache"}, n_tokens=612)
    rep = {"active": True, "mode": "fast", "line": None, "line_spelling": modes.describe_line(res), "levers_requested": list(res.levers), "levers_off": list(res.levers_off),
           "levers_applied": [], "levers_unavailable": [], "levers_pending": list(res.levers), "hooks": list(res.hooks), "notes": list(res.notes), "n_gpu": 1,
           "compile": modes.COMPILE_STATE}
    line = report.activation_line(rep)
    assert " n_gpu=1 sharding=none compile=none" in line and line.index(" compile=none") < line.index(" levers_off=castcache") < line.index(" notes=")
    assert report.compile_field({}) == " compile=none" and " sharding=none compile=none" in report.activation_line({k: v for k, v in rep.items() if k != "compile"})
    drep = dict(rep, active=False, dry_run=True, env=dict(res.exports))
    assert " n_gpu=1 sharding=none compile=none" in report.activation_line(drep)
    rep = stack.activate("fast", dry_run=True, home=HOME, environ={OFF: "compile", "PATH": os.environ.get("PATH", "")}, log=False)
    assert rep.get("compile") == "none" and rep.get("active") is False, rep.get("reason")
    if rep.get("reason") == report.DRY_RUN_OK:                                        # the pinned upstream importable on this box: the full dry run (elsewhere it stops at the pin, by name)
        assert any("compile — nothing to leave off" in n for n in rep["notes"]) and " compile=none" in report.activation_line(rep)
    bad = stack.activate("fast", dry_run=True, home=HOME, environ={OFF: "nope,compile", "PATH": os.environ.get("PATH", "")}, log=False)
    assert bad["active"] is False and ("names nope: not a lever of the fast line" in bad["reason"] or bad["reason"] == rep.get("reason")), bad.get("reason")


def _py_files(root, skip=()):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in skip and x != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(d, f)


def test_no_compile_call_in_the_kit_or_on_stocks_inference_path():
    """The premise of compile=none: the kit's own code (opt/, tests aside) calls no torch.compile / TorchScript compile, and neither does the
    pinned upstream's package source under stock/src (its tests and docs aside; `@torch.compiler.disable` / `@torch.jit.ignore` are opt-OUT
    markers, not compilation). A hit here means a compile behaviour now exists: the modes inherit stock's untouched and the word must say so."""
    hits = []
    for f in _py_files(os.path.join(KIT_DIR, "opt"), skip=("tests",)):
        hits += [f"{f}:{i}" for i, l in enumerate(open(f, encoding="utf-8", errors="replace"), 1) if COMPILE_CALL.search(l) and not l.lstrip().startswith("#")]
    src = os.path.join(KIT_DIR, "stock", "src")
    pkg = os.path.join(src, "openfold3") if os.path.isdir(os.path.join(src, "openfold3")) else src
    if os.path.isdir(pkg):
        for f in _py_files(pkg, skip=("tests", "docs")):
            hits += [f"{f}:{i}" for i, l in enumerate(open(f, encoding="utf-8", errors="replace"), 1) if COMPILE_CALL.search(l) and not l.lstrip().startswith("#")]
    assert hits == [], hits


def test_run_sh_names_the_flag_on_the_three_verbs():
    head = open(os.path.join(KIT_DIR, "run.sh"), encoding="utf-8").read().split("set -euo pipefail")[0]
    assert head.count("[--no-compile]") == 3 and all(f"run.sh {v} " in head for v in ("pred", "check", "warm"))
