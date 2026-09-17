"""``--no-compile`` (cli.no_compile_alias): the opt-out of the kit's ``compile`` lever is an ALIAS of the ablation switch — ``compile`` appended to
``MODEL_OPT_LEVERS_OFF`` for the process when the mode's selection carries the lever (fast / big), nothing under off / exact (which never
compile; the flag is accepted there, never refused) — and the ACTIVE line's ``compile=`` word (report.compile_word): ``on`` | ``off:user`` |
``off:mode``, ``stepped_aside:<reason>`` on the LEVER line after a pass whose compiled callable raised and ran eagerly."""
import os

import pytest

from af3_torch_opt import cli, modes, report, stack

ENV = modes.ENV_LEVERS_OFF


def active(err: str) -> str:
    return next(l for l in err.splitlines() if " ACTIVE mode=" in l)


@pytest.mark.parametrize("mode,word,env_after", [("fast", "off:user", "compile"), ("big", "off:user", "compile"), ("exact", "off:mode", None), ("off", "off:mode", None)])
def test_check_no_compile(box, capsys, monkeypatch, mode, word, env_after):
    monkeypatch.setenv(ENV, "")                                                         # blank = unset for the switch; registered with monkeypatch so the alias's write is undone at teardown
    assert cli.main(["check", "--mode", mode, "--no-compile"]) == 0
    act = active(capsys.readouterr().err)
    assert f" compile={word}" in act, act
    assert (os.environ.get(ENV) or None) == env_after                                              # one mechanism: the switch's variable carries the name (fast / big) or stays unset (off / exact)
    rep = stack.status()
    assert "compile" not in rep["levers"]
    if env_after:
        assert rep["levers_off"] == ["compile"] and act.endswith(" levers_off=compile")
    else:
        assert rep["levers_off"] == [] and "levers_off=" not in act


def test_check_default_compile_on(box, capsys, monkeypatch):
    monkeypatch.setenv(ENV, "")
    for mode, word in (("fast", "on"), ("big", "on"), ("exact", "off:mode"), ("off", "off:mode")):
        assert cli.main(["check", "--mode", mode]) == 0
        act = active(capsys.readouterr().err)
        assert f" compile={word}" in act and "levers_off=" not in act, act
    assert not os.environ.get(ENV)


def test_alias_extends_an_existing_list(box, capsys, monkeypatch):
    """A run already dropping a lever keeps it: --no-compile appends, in order, once."""
    monkeypatch.setenv(ENV, "stepgraph")
    assert cli.main(["check", "--mode", "fast", "--no-compile"]) == 0
    act = active(capsys.readouterr().err)
    assert os.environ[ENV] == "stepgraph,compile" and act.endswith(" compile=off:user levers_off=stepgraph,compile"), act
    monkeypatch.setenv(ENV, "compile")                                                  # named by the user already: the flag adds nothing
    assert cli.main(["check", "--mode", "fast", "--no-compile"]) == 0
    assert os.environ[ENV] == "compile" and " compile=off:user levers_off=compile" in capsys.readouterr().err


def test_pred_no_compile_reaches_the_model_process(box, capsys, monkeypatch):
    """pred --no-compile under fast: the model process is launched without `compile` in --levers and with the variable in its environment; the
    lever's LEVER line reads state=off reason=levers_off; the run is whole (rc 0, not partial)."""
    import json
    from .conftest import stub_calls
    monkeypatch.setenv(ENV, "")
    inp = os.path.join(str(box["tmp"]), "in"); os.makedirs(inp, exist_ok=True)
    with open(os.path.join(inp, "tiny.json"), "w") as fh: json.dump({"name": "tiny", "sequences": [], "modelSeeds": [1]}, fh)
    assert cli.main(["pred", "--mode", "fast", "--no-compile", "--json_path", os.path.join(inp, "tiny.json"), "--output_dir", str(box["tmp"] / "out_nc")]) == 0
    err = capsys.readouterr().err
    act = active(err)
    assert " compile=off:user" in act and act.endswith(" levers_off=compile"), act
    fwd = [c for c in stub_calls(box) if c and os.path.basename(c[1]) == "forward.py"][-1]
    assert "compile" not in fwd[fwd.index("--levers") + 1].split(","), fwd
    lever = next(l for l in err.splitlines() if " LEVER name=compile " in l)
    assert " state=off " in lever and " reason=levers_off " in lever, lever
    assert " PARTIAL " not in err, err


def test_compile_word():
    assert report.compile_word({"levers": ["bf16w", "compile"], "levers_off": []}) == "on"
    assert report.compile_word({"levers": ["bf16w"], "levers_off": ["compile"]}) == "off:user"
    assert report.compile_word({"levers": ["stepgraph"], "levers_off": []}) == "off:mode"
    assert report.compile_word({"levers": ["compile"], "dead": {"compile": "eager x2: dit_transition.0=RuntimeError boom"}}) == "stepped_aside:eager_x2"
    state, reason, ev = report.lever_state("compile", {"mode": "fast", "levers": ["compile"], "levers_applied": ["compile"], "compiled": 13, "dead": {"compile": "eager x1: a=b"}})
    assert state == "on" and ev["compile"] == "stepped_aside:eager_x1" and ev["dead"] == 1 and list(ev)[-1] == "dead"      # dead=1 stays the line's last word
    assert report.compile_word({"levers": ["compile"], "dead": {"compile": "RuntimeError('stub kernel error')"}}) == "stepped_aside:RuntimeErrorstub_kernel_error"   # blank-free, quote-free


def test_parser_accepts_the_flag_on_pred_and_check():
    p = cli.build_parser()
    assert p.parse_args(["check", "--mode", "fast", "--no-compile"]).no_compile is True
    assert p.parse_args(["pred", "--output_dir", "/x", "--no-compile"]).no_compile is True
    assert p.parse_args(["check"]).no_compile is False
