"""The command line and the environment contract on a box without the upstream: `check` is a dry run that prints the DRY-RUN line and
exits 3 (the stock is not installed here); `fast` is not a mode; a disagreeing ESMC_OPT is refused; the activation line grammar;
the partial-activation refusal."""
import os
import re
import subprocess
import sys

import pytest

from esmc_opt import cli, report, stack

from ._paths import OPT, TREE

PY = [sys.executable, "-s", "-m", "esmc_opt"]
LINE = re.compile(r"^\[esmc-opt\] (ACTIVE|DRY-RUN|NOT ACTIVE:|APPLIED)")


def _run(args, env=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith("ESMC_")}
    e.update({"PYTHONPATH": OPT, "PYTHONDONTWRITEBYTECODE": "1"})
    e.update(env or {})
    return subprocess.run(PY + args, capture_output=True, text=True, env=e)


def test_check_dry_run_line():
    out = _run(["check", "--variant", "6b", "--mode", "exact"])
    assert out.returncode == 3, out.stderr[-600:]
    lines = [l for l in out.stdout.splitlines() if l.startswith("[esmc-opt] DRY-RUN")]
    assert lines, out.stdout
    assert "mode=exact variant=6b regime=b1 levers=pipe,fused levers_out=none" in lines[0]
    assert "would_refuse=" in lines[0]


def test_unknown_mode_word_refused():
    """Two modes: off | exact (no tier-2 composition for this model). Any other word is refused by resolve with the table in the line."""
    out = _run(["check", "--variant", "6b", "--mode", "turbo"])
    assert out.returncode == 3 and "is not one of off|exact" in out.stdout, (out.stdout, out.stderr[-200:])



def test_env_mode_disagreement():
    out = _run(["check", "--variant", "6b", "--mode", "exact"], env={"ESMC_OPT": "off"})
    assert out.returncode == 3
    assert "disagrees" in out.stdout


def test_activation_line_grammar():
    rep = {"active": True, "mode": "exact", "variant": "6b", "regime": "b1", "levers": ("pipe", "fused"), "levers_out": {"fused": "x"}, "upstream": {}, "gpu": {"name": "NVIDIA H100 80GB HBM3", "mem_mib": 81559}, "package_version": "0.1.0"}
    assert LINE.match(report.activation_line(rep))
    rep["active"] = False
    rep["reason"] = "no visible GPU"
    line = report.activation_line(rep)
    assert line.startswith("[esmc-opt] NOT ACTIVE: no visible GPU")
    assert LINE.match(report.activation_line(None))


def test_drift_is_named_on_the_line_never_refused(monkeypatch):
    """stack.version_gate: the one refusal is esm (the stock) absent; a stack pin off its version or absent (torch / triton / flash_attn /
    transformer_engine) is DRIFT — returned by name and printed at the end of the ACTIVE / DRY-RUN line as `drift=<dist>:<installed>(pin_<pinned>),…`;
    an environment at the pins prints the line with no drift field at all (the line every table expects, unchanged)."""
    p = {"pins": {"torch": "2.11.0", "triton": "3.6.0", "flash_attn": "2.7.4.post1", "transformer_engine": "2.15.0"}, "upstream": {"esm": {"version": "3.4.0"}}}
    at_pins = {"esm": "3.4.0", "torch": "2.11.0", "triton": "3.6.0", "flash_attn": "2.7.4.post1", "transformer_engine": "2.15.0"}
    monkeypatch.setattr(stack, "upstream_versions", lambda: dict(at_pins))
    ok, have, why, drift = stack.version_gate(p, ["pipe", "fused"])
    assert (ok, why, drift) == (True, None, {})
    monkeypatch.setattr(stack, "upstream_versions", lambda: dict(at_pins, triton="3.5.1", transformer_engine=None, torch="2.11.0+cu130"))
    ok, have, why, drift = stack.version_gate(p, ["pipe", "fused"])
    assert ok and why is None and drift == {"torch": "2.11.0+cu130(pin_2.11.0)", "triton": "3.5.1(pin_3.6.0)", "transformer_engine": "absent(pin_2.15.0)"}   # named, in DRIFT_DISTS order
    rep = {"active": True, "mode": "exact", "variant": "6b", "regime": "b1", "levers": ["pipe", "fused"], "levers_out": ["fused"], "upstream": have,
           "gpu": {"name": "NVIDIA H100", "sm": "sm90"}, "package_version": "0", "drift": drift}
    line = report.activation_line(rep)
    assert line.endswith(" drift=torch:2.11.0+cu130(pin_2.11.0),triton:3.5.1(pin_3.6.0),transformer_engine:absent(pin_2.15.0)") and LINE.match(line)
    assert " drift=" not in report.activation_line(dict(rep, drift={}))                     # at the pins: no field
    assert report.activation_line(dict(rep, active=None, dry_run=True)).endswith("(pin_2.15.0)")   # the DRY-RUN line carries it too
    monkeypatch.setattr(stack, "upstream_versions", lambda: dict(at_pins, esm=None))
    ok, have, why, drift = stack.version_gate(p, ["pipe"])
    assert not ok and why.startswith("the stock package esm is not installed") and drift == {}


def test_kit_switch_refused_by_name():
    """One mode table: a kit-internal switch in the environment under a named mode is refused by name (rc 3); names the kits never read are ignored."""
    out = _run(["check", "--variant", "6b", "--mode", "exact"], env={"ESMC_KIT": "x"})
    assert out.returncode == 3, (out.stdout, out.stderr[-300:])
    assert "kit-internal switch" in out.stdout and "ESMC_KIT" in out.stdout
    out = _run(["check", "--variant", "6b", "--mode", "exact"], env={"KIT_HOME_VAR": "X", "KIT_PYTHONPATH": "opt", "ESMC_OPT_KIT": "/x", "MO_SDK_SOMETHING": "1"})
    assert "kit-internal switch" not in out.stdout                       # names the kits never read are ignored (a caller's bookkeeping)


def test_mode_off_reaches_every_verb():
    """`--mode off` is never refused by a verb: check reports the mode-off resolution (rc 0)."""
    out = _run(["check", "--variant", "300m", "--mode", "off"])
    assert out.returncode == 0 and "NOT ACTIVE: mode off" in out.stdout, (out.stdout, out.stderr[-300:])


# ------------------------------------------------------------------------------------------------------------ the partial state
# A PARTIAL activation — a lever of the mode's set not applied in full on the loaded model (stack.partial_levers from the kits' own apply
# records) — is the mode's refusal BY NAME at the load (a mode is all of its levers): the APPLIED line names it, then `NOT ACTIVE: partial
# activation — …`, then PartialActivation raised from ESMC.from_pretrained. The model-side kit is a stand-in here.

def _rec(applied, fallback, kits):
    return {"model_index": 1, "trigger": "explicit", "levers_applied": list(applied), "levers_fallback": list(fallback), "kits": kits}


def test_partial_levers_is_the_kits_own_record():
    """levers_fallback (a kit's refusal) and an applied lever whose record names a fallback are the partial list; the full composition is not."""
    assert stack.partial_levers(_rec(["pipe", "fused"], [], {"pipe": {"applied": True}, "fused": {"applied": True}})) == []
    assert stack.partial_levers(_rec(["pipe"], ["fused"], {"pipe": {"applied": True}, "fused": {"applied": False, "refused": "KitRefused: x"}})) == ["fused"]
    assert stack.partial_levers(_rec(["pipe", "fused"], [], {"pipe": {"applied": True, "fallback": "a named step-aside"}, "fused": {"applied": True}})) == ["pipe"]


def test_applied_line_names_the_partial_state():
    rep = {"variant": "6b", "regime": "b1"}
    rec = dict(_rec(["pipe"], ["fused"], {}), t_apply_s=0.1, partial=["fused"])
    line = report.applied_line(rep, rec)
    assert line.startswith("[esmc-opt] APPLIED model#1 variant=6b regime=b1 levers_applied=pipe levers_fallback=fused t_apply=0.10") and line.endswith(" partial=fused")
    assert "partial=" not in report.applied_line(rep, dict(rec, partial=[]))


def test_apply_to_refuses_a_partial_model_by_name(monkeypatch, capsys):
    """stack.apply_to (the wrapper on ESMC.from_pretrained runs it): a lever of the active mode's set whose kit refuses -> the APPLIED line
    names it (levers_fallback= / partial=), then the NOT ACTIVE partial line, then PartialActivation raised from the load — on every route,
    the SDK / autoload route included: no client is returned under the mode's name with a subset of its levers."""
    rep = stack._base_report("exact", "6b", "b1", "test")
    rep.update({"active": True, "levers": ("pipe", "fused")})
    monkeypatch.setitem(stack._STATE, "report", rep)
    monkeypatch.setitem(stack._STATE, "pipe", {"applied": True})
    monkeypatch.setitem(stack._STATE, "model_index", 0)
    monkeypatch.setattr(stack, "_apply_model_lever", lambda name, client, rep: {"applied": False, "refused": "KitRefused: cannot build here"})
    with pytest.raises(stack.PartialActivation) as ei:
        stack.apply_to(object(), trigger="from_pretrained")
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("[esmc-opt] APPLIED model#1 variant=6b regime=b1 levers_applied=pipe levers_fallback=fused ") and out[0].endswith(" partial=fused")
    assert out[1] == "[esmc-opt] NOT ACTIVE: partial activation — fused: KitRefused: cannot build here; a mode is all of its levers: the load is refused; `ESMC_OPT=off` runs the stock SDK without the kit"
    assert ei.value.partial == ["fused"] and stack.status()["partial"] == ["fused"]
    monkeypatch.setattr(stack, "_apply_model_lever", lambda name, client, rep: {"applied": True})   # the full set: the record returns, no refusal
    rec = stack.apply_to(object(), trigger="from_pretrained")
    assert rec["partial"] == [] and rec["levers_applied"] == ["pipe", "fused"]
    assert "partial=" not in capsys.readouterr().out.splitlines()[-1]   # the APPLIED line the forward rows expect; no NOT ACTIVE line


def test_every_model_built_gets_the_levers_and_its_own_applied_line(monkeypatch, capsys):
    """A second (third, …) client built while the mode is active gets the levers too: APPLIED model#2 names THAT model's variant (from
    the SDK name it was loaded by); a model whose variant would need another lever set than the process's is refused by name."""
    rep = stack._base_report("exact", "6b", "b1", "test")
    rep.update({"active": True, "levers": ("pipe", "fused")})
    monkeypatch.setitem(stack._STATE, "report", rep)
    monkeypatch.setitem(stack._STATE, "pipe", {"applied": True})
    monkeypatch.setitem(stack._STATE, "model_index", 0)
    monkeypatch.setattr(stack, "_apply_model_lever", lambda name, client, rep: {"applied": True})
    stack.apply_to(object(), trigger="from_pretrained", model_name="esmc_6b")
    rec2 = stack.apply_to(object(), trigger="from_pretrained", model_name="esmc_300m")      # another size, the same lever set: applied and labelled
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("[esmc-opt] APPLIED model#1 variant=6b regime=b1 levers_applied=pipe,fused levers_fallback=none ")
    assert out[1].startswith("[esmc-opt] APPLIED model#2 variant=300m regime=b1 levers_applied=pipe,fused levers_fallback=none ")
    assert rec2["model_index"] == 2 and rec2["variant"] == "300m" and [m["variant"] for m in stack.status()["models"]] == ["6b", "300m"]
    rec3 = stack.apply_to(object(), trigger="explicit")                                     # no name: the process's variant labels it
    assert rec3["variant"] == "6b" and capsys.readouterr().out.startswith("[esmc-opt] APPLIED model#3 variant=6b ")
    real = stack.modes.resolve
    monkeypatch.setattr(stack.modes, "resolve", lambda mode, variant, regime=None: {**real(mode, variant, regime), "levers": ("pipe",)} if variant == "600m" else real(mode, variant, regime))
    with pytest.raises(stack.ActivationError) as ei:
        stack.apply_to(object(), trigger="from_pretrained", model_name="esmc_600m")
    assert "one lever set per process" in str(ei.value) and "model#4" in str(ei.value)


def test_partial_refused_line_is_the_family_grammar():
    """The one formatter of the partial refusal (report.partial_refused_line, printed by stack.refuse_partial right after the APPLIED line):
    `NOT ACTIVE: partial activation — <detail>; a mode is all of its levers: the load is refused; <the mode-off escape>`, <detail> = the
    levers first, each with its kit record's own reason (report.partial_detail)."""
    recs = {"fused": {"applied": False, "refused": "KitRefused: fixture"}, "pipe": {"applied": True, "fallback": "a named step-aside"}}
    detail = report.partial_detail(["fused", "pipe"], recs)
    assert detail == "fused: KitRefused: fixture, pipe: a named step-aside"
    line = report.partial_refused_line(detail)
    assert line == f"{report.PREFIX} NOT ACTIVE: partial activation — {detail}; a mode is all of its levers: the load is refused; {report.MODE_OFF_ESCAPE}"
    assert LINE.match(line) and "allow" not in line and not hasattr(report, "partial_line")
    assert report.partial_detail(["fused"], None) == "fused" and report.partial_detail([], None) == "none"
    with pytest.raises(stack.PartialActivation) as ei:
        stack.refuse_partial(["fused"], recs)
    assert ei.value.partial == ["fused"] and isinstance(ei.value, stack.ActivationError) and str(ei.value) == "partial activation — fused: KitRefused: fixture"


def test_check_has_no_partial_state_and_no_flag():
    """`check` applies nothing and the partial state of this package is known only at the load: no flag words it (there is no
    --allow-partial on any verb); a refusal reason exits 3, none exits 0 (cli.EXIT_* by name)."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["check", "--variant", "6b", "--allow-partial"])
    a = cli.build_parser().parse_args(["check", "--variant", "6b", "--mode", "exact"])
    assert not hasattr(a, "allow_partial")
    assert (cli.EXIT_OK, cli.EXIT_USAGE, cli.EXIT_NOT_ACTIVE) == (0, 2, 3)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["pred", "--variant", "6b", "--input", "x.fa", "--out", "o"])   # no inference command: the kit is a library drop-in
