import sys
"""One resolution per run: the primary process resolves the mode line once and exports it frozen (modes.ENV_RESOLVED); a process the run
starts (upstream 0.5.0's forkserver DataLoader workers re-import openfold3 and re-enter the hooks) INHERITS it verbatim — no line lookup, no
token count, no gate, no ACTIVE line, no exit tally."""
import io, json, os

import pytest

from openfold3_ob0_opt import modes, report, stack
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()


def _primary(**kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith(modes.SWITCH_PREFIXES) and not k.startswith("OPENFOLD3_OB0_OPT")}
    return modes.resolve("fast", HOME, environ=env, n_tokens=384, **kw), env


def test_freeze_thaw_roundtrip():
    res, _ = _primary()
    back = modes.thaw(modes.freeze(res))
    assert back.inherited is True and res.inherited is False
    back.inherited = False
    assert back == res                                                                     # every field survives verbatim (exports, hook dirs, gates' words, notes)
    with pytest.raises(ValueError, match=modes.ENV_RESOLVED):
        modes.thaw("{not json")
    with pytest.raises(ValueError, match=modes.ENV_RESOLVED):
        modes.thaw(json.dumps({"line": None}))                                             # required fields missing


def test_a_started_process_inherits_the_frozen_resolution_without_re_resolving(monkeypatch):
    res, env = _primary()
    child_env = dict(env); child_env[modes.ENV_RESOLVED] = modes.freeze(res)
    def boom(*a, **k): raise AssertionError("re-resolved in a child process")
    for name in ("line_of", "conf_gate", "graphs_gate", "pending_levers", "foreign_params"):
        if hasattr(modes, name):
            monkeypatch.setattr(modes, name, boom)                                         # nothing of the resolver runs in the child
    got = modes.resolve("fast", HOME, environ=child_env, n_tokens=None)                    # the child never counted tokens
    assert got.inherited is True and got.n_tokens == 384 and got.exports == res.exports and got.hook_dirs == res.hook_dirs and got.conflicts == res.conflicts
    assert got.size_gate == res.size_gate and got.conf_gate == res.conf_gate and got.levers == res.levers
    with pytest.raises(ValueError, match="inherits"):
        modes.resolve("exact", HOME, environ=child_env)                                    # a child cannot ask for another mode than the run's


def test_an_inherited_activation_prints_nothing_and_registers_no_exit_tally(monkeypatch):
    res, env = _primary()
    rep = {"mode": "fast", "active": True, "inherited": True, "line": None, "levers": res.levers}
    buf = io.StringIO()
    report.log_activation(dict(rep), stream=buf)
    assert buf.getvalue() == ""                                                             # an inherited activation: silent
    failed = dict(rep, active=False, reason="hook x failed")
    report.log_activation(failed, stream=buf)
    assert buf.getvalue().startswith(f"{report.PREFIX} {report.WORKER_WORD} ") and buf.getvalue().count("\n") == 1   # a child's failure: under its own word, never ACTIVE
    calls = []
    import opt_core.report
    monkeypatch.setattr(opt_core.report, "register_exit_tally", lambda *a, **k: calls.append(1))
    report.register_exit_tally(lambda: rep, lambda: 0, inherited=True)
    assert calls == []                                                                      # no exit tally from a started process
    report.register_exit_tally(lambda: rep, lambda: 0)
    assert calls == [1]                                                                     # the primary process registers one


def test_activation_exports_the_frozen_resolution_once(monkeypatch):
    """stack.activate in the primary process puts modes.freeze(res) into the environment the run's children inherit; a child's activate
    (the variable present) reports inherited, applies without re-evaluating the gates, and prints nothing."""
    src = open(stack.__file__, encoding="utf-8").read()
    assert "os.environ[modes.ENV_RESOLVED] = modes.freeze(res)" in src and "if res.inherited:" in src


def test_the_variable_is_declared_and_stripped_for_stock():
    from openfold3_ob0_opt import _autoload, env
    assert modes.ENV_RESOLVED in _autoload.DECLARED and modes.ENV_RESOLVED.startswith(_autoload.ENV)
    assert any(modes.ENV_RESOLVED.startswith(p) for p in env.must_be_absent(HOME))          # the stock child never sees it


def test_the_pth_hook_inherits_in_a_process_the_run_started():
    """upstream's worker interpreters inherit the primary's environment — OPENFOLD3_OB0_OPT=fast, OPENFOLD3_OB0_OPT_UNDECLARED_X=1 (a name the env route refuses on its own)
    AND the frozen resolution (modes.ENV_RESOLVED): the .pth hook (_autoload.install) re-judges nothing on the env route there and installs the
    core's finder, whose trigger thaws the resolution (stack.activate). Without the frozen resolution the same variables are the env route's own
    call and an undeclared name is refused by name (exit 3)."""
    import pytest
    from opt_core import autoload as core_autoload
    from openfold3_ob0_opt import _autoload
    assert _autoload.RESOLVED == modes.ENV_RESOLVED and _autoload.RESOLVED in _autoload.DECLARED
    res = modes.resolve("fast", HOME, environ={})
    child = {"OPENFOLD3_OB0_OPT": "fast", "OPENFOLD3_OB0_OPT_UNDECLARED_X": "1", modes.ENV_RESOLVED: modes.freeze(res)}
    core_autoload.disarm(_autoload.spec())
    try:
        f = _autoload.install(child)
        assert f is not None and f in sys.meta_path                                         # armed: the activation runs at the trigger import and inherits
    finally:
        core_autoload.disarm(_autoload.spec())
    primary = {k: v for k, v in child.items() if k != modes.ENV_RESOLVED}
    with pytest.raises(SystemExit) as ei:
        _autoload.install(primary)
    assert ei.value.code == 3


def test_every_package_switch_a_lever_or_line_uses_is_declared():
    """The .pth hook exits 3 on an undeclared OPENFOLD3_OB0_OPT_* name (_autoload.install): every such switch a registry lever reads (Lever.env_keys),
    a line exports (Line.env), or a cell module lists (modes.ROLLOUT_ENVS / PAIR_ENVS) is in _autoload.DECLARED — so a caller setting a lever's own
    variable (e.g. the rollout gate OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS=0) never trips the gate."""
    from openfold3_ob0_opt import _autoload, registry
    prefix = "OPENFOLD3_OB0_OPT_"
    used = {k for lv in registry.LEVERS.values() for k in lv.env_keys if k.startswith(prefix)}
    used |= {k for ln in modes.LINES.values() for k in ln.env if k.startswith(prefix)}
    used |= {k for k in tuple(modes.ROLLOUT_ENVS) + tuple(modes.PAIR_ENVS) if k.startswith(prefix)}
    missing = sorted(used - set(_autoload.DECLARED))
    assert not missing, missing
    assert "OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS" in _autoload.DECLARED and "OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS" in modes.ROLLOUT_ENVS
