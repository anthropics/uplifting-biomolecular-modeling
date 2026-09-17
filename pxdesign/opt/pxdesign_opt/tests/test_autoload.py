"""The env route: PXDESIGN_OPT unset/off installs nothing (and imports nothing of the core); a kit mode runs the core pin gate, then installs
the core's finder (opt_core.autoload) from this kit's AutoloadSpec, which fires on the first import of `pxdesign`; the accepted names are
exactly the mode table's (off|exact|fast|big); any other value (e.g. `tf32` — a lever, `rows` — not a mode) refuses at the trigger. The .pth line itself — generated,
inert without a kit mode, the gate's refusal with exit status 3 under one — is proven in child interpreters by test_core_gate_routes.py."""
import sys

import pytest

from opt_core.autoload import Finder
from pxdesign_opt import _autoload


def _clear():
    for f in list(sys.meta_path):
        if isinstance(f, Finder):
            sys.meta_path.remove(f)


def test_unset_and_off_install_nothing():
    _clear()
    assert _autoload.install({}) is None
    assert _autoload.install({"PXDESIGN_OPT": ""}) is None
    assert _autoload.install({"PXDESIGN_OPT": "off"}) is None
    assert not any(isinstance(f, Finder) for f in sys.meta_path)


def test_spec_is_this_kits_table():
    sp = _autoload.spec()
    assert (sp.env, sp.package, sp.tag, sp.triggers, sp.exit_not_active) == ("PXDESIGN_OPT", "pxdesign_opt", "pxdesign-opt", ("pxdesign",), 3)
    from pxdesign_opt.modes import MODES
    assert tuple(sp.modes) == MODES == ("off", "exact", "fast", "big")


@pytest.mark.parametrize("value", ["tf32", "rows"])
def test_unknown_mode_refuses_at_the_trigger(capsys, value):
    """A value outside the table (`rows` included: not a mode) installs the finder (nothing printed at start) and REFUSES when `pxdesign` is
    first imported: the finder's line naming the accepted set, exit 3 — enable() is never reached."""
    _clear()
    f = _autoload.install({"PXDESIGN_OPT": value})
    assert isinstance(f, Finder) and capsys.readouterr().err == ""
    with pytest.raises(SystemExit) as e:
        f._fire("pxdesign")
    err = capsys.readouterr().err
    assert e.value.code == 3 and f"NOT ACTIVE: unknown PXDESIGN_OPT='{value}'" in err and "off|exact" in err, err
    _clear()


@pytest.mark.parametrize("mode", ["exact", "Exact "])
def test_kit_mode_installs_one_finder_that_fires_on_pxdesign(monkeypatch, mode):
    _clear()
    f = _autoload.install({"PXDESIGN_OPT": mode})
    assert isinstance(f, Finder) and sys.meta_path[0] is f
    assert _autoload.install({"PXDESIGN_OPT": mode}) is f                      # idempotent
    calls = []
    import pxdesign_opt
    monkeypatch.setattr(pxdesign_opt, "enable", lambda mode, strict=False, trigger=None, **kw: calls.append((mode, strict, trigger)))
    f._fire("pxdesign")
    assert calls == [("exact", True, "pxdesign")] and f not in sys.meta_path        # the value folded onto the table's name
    _clear()


def test_finder_ignores_other_modules():
    _clear()
    f = _autoload.install({"PXDESIGN_OPT": "exact"})
    assert f.find_spec("protenix") is None and f.find_spec("pxdesign.model") is None
    _clear()


def test_refused_activation_on_the_env_route_exits_3(monkeypatch):
    """`PXDESIGN_OPT=exact` with a refused activation stops the process with exit status 3: stock never runs silently under the switch."""
    import pxdesign_opt
    _clear()
    f = _autoload.install({"PXDESIGN_OPT": "exact"})

    def refuse(mode, strict=False, trigger=None, **kw):
        raise pxdesign_opt.ActivationError("no visible GPU")

    monkeypatch.setattr(pxdesign_opt, "enable", refuse)
    with pytest.raises(SystemExit) as e:
        f._fire("pxdesign")
    assert e.value.code == 3
    assert f not in sys.meta_path
    _clear()


def test_disarm_removes_this_kits_finder(monkeypatch):
    _clear()
    monkeypatch.setenv("PXDESIGN_OPT", "exact")
    f = _autoload.install()
    assert f in sys.meta_path and _autoload.disarm() is True and f not in sys.meta_path
    _clear()
