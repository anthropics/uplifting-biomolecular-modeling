"""Mode tables, the precedence rule, and the env-script transcription held to a real script sourced in a clean bash subshell."""
import os
import shutil

import pytest

from opt_core import modes

TABLE = modes.ModeTable(modes=("off", "exact", "fast", "mem64"), default="fast")


def test_table_validation():
    assert TABLE.kit_modes == ("exact", "fast", "mem64")
    with pytest.raises(ValueError):
        modes.ModeTable(modes=("exact", "fast"), default="fast")            # no off
    with pytest.raises(ValueError):
        modes.ModeTable(modes=("off", "Fast"), default="off")               # not lowercase
    with pytest.raises(ValueError):
        modes.ModeTable(modes=("off", "fast", "fast"), default="fast")      # duplicate
    with pytest.raises(ValueError):
        modes.ModeTable(modes=("off", "fast"), default="exact")            # default outside


def test_check_normalises_and_refuses():
    assert TABLE.check(" Exact ") == "exact" and TABLE.check(None) == "fast" and TABLE.check("") == "fast"
    with pytest.raises(modes.ModeError) as e:
        TABLE.check("turbo")
    assert str(e.value) == "unknown mode 'turbo' (expected off|exact|fast|mem64)"


def test_mode_argument_precedence():
    assert modes.mode_argument("exact", "off", TABLE) == "exact"
    assert modes.mode_argument(None, "off", TABLE) == "off"
    assert modes.mode_argument(None, "", TABLE) == "fast"
    assert modes.mode_argument(None, None, TABLE) == "fast"
    with pytest.raises(modes.ModeError):
        modes.mode_argument(None, "turbo", TABLE)


SCRIPT = """#!/bin/bash
export KIT_HOME=/opt/kit                            # L1 forced
export KIT_LEVERS="${KIT_LEVERS:-a,b,c}"           # L2 defaulted
export KIT_FLAG=${KIT_FLAG:-1}                     # L3 defaulted
unset KIT_STALE                                    # L4
export PYTHONPATH="/opt/kit/src:/opt/kit/third:${PYTHONPATH}"   # L5 prepend
export KIT_LEVERS="${KIT_LEVERS},d"                # L6 forced from the current value
"""


def transcription(environ):
    s = modes.EnvScript("exact")
    s.export("KIT_HOME", "/opt/kit", line="L1")
    s.default("KIT_LEVERS", "a,b,c", line="L2")
    s.default("KIT_FLAG", "1", line="L3")
    s.unset("KIT_STALE", line="L4")
    s.prepend_path("PYTHONPATH", ["/opt/kit/src", "/opt/kit/third"], line="L5")
    res = s.apply(environ)
    s.export("KIT_LEVERS", res.env["KIT_LEVERS"] + ",d", line="L6")     # a statement reading the current value: transcribed from the resolution so far
    return s.apply(environ)


PRESETS = [
    {},
    {"KIT_LEVERS": "x", "KIT_FLAG": "0"},
    {"KIT_STALE": "1", "PYTHONPATH": "/home/u/lib"},
    {"PYTHONPATH": "/opt/kit/src:/home/u/lib", "KIT_FLAG": ""},
]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash needed to source the script")
@pytest.mark.parametrize("preset", PRESETS)
def test_transcription_matches_the_sourced_script(tmp_path, preset):
    path = tmp_path / "env.sh"
    path.write_text(SCRIPT)
    sourced = modes.source_env_sh(str(path), preset)
    res = transcription(preset)
    for name in ("KIT_HOME", "KIT_LEVERS", "KIT_FLAG", "PYTHONPATH"):
        assert res.env[name] == sourced[name], name                    # byte-equal, the trailing ':' of an empty PYTHONPATH included
    assert "KIT_STALE" not in sourced and "KIT_STALE" in res.unset
    assert res.pythonpath[:2] == ("/opt/kit/src", "/opt/kit/third")


def test_prepend_if_set_idiom():
    s = modes.EnvScript("fast").prepend_path("PYTHONPATH", ["/a"], if_set=True)
    assert s.apply({}).env["PYTHONPATH"] == "/a" and s.apply({"PYTHONPATH": "/b"}).env["PYTHONPATH"] == "/a:/b"
    t = modes.EnvScript("fast").prepend_path("PYTHONPATH", ["/a"])
    assert t.apply({}).env["PYTHONPATH"] == "/a:" and t.apply({"PYTHONPATH": "/a"}).env["PYTHONPATH"] == "/a:/a"


def test_default_keeps_a_preset_value_and_notes_it():
    res = transcription({"KIT_FLAG": "0"})
    assert res.env["KIT_FLAG"] == "0" and any("KIT_FLAG pre-set to '0': kept (L3)" == n for n in res.notes)
    assert transcription({}).notes == []


def test_install_env_applies_and_returns_previous(monkeypatch):
    env = {"KIT_STALE": "1", "KIT_FLAG": "9"}
    res = transcription(env)
    before = modes.install_env(res, env)
    assert env["KIT_HOME"] == "/opt/kit" and "KIT_STALE" not in env and env["KIT_FLAG"] == "9"
    assert before["KIT_STALE"] == "1" and before["KIT_HOME"] is None and before["KIT_FLAG"] == "9"


def test_resolution_as_dict_is_json_ready():
    import json
    d = transcription({}).as_dict()
    json.dumps(d)
    assert d["mode"] == "exact" and d["refuse"] is None and d["extra"] == {}


def test_default_keep_empty_mirrors_the_dash_form():
    s = modes.EnvScript("fast").default("KIT_X", "1", keep_empty=True)                 # ${KIT_X-1}
    t = modes.EnvScript("fast").default("KIT_X", "1")                                  # ${KIT_X:-1}
    assert s.apply({}).env["KIT_X"] == "1" and s.apply({"KIT_X": ""}).env["KIT_X"] == "" and s.apply({"KIT_X": "7"}).env["KIT_X"] == "7"
    assert t.apply({}).env["KIT_X"] == "1" and t.apply({"KIT_X": ""}).env["KIT_X"] == "1" and t.apply({"KIT_X": "7"}).env["KIT_X"] == "7"


def test_unknown_mode_message_is_the_kits_when_given():
    """A kit whose usage text for an unknown mode differs from the core's composes it; the core's text is the default."""
    t = modes.ModeTable(modes=("off", "exact", "fast"), default="fast", unknown_message=lambda v: f"mode {v} is not shipped by this kit")
    with pytest.raises(modes.ModeError) as e:
        t.check("turbo")
    assert str(e.value) == "mode turbo is not shipped by this kit"
    with pytest.raises(modes.ModeError) as e:
        modes.mode_argument("turbo", None, t)
    assert str(e.value) == "mode turbo is not shipped by this kit"
    with pytest.raises(modes.ModeError) as e:
        TABLE.check("turbo")
    assert str(e.value) == "unknown mode 'turbo' (expected off|exact|fast|mem64)"
