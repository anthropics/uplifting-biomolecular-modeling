"""Frozen-weights gate (4e-bp), keyed on PROTENIX_ROOT_FROZEN=1: the stock CLI downloads what it does not
find under PROTENIX_ROOT_DIR (stock/src/runner/inference.py:291-347) — under the switch the kit refuses by name before anything imports the
stock runner: `pred` / `warm` on the CLI route, the start-up hook in every process on the env route. The rule lives in the LEAF module
protenix_opt/_frozen.py (stdlib only); the checkpoint gated is the effective --model_name's (default the stock's, batch_inference.py:609-614);
--use_template adds the two template caches; an ESM model name is refused by name. The switch unset: nothing."""
import os
import subprocess
import sys

import pytest

from protenix_opt import _frozen, cli, manifest

FIVE = list(_frozen.DATA_CACHES) + [f"checkpoint/{_frozen.STOCK_DEFAULT_MODEL_NAME}.pt"]


def _root_with(tmp_path, rels):
    for rel in rels:
        p = tmp_path / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")
    return str(tmp_path)


def test_switch_unset_means_nothing(tmp_path):
    assert _frozen.check({}, []) is None
    assert _frozen.check({"PROTENIX_ROOT_DIR": str(tmp_path)}, ["--model_name", "anything"]) is None


def test_relpaths_follow_the_effective_model_name_and_template_switch():
    assert _frozen.relpaths(_frozen.STOCK_DEFAULT_MODEL_NAME, False) == FIVE
    assert _frozen.effective([]) == {"model_name": _frozen.STOCK_DEFAULT_MODEL_NAME, "use_template": False}
    assert _frozen.effective(["--model_name", "protenix-v2", "-n", "other", "--use_template", "true"]) == {"model_name": "other", "use_template": True}   # the last occurrence wins, as click's
    assert _frozen.effective(["--model_name=protenix-v2"])["model_name"] == "protenix-v2"
    assert _frozen.relpaths("protenix-v2", True)[-3:] == [*_frozen.TEMPLATE_CACHES, "checkpoint/protenix-v2.pt"]


def test_check_names_every_absent_file_of_this_invocation(tmp_path):
    env = {"PROTENIX_ROOT_FROZEN": "1", "PROTENIX_ROOT_DIR": str(tmp_path)}
    why = _frozen.check(env, [])
    assert why and str(FIVE) in why and "runner/inference.py:291-347" in why and _frozen.STOCK_DEFAULT_MODEL_NAME in why
    _root_with(tmp_path, _frozen.DATA_CACHES)
    assert "['checkpoint/foo.pt']" in _frozen.check(env, ["--model_name", "foo"])                       # the effective model's checkpoint, not a constant
    assert _frozen.check(env, ["--model_name", "foo", "--use_template", "true"]).count("common/") == 2   # + the two template caches
    _root_with(tmp_path, ("checkpoint/foo.pt",))
    assert _frozen.check(env, ["--model_name", "foo"]) is None
    assert "ESM checkpoints" in _frozen.check(env, ["--model_name", "protenix_base_esm_v0.5.0"])       # refused by name, files or not
    assert str(FIVE) in _frozen.check({"PROTENIX_ROOT_FROZEN": "1"}, [])                                  # root unset: everything missing


def test_pred_refuses_by_name_under_the_switch_and_forwards_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PROTENIX_ROOT_FROZEN", "1"); monkeypatch.setenv("PROTENIX_ROOT_DIR", str(tmp_path))
    for k in [k for k in os.environ if k.startswith("PROTENIX_OPT")]:
        monkeypatch.delenv(k)
    rc = cli.main(["pred", "--mode", "off", "--input", "x.json", "--out_dir", str(tmp_path / "out"), "--model_name", "protenix-v2"])
    err = capsys.readouterr().err
    assert rc == 3
    lines = [l for l in err.splitlines() if "NOT ACTIVE: PROTENIX_ROOT_FROZEN=1 and frozen inputs absent" in l]
    assert len(lines) == 1 and "checkpoint/protenix-v2.pt" in lines[0] and "runner/inference.py:291-347" in lines[0]
    cmd, env, stripped = cli.stock_command(["--input", "x.json"], str(tmp_path / "proof.json"))
    assert env.get("PROTENIX_ROOT_FROZEN") == "1"                      # the stock child inherits the switch: its own hook checks again at start


def test_hook_refuses_at_the_leaf_without_importing_the_package(tmp_path):
    """In a fresh interpreter the hook's refusal under the switch imports nothing of the package beyond the hook and the leaf module —
    no stack, no manifest, no opt_core (the stock child pays a few isfile calls at start). Two forms: the switch in the process environment
    (the real start-up form: the import itself refuses, rc 3, one line) and the function called explicitly (the module census)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_OPT") and k != "PROTENIX_ROOT_FROZEN"}
    env.update(PROTENIX_ROOT_DIR=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", "import protenix_opt._autoload; print('STOCK RAN')"], env=dict(env, PROTENIX_ROOT_FROZEN="1"), capture_output=True, text=True)
    assert r.returncode == 3 and "STOCK RAN" not in r.stdout and r.stderr.count("NOT ACTIVE: PROTENIX_ROOT_FROZEN=1 and frozen inputs absent") == 1, r.stderr[-400:]
    code = ("import os, sys, protenix_opt._autoload as A\n"
            "try:\n"
            "    A.refuse_frozen_root_incomplete(dict(os.environ, PROTENIX_ROOT_FROZEN='1'), ['--model_name', 'protenix-v2'])\n"
            "except SystemExit as e:\n"
            "    print('EXIT', e.code, sorted(m for m in sys.modules if m.startswith(('protenix_opt', 'opt_core'))))\n")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("EXIT 3 "), r.stdout + r.stderr[-400:]
    mods = eval(r.stdout.split(" ", 2)[2])
    assert "protenix_opt._frozen" in mods and not any(m in mods for m in ("protenix_opt.stack", "protenix_opt.manifest", "protenix_opt.cli")) and not any(m.startswith("opt_core") for m in mods), mods
    assert "checkpoint/protenix-v2.pt" in r.stderr
    r2 = subprocess.run([sys.executable, "-c", "import protenix_opt._autoload as A; A.refuse_frozen_root_incomplete({}); print('quiet')"], env=env, capture_output=True, text=True)
    assert r2.stdout.strip() == "quiet", r2.stderr[-300:]                                                # the switch unset: nothing
