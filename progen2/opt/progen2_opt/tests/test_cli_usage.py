"""The command line: help, exit codes, the refusals by name, the flags that are (and are not) on the surface — the stock scripts' own
flags plus the mode word and the one kit extension (--input/--out_dir)."""
import os
import re
import subprocess
import sys

import pytest

from progen2_opt import cli
from .conftest import run_package, subprocess_env


def _run(*args, **env):
    r = subprocess.run([sys.executable, "-m", "progen2_opt", *args], env=subprocess_env(env), capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_help_and_unknown():
    rc, out = _run("--help")
    assert rc == 0 and out.startswith("usage: python -m progen2_opt")
    assert "--model progen2-large" in out and "--input FILE --out_dir DIR" in out and "progen2-bfd90 is accepted for progen2-BFD90" in out
    assert "--variant" not in out and "--det" not in out and "PROGEN2_OPT" not in out                # the retired surface is not documented
    assert _run()[0] == cli.EXIT_USAGE and _run("bogus")[0] == cli.EXIT_USAGE
    rc, out = _run("sample", "--help")
    assert rc == 0 and out.startswith("usage: python -m progen2_opt")


def test_check_lines():
    rc, out = _run("check", "--mode", "exact", "--model", "progen2-small")
    assert "[progen2-opt] DRY-RUN mode=exact variant=small kit=sample=serving:pipeline_v0_4 score=forward:v0_score_r3_1+v0_ew" in out
    assert "on=sample:oneread_mmap+sampler_exact+resident_rotary+static_kv,score:rotary_tables+one_forward_per_direction+host_pipeline+ew:gelu" in out
    from progen2_opt import stack
    if stack.gpu_info().get("name"):                                                                # a CUDA box: nothing about it is a refusal (another stack, another card are NOTES)
        assert "would_refuse=" not in out, out
    else:                                                                                           # a box without a CUDA device: the levers cannot run — the dry run says the verbs refuse
        assert f"would_refuse={stack.NO_CUDA_REFUSAL!r}" in out, out
    rc, out = _run("check", "--mode", "off", "--model", "progen2-small")                          # check_pins always runs: its rc decides (this box may not be the pinned stack)
    m = re.search(r"check_pins rc=(\d+)", out)
    assert m, out
    assert int(m.group(1)) != 2, ("check_pins exited 2 (argparse: unrecognized/missing arguments) — cmd_check() is sending it a "
                                   "flag it doesn't accept:\n" + out)                            # a real, non-tautological guard: whatever this box's stack is, the ARGS must always parse
    assert "DRY-RUN mode=off variant=small kit=stock" in out and rc == (0 if m.group(1) == "0" else cli.EXIT_FAIL), out
    rc, out = _run("check", "--mode", "exact", "--model", "progen2-bfd90")                          # the lowercase alias of progen2-BFD90
    assert "DRY-RUN mode=exact variant=bfd90 " in out, out


def test_check_pins_argv_is_accepted_and_healthy_tree_exits_0():
    """The exact argv cmd_check() sends (stack.check_pins_argv(), the one source of truth for it) is valid check_pins.py args —
    a regression guard for a since-removed flag (--self) that once made every `progen2-opt check` fail unconditionally,
    regardless of pinning. Runs check_pins.py directly, skipping the box-dependent interpreter/packages checks (this sandbox is
    not the pinned stack and that is a fact about the box, not the tree): the stock-files check alone must exit 0."""
    from progen2_opt import stack
    script = os.path.join(stack.tree_home(), "stock", "check_pins.py")
    cmd = [stack.python(), "-I", script] + stack.check_pins_argv() + ["--skip", "interpreter", "--skip", "packages"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (r.returncode, cmd, r.stdout, r.stderr)


def test_refusals_by_name(tmp_path):
    rc, out = _run("sample", "--mode", "turbo", "--model", "progen2-small")                      # an unknown mode word: the modes named, exit 2
    assert rc == cli.EXIT_USAGE and "unknown mode 'turbo' (choose from exact, off)" in out, out
    for word in ("fast", "big"):                                                              # the family's other mode words: not shipped for this model — refused by name, never mapped onto exact
        rc, out = _run("score", "--mode", word, "--model", "progen2-small")
        assert rc == cli.EXIT_USAGE and f"--mode {word}: not shipped for this model (modes: exact | off)" in out, out
    rc, out = _run("sample", "--model", "progen2-huge")                                          # an unknown size: the stock's own choices named, exit 2
    assert rc == cli.EXIT_USAGE and "invalid choice 'progen2-huge'" in out and "progen2-BFD90" in out and "progen2-xlarge" in out, out
    for verb in ("sample", "score"):                                                             # the retired escape word: unknown to every verb (a mode is all of its levers: nothing runs under its name with a subset)
        rc, out = _run(verb, "--model", "progen2-small", "--allow-partial")
        assert rc == cli.EXIT_USAGE and "unrecognized arguments: --allow-partial" in out, (verb, out)
    for flags, named in (("--fp16", "false"), ("--rng-deterministic", "False"), ("--device", "cuda:1"), ("--sanity", "true")):   # the stock's own settings are not refusals under exact (test_routes: where they go); only the cpu device is (no lever runs there)
        rc, out = _run("score", "--mode", "exact", "--model", "progen2-small", flags, named, PROGEN2_WEIGHTS=str(tmp_path / "none"))
        assert f"NOT ACTIVE: {flags}" not in out and "not a setting mode exact serves" not in out, out
    rc, out = _run("score", "--mode", "exact", "--model", "progen2-small", "--device", "cpu", PROGEN2_WEIGHTS=str(tmp_path / "none"))
    assert rc == cli.EXIT_NOT_ACTIVE and "[progen2-opt] NOT ACTIVE: --device cpu: the levers run on CUDA" in out and "`--mode off --device cpu` runs the stock on the cpu" in out, out


def test_input_is_the_one_kit_extension(tmp_path):
    items = tmp_path / "items.jsonl"; items.write_text('{"item_id": "a"}\n')
    rc, out = _run("sample", "--mode", "off", "--model", "progen2-small", "--input", str(items), "--out_dir", str(tmp_path / "o"), "--t", "0.5", "--context", "1M")
    assert rc == cli.EXIT_USAGE and "--input names them per item: --t --context cannot be given on the command line with --input" in out, out
    rc, out = _run("score", "--mode", "off", "--model", "progen2-small", "--input", str(items), "--out_dir", str(tmp_path / "o"), "--context", "1M2")
    assert rc == cli.EXIT_USAGE and "--input names them per item: --context" in out, out
    rc, out = _run("sample", "--mode", "off", "--input", str(items))                              # the job form needs its out dir
    assert rc == cli.EXIT_USAGE and "--out_dir" in out, out
    rc, out = _run("score", "--mode", "off", "--out_dir", str(tmp_path / "o"))                    # a single call writes nothing: --out_dir alone is refused, on both commands
    assert rc == cli.EXIT_USAGE and "score --out_dir goes with --input" in out, out
    rc, out = _run("sample", "--mode", "exact", "--model", "progen2-small", "--out_dir", str(tmp_path / "o"))
    assert rc == cli.EXIT_USAGE and "sample --out_dir goes with --input" in out, out
    rc, out = _run("sample", "--mode", "off", "--input", str(tmp_path / "none.jsonl"), "--out_dir", str(tmp_path / "o"))
    assert rc == cli.EXIT_USAGE and "no such file" in out, out
    old = tmp_path / "old.jsonl"; old.write_text('{"item_id": "a", "seed": 7}\n')                # the retired item key: refused by name, exit 2
    rc, out = _run("sample", "--mode", "off", "--input", str(old), "--out_dir", str(tmp_path / "o"))
    assert rc == cli.EXIT_USAGE and "'seed' is retired" in out and "rng_seed" in out, out


def test_kit_side_names_are_no_switch():
    rc, out, _ = run_package(["check", "--mode", "exact", "--model", "progen2-small"], {"PROGEN2_KIT_HOME": "/x/opt", "KIT_HOME_VAR": "PROGEN2_KIT_HOME", "KIT_PYTHONPATH": "opt", "PROGEN2_KIT": "v0_ew"}, gate_ok=True)
    assert "would_refuse=None" in out or "kit switch" not in out, out                          # names the kits never read on the route are ignored (no prefix rule)


def test_no_environment_variable_selects_a_mode_or_a_size():
    """PROGEN2_OPT / PROGEN2_VARIANT are nothing to the package: the mode is --mode (default exact), the size is --model (the stock's default)."""
    rc, out, _ = run_package(["check", "--mode", "off"], {"PROGEN2_OPT": "exact", "PROGEN2_VARIANT": "small"}, gate_ok=True)
    assert "DRY-RUN mode=off variant=None kit=stock" in out, out                                  # neither variable is read: mode off as given, no size named
    rc, out, _ = run_package(["check"], {"PROGEN2_OPT": "off"}, gate_ok=True)
    assert "DRY-RUN mode=exact variant=None" in out, out                                          # the package default, whatever the variable says


# The surface, per command: the stock scripts' own flags + --mode / --model + the one flag upstream lacks (--input / --out_dir). Anything
# else is unknown to argparse; `sample`, `score` and `check` are the only commands past `install` (run.sh's own).
SAMPLE_PY = {"--model", "--device", "--rng-seed", "--rng-deterministic", "--p", "--t", "--max-length", "--num-samples", "--fp16", "--context", "--sanity"}   # sample.py main() L112-122
LIKELIHOOD_PY = {"--model", "--device", "--rng-seed", "--rng-deterministic", "--fp16", "--context", "--sanity"}                                                 # likelihood.py main() L122-128
SURFACE = {
    "sample": SAMPLE_PY | {"--mode", "--input", "--out_dir"},
    "score": LIKELIHOOD_PY | {"--mode", "--input", "--out_dir"},
    "check": {"--mode", "--model"},
}
# switches folded into a default or retired (spelled without their dashes; the test adds them): refused by every command
REMOVED = ("variant", "det", "served", "batch", "no-pins", "python", "boot-timeout", "timeout", "route", "recipe", "graphs", "witness", "step-clock", "allow-partial",
           "config", "json", "kit-batches", "socket")
_ARGV = {"sample": ["--model", "progen2-small"], "score": ["--model", "progen2-small"], "check": ["--model", "progen2-small"]}


def test_stock_flag_sets_are_the_scripts_own(tree):
    """The flag names above are read back from the stock files' own `parser.add_argument('--…')` lines: the surface is theirs, not a copy that can drift."""
    for script, want in (("sample.py", SAMPLE_PY), ("likelihood.py", LIKELIHOOD_PY)):
        src = open(os.path.join(tree, "stock", "src", "progen2", script)).read()
        assert set(re.findall(r"add_argument\('(--[a-z0-9-]+)'", src)) == want, script


def _options(command):
    """The long options a command's parser accepts (captured from argparse's own registry, no parse)."""
    import argparse
    seen = {}
    real = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        seen["opts"] = {o for a in self._actions for o in a.option_strings if o.startswith("--")} - {"--help"}
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    try:
        try:
            cli.COMMANDS[command](_ARGV[command])
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real
    return seen["opts"]


@pytest.mark.parametrize("command", sorted(SURFACE))
def test_surface_is_exactly_the_documented_options(command):
    assert _options(command) == SURFACE[command], command


@pytest.mark.parametrize("name", REMOVED)
def test_removed_switches_are_refused(name):
    for command, argv in _ARGV.items():
        rc, out = _run(command, *argv, "--" + name, "5")
        assert rc == 2 and ("unrecognized arguments" in out or "invalid choice" in out), (command, name, out[-300:])


def test_the_server_verbs_and_the_config_word_are_gone():
    """Generation runs in the calling process: there is no server to warm, stop or ask — `warm` / `serve` are unknown commands (exit 2) — and no
    per-card config: `--config` is unknown to every command (the card is read from the device at activation)."""
    assert set(cli.COMMANDS) == {"sample", "score", "check"}
    for gone in (["warm", "--model", "progen2-small"], ["serve", "--model", "progen2-small", "stop"], ["serve", "--model", "progen2-small", "status"]):
        rc, out = _run(*gone)
        assert rc == cli.EXIT_USAGE and f"unknown command {gone[0]!r}" in out, (gone, out[-300:])
    for command, argv in _ARGV.items():
        rc, out = _run(command, "--config", "h100", *argv)
        assert rc == cli.EXIT_USAGE and "unrecognized arguments: --config" in out, (command, out[-300:])


def test_run_sh_exit_codes_by_name(tmp_path):
    """`run.sh` on a python where the kit package is not importable: `NOT ACTIVE: kit package not installed` and exit 3 (nothing can
    activate); on this python, a flag off the surface (`--config`, gone with the per-card configs) is the package's usage error, exit 2."""
    import stat
    from .conftest import TREE
    bare = tmp_path / "bare_python.sh"                                    # the same interpreter without site-packages: the package is not importable there
    bare.write_text(f"#!/bin/sh\nexec {sys.executable} -S \"$@\"\n"); bare.chmod(bare.stat().st_mode | stat.S_IEXEC)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PROGEN2_", "PYTHON"))}
    env["PROGEN2_PYTHON"] = str(bare)
    p = subprocess.run(["bash", os.path.join(TREE, "run.sh"), "check"], capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    assert p.returncode == cli.EXIT_NOT_ACTIVE == 3 and "[progen2-opt] NOT ACTIVE: kit package not installed" in p.stderr and "pip install -e" in p.stderr, (p.returncode, p.stderr[-400:])
    env2 = dict(subprocess_env(), PROGEN2_PYTHON=sys.executable)           # this interpreter: the package importable, every flag the package's
    q = subprocess.run(["bash", os.path.join(TREE, "run.sh"), "check", "--config", "h100", "--model", "progen2-small"], capture_output=True, text=True, env=env2, cwd=str(tmp_path), timeout=120)
    assert q.returncode == cli.EXIT_USAGE == 2 and "unrecognized arguments: --config" in q.stderr, (q.returncode, q.stderr[-300:])
    import glob
    assert glob.glob(os.path.join(TREE, "configs", "*.env")) == []                # no per-card config ships: the card is read from the device
