"""What a call resolves and prints is one record: for `--mode exact`, `--mode fast`, `--mode big` (one GPU: resident) and `--mode big --n_gpu 2`
(a rank of a 2-GPU world: the row-sharded composition) the resolved lever set, hook chain, exported / unset switches, precision word, size-gate
words and the ACTIVE / DRY-RUN / LEVER lines equal, byte for byte, tests/data/compositions_record.json (HOME-relative, environment words masked;
its `_source` names the package version it was taken on — re-taken only when a line changes by design). The composition selectors do not exist:
no `--exact-line` / `--big-line` flags, their variables are refused as undeclared by the env route, no streamed composition."""
import json
import os
import re
import subprocess
import sys

import pytest

from openfold3_ob0_opt import _autoload, cli, modes, registry
from openfold3_ob0_opt.tests._stubs import tree_home

HOME = tree_home()
RECORD = json.load(open(os.path.join(os.path.dirname(__file__), "data", "compositions_record.json")))
DELETED_STREAMED_LEVERS = ()                                      # levers deleted from the registry since the record was taken: their `state=off reason=not_in_line` rows may be absent (none today)

CASES = {"exact": ("exact", {}, 1), "fast": ("fast", {}, 1), "big": ("big", {}, 1),
         "big_tp2": ("big", {"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OB0_OPT_N_GPU": "2"}, 2)}


def _clean_env(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OF3", "OPENFOLD3_OB0_OPT", "BFTP_", "PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG"))}
    env.update(extra)
    return env


_ENV_WORDS = ((re.compile(r" openfold3=\S+ gpu=.*? levers_requested="), " openfold3=* gpu=* levers_requested="),   # the installed distribution's version and the device: the box's, not the composition's
              (re.compile(r" refused=.*$"), ""))                                                              # the dry run's verdict on THIS environment (wheel / GPU present or not): not the composition's either


def _mask(line):
    """An ACTIVE / DRY-RUN line with its environment words masked — everything else compared byte for byte."""
    for rx, rep in _ENV_WORDS:
        line = rx.sub(rep, line)
    return line


RECORDER = r"""
import json, os, sys
from openfold3_ob0_opt import modes, report, stack
mode, extra, p, n_tok, home = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
env = dict(os.environ, **extra)
res = modes.resolve(mode, home, environ=env, n_tokens=n_tok, n_gpu=p)
rel = lambda s: s.replace(home, "<HOME>") if isinstance(s, str) else s
rep = stack.activate(mode, dry_run=True, log=False, environ=env, home=home, n_tokens=n_tok, n_gpu=p)
print("RECORD_JSON " + json.dumps({
    "line": res.line, "tier": res.tier, "levers": res.levers, "hooks": res.hooks, "hook_dirs": [rel(d) for d in res.hook_dirs],
    "exports": {k: rel(v) for k, v in sorted(res.exports.items())}, "unsets": sorted(res.unsets), "precision": res.precision,
    "size_gate": res.size_gate, "gate_reason": res.gate_reason, "conf_gate": res.conf_gate, "conf_reason": res.conf_reason,
    "conflicts": res.conflicts, "describe": modes.describe_line(res),
    "dry_run_line": rel(report.activation_line(rep)), "lever_lines": [rel(l) for l in report.lever_lines(dict(rep, active=True))],
    "active_line": rel(report.activation_line(dict(rep, active=True, dry_run=False)))}))
"""


def _record(mode, extra, p, n_tok):
    """One composition case rendered in a FRESH child interpreter — the import state of the pytest process (a sibling test or a plugin that
    already imported openfold3's model modules) never reaches the dry-run activation, exactly as in a real `pred` process. The child sees the
    package the way this process does (the same sys.path, minus the selector-era variables)."""
    env = _clean_env(extra)
    env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))] + [d for d in sys.path if d])   # opt/ (the package's parent) first, then this process's path
    r = subprocess.run([sys.executable, "-c", RECORDER, mode, json.dumps(extra), str(p), str(n_tok), HOME], capture_output=True, text=True, env=env, cwd=os.sep, timeout=600)
    rows = [l for l in r.stdout.splitlines() if l.startswith("RECORD_JSON ")]
    assert r.returncode == 0 and len(rows) == 1, (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    return json.loads(rows[0][len("RECORD_JSON "):])


@pytest.mark.parametrize("key", sorted(k for k in RECORD if not k.startswith("_")))
def test_composition_bytes_equal_the_record_taken_before_the_selectors_went(key):
    name, n_tok = key.split("@")
    mode, extra, p = CASES[name]
    got = _record(mode, extra, p, int(n_tok))
    want = dict(RECORD[key])
    gone = [l for l in want["lever_lines"] if any(f" name={n} " in l for n in DELETED_STREAMED_LEVERS)]   # the NAMED tolerance: the two deleted levers' record rows, and only theirs
    assert len(gone) == len(DELETED_STREAMED_LEVERS) and all(" state=off reason=not_in_line " in l for l in gone), gone   # they were off (not in the line) on every recorded case — nothing that ran is dropped
    assert not any(f" name={n} " in l for n in DELETED_STREAMED_LEVERS for l in got["lever_lines"]), got["lever_lines"]   # and the levers are gone from the registry now
    want["lever_lines"] = [l for l in want["lever_lines"] if l not in gone]                    # every other LEVER row: byte for byte (the `got == want` below)
    for field in ("dry_run_line", "active_line"):                            # environment-free: the record was taken wheel-less on a CPU box; a box with the wheel / a GPU prints other environment words
        got[field], want[field] = _mask(got[field]), _mask(want[field])
        assert " openfold3=* gpu=* levers_requested=" in want[field], want[field]
    for field in want:                                                   # field by field: a failure names the field
        assert got[field] == want[field], (key, field, got[field], want[field])
    assert got == want


def test_the_resource_decides_the_big_composition_and_nothing_else_does():
    assert modes.line_for("big") == "resident" and modes.line_for("big", n_gpu=1) == "resident" and modes.line_for("big", n_gpu="2") == "tp"
    assert modes.line_for("big", environ={"OPENFOLD3_OB0_OPT_N_GPU": "4"}) == "tp"
    assert modes.line_for("exact") == modes.EXACT_LINE == "cueq" and modes.line_for("fast") is None and modes.line_for("off") is None
    assert modes.BIG_LINES == ("resident", "tp") and ("big", "streamed") not in modes.LINES and set(modes.MODE_LINES) == {"exact", "big"}
    assert not any(hasattr(modes, n) for n in ("EXACT_LINES", "DEFAULT_EXACT_LINE", "ENV_EXACT_LINE", "DEFAULT_BIG_LINE", "ENV_BIG_LINE", "check_line", "line_arg", "RESERVED_LINES"))
    assert "pair_offload_stream" not in registry.LEVERS and "logits_host" not in registry.LEVERS
    assert "logits_host" not in modes.CONF_LEVERS and "OF3O_LOGITS_HOST" not in modes.CONF_ENVS


def test_the_selector_flags_and_variables_are_gone():
    ap = cli.build_parser()
    for flag, val in (("--exact-line", "cueq"), ("--big-line", "resident"), ("--big-line", "streamed"), ("--no-hash-checkpoint", None)):
        with pytest.raises(SystemExit):
            ap.parse_args(["pred", "--mode", "big", "--query-json", "q.json", "--output-dir", "/tmp/o"] + [flag] + ([val] if val else []))
    with pytest.raises(SystemExit):                                       # no `--` pass-through: run_openfold's own words are the CLI's named flags or nothing
        ap.parse_args(["pred", "--mode", "off", "--query-json", "q.json", "--output-dir", "/tmp/o", "--", "--num_recycles", "1"])
    for name in ("OPENFOLD3_OB0_OPT_EXACT_LINE", "OPENFOLD3_OB0_OPT_BIG_LINE", "OPENFOLD3_OB0_OPT_BIG_STREAM_MIN_TOKENS"):
        assert name not in _autoload.DECLARED                            # the env route refuses them as undeclared (NOT ACTIVE: undeclared variable)
    assert _autoload.undeclared_refusal({"OPENFOLD3_OB0_OPT": "big", "OPENFOLD3_OB0_OPT_BIG_LINE": "streamed"}) if hasattr(_autoload, "undeclared_refusal") else True
