"""The fast line runs on the stock runner yaml on the CLI route (Line.runner_yaml, composed under by cli.row_yaml; its Lightning word bf16-mixed is
upstream's spelling of the fast composition's precision, modes.FAST_PRECISION = bf16 — precision.py sets it where the Trainer reads it on every route,
test_precision_param.py); the big lines' word is read from their runner yaml (stock_pred.effective_precision — the one reader); the resolution records
the word and the ACTIVE line prints it; there is no precision switch — OPENFOLD3_OB0_OPT_PRECISION is an undeclared name the .pth hook refuses."""
import os
import subprocess
import sys

from openfold3_ob0_opt import cli, modes, report, stock_pred
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
CLEAN = {k: v for k, v in os.environ.items() if not k.startswith(modes.SWITCH_PREFIXES + ("OPENFOLD3_OB0_OPT",))}


def test_the_fast_line_runs_on_the_stock_runner_yaml():
    fast = modes.LINES[("fast", None)]
    assert fast.runner_yaml == modes.STOCK_YAML                                              # it differs from stock by its levers alone
    assert cli.line_member(fast, "fast") == modes.STOCK_YAML == cli.line_member(None, "fast")
    assert cli.row_yaml(HOME, fast, mode="fast") == os.path.join(HOME, modes.STOCK_YAML)
    assert stock_pred.effective_precision(os.path.join(HOME, modes.STOCK_YAML)) == "bf16-mixed"     # the member's Lightning word: upstream's spelling of the composition's precision (precision.BF16_MIXED)
    res = modes.resolve("fast", HOME, environ=CLEAN)
    assert res.precision == modes.FAST_PRECISION == "bf16"                                    # the composition's word, never a switch and never a file's
    line = report.activation_line({"active": True, "mode": "fast", "line": None, "openfold3_version": _stubs.PIN, "precision": res.precision, "levers_requested": [], "hooks": []})
    assert " precision=bf16 " in line or line.endswith(" precision=bf16")
    big = modes.resolve("big", HOME, environ=CLEAN)                                        # the big lines: their runner yaml's word (the one reader)
    assert big.precision == stock_pred.effective_precision(os.path.join(HOME, modes.LINES[("big", big.line)].runner_yaml)) == "bf16-mixed"
    assert modes.resolve("exact", HOME, environ=CLEAN).precision is None   # exact: the stock configuration by det level, no word of its own


def test_there_is_no_precision_switch():
    assert not any("--precision" in getattr(act, "option_strings", ()) for act in cli.build_parser()._actions)
    for sub in [act for act in cli.build_parser()._actions if act.__class__.__name__ == "_SubParsersAction"]:
        for name, sp in sub.choices.items():
            assert not any("--precision" in getattr(act, "option_strings", ()) for act in sp._actions), name
    code = "import os; from openfold3_ob0_opt import _autoload; _autoload.install(dict(os.environ))"
    env = {**CLEAN, "OPENFOLD3_OB0_OPT": "fast", "OPENFOLD3_OB0_OPT_PRECISION": "bf16", "PYTHONPATH": os.pathsep.join(sys.path)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "undeclared variable(s) OPENFOLD3_OB0_OPT_PRECISION" in r.stderr, r.stderr
