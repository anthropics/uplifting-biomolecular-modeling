"""run.sh reads its two-token options safely: `--weights`, `--model_name` (install) and `--mode` (score) given as the LAST token, with no
value after them, end the script with the usage text and exit code 2 — they never spin (a `shift 2` with one argument left shifts nothing,
so an unguarded loop would read the same token forever)."""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_SH = os.path.normpath(os.path.join(HERE, "..", "..", "..", "run.sh"))


def _run(args, tmp_path):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "TMPDIR": str(tmp_path), "LANG": "C"}
    return subprocess.run(["bash", RUN_SH, *args], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(not os.path.isfile(RUN_SH), reason="run.sh is not beside this package (installed copy)")
@pytest.mark.parametrize("args", [["install", "--weights"], ["install", "--model_name"], ["install", "--weights=/nonexistent", "--model_name"], ["score", "--mode"]])
def test_a_two_token_option_without_its_value_is_usage_not_a_spin(args, tmp_path):
    try:
        r = _run(args, tmp_path)
    except subprocess.TimeoutExpired:                      # the unguarded form: `shift 2` fails, the loop re-reads the same token
        pytest.fail(f"run.sh {' '.join(args)} did not return (option loop spins on a trailing {args[-1]})")
    assert r.returncode == 2, (args, r.returncode, r.stderr[-400:])
    assert "run.sh install" in r.stderr and "run.sh score" in r.stderr      # the usage text (the script's header lines)


def test_bash_reads_the_script(tmp_path):
    r = subprocess.run(["bash", "-n", RUN_SH], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
