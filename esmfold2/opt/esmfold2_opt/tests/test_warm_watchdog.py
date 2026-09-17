"""warm runs one pred in a subprocess with the transcript teed line by line (opt_core.process.run_logged, no deadline): every line kept,
pred's exit rule followed (pred's own code: 3 = levers not active), the driver file set counted."""
import os
import pathlib
import sys
import time

import pytest

from esmfold2_opt import warm


def _pid_gone(pid: int, wait_s: float = 5.0) -> bool:
    for _ in range(int(wait_s / 0.05)):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_run_logged_normal_completion_keeps_every_line():
    lines = []
    rc = warm.run_logged(["bash", "-c", "echo a; echo b 1>&2; exit 3"], lines.append)
    assert rc == 3 and sorted(ln.strip() for ln in lines) == ["a", "b"]


def test_warm_counts_the_driver_file_set(tmp_path, monkeypatch):
    """warm counts the predictions where pred writes them (outputs.py: <out_dir>/cif_all/*.cif) and reports PASS on an applied run."""
    from esmfold2_opt import outputs
    items = tmp_path / "items.json"
    items.write_text('[{"id": "w", "sequences": [{"type": "protein", "id": "A", "sequence": "MKV"}]}]')
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/bash\n"                                         # a pred stand-in: the activation lines, then the driver file set
                           "while [ $# -gt 0 ]; do [ \"$1\" = --out_dir ] && OUT=$2; shift; done\n"
                           "echo '[esmfold2-opt] ACTIVE mode=fast variant=fast server_mode=opt14_msa'\n"
                           "echo '[esmfold2-opt] APPLIED model#0 mode=fast variant=fast'\n"
                           f"mkdir -p $OUT/{outputs.CIF_DIR} && echo data_w > $OUT/{outputs.CIF_DIR}/w__fast__s0_x0.cif\n")
    fake_python.chmod(0o755)
    monkeypatch.setattr(warm, "public_items", lambda: str(items))
    monkeypatch.setattr(sys, "executable", str(fake_python))
    res = warm.run("fast", "fast", log_path=str(tmp_path / "warm.log"), echo=False)
    assert res["status"] == "PASS" and res["predictions"] == 1 and res["exit_code"] == 0, res
    assert res["activation"].startswith("[esmfold2-opt] ACTIVE mode=fast") and res["applied"].startswith("[esmfold2-opt] APPLIED")
    assert "predictions=1" in warm.summary_line(res) and "reason=" not in warm.summary_line(res)


def test_warm_follows_pred_exit_rule(tmp_path, monkeypatch, capsys):
    """warm's exit is pred's: a pred that is NOT ACTIVE (exits 3) is warm's 3, named; a pred that exits 0 with a prediction and an APPLIED line is a PASS."""
    from esmfold2_opt import cli, outputs, stack
    items = tmp_path / "items.json"
    items.write_text('[{"id": "w", "sequences": [{"type": "protein", "id": "A", "sequence": "MKV"}]}]')
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/bash\n"                                         # a pred stand-in: NOT ACTIVE (exit 3) unless WARM_TEST_ACTIVE=1 is in its environment
                           "while [ $# -gt 0 ]; do [ \"$1\" = --out_dir ] && OUT=$2; shift; done\n"
                           "echo '[esmfold2-opt] ACTIVE mode=fast variant=fast server_mode=opt14_msa'\n"
                           "echo '[esmfold2-opt] APPLIED model#0 mode=fast variant=fast'\n"
                           f"mkdir -p $OUT/{outputs.CIF_DIR} && echo data_w > $OUT/{outputs.CIF_DIR}/w__fast__s0_x0.cif\n"
                           "[ \"${WARM_TEST_ACTIVE:-0}\" = 1 ] || exit 3\n")
    fake_python.chmod(0o755)
    monkeypatch.setattr(warm, "public_items", lambda: str(items))
    monkeypatch.setattr(sys, "executable", str(fake_python))
    res = warm.run("fast", "fast", log_path=str(tmp_path / "warm.log"), echo=False)
    assert res["status"] == "FAIL" and res["exit_code"] == 3 and res["reason"] == "pred exited 3 (levers not active: the NOT ACTIVE line names why)"
    monkeypatch.setattr(cli, "check_variant", lambda v: v)
    monkeypatch.setattr(stack, "_effective_variant", lambda v: v)
    assert cli.cmd_warm(["--mode", "fast", "--variant", "fast", "--log", str(tmp_path / "w3.log"), "--quiet"]) == cli.EXIT_NOT_ACTIVE   # pred's NOT ACTIVE is warm's 3
    monkeypatch.setenv("WARM_TEST_ACTIVE", "1")                                        # the stand-in now activates: exit 0, a prediction, an APPLIED line
    res = warm.run("fast", "fast", log_path=str(tmp_path / "warm2.log"), echo=False)
    assert res["status"] == "PASS" and res["exit_code"] == 0 and "--allow-partial" not in res["command"]
    assert cli.cmd_warm(["--mode", "fast", "--variant", "fast", "--log", str(tmp_path / "w4.log"), "--quiet"]) == cli.EXIT_OK
    with pytest.raises(SystemExit):                                                     # no opt-out flag exists: --allow-partial is a usage error of the verb
        cli.cmd_warm(["--mode", "fast", "--variant", "fast", "--log", str(tmp_path / "w5.log"), "--quiet", "--allow-partial"])


