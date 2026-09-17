"""``mem.rowpair.launch.run_rank_processes`` — teardown after a rank fails. A rank that EXITS with a non-zero status (an item failed on it) is
announced (``[<tag>] RANKEXIT rank=<r> exitcode=<c> grace_s=<g>`` on the launcher's stderr) and its peers run on for ``fail_grace_s``, so the
output rank still writes what it has: rank 1 exits 3 at once, rank 0 sleeps 2 s, writes a file and exits — the file exists, the launch raises
``RankFailed`` naming rank 1 / 3, rank 0's record is ok. A rank that DIES ON A SIGNAL — or whose wrapper reports the signal as 128+n — tears
its peers down at once (rank 0's 30 s sleep is cut, its file never appears, its record says torn_down). Past ``fail_grace_s`` the peers are
torn down and the event still names the rank that exited. Two real interpreters per case, CPU only, seconds.

Run: ``python -m pytest tests/test_rowpair_launch_05183.py -q -rfE``.
"""
from __future__ import annotations

import os
import signal
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from opt_core.mem.rowpair import launch  # noqa: E402


def _env():
    return {k: v for k, v in os.environ.items() if k not in launch.TORCHRUN_NAMES and k != launch.ENV_TAG}   # no inherited tag: the line reads [rowpair]


def _rank0_writes(path: str, after_s: float, rc: int = 0) -> list:
    return [sys.executable, "-c", f"import sys, time; time.sleep({after_s}); open({path!r}, 'w').write('last item'); sys.exit({rc})"]


def _launch(tmp_path, capsys, argv_of, **kw):
    t0 = time.monotonic()
    with pytest.raises(launch.RankFailed) as ei:
        launch.run_rank_processes(2, argv_of=argv_of, mode="big", log_dir=str(tmp_path / "logs"), cpu_ok=True, run_timeout_s=120, env=_env(), **kw)
    wall = time.monotonic() - t0
    exits = [ln for ln in capsys.readouterr().err.splitlines() if " RANKEXIT " in ln]
    return ei.value, {r["rank"]: r for r in launch.last_run()}, wall, exits


def test_a_rank_that_exits_nonzero_lets_the_output_rank_finish_writing(tmp_path, capsys):
    out = str(tmp_path / "item.txt")
    e, recs, wall, exits = _launch(tmp_path, capsys, lambda r: _rank0_writes(out, 2.0) if r == 0 else [sys.executable, "-c", "import sys; sys.exit(3)"])
    assert (e.event, e.rank, e.exitcode) == ("rank_failed", 1, 3), (e.event, e.rank, e.exitcode)
    assert os.path.exists(out) and open(out).read() == "last item"                            # rank 0 was NOT killed mid-write
    assert recs[0]["ok"] is True and recs[0]["rc"] == 0 and recs[0]["error"] is None, recs
    assert recs[1]["rc"] == 3 and recs[1]["error"] == "rank_failed", recs
    assert exits == [f"[rowpair] RANKEXIT rank=1 exitcode=3 grace_s={launch.FAIL_GRACE_S:g}"], exits    # announced once, at detection
    assert 2.0 <= wall < 30.0, wall


def test_every_rank_exiting_nonzero_names_the_first_and_keeps_every_transcript(tmp_path, capsys):
    out = str(tmp_path / "item.txt")
    e, recs, wall, exits = _launch(tmp_path, capsys, lambda r: _rank0_writes(out, 3.0, rc=3) if r == 0 else [sys.executable, "-c", "import sys; print('rank1 refused', flush=True); sys.exit(3)"])
    assert (e.event, e.rank, e.exitcode) == ("rank_failed", 1, 3), (e.event, e.rank, e.exitcode)
    assert os.path.exists(out)
    assert recs[0]["rc"] == 3 and recs[0]["error"] == "rank_failed" and recs[1]["rc"] == 3, recs
    assert "rank1 refused" in open(recs[1]["log"]).read()
    assert len(exits) == 1, exits
    assert wall < 30.0, wall


@pytest.mark.parametrize("death,code", [("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", -signal.SIGKILL),
                                        ("import sys; sys.exit(137)", 137)],                    # a wrapper (bash -c, conda run, ...) reporting its child's SIGKILL as 128+9
                         ids=["signal", "128+n"])
def test_a_rank_that_dies_on_a_signal_tears_the_rest_down_at_once(tmp_path, capsys, death, code):
    out = str(tmp_path / "item.txt")
    e, recs, wall, exits = _launch(tmp_path, capsys, lambda r: _rank0_writes(out, 30.0) if r == 0 else [sys.executable, "-c", death])
    assert (e.event, e.rank, e.exitcode) == ("rank_failed", 1, code), (e.event, e.rank, e.exitcode)
    assert not os.path.exists(out)                                                              # rank 0 was torn down inside its 30 s sleep
    assert recs[0]["rc"] == -signal.SIGTERM and recs[0]["error"] == "torn_down", recs
    assert exits == [], exits                                                                   # no grace announced: nothing more comes of this launch
    assert wall < 20.0, wall


@pytest.mark.parametrize("kw", [{"fail_grace_s": 1.0}, {"fail_grace_s": None, "finish_grace_s": 1.0}], ids=["fail_grace_s", "finish_grace_s_fallback"])
def test_past_the_grace_the_peers_are_torn_down_and_the_exited_rank_is_still_the_event(tmp_path, capsys, kw):
    out = str(tmp_path / "item.txt")
    e, recs, wall, exits = _launch(tmp_path, capsys, lambda r: _rank0_writes(out, 30.0) if r == 0 else [sys.executable, "-c", "import sys; sys.exit(3)"], **kw)
    assert (e.event, e.rank, e.exitcode) == ("rank_failed", 1, 3), (e.event, e.rank, e.exitcode)
    assert not os.path.exists(out)
    assert recs[0]["rc"] == -signal.SIGTERM and recs[0]["error"] == "torn_down", recs
    assert exits == ["[rowpair] RANKEXIT rank=1 exitcode=3 grace_s=1"], exits
    assert 1.0 <= wall < 20.0, wall
