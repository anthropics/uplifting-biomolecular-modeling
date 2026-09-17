"""`--mode big --n_gpu P>1`: the P rank processes `pred` launches are separate interpreters, so anything a rank orders by Python `str`
hash (set / dict-of-set iteration in a featuriser) is byte-equal across ranks only when every rank runs under ONE `PYTHONHASHSEED`. The
family launcher (opt_core.mem.rowpair.launch.run_rank_processes, the call rowpair.run_ranks makes) states that seed in every rank's
environment (opt_core.mem.rowpair.rankdata): a `PYTHONHASHSEED` naming a seed in the launching environment is kept (source `inherited`),
else the core's default is exported to all ranks (source `default`); it prints `[protenix-v1-opt] RANKENV hashseed=<v>
source=default|inherited ranks=P` once per launch on stderr (the tag is the kit's: run_ranks hands the launcher its base environment with
launch.ENV_TAG set).

Driven through rowpair.run_ranks over the REAL launcher on a CPU box: the rank command line is replaced by a one-line printer of the seed the
rank interpreter sees, the launcher admits P=2 without visible devices and isolates none; each rank's transcript is
<out_dir>/rowpair/rank<r>.log (rowpair.rank_log). No torch group, no GPU.
"""
import os
import sys

import pytest

from opt_core.mem.rowpair import launch as L
from opt_core.mem.rowpair import rankdata as RD

from protenix_v1_opt import report as R
from protenix_v1_opt import rowpair

SEED_PRINTER = [sys.executable, "-c", "import os;print('seed', os.environ.get('PYTHONHASHSEED'))"]
KIT_ARGS = ["--mode", "big", "--n_gpu", "2", "--input", "x.json", "--out_dir", "o"]


@pytest.fixture
def cpu_ranks(monkeypatch):
    """rowpair.run_ranks on a CPU box: the rank command is SEED_PRINTER; the real launcher runs with cpu_ok (no visible-device rule, no device
    census) and without device isolation; no rank environment or line tag is inherited from the test process."""
    for k in list(os.environ):
        if k.startswith("ROWPAIR_") or k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(rowpair, "rank_argv", lambda kit_args: list(SEED_PRINTER))
    real = L.run_rank_processes
    monkeypatch.setattr(L, "run_rank_processes", lambda *a, **kw: real(*a, **dict(kw, cpu_ok=True, isolate_devices=False)))
    return rowpair.run_ranks


def rank_seeds(out_dir: str, P: int) -> list:
    """The one `seed <value>` line of each rank's transcript, in rank order."""
    vals = []
    for r in range(P):
        with open(rowpair.rank_log(out_dir, r), encoding="utf-8") as fh:
            lines = [ln.split() for ln in fh if ln.startswith("seed ")]
        assert len(lines) == 1, (r, lines)
        vals.append(lines[0][1] if len(lines[0]) > 1 else "")
    return vals


def rankenv_lines(err: str) -> list:
    """The launcher's RANKENV lines in a captured stderr."""
    return [ln for ln in err.splitlines() if " RANKENV " in f" {ln} "]


def test_every_rank_runs_under_one_hash_seed_when_the_caller_sets_none(cpu_ranks, monkeypatch, tmp_path, capfd):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    recs = cpu_ranks(2, KIT_ARGS, str(tmp_path))
    assert [rec["rc"] for rec in recs] == [0, 0]
    seeds = rank_seeds(str(tmp_path), 2)
    assert seeds[0] == seeds[1] and seeds[0].isdigit(), seeds              # one value, a real seed (not None / random), on both ranks
    assert seeds[0] == RD.hash_seed(os.environ)[0] == RD.HASHSEED_DEFAULT   # the core's default, the value its census names
    line = RD.rankenv_line(R.TAG, os.environ, 2)                            # the core's own words under the kit's tag: `[protenix-v1-opt] RANKENV hashseed=0 source=default ranks=2`
    assert rankenv_lines(capfd.readouterr().err) == [line] and "source=default" in line and line.startswith(R.PREFIX + " RANKENV "), line


def test_a_callers_hash_seed_reaches_every_rank_unchanged(cpu_ranks, monkeypatch, tmp_path, capfd):
    monkeypatch.setenv("PYTHONHASHSEED", "123")
    recs = cpu_ranks(2, KIT_ARGS, str(tmp_path))
    assert [rec["rc"] for rec in recs] == [0, 0]
    assert rank_seeds(str(tmp_path), 2) == ["123", "123"]
    line = RD.rankenv_line(R.TAG, os.environ, 2)
    assert rankenv_lines(capfd.readouterr().err) == [line] and "hashseed=123 source=inherited ranks=2" in line, line
