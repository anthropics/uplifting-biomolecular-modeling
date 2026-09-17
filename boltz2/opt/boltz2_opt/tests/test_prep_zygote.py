"""prep.Zygote: every input is parsed in a FRESH child forked from a zygote that parsed nothing — the process context `boltz predict <one yaml>`
parses in (RDKit's conformer RNG is process-global and unseeded upstream: the Nth SMILES embedding of one process ≠ the first of a fresh one)."""
import os
import re
import subprocess
import sys

import pytest

from .. import prep

TARGET = "boltz2_opt.tests._prep_targets"


@pytest.fixture()
def zyg():
    z = prep.Zygote(preload=(TARGET,)).start(forked_at="test")
    yield z
    z.close()


def test_every_call_runs_in_a_fresh_child_of_the_zygote(zyg):
    from . import _prep_targets
    _prep_targets.CALLS = 99                                   # the parent's copy; the zygote imported its own (CALLS 0) and every child forks from THAT: 0 -> 1 each time
    r1 = zyg.run(f"{TARGET}:bump", tag="a")
    r2 = zyg.run(f"{TARGET}:bump", tag="b", payload={"k": [1, 2.5, None, "s"]})
    assert r1.rc == 0 and r2.rc == 0, (r1, r2)
    m1 = re.search(r"pid=(\d+) calls=(\d+)", r1.out); m2 = re.search(r"pid=(\d+) calls=(\d+)", r2.out)
    assert m1.group(2) == "1" and m2.group(2) == "1", "each child starts from the zygote's state: no call sees another call's side effects"
    assert len({int(m1.group(1)), int(m2.group(1)), os.getpid(), zyg.pid}) == 4, "two distinct children, neither the parent nor the zygote"
    assert r1.err == "err tag=a\n" and r2.out.endswith("payload={'k': [1, 2.5, None, 's']}\n"), "stdout / stderr captured verbatim; kwargs round-trip"
    assert _prep_targets.CALLS == 99, "the parent's own module is untouched"
    d = zyg.describe()
    assert d["forked_at"] == "test" and d["calls"] == 2 and d["cuda_initialized_at_fork"] is False and d["zygote_pid"] == zyg.pid


def test_a_raising_target_is_rc_1_with_its_traceback_and_a_dying_child_is_rc_2(zyg):
    r = zyg.run(f"{TARGET}:bump", tag="c", fail=True)
    assert r.rc == 1 and r.err.rstrip().endswith("ValueError: boom c") and "Traceback (most recent call last)" in r.err and "out tag=c" in r.out
    r = zyg.run("no_such_module_xyz:f")
    assert r.rc == 1 and "ModuleNotFoundError" in r.err
    r = zyg.run(f"{TARGET}:bump", tag="d", die=7)
    assert r.rc == 2 and "died without reporting (exit 7)" in r.err
    assert zyg.run(f"{TARGET}:bump", tag="e").rc == 0, "the zygote serves on after a child failed"


def test_close_is_idempotent_and_a_closed_zygote_restarts_on_use():
    z = prep.Zygote(preload=()).start(forked_at="test"); pid = z.pid
    z.close(); z.close()
    assert z.pid is None
    r = z.run(f"{TARGET}:bump", tag="again")                   # run() starts it again (first_use)
    assert r.rc == 0 and z.pid not in (None, pid) and z.forked_at == "first_use"
    z.close()


def test_the_process_wide_zygote_is_one_and_names_where_it_was_forked(capsys):
    prep.close()
    z1 = prep.zygote(); z2 = prep.zygote()
    assert z1 is z2 and z1.forked_at == "first_use" and z1.pid
    line = capsys.readouterr().err
    assert line.startswith("[boltz2-opt prep] zygote pid=") and "forked_at=first_use" in line and "cuda_initialized=False" in line
    prep.close()
    z3 = prep.start(forked_at="launch", preload=())
    assert prep.zygote() is z3 and z3.forked_at == "launch"
    prep.close(); capsys.readouterr()


def test_parse_is_run_of_parse_one_with_the_callers_kwargs(monkeypatch, zyg):
    seen = {}
    monkeypatch.setattr(zyg, "run", lambda target, **kw: seen.update(target=target, **kw) or prep.Result(0, "o", "e"))
    r = zyg.parse("/in/a.yaml", "/out/boltz_results_a", "/cache/ccd.pkl", "/cache/mols", boltz2=True, preprocessing_threads=1)
    assert r == prep.Result(0, "o", "e") and seen == {"target": prep.PARSE_ONE, "yaml": "/in/a.yaml", "out_dir": "/out/boltz_results_a", "ccd_path": "/cache/ccd.pkl",
                                                        "mol_dir": "/cache/mols", "boltz2": True, "preprocessing_threads": 1}
    assert prep.PARSE_ONE == "boltz2_opt.prep:parse_one" and prep.PRELOAD == ("boltz.main",)


def test_worker_launch_forks_the_zygote_before_anything_else():
    """worker_launch.main: prep.start is the first thing after the usage check — before the kernels / phase hooks, routes, attachments and the
    script (so the zygote has no CUDA context, no kit hook, nothing parsed)."""
    src = open(os.path.join(os.path.dirname(prep.__file__), "worker_launch.py")).read()
    body = src[src.index("def main("):]
    first = body.index('prep.start(forked_at="launch")')
    for later in ("from . import kernels", "sys.meta_path.insert", "kernels.route(", "_attach(", "runpy.run_path("):
        assert first < body.index(later), later


def test_the_worker_bases_parse_through_the_zygote_with_main_py_s_arguments():
    from .. import stack
    src_dir = os.path.join(stack.tree_dir(), "opt", "forward", "trunk_levers", "src")
    for base in ("bz_worker_lev.py", "bz_worker_levf2.py"):
        src = open(os.path.join(src_dir, base)).read()
        assert "PREP = _prep.zygote()" in src and "res = PREP.parse(str(data), str(out_dir), str(ccd_path), str(mol_dir), **PROCESS_INPUTS_KW)" in src, base
        assert 'PROCESS_INPUTS_KW = dict(dict(use_msa_server=False, msa_server_url="https://api.colabfold.com", msa_pairing_strategy="greedy", preprocessing_threads=1, max_msa_seqs=8192), boltz2=True,\n                         **{k: OPTS[k] for k in ("use_msa_server", "msa_server_url", "msa_pairing_strategy", "preprocessing_threads", "max_msa_seqs") if k in OPTS})' in src, base
        body = src[src.index("def process_item("):src.index("# ---------------- RNG state capture")]
        assert "process_inputs(" not in body and "check_inputs(" not in body and "EmbedMolecule" not in src, f"{base}: nothing is parsed in the worker process itself"


# ---------------------------------------------------------------- the defect itself, with RDKit (skipped where RDKit is absent) ------------------------------------
def test_smiles_conformers_through_the_zygote_equal_a_fresh_process_first_embedding():
    """Two SMILES inputs parsed in one worker's life both get the conformer a fresh process gives (= stock's `boltz predict <one yaml>`);
    parsed in one process one after the other, the second would not (RDKit's process-global ETKDG RNG)."""
    pytest.importorskip("rdkit")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    fresh = subprocess.run([sys.executable, "-c", f"from {TARGET} import print_conformer_digest as p; p()"], capture_output=True, text=True, env=env, timeout=300)
    assert fresh.returncode == 0, fresh.stderr[-2000:]
    reference = fresh.stdout.strip().splitlines()[-1]          # embedding #1 of a fresh process: what stock's per-input process computes
    fresh2 = subprocess.run([sys.executable, "-c", f"from {TARGET} import print_conformer_digest as p; p()"], capture_output=True, text=True, env=env, timeout=300)
    assert fresh2.stdout.strip().splitlines()[-1] == reference, "a fresh process's first embedding is reproducible across processes (the premise)"
    z = prep.Zygote(preload=(TARGET, "rdkit.Chem.AllChem")).start(forked_at="test")
    try:
        first = z.run(f"{TARGET}:print_conformer_digest"); second = z.run(f"{TARGET}:print_conformer_digest")
    finally:
        z.close()
    assert first.rc == 0 and second.rc == 0, (first.err[-1500:], second.err[-1500:])
    assert first.out.strip() == reference and second.out.strip() == reference, "every input parsed through the zygote is a fresh process's first embedding"
    same_process = subprocess.run([sys.executable, "-c", f"from {TARGET} import conformer_digest as d; print(d()); print(d())"], capture_output=True, text=True, env=env, timeout=300)
    d1, d2 = same_process.stdout.strip().splitlines()[-2:]
    assert d1 == reference and d2 != reference, "the defect this construct exists for: the second embedding of ONE process differs (RDKit ETKDG, no randomSeed)"
