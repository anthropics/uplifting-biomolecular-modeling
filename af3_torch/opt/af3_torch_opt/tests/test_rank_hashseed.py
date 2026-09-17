"""`big --n_gpu P`, P > 1: the P ranks are `spawn` interpreters of ONE launch and start with ONE `PYTHONHASHSEED`, decided once per run by
the shared core's launcher (`opt_core.mem.rowpair.launch.run_sharded` on `rankdata.hash_seed`: an exported decimal integer in [0, 4294967295]
is kept — source=inherited — unset, empty, `random` or any other value gives "0" — source=default), so no step that orders bytes by str hash
can make two ranks disagree; the launcher prints it once per launch as `[<tag>] RANKENV hashseed=<v> source=<s> [parent=<repr>] ranks=P`
(`rankdata.rankenv_line`). The kit adds no seed logic. Its part, held here: the wrapper hands the model process the kit's tag
(`ROWPAIR_TAG=af3-torch-opt`, under P > 1 only) so the line reads `[af3-torch-opt] RANKENV …`, and relays that one line verbatim on stderr from
the model process's transcript; `forward.main_sharded` hands the launcher exactly P with no per-rank environment of its own — exercised over
the REAL launcher (two gloo ranks on CPU, torch required; skipped by name without it); and the model process's base environment
(`stack.model_process_env`) carries an exported `PYTHONHASHSEED` through unchanged and adds neither a seed nor a tag."""
import json
import os

import pytest

from af3_torch_opt import stack
from af3_torch_opt.report import PREFIX, RANKENV_MARK, TAG
from opt_core.mem.rowpair import launch, rankdata

PROBE_TEXT = "rowpair hash seed probe"        # hashed in every rank: equal str hashes <=> one seed
EXPECT = {None: ("0", "default"), "123": ("123", "inherited"), "random": ("0", "default")}   # exported PYTHONHASHSEED → the ranks' (value, source): a decimal integer is kept, anything else gives 0


def _parent_env(preset):
    """The launching process's environment as far as the seed decision reads it."""
    return {} if preset is None else {rankdata.HASHSEED_ENV: preset}


def _rank_probe(argv):
    """The rank body run_sharded spawns in place of forward._rank_entry (module level: `spawn` pickles it by module path). Records what THIS
    interpreter started with beside the report the argv names; rank 0 also leaves the minimal report a model process leaves. Returns its rc."""
    from af3_torch_opt import forward
    a = forward.parse_args(argv)
    r = int(launch.rank())
    facts = {"rank": r, "hashseed": os.environ.get(rankdata.HASHSEED_ENV, ""), "str_hash": hash(PROBE_TEXT), "tag": os.environ.get(launch.ENV_TAG)}
    with open(os.path.join(os.path.dirname(os.path.abspath(a.report)), "rank%d.json" % r), "w", encoding="utf-8") as fh:
        json.dump(facts, fh)
    if r == 0:
        forward._write(a.report, {"items": [], "ok": True, "dtk": bool(a.dtk), "n_gpu": int(a.n_gpu)})
    return 0


def _forward_argv(tmp_path, n_gpu):
    return ["--kit", str(tmp_path / "af3t"), "--dtk-home", str(tmp_path / "dtk"), "--params", str(tmp_path / "weights.bin"), "--report", str(tmp_path / "forward.json"),
            "--n-gpu", str(n_gpu), "--item", f"a=1={tmp_path / 'batch.npz'}={tmp_path / 'result.npz'}"]


def test_the_relay_mark_is_the_cores_line_under_the_kit_tag():
    """The wrapper relays transcript lines carrying RANKENV_MARK: exactly the core's RANKENV line rendered under this kit's tag, not another tag's."""
    assert TAG == "af3-torch-opt" and RANKENV_MARK == f"{PREFIX} RANKENV " == f"[{TAG}] RANKENV "
    assert rankdata.rankenv_line(TAG, {}, 2).startswith(RANKENV_MARK) and rankdata.rankenv_line(TAG, {}, 2) == f"{PREFIX} RANKENV hashseed=0 source=default ranks=2"
    assert RANKENV_MARK not in rankdata.rankenv_line("rowpair", {}, 2)


def test_the_model_process_environment_carries_an_exported_hash_seed_and_adds_no_seed_or_tag(box, monkeypatch):
    """The featurise process (jax) and the forward process — the parent every rank is spawned from (torch) — start from stack.model_process_env:
    an exported PYTHONHASHSEED rides through unchanged; an absent one stays absent (the launcher decides for the ranks); the line tag is not part
    of this base environment (the wrapper adds it to the forward process's, under --n_gpu > 1 only)."""
    monkeypatch.delenv(launch.ENV_TAG, raising=False)
    monkeypatch.setenv(rankdata.HASHSEED_ENV, "7")
    for jax in (False, True):
        env = stack.model_process_env(jax=jax)
        assert env[rankdata.HASHSEED_ENV] == "7" and launch.ENV_TAG not in env, jax
    monkeypatch.delenv(rankdata.HASHSEED_ENV)
    for jax in (False, True):
        assert rankdata.HASHSEED_ENV not in stack.model_process_env(jax=jax), jax


def test_the_wrapper_hands_the_kit_tag_and_relays_the_rankenv_line_once(box, monkeypatch, capsys):
    """cli pred: at --n_gpu 2 the model process runs with ROWPAIR_TAG=<kit tag> and its launcher's `[af3-torch-opt] RANKENV …` line appears once
    on the wrapper's stderr, verbatim, before DONE; at P = 1 the model process is handed no tag, nothing launches, no RANKENV line exists."""
    from .test_n_gpu import _pred
    monkeypatch.delenv(launch.ENV_TAG, raising=False)
    rc, err, fwd = _pred(box, monkeypatch, capsys, 2)
    line = rankdata.rankenv_line(TAG, {}, 2)                                 # `[af3-torch-opt] RANKENV hashseed=0 source=default ranks=2` (the stub launcher's seed is the default one)
    lines = err.splitlines()
    assert rc == 0 and fwd["rowpair_tag"] == TAG, (rc, fwd.get("rowpair_tag"))
    assert lines.count(line) == 1, err[-3000:]
    assert lines.index(line) < next(i for i, l in enumerate(lines) if l.startswith(f"{PREFIX} DONE ")), err[-3000:]
    rc1, err1, fwd1 = _pred(box, monkeypatch, capsys, 1)
    assert rc1 == 0 and fwd1["rowpair_tag"] is None and "RANKENV" not in err1, err1[-3000:]


@pytest.mark.parametrize("preset", list(EXPECT))
def test_main_sharded_over_the_real_launcher_starts_every_rank_with_one_hash_seed(tmp_path, monkeypatch, capfd, preset):
    """forward.main_sharded drives the REAL launch.run_sharded (a spy swaps only the rank body for _rank_probe and asks for gloo on CPU): the kit
    hands it exactly P, the one argv, mode="big", nccl_timeout_s, log_dir, return_records and NO env / base / per-rank mapping; the two spawned
    ranks start with one PYTHONHASHSEED (EXPECT: an exported integer kept, else "0") and hash a str identically; the launcher's ONE line reads
    `[af3-torch-opt] RANKENV hashseed=<v> source=<s> ranks=2` under the tag the wrapper hands the model process; the launching process keeps its
    own value; rank 0's report gains the rank census (world, ranks)."""
    pytest.importorskip("torch")              # run_sharded's P > 1 form spawns through torch.multiprocessing and joins a gloo group
    from af3_torch_opt import forward
    monkeypatch.setenv(launch.ENV_TAG, TAG)                                   # what the wrapper exports to the model process under --n_gpu > 1
    if preset is None:
        monkeypatch.delenv(rankdata.HASHSEED_ENV, raising=False)
    else:
        monkeypatch.setenv(rankdata.HASHSEED_ENV, preset)
    real_run_sharded, calls = launch.run_sharded, []

    def spy_run_sharded(n_gpu, entry, *args, **kwargs):
        calls.append({"n_gpu": n_gpu, "entry": entry, "args": args, "kwargs": dict(kwargs), "hashseed_at_launch": os.environ.get(rankdata.HASHSEED_ENV)})
        return real_run_sharded(n_gpu, _rank_probe, *args, backend="gloo", cpu_ok=True, **kwargs)

    monkeypatch.setattr(launch, "run_sharded", spy_run_sharded)
    argv = _forward_argv(tmp_path, 2)
    assert forward.main_sharded(forward.parse_args(argv), argv) == 0
    assert len(calls) == 1, calls
    c = calls[0]
    assert c["n_gpu"] == 2 and c["entry"] is forward._rank_entry and c["args"] == (argv,)
    assert c["kwargs"] == {"mode": "big", "nccl_timeout_s": forward.RANK_TIMEOUT_S, "log_dir": str(tmp_path / "ranks"), "return_records": True}, c["kwargs"]
    assert c["hashseed_at_launch"] == preset and os.environ.get(rankdata.HASHSEED_ENV) == preset      # neither set nor cleared by the kit; the launcher restores its export
    r0, r1 = (json.load(open(tmp_path / f"rank{r}.json", encoding="utf-8")) for r in (0, 1))
    want, source = EXPECT[preset]
    assert (r0["rank"], r1["rank"]) == (0, 1) and r0["tag"] == r1["tag"] == TAG
    assert r0["hashseed"] == r1["hashseed"] == want, (r0, r1)             # one value in both interpreters
    assert r0["str_hash"] == r1["str_hash"], (r0, r1)                     # hence str-hash-ordered bytes agree across ranks
    line = rankdata.rankenv_line(TAG, _parent_env(preset), 2)
    assert line.startswith(RANKENV_MARK) and f"hashseed={want} source={source}" in line and line.endswith("ranks=2"), line
    err = capfd.readouterr().err
    assert err.count(line) == 1, (line, err[-3000:])
    rep = forward._read(str(tmp_path / "forward.json"))
    assert rep["ok"] is True and rep["world"] == 2 and [r["rank"] for r in rep["ranks"]] == [0, 1], rep
