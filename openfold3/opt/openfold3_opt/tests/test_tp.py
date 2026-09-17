"""The big/tp line: the mode table's refusals (the line activates only inside a rank process; the world size is 2|4|8 by name), the
launcher's chunk plan and pinned yaml, the rank processes' environment, the fail-fast wait and the census of the ranks' own lines —
on CPU with stub ranks (no GPU, no add-on import: tp.py's process machinery only)."""
import argparse
import json
import os
import sys

import pytest

from openfold3_opt import modes, tp
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()


def test_tp_line_is_a_rank_only_line():
    r = modes.resolve("big", HOME, environ={}, n_gpu="2")
    assert r.line == "tp" and r.tier == "tolerance" and r.hooks == ["cells", "tp_rowpair", "fast_inference"] and "loader_workers" in r.levers and not modes.graphed("big", "tp")
    assert any("not a rank" in c for c in r.conflicts), r.conflicts                              # the env/.pth route cannot spawn the ranks
    r = modes.resolve("big", HOME, environ={"OF3TP_RANK": "1", "OF3TP_WORLD": "4"}, n_gpu="4")
    assert not r.conflicts and r.exports["OF3TP_TRIMUL_SUB"] == "128" and r.exports["OF3_FAST_INIT"] == "1" and "OF3_CUDA_GRAPHS" in r.unsets
    assert not any("OF3TP_WORLD" in n or "OF3TP_RANK" in n for n in r.notes)                    # the run's variables are not opt-ins (modes.RUN_VARS)
    r = modes.resolve("big", HOME, environ={"OF3TP_RANK": "0", "OF3TP_WORLD": "3", "OPENFOLD3_OPT_N_GPU": "2"})
    assert any("2|4|8" in c for c in r.conflicts), r.conflicts
    r = modes.resolve("big", HOME, environ={"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OF3_CUDA_GRAPHS": "1"}, n_gpu="2")
    assert any("OF3_CUDA_GRAPHS" in c for c in r.conflicts)                                    # a graphed switch preset contradicts the line
    assert r.entry_hook.endswith(os.path.join("opt", "openfold3_opt", "hooks", "cells", "sitecustomize.py")) and r.exports[modes.PAIRFUSED_CHAIN_ENV].endswith(os.path.join("opt", "openfold3_opt", "tp_rowpair", "hook"))   # the cells hook is the entry; it chains the row-sharded line's hook


def test_n_gpu_values_are_refused_by_name():
    for bad in (None, "1", "3", "16", "abc", ""):
        with pytest.raises(ValueError, match="refused: the big/tp line runs on --n_gpu P GPUs, P in 2\\|4\\|8"):
            modes.check_tp_world(bad, environ={})
    # world 1 in particular: the line is the multi-rank form — at world 1 nothing of it may run: refused by name here
    r = modes.resolve("big", HOME, environ={"OF3TP_RANK": "0", "OF3TP_WORLD": "1", "OPENFOLD3_OPT_N_GPU": "2"})
    assert any("2|4|8" in c for c in r.conflicts) and r.exports.get("OF3TP_TRIMUL_SUB") == "128"
    assert modes.check_tp_world(None, environ={"OPENFOLD3_OPT_N_GPU": " 8 "}) == 8 and modes.check_tp_world("2", environ={}) == 2
    assert modes.n_gpu_of(None, environ={}) == 1 and modes.n_gpu_of("4", environ={}) == 4 and modes.n_gpu_of(None, environ={"OPENFOLD3_OPT_N_GPU": "2"}) == 2
    for bad in ("0", "-2", "x", "2.5"):
        with pytest.raises(ValueError, match="a positive integer is required"):
            modes.n_gpu_of(bad, environ={})
    with pytest.raises(ValueError, match="refused: n_gpu=3 is not a GPU count this kit runs"):
        modes.n_gpu_of("3", environ={})
    assert modes.N_GPU_VALUES == ("1", "2", "4", "8") and modes.LINE_PARAMS[("big", "tp")].values == ("2", "4", "8")
    assert modes.line_arg("big", environ={}, n_gpu="2") == "tp" and modes.line_arg("big", environ={"OPENFOLD3_OPT_N_GPU": "2"}) == "tp"
    assert modes.line_arg("big", environ={}) == modes.DEFAULT_BIG_LINE and modes.line_arg("fast", environ={}) is None


def test_chunk_plan_and_pinned_yaml(tmp_path):
    q = {"queries": {"a": {"chains": [{"molecule_type": "protein", "chain_ids": ["A", "B"], "sequence": "M" * 700}, {"molecule_type": "ligand", "chain_ids": ["L"], "smiles": "CCO"}]}}}
    qj = tmp_path / "q.json"
    qj.write_text(json.dumps(q))
    assert tp.polymer_residues(str(qj)) == 1400
    assert [tp.chunk_for(n) for n in (500, 2500, 2501, 4000, 4001, 6500, 6501, 20000)] == [128, 128, 64, 64, 32, 32, 16, 16]
    assert tp.chunk_of_run(str(qj), environ={}) == (128, "plan for 1400 polymer residues (CHUNK_PLAN)")
    assert tp.chunk_of_run(str(qj), environ={"OF3TP_CHUNK": "24"}) == (24, "caller (OF3TP_CHUNK)")
    with pytest.raises(ValueError):
        tp.chunk_of_run(str(qj), environ={"OF3TP_CHUNK": "big"})
    yaml = pytest.importorskip("yaml")
    base = tmp_path / "base.yml"
    base.write_text("model_update:\n  presets: [predict]\n  custom:\n    settings:\n      memory:\n        eval:\n          use_deepspeed_evo_attention: false\n          use_cueq_triangle_kernels: true\nexperiment_settings:\n  seeds: [42]\n")   # a caller yaml with a fused kernel on (stock's cueq det yaml)
    pins = tp.pinned_yaml(str(base), 32, str(tmp_path / "out" / "tp_predict.yml"))
    ev = yaml.safe_load(open(tmp_path / "out" / "tp_predict.yml"))["model_update"]["custom"]["settings"]["memory"]["eval"]
    assert all(ev[f] is False and pins[f] is False for f in tp.KERNEL_FLAGS_OFF), ev        # the fused pair kernels pinned off — a caller's cueq=true too (the ranks' trunk refuses them on)
    src = open(os.path.join(os.path.dirname(tp.__file__), "tp_rowpair", "trunk.py"), encoding="utf-8").read()
    assert sorted(tp.KERNEL_FLAGS_OFF) == sorted(eval(src.split("KERNEL_FLAGS = ", 1)[1].split("\n", 1)[0].split("#")[0])), "the launcher pins exactly the flags the ranks refuse"
    doc = yaml.safe_load(open(tmp_path / "out" / "tp_predict.yml"))
    ev = doc["model_update"]["custom"]["settings"]["memory"]["eval"]
    assert ev == {"use_deepspeed_evo_attention": False, "use_cueq_triangle_kernels": False, "chunk_size": 32, "use_triton_triangle_kernels": False, "use_lma": False,
                  "offload_inference": {"confidence_heads": False, "msa_module": False},
                  "per_sample_token_cutoff": None, "per_sample_atom_cutoff": None}
    arch = doc["model_update"]["custom"]["architecture"]
    assert arch["pairformer"] == {"tune_chunk_size": False} and arch["heads"]["pairformer_embedding"]["pairformer"] == {"tune_chunk_size": False}
    assert len(pins["tuners"]) == 5 and doc["experiment_settings"] == {"seeds": [42]} and doc["model_update"]["presets"] == ["predict"]


def test_rank_env_strips_the_route_and_pins_one_gpu():
    base = {"OPENFOLD3_OPT": "big", "OPENFOLD3_OPT_N_GPU": "4", "OPENFOLD3_OPT_HOME": "/h", "PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    env = tp.rank_env(base, 2, 4, "2", 29999, 64)
    assert "OPENFOLD3_OPT" not in env and "OPENFOLD3_OPT_N_GPU" not in env and env["OPENFOLD3_OPT_HOME"] == "/h" and env["PATH"] == "/bin"
    assert {k: env[k] for k in modes.RUN_VARS} == {"OF3TP_WORLD": "4", "OF3TP_RANK": "2", "OF3TP_ADDR": "127.0.0.1", "OF3TP_PORT": "29999", "OF3TP_CHUNK": "64"}
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_every_package_name_a_rank_inherits_is_declared_to_the_gate(tmp_path):
    """A rank's environment is `launch_env` (the launcher's own + the per-launch records directory, manifest.RECORDS_ENV) through `rank_env`;
    the .pth gate (`_autoload.install`) runs FIRST in every rank's interpreter and exits 3 on any OPENFOLD3_OPT* name outside
    `_autoload.DECLARED`. So every name under the prefix a rank can inherit — from a launcher that carries every declared name (the mode, its
    line parameter, the tree home, the resolver's exported cell / hook variables) — must be declared, and the gate must let the rank's
    environment through, inert (the rank takes its route from argv: OPENFOLD3_OPT itself is stripped)."""
    from openfold3_opt import _autoload, manifest
    launcher = {**{k: "1" for k in _autoload.DECLARED}, "OPENFOLD3_OPT": "big", "OPENFOLD3_OPT_N_GPU": "2", "OPENFOLD3_OPT_HOME": "/h",
                "PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0,1"}
    launcher.pop(manifest.RECORDS_ENV, None)                                  # the launcher never carries it: launch_env adds it
    base = tp.launch_env(str(tmp_path), 2, environ=launcher)
    assert base[manifest.RECORDS_ENV] == str(tmp_path) and base["PYTHONHASHSEED"] == "0"          # + the launch's ONE hash seed (opt_core rankdata: the launcher names none -> 0)
    assert {k: v for k, v in base.items() if k not in (manifest.RECORDS_ENV, "PYTHONHASHSEED")} == launcher
    for r in range(2):
        env = tp.rank_env(base, r, 2, str(r), 29999, 64)
        names = sorted(k for k in env if k.startswith(_autoload.ENV))
        assert manifest.RECORDS_ENV in names and env[manifest.RECORDS_ENV] == str(tmp_path)
        assert env["TMPDIR"] == tp.rank_tmpdir(str(tmp_path), r) == os.path.join(str(tmp_path), f"rank{r}", "tmp")   # a private temporary directory per rank, under the records directory
        assert env["PYTHONHASHSEED"] == "0"                                   # every rank interpreter starts with the launch's one seed
        assert not [k for k in names if k not in _autoload.DECLARED], [k for k in names if k not in _autoload.DECLARED]
        assert _autoload.install(dict(env)) is None                           # the gate in the rank: no NOT ACTIVE / SystemExit 3, and nothing installed (no mode variable in a rank)


STUB = """
import os, sys, time
r, w = int(os.environ["OF3TP_RANK"]), int(os.environ["OF3TP_WORLD"])
od = sys.argv[1]; fail = sys.argv[2] if len(sys.argv) > 2 else ""
print(f"[tp_rowpair hook r{r}] armed: world={w} rank={r} mode=S")
print(f"[rowpair r{r}] process group ready: backend=gloo world={w} mode=S nccl=n/a")
if fail == "die" and r == 1:
    print("boom"); sys.exit(7)
v = "n/a" if w == 1 else ("False" if fail == "feats" else "True")           # the cross-rank input digest gate's line (tp_rowpair/model.feature_digest), one per forward
print(f"[rowpair r{r}] [feats] rank {r} shared 0123456789abcdef keys 60 tensor_leaves=50 nontensor_leaves=2:/query_id/0,/seed/0 nontensor_unhashed=none (host-parked keys out of the shared digest: msa; ranks_identical={v}; 0.1s)")
os.makedirs(os.path.join(od, "q", "seed_42"), exist_ok=True)                # upstream's bookkeeping, written by EVERY rank (the runner's timer and config)
open(os.path.join(od, "q", "seed_42", "timing.json"), "w").write("{}")
open(os.path.join(od, "q", "experiment_config.json"), "w").write("{}")
if r == 0:
    open(os.path.join(od, "pred_model.cif"), "w").write("data_x\\n")
    if os.environ.get("OPENFOLD3_OPT_RECORDS"):                          # the launcher's records directory: the rank's run record goes there, never beside the outputs
        open(os.path.join(os.environ["OPENFOLD3_OPT_RECORDS"], "rank0.json"), "w").write('{"exit_code": 0, "exit_rule": {"reason": "ok"}}')
if fail == "copy" and r == 1:
    open(os.path.join(od, "pred_model.cif"), "w").write("data_y\\n")
n = 100; a = r * (n // w); b = (r + 1) * (n // w)
print(f"[rowpair r{r}] [model] trunk done N={n} P={w} rows {a}:{b} in 0.1s")
print(f"[openfold3-opt tp rank {r}] allocator peak: max_allocated_mib={100 + r} max_reserved_mib={200 + r}")
time.sleep(0.2 if r != 0 else 1.5)
"""


def _run(tmp_path, world, fail=""):
    out = tmp_path / "out"
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    procs = tp.spawn_ranks(lambda r, od: [sys.executable, str(stub), od, fail], str(out), world, [str(i) for i in range(world)], 29517, 64,
                           base_env={**os.environ, "OPENFOLD3_OPT": "big"})
    failed = tp.wait_ranks(procs, poll_s=0.05)
    block = tp.census(str(out), procs, world, "S", "gloo", 64, "test", 29517, {"chunk_size": 64}, {"period_s": 1.0, "samples": 0, "peak_mib": {}})
    return out, procs, failed, block


def test_launcher_spawns_censuses_and_tiles_the_rows(tmp_path):
    out, procs, failed, block = _run(tmp_path, 4)
    assert failed is None and [p["rc"] for p in procs] == [0, 0, 0, 0]
    assert block["shard_map"] == {"0": [0, 25], "1": [25, 50], "2": [50, 75], "3": [75, 100]} and block["shard_map_covers_N"] and block["n_tokens"] == 100
    assert block["groups_reported"] == [["gloo", "4", "S"]] and block["ranks_alive_at_exit"] == 4 and block["ranks_identical"] is True and block["outputs_identical"] is None and not block["reasons"]   # the gate's verdict carried; outputs: no rank but 0 wrote one
    assert [r["feats"] for r in block["ranks"]] == [{"gates": 1, "ranks_identical": True, "shared": "0123456789abcdef"}] * 4
    assert block["peak_mib_per_rank"] == {"0": None, "1": None, "2": None, "3": None} and block["peak_mib_sum"] == 0     # no nvidia-smi on a CPU box: recorded, not invented
    assert [r["allocator_mib"] for r in block["ranks"]] == [{"max_allocated": 100 + r, "max_reserved": 200 + r} for r in range(4)]
    assert block["peak_max_mib_per_rank"] == {"0": 200, "1": 201, "2": 202, "3": 203} and block["peak_max_mib_max"] == 203 and block["peak_max_mib_sum"] == 806
    assert (out / "_tp" / "rank1.log").read_text().count("armed: world=4 rank=1") == 1 and not any(n.endswith(".json") and "manifest" in n for n in os.listdir(out))
    assert [r["outputs"]["identical_to_rank0"] for r in block["ranks"][1:]] == [None, None, None]       # ranks > 0 wrote no prediction: nothing to compare
    assert all(r["outputs"]["prediction_files"] == 0 and r["outputs"]["other_files"] == ["q/experiment_config.json", "q/seed_42/timing.json"] for r in block["ranks"][1:])   # their bookkeeping named


def test_census_names_a_rank_that_never_armed(tmp_path):
    out = tmp_path / "out"
    (out / "_tp").mkdir(parents=True)
    alloc = "[openfold3-opt tp rank R] allocator peak: max_allocated_mib=1 max_reserved_mib=2\n"
    feats = "[rowpair rR] [feats] rank R shared 0123456789abcdef keys 60 (host-parked keys out of the shared digest: msa; ranks_identical=True; 0.1s)\n"   # the input digest gate's line, so the reasons below stay the two this case is about
    (out / "_tp" / "rank1.log").write_text("[rowpair r1] process group ready: backend=gloo world=2 mode=S nccl=n/a\n" + feats.replace("R", "1") + "[rowpair r1] [model] trunk done N=10 P=2 rows 5:10 in 0.1s\n" + alloc)
    (out / "_tp" / "rank0.log").write_text("[tp_rowpair hook r0] armed: world=2 rank=0 mode=S\n[rowpair r0] process group ready: backend=gloo world=2 mode=S nccl=n/a\n" + feats.replace("R", "0") + "[rowpair r0] [model] trunk done N=10 P=2 rows 0:5 in 0.1s\n" + alloc)
    procs = [{"rank": r, "gpu": str(r), "rc": 0, "started": 0.0, "ended": 1.0, "log": str(out / "_tp" / f"rank{r}.log"), "dir": str(out if r == 0 else out / "_tp" / f"rank{r}")} for r in (0, 1)]
    block = tp.census(str(out), procs, 2, "S", "gloo", 64, "test", 1, {}, {"period_s": 1.0, "samples": 0, "peak_mib": {}})
    assert block["reasons"] == ["n_gpu_mismatch requested=2 active=1",                                   # the axis fail-closed comes first: one of two ranks armed and joined
                                "rank 1: no `[tp_rowpair hook r1] armed:` / `[of3tp_hook r1] armed:` line in its transcript (the line's hook did not arm)"]


def test_launcher_fails_fast_and_names_the_rank(tmp_path):
    out, procs, failed, block = _run(tmp_path, 2, fail="die")
    assert failed == 1 and procs[1]["rc"] == 7 and procs[0]["rc"] not in (None, 0)                 # rank 0 (sleeping) was terminated
    assert any(r.startswith("rank 1 exited rc=7") for r in block["reasons"]) and block["ranks_alive_at_exit"] == 0


def test_launcher_names_a_rank_whose_copy_differs(tmp_path):
    out, procs, failed, block = _run(tmp_path, 2, fail="copy")
    assert failed is None and block["outputs_identical"] is False and block["ranks_identical"] is True and any("differ from rank 0" in r for r in block["reasons"])
    assert block["ranks"][1]["outputs"] == {"prediction_files": 1, "identical_to_rank0": False, "other_files": ["q/experiment_config.json", "q/seed_42/timing.json"]}


def test_rank_log_parser_reads_the_add_ons_lines(tmp_path):
    log = tmp_path / "r.log"
    log.write_text("[tp_rowpair hook r2] armed: world=8 rank=2 mode=S\n[rowpair r2] process group ready: backend=nccl world=8 mode=S nccl=2.27.3\n"
                   "[rowpair r2] [model] trunk done N=2104 P=8 rows 526:789 in 12.3s\n")
    rec = tp.parse_rank_log(str(log))
    assert rec == {"armed": "world=8 rank=2 mode=S", "group": {"backend": "nccl", "world": "8", "mode": "S", "nccl": "2.27.3"}, "rows": [526, 789], "N": 2104, "alloc": None, "triatt": None, "trimul": None, "refusal": None, "fallbacks": {}, "feats": None}
    assert tp.parse_rank_log(str(tmp_path / "absent.log")) == {"armed": None, "group": None, "rows": None, "N": None, "alloc": None, "triatt": None, "trimul": None, "refusal": None, "fallbacks": {}, "feats": None}
    log.write_text(log.read_text() + "[openfold3-opt tp rank 2] allocator peak: max_allocated_mib=41000 max_reserved_mib=45056\n")
    assert tp.parse_rank_log(str(log))["alloc"] == {"max_allocated_mib": 41000, "max_reserved_mib": 45056}
    log.write_text("[openfold3-opt tp rank 2] allocator peak: no cuda\n")
    assert tp.parse_rank_log(str(log))["alloc"] == {"unavailable": "no cuda"}
    feats = "[rowpair r2] [feats] rank 2 shared 89abcdef01234567 keys 61 tensor_leaves=51 nontensor_leaves=2:/query_id/0,/seed/0 nontensor_unhashed=none (host-parked keys out of the shared digest: msa; ranks_identical=V; 4.2s)\n"
    log.write_text(feats.replace("V", "True"))                                # the cross-rank input digest gate's line, read back: its verdict and shared digest
    assert tp.parse_rank_log(str(log))["feats"] == {"gates": 1, "ranks_identical": True, "shared": "89abcdef01234567"}
    log.write_text(feats.replace("V", "True") + feats.replace("V", "n/a"))       # two forwards: n/a (no group) never lowers a True
    assert tp.parse_rank_log(str(log))["feats"] == {"gates": 2, "ranks_identical": True, "shared": "89abcdef01234567"}
    log.write_text(feats.replace("V", "True") + feats.replace("V", "False"))     # one False anywhere is False
    assert tp.parse_rank_log(str(log))["feats"]["ranks_identical"] is False
    log.write_text(feats.replace("V", "n/a") + "[rowpair r2] [feats] batch sync: none (data_form=rank0_bcast: this rank holds rank 0's batch from the transfer hook); the cross-rank digest gate follows\n")
    assert tp.parse_rank_log(str(log))["feats"] == {"gates": 1, "ranks_identical": None, "shared": "89abcdef01234567"}   # the sync note is not a gate line


def test_census_carries_the_input_digest_gates_verdict(tmp_path):
    """The launcher's `ranks_identical` is the cross-rank INPUT digest gate's verdict as the ranks logged it (`[feats] rank r … ranks_identical=`):
    True when every rank said True, False (a named reason) when any said False, None when a rank had no verdict — and a rank without the
    line is named (the gate runs on every rank of a P>1 group before the forward). `outputs_identical` is the separate rank-copy match."""
    alloc = "[openfold3-opt tp rank R] allocator peak: max_allocated_mib=1 max_reserved_mib=2\n"

    def logs(v0, v1):
        out = tmp_path / f"out_{v0}_{v1}".replace("/", "")
        (out / "_tp").mkdir(parents=True)
        for r, v in ((0, v0), (1, v1)):
            feats = "" if v is None else f"[rowpair r{r}] [feats] rank {r} shared 0123456789abcdef keys 60 (host-parked keys out of the shared digest: msa; ranks_identical={v}; 0.1s)\n"
            (out / "_tp" / f"rank{r}.log").write_text(f"[tp_rowpair hook r{r}] armed: world=2 rank={r} mode=S\n[rowpair r{r}] process group ready: backend=gloo world=2 mode=S nccl=n/a\n"
                                                    + feats + f"[rowpair r{r}] [model] trunk done N=10 P=2 rows {5 * r}:{5 * r + 5} in 0.1s\n" + alloc.replace("R", str(r)))
        procs = [{"rank": r, "gpu": str(r), "rc": 0, "started": 0.0, "ended": 1.0, "log": str(out / "_tp" / f"rank{r}.log"), "dir": str(out if r == 0 else out / "_tp" / f"rank{r}")} for r in (0, 1)]
        return tp.census(str(out), procs, 2, "S", "gloo", 64, "test", 1, {}, {"period_s": 1.0, "samples": 0, "peak_mib": {}})

    block = logs("True", "True")
    assert block["ranks_identical"] is True and block["outputs_identical"] is None and block["reasons"] == [], block["reasons"]
    assert [r["feats"] for r in block["ranks"]] == [{"gates": 1, "ranks_identical": True, "shared": "0123456789abcdef"}] * 2
    block = logs("True", None)                                                  # rank 1 never wrote the gate line
    assert block["ranks_identical"] is None and block["reasons"] == ["rank 1: no `[feats] rank 1 … ranks_identical=` line in its transcript (the cross-rank input digest gate did not report)"], block["reasons"]
    block = logs("False", "False")                                              # the gate said the ranks' inputs differ (the ranks refuse by name too)
    assert block["ranks_identical"] is False and block["reasons"] == [f"rank {r}: its input digest gate reported ranks_identical=False (the ranks featurised the query differently: feats_ranks_differ)" for r in (0, 1)]
    block = logs("n/a", "n/a")                                                  # no multi-rank group behind the line: no verdict, nothing claimed, nothing failed
    assert block["ranks_identical"] is None and block["reasons"] == []


def test_rank_branch_of_cmd_pred_runs_to_the_allocator_line(tmp_path, monkeypatch, capsys):
    """A rank process (OF3TP_RANK set) takes cmd_pred's in-process branch: enable, the predict, the rank's allocator line on stderr, the
    gated manifest — driven end to end with the predict and the stack stubbed (no GPU, no weights)."""
    monkeypatch.delenv(modes.NOISE_SYNC_SWITCH, raising=False)            # the rank branch sets the replicated-sync policy switch in os.environ; restored at teardown
    import argparse
    import openfold3_opt
    from openfold3_opt import cli
    calls = {}
    monkeypatch.setenv("OF3TP_RANK", "1"); monkeypatch.setenv("OF3TP_WORLD", "2"); monkeypatch.setenv("OPENFOLD3_OPT_HOME", HOME)
    monkeypatch.setattr(cli, "_ckpt", lambda c: str(tmp_path / "w.pt"))
    monkeypatch.setattr(cli, "weights_check", lambda *a, **k: ({"path": "w.pt"}, 0))
    monkeypatch.setattr(openfold3_opt, "enable", lambda mode, **k: calls.setdefault("enable", (mode, k)) and {"mode": mode, "active": True})
    monkeypatch.setattr(cli, "stock_argv", lambda a, home, ckpt, q, line=None, yml=None: ["--runner-yaml", str(tmp_path / "r.yml")])
    monkeypatch.setattr(cli, "run_predict_inprocess", lambda argv: calls.setdefault("predict", argv) and 0)
    monkeypatch.setattr(cli, "gated_manifest", lambda out, rep, what, rc, expected, **k: calls.setdefault("manifest", (what, rc, expected)) and 0)
    a = argparse.Namespace(mode="big", ckpt=None, det=0, n_gpu="2", items=None, query_json=str(tmp_path / "q.json"),
                           output_dir=str(tmp_path / "out"), num_model_seeds=None,
                           num_diffusion_samples=None, use_templates=None, use_msa_server=None, runner_yaml=None)
    assert cli.cmd_pred(a) == 0
    err = capsys.readouterr().err
    assert "[openfold3-opt tp rank 1] allocator peak:" in err, err                     # the allocator instrument line, from the rank branch itself
    assert calls["enable"][0] == "big" and str(calls["enable"][1]["n_gpu"]) == "2" and calls["enable"][1]["strict"] is True
    assert calls["manifest"] == ("pred --mode big (tp rank 1)", 0, None)             # ranks > 0 expect no structures


def test_n_gpu_route_rules_are_refused_by_name(tmp_path, monkeypatch, capsys):
    """`--n_gpu P` / OPENFOLD3_OPT_N_GPU is the memory mode's multi-GPU axis (opt_core.mem.ngpu): P=1 is every route's; P>1 under big with no
    line named IS the tp line; P>1 under exact/fast/off is refused with ngpu's exact
    sentence — the CLI route exits 2 (usage) naming it; the env route resolves with a conflict (NOT ACTIVE 3 at activation)."""
    import argparse
    from opt_core.mem import ngpu
    from openfold3_opt import cli
    assert ngpu.REFUSE_MODE == "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)"
    # P > 1 with no big line named selects tp; P = 1 keeps the default line; an explicit one-GPU line with P > 1 is refused
    assert modes.line_arg("big", environ={}, n_gpu="2") == "tp" and modes.line_arg("big", environ={"OPENFOLD3_OPT_N_GPU": "4"}) == "tp"
    assert modes.line_arg("big", environ={}, n_gpu=None) == modes.DEFAULT_BIG_LINE and modes.line_arg("big", environ={}, n_gpu="1") == modes.DEFAULT_BIG_LINE
    assert modes.check_n_gpu_route("big", "tp", "2", environ={}) == 2 and modes.check_n_gpu_route("big", "resident", None, environ={}) == 1
    assert modes.check_n_gpu_route("fast", None, "1", environ={}) == 1 and modes.check_n_gpu_route("exact", "sampler", None, environ={"OPENFOLD3_OPT_N_GPU": "1"}) == 1
    with pytest.raises(ValueError, match="refused: n_gpu=2 runs the big/tp line \\(the row-sharded pair stack\\); the selected line is big/resident, a one-GPU line"):
        modes.check_n_gpu_route("big", "resident", "2", environ={})
    with pytest.raises(ValueError, match="refused: the big/tp line runs on --n_gpu P GPUs, P in 2\\|4\\|8 \\(got n_gpu=1\\)"):
        modes.check_n_gpu_route("big", "tp", "1", environ={})
    for mode, line in (("exact", "sampler"), ("exact", "composed"), ("fast", None), ("off", None)):
        with pytest.raises(ValueError) as ei:
            modes.check_n_gpu_route(mode, line, "4", environ={})
        assert str(ei.value) == ngpu.REFUSE_MODE, str(ei.value)
        f = modes.foreign_params(mode, line, environ={"OPENFOLD3_OPT_N_GPU": "8"})                 # the env route: the same sentence, the variable named
        assert len(f) == 1 and f[0].startswith(ngpu.REFUSE_MODE) and "OPENFOLD3_OPT_N_GPU=8" in f[0], f
        assert not modes.foreign_params(mode, line, args={"n_gpu": 1}, environ={"OPENFOLD3_OPT_N_GPU": "1"})   # P = 1 is never foreign
    # env route under big: the variable selects tp (a non-rank then refuses: cannot spawn)
    r = modes.resolve("big", HOME, environ={"OPENFOLD3_OPT_N_GPU": "2"})
    assert r.line == "tp" and any("not a rank" in c for c in r.conflicts), (r.line, r.conflicts)
    assert not modes.foreign_params("big", "tp", args={"n_gpu": "2"}, environ={"OPENFOLD3_OPT_N_GPU": "2"})       # the owner: nothing foreign
    for mode in ("exact", "fast"):                                                  # the env route with a mode named: the resolution carries the conflict (NOT ACTIVE at activation)
        r = modes.resolve(mode, HOME, environ={"OPENFOLD3_OPT_N_GPU": "2"})
        assert any(c.startswith(ngpu.REFUSE_MODE) for c in r.conflicts), (mode, r.conflicts)
        assert not modes.resolve(mode, HOME, environ={"OPENFOLD3_OPT_N_GPU": "1"}).conflicts
    assert not modes.resolve("exact", HOME, environ={}).conflicts
    # the env route with `off` selected: the hook's gate refuses P > 1 before anything is imported (the .pth form; SystemExit 3 with the line); P = 1 passes
    from openfold3_opt import _autoload
    assert _autoload.LINE_PARAMS == {p.env: (f"{m}/{l}" if l else m) for (m, l), p in modes.LINE_PARAMS.items()}
    for env in ({"OPENFOLD3_OPT": "off", "OPENFOLD3_OPT_N_GPU": "2"}, {"OPENFOLD3_OPT": " OFF ", "OPENFOLD3_OPT_N_GPU": "4"}):
        with pytest.raises(SystemExit) as ex:
            _autoload.install(environ=env)
        assert ex.value.code == 3
        err = capsys.readouterr().err
        assert "[openfold3-opt] NOT ACTIVE: OPENFOLD3_OPT_N_GPU=" in err and "is the big/tp line's parameter" in err and "OPENFOLD3_OPT='off'" in err, err
    assert _autoload.install(environ={}) is None and _autoload.install(environ={"OPENFOLD3_OPT_N_GPU": "2"}) is None and _autoload.install(environ={"OPENFOLD3_OPT": "off"}) is None
    assert _autoload.install(environ={"OPENFOLD3_OPT": "off", "OPENFOLD3_OPT_N_GPU": "1"}) is None
    # CLI route: pred with --n_gpu 2 under exact / fast / off -> usage exit, the refusal named; under big -> the tp launcher
    monkeypatch.setenv("OPENFOLD3_OPT_HOME", HOME); monkeypatch.delenv("OPENFOLD3_OPT_N_GPU", raising=False)
    monkeypatch.setattr(cli, "_ckpt", lambda c: str(tmp_path / "w.pt"))
    def ns(mode, n):
        return argparse.Namespace(mode=mode, ckpt=None, det=0, n_gpu=n, items=None, query_json=str(tmp_path / "q.json"),
                                  output_dir=str(tmp_path / "out"), num_model_seeds=None,
                                  num_diffusion_samples=None, use_templates=None, use_msa_server=None, runner_yaml=None)
    for mode, want in (("exact", ngpu.REFUSE_MODE), ("fast", ngpu.REFUSE_MODE), ("off", ngpu.REFUSE_MODE)):
        assert cli.cmd_pred(ns(mode, "2")) == cli.EXIT_USAGE, mode
        err = capsys.readouterr().err
        assert want in err, (mode, err)
    # `--mode big --n_gpu 2` alone is the tp line: the launcher runs, and on a box with ONE visible GPU it refuses by name with opt_core.mem.ngpu's
    # sentence (exit 3) — never a one-card run (CUDA_VISIBLE_DEVICES names one device here; tp.gpu_census reads it before nvidia-smi)
    monkeypatch.setattr(cli, "weights_check", lambda *a, **k: ({"path": "w.pt"}, 0))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert cli.cmd_pred(ns("big", "2")) == cli.EXIT_NOT_ACTIVE
    err = capsys.readouterr().err
    assert "[openfold3-opt tp] NOT ACTIVE: refused: n_gpu=2 visible=1" in err and ngpu.visible_refusal(2, 1) == "refused: n_gpu=2 visible=1", err
    monkeypatch.setenv("OPENFOLD3_OPT_N_GPU", "8")                                   # the variable alone, CLI route: it selects the tp line
    a8 = argparse.Namespace(det=0, n_gpu=None)
    cli.route_options(a8, "big")
    assert a8.line == "tp"


def test_census_fails_closed_when_fewer_ranks_joined_than_requested(tmp_path):
    """The axis fail-closed (class: n_gpu silent drop): `--n_gpu 2` whose ranks never armed / joined a 2-wide group is `n_gpu_mismatch
    requested=2 active=Q` FIRST among the census reasons (the launcher maps it to NOT ACTIVE, exit 3) — never a pass on one GPU."""
    out = tmp_path / "out"; (out / "_tp").mkdir(parents=True)
    for r in (0, 1):
        (out / "_tp" / f"rank{r}.log").write_text("some upstream output\n")            # no armed / group / rows lines: the add-on never armed in these ranks
    procs = [{"rank": r, "gpu": str(r), "rc": 0, "started": 0.0, "ended": 1.0, "log": str(out / "_tp" / f"rank{r}.log"), "dir": str(out if r == 0 else out / "_tp" / f"rank{r}")} for r in (0, 1)]
    mem = {"period_s": 1.0, "samples": 0, "peak_mib": {}}
    block = tp.census(str(out), procs, 2, "S", "nccl", 128, "plan", 0, {}, mem)
    assert block["reasons"][0] == "n_gpu_mismatch requested=2 active=0" and block["n_gpu_active"] == 0, block["reasons"][:3]
    # one rank armed in a 1-wide group (the axis dropped inside the rank) is still a mismatch: `active` counts only ranks in a 2-wide group
    (out / "_tp" / "rank0.log").write_text("[tp_rowpair hook r0] armed: world=1 rank=0 mode=S\n[rowpair r0] process group ready: backend=nccl world=1 mode=S nccl=2.27.3\n")
    block = tp.census(str(out), procs, 2, "S", "nccl", 128, "plan", 0, {}, mem)
    assert block["reasons"][0] == "n_gpu_mismatch requested=2 active=0", block["reasons"][:3]


# ------------------------------------------------------------------------------------------ pre-flight row layout (input too small for P) ----


def test_judge_templates_adopts_a_rank_judgement_and_judges_an_unjudged_run_once(monkeypatch, capsys):
    """The launcher's template judge: rank 0's own judgement (its record says judged — every rank runs cmd_pred's guard) stands and only the census
    word is taken from it (nothing re-printed, the code is the rank's); a run whose rank 0 did not judge is judged here once against rank 0's
    featurised record (TEMPLATES DROPPED printed once, exit 5 unless --allow-template-drop)."""
    import argparse
    from openfold3_opt import cli, templ_census
    a = argparse.Namespace(allow_template_drop=False)
    # (1) rank 0 judged a drop and exited 5: adopted, census word set, no line printed here
    man = {"templates": {"declared": {"q": 1}, "featurised": {"q": {"slots": 4, "real_slots": 0}}, "dropped": [{"query": "q"}], "judged": True}}
    block = {"fallbacks": {}}
    code, reason, rec = tp.judge_templates(a, man, 5, "rank 0 exited rc=5", block, (0,))
    assert (code, rec is man["templates"], block["fallbacks"]) == (5, True, {"templates_dropped": 1}) and capsys.readouterr().err == ""
    # (2) rank 0 returned 0 without judging (declaration held only here): judged once in this process
    monkeypatch.setattr(templ_census, "declared", lambda: {"q": 1})
    man = {"templates": {"declared": {}, "featurised": {"q": {"slots": 4, "real_slots": 0, "key": "template_pseudo_beta_mask"}}, "dropped": [], "judged": False}}
    block = {"fallbacks": {}}
    code, reason, rec = tp.judge_templates(a, man, 0, "ok", block, (0,))
    err = capsys.readouterr().err
    assert code == cli.EXIT_TEMPLATES_DROPPED and block["fallbacks"] == {"templates_dropped": 1} and err.count("TEMPLATES DROPPED") == 1, (code, block, err)
    code, reason, rec = tp.judge_templates(argparse.Namespace(allow_template_drop=True), man, 0, "ok", {"fallbacks": {}}, (0,))
    assert code == 0 and "accepted by --allow-template-drop" in reason
