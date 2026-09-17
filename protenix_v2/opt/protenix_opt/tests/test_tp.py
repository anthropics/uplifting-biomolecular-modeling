"""CPU tests for the multi-GPU line (tp.py): the selector and its refusals, the ranks' environment, the launcher command,
the events read from the ranks' logs (world size, banners, phase logs) and their gate, the ranks' activation records reconciled,
the shard map from the unit's own Layout, and the CLI routes: `--n_gpu` > 1 with another mode is usage, `big` without N is NOT ACTIVE,
`check --mode big --n_gpu P` on a box without P GPUs is NOT ACTIVE by name, the env route refuses PROTENIX_OPT=big and a bare
PROTENIX_OPT_N_GPU > 1 in-process."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import protenix_opt
from protenix_opt import _autoload, cli, det, kits, stack, tp
from protenix_opt.tests.test_cli_passthrough import fake_stock  # noqa: F401  (the stock CLI stand-in: stock_pred_params parses the stock argv through it)

BANNER = ("[launch_core] P={n} torch=2.13.0+cu130 cuda=13.0 nccl=2.29.7 python=3.11.5 env={{}}\n"
          "[launch] layout block B = 128 for N~1592, P={n} (auto: largest of 128/64/32/16 with no empty rank)\n"
          "[launch] TRIMUL/TRIATT env (PairCore v3 recommended): N~1592 P={n} shard=0.6GB bcache=device\n")
RANK = "[launch_core] rank {r}/{n} device=cuda:{r} name=NVIDIA H100 80GB HBM3 cc=9.0 mem=79.2 GiB driver=580.95.05\n"
ROUTE = "".join(f"[protenix-opt] TP-ROUTE {m} -> {c} names={len(ns)} core=/venv/lib/python3.11/site-packages/opt_core/mem/rowpair/x.py\n" for m, (c, ns) in tp.tp_route.ROUTES.items())
ROUTE += "".join(f"[protenix-opt] TP-BIND {m} -> {b} names={len(ns)} core=?\n" for m, (b, ns) in tp.tp_route.BIND.items())   # + the engine seams bound to the kit's bindings
APPLIED = ROUTE + "[ptx_tp r{r}] APPLIED P={n} feat=bcast relp=lazy zinit_recompute=0 extras=['atom_local'] seams: pairformer=tp msa=tp diffusion=tp confidence=tp summary=tp trimul=tp triatt=tp rowlocal=tp trunk=tp\n"
EFFECTIVE = ("per-rank peak alloc GiB = [19.062, 18.91] (max 19.06) / single-card peak = n/a GiB  [N_token=1592, P={n}, seams: pairformer=tp msa=tp diffusion=tp "
             "confidence=tp summary=tp trimul=tp triatt=tp rowlocal=tp trunk=tp; tiling: N~1592 P={n} shard=0.6GB]\n")
RANKS_ON_ONE_LINE = "[launch_core] rank 1/2 device=cuda:1 name=NVIDIA H100 80GB HBM3 cc=9.0 mem=79.2 GiB driver=580.95.05[launch_core] rank 0/2 device=cuda:0 name=NVIDIA H100 80GB HBM3 cc=9.0 mem=79.2 GiB driver=580.95.05\n"


def _phase_log(tp_dir, r, phases=("trunk_start", "trunk_end", "diffusion_start", "diffusion_end", "confidence_start", "confidence_end")):
    with open(os.path.join(tp_dir, f"phases_rank{r}.jsonl"), "w") as fh:
        for i, p in enumerate(phases):
            fh.write(json.dumps({"t": i, "rank": r, "phase": p, "cuda": {"max_alloc_GiB": 1.0 + i}}) + "\n")


def _run_dir(tmp_path, n=2, ranks=None, logs=None, applied=None, extra=""):
    out = tmp_path / "out"; tp_dir = out / tp.PHASE_LOG_DIR; tp_dir.mkdir(parents=True)
    text = (BANNER.format(n=n) + "".join(RANK.format(r=r, n=n) for r in (range(n) if ranks is None else ranks))
            + "".join(APPLIED.format(r=r, n=n) for r in (range(n) if applied is None else applied)) + EFFECTIVE.format(n=n) + extra)
    (tp_dir / tp.LAUNCH_LOG).write_text(text)
    for r in (range(n) if logs is None else logs):
        _phase_log(str(tp_dir), r)
    return str(out)


# ------------------------------------------------------------------------------------------------------------ selector
def test_split_n_gpu_forms_and_refusals():
    assert tp.split_n_gpu(["--n_gpu", "4", "pred", "-i", "x"]) == (4, ["pred", "-i", "x"])
    assert tp.split_n_gpu(["pred", "--n_gpu=2"]) == (2, ["pred"])
    assert tp.split_n_gpu(["pred", "--n_gpu", "1"]) == (1, ["pred"]), "an explicit --n_gpu 1 is accepted (the single-GPU line)"
    assert tp.split_n_gpu(["pred"]) == (None, ["pred"])
    for bad in (["--n_gpu"], ["--n_gpu", "two"], ["--n_gpu", "0"], ["--n_gpu=-1"]):
        with pytest.raises(tp.TpError):
            tp.split_n_gpu(bad)
    with pytest.raises(tp.TpError):
        tp._positive_int("2.5", "--n_gpu")


def test_selection_flag_env_default_and_disagreement():
    assert tp.selection(2, {}) == (2, tp.FLAG_NGPU)
    assert tp.selection(None, {tp.ENV_NGPU: "3"}) == (3, tp.ENV_NGPU)
    assert tp.selection(None, {}) == (1, "default"), "absent --n_gpu == --n_gpu 1 (explicit default, never auto-detected)"
    assert tp.selection(1, {}) == (1, tp.FLAG_NGPU)
    assert tp.selection(2, {tp.ENV_NGPU: "2"}) == (2, tp.FLAG_NGPU)
    with pytest.raises(tp.TpError):
        tp.selection(2, {tp.ENV_NGPU: "3"})
    with pytest.raises(tp.TpError):
        tp.n_gpu_from_env({tp.ENV_NGPU: "many"})
    assert tp.DEFAULT_NGPU == 1 and tp.FLAG_NGPU == "--n_gpu" and tp.ENV_NGPU == "PROTENIX_OPT_N_GPU"


def test_refusal_rule_and_line_selection():
    """n_gpu > 1 is big's; the refusal sentence and the token text are the shared core's (opt_core.mem.ngpu), byte for byte."""
    from opt_core.mem import ngpu
    assert tp.refusal("fast", 1) is None and tp.refusal("exact", 1) is None and tp.refusal("off", 1) is None, "n_gpu 1 runs under any mode"
    assert tp.refusal("big", 1) is None and tp.refusal("big", 2) is None and tp.refusal("big", 8) is None
    for mode in ("exact", "fast", "off"):
        assert tp.refusal(mode, 2) == "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)" == ngpu.REFUSE_MODE
    assert tp.line_selected("big", 2) and not tp.line_selected("big", 1) and not tp.line_selected("big", None) and not tp.line_selected("fast", 2)
    assert tp.LINE_MODE == "big" and tp.BASE_MODE == "fast" and tp.LINE_ARGS == ("--lazy-relp", "--lift-guard")
    assert "--dap" not in tp.LINE_ARGS, "the line never passes --dap"


def test_active_fields_token_text_is_exact(monkeypatch):
    """Tools outside this tree grep these tokens byte for byte: `n_gpu=P sharding=rowpair` for the multi-GPU line,
    `n_gpu=1 sharding=none` for every single-GPU process (ACTIVE, FINAL and EXIT lines)."""
    from protenix_opt import report
    monkeypatch.delenv(tp.HASHSEED_ENV, raising=False)                                  # the line states the launching process's seed: unset here -> hashseed=0 source=default
    assert tp.active_fields(2) == "n_gpu=2 sharding=rowpair" and tp.active_fields(4) == "n_gpu=4 sharding=rowpair" and tp.active_fields(8) == "n_gpu=8 sharding=rowpair"
    assert tp.active_fields(1) == "n_gpu=1 sharding=none"
    assert tp.SCHEME == "rowpair" and tp.IMPL == "ptx_tp" and tp.STRATEGY == "F7.tensor_parallel"
    assert report.ngpu_fields({}) == "n_gpu=1 sharding=none" and report.ngpu_fields({"n_gpu": 2, "sharding": "rowpair"}) == "n_gpu=2 sharding=rowpair"
    base = {"active": True, "mode": "big", "protenix_version": "2.0.0", "gpu": None, "levers_applied": ["deadskip"], "levers_fallback": []}
    assert " n_gpu=1 sharding=none " in report.activation_line(base) and " n_gpu=1 sharding=none " in report.final_line(base)
    assert "ACTIVE mode=big line=tp n_gpu=2 sharding=rowpair impl=ptx_tp routed=dist,blockreduce,bcast,contract->opt_core.mem.rowpair bound=trunk,template,diffusion,pairformer,msa->protenix_opt.tp_bind across_P=tier2(gemm_m=rows_per_rank) inplace_chunk=256 noise_sync=bcast launcher=torchrun_loopback hashseed=0 source=default ranks=2 base=fast " in tp.active_line("[protenix-opt]", 2, base)


def test_rank_env_composes_the_base_mode_through_the_env_route(monkeypatch, tmp_path):
    monkeypatch.setenv(tp.ENV_NGPU, "2"); monkeypatch.setenv("PYTHONPATH", "/elsewhere"); monkeypatch.setenv("PROTENIX_OPT", "big")
    env = tp.rank_env(2, str(tmp_path), exports={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PTX_FPF_XL": "1"})
    assert env["PROTENIX_OPT"] == tp.BASE_MODE, "every rank activates the base mode through the installed .pth"
    assert tp.ENV_NGPU not in env, "a rank is not a selection"
    assert env["PYTHONPATH"].split(":") == [tp.unit_dir(), "/elsewhere"], "the unit first on the ranks' path"
    assert env["PTX_TP_PHASE_LOG"] == os.path.join(str(tmp_path), tp.PHASE_LOG_DIR)
    assert env["PTX_LEVER_REPORT"] == os.path.join(str(tmp_path), tp.PHASE_LOG_DIR, tp.LEVER_REPORT)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True", "the allocator policy precedes the rank's CUDA initialisation"
    assert env["PTX_BLK_GRAPH"] == "0" and env["PTX_SAMPLER_GRAPH"] == "0", "the graph levers stay uninstalled in every rank (TP_DROPPED)"
    assert "PTX_FPF_XL" not in env, "only the pre-CUDA keys are exported by the parent; the rank's own activation exports the rest"


HASHSEED_CASES = (                                                                       # the caller's PYTHONHASHSEED -> (the ranks' value, the ACTIVE line's words for 2 ranks)
    (None, "0", "hashseed=0 source=default ranks=2"),                                    # unset: the default seed
    ("7", "7", "hashseed=7 source=inherited ranks=2"),                                   # a decimal integer in the interpreter's range: kept
    ("random", "0", "hashseed=0 source=default parent='random' ranks=2"),                # per-interpreter random seeds are exactly what the line removes: 0, and the caller's word is named
    ("", "0", "hashseed=0 source=default ranks=2"),                                      # empty is unset (the interpreter's own reading of an empty value)
    ("007", "7", "hashseed=7 source=inherited ranks=2"),                                 # kept in its decimal spelling
    ("4294967295", "4294967295", "hashseed=4294967295 source=inherited ranks=2"),        # the top of the range
    ("4294967296", "0", "hashseed=0 source=default parent='4294967296' ranks=2"),        # out of range: not kept, named
    ("-1", "0", "hashseed=0 source=default parent='-1' ranks=2"),
    ("seed with spaces and length", "0", "hashseed=0 source=default parent='seed with spaces' ranks=2"),   # a discarded value is named by its first 16 characters
)


def test_rank_env_starts_every_rank_with_one_str_hash_seed(monkeypatch, tmp_path):
    """The ranks are separate interpreters (torchrun's workers inherit the launcher's environment unchanged): the run has ONE PYTHONHASHSEED by
    the shared core's rule (opt_core.mem.rowpair.rankdata) — a decimal integer in [0, 4294967295] the caller exported is kept (source
    inherited); unset, empty, `random` or anything else becomes 0 (source default) and a present value that could not be kept is named
    (`parent=`) — exported by rank_env, stated on the ACTIVE line as the core's word `hashseed=<v> source=<default|inherited> [parent=…] ranks=P`
    byte for byte, and kept in the run's record (hashseed / hashseed_source) — env, word and fields from ONE reading (launch_env)."""
    from opt_core.mem.rowpair import rankdata
    assert tp.HASHSEED_ENV == rankdata.HASHSEED_ENV == "PYTHONHASHSEED"
    for pre, value, words in HASHSEED_CASES:
        if pre is None:
            monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        else:
            monkeypatch.setenv("PYTHONHASHSEED", pre)
        explicit = {"PATH": "/bin", **({} if pre is None else {"PYTHONHASHSEED": pre})}
        env, word, fields = tp.launch_env(2, str(tmp_path))                                # ONE reading of the parent's environment (os.environ): env + word + record fields ...
        based = tp.rank_env(2, str(tmp_path), base=explicit)                             # ... and an explicit base mapping: one rule
        assert env["PYTHONHASHSEED"] == based["PYTHONHASHSEED"] == value == fields["hashseed"], (pre, env["PYTHONHASHSEED"], based["PYTHONHASHSEED"])
        assert word == words == tp.hashseed_token(2) == tp.hashseed_token(2, explicit) == rankdata.hashseed_word(explicit, 2), (pre, word)
        assert fields == rankdata.hashseed_fields(explicit) and ("hashseed_parent" in fields) == ("parent=" in words)
        assert f" launcher=torchrun_loopback {words} base=fast " in tp.active_line("[protenix-opt]", 2, {}, hashseed=word), "the launch's word, stated before any rank starts"
        assert f" launcher=torchrun_loopback {words} base=fast " in tp.active_line("[protenix-opt]", 2, {}), "a caller without a launch reads the same word"
    monkeypatch.setenv("PYTHONHASHSEED", "123")                                          # the caller's integer wins, in the ranks, in the record and on the line, for any P
    env, word, fields = tp.launch_env(4, str(tmp_path))
    assert env["PYTHONHASHSEED"] == "123" and fields == {"hashseed": "123", "hashseed_source": "inherited"} and word == "hashseed=123 source=inherited ranks=4"
    assert " hashseed=123 source=inherited ranks=4 " in tp.active_line("[protenix-opt]", 4, {}, hashseed=word)
    assert tp.launch_env(2, str(tmp_path), base={"PATH": "/bin", "PYTHONHASHSEED": "random"})[1:] == ("hashseed=0 source=default parent='random' ranks=2", {"hashseed": "0", "hashseed_source": "default", "hashseed_parent": "random"})


def test_spawned_rank_interpreters_report_one_hash_seed(monkeypatch, tmp_path):
    """Two interpreters started on rank_env's environment (what Popen hands the launcher, and torchrun every worker) report the same non-empty
    PYTHONHASHSEED and the same str hash — with the caller's seed unset (0), an integer exported (kept) and `random` exported (0: the case
    where two interpreters would otherwise draw two seeds)."""
    probe = "import os; print(os.environ.get('PYTHONHASHSEED'), hash('rowpair'))"
    for pre, value in ((None, "0"), ("123", "123"), ("random", "0")):
        if pre is None:
            monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        else:
            monkeypatch.setenv("PYTHONHASHSEED", pre)
        env = tp.rank_env(2, str(tmp_path))
        outs = [subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True).stdout.split() for _ in range(2)]
        assert outs[0] == outs[1] and outs[0][0] == value and len(outs[0]) == 2, (pre, outs)


def test_command(tmp_path):
    cmd = tp.command(4, ["--input", "x", "--out_dir", "o"], str(tmp_path))
    assert cmd[:3] == [sys.executable, "-m", tp.LAUNCHER_MODULE]
    assert cmd[3:5] == ["--nproc", "4"] and all(a in cmd for a in tp.LINE_ARGS) and "--dap" not in cmd and "--det" not in cmd
    assert cmd[cmd.index("--") + 1:] == ["pred", "--input", "x", "--out_dir", "o"], "the stock arguments pass through after `--`"
    assert cmd[cmd.index("--out") + 1] == os.path.join(str(tmp_path), tp.PHASE_LOG_DIR)


# ------------------------------------------------------------------------------------------------------------ events + gate
def test_events_ok_when_every_rank_reports_and_finishes(tmp_path):
    ev = tp.events(_run_dir(tmp_path, 2), 2)
    assert ev["ok"] is True and ev["reason"] == "" and ev["world_size"] == 2 and ev["backend"]["nccl"] == "2.29.7"
    assert sorted(ev["ranks"]) == ["0", "1"] and ev["ranks"]["1"]["device"] == "cuda:1" and ev["layout"]["block"] == 128
    assert ev["phases"]["0"]["last"] == tp.FINAL_PHASE and ev["phases"]["0"]["samples_finished"] == 1 and ev["phases"]["1"]["max_alloc_GiB"] == 6.0
    assert ev["stock_arg_delta"] == ["--trimul_kernel", "torch"] and ev["line_args"] == list(tp.LINE_ARGS) and ev["n_gpu"] == 2 and ev["sharding"] == "rowpair"
    assert ev["seams_applied"]["1"]["seams"]["pairformer"] == "tp" and ev["seams_applied"]["0"]["relp"] == "lazy" and ev["parse_failures"] == {}
    assert ev["seams_effective"]["confidence"] == "tp" and "seam_policy" not in ev, "the forward's seams record is named"
    assert ev["backend"]["collective"] == "nccl" and "dist.py" in ev["backend"]["rule"] and ev["pg_timeout_s"] == 1800
    assert ev["shard_map"] == {"n_tokens": 1592, "world": 2, "block": 128, "replicated": False, "rows_per_rank": [[0, 896], [896, 1592]]}
    assert ev["not_applied"] == {} and ev["dropped_levers"] == list(tp.TP_DROPPED) and ev["runmeta"]["n_token"] == 1592


def test_a_rank_whose_unit_layers_did_not_route_to_the_core_is_not_ok(tmp_path):
    out = _run_dir(tmp_path / "r", 2)
    for r in range(2):                                # per-rank files: rank 1's log has no TP-ROUTE lines
        d = tmp_path / "r" / "out" / tp.RANK_LOG_DIR / "run0" / "attempt_0" / str(r); d.mkdir(parents=True)
        text = BANNER.format(n=2) + RANK.format(r=r, n=2) + APPLIED.format(r=r, n=2) + (EFFECTIVE.format(n=2) if r == 0 else "")
        (d / "stderr.log").write_text(text if r == 0 else text.replace(ROUTE, ""))
    ev = tp.events(out, 2, block=128)
    assert ev["ok"] is False and ev["routed"] == {0: sorted(tp.tp_route.ROUTES), 1: []}, ev["routed"]
    assert ev["reason"].startswith("ranks whose carried-unit layers did not route to the core: rank1: routed none of "), ev["reason"]
    (tmp_path / "r" / "out" / tp.RANK_LOG_DIR / "run0" / "attempt_0" / "1" / "stderr.log").write_text(BANNER.format(n=2) + RANK.format(r=1, n=2) + APPLIED.format(r=1, n=2))
    ev = tp.events(out, 2, block=128)
    assert ev["ok"] is True and tp.routed_count(ev) == 2, ev["reason"]


def test_events_read_banners_that_share_a_line(tmp_path):
    out = _run_dir(tmp_path, 2, ranks=[])
    with open(os.path.join(out, tp.PHASE_LOG_DIR, tp.LAUNCH_LOG), "a") as fh:
        fh.write(RANKS_ON_ONE_LINE)                   # the ranks' stderr interleaves: two banners on one line (seen on a box)
    ev = tp.events(out, 2)
    assert sorted(ev["ranks"]) == ["0", "1"] and ev["ok"] is True, ev["reason"]


def test_events_name_every_way_a_run_falls_short(tmp_path):
    assert "reason=n_gpu_mismatch requested=4 active=2" in tp.events(_run_dir(tmp_path / "a", 2), 4)["reason"]
    assert "ranks without a banner: [1]" in tp.events(_run_dir(tmp_path / "b", 2, ranks=[0]), 2)["reason"]
    assert "ranks without a phase log: [1]" in tp.events(_run_dir(tmp_path / "c", 2, logs=[0]), 2)["reason"]
    out = _run_dir(tmp_path / "d", 2)
    _phase_log(os.path.join(out, tp.PHASE_LOG_DIR), 1, phases=("trunk_start", "diffusion_start"))
    ev = tp.events(out, 2)
    assert ev["ok"] is False and "did not reach confidence_end: [1]" in ev["reason"]
    assert tp.events(str(tmp_path / "nothing"), 2)["reason"].startswith("reason=n_gpu_mismatch requested=2 active=unread")
    ev = tp.events(_run_dir(tmp_path / "e", 2, applied=[0]), 2)
    assert ev["ok"] is False and "without the unit's APPLIED line: [1]" in ev["reason"]
    out = _run_dir(tmp_path / "f", 2, applied=[0], extra=APPLIED.format(r=1, n=2).replace("pairformer=tp", "pairformer=other"))
    ev = tp.events(out, 2)
    assert ev["ok"] is False and "rank1:pairformer seam pairformer=tp not readable on rank 1's APPLIED line" in ev["reason"]
    out = _run_dir(tmp_path / "g", 2, extra="[ptx_tp r1] NOT applied (PTX_TP='2'); stock behaviour\n")
    ev = tp.events(out, 2)
    assert ev["ok"] is False and "refused to apply in rank(s) 1: PTX_TP='2'" in ev["reason"]
    glued = APPLIED.format(r=1, n=2).rstrip("\n") + "2026-09-01 10:20:01,001 INFO [protenix.runner] some stock logger line\n"   # a shared-pipe glue seen at 8 ranks
    out = _run_dir(tmp_path / "i", 2, applied=[0], extra=glued)
    ev = tp.events(out, 2)
    assert ev["ok"] is True and ev["seams_applied"]["1"]["seams"]["trunk"] == "tp" and ev["parse_failures"] == {}, ev["reason"]
    out = _run_dir(tmp_path / "j", 2, applied=[0], extra=APPLIED.format(r=1, n=2).replace(" trunk=tp", ""))
    ev = tp.events(out, 2)
    assert ev["ok"] is False and "rank1:trunk" in ev["parse_failures"] and "not a fallback" in ev["reason"]
    out = _run_dir(tmp_path / "k", 2, extra="[ptx_tp r0] N_token=190 < P*B=256 (or labels given): replicated regime -> STOCK main loop on every rank\n")
    ev = tp.events(out, 2)
    assert ev["ok"] is False and ev["regime"] == "replicated" and "REPLICATED regime" in ev["reason"] and ev["replicated"] == [{"n_tokens": 190, "p_times_b": 256}]
    out = _run_dir(tmp_path / "l", 2, extra="[ptx_tp r0] WARNING layout B=128: ranks [7] own ZERO rows at N=1592, P=8 (trunk is R==0-safe; diffusion/confidence tails may not be) -- launch via ptx_tp.launch (auto-B) or set PTX_TP_B to one of 64/32/16\n")
    ev = tp.events(out, 2)
    assert ev["ok"] is False and ev["empty_ranks"] == [{"block": 128, "ranks": [7], "n_tokens": 1592, "world": 8}] and "zero-row ranks" in ev["reason"]
    ev = tp.events(_run_dir(tmp_path / "m", 2), 2, block=64)
    assert ev["ok"] is True and ev["regime"] == "tp" and ev["shard_map"]["block"] == 64 and ev["shard_map"]["n_tokens"] == ev["runmeta"]["n_token"]


# ------------------------------------------------------------------------------------------------------------ ranks' records
def _records(path, recs, tallies=None):
    with open(path, "w") as fh:
        for pid, rec in recs.items():
            fh.write(json.dumps({"pid": pid, "clisampler": {"installed": True, "sampler": {"captures": 0, "replays": 0}}, **(tallies or {})}) + "\n")
            fh.write(json.dumps({"pid": pid, "protenix_opt": rec, "ptx_tp": {"rank": str(pid % 100), "seams": "pairformer=tp", "diffusion": {"mode": "tp", "precision": "fp32", "skip_amp": True, "N": 1592, "P": 2}}}) + "\n")


def test_reconcile_ranks_applied_in_all_fallen_back_in_any(tmp_path):
    p = str(tmp_path / "lever_report.jsonl")
    base = {"mode": "big", "levers_applied": ["a", "b", "c"], "levers_fallback": [], "fallback_reasons": {}, "partial": False}
    _records(p, {11: {"active": True, "levers_applied": ["a", "b", "c"], "levers_fallback": []},
                 12: {"active": True, "levers_applied": ["a", "b"], "levers_fallback": ["c"], "fallback_reasons": {"c": "no cell"}}})
    r = tp.reconcile_ranks(base, p, 2)
    assert r["levers_applied"] == ["a", "b"] and r["levers_fallback"] == ["c"] and r["partial"] is True
    assert r["fallback_reasons"]["c"] == "rank pid 12: no cell" and r["reconciled"]["ranks_reported"] == 2 and r["reconciled"]["moves"] == {"c": "applied -> fallback"}
    _records(p, {11: {"active": True, "levers_applied": ["a", "b", "c"], "levers_fallback": []}, 12: {"active": True, "levers_applied": ["a", "b", "c"], "levers_fallback": []}})
    r = tp.reconcile_ranks(base, p, 2)
    assert r["partial"] is False and r["levers_fallback"] == [] and r["levers_applied"] == ["a", "b", "c"]
    r = tp.reconcile_ranks(base, p, 3)
    assert r["partial"] is True and "rank_record_missing" in r["levers_fallback"] and "2 of 3 ranks" in r["fallback_reasons"]["rank_record_missing"]
    r = tp.reconcile_ranks(base, str(tmp_path / "absent.jsonl"), 2)
    assert r["partial"] is True and r["levers_fallback"] == ["rank_record_missing"] and r["reconciled"]["ranks_reported"] == 0
    _records(p, {11: {"active": False, "reason": "stub refused", "levers_applied": [], "levers_fallback": []}, 12: {"active": True, "levers_applied": ["a", "b", "c"], "levers_fallback": []}})
    r = tp.reconcile_ranks(base, p, 2)
    assert "rank_not_active" in r["levers_fallback"] and r["levers_applied"] == [] and "stub refused" in r["fallback_reasons"]["rank_not_active"]


def test_reconcile_ranks_drops_the_graph_levers_by_name_and_reads_the_tallies(tmp_path):
    """A graph lever the ranks report fallen back (left uninstalled by TP_PRE) is dropped by the line, never a fallback; the tallies
    say which applied levers executed (deadskip n=41), which were inert under TP (trimul_core lever_calls 0, k2b served 0), which have no counter."""
    p = str(tmp_path / "lever_report.jsonl")
    base = {"mode": "big", "levers_applied": ["trimul_core", "k2b_flash_triattention", "deadskip", "lazy_init", "stackgraph", "sampler_graph"], "levers_fallback": [], "fallback_reasons": {}, "partial": False}
    tall = {"trimul_routes": {"lever_calls": 0, "passthrough": 0}, "blk_att_served": {"k2b": 0, "cueq": 0}, "deadskip": {"n": 41}}
    _records(p, {11: {"active": True, "levers_applied": ["trimul_core", "k2b_flash_triattention", "deadskip", "lazy_init"], "levers_fallback": ["stackgraph", "sampler_graph"],
                      "fallback_reasons": {"stackgraph": "no marker", "sampler_graph": "no marker"}},
                 12: {"active": True, "levers_applied": ["trimul_core", "k2b_flash_triattention", "deadskip", "lazy_init"], "levers_fallback": ["stackgraph", "sampler_graph"]}}, tallies=tall)
    r = tp.reconcile_ranks(base, p, 2)
    assert r["partial"] is False and r["levers_fallback"] == [] and r["levers_dropped_by_line"] == ["stackgraph", "sampler_graph"]
    assert r["levers_applied"] == ["deadskip", "k2b_flash_triattention", "lazy_init", "trimul_core"]
    assert r["levers_executed"] == ["deadskip"] and r["levers_inert_under_tp"] == ["k2b_flash_triattention", "trimul_core"] and r["levers_env_only"] == ["lazy_init"]
    assert r["execution"]["inert_under_tp"]["trimul_core"] == {"11": 0, "12": 0} and r["execution"]["executed"]["deadskip"] == {"11": 41, "12": 41}
    assert r["reconciled"]["moves"] == {"stackgraph": "applied -> dropped_by_line", "sampler_graph": "applied -> dropped_by_line"}
    line = tp.execution_line("[p]", r)
    assert line.startswith("[p] EXECUTION line=tp levers_executed=deadskip:41 levers_inert_under_tp=k2b_flash_triattention:0,trimul_core:0 env_only=lazy_init dropped_by_line=stackgraph,sampler_graph")


# ------------------------------------------------------------------------------------------------------------ the unit
def test_unit_is_carried_and_importable():
    assert "PTX_TP" in kits.KITS and os.path.isfile(os.path.join(tp.unit_dir(), "ptx_tp", "launch.py"))
    rels = tp.unit_rels()
    assert len(rels) == len(kits.tree_files(kits.kit_dir("PTX_TP"))) > 0 and all(r.startswith("forward/PTX_TP/") for r in rels), len(rels)
    problems = kits.missing_kit_files(rels)
    assert problems == [], problems
    sm = tp.shard_map(1592, 2)
    assert sm == {"n_tokens": 1592, "world": 2, "block": 128, "replicated": False, "rows_per_rank": [[0, 896], [896, 1592]]}, "the row grid: 13 blocks of 128, 7 per rank"
    r = tp.shard_map(100, 4)
    assert r["replicated"] is True and r["rows_per_rank"] == [] and "cannot shard" in r["refused"]           # reported as refused, never launched
    from opt_core.mem.rowpair import dist as core_dist                                                        # the one row-grid arithmetic (serves the unit's ptx_tp.dist in every rank)
    assert tp._layout(1592, 2, 128).bounds == core_dist.Layout(1592, 2, 0, B=128).bounds


def test_preflight_names_the_gap_without_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    pf = tp.preflight(2)
    assert pf["n_gpu"] == 2 and pf["unit_files"] == len(tp.unit_rels()) and "selector" not in pf
    if pf["ok"]:
        pytest.skip("a box with >= 2 visible GPUs despite CUDA_VISIBLE_DEVICES='' (torch imported earlier)")
    assert any(w in pf["reason"] for w in ("visible", "torch", "NCCL", tp.AUTOLOAD_PTH)), pf["reason"]
    if "visible" in pf["reason"]:
        assert pf["reason"] == f"refused: n_gpu=2 visible={pf['visible_gpus']}", "the shared core's sentence, byte for byte (never shrunk to the visible count)"
    if tp.autoload_installed() is None:
        assert tp.AUTOLOAD_PTH in pf["reason"], "a PYTHONPATH-only package (no .pth) is named as the gap before torch is consulted"
    monkeypatch.setattr(tp, "autoload_installed", lambda executable=None: None)
    assert tp.AUTOLOAD_PTH in tp.preflight(2)["reason"]


# ------------------------------------------------------------------------------------------------------------ the CLI routes
def test_cli_n_gpu_above_one_with_another_mode_is_usage_and_one_is_every_modes_path(monkeypatch, capsys):
    monkeypatch.setattr(protenix_opt, "MODES", {"exact": ["a"], "fast": ["b"], "big": ["a"], "off": []}, raising=False)
    monkeypatch.delenv("PROTENIX_OPT", raising=False); monkeypatch.delenv(tp.ENV_NGPU, raising=False)
    for mode in ("fast", "exact", "off"):
        assert cli.main(["pred", "--mode", mode, "--n_gpu", "2", "--input", "x", "--out_dir", "o"]) == cli.EXIT_USAGE
        assert "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)" in capsys.readouterr().err
    assert cli.DEFAULT_MODE == "fast" and cli.main(["pred", "--n_gpu", "2", "--input", "x", "--out_dir", "o"]) == cli.EXIT_USAGE   # no --mode = fast: the same sentence, before anything runs
    assert "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)" in capsys.readouterr().err and not os.path.exists("o")
    assert cli.main(["pred", "--mode", "big", "--n_gpu", "x", "--input", "x"]) == cli.EXIT_USAGE
    assert cli.main(["pred", "--mode", "big", "--n_gpu", "0", "--input", "x"]) == cli.EXIT_USAGE
    assert cli.main(["check", "--mode", "fast", "--n_gpu", "2"]) == cli.EXIT_USAGE
    assert "refused: n_gpu>1 requires --mode big" in capsys.readouterr().err
    assert not hasattr(tp, "FLAG_REFERENCE") and not hasattr(tp, "execution_reference")                                # the line's own options are --n_gpu only
    assert "unknown --mode" not in capsys.readouterr().err


def test_cli_n_gpu_one_is_the_single_line_under_big(monkeypatch, capsys):
    """`--n_gpu 1` (or no flag) with --mode big never enters the multi-GPU route (tp_check / tp_pred_route): the single line's own
    check runs in-process."""
    monkeypatch.setattr(protenix_opt, "MODES", {"exact": ["a"], "fast": ["b"], "big": ["a"], "off": []}, raising=False)
    monkeypatch.delenv("PROTENIX_OPT", raising=False); monkeypatch.delenv(tp.ENV_NGPU, raising=False)
    called = {}
    monkeypatch.setattr(cli, "tp_check", lambda core, n, a: called.setdefault("tp", n) and 0)
    monkeypatch.setattr(cli.det, "plan", lambda: (_ for _ in ()).throw(cli.CliError("single-line check reached")))   # the single line's first step (under --det 1)
    for argv in (["check", "--mode", "big", "--n_gpu", "1", "--det", "1"], ["check", "--mode", "big", "--det", "1"]):
        assert cli.main(argv) == cli.EXIT_USAGE and "single-line check reached" in capsys.readouterr().err
        assert "tp" not in called, argv


def test_cli_check_big_without_n_gpus_is_not_active_by_name(monkeypatch, capsys):
    monkeypatch.setattr(protenix_opt, "MODES", {"exact": ["a"], "fast": ["b"], "big": ["a"], "off": []}, raising=False)
    monkeypatch.delenv("PROTENIX_OPT", raising=False); monkeypatch.delenv(tp.ENV_NGPU, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(cli, "activate", lambda core, mode, dry_run=False, det_level=None: {"mode": mode, "dry_run": True, "active": False, "env": {"X": "1"},
                                                                                             "levers_applied": ["b"], "levers_fallback": [], "partial": False})
    monkeypatch.setattr(tp, "preflight", lambda n: {"ok": False, "reason": "refused: n_gpu=2 visible=0", "n_gpu": n,
                                                    "unit_files": len(tp.unit_rels()), "visible_gpus": 0})
    assert cli.main(["check", "--mode", "big", "--n_gpu", "2"]) == cli.EXIT_NOT_ACTIVE
    err = capsys.readouterr().err
    assert "NOT ACTIVE: mode=big line=tp n_gpu=2 sharding=rowpair impl=ptx_tp base=fast" in err and "reason=refused: n_gpu=2 visible=0" in err


def test_cli_pred_line_runs_the_launcher_and_gates_on_the_ranks(monkeypatch, tmp_path, capsys, fake_stock):
    """The route end to end with the launcher replaced: preflight ok, the base mode's dry run, the launcher writes 2 ranks' logs and
    records + the stock output layout; a launcher that loses a rank makes the run FAIL by name."""
    from protenix_opt.tests import _stock_stub
    monkeypatch.setattr(protenix_opt, "MODES", {"exact": ["a"], "fast": ["b"], "big": ["a"], "off": []}, raising=False)
    monkeypatch.delenv("PROTENIX_OPT", raising=False); monkeypatch.delenv(tp.ENV_NGPU, raising=False); monkeypatch.delenv(tp.HASHSEED_ENV, raising=False)
    monkeypatch.setattr(cli, "activate", lambda core, mode, dry_run=False, det_level=None: {"mode": mode, "dry_run": True, "active": False, "env": {"PYTORCH_CUDA_ALLOC_CONF": "x"},
                                                                                             "levers_applied": ["b"], "levers_fallback": [], "partial": False, "protenix_version": "2.0.0"})
    monkeypatch.setattr(tp, "preflight", lambda n: {"ok": True, "reason": "", "n_gpu": n, "unit_files": 1, "visible_gpus": n, "nccl": "2.29.7"})
    seen = {}

    def fake_run(n, rest, records, label=tp.LINE, env=None, lose_rank=False):
        seen.update(n=n, rest=list(rest), env=dict(env or {}), records=records)
        out_dir = rest[rest.index("--out_dir") + 1]                                     # stock's outputs go to --out_dir; the line's records to the records dir
        assert not records.startswith(out_dir)
        tp_dir = os.path.join(records, tp.PHASE_LOG_DIR); os.makedirs(tp_dir, exist_ok=True)
        open(os.path.join(tp_dir, tp.LAUNCH_LOG), "w").write(BANNER.format(n=n) + "".join(RANK.format(r=r, n=n) + APPLIED.format(r=r, n=n) for r in range(n)) + EFFECTIVE.format(n=n))
        for r in range(n if not seen.get("lose") else n - 1):
            _phase_log(tp_dir, r)
        _records(env["PTX_LEVER_REPORT"], {100 + r: {"active": True, "levers_applied": ["b"], "levers_fallback": []} for r in range(n)})
        params = {"input": "x", "out_dir": out_dir, "seeds": "101", "num_samples": 5, "model_name": "protenix-v2"}
        _stock_stub.write_outputs(params)
        return 0, os.path.join(tp_dir, tp.LAUNCH_LOG)

    monkeypatch.setattr(tp, "run", fake_run)
    out = tmp_path / "o"
    inp = tmp_path / "in.json"; inp.write_text(json.dumps([{"name": "x", "sequences": [{"proteinChain": {"sequence": "A" * 800, "count": 2}}]}]))
    rc = cli.main(["pred", "--mode", "big", "--n_gpu", "2", "--input", str(inp), "--out_dir", str(out), "--seeds", "101"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_OK, err
    assert seen["n"] == 2 and seen["env"]["PROTENIX_OPT"] == "fast" and seen["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "x"
    assert seen["env"][tp.HASHSEED_ENV] == "0", "the environment handed to the launcher carries the run's one str-hash seed"
    assert "ACTIVE mode=big line=tp n_gpu=2 sharding=rowpair impl=ptx_tp routed=dist,blockreduce,bcast,contract->opt_core.mem.rowpair bound=trunk,template,diffusion,pairformer,msa->protenix_opt.tp_bind across_P=tier2(gemm_m=rows_per_rank) inplace_chunk=256 noise_sync=bcast launcher=torchrun_loopback hashseed=0 source=default ranks=2 base=fast" in err and "EXECUTION line=tp" in err and " diffusion_regime=tp,fp32 " in err and "TP-LAYOUT items=1 tokens_min=1600 tokens_max=1600 n_gpu=2 block=128" in err
    assert seen["env"][tp.ENV_BLOCK] == "128" and not os.path.exists(out / tp.PHASE_LOG_DIR)      # the line's records never land under --out_dir
    small = tmp_path / "small.json"; small.write_text(json.dumps([{"name": "s", "sequences": [{"proteinChain": {"sequence": "A" * 20, "count": 1}}]}]))
    assert cli.main(["pred", "--mode", "big", "--n_gpu", "2", "--input", str(small), "--out_dir", str(tmp_path / "q"), "--seeds", "101"]) == cli.EXIT_NOT_ACTIVE
    assert "no layout block of 128/64/32/16 serves every item" in capsys.readouterr().err
    finals = [l for l in err.splitlines() if "] FINAL " in l]
    assert len(finals) == 1 and finals[0].startswith("[protenix-opt] FINAL mode=big line=tp n_gpu=2 sharding=rowpair levers=") and " world_size=2 ranks=2 finished=2 backend=nccl" in finals[0] and "ok=true" in finals[0], finals
    assert " partial=false " in finals[0] and not [f for f in os.listdir(out) if f.endswith(".json")], "no kit file at the output directory's root (the stand-in's stock tree only)"
    seen["lose"] = True
    rc = cli.main(["pred", "--mode", "big", "--n_gpu", "2", "--input", str(inp), "--out_dir", str(tmp_path / "p"), "--seeds", "101"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_FAIL and "ok=false" in err and "ranks without a phase log: [1]" in err


# ------------------------------------------------------------------------------------------------------------ the env route
def test_env_route_refuses_the_line_in_process_and_n_gpu_above_one_beside_another_mode(tmp_path, monkeypatch):
    """PROTENIX_OPT=big with PROTENIX_OPT_N_GPU=P>1 in a process that imports the kit: NOT ACTIVE by name naming the launcher, exit 3 (never
    one GPU under the line's name; P=1 or no count is the mode's single-GPU line and activates); PROTENIX_OPT_N_GPU>1 beside another selection:
    NOT ACTIVE by name, exit 3 (the autoload's gate, before any import); a malformed count is refused by name."""
    assert "big" in _autoload.MODES and tp.ENV_NGPU in _autoload.DECLARED
    monkeypatch.setenv(tp.ENV_NGPU, "2")
    rep = stack.activate("big")
    assert rep["active"] is False and rep.get("refused") is True and "multi-GPU line" in rep["reason"] and "--n_gpu P" in rep["reason"]
    with pytest.raises(protenix_opt.ActivationError):
        stack.activate("big", strict=True)
    monkeypatch.setenv(tp.ENV_NGPU, "two")
    rep = stack.activate("big")
    assert rep["active"] is False and rep.get("refused") is True and "PROTENIX_OPT_N_GPU" in rep["reason"] and "positive" in rep["reason"]
    monkeypatch.delenv(tp.ENV_NGPU)
    opt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(_autoload.__file__))))
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_OPT")}
    env.update(PROTENIX_OPT="fast", PROTENIX_OPT_N_GPU="2", PYTHONPATH=os.pathsep.join([os.path.join(opt, "opt")] + [p for p in sys.path if p]))
    r = subprocess.run([sys.executable, "-c", "import protenix_opt._autoload as A; A.install(); print('INSTALLED')"], env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "NOT ACTIVE: PROTENIX_OPT_N_GPU=2 set with PROTENIX_OPT=fast: refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)" in r.stderr and "INSTALLED" not in r.stdout, r.stderr
    env["PROTENIX_OPT"] = "off"
    r = subprocess.run([sys.executable, "-c", "import protenix_opt._autoload as A; A.install(); print('INSTALLED')"], env=env, capture_output=True, text=True)
    assert r.returncode == 3, "the GPU count beside PROTENIX_OPT=off is a mistyped selection too"


def test_classification_reads_a_pre_exported_allocator_policy_as_in_effect():
    """The ranks initialise CUDA before the entry imports the kit; when the launching process exported the mode's allocator policy
    first (tp.rank_env, PRE_CUDA_KEYS) the kit's note says so and xl_policy is not a fallback; the plain 'initialised before' note still is."""
    environ = {"PTX_FPF_XL": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    ok_note = f"{stack.ALLOC_CONF}=expandable_segments:True was set before CUDA initialised (exported by the launching process; env.sh keeps a caller's value): the allocator policy is in effect"
    bad_note = f"CUDA was initialised before activation: {stack.ALLOC_CONF} cannot take effect in this process"
    on_ok, fb_ok, _, why_ok = stack._classify("fast", [], environ, notes=[ok_note])
    on_bad, fb_bad, _, why_bad = stack._classify("fast", [], environ, notes=[bad_note])
    assert "allocator policy not in effect" in why_bad.get("xl_policy", ""), why_bad
    assert "allocator policy not in effect" not in why_ok.get("xl_policy", ""), why_ok.get("xl_policy")


def test_block_rule_keeps_every_item_out_of_the_replicated_regime_and_every_rank_non_empty():
    """The unit's own Layout decides: no item replicated (N < P*B) AND no zero-row rank for any item — the launcher's auto-B table
    (runs seen: 1,592/8 -> 32, 1,956/8 -> 128, 4,076/8 -> 128, 600/4 -> 64, 1,592-2,500/4 -> 128, 400/2 -> 128, 185-239/2 -> 64)
    and the hazard bands where a plain 'lo >= P*B' rule would leave a rank empty (1,592/8 at B=128: rows 256x6, 56, 0)."""
    assert tp.block_for([1592], 2)[0] == 128 and tp.block_for([400], 2)[0] == 128
    assert tp.block_for([1956], 8)[0] == 128 and tp.block_for([4076], 8)[0] == 128 and tp.block_for([1592, 2500], 4)[0] == 128 and tp.block_for([2956], 8)[0] == 128
    assert tp.block_for([185, 190, 235, 239], 2)[0] == 64, "the b200 items on 2 ranks: 128 would replicate (190 < 256); 64 keeps every item on the line"
    assert tp.block_for([1592], 8)[0] == 32 and tp.block_for([600], 4)[0] == 64 and tp.block_for([992], 8)[0] == 64 and tp.block_for([1012], 8)[0] == 64
    # an input that needs a grid finer than the supported 128/64/32 (B=16) is refused by name
    b, why = tp.block_for([250], 8)
    assert b is None and "B=16 layout grid" in why and "not supported" in why and "at least 256 tokens" in why, why
    assert tp.BLOCKS_SUPPORTED == (128, 64, 32)
    for N, P in ((1592, 8), (2500, 8), (600, 4), (1200, 8), (2049, 8), (513, 4)):            # served by the 128 / 64 / 32 grids
        B, why = tp.block_for([N], P)
        assert B is not None, (N, P, why)
        sm = tp.shard_map(N, P, B)
        assert not sm["replicated"] and all(b - a > 0 for a, b in sm["rows_per_rank"]), (N, P, B, sm)
    B, why = tp.block_for([120], 8)
    assert B is None and "cannot shard" in why and "no layout block" in why
    B, why = tp.block_for([1025], 8)                                                             # 128/64/32 leave a rank empty; only B=16 serves it: not supported -> refused by name
    assert B is None and "B=16 layout grid" in why and "not supported" in why
    assert tp.rows_line([1592], 8, 32).startswith("rows[1592]=0-224,224-448,")


def test_item_tokens_and_input_path(tmp_path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps([{"name": "a", "sequences": [{"proteinChain": {"sequence": "A" * 100, "count": 2}}, {"ligand": {"ligand": "CCO"}}]},
                             {"name": "b", "sequences": [{"proteinChain": {"sequence": "A" * 300, "count": 1}}]}]))
    assert tp.item_tokens(str(p)) == [200, 300] and tp.input_path(["pred", "--input", str(p), "--seeds", "1"]) == str(p) and tp.input_path(["pred"]) is None


def test_execution_line_names_counts(tmp_path):
    rep = {"execution": {"executed": {"blk2_block_path": {"251": 80}, "deadskip": {"251": 41}}, "inert_under_tp": {"trimul_core": {"251": 0}}},
           "levers_env_only": ["lazy_init"], "levers_dropped_by_line": ["stackgraph"]}
    line = tp.execution_line("[protenix-opt]", rep)
    assert line == "[protenix-opt] EXECUTION line=tp levers_executed=blk2_block_path:80,deadskip:41 levers_inert_under_tp=trimul_core:0 env_only=lazy_init dropped_by_line=stackgraph"

def test_per_rank_stderr_files_are_the_ledger_source_when_every_rank_has_one(tmp_path):
    """torchrun's per-rank stderr files (PET_LOG_DIR / PET_REDIRECTS / PET_TEE in the rank env) carry every rank's ledger lines without
    the shared pipe's interleaving or loss: with a file per rank the pipe decides nothing about the ranks (a rank's APPLIED line lost
    on the pipe is read from its file); with an incomplete set the pipe is the source and the loss reads as a missing line, named."""
    env = tp.rank_env(2, str(tmp_path / "o"))
    assert env["PET_LOG_DIR"] == str(tmp_path / "o" / tp.RANK_LOG_DIR) and env["PET_REDIRECTS"] == "3" and env["PET_TEE"] == "3"
    # the pipe lost rank 1's APPLIED line
    out = _run_dir(tmp_path / "a", 2, applied=[0])
    ev = tp.events(out, 2, block=128)
    assert ev["ok"] is False and "ranks without the unit's APPLIED line: [1]" in ev["reason"] and ev["ledger_source"]["form"] == "shared_pipe"
    # the same run with a file per rank: every ledger line present in its own file
    for r in range(2):
        d = tmp_path / "a" / "out" / tp.RANK_LOG_DIR / "run0" / "attempt_0" / str(r); d.mkdir(parents=True)
        (d / "stderr.log").write_text(BANNER.format(n=2) + RANK.format(r=r, n=2) + APPLIED.format(r=r, n=2))
        (d / "stdout.log").write_text(EFFECTIVE.format(n=2) if r == 0 else "")          # the effective-seams line is a stdout line
    ev = tp.events(out, 2, block=128)
    assert ev["ok"] is True and ev["ledger_source"]["form"] == "per_rank_files" and sorted(ev["ledger_source"]["files"]) == ["0", "1"], ev["reason"]
    assert all(len(ps) == 2 for ps in ev["ledger_source"]["files"].values())
    assert sorted(ev["seams_applied"]) == ["0", "1"]
    assert ev["seams_effective"] and ev["seams_effective"]["pairformer"] == "tp" and ev["shard_map"]["n_tokens"] > 0, (ev["seams_effective"], ev["shard_map"])
    # one rank's file missing -> the pipe form again, named
    import shutil
    shutil.rmtree(tmp_path / "a" / "out" / tp.RANK_LOG_DIR / "run0" / "attempt_0" / "1")
    ev = tp.events(out, 2, block=128)
    assert ev["ok"] is False and ev["ledger_source"]["form"] == "shared_pipe" and "ranks [0] of 2" in ev["ledger_source"]["note"]


def test_absent_effective_line_fails_by_name(tmp_path):
    """No effective N_token line and no runmeta -> no shard map recorded -> ok=false BY NAME (the gate lives in the reason chain)."""
    out = _run_dir(tmp_path, 2)
    tp_dir = os.path.join(out, tp.PHASE_LOG_DIR); text = open(os.path.join(tp_dir, tp.LAUNCH_LOG)).read()
    open(os.path.join(tp_dir, tp.LAUNCH_LOG), "w").write(text.replace(EFFECTIVE.format(n=2), ""))
    ev = tp.events(out, 2, block=128)
    assert ev["ok"] is False and ev["shard_map"] is None and "no shard map recorded" in ev["reason"], ev["reason"]


def test_diffusion_regime_and_rank_labels_come_from_the_ranks_records(tmp_path):
    """Every rank's record carries the unit's own diffusion statement (report.unit_record: diffusion.report()['modes']['sample_diffusion'])
    and its rank id; the reconcile reads the regime (one statement -> the regime; a disagreement or an absence is named) and labels the
    records rank<r>:pid<pid>; the FINAL line carries diffusion=<mode>,<fp32|bf16> and ledger=<form>."""
    path = tmp_path / "lever_report.jsonl"
    def rec(pid, rank, mode="tp", skip_amp=True):
        return json.dumps({"pid": pid, "protenix_opt": {"active": True, "levers_applied": ["deadskip"], "levers_fallback": [], "fallback_reasons": {}},
                           "ptx_tp": {"rank": str(rank), "seams": "pairformer=tp", "diffusion": {"mode": mode, "precision": "fp32" if skip_amp else "bf16", "skip_amp": skip_amp, "N": 1592, "P": 2}}})
    path.write_text(rec(551, 0) + "\n" + rec(552, 1) + "\n")
    r = tp.reconcile_ranks({"levers_applied": ["deadskip"], "levers_fallback": [], "fallback_reasons": {}}, str(path), 2)
    assert r["reconciled"]["records"] == ["rank0:pid551", "rank1:pid552"]
    assert r["diffusion_regime"]["mode"] == "tp" and r["diffusion_regime"]["precision"] == "fp32" and r["diffusion_regime"]["N"] == 1592
    path.write_text(rec(551, 0) + "\n" + rec(552, 1, mode="tp", skip_amp=False) + "\n")
    r2 = tp.reconcile_ranks({"levers_applied": ["deadskip"], "levers_fallback": [], "fallback_reasons": {}}, str(path), 2)
    assert r2["diffusion_regime"]["mode"] == "disagree" and set(r2["diffusion_regime"]["ranks"]) == {"rank0:pid551", "rank1:pid552"}
    assert r2["partial"] is True and "diffusion_regime_disagree" in r2["levers_fallback"] and "disagree across the ranks" in r2["fallback_reasons"]["diffusion_regime_disagree"]
    assert r["partial"] is False and not [f for f in r["levers_fallback"] if f.startswith("diffusion_regime")]
    path.write_text(json.dumps({"pid": 7, "protenix_opt": {"active": True, "levers_applied": [], "levers_fallback": [], "fallback_reasons": {}}}) + "\n")
    r3 = tp.reconcile_ranks({"levers_applied": [], "levers_fallback": [], "fallback_reasons": {}}, str(path), 1)
    assert r3["diffusion_regime"]["mode"] == "unread" and r3["reconciled"]["records"] == ["rank?:pid7"]
    assert r3["partial"] is True and "diffusion_regime_unread" in r3["levers_fallback"]           # silence is the defect: unread = a named fallback (exit 3 by the partial gate)
    out = _run_dir(tmp_path / "ev", 2); ev = tp.events(out, 2, block=128)
    line = tp.final_line("[protenix-opt]", dict(r, mode="big", levers_applied=["deadskip"], levers_fallback=[], partial=False), ev, 2)
    assert " diffusion_regime=tp,fp32 " in line and " ledger=shared_pipe " in line and line.startswith("[protenix-opt] FINAL mode=big line=tp ")
    path4 = tmp_path / "lever_report_replicated.jsonl"
    path4.write_text(rec(601, 0, mode="replicated") + "\n" + rec(602, 1, mode="replicated") + "\n")
    r4 = tp.reconcile_ranks({"levers_applied": ["deadskip"], "levers_fallback": [], "fallback_reasons": {}}, str(path4), 2)
    assert r4["diffusion_regime"]["mode"] == "replicated" and r4["partial"] is True and "diffusion_regime_replicated" in r4["levers_fallback"]   # the gate open in the ranks = a named fallback
    path5 = tmp_path / "lever_report_bf16.jsonl"
    path5.write_text(rec(701, 0, skip_amp=False) + "\n" + rec(702, 1, skip_amp=False) + "\n")
    r5 = tp.reconcile_ranks({"levers_applied": ["deadskip"], "levers_fallback": [], "fallback_reasons": {}}, str(path5), 2)
    assert (r5["diffusion_regime"]["mode"], r5["diffusion_regime"]["precision"]) == ("tp", "bf16")
    assert r5["partial"] is True and "diffusion_regime_bf16" in r5["levers_fallback"] and "bf16 autocast" in r5["fallback_reasons"]["diffusion_regime_bf16"]   # a bf16 sampler in the ranks = a named fallback (the guard lift keeps fp32)
    assert line.count("diffusion=") == 1                                                     # the seam token only; the regime has its own name


def test_the_replicated_tensor_guards_are_the_det_recipe(monkeypatch, tmp_path):
    """The unit's checksum guards key on the recipe switch itself (ptx_tp.det_recipe reads PTX_DET): the rank environment carries no guard
    switch of its own, under the recipe or outside it (fast's kernels are not bitwise across ranks: the checksums are a statement under the
    recipe only)."""
    for base in ({"PATH": "/bin"}, {"PATH": "/bin", "PTX_DET": "1"}):
        env = tp.rank_env(2, str(tmp_path), base=base)
        assert env[tp.tp_route.ENV] == tp.tp_route.WORD and tp.DET_GUARDS == "PTX_DET" == det.DET_SWITCH
        assert not [k for k in env if k.startswith("PTX_TP_DEBUG")], sorted(env)
        assert env.get("PTX_DET") == base.get("PTX_DET")


def test_n_gpu_reaches_the_launcher_and_a_short_world_is_refused_by_name(monkeypatch, tmp_path, capsys):
    """`pred --mode big --n_gpu P`: P reaches the multi-GPU route and the launcher's argv (`--nproc P`), never folded to one GPU; a run
    whose ranks report another world size is NOT ok by the standard words (reason=n_gpu_mismatch requested=P active=Q)."""
    monkeypatch.setattr(protenix_opt, "MODES", {"exact": ["a"], "fast": ["b"], "big": ["a"], "off": []}, raising=False)
    monkeypatch.delenv("PROTENIX_OPT", raising=False); monkeypatch.delenv(tp.ENV_NGPU, raising=False)
    seen = {}
    def fake_route(core, n_gpu, rest, *a, **k):
        seen["n_gpu"] = n_gpu; seen["argv"] = tp.command(n_gpu, rest, str(tmp_path)); return cli.EXIT_NOT_ACTIVE
    monkeypatch.setattr(cli, "tp_pred_route", fake_route)
    for p in (2, 4, 8):
        cli.main(["pred", "--mode", "big", "--n_gpu", str(p), "--input", "x", "--out_dir", str(tmp_path)])
        assert seen["n_gpu"] == p and seen["argv"][seen["argv"].index("--nproc") + 1] == str(p), seen
    ev = tp.events(_run_dir(tmp_path / "w", 1), 2)     # the ranks report a world of 1 under a request of 2
    assert ev["ok"] is False and ev["reason"].startswith("reason=n_gpu_mismatch requested=2 active=1"), ev["reason"]
