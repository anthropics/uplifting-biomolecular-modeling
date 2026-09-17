"""The n_gpu axis of big: the refusals by name in the core's words, the exact ACTIVE/EXIT tokens, the P set, the template
all-dummy refusal, prev_free in the lever order, the rowpair LEVER line, and the rowpair adapter's binding invariants."""
import json
import os
import re

import pytest

from af3_torch_opt import big, cli, modes, registry, report, stack
from opt_core.mem import ngpu

from .test_cli_manifest import _inputs

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
HOME = os.path.dirname(OPT)


def _read(*rel):
    with open(os.path.join(HOME, *rel), encoding="utf-8") as f:
        return f.read()


def test_p_set_and_default():
    assert modes.N_GPU_DEFAULT == 1
    assert modes.N_GPU_SUPPORTED == (1, 2, 4, 8)                                 # the P set `--mode big --n_gpu P` does not refuse
    assert modes.N_GPU_MODES == ("big",) and ngpu.MEMORY_MODE == "big"


def test_served_set_refuses_every_p_outside_it_by_name(box, monkeypatch):
    """A P outside the served set under big is the core-worded served-set refusal, whatever is visible; under off/fast the mode rule's."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    for P in (3, 5, 6, 7, 16):
        assert stack.n_gpu_gate("big", P) == (P, f"refused: n_gpu={P} not in the supported set {{1,2,4,8}}", None)
        assert stack.n_gpu_gate("fast", P)[1] == ngpu.REFUSE_MODE
    for P in (2, 4, 8):
        assert stack.n_gpu_gate("fast", P)[1] == ngpu.REFUSE_MODE and stack.n_gpu_gate("off", P)[1] == ngpu.REFUSE_MODE
    rep = stack.check("big", n_gpu=3)
    assert not rep["active"] and "refused: n_gpu=3 not in the supported set {1,2,4,8}" in rep["reason"] and rep["n_gpu"] == 3   # never shrunk to 1
    assert stack.check("big", n_gpu=1)["n_gpu"] == 1


MECHANISM_SET = (1, 2, 4)     # the P values the carried row-sharding mechanism admits; the tests below exercise the gate's other rules under it


def test_tokens_exact_text():
    """The activation line's n_gpu/sharding tokens, in order relative to dtk=."""
    assert report.kv(*stack.n_gpu_fields(1)) == "n_gpu=1 sharding=none"
    assert report.kv(*stack.n_gpu_fields(2)) == "n_gpu=2 sharding=rowpair"
    assert report.kv(*stack.n_gpu_fields(4)) == "n_gpu=4 sharding=rowpair"
    on = report.activation_line({"active": True, "mode": "big", "n_gpu": 2})
    assert " n_gpu=2 sharding=rowpair " in on and on.index(" dtk=") < on.index(" n_gpu=2 ")
    assert " n_gpu=1 sharding=none " in report.activation_line({"active": True, "mode": "fast", "n_gpu": 1})
    assert " n_gpu=1 sharding=none " in report.activation_line({"active": True, "mode": "fast"})          # absent == 1
    off = report.activation_line({"active": False, "mode": "fast", "n_gpu": 2, "reason": ngpu.REFUSE_MODE})
    assert off.endswith("reason=" + ngpu.REFUSE_MODE) and " n_gpu=2 " in off


def test_refusals_by_name_in_the_cores_words(monkeypatch):
    monkeypatch.setattr(modes, "N_GPU_SUPPORTED", MECHANISM_SET)
    assert stack.n_gpu_gate("fast", 2) == (2, ngpu.REFUSE_MODE, None)
    assert stack.n_gpu_gate("off", 4)[1] == ngpu.REFUSE_MODE
    assert ngpu.REFUSE_MODE == "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)"
    assert stack.n_gpu_gate("big", 1) == (1, None, None)
    assert stack.n_gpu_gate("fast", 1) == (1, None, None)                     # P = 1 passes under every mode, no probe
    P, why, k = stack.n_gpu_gate("big", 2, {"CUDA_VISIBLE_DEVICES": "0"})
    assert (P, why, k) == (2, "refused: n_gpu=2 visible=1 — use --n_gpu <= 1 on this box, or expose 2 GPUs (CUDA_VISIBLE_DEVICES); a smaller P is never substituted", 1)          # the core's words + the knob (the visible count and --n_gpu), never a smaller P substituted
    assert stack.n_gpu_gate("big", 4, {"CUDA_VISIBLE_DEVICES": "0,1"})[1] == "refused: n_gpu=4 visible=2 — use --n_gpu <= 2 on this box, or expose 4 GPUs (CUDA_VISIBLE_DEVICES); a smaller P is never substituted"
    assert stack.n_gpu_gate("big", 2, {"CUDA_VISIBLE_DEVICES": "3,5"}) == (2, None, 2)
    assert stack.n_gpu_gate("big", 3, {"CUDA_VISIBLE_DEVICES": "0,1,2"})[1] == "refused: n_gpu=3 not in the supported set {1,2,4}"
    for bad in (0, -1, "x", "2.5", True):
        P, why, _ = stack.n_gpu_gate("big", bad)
        assert P == 0 and why.startswith("refused: n_gpu=") and "positive integer" in why, (bad, why)
    assert stack.visible_gpus({"CUDA_VISIBLE_DEVICES": ""}) == 0
    assert stack.visible_gpus({"CUDA_VISIBLE_DEVICES": "1, 2"}) == 2


def test_check_reports_the_refusal_and_never_a_smaller_p(box, monkeypatch):
    monkeypatch.setattr(modes, "N_GPU_SUPPORTED", MECHANISM_SET)
    rep = stack.check("fast", n_gpu=2)
    assert not rep["active"] and ngpu.REFUSE_MODE in rep["reason"] and rep["n_gpu"] == 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    rep = stack.check("big", n_gpu=2)
    assert not rep["active"] and "refused: n_gpu=2 visible=1" in rep["reason"] and rep["n_gpu"] == 2   # never shrunk to 1
    rep1 = stack.check("big", n_gpu=1)
    assert rep1["n_gpu"] == 1 and rep1["sharding"] == "none"


def test_cli_parser_carries_n_gpu():
    ap = cli.build_parser()
    a = ap.parse_args(["pred", "--output_dir", "/tmp/x", "--json_path", "a.json"])
    assert a.n_gpu == 1
    a = ap.parse_args(["pred", "--mode", "big", "--n_gpu", "4", "--output_dir", "/tmp/x", "--json_path", "a.json"])
    assert a.n_gpu == 4
    a = ap.parse_args(["check", "--mode", "big", "--n_gpu", "2"])
    assert a.n_gpu == 2


def test_forward_carries_the_axis_and_the_template_refusal():
    fwd = _read("opt", "af3_torch_opt", "forward.py")
    fwd_impl = _read("opt", "af3_torch_opt", "forward_impl.py")   # _run_trunk_prev_free lives here, imported by forward.py
    for needle in ('"--n-gpu"', "def main_sharded(", "launch.run_sharded(", "def _rank_entry(", "rowpair_xfold.py", '"--templates-declared"',
                   '"peak_mem_gb_ranks"'):
        assert needle in fwd, needle
    assert "def _run_trunk_prev_free(" in fwd_impl and "ConsumeOnce(emb)" in fwd_impl
    c = _read("opt", "af3_torch_opt", "cli.py")
    assert '"--n-gpu", str(rep["n_gpu"])' in c
    assert "--templates-declared" in c and "templates_declared(" in c
    assert "**dict(stack.n_gpu_fields(rep[\"n_gpu\"]))" in c                    # the DONE line's EXIT evidence


def test_templates_declared_counts_the_fold_inputs_lists(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"sequences": [{"protein": {"id": "A", "sequence": "AC", "templates": [{"mmcif": "x"}, {"mmcif": "y"}]}},
                                          {"protein": {"id": "B", "sequence": "AC", "templates": []}}, {"ligand": {"id": "L", "ccdCodes": ["ATP"]}},
                                          {"protein": {"id": "C", "sequence": "AC"}}]}))
    assert cli.templates_declared(str(p)) == 2
    q = tmp_path / "b.json"; q.write_text(json.dumps({"sequences": [{"protein": {"id": "A", "sequence": "AC", "templates": None}}]}))
    assert cli.templates_declared(str(q)) == 0
    assert cli.templates_declared(str(tmp_path / "missing.json")) == 0


def test_prev_free_is_a_shipped_big_lever():
    assert "prev_free" in big.LEVER_ORDER
    assert registry.STRATEGY["prev_free"] == "F7.chunked_eval" and registry.IMPL["prev_free"] == ("forward_impl._run_trunk_prev_free", "kit")
    assert "prev_free" in registry.BIG_LEVERS
    sel = {"levers": list(big.LEVER_ORDER), "settings": {}}
    line, (state, _) = report.lever_line("prev_free", {"mode": "big", "big": sel, "big_record": {"prev_free_calls": 2, "items_ok": 2}})
    assert state == "on" and " strategy=F7.chunked_eval " in line and line.endswith("calls=2")
    line, (state, reason) = report.lever_line("prev_free", {"mode": "big", "big": sel, "big_record": {"prev_free_calls": 0, "items_ok": 0}})
    assert state == "skipped" and reason == "no_item_completed_its_trunk"


def test_rowpair_lever_line():
    assert registry.STRATEGY["rowpair"] == "F7.tensor_parallel" and registry.IMPL["rowpair"] == ("opt_core.mem.rowpair", "core")
    line, (state, reason) = report.lever_line("rowpair", {"mode": "fast", "n_gpu": 1})
    assert (state, reason) == ("off", "n_gpu=1") and " n_gpu=1 sharding=none" in line and " strategy=F7.tensor_parallel " in line
    rec = {"P": 2, "align": 32, "trimul": "rowpair_rows", "triattn": "flash_rows", "heads": "sharded_rows", "diffusion": "zcond_rows+dit_local_queries",
           "dtk": "skipped:n_gpu>1", "core_version": "0.4.3", "prev_free": "rowpair_carry", "stats": {"items": 1, "pair_blocks": 530, "gathers_in_trunk": 0}}
    line, (state, _) = report.lever_line("rowpair", {"mode": "big", "n_gpu": 2, "rowpair_record": rec})
    assert state == "on" and re.search(r" n_gpu=2 sharding=rowpair ranks=2 align=32 trimul=rowpair_rows triattn=flash_rows heads=sharded_rows diffusion=zcond_rows\+dit_local_queries dtk=skipped:n_gpu>1 ", line), line
    assert " gathers_in_trunk=0" in line
    line, (state, reason) = report.lever_line("rowpair", {"mode": "big", "n_gpu": 2})
    assert (state, reason) == ("skipped", "no_rowpair_record(forward.json)")


def test_rowpair_adapter_binds_the_core_seams_and_rebinds_no_class():
    """rowpair_xfold.py drives the shared core's row-sharded seams (one implementation per mechanism: the adapter passes the xfold modules' own
    layers as callables) — it rebinds no xfold class, calls no torch.distributed primitive, and names every seam driver it binds."""
    src = _read("opt", "af3_torch_opt", "rowpair_xfold.py")
    assert not re.search(r"^\s*\w+\.\w+\.forward = _", src, re.M), "no class-level rebinding under 0.4.3"
    assert "torch.distributed" not in src and "dist.all_reduce(" not in src
    assert "conf_full_matrices_rank0" in src and "xfold_replicated" in src          # the replicated-by-design tensors are NAMED


# ---- NGPU-ASSERT (fail-closed): the n_gpu the model process REPORTS (forward.json `n_gpu`, and under n_gpu > 1 the launcher's rank
#      census `world`) must equal the requested --n_gpu, else `NOT ACTIVE … reason=n_gpu_mismatch requested=P active=Q` and rc 3 —
#      never a pass; and the requested P reaches the model process's command line (`--n-gpu P`, read back from the stub's report) ----

def _pred(box, monkeypatch, capsys, n_gpu, **stub_env):
    monkeypatch.setattr(modes, "N_GPU_SUPPORTED", (1, 2, 4))                 # the mechanism's P set for this test (a subset of the shipped (1, 2, 4, 8); matches the visible-device count set on the next line)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")                   # four visible devices: the visible rule passes for P <= 4 on a CPU box
    for k, v in stub_env.items():
        monkeypatch.setenv(k, str(v))
    inp = _inputs(box["tmp"])[0]
    out = os.path.join(box["tmp"], f"o-p{n_gpu}-{len(stub_env)}")
    rc = cli.main(["pred", "--json_path", inp, "--output_dir", out, "--mode", "big", "--n_gpu", str(n_gpu)])
    err = capsys.readouterr().err
    fwd = cli.last_run()["reports"]["forward"]                                # the forward step's own report, kept on the run record (the work dir goes after a clean pred)
    return rc, err, fwd


def test_requested_p_reaches_the_model_process(box, monkeypatch, capsys):
    rc, err, fwd = _pred(box, monkeypatch, capsys, 2)
    assert fwd["n_gpu_received"] == 2 and fwd["n_gpu"] == 2 and fwd["world"] == 2, fwd          # `--n-gpu 2` on the forward command line
    cmd = [l for l in err.splitlines() if l.startswith("[af3-torch-opt] COMMAND step=forward")][0]
    assert " --n-gpu 2 " in cmd and "--rank-timeout" not in cmd and "--rowpair-block" not in cmd, cmd   # the group timeout and the row block are the model process's constants
    from af3_torch_opt import forward as fwdmod
    assert fwdmod.RANK_TIMEOUT_S == 1800.0 and fwdmod.ROWPAIR_BLOCK is None
    src = _read("opt", "af3_torch_opt", "forward.py")
    assert "nccl_timeout_s=RANK_TIMEOUT_S" in src and '"B": ROWPAIR_BLOCK' in src and "--rank-timeout" not in src and "--rowpair-block" not in src
    active = [l for l in err.splitlines() if l.startswith("[af3-torch-opt] ACTIVE ")][0]
    assert active.startswith("[af3-torch-opt] ACTIVE mode=big lever_set=big levers=bf16w+trimul+triattn+transition+apb+resid_fold+attn_epi+tmpl_trimul+pwa_lnl+opm+pwa_msa+hoist+compile+sbatch+atom_window+token_agg+atom_rows+prologue+glu_proj+trimul_exact dtk=1 n_gpu=2 sharding=rowpair params="), active
    assert rc == 0 and "ACTIVE mode=big" in err and " n_gpu=2 sharding=rowpair " in err and "n_gpu_mismatch" not in err, err[-3000:]
    done = [l for l in err.splitlines() if l.startswith("[af3-torch-opt] DONE ")][0]
    assert " n_gpu=2 sharding=rowpair " in done, done
    (templates,) = [l for l in err.splitlines() if l.startswith("[af3-torch-opt] TEMPLATES ")]
    assert templates.endswith(" form=row_born"), templates                                          # --n_gpu 2: each rank computes its rows of the template pair inputs (the census word of the row-born template path)


def test_a_model_process_that_ran_another_p_is_not_active(box, monkeypatch, capsys):
    rc, err, fwd = _pred(box, monkeypatch, capsys, 2, STUB_NGPU_REPORT=1, STUB_WORLD_REPORT=1)   # the axis dropped inside: folded on one GPU
    assert fwd["n_gpu_received"] == 2 and fwd["n_gpu"] == 1
    assert rc == 3, (rc, err[-2000:])
    line = [l for l in err.splitlines() if l.startswith("[af3-torch-opt] NOT ACTIVE")]
    assert line and "reason=n_gpu_mismatch requested=2 active=1 world=1" in line[0], err[-3000:]
    done = [l for l in err.splitlines() if l.startswith("[af3-torch-opt] DONE ")][0]
    assert " ok=0 rc=3 " in done, done


def test_a_short_rank_census_is_not_active(box, monkeypatch, capsys):
    rc, err, fwd = _pred(box, monkeypatch, capsys, 4, STUB_WORLD_REPORT=3)                          # rank 0 says 4, the launcher ran 3
    assert rc == 3 and "reason=n_gpu_mismatch requested=4 active=4 world=3" in err, err[-3000:]


def test_p1_passes_the_assertion(box, monkeypatch, capsys):
    rc, err, fwd = _pred(box, monkeypatch, capsys, 1)
    assert rc == 0 and fwd["n_gpu"] == 1 and " n_gpu=1 sharding=none " in err and "n_gpu_mismatch" not in err
    t = [l for l in err.splitlines() if l.startswith("[af3-torch-opt] TEMPLATES ")]
    assert t and " items=1 declared=0 live=0 refused=0" in t[0] and t[0].endswith(" real=0/4 form=dense"), t   # the template census line (one GPU: the stock embedder's dense statement); DONE carries templates=live/declared
    assert " templates=0/0 " in [l for l in err.splitlines() if l.startswith("[af3-torch-opt] DONE ")][0]


# ---- no silent switch to a replicated path under `--mode big --n_gpu P`: the GATE TABLE forward.ROWPAIR_GATES names every
#      lever's state under P > 1, and rowpair_xfold.py implements exactly those states ----------------------------------------

def _gates():
    src = _read("opt", "af3_torch_opt", "forward.py")
    body = src[src.index("ROWPAIR_GATES = ("):src.index(")\n\n\ndef main_sharded")]
    return dict(re.findall(r'\("(\w+)", "([^"]+)"\)', body))


def test_gate_table_covers_every_kit_lever():
    gates = _gates()
    for lever in list(registry.LEVERS) + list(big.LEVER_ORDER):
        if lever in ("graph_drop",):                     # graph_drop IS the P = 1 big lever that drops the graphs; under P > 1 graph_pairformer/stepgraph rows say off
            continue
        assert lever in gates, f"lever {lever} has no row in forward.ROWPAIR_GATES"
    for name in ("trimul", "triattn", "transition", "apb", "hoist", "stepgraph", "dtk", "prev_free", "diff_free", "bf16w", "compile"):
        assert name in gates, name
    assert all(v.split(":")[0] in ("replaced", "composes", "off", "skipped", "subsumed", "n/a", "bcast") for v in gates.values()), gates


def test_gates_are_forced_in_the_model_process_source():
    fwd = _read("opt", "af3_torch_opt", "forward.py")
    fwd_impl = _read("opt", "af3_torch_opt", "forward_impl.py")   # _forward_samples (the prev_free_mode statement) lives here now
    rpx = _read("opt", "af3_torch_opt", "rowpair_xfold.py")
    gates = _gates()
    assert gates["dtk"].startswith("skipped") and 'rep["dtk_state"] = "skipped:n_gpu>1' in fwd and "if a.dtk and tp is not None:" in fwd
    assert gates["hoist"].startswith("replaced") and "dh.use_hoist = False" in rpx and "pair_cond_rows(" in rpx and "PairBiasCache(" in rpx
    assert gates["stepgraph"].startswith("off") and "dh.use_step_graph = False" in rpx
    body = rpx.split("def trimul_fns")[1].split("\ndef ")[0]                       # the trimul binding: the module's own Linear layers + the core contraction; the core's fused rows
    assert gates["trimul"] == "replaced:rowpair_rows" and "TriMulFns(" in body and "lnl_fused" not in body       # the core's fused TriMul rows over the eager callables whenever the trimul lever is on
    assert "RF.fused_trimul_fns(trimul_weights(mod, outgoing), stock, eps=TRIMUL_EPS)" in body and 'if not _kernel_on("trimul"):' in body   # (tests/test_tp_trimul_rows.py); lever off = the callables alone
    assert gates["triattn"].startswith("composes") and "flash_triangle_attention(" in rpx and "flash_supported(" in rpx
    assert gates["apb"].startswith("composes") and "scaled_dot_product_attention(" in rpx and "apb_local_queries(" in rpx
    assert gates["prev_free"].startswith("subsumed") and 'item["prev_free_mode"] = "rowpair_carry"' in fwd_impl
    # no size threshold switches a statement to a replicated pair: the adapter has no such constant
    assert not re.search(r"REPLICATE_BELOW|GATHER_BELOW|TP_ABOVE|if\s+N\s*[<>]=?\s*\d{3,}", rpx)


def test_a_reduced_protocol_is_named_on_the_settings_line():
    """A memory-reach run of the tensor-parallel line is spelt with the stock knobs (`--num_recycles 1 --num_diffusion_samples 1
    --diffusion_steps 2`): any knob off its default prints the SETTINGS line with the resolved values and records them in the run record's
    activation report — a reduced protocol is a named event, never silent; there is no named preset."""
    cli_src = _read("opt", "af3_torch_opt", "cli.py")
    assert '"--settings"' not in cli_src and "SETTINGS_PRESETS" not in _read("opt", "af3_torch_opt", "modes.py")
    assert 'line("SETTINGS", num_recycles=a.num_recycles, num_diffusion_samples=a.num_diffusion_samples' in cli_src and 'rep["settings"]' in cli_src and "def protocol_named(" in cli_src


def test_stage_boundaries_are_a_census_entry_not_a_comment():
    """boundary_sync is RECORDED (stats['boundary_sync'] / stats['boundaries']) by rowpair_xfold.stage_boundary at both stage boundaries of the
    n_gpu > 1 driver and printed on the STAGES line."""
    fwd_impl = _read("opt", "af3_torch_opt", "forward_impl.py")   # _forward_samples (the stage-boundary calls + the STAGES print) lives here now
    rpx = _read("opt", "af3_torch_opt", "rowpair_xfold.py")
    assert 'rpx.stage_boundary("trunk_done")' in fwd_impl and 'rpx.stage_boundary("diffusion_done")' in fwd_impl
    assert 'st["boundary_sync"] = BOUNDARY_SYNC' in rpx and 'BOUNDARY_SYNC = "barrier"' in rpx
    assert "boundary_sync=%s boundaries=%s" in fwd_impl                                   # the STAGES line carries the word


# ---- the core's fused TriMul rows under n_gpu > 1: rank 0's F2.trimul_rows line is relayed verbatim and its record kept on the run record (cli.last_run()); the
#      outcome rule is the core's (every decline reason on that line is a documented gate, exit 0; where the fused rows cannot run the core raises
#      RowpairRefused inside the fold, naming `--mode off` / ROWPAIR_TRIMUL_KERNELS=torch: the item FAILS by name, rc 1) ----

def test_the_f2_line_is_relayed_and_its_record_kept(box, monkeypatch, capsys):
    rc, err, fwd = _pred(box, monkeypatch, capsys, 2, STUB_ROWPAIR_F2="below_gate@2016;c=64/64@968;served@26048")
    assert rc == 0, err[-3000:]
    assert f"{report.PREFIX} LEVER name=F2.trimul_rows state=on served=26048 fallback=2984" in err               # relayed verbatim, once, under the kit's tag (the wrapper hands the model process ROWPAIR_TAG)
    assert "lever=trimul_rows" not in err and "partial=none" in [l for l in err.splitlines() if l.startswith("[af3-torch-opt] DONE ")][0]
    M = cli.last_run()                                                                                             # the run record in memory (no manifest file)
    assert M["activation"]["trimul_rows"]["fallback_by"] == {"below_gate": 2016, "c=64/64": 968} and M["activation"]["trimul_rows"]["served"] == 26048


def test_a_refusal_raised_inside_the_fold_fails_the_item_by_name(box, monkeypatch, capsys):
    words = "RowpairRefused('F2.trimul_rows (fpf_trimul_v4): cannot run in this process — no cells row and no safe settings for cc 8.6; run the kit with `--mode off`, or opt this lever out with ROWPAIR_TRIMUL_KERNELS=torch')"
    rc, err, fwd = _pred(box, monkeypatch, capsys, 2, STUB_FAIL="forward:a", STUB_FAIL_ERROR=words)
    assert rc == 1, err[-3000:]
    assert f"[af3-torch-opt] FAILED item=a seed=1 error={words}" in err
    done = [l for l in err.splitlines() if l.startswith("[af3-torch-opt] DONE ")][0]
    assert " ok=0 rc=1 " in done and "failed=a" in done, done


def test_positions_sync_is_bcast_and_recorded_in_the_cores_words():
    """The diffusion roll-out's replicated-positions policy is the one of every xP row (rank 0 broadcast per denoiser call): no environment word
    names another; the schedule census records det / noise_sync=bcast / diff_noise=bcast_rank0_state once per process and the lever stats say bcast."""
    import pytest
    pytest.importorskip("torch")                                              # rowpair_xfold imports torch (the model process's module)
    from af3_torch_opt import rowpair_xfold as rpx
    src = open(rpx.__file__, encoding="utf-8").read()
    assert rpx.POSITIONS_SYNC == "bcast" and "ROWPAIR_XFOLD_POSITIONS_SYNC" not in src and '"guard"' not in src
    seen = []
    class _EV:
        @staticmethod
        def record_schedule(**kw): seen.append(kw)
    saved = {k: rpx.STATE.get(k) for k in ("C", "sync_recorded", "stats")}
    try:
        rpx.STATE["C"] = {"EV": _EV}; rpx.STATE["sync_recorded"] = False; rpx.STATE["stats"] = {**(saved["stats"] or {}), "levers": {}}
        assert rpx._positions_sync_mode() == "bcast" and rpx._positions_sync_mode() == "bcast"
        assert seen == [{"det": rpx.det_level(), "noise_sync": "bcast", "diff_noise": "bcast_rank0_state"}], seen      # recorded once
        assert rpx.STATE["stats"]["levers"].get("positions_sync") == "bcast"
    finally:
        for k, v in saved.items():
            if v is not None: rpx.STATE[k] = v
