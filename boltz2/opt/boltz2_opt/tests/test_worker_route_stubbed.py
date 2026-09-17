"""pred --mode exact|fast on a stand-in worker: staging contract, batch json, child environment, evidence, lines, tally, run record."""
import json
import os

import pytest

from .. import manifest as mf
from .. import modes, report as rep, stack, worker
from . import _stubs


@pytest.fixture(autouse=True)
def _tally():
    rep.reset_tally(); yield; rep.reset_tally()


def test_child_env_strips_kit_switches_and_package_switch_then_sets_the_row(monkeypatch):
    monkeypatch.setenv("BOLTZ2_OPT", "exact"); monkeypatch.setenv("BOLTZ_TRIATTN", "flash"); monkeypatch.setenv("BOLTZ_LEVERS", "all")
    monkeypatch.setenv("ROWPAIR_MSA_HOST", "off"); monkeypatch.setenv("CUEQ_DEFAULT_CONFIG", "0"); monkeypatch.setenv("BOLTZ_CACHE", "/c")
    env = stack.child_env("exact")
    assert "BOLTZ2_OPT" not in env and "BOLTZ_TRIATTN" not in env and "ROWPAIR_MSA_HOST" not in env and "CUEQ_DEFAULT_CONFIG" not in env
    assert env["BOLTZ_LEVERS"] == "resid,mask2" and env["BOLTZ_SAMPLER_ROLLOUT"] == "graph" and "BOLTZ_GRAPH_DIFFUSION" not in env and env["BOLTZ_DIT_HOIST"] == "2" and env["BOLTZ_PAIRBLOCK"] == "cueq"
    assert env["BOLTZ_CACHE"] == "/c"
    assert "BOLTZ_TRIATTN" not in stack.child_env("big") and "BOLTZ_TRIATTN" not in stack.child_env("fast") and stack.child_env("big")["BOLTZ_PAIRFUSE"] == modes.MODES["big"]["env"]["BOLTZ_PAIRFUSE"] and stack.child_env("fast")["BOLTZ_PAIRFUSE"] == modes.MODES["fast"]["env"]["BOLTZ_PAIRFUSE"] and {stack.child_env(m)["BOLTZ_PAIRFUSE"].split(",")[0] for m in ("fast", "big")} == {"bf16"}
    assert stack.child_env("fast")["BOLTZ_PAIRBLOCK"] == modes.env_row("fast")["BOLTZ_PAIRBLOCK"] != "cueq" and stack.child_env("exact")["BOLTZ_PAIRBLOCK"] == "cueq" and stack.child_env("exact")["BOLTZ_FPF_TRIMUL"] == "exact"


def test_write_batch_is_the_kit_workers_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(stack, "cache_dir", lambda: "/cache")
    ys = _stubs.write_yamls(str(tmp_path))
    items = worker.items_from_yamls(ys)
    p = stack.write_batch(str(tmp_path), "t", items, [0, 1], str(tmp_path / "out"))
    b = json.load(open(p))
    assert set(b) == {"tag", "items", "seeds", "cache", "checkpoint", "out_dir", "kit_dir", "write_full_pae", "write_full_pde", "output_format", "step_scale", "predict", "options"} and b["options"] == {"affinity": {}}   # options: only what the caller gave
    assert b["predict"] == {"recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1, "max_parallel_samples": 5} and (b["write_full_pae"], b["write_full_pde"], b["output_format"], b["step_scale"]) == (False, False, "mmcif", 1.5), "upstream's defaults"
    assert b["items"][0] == {"name": "a", "uid": "a", "yaml": ys[0], "seeds": [0, 1]} and b["checkpoint"] == "/cache/boltz2_conf.ckpt"
    assert b["kit_dir"] == os.path.abspath(str(tmp_path)) and b["out_dir"] == os.path.abspath(str(tmp_path / "out")), "the launch's kit directory (the worker's records) beside the predictions' directory"
    with pytest.raises(ValueError):
        worker.items_from_yamls([ys[0], ys[0]])


def test_worker_command_is_the_self_test_arm_with_the_rows_kernel_state():
    exact = stack.worker_command("/b.json", "exact"); dd = exact.index("--")
    assert exact[1:3] == ["-m", "boltz2_opt.worker_launch"], "every worker mode runs under the launcher with the template guard attached"
    assert exact[3:7] == ["--route", "fpf_trimul", "--attach", ",".join(modes.attachments("exact"))] and exact[6].startswith("templ,trimul,transition,pairblock,triattn_exact,sampler,"), "exact: the exact TriMul routed, the core pair-track adapters and the engine adapters attached in row order"
    assert exact[7:dd] == stack.kernels_opts("exact") == ["--kernels-route", "exact", "--kernels-expect", "cueq_triatt=engaged,cueq_trimul=engaged", "--kernels-settings", "defaults", "--kernels-ngpu", "1", "--kernels-mode", "exact"], "exact: the KERNELS census expects upstream's two cuEquivariance accelerators engaged (modes.kernels_expected)"
    assert exact[dd + 1:] == ["bz_worker.py", "--batch", "/b.json", "--mode", "fast", "--kernels", "on", "--num_workers", "1", "--pipeline", "1", "--keep_on_gpu", "1"], "exact: the row's --kernels on (D65)"
    fast = stack.worker_command("/b.json", "fast")
    assert modes.routed_kernels("fast") == [] and fast[1:5] == ["-m", "boltz2_opt.worker_launch", "--attach", ",".join(modes.attachments("fast"))], "fast: no kernel route of this tree's (the TriMul kernels are the core provider's by their core names), the adapters plus the fused MSA-module kernels' attach hook"
    assert fast[5:fast.index("--")] == stack.kernels_opts("fast") and stack.kernels_opts("fast")[1] == "fast"
    big = stack.worker_command("/b.json", "big")
    assert big[3:7] == ["--route", ",".join(modes.routed_kernels("big")), "--attach", "templ,trimul,transition,pairblock,pairfuse,sampler,exactln,waste,msa,msa2,templskip,atom,precision,xl,writer,prefetch"] and big[7:big.index("--")] == stack.kernels_opts("big")
    b2 = stack.worker_command("/b.json", "big", 2)
    assert b2[b2.index("--kernels-route") + 1] == "big_x2" and b2[b2.index("--kernels-expect") + 1] == "cueq_triatt=engaged,cueq_trimul=off-by-route:replaced_by_rowpair" and b2[b2.index("--kernels-ngpu") + 1] == "2", "n_gpu > 1: the route word gains _x<P>, the row-sharded pair stack replaces the whole-tensor TriMul kernels by name"


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_pred_exact_runs_the_worker_and_reports(tmp_path, monkeypatch, capsys):
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch)
    monkeypatch.setenv("BOLTZ2_OPT", "exact")
    ys = _stubs.write_yamls(str(tmp_path)); out = str(tmp_path / "out")
    rc = worker.run("exact", ys, out, [0, 1], tag="t")
    text = capsys.readouterr().out
    assert rc == 0, text
    assert "[boltz2-opt] ACTIVE mode=exact route=worker gpu=NVIDIA H100 80GB HBM3 n_gpu=1 sharding=none levers=resid,mask2,graph_sampler,dit_hoist,fpf_trimul_exact,fused_transition,pairblock" in text
    assert "[boltz2-opt] APPLIED levers=resid,mask2 graph=graph hoist=2 f2=- items=4 n_replay=199 captured_with_cache=1" in text, text
    assert os.path.isfile(os.path.join(out, "by_seed", "a", "s1", "a_model_0.cif"))
    m = mf.LAST
    assert m["mode"] == "exact" and m["route"] == "pred" and m["report"]["active"] is True and m["evidence"]["n_items"] == 4
    assert m["env_row"] == modes.env_row("exact") and len(m["outputs"]) == 4 and m["rc"] == 0
    assert m["tally"] == {"mode": "exact", "route": "worker", "predictions": 4, "ok": 4, "failed": 0, "rc": 0}
    kit = os.path.join(out, "_kit")                                               # the launch directory keeps the staged worker and the batch json; the worker's records left with the verb
    assert os.path.isfile(os.path.join(kit, "batch_t.json")) and not any(os.path.exists(p) for p in worker.launch_records(kit, "t")), sorted(os.listdir(kit))
    assert sorted(f for f in os.listdir(out) if not f.startswith("_kit")) == ["by_seed", "t_worker.log"], "nothing but the predictions and the worker's transcript beside the launch directory"
    rep._print_tally()
    assert "[boltz2-opt] EXIT mode=exact route=worker n_gpu=1 sharding=none predictions=4 ok=4 failed=0 rc=0" in capsys.readouterr().out


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_pred_exact_is_the_kernels_on_row(tmp_path, monkeypatch, capsys):
    """exact (D65): the row resid,mask2 + graph + hoist with the worker's --kernels on; the ACTIVE line names the mode and kernels=on."""
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch)
    monkeypatch.setenv("BOLTZ2_OPT", "exact")
    ys = _stubs.write_yamls(str(tmp_path)); out = str(tmp_path / "out")
    rc = worker.run("exact", ys, out, [0, 1], tag="t")
    text = capsys.readouterr().out
    assert rc == 0, text
    active = [l for l in text.splitlines() if l.startswith("[boltz2-opt] ACTIVE mode=exact ")]
    assert len(active) == 1 and "route=worker gpu=NVIDIA H100 80GB HBM3" in active[0] and "resid,mask2" in active[0] and active[0].endswith("kernels=on"), active
    assert "[boltz2-opt] APPLIED levers=resid,mask2 graph=graph hoist=2 f2=- items=4 n_replay=199 captured_with_cache=1" in text
    m = mf.LAST
    assert m["mode"] == "exact" and m["env_row"] == modes.env_row("exact") == {"BOLTZ_LEVERS": "resid,mask2", "BOLTZ_GRAPH_DIFFUSION": "graph", "BOLTZ_DIT_HOIST": "2", "BOLTZ_FPF_TRIMUL": "exact", "BOLTZ_FPF_TRIMUL_PROVIDER": "exact", "BOLTZ_TRANSITION": "exact", "BOLTZ_PAIRBLOCK": "cueq"} and m["rc"] == 0
    assert m["tally"] == {"mode": "exact", "route": "worker", "predictions": 4, "ok": 4, "failed": 0, "rc": 0}


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_pred_fast_requires_the_block_adapters_evidence(tmp_path, monkeypatch, capsys):
    """fast: the core pair-track adapters' own reports (pairblock_report / transition_report / trimul_report, written by the attach hook) are
    the evidence; a missing report is 'lever not in force' (exit 3, no APPLIED line)."""
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch)
    ys = _stubs.write_yamls(str(tmp_path), ("a",))
    assert worker.run("fast", ys, str(tmp_path / "o1"), [0]) == 0
    text = capsys.readouterr().out
    assert "[boltz2-opt] APPLIED" in text and "f2=-" in text, text
    m = mf.LAST
    assert m["evidence"]["pairblock_report"]["variant"] == "k2b" and m["evidence"]["transition_report"]["variant"] == "fast" and "fpf_trimul" in m["evidence"]["trimul_report"]["applied"]
    _stubs.install_fake_stage(monkeypatch, pair_reports=False)                                 # the adapters' reports missing = no evidence
    rc = worker.run("fast", ys, str(tmp_path / "o2"), [0])
    text = capsys.readouterr().out
    assert rc == 3 and "no pairblock_report" in text and "no transition_report" in text and "APPLIED" not in text, text
    rc = worker.run("exact", ys, str(tmp_path / "o3"), [0])
    assert rc == 3 and "no pairblock_report" in capsys.readouterr().out, "exact carries the same adapters (cueq / exact variants)"

def test_child_env_carries_the_evidence_switch_for_fast_only(monkeypatch):
    monkeypatch.delenv("BOLTZ_TRIATTN_REPORT", raising=False)
    assert stack.child_env("fast")["BOLTZ_TRIATTN_REPORT"] == "1"
    assert "BOLTZ_TRIATTN_REPORT" not in stack.child_env("exact")
    monkeypatch.setenv("BOLTZ_TRIATTN_REPORT", "1")                                            # a caller's copy is stripped with the switch family, then set by the mode
    assert "BOLTZ_TRIATTN_REPORT" not in stack.child_env("exact") and stack.child_env("fast")["BOLTZ_TRIATTN_REPORT"] == "1"


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_missing_evidence_is_a_refusal_not_a_silent_stock_run(tmp_path, monkeypatch, capsys):
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch, evidence_ok=False)
    ys = _stubs.write_yamls(str(tmp_path), ("a",))
    rc = worker.run("exact", ys, str(tmp_path / "out"), [0])
    text = capsys.readouterr().out
    assert rc == 3 and "[boltz2-opt] NOT ACTIVE: " in text and "hoist not applied" in text and "APPLIED" not in text
    m = mf.LAST
    assert m["report"]["active"] is False and "hoist not applied" in m["report"]["reason"]


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_failed_prediction_counts_in_the_tally(tmp_path, monkeypatch, capsys):
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch, fail_items=1)
    ys = _stubs.write_yamls(str(tmp_path)); rc = worker.run("exact", ys, str(tmp_path / "out"), [0])
    assert rc == 1
    assert rep.tally_snapshot() == {"mode": "exact", "route": "worker", "predictions": 2, "ok": 1, "failed": 1, "rc": 1}


def test_gate_refusal_before_anything_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stack, "gpu_probe", lambda: None)
    monkeypatch.setattr(stack, "pins_check", lambda ckpt=False: ([], {"version": "2.2.1", "pinned": True}))
    monkeypatch.setenv("BOLTZ_CACHE", str(tmp_path / "nocache"))
    ys = _stubs.write_yamls(str(tmp_path), ("a",))
    rc = worker.run("exact", ys, str(tmp_path / "out"), [0])
    text = capsys.readouterr().out
    assert rc == 3 and text.startswith("[boltz2-opt] NOT ACTIVE:") and "ACTIVE mode=" not in text
    assert not os.path.isdir(tmp_path / "out" / "_kit")
    assert mf.LAST["rc"] == 3


def test_off_has_no_worker_route(capsys):
    assert worker.run("off", [], "/tmp/none", [0]) == 2


# --- the exit rule on partial activation (stack.partial_activation): a kit fallback of a row lever exits EXIT_NOT_ACTIVE unless --allow-partial
# is passed and recorded; a documented gate is recorded and exits by the outputs; outputs short are `incomplete`, EXIT_FAILED
def _lr(**kw):
    base = {"applied": True, "levers": ["resid", "mask2"], "stats": {"mask_scope_trivial": 52}, "torch": "2.12.0"}
    base.update(kw); return base


MASK2_FALLBACK = "mask2: 3 calls could not classify the pair mask (mask_scope_error) and ran the stock mask arithmetic"   # stack.partial_activation's string
REFUSED_LINE = f"[boltz2-opt] NOT ACTIVE: partial activation — {MASK2_FALLBACK}; exit 3 (--allow-partial records and proceeds)"   # the family's line, literal
ALLOWED_LINE = f"[boltz2-opt] PARTIAL allowed: {MASK2_FALLBACK} (--allow-partial, recorded)"


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_pred_kit_fallback_exits_not_active_unless_allow_partial(tmp_path, monkeypatch, capsys):
    """A trunk lever of the row that ran without its optimization (mask2: calls whose mask the scope could not classify, stats
    mask_scope_error) is a partial activation: the verb exits EXIT_NOT_ACTIVE unless --allow-partial is passed, which records it."""
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch, lever_report=_lr(stats={"mask_scope_error": 3, "mask_scope_trivial": 49}))
    ys = _stubs.write_yamls(str(tmp_path))
    rc = worker.run("exact", ys, str(tmp_path / "o1"), [0], tag="t")
    text = capsys.readouterr().out
    assert rc == rep.EXIT_NOT_ACTIVE and REFUSED_LINE in text.splitlines() and "APPLIED" not in text and text.count("NOT ACTIVE") == 1
    m = mf.LAST
    assert m["report"]["partial"] is True and m["report"]["levers_fallback"] == [MASK2_FALLBACK] and m["report"]["allow_partial"] is False and m["rc"] == 0
    assert m["report"]["active"] is False and m["report"]["reason"] == f"partial activation — {MASK2_FALLBACK}"
    rc = worker.run("exact", ys, str(tmp_path / "o2"), [0], tag="t", allow_partial=True)
    text = capsys.readouterr().out
    assert rc == 0 and ALLOWED_LINE in text.splitlines() and "[boltz2-opt] APPLIED " in text and "NOT ACTIVE" not in text
    m = mf.LAST
    assert m["report"]["partial"] is True and m["report"]["allow_partial"] is True and {k: m["report"]["outputs"][k] for k in ("expected", "ok", "failed", "status")} == {"expected": 2, "ok": 2, "failed": 0, "status": "complete"}
    assert stack.partial_activation("exact", {"mask_scope_errors": 0}) == ([], {}), "zero is quiet"


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_pred_outputs_short_is_incomplete_and_failed(tmp_path, monkeypatch, capsys):
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch, fail_items=1)
    ys = _stubs.write_yamls(str(tmp_path)); out = str(tmp_path / "o")
    assert worker.run("exact", ys, out, [0], tag="t") == rep.EXIT_FAILED
    m = mf.LAST
    assert {k: m["report"]["outputs"][k] for k in ("expected", "ok", "failed", "status")} == {"expected": 2, "ok": 1, "failed": 1, "status": "incomplete"} and m["report"]["active"] is True


def test_warm_is_one_pred_of_the_carried_public_yaml(tmp_path, monkeypatch):
    """`warm` = pred (worker.run) on the tree's inputs/1BRS_x2_barnase_barstar.yaml (398 tokens: above every routed kernel's token floor, so
    each compiles) at one seed, tag warm, run-record flag warm: true; the README's example input is its one-complex sibling (199 tokens)."""
    from .. import warm
    y = warm.warm_input()
    assert y == os.path.join(stack.tree_dir(), "inputs", "1BRS_x2_barnase_barstar.yaml") and os.path.isfile(y)
    text = open(y).read()
    assert text.count("- protein:") == 2 and text.count("msa: empty") == 2 and "id: [A, B]" in text and "id: [C, D]" in text, "two barnase + barstar complexes, single-sequence"
    seqs = [l.split("sequence:")[1].strip() for l in text.splitlines() if l.strip().startswith("sequence:")]
    assert [len(q) for q in seqs] == [110, 89] and 2 * sum(len(q) for q in seqs) == 398, "PDB 1BRS barnase (110) + barstar (89), two copies each = 398 tokens"
    floors = [int(v) for m in ("exact", "fast", "big") for k, v in modes.env_row(m).items() if k.endswith("_MIN_TOKENS")]
    y2 = warm.warm_input(warm.WARM_INPUT_2); t2 = open(y2).read()
    n2 = sum(len(l.split("sequence:")[1].strip()) * (l2.count(",") + 1) for l, l2 in zip([x for x in t2.splitlines() if x.strip().startswith("sequence:")], [x for x in t2.splitlines() if x.strip().startswith("id:")]))
    assert n2 == 576, "the second warm input: barnase x2 + barstar x4 = 576 tokens"
    triton_floors = [int(v) for m in ("exact", "fast", "big") for k, v in modes.env_row(m).items() if k.endswith("_MIN_TOKENS") and k != "BOLTZ_PAIRFUSE_TRIATT_MIN_TOKENS"]
    assert triton_floors and max(triton_floors) <= 398, ("warm's first input reaches every routed Triton kernel's token floor (both inputs = both specialization classes)", triton_floors)
    assert floors and max(floors) <= n2, ("warm's inputs together reach every routed kernel's token floor (the CUDA triangle-attention row's 512: the 576-token input; a prebuilt, no JIT class)", floors)
    ex = open(os.path.join(stack.tree_dir(), "inputs", "1BRS_barnase_barstar.yaml")).read()
    assert [len(l.split("sequence:")[1].strip()) for l in ex.splitlines() if l.strip().startswith("sequence:")] == [110, 89], "the README's example: one complex, 199 tokens"
    calls = []
    monkeypatch.setattr(worker, "run", lambda mode, yamls, out_dir, seeds, **kw: calls.append((mode, yamls, out_dir, seeds, kw)) or 0)
    assert warm.run("fast", str(tmp_path / "w"), seed=3, allow_partial=True, n_gpu=1) == 0
    y2 = warm.warm_input(warm.WARM_INPUT_2)
    y3 = warm.warm_input(warm.WARM_INPUT_3); y4 = warm.warm_input(warm.WARM_INPUT_4)
    assert y2 == os.path.join(stack.tree_dir(), "inputs", "1BRS_a2b4_barnase_barstar.yaml") and os.path.isfile(y2) and warm.warm_inputs() == [y, y3, y2, y4]
    def _tokens(p):                                                                     # sum over entities: len(sequence) x number of chain ids
        t = open(p).read(); s = [x for x in t.splitlines() if x.strip().startswith("sequence:")]; i = [x for x in t.splitlines() if x.strip().startswith("id:")]
        return sum(len(a.split("sequence:")[1].strip()) * (b.count(",") + 1) for a, b in zip(s, i)), t
    n3, t3 = _tokens(y3); n4, t4 = _tokens(y4)
    assert n3 == 480 == 30 * 16 and 300 <= n3 < 512 and t3.count("msa: empty") == 2, "barnase x4 + trp-cage x2: the N == 0 (mod 16) class INSIDE the K2B window (no class the warm ladder meets compiles after warm)"
    assert n4 == 1056 == 66 * 16 and n4 >= 1024 and t4.count("msa: empty") == 3, "barnase x6 + barstar x4 + trp-cage x2: the >= 1,024-token class"
    t2 = open(y2).read(); seqs2 = [l.split("sequence:")[1].strip() for l in t2.splitlines() if l.strip().startswith("sequence:")]
    assert "id: [A, B]" in t2 and "id: [C, D, E, F]" in t2 and t2.count("msa: empty") == 2 and [len(q) for q in seqs2] == [110, 89] and 2 * 110 + 4 * 89 == 576 == 36 * 16, "barnase x2 + barstar x4 = 576 tokens: the N == 0 (mod 16) class (398 = 14 mod 16 is the other)"
    assert calls == [("fast", [y, y3, y2, y4], str(tmp_path / "w"), [3], {"tag": "warm", "extra_record": {"warm": True}, "allow_partial": True, "n_gpu": 1})]


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_warm_runs_the_worker_route_end_to_end_stubbed(tmp_path, monkeypatch, capsys):
    from .. import warm
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch)
    out = str(tmp_path / "w")
    rc = warm.run("exact", out)
    text = capsys.readouterr().out
    assert rc == 0, text
    rec = os.path.splitext(os.path.basename(warm.WARM_INPUT))[0]                      # the record is the warm input's stem (inputs/1BRS_x2_barnase_barstar.yaml)
    assert os.path.isfile(os.path.join(out, "by_seed", rec, "s0", f"{rec}_model_0.cif"))
    m = mf.LAST
    assert m["mode"] == "exact" and m["warm"] is True and m["tally"]["predictions"] == 4 and m["rc"] == 0


def test_a_name_that_is_not_a_mode_is_refused_by_name(tmp_path, monkeypatch):
    """A name outside modes.MODE_NAMES never runs and is never aliased: the worker route raises before anything is staged."""
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch)
    ys = _stubs.write_yamls(str(tmp_path)); out = str(tmp_path / "out")
    for name in ("exact_k", "exact_nk", "exact_fpf", "big_exact", "turbo"):
        with pytest.raises(ValueError):
            worker.run(name, ys, out, [0, 1], tag="t")
    assert not os.path.exists(os.path.join(out, "_kit")), "nothing ran"
