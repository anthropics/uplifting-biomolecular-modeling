"""The command line's mode rule, check's dry run and exit codes, the report lines, and the evidence reader on synthetic worker logs."""
import json
import os
import subprocess
import sys

import pytest

from .. import cli, modes, report as rep, stack


def test_mode_resolution_rules(monkeypatch):
    monkeypatch.delenv("BOLTZ2_OPT", raising=False)
    assert cli.resolve_mode(None) == "fast" and cli.resolve_mode("Exact") == "exact"          # modes.DEFAULT_MODE: fast, the package default
    monkeypatch.setenv("BOLTZ2_OPT", "fast")
    assert cli.resolve_mode(None) == "fast" and cli.resolve_mode("fast") == "fast"
    with pytest.raises(SystemExit) as e:
        cli.resolve_mode("exact")
    assert e.value.code == 2
    monkeypatch.delenv("BOLTZ2_OPT", raising=False)
    with pytest.raises(ValueError):
        cli.resolve_mode("turbo")
    for word in ("BOLTZ2_OPT_MODE", "BOLTZ2_BIG_XL_FREE", "BOLTZ2_BIG_ALLOW_PARTIAL"):       # a mistyped selection and a memory-line word alike: refused by name before anything runs (exit 2)
        monkeypatch.setenv(word, "1")
        with pytest.raises(SystemExit) as e:
            cli.resolve_mode("big")
        assert e.value.code == 2
        monkeypatch.delenv(word)


def test_check_off_and_unknown_and_disagreement_exit_codes(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}
    r = subprocess.run([sys.executable, "-m", "boltz2_opt", "check", "--mode", "off"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.startswith("[boltz2-opt] DRY-RUN mode=off route=stock_cli")
    r = subprocess.run([sys.executable, "-m", "boltz2_opt", "check", "--mode", "turbo"], capture_output=True, text=True, env=env)
    assert r.returncode == 2
    r = subprocess.run([sys.executable, "-m", "boltz2_opt", "check", "--mode", "exact"], capture_output=True, text=True, env=dict(env, BOLTZ2_OPT="fast"))
    assert r.returncode == 2 and "disagrees" in r.stderr


def test_allow_partial_is_refused_on_the_stock_route(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}
    r = subprocess.run([sys.executable, "-m", "boltz2_opt", "pred", "--mode", "off", "--allow-partial", "--input", "x.yaml", "--out_dir", str(tmp_path)], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "--allow-partial names a kit fallback of a row lever" in r.stdout + r.stderr


def test_check_exact_dry_run_applies_nothing_and_names_the_gate(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}; env.pop("BOLTZ_CACHE", None)
    r = subprocess.run([sys.executable, "-m", "boltz2_opt", "check", "--mode", "exact", "--json"], capture_output=True, text=True, env=env)
    assert "[boltz2-opt] DRY-RUN mode=exact route=worker" in r.stdout and "levers=resid,mask2,rollout,dit_hoist,align_aligncap,dit_par,dit_mask,dit_sba,dit_smx,dit_glue,fpf_trimul_exact,fused_transition,pairblock,triattn_exact,templ_skip,exactln,exactln_resid,graph_trunk,waste_chunkcast,waste_opmmask,waste_opmdiv,waste_ctorskip,msa_pwa_exact,msa_trans2_exact,atom_keys_gather,atom_glue_hoist" in r.stdout, r.stdout + r.stderr
    assert "env=BOLTZ_LEVERS=resid,mask2 BOLTZ_SAMPLER_ROLLOUT=graph BOLTZ_DIT_HOIST=2 BOLTZ_SAMPLER_ALIGN=aligncap BOLTZ_DIT_EXACT=par,mask,sba,smx,glue BOLTZ_FPF_TRIMUL=exact BOLTZ_FPF_TRIMUL_PROVIDER=exact BOLTZ_TRANSITION=exact BOLTZ_PAIRBLOCK=cueq BOLTZ_TRIATTN_EXACT=1 BOLTZ_TEMPL_SKIP=1 BOLTZ_EXACTLN=core BOLTZ_EXACTLN_RESID=on BOLTZ_GRAPH_TRUNK=pf,pfnoseq,templ BOLTZ_GRAPH_TRUNK_MAX_TOKENS=300 BOLTZ_WASTE=chunkcast,opmmask,opmdiv,ctorskip BOLTZ_MSA2=pwa2x,trans2x BOLTZ_ATOM=keys,glue BOLTZ_WRITER=overlap BOLTZ_PREFETCH=persistent" in r.stdout
    assert r.returncode in (0, 3)
    if r.returncode == 3:
        assert "[boltz2-opt] NOT ACTIVE:" in r.stdout
    plan = json.loads(r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1])
    assert plan["mode"] == "exact" and set(plan) >= {"levers", "env", "stage", "target_gpu", "gpu_class"}
    for name in ("exact_k", "turbo"):                                        # a name that is not a mode is refused by name (exit 2), never run
        r = subprocess.run([sys.executable, "-m", "boltz2_opt", "check", "--mode", name], capture_output=True, text=True, env=env)
        assert r.returncode == rep.EXIT_USAGE and "DRY-RUN" not in r.stdout, (name, r.returncode, r.stdout, r.stderr)


def test_report_lines_and_tally(capsys):
    rep.reset_tally()
    rep.arm_tally("exact", "worker"); rep.count(ok=2); rep.count(failed=1); rep.set_rc(1)
    assert rep.tally_line() == "[boltz2-opt] EXIT mode=exact route=worker n_gpu=1 sharding=none predictions=3 ok=2 failed=1 rc=1 levers_headroom_gated=-"
    rep.tally_headroom(2)                                        # the sampler levers' memory-headroom gate acted on two predictions (stack.evidence headroom_gated)
    assert rep.tally_line().endswith(" rc=1 levers_headroom_gated=rollout,dit_hoist:2") and rep.tally_snapshot()["headroom_gated"] == 2
    rep._print_tally(); rep._print_tally()
    assert capsys.readouterr().out.count("EXIT mode=exact") == 1, "the tally prints once"
    assert rep.not_active_line("x") == "[boltz2-opt] NOT ACTIVE: x"
    assert rep.active_line({"mode": "fast", "route": "worker", "gpu": "H100", "levers_applied": ["a"], "levers_fallback": [], "worker": "w"}) == "[boltz2-opt] ACTIVE mode=fast route=worker gpu=H100 n_gpu=1 sharding=none levers=a fallbacks=- worker=w " + rep.version_tokens() + " kernels=- compile=off:none_in_kit"
    assert rep.active_line({"mode": "exact", "route": "worker", "gpu": "H100", "levers_applied": ["resid", "mask2"], "levers_fallback": [], "worker": "w", "kernels": "on"}).endswith("mode=exact route=worker gpu=H100 n_gpu=1 sharding=none levers=resid,mask2 fallbacks=- worker=w " + rep.version_tokens() + " kernels=on compile=off:none_in_kit")
    assert (rep.EXIT_OK, rep.EXIT_FAILED, rep.EXIT_USAGE, rep.EXIT_NOT_ACTIVE) == (0, 1, 2, 3)
    rep.reset_tally()


TRIMUL_OK = {"applied": ["fpf_trimul"], "disabled": {}, "line": "[boltz2-opt] LEVER name=F2.trimul state=on", "census": {"served": 96, "fallback": {"below_min_tokens": 8, "c_z": 4}, "errors": {}}, "gate": {"ok": True, "reason": None}}   # boltz2_opt.trimul.report() of a run whose every fallback is a declared one
TRIMUL_EXACT_OK = {**TRIMUL_OK, "applied": ["fpf_trimul_exact"], "variant": "exact"}   # the exact row's TriMul variant (BOLTZ_FPF_TRIMUL=exact)
TRANSITION_OK = {"applied": ["fused_transition"], "disabled": {}, "line": "[boltz2-opt] LEVER name=F5.transition state=on", "census": {"served": {"unchunked": 300}, "fallback": {"autocast_off": 260, "chunked": 16}, "errors": {}}, "gate": {"ok": True, "idle": False, "reason": None}}   # boltz2_opt.transition.report(); variant added per row by _log
PAIRBLOCK_OK = {"applied": ["pairblock"], "disabled": {}, "line": "[boltz2-opt] LEVER name=F1.pairblock state=on", "census": {"served": {"cueq": 552}, "fallback": {}, "errors": {}}, "gate": {"ok": True, "idle": False, "reason": None}}   # boltz2_opt.pairblock.report(); variant added per row by _log


PREFETCH_OK = {"installed": True, "disposition": None, "switch": "persistent", "applied": ["prefetch"], "served": 2, "fallback": 0, "fallback_by": {}, "forks_avoided": 1, "ref_forks": 1,
               "forked_before_cuda": True, "helper_cuda_initialized": False, "pin": "parent", "bit_compare": "equal:78", "refused": None, "gate": {"ok": True, "why": []}}
WRITER_OK = {"installed": True, "disposition": None, "switch": "overlap", "applied": ["writer_overlap"], "backend": "process", "items": 2, "submitted": 2, "served": 2, "inline": 0, "inline_by": {},
             "fallback": 0, "fallback_by": {}, "failed": 0, "errors": [], "moves": 2, "moves_done": 2, "queued_max": 1, "join_s": 0.2, "bytes_written": 4096, "helper_pid": 4321, "forked_before_cuda": True,
             "helper_cuda_initialized": False, "dead": None, "gate": {"ok": True, "why": [], "idle": False}}   # boltz2_opt.writer.report() of a healthy run (joined, every write / move landed)
TEMPL_OK = {"installed": True, "items_checked": 0, "declared": 0, "live": 0, "refused": [], "per_item": []}   # boltz2_opt.templates.report() of a run whose inputs declare no template
MSA_OK = {"applied": ["fpf_opm", "fpf_pwa"], "disabled": {}, "units": {u: {"lever": l, "census": {"served": 48, "fallback": 0, "fallback_by": {}, "errors": {}, "calls": 48}, "gate": {"ok": True, "idle": False, "reason": None}, "cfg": c, "tma": True} for u, l, c in (("opm", "fpf_opm", "f1"), ("pwa", "fpf_pwa", "default"))},
          "lines": [], "gate": {"ok": True, "idle": False, "reason": None}, "patched": ["OuterProductMean.forward", "PairWeightedAveraging.forward"], "variant": "opm,pwa", "expected": []}   # boltz2_opt.msa_kernels.report() of a healthy fast run: both units served every call


def msa_ok_for(row):
    """MSA_OK restricted to the units the row's BOLTZ_FPF_MSA word names (variant = the word verbatim), None when the row carries none."""
    word = (row or {}).get("BOLTZ_FPF_MSA")
    if not word:
        return None
    units = [u for u in word.split(",") if u]
    return {**MSA_OK, "applied": [MSA_OK["units"][u]["lever"] for u in units], "units": {u: MSA_OK["units"][u] for u in units}, "variant": word}



def _log(levers, graph="graph", hoist=2, f2=None, items=2, replay=199, cwc=1, fallbacks=0, trimul="row", templ=TEMPL_OK, transition="row", pairblock="row", mode=None, msa="row", adapters="row"):
    """A worker log of a healthy run; trimul / transition / pairblock = "row" fills the adapter report the MODE's row implies (exact: the exact
    variants; fast / big: the Tier-2 ones), a dict is used as given, None omits the report."""
    row = modes.env_row(mode) if mode in modes.MODES else {}
    if row and not row.get("BOLTZ_GRAPH_DIFFUSION") and graph == "graph":
        graph = "off"                                  # the rows carry the roll-out (BOLTZ_SAMPLER_ROLLOUT) in place of the per-step graph patch: the patch is inert, its per-item words say so
    log = {"env": {"boltz_levers": levers, "graph_diffusion": graph, "dit_hoist": hoist, "f2_patch_active": f2}, "lever_report": {"applied": bool(levers), "levers": levers, "stats": {}},
           "per_item": [{"name": "a", "seed": s, "graph_sampler_mode": graph, "graph_n_replay": replay if graph == "graph" else 0, "hoist_hoist_level": hoist, "hoist_captured_with_cache": cwc if graph == "graph" else 0, "hoist_capture_stock_fallbacks": fallbacks} for s in range(items)]}
    if adapters == "row":                              # the engine adapters the row attaches beyond the pair-track cells (roll-out, DITEXACT, layer_norm replica, trunk graphs, PAIRFUSE, atom, WASTE, MSA2, CONF): healthy reports
        from . import _faster
        log.update(_faster.reports(row, items, replay=replay))
    elif isinstance(adapters, dict):
        log.update(adapters)
    if trimul == "row":
        trimul = TRIMUL_EXACT_OK if row.get("BOLTZ_FPF_TRIMUL") == "exact" else TRIMUL_OK
    if transition == "row":
        transition = {**TRANSITION_OK, "variant": row.get("BOLTZ_TRANSITION", "exact")}
    if pairblock == "row":
        v = row.get("BOLTZ_PAIRBLOCK", "cueq")
        pairblock = {**PAIRBLOCK_OK, "variant": v, **({"applied": ["pairblock", "flash_triattn"], "census": {"served": {v: 500, "cueq_below_gate": 52}, "fallback": {}, "errors": {}}} if v != "cueq" else {})}
    if trimul is not None:
        log["trimul_report"] = trimul               # read only by the rows that carry BOLTZ_FPF_TRIMUL (exact, fast, big)
    if templ is not None:
        log["templ_report"] = templ                 # the template guard's census (every worker mode attaches it)
    if row.get("BOLTZ_WRITER") and "writer_report" not in log:
        log["writer_report"] = dict(WRITER_OK)              # the background writer's census (rows that carry BOLTZ_WRITER: every row)
    if row.get("BOLTZ_PREFETCH") and "prefetch_report" not in log:
        log["prefetch_report"] = dict(PREFETCH_OK)          # the persistent featurizer's census (rows that carry BOLTZ_PREFETCH: exact, fast)
    if transition is not None:
        log["transition_report"] = transition       # read only by the rows that carry BOLTZ_TRANSITION (exact, fast)
    if pairblock is not None:
        log["pairblock_report"] = pairblock         # read only by the rows that carry BOLTZ_PAIRBLOCK (exact, fast)
    if msa == "row":
        msa = msa_ok_for(row)
    if msa is not None:
        log["msa_report"] = msa                     # read only by the rows that carry BOLTZ_FPF_MSA (fast)
    return log


def test_evidence_reader():
    full = ["resid", "mask2"]
    kon = ["resid", "mask2"]                          # the kernels-on rows' trunk levers (exact, fast)
    ev, problems = stack.evidence("exact", _log(kon, mode="exact"))
    assert problems == [] and ev["n_items"] == 2 and ((ev.get("sampler_report") or {}).get("module") or {}).get("stats", {}).get("replays") == 2 * 199, "the roll-out replays S-1 boundaries per prediction"
    ev, problems = stack.evidence("exact", _log(kon[:1], mode="exact"))
    assert any("trunk levers not reported applied by the worker: mask2" in p for p in problems)
    ev, problems = stack.evidence("exact", _log(kon, replay=100, mode="exact"))
    assert any("roll-out: captures=2 replays=200 for 2 item(s)" in p and "398 replays" in p for p in problems), "the roll-out replays S-1 boundaries per prediction"
    ev, problems = stack.evidence("exact", _log(kon, fallbacks=1, mode="exact"))
    assert any("capture_stock_fallbacks!=0 (hoist over the roll-out)" in p for p in problems)
    ev, problems = stack.evidence("exact", _log(kon, mode="exact", adapters=None))     # the row's engine adapters write their reports at exit: each is required evidence, by name
    for key in ("sampler_report", "ditexact_report", "triattn_exact_report", "exactln_report", "templskip_report", "graph_report", "atom_report", "waste_report", "msa2_report"):
        assert any(f"no {key} in the worker log" in p for p in problems), key
    ev, problems = stack.evidence("fast", _log(kon, mode="fast", adapters=None))
    for key in ("sampler_report", "exactln_report", "graph_report", "pairfuse_report", "atom_report", "waste_report", "msa2_report", "conf_report"):
        assert any(f"no {key} in the worker log" in p for p in problems), key
    ev, problems = stack.evidence("big", _log(kon, f2=None, mode="big", adapters=None))   # big on fast's pair track: the driver's / block adapters' reports are required evidence as on fast; the flash patch is not its base
    for key in ("exactln_report", "pairfuse_report", "atom_report", "waste_report", "msa2_report", "xl_report"):
        assert any(f"no {key} in the worker log" in p for p in problems), key
    assert not any("no graph_report in the worker log" in p for p in problems), "big names the trunk graphs off by rule: their report is not its evidence"
    assert not any("flash tri-attention" in p for p in problems), problems
    ev, problems = stack.evidence("fast", _log(kon, mode="fast"))                       # fast: the core block adapters' reports are the evidence (filled per row by _log)
    assert problems == [], problems
    ev, problems = stack.evidence("fast", _log(kon, mode="fast", pairblock=None))
    assert any("no pairblock_report" in p for p in problems)
    ev, problems = stack.evidence("fast", _log(kon, mode="fast", msa=None))                    # fast carries the fused MSA-module kernels: their report is required evidence
    assert any("no msa_report" in p for p in problems)
    bad = {**MSA_OK, "gate": {"ok": False, "idle": False, "reason": "opm: unexpected fallback: fallback_by={dims_not_pinned: 1}"}}
    ev, problems = stack.evidence("fast", _log(kon, mode="fast", msa=bad))
    assert any("fused MSA-module kernels gate refused" in p for p in problems)
    ev, problems = stack.evidence("fast", _log(kon, mode="fast", msa={**MSA_OK, "applied": ["fpf_opm"], "variant": "opm"}))
    assert any("not reported applied: fpf_pwa" in p for p in problems) and any("variant 'opm'" in p for p in problems)
    ev, problems = stack.evidence("exact", None)
    assert any("no worker log" in p for p in problems)
    ev, problems = stack.evidence("exact", _log(kon, mode="exact"), stdout_text="[sitecustomize] boltz_trunk_levers not applied: x\n")
    assert any("the kit reported a lever not applied" in p for p in problems)
    ev, problems = stack.evidence("exact", _log(["resid", "mask2"], mode="exact"))              # the kernels-on row wants resid,mask2 only (D65)
    assert problems == [] and ev["levers"] == ["resid", "mask2"]
    ev, problems = stack.evidence("exact", _log(["resid"], mode="exact"))
    assert any("trunk levers not reported applied by the worker: mask2" in p for p in problems)



def _f2_line(stats, min_tokens=300, errors=()):
    return "[boltz_flash_triattn_patch] " + json.dumps({"applied": True, "mode": "flash", "min_tokens": min_tokens, "stats": stats, "n_flash": stats.get("flash_calls", 0),
                                                        "n_stock": stats.get("stock_calls", 0), "errors": list(errors)}) + "\n"


def test_big_evidence_reads_the_flash_patchs_own_counters(monkeypatch):
    """On the memory row's `flash` pair-stack base (modes.BIG_ATTN_BASE = "flash") the F2 report line (the kit's STATS at exit) is the evidence for the flash
    tri-attention patch: its absence, an exception fallback, or no flash call while calls at or above the size gate went stock are each 'lever not in force';
    size-gated stock calls below the gate are the lever's contract."""
    _flash_base(monkeypatch)
    kon = ["resid", "mask2"]
    ev, problems = stack.evidence("big", _big_log(), stdout_text="[boltz_trunk_levers] APPLIED\n")
    assert any("no flash tri-attention report line" in p for p in problems)
    ok = {"calls": 5, "flash_calls": 4, "stock_calls": 1, "stock|n<300|pairformer.start|N=199|use_kernels=True": 1, "flash|pairformer.start|q=[1, 398, 4, 398, 32]|torch.bfloat16|chunk=None|use_kernels=True": 4}
    ev, problems = stack.evidence("big", _big_log(), stdout_text=_f2_line(ok))
    assert problems == [] and ev["f2_report"]["flash_calls"] == 4 and ev["f2_report"]["stock_calls"] == 1 and ev["f2_report"]["exceptions"] == {} and ev["f2_report"]["stock_above_gate"] == {}
    assert "f2=True(flash_calls=4,stock_calls=1,fallbacks=0)" in rep.applied_line(ev)
    exc = dict(ok, **{"stock|exception:RuntimeError": 2, "stock_calls": 3})
    ev, problems = stack.evidence("big", _big_log(), stdout_text=_f2_line(exc, errors=["flash path raised RuntimeError(...) at pairformer.start N=398"]))
    assert any("fell back to stock at run time" in p and "stock|exception:RuntimeError" in p for p in problems)
    assert "fallbacks=2" in rep.applied_line(ev)
    kfn = dict(ok, **{"kfn_stock|exception:ValueError": 1})
    assert any("fell back to stock" in p for p in stack.evidence("big", _big_log(), stdout_text=_f2_line(kfn))[1])
    never = {"calls": 3, "flash_calls": 0, "stock_calls": 3, "stock|dtype_float32|pairformer.start|N=398|use_kernels=True": 3}
    ev, problems = stack.evidence("big", _big_log(), stdout_text=_f2_line(never))
    assert any("never engaged" in p and "N=398" in p for p in problems)
    gated = {"calls": 3, "flash_calls": 0, "stock_calls": 3, "stock|n<300|pairformer.start|N=199|use_kernels=True": 3}
    ev, problems = stack.evidence("big", _big_log(), stdout_text=_f2_line(gated))
    assert problems == [], "every call below the gate on the stock path is the lever's contract, not a failure"
    assert stack.f2_report("[boltz_flash_triattn_patch] APPLIED flash_triattn version=1\n") is None
    ev, problems = stack.evidence("exact", _log(kon, mode="exact"), stdout_text=_f2_line(ok))
    assert problems == [] and ev["f2_report"] is None, "exact reads no F2 line (the patch is not in its row)"


def test_gpu_class_check_compares_memory_not_only_the_name():
    """MODEL_OPT_TARGET_GPU=H100 is the 80 GB class: an H100 NVL (95830 MiB) matches the name and fails the class; data, never a gate."""
    h100 = {"name": "NVIDIA H100 80GB HBM3", "memory_mib": 81559, "driver": "580.95.05"}
    c = stack.gpu_class_check(h100, "H100")
    assert c["supported"] is True and c["name_match"] and c["memory_match"] and c["class_memory_mib"] == 81559 and stack.gpu_class_note(c) is None
    nvl = stack.gpu_class_check({"name": "NVIDIA H100 NVL", "memory_mib": 95830}, "H100")
    assert nvl["supported"] is False and nvl["name_match"] is True and nvl["memory_match"] is False
    assert "memory 95830 MiB is not the class's 81559 MiB" in stack.gpu_class_note(nvl)
    a100 = stack.gpu_class_check({"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920}, "H100")
    assert a100["supported"] is False and a100["name_match"] is False and a100["memory_match"] is True and "the GPU is NVIDIA A100-SXM4-80GB" in stack.gpu_class_note(a100)
    assert stack.gpu_class_check(None, "H100")["supported"] is None and stack.gpu_class_check(h100, None)["supported"] is None
    other = stack.gpu_class_check({"name": "NVIDIA H200", "memory_mib": 143771}, "H200")
    assert other["supported"] is None and other["name_match"] is True and other["class_memory_mib"] is None, "no class row: nothing to compare the memory with"


def test_the_card_is_judged_by_compute_capability_name_and_memory_are_notes():
    """GPU_CLASSES is keyed by what the cells are measured by — compute capability; the class name and memory totals are labels. With the
    capability reported (nvidia-smi compute_cap) `supported` is the capability match; a name / memory difference is check's NOTE only."""
    a100 = {"name": "NVIDIA A100 80GB PCIe", "memory_mib": 81920, "driver": "580.95.05", "compute_capability": "8.0"}
    c = stack.gpu_class_check(a100, "A100")
    assert c["supported"] is True and c["capability_match"] and c["memory_match"] and c["card_class"] == "A100" and stack.gpu_class_note(c) is None
    a40 = stack.gpu_class_check(dict(a100, name="NVIDIA A100-SXM4-40GB", memory_mib=40960), "A100")
    assert a40["supported"] is True and a40["memory_match"] is True, "the 40 GB member is the same class (capability 8.0)"
    h_on_a = stack.gpu_class_check({"name": "NVIDIA H100 80GB HBM3", "memory_mib": 81559, "compute_capability": "9.0"}, "A100")
    assert h_on_a["supported"] is False and h_on_a["capability_match"] is False and "compute capability 9.0, not the class's 8.0" in stack.gpu_class_note(h_on_a)
    odd = stack.gpu_class_check(dict(a100, name="NVIDIA A800 80GB", memory_mib=81251), "A100")          # capability 8.0, another label: supported, with a note
    assert odd["supported"] is True and odd["name_match"] is False and "the GPU is NVIDIA A800 80GB" in stack.gpu_class_note(odd)
    assert stack.card_class_of({"name": "NVIDIA H200", "compute_capability": "9.0"}) == "H100", "capability decides the class; the name is a label"
    assert stack.card_class_of({"name": "Tesla T4", "compute_capability": "7.5"}) is None and stack.card_class_of(None) is None


def test_the_only_card_gate_is_compute_capability(monkeypatch):
    """capability_reasons: below 8.0 → one reason BY NAME (gate refuses before any launch); 8.0 / 9.0 / an unreported capability → nothing."""
    assert stack.capability_reasons({"name": "Tesla T4", "compute_capability": "7.5"})[0].startswith("GPU Tesla T4 has compute capability 7.5, below 8.0:")
    for gpu in ({"name": "NVIDIA A100 80GB PCIe", "compute_capability": "8.0"}, {"name": "NVIDIA H100 80GB HBM3", "compute_capability": "9.0"},
                {"name": "NVIDIA B200", "compute_capability": "10.0"}, {"name": "some card", "compute_capability": None}, {"name": "x"}, None):
        assert stack.capability_reasons(gpu) == []
    assert stack.parse_cc("8.0") == (8, 0) and stack.parse_cc("10.3") == (10, 3) and stack.parse_cc("") is None and stack.parse_cc("N/A") is None
    monkeypatch.setattr(stack, "gpu_probe", lambda: {"name": "Tesla T4", "memory_mib": 15360, "driver": "550.1", "compute_capability": "7.5"})
    plan = stack.gate("exact", need_cache=False, check_pins=False)
    assert any("compute capability 7.5, below 8.0" in r for r in plan["reasons"]), plan["reasons"]
    assert stack.gate("off", need_cache=False, check_pins=False)["reasons"] == [], "the stock route is upstream as shipped: its own words decide"


def test_a100_env_mirrors_h100_env():
    """configs/a100.env is configs/h100.env with the target label A100: the same lines but the header (card / tested-on), the target default and
    the probe's self-reference — one deployment shape for both cards."""
    cfg = os.path.join(stack.tree_dir(), "configs")
    h, a = open(os.path.join(cfg, "h100.env")).read().splitlines(), open(os.path.join(cfg, "a100.env")).read().splitlines()
    card = [l for l in a if l.startswith("#") and a.index(l) in range(9, 9 + 16)]           # the per-card difference list (comment block after "Tested on"): the one place a reader finds them
    assert a[9].startswith("# Per-card differences on compute capability 8.0") and len(card) == 16
    for name in ("fused_transition", "pairblock", "CARD_ROWS", "below_min_rows", "above_max_rows", "trailing_piece", "library_op", "fpf_opm", "385 tokens", "CARD_MIN_TOKENS", "tier word", "8.0 cells", "dit_smx", "dit_glue", "msa_trans2_exact"):
        assert any(name in l for l in card), name
    from .. import msa_kernels, pairblock, transition
    assert set(modes.CARD_DROPS) == {"8.0", "10.0"} and any("msa_pwa_exact:pin:cc80" in l for l in card), "the levers that leave a row on 8.0 are named in the list: the MSA PairWeightedAveraging's exact cell"
    assert not any("dit_smx:unproven_cc" in l or "msa_trans2_exact:pin" in l for l in card), "the replicas proven on 8.0 are no longer listed as leaving"
    ranges = [f"[{lo:,}, {hi:,}) rows" for t in (pairblock.CARD_ROWS,) for lo, hi, _piece in t["8.0"].values()] + [f"{piece:,} rows" for t in (pairblock.CARD_ROWS,) for _lo, _hi, piece in t["8.0"].values()] + [f"{n} tokens" for n in msa_kernels.CARD_MIN_TOKENS["8.0"].values()]
    assert set(pairblock.CARD_ROWS) == set(msa_kernels.CARD_MIN_TOKENS) == {"8.0"} and not hasattr(transition, "CARD_ROWS") and all(any(w in l for l in card) for w in ranges), ("the list names every per-card range / piece / floor with its value", ranges)
    assert any("fused_transition" in l and "exact_unvouched" in l for l in card) or any("fused_transition" in l and "no kit row rule" in l for l in card), "the transition's per-card statement is the provider's vouch table, named in the list"
    assert not any("native_triattn" in l for l in card) and any("triatt=core.fast" in l for l in card), "fast / big: the tri-attention site is the provider's tier word on every card (no row pick leaves by name)"
    a = a[:9] + a[9 + 16:]
    assert len(h) == len(a)
    differ = [i + 1 for i, (x, y) in enumerate(zip(h, a)) if x != y]
    assert differ == [1, 2, 3, 9, 11, 22], differ
    assert "MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-A100}" in a[10] and "MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-H100}" in h[10]
    assert all(x.replace("h100", "a100").replace("H100", "A100") == y or i + 1 in (1, 9, 11) for i, (x, y) in enumerate(zip(h, a))), "every other difference is the label"
    assert set(stack.GPU_CLASSES) == {"H100", "A100"} and stack.GPU_CLASSES["A100"]["compute_capability"] == "8.0" and stack.GPU_CLASSES["H100"] == {"compute_capability": "9.0", "memory_mib": 81559, "memory_mib_members": (81559,)}



XL_UNIT_LEVERS = ["xl_trans", "xl_cond", "xl_free", "relpos_lazy"]


def _xl_exit(partial=(), allow=False):
    """The adapter's exit gate block (opt_core.mem.record.AppliedRecord.exit_gate) with the given partial levers."""
    return {"exit_code": (0 if (not partial or allow) else 3), "partial": list(partial), "gated": [], "allow_partial": allow, "incomplete": None,
            "refused": [], "opt_out": "--allow-partial", "allow_partial_source": ("kit" if allow else None),
            "reasons": {n: "predict_step0: never marked" for n in partial}}


def _flash_base(monkeypatch):
    """Put the memory row on its `flash` pair-stack base for one test (modes.BIG_ATTN_BASE = "flash": the staged flash triangle-attention patch in the
    stock Pairformer statements instead of fast's fused pair track) — the F2 report line is that base's evidence."""
    env = {k: v for k, v in modes.MODES["big"]["env"].items() if k not in modes._PAIRTRACK_WORDS}
    xl0 = list(env).index("BOLTZ_PRECISION")
    items = list(env.items()); env = dict(items[:xl0] + list(modes._F2_ROW.items()) + items[xl0:])
    monkeypatch.setitem(modes.MODES["big"], "env", env)
    monkeypatch.setitem(modes.MODES["big"], "levers", [l for l in modes.MODES["big"]["levers"] if l not in ("pairblock", "fused_transition", "pairfuse", "condproj")])
    monkeypatch.setitem(modes.MODES["big"], "attach", [a for a in modes.MODES["big"]["attach"] if a not in ("transition", "pairblock", "pairfuse", "conf")])
    monkeypatch.setitem(modes.MODES["big"], "off", {k: v for k, v in modes.MODES["big"]["off"].items() if k != "condproj"})


def _big_log(**xl):
    """A worker log of the memory row with every trunk lever reported applied and one item run; xl_report (boltz2_opt.big.report()) as given (None = absent)."""
    rep = {"applied": ["expandable_segments"] + XL_UNIT_LEVERS, "mode": "big", "base": "fast",
           "patched": ["boltz.model.layers.transition.Transition.forward", "boltz.model.modules.diffusion_conditioning.DiffusionConditioning.forward",
                       "boltz.model.modules.confidencev2.ConfidenceModule.forward", "boltz.model.modules.encodersv2.RelativePositionEncoder.forward"],
           "stats": {"trans_rowchunked_calls": 40, "trans_stock_calls": 8, "cond_rowchunked_calls": 2, "free_events": 2, "free_tensors": 6, "free_bytes": 3 * 2**30,
                     "relpos_lazy_released": 4, "relpos_lazy_bytes": 2**30, "relpos_lazy_recomputed": 2},
           "settings": {"xl_trans": {"rows": 256}, "xl_cond": {"rows": 256, "fp32": 1}, "xl_free": {}, "relpos_lazy": {}},
           "env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "alloc_effective": "true"},
           "record": {"mode": "big", "refused": [], "off_by_flag": [], "on_by_flag": []}, "exit": _xl_exit(), "n_units": 2}
    templated = bool(xl.pop("templated", False))
    if xl.get("absent"):
        rep = None
    else:
        for k, v in xl.items():
            if k in ("stats", "env", "record"):
                rep[k].update(v)
            else:
                rep[k] = v
    brow = modes.env_row("big"); hoist = int(brow["BOLTZ_DIT_HOIST"]) if brow.get("BOLTZ_DIT_HOIST") else None
    item = {"name": "a", "seed": 101, "graph_sampler_mode": ("stock" if brow.get("BOLTZ_SAMPLER_ROLLOUT") else None), "graph_n_replay": None}
    if hoist:                                                           # the sampler group rides the memory row up to the card's token ceiling (0.3.13): the hoist's per-item census as on fast
        item.update({"hoist_hoist_level": hoist, "hoist_captured_with_cache": 1, "hoist_capture_stock_fallbacks": 0, "hoist_cache_released": False, "hoist_headroom_gated": False, "hoist_token_gated": False})
    log = {"env": {"boltz_levers": ["resid", "mask2"], "f2_patch_active": True, **({"dit_hoist": hoist} if hoist else {})}, "per_item": [item], "events": [],
           "trimul_report": dict(TRIMUL_OK), "templ_report": dict(TEMPL_OK), "msa_report": msa_ok_for(modes.env_row("big"))}   # big carries the fused outer-product mean as fast does (its BOLTZ_FPF_MSA units)
    from . import _faster
    log.update(_faster.reports(modes.env_row("big"), 1))          # + fast's engine adapters the memory row carries (layer_norm replica, trunk graphs, WASTE, MSA dim-64 transition, atom kernels, PAIRFUSE): their exit reports, as on fast
    if modes.env_row("big").get("BOLTZ_TRANSITION"):
        log["transition_report"] = {**TRANSITION_OK, "variant": modes.env_row("big")["BOLTZ_TRANSITION"]}   # fast's fused transition (the pair-track base, 0.3.4)
    if modes.env_row("big").get("BOLTZ_PAIRBLOCK"):
        log["pairblock_report"] = {**PAIRBLOCK_OK, "variant": modes.env_row("big")["BOLTZ_PAIRBLOCK"], "applied": ["pairblock", "flash_triattn"], "min_tokens": 300,
                                   "census": {"served": {"flash": 520, "cueq_below_gate": 32}, "fallback": {}, "errors": {}}}   # fast's fused block with the flash core from 300 tokens
    if modes.env_row("big").get("BOLTZ_WRITER"):
        log["writer_report"] = dict(WRITER_OK)                     # the background writer's census (beside the featurizer on big as on fast)
    if modes.env_row("big").get("BOLTZ_PREFETCH"):
        log["prefetch_report"] = dict(PREFETCH_OK)                 # the persistent featurizer's census (attached last on big as on fast)
    if rep is not None:
        log["xl_report"] = rep
    if templated:                                                    # a run whose passes carried live template slots: the template module ran (templskip_report live > 0, skipped 0)
        tr = dict(log.get("templskip_report") or {}); c = dict(tr.get("census") or {})
        c.update(live=c.get("calls", 4) or 4, skipped=0, served=0); tr["census"] = c; log["templskip_report"] = tr
    return log


def test_attachment_probe_names_a_missing_core_lever(monkeypatch):
    """stack.gate: a memory mode whose adapter is not importable, or carries no installed levers, is a named reason (check → NOT ACTIVE, pred
    refuses before any worker starts); an adapter with levers passes the probe."""
    import importlib.util as iu
    monkeypatch.setattr(iu, "find_spec", lambda name, *a, **k: None)
    got = stack.attachment_problems("big")
    templ_missing = "attachment templ: boltz2_opt.templates is not importable (the mode refuses by name)"
    from ..worker_launch import ATTACH as _ATTACH
    assert got == [f"attachment {a}: {_ATTACH[a]['module']} is not importable (the mode refuses by name)" for a in modes.attachments("big")] and got[0] == templ_missing and got[-3:] == ["attachment xl: boltz2_opt.big is not importable (the mode refuses by name)", "attachment writer: boltz2_opt.writer is not importable (the mode refuses by name)", "attachment prefetch: boltz2_opt.prefetch is not importable (the mode refuses by name)"], got   # one named reason per attachment of the memory row, in row order: fast's adapters it carries, the xl hook, the featurizer last
    from ..worker_launch import ATTACH
    missing = lambda m: [f"attachment {a}: {ATTACH[a]['module']} is not importable (the mode refuses by name)" for a in modes.attachments(m)]   # noqa: E731 — one named reason per attachment of the row, in row order
    assert stack.attachment_problems("fast") == missing("fast") and stack.attachment_problems("exact") == missing("exact") and stack.attachment_problems("off") == []
    assert missing("exact")[:4] == [templ_missing] + [f"attachment {a}: boltz2_opt.{m} is not importable (the mode refuses by name)" for a, m in (("trimul", "trimul"), ("transition", "transition"), ("pairblock", "pairblock"))]
    monkeypatch.undo()
    assert stack.attachment_problems("big") == [] == stack.attachment_problems("fast") == stack.attachment_problems("exact"), "the adapters as shipped carry their levers / guards; the census modules are staged files"
    import types, importlib as il
    monkeypatch.setattr(il, "import_module", lambda name: types.SimpleNamespace(LEVERS=(), GUARDS=()))
    assert stack.attachment_problems("big") == [f"attachment {a}: {_ATTACH[a]['module']} has no {'guards' if a in modes.GUARD_ATTACHMENTS else 'levers'} installed yet (the mode refuses by name)" for a in modes.attachments("big")] and stack.attachment_problems("big")[-3] == "attachment xl: boltz2_opt.big has no levers installed yet (the mode refuses by name)"


F2_OK_LINE = '[boltz_flash_triattn_patch] {"calls": 4, "flash_calls": 3, "stock_calls": 1, "min_tokens": 300, "max_n": 400}'


def test_xl_gate_state_table():
    """stack.evidence on the memory rows: the adapter's own report (xl_report) is the evidence, and every degraded state is a named problem;
    the good state has none."""
    good = stack.evidence("big", _big_log(), "")[1]
    assert good == [], good                                                             # on fast's pair track the block / driver reports in the log are the evidence; no F2 line is read
    states = [
        (dict(absent=True), "no xl_report in the worker log"),
        (dict(applied=["expandable_segments", "xl_trans", "xl_cond"]), "memory levers not reported applied: xl_free,relpos_lazy"),
        (dict(applied=["expandable_segments"] + XL_UNIT_LEVERS + ["trimul_rowchunk"]), "memory levers applied outside the row: trimul_rowchunk"),
        (dict(stats={"trans_rowchunked_calls": 0}, templated=True), "memory levers never acted although items ran (size gate BOLTZ_XL_MIN_TOKENS=0): xl_trans"),   # a templated pass ran the template stack: its transitions were the lever's to chunk
        (dict(stats={"cond_rowchunked_calls": 0, "free_events": 0, "relpos_lazy_released": 0}), "memory levers never acted although items ran (size gate BOLTZ_XL_MIN_TOKENS=0): xl_cond,xl_free,relpos_lazy"),
        (dict(record={"refused": [{"lever": "relpos_lazy", "precondition": "hooks.relpos_cls", "reason": "no hook"}]}), "memory levers refused at apply: relpos_lazy: hooks.relpos_cls: no hook"),
        (dict(exit={}), "no exit gate in xl_report"),
        (dict(exit=_xl_exit(partial=["xl_cond"])), "memory levers partial per the adapter's census (exit gate refused): xl_cond: predict_step0: never marked"),
        (dict(exit=_xl_exit(partial=["xl_cond"], allow=True)), "memory levers partial per the adapter's census (exit gate refused): xl_cond"),
        (dict(env={"PYTORCH_CUDA_ALLOC_CONF": None}), "allocator setting not in the worker process (xl_report.env.PYTORCH_CUDA_ALLOC_CONF=None)"),
        (dict(env={"alloc_effective": "false"}), "allocator setting carried in the environment but not effective in the worker process"),
    ]
    for state, expect in states:
        problems = stack.evidence("big", _big_log(**state), F2_OK_LINE)[1]
        assert any(p.startswith(expect) for p in problems), (state, expect, problems)
    # every pass untemplated (templ_skip elided the template stack) and the layer driver on the C=128 stacks: the row-chunked transition had no caller — named, not a problem
    ev, problems = stack.evidence("big", _big_log(stats={"trans_rowchunked_calls": 0}), F2_OK_LINE)[:2]
    assert problems == [] and ev.get("xl_trans_idle_by") == "pairfuse@c128+templ_skip", (problems, ev.get("xl_trans_idle_by"))
    # the attach route's refusal line is a forbidden log pattern
    assert "[boltz2-opt attach] REFUSED" in stack.FORBIDDEN_LOG_PATTERNS
    assert stack.evidence("big", _big_log(), F2_OK_LINE + "\n[boltz2-opt attach] REFUSED xl: ModuleNotFoundError")[1][0].startswith("the kit reported a lever not applied")
    # the other modes read no XL evidence
    assert "xl_report" in stack.evidence("exact", {"env": {"boltz_levers": []}, "per_item": []}, "")[0]


def test_trimul_gate_state_table():
    """stack.evidence on the rows that carry the fused TriMul (fast, big): the adapter's own report (trimul_report) is the evidence — a
    missing report, a lever not applied, a refused core gate (kernel error / unexpected fallback / served nothing) are named problems."""
    kon = ["resid", "mask2"]
    f2 = '[boltz_flash_triattn_patch] {"calls": 4, "flash_calls": 3, "stock_calls": 1, "min_tokens": 300, "max_n": 400}'
    assert stack.evidence("fast", _log(kon, f2=True, mode="fast"), f2)[1] == []
    states = [
        (None, "no trimul_report in the worker log"),
        ({**TRIMUL_OK, "applied": []}, "fused TriMul fpf_trimul not reported applied"),
        ({**TRIMUL_OK, "gate": {"ok": False, "reason": "unexpected fallback: fallback=dtype:2"}}, "fused TriMul gate refused: unexpected fallback: fallback=dtype:2"),
        ({**TRIMUL_OK, "gate": {"ok": False, "reason": "kernel errors: errors=RuntimeError:1"}}, "fused TriMul gate refused: kernel errors"),
    ]
    for rep_, expect in states:
        problems = stack.evidence("fast", _log(kon, f2=True, trimul=rep_, mode="fast"), f2)[1]
        assert any(p.startswith(expect) for p in problems), (rep_, expect, problems)
    idle = {**TRIMUL_OK, "census": {"served": 0, "fallback": {"below_min_tokens": 8}, "errors": {}}, "gate": {"ok": True, "idle": True, "reason": "idle: …"}}
    ev, problems = stack.evidence("fast", _log(kon, f2=True, trimul=idle, mode="fast"), f2)
    assert problems == [] and ev["trimul_idle"] is True, "an input below the size gate: the installed lever idles (boltz2_opt.trimul.verdict), the run is active"
    assert any(p.startswith("no trimul_report") for p in stack.evidence("exact", _log(kon, trimul=None, mode="exact"), "")[1]), "the exact row carries the exact TriMul: its report is required"
    for key, what in (("transition", "fused transition"), ("pairblock", "fused triangle-attention block")):   # the core pair-track adapters: missing report, not applied, wrong variant, refused gate, served nothing = named problems; idle on declared paths = active
        base = dict(TRANSITION_OK if key == "transition" else PAIRBLOCK_OK, variant=modes.env_row("exact")["BOLTZ_" + key.upper()])
        for rep_, expect in ((None, f"no {key}_report in the worker log"), ({**base, "applied": []}, f"{what} not reported applied"),
                             ({**base, "variant": "zzz"}, f"{what} variant 'zzz'"), ({**base, "gate": {"ok": False, "reason": "unexpected fallback: x"}}, f"{what} gate refused: unexpected fallback: x"),
                             ({**base, "census": {"served": {}, "fallback": {"kernels_off": 3}, "errors": {}}}, f"{what} installed but served no call")):
            problems = stack.evidence("exact", _log(kon, **{key: rep_}, mode="exact"), "")[1]
            assert any(p.startswith(expect) for p in problems), (key, rep_, expect, problems)
        idle_ = {**base, "census": {"served": {}, "fallback": {"below_min_rows": 96}, "errors": {}}, "gate": {"ok": True, "idle": True, "reason": "idle: …"}}   # every call under the card's row floor (the adapter's verdict)
        ev, problems = stack.evidence("exact", _log(kon, **{key: idle_}, mode="exact"), "")
        assert not any(what in p for p in problems) and ev[key + "_idle"] is True, (key, problems)


def test_trimul_verdict_idles_only_on_declared_stock_paths():
    """boltz2_opt.trimul.verdict over the core ladder (opt_core.trimul.Lever, no provider import needed): served 0 with only declared fallbacks =
    idle (ok); an undeclared fallback or a kernel error = the core's refusal; served > 0 = the core's ok.  The declared words: none under a
    tolerance tier word (the provider names a row for every cell), `library_op:<row>` under the exact word (the cells whose exact row is the
    library op: the engine's own call serves), `above_max_tokens` when a row names a ceiling."""
    from boltz2_opt import trimul as tm
    from opt_core import trimul as T
    class P:                                    # a provider stub: eligible below is never reached in these censuses
        name = "trimul:exact"; resolved_from = "core"
        def describe(self): return "trimul:tmk3_exact@exact (stub)"
    def lever(counts, expected=tm.EXPECTED_OF["exact"], mode="exact"):
        lv = T.Lever(tm.TAG, mode, provider=P(), min_tokens=0, expected=expected)
        for (d, key), n in counts.items():
            for _ in range(n):
                lv._count(d, key)
        return lv
    assert tm.EXPECTED_OF == {"1": (), "exact": ("library_op",)} and not hasattr(tm, "min_tokens") and tm.requested({"BOLTZ_FPF_TRIMUL": "1"}) and tm.requested({"BOLTZ_FPF_TRIMUL": "exact"}) and not tm.requested({})
    v = tm.verdict(lever({("outgoing", "fallback:library_op:cueq"): 4, ("incoming", "fallback:library_op:cueq"): 4}))
    assert v["ok"] is True and v["idle"] is True and v["reason"] == tm.IDLE, "every cell of the input names the library op on this card: the installed lever idles, gate open"
    assert v.get("aside") or str(v.get("core_reason") or "").startswith("mode exact routed"), ("the core either names that census its own aside (opt_core >= 0.5.211: gate ok, "
                                                                                             "words aside:<word>) or refuses it as 'routed … served 0' (older cores) — idle either way", v)
    v = tm.verdict(lever({("outgoing", "fallback:library_op:cueq"): 4, ("incoming", "fallback:dtype"): 1}))
    assert v["ok"] is False and v["idle"] is False and "unexpected fallback" in v["reason"]
    v = tm.verdict(lever({("outgoing", "error:RuntimeError"): 1, ("outgoing", "served:trimul:exact"): 3}))
    assert v["ok"] is False and "kernel errors" in v["reason"]
    v = tm.verdict(lever({("outgoing", "served:trimul:exact"): 48, ("incoming", "served:trimul:exact"): 48, ("incoming", "fallback:library_op:torch_math"): 1}))
    assert v == {"ok": True, "idle": False, "reason": None}
    v = tm.verdict(lever({("outgoing", "served:trimul:exact"): 40, ("incoming", "fallback:tmk3_exact:launch_grid_y>65535"): 1}))
    assert v["ok"] is False and "unexpected fallback" in v["reason"], "a kernel row's refusal with the tensors in hand is not a declared path: the gate refuses (the tier word's table admits only rows that serve the shape)"
    v = tm.verdict(lever({("outgoing", "fallback:below_min_tokens"): 4}, expected=tm.EXPECTED_OF["1"], mode="fast"))
    assert v["ok"] is False and "unexpected fallback" in v["reason"], "no token gate of this tree's: below_min_tokens is nobody's word"
    v = tm.verdict(lever({("outgoing", "served:trimul:fast"): 4, ("incoming", "fallback:above_max_tokens"): 2}, expected=tm.EXPECTED_OF["1"] + (tm.ABOVE_MAX,), mode="fast"))
    assert v == {"ok": True, "idle": False, "reason": None}, "a row's ceiling is a declared word"


def test_template_guard_is_required_evidence_on_the_worker_route():
    """Every worker mode attaches the template guard (modes.attachments: templ) — its templ_report is required evidence under pred / warm
    (attached=True; attached=False = a worker launched bare, no kit mode); an ALL DUMMY templates line in the worker's output
    is a named event, not a problem."""
    kon = ["resid", "mask2"]
    assert all("templ" in modes.attachments(m) for m in ("exact", "fast", "big"))
    assert modes.lever_attachments("exact") == ["trimul", "transition", "pairblock", "triattn_exact", "sampler", "ditexact", "exactln", "waste", "msa2", "graph", "templskip", "atom", "writer", "prefetch"] and modes.lever_attachments("exact")[:3] == modes.lever_attachments("fast")[:3]
    assert modes.lever_attachments("fast") == ["trimul", "transition", "pairblock", "pairfuse", "sampler", "exactln", "waste", "msa", "msa2", "conf", "graph", "templskip", "atom", "writer", "prefetch"]
    ev, problems = stack.evidence("exact", _log(kon, templ=None, mode="exact"))
    assert any("no templ_report" in p for p in problems)
    ev, problems = stack.evidence("exact", _log(kon, templ=None, mode="exact"), attached=False)
    assert problems == [], problems
    all_dummy = "[boltz2-opt] TEMPLATES ALL DUMMY record=8IO9: template slots all dummy (declared=4 live_slots=0/4); the input runs untemplated, as `boltz predict` runs it\n"
    ev, problems = stack.evidence("exact", _log(kon, mode="exact"), stdout_text=all_dummy)
    assert problems == [] and ev["template_all_dummy"] == [all_dummy.strip()], "an all-dummy templated input is a named event and runs as stock runs it; never a refusal"
    dropped = "[boltz2-opt] TEMPLATE DROPPED record=7VCU template=t1 chain=A reason=unresolved_aligned_residues 0/212\n"
    ev, problems = stack.evidence("exact", _log(kon, mode="exact"), stdout_text=dropped)
    assert problems == [] and ev["template_dropped"] == [dropped.strip()], "a dropped template is a named event"
    templ = dict(TEMPL_OK, items_checked=1, declared=4, live=1, per_item=[{"record": "7VCU", "declared": 4, "names": 1, "slots": 1, "live_slots": 1}])
    ev, problems = stack.evidence("exact", _log(kon, templ=templ, mode="exact"))
    assert problems == [] and "templates=declared:4,live:1,items:1" in rep.applied_line(ev)
    assert "templates=none" in rep.applied_line(stack.evidence("exact", _log(kon, mode="exact"))[0])



def test_template_guard_census_pure_functions():
    """templates.featurizer_census / item_census / check_batch on plain records: the guard counts COORDINATE-BEARING aligned tokens
    (template_mask_cb | template_mask_frame), not index-mapped ones — a declared template whose chain carries no resolved coordinates is
    DROPPED by name (structure-free: upstream would feed its residue types alone); an item whose every slot is structure-free is REFUSED by
    name (TemplatesAllDummy); an item that declares nothing is untouched."""
    from boltz2_opt import templates as T

    class TI:
        def __init__(self, name, chain, st, en):
            self.name, self.query_chain, self.query_st, self.query_en = name, chain, st, en

    class Rec:
        def __init__(self, rid, templates):
            self.id, self.templates = rid, templates
    tis = [TI("t1", "A", 0, 4), TI("t1", "B", 0, 3), TI("t2", "A", 1, 3)]
    token_asym = [0, 0, 0, 0, 1, 1, 1]; chain_asym = {"A": 0, "B": 1}
    mask = [[1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0]]      # index-mapped: slot t1: 3 A tokens, 0 B tokens; slot t2: 2 A tokens (a structurally VOID template)
    cb = [[1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]        # coordinates: t1 has a CB at token 0 and a frame at token 1; t2 has none anywhere
    fr = [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    coord = T.coord_mask({"template_mask_cb": cb, "template_mask_frame": fr})
    assert coord == [[1.0, 1.0, 0, 0, 0, 0, 0][:2] + [0.0] * 5, [0.0] * 7]
    rows = T.featurizer_census("X", tis, token_asym, chain_asym, ["t1", "t2"], mask, coord)
    assert [(r["template"], r["chain"], r["resolved_tokens"], r["aligned_tokens"], r["declared_residues"], r["dropped"]) for r in rows] == \
        [("t1", "A", 2, 3, 4, False), ("t1", "B", 0, 0, 3, True), ("t2", "A", 0, 2, 2, True)], rows
    lines = T.census_lines(rows)
    assert sum(l.startswith(T.DROPPED_MARK) for l in lines) == 2 and any("aligned_residues_resolved=2/3 declared_residues=4" in l for l in lines), lines
    assert any(l.endswith("template=t2 chain=A reason=unresolved_aligned_residues 0/2") for l in lines), "a structurally void template (mapped, no coordinates) is DROPPED by name"
    import pytest
    with pytest.raises(KeyError):
        T.coord_mask({"template_mask": mask})                      # a feature dict without the coordinate masks is refused, never read as live
    T._STATE.update(per_item=[], seen=set(), refused=[])
    feats = {"record": [Rec("X", tis)], "template_mask": [mask], "template_mask_cb": [cb], "template_mask_frame": [fr]}
    fresh = T.check_batch(feats)
    assert fresh == [{"record": "X", "declared": 3, "names": 2, "slots": 2, "live_slots": 1}]
    assert T.check_batch(feats) == [], "one census per record"
    zero = [[0] * 7]
    assert T.check_batch({"record": [Rec("N", [])], "template_mask": [zero], "template_mask_cb": [zero], "template_mask_frame": [zero]}) == [], "no template declared: untouched"
    v = T.check_batch({"record": [Rec("V", tis)], "template_mask": [[[1] * 7, [1] * 7]], "template_mask_cb": [[[0] * 7, [0] * 7]], "template_mask_frame": [[[0] * 7, [0] * 7]]})
    assert v == [{"record": "V", "declared": 3, "names": 2, "slots": 2, "live_slots": 0}], "every slot index-mapped but structure-free: counted (all dummy), no refusal — the input runs as stock runs it"
    d = T.check_batch({"record": [Rec("D", tis)], "template_mask": [[[0] * 7, [0] * 7]], "template_mask_cb": [[[0] * 7, [0] * 7]], "template_mask_frame": [[[0] * 7, [0] * 7]]})
    assert d[0]["live_slots"] == 0, "nothing mapped at all: counted, no refusal"
    r = T.report()
    assert r["items_checked"] == 3 and r["all_dummy"] == ["V", "D"] and r["declared"] == 9 and r["live"] == 1
    T._STATE.update(per_item=[], seen=set(), all_dummy=[])


def test_templates_line_words_one_gpu_and_n_gpu(capsys, monkeypatch):
    """The per-input TEMPLATES line: `… slots=<T> live_slots=<j> real=<j>/<T>` on every mode of the kit; under `--n_gpu P` (boltz2_opt.rowpair
    imported with its world size P > 1) the same line ends `form=row_born n_gpu=<P>` — the template pair features are born as each rank's rows
    (rowpair._template_rows) — and the census row records form=row_born. Untemplated records print nothing either way."""
    import sys, types
    from boltz2_opt import templates as T

    class TI:
        def __init__(self, name, chain, st, en):
            self.name, self.query_chain, self.query_st, self.query_en = name, chain, st, en

    class Rec:
        def __init__(self, rid, templates):
            self.id, self.templates = rid, templates
    tis = [TI("t1", "A", 0, 4), TI("t2", "A", 1, 3)]
    live = {"template_mask": [[[1] * 5, [1] * 5]], "template_mask_cb": [[[1, 1, 0, 0, 0], [0] * 5]], "template_mask_frame": [[[0] * 5, [0] * 5]]}
    T._STATE.update(per_item=[], seen=set(), all_dummy=[])
    monkeypatch.delitem(sys.modules, "boltz2_opt.rowpair", raising=False)          # one GPU: the row-sharded pair stack is not imported
    assert T.tp_n_gpu() == 1
    assert T.check_batch(dict(live, record=[Rec("one", tis)])) == [{"record": "one", "declared": 2, "names": 2, "slots": 2, "live_slots": 1}]
    out = capsys.readouterr().err.strip().splitlines()
    assert out == ["[boltz2-opt] TEMPLATES record=one declared=2 names=2 slots=2 live_slots=1 real=1/2"], out
    monkeypatch.setitem(sys.modules, "boltz2_opt.rowpair", types.SimpleNamespace(n_gpu=lambda: 2))   # --n_gpu 2: the installed module's world size
    assert T.tp_n_gpu() == 2
    rows = T.check_batch(dict(live, record=[Rec("two", tis)]))
    assert rows == [{"record": "two", "declared": 2, "names": 2, "slots": 2, "live_slots": 1, "form": "row_born", "n_gpu": 2}], rows
    out = capsys.readouterr().err.strip().splitlines()
    assert out == ["[boltz2-opt] TEMPLATES record=two declared=2 names=2 slots=2 live_slots=1 real=1/2 form=row_born n_gpu=2"], out
    dummy = {"template_mask": [[[1] * 5, [1] * 5]], "template_mask_cb": [[[0] * 5, [0] * 5]], "template_mask_frame": [[[0] * 5, [0] * 5]]}
    T.check_batch(dict(dummy, record=[Rec("dud", tis)]))                        # all slots structure-free under --n_gpu 2: the named event, as on one GPU
    out = capsys.readouterr().err.strip().splitlines()
    assert out[0].endswith("live_slots=0 real=0/2 form=row_born n_gpu=2") and out[1].startswith(T.ALL_DUMMY_MARK + " record=dud"), out
    assert T.check_batch(dict(live, record=[Rec("none", [])])) == [] and capsys.readouterr().err == ""
    from boltz2_opt import report as RP                                        # the caller echoes the worker's rows after the run (report.run_exit): the same words
    echoed = RP.template_lines({"templ_report": T.report()})
    assert echoed == ["[boltz2-opt] TEMPLATES record=one declared=2 names=2 slots=2 live_slots=1 real=1/2",
                      "[boltz2-opt] TEMPLATES record=two declared=2 names=2 slots=2 live_slots=1 real=1/2 form=row_born n_gpu=2",
                      "[boltz2-opt] TEMPLATES record=dud declared=2 names=2 slots=2 live_slots=0 real=0/2 form=row_born n_gpu=2",
                      T.ALL_DUMMY_MARK + " record=dud: template slots all dummy (declared=2 live_slots=0/2); the input runs untemplated, as `boltz predict` runs it"], echoed
    assert RP.template_lines({"templ_report": None}) == [] and RP.template_lines({}) == []
    T._STATE.update(per_item=[], seen=set(), all_dummy=[])


def test_lever_lines_speak_the_cores_grammar_for_every_row_lever():
    """After APPLIED, one LEVER line per lever of the row (report.lever_lines over opt_core.report.lever_line): name, state, impl, origin and a
    strategy id of the line's form (opt_core.report.strategy_form: F<k>.<name> or LOCAL.boltz2.<name>), evidence pairs from the worker's log."""
    from opt_core.report import strategy_form
    from boltz2_opt import registry

    def parse_line(line):
        head, _, rest = line.partition(" LEVER ")
        assert head == rep.PREFIX, line
        return dict(tok.split("=", 1) for tok in rest.split(" ") if "=" in tok)
    for name, L in registry.LEVERS.items():
        assert strategy_form(L["strategy"]) == L["strategy"] and L["origin"] in ("kit", "core") and L["impl"]
    kon = ["resid", "mask2"]
    ev, problems = stack.evidence("fast", _log(kon, mode="fast"))
    assert problems == [], problems
    report = {"mode": "fast", "levers_fallback": [], "gates": {}}
    lines = rep.lever_lines("fast", ev, report)
    assert len(lines) == len(modes.levers("fast"))
    parsed = {parse_line(l)["name"]: parse_line(l) for l in lines}
    assert set(parsed) == set(modes.levers("fast"))
    assert parsed["flash_triattn"]["state"] == "on" and parsed["flash_triattn"]["strategy"] == "F1.flash_triatt"
    assert parsed["pairblock"]["state"] == "on" and parsed["pairblock"]["origin"] == "core" and parsed["fused_transition"]["state"] == "on"
    assert parsed["fpf_trimul"]["state"] == "on" and parsed["fpf_trimul"]["origin"] == "core" and parsed["fpf_trimul"]["served"] == "96"
    assert parsed["resid"]["state"] == "on" and parsed["rollout"]["replays"] == str(199 * 2) and parsed["rollout"]["captures"] == "2"
    assert parsed["pairfuse"]["providers"] == f"trimul=v4,attn={modes.env_row('fast')['BOLTZ_PAIRBLOCK']},transition=pair_fused" and parsed["pairblock"]["superseded_by"] == "pairfuse@c128" == parsed["fused_transition"]["superseded_by"] == parsed["fpf_trimul"]["superseded_by"], "the layer driver names its providers, the block adapters say who serves the C=128 stacks"
    assert parsed["dit_fused"]["numerics"] == "bf16_token_transformer" and parsed["atom_gemm"]["word"] == "bf16" and parsed["msa_trans2"]["numerics"] == "fused_rounding", "the fast row's precision levers name their numerics class"
    assert parsed["waste_ctorskip"]["scope"] == "process_start" and parsed["graph_trunk"]["units"] == "pf,pfnoseq,templ" and parsed["graph_trunk"]["bodies"] == "pf:pairfuse.pfm_forward,pfnoseq:pairfuse.pfnm_forward", "the fast row captures the pair stacks with the PAIRFUSE driver's bodies (graph BODIES table)"
    evb, _ = stack.evidence("big", _big_log(), stdout_text="")
    pb = {parse_line(l)["name"]: parse_line(l) for l in rep.lever_lines("big", evb, {"mode": "big", "levers_fallback": [], "gates": {}})}
    assert pb["flash_triattn"]["state"] == "on" and pb["flash_triattn"]["core"] == modes.env_row("big")["BOLTZ_PAIRBLOCK"] and pb["flash_triattn"]["flash_calls"] == "520" and pb["pairfuse"]["state"] == "on" \
        and pb["xl_trans"]["superseded_by"] == "pairfuse@c128" == pb["pairblock"]["superseded_by"], "big on fast's pair track: the flash core's calls are the fused block's census, the row-chunked Transition names the driver that supersedes it"
    evt = dict(evb); evt["tp_report"] = {"installed": True, "n_gpu": 2, "tpx": {"triatt": {"word": "flash_triattn", "state": "on", "reason": None, "env": "ROWPAIR_TRIATT_CORE"}}}; evt.pop("pairblock_report", None); evt["f2"] = None
    assert rep.lever_state("flash_triattn", evt, {"mode": "big", "n_gpu": 2})[0:2] == ("on", None), "n_gpu > 1 off the flash patch: the row-sharded trunk's row-block attention core hosts the flash kernel (rowpair TPX word)"
    assert "min_rows" not in parsed["fused_transition"], "no row floor in the report (compute capability 9.0): no min_rows token — the line is unchanged"
    assert "min_rows" not in parsed["pairblock"] and "max_rows" not in parsed["pairblock"]
    eva, _ = stack.evidence("exact", _log(kon, transition={**TRANSITION_OK, "variant": "exact", "min_rows": 3137, "max_rows": 3145632, "piece_rows": 1048544},
                                             pairblock={**PAIRBLOCK_OK, "variant": "cueq", "min_rows": 5857, "max_rows": 3145632, "piece_rows": 1048544}, mode="exact"))   # a card with served row rules (8.0): <adapter>_report.min_rows / max_rows / piece_rows
    pa = {parse_line(l)["name"]: parse_line(l) for l in rep.lever_lines("exact", eva, {"mode": "exact", "levers_fallback": [], "gates": {}})}
    assert (pa["fused_transition"]["min_rows"], pa["fused_transition"]["max_rows"], pa["fused_transition"]["piece_rows"]) == ("3137", "3145632", "1048544") and pa["fused_transition"]["state"] == "on", "the card's row rule is a token triple of the fused-transition line"
    assert (pa["pairblock"]["min_rows"], pa["pairblock"]["max_rows"], pa["pairblock"]["piece_rows"]) == ("5857", "3145632", "1048544") and pa["pairblock"]["state"] == "on", "... and of the pairblock line"


def test_ditexact_smx_declared_declines_count_and_undeclared_ones_refuse():
    """The fused softmax replica serves its declared size range: rows outside it take torch's statements by a DECLARED word (ditexact.SMX_DECLARED,
    n_range) — smx_fused + declared declines == 24 * calls passes; an undeclared decline word, or a class whose bit comparison was not served, refuses."""
    from . import _faster
    from .. import ditexact as DX
    row = dict(modes.env_row("exact"))
    def log(census_patch):
        lg = {"env": row, "events": [], "attach": {}, "per_item": [{"name": "a", "seed": 1, "graph_sampler_mode": "off", "graph_n_replay": 0, "hoist_hoist_level": 2, "hoist_captured_with_cache": 0, "hoist_capture_stock_fallbacks": 0}]}
        lg.update(_faster.reports(row, 1))
        c = lg["ditexact_report"]["census"]; c.update(census_patch)
        return lg
    calls = _faster.reports(row, 1)["ditexact_report"]["census"]["calls"]
    assert "n_range" in DX.SMX_DECLARED
    _, pr = stack.evidence("exact", log({"smx_fused": 24 * calls - 48, "smx_declined_by": {"n_range": 48}}), "", attached=False)
    assert not [p for p in pr if p.startswith("DITEXACT")], pr
    _, pr = stack.evidence("exact", log({"smx_fused": 24 * calls - 48, "smx_declined_by": {"layout": 48}}), "", attached=False)
    assert any("undeclared word" in p for p in pr), pr
    _, pr = stack.evidence("exact", log({"smx_fused": 24 * calls - 48, "smx_declined_by": {}}), "", attached=False)
    assert any("!= 24*calls" in p for p in pr), pr
    _, pr = stack.evidence("exact", log({"smx_bitcmp": {"N400": "differs"}}), "", attached=False)
    assert any("bit comparison not served" in p for p in pr), pr


def test_a_core_safe_settings_net_is_a_partial_activation_named_on_active():
    """A core lever that served an UNPINNED cell with its SAFE settings switches every launch of that lever to the safe settings for the rest of
    the process (the pinned cells too): the kit rows pin every cell, so a `settings=safe:<cell>` tail on a core census line is a partial
    activation — `fallbacks=<lever>:safe(<cell>)` on the ACTIVE line, exit 3 unless --allow-partial — never a silent slowdown."""
    from . import _faster
    from .. import report as rep
    row = dict(modes.env_row("fast"))
    def log(tline):
        lg = {"env": row, "events": [], "attach": {}, "per_item": [{"name": "a", "seed": 1, "graph_sampler_mode": "off", "graph_n_replay": 0, "hoist_hoist_level": 2, "hoist_captured_with_cache": 0, "hoist_capture_stock_fallbacks": 0}]}
        lg.update(_faster.reports(row, 1))
        lg["transition_report"] = {"applied": ["fused_transition"], "disabled": {}, "line": tline, "census": {"served": {"transition": 10}, "fallback": {}, "errors": {}}, "gate": {"ok": True, "idle": False}, "patched": ["Transition.forward"]}
        return lg
    clean = "[boltz2-opt] LEVER name=F5.transition state=on impl=fpf origin=core served=2800 fallback=0 fallback_by=none cells=111d4704 served_certified=14000 variants=v2_fold:2800 variant=fast ln=fused keys=fpf.transition:9.0|*"
    netted = clean.replace("keys=fpf.transition:9.0|*", "keys=fpf.transition:9.0|*+safe settings=safe:no_cell:768x1536")
    ev, _ = stack.evidence("fast", log(clean), "", attached=False)
    assert ev["safe_nets"] == {} and stack.partial_activation("fast", ev)[0] == []
    ev, _ = stack.evidence("fast", log(netted), "", attached=False)
    assert ev["safe_nets"] == {"fused_transition": "no_cell:768x1536"}
    fallbacks, _ = stack.partial_activation("fast", ev)
    assert fallbacks == ["fused_transition:safe(no_cell:768x1536)"]
    line = rep.active_line({"mode": "fast", "route": "worker", "gpu": "H100", "levers_applied": modes.levers("fast"), "levers_fallback": fallbacks, "worker": "bz_worker_levf2.py", "kernels": "on"})
    assert "fallbacks=fused_transition:safe(no_cell:768x1536)" in line, line
    assert "exit 3" in rep.partial_line(fallbacks, allow_partial=False) and "fused_transition:safe(no_cell:768x1536)" in rep.partial_line(fallbacks, allow_partial=True)
    # the same tail arriving on the pass' stdout (a core line the adapters do not keep) is found too
    ev, _ = stack.evidence("fast", log(clean), netted + "\n", attached=False)
    assert ev["safe_nets"] == {"fused_transition": "no_cell:768x1536"}


def test_r5a2_class_census_gate_refuses_uncompared_service_and_underbudget_proofs():
    """Any adapter census in the agreed per-class shape is judged — replica output served by a class that is not `proven`, a
    `proven` class whose compares covered fewer output elements than its budget, an unknown state; the MSA exact cells' totals must account for
    every call and serve only through a proven class."""
    ok = {"x_report": {"classes": {"c1": {"state": "proven", "compared_calls": 3, "compared_elems": 4_000_000}, "c2": {"state": "floor", "compared_calls": 0, "compared_elems": 0},
                                   "c3": {"state": "comparing", "compared_calls": 2, "compared_elems": 20_000, "served_uncompared": 0}}}}
    assert stack.r5a2_findings(ok) == [] and stack.r5a2_findings({}) == [] and stack.r5a2_findings({"y_report": {"classes": {"400x8192:u": [2, 4]}}}) == []   # an older shape is not this rule's
    bad = {"x_report": {"classes": {"c3": {"state": "comparing", "compared_calls": 2, "compared_elems": 20_000, "served_uncompared": 5}}},
           "y_report": {"pwa2_census": {"classes": {"33x2:u": {"state": "proven", "compared_calls": 1, "compared_elems": 4224}}}},
           "z_report": {"trans2_census": {"selftest": {"201x64:0": {"state": "locked"}}}}}
    f = stack.r5a2_findings(bad)
    assert len(f) == 3 and any("served 5 call(s) of replica output uncompared while comparing" in x for x in f) and any("proven on 4224 compared output elements < budget 1000000" in x for x in f) and any("state 'locked'" in x for x in f), f
    kon = ["resid", "mask2"]
    log = _log(kon, mode="exact"); mr = log["msa2_report"]
    mr["pwa2_census"]["classes"]["400x8192:u"]["state"] = "comparing"          # served 63 calls, yet no class is proven: replica output left the cell unproven
    ev, problems = stack.evidence("exact", log)
    assert any("MSA2 pwa2x served 63 call(s) with no class proven" in p for p in problems), problems
    log = _log(kon, mode="exact"); log["msa2_report"]["pwa2_census"]["calls"] = 70    # 70 != 63 served + 1 compared + 0 fallback
    ev, problems = stack.evidence("exact", log)
    assert any("MSA2 pwa2x accounting: calls=70" in p for p in problems), problems
    log = _log(kon, mode="exact"); log["msa2_report"]["pwa2_census"]["fallback_by"] = {"small_class:33x2:u": 16, "undetermined:201x400:c": 1}; log["msa2_report"]["pwa2_census"]["fallback"] = 16; log["msa2_report"]["pwa2_census"]["calls"] = 63 + 1 + 16
    ev, problems = stack.evidence("exact", log)
    assert not any("MSA2" in p for p in problems), ("the floor and an undetermined selecting compare are declared words", problems)

