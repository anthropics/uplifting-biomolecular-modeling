"""Test fixtures: the pinned upstream wheel unpacked as an importable `boltz` distribution (no torch needed for what the tests import),
a fake model class for the late-activation rule, and a fake kit worker for the worker route."""
import json
import os
import shutil
import sys
import tempfile
import zipfile

from .. import stack

WHEEL = os.path.join(stack.tree_dir(), "stock", "boltz-2.2.1-py3-none-any.whl")


def unpack_wheel(dst):
    """Unpack stock/boltz-2.2.1-py3-none-any.whl into `dst` (boltz/ + boltz-2.2.1.dist-info/); returns dst."""
    with zipfile.ZipFile(WHEEL) as z:
        z.extractall(dst)
    return dst


def fake_worker_script(evidence_ok=True, f2=None, fail_items=0, f2_stats=None, f2_report=True, lever_report=None, pair_reports=True, kernels="pass", mode="fast", skip_items=(), affinity_json=True, oom_items=()):
    """A stand-in for the kit worker: writes stock's file set per (item, seed) and a worker log with the kit's evidence keys; under
    BOLTZ_TRIATTN_REPORT=1 with the patch active it prints the flash patch's exit report line with `f2_stats` (the kit's STATS keys).
    `kernels` stands in for the KERNELS census the real worker's boltz2_opt.kernels prints (no GPU here): "pass" = the pass line + record of
    `mode`'s route with every expected accelerator engaged (kernels.format_line / modes.kernels_expected: the kit's own grammar and
    expectations); "refuse" = the REQUIRE guard's refusal (KERNELS-REFUSED line, the pass line with verdict REFUSED, exit 5); "contradiction"
    = a pass line whose record says an off-by-route accelerator was called (verdict REFUSED, exit 0: the parent must refuse); "nostep" = a pass line
    whose verdict is NO-STEP although inputs were accepted (kernels.verdict_word: no predict_step ran); "none" = no line. `skip_items`: input
    names the stand-in treats as skipped by the stock parser exactly as the real worker does (skipped.line once per input, a `skipped` log
    entry, no outputs, no PHASE / PEAK line, no predict_step: with every input skipped the census verdict is NO-STEP). `oom_items`: input
    names whose every (input, seed) unit boltz's own predict_step skips on CUDA out of memory, as the real worker sees it (boltz's WARNING
    line on stdout, a `failed` log entry with skipped.oom_reason, no outputs; predict_step ran, so the census is unaffected)."""
    if lever_report is None:   # the kit's report() shape (boltz_trunk_levers.py report()): the row's levers applied, no scope error
        lever_report = {"applied": True, "levers": ["resid", "mask2"], "stats": {"mask_scope_trivial": 52, "pf_layer_resid": 256}, "torch": "2.12.0"}
    if f2_stats is None:
        f2_stats = {"calls": 4, "flash_calls": 3, "stock_calls": 1, "stock|n<300|pairformer.start|N=199|use_kernels=False": 1, "flash|pairformer.start|q=[1, 398, 4, 398, 32]|torch.bfloat16|chunk=None|use_kernels=False": 3}
    return f'''
import argparse, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--batch"); ap.add_argument("--mode"); ap.add_argument("--kernels"); ap.add_argument("--num_workers"); ap.add_argument("--pipeline"); ap.add_argument("--keep_on_gpu")
a = ap.parse_args(); B = json.load(open(a.batch)); OUT = B["out_dir"]
assert "BOLTZ2_OPT" not in os.environ, "BOLTZ2_OPT must not reach the worker"
env = os.environ
def LEV(v):   # the row's levers, literally (BOLTZ_LEVERS=resid,mask2): the stub reports what the row asked for, nothing more
    return [x for x in v.split(",") if x]
LOG = {{"events": [], "per_item": [], "env": {{"boltz_levers": LEV(env.get("BOLTZ_LEVERS", "")),
        "graph_diffusion": env.get("BOLTZ_GRAPH_DIFFUSION") if {evidence_ok!r} else "stock", "dit_hoist": int(env.get("BOLTZ_DIT_HOIST", "0")) if {evidence_ok!r} else 0,
        "f2_patch_active": {f2!r}, "BOLTZ_TRIATTN": env.get("BOLTZ_TRIATTN")}}, "lever_report": {lever_report!r}}}
if env.get("BOLTZ_WRITER"):     # the background writer's census (boltz2_opt.writer.report(), merged by the attach hook in the real worker): joined, every write / move landed
    LOG["writer_report"] = {{"installed": True, "disposition": None, "switch": env["BOLTZ_WRITER"], "applied": ["writer_overlap"], "backend": "process", "items": 2, "submitted": 2, "served": 2, "inline": 0,
                            "inline_by": {{}}, "fallback": 0, "fallback_by": {{}}, "failed": 0, "errors": [], "moves": 2, "moves_done": 2, "queued_max": 1, "join_s": 0.1, "bytes_written": 4096,
                            "helper_pid": 4321, "forked_before_cuda": True, "helper_cuda_initialized": False, "dead": None, "gate": {{"ok": True, "why": [], "idle": False}}}}
if env.get("BOLTZ_PREFETCH"):   # the persistent featurizer's census (boltz2_opt.prefetch.report(), merged by the attach hook in the real worker): every input served, first input bit-compared
    LOG["prefetch_report"] = {{"installed": True, "disposition": None, "switch": env["BOLTZ_PREFETCH"], "applied": ["prefetch"], "served": 2, "fallback": 0, "fallback_by": {{}}, "forks_avoided": 1, "ref_forks": 1,
                              "forked_before_cuda": True, "helper_cuda_initialized": False, "pin": "parent", "bit_compare": "equal:78", "refused": None, "gate": {{"ok": True, "why": []}}}}
LOG["templ_report"] = {{"installed": True, "items_checked": 0, "declared": 0, "live": 0, "refused": [], "per_item": []}}   # the template guard's census (boltz2_opt.templates.report(), merged by the attach hook in the real worker): no input declared a template
if env.get("BOLTZ_FPF_TRIMUL") in ("1", "exact"):   # the fused TriMul adapter's report (boltz2_opt.trimul.report(), merged by the attach hook in the real worker): every fallback a declared one
    LOG["trimul_report"] = {{"applied": ["fpf_trimul" if env["BOLTZ_FPF_TRIMUL"] == "1" else "fpf_trimul_exact"], "disabled": {{}}, "line": "[boltz2-opt] LEVER name=F2.trimul state=on", "census": {{"served": 96, "fallback": {{"below_min_tokens": 8}}, "errors": {{}}}}, "gate": {{"ok": True, "reason": None}}, "variant": env["BOLTZ_FPF_TRIMUL"]}}
for _sw, _key, _lever in ((("BOLTZ_TRANSITION", "transition_report", "fused_transition"), ("BOLTZ_PAIRBLOCK", "pairblock_report", "pairblock"), ("BOLTZ_TRIATTN_EXACT", "triattn_exact_report", "triattn_exact"), ("BOLTZ_EXACTLN", "exactln_report", "exactln"), ("BOLTZ_TEMPL_SKIP", "templskip_report", "templ_skip"), ("BOLTZ_PAIRFUSE", "pairfuse_report", "pairfuse")) if {pair_reports!r} else ()):   # the core pair-track adapters' reports (boltz2_opt.transition / .pairblock / .exactln / .pairfuse report()): served, every fallback a declared one
    if env.get(_sw):
        LOG[_key] = {{"applied": [_lever] + (["flash_triattn"] if env.get(_sw) in ("flash", "k2b") else []) + (["exactln_resid"] if _lever == "exactln" and env.get("BOLTZ_EXACTLN_RESID") == "on" else []) + (["pairblock_c64"] if (_key == "pairblock_report" and env.get("BOLTZ_PAIRBLOCK_C64") == "1") else []), "disabled": {{}}, "line": "[boltz2-opt] LEVER name=" + _key + " state=on", "census": {{"served": {{"x": 100}}, "fallback": {{}}, "errors": {{}}}}, "gate": {{"ok": True, "idle": False, "reason": None}}, "variant": env[_sw]}}
        if _lever == "exactln" and env.get("BOLTZ_EXACTLN_RESID") == "on":   # the residual pass rides the replica (boltz2_opt.exactln.report()["resid"]): fused calls, every parked LayerNorm taken
            LOG[_key]["resid"] = {{"fused": 90, "ln_taken": 90, "ln_unused": 0, "below_min_numel": 10}}
if env.get("BOLTZ_FPF_MSA") and {pair_reports!r}:   # the fused MSA-module kernels' report (boltz2_opt.msa_kernels.report(), merged by the attach hook in the real worker): every unit served, no fallback
    _us = [u for u in env["BOLTZ_FPF_MSA"].split(",") if u]; _lev = {{"opm": "fpf_opm", "pwa": "fpf_pwa"}}
    LOG["msa_report"] = {{"applied": [_lev[u] for u in _us], "disabled": {{}}, "units": {{u: {{"lever": _lev[u], "census": {{"served": 48, "fallback": 0, "fallback_by": {{}}, "errors": {{}}, "calls": 48}}, "gate": {{"ok": True, "idle": False, "reason": None}}, "cfg": "f1" if u == "opm" else "default", "tma": True}} for u in _us}},
                         "lines": [], "gate": {{"ok": True, "idle": False, "reason": None}}, "patched": [], "variant": ",".join(_us), "expected": []}}
if env.get("BOLTZ_PAIRFUSE") and {pair_reports!r}:   # the layer driver's providers word on its report
    LOG["pairfuse_report"].update(residency=env["BOLTZ_PAIRFUSE"], providers="trimul=v4,attn=" + (env.get("BOLTZ_PAIRBLOCK") or "cueq") + ",transition=pair_fused")
if env.get("BOLTZ_GRAPH_TRUNK", "").strip().lower() not in ("", "0", "off", "none") and {pair_reports!r}:   # the trunk CUDA-graph lever's report (boltz2_opt.graph.report(), merged by the attach hook in the real worker): units applied, replays served, no capture error
    _gu = [u for u in env["BOLTZ_GRAPH_TRUNK"].split(",") if u]
    LOG["graph_report"] = {{"applied": ["graph_trunk"], "units": _gu, "disabled": {{}}, "line": "[boltz2-opt] LEVER name=graph_trunk state=on", "census": {{u: ({{"replayed": 10, "captured": 1, "eager:warm": 1}} if u != "templ" else {{"boundaries_served": 4, "boundaries_built": 1}}) for u in _gu}},
                           "gate": {{"ok": True, "idle": False, "reason": None}}, "patched": _gu, "captures": [], "capture_s_total": 0.3, "pool_gib_max": 1.0, "static_gib": 0.1, "generations_evicted": 0, "resident": [], "errors": {{}},
                           "min_tokens": int(env.get("BOLTZ_GRAPH_TRUNK_MIN_TOKENS", "0") or 0), "max_tokens": int(env.get("BOLTZ_GRAPH_TRUNK_MAX_TOKENS", "0") or 0), "keep": 1, "templ_patched": [], "templ_refused": {{}}, "switch": "BOLTZ_GRAPH_TRUNK", "spec": env["BOLTZ_GRAPH_TRUNK"],
                           "bodies": {{u: ({{"pf": "pairfuse.pfm_forward", "pfnoseq": "pairfuse.pfnm_forward"}}[u] if env.get("BOLTZ_PAIRFUSE") else "stock") for u in _gu if u in ("pf", "pfnoseq")}}, "mask_predicate": 0, "primed_missing": 0, "src_sha": {{}}}}
if env.get("BOLTZ_DIT_EXACT") and {pair_reports!r}:   # the DITEXACT adapter's report (boltz2_opt.ditexact.report()): installed on the token transformer, every denoiser call served by the schedule
    _ws = [w for w in env["BOLTZ_DIT_EXACT"].split(",") if w]; _lv = {{"par": "dit_par", "mask": "dit_mask", "sba": "dit_sba", "smx": "dit_smx", "glue": "dit_glue"}}
    LOG["ditexact_report"] = {{"applied": [_lv[w] for w in _ws], "words": _ws, "variant": ",".join(_ws), "tune": {{"n_spath": 1, "prefetch": 1}},
                              "census": {{"installed": 1, "install_refused": {{}}, "calls": 400, "calls_par": 400 if "par" in _ws else 0, "calls_prev": 0, "layers": 9600, "mask_skipped": 400 if "mask" in _ws else 0, "mask_applied": 0 if "mask" in _ws else 400, "sba_fused": 9600 if ("sba" in _ws and "smx" not in _ws) else 0, "sba_torch": 0, "sba_torch_by": {{}}, "smx_fused": 9600 if "smx" in _ws else 0, "smx_bitcmp": ({{"n400": "served"}} if "smx" in _ws else {{}}), "smx_declined_by": {{}}, "glue_fused": 48000 if "glue" in _ws else 0, "glue_bitcmp": ({{"N400": "served"}} if "glue" in _ws else {{}}), "glue_declined_by": {{}}, "cc_refused": {{}}, "nvrtc_compile_s": ({{"smx": 2.1, "glue": 3.4}} if "glue" in _ws else {{}}), "prev_by": {{}}, "capturing_calls": 2}},
                              "gate": {{"ok": True, "idle": False, "reason": None}}, "line": "[boltz2-opt] DITEXACT applied=" + ",".join(_lv[w] for w in _ws), "expected": ["no_hoist_cache"]}}
if env.get("BOLTZ_ATOM") and {pair_reports!r}:        # the atom-attention adapter's report (boltz2_opt.atom.report()): every unit served, no fallback
    _au = [u for u in env["BOLTZ_ATOM"].split(",") if u]; _alv = {{"keys": "atom_keys_gather", "glue": "atom_glue_hoist", "fused": "atom_fused"}}; _gw = env.get("BOLTZ_ATOM_GEMM", "ieee") or "ieee"
    LOG["atom_report"] = {{"applied": [_alv[u] for u in _au] + (["atom_gemm"] if _gw != "ieee" else []), "variant": ",".join(_au), "gemm": _gw, "module": {{"refreshes": 2}},
                          "units": {{u: {{"lever": _alv[u], "census": {{"served": 400, "fallback_by": {{}}, "errors": {{}}, "installed": 1}}}} for u in _au}}, "gate": {{"ok": True, "idle": False, "reason": None}}, "lines": []}}
if env.get("BOLTZ_WASTE") and {pair_reports!r}:       # the WASTE adapter's report (boltz2_opt.waste.report()): every unit on, the hoists acted
    _wu = [u for u in env["BOLTZ_WASTE"].split(",") if u]; _wlv = {{"chunkcast": "waste_chunkcast", "opmmask": "waste_opmmask", "opmdiv": "waste_opmdiv", "ctorskip": "waste_ctorskip"}}
    LOG["waste_report"] = {{"applied": [_wlv[u] for u in _wu], "requested": _wu, "switch": "BOLTZ_WASTE", "lines": [], "patched": ["Transition", "PairWeightedAveraging", "OuterProductMean"],
                           "module": {{"applied": True, "levers": _wu, "state": {{u: {{"state": "on", "reason": "", "patched": ["Transition"] if u == "chunkcast" else (["initialize.trunc_normal_init_"] if u == "ctorskip" else [])}} for u in _wu}}, "over": {{"transition": "boltz2_opt.transition", "pwa": "boltz.model.layers.pair_averaging", "opm": "boltz.model.layers.outer_product_mean"}},
                                       "stats": {{"transition_chunked_calls": 32, "transition_delegated_calls": 900, "transition_cast_hoisted": 32, "pwa_chunked_calls": 16, "pwa_delegated_calls": 0, "pwa_cast_hoisted": 16, "opm_chunked_calls": 16, "opm_delegated_calls": 0, "opm_cast_hoisted": 16, "opm_nummask_computed": 2, "opm_nummask_reused": 14, "opm_div_out": 128, "ctorskip_calls": 472, "ctorskip_elements": 250000000}}, "nummask_entries": 0}},
                           "gate": {{"ok": True, "idle": False, "reason": None}}}}
if env.get("BOLTZ_MSA2") and {pair_reports!r}:        # the MSA-module levers' adapter report (boltz2_opt.msa2.report()): the unit applied, every dim-64 transition call served, the exact unit's self-check classes passed
    _mu = [u for u in env["BOLTZ_MSA2"].split(",") if u]
    LOG["msa2_report"] = {{"applied": _mu, "requested": _mu, "units": {{u: {{"state": "on", "mode": 1 if u == "trans2x" else 0}} for u in _mu}}, "patched": ["Transition.forward"], "lines": [],
                          "trans2_census": {{"calls": 40, "served": (38 if "trans2x" in _mu else 40), "compared": (1 if "trans2x" in _mu else 0), "undetermined": 0, "fallback": (1 if "trans2x" in _mu else 0), "fallback_by": ({{"small_class:4x64:0": 1}} if "trans2x" in _mu else {{}}), "shapes": {{}},
                                            "selftest": ({{"524288x64:32": {{"state": "proven", "sel": None, "candidates_tested": 0, "undetermined": 0, "compared_calls": 1, "compared_elems": 33554432, "ms": 50.0}}, "4x64:0": {{"state": "floor", "sel": None, "candidates_tested": 0, "undetermined": 0, "compared_calls": 0, "compared_elems": 0, "ms": 0.0}}}} if "trans2x" in _mu else {{}}), "floor": {{"rows": 65536}}, "lock": {{"elems": 1000000, "calls": 1}}}}}}
    if "pwa2x" in _mu:
        LOG["msa2_report"]["pwa2_census"] = {{"calls": 64, "served": 63, "compared": 1, "undetermined": 0, "fallback": 0, "fallback_by": {{}}, "classes": {{"400x8192:u": {{"state": "proven", "sel": [2, 4], "candidates_tested": 3, "undetermined": 0, "compared_calls": 1, "compared_elems": 209715200, "ms": 900.0}}}}, "selftest_ms": 900.0, "floor": {{"S": 16, "rows": 65536}}, "lock": {{"elems": 1000000, "calls": 1}}}}
if env.get("BOLTZ_CONF") and {pair_reports!r}:        # the CONF levers' report (boltz2_opt.conf.report(), merged by the attach hook in the real worker): every word applied
    _ws = [w for w in env["BOLTZ_CONF"].split(",") if w]
    LOG["conf_report"] = {{"applied": _ws, "words": _ws, "disabled": {{}}, "line": ["[boltz2-opt] LEVER name=conf." + w + " state=on" for w in _ws], "census": {{w: {{"calls": 8}} for w in _ws}}, "gate": {{"ok": True, "idle": False, "reason": None}}, "switch": "BOLTZ_CONF"}}
if env.get("BOLTZ_PRECISION"):   # the precision units' report (boltz2_opt.precision.report()): every unit of the word served, `off:<unit>` = an ablation entry [PRECISION]
    _pt = [t.strip() for t in env["BOLTZ_PRECISION"].split(",") if t.strip()]; _pon = [t for t in _pt if not t.startswith("off:")]; _poff = [t[4:] for t in _pt if t.startswith("off:")]
    LOG["precision_report"] = {{"applied": list(_pon), "disabled": {{u: "ablation" for u in _poff}}, "variant": env["BOLTZ_PRECISION"], "lines": [], "line": None, "patched": [],
                               "units": {{**{{u: {{"state": "on", "lever": u, "census": {{("attn_calls" if u == "seq_bf16" else "calls"): 400, **({{"flag_restored": 400}} if u == "dit_tf32" else {{}})}}}} for u in _pon}}, **{{u: {{"state": "off", "reason": "ablation", "lever": u}} for u in _poff}}}},
                               "gate": {{"ok": True, "idle": False, "idle_units": [], "reason": None}}, "bundle": {{"tf32_flag_now": False, "matmul_precision": "highest"}}}}
print("[boltz_trunk_levers] APPLIED levers=" + env.get("BOLTZ_LEVERS", ""))
SKIP = set({list(skip_items)!r}); OOM = set({list(oom_items)!r})
from boltz2_opt import skipped as _sk
for it in B["items"]:
    if it["name"] in SKIP:                      # the real worker's process_item: stock's parser skipped the input -> one SKIPPED line, a log entry, nothing predicted
        print("Failed to process " + it["yaml"] + ". Skipping. Error: 'GLY1_'.", flush=True)
        _why = _sk.reason("KeyError: 'GLY1_'"); print(_sk.line(it["name"], _why), flush=True)
        LOG.setdefault("skipped", []).append({{"name": it["name"], "uid": it.get("uid"), "yaml": it["yaml"], "reason": _why}})
k = 0
for it in B["items"]:
    for s in it["seeds"]:
        if it["name"] in SKIP:
            continue
        d = os.path.join(OUT, "by_seed", it["name"], f"s{{s}}"); os.makedirs(d, exist_ok=True)
        k += 1
        if k > len(B["items"]) * len(B["seeds"]) - {fail_items}:
            continue
        if it["name"] in OOM:                   # the real worker's predict_step wrapper saw boltz's {{"exception": True}}: boltz's own WARNING line, a `failed` entry, nothing written
            print("| WARNING: ran out of memory, skipping batch", flush=True)
            LOG.setdefault("failed", []).append({{"name": it["name"], "uid": it.get("uid"), "seed": s, "reason": _sk.oom_reason(66.0)}}); continue
        for f in (f"{{it['name']}}_model_0.cif", f"pae_{{it['name']}}_model_0.npz", f"plddt_{{it['name']}}_model_0.npz", f"confidence_{{it['name']}}_model_0.json"):
            open(os.path.join(d, f), "w").write(f + " " + str(s) + "\\n")
        if {affinity_json!r} and "affinity" in open(it["yaml"]).read():      # an input declaring the affinity property: upstream's affinity leg wrote its json beside the structure
            open(os.path.join(d, f"affinity_{{it['name']}}.json"), "w").write('{{"affinity_pred_value": 0.5}}')
        LOG["per_item"].append({{"name": it["name"], "seed": s, "model_s": 1.0, "graph_sampler_mode": "graph" if {evidence_ok!r} else "stock", "graph_n_replay": 199 if {evidence_ok!r} else 0,
                                "hoist_hoist_level": 2, "hoist_captured_with_cache": 1 if {evidence_ok!r} else 0, "hoist_capture_stock_fallbacks": 0}})
if (env.get("BOLTZ_SAMPLER_ROLLOUT") or env.get("BOLTZ_SAMPLER_DIT") or env.get("BOLTZ_SAMPLER_ALIGN")) and {pair_reports!r}:   # the sampler adapter's report (boltz2_opt.sampler.report()): one capture per prediction, S-1 replays each, nothing out of scope
    _n = len(LOG["per_item"]) if {evidence_ok!r} else 0; _S = int((B.get("settings") or {{}}).get("sampling_steps") or 200)
    _app = (["rollout"] if env.get("BOLTZ_SAMPLER_ROLLOUT") else []) + (["align_" + env["BOLTZ_SAMPLER_ALIGN"]] if env.get("BOLTZ_SAMPLER_ALIGN") else []) + (["dit_fused"] if env.get("BOLTZ_SAMPLER_DIT") else [])
    LOG["sampler_report"] = {{"applied": _app, "levers": {{l: {{"state": "on", "word": {{"rollout": env.get("BOLTZ_SAMPLER_ROLLOUT"), "dit_fused": env.get("BOLTZ_SAMPLER_DIT")}}.get(l, env.get("BOLTZ_SAMPLER_ALIGN"))}} for l in _app}},
                             "words": {{}}, "line": "[boltz2-opt] LEVER name=rollout state=on", "gate": {{"ok": True, "reason": None}}, "patched": ["AtomDiffusion.sample"],
                             "module": {{"stats": {{"calls": _n, "samples": _n, "captures": _n, "replays": _n * (_S - 1), "capture_failed": 0, "capture_s": [1.5] * _n, "kabsch": {{"aligncap": "aligncap", "jacobi64": "device"}}.get(env.get("BOLTZ_SAMPLER_ALIGN") or "", "torch"), "div_recipe": "double", "scope": {{}}, "guard_prints": 0}},
                                         "dit": ({{"stats": {{"gemm": "bf16", "attn": "flash", "layers": 24, "launches_per_layer": 7, "calls": 400 * _n, "served": 400 * _n, "scope": {{}}, "packs": 1, "pack_gib": 0.4, "weights_mib": 380, "out_dtype_probe": "fp32", "mask_fold": True}}}} if env.get("BOLTZ_SAMPLER_DIT") else None)}}}}
os.makedirs(B["kit_dir"], exist_ok=True); json.dump(LOG, open(os.path.join(B["kit_dir"], B["tag"] + "_worker_log.json"), "w"), indent=1)   # stack.worker_log_path's rule
for _it in [x for x in B["items"] if x["name"] not in SKIP]:   # the per-item instrument lines the real worker's phase.py prints in the model process (relayed by pred)
    print(f"PHASE item={{_it['name']}} lm_s=- trunk_s=1.000 sampler_s=2.000 conf_s=0.500 total_s=3.600 cond_s=0.100 fwd_s=3.600", flush=True)
    print(f"[boltz2-opt] PEAK item={{_it['name']}} alloc_gib=5.25 reserved_gib=6.50", flush=True)
KMODE, KWHAT = {mode!r}, {kernels!r}
if KWHAT != "none":                          # the KERNELS census stand-in: the kit's own line grammar and expectations (boltz2_opt.kernels / modes.kernels_expected)
    from boltz2_opt import kernels as _k, modes as _m
    _P = int(env.get("BOLTZ_TP", "1") or 1); _route = _k.route_word(KMODE, _P); _exp = _m.kernels_expected(KMODE, n_gpu=_P)
    _words = {{a: (w if w != "engaged" else "engaged:stub@0.0-cpu[served=1,byrule=0]") for a, w in _exp.items()}}
    _refused = []
    if KWHAT == "refuse":
        _words["cueq_triatt"] = "absent:cuequivariance_ops_torch(ModuleNotFoundError)"; _refused = [{{"accelerator": "cueq_triatt", "word": _words["cueq_triatt"], "expected": "engaged"}}]
        print(f"[boltz2-opt {{KMODE}}] " + _k.REFUSED_MARK + f"{{_route}} settings=defaults cueq_triatt={{_words['cueq_triatt']}} (expected engaged); exit 5", flush=True)
    if KWHAT == "contradiction":
        _a = next((a for a, w in _exp.items() if w.startswith("off-by-route")), "cueq_trimul"); _words[_a] = "engaged:stub@0.0-cpu[served=3,byrule=0]"
        _refused = [{{"accelerator": _a, "word": _words[_a], "expected": _exp[_a]}}]
    _stepped = KWHAT != "nostep" and any(x["name"] not in SKIP for x in B["items"])   # kernels.verdict_word: NO-STEP when no predict_step ran
    _verdict = "NO-STEP" if not _stepped else ("PASS" if not _refused else "REFUSED(" + ",".join(r["accelerator"] for r in _refused) + ")")
    print(_k.format_line(KMODE, _route, "defaults", _words, _verdict, ["trifast=n/a-upstream:no_caller", f"n_gpu={{_P}}"] + ([f"rank={{env['ROWPAIR_RANK']}}"] if env.get("ROWPAIR_RANK") else [])), flush=True)
    json.dump({{"route": _route, "mode": KMODE, "settings": "defaults", "expected": _exp, "words": _words, "verdict": _verdict, "refused": _refused}}, open(os.path.join(B["kit_dir"], B["tag"] + "_kernels_census.json"), "w"), indent=1)
    if KWHAT == "refuse":
        sys.exit(5)
if {f2!r} and env.get("BOLTZ_TRIATTN_REPORT") == "1" and {f2_report!r}:
    S = {f2_stats!r}
    print("[boltz_flash_triattn_patch] " + json.dumps({{"applied": True, "mode": "flash", "min_tokens": int(env.get("BOLTZ_TRIATTN_MIN_TOKENS", "0")), "stats": S,
                                                     "n_flash": S.get("flash_calls", 0), "n_stock": S.get("stock_calls", 0), "errors": [k for k in S if "exception" in k][:1] and ["flash path raised"]}}), flush=True)
sys.exit(0)
'''


def install_fake_stage(monkeypatch, evidence_ok=True, f2=None, fail_items=0, f2_stats=None, f2_report=True, lever_report=None, pair_reports=True, kernels="pass", skip_items=(), affinity_json=True, oom_items=()):
    """Replace stack.stage with one that writes the fake worker; keep the real batch/env/log logic."""
    def fake_stage(mode, workdir):
        os.makedirs(workdir, exist_ok=True)
        w = os.path.join(workdir, "bz_worker.py"); open(w, "w").write(fake_worker_script(evidence_ok, f2, fail_items, f2_stats, f2_report, lever_report, pair_reports, kernels, mode, skip_items, affinity_json, oom_items))
        return {"files": {"bz_worker.py": "fake"}, "worker": w, "variant_line": "fake variant", "variant_sha256": "0" * 64}
    monkeypatch.setattr(stack, "stage", fake_stage)


def gated_ok(monkeypatch, gpu_name="NVIDIA H100 80GB HBM3", cache="/tmp/fake_cache"):
    """A box that passes the gate: a GPU, the pin, the cache."""
    real_gate = stack.gate

    def fake_gate(mode, need_gpu=True, need_cache=True, check_pins=True, n_gpu=1, refresh_weights=False):
        plan = real_gate(mode, need_gpu=False, need_cache=False, check_pins=False, n_gpu=n_gpu)
        plan.update(gpu={"name": gpu_name, "memory_mib": 81559, "driver": "580.95.05"}, pins={"version": "2.2.1", "pinned": True}, cache=cache, reasons=[])
        return plan
    monkeypatch.setattr(stack, "gate", fake_gate)
    monkeypatch.setattr(stack, "cache_dir", lambda: cache)


def write_yamls(d, names=("a", "b")):
    out = []
    for n in names:
        p = os.path.join(d, f"{n}.yaml"); open(p, "w").write("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKT\n"); out.append(p)
    return out


def run_staged_worker_here(monkeypatch):
    """Replace stack.run_worker with a plain subprocess of the staged worker under THIS interpreter and sys.path (no launcher, no pip
    install): worker.run's own batch / env / log / relay / accounting logic runs for real around it."""
    import subprocess

    def fake_run_worker(mode, workdir, batch, log_path, env=None, n_gpu=1, settings_word="defaults", num_workers=None):
        from .. import modes
        argv = [sys.executable, os.path.join(workdir, "bz_worker.py"), "--batch", batch] + [t for k, v in modes.worker_args(mode).items() for t in (f"--{k}", str(v))]
        e = dict(env or os.environ); e["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        with open(log_path, "w") as lf:
            return subprocess.run(argv, env=e, stdout=lf, stderr=subprocess.STDOUT, timeout=300).returncode
    monkeypatch.setattr(stack, "run_worker", fake_run_worker)
