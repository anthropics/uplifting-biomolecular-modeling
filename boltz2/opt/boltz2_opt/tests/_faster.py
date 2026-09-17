"""Healthy-run reports of the engine adapters the exact / fast rows attach beyond the pair-track cells (sampler roll-out, DITEXACT, ATen
layer_norm replica, trunk CUDA graphs, PAIRFUSE, atom attention, WASTE, MSA2, CONF), shaped like each adapter's report() — for the evidence
tests that build a worker log by hand (test_cli_and_evidence._log) the way tests/_stubs.py's stand-in worker writes them."""
from typing import Dict


def reports(env: Dict[str, str], n_pred: int, replay: int = 199, healthy: bool = True) -> Dict[str, dict]:
    """{report_key: report} for every adapter word of `env` (a mode's row); n_pred predictions ran, each replayed `replay` boundaries."""
    out: Dict[str, dict] = {}
    gate = {"ok": True, "idle": False, "reason": None}
    if env.get("BOLTZ_SAMPLER_ROLLOUT") or env.get("BOLTZ_SAMPLER_DIT") or env.get("BOLTZ_SAMPLER_ALIGN"):
        app = (["rollout"] if env.get("BOLTZ_SAMPLER_ROLLOUT") else []) + (["align_" + env["BOLTZ_SAMPLER_ALIGN"]] if env.get("BOLTZ_SAMPLER_ALIGN") else []) + (["dit_fused"] if env.get("BOLTZ_SAMPLER_DIT") else [])
        words = {"rollout": env.get("BOLTZ_SAMPLER_ROLLOUT"), "dit_fused": env.get("BOLTZ_SAMPLER_DIT")}
        out["sampler_report"] = {"applied": app, "levers": {l: {"state": "on", "word": words.get(l, env.get("BOLTZ_SAMPLER_ALIGN"))} for l in app}, "words": {}, "line": "", "gate": {"ok": True, "reason": None},
                                 "patched": ["AtomDiffusion.sample"],
                                 "module": {"stats": {"calls": n_pred, "samples": n_pred, "captures": n_pred, "replays": n_pred * replay, "capture_failed": 0, "capture_s": [1.5] * n_pred, "kabsch": {"aligncap": "aligncap", "jacobi64": "device"}.get(env.get("BOLTZ_SAMPLER_ALIGN") or "", "torch"),
                                                      "div_recipe": "double", "scope": {}, "guard_prints": 0},
                                            "dit": ({"stats": {"gemm": "bf16", "attn": "flash", "layers": 24, "launches_per_layer": 7, "calls": 400 * n_pred, "served": 400 * n_pred, "scope": {}, "packs": 1,
                                                               "pack_gib": 0.4, "weights_mib": 380, "out_dtype_probe": "fp32", "mask_fold": True}} if env.get("BOLTZ_SAMPLER_DIT") else None)}}
    if env.get("BOLTZ_DIT_EXACT"):
        ws = [w for w in env["BOLTZ_DIT_EXACT"].split(",") if w]; lv = {"par": "dit_par", "mask": "dit_mask", "sba": "dit_sba", "smx": "dit_smx", "glue": "dit_glue"}; calls = 400 * n_pred
        out["ditexact_report"] = {"applied": [lv[w] for w in ws], "words": ws, "variant": ",".join(ws), "tune": {"n_spath": 1, "prefetch": 1},
                                  "census": {"installed": 1, "install_refused": {}, "calls": calls, "calls_par": calls if "par" in ws else 0, "calls_prev": 0, "layers": 24 * calls,
                                             "mask_skipped": calls if "mask" in ws else 0, "mask_applied": 0 if "mask" in ws else calls, "sba_fused": 24 * calls if ("sba" in ws and "smx" not in ws) else 0,
                                             "sba_torch": 0, "sba_torch_by": {}, "smx_fused": 24 * calls if "smx" in ws else 0, "smx_bitcmp": ({"n400": "served"} if "smx" in ws else {}), "smx_declined_by": {}, "glue_fused": (120 * calls if "glue" in ws else 0), "glue_bitcmp": ({"N400": "served"} if "glue" in ws else {}), "glue_declined_by": {}, "cc_refused": {}, "nvrtc_compile_s": ({"smx": 2.1, "glue": 3.4} if "glue" in ws else {}),
                                             "prev_by": {}, "capturing_calls": 2},
                                  "gate": dict(gate), "line": "", "expected": ["no_hoist_cache"]}
    if env.get("BOLTZ_TEMPL_SKIP"):                                  # the dummy-template elision's report (boltz2_opt.templskip.report()): every untemplated pass skipped
        out["templskip_report"] = {"applied": ["templ_skip"], "variant": env["BOLTZ_TEMPL_SKIP"], "disabled": {}, "line": "", "census": {"calls": 4 * n_pred, "skipped": 4 * n_pred, "live": 0, "capturing": 0, "no_mask": 0, "errors": 0, "served": 4 * n_pred, "shapes": {}},
                                   "gate": {"ok": True, "idle": False, "reason": None}, "patched": ["TemplateV2Module.forward"]}
    if env.get("BOLTZ_TRIATTN_EXACT"):                              # the exact word binding of the block's library core (boltz2_opt.triattn_exact.report()): every call served through the provider, the library op named its row
        out["triattn_exact_report"] = {"applied": ["triattn_exact"], "variant": env["BOLTZ_TRIATTN_EXACT"], "disabled": {}, "line": "",
                                       "census": {"calls": 96 * n_pred, "served": 96 * n_pred, "fallback": {}, "errors": {}, "rows": {"cueq": 96 * n_pred}, "kernel": {"served": 0, "calls": 0, "refused": {}}},
                                       "facts": {"word": "exact", "row": "cueq", "cc": "9.0", "stack": "9.0|torch2.12.0+cu130|cueq0.10.0", "at": "bf16xD32xH4xN512"},
                                       "evidence": "triattn_exact: served 0/0 calls (refused: {})", "gate": dict(gate)}
    if env.get("BOLTZ_EXACTLN"):
        out["exactln_report"] = {"applied": ["exactln"], "variant": env["BOLTZ_EXACTLN"], "disabled": {}, "line": "", "census": {"served": {"C128:float32>bfloat16:contig": 900}, "fallback": {"below_min_numel": 40}, "errors": {}},
                                 "gate": dict(gate)}
    if env.get("BOLTZ_PAIRFUSE"):
        out["pairfuse_report"] = {"applied": ["pairfuse"], "variant": env["BOLTZ_PAIRFUSE"], "residency": env["BOLTZ_PAIRFUSE"], "disabled": {}, "line": "",
                                  "census": {"served": {"pairformer": 64 * 4 * n_pred, "noseq": 0, "msa_pair": 4 * 4 * n_pred}, "fallback": {"c_z:64": 8 * n_pred}, "errors": {}, "sites": {}},
                                  "providers": "trimul=v4,attn=" + (env.get("BOLTZ_PAIRBLOCK") or "cueq") + ",transition=pair_fused", "gate": dict(gate)}
    if env.get("BOLTZ_GRAPH_TRUNK", "").strip().lower() not in ("", "0", "off", "none"):
        gu = [u for u in env["BOLTZ_GRAPH_TRUNK"].split(",") if u]
        out["graph_report"] = {"applied": ["graph_trunk"], "units": gu, "disabled": {}, "line": "", "census": {u: ({"replayed": 10, "captured": 1, "eager:warm": 1} if u != "templ" else {"boundaries_served": 4, "boundaries_built": 1}) for u in gu},
                               "gate": dict(gate), "patched": gu, "captures": [], "capture_s_total": 0.3, "pool_gib_max": 1.0, "static_gib": 0.1, "generations_evicted": 0, "resident": [], "errors": {},
                               "min_tokens": int(env.get("BOLTZ_GRAPH_TRUNK_MIN_TOKENS", "0") or 0), "max_tokens": int(env.get("BOLTZ_GRAPH_TRUNK_MAX_TOKENS", "0") or 0), "keep": 1, "switch": "BOLTZ_GRAPH_TRUNK", "spec": env["BOLTZ_GRAPH_TRUNK"],
                               "bodies": {u: ({"pf": "pairfuse.pfm_forward", "pfnoseq": "pairfuse.pfnm_forward"}[u] if env.get("BOLTZ_PAIRFUSE") else "stock") for u in gu if u in ("pf", "pfnoseq")}, "mask_predicate": 0, "primed_missing": 0, "src_sha": {}}
    if env.get("BOLTZ_ATOM"):
        au = [u for u in env["BOLTZ_ATOM"].split(",") if u]; alv = {"keys": "atom_keys_gather", "glue": "atom_glue_hoist", "fused": "atom_fused"}; gw = env.get("BOLTZ_ATOM_GEMM", "ieee") or "ieee"
        out["atom_report"] = {"applied": [alv[u] for u in au] + (["atom_gemm"] if gw != "ieee" else []), "variant": ",".join(au), "gemm": gw, "module": {"refreshes": 2},
                              "units": {u: {"lever": alv[u], "census": {"served": 400 * n_pred, "fallback_by": {}, "errors": {}, "installed": 1}} for u in au}, "gate": dict(gate), "lines": []}
    if env.get("BOLTZ_WASTE"):
        wu = [u for u in env["BOLTZ_WASTE"].split(",") if u]; wlv = {"chunkcast": "waste_chunkcast", "opmmask": "waste_opmmask", "opmdiv": "waste_opmdiv", "ctorskip": "waste_ctorskip"}
        out["waste_report"] = {"applied": [wlv[u] for u in wu], "requested": wu, "switch": "BOLTZ_WASTE", "lines": [], "patched": ["Transition", "PairWeightedAveraging", "OuterProductMean"],
                               "module": {"applied": True, "levers": wu, "state": {u: {"state": "on", "reason": "", "patched": []} for u in wu}, "over": {},
                                          "stats": {"transition_chunked_calls": 32, "transition_delegated_calls": 900, "transition_cast_hoisted": 32, "pwa_chunked_calls": 16, "pwa_delegated_calls": 0, "pwa_cast_hoisted": 16,
                                                    "opm_chunked_calls": 16, "opm_delegated_calls": 0, "opm_cast_hoisted": 16, "opm_nummask_computed": 2, "opm_nummask_reused": 14, "opm_div_out": 128,
                                                    "ctorskip_calls": 472, "ctorskip_elements": 250000000}, "nummask_entries": 0},
                               "gate": dict(gate)}
    if env.get("BOLTZ_MSA2"):
        mu = [u for u in env["BOLTZ_MSA2"].split(",") if u]
        out["msa2_report"] = {"applied": mu, "requested": mu, "units": {u: {"state": "on", "mode": 1 if u == "trans2x" else 0} for u in mu}, "patched": ["Transition.forward"], "lines": [],
                              "trans2_census": {"calls": 40, "served": (38 if "trans2x" in mu else 40), "compared": (1 if "trans2x" in mu else 0), "undetermined": 0, "fallback": (1 if "trans2x" in mu else 0), "fallback_by": ({"small_class:4x64:0": 1} if "trans2x" in mu else {}), "shapes": {},
                                                "selftest": ({"524288x64:32": {"state": "proven", "sel": None, "candidates_tested": 0, "undetermined": 0, "compared_calls": 1, "compared_elems": 33554432, "ms": 50.0}, "4x64:0": {"state": "floor", "sel": None, "candidates_tested": 0, "undetermined": 0, "compared_calls": 0, "compared_elems": 0, "ms": 0.0}} if "trans2x" in mu else {}), "floor": {"rows": 65536}, "lock": {"elems": 1000000, "calls": 1}}}
        if "pwa2x" in mu:
            out["msa2_report"]["pwa2_census"] = {"calls": 64, "served": 63, "compared": 1, "undetermined": 0, "fallback": 0, "fallback_by": {}, "classes": {"400x8192:u": {"state": "proven", "sel": [2, 4], "candidates_tested": 3, "undetermined": 0, "compared_calls": 1, "compared_elems": 209715200, "ms": 900.0}}, "selftest_ms": 900.0, "floor": {"S": 16, "rows": 65536}, "lock": {"elems": 1000000, "calls": 1}}
    if env.get("BOLTZ_CONF"):
        cw = [w for w in env["BOLTZ_CONF"].split(",") if w]
        out["conf_report"] = {"applied": cw, "words": cw, "disabled": {}, "line": [], "census": {w: {"calls": 8} for w in cw}, "gate": dict(gate), "switch": "BOLTZ_CONF"}
    if env.get("BOLTZ_PRECISION"):                                   # the precision units' report (boltz2_opt.precision.report()): every unit of the word served; `off:<unit>` = an ablation entry [PRECISION]
        pt = [t.strip() for t in env["BOLTZ_PRECISION"].split(",") if t.strip()]; pon = [t for t in pt if not t.startswith("off:")]; poff = [t[4:] for t in pt if t.startswith("off:")]
        out["precision_report"] = {"applied": list(pon), "disabled": {u: "ablation" for u in poff}, "variant": env["BOLTZ_PRECISION"], "lines": [], "line": None, "patched": [],
                                   "units": {**{u: {"state": "on", "lever": u, "census": {("attn_calls" if u == "seq_bf16" else "calls"): 2 * n_pred, **({"flag_restored": 2 * n_pred} if u == "dit_tf32" else {})}} for u in pon},
                                             **{u: {"state": "off", "reason": "ablation", "lever": u} for u in poff}},
                                   "gate": dict(gate, idle_units=[]), "bundle": {"tf32_flag_now": False, "matmul_precision": "highest"}}
    return out
