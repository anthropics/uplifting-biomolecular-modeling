"""One LEVER line per lever per process in the shared per-lever grammar and key order (opt_core.report prefix + kv):
name, state, [reason], impl, origin, [served, fallback, fallback_by, gated, gated_by, retried, retried_by, min_tokens|floor], extras, lever=<kit word> last;
reasons are single tokens; no blank inside any value."""
import re
import pytest

from protenix_v1_opt import modes, report as R
from protenix_v1_opt.tests._ditfast_account import DF_OK as _DF_OK

FAST = modes.resolve("fast"); EXACT = modes.resolve("exact")


def _fast_account():
    return {"cfg": {"trimul": "fast"}, "counts": {"dit": {"apb:fp16": 48}, "pf": {"t2:pf@9.0": 96}, "opm": {"t2:opm@9.0": 8}, "pwa": {"t2:pwa@9.0": 6}, "atom": {"apb:tf32rn": 12}, "trimul": {"fast": 40}, "triattn": {"gflash": 0, "stock:gate": 96}, "transition": {"ttr:C=128": 40}},
            "keep_pool": {"installed": True, "skipped_total": 3, "passed_total": 1, "errors": 0}, "summary_hostidx": {"installed": True, "samples": 5, "delegated": 0, "errors": 0}, "lazy_init": {"installed": True, "constructs": 1, "lazy_construct_s": 3.9, "patched": 12}, "ditfast": _DF_OK, "templ": {"template_dedupe": {"installed": True, "calls": 1, "evaluated": 2, "reused": 2, "stock": {}}, "tmpl_triatt": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul_exact": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_xtr": {"on": True, "routed_total": 4, "stock": {}, "fallback": {}}, "tmpl_pairfused": {"on": True, "calls": 8}}, "dit_attn_exact": {"installed": True, "calls": 72, "routes": {"kernel": 72}}, "apb": {"dit": {"engaged": True, "fp16": True, "opd": "fp16", "cell_key": "9.0", "installed_on": 24, "calls": 48}, "atom": {"engaged": True, "opd": "tf32rn", "cell_key": "9.0", "installed_on": 6, "calls": 12}}, "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}, "sampler": {"graphs": True, "prep": {"on": True, "parts": "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec", "poison": "ok", "stats": {}, "aside": None}, "hoist_installed": True, "sampler": {"replays": 100}, "hoist": {"hits": 10, "records": 1}}}


def _keys(line):
    return [kv.split("=", 1)[0] for kv in line.split(" LEVER ", 1)[1].split(" ")]


def test_grammar_order_and_tokens():
    line = R.lever_line("gflash", "on", served=3)
    assert line == "[protenix-v1-opt] LEVER name=F1.flash_triatt state=on impl=opt_core/attn/pair_fused.py origin=core strategy=F1.flash_triatt served=3 lever=gflash"
    assert re.fullmatch('\\[protenix\\-v1\\-opt\\]\\ LEVER\\ name=F2\\.fpf_trimul_fast\\ state=on\\ impl=opt_core/kernels/\\w+\\ origin=core\\ strategy=F2\\.fpf_trimul_fast\\ lever=fast', R.lever_line("fast", "on")), R.lever_line("fast", "on")   # impl: class-tolerant (the shared core's TriMul provider path)
    assert " why=a_b_c " in R.lever_line("sg", "skipped", "x y", why="a b c")           # no blank survives inside a value
    with pytest.raises(ValueError):
        R.lever_line("sg", "skipped")
    with pytest.raises(ValueError):
        R.lever_line("sg", "live")
    assert set(R.STRATEGY_IDS) >= set(modes.LEVERS) | {"exact", "fast", "memory", "det"}


def test_fast_floor_off_run_prints_one_line_per_lever_in_key_order():
    rep = {"mode": "fast", "levers": _fast_account(), "items": [{"N_token": 185}], "triattn_floor": {"exported": 300},
           "det_report": None, "memory": None}
    v = R.verdict(rep, "fast", FAST.levers, allow_partial=False)
    lines = R.lever_lines(v["evidence"], rep)
    assert [l.rsplit(" lever=", 1)[1] for l in lines] == ["fast", "gflash", "tricuda", "ttr", "sg", "hoist", "keep_pool", "summary_hostidx", "ditattn", "ditattnfp16", "atomattn", "lazy_init", "sampler_prep", "template_dedupe", "tmpl_triatt", "tmpl_trimul", "tmpl_xtr", "tmpl_pairfused", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "gblock", "triexact", "xtr", "dit_attn_exact", "atom_attn_exact", "tmpl_trimul_exact", "memory", "det"]   # every lever once: the mode's, then the levers it does not compose (off), then the package rows
    for l in lines:
        ks = _keys(l)
        assert ks[:2] == ["name", "state"] and ks[-1] == "lever" and "impl" in ks and ks[ks.index("impl") + 1 : ks.index("impl") + 3] == ["origin", "strategy"]
        assert all(" " not in kv.split("=", 1)[1] for kv in l.split(" LEVER ", 1)[1].split(" "))
    assert lines[1] == "[protenix-v1-opt] LEVER name=F1.flash_triatt state=on impl=opt_core/attn/pair_fused.py origin=core strategy=F1.flash_triatt served=0 gated=96 gated_by=stock:gate:96 min_tokens=300 floor=floor-off lever=gflash"
    assert lines[2] == "[protenix-v1-opt] LEVER name=F1.flash_triatt state=skipped reason=aside impl=opt_core/kernels/triattn origin=core strategy=F1.flash_triatt served=0 row=none host=gflash aside=no_core_call tier=fast lever=tricuda"   # gflash made no core call: the word served nothing by design (exit 0, never partial)
    assert lines[6] == "[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.keep_pool state=on impl=lib/ptx1_keep_pool.py origin=kit strategy=LOCAL.protenix_v1.keep_pool served=3 gated=1 gated_by=passed:1 lever=keep_pool"
    assert lines[7] == "[protenix-v1-opt] LEVER name=F6.host_sync_elimination state=on impl=lib/ptx1_summary_host.py origin=kit strategy=F6.host_sync_elimination served=5 lever=summary_hostidx"
    assert lines[8] == "[protenix-v1-opt] LEVER name=F5.flash_attn_dense state=on impl=ptxfpf/apb_ptx1.py origin=kit strategy=F5.flash_attn_dense served=48 cell=fp16@9.0 lever=ditattn"
    assert lines[11] == "[protenix-v1-opt] LEVER name=F6.weights_residency_init state=on impl=lib/ptx1_lazy_init.py origin=kit strategy=F6.weights_residency_init served=1 construct_s=3.9 patched=12 lever=lazy_init"
    assert lines[13] == "[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.template_dedupe state=on impl=ptxfpf/ptx1_templ.py origin=kit strategy=LOCAL.protenix_v1.template_dedupe served=2 gated=2 gated_by=reused:2 lever=template_dedupe"
    assert lines[14] == "[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.tmpl_triatt state=on impl=ptxfpf/ptx1_templ.py origin=kit strategy=LOCAL.protenix_v1.tmpl_triatt served=8 lever=tmpl_triatt"
    assert lines[18] == "[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.pfattn state=on impl=ptxfpf/trunk2_ptx1.py origin=kit strategy=LOCAL.protenix_v1.pfattn served=96 cell=pf@9.0 lever=pfattn"
    assert lines[19] == "[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.opm_fused state=on impl=ptxfpf/trunk2_ptx1.py origin=kit strategy=LOCAL.protenix_v1.opm_fused served=8 cell=opm@9.0 lever=opm_fused"
    assert lines[20] == "[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.pwa_fused state=on impl=ptxfpf/trunk2_ptx1.py origin=kit strategy=LOCAL.protenix_v1.pwa_fused served=6 cell=pwa@9.0 lever=pwa_fused"
    assert lines[25] == "[protenix-v1-opt] LEVER name=F1.triatt_lean_exact state=off reason=not_in_mode:fast impl=opt_core/attn/pair_fused.py origin=core strategy=F1.triatt_lean_exact lever=gblock"
    assert lines[26] == "[protenix-v1-opt] LEVER name=F1.flash_triatt state=off reason=not_in_mode:fast impl=opt_core/kernels/triattn origin=core strategy=F1.flash_triatt lever=triexact"
    assert lines[28] == "[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.dit_attn_exact state=off reason=not_in_mode:fast impl=opt_core/kernels/apb/dit_exact origin=core strategy=LOCAL.protenix_v1.dit_attn_exact lever=dit_attn_exact"
    assert lines[15] == "[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.tmpl_trimul state=on impl=ptxfpf/ptx1_templ.py origin=kit strategy=LOCAL.protenix_v1.tmpl_trimul served=8 lever=tmpl_trimul"
    assert lines[31] == "[protenix-v1-opt] LEVER name=F7.chunked_eval state=off reason=not_in_mode:fast impl=infer_setting origin=kit strategy=F7.chunked_eval lever=memory"
    assert lines[32] == "[protenix-v1-opt] LEVER name=F4.det_recipe state=off reason=det0 impl=det.py origin=kit strategy=F4.det_recipe lever=det"


def test_a_fallback_is_skipped_with_a_one_token_reason_and_its_counters():
    acct = _fast_account(); acct["counts"]["trimul"]["error:RuntimeError"] = 2
    rep = {"mode": "fast", "levers": acct, "items": [{"N_token": 185}], "triattn_floor": {"exported": 300}}
    v = R.verdict(rep, "fast", FAST.levers, allow_partial=False)
    assert re.fullmatch('\\[protenix\\-v1\\-opt\\]\\ LEVER\\ name=F2\\.fpf_trimul_fast\\ state=skipped\\ reason=fallback\\ impl=opt_core/kernels/\\w+\\ origin=core\\ strategy=F2\\.fpf_trimul_fast\\ served=40\\ fallback=2\\ fallback_by=error:RuntimeError:2\\ lever=fast', R.lever_lines(v["evidence"], rep)[0]), R.lever_lines(v["evidence"], rep)[0]   # impl: class-tolerant
    v = R.verdict({"mode": "fast", "levers": {"error": "boom went the import"}, "items": []}, "fast", FAST.levers, allow_partial=False)
    first = R.lever_lines(v["evidence"], {"mode": "fast"})[0]
    assert first.startswith("[protenix-v1-opt] LEVER name=F2.fpf_trimul_fast state=skipped reason=fallback ") and "fallback_by=account:boom_went_the_import" in first


def test_exact_and_big_rows():
    rep = {"mode": "exact", "levers": {"cfg": {"trimul": "exact"}, "counts": {"dit": {"apb:fp16": 48}, "pf": {"t2:pf@9.0": 96}, "opm": {"t2:opm@9.0": 8}, "pwa": {"t2:pwa@9.0": 6}, "atom": {"apb:tf32rn": 12}, "trimul": {"exact": 40}, "triattn": {"gblock": 960}, "transition": {"xtr:C=128": 480, "stock:C=384": 22}},
           "trimul_buffers": {"state": "on", "released": 3, "calls": 4},
           "keep_pool": {"installed": True, "skipped_total": 3, "passed_total": 1, "errors": 0}, "summary_hostidx": {"installed": True, "samples": 5, "delegated": 0, "errors": 0}, "lazy_init": {"installed": True, "constructs": 1, "lazy_construct_s": 3.9, "patched": 12}, "ditfast": _DF_OK, "templ": {"template_dedupe": {"installed": True, "calls": 1, "evaluated": 2, "reused": 2, "stock": {}}, "tmpl_triatt": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul_exact": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_xtr": {"on": True, "routed_total": 4, "stock": {}, "fallback": {}}, "tmpl_pairfused": {"on": True, "calls": 8}}, "dit_attn_exact": {"installed": True, "calls": 72, "routes": {"kernel": 72}}, "apb": {"dit": {"engaged": True, "fp16": True, "opd": "fp16", "cell_key": "9.0", "installed_on": 24, "calls": 48}, "atom": {"engaged": True, "opd": "tf32rn", "cell_key": "9.0", "installed_on": 6, "calls": 12}}, "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}, "sampler": {"graphs": True, "prep": {"on": True, "parts": "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec", "poison": "ok", "stats": {}, "aside": None}, "hoist_installed": True, "sampler": {"replays": 9}, "hoist": {"hits": 1}}}, "items": [{"N_token": 705}],
           "det_report": {"det_scatter_active": True, "warn_only": True}}
    v = R.verdict(rep, "exact", EXACT.levers, allow_partial=False)
    lines = R.lever_lines(v["evidence"], rep)
    assert [l.rsplit(" lever=", 1)[1] for l in lines] == ["exact", "gblock", "triexact", "xtr", "sg", "hoist", "keep_pool", "summary_hostidx", "dit_attn_exact", "lazy_init", "sampler_prep", "template_dedupe", "tmpl_triatt", "tmpl_trimul_exact", "tmpl_xtr", "atom_attn_exact", "gflash", "tricuda", "ttr", "ditattn", "ditattnfp16", "atomattn", "tmpl_trimul", "tmpl_pairfused", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "memory", "det"]
    assert re.fullmatch('\\[protenix\\-v1\\-opt\\]\\ LEVER\\ name=F2\\.fpf_trimul_exact\\ state=on\\ impl=opt_core/kernels/\\w+\\ origin=core\\ strategy=F2\\.fpf_trimul_exact\\ served=40\\ buf_release=on\\ buf_released=3\\ lever=exact', lines[0]), lines[0]   # impl: class-tolerant (the shared core's TriMul provider path)
    assert lines[1] == "[protenix-v1-opt] LEVER name=F1.triatt_lean_exact state=on impl=opt_core/attn/pair_fused.py origin=core strategy=F1.triatt_lean_exact served=960 lever=gblock"
    assert lines[2] == "[protenix-v1-opt] LEVER name=F1.flash_triatt state=skipped reason=aside impl=opt_core/kernels/triattn origin=core strategy=F1.flash_triatt served=0 row=none host=gblock aside=no_core_call tier=exact lever=triexact"   # no provider census in this account: gblock made no core call through the word (by design, exit 0)
    assert lines[3] == "[protenix-v1-opt] LEVER name=LOCAL.fused_transition state=on impl=opt_core/kernels/transition origin=core strategy=LOCAL.fused_transition served=480 gated=22 gated_by=stock:C=384:22 lever=xtr"
    assert lines[32] == "[protenix-v1-opt] LEVER name=F4.det_recipe state=on impl=det.py origin=kit strategy=F4.det_recipe det=1 det_scatter=1 warn_only=1 lever=det"
    mem = R.lever_lines({}, {"mode": "big", "memory": {"applied": {"infer_setting.chunk_size": 128, "infer_setting.dynamic_chunk_size": False}}})
    assert mem[-2] == "[protenix-v1-opt] LEVER name=F7.chunked_eval state=on impl=infer_setting origin=kit strategy=F7.chunked_eval chunk_size=128 dynamic_chunk_size=False lever=memory"



def test_every_strategy_id_is_canonical_in_the_core_vocabulary():
    """report.STRATEGY_IDS are canonical ids of opt_core/STRATEGIES.json (never an alias): the core's lever_line refuses anything else."""
    import json, os
    import opt_core
    sj = json.load(open(os.path.join(os.path.dirname(opt_core.__file__), "STRATEGIES.json"), encoding="utf-8"))
    canonical = {c["id"] for c in sj["canonical"]}
    local = {v for v in R.STRATEGY_IDS.values() if v.startswith("LOCAL.protenix_v1.")}       # the core's own kit-local form (opt_core.report.check_strategy: LOCAL.<kit>.<name>), this kit's name only
    from opt_core import strategies as CORE_R
    assert all(CORE_R.check_strategy(v) == v for v in local), sorted(local)
    assert set(R.STRATEGY_IDS.values()) - local <= canonical, sorted(set(R.STRATEGY_IDS.values()) - local - canonical)
    assert not set(R.STRATEGY_IDS.values()) & set(sj["aliases"]), sorted(set(R.STRATEGY_IDS.values()) & set(sj["aliases"]))


def test_a_core_refusal_is_a_fallback_for_the_fused_block_and_transition():
    """`fallback:<lever>:<reason>` (opt_core.attn.pair_fused refused the call before any launch -> stock for that call) is a FALLBACK: the
    lever line is `skipped reason=fallback` with the counter, the run partial."""
    rep = {"mode": "exact", "levers": {"cfg": {"trimul": "exact"}, "counts": {"dit": {"apb:fp16": 48}, "pf": {"t2:pf@9.0": 96}, "opm": {"t2:opm@9.0": 8}, "pwa": {"t2:pwa@9.0": 6}, "atom": {"apb:tf32rn": 12}, "trimul": {"exact": 40}, "triattn": {"gblock": 900, "fallback:gblock:no-cell:fpf:prologue:128x4x32:9.0|3.3": 60},
                                        "transition": {"xtr:C=128": 480}},
                                        "keep_pool": {"installed": True, "skipped_total": 3, "passed_total": 1, "errors": 0}, "summary_hostidx": {"installed": True, "samples": 5, "delegated": 0, "errors": 0}, "lazy_init": {"installed": True, "constructs": 1, "lazy_construct_s": 3.9, "patched": 12}, "ditfast": _DF_OK, "templ": {"template_dedupe": {"installed": True, "calls": 1, "evaluated": 2, "reused": 2, "stock": {}}, "tmpl_triatt": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul_exact": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_xtr": {"on": True, "routed_total": 4, "stock": {}, "fallback": {}}, "tmpl_pairfused": {"on": True, "calls": 8}}, "dit_attn_exact": {"installed": True, "calls": 72, "routes": {"kernel": 72}}, "apb": {"dit": {"engaged": True, "fp16": True, "opd": "fp16", "cell_key": "9.0", "installed_on": 24, "calls": 48}, "atom": {"engaged": True, "opd": "tf32rn", "cell_key": "9.0", "installed_on": 6, "calls": 12}}, "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}, "sampler": {"graphs": True, "prep": {"on": True, "parts": "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec", "poison": "ok", "stats": {}, "aside": None}, "hoist_installed": True, "sampler": {"replays": 9}, "hoist": {"hits": 1}}}, "items": [{"N_token": 705}]}
    v = R.verdict(rep, "exact", EXACT.levers, allow_partial=False)
    assert "gblock" in (v["partial"] or []) and v["exit_code"] == R.EXIT_NOT_ACTIVE
    line = R.lever_lines(v["evidence"], rep)[1]
    assert line.startswith("[protenix-v1-opt] LEVER name=F1.triatt_lean_exact state=skipped reason=fallback ") and "fallback=60" in line and line.endswith(" lever=gblock")
