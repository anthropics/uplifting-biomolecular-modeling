"""The ablation switch MODEL_OPT_LEVERS_OFF (ablation.py): names validated by name against the mode's lever set, the arm words leaving the
arm string inside modes.resolve, the memory levers OFF in the big line's selection, the ARMED / ACTIVE `ablated=` token, the ablated
levers' `state=off reason=ablated` LEVER lines, and the refusals (unknown / outside the mode / emptying / --mode off). No GPU, no torch."""
import json
import sys
import types

import pytest

from protenix_v1_opt import ablation as A, big as B, cli, modes as M, report as R
from protenix_v1_opt.tests._ditfast_account import DF_OK as _DF_OK

CPU_OFF = {"recycle_carry": False, "cache_release": False, "expandable_segments": False}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(A.ENV, raising=False)
    B.reset()
    yield
    B.reset()


def _input(tmp_path, n):
    p = tmp_path / f"item{n}.json"
    p.write_text(json.dumps([{"name": f"item{n}", "sequences": [{"proteinChain": {"sequence": "A" * n, "count": 1}}]}]))
    return str(p)


def test_the_words_are_the_sibling_kits():
    assert (A.ENV, A.REASON, A.TOKEN) == ("MODEL_OPT_LEVERS_OFF", "ablated", "ablated") and R.ABLATED == A.REASON


def test_requested_parses_the_variable():
    assert A.requested({}) == [] and A.requested({A.ENV: ""}) == [] and A.requested({A.ENV: " , "}) == []
    assert A.requested({A.ENV: " sg, hoist ,sg"}) == ["sg", "hoist"]                      # stripped, folded, request order


def test_arm_words_and_reduction():
    assert A.arm_words("exact+gblock+xtr+sg+hoist") == ("exact", "gblock", "xtr", "sg", "hoist")
    assert A.arm_words("stock+gblock") == ("gblock",) and A.arm_words("stock") == ()
    assert A.reduce_arm("exact+gblock+xtr+sg+hoist", ["hoist", "sg"]) == "exact+gblock+xtr"
    assert A.reduce_arm("exact+gblock+xtr+sg+hoist", ["exact"]) == "stock+gblock+xtr+sg+hoist"   # the trimul word ablated = the stock triangle multiplication
    assert A.reduce_arm("fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused", ["gflash", "ttr"]) == "fast+tricuda+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused"
    assert A.token([]) == "" and A.token(["hoist", "sg"]) == " ablated=hoist,sg"
    assert A.switches(["recycle_carry", "sg"], B.LINE) == {"recycle_carry": False} and A.split(["sg", "recycle_carry"], B.LINE) == (("sg",), ("recycle_carry",))


def test_unset_is_the_mode_byte_for_byte():
    for m in ("exact", "fast"):
        res = M.resolve(m)
        assert res.arm == res.mode_arm == M.KIT_MODES[m].arm and res.ablated == ()
    rep = {"mode": "fast", "arm": "fast+gflash+ttr+sg+hoist", "cfg": {"trimul": "fast", "gflash": True}, "levers": {}, "gpu": {"name": "H100", "sm": "9.0"}}
    assert "ablated" not in R.activation_line(rep) and "ablated" not in R.armed_line(rep)


def test_resolve_withholds_the_named_arm_words(monkeypatch):
    monkeypatch.setenv(A.ENV, "hoist, sg")
    res = M.resolve("exact")
    assert (res.arm, res.trimul, res.levers, res.ablated, res.mode_arm) == ("exact+gblock+triexact+xtr+keep_pool+summary_hostidx+dit_attn_exact+lazy_init+template_dedupe+sampler_prep+atom_attn_exact+tmpl_triatt+tmpl_xtr+tmpl_trimul_exact", "exact", ("gblock", "triexact", "xtr", "keep_pool", "summary_hostidx", "dit_attn_exact", "lazy_init", "template_dedupe", "sampler_prep", "atom_attn_exact", "tmpl_triatt", "tmpl_xtr", "tmpl_trimul_exact"), ("hoist", "sg"), "exact+gblock+triexact+xtr+sg+hoist+keep_pool+summary_hostidx+dit_attn_exact+lazy_init+template_dedupe+sampler_prep+atom_attn_exact+tmpl_triatt+tmpl_xtr+tmpl_trimul_exact")
    assert res.tier == 1 and res.line == "T1+hoist"                                            # the mode's identity: an ablation is a run of the mode, not another mode
    monkeypatch.setenv(A.ENV, "exact")
    res = M.resolve("exact")
    assert (res.arm, res.trimul, res.levers) == ("stock+gblock+triexact+xtr+sg+hoist+keep_pool+summary_hostidx+dit_attn_exact+lazy_init+template_dedupe+sampler_prep+atom_attn_exact+tmpl_triatt+tmpl_xtr+tmpl_trimul_exact", "stock", ("gblock", "triexact", "xtr", "sg", "hoist", "keep_pool", "summary_hostidx", "dit_attn_exact", "lazy_init", "template_dedupe", "sampler_prep", "atom_attn_exact", "tmpl_triatt", "tmpl_xtr", "tmpl_trimul_exact"))
    monkeypatch.setenv(A.ENV, "gflash")
    res = M.resolve("fast")
    assert res.arm == "fast+tricuda+ttr+sg+hoist+keep_pool+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+sampler_prep+pfattn+opm_fused+pwa_fused+cond_dedupe+dit_fused+dit_lowp+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused" and res.triattn_floor is None and M.resolve("fast", environ={}).triattn_floor == M.TRIATTN_FLOOR   # the floor leaves with the lever


def test_refusals_by_name(monkeypatch):
    monkeypatch.setenv(A.ENV, "nonesuch")
    with pytest.raises(RuntimeError, match=r"MODEL_OPT_LEVERS_OFF=nonesuch: unknown lever name\(s\) \['nonesuch'\]"):
        M.resolve("fast")
    monkeypatch.setenv(A.ENV, "gflash")                                                          # a lever of fast, not of exact
    with pytest.raises(RuntimeError, match=r"\['gflash'\] not a lever of mode exact \(its set: \['exact', 'gblock', 'triexact', 'xtr', 'sg', 'hoist', 'keep_pool', 'summary_hostidx', 'dit_attn_exact', 'lazy_init', 'template_dedupe', 'sampler_prep', 'atom_attn_exact', 'tmpl_triatt', 'tmpl_xtr', 'tmpl_trimul_exact'\]\)"):
        M.resolve("exact")
    monkeypatch.setenv(A.ENV, "recycle_carry")                                                   # a memory lever: big's, refused under fast by name (known, outside the mode)
    with pytest.raises(RuntimeError, match=r"\['recycle_carry'\] not a lever of mode fast"):
        M.resolve("fast")
    monkeypatch.setenv(A.ENV, "fast,gflash,tricuda,ttr,sg,hoist,keep_pool,summary_hostidx,ditattn,ditattnfp16,atomattn,lazy_init,template_dedupe,tmpl_triatt,sampler_prep,pfattn,opm_fused,pwa_fused,cond_dedupe,dit_fused,dit_lowp,atom_fused,tmpl_trimul,tmpl_xtr,tmpl_pairfused")
    with pytest.raises(RuntimeError, match="removes every lever of mode fast: that is `--mode off`, not an ablation"):
        M.resolve("fast")
    with pytest.raises(A.AblationError, match="refused under --mode off"):
        A.validate("off", ["sg"], M.STOCK_ARM)


def test_the_armed_and_active_lines_gain_the_token_at_the_end(monkeypatch):
    rep = {"mode": "exact", "arm": "exact+gblock+xtr", "line": "T1+hoist", "tier": 1, "det": True, "protenix_version": "1.1.0", "package_version": "0",
           "cfg": {"trimul": "exact", "gblock": True, "xtr": True, "sg": False, "hoist": False}, "levers": {"sampler": {"graphs": False, "prep": {"on": False, "aside": "no_sampler_graph"}, "hoist_installed": False, "fastln": {}}},
           "gpu": {"name": "H100", "sm": "9.0"}, "levers_ablated": ["hoist", "sg"]}
    assert R.activation_line(rep).endswith(" gpu='H100' sm=9.0 ablated=hoist,sg")
    assert " package=0 ablated=hoist,sg (the levers apply" in R.armed_line(rep)


def test_the_ablated_levers_lever_lines_read_off_ablated(monkeypatch):
    monkeypatch.setenv(A.ENV, "hoist,sg")
    res = M.resolve("exact")
    rep = {"mode": "exact", "levers_ablated": list(res.ablated),
           "levers": {"cfg": {"trimul": "exact"}, "counts": {"dit": {"apb:fp16": 48}, "pf": {"t2:pf@9.0": 96}, "opm": {"t2:opm@9.0": 8}, "pwa": {"t2:pwa@9.0": 6}, "atom": {"apb:tf32rn": 12}, "trimul": {"exact": 40}, "triattn": {"gblock": 960}, "transition": {"xtr:C=128": 480}},
                      "sampler": {"graphs": False, "prep": {"on": False, "aside": "no_sampler_graph"}, "hoist_installed": False}, "keep_pool": {"installed": True, "skipped_total": 1, "passed_total": 2, "errors": 0}, "summary_hostidx": {"installed": True, "samples": 5, "delegated": 0, "errors": 0}, "lazy_init": {"installed": True, "constructs": 1, "lazy_construct_s": 3.9, "patched": 12}, "ditfast": _DF_OK, "templ": {"template_dedupe": {"installed": True, "calls": 1, "evaluated": 2, "reused": 2, "stock": {}}, "tmpl_triatt": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul_exact": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_xtr": {"on": True, "routed_total": 4, "stock": {}, "fallback": {}}, "tmpl_pairfused": {"on": True, "calls": 8}}, "dit_attn_exact": {"installed": True, "calls": 72, "routes": {"kernel": 72}}, "apb": {"dit": {"engaged": True, "fp16": True, "opd": "fp16", "cell_key": "9.0", "installed_on": 24, "calls": 48}, "atom": {"engaged": True, "opd": "tf32rn", "cell_key": "9.0", "installed_on": 6, "calls": 12}}, "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}},
           "items": [{"N_token": 705}], "det_report": None}
    v = R.verdict(rep, res.trimul, res.levers, allow_partial=False)
    assert v["partial"] == [] and set(v["evidence"]) == {"exact", "gblock", "triexact", "xtr", "keep_pool", "summary_hostidx", "dit_attn_exact", "lazy_init", "template_dedupe", "sampler_prep", "atom_attn_exact", "tmpl_triatt", "tmpl_xtr", "tmpl_trimul_exact"}   # the exit rule judges the kept set
    lines = R.lever_lines(v["evidence"], rep)
    assert [l.rsplit(" lever=", 1)[1] for l in lines] == ["exact", "gblock", "triexact", "xtr", "keep_pool", "summary_hostidx", "dit_attn_exact", "lazy_init", "sampler_prep", "template_dedupe", "tmpl_triatt", "tmpl_trimul_exact", "tmpl_xtr", "atom_attn_exact", "gflash", "tricuda", "ttr", "sg", "hoist", "ditattn", "ditattnfp16", "atomattn", "tmpl_trimul", "tmpl_pairfused", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "memory", "det"]
    assert lines[4] == "[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.keep_pool state=on impl=lib/ptx1_keep_pool.py origin=kit strategy=LOCAL.protenix_v1.keep_pool served=1 gated=2 gated_by=passed:2 lever=keep_pool"
    by = {l.rsplit(" lever=", 1)[1]: l for l in lines}
    assert by["sg"] == "[protenix-v1-opt] LEVER name=F3.cuda_graph_sampler state=off reason=ablated impl=lib/kit112_src/infopt_graphs origin=kit strategy=F3.cuda_graph_sampler lever=sg"
    assert " state=off reason=ablated " in by["hoist"] and " reason=not_in_mode:exact " in by["gflash"]
    monkeypatch.setenv(A.ENV, "exact")                                                           # the trimul word: exact off/ablated, fast off/not_in_mode
    res = M.resolve("exact")
    rep = dict(rep, levers_ablated=list(res.ablated)); rep["levers"] = dict(rep["levers"], cfg={"trimul": "stock"})
    v = R.verdict(rep, res.trimul, res.levers, allow_partial=False)
    by = {l.rsplit(" lever=", 1)[1]: l for l in R.lever_lines(v["evidence"], rep)}
    assert " state=off reason=ablated " in by["exact"] and " reason=not_in_mode:exact " in by["fast"] and "gblock" in v["evidence"]


def test_big_memory_levers_are_off_in_the_lines_selection_by_name(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(A.ENV, "recycle_carry, relp_lean,ttr")
    monkeypatch.setattr(sys, "argv", ["protenix", "pred", "--input", _input(tmp_path, 1948)])                 # above the memory gate (big.MEMORY_GATE_TOKENS), below the TriMul gate: every memory lever but trimul_torch in the line
    monkeypatch.setattr(B, "_module_pinned", lambda lever, mod: None)
    for name in ("_install_relp_lean", "_install_recycle_carry", "_install_diffusion_cond_chunk", "_install_conf_head_chunk", "_install_msa_zfree", "_install_diffcache_free"):
        if hasattr(B, name):                                                                       # the installers reach into the upstream's modules (no protenix in this interpreter): stubbed; the lines are the subject here
            monkeypatch.setattr(B, name, lambda *a: (lambda: None))
    res = M.resolve("big")
    assert res.arm == "fast+gflash+tricuda+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused" and res.levers == ("gflash", "tricuda", "summary_hostidx", "ditattn", "ditattnfp16", "atomattn", "lazy_init", "template_dedupe", "tmpl_triatt", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "atom_fused", "tmpl_trimul", "tmpl_xtr", "tmpl_pairfused") and res.ablated == ("recycle_carry", "relp_lean", "ttr") and res.mode_arm == "fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused"
    B.arm(res, switches={"cache_release": False, "expandable_segments": False})                   # the CPU box's own switches; recycle_carry is off by the ablation
    f = B.fields()
    assert "relp_lean" not in f["levers"] and "recycle_carry" not in f["levers"] and {"recycle_carry", "relp_lean"} <= set(f["off_by_flag"])
    rows = B.lever_rows("big", ablated=res.ablated)
    by = {l.split(" name=", 1)[1].split(" ", 1)[0]: l for l in rows}
    assert by["relp_lean"].startswith("[protenix-v1-opt] LEVER name=relp_lean state=off reason=ablated impl=") and by["relp_lean"].endswith(" origin=kit strategy=" + B.TABLE["relp_lean"]["strategy"])
    assert " state=off reason=ablated " in by["recycle_carry"] and " state=off reason=flag " in by["cache_release"]   # the ablation's word vs the plain switch's
    assert sum(1 for l in rows if " name=relp_lean " in l) == 1 and not any(" name=chunk_pair " in l for l in rows)
    rep = {"mode": "big", "levers_ablated": list(res.ablated), "memory": {"applied": {"infer_setting.chunk_size": 128}}}
    tail = R.lever_lines({}, rep)
    assert any(l.endswith(" lever=ttr") and " reason=ablated " in l for l in tail) and tail[-2].endswith("chunk_size=128 lever=memory")
    B.reset(); monkeypatch.setenv(A.ENV, "chunk_pair")
    res = M.resolve("big"); B.arm(res, switches=dict(CPU_OFF))
    r = types.SimpleNamespace(configs=types.SimpleNamespace(infer_setting=types.SimpleNamespace(chunk_size=256, dynamic_chunk_size=True), triangle_multiplicative="cuequivariance"))
    r.model = types.SimpleNamespace(configs=r.configs); r.predict = lambda data: None
    mem = B.apply(r, res)
    assert mem["applied"] == {} and r.configs.infer_setting.chunk_size == 256                       # the preset is chunk_pair's: withheld with it
    assert R.lever_lines({}, {"mode": "big", "levers_ablated": ["chunk_pair"], "memory": mem})[-2] == "[protenix-v1-opt] LEVER name=F7.chunked_eval state=off reason=ablated impl=infer_setting origin=kit strategy=F7.chunked_eval lever=memory"
    B.reset(); monkeypatch.setenv(A.ENV, "sg")                                                     # off under big at every size: not a lever of the mode
    with pytest.raises(RuntimeError, match=r"\['sg'\] not a lever of mode big"):
        M.resolve("big")


def test_check_and_pred_refuse_under_mode_off_by_name(monkeypatch, capsys):
    monkeypatch.setenv(A.ENV, "sg")
    rep = cli.dry_run_report("off")
    assert rep["ok"] is False and rep["gates"] == [A.off_refusal(["sg"])]
    monkeypatch.setattr(cli.K, "frozen_weights_check", lambda **kw: {})
    assert cli.cmd_pred(["--mode", "off", "--input", "x.json", "--out_dir", "o"]) == R.EXIT_NOT_ACTIVE
    err = capsys.readouterr().err
    assert "[protenix-v1-opt] NOT ACTIVE: MODEL_OPT_LEVERS_OFF=sg refused under --mode off" in err and "[protenix-v1-opt] EXIT mode=off rc=3" in err


def test_check_shows_the_ablation_it_would_run(monkeypatch):
    monkeypatch.setenv(A.ENV, "hoist")
    monkeypatch.setattr(cli, "dry_run_report", cli.dry_run_report)
    from protenix_v1_opt import stack
    monkeypatch.setattr(stack, "gates", lambda res, det, refresh_weights=False: [])
    monkeypatch.setattr(stack, "torch_gpu_info", lambda: {})
    rep = cli.dry_run_report("fast")
    assert rep["arm"] == "fast+gflash+tricuda+ttr+sg+keep_pool+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+sampler_prep+pfattn+opm_fused+pwa_fused+cond_dedupe+dit_fused+dit_lowp+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused" and rep["levers_ablated"] == ["hoist"] and rep["mode_arm"] == "fast+gflash+tricuda+ttr+sg+hoist+keep_pool+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+sampler_prep+pfattn+opm_fused+pwa_fused+cond_dedupe+dit_fused+dit_lowp+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused" and rep["ok"] is True
