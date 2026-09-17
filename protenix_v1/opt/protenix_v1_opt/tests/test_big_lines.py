"""The memory line's printed lines are pinned byte for byte: the BIG census line and every LEVER row at P=1 — above the TriMul size gate
(2956), between the memory gate and the TriMul gate (1948: the lines the line has always printed below 2048) and at or below the memory gate
(948: since kit 0.2.44 the same lines as 1948 — the memory gate is 0) — and at n_gpu 2 (the four superseded levers off BY NAME through the in-process
selection) equal the pinned lines below; the census exit lines name the opt-out of the route — `--allow-partial` on the verbs,
`PROTENIX_V1_OPT_ALLOW_PARTIAL=1` on the environment route."""
import json
import os
import sys

import pytest

from protenix_v1_opt import big as B, modes as M, ngpu as NGPU

CPU_OFF = {"recycle_carry": False, "cache_release": False, "expandable_segments": False}     # the CUDA levers, off by name on a CPU box (their rows read state=off reason=flag)

EXPECTED = {
 "p1_2956": {
  "big_line": "[protenix-v1-opt] BIG line=big tokens=2956 source=estimate base=fast arm=fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused levers=chunk_pair,drop_bond_mask,relp_lean,diffusion_cond_chunk,conf_head_chunk,msa_zfree,diffcache_free,trimul_torch exact=measured refused=none off=cache_release,expandable_segments,recycle_carry on=none property_off=fast,gflash,hoist,keep_pool,sg allocator=none",
  "rows": [
   "[protenix-v1-opt] LEVER name=drop_bond_mask state=on impl=protenix_v1_opt.big origin=kit strategy=LOCAL.protenix_v1.drop_bond_mask exact=bitwise scope=unit key=bond_mask sites=InferenceRunner.predict",
   "[protenix-v1-opt] LEVER name=relp_lean state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=bitwise scope=unit rows=512 sites=RelativePositionEncoding.generate_relp",
   "[protenix-v1-opt] LEVER name=diffusion_cond_chunk state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=measured scope=unit rows=256 sites=DiffusionConditioning.prepare_cache",
   "[protenix-v1-opt] LEVER name=conf_head_chunk state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=measured scope=unit rows=256 sites=ConfidenceHead.memory_efficient_forward",
   "[protenix-v1-opt] LEVER name=msa_zfree state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=bitwise scope=unit sites=MSABlock.forward,MSAModule.forward",
   "[protenix-v1-opt] LEVER name=diffcache_free state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=bitwise scope=unit sites=DiffusionConditioning.prepare_cache,AtomAttentionEncoder.prepare_cache,Protenix.run_confidence_head",
   "[protenix-v1-opt] LEVER name=trimul_torch state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=measured scope=unit triangle_multiplicative=torch sites=InferenceRunner.configs.triangle_multiplicative,Protenix.configs_(the_same_ConfigDict,_protenix.py:98)",
   "[protenix-v1-opt] LEVER name=cache_release state=off reason=flag impl=opt_core.mem.allocator origin=core strategy=F7.cache_release",
   "[protenix-v1-opt] LEVER name=expandable_segments state=off reason=flag impl=opt_core.mem.allocator origin=core strategy=F7.expandable_segments",
   "[protenix-v1-opt] LEVER name=recycle_carry state=off reason=flag impl=opt_core.mem.ckpt origin=core strategy=F7.recycle_carry"
  ]
 },
 "p1_1948": {
  "big_line": "[protenix-v1-opt] BIG line=big tokens=1948 source=estimate base=fast arm=fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused levers=chunk_pair,drop_bond_mask,relp_lean,diffusion_cond_chunk,conf_head_chunk,msa_zfree,diffcache_free exact=measured refused=none off=cache_release,expandable_segments,recycle_carry on=none property_off=hoist,keep_pool,sg,trimul_torch allocator=none",
  "rows": [
   "[protenix-v1-opt] LEVER name=drop_bond_mask state=on impl=protenix_v1_opt.big origin=kit strategy=LOCAL.protenix_v1.drop_bond_mask exact=bitwise scope=unit key=bond_mask sites=InferenceRunner.predict",
   "[protenix-v1-opt] LEVER name=relp_lean state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=bitwise scope=unit rows=512 sites=RelativePositionEncoding.generate_relp",
   "[protenix-v1-opt] LEVER name=diffusion_cond_chunk state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=measured scope=unit rows=256 sites=DiffusionConditioning.prepare_cache",
   "[protenix-v1-opt] LEVER name=conf_head_chunk state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=measured scope=unit rows=256 sites=ConfidenceHead.memory_efficient_forward",
   "[protenix-v1-opt] LEVER name=msa_zfree state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=bitwise scope=unit sites=MSABlock.forward,MSAModule.forward",
   "[protenix-v1-opt] LEVER name=diffcache_free state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=bitwise scope=unit sites=DiffusionConditioning.prepare_cache,AtomAttentionEncoder.prepare_cache,Protenix.run_confidence_head",
   "[protenix-v1-opt] LEVER name=cache_release state=off reason=flag impl=opt_core.mem.allocator origin=core strategy=F7.cache_release",
   "[protenix-v1-opt] LEVER name=expandable_segments state=off reason=flag impl=opt_core.mem.allocator origin=core strategy=F7.expandable_segments",
   "[protenix-v1-opt] LEVER name=recycle_carry state=off reason=flag impl=opt_core.mem.ckpt origin=core strategy=F7.recycle_carry",
   "[protenix-v1-opt] LEVER name=trimul_torch state=skipped reason=below_gate impl=configs.triangle_multiplicative origin=kit strategy=F7.chunked_eval gate=n_token<=2048 n=1948"
  ]
 },
 "p2_2956": {
  "big_line": "[protenix-v1-opt] BIG line=big tokens=2956 source=estimate base=fast arm=fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused levers=chunk_pair,drop_bond_mask,trimul_torch exact=measured refused=none off=cache_release,conf_head_chunk,diffusion_cond_chunk,expandable_segments,recycle_carry,relp_lean on=none property_off=atom_fused,atomattn,cond_dedupe,diffcache_free,dit_fused,dit_lowp,ditattn,ditattnfp16,fast,gflash,hoist,keep_pool,msa_zfree,opm_fused,pfattn,pwa_fused,sg,summary_hostidx,ttr allocator=none",
  "rows": [
   "[protenix-v1-opt] LEVER name=drop_bond_mask state=on impl=protenix_v1_opt.big origin=kit strategy=LOCAL.protenix_v1.drop_bond_mask exact=bitwise scope=unit key=bond_mask sites=InferenceRunner.predict",
   "[protenix-v1-opt] LEVER name=trimul_torch state=on impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval exact=measured scope=unit triangle_multiplicative=torch sites=InferenceRunner.configs.triangle_multiplicative,Protenix.configs_(the_same_ConfigDict,_protenix.py:98)",
   "[protenix-v1-opt] LEVER name=cache_release state=off reason=flag impl=opt_core.mem.allocator origin=core strategy=F7.cache_release",
   "[protenix-v1-opt] LEVER name=conf_head_chunk state=off reason=flag impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval",
   "[protenix-v1-opt] LEVER name=diffusion_cond_chunk state=off reason=flag impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval",
   "[protenix-v1-opt] LEVER name=expandable_segments state=off reason=flag impl=opt_core.mem.allocator origin=core strategy=F7.expandable_segments",
   "[protenix-v1-opt] LEVER name=recycle_carry state=off reason=flag impl=opt_core.mem.ckpt origin=core strategy=F7.recycle_carry",
   "[protenix-v1-opt] LEVER name=relp_lean state=off reason=flag impl=protenix_v1_opt.big origin=kit strategy=F7.chunked_eval",
   "[protenix-v1-opt] LEVER name=msa_zfree state=off reason=property_gate impl=big.py origin=kit strategy=F7.chunked_eval gate=n_gpu<=1",
   "[protenix-v1-opt] LEVER name=diffcache_free state=off reason=property_gate impl=big.py origin=kit strategy=F7.chunked_eval gate=n_gpu<=1"
  ]
 }
}
EXPECTED["p1_948"] = {"big_line": EXPECTED["p1_1948"]["big_line"].replace("tokens=1948", "tokens=948"),      # kit 0.2.44: the memory gate is 0 — a 948-token run arms
                      "rows": [r.replace(" n=1948", " n=948") for r in EXPECTED["p1_1948"]["rows"]]}             # exactly what a 1,948-token run arms (trimul_torch waits for 2048)


def _arm(monkeypatch, tmp_path, tag, n, P, **kw):
    B.reset()
    monkeypatch.setenv("ROWPAIR_WORLD", str(P)) if P > 1 else monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    if P > 1:
        env = dict(os.environ); NGPU.large_input_regime(env)
        for k, v in env.items():
            if k not in os.environ:
                monkeypatch.setenv(k, v)
    p = tmp_path / f"in_{tag}.json"; p.write_text(json.dumps([{"name": "x", "sequences": [{"proteinChain": {"sequence": "A" * n, "count": 1}}]}]))
    monkeypatch.setattr(sys, "argv", ["protenix", "pred", "--input", str(p)])
    monkeypatch.setattr(B, "_module_pinned", lambda lever, mod: None)
    for name in ("_install_relp_lean", "_install_recycle_carry", "_install_diffusion_cond_chunk", "_install_conf_head_chunk", "_install_msa_zfree", "_install_diffcache_free"):
        monkeypatch.setattr(B, name, lambda *a: (lambda: None))
    B.arm(M.resolve("big"), switches=dict(CPU_OFF), **kw)


@pytest.mark.parametrize("tag,n,P", [("p1_2956", 2956, 1), ("p1_1948", 1948, 1), ("p1_948", 948, 1), ("p2_2956", 2956, 2)])
def test_the_big_and_lever_lines_are_the_lines_of_record(monkeypatch, tmp_path, tag, n, P):
    _arm(monkeypatch, tmp_path, tag, n, P)
    assert B.line() == EXPECTED[tag]["big_line"]
    assert B.lever_rows("big") == EXPECTED[tag]["rows"]
    if P > 1:
        assert B.fields()["off_by_flag"] == sorted(set(CPU_OFF) | set(NGPU.SUPERSEDED_LEVERS))
    B.reset()


@pytest.mark.parametrize("opt_out,word", [(None, "--allow-partial"), ("PROTENIX_V1_OPT_ALLOW_PARTIAL=1", "PROTENIX_V1_OPT_ALLOW_PARTIAL=1")])
def test_the_census_exit_lines_name_the_routes_opt_out(monkeypatch, tmp_path, opt_out, word):
    _arm(monkeypatch, tmp_path, "p1_1948", 1948, 1, opt_out=opt_out)
    partial, reason = B.exit_join([], None, {}, False, True)                     # nothing observed on any unit: census-partial, refused
    assert B.exit_lines() == [f"[protenix-v1-opt] NOT ACTIVE: big partial — levers=chunk_pair,drop_bond_mask,relp_lean,diffusion_cond_chunk,conf_head_chunk,msa_zfree,diffcache_free (chunk_pair: <no unit observed>; drop_bond_mask: <no unit observed>; relp_lean: <no unit observed>; diffusion_cond_chunk: <no unit observed>; conf_head_chunk: <no unit observed>; msa_zfree: <no unit observed>; diffcache_free: <no unit observed>); exit 3 ({word} records and proceeds)"]
    partial, reason = B.exit_join([], None, {}, True, True)                      # the opt-out: recorded, proceeds
    assert B.exit_lines() == [f"[protenix-v1-opt] PARTIAL allowed: levers=chunk_pair,drop_bond_mask,relp_lean,diffusion_cond_chunk,conf_head_chunk,msa_zfree,diffcache_free (chunk_pair: <no unit observed>; drop_bond_mask: <no unit observed>; relp_lean: <no unit observed>; diffusion_cond_chunk: <no unit observed>; conf_head_chunk: <no unit observed>; msa_zfree: <no unit observed>; diffcache_free: <no unit observed>) ({word}, recorded)"]
    B.reset()


def test_the_route_names_its_opt_out():
    assert B.route_opt_out({"trigger": "runner"}) == "PROTENIX_V1_OPT_ALLOW_PARTIAL=1" and B.route_opt_out({"trigger": None}) is None and B.route_opt_out(None) is None
