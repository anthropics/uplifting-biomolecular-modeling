"""The `big` memory mode (big.py over opt_core.mem): the composition and its lever-property gates sized on the run's input, the line and
the strategy ids canonical, the activation flow on a fake runner (arm → apply → per-item census → exit gate)
without torch, the LEVER / BIG / census lines. No GPU: the levers that need CUDA are turned off by their flags here and proven on the dev
boxes."""
import json
import os
import sys
import types

import pytest

from protenix_v1_opt import big as B, modes as M, report as R, stack

CPU_OFF = {"recycle_carry": False, "cache_release": False, "expandable_segments": False}   # the CUDA levers: off by flag on a CPU box (named on the BIG line)


def _input(tmp_path, n, name="item", extra=()):
    p = tmp_path / f"{name}{n}.json"
    p.write_text(json.dumps([{"name": f"{name}{n}", "sequences": [{"proteinChain": {"sequence": "A" * n, "count": 1}}] + list(extra)}]))
    return str(p)


@pytest.fixture(autouse=True)
def _fresh():
    B.reset()
    yield
    B.reset()


def test_the_line_and_the_table_name_every_lever_once():
    assert B.LINE == ("expandable_segments", "cache_release", "chunk_pair", "drop_bond_mask", "relp_lean", "recycle_carry", "diffusion_cond_chunk", "conf_head_chunk", "msa_zfree", "diffcache_free", "trimul_torch")
    assert B.LINE.index("cache_release") < B.LINE.index("recycle_carry")            # recycle_carry's seam flush is the cache_release lever's call (opt_core.mem.ckpt)
    assert B.LINE.index("relp_lean") < B.LINE.index("diffusion_cond_chunk")
    assert set(B.LINE) <= set(B.TABLE) and B.SIZE_GATED_LEVERS == B.MEMORY_GATED_LEVERS + ("trimul_torch",)
    assert B.MEMORY_GATED_LEVERS == ("cache_release", "chunk_pair", "recycle_carry", "diffusion_cond_chunk", "conf_head_chunk") and B.MEMORY_GATE_TOKENS == 0                  # kit 0.2.44: the five engage at every sized input (was 1023)
    assert B.ROWPAIR_SITELESS == ("msa_zfree", "diffcache_free") and "msa_zfree" not in B.SIZE_GATED_LEVERS and "diffcache_free" not in B.SIZE_GATED_LEVERS   # size-blind
    assert B.SETTINGS["msa_zfree"] == B.SETTINGS["diffcache_free"] == {} and B.TABLE["msa_zfree"]["strategy"] == B.TABLE["diffcache_free"]["strategy"] == "F7.chunked_eval"   # storage-release levers: no settings, one strategy id < B.TRIMUL_GATE_TOKENS
    assert {"cache_release", "recycle_carry"} <= set(B.MEMORY_GATED_LEVERS)         # gated together: the core refuses recycle_carry's seam flush without cache_release
    assert not set(B.MEMORY_GATED_LEVERS) & {"expandable_segments", "drop_bond_mask", "relp_lean"}   # the three free levers stay size-blind
    assert set(B.KIT_LEVERS) | set(B.CORE_LEVERS) == set(B.LINE)


def test_every_strategy_id_has_the_lever_lines_form():
    from opt_core import report as CORE_R
    for lever, row in B.TABLE.items():
        assert CORE_R.strategy_form(row["strategy"]) == row["strategy"], lever                    # catalogue membership: test_strategy_catalogue.py
        assert row["class"] and row["evidence"] and (row["adopted"] == "yes" or row["adopted"].startswith("yes ("))


def test_the_gates(tmp_path):
    MEM = B.MEMORY_GATED_LEVERS
    for n in (None,):                                                                                                 # at or below the memory gate (and unsized): the DiT hoist and the sampler graphs off (EVERY size), the
        g = B.gates_for(n, {})                                                                                        # five memory levers and trimul_torch not applied — the free levers only
        assert g["drop"] == ("hoist", "sg", "keep_pool", "sampler_prep", "dit_fused", "dit_lowp") == B.BASE_LEVERS_OFF and g["levers_off"] == MEM + ("trimul_torch",), n
        assert set(g["off_by_property"]) == {"hoist", "sg", "keep_pool", "trimul_torch", *MEM} and g["sized"] is (n is not None), n
        assert g["memory_gate_tokens"] == 0 and g["trimul_gate_tokens"] == 2048 and g["gates"] == dict({lv: 0 for lv in MEM}, trimul_torch=2048), n
        assert all(f"N_token {n if n is not None else 'unsized'} <= 0 (the memory gate)" in g["off_by_property"][lv] for lv in MEM), n
    for n in (1, 400, 948, 1023, 1024, 2048):                                                                             # above the memory gate, at or below the TriMul gate: the five arm, trimul_torch waits for its gate
        g = B.gates_for(n, {})
        assert g["levers_off"] == ("trimul_torch",) and set(g["off_by_property"]) == {"hoist", "sg", "keep_pool", "trimul_torch"}, n
    g = B.gates_for(2049, {})
    assert g["levers_off"] == () and set(g["off_by_property"]) == {"hoist", "sg", "keep_pool", "fast", "gflash"}                 # > 2048: trimul_torch on; fast / gflash disengaged by property
    assert B.gates_for(5000, {"PROTENIX_V1_BIG_TRIMUL_TORCH_TOKENS": "6000"})["levers_off"] == ("trimul_torch",)   # the TriMul gate reads the caller's setting
    assert B.gates_for(400, {"PROTENIX_V1_BIG_TRIMUL_TORCH_TOKENS": "0"})["levers_off"] == ()                      # the two gates are independent thresholds
    assert B.memory_gate_tokens() == B.SETTINGS["size"]["memory_gate_tokens"] == B.MEMORY_GATE_TOKENS                # the one statement of the memory gate (no caller name reads it: the size statement decides the side)
    with pytest.raises(B.BigError, match="PROTENIX_V1_BIG_TRIMUL_TORCH_TOKENS='x': an integer token count"):
        B.gates_for(10, {"PROTENIX_V1_BIG_TRIMUL_TORCH_TOKENS": "x"})


def test_the_composition_is_sized_on_the_largest_item(tmp_path):
    c = B.compose({}, ["protenix", "pred", "--input", _input(tmp_path, 948)])                                     # below the TriMul gate (the memory gate is 0 since kit 0.2.44): every lever but trimul_torch
    assert (c["line"], c["arm"], c["base"], c["base_arm"], c["drop"]) == ("big", "fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused", "fast", "fast+gflash+tricuda+ttr+sg+hoist+keep_pool+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+sampler_prep+pfattn+opm_fused+pwa_fused+cond_dedupe+dit_fused+dit_lowp+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused", ("hoist", "sg", "keep_pool", "sampler_prep", "dit_fused", "dit_lowp"))
    assert c["levers"] == B.compose({}, ["protenix", "pred", "--input", _input(tmp_path, 1948)])["levers"]        # kit 0.2.44: the memory gate is 0 — 948 tokens arms what 1948 arms
    assert c["size"] == {"n_token": 948, "source": "estimate", "input": c["size"]["input"], "per_item": {"item948": 948}, "unsized": None, "reason": c["size"]["reason"]}
    c = B.compose({}, ["protenix", "pred", "--input", _input(tmp_path, 1948)])                                    # above the memory gate, at or below the TriMul gate: every lever but trimul_torch
    assert c["arm"] == "fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused" and c["levers"] == ("expandable_segments", "cache_release", "chunk_pair", "drop_bond_mask", "relp_lean", "recycle_carry", "diffusion_cond_chunk", "conf_head_chunk", "msa_zfree", "diffcache_free")
    assert B.compose({B.N_TOKEN_ENV: "1948"}, ["protenix", "pred", "--input", _input(tmp_path, 948)])["levers"] == c["levers"]   # the caller's size statement puts a small run on the above-gate composition (the pre-gate behaviour, by statement)
    c = B.compose({}, ["protenix", "pred", f"--input={_input(tmp_path, 2956)}"])
    assert c["arm"] == "fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused" and c["drop"] == ("hoist", "sg", "keep_pool", "sampler_prep", "dit_fused", "dit_lowp") and c["levers"] == B.LINE
    two = tmp_path / "two.json"
    two.write_text(json.dumps([{"name": "a", "sequences": [{"proteinChain": {"sequence": "A" * 100, "count": 2}}]},
                               {"name": "b", "sequences": [{"proteinChain": {"sequence": "A" * 3000, "count": 1}}, {"ligand": {"ligand": "CCD_ATP", "count": 1}}]}]))
    c = B.compose({}, ["x", "--input", str(two)])
    assert c["size"]["n_token"] == 3000 and c["size"]["per_item"] == {"a": 200, "b": 3000} and c["size"]["unsized"] == {"b": ["ligand"]}   # a non-polymer entry is named, never refused
    c = B.compose({B.N_TOKEN_ENV: "5000"}, ["x", "--input", str(two)])
    assert c["size"]["source"] == "environment" and c["size"]["n_token"] == 5000 and "trimul_torch" in c["levers"]      # the caller's statement wins
    c = B.compose({}, ["protenix", "pred"])
    assert c["size"]["source"] == "unsized" and c["arm"] == "fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused" and c["levers"] == ("expandable_segments", "drop_bond_mask", "relp_lean", "msa_zfree", "diffcache_free")   # unsized: the same arm, below both size gates
    with pytest.raises(B.BigError, match="PROTENIX_V1_BIG_SIZE_N_TOKEN='0': a positive token count"):
        B.compose({B.N_TOKEN_ENV: "0"}, ["x"])


def test_the_mode_is_one_composition(tmp_path):
    """`--mode big` is ONE lever set (the sized composition): no line selector, by flag or by environment."""
    c = B.compose({"PROTENIX_V1_OPT_EVIDENCE": "lean"}, ["x", "--input", _input(tmp_path, 2048)])          # a variable the package no longer reads
    assert (c["line"], c["base"]) == ("big", "fast") and c["arm"] != M.STOCK_ARM and set(c["levers"]) <= set(B.LINE)
    assert not hasattr(B, "line_request") and not hasattr(B, "EVIDENCE_ENV")


def test_resolve_big_is_the_sized_composition(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["protenix", "pred", "--input", _input(tmp_path, 3000)])
    res = M.resolve("big")
    assert (res.mode, res.arm, res.trimul, res.levers, res.tier, res.line) == ("big", "fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused", "fast", ("gflash", "tricuda", "ttr", "summary_hostidx", "ditattn", "ditattnfp16", "atomattn", "lazy_init", "template_dedupe", "tmpl_triatt", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "atom_fused", "tmpl_trimul", "tmpl_xtr", "tmpl_pairfused"), 2, "big")
    assert res.memory_preset is M.MEMORY_PRESET and res.triattn_floor == M.TRIATTN_FLOOR
    B.reset(); monkeypatch.setattr(sys, "argv", ["protenix", "pred", "--input", _input(tmp_path, 500)])
    res = M.resolve("big")
    assert res.arm == "fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused" != M.KIT_MODES["fast"].arm and res.levers == ("gflash", "tricuda", "ttr", "summary_hostidx", "ditattn", "ditattnfp16", "atomattn", "lazy_init", "template_dedupe", "tmpl_triatt", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "atom_fused", "tmpl_trimul", "tmpl_xtr", "tmpl_pairfused")      # below the size gate: the same arm, no hoist / sg
    assert dict(M.DETERMINISM)["big"] == "band" and M.KIT_MODES["big"].tier == 2
    for m in ("exact", "fast", "off"):
        assert M.resolve(m).memory_preset is None


class _Rec:
    """A protenix runner stand-in: configs with the fields the levers set, a model with its own configs, a predict that records."""

    def __init__(self):
        ns = types.SimpleNamespace
        self.configs = ns(infer_setting=ns(chunk_size=256, dynamic_chunk_size=True, sample_diffusion_chunk_size=5), triangle_multiplicative="cuequivariance", skip_amp=ns(sample_diffusion=True))
        self.model = ns(configs=self.configs)
        self.calls = []

    def predict(self, data):
        self.calls.append(dict(data.get("input_feature_dict") or {}))
        return "prediction"


def _armed(monkeypatch, tmp_path, n_token, environ=None, argv_input=True, **arm_kw):
    """Arm the line on a CPU box: the CUDA levers off by name through the in-process selection (CPU_OFF: the switches a caller in this
    process states — arm's `switches`); `environ` = the caller's size statements."""
    for k, v in (environ or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "argv", ["protenix", "pred"] + (["--input", _input(tmp_path, n_token)] if argv_input else []))
    monkeypatch.setattr(B, "_module_pinned", lambda lever, mod: None)
    installed = []
    for name in ("_install_relp_lean", "_install_recycle_carry", "_install_diffusion_cond_chunk", "_install_conf_head_chunk", "_install_msa_zfree", "_install_diffcache_free"):
        monkeypatch.setattr(B, name, (lambda nm: (lambda *a: installed.append((nm, a)) or (lambda: installed.append(("undo", nm)))))(name))
    res = M.resolve("big")
    B.arm(res, switches=dict(CPU_OFF, **arm_kw.pop("switches", {})), **arm_kw)
    return res, installed


def test_arm_apply_and_the_lines_on_a_fake_runner(monkeypatch, tmp_path, capsys):
    res, installed = _armed(monkeypatch, tmp_path, 2956)
    f = B.fields()
    assert f["line"] == "big" and f["arm"] == "fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused" and f["n_token_estimate"] == 2956 and f["n_token_source"] == "estimate"
    assert f["levers"] == ["chunk_pair", "drop_bond_mask", "relp_lean", "diffusion_cond_chunk", "conf_head_chunk", "msa_zfree", "diffcache_free", "trimul_torch"]
    assert f["off_by_flag"] == ["cache_release", "expandable_segments", "recycle_carry"] and f["refused"] == [] and f["exact"] == "measured"
    assert f["exact_per_lever"] == {"chunk_pair": "bitwise", "drop_bond_mask": "bitwise", "relp_lean": "bitwise", "diffusion_cond_chunk": "measured", "conf_head_chunk": "measured", "msa_zfree": "bitwise", "diffcache_free": "bitwise", "trimul_torch": "measured"}
    r = _Rec()
    mem = B.apply(r, res)
    assert mem == {"applied": {"infer_setting.chunk_size": 128, "infer_setting.dynamic_chunk_size": False, "triangle_multiplicative": "torch"},
                   "replaced": {"infer_setting.chunk_size": 256, "infer_setting.dynamic_chunk_size": True, "triangle_multiplicative": "cuequivariance"}}
    assert (r.configs.infer_setting.chunk_size, r.configs.infer_setting.dynamic_chunk_size, r.configs.triangle_multiplicative) == (128, False, "torch")
    assert r.configs.infer_setting.sample_diffusion_chunk_size == 5 and r.configs.skip_amp.sample_diffusion is True      # nothing else touched
    assert [n for n, _ in installed] == ["_install_relp_lean", "_install_diffusion_cond_chunk", "_install_conf_head_chunk", "_install_msa_zfree", "_install_diffcache_free"] and installed[0][1] == (512,) and installed[2][1] == (256,) and installed[3][1] == ()
    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert line == ("[protenix-v1-opt] BIG line=big tokens=2956 source=estimate base=fast arm=fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused "
                    "levers=chunk_pair,drop_bond_mask,relp_lean,diffusion_cond_chunk,conf_head_chunk,msa_zfree,diffcache_free,trimul_torch exact=measured refused=none "
                    "off=cache_release,expandable_segments,recycle_carry on=none property_off=fast,gflash,hoist,keep_pool,sg allocator=none") == B.line()
    # two items: the levers mark per unit; drop_bond_mask acts before the class predict; the settings are re-read on the item
    assert r.predict({"sample_name": "u1", "N_token": 2956, "input_feature_dict": {"bond_mask": "T", "x": 1}}) == "prediction"
    assert r.calls[-1] == {"x": 1}                                                                   # bond_mask dropped before the stock predict saw the dict
    B._mark("relp_lean")                                                                             # a statement reached outside an item: noted on the record, never raised into the fold, never counted
    assert any("relp_lean: mark outside a unit" in n for n in B._rec().notes)
    r.configs.infer_setting.chunk_size = 64                                                          # something reset the setting between items: a NAMED fallback on the next unit
    r.predict({"sample_name": "u2", "N_token": 700, "input_feature_dict": {}})
    cen = B._rec().census()
    assert cen["units"]["u1"]["ran"] == ["drop_bond_mask", "chunk_pair", "trimul_torch"] and cen["units"]["u1"]["absent"] == ["relp_lean", "diffusion_cond_chunk", "conf_head_chunk", "msa_zfree", "diffcache_free"]
    assert cen["units"]["u2"]["skipped"] == {"drop_bond_mask": "the item carries no bond_mask"} and "chunk_pair" in cen["units"]["u2"]["fallback"]
    # the exit rule's big part: fast / gflash gated for the whole run above 2048 are disengaged by property; the census partial joins
    evidence = {"fast": {"served": 0, "fallback": {}, "gated": {"stock:path": 96}}, "gflash": {"served": 0, "fallback": {}, "gated": {"stock:chunk": 96}}, "ttr": {"served": 10, "fallback": {}, "gated": {}}}
    partial, reason = B.exit_join(["fast", "gflash"], "fast served 0 calls (gated {'stock:path': 96}); gflash served 0 calls (gated {'stock:chunk': 96})", evidence, False, True)
    assert "fast" not in partial and "gflash" not in partial and B.fields()["excused_by_property"].keys() == {"fast", "gflash"}
    assert {"relp_lean@u1", "chunk_pair@u2"} <= set(partial) and not any(n.startswith("size_gate") for n in partial), partial   # never marked / fell back; the size-gate crossing of u2 is NOT partial
    assert "size_gate" not in reason and B._STATE["gate"]["exit_code"] == R.EXIT_NOT_ACTIVE                                 # (the exit here is the two levers', not the crossing's)
    cross = B._STATE["gate"]["size_gate_crossings"]
    assert [c["unit"] for c in cross] == ["u2"] and cross[0]["n_token"] == 700 and len(cross[0]["crossed"]) == 1               # 0 < 700 < 2048 < 2956: sized above the TriMul gate, the composition kept (the memory gate, 0 since kit 0.2.44, is not crossed)
    assert cross[0]["kept"] == ["trimul_torch on"]                                                                          # the crossed gate's lever with its kept state
    assert ("[protenix-v1-opt] NOTE size gate re-decided at featurization: unit u2 N_token=700 (sized 2956, estimate): "
            "trimul_torch on — the composition the process was sized for is kept for this item; the run proceeds") in capsys.readouterr().err
    rows = B.lever_rows("big")
    assert rows[0] == "[protenix-v1-opt] LEVER name=drop_bond_mask state=on impl=protenix_v1_opt.big origin=kit strategy=LOCAL.protenix_v1.drop_bond_mask exact=bitwise scope=unit key=bond_mask sites=InferenceRunner.predict"
    assert any(l.startswith("[protenix-v1-opt] LEVER name=recycle_carry state=off reason=flag ") for l in rows)
    assert not any(" name=chunk_pair " in l for l in rows) and len(rows) == len(B.TABLE) - 1     # chunk_pair's evidence is the `memory` row
    ex = B.exit_lines()
    assert len(ex) == 1 and ex[0].startswith("[protenix-v1-opt] NOT ACTIVE: big partial")
    assert B.lever_rows("fast") == [] and B.report({"mode": "fast"}) is None and B.report({"mode": "big", "memory": mem}) == mem


def test_allow_partial_records_and_proceeds(monkeypatch, tmp_path):
    res, _ = _armed(monkeypatch, tmp_path, 1948)
    r = _Rec(); B.apply(r, res)
    r.predict({"sample_name": "u1", "N_token": 1948, "input_feature_dict": {"bond_mask": 1}})
    partial, reason = B.exit_join([], None, {}, True, True)
    assert partial and B._STATE["gate"]["allow_partial"] is True and B._STATE["gate"]["exit_code"] == R.EXIT_OK
    assert B.exit_lines()[0].startswith("[protenix-v1-opt] PARTIAL allowed: ")


def test_a_complete_run_has_no_census_line(monkeypatch, tmp_path):
    res, _ = _armed(monkeypatch, tmp_path, 1948)
    r = _Rec(); B.apply(r, res)
    orig = r.predict.__wrapped__
    def predict(data, *a, **kw):                                                   # the patched statements' marks, as the GPU path makes them inside the item
        for lv in ("relp_lean", "diffusion_cond_chunk", "conf_head_chunk", "msa_zfree", "diffcache_free"):
            B._mark(lv)
        return orig(data, *a, **kw)
    unit = B.before_predict(r, {"sample_name": "u1", "N_token": 1948, "input_feature_dict": {"bond_mask": 1}})
    predict({"input_feature_dict": {}}); B.after_predict(unit)
    partial, reason = B.exit_join([], None, {}, False, True)
    assert (partial, reason) == ([], None) and B.exit_lines() == [] and B._STATE["gate"]["census"]["ok"] is True
    v = R.verdict({"mode": "big", "levers": None, "items": []}, "stock", (), allow_partial=False)   # the lean line's shape (stock arm): no kit evidence, the census complete
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK


def test_refusals_are_by_name(monkeypatch, tmp_path):
    """The line is ONE lever set: a PROTENIX_V1_BIG_* name other than the two size statements — a retired lever switch, a retired setting
    word, a typo — is refused by name at composition (and at interpreter start: _autoload); a setting outside its domain and a switch this
    process states for a name the registry does not hold are refused by name; a lever a caller in this process switches off is off BY NAME."""
    monkeypatch.setattr(sys, "argv", ["protenix", "pred", "--input", _input(tmp_path, 1500)])                    # above the memory gate: every switchable lever in the line
    monkeypatch.setattr(B, "_module_pinned", lambda lever, mod: None)
    for name, value in (("PROTENIX_V1_BIG_BOGUS", "1"), ("PROTENIX_V1_BIG_RELP_LEAN", "0"), ("PROTENIX_V1_BIG_RELP_LEAN_ROWS", "64"), ("PROTENIX_V1_BIG_ALLOW_PARTIAL", "1")):
        reason = rf"undeclared {name}: the line's levers are one set, not switchable; the size statements it reads are PROTENIX_V1_BIG_SIZE_N_TOKEN, PROTENIX_V1_BIG_TRIMUL_TORCH_TOKENS"
        with pytest.raises(B.BigError, match=reason):
            B.compose({name: value}, ["protenix", "pred"])                              # the composition refuses it (modes.resolve words it as the mode's refusal:)
        B.reset(); monkeypatch.setenv(name, value)
        with pytest.raises(RuntimeError, match=reason):
            M.resolve("big")
        monkeypatch.delenv(name)
    B.reset(); monkeypatch.setitem(B.SETTINGS, "relp_lean", {"rows": 0})
    with pytest.raises(B.BigError, match="big refused — relp_lean: rows: rows=0: a positive row count"):
        B.arm(M.resolve("big"), switches=dict(CPU_OFF))
    monkeypatch.setitem(B.SETTINGS, "relp_lean", {"rows": 512})
    B.reset()
    with pytest.raises(B.BigError, match=r"big refused — big: switch.bogus_lever: switch 'bogus_lever' names no registered lever and no line lever"):
        B.arm(M.resolve("big"), switches=dict(CPU_OFF, bogus_lever=True))
    B.reset()
    res = M.resolve("big"); B.arm(res, switches=dict(CPU_OFF, relp_lean=False, diffusion_cond_chunk=False))
    assert B.fields()["off_by_flag"] == ["cache_release", "diffusion_cond_chunk", "expandable_segments", "recycle_carry", "relp_lean"]   # named, in name order; expandable_segments is in the line at every size (no graph pool under the mode): its switch removes it by name
    r = _Rec(); del r.configs.infer_setting.dynamic_chunk_size
    with pytest.raises(stack.ActivationError, match=r"big refused — chunk_pair: configs.infer_setting.dynamic_chunk_size: the runner's configs carry no `infer_setting.dynamic_chunk_size`"):
        B.apply(r, res)
    with pytest.raises(stack.ActivationError, match="the stock runner carries no `configs`"):
        B.apply(types.SimpleNamespace(model=None), res)


def test_the_pinned_file_rule(monkeypatch, tmp_path):
    fake = types.ModuleType("protenix.model.modules.embedders"); fake.__file__ = str(tmp_path / "embedders.py")
    (tmp_path / "embedders.py").write_text("# not the pinned file\n")
    monkeypatch.setitem(sys.modules, "protenix.model.modules.embedders", fake)
    import importlib
    monkeypatch.setattr(importlib, "import_module", lambda name: sys.modules[name])
    with pytest.raises(B.RefusalError) as ei:
        B._module_pinned("relp_lean", "protenix.model.modules.embedders")
    assert ei.value.refusal.precondition == "sha256.protenix.model.modules.embedders" and "the lever copies statements of the pinned file only" in ei.value.refusal.reason


def test_the_flag_is_the_census_opt_out(monkeypatch, tmp_path):
    """The census opt-out is the verb's `--allow-partial` (or the environment route's word), handed to the core at arm and at the gate:
    without it a census-partial unit is exit 3; with it the names are recorded and joined into the PARTIAL allowed line, exit 0. The
    old variable PROTENIX_V1_BIG_ALLOW_PARTIAL is refused by name (test_refusals_are_by_name)."""
    res, _ = _armed(monkeypatch, tmp_path, 1948)
    r = _Rec(); B.apply(r, res)
    r.predict({"sample_name": "u1", "N_token": 1948, "input_feature_dict": {"bond_mask": 1}})         # the patched statements never mark on the fake runner: u1 is census-partial
    partial, reason = B.exit_join([], None, {}, False, True)
    g = B._STATE["gate"]
    assert {"relp_lean@u1", "diffusion_cond_chunk@u1", "conf_head_chunk@u1"} <= set(partial) and g["allow_partial"] is False and g["exit_code"] == R.EXIT_NOT_ACTIVE
    partial, reason = B.exit_join([], None, {}, True, True)                                             # the kit's flag
    g = B._STATE["gate"]
    assert g["allow_partial"] is True and g["allow_partial_source"] == "kit" and g["exit_code"] == R.EXIT_OK
    assert B.exit_lines()[0].startswith("[protenix-v1-opt] PARTIAL allowed: ")


def test_the_size_statements_are_the_only_names_read_and_the_kit_reads_them(monkeypatch):
    """PROTENIX_V1_BIG_SIZE_N_TOKEN and PROTENIX_V1_BIG_TRIMUL_TORCH_TOKENS are the two names the line reads (big.GATE_SETTINGS ==
    _autoload.BIG_DECLARED): line_settings reads them kit-side into the settings the core records (ctx.settings), a malformed value is a
    BigError by name, the defaults are the line's SETTINGS row (stated once); the .pth hook refuses every other PROTENIX_V1_BIG_* name
    at interpreter start like a mistyped package switch."""
    from protenix_v1_opt import _autoload as A, big as B
    assert set(B.GATE_SETTINGS) == {B.TRIMUL_GATE_ENV, B.N_TOKEN_ENV} == set(A.BIG_DECLARED) and B.POLICY_LEVERS == ("size",)
    env = {B.N_TOKEN_ENV: "2956", B.TRIMUL_GATE_ENV: "2048"}
    s = B.line_settings(env)
    assert s["size"]["n_token"] == 2956 and s["trimul_torch"]["tokens"] == 2048 and s["relp_lean"] == B.SETTINGS["relp_lean"]
    assert B.line_settings({}) == {k: dict(v) for k, v in B.SETTINGS.items()}
    assert B.size_of_run(env, ["x"])["n_token"] == 2956 and B.gate_tokens(B.TRIMUL_GATE_ENV, {}) == B.SETTINGS["trimul_torch"]["tokens"] == B.TRIMUL_GATE_TOKENS == 2048
    assert B.gate_tokens(B.TRIMUL_GATE_ENV, {B.TRIMUL_GATE_ENV: "0"}) == 0
    with pytest.raises(B.BigError, match="TRIMUL_TORCH_TOKENS='-1': a non-negative token count"):
        B.gate_tokens(B.TRIMUL_GATE_ENV, {B.TRIMUL_GATE_ENV: "-1"})
    refused = []
    f = A.install({"PROTENIX_V1_BIG_SG_GATE_TOKENS": "0", B.N_TOKEN_ENV: "5"}, exit=refused.append)      # a name of no size statement: refused at interpreter start, by name
    assert f is None and refused == [A.EXIT_NOT_ACTIVE]
    assert A.install({B.N_TOKEN_ENV: "5", B.TRIMUL_GATE_ENV: "4096"}, exit=refused.append) is None and refused == [A.EXIT_NOT_ACTIVE]   # the two statements pass (no mode: nothing installed)


def test_a_size_gate_crossing_is_a_note_never_partial(monkeypatch, capsys):
    """The exit rule with ONLY a size-gate crossing on the census: nothing partial, no reason, the exit code stays the run's own; the crossing
    is recorded (`size_gate_crossings`, with the kept lever states)."""
    class _FakeRec:
        refused_names = set()

        def exit_gate(self, rc, expect_units, allow_partial):
            return {"partial": [], "reasons": {}, "census": {"units": {}}, "exit_code": rc, "allow_partial": bool(allow_partial)}
    monkeypatch.setattr(B, "_rec", lambda: _FakeRec())
    monkeypatch.setitem(B._STATE, "composition", {"gates": {"off_by_property": {}}})
    entry = {"unit": "u9", "n_token": 700, "estimate": 2956, "crossed": ["trimul_torch gate 2048: sized 2956 (estimate), the item has N_token=700"], "kept": ["trimul_torch on"]}
    monkeypatch.setitem(B._STATE, "size_gate_crossings", [entry])
    partial, reason = B.exit_join([], None, {}, False, True)
    assert partial == [] and reason is None
    gate = B._STATE["gate"]
    assert gate["names"] == [] and gate["exit_code"] == R.EXIT_OK and gate["size_gate_crossings"] == [entry]


def test_at_or_below_the_memory_gate_the_line_is_the_free_levers(monkeypatch, tmp_path, capsys):
    """Sized at or below MEMORY_GATE_TOKENS the five memory-cost levers do not arm (nothing installed, the runner's configs untouched — the
    upstream's own chunk ladder), the BIG line names them under property_off, their LEVER rows read `skipped reason=below_gate …
    gate=n_token<=1023 n=<sized N>` (chunk_pair's too: no `memory` settings were applied), the composed label is bitwise; an item above the
    gate in that process is NAMED (both gates it crosses) and keeps the composition; between the two gates the composition is the pre-gate one."""
    monkeypatch.setattr(B, "MEMORY_GATE_TOKENS", 1023); monkeypatch.setitem(B.SETTINGS["size"], "memory_gate_tokens", 1023)   # the mechanism at a stated threshold (the shipped gate is 0 since kit 0.2.44)
    res, installed = _armed(monkeypatch, tmp_path, 948)
    f = B.fields()
    assert f["levers"] == ["drop_bond_mask", "relp_lean", "msa_zfree", "diffcache_free"] and f["off_by_flag"] == ["expandable_segments"] and f["exact"] == "bitwise"   # cache_release / recycle_carry: not in the line at this size, so the CPU box's switches for them change nothing
    assert tuple(f["gates"]["levers_off"]) == B.MEMORY_GATED_LEVERS + ("trimul_torch",) and f["gates"]["memory_gate_tokens"] == 1023
    assert set(f["off_by_property"]) == {"hoist", "sg", "keep_pool", "trimul_torch", *B.MEMORY_GATED_LEVERS}
    r = _Rec()
    mem = B.apply(r, res)
    assert mem == {"applied": {}, "replaced": {}}                                                                    # chunk_pair below its gate: the configs keep the upstream's dynamic chunk ladder
    assert (r.configs.infer_setting.chunk_size, r.configs.infer_setting.dynamic_chunk_size, r.configs.triangle_multiplicative) == (256, True, "cuequivariance")
    assert [n for n, _ in installed] == ["_install_relp_lean", "_install_msa_zfree", "_install_diffcache_free"]                                                       # recycle_carry / diffusion_cond_chunk / conf_head_chunk: nothing patched
    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert line == ("[protenix-v1-opt] BIG line=big tokens=948 source=estimate base=fast arm=fast+gflash+tricuda+ttr+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+pfattn+opm_fused+pwa_fused+cond_dedupe+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused levers=drop_bond_mask,relp_lean,msa_zfree,diffcache_free exact=bitwise refused=none "
                    "off=expandable_segments on=none property_off=cache_release,chunk_pair,conf_head_chunk,diffusion_cond_chunk,hoist,keep_pool,recycle_carry,sg,trimul_torch allocator=none") == B.line()
    rows = B.lever_rows("big")
    by = {l.split(" name=", 1)[1].split(" ", 1)[0]: l for l in rows}
    assert len(rows) == len(B.TABLE) and set(by) == set(B.TABLE)                                                      # one row per lever, chunk_pair's included below its gate
    for lv in B.MEMORY_GATED_LEVERS:
        row = B.TABLE[lv]
        assert by[lv] == f"[protenix-v1-opt] LEVER name={lv} state=skipped reason=below_gate impl={row['impl'].split(' ', 1)[0]} origin={row['origin']} strategy={row['strategy']} gate=n_token<=1023 n=948", by[lv]
    assert by["trimul_torch"] == "[protenix-v1-opt] LEVER name=trimul_torch state=skipped reason=below_gate impl=configs.triangle_multiplicative origin=kit strategy=F7.chunked_eval gate=n_token<=2048 n=948"
    assert " state=on " in by["drop_bond_mask"] and " state=on " in by["relp_lean"] and " state=off reason=flag " in by["expandable_segments"]
    assert by["msa_zfree"].startswith("[protenix-v1-opt] LEVER name=msa_zfree state=on ") and by["diffcache_free"].startswith("[protenix-v1-opt] LEVER name=diffcache_free state=on ")   # size-blind: in the line below the gate too
    # an item above the gate in a process sized below it: named (both gates' levers with their kept state), never re-composed; cache_release does not act at unit end
    r.predict({"sample_name": "u1", "N_token": 1500, "input_feature_dict": {"bond_mask": 1}})
    err = capsys.readouterr().err
    assert ("NOTE size gate re-decided at featurization: unit u1 N_token=1500 (sized 948, estimate): cache_release off, chunk_pair off, recycle_carry off, diffusion_cond_chunk off, "
            "conf_head_chunk off — the composition the process was sized for is kept for this item; the run proceeds") in err
    cen = B._rec().census()
    assert cen["units"]["u1"]["ran"] == ["drop_bond_mask"] and "cache_release" not in cen["units"]["u1"]["ran"] and cen["units"]["u1"]["absent"] == ["relp_lean", "msa_zfree", "diffcache_free"]
    # between the gates (1024..2048) the composition is the pre-gate line; unsized = below both gates
    B.reset()
    assert B.compose({}, ["x", "--input", _input(tmp_path, 1024)])["levers"] == tuple(x for x in B.LINE if x != "trimul_torch")
    B.reset()
    res, _ = _armed(monkeypatch, tmp_path, 5, argv_input=False)
    rows = {l.split(" name=", 1)[1].split(" ", 1)[0]: l for l in B.lever_rows("big")}
    assert rows["recycle_carry"].endswith(" gate=n_token<=1023 n=unsized") and B.fields()["n_token_source"] == "unsized"


def test_the_stock_attention_package_is_preloaded_ahead_of_the_trunk_on_a_cuda_process_only():
    """apply() imports the upstream's attention package (cuequivariance_torch) before the forward on a CUDA process — the upstream's own lazy
    import inside the first item's pair stack measured ~60 s there; on a CPU process (this box) nothing is imported and the record says why.
    Never a refusal either way."""
    r = B.preload_stock_attention()
    assert r["package"] == B.STOCK_ATTENTION_PACKAGE == "cuequivariance_torch"
    assert (r["preloaded"] is True and r["seconds"] >= 0) or (r["preloaded"] is False and r["reason"])


def test_the_release_levers_are_left_out_under_the_row_sharded_line_by_name():
    """msa_zfree / diffcache_free hold no site under `--n_gpu P>1` (tp.py replaces MSAModule.forward and the sampler / confidence with the
    row-sharded statements): gates_for leaves them out of the composition in every rank and names them replaced_by_rowpair — never a
    `served 0 calls` at exit; at P == 1 both are in the line at every size (size-blind); their LEVER row under P>1 reads property_gate n_gpu<=1."""
    from opt_core.mem.rowpair import launch as L
    g1 = B.gates_for(1948, {})
    assert "msa_zfree" not in g1["levers_off"] and "diffcache_free" not in g1["levers_off"] and g1["size_gated_off"] == ("trimul_torch",)
    g0 = B.gates_for(948, {})
    assert "msa_zfree" not in g0["levers_off"] and "diffcache_free" not in g0["levers_off"]                        # size-blind: below the memory gate too
    g2 = B.gates_for(1948, {L.ENV_WORLD: "2"})
    assert g2["levers_off"] == ("msa_zfree", "diffcache_free", "trimul_torch") and g2["size_gated_off"] == ("trimul_torch",)
    for lv in B.ROWPAIR_SITELESS:
        assert g2["off_by_property"][lv].startswith("replaced_by_rowpair at n_gpu 2"), g2["off_by_property"][lv]


def test_free_storage_releases_a_tensor_in_place_and_is_idempotent():
    """The release levers' one primitive: the tensor object stays valid with 0-byte storage; a second call frees nothing; non-tensors free nothing."""
    torch = pytest.importorskip("torch")
    t = torch.zeros(64, 64)
    n = B.free_storage(t)
    assert n == 64 * 64 * 4 and t.untyped_storage().nbytes() == 0 and B.free_storage(t) == 0 and B.free_storage(None) == 0


def test_the_release_levers_wrap_the_stock_classes_and_undo(monkeypatch):
    """With the stock package importable: msa_zfree wraps MSABlock.forward / MSAModule.forward, diffcache_free wraps the two prepare_cache
    methods and Protenix.run_confidence_head (class level, __wrapped__ = the previous owner); each lever's undo restores the originals."""
    pytest.importorskip("torch")
    pf = pytest.importorskip("protenix.model.modules.pairformer")
    df = pytest.importorskip("protenix.model.modules.diffusion")
    tf = pytest.importorskip("protenix.model.modules.transformer")
    px = pytest.importorskip("protenix.model.protenix")
    monkeypatch.setattr(B, "_module_pinned", lambda lever, mod: None)
    b0, m0, c0, e0, r0 = pf.MSABlock.forward, pf.MSAModule.forward, df.DiffusionConditioning.prepare_cache, tf.AtomAttentionEncoder.prepare_cache, px.Protenix.run_confidence_head
    undo1 = B._install_msa_zfree(); undo2 = B._install_diffcache_free()
    try:
        assert pf.MSABlock.forward is not b0 and pf.MSABlock.forward.__wrapped__ is b0 and pf.MSAModule.forward.__wrapped__ is m0
        assert df.DiffusionConditioning.prepare_cache.__wrapped__ is c0 and tf.AtomAttentionEncoder.prepare_cache.__wrapped__ is e0
        assert px.Protenix.run_confidence_head.__wrapped__ is r0
    finally:
        undo1(); undo2()
    assert (pf.MSABlock.forward, pf.MSAModule.forward, df.DiffusionConditioning.prepare_cache, tf.AtomAttentionEncoder.prepare_cache, px.Protenix.run_confidence_head) == (b0, m0, c0, e0, r0)
