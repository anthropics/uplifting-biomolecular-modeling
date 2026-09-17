"""The big mode: the package side (big.py, modes wiring, the launcher's argv and record grammar) and the lever module's pure parts
(big_levers.py) — no model stack here; the lever bodies run only in the model process on a GPU box. The literals other tools read as text (MODES, KIT_MODES) are
asserted as literals."""
import ast
import os
import re

import json
import pytest

from af3_jax_opt import big, modes, registry as kit_registry, report, stack
from opt_core import modes as core_modes

WHY_040 = "the memory mode lands with core >= 0.4.0 (opt_core.mem.registry/record/compose/ckpt/jax_mem); below it big refuses by name (test_gate_refuses_by_name_below_core_040 runs at every core)"
from .conftest import assert_no_markers, chain, render
import inspect
from af3_jax_opt import cli, inputs, modes, stack, warm
from .conftest import PKG
from af3_jax_opt import modes, report


def _mem():
    """(registry, compose, big_launch, big_levers) — skipped BY NAME when the core beside the kit predates opt_core.mem.registry."""
    registry = pytest.importorskip("opt_core.mem.registry", reason=WHY_040)
    compose = pytest.importorskip("opt_core.mem.compose", reason=WHY_040)
    from af3_jax_opt import big_launch, big_levers
    return registry, compose, big_launch, big_levers

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STOCK_NET = os.path.join(PKG, "..", "..", "stock", "src", "src", "alphafold3", "model", "network")
STOCK_DH = os.path.join(STOCK_NET, "diffusion_head.py")
HOIST = os.path.join(PKG, "..", "forward", "flashpairformer", "af3_flashpairformer", "diffusion_hoist.py")
LINE = ("samples_per_pass", "transition_shard", "cond_shard", "logits_shard", "trimul_chunk")
OK = ("[af3-jax-opt] ACTIVE mode=big:transition_shard,cond_shard,trimul_chunk base=fast drop=none exact=measured "
      "refused=none off=samples_per_pass,logits_shard on=none allocator=none line=big n_gpu=1")   # 0.3.26: the one-GPU line = the registered line minus the not-composed levers, off BY NAME
CENSUS = ("[af3-jax-opt] BIG census units=1 applied=transition_shard,cond_shard,trimul_chunk refused=none "
          "traced=samples_per_pass=1,transition_shard=1,cond_shard=2,logits_shard=1,trimul_chunk=2 verdict=ok n_gpu=1")
LEVERS = "[af3-jax-opt] LEVERS active=L-ATTNCFG+L-GLUT af3_pallas_levers.py=x script=run_alphafold_fast.py sha256=y"
L1 = "Featurisation prefetch enabled: 3 worker process(es), 4 item(s) ahead.\nOutput writer enabled: result extraction and output writing run on one writer thread behind the next fold job."   # the two row levers' lines (L1, WRITER) as one transcript element
SERVED_OFF = "[af3-jax-opt] SERVED trimul fused=0 fallback=0 triatt fused=0 fallback=0 fallback_shapes=none hoist=off tiles=own:9.0 cc=9.0 dattn=10 dattn_sites=diffusion:2,pairformer_single:8 ttr=off ttr_routed=0 ttr_fallback=none sbf16=16 sbf16_sites=atom:12,token:4 txla=24 txla_rows=cuda_sm90a:20,k2b_aot:4 txla_aside=none atomattn=7 atomattn_sites=diffusion_decoder:3,diffusion_encoder:3,evoformer_encoder:1 atomattn_fallback=none hlog=2 hlog_dtype=float32 cshare=2 achoist=2 achoist_sites=enc:2,dec:2,passed:2 achoist_aside=none cnoise=off cnoise_rule=none tcd=24 tcd_rows=cd_trimul:24 tcd_aside=none tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_uncovered=af3_tmpl_c64_ch64_incoming:2" + "\n[af3-jax-opt] LNP served=150 word=fast rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4 routed=adaptive:40,axis:2,channels:12 aside=none uncovered=none"   # big at one GPU: the add-on disengaged, DATTN inherited from fast, TTR not in the composition
SERVED_ON = SERVED_OFF.replace("trimul fused=0", "trimul fused=48")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("AF3_JAX_"):
            monkeypatch.delenv(k, raising=False)


def test_gate_refuses_by_name_below_core_040(monkeypatch):
    """With the core's lever registry absent (core < 0.4.0, simulated here at any core) the mode is NOT ACTIVE by name — resolve raises
    UnsupportedMode with the reason, activation reports it; nothing is composed silently and the other modes are untouched."""
    import sys
    monkeypatch.setitem(sys.modules, "opt_core.mem.registry", None)         # importing opt_core.mem.registry raises ImportError, as on a core without it
    why = big.gate()
    assert why and why.startswith("big: core_missing:opt_core.mem.registry") and "0.4.0" in why, why
    with pytest.raises(modes.UnsupportedMode):
        modes.resolve("big")
    assert modes.resolve("exact")["levers"] == modes.KIT_MODES["exact"]["levers"] and modes.resolve("fast")["cache_class"] == "__fast"


def test_modes_literal_carries_big_as_the_one_memory_mode():
    _, _, big_launch, big_levers = _mem()
    src = open(os.path.join(PKG, "modes.py"), encoding="utf-8").read()
    m = re.search(r"(?m)^MODES = \((.*)\)$", src)
    assert m, "MODES is a literal tuple statement (read by regex from outside the package)"
    literal = tuple(x.strip().strip('"') for x in m.group(1).split(",") if x.strip())
    assert literal == modes.MODES == ("off", "exact", "fast", "big") and modes.BIG == big.MODE == "big" and modes.DEFAULT_MODE != modes.BIG
    assert modes.BIG_LINES == {"big": "fast"} and big.LINE == "big" and set(big_launch.INSTALL) == set(modes.BIG_LINES) == set(big_levers.LINE_LEVERS)   # ONE composition: the mode's own line on the fast base; no evidence toggle, no alias
    for m_ in modes.MODES:                                                   # read per mode from outside the package
        assert isinstance(modes.KIT_MODES[m_]["kits"], tuple) and isinstance(modes.KIT_MODES[m_]["levers"], list)
    assert modes.KIT_MODES["big"]["launcher"] == "big" and modes.KIT_MODES["big"]["cache_class"] == "__big" and set(modes.KIT_MODES["big"]["kits"]) == {"fast_inference", "pallas", "fpf"}


def test_compose_big_validates_the_literal():
    registry, compose, big_launch, big_levers = _mem()
    table = core_modes.ModeTable(modes=tuple(m for m in modes.MODES if m != "big"), default=modes.DEFAULT_MODE)
    new, bl = compose.compose_big(table, modes.BIG_BASE, big.levers(), name="big")
    assert new.modes == modes.MODES and bl.base == "fast" and bl.levers == big.levers() == LINE


def test_lever_ids_agree_across_the_tables():
    registry, compose, big_launch, big_levers = _mem()
    assert big_levers.LINE == big.levers() == LINE                                                       # the registered line (five levers, the launcher's / the record's vocabulary)
    assert tuple(x.lower() for x in modes.KIT_MODES["big"]["levers"]) == tuple(l for l in LINE if l.upper() not in modes.BIG_NOT_COMPOSED) == ("transition_shard", "cond_shard", "trimul_chunk")   # the one-GPU composition: the line minus the not-composed levers (0.3.26, measured)
    assert set(modes.BIG_NOT_COMPOSED) == {"SAMPLES_PER_PASS", "LOGITS_SHARD"} and modes.BIG_NGPU_LEVERS == []   # 0.3.38: the P > 1 line composes no extra memory lever (SAMPLES_PER_PASS off by name there too)
    assert not hasattr(big, "FLAG_PREFIX") and not hasattr(stack, "FLAG_PREFIX") and not hasattr(big, "RECORD_NAME") and not hasattr(big_launch, "RECORD_NAME")   # no switch names anywhere in the wrapper; the launcher writes no record file: its evidence is the ACTIVE and census lines
    assert modes.OFF_ARG == big_launch.OFF_ARG and not hasattr(modes, "ALLOW_PARTIAL_ARG") and not hasattr(big_launch, "ALLOW_PARTIAL_ARG")   # the wrapper's word is the launcher's; no opt-out word exists
    for lv in modes.KIT_MODES["big"]["levers"]:
        row = kit_registry.LEVERS[lv]
        assert row["kind"] in modes.GRAPH_KINDS and row["impl"] == "jax" and row["switch"].startswith("--mode big") and "BIG" not in row["switch"] and modes.INPROCESS_MODULES[lv] == "big_levers.py"
        report._core.strategy_form(row["strategy"])                    # a strategy id of the LEVER line's form (catalogue membership: test_strategy_catalogue.py)
    assert set(modes.BIG_DISENGAGED) < set(modes.KIT_MODES["fast"]["levers"]) and set(modes.BIG_FALLBACK) < set(modes.KIT_MODES["exact"]["levers"])
    assert modes.base_levers_of("big") == ["FIX1", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "LNP", "GLUT", "ATTNCFG"] and modes.fpf_env_disengaged() == {"AF3_FLASHPAIRFORMER": "off", "AF3_DIFFUSION_HOIST": "0"}


def test_resolution_is_the_one_composition_on_the_fast_base():
    registry, compose, big_launch, big_levers = _mem()
    res = modes.resolve("big", "/c")
    assert res["base"] == "fast" and res["cache_class"].startswith("__big-big-") and res["cache_class"] == modes.cache_class_of("big") == "__big" + big.cache_suffix()
    assert res["levers"] == ["FIX1", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "LNP", "GLUT", "ATTNCFG", "TRANSITION_SHARD", "COND_SHARD", "TRIMUL_CHUNK", "L1", "WRITER"] == modes.levers_of("big")
    assert res["row_levers"] == ["L1", "WRITER"] and res["script"] == modes.FAST_SCRIPT and set(res["disengaged"]) == set(modes.BIG_DISENGAGED) | set(modes.BIG_DISENGAGED_AT_EDGE)   # size unknown: the conservative composition
    sized = modes.resolve("big", "/c", region="reach", size={"n_padded": 4096, "n_gpu": 1})   # one GPU, at or below the fused pair kernels' serve edge: FPF_TRIATT + TTR kept (0.00 GiB measured, faster), ATTNCFG's site not vacated
    assert sized["levers"] == ["FIX1", "FPF_TRIATT", "TTR", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "COND_SHARE", "LNP", "GLUT", "TRANSITION_SHARD", "COND_SHARD", "TRIMUL_CHUNK", "L1", "WRITER"]   # 0.3.26: COND_SHARE engages in reach on one GPU (no chunked sampler composed); SAMPLES_PER_PASS / LOGITS_SHARD not composed
    assert set(sized["disengaged"]) == set(modes.BIG_DISENGAGED) and sized["env"]["AF3_FLASHPAIRFORMER"] == "triatt" and sized["env"]["AF3_DIFFUSION_HOIST"] == "0" and sized["env"]["AF3_JAX_TTR"] == "big" and "AF3P_ATTN_CFG" not in sized["env"] and sized["env"]["AF3P_GLU_T"] == "1"
    for size in ({"n_padded": 6144, "n_gpu": 1}, {"n_padded": 2048, "n_gpu": 2}, None):          # past the edge / row-sharded / unknown: the fused pair levers leave, ATTNCFG returns at the vacated site
        assert modes.resolve("big", "/c", region="reach", size=size)["levers"] == res["levers"]
    assert res["env"]["AF3_FLASHPAIRFORMER"] == "off" and res["env"]["AF3_DIFFUSION_HOIST"] == "0" and set(modes.pallas_levers()).issubset(res["env"]) and set(modes.fpf_env()) <= set(res["env"]) and res["env"]["AF3_JAX_DATTN"] == "big" and "AF3_JAX_TTR" not in res["env"]   # DATTN names the shared core's provider by the memory mode's tier word (modes.tier_word_env)
    assert [os.path.basename(x) for x in res["launcher"]][0] == modes.BIG_LAUNCHER == big.LAUNCHER and res["launcher"][4:] == ["--line", "big", "--base", "fast", modes.OFF_ARG, "samples_per_pass,logits_shard"]
    assert os.path.isfile(res["launcher"][0]) and os.path.isfile(os.path.join(res["launcher"][1], "opt_core", "mem", "registry.py")) and os.path.isfile(res["launcher"][3])
    assert big_launch.parse_argv(res["launcher"] + ["/x/run_alphafold_fast.py", "--json_path=a"])[3:] == ("big", "fast", ("samples_per_pass", "logits_shard"), "/x/run_alphafold_fast.py", ["--json_path=a"])   # one device: the not-composed levers on --off, by name
    for mode in ("exact", "fast"):                                           # the other modes resolve as their table rows: the big rows are inert outside big
        assert modes.levers_of(mode) == modes.KIT_MODES[mode]["levers"] and modes.cache_class_of(mode) == modes.KIT_MODES[mode]["cache_class"]
        assert modes.resolve(mode)["base"] is None and modes.resolve(mode)["disengaged"] == {} and not any("BIG" in k for k in modes.resolve(mode)["env"])


def test_the_mode_has_one_lever_set_and_no_caller_switch():
    """big's lever set is fixed: the registry's resolution of the line with no switch set — every memory lever, in order, at its registered
    settings; nothing in any environment selects a sub-line (the wrapper reads and writes no memory-lever switch name; a --n_gpu P > 1
    composition names the levers it switches off on the launcher's command line). The cache class suffix is that lever set's, the one the printed lines carry."""
    registry, compose, big_launch, big_levers = _mem()
    sel = big.selection()
    assert sel["levers"] == LINE and sel["off"] == sel["on"] == () and sel["flags"] == {} and sel["refusals"] == [] and sel["allow_partial"] is False and sel["line"] == "big"
    assert "environ" not in inspect.signature(big.selection).parameters and "environ" not in inspect.signature(big.cache_suffix).parameters
    assert "environ" not in inspect.signature(modes.levers_of).parameters and "environ" not in inspect.signature(modes.cache_class_of).parameters and not hasattr(big, "flags_env")
    assert re.fullmatch(r"-big-[0-9a-f]{8}", big.cache_suffix()) and modes.cache_class_of("big") == "__big" + big.cache_suffix()
    recorded = set(re.findall(r"__big(-big-[0-9a-f]{8})", open(os.path.join(PKG, "tests", "printed_lines.json"), encoding="utf-8").read()))
    assert recorded == {big.cache_suffix()}                                # the class the recorded big / big ×2 lines name is this lever set's class
    assert {**modes.mode_env("big"), "AF3_JAX_DATTN": "big", "AF3_JAX_TRIATT_XLA": "big", **modes.TREE_LEVER_ENV["SAMPLER_BF16"], **modes.TREE_LEVER_ENV["ATOM_ATTN"], "AF3_JAX_LNP": "big"} == modes.resolve("big")["env"] == modes.lever_env("big", modes.levers_of("big"))   # size unknown: every provider-bound lever of that composition carries the word big
    assert not any("BIG" in k for k in modes.resolve("big", "/c")["env"]) and modes.resolve("big", "/c")["launcher"][-2:] == [modes.OFF_ARG, "samples_per_pass,logits_shard"]   # one device: the not-composed levers off BY NAME on the launcher's word, nothing else switched


def _switch_name(which: str) -> str:
    """A memory-lever switch name of the retired variable grammar (<stem>BIG_<LEVER>[_<SETTING>] | …LINE | …ALLOW_PARTIAL) — composed here; the kit spells none and reads none."""
    pre = stack.ENV_PREFIX + "BIG_"
    return {"lever": pre + "COND_SHARD", "setting": pre + "TRANSITION_SHARD_ROWS", "line": pre + "LINE", "allow_partial": pre + "ALLOW_PARTIAL", "bogus": pre + "BOGUS"}[which]


@pytest.mark.parametrize("which,value", [("lever", "0"), ("setting", "128"), ("line", "big"), ("allow_partial", "1"), ("bogus", "1")])
def test_a_callers_memory_lever_name_is_an_undeclared_variable(box, capsys, monkeypatch, which, value):
    """Any memory-lever switch name in the caller's environment — a lever's, a setting's, the registry's line word, its census opt-out, or a
    made-up one under the same stem — is refused by name like every other AF3_JAX_* name the package does not declare: `pred` and `check` exit 3,
    NOT ACTIVE, before anything resolves or launches."""
    name = _switch_name(which)
    assert name.startswith(stack.ENV_PREFIX) and stack.undeclared_env({name: value}) == [name] and name not in stack.DECLARED_ENV
    monkeypatch.setenv(name, value)
    inp = box.input_json("envname", seeds=(1,)); out = os.path.join(box.root, "o_envname")
    for argv in (["check", "--variant", "p2", "--mode", "big", "--n_est", "4000"],
                 ["pred", "--variant", "p2", "--mode", "big", "--n_est", "4000", "--json_path", inp, "--output_dir", out],
                 ["pred", "--variant", "p2", "--mode", "big", "--n_gpu", "2", "--json_path", inp, "--output_dir", out]):
        stack._REPORT = None; stack._LAUNCHED.clear()
        rc, txt = _run_ngpu(argv, capsys)
        assert rc == 3 and "NOT ACTIVE" in txt and "undeclared" in txt and name in txt, (argv, rc, txt)
    assert not os.path.exists(out) and not os.path.exists(box.record)         # nothing launched, nothing written


def test_active_and_census_lines_parse_in_the_core_grammar():
    registry, compose, big_launch, big_levers = _mem()
    a = big.parse_active(OK)
    assert a["levers"] == tuple(l for l in LINE if l.upper() not in modes.BIG_NOT_COMPOSED) and a["off"] == "samples_per_pass,logits_shard" and a["base"] == "fast" and a["line"] == "big" and a["refused"] == "none" and a["drop"] == "none" and a["n_gpu"] == "1"
    c = big.parse_census(CENSUS)
    assert c["applied"] == tuple(l for l in LINE if l.upper() not in modes.BIG_NOT_COMPOSED) and c["refused"] == () and c["verdict"] == "ok" and c["traced"]["cond_shard"] == 2
    assert big.parse_active("[af3-jax-opt] ACTIVE mode=big variant=p2 script=run_alphafold_fast.py") is None   # the wrapper's own activation line is not the launcher's
    assert all(modes.is_evidence_line(x) for x in (OK, CENSUS, LEVERS)) and not modes.is_evidence_line("I0902 some model log line")
    from opt_core.mem import record as core_record                          # the ACTIVE line is the core record's own: compose one here and parse it back
    sel = registry.selection(LINE)
    rec = core_record.AppliedRecord.from_selection(sel, prefix=big.PREFIX, base="fast")
    parsed = big.parse_active(rec.active_line("af3-jax-opt", line="big", n_gpu=1))
    assert parsed is not None and parsed["levers"] == () and parsed["base"] == "fast" and parsed["line"] == "big" and parsed["refused"] == "none"


def test_evidence_composes_the_launcher_lines_with_the_base():
    registry, compose, big_launch, big_levers = _mem()
    assert big.lever_evidence([OK, CENSUS])["ok"]
    ev = modes.lever_evidence("big", [OK, CENSUS, LEVERS, L1, SERVED_OFF])
    assert ev["ok"] and ev["reason"] is None and [d["state"] for d in ev["per_lever"].values()] == ["on"] * 13
    assert ev["per_lever"]["TRIMUL_CHUNK"]["evidence"] == "applied traced=2" and ev["per_lever"]["GLUT"]["state"] == "on" and ev["per_lever"]["DATTN"]["evidence"] == "dattn=10 sites=diffusion:2,pairformer_single:8"
    nod = modes.lever_evidence("big", [OK, CENSUS, LEVERS, L1, SERVED_OFF.replace("dattn=10", "dattn=off")])
    assert not nod["ok"] and "DATTN not traced: dattn=off" in nod["reason"] and nod["per_lever"]["DATTN"]["state"] == "skipped"   # the inherited lever: traced or levers_short, never silent
    lines = report.lever_lines("big", ev["per_lever"])                    # one LEVER line per lever of the composition in the core's pinned schema, canonical strategy ids
    assert len(lines) == 13 and all(ln.startswith("[af3-jax-opt] LEVER name=") for ln in lines)
    assert sum("name=F7.chunked_eval state=on impl=jax origin=kit" in ln for ln in lines) == 3 and any("lever=TRIMUL_CHUNK mode=big" in ln for ln in lines)
    assert not modes.lever_evidence("big", [OK, CENSUS, LEVERS, L1])["ok"] and "DATTN not traced: dattn=absent" in modes.lever_evidence("big", [OK, CENSUS, LEVERS, L1])["reason"]   # no SERVED line: the inherited DATTN has no evidence — levers_short by name
    NOD = ["FIX1", "GLUT", "ATTNCFG"] + [l.upper() for l in LINE] + ["L1"]
    assert modes.lever_evidence("big", [OK, CENSUS, LEVERS, L1], NOD)["ok"]   # a composition without DATTN (e.g. --n_gpu > 1 supersedes it): exact's evidence + the census suffice
    assert not modes.lever_evidence("big", [OK, CENSUS, L1])["ok"]         # the fallback kernels' LEVERS line is required too
    no_census = modes.lever_evidence("big", [OK, LEVERS, L1])
    assert not no_census["ok"] and "no BIG census line" in no_census["reason"] and no_census["per_lever"]["COND_SHARD"]["state"] == "skipped"
    partial = modes.lever_evidence("big", [OK, CENSUS.replace("verdict=ok", "verdict=partial").replace("refused=none", "refused=cond_shard"), LEVERS, L1])
    assert not partial["ok"] and "census verdict=partial" in partial["reason"] and partial["per_lever"]["COND_SHARD"]["state"] == "skipped"
    served = modes.lever_evidence("big", [OK, CENSUS, LEVERS, L1, SERVED_ON])
    assert not served["ok"] and "disengaged fast kernels served: FPF_TRIMUL served fused=48" in served["reason"]   # a disengaged fast kernel that served = a defect, named
    assert not big.lever_evidence([OK.replace("refused=none", "refused=cond_shard"), CENSUS])["ok"]
    assert not big.lever_evidence([OK.replace("line=big", "line=fpf"), CENSUS])["ok"]
    short = OK.replace("big:transition_shard,cond_shard,trimul_chunk", "big:transition_shard")
    assert "selection" in big.lever_evidence([short, CENSUS])["reason"]
    assert big.lever_evidence([short, CENSUS.replace("applied=transition_shard,cond_shard,trimul_chunk", "applied=transition_shard")], expected=["TRANSITION_SHARD"])["ok"]   # the expectation is the composition handed down
    assert not big.lever_evidence([LEVERS])["ok"] and "did not run" in big.lever_evidence([LEVERS])["reason"]
    tp_ok = OK.replace("big:transition_shard,cond_shard,trimul_chunk", "big:transition_shard").replace("off=samples_per_pass,logits_shard", "off=samples_per_pass,logits_shard,cond_shard,trimul_chunk").replace("n_gpu=1", "n_gpu=2 sharding=rowpair")   # --n_gpu 2: the superseded levers off by their own flags, SAMPLES_PER_PASS off by name as on one GPU (modes.with_n_gpu, 0.3.38)
    tp_census = CENSUS.replace("applied=transition_shard,cond_shard,trimul_chunk", "applied=transition_shard")
    assert big.parse_active(tp_ok)["off"] == "samples_per_pass,logits_shard,cond_shard,trimul_chunk" and big.lever_evidence([tp_ok, tp_census], expected=["TRANSITION_SHARD"])["ok"]
    assert not big.lever_evidence([tp_ok, tp_census])["ok"]                # with no expectation handed down the full lever set is expected


def test_evidence_at_n_gpu_2_expects_the_one_gpu_lines_memory_levers_kept():
    """Under --n_gpu P > 1 modes.with_n_gpu composes NO extra memory lever (BIG_NGPU_LEVERS is empty: SAMPLES_PER_PASS stays off by name as on
    one GPU); the gate's expectation on the launcher's ACTIVE line is the one-GPU line's memory levers minus the ones the row-sharded stack
    supersedes — transition_shard alone. An ACTIVE line that still applies samples_per_pass at P = 2 is one lever MORE than the selection:
    levers_short by name, as is a kept lever genuinely absent; P = 1 is unchanged."""
    registry, compose, big_launch, big_levers = _mem()
    P_ = "[af3-jax-opt]"
    res2 = modes.with_n_gpu(modes.resolve("big", "/c"), 2)
    assert [l for l in res2["levers"] if l in set(modes.KIT_MODES["big"]["levers"]) | set(modes.BIG_NGPU_LEVERS) | set(modes.BIG_NOT_COMPOSED)] == ["TRANSITION_SHARD"]
    x2_ok = OK.replace("big:transition_shard,cond_shard,trimul_chunk", "big:transition_shard").replace("off=samples_per_pass,logits_shard", "off=samples_per_pass,logits_shard,cond_shard,trimul_chunk").replace("n_gpu=1", "n_gpu=2 sharding=rowpair")
    x2_census = CENSUS.replace("applied=transition_shard,cond_shard,trimul_chunk", "applied=transition_shard").replace("n_gpu=1", "n_gpu=2")
    levers1 = LEVERS.replace("L-ATTNCFG+L-GLUT", "L-ATTNCFG")               # GLUT superseded by ROWPAIR at P > 1 (modes.ROWPAIR_SUPERSEDES)
    inst = f"{P_} ROWPAIR installed n_gpu=2 sites=31 heads=sharded conf=sharded diffusion=sharded schedule=gather library=ab12cd34"
    fam_on = f"{P_} LEVER name=rowpair state=on impl=opt_core.mem.rowpair_jax@0.4.1 origin=core n_gpu=2 sharding=rowpair xla_peak_gb_max=31.5 schedule=gather kernel=xla sites=trunk,model,heads"
    x2 = [x2_ok, x2_census, levers1, L1, SERVED_OFF, inst, fam_on]
    ev = modes.lever_evidence("big", x2, res2["levers"], n_gpu=2)
    assert ev["ok"] and ev["reason"] is None, ev["reason"]
    assert "SAMPLES_PER_PASS" not in ev["per_lever"] and ev["per_lever"]["TRANSITION_SHARD"]["state"] == "on" and all(d["state"] == "on" for d in ev["per_lever"].values()), ev["per_lever"]
    withsp = modes.lever_evidence("big", [x2_ok.replace("big:transition_shard", "big:samples_per_pass,transition_shard"), x2_census.replace("applied=transition_shard", "applied=samples_per_pass,transition_shard"), levers1, L1, SERVED_OFF, inst, fam_on], res2["levers"], n_gpu=2)
    assert not withsp["ok"] and withsp["reason"] == "levers active=samples_per_pass,transition_shard: the selection is transition_shard"   # the pre-0.3.38 launcher's composition: one lever more than the selection, levers_short by name
    nokept = modes.lever_evidence("big", [x2_ok.replace("big:transition_shard", "big:"), x2_census, levers1, L1, SERVED_OFF, inst, fam_on], res2["levers"], n_gpu=2)
    assert not nokept["ok"]                                                   # the kept memory lever missing from the ACTIVE line: levers_short too
    ev1 = modes.lever_evidence("big", [OK, CENSUS, LEVERS, L1, SERVED_OFF])   # P = 1: the one-GPU line's expectation, unchanged
    assert ev1["ok"] and "SAMPLES_PER_PASS" not in ev1["per_lever"]
    assert not modes.lever_evidence("big", [x2_ok.replace("n_gpu=2 sharding=rowpair", "n_gpu=1"), x2_census, LEVERS, L1, SERVED_OFF])["ok"]   # P = 1 with the P = 2 words: not the one-GPU selection (cond_shard, trimul_chunk missing)


def _def_params(path, qualname):
    """The positional/keyword parameter names of ``def <name>`` (module level) or ``def <method>`` of ``<Class>.<method>`` in the file at
    ``path`` (AST-only, no import needed) -- None if the file or the def is absent. Existence-and-shape only, never a content digest:
    stock/src is byte-pinned to the upstream release (stock/PINS.json) and never edited in place, so drift can only happen through a
    deliberate, reviewed pin-bump commit -- this is a canary that big_levers.py's own transcription still targets a same-shaped def,
    not a content check."""
    if not os.path.isfile(path):
        return None
    src = open(path, encoding="utf-8").read()
    owner, _, meth = qualname.partition(".")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and not meth and node.name == owner:
            return tuple(a.arg for a in node.args.args)
        if isinstance(node, ast.ClassDef) and meth and node.name == owner:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == meth:
                    return tuple(a.arg for a in sub.args.args)
    return None


def test_transcription_bases_are_the_stock_sources():
    """The transcriptions (the chunked sampler, the sharded conditioning, the sharded transformer pair path, the row-blocked
    TriangleMultiplication) each hand-copy a named stock/hoist/GLUT function; this checks the named function still EXISTS with the
    expected parameter shape (not a content digest -- doctrine: no digests of in-repo content, at runtime or in tests. Staleness
    protection instead comes from stock/src being byte-pinned to the upstream release and never edited in place, so the transcription
    base can only change through a deliberate, reviewed pin-bump commit)."""
    assert _def_params(STOCK_DH, "sample") == ("denoising_step", "batch", "key", "config")
    assert _def_params(STOCK_DH, "DiffusionHead._conditioning") == ("self", "batch", "embeddings", "noise_level", "use_conditioning")
    assert _def_params(HOIST, "_pair_conditioning") == ("self", "batch", "embeddings", "use_conditioning")
    assert _def_params(os.path.join(STOCK_NET, "featurization.py"), "create_relative_encoding") == ("seq_features", "max_relative_idx", "max_relative_chain")
    assert _def_params(STOCK_DH, "no_such_function") is None
    assert _def_params(os.path.join(STOCK_NET, "diffusion_transformer.py"), "Transformer.__call__") == ("self", "act", "mask", "single_cond", "pair_cond")
    assert _def_params(os.path.join(STOCK_NET, "modules.py"), "TriangleMultiplication.__call__") == ("self", "act", "mask")
    assert _def_params(os.path.join(PKG, "..", "forward", "pallas_addon", "patches", "af3_pallas_levers.py"), "make_patched_trimul_class") == ("modules",)


def test_registered_levers_declare_their_contract():
    registry, compose, big_launch, big_levers = _mem()
    for name in big_levers.LINE:
        lv = registry.LEVERS[name]
        assert lv.exact == "measured" and lv.frameworks == ("jax",) and lv.settings and lv.applies is not None and lv.module.endswith("big_levers")   # registered by this kit's module (af3_jax_opt.big_levers here; big_levers in the model process)
        assert set(big_levers.SETTINGS[name]) == set(lv.settings)          # the kit's defaults name exactly the declared settings
    assert registry.LEVERS["samples_per_pass"].family == "ckpt"              # this engine's JAX transcription replaces the core's torch lever of the same name (opt_core.mem.ckpt)
    assert big.gate() is None


def test_n_gpu_is_read_by_name():
    registry, compose, big_launch, big_levers = _mem()
    assert big_launch.ENV_N_GPU == stack.ENV_N_GPU == modes.ENV_N_GPU     # the axis variable: the wrapper sets it from --n_gpu (modes.with_n_gpu), the launcher reads it
    assert big_launch.n_gpu_requested({}) == 1 and big_launch.n_gpu_requested({"AF3_JAX_N_GPU": "4"}) == 4
    for bad in ("0", "-1", "two"):
        with pytest.raises(SystemExit) as e:
            big_launch.n_gpu_requested({"AF3_JAX_N_GPU": bad})
        assert "AF3_JAX_N_GPU" in str(e.value.code) and "(rc 2)" in str(e.value.code)
    res2 = modes.with_n_gpu(modes.resolve("big", "/c"), 2)                # P=2: the superseded memory levers leave the composition, named on the launcher's --off argument; the evidence expects exactly the rest
    memory = [l for l in res2["levers"] if l in modes.KIT_MODES["big"]["levers"]]
    assert "SAMPLES_PER_PASS" not in res2["levers"] and modes.BIG_NGPU_LEVERS == []   # 0.3.38: nothing composed back under P > 1 (SAMPLES_PER_PASS stays off by name, as on one GPU)
    assert memory == ["TRANSITION_SHARD"] and res2["launcher"][-2:] == [modes.OFF_ARG, "samples_per_pass,logits_shard,cond_shard,trimul_chunk"] and res2["launcher"][:-2] == modes.resolve("big", "/c")["launcher"][:-2]   # P > 1 rewrites the one --off word: the not-composed levers + the superseded memory levers
    assert not any("BIG" in k for k in res2["env"]) and not hasattr(__import__("af3_jax_opt._autoload", fromlist=["_"]), "PROTOCOL_ENV")   # no switch name in the ×P model process's environment; the start-up hook has no exemption list


def test_the_launchers_argv_selection_is_the_line_minus_off(box, capsys):
    """The wrapper→model-process hand-off is the launcher's command line alone: with_n_gpu names the memory levers it switches off on
    ``--off``, big_launch.parse_argv reads them back, and the switch mapping the launcher hands opt_core.mem.apply
    (``switches={lever: False}``) resolves to the line minus those levers, in order, settings the kit's; at one device nothing is named and the
    selection is the mode's one lever set. Nothing of the model process's environment is read for it."""
    registry, compose, big_launch, big_levers = _mem()
    pre = modes.resolve("big", "/c")["launcher"]
    script, flags = os.path.join(box.root, "repo", "run_alphafold_fast.py"), ["--json_path=/x/a.json", "--output_dir=/x/o"]
    got = big_launch.parse_argv(pre + [script] + flags)
    assert got[3:7] == ("big", "fast", ("samples_per_pass", "logits_shard"), script) and got[7] == flags   # the one-GPU line's --off word
    x2 = modes.with_n_gpu(modes.resolve("big", "/c"), 2)["launcher"]
    got = big_launch.parse_argv(x2 + [script] + flags)
    assert got[5:7] == (("samples_per_pass", "logits_shard", "cond_shard", "trimul_chunk"), script) and got[7] == flags   # 0.3.38: P > 1 keeps samples_per_pass off by name too
    core = pre[:-2]                                                                                         # the prefix without its --off word
    for bad in (core + ["--off"], core + [modes.OFF_ARG, "cond_shard"], core[:-2] + [script], core + ["--allow-partial", script]):   # no opt-out word in the grammar
        with pytest.raises(SystemExit):
            big_launch.parse_argv(bad)
    with pytest.raises(SystemExit) as e:
        big_launch.lever_switches(("nonesuch",), registry)
    assert "--off names no registered lever: nonesuch" in str(e.value.code) and "(rc 2)" in str(e.value.code)
    off = got[5]
    switches = big_launch.lever_switches(off, registry)
    assert switches == {n: False for n in off} and big_launch.lever_switches((), registry) == {}
    sel = registry.selection(big_levers.LINE, switches=switches)
    assert tuple(sel.levers) == tuple(l for l in big_levers.LINE if l not in off) == ("transition_shard",) and tuple(sel.off_by_flag) == off   # 0.3.38: the P > 1 selection = the one-GPU line's memory levers the row-sharded stack keeps
    assert tuple(sel.on_by_flag) == () and not sel.refusals and sel.allow_partial is False
    with pytest.raises(TypeError):
        registry.selection(big_levers.LINE, big.PREFIX, switches)                # the retired positional (prefix, environ) form: loud, never a silent read
    one = registry.selection(big_levers.LINE)
    assert tuple(one.levers) == big_levers.LINE == big.selection()["levers"] and tuple(one.off_by_flag) == ()
    inp = box.input_json("apz", seeds=(1,)); out = os.path.join(box.root, "o_apz")
    rc, txt = _run_ngpu(["pred", "--variant", "p2", "--mode", "fast", "--json_path", inp, "--output_dir", out], capsys)
    rec = box.stub_record()
    assert rec["launcher"] == "fpf_launch.py" and rec["argv"][1].endswith(modes.FPF_LAUNCHER) and "--allow-partial" not in rec["argv"] and not any("PARTIAL" in k for k in rec["env"] if k.startswith("AF3"))   # the launcher's argv is the fixed prefix: nothing of the caller rides on it


def test_activation_gates_and_records_the_lever_set(box):
    _mem()
    rep = stack.check("big", "p2")
    assert rep["mode"] == "big" and rep["big"]["base"] == "fast" and rep["big"]["levers"] == LINE and rep["big"]["off"] == () and rep["cache_class"] == "__big" + big.cache_suffix()
    assert rep["levers"] == modes.levers_of("big") and rep["launcher"] == ["big_launch.py"]
    assert stack.check("exact", "p2")["reason"] is None or "BIG" not in str(stack.check("exact", "p2")["reason"])


def test_region_rule_is_decided_up_front():
    """The memory mode's two regions (modes.big_region): at or below the memory line's size boundary (modes.BIG_N_STAR padded tokens) on one
    device the fast line lever for lever; above it, under --n_gpu > 1, or when the size cannot be read, the memory levers' program."""
    tile, n_star = modes.kernel_tile(), modes.BIG_N_STAR
    assert n_star % tile == 0 and n_star == 1408 and modes.n_star_for_card(81559) == 1408 and modes.n_star_for_card(40536) == 1408 and modes.n_star_for_card(None) == 1408
    r = modes.big_region(400)
    assert (r["region"], r["n_padded"], r["n_star"]) == ("fast", 448, n_star) and "pads to 448 <= n_star=1408" in r["rule"]
    assert modes.big_region(n_star)["region"] == "fast" and modes.big_region(n_star - 63)["n_padded"] == n_star
    assert modes.big_region(n_star + 1)["region"] == "reach" and modes.big_region(n_star + 1)["n_padded"] == n_star + tile
    assert modes.big_region(400, n_gpu=2)["region"] == "reach" and "n_gpu=2" in modes.big_region(400, n_gpu=2)["rule"]
    assert modes.big_region(None)["region"] == "reach" and "not readable" in modes.big_region(None)["rule"]
    for reg in (modes.big_region(400), modes.big_region(4000), modes.big_region(None), modes.big_region(400, n_gpu=2)):
        assert_no_markers(f"{report.PREFIX} {modes.region_note(reg)}")            # the NOTE line carries no log-scan failure marker in either region
    assert modes.effective_mode("big", "fast") == "fast" and modes.effective_mode("big", "reach") == "big" and modes.effective_mode("exact", "fast") == "exact"
    note = modes.region_note(modes.big_region(400))
    assert note.startswith("BIG NOTE region=fast n_est=400 n_star=1408: memory levers transition_shard,cond_shard,trimul_chunk inactive by size") and " " not in note.split(":")[0].split("NOTE ")[1].replace(" n_est", "").replace(" n_star", "")
    assert modes.region_note(modes.big_region(4000)).startswith("BIG NOTE region=reach n_est=4000 n_star=1408: memory levers engaged") and "FPF_HOIST,FPF_TRIMUL,HOIST_LOGITS,TRIMUL_CD inactive by property" in modes.region_note(modes.big_region(4000)) and "FPF_TRIATT+TTR kept" in modes.region_note(modes.big_region(4000))
    assert "FPF_HOIST,FPF_TRIATT,FPF_TRIMUL,HOIST_LOGITS,TRIMUL_CD,TTR inactive by property" in modes.region_note(modes.big_region(6000)) and "kept" not in modes.region_note(modes.big_region(None))


def test_region_fast_is_the_fast_line_lever_for_lever(box):
    """big resolved in region fast IS the fast line: launcher, levers, flags (kernel_tile padding), environment, cache class, evidence rules —
    labelled mode big; region reach is the memory levers' composition (today's), byte for byte."""
    _mem()
    f, bf, br = modes.resolve("fast", "/c"), modes.resolve("big", "/c", region="fast"), modes.resolve("big", "/c", region="reach")
    assert {k: v for k, v in bf.items() if k not in ("mode", "base", "disengaged", "region", "env")} == {k: v for k, v in f.items() if k not in ("mode", "base", "disengaged", "region", "env")}
    words = set(modes.TIER_WORD_SWITCHES.values())                                                                            # the provider's tier word is the mode's own in both regions: big names big
    assert bf["env"] == f["env"] and all(f["env"][w] in ("fast", "1") for w in words if w in f["env"])   # D1: region fast is the fast line — the same words (fast) on every provider face
    assert (bf["mode"], bf["base"], bf["region"], bf["disengaged"]) == ("big", "fast", "fast", {}) and bf["launcher"][0].endswith("fpf_launch.py")
    assert br == modes.resolve("big", "/c") and br["region"] == "reach" and br["launcher"][0].endswith("big_launch.py") and set(modes.BIG_DISENGAGED) == {"FPF_TRIMUL", "FPF_HOIST", "HOIST_LOGITS", "ATOM_COND_HOIST", "TRIMUL_CD"} and set(modes.BIG_DISENGAGED_AT_EDGE) == {"FPF_TRIATT", "TTR", "COND_SHARE"}
    dis = set(modes.big_disengaged(None))                                                                                       # size unknown: both sets
    assert not dis & set(br["levers"]) and dis <= set(bf["levers"])            # region fast ⊇ the fast line's levers; region reach (size unknown) engages none of them
    assert modes.cache_class_of("big", region="fast") == modes.KIT_MODES["fast"]["cache_class"] == "__fast" and modes.cache_class_of("big").startswith("__big-big-")
    assert [t for t in bf["flags"] if t.startswith("--buckets=")] == [t for t in br["flags"] if t.startswith("--buckets=")] == modes.pad_flags("fast")   # ONE padding in both regions
    served = "[af3-jax-opt] SERVED trimul fused=3 fallback=0 triatt fused=2 fallback=0 fallback_shapes=none hoist=1 tiles=own:9.0 cc=9.0 dattn=5 dattn_sites=diffusion:5 ttr=3 ttr_routed=1 ttr_fallback=none sbf16=16 sbf16_sites=atom:12,token:4 txla=24 txla_rows=cuda_sm90a:20,k2b_aot:4 txla_aside=none atomattn=7 atomattn_sites=diffusion_decoder:3,diffusion_encoder:3,evoformer_encoder:1 atomattn_fallback=none hlog=2 hlog_dtype=float32 cshare=2 achoist=2 achoist_sites=enc:2,dec:2,passed:2 achoist_aside=none cnoise=off cnoise_rule=none tcd=24 tcd_rows=cd_trimul:24 tcd_aside=none tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_uncovered=af3_tmpl_c64_ch64_incoming:2" + "\n[af3-jax-opt] LNP served=150 word=fast rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4 routed=adaptive:40,axis:2,channels:12 aside=none uncovered=none"
    inst = "af3_flashpairformer: mode=both (TriangleMultiplication=FlashTriangleMultiplication, GridSelfAttention=FlashGridSelfAttention), jax 0.10.2 backend gpu"
    L1 = "Featurisation prefetch enabled: 3 worker process(es), 4 item(s) ahead.\nOutput writer enabled: result extraction and output writing run on one writer thread behind the next fold job."   # the two row levers' lines (L1, WRITER) as one transcript element
    assert modes.lever_evidence("big", [inst, L1, served], bf["levers"], region="fast") == modes.lever_evidence("fast", [inst, L1, served], f["levers"])
    rep = stack.check("big", "p2", region=modes.big_region(400))                        # a cold fast class: named (partial_conditions), active (the same rule as fast)
    assert rep["active"] and rep["partial_conditions"] == ["cold_cache"] and rep["mode"] == "big" and rep["region"]["region"] == "fast" and rep["levers"] == f["levers"] and rep["launcher"] == ["fpf_launch.py"] and rep["cache_class"] == "__fast"
    assert rep["cache_dir"] == stack.cache_dir(rep["key"], "fast") and report.activation_line(rep).endswith(" region=fast n_est=400 n_star=1408")


def test_token_estimate_reads_the_fold_input(tmp_path):
    """inputs.token_estimate: polymer residues per copy, ligand SMILES heavy atoms, a CCD ligand a named constant per code; the largest input of
    a directory; an unreadable input makes the estimate None (the region rule then names it)."""
    from af3_jax_opt import inputs
    from .conftest import ligand
    a = render("a", [chain("A", "MKTAYIAKQR"), {"protein": {"id": ["B", "C"], "sequence": "ACDE"}}, {"ligand": {"id": "L", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}}, ligand("M", ["ATP", "MG"])])
    (tmp_path / "a.json").write_text(json.dumps(a)); (tmp_path / "b.json").write_text(json.dumps(render("b", [chain("A", "M" * 500)])))
    est = inputs.token_estimate(str(tmp_path / "a.json"))
    assert est == {"n_est": 10 + 2 * 4 + 13 + 2 * inputs.LIGAND_CCD_TOKENS, "files": 1, "per_file": {"a.json": 231}, "unreadable": []} and inputs.smiles_heavy_atoms("[NH4+].[Cl-]") == 2
    assert inputs.token_estimate(str(tmp_path))["n_est"] == 500 and inputs.token_estimate(str(tmp_path))["files"] == 2
    (tmp_path / "c.json").write_text(json.dumps([{"name": "x"}]))                                                   # the alphafoldserver list dialect: not readable as an alphafold3 fold input
    est = inputs.token_estimate(str(tmp_path))
    assert est["n_est"] is None and est["unreadable"] and est["unreadable"][0].startswith("c.json:") and modes.big_region(est["n_est"])["region"] == "reach"


# --- merged from test_big_line.py (file consolidation, tests unchanged) ---

def _run(argv, capsys):
    rc = cli.main(argv)
    out = capsys.readouterr()
    return rc, out.err, out.out


def _reset():
    stack._REPORT = None
    stack._LAUNCHED.clear()


def _input_dir(root: str, name: str, n_residues: int) -> str:
    """A directory holding ONE alphafold3-dialect fold input of `n_residues` protein residues (token estimate = n_residues), one seed."""
    d = os.path.join(root, f"in_{name}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}.json"), "w") as f:
        json.dump(render(name, [chain("A", "M" * n_residues)], seeds=(1,)), f)
    return d


AT_N_STAR = modes.BIG_N_STAR - 8                                       # 1400: pads to exactly n_star (1408) by the kernel tile — the boundary is region fast
SIZES = [("below", 400, None, "fast"), ("at_n_star", AT_N_STAR, None, "fast"), ("exactly_n_star", modes.BIG_N_STAR, None, "fast"),
         ("above", modes.BIG_N_STAR + 1, None, "reach"), ("large", 4000, None, "reach")]   # (label, n_est, -, region): the size rule alone decides the region


def _mem_guard():
    pytest.importorskip("opt_core.mem.registry", reason="the memory mode needs the core's lever registry (opt_core >= 0.4.0: opt_core.mem.registry)")


@pytest.mark.parametrize("label,n_est,forced,region", SIZES, ids=[s[0] for s in SIZES])
def test_the_resolver_decides_region_program_and_class(box, label, n_est, forced, region):
    """effective_line by --n_est and by an input of that size agree; region fast = the fast line's class and program, region reach = the
    memory levers' class (`__big-big-<sha8>`, cache_class_of's) and program; the boundary input (pads to exactly n_star) is region fast."""
    _mem_guard()
    by_flag = modes.effective_line("big", n_est=n_est)
    by_input = modes.effective_line("big", input_path=_input_dir(box.root, label, n_est))
    assert by_flag["region"] == by_input["region"] == region and by_flag["cache_class"] == by_input["cache_class"] and by_flag["mode_effective"] == by_input["mode_effective"]
    assert by_input["record"]["estimate"]["n_est"] == n_est and by_input["record"]["estimate"]["files"] == 1 and by_flag["record"]["estimate"]["given_by"] == "--n_est"
    if region == "fast":
        assert by_flag["cache_class"] == modes.KIT_MODES["fast"]["cache_class"] == "__fast" and by_flag["mode_effective"] == "fast"
    else:
        assert by_flag["cache_class"] == modes.cache_class_of("big") and re.fullmatch(r"__big-big-[0-9a-f]{8}", by_flag["cache_class"]) and by_flag["mode_effective"] == "big"
    if label == "at_n_star":
        assert by_flag["record"]["n_padded"] == modes.BIG_N_STAR
    assert modes.effective_line("big", record=by_input["record"]) == by_input            # activate's path: a resolved record re-derives the same line, nothing re-read


@pytest.mark.parametrize("label,n_est,forced,region", SIZES, ids=[s[0] for s in SIZES])
def test_warm_builds_the_class_pred_reads_for_the_same_input(box, capsys, label, n_est, forced, region):
    """END TO END on the stub box: `warm --mode big --input_dir D` then `pred --mode big --input_dir D` — warm's class directory IS the
    --cache_dir pred's model process runs with, pred's activation finds the class warm (no cold_cache refusal), and both equal the class of
    `check --mode big --n_est <n>`; in region fast that directory is fast's own class (a `warm --mode fast` serves it too)."""
    _mem_guard()
    d = _input_dir(box.root, label, n_est)
    extra = []                                                            # the command line carries no region force: the size rule decides
    rc, err, _ = _run(["warm", "--variant", "p2", "--mode", "big", "--input_dir", d, *extra], capsys)
    assert rc == (0 if region == "fast" else 3) and " WARM " in err, err               # the stub launcher of the reach program prints no lever evidence: WARM PARTIAL (levers_short, rc 3 by name) — the class is built all the same, which is the point here
    wline = [ln for ln in err.splitlines() if " WARM " in ln][0]
    warm_cache = re.search(r" cache=(\S+) ", wline).group(1)
    act = [ln for ln in err.splitlines() if " ACTIVE mode=big" in ln][0]
    assert act.endswith(f" region={region} n_est={n_est} n_star={modes.BIG_N_STAR}") and err.count(f"BIG NOTE region={region} ") == 1   # warm prints pred's region tokens and pred's ONE NOTE line
    warm_flags = box.stub_record()["flags"]
    assert f"--cache_dir={warm_cache}" in warm_flags and f"--input_dir={d}" in warm_flags
    _reset()
    out = os.path.join(box.root, f"out_{label}")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "big", "--input_dir", d, "--output_dir", out, *extra], capsys)
    assert "NOT ACTIVE" not in err and "cold_cache" not in err, err                       # the defect this guards: warm's class was not the class pred read at or below n_star → cold_cache, rc 3
    pact = [ln for ln in err.splitlines() if " ACTIVE mode=big" in ln][0]
    assert pact.endswith(f" region={region} n_est={n_est} n_star={modes.BIG_N_STAR}")
    assert f"--cache_dir={warm_cache}" in box.stub_record()["flags"]                       # key(warm) == key(pred)
    if region == "fast":
        assert rc == 0, err
        assert warm_cache == box.cache_dir("fast") and box.stub_record()["launcher"] == "fpf_launch.py"
    else:
        assert warm_cache == box.cache_dir("big") and box.stub_record()["launcher"] == "big_launch.py"
    _reset()
    rc, err, _ = _run(["check", "--variant", "p2", "--mode", "big", "--n_est", str(n_est), *extra], capsys)             # check by --n_est: the same class, now warm
    assert rc == 0 and f"cache={warm_cache} " in err and "partial=" not in err, err


def test_warm_big_with_nothing_to_size_is_refused_by_name(box, capsys):
    """`warm --mode big` with no --input_dir and no --n_est: refused BY NAME (rc 2, modes.BIG_UNSIZED) before anything
    activates or is written — never the kit's default inputs sized silently, never a class warmed blind; --n_est / --input_dir each
    size it; the Python API raises modes.BigUnsized (a ValueError)."""
    _mem_guard()
    rc, err, _ = _run(["warm", "--variant", "p2", "--mode", "big"], capsys)
    assert rc == 2 and modes.BIG_UNSIZED in err and "big: the program depends on the input size — pass --input_dir or --n_est N" in err, err
    assert " ACTIVE " not in err and " WARM " not in err and not os.path.exists(box.record)                                       # nothing activated, no model process launched
    assert not os.path.isdir(os.path.join(box.cache_root, "warm_records")) and not os.path.isdir(box.cache_dir("big")) and not os.path.isdir(box.cache_dir("fast"))
    with pytest.raises(modes.BigUnsized, match="depends on the input size"):
        warm.warm("p2", mode="big")
    with pytest.raises(ValueError):
        warm.warm("p2", mode="big")
    _reset()
    rc, err, _ = _run(["warm", "--variant", "p2", "--mode", "big", "--n_est", "400", "--allow-partial"], capsys)   # sized by --n_est: fast's class, the kit's default inputs warmed
    assert rc == 0 and f" cache={box.cache_dir('fast')} " in err and "region=fast n_est=400" in err, err
    _reset()
    _reset()
    rc, err, _ = _run(["warm", "--variant", "p2", "--mode", "fast"], capsys)                                       # every other mode: no size needed, its own class
    assert rc == 0 and f" cache={box.cache_dir('fast')} " in err and "region=" not in err


def test_the_resolver_alone_and_by_name(tmp_path):
    """effective_line's records: every other mode is itself; big with nothing to size names the size-unknown rule (check's dry run, the
    Python API) or raises under require_size (warm); n_gpu > 1 sizes it otherwise; --n_est wins over the input; an unreadable
    input is region reach by pred's rule (the class pred reads for it); an unknown mode / region / n_gpu is refused in words."""
    _mem_guard()
    assert modes.effective_line("fast") == {"mode": "fast", "mode_effective": "fast", "region": None, "record": None, "cache_class": "__fast"}
    assert modes.effective_line("exact")["cache_class"] == modes.effective_line("off")["cache_class"] == "" and modes.effective_line("exact", require_size=True)["region"] is None
    blind = modes.effective_line("big")
    assert blind["region"] == "reach" and blind["record"]["estimate"]["unreadable"] == ["no input given"] and "not readable up front" in blind["record"]["rule"] and blind["mode_effective"] == "big"
    with pytest.raises(modes.BigUnsized) as e:
        modes.effective_line("big", require_size=True)
    assert str(e.value) == modes.BIG_UNSIZED and isinstance(e.value, ValueError) and all(w in modes.BIG_UNSIZED for w in ("--input_dir", "--n_est N"))
    assert modes.effective_line("big", n_gpu=2, require_size=True)["region"] == "reach" and "n_gpu=2" in modes.effective_line("big", n_gpu=2)["record"]["rule"]
    big = _input_dir(str(tmp_path), "big", 3000)
    won = modes.effective_line("big", input_path=big, n_est=400)
    assert won["region"] == "fast" and won["record"]["estimate"] == {"n_est": 400, "files": 0, "per_file": {}, "unreadable": [], "given_by": "--n_est"}
    (tmp_path / "in_big" / "list.json").write_text(json.dumps([{"name": "x"}]))                                  # an unreadable input beside it: pred's estimator makes n_est None → reach, named
    unread = modes.effective_line("big", input_path=big, require_size=True)
    assert unread["region"] == "reach" and unread["record"]["estimate"]["unreadable"] and unread["record"]["n_est"] is None
    with pytest.raises(modes.UnsupportedMode, match="unknown mode"):
        modes.effective_line("huge")
    with pytest.raises(ValueError, match="a positive integer is required"):
        modes.effective_line("big", n_est=10, n_gpu="two")


def test_one_home_for_the_decision():
    """The resolver is the only place the verbs decide the memory mode's region, effective mode and class: warm.py, cli.py and stack.py call
    modes.effective_line and none of them calls big_region / effective_mode / the input estimator themselves; the estimator the resolver
    sizes an input with is pred's (inputs.token_estimate), read in modes.effective_line alone."""
    src = {name: open(os.path.join(PKG, name), encoding="utf-8").read() for name in ("warm.py", "cli.py", "stack.py")}
    for name, text in src.items():
        assert ".effective_line(" in text, name
        assert not re.search(r"(?<![\w_])_?modes\.big_region\(", text) and "modes.effective_mode(" not in text and "_modes.effective_mode(" not in text, name
        assert "token_estimate(" not in text, name                                                        # the one reader of the input's size is called by the resolver
    assert "cache_class_of(" not in src["warm.py"] and "cache_class_of(" not in src["cli.py"]
    body = inspect.getsource(modes.effective_line)
    assert "token_estimate(" in body and "big_region(" in body and "cache_class_of(" in body and "effective_mode(" in body and "BigUnsized" in body
    assert [a.dest for a in cli.build_parser()._subparsers._group_actions[0].choices["warm"]._actions if a.dest in ("region", "n_est")] == ["n_est"]             # warm takes pred's / check's sizing option; no region force on any verb
    for verb in ("pred", "check"):
        dests = {a.dest for a in cli.build_parser()._subparsers._group_actions[0].choices[verb]._actions}
        assert "n_est" in dests and "region" not in dests                    # one sizing option on every verb; no region force


# --- merged from test_ngpu.py (file consolidation, tests unchanged) ---

ngpu = pytest.importorskip("opt_core.mem.ngpu", reason="opt_core.mem.ngpu (the n_gpu producer) lands with opt_core >= 0.4.0; below it every verb is NOT ACTIVE producer_missing (test_core_missing)")



def test_tokens_are_the_cores_words():
    assert report.ngpu_fields(1) == "n_gpu=1 sharding=none" == ngpu.active_fields(1)
    assert report.ngpu_fields(2) == "n_gpu=2 sharding=rowpair" and report.ngpu_fields(8) == "n_gpu=8 sharding=rowpair"


def test_p1_is_the_unchanged_composition():
    for m in modes.MODES:
        res = modes.resolve(m, "/c")
        out = modes.with_n_gpu(res, 1)
        assert out["n_gpu"] == 1 and out["sharding"] == "none" and out["superseded"] == []
        assert {k: out[k] for k in res} == res                            # byte-identical composition: launcher, levers, env, cache class


@pytest.mark.parametrize("mode", [m for m in ("off", "exact", "fast") if m in modes.MODES])
def test_p_gt_1_outside_the_memory_mode_is_refused_by_name(mode):
    with pytest.raises(ngpu.NGpuRefused) as e:
        modes.with_n_gpu(modes.resolve(mode, "/c"), 2)
    assert e.value.reason == ngpu.REFUSE_MODE == "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)"


def _run_ngpu(argv, capsys):
    from af3_jax_opt import cli
    rc = cli.main(argv)
    o = capsys.readouterr()
    return rc, o.err + o.out


def test_pred_n_gpu_above_one_requires_mode_big(box, capsys):
    """`pred --n_gpu 2` under --mode exact | fast | off, and with no --mode (the default, fast): NOT ACTIVE by name — the core's sentence
    'n_gpu>1 requires --mode big' — exit 3 before any model process is launched; only --mode big takes P > 1."""
    from af3_jax_opt import stack
    inp = box.input_json("ng", seeds=(1,))
    for extra in (["--mode", "exact"], ["--mode", "fast"], ["--mode", "off"], []):
        stack._REPORT = None; stack._LAUNCHED.clear()
        out = os.path.join(box.root, "o_ng_" + (extra[1] if extra else "default"))
        rc, txt = _run_ngpu(["pred", "--variant", "p2", *extra, "--n_gpu", "2", "--json_path", inp, "--output_dir", out], capsys)
        assert rc == 3 and "NOT ACTIVE" in txt and "n_gpu>1 requires --mode big" in txt and ngpu.REFUSE_MODE in txt, (extra, rc, txt)
        assert not os.path.exists(out) and not os.path.exists(box.record), extra            # nothing launched, nothing written


def test_activation_refuses_by_name(box, capsys):
    """exact --n_gpu 2 → NOT ACTIVE with the core's sentence, rc 3; the P=1 ACTIVE line carries the resource tokens (explicit 1 == absent)."""
    from af3_jax_opt import stack
    rc, txt = _run_ngpu(["check", "--variant", "p2", "--mode", "exact", "--n_gpu", "2"], capsys)
    assert rc == 3 and "NOT ACTIVE" in txt and ngpu.REFUSE_MODE in txt, txt
    stack._REPORT = None
    rc, txt = _run_ngpu(["check", "--variant", "p2", "--mode", "off"], capsys)
    assert rc == 0 and " n_gpu=1 sharding=none" in txt, txt
    stack._REPORT = None
    rc, txt2 = _run_ngpu(["check", "--variant", "p2", "--mode", "off", "--n_gpu", "1"], capsys)
    assert rc == 0 and " n_gpu=1 sharding=none" in txt2
    stack._REPORT = None
    box.warm_cache(mode="exact")
    rc, txt = _run_ngpu(["check", "--variant", "p2", "--mode", "exact"], capsys)
    assert rc == 0 and re.search(r"ACTIVE mode=exact .* n_gpu=1 sharding=none", txt), txt


@pytest.mark.skipif(ngpu.MEMORY_MODE not in modes.MODES, reason="the memory mode's rows are not in this tree's mode table")
def test_memory_mode_p2_composition_and_visible_rule(box, capsys):
    from af3_jax_opt import stack
    res = modes.with_n_gpu(modes.resolve(ngpu.MEMORY_MODE, "/c"), 2)
    assert res["env"][modes.ENV_N_GPU] == "2" and modes.ROWPAIR in res["levers"] and res["cache_class"] == modes.resolve(ngpu.MEMORY_MODE, "/c")["cache_class"]
    assert res["launcher"][-2:] == [modes.OFF_ARG, ",".join(["samples_per_pass", "logits_shard"] + [lv.lower() for lv in res["superseded"] if lv != "GLUT" and lv not in modes.TREE_LEVER_ENV])] == [modes.OFF_ARG, "samples_per_pass,logits_shard,cond_shard,trimul_chunk"] and "AF3P_GLU_T" not in res["env"]   # 0.3.38: the not-composed levers stay off at P > 1 too
    assert "DATTN" in res["superseded"] and not any(k in res["env"] for k in modes.TREE_LEVER_ENV["DATTN"]) and "dattn" not in res["launcher"][-1]   # a superseded tree lever leaves by its own switch: never installed in the ranks, not a name on --off
    assert all(l not in res["levers"] for l in modes.ROWPAIR_SUPERSEDES) and set(res["superseded"]) <= set(modes.ROWPAIR_SUPERSEDES)
    stack._REPORT = None
    rc, txt = _run_ngpu(["check", "--variant", "p2", "--mode", ngpu.MEMORY_MODE, "--n_gpu", "2"], capsys)   # the stub box's nvidia-smi reports ONE H100
    assert rc == 3 and "refused: n_gpu=2 visible=1" in txt, txt


def test_rowpair_evidence_by_name():
    """n_gpu > 1: the adapter's install line + the family's lever line state=on = applied; a refusal, a missing install line, a missing or
    non-on family line, or an n_gpu disagreeing with the wrapper's are each a named reason (the exit rule's levers_short)."""
    P_ = "[af3-jax-opt]"
    inst = f"{P_} ROWPAIR installed n_gpu=2 sites=31 heads=sharded conf=sharded diffusion=sharded schedule=gather library=ab12cd34"
    fam_on = f"{P_} LEVER name=rowpair state=on impl=opt_core.mem.rowpair_jax@0.4.1 origin=core n_gpu=2 sharding=rowpair xla_peak_gb_max=31.5 schedule=gather kernel=xla sites=trunk,model,heads"
    ok = modes.rowpair_evidence([inst, fam_on], 2)
    assert ok["ok"] and ok["installed"] == 2 and ok["family_state"] == "on" and "heads=sharded" in ok["installed_fields"]
    assert modes.rowpair_evidence([inst, fam_on], 4)["reason"] == "ROWPAIR installed n_gpu=2, the wrapper asked 4"
    assert modes.rowpair_evidence([f"{P_} ROWPAIR refused: install: the b21 module lacks conditioning_factory"], 2)["reason"].startswith("ROWPAIR refused: install:")
    assert "no 'ROWPAIR installed' line" in modes.rowpair_evidence([fam_on], 2)["reason"]
    assert "no family LEVER name=rowpair line" in modes.rowpair_evidence([inst], 2)["reason"]
    bad = modes.rowpair_evidence([inst, f"{P_} LEVER name=rowpair state=skipped reason=evidence_error:ValueError:x n_gpu=2 sharding=rowpair"], 2)
    assert not bad["ok"] and bad["reason"].startswith("ROWPAIR family line state=skipped")
    pl = modes.per_lever(ngpu.MEMORY_MODE if ngpu.MEMORY_MODE in modes.MODES else "fast", [modes.ROWPAIR], {"rowpair": ok}, [inst, fam_on])
    assert pl[modes.ROWPAIR]["state"] == "on" and pl[modes.ROWPAIR]["name"] == "F7.tensor_parallel"
    assert modes.per_lever("fast", [modes.ROWPAIR], {"rowpair": bad}, [])[modes.ROWPAIR] == {"name": "F7.tensor_parallel", "state": "skipped", "reason": bad["reason"], "evidence": None}


def test_n_gpu_is_fail_closed_against_the_launchers_report():
    """The n_gpu the model process REPORTS (the memory mode launcher's ACTIVE line) must equal the request: a dropped axis (requested 2,
    active 1), an unreported one, or a ROWPAIR install under P=1 are each `n_gpu_mismatch ...` — never a pass (levers_short, rc 3)."""
    P_ = "[af3-jax-opt]"
    launcher = lambda n: f"{P_} ACTIVE mode=big:samples_per_pass,transition_shard base=fast drop=none refused=none off=cond_shard on=none allocator=none line=big n_gpu={n}" + (" sharding=rowpair" if n > 1 else "")
    wrapper = f"{P_} ACTIVE mode=big variant=p2 script=run_alphafold_fast.py launcher=big_launch.py n_gpu=2 sharding=rowpair"   # the wrapper's own line never stands in for the launcher's
    inst = f"{P_} ROWPAIR installed n_gpu=2 sites=31 heads=sharded conf=sharded diffusion=sharded schedule=gather library=ab12cd34"
    assert modes.n_gpu_evidence([wrapper, launcher(2), inst], 2, True)["ok"]
    assert modes.n_gpu_evidence([wrapper, launcher(1)], 2, True)["reason"] == "n_gpu_mismatch requested=2 active=1"
    assert modes.n_gpu_evidence([wrapper], 2, True)["reason"].startswith("n_gpu_mismatch requested=2 active=unreported")
    assert modes.n_gpu_evidence([launcher(1), inst], 1, False)["reason"].startswith("n_gpu_mismatch requested=1 active=1 but ROWPAIR installed")
    assert modes.n_gpu_evidence([launcher(1)], 1, False)["ok"]
    ev = modes._lever_evidence("big", [wrapper, launcher(1)], modes.with_n_gpu(modes.resolve("big", "/c"), 2)["levers"], n_gpu=2)
    assert not ev["ok"] and "n_gpu_mismatch requested=2 active=1" in ev["reason"] and ev["n_gpu"]["active"] == 1


@pytest.mark.skipif(ngpu.MEMORY_MODE not in modes.MODES, reason="the memory mode's rows are not in this tree's mode table")
def test_requested_n_gpu_reaches_the_model_process(box, capsys, monkeypatch):
    """`pred --mode big --n_gpu 2` on a box reporting two H100s: the launch the wrapper makes carries AF3_JAX_N_GPU=2 to the model process
    (the stub fork records its environment), the superseded single-card levers off by their own flags, and the wrapper's exit rule then
    demands the launcher's n_gpu=2 + the ROWPAIR evidence — the stub prints neither, so the run is levers_short by name (rc 3), never a pass."""
    from af3_jax_opt import stack
    from .conftest import GPU_H100
    monkeypatch.setattr(stack, "gpu_info", lambda index=0: {**GPU_H100, "count": 2})
    stack._REPORT = None
    box.warm_cache(mode=ngpu.MEMORY_MODE)                                 # the memory mode's class warm (one class for every P): pred launches, the stub fork records the launch
    inp = box.input_json("n2", seeds=(1,))
    out = os.path.join(box.root, "out_big2")
    rc, txt = _run_ngpu(["pred", "--variant", "p2", "--mode", ngpu.MEMORY_MODE, "--n_gpu", "2", "--json_path", inp, "--output_dir", out], capsys)
    rec = box.stub_record()
    assert rec["launcher"] == "big_launch.py" and rec["env"].get(modes.ENV_N_GPU) == "2", rec
    i = rec["argv"].index(modes.OFF_ARG)
    assert rec["argv"][i - 2: i + 2] == ["--base", "fast", modes.OFF_ARG, "samples_per_pass,logits_shard,cond_shard,trimul_chunk"] and rec["argv"][i + 2].endswith("run_alphafold_fast.py") and "AF3P_GLU_T" not in rec["env"] and "AF3_JAX_DATTN" not in rec["env"]
    assert not any("BIG" in k for k in rec["env"])                          # the ×P model process's environment carries no memory-lever switch
    assert rc == 3 and "n_gpu_mismatch requested=2 active=unreported" in txt, txt        # the stub launcher reports no axis: fail-closed



def test_headroom_estimate_is_a_note_not_a_refusal(capsys):
    """The NCCL-headroom rule (`mesh.mem_fraction_gate` with MEM_FRACTION_CEILINGS) is a memory ESTIMATE: within its ceiling the core's record
    passes through unchanged and nothing is printed (the region served before is byte-identical); over it the adapter prints ONE
    `ROWPAIR NOTE mem_headroom: <words>; proceeding — may exhaust device memory` line and returns the record with the same two tokens plus `note` — never
    `ROWPAIR refused`, never an exit. The wrapper records the note and the evidence stays ok."""
    from af3_jax_opt.inprocess import rowpair as rp
    from opt_core.mem import MemLeverRefused
    P_ = report.PREFIX

    class Mesh:                                                            # the two RowMesh facts headroom() reads
        n_gpu, lever = 8, "rowpair"
        def pool_fraction(self): return 0.95

    class GateWithin:
        @staticmethod
        def mem_fraction_gate(rmesh, ceilings): return {"mem_fraction_ceiling": 0.9, "xla_pool_fraction": 0.85}

    class GateOver:
        @staticmethod
        def mem_fraction_gate(rmesh, ceilings):
            raise MemLeverRefused(rmesh.lever, "refused: n_gpu=8 needs the XLA pool fraction <= 0.9 (live 0.950; env XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_CLIENT_MEM_FRACTION=unset); NCCL communicators allocate outside XLA's pool")

    assert rp.headroom(GateWithin, Mesh(), MemLeverRefused) == {"mem_fraction_ceiling": 0.9, "xla_pool_fraction": 0.85}
    assert capsys.readouterr().out == ""                                   # within the ceiling: no line, the record as the core returned it
    rec = rp.headroom(GateOver, Mesh(), MemLeverRefused)                   # over: no SystemExit
    out = capsys.readouterr().out
    assert rec["mem_fraction_ceiling"] == 0.9 and rec["xla_pool_fraction"] == 0.95 and rec["note"].startswith("n_gpu=8 needs the XLA pool fraction <= 0.9")
    assert out.count("\n") == 1 and out.startswith(f"{P_} ROWPAIR NOTE mem_headroom: n_gpu=8 needs the XLA pool fraction <= 0.9") and out.rstrip().endswith("; proceeding — may exhaust device memory")
    assert "ROWPAIR refused" not in out and modes.ROWPAIR_REFUSED_RX.search(out) is None
    assert_no_markers(out)                                                # the NOTE line carries no log-scan failure marker (its estimate words are the core's, the leading `refused:` dropped)
    lines = [out.strip(), f"{P_} ROWPAIR installed n_gpu=8 sites=18 heads=sharded conf=sharded diffusion=sharded n_sites=18 library=x schedule=gather mem_fraction_ceiling=0.9 xla_pool_fraction=0.95",
             f"{P_} LEVER name=rowpair state=on impl=opt_core.mem.rowpair_jax@0 origin=core n_gpu=8 sharding=rowpair"]
    ev = modes.rowpair_evidence(lines, 8)
    assert ev["ok"] and ev["reason"] is None and ev["notes"] == ["mem_headroom: " + rec["note"] + "; proceeding — may exhaust device memory"]
    assert modes.rowpair_evidence(lines[1:], 8)["notes"] == []                # no note printed: the record of a run within the ceiling is unchanged


def test_every_size_pads_to_a_multiple_of_n_gpu():
    """The shared recipe's size contract (opt_core.mem.rowpair_jax.shard.pad_plan: every N has a plan, never a refusal): under n_gpu > 1 the
    adapter holds the runner's padded token length to a multiple of P. The fork's buckets (32 ... 5120) already are, so a bucketed input keeps its
    stock bucket; above the largest bucket the fork pads to the token count itself (6426 stays 6426) and the guard adds at most P - 1 tokens."""
    import bisect
    from af3_jax_opt import settings
    from af3_jax_opt.inprocess import rowpair as rp
    from opt_core.mem.rowpair_jax import shard
    buckets = settings.upstream_buckets()

    def stock(num_tokens, bks):                                          # the fork's rule (pipeline.calculate_bucket_size): the first bucket that holds it, else the count itself
        i = bisect.bisect_left(list(bks), num_tokens)
        return num_tokens if i == len(bks) else bks[i]

    assert stock(6426, buckets) == 6426 and stock(992, buckets) == 1024 and stock(57, buckets) == 64
    for P in (2, 4, 8):
        seen = []
        f = rp.padded_bucket_size(stock, P, on_pad=lambda n, b, padded: seen.append((n, b, padded)))
        assert f.__wrapped__ is stock and f.n_gpu == P
        for n in list(range(1, 5121, 37)) + buckets + [5121, 6114, 6426, 6427, 6432, 7824, 8000, 12345, 16384]:
            b, padded = stock(n, buckets), f(n, buckets)
            assert padded % P == 0 and b <= padded < b + P and padded == shard.pad_plan(b, P)["n_padded"], (P, n, b, padded)
            if n <= buckets[-1]:
                assert padded == b, (P, n)                               # every fork bucket divides by 2, 4 and 8: bucketed inputs are untouched
        assert (6426, 6426, 6428) in seen if P == 4 else (6427, 6427, {2: 6428, 8: 6432}[P]) in seen
        assert all(b != padded for _, b, padded in seen)                # the pad line fires only when the size moved
    assert rp.BUCKET_SITE == "alphafold3.model.pipeline.pipeline.calculate_bucket_size"
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "stock", "src", "src", "alphafold3", "model", "pipeline", "pipeline.py"), encoding="utf-8").read()
    assert "def calculate_bucket_size(" in src and "padded_token_length = calculate_bucket_size(" in src      # the site exists in the pinned stock and is called by module name


def test_pool_fraction_rule_is_decided_up_front(box, monkeypatch):
    """--n_gpu 8: the row-sharded stack's precondition (inprocess/rowpair.py MEM_FRACTION_CEILINGS {8: 0.9}, the one table, read by the wrapper)
    against the fraction the model process would resolve — the image's own XLA_CLIENT_MEM_FRACTION=0.95 is not the caller's: the kit writes 0.9
    under that name (source=kit); a caller's compliant value is kept (source=user); a caller's value over the ceiling, or both jaxlib names set,
    is a contradiction refused by name; P = 2 / 4: nothing applies."""
    from af3_jax_opt import report, stack
    image = stack.pins()["image"]["env"]
    assert modes.rowpair_mem_fraction_ceilings() == {8: 0.9} and modes.rowpair_mem_fraction_ceiling(8) == 0.9 and modes.rowpair_mem_fraction_ceiling(16) == 0.9
    assert modes.rowpair_mem_fraction_ceiling(2) is None and modes.rowpair_mem_fraction_ceiling(4) is None and image["XLA_CLIENT_MEM_FRACTION"] == "0.95"
    base = {k: v for k, v in os.environ.items() if "MEM_FRACTION" not in k}
    r = stack.xla_pool_fraction(8, dict(base, **image))                           # the container's environment = the image's variables: not the caller's
    assert (r["applies"], r["source"], r["name"], r["value"], r["env"], r["refused"]) == (True, "kit", "XLA_CLIENT_MEM_FRACTION", "0.9", {"XLA_CLIENT_MEM_FRACTION": "0.9"}, None)
    assert stack.xla_pool_fraction(8, dict(base))["env"] == {"XLA_CLIENT_MEM_FRACTION": "0.9"}    # nothing set anywhere: composed from the image, over the ceiling: written
    env = stack.model_process_env(dict(base, **image), n_gpu=8)
    assert env["XLA_CLIENT_MEM_FRACTION"] == "0.9" and "XLA_PYTHON_CLIENT_MEM_FRACTION" not in env and stack.model_process_env(dict(base, **image), n_gpu=2)["XLA_CLIENT_MEM_FRACTION"] == "0.95"
    assert stack.model_process_env(dict(base, **image))["XLA_CLIENT_MEM_FRACTION"] == "0.95"       # P = 1: the image's pool untouched
    r = stack.xla_pool_fraction(8, dict(base, XLA_CLIENT_MEM_FRACTION="0.85"))                    # the caller's own, within the precondition: kept, named source=user
    assert (r["source"], r["value"], r["env"], r["refused"]) == ("user", "0.85", {}, None) and stack.model_process_env(dict(base, XLA_CLIENT_MEM_FRACTION="0.85"), n_gpu=8)["XLA_CLIENT_MEM_FRACTION"] == "0.85"
    r = stack.xla_pool_fraction(8, dict(base, XLA_CLIENT_MEM_FRACTION="0.97"))                    # the caller's own, over the ceiling: a contradiction, refused by name
    assert r["source"] == "user" and r["refused"].startswith("n_gpu=8 needs the XLA pool fraction <= 0.9") and "XLA_CLIENT_MEM_FRACTION=0.97" in r["refused"] and r["env"] == {}
    r = stack.xla_pool_fraction(8, dict(base, XLA_PYTHON_CLIENT_MEM_FRACTION="0.8", **image))    # both jaxlib names would be set (the caller's + the image's): refused by name
    assert r["refused"].startswith("n_gpu=8 needs ONE XLA pool fraction variable") and "XLA_CLIENT_MEM_FRACTION=0.95" in r["refused"] and "XLA_PYTHON_CLIENT_MEM_FRACTION=0.8" in r["refused"]
    assert stack.xla_pool_fraction(2, dict(base, XLA_CLIENT_MEM_FRACTION="0.97")) == {"applies": False, "n_gpu": 2, "ceiling": None, "name": None, "value": None, "source": None, "env": {}, "refused": None}
    note = report.xla_pool_note(stack.xla_pool_fraction(8, dict(base, **image)))
    assert note == "[af3-jax-opt] NOTE xla_mem_fraction=0.9 source=kit name=XLA_CLIENT_MEM_FRACTION n_gpu=8 (the row-sharded stack's precondition: pool fraction <= 0.9; NCCL communicators allocate outside XLA's pool)"
    assert_no_markers(note)


def test_pool_fraction_rule_at_activation(box, capsys, monkeypatch):
    """`check --mode big --n_gpu 8` on a box reporting eight H100s: the container's image value → ACTIVE + ONE NOTE `xla_mem_fraction=0.9
    source=kit`; the caller's 0.97 → NOT ACTIVE by name (rc 3) before anything launches; `pred` carries the written fraction to the model process."""
    from af3_jax_opt import stack
    from .conftest import GPU_H100
    monkeypatch.setattr(stack, "gpu_info", lambda index=0: {**GPU_H100, "count": 8})
    monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.95"); monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)   # the container's environment: the image's own value
    stack._REPORT = None
    rc, txt = _run_ngpu(["check", "--variant", "p2", "--mode", ngpu.MEMORY_MODE, "--n_gpu", "8", "--allow-partial"], capsys)
    assert rc == 0 and "n_gpu=8 sharding=rowpair" in txt and txt.count("[af3-jax-opt] NOTE xla_mem_fraction=0.9 source=kit name=XLA_CLIENT_MEM_FRACTION n_gpu=8 ") == 1, txt
    stack._REPORT = None
    box.warm_cache(mode=ngpu.MEMORY_MODE)
    inp = box.input_json("n8", seeds=(1,))
    rc, txt = _run_ngpu(["pred", "--variant", "p2", "--mode", ngpu.MEMORY_MODE, "--n_gpu", "8", "--json_path", inp, "--output_dir", os.path.join(box.root, "out_big8")], capsys)
    rec = box.stub_record()
    assert rec["env"].get("XLA_CLIENT_MEM_FRACTION") == "0.9" and "XLA_PYTHON_CLIENT_MEM_FRACTION" not in rec["env"] and rec["env"].get(modes.ENV_N_GPU) == "8", rec["env"]
    assert "NOTE xla_mem_fraction=0.9 source=kit" in txt
    stack._REPORT = None; stack._LAUNCHED.clear()
    monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.97")                                          # the caller's own value over the ceiling
    rc, txt = _run_ngpu(["check", "--variant", "p2", "--mode", ngpu.MEMORY_MODE, "--n_gpu", "8", "--allow-partial"], capsys)
    assert rc == 3 and "NOT ACTIVE" in txt and "n_gpu=8 needs the XLA pool fraction <= 0.9" in txt and "XLA_CLIENT_MEM_FRACTION=0.97" in txt, txt
    stack._REPORT = None
    monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.88")                                          # the caller's own value within the ceiling: kept, named
    rc, txt = _run_ngpu(["check", "--variant", "p2", "--mode", ngpu.MEMORY_MODE, "--n_gpu", "8", "--allow-partial"], capsys)
    assert rc == 0 and "NOTE xla_mem_fraction=0.88 source=user name=XLA_CLIENT_MEM_FRACTION n_gpu=8 " in txt, txt


def test_samples_per_pass_composes_with_the_sampler_scope_lever():
    """SAMPLER_BF16 wraps diffusion_head.sample to raise its scope; the memory mode's SAMPLES_PER_PASS then wraps THAT: its applied-twice guard
    must not mistake the scope wrapper for its own chunked sampler, and its chunked body (which re-implements sample) must enter the scope."""
    from af3_jax_opt.inprocess import sampler_bf16 as sb
    from af3_jax_opt import big_levers as bl
    calls = []
    def stock_sample(denoising_step, batch, key, config):
        calls.append(sb._STATE["depth"]); return "stock"
    scoped = sb._scoped_sample(stock_sample)
    assert getattr(scoped, "__wrapped_stock__", None) is None and scoped._sampler_bf16_scope is sb.scope     # not the memory mode's mark
    assert scoped(denoising_step=None, batch=None, key=None, config=None) == "stock" and calls == [1] and sb._STATE["depth"] == 0
    assert bl._scope_of(scoped) is sb.scope                                   # the chunked sampler reads the scope off the function it wraps …
    chunked_like = lambda **k: None; chunked_like.__wrapped_stock__ = scoped
    assert bl._scope_of(chunked_like) is sb.scope                             # … or off any sampler beneath it
    import contextlib
    assert bl._scope_of(stock_sample) is contextlib.nullcontext               # no scope lever: a null context
    with bl._scope_of(scoped)():
        assert sb._STATE["depth"] == 1
    assert sb._STATE["depth"] == 0

# ---- 0.3.24: the fused pair levers KEPT in region reach are judged by their SERVED words (a transcript of `pred --mode big --n_est 4000`
# at 2,048 padded tokens on one H100, kit 0.3.23: FPF_TRIATT served fused=24, TTR traced 38 sites) — the census must word them `on`, not `skipped`
REACH_TRANSCRIPT = ['[af3-jax-opt] ACTIVE mode=big:transition_shard,cond_shard,trimul_chunk base=fast drop=none exact=measured refused=none off=samples_per_pass,logits_shard on=none allocator=none line=big n_gpu=1', '[af3-jax-opt] BIG census units=1 applied=transition_shard,cond_shard,trimul_chunk refused=none traced=transition_shard=2,cond_shard=10,trimul_chunk=24 verdict=ok n_gpu=1', '[af3-jax-opt] SERVED trimul fused=0 fallback=0 triatt fused=24 fallback=0 fallback_shapes=none hoist=off tiles=own:9.0 cc=9.0 dattn=18 dattn_sites=diffusion:10,pairformer_single:8 ttr=38 ttr_routed=8 ttr_fallback=none sbf16=80 sbf16_sites=atom:60,token:20 txla=24 txla_rows=cuda_sm90a:20,k2b_aot:4 txla_aside=none atomattn=22 atomattn_sites=diffusion_decoder:10,diffusion_encoder:10,evoformer_encoder:2 atomattn_fallback=none hlog=off hlog_dtype=none cshare=2 achoist=off achoist_sites=none achoist_aside=none cnoise=off cnoise_rule=none tcd=24 tcd_rows=cd_trimul:24 tcd_aside=none', '[af3-jax-opt] LNP served=150 word=big rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4 routed=adaptive:40,axis:2,channels:12 aside=none uncovered=none', '[af3-jax-opt] KERNELS route=big mode=big flash_impl=engaged:triton@tokamax0.0.12 pair_attn=fpf trimul=glut+w:TriangleMultiplication glu_impl=engaged:triton@tokamax0.0.12 xla_flags=upstream requested=triton:fork_default tokamax=0.0.12 device=NVIDIA_H100_80GB_HBM3 cc=9.0 autotune=ok dpa_calls=triton:18 glu_calls=None:120 probe=ok verdict=ok']
REACH_FUSED = modes.resolve("big", "/c", region="reach", size={"n_padded": 2048, "n_gpu": 1})["levers"]   # the sized reach composition the transcript was taken under (fused pair kernels kept; 0.3.26: COND_SHARE engaged, SAMPLES_PER_PASS / LOGITS_SHARD not composed)


def test_kept_fused_levers_in_reach_are_judged_by_their_served_words():
    _mem()
    glut_line = "[af3-jax-opt] LEVERS active=L-GLUT af3_pallas_levers.py=x script=run_alphafold_fast.py sha256=y"
    ev = modes.lever_evidence("big", REACH_TRANSCRIPT + [glut_line, L1], REACH_FUSED)
    pl = ev["per_lever"]
    assert ev["ok"] and ev["reason"] is None, ev["reason"]
    assert pl["FPF_TRIATT"]["state"] == "on" and pl["FPF_TRIATT"].get("reason") is None, pl["FPF_TRIATT"]     # served fused=24 fallback=0 (the kernel levers' record carries the counts as fields)
    assert pl["TTR"]["state"] == "on" and pl["TTR"]["evidence"] == "ttr=38 routed=8", pl["TTR"]
    assert pl["TRIATT_XLA"]["state"] == "on" and "cuda_sm90a:20,k2b_aot:4" in pl["TRIATT_XLA"]["evidence"] and [d["state"] for d in pl.values()] == ["on"] * 15, {k: d["state"] for k, d in pl.items()}   # 0.3.32: 15 levers (+ LNP); 0.3.26: 14 levers (COND_SHARE engaged in reach on one GPU; SAMPLES_PER_PASS / LOGITS_SHARD not composed)
    lv = {ln.split(" lever=")[1].split()[0]: ln for ln in report.lever_lines("big", pl)}
    assert " state=on " in lv["FPF_TRIATT"] and "served=24" in lv["FPF_TRIATT"] and " state=on " in lv["TTR"] and "ttr=38" in lv["TTR"] and "reason=" not in lv["FPF_TRIATT"] + lv["TTR"], (lv["FPF_TRIATT"], lv["TTR"])
    # a kept fused kernel that served nothing is levers_short by name (never a silent stock path); a DISENGAGED kernel that served is a defect, named
    short = modes.lever_evidence("big", [l.replace("triatt fused=24", "triatt fused=0") for l in REACH_TRANSCRIPT] + [glut_line, L1], REACH_FUSED)
    assert not short["ok"] and "FPF_TRIATT not served: fused=0 fallback=0" in short["reason"] and short["per_lever"]["FPF_TRIATT"]["state"] == "skipped"
    defect = modes.lever_evidence("big", [l.replace("trimul fused=0", "trimul fused=24") for l in REACH_TRANSCRIPT] + [glut_line, L1], REACH_FUSED)
    assert not defect["ok"] and "disengaged fast kernels served: FPF_TRIMUL served fused=24" in defect["reason"]


def test_memory_line_drops_the_fused_pair_kernels_below_the_cc_floor():
    """0.3.36: on the card class below modes.BIG_FUSED_CC_FLOOR (cc 8.0, the A100s) the fused pair kernels FPF_TRIATT + TTR are memory
    holders inside the memory line (+1.83 GiB allocator peak at 2,048 padded tokens, measured) and leave BY NAME; exact's ATTNCFG serves the
    triangle attention site (BIG_FALLBACK); cc >= 9.0 and 'no GPU visible' keep them (0.00 GiB there). Row-sharded runs are unchanged."""
    size = {"n_padded": 2048, "n_gpu": 1}
    keep = modes.resolve("big", "/c", region="reach", size={**size, "cc": 9.0})
    none = modes.resolve("big", "/c", region="reach", size=size)
    drop = modes.resolve("big", "/c", region="reach", size={**size, "cc": 8.0})
    assert keep["levers"] == none["levers"] and "FPF_TRIATT" in keep["levers"] and "TTR" in keep["levers"] and "ATTNCFG" not in keep["levers"]
    assert "FPF_TRIATT" not in drop["levers"] and "TTR" not in drop["levers"] and "ATTNCFG" in drop["levers"] and "COND_SHARE" in drop["levers"] and "LNP" in drop["levers"]
    assert set(drop["disengaged"]) - set(keep["disengaged"]) == {"FPF_TRIATT", "TTR"} and all("cc=8.0 < 9.0" in drop["disengaged"][lv] for lv in ("FPF_TRIATT", "TTR"))
    reg = {**modes.big_region(2000, 1, 81920), "cc": 8.0}
    assert reg["region"] == "reach" and "FPF_TRIATT+TTR off by the card rule (cc=8.0 < 9.0" in modes.region_note(reg)
    assert "kept" not in modes.big_region(2000, 1, 81920, cc=8.0)["rule"] and "off on this card class (cc=8.0 < 9.0" in modes.big_region(2000, 1, 81920, cc=8.0)["rule"]
    assert "kept (0.00 GiB measured, faster)" in modes.region_note({**modes.big_region(2000, 1, 81559), "cc": 9.0})
    x2 = modes.resolve("big", "/c", region="reach", size={"n_padded": 2048, "n_gpu": 2, "cc": 8.0})
    assert "FPF_TRIATT" not in x2["levers"] and "[n_gpu=2 > 1" in x2["disengaged"]["FPF_TRIATT"]

