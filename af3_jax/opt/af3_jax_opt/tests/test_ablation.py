"""The ablation switch MODEL_OPT_LEVERS_OFF (ablation.py, modes.with_levers_off): the request's words, the refusals by name, each lever's exit
through the switch the kit already composes, the class kept, and the printed lines (ACTIVE ablated=…, LEVER state=off reason=ablated) on the stub box."""
import os

import pytest

from af3_jax_opt import ablation, big, cli, modes, registry, stack


def test_requested_words(monkeypatch):
    monkeypatch.delenv(ablation.ENV, raising=False)
    assert ablation.requested() == [] and ablation.requested({}) == [] and ablation.requested({ablation.ENV: " , "}) == []
    assert ablation.requested({ablation.ENV: "fpf_hoist, TTR,,ttr ,Fpf_Hoist"}) == ["FPF_HOIST", "TTR"]          # case-folded to the registry's spelling, blanks ignored, duplicates folded, request order
    assert ablation.requested({ablation.ENV: "nosuch,GLUT"}) == ["nosuch", "GLUT"]                                # an unknown name is kept as given: validate() refuses it by name
    assert ablation.ENV == "MODEL_OPT_LEVERS_OFF" and not ablation.ENV.startswith(stack.ENV_PREFIX)               # a MODEL_OPT_ name: the undeclared-AF3_JAX_ gate never sees it
    assert ablation.TOKEN == "ablated" == ablation.REASON


def test_refusals_by_name():
    fast = modes.resolve("fast")["levers"]
    with pytest.raises(ablation.AblationError, match="mode off applies no lever"):
        ablation.validate("off", ["GLUT"], [])
    with pytest.raises(ablation.AblationError, match=r"NOSUCH: not a lever of this kit \(registry levers: L1, WRITER, FIX1, GLUT, "):
        ablation.validate("fast", ["NOSUCH"], fast)
    with pytest.raises(ablation.AblationError, match=r"GLUT: not in the composition mode fast runs here \(FIX1\+FPF_TRIMUL\+FPF_TRIATT\+FPF_HOIST\+TTR\+DATTN\+TRIATT_XLA\+SAMPLER_BF16\+ATOM_ATTN\+TRIMUL_CD\+HOIST_LOGITS\+COND_SHARE\+ATOM_COND_HOIST\+LNP\+L1\+WRITER\)"):
        ablation.validate("fast", ["GLUT"], fast)
    with pytest.raises(ablation.AblationError, match="FIX1 is the kit script itself"):
        ablation.validate("fast", ["FIX1"], fast)
    x2 = modes.with_n_gpu(modes.resolve("big"), 2)["levers"]
    with pytest.raises(ablation.AblationError, match="ROWPAIR is the --n_gpu axis"):
        ablation.validate("big", ["ROWPAIR"], x2, n_gpu=2)
    with pytest.raises(ablation.AblationError, match=r"COND_SHARD: not in the composition mode big --n_gpu 2 runs here"):   # superseded by ROWPAIR at P=2: not in the composition, said so
        ablation.validate("big", ["COND_SHARD"], x2, n_gpu=2)
    assert ablation.validate("fast", ["TTR", "FPF_HOIST"], fast) == ["TTR", "FPF_HOIST"] and ablation.validate("fast", [], fast) == []
    with pytest.raises(ablation.AblationError, match="NOSUCH: not a lever"):                                       # with_levers_off validates by the same words
        modes.with_levers_off(modes.resolve("fast"), ["NOSUCH"])


def test_no_request_changes_nothing():
    for mode in ("exact", "fast", "big"):
        res = modes.with_n_gpu(modes.resolve(mode, "/c"), 1)
        assert modes.with_levers_off(res, []) is res and "ablated" not in res


def test_each_lever_leaves_through_its_own_switch():
    fast = modes.resolve("fast", "/c")
    r = modes.with_levers_off(fast, ["FPF_HOIST"])
    assert r["levers"] == ["FIX1", "FPF_TRIMUL", "FPF_TRIATT", "TTR", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "TRIMUL_CD", "HOIST_LOGITS", "COND_SHARE", "ATOM_COND_HOIST", "LNP", "L1", "WRITER"] and r["ablated"] == ["FPF_HOIST"]
    assert r["env"] == {**fast["env"], "AF3_DIFFUSION_HOIST": "0"} and r["flags"] == fast["flags"] and r["launcher"] == fast["launcher"]
    assert r["cache_class"] == fast["cache_class"] == "__fast" and r["script"] == fast["script"]                    # the class stays the mode's (the --n_gpu rule: one class per lever line)
    assert modes.with_levers_off(fast, ["FPF_TRIMUL"])["env"]["AF3_FLASHPAIRFORMER"] == "triatt"
    assert modes.with_levers_off(fast, ["FPF_TRIATT"])["env"]["AF3_FLASHPAIRFORMER"] == "trimul"
    both = modes.with_levers_off(fast, ["FPF_TRIATT", "FPF_TRIMUL", "TTR", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "TRIMUL_CD", "HOIST_LOGITS", "COND_SHARE", "ATOM_COND_HOIST", "LNP"])
    assert both["env"] == {"AF3_FLASHPAIRFORMER": "off", "AF3_DIFFUSION_HOIST": "1"} and both["levers"] == ["FIX1", "FPF_HOIST", "L1", "WRITER"]
    assert modes.with_levers_off(fast, ["SAMPLER_BF16"])["env"] == {k: v for k, v in fast["env"].items() if k != "AF3_JAX_SAMPLER_BF16"}   # SAMPLER_BF16 leaves through its own variable alone   # the tree levers' variables gone, the add-on's word off, the hoist's own still on
    assert modes.fpf_kernel_switch(fast["levers"]) == modes.fpf_env()["AF3_FLASHPAIRFORMER"] == "both"               # the package default IS the full row's word
    nol1 = modes.with_levers_off(fast, ["L1"])
    assert nol1["flags"] == ["--output_writer"] + modes.pad_flags("fast") and nol1["row_levers"] == ["WRITER"] and nol1["row"] == fast["row"] and "L1" not in nol1["levers"]   # L1's two flags leave the line; WRITER's flag and the padding policy's --buckets stay
    norow = modes.with_levers_off(fast, ["L1", "WRITER"])
    assert norow["flags"] == modes.pad_flags("fast") and norow["row_levers"] == [] and norow["row"] == ""                                        # every row lever off: no row at all
    exact = modes.resolve("exact", "/c")
    e = modes.with_levers_off(exact, ["GLUT", "ATTNCFG"])
    assert e["env"] == {} and e["levers"] == ["FIX1", "L1", "WRITER"] and e["cache_class"] == "" and e["ablated"] == ["GLUT", "ATTNCFG"]
    assert modes.lever_switch_vars() == {"GLUT": "AF3P_GLU_T", "ATTNCFG": "AF3P_ATTN_CFG", "DATTN": "AF3_JAX_DATTN", "TTR": "AF3_JAX_TTR", "TRIATT_XLA": "AF3_JAX_TRIATT_XLA", "SAMPLER_BF16": "AF3_JAX_SAMPLER_BF16", "ATOM_ATTN": "AF3_JAX_ATOM_ATTN", "TRIMUL_CD": "AF3_JAX_TRIMUL_CD", "LNP": "AF3_JAX_LNP", "HOIST_LOGITS": "AF3_JAX_HOIST_LOGITS", "COND_SHARE": "AF3_JAX_COND_SHARE", "ATOM_COND_HOIST": "AF3_JAX_ATOM_COND_HOIST"}


def test_memory_levers_join_the_launchers_off_word():
    reach = modes.with_n_gpu(modes.resolve("big", "/c", region="reach"), 1)
    r = modes.with_levers_off(reach, ["COND_SHARD", "DATTN"])
    assert r["launcher"] == reach["launcher"][:-2] + [modes.OFF_ARG, "samples_per_pass,logits_shard,cond_shard"] and "AF3_JAX_DATTN" not in r["env"] and "AF3_JAX_DATTN" in reach["env"]
    assert r["cache_class"] == reach["cache_class"] and r["levers"] == [l for l in reach["levers"] if l not in ("COND_SHARD", "DATTN")]
    x2 = modes.with_n_gpu(modes.resolve("big", "/c", region="reach"), 2)
    i = x2["launcher"].index(modes.OFF_ARG)
    r2 = modes.with_levers_off(x2, ["TRANSITION_SHARD"], n_gpu=2)
    assert r2["launcher"][i + 1] == x2["launcher"][i + 1] + ",transition_shard" and r2["launcher"].count(modes.OFF_ARG) == 1   # ONE --off word: the superseded set, extended
    assert "ROWPAIR" in r2["levers"] and "TRANSITION_SHARD" not in r2["levers"]
    regfast = modes.resolve("big", "/c", region="fast")                                                        # region fast IS the fast line: its levers ablate as fast's do, a memory lever is not in that composition
    assert modes.with_levers_off(regfast, ["FPF_HOIST"])["env"]["AF3_DIFFUSION_HOIST"] == "0"
    with pytest.raises(ablation.AblationError, match="COND_SHARD: not in the composition mode big runs here"):
        modes.with_levers_off(regfast, ["COND_SHARD"])


def test_evidence_judges_the_reduced_composition():
    """A kept lever is judged as the mode judges it; an ablated FPF kernel's zero served count is not a shortfall; both kernels ablated: no install line wanted."""
    lines = ["af3_flashpairformer: mode=triatt (TriangleMultiplication=TriangleMultiplication, GridSelfAttention=FlashGridSelfAttention), jax 0.10.2 backend gpu",
             "[af3-jax-opt] SERVED trimul fused=0 fallback=0 triatt fused=2 fallback=0 fallback_shapes=none hoist=5 tiles=own:9.0 cc=9.0 dattn=53 dattn_sites=diffusion:5,pairformer_single:48 ttr=4 ttr_routed=2 ttr_fallback=none sbf16=16 sbf16_sites=atom:12,token:4 txla=24 txla_rows=cuda_sm90a:20,k2b_aot:4 txla_aside=none atomattn=7 atomattn_sites=diffusion_decoder:3,diffusion_encoder:3,evoformer_encoder:1 atomattn_fallback=none hlog=2 hlog_dtype=float32 cshare=2 achoist=2 achoist_sites=enc:2,dec:2,passed:2 achoist_aside=none cnoise=off cnoise_rule=none tcd=24 tcd_rows=cd_trimul:24 tcd_aside=none tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_uncovered=af3_tmpl_c64_ch64_incoming:2", "[af3-jax-opt] LNP served=150 word=fast rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4 routed=adaptive:40,axis:2,channels:12 aside=none uncovered=none",
             "Featurisation prefetch enabled: 3 worker process(es), 4 item(s) ahead.", "Output writer enabled: result extraction and output writing run on one writer thread behind the next fold job."]
    kept = [l for l in modes.resolve("fast")["levers"] if l != "FPF_TRIMUL"]
    ev = modes.lever_evidence("fast", lines, kept)
    assert ev["ok"], ev["reason"]
    full = modes.lever_evidence("fast", lines, modes.resolve("fast")["levers"])                               # the mode proper on the same transcript: mode=triatt is not the row's word
    assert not full["ok"] and "af3_flashpairformer mode=triatt, the tree sets both" in full["reason"]
    none = modes.lever_evidence("fast", lines[1:], ["FIX1", "FPF_HOIST", "TTR", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "TRIMUL_CD", "HOIST_LOGITS", "COND_SHARE", "ATOM_COND_HOIST", "LNP", "L1", "WRITER"])                 # both kernels ablated: AF3_FLASHPAIRFORMER=off prints no install line, none is wanted
    assert none["ok"], none["reason"]
    pl = modes.ablated_per_lever(["FPF_TRIMUL", "L1"])
    assert pl["FPF_TRIMUL"] == {"name": registry.LEVERS["FPF_TRIMUL"].get("strategy") or "FPF_TRIMUL", "state": "off", "reason": "ablated"} and pl["L1"]["state"] == "off"


def test_check_names_the_ablation_on_the_active_line(box, monkeypatch, capsys):
    import json
    monkeypatch.setenv(ablation.ENV, "fpf_hoist,TTR")
    assert cli.main(["check", "--variant", "p2", "--mode", "fast", "--json"]) == 0
    cap = capsys.readouterr()
    active = [ln for ln in cap.err.splitlines() if ln.startswith("[af3-jax-opt] ACTIVE ")][-1]
    assert " levers=FIX1+FPF_TRIMUL+FPF_TRIATT+DATTN+TRIATT_XLA+SAMPLER_BF16+ATOM_ATTN+TRIMUL_CD+HOIST_LOGITS+COND_SHARE+ATOM_COND_HOIST+LNP+L1+WRITER ablated=FPF_HOIST,TTR " in active, active
    rep = json.loads(cap.out)
    assert rep["levers_ablated"] == ["FPF_HOIST", "TTR"] and rep["levers_applied"] == ["FIX1", "FPF_TRIMUL", "FPF_TRIATT", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "TRIMUL_CD", "HOIST_LOGITS", "COND_SHARE", "ATOM_COND_HOIST", "LNP", "L1", "WRITER"]
    assert rep["lever_env"] == {"AF3_FLASHPAIRFORMER": "both", "AF3_DIFFUSION_HOIST": "0", "AF3_JAX_DATTN": "1", "AF3_JAX_TRIATT_XLA": "fast", "AF3_JAX_SAMPLER_BF16": "1", "AF3_JAX_ATOM_ATTN": "1", "AF3_JAX_TRIMUL_CD": "fast", "AF3_JAX_HOIST_LOGITS": "1", "AF3_JAX_COND_SHARE": "1", "AF3_JAX_ATOM_COND_HOIST": "1", "AF3_JAX_LNP": "fast"} and rep["cache_class"] == "__fast"


def test_check_without_the_variable_has_no_token(box, monkeypatch, capsys):
    monkeypatch.delenv(ablation.ENV, raising=False)
    assert cli.main(["check", "--variant", "p2", "--mode", "fast"]) == 0
    assert "ablated=" not in capsys.readouterr().err


@pytest.mark.parametrize("mode,value,words", [
    ("fast", "NOSUCH", "reason=MODEL_OPT_LEVERS_OFF refused — MODEL_OPT_LEVERS_OFF=NOSUCH: NOSUCH: not a lever of this kit"),
    ("exact", "FPF_HOIST", "FPF_HOIST: not in the composition mode exact runs here (FIX1+GLUT+ATTNCFG+L1+WRITER)"),
    ("fast", "FIX1", "FIX1 is the kit script itself"),
    ("off", "L1", "mode off applies no lever, there is nothing to ablate"),
])
def test_refused_requests_exit_3_by_name(box, monkeypatch, capsys, mode, value, words):
    monkeypatch.setenv(ablation.ENV, value)
    assert cli.main(["check", "--variant", "p2", "--mode", mode]) == 3
    err = capsys.readouterr().err
    assert "[af3-jax-opt] NOT ACTIVE mode=" in err and words in err, err


def test_pred_prints_the_ablated_levers_lever_lines(box, monkeypatch, capsys, tmp_path):
    box.warm_cache(mode="fast")
    monkeypatch.setenv(ablation.ENV, "FPF_HOIST")
    out = tmp_path / "out"
    rc = cli.main(["pred", "--variant", "p2", "--mode", "fast", "--output_dir", str(out), "--json_path", box.input_json()])
    err = capsys.readouterr().err
    assert rc == 0, err
    rec = box.stub_record()
    assert rec["env"].get("AF3_DIFFUSION_HOIST") == "0" and rec["env"].get("AF3_FLASHPAIRFORMER") == "both" and ablation.ENV in rec["env"]   # the hoist's word at off in the model process; the variable itself rides along unread
    lever = {ln.split(" lever=")[1].split()[0]: ln for ln in err.splitlines() if ln.startswith("[af3-jax-opt] LEVER ")}
    assert " state=off reason=ablated " in lever["FPF_HOIST"] and " mode=fast " in lever["FPF_HOIST"], lever
    assert all(" state=on " in lever[l] for l in ("FIX1", "FPF_TRIMUL", "FPF_TRIATT", "TTR", "DATTN", "L1")), lever
    active = [ln for ln in err.splitlines() if ln.startswith("[af3-jax-opt] ACTIVE ")][-1]
    assert " ablated=FPF_HOIST " in active and "+FPF_HOIST" not in active
    done = [ln for ln in err.splitlines() if ln.startswith("[af3-jax-opt] DONE ")][-1]
    assert " status=ok " in done and " rc=0 " in done and done.endswith(" ablated=FPF_HOIST"), done          # the DONE line names the ablation too (last field)


def test_pred_big_reach_memory_lever_off(box, monkeypatch, capsys, tmp_path):
    box.warm_cache(mode="fast"); box.warm_cache(mode="big")
    import json
    monkeypatch.setenv(ablation.ENV, "trimul_chunk,GLUT")
    rc = cli.main(["check", "--variant", "p2", "--mode", "big", "--n_est", "4000", "--json"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    active = [ln for ln in cap.err.splitlines() if ln.startswith("[af3-jax-opt] ACTIVE ")][-1]
    assert " ablated=TRIMUL_CHUNK,GLUT " in active and "TRIMUL_CHUNK" not in active.split(" levers=")[1].split()[0] and "+GLUT" not in active, active
    rep = json.loads(cap.out)
    assert f" {modes.OFF_ARG} samples_per_pass,logits_shard,trimul_chunk " in rep["command"] and "AF3P_GLU_T" not in rep["lever_env"] and rep["lever_env"].get("AF3_FLASHPAIRFORMER") == "triatt" and "AF3P_ATTN_CFG" not in rep["lever_env"], rep["command"]   # 4,032 padded on one GPU: the fused triangle attention kept (ATTNCFG's site not vacated), GLUT ablated by request
    assert rep["cache_class"] == modes.cache_class_of("big", "reach")                                      # the memory line's class, unchanged by the ablation


@pytest.mark.parametrize("mode,value,levers,fpf_word", [
    ("fast", "FPF_TRIATT", "FIX1+FPF_TRIMUL+FPF_HOIST+TTR+DATTN+TRIATT_XLA+SAMPLER_BF16+ATOM_ATTN+TRIMUL_CD+HOIST_LOGITS+COND_SHARE+ATOM_COND_HOIST+LNP+L1+WRITER", "trimul"),
    ("fast", "FPF_TRIMUL,FPF_TRIATT", "FIX1+FPF_HOIST+TTR+DATTN+TRIATT_XLA+SAMPLER_BF16+ATOM_ATTN+TRIMUL_CD+HOIST_LOGITS+COND_SHARE+ATOM_COND_HOIST+LNP+L1+WRITER", "off"),
    ("exact", "GLUT", "FIX1+ATTNCFG+L1+WRITER", None),
])
def test_pred_with_kernel_levers_ablated_passes_its_gates(box, monkeypatch, capsys, tmp_path, mode, value, levers, fpf_word):
    """A kernel lever ablated: the KERNELS line's REQUIRE guard expects what the REDUCED composition implies (verdict=ok), the kept levers pass
    all-or-refuse, DONE rc=0 — on the stub box, for one FlashPairformer kernel off, both off, and exact's GLUT off."""
    box.warm_cache(mode=mode)
    monkeypatch.setenv(ablation.ENV, value)
    rc = cli.main(["pred", "--variant", "p2", "--mode", mode, "--output_dir", str(tmp_path / "out"), "--json_path", box.input_json()])
    err = capsys.readouterr().err
    assert rc == 0, err
    active = [ln for ln in err.splitlines() if ln.startswith("[af3-jax-opt] ACTIVE ")][-1]
    assert f" levers={levers} ablated={value} " in active, active
    kern = [ln for ln in err.splitlines() if ln.startswith("[af3-jax-opt] KERNELS ")]
    assert kern and " verdict=ok" in kern[-1], kern
    done = [ln for ln in err.splitlines() if ln.startswith("[af3-jax-opt] DONE ")][-1]
    assert " status=ok " in done and " rc=0 " in done and done.endswith(f" ablated={value}"), done
    rec = box.stub_record()
    if fpf_word is not None:
        assert rec["env"].get("AF3_FLASHPAIRFORMER") == fpf_word, rec["env"]                                  # the add-on's word rewritten to what remains
    else:
        assert "AF3P_GLU_T" not in rec["env"] and rec["env"].get("AF3P_ATTN_CFG"), rec["env"]                  # GLUT's variable gone, ATTNCFG's kept


def test_row_words_pass_by_name(monkeypatch):
    """The support library's row words (triattn_xla:<row>) are not kit levers: requested() skips them, validate() passes them by name against the
    composition (TRIATT_XLA must run, the row must be one of the pinned core's), token() words them ``rows_off=`` after ``ablated=``."""
    fast = modes.resolve("fast")["levers"]
    assert "TRIATT_XLA" in fast and "triattn_native" in ablation.core_rows()                     # the pinned core carries the row
    env = {ablation.ENV: "sampler_bf16, triattn_xla:triattn_native,,triattn_xla:triattn_native"}
    assert ablation.requested(env) == ["SAMPLER_BF16"]                                        # the row word is not a kit lever
    assert ablation.row_words(env) == ["triattn_xla:triattn_native"]                              # duplicates folded, as spelled
    assert ablation.validate("fast", ablation.requested(env), fast, environ=env) == ["SAMPLER_BF16"]
    assert ablation.token(["SAMPLER_BF16"], env) == " ablated=SAMPLER_BF16 rows_off=triattn_xla:triattn_native"
    assert ablation.token([], {ablation.ENV: "triattn_xla:cuda_sm90a"}) == " rows_off=triattn_xla:cuda_sm90a" and ablation.token([], {}) == ""
    monkeypatch.setenv(ablation.ENV, "triattn_xla:triattn_native")
    assert ablation.requested() == [] and ablation.row_words() == ["triattn_xla:triattn_native"] and ablation.validate("fast", [], fast) == []


@pytest.mark.parametrize("value,mode,words", [
    ("triattn_xla:triattn_native", "off", "mode off applies no lever"),
    ("triattn_xla", "fast", "names every row"),
    ("triattn_xla:nosuch", "fast", "not a row of opt_core.kernels.triattn_xla"),
    ("TRIATTN_XLA:TRIATTN_NATIVE", "fast", "case-sensitively"),
    ("TRIATT_XLA,triattn_xla:triattn_native", "fast", "TRIATT_XLA is ablated in this run"),
    ("triattn_xla:triattn_native", "exact", "TRIATT_XLA is not in the composition"),
])
def test_row_words_refused_by_name(value, mode, words):
    env = {ablation.ENV: value}
    comp = [] if mode == "off" else modes.resolve(mode)["levers"]
    with pytest.raises(ablation.AblationError, match=words):
        ablation.validate(mode, ablation.requested(env), comp, environ=env)
