"""The mode table and its sources: the rows read out of the add-ons' own files, the launchers, the lever environments, the registry."""
import os
import re

import pytest

from af3_jax_opt import modes, registry, stack
from .conftest import FPF, KIT, PALLAS, PKG, TREE


def _row_of(path, name):
    rx = re.compile(r'^' + name + r'=(?:\$\{\w+:-)?"(.*)"\}?\s*$')
    for ln in open(path, encoding="utf-8"):
        m = rx.match(ln)
        if m:
            return m.group(1)
    raise AssertionError(name)


def test_mode_table():
    assert modes.MODES == ("off", "exact", "fast", "big") and set(modes.KIT_MODES) == set(modes.MODES)
    assert modes.DEFAULT_MODE == "fast"
    assert modes.KIT_MODES["off"] == {"script": "run_alphafold.py", "levers": [], "launcher": None, "cache_class": "", "kits": ()}
    assert modes.KIT_ROW == "FAST" and set(modes.ROW_LEVER_FLAGS) == {"L1", "WRITER"}
    assert modes.KIT_MODES["exact"]["cache_class"] == "" and modes.KIT_MODES["fast"]["cache_class"] == "__fast"
    assert modes.KIT_MODES["exact"]["kits"] == ("fast_inference", "pallas") and modes.KIT_MODES["fast"]["kits"] == ("fast_inference", "fpf")


def test_rows_are_the_kits_rows_file():
    """An independent read of the base kit's rows.sh: the COMMON and FAST rows the package resolves are those lines."""
    st = os.path.join(KIT, "rows.sh")
    rows = modes.kit_rows()
    for name in modes.ROW_NAMES:
        assert rows[name] == _row_of(st, name), name
    assert modes.resolve("exact")["row"] == rows[modes.KIT_ROW] == modes.resolve("fast")["row"]                       # the row the row levers select from
    assert modes.resolve("exact")["row_levers"] == ["L1", "WRITER"] == modes.resolve("fast")["row_levers"]
    assert modes.resolve("exact")["flags"][:2] == ["--featurisation_workers=3", "--featurisation_prefetch=4"] == modes.resolve("fast")["flags"][:2]      # L1's flags as written in the row
    for md in ("exact", "fast"):
        assert modes.resolve(md)["flags"][2] == "--output_writer" and modes.resolve(md)["flags"][3:] == modes.pad_flags(md) and len(modes.resolve(md)["flags"]) == 4   # WRITER's flag + the padding policy's --buckets (test_padding_policy_by_mode)
    assert modes.resolve("off")["flags"] == modes.pad_flags("off") == modes.pad_flags("fast")                                                        # the stock route pads as every mode does: ONE --buckets list
    assert modes.resolve("exact", "/c")["flags"] == modes.split_row(rows["FAST"], "/c") + modes.pad_flags("exact")                                    # the row lever = the whole row, in row order, then the padding
    assert modes.resolve("exact")["levers"] == ["FIX1", "GLUT", "ATTNCFG", "L1", "WRITER"]
    assert modes.resolve("fast")["levers"] == ["FIX1", "FPF_TRIMUL", "FPF_TRIATT", "FPF_HOIST", "TTR", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "TRIMUL_CD", "HOIST_LOGITS", "COND_SHARE", "ATOM_COND_HOIST", "LNP", "L1", "WRITER"]
    assert modes.resolve("off")["row"] == "" and modes.resolve("off")["flags"] == modes.pad_flags("off")            # the stock route: no kit row, the ONE padding list (upstream's own --buckets)
    assert "$CACHE" not in rows["FAST"] and "$CACHE" in rows["COMMON"]


def test_common_flags_are_the_common_row_minus_the_composed_ones():
    common = modes.kit_rows()["COMMON"].split()
    assert modes.common_flags() == [t for t in common if not t.startswith("$") and t.split("=", 1)[0] not in modes.COMPOSED_KEYS]
    assert modes.common_flags() == ["--norun_data_pipeline"]


def test_pallas_levers_are_the_addons_tier1_set():
    assert modes.pallas_levers() == modes.PALLAS_LEVERS == {"AF3P_GLU_T": "1", "AF3P_ATTN_CFG": "64,64,4,3"}   # the Tier-1 set; no AF3P_PAIR_CHUNK
    assert modes.pallas_levers() is not modes.PALLAS_LEVERS                                             # a fresh copy each call (callers compose on it)
    src = open(os.path.join(PALLAS, "patches", "af3_pallas_levers.py"), encoding="utf-8").read()        # the add-on's apply_levers() reads each variable the table sets
    assert all(f'os.environ.get("{k}"' in src for k in modes.PALLAS_LEVERS), [k for k in modes.PALLAS_LEVERS if f'os.environ.get("{k}"' not in src]
    assert modes.resolve("exact")["env"] == modes.pallas_levers() and modes.resolve("fast")["env"] == {**modes.fpf_env(), **modes.TREE_LEVER_ENV["TTR"], **modes.TREE_LEVER_ENV["DATTN"], **modes.TREE_LEVER_ENV["TRIATT_XLA"], **modes.TREE_LEVER_ENV["SAMPLER_BF16"], **modes.TREE_LEVER_ENV["ATOM_ATTN"], **modes.TREE_LEVER_ENV["TRIMUL_CD"], **modes.TREE_LEVER_ENV["HOIST_LOGITS"], **modes.TREE_LEVER_ENV["COND_SHARE"], **modes.TREE_LEVER_ENV["ATOM_COND_HOIST"], **modes.TREE_LEVER_ENV["LNP"]}   # the add-on's row + the tree's in-process lever switches


def test_fpf_env_is_the_packages_switches():
    src = open(os.path.join(FPF, "af3_flashpairformer", "__init__.py"), encoding="utf-8").read()
    env = modes.fpf_env()
    assert set(env) == {"AF3_FLASHPAIRFORMER", "AF3_DIFFUSION_HOIST"}
    assert f'"AF3_FLASHPAIRFORMER", "{env["AF3_FLASHPAIRFORMER"]}"' in src                # the package default of the kernel switch
    assert env["AF3_DIFFUSION_HOIST"] in ("1", "true", "on", "yes") and f'"{env["AF3_DIFFUSION_HOIST"]}"' in src


def test_launchers():
    assert modes.launcher(None) == []
    lev = modes.launcher("levers")
    assert len(lev) == 2 and lev[0] == os.path.join(PKG, "levers_launch.py") and lev[1] == os.path.join(PALLAS, "patches")
    assert os.path.isfile(lev[0]) and os.path.isfile(os.path.join(lev[1], "af3_pallas_levers.py"))
    fpf = modes.launcher("fpf")
    assert fpf == [os.path.join(PKG, "fpf_launch.py"), os.path.join(FPF, "run_alphafold_flashpairformer.py")] and all(os.path.isfile(p) for p in fpf)
    with pytest.raises(modes.UnsupportedMode):
        modes.launcher("other")


def test_unknown_mode():
    with pytest.raises(modes.UnsupportedMode):
        modes.resolve("turbo")


def test_exact_only_flags_are_the_fast_rows_names():
    fast = modes.kit_rows()["FAST"]
    assert modes.exact_only_flags() == sorted({t.split("=", 1)[0] for t in modes.split_row(fast) if t.startswith("--")})
    assert modes.exact_only_flags() == ["--featurisation_prefetch", "--featurisation_workers", "--output_writer"]
    assert modes.exact_only(["--featurisation_workers=3", "--num_recycles=10", "--featurisation_prefetch=2"]) == ["--featurisation_workers=3", "--featurisation_prefetch=2"]
    assert modes.exact_only(["--num_recycles=10"]) == []


def test_every_mode_lever_is_registered():
    for mode, spec in modes.KIT_MODES.items():
        for lever in spec["levers"]:
            assert lever in registry.LEVERS, (mode, lever)
            for rel in registry.LEVERS[lever]["touches"]:
                assert os.path.isfile(os.path.join(stack.opt_home(), "forward", rel)), rel
    assert {l for l in registry.LEVERS if l.startswith("FPF")} == set(modes.KIT_MODES["fast"]["levers"]) - {"FIX1", "L1", "WRITER", "TTR", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "TRIMUL_CD", "HOIST_LOGITS", "COND_SHARE", "ATOM_COND_HOIST", "LNP"}
    assert set(modes.TREE_LEVER_ENV) == set(modes.INPROCESS_MODULES) - set(modes.KIT_MODES["big"]["levers"]) - set(modes.BIG_NOT_COMPOSED) - {"ROWPAIR"} == {"TTR", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "TRIMUL_CD", "HOIST_LOGITS", "COND_SHARE", "ATOM_COND_HOIST", "LNP"} <= set(modes.KIT_MODES["fast"]["levers"])   # the tree's own in-process levers: fast's row
    assert set(registry.LEVERS) == {l for spec in modes.KIT_MODES.values() for l in spec["levers"]} | {"ROWPAIR"} | set(modes.BIG_NOT_COMPOSED)   # + the registered memory levers the one-GPU line does not compose (SAMPLES_PER_PASS rides the row-sharded line; LOGITS_SHARD kept registered with its number)      # every registered lever is some mode's row (ROWPAIR joins big at --n_gpu > 1): no available-not-default lever
    assert {"GLUT", "ATTNCFG"} == set(modes.KIT_MODES["exact"]["levers"]) - {"FIX1", "L1", "WRITER"}
    assert "L1" in modes.KIT_MODES["exact"]["levers"] and "L1" in modes.KIT_MODES["fast"]["levers"]     # featurisation prefetch: a row lever in both kit rows
    assert "FIX1" in modes.KIT_MODES["exact"]["levers"] and "FIX1" in modes.KIT_MODES["fast"]["levers"]


def test_row_lever_flags_partition_the_fast_row():
    """Every FAST-row flag belongs to exactly one row lever and every row lever finds its flags in the row (the map and the row agree)."""
    row = modes.split_row(modes.kit_rows()["FAST"])
    names = [t.split("=", 1)[0] for t in row if t.startswith("--")]
    owned = [f for flags in modes.ROW_LEVER_FLAGS.values() for f in flags]
    assert sorted(names) == sorted(owned) and len(owned) == len(set(owned))
    assert modes.row_lever_flags(["FIX1", "GLUT"]) == [] and modes.row_lever_flags(["L1"], "/c") == ["--featurisation_workers=3", "--featurisation_prefetch=4"]   # row order


def test_lever_evidence_rules():
    """One line per lever per process: the LEVERS line (exact), the install + SERVED lines with the hoist token (fast), the kit
    script's own line per row lever."""
    L1 = "Featurisation prefetch enabled: 3 worker process(es), 4 item(s) ahead.\nOutput writer enabled: result extraction and output writing run on one writer thread behind the next fold job."   # the two row levers' lines (L1, WRITER) as one transcript element
    lev = "[af3-jax-opt] LEVERS active=L-ATTNCFG+L-GLUT af3_pallas_levers.py=x script=run_alphafold_fast.py sha256=y"
    assert modes.lever_evidence("exact", [lev, L1])["ok"]
    ev = modes.lever_evidence("exact", [lev])
    assert not ev["ok"] and ev["reason"].startswith("L1 printed no")
    assert not modes.lever_evidence("exact", [L1])["ok"]
    inst = "af3_flashpairformer: mode=both (TriangleMultiplication=FlashTriangleMultiplication, GridSelfAttention=FlashGridSelfAttention), jax 0.10.2 backend gpu"
    served = "[af3-jax-opt] SERVED trimul fused=3 fallback=0 triatt fused=2 fallback=0 fallback_shapes=none hoist=1 tiles=own:9.0 cc=9.0 dattn=53 dattn_sites=diffusion:5,pairformer_single:48 ttr=3 ttr_routed=1 ttr_fallback=none sbf16=16 sbf16_sites=atom:12,token:4 txla=24 txla_rows=cuda_sm90a:20,k2b_aot:4 txla_aside=none atomattn=7 atomattn_sites=diffusion_decoder:3,diffusion_encoder:3,evoformer_encoder:1 atomattn_fallback=none hlog=2 hlog_dtype=float32 cshare=2 achoist=2 achoist_sites=enc:2,dec:2,passed:2 achoist_aside=none cnoise=off cnoise_rule=none tcd=24 tcd_rows=cd_trimul:24 tcd_aside=none tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_uncovered=af3_tmpl_c64_ch64_incoming:2" + "\n[af3-jax-opt] LNP served=150 word=fast rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4 routed=adaptive:40,axis:2,channels:12 aside=none uncovered=none"
    FASTD = modes.resolve("fast")["levers"]                                                                # the fast row: every tree lever's evidence rule exercised
    ev = modes.lever_evidence("fast", [inst, L1, served], FASTD)
    assert ev["ok"] and ev["hoist"] == "1" and ev["served_via"] == "traced" and ev["served"]["FPF_TRIMUL"] == {"fused": 3, "fallback": 0} and ev["tiles"] == "own:9.0" and ev["dattn"] == "53" and ev["ttr"] == "3" and ev["cnoise"] == "off"
    assert modes.lever_evidence("fast", [inst, L1, served.replace("tiles=own:9.0", "tiles=own")])["ok"]      # the tiles word: own | own:<cc>
    sf = modes.lever_evidence("fast", [inst, L1, served.replace("trimul fused=3 fallback=0", "trimul fused=3 fallback=2").replace("fallback_shapes=none", "fallback_shapes=trimul:600x600x128:bfloat16")])
    assert not sf["ok"] and sf["softened"] and sf["fell_back"] == ["FPF_TRIMUL not served: fused=3 fallback=2 at trimul:600x600x128:bfloat16"] and sf["reason"] == sf["fell_back"][0]   # engaged + some fallback: softened
    nf = modes.lever_evidence("fast", [inst, L1, served.replace("trimul fused=3 fallback=0", "trimul fused=0 fallback=2").replace("fallback_shapes=none", "fallback_shapes=trimul:600x600x128:bfloat16")])
    assert not nf["ok"] and nf["softened"] and nf["fell_back"] == ["FPF_TRIMUL aside:pad64:2 at trimul:600x600x128:bfloat16"]          # every call fell back at a DECLARED shape (a padded size not a multiple of the tile): a step-aside by name, softened (exit 0)
    uf = modes.lever_evidence("fast", [inst, L1, served.replace("trimul fused=3 fallback=0", "trimul fused=0 fallback=0")])
    assert not uf["ok"] and not uf["softened"] and "FPF_TRIMUL not served: fused=0 fallback=0" in uf["reason"]                             # never engaged and named nothing: levers_short proper (exit 3)
    hf = modes.lever_evidence("fast", [inst, L1, served.replace("trimul fused=3 fallback=0", "trimul fused=3 fallback=2").replace("fallback_shapes=none", "fallback_shapes=trimul:600x600x128:bfloat16").replace("hoist=1", "hoist=off")])
    assert not hf["ok"] and not hf["softened"] and "FPF_HOIST not traced" in hf["reason"] and "FPF_TRIMUL not served: fused=3 fallback=2" in hf["reason"]   # a hard shortfall beside it: not softened
    for bad, why, softened in ((served.replace("ttr=3", "ttr=off"), "TTR not traced: ttr=off", False), (served.replace("ttr_fallback=none", "ttr_fallback=rows:2"), "TTR aside at rows:2", True),
                               (served.split(" ttr=")[0], "TTR not traced: ttr=absent", False)):
        ev = modes.lever_evidence("fast", [inst, L1, bad], FASTD)
        assert not ev["ok"] and why in ev["reason"] and bool(ev.get("softened")) is softened, (why, ev["reason"])   # the tree's levers: served or named, never silent; a DECLARED fallback is a named step-aside (softened, exit 0)
    assert modes.lever_evidence("fast", [inst, L1, served.replace(" cnoise=off cnoise_rule=none", "")])["ok"]   # the cnoise words are constants: no lever reads them
    ev = modes.lever_evidence("fast", [inst, L1, served.replace("dattn=53", "dattn=off")], FASTD)
    assert not ev["ok"] and "DATTN not traced: dattn=off" in ev["reason"] and ev["per_lever"]["DATTN"] == {"name": "F5.flash_attn_dense", "state": "skipped", "reason": "dattn=off", "evidence": None}
    assert "DATTN not traced: dattn=absent" in modes.lever_evidence("fast", [inst, L1, served.split(" dattn=")[0]], FASTD)["reason"]
    ev = modes.lever_evidence("fast", [inst, L1, served.replace("tiles=own:9.0 cc=9.0", "tiles=fallback:8.0->9.0 cc=8.0")])
    assert not ev["ok"] and "FPF tile table not this GPU's own: tiles=fallback:8.0->9.0 cc=8.0" in ev["reason"]           # the add-on's degraded configuration, named
    assert "FPF tile table not this GPU's own: tiles=absent cc=absent" in modes.lever_evidence("fast", [inst, L1, served.replace(" tiles=own:9.0 cc=9.0", "")])["reason"]
    for bad, why in ((served.replace("hoist=1", "hoist=off"), "FPF_HOIST not traced: hoist=off"), (served.replace(" hoist=1", ""), "FPF_HOIST not traced: hoist=absent"),
                     (served.replace("hoist=1", "hoist=uncounted"), "FPF_HOIST not traced: hoist=uncounted"),
                     (served.replace("hoist=1", "hoist=probe_error:ImportError"), "FPF_HOIST not traced: hoist=probe_error:ImportError")):
        ev = modes.lever_evidence("fast", [inst, L1, bad])
        assert not ev["ok"] and why in ev["reason"], why
    untraced = "[af3-jax-opt] SERVED trimul fused=0 fallback=0 triatt fused=0 fallback=0 fallback_shapes=none hoist=0 tiles=own:9.0 cc=9.0 dattn=0 dattn_sites=none ttr=0 ttr_routed=0 ttr_fallback=none sbf16=16 sbf16_sites=atom:12,token:4 txla=24 txla_rows=cuda_sm90a:20,k2b_aot:4 txla_aside=none atomattn=7 atomattn_sites=diffusion_decoder:3,diffusion_encoder:3,evoformer_encoder:1 atomattn_fallback=none hlog=2 hlog_dtype=float32 cshare=2 achoist=2 achoist_sites=enc:2,dec:2,passed:2 achoist_aside=none cnoise=off cnoise_rule=none tcd=24 tcd_rows=cd_trimul:24 tcd_aside=none tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_uncovered=af3_tmpl_c64_ch64_incoming:2" + "\n[af3-jax-opt] LNP served=150 word=fast rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4 routed=adaptive:40,axis:2,channels:12 aside=none uncovered=none"
    ev = modes.lever_evidence("fast", [inst, L1, untraced])
    assert not ev["ok"] and "FPF_TRIMUL not served" in ev["reason"] and "FPF_HOIST not traced" in ev["reason"] and "DATTN not traced: dattn=0" in ev["reason"] and "TTR not traced: ttr=0" in ev["reason"]
    assert modes.lever_evidence("off", [])["ok"] and modes.is_evidence_line(L1) and modes.is_evidence_line(served) and not modes.is_evidence_line("Running model inference with seed 1 took 1.23 seconds.")


def test_per_lever_census_and_lever_lines():
    """Every lever of the composition is on (its evidence in the transcript) or skipped with the reason; one LEVER line each, in the core's grammar."""
    from af3_jax_opt import report
    L1 = "Featurisation prefetch enabled: 3 worker process(es), 4 item(s) ahead.\nOutput writer enabled: result extraction and output writing run on one writer thread behind the next fold job."   # the two row levers' lines (L1, WRITER) as one transcript element
    lev = "[af3-jax-opt] LEVERS active=L-ATTNCFG+L-GLUT af3_pallas_levers.py=x script=run_alphafold_fast.py sha256=y"
    ev = modes.lever_evidence("exact", [lev, L1])
    pl = ev["per_lever"]
    assert list(pl) == ["FIX1", "GLUT", "ATTNCFG", "L1", "WRITER"] and all(d["state"] == "on" for d in pl.values())
    assert pl["L1"]["name"] == "F6.item_ordering_prefetch" and pl["GLUT"]["name"] == "GLUT" and pl["FIX1"]["evidence"] == "in-script"
    lines = report.lever_lines("exact", pl)
    assert lines[0] == "[af3-jax-opt] LEVER name=F3.xla_flags_compile state=on impl=host origin=kit lever=FIX1 mode=exact site=fast_inference/patches/patched_files/run_alphafold.py evidence=in-script"
    assert lines[1].startswith("[af3-jax-opt] LEVER name=GLUT state=on impl=pallas_triton origin=kit lever=GLUT mode=exact site=pallas_addon/patches/af3_pallas_levers.py evidence=")
    assert lines[3] == "[af3-jax-opt] LEVER name=F6.item_ordering_prefetch state=on impl=host origin=kit lever=L1 mode=exact site=fast_inference/patches/patched_files/run_alphafold.py evidence=line"
    assert all("impl" in registry.LEVERS[l] and "origin" not in registry.LEVERS[l] for l in registry.LEVERS)   # origin is kit for every lever of this tree (report.lever_lines)
    ev = modes.lever_evidence("exact", [lev.replace("L-ATTNCFG+L-GLUT", "L-GLUT")], modes.KIT_MODES["exact"]["levers"])
    pl = ev["per_lever"]
    assert pl["ATTNCFG"] == {"name": "ATTNCFG", "state": "skipped", "reason": "not on the LEVERS line (active=L-GLUT)", "evidence": None}
    assert pl["L1"]["state"] == "skipped" and pl["L1"]["reason"] == "no line"
    assert report.lever_lines("exact", pl)[2] == "[af3-jax-opt] LEVER name=ATTNCFG state=skipped reason=not_on_the_LEVERS_line_(active=L-GLUT) impl=tokamax_config origin=kit lever=ATTNCFG mode=exact site=pallas_addon/patches/af3_pallas_levers.py"
    inst = "af3_flashpairformer: mode=both (TriangleMultiplication=FlashTriangleMultiplication, GridSelfAttention=FlashGridSelfAttention), jax 0.10.2 backend gpu"
    served = "[af3-jax-opt] SERVED trimul fused=3 fallback=0 triatt fused=2 fallback=0 fallback_shapes=none hoist=1 tiles=own:9.0 cc=9.0 dattn=53 dattn_sites=diffusion:5,pairformer_single:48 ttr=3 ttr_routed=1 ttr_fallback=none sbf16=16 sbf16_sites=atom:12,token:4 txla=24 txla_rows=cuda_sm90a:20,k2b_aot:4 txla_aside=none atomattn=7 atomattn_sites=diffusion_decoder:3,diffusion_encoder:3,evoformer_encoder:1 atomattn_fallback=none hlog=2 hlog_dtype=float32 cshare=2 achoist=2 achoist_sites=enc:2,dec:2,passed:2 achoist_aside=none cnoise=off cnoise_rule=none tcd=24 tcd_rows=cd_trimul:24 tcd_aside=none tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_uncovered=af3_tmpl_c64_ch64_incoming:2" + "\n[af3-jax-opt] LNP served=150 word=fast rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4 routed=adaptive:40,axis:2,channels:12 aside=none uncovered=none"
    pl = modes.lever_evidence("fast", [inst, L1, served])["per_lever"]
    assert list(pl) == ["FIX1", "FPF_TRIMUL", "FPF_TRIATT", "FPF_HOIST", "TTR", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "TRIMUL_CD", "HOIST_LOGITS", "COND_SHARE", "ATOM_COND_HOIST", "LNP", "L1", "WRITER"] and [d["state"] for d in pl.values()] == ["on"] * 16 and pl["DATTN"]["evidence"] == "dattn=53 sites=diffusion:5,pairformer_single:48"   # the fast row
    assert pl["TTR"] == {"name": "LOCAL.fused_transition", "state": "on", "reason": None, "evidence": "ttr=3 routed=1"}
    assert pl["FPF_TRIMUL"]["name"] == "F2.fpf_trimul_fast" and pl["FPF_TRIATT"]["fields"] == {"served": 2, "fallback": 0} and pl["FPF_TRIATT"]["evidence"] is None and pl["FPF_HOIST"]["evidence"] == "hoist=1"
    pl = modes.lever_evidence("fast", [inst, L1, served.replace("hoist=1", "hoist=off").replace("triatt fused=2 fallback=0", "triatt fused=0 fallback=2")])["per_lever"]
    assert pl["FPF_HOIST"] == {"name": "LOCAL.step_invariant_hoist", "state": "skipped", "reason": "hoist=off", "evidence": None} and pl["FPF_TRIATT"]["reason"] == "fused=0 fallback=2"
    assert modes.lever_evidence("off", [])["per_lever"] == {}


def test_mode_env():
    """The model-process variables per mode: the add-on rows plus the tree's own in-process lever switches."""
    assert modes.mode_env("off") == {} and modes.mode_env("exact") == modes.pallas_levers() and not any(k.startswith("AF3_JAX_") for k in modes.mode_env("exact"))
    assert modes.mode_env("fast") == modes.fpf_env() and modes.lever_env("fast", modes.KIT_MODES["fast"]["levers"]) == modes.resolve("fast")["env"] == {**modes.fpf_env(), **modes.TREE_LEVER_ENV["TTR"], **modes.TREE_LEVER_ENV["DATTN"], **modes.TREE_LEVER_ENV["TRIATT_XLA"], **modes.TREE_LEVER_ENV["SAMPLER_BF16"], **modes.TREE_LEVER_ENV["ATOM_ATTN"], **modes.TREE_LEVER_ENV["TRIMUL_CD"], **modes.TREE_LEVER_ENV["HOIST_LOGITS"], **modes.TREE_LEVER_ENV["COND_SHARE"], **modes.TREE_LEVER_ENV["ATOM_COND_HOIST"], **modes.TREE_LEVER_ENV["LNP"]}   # mode_env: the add-on rows; lever_env: + the tree's switch
    assert modes.lever_env("fast", ["FIX1", "FPF_TRIMUL", "FPF_TRIATT", "FPF_HOIST", "L1"]) == modes.fpf_env()   # a composition without the tree's levers carries none of their switches
    assert set(modes.INPROCESS_MODULES) <= set(registry.LEVERS) and set(modes.TREE_LEVER_ENV) <= set(registry.LEVERS)
    for rel in modes.INPROCESS_MODULES.values():
        assert os.path.isfile(os.path.join(stack.opt_home(), "af3_jax_opt", rel)), rel
    for env in modes.TREE_LEVER_ENV.values():
        assert all(k.startswith(stack.ENV_PREFIX) and k in stack.LEVER_SWITCH_ENV and k in stack.DECLARED_ENV for k in env)   # the package's prefix (stripped from callers) and declared: a model process carrying the hook accepts them
def test_inprocess_levers_are_stdlib_at_import():
    """inprocess/dattn.py and inprocess/ttr.py import nothing of jax/alphafold3 at import (the wrapper never loads them); their switches are the table's."""
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location("af3_jax_opt.inprocess.dattn", os.path.join(stack.opt_home(), "af3_jax_opt", "inprocess", "dattn.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert mod.ENV_SWITCH == "AF3_JAX_DATTN" and modes.TREE_LEVER_ENV["DATTN"] == {mod.ENV_SWITCH: "1"} and mod.wanted({"AF3_JAX_DATTN": "1"}) and not mod.wanted({})
    assert mod.report() == {"installed": False, "word": "fast", "traced": 0, "sites": {}, "rows": {}, "aside": {}, "uncovered": {}, "implementation": "none"} and "jax" not in _sys.modules and "tokamax" not in _sys.modules   # nothing served yet: no implementation named (the provider's rows say which at run time)
    for name in ("ttr",):                                                           # the tree's other in-process lever: stdlib at import too, switch name as the table says
        sp = importlib.util.spec_from_file_location(f"af3_jax_opt.inprocess.{name}", os.path.join(stack.opt_home(), "af3_jax_opt", "inprocess", f"{name}.py"))
        m2 = importlib.util.module_from_spec(sp); sp.loader.exec_module(m2)
        assert modes.TREE_LEVER_ENV[name.upper()] == {m2.ENV_SWITCH: ("fast" if name.upper() in modes.TIER_WORD_SWITCHES else "1")} and m2.wanted({m2.ENV_SWITCH: "1"}) and not m2.wanted({}) and m2.report()["installed"] is False   # a TIER_WORD_SWITCHES lever's switch carries a provider tier word (its DEFAULT_WORD in the table; the composition's at launch)
    assert "jax" not in _sys.modules and "tokamax" not in _sys.modules and "opt_core.kernels.fpf_pallas.transition_pallas" not in _sys.modules



def test_padding_policy_by_mode(monkeypatch):
    """PAD_POLICY: off / exact pad by the fork's own --buckets default, verbatim (nothing on the line: like-for-like with stock, bitwise where
    the mode is); fast pads to the next multiple of the shared core's KERNEL_TILE — `--buckets=<every multiple of the tile up to the fork's
    largest bucket>` on the line, before the caller's flags (a caller's own --buckets wins by absl order) — the multiple the fused pair kernels
    serve; big (at any n_gpu) pads by the fork's list."""
    from af3_jax_opt import settings, stock_pred
    stock = settings.upstream_buckets()
    assert stock[:3] == [32, 64, 128] and stock[-1] == 5120 and stock == sorted(stock)
    tile = modes.kernel_tile()
    assert tile in (64, 128) and all(b % tile == 0 for b in stock if b >= 128)              # every default bucket from 128 up is a tile multiple: af3_buckets(kernel_tile(n)) == af3_buckets(n)
    assert modes.PAD_POLICY == {"off": "kernel_tile", "exact": "kernel_tile", "fast": "kernel_tile", "big": "kernel_tile"}   # ONE padding in every mode
    assert modes.pad_buckets("af3_buckets") == [] and modes.pad_buckets("af3_buckets_served") == [b for b in stock if b % tile == 0]
    served = modes.pad_buckets("kernel_tile")
    assert served == list(range(tile, 5120 + 1, tile)) and set(b for b in stock if b >= 128) <= set(served)
    with pytest.raises(ValueError, match="unknown padding policy"):
        modes.pad_buckets("tile128")
    want = f"--buckets={','.join(map(str, served))}"
    for mode in modes.MODES:
        toks = [t for t in modes.resolve(mode, "/c")["flags"] if t.startswith("--buckets=")]
        assert toks == [want], (mode, toks)                                       # every mode pads by the ONE list
        assert modes.pad_flags(mode) == toks
    monkeypatch.setenv("AF3_JAX_PARAMS_ROOT", "/params"); monkeypatch.setenv("AF3_JAX_REPO", "/repo")
    fast = modes.resolve("fast", "/c")
    argv = stock_pred.compose("p2", "/c", ["--json_path=x.json"], script=fast["script"], mode_flags=fast["flags"], launcher=fast["launcher"], py="/py")
    assert argv.index(want) < argv.index("--json_path=x.json") and argv.count(want) == 1     # the mode's token before the caller's flags (a caller's own --buckets wins by absl order and is then held to the exit rule)
    mine = stock_pred.compose("p2", "/c", ["--json_path=x.json", "--buckets=256,512"], script=fast["script"], mode_flags=fast["flags"], launcher=fast["launcher"], py="/py")
    assert [t for t in mine if t.startswith("--buckets=")] == [want, "--buckets=256,512"] and mine[-1] == "--buckets=256,512"   # a caller's own list rides last and wins by absl order
    assert [t for t in modes.with_n_gpu(modes.resolve(modes.BIG, "/c"), 2)["flags"] if t.startswith("--buckets=")] == [want]   # the row-sharded stack pads by the same list (every entry divides by 2, 4 and 8)
    assert all(b % 8 == 0 for b in served)

import json


# --- the class's tokamax autotuning table (K.39): <class>/tokamax_autotune.json, the TOKAMAX line, the cache-miss policy

def test_tokamax_table_path_is_upstreams(tmp_path):
    """One path, upstream's: run_alphafold.py keeps its tokamax table at <--cache_dir>/tokamax_autotune.json (written when tokamax.autotune succeeds,
    loaded at ModelRunner init); the package reads the same file of the class and composes --cache_dir=<class> for the kit script (warm and pred alike)."""
    import re
    from af3_jax_opt import det, modes, stock_pred
    src = open(os.path.join(TREE, "stock", "src", "run_alphafold.py"), encoding="utf-8").read()
    kit = open(os.path.join(TREE, "opt", "forward", "fast_inference", "patches", "patched_files", "run_alphafold.py"), encoding="utf-8").read()
    for text in (src, kit):
        assert re.search(r"os\.path\.join\(_CACHE_DIR\.value, 'tokamax_autotune\.json'\)", text) and "print(f'Loading tokamax autotune cache from {path}')" in text
        assert "print(f'Tokamax autotune cache saved to {self._autotune_cache_path}')" in text
    assert det.TOKAMAX_TABLE == "tokamax_autotune.json" and det.tokamax_table_path("/c/K") == "/c/K/tokamax_autotune.json" and det.tokamax_table_path(None) is None
    for mode in ("exact", "fast"):                                            # warm and pred compose the same cache flag for the kit script: the class dir
        res = modes.resolve(mode, "/c/K")
        assert stock_pred.cache_flags(res["script"], "/c/K") == ["--cache_dir=/c/K"]
    assert det.TOKAMAX_RX["unavailable"].match("Tokamax autotune unavailable (RuntimeError: AttributeError: x); kernel configs from tokamax's packaged tables, else tokamax_autotuning_cache_miss_fallback=heuristics.")
    assert "print(f'Tokamax autotune unavailable ({type(e).__name__}: {cause}); kernel configs from '" in kit and "tokamax_autotuning_cache_miss_fallback={policy}" in kit and "Tokamax autotune unavailable" not in src


def test_tokamax_account_states_and_line(tmp_path):
    from af3_jax_opt import det
    d = str(tmp_path / "K"); os.makedirs(d)
    absent = det.tokamax_table(d)
    assert absent == {"path": os.path.join(d, "tokamax_autotune.json"), "present": False, "bytes": None, "sha256": None, "entries": None}
    acc = det.tokamax_account(absent, absent, ["Tokamax autotune unavailable (RuntimeError: boom); kernel configs from tokamax's packaged tables, else tokamax_autotuning_cache_miss_fallback=heuristics."], ["py", "run_alphafold_fast.py", "--cache_dir=" + d])
    assert acc["state"] == "absent" and acc["autotune"] == "unavailable:RuntimeError" and acc["policy"] == "heuristics" and acc["entries"] is None
    assert det.tokamax_line(acc) == f"[af3-jax-opt] TOKAMAX table={d}/tokamax_autotune.json state=absent entries=na sha256=none policy=heuristics autotune=unavailable:RuntimeError"
    with open(os.path.join(d, "tokamax_autotune.json"), "w") as f:
        json.dump({"device_kind": "NVIDIA H100 80GB HBM3", "data": [[{"op": 1}, {"cfg": 1}], [{"op": 2}, {"cfg": 2}]]}, f)
    written = det.tokamax_table(d)
    assert written["present"] and written["entries"] == 2 and len(written["sha256"]) == 64 and written["bytes"] > 0
    acc = det.tokamax_account(absent, written, [f"Tokamax autotune cache saved to {d}/tokamax_autotune.json"], ["x"])
    assert (acc["state"], acc["autotune"], acc["entries"]) == ("written", "saved", 2)
    assert det.tokamax_line(acc) == f"[af3-jax-opt] TOKAMAX table={d}/tokamax_autotune.json state=written entries=2 sha256={written['sha256'][:16]} policy=heuristics autotune=saved"
    acc = det.tokamax_account(written, written, [f"Loading tokamax autotune cache from {d}/tokamax_autotune.json"], ["x", "--tokamax_autotuning_cache_miss_fallback=autotune"])
    assert (acc["state"], acc["autotune"], acc["policy"]) == ("loaded", "loaded", "autotune")          # a caller's pass-through flag is named, never assumed
    changed = dict(written, sha256="0" * 64)
    assert det.tokamax_account(written, changed, [], ["x"])["state"] == "rewritten" and det.tokamax_account(absent, absent, [], ["x"])["autotune"] == "none"
    with open(os.path.join(d, "tokamax_autotune.json"), "w") as f:
        f.write("")                                                           # an unreadable table (tokamax 0.0.12 ships its H100 GLU table as an empty file): present, entries unknown
    assert det.tokamax_table(d)["entries"] is None and det.tokamax_table(d)["present"]
    assert det.tokamax_policy(["a", "--tokamax_autotuning_cache_miss_fallback", "error"]) == "error" and det.tokamax_policy([]) == det.TOKAMAX_POLICY_DEFAULT == "heuristics"
    assert det.is_tokamax_line("Loading tokamax autotune cache from /x\n") and not det.is_tokamax_line("[af3-jax-opt] TOKAMAX table=/x")


def test_patch_01_describes_the_patched_script():
    """patches/01_run_alphafold_fast_inference.diff applied to stock/src/run_alphafold.py yields patches/patched_files/run_alphafold.py byte for byte."""
    import shutil, subprocess, tempfile
    fi = os.path.join(TREE, "opt", "forward", "fast_inference", "patches")
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(TREE, "stock", "src", "run_alphafold.py"), os.path.join(tmp, "run_alphafold.py"))
        p = subprocess.run(["patch", "-p1", "-s", "-i", os.path.join(fi, "01_run_alphafold_fast_inference.diff")], cwd=tmp, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        assert open(os.path.join(tmp, "run_alphafold.py"), "rb").read() == open(os.path.join(fi, "patched_files", "run_alphafold.py"), "rb").read()


def test_ttr_names_the_providers_tier_word_per_mode():
    """TTR hands the shared core's transition face the composition's TIER word: fast in mode fast, big in mode big (both regions); `1` = its default word;
    exact does not name the lever (every transition row is tolerance class: exact keeps the stock module by name)."""
    from af3_jax_opt.inprocess import ttr
    assert ttr.TIER_WORDS == ("fast", "exact", "big") and ttr.DEFAULT_WORD == "fast" and ttr.FORM == "af3" and ttr.ACTIVATION == "swiglu"
    assert ttr.word({}) == "fast" and ttr.word({"AF3_JAX_TTR": "1"}) == "fast" and ttr.word({"AF3_JAX_TTR": "big"}) == "big" and not ttr.wanted({"AF3_JAX_TTR": "0"}) and ttr.wanted({"AF3_JAX_TTR": "big"})
    assert ttr.report() == {"installed": False, "word": ttr.word(), "fused": 0, "routed": 0, "rows": {}, "fallback": {}, "uncovered": {}}
    assert modes.TIER_WORD_SWITCHES == {"DATTN": "AF3_JAX_DATTN", "TTR": "AF3_JAX_TTR", "TRIATT_XLA": "AF3_JAX_TRIATT_XLA", "TRIMUL_CD": "AF3_JAX_TRIMUL_CD", "LNP": "AF3_JAX_LNP"} and not set(modes.TIER_WORD_SWITCHES) & set(modes.KIT_MODES["exact"]["levers"])
    assert modes.resolve("fast", "/c")["env"]["AF3_JAX_TTR"] == "fast" and modes.resolve("big", "/c", region="fast")["env"]["AF3_JAX_TTR"] == "fast"
    reach = modes.resolve("big", "/c", region="reach", size={"n_padded": 4096, "n_gpu": 1})
    assert "TTR" in reach["levers"] and reach["env"]["AF3_JAX_TTR"] == "big" and "AF3_JAX_TTR" not in modes.resolve("exact", "/c")["env"]
    assert modes.tier_word_env("big", ["TTR"], {"AF3_JAX_TTR": "fast", "X": "1"}) == {"AF3_JAX_TTR": "big", "X": "1"} and modes.tier_word_env("big", [], {"AF3_JAX_TTR": "fast"}) == {"AF3_JAX_TTR": "fast"}
