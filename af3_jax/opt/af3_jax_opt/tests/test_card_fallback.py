"""On a part the core's Pallas serve layer REFUSES (below cc 8.0; an 8.x/9.x/10.x part without its own table is served the generation's safe
rows, tiles=safe:<gen>, and installs) the FlashPairformer kernels (FPF_TRIMUL / FPF_TRIATT) are HELD by name and the run is refused before
launch (fpf_launch.refuse_held, exit 3 — a mode is all of its levers, nothing waives it); should the add-on be imported past that gate the
SERVED line names `held=triatt:fallback:no_tiles_cc80(triattn),trimul:fallback:no_tiles_cc80(trimul)`, stock classes serving, levers_short
naming the word — while FPF_HOIST / TTR / DATTN / L1 engage on their own evidence (no longer taken down by the add-on's import). With a
tile table the SERVED line carries no `held=` field and nothing changes. No jax, no GPU."""
import sys
import types

import pytest

from af3_jax_opt import modes

L1 = "Featurisation prefetch enabled: 3 worker process(es), 4 item(s) ahead.\nOutput writer enabled: result extraction and output writing run on one writer thread behind the next fold job."   # the two row levers' lines (L1, WRITER) as one transcript element
INST_H100 = "af3_flashpairformer: mode=both (TriangleMultiplication=FlashTriangleMultiplication, GridSelfAttention=FlashGridSelfAttention), jax 0.10.2 backend gpu"
INST_HELD = "af3_flashpairformer: mode=both (TriangleMultiplication=TriangleMultiplication, GridSelfAttention=GridSelfAttention), jax 0.10.2 backend gpu"
SERVED_H100 = "[af3-jax-opt] SERVED trimul fused=3 fallback=0 triatt fused=2 fallback=0 fallback_shapes=none hoist=1 tiles=own:9.0 cc=9.0 dattn=53 dattn_sites=diffusion:5,pairformer_single:48 ttr=3 ttr_routed=1 ttr_fallback=none sbf16=16 sbf16_sites=atom:12,token:4 txla=24 txla_rows=cuda_sm90a:20,k2b_aot:4 txla_aside=none atomattn=7 atomattn_sites=diffusion_decoder:3,diffusion_encoder:3,evoformer_encoder:1 atomattn_fallback=none hlog=2 hlog_dtype=float32 cshare=2 achoist=2 achoist_sites=enc:2,dec:2,passed:2 achoist_aside=none cnoise=off cnoise_rule=none tcd=24 tcd_rows=cd_trimul:24 tcd_aside=none tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2 tcd_uncovered=af3_tmpl_c64_ch64_incoming:2" + "\n[af3-jax-opt] LNP served=150 word=fast rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4 routed=adaptive:40,axis:2,channels:12 aside=none uncovered=none"
SERVED_A100 = ("[af3-jax-opt] SERVED trimul fused=0 fallback=0 triatt fused=0 fallback=0 fallback_shapes=none hoist=1 tiles=none:no-tiles:8.0 cc=8.0 dattn=53 "
               "dattn_sites=diffusion:5,pairformer_single:48 ttr=3 ttr_routed=1 ttr_fallback=none cnoise=off cnoise_rule=none tcd=off tcd_rows=none tcd_aside=none tcd_word=none tcd_uncovered=none held=triatt:fallback:no_tiles_cc80(triattn),trimul:fallback:no_tiles_cc80(trimul)") + "\n[af3-jax-opt] LNP served=150 word=fast rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4 routed=adaptive:40,axis:2,channels:12 aside=none uncovered=none"


def test_held_kernels_are_named_and_the_other_levers_engage():
    ev = modes.lever_evidence("fast", [INST_HELD, L1, SERVED_A100])
    assert not ev["ok"] and ev["held"] == {"FPF_TRIMUL": "fallback:no_tiles_cc80(trimul)", "FPF_TRIATT": "fallback:no_tiles_cc80(triattn)"}
    assert "FPF_TRIMUL held: fallback:no_tiles_cc80(trimul) (no tile table for cc=8.0: the stock class served every call)" in ev["reason"]
    assert "FPF_TRIATT held: fallback:no_tiles_cc80(triattn)" in ev["reason"]
    for other in ("FPF_HOIST", "DATTN", "TTR", "L1 ", "tile table not this GPU's own"):
        assert other not in ev["reason"], (other, ev["reason"])                                   # decoupled: only the held kernels are short
    pl = ev["per_lever"]
    assert pl["FPF_TRIMUL"]["state"] == "skipped" and pl["FPF_TRIMUL"]["reason"] == "fallback:no_tiles_cc80(trimul)"
    assert pl["FPF_TRIATT"]["reason"] == "fallback:no_tiles_cc80(triattn)"
    assert [pl[k]["state"] for k in ("FPF_HOIST", "TTR", "DATTN", "L1")] == ["on"] * 4


def test_one_table_present_engages_that_kernel():
    one = SERVED_A100.replace("trimul fused=0", "trimul fused=3").replace("held=triatt:fallback:no_tiles_cc80(triattn),trimul:fallback:no_tiles_cc80(trimul)", "held=triatt:fallback:no_tiles_cc80(triattn)")
    ev = modes.lever_evidence("fast", [INST_HELD, L1, one])
    assert ev["per_lever"]["FPF_TRIMUL"]["state"] == "on" and ev["per_lever"]["FPF_TRIATT"]["reason"] == "fallback:no_tiles_cc80(triattn)"
    assert "FPF_TRIMUL" not in ev["reason"] and "FPF_TRIATT held" in ev["reason"]


def test_a_served_line_without_held_is_read_as_before():
    ev = modes.lever_evidence("fast", [INST_H100, L1, SERVED_H100])
    assert ev["ok"] and "held" not in ev and modes.SERVED_LINE_RX.search(SERVED_H100).group(35) is None


def test_the_tree_launcher_prints_held_only_when_held(monkeypatch):
    from af3_jax_opt import fpf_launch
    fake = types.ModuleType("af3_flashpairformer")
    st = {"tile_table": "own:9.0", "compute_capability": "9.0", "held": {}}
    fake.served_report = lambda: {}
    fake.status = lambda: dict(st)
    monkeypatch.setitem(sys.modules, "af3_flashpairformer", fake)
    h100 = fpf_launch._served_line()
    assert h100 == "[af3-jax-opt] SERVED trimul fused=0 fallback=0 triatt fused=0 fallback=0 fallback_shapes=none hoist=off tiles=own:9.0 cc=9.0 dattn=off dattn_sites=none ttr=off ttr_routed=0 ttr_fallback=none sbf16=off sbf16_sites=none txla=off txla_rows=none txla_aside=none atomattn=off atomattn_sites=none atomattn_fallback=none hlog=off hlog_dtype=none cshare=off achoist=off achoist_sites=none achoist_aside=none cnoise=off cnoise_rule=none tcd=off tcd_rows=none tcd_aside=none tcd_word=none tcd_uncovered=none"
    assert " held=" not in h100
    st.update(tile_table="none:no-tiles:8.0", compute_capability="8.0", held={"trimul": "fallback:no_tiles_cc80(trimul)", "triatt": "fallback:no_tiles_cc80(triattn)"})
    a100 = fpf_launch._served_line()
    assert a100.endswith(" held=triatt:fallback:no_tiles_cc80(triattn),trimul:fallback:no_tiles_cc80(trimul)") and a100.startswith(h100.replace("tiles=own:9.0 cc=9.0", "tiles=none:no-tiles:8.0 cc=8.0"))
    m = modes.SERVED_LINE_RX.search(a100)
    assert m and m.group(35) == "triatt:fallback:no_tiles_cc80(triattn),trimul:fallback:no_tiles_cc80(trimul)"


def test_refuse_held_exits_3_by_name_before_launch(monkeypatch, capsys):
    """fpf_launch.refuse_held: a kernel the core refused on this GPU (patch.status()['held']) refuses the run BEFORE the launcher — ONE line,
    exit 3, nothing waives it: a mode is all of its levers, it never runs under its name with the stock class standing in (exact / off are the
    modes without the kernel)."""
    from af3_jax_opt import fpf_launch
    fake = types.ModuleType("af3_flashpairformer")
    held = {"trimul": "fallback:no_tiles_cc75(trimul)", "triatt": "fallback:no_tiles_cc75(triattn)"}
    fake.status = lambda: {"held": dict(held)}
    monkeypatch.setitem(sys.modules, "af3_flashpairformer", fake)
    with pytest.raises(SystemExit) as e:
        fpf_launch.refuse_held()
    assert e.value.code == 3
    line = capsys.readouterr().out.strip()
    assert line == ("[af3-jax-opt] NOT ACTIVE: FPF_TRIATT=fallback:no_tiles_cc75(triattn),FPF_TRIMUL=fallback:no_tiles_cc75(trimul): the lever cannot engage on this GPU "
                    "(the core's Pallas serve layer refuses its compute capability); exit 3 (a mode is all of its levers; --mode exact and --mode off run without it)")
    assert modes.refusal_reason(["noise", line]) == line.split("NOT ACTIVE: ", 1)[1] and modes.is_evidence_line(line)
    assert "allow_partial" not in fpf_launch.refuse_held.__code__.co_varnames           # no waiver parameter
    fake.status = lambda: {"held": {}}                                        # every kernel installed (own or safe tiles): nothing to refuse
    fpf_launch.refuse_held()
    assert capsys.readouterr().out == ""

def test_no_opt_out_surface():
    """A mode is all of its levers — there is nothing to opt out with: no --allow-partial flag on the cli, no launcher word, no environment name."""
    from af3_jax_opt import _autoload, big_launch, cli, fpf_launch, stack
    assert not any(hasattr(m, "ALLOW_PARTIAL_ARG") for m in (modes, fpf_launch, big_launch)) and not hasattr(modes, "allow_partial_launcher")
    assert not hasattr(stack, "ENV_ALLOW_PARTIAL") and not any(n.endswith("ALLOW_PARTIAL") for n in stack.DECLARED_ENV + _autoload.DECLARED_ENV)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["pred", "--variant", "p2", "--allow-partial"])              # argparse: unrecognized

