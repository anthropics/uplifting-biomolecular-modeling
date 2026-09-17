"""DATTN — the DiT's and the pairformer's dense pair-biased attention served by the shared core's provider (opt_core.kernels.pallas
serve.attention) by TIER word: the switch's word, the census line's grammar (printed by the model process, read by the wrapper's LEVER line
and by the kernels gate), the memory mode's word in both regions, and the provider's face carrying this model's two families on the stack the
kit runs (no winner is asserted: which arm serves a cell is the provider's measurement, printed at run time, never a constant here)."""
import pytest

from af3_jax_opt import kernels, modes, registry
from af3_jax_opt.inprocess import dattn


def test_switch_word_and_wanted():
    assert dattn.ENV_SWITCH == "AF3_JAX_DATTN" and dattn.DEFAULT_WORD == "fast" and dattn.BIG_WORD == "big" and set(dattn.TIER_WORDS) == {"fast", "exact", "big"}
    assert not dattn.wanted({}) and not dattn.wanted({"AF3_JAX_DATTN": ""}) and not dattn.wanted({"AF3_JAX_DATTN": "0"})
    assert dattn.wanted({"AF3_JAX_DATTN": "1"}) and dattn.wanted({"AF3_JAX_DATTN": "big"}) and dattn.wanted({"AF3_JAX_DATTN": "tokamax@triton"})
    assert dattn.word({"AF3_JAX_DATTN": "1"}) == "fast" and dattn.word({"AF3_JAX_DATTN": "big"}) == "big" and dattn.word({"AF3_JAX_DATTN": "tokamax@triton"}) == "tokamax@triton"
    assert dattn.KINDS == {"diffusion": "af3_dit", "pairformer_single": "af3_single"} and dattn.REBINDS == ("alphafold3.model.network.diffusion_transformer:self_attention",)


def test_census_line_round_trip_and_impls():
    rep = {"word": "fast", "traced": 53, "rows": {"tokamax@triton": 5, "tokamax@xla": 48}, "aside": {}, "uncovered": {}}
    line = dattn.census_line(rep)
    assert line == "[af3-jax-opt] DATTN word=fast served=53 rows=tokamax@triton:5,tokamax@xla:48 aside=none uncovered=none"
    rec = dattn.scan(["noise", line, "more noise"])
    assert rec["word"] == "fast" and rec["served"] == 53 and rec["rows"] == {"tokamax@triton": 5, "tokamax@xla": 48} and rec["aside"] == {} and rec["uncovered"] == {} and rec["line"] == line
    assert dattn.scan(["nothing here"]) is None and dattn.scan([]) is None
    later = dattn.census_line({"word": "big", "traced": 0, "rows": {}, "aside": {"none:unknown_word": 53}, "uncovered": {"af3_dit_s5_h16_d40": 2}})
    assert dattn.scan([line, later])["word"] == "big" and dattn.scan([line, later])["aside"] == {"none:unknown_word": 53} and dattn.scan([line, later])["uncovered"] == {"af3_dit_s5_h16_d40": 2}   # the LAST line wins
    # the tokamax implementation words the served arms call under (what the KERNELS census counts); non-tokamax rows make no counted call
    assert dattn.census_impls({"tokamax@triton": 5, "tokamax@xla": 48}) == ["triton", "xla"]
    assert dattn.census_impls({"tokamax": 3}) == ["None"] and dattn.census_impls({"cudnn": 3, "xla": 1, "pallas_attn": 2}) == [] and dattn.census_impls({}) == []
    assert dattn.census_impls(["tokamax@cudnn:7"]) == ["cudnn"]


def test_kernels_gate_reads_the_served_rows_from_the_census_line():
    exp = kernels.expected("fast", ["FPF_TRIATT", "FPF_TRIMUL", "DATTN"], "triton")
    reading = {**exp, "attn_impls": "cudnn,triton,xla"}
    small = [dattn.census_line({"word": "fast", "traced": 53, "rows": {"tokamax@triton": 5, "tokamax@xla": 48}, "aside": {}, "uncovered": {}})]
    sc = {"probe": reading, "probes": 1, "traps": {k: [] for k in kernels.TRAPS}, "census": {"dpa": {"triton": 5, "xla": 48}, "glu": {}, "lines": 1}, "refused_inprocess": False}
    assert kernels.lever_impls(["DATTN"], small) == {"DATTN": ("triton", "xla")}
    assert kernels.refusal(exp, sc, 0, lever_impls=kernels.lever_impls(["DATTN"], small)) == []          # the xla calls are DATTN's served row tokamax@xla: expected by name
    only_triton = [dattn.census_line({"word": "fast", "traced": 53, "rows": {"tokamax@triton": 53}, "aside": {}, "uncovered": {}})]
    why = kernels.refusal(exp, sc, 0, lever_impls=kernels.lever_impls(["DATTN"], only_triton))
    assert len(why) == 1 and why[0].startswith("dpa_calls: 48 attention call(s) under implementation=xla, the route requests triton")
    assert kernels.refusal(exp, sc, 0, lever_impls=kernels.lever_impls(["DATTN"], [])) == why            # no census line: nothing declared
    assert kernels.refusal(exp, sc, 0, lever_impls=kernels.lever_impls(["FPF_TRIATT"], small)) == why   # DATTN not in the composition: its line is not read
    cud = {**sc, "census": {"dpa": {"triton": 5, "None": 2}, "glu": {}, "lines": 1}}
    assert kernels.refusal(exp, cud, 0, lever_impls=kernels.lever_impls(["DATTN"], small)) and kernels.refusal(exp, cud, 0, lever_impls=kernels.lever_impls(["DATTN"], [dattn.census_line({"word": "tokamax", "traced": 2, "rows": {"tokamax": 2}, "aside": {}, "uncovered": {}})])) == []


def test_the_memory_line_names_the_big_word_and_the_fast_line_names_fast():
    fast = modes.resolve("fast", "/c")
    assert fast["env"]["AF3_JAX_DATTN"] == "1" and dattn.word(fast["env"]) == "fast"
    reach = modes.resolve("big", "/c")
    assert "DATTN" in reach["levers"] and reach["region"] == "reach" and reach["env"]["AF3_JAX_DATTN"] == "big" and dattn.word(reach["env"]) == "big"
    region_fast = modes.resolve("big", "/c", region="fast")
    assert region_fast["region"] == "fast" and region_fast["levers"] == fast["levers"] and region_fast["env"] == fast["env"] and dattn.word(region_fast["env"]) == "fast"   # at or below N* big IS the fast line: its program, its cache class, its word
    assert modes.tier_word_env("fast", ["DATTN"], {"AF3_JAX_DATTN": "1", "X": "y"}) == {"AF3_JAX_DATTN": "1", "X": "y"} and modes.tier_word_env("big", ["DATTN"], {"AF3_JAX_DATTN": "1"}) == {"AF3_JAX_DATTN": "big"}
    assert modes.tier_word_env("big", ["DATTN"], {"AF3_JAX_DATTN": "1"}) == {"AF3_JAX_DATTN": "big"} and modes.tier_word_env("big", ["TTR"], {"A": "1"}) == {"A": "1"}
    two = modes.with_n_gpu(modes.resolve("big", "/c"), 2)
    assert "AF3_JAX_DATTN" not in two["env"]                                  # ROWPAIR owns the site at n_gpu > 1: the lever leaves by its switch


def test_lever_line_quotes_the_census_line():
    served = {"dattn": "53", "dattn_sites": "diffusion:5,pairformer_single:48"}
    line = dattn.census_line({"word": "fast", "traced": 53, "rows": {"tokamax@triton": 53}, "aside": {}, "uncovered": {}})
    rec = modes.per_lever("fast", ["DATTN"], served, [line])
    d = rec["DATTN"] if isinstance(rec, dict) and "DATTN" in rec else [r for r in (rec if isinstance(rec, list) else rec.values()) if (r.get("lever") == "DATTN" if isinstance(r, dict) else False)][0]
    assert d["state"] == "on" and d["evidence"] == "dattn=53 sites=diffusion:5,pairformer_single:48 word=fast rows=tokamax@triton:53 aside=none uncovered=none"
    no_line = modes.per_lever("fast", ["DATTN"], served, [])
    d2 = no_line["DATTN"] if isinstance(no_line, dict) and "DATTN" in no_line else [r for r in (no_line if isinstance(no_line, list) else no_line.values()) if (r.get("lever") == "DATTN" if isinstance(r, dict) else False)][0]
    assert d2["state"] == "on" and d2["evidence"] == "dattn=53 sites=diffusion:5,pairformer_single:48"   # no census line on the transcript: the SERVED count alone
    off = modes.per_lever("fast", ["DATTN"], {"dattn": "0", "dattn_sites": "none"}, [dattn.census_line({"word": "nonsense", "traced": 0, "rows": {}, "aside": {"none:unknown_word": 53}, "uncovered": {}})])
    d3 = off["DATTN"] if isinstance(off, dict) and "DATTN" in off else [r for r in (off if isinstance(off, list) else off.values()) if (r.get("lever") == "DATTN" if isinstance(r, dict) else False)][0]
    assert d3["state"] != "on" and "aside=none:unknown_word:53" in d3.get("reason", "")


def test_registry_names_the_provider_not_a_pinned_implementation():
    d = registry.LEVERS["DATTN"]
    assert d["impl"] == "opt_core.kernels.pallas" and d["strategy"] == "F5.flash_attn_dense" and d["site"] == "opt/af3_jax_opt/inprocess/dattn.py" and "serve.attention" in d["kernel"]
    assert "TIER word" in d["name"] and "AF3_JAX_DATTN=1|big" in d["switch"]
    assert not hasattr(dattn, "IMPLEMENTATION")                              # no kit-side implementation pin: the provider's cells decide, the census line says what served


def test_the_provider_face_carries_both_families_on_this_stack():
    """The shared core's attention face knows this model's two dense-attention families and has a measured forward cell for each on the kit's jax
    line for both cards at the ladder's token counts — the tier words resolve THERE (which arm wins is the provider's measurement, not asserted)."""
    P = pytest.importorskip("opt_core.kernels.pallas")
    fams = P.families("attn")
    assert "af3_dit_s5_h16_d48" in fams and "af3_single_h16_d24" in fams
    assert P.family("attn", kind="af3_dit", heads=16, head_dim=48, n_seq=5) == "af3_dit_s5_h16_d48" and P.family("attn", kind="af3_single", heads=16, head_dim=24) == "af3_single_h16_d24"
    for cc in ("9.0", "8.0"):
        for fam in ("af3_dit_s1_h16_d48", "af3_dit_s5_h16_d48", "af3_single_h16_d24"):      # depth 1 = one sample at trace under the sampler's vmap: the family's measured depth serves it
            for n in (448, 832, 1216):
                for word in ("fast", "big", "exact"):
                    sel = P.select("0.10", cc, "bf16", "attn", fam, n, word=word)
                    assert sel.cell_key is not None and sel.candidates, (cc, fam, n, word)
                    if word == "exact":
                        assert [str(c) for c in sel.candidates][-1] == "xla"          # exact names the stock statement by the provider's rule (no arm of these cells is bitwise the stock lines)


def test_the_kernels_reader_collects_dattns_census_line_so_named_xla_arms_pass_the_gate():
    """A 256-token input on an H100 (a heterodimer of the identity set): the provider served the pairformer's single attention through tokamax@xla
    (its measured cell at that bucket) and the DiT through tokamax@triton; the KERNELS census therefore counts dpa_calls=triton:2,xla:8 while the
    route requests triton. The wrapper's line collector must hand the DATTN census line to the reader (kernels.is_kernels_line), else lever_impls
    sees no declared implementation and the completed run is refused (0.3.32's defect: DONE status=FAILED reason=kernels_refused … 8 attention
    call(s) under implementation=xla)."""
    from af3_jax_opt import kernels
    transcript = [
        "[af3-jax-opt] KERNELS-PROBE flag=triton dpa_site=tokamax triatt_site=fpf trimul_site=fpf transition_site=stock+w:ProviderTransitionBlock attn_impls=cudnn,mosaic_gpu,mosaic_tpu,triton,xla,xla_chunked attn_class=PallasTritonFlashAttention attn_supported=1 glu_impls=mosaic,triton,xla glu_chain=triton,mosaic,xla glu_head=triton tokamax=0.0.12 jax=0.10.2 backend=gpu device=NVIDIA_H100_80GB_HBM3 cc=9.0 xla_flags=--xla_gpu_enable_triton_gemm=false prealloc=true mem_fraction=0.95 first=glu pid=22202 expect=flag=triton;dpa_site=tokamax;triatt_site=fpf;trimul_site=fpf;glu_head=triton;backend=gpu;attn_supported=1;attn_class=present verdict=ok",
        "Running model inference with seed 1 took 9.99 seconds.",
        "[af3-jax-opt] DATTN word=fast served=10 rows=tokamax@triton:2,tokamax@xla:8 aside=none uncovered=none",
        "[af3-jax-opt] KERNELS-CENSUS pid=22202 dpa_calls=triton:2,xla:8 glu_calls=None:24 probe_lines=1",
    ]
    klines = [ln for ln in transcript if kernels.is_kernels_line(ln)]
    assert any(ln.startswith("[af3-jax-opt] DATTN ") for ln in klines)
    acc = kernels.account(mode="fast", n_gpu=1, levers=list(modes.KIT_MODES["fast"]["levers"]), argv=["run_alphafold.py"], env={}, lines=klines, rc=0)
    assert acc["ok"] and acc["reasons"] == [], acc["reasons"]
    without = [ln for ln in klines if not ln.startswith("[af3-jax-opt] DATTN ")]
    acc2 = kernels.account(mode="fast", n_gpu=1, levers=list(modes.KIT_MODES["fast"]["levers"]), argv=["run_alphafold.py"], env={}, lines=without, rc=0)
    assert not acc2["ok"] and "implementation=xla" in acc2["reasons"][0]   # the census line is what names those arms
