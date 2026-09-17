"""LNP — the standalone-LayerNorm provider binding (inprocess/lnp.py): the unit decision from the call's facts, the word the switch names, the
census line and its reading by modes (CPU only; the provider's rows run on the GPU box, judged there by the line's counts)."""
import importlib.util, os
from af3_jax_opt import fpf_launch, modes, registry

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("af3_jax_opt.inprocess.lnp", os.path.join(HERE, "inprocess", "lnp.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


LNP = _load()
MEASURED = ("pair_c128", "tmpl_c64", "tmpl_c64_t4", "msa_c64_s1024", "msa_c256_s512", "atom_c128", "dit_c768_s5", "single_c384")   # the provider's ln families (PALLAS_CELLS.json, jax 0.10 line)
PLAIN = dict(create_scale=True, create_offset=True, axis=-1, param_axis=None, eps=1e-5, upcast=True, is_16bit=True, measured=MEASURED)


def test_the_switch_and_the_word():
    assert LNP.ENV_SWITCH == "AF3_JAX_LNP" == next(iter(modes.TREE_LEVER_ENV["LNP"])) == modes.TIER_WORD_SWITCHES["LNP"]
    assert not LNP.wanted({}) and not LNP.wanted({"AF3_JAX_LNP": "0"}) and LNP.wanted({"AF3_JAX_LNP": "1"}) and LNP.wanted({"AF3_JAX_LNP": "big"})
    assert LNP.word({"AF3_JAX_LNP": "1"}) == "fast" and LNP.word({"AF3_JAX_LNP": "big"}) == "big" and LNP.word({"AF3_JAX_LNP": "cd_ln"}) == "cd_ln"
    assert modes.lever_env("fast", modes.KIT_MODES["fast"]["levers"])["AF3_JAX_LNP"] == "fast"                    # fast: the tier word fast
    assert modes.resolve("big", "/c")["env"]["AF3_JAX_LNP"] == "big"                                          # big's composition: the tier word big, literally
    assert modes.resolve("big", "/c", region="fast")["env"]["AF3_JAX_LNP"] == "fast"                            # region fast IS the fast line (word fast)
    assert "LNP" in modes.KIT_MODES["fast"]["levers"] and "LNP" not in modes.KIT_MODES["exact"]["levers"]          # exact keeps the stock LayerNorm by name
    assert "LNP" in modes.ROWPAIR_SUPERSEDES and registry.LEVERS["LNP"]["strategy"] == "F5.row_layernorm"
    assert ("lnp", "LNP") in fpf_launch.TREE_LEVERS and LNP.REBINDS == ("alphafold3.model.components.haiku_modules:LayerNorm",)


def test_units_are_decided_from_the_call():
    c = LNP.classify
    assert c(shape=(832, 832, 128), **PLAIN)[:2] == ("pair", {"unit": "pair", "n_tokens": 832})                    # pair plane
    assert c(shape=(5, 832, 832, 128), **PLAIN)[0] == "pair"                                                        # batched (per-sample) pair plane
    assert c(shape=(832, 832, 64), scope="template_embedding/single_template_embedding/output_layer_norm", **PLAIN)[:2] == ("tmpl", {"unit": "tmpl", "n_tokens": 832})
    assert c(shape=(4, 448, 448, 64), scope="template_embedding/x", **PLAIN)[1] == {"unit": "tmpl", "n_tokens": 448, "t": 4}   # a measured template count: its own family
    assert c(shape=(3, 448, 448, 64), scope="template_embedding/x", **PLAIN)[1] == {"unit": "tmpl", "n_tokens": 448}           # an unmeasured count: the plain template family
    assert c(shape=(1024, 832, 64), scope="evoformer/msa_stack/outer_product_mean/layer_norm_input", **PLAIN)[:2] == ("msa", {"unit": "msa", "n_tokens": 832, "n_seq": 1024})
    assert c(shape=(1024, 1024, 64), scope="evoformer/msa_stack/msa_attention1/act_norm", **PLAIN)[0] == "msa"    # a square MSA plane (S == N): the scope decides, not the shape
    assert c(shape=(300, 832, 64), scope="evoformer/msa_stack/x", **PLAIN)[1]["n_seq"] == 1024                     # the provider's depth rule: nearest measured depth at or above
    assert c(shape=(2048, 832, 64), scope="evoformer/msa_stack/x", **PLAIN)[1]["n_seq"] == 1024                    # ... else the deepest measured
    assert LNP.msa_depth(64, 300, ()) == 300                                                                        # no table: the fact as it is


def test_what_the_stock_class_keeps_by_rule():
    c = LNP.classify
    assert c(shape=(832, 384), **PLAIN)[2] == "rank"                                                                # single [N, 384]: rank 2 (and C = 384 is no row's)
    assert c(shape=(5, 832, 384), **PLAIN)[2] == "channels" and c(shape=(5, 832, 768), **PLAIN)[2] == "channels"   # single / diffusion-transformer widths: no provider row
    assert c(shape=(832, 832, 128), **{**PLAIN, "create_offset": False})[2] == "adaptive"                          # adaptive LayerNorm (no offset): the diffusion transformer's pair_input_layer_norm
    assert c(shape=(5, 4480, 128), **{**PLAIN, "create_scale": False, "create_offset": False})[2] == "adaptive"    # atom transformer adaptive norms
    assert c(shape=(128, 832, 832), **{**PLAIN, "axis": 0, "param_axis": (0,)})[2] == "axis"                       # the triangle multiplication's centre norm
    assert c(shape=(256, 832, 128), **PLAIN)[2] == "rows"                                                           # a pair row block (memory line): not the measured plane
    assert c(shape=(256, 832, 64), scope="template_embedding/x", **PLAIN)[2] == "rows"                             # a template row block
    assert c(shape=(832, 832, 128), **{**PLAIN, "eps": 1e-6})[2] == "eps"
    assert c(shape=(832, 832, 128), **{**PLAIN, "upcast": False})[2] == "no_upcast"
    assert c(shape=(832, 832, 96), **PLAIN)[2] == "channels" and c(shape=(832, 832, 256), **PLAIN)[2] == "channels"   # not a power of two / no unit at that width


def test_the_line_and_its_reading():
    rep = {"installed": True, "word": "big", "traced": 150, "rows": {"cd_ln": 62, "xla": 88}, "units": {"pair": 58, "msa": 88, "tmpl": 4}, "routed": {"adaptive": 40, "channels": 12},
           "aside": {}, "uncovered": {"pair_c128": 2}}
    ln = LNP.line(rep)
    assert ln == "[af3-jax-opt] LNP served=150 word=big rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4 routed=adaptive:40,channels:12 aside=none uncovered=pair_c128:2"
    f = modes.lnp_evidence(["noise", ln])
    assert f == {"lnp": "150", "lnp_word": "big", "lnp_rows": "cd_ln:62,xla:88", "lnp_units": "msa:88,pair:58,tmpl:4", "lnp_routed": "adaptive:40,channels:12", "lnp_aside": "none", "lnp_uncovered": "pair_c128:2"}
    assert LNP.line({**rep, "installed": False}).startswith("[af3-jax-opt] LNP served=off ") and modes.lnp_evidence([])["lnp"] == "absent"
    assert modes.LNP_LINE_RX in modes.EVIDENCE_RXS


def test_evidence_is_fail_closed():
    inst = "af3_flashpairformer: mode=both (TriangleMultiplication=FlashTriangleMultiplication, GridSelfAttention=FlashGridSelfAttention), jax 0.10.2 backend gpu"
    L1 = "Featurisation prefetch enabled: 3 worker process(es), 4 item(s) ahead.\nOutput writer enabled: result extraction and output writing run on one writer thread behind the next fold job."
    from af3_jax_opt.tests.test_card_fallback import SERVED_H100
    served, lnp = SERVED_H100.split("\n")
    levers = modes.resolve("fast")["levers"]
    assert modes.lever_evidence("fast", [inst, L1, served, lnp], levers)["ok"]
    ev = modes.lever_evidence("fast", [inst, L1, served], levers)                                                  # no LNP line: levers_short by name
    assert not ev["ok"] and "LNP not served: lnp=absent" in ev["reason"] and ev["per_lever"]["LNP"]["state"] == "skipped"
    ev = modes.lever_evidence("fast", [inst, L1, served, lnp.replace("served=150", "served=0")], levers)            # installed, nothing served
    assert not ev["ok"] and "LNP not served: lnp=0" in ev["reason"]
    ev = modes.lever_evidence("fast", [inst, L1, served, lnp.replace("aside=none", "aside=cd_ln:levers_off:3")], levers)   # a provider refusal by name at a routed unit: named, levers_short
    assert not ev["ok"] and "LNP refused at cd_ln:levers_off:3" in ev["reason"]
    kept = [l for l in levers if l != "LNP"]                                                                        # MODEL_OPT_LEVERS_OFF=LNP: the reduced composition wants no LNP line
    assert modes.lever_evidence("fast", [inst, L1, served], kept)["ok"]
    on = modes.lever_evidence("fast", [inst, L1, served, lnp], levers)["per_lever"]["LNP"]
    assert on["state"] == "on" and on["evidence"].startswith("lnp=150 word=fast rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4")
