"""modes.py: the mode list, the default, and the kit tables read from the kit files."""
import pytest

from progen2_opt import modes


def test_modes_and_default():
    assert modes.MODES == ("exact", "off")
    assert modes.DEFAULT_MODE == "exact"
    for gone in ("RECIPES", "DEFAULT_RECIPE", "RECIPE_ENV", "RETIRED", "RECIPE_MODES"):                                  # one composition per mode: no lever inside a mode, no retired-name table
        assert not hasattr(modes, gone), gone
    assert set(modes.KIT_MODES) == {"exact"}
    assert modes.kit_mode("exact") is modes.KIT_MODES["exact"] and modes.kit_mode("exact").sample == "serving"
    for bad in ("exact-r2", "fast", "turbo"):
        with pytest.raises(ValueError, match="unknown kit mode"):
            modes.kit_mode(bad)
    assert modes.VARIANTS == ("small", "medium", "oas", "base", "large", "bfd90", "xlarge")
    assert modes.UPSTREAM_NAME["bfd90"] == "progen2-BFD90" and modes.UPSTREAM_NAME["small"] == "progen2-small"


def test_kit_tables_read_from_kit_files(home):
    import os
    serving = os.path.join(home, modes.SERVING_KIT_RELPATH)
    auto = modes.generation_levers(serving)
    assert auto == ("oneread_mmap", "sampler_exact"), auto                     # the generation kit's own lever table (progen2_decode.py AUTO_LEVERS)
    comp = modes.decode_component(serving)
    assert comp["present"] and comp["module"] == "kit_t1" and comp["level"] == "t3s" and comp["levers"] == ["resident_rotary", "static_kv"]
    assert modes.GENERATION_MODULE == "progen2_decode" and modes.ROUTES == ("sample", "score") and modes.NOT_SHIPPED_MODES == ("fast", "big")
    ew = modes.ew_exact_on(os.path.join(home, modes.EW_KIT_RELPATH))
    assert ew == ("gelu", "rotary", "residual", "glue", "ln"), ew                # the elementwise kit's exact table (v0_ew/patches.py)
    consts = modes.scoring_kit_constants(os.path.join(home, modes.SCORING_KIT_RELPATH))
    assert consts["KIT"] == "v0_score_r3_1" and consts["MAX_ROWS_DEFAULT"] >= 1


def comp_dir(home):
    import os
    return os.path.join(home, modes.SERVING_KIT_RELPATH, "components", "plm_transfer_t1_v0")


def test_resolve_exact_and_off(home):
    res = modes.resolve("exact", "small", home)
    assert res.routes == {"sample": "serving", "score": "forward"}
    assert res.optimizations["sample"] == ["oneread_mmap", "sampler_exact", "resident_rotary", "static_kv"]           # one token per lever: the kit's table + the component's levers
    assert res.optimizations["score"][:3] == list(modes.SCORING_OPTIMIZATIONS) == ["rotary_tables", "one_forward_per_direction", "host_pipeline"] and res.optimizations["score"][3] == "ew:gelu"
    assert set(res.kit_dirs) == {"serving", "decode", "forward", "scoring", "ew"} and res.kit_dirs["decode"] == comp_dir(home)   # the decode component is a lever of the mode: its dir is gated like a kit dir
    assert modes.describe_line(res) == "sample=serving:pipeline_v0_4 score=forward:v0_score_r3_1+v0_ew"
    off = modes.resolve("off", "small", home)
    assert off.routes == {"sample": None, "score": None} and off.kit_dirs == {}


def test_unknown_variant_refused():
    with pytest.raises(ValueError):
        modes.check_variant("huge")


def test_model_names_are_the_stocks():
    """--model takes the stock's own checkpoint names (sample.py / likelihood.py `models`), plus the one lowercase alias of progen2-BFD90."""
    assert modes.MODEL_NAMES == ("progen2-small", "progen2-medium", "progen2-oas", "progen2-base", "progen2-large", "progen2-BFD90", "progen2-xlarge")
    assert [modes.variant_of_model(n) for n in modes.MODEL_NAMES] == list(modes.VARIANTS)
    assert modes.variant_of_model("progen2-bfd90") == "bfd90" and modes.MODEL_ALIASES == {"progen2-bfd90": "bfd90"}
    assert modes.STOCK_DEFAULT_MODEL == {"sample": "progen2-large", "score": "progen2-base"}          # sample.py L112 / likelihood.py L122
    for bad in ("small", "progen2-SMALL", "progen2-huge", ""):
        with pytest.raises(ValueError, match="invalid choice"):
            modes.variant_of_model(bad)
    for gone in ("DEFAULT_VARIANT",):
        assert not hasattr(modes, gone), gone
