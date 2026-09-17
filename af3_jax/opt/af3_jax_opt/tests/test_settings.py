"""The stock command line's own knobs pass through; the package presets nothing. Upstream's defaults are read from the stock source
(settings.upstream_defaults), no stock flag is refused in any mode, and the model-shape flags a caller states
are picked out as the stated model shape (settings.stated)."""
import os
import re

import pytest

from .conftest import TREE
from af3_jax_opt import modes, settings, stock_pred


def test_upstream_defaults_are_read_from_the_source():
    """An independent parse of stock/src/run_alphafold.py: each DEFINE_* block's default, by name; the fork's cache flag default reads the same way."""
    src = open(os.path.join(TREE, "stock", "src", "run_alphafold.py"), encoding="utf-8").read()
    for name in settings.STOCK_FLAG_NAMES:
        block = src[src.index(f"'{name}'"):]
        block = block[:block.index(")")]
        m = re.search(r"'%s',\s*(?:default=)?'?([\w.]+)'?," % name, block)
        want = m.group(1)
        assert settings.upstream_defaults()[name] == want, name
    assert settings.upstream_defaults() == {"num_recycles": "10", "num_diffusion_samples": "5", "flash_attention_implementation": "triton"}
    assert "resolve_msa_overlaps" not in settings.STOCK_FLAG_NAMES and "save_terms_of_use" not in settings.STOCK_FLAG_NAMES
    assert re.search(r"'resolve_msa_overlaps',\s*True", src) and re.search(r"'save_terms_of_use',\s*True", src)
    assert settings.FIXED_SEEDS == (1,)
    assert not hasattr(settings, "PRESETS") and not hasattr(settings, "DEFAULT_PRESET")             # nothing is preset: --mode plus the stock flags


def test_stated_picks_the_model_shape_flags():
    toks = ["--num_recycles=3", "--buckets=256,512", "--flash_attention_implementation=xla", "--resolve_msa_overlaps=false", "--num_diffusion_samples", "2"]
    assert settings.stated(toks) == ["--num_recycles=3", "--flash_attention_implementation=xla", "--num_diffusion_samples"]
    assert settings.stated([]) == [] and settings.flag_value(toks, "--num_recycles") == "3"


def test_cache_flags_by_script_and_caller():
    """off composes the EMPTY class unless the caller names a --cache_dir (then none: the caller's rides last); the kit script always its class."""
    assert stock_pred.cache_flags(modes.STOCK_SCRIPT, "/c", []) == ["--cache_dir="] == [settings.EMPTY_CACHE_FLAG]
    assert stock_pred.cache_flags(modes.STOCK_SCRIPT, "/c", ["--cache_dir=/mine"]) == [] == stock_pred.cache_flags(modes.STOCK_SCRIPT, "/c", ["--cache_dir", "/mine"])
    assert stock_pred.cache_flags(modes.FAST_SCRIPT, "/c", ["--x"]) == ["--cache_dir=/c"]
    assert stock_pred.caller_cache_dir(["--a", "--cache_dir=/one", "--cache_dir", "/two"]) == "/two" and stock_pred.caller_cache_dir([]) is None


def test_rows_carry_no_box_path():
    assert modes.ROW_NAMES == ("COMMON", "FAST")
    assert not [t for name in modes.ROW_NAMES for t in modes.kit_rows()[name].split() if t.startswith("/") or "/vol" in t]


# --- merged from test_buckets_line.py (file consolidation, tests unchanged) ---

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the af3_jax/ tree


def _argv(user):
    return ["python", "run_alphafold.py", "--norun_data_pipeline", *modes.pad_flags("fast", HOME), "--output_dir=/o", "--json_path=/j", *user]


def test_mode_list_without_a_caller_flag():
    eff = stock_pred.effective_buckets(_argv([]), HOME)
    assert eff["source"] == "mode" and eff["occurrences"] == 1
    assert eff["buckets"] == [int(t) for t in modes.pad_flags("fast", HOME)[0].split("=", 1)[1].split(",")]
    assert stock_pred.buckets_line(eff).startswith("[af3-jax-opt] BUCKETS buckets=" + ",".join(map(str, eff["buckets"])) + " source=mode occurrences=1")


def test_a_callers_later_flag_wins():
    for user in (["--buckets=256,512"], ["--buckets", "256,512"]):
        eff = stock_pred.effective_buckets(_argv(user), HOME)
        assert eff == {"buckets": [256, 512], "source": "caller", "occurrences": 2}, eff
        assert "buckets=256,512 source=caller occurrences=2" in stock_pred.buckets_line(eff)


def test_nothing_on_the_line_is_the_forks_default():
    eff = stock_pred.effective_buckets(["python", "run_alphafold.py", "--json_path=/j"], HOME)
    assert eff["source"] == "fork_default" and eff["occurrences"] == 0 and eff["buckets"] == settings.upstream_buckets(HOME)
