"""Upstream's knobs come from stock/PINS.json cli_defaults (their names and defaults); the deterministic recipe is the kit's literal."""
import argparse
import os

import pytest

from pxdesign_opt import det, options

UPSTREAM_ARGV = ["--N_step", "400", "--N_sample", "5", "--dtype", "bf16", "--eta_type", "const", "--eta_min", "2.5", "--eta_max", "2.5",
                 "--num_workers", "16", "--use_msa", "true", "--use_fast_ln", "true"]


def test_no_flag_is_upstreams_defaults(tree):
    pins = options.read_pins(os.path.join(tree, "stock", "PINS.json"))
    o = options.resolve(pins=pins)
    assert o.argv() == UPSTREAM_ARGV                                                     # upstream's documented defaults, rendered in OPTION_ORDER
    assert o.values == o.defaults == {k: str(pins["cli_defaults"][k]).lower() if isinstance(pins["cli_defaults"][k], bool) else str(pins["cli_defaults"][k]) for k in options.OPTION_ORDER}
    assert o.given == {} and o.seeds == () and o.seeds_arg() is None                     # no --seeds: nothing rendered (upstream derives one from the clock)
    assert not hasattr(o, "env")                                                              # LAYERNORM_TYPE is stock's environment variable, not an option: nothing here couples to it
    assert sorted(pins) == ["archive_recipe", "ccd_cache", "cli_defaults", "model", "pinned_stack", "pins", "python", "stock_environment", "upstream", "variants", "weights"]   # cli_defaults is the one knob table


def test_given_words_pass_through(tree):
    pins = options.read_pins(os.path.join(tree, "stock", "PINS.json"))
    o = options.resolve(N_sample=2, seeds="317000,317001", pins=pins)
    assert o.values["N_sample"] == "2" and o.seeds == (317000, 317001) and o.seeds_arg() == "317000,317001"
    assert o.given == {"N_sample": "2", "seeds": "317000,317001"}
    assert options.resolve(N_sample=5, pins=pins).given == {"N_sample": "5"}              # a given word equal to the default is still the caller's word
    assert options.resolve(eta_min="3", dtype="fp32", use_msa="false", pins=pins).argv()[4:] == ["--dtype", "fp32", "--eta_type", "const", "--eta_min", "3", "--eta_max", "2.5", "--num_workers", "16", "--use_msa", "false", "--use_fast_ln", "true"]
    q = options.resolve(use_fast_ln="false", pins=pins)
    assert q.values["use_fast_ln"] == "false" and q.given == {"use_fast_ln": "false"}
    assert options.given_argv(o) == ["--seeds", "317000,317001", "--N_sample", "2"]


def test_the_flags_are_upstreams_names():
    ap = argparse.ArgumentParser(); options.add_arguments(ap)
    a = ap.parse_args(["--N_step", "10", "--eta_max", "3.5", "--use_fast_ln", "false", "--seeds", "7"])
    o = options.from_args(a, pins={"cli_defaults": {"N_step": 400, "N_sample": 5, "dtype": "bf16", "eta_type": "const", "eta_min": 2.5, "eta_max": 2.5, "num_workers": 16, "use_msa": True, "use_fast_ln": True}})
    assert o.given == {"N_step": "10", "eta_max": "3.5", "use_fast_ln": "false", "seeds": "7"} and o.argv()[:2] == ["--N_step", "10"]
    flags = sorted(o for a in ap._actions for o in a.option_strings if o.startswith("--"))
    assert flags == sorted(["--help", "--seeds"] + [f"--{k}" for k in options.OPTION_ORDER])   # upstream's knob names + --seeds, nothing else
    with pytest.raises(SystemExit):
        ap.parse_args(["--N_step", "x"])                                                   # upstream's int knobs stay ints


def test_recipe_is_the_kits_literal(tree):
    """The recipe's one variable, and upstream's seeding call it names (protenix/utils/seed.py: seed_everything(seed, deterministic))."""
    assert det.RECIPE_ENV == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    src = open(os.path.join(tree, "stock", "src", "Protenix", "protenix", "utils", "seed.py"), encoding="utf-8").read()
    assert "def seed_everything(" in src and "deterministic" in src and "CUBLAS_WORKSPACE_CONFIG" in src
    env = {}
    assert det.apply_env(0, env) == {} and env == {}
    assert det.apply_env(1, env) == det.RECIPE_ENV and env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_console_argv_maps_to_inference_argv(monkeypatch):
    """The in-process route forwards exactly as the console script does: the shared options as `common`, the rest verbatim."""
    import sys, types
    from pxdesign_opt import infer_loop
    calls = []
    cli = types.ModuleType("pxdesign.runner.cli")
    cli.build_argv = lambda common, extra: (calls.append((dict(common), list(extra))) or ["<built>"])
    pkg = types.ModuleType("pxdesign"); runner = types.ModuleType("pxdesign.runner")
    monkeypatch.setitem(sys.modules, "pxdesign", pkg); monkeypatch.setitem(sys.modules, "pxdesign.runner", runner); monkeypatch.setitem(sys.modules, "pxdesign.runner.cli", cli)
    argv = infer_loop.upstream_argv("t.json", "out", "/ckpt", ["--N_step", "400", "--N_sample", "5", "--dtype", "bf16", "--eta_type", "const", "--eta_min", "2.5", "--eta_max", "2.5", "--num_workers", "16", "--use_msa", "true", "--use_fast_ln", "true"], "317000", ["--foo", "1"])
    assert infer_loop.to_inference_argv(argv) == ["<built>"]
    common, extra = calls[0]
    assert common == {"input_json_path": "t.json", "dump_dir": "out", "N_step": "400", "N_sample": "5", "dtype": "bf16", "eta_type": "const", "eta_min": "2.5", "eta_max": "2.5"}
    assert extra == ["--num_workers", "16", "--use_msa", "true", "--use_fast_ln", "true", "--seeds", "317000", "--load_checkpoint_dir", "/ckpt", "--foo", "1"]
    assert infer_loop.upstream_argv("t.json", "out", "/ckpt", ["--N_step", "400"], None) == ["-i", "t.json", "-o", "out", "--N_step", "400", "--load_checkpoint_dir", "/ckpt"]   # no --seeds: none rendered
