"""Sampling settings are rf3 fold's own key=value overrides, passed through verbatim: with none the argv beyond the upstream CLI is
the inputs, the output directory, the weights and the seed; the kit's own four keys are refused by name; describe() reports the pin's
copy of upstream's defaults overlaid with the caller's overrides."""

import pytest

from .. import settings, stack, stock_fold


def test_no_override_is_upstreams_defaults_plus_the_seed():
    assert settings.overrides(None, 42) == ["seed=42"] == settings.overrides([], 42)
    assert settings.overrides((), None) == []
    assert stock_fold.command("/v/bin/python", "/in.json", "/out", "/w.ckpt", (), 42) == \
        ["/v/bin/python", "-m", "rf3.cli", "fold", "inputs=/in.json", "out_dir=/out", "ckpt_path=/w.ckpt", "seed=42"]


def test_stock_route_environment():
    """The stock arm's environment is the caller's: the kits' names stripped (stock/PINS.json), PYTHONDONTWRITEBYTECODE added, nothing else."""
    base = {"PATH": "/usr/bin", "HOME": "/h", "RF3_HOIST": "1", "FPF_RF3_PAD": "16", "PYTHONPATH": "/x"}
    env, stripped = stock_fold.clean_env(stack.pins(), environ=base)
    assert stripped == ["FPF_RF3_PAD", "RF3_HOIST"]
    assert env == {"PATH": "/usr/bin", "HOME": "/h", "PYTHONPATH": "/x", "PYTHONDONTWRITEBYTECODE": "1"}


def test_overrides_pass_through_verbatim_in_order_before_the_seed():
    toks = ["diffusion_batch_size=1", "num_steps=200", "early_stopping_plddt_threshold=null"]
    assert settings.overrides(toks, 0) == toks + ["seed=0"]
    assert stock_fold.command("/p", "/i.json", "/o", "/w", ["n_recycles=1", "diffusion_batch_size=1", "num_steps=2", "early_stopping_plddt_threshold=null"], 1)[4:] == \
        ["inputs=/i.json", "out_dir=/o", "ckpt_path=/w", "n_recycles=1", "diffusion_batch_size=1", "num_steps=2", "early_stopping_plddt_threshold=null", "seed=1"]
    assert settings.overrides(["verbose=true"], None) == ["verbose=true"]                       # any upstream key: hydra judges it, not the kit


def test_the_kits_own_keys_and_non_overrides_are_refused_by_name():
    for bad, words in (("inputs=/x.json", "inputs= is written by the kit from --input"),
                       ("out_dir=/o", "out_dir= is written by the kit from --out_dir"), ("ckpt_path=/w", "ckpt_path= is written by the kit from --ckpt"),
                       ("fast", "not a key=value override"), ("=5", "not a key=value override")):
        with pytest.raises(ValueError, match=words.replace("(", "\\(")):
            settings.check([bad])
    assert settings.check(["seed=7"]) == ["seed=7"] and settings.overrides(["seed=7"], None) == ["seed=7"]   # rf3 fold's own seed override passes
    with pytest.raises(ValueError, match="together with --seeds"):
        settings.overrides(["seed=7"], 7)                                                    # --seeds AND seed=: the seed named twice


def test_effective_settings_are_the_pinned_cli_defaults_overlaid():
    """stock/PINS.json "settings"."cli_defaults" (the pin's copy of rf3.yaml:12-21) is what describe() reports with no override; an
    override of one of KEYS replaces that key's value (hydra scalars parsed); the seed is not a settings key; PINS.json names no preset."""
    pins = stack.pins()["settings"]
    cli = {k: v for k, v in pins["cli_defaults"].items() if k not in ("seed", "source")}
    assert settings.upstream_defaults() == cli == {"n_recycles": 10, "diffusion_batch_size": 5, "num_steps": 50, "early_stopping_plddt_threshold": 0.5}
    assert settings.describe(()) == {**cli, "overrides": []}
    d = settings.describe(["num_steps=200", "early_stopping_plddt_threshold=null", "verbose=true"])
    assert d == {**cli, "num_steps": 200, "early_stopping_plddt_threshold": None, "overrides": ["num_steps=200", "early_stopping_plddt_threshold=null", "verbose=true"]}
    assert settings.describe(["early_stopping_plddt_threshold=0"])["early_stopping_plddt_threshold"] == 0
    assert pins["cli_defaults"]["seed"] is None and "seed" not in settings.KEYS
    assert not any(k.startswith("preset") for k in pins) and not hasattr(settings, "PRESETS")
