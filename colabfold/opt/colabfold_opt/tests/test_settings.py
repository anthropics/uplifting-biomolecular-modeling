from colabfold_opt import settings


def test_no_flags_is_upstreams_defaults():
    assert (settings.models_per_seed(), settings.num_seeds()) == (5, 1)                      # batch.py:1877 / :1864
    assert (settings.models_per_seed([]), settings.num_seeds(None)) == (5, 1)
    d = settings.describe([])
    assert d["stock_flags"] == [] and (d["models_per_seed"], d["num_seeds"]) == (5, 1)
    assert d["defaults"]["--num-models"] == 5 and d["defaults"]["--random-seed"] == 0 and d["defaults"]["--model-type"] == "auto"
    assert d["model_config_defaults"] == {"num_recycle": 20, "recycle_early_stop_tolerance": 0.5}
    assert set(d) == {"stock_flags", "models_per_seed", "num_seeds", "defaults", "model_config_defaults"}


def test_passed_stock_flags_are_read_at_stocks_names():
    row = ["--num-recycle", "4", "--num-models", "2", "--random-seed", "7", "--num-seeds", "1"]
    assert (settings.models_per_seed(row), settings.num_seeds(row)) == (2, 1)
    assert settings.flag_value(row, "--num-recycle") == "4" and settings.flag_value(row, "--random-seed") == "7"
    assert settings.flag_value(row, "--model-type") == "auto"                                 # not passed: upstream's default
    assert settings.models_per_seed(["--num-models=3"]) == 3                                  # the `--flag=value` form
    assert settings.num_seeds(["--num-seeds", "2", "--num-seeds", "4"]) == 4                  # the last occurrence wins, as argparse reads it
    d = settings.describe(row)
    assert d["stock_flags"] == row and (d["models_per_seed"], d["num_seeds"]) == (2, 1)


def test_the_module_carries_no_preset_table():
    assert not hasattr(settings, "PRESETS") and not hasattr(settings, "DEFAULT_PRESET") and not hasattr(settings, "flags")
    assert settings.UPSTREAM_DEFAULTS["--num-recycle"] == (None, 1842) and settings.UPSTREAM_DEFAULTS["--disable-unified-memory"] == (False, 2089)
