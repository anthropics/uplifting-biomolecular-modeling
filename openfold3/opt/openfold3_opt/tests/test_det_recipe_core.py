"""det.py's level-1 recipe is the core's Recipe shape (opt_core.precision.recipe.torch_recipe) with the kit's words unchanged."""
from openfold3_opt import det


def test_words_unchanged():
    assert det.ENV == {"OF3_DETERMINISTIC": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"} == dict(det.recipe(1).env)   # the kit's words == the core recipe of the kit switch
    assert det.recipe(1).level == 1 and det.recipe(0).level == 0 and dict(det.recipe(0).env) == {}
    assert det.GRAPHED_EXTRA == {"OF3_GRAPHS_STRICT": "1"}
    env = {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    added = det.apply_env(1, graphed=True, environ=env)
    assert env == {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "OF3_DETERMINISTIC": "1", "OF3_GRAPHS_STRICT": "1"} and "CUBLAS_WORKSPACE_CONFIG" not in added
    assert det.apply_env(0, environ={}) == {}
    assert det.describe(1).startswith("det=1 env=CUBLAS_WORKSPACE_CONFIG,OF3_DETERMINISTIC ") and det.describe(0).startswith("det=0")
