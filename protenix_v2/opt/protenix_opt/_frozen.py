"""The frozen-weights rule, LEAF-LEVEL: imported by the start-up hook in every process, so this module imports nothing of the
package, the core or the stack — stdlib `os` only.

The stock CLI DOWNLOADS what it does not find under PROTENIX_ROOT_DIR (stock/src/runner/inference.py:291-347 `download_inference_cache`,
called by runner/batch_inference.py:427 at every `protenix pred`): the four data caches (:299-316; paths configs_data.py:251-260), the two
template caches when `--use_template` is true (:318-336; configs_data.py:243-248), the checkpoint `{load_checkpoint_dir}/{model_name}.pt`
(:338-347; load_checkpoint_dir = PROTENIX_ROOT_DIR/checkpoint, configs_inference.py:29; model_name = `-n/--model_name`, default
protenix_base_default_v1.0.0, batch_inference.py:609-614) and, for a model name containing "esm" or "ism", the ESM checkpoints (:349-378).
A frozen-weights run must never reach that fallback: with the switch PROTENIX_ROOT_FROZEN=1 the kit
refuses BY NAME before anything imports the stock runner — `cli.cmd_pred` on the CLI route, `_autoload.install` at interpreter
start in every process on the env route (the stock child of `pred` included). The switch unset: nothing here runs.
"""
import os

SWITCH = "PROTENIX_ROOT_FROZEN"          # =1: the weights root is frozen — an absent frozen input is a refusal, never a download
ROOT_ENV = "PROTENIX_ROOT_DIR"           # the stock's weights root AND download target (configs_data.py:22) — one variable, so no separate cache root exists
STOCK_DEFAULT_MODEL_NAME = "protenix_base_default_v1.0.0"                       # batch_inference.py:609-614 / configs_base.py:54
DATA_CACHES = ("common/components.cif", "common/components.cif.rdkit_mol.pkl", "common/obsolete_release_date.csv", "common/clusters-by-entity-40.txt")   # configs_data.py:251-260
TEMPLATE_CACHES = ("common/release_date_cache.json", "common/obsolete_to_successor.json")                                                            # configs_data.py:243-248, downloaded under --use_template true (inference.py:318-336)
ESM_MARKERS = ("esm", "ism")            # a model name carrying one downloads the ESM checkpoints too (inference.py:349-378): refused by name under the switch


def switch_on(environ) -> bool:
    return (environ.get(SWITCH) or "").strip() == "1"


def _option(argv, names, default):
    """The LAST value of an option among ``names`` in ``argv`` (`--opt value` or `--opt=value`; click keeps the last occurrence)."""
    val = default
    it = iter(range(len(argv)))
    for i in it:
        a = argv[i]
        for n in names:
            if a == n and i + 1 < len(argv):
                val = argv[i + 1]
            elif a.startswith(n + "="):
                val = a[len(n) + 1:]
    return val


def effective(argv) -> dict:
    """What the stock CLI would resolve from ``argv`` (the arguments after `pred`, presets already prepended): the model name and the
    template switch."""
    model = _option(argv, ("-n", "--model_name"), STOCK_DEFAULT_MODEL_NAME)
    tmpl = str(_option(argv, ("--use_template",), "false")).strip().lower() in ("1", "true", "yes", "y", "t")
    return {"model_name": model, "use_template": tmpl}


def relpaths(model_name: str, use_template: bool) -> list:
    """Every file the stock CLI would download for this invocation, relative to PROTENIX_ROOT_DIR."""
    return list(DATA_CACHES) + (list(TEMPLATE_CACHES) if use_template else []) + [f"checkpoint/{model_name}.pt"]


def check(environ, argv) -> str | None:
    """The refusal reason under the switch (None = nothing to refuse: the switch is off, or every frozen input is a file under the root).
    A model name that downloads the ESM checkpoints is refused by name: those files are not frozen inputs of this kit."""
    if not switch_on(environ):
        return None
    eff = effective(argv)
    root = environ.get(ROOT_ENV)
    if any(m in eff["model_name"] for m in ESM_MARKERS):
        return (f"{SWITCH}=1 and --model_name {eff['model_name']} downloads the ESM checkpoints beside its own (stock/src/runner/inference.py:349-378); "
                f"they are not frozen inputs of this kit — choose a checkpoint under {ROOT_ENV}/checkpoint/")
    rels = relpaths(eff["model_name"], eff["use_template"])
    missing = rels if not root else [r for r in rels if not os.path.isfile(os.path.join(root, r))]
    if not missing:
        return None
    return (f"{SWITCH}=1 and frozen inputs absent under {ROOT_ENV}={root!r}: {missing} — the stock CLI would download them "
            f"(stock/src/runner/inference.py:291-347; model_name {eff['model_name']}{', --use_template on' if eff['use_template'] else ''}); nothing runs")
