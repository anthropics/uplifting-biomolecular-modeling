"""Stock's own prediction knobs — the package adds no settings of its own.

`pred` takes `colabfold_batch`'s own command line — its two positionals and its options, verbatim, at stock's names and defaults
(colabfold/batch.py `main()`: `options_table()` reads its `add_argument` calls from the pinned wheel itself, UPSTREAM_DEFAULTS cites the
`default=` line of each option the package reads back). No option → upstream's defaults exactly: `--model-type auto` →
`alphafold2_multimer_v3` for a complex (batch.py:1670-1672), 5 models in order 1-5, `--num-recycle` unset → the multimer-v3 model config's
20 recycles with early stop at tolerance 0.5 (alphafold/model/config.py:701-702; colabfold/alphafold/models.py:148-153 overrides only when
set), one seed, `--random-seed 0` (batch.py:1870), rank `auto` → `multimer` for a complex (batch.py:1274-1275), no templates, no relax,
recompile padding 10, unified memory as the environment leaves it (batch.py:4-6). The MSA is the a3m input itself (`--msa-mode` is not
consulted for a3m input: batch.py:1437-1441). A kit mode accepts the same flags as `--mode off` does: no lever reads them.

The item-output expectation `pred`'s verdict and `warm` count against is `models_per_seed(flags)` × `num_seeds(flags)` ranked model files per
item: the passed `--num-models` / `--num-seeds`, else upstream's defaults 5 × 1 — and 0 under `--msa-only`, which `main()` turns into
`num_models = 0` (batch.py:2160-2161: MSAs and input features only, no model built, no `<job>.done.txt`). The bare options `pred` reads back
(`flag_present`): `--msa-only`, `--zip` (each job's files moved into `<job>.result.zip`, batch.py:1643-1651), `--overwrite-existing-results`
(finished jobs are recomputed instead of skipped, batch.py:1406-1412 / :2222) and `--af3-json` (the AlphaFold 3 input JSON is written and
`main()` returns before `run()`, batch.py:2184-2200). `describe(flags)` records the passed options and upstream's defaults in `pred`'s launch
record. `--data <dir>` is the parameters root: passed, it names the root for the run and its gates and rides
verbatim; absent, the CLI adds `--data $COLABFOLD_OPT_DATA_DIR` (stack.ENV_DATA).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

_TABLE: Dict[str, object] = {}


def options_table() -> Dict[str, object]:
    """colabfold_batch's argument table, read from the pinned wheel's own parser (stock/PINS.json upstream.colabfold.wheel.file:
    colabfold/batch.py `main()`, its `add_argument` calls): `{"positionals": [names], "valued": {options taking one value},
    "bare": {options taking none (store_true/false)}, "optional_value": {options whose value may follow or not (nargs='?')}}`.
    `pred` reads it only to find the two positionals among the caller's tokens (inputs.split_argv). Cached per process."""
    if _TABLE:
        return _TABLE
    import ast, os, zipfile
    from . import stack
    wheel = os.path.join(stack.tree_home(), stack.pins()["upstream"]["colabfold"]["wheel"]["file"])
    with zipfile.ZipFile(wheel) as z:
        tree = ast.parse(z.read("colabfold/batch.py").decode("utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    valued, bare, optional, pos = set(), set(), set(), []
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            kw = {k.arg: (k.value.value if isinstance(k.value, ast.Constant) else None) for k in node.keywords}
            if not names:
                continue
            if not names[0].startswith("-"):
                pos.append(names[0]); continue
            (bare if kw.get("action") in ("store_true", "store_false", "store_const", "count", "help", "version") else valued).update(names)
            if kw.get("nargs") == "?":
                optional.update(names)
    _TABLE.update(positionals=pos, valued=frozenset(valued), bare=frozenset(bare), optional_value=frozenset(optional))
    return _TABLE

# upstream's CLI defaults: option -> (default, the line of `default=` in colabfold/batch.py main())
UPSTREAM_DEFAULTS: Dict[str, tuple] = {
    "--msa-mode": ("mmseqs2_uniref_env", 1760),
    "--pair-mode": ("unpaired_paired", 1774),
    "--templates": (False, 1790),
    "--num-recycle": (None, 1842),
    "--recycle-early-stop-tolerance": (None, 1849),
    "--num-ensemble": (1, 1857),
    "--num-seeds": (1, 1864),
    "--random-seed": (0, 1870),
    "--num-models": (5, 1877),
    "--model-type": ("auto", 1887),
    "--model-order": ("1,2,3,4,5", 1898),
    "--use-dropout": (False, 1909),
    "--num-relax": (0, 1968),
    "--rank": ("auto", 2009),
    "--sort-queries-by": ("length", 2075),
    "--disable-unified-memory": (False, 2089),
    "--recompile-padding": (10, 2096),
}
MODEL_CONFIG_DEFAULTS = {"num_recycle": 20, "recycle_early_stop_tolerance": 0.5}   # alphafold/model/config.py:701-702 (multimer_v3, applied when --num-recycle is unset)


def flag_value(stock_flags: Optional[Sequence[str]], flag: str):
    """The value passed for `flag` among the stock flags (`--flag v` or `--flag=v`; the last occurrence wins, as argparse reads it),
    else upstream's default for it (UPSTREAM_DEFAULTS; None for a flag outside that table, e.g. `--data`, whose upstream default is a
    per-user directory)."""
    val = UPSTREAM_DEFAULTS[flag][0] if flag in UPSTREAM_DEFAULTS else None
    args = list(stock_flags or [])
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            val = args[i + 1]
        elif a.startswith(flag + "="):
            val = a.split("=", 1)[1]
    return val


MSA_ONLY, ZIP, OVERWRITE, AF3_JSON = "--msa-only", "--zip", "--overwrite-existing-results", "--af3-json"   # colabfold_batch's bare options `pred` reads back (batch.py:1754, :2065, :2058, :2114)


def flag_present(stock_flags: Optional[Sequence[str]], flag: str) -> bool:
    """A bare (store_true) option of colabfold_batch among the stock flags, spelled in full as its parser defines it (`--zip`, `--msa-only`, …)."""
    return any(a == flag for a in (stock_flags or []))


def models_per_seed(stock_flags: Optional[Sequence[str]] = None) -> int:
    """Ranked model files per item and seed: the passed `--num-models`, else upstream's default 5 (batch.py:1877); 0 under `--msa-only`
    (`main()` sets `num_models = 0`, batch.py:2160-2161)."""
    if flag_present(stock_flags, MSA_ONLY):
        return 0
    return int(flag_value(stock_flags, "--num-models"))


def num_seeds(stock_flags: Optional[Sequence[str]] = None) -> int:
    """Seeds per item: the passed `--num-seeds`, else upstream's default 1 (batch.py:1864)."""
    return int(flag_value(stock_flags, "--num-seeds"))


def describe(stock_flags: Optional[Sequence[str]] = None) -> dict:
    """The settings as recorded in the manifest: the stock flags passed through (verbatim), the item-output expectation (models per seed ×
    seeds) and upstream's defaults for every flag not passed."""
    return {"stock_flags": list(stock_flags or []), "models_per_seed": models_per_seed(stock_flags), "num_seeds": num_seeds(stock_flags),
            "defaults": {k: v[0] for k, v in UPSTREAM_DEFAULTS.items()}, "model_config_defaults": dict(MODEL_CONFIG_DEFAULTS)}
