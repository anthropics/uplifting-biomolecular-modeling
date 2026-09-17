"""Upstream's own `opendde pred` knobs on the package's command line — stock's flag names, passed through verbatim — and the tree's stock base.

`pred` (every mode alike) accepts the upstream knobs below with upstream's spellings (runner/batch_inference.py, the `pred` = `predict` click
options; upstream's defaults are pinned in stock/PINS.json "cli_defaults"). A flag the caller states is handed to the engine exactly as
stated; a flag the caller leaves out is not passed, so upstream's own default applies — with ONE exception, the stock base of this tree
(README "Stock"): upstream's `--dtype bf16` autocast switch and its documented fused LayerNorm, `LAYERNORM_TYPE=fast_layernorm`. `stock` means
that pair here, on every route and mode (`off` included): `--dtype bf16` is passed unless the caller states `--dtype`, and `LAYERNORM_TYPE` is
exported unless the caller already set it; a caller's own value runs as requested with one NOTE line. Every other upstream option passes
untouched after `--`. There is no settings table in this tree.

  FLAGS            the knobs the package itself must read (in the order they are handed to the engine): the structures a run owes
                   (`--seeds`, `--sample`: outputs.expected_structures), the templates gate (`--use_template`: templates.py / frozen.py), the
                   frozen-weights gate (`--use_msa`, `--use_rna_msa`), the PRED line's `dims=` token and the manifest's record.
  add_arguments(p) declares them on a parser: long name + upstream's short alias, default None (= not stated).
  stated(a)        {flag: value} the caller stated;  stated_args(a) the same as engine arguments, in FLAGS order.
  effective(...)   {flag: value} in force: stated, else a trailing occurrence after `--` (the last one wins, as click reads them), else
                   upstream's default (PINS "cli_defaults") — strings, booleans spelled `true` / `false`.
  fields(...)      the PRED line's token: `flags=<the stated flags, comma-joined | upstream_defaults> dims=cycle:<c>,step:<p>,sample:<e>,use_msa:<b>`.
  describe(...)    the manifest's "stock_flags" block.
  base_args(...)   `['--dtype', 'bf16']` unless the caller stated `--dtype` (on `pred` or after `--`): the stock base's flag, every mode.
  run_seed(...)    the ONE seed decision of a `--n_gpu P>1` run (`RunSeed(value, source, args)`, source `cli` | `json` | `drawn`): the launcher's
                   parent makes it once so every rank process runs the same seeds; `run_seed_fields` renders its LAUNCH-line words.
  stock_env / apply_stock_env / stock_env_notes   the stock base's environment (`STOCK_ENV`): exported when unset; a caller's other value stays and
                   is named in one NOTE sentence.
  dtype_note(...)  the one NOTE sentence printed when the `--dtype` in force is not the base's; None on the base. Informational — never a refusal.
"""
from __future__ import annotations

import json
import os
import random
from collections import namedtuple

# flag -> (upstream short alias | None, runner/batch_inference.py line of the click option) — cited in --help
UPSTREAM = {
    "cycle": ("-c", 705), "step": ("-p", 706), "sample": ("-e", 707), "dtype": (None, 710), "model_name": ("-n", 725),
    "use_msa": (None, 737), "use_template": (None, 785), "use_rna_msa": (None, 791), "need_atom_confidence": (None, 804),
    "trimul_kernel": (None, 749), "triatt_kernel": (None, 755),                               # upstream's triangle-kernel selectors (auto | cuequivariance | torch): run as stated;
    "seeds": ("-s", 699),                                                                        # a stated value other than `auto` is a stock knob (KERNEL_KNOBS, stock_knobs)
}
FLAGS = tuple(UPSTREAM)                              # the order the stated flags are handed to the engine (`--seeds` last, then the checkpoint path)
DIMS = ("cycle", "step", "sample", "use_msa")        # the PRED line's dims= token
KERNEL_KNOBS = {"triatt_kernel": "cueq_triatt", "trimul_kernel": "cueq_trimul"}   # upstream's triangle-kernel flags -> the KERNELS census site each selects (lncensus.KERNEL_SITES)
KERNEL_AUTO = "auto"                                 # upstream's default for both (stock/PINS.json cli_defaults): the engine resolves it on the device; any other STATED value
                                                     # is the caller's choice of upstream's own kernel — the census expects what the flag says and the kit's triangle levers of
                                                     # that site step aside by name (opendde_opt/stockknob.py)
BASE_DTYPE = "bf16"                                  # the stock base's `--dtype` (README "Stock"): upstream's bf16 autocast switch, passed unless the caller states --dtype
LN_ENV = "LAYERNORM_TYPE"
STOCK_ENV = {LN_ENV: "fast_layernorm"}              # the stock base's environment, every route and mode: upstream's fused LayerNorm CUDA kernel, read once at import
                                                     # (opendde/model/triangular/layers.py) — stock/PINS.json "reads"; exported unless the caller set the name
STOCK_BASE = "bf16_fastln"                           # the one name of that pair


def cli_defaults(tree: str) -> dict:
    """Upstream's `pred` defaults as pinned in stock/PINS.json "cli_defaults" (runner/batch_inference.py:692-913)."""
    with open(os.path.join(tree, "stock", "PINS.json")) as fh:
        return json.load(fh).get("cli_defaults", {})


def _spell(v) -> str:
    """A default's command-line spelling: booleans as click reads them (`true` / `false`), None as ''."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return "" if v is None else str(v)


def add_arguments(p, tree: str | None = None) -> None:
    """Declare upstream's knobs on ``p``: `--<flag>` (+ upstream's short alias), default None = not stated, upstream's default named in the help."""
    defaults = cli_defaults(tree) if tree else {}
    for flag, (short, line) in UPSTREAM.items():
        names = [f"--{flag}"] + ([short] if short else [])
        d = defaults.get(flag)
        dtext = "the input JSON's modelSeeds, else a random seed" if flag == "seeds" and d is None else _spell(d)
        if flag == "dtype":
            dtext = f"{BASE_DTYPE} — the stock base of this tree, passed by the package (upstream's own default is {_spell(d) or 'fp32'})"
        p.add_argument(*names, dest=flag, default=None, metavar=flag.upper(),
                       help=f"upstream `opendde pred {names[0]}` (runner/batch_inference.py:{line}); passed through as stated — default when absent: {dtext}")


def stated(a, extra: list[str] | None = None) -> dict:
    """{flag: value} for the upstream flags the caller stated, in FLAGS order: on `pred` itself (argparse namespace ``a``) or, given ``extra``,
    after `--` (the last occurrence, as click reads them). Both reach the engine verbatim."""
    out = {}
    for f in FLAGS:
        v = getattr(a, f, None)
        if v is None and extra:
            v = _last(list(extra), f)
        if v is not None:
            out[f] = str(v)
    return out


def stock_knobs(st: dict) -> dict:
    """{flag: value} for the triangle-kernel flags (KERNEL_KNOBS) the caller STATED with a value other than upstream's `auto` — on `pred` or after
    `--` (``st`` = stated(a, extra)). Empty when neither is stated (the shipped lines, unchanged)."""
    return {f: st[f] for f in KERNEL_KNOBS if st.get(f) not in (None, "", KERNEL_AUTO)}


def stated_in(argv: list[str]) -> dict:
    """{flag: value} for the KERNEL_KNOBS flags in a raw engine argument list (`--flag v` or `--flag=v`, the last occurrence wins) — the stock
    caller's reading of the arguments it hands upstream unchanged (stock_pred), the same rule as ``stated``."""
    out = {}
    toks = list(argv or [])
    for i, tok in enumerate(toks):
        for f in KERNEL_KNOBS:
            if tok == f"--{f}" and i + 1 < len(toks):
                out[f] = str(toks[i + 1])
            elif tok.startswith(f"--{f}="):
                out[f] = tok.split("=", 1)[1]
    return out


def stated_args(a) -> list[str]:
    """The flags stated on `pred` itself as engine arguments, verbatim, in FLAGS order: ['--cycle', '10', …] (those after `--` travel in the
    pass-through list, untouched)."""
    out: list[str] = []
    for f, v in stated(a).items():
        out += [f"--{f}", v]
    return out


def _last(extra: list[str], flag: str):
    """The value of the last `--<flag> <v>` / `--<flag>=<v>` (or upstream's short alias `<short> <v>`) among the pass-through arguments, else None."""
    short = UPSTREAM[flag][0]
    val = None
    for i, tok in enumerate(extra):
        if (tok == f"--{flag}" or (short and tok == short)) and i + 1 < len(extra):
            val = extra[i + 1]
        elif tok.startswith(f"--{flag}="):
            val = tok.split("=", 1)[1]
    return val


def effective(a, extra: list[str] | None = None, tree: str | None = None) -> dict:
    """{flag: value} in force for this run: stated, else the last occurrence after `--`, else the stock base's `--dtype` (``BASE_DTYPE``, which
    ``base_args`` passes), else upstream's default (strings; '' = upstream decides at run time, e.g. seeds from the input JSON)."""
    defaults = dict(cli_defaults(tree) if tree else {}, dtype=BASE_DTYPE)
    st = stated(a, extra)
    return {f: (st[f] if f in st else _spell(defaults.get(f))) for f in FLAGS}


def base_args(a, extra: list[str] | None = None) -> list[str]:
    """The stock base's flag: `['--dtype', BASE_DTYPE]` unless the caller stated `--dtype` on `pred` or after `--` (the caller's value passes as stated)."""
    return [] if "dtype" in stated(a, extra) else ["--dtype", BASE_DTYPE]


RUNSEED_SPACE = (1, 65536)                           # upstream's own random-seed space: runner/inference.py `_resolve_job_seed_schedule` draws `random.randint(1, 65536)` for a job with no seeds
RUNSEED_SOURCES = ("cli", "json", "drawn")           # where a `--n_gpu P>1` run's seeds come from: `--seeds` stated | every job's `modelSeeds` | one seed drawn by the launcher's parent
RunSeed = namedtuple("RunSeed", "value source args")  # value, by source: cli -> the stated `--seeds` text (str, as typed: "7", "5,6") | json -> None | drawn -> the drawn int; args: the engine arguments stated to every rank (`['--seeds', v]` when drawn, else [])


def run_seed(st: dict, jobs: list, n_gpu: int, draw=None) -> RunSeed:
    """The ONE seed decision of a ``--n_gpu P>1`` run, made once by the launcher's parent so every rank process runs the same seeds (upstream's
    precedence per job: ``--seeds`` > that job's ``modelSeeds`` > a random seed — drawn per PROCESS, so P rank processes would each draw their own):
    ``st`` = :func:`stated` (on `pred` or after `--`), ``jobs`` = the query's jobs.

      * ``--seeds`` stated                  -> ``RunSeed(<stated>, "cli", [])``: every rank states it itself, nothing is added;
      * every job carries ``modelSeeds``    -> ``RunSeed(None, "json", [])``: upstream reads them per job on every rank;
      * no job carries ``modelSeeds``       -> ``RunSeed(v, "drawn", ["--seeds", str(v)])``: ``v = draw()`` ONCE (default: ``random.randint`` over
                                               :data:`RUNSEED_SPACE`, upstream's own space), handed to every rank exactly as a caller's ``--seeds v``;
      * some jobs carry ``modelSeeds``, some do not -> ValueError ``runseed: <k> of <n> jobs carry no modelSeeds under --n_gpu <P> — state --seeds,
                                               or give every job modelSeeds`` (the caller refuses the run by name before any rank starts).

    A job's ``modelSeeds`` counts when it is a non-empty list (upstream's own test). At P == 1 nothing here applies: upstream decides in its one process."""
    if "seeds" in st:
        return RunSeed(st["seeds"], "cli", [])
    n = len(jobs or [])
    missing = sum(1 for j in (jobs or []) if not j.get("modelSeeds"))
    if n and missing == 0:
        return RunSeed(None, "json", [])
    if missing < n:
        raise ValueError(f"runseed: {missing} of {n} jobs carry no modelSeeds under --n_gpu {int(n_gpu)} — state --seeds, or give every job modelSeeds")
    v = int(draw() if draw is not None else random.randint(*RUNSEED_SPACE))
    return RunSeed(v, "drawn", ["--seeds", str(v)])


def run_seed_fields(seed: RunSeed) -> str:
    """The LAUNCH line's words for the run's seed decision: ``runseed=<v|-> source=<cli|json|drawn>`` (``-`` = the jobs' own modelSeeds)."""
    return f"runseed={'-' if seed.value is None else seed.value} source={seed.source}"


def fields(eff: dict, st: dict) -> str:
    """The PRED line's settings token: which upstream flags were stated (or `upstream_defaults`) and the run's dims."""
    names = ",".join(st) if st else "upstream_defaults"
    return f"flags={names} dims=" + ",".join(f"{k}:{eff.get(k, '')}" for k in DIMS)


def describe(eff: dict, st: dict, tree: str | None = None, environ=None) -> dict:
    """The manifest's "stock_flags" block: the stated flags, the values in force, where upstream's defaults come from, the stock base and its environment."""
    env = os.environ if environ is None else environ
    src = (cli_defaults(tree).get("source") if tree else None) or "stock/PINS.json cli_defaults"
    return {"stated": dict(st), "effective": dict(eff), "defaults_source": src, "base": STOCK_BASE,
            "stock_env": {k: env.get(k, v) for k, v in STOCK_ENV.items()}}


def ln_requested(environ=None) -> bool:
    """True when the model process is asked for upstream's fused LayerNorm (`LAYERNORM_TYPE=fast_layernorm`: the stock base, or the caller's own export)."""
    env = os.environ if environ is None else environ
    return env.get(LN_ENV) == STOCK_ENV[LN_ENV]


def stock_env(environ=None) -> dict:
    """The ``STOCK_ENV`` names this process still has to export — {} when every name is already set, as the base or as the caller chose."""
    env = os.environ if environ is None else environ
    return {k: v for k, v in STOCK_ENV.items() if env.get(k) in (None, "")}


def stock_env_notes(environ=None) -> list[str]:
    """One NOTE sentence per ``STOCK_ENV`` name the caller set to another value: the run proceeds as requested; the sentence names the base. Never a refusal."""
    env = os.environ if environ is None else environ
    return [f"NOTE {k}={env.get(k)!r} requested — running as requested; the stock base of this tree (every exact/fast/big word) is {k}={v}"
            for k, v in STOCK_ENV.items() if env.get(k) not in (None, "", v)]


def apply_stock_env(environ=None) -> dict:
    """Export ``stock_env()`` into ``environ`` (default ``os.environ``); returns what was exported."""
    env = os.environ if environ is None else environ
    out = stock_env(env)
    env.update(out)
    return out


def dtype_note(dtype: str | None, mode: str | None = None) -> str | None:
    """The NOTE sentence for a `--dtype` in force other than the base's (``BASE_DTYPE``), else None: the run proceeds as requested; the sentence names the base."""
    d = (dtype or BASE_DTYPE).lower()
    if d == BASE_DTYPE:
        return None
    m = (mode or "off").lower()
    where = "on the stock arm" if m == "off" else f"under {m}"
    return (f"NOTE --dtype {d} requested {where} — running upstream's {d} switch as requested; the stock base of this tree (every exact/fast/big word) is "
            f"--dtype {BASE_DTYPE} + {LN_ENV}={STOCK_ENV[LN_ENV]}")


def upstream_layernorm_word() -> str | None:
    """None unless upstream's triangular layers are already imported in this process with an implementation other than the base's — then the word
    naming it (the env route: the stock CLI imported its model before the kit activated)."""
    import sys
    m = sys.modules.get("opendde.model.triangular.layers")
    if m is None or getattr(m, "_use_fast_layer_norm", None) is None:
        return None
    return None if m._use_fast_layer_norm else "torch"
