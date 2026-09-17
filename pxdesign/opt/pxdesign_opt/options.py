"""The `pxdesign infer` options every mode passes through: upstream's own knobs, with upstream's names and defaults.

The knob table is `stock/PINS.json` "cli_defaults" (upstream's documented defaults: `runner/cli.py:88-131` shared options,
`configs/configs_infer.py:21-31`, `configs/configs_base.py:60-62`): `--N_step 400 --N_sample 5 --dtype bf16 --eta_type const --eta_min 2.5
--eta_max 2.5 --num_workers 16 --use_msa true --use_fast_ln true`, rendered explicitly on every route in that order (`OPTION_ORDER`) with the
caller's value where one is given; anything else upstream accepts passes through unchanged (cli.py). `--seeds` is upstream's too: absent = upstream
derives one seed from `time.time_ns()` (`runner/inference.py:233`) — on every mode (the kit modes' in-process route derives it with upstream's
own `derive_seed`, `infer_loop.run`). Boolean knobs take upstream's own words (`get_bool_value`, `protenix/config/extend_types.py:48-55`:
true|t|yes|y|1, false|f|no|n|0; `bool_word` renders them as true|false).
`LAYERNORM_TYPE` is the variable `--use_fast_ln` means to set: Protenix reads it from the environment at import
(`protenix/openfold_local/model/primitives.py:49-51`), before upstream's own `configure_runtime_env` sets it (`pxdesign/utils/infer.py:492-503`),
so the design verb sets it from the flag before anything imports Protenix, identically on every mode, `off` included (`apply_layernorm_env`:
`true`, the default -> `fast_layernorm`, Protenix's fused LayerNorm; `false` -> unset, the plain non-fused LayerNorm); configs/<gpu>.env exports the
default's word for the environment route (`PXDESIGN_OPT=<mode> pxdesign infer …`), where the caller's environment decides as with bare upstream.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .stamps import FALSE_WORDS, TRUE_WORDS                 # upstream's boolean words (get_bool_value), one table

OPTION_ORDER = ("N_step", "N_sample", "dtype", "eta_type", "eta_min", "eta_max", "num_workers", "use_msa", "use_fast_ln")   # rendered in this order
INT_OPTIONS = ("N_step", "N_sample", "num_workers")
BOOL_OPTIONS = ("use_msa", "use_fast_ln")
LAYERNORM_ENV = "LAYERNORM_TYPE"
LAYERNORM_VALUE = "fast_layernorm"                          # Protenix's fused LayerNorm word = configs/<gpu>.env's export; any other value or none selects OpenFold's LayerNorm


def _pins_path() -> str:
    from .stack import tree_home
    return os.path.join(tree_home(), "stock", "PINS.json")


def read_pins(path: Optional[str] = None) -> dict:
    with open(path or _pins_path(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _word(v) -> str:
    """A command-line word for a value: Python booleans (the pins' defaults) as upstream spells them (true|false), numbers and strings as given —
    boolean KNOB words are canonicalised by `bool_word` at parse time, never here (so `--eta_type`, `--dtype` words pass verbatim)."""
    if v is True:
        return "true"
    if v is False:
        return "false"
    return str(v)


def bool_word(v: str) -> str:
    """argparse `type` for a boolean knob: any word upstream's `get_bool_value` accepts, rendered true|false; anything else is the error upstream
    itself raises on it (ArgumentTypeError here, exit 2)."""
    import argparse
    w = str(v).strip().lower()
    if w in TRUE_WORDS:
        return "true"
    if w in FALSE_WORDS:
        return "false"
    raise argparse.ArgumentTypeError(f"invalid boolean word {v!r} (upstream accepts {'|'.join(TRUE_WORDS)} / {'|'.join(FALSE_WORDS)})")


def defaults(pins: Optional[dict] = None) -> Dict[str, str]:
    """Upstream's defaults for the knobs of OPTION_ORDER, as command-line words (stock/PINS.json cli_defaults)."""
    cd = (pins or read_pins())["cli_defaults"]
    return {k: _word(cd[k]) for k in OPTION_ORDER}


@dataclass(frozen=True)
class Options:
    values: Dict[str, str]                                # every knob of OPTION_ORDER: the caller's word where given, else upstream's default
    given: Dict[str, str]                                 # what the caller gave (knobs, seeds), as words
    seeds: Tuple[int, ...]                                # () = not given: upstream derives one seed from the clock
    defaults: Dict[str, str] = field(default_factory=dict)

    def argv(self) -> List[str]:
        out: List[str] = []
        for k in OPTION_ORDER:
            out += [f"--{k}", self.values[k]]
        return out

    def seeds_arg(self) -> Optional[str]:
        return ",".join(str(s) for s in self.seeds) if self.seeds else None

    def as_dict(self) -> dict:
        return {"values": dict(self.values), "given": dict(self.given), "defaults": dict(self.defaults), "seeds": list(self.seeds), "argv": self.argv()}


def resolve(*, N_step=None, N_sample=None, dtype=None, eta_type=None, eta_min=None, eta_max=None, num_workers=None, use_msa=None,
            use_fast_ln=None, seeds=None, pins: Optional[dict] = None) -> Options:
    """The knob values for one job: upstream's defaults with the caller's words applied."""
    base = defaults(pins)
    given_raw = {"N_step": N_step, "N_sample": N_sample, "dtype": dtype, "eta_type": eta_type, "eta_min": eta_min, "eta_max": eta_max,
                 "num_workers": num_workers, "use_msa": use_msa, "use_fast_ln": use_fast_ln}
    given = {k: (bool_word(v) if k in BOOL_OPTIONS and not isinstance(v, bool) else _word(v)) for k, v in given_raw.items() if v is not None}   # boolean knobs: upstream's words canonicalised; other knobs verbatim
    values = dict(base, **given)
    sd: Tuple[int, ...] = ()
    if seeds not in (None, "", ()):
        sd = tuple(int(s) for s in (seeds.split(",") if isinstance(seeds, str) else seeds))
        given["seeds"] = ",".join(str(s) for s in sd)
    return Options(values=values, given=given, seeds=sd, defaults=base)


def apply_layernorm_env(opts: "Options", env) -> None:
    """Set `LAYERNORM_TYPE` in `env` (os.environ, or the stock child's mapping) from the job's `--use_fast_ln` word: `true` -> `fast_layernorm`,
    `false` -> removed (absent selects the plain, non-fused LayerNorm). The default flag gives the word configs/<gpu>.env exports."""
    if opts.values["use_fast_ln"] == "true":
        env[LAYERNORM_ENV] = LAYERNORM_VALUE
    else:
        env.pop(LAYERNORM_ENV, None)


def add_arguments(ap) -> None:
    """The knob flags on a parser: upstream's names; default None = not given (upstream's default applies and is rendered)."""
    for k in OPTION_ORDER:
        if k in INT_OPTIONS:
            ap.add_argument(f"--{k}", type=int, default=None)
        elif k in BOOL_OPTIONS:
            ap.add_argument(f"--{k}", type=bool_word, default=None, metavar="|".join(("true", "false")))   # upstream's boolean words, rendered true|false
        else:
            ap.add_argument(f"--{k}", default=None)             # strings and floats verbatim (upstream parses them)
    ap.add_argument("--seeds", default=None, help="comma-separated; absent = upstream derives one seed from the clock (every mode)")


def from_args(args, pins: Optional[dict] = None) -> Options:
    return resolve(**{k: getattr(args, k, None) for k in OPTION_ORDER}, seeds=getattr(args, "seeds", None), pins=pins)


def given_argv(opts: Options) -> List[str]:
    """The caller's given words as flags (for handing the same job to the stock caller): --seeds when given, then the knobs in OPTION_ORDER."""
    out: List[str] = ["--seeds", opts.seeds_arg()] if opts.seeds else []
    for k in OPTION_ORDER:
        if k in opts.given:
            out += [f"--{k}", opts.given[k]]
    return out
