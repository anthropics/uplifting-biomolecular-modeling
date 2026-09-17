"""Modes, the seven ProGen2 sizes and how a package mode resolves to the kits.

ProGen2 ships two documented commands, and the kit is different for each:

  ``sample``   generation (`sample.py`)   -> the GENERATION kit `opt/serving/pipeline_v0_4` (`progen2_decode.py`, driven in-process by
               generate.py): the model loaded once in the stock order, every exact-class lever on (the kit's own lever table + the
               decode component `components.json` names), every unit the same block bytes as the stock CLI.
  ``score``    likelihood (`likelihood.py`) -> the SCORING kit `opt/forward/engines/progen2/kits/v0_score_r3_1` (`apply(model, ew=True,
               size=...)` after `create_model`, its `Scorer` for the forwards; module B = `v0_ew` composed first).

Package modes (``KIT_MODES``, the one place to change what a mode means; ``MODES`` / ``DEFAULT_MODE`` are the one list and the one
default every command reads):
  exact   the kits above — bit-for-bit the stock's outputs at the kits' own exact shapes (their shape tables)
          — the package DEFAULT.
          One composition per mode: no lever inside it.
  off     stock: the upstream CLIs in a clean subprocess (stock_cli.py); nothing from a kit on the path.

Sizes: the stock's own ``--model`` names (``MODEL_NAMES``: sample.py / likelihood.py's `choices=models`, `progen2-small` … `progen2-xlarge`;
``variant_of_model`` also accepts the all-lowercase `progen2-bfd90` for `progen2-BFD90`); inside the package a size is its short word
(``VARIANTS``: `small` … `bfd90` … `xlarge`, the token every printed line carries as ``variant=``) and ``UPSTREAM_NAME`` maps it back to the
checkpoint name. No environment variable selects a mode or a size: the mode is ``--mode``, the size is ``--model``; the kits read no
environment variable of their own. The kit tables are read from the kit files themselves (``generation_levers``, ``decode_component``,
``ew_exact_on``). The scoring kit ships no such tuple (its optimizations
are the numbered list in its module docstring): the package names them by its own registry ids (``SCORING_OPTIMIZATIONS`` = `registry.py`'s rows for that
kit — the one place they are spelled), and the ACTIVE line prints those ids beside the elementwise kit's tuple read from its bytes.
"""
from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

VARIANTS: Tuple[str, ...] = ("small", "medium", "oas", "base", "large", "bfd90", "xlarge")
UPSTREAM_NAME: Dict[str, str] = {v: ("progen2-BFD90" if v == "bfd90" else f"progen2-{v}") for v in VARIANTS}
SIZE_CLASS: Dict[str, str] = {"small": "151M", "medium": "764M", "oas": "764M", "base": "764M", "large": "2.7B", "bfd90": "2.7B", "xlarge": "6.4B"}   # sample.py main() L103-107 (the four lists)
MODEL_NAMES: Tuple[str, ...] = tuple(UPSTREAM_NAME[v] for v in VARIANTS)                    # the stock's --model choices (sample.py / likelihood.py main(): `models`)
MODEL_ALIASES: Dict[str, str] = {"progen2-bfd90": "bfd90"}                                       # the one extra spelling accepted: the all-lowercase form of progen2-BFD90
STOCK_DEFAULT_MODEL: Dict[str, str] = {"sample": "progen2-large", "score": "progen2-base"}      # sample.py L112 / likelihood.py L122: each script's own --model default
ROUTES: Tuple[str, ...] = ("sample", "score")

MODES: Tuple[str, ...] = ("exact", "off")
DEFAULT_MODE = "exact"
NOT_SHIPPED_MODES: Tuple[str, ...] = ("fast", "big")   # the family's other mode words: no such composition ships for this model — refused by name (cli: exit 2), never mapped onto exact


def variant_of_model(name: str) -> str:
    """The size word of a stock ``--model`` name (the seven `progen2-<size>` names, or the lowercase alias of progen2-BFD90); ValueError names the choices."""
    for v, up in UPSTREAM_NAME.items():
        if name == up:
            return v
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    raise ValueError(f"--model: invalid choice {name!r} (choose from {', '.join(MODEL_NAMES)}; progen2-bfd90 is accepted for progen2-BFD90)")

SERVING_KIT_RELPATH = os.path.join("serving", "pipeline_v0_4")                                  # under opt/
FORWARD_ROOT_RELPATH = "forward"                                                                  # under opt/: the scoring kit's tree root (namespace engines.progen2.kits)
SCORING_KIT_MODULE = "engines.progen2.kits.v0_score_r3_1"
SCORING_KIT_RELPATH = os.path.join(FORWARD_ROOT_RELPATH, "engines", "progen2", "kits", "v0_score_r3_1")
EW_KIT_MODULE = "engines.progen2.kits.v0_ew"
EW_KIT_RELPATH = os.path.join(FORWARD_ROOT_RELPATH, "engines", "progen2", "kits", "v0_ew")
GENERATION_MODULE = "progen2_decode"                   # opt/serving/pipeline_v0_4/progen2_decode.py: the generation kit's module (imported with the kit dir on sys.path)
GENERATION_LEVER_TABLE = "AUTO_LEVERS"                 # progen2_decode.py: the kit's own name for the tuple of levers its load turns on beside the decode component
EW_EXACT_TABLE = "EXACT_LEVERS"                        # v0_ew/patches.py: the kit's own name for the tuple of the elementwise kit's fused-kernel names
COMPONENTS_FILE = "components.json"


@dataclass(frozen=True)
class KitMode:
    sample: Optional[str]                     # "serving" | None
    score: Optional[str]                      # "forward" | None
    what: str = ""


KIT_MODES: Dict[str, KitMode] = {
    "exact": KitMode("serving", "forward", "the kits (opt/serving/pipeline_v0_4 for sample; opt/forward/…/v0_score_r3_1 + v0_ew for score): the generation kit in-process (every exact-class lever on), "
                                           "the scoring kit + v0_ew for likelihood (B=1 per direction, the exact class on every size)"),
}


# --------------------------------------------------------------------------------------------------- the kit tables, read from the kit files
def _assigned_tuple(path: str, name: str) -> Optional[tuple]:
    """The tuple/list literal assigned to the kit's own identifier ``name`` anywhere in the file (module level or inside a function), or None."""
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                v = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(v, (tuple, list)):
                return tuple(v)
    return None


def generation_levers(serving_dir: str) -> tuple:
    """The names the generation kit's load turns on (its own `AUTO_LEVERS` tuple in `progen2_decode.py`), read from the file."""
    path = os.path.join(serving_dir, GENERATION_MODULE + ".py")
    t = _assigned_tuple(path, GENERATION_LEVER_TABLE)
    if t is None:
        raise ValueError(f"no {GENERATION_LEVER_TABLE} tuple in {path}")
    return t


def decode_component(serving_dir: str) -> dict:
    """`components.json` ``decode_kit``: the decode component the load installs (`components/<dir>/<module>.py` at `level`) and the `levers` it names."""
    with open(os.path.join(serving_dir, COMPONENTS_FILE), "r", encoding="utf-8") as fh:
        c = json.load(fh)
    dk = c.get("decode_kit") or {}
    out = {"dir": dk.get("dir"), "module": dk.get("module"), "level": dk.get("level"), "levers": [str(x) for x in (dk.get("levers") or [])], "present": False}
    if dk.get("dir"):
        cand = os.path.join(serving_dir, "components", dk["dir"])
        out["present"] = os.path.isdir(cand)
        out["path"] = cand
    return out


def ew_exact_on(ew_dir: str) -> tuple:
    """The elementwise kit's fused-kernel names (module B of the scoring kit; its exact table, `v0_ew/patches.py:32`), read from the file."""
    path = os.path.join(ew_dir, "patches.py")
    t = _assigned_tuple(path, EW_EXACT_TABLE)
    if t is None:
        raise ValueError(f"no {EW_EXACT_TABLE} tuple in {path}")
    return t

def scoring_kit_constants(scoring_dir: str) -> dict:
    """``KIT`` / ``ARM`` / ``ROUTE`` / ``MAX_ROWS_DEFAULT`` of the scoring kit, read from its file."""
    out = {}
    with open(os.path.join(scoring_dir, "__init__.py"), "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("KIT", "ARM", "ROUTE", "MAX_ROWS_DEFAULT"):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out


# ------------------------------------------------------------------------------------------------------------------- resolution
@dataclass
class Resolution:
    mode: str
    variant: Optional[str]
    upstream_name: Optional[str]
    routes: Dict[str, Optional[str]]              # route -> kit class dir ("serving" | "forward") or None (off)
    optimizations: Dict[str, list]                # route -> the names the kits' own tables carry (sample: the generation kit's lever table + the decode component; score: the scoring kit's rows + v0_ew's exact table)
    kit_dirs: Dict[str, str]                      # class -> absolute kit dir (+ "decode": the decode component's dir)
    notes: list = field(default_factory=list)


SCORING_OPTIMIZATIONS = ("rotary_tables", "one_forward_per_direction", "host_pipeline")   # the scoring kit's levers by name (its exact-length batching runs at B=1 in this package; resident_fp16 is the kit's option, off)


def kit_mode(mode: str) -> KitMode:
    """The ONE resolution point: mode -> the composition."""
    if mode not in KIT_MODES:
        raise ValueError(f"unknown kit mode {mode!r}; expected one of {sorted(KIT_MODES)} (or 'off')")
    return KIT_MODES[mode]


def check_variant(variant: Optional[str]) -> Optional[str]:
    if variant is None:
        return None
    v = variant.strip().lower()
    if v not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    return v


def resolve(mode: str, variant: Optional[str], home: str) -> Resolution:
    """Resolve a package mode to the kit dirs and the names their own tables carry (read from the kit files under `home` = progen2/opt)."""
    variant = check_variant(variant)
    up = UPSTREAM_NAME.get(variant) if variant else None
    if mode == "off":
        return Resolution(mode, variant, up, {"sample": None, "score": None}, {"sample": [], "score": []}, {}, ["stock: nothing from a kit on the path"])
    km = kit_mode(mode)
    serving = os.path.join(home, SERVING_KIT_RELPATH)
    scoring = os.path.join(home, SCORING_KIT_RELPATH)
    ew = os.path.join(home, EW_KIT_RELPATH)
    notes = []
    on = {"sample": [], "score": []}
    kit_dirs = {}
    if km.sample == "serving":
        kit_dirs["serving"] = serving
        comp = decode_component(serving)
        kit_dirs["decode"] = comp.get("path") or os.path.join(serving, "components", str(comp.get("dir")))   # the component is a lever of the mode: its dir absent = the kit tree is not whole (the activation's install error)
        on["sample"] = list(generation_levers(serving)) + [str(x) for x in (comp.get("levers") or [])]
    if km.score == "forward":
        kit_dirs["forward"] = os.path.join(home, FORWARD_ROOT_RELPATH)
        kit_dirs["scoring"] = scoring
        kit_dirs["ew"] = ew
        on["score"] = list(SCORING_OPTIMIZATIONS) + [f"ew:{n}" for n in ew_exact_on(ew)]
    return Resolution(mode, variant, up, {"sample": km.sample, "score": km.score}, on, kit_dirs, notes)


def describe_line(res: Resolution, route: Optional[str] = None) -> str:
    """The mode's kit spelling for the activation line: ``sample=serving:pipeline_v0_4 score=forward:v0_score_r3_1+v0_ew``."""
    parts = []
    for r in ("sample", "score"):
        if route and r != route:
            continue
        cls = res.routes.get(r)
        if cls == "serving":
            parts.append("sample=serving:" + os.path.basename(SERVING_KIT_RELPATH))
        elif cls == "forward":
            parts.append("score=forward:" + os.path.basename(SCORING_KIT_RELPATH) + "+" + os.path.basename(EW_KIT_RELPATH))
        else:
            parts.append(f"{r}=stock")
    return " ".join(parts)
