"""Stock's own knobs pass through; nothing is preset. The package writes no model-shape flag in any mode: ``--num_recycles``,
``--num_diffusion_samples``, ``--flash_attention_implementation`` (run_alphafold.py's own names and defaults: STOCK_FLAG_NAMES, read from the
stock source by upstream_defaults), ``--buckets`` and every other flag of the stock command line go on the model process's command line exactly
as the caller states them, after the composed flags (absl: the last occurrence of a flag wins), or stay at the fork's own defaults when
unstated. What the package composes itself (stock_pred.compose): the COMMON row's input flags (modes.common_flags), the variant's weights
flags, the mode's padding flag (modes.pad_flags) and the cache class — on ``off`` the EMPTY class ``--cache_dir=`` (tokamax autotune skipped,
the JAX cache at ./jax under the pass's output directory) unless the caller names a ``--cache_dir`` of their own, which is then the only one
on the line; under the kit modes ``--cache_dir=$CACHE`` (the levers' class: autotune results, featurisation memo, serialized executables) —
the mode's class directory, or the caller's own ``--cache_dir`` when they name one (the levers then read and write that directory). No stock
flag is refused in any mode."""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

FLASH_FLAG_NAME = "flash_attention_implementation"                     # upstream's accelerator switch (run_alphafold.py:389-402: triton | cudnn | xla, default triton)
STOCK_FLAG_NAMES = ("num_recycles", "num_diffusion_samples", FLASH_FLAG_NAME)   # the model shape, in the stock CLI's own names and defaults (run_alphafold.py:402-407 num_recycles 10, :408-413 num_diffusion_samples 5, :389-401 flash_attention_implementation triton)
CACHE_FLAG_NAME = "cache_dir"                                        # the fork's cache flag; its default is where the pass-through caches
EMPTY_CACHE_FLAG = f"--{CACHE_FLAG_NAME}="                             # the EMPTY cache class, mode off's composition when the caller names no --cache_dir (autotune skipped; JAX cache ./jax under the process cwd)
CWD_JAX_CACHE = "<output_dir>/jax"                                   # where that JAX cache lands: the model process runs with cwd = the pass's output directory (cli.pred)
FIXED_SEEDS = (1,)                                             # a fixed seed; the kit's own inputs carry seeds 1-5 (its tested shape)
STOCK_SOURCE_RELPATH = os.path.join("stock", "src", "run_alphafold.py")
# flags.DEFINE_<kind>('name', <default> | default=<default>, ...): the default is the first value after the name (absl's positional form)
_DEFINE_RX = re.compile(r"flags\.DEFINE_\w+\(\s*'(?P<name>\w+)'\s*,\s*(?:default\s*=\s*)?(?P<default>'[^']*'|[-\w.]+)", re.S)


def upstream_defaults(tree: Optional[str] = None, names: Tuple[str, ...] = STOCK_FLAG_NAMES) -> Dict[str, str]:
    """{flag name: default as written} for ``names`` (STOCK_FLAG_NAMES), read from the stock source's ``flags.DEFINE_*`` calls."""
    from .stack import tree_home
    path = os.path.join(tree or tree_home(), STOCK_SOURCE_RELPATH)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    found = {m.group("name"): m.group("default").strip("'") for m in _DEFINE_RX.finditer(src) if m.group("name") in names}
    missing = [n for n in names if n not in found]
    if missing:
        raise RuntimeError(f"{path}: flags {missing} not found (the stock source changed shape)")
    return {n: found[n] for n in names}


BUCKETS_FLAG_NAME = "buckets"                                        # the fork's token buckets (flags.DEFINE_list): an input pads to the first bucket that holds it
_DEFINE_LIST_RX = re.compile(r"flags\.DEFINE_list\(\s*'(?P<name>\w+)'\s*,\s*(?:#[^\n]*\n\s*)*\[(?P<items>[^\]]*)\]", re.S)


def upstream_buckets(tree: Optional[str] = None) -> List[int]:
    """The fork's default ``--buckets`` (run_alphafold.py ``flags.DEFINE_list('buckets', [...])``), as integers in the written order."""
    from .stack import tree_home
    path = os.path.join(tree or tree_home(), STOCK_SOURCE_RELPATH)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    for m in _DEFINE_LIST_RX.finditer(src):
        if m.group("name") == BUCKETS_FLAG_NAME:
            items = [t.strip().strip("'\"") for t in m.group("items").replace("\n", " ").split(",") if t.strip()]
            out = [int(t) for t in items]
            if not out or out != sorted(out):
                raise RuntimeError(f"{path}: --{BUCKETS_FLAG_NAME} default {out} is empty or not increasing (the stock source changed shape)")
            return out
    raise RuntimeError(f"{path}: flags.DEFINE_list('{BUCKETS_FLAG_NAME}', [...]) not found (the stock source changed shape)")


def stated(tokens: List[str], names: Tuple[str, ...] = STOCK_FLAG_NAMES) -> List[str]:
    """The caller's own tokens for ``names`` (``--<name>=<v>`` / ``--<name>``), in the order given — the model shape a process ran at when it
    differs from the fork's defaults (they key the compiled programs of the class)."""
    return [t for t in tokens if t.split("=", 1)[0] in {"--" + n for n in names}]


def flag_value(tokens: List[str], name: str) -> Optional[str]:
    for t in tokens:
        if t == name:
            return "true"
        if t.startswith(name + "="):
            return t.split("=", 1)[1]
    return None


def flag_false(tokens: List[str], name: str) -> bool:
    """True when a BOOLEAN stock flag is switched OFF on the command line in any spelling absl accepts (``--noNAME``, ``--NAME=false|0|no|f|n``);
    the last statement wins. ``name`` without dashes (e.g. ``run_inference``)."""
    off = False
    for t in tokens:
        if t == f"--no{name}":
            off = True
        elif t == f"--{name}" or t.startswith(f"--{name}="):
            v = t.split("=", 1)[1].strip().lower() if "=" in t else "true"
            off = v in ("false", "0", "no", "f", "n")
    return off


def flag_int(tokens: List[str], name: str) -> Optional[int]:
    """The integer a VALUED stock flag states on the command line, in either spelling absl accepts (``--name=3`` or ``--name 3``);
    None when the flag is absent. The last statement wins, as absl parses. A value that is not an integer is left to the stock
    script to reject (None here: the wrapper never second-guesses upstream's own flag validation)."""
    val: Optional[str] = None
    for i, t in enumerate(tokens):
        if t.startswith(name + "="):
            val = t.split("=", 1)[1]
        elif t == name and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            val = tokens[i + 1]
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None
