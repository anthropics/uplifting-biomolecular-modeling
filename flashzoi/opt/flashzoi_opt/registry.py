"""The component registry, sourced from the kit's own tables (modes.kit_table: LEVERS, LEVER_CLASS, PINS) — never transcribed.

One entry per kit component: the name (the kit's), the kit's own class statement (LEVER_CLASS), the kit file that implements it and the
switch that turns it on. In this kit every component is ON under the apply line `kit.KitRunner(model, ...)` (its `components` argument
defaults to the kit's ALL); the package passes no component list — the kit's default IS the component composition of every kit mode — so the
switch of every component is that apply line. The one kit mode, `exact`, is the kit's defaults (numerics='tf32', carried here under the
mode: MODE_KNOBS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from . import modes

KIT_KERNEL_FILE = "fz_exact.py"                   # the kit's kernels (sha-gated by the kit at import)
KIT_WRAPPER_FILE = "_wrap.py"                     # the kit's wrapper: KitRunner (the apply line), the predict_tracks route, the refusals
APPLY_LINE = "kit.KitRunner(model, **modes.MODE_ARGS[mode])"   # the switch of every component (the kit's apply line; the mode's one argument)
FIELD = "forward"                                 # the one optimization class of this kit


@dataclass(frozen=True)
class Component:
    name: str
    field: str
    kit_class: str            # the kit's own LEVER_CLASS statement, verbatim
    source: str               # the kit file implementing it
    switch: str               # what turns it on (the apply line; every component is in the kit's default set)
    tier: str = "T1"          # the tier of the MODE is its knob (MODE_KNOBS)


MODE_KNOBS = {"exact": {"numerics": "tf32", "tier": "T1", "note": "the kit's default: the stock's TF32 class (torch's defaults, cuDNN TF32 on, plus matmul TF32 for the head GEMM) set inside each call and restored — bitwise to stock"}}


def components(kit_root: Optional[str] = None, table: Optional[dict] = None) -> Dict[str, Component]:
    """The registry from the kit's tables (read from the kit files under `kit_root`, or a table already read)."""
    t = table if table is not None else modes.kit_table(kit_root)
    out = {}
    for name in t["LEVERS"]:
        cls = t["LEVER_CLASS"].get(name)
        if cls is None:
            raise KeyError(f"kit component {name!r} has no LEVER_CLASS entry in the kit's table")
        out[name] = Component(name=name, field=FIELD, kit_class=cls, source=f"{KIT_KERNEL_FILE} (wired by {KIT_WRAPPER_FILE} KitRunner)", switch=APPLY_LINE)
    return out


def component_names(kit_root: Optional[str] = None, table: Optional[dict] = None) -> tuple:
    return tuple(components(kit_root, table))


def device_names(kit_root: Optional[str] = None, table: Optional[dict] = None) -> tuple:
    t = table if table is not None else modes.kit_table(kit_root)
    return tuple(t["PINS"].get("device_names") or ())
