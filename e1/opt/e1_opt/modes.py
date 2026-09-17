"""THE mode table — the one place that says what a package mode is (every command, the autoload hook and run.sh read it).

  off     stock: the upstream CLI (`python -m E1.tools.score ...`) in a clean subprocess — every kit variable stripped from the
          environment and proved absent by the subprocess itself, no kit directory on sys.path, no component applied (stock_score.py).
  exact   the kit (opt/forward/engines/e1/kits/eager, mode word `eager`): ONE lever set applied to every forward of the model at its
          construction — on the command line the same upstream CLI in a subprocess with E1_OPT=exact (kit_score.py), in a program
          `e1_opt.enable("exact")` / the E1_OPT autoload arming the E1Predictor / E1Scorer constructors (stack.py). Outputs are the
          stock's, bit for bit (against the pinned stock under the deterministic recipe, det.py); the mode
          is all of its levers — one that cannot run on the host is the mode's refusal by name, never a subset.

Every mode that runs names the accelerators the pinned stock ENGAGES in its model process (`ModeEntry.kernels`, names from
accel.ACCELERATORS): the KERNELS proof (accel.py) reads what the process bound and words it against the expected words computed from this
field (which accelerators) and the pins module (at which version) — an accelerator that is absent or fell back on the host is NAMED on that
line; under `exact`, whose levers run on those accelerators, it is the mode's refusal by name. Upstream E1 engages flash-attn and the hub
Triton RMSNorm whenever they import and always compiles flex-attention; the pinned stock is that environment, so both modes expect all three.

The kit's own constants this table depends on (KIT, MODE) are read out of the kit's entry file by AST — never typed here — so a kit tree
whose constants disagree fails `lock_check` (a refusal by name at activation) rather than running under a stale name. `fast` and `big` are not modes of this kit (no
tolerance-class levers, no separate memory lever set): asked for, they are refused by name.
"""
from __future__ import annotations

import ast
import os
from typing import Dict, NamedTuple, Optional, Tuple

MODES = ("off", "exact")                             # the names the switch accepts, the stock word first
DEFAULT_MODE = "exact"
KIT_MODE = "eager"                                   # the kit's own mode word (locked against the kit's MODE constant below)
KERNELS_ALL = ("flash_attn", "hub_layernorm", "flex_attention")   # == accel.ACCELERATORS (accel.py asserts the equality; typed here so this table stays import-free)
NOT_SHIPPED = ("fast", "big")                       # the family's other mode words: refused by name here


class ModeEntry(NamedTuple):
    mode: str
    runner: str                                      # the package module that runs the command in this mode
    kit_mode: Optional[str]                          # the kit's mode word (None = no kit)
    active: bool                                     # does this mode apply a component
    what: str                                        # what the mode does (never how fast)
    kernels: Tuple[str, ...]                         # the accelerators the mode's model process must engage (accel.py)


MODE_TABLE: Dict[str, ModeEntry] = {
    "off": ModeEntry("off", "e1_opt.stock_score", None, False,
                     "stock e1 (the upstream CLI in a clean subprocess; no environment set, no component applied)", KERNELS_ALL),
    "exact": ModeEntry("exact", "e1_opt.kit_score", KIT_MODE, True,
                       "the kit: its one lever set on every forward, applied at the model's construction", KERNELS_ALL),
}
KIT_MODES = tuple(m for m, e in MODE_TABLE.items() if e.active)   # the modes that run the kit (exact): the kit route's word on the KERNELS line
assert tuple(MODE_TABLE) == MODES == ("off", "exact") and KIT_MODES == ("exact",)


def kernels_expected(mode: Optional[str]) -> Tuple[str, ...]:
    """The accelerators the mode's model process must engage."""
    return tuple(entry(mode).kernels)


def check_mode(mode: Optional[str]) -> str:
    """The mode name, normalised; ValueError for a name outside the table (the family's other words named as not shipped)."""
    m = (mode or DEFAULT_MODE).strip().lower()
    if m in NOT_SHIPPED:
        raise ValueError(f"mode {m}: not shipped by this kit (modes: {'|'.join(MODES)})")
    if m not in MODE_TABLE:
        raise ValueError(f"unknown mode {mode!r} (expected {'|'.join(MODES)})")
    return m


def entry(mode: Optional[str]) -> ModeEntry:
    return MODE_TABLE[check_mode(mode)]


# ------------------------------------------------------------------------------------------ the kit's own constants, read from its file
def kit_constants(kit_dir: str) -> dict:
    """{KIT, MODE, file} read by AST from the kit's entry `__init__.py` (KIT = the kit version string, MODE = its mode word). No import, no torch."""
    path = os.path.join(kit_dir, "__init__.py")
    tree = ast.parse(open(path, "r", encoding="utf-8").read(), filename=path)
    out = {"file": path, "KIT": None, "MODE": None}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("KIT", "MODE"):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def kit_version(kit_dir: str) -> Optional[str]:
    return kit_constants(kit_dir).get("KIT")


def lock_check(kit_dir: str) -> list:
    """Every disagreement between this table and the kit's own constants (empty = locked)."""
    kc = kit_constants(kit_dir)
    bad = []
    if not kc.get("KIT"):
        bad.append(f"KIT not found in {kc['file']}")
    if kc.get("MODE") != KIT_MODE:
        bad.append(f"the kit's MODE is {kc.get('MODE')!r}, the package expects {KIT_MODE!r} ({kc['file']})")
    return bad
