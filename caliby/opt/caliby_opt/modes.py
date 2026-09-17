"""Modes, variants and the mode table: how a package mode resolves to one row of the kits' lever switches.

This module is the one place the table lives: every row's switches are stated here (``ROWS``) and exported as written; the
xattempt_addon's HOWTO (opt/forward/xattempt_addon/HOWTO.md, "Enable" and "Ensemble-conditioned design") documents the same lines. The rows:

  mode    variant      row              switches
  off     single|e32   —                stock: no switch set, the pinned upstream tree
  fast    single       single/serial    the kit's row "X": eight switches, CALIBY_X_CLEAN=0, one clean worker
  fast    single       single/loader    row "XL": the same switches with CALIBY_X_CLEAN=loader; clean workers N>1 (the row's own N is 4)
  exact   ensemble32   ensemble32       the kit's ensemble line: twelve switches (no clean lever: N>1 clean workers are upstream's own
                                        joblib-parallel clean_pdbs there)
  exact   single       refused by name  the single-sequence route differs from stock deterministically (energy last-bit class): its rows
                                        ship as fast, tier 2
  fast    ensemble32   ensemble32       the same row as exact: a bit-identical row satisfies fast's guarantee, and no other ensemble lever exists

Each variant has one kit ROW; ``KIT_MODE_OF`` names the mode that states that row's guarantee: ``fast`` on ``single`` — tier 2, the
forward-pass band guarantee: deterministic, Potts energies within the last bit of stock's, hence the same sequences as stock except where
such a last-bit difference flips a sampled residue (within stock's seed-to-seed variation) — and ``exact`` on
``ensemble32`` — tier 1, stock's outputs; ``fast`` on ``ensemble32`` runs that same row (``MODES_OF``). ``exact`` on ``single`` (no
bit-identical row exists there) and an unknown name are refused by name (``mode_refusal``, the one refusal every route
surfaces: the command line with exit 2, the ``.pth`` route and ``check()`` as their NOT ACTIVE line). Without ``--mode`` the mode is
``fast`` on both variants (``DEFAULT_MODE``), never ``exact``.

``resolve(mode, variant, clean_workers)`` picks the row. For ``fast/single`` the two rows differ in one lever (the clean
lever, ``CALIBY_X_CLEAN``): its value is inert at one clean worker (with one clean worker the stock serial ``clean_pdbs`` path runs
whatever the switch says), so the resolver exports the serial row for ``clean_workers == 1`` (the default) and the ``loader`` row for
``clean_workers > 1``. The ``ensemble32`` row carries no clean lever: ``clean_workers > 1`` there is upstream's own parallel clean
(``caliby.clean_pdbs(num_workers=N)``, joblib), the row unchanged. ``default_mode(variant)`` is ``fast``, the command line's
mode: the command line (cli.py, run.sh) falls back to it when neither ``--mode`` nor ``CALIBY_OPT`` is given; the ``.pth`` route of a
user's own script (_autoload.py) activates only under ``CALIBY_OPT=fast`` (single) or ``CALIBY_OPT=exact`` with
``CALIBY_VARIANT=ensemble32`` — with ``CALIBY_OPT`` unset that script runs stock, nothing activated.

Row keys are the kits' own switch names (``CALIBY_FAST_*``: defined by opt/forward/fast_inference's patches; ``CALIBY_X_*``: opt/forward/xattempt_addon's own; the code of all of them is served from opt/forward/xattempt_addon/fast/);
their meaning per value is the kits' own text (registry.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from . import stack

MODES: Tuple[str, ...] = ("off", "exact", "fast")
VARIANTS: Tuple[str, ...] = ("single", "ensemble32")
DEFAULT_VARIANT = "single"
KIT_MODE_OF: Dict[str, str] = {"single": "fast", "ensemble32": "exact"}   # the mode that states each variant's row guarantee (single's row is tier 2, ensemble32's tier 1)
MODES_OF: Dict[str, Tuple[str, ...]] = {"single": ("off", "fast"), "ensemble32": ("off", "exact", "fast")}   # the modes that resolve per variant; fast on ensemble32 = the exact row
DEFAULT_MODE = "fast"                                                  # the mode when neither --mode nor CALIBY_OPT is given, both variants; never exact
TIER_OF: Dict[str, int] = {"exact": 1, "fast": 2}                       # exact: stock's outputs; fast: the forward-pass band guarantee
REFUSED: Dict[Tuple[str, str], str] = {                                # (mode, variant) refused by name -> the words (mode_refusal: the one refusal site)
    ("exact", "single"): ("exact refused: the single-sequence route differs from stock deterministically (energy last-bit class); "
                          "use --mode fast on single (same levers, tier-2 band guarantee) or the ensemble route (--mode exact, tier 1)"),
}
CLEAN_LEVER = "loader"                                                 # fast/single with --clean_workers > 1: the parallel clean row (single/loader, CALIBY_X_CLEAN=loader)
SWITCH_RE = re.compile(r"\b(CALIBY_(?:FAST|X)_[A-Z_]+)=(\S+)")
@dataclass(frozen=True)
class Row:
    key: str                          # single/serial | single/loader | ensemble32
    variant: str
    switches: str                     # the row's NAME=value switches, in the order exported
    label: str                        # the kits' own name for the row
    clean_workers: Optional[int]      # the writer's --clean_workers the row is stated with (None: not part of the row)
    ensemble: int = 0                 # the writer's --ensemble the row is stated with (0: single-structure design)
    note: str = ""


SINGLE_SWITCHES = ("CALIBY_FAST_SAMPLER=2 CALIBY_FAST_POTTS_PARAMS=2 CALIBY_X_SPARSE_EXACT=2 CALIBY_X_LCP=1 CALIBY_X_CLEAN={clean} "
                   "CALIBY_X_BG_CIF=fork CALIBY_X_MULTISEQ=1 CALIBY_X_CIF_WORKERS=8")          # the single-sequence rows differ in the clean lever only
ENSEMBLE_SWITCHES = ("CALIBY_FAST_SAMPLER=2 CALIBY_FAST_POTTS_PARAMS=2 CALIBY_X_SPARSE_EXACT=2 CALIBY_X_LCP=1 CALIBY_X_TIED_DET=1 CALIBY_X_ENS_WORKERS=8 "
                     "CALIBY_X_PP_CACHE=1 CALIBY_X_BG_CIF=fork CALIBY_X_PP_NOSYNC=1 CALIBY_X_PP_FASTPDB=1 CALIBY_X_MULTISEQ=1 CALIBY_X_CIF_WORKERS=8")

ROWS: Dict[str, Row] = {
    "single/serial": Row("single/serial", "single", SINGLE_SWITCHES.format(clean="0"), "X", 1,
                         note="serial clean (CALIBY_X_CLEAN=0, one clean worker)"),
    "single/loader": Row("single/loader", "single", SINGLE_SWITCHES.format(clean="loader"), "XL", 4,
                         note="in-process parallel clean (CALIBY_X_CLEAN=loader, N clean workers; the row's own N is 4)"),
    "ensemble32": Row("ensemble32", "ensemble32", ENSEMBLE_SWITCHES, "ensemble command", 1, ensemble=32,
                      note="the twelve-switch ensemble row (no clean lever: N clean workers are upstream's joblib clean; CALIBY_X_ENS_WORKERS=8 conformer workers)"),
}

# The other value the kits state for a row's switch, and the flag that selects it (facts; the resolver applies the flag, never a policy decision).
ALTERNATIVES: Dict[str, str] = {
    "clean lever": "fast/single: CALIBY_X_CLEAN=0 (one clean worker; the default) | loader (--clean_workers N>1)",
    "ensemble workers": "exact/ensemble32: CALIBY_X_ENS_WORKERS=8 (the row's conformer workers; no flag changes it)",
}


@dataclass(frozen=True)
class Resolution:
    mode: str
    variant: str
    row: Optional[Row]
    env: Dict[str, str]               # the switches to export (empty for off)
    clean_workers: int
    source: Optional[str]             # "modes.py:<row key>"
    label: Optional[str]              # the row's label
    alternatives: Tuple[str, ...]     # keys of ALTERNATIVES that apply to the row

    @property
    def line(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.env.items())


def line_switches(text: str) -> Dict[str, str]:
    """The ``NAME=value`` pairs on a switch line, in the order written, up to a shell comment (``  # ...``); shell parameter
    defaults like ``${EW:-4}`` resolve to the default the line states."""
    text = re.split(r"\s#", text, maxsplit=1)[0]
    out: Dict[str, str] = {}
    for name, value in SWITCH_RE.findall(text):
        m = re.fullmatch(r"\$\{[A-Z_]+:-([^}]*)\}", value)
        if m:
            value = m.group(1)
        out[name] = re.sub(r"[`'\".,;)]+$", "", value).strip("`'\"")
    return out


def row_switches(row: Row) -> Dict[str, str]:
    """The ``NAME=value`` pairs of the row, in the order exported."""
    out = line_switches(row.switches)
    if not out:
        raise RuntimeError(f"row {row.key} carries no CALIBY_* switch: {row.switches!r}")
    return out


def is_kit_mode(mode: Optional[str]) -> bool:
    """A kit mode (the levers loaded: fast, exact) as opposed to ``off``; the lever-module set of a kit mode is the ``exact`` tree state."""
    return mode in TIER_OF


def default_mode(variant: Optional[str] = None) -> str:
    """``fast``: the command line's mode when neither --mode nor CALIBY_OPT is given, on both variants (never ``exact``)."""
    variant = variant or DEFAULT_VARIANT
    if variant not in MODES_OF:
        raise ValueError(f"unknown variant {variant!r} (expected {'|'.join(VARIANTS)})")
    return DEFAULT_MODE


def mode_refusal(mode: str, variant: Optional[str] = None) -> Optional[str]:
    """The one refusal site for a (mode, variant) pair: None when the pair resolves (``MODES_OF``: off and fast on both variants, exact on
    ensemble32), else the words — an unknown name, ``exact`` on single (``REFUSED``)."""
    variant = variant or DEFAULT_VARIANT
    if mode not in MODES:
        return f"unknown mode {mode!r} (expected {'|'.join(MODES)})"
    if variant not in VARIANTS:
        return f"unknown variant {variant!r} (expected {'|'.join(VARIANTS)})"
    if mode in MODES_OF[variant]:
        return None
    return REFUSED[(mode, variant)]


def row_for(mode: str, variant: str, clean_workers: Optional[int] = None) -> Optional[Row]:
    why = mode_refusal(mode, variant)
    if why:
        raise ValueError(why)
    if mode == "off":
        return None
    if variant == "single":
        if (clean_workers or 1) > 1:
            return ROWS["single/loader"]
        return ROWS["single/serial"]
    if variant == "ensemble32":
        return ROWS["ensemble32"]
    raise ValueError(f"unknown variant {variant!r} (expected {'|'.join(VARIANTS)})")


def resolve(mode: str, variant: Optional[str] = None, clean_workers: Optional[int] = None) -> Resolution:
    """A package mode + variant -> the kit row and the switches it exports: the whole row, never a subset (a mode is all of its
    levers; a lever that cannot run makes the mode refuse by name, activate.partial_exit). ``clean_workers`` is the writer's
    --clean_workers (default 1; upstream's clean_pdbs num_workers): for fast/single N>1 selects the parallel-clean row, elsewhere it is
    upstream's own parallel clean and the row is unchanged. A pair the table refuses (``mode_refusal``) raises ValueError with its words
    before anything else is looked at."""
    variant = variant or DEFAULT_VARIANT
    why = mode_refusal(mode, variant)
    if why:
        raise ValueError(why)
    cw = 1 if clean_workers is None else int(clean_workers)
    if cw < 1:
        raise ValueError("clean_workers must be >= 1")
    if mode == "off":
        return Resolution(mode, variant, None, {}, cw, None, None, ())
    row = row_for(mode, variant, cw)
    alts = ("clean lever",) if variant == "single" else ("ensemble workers",)
    return Resolution(mode, variant, row, row_switches(row), cw, f"modes.py:{row.key}", row.label, alts)


def table() -> Dict[str, Dict[str, str]]:
    """Every row's switches (for check --json and the tests)."""
    return {k: row_switches(r) for k, r in ROWS.items()}


def jit_cache_key() -> str:
    """The stack key configs/<gpu>.env uses for the Triton cache directory (stack.stack_key: the shared core's rule over torch's values)."""
    return stack.stack_key()
