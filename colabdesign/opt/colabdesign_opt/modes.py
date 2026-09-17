"""The mode table — the ONE place a mode name becomes a lever set, a numerics class and a guarantee. Modes are named by what they
GUARANTEE against stock (`off`) (`opt_core.modes`: the mode contract), not by what they switch on; the lever SETS are derived from the lever
registry (registry.py, the one source of truth; `levers_of`) — they resolve to:

| mode | levers (registry order) | guarantee vs stock (`off`) | gates | numerics class |
|---|---|---|---|---|
| `off` | none — BindCraft's design step on stock ColabDesign in a subprocess proven clean of all the kit's variables and modules (stock_launch.py) | is stock | — | stock |
| `exact` | `compilecache` + `lowercache` + `parcompile` + `hoist_prev` — every registered lever of numerics class `exact` (the persistent compile cache, the persisted design-program executables that skip jax's per-trajectory trace + lower, XLA's module-parallel compile, the device-resident recycle state) | stock's bytes from the kit's process: the same design states, trajectory and files bit for bit at the same seed | none | exact (tier 1) |
| `fast` (default) | `exact`'s four + `nosub` + `nosub_fn` (both executables unchunked) + `trimul` (fused Pallas triangle multiplication) + `triatt` (Pallas flash attention v2, supersedes `pallas`, which `fast-no-triatt` restores) + `opm_fold` + `ln` + `proj` + `transition` (fused Transition MLP where the kernel provider names a fused row for the call's cell) + `txla` (forward-only attention calls on the core's bridge; rows by its own table) | never bitwise; held PER STEP to the accuracy band against stock's own reference design states (gradient / logit cosine, per-step sequence argmax inside stock's replicate floor, every loss term) — whole trajectories may diverge (BindCraft's discrete stage decisions amplify last-bit differences); run-to-run bitwise | nosub / nosub_fn: stock's programs at <= 384 tokens (`reason=gated`); transition: the stock class, per call, where the provider's measured cell names the stock statement (`fallback_by=cell_xla:n`) | precision (tier 2) |

Any word outside the table is an unknown mode (`unknown mode … (expected off|exact|fast, or <mode>-no-<lever>…)`, exit 3). There is no
`big` mode (no memory lever; one GPU per design).

The kit rule (levers.py): a kit accepts everything stock accepts — a lever that cannot engage on the host steps aside BY NAME (`LEVER name=<x>
state=skipped reason=cannot_run|no_attention_kernel …`) and the mode runs the rest of its set; the ACTIVE line's `levers=` lists what is on and
`skipped=` what stepped aside, EVIDENCE carries `skipped=`, the exit code is the design's. Only configuration refuses (exit 3): an unknown
word, the variable disagreeing with `--mode`, colabdesign off its pinned commit, the core pin gate.

The mode word's subtractive form `<mode>-no-<lever>[-no-<lever>...]` (names.ABLATION_SEP; `resolve`) runs a kit-route mode WITHOUT the named
members of its lever set — an A/B attribution run, never one of the table's modes: it resolves to tier none, class `ablation`, the remaining
levers in registry order (a dropped lever that supersedes another RESTORES the other, named `restored=`), and every line that names the mode
carries the word itself plus `base=<mode> ablated=<levers>` (the ACTIVE line) / `ablated=<levers>` (EVIDENCE, EXIT); each dropped lever prints
its LEVER line with `state=off reason=ablated`. A base outside the kit route, a lever outside the base's set, a repeated lever, a replaced
lever alone, or a word that drops every lever of the base is an unknown mode like any other.

`tier`: 1 = bitwise identical to stock (`exact`), 2 = per step within the accuracy band against stock's reference design states (`fast`). The
package variable COLABDESIGN_OPT holds a mode name for the `.pth` route (`_autoload.py`); the CLI's `--mode` and the variable must agree when
both are given (cli.py). No lever has a switch outside this table and the subtractive word: a mode IS its lever set.
"""
from __future__ import annotations

import os
from typing import Dict, NamedTuple, Optional, Tuple

import re

from opt_core.modes import ModeError, ModeTable, levers_label

from . import registry
from .names import ABLATION_CLASS, ABLATION_SEP, DEFAULT_MODE, KIT_ROUTE_MODES, MODE_FAST, MODE_NAMES, MODE_OFF, split_mode_word

MODE_EXACT = "exact"                                        # the bitwise tier's name: in the table only while the registry holds an exact-class lever (levers_of)
LEVER_ID = re.compile(r"[a-z][a-z0-9_]*$")                  # a lever id's form (registry.LEVERS keys): the subtractive word splits on names.ABLATION_SEP unambiguously

ENV = "COLABDESIGN_OPT"                                     # the mode, for the .pth route and as the CLI's default
SIZE_GATE_TOKENS = 384                                      # stock's own sub-batch rule: above it both executables are traced with subbatch_size=4 (colabdesign/af/prep.py:25-32)
STOCK_SUBBATCH = 4                                          # the chunk stock sets above the gate (prep.py:31)


class Mode(NamedTuple):
    name: str
    levers: Tuple[str, ...]                                 # registry.LEVERS ids, applied in this order
    tier: Optional[int]                                     # 1 bitwise vs stock | 2 within stock's band | None (stock itself)
    numerics_class: Optional[str]                           # the mode's numerics word against stock, DERIVED from its levers' registry classes (class_of): exact | precision | approx; None = stock itself
    guarantee: str                                          # the mode's line of the table above, in one sentence
    route: str                                              # "stock" (stock_launch.py) | "kit" (kit_launch.py)


def candidates_of(mode: str) -> Tuple[str, ...]:
    """The candidate levers of a kit-route mode, registry order: fast = every registered lever, exact = the exact-class levers."""
    if mode == MODE_FAST:
        return registry.ORDER
    if mode == MODE_EXACT:
        return registry.levers_of_numerics("exact")
    return ()


def mode_set(candidates: Tuple[str, ...]) -> Tuple[str, ...]:
    """registry.mode_set: the candidates minus every candidate another candidate supersedes."""
    return registry.mode_set(candidates)


def levers_of(mode: str) -> Tuple[str, ...]:
    """The lever set of a kit-route mode, derived from the registry: its candidates minus the ones a candidate supersedes; registry order."""
    return mode_set(candidates_of(mode))


def class_of(levers: Tuple[str, ...]) -> Optional[str]:
    """A lever set's numerics word against stock — the widest class among its levers in registry.NUMERICS order (exact < precision < approx):
    `exact` iff every lever is exact-class; None for the empty set (stock itself)."""
    if not levers:
        return None
    return max((registry.LEVERS[l].numerics for l in levers), key=registry.NUMERICS.index)


MODES: Dict[str, Mode] = {
    MODE_OFF: Mode(MODE_OFF, (), None, None, "stock: the design script in a subprocess proven clean of every kit variable and module", "stock"),
    **({MODE_EXACT: Mode(MODE_EXACT, levers_of(MODE_EXACT), 1, class_of(levers_of(MODE_EXACT)), "identical outputs to stock, from the kit's process (same seed: the same design states, trajectory and files, bit for bit); today: the persistent compile cache, the module-parallel compile and the device-resident recycle state", "kit")}
       if levers_of(MODE_EXACT) else {}),
    MODE_FAST: Mode(MODE_FAST, levers_of(MODE_FAST), 2, class_of(levers_of(MODE_FAST)), f"every lever: the exact-class ones + nosub/nosub_fn + the kit's Pallas kernels (flash attention on every eligible attention call, "
                    f"fused triangle multiplication on every call): never bitwise vs stock, held to the per-step accuracy band against stock's reference design states, "
                    f"run-to-run bitwise; per-process fallback census, fail-closed; above {SIZE_GATE_TOKENS} tokens both executables (forward-only and forward+backward) are also traced unchunked", "kit"),
}


def _unknown(name) -> str:
    """The ONE unknown-mode text: the core's form, extended by the subtractive word's clause."""
    return (f"unknown mode {name!r} (expected {'|'.join(MODES)}, or "
            + ", ".join(f"{m}{ABLATION_SEP}<lever>[{ABLATION_SEP}<lever>...] without the named levers of {m}'s {levers_label(MODES[m].levers)}" for m in KIT_ROUTE_MODES) + ")")


TABLE = ModeTable(modes=MODE_NAMES, default=DEFAULT_MODE, unknown_message=_unknown)  # validated by the core: unique names, contains off, default in the table; a name outside it raises `unknown mode …`
assert tuple(MODES) == MODE_NAMES and all(MODES[m].route == "kit" for m in KIT_ROUTE_MODES) and MODES[MODE_OFF].route == "stock"   # names.py's literal tuple (the .pth hook's copy) == the derived table


class Resolved(NamedTuple):
    mode: str                                               # the word as given (normalised): a table mode, or its subtractive form
    levers: Tuple[str, ...]                                 # the levers this run installs, registry order
    tier: Optional[int]
    numerics_class: object                                  # the table's word (exact | precision | approx), None (stock), or names.ABLATION_CLASS for a subtractive word
    route: str
    line: str                                               # the guarantee sentence (the ACTIVE line's `line=`)
    base: Optional[str] = None                              # subtractive word only: the table mode it subtracts from
    ablated: Tuple[str, ...] = ()                           # subtractive word only: the levers dropped, registry order
    restored: Tuple[str, ...] = ()                          # subtractive word only: levers a dropped lever replaced, back in the run's set (registry Supersession), registry order


def resolve(mode: Optional[str]) -> Resolved:
    """Resolve a mode word (None / empty = the default) against the table, or its subtractive form against the base's lever set; anything else
    raises ValueError with the one unknown-mode text."""
    base, drops = split_mode_word(mode)
    if not drops:
        try:
            name = TABLE.check(mode)
        except ModeError as e:
            raise ValueError(str(e)) from None
        m = MODES[name]
        return Resolved(m.name, m.levers, m.tier, m.numerics_class, m.route, m.guarantee)
    m = MODES.get(base)
    universe = candidates_of(base) if m is not None else ()                     # the base's candidate levers (registry order): its set plus the levers that members of it replace
    ok = (m is not None and base in KIT_ROUTE_MODES and all(LEVER_ID.match(d) for d in drops) and len(set(drops)) == len(drops) and set(drops) <= set(universe))
    kept = mode_set(tuple(l for l in universe if l not in drops)) if ok else ()
    ok = ok and bool(kept) and all(d in mode_set(tuple(l for l in universe if l not in (set(drops) - {d}))) for d in drops)   # every drop takes a lever OUT of the run (a replaced lever is dropped only together with its replacer); something is left
    if not ok:
        raise ValueError(_unknown(str(mode).strip()))
    ablated = tuple(l for l in universe if l in drops)
    restored = tuple(l for l in kept if l not in m.levers)                      # replaced levers back in the set because their replacer is dropped: named on every line (registry Supersession)
    word = base + "".join(ABLATION_SEP + l for l in ablated)                  # canonical: registry order (fast-no-b-no-a names the same run as fast-no-a-no-b)
    return Resolved(word, kept, None, ABLATION_CLASS, m.route,
                    f"{ABLATION_CLASS}: {base} without {','.join(ablated)}{(' (restored: ' + ','.join(restored) + ')') if restored else ''} - an A/B attribution run, none of the table's modes, no guarantee against stock claimed",
                    base, ablated, restored)


def describe(res: Resolved) -> str:
    return (f"mode={res.mode} levers={levers_label(res.levers)} route={res.route}" + (f" base={res.base} ablated={','.join(res.ablated)}" if res.ablated else "")
            + (f" restored={','.join(res.restored)}" if res.restored else ""))


def mode_from_env(environ=None) -> Optional[str]:
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV) or "").strip().lower()
    return v or None


def table_rows() -> list:
    """README/CHANGES §Modes, rendered: one markdown row per mode from MODES (the docs are locked to these rows by the tests)."""
    head = ["| mode | levers | numerics class | tier | guarantee vs stock (`off`) | arm |", "|---|---|---|---|---|---|"]
    return head + [f"| `{m.name}`{' (default)' if m.name == DEFAULT_MODE else ''} | {levers_label(m.levers)} | {m.numerics_class if m.numerics_class is not None else 'stock'} | "
                   f"{m.tier if m.tier is not None else chr(8212)} | {m.guarantee} | {'kit_launch.py' if m.route == 'kit' else 'stock_launch.py'} |" for m in MODES.values()]
