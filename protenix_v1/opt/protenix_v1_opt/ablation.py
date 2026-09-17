"""The ablation switch: ``MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]`` runs a mode WITHOUT the named levers of its set — one A/B
attribution run, never one of the kit's modes. The variable name, the ``ablated=`` token and the ``state=off reason=ablated`` LEVER line are
the grammar every kit of the shared core uses.

Semantics on this kit, whose switch is the arm string ``levers_ptx1.apply()`` takes (``<trimul>[+lever...]``, modes.KIT_MODES):

* names are the mode's lever words, comma-separated, whitespace ignored, duplicates folded: the arm's triangle-multiplication word
  (``exact`` | ``fast``), the arm's levers (modes.LEVERS: gblock gflash xtr ttr sg hoist) and, under ``big``, the memory levers of
  ``big.LINE``; unset or blank = no ablation and the run is byte for byte the mode's (no ``ablated=`` token anywhere);
* every name must be a lever of the mode's set — an unknown name, a lever the mode does not compose (``sg`` under big, ``gflash`` under
  exact, a memory lever outside big), a list that empties the mode, and any name under ``--mode off`` are refused BY NAME
  (``NOT ACTIVE: … MODEL_OPT_LEVERS_OFF …``, exit 3); nothing runs under a name it does not have;
* applied inside ``modes.resolve()``, so every route reads one resolution (``run.sh pred|check``, the console entry point, the drop-in
  environment route, the ``--n_gpu`` rank processes, which inherit the variable): an arm word leaves the arm string (the trimul word ablated =
  the stock triangle multiplication, ``stock+…``), so the kit never installs it; a memory lever is the big line's in-process selection OFF by
  name (opt_core.mem.apply ``switches={lever: False}``, the mechanism the ×P line already uses), so the core never applies it;
* the ARMED and ACTIVE lines gain ``ablated=<names>`` (request order), each ablated lever's LEVER line reads ``state=off reason=ablated``,
  and the levers kept engage all-or-refuse exactly as the mode does (the exit rule judges the kept set).
"""
from __future__ import annotations

import os
from typing import List, Mapping, Optional, Sequence, Tuple

ENV = "MODEL_OPT_LEVERS_OFF"
REASON = "ablated"                                      # the LEVER line's reason token for an ablated lever
TOKEN = "ablated"                                       # the ARMED / ACTIVE lines' key: ablated=<lever,…>
TRIMUL_WORDS = ("exact", "fast")                        # the arm's triangle-multiplication words that are levers (``stock`` is the absence of one)


class AblationError(ValueError):
    """A refusal by name; str(e) is the NOT ACTIVE reason."""


def requested(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The lever names in ``MODEL_OPT_LEVERS_OFF`` (comma-separated, stripped, duplicates folded, order kept); [] when unset or blank."""
    env = os.environ if environ is None else environ
    out: List[str] = []
    for tok in (env.get(ENV) or "").split(","):
        name = tok.strip()
        if name and name not in out:
            out.append(name)
    return out


def arm_words(arm: str) -> Tuple[str, ...]:
    """The lever words an arm string carries: its trimul word when it is one of TRIMUL_WORDS, then its levers, in arm order."""
    parts = [p for p in str(arm).replace(",", "+").split("+") if p]
    if not parts:
        return ()
    head, rest = parts[0], parts[1:]
    if head in TRIMUL_WORDS:
        return (head, *rest)
    return tuple(rest if head == "stock" else parts)


def mode_set(arm: str, memory_levers: Sequence[str] = ()) -> Tuple[str, ...]:
    """The mode's lever set an ablation may name: the arm's words then the memory levers (big.LINE under big, () elsewhere)."""
    return (*arm_words(arm), *tuple(memory_levers))


def validate(mode: str, names: Sequence[str], arm: str, memory_levers: Sequence[str] = (), known: Sequence[str] = ()) -> None:
    """AblationError (by name) unless every requested name is a lever of `mode`'s set and the request leaves the mode something to apply.
    `known` = every lever word this kit has (the grammar's trimul words + levers + the memory levers): a name outside it is `unknown`, a known
    name outside the mode's set is `not a lever of mode <mode>` with the set spelled out."""
    if not names:
        return
    if mode == "off":
        raise AblationError(off_refusal(names))
    allowed = mode_set(arm, memory_levers)
    universe = set(known) | set(allowed)
    unknown = [n for n in names if n not in universe]
    if unknown:
        raise AblationError(f"{ENV}={','.join(names)}: unknown lever name(s) {unknown} (this kit's levers: {sorted(universe)})")
    outside = [n for n in names if n not in allowed]
    if outside:
        raise AblationError(f"{ENV}={','.join(names)}: {outside} not a lever of mode {mode} (its set: {list(allowed)}); an ablation names levers the mode composes")
    left = [x for x in allowed if x not in names]
    if not left:
        raise AblationError(f"{ENV}={','.join(names)}: the request removes every lever of mode {mode}: that is `--mode off`, not an ablation")


def off_refusal(names: Sequence[str]) -> str:
    """The reason the stock route gives: it applies no lever, so an ablation request under it is refused by name."""
    return f"{ENV}={','.join(names)} refused under --mode off: the stock route applies no lever, there is nothing to ablate (unset the variable)"


def reduce_arm(arm: str, names: Sequence[str]) -> str:
    """The arm string without the ablated words: an ablated trimul word becomes ``stock``, ablated levers leave the list (arm order kept)."""
    parts = [p for p in str(arm).replace(",", "+").split("+") if p]
    if not parts:
        return arm
    head, rest = parts[0], parts[1:]
    if head not in TRIMUL_WORDS and head != "stock":            # a lever-only arm: stock trimul implied
        head, rest = "stock", parts
    if head in names:
        head = "stock"
    kept = [lv for lv in rest if lv not in names]
    return "+".join([head, *kept])


def split(names: Sequence[str], memory_levers: Sequence[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(arm words ablated, memory levers ablated), each in request order."""
    mem = set(memory_levers)
    return tuple(n for n in names if n not in mem), tuple(n for n in names if n in mem)


def switches(names: Sequence[str], memory_levers: Sequence[str]) -> dict:
    """The big line's in-process selection for the ablated memory levers: {lever: False} (opt_core.mem.apply `switches`)."""
    mem = set(memory_levers)
    return {n: False for n in names if n in mem}


def token(names: Sequence[str]) -> str:
    """`` ablated=<a,b>`` for the ARMED / ACTIVE lines; the empty string when nothing is ablated (the lines read exactly as before)."""
    return f" {TOKEN}={','.join(names)}" if names else ""
