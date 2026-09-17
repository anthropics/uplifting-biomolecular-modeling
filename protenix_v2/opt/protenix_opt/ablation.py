"""The ablation switch: ``MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]`` runs a mode WITHOUT the named levers of its set — one A/B
attribution run, never one of the kit's modes and never a row's setting.

Semantics (the same on every route the package activates: ``run.sh pred|check``, the console entry point, the drop-in environment
route, the multi-GPU line's ranks, the stock CLI's autoloading child processes — the variable stays in the environment they inherit):

* names are registry lever names (``registry.LEVERS``), comma-separated, whitespace ignored, duplicates folded; unset or blank = no
  ablation and the run is byte for byte the mode's (no ``ablated=`` token anywhere);
* every name must be a lever of the mode's set (``modes.MODES[mode]``; big's is ``modes.big_levers()``) and switch-driven (its
  registry row names the switches it reads: ``env_keys`` + ``conditional``) — an unknown name, a lever outside the mode, a lever the
  big line applies without a switch (its memory levers), a list that empties the mode, and any name under ``--mode off`` are refused
  BY NAME (``NOT ACTIVE: MODEL_OPT_LEVERS_OFF …``, exit 3); nothing runs under a name it does not have;
* applied after ``modes.resolve()``: each ablated lever's switches leave the mode's exports (env.sh's delta, the README row's pre/post
  exports, the package's extras) and are unset in the process if a caller had set them, so the unit reads its switch absent and does
  not engage; the kit's own applied record is then read back — an ablated lever whose marker still says it engaged makes the run
  refuse by name rather than report an ablation that did not happen;
* the ACTIVE / DRY-RUN / FINAL lines gain ``ablated=<names>`` (in request order), each ablated lever's LEVER line reads
  ``state=off reason=ablated``, and the levers kept engage all-or-refuse exactly as the mode does.
"""
from typing import Dict, List, Mapping, Optional, Tuple
import os
import sys

from .registry import LEVERS

ENV = "MODEL_OPT_LEVERS_OFF"
REASON = "ablated"                                      # the LEVER line's reason token for an ablated lever
TOKEN = "ablated"                                       # the ACTIVE / DRY-RUN / FINAL lines' key (ablated=<names>)


class AblationError(ValueError):
    """A MODEL_OPT_LEVERS_OFF request the kit refuses by name (the activation's NOT ACTIVE reason)."""


def requested(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The lever names ``MODEL_OPT_LEVERS_OFF`` lists, in order, duplicates folded; [] when unset or blank."""
    raw = (os.environ if environ is None else environ).get(ENV) or ""
    out: List[str] = []
    for w in raw.replace(" ", "").split(","):
        if w and w not in out:
            out.append(w)
    return out


def switches(name: str) -> Tuple[str, ...]:
    """The switches an ablation of ``name`` removes: the registry row's env_keys and its runtime-probed (conditional) switches."""
    lv = LEVERS[name]
    return tuple(dict.fromkeys(lv.env_keys + lv.conditional))


def mode_levers(mode: str) -> List[str]:
    from .modes import MODES, big_levers
    return list(big_levers() if mode == "big" else (MODES.get(mode) or []))


def validate(mode: str, names: List[str], row=None, row_key: str = None) -> List[str]:
    """``names`` checked against ``mode``: raises AblationError naming every unknown lever, every lever outside the mode's set, every
    lever with no switch to remove, and a request that leaves the mode with no lever; returns ``names`` unchanged otherwise."""
    if not names:
        return []
    if mode == "off":
        raise AblationError(f"{ENV}={','.join(names)}: mode off applies no lever, there is nothing to ablate (unset {ENV} for the stock route)")
    members = mode_levers(mode)
    unknown = [n for n in names if n not in LEVERS]
    outside = [n for n in names if n in LEVERS and n not in members]
    switchless = [n for n in names if n in LEVERS and n in members and not switches(n)]
    problems = []
    if unknown:
        problems.append(f"{','.join(unknown)}: not a lever of this kit (registry levers: {','.join(LEVERS)})")
    if outside:
        problems.append(f"{','.join(outside)}: not in mode {mode}'s lever set ({','.join(members)})")
    if switchless:
        problems.append(f"{','.join(switchless)}: applied by the big line without a switch of its own, not removable by {ENV}")
    from .registry import in_row
    if row is not None:                                                 # per-card membership: a lever that is not part of the arm on this row cannot be ablated here — refused by name
        for n in [x for x in names if x in members]:
            lv = LEVERS[n]
            repl = [r for r in lv.replaced_by if r in members and r not in names and (not LEVERS[r].row_dependent or in_row(LEVERS[r], row))]
            if repl:
                problems.append(f"{n}: not part of the arm on kernel key {row_key or '?'} (its slot is served by {','.join(repl)} there — ablate {','.join(repl)} instead)")
            elif lv.row_dependent and not in_row(lv, row):
                problems.append(f"{n}: not part of the arm on kernel key {row_key or '?'} (the compositions row does not list it)")
    for m in members:                                                   # a lever left on that requires an ablated one: refused by name (ablate both), never implied
        if m not in names:
            if row is not None and LEVERS[m].row_dependent and not in_row(LEVERS[m], row):
                continue                                                #   … a row-dependent lever the compositions row does not list is not on here: it requires nothing (dit_fused on the cc-8.0 rows, which list dit_attn without it)
            need = [r for r in LEVERS[m].requires if r in names]
            if need:
                problems.append(f"{','.join(need)}: required by {m}, which stays on (ablate {m} with it)")
    if not problems and not [n for n in members if n not in names]:
        problems.append(f"the request removes every lever of mode {mode}: that is --mode off, not an ablation")
    if problems:
        raise AblationError(f"{ENV} refused — " + "; ".join(problems))
    return list(names)


def pre_switches(names: List[str], row) -> List[str]:
    """The ablated levers' switches that are README-row PRE exports on this row (read BY env.sh: PTX_T_TRIMUL, PTX_T_ATT): their ablation
    re-sources env.sh without them (``modes.resolve(skip_pre=...)``) so env.sh's own default word for the slot stands — removing the derived
    export instead would leave the slot empty for the lever that keeps it (k2b when triattn_cuda is ablated)."""
    pre = set((row or {}).get("pre", {}))
    out: List[str] = []
    for n in names:
        for k in LEVERS[n].env_keys:
            if k in pre and k not in out:
                out.append(k)
    return out


def restored_words(names: List[str], row) -> Dict[str, str]:
    """For each ablated lever selected by the VALUE of a switch env.sh reads (registry ``words`` on a README-row PRE switch), the word of the
    lever it replaces on this row (the registry lever whose ``replaced_by`` names it and which declares a word of its own for the same
    switch) — exported in the ablated word's place so the run without the lever is the lever it displaced, not env.sh's bare default.
    ``{switch: word}``; a slot whose displaced lever has no word of its own (k2b_flash_triattention under PTX_T_ATT) is simply left to
    env.sh's default, which IS that lever."""
    pre = set((row or {}).get("pre", {}))
    out: Dict[str, str] = {}
    for n in names:
        mine = {k: w for k, w in LEVERS[n].words if k in pre}
        if not mine:
            continue
        for other in LEVERS.values():
            if n in other.replaced_by and other.name not in names:
                for k, w in other.words:
                    if k in mine and k not in out:
                        out[k] = w
    return out


def apply(res, names: List[str], resourced=(), restored=None) -> Dict[str, List[str]]:
    """Remove the ablated levers' switches from a ``modes.Resolution`` in place: out of ``exports`` / ``extras`` / ``pre_exports`` and
    into ``unsets`` (a caller's pre-set value must not keep the lever on). Returns ``{lever: [switch, …]}`` — every switch the lever's
    registry row names, whether or not this row exported it. A lever whose pre switch is in ``resourced`` (pre_switches: env.sh was
    sourced again without it) keeps its runtime-probed (conditional) keys as env.sh now set them: only its own env_keys leave; a switch in
    ``restored`` (restored_words: it now carries the displaced lever's word) stays exported."""
    removed: Dict[str, List[str]] = {}
    resourced = set(resourced); restored = dict(restored or {})
    for n in names:
        keys = list(switches(n))
        if resourced & set(LEVERS[n].env_keys):
            keys = [k for k in keys if k in LEVERS[n].env_keys]
        for k in keys:
            if k in restored:                      # the switch now carries the displaced lever's word (restored_words): it stays exported with that word
                continue
            res.exports.pop(k, None); res.extras.pop(k, None); res.pre_exports.pop(k, None)
            if k not in res.unsets:
                res.unsets.append(k)
        removed[n] = keys
    return removed


def leaks(names: List[str], applied: List[str], environ: Mapping[str, str]) -> Dict[str, str]:
    """Ablated levers the process engaged anyway, by the kit's own evidence: a marker-probed lever whose family marker reads on, an
    env-probed lever whose switches are still all set (to ITS word, for a lever env.sh selects by the switch's value). ``{lever: evidence}``;
    empty = the ablation took effect."""
    from .stack import MARKERS, BAD, MODULE_PROOF
    out: Dict[str, str] = {}
    for n in names:
        lv = LEVERS[n]
        if lv.probe == "marker" and n in MARKERS:
            if lv.words and any(environ.get(k) != w for k, w in lv.words):   # a word-selected provider (trimul_tx_exact = PTX_E_TRIMUL=tx): ablated, the switch carries the displaced lever's word — the family marker (FPF:enabled(trimul_in,trimul_out)) is the slot's and reads on for the provider that took the slot back, so it is not this lever's
                continue
            family, good_marks = MARKERS[n]
            hits = [a for a in applied if a.startswith(family) and any(m.casefold() in a.casefold() for m in good_marks) and not any(b in a for b in BAD)]
            proof = MODULE_PROOF.get(n)                                  # a lever proven by its provider module: the marker counts as this lever's only with the module loaded
            if hits and (proof is None or proof in sys.modules):
                out[n] = hits[0]
        elif lv.env_keys and all(environ.get(k) for k in lv.env_keys):
            if lv.words and any(environ.get(k) != w for k, w in lv.words):   # a word-selected lever: the switch now carries the displaced lever's word — set, but not to this lever
                continue
            out[n] = "switch still set: " + ",".join(f"{k}={environ.get(k)}" for k in lv.env_keys)
    return out


def token(names: List[str]) -> str:
    """`` ablated=<a,b>`` for the ACTIVE / DRY-RUN / FINAL lines; the empty string when nothing is ablated (the lines stay as they were)."""
    return f" {TOKEN}={','.join(names)}" if names else ""
