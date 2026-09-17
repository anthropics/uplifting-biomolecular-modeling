"""``MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]`` — the uniform ablation switch: one run of a mode
WITHOUT the named levers of its set. An A/B attribution device, never a mode and never a row setting; unset or blank = the mode byte
for byte (no ``withheld=`` token anywhere). The kit's own words keep working as they are (``--mode``, the FPF arm grammar, the big
line's selection): this module only removes names from what :func:`modes.resolve` returns, at that one place, so every route that
resolves a mode — ``run.sh pred|check|warm``, ``python -m rosettafold3_opt``, the drop-in ``ROSETTAFOLD3_OPT=<mode>`` hook in the fold
process and its rank processes (the variable rides the environment they inherit) — sees the same row.

Semantics:

* names are registry lever names (``registry.LEVERS``), comma-separated, whitespace ignored, duplicates folded, request order kept;
* every name must be a lever of the mode's set — the row's switch levers, its FPF arm's components, its package levers
  (``modes.KIT_LEVERS``), and under ``big`` the line's memory levers by their registry names (``big_<lever>``); an unknown name, a
  lever outside the mode's set, the memory policy ``mem`` (it rides every kit row, it is not a lever of a set), ``rowpair`` (it IS the
  ``--n_gpu`` line), a request that empties the mode, and any name under ``--mode off`` are refused BY NAME (``ValueError`` out of
  ``modes.resolve``: the CLI's ERROR / the hook's NOT ACTIVE line, exit 3) — nothing runs under a name it does not have;
* dependent levers are never implied: ``graph`` and ``graph_safe_ops`` ride one switch (``RF3_CUDAGRAPH``; the safe-op rewrites are on
  iff the sampler graph is) and are withheld together or not at all; a lever that stays and requires a withheld one is refused by name
  with the fix spelled out (``mkdit`` and ``big_atom_pair_local`` need ``hoist``; ``fpf_tg`` needs ``fpf_sapb`` or ``fpf_apb``;
  ``fpf_res`` needs ``fpf_trimul`` or ``fpf_ttr`` to fuse into);
* how a name leaves the row: a switch lever's variable is exported ``0`` (``RF3_CUDAGRAPH`` / ``RF3_HOIST``; the FPF arm's ``@L1`` step,
  which re-sets those flags in-process, leaves the arm with it so the flags read what the row exports); an FPF component leaves the arm
  string (``fast`` trimul -> ``stock``); a package lever leaves ``kit_levers``; a big memory lever becomes a named ``off`` switch of the
  line's selection (``big.switches_for``: ``off_by_flag`` on the BIG line, ``state=off reason=flag`` on its LEVER line);
* ``compile`` (``run.sh --no-compile``) is the release tree's word for a lever this kit does not have as a class (ABSENT): accepted in every
  mode, ``off`` included, it withholds nothing and the lines say ``compile=none`` — never refused, never silent;
* the ACTIVE / DRY-RUN / EXIT lines gain ``withheld=<names>`` (request order), each withheld lever's LEVER line reads ``state=off
  reason=withheld``, and the levers kept engage all-or-refuse exactly as the mode does (``xtr`` composed with a withheld ``ttr`` serves
  the widths ``ttr`` served: pf.ttr_keys reads the arm as applied).
"""
import os
from dataclasses import replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

ENV = "MODEL_OPT_LEVERS_OFF"
TOKEN = "withheld"                                        # the ACTIVE / DRY-RUN / EXIT lines' key (withheld=<names>)
REASON = "withheld"                                       # the LEVER line's reason token for a withheld lever
BIG_ROW = "big"                                       # the memory mode's row name (modes.KIT_MODES["big"], stack.BIG_MODE)
BIG_PREFIX = "big_"                                   # registry names of the big line's memory levers (registry.py) = this prefix + big.LEVERS name
NOT_WITHHOLDABLE = {                                      # registry names no request can remove, with the sentence
    "mem": "the memory policy rides every kit row (mem.POLICY); it is not a lever of a mode's set",
    "rowpair": "the row-sharded pair stack IS the --n_gpu line (big --n_gpu P); run --n_gpu 1 instead",
}
ABSENT = {                                                # release-tree lever words this kit does not have AS A CLASS: accepted in every mode (off too), withhold nothing, reported <name>=none
    "compile": "this kit has no torch.compile lever and upstream rf3 compiles nothing (run.sh --no-compile = MODEL_OPT_LEVERS_OFF=compile): a no-op, reported compile=none",
}
TOKEN_ABSENT = "none"                                     # the ACTIVE / DRY-RUN / EXIT lines' value for an absent word: ` compile=none`
TOGETHER = (("graph", "graph_safe_ops"),)                 # levers that ride one switch: withheld together or not at all
REQUIRES: Dict[str, Tuple[str, ...]] = {                  # lever -> levers it needs (all of them): a kept lever whose need is withheld is refused by name
    "mkdit": ("hoist",),                                  # the pair-bias layout lives in the RF3_HOIST roll-out cache (mkdit.py refuses without it)
    "big_atom_pair_local": ("hoist",),                  # the hoisted prefix is the patched site (levers._atom_pair_applies)
    "confln": ("confhoist",),                             # the whole-tensor layer norms it replaces are confhoist's hoisted prologue statements
}
REQUIRES_ANY: Dict[str, Tuple[str, ...]] = {              # lever -> levers of which at least one must stay
    "fpf_tg": ("fpf_sapb", "fpf_apb"),                    # the captured pairformer stack needs the graph-safe pair-bias attention (cached constant or the Triton kernel)
    "fpf_res": ("fpf_trimul", "fpf_ttr"),                 # the fused residual adds ride the FPF trimul / transition kernels' epilogues
}


class WithholdError(ValueError):
    """A MODEL_OPT_LEVERS_OFF request the kit refuses by name (modes.resolve raises it: the CLI's ERROR line / the hook's NOT ACTIVE, exit 3)."""


def _requested_raw(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The lever names ``MODEL_OPT_LEVERS_OFF`` lists, in order, whitespace ignored, duplicates folded; [] when unset or blank."""
    raw = (os.environ if environ is None else environ).get(ENV) or ""
    out: List[str] = []
    for w in raw.replace(" ", "").replace("\t", "").split(","):
        if w and w not in out:
            out.append(w)
    return out


def requested(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The lever names ``MODEL_OPT_LEVERS_OFF`` lists, in order, whitespace ignored, duplicates folded, the ABSENT words (``compile``) left
    out (:func:`requested_absent`); [] when unset or blank."""
    return [n for n in _requested_raw(environ) if n not in ABSENT]


def requested_absent(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The ABSENT words the request names (``compile``), in order; they withhold nothing and are reported ``<name>=none``."""
    return [n for n in _requested_raw(environ) if n in ABSENT]


def absent_token(names: Sequence[str]) -> str:
    """`` compile=none`` for each absent word requested ('' for none)."""
    return "".join(f" {n}={TOKEN_ABSENT}" for n in names)


def token(names: Sequence[str]) -> str:
    """`` withheld=<a,b>`` for the ACTIVE / DRY-RUN / EXIT lines; the empty string when nothing is withheld (the lines stay as they were)."""
    return f" {TOKEN}={','.join(names)}" if names else ""


def big_names() -> Tuple[str, ...]:
    """The big line's memory levers by registry name (big_<lever>), in line order."""
    from . import big as _big
    return tuple(BIG_PREFIX + lv for lv in _big.LEVERS)


def mode_set(res) -> List[str]:
    """Every registry name a request may remove from this resolution: the row's levers (switch + FPF + package) and, under big, the
    line's memory levers."""
    names = list(res.levers)
    if res.row == BIG_ROW:
        names += [n for n in big_names() if n not in names]
    return names


def validate(res, names: Sequence[str]) -> List[str]:
    """``names`` against the resolution: raises WithholdError naming every unknown lever, every lever outside the mode's set, every name
    that cannot be withheld, every broken dependency, and a request that leaves the mode with no lever; returns the names otherwise."""
    from .registry import LEVERS
    names = list(names)
    if not names:
        return []
    if res.row == "off":
        raise WithholdError(f"{ENV}={','.join(names)}: --mode off applies no lever, there is nothing to withhold (unset {ENV} for the stock route)")
    members = mode_set(res)
    problems: List[str] = []
    unknown = [n for n in names if n not in LEVERS]
    fixed = [n for n in names if n in NOT_WITHHOLDABLE]
    outside = [n for n in names if n in LEVERS and n not in NOT_WITHHOLDABLE and n not in members]
    if unknown:
        problems.append(f"{','.join(unknown)}: not a lever of this kit (registry levers: {','.join(LEVERS)})")
    for n in fixed:
        problems.append(f"{n}: {NOT_WITHHOLDABLE[n]}")
    if outside:
        problems.append(f"{','.join(outside)}: not in mode {res.row}'s lever set ({','.join(members)})")
    for group in TOGETHER:
        named = [n for n in group if n in names]
        if named and len(named) != len([n for n in group if n in members]):
            rest = [n for n in group if n in members and n not in names]
            problems.append(f"{','.join(named)}: rides one switch with {','.join(rest)} (RF3_CUDAGRAPH: the graph-safe op rewrites are on iff the sampler graph is) — "
                            f"withhold {','.join(group)} together")
    kept = [m for m in members if m not in names]
    for m in kept:
        need = [r for r in REQUIRES.get(m, ()) if r in names]
        if need:
            problems.append(f"{','.join(need)}: required by {m}, which stays on (withhold {m} with it)")
        anyof = REQUIRES_ANY.get(m)
        if anyof and any(a in members for a in anyof) and not any(a in kept for a in anyof):
            problems.append(f"{','.join(a for a in anyof if a in names)}: {m} stays on and needs one of {','.join(anyof)} (withhold {m} with them)")
    from . import modes as M
    switch_names = [lv for lvs in M.LEVERS_OF_SWITCH.values() for lv in lvs]
    if any(n in switch_names for n in names):                                   # a switch lever leaves with the arm's whole lever step (@L1[.sub]): its sub-step levers cannot ride without it
        riding = [lv for lvs in M.LEVERS_OF_SUB.values() for lv in lvs if lv in members and lv not in names]
        if riding:
            problems.append(f"{','.join(riding)}: rides the FPF arm's lever step, which leaves with {','.join(n for n in names if n in switch_names)} (withhold {','.join(riding)} with it)")
    if not problems and not [m for m in res.levers if m not in names]:
        problems.append(f"the request removes every lever of mode {res.row}: that is --mode off, not an ablation")
    if problems:
        raise WithholdError(f"{ENV} refused — " + "; ".join(problems))
    return names


def _arm_without(arm: Optional[str], components_off: Sequence[str], drop_lever_step: bool, keep_seam: bool, subs_off: Sequence[str] = ()) -> Optional[str]:
    """The FPF arm string with the named components removed (``fast`` trimul -> ``stock``: the word leaves the body) and, when asked,
    without its ``@L`` step (modes.fpf_components' grammar: parts in any order, an empty body = ``stock``). An arm left with no lever
    is ``None`` — the adapter is not imported, the row is arm-less like an X-switches row — unless a package lever that installs through
    the adapter's seam stays (``keep_seam``: modes.KIT_LEVERS_ON_FPF_SEAM), which keeps ``stock`` (the adapter imported, no kernel)."""
    if not arm:
        return arm
    body, _, lstep = arm.partition("@")
    parts = [p for p in body.split("+") if p and p.partition(".")[0] not in components_off]   # a sub-worded component (fast.fast, apb.fast, xmul.eager) leaves with its component
    from . import modes as M
    names_lever = [p for p in parts if M.LEVERS_OF_FPF.get(p.partition(".")[0])]
    if not names_lever:
        if not keep_seam:
            return None
        parts = ["stock"]
    out = "+".join(parts)
    if lstep and not drop_lever_step:
        step, *subs = lstep.split(".")
        out += "@" + ".".join([step] + [s for s in subs if s not in subs_off])   # a withheld sub-step lever (warm) leaves the step; the step itself stays
    return out


def withhold(res, names: Optional[Sequence[str]] = None, environ: Optional[Mapping[str, str]] = None):
    """The resolution with the requested levers removed (a new ``modes.Resolution``; ``res`` unchanged). ``names`` None reads the
    environment. No request: ``res`` itself, untouched. A refused request raises :class:`WithholdError` (a ValueError)."""
    absent = requested_absent(environ) if names is None else [n for n in names if n and n in ABSENT]
    names = requested(environ) if names is None else [n for n in names if n and n not in ABSENT]
    if absent:
        res = replace(res, absent=list(dict.fromkeys(absent)))               # accepted in every mode, withholds nothing: the census says <name>=none
    if not names:
        return res
    names = validate(res, names)
    from . import modes as M
    switches = dict(res.switches)
    switch_off = [sw for sw, lvs in M.LEVERS_OF_SWITCH.items() if any(lv in names for lv in lvs)]
    for sw in switch_off:
        switches[sw] = "0"                                                  # exported 0: rf3.graph_flags reads the lever off (RF3_CUDAGRAPH "0" = eager sampler, no safe-op rewrites; RF3_HOIST "0")
    comp_of = {lv: comp for comp, lvs in M.LEVERS_OF_FPF.items() for lv in lvs}
    components_off = [comp_of[n] for n in names if n in comp_of]
    kit_levers = [lv for lv in res.kit_levers if lv not in names]
    subs_off = [sub for sub, lvs in M.LEVERS_OF_SUB.items() if any(lv in names for lv in lvs)]
    arm = _arm_without(res.fpf_arm, components_off, drop_lever_step=bool(switch_off), keep_seam=any(lv in M.KIT_LEVERS_ON_FPF_SEAM for lv in kit_levers),
                       subs_off=subs_off)
    components, lever_state = M.fpf_components(arm) if arm else ([], None)
    levers = [lv for lv in res.levers if lv not in names]
    levers_off = list(res.levers_off) + [lv for lv in res.levers if lv in names and lv not in res.levers_off]
    notes = list(res.notes) + [f"{ENV}: withheld {','.join(names)}" + (f" (arm {res.fpf_arm} -> {arm})" if arm != res.fpf_arm else "")]
    return replace(res, switches=switches, levers=levers, levers_off=levers_off, notes=notes, fpf_arm=arm, fpf_components=components,
                   fpf_lever_state=lever_state, kit_levers=kit_levers, withheld=list(names))


def without_request(fn: Callable, *args, **kwargs):
    """``fn(*args, **kwargs)`` with the request masked (the mode as it is): the activation report of a REFUSED request still names the
    mode's own row and levers on its NOT ACTIVE line."""
    saved = os.environ.pop(ENV, None)
    try:
        return fn(*args, **kwargs)
    finally:
        if saved is not None:
            os.environ[ENV] = saved


def big_switches(names: Optional[Sequence[str]] = None, environ: Optional[Mapping[str, str]] = None) -> Dict[str, bool]:
    """``{<big lever>: False}`` for every withheld ``big_<lever>`` name — the big line's own selection switches (big.switches_for
    merges them: the lever is ``off_by_flag`` on the BIG line, ``state=off reason=flag`` on its LEVER line). Validation is
    :func:`withhold`'s (modes.resolve ran first on every route that composes the line)."""
    names = requested(environ) if names is None else list(names)
    return {n[len(BIG_PREFIX):]: False for n in names if n.startswith(BIG_PREFIX)}
