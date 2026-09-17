"""Card classes and lever support: the ONE registry of which compute architectures (``sm`` classes) each lever runs on, the reader
of the box's ``sm`` class (torch when already imported, else nvidia-smi — never imported here), the device's actual memory,
and the words an activation line carries when a lever is off because the card cannot run it.

Contract.
  * An ``sm`` class is ``sm<major><minor>`` of the CUDA compute capability: ``sm80`` (A100), ``sm90`` (H100, H200), ``sm100`` (B200),
    ``sm103`` (B300). Detection reads the DEVICE (compute capability, total memory); card names are labels for tables and reports only —
    no gate here or in a kit derives a memory size or a capability from a name (:data:`CARD_SM` maps a card label to its class for
    tables; :func:`device_memory` reads what the card has).
  * A lever's support is DECLARED once (:func:`declare`), by whoever owns the lever — a family module for a shared strategy, a kit's
    registry module for a kit-local lever — as: the ``sm`` classes it is TESTED on (an equality or band test record exists), an
    optional floor ``min_sm`` (below it the mechanism cannot run: an FP8 or TMA kernel below ``sm90``), and named exclusions
    ``{sm: reason_word}`` (a class at or above the floor where the lever is known not to run — e.g. a fused kernel whose tensor-memory
    request exceeds the class's limit: ``{"sm100": "tmem_528_gt_512"}``). One producer per lever: re-declaring a lever with different
    content raises unless ``replace=True``.
  * :func:`supports` turns (lever, sm) into ONE verdict word: ``supported`` (tested on this class) · ``uncertified`` (at or above the
    floor, not excluded, no test record on this class: it runs, the test record is owed) · ``unsupported:<word>`` (excluded, or
    ``below_<min_sm>``) · ``undeclared`` (nobody declared the lever) · ``no_gpu`` (no class could be read). Nothing is guessed either way.
  * :func:`lever_state` turns the verdict into the activation line's words for :func:`opt_core.report.lever_line`: ``state=on`` for
    supported; ``state=on card_support=uncertified:<sm>`` / ``card_support=undeclared`` (named, so a census can count them: a class no
    engine was measured on is where the lever engages and says so — a measurement is a word, never a state); ``state=off
    reason=unsupported_card:<sm> card_reason=<word>`` for unsupported (the mechanism's floor or an exclusion: it cannot run there). A mode
    whose REQUIRED lever cannot run calls :func:`require`, which raises :class:`ArchRefused` (a :class:`~opt_core.modes.ModeError` carrying
    ``cannot_run = True``) naming lever, class and reason — the kit refuses the mode with that sentence, never runs a subset of it.
  * :func:`device_memory` reads the device's total and free bytes (torch ``mem_get_info`` when torch is imported in this process, else
    nvidia-smi's total): size gates budget against what the card HAS.
    Allocator-aware torch budgeting (reserved-but-unallocated slack counted as free) is :func:`opt_core.mem.budget.device_free_bytes`.
  * :func:`card_table` renders ``{lever: {sm: word}}`` over the classes of :data:`SM_CLASSES` — the CARD column of a kit's applicability
    table is a projection of it.

Standard library at import; Python 3.8 syntax.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from . import modes, report

__all__ = ["SM_CLASSES", "CARD_SM", "Support", "Verdict", "ArchRefused", "sm_of", "sm_key", "declare", "declared", "supports",
           "lever_state", "require", "card_table", "card_line", "current_sm", "device_memory",
           "REASON_UNSUPPORTED"]

# ------------------------------------------------------------------------------------------------------------------ classes and cards

SM_CLASSES = {                      # sm class -> architecture word and the card labels of that class (labels only; detection never uses them)
    "sm80": {"arch": "ampere", "cards": ("A100",)},
    "sm90": {"arch": "hopper", "cards": ("H100", "H200")},
    "sm100": {"arch": "blackwell", "cards": ("B200",)},
    "sm103": {"arch": "blackwell_ultra", "cards": ("B300",)},
}
CARD_SM = {card: sm for sm, row in SM_CLASSES.items() for card in row["cards"]}
REASON_UNSUPPORTED = "unsupported_card"      # the reason= word of an off line: reason=unsupported_card:<sm>


def sm_of(x) -> Optional[str]:
    """``sm<major><minor>`` from any of: ``"9.0"`` · ``(9, 0)`` · ``"sm90"`` / ``"sm_90"`` / ``"SM90a"`` · a probe dict with ``sm`` or ``cc``
    (:func:`opt_core.gates.torch_gpu_probe` / :func:`~opt_core.gates.nvidia_smi_probe`) · an object with ``major``/``minor`` (torch device
    properties) or a device's ``compute_capability``. None when nothing names a capability."""
    if x is None:
        return None
    if isinstance(x, Mapping):
        return sm_of(x.get("sm") or x.get("cc"))
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return "sm%d%d" % (int(x[0]), int(x[1]))
    if hasattr(x, "major") and hasattr(x, "minor"):
        return "sm%d%d" % (int(x.major), int(x.minor))
    if hasattr(x, "compute_capability"):
        return sm_of(getattr(x, "compute_capability"))
    s = str(x).strip().lower().replace("_", "")
    if s.startswith("sm"):
        digits = "".join(c for c in s[2:] if c.isdigit())
        return ("sm" + digits) if len(digits) >= 2 else None
    if "." in s:
        major, _, minor = s.partition(".")
        if major.isdigit() and minor[:1].isdigit():
            return "sm%d%d" % (int(major), int(minor[0]))
    return None


def sm_key(sm: str) -> Tuple[int, int]:
    """(major, minor) of an ``sm`` class for ordering: the last digit is the minor (``sm90`` -> (9, 0), ``sm103`` -> (10, 3))."""
    s = sm_of(sm)
    if s is None:
        raise ValueError("sm_key: %r does not name an sm class" % (sm,))
    digits = s[2:]
    return int(digits[:-1]), int(digits[-1])


# ------------------------------------------------------------------------------------------------------------------ the registry


@dataclass(frozen=True)
class Support:
    """One lever's declaration: tested classes, the floor, the named exclusions, and a free note (a citation, never parsed)."""

    lever: str
    certified: Tuple[str, ...] = ()
    min_sm: Optional[str] = None
    exclude: Tuple[Tuple[str, str], ...] = ()          # ((sm, reason_word), ...) sorted by class
    note: Optional[str] = None

    def excluded(self) -> Dict[str, str]:
        return dict(self.exclude)


@dataclass(frozen=True)
class Verdict:
    """``ok`` = the lever may be switched on on this class; ``word`` = the one census word (see the module contract)."""

    ok: bool
    word: str
    sm: Optional[str]
    certified: bool = False


class ArchRefused(modes.ModeError):
    """A mode's REQUIRED lever cannot run on this card: the mode refuses by name (never runs silently without the lever). A cannot-run
    event (``opt_core.gates.is_cannot_run``)."""

    cannot_run = True


_REGISTRY: Dict[str, Support] = {}


def _word_ok(word: str) -> str:
    w = str(word)
    if not w or any(c.isspace() for c in w) or "=" in w:
        raise ValueError("arch: reason word %r must be one blank-free token without '='" % (word,))
    return w


def declare(lever: str, *, certified: Iterable = (), min_sm: Optional[str] = None, exclude: Optional[Mapping] = None,
            note: Optional[str] = None, replace: bool = False) -> Support:
    """Declare ``lever``'s support (one producer per lever). ``certified``: classes with a test record; ``min_sm``: the floor;
    ``exclude``: ``{sm: reason_word}`` for classes the lever is known not to run on. The same content declared twice is a no-op; different
    content raises ``ValueError`` unless ``replace``."""
    lever = str(lever)
    if not lever or any(c.isspace() for c in lever):
        raise ValueError("arch.declare: lever id %r must be one blank-free token" % (lever,))
    cert = []
    for c in certified:
        s = sm_of(c)
        if s is None:
            raise ValueError("arch.declare(%s): certified entry %r is not an sm class" % (lever, c))
        cert.append(s)
    floor = None
    if min_sm is not None:
        floor = sm_of(min_sm)
        if floor is None:
            raise ValueError("arch.declare(%s): min_sm %r is not an sm class" % (lever, min_sm))
    excl = []
    for k, v in dict(exclude or {}).items():
        s = sm_of(k)
        if s is None:
            raise ValueError("arch.declare(%s): excluded entry %r is not an sm class" % (lever, k))
        excl.append((s, _word_ok(v)))
    for s in cert:
        if floor is not None and sm_key(s) < sm_key(floor):
            raise ValueError("arch.declare(%s): certified class %s is below min_sm %s" % (lever, s, floor))
        if s in dict(excl):
            raise ValueError("arch.declare(%s): class %s is both certified and excluded" % (lever, s))
    sup = Support(lever=lever, certified=tuple(sorted(set(cert), key=sm_key)), min_sm=floor,
                  exclude=tuple(sorted(excl, key=lambda kv: sm_key(kv[0]))), note=note)
    have = _REGISTRY.get(lever)
    if have is not None and have != sup and not replace:
        raise ValueError("arch.declare(%s): already declared with different content (%r); one producer per lever — pass replace=True "
                         "to re-declare deliberately" % (lever, have))
    _REGISTRY[lever] = sup
    return sup


def declared(lever: Optional[str] = None):
    """The :class:`Support` of ``lever`` (None when undeclared), or the whole registry as ``{lever: Support}`` when no lever is named."""
    if lever is None:
        return dict(_REGISTRY)
    return _REGISTRY.get(str(lever))


def _forget(lever: str) -> None:                          # tests only: remove a declaration
    _REGISTRY.pop(str(lever), None)


def supports(lever: str, sm) -> Verdict:
    """The verdict for ``lever`` on class ``sm`` (any form :func:`sm_of` reads; None / unreadable -> ``no_gpu``)."""
    s = sm_of(sm)
    sup = _REGISTRY.get(str(lever))
    if s is None:
        return Verdict(ok=False, word="no_gpu", sm=None)
    if sup is None:
        return Verdict(ok=True, word="undeclared", sm=s)
    excl = sup.excluded()
    if s in excl:
        return Verdict(ok=False, word="unsupported:" + excl[s], sm=s)
    if sup.min_sm is not None and sm_key(s) < sm_key(sup.min_sm):
        return Verdict(ok=False, word="unsupported:below_" + sup.min_sm, sm=s)
    if s in sup.certified:
        return Verdict(ok=True, word="supported", sm=s, certified=True)
    return Verdict(ok=True, word="uncertified", sm=s)


def lever_state(lever: str, sm, *, strict: bool = False) -> dict:
    """The activation line's words for ``lever`` on ``sm``: ``{"state", "reason", "evidence", "verdict", "words"}`` to pass to
    ``report.lever_line(tag, name, d["state"], reason=d["reason"], impl=..., origin=..., **d["evidence"])``.
    supported -> on; uncertified / undeclared -> on with ``card_support=<word>`` (also listed under ``words``: the named uncertainty the
    lever engages under); unsupported / no_gpu -> off, ``reason=unsupported_card:<sm|none>``, ``card_reason=<word>``. ``strict`` is
    accepted from callers that thread it and changes nothing: a measurement is a word, never a state."""
    del strict
    v = supports(lever, sm)
    tag = v.sm or "none"
    if v.word == "supported":
        return {"state": "on", "reason": None, "evidence": {"sm": tag}, "verdict": v, "words": []}
    if v.ok:                                              # uncertified / undeclared: engages, named
        cs = v.word + (":" + tag if v.word == "uncertified" else "")
        return {"state": "on", "reason": None, "evidence": {"sm": tag, "card_support": cs}, "verdict": v, "words": ["card_support=" + cs]}
    word = v.word.split(":", 1)[1] if v.word.startswith("unsupported:") else v.word
    return {"state": "off", "reason": REASON_UNSUPPORTED + ":" + tag, "evidence": {"card_reason": word}, "verdict": v, "words": []}


def require(lever: str, sm, *, mode: str, strict: bool = False) -> Verdict:
    """A mode's REQUIRED lever: return the verdict when the lever can run (certified or not — an uncertified class is a word on the
    line), else raise :class:`ArchRefused` with the sentence the kit refuses the mode with."""
    st = lever_state(lever, sm, strict=strict)
    if st["state"] == "on":
        return st["verdict"]
    v = st["verdict"]
    raise ArchRefused("mode %r needs lever %r, which cannot run on this card (%s: %s)" % (mode, str(lever), v.sm or "no GPU class read", v.word))


def card_table(levers: Optional[Iterable[str]] = None, sms: Optional[Sequence] = None) -> Dict[str, Dict[str, str]]:
    """``{lever: {sm: word}}`` over ``sms`` (default: every class of :data:`SM_CLASSES`) for ``levers`` (default: every declared lever)."""
    classes = [sm_of(s) for s in (sms if sms is not None else sorted(SM_CLASSES, key=sm_key))]
    names = list(levers) if levers is not None else sorted(_REGISTRY)
    return {name: {s: supports(name, s).word for s in classes} for name in names}


# ------------------------------------------------------------------------------------------------------------------ readers: class and memory of the box


def current_sm(index: int = 0, *, probe: Optional[Mapping] = None) -> Tuple[Optional[str], dict]:
    """(sm class or None, the probe dict) of GPU ``index``. A given ``probe`` (a gates probe dict) is used as is; else torch's device
    properties when torch is ALREADY imported in this process (this reader never imports torch by itself), else nvidia-smi."""
    from . import gates
    if probe is None:
        torch = sys.modules.get("torch")
        if torch is not None and getattr(getattr(torch, "cuda", None), "is_available", lambda: False)():
            probe = gates.torch_gpu_probe(index)
        else:
            probe = gates.nvidia_smi_probe(index=index)
    return sm_of(probe), dict(probe)


def device_memory(index: int = 0) -> dict:
    """``{"total_bytes", "free_bytes", "source"}`` of GPU ``index``: torch ``cuda.mem_get_info`` when torch is already imported and CUDA is
    available (free = the driver's figure), else nvidia-smi's total with ``free_bytes`` None; no GPU -> both None with the source naming why."""
    torch = sys.modules.get("torch")
    if torch is not None and getattr(getattr(torch, "cuda", None), "is_available", lambda: False)():
        free, total = torch.cuda.mem_get_info(index)
        return {"total_bytes": int(total), "free_bytes": int(free), "source": "torch"}
    from . import gates
    p = gates.nvidia_smi_probe(index=index)
    mib = p.get("memory_mib")
    return {"total_bytes": (int(mib) * 1024 * 1024) if mib is not None else None, "free_bytes": None, "source": p.get("probe")}


def card_line(tag: str, name: str, sm, *, impl: str, origin: str, strategy: Optional[str] = None, strict: bool = False, **fields) -> str:
    """The lever's activation line on this card through :func:`opt_core.report.lever_line` with :func:`lever_state`'s words (on or
    off). The declaration is looked up under the line's ``name``; when that is undeclared and a canonical ``strategy`` id is given, under
    the strategy id (a shared strategy is declared once under its canonical id, whatever a kit's line name)."""
    key = name if (declared(name) is not None or strategy is None) else strategy
    st = lever_state(key, sm, strict=strict)
    ev = dict(st["evidence"])
    ev.update(fields)
    return report.lever_line(tag, name, st["state"], reason=st["reason"], impl=impl, origin=origin, strategy=strategy, **ev)
