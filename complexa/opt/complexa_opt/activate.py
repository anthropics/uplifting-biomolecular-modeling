"""Activation inside upstream's generation process (``python -m proteinfoundation.generate``): ``enable(mode)``, called by the autoload
finder right after ``proteinfoundation.proteina`` has been imported and before the checkpoint loads.

``enable`` resolves the mode's lever set (``modes.KIT_MODES``: the same set on every card), installs every lever of it (``levers.install``:
class / function patches of upstream's pinned code; a lever that cannot be installed raises and the whole mode is refused by name — one
``[complexa-opt] LEVER-ERROR lever=<name> reason=<…>`` line, one ``NOT ACTIVE`` line, ``ActivationError``, the finder exits 3; never a subset
under the mode's name), prints ONE activation line::

    [complexa-opt] ACTIVE mode=<m> tier=<exact|tolerance> levers=<a,b,…> proteinfoundation=<version> torch=<version> gpu=<name>(cc<X.Y>) gpu_class=<H100|A100|other> pid=<pid> trigger=<module>

(``gpu_class=other`` names a card outside the two the configs name (H100, A100): the levers engage all the same — the card found is named on the
line, never a reason to disengage), writes the process's activation record ``<COMPLEXA_OPT_RECORD>/kit_<pid>.json``
(state ``armed``), and registers the exit report: one ``LEVER`` line per lever of the set with its census counters (``levers.status``:
``state=on`` when the lever engaged during the run; ``state=skipped reason=padded_batches`` when pair_assembly's declared input gate took every
call — binder lengths differ within the batches (a binder_length range), upstream's padded assembly, same bytes: named, exit unaffected;
``state=skipped reason=never_engaged`` when its site was never reached — the kit route reads that as a partial activation and exits 3), one
``TALLY`` line::

    [complexa-opt] TALLY pid=<pid> mode=<m> state=<complete|gated|partial> predict_steps=<n> forwards=<n> levers_on=<a,b,…|none> levers_skipped=<…|none> peak_alloc_gib=<x> levers_gated=<…|none>

(every lever of the set in exactly one of the three lists; torch's peak allocation is a diagnostic, not a device-level peak), and the record
rewritten final (``gated`` and ``partial`` name the two non-``on`` lists). Without ``COMPLEXA_OPT_RECORD`` (``COMPLEXA_OPT=<mode> complexa
generate …`` by hand) the lines are printed and no record is written.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

from opt_core import report as core_report

from . import TAG, ActivationError, __version__, levers as _levers, modes

PREFIX = core_report.prefix(TAG)
KNOWN_CLASSES = {"9.0": "H100", "8.0": "A100"}                 # compute capability -> the card classes configs/*.env name; any other card reads gpu_class=other
_STATE: dict = {}


def _gpu() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"name": None, "cc": None, "label": "none", "class": "none"}
    i = torch.cuda.current_device()
    name = torch.cuda.get_device_name(i)
    maj, mnr = torch.cuda.get_device_capability(i)
    cc = f"{maj}.{mnr}"
    return {"name": name, "cc": cc, "label": f"{name.replace(' ', '_')}(cc{cc})", "class": KNOWN_CLASSES.get(cc, "other")}


def _upstream_version() -> str:
    try:
        from importlib.metadata import version
        return version("proteinfoundation")
    except Exception:                                              # a checkout without metadata: named, not fatal
        return "unknown"


def _record_path() -> Optional[str]:
    d = os.environ.get(modes.ENV_RECORD)
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"kit_{os.getpid()}.json")


def _write_record(final: bool) -> None:
    path = _STATE.get("record_path")
    if not path:
        return
    lev = {}
    census = _levers.census()
    for n in _STATE["levers"]:
        state, reason = _levers.status(n) if final else (("on", None) if _levers.engaged(n) else ("armed", None))
        lev[n] = {"state": state, "tier": _levers.LEVERS[n].tier, "counters": census.get(n, {})}
        if reason is not None:
            lev[n]["reason"] = reason
    doc = {"schema": "complexa_opt.kit_record/1", "package_version": __version__, "pid": os.getpid(), "mode": _STATE["mode"], "tier": _STATE["tier"],
           "levers_requested": list(_STATE["levers"]), "levers": lev, "gpu": _STATE["gpu"], "proteinfoundation": _STATE["upstream"],
           "torch": _STATE["torch"], "trigger": _STATE["trigger"], "t_enable": _STATE["t_enable"], "final": final}
    doc.update(_levers.counts())
    if final:
        doc["t_exit"] = round(time.time(), 3)
        doc["peak_alloc_gib"] = _peak_gib()
        doc["gated"] = sorted(n for n in lev if _levers.gated(n))                                   # idle by their declared input gate (levers.GATES): named, exit unaffected
        doc["partial"] = sorted(n for n, v in lev.items() if v["state"] != "on" and n not in doc["gated"])   # never engaged: the mode is partial (the kit route exits 3)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _peak_gib() -> Optional[float]:
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if torch.cuda.is_available() else None
    except Exception:
        return None


def active_line() -> str:
    s = _STATE
    return core_report.line(PREFIX, "ACTIVE", ("mode", s["mode"]), ("tier", s["tier"]), ("levers", ",".join(s["levers"])), ("proteinfoundation", s["upstream"]),
                            ("torch", s["torch"]), ("gpu", s["gpu"]["label"]), ("gpu_class", s["gpu"]["class"]), ("pid", os.getpid()), ("trigger", s["trigger"]))


def lever_lines() -> list:
    out = []
    census = _levers.census()
    for n in _STATE.get("levers", ()):
        c = census.get(n, {})
        pairs = [(k, c[k]) for k in sorted(c)]
        state, reason = _levers.status(n)
        out.append(core_report.lever_line(TAG, n, state, *pairs, reason=reason, impl="complexa_opt.levers", origin="kit", tier=_levers.LEVERS[n].tier))
    return out


def buckets() -> dict:
    """``{on, gated, skipped}``: every lever of the set in exactly one — engaged; idle by its declared gate (``levers.GATES``); never engaged."""
    names = _STATE.get("levers", ())
    on = [n for n in names if _levers.engaged(n)]
    gated = [n for n in names if n not in on and _levers.gated(n)]
    return {"on": on, "gated": gated, "skipped": [n for n in names if n not in on and n not in gated]}


def tally_state(b: dict) -> str:
    """``partial`` when a lever never engaged; ``gated`` when none did that but a declared gate idled one; else ``complete``."""
    return "partial" if b["skipped"] else ("gated" if b["gated"] else "complete")


def tally_line() -> str:
    c = _levers.counts()
    b = buckets()
    return core_report.line(PREFIX, "TALLY", ("pid", os.getpid()), ("mode", _STATE.get("mode")), ("state", tally_state(b)),
                            ("predict_steps", c["predict_steps"]), ("forwards", c["forwards"]), ("levers_on", ",".join(b["on"]) or "none"),
                            ("levers_skipped", ",".join(b["skipped"]) or "none"), ("peak_alloc_gib", _peak_gib()), ("levers_gated", ",".join(b["gated"]) or "none"))


def _at_exit() -> None:
    try:
        for ln in lever_lines():
            core_report.emit(ln)
        core_report.emit(tally_line())
        _write_record(final=True)
    except Exception as e:                                         # the exit report must never turn a finished run into a traceback
        sys.stderr.write(f"{PREFIX} NOTE exit_report=failed: {type(e).__name__}: {e}\n")


def enable(mode: Optional[str] = None, *, strict: bool = True, trigger: Optional[str] = None) -> dict:
    """Install the levers of ``mode`` in this process and print the ACTIVE line. ``ActivationError`` (after ONE ``NOT ACTIVE`` line) for a
    word that is not a kit mode or a lever that cannot be installed. ``strict`` is the finder's contract word (a refusal always raises here)."""
    if _STATE.get("mode"):
        return dict(_STATE)                                        # a second trigger in the same process: already active
    try:
        m = modes.resolve(mode, environ={})
    except modes.ModeError as e:
        core_report.emit(core_report.not_active_line(TAG, str(e), head=f"mode={mode}"))
        raise ActivationError(str(e)) from None
    names = modes.levers_of(m)
    if not names:
        words = f"mode {m} is the stock route: it activates nothing inside upstream's process (unset {modes.ENV_MODE} or run the kit's `design --mode {m}`)"
        core_report.emit(core_report.not_active_line(TAG, words, head=f"mode={m}"))
        raise ActivationError(words)
    import torch
    try:
        _levers.install(names)
    except _levers.LeverError as e:
        words = f"mode {m} refused: {e} — a mode is all of its levers or none; select {modes.ESCAPE} for stock"
        core_report.emit(core_report.not_active_line(TAG, words, head=f"mode={m}"))
        raise ActivationError(words) from None
    _STATE.update(mode=m, tier=modes.TIERS[m], levers=tuple(names), gpu=_gpu(), upstream=_upstream_version(), torch=torch.__version__,
                  trigger=trigger or "none", t_enable=round(time.time(), 3), record_path=_record_path(), active=True)
    core_report.emit(active_line())
    _write_record(final=False)
    import atexit
    atexit.register(_at_exit)
    return dict(_STATE)
