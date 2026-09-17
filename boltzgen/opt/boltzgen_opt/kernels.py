"""The kit modes' policy over the accelerator census (census.py — the per-process monitor every model process carries): what a route
EXPECTS of upstream's two cuEquivariance accelerators, what a kit mode REFUSES, and the caller's ACCOUNT of a run's census lines.

* ``upstream_resolution`` reads upstream's own gate from the configure log (``Using kernels: <bool> [device capability: (M, m)]``,
  printed by upstream's ``cli/boltzgen.py`` at configure) and ``expectation`` crosses it with the mode table (modes.kernels_of: the accelerators a mode routes through
  unchanged, those a lever of the mode replaces): ``{"expect": on|unknown|off:<reason>, "words": {accel: engaged|off-by-route:<reason>}}``.
* ``refuse_if_absent`` is the kit modes' gate in an armed KIT process (``_autoload.arm_census``): a mode that routes through the accelerators
  cannot activate where the library is absent — the NOT ACTIVE line naming the escape, exit 3 before the step. Upstream's own processes
  (mode ``off``) never import this module: an absent library there is upstream's to meet and the census line says ``absent``.
* ``verdict`` / ``census_of`` are the caller's account of a run's KERNELS lines against the expectation (design.py records it in
  ``opt_manifest.json``; report-only — nothing exits on it).
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from . import census
from .census import ACCELERATORS, KERNEL_FREE_STEPS, LINE_VERB

UPSTREAM_RESOLUTION = re.compile(r"^Using kernels: (True|False|None) \[device capability: \((\d+), (\d+)\)\]\s*$")   # upstream's configure-time line (cli/boltzgen.py)
REFUSED_KINDS: Tuple[str, ...] = ("absent", "fallback")   # a word of these kinds where the route expects the accelerator is a finding the account (verdict) names


# --------------------------------------------------------------------------------------------------------- upstream's own gate
def upstream_resolution(configure_log: str) -> dict:
    """Upstream's ``Using kernels: <bool> [device capability: (M, m)]`` line from the configure log (upstream's cli/boltzgen.py):
    ``{"use_kernels": bool|None, "cc": (M, m)|None, "line": str|None}`` — ``line`` None when upstream printed none."""
    res = {"use_kernels": None, "cc": None, "line": None}
    try:
        with open(configure_log, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                m = UPSTREAM_RESOLUTION.match(ln.strip())
                if m:
                    res = {"use_kernels": {"True": True, "False": False, "None": None}[m.group(1)], "cc": (int(m.group(2)), int(m.group(3))), "line": ln.strip()}
    except OSError:
        pass
    return res


def expectation(mode: str, resolution: dict, kernels_of=None) -> dict:
    """What the route expects per accelerator, from the mode table (modes.kernels_of: KIT_MODES[mode].kernels / kernels_replaced; ``off``
    routes through every upstream accelerator) and upstream's own resolution:
    ``{"expect": "on"|"unknown"|"off:<reason>", "words": {accel: "engaged"|"off-by-route:<reason>"}}``. ``unknown``: upstream printed no
    resolution line at configure. ``off:upstream(cc=M.m<8)``: upstream's ``auto`` switched the kernels off on this card;
    ``off:use_kernels=false``: the caller's own ``--use_kernels false`` — upstream's switch, passed through as upstream takes it (the model
    runs its torch paths; the census then expects no library call)."""
    from . import modes as _modes
    routed, replaced = _modes.kernels_of(mode) if kernels_of is None else kernels_of(mode)
    words: Dict[str, str] = {}
    if resolution.get("line") is None:
        return {"expect": "unknown", "words": {a: "engaged" for a in routed}}
    cc = resolution.get("cc") or (0, 0)
    if bool(resolution.get("use_kernels")):
        expect = "on"
    elif cc[0] >= 8:
        expect = "off:use_kernels=false"
    else:
        expect = f"off:upstream(cc={cc[0]}.{cc[1]}<8)"
    for a in ACCELERATORS:
        if a in replaced:
            words[a] = f"off-by-route:{replaced[a]}"
        elif a in routed:
            words[a] = "engaged" if expect == "on" else f"off-by-route:use_kernels=False({expect[4:]})"
        else:
            raise ValueError(f"mode {mode!r} neither routes nor replaces accelerator {a!r} (modes.KIT_MODES kernels / kernels_replaced)")
    return {"expect": expect, "words": words}


# ------------------------------------------------------------------------------------------------------------- the kit's refusal
def refuse_if_absent(p: Optional[Dict[str, str]] = None) -> Optional[str]:
    """After an early/eager ``arm`` in a KIT mode's process: the NOT ACTIVE line when the mode routes through the accelerators (payload
    ``expect=on``) and this process cannot provide one — the mode cannot activate here; the caller prints it and exits ``EXIT_NOT_ACTIVE``
    before anything of the step runs (_autoload.arm_census). ``None`` when nothing is to refuse, and always ``None`` for upstream's own
    processes (``mode=off``: an absent library is upstream's to meet, never the kit's refusal)."""
    p = p or census.armed() or {}
    miss = census.absent()
    if p.get("mode", "off") == "off" or p.get("expect", "on") != "on" or not miss:
        return None
    step = p.get("step") or os.environ.get("BOLTZGEN_PIPELINE_STEP") or "?"
    words = " ".join(f"{a}=absent:{census._clean(why)}" for a, why in miss.items())
    from .report import not_active_line
    return not_active_line(f"mode {p.get('mode')} routes through the cuEquivariance kernels and this process cannot provide them "
                           f"(step={step} {words}); exit 3 before the step — `--use_kernels false` (upstream's switch) runs the torch paths, "
                           f"`--mode off` runs upstream alone")


# ----------------------------------------------------------------------------------------------------------- the caller's account
def census_of(lines: List[str], exp: dict, expected_lines: int, where: str = "") -> dict:
    """A run's kernels record from its log lines (``opt_manifest.json`` ``kernels_census``): the account ``verdict`` builds over the KERNELS
    lines, the PEAK lines beside it (``peak``)."""
    out = verdict(census.parse_lines(lines), exp, expected_lines, where=where)
    out["peak"] = census.parse_peak_lines(lines)
    return out


def verdict(parsed: List[dict], exp: dict, expected_lines: int, where: str = "") -> dict:
    """The account of a run's lines: ``{"ok", "findings": [str], "lines": [raw], "words": {accel: word of the last line}, "expected"}``.
    Findings: fewer KERNELS lines than model processes the run launched (``expected_lines``); a word whose kind is not the expected kind
    (``engaged`` expected and ``absent`` / ``fallback`` / ``off-by-route`` found, or ``off-by-route`` expected and anything else found).
    Report-only: the record names what the words show; nothing exits on it."""
    findings: List[str] = []
    if len(parsed) < expected_lines:
        findings.append(f"{len(parsed)} {LINE_VERB} line(s) where {expected_lines} model process(es) ran{where}: a process printed no census (the library was never armed there)")
    for rec in parsed:
        for a in ACCELERATORS:
            want = exp["words"][a]
            want_kind = want.split(":", 1)[0]
            got, got_kind = rec["words"][a], rec["kinds"][a]
            if got_kind == "n/a-upstream" and want_kind == "engaged" and rec["step"] in KERNEL_FREE_STEPS:
                continue                                                  # a step whose network has no call site (KERNEL_FREE_STEPS): nothing to engage there
            if got_kind != want_kind:
                findings.append(f"route={rec['route']} step={rec['step']} {a}={got} (expected {want})")
    words = {a: parsed[-1]["words"][a] for a in ACCELERATORS} if parsed else {}
    return {"ok": not findings, "findings": findings, "lines": [r["raw"] for r in parsed], "words": words, "expected": dict(exp["words"])}
