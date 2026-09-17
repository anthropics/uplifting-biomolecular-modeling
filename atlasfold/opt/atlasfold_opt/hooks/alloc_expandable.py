"""Lever alloc_expandable — torch's CUDA caching allocator with EXPANDABLE SEGMENTS for the kit process (opt_core strategy
F7.expandable_segments; the one writer of the setting is ``opt_core.mem.torch_alloc.write_conf``).  Not a numerics lever: no tensor, kernel
or draw changes — the row stays bitwise (`--mode exact --det 1` == `--mode off --det 1`); what changes is the RESERVED peak: the allocator
grows and shrinks mapped segments instead of caching fixed blocks it cannot reuse across the trunk / sampler / confidence phases.

Mechanism: ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.  torch reads the variable at its first CUDA allocation; this kit activates
in-process AFTER its device probe has initialised CUDA, so the setting is handed to the live allocator through torch's runtime surface
(``torch.cuda.memory._set_allocator_settings``, via ``opt_core.mem.torch_alloc.write_conf(conf, via="runtime_api")``) AND exported into
``os.environ`` for any worker the process forks (``--gpu-ids``); before CUDA is initialised the export alone suffices (``via=env``).
An operator's own ``PYTORCH_CUDA_ALLOC_CONF`` is never overwritten: one that already names ``expandable_segments:True`` is reported
``via=present``; any other value leaves the stock policy in place, the lever prints ``state=skipped reason=user_conf:<value>`` (installed
and inert — never a partial activation).  Beside a CUDA-graph lever in the row (fast's ``denoiser_graph``) the lever steps aside BY NAME
(``graph_lever_in_row:denoiser_graph``; the core does not compose its allocator policy with graph capture by default — ``AFO_ALLOC_EXPANDABLE=graphs``
composes them on request): exact and big carry the policy, fast prints why it does not.  ``AFO_ALLOC_EXPANDABLE=0`` (printed
``state=skipped reason=AFO_ALLOC_EXPANDABLE=0``) or ``MODEL_OPT_LEVERS_OFF=alloc_expandable`` keep the stock allocator policy.
ORDER: with the row's report levers (no lever allocates device memory at install; the model's first allocation comes after activation).
LEVER line: name=F7.expandable_segments impl=torch.cuda.memory[expandable_segments:True] origin=core via=<runtime_api|env|present>
conf=<PYTORCH_CUDA_ALLOC_CONF> effective=<true|false|pending> source=<snapshot|no-cuda|…> (the allocator's own record, opt_core.mem.torch_alloc.effective).
Class: exact (allocator policy)."""
import os

from . import Installed

LEVER = "alloc_expandable"
NAME = "F7.expandable_segments"
SWITCH_ENV = "AFO_ALLOC_EXPANDABLE"
CONF_ENV = "PYTORCH_CUDA_ALLOC_CONF"
WANT = "expandable_segments:True"


def requested() -> bool:
    """AFO_ALLOC_EXPANDABLE: unset / anything but 0|off|false -> on."""
    return (os.environ.get(SWITCH_ENV) or "1").strip().lower() not in ("0", "off", "false", "no")


def user_conf():
    """(state, value): 'absent' | 'present' (already names expandable_segments:True) | 'other' (names another configuration)."""
    have = (os.environ.get(CONF_ENV) or "").strip()
    if not have:
        return "absent", have
    parts = [p.strip() for p in have.split(",") if p.strip()]
    return ("present" if WANT in parts else "other"), have


GRAPH_LEVERS = ("denoiser_graph",)                                    # CUDA-graph levers: the core's allocator policy is not composed with graph capture by default (opt_core.mem.torch_alloc.export: graph private pools are not returned under it) — beside one, this lever steps aside BY NAME unless AFO_ALLOC_EXPANDABLE=graphs


def graph_lever_in_row(mode: str):
    """The first CUDA-graph lever riding this process's row (the mode's row less MODEL_OPT_LEVERS_OFF), or None."""
    try:
        from ..modes import MODES
        row = list(MODES.get(mode, []))
    except Exception:  # noqa: BLE001
        return None
    off = {w.strip() for w in (os.environ.get("MODEL_OPT_LEVERS_OFF") or "").split(",") if w.strip()}
    for g in GRAPH_LEVERS:
        if g in row and g not in off:
            return g
    return None


def _skipped(tag: str, reason: str) -> Installed:
    """Installed-and-inert: the stock allocator policy stays, the LEVER line says why (state=skipped reason=<why>) — never a partial activation."""
    def line():
        from opt_core import report as R
        return R.lever_line(tag, NAME, "skipped", reason=reason, impl=f"torch.cuda.memory[{WANT}]", origin="core", lever=LEVER)
    return Installed(LEVER, True, lines=[line], facts={"via": "none", "reason": reason})


def graph_gate_exceeded(inputs: dict):
    """``exceeded:<min_bucket>><gate>`` when EVERY record of the run pads above the CUDA-graph lever's token gate (denoiser_graph steps aside
    `above_gate` for all of them: no capture can happen in this process, so its allocator constraint does not apply and expandable segments
    engage as in the rows without the graph lever); None when some record can capture (the graph lever keeps the allocator: this lever steps
    aside by name) or when the run's inputs are unknown at start-up (no ``--input-fasta`` hint)."""
    try:
        from . import denoiser_graph as DG
        gate = int(DG.max_tokens())
        mb = int((inputs or {}).get("min_bucket") or 0)
    except Exception:  # noqa: BLE001
        return None
    if mb and mb > gate:
        return f"exceeded:{mb}>{gate}"
    return None


def install(mode: str, tag: str, ctx: dict) -> Installed:
    switch = (os.environ.get(SWITCH_ENV) or "1").strip().lower()
    if not requested():
        return _skipped(tag, f"{SWITCH_ENV}=0")
    state, have = user_conf()
    if state == "other":
        return _skipped(tag, f"user_conf:{have.replace(' ', '')[:60]}")
    g = graph_lever_in_row(mode)
    graph_gate = None
    if g is not None and switch != "graphs":
        graph_gate = graph_gate_exceeded((ctx or {}).get("inputs") or {})
        if graph_gate is None:
            return _skipped(tag, f"graph_lever_in_row:{g}")        # compose with graph capture only on request (AFO_ALLOC_EXPANDABLE=graphs) — unless NO input of this run can capture (below)
    try:
        import torch
        from opt_core.mem import torch_alloc as TA
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"import:{type(e).__name__}:{str(e)[:80]}")
    facts = {"conf": have or WANT, "via": "present"}
    if graph_gate is not None:
        facts["graph_gate"] = graph_gate                                   # engaged beside the graph lever because no input of this run can capture (LEVER … graph_gate=exceeded:<bucket>><gate>)
    if state == "absent":
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                TA.write_conf(WANT, via="runtime_api")            # the live allocator + os.environ (workers inherit it)
                facts["via"] = "runtime_api"
            else:
                TA.write_conf(WANT)                               # os.environ: torch reads it at its first CUDA allocation
                facts["via"] = "env"
        except Exception as e:  # noqa: BLE001 — a torch without the runtime surface: the stock policy stays, by name
            return _skipped(tag, f"write:{type(e).__name__}")
        facts["conf"] = os.environ.get(CONF_ENV, WANT)

    def effective():
        """(value, source): the allocator's own record (opt_core.mem.torch_alloc.effective): true|false|pending, snapshot|no-cuda|pending:…|unreadable:…"""
        try:
            f = TA.effective()
            v = f.get("expandable")
            return ("pending" if v is None else ("true" if v else "false")), str(f.get("source", "?")).replace(" ", "")
        except Exception as e:  # noqa: BLE001
            return "unreadable", type(e).__name__

    def line() -> str:
        from opt_core import report as R
        eff, src = effective()
        extra = {"graph_gate": facts["graph_gate"]} if facts.get("graph_gate") else {}
        return R.lever_line(tag, NAME, "on", impl=f"torch.cuda.memory[{WANT}]", origin="core", lever=LEVER,
                            via=facts["via"], conf=str(facts["conf"]).replace(" ", ""), effective=eff, source=src, **extra)

    def gate():
        from opt_core.gates import Gate
        eff, src = effective()
        ok = not (eff == "false" and src == "snapshot")           # the live allocator's OWN record says the policy did not take: refuse by name (a host without CUDA / an unreadable record is printed, not refused)
        return Gate(name=NAME, ok=ok, reason=(None if ok else f"allocator record: expandable_segments not effective (via={facts['via']})"),
                    details=dict(facts, effective=eff, source=src), words=())
    return Installed(LEVER, True, lines=[line], gates=[gate], facts=facts)
