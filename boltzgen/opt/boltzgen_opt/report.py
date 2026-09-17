"""Observability for boltzgen_opt: the activation line and the exit tally (both on stderr, prefixed ``[boltzgen-opt]``).

* activation — ``ACTIVE mode=<m> form=inproc|process switches=<BG_GRAPH=graph,...> boltzgen=<v> gpu=<name(<MiB>)>
  levers=<a,b,...> unavailable=<x,...> fallback=<x,...> dropped=<ENV,...> partial=<x,...> … fallbacks=<none|x,...>``
  (or ``DRY-RUN ...`` / ``NOT ACTIVE: <reason>``), formatted from the activation report returned by ``boltzgen_opt.enable()`` /
  ``status()``; ``partial=`` names the levers of the mode that could not run here (the activation is partial); ``fallbacks=`` is the last word of every ACTIVE line — ``none`` when no lever of the mode fell back, else the
  levers that did (the positive word beside ``fallback=``, which prints only when non-empty);
  the process form (`design`) prints the same line for the child it launches, with the levers the child's own stderr lines showed
  (stack.RUNNER_LINES);
* exit — the lever modules' own counters of this process (``xa_fastinit.STATS``, ``xa_hoist.STATS``, ``bg_graph_patch.STATS``,
  ``bg_hook._acc``), read in memory at interpreter exit — the same numbers the runner prints on its ``[xa_run] ... stats:`` line —
  and when no lever module was ever loaded the tally says so (never silent). ``register_exit_tally()`` is called before the lever
  modules are imported.
"""
from __future__ import annotations

import os
import sys

from opt_core import report as core_report

from . import TAG, print_fresh                         # the package's one tag (boltzgen_opt/__init__.py)

PREFIX = core_report.prefix(TAG)          # "[boltzgen-opt]"
NOT_ACTIVE = "NOT ACTIVE"
# the partial-activation refusal (held by the kit's unit tests): a mode is all of its levers — one that could not run here
# (a kernel that does not import, compile or launch, an unsupported shape or dtype) makes the mode refuse by name:
# ``<PREFIX> NOT ACTIVE: partial activation — <detail>; a mode is all of its levers: exit 3``; <detail> is the levers, their reasons, the
# verb's place. There is no opt-out: nothing runs under the mode's name with a subset of its levers.
_NOT_ACTIVE_HEAD = f"{PREFIX} {NOT_ACTIVE}: "          # what not_active_line() puts before a reason; partial_reason() is the refusal without it
TALLY_MODULES = ("xa_fastinit", "xa_hoist", "bg_graph_patch", "bg_hook")
TALLY_SKIP_KEYS = ("loads", "graphs", "keys")          # per-load / per-graph lists: sizes only
TALLY_MAX_FIELDS = 40


def _join(items) -> str:
    if not items:
        return "none"
    return ",".join(str(x) for x in items)


def gpu_label(gpu) -> str:
    if isinstance(gpu, dict):
        name = gpu.get("name")
        mem = gpu.get("memory_total_mib")
        if not name:
            return "none"
        return f"{name}({mem}MiB)" if mem else str(name)
    return str(gpu) if gpu else "none"


def activation_line(rep: dict) -> str:
    """One line from an activation report (enable()/status()/check)."""
    if rep is None:
        return not_active_line("no activation report")
    if not rep.get("active"):
        if rep.get("dry_run"):
            head = "DRY-RUN"
        else:
            return not_active_line(rep.get("reason", "unknown"))
    else:
        head = "ACTIVE"
    parts = [head, f"mode={rep.get('mode')}", f"form={rep.get('form', 'inproc')}", f"switches={rep.get('switches') or 'stock'}",
             f"boltzgen={rep.get('boltzgen_version') or '?'}", f"gpu={gpu_label(rep.get('gpu'))}",
             f"levers={_join(rep.get('levers_applied'))}"]
    if rep.get("levers_unavailable"):
        parts.append(f"unavailable={_join(rep['levers_unavailable'])}")
    if rep.get("levers_fallback"):
        parts.append(f"fallback={_join(rep['levers_fallback'])}")
    if rep.get("dropped_env"):
        parts.append(f"dropped={_join(rep['dropped_env'])}")
    if rep.get("partial"):
        parts.append(f"partial={_join(rep.get('levers_fallback')) if rep.get('levers_fallback') else 1}")
    if rep.get("hook_first"):
        parts.append("after=sitecustomize")
    if os.environ.get("BOLTZGEN_PIPELINE_STEP") and rep.get("form", "inproc") == "inproc":
        parts.append(f"step={os.environ['BOLTZGEN_PIPELINE_STEP']}")
    if head == "DRY-RUN" and rep.get("reason"):
        parts.append(("note: " if rep.get("mode") == "off" else "would-refuse: ") + rep["reason"])
    if head == "ACTIVE":
        parts.append(f"fallbacks={_join(rep.get('levers_fallback')) if rep.get('levers_fallback') else 'none'}")   # the last word: `none` says no lever fell back
    return f"{PREFIX} " + " ".join(parts)


def emit(rep: dict) -> str:
    return print_fresh(activation_line(rep))


def not_active_line(reason: str) -> str:
    """``<PREFIX> NOT ACTIVE: <reason>`` — every refusal line of the package (a gate, a failed run, a partial activation)."""
    return core_report.not_active_line(TAG, reason)


def partial_detail(levers, reasons=None, where: str = None) -> str:
    """<detail> of a partial activation: the levers that fell back first, then their reasons (``lever: reason; ...``), then the
    verb's own place (``in the run: the outputs under <dir> are not the mode's``)."""
    d = f"{_join(levers)} fell back"
    if reasons:
        d += " (" + "; ".join(f"{k}: {v}" for k, v in reasons.items()) + ")"
    if where:
        d += f" {where}"
    return d


def partial_reason(levers, reasons=None, where: str = None) -> str:
    """The refusal sentence of a partial activation — the manifest's ``reason`` and the body of its NOT ACTIVE line (exit EXIT_NOT_ACTIVE)."""
    return f"partial activation — {partial_detail(levers, reasons, where)}; a mode is all of its levers: exit {core_report.EXIT_NOT_ACTIVE}"


def partial_exit_line(levers, reasons=None, where: str = None) -> str:
    """``<PREFIX> NOT ACTIVE: partial activation — <detail>; a mode is all of its levers: exit 3`` — the one line of a partial activation."""
    return not_active_line(partial_reason(levers, reasons, where))


def step_refusal_reason(mode: str, refused: dict, step: str = None) -> str:
    """The refusal sentence of a model step a planned lever cannot serve (stack.arm_step_gate, before the model loads): ``mode=<m> cannot
    serve this step[ (<name>)]: <lever> — <its module's reason>[; …]; a mode is all of its levers, nothing of the step ran: exit 3 (the stock
    path is --mode off / BOLTZGEN_OPT=off)``."""
    which = f"this step ({step})" if step else "this step"
    body = "; ".join(f"{k} — {v}" for k, v in refused.items())
    return (f"mode={mode} cannot serve {which}: {body}; a mode is all of its levers, nothing of the step ran: exit {core_report.EXIT_NOT_ACTIVE} "
            f"(the stock path is --mode off / BOLTZGEN_OPT=off)")


def forced_exit(code: int) -> None:
    """End the process with ``code`` now, the kit's pending EXIT lines printed first (the core's forced_exit: os._exit after the tallies)."""
    core_report.forced_exit(code)


DESIGNS = "DESIGNS"                                      # the verb of the design-census line
NA = "n/a"                                               # a census field the run does not have (no design-generating step among its steps)


def designs_line(c: dict) -> str:
    """``<PREFIX> DESIGNS step=<step> requested=<n> produced=<m> oom_skipped=<k> featurizer_skipped=<j>[ stale=<s>][ reuse=true]`` — the
    run's design census (design.designs_census), every mode: the design files its design-generating step asked upstream to write and
    wrote, and the batches upstream's own handlers skipped (an out-of-memory batch, a featurizer exception) over every model step of the
    run. ``step=none requested=n/a produced=n/a`` when the run's steps generate no designs; ``stale`` (only when > 0) = design files of an
    earlier run into the same directory this run did not rewrite; ``reuse=true`` = upstream's ``--reuse`` kept existing designs."""
    pairs = [("step", c.get("step") or "none"),
             ("requested", NA if c.get("requested") is None else c["requested"]),
             ("produced", NA if c.get("produced") is None else c["produced"]),
             ("oom_skipped", c.get("oom_skipped", 0)), ("featurizer_skipped", c.get("featurizer_skipped", 0))]
    if c.get("stale"):
        pairs.append(("stale", c["stale"]))
    if c.get("reuse"):
        pairs.append(("reuse", "true"))
    return core_report.line(PREFIX, DESIGNS, *pairs)


def say(line: str) -> None:
    """A finished line (one of the formatters above) on stderr, starting a line of its own (print_fresh)."""
    print_fresh(line)


def note(msg: str) -> None:
    print_fresh(f"{PREFIX} {msg}")


# ------------------------------------------------------------------------------------------------------------------ exit tally
def _fmt(v):
    if isinstance(v, (list, tuple, dict)):
        return f"n={len(v)}"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def tally_fields(modules=None) -> list:
    """``(module, key, value)`` triples from the kit modules' own counters that are loaded in this process."""
    modules = sys.modules if modules is None else modules
    out = []
    for name in TALLY_MODULES:
        mod = modules.get(name)
        if mod is None:
            continue
        stats = getattr(mod, "STATS", None)
        if isinstance(stats, dict):
            for k, v in stats.items():
                if k in TALLY_SKIP_KEYS:
                    out.append((name, k, f"n={len(v)}" if hasattr(v, "__len__") else _fmt(v)))
                else:
                    out.append((name, k, _fmt(v)))
        acc = getattr(mod, "_acc", None)
        if isinstance(acc, dict):
            for k, v in acc.items():
                out.append((name, k, _fmt(v[1] if isinstance(v, (list, tuple)) and len(v) == 2 else v)))
    return out


def tally_line(modules=None) -> str:
    fields = tally_fields(modules)
    if not fields:
        return f"{PREFIX} EXIT tally: no lever module loaded in this process ({'/'.join(TALLY_MODULES)})"
    body = " ".join(f"{m}.{k}={v}" for m, k, v in fields[:TALLY_MAX_FIELDS])
    more = f" (+{len(fields) - TALLY_MAX_FIELDS} more)" if len(fields) > TALLY_MAX_FIELDS else ""
    return f"{PREFIX} EXIT tally: {body}{more}"




def register_exit_tally() -> bool:
    """Print this kit's EXIT tally line once at interpreter exit (the core's registration: one per tag; a failing line function prints a
    tally-failed line, never nothing). Idempotent; off under ``BOLTZGEN_OPT_QUIET=1``."""
    if os.environ.get("BOLTZGEN_OPT_QUIET") == "1":
        return False
    return core_report.register_exit_tally(TAG, tally_line)


EXAMPLES = {   # example ACTIVE lines (process form, on the pinned H100) for readers of the line grammar — field order, `fallbacks=` last; the lever and switch lists shown predate `async_writer` / `HL_ASYNC_WRITER=1` and are not the current mode table's
    'active_exact': "[boltzgen-opt] ACTIVE mode=exact form=process switches=BG_GRAPH=graph,XA_FAST_INIT=1,XA_HOIST=1 boltzgen=0.3.2 gpu=NVIDIA H100 80GB HBM3(81559MiB) levers=graph_sampler,fastinit,hoist,inproc dropped=PYTHONPATH fallbacks=none",
    'active_big': "[boltzgen-opt] ACTIVE mode=big form=process switches=BG_GRAPH=off,XA_FAST_INIT=1,XA_HOIST=0,SZ_TD_CHUNK=64,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True boltzgen=0.3.2 gpu=NVIDIA H100 80GB HBM3(81559MiB) levers=fastinit,td_chunk,inproc dropped=PYTHONPATH fallbacks=none",
}
