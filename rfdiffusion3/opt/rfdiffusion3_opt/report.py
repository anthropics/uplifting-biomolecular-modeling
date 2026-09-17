"""Observability for rfdiffusion3_opt: the activation line, the application line, the exit rule's lines and the exit tally (all on
stderr, prefixed ``[rfdiffusion3-opt]``, each at a line start through the package's one printer ``_emit.emit``), and the exit codes every
command returns.

* activation — ``ACTIVE mode=exact env=RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1 interpreter=patched(10/10) foundry=<version> gpu=<name(smNN)> kit=<version>
  applied=at-import attn_pin=1`` (or ``DRY-RUN ...`` / ``NOT ACTIVE: <reason>`` / ``NOT ACTIVE mode=off ...``), formatted from the activation report
  returned by ``rfdiffusion3_opt.enable()`` / ``status()``; ``not_active_line(reason)`` is the one formatter of a refusal by name;
* application — ``APPLIED rfd3.model.hoist RFD3_HOIST=True parts=dedup,downcast,pll,valid,where RFD3_COMPILE=False RFD3_TOKEN_SDPA=False`` once, right after the
  kit's module has read the environment (its own ``describe()``, hoist.py:281);
* the exit rule — a run is PARTIAL when a planned lever has no evidence of application: the kit's module read ``RFD3_HOIST`` off at its
  import (``APPLIED ... RFD3_HOIST=False``) or was never imported by a completed run (``APPLIED none``). The family line of a partial
  exit (``partial_line``): ``NOT ACTIVE: partial activation — <lever (reason)>; exit 3 (--allow-partial records and proceeds)``,
  exit EXIT_NOT_ACTIVE; with ``--allow-partial`` (``RFDIFFUSION3_OPT_ALLOW_PARTIAL=1`` in the environment) ``PARTIAL allowed: <detail>
  (--allow-partial, recorded)``, the exit is the run's own and the manifest records ``allow_partial``; a run that failed on its own
  keeps its own code (``verdict``: failed ranks before partial) and the partial is a record — ``PARTIAL recorded: <detail> — the run
  failed on its own, exit <code> …`` (no exit-3 tail), ``partial`` / ``partial_reason`` in the manifest. Every verb prints it through
  ``partial_line`` (``design`` and the ``.pth`` route at the kit module's import, ``warm`` from its child's lines); ``verdict`` is the
  one decision of a completed ``design``. A refusal by name (a deployment gate, a line the mode cannot serve) has no allowance;
* the warnings channel — upstream's engine construction installs a blanket ``warnings.filterwarnings("ignore")``
  (``foundry/utils/logging.py:123``, from ``configure_minimal_inference_logging``, ``inference_engines/base.py:63-64``), which would
  swallow the deterministic recipe's only named event (the ``UserWarning`` ``torch.use_deterministic_algorithms(True, warn_only=True)``
  raises per nondeterministic op; ``warn_only=False`` changes the numerics and is not an option here).
  ``reopen_warnings()`` removes that blanket entry once the kit's module has been imported — during the model build, after the
  construction — so the warnings of the roll-out reach stderr again under Python's own filters (``default``: once per op site);
* exit — the kit modules' own counters (``hoist.HOIST_STATS`` rollouts/entries_last/hits_last; the graph
  sampler's ``GRAPH_STATS``; the fused transition's ``FZT_STATS`` — fused and stock-routed call counts by cell), read in memory at
  interpreter exit, and the ``WARNINGS`` line: ``warnings.filters`` as found at exit (the channel's state is never silent);
  when the module was never loaded the tally says so. ``register_exit_tally()`` is called at activation, before the upstream package
  is imported.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence

from opt_core import report as core_report
from opt_core.report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE, PARTIAL_ALLOWED, PARTIAL_REFUSED   # the exit table (the core's): 0 ok · 1 failed · 2 usage · 3 not active / not stock / partial

from ._autoload import TAG                                                          # the kit's line tag, one spelling: "rfdiffusion3-opt" (import-free in the .pth module)
from . import _emit
from ._emit import fresh
from .census import GRAPH_MODULE                                                       # the kit's graph sampler module, named once (kernels reads its GRAPH_STATS too)

PREFIX = core_report.prefix(TAG)                                                    # "[rfdiffusion3-opt]"
EXIT_KERNELS = 5                                                                    # the KERNELS guard's refusal (census.EXIT_KERNELS): an accelerator the route expects is absent / fell back — beside the core's 0 ok · 1 failed · 2 usage · 3 not active
HOIST_MODULE = "rfd3.model.hoist"
FZT_MODULE = "rfd3.model.layers.layer_utils"
INIT_CHUNK_MODULE = "rfd3.model.layers.layer_utils"                                                   # [RFD3_INIT_CHUNK] the row-blocked initialiser embedding (init_chunk_describe)
ENV_ALLOW_PARTIAL = "RFDIFFUSION3_OPT_ALLOW_PARTIAL"                               # =1: the environment form of --allow-partial (beside RFDIFFUSION3_OPT=<mode>)
_MODULES: Dict[str, object] = {"hoist": None, "graph": None, "fzt": None, "init_chunk": None}                        # the kit modules as last reported (the tally after a failed import)


def not_active_line(reason: str) -> str:
    """``[rfdiffusion3-opt] NOT ACTIVE: <reason>`` — the one refusal line of the package (the core's grammar)."""
    return core_report.not_active_line(TAG, reason)


def allow_partial_env(environ=None) -> bool:
    """The environment form of ``--allow-partial``: ``RFDIFFUSION3_OPT_ALLOW_PARTIAL=1`` (stripped from the stock arm's child with the
    package's other switch, stack.PACKAGE_ENV) — the core's one reader."""
    return core_report.allow_partial(False, ENV_ALLOW_PARTIAL, environ)


def partial_detail(levers: Sequence[str], reason: Optional[str]) -> str:
    """``<detail>`` of the family line: the levers without evidence of application by name, with this engine's own reason."""
    return ", ".join(f"{lever} ({reason or 'no reason recorded'})" for lever in levers) or "-"


PARTIAL_RECORDED = "PARTIAL recorded: {detail} — the run failed on its own, exit {code} (opt_manifest.json: partial, partial_reason)"


def partial_line(levers: Sequence[str], reason: Optional[str], allow_partial: bool = False, exit_code: Optional[int] = None) -> str:
    """The one line of a partial activation — every verb prints it through here: the core's ``PARTIAL_REFUSED`` (``NOT ACTIVE: partial
    activation — …; exit 3 (--allow-partial records and proceeds)``) without ``--allow-partial``, the core's ``PARTIAL_ALLOWED`` with it. A
    run that failed on its own (``exit_code`` given, neither 0 nor 3) keeps its own code: the partial is a RECORD (``PARTIAL recorded: …``,
    no exit-3 tail), written to the manifest beside that code."""
    detail = partial_detail(levers, reason)
    if exit_code not in (None, EXIT_OK, EXIT_NOT_ACTIVE) and not allow_partial:
        return f"{PREFIX} " + PARTIAL_RECORDED.format(detail=detail, code=exit_code)
    return (PARTIAL_ALLOWED if allow_partial else PARTIAL_REFUSED).format(prefix=PREFIX, detail=detail)


def verdict(rc: int, rep: Optional[dict], allow_partial: bool) -> dict:
    """The exit rule's one decision after a ``design`` run, in this precedence — one code per condition: the run's own non-zero code
    (a refusal at the kit module's import already carries EXIT_NOT_ACTIVE; a failed run is never conflated with partial); then a
    completed run whose activation report is partial (a planned lever without evidence of application) -> EXIT_NOT_ACTIVE unless
    ``allow_partial`` (recorded; the run's own code); else the run's code. Returns {exit_code, partial: [levers], partial_reason,
    allow_partial, status: ok|failed|not_active|partial|partial_allowed}."""
    rep = rep or {}
    partial = list(rep.get("levers_fallback") or []) if rep.get("partial") else []
    reason = rep.get("partial_reason")
    if rc == EXIT_NOT_ACTIVE:
        status, code = "not_active", EXIT_NOT_ACTIVE
    elif rc != EXIT_OK:
        status, code = "failed", rc
    elif partial and not allow_partial:
        status, code = "partial", EXIT_NOT_ACTIVE
    elif partial:
        status, code = "partial_allowed", EXIT_OK
    else:
        status, code = "ok", EXIT_OK
    return {"exit_code": code, "partial": partial, "partial_reason": reason, "allow_partial": bool(allow_partial), "status": status}


def note_module(module, which: str = "hoist") -> None:
    """Remember a kit module once reported, so the exit tally reads its counters even after an import the exit rule refused."""
    _MODULES[which] = module


BLANKET_IGNORE = ("ignore", None, Warning, None, 0)                            # what warnings.filterwarnings("ignore") inserts (logging.py:123)


def reopen_warnings() -> int:
    """Remove upstream's blanket ``ignore`` filter(s) from ``warnings.filters`` — every other filter (Python's defaults, ``-W`` /
    PYTHONWARNINGS, upstream's targeted ones, the recipe's own) stays, in order, re-added through the public API so the module's
    filter cache is invalidated. Returns the number removed (0: the channel was open)."""
    kept = [tuple(f) for f in warnings.filters if tuple(f) != BLANKET_IGNORE]
    removed = len(warnings.filters) - len(kept)
    if removed:
        warnings.resetwarnings()
        for action, message, category, module, lineno in reversed(kept):        # front insertion in reverse order keeps the order
            warnings.filterwarnings(action, message=getattr(message, "pattern", message) or "", category=category or Warning,
                                    module=getattr(module, "pattern", module) or "", lineno=lineno or 0)
    return removed


def filter_str(f) -> str:
    action, message, category, module, lineno = f
    return f"{action}:{getattr(category, '__name__', category) or '*'}:{getattr(message, 'pattern', message) or '*'}:{getattr(module, 'pattern', module) or '*'}" + (f":{lineno}" if lineno else "")


def warnings_line(pid: int | None = None) -> str:
    """``WARNINGS pid=<pid> filters=<n> [<action>:<category>:<message>:<module> …]`` — ``warnings.filters`` as found (front first)."""
    pid = os.getpid() if pid is None else pid
    fs = list(warnings.filters)
    return f"{PREFIX} WARNINGS pid={pid} filters={len(fs)} blanket_ignore={sum(tuple(f) == BLANKET_IGNORE for f in fs)} [" + " ".join(filter_str(f) for f in fs) + "]"


def _gpu_str(gpu) -> str:
    if isinstance(gpu, dict):
        sm = gpu.get("sm")
        return f"{gpu.get('name')}" + (f"(sm{sm})" if sm else "")
    return str(gpu) if gpu else "none"


def activation_line(rep: dict) -> str:
    mode = rep.get("mode")
    if "env" not in rep:                                     # not a mode, or refused before resolution
        return not_active_line(rep.get("reason", "unknown reason"))
    if rep.get("dry_run"):
        head = "DRY-RUN"
    elif rep.get("active"):
        head = "ACTIVE"
    elif mode == "off":
        head = "NOT ACTIVE"
    else:
        return not_active_line(rep.get("reason", "unknown reason"))
    interp = rep.get("interpreter") or {}
    fields = [f"mode={mode}", f"env={rep.get('env_line', 'none')}",
              f"interpreter={interp.get('label', 'unknown')}({interp.get('detail', '?')})",
              f"foundry={rep.get('foundry_version')}", f"gpu={_gpu_str(rep.get('gpu'))}",
              f"kit={rep.get('kit_version')}", f"applied={rep.get('applied', 'n/a')}"]
    if rep.get("attn_pin"):                                    # the kit routes pin the pre-flight atom-attention decision for the process (routes.Route.attn_pin; kernels ATTN_PREFLIGHT `pinned=1`)
        fields.append("attn_pin=1")
    if rep.get("alloc_conf"):                                  # torch's allocator configuration as launched (configs/h100.env: expandable_segments:True, part of the documented launch line)
        fields.append(f"alloc_conf={rep['alloc_conf']}")
    if isinstance(rep.get("det"), dict):                       # the deterministic recipe's level (det.py): 0 = upstream's default numerics
        fields.append(f"det={rep['det'].get('level', 0)}")
    if rep.get("trigger"):
        fields.append(f"trigger={rep['trigger']}")
    if mode == "off":
        fields.append("(stock: nothing exported)")
        if rep.get("env_present"):
            fields.append(f"env_present={','.join(rep['env_present'])} (kit names are stripped from the stock arm's child; upstream's own switches pass through)")
    if rep.get("dry_run") and rep.get("findings"):
        fields.append(f"would_refuse={' | '.join(rep['findings'])!r}")
    return f"{PREFIX} {head} " + " ".join(fields)


def pins_line(note: str) -> str:
    """``[rfdiffusion3-opt] PINS <finding>`` — one finding of stock/check_pins.py (the distribution's install source / commit, foundry/utils/torch.py):
    reported on every route, never a refusal."""
    return f"{PREFIX} PINS {note}"


def applied_line(module) -> str:
    try:
        d = module.describe()
    except Exception as e:  # noqa: BLE001
        return f"{PREFIX} APPLIED {HOIST_MODULE} (describe failed: {e!r}) RFD3_HOIST={getattr(module, 'HOIST', None)}"
    return (f"{PREFIX} APPLIED {HOIST_MODULE} RFD3_HOIST={d.get('RFD3_HOIST')} parts={d.get('parts')} "
            f"RFD3_COMPILE={d.get('RFD3_COMPILE')} RFD3_TOKEN_SDPA={d.get('RFD3_TOKEN_SDPA')}")


def graph_stats() -> dict | None:
    """The graph sampler's process tally (patched cudagraph_sampler.GRAPH_STATS: installs, captures, hits, fallbacks), or None when
    the module was never imported in this process (the switch off, or no sampler constructed)."""
    mod = sys.modules.get(GRAPH_MODULE) or _MODULES["graph"]
    v = getattr(mod, "GRAPH_STATS", None) if mod is not None else None
    return dict(v) if isinstance(v, dict) else None


def graph_lever_line(stats: dict | None) -> str:
    """The graph sampler's state as the core's LEVER line (opt_core.report.lever_line — the grammar is the core's: ``LEVER name=<switch>
    state=on|skipped [reason=<word>] impl=cuda_graphs origin=kit installs=<n> captures=<n> hits=<n> fallbacks=<n> declined=<n>``). ``state=on``
    when at least one roll-out of the process ran the captured step (a roll-out either captures a graph or reuses one: captures + hits >= 1) and
    none left it; ``state=skipped reason=<word>`` otherwise — ``not_imported`` (no sampler read the switch), ``not_installed`` (installs=0: no
    default sampler wrapped), ``capture_failed:<n>`` (fallbacks: the eager denoiser served them), ``not_driven:<kinds>`` (declined: roll-out kinds
    the sampler does not drive ran the stock loop), ``no_rollout`` (installed, no roll-out ran)."""
    from .modes import KIT_GRAPH_SWITCH                   # named at call time: modes is importable wherever report is (the kit routes), no cycle at import
    s = stats or {}
    n = {k: int(s.get(k) or 0) for k in ("installs", "captures", "hits", "fallbacks", "declined")}
    if stats is None:
        state, reason = "skipped", "not_imported"
    elif not n["installs"]:
        state, reason = "skipped", "not_installed"
    elif n["fallbacks"] > 0:
        state, reason = "skipped", f"capture_failed:{n['fallbacks']}"
    elif n["declined"] > 0:
        state, reason = "skipped", "not_driven:" + str(s.get("declined_by") or "unknown").replace(" ", "")
    elif n["captures"] + n["hits"] == 0:
        state, reason = "skipped", "no_rollout"
    else:
        state, reason = "on", None
    return core_report.lever_line(TAG, KIT_GRAPH_SWITCH, state, reason=reason, impl="cuda_graphs", origin="kit", **n)


def fzt_fallback_line(unserved: dict, calls: int) -> str:
    """``<PREFIX> FALLBACK lever=RFD3_FZT reason=no-cell cells=<C>x<NH>,… core=<the core's refusal words>[;…] route=stock-transition calls=<n>``:
    the fused transition's named fallback — cells of the route (patched layer_utils.FZT_CELLS) the core's cell table does not serve on this
    stack (its compute capability / Triton line; `cells_unserved` of fzt_describe()) ran the stock transition, `calls` times in all. One
    line per process (stack.report_fzt), which also marks the activation partial."""
    cells = ",".join(sorted(unserved))
    core = ";".join(sorted({str(v) for v in unserved.values()}))
    return f"{PREFIX} FALLBACK lever=RFD3_FZT reason=no-cell cells={cells} core={core} route=stock-transition calls={int(calls)}"


def fzt_stats() -> dict | None:
    """The fused transition's process tally (patched layers/layer_utils.fzt_describe(): RFD3_FZT, installs, fused_calls, stock_calls,
    cells_fused, cells_stock, modules_packed, cells, opt_core_cells_sha256), or None when the module was
    never imported in this process."""
    mod = sys.modules.get(FZT_MODULE) or _MODULES["fzt"]
    fn = getattr(mod, "fzt_describe", None) if mod is not None else None
    return dict(fn()) if callable(fn) else None


def init_chunk_stats() -> dict | None:
    """[RFD3_INIT_CHUNK] the row-blocked initialiser embedding's process tally (patched layers/blocks.init_chunk_describe(): RFD3_INIT_CHUNK,
    calls, chunked_calls, blocks, max_atoms, pairs_per_block), or None when the module was never imported in this process."""
    mod = sys.modules.get(INIT_CHUNK_MODULE) or _MODULES["init_chunk"]
    fn = getattr(mod, "init_chunk_describe", None) if mod is not None else None
    if fn is None:
        return None
    try:
        v = fn()
    except Exception:  # noqa: BLE001
        return None
    return dict(v) if isinstance(v, dict) else None


def gather_stats() -> dict | None:
    """[RFD3_GATHER_ATTN] the gather_attn lever's process tally (gather.describe(): RFD3_GATHER_ATTN, installs, unavailable, served, stock, stock_by,
    cells_served, idx_builds, kernel, strategy, active), or None when the switch was never read on in this process."""
    from . import gather as _g
    d = _g.describe()
    return d if d.get(_g.SWITCH) else None


def memory_stats() -> dict | None:
    mod = sys.modules.get(HOIST_MODULE) or _MODULES["hoist"]      # the module as reported, even after an import the exit rule refused
    if mod is None:
        return None
    out = {"RFD3_HOIST": getattr(mod, "HOIST", None), "parts": ",".join(sorted(getattr(mod, "PARTS", []) or []))}
    v = getattr(mod, "HOIST_STATS", None)                     # the module's own roll-out cache tally
    if isinstance(v, dict):
        out["hoist_stats"] = dict(v)
    g = graph_stats()
    if g is not None:
        out["graph_stats"] = g
    f = fzt_stats()
    if f is not None:
        out["fzt_stats"] = {k: v for k, v in f.items() if k in ("RFD3_FZT", "installs", "fused_calls", "stock_calls", "modules_packed", "cells_fused", "cells_stock")}
    c = compile_stats()
    if c is not None:
        out["compile_stats"] = {k: v for k, v in c.items() if k in ("RFD3_COMPILE", "dynamic", "installs", "wrapped", "declared", "graphs", "graph_breaks", "frames_ok", "frames_total", "frames_skipped", "frames_skipped_allowed", "recompile_limit_hits", "cache_entries_max", "cache_size_limit")}
    ic = init_chunk_stats()
    if ic is not None:
        out["init_chunk_stats"] = {k: v for k, v in ic.items() if k in ("RFD3_INIT_CHUNK", "calls", "chunked_calls", "blocks", "max_atoms", "pairs_per_block")}
    g2 = gather_stats()
    if g2 is not None:
        out["gather_stats"] = {k: v for k, v in g2.items() if k in ("RFD3_GATHER_ATTN", "installs", "unavailable", "served", "stock", "stock_by", "cells_served", "idx_builds", "kernel", "strategy")}
    return out


def compile_stats() -> dict | None:
    """[RFD3_COMPILE] the compile lever's process tally (patched hoist.compile_describe(): RFD3_COMPILE, installs, targets, wrapped, and
    dynamo's own counters unique_graphs / graph_breaks / frames), or None when the module was never imported in this process."""
    mod = sys.modules.get(HOIST_MODULE) or _MODULES["hoist"]
    fn = getattr(mod, "compile_describe", None) if mod is not None else None
    return dict(fn()) if callable(fn) else None


def compile_lever_line() -> str | None:
    """The shared core's LEVER line for RFD3_COMPILE (opt_core.capture.compile CompileRecord.line via patched hoist.compile_lever_line), or None
    when the lever is off / the module was never imported / the pass was refused at the engine's initialize for a computation the mode does
    not drive (census.not_served: no roll-out ran, the compile applies at the first one — nothing to state about it)."""
    from . import census
    if census.not_served() is not None:
        return None
    mod = sys.modules.get(HOIST_MODULE) or _MODULES["hoist"]
    fn = getattr(mod, "compile_lever_line", None) if mod is not None else None
    return fn(PREFIX.strip("[]")) if callable(fn) else None



def tally_fields(stats: dict) -> list:
    out = [f"RFD3_HOIST={stats.get('RFD3_HOIST')}", f"parts={stats.get('parts')}"]
    for k in ("hoist_stats", "graph_stats", "fzt_stats", "init_chunk_stats", "compile_stats", "gather_stats"):
        for kk, vv in (stats.get(k) or {}).items():
            out.append(f"{k.replace('hoist_', '').replace('graph_stats', 'graph').replace('fzt_stats', 'fzt').replace('compile_stats', 'compile').replace('gather_stats', 'gather').replace('init_chunk_stats', 'init_chunk')}.{kk}={vv}")
    return out


def exit_tally_line(pid: int | None = None) -> str:
    pid = os.getpid() if pid is None else pid
    stats = memory_stats()
    if stats:
        return f"{PREFIX} EXIT pid={pid} source=memory " + " ".join(tally_fields(stats))
    return f"{PREFIX} EXIT pid={pid} no lever counters: {HOIST_MODULE} was never loaded in this process"


_BEFORE_EXIT: List = []                                                             # callables run first at exit (stack: the graph lever's evidence line on upstream's own command line)


def before_exit(fn) -> None:
    """Run ``fn`` (no arguments, returns None) ahead of the exit tally — once registered, once run."""
    if fn not in _BEFORE_EXIT:
        _BEFORE_EXIT.append(fn)


def _exit_lines() -> str:
    """The exit tally and the warnings census, as the core's one exit hook prints them (two physical lines)."""
    for pre in list(_BEFORE_EXIT):
        try:
            pre()
        except Exception as e:  # noqa: BLE001
            emit(f"{PREFIX} EXIT {getattr(pre, '__name__', 'hook')} failed: {e!r}")
    lines = []
    for fn in (exit_tally_line, compile_lever_line, warnings_line):
        try:
            line = fn()
        except Exception as e:  # noqa: BLE001
            line = f"{PREFIX} EXIT {fn.__name__} failed: {e!r}"
        if line is not None:                                # compile_lever_line is None with the compile lever off
            lines.append(line)
    return fresh("\n".join(lines))                       # handed to the core's exit hook already at a line start (the core writes it as given)


def register_exit_tally() -> None:
    """Print the exit tally at interpreter exit — the core's hook, once per process (opt_core.report.register_exit_tally)."""
    core_report.register_exit_tally(TAG, _exit_lines)


emit = _emit.emit                                        # the package's one printer (_emit): every line fresh, flushed
