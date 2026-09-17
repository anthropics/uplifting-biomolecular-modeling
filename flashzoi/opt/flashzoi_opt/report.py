"""Observability for flashzoi_opt: every `[flashzoi-opt]` line the package prints is formatted HERE and nowhere else (the format
strings below are the contract a reader keys on). All lines go to stderr.

* activation — ACTIVE_FMT, printed ONCE per process at the first attach of the kit to a model (after the kit's own apply line):
  ``[flashzoi-opt] ACTIVE mode=<m> variant=- gpu=<torch.cuda.get_device_name()> cc=<major>.<minor> components=<the kit's LEVERS
  joined by ','> (partial: <unavailable or none>)``, with `` drift=[<item>; ...]`` appended (DRIFT_SUFFIX) when the environment is off
  the kit's pins — another GPU class, stack versions, a TF32 override variable: named, the run goes on;
* refusal — NOT_ACTIVE_FMT ``[flashzoi-opt] NOT ACTIVE: <reason>`` (the CLI exits 3): the stock not at its pin, no CUDA device, the kit
  tree missing, the kit cannot run here (a kernel that does not compile or launch, a model shape it does not serve — a mode is all of
  its levers or refuses by name), a partial activation without --allow-partial;
* dry run — DRY_RUN_FMT (``check``: the resolution and the gates on this machine, nothing applied);
* per item — PRED_FMT ``[flashzoi-opt] pred <item> s0 samples=1 <wall>s`` (one prediction: the documented call incl. host
  materialisation; the seed axis is inert — s0 names it);
* exit — EXIT_FMT ``[flashzoi-opt] EXIT mode=<m> items=<n> ok=<k> failed=<f> wall=<s>s partial=<components_fallback or none>`` at
  interpreter exit, once per process (register_exit_tally(); the counters are this module's ITEMS, moved by note_item(); partial = the
  kit's evidence read back at exit, stack.settle() — the env / in-process route's record of a partial activation, which cannot set
  the host process's exit: the CLI verbs pred and warm are the gated form); never silent;
* warm — WARM_FMT; the weights-on-device moment — READY_FMT.
"""
from __future__ import annotations

import atexit
import os
import sys
import time

PREFIX = "[flashzoi-opt]"
VARIANT = "-"                                             # this model has no variant axis; the field keeps the line's fixed shape

ACTIVE_FMT = "{prefix} ACTIVE mode={mode} variant={variant} gpu={gpu} cc={cc} components={components} (partial: {partial})"
DRIFT_SUFFIX = " drift=[{items}]"                # appended to the ACTIVE / DRY-RUN line when the environment is off the kit's pins (GPU class, stack pins, TF32 overrides): named, never refused
NOT_ACTIVE_FMT = "{prefix} NOT ACTIVE: {reason}"
ASIDE_FMT = "{prefix} ASIDE model={index}: {reason} — not attached; upstream's own forward serves this model"          # a model the kit's kernels are not built for (CPU / non-float32): served as shipped, said once
PARTIAL_FMT = "{prefix} NOT ACTIVE: partial activation — {detail}; exit {exit_code} (--allow-partial records and proceeds)"   # the family grammar of a partial exit: detail = the levers, then the reason
PARTIAL_ALLOWED_FMT = "{prefix} PARTIAL allowed: {detail} (--allow-partial, recorded)"                                        # the same detail under --allow-partial (the exit is the run's own)
DRY_RUN_FMT = "{prefix} DRY-RUN mode={mode} variant={variant} gpu={gpu} cc={cc} components={components} knobs={knobs} stack_key={stack_key} would_refuse={would_refuse}"
PRED_FMT = "{prefix} pred {item} s0 samples=1 {wall:.3f}s"
EXIT_FMT = "{prefix} EXIT mode={mode} items={items} ok={ok} failed={failed} wall={wall:.1f}s partial={partial}"
READY_FMT = "{prefix} ready replicates={n} t={t:.2f}s"
MULTI_SUFFIX = " multi={n}"                                # appended to the ACTIVE line under pred --jobs (n = the jobs of the file)
MULTI_PREFIX = "[flashzoi-opt multi]"                      # the multi-job form's own lines (the resident-process line grammar: ready / item / DONE)
MULTI_READY_FMT = "{mprefix} ready mode={mode} load={load:.3f}s jobs={n}"                                                    # once, after the one load + apply (load = load_once_s)
MULTI_ITEM_FMT = "{mprefix} item {index}/{n} {name} items={items} ok={ok} failed={failed} wall={wall:.3f}s rc={rc}"          # ONE line per job: the job's wall inside the resident process
MULTI_DONE_FMT = "{mprefix} DONE jobs={n} ok={ok} failed={failed} load={load:.3f}s wall={wall:.3f}s"                         # once; wall = every job, after the ready line
WARM_FMT = "{prefix} WARM {status} mode={mode} forwards={forwards} triton_cache={before}->{after} wall={wall:.1f}s{reason}"


ITEMS = {"items": 0, "ok": 0, "failed": 0}               # the exit tally's counters (note_item)
_T0 = time.perf_counter()
_REGISTERED = {"done": False, "mode": None}
_ACTIVE_PRINTED = {"done": False}


def _join(items) -> str:
    return ",".join(str(x) for x in items) if items else "none"


def gpu_fields(rep: dict) -> tuple:
    g = (rep or {}).get("gpu") or {}
    return (g.get("name") or "none", g.get("cc") or "none")


def activation_line(rep: dict) -> str:
    """The ACTIVE line for an activation report (components = the mode's component set: the kit's LEVERS verbatim)."""
    gpu, cc = gpu_fields(rep)
    return ACTIVE_FMT.format(prefix=PREFIX, mode=rep.get("mode"), variant=VARIANT, gpu=gpu, cc=cc, components=_join(rep.get("components_planned")),
                             partial=_join(rep.get("components_unavailable"))) + multi_suffix(rep) + drift_suffix(rep)


def drift_suffix(rep: dict) -> str:
    """` drift=[a; b]` when the report names environment drift; empty otherwise (the line unchanged)."""
    items = [str(x) for x in ((rep or {}).get("drift") or [])]
    return DRIFT_SUFFIX.format(items="; ".join(items)) if items else ""


def multi_suffix(rep: dict) -> str:
    """` multi=<n>` when the report carries the multi-job form's job count; empty otherwise (the line unchanged)."""
    n = (rep or {}).get("multi")
    return MULTI_SUFFIX.format(n=int(n)) if n else ""


def not_active_line(reason: str) -> str:
    return NOT_ACTIVE_FMT.format(prefix=PREFIX, reason=reason)


def partial_line(detail: str, exit_code: int | None = None) -> str:
    """The partial-activation line: the refusal (PARTIAL_FMT, `exit_code` = the code the caller exits with) or, with `exit_code` None,
    the --allow-partial form (PARTIAL_ALLOWED_FMT)."""
    if exit_code is None:
        return PARTIAL_ALLOWED_FMT.format(prefix=PREFIX, detail=detail)
    return PARTIAL_FMT.format(prefix=PREFIX, detail=detail, exit_code=int(exit_code))


def dry_run_line(rep: dict) -> str:
    gpu, cc = gpu_fields(rep)
    return DRY_RUN_FMT.format(prefix=PREFIX, mode=rep.get("mode"), variant=VARIANT, gpu=gpu, cc=cc, components=_join(rep.get("components_planned")),
                              knobs=rep.get("knobs_line") or "none", stack_key=rep.get("stack_key") or "unknown",
                              would_refuse=repr(rep.get("would_refuse")) if rep.get("would_refuse") else "none") + multi_suffix(rep) + drift_suffix(rep)


def multi_ready_line(mode: str, load_once_s: float, n: int) -> str:
    return MULTI_READY_FMT.format(mprefix=MULTI_PREFIX, mode=mode, load=float(load_once_s), n=int(n))


def multi_item_line(index: int, n: int, name: str, counts: dict, wall_s: float) -> str:
    """The per-job line of the multi-job form (one per job; rc = 0 when every item of the job is ok, else 1)."""
    rc = 0 if counts.get("failed", 0) == 0 else 1
    return MULTI_ITEM_FMT.format(mprefix=MULTI_PREFIX, index=int(index), n=int(n), name=name, items=counts.get("items", 0), ok=counts.get("ok", 0),
                                 failed=counts.get("failed", 0), wall=float(wall_s), rc=rc)


def multi_done_line(n: int, ok: int, failed: int, load_once_s: float, wall_s: float) -> str:
    return MULTI_DONE_FMT.format(mprefix=MULTI_PREFIX, n=int(n), ok=int(ok), failed=int(failed), load=float(load_once_s), wall=float(wall_s))


def pred_line(item: str, wall_s: float) -> str:
    return PRED_FMT.format(prefix=PREFIX, item=item, wall=float(wall_s))


def ready_line(n_replicates: int, t_load_s: float) -> str:
    return READY_FMT.format(prefix=PREFIX, n=int(n_replicates), t=float(t_load_s))


def exit_tally_line(mode: str | None = None, wall_s: float | None = None) -> str:
    from . import stack                                   # the kit's evidence read back (settle) for the partial field; a no-op without runners
    wall = (time.perf_counter() - _T0) if wall_s is None else float(wall_s)
    return EXIT_FMT.format(prefix=PREFIX, mode=mode or _REGISTERED["mode"] or "none", items=ITEMS["items"], ok=ITEMS["ok"], failed=ITEMS["failed"], wall=wall,
                           partial=_join(stack.settle().get("components_fallback")))


def warm_line(res: dict) -> str:
    tc = res.get("triton_cache_files") or {}
    return WARM_FMT.format(prefix=PREFIX, status=res.get("status"), mode=res.get("mode"), forwards=res.get("forwards"), before=tc.get("before"), after=tc.get("after"),
                           wall=float(res.get("wall_s") or 0.0), reason=(f" reason={res['reason']}" if res.get("reason") else ""))


def emit(line: str, stream=None) -> str:
    print(line, file=stream or sys.stderr, flush=True)
    return line


def log_activation(rep: dict, stream=None) -> str:
    """Print the ACTIVE line once per process (the first attach); later calls return the line without printing."""
    line = activation_line(rep)
    if not _ACTIVE_PRINTED["done"]:
        _ACTIVE_PRINTED["done"] = True
        emit(line, stream)
    return line


def log_refusal(reason: str, stream=None) -> str:
    return emit(not_active_line(reason), stream)


def aside_line(index: int, reason: str) -> str:
    return ASIDE_FMT.format(prefix=PREFIX, index=index, reason=reason)


def log_aside(index: int, reason: str, stream=None) -> str:
    return emit(aside_line(index, reason), stream)


def log_partial(detail: str, exit_code: int | None = None, stream=None) -> str:
    return emit(partial_line(detail, exit_code), stream)


def note_item(ok: bool) -> None:
    ITEMS["items"] += 1
    ITEMS["ok" if ok else "failed"] += 1


def _print_exit_tally() -> None:
    try:
        line = exit_tally_line()
    except Exception as e:  # noqa: BLE001
        line = f"{PREFIX} EXIT tally failed: {e!r}"
    try:
        sys.stderr.write(line + "\n"); sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def register_exit_tally(mode: str) -> None:
    """Print the exit tally at interpreter exit (once per process; the mode named at registration)."""
    _REGISTERED["mode"] = mode
    if _REGISTERED["done"]:
        return
    _REGISTERED["done"] = True
    atexit.register(_print_exit_tally)


def pid() -> int:
    return os.getpid()
