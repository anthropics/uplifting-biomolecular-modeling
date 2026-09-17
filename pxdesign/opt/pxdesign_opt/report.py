"""Observability for pxdesign_opt: the activation line, the application line, the run line and the exit tally (all on stderr,
prefixed ``[pxdesign-opt]``), composed from the core's line primitives (`opt_core.report`: prefix, join, kv, gpu_label, the exit-tally
registration); the bytes of every line are this kit's.

* activation — ``ACTIVE mode=<m> pxdesign=<v> protenix=<v> pxdbench=<v> gpu=<name(smNN)> levers=<h1,...> env=<K=V,...>
  applied=deferred|installed package_levers=<names|none> tier=<1|2>[ notes=<sentence; …>]`` (or ``DRY-RUN ...`` / ``NOT ACTIVE: <reason>`` /
  ``STOCK mode=off ...`` for the stock arm), formatted from the activation report returned by ``pxdesign_opt.enable()`` / ``status()``; the
  ``notes=`` tail (present only when there is something to name) carries the environment facts the kit names and never refuses on — a GPU
  class outside the measured ones, a torch build other than the pin, a target-GPU mismatch, TF32 library overrides (`stack.environment_notes`);
* application — ``APPLIED model#<i> levers=<...> fallbacks=<...>`` once per runner the hoist was applied to, with what the kit's own
  records show patched (the class attributes hoist rebinds, the model's ready flag);
* package — ``PACKAGE model#<i> levers=<planned> applied=<...> fallbacks=<...> <evidence fields>`` once per runner, only when the mode
  plans a package lever (registry.PACKAGE_LEVERS: `big` and `fast` do);
* levers — at exit, one ``LEVER name=tf32 … strategy=F4.tf32_matmul`` and one ``LEVER name=sdedup … strategy=F7.row_dedup`` line (opt_core.report.lever_line
  grammar: state=on with the live policy words / the RowDedup words class=tolerance served= fallback= fallback_by=nonuniform:<n> rows_saved=, state=off on a mode that does not select
  them, state=skipped with the reason when selected and not applied);
* run — ``DONE designs=<n> tasks=<t> seeds=<s> skipped_existing=<k> s/design=<x> s/task=<y> load_s=<z> … expected=<N>`` after a design job
  (``INCOMPLETE …`` when the count falls short), or ``NOTHING RAN: …`` when upstream skipped every (task, seed) pair of the job as already
  dumped — no mode claim over outputs an earlier run wrote (`nothing_ran_line`);
* exit — the kit module's own counters of this process (``pxd_xattempt.hoist.stats()``: prepares, hits per replay site, prepare
  seconds), read in memory at interpreter exit; when the lever module was never loaded the tally says so (never silent).
  ``register_exit_tally()`` is called at activation, before the kit module is imported.
"""
from __future__ import annotations

import re
import sys

from opt_core import report as _core

from . import stamps as _stamps

from . import TAG                                         # "pxdesign-opt": the one spelling of the kit's tag (pxdesign_opt/__init__.py)
from .outputs import MANIFEST_NAME as _MANIFEST_NAME

PREFIX = _core.prefix(TAG)
LEVER_MODULE = "pxd_xattempt.hoist"
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = _core.EXIT_OK, _core.EXIT_FAIL, _core.EXIT_USAGE, _core.EXIT_NOT_ACTIVE

gpu_label = _core.gpu_label
_join = _core.join


def _versions(rep: dict) -> str:
    up = rep.get("upstream") or {}
    return f"pxdesign={up.get('pxdesign') or '?'} protenix={up.get('protenix') or '?'} pxdbench={up.get('pxdbench') or '?'}"


def activation_line(rep: dict) -> str:
    vers = _versions(rep)
    notes = ("; ".join(str(n) for n in rep["notes"])) if rep.get("notes") else ""
    tail = f" notes={notes}" if notes else ""                                     # environment facts named, never refused on (stack.environment_notes); absent when there is nothing to name
    if rep.get("mode") == "off":
        head = "DRY-RUN" if rep.get("dry_run") else "STOCK"
        return f"{PREFIX} {head} mode=off stock {vers} gpu={gpu_label(rep.get('gpu'))} levers=none env=none{tail}"
    if rep.get("dry_run"):
        head = "DRY-RUN"
    elif rep.get("active"):
        head = "ACTIVE"
    else:
        return _core.not_active_line(TAG, rep.get("reason") or "no reason recorded", f"mode={rep.get('mode')} {vers} gpu={gpu_label(rep.get('gpu'))}")
    env = ",".join(f"{k}={v}" for k, v in (rep.get("env") or {}).items()) or "none"
    line = (f"{PREFIX} {head} mode={rep.get('mode')} {vers} gpu={gpu_label(rep.get('gpu'))} levers={_join(rep.get('levers_planned') or rep.get('levers_applied'))} "
            f"env={env} applied={rep.get('applied') or 'deferred'} package_levers={_join(rep.get('package_levers'))} tier={rep.get('tier')}")
    if rep.get("dry_run") and rep.get("would_refuse"):
        line += f" would_refuse={rep['would_refuse']!r}"
    return line + tail


def log_activation(rep: dict) -> None:
    _stamps.emit(activation_line(rep))



def applied_line(app: dict) -> str:
    """The hoist's line (levers = the hoist families h1..h5; printed once per model the lever is installed on)."""
    line = f"{PREFIX} APPLIED model#{app.get('index')} levers={_join(app.get('levers_applied'))} fallbacks={_join(app.get('levers_fallback'))}"
    reasons = {k: v for k, v in (app.get("fallback_reasons") or {}).items() if k in (app.get("levers_fallback") or [])}
    if reasons:
        line += f" reasons={reasons}"
    return line


def package_line(app: dict) -> str:
    """The package levers' line: planned, applied, fallbacks, and each lever's evidence field (opt_core.report.kv)."""
    pk = app.get("package") or {}
    line = (f"{PREFIX} PACKAGE model#{app.get('index')} levers={_join(pk.get('planned'))} applied={_join(pk.get('applied'))} fallbacks={_join(pk.get('fallback'))} "
            + _core.kv(**(pk.get("fields") or {})))
    reasons = pk.get("reasons") or {}
    if reasons:
        line += f" reasons={reasons}"
    return line.rstrip()


def log_applied(app: dict) -> None:
    lines = [applied_line(app)] + ([package_line(app)] if (app.get("package") or {}).get("planned") else [])   # the PACKAGE line only when the mode plans a package lever (none ships)
    for line in lines:
        _stamps.emit(line)


def done_line(res: dict) -> str:
    """s/design and s/task are the amortised sampling-phase cost with the one-time load_s beside them; `n/a` when the load is not separable;
    `expected` (the last field) is tasks × seeds × N_sample, the count `designs` is checked against — the first word is ``DONE`` when they
    agree (or the expectation is unknown) and ``INCOMPLETE`` when the count falls short or overshoots (the design verb then exits 1);
    `scope` says what was counted: ``job`` = exactly this job's (task, seed) pairs, ``dir`` = every design under the task directories
    (the job's seeds unknown — a degraded count, named here); `skipped_existing` = the job's pairs upstream skipped as already dumped
    (their designs, written by an earlier run, are inside `designs`; outputs.count_designs)."""
    v = lambda k: "n/a" if res.get(k) is None else res.get(k)  # noqa: E731
    word = "DONE" if res.get("expected") is None or res.get("n_designs") == res.get("expected") else "INCOMPLETE"
    return (f"{PREFIX} {word} mode={res.get('mode')} designs={res.get('n_designs')} tasks={v('n_tasks')} seeds={v('n_seeds')} skipped_existing={v('skipped_existing')} "
            f"s/design={v('s_per_design')} s/task={v('s_per_task')} load_s={v('load_s')} wall_s={res.get('wall_s')} out_dir={res.get('out_dir')} scope={v('scope')} expected={v('expected')}")


def nothing_ran_line(res: dict) -> str:
    """The run line of a job whose every (task, seed) pair upstream skipped as already dumped (its resume rule: `SUCCESS_FILE` present in the
    pair's dump dir): nothing was sampled in this process, so no mode claim is made over the designs on disk — no DONE line, and
    `opt_manifest.json` is left as the earlier run wrote it (``nothing_ran_notes``: what the invocation still wrote there, e.g. the stock
    child's environment proof on ``--mode off``). The exit code is the verb's rule on the designs as found: 0 when they equal tasks × seeds ×
    N_sample, EXIT_FAIL when they do not (short or overshooting; said on the line)."""
    notes = "".join(f"; {n}" for n in (res.get("nothing_ran_notes") or []))
    tail = "" if res.get("complete") else f"; designs on disk != tasks x seeds x N_sample: exit {EXIT_FAIL}"
    return (f"{PREFIX} NOTHING RAN: {res.get('skipped_existing')}/{res.get('n_pairs')} pairs already dumped (upstream resume) — outputs pre-exist from an earlier run "
            f"(designs={res.get('n_designs')} expected={res.get('expected')}); no mode claim made (requested={res.get('mode')}); {_MANIFEST_NAME} not written{notes}; "
            f"out_dir={res.get('out_dir')}{tail}")


DONE_WORDS = ("DONE", "INCOMPLETE")
DONE_RE = re.compile(r"^\[pxdesign-opt\] (?P<word>DONE|INCOMPLETE) mode=(?P<mode>\S+) designs=(?P<designs>\S+) tasks=(?P<tasks>\S+) seeds=(?P<seeds>\S+) .* scope=(?P<scope>job|dir|n/a) expected=(?P<expected>\d+|n/a)$")
KERNELS_RE = _stamps.KERNELS_RE                                                   # the KERNELS line's grammar (stamps.py is the source)
STAMP_EXAMPLES = _stamps.EXAMPLES


def log(msg: str) -> None:
    """One kit line on stderr, on a fresh line (stamps.emit: the writer of every line this module prints)."""
    _stamps.emit(f"{PREFIX} {msg}")


def log_done(res: dict) -> None:
    """The run line: NOTHING RAN when upstream skipped every pair of the job, else DONE / INCOMPLETE."""
    _stamps.emit(nothing_ran_line(res) if res.get("nothing_ran") else done_line(res))


def exit_tally() -> str:
    mod = sys.modules.get(LEVER_MODULE)
    if mod is None:
        return f"{PREFIX} EXIT tally: lever module {LEVER_MODULE} never loaded in this process (no lever applied)"
    try:
        st = mod.stats() if hasattr(mod, "stats") else getattr(mod, "_STATE", {}).get("stats", {})
    except Exception as e:  # noqa: BLE001
        return f"{PREFIX} EXIT tally: {LEVER_MODULE}.stats() failed: {e!r}"
    hits = st.get("hits") or {}
    ps = st.get("prepare_s") or []
    installed = bool(getattr(mod, "_STATE", {}).get("installed"))
    from . import stack
    rep = stack.status()
    partial = f" partial=1 fallbacks={_join((rep.get('levers_fallback') or []) + (rep.get('package_levers_fallback') or []))}" if rep.get("partial") else " partial=0"
    tally = (f"{PREFIX} EXIT tally: installed={installed} prepares={st.get('prepares')} prepare_s_total={round(sum(ps), 3)} "
             f"hits={','.join(f'{k}={v}' for k, v in sorted(hits.items())) or 'none'}{partial}")
    if rep.get("mode") in (None, "off") or not rep.get("active"):
        return tally
    return "\n".join([tally] + stack.lever_lines())


def register_exit_tally() -> bool:
    """Register this kit's EXIT tally line with the core (once per process); True when registered now."""
    return _core.register_exit_tally(TAG, exit_tally)
