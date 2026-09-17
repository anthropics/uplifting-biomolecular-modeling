
"""Observability for mosaic_opt: the activation line, the driver's run line and the exit tally (all on stderr, prefixed ``[mosaic-opt]``).

* activation — ``ACTIVE mode=exact route=driver|in-process row=<tag>[<levers>][-off[<ids>]][-aside[<id>[:<word>],…]] levers=<a,b,...>[ levers_off=<ids>]
  unavailable=<x,...> p1=<dump|load|off>:<dir>|none(<reason>) jax=<v> cuda12-plugin=<v> gpu=<name(smNN)>`` (the in-process route's call-site levers
  step aside by name on the row token, ``-aside[P2:driver_only]``, and are neither ``unavailable=`` nor partial; plus `` partial=<levers>`` when the
  activation is partial — a lever the route should apply and did not: on the driver route the levers the driver's manifest does not show applied — and
  `` allow_partial=1`` when the caller opted to proceed on it, then ``mosaic.fast=<not-installed|kit-bytes@path|ABSENT(...)@path>``: the
  installed lever files against the kit's own set), or ``DRY-RUN ...`` (`` partial=<levers>`` when the plan is partial: the in-process route) /
  ``NOT ACTIVE: <reason>``, formatted from the activation report returned by ``mosaic_opt.enable()`` / ``status()``;
* run — the kit driver's own ``[run] <tag>: ...``, ``PEAK …`` and ``INPUT …`` lines are passed through under the tag; ``design`` adds
  ``DONE`` with the run fields the driver recorded (outputs.py), the same ``partial=`` / ``incomplete=`` fields, and `` forced=upstream_pins(…)``
  (also on ACTIVE as a note) when ``MOSAIC_OPT_FORCE=1`` let an upstream package off its pinned commit through — the override rides every
  line that carries the mode word, never only the WARNING;
* exit tally — once per process at interpreter exit: the mode and route, then P1 as the shared primitive reports it from the P1 directory
  (``jax_cache=<dir|off> autotune=<load|dump|off> via=<…> autotune_sha256=<12 hex|none> cache_entries=<n>``: `opt_core.jax_design.pcc.evidence_fields`);
  a kit mode whose P1 was not applied in the process (switched off by name, or the transparent form stepping aside without
  MOSAIC_OPT_CACHE_ROOT: fast, big) says ``p1=none (…)`` and points at the design process's LEVER lines; when no lever was applied in
  the process (no mode, ``off``) it says so (``jax_cache=off``). `` levers_off=<ids>`` rides ACTIVE / DRY-RUN when the ablation switch
  (``MODEL_OPT_LEVERS_OFF``) names levers of the row.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional

from opt_core.report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE   # noqa: F401  the shared core's exit table (cli re-exports them; _autoload.py's import-free 3 is EXIT_NOT_ACTIVE)
from opt_core.report import PARTIAL_ALLOWED, PARTIAL_REFUSED, join as _join
from opt_core.report import register_exit_tally as _core_register_exit_tally

from . import TAG                                                        # the kit's one tag spelling (the package root, import-free)

PREFIX = f"[{TAG}]"
PARTIAL_HEAD = "NOT ACTIVE: partial activation — "                      # the shared core's partial-exit grammar: the literal parts of both lines
PARTIAL_TAIL = "(--allow-partial records and proceeds)"
ALLOWED_HEAD = "PARTIAL allowed: "
ALLOWED_TAIL = "(--allow-partial, recorded)"
_TALLY: dict = {"fn": None}                                             # the exit tally's composer: the last registration wins (register_exit_tally)


def gpu_label(gpu) -> str:
    """`<name>(<smNN>,<MiB>MiB)`: the card and its memory size, both recorded on every activation line (the memory size is what tells an H100
    80GB from an H100 NVL)."""
    if isinstance(gpu, dict):
        name = gpu.get("name")
        sm = gpu.get("sm")
        mem = gpu.get("memory_mib")
        if not name:
            return "none"
        inner = ",".join(x for x in (sm or "", f"{mem}MiB" if mem else "") if x)
        return f"{name}({inner})" if inner else str(name)
    return str(gpu) if gpu else "none"


def _versions(rep: dict) -> str:
    up = rep.get("upstream") or {}
    return f"jax={up.get('jax')} cuda12-plugin={up.get('jax-cuda12-plugin')} mosaic={up.get('mosaic')}"


def _p1(rep: dict) -> str:
    """`p1=<load|dump|off>:<dir>` (off = the transparent form: the compilation cache alone, no autotune pin), `p1=none(<reason>)` when P1
    steps aside by name (the transparent form without MOSAIC_OPT_CACHE_ROOT), `p1=none` when the row carries no P1."""
    p1 = rep.get("p1") or {}
    if not p1.get("cache_dir"):
        why = p1.get("aside") or (rep.get("levers_aside") or {}).get("P1")
        return f"p1=none({str(why).replace(' ', '_')})" if why else "p1=none"
    return f"p1={p1.get('autotune') or '?'}:{p1.get('cache_dir')}"


def _levers_off(rep: dict) -> str:
    """`` levers_off=<ids>`` when the ablation switch (MODEL_OPT_LEVERS_OFF) names levers of the row; empty otherwise."""
    off = rep.get("levers_off") or []
    return f" levers_off={_join(off)}" if off else ""


def _fast(rep: dict) -> str:
    """` mosaic.fast=<label>`: the installed lever files (site-packages/mosaic/fast/) checked against the kit's own file set —
    `stack.installed_fast_check` worded by `stack.installed_fast_label`; absent from the line when the report carries no check result (an early refusal)."""
    fc = rep.get("kit_install")
    if not fc:
        return ""
    from .stack import installed_fast_label
    return f" mosaic.fast={installed_fast_label(fc)}"


def partial_field(rep: dict | None) -> str:
    """`` partial=<levers>`` when the activation (or the dry-run plan) is partial — on the in-process route with the reason the lever was
    not applied — then `` allow_partial=1`` when the caller opted to proceed on it; empty when whole."""
    rep = rep or {}
    levers = partial_levers(rep)
    if not levers:
        return ""
    s = f" partial={_join(levers)}"
    if rep.get("levers_unavailable"):                                                       # the in-process route: a lever it should apply and did not, with its reason (a call-site lever stepping aside
        s += f" (not applied on this route: {rep.get('unavailable_reason') or 'no reason recorded'})"   # by name is not here: it rides row=…-aside[P2:driver_only])
    elif rep.get("levers_fallback"):
        s += " (the driver's manifest does not show them applied)"
    if rep.get("allow_partial"):
        s += " allow_partial=1"
    return s


def activation_line(rep: dict | None) -> str:
    rep = rep or {}
    mode = rep.get("mode")
    head = f"mode={mode} route={rep.get('route') or 'none'}"
    if rep.get("active"):
        line = (f"{PREFIX} ACTIVE {head} row={rep.get('row_line') or 'none'} levers={_join(rep.get('levers_applied'))}{_levers_off(rep)} "
                f"unavailable={_join(rep.get('levers_unavailable'))} {_p1(rep)} {_versions(rep)} gpu={gpu_label(rep.get('gpu'))}")
        return line + partial_field(rep) + _fast(rep) + _notes(rep)
    reason = rep.get("reason") or "no reason given by mosaic_opt.enable()"
    if rep.get("dry_run") and rep.get("row_line"):
        tg = rep.get("target_gpu")
        st = rep.get("shape") or {}
        shape = f" shape={st.get('key')}({st.get('tokens')} tokens)" if st.get("key") else ""
        feats = f" features={'present' if (rep.get('features') or {}).get('present') else 'absent'}" if rep.get("route") == "driver" and rep.get("mode") != "off" else ""
        return (f"{PREFIX} DRY-RUN {head} row={rep.get('row_line')}{shape} "
                f"levers={_join(rep.get('levers_planned'))}{_levers_off(rep)} {_p1(rep)}{feats} "
                f"{_versions(rep)} gpu={gpu_label(rep.get('gpu'))}" + (f" target_gpu={tg}" if tg else "")
                + f" env_vars={len(rep.get('env') or {})}" + partial_field(rep) + _fast(rep)
                + (f" would_refuse={rep['would_refuse']!r}" if rep.get("would_refuse") else "")
                + _notes(rep))
    return f"{PREFIX} NOT ACTIVE: {reason}" + (f" ({head})" if mode else "")


def _notes(rep: dict) -> str:
    """`` notes=<note>; <note>…`` — what the environment differs in (the stack's versions against the kit's pins, a target-GPU mismatch, extra
    files beside the installed lever files, a cold shape): named on the activation line, never a reason to refuse; empty when there is none."""
    return (" notes=" + "; ".join(str(n) for n in rep["notes"])) if rep.get("notes") else ""


def log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)




ENV_ALLOW_PARTIAL = "MOSAIC_OPT_ALLOW_PARTIAL"                   # =1: the in-process route's opt-out of the partial verdict (MOSAIC_OPT=<mode> in a program's environment,
                                                               # where the environment is that route's only surface); the CLI verbs read --allow-partial only and ignore it


def allow_partial_env(environ: Optional[dict] = None) -> bool:
    """``MOSAIC_OPT_ALLOW_PARTIAL=1`` in the environment of the in-process route (``MOSAIC_OPT=<mode>``, ``enable()``): a partial activation
    is recorded and the program proceeds. Read by that route only."""
    environ = os.environ if environ is None else environ
    return (environ.get(ENV_ALLOW_PARTIAL) or "").strip().lower() in ("1", "true", "yes", "on")


def partial_levers(rep: Optional[dict]) -> List[str]:
    """The levers of a PARTIAL activation (``partial`` on the report): the ones that fell back on the driver route (``levers_fallback``:
    requested by the row, not shown applied by the driver's manifest) or the levers the in-process route should apply and did not
    (``levers_unavailable``; the driver's call-site levers are not among them — they step aside by name, ``levers_driver_only``); [] when
    the activation is whole."""
    rep = rep or {}
    if not rep.get("partial"):
        return []
    return list(rep.get("levers_fallback") or rep.get("levers_unavailable") or [])


def allow_partial(flag: bool = False) -> bool:
    """``--allow-partial``: the CLI verbs' opt-out of the partial verdict (the environment word is not read here)."""
    return bool(flag)


def partial_detail(rep: dict, what: str) -> str:
    """``<detail>`` of the partial verdict: the lever names first, then the reason — the in-process plan's own (``unavailable_reason``)
    or the driver's manifest not showing them applied (``what``: where its results file is)."""
    levers = ",".join(partial_levers(rep))
    if rep.get("levers_unavailable") and not rep.get("levers_fallback"):
        return f"{levers} — the plan is partial on this box ({rep.get('unavailable_reason') or 'not applied'})"
    return f"{levers} — the driver's manifest does not show them applied ({what})"


def exit_for(rc: int, failed: bool, rep: dict, allowed: bool, what: str = "design") -> int:
    """The verbs' exit from the driver's exit status and the activation report: EXIT_FAIL when the driver failed, was killed or left the
    outputs incomplete (``failed``), EXIT_NOT_ACTIVE when the activation is partial (``rep['partial']``: levers of the row the driver's
    manifest does not show applied, or the in-process plan's call-site levers) and not allowed, else EXIT_OK. Records ``allow_partial``
    on the report; prints the ONE partial verdict (report.partial_verdict_line)."""
    rep["allow_partial"] = bool(allowed)
    if failed or rc != 0:
        return EXIT_FAIL
    if partial_levers(rep):
        log(partial_verdict_line(partial_detail(rep, what), allowed, EXIT_NOT_ACTIVE))
        if not allowed:
            return EXIT_NOT_ACTIVE
    return EXIT_OK


def partial_verdict_line(detail: str, allowed: bool, exit_code: int = EXIT_NOT_ACTIVE) -> str:
    """The ONE partial-activation verdict (every verb prints it through here; ``detail``: the lever names first, then the reason):
    refused — ``<PREFIX> NOT ACTIVE: partial activation — <detail>; exit 3 (--allow-partial records and proceeds)``;
    allowed — ``<PREFIX> PARTIAL allowed: <detail> (--allow-partial, recorded)``."""
    if allowed:
        return PARTIAL_ALLOWED.format(prefix=PREFIX, detail=detail)
    return PARTIAL_REFUSED.format(prefix=PREFIX, detail=detail, exit_code=exit_code)


def log_activation(rep: dict) -> None:
    log(activation_line(rep))


DRIVER_LINE_PREFIXES = ("[run]", "PEAK ", "INPUT ")  # the driver's own summary lines the package relays under its tag: `[run] <tag>: …`, the report-only `PEAK item=<tag> peak_bytes_in_use_gib=<x> …` and the input-option lines `INPUT target_fasta records=<n> used=1 (--first-record)` / `INPUT epitope residues=<spec> …` (tools/recipe.py input_lines)


def run_line(text: str) -> str:
    """The driver's `[run] ...` / `PEAK ...` line, passed through under the package tag."""
    return f"{PREFIX} {text.strip()}"


def relay(text: str) -> str:
    """A driver output line as the package logs it: its summary lines (DRIVER_LINE_PREFIXES) under the tag, every other line verbatim."""
    return run_line(text) if text.startswith(DRIVER_LINE_PREFIXES) else text


def forced_field(rep: dict | None) -> str:
    """`` forced=upstream_pins(<package>,…)`` when MOSAIC_OPT_FORCE=1 let upstream off its pinned commit through (stack.forced_word); empty otherwise."""
    return f" forced={rep['forced']}" if (rep or {}).get("forced") else ""


def done_line(mode: str, run: dict, out_dir: str, rc: int, rep: dict | None = None, incomplete: str | None = None) -> str:
    return (f"{PREFIX} DONE mode={mode} rc={rc} loss_traj_sha256={str(run.get('loss_traj_sha256'))[:16]} best2={run.get('best2_sha16')} "
            f"seq={run.get('seq_stage2_best')} iptm={run.get('refold_iptm')}" + partial_field(rep) + forced_field(rep)
            + (f" incomplete={incomplete}" if incomplete else "") + f" out={out_dir}")


def tally_line(mode: Optional[str], route: Optional[str], cache_dir: Optional[str], autotune: Optional[str] = None, via: Optional[str] = None,
               why_none: Optional[str] = None) -> str:
    """The exit tally: which mode and route this process ran, then P1 as the shared primitive reports it —
    `opt_core.report.kv(**opt_core.jax_design.pcc.evidence_fields(record))` = ``jax_cache=<dir> autotune=<load|dump|off> via=<…> autotune_sha256=<12 hex|none>
    cache_entries=<n>`` read from the P1 directory at exit (``autotune``: the phase the row ran — ``off`` for the transparent form, the cache
    without the pin — else read from the directory; ``via``: the route's record, ``exports`` for the driver route); ``cache_dir`` None with a kit
    mode = P1 not applied in this process (``mode=<m> route=<r> p1=none (…) jax_cache=off …``: switched off by name, or the transparent form
    stepping aside — ``why_none``, e.g. MOSAIC_OPT_CACHE_ROOT unset — the mode's per-step levers reported on the design process's LEVER lines),
    and with no mode or ``off`` = no lever applied in this process (``jax_cache=off …``) — never silence."""
    from opt_core.jax_design import pcc
    from opt_core.report import kv
    if not cache_dir:
        if mode in (None, "off"):
            return f"{PREFIX} EXIT pid={os.getpid()} no lever applied in this process " + kv(**pcc.evidence_fields(None))
        why = f"P1 steps aside: {why_none}" if why_none else "P1 not applied in this process"
        return (f"{PREFIX} EXIT pid={os.getpid()} mode={mode} route={route} p1=none ({why}; the mode's per-step levers report on the design process's LEVER lines) "
                + kv(**pcc.evidence_fields(None)))
    phase = autotune if autotune in ("load", "dump", "off") else ("load" if os.path.isfile(pcc.autotune_file_of(cache_dir)) else "dump")
    record = {"cache_dir": cache_dir, "autotune": phase, "applied_via": via or ("exports" if route == "driver" else None)}
    return f"{PREFIX} EXIT pid={os.getpid()} mode={mode} route={route} P1 " + kv(**pcc.evidence_fields(record))


def register_exit_tally(fn: Callable[[], str]) -> None:
    """Print `fn()` once at interpreter exit through the core's tally (`opt_core.report.register_exit_tally`, one registration per tag,
    never silent); the last `fn` registered here is the one composed at exit."""
    _TALLY["fn"] = fn
    _core_register_exit_tally(TAG, _compose_tally)


def _compose_tally() -> str:
    fn = _TALLY["fn"]
    return fn() if fn is not None else tally_line(None, None, None)

