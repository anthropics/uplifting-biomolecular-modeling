"""The evidence lines: one format string per line, all on stderr, prefixed ``[colabfold-opt]`` (``[colabfold-opt stock]`` for the stock
route). The activation report dict (stack.py) is the one source of every field.

  ACTIVE     the mode's levers are on in this process, printed once at activation (before the stock ``run``)
  DRY-RUN    `check`: the same fields as ACTIVE, nothing applied
  NOT ACTIVE the mode could not be activated — the process that requested it exits 3. A mode is all of its levers: a lever of the mode
             that cannot run on this stack (no tile table for the part, its requirement raised, its patch did not land) refuses the mode by
             name (`NOT ACTIVE: <LEVER>=<word>: …; exit 3 (--mode off runs stock)`, LEVER_REFUSED_FMT / `NOT ACTIVE: enable() failed:
             <LEVER>: <why>`) — never a run under the mode's name with a subset of its levers; a lever applied whose kernel no call of the run
             reached is a PARTIAL activation: the line `NOT ACTIVE: partial activation — <levers>: <reason>; exit 3`
             (PARTIAL_EXIT_FMT: `pred`'s `kernel_not_engaged` verdict, the strict hook's own at exit). The environment is never a refusal:
             the GPU's compute capability, the jax version, another GPU model than the configuration's, a compile-cache miss are notes on
             the ACTIVE line (`notes=…`) and every lever engages. A kernel that engaged but fell back beyond the documented class prints
             `PARTIAL fallback_excess: <detail>; the calls that fell back ran the stock operation — recorded, exit 0` (FALLBACK_EXCESS_FMT)
             and never changes the exit code
  IDLE       colabfold_batch built no model on this run — `--num-models 0` / `--msa-only` (`no_model_run=num_models_0`), every job of
             the input already complete in the results directory and kept (`all_jobs_done`: colabfold skips finished jobs), `--af3-json`
             (`af3_json`: it returns before its `run()`): the mode's applied levers had no model call to serve (`levers=not_applicable:<names>`;
             `levers=none` when nothing was applied) — by name, not a partial activation; the run exits as stock does (IDLE_FMT: `pred`'s
             verdict in every mode, the strict hook's own line when it is the entry point). The idle levers' LEVER lines carry `no_model_run=<case>`
  EXIT       at interpreter exit, one line per lever: the carried kit's counters, ``af2_pallas_attn._STATE`` (:16), its fields
             verbatim in the kit's order (`EXIT _STATE ...`, in the modes applying the kernel)
  LEVER      at interpreter exit, one line per lever of the table (registry.LEVERS in a mode): `LEVER name=<lever> state=on <counters>`
             (applied: DEVICE_RESIDENT's calls/uploads/hits/fetched_bytes, AF_PALLAS_ATTN's calls/fallbacks), `state=off reason=not_in_mode`
             (the mode does not carry it), `state=skipped reason=<why>` (in the mode, not applied: a refusal, a card fallback)
  STOCK      the stock route's proof, printed by the launcher before ``colabfold_batch`` starts
  FAILED     the `pred` verdict after the model process exits (manifest.verdict): reason=missing_outputs | model_process_failed | killed,
             then the detail (a kit mode without its launch's activation report is NOT ACTIVE, rc 3; a partial activation prints the
             partial-activation line above)

`tokens=<min>-<max>` on the ACTIVE / DRY-RUN lines is the run's size: residues of the complex including every copy of a chain (Σ length ×
cardinality from the a3m header; inputs.tokens), the smallest and largest query of the run (`none` on a dry run: no queries).
`not_wired=<names>` is appended to the ACTIVE line when a kit switch no mode sets (registry.NOT_WIRED) is on in the model process.
"""
from __future__ import annotations

import atexit
import json
import sys
from typing import Optional

from opt_core import report as _core_report


PREFIX = "[colabfold-opt]"
STOCK_PREFIX = "[colabfold-opt stock]"

ACTIVE_FMT = "{prefix} ACTIVE mode={mode} levers={levers} tokens={tokens} queries={queries} colabfold={colabfold} alphafold_colabfold={alphafold_colabfold} jax={jax} key={key} gpu={gpu}"
DRY_RUN_FMT = "{prefix} DRY-RUN mode={mode} levers={levers} tokens={tokens} queries={queries} colabfold={colabfold} alphafold_colabfold={alphafold_colabfold} jax={jax} key={key} gpu={gpu}"
NOT_ACTIVE_FMT = "{prefix} NOT ACTIVE: {reason}"
ABLATED = "ablated"                                        # the ACTIVE / DRY-RUN token and the LEVER line's reason for a lever MODEL_OPT_LEVERS_OFF removed from the mode (ablation.TOKEN / .REASON)
PARTIAL_EXIT_FMT = "{prefix} NOT ACTIVE: partial activation — {detail}; exit {code}"   # the line of a partial activation (the fixed parts byte-literal)
FALLBACK_EXCESS_FMT = "{prefix} PARTIAL fallback_excess: {detail}; the calls that fell back ran the stock operation — recorded, exit 0"   # the kernel engaged and fell back beyond the documented class: named once, never an exit
LEVER_REFUSED_FMT = "{prefix} NOT ACTIVE: {detail}; exit {code} (--mode off runs stock)"   # a lever of the mode that cannot run on this GPU: a mode is all of its levers — the mode refuses the run by name
IDLE_FMT = "{prefix} IDLE no_model_run={case} levers={levers} jobs={jobs}: {detail}; exit {code}"   # colabfold_batch built no model (manifest.NO_MODEL_RUN_CASES): the applied levers had no call to serve — by name, never partial, the run's own exit code
IDLE_LEVERS = "not_applicable"                             # the IDLE line's levers= word ahead of the idle levers' names (`levers=not_applicable:<a>,<b>`; `levers=none` when no lever was applied)
EXIT_NOT_ACTIVE = _core_report.EXIT_NOT_ACTIVE                                                    # the named exit code of a refused or partial activation (cli.EXIT_NOT_ACTIVE, stack.hook_run, _autoload.py)
OFF_FMT = "{prefix} NOT ACTIVE mode=off (stock: nothing applied)"
EXIT_FMT = "{prefix} EXIT _STATE {fields}"
EXIT_NONE_FMT = "{prefix} EXIT _STATE none: {module} was not imported in this process"
TAG = "colabfold-opt"                                                                               # PREFIX = opt_core.report.prefix(TAG)
LEVER_VERB = _core_report.LEVER                                                                     # the per-lever line: `[colabfold-opt] LEVER name=<lever> state=<on|off|skipped> [reason=<why>] impl=<module> origin=<kit|core> strategy=<canonical id> <evidence k=v>` (opt_core report.lever_line)
STOCK_FMT = "{prefix} STOCK cli={cli} env_prefixes_absent={prefixes} kit_dirs={kit_dirs} proof={proof}"
FAILED_FMT = "{prefix} FAILED reason={reason} {detail}"

_TALLY = {"registered": False, "state": None, "module": None, "on_exit": None, "levers": {}}   # levers: name -> its _STATE dict (LEVER_EXIT_FMT lines)


def slug(s: Optional[str]) -> str:
    """A line field carries no whitespace: `NVIDIA H100 80GB HBM3` -> `NVIDIA_H100_80GB_HBM3`."""
    return "none" if not s else "".join(c if (c.isalnum() or c in ".+-|") else "_" for c in str(s).strip())


def _fields(rep: dict) -> dict:
    lo, hi = rep.get("tokens_min"), rep.get("tokens_max")
    return {"prefix": PREFIX, "mode": rep.get("mode"), "levers": ",".join(rep.get("levers_applied") or rep.get("levers") or []) or "none",
            "tokens": f"{lo}-{hi}" if lo is not None else "none",
            "queries": rep.get("queries") if rep.get("queries") is not None else "none", "colabfold": rep.get("colabfold_version"),
            "alphafold_colabfold": rep.get("alphafold_colabfold_version"), "jax": rep.get("jax_version"), "key": rep.get("key") or "none",
            "gpu": slug((rep.get("gpu") or {}).get("name")), "reason": rep.get("reason")}


def activation_line(rep: dict, dry_run: bool = False) -> str:
    """One of the format strings above, filled from the report; `notes=...` (the environment, named: another GPU than the configuration's, a
    compute capability or a jax outside the tested ones) and, on a dry run, `would_refuse=...` are appended after the last field when present."""
    tail = (" " + _core_report.kv(*rep["n_gpu_fields"])) if rep.get("n_gpu_fields") else ""   # big: `n_gpu=P sharding=rowpair|none` (opt_core.mem.rowpair_jax.evidence.active_fields)
    if rep.get("levers_ablated"):                                        # MODEL_OPT_LEVERS_OFF removed levers from the mode for this run (ablation.py): `ablated=<a>,<b>` in request order; absent otherwise, so an un-ablated run's lines read exactly as before
        tail += f" {ABLATED}=" + ",".join(rep["levers_ablated"])
    tail += (" notes=" + "; ".join(rep["notes"])) if rep.get("notes") else ""
    on = sorted(n for n, v in (rep.get("switches_not_wired") or {}).items() if v)
    if on:
        tail += " not_wired=" + ",".join(on)
    for n, w in (rep.get("lever_fallbacks") or {}).items():             # a table-backed lever held on this part, by name: <LEVER>=fallback:no_tiles_cc<NN>(<kind>)
        tail += f" {n}={w}"
    if rep.get("mode") == "off":
        return OFF_FMT.format(prefix=PREFIX)
    if dry_run:
        if rep.get("would_refuse"):
            tail += f" would_refuse={rep['would_refuse']!r}"
        return DRY_RUN_FMT.format(**_fields(rep)) + tail
    if rep.get("active"):
        return ACTIVE_FMT.format(**_fields(rep)) + tail
    return NOT_ACTIVE_FMT.format(prefix=PREFIX, reason=rep.get("reason"))


def partial_exit_line(detail: str) -> str:
    """The family line of a partial activation that exits by name (every entry point: `pred`'s verdict, the strict hook at the model
    process's exit; `warm` prints it through `pred`): the lever names first, then the engine's reason, in `detail`."""
    return PARTIAL_EXIT_FMT.format(prefix=PREFIX, detail=detail, code=EXIT_NOT_ACTIVE)


def idle_line(case: str, idle, jobs, detail: str, code: int = 0) -> str:
    """The ONE line of a run on which colabfold_batch built no model (manifest.no_model_run: `num_models_0` | `all_jobs_done` | `af3_json`):
    the case by name, the applied levers left without a call (`levers=not_applicable:<names>`, `none` when nothing was applied — `--mode off`,
    or a hook never entered), the job count, what colabfold_batch did instead in `detail`, and the run's own exit code. Printed by the process
    that reads the facts: `pred` from its verdict (every mode), the model process itself when it is the entry point (the env route on its own)."""
    names = list(idle or [])
    return IDLE_FMT.format(prefix=PREFIX, case=case, levers=(IDLE_LEVERS + ":" + ",".join(names)) if names else "none",
                           jobs=jobs if jobs is not None else "none", detail=detail, code=code)


def fallback_excess_line(detail: str) -> str:
    """The ONE line of a run whose kernel engaged but fell back to the stock operation beyond the documented class of calls
    (manifest.excess_fallbacks, `fallback_excess`): the levers, the counters and the class in `detail`; the run keeps its own verdict and exit
    code (those calls computed what stock computes).  Printed by the process that reads the counters: `pred` from its verdict, the model
    process itself when it is the entry point."""
    return FALLBACK_EXCESS_FMT.format(prefix=PREFIX, detail=detail)


def lever_fallback_detail(rep: dict) -> str:
    """`detail` of a mode whose lever(s) cannot run on this GPU: `<LEVER>=<word>[,…]: the lever cannot run on this GPU (no tile table
    for its compute capability) — a mode is all of its levers`."""
    fb = rep.get("lever_fallbacks") or {}
    return (",".join(f"{n}={w}" for n, w in fb.items()) or "none") + ": the lever cannot run on this GPU (no tile table for its compute capability) — a mode is all of its levers"


def lever_refused_line(rep: dict) -> str:
    """The ONE line of a mode refused because a lever cannot run on this GPU: the lever(s) and their words, the exit code, the one other
    route (`--mode off` runs stock) — never a run under the mode's name with a subset of its levers."""
    return LEVER_REFUSED_FMT.format(prefix=PREFIX, detail=lever_fallback_detail(rep), code=EXIT_NOT_ACTIVE)


def stock_line(proof: dict) -> str:
    return STOCK_FMT.format(prefix=STOCK_PREFIX, cli=proof["cli"], prefixes=",".join(proof["prefixes"]),
                            kit_dirs=",".join(proof["kit_dirs"]) or "none", proof="ok" if proof["ok"] else "FAILED")


def failed_line(v: dict) -> str:
    return FAILED_FMT.format(prefix=PREFIX, reason=v.get("reason"), detail=v.get("detail") or "")


def exit_line(state: Optional[dict], module: str) -> str:
    """The kit's ``_STATE`` dict as `k=v` pairs in its own key order (af2_pallas_attn.py:16: enabled, calls, fallbacks)."""
    if state is None:
        return EXIT_NONE_FMT.format(prefix=PREFIX, module=module)
    return EXIT_FMT.format(prefix=PREFIX, fields=" ".join(f"{k}={v}" for k, v in state.items()))


def lever_line(name: str, state: str, reason: Optional[str] = None, **evidence) -> str:
    """The per-lever evidence line (one per lever in each process that runs a kit mode, printed at interpreter exit with the lever's final counters) —
    the core's `report.lever_line` bound to this kit's tag and the registry's impl / origin of the lever: `state` on = applied and live
    in this process (its counters follow), off = not in this mode's lever set, skipped = in the set but not applied (`reason` mandatory:
    a refusal, a card fallback)."""
    from . import registry as _registry
    lv = _registry.LEVERS.get(name)
    if state != "on" and not reason:
        reason = "none"
    return _core_report.lever_line(TAG, name, state, reason=reason, impl=lv.impl if lv else None, origin=lv.origin if lv else None,
                                   strategy=(lv.strategy or None) if lv else None, **evidence)


def line(tag: str, **kv) -> str:
    return " ".join([PREFIX, tag] + [f"{k}={v}" for k, v in kv.items()])


def emit(text: str, stream=None) -> None:
    (stream or sys.stderr).write(text + "\n")
    (stream or sys.stderr).flush()


def print_activation(rep: dict, dry_run: bool = False) -> None:
    emit(activation_line(rep, dry_run))


def dump(rep: dict) -> str:
    return json.dumps(rep, indent=1, sort_keys=True, default=str)


# ------------------------------------------------------------------------------------------------------------- exit tally
def _print_exit_tally() -> None:
    lines = []
    if _TALLY["module"] is not None:                                     # the carried kit's _STATE line: only when a mode applies the kernel
        try:
            lines.append(exit_line(_TALLY["state"], _TALLY["module"]))
        except Exception as e:  # noqa: BLE001
            lines.append(f"{PREFIX} EXIT tally failed: {e!r}")
    for name, (state, reason, st) in _TALLY["levers"].items():             # one LEVER line per lever of the table, its live counters read now
        try:
            st = st() if callable(st) else st
            if isinstance(st, str):                                       # the lever rendered its own line (the n_gpu axis: opt_core.mem.rowpair_jax.evidence.line)
                lines.append(st)
                continue
            evidence = {k: v for k, v in st.items() if k != "enabled"} if (state == "on" and isinstance(st, dict)) else {}
            lines.append(lever_line(name, state, reason, **evidence))
        except Exception as e:  # noqa: BLE001
            lines.append(f"{PREFIX} {LEVER_VERB} name={name} tally failed: {e!r}")
    for text in lines:
        try:
            emit(text)
        except Exception:  # noqa: BLE001
            pass
    if _TALLY["on_exit"]:
        try:
            _TALLY["on_exit"](_TALLY["state"])
        except Exception:  # noqa: BLE001
            pass


def register_exit_tally(state: Optional[dict], module: Optional[str], on_exit=None, levers: Optional[dict] = None) -> None:
    """Print the EXIT line(s) at interpreter exit, once per process. ``state``: the carried kit's ``_STATE`` dict itself (read in memory at
    exit, so the counters are the final ones); None when the kit was not imported. ``module``: its module name — None when the mode does
    not apply the kernel (no `_STATE` line then). ``levers``: lever name -> (state, reason, its ``_STATE`` dict or None), one LEVER line
    each (lever_line). ``on_exit(state)``: a callback (the manifest's exit record)."""
    _TALLY["state"], _TALLY["module"] = state, module
    _TALLY["levers"] = dict(levers or {})
    if on_exit is not None:
        _TALLY["on_exit"] = on_exit
    if _TALLY["registered"]:
        return
    _TALLY["registered"] = True
    atexit.register(_print_exit_tally)


def tally_registered() -> bool:
    return bool(_TALLY["registered"])


def reset_tally_for_tests() -> None:
    """Unregister the exit tally (in-process activations in a test run would print it after the run's summary)."""
    if _TALLY["registered"]:
        atexit.unregister(_print_exit_tally)
    _TALLY.update(registered=False, state=None, module=None, on_exit=None, levers={})
