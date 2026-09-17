"""The generation run of a kit mode (``exact`` / ``fast`` / ``big``): the stock route's own command with the mode exported into the child.

``run()`` shares every step with ``stock_design.run`` but two. (1) The child's environment is the stock route's cleaned environment PLUS
``COMPLEXA_OPT=<mode>`` and ``COMPLEXA_OPT_RECORD=<out>/kit_records`` (``ENV-KIT`` line: what was removed, what was exported). (2) Before
the launch the autoload hook is proven live in the interpreter upstream's console script runs (``stack.hook_probe``: a ``python -c`` child
in that environment must come up holding ``complexa_opt._autoload`` with its finder armed — ``HOOK ok python=… pth=… finder=armed``;
otherwise nothing is launched: ``NOT ACTIVE: reason=hook_missing …`` names the install step, exit 3 — the levers could not have reached the
generation process and the run would have been stock under a kit mode's name). The child is then the same ``complexa generate`` command
(``stock_design.compose``); in the ``python -m proteinfoundation.generate`` process it spawns, the hook installs the mode's levers and prints
``ACTIVE`` / ``LEVER`` / ``TALLY`` (``activate.py``) into the relayed log output, and leaves ``kit_records/kit_<pid>.json``. After the child
the records are read back (``read_records``: ``KIT-RECORD`` line) and enter the verdict with the census: exit 0 complete and fully active
— a lever idle by its declared input gate (``levers.GATES``: pair_assembly when binder lengths differ within the batches, a binder_length
range) is named on a ``NOTE levers_gated=…`` line, on KIT-RECORD and EXIT and in the manifest, and prices nothing — 1 the child failed or
wrote fewer designs than asked, 3 no generation process activated the mode or a lever of the set never engaged (a partial activation — the
EXIT line names the levers).
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Optional, Sequence

from . import levers as _levers, modes, report as _report, settings as _settings, stack, stock_design as _sd

RECORDS_DIR = "kit_records"


def read_records(records_dir: str, mode: str, lever_set: Sequence[str]) -> dict:
    """Summarise the activation records the generation process(es) left: ``{processes, mode, state, levers_on, levers_gated, levers_skipped,
    gates, predict_steps, forwards, peak_alloc_gib, records, partial}``. Every lever of the set lands in exactly one list: ``levers_on`` (``on``
    in every final record), ``levers_gated`` (``on`` or idle by its declared gate — ``levers.GATES``: pair_assembly on padded batches — in every
    final record, gated in at least one; ``gates`` = ``{lever: {reason, <gate counter>: <sum over records>}}``), else ``levers_skipped``.
    ``state`` = ``absent`` (no final record: the hook never fired or the process died before its exit report), ``partial`` (``partial`` is then
    the priced list: levers skipped, a record of another mode, a record never rewritten final), ``gated`` (nothing priced, a gated lever),
    else ``complete``."""
    recs = []
    for p in sorted(glob.glob(os.path.join(records_dir, "kit_*.json"))):
        try:
            with open(p, encoding="utf-8") as fh:
                recs.append(json.load(fh))
        except (OSError, ValueError):
            recs.append({"path": p, "unreadable": True, "final": False})
    final = [r for r in recs if r.get("final")]
    on, gated, skipped, gates = [], [], [], {}
    for n in lever_set:
        entries = [((r.get("levers") or {}).get(n) or {}) for r in final]
        if final and all(e.get("state") == "on" for e in entries):
            on.append(n)
        elif final and n in _levers.GATES and all(e.get("state") == "on" or (e.get("state") == "skipped" and e.get("reason") == _levers.GATES[n].reason) for e in entries):
            gated.append(n)
            g = _levers.GATES[n]
            gates[n] = {"reason": g.reason, g.counter: sum(int((e.get("counters") or {}).get(g.counter) or 0) for e in entries)}
        else:
            skipped.append(n)
    wrong_mode = [r.get("pid") for r in final if r.get("mode") != mode]
    if not final:
        partial = ["no_activation_record"]
    else:
        partial = list(skipped) + (["record_of_another_mode"] if wrong_mode else []) + (["record_not_final"] if len(final) != len(recs) else [])
    state = "absent" if not final else ("partial" if partial else ("gated" if gated else "complete"))
    peak = max((r.get("peak_alloc_gib") or 0.0 for r in final), default=None)
    return {"processes": len(recs), "final": len(final), "mode": mode, "state": state, "levers_on": on, "levers_gated": gated,
            "levers_skipped": skipped if final else list(lever_set), "gates": gates, "wrong_mode_pids": wrong_mode,
            "predict_steps": sum(int(r.get("predict_steps") or 0) for r in final), "forwards": sum(int(r.get("forwards") or 0) for r in final),
            "peak_alloc_gib": peak, "records": [os.path.relpath(p, os.path.dirname(records_dir)) for p in sorted(glob.glob(os.path.join(records_dir, "kit_*.json")))],
            "partial": partial}


def run(*, mode: str, out_dir: str, input_path: Optional[str] = None, overrides: Sequence[str] = ()) -> dict:
    """One generation run in kit ``mode``. Returns the run record (``exit_code``, census, manifest path, invocation)."""
    t0 = time.time()
    route = modes.route_of(mode)
    lever_set = modes.levers_of(mode)
    values = _settings.values_of(overrides)
    item, entry = (_sd._inputs.load_entry(input_path) if input_path else (None, None))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    w = stack.weights_gate()
    _report.emit(w["line"])
    if not w["pinned"]:
        _report.emit(_report.refused("weights: " + "; ".join(w["bad"]), mode))
        return {"exit_code": _report.EXIT_NOT_ACTIVE, "reason": "weights"}
    cs = stack.console_script()
    cfg_path = stack.pipeline_config()
    records_dir = os.path.join(out_dir, RECORDS_DIR)
    os.makedirs(records_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(records_dir, "kit_*.json")):     # a second run into the same directory: this run's records only
        os.remove(stale)
    exported = {modes.ENV_MODE: mode, modes.ENV_RECORD: records_dir}
    env, removed = _sd.clean_env(export=exported, keep_kit_paths=True)
    probe = stack.hook_probe(env, python=cs.get("interpreter"), cwd=out_dir)
    here = os.path.realpath(stack.package_dir())
    if probe.get("finder_armed") and probe.get("package") and os.path.realpath(probe["package"]) != here:
        _report.emit(_report.refused(f"reason=hook_stale: {probe.get('python')} resolves complexa_opt to {probe['package']}, not this tree's {here} — the levers that would run are not this pin's (run.sh install from this tree)", mode))
        return {"exit_code": _report.EXIT_NOT_ACTIVE, "reason": "hook_stale", "probe": probe}
    if not (probe.get("autoload_loaded") and probe.get("finder_armed")):
        why = (f"reason=hook_missing: {stack.PTH_FILE} did not arm complexa_opt._autoload in {probe.get('python')} under {modes.ENV_MODE}={mode} "
               f"(pth={probe.get('pth') or 'absent'} loaded={probe.get('autoload_loaded')} armed={probe.get('finder_armed')} rc={probe.get('rc')}"
               f"{'; ' + probe['stderr'] if probe.get('stderr') else ''}) — run.sh install puts the kit into the interpreter upstream's `complexa` runs")
        _report.emit(_report.refused(why, mode))
        return {"exit_code": _report.EXIT_NOT_ACTIVE, "reason": "hook_missing", "probe": probe}
    _report.emit(_report.hook_line(probe, mode))
    _report.emit(_report.env_kit_line(removed, exported, probe.get("present")))
    argv = _sd.compose(console_script=cs["path"], config_path=cfg_path, weights_dir=w["dir"], item=item, entry=entry, overrides=overrides)
    eff_item, eff_run, root, before = _sd.prelaunch(out_dir, item, overrides, values)
    inv = _sd._launch(route, item, eff_run, argv, env, out_dir, os.path.join(out_dir, _sd.DESIGN_LOG))
    summary = read_records(records_dir, mode, lever_set)
    _report.emit(_report.kit_record_line(summary))
    for n in summary["levers_gated"]:                                        # a lever idle by its declared input gate: one named line with the count (exit unaffected)
        g = _levers.GATES[n]
        _report.emit(_report.note_line("levers_gated", f"{n}:{g.reason}", f"{g.words} ({g.counter}={summary['gates'][n][g.counter]} calls over {summary['final']} process(es); LEVER line above)"))
    active = summary["state"] in ("complete", "gated")
    rep = {"active": active, "mode": mode, "route": route, "package_version": stack.__version__, "levers": list(lever_set),
           "levers_on": summary["levers_on"], "gated": [f"{n}:{summary['gates'][n]['reason']}" for n in summary["levers_gated"]], "partial": summary["partial"],
           "reason": (None if active else f"activation {summary['state']}")}
    return _sd.finish(mode=mode, route=route, rep=rep, t0=t0, out_dir=out_dir, argv=argv, inv=inv, values=values, item=item, eff_item=eff_item, eff_run=eff_run,
                      root=root, before=before, entry=entry, input_path=input_path, overrides=overrides, w=w, cs=cs, cfg_path=cfg_path,
                      env_block={"path": None, "ok": True, "removed": removed, "exported": exported, "hook": probe}, kit_record=summary)
