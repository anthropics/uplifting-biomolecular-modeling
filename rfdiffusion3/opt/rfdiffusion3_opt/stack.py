"""Activation: kit bytes, interpreter state, pins, GPU, the environment, and the export of the mode's delta.

`activate(mode)` resolves the mode in the one table (modes.resolve), gates it — the kit's tree complete (registry.check_tree), this
interpreter's ten target files in the kit's `patched` state (registry.classify; the kit's install.sh is the only way to get there), a
visible GPU (nvidia-smi, no torch), and an environment that does not contradict the mode (no lever name set to another value) — then
exports the delta into os.environ. The upstream distribution's pin (stock/check_pins.py: install source / commit, foundry/utils/torch.py)
is REPORTED (`pins`, one `PINS …` note line per finding), never a gate: a checkout upstream itself runs is not refused. The deterministic
recipe's names (FOUNDRY_DET_SCATTER, CUBLAS_WORKSPACE_CONFIG) are the caller's and are reported (`det_env`), never refused. A design
line the mode cannot serve — upstream's `compile_model=true` (routes.COMPILE_REFUSED), classifier-free guidance, the low-memory
tokenization or the symmetry sampler (modes.UNSERVED) — is refused by name (`NOT ACTIVE: …`, line_refusal): nothing is exported and nothing of the run starts;
`--mode off` serves it. That is the whole application: the kit's `rfd3.model.hoist` reads `RFD3_HOIST`
when it is imported (hoist.py:18), so the export must precede that import. `activate()` is refused by name once that module is in
sys.modules (late activation), when it reports a lever state other than the mode's, and for a second mode in one process; it is
idempotent for the same mode. `activate(mode, dry_run=True)` gates and reports without exporting (`check`), and imports nothing of
the upstream stack.

The report names what was checked (`interpreter`, `kit`, `pins`, `gpu`, `env`) and, after export, `applied="at-import"` with the
core's shared report keys — `levers_applied` (the delta's names, `["RFD3_HOIST"]`), `levers_fallback`, `levers_unavailable`,
`partial`, `partial_reason`, `allow_partial` — empty / False / None before export and for `off`; the `APPLIED` line (report_applied,
called by the autoload hook or by `design` after the import) shows what the kit's module actually read. A module that read the lever
off, or a completed run that never imported it, is a PARTIAL activation: the lever moves to the fallback keys with the reason, and the
exit rule (report.verdict, report.partial_line) refuses the run — exit 3 — unless `--allow-partial` / RFDIFFUSION3_OPT_ALLOW_PARTIAL=1
records the allowance (`allow_partial`). `line_refusal` is the gate on the design line's route: upstream's `compile_model=true` under
a kit mode has no route (modes.COMPILE_REFUSED names why; `--mode off` runs that line as stock), exit 3.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

from opt_core.gates import dist_version, nvidia_smi_probe

from . import ActivationError, __version__
from .census import ALLOC_CONF_ENV                                             # torch's allocator variable, one spelling (the ACTIVE and KERNELS lines' alloc_conf=)
from . import det as _det, census, modes, registry
from . import report as _report
from ._autoload import ENV_NAMES, core_gate

PACKAGE_ENV = ENV_NAMES                                                   # the package's own switches (_autoload.ENV_NAMES): never seen by the stock arm's child
DET_RECIPE_NAMES = ("FOUNDRY_DET_SCATTER", "CUBLAS_WORKSPACE_CONFIG")     # the deterministic recipe: the caller's choice (torch's / foundry's names), tolerated under both kit modes and reported (rep['det_env'])s; the stock arm's child strips them like every must-be-absent name
_STATE: Dict[str, object] = {"report": None, "hook": None}


pins = registry.pins                                                      # stock/PINS.json (registry: the one reader, shared with the stock caller)
_check_pins_module = registry.check_pins_module


def lever_names() -> Tuple[str, ...]:
    """Every environment name the stock arm must not see (stock/PINS.json "stock_environment.must_be_absent")."""
    return tuple(pins()["stock_environment"]["must_be_absent"])


def upstream_switches() -> Tuple[str, ...]:
    return tuple(pins()["stock_environment"]["upstream_switches_unset"])


def pin_check(expect: str = "any") -> Tuple[List[str], dict]:
    """stock/check_pins.py in this interpreter: (bad, detail)."""
    cp = _check_pins_module()
    return cp.check(pins(), expect)


def foundry_version() -> Optional[str]:
    """The installed upstream distribution's version (opt_core.gates.dist_version), None when it is not installed."""
    return dist_version(pins()["upstream"]["foundry"]["distribution"])


def gpu_info() -> Optional[dict]:
    """The first visible GPU from nvidia-smi — the core's probe (name, cc, sm, memory_mib) without importing torch; None when there is none."""
    g = nvidia_smi_probe()
    if g.get("probe") != "nvidia-smi" or not g.get("name"):
        return None
    return {"name": g["name"], "compute_cap": g["cc"], "sm": (g["cc"] or "").replace(".", "") or None, "memory_mib": g["memory_mib"], "probe": g["probe"]}


def interpreter_state(site: Optional[str] = None) -> dict:
    """The ten target files' states in this interpreter (registry.classify) with the summary label."""
    states = registry.classify(site)
    label, detail = registry.summarize(states)
    return {"label": label, "detail": detail, "states": states, "site": site or registry.site_dir(), "python": sys.executable}


def environment_findings(res: modes.Resolution) -> Tuple[List[str], Dict[str, str]]:
    """(refusals, reported): lever names already in the environment that contradict the mode, and the caller's deterministic-recipe names that are set.
    A pre-set switch agrees with the mode only in the mode table's literal spelling (`RFD3_HOIST=1`, modes.KIT_MODES): the kit's own reader
    takes `1|true|on` (hoist.py), the package compares the literal and refuses the other spellings by name — never a silent difference."""
    bad, det_env = [], {}
    env = os.environ
    for name in lever_names():
        if name in PACKAGE_ENV or name not in env:
            continue
        if name in DET_RECIPE_NAMES:                          # the caller's deterministic recipe: tolerated, reported
            det_env[name] = env[name]
            continue
        if name in res.env:
            if env[name] != res.env[name]:
                bad.append(f"{name}={env[name]!r} in the environment disagrees with mode {res.mode} ({name}={res.env[name]}, the mode table's literal spelling — the only one this package accepts)")
            continue
        bad.append(f"{name}={env[name]!r} in the environment is not part of mode {res.mode} (a lever subset is not a mode)")
    for name in upstream_switches():
        if name in env:
            bad.append(f"{name}={env[name]!r} in the environment: upstream's own switch forces the sparse atom-attention path, which the kit's levers do not serve — `--mode off` runs it as stock")
    return bad, det_env


def kit_state_findings(res: modes.Resolution) -> List[str]:
    """Late activation and lever state, from the kit module's own attributes when it is already imported."""
    out: List[str] = []
    mod = sys.modules.get(modes.KIT_HOIST_MODULE)
    if mod is not None:
        want = res.env.get(modes.KIT_SWITCH) == "1"
        got = bool(getattr(mod, "HOIST", False))
        out.append(f"late activation: {modes.KIT_HOIST_MODULE} is already imported and read {modes.KIT_SWITCH} at its import (hoist.py:18): "
                   f"the lever is {'on' if got else 'off'}, mode {res.mode} needs it {'on' if want else 'off'}; export the delta before the process imports rfd3")
    mod = sys.modules.get(modes.KIT_FZT_MODULE)
    if mod is not None:
        want = res.env.get(modes.KIT_FZT_SWITCH) == "1"
        got = bool(getattr(mod, "FZT", False))
        out.append(f"late activation: {modes.KIT_FZT_MODULE} is already imported and read {modes.KIT_FZT_SWITCH} at its import: "
                   f"the lever is {'on' if got else 'off'}, mode {res.mode} needs it {'on' if want else 'off'}; export the delta before the process imports rfd3")
    return out


def activate(mode: str, *, strict: bool = False, trigger: Optional[str] = None, dry_run: bool = False, hook: bool = True,
             allow_partial: Optional[bool] = None, argv: Optional[List[str]] = None, det: Optional[int] = None) -> dict:
    prev = _STATE["report"]
    try:
        res = modes.resolve(mode)
    except ValueError as e:
        rep = {"active": False, "mode": modes.normalize(mode), "reason": str(e), "dry_run": dry_run}
        return _finish(rep, strict, dry_run)
    if prev is not None and not dry_run:
        if prev.get("mode") == res.mode:
            return prev
        rep = {"active": False, "mode": res.mode, "reason": f"mode {prev.get('mode')} was already {'activated' if prev.get('active') else 'resolved'} in this process; a second mode is refused", "dry_run": False}
        return _finish(rep, strict, dry_run)

    rep: Dict[str, object] = {"active": False, "mode": res.mode, "variant": res.variant, "env": dict(res.env), "env_line": modes.describe_env(res.env),
                              "tier": res.tier, "kit_class": res.kit_class, "activation_point": res.activation_point, "dry_run": dry_run,
                              "trigger": trigger, "package_version": __version__, "python": sys.executable,
                              # the core's shared report keys: the levers are the delta's names; applied only once exported
                              "levers_planned": sorted(res.env), "levers_applied": [], "levers_fallback": [], "levers_unavailable": [], "partial": False,
                              "partial_reason": None, "allow_partial": False,
                              "alloc_conf": os.environ.get(ALLOC_CONF_ENV) or "unset"}   # torch's allocator configuration as launched (configs/h100.env sets expandable_segments:True — part of the documented launch line), printed on the ACTIVE line; the KERNELS line reads it back at the pass's end
    findings: List[str] = []
    try:                                                                   # the deterministic recipe's level: --det, else RFDIFFUSION3_OPT_DET (the .pth route); a stock-side switch, never a mode
        det_level = _det.level(det) if det is not None else _det.level_from_env()
    except ValueError as e:
        det_level = 0
        findings.append(str(e))
    rep["det"] = {"level": det_level}
    # the core this kit stands on, as the core pin gate states it (_autoload.core_gate, statement one of every entry: the process got here, so
    # the importable opt_core IS the one opt/pyproject.toml [tool.opt_core] pins — the same producer records the fact; it is not a second gate)
    core = core_gate()
    rep["core"] = dict(core["installed"], ok=True, pinned=core["pinned"])
    # the kit bytes
    kv = registry.check_tree()
    rep["kit"] = {k: kv[k] for k in ("ok", "kit", "checked", "missing", "stray", "version", "modified", "git_check") if k in kv}
    rep["kit_version"] = kv.get("version")
    if not kv["ok"]:
        findings.append("kit bytes: " + (kv.get("reason") or f"missing {kv['missing']} stray {kv['stray']} modified {kv['modified']}"))
    # this interpreter
    st = interpreter_state()
    rep["interpreter"] = {"label": st["label"], "detail": st["detail"], "site": st["site"], "python": st["python"], "states": st["states"]}
    pin_notes, pin_detail = pin_check("any")                               # the upstream pin: REPORTED (one PINS note line per finding), never a gate
    rep["pins"] = {"ok": not pin_notes, "notes": pin_notes, "distribution": pin_detail.get("distribution"), "stack": pin_detail.get("stack"), "pinned_stack": pin_detail.get("pinned_stack")}
    rep["foundry_version"] = (pin_detail.get("distribution") or {}).get("version") or foundry_version()
    for n in pin_notes:
        _report.emit(_report.pins_line(n))
    rep["gpu"] = gpu_info()
    if res.mode == "off":
        rep["reason"] = "mode off: stock — nothing exported; `design --mode off` runs the upstream command line in the pristine interpreter"
        rep["findings"] = findings
        rep["applied"] = "none"
        # report only: the stock routes (`design`/`warm --mode off`) strip these names from the child's environment and prove their absence
        rep["env_present"] = sorted(n for n in list(lever_names()) + list(upstream_switches()) if n in os.environ and n not in PACKAGE_ENV)
        _STATE["report"] = rep if not dry_run else _STATE["report"]
        _report.emit(_report.activation_line(rep))
        return rep
    if st["label"] != "patched":
        findings.append(f"the kit is not installed in this interpreter (rfd3 tree {st['label']}: {st['detail']}): apply the kit's own installer in the patched environment — `bash {os.path.join(registry.kit_home(), 'install.sh')}` (the add-on README.md, Install state) — never in the pristine one")
    if rep["gpu"] is None:
        findings.append("no visible GPU (nvidia-smi found none): the mode runs on CUDA only")
    rep["attn_pin"] = bool(getattr(modes.ROUTES.get(res.mode), "attn_pin", False))   # the kit routes pin the pre-flight atom-attention decision (routes.Route.attn_pin): the ACTIVE line says so
    line_tokens = list(sys.argv[1:] if argv is None else argv) if (trigger is not None or argv is not None) else []   # the design line: `design`'s composed tokens, or upstream's own command line at the autoload trigger
    env_bad, det_env = environment_findings(res)
    findings.extend(env_bad)
    rep["det_env"] = det_env                                               # the caller's deterministic-recipe names as set (reported, never refused)
    findings.extend(kit_state_findings(res))
    if line_tokens:
        refused = line_refusal(line_tokens, res.mode)                      # a line the mode cannot serve (compile_model=true, classifier-free guidance, low_memory_mode, the symmetry sampler): refused by name, nothing of the run starts
        if refused:
            findings.append(refused)
    rep["findings"] = findings
    if findings:
        rep["reason"] = findings[0] if len(findings) == 1 else f"{len(findings)} findings: " + " | ".join(findings)
        rep["applied"] = "no"
        return _finish(rep, strict, dry_run)
    if dry_run:
        rep["applied"] = "dry-run"
        _report.emit(_report.activation_line(rep))
        return rep
    os.environ.update(res.env)
    if det_level and not dry_run:
        rep["det"] = _det.apply(det_level)                                 # before upstream is imported: the variables, torch's switches, the fixed-order scatter-mean (det.py)
    _report.register_exit_tally()
    _report.before_exit(report_graph)                                      # upstream's own command line (the .pth route) has no verb after the run: the graph lever's line at exit
    _report.before_exit(report_fzt)                                        # likewise the fused transition's line (its call counts exist only after the roll-out)
    _report.before_exit(report_init_chunk)                                 # likewise the row-blocked initialiser embedding's line (exact, fast)
    _report.before_exit(report_gather)                                     # likewise the gather_attn lever's line (fast: its served counts exist only after the roll-out)
    if hook:                                                               # the KERNELS census (census.py): the route this line runs under, observed from here to the pass's end
        line_tokens = list(sys.argv[1:] if argv is None else argv)
        try:
            route = modes.route_of(res.mode, line_tokens)
        except ValueError:                                                 # reachable only without a design line (enable() in a host process whose argv is not rfd3's; a design line was gated above): the mode's own route
            route = modes.route_of(res.mode, [])
        census.observe(route.name, res.mode, line_tokens)
        _report.before_exit(census.finish_at_exit)                        # the .pth route: the KERNELS line at exit when no verb printed it (a refusal during the run already exited 5)
    rep["active"] = True
    rep["applied"] = "at-import"
    rep["levers_applied"] = sorted(res.env)
    rep["allow_partial"] = _report.allow_partial_env() if allow_partial is None else bool(allow_partial)
    _STATE["report"] = rep
    _STATE["hook"] = AppliedHook.install() if hook else None              # the APPLIED line and the exit rule at the kit module's import, on every route (warm's child: the warm verb applies the rule from the child's line)
    _report.emit(_report.activation_line(rep))
    return rep


def _finish(rep: dict, strict: bool, dry_run: bool) -> dict:
    if not dry_run:
        _STATE["report"] = rep
    _report.emit(_report.activation_line(rep))
    if strict and not rep.get("active") and rep.get("mode") != "off":
        raise ActivationError(rep.get("reason", "not active"))
    return rep


def status() -> dict:
    rep = _STATE["report"]
    if rep is None:
        return {"active": False, "reason": "enable() has not run in this process"}
    return rep


def _mark_partial(rep: dict, reason: str, levers: Optional[List[str]] = None) -> None:
    """Planned levers without evidence of application (all of them, or the named ones) move to the fallback keys; the report is partial
    with the reason (reasons accumulate when two levers fail separately)."""
    levers = list(rep["levers_applied"]) if levers is None else [n for n in levers if n in rep["levers_applied"]]
    if not levers:
        return
    rep["levers_fallback"] = list(rep.get("levers_fallback") or []) + levers
    rep["levers_unavailable"] = list(rep.get("levers_unavailable") or []) + levers
    rep["levers_applied"] = [n for n in rep["levers_applied"] if n not in levers]
    rep["partial"] = True
    rep["partial_reason"] = reason if not rep.get("partial_reason") else f"{rep['partial_reason']}; {reason}"


def report_applied(module=None) -> str:
    """The APPLIED line from the kit module's own describe() (after its import); also records it in the report. With the module
    absent — a completed run that never imported it — an active report is marked partial: the lever has no evidence of application."""
    module = module or sys.modules.get(modes.KIT_HOIST_MODULE)
    rep = _STATE["report"]
    if module is None:
        line = f"{_report.PREFIX} APPLIED none: {modes.KIT_HOIST_MODULE} is not imported"
        if rep is not None and rep.get("active") and rep.get("levers_applied"):
            rep["applied"] = "none"
            _mark_partial(rep, f"{modes.KIT_HOIST_MODULE} was never imported in this process: no evidence that {modes.KIT_SWITCH} was read (hoist.py:18)", [modes.KIT_SWITCH])
    else:
        line = _report.applied_line(module)
        _report.note_module(module)
        reopened = _report.reopen_warnings()                    # the model build is after the engine construction: the channel re-opens here
        if rep is not None:
            rep["warnings_reopened"] = reopened
            try:
                rep["applied_describe"] = module.describe()
                got = bool(getattr(module, "HOIST", False))
                agreed = got == (rep.get("env", {}).get(modes.KIT_SWITCH) == "1")
                rep["applied"] = "read-at-import" if agreed else "MISMATCH"
                if not agreed and rep.get("levers_applied"):        # the module did not read the delta: the lever is not applied
                    _mark_partial(rep, f"{modes.KIT_HOIST_MODULE} read {modes.KIT_SWITCH} {'on' if got else 'off'} at its import (hoist.py:18), mode {rep.get('mode')} exported it {'off' if got else 'on'}: the environment changed between the export and the import", [modes.KIT_SWITCH])
            except Exception as e:  # noqa: BLE001
                rep["applied_describe"] = repr(e)
    _report.emit(line)
    return line


def _planned(switch: str) -> Optional[dict]:
    """The activation report when the active mode planned `switch` AND the pass was not refused before its first roll-out, else None — the one
    gate of the four evidence reporters below (verb and exit hooks alike). A pass refused at the engine's initialize for a computation the mode
    does not drive (census.not_served: its NOT ACTIVE line is printed, nothing ran) has no lever evidence to report and no partial verdict to
    reach: the reporters print nothing."""
    rep = _STATE["report"]
    if rep is None or not rep.get("active") or switch not in (rep.get("env") or {}) or census.not_served() is not None:
        return None
    return rep


def report_graph() -> Optional[str]:
    """The graph sampler's evidence lines, once the run (or the model build) is over, when the mode planned `RFD3_CUDAGRAPH` (exact, fast):
    `APPLIED rfd3.model.cudagraph_sampler installs=<n> captures=<n> hits=<n> fallbacks=<n> declined=<n> declined_by=<options|none>` from the
    module's own process tally (patched cudagraph_sampler.GRAPH_STATS), then the lever's state in the core's LEVER grammar (graph_lever_line):
    `LEVER name=RFD3_CUDAGRAPH state=on|skipped [reason=…] impl=cuda_graphs origin=kit installs=… captures=… hits=… fallbacks=… declined=…`.
    A planned graph switch the sampler never installed — the module never imported, or no default sampler wrapped —, a capture that failed and
    was served eagerly (fallbacks > 0), or a roll-out the sampler does not drive that reached it and ran the stock loop (declined > 0:
    classifier-free guidance / the chunked pair embedder driven past the mode's refusal through upstream's API, a CPU tensor) is a PARTIAL
    activation of that lever — `state=skipped` with the reason. None when the mode did not plan the switch or nothing is active."""
    rep = _planned(modes.KIT_GRAPH_SWITCH)
    if rep is None:
        return None
    if rep.get("graph_line"):                                              # reported once per process (the verb after the run, else the exit hook)
        return rep["graph_line"]
    stats = _report.graph_stats()
    rep["graph_stats"] = stats
    if stats is None:
        line = f"{_report.PREFIX} APPLIED none: {modes.KIT_GRAPH_MODULE} is not imported"
        _mark_partial(rep, f"{modes.KIT_GRAPH_MODULE} was never imported in this process: no sampler read {modes.KIT_GRAPH_SWITCH} (inference_sampler.py:610-611)", [modes.KIT_GRAPH_SWITCH])
    else:
        _report.note_module(sys.modules.get(modes.KIT_GRAPH_MODULE), "graph")
        line = f"{_report.PREFIX} APPLIED {modes.KIT_GRAPH_MODULE} " + " ".join(f"{k}={v}" for k, v in stats.items())
        if not stats.get("installs"):
            _mark_partial(rep, f"{modes.KIT_GRAPH_MODULE} wrapped no sampler (installs=0): the graph roll-out never engaged", [modes.KIT_GRAPH_SWITCH])
        elif int(stats.get("fallbacks") or 0) > 0:      # a capture that FAILED and was served by the eager denoiser (`[RFD3_CUDAGRAPH] …; eager denoiser` in the log): never silent
            _mark_partial(rep, f"{modes.KIT_GRAPH_MODULE} fell back to the eager denoiser {stats['fallbacks']} time(s) (fallbacks>0: the `[RFD3_CUDAGRAPH]` events in the log name each)", [modes.KIT_GRAPH_SWITCH])
        elif int(stats.get("declined") or 0) > 0:       # a roll-out the graph sampler does not drive reached it (declined_by names the kind) and ran the stock loop: the lever did not serve that roll-out — never under the mode's name silently
            _mark_partial(rep, f"{modes.KIT_GRAPH_MODULE} did not drive {stats['declined']} roll-out(s) (declined_by={stats.get('declined_by')}: the stock sampler ran them eagerly)", [modes.KIT_GRAPH_SWITCH])
    rep["graph_line"] = line
    _report.emit(line)
    lever = _report.graph_lever_line(stats)
    rep["graph_lever_line"] = lever
    _report.emit(lever)
    return line


def report_fzt() -> Optional[str]:
    """The fused transition's evidence line, once the run (or the model build) is over, when the mode planned `RFD3_FZT`:
    `APPLIED rfd3.model.layers.layer_utils RFD3_FZT=True installs=1 fused_calls=<n> stock_calls=<n> ...` from the module's own process
    tally (patched layer_utils.fzt_describe()). A planned switch the module never acted on — the module never imported, or the route
    not installed (installs=0: the module read the switch off) — is a PARTIAL activation of that lever; so is a cell of the route the
    core's table does not serve on this stack (`cells_unserved`: its calls ran the stock transition), which is also named on its own
    `FALLBACK lever=RFD3_FZT reason=no-cell cells=… core=… route=stock-transition calls=<n>` line (report.fzt_fallback_line), once per
    process. None when the mode did not plan the switch or nothing is active."""
    rep = _planned(modes.KIT_FZT_SWITCH)
    if rep is None:
        return None
    if rep.get("fzt_line"):                                                # reported once per process (the verb after the run, else the exit hook)
        return rep["fzt_line"]
    stats = _report.fzt_stats()
    rep["fzt_stats"] = stats
    fallback = None
    if stats is None:
        line = f"{_report.PREFIX} APPLIED none: {modes.KIT_FZT_MODULE} is not imported"
        _mark_partial(rep, f"{modes.KIT_FZT_MODULE} was never imported in this process: nothing read {modes.KIT_FZT_SWITCH}", [modes.KIT_FZT_SWITCH])
    else:
        _report.note_module(sys.modules.get(modes.KIT_FZT_MODULE), "fzt")
        line = f"{_report.PREFIX} APPLIED {modes.KIT_FZT_MODULE} " + " ".join(f"{k}={v}" for k, v in stats.items() if not isinstance(v, dict))
        if not stats.get("installs"):
            _mark_partial(rep, f"{modes.KIT_FZT_MODULE} read {modes.KIT_FZT_SWITCH} off at its import (installs=0): Transition._forward_impl is the stock body", [modes.KIT_FZT_SWITCH])
        unserved = stats.get("cells_unserved") or {}                      # cells of the route the core's table does not serve on this stack (its cc / Triton line): a NAMED fallback to the stock transition
        if unserved:
            calls = sum(int(n) for k, n in (stats.get("cells_stock") or {}).items() if str(k).startswith("no-cell:"))
            fallback = _report.fzt_fallback_line(unserved, calls)
            _mark_partial(rep, f"opt_core serves no fused transition cell for {','.join(sorted(unserved))} on this stack ({'; '.join(sorted(set(unserved.values())))}): "
                               f"those calls ran the stock transition ({calls} call(s))", [modes.KIT_FZT_SWITCH])
    rep["fzt_line"] = line
    _report.emit(line)
    if fallback is not None:
        rep["fzt_fallback_line"] = fallback
        _report.emit(fallback)
    return line


def report_init_chunk() -> Optional[str]:
    """The row-blocked initialiser embedding's evidence line, once the run is over, when the mode planned `RFD3_INIT_CHUNK` (exact,
    fast): `APPLIED rfd3.model.layers.blocks RFD3_INIT_CHUNK=True calls=<n> chunked_calls=<n> blocks=<n> max_atoms=<n> pairs_per_block=<n>`
    from the patched module's own tally (layers/blocks.init_chunk_describe()). `chunked_calls=0` is not partial: an item of at most
    pairs_per_block // L_atom rows per block (up to 1 024 atoms) runs stock's single block, statement for statement. A planned switch the
    module never read on (the module not imported, or RFD3_INIT_CHUNK=False at its import) IS a partial activation of that lever. None
    when the mode did not plan the switch or nothing is active."""
    rep = _planned(modes.KIT_INIT_CHUNK_SWITCH)
    if rep is None:
        return None
    if rep.get("init_chunk_line"):                                         # reported once per process
        return rep["init_chunk_line"]
    stats = _report.init_chunk_stats()
    rep["init_chunk_stats"] = stats
    if stats is None:
        line = f"{_report.PREFIX} APPLIED none: {modes.KIT_INIT_CHUNK_MODULE} is not imported"
        _mark_partial(rep, f"{modes.KIT_INIT_CHUNK_MODULE} was never imported in this process: nothing read {modes.KIT_INIT_CHUNK_SWITCH}", [modes.KIT_INIT_CHUNK_SWITCH])
    else:
        _report.note_module(sys.modules.get(modes.KIT_INIT_CHUNK_MODULE), "init_chunk")
        line = f"{_report.PREFIX} APPLIED {modes.KIT_INIT_CHUNK_MODULE} " + " ".join(f"{k}={v}" for k, v in stats.items() if not isinstance(v, dict))
        if not stats.get(modes.KIT_INIT_CHUNK_SWITCH):
            _mark_partial(rep, f"{modes.KIT_INIT_CHUNK_MODULE} read {modes.KIT_INIT_CHUNK_SWITCH} off at its import: SinusoidalDistEmbed.forward runs the stock body", [modes.KIT_INIT_CHUNK_SWITCH])
    rep["init_chunk_line"] = line
    _report.emit(line)
    return line


def report_gather() -> Optional[str]:
    """The gather_attn lever's evidence line, once the run is over, when the mode planned `RFD3_GATHER_ATTN` (fast): `APPLIED
    rfdiffusion3_opt.gather RFD3_GATHER_ATTN=True installs=1 served=<n> stock=<n> stock_by=<reason:n,…|none> cells_served=<cell:n,…> idx_builds=<n>
    kernel=<version> strategy=F5.gather_attn` from the lever's process tally (gather.STATS: the kernels observer installs it at the engine's
    initialize). A planned switch the process never acted on (no engine initialized: install never ran), or a kernel that could not be installed
    (`unavailable=<reason>`), is a PARTIAL activation of that lever — named here and, for a pass that ran, refused by the KERNELS guard (atom_attn
    reads fallback:gather-…). None when the mode did not plan the switch or nothing is active."""
    rep = _planned(modes.KIT_GATHER_SWITCH)
    if rep is None:
        return None
    if rep.get("gather_line"):
        return rep["gather_line"]
    from . import gather as _g
    stats = _g.describe()
    rep["gather_stats"] = stats
    flat = {k: v for k, v in stats.items() if not isinstance(v, dict) and k not in ("active",)}
    by = ",".join(f"{k}:{v}" for k, v in sorted((stats.get("stock_by") or {}).items())) or "none"
    cells = ",".join(f"{k}:{v}" for k, v in sorted((stats.get("cells_served") or {}).items())) or "none"
    line = f"{_report.PREFIX} APPLIED rfdiffusion3_opt.gather " + " ".join(f"{k}={v}" for k, v in flat.items()) + f" stock_by={by} cells_served={cells}"
    if not stats.get(modes.KIT_GATHER_SWITCH):
        line = f"{_report.PREFIX} APPLIED none: rfdiffusion3_opt.gather.install never ran (no engine initialized in this process)"
        _mark_partial(rep, f"{modes.KIT_GATHER_SWITCH} planned, gather.install never ran (no engine initialized)", [modes.KIT_GATHER_SWITCH])
    elif stats.get("unavailable"):
        _mark_partial(rep, f"{modes.KIT_GATHER_SWITCH} planned, the gather kernel could not be installed: {stats['unavailable']}", [modes.KIT_GATHER_SWITCH])
    rep["gather_line"] = line
    _report.emit(line)
    return line


def partial_levers(rep: Optional[dict] = None) -> Tuple[List[str], Optional[str]]:
    """(levers, reason) of a partial activation report — ([], None) when the report is not partial (or there is none)."""
    rep = status() if rep is None else rep
    if not rep.get("partial"):
        return [], None
    return list(rep.get("levers_fallback") or []), rep.get("partial_reason")


class AppliedHook:
    """The kit-side import hook of an ACTIVE process (duck-typed meta-path finder, installed by `activate()`): when the kit's module
    `rfd3.model.hoist` is imported it lets the module's own body run, then reports what the module read (`report_applied`: the
    APPLIED line, the record in the activation report) and applies the exit rule there, before any roll-out — a module that read the
    lever off is a PARTIAL activation: the family line is printed and the process exits 3 (the import is refused), unless the report's
    `allow_partial` records the allowance. One-shot: it removes itself when it fires. `applied` is None until it ran; `partial` is the
    exit rule's record ({} when the run is not partial)."""

    def __init__(self):
        self.applied = None
        self.partial = None

    @classmethod
    def install(cls) -> "AppliedHook":
        for f in sys.meta_path:
            if isinstance(f, cls):
                return f
        hook = cls()
        sys.meta_path.insert(0, hook)
        return hook

    def remove(self) -> None:
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass

    def find_spec(self, fullname, path=None, target=None):
        if fullname != modes.KIT_HOIST_MODULE or self.applied is not None:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)                                # the kit module's own body first, then the report of what it read
            self._report(module)
        spec.loader.exec_module = exec_module
        return spec

    def _report(self, module) -> None:
        self.applied = True
        self.remove()
        report_applied(module)
        levers, reason = partial_levers()
        if not levers:
            self.partial = {}
            return
        allowed = bool(status().get("allow_partial"))
        self.partial = {"levers": levers, "reason": reason, "allowed": allowed}
        _report.emit(_report.partial_line(levers, reason, allowed))
        if not allowed:
            sys.exit(_report.EXIT_NOT_ACTIVE)            # before any roll-out: the import is refused, the process ends here


def applied_hook() -> Optional[AppliedHook]:
    """The hook of this process's activation (None before an activation)."""
    return _STATE["hook"]


def line_refusal(tokens: List[str], mode: str) -> Optional[str]:
    """The NOT ACTIVE reason when `mode` cannot serve the design line — upstream's compile_model=true under exact | fast (modes.COMPILE_REFUSED:
    no route) or a computation the mode's levers do not drive (modes.UNSERVED: classifier-free guidance on, low_memory_mode on, the symmetry sampler) —, else None;
    `--mode off` runs every such line as stock. Both routes gate through here: `design` on its composed line, the autoload finder on the
    process's own argv at the trigger; a spelling this reader cannot see is caught at the engine's initialize (census.unserved_by_engine)."""
    if mode == "off":
        return None
    return modes.route_refusal(tokens, mode) or modes.line_unserved(tokens, mode)


def stock_python() -> str:
    """The pristine interpreter for the stock arm: RFDIFFUSION3_STOCK_PYTHON when set, else this interpreter (which then must be pristine)."""
    return os.environ.get("RFDIFFUSION3_STOCK_PYTHON") or sys.executable


def strip_env(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """A copy of the environment without the package's switches and the kit's lever names (for the stock arm). Upstream's own switches
    (stock/PINS.json `upstream_switches_unset`: RFD3_DENSE_SDPA_ATTENTION, RFD3_LOW_MEMORY_MODE) and torch's variables are stock's own knobs and
    pass through unchanged — the stock child reports them on its STOCK line (`upstream_env=`)."""
    environ = os.environ if environ is None else environ
    drop = set(PACKAGE_ENV) | set(lever_names())
    return {k: v for k, v in environ.items() if k not in drop}
