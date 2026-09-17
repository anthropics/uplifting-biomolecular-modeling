"""``design --pack K`` — the packed line: K resident workers of the exact line on ONE GPU under uncapped CUDA MPS, started by the packing
launcher ``common/mps_packing/mps_workers.sh`` (one copy for every model directory of the release tree: fresh MPS pipe / log directories
exported before the workers initialise CUDA, the memory estimate (a NOTE, never a gate), the daemon's ``quit`` in teardown, the OOM scan of
every worker log — mps_workers.sh:7-13,21-35). K is an axis of the line (``--pack K``, K >= 1), never a mode: every worker runs the packed
resolution (modes.resolve(..., served=True): the mode's own line on the same driver; each worker's TorchScript executor
history is the stock command line's own per-process history, its first design (``startnum + W * n``) the fresh-process numerics class and
every later one the warmed class); K = 1 runs through the same launcher, so a solo pass and a packed pass differ in K alone.

The request is ``design``'s (design.prepare: upstream's hydra overrides for one target); the target runs in every worker, its designs
sliced K ways: worker W designs ``num_designs / K`` of them from ``startnum + W * num_designs / K`` (slice_cases; a count that does not
divide by K is refused by name). The outputs are upstream's own layout — every worker writes its slice at the case's prefix
(``<prefix>_<i>.pdb / .trb``, disjoint indices) — plus the run record in ``<out>``: per worker the driver's ``w<W>_timings.json`` (its evidence
record) and the launcher's ``mps_worker_<W>.log`` (mps_workers.sh:29), the launcher's own log ``mps_workers.log``, and
``opt_manifest.json``. Per worker the evidence lines are read back (design.read_evidence) and the
exit tally fed (report.add_tally_source); the per-worker cases files live in a scratch directory outside ``<out>``.

The launcher's environment (mps_workers.sh:11-29): ``K``; ``WORKER_GB`` — the per-worker device footprint the memory estimate multiplies
(cli.PACK_WORKER_GB_DEFAULT, or MODEL_OPT_PACK_WORKER_GB from configs/<card>.env; K x it + headroom above the card's total is NOTED by the
launcher and the launch proceeds, an actual OOM then surfaces through the launcher's scan, rc 6); ``LOG_DIR`` = ``<out>``; ``MPS_DIR_ROOT`` (a
fresh temporary directory: the daemon's pipe directory holds a Unix socket); on top of the line's driver environment (stack.driver_environment:
the mode's row). The worker command template is
stack.driver_command with the per-worker slots (cases file, tag) spelled ``w{W}`` and every argument shell-quoted (the launcher substitutes
``{W}`` and backgrounds each worker under ``bash -c``).

Exit rule (report.verdict, as on the plain route): refused before anything ran → 3 (``NOT ACTIVE: …``: the missing launcher or MPS control
binary, the mode facts; a request that does not slice; the launcher's own refusal line, mps_workers.sh:20,24 — its rc 4 / 5 named on
the line); the launcher's OOM verdict (:35, rc 6) or a worker that exited non-zero → 1 (failed); a worker with a lever on the stock path → 3
``NOT ACTIVE: partial activation``, the outputs kept; designs short of the request → 1.
"""
from __future__ import annotations

import json
import os
import sys
import re
import shlex
import tempfile
from typing import Dict, List, Optional, Tuple

from . import design, manifest as _manifest, outputs as _outputs, report as _report, stack
from .modes import numerics, resolve
from .registry import LEVERS
from .report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE   # noqa: F401 — the codes' one home is report.py

WORKER_SLOT = "w{W}"                                               # the launcher's placeholder in the three per-worker slots (mps_workers.sh:4,28)
WORKER_LOG = "mps_worker_{W}.log"                                  # the launcher's per-worker log name under LOG_DIR (mps_workers.sh:29)
LAUNCHER_LOG = "mps_workers.log"                                   # the launcher's own lines ([mps_workers] ...), captured by the package
LAUNCHER_LINE_RE = re.compile(r"^\[mps_workers\] ")
LAUNCHER_RC: Dict[int, str] = {                                    # the launcher's own exit codes (mps_workers.sh:20,24,35; a worker's rc otherwise, :32 — the memory estimate :19 is a NOTE, no code)
    4: "nvidia-cuda-mps-control not on PATH (mps_workers.sh:20); nothing ran",
    5: "the MPS control daemon failed to start (mps_workers.sh:24); nothing ran",
    6: "out-of-memory text in a worker log: the launcher's OOM scan (mps_workers.sh:35) — the outputs of this launch are not exact-class",
}
LAUNCHER_REFUSALS: Tuple[Tuple[int, str], ...] = (                # the launcher's refusal lines before any worker starts (mps_workers.sh:20,24) -> its code
    (4, "[mps_workers] nvidia-cuda-mps-control not found"), (5, "[mps_workers] MPS control daemon failed to start"))
LAUNCHER_NOTE_PREFIX = "[mps_workers] NOTE: "                       # the launcher's memory-estimate line (mps_workers.sh:19): K x WORKER_GB + HEADROOM_GB > the card — noted, the launch proceeds
GM1_NOTE = "requested worker footprint exceeds the card by estimate; proceeding — may OOM"   # the package's words for it (report.note_line), the launcher's arithmetic appended
LAUNCHER_OOM_LINE = "[mps_workers] WARNING: OOM text found in a worker log"   # the OOM scan's line (mps_workers.sh:35, rc 6)
WORKER_RC_RE = re.compile(r"^\[mps_workers\] worker (?P<w>\d+) rc=(?P<rc>\d+)\s*$")   # one per worker that ran (mps_workers.sh:32)


class ServeError(ValueError):
    """A refused request, with the named fact."""


def slice_cases(cases: List[dict], k: int) -> List[List[dict]]:
    """The per-worker cases lists: worker W runs every case with num_designs / K designs from startnum + W * (num_designs / K)
    Refuses a case whose num_designs does not divide by K."""
    if k < 1:
        raise ServeError(f"--pack {k}: K must be >= 1")
    out: List[List[dict]] = [[] for _ in range(k)]
    for c in cases:
        n, base = int(c["num_designs"]), int(c.get("startnum", 0))
        if n % k:
            raise ServeError(f"case {c['name']}: num_designs {n} does not slice into K={k} workers (the packing rule gives every worker the same "
                             f"designs-per-worker count); make inference.num_designs a multiple of {k}")
        per = n // k
        for w in range(k):
            d = dict(c)
            d["num_designs"], d["startnum"] = per, base + w * per
            out[w].append(d)
    return out


def worker_template(res, out_dir: str, work_dir: str, rfd: str, weights: str, python: Optional[str] = None) -> Tuple[str, List[str]]:
    """The launcher's one worker command template (a shell line with `w{W}` in the cases file and the tag) and the argv it was rendered from
    (stack.driver_command on the packed resolution): every worker writes at the cases' own prefixes under `out_dir`, its run-record files
    tagged `w<W>`; its cases file is `<work_dir>/w<W>/cases.json`."""
    cases = os.path.join(work_dir, WORKER_SLOT, "cases.json")
    argv = stack.driver_command(res, cases, out_dir, WORKER_SLOT, rfd, weights, python)
    return " ".join(shlex.quote(a) for a in argv), argv


def launcher_command(k: int, template: str) -> List[str]:
    return ["bash", stack.pack_launcher(), template]


def launcher_environment(base_env: Dict[str, str], k: int, worker_gb: float, out_dir: str) -> Dict[str, str]:
    """The launcher process's environment: the packed line's driver environment + the launcher's own inputs (mps_workers.sh:15,22,29)."""
    env = dict(base_env)
    # MPS_DIR_ROOT outside the output directory: the launcher creates the control daemon's pipe directory under it (mps_workers.sh:22-23, a Unix socket
    # `control_privileged` that outlives the daemon) — the output directory holds regular files only
    env.update({"K": str(int(k)), "WORKER_GB": f"{float(worker_gb):g}", "LOG_DIR": out_dir, "MPS_DIR_ROOT": tempfile.mkdtemp(prefix="rfdiffusion1_mps_")})
    return env


def launcher_lines(log_path: str) -> List[str]:
    try:
        return [ln.rstrip("\n") for ln in open(log_path, encoding="utf-8", errors="replace") if LAUNCHER_LINE_RE.match(ln)][:40]
    except OSError as e:
        return [f"<launcher log unreadable: {e!r}>"]


def run(mode: Optional[str], overrides=None, *, k: int = 1, worker_gb: float = 10.0, tag: str = "run", timeout: Optional[float] = None,
        python: Optional[str] = None, dry_run: bool = False, det_flag: bool = False) -> Tuple[int, dict]:
    """The packed pass over the command line's one target (design.prepare). Returns (exit code, the manifest dict). Refusals happen before
    anything runs (exit 3 with the named fact; exit 2 for a command line that names no target)."""
    try:
        cases, out_dir = design.prepare(overrides)
        slices = slice_cases(cases, k)
    except design.UsageError as e:
        sys.stderr.write(f"rfdiffusion1-opt design --pack: {e}\n")
        return design.EXIT_USAGE, {"status": "usage", "reason": str(e)}
    except (design.DesignError, ServeError) as e:
        _report.emit(_report.not_active_line(str(e)))
        return EXIT_NOT_ACTIVE, {"status": "refused", "reason": str(e)}
    m, refused, overrides = design.request(mode, overrides)                                                # what the kit line cannot serve: refused by name, exit 3, nothing runs (`--mode off` runs it, unpacked)
    if refused:
        return refused
    rep = stack.activate(mode, overrides, dry_run=dry_run, served=True, det=det_flag)
    if rep.get("reason") or rep.get("would_refuse") or not (rep.get("active") or dry_run):
        return EXIT_NOT_ACTIVE, {"status": "refused", "reason": rep.get("reason") or "; ".join(rep.get("would_refuse") or []), "activation": rep}
    res = resolve(rep["mode"], overrides if overrides is not None else rep.get("overrides"), served=True, det=det_flag)
    rfd, weights = rep.get("rfd_root"), rep.get("weights")
    wgb = float(worker_gb)
    work = design.scratch_dir(tag)
    template, argv = worker_template(res, out_dir, work, rfd, weights, python)
    n_designs = sum(c["num_designs"] for c in cases)
    man = _manifest.start(rep, res, cases, out_dir, tag, det=det_flag)
    man["serve"] = {"k": int(k), "worker_gb": wgb, "launcher": stack.pack_launcher(),
                    "launcher_cmd": launcher_command(k, template), "template": template, "template_argv": argv, "worker_slot": WORKER_SLOT,
                    "worker_log": WORKER_LOG, "slices": [[{"name": c["name"], "startnum": c["startnum"], "num_designs": c["num_designs"]} for c in s] for s in slices],
                    "driver_pythonpath": None}
    if dry_run:
        man["status"] = "dry-run"
        return EXIT_OK, man
    os.makedirs(out_dir, exist_ok=True)
    for w, s in enumerate(slices):
        design.write_cases(s, os.path.join(work, WORKER_SLOT.format(W=w)), out_dir)                                # <work>/w<W>/cases.json
    env, dropped = stack.driver_environment(res, rep["gpu"])
    man["env_dropped"] = dropped
    man["serve"]["driver_pythonpath"] = env.get("PYTHONPATH")
    lenv = launcher_environment(env, k, wgb, out_dir)
    man["serve"]["launcher_env"] = {n: lenv[n] for n in ("K", "WORKER_GB", "HEADROOM_GB", "LOG_DIR", "MPS_DIR_ROOT") if n in lenv}
    _report.register_exit_tally()                                                                          # as design.run: registered once a pass runs
    _report.emit(_report.run_line(res.mode, len(cases), n_designs, out_dir, f"served K={k}"))
    log_path = os.path.join(out_dir, LAUNCHER_LOG)
    rc, wall, t0 = design._run(launcher_command(k, template), lenv, os.getcwd(), log_path, timeout)   # relative typed paths resolve where upstream would resolve them: the working directory
    man["serve"].update(rc=rc, wall_s=round(wall, 2), t0=t0, log=log_path, lines=launcher_lines(log_path), rc_fact=LAUNCHER_RC.get(rc))
    man["serve"]["notes"] = launcher_notes(log_path)                                                     # the memory estimate when it exceeded the card: named, printed, never a gate
    for ln in man["serve"]["notes"]:
        _report.emit(_report.note_line(f"{GM1_NOTE} (launcher: {ln[len(LAUNCHER_NOTE_PREFIX):].split(' — ')[0]})"))
    man["driver_passes"] = []
    names = [c["name"] for c in cases]
    refusal = launcher_refusal(log_path)
    man["serve"]["refusal_line"] = refusal[1] if refusal else None
    if refusal is not None:                                                                               # the launcher refused before any worker ran: its own line says so
        man["status"], man["reason"] = "refused", LAUNCHER_RC[refusal[0]] + (f" (launcher rc={rc})" if rc != refusal[0] else "")
        man["outputs"] = {"n_pdb": 0, "expected": n_designs, "per_worker": {}}
        _report.emit(_report.not_active_line(man["reason"]))
        _manifest.finish(man, out_dir)
        return EXIT_NOT_ACTIVE, man
    failed = None                                                                                         # the facts the verdict weighs (report.verdict)
    partial, partial_reasons = [], []
    if launcher_oom(log_path) or rc == 6:                                                                 # the launcher's own verdict: above every worker's (mps_workers.sh:35)
        failed = LAUNCHER_RC[6]
    for w in range(k):
        wtag = WORKER_SLOT.format(W=w)
        wlog = os.path.join(out_dir, WORKER_LOG.format(W=w))
        timings = os.path.join(out_dir, f"{wtag}_timings.json")
        ev = design.read_evidence(wlog, list(res.levers))
        io = design.io_counters(wlog)
        n_run = design.designs_computed(timings)
        if not design.all_skipped(timings):                                                               # a worker whose every requested design already existed (upstream's inference.cautious rule) made no forward call: no per-call counters to hold the levers to
            ev["forbidden"] += design.io_defects(io, [l for l in res.levers if l in ev["applied"]])        # lever IO1's proof per worker
            ev["forbidden"] += design.counter_defects(timings, [l for l in res.levers if l in ev["applied"]])   # the kits' final counters per worker (design.COUNTER_RULES)
        observed = _report.numerics_from_timings(timings)                              # the worker's torch numerics read-back against the line's declaration (design.numerics_defects): exact never runs TF32
        num = numerics(res.mode, observed, levers=res.levers) if "error" not in observed else None
        if os.path.isfile(timings):
            ev["forbidden"] += design.numerics_defects(num)
        _report.emit(_report.evidence_line(ev["applied"], ev["missing"], ev["forbidden"], ev.get("log_error")))
        _report.emit(_report.numerics_line(wtag, observed, num))                        # the worker's NUMERICS line on this process's output (numerics_source=torch when the record is readable)
        if "IO1" in res.levers:
            _report.emit(_report.io_line(wtag, io))
        stack.record_pass(ev)
        if ev["applied"]:
            stack.confirm_active(f"{wtag}: applied-lines {','.join(ev['applied'])}")     # the ACTIVE documented line follows the first worker's application evidence
        _report.add_tally_source(wtag, timings, wlog)
        wrc = worker_rc(log_path, w)
        man["driver_passes"].append({"designs_computed": ({"computed": n_run[0], "skipped_existing": n_run[1]} if n_run is not None else None), "tag": wtag, "io": io, "cases": names, "rc": wrc, "cmd": template.replace("{W}", str(w)), "log": wlog,
                                     "timings_json": timings if os.path.isfile(timings) else None, "evidence": ev,
                                     "numerics": num if num is not None else {"source": "unreadable", "error": observed.get("error")},
                                     "outputs": _outputs.list_outputs(slices[w], out_dir)})
        if wrc != 0 and failed is None:
            failed = f"worker {wtag}: rc={wrc}" if wrc is not None else f"worker {wtag}: no `[mps_workers] worker {w} rc=` line (the worker never reported; launcher rc={rc})"
        if ev["missing"] or ev["forbidden"]:                                                              # a lever on the stock path in a worker (every worker's list kept)
            partial += [l for l in ev["missing"] if l not in partial]
            partial_reasons.append(f"worker {wtag}: levers without their evidence line {ev['missing'] or 'none'}; forbidden lines {len(ev['forbidden'])}")
    if rc != 0 and failed is None:                                                                        # the launcher's rc with no line of its own and every worker rc 0
        failed = f"launcher rc={rc}"
    n_pdb = sum((p["outputs"] or {}).get("n_pdb", 0) for p in man["driver_passes"])
    man["outputs"] = {"n_pdb": n_pdb, "expected": n_designs, "per_worker": {p["tag"]: p["outputs"] for p in man["driver_passes"]}}
    v = _report.verdict(failed=failed, partial=partial, partial_reason="; ".join(partial_reasons) or None, n_pdb=n_pdb, expected=n_designs)
    man.update(status=v["status"], partial=v["partial"], partial_reason=v["partial_reason"], incomplete=v["incomplete"], exit_code=v["exit_code"])
    if v["reason"]:
        man["reason"] = v["reason"]
    if v["incomplete"]:
        man["outputs_note"] = f"expected {n_designs} designs over {k} workers, found {n_pdb}"
    _report.emit_verdict(v)
    _manifest.finish(man, out_dir)
    return v["exit_code"], man


def launcher_notes(launcher_log: str) -> List[str]:
    """The launcher's NOTE lines (LAUNCHER_NOTE_PREFIX: the memory estimate exceeded the card — mps_workers.sh:19 prints it and proceeds)."""
    return [ln for ln in _launcher_lines(launcher_log) if ln.startswith(LAUNCHER_NOTE_PREFIX)]


def launcher_refusal(launcher_log: str) -> Optional[Tuple[int, str]]:
    """The launcher's own refusal before any worker started — (its code, the line) from LAUNCHER_REFUSALS (mps_workers.sh:20,24) when
    such a line was printed and no `worker <W> rc=` line follows; None otherwise (a worker's rc returned through mps_workers.sh:32 is the
    worker's own code; the memory-estimate line is a NOTE, launcher_notes, never a refusal)."""
    lines = _launcher_lines(launcher_log)
    if any(WORKER_RC_RE.match(ln) for ln in lines):
        return None
    for code, head in LAUNCHER_REFUSALS:
        for ln in lines:
            if ln.startswith(head):
                return code, ln
    return None


def launcher_oom(launcher_log: str) -> bool:
    """True when the launcher's OOM scan printed its line (mps_workers.sh:35): the launch's outputs are not exact-class whatever the workers said."""
    return any(ln.startswith(LAUNCHER_OOM_LINE) for ln in _launcher_lines(launcher_log))


def _launcher_lines(launcher_log: str) -> List[str]:
    try:
        return [ln.strip() for ln in open(launcher_log, encoding="utf-8", errors="replace")]
    except OSError:
        return []


def worker_rc(launcher_log: str, w: int) -> Optional[int]:
    """Worker W's exit code from the launcher's own line `[mps_workers] worker <W> rc=<rc>` (mps_workers.sh:32); None when it never printed."""
    rx = re.compile(rf"^\[mps_workers\] worker {w} rc=(\d+)\s*$")
    try:
        for ln in open(launcher_log, encoding="utf-8", errors="replace"):
            m = rx.match(ln.strip())
            if m:
                return int(m.group(1))
    except Exception:  # noqa: BLE001
        pass
    return None

