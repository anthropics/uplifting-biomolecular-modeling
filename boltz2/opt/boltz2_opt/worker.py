"""``pred --mode exact|fast``: the kits' persistent worker on the caller's YAMLs — the kit route.

One process per call: the mode's kit files are staged (stack.stage), the worker variant is derived by the sampler add-on's own script, the
batch json is written from the YAMLs (one record per YAML; the record id is the YAML stem, which is the name stock gives the output
directory, bz_worker_lev.py:281-282), and the kit worker runs under the mode's activation row with the kits' probe arm's arguments
(modes.WORKER_ARGS). Outputs land where the worker puts them — ``<out_dir>/by_seed/<name>/s<seed>/`` with stock's file set
(``<name>_model_0.cif``, ``pae_<name>_model_0.npz``, ``plddt_<name>_model_0.npz``, ``confidence_<name>_model_0.json``) — plus the
worker's transcript ``<tag>_worker.log``; the launch's kit directory ``<out_dir>/_kit/`` holds the staged worker and the batch json (the
worker's own records — its log json, the KERNELS record, the affinity leg's hand-over — are written there and removed once folded into the
lines, remove_launch_records). The prediction settings are the call's — `boltz predict`'s own options (recycling_steps, sampling_steps,
diffusion_samples, max_parallel_samples, step_scale, write_full_pae, write_full_pde, output_format), upstream's defaults for the ones not given
(settings.worker_settings; the batch json), kernels per the mode row; ``pred --mode off`` with the same options is the stock arm of the same shape.

The worker's own log is the evidence that the row was in force (stack.evidence): missing evidence, a kit "not applied" line or a worker
exit code other than 0 is a failure of the run, never a silent stock result. Every expected (input, seed) unit is accounted (skipped.units):
ok, or failed with a named reason — an input the stock parser skipped is named by the worker's ``[boltz2-opt] SKIPPED item=<name>
reason=…`` line (relayed here), an (input, seed) unit boltz's own predict_step skipped on CUDA out of memory by the worker's ``failed``
entry (``FAILED item=<name> seed=<s> reason=upstream skipped batch: CUDA out of memory (…)``; boltz's ``| WARNING: ran out of memory,
skipping batch`` line relayed) — recorded under ``outputs.failed_units`` and counted failed; the exit is then 1 (outputs short). An input
declaring upstream's affinity property gets upstream's affinity leg after its structure pass (the worker's run_affinity_leg → affinity_leg.py,
stock's second model verbatim in a clean interpreter); its unit is complete only with ``affinity_<name>.json`` (expected_files).
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import sys
from typing import List, Optional

from . import __version__, manifest as mf, modes, report as rep, rowpair_msa, settings as _settings, skipped, stack


from .worker_launch import KERNELS_CENSUS  # noqa: E402  — the census record's name beside the worker log (in the launch's kit directory)


AFFINITY_JSON = "affinity_{name}.json"                              # upstream's BoltzAffinityWriter output, beside the structure files of an input declaring the property


def affinity_inputs(items: List[dict]) -> List[str]:
    """Names of the inputs whose YAML declares upstream's affinity property (``properties: - affinity: {binder: <chain>}``; boltz schema.py
    reads the property key case-insensitively, so does this). Upstream runs a second model pass for those after the structure pass and the kit
    worker runs that leg verbatim (bz_worker_lev*.py run_affinity_leg → boltz2_opt.affinity_leg), so such an input's unit is complete only
    with its affinity_<name>.json (expected_files)."""
    import yaml
    names = []
    for it in items:
        try:
            doc = yaml.safe_load(open(it["yaml"])) or {}
        except (OSError, yaml.YAMLError):
            continue                                                # a YAML upstream cannot read is upstream's to name (process_inputs: Failed to process … Skipping)
        props = doc.get("properties") if isinstance(doc, dict) else None
        if isinstance(props, list) and any(isinstance(p, dict) and p and next(iter(p)).lower() == "affinity" for p in props):
            names.append(it["name"])
    return names


STRUCTURE_SUFFIX = {"mmcif": "cif", "pdb": "pdb"}                   # upstream's BoltzWriter: --output_format mmcif writes <name>_model_<k>.cif, pdb writes .pdb


def expected_files(name: str, affinity: bool, output_format: str = "mmcif") -> List[str]:
    """The files whose presence makes an (input, seed) unit complete: the first structure ``<name>_model_0.<cif|pdb>`` (by --output_format),
    and ``affinity_<name>.json`` for an input declaring the affinity property."""
    return [f"{name}_model_0.{STRUCTURE_SUFFIX[output_format]}"] + ([AFFINITY_JSON.format(name=name)] if affinity else [])


INPUT_SUFFIXES = (".fa", ".fas", ".fasta", ".yml", ".yaml")            # boltz predict's own input forms (boltz/main.py check_inputs): these files, or a directory of them


def expand_inputs(paths: List[str]) -> List[str]:
    """The input files `boltz predict` would take for these paths, in order: a file is itself; a directory is its entries (sorted), where —
    exactly as boltz's check_inputs — a sub-directory or a file of another type is an error by name. A missing path is FileNotFoundError."""
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for e in sorted(os.listdir(p)):
                q = os.path.join(p, e)
                if os.path.isdir(q):
                    raise ValueError(f"Found directory {q} instead of .fasta or .yaml (boltz predict refuses a nested directory)")
                if os.path.splitext(e)[1].lower() not in INPUT_SUFFIXES:
                    raise ValueError(f"Unable to parse filetype {os.path.splitext(e)[1]} ({q}), please provide a .fasta or .yaml file (boltz predict's input types)")
                files.append(q)
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def items_from_yamls(paths: List[str]) -> List[dict]:
    """[{name, uid, yaml}] from boltz predict's input forms (expand_inputs: YAML / FASTA files, directories of them); the record id is the file
    stem, as in stock, and must be unique across the pass. (`yaml` = the input file's path, whichever of stock's types it is.)"""
    items, seen = [], set()
    for y in expand_inputs(paths):
        name = os.path.splitext(os.path.basename(y))[0]
        if name in seen:
            raise ValueError(f"two inputs share the record name {name!r} (the file stem is the record id)")
        seen.add(name); items.append({"name": name, "uid": name, "yaml": os.path.abspath(y)})
    return items


def run(mode: str, yamls: List[str], out_dir: str, seeds: List[int], *, tag: Optional[str] = None,
        extra_record: Optional[dict] = None, allow_partial: bool = False, n_gpu=1, settings: Optional[dict] = None) -> int:
    """The whole route; returns the process exit code (report.EXIT_*). Exit rule: any evidence problem → EXIT_NOT_ACTIVE; a kit fallback
    of a row lever (stack.partial_activation) → the line ``NOT ACTIVE: partial activation — <detail>; exit 3 (--allow-partial records and
    proceeds)`` and EXIT_NOT_ACTIVE, unless ``allow_partial`` accepts it (named on the line ``PARTIAL allowed: …``; report.allow_partial /
    levers_fallback / partial in the run record) and the exit is the run's own; a documented gate is recorded and does not change
    the exit; outputs short of the expected set → ``incomplete``, EXIT_FAILED (a worker that exits non-zero is EXIT_FAILED, every reason named).
    ``settings``: the caller's `boltz predict` options ({knob: value}; _settings.KNOBS — upstream's names, upstream's defaults for the ones not
    given): the worker's predict_args / step_scale / write_full_pae / write_full_pde / output_format for every input of the pass; a bad value or a
    knob the worker line cannot serve (--num_workers, --no_kernels: _settings.WORKER_REFUSED) is a usage error naming it, nothing launched."""
    row = modes.resolve(mode)
    if row["route"] != modes.PINNED_ROUTE:
        rep.say(rep.not_active_line(f"mode {row['mode']} has no worker route")); return rep.EXIT_USAGE
    rep.arm_tally(row["mode"], row["route"])
    plan = stack.gate(row["mode"], need_gpu=True, n_gpu=n_gpu)
    row = modes.resolve(row["mode"])                                   # the row on THIS card (stack.gate set it: modes.CARD_DROPS levers leave the row by name)
    P = int(plan["n_gpu"] or 1)
    rep.arm_tally(row["mode"], row["route"], n_gpu=P)
    if plan["reasons"]:
        rep.say(rep.not_active_line("; ".join(plan["reasons"]))); rep.set_rc(rep.EXIT_NOT_ACTIVE)
        mf.record(mf.build("pred", row["mode"], report={"active": False, "reason": plan["reasons"]}, rc=rep.EXIT_NOT_ACTIVE, extra=extra_record))
        return rep.EXIT_NOT_ACTIVE
    try:
        items = items_from_yamls(yamls)
    except (FileNotFoundError, ValueError) as e:
        rep.say(rep.not_active_line(f"inputs: {e}")); rep.set_rc(rep.EXIT_USAGE); return rep.EXIT_USAGE
    aff = set(affinity_inputs(items))                                # units of these inputs are complete only with upstream's affinity_<name>.json (the worker runs that leg)
    try:
        eff = _settings.worker_settings(mode, settings)                  # `boltz predict`'s own options over upstream's defaults; a knob the worker line cannot serve is named
    except ValueError as e:
        rep.say(rep.not_active_line(f"settings: {e}")); rep.set_rc(rep.EXIT_USAGE); return rep.EXIT_USAGE
    modes.set_run_drops(eff["run_off"]); modes.set_n_gpu(P); row = modes.resolve(mode)       # levers this run takes off by name for an option they cannot carry (modes.RUN_DROPS; {} = the row as tabled):
                                                                      # stated once, before staging — every resolve() of this process (stage, child_env, evidence, the lines) agrees
    out_dir = os.path.abspath(out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = tag or "pred"
    workdir = os.path.join(out_dir, "_kit")
    try:
        staged = stack.stage(row["mode"], workdir)
    except RuntimeError as e:
        rep.say(rep.not_active_line(str(e))); rep.set_rc(rep.EXIT_NOT_ACTIVE); return rep.EXIT_NOT_ACTIVE
    batch = stack.write_batch(workdir, tag, items, seeds, out_dir, settings=eff)
    env = stack.child_env(row["mode"], n_gpu=P)
    report = {"active": True, "mode": row["mode"], "route": row["route"], "tier": row["tier"], "gpu": (plan["gpu"] or {}).get("name"), "n_gpu": P, "settings": dict(eff),
              "levers_applied": list(row["levers"]) + (["rowpair_tp"] if P > 1 else []), "levers_fallback": [], "partial": False, "env": dict(row["env"], **stack.tp_exports(env, P)), "card_off": dict(row.get("card_off") or {}), "off": dict(row.get("off") or {}), "run_off": dict(row.get("run_off") or {}), "tp_off": dict(row.get("tp_off") or {}),
              "worker": stack.worker_name(row["mode"]), "kernels": row["kernels"], "boltz_version": (plan["pins"] or {}).get("version"), "package_version": __version__}
    rep.say(rep.active_line(report))
    log_path = os.path.join(out_dir, f"{tag}_worker.log")
    import time as _time
    t_launch = _time.time()
    rc = stack.run_worker(row["mode"], workdir, batch, log_path, env=env, n_gpu=P, settings_word=_settings.settings_word(settings), num_workers=eff["num_workers"])
    try:                                                              # fold the launch's records into the lines and the run record; they leave with the verb (finally) — a run that dies before this point keeps them
        wlog = stack.read_worker_log(stack.worker_log_path(workdir, tag))
        text = open(log_path, errors="replace").read() if os.path.isfile(log_path) else ""
        ev, problems = stack.evidence(row["mode"], wlog, text, n_gpu=P, rank_worker_logs=stack.rank_logs(batch, P) if P > 1 else None, sampling_steps=eff["sampling_steps"])
        rank_logs = glob.glob(os.path.join(workdir, "ranks", "rank*.log")) if P > 1 else []
        for line in relay_lines(text, rank_logs):                     # the worker's reader-facing lines (KERNELS / KERNELS-REFUSED / PHASE / PEAK: RELAY_LINES), verbatim on this process's stdout — the stock route's reach it directly
            rep.say(line)
        rep.say(peak_summary(text, rank_logs, P))
        census_path = os.path.join(workdir, KERNELS_CENSUS.format(tag=tag))
        from . import kernels as _kernels
        census = _kernels.read_record(census_path, since=t_launch)   # this pass's record (one written before the launch is an earlier pass's: ignored, never deleted)
        fallbacks, gates = stack.partial_activation(row["mode"], ev) if wlog is not None else ([], {})
        report.update({"levers_fallback": fallbacks, "partial": bool(fallbacks), "gates": gates, "allow_partial": bool(allow_partial)})
        def missing(name, s):                                            # the first expected file the unit lacks, else None (skipped.units)
            have = set(stack.output_files(out_dir, name, s))
            return next((f for f in expected_files(name, name in aff, eff["output_format"]) if f not in have), None)
        acct = skipped.units(items, seeds, skipped.skipped_from_log(wlog), missing, failed=skipped.failed_from_log(wlog))
        n_ok, n_fail = acct["ok"], acct["failed"]                   # every expected (input, seed) unit: ok, or failed with a named reason (the worker's SKIPPED reason for an input the stock parser skipped, its `failed` reason for a batch boltz skipped on CUDA out of memory)
        for u in acct["failed_units"]:
            rep.say(skipped.failed_line(u["name"], u["seed"], u["reason"]))
        rep.count(ok=n_ok, failed=n_fail)
        rep.tally_headroom(ev.get("headroom_gated") or 0)               # predictions the sampler levers' memory-headroom gate sampled on the stock path (EXIT levers_headroom_gated=)
        report["outputs"] = {k: acct[k] for k in ("expected", "ok", "failed", "status", "failed_units", "skipped")}
        parse_failed_all = bool(acct.get("all_skipped")) and bool(items)      # every input failed to parse in the worker (the stock parser skipped each one, named on its SKIPPED /
        if parse_failed_all:                                            # FAILED lines): the parser's error is THE reason of this run — the levers' gates ('… served no call over N item(s)')
            problems[:] = [parse_failure_reason(items, acct)]            # do not speak for a pass in which nothing was parsed; exit 1 below, as the stock route exits for them
        if rc != 0:
            problems.append(f"worker exit code {rc} (log: {log_path})")
        if rc == rep.EXIT_KERNELS:                                     # the REQUIRE guard refused in the worker: its KERNELS-REFUSED line (echoed above) names accelerator and route — exit 5, not the generic worker failure
            problems[:] = [p_ for p_ in problems if not p_.startswith("worker exit code")] + [f"KERNELS REQUIRE refused in the worker process (exit {rep.EXIT_KERNELS})"]
        exit_code = rep.run_exit(report, ev, problems, fallbacks, rc, n_fail, allow_partial)
        if parse_failed_all and exit_code == rep.EXIT_NOT_ACTIVE:        # nothing parsed: every unit FAILED by name -> exit 1 (report.EXIT_FAILED), the kit's rule for inputs the stock
            exit_code = rep.EXIT_FAILED; rep.set_rc(exit_code)          # parser skipped on both routes (cli.cmd_pred_off exits 1 for the same inputs) — not the levers' 3
        k_rc = rep.kernels_exit(row["mode"], rc, [census or ev.get("kernels")] if (census or ev.get("kernels")) else [], where=f"{KERNELS_CENSUS.format(tag=tag)} / the KERNELS line in {os.path.basename(log_path)}",
                                all_skipped=acct["all_skipped"])   # every requested input SKIPPED by the stock parser: the census's NO-STEP is 'nothing ran', the SKIPPED / FAILED lines and exit 1 speak
        if k_rc == rep.EXIT_KERNELS or (k_rc == rep.EXIT_NOT_ACTIVE and exit_code == rep.EXIT_OK):   # the census's rules over the run's own exit: a refusal is 5 whatever else held; a missing record turns a clean pass into 3
            exit_code = k_rc; report["active"] = False; report["reason"] = "; ".join([x for x in [report.get("reason"), f"KERNELS exit {k_rc}"] if x]); rep.set_rc(exit_code)
        mf.record(mf.build("pred", row["mode"], report=report, evidence=ev, inputs=mf.input_records(items), outputs=mf.output_records(out_dir, items, seeds),
                                   settings=_settings.manifest_settings(row["mode"], eff, seeds, P), staged=staged,
                                   command=stack.worker_command(batch, row["mode"], P, _settings.settings_word(settings), eff["num_workers"]), rc=rc, kernels_census=census,
                                   extra=dict(extra_record or {}, worker_log=stack.worker_log_path(workdir, tag), worker_stdout=log_path, kernels_census_path=census_path)))
        return exit_code
    finally:
        remove_launch_records(workdir, tag, P)




def parse_failure_reason(items: list, acct: dict) -> str:
    """The run's reason when EVERY input failed to parse in the worker (skipped.units ``all_skipped``): ``inputs: N/N failed to parse in the
    worker — <the first unit's reason: boltz's own error, e.g. stock parser skipped the input (FileNotFoundError: … 'x/msa/a.csv')> (first
    failure: item <name>); nothing was predicted``. One SKIPPED line per input and one FAILED line per (input, seed) unit above name each."""
    units = acct.get("failed_units") or []
    first = units[0] if units else {"name": (items[0]["name"] if items else "?"), "reason": skipped.REASON}
    n = len(items)
    return f"inputs: {n}/{n} failed to parse in the worker — {first['reason']} (first failure: item {first['name']}); nothing was predicted"


def launch_records(kit_dir: str, tag: str, n_gpu: int = 1) -> List[str]:
    """The worker→caller records of one launch under its kit directory (rank 0's and every rank r > 0's, stack.rank_dir): the worker's log json
    (stack.worker_log_path), the KERNELS record (worker_launch.KERNELS_CENSUS) and the affinity leg's hand-over directory (bz_worker_lev.py
    run_affinity_leg: consumed by the clean interpreter the same worker starts, nothing later reads it)."""
    dirs = [kit_dir] + [stack.rank_dir(kit_dir, r) for r in range(1, int(n_gpu or 1))]
    return [p for d in dirs for p in (stack.worker_log_path(d, tag), os.path.join(d, KERNELS_CENSUS.format(tag=tag)), os.path.join(d, "_affinity_leg"))]


def remove_launch_records(kit_dir: str, tag: str, n_gpu: int = 1) -> List[str]:
    """Remove the launch's records (launch_records) once the caller folded them into its lines and run record; returns the paths removed.
    What stays under the kit directory: the staged worker, the batch json(s) and the ranks' own logs of an n_gpu run."""
    gone = []
    for p in launch_records(kit_dir, tag, n_gpu):
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True); gone.append(p)
        elif os.path.isfile(p):
            try:
                os.remove(p); gone.append(p)
            except OSError:
                pass
    return gone


RELAY_LINES = (                                                     # the model process's lines a caller's transcript must carry on every route (a caller reads one stdout): relayed verbatim by pred (and warm, which is pred)
    skipped.LINE_RE,                                                #   an input the stock parser skipped, named once by the worker (skipped.line)
    skipped.STOCK_OOM_RE,                                           #   boltz's own words when its predict_step skips a batch on CUDA out of memory (the unit's FAILED line names it: skipped.oom_reason)
    r"\[boltz2-opt [^\]]*\] KERNELS route=.*",                       #   the KERNELS census pass line (kernels.LINE_MARK; one per model process, rank=<r> at n_gpu > 1)
    r"\[boltz2-opt [^\]]*\] KERNELS-REFUSED route=.*",               #   the REQUIRE guard's refusal (kernels.REFUSED_MARK)
    r"PHASE item=.*",                                               #   the per-item timing line (phase.py; rank 0's only at n_gpu > 1)
    r"\[boltz2-opt\] PEAK item=.*", r"\[boltz2-opt\] PEAK-NOTE .*",   #   the per-item allocator peak (phase.py; every rank's)
    re.escape(f"{rep.PREFIX} [{rowpair_msa.FEATS_WHAT}] rank ") + r".*",   #   the per-batch input-feature census at n_gpu > 1 (rowpair_msa.feats_census, the core's line; every rank's: digest … feats_ranks_equal=)
    r"(?:[\w.]*\.)?RowpairRefused: .*",                              #   a x P lever that cannot run in the rank process names itself and its one opt-out (opt_core.mem.rowpair.RowpairRefused,
)                                                                   #   the traceback's last line); the run then exits failed (the units' FAILED lines + `worker exit code`), never degraded


def relay_lines(text: str, rank_logs: List[str]) -> List[str]:
    """The RELAY_LINES of the worker transcript in order, then — at n_gpu > 1 — those of every other rank's own transcript (``ranks/rank<r>.log``,
    r > 0; their PHASE lines stay there: one timing line per item, rank 0's). One function, one list: what the worker printed, verbatim."""
    import re
    pat = re.compile("^(?:" + "|".join(RELAY_LINES) + ")$", re.M)
    out = [m.group(0) for m in pat.finditer(text)]
    for p_ in sorted(rank_logs):
        m = re.search(r"rank(\d+)\.log$", os.path.basename(p_))
        if m and int(m.group(1)) > 0 and os.path.isfile(p_):
            out += [x.group(0) for x in pat.finditer(open(p_, errors="replace").read()) if not x.group(0).startswith("PHASE ")]
    return out


def peak_summary(text: str, rank_logs: List[str], n_gpu: int) -> str:
    """``[boltz2-opt] PEAK-SUMMARY items=<n> rank0_max_alloc_gib=<f> rankmax_alloc_gib=<f> rankmax_reserved_gib=<f> ranks=<P>`` reduced from the
    per-item PEAK lines (phase.PEAK_PREFIX): at n_gpu = 1 the worker transcript is rank 0; at n_gpu > 1 every rank's own transcript
    (``<workdir>/ranks/rank<r>.log``, the core launcher's) is read, rank 0's being ``rank0.log``."""
    import re
    pat = re.compile(r"\[boltz2-opt\] PEAK item=(\S+) alloc_gib=([0-9.]+) reserved_gib=([0-9.]+)$", re.M)
    per_rank = {0: text} if int(n_gpu or 1) == 1 or not rank_logs else {}
    for p_ in rank_logs:
        m = re.search(r"rank(\d+)\.log$", os.path.basename(p_))
        if m and os.path.isfile(p_):
            per_rank[int(m.group(1))] = open(p_, errors="replace").read()
    rows = sorted({(int(k), it, float(a), float(r)) for k, t in per_rank.items() for it, a, r in pat.findall(t)})
    f = lambda xs: f"{max(xs):.2f}" if xs else "-"   # noqa: E731
    return (f"{rep.PREFIX} PEAK-SUMMARY items={len({r[1] for r in rows})} rank0_max_alloc_gib={f([a for k, _, a, _ in rows if k == 0])} "
            f"rankmax_alloc_gib={f([a for _, _, a, _ in rows])} rankmax_reserved_gib={f([r for _, _, _, r in rows])} ranks={int(n_gpu or 1)}")


def parse_seeds(s: str) -> List[int]:
    return [int(x) for x in str(s).replace(";", ",").split(",") if x.strip()]
