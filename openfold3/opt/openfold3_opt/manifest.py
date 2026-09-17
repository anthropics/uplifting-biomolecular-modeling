"""The run record of a ``pred`` / ``warm`` call — what ran, in which mode, on what — built in memory for the exit rule (``build``). No file is
written beside the outputs. A RANK of the row-sharded line hands its record to the launcher that spawned it: ``dump`` writes it as
``<records dir>/rank<r>.json`` into the launcher's per-launch temporary directory (``RECORDS_ENV``, set by ``tp.launch`` for its ranks and removed
when the launcher has read rank 0's record).

Schema ``openfold3_opt/2``: the activation report; the lever record as of the write — ``levers_requested`` (the line's list),
``levers_applied`` / ``levers_unavailable`` / ``levers_pending`` from the add-ons' own records (stack.levers_record), ``partial`` and
``arm_complete`` (False when a requested lever is unavailable or a lever module was not the line's carrier's; None while pending;
stock runs: the environment proof's verdict); the versions, the GPU; the checkpoint (path, bytes, sha256, the pinned digest and
``is_pinned``; ``hashed: false`` in the stock caller's own proof, whose parent process hashed it) and ``weights_pinned`` at top level; the stock configuration's sha256, the
the command and argv (upstream's knobs exactly as given), the exit code, the number of structures written, the deterministic level, ``upstream_fix`` (the IDs
of the upstream-issue fixes the run applied under ``--upstream-fix``; ``[]`` without the flag) and (stock runs) the environment
proof.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import platform
from typing import Optional

from . import __version__
from . import env as _env
from . import report as _report
from . import modes

SCHEMA = "openfold3_opt/2"
RECORDS_ENV = "OPENFOLD3_OPT_RECORDS"                     # a rank's environment: the launcher's per-launch records directory (tp.launch); unset everywhere else


def sha256_file(path: str, *, strict: bool = False, home: Optional[str] = None) -> Optional[str]:
    """The sha256 of one file. This is stock/check_pins.py's hasher ON PURPOSE, not opt_core.gates.sha256_file: the stock route
    (`stock_pred`, `--mode off`) records the runner yaml's digest through this function and must load NOTHING of the core (its proof line
    `NOT STOCK: … core [...]` refuses the run otherwise), and the pin checker runs in the stock venv where opt_core is absent. None when the
    file cannot be read, or the OSError itself with `strict=True` (a carried file or the checkpoint that cannot be read is an error naming
    the file, never a 'differs' verdict)."""
    try:
        return _env.check_pins(home).sha256_file(path)
    except OSError:
        if strict:
            raise
        return None


def checkpoint_info(path, hash_it: bool = True, home: Optional[str] = None) -> dict:
    """The checkpoint record: the weights gate's dict when the caller already ran it (stack.weights_gate: sha256, the pinned digest,
    `is_pinned`), else the gate run here (`hash_it=False` records `hashed: false`, `is_pinned: None`)."""
    if isinstance(path, dict):
        return dict(path)
    if not path:
        return {"path": None, "exists": False, "hashed": False, "is_pinned": None}
    from . import stack
    return stack.weights_gate(path, home=home, hash_it=hash_it)


def count_structures(out_dir: Optional[str]) -> Optional[int]:
    """The writer's `<query>/seed_<s>/<query>_seed_<s>_sample_<k>_model.cif` files under out_dir (openfold3/core/runners/writer.py:234,241)
    whose atom coordinates are all finite: a file with a `nan` / `inf` in `_atom_site.Cartn_{x,y,z}` is not a structure — it is named once on
    stderr (`OUTPUT REJECTED file=<path> reason=nan_coordinates|unreadable`) and left out of the count the exit rule compares with the
    expected number (`rejected_structures` holds the names)."""
    if not out_dir or not os.path.isdir(out_dir):
        return None
    files = sorted(glob.glob(os.path.join(out_dir, "**", "seed_*", "*_model.cif"), recursive=True))
    so = structure_only(out_dir)                                            # structure first: a model file whose confidence was not written is NOT a structure of the tally
    ok = 0
    for f in files:
        if os.path.abspath(f) in so:
            if f not in uncounted_structures:
                uncounted_structures[f] = "confidence_not_written"
                import sys
                print(f"[{_report.TAG}] OUTPUT NOT COUNTED file={f} reason=confidence_not_written (structure-only model file: pLDDT/B-factor column 0.00)", file=sys.stderr, flush=True)
            continue
        reason = coordinate_defect(f)
        if reason is None:
            ok += 1
            continue
        if f not in rejected_structures:
            rejected_structures[f] = reason
            import sys
            print(f"[{_report.TAG}] OUTPUT REJECTED file={f} reason={reason}", file=sys.stderr, flush=True)
    return ok


rejected_structures: dict = {}                                            # path -> reason, every model file this process refused to count (named once each)


uncounted_structures: dict = {}                                             # path -> "confidence_not_written" (structure first: named once on stderr, left out of n_cif)


def structure_first_block(out_dir: Optional[str]) -> dict:
    """The core's structure-first scan of ``out_dir`` (opt_core.mem.rowpair.structure_first.scan): ``{files, confidence_not_written, structure_only_paths, notes}``;
    an empty block when the core module is absent (an older opt_core: no sidecars can exist)."""
    try:
        from opt_core.mem.rowpair.structure_first import scan
    except ImportError:
        return {"files": 0, "confidence_not_written": [], "structure_only_paths": [], "notes": []}
    return scan(out_dir)


def structure_only(out_dir: Optional[str]) -> set:
    """Absolute paths of the model files under ``out_dir`` whose sidecar says ``confidence_written: false``."""
    return {os.path.abspath(p) for p in structure_first_block(out_dir).get("structure_only_paths", [])}


def coordinate_defect(path: str) -> Optional[str]:
    """None when every `_atom_site.Cartn_x/y/z` value of the mmCIF parses as a finite float (or the file carries no atom_site loop to judge);
    else the reason word: `nan_coordinates` (a nan or inf coordinate, or a coordinate token that is not a number) or `unreadable:<Exc>`. Rows
    are read from the `_atom_site` loop as whitespace-separated tokens (the writer quotes no coordinate)."""
    import math
    try:
        with open(path, errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError as e:
        return f"unreadable:{type(e).__name__}"
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() == "loop_":
            j, names = i + 1, []
            while j < n and lines[j].startswith("_"):
                names.append(lines[j].strip()); j += 1
            if "_atom_site.Cartn_x" in names:
                idx = [names.index(f"_atom_site.Cartn_{a}") for a in ("x", "y", "z")]
                k = j
                while k < n and lines[k].strip() and not lines[k].startswith(("_", "loop_", "#")):
                    toks = lines[k].split()
                    try:
                        vals = [float(toks[m]) for m in idx]
                    except (IndexError, ValueError):
                        return "nan_coordinates"
                    if not all(math.isfinite(v) for v in vals):
                        return "nan_coordinates"
                    k += 1
                i = k
                continue
            i = j
            continue
        i += 1
    return None                                                           # no atom_site loop: nothing to judge, the file counts as written


def _stack_block(rep: dict) -> dict:
    """opt_core.manifest.stack_block: the box as found + the core block (version, package_dir). Imported here, not at module level:
    the stock child imports this module after its proof and may hold no core module beyond the proof machinery."""
    from opt_core import manifest as _core_manifest
    return _core_manifest.stack_block(gpu=rep.get("gpu"), torch_version=rep.get("torch") or modes.torch_version())


def build(report: Optional[dict], *, command: str, argv=None, exit_code=None, out_dir=None, checkpoint=None, runner_yaml=None,
          det=None, stock_proof=None, extra=None, upstream_fix=None) -> dict:
    rep = dict(report or {})
    if rep.get("active"):
        from . import stack
        rep.update(stack.levers_record(rep))                                # the add-ons' own records as of the write, not the line's list
    arm = rep.get("arm_complete")
    if not rep.get("active") and stock_proof is not None:
        arm = bool(stock_proof.get("ok")) if isinstance(stock_proof, dict) else None
    ckpt = checkpoint_info(checkpoint)
    man = {
        "schema": SCHEMA,
        "written_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": rep.get("mode"),
        "line": rep.get("line"),
        "active": bool(rep.get("active")),
        "levers_requested": list(rep.get("levers_requested") or []),      # the line's levers (switched on at activation; the add-ons apply them at import)
        "levers_applied": list(rep.get("levers_applied") or []),          # what the add-ons' own records say they installed, as of this write
        "levers_unavailable": list(rep.get("levers_unavailable") or []),  # requested, target imported, not installed (the kit's fallback)
        "levers_pending": list(rep.get("levers_pending") or []),          # requested, target not imported yet
        "levers_off": list(rep.get("levers_off") or []),                  # the line's levers MODEL_OPT_LEVERS_OFF left off for this run (modes.apply_levers_off; [] = the line as written)
        "compile": rep.get("compile") or "none",                          # the ACTIVE line's compile= word (modes.COMPILE_STATE: no mode of this kit compiles; --no-compile is a named no-op)
        "partial": bool(rep.get("partial")),
        "arm_complete": arm,
        "precision": rep.get("precision"),                                # the fast line's precision word (fp32 | bf16), None under any other mode
        "size_gate": rep.get("size_gate"),                                # the graph gate's fragment (opt_core.mem.graph_gate: graph=capture|eager:<reason>|off) or None
        "gate_reason": rep.get("gate_reason"),                                # the CUDA-graph size gate's word for this call (modes.graphs_gate), None when it does not apply
        "conf_gate": rep.get("conf_gate"),                                    # the confidence levers' size-gate fragment (modes.conf_gate: conf=served|gated:lt_min|n_tok_unknown min_tokens=N), None when the line carries none
        "conf_reason": rep.get("conf_reason"),                                # its decision word alone
        "reach_gate": rep.get("reach_gate"),                                  # the resident line's reach-gate fragment (modes.reach_gate), None when it did not apply
        "reach_reason": rep.get("reach_reason"),
        "reach_off": list(rep.get("reach_off") or []),                    # the levers it set aside for this run
        "n_tokens": rep.get("n_tokens"),                                  # the query's polymer token count the gate read (inputs.polymer_tokens)
        "lever_evidence": rep.get("lever_evidence"),
        "weights_pinned": ckpt.get("is_pinned"),
        "kit_counters": _report.kit_counters(),                             # the add-ons' own counters at write time (exit tally)
        "hooks": list(rep.get("hooks") or []),
        "reason": rep.get("reason"),
        "package_version": rep.get("package_version") or __version__,
        "openfold3_version": rep.get("openfold3_version") or modes.openfold3_version(),
        "torch": rep.get("torch") or modes.torch_version(),
        "triton": rep.get("triton") or modes.triton_version(),
        "python": platform.python_version(),
        "gpu": rep.get("gpu"),
        "checkpoint": ckpt,
        "runner_yaml": {"path": runner_yaml, "sha256": sha256_file(runner_yaml) if runner_yaml else None},
        "det": det,
        "upstream_fix": list(upstream_fix or []),                          # --upstream-fix: the IDs of the upstream-issue fixes applied on this run (upstream_fix.py); [] = upstream's behaviour as shipped
        "command": command,
        "argv": list(argv) if argv is not None else None,
        "exit_code": exit_code,
        "n_cif": count_structures(out_dir),
        "openfold3_opt_env": os.environ.get("OPENFOLD3_OPT"),
        "env": rep.get("env"),
        "stock_env_proof": stock_proof,
        "activation_report": rep,
        "stack": _stack_block(rep),
    }
    sf = structure_first_block(out_dir)                                     # structure first: the sidecars under the output dir -> record block + notes (n_cif above excludes structure-only files)
    if sf["files"]:
        man["structure_first"] = {"files": sf["files"], "confidence_not_written": sf["confidence_not_written"]}
        if sf["notes"]:
            man["notes"] = list(man.get("notes") or []) + sf["notes"]
    if extra:
        man.update(extra)
    return man


def dump(path: str, man: dict) -> str:
    """Write a built record to ``path`` atomically (a rank's hand-over to its launcher: build first, gate, then dump — the record's `exit_code` is the process's)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
