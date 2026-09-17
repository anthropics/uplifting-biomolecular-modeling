"""The kit route (mode ``exact``): the kit's own executable on the stock arguments plus the mode's lever flags, from a staged copy.

Base variants: ``python <stage>/mpnn_pdb_parser/kit/mpnn_worker2.py --jsonl_path <parsed> --chain_id_jsonl <chains> --out_folder
<out> --path_to_model_weights $MPNN_DIR/<variant>_model_weights --model_name v_48_020 <stock options> <lever flags>`` — the README row
with its stock options replaced by the pass's (settings.kit_argv; modes.resolve keeps the lever flags only). The parsed.jsonl comes from the kit's read-once
parser ``kit/fast_parse.py`` when the input is a directory (the stock helper's bytes, the helper's own entry order; a parsed.jsonl the
caller gives is read as given); ``<chains>`` is the caller's --chain_id_jsonl verbatim, else ``<stage>/unassigned.jsonl`` (inputs.write_unassigned:
one JSON line ``null`` — the worker's required argument carrying upstream's default, protein_mpnn_run.py's chain_id_dict None: every chain of
each entry designed, none fixed; removed with the stage); the weights directory is the variant's unless the pass names it by upstream's own
selectors (settings.base_weights_dir). One PDB file (protein_mpnn_run.py's ``--pdb_path``) is parsed by the same parser from a staged directory
holding that file alone (upstream's parse_PDB reads it with the same statements as its directory helper) and its chains are assigned as
protein_mpnn_run.py assigns a ``--pdb_path`` input's (inputs.write_pdb_assignment: ``--pdb_path_chains``, else every chain; upstream reads no
``--chain_id_jsonl`` for one file, so neither route applies one). A pass the worker cannot serve never reaches this module: ``design`` refuses it by name
first (settings.worker_refuses). The worker reads ``MPNN_DIR`` itself (stack.kit_env carries it into the child) for protein_mpnn_utils and the checkout's commit.

One CMD line (report.log_cmd: route ``kit``, the variant, the executable's whole argv) precedes each process; the scratch directory is
temporary and removed when the pass ends. The kit's own lines reach the caller's streams and the kit lines
(``DEVICE CELL``, ``hybrid_gemm: PROBE ...``, ``RUN_WALL_S``, the worker's timing json line) are returned for the manifest, unedited.
The worker's timing json line is also its own record of the flags it ran with (``mode``, ``chunk_gemm``, ``graph_rng``, ... after its own
CPU rule and any other rule of the carried bytes): ``lever_evidence`` reads the requested composition against it — the evidence of
application behind the ``partial`` verdict.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Tuple, Dict, List, Optional

from . import inputs as _inputs
from . import report as _report
from . import settings as _settings
from . import stack
from . import stage as _stage
from .modes import LOWMEM as _modes_LOWMEM, Resolution

KIT_LINES = {"device_cell": r"^DEVICE CELL:.*$", "probe": r"^.*hybrid_gemm: PROBE (PASS|FAIL).*$", "run_wall": r"^.*RUN_WALL_S.*$"}


class KitRunError(RuntimeError):
    """The kit route could not be launched (no MPNN_DIR / checkpoint, an unsupported option)."""


def kit_lines(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key, pat in KIT_LINES.items():
        hits = [ln for ln in text.splitlines() if re.match(pat, ln)]
        if hits:
            out[key] = hits[:8]
    return out


def worker_record(text: str) -> Optional[dict]:
    """The worker's own end-of-run JSON line (its timings and the hybrid_gemm probe cell: device, sm_arch, torch, cuda, verdict; or the record of a job it
    refused by name before any output, ``refused``), unedited."""
    for ln in reversed(text.splitlines()):
        ln = ln.strip()
        if ln.startswith("{") and ('"hybrid_gemm_probe"' in ln or '"refused"' in ln):
            try:
                return json.loads(ln)
            except json.JSONDecodeError:
                return None
    return None


def probe_word(cell: dict | None) -> str:
    """The worker's probe cell (its end-of-run record's ``hybrid_gemm_probe``) -> the one word of its verdict sentence: ``PASS`` | ``FAIL``
    (``PROBE PASS -> ...`` / ``PROBE FAIL -> ...``, mpnn_worker2.py decide_hybrid), ``not requested`` when the worker was not asked,
    ``unobserved`` when the cell says neither (named, never guessed)."""
    verdict = str((cell or {}).get("verdict") or "")
    m = re.match(r"\s*PROBE (PASS|FAIL)\b", verdict)
    if m:
        return m.group(1)
    return "not requested" if verdict.startswith("not requested") else "unobserved"


def lever_evidence(res: Resolution, record: Optional[dict]) -> dict:
    """Each requested lever (``res.levers``) against the worker's own end-of-run record: ``applied`` (the record shows it on),
    ``dropped`` (the worker ran without it — its CPU rule, or any rule of the carried bytes), ``gated`` (the kit's own gate switched it off
    and says so: the hybrid_gemm probe verdict), ``unobserved`` (the record carries no field for it: named, never judged). A ``--mode``
    value is read from the record's ``mode`` field. Without a record (a worker that died before its last line)
    nothing is judged and ``detection`` names the absence."""
    ev: dict = {"applied": [], "dropped": [], "gated": {}, "unobserved": [], "detection": None}
    if not record:
        ev["detection"] = "none (the worker printed no end-of-run record)"
        return ev
    mode_value = next((res.flags[i + 1] for i in range(len(res.flags) - 1) if res.flags[i] == "--mode"), None)
    for name in res.levers:
        if name == "hybrid_gemm" and record.get("hybrid_gemm") and not record.get("hybrid_gemm_active"):
            probe = record.get("hybrid_gemm_probe") if isinstance(record.get("hybrid_gemm_probe"), dict) else {}
            ev["gated"][name] = "hybrid_gemm probe: " + probe_word(probe)
        elif name in record:
            (ev["applied"] if record[name] else ev["dropped"]).append(name)
        elif name == mode_value and "mode" in record:
            (ev["applied"] if record["mode"] == name else ev["dropped"]).append(name)
        else:
            ev["unobserved"].append(name)
    return ev


def _launch(cmd: List[str], env: dict, cwd: Optional[str], echo: bool = True) -> dict:
    """Run a kit executable to the end; its two streams are relayed to the caller's (``echo``) once it exits and read for the kit lines and the worker record."""
    t0 = time.time()
    stack.mark_launched()
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_b, err_b = proc.communicate()
    text = out_b.decode(errors="replace") + "\n" + err_b.decode(errors="replace")
    if echo:
        sys.stdout.write(out_b.decode(errors="replace")); sys.stderr.write(err_b.decode(errors="replace"))
        sys.stdout.flush(); sys.stderr.flush()
    return {"cmd": cmd, "cwd": cwd, "rc": proc.returncode, "wall_s": time.time() - t0, "lines": kit_lines(text), "worker_record": worker_record(text)}

ONE_PDB = "one_pdb"                # <stage>/one_pdb/: the staged directory holding the one PDB file of a single-PDB input, for the parser
INPUTS_READY = "inputs.ready"      # <stage>/inputs.ready: created by run() once the parse step has written the parsed jsonl; the worker (started
                                   # first, --inputs_ready) does its input-independent start-up meanwhile and reads its inputs when the file exists

def _launch_overlapped(parse_cmd: List[str], worker_cmd: List[str], ready: str, env: dict, echo: bool = True) -> Tuple[dict, Optional[dict]]:
    """The parse step and the worker started together: the worker's input-independent start-up (interpreter, torch, CUDA context, weights,
    the draw-kernel probe) runs while the inputs are parsed; `ready` is created when the parse step exits 0 and the worker reads the parsed
    jsonl then. A failing parse step ends the worker before it has read anything. Streams are relayed as each child exits (parse first);
    returns (parse record, worker record | None when the parse step failed)."""
    t0 = time.time()
    stack.mark_launched()
    pp = subprocess.Popen(parse_cmd, env=env, cwd=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wp = subprocess.Popen(worker_cmd, env=env, cwd=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_b, err_b = pp.communicate()
    ptext = out_b.decode(errors="replace") + "\n" + err_b.decode(errors="replace")
    if echo:
        sys.stdout.write(out_b.decode(errors="replace")); sys.stderr.write(err_b.decode(errors="replace")); sys.stdout.flush(); sys.stderr.flush()
    pr = {"cmd": parse_cmd, "cwd": None, "rc": pp.returncode, "wall_s": time.time() - t0, "lines": kit_lines(ptext), "worker_record": None, "step": "parse"}
    if pp.returncode != 0:
        wp.kill(); wp.communicate()
        return pr, None
    open(ready, "w").close()
    out_b, err_b = wp.communicate()
    wtext = out_b.decode(errors="replace") + "\n" + err_b.decode(errors="replace")
    if echo:
        sys.stdout.write(out_b.decode(errors="replace")); sys.stderr.write(err_b.decode(errors="replace")); sys.stdout.flush(); sys.stderr.flush()
    wr = {"cmd": worker_cmd, "cwd": None, "rc": wp.returncode, "wall_s": time.time() - t0, "lines": kit_lines(wtext), "worker_record": worker_record(wtext), "step": "design"}
    return pr, wr

def run(res: Resolution, input_path: str, out_dir: str, stock_args: Optional[List[str]] = None, chain_id_jsonl: Optional[str] = None,
        fixed_positions_jsonl: Optional[str] = None, echo: bool = True, python: str = sys.executable) -> dict:
    """One kit pass for a resolved kit mode. Returns {route, rc, wall_s, commands, parsed, assigned (the caller's --chain_id_jsonl, else None), n_inputs, lines, worker_record[, lowmem]}."""
    os.makedirs(out_dir, exist_ok=True)
    env = stack.kit_env()
    rec: dict = {"route": res.route, "variant": res.variant, "commands": [], "rc": None, "wall_s": 0.0, "n_inputs": None, "lines": {}, "worker_record": None}
    pairs = _settings.parse(stock_args, res.variant)
    knobs = _settings.kit_argv(pairs, res.variant)                            # the stock options given, for the kit executable (minus the worker's built-ins)
    stage_dir = tempfile.mkdtemp(prefix="proteinmpnn_opt_stage_")
    try:
        mdir = stack.mpnn_dir()
        if not mdir or not os.path.isfile(os.path.join(mdir, "protein_mpnn_utils.py")):
            raise KitRunError(f"{stack.ENV_MPNN_DIR}={mdir!r} does not hold protein_mpnn_utils.py (the worker imports it from MPNN_DIR)")
        paths = _stage.stage_base(stage_dir)
        if res.transform == _modes_LOWMEM:                                 # the row's worker runs as kit/mpnn_worker2_lowmem.py (stage.stage_lowmem)
            rec["lowmem"] = {"executable": _stage.LOWMEM_WORKER}
        kind = _inputs.classify_base(input_path)
        rec["order"] = "given"                                             # a parsed jsonl or one PDB file: the caller's order (a directory: the parser's, below)
        parsed = os.path.join(out_dir, _inputs.PARSED)
        parse_cmd = None
        if kind == "dir":
            parse_cmd = [python, paths["fast_parse"], "--input_path", os.path.abspath(input_path), "--output_path", parsed]
            rec["parser"] = "kit"                                          # the kit's read-once parser (the stock helper's bytes)
            rec["order"] = "parse helper (file-system order)"
        elif kind == "pdb":                                                # one PDB file: the same parser on a staged directory holding that file alone, run to the end first (one file)
            one = os.path.join(stage_dir, ONE_PDB)
            os.makedirs(one)
            os.symlink(os.path.abspath(input_path), os.path.join(one, os.path.basename(input_path)))
            pcmd = [python, paths["fast_parse"], "--input_path", one, "--output_path", parsed]
            rec["parser"] = "kit"
            _report.log_cmd("kit", res.variant, pcmd)
            r = _launch(pcmd, env, None, echo)
            r["step"] = "parse"
            rec["commands"].append(r); rec["wall_s"] += r["wall_s"]
            if r["rc"] != 0:
                rec["rc"] = r["rc"]
                return rec
        else:
            parsed = os.path.abspath(input_path)
        rec["parsed"] = parsed
        if kind == "pdb":                                                  # protein_mpnn_run.py assigns a --pdb_path input's chains itself (--pdb_path_chains, else every chain) and reads no --chain_id_jsonl for it
            rec["assigned"] = None
            rec["pdb_path_chains"] = _settings.pdb_path_chains(pairs)
            chains = _inputs.write_pdb_assignment(parsed, rec["pdb_path_chains"], os.path.join(stage_dir, _inputs.PDB_ASSIGNED))
        else:
            rec["assigned"] = os.path.abspath(chain_id_jsonl) if chain_id_jsonl else None   # the caller's --chain_id_jsonl verbatim, or none
            chains = rec["assigned"] or _inputs.write_unassigned(os.path.join(stage_dir, _inputs.UNASSIGNED))   # none given: the worker's required --chain_id_jsonl carries upstream's default (`null` -> chain_id_dict None: every chain designed)
        weights = _settings.base_weights_dir(pairs, res.variant, mdir)
        cmd = [python, paths["worker_lowmem"] if res.transform == _modes_LOWMEM else paths["worker"], "--jsonl_path", parsed, "--chain_id_jsonl", chains, "--out_folder", os.path.abspath(out_dir),
               "--path_to_model_weights", weights] + knobs
        if fixed_positions_jsonl:
            cmd += ["--fixed_positions_jsonl", os.path.abspath(fixed_positions_jsonl)]
        cmd += list(res.flags)
        if parse_cmd is not None:
            ready = os.path.join(stage_dir, INPUTS_READY)
            cmd += ["--inputs_ready", ready]
            _report.log_cmd("kit", res.variant, parse_cmd)                 # the CMD lines: the parse step's argv and the worker's whole argv, before they start
            _report.log_cmd("kit", res.variant, cmd)
            r, rw = _launch_overlapped(parse_cmd, cmd, ready, env, echo)
            rec["commands"].append(r); rec["wall_s"] = r["wall_s"]
            if rw is None:
                rec["rc"] = r["rc"]
                return rec
            rec["n_inputs"] = _inputs.parse_count(parsed)
            rec["commands"].append(rw); rec["wall_s"] = rw["wall_s"]; rec["rc"] = rw["rc"]; rec["lines"] = rw["lines"]; rec["worker_record"] = rw.get("worker_record")
        else:
            rec["n_inputs"] = _inputs.parse_count(parsed)
            _report.log_cmd("kit", res.variant, cmd)                       # the CMD line: the worker's whole argv, before it starts
            r = _launch(cmd, env, None, echo)
            r["step"] = "design"
            rec["commands"].append(r); rec["wall_s"] += r["wall_s"]; rec["rc"] = r["rc"]; rec["lines"] = r["lines"]; rec["worker_record"] = r.get("worker_record")
    finally:
        _stage.remove(stage_dir)
    return rec
