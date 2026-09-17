"""``design`` — the upstream tool's own action, on upstream's own input and nothing else: the hydra ``KEY=VALUE`` overrides of one
``scripts/run_inference.py`` invocation (``inference.input_pdb=… 'contigmap.contigs=[…]' ['ppi.hotspot_res=[…]'] inference.output_prefix=…
inference.num_designs=… inference.design_startnum=… …``: one target, outputs at the typed prefix exactly as upstream writes them, the run
record — ``opt_manifest.json``, ``run.log``, ``run_timings.json`` — beside them in the prefix's directory). Internally a request is a list
of case rows (``{name, pdb, contigs, hotspots, num_designs, startnum, prefix}`` — the resident driver's ``--cases`` file): the command line's
one target (case_from_overrides), or ``warm``'s bundled example targets (warm.py hands its rows to run(); their prefixes are
``<out_dir>/<name>/des``, outputs.case_prefix).

* ``--mode off``: per case, the stock caller (stock_cli.py) in a clean subprocess: ``python -s $RFD_ROOT/scripts/run_inference.py`` with the
  typed overrides verbatim (``inference.model_directory_path=$WEIGHTS`` appended when not typed; a warm-up row without a typed prefix is
  composed from the row: its inputs, ``inference.output_prefix=<out>/<case>/des``, ``num_designs`` / ``design_startnum``),
  plus the recipe's seed under ``--det 1`` (det.py), plus a ``stock_env_proof.json`` per case.
* ``--mode exact|fast`` (a driver mode): the mode's line from modes.resolve — the resident driver with the README flag
  set and the mode's environment row (stack.driver_command / driver_environment), ONE resident process for all cases (the kit's tested shape:
  the model, the JIT specialisations and the graphs persist across cases), its evidence lines and its numerics record read back from the log
  and the timings file (registry.Lever.evidence, report.numerics_from_timings). The launch's values
  of upstream's keys are composed on every configuration the driver builds (driver_run.py): typed-or-default on the keys the driver
  hard-codes (modes.DRIVER_FIXED), every other typed override verbatim (guiding potentials, noise schedules, partial diffusion, …: read there
  by upstream's own sampler code). What the kit line cannot serve — symmetric oligomers, cyclic peptides, fold conditioning, sequence
  inpainting, a typed architecture / trained-schedule key, another ``--config-name``, a Hydra application flag (upstream_args.refused, each
  with its mechanism) — is refused by name before anything runs: ``[rfdiffusion1-opt] NOT ACTIVE: mode=<m> cannot serve …``, exit 3
  (refuse_unserved); ``--mode off`` runs it.
* ``--pack K``: the packed line (serve.py).

The exit rule after the pass is report.verdict.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

from . import det, manifest as _manifest, outputs as _outputs, report as _report, stack, upstream_args
from .modes import numerics
from .modes import MODE_NAMES, MODES, Resolution, default_mode, resolve
from .registry import FORBIDDEN_LINES, KIT_DIRS, LEVERS
from opt_core.oom import is_oom                                                       # the core's one out-of-memory classifier: an OOM a driver names in its log is never a pass (read_evidence)
from .report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE   # noqa: F401 — the codes' one home is report.py; re-exported for callers

NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")                                              # a case row's name (it becomes a directory / a log label)
NULL = "null"                                                                          # hydra's null word (base.yaml: input_pdb, contigs, hotspot_res default to it)
UNSERVED_FMT = "mode={mode} cannot serve {reasons} — refused by name, nothing ran; `--mode off` runs this request on upstream's command line (scripts/run_inference.py)"


class DesignError(ValueError):
    """A refused input or run, with the named fact."""


class UsageError(DesignError):
    """The command line names no target (exit 2)."""


# ------------------------------------------------------------------------------------------------------------------------- cases
def case_from_overrides(overrides, cwd: Optional[str] = None) -> dict:
    """The one target of a run_inference.py invocation, read from the typed hydra overrides with upstream's defaults and upstream's own
    start-number rule (upstream_args.target): a case row whose `prefix` is the typed inference.output_prefix (absolute)."""
    ua = upstream_args.parse([str(o) for o in (overrides or [])])
    t = upstream_args.target(ua, cwd)
    if t.pdb == NULL and t.contigs == NULL and not upstream_args.switched_on("scaffoldguided.scaffoldguided", ua.value("scaffoldguided.scaffoldguided")):   # fold-conditioned design names its target by scaffoldguided.* (a --mode off request)
        raise UsageError("no target: type upstream's overrides (inference.input_pdb=… 'contigmap.contigs=[…]' inference.output_prefix=… …) as scripts/run_inference.py takes them")
    row = {"name": NAME_RE.sub("_", t.name) if not NAME_RE.match(t.name) else t.name, "pdb": t.pdb, "contigs": t.contigs, "hotspots": t.hotspots,
           "num_designs": t.num_designs, "startnum": t.startnum, "prefix": t.prefix,
           "startnum_compose": ua.value("inference.design_startnum")}                     # the typed token verbatim (None when untyped): what the driver composes on its configuration, as the stock command line carries it (driver_run.case_startnums)
    if t.ckpt_override:
        row["ckpt_path"] = t.ckpt_override
    return row


def write_cases(cases: List[dict], directory: str, out_dir: str, name: str = "cases.json") -> str:
    """The driver's cases file: every row with its output prefix spelled out (`prefix`: the typed one, or `<out_dir>/<name>/des` for a warm-up row — outputs.case_prefix)."""
    os.makedirs(directory, exist_ok=True)
    p = os.path.join(directory, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump([dict(c, prefix=_outputs.case_prefix(c, out_dir)) for c in cases], fh, indent=1)
    return p


def scratch_dir(tag: str) -> str:
    """The directory the pass's cases file(s) go to: outside the outputs (upstream's layout stays upstream's)."""
    return tempfile.mkdtemp(prefix=f"rfdiffusion1_opt_{tag}_")


def refusals(mode: Optional[str], overrides) -> List[str]:
    """What the kit line of `mode` cannot serve in this request, one named reason per token (upstream_args.refused); empty = it serves the
    request whole. `off` (upstream's command line) serves everything; no mode word = the default kit line (modes.DEFAULT_MODE)."""
    if (mode or default_mode()) == "off":
        return []
    return upstream_args.refused(upstream_args.parse([str(o) for o in (overrides or [])]))


def refuse_unserved(mode: Optional[str], overrides) -> Optional[Tuple[int, dict]]:
    """The one refusal of a request the kit line cannot serve, shared by design, the packed line, check and warm: prints
    ``[rfdiffusion1-opt] NOT ACTIVE: mode=<m> cannot serve <feature [token]: mechanism; …> — refused by name, nothing ran; `--mode off` runs …``
    and returns (EXIT_NOT_ACTIVE, {status: refused, reason, refused}) before anything is resolved, armed or run; None when the mode serves it."""
    m = mode or default_mode()
    why = refusals(m, overrides)
    if not why:
        return None
    reason = UNSERVED_FMT.format(mode=m, reasons="; ".join(why))
    _report.emit(_report.not_active_line(reason))
    return EXIT_NOT_ACTIVE, {"status": "refused", "reason": reason, "mode": m, "refused": why}


def compose_overrides(mode: Optional[str], overrides) -> List[str]:
    """The typed tokens the mode's line takes: all of them verbatim on `off` (upstream's parser answers for each); on a kit line the override
    tokens (a served ``--config-name base`` names what the resident driver composes anyway and is left out)."""
    ov = [str(o) for o in (overrides or [])]
    return ov if (mode or default_mode()) == "off" else upstream_args.parse(ov).compose_tokens()


def request(mode: Optional[str], overrides) -> Tuple[str, Optional[Tuple[int, dict]], Optional[List[str]]]:
    """The one admission every route runs before anything is resolved, armed or created (design, the packed line, check, warm): the effective
    mode (typed, else RFDIFFUSION1_OPT, else modes.DEFAULT_MODE), the refusal of what that mode's line cannot serve (refuse_unserved: the NOT
    ACTIVE line printed, (EXIT_NOT_ACTIVE, record)) or None, and the typed tokens as the mode's line takes them (compose_overrides; None when
    refused or when nothing was typed)."""
    m = stack._mode_from_env(mode) or default_mode()
    refused = refuse_unserved(m, overrides)
    return m, refused, (None if refused or overrides is None else compose_overrides(m, overrides))


def kit_dir_of(prefix_case: dict, out_dir: str) -> str:
    """Where a case's run-record files go: `out_dir` itself (the typed prefix's directory: no directory upstream lacks), `<out_dir>/<name>/` for a warm-up row without a prefix."""
    return out_dir if prefix_case.get("prefix") else os.path.join(out_dir, prefix_case["name"])


# --------------------------------------------------------------------------------------------------------------------- stock arm
def stock_overrides(case: dict, out_dir: str, weights: str, res: Resolution, det_flag: bool = False) -> List[str]:
    """The override list of the stock arm for one case. The command line's target (`prefix` set): the typed overrides verbatim,
    `inference.model_directory_path` appended when not typed, the recipe's seed under `--det 1`. A warm-up row (no typed prefix): the base driver's
    own per-case composition (rfd_bench.py:100-103, build_conf) — the row's inputs, `inference.output_prefix=<out>/<case>/des`, `num_designs` /
    `design_startnum` — then the seed under `--det 1`, the typed overrides, its checkpoint override."""
    typed = list(res.settings.stock_overrides)
    typed_keys = {o.split("=", 1)[0] for o in typed}
    if case.get("prefix"):
        ov = list(typed)
        if "inference.model_directory_path" not in typed_keys and weights:
            ov.append(f"inference.model_directory_path={weights}")
        return ov + det.stock_overrides(det_flag, typed)
    prefix = _outputs.case_prefix(case, out_dir)
    ov = [f"inference.input_pdb={case['pdb']}", f"inference.output_prefix={prefix}", f"inference.model_directory_path={weights}",
          f"inference.num_designs={case['num_designs']}", f"inference.design_startnum={case.get('startnum', 0)}",
          f"contigmap.contigs={case['contigs']}", f"ppi.hotspot_res={case['hotspots']}"]
    ov += det.stock_overrides(det_flag, typed)
    ov += typed
    if case.get("ckpt"):
        ov.append(f"inference.ckpt_override_path={os.path.join(weights, case['ckpt'])}")
    return ov


def stock_environment(environ=None) -> Tuple[Dict[str, str], List[str], List[str]]:
    """The stock process's environment: every name under stock/PINS.json must_be_absent_prefixes stripped except the allowed exceptions,
    DGL's backend word set, PYTHONPATH entries inside a kit directory removed. Returns (env, stripped names, the names to prove absent)."""
    environ = os.environ if environ is None else environ
    se = stack.pins()["stock_environment"]
    absent, allowed = list(se["must_be_absent_prefixes"]), list(se.get("allowed_exceptions", []))
    env, stripped = {}, []
    for k, v in environ.items():
        hit = any((n.endswith("_") and k.startswith(n)) or k == n for n in absent) and k not in allowed
        if hit:
            stripped.append(k)
            continue
        env[k] = v
    env.setdefault(*stack.DGL_BACKEND)
    kits = [stack.kit_dir(k) for k in KIT_DIRS]
    if env.get("PYTHONPATH"):
        keep = [p for p in env["PYTHONPATH"].split(os.pathsep) if p and not any(os.path.realpath(p).startswith(os.path.realpath(d)) for d in kits)]
        env["PYTHONPATH"] = os.pathsep.join(keep)
        if not env["PYTHONPATH"]:
            del env["PYTHONPATH"]
    return env, sorted(stripped), absent


def stock_command(case: dict, out_dir: str, rfd: str, weights: str, res: Resolution, python: Optional[str] = None, det_flag: bool = False) -> List[str]:
    """``python -s -m rfdiffusion1_opt.stock_cli --proof-json ... --env-absent ... --kit-dirs ... --rfd-root ... -- <overrides>``."""
    se = stack.pins()["stock_environment"]
    proof = os.path.join(kit_dir_of(case, out_dir), "stock_env_proof.json")
    kits = os.pathsep.join(stack.kit_dir(k) for k in KIT_DIRS)
    return ([python or sys.executable, "-s", "-m", "rfdiffusion1_opt.stock_cli", "--proof-json", proof, "--env-absent", ",".join(se["must_be_absent_prefixes"]),
             "--allowed", ",".join(se.get("allowed_exceptions", [])), "--kit-dirs", kits, "--rfd-root", rfd, "--"] + stock_overrides(case, out_dir, weights, res, det_flag))


# ------------------------------------------------------------------------------------------------------------------- evidence
# the kits' own final counters (<tag>_timings.json, report.STATS_KEYS) a lever must satisfy at the end of a pass: (timings key, counter,
# predicate on its value, the words of the defect). A pass whose lever names a degraded path here is `partial` — the same exit rule as a
# missing applied-line: the outputs are kept and the pass exits 3 (report.verdict).
COUNTER_RULES = {
    "W1": (("fullgraph_final", "eager_fallback_calls", lambda v: v == 0, "forward calls served eagerly after a failed capture"),
           ("fullgraph_final", "capture_errors", lambda v: not v, "CUDA-graph capture errors"),
           ("fullgraph_final", "n_capture", lambda v: v >= 1, "no graph was captured")),
    # K2's `n_fallback` counts the LayerNorm calls the lever ROUTES to torch by rule (inputs below RFD_TRITON_LN_MIN_NUMEL, a width outside
    # 8..1024, CPU tensors — opt_core/kernels/rfd_layernorm.py triton_layer_norm): routing, not degradation, so it is not gated; `reason` is non-empty only when the Triton
    # path itself was lost (import / fast-launch failure), and `n_triton == 0` means no call took the kernel — both are defects of the line
    "K2": (("triton_ln_final", "n_triton", lambda v: v > 0, "no LayerNorm call ran the Triton kernel"),
           ("triton_ln_final", "reason", lambda v: not v, "the lever recorded a degraded path")),
}


def numerics_defects(num: Optional[dict]) -> List[str]:
    """The numerics gate: the driver's per-case torch read-back (numerics(), source=torch) must equal the line's declaration — TF32 planned ⇒
    matmul and cuDNN TF32 both on in every case; not planned ⇒ both off (exact never runs TF32). A mismatch, a record lacking either TF32 key,
    or an unreadable record on a line that declares a policy, is a defect line of the pass (the same exit rule as a missing applied-line),
    never a note."""
    if num is None:
        return ["numerics: the driver's torch numerics record is unreadable (no per-case `precision` block) — the line's TF32 state is unproven"]
    if num.get("source") != "torch":
        return []
    return ([f"numerics: {k} absent from the driver's torch record — the line's TF32 state is unproven" for k in (num.get("missing") or [])]
            + [f"numerics: {k} recorded {num.get(k)!r}, the line declares {NUMERICS_DECL(num, k)!r}" for k in (num.get("mismatch") or [])])


def NUMERICS_DECL(num: dict, key: str):
    from .modes import NUMERICS
    pol = num.get("policy")
    for m, d in NUMERICS.items():
        if d.get("policy") == pol:
            return d.get(key) if key != "dtype" else "torch.float32"
    return None


def designs_computed(timings_path: str) -> Optional[Tuple[int, int]]:
    """(computed, skipped) for the pass, from the driver's ``<tag>_timings.json``: the design rows it computed (``cases[].designs``) and the
    designs upstream's ``inference.cautious`` rule skipped because their ``<prefix>_<i>.pdb`` existed (``cases[].skipped_existing``:
    stock/src/scripts/run_inference.py:78-82, the driver's same rule). None when the file is unreadable."""
    try:
        with open(timings_path, encoding="utf-8") as fh:
            r = json.load(fh)
    except (OSError, ValueError):
        return None
    cases = [c for c in (r.get("cases") or []) if isinstance(c, dict)]
    return (sum(len(c.get("designs") or []) for c in cases), sum(len(c.get("skipped_existing") or []) for c in cases))


def all_skipped(timings_path: str) -> bool:
    """True when the pass computed no design because every requested one already existed (upstream's inference.cautious rule): such a pass
    made no forward call, so the per-call counters (IO1's confirmed writes, W1's captures, K2's calls) have nothing to hold the levers to."""
    n = designs_computed(timings_path)
    return n is not None and n[0] == 0 and n[1] > 0


def counter_defects(timings_path: str, levers: List[str]) -> List[str]:
    """The defects the kits' final counters name for these levers (empty = none): per COUNTER_RULES, a counter absent, or failing its
    predicate, is one defect line `<lever>: <words> (<key>.<counter>=<value>)`; an unreadable / unfinished timings file is one defect per
    gated lever (the pass did not reach its final counters)."""
    gated = [l for l in levers if l in COUNTER_RULES]
    if not gated:
        return []
    try:
        r = json.load(open(timings_path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{l}: final counters unreadable ({timings_path}: {e!r})" for l in gated]
    out = []
    for l in gated:
        for key, counter, ok, words in COUNTER_RULES[l]:
            block = r.get(key)
            if not isinstance(block, dict) or counter not in block:
                out.append(f"{l}: {words} — counter absent ({key}.{counter}; the pass did not write its final counters)")
                continue
            v = block[counter]
            try:
                good = bool(ok(v))
            except (TypeError, ValueError):
                good = False
            if not good:
                out.append(f"{l}: {words} ({key}.{counter}={v!r})")
    return out


IO_FINAL = "PDBIO_FINAL"                                             # lever IO1's exit line in the driver log (driver_run: `... PDBIO_FINAL {json}`)


def io_counters(log_path: str) -> Optional[dict]:
    """Lever IO1's final counters from the driver log (the last `PDBIO_FINAL {json}` line), or None when the line is absent / unreadable."""
    try:
        text = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    found = None
    for line in text.splitlines():
        k = line.find(IO_FINAL + " {")
        if k >= 0:
            try:
                found = json.loads(line[k + len(IO_FINAL) + 1:])
            except ValueError:
                found = None
    return found if isinstance(found, dict) else None


def io_defects(counters: Optional[dict], levers: List[str]) -> List[str]:
    """The defects lever IO1's counters name (empty = none, or IO1 not among the levers): the exit line absent, a byte mismatch against upstream's
    writer, or no call confirmed against it — the writer's proof runs in every mode that carries the lever, never as a debug switch."""
    if "IO1" not in levers:
        return []
    if counters is None:
        return [f"IO1: final counters absent ({IO_FINAL}; the pass did not reach the writer's exit line)"]
    out = []
    if counters.get("n_mismatch"):
        out.append(f"IO1: bytes unlike rfdiffusion.util's writer ({IO_FINAL}.n_mismatch={counters.get('n_mismatch')!r} on {counters.get('mismatches')})")
    if not counters.get("n_verified"):
        out.append(f"IO1: no call confirmed against rfdiffusion.util's writer ({IO_FINAL}.n_verified={counters.get('n_verified')!r})")
    return out


def read_evidence(log_path: str, levers: List[str]) -> dict:
    """Which levers printed their own applied-line in the driver log, which did not, and the forbidden lines found. A forbidden line is
    one of registry.FORBIDDEN_LINES or a line the core's classifier reads as an out-of-memory (opt_core.oom.is_oom on the line's text:
    a driver that met an OOM — raised, or recorded by a lever's own counters such as `capture_errors` — never yields a pass)."""
    log_error = None
    try:
        text = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError as e:
        text, log_error = "", repr(e)                                            # an unreadable log: every lever reads as missing, and the cause is named beside it
    lines = text.splitlines()
    applied, missing = [], []
    for lid in levers:
        lv = LEVERS[lid]
        if not lv.evidence:
            continue
        rx = re.compile(lv.evidence)
        (applied if any(rx.search(ln) for ln in lines) else missing).append(lid)
    forbidden = [ln for ln in lines if any(re.search(f, ln) for f in FORBIDDEN_LINES) or is_oom(RuntimeError(ln))]
    return {"applied": applied, "missing": missing, "forbidden": forbidden[:20], "n_lines": len(lines), "log_error": log_error}


# ------------------------------------------------------------------------------------------------------------------------ runs
def _run(cmd: List[str], env: Dict[str, str], cwd: str, log_path: str, timeout: Optional[float]) -> Tuple[int, float, float]:
    """Run one child process with its output appended to log_path. Returns (rc, wall seconds, the start time)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    with open(log_path, "ab") as log:
        log.write(("$ " + " ".join(cmd) + "\n").encode())
        log.flush()
        try:
            p = subprocess.run(cmd, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\n{_report.PREFIX} TIMEOUT after {timeout}s\n".encode())
            rc = 124
    return rc, time.time() - t0, t0


def prepare(overrides) -> Tuple[List[dict], str]:
    """(cases, out_dir) of a command line: its one target read from the typed overrides (case_from_overrides); out_dir = the directory of the
    typed prefix, where the run record goes beside the outputs. Raises UsageError (no target typed) by name."""
    case = case_from_overrides(overrides)
    return [case], os.path.dirname(case["prefix"])


def run(mode: Optional[str], overrides=None, *, cases: Optional[List[dict]] = None, out_dir: Optional[str] = None, tag: str = "run",
        timeout: Optional[float] = None, python: Optional[str] = None, dry_run: bool = False, det_flag: bool = False
        ) -> Tuple[int, dict]:
    """The design pass. Returns (exit code, the manifest dict). The request is the command line's one target (`overrides`, prepare), or —
    ``warm`` only — the rows `cases` it loaded from the kit's bundled example targets with their scratch `out_dir` (warm.py). Refusals happen
    before anything runs (exit 3 with the named fact; exit 2 for a command line that names no target); after the pass the exit rule is
    report.verdict — a partial activation (a lever without its evidence line, or a forbidden line) exits 3 with the outputs kept; outputs
    short of the request exit 1 whatever else happened; on ``--mode off`` the stock process's own exit code is the pass's."""
    try:
        if cases is None:
            cases, out_dir = prepare(overrides)
        else:
            out_dir = os.path.abspath(out_dir)
    except UsageError as e:
        sys.stderr.write(f"rfdiffusion1-opt design: {e}\n")
        return EXIT_USAGE, {"status": "usage", "reason": str(e)}
    except DesignError as e:
        _report.emit(_report.not_active_line(str(e)))
        return EXIT_NOT_ACTIVE, {"status": "refused", "reason": str(e)}
    m, refused, overrides = request(mode, overrides)                                 # what the kit line cannot serve (the default kit line included): refused by name, exit 3, nothing runs (`--mode off` runs it)
    if refused:
        return refused
    typed_det = upstream_args.parse(overrides or []).value(det.SEED_KEY)
    if det_flag and typed_det is not None and not upstream_args.as_bool(typed_det, True):   # --det 1 asks for the seed, the typed key says otherwise: the typed key wins on both arms — named
        _report.emit(_report.note_line(f"--det 1 overridden by the typed {det.SEED_KEY}={typed_det}: both arms run unseeded"))
    rep = stack.activate(mode, overrides, dry_run=dry_run, det=det_flag)
    if rep.get("reason") or rep.get("would_refuse") or not (rep.get("active") or dry_run):
        return EXIT_NOT_ACTIVE, {"status": "refused", "reason": rep.get("reason") or "; ".join(rep.get("would_refuse") or []), "activation": rep}
    res = resolve(rep["mode"], overrides if overrides is not None else rep.get("overrides"), det=det_flag)   # the mode is the process's; the typed overrides and --det are this launch's
    rfd, weights = rep.get("rfd_root"), rep.get("weights")
    n_designs = sum(c["num_designs"] for c in cases)
    man = _manifest.start(rep, res, cases, out_dir, tag, det=det_flag)
    man["det_effective"] = int(upstream_args.as_bool(typed_det, bool(det_flag))) if typed_det is not None else int(bool(det_flag))
    work = scratch_dir(tag)
    cases_file = os.path.join(work, "cases.json")
    if dry_run:
        man["status"] = "dry-run"
        if res.attach == "driver":
            man["driver_cmd"] = stack.driver_command(res, cases_file, out_dir, tag, rfd, weights, python)
        else:
            man["stock_cmds"] = [stock_command(c, out_dir, rfd, weights, res, python, det_flag) for c in cases]
        return EXIT_OK, man
    write_cases(cases, work, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    _report.register_exit_tally()                                                   # the exit tally belongs to a process that runs a pass: registered here, never on a usage / refusal exit
    _report.emit(_report.run_line(res.mode, len(cases), n_designs, out_dir, "per-case" if res.attach == "stock-cli" else "resident"))
    failed = not_active = None                                                     # the facts the verdict weighs (report.verdict), each named once
    partial, partial_reasons = [], []
    stock_rc = None
    if res.attach == "stock-cli":
        env, stripped, absent = stock_environment()
        man["stock"] = {"env_stripped": stripped, "must_be_absent": absent, "det": bool(det_flag), "passes": []}
        for c in cases:
            case_dir = kit_dir_of(c, out_dir)
            os.makedirs(case_dir, exist_ok=True)
            cmd = stock_command(c, out_dir, rfd, weights, res, python, det_flag)
            rc, wall, t0 = _run(cmd, env, os.getcwd() if c.get("prefix") else case_dir, os.path.join(case_dir, "stock.log"), timeout)
            proof_p = os.path.join(case_dir, "stock_env_proof.json")
            proof = json.load(open(proof_p)) if os.path.isfile(proof_p) else {"ok": False, "error": "no proof written"}
            man["stock"]["passes"].append({"case": c["name"], "rc": rc, "cmd": cmd, "proof": proof, "log": os.path.join(case_dir, "stock.log")})
            if proof.get("ok"):
                stack.confirm_active(f"case {c['name']}: stock environment proof ok")   # the ACTIVE documented line follows the first proof
            if not proof.get("ok"):                                                # the stock process's environment was not proven clean: not the stock line
                not_active = f"case {c['name']}: the stock process's environment proof is not ok ({proof.get('error') or proof.get('findings') or proof})"
                _report.emit(_report.not_active_line(not_active))
                break
            if rc != 0:
                failed = f"case {c['name']}: the stock process exited {rc}"
                stock_rc = rc
                break
    else:
        env, dropped = stack.driver_environment(res, rep["gpu"])
        man["env_dropped"] = dropped
        man["driver_passes"] = []
        cwd = os.getcwd() if any(c.get("prefix") for c in cases) else out_dir       # the hydra form: relative typed paths resolve where upstream would resolve them
        cmd = stack.driver_command(res, cases_file, out_dir, tag, rfd, weights, python)
        log_path = os.path.join(out_dir, f"{tag}.log")
        rc, wall, t0 = _run(cmd, env, cwd, log_path, timeout)
        timings = os.path.join(out_dir, f"{tag}_timings.json")
        ev = read_evidence(log_path, list(res.levers))
        io = io_counters(log_path)
        n_run = designs_computed(timings)
        if not all_skipped(timings):                                                   # a pass whose every requested design already existed (upstream's inference.cautious rule) made no forward call: no per-call counters to hold the levers to
            ev["forbidden"] += io_defects(io, [l for l in res.levers if l in ev["applied"]])   # lever IO1's proof against upstream's writer: a mismatch or an unverified pass is a forbidden line
            ev["forbidden"] += counter_defects(timings, [l for l in res.levers if l in ev["applied"]])   # the kits' own final counters of the levers that printed their applied-line: a lever whose counters name a degraded path is a forbidden line of the pass
        observed = _report.numerics_from_timings(timings)                             # the driver's own torch numerics record against the line's declaration — gated: a TF32 state other than the line's is a defect of the pass
        num = numerics(res.mode, observed, levers=res.levers) if "error" not in observed else None
        if rc == 0 or os.path.isfile(timings):
            ev["forbidden"] += numerics_defects(num)
        _report.emit(_report.evidence_line(ev["applied"], ev["missing"], ev["forbidden"], ev.get("log_error")))
        stack.record_pass(ev)                                                    # the report's levers_applied / levers_unavailable / partial
        _report.add_tally_source(tag, timings, log_path)
        _report.emit(_report.numerics_line(tag, observed, num))
        if "IO1" in res.levers:
            _report.emit(_report.io_line(tag, io))
        man["driver_passes"].append({"designs_computed": ({"computed": n_run[0], "skipped_existing": n_run[1]} if n_run is not None else None), "tag": tag, "cases": [c["name"] for c in cases], "rc": rc, "cmd": cmd, "log": log_path,
                                     "timings_json": timings if os.path.isfile(timings) else None, "evidence": ev, "io": io,
                                     "numerics": num if num is not None else {"source": "unreadable", "error": observed.get("error")}})
        if ev["applied"]:
            stack.confirm_active(f"pass {tag}: applied-lines {','.join(ev['applied'])}")   # the ACTIVE documented line follows the first application evidence
        if rc != 0 and not ev["applied"]:                                       # the driver died before any lever applied (an import or set-up failure): activation failed in the driver — NOT ACTIVE, exit 3, never a plain failure
            not_active = f"pass {tag}: the driver exited {rc} before any lever applied — activation failed in the driver (log {log_path})"
            _report.emit(_report.not_active_line(not_active))
        elif rc != 0:
            failed = f"pass {tag}: the driver process exited {rc}"
        elif ev["missing"] or ev["forbidden"]:                                       # a lever on the stock path: not the mode's line
            partial += [l for l in ev["missing"] if l not in partial]
            partial_reasons.append(f"pass {tag}: levers without their evidence line {ev['missing'] or 'none'}; forbidden lines {len(ev['forbidden'])}")
    man["outputs"] = _outputs.list_outputs(cases, out_dir)
    v = _report.verdict(failed=failed, not_active=not_active, partial=partial, partial_reason="; ".join(partial_reasons) or None,
                        n_pdb=man["outputs"]["n_pdb"], expected=n_designs)
    if stock_rc is not None:
        v["exit_code"] = stock_rc                                                  # the stock command line's own exit code, passed through
    man.update(status=v["status"], partial=v["partial"], partial_reason=v["partial_reason"], incomplete=v["incomplete"], exit_code=v["exit_code"])
    if v["reason"]:
        man["reason"] = v["reason"]
    if v["incomplete"]:
        man["outputs_note"] = f"expected {n_designs} designs, found {man['outputs']['n_pdb']}"
    _report.emit_verdict(v)
    _manifest.finish(man, out_dir)
    return v["exit_code"], man
