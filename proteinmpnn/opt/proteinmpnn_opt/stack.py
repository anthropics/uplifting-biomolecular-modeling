"""Activation: the tree's paths, the carried kit bytes, the stock pin, the GPU, and the resolved mode armed for this process.

``activate(mode, variant)`` resolves the mode by name to the kit's README row (modes.resolve), records the core pin gate's facts (``core_gate``:
statement one of every entry, so a mismatched core has refused before this point; ``opt_core`` in the report), gates — the kit's own required
files are present under the kit directories (``check_kits``; the tree's git commit names the bytes, nothing is re-hashed here), the stock checkout
and the weights are present (``stock/check_pins.py`` in a subprocess: absent things refuse, and so does a checkout whose HEAD is another commit than
the pin — it changes what stock means; a checkout whose HEAD cannot be read (no .git) runs and
the report says so under ``pins.notes``; the weights file the pass loads — the
variant's directory and the pass's ``--model_name`` — is digested there and named, never refused unless absent: ``weights`` in the report
{file, sha256, pinned, verdict pinned|not_pinned, line} = ``pins.detail.weights``, its line printed after the activation line by
``report.log_activation``), a GPU
is identified (``nvidia-smi``, no torch import) and, where the configuration names a target class (``MODEL_OPT_TARGET_GPU`` with
``MODEL_OPT_TARGET_GPU_MEM_MIB``), compared by card name and memory total (``gpu_matches_target``: reported on the line and in the report,
not refused) — and returns the activation report. The kit is a driver, so nothing is applied at activation: the report is the plan
(``applied: armed``) and the launch (``design``) is where the kit's own executable runs on it; the worker's own lines (``DEVICE CELL``, the
probe verdict) are read from that run. ``activate(..., dry_run=True)`` is ``check``: the same resolution and gates, nothing launched,
``DRY-RUN`` line.

Late activation: ``enable()`` after this process has launched a kit worker (``mark_launched``) is refused by name; a second, different
(mode, variant) in one process is refused by name; the same pair is idempotent.

A mode is all of its levers: it engages every lever of its line or refuses by name, never runs under its name with a subset. Without a
CUDA device the worker's CUDA-graph levers cannot run (its own CPU rule, modes.cpu_disabled_levers): the report names them in ``partial`` (and
under ``levers_unavailable``) and the activation REFUSES — ``design`` / ``warm``: ActivationError, exit 3, nothing launched; ``check``: the reason
on the DRY-RUN line, exit 3 — unless ``--allow-partial`` (or ``PROTEINMPNN_OPT_ALLOW_PARTIAL=1``; ``allow_partial``) is given: the one recorded
override, under which the levers that can run do and the lines say ``allow_partial=yes``. After a launch the worker's own end-of-run record is
read against the request (kit_run.lever_evidence): a requested lever the record shows off lands in ``levers_fallback`` and ``partial`` (exit 3
unless ``--allow-partial``); ``enable()`` records it and cannot set the host process's exit — the command line is the gated form. What is merely
untested (another GPU model or arch, another torch or driver) is named on the lines and never a refusal. The exact line requests the worker's
probe-gated ``--hybrid_gemm``: at activation ``probe.verdict`` is ``measured at launch`` (``probe.requested`` true) and ``probe.kit_observed``
carries the kit README's own observation for this GPU's compute capability; after the launch ``design`` folds the worker's own verdict word into
``probe.verdict`` (``PASS`` | ``FAIL`` | ``unobserved``, kit_run.probe_word; the whole cell under ``probe.worker``) and prints it as ``probe=`` on
the EXIT line. On a FAIL the worker refuses the job by name before any output (its ``hybrid_gemm: REFUSED`` line, ``refused`` in its record,
rc 3): the lever cannot engage bit-identically on this device and the line is all of its levers; ``design`` exits 3 naming ``--hybrid_gemm 0``,
the line without the lever by name (``--allow-partial`` does not apply: nothing partial ran).
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from opt_core import gates as core_gates, home as core_home

from . import ActivationError, __version__, core_gate
from . import modes

ENV_MODE = "PROTEINMPNN_OPT"
ENV_VARIANT = "PROTEINMPNN_VARIANT"
ENV_HOME = "PROTEINMPNN_OPT_HOME"
ENV_MPNN_DIR = "MPNN_DIR"
ENV_TARGET_GPU = "MODEL_OPT_TARGET_GPU"
ENV_TARGET_GPU_MEM = "MODEL_OPT_TARGET_GPU_MEM_MIB"                         # the target class's memory total (nvidia-smi MiB); the match asserts it
ENV_ALLOW_PARTIAL = "PROTEINMPNN_OPT_ALLOW_PARTIAL"                        # the environment spelling of --allow-partial (=1): a partial activation is recorded and the pass proceeds
PACKAGE_ENV: Tuple[str, ...] = (ENV_MODE, ENV_VARIANT, ENV_HOME, ENV_ALLOW_PARTIAL)   # read by this package only; stripped from every subprocess
DATA_ENV: Tuple[str, ...] = (ENV_MPNN_DIR,)                                # deployment parameters (configs/<gpu>.env)
INTERPRETER_ENV: Tuple[str, ...] = ("PYTHONSAFEPATH",)                       # would stop the upstream scripts importing their siblings (script-directory imports)

PINS = os.path.join("stock", "PINS.json")
CHECK_PINS = os.path.join("stock", "check_pins.py")

_STATE: Dict[str, object] = {"report": None, "launched": 0}


# ----------------------------------------------------------------------------------------------------------------- paths
def tree_home() -> str:
    """The proteinmpnn/ directory: $PROTEINMPNN_OPT_HOME, else $MODEL_OPT, else the parent of the package's opt/ (editable install; opt_core.home)."""
    return core_home.tree_home(__file__, env_home=ENV_HOME)


def kit_home() -> str:
    return os.path.join(tree_home(), "opt", "forward")


def worker_dir() -> str:
    return os.path.join(kit_home(), modes.WORKER_DIR)


def parser_dir() -> str:
    return os.path.join(kit_home(), modes.PARSER_DIR)


def stock_dir() -> str:
    return os.path.join(tree_home(), "stock")


def mpnn_dir() -> Optional[str]:
    return os.environ.get(ENV_MPNN_DIR) or None


def pins() -> dict:
    with open(os.path.join(tree_home(), PINS), encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------------------------------------------- the carried bytes
sha256 = core_gates.sha256_file             # the one file-digest rule (opt_core.gates): copy-fidelity checks (an overlay landed intact), nothing more

# The kit's own required files, by kit directory — a presence gate at activation, before staging touches them (stage.py's own copy raises on
# a missing file too; this names the gap earlier, with the kit directory it's under). The tree's git commit names the bytes; nothing is re-hashed here.
KIT_REQUIRED_FILES: Dict[str, Tuple[str, ...]] = {
    modes.PARSER_DIR: ("kit/fast_parse.py",),
    modes.WORKER_DIR: ("addon/mpnn_worker2.py",),
}


def check_kits() -> Tuple[int, List[str]]:
    """The kit's own required files (KIT_REQUIRED_FILES) are present under the two opt/forward/ directories."""
    bad: List[str] = []
    n_ok = 0
    for sub, rels in KIT_REQUIRED_FILES.items():
        for rel in rels:
            if os.path.exists(os.path.join(kit_home(), sub, rel)):
                n_ok += 1
            else:
                bad.append(f"{sub}/{rel}: missing")
    return n_ok, bad


# ----------------------------------------------------------------------------------------------------------------- the stock pin
def check_pins(variant: str, model_name: Optional[str] = None, weights_file: Optional[str] = None) -> dict:
    """stock/check_pins.py --variant V [--model-name N] [--weights-file F] --json in a subprocess (python -I, standard library): {pinned, findings,
    notes, detail} — ``findings`` are what a pass cannot run without (the checkout, the weights file: refused by name), ``notes`` are facts reported and never refused (the checkout's HEAD is not the pinned commit, the checkout is not a git
    checkout); ``detail["weights"]`` is the record of the weights file the pass loads — ``weights_file`` when the pass's own selectors name one
    (settings.weights_path), else the variant's default (its digest, its PINS.json name or None, the verdict, the line)."""
    cmd = [sys.executable, "-I", os.path.join(tree_home(), CHECK_PINS), "--variant", variant, "--json", "--quiet"]
    if model_name:
        cmd += ["--model-name", str(model_name)]
    if weights_file:
        cmd += ["--weights-file", str(weights_file)]
    env = {k: v for k, v in os.environ.items() if k not in PACKAGE_ENV}
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        return {"pinned": False, "findings": [f"check_pins.py could not run: {e!r}"], "detail": {}}
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"pinned": False, "findings": [f"check_pins.py rc {out.returncode}: {out.stderr.strip()[-400:]}"], "detail": {}}


def commit8(detail: dict) -> Optional[str]:
    c = detail.get("head")
    return c[:8] if c else "unknown"                  # a checkout without .git: the .fa headers say git_hash=unknown too


# ----------------------------------------------------------------------------------------------------------------- the GPU
def gpu_info() -> Optional[dict]:
    """The first visible GPU by nvidia-smi: {name, mem_mib, cc, sm, count}; None without a driver."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    rows = [r.strip() for r in out.stdout.strip().splitlines() if r.strip()]
    name, mem, cc = [x.strip() for x in rows[0].split(",")][:3]
    sm = "sm_" + cc.replace(".", "") if cc else None
    try:
        mem_mib = int(float(mem))
    except ValueError:
        mem_mib = None
    return {"name": name, "mem_mib": mem_mib, "cc": cc or None, "sm": sm, "count": len(rows)}


def gpu_matches_target(gpu: Optional[dict], target: Optional[str], target_mem_mib: Optional[int] = None, tolerance: float = 0.01) -> Optional[bool]:
    """The box's GPU against the configuration's target: the card name (substring) AND, when the configuration states the class's memory
    total (``MODEL_OPT_TARGET_GPU_MEM_MIB``), the nvidia-smi total within ``tolerance`` of it — a card whose name contains the target but
    carries another memory size (an NVL or a 40 GB part) is not the target. None when no target or no GPU."""
    if not target or not gpu:
        return None
    name_ok = target.replace("-", " ").lower() in (gpu.get("name") or "").replace("-", " ").lower()
    if not name_ok:
        return False
    if target_mem_mib:
        mem = gpu.get("mem_mib")
        if mem is None:
            return False
        return abs(int(mem) - int(target_mem_mib)) <= tolerance * int(target_mem_mib)
    return True


def target_gpu_mem_mib() -> Optional[int]:
    v = os.environ.get(ENV_TARGET_GPU_MEM)
    try:
        return int(v) if v else None
    except ValueError:
        return None


def stack_key(gpu: Optional[dict] = None) -> str:
    """torch<version sans local tag>-cu<CUDA sans dot>[-sm<cc digits>] from package metadata (no torch import); raises
    importlib.metadata.PackageNotFoundError when torch's package metadata is absent (never a silent 'unknown': a wrong-but-plausible key
    read from a caller that keys a cache directory or a report field on it is a worse failure mode than a loud one here)."""
    ver = md.version("torch")
    base, _, local = ver.partition("+")
    key = f"torch{base}" + (f"-{local}" if local else "")
    if gpu and gpu.get("cc"):
        key += "-sm" + gpu["cc"].replace(".", "")
    return key


# ----------------------------------------------------------------------------------------------------------------- activation
def _refuse_late(what: str) -> None:
    if int(_STATE["launched"]) > 0:
        raise ActivationError(f"late activation refused: {what} after this process launched a kit worker ({_STATE['launched']} launch(es)); "
                              "one mode per process — start a new process for another mode")


def allow_partial_env() -> bool:
    """The environment spelling of ``--allow-partial`` (``PROTEINMPNN_OPT_ALLOW_PARTIAL=1``), read here only: the command line ORs its
    flag with it (cli._allow_partial), the activation report records it."""
    return os.environ.get(ENV_ALLOW_PARTIAL, "") == "1"


def activate(mode: Optional[str], variant: Optional[str], dry_run: bool = False, model_name: Optional[str] = None, opt_out: Optional[List[str]] = None,
             weights_file: Optional[str] = None, bb_batch: Optional[int] = None, allow_partial: Optional[bool] = None) -> dict:
    """Resolve + gate; the activation report. Raises ActivationError (named) when a gate refuses a non-dry run; a dry run reports the
    refusal in ``reason`` with ``active`` False. ``model_name`` (base variants; the pass's --model_name, else upstream's default) selects the weights file the stock pin
    check digests and names — the file the pass loads. ``opt_out``: probe-gated levers the command line leaves out by name (``--hybrid_gemm 0``);
    ``bb_batch``: the command line's backbones per worker batch (modes.resolve; None = the mode line's own)."""
    try:
        res = modes.resolve(mode, variant, kit_home(), opt_out=opt_out, bb_batch=bb_batch)
    except modes.ModeError as e:
        if dry_run:
            return {"active": False, "dry_run": True, "mode": mode, "variant": variant, "reason": str(e), "package_version": __version__}
        raise ActivationError(str(e)) from e
    rep: dict = {"active": False, "dry_run": dry_run, "mode": res.mode, "variant": res.variant, "route": res.route, "line": modes.describe_line(res),
                 "executable": res.executable, "flags": list(res.flags), "levers_applied": list(res.levers), "levers_fallback": [], "levers_unavailable": [],
                 "partial": [], "allow_partial": allow_partial_env() if allow_partial is None else bool(allow_partial), "opted_out": list(res.opted_out), "bb_batch": res.bb_batch,
                 "package_version": __version__, "weights": None, "applied": None, "reason": None, "tree_home": tree_home()}
    reasons: List[str] = []
    core = core_gate()                                      # the core pin gate (statement one of every entry, the stock route included; the same producer fills the report — a mismatch has refused with its line and exit 3 before this point)
    rep["opt_core"] = dict(core["installed"], ok=True, pinned=core["pinned"])
    # the carried bytes (kit routes)
    if res.route != "stock":
        n_ok, bad = check_kits()
        rep["kits"] = {"n_files": n_ok + len(bad), "present": not bad, "bad": bad[:10]}
        if bad:
            reasons.append(f"{len(bad)} carried kit file(s) missing: {bad[0]}")
    # the stock pin (every route: off runs the pinned checkout, exact reproduces it); the weights file of the pass is named there, not gated
    pc = check_pins(res.variant, model_name, weights_file)
    rep["pins"] = pc
    if not pc.get("pinned"):
        reasons.append("stock pin: " + "; ".join(pc.get("findings") or ["unknown"]))
    detail = pc.get("detail") or {}
    rep["weights"] = detail.get("weights")
    rep["proteinmpnn_version"] = commit8(detail)
    rep["mpnn_dir"] = mpnn_dir()
    # the GPU
    gpu = gpu_info()
    rep["gpu"] = gpu
    rep["target_gpu"] = os.environ.get(ENV_TARGET_GPU)
    rep["target_gpu_mem_mib"] = target_gpu_mem_mib()
    rep["gpu_matches_target"] = gpu_matches_target(gpu, rep["target_gpu"], rep["target_gpu_mem_mib"])
    rep["stack_key"] = stack_key(gpu)
    if res.route == "worker":
        cpu_off = [f.lstrip("-") for f in modes.cpu_disabled_levers(kit_home())]
        if gpu is None:
            rep["levers_unavailable"] = [l for l in res.levers if l in cpu_off]
            rep["partial"] = list(rep["levers_unavailable"])
            if rep["partial"] and not rep["allow_partial"]:      # a mode is all of its levers: these cannot run here — refuse by name, never run a subset under the mode's name
                reasons.append(f"mode {res.mode} is all of its levers and {','.join(rep['partial'])} cannot run on this box (no CUDA device: the worker's "
                               f"CUDA-graph levers) — --mode off runs the stock command line here; --allow-partial (or {ENV_ALLOW_PARTIAL}=1) runs the levers that can, recorded")
        rep["probe"] = {"requested": res.hybrid_gemm, "verdict": "not requested" if not res.hybrid_gemm else "measured at launch",
                        "kit_observed": modes.probe_verdict_kit_observed(kit_home(), gpu.get("sm") if gpu else None)}
    else:
        rep["probe"] = {"requested": False, "verdict": "not applicable", "kit_observed": None}
    if reasons:
        rep["reason"] = "; ".join(reasons)
        if not dry_run:
            raise ActivationError(rep["reason"])
        return rep
    rep["active"] = not dry_run
    rep["applied"] = "armed" if not dry_run else None
    return rep


def enable(mode: Optional[str] = None, variant: Optional[str] = None, model_name: Optional[str] = None, opt_out: Optional[List[str]] = None,
           weights_file: Optional[str] = None, bb_batch: Optional[int] = None, allow_partial: Optional[bool] = None) -> dict:
    mode = modes.check_mode(mode if mode is not None else os.environ.get(ENV_MODE))
    variant = modes.check_variant(variant if variant is not None else os.environ.get(ENV_VARIANT))
    _refuse_late(f"enable({mode!r}, {variant!r})")
    prev = _STATE["report"]
    if isinstance(prev, dict) and prev.get("active"):
        if (prev["mode"], prev["variant"]) == (mode, variant) and (bb_batch is None or prev.get("bb_batch") == bb_batch):
            return prev
        raise ActivationError(f"mode {prev['mode']} (variant {prev['variant']}) is already active in this process; enable({mode!r}, {variant!r}) "
                              "refused — one mode per process")
    rep = activate(mode, variant, dry_run=False, model_name=model_name, opt_out=opt_out, weights_file=weights_file, bb_batch=bb_batch, allow_partial=allow_partial)
    _STATE["report"] = rep
    return rep


def status() -> dict:
    rep = _STATE["report"]
    if isinstance(rep, dict):
        return rep
    return {"active": False, "reason": "not activated: enable(mode, variant) has not been called in this process"}


def mark_launched(n: int = 1) -> None:
    _STATE["launched"] = int(_STATE["launched"]) + n
    rep = _STATE["report"]
    if isinstance(rep, dict):
        rep["applied"] = "launched"


def reset_for_tests() -> None:
    _STATE["report"] = None
    _STATE["launched"] = 0


# ----------------------------------------------------------------------------------------------------------------- subprocess env
def kit_env(base: Optional[dict] = None) -> dict:
    """The environment for a kit executable: the caller's, the package's own variables and PYTHONSAFEPATH removed, MPNN_DIR set, no .pyc in the tree."""
    env = {k: v for k, v in (base if base is not None else os.environ).items() if k not in PACKAGE_ENV and k not in INTERPRETER_ENV}
    if mpnn_dir():
        env[ENV_MPNN_DIR] = mpnn_dir()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def resolve_mode_and_variant(cli_mode: Optional[str], cli_variant: Optional[str]) -> Tuple[str, str]:
    """The mode: --mode, else $PROTEINMPNN_OPT; neither is a usage error naming off|exact (modes.check_mode: this engine has no default mode); a
    --mode that disagrees with a set $PROTEINMPNN_OPT is refused. The variant: --variant, else $PROTEINMPNN_VARIANT, else modes.DEFAULT_VARIANT."""
    env_mode, env_var = os.environ.get(ENV_MODE), os.environ.get(ENV_VARIANT)
    if cli_mode and env_mode and cli_mode != env_mode:
        raise ActivationError(f"--mode {cli_mode} disagrees with {ENV_MODE}={env_mode} in the environment: unset one")
    if cli_variant and env_var and cli_variant != env_var:
        raise ActivationError(f"--variant {cli_variant} disagrees with {ENV_VARIANT}={env_var} in the environment: unset one")
    variant = modes.check_variant(cli_variant or env_var)
    return modes.check_mode(cli_mode or env_mode), variant
