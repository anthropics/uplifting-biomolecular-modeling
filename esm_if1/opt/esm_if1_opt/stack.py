"""What the tree carries and what the box has — the facts ``check`` reports and ``design`` stages; nothing here refuses a run.

* Tree: ``tree_home()`` = the ``esm_if1/`` directory (the core's ``home.tree_home``: the package's own location, else ``MODEL_OPT``), which
  carries ``stock/fair-esm-2b369911.tar.gz`` and its extracted ``stock/src/`` — upstream's example script
  ``examples/inverse_folding/sample_sequences.py`` lives there: ``--mode off`` runs it (``stock_script``); ``--mode fast`` runs ``batched.py``.
* Upstream pin: ``UPSTREAM_COMMIT`` / ``FAIR_ESM_VERSION``; the installed ``fair-esm`` and ``torch`` versions come from distribution metadata.
* Weights: upstream's loader reads ``$TORCH_HOME/hub/checkpoints/esm_if1_gvp4_t16_142M_UR50.pt`` (downloading it when absent);
  ``ESM_IF1_WEIGHTS`` optionally names a local copy: ``stage_torch_home`` then makes that cache entry a symlink to it (no copy, no download);
  ``WEIGHTS_BYTES`` / ``WEIGHTS_SHA256`` are the expected values ``check`` prints beside the file's own — reported, never a gate.
* State: ``MODEL_OPT_STATE`` (default ``~/.cache/esm_if1_opt``) holds ``torch_home/`` unless ``TORCH_HOME`` is set.
* GPU: ``gpu()`` probes ``nvidia-smi``; ``target_match`` compares it with ``MODEL_OPT_TARGET_GPU`` through the core's class table — reported.
* Environment: the stock child runs with ``MUST_BE_ABSENT_PREFIXES`` and ``INTERPRETER_ENV`` stripped (``stock_env``) and proves them absent
  itself (``stock_design`` over ``opt_core.stock_proof.env_proof``).
* Activation: ``enable(mode)`` -> the report of ``off`` (inactive: stock) or ``fast`` (active: lever ``batched_sampling``, on both of
  upstream's routes); both need ``fair-esm`` and ``torch`` installed, else ``ActivationError`` (exit 3 at the command line); an unknown
  mode word raises ``ActivationError`` too (the command line answers it earlier with a usage line, exit 2).
"""
from __future__ import annotations

import os
import sys
from typing import Mapping, Optional

from opt_core import gates, home

from . import ActivationError, __version__, modes

UPSTREAM_REPO = "https://github.com/facebookresearch/esm"
UPSTREAM_COMMIT = "2b369911bb5b4b0dda914521b9475cad1656b2ac"
FAIR_ESM_VERSION = "2.0.1"
ARCHIVE = "stock/fair-esm-2b369911.tar.gz"
SCRIPT_REL = os.path.join("stock", "src", "examples", "inverse_folding", "sample_sequences.py")
MODEL_NAME = "esm_if1_gvp4_t16_142M_UR50"
WEIGHTS_FILE = MODEL_NAME + ".pt"
WEIGHTS_BYTES = 1700450121
WEIGHTS_SHA256 = "be4ba36edec22a9bfaa4946ff6b2815f1f19d8a3d7e0eada8b796d5a0eae9fd4"     # the checkpoint's one digest: `check` prints it; nothing gates on it
ENV_WEIGHTS = "ESM_IF1_WEIGHTS"
ENV_STATE = "MODEL_OPT_STATE"
ENV_TARGET_GPU = "MODEL_OPT_TARGET_GPU"
INTERPRETER_ENV = ("PYTHONSAFEPATH",)                                                     # interpreter switches a stock child does not inherit
TIMING_ENV = "ESM_IF1_TIMING_JSONL"                                                      # lines.TIMING_ENV (held equal by the tests)
MUST_BE_ABSENT_PREFIXES = (modes.ENV_MODE, "ESM_IF1_KIT", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")   # stripped from the stock child's environment (stock_env) and proven absent there (opt_core.stock_proof.env_proof): the mode variable, the reserved ESM_IF1_KIT prefix, TF32 / cuBLAS / loader switches
_REPORT: Optional[dict] = None


def tree_home() -> str:
    return home.tree_home(__file__, require=(SCRIPT_REL, "opt"))


def stock_script() -> str:
    """Upstream's example script in the tree (``stock/src/examples/inverse_folding/sample_sequences.py``, byte-identical to the pinned commit):
    what ``--mode off`` runs."""
    return os.path.join(tree_home(), SCRIPT_REL)


def state_dir(environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    return os.path.abspath(env.get(ENV_STATE) or os.path.join(os.path.expanduser("~"), ".cache", "esm_if1_opt"))


def torch_home(environ: Optional[Mapping[str, str]] = None) -> str:
    """torch.hub's cache root for this pass, absolute: ``$TORCH_HOME`` when set, else ``<state dir>/torch_home`` (a stock child receives it as TORCH_HOME)."""
    env = os.environ if environ is None else environ
    return os.path.abspath(env.get("TORCH_HOME") or os.path.join(state_dir(env), "torch_home"))


def hub_entry(th: Optional[str] = None) -> str:
    return os.path.join(th or torch_home(), "hub", "checkpoints", WEIGHTS_FILE)


def weights_env() -> Optional[str]:
    v = os.environ.get(ENV_WEIGHTS)
    return os.path.abspath(v) if v else None


def stage_torch_home() -> dict:
    """Make upstream's loader find the checkpoint without a download when ``ESM_IF1_WEIGHTS`` names a local file: the torch.hub cache entry
    becomes a symlink to it (an entry that is already a regular file is kept). Returns ``{file, torch_home, hub_entry: linked|present|kept|
    absent, source}`` — the WEIGHTS line's fields. Never raises on the checkpoint's content and never reads it."""
    th = torch_home()
    entry = hub_entry(th)
    src = weights_env()
    rec = {"file": WEIGHTS_FILE, "torch_home": th, "source": (f"{ENV_WEIGHTS}={src}" if src else "torch.hub")}
    if not src or not os.path.isfile(src):
        rec["hub_entry"] = "present" if os.path.isfile(entry) else "absent"
        if src:
            rec["source"] += "(not a file)"
        return rec
    os.makedirs(os.path.dirname(entry), exist_ok=True)
    if os.path.islink(entry):
        if os.path.realpath(entry) == os.path.realpath(src):
            rec["hub_entry"] = "linked"
            return rec
        os.unlink(entry)
    elif os.path.exists(entry):
        rec["hub_entry"] = "kept" if not os.path.samefile(entry, src) else "linked"
        return rec
    os.symlink(src, entry)
    rec["hub_entry"] = "linked"
    return rec


def cache_word() -> str:
    """After a stock child ran: ``hit`` when the torch.hub cache entry still resolves to the ``ESM_IF1_WEIGHTS`` file, ``miss`` when it does
    not (something replaced it — a download), ``na`` when no ``ESM_IF1_WEIGHTS`` file is given."""
    src = weights_env()
    if not src or not os.path.isfile(src):
        return "na"
    entry = hub_entry()
    return "hit" if os.path.exists(entry) and os.path.realpath(entry) == os.path.realpath(src) else "miss"


def gpu() -> dict:
    """``{name, cc, memory_mib, probe}`` of GPU 0 (nvidia-smi; no torch import)."""
    return gates.nvidia_smi_probe(keys=("name", "cc", "memory_mib", "probe"))


def target_match(g: Optional[Mapping] = None) -> dict:
    """``{target, match: bool|None, reason}`` of ``MODEL_OPT_TARGET_GPU`` against the probed GPU — reported, never enforced."""
    target = os.environ.get(ENV_TARGET_GPU) or None
    if not target:
        return {"target": None, "match": None, "reason": None}
    gate = gates.gpu_class_check(g if g is not None else gpu(), target)
    return {"target": target, "match": bool(gate.ok), "reason": gate.reason}


def stack_key() -> str:
    """A short label of the installed torch build from distribution metadata (no import), e.g. ``torch2.4.0-cu121``; ``unknown`` when absent."""
    v = gates.dist_version("torch")
    if not v:
        return "unknown"
    base, _, local = v.partition("+")
    return f"torch{base}-{local or 'cpu'}"


def esm_version() -> Optional[str]:
    """The installed ``fair-esm`` distribution's version (metadata; no import), or None."""
    return gates.dist_version("fair-esm")


def facts() -> dict:
    """What the tree carries and what the box has, for ``check`` and the run record (no hashing here)."""
    w = weights_env()
    entry = hub_entry()
    return {"upstream": {"repo": UPSTREAM_REPO, "commit": UPSTREAM_COMMIT, "distribution": "fair-esm", "version": FAIR_ESM_VERSION,
                         "archive": ARCHIVE, "installed": esm_version()},
            "script": SCRIPT_REL, "torch": gates.dist_version("torch"), "stack_key": stack_key(),
            "weights": {"file": WEIGHTS_FILE, "env": w, "env_is_file": bool(w and os.path.isfile(w)), "hub_entry": entry,
                        "hub_entry_present": os.path.isfile(entry), "torch_home": torch_home()}}


def enable(mode: Optional[str] = None) -> dict:
    """The activation report of ``mode`` for this process: ``off`` -> the stock route (inactive by definition); ``fast`` -> active, its
    lever ``batched_sampling`` applied on both of upstream's routes (single-chain and ``--multichain-backbone``) — the mode has no gate and
    no partial state. Both need ``fair-esm`` and ``torch`` installed for the child to run (``ActivationError`` otherwise); an unknown
    mode word -> ``ActivationError`` (the table's usage sentence)."""
    global _REPORT
    try:
        m = modes.resolve(mode)
    except modes.ModeError as e:
        raise ActivationError(str(e)) from None
    if _REPORT is not None and _REPORT.get("mode") != m:
        raise ActivationError(f"mode {_REPORT['mode']} was resolved in this process; a second mode ({m}) is refused")
    missing = [d for d in ("fair-esm", "torch") if not gates.dist_version(d)]
    if missing:
        raise ActivationError(f"mode {m} cannot run: {', '.join(missing)} not installed on {sys.executable} (run.sh install; STOCK.md)")
    levers = list(modes.levers(m))
    rep = {"mode": m, "route": modes.route_of(m), "active": bool(levers), "package_version": __version__, "levers": levers,
           "levers_applied": list(levers), "partial": [], "reason": f"mode {m}: {modes.WORDS[m]}"}
    _REPORT = rep
    return dict(rep)


def kit_env(out_dir: str, base: Optional[Mapping[str, str]] = None) -> dict:
    """The kit child's environment (``--mode fast``): this process's, plus ``TORCH_HOME`` (absolute), ``ESM_IF1_TIMING_JSONL=<out>/timing.jsonl``
    and ``PYTHONDONTWRITEBYTECODE=1``. Nothing is stripped: the kit child is the kit."""
    env = dict(os.environ if base is None else base)
    env["TORCH_HOME"] = torch_home(env)
    env[TIMING_ENV] = os.path.join(os.path.abspath(out_dir), "timing.jsonl")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def stock_env(out_dir: str, base: Optional[Mapping[str, str]] = None) -> tuple:
    """The stock child's environment: ``base`` (this process's) without the names under ``MUST_BE_ABSENT_PREFIXES`` and ``INTERPRETER_ENV``
    (``opt_core.stock_proof.strip_env``), plus ``TORCH_HOME`` (absolute), ``ESM_IF1_TIMING_JSONL=<out>/timing.jsonl`` and
    ``PYTHONDONTWRITEBYTECODE=1``. PYTHONPATH is inherited unchanged: the child is a module of this package and this engine carries no lever
    directory to remove. Returns ``(env, removed_names)``."""
    from opt_core import stock_proof
    env, names = stock_proof.strip_env(os.environ if base is None else base, MUST_BE_ABSENT_PREFIXES, INTERPRETER_ENV)
    env["TORCH_HOME"] = torch_home(env)
    env[TIMING_ENV] = os.path.join(os.path.abspath(out_dir), "timing.jsonl")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env, sorted(names)


def status() -> dict:
    return dict(_REPORT) if _REPORT is not None else {"active": False, "reason": "no mode resolved in this process"}
