"""Activation: kit paths, pins, the GPU, the gates, and the lever application through the kit's own statements.

Chai-1 has no model class and no import-time activation point in the kit: ``kit/chai_worker.py`` is a script that installs its levers
at import of ITS OWN process (three module attributes, W1 / W5) and folds through its own loop (W2). The package therefore has two routes:

* ``pred --mode exact|fast|big`` — the kit driver runs as shipped in a subprocess (driver.py): the driver's levers are the driver's own;
  for a row with an eager line (``fast``) the package installs it on top (``apply_eager``: the eager stack's ``Components`` /
  ``build_parts`` / ``make_loader`` with the DSTEP add-on's ``build_lever_parts``; the stack's own ``install`` for a row without DSTEP
  levers) in the driver's ``env_report`` window — after the driver's W1/W5 block and its recipe, before the first fold. This is the
  route every kit mode runs.
* ``chai1_opt.enable(mode)`` / ``CHAI1_OPT=<mode>`` — a program that calls ``chai_lab.chai1.run_inference`` itself. ``activate()`` resolves
  and gates the mode, then executes the kit's OWN lever statements (the driver's top-level W1/W5 block, selected by AST from the kit
  file, compiled with its own line numbers, never transcribed: ``kit_lever_statements``) in a namespace holding ``levels`` = the mode's
  in-process driver levers (modes.KitMode.in_process), applies the recipe, then (``fast``) installs the eager line the same way the
  driver route does. W2 lives in the driver loop and is reported ``not applicable``. ``classify`` reads what the kits' own state shows
  applied (the three attributes hold the kit's functions; under ``fast`` ``chai1.load_exported`` holds the eager stack's loader, which
  wraps the kit's, and the handle's denoiser carries the add-on's policies), never a value the package asserts. ``det=1`` applies the
  deterministic recipe in this process (det.apply: the kit's four statements) at the driver's own placement — after the lever block,
  before the first CUDA call (the driver's ``DET_MODE = cp.apply_deterministic_mode()`` statement) and before the eager stack reads it (``install`` / the package's DSTEP path graph
  the denoiser step only when deterministic algorithms are off); the .pth route reads the level from the kit's own switch
  ``CHAI_DETERMINISTIC``.

A row's implied levers (modes.OPTIN_LEVERS: ``tf32``, ``alloc``; modes.KitMode.implied_optin) ride on either route: gated by
``modes.optin_refusal`` (unknown, or not admitted by the mode → refused by name), applied by precision.py after the recipe and before the
eager stack installs, probed from torch's own flag by ``classify``; a requested opt-in that does not show applied is a refusal like any
lever of the mode.

Every route notes ``MODEL_OPT_TARGET_GPU`` (configs/<gpu>.env) against the visible GPU's name: a mismatch is a note in the report and on
the ACTIVE / DRY-RUN line (``notes=...``), never a refusal.

Late activation: allowed any time after ``chai_lab.chai1`` is imported and before the first fold; refused (ActivationError) once the
traced ESM is loaded (``esm._esm_model`` non-empty — the first thing a fold loads) or ``chai1.load_exported`` is no longer upstream's own
function (the kit or another shim already patched it). ``activate(..., dry_run=True)`` resolves, gates and reports without importing
torch or touching anything (``check``).
"""
from __future__ import annotations

import ast
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import platform
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from . import ActivationError, __version__
from . import det as _det
from . import _core
from . import hoist, modes, registry
from . import alloc as _alloc
from . import precision as _precision
from . import pairtrack as _pairtrack
from . import report as _report

ENV = modes.ENV
ENV_HOME, ENV_KIT, ENV_EAGER, ENV_DSTEP, ENV_ERRATA02 = "CHAI1_OPT_HOME", "CHAI1_OPT_KIT", "CHAI1_OPT_EAGER", "CHAI1_OPT_DSTEP", "CHAI1_OPT_ERRATA02"
ENV_ESM_MEMO_SCOPE = "CHAI1_OPT_ESM_MEMO_SCOPE"                # the W5 ESM-embedding memo's reach: global (every input of the process) | input (one input)
ESM_MEMO_SCOPES = ("global", "input")
ENV_STRICT_STACK = "CHAI1_OPT_STRICT_STACK"                    # =1: a torch/CUDA stack other than the pinned one is refused by name (gates); unset/0: recorded (stack_line) and the run proceeds — stock/check_pins.py ENV_STRICT_STACK restated (that file runs alone under python -I; the tests lock the pair)
ENV_ALLOW_PARTIAL = "CHAI1_OPT_ALLOW_PARTIAL"                  # =1: the opt-out of the partial-activation refusal on the CHAI1_OPT (in-process) route — that route only; pred / the driver take --allow-partial and never read it
ENV_AOTI_HOST_ISA = "CHAI1_OPT_AOTI_HOST_ISA"                  # the host CPU's ISA word declared by hand for the compiled step's ahead-of-time packages (chai1_fastln.cpu_isa: overrides the probe; e.g. AVX2 refuses AVX-512 launchers by name)
DECLARED_ENV = (ENV, ENV_HOME, ENV_KIT, ENV_EAGER, ENV_DSTEP, ENV_ERRATA02, ENV_ESM_MEMO_SCOPE, ENV_STRICT_STACK, ENV_ALLOW_PARTIAL, ENV_AOTI_HOST_ISA)     # every CHAI1_OPT* name this package reads (the .pth hook restates it)
DECLARED_BIG_ENV = ("CHAI1_BIG_MSA_CHUNK_MIN_N", "CHAI1_BIG_NOGRAPH_MIN_N", "CHAI1_BIG_TRUNK_CHUNK_MIN_N", "CHAI1_BIG_OPM_CHUNK_MIN_N", "CHAI1_BIG_HOIST2_MAX_N")   # every CHAI1_BIG_* name this package reads: the two size gates (big.GATE_WORDS; the .pth hook restates it)
ENV_LEVERS_OFF = modes.ENV_LEVERS_OFF                            # the ablation word (one across the model-opt kits), resolved once in modes.py (apply_levers_off / levers_off_refusal)
UPSTREAM_DIST = "chai_lab"
TRIGGER_MODULE = "chai_lab.chai1"
ESM_MODULE = "chai_lab.data.dataset.embeddings.esm"
T0 = time.time()

_STATE: Dict[str, object] = {"report": None, "namespace": None, "eager": None}


# ------------------------------------------------------------------------------------------------------------------ paths
def tree_home() -> str:
    """The chai1/ directory: CHAI1_OPT_HOME, else MODEL_OPT (configs/*.env), else derived from this package's location."""
    for k in (ENV_HOME, "MODEL_OPT"):
        v = os.environ.get(k)
        if v:
            return os.path.abspath(v)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def kit_home() -> str:
    """The carried kit directory (opt/forward/fast_inference), or CHAI1_OPT_KIT."""
    v = os.environ.get(ENV_KIT)
    return os.path.abspath(v) if v else os.path.join(tree_home(), "opt", "forward", "fast_inference")


def eager_home() -> str:
    """The carried eager stack directory (opt/forward/eager_trunk), or CHAI1_OPT_EAGER."""
    v = os.environ.get(ENV_EAGER)
    return os.path.abspath(v) if v else os.path.join(tree_home(), "opt", "forward", "eager_trunk")


def dstep_home() -> str:
    """The carried DSTEP add-on directory (opt/forward/dstep_megakernel), or CHAI1_OPT_DSTEP."""
    v = os.environ.get(ENV_DSTEP)
    return os.path.abspath(v) if v else os.path.join(tree_home(), "opt", "forward", "dstep_megakernel")


def errata02_home() -> str:
    """The carried errata_02 directory (opt/forward/errata_02: the amended worker), or CHAI1_OPT_ERRATA02."""
    v = os.environ.get(ENV_ERRATA02)
    return os.path.abspath(v) if v else os.path.join(tree_home(), "opt", "forward", "errata_02")


def kit_pkg_dir() -> str:
    """The kit's module directory (kit/: chai_proto) — first on sys.path for the worker the tree runs."""
    return os.path.join(kit_home(), "kit")


def driver_path() -> str:
    """The worker the tree runs (modes.WORKER_RELPATH under the errata directory)."""
    return os.path.join(errata02_home(), modes.WORKER_RELPATH)


def eager_stack_path() -> str:
    return os.path.join(eager_home(), modes.EAGER_STACK_RELPATH)


def dstep_stackx_path() -> str:
    return os.path.join(dstep_home(), modes.DSTEP_STACKX_RELPATH)


def kit_present() -> Tuple[bool, str]:
    """The files the routes run are on disk: the kit's module (chai_proto), the worker the tree runs, the eager stack's and
    the DSTEP add-on's entry points. The first missing one is the refusal."""
    p = os.path.join(kit_home(), modes.PROTO_RELPATH)
    if not os.path.isfile(p):
        return False, f"kit file missing: {p} (set {ENV_KIT} or {ENV_HOME}; the kit is opt/forward/fast_inference of the tree)"
    p = os.path.join(errata02_home(), modes.WORKER_RELPATH)
    if not os.path.isfile(p):
        return False, f"errata file missing: {p} (set {ENV_ERRATA02} or {ENV_HOME}; the errata is opt/forward/errata_02 of the tree)"
    p = os.path.join(eager_home(), modes.EAGER_STACK_RELPATH)
    if not os.path.isfile(p):
        return False, f"eager stack file missing: {p} (set {ENV_EAGER} or {ENV_HOME}; the eager stack is opt/forward/eager_trunk of the tree)"
    p = os.path.join(dstep_home(), modes.DSTEP_STACKX_RELPATH)
    if not os.path.isfile(p):
        return False, f"DSTEP add-on file missing: {p} (set {ENV_DSTEP} or {ENV_HOME}; the add-on is opt/forward/dstep_megakernel of the tree)"
    return True, kit_home()


def pins() -> dict:
    with open(os.path.join(tree_home(), "stock", "PINS.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _check_pins_module():
    p = os.path.join(tree_home(), "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("chai1_stock_check_pins", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pins_check() -> Tuple[List[str], dict]:
    """The stock pin check (stock/check_pins.py): ``(bad, detail)``; bad = one line per refusal (empty = pinned). The torch / CUDA stack
    is never a refusal here (``strict=False``): the strict switch is applied by :func:`gates`, which names it."""
    return _check_pins_module().check(strict=False)


def stack_pinning(environ=None) -> dict:
    """The torch / CUDA stack record every route carries (base_report -> the activation report, the NOTE line), computed by
    the one function that owns it, ``stock/check_pins.py`` ``stack_verdict``: ``{"stack_pinned": True|False|None, "stack_line": None |
    "STACK not pinned: torch <v>+<cuda> (pinned: torch 2.13.0+cu130)", "stack_strict": bool}``. ``stack_pinned`` None (with a
    ``STACK unknown: ...`` line) only when the check itself could not run — named, never read as pinned. A malformed
    ``CHAI1_OPT_STRICT_STACK`` value is a ValueError naming it (gates turns it into a refusal)."""
    environ = os.environ if environ is None else environ
    mod = _check_pins_module()
    strict = mod.strict_stack((), environ)
    try:
        t = mod.stack_verdict(pins())
    except Exception as e:  # noqa: BLE001
        return {"stack_pinned": None, "stack_line": f"STACK unknown: the stack verdict of stock/check_pins.py failed to run: {e!r}", "stack_strict": strict}
    return {"stack_pinned": bool(t["pinned"]), "stack_line": t["line"], "stack_strict": strict}


def dist_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _metadata_cuda() -> Optional[str]:
    """torch's CUDA version from its installed files without importing it (stock/check_pins.py ``torch_build``), or None."""
    try:
        return _check_pins_module().torch_build()[1]
    except Exception:  # noqa: BLE001
        return None


def torch_version() -> Optional[str]:
    """torch's version with its local tag (``2.13.0+cu130``): ``torch.__version__`` when torch is imported in this process, else the installed
    distribution's build read WITHOUT importing it (stock/check_pins.py ``torch_build``: the metadata version completed from torch/version.py) —
    one answer on every route of a box, the parent of the stock route (no torch import) included."""
    if "torch" in sys.modules:
        return getattr(sys.modules["torch"], "__version__", None)
    try:
        return _check_pins_module().torch_build()[0]
    except Exception:  # noqa: BLE001                                       # an unreadable checker: the bare metadata version, as before
        return dist_version("torch")


# ------------------------------------------------------------------------------------------------------------------- GPU
def gpu_info(use_torch: bool = False) -> dict:
    """``{"visible": bool, "name", "cc", "mem_mib", "source"}`` — from torch when it is already imported (or asked for), else nvidia-smi."""
    if use_torch or "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                return {"visible": True, "name": p.name, "cc": f"{p.major}.{p.minor}", "mem_mib": int(p.total_memory // 2**20),
                        "source": "torch", "cuda": torch.version.cuda}
            return {"visible": False, "name": None, "cc": None, "mem_mib": None, "source": "torch", "cuda": torch.version.cuda}
        except Exception as e:  # noqa: BLE001
            return {"visible": False, "name": None, "cc": None, "mem_mib": None, "source": f"torch: {e!r}"}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        line = (r.stdout.strip().splitlines() or [""])[0]
        if r.returncode == 0 and line:
            name, cc, mem = [x.strip() for x in line.split(",")[:3]]
            return {"visible": True, "name": name, "cc": cc, "mem_mib": int(float(mem)), "source": "nvidia-smi"}
        return {"visible": False, "name": None, "cc": None, "mem_mib": None, "source": "nvidia-smi: " + (r.stderr.strip()[:120] or "no device")}
    except Exception as e:  # noqa: BLE001
        return {"visible": False, "name": None, "cc": None, "mem_mib": None, "source": f"nvidia-smi: {e!r}"}


def stack_key(torch_v: Optional[str] = None, cuda: Optional[str] = None, cc: Optional[str] = None) -> str:
    """``torch<version sans local tag>-cu<CUDA sans dot>-sm<cc digits>``: the stack label of the activation report (``stack_key``); a caller
    keying per-stack caches uses the same spelling."""
    tv = torch_v or torch_version() or "unknown"
    tv = tv.split("+")[0]
    if cuda is None:
        m = re.search(r"\+cu(\d+)", torch_v or torch_version() or "")
        cuda = m.group(1) if m else (gpu_info().get("cuda") or _metadata_cuda() or "").replace(".", "") or "unknown"
    cc = (cc or gpu_info().get("cc") or "unknown").replace(".", "")
    return f"torch{tv}-cu{str(cuda).replace('.', '')}-sm{cc}"


# ---------------------------------------------------------------------------------------------------------------- weights
def weights_check(downloads_dir: Optional[str] = None) -> Tuple[bool, str, List[str]]:
    """CHAI_DOWNLOADS_DIR must name a directory holding the 8 weight files of stock/PINS.json (nothing is fetched at run time)."""
    d = downloads_dir or os.environ.get("CHAI_DOWNLOADS_DIR")
    if not d:
        return False, "CHAI_DOWNLOADS_DIR is not set — see README Variables (the directory with chai-lab's downloads layout holding the weight files of stock/PINS.json; e.g. export CHAI_DOWNLOADS_DIR=/data/chai_downloads)", []
    if not os.path.isdir(d):
        return False, f"CHAI_DOWNLOADS_DIR={d} does not exist", []
    missing = [w["local"] for w in pins()["weights"]["files"] if not os.path.isfile(os.path.join(d, w["local"]))]
    if missing:
        return False, f"CHAI_DOWNLOADS_DIR={d} lacks {len(missing)} of the weight files (stock/PINS.json): {missing[:3]}{'...' if len(missing) > 3 else ''}", missing
    return True, d, []


def esm_memo_scope(environ=None) -> str:
    """``CHAI1_OPT_ESM_MEMO_SCOPE``: ``global`` (default: the W5 memo of per-sequence ESM2 embeddings serves every input of the process — a
    sequence embedded once is never embedded again) or ``input`` (the memo is emptied when an input's feature context is built: chains
    repeated INSIDE one input still hit, every input computes its own embeddings — every input pays its own embedding cost, as stock does).
    Any other word is refused by name (ActivationError)."""
    environ = os.environ if environ is None else environ
    v = str(environ.get(ENV_ESM_MEMO_SCOPE, "") or "global").strip().lower()
    if v not in ESM_MEMO_SCOPES:
        raise ActivationError(f"{ENV_ESM_MEMO_SCOPE}={environ.get(ENV_ESM_MEMO_SCOPE)!r} is not one of {'|'.join(ESM_MEMO_SCOPES)}")
    return v


ESM_SCOPE_STATS = {"scope": None, "clears": 0}


def install_esm_memo_scope(chai1_mod, namespace: Optional[dict], scope: str) -> None:
    """Give the W5 memo (``ESM_CACHE`` in the namespace the kit's statements run in) its scope: under ``input`` the memo is emptied at the
    start of every ``chai_lab.chai1.make_all_atom_feature_context`` call (one call = one input's features on every route); under
    ``global`` nothing is wrapped. Idempotent; counted (``ESM_SCOPE_STATS``: scope, clears)."""
    ESM_SCOPE_STATS["scope"] = scope
    if scope != "input":
        return
    if not isinstance(namespace, dict) or "levels" not in namespace:            # the kit statements' namespace carries `levels`; anything else is a wiring error, named
        raise ActivationError(f"{ENV_ESM_MEMO_SCOPE}=input: the namespace handed to install_esm_memo_scope is not the kit's (no `levels`) — the memo cannot be scoped")
    cache = namespace.get("ESM_CACHE")
    if "W5" in set(namespace["levels"]) and not isinstance(cache, dict):
        raise ActivationError(f"{ENV_ESM_MEMO_SCOPE}=input: W5 is among the levels but its memo (ESM_CACHE) is not in the kit's namespace — the memo cannot be scoped")
    cur = chai1_mod.make_all_atom_feature_context
    if getattr(cur, "chai1_opt_esm_scope", False):
        return
    orig = cur

    def make_all_atom_feature_context(*args, **kw):
        if isinstance(cache, dict):                                           # no W5 in this process: nothing memoised, nothing to empty
            cache.clear(); ESM_SCOPE_STATS["clears"] += 1
        return orig(*args, **kw)
    for attr in ("chai1_opt_lever",):                                        # keep the msa_form wrapper's mark visible when it sits underneath
        if hasattr(orig, attr):
            setattr(make_all_atom_feature_context, attr, getattr(orig, attr))
    make_all_atom_feature_context.chai1_opt_esm_scope = True
    make_all_atom_feature_context.__wrapped__ = orig
    chai1_mod.make_all_atom_feature_context = make_all_atom_feature_context


def esm_scope_tally_fields() -> list:
    sc = ESM_SCOPE_STATS["scope"]
    return [f"esm_memo_scope={sc} esm_memo_clears={ESM_SCOPE_STATS['clears']}"] if sc else []


MSA_FORMS: List[dict] = []                                   # one record per feature context the driver builds (install_msa_form)


def install_msa_form(chai1_mod) -> None:
    """The driver's MSA-form statement: every fold's ``msa_directory`` follows chai-lab's own form for the item — the ``--msa_dir``
    directory when at least one chain of the FASTA has its ``.aligned.pqt`` there, ``None`` (upstream's empty MSA context) otherwise —
    the rule the ``off`` route folds by (``stock_fold.msa_directory_for``, the one function both routes use), so every kit mode folds the
    features stock's own command line folds for the same item and alignments. Wraps ``chai_lab.chai1.make_all_atom_feature_context`` (the
    worker builds every feature context through it, keywords only); prints one ``MSA form=`` line per context and records it
    (``MSA_FORMS``: the EXIT tally carries the census)."""
    from . import stock_fold as _stock_fold
    orig = chai1_mod.make_all_atom_feature_context
    if getattr(orig, "chai1_opt_lever", None) == "msa_form":
        return

    def make_all_atom_feature_context(*args, **kw):
        fasta = kw.get("fasta_file"); given = kw.get("msa_directory")
        d, present, n_chains = _stock_fold.msa_directory_for(str(fasta), str(given) if given is not None else None)
        depth_chains, depths = _stock_fold.msa_depths(str(fasta), str(given) if given is not None else None)
        kw["msa_directory"] = d
        item = os.path.basename(str(fasta))
        MSA_FORMS.append({"item": item, "msa_form": "directory" if d else "none", "aligned_pqt_present": len(present), "n_chains": n_chains, "depths": list(depths)})
        sys.stderr.write(f"[{_core.TAG}] MSA form={'directory' if d else 'none'} item={item} aligned_pqt={len(present)}/{n_chains} "
                         f"(chai-lab's own form, the off route's rule)\n")
        sys.stderr.write(_report.msa_depth_line(f"[{_core.TAG}]", item, depth_chains, depths, given) + "\n")
        return orig(*args, **kw)
    make_all_atom_feature_context.chai1_opt_lever = "msa_form"
    chai1_mod.make_all_atom_feature_context = make_all_atom_feature_context


def msa_form_fields() -> Optional[str]:
    """``msa_form_directory=<n> msa_form_none=<n>`` for the EXIT tally (None before any context was built)."""
    if not MSA_FORMS:
        return None
    n_dir = sum(1 for r in MSA_FORMS if r["msa_form"] == "directory")
    return f"msa_form_directory={n_dir} msa_form_none={len(MSA_FORMS) - n_dir}"


def weights_memo_dir() -> str:
    """The kit's cache root for the weights digest memo: ``<XDG_CACHE_HOME or ~/.cache>/chai1_opt`` (the memo file is
    ``digest_memo.MEMO_NAME`` = ``weights_digests.json`` in it). A file matches by digest only — size/mtime/inode select the memo entry, never
    decide the match (``digest_memo``)."""
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "chai1_opt")


def memo_dir_unwritable(memo_dir: str) -> Optional[str]:
    """None when the digest memo's directory can be created and written; else the reason (``<dir>: <error>``). A read-only cache root (a
    read-only mount, another user's directory) is a note on the activation line and digests computed afresh — never a refusal."""
    import tempfile
    try:
        os.makedirs(memo_dir, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".probe.", dir=memo_dir)
        os.close(fd); os.unlink(probe)
        return None
    except OSError as e:
        return f"{memo_dir}: {e.strerror or e.__class__.__name__}"


def weights_memo_note(wid: dict) -> Optional[str]:
    """The note a read-only memo directory carries on the activation line (None when the memo is writable)."""
    if not wid or not wid.get("memo_unwritable"):
        return None
    return f"WEIGHTS memo unwritable ({wid['memo_unwritable']}): digests computed afresh, not memoised"


def weights_match(d: str, afresh: bool = False) -> dict:
    """The checkpoint by digest: the sha256 digest of each pinned weight file in ``d`` against ``stock/PINS.json`` — ``pinned`` (every file has
    the PINS digest) or ``unknown`` (the files that differ, named by digest). A note, never a gate: an unknown checkpoint is a
    WARNING by name on the ACTIVE / DRY-RUN line and the run proceeds in every mode. Every digest goes through ``digest_memo.digest(path,
    weights_memo_dir(), refresh=afresh)``: a memo entry selected by the file's (realpath, st_size, st_mtime_ns, st_ino) is read without hashing
    and the record's words (``words``) then carry ``(cached digest <utc>)`` while the ACTIVE line's last token reads ``weights_digest=cached@<utc>``;
    otherwise the file is hashed in full and its entry written (atomically, only after the hash completed). ``afresh=True`` — ``check``, every dry run — hashes every file afresh and rewrites its entry. A file matches by digest only —
    size/mtime/inode select the memo entry, never decide the match."""
    from . import digest_memo
    t0 = time.time(); unknown = []; n = 0; cached_utcs = []; memo_dir = weights_memo_dir()
    unwritable = memo_dir_unwritable(memo_dir)                       # a read-only cache root: named, digests computed afresh, nothing memoised — never a refusal
    for w in pins()["weights"]["files"]:
        if unwritable:
            digest, cached_utc = digest_memo.sha256_file(os.path.realpath(os.path.join(d, w["local"]))), None
        else:
            digest, cached_utc = digest_memo.digest(os.path.join(d, w["local"]), memo_dir, refresh=afresh)
        n += 1
        if cached_utc is not None:
            cached_utcs.append(str(cached_utc))
        if digest != w["sha256"]:
            unknown.append({"local": w["local"], "sha256": digest})
    status = "unknown" if unknown else "pinned"; cached_utc = min(cached_utcs) if cached_utcs else None
    return {"status": status, "files": n, "unknown": unknown, "seconds": round(time.time() - t0, 1), "hashed": n - len(cached_utcs), "cached": len(cached_utcs),
            "cached_utc": cached_utc, "afresh": bool(afresh) or bool(unwritable), "memo": os.path.join(memo_dir, digest_memo.MEMO_NAME),
            "memo_unwritable": unwritable,                                  # None, or the reason the cache root cannot be written (the note names it)
            "words": digest_memo.word(status, cached_utc)}                 # the record's human words: `pinned` | `unknown` + ` (cached digest <utc>)` on a memo hit


def weights_digest_token(wid: dict) -> str:
    """The ACTIVE / DRY-RUN line's last token value, whitespace-free: ``fresh`` (every digest computed by this process) or ``cached@<utc>``
    (digests read from the memo; the oldest cached time). The human words ``(cached digest <utc>)`` live in the record (``words``), never
    inside an ACTIVE-line token."""
    return "fresh" if not wid.get("cached") else f"cached@{wid.get('cached_utc')}"


def weights_note(wid: dict) -> Optional[str]:
    """The WARNING an unknown checkpoint carries on the activation line (None for the pinned checkpoint)."""
    if not wid or wid.get("status") != "unknown":
        return None
    named = ", ".join(f"{u['local']}:{u['sha256'][:12]}" for u in wid["unknown"][:3]) + (" …" if len(wid["unknown"]) > 3 else "")
    return f"WEIGHTS unknown sha={named} ({len(wid['unknown'])}/{wid['files']} files not the pinned checkpoint, stock/PINS.json); proceeding"


# --------------------------------------------------------------------------------------------------- the kit's own statements
def kit_lever_statements(path: Optional[str] = None) -> Tuple[List[ast.stmt], str]:
    """The driver's top-level W1/W5 block: from the ``_MODULE_CACHE = {}`` assignment through the ``if "W5" in levels:`` block, as AST
    nodes with the kit file's own line numbers. Returns ``(nodes, source_slice)``."""
    path = path or driver_path()
    src = open(path, "r", encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    start = end = None
    for i, node in enumerate(tree.body):
        if start is None and isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_MODULE_CACHE" for t in node.targets):
            start = i
        if start is not None and isinstance(node, ast.If) and isinstance(node.test, ast.Compare) \
                and isinstance(node.test.left, ast.Constant) and node.test.left.value == "W5":
            end = i
            break
    if start is None or end is None:
        raise ActivationError(f"the kit driver's W1/W5 block was not found in {path}: the kit bytes are not the ones this package wraps")
    nodes = tree.body[start:end + 1]
    lines = src.splitlines()
    return nodes, "\n".join(lines[nodes[0].lineno - 1:nodes[-1].end_lineno])


def apply_kit_levers(levels, chai1_mod, esm_mod, torch_mod) -> dict:
    """Execute the kit's W1/W5 statements with ``levels`` bound (the driver's own namespace shape). Returns the namespace."""
    nodes, _ = kit_lever_statements()
    ns = {"__name__": "chai1_opt.kit_levers", "__file__": driver_path(), "levels": set(levels), "chai1": chai1_mod, "esm_mod": esm_mod,
          "torch": torch_mod, "time": time, "os": os, "sys": sys}
    code = compile(ast.Module(body=nodes, type_ignores=[]), driver_path(), "exec")
    exec(code, ns)  # noqa: S102  the kit's own statements, byte-for-byte from the carried file
    return ns


def classify(chai1_mod=None, esm_mod=None) -> Dict[str, List[str]]:
    """What the kits' own state shows applied: the attribute probes of registry.LEVERS and the eager probes against the installed
    StackHandle (never a value asserted by the package). With the eager stack installed, ``chai1.load_exported`` holds its loader and
    the driver kit's function is the one the handle wraps (``StackHandle.orig``): the W1 probe looks through the handle."""
    applied, off = [], []
    h = _STATE["eager"]
    for name, (_, modname, attr, fn) in registry.ATTR_PROBES.items():
        mod = sys.modules.get(modname)
        if modname == "chai_lab.chai1" and chai1_mod is not None:
            mod = chai1_mod
        if modname == ESM_MODULE and esm_mod is not None:
            mod = esm_mod
        obj = getattr(mod, attr, None) if mod is not None else None
        if h is not None and obj is not None and obj is getattr(h, "loader", None):
            obj = getattr(h, "orig", None)                                 # the function the eager loader wraps
        (applied if getattr(obj, "__name__", None) == fn else off).append(name)
    mod = chai1_mod if chai1_mod is not None else sys.modules.get(TRIGGER_MODULE)
    le = getattr(mod, "load_exported", None) if mod is not None else None
    for name, (_, levers) in registry.EAGER_PROBES.items():
        (applied if eager_installed_levers(h, le) == levers else off).append(name)
    installed_dstep = dstep_installed_levers(h, le)
    for name, (_, lever) in registry.DSTEP_PROBES.items():
        (applied if lever in installed_dstep else off).append(name)
    for name in registry.PAIRTRACK_PROBES:                                  # the trunk levers: pairtrack's install record names the lever
        (applied if name in _pairtrack.applied() else off).append(name)
    torch_mod = sys.modules.get("torch")
    for name in registry.FLAG_PROBES:                                       # the package's opt-in levers: torch's own flag (precision.probe)
        (applied if torch_mod is not None and _precision.probe(name, torch_mod) else off).append(name)
    for name in registry.ENV_PROBES:                                        # alloc: the environment carries the core policy's configuration
        (applied if _alloc.applied() else off).append(name)
    if registry.BIG_PROBES:                                               # the memory line: the applied record of opt_core.mem names the lever
        from . import big as _big
        rec = _big.record()
        on = set(rec.levers) if rec is not None and not rec.refused else set()
        for name, (_, lever) in registry.BIG_PROBES.items():
            (applied if lever in on else off).append(name)
    return {"applied": applied, "off": off}


# ------------------------------------------------------------------------------------------------------------- opt-in levers
def optin_env(optin: Tuple[str, ...]) -> Dict[str, str]:
    """The environment rows the requested opt-in levers need in a CHILD process (the driver route's child): ``alloc``'s allocator
    configuration; {} for levers applied in-process. LeverUnavailable of the lever's module when the core cannot provide it."""
    rows: Dict[str, str] = {}
    if "alloc" in optin:
        rows.update(_alloc.env_row())
    return rows


def apply_optin_pre(optin: Tuple[str, ...], torch, *, export_alloc: bool = True) -> dict:
    """Apply the opt-in levers that must be in place BEFORE the eager stack installs: ``tf32`` (torch's flag; the hoisted step's graph
    capture reads it) and ``alloc`` (exported into this process before its first CUDA allocation — ``export_alloc=False`` when the caller
    already exported it earlier in the process, as the driver does at start). Returns the facts for the report; raises the lever module's
    exception (its wording is the refusal)."""
    facts = dict(_precision.apply(tuple(n for n in optin if n in registry.FLAG_PROBES), torch))
    if "alloc" in optin and export_alloc:
        facts.update(_alloc.export())
    return facts





def dstep_installed_levers(handle, load_exported) -> Tuple[str, ...]:
    """The DSTEP levers the installed handle's denoiser carries (``chai1_fastln.stackx.build_lever_parts``: the HoistedDiffusionWrapper's
    ``hoister`` is ``hoist2``; its ``compile`` mode is set and did not step aside; its ``dit_attn`` Router is bound), empty when the attribute
    is not the handle's loader or the denoiser carries none."""
    if handle is None or load_exported is None or load_exported is not getattr(handle, "loader", None):
        return ()
    dw = (getattr(handle, "parts", None) or {}).get("diffusion")
    if dw is None:
        return ()
    out = []
    if getattr(dw, "hoister", "base") == "hoist2":                          # the eager stack's value-taint hoister (stackx HOIST_LEVERS)
        out.append("hoist2")
    if getattr(dw, "compile", None) and not getattr(dw, "compile_failed", None):   # the compiled per-step function (stackx COMPILE_LEVERS); stepped aside = off, by name
        out.append("compiled")
    if getattr(dw, "dit_attn", None) is not None:                                     # the DiT token attention through kernels.apb (stackx ATTN_LEVERS)
        out.append("dit_attn")
    return tuple(out)


def eager_installed_levers(handle, load_exported) -> Optional[str]:
    """The eager lever the installed handle serves through ``load_exported`` (``chai1_eager.stack.build_parts``: a handle with a trunk part
    is the ``tier1`` line), or None when the attribute is not the handle's loader."""
    if handle is None or load_exported is None or load_exported is not getattr(handle, "loader", None):
        return None
    trunk = (getattr(handle, "parts", None) or {}).get("trunk")
    if trunk is None:
        return None
    return "tier1"


def eager_module():
    """``chai1_eager.stack`` from the carried eager directory (first on sys.path, like the kit's ``kit/`` for chai_proto)."""
    home = eager_home()
    if home not in sys.path:
        sys.path.insert(0, home)
    mod = importlib.import_module(modes.EAGER_PACKAGE + ".stack")
    want = eager_stack_path()
    if not os.path.samefile(mod.__file__, want):
        raise ActivationError(f"{modes.EAGER_PACKAGE}.stack resolved to {mod.__file__}, not the carried {want}")
    return mod


def dstep_module():
    """``chai1_fastln.stackx`` from the carried DSTEP directory (on sys.path like the eager directory; the add-on imports ``chai1_eager``)."""
    home = dstep_home()
    if home not in sys.path:
        sys.path.insert(0, home)
    eager_module()                                                          # chai1_eager resolvable first (the add-on's own import)
    mod = importlib.import_module(modes.DSTEP_PACKAGE + ".stackx")
    want = dstep_stackx_path()
    if not os.path.samefile(mod.__file__, want):
        raise ActivationError(f"{modes.DSTEP_PACKAGE}.stackx resolved to {mod.__file__}, not the carried {want}")
    return mod


def apply_eager(levers: str, device: str = "cuda:0", dstep: Tuple[str, ...] = (), pairtrack: Tuple[str, ...] = (), mode: str = "fast"):
    """Install the eager stack's lever through its own ``install`` (``chai1_eager/stack.py``): patches ``chai1.load_exported`` in
    place, on top of whatever holds it (the driver's W1 memo on the kit routes). ``graphed`` is left to the stack's own rule
    (graphs unless deterministic algorithms are on — the recipe must be applied before this call). With ``dstep`` levers the denoiser
    is the add-on's (``chai1_fastln.stackx.build_lever_parts`` over the stack's ``build_parts``, served by the stack's ``make_loader``
    — the add-on's own recipe, DSTEP README.md). On either route the installed denoiser and trunk are adopted by ``hoist`` (the
    item-keyed hoist, released at every item boundary and at the item's confidence head — the handle's loader wrapped) before the handle is kept. ``pairtrack``: the trunk levers (``pairtrack.LEVERS``)
    installed on the adopted eager trunk for ``mode`` (``pairtrack.install``: exact = its bitwise constructions, fast = Tier 2). Returns the
    StackHandle."""
    # --- the row's host-side levers (modes.KitMode.postproc: postproc.py rankcc / tailasync, featfast.py confmemo / prefetch) — install-style
    #     rebindings of the fold body's module attributes; both routes converge here (chai1_eager.stack.install is not the DSTEP rows' path) ---
    _pp = tuple(modes.kit_mode(mode).postproc) if mode in modes.KIT_MODES else ()
    if _pp:
        from . import featfast as _featfast, postproc as _postproc
        _postproc.install(tuple(n for n in _pp if n in _postproc.LEVERS))
        _featfast.install(tuple(n for n in _pp if n in _featfast.LEVERS))
    # --- end host-side levers ---

    S = eager_module()
    if levers not in S.LEVERS or levers == "stock":
        raise ActivationError(f"eager lever {levers!r} is not one of the stack's own ({', '.join(S.LEVERS)})")
    if _STATE["eager"] is not None:
        raise ActivationError("the eager stack is already installed in this process (one mode per process)")
    from . import big                                                         # the memory mode's graphed override + instance hooks (None / no-op outside big)
    if not dstep:
        h = S.install(levers=levers, device=device, graphed=big.graphed_override())   # None = the stack's own rule; False under big's nograph
        hoist.adopt(h.parts, S, handle=h)                                      # + the handle: its loader serves the confidence head that releases the item's denoiser state
        if pairtrack:                                                           # the trunk levers on the adopted eager trunk, BEFORE the memory line: big's
            _pairtrack.install(h, S, tuple(pairtrack), mode=mode)               # trunk_chunk wraps cfg_fn last and keeps their plug below its token gate
        big.on_adopted(h.parts, S)                                            # the memory levers on the adopted instances (no-op outside big)
        _STATE["eager"] = h
        return h
    X = dstep_module()
    bad = [lv for lv in dstep if lv not in X.ALL_LEVERS]
    if bad:
        raise ActivationError(f"DSTEP lever(s) {','.join(bad)} not among the add-on's own ({', '.join(X.ALL_LEVERS)})")
    import torch  # noqa: PLC0415  (the recipe is applied before this call; the stack's own graphed rule reads torch's state)
    graphed = not torch.are_deterministic_algorithms_enabled()
    if big.graphed_override() is not None:
        graphed = big.graphed_override()                                      # big's nograph lever: un-graphed whatever the recipe says
    comps = S.Components(device=device)
    base = {levers: S.build_parts(comps, levers, graphed=graphed)}
    parts = X.build_lever_parts(comps, base, levers, tuple(dstep), graphed=graphed, mode=mode)   # the mode's tier word is what its provider-served levers ask by
    loader = S.make_loader(comps, parts)
    comps.C1.load_exported = loader
    h = S.StackHandle(comps.C1, comps.orig_load, loader, parts)
    hoist.adopt(parts, S, handle=h)                                             # as above: the confidence head served through h.loader releases at its entry
    if pairtrack:                                                               # before the memory line, as above
        _pairtrack.install(h, S, tuple(pairtrack), mode=mode)
    big.on_adopted(parts, S)
    _STATE["eager"] = h
    return h


def dstep_stand_down(lever: str, on: bool) -> Optional[str]:
    """Per-item hook for a memory line (big.py's gate table calls it at the item's trunk call, beside its `graphed` switch): with ``on``
    the adopted denoiser runs THIS item on the base hoister — ``lever`` (``hoist2``) stood down by name, its larger hoist cache being what a
    40 GB card cannot hold at the top crops; ``on`` False restores the row's hoister for the next item. Returns the hoister the next item
    will run (None when no eager stack / no denoiser is installed, or ``lever`` is not the denoiser's hoister — nothing to stand down).
    base and hoist2 are bitwise identical: the stand-down moves memory and speed only, never the outputs."""
    h = _STATE["eager"]
    dw = (getattr(h, "parts", None) or {}).get("diffusion") if h is not None else None
    if dw is None or getattr(dw, "hoister", "base") != lever:
        return None
    dw.stand_down = "base" if on else None
    return dw.hoister_now() if hasattr(dw, "hoister_now") else (dw.stand_down or dw.hoister)


def eager_stats() -> Optional[dict]:
    """The installed handle's own counters (``StackHandle.stats``): diffusion events (hoist builds, graph-capture fallbacks), the DSTEP
    policy's counters and the item-keyed hoist's (``hoist.stats``: precomputes, releases); None when no eager stack is installed."""
    h = _STATE["eager"]
    if h is None:
        return None
    try:
        out = h.stats()
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    dw = (getattr(h, "parts", None) or {}).get("diffusion")
    pol = getattr(dw, "ln_policy", None) if dw is not None else None
    if pol is not None:
        out["dstep_ln"] = {"mode": getattr(pol, "mode", None), "n_fast": getattr(pol, "n_fast", None), "n_slow": getattr(pol, "n_slow", None)}
    if dw is not None and getattr(dw, "dit_attn", None) is not None:
        try:
            DA = sys.modules.get("chai1_fastln.dit_attn")
            out["dstep_dit_attn"] = DA.report() if DA is not None else {"installed": True}
        except Exception:  # noqa: BLE001
            out["dstep_dit_attn"] = {"installed": True}
    if dw is not None and getattr(dw, "compile", None):
        out["dstep_compile"] = {"mode": dw.compile, "failed": getattr(dw, "compile_failed", None),
                                "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR") or "(inductor default)"}
        if getattr(dw, "aoti", None) is not None:                                      # the compiled step's ahead-of-time route (chai1_fastln.aoti, bound by stackx.build_lever_parts)
            try:
                out["dstep_aoti"] = dw.aoti.report()
            except Exception:  # noqa: BLE001
                out["dstep_aoti"] = {"n_loaded": 0, "n_asked": 0}
    hs = hoist.stats(dw)
    if hs is not None:
        out["hoist"] = hs
    return out


def late_activation_reason(chai1_mod, esm_mod) -> Optional[str]:
    if getattr(esm_mod, "_esm_model", None):
        return "the traced ESM model is already loaded (a fold has started): enable() must run before the first fold"
    le = getattr(chai1_mod, "load_exported", None)
    if getattr(le, "__module__", None) != "chai_lab.chai1":
        return (f"chai1.load_exported is {getattr(le, '__module__', None)}.{getattr(le, '__name__', None)}, not upstream's own function: "
                "a lever is already applied in this process (or another shim patched it)")
    return None


# ------------------------------------------------------------------------------------------------------------ activation
def _refuse(rep: dict, reason: str, strict: bool, dry_run: bool) -> dict:
    rep.update(active=False, reason=reason)
    _report.print_activation(rep)
    prev = _STATE["report"]
    if not dry_run and not (prev is not None and prev.get("active")):      # an active mode's report is never displaced by a refusal
        _STATE["report"] = rep
    if strict and not dry_run:
        raise ActivationError(reason)
    return rep


def base_report(mode: str, trigger: Optional[str], dry_run: bool, det: int = 0, optin: Tuple[str, ...] = (),
                n_gpu: int = 1) -> dict:
    ok, kh = kit_present()
    try:
        cert = stack_pinning()                                        # every route starts here: the STACK record rides on every report (off included)
    except Exception as e:  # noqa: BLE001                                 # a malformed CHAI1_OPT_STRICT_STACK or an unreadable checker: named on the record; gates refuses on it
        cert = {"stack_pinned": None, "stack_line": f"STACK unknown: {e}", "stack_strict": None}
    modes.apply_levers_off()                                               # MODEL_OPT_LEVERS_OFF re-read: the rows this report describes are the reduced ones
    return {"active": False, "mode": mode, "optin": list(optin), "n_gpu": int(n_gpu), "levers_off_env": list(modes.levers_off_for(mode)),
            "levels": modes.driver_levels(mode) if mode in modes.KIT_MODES else None,
            "eager": modes.eager_lever(mode) if mode in modes.KIT_MODES else None, "dstep": modes.kit_mode(mode).dstep_arg if mode in modes.KIT_MODES and modes.kit_mode(mode).dstep else None,
            "route": "in-process", "levers_applied": [], "levers_not_applicable": [], "levers_off": [], "partial": False,
            "chai_lab_version": dist_version(UPSTREAM_DIST), "torch_version": torch_version(), "python": platform.python_version(),
            "package_version": __version__, "kit_home": kh if ok else None, "tree_home": tree_home(), "trigger": trigger,
            "dry_run": dry_run, "det": int(det), "gpu": None, "stack_key": None, "target_gpu": os.environ.get("MODEL_OPT_TARGET_GPU") or None,
            "stack_pinned": cert["stack_pinned"], "stack_line": cert["stack_line"], "stack_strict": cert["stack_strict"],
            "notes": [], "reason": None, "t_s": round(time.time() - T0, 2)}


def strict_stack_refusal(rep: dict) -> Optional[str]:
    """The strict switch applied to the report's STACK record (base_report's ``stack_pinned`` / ``stack_line`` / ``stack_strict``): None when
    the stack is the pinned one or the switch is off; else the refusal by name. A record the checker could not compute (``stack_pinned``
    None) is a refusal whatever the switch says — never read as pinned."""
    if "stack_pinned" not in rep:                                       # a bare report (tests): compute the record here, once
        rep.update(stack_pinning())
    if rep.get("stack_pinned") is None:
        return f"{rep.get('stack_line') or 'STACK unknown'} (the stack record could not be computed; refused rather than assumed pinned)"
    if rep.get("stack_strict") and not rep["stack_pinned"]:
        return (f"{rep['stack_line']} — refused under {ENV_STRICT_STACK}=1 (strict: every mode runs on the pinned torch/CUDA stack only; "
                f"unset it to run here with the STACK record)")
    return None


def gates(rep: dict, need_gpu: bool = True, use_torch: bool = False) -> Optional[str]:
    """Declared variables -> kit present -> stock pin -> the STACK record's strict switch -> the mode's capability on this box (a C compiler
    for Triton-kernel modes) -> weights -> GPU (the core pin is activate's statement one, before this). Returns the first refusal (None = all
    pass); fills the report's facts."""
    from . import _autoload
    bad = _autoload.undeclared(os.environ)
    if bad:
        return f"undeclared variable(s) {', '.join(bad)} (this package reads {', '.join(DECLARED_ENV + DECLARED_BIG_ENV)})"
    try:
        rep["esm_memo_scope"] = esm_memo_scope()                          # the W5 memo's scope word, validated before anything loads
    except ActivationError as e:
        return str(e)
    ok, why = kit_present()                                                # (the core pin is gated before this: activate's statement one, _core.core_gate)
    if not ok:
        return why
    from opt_core.oom import is_oom
    try:
        bad, detail = pins_check()
    except Exception as e:  # noqa: BLE001
        if is_oom(e): raise                                       # noqa: E701
        return f"stock pin check failed to run: {e!r}"
    rep["pins"] = detail
    if bad:
        return "stock pin: " + "; ".join(bad)
    why = strict_stack_refusal(rep)                                        # a stack other than the pinned one: recorded (base_report) and run; refused by name only under CHAI1_OPT_STRICT_STACK=1
    if why:
        return why
    refusal, sn = mode_stack_check(modes.kit_mode(rep["mode"])) if rep.get("mode") in modes.KIT_MODES else (None, None)
    if refusal:                                                            # the mode's kernels cannot build on this box: by name, before anything loads
        return refusal
    if sn:
        rep.setdefault("notes", []).append(sn)
    wok, wwhy, _ = weights_check()
    rep["weights_dir"] = wwhy if wok else None
    if not wok:
        return wwhy
    afresh = bool(rep.get("dry_run")) and rep.get("trigger") != "pred"       # `check` / an API dry run hash every pinned file afresh; `pred`'s pre-flight gate reads the
    wid = weights_match(wwhy, afresh=afresh)                       # (realpath, size, mtime_ns, ino)-keyed digest memo (a full re-hash of every file per run otherwise) — pinned | unknown by digest: a note
    if wid is not None:
        rep["weights"] = wid
        for note in (weights_note(wid), weights_memo_note(wid)):
            if note:
                rep.setdefault("notes", []).append(note)
    g = gpu_info(use_torch=use_torch)
    rep["gpu"] = g
    rep["stack_key"] = stack_key(cc=g.get("cc")) if g.get("visible") else None
    if need_gpu and not g.get("visible"):
        return f"no CUDA device visible ({g.get('source')})"
    note = target_gpu_note(rep.get("target_gpu"), g)
    if note:
        rep.setdefault("notes", []).insert(0, note)
    return None


C_COMPILERS = ("cc", "gcc", "clang")                     # Triton builds each kernel's launcher with $CC, else the first of these on PATH


def c_compiler() -> Optional[str]:
    """The C compiler Triton's launcher build would use on this box (``$CC`` when it resolves, else cc / gcc / clang on PATH), or None."""
    cc = os.environ.get("CC")
    found = shutil.which(cc) if cc else None
    for name in C_COMPILERS:
        found = found or shutil.which(name)
    return found


def mode_stack_check(km, compiler: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """``(refusal, note)``: the row's real capability check on this box, by its ``stack_rule`` (modes.KitMode). ``triton`` (fast, big) — the
    DSTEP add-on's and the shared core's Triton kernels build their launchers with a C compiler at first use, so without one (``$CC`` / cc /
    gcc / clang on PATH) the mode is refused by name before anything loads: ``mode <m> needs a C compiler for its Triton kernels ($CC / cc /
    gcc / clang: none on PATH) — --mode exact and --mode off need none``. ``bitwise`` (exact) — nothing to check on the box: exact is bitwise vs
    stock on one and the same torch / CUDA stack (the pinned one: ``KitMode.stack``); on another stack the run
    carries the STACK record (base_report ``stack_line``). ``compiler`` defaults to
    the box's (:func:`c_compiler`). No note is produced by either rule."""
    if getattr(km, "stack_rule", None) != "triton":
        return None, None
    compiler = compiler if compiler is not None else c_compiler()
    if not compiler:
        return (f"mode {km.name} needs a C compiler for its Triton kernels ($CC / {' / '.join(C_COMPILERS)}: none on PATH) "
                f"— --mode exact and --mode off need none"), None
    return None, None


def target_gpu_note(target: Optional[str], g: Optional[dict]) -> Optional[str]:
    """``MODEL_OPT_TARGET_GPU=<t> but the GPU is <name>: not the GPU this configuration targets`` when the target name is not part of
    the visible GPU's name (case-insensitive); None when they agree, when no target is set or when no GPU name is known."""
    name = (g or {}).get("name")
    if target and name and target.lower() not in name.lower():
        return f"MODEL_OPT_TARGET_GPU={target} but the GPU is {name}: not the GPU this configuration targets"
    return None


PARTIAL_ENV_NOTE = ("partial activation requested: {env}={raw!r} leaves lever(s) {off} of mode {mode} off for this run (an ablation, not a benchmark configuration); "
                    "named here and in the report (partial, levers_off, levers_off_env); the fold proceeds on the levers in levers_applied")
PARTIAL_NOTE = "partial activation: lever(s) {off} not applied — decided at activation (the run's own switches), named here and in the report (partial, levers_off); the fold proceeds on the levers that did"
PARTIAL_REFUSAL = ("partial activation: lever(s) {off} of mode {mode} not applied (the run's own switches turned them off) — a mode runs with every one of "
                   "its levers or not at all; --allow-partial (pred) or " + ENV_ALLOW_PARTIAL + "=1 (the " + ENV + " route) runs on the levers that did apply")


def allow_partial_from_env(environ=None) -> bool:
    """``CHAI1_OPT_ALLOW_PARTIAL`` — read on the CHAI1_OPT (in-process) route only (_autoload), never by pred / the driver: unset / ``0`` = no
    (a partial activation is refused by name), ``1`` = yes (recorded, the process proceeds); any other value is a ValueError naming the variable."""
    v = (os.environ if environ is None else environ).get(ENV_ALLOW_PARTIAL)
    if v is None or v == "" or v == "0":
        return False
    if v == "1":
        return True
    raise ValueError(f"{ENV_ALLOW_PARTIAL}={v!r}: expected 1 (a partial activation proceeds, recorded) or 0 / unset (refused by name)")


def partial_reason(rep: dict, allow_partial: bool = False) -> Optional[str]:
    """The one rule for a PARTIAL activation (a lever of the mode the kits' state does not show applied — a lever the run's own switches
    turned off), both routes: a mode is the whole of its lever set, so the activation is REFUSED by name (the returned reason; NOT ACTIVE,
    exit 3) unless the caller opted out explicitly — ``pred --allow-partial`` on the command line (the driver's flag; that route reads no
    environment word for it), ``CHAI1_OPT_ALLOW_PARTIAL=1`` on the CHAI1_OPT (in-process) route, ``enable(allow_partial=True)`` in a caller's
    program. With the opt-out the activation proceeds on the levers that did apply and says so: the report carries ``partial`` = True,
    ``levers_off``, ``partial_note`` and ``allow_partial`` = True; the ACTIVE
    line carries ``PARTIAL off=<levers>`` and one NOTE line names it (``report.print_activation``). The same flag is the opt-out of the memory
    mode's per-item gate at exit (``big.exit_gate``). Returns None when there is nothing to refuse."""
    env_off = [n for n in (rep.get("levers_off_env") or [])]
    if env_off:                                                            # MODEL_OPT_LEVERS_OFF: its levers are off levers of a partial activation, named — and need no
        merged = list(rep.get("levers_off") or [])                         # opt-out (the variable is the explicit request)
        rep["levers_off"] = merged + [n for n in env_off if n not in merged]
        rep["partial"] = True
    if not rep.get("partial"):
        return None
    off = ",".join(rep.get("levers_off") or []) or "?"
    state_off = [n for n in (rep.get("levers_off") or []) if n not in env_off]   # levers the run's own switches turned off (the kits' state): these need the opt-out
    if state_off and not allow_partial:
        return PARTIAL_REFUSAL.format(off=",".join(state_off), mode=rep.get("mode"))
    rep["partial_note"] = (PARTIAL_NOTE.format(off=off) if state_off else
                           PARTIAL_ENV_NOTE.format(env=modes.ENV_LEVERS_OFF, raw=os.environ.get(modes.ENV_LEVERS_OFF, ""), off=off, mode=rep.get("mode")))
    if allow_partial:
        rep["allow_partial"] = True
    return None


def activate(mode: str, *, strict: bool = False, trigger: Optional[str] = None, dry_run: bool = False,
             det: int = 0, route: Optional[str] = None, allow_partial: bool = False, optin=None, n_gpu=None) -> dict:
    """Resolve, gate and (unless dry_run) apply the mode's in-process levers through the kit's own statements, plus the requested opt-in
    levers (``optin``: names or a comma list; modes.OPTIN_LEVERS) at ``n_gpu`` GPUs per fold (None = 1; ngpu.py: P ∈ {1}, a larger P is
    refused by name). See the module doc. A dry run reports the route (``driver``: the whole lever list) unless
    ``route="in-process"`` is asked for."""
    from . import ngpu
    mode = (mode or "").strip().lower()
    route = route or ("driver" if dry_run else "in-process")
    modes.apply_levers_off()                                        # MODEL_OPT_LEVERS_OFF: the rows reduced by its names before anything reads them (implied opt-ins included)
    optin = modes.with_implied(mode, modes.parse_optin(optin))     # the row's implied opt-ins (big: alloc) ride every request
    rep = base_report(mode, trigger, dry_run, det, optin)
    rep["route"] = route
    if mode not in modes.MODES:
        return _refuse(rep, f"unknown mode {mode!r} (expected {'|'.join(modes.MODES)})", strict, dry_run)
    why = modes.levers_off_refusal(mode)                            # a name the mode does not compose, or a lever it cannot switch off on its own: by name, nothing applied
    if why:
        return _refuse(rep, why, strict, dry_run)
    import io
    sink = io.StringIO()
    try:                                                            # statement one: the pin gate (an absent or stale core), then the producers check (an older core) — named, before any opt_core import
        facts = _core.core_gate(stream=sink)
    except SystemExit:
        return _refuse(rep, sink.getvalue().strip().split(" NOT ACTIVE: ", 1)[-1], strict, dry_run)
    refusal = _core.producers_refusal()
    if refusal is not None:
        return _refuse(rep, refusal, strict, dry_run)
    rep["opt_core"] = {"ok": True, "version": facts["installed"]["version"], "pinned": facts["pinned"]["version"]}
    from opt_core.oom import is_oom                                 # an out-of-memory error is re-raised first at every handler below: never a refusal, never a stock route
    why = modes.optin_refusal(mode, optin)
    if why:
        return _refuse(rep, why, strict, dry_run)
    try:
        rep["n_gpu"] = ngpu.resolve(n_gpu, mode)
    except ValueError as e:
        return _refuse(rep, f"usage: {e}", strict, dry_run)
    except Exception as e:  # noqa: BLE001 — opt_core.mem.ngpu.NGpuRefused carries the sentence
        if is_oom(e): raise                                       # noqa: E701
        return _refuse(rep, str(getattr(e, "reason", e)), strict, dry_run)
    prev = _STATE["report"]
    if prev is not None and prev.get("active") and not dry_run:
        if prev["mode"] == mode:
            return prev
        return _refuse(rep, f"mode {prev['mode']} is already active in this process; a process has one mode (levers patch process-wide)",
                       strict, dry_run)
    if mode == "off":
        rep.update(active=False, reason="off: stock — nothing applied, no environment set", levels=None)
        _report.print_activation(rep)
        if not dry_run:
            _STATE["report"] = rep
        return rep
    why = modes.refusal(mode, in_process=(route == "in-process"))
    if why:
        return _refuse(rep, why, strict, dry_run)
    km = modes.kit_mode(mode)
    why = gates(rep, need_gpu=True, use_torch=not dry_run)
    if why:
        return _refuse(rep, why, strict, dry_run)
    rep["levers_not_applicable"] = [] if route == "driver" else [n for n in km.lever_names if n not in km.in_process]
    if km.memory or km.memory_gated:
        from . import big
        why = big.refusal()                                                    # the memory-mode library absent from the core: refused by name
        if why:
            return _refuse(rep, why, strict, dry_run)
        if dry_run:
            rep["big"] = big.dry_run_fields(mode)                            # big's line, or exact / fast's gated levers and their gate
    if dry_run:
        rep.update(active=False, reason=None, levers_applied=(list(km.lever_names) if route == "driver" else list(km.in_process)) + list(optin))
        partial_reason(rep, allow_partial)                                  # MODEL_OPT_LEVERS_OFF named on the dry run too (levers_off, PARTIAL off=…); nothing to refuse here
        _report.print_activation(rep, dry_run=True)
        return rep
    # --- the real thing: the kit's own statements in this process
    _report.register_exit_tally()
    try:
        import torch
        chai1_mod = importlib.import_module(TRIGGER_MODULE)
        esm_mod = importlib.import_module(ESM_MODULE)
    except Exception as e:  # noqa: BLE001
        if is_oom(e): raise                                       # noqa: E701
        return _refuse(rep, f"upstream import failed: {e!r}", strict, dry_run)
    rep["torch_version"] = torch.__version__
    late = late_activation_reason(chai1_mod, esm_mod)
    if late:
        return _refuse(rep, "late activation refused: " + late, strict, dry_run)
    driver_levers = [n for n in km.in_process if n in modes.DRIVER_LEVERS]
    try:
        ns = apply_kit_levers(driver_levers, chai1_mod, esm_mod, torch)
    except Exception as e:  # noqa: BLE001
        if is_oom(e): raise                                       # noqa: E701
        return _refuse(rep, f"the kit's lever statements failed: {e!r}", strict, dry_run)
    _STATE["namespace"] = ns
    from . import forward_timer as _ft
    _ft.install(chai1_mod, _report.PREFIX)                               # the FORWARD line around run_folding_on_context (timing only)
    install_esm_memo_scope(chai1_mod, ns, rep["esm_memo_scope"])         # the W5 memo's scope (gates read and validated the word)
    c = classify(chai1_mod, esm_mod)
    if not [n for n in c["applied"] if n in driver_levers]:
        rep.update(levers_applied=c["applied"], levers_off=[n for n in c["off"] if n in km.in_process], partial=True)
        return _refuse(rep, "no lever shows applied after the kit's statements ran", strict, dry_run)
    try:
        rep["det_applied"] = _det.apply(det, torch)                       # the recipe, at the driver's own placement (after its W1/W5 block)
    except Exception as e:  # noqa: BLE001
        if is_oom(e): raise                                       # noqa: E701
        return _refuse(rep, f"the deterministic recipe (det={det}) failed: {e!r}", strict, dry_run)
    try:
        rep["optin_applied"] = apply_optin_pre(optin, torch)               # tf32 / alloc: after the recipe, before the eager stack installs (its graph capture and first CUDA allocation read them)
    except _alloc.LeverUnavailable as e:
        return _refuse(rep, str(e), strict, dry_run)
    except Exception as e:  # noqa: BLE001
        if is_oom(e): raise                                       # noqa: E701
        return _refuse(rep, f"the opt-in lever(s) {','.join(optin)} failed to apply: {e!r}", strict, dry_run)
    if km.memory or km.memory_gated:
        from . import big
        try:
            big.apply_line(rep, mode=mode, allow_partial=allow_partial, opt_out=f"{ENV_ALLOW_PARTIAL}=1")   # the memory levers through opt_core.mem (big's line; exact / fast's gated levers): a refused lever is named; this route's opt-out word
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise                                       # noqa: E701
            return _refuse(rep, f"big refused — {e}", strict, dry_run)
    if km.eager and km.eager in km.in_process:
        try:
            from . import jit as _jit
            rep["jit"] = _jit.apply()                                                  # the JIT caches keyed under MODEL_OPT_JIT_ROOT (if set) before a lever compiles anything; one NOTE line
            sys.stderr.write(f"{_report.PREFIX} NOTE {_jit.note(rep['jit'])}\n")
            apply_eager(km.eager, dstep=km.dstep, pairtrack=km.pairtrack, mode=km.name)   # the kits' own installs, after the recipe they read (the row as reduced by MODEL_OPT_LEVERS_OFF)
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise                                       # noqa: E701
            return _refuse(rep, f"the eager stack ({km.eager}{'+' + km.dstep_arg if km.dstep else ''}"
                                f"{'+' + ','.join(km.pairtrack) if km.pairtrack else ''}) failed to install: {e!r}", strict, dry_run)
    rep["numerics"] = _precision.numerics(torch)                           # the process numerics signature the recipe and the levers left in force
    c = classify(chai1_mod, esm_mod)
    wanted = tuple(km.in_process) + optin
    rep.update(levers_applied=[n for n in wanted if n in c["applied"]], levers_off=[n for n in c["off"] if n in wanted],   # in the row's own order (the attribute-probed host levers ride after the trunk levers, as in_process lists them)
               partial=bool(set(wanted) - set(c["applied"])))
    missing = [n for n in ((km.eager,) if km.eager else ()) + km.dstep + km.pairtrack if n in km.in_process and n not in c["applied"]]
    if missing:
        return _refuse(rep, f"the eager stack lever(s) {','.join(missing)} do not show installed after the install ran", strict, dry_run)
    missing = [n for n in optin if n not in c["applied"]]
    if missing:
        return _refuse(rep, f"the opt-in lever(s) {','.join(missing)} do not show applied after precision.apply ran", strict, dry_run)
    why = partial_reason(rep, allow_partial)
    if why:
        return _refuse(rep, why, strict, dry_run)
    rep.update(active=True, reason=None)
    warm_imports(rep)                                                        # the core's early library imports, once, before the first fold (in-process route)
    _report.print_activation(rep)
    _STATE["report"] = rep
    return rep


WARM_LIBRARIES = ("torch", "cuequivariance_ops_torch", "cuequivariance_torch")   # the order matters: torch before any library that links against it


def warm_imports(rep=None) -> dict:
    """``opt_core.warm_imports()`` once at start-up — after the allocator / JIT-cache environment is settled and before the first fold: the
    stack's heavy model libraries imported early and shallow through the core's trampoline (idempotent, never raises, no bytes change; a
    library already imported or absent is a word). The activation report gains ``warm_imports`` = the core's per-library words."""
    import opt_core
    words = opt_core.warm_imports(libraries=WARM_LIBRARIES, origin="chai1_opt")      # torch FIRST: a CUDA-extension library imported before torch can fail to load on some wheels
    if rep is not None:
        rep["warm_imports"] = dict(words)
    return words


def status() -> dict:
    rep = _STATE["report"]
    return rep if rep is not None else {"active": False, "reason": "enable() has not run in this process"}


def kit_namespace() -> Optional[dict]:
    """The namespace the kit's statements ran in (LOAD_LOG, ESM_STATS, _MODULE_CACHE ...), for the exit tally."""
    return _STATE["namespace"]


def reset_for_tests() -> None:
    _STATE["report"] = None; _STATE["namespace"] = None; _STATE["eager"] = None
    modes.apply_levers_off()                                               # the mode table as the current environment composes it
    _pairtrack.reset_for_tests()
    from . import featfast as _featfast, postproc as _postproc                 # the host-side levers' process state
    _postproc.reset_for_tests(); _featfast.reset_for_tests()
