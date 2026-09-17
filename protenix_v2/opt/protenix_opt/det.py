"""``--det 1``: the kit's own deterministic recipe, applied on both arms.

The recipe, held here as data:

* ``DET_ENV`` — the two environment names the block exports (``CUBLAS_WORKSPACE_CONFIG=:4096:8 PTX_DET=1``); under ``PTX_DET=1`` the kit's
  ``src/sitecustomize.py`` calls ``torch.use_deterministic_algorithms(True, warn_only=True)`` at interpreter start;
* ``DETREF_FILES`` — the detref scatter copy: ``$FPF_HOME/src/detref/scatter_utils.py`` over the installed ``protenix/utils/scatter_utils.py``
  (backed up first) and ``det_segment_reduce.py`` beside it, both removed / restored when the run ends;
* ``STOCK_PYTHONPATH_REL`` — the stock arm keeps ``$FPF_HOME/src`` on PYTHONPATH and nothing else of the kit (the sitecustomize is inert
  without lever env; its PTX_DET block is the only thing that runs).

``plan()`` names every precondition (detref files present, the installed protenix package and its writable
``utils`` directory, no backup left by an earlier run, the target not already carrying the detref bytes, no stale ``det_segment_reduce.py``);
``Patch.apply()`` performs the copy and ``Patch.restore()`` undoes it (idempotent; registered with atexit as the safety net); ``Patch.record()``
describes what was applied and restored.

The recipe changes no lever: under ``PTX_DET=1`` the kit installs the full lever set of the mode, the graphed sampler included (the
detref scatter keeps no state across calls and builds its gather table inside the captured step), so ``--det 1`` measures exactly the
arm ``--det 0`` ships.
"""
from __future__ import annotations

import atexit
import signal
import importlib.util
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

from . import _core  # noqa: F401
from . import stack
from opt_core import det as _core_det

LEVELS = (0, 1)                                                              # --det: 0 = nothing applied (the default); 1 = the recipe below
DEFAULT = 0
DET_ENV: Dict[str, str] = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PTX_DET": "1"}
RECIPE = _core_det.Recipe(level=1, env=DET_ENV)                              # the recipe's environment as the core's shape (apply_env); its one PYTHONPATH entry is per kit home (stock_exception)
DETREF_RELDIR = os.path.join("src", "detref")                                # $FPF_HOME/src/detref
DETREF_FILES: Tuple[str, ...] = ("scatter_utils.py", "det_segment_reduce.py")   # copied into <protenix package>/utils/; the first replaces a stock file
TARGET_RELDIR = "utils"                                                      # <protenix package>/utils
REPLACED_FILE = "scatter_utils.py"                                           # the stock file the copy replaces: backed up before, restored after
STOCK_PYTHONPATH_REL = "src"                                                 # the stock arm's one kit PYTHONPATH entry: $FPF_HOME/src
BACKUP_SUFFIX = ".protenix_opt_det.orig"                                     # <target>.protenix_opt_det.orig: the backup of the replaced stock file
DET_SWITCH = "PTX_DET"                                                       # the recipe's switch the kit keys on; the package keys on the same one


def is_det(environ=None) -> bool:
    """Whether the recipe is on in ``environ`` (the process environment by default): ``PTX_DET=1``."""
    environ = os.environ if environ is None else environ
    return environ.get(DET_SWITCH, "0") == "1"


# ---------------------------------------------------------------------------------------------------------------- paths ----
def detref_dir(fpf_home: Optional[str] = None) -> str:
    return os.path.join(fpf_home or stack.kit_home(), DETREF_RELDIR)


def stock_pythonpath(fpf_home: Optional[str] = None) -> str:
    """The one kit entry the stock arm keeps on PYTHONPATH under the recipe: $FPF_HOME/src."""
    return os.path.join(fpf_home or stack.kit_home(), STOCK_PYTHONPATH_REL)


def protenix_package_dir() -> Optional[str]:
    """The installed ``protenix`` package directory (importlib's spec: the package is located, not imported), or None when absent."""
    try:
        spec = importlib.util.find_spec("protenix")
    except (ImportError, ValueError):                                     # a broken namespace/parent on sys.path: counted as absent
        return None
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(os.path.abspath(spec.origin))


def dir_writable(path: str) -> bool:
    """Whether this process can create a file in ``path`` (probed, not inferred from mode bits)."""
    try:
        fd, tmp = tempfile.mkstemp(prefix=".protenix_opt_det_probe_", dir=path)
    except OSError:
        return False
    os.close(fd)
    os.unlink(tmp)
    return True


# ----------------------------------------------------------------------------------------------------------------- plan ----
def plan(fpf_home: Optional[str] = None, package_dir: Optional[str] = None) -> dict:
    """What ``--det 1`` would do here and every reason it cannot: ``{"env", "patch": [{"source", "target", "backup", "sha256_source",
    "sha256_target_before", "replaces"}], "stock_pythonpath", "package_dir", "problems": [...]}``."""
    h = fpf_home or stack.kit_home()
    out = {"level": 1, "env": dict(DET_ENV), "patch": [], "stock_pythonpath": stock_pythonpath(h), "package_dir": None, "problems": []}
    from . import kits
    rels = [kits.kit_rel("flashpairformer", f"{DETREF_RELDIR}/{f}") for f in DETREF_FILES]
    problems = kits.missing_kit_files(rels)
    out["problems"] += [f"detref {p}" for p in problems]
    pkg = package_dir if package_dir is not None else protenix_package_dir()
    out["package_dir"] = pkg
    if pkg is None:
        out["problems"].append("protenix package not installed: no target for the detref copy (<protenix package>/utils/)")
    target_dir = os.path.join(pkg, TARGET_RELDIR) if pkg else None
    if target_dir and not os.path.isdir(target_dir):
        out["problems"].append(f"protenix package has no {TARGET_RELDIR}/ directory at {pkg}: not the pinned protenix layout")
        target_dir = None
    if target_dir and not dir_writable(target_dir):
        out["problems"].append(f"{target_dir} is not writable by this process: the detref copy cannot be placed (and restored)")
    for f in DETREF_FILES:
        src = os.path.join(detref_dir(h), f)
        tgt = os.path.join(target_dir, f) if target_dir else None
        entry = {"source": src, "target": tgt, "backup": (tgt + BACKUP_SUFFIX) if (tgt and f == REPLACED_FILE) else None,
                 "sha256_source": stack.sha256_file(src) if os.path.isfile(src) else None, "sha256_target_before": None, "replaces": f == REPLACED_FILE}
        if tgt and os.path.exists(tgt):
            entry["sha256_target_before"] = stack.sha256_file(tgt)
        if tgt and f == REPLACED_FILE:
            if not os.path.isfile(tgt):
                out["problems"].append(f"{tgt}: the stock file the copy replaces is missing: not the pinned protenix layout")
            elif entry["sha256_target_before"] == entry["sha256_source"]:
                out["problems"].append(f"{tgt}: already carries the detref bytes (an earlier run did not restore it): restore the stock file by hand "
                                       f"({entry['backup']} if present) before --det 1 runs again")
            if os.path.exists(entry["backup"]):
                out["problems"].append(f"{entry['backup']}: a backup from an earlier run is present (never overwritten): restore it by hand "
                                       f"(move it back over {tgt}) before --det 1 runs again")
        elif tgt and os.path.exists(tgt):
            out["problems"].append(f"{tgt}: present before the copy (a stale det file from an earlier run): remove it by hand before --det 1 runs again")
        out["patch"].append(entry)
    return out


# ---------------------------------------------------------------------------------------------------------------- patch ----
class DetError(RuntimeError):
    """The recipe cannot be applied or restored; the message names every reason."""


class Patch:
    """The applied detref copy: ``apply()`` once, ``restore()`` any number of times (the first one restores, the rest are no-ops)."""

    def __init__(self, plan_: dict):
        if plan_["problems"]:
            raise DetError("--det 1 refused: " + "; ".join(plan_["problems"]))
        self.plan = plan_
        self.applied = False
        self.restored = False
        self.entries: List[dict] = []
        self.restore_problems: List[str] = []

    def apply(self) -> "Patch":
        for e in self.plan["patch"]:
            rec = dict(e, sha256_after=None, restored=False)
            if e["replaces"]:
                if os.path.exists(e["backup"]):                               # re-checked at apply time: never overwrite a backup
                    raise DetError(f"--det 1 refused: {e['backup']} appeared before the copy (another run in progress?)")
                shutil.copy2(e["target"], e["backup"])
            shutil.copy2(e["source"], e["target"])
            rec["sha256_after"] = stack.sha256_file(e["target"])
            if rec["sha256_after"] != e["sha256_source"]:
                self.entries.append(rec); self.applied = True
                self.restore()
                raise DetError(f"--det 1: {e['target']} is not the detref bytes after the copy ({rec['sha256_after'][:16]}… != {e['sha256_source'][:16]}…)")
            self.entries.append(rec)
        self.applied = True
        atexit.register(self.restore)                                         # the safety net: a crash after apply() still restores
        return self

    def restore(self) -> None:
        if not self.applied or self.restored:
            return
        for rec in self.entries:
            try:
                if rec["replaces"]:
                    shutil.copy2(rec["backup"], rec["target"])
                    got = stack.sha256_file(rec["target"])
                    if got != rec["sha256_target_before"]:
                        self.restore_problems.append(f"{rec['target']}: restored bytes {got[:16]}… are not the original {str(rec['sha256_target_before'])[:16]}… "
                                                     f"(backup kept at {rec['backup']})")
                        continue
                    os.unlink(rec["backup"])
                else:
                    if os.path.exists(rec["target"]):
                        os.unlink(rec["target"])
                rec["restored"] = True
            except OSError as e:                                               # an unwritable or vanished target: named, never silent
                self.restore_problems.append(f"{rec['target']}: restore failed: {e!r}")
        self.restored = True

    def record(self, stock_exception: Optional[dict] = None) -> dict:
        """What level 1 applied: the environment, the package directory, every file's source / target / backup / digests / restored flag, the restore problems."""
        return {"level": 1, "env": dict(self.plan["env"]), "package_dir": self.plan["package_dir"],
                "patch": [{k: rec[k] for k in ("source", "target", "backup", "sha256_before", "sha256_after", "restored")}
                          for rec in ({**r, "sha256_before": r["sha256_target_before"]} for r in self.entries)],
                "restore_problems": list(self.restore_problems), "stock_exception": stock_exception}


def apply_env(environ=None) -> Dict[str, Optional[str]]:
    """Export ``DET_ENV`` into ``environ`` (the process environment by default); returns the values replaced (None = was unset)."""
    return _core_det.apply_env(RECIPE, environ)


def stock_exception(fpf_home: Optional[str] = None) -> dict:
    """The one named exception to the stock env-clean rule under the recipe: the det names present, ``$FPF_HOME/src`` the only kit entry
    on the path, the kit's sitecustomize the process's (inert without lever env)."""
    h = fpf_home or stack.kit_home()
    src = stock_pythonpath(h)
    return dict(_core_det.stock_exception(_core_det.Recipe(level=1, env=DET_ENV, pythonpath=(src,))),
                sitecustomize=os.path.join(src, "sitecustomize.py"), modules=["sitecustomize"])


# ---------------------------------------------------------------------------------------------------------------- lines ----
def lines(plan_: dict, prefix: str) -> List[str]:
    """The lines ``check --det 1`` prints: the env names, each detref copy (source sha, target, backup), the package dir; ok or the problems."""
    out = [f"{prefix} DET env {' '.join(f'{k}={v}' for k, v in plan_['env'].items())}; stock arm PYTHONPATH={plan_['stock_pythonpath']}"]
    for e in plan_["patch"]:
        src_sha = (e["sha256_source"] or "absent")[:16]
        out.append(f"{prefix} DET copy {e['source']} ({src_sha}…) -> {e['target'] or '<no protenix package>'}"
                   + (f" (backup {e['backup']})" if e["backup"] else " (removed at the end)"))
    out.append(f"{prefix} DET protenix package {plan_['package_dir'] or 'NOT INSTALLED'}")
    if plan_["problems"]:
        out.append(f"{prefix} DET PROBLEMS {len(plan_['problems'])}")
        out += [f"{prefix} DET   {p}" for p in plan_["problems"]]
    else:
        out.append(f"{prefix} DET ok: detref present, target writable, no backup or stale copy present")
    return out


# ------------------------------------------------------------------------------------------------------------ signal guard ----
GUARDED_SIGNALS = ("SIGTERM", "SIGHUP", "SIGINT")                           # a termination while the detref copy is in place must unwind, not vanish


class Terminated(SystemExit):
    """Raised from the guard's handler: a SystemExit (code 128 + signo) so every ``finally`` and the atexit safety net run — the detref copy
    is restored on the way out instead of being left over the stock file (the next ``--det 1`` would then refuse by name until restored by hand)."""


def guard_signals(patch: Optional["Patch"]) -> dict:
    """While a Patch is applied: TERM / HUP / INT raise ``Terminated`` in the main thread (Python unwinds: cli's finally -> Patch.restore()).
    Returns the previous handlers for ``unguard_signals``; a no-op ({}) without a patch or off the main thread."""
    prev: dict = {}
    if patch is None:
        return prev
    for name in GUARDED_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            prev[sig] = signal.signal(sig, _raise_terminated)
        except (ValueError, OSError):                                       # not the main thread / not settable here: the atexit net remains
            pass
    return prev


def unguard_signals(prev: dict) -> None:
    for sig, h in (prev or {}).items():
        try:
            signal.signal(sig, h if h is not None else signal.SIG_DFL)
        except (ValueError, OSError):
            pass


def _raise_terminated(signo, _frame):
    raise Terminated(128 + int(signo))
