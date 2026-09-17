"""The gate of the ENV route: run by run.sh only when the mode came from ``BOLTZ2_OPT`` (no ``--mode`` on the command line) and
names a mode other than off. That route is only as real as the kit's hook — ``boltz2_opt_autoload.pth`` processed by site.py at interpreter
start — because the same hook is what makes ``BOLTZ2_OPT=<mode>`` REFUSE by name in every stock ``boltz`` process on the box (opt/boltz2_opt/
_autoload.py): with the package merely importable (PYTHONPATH, a checkout) and the hook not live, such a process runs STOCK silently. The check
is HOOK-LIVE, not file presence: this script runs as a fresh ``python`` interpreter (the route's own form: no -I, so the user site and PYTHONPATH count exactly as they do for the route) and asks whether ``boltz2_opt._autoload`` is already in
``sys.modules`` when it starts (the .pth's effect) and whether that module's bytes are this tree's; the file search is the DIAGNOSTIC of the
refusal line (present but not processed / absent from the searched sites / a stale copy). Stdlib only; run by path (no package import).
The ``--mode`` route is not gated: run.sh imports the package directly and its ACTIVE line is the evidence.
"""
import hashlib
import os
import site
import sys

PTH = "boltz2_opt_autoload.pth"
HOOK = "boltz2_opt._autoload"
EXIT_NOT_ACTIVE = 3
TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sha(path):   # private on purpose: this gate runs at interpreter start from the .pth, before any mode is named — importing opt_core here would break the no-core-at-start rule (_autoload, test_autoload)
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def site_dirs():
    dirs = list(site.getsitepackages()) if hasattr(site, "getsitepackages") else []
    user = site.getusersitepackages() if hasattr(site, "getusersitepackages") else None
    return [d for d in dirs if os.path.isdir(d)], (user if user and os.path.isdir(user) else None)   # the user site is searched whether or not it is enabled: the diagnostic says which


def diagnostic():
    """Where a copy of the .pth sits, if anywhere: a processed site dir (then its line did not import the hook), the user site while it is
    disabled (PYTHONNOUSERSITE / -s / a venv: site.py did not read it; enabled, a user-site install is processed and the hook is live), or
    beside a PYTHONPATH / sys.path entry (site.py processes no .pth there)."""
    dirs, user = site_dirs()
    for d in dirs:
        if os.path.isfile(os.path.join(d, PTH)):
            return f"present at {os.path.join(d, PTH)} but not processed (site.py read it and the hook did not import: a broken line, or site disabled)"
    if user and os.path.isfile(os.path.join(user, PTH)):
        return (f"present at {os.path.join(user, PTH)} (the user site) but not processed: " + ("site.py read it and the hook did not import (a broken line)" if site.ENABLE_USER_SITE else "the user site is disabled in this interpreter (PYTHONNOUSERSITE, -s, or a venv)"))
    for p in [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x] + [x for x in sys.path if x]:
        if os.path.isfile(os.path.join(p, PTH)):
            return f"present at {os.path.join(p, PTH)} beside a sys.path/PYTHONPATH entry — site.py processes no .pth file there (a copy on PYTHONPATH proves nothing)"
    return f"absent from the searched sites ({os.pathsep.join(dirs + ([user] if user else [])) or 'no site directory'})"


def main() -> int:
    mod = sys.modules.get(HOOK)
    if mod is not None:
        ours = os.path.join(TREE, "opt", "boltz2_opt", "_autoload.py"); theirs = getattr(mod, "__file__", None)
        if theirs and os.path.isfile(theirs) and _sha(theirs) == _sha(ours):
            return 0
        why = f"a stale copy: the hook loaded from {theirs} (sha256 {_sha(theirs)[:12] if theirs and os.path.isfile(theirs) else '?'}), not this tree's {ours} ({_sha(ours)[:12]})"
    else:
        why = diagnostic()
    sys.stderr.write(f"run.sh: NOT ACTIVE (RF-6): the kit's hook {HOOK} is not live in {sys.executable} — {why}; the env route BOLTZ2_OPT=<mode> "
                     f"runs stock silently in a stock process without it; install the package into that interpreter's site: {TREE}/run.sh install\n")
    return EXIT_NOT_ACTIVE


if __name__ == "__main__":
    sys.exit(main())
