"""install — the kit's own installer, run in place: `bash opt/forward/xattempt_addon/install.sh <flag>` in a chosen interpreter.

Every state change of the ten target files goes through the kit's `install.sh` (apply, --rebind, --uninstall); the package never copies a kit
file. `install.sh` locates site-packages with the `python` on PATH (install.sh:16), so each call runs with the interpreter's bin
directory first on PATH and with the package switch and every lever name stripped from the environment (the installer's own
`import rfd3` must not trigger this package's autoload). Exit codes are the installer's: 0 ok, 2 no rfd3 (the command line names it in
one line before the installer runs: `[rfdiffusion3-opt] install <action>: rfd3 is not importable in <python> …`), 3 sha mismatch after
install, 4 refused (a file in an unknown state; the package never passes --force). `classify` prints the installer's one
`OVERLAY_PRESTATE: pristine|current|previous:<id>|foreign:<files>` line; `rebind` is its start-up verb for a fresh environment (that line, then on pristine / previous
the overlay of this tree installed and `OVERLAY_APPLIED: 10 files id=<version>/<overlay id>`; foreign: exit 4, nothing touched).

    python -m rfdiffusion3_opt.install status|classify|rebind|apply|check-only|uninstall [--python <interpreter>]
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Optional

from . import registry

FLAGS = {"status": "--status", "classify": "--classify", "rebind": "--rebind", "apply": "apply", "check-only": "--check-only", "uninstall": "--uninstall"}
EXIT_NO_RFD3 = 2                                                          # the installer's own code for an interpreter without rfd3
EXIT_REFUSED = 4


class InstallRefused(RuntimeError):
    """install.sh exit 4: a target file is in a state the kit does not know (never overridden with --force by the package)."""


def _env(python: str) -> Dict[str, str]:
    from . import stack
    env = {k: v for k, v in os.environ.items() if k not in stack.PACKAGE_ENV and k not in stack.lever_names()}
    env["PATH"] = os.path.dirname(os.path.abspath(python)) + os.pathsep + env.get("PATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run(action: str, python: Optional[str] = None, kit: Optional[str] = None) -> subprocess.CompletedProcess:
    if action not in FLAGS:
        raise ValueError(f"action must be one of {'|'.join(FLAGS)}")
    kit = kit or registry.kit_home()
    script = os.path.join(kit, "install.sh")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"the kit installer is missing: {script}")
    python = python or sys.executable
    cmd: List[str] = ["bash", script, FLAGS[action]]
    return subprocess.run(cmd, env=_env(python), cwd=kit, capture_output=True, text=True)


def parse_status(text: str) -> Dict[str, str]:
    """`--status` output (`site-packages: <dir>` then `  <state>  <file>` lines) -> {file: state} (+ "site-packages")."""
    out = {}
    for line in text.splitlines():
        if line.startswith("site-packages: "):
            out["site-packages"] = line.split(": ", 1)[1].strip()
        elif line.startswith("  "):
            parts = line.split()
            if len(parts) == 2:
                out[parts[1]] = parts[0]
    return out


def apply(python: Optional[str] = None, kit: Optional[str] = None) -> str:
    r = run("apply", python, kit)
    if r.returncode == EXIT_REFUSED:
        raise InstallRefused((r.stdout + r.stderr).strip())
    if r.returncode != 0:
        raise RuntimeError(f"install.sh exited {r.returncode}: {(r.stdout + r.stderr).strip()[-600:]}")
    return r.stdout.strip()


def main(argv: Optional[List[str]] = None) -> int:
    from ._autoload import core_gate
    core_gate()                                              # the core pin gate (every action reads stack, which imports the core): NOT ACTIVE + exit 3 by name
    import argparse
    ap = argparse.ArgumentParser(prog="python -m rfdiffusion3_opt.install", description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=sorted(FLAGS))
    ap.add_argument("--python", default=None, help="the interpreter whose rfd3 tree is meant (default: this one)")
    a = ap.parse_args(argv)
    python = a.python or sys.executable
    probe = subprocess.run([python, "-c", "import rfd3"], env=_env(python), capture_output=True, text=True)   # the installer's own first step (install.sh:16), named here instead of its traceback
    if probe.returncode != 0:
        from ._autoload import TAG
        cause = ([ln for ln in probe.stderr.splitlines() if ln.strip()] or ["exit %d" % probe.returncode])[-1]
        from ._emit import emit
        emit(f"[{TAG}] install {a.action}: rfd3 is not importable in {python} ({cause}): the kit installs over the pinned upstream package "
             f"(stock/PINS.json), which this interpreter does not have; exit {EXIT_NO_RFD3}")
        return EXIT_NO_RFD3
    r = run(a.action, python)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
