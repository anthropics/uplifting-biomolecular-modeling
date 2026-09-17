"""nostdin — the verbs run with no standard input: this process's fd 0 off any pipe, every child launched with stdin=/dev/null.

Why: under `--use_template true` upstream aligns each template hit with kalign (opendde/data/tools/kalign.py runs
`subprocess.run(cmd, capture_output=True)`, stdin INHERITED). kalign 3.3.5 reads a standard input that is not a terminal before its
`-i` file, so when the process tree's fd 0 is an open pipe that never reaches EOF — a container runtime's exec channel, `launcher |
run.sh pred …`, a job runner's stdin — template featurization blocks forever at `Building template features`. With fd 0 on /dev/null
kalign returns at once. No verb of this package reads stdin, so the package takes its process tree off the caller's stdin instead of
asking every caller to append `</dev/null`; nothing upstream computes changes.

  detach()       this process (cli.main before any verb runs; opendde_opt.enable — the `OPENDDE_OPT=<mode>` autoload finder's and a library
                 caller's activation — before any lever installs): when fd 0 is not a terminal — a pipe, a file, a socket, closed — it is
                 re-pointed at /dev/null (os.dup2) and sys.stdin rebound, so in-process upstream (the kit lines run the stock CLI in this
                 process) and everything it forks read EOF. A terminal is left alone: kalign's own test is isatty (a tty never triggers the
                 read), and an interactive debugger keeps working. Returns True when it re-pointed fd 0.
  CHILD_STDIN    the `stdin=` of every child the package launches itself (the stock arm's subprocess, warm's pred): subprocess.DEVNULL — explicit at each launch, so a launcher called from a process that never ran
                 detach() (a test, a library caller) is off the caller's stdin too.
"""
from __future__ import annotations

import os
import subprocess
import sys

CHILD_STDIN = subprocess.DEVNULL


def detach() -> bool:
    """Point this process's fd 0 at /dev/null unless it is a terminal; rebind sys.stdin to the new fd 0. True when fd 0 was re-pointed."""
    try:
        if os.isatty(0):
            return False
    except OSError:                                                              # no fd 0 at all: give the process one, on /dev/null
        pass
    fd = os.open(os.devnull, os.O_RDONLY)
    try:
        if fd != 0:
            os.dup2(fd, 0)
    finally:
        if fd != 0:
            os.close(fd)
    try:
        sys.stdin = open(0, "r", closefd=False)                                   # noqa: SIM115 — the process's stdin object for the rest of its life
    except OSError:
        sys.stdin = None
    return True
