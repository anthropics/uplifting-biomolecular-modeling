"""Interpreter-start hook behind ``genie3_opt_autoload.pth`` — the env route ``GENIE3_OPT=<mode> genie3 generate …``.

The ``.pth`` file is GENERATED at build / install time by the in-tree build backend (``opt/_build_backend.py``, declared in
``pyproject.toml``) and its one line is exactly ``import genie3_opt._autoload``: `site.py` runs it in every interpreter that starts
with the package installed.  What the hook does at start-up is bounded and cheap: read ``GENIE3_OPT`` and ``GENIE3_OPT_AUTOLOAD``;
when ``GENIE3_OPT`` is set, put a one-shot `importlib` meta-path finder in front of ``sys.meta_path``.  Nothing of torch, genie3
or the rest of this package is imported at start-up.

When the process later imports top-level ``genie3`` the finder fires ONCE and removes itself.  ``GENIE3_OPT=off`` (or empty): the
import proceeds, the process is stock, untouched.  A kit mode named (``exact`` / ``fast`` / any other word: the design verb resolves
it against the mode table) is SERVED OR REFUSED BY NAME — the stock import never proceeds under a kit mode's name:

* the process is upstream's console script at its ``generate`` command (``genie3 generate -c <experiment.yaml> [--verbose] [--log-dir D]
  [--num-devices N] [--shard-id K] [--num-shards M]``, src/genie3/cli.py): the kit's own ``design`` verb carries the mode — one line
  ``[genie3-opt] NOTE env-route: …`` names the hand-over and the process is REPLACED (`os.execv`) by ``python -m genie3_opt design
  --mode <m> <the same generate arguments>`` (`design_argv`: the design verb accepts upstream's generate flags under their own
  spellings).  From there everything is the design verb's: the shared core's pin gate (statement one of `python -m genie3_opt`), the
  activation report, the refusals by name (exit 3), the outputs in upstream's layout plus the kit's manifest, the exit code;
* any other importer of ``genie3`` under a kit mode (upstream's ``run`` / ``evaluate`` / ``train`` / ``status`` commands, ``python -m
  genie3.cli …``, a tool that imports the library): one line ``[genie3-opt] NOT ACTIVE: env-route: GENIE3_OPT=<mode> names a kit mode at
  `<command>` … refused (exit 3)`` naming the two honest routes — ``GENIE3_OPT=off`` runs it as stock, ``GENIE3_OPT_AUTOLOAD=0`` disarms
  the hook for a tool that imports genie3 on purpose — and the process ENDS with exit code 3 (codes.EXIT_NOT_ACTIVE) before anything of
  upstream runs (`SystemExit` raised through the importer's ``import genie3``).

``GENIE3_OPT_AUTOLOAD=0`` disarms the hook entirely (no finder, whatever ``GENIE3_OPT`` says).
"""
from __future__ import annotations

import importlib.abc
import os
import sys
from typing import List, Optional

ENV_MODE, ENV_DISARM = "GENIE3_OPT", "GENIE3_OPT_AUTOLOAD"     # nothing of the package is imported here: at interpreter start sys.modules holds the top module and this one only
UPSTREAM = "genie3"
UPSTREAM_SCRIPT, UPSTREAM_GENERATE = "genie3", "generate"      # upstream's console script (pyproject [project.scripts]: genie3 = genie3.cli:main) and the one command a kit mode serves
PATH_FLAGS = ("-c", "--config", "--log-dir")                    # upstream's path-valued generate flags (src/genie3/cli.py): resolved against the caller's cwd at the hand-over
FINDER = None


def design_argv(mode: str, argv: List[str], python: Optional[str] = None) -> Optional[List[str]]:
    """``<bin>/genie3 generate <args…>`` → ``[python, "-m", "genie3_opt", "design", "--mode", <mode>, <args…>]``: the kit's design verb
    carrying the mode over the same arguments (upstream's generate flags ``-c/--config``, ``--verbose``, ``--log-dir``, ``--num-devices``,
    ``--shard-id``, ``--num-shards`` are the design verb's own spellings); None when ``argv`` is not upstream's console script at its
    generate command."""
    if len(argv) < 2 or os.path.basename(argv[0]) != UPSTREAM_SCRIPT or argv[1] != UPSTREAM_GENERATE:
        return None
    rest, path_next = [], False
    for a in argv[2:]:                                                        # the request file and the log directory are found where the caller points: made absolute against the
        if path_next:                                                         # caller's cwd before the hand-over (paths INSIDE the request resolve against GENIE3_ROOT, the cwd of every kit line)
            a, path_next = os.path.abspath(a), False
        elif a in PATH_FLAGS:
            path_next = True
        elif any(a.startswith(f + "=") for f in PATH_FLAGS if f.startswith("--")):
            f, v = a.split("=", 1)
            a = f + "=" + os.path.abspath(v)
        rest.append(a)
    return [python or sys.executable, "-m", "genie3_opt", "design", "--mode", mode] + rest


def importer_words(argv: List[str]) -> str:
    """What imported genie3, in the caller's terms: ``genie3 evaluate`` (a console script and its command), ``python -c …`` / ``python -m <module>``
    (the interpreter's own spellings of argv[0]), ``python <script>``, or an embedded interpreter."""
    if not argv or not argv[0]:
        return "an embedded interpreter importing genie3"
    head = argv[0]
    if head == "-c":
        return "python -c …"
    if head == "-m":
        return "python -m " + (argv[1] if len(argv) > 1 else "<module>")
    base = os.path.basename(head)
    if base.endswith(".py"):
        return "python " + base
    return " ".join([base] + list(argv[1:2]))


def refused_line(mode_word: str, argv: List[str]) -> str:
    """The env route's refusal by name: what ``GENIE3_OPT`` is set to, at which importer, and the two honest routes."""
    from .codes import EXIT_NOT_ACTIVE, TAG                                    # loaded at the trigger, not at interpreter start
    return (f"[{TAG}] NOT ACTIVE: env-route: {ENV_MODE}={mode_word!r} is set at `{importer_words(argv)}` — a kit mode serves upstream's "
            f"`{UPSTREAM_SCRIPT} {UPSTREAM_GENERATE} …` (carried by `genie3-opt design --mode {mode_word.strip().lower()}`) and nothing else; "
            f"refused (exit {EXIT_NOT_ACTIVE}), nothing of upstream ran: run it with {ENV_MODE}=off for the stock path, or {ENV_DISARM}=0 to leave "
            f"a tool that imports genie3 untouched")


class _Finder(importlib.abc.MetaPathFinder):
    """Wakes on the first `import genie3`, removes itself, and applies the env-route rule: off = stock untouched; a kit mode = the design
    verb by `os.execv` at `genie3 generate`, else refused by name and the process ends (exit 3)."""

    def __init__(self, mode: str):
        self.mode = mode
        self.fired = False

    def find_spec(self, fullname, path=None, target=None):
        if self.fired or not (fullname == UPSTREAM or fullname.startswith(UPSTREAM + ".")):
            return None
        self.fired = True
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        m = self.mode.strip().lower()
        if m in ("", "off"):
            return None
        from .codes import EXIT_NOT_ACTIVE, TAG                                # the line tag and the exit code, read only when a kit mode is named and the upstream import wakes the finder
        cmd = design_argv(m, list(sys.argv))
        if cmd is not None:                                                   # upstream's `genie3 generate …`: the kit's design verb carries the mode over the same arguments — this process becomes it
            sys.stderr.write(f"[{TAG}] NOTE env-route: {ENV_MODE}={self.mode!r} at `{UPSTREAM_SCRIPT} {UPSTREAM_GENERATE}` — carried by the kit's design verb: {' '.join(cmd[1:])} "
                             f"(paths inside the request resolve against GENIE3_ROOT, the cwd of every kit line)\n")
            sys.stderr.flush()
            try:
                os.execv(cmd[0], cmd)
            except OSError as e:                                              # the interpreter could not be re-executed: refused by name — never the stock import under the mode's name
                sys.stderr.write(f"[{TAG}] NOT ACTIVE: env-route: the design verb could not start ({e}); refused (exit {EXIT_NOT_ACTIVE}), nothing of upstream ran\n")
                sys.stderr.flush()
                raise SystemExit(EXIT_NOT_ACTIVE)
        sys.stderr.write(refused_line(self.mode, list(sys.argv)) + "\n")
        sys.stderr.flush()
        raise SystemExit(EXIT_NOT_ACTIVE)                                     # through the importer's `import genie3`: the process ends 3 before anything of upstream runs


def install(environ=None) -> bool:
    """Install the finder when GENIE3_OPT is set to anything and the disarm variable is not 0. Returns whether it was installed."""
    global FINDER
    environ = os.environ if environ is None else environ
    mode = environ.get(ENV_MODE)
    if mode is None or environ.get(ENV_DISARM) == "0" or FINDER is not None:
        return False
    FINDER = _Finder(mode)
    sys.meta_path.insert(0, FINDER)
    return True


def uninstall() -> None:
    global FINDER
    if FINDER is not None:
        try:
            sys.meta_path.remove(FINDER)
        except ValueError:
            pass
        FINDER = None


install()
