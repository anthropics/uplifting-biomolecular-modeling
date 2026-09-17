"""``python -m caliby_opt.design_run --mode M --variant V ... -- <writer args>``: the process that designs.

One process per design pass. Its first statement is the core pin gate (``stack.core_gate``: an absent, older, newer or edited shared
core is one NOT ACTIVE line and exit 3 before anything of the core is imported). It (1) activates the mode here —
``caliby_opt.enable(mode, variant)``: the installed-tree proof, the kits' files proven and loaded by import hook, the exported row, the
activation line, the exit tally — before anything upstream is imported; (2) writes ``opt_manifest.json`` with the
activation report, completed at exit with the levers that fell back (``activate.completion``); (3) runs the tree's design writer, the kit's own ``tests/xcaliby_design.py``, in this same process as a
script (``runpy``, ``__main__``) with the writer arguments after ``--``; the writer makes the upstream calls in its order (clean_pdbs
-> load_model -> seed -> sample / ensemble_sample) and writes its file set (``seq_des_outputs.csv``, ``samples/``,
``cleaned/``, ``timing.json``); this module prints STACK before it and OUTPUTS_WRITTEN after it (report.py: the process's last design output is on disk). Exit: the writer's code;
an activation refusal exits EXIT_NOT_ACTIVE (3), and so does a run in which a lever of the row could not run at call time — a mode is
all of its levers, never a subset under its name (``activate.completion``: ``levers_fallback`` / ``partial`` in the manifest,
``partial=`` on the EXIT line; ``activate.partial_exit`` says NOT ACTIVE by name). This process owns its exit code (``report.exit_owned``):
the environment route's exit gate, which ends a host script's partial run with the same code, stands down here.

The stock arm is not this module: ``stock_design.py`` is the stock caller (a clean subprocess with every kit and package switch
stripped and proven absent), which runs the same writer after its proofs.
"""
from __future__ import annotations

import os
import traceback
import runpy
import sys
from typing import List, Optional

from .stack import core_gate

core_gate()                                                            # the core pin gate before anything of the core (activate / cli -> report import opt_core): a child started by hand refuses like every entry

from . import ActivationError, enable, manifest, stack  # noqa: E402
from . import activate as _activate  # noqa: E402
from . import report as _report  # noqa: E402
from .cli import EXIT_NOT_ACTIVE  # noqa: E402

WRITER_RELPATH = os.path.join("tests", "xcaliby_design.py")


def writer_path() -> str:
    return os.path.join(stack.kit_dir(stack.KIT_ADDON), WRITER_RELPATH)


def run_main(path: str, args: List[str]) -> int:
    """``path`` as ``__main__`` in this process (runpy) with ``sys.argv`` = [path] + args; returns its exit code (``SystemExit``: None -> 0,
    an int -> itself, anything else -> printed, 1; a plain return -> 0; an exception the script raised -> its traceback printed, 1, so that
    the caller's exit rule still runs: the completion record, NOT ACTIVE by name when a lever could not run, the manifest). The one script
    runner: the kit's writer (``run_writer``: design_run, stock_design)."""
    sys.argv = [path] + list(args)
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        sys.stderr.write(str(code) + "\n")
        return 1
    except Exception:                                                  # the script's own failure (a lever's named stop included): printed whole, exit code 1; nothing rerouted or retried
        traceback.print_exc()
        return 1
    return 0


def run_writer(args: List[str]) -> int:
    """The kit's writer as ``__main__`` in this process with ``args``; returns its exit code."""
    return run_main(writer_path(), args)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m caliby_opt.design_run")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--clean_workers", type=int, default=None)
    ap.add_argument("--model_name", default=None, help="the run's checkpoint (the weights step checks its file when it is a pinned name)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("writer_args", nargs=argparse.REMAINDER, help="-- then the writer's arguments")
    a = ap.parse_args(argv)
    wargs = list(a.writer_args)
    if wargs and wargs[0] == "--":
        wargs = wargs[1:]
    try:
        rep = enable(a.mode, a.variant, clean_workers=a.clean_workers, ckpt=a.model_name)
    except ActivationError:
        manifest.write(a.out_dir, None, command="design", argv=sys.argv, exit_code=EXIT_NOT_ACTIVE, extra={"reason": "activation refused (see stderr)"})
        return EXIT_NOT_ACTIVE
    _report.exit_owned()                                               # this process decides its own exit code below (partial_exit / modules_exit); the environment route's exit gate stands down
    manifest.write(a.out_dir, rep, command="design", argv=sys.argv, settings={"writer_args": wargs})
    _report.say(_report.stack_line())                                  # the process's stack facts, once, before the writer runs
    rc = run_writer(wargs)
    if rc == 0:
        _report.outputs_written(a.out_dir)                                      # the writer returned: seq_des_outputs.csv and timing.json, its last writes, are on disk (the background CIF writer joined before run_seq_des returned)
    done = _activate.completion(rep)
    rc = _activate.partial_exit(rc, done, a.mode, EXIT_NOT_ACTIVE)     # a lever that could not run: the mode refuses by name
    rc = _activate.modules_exit(rc, done, EXIT_NOT_ACTIVE)            # fail-closed: a lever module not sourced from its kit file is not an exact run
    manifest.update(a.out_dir, exit_code=rc, **done)
    return rc


if __name__ == "__main__":
    sys.exit(main())
