"""The tree's one stock caller: ``python -s -m caliby_opt.stock_design --variant V --out_dir OUT -- <writer args>``.

"Stock" here is the upstream Python API on the pinned upstream tree with nothing from the kits on the path, proven rather than
assumed. Run by ``caliby-opt design --mode off`` (and ``warm --mode off``) in a clean subprocess — every variable starting with
CALIBY_FAST_ / CALIBY_X_ / CALIBY_OPT / CALIBY_VARIANT stripped, the tree reached through MODEL_OPT — this process, after the core pin
gate (``stack.core_gate``, its first statement: an absent, older, newer or edited shared core is one NOT ACTIVE line and exit 3 before
anything of the core is imported):

  1. proves its environment: no kit or package switch present (``stack.stock_env_proof``; the data-path names upstream reads —
     MODEL_PARAMS_DIR, HF_HUB_OFFLINE, PDB_MIRROR_PATH, CCD_MIRROR_PATH — are recorded, never stripped);
  2. proves the installed tree: ``stack.tree_state() == "stock"`` and the kit's tree digest equals stock/PINS.json
     "tree_digest_upstream" (``enable("off")`` does both; a patched tree is the ``NOT STOCK`` refusal, exit 3, with the reinstall
     command) — so the modules the writer imports are the installed upstream files, no import hook installed, nothing served over them;
  3. writes ``stock_env_proof.json`` and ``opt_manifest.json`` in the output directory;
  4. prints the STACK line and runs the design writer — the kit's ``tests/xcaliby_design.py`` on the upstream API, in this process,
     with the writer arguments given (settings.py) — the same writer and file set as the exact arm, so the arms differ in the mode alone;
     prints OUTPUTS_WRITTEN once the writer returned (its seq_des_outputs.csv and timing.json are its last writes).

The writer is a kit file; it reads the switches only to echo them into ``timing.json``, which under this caller records every switch empty.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

from .stack import core_gate

core_gate()                                                            # the core pin gate before anything of the core (activate / report import opt_core): a child started by hand refuses like every entry

from . import ActivationError, enable, manifest, stack  # noqa: E402
from . import activate as _activate  # noqa: E402
from . import report as _report  # noqa: E402
from .design_run import run_writer  # noqa: E402
from .report import EXIT_NOT_ACTIVE  # noqa: E402

PROOF_NAME = "stock_env_proof.json"


def prove() -> dict:
    """The environment proof of a stock process (``stack.stock_env_proof`` plus what this interpreter can say about itself): no kit or
    package switch present, the data-path names read, TQDM_DISABLE as found (``1`` under the stock caller), the user site off, no kit
    directory on ``sys.path``."""
    proof = stack.stock_env_proof()
    proof["tqdm_disable"] = os.environ.get("TQDM_DISABLE")
    proof["sys_flags"] = {"no_user_site": bool(sys.flags.no_user_site), "argv0": sys.argv[0]}
    proof["kit_dirs_on_sys_path"] = [p for p in sys.path if os.sep + "opt" + os.sep + "forward" + os.sep in p]
    return proof


def write_proof(out_dir: str, proof: dict) -> str:
    path = os.path.join(out_dir, PROOF_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(proof, fh, indent=1)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -s -m caliby_opt.stock_design")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("writer_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    wargs = list(a.writer_args)
    if wargs and wargs[0] == "--":
        wargs = wargs[1:]
    os.makedirs(a.out_dir, exist_ok=True)
    proof = prove()
    try:
        rep = enable("off", a.variant)
    except ActivationError as e:
        proof["verdict"] = f"REFUSED: {e}"
        write_proof(a.out_dir, proof)
        manifest.write(a.out_dir, None, command="design", argv=sys.argv, exit_code=EXIT_NOT_ACTIVE, extra={"reason": str(e)})
        return EXIT_NOT_ACTIVE
    proof["tree_digest"] = rep.get("tree_digest")
    proof["tree_state"] = rep.get("tree_state")
    proof["verdict"] = "STOCK"
    write_proof(a.out_dir, proof)
    _report.exit_owned()                                               # this process decides its own exit code below (modules_exit)
    manifest.write(a.out_dir, rep, command="design", argv=sys.argv, settings={"writer_args": wargs})
    _report.say(_report.stack_line())                                  # the process's stack facts, once, before the writer runs
    rc = run_writer(wargs)
    if rc == 0:
        _report.outputs_written(a.out_dir)                             # the writer returned: seq_des_outputs.csv and timing.json, its last writes, are on disk
    done = _activate.completion(rep)
    rc = _activate.modules_exit(rc, done, EXIT_NOT_ACTIVE)             # fail-closed: a kit file loaded in a stock process is not a stock run
    manifest.update(a.out_dir, exit_code=rc, modules_state=done["modules_state"], modules_wrong=done["modules_wrong"])
    return rc


if __name__ == "__main__":
    sys.exit(main())
