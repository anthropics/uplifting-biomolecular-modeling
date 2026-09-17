"""warm — one design of the smallest public example structure in the mode, through ``design`` (cli.run_design) at upstream's
settings, one sequence and seed 0, so that the one cost a box keeps across processes is paid before a measured run: the Triton JIT of the fused LCP
kernel (``CALIBY_X_LCP=1``, into the persistent cache directory configs/<gpu>.env keys by the running stack). What ``design`` pays per
invocation recurs in every invocation, this one included: in its design process the activation proofs, the model load and the
sampler's CUDA-graph capture (``CALIBY_FAST_SAMPLER=2``). The outputs go to a scratch
directory under the state directory and are removed on success unless ``keep`` or an ``out_dir`` is given; ``opt_manifest.json``
there records the run as ``command: warm``. Exit code: the design's (EXIT_NOT_ACTIVE when a lever of the row could not run: the
design process's own NOT ACTIVE line first, then this module's; the scratch directory is kept — its manifest is the record).
"""
from __future__ import annotations

import os
import shutil
import time
from typing import Callable, Optional

from . import inputs as _inputs, manifest, report as _report, stack

WARM_SEED = 0                                                          # the warm design's seed (upstream's YAML default)
WARM_NUM_SEQS = 1                                                      # one sequence of the smallest public input pays the JIT


def run(mode: str, variant: str, design: Callable[..., int], *, out_dir: Optional[str] = None, keep: bool = False) -> int:
    files = [_inputs.smallest_public_input()]
    scratch = out_dir or os.path.join(stack.state_dir(), "warm", f"{mode}-{variant}-{int(time.time())}")
    rc = design(mode, variant, files, scratch, seed=WARM_SEED, num_seqs_per_pdb=WARM_NUM_SEQS, command="warm")
    man = manifest.read(scratch) if os.path.isfile(os.path.join(scratch, manifest.FILENAME)) else {}
    if man.get("partial"):
        _report.say(f"{_report.PREFIX} warm: a lever of the row could not run (levers_fallback={','.join(man.get('levers_fallback') or [])}) "
                    f"rc={rc}; outputs kept at {scratch}")
    elif not out_dir and rc == 0 and not keep:
        shutil.rmtree(scratch, ignore_errors=True)
    return rc
