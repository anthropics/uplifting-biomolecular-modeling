"""`CALIBY_FAST_SAMPLER`'s served-call predicate (lever code in the kit's `fast/potts.py`, `sample_potts`): the switch word, the ONE
predicate that decides whether a sampler call is the fast sampler's (levels 1 / 2) or upstream's sweep loop, and the stock-sampler line
(counted once per line by the report's stderr probe — ``report.FALLBACK_MARKERS`` — the one census of the event). `caliby_opt.multiseq`
composes the same predicate (the concurrent path rides the fast sampler's graph path). No torch here: the lever file passes plain values in.

Served: proposal ``dlmc``, no rejection step, no trajectory requested, a differentiable penalty — upstream's own sampling configuration — on
CUDA tensors. Any other call under `CALIBY_FAST_SAMPLER` >= 1 (a sampling override naming another proposal or a rejection step, CPU
tensors) takes upstream's sweep loop for that call, announced once per call on stderr as ``[CALIBY_FAST_SAMPLER] <reason> -> stock sampler
for this call`` and counted (``report.FALLBACK_MARKERS``: a lever of the row that could not run — the run ends NOT ACTIVE by name).
"""
import sys
from typing import Optional

SWITCH = "CALIBY_FAST_SAMPLER"
FALLBACK_MARKER = f"[{SWITCH}]"


def refusal(proposal: str, rejection_step: bool, differentiable_penalty: bool, is_cuda: bool, return_trajectory: bool = False) -> Optional[str]:
    """None when the fast sampler serves the call, else the reason it is upstream's sweep loop."""
    if proposal != "dlmc" or rejection_step or not differentiable_penalty or return_trajectory:
        return "sampler configuration outside the fast path (proposal/rejection_step/differentiable_penalty/return_trajectory)"
    if not is_cuda:
        return "CPU tensors"
    return None


def note_stock(reason: str) -> None:
    """The stock-sampler line on stderr, once per call the fast sampler does not serve (the report's stderr probe counts it)."""
    print(f"[CALIBY_FAST_SAMPLER] {reason} -> stock sampler for this call (the upstream sweep loop)", file=sys.stderr, flush=True)
