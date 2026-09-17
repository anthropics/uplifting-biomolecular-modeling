"""X010 concurrent multi-sequence sampling (`CALIBY_X_MULTISEQ`, lever code in the kit's `fast/potts.py`, `sample_potts_multiseq`): the
switch word, the predicate that decides whether a call is served concurrently or as the serial calls, the serial-fallback line, and the
census the exit line reads (`report.multiseq_state`). No torch here: the lever file passes plain values in.

Served: the fast sampler's graph path (`CALIBY_FAST_SAMPLER` >= 2) on the calls the fast sampler itself serves (`fast_sampler.refusal`, the
one predicate: CUDA tensors, proposal ``dlmc``, no rejection step, a differentiable penalty, no trajectory requested) — exactly the calls
`sample_potts` hands to the level-2 fast sampler; every other call is the serial calls, announced once per call on stderr as
``[CALIBY_X_MULTISEQ] <reason> -> serial fallback (...)`` and counted.
"""
import sys
from typing import Optional

from . import fast_sampler
from .env_words import int_word

SWITCH = "CALIBY_X_MULTISEQ"
FALLBACK_MARKER = f"[{SWITCH}]"
STATE = {"calls": 0, "sequences": 0, "serial_calls": 0, "serial_reasons": {}, "rng": None}


def level() -> int:
    """0 = the serial calls (unset), 1 = concurrent; anything else is a usage error (`env_words.UsageError`)."""
    value = int_word(SWITCH, default=0)
    if value > 1:
        from .env_words import UsageError
        raise UsageError(f"[{SWITCH}] malformed value {value!r}: expected 0 | 1")
    return value


def refusal(fast_sampler_level: int, proposal: str, rejection_step: bool, differentiable_penalty: bool, is_cuda: bool,
            return_trajectory: bool = False) -> Optional[str]:
    """None when the concurrent path serves the call, else the reason it is the serial calls."""
    if fast_sampler_level < 2:
        return f"CALIBY_FAST_SAMPLER={fast_sampler_level} (the concurrent path rides the fast sampler's graph path, level 2)"
    return fast_sampler.refusal(proposal, rejection_step, differentiable_penalty, is_cuda, return_trajectory)


def note_serial(reason: str) -> None:
    """The serial-fallback line (stderr) and its count: once per refused call."""
    print(f"[CALIBY_X_MULTISEQ] {reason} -> serial fallback (the sequences of this call sampled one call at a time)", file=sys.stderr, flush=True)
    STATE["serial_calls"] += 1
    STATE["serial_reasons"][reason] = STATE["serial_reasons"].get(reason, 0) + 1


def note_served(n_seqs: int, rng: Optional[dict] = None) -> None:
    """One concurrent call served: ``n_seqs`` sequences; ``rng`` = the call's generator positions (seed, base, increment, unit, after)."""
    STATE["calls"] += 1
    STATE["sequences"] += int(n_seqs)
    if rng is not None:
        STATE["rng"] = dict(rng)


def census_word() -> str:
    """``<calls>calls/<sequences>seqs[,serial:<n>]`` — the exit line's ``multiseq=`` value for a process whose potts module is the kit's."""
    word = f"{STATE['calls']}calls/{STATE['sequences']}seqs"
    if STATE["serial_calls"]:
        word += f",serial:{STATE['serial_calls']}"
    return word
