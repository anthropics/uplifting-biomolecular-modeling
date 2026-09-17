"""Modes: the one table of this engine's mode names and what each resolves to.

``MODES`` = ``off | fast``. ``off`` is STOCK: upstream's example script ``sample_sequences.py`` as shipped
(``stock/src/examples/inverse_folding/``), one interpreter per structure and one ``model.sample()`` call per requested sequence, unseeded,
run in a child process proven clean of any kit first (route word ``stock``). ``fast`` — the default word — is the kit's one tier: batched
sampling (``batched.py``, route word ``kit``): the same unmodified upstream encoder and decoder driven over ``--batch_size`` (backbone,
sample) rows per forward under ``torch.inference_mode``, every input parsed once, the model loaded once, ``torch.manual_seed(--seed)`` once
after the load; its one lever is ``batched_sampling`` (batched multinomial sampling draws from the same distribution as upstream's
one-sequence calls, not the same bits). The mode comes from ``--mode``, else ``ESM_IF1_OPT``, else the default; the two must agree when
both are given (``ModeConflict``); any other word is a usage error (``ModeError``, exit 2) — the kit never silently runs another mode instead.
"""
from __future__ import annotations

from typing import Mapping, Optional

from opt_core.modes import ModeError, ModeTable, mode_argument  # noqa: F401  (ModeError re-exported: the usage-error type callers catch)

MODES = ("off", "fast")
OFF, FAST = MODES
STOCK = "stock"                                    # the route word of --mode off: upstream's script in the proven stock child
KIT = "kit"                                        # the route word of --mode fast: the batched driver (batched.py)
BATCHED = "batched_sampling"                       # the one lever: B (backbone, sample) rows per forward instead of one sequence per call
KIT_MODES = {FAST: (BATCHED,)}                     # mode -> lever set
DEFAULT_MODE = FAST
ENV_MODE = "ESM_IF1_OPT"                           # the environment route of the mode (this command line and run.sh only)
ROUTES = {OFF: STOCK, FAST: KIT}
WORDS = {
    OFF: "stock: upstream's sample_sequences.py as shipped, one interpreter per structure, one model.sample() call per sequence, unseeded; "
         "nothing of the kit applies",
    FAST: "batched sampling: the unmodified upstream encoder and decoder over --batch_size (backbone, sample) rows per forward under "
          "torch.inference_mode, inputs parsed once, the model loaded once, torch.manual_seed(--seed) once after the load",
}

TABLE = ModeTable(MODES, DEFAULT_MODE)


class ModeConflict(ModeError):
    """``--mode`` and ``ESM_IF1_OPT`` name different modes."""


def resolve(cli_value: Optional[str], environ: Optional[Mapping[str, str]] = None) -> str:
    """The run's mode name. ``ModeError`` (usage, exit 2) for an unknown name or a conflict."""
    import os
    environ = os.environ if environ is None else environ
    env_value = environ.get(ENV_MODE)
    if cli_value and str(cli_value).strip() and env_value and str(env_value).strip():
        a, b = TABLE.check(cli_value), TABLE.check(env_value)
        if a != b:
            raise ModeConflict(f"--mode {a} disagrees with {ENV_MODE}={b}: give one")
    return mode_argument(cli_value, env_value, TABLE)


def is_kit(mode: str) -> bool:
    """True for the kit-tier words (every mode but ``off``): ``fast`` here."""
    return mode != OFF


def levers(mode: str) -> tuple:
    """The lever set of ``mode``: ``(batched_sampling,)`` for ``fast``, empty for ``off``."""
    return tuple(KIT_MODES.get(mode, ()))


def route_of(mode: str) -> str:
    """``off`` -> ``stock``; ``fast`` -> ``kit``."""
    return ROUTES[mode]
