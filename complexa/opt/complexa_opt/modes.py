"""Modes: the one table of this engine's mode names and what each resolves to.

``MODES`` is the list every command reads. ``off`` = stock (route word ``stock``): upstream's own ``complexa generate`` command on the
shipped pipeline configuration in a proven-clean child process, upstream's own knobs given the stock way — Hydra overrides after ``--``,
verbatim (``settings.py``). ``exact`` / ``fast`` / ``big`` (route word ``kit``) run the SAME command with ``COMPLEXA_OPT=<mode>`` in the
child's environment: the kit's autoload hook (``_autoload.py``, laid by the install's ``complexa_opt_autoload.pth``) then installs the mode's
lever set (``KIT_MODES``: the names of ``levers.LEVERS``) inside upstream's generation process right after ``proteinfoundation.proteina`` is
imported — ``exact``: the four exact-class levers (designs byte-identical to stock's); ``big``: those plus the row-blocked pair bias (peak
memory the lowest of the modes at large targets, still byte-identical); ``fast``: the exact four plus the fused pair bias and SDPA
attention (tolerance class: designs differ from stock's at the level of stock's own seed-to-seed variation). One lever set per mode on every
card this kit serves (``configs/h100.env``, ``configs/a100.env`` name the card, not the levers). A mode is all of its levers: one that cannot
be installed makes the generation process refuse by name (exit 3), never run under the mode's name with a subset. ``DEFAULT_MODE`` =
``fast``. The mode of a run: ``--mode`` when given, else ``COMPLEXA_OPT``, else the default (``opt_core.modes.mode_argument``); a ``--mode``
that disagrees with a set ``COMPLEXA_OPT`` is refused (``ModeConflict``).
"""
from __future__ import annotations

from typing import Mapping, Optional, Tuple

from opt_core.modes import ModeError, ModeTable, mode_argument

OFF, EXACT, FAST, BIG = "off", "exact", "fast", "big"
MODES = (OFF, EXACT, FAST, BIG)
STOCK_MODES = (OFF,)                                # resolves no lever: the stock route
EXACT_SET: Tuple[str, ...] = ("onehot_f32", "target_hoist", "pair_assembly", "loop_desync")
KIT_MODES = {                                       # mode -> its lever set (levers.LEVERS names), the same on every card
    EXACT: EXACT_SET,
    FAST: EXACT_SET + ("pair_bias_fused", "attn_sdpa"),
    BIG: EXACT_SET + ("pair_bias_rows",),
}
TIERS = {EXACT: "exact", FAST: "tolerance", BIG: "exact"}          # the identity class each mode is held to (big's levers are all exact-class)
DEFAULT_MODE = FAST
SERVED = MODES                                      # every mode of the table runs
ENV_MODE = "COMPLEXA_OPT"                           # the environment route of the mode word: read by the command line and, inside the generation process, by the autoload hook
ENV_RECORD = "COMPLEXA_OPT_RECORD"                  # the directory the generation process writes its activation record into (set by the kit route, read by the hook)
ROUTES = {OFF: "stock", EXACT: "kit", FAST: "kit", BIG: "kit"}
ROUTE_WORDS = {"stock": "upstream's `complexa generate`, the binder generation stage, with the caller's Hydra overrides after `--` verbatim, in a proven-clean child",
               "kit": "the same `complexa generate` command with COMPLEXA_OPT=<mode> in the child's environment: the autoload hook installs the mode's levers inside upstream's generation process"}
ESCAPE = "--mode off"                               # the one-flag escape every refusal of a kit mode names

TABLE = ModeTable(MODES, DEFAULT_MODE, unknown_message=lambda v: f"unknown mode {v!r}: the modes are {'|'.join(MODES)} ({OFF} runs stock; {EXACT}|{FAST}|{BIG} run the kit's lever sets)")


class ModeConflict(ModeError):
    """``--mode`` and ``COMPLEXA_OPT`` both given and naming different modes."""


def requested(cli_value: Optional[str], environ: Optional[Mapping[str, str]] = None) -> str:
    """The mode word a run asks for, unjudged: ``--mode``, else ``COMPLEXA_OPT``, else ``DEFAULT_MODE`` (what a refusal line names)."""
    import os
    environ = os.environ if environ is None else environ
    for v in (cli_value, environ.get(ENV_MODE)):
        if v is not None and str(v).strip():
            return str(v).strip().lower()
    return DEFAULT_MODE


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


def route_of(mode: str) -> str:
    """``stock`` or ``kit``; ValueError for a word that is not a mode."""
    if mode not in ROUTES:
        raise ValueError(TABLE.unknown_message(mode))
    return ROUTES[mode]


def levers_of(mode: str) -> Tuple[str, ...]:
    """The mode's lever set (empty for ``off``)."""
    return tuple(KIT_MODES.get(mode, ()))
