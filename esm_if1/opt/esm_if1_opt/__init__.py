"""esm_if1_opt — ESM-IF1 (Meta FAIR's GVP-Transformer inverse-folding model, 142M parameters) sequence design behind one command line.

Modes (``modes.MODES``, the one table): ``off`` is STOCK — upstream's example script ``sample_sequences.py`` as shipped, one interpreter per
structure, one ``model.sample()`` call per sequence, unseeded, each in a child process proven clean of any kit first (``stock_design.py`` over
``opt_core.stock_proof``); ``fast`` (the default word) is the kit's one tier — batched sampling, ``batched.py``: upstream's unmodified encoder
and decoder over B (backbone, sample) rows per forward under ``torch.inference_mode``, inputs parsed once, the model loaded once,
``torch.manual_seed(--seed)`` once after the load (lever ``batched_sampling``, run in the kit child ``kit_design.py``); any other mode word is a
usage error, exit 2. The command line takes upstream ``sample_sequences.py``'s own arguments with its names and defaults (``pdbfile``, ``--chain``, ``--temperature``,
``--outpath``, ``--num-samples``, ``--multichain-backbone`` / ``--singlechain-backbone``, ``--nogpu``), the driver's ``--input <dir|file>`` /
``--out <dir>``, ``--seed N`` and ``--batch_size B`` (inputs of every pass), ``--det [0|1]`` (never applies here) and ``run.sh --config <card>``.
No variant axis: one checkpoint (``esm_if1_gvp4_t16_142M_UR50``).

Entry points:
    esm_if1_opt.enable(mode=None) -> report      resolve the mode for this process (``off``: inactive, stock; ``fast``: active, batched_sampling; an unknown word raises)
    esm_if1_opt.status() -> report               the last report, or {"active": False, "reason": ...}
    esm_if1-opt design|check ...                 the command line (cli.py); ``ESM_IF1_OPT=<mode>`` in the environment is honoured by this
                                                 command line and run.sh only.
Statement one of every entry (``python -m esm_if1_opt`` / the console script, the two functions above) is ``core_gate()``: the core pin gate
(``_core_gate.gate`` — ``common/opt_core/kit_template/_core_gate.py`` byte for byte) compares ``opt/pyproject.toml`` ``[tool.opt_core]`` (a
path plus a MINIMUM version) with the installed core's own ``__version__``, without importing anything of the core, and refuses by name —
one ``[esm_if1-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …`` line on stderr, ``SystemExit(3)``
— on an absent or older core. Importing the package itself stays inert: this module imports nothing of the core and no sibling module at
module level (the proven stock child relies on it).
"""
from __future__ import annotations

__version__ = "0.4.2"
TAG = "esm_if1-opt"                     # the one spelling of this kit's line tag (``[esm_if1-opt]``): lines.TAG carries the same bytes (tests hold them equal)


class ActivationError(RuntimeError):
    """Named refusal: an unknown mode word, or the mode cannot run in this process (fair-esm / torch not installed)."""


def core_gate() -> dict:
    """Statement one of every entry: the core pin gate (``_core_gate.gate``) under this kit's tag. Returns the gate's facts; an absent /
    mismatched core or an unreadable pin is its one NOT ACTIVE line on stderr and ``SystemExit(3)`` — never a traceback."""
    from ._core_gate import gate
    return gate(__file__, tag=TAG)


def enable(mode: str | None = None) -> dict:
    """Resolve ``mode`` for this process and return the report (``stack.enable``). Statement one is the core pin gate."""
    core_gate()
    from . import stack
    return stack.enable(mode)


def status() -> dict:
    """The last report of this process, or ``{"active": False, "reason": ...}`` before any call. Statement one is the core pin gate."""
    core_gate()
    from . import stack
    return stack.status()


__all__ = ["__version__", "TAG", "ActivationError", "core_gate", "enable", "status"]
