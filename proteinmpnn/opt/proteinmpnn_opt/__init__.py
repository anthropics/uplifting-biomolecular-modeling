"""proteinmpnn_opt — the ProteinMPNN inference optimizations behind one interface.

Modes (``modes.MODES``): ``off`` (stock: the upstream command line in a clean subprocess), ``exact`` (the kit's bit-identical line).
proteinmpnn ships no fast tier: ``fast`` (``--mode``, ``PROTEINMPNN_OPT``, ``enable()``) is refused by name before anything resolves
(``modes.UnsupportedMode``, a ``ValueError``; exit 3) with the pointer ``select --mode exact``. Variants (``modes.VARIANTS``): ``soluble``
(the default), ``vanilla`` — the two upstream weight sets; one per process.

Entry points:
    proteinmpnn_opt.enable(mode=None, variant=None) -> report    resolve and gate the mode on this box for this process (idempotent per
                                                                  process; refused by name after a kit worker has been launched here)
    proteinmpnn_opt.status() -> report                            the last activation report, or {"active": False, "reason": ...}
    proteinmpnn-opt design|check|warm ...                         the command line (cli.py); ``PROTEINMPNN_OPT=<mode>`` and
                                                                  ``PROTEINMPNN_VARIANT=<variant>`` in the environment are honoured by
                                                                  this command line and run.sh only.
Statement one of every entry (the two functions above, ``python -m proteinmpnn_opt`` / the console script, ``run.sh``, ``configs/<gpu>.env``,
a caller staging by hand through ``stage``) is ``core_gate()``: the core pin gate (``_core_gate.gate`` —
``common/opt_core/kit_template/_core_gate.py`` byte for byte) compares ``opt/pyproject.toml`` ``[tool.opt_core]``'s pinned minimum version
with the installed core's ``__version__`` literal, without importing anything of the core, and refuses by name — one
``[proteinmpnn-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …`` line on stderr,
``SystemExit(3)`` — on an absent or older core. Importing the package
itself stays inert: this module imports nothing of the core and no sibling module at module level.

The stock entry point is a script (``python protein_mpnn_run.py``): there is no import to hook, so the
environment route cannot attach to the stock command line and this package installs no site hook.
The kit's optimizations are drivers — a different executable run on the stock arguments — so activation here means: the mode is resolved
to the kit's own documented command line, gated (kit files present, stock pin, GPU), and armed for the launches this process makes.
A mode is all of its levers: ``enable()`` refuses by name (ActivationError) when a lever of the line cannot run on this box (no CUDA device:
the worker's CUDA-graph levers, named in ``partial`` / ``levers_unavailable``) unless ``--allow-partial`` or ``PROTEINMPNN_OPT_ALLOW_PARTIAL=1``
is given (recorded as ``allow_partial``: the levers that can run do); the worker refuses a job by name when its on-device probe fails a
probe-gated lever; a lever the end-of-run record shows off after a launch is ``partial`` too, and ``enable()`` cannot set the host
process's exit — the command line is the gated form (``design`` / ``warm`` exit 3; ``check`` exits 3 on such a plan). What is merely
untested (another GPU, driver, torch) is named on the lines and never refused.
"""
from __future__ import annotations

__version__ = "0.4.2"
TAG = "proteinmpnn-opt"                 # the one spelling of this kit's line tag (``[proteinmpnn-opt]``): report.PREFIX and the core pin gate's line take it from here


class ActivationError(RuntimeError):
    """Named refusal of an activation (unknown mode/variant, a gate that failed, late activation, a second mode in one process)."""


def core_gate() -> dict:
    """Statement one of every entry: the core pin gate (``_core_gate.gate``) under this kit's tag. Returns the gate's facts
    (``{"pinned": {path, version, pyproject}, "installed": {package_dir, root, version}, "tag"}``);
    an absent / older / unversioned core or an unreadable pin is its one NOT ACTIVE line on stderr and ``SystemExit(3)``
    (``_core_gate.CoreGateRefused``) — never a traceback. Imports nothing of the core; touches no ``sys.path``."""
    from ._core_gate import gate
    return gate(__file__, tag=TAG)


def enable(mode: str | None = None, variant: str | None = None) -> dict:
    """Resolve and gate ``mode`` for ``variant`` on this box; return the activation report (see ``stack.activate``).

    Statement one is the core pin gate (an absent / mismatched core: its NOT ACTIVE line and ``SystemExit(3)``). Idempotent per process
    for the same (mode, variant); a different pair after activation, or any call after a kit worker has been launched by this process,
    raises ``ActivationError`` naming the reason.
    """
    core_gate()
    from . import stack
    return stack.enable(mode, variant)


def status() -> dict:
    """The last activation report of this process, or ``{"active": False, "reason": ...}`` before any activation. Statement one is the core pin gate."""
    core_gate()
    from . import stack
    return stack.status()


__all__ = ["__version__", "TAG", "ActivationError", "core_gate", "enable", "status"]
