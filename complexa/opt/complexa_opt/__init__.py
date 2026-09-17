"""complexa_opt — Proteina-Complexa (NVIDIA's 160M-parameter flow-matching binder generator: 400 integration steps over the binder's CA
coordinates and per-residue latents against a fixed target, then a frozen autoencoder decodes atoms AND sequence) behind one interface.

Modes (``modes.MODES``, the one table): ``off`` (stock — route word ``stock``: upstream's own ``complexa generate`` command on the shipped
pipeline configuration, taking upstream's own knobs as Hydra overrides after ``--``, verbatim; none given = the shipped configuration) and
the kit modes ``exact`` / ``fast`` / ``big`` (route word ``kit``): the SAME command with ``COMPLEXA_OPT=<mode>`` in the child's
environment, under which the autoload hook (``_autoload.py``, laid by ``complexa_opt_autoload.pth``) installs the mode's lever set
(``modes.KIT_MODES``, ``levers.py``) inside upstream's generation process (``activate.enable``). No mode given asks for the default, ``fast``.
No variant axis: one checkpoint pair (``complexa.ckpt`` + ``complexa_ae.ckpt``, the protein-target model).

Entry point: the command line, ``complexa-opt design|check ...`` (cli.py; ``python -m complexa_opt``; ``run.sh`` wraps it);
``COMPLEXA_OPT=<mode>`` in the environment is honoured by the command line and run.sh and — inside upstream's own processes — by the autoload
hook. Statement one of every entry (``python -m complexa_opt`` / the console script, ``run.sh``, ``configs/<gpu>.env``, the hook under a kit
mode) is
``core_gate()``: the core pin gate (``_core_gate.gate`` — ``common/opt_core/kit_template/_core_gate.py`` byte for byte) compares
``opt/pyproject.toml`` ``[tool.opt_core]`` (``path`` + a MINIMUM ``version``) with the installed core's ``__version__`` literal, located
without importing anything of the core, and refuses by name — one ``[complexa-opt] NOT ACTIVE: reason=core_missing:opt_core |
core_mismatch: … | core_pin_unreadable: …`` line on stderr, ``SystemExit(3)`` — on an absent or older core (newer passes: the pin is a floor).
Importing the package itself stays inert: this module imports nothing of the core and no sibling module at module level.

The stock route is ONE upstream command — upstream's console script running upstream's installed code, nothing staged or patched — in a
proven-clean child process (``stock_design.py``): the environment stripped of this package's variables and proven so (``ENV-CLEAN``), the run
directory as working directory, the child's log output relayed; around it the package prints its ``[complexa-opt]`` lines (NOT ACTIVE,
WEIGHTS, ENV-CLEAN, INVOCATION, OUTPUTS, EXIT) and writes ``opt_manifest.json`` — what ran, and how many designs it wrote against how many
the command asked for. The kit route (``kit_design.py``) is the same command and the same census with the mode exported into the child and
the hook's presence proven first (``HOOK``); the generation process prints ACTIVE / LEVER / TALLY and leaves an activation record the parent
reads back (``KIT-RECORD``) before its EXIT verdict.
"""
from __future__ import annotations

__version__ = "0.3.1"
TAG = "complexa-opt"                    # the one spelling of this kit's line tag (``[complexa-opt]``)


class ActivationError(RuntimeError):
    """Named refusal of an activation (a kit tier this engine does not ship, an unknown mode, a gate that failed)."""


def enable(mode=None, *, strict: bool = True, trigger=None) -> dict:
    """The autoload finder's entry (``opt_core.autoload``): install ``mode``'s levers in THIS process — upstream's generation process — and
    print the ACTIVE line (``activate.enable``); ``ActivationError`` after one NOT ACTIVE line when the mode cannot be activated."""
    from .activate import enable as _enable
    return _enable(mode, strict=strict, trigger=trigger)


def core_gate() -> dict:
    """Statement one of every entry: the core pin gate (``_core_gate.gate``) under this kit's tag. Returns the gate's facts; an absent /
    mismatched / pre-manifest core or an unreadable pin is its one NOT ACTIVE line on stderr and ``SystemExit(3)`` — never a traceback."""
    from ._core_gate import gate
    return gate(__file__, tag=TAG)


__all__ = ["__version__", "TAG", "ActivationError", "core_gate", "enable"]
