"""boltzgen_opt — explicit interface to the BoltzGen inference optimizations.

    import boltzgen_opt
    report = boltzgen_opt.enable("exact")     # or "fast" | "big" | "off"; idempotent, once per process
    boltzgen_opt.status()                     # the last activation report

or, without code changes, `BOLTZGEN_OPT=exact boltzgen run spec.yaml ...` / `BOLTZGEN_OPT=exact python my_script.py`: the
package's .pth installs a lazy import hook that activates the mode the first time the upstream package `boltzgen` is imported, and
in every process of upstream's per-step pipeline (each starts with the activated parent's environment: _autoload.py). Nothing is
imported at interpreter start beyond this module; torch and the upstream package load only when a mode is activated. Statement one
of every entry — `python -m boltzgen_opt` / `boltzgen-opt`, `enable()`, `status()`, `ActivationError`, the hook's trigger, `run.sh`,
`configs/h100.env` — is the core pin gate (`core_gate()` below → `_core_gate.py`, a byte-for-byte copy of the shared core's template):
the `opt/pyproject.toml` `[tool.opt_core]` pin (path + minimum version) is compared with the installed core's own `__version__` before
anything of the core is imported, and an absent or older core is one `[boltzgen-opt] NOT ACTIVE: reason=core_missing:opt_core |
core_mismatch: … | core_pin_unreadable: …` line on stderr and exit 3. Importing the package, `MODES` and `DEFAULT_MODE` touch nothing of the core.

The lever modules activate by process environment (their `src` directories on PYTHONPATH, `fast_inference/src/sitecustomize.py`
importing the seed hook, the runner `xattempt_addon/src/xa_run.py` importing its lever modules — modes.py). `enable()` does
in-process what that launch does, after the gates (lever directories present, pins, a visible GPU — named on a NOTE line when it is
not the pinned card —, no model instance yet, no lever module imported yet). The command line (`boltzgen-opt design`) launches the
runner in a child process instead; the in-process pipeline (`inproc`) is only that form's. A mode is all of its levers: a partial
activation (`partial` on the report: the levers in `levers_fallback` could not run on this box) is a refusal by name — the ACTIVE
line's `fallback=`/`partial=`, one NOT ACTIVE line, the manifest's `partial`; the env route and `enable(strict=True)` end the
process (exit 3), `design`/`warm` exit 3 after the run, `enable()` prints the line and hands its caller the report. An untested card or library patch level is never that: it is named and the levers engage.

Modes (modes.MODES; each mode's levers: `CHANGES.md`): "exact" = the bit-identical lever set (outputs byte-identical to seeded stock
at the same seed), "fast" = exact's levers plus the tolerance-class levers (the default), "big" = the single-GPU lowest-memory
lever set, "off" = stock (upstream alone in a clean subprocess per pipeline step).
"""
from __future__ import annotations

__version__ = "0.5.2"
__all__ = ["enable", "status", "ActivationError", "MODES", "DEFAULT_MODE", "__version__"]
TAG = "boltzgen-opt"                       # the one tag of every line the package prints (`[boltzgen-opt] …`): report.PREFIX, _autoload's lines, the core pin gate's line, the .pth guard's


def print_fresh(line: str, stream=None) -> str:
    """Print one of the package's lines so that it starts a line of its own: a line feed, the line, a line feed, flushed (``stream``:
    stderr). Upstream's progress bar (tqdm on stderr) holds the cursor mid-line — ``\r``, the bar, no line feed — while a step runs; a
    line written behind such a fragment would not begin at column 0, and every reader of these lines anchors its grammar there
    (``^…$``). Every census / activation / refusal line of the package goes through here."""
    import sys
    out = stream if stream is not None else sys.stderr
    out.write("\n" + line + "\n")
    out.flush()
    return line


def core_gate() -> dict:
    """The core pin gate, statement one of every entry: ``_core_gate.gate`` on this package under the tag of its lines
    (``TAG``). Returns the gate's facts (``{"pinned": {...}, "installed": {...}, "tag"}``); an absent, mismatched or pre-manifest core
    or an unreadable pin is its one NOT ACTIVE line on stderr and ``SystemExit(3)`` (``_core_gate.CoreGateRefused``) — never a traceback,
    never a silent stock run. Imports nothing of the core."""
    from ._core_gate import gate
    return gate(__file__, tag=TAG)


def enable(mode: "str | None" = None, *, strict: bool = False, dry_run: bool = False) -> dict:
    """Activate ``mode`` in this process (idempotent). Returns the activation report; ``strict=True`` raises ActivationError on a
    refusal instead of returning an inactive report; ``dry_run=True`` resolves and gates without applying (what `check` does).
    Statement one is the core pin gate."""
    core_gate()
    from . import stack
    return stack.activate(mode, dry_run=dry_run, strict=strict)


def status() -> dict:
    """The last activation report of this process. Statement one is the core pin gate."""
    core_gate()
    from . import stack
    return stack.status()


def __getattr__(name):                                                    # PEP 562: nothing heavy at import
    if name == "ActivationError":
        core_gate()                                                       # the class lives in stack.py, which imports the core
        from .stack import ActivationError
        return ActivationError
    if name in ("MODES", "DEFAULT_MODE"):
        from . import modes
        return getattr(modes, name)
    raise AttributeError(name)
