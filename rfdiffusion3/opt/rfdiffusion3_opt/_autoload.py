"""The kit's autoload hook, run at interpreter start by `rfdiffusion3_opt_autoload.pth` (the generated guard line of `opt/_build_backend.py`,
which imports `rfdiffusion3_opt._autoload` and turns a refusal here into the process's exit code).

Import-free until RFDIFFUSION3_OPT names a selection: the .pth runs in every interpreter on the box, the stock arm's processes included,
and those load nothing of the core. An undeclared name under the package prefix (RFDIFFUSION3_OPT_<x> outside ENV_NAMES — a mistyped
switch) is refused here, in every process: the NOT ACTIVE line and exit 3. With RFDIFFUSION3_OPT=exact|fast the core pin gate runs first (`core_gate()` below: the importable
`opt_core` is the one opt/pyproject.toml [tool.opt_core] pins, else `NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … |
core_pin_unreadable: …` and exit 3 — written, flushed, `os._exit`: a SystemExit raised under a .pth line would not end the process
cleanly), then the core's finder (`opt_core.autoload`) is installed from this kit's AutoloadSpec: it waits for the first import of the upstream package `rfd3`, lets that package's own body run (`rfd3/__init__.py` imports
`pydantic` and `packaging` only — nothing of `rfd3.model`), removes itself and calls `rfdiffusion3_opt.enable(mode, strict=True,
trigger="rfd3")`, which gates the mode and exports its environment delta; the kit's module `rfd3.model.hoist` reads `RFD3_HOIST` at its
own, later import (hoist.py:18), so the delta is in os.environ in time by construction (a process that imported `rfd3.model.hoist`
before the activation is refused by name: stack.kit_state_findings). When the mode cannot be activated (kit not installed in this
interpreter, pins, no GPU) the package prints its NOT ACTIVE line and the process exits 3 — stock never
runs silently under RFDIFFUSION3_OPT. An unknown mode is refused at interpreter start with the kit's line and exit 3
(`on_unknown="exit_now"`); a bare `importlib.util.find_spec("rfd3")` probe leaves the finder armed. With RFDIFFUSION3_OPT unset or `off`
nothing is installed. The activation installs the APPLIED hook (stack.AppliedHook): the one line of what the kit's module actually read,
and the exit rule at that import — the same hook on every route.
"""
import os
import sys

ENV = "RFDIFFUSION3_OPT"
TAG = "rfdiffusion3-opt"                         # the kit's line tag, the one spelling: report.TAG, the .pth guard line (tag=) and the core pin gate take it from here
TRIGGERS = ("rfd3",)
MODES = ("exact", "fast", "off")                 # modes.MODES, copied import-free (this module runs before the package imports); locked by test_merge_locks
EXIT_NOT_ACTIVE = 3                              # report.EXIT_NOT_ACTIVE, copied import-free; locked by test_merge_locks
ENV_NAMES = ("RFDIFFUSION3_OPT", "RFDIFFUSION3_OPT_ALLOW_PARTIAL", "RFDIFFUSION3_OPT_CACHE", "RFDIFFUSION3_OPT_DET")   # the declared switches (this module / report.ENV_ALLOW_PARTIAL / manifest.CACHE_ENV); locked by test_merge_locks


def spec():
    """This kit's AutoloadSpec for the core's finder (imports the core: called under a set variable, or from the package)."""
    from opt_core.autoload import AutoloadSpec
    return AutoloadSpec(env=ENV, package=__package__, tag=TAG, triggers=TRIGGERS, modes=MODES, exit_not_active=EXIT_NOT_ACTIVE,
                        on_unknown="exit_now", fold_mode=True,          # the selection is stripped and case-folded onto a declared name (` Exact ` selects exact)
                        line_of=lambda what, value: f"[{TAG}] NOT ACTIVE: '{value}' is not a mode ({'|'.join(MODES)})")   # the table's own wording (modes.TABLE)


def core_gate() -> dict:
    """The kit's core pin gate with the kit's tag — statement one of every entry (`python -m rfdiffusion3_opt` and the console script through
    `__main__`, `run.sh`, `configs/h100.env`, this module under a set RFDIFFUSION3_OPT, `enable()`, `status()`, the `install` and
    `stock_design` child modules), before the first `opt_core` import: `_core_gate.gate` compares opt/pyproject.toml [tool.opt_core] with the
    `opt_core` this interpreter would import (located, not imported) and refuses by name — one `[rfdiffusion3-opt] NOT ACTIVE:
    reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …` line, SystemExit(3) — or returns the facts
    {"pinned": {path, version, pyproject}, "installed": {package_dir, root, version}, "tag"}."""
    from ._core_gate import gate
    return gate(__file__, tag=TAG)


def disarm() -> bool:
    """Remove the mode finder before an explicit activation — an explicit enable() (`design --mode ...`) wins over the RFDIFFUSION3_OPT
    route. True when one was armed."""
    if FINDER is None or not FINDER.armed:
        return False
    FINDER.armed = False
    FINDER.remove()
    return True


def _refuse(reason: str) -> None:
    """The kit's NOT ACTIVE line and exit code at interpreter start (a SystemExit inside a .pth surfaces as status 1 with a traceback)."""
    from ._emit import emit
    emit(f"[{TAG}] NOT ACTIVE: {reason}")
    os._exit(EXIT_NOT_ACTIVE)


FINDER = None
_undeclared = sorted(k for k in os.environ if k.startswith(ENV + "_") and k not in ENV_NAMES)
if _undeclared:
    _refuse(f"undeclared {'/'.join(_undeclared)} in the environment (the package's switches are {'|'.join(ENV_NAMES)})")
if (os.environ.get(ENV) or "").strip().lower() not in ("", "off"):
    try:
        core_gate()                                  # the core pin gate, before the first opt_core import: its NOT ACTIVE line is written and flushed ...
    except SystemExit as _x:
        sys.stderr.flush()
        os._exit(_x.code if isinstance(_x.code, int) else EXIT_NOT_ACTIVE)   # ... and the process ends 3 here (a SystemExit under a .pth line does not end the interpreter cleanly)
    from opt_core.autoload import install
    FINDER = install(spec())
