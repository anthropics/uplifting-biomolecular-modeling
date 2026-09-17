"""The kit's autoload hook, run at interpreter start by `esmfold2_opt_autoload.pth` (the guarded `import esmfold2_opt._autoload` line that
opt/_build_backend.py generates: a SystemExit of this module becomes its exit code, an import failure under ESMFOLD2_OPT the NOT ACTIVE line and exit 3).

Import-free until ESMFOLD2_OPT names a selection: the .pth runs in every interpreter on the box, the stock arm's processes included,
and those load nothing of the core. An undeclared name under the package prefix (ESMFOLD2_OPT_<x> outside ENV_NAMES — a mistyped
switch) is refused here, in every process: the NOT ACTIVE line and exit 3. With ESMFOLD2_OPT=exact|fast the core's finder
(`opt_core.autoload`) is installed from this kit's AutoloadSpec: it waits for the first import of a trigger package —
`transformers.models.esmfold2` (the model) or `esm.models.esmfold2` (the input builder), whichever a program imports first — lets that
package's own body run, removes itself and calls `esmfold2_opt.enable(mode, strict=True, trigger=<name>)`; the variant is read from
ESMFOLD2_VARIANT by the package's own resolver at that moment (stack.py). When the mode cannot be activated (no variant named, pins, no
GPU, kit missing) the package prints its NOT ACTIVE line and the process exits 3 — stock never runs silently under ESMFOLD2_OPT. An
unknown mode (outside MODES) is refused at interpreter start with the kit's line and exit 3 (`on_unknown="exit_now"`); a bare
`importlib.util.find_spec(<trigger>)` probe leaves the finder armed. Until the trigger nothing else is imported (no torch, no upstream);
with ESMFOLD2_OPT unset or "off" nothing is installed. enable() arms the application on `ESMFold2InputBuilder.fold`: the kit's
configure() runs on each model at its first fold (stack.py), the only point where a loaded model exists.
"""
import os
import sys

ENV, ENV_VARIANT = "ESMFOLD2_OPT", "ESMFOLD2_VARIANT"
TAG = "esmfold2-opt"
TRIGGERS = ("transformers.models.esmfold2", "esm.models.esmfold2")
MODES = ("exact", "fast", "off", "big")
EXIT_NOT_ACTIVE = 3                      # report.EXIT_NOT_ACTIVE, copied import-free (this module runs before the package imports); locked by test_merge_locks
ENV_NAMES = ("ESMFOLD2_OPT", "ESMFOLD2_OPT_FORCE", "ESMFOLD2_OPT_HOME", "ESMFOLD2_OPT_KIT", "ESMFOLD2_OPT_WEIGHTS_MEMO_DIR", "ESMFOLD2_OPT_CACHE_SCOPE", "ESMFOLD2_OPT_REQUIRE_FAST_ENV", "ESMFOLD2_OPT_ABLATE")   # the declared switches (stack.py / report.py / attn.py: … the weights digest memo directory, the cache scope, the ablation variable (ablation.py), the fast-environment fail-loud switch); locked by test_merge_locks


def spec():
    """This kit's AutoloadSpec for the core's finder (imports the core: called under a set variable, or from the package)."""
    from opt_core.autoload import AutoloadSpec
    return AutoloadSpec(env=ENV, package=__package__, tag=TAG, triggers=TRIGGERS, modes=MODES, exit_not_active=EXIT_NOT_ACTIVE,
                        on_unknown="exit_now", fold_mode=True)         # the selection is stripped and case-folded onto a declared mode (` Fast ` selects fast)


def disarm() -> bool:
    """Remove the mode finder before an explicit activation — an explicit enable() wins over the ESMFOLD2_OPT route. True when one was
    armed."""
    if FINDER is None or not FINDER.armed:
        return False
    FINDER.armed = False
    FINDER.remove()
    return True


def _refuse(reason: str) -> None:
    """The kit's NOT ACTIVE line and exit code at interpreter start (a SystemExit inside a .pth surfaces as status 1 with a traceback)."""
    sys.stderr.write(f"[{TAG}] NOT ACTIVE: {reason}\n")
    sys.stderr.flush()
    os._exit(EXIT_NOT_ACTIVE)


FINDER = None
_undeclared = sorted(k for k in os.environ if k.startswith(ENV + "_") and k not in ENV_NAMES)
if _undeclared:
    _refuse(f"undeclared {'/'.join(_undeclared)} in the environment (the package's switches are {'|'.join(ENV_NAMES)})")
_sel = (os.environ.get(ENV) or "").strip().lower()
if (os.environ.get(ENV) or "").strip().lower() not in ("", "off"):
    from ._core_gate import gate as _core_gate                           # statement one under a named mode: the importable opt_core is the one this kit pins ([tool.opt_core]
    from ._producers import kit_anchor as _kit_anchor, refusal as _producers_refusal   # beside $ESMFOLD2_OPT_HOME or above the package), read on disk, nothing of the core imported —
    _core_gate(_kit_anchor(__file__))                                    # else `NOT ACTIVE: reason=core_missing|core_mismatch|core_pin_unreadable …`, exit 3
    _why = _producers_refusal()                                          # (producer_missing:<modules>, exit 3) — stat calls only
    if _why:
        _refuse(_why)
    try:
        from opt_core.autoload import install
    except ImportError as _e:
        _refuse(f"core_missing:opt_core.autoload ({_e})")
    FINDER = install(spec())
