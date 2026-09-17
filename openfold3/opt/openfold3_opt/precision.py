"""The fast mode's precision on every route: upstream's ``pl_trainer_args.precision`` is ``bf16-mixed`` where the Lightning Trainer reads it.

The CLI composes the fast runner yaml with that value (cli.row_yaml, runner_yaml.compose); the env/.pth route passes no yaml of the kit's. So the fast mode's
activation patches ``openfold3.entry_points.experiment_runner.ExperimentRunner.__init__`` through the core's import-time patch
(``opt_core.autoload.patch_attr_at_import``: now when the module is imported, else when upstream imports it; fail-closed by name): after
upstream has built the runner from its own config, ``self.pl_trainer_args.precision`` is set to ``BF16_MIXED`` — silently when the config
already says so (the CLI's composed yaml), with ONE stderr line naming the old value when it did not (a caller's yaml or upstream's default
on the env route: the fast mode is one composition, its precision wins and is named — the CLI's rule for a caller's own yaml, runner_yaml.compose).
"""
from __future__ import annotations

import functools
import sys

TAG = "openfold3-opt"
TARGET = "openfold3.entry_points.experiment_runner"        # upstream: ExperimentRunner.__init__ keeps experiment_config.pl_trainer_args; .trainer builds pl.Trainer(**pl_trainer_args)
ATTR = "ExperimentRunner.__init__"
BF16_MIXED = "bf16-mixed"                                  # upstream's pl_trainer_args.precision word of the fast composition (modes.FAST_PRECISION = bf16)
PATCH = None                                               # the opt_core.autoload.AttrPatch once installed (report: PATCH.record())


def _make_init(original):
    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        pl = getattr(self, "pl_trainer_args", None)
        if pl is None or not hasattr(pl, "precision"):        # upstream changed shape: named, never a silent fp32 fast run
            raise RuntimeError(f"[{TAG}] precision_bf16: {TARGET}.ExperimentRunner carries no pl_trainer_args.precision after __init__")
        was = pl.precision
        if str(was) != BF16_MIXED:
            sys.stderr.write(f"[{TAG}] fast: pl_trainer_args.precision {was!r} -> {BF16_MIXED!r} (the fast mode's precision)\n")
            sys.stderr.flush()
            pl.precision = BF16_MIXED
    return __init__


def install():
    """Arm (or apply) the patch; idempotent. Returns the core's AttrPatch."""
    global PATCH
    from opt_core import autoload as _core_autoload
    PATCH = _core_autoload.patch_attr_at_import(TARGET, ATTR, _make_init, tag=TAG, name="precision_bf16")
    return PATCH
