"""Lever `nosub_fn` — the forward-only executable (`fn`: ColabDesign's recycle-0, no-gradient forward that feeds `prev` into the graded step)
traced WITHOUT ColabDesign's sub-batch chunking at every size, exactly as lever `nosub` traces the forward+backward executable: the same
operations, one unchunked program (source `kit`) where stock chunks by 4 above 384 tokens (`colabdesign/af/prep.py:30`). At or below 384 tokens
stock's `fn` is unchunked already: the lever's size gate reads `skipped reason=gated` (stock's program). Above the gate the unchunked program's
GEMM shapes and XLA's fusion differ from the chunked one's: the same mathematics, a different floating-point association — numerics class
`precision`, never bitwise there. The decision and the hook are `nosub.py`'s (POLICY fn=kit); this module is the lever's protocol face
(registry.py): install / installed / uninstall / off_line / evidence, its LEVER line printed by the hook at model build:

    [colabdesign-opt] LEVER name=nosub_fn state=<on|skipped> [reason=gated] impl=subbatch_policy@<ver> origin=core tokens=<T> fn_subbatch=none
    fn_subbatch_source=kit stock_fn_subbatch=<4|none> gate=… 
"""
from __future__ import annotations

from opt_core import report as _core_report

from . import nosub
from .names import LEVER_NOSUB_FN, TAG

NUMERICS = "precision"                       # registry.LEVERS[nosub_fn].numerics
REFUSALS = ()                                # nothing of this lever can fail to install (pure Python over colabdesign's prep)
IMPL = nosub.IMPL


def install() -> None:
    """The forward-only executable unchunked at every size (nosub.POLICY fn=kit) through nosub's hook. Call before any `mk_afdesign_model(...)`."""
    nosub.install_hook()
    nosub.POLICY["fn"] = nosub.KIT


def installed() -> bool:
    return nosub.POLICY["fn"] == nosub.KIT and nosub.hook_installed()


def uninstall() -> None:
    nosub.POLICY["fn"] = nosub.STOCK
    nosub.uninstall_hook()


def off_line(reason: str) -> str:
    """The lever's line in a run that does not select it: state=off with the reason."""
    return _core_report.lever_line(TAG, LEVER_NOSUB_FN, "off", reason=reason, impl=IMPL, origin="core")


def evidence() -> dict:
    return {"builds": nosub.builds()}


def lines() -> list:
    """The nosub_fn LEVER lines printed in this process (one per distinct build)."""
    return [l for b in nosub.builds() for l in b["lines"] if f" name={LEVER_NOSUB_FN} " in l]
