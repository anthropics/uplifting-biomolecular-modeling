"""Card classes for this kit's levers (the release tree's ONE arch registry, :mod:`opt_core.arch`): which ``sm`` classes each lever's
KIT implementation supports, which classes it is known not to run on and why, and their per-class projection (``table()``).
One producer: every declaration of an ``LOCAL.esmfold2.<flag>`` key is made here; the LEVER lines
(report.lever_lines) carry the registry's words for the box's class beside the kit's own record of what ran (the carried driver's device
policy substitutes or disables the same levers by name at run time: ``driver/ef2_w4.py`` ``_apply_device_policy`` / ``_disable_lever``).
"""
from __future__ import annotations

from typing import Dict

from .registry import LEVERS, STRATEGY

SM_TESTED = ("sm90",)                   # the class every lever of this kit supports (H100 / H200)
SM80_TESTED = ("fused", "tg", "sg", "eg", "ec", "fc", "pb", "msa", "t3", "t5", "t6", "tx", "t10", "mk", "t11", "t12", "t13", "t14",   # the levers whose kit implementation also runs on the 8.0 class
               "ls", "rg", "ax", "af", "fz", "m15", "m16", "m17", "mh", "t15", "t15msa", "trimul", "glue", "disto",       # (A100, sm80, configs/a100.env: the exact and fast sets of
               "ro", "kd", "dit", "x2b", "x3", "x6", "x7", "x8", "x10", "x4", "xln")                                                                  # both model families — exact bit-identical to the library's fused
                                         # backend there under --det 1, fast within its tier, t10 / t3 / t15 on their sm_80 launch rows, tx on the shared core's 8.0 TriMul
                                         # cells, the XL storage levers under fast / big and x4 under big)
SM_TESTED_BY_FLAG: Dict[str, tuple] = {flag: SM_TESTED + ("sm80",) for flag in SM80_TESTED}   # per lever: the classes it supports (default SM_TESTED)
SM90A_ONLY = ("t16",)                      # the CuTe / NVRTC transition kernel: sm_90a code (wgmma, TMA, setmaxnreg) — no other class loads it; the package keeps it in the
                                            # 9.0 class's set alone (registry `classes`)
SM_EXCLUDE: Dict[str, Dict[str, str]] = {  # classes a lever's kit implementation is known not to run on, with the reason word
    "t10": {"sm100": "tmem_528_gt_512"},   # the fused pair-transition kernel's tensor-memory request exceeds the sm100 limit (device-disabled by the driver's probe)
    **{flag: {sm: "sm_90a_code" for sm in ("sm80", "sm100", "sm103")} for flag in SM90A_ONLY},   # every class of the registry (opt_core.arch.SM_CLASSES) but sm90
}
SM_FLOOR: Dict[str, str] = {}              # no lever of this kit has a hard floor among the registry's classes (the small-shared-memory policy is per device, not per class)


def arch_id(flag: str) -> str:
    """The opt_core.arch key of this kit's IMPLEMENTATION of lever ``flag``: ``LOCAL.esmfold2.<flag>``. Support on a card is a property
    of the kit's kernels, not of the shared strategy the lever files under (registry.STRATEGY); a family module's declaration of the
    shared id is the family's statement, this is the kit's."""
    return f"LOCAL.esmfold2.{flag}"


def declare() -> Dict[str, object]:
    """Declare every registry lever's card support ONCE (idempotent in opt_core.arch: the same content re-declared is a no-op). Returns
    ``{flag: Support}``."""
    from opt_core import arch
    return {flag: arch.declare(arch_id(flag), certified=SM_TESTED_BY_FLAG.get(flag, SM_TESTED), min_sm=SM_FLOOR.get(flag), exclude=SM_EXCLUDE.get(flag),
                               note=f"esmfold2 {flag}: files under {STRATEGY.get(flag, 'LOCAL.unfiled')}")
            for flag in LEVERS}


def table(sms=None) -> Dict[str, Dict[str, str]]:
    """``{flag: {sm: word}}`` for every registry lever over the registry's classes."""
    from opt_core import arch
    declare()
    t = arch.card_table([arch_id(f) for f in LEVERS], sms)
    return {flag: t[arch_id(flag)] for flag in LEVERS}


def words_for(flag: str, sm) -> dict:
    """The LEVER line's card fields for ``flag`` on class ``sm`` (opt_core.arch.lever_state): ``sm=<class|none>`` + ``card_support=<word>``
    when the class holds no measured row, or ``card=unsupported_card:<class> card_reason=<word>`` when the lever cannot run there."""
    from opt_core import arch
    declare()
    st = arch.lever_state(arch_id(flag), sm)
    fields = dict(st["evidence"])
    if st["state"] == "off":
        fields["card"] = st["reason"]
    return fields
