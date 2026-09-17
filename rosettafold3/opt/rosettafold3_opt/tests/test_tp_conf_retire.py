"""rosettafold3_opt.tp_conf.retire_after_embed — the trunk park's early retirement never fires on the recycle-0
early-stop probe pass (``last_use=False``: the trunk shard is read again after it), fires on the sample passes, and is silent on a core without
``ZTrunkPlan.retire``. Torch-free: a stand-in plan records the calls. (The failure this guards, seen under ``big --n_gpu 2`` with the
upstream early-stop probe on: the probe's park was retired, ``ZTrunkPlan.end(0, device_needed_next=True)`` refused by name, the item raised, the
mode's audit then read ``feature_park: fold1: never marked`` -> exit 3.)

Run: ``python -m pytest opt/rosettafold3_opt/tests/test_tp_conf_retire.py -q``."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from rosettafold3_opt import tp_conf  # noqa: E402


class _Plan(object):
    def __init__(self, passes, parked=True):
        self.passes, self.parked, self.calls = passes, parked, []

    def retire(self, i):
        self.calls.append(int(i))
        if i != self.passes - 1:
            return "kept:not_last"
        return f"pass{i}@embed" if self.parked else "nothing_parked"


class _OldPlan(object):                                                  # a core before 0.5.214: no retire()
    passes = 1


def test_probe_pass_is_never_retired():
    plan = _Plan(passes=1, parked=True)                                   # the probe: its own one-pass plan, the shard parked, read again after
    assert tp_conf.retire_after_embed(plan, 0, False) == tp_conf.RETIRE_KEPT_PROBE == "kept:probe"
    assert plan.calls == []                                               # the plan was not touched: end(0, device_needed_next=True) restores the shard


def test_last_sample_pass_retires_and_earlier_passes_keep():
    plan = _Plan(passes=3, parked=True)                                   # D = 3 confidence passes (last_use=None: the core decides the form)
    assert tp_conf.retire_after_embed(plan, 0, None) == "kept:not_last"
    assert tp_conf.retire_after_embed(plan, 1, None) == "kept:not_last"
    assert tp_conf.retire_after_embed(plan, 2, None) == "pass2@embed"
    assert plan.calls == [0, 1, 2]
    single = _Plan(passes=1, parked=False)                                # D = 1, resident / in place: the plan's own word
    assert tp_conf.retire_after_embed(single, 0, None) == "nothing_parked"
    assert tp_conf.retire_after_embed(_Plan(passes=1, parked=True), 0, True) == "pass0@embed"   # an explicit last use retires too


def test_core_without_retire_is_silent():
    assert tp_conf.retire_after_embed(_OldPlan(), 0, None) is None
    assert tp_conf.retire_after_embed(_OldPlan(), 0, False) is None


if __name__ == "__main__":
    for t in (test_probe_pass_is_never_retired, test_last_sample_pass_retires_and_earlier_passes_keep, test_core_without_retire_is_silent):
        t(); print(f"RESULT {t.__name__} PASS")
