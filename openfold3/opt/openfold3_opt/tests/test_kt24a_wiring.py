"""Structure first + pinned-pool shrink on the tp line: the hook WIRING of this kit (the mechanism lives once in opt_core.mem.rowpair.structure_first /
heads.pinned_shrink and is tested there): the seams are declared, the hooks call them, no copy of the module and no fault-injection switch is in the kit."""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))


def read(rel):
    with open(os.path.join(PKG, rel), encoding="utf-8") as fh:
        return fh.read()


def test_seams_declared():
    from openfold3_opt.tp_rowpair import core as C
    assert set(("capture", "write_early", "mark_final", "scan", "utc")) <= set(C.SEAMS["structure_first"])
    assert "pinned_shrink" in C.SEAMS["heads"]


def test_hooks_call_the_core_and_nothing_is_duplicated():
    model = read("tp_rowpair/model.py"); conf = read("tp_rowpair/confidence.py"); man = read("manifest.py"); env = read("tp_rowpair/env.py")
    assert model.count('C.fn("structure_first", "write_early")') == 1 and model.count('C.fn("structure_first", "capture")') == 2 and model.count('C.fn("structure_first", "mark_final")') == 1
    assert re.search(r'\[conf\] .* confidence heads start', model)
    assert 'C.fn("heads", "pinned_shrink")' in conf
    assert "structure_first_block" in man and "OUTPUT NOT COUNTED" in man
    assert "OF3TP_STRUCTURE_FIRST" in env and "OF3TP_FAIL_IN_CONFIDENCE" not in env and "FAIL_IN_CONFIDENCE" not in model
    assert not os.path.exists(os.path.join(PKG, "tp_rowpair", "structure_first.py"))
