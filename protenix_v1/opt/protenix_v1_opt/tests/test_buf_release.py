"""The exact TriMul's persistent per-N scratch buffers (the routed fpf_trimul.kernels `_BUF`) are released at every item entry: one
item's set resident, never every distinct N of the process; only for the exact TriMul; the record
reaches the trimul lever's evidence (kit_evidence buf_release) and its LEVER line."""
import sys
import types

from protenix_v1_opt import kit as K, modes, report as R, stack

EXACT = modes.resolve("exact"); FAST = modes.resolve("fast")


def _fake_kernels(monkeypatch, n):
    mod = types.ModuleType(stack.TRIMUL_KERNELS_MODULE)
    mod._BUF = {("ptx1", 128, 16 * (i + 1), "bf16", "cuda:0"): [object()] for i in range(n)}
    monkeypatch.setitem(sys.modules, stack.TRIMUL_KERNELS_MODULE, mod)
    return mod


def _reset(monkeypatch):
    monkeypatch.setattr(stack, "_BUFREL", {"state": None, "released": 0, "calls": 0})


def test_exact_releases_every_item(monkeypatch):
    _reset(monkeypatch)
    assert stack.release_trimul_buffers(EXACT) == 0 and stack._BUFREL["state"] == "na:not_imported"     # the first item: the kernel module is not imported yet
    mod = _fake_kernels(monkeypatch, 3)
    assert stack.release_trimul_buffers(EXACT) == 3 and mod._BUF == {}
    mod._BUF[("ptx1", 128, 64, "bf16", "cuda:0")] = [object()]
    assert stack.release_trimul_buffers(EXACT) == 1
    assert stack._BUFREL == {"state": "on", "released": 4, "calls": 2}
    ev = R.kit_evidence({"cfg": {"trimul": "exact"}, "counts": {"dit": {"apb:fp16": 48}, "atom": {"apb:tf32rn": 12}, "trimul": {"exact": 8}}, "trimul_buffers": dict(stack._BUFREL)}, "exact", EXACT.levers)
    assert ev["exact"]["buf_release"] == {"state": "on", "released": 4, "calls": 2}
    assert R.lever_lines(ev, {"mode": "exact"})[0].endswith("served=8 buf_release=on buf_released=4 lever=exact")


def test_fast_has_no_table_and_says_so(monkeypatch):
    _reset(monkeypatch); mod = _fake_kernels(monkeypatch, 2)
    assert stack.release_trimul_buffers(FAST) is None and len(mod._BUF) == 2 and stack._BUFREL["state"] == "na:trimul_fast"



def test_a_kernel_module_without_the_table_is_named(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setitem(sys.modules, stack.TRIMUL_KERNELS_MODULE, types.ModuleType(stack.TRIMUL_KERNELS_MODULE))
    assert stack.release_trimul_buffers(EXACT) is None and stack._BUFREL["state"] == "off:no_BUF_table"


def test_the_routed_kernel_really_has_the_table():
    """The release relies on opt_core/kernels/fpf_trimul/kernels.py keeping its buffers in a module dict `_BUF` keyed (key, C, Np, dtype, device)."""
    import os, re
    from opt_core import kernels as KR
    src = open(os.path.join(os.path.dirname(KR.__file__), "fpf_trimul", "kernels.py"), encoding="utf-8").read()
    assert re.search(r"^_BUF = \{\}", src, re.M) and "_BUF[k] = ent" in src        # planes of a keyed call are kept in the module table
    assert re.search(r"def trimul_forward\(.*key=", src)                            # the carried lever passes key="ptx1" (levers_ptx1.py:89)
