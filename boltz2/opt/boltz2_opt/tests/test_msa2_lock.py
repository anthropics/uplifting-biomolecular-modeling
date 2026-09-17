"""msa2 lock discipline — CPU tests of boltz2_opt.msa2._Lock and the LEVER/census rendering: floor, discrimination (exactly one distinct
candidate), undetermined re-attempts, budget -> proven, first mismatch -> refused; and pwa2.signature / distinct_candidates de-duplication."""
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.normpath(os.path.join(HERE, "..", ".."))


def _msa2():
    name = "boltz2_opt.msa2"
    if name in sys.modules:
        return sys.modules[name]
    if "boltz2_opt" not in sys.modules:
        pkg = types.ModuleType("boltz2_opt"); pkg.__path__ = [os.path.join(OPT, "boltz2_opt")]; sys.modules["boltz2_opt"] = pkg
    spec = importlib.util.spec_from_file_location(name, os.path.join(OPT, "boltz2_opt", "msa2.py"))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def _pwa2_cpu():
    """the core cell's (opt_core.ops.msa_pwa2) pure-Python helpers without importing triton: exec only the candidates/signature/distinct_candidates defs."""
    spec = importlib.util.find_spec("opt_core.ops.msa_pwa2")
    assert spec is not None and spec.origin, "opt_core.ops.msa_pwa2"
    src = open(spec.origin).read()
    ns = {"List": list, "Tuple": tuple}
    for fn in ("def candidates(", "def signature(", "def distinct_candidates("):
        i = src.index(fn); j = src.index("\n\n\n", i)
        exec("from typing import List, Tuple\n" + src[i:j], ns)
    return ns


def test_lock_budget_and_floor():
    M = _msa2(); L = M._Lock(lock_elems=1_000_000, lock_calls=1)
    k = (400, 7311, True)
    assert L.first(k, {(0, 0): True}, elems=400 * 7311 * 64) == "selected" and L.get(k)["state"] == "proven"        # one distinct candidate, big call: proven at once
    k2 = (300, 400, False)                                                                                              # 120000 rows -> 7.68e6 elems: proven at once too
    assert L.first(k2, {(1, 44): True, (1, 12): False, (0, 0): False}, elems=300 * 400 * 64) == "selected" and L.get(k2)["sel"] == (1, 44)
    k3 = (64, 300, True)
    small = 64 * 300 * 64 // 2                                                                                          # pretend a sub-budget call: stays comparing
    L3 = M._Lock(lock_elems=10_000_000)
    assert L3.first(k3, {(0, 0): True}, elems=small) == "selected" and L3.get(k3)["state"] == "comparing"
    assert L3.next(k3, True, elems=small) == "ok" and L3.get(k3)["state"] == "comparing"
    for _ in range(20):
        L3.next(k3, True, elems=small)
    assert L3.get(k3)["state"] == "proven" and L3.get(k3)["compared_elems"] >= 10_000_000
    L.floor((33, 2, False)); assert L.get((33, 2, False))["state"] == "floor" and L.count("floor") == 1


def test_lock_discrimination_and_refusal():
    M = _msa2(); L = M._Lock()
    k = (201, 400, True)
    assert L.first(k, {(2, 9): True, (2, 41): True, (0, 0): False}, elems=201 * 400 * 64) == "undetermined"            # two distinct candidates tie: undetermined, no selection
    st = L.get(k); assert st["state"] == "comparing" and st["sel"] is None and st["undetermined"] == 1 and st["compared_elems"] == 0
    assert L.first(k, {(2, 9): True, (2, 41): True, (0, 0): True}, elems=1) == "undetermined" and st["undetermined"] == 2    # all equal: still undetermined
    assert L.first(k, {(2, 9): True, (2, 41): False, (0, 0): False}, elems=201 * 400 * 64) == "selected" and st["state"] == "proven" and st["sel"] == (2, 9)
    k2 = (995, 7459, True)
    assert L.first(k2, {(2, 3): False, (2, 35): False, (0, 0): False}, elems=10) == "refused" and L.get(k2)["state"] == "refused"
    k3 = (500, 2048, True); L2 = M._Lock(lock_elems=10**12)
    assert L2.first(k3, {(1, 52): True, (1, 20): False, (0, 0): False}, elems=500 * 2048 * 64) == "selected" and L2.get(k3)["state"] == "comparing"
    assert L2.next(k3, False, elems=1) == "refused" and L2.get(k3)["state"] == "refused"                                # a later mismatch refuses for good
    assert L2.next(k3, True, elems=10**13) == "refused" and L2.get(k3)["state"] == "refused"                            # ... and never resurrects or counts
    w = L2.words(M._cls_word); assert "500x2048:c=(1,52):refused:cand3/und0/calls1/elems65536000@" in w, w


def test_signature_dedup():
    P = _pwa2_cpu()
    assert P["signature"](0, 0) == P["signature"](1, 16) == P["signature"](1, 48) == (16, 0)
    assert P["signature"](1, 18) != P["signature"](1, 50) and P["signature"](2, 8) == (8, 0) and P["signature"](2, 27) == (8, 27)
    assert P["distinct_candidates"](400) == [(0, 0)]                                          # N%64 = N%32 = 16: every spelling is the ascending order
    assert P["distinct_candidates"](1000) == [(0, 0), (1, 40), (1, 8)]
    assert P["distinct_candidates"](402) == [(1, 18)] + [c for c in P["distinct_candidates"](402)[1:]] and (0, 0) in P["distinct_candidates"](402)
    assert len(P["distinct_candidates"](402)) == 2                                            # 402%32 == 402%64 == 18: one residue spelling + ascending
    d = P["distinct_candidates"](411); assert d == [(2, 27), (0, 0)]                         # 411%32 == 411%64 == 27
    d = P["distinct_candidates"](995); assert d == [(2, 3), (2, 35), (0, 0)]                 # 995%32 = 3, 995%64 = 35: two k8 spellings + ascending
    for n in (61, 103, 118, 199, 256, 300, 384, 385, 612, 995, 1292, 1340, 1400, 2048):
        sigs = [P["signature"](*c) for c in P["distinct_candidates"](n)]
        assert len(sigs) == len(set(sigs)) >= 1


def test_lever_line_and_census_render():
    M = _msa2()
    M._STATE["applied"] = ["pwa2x", "trans2x"]; M._STATE["requested"] = ["pwa2x", "trans2x"]
    M._STATE["state"]["pwa2x"] = {"state": "on", "reason": None, "mode": "exact"}; M._STATE["state"]["trans2x"] = {"state": "on", "reason": None, "mode": "exact"}
    LP, LT = M._P2_LOCK, M._T2_LOCK
    LP.classes.clear(); LT.classes.clear()
    LP.first((400, 7311, True), {(0, 0): True}, elems=400 * 7311 * 64, ms=120.0); LP.floor((33, 2, False)); LP.first((201, 400, True), {(2, 9): True, (2, 41): True}, elems=1)
    M._P2.update({"calls": 20, "served": 15, "compared": 2, "undetermined": 1, "fallback": 3, "fallback_by": {"small_class:33x2:u": 2, "undetermined:201x400:c": 1}})
    M._P2["selftest_ms"] = 120.0
    LT.next((2924400, "float32", 32), True, elems=2924400 * 64); LT.floor((66, "bfloat16", 0))
    M._T2.update({"calls": 30, "served": 22, "compared": 1, "fallback": 7, "fallback_by": {"small_class:66xbfloat16:0": 7}})
    r = M.report()
    pc, tc = r["pwa2_census"], r["trans2_census"]
    assert pc["classes"]["400x7311:c"]["state"] == "proven" and pc["classes"]["400x7311:c"]["sel"] == [0, 0] and pc["classes"]["33x2:u"]["state"] == "floor"
    assert pc["classes"]["201x400:c"]["state"] == "comparing" and pc["classes"]["201x400:c"]["undetermined"] == 1 and pc["floor"] == {"S": 16, "rows": 65536} and pc["lock"]["elems"] == 1_000_000
    assert tc["selftest"]["2924400xfloat32:32"]["state"] == "proven" and tc["selftest"]["66xbfloat16:0"]["state"] == "floor"
    lines = {l.split("name=")[1].split()[0]: l for l in r["lines"]}
    lp, lt = lines["msa_pwa_exact"], lines["msa_trans2_exact"]
    for tok in ("served=15", "compared=2", "undetermined=1", "selftest=classes:3,proven:1,comparing:1,refused:0,floor:1", "floor=S>=16,SxN>=65536", "lock=elems>=1000000",
                "400x7311:c=(0,0):proven:cand1/und0/calls1/elems187161600@120ms", "33x2:u=-:floor:", "201x400:c=-:comparing:cand2/und1/calls0/elems0", "small_class:33x2:u:2"):
        assert tok in lp, (tok, lp)
    for tok in ("compared=1", "selftest=classes:2,proven:1,comparing:0,refused:0,floor:1", "floor=rows>=65536", "2924400xfloat32:32=-:proven:cand0/und0/calls1/elems187161600", "small_class:66xbfloat16:0:7"):
        assert tok in lt, (tok, lt)
