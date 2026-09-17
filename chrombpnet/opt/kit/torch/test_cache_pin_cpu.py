#!/usr/bin/env python
"""CPU test: (1) pin_gate passes when every shipped record's kernels_sha256_16 equals the package's kernels.py sha and raises naming the dir when
one does not; (2) kernels_jit_sha256_16 ignores an edit outside the @triton.jit bodies and changes on an edit inside one; (3) the wording used
when the shipped cache was not applied in the process makes no HIT claim; (4) the shipped cache dirs of this tree carry records whose pin equals
the package's kernels.py sha (reported as skipped when no shipped cache is present)."""
import os, sys, json, tempfile, hashlib, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chrombpnet_k1.cache_identity as CI
import chrombpnet_k1.forward as F
HERE = os.path.dirname(os.path.abspath(__file__)); KP = os.path.join(HERE, "chrombpnet_k1", "kernels.py")
cases = 0
# (1) the gate on fakes
def fake_kit(record_sha):
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "chrombpnet_k1")); shutil.copy(KP, os.path.join(d, "chrombpnet_k1", "kernels.py"))
    os.makedirs(os.path.join(d, "triton_cache_x")); json.dump({"kernels_sha256_16": record_sha, "kernels_jit_sha256_16": CI.kernels_jit_sha256_16(KP)}, open(os.path.join(d, "triton_cache_x", "JIT_IDENTITY.json"), "w"))
    json.dump({"entries": [{"arch": "cuda-90-32", "dir": "triton_cache_x", "sha256sums_sha256_16": "0" * 16}]}, open(os.path.join(d, "triton_cache_of_record.json"), "w")); return d
mine = hashlib.sha256(open(KP, "rb").read()).hexdigest()[:16]
rep = CI.pin_gate(fake_kit(mine)); assert len(rep) == 1 and rep[0]["pin_equal"] and rep[0]["record_kernels_sha256_16"] == mine, rep; cases += 1
try: CI.pin_gate(fake_kit("11993b0f239f0252")); raise SystemExit("the gate passed a stale pin")
except RuntimeError as e: assert "PIN EQUALITY GATE REFUSED" in str(e) and "triton_cache_x" in str(e) and "11993b0f239f0252" in str(e) and "jit sources EQUAL" in str(e), str(e)
cases += 1
# (2) the jit-source sha: helper-invariant, kernel-sensitive
src = open(KP).read(); jit0 = CI.kernels_jit_sha256_16(KP); names = [n for n, _ in CI.kernels_jit_sources(KP)]; assert len(names) >= 4 and "_gap_serial_kernel" in names, names
t = tempfile.mkdtemp(); p2 = os.path.join(t, "kernels.py"); open(p2, "w").write(src + "\n\ndef _helper_added_for_the_test():\n    return 1\n")
assert CI.kernels_jit_sha256_16(p2) == jit0 and hashlib.sha256(open(p2, "rb").read()).hexdigest()[:16] != mine; cases += 1
i = src.index("def _gap_serial_kernel"); j = src.index("\n", src.index("tl.", i)); p3 = os.path.join(t, "kernels2.py"); open(p3, "w").write(src[:j] + "   # an edit inside the kernel body" + src[j:])
assert CI.kernels_jit_sha256_16(p3) != jit0; cases += 1
# (3) the not-applied path's wording (no HIT claim)
F._CACHE_SKIP.clear(); F._CACHE_SKIP["note"] = "shipped cache NOT APPLICABLE (stage-1 pins differ: test)"
w = F.cache_witness(None, {}, {"make_ttir": 0, "make_cubin": 0}, F._JIT_RUNS.get("n0", 0) if hasattr(F, "_JIT_RUNS") else 0)
assert w["applied"] is False and "HIT" not in w["verdict"] and "served every kernel" not in w["verdict"] and "shipped cache not applicable" in w["verdict"] and "stage-1 pins differ" in w["verdict"], w["verdict"]; cases += 1
d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "e1")); open(os.path.join(d, "e1", "k.json"), "w").write("{}")
snap = F.cache_snapshot(d); w2 = F.cache_witness(d, snap, {"make_ttir": 0, "make_cubin": 0}, F._JIT_RUNS.get("n0", 0) if hasattr(F, "_JIT_RUNS") else 0)
assert w2["applied"] is True and w2["verdict"].startswith("shipped cache APPLIED — HIT"), w2["verdict"]; cases += 1
F._CACHE_SKIP.clear()
# (4) the shipped cache dirs' own records
lst = os.path.join(HERE, "triton_cache_of_record.json")
if os.path.isfile(lst):
    rep = CI.pin_gate(HERE); assert rep and all(r["pin_equal"] for r in rep), rep; print("   shipped caches pinned to this package's kernels.py:", [(r["dir"], r["record_kernels_sha256_16"]) for r in rep]); cases += 1
else: print("   (no triton_cache_of_record.json here: the frozen-dir case skipped — the freeze runs the gate itself)")
print(f"PASS test_cache_pin_cpu: {cases} cases (the pin gate passes on equal / refuses a stale pin by dir; the jit-source sha helper-invariant + kernel-sensitive; CHK-CBP-54 wording; the frozen dir's records == the package's pin when caches are shipped)")
