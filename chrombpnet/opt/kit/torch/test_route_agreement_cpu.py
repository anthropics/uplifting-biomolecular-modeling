#!/usr/bin/env python
"""CPU test: the package's arch-keyed default_route (arch_tiles.json) and the device-class table of the fast kit
(opt/kit_ho/tf/chrombpnet_fastkit/fastdefault.py) agree, per (device class, mode), on whether K1 is the route; a class absent from that table is
reported, not failed. The table file must exist (no skip)."""
import os, sys, ast, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chrombpnet_k1.forward as F
ARCH_CLASSES = {"cuda-90": ("H100", "H200"), "cuda-89": ("L40S",), "cuda-80": ("A100",), "cuda-100": ("B200",)}    # the classes each arch serves (sm_90 H100/H200; sm_89 L40S; sm_80 A100; sm_100 B200)
def owner_table(path):
    src = open(path).read(); tree = ast.parse(src); tab = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "K1_CLASS_TABLE" for t in n.targets): tab = ast.literal_eval(n.value)
    assert isinstance(tab, dict) and tab, f"no K1_CLASS_TABLE dict literal in {path}"
    out = {}
    for k, v in tab.items():
        cls = k.replace("NVIDIA ", "").strip()
        if isinstance(v, str): out[cls] = v.startswith("default")                      # string form: 'default …' = K1 by default
        elif isinstance(v, dict): out[cls] = ({"prod": bool(v["k1_by_mode"].get("prod")), "det": bool(v["k1_by_mode"].get("det"))} if isinstance(v.get("k1_by_mode"), dict) else {"prod": bool(v.get("k1")), "det": bool(v.get("k1"))})   # the per-mode form {'k1_by_mode': {prod, det}} or the single bool (= both modes)
        else: raise AssertionError(f"unknown table row form for {k}: {type(v)}")
    return out, hashlib.sha256(src.encode()).hexdigest()[:16]
paths = [p for p in [os.environ.get("OWNER_FASTDEFAULT"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kit_ho", "tf", "chrombpnet_fastkit", "fastdefault.py")] if p and os.path.isfile(p)]
assert paths, "no class table to compare against (set OWNER_FASTDEFAULT=<tree>/opt/kit_ho/tf/chrombpnet_fastkit/fastdefault.py) — RED"
cases = 0; report = []
for p in paths:
    tab, sha = owner_table(p); bad = []
    for arch, classes in ARCH_CLASSES.items():
        for mode in ("prod", "det"):                                                # the route is per (class, MODE)
            mine = F.default_route(arch, mode)[0] == "k1"
            for cls in classes:
                if cls not in tab: continue                                         # a class absent from the fast kit's table = no claim to disagree with (reported)
                theirs = tab[cls][mode] if isinstance(tab[cls], dict) else bool(tab[cls])
                if theirs != mine: bad.append(f"{arch}/{cls}/{mode}: package default_route {'k1' if mine else 'not-k1'} vs the class table {'k1' if theirs else 'not-k1'}")
                cases += 1
    report.append({"table": p, "sha256_16": sha, "classes": tab, "checked": [c for cs in ARCH_CLASSES.values() for c in cs if c in tab], "absent": [c for cs in ARCH_CLASSES.values() for c in cs if c not in tab]})
    assert not bad, f"ROUTE DISAGREEMENT (RED) vs {p} ({sha}): {bad}"
print("route agreement:", json.dumps(report, default=str)[:900])
assert cases >= 4, f"too few (arch, class) pairs checked: {cases}"
print(f"PASS test_route_agreement_cpu: {cases} cases (the package's arch-keyed default_route == every class table checked, by bytes: {[r['sha256_16'] for r in report]})")
