#!/usr/bin/env python
"""CPU test (per (arch, MODE); TF-free, torch-free): the K1 package's arch-keyed default_route (arch_tiles.json, a per-mode dict
{'prod': {route, basis}, 'det': {route, basis}} or the older single {route, basis} = both modes) is THE ROUTE TABLE; the kit's class table
(fastdefault.K1_CLASS_TABLE: a string row 'default …' = K1 in both modes / 'STOCK ROUTE …' / 'REFUSED …' = never K1; or a dict row {'k1_by_mode':
{'prod': bool, 'det': bool}}) must AGREE on every (arch, class, mode); arch_default_route(kit_root, arch, mode) must return the package's answer.
Run under python -I from tf/: python -I test_route_agreement_cpu.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chrombpnet_fastkit import fastdefault as F
kit_root = os.path.join(os.path.dirname(os.path.dirname(HERE)), "kit"); p = os.path.join(kit_root, "torch", "chrombpnet_k1", "arch_tiles.json")
m = json.load(open(p)); ent = m.get("entries") or {}
def pkg_route(arch, mode):
    e = ent.get(arch)
    if not e: return "stock"
    r = e.get("default_route") or {"route": "k1"}
    if "route" not in r: r = r.get(mode) or {"route": "stock"}
    return r["route"]
n = 0; per_mode_arches = set()
for cls, arch in F.CLASS_ARCH.items():
    for mode in ("prod", "det"):
        theirs = pkg_route(arch, mode); theirs_k1 = (theirs == "k1")
        mine_k1 = F.class_route(cls, {"cuda-90": "9.0", "cuda-80": "8.0", "cuda-89": "8.9", "cuda-100": "10.0"}[arch], mode)[0]
        got = F.arch_default_route(kit_root, arch, mode); got_route = got[0] if got else "stock"
        assert theirs_k1 == mine_k1, (cls, arch, mode, "the package", theirs, "the class table", mine_k1)
        assert got_route == theirs, (cls, arch, mode, "arch_default_route()", got_route, "the package", theirs)
        e = ent.get(arch)
        if e and isinstance(e.get("default_route"), dict) and "route" not in e["default_route"]: per_mode_arches.add(arch)
        n += 1
    assert F.arch_key({"cuda-90": "9.0", "cuda-80": "8.0", "cuda-89": "8.9", "cuda-100": "10.0"}[arch], cls) == arch
assert "cuda-89" in per_mode_arches and pkg_route("cuda-89", "prod") != "k1" and pkg_route("cuda-89", "det") == "k1", ("cuda-89 must be per-mode: prod -> tf/stock, det -> k1", pkg_route("cuda-89", "prod"), pkg_route("cuda-89", "det"))
assert isinstance(F.K1_CLASS_TABLE["L40S"], dict) and F.K1_CLASS_TABLE["L40S"]["k1_by_mode"] == {"prod": False, "det": True}, F.K1_CLASS_TABLE["L40S"]
print(f"ROUTE AGREEMENT OK: {n} (arch, class, mode) cases — the package's per-mode default_route == the class table == arch_default_route(mode); per-mode arches {sorted(per_mode_arches)}; arch_tiles.json sha16 {__import__('hashlib').sha256(open(p,'rb').read()).hexdigest()[:16]}")
