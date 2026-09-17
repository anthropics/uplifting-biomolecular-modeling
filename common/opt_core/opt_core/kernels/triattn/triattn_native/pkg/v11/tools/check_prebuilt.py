#!/usr/bin/env python3
"""check_prebuilt.py -- verify every prebuilt extension record of this package against the files beside it and the package's sources, and
(re)write prebuilt/INDEX.json from those records.  No torch, no GPU: digests and fields only (the byte gate is test_pkg.py's job).

    python tools/check_prebuilt.py                 one line per <stack tag>/<ext>: so_sha256 == the .so, source_sha256 == triattn_pkg/<dir>/csrc,
                                                   module_sha256 == triattn_pkg/<dir>/<module>.py, arch == the extension's architecture, loadcheck state;
                                                   exit 1 on any mismatch (records of an earlier generation without digest fields are reported as such)
    python tools/check_prebuilt.py --write-index   also rewrite prebuilt/INDEX.json's `stacks` section from the records (keeps `verified` / prose keys)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
PAYLOAD = os.path.join(PKG, "triattn_pkg")
PREBUILT = os.path.join(PAYLOAD, "prebuilt")
EXTS = {"triattn_sm90_ext": ("cuda", "cuda", "triattn_cuda", "sm_90a"), "triattn_mw_ext_g3x4": ("cuda_c", "cuda_c", "triattn_mw", "sm_90a"),
        "triattn_m1_ext": ("cuda_b", "cuda_b", "triattn_m1", "sm_90a"), "triattn_sm80_ext": ("cuda_80", "cuda_80", "triattn_sm80", "sm_80")}   # ext -> (route, dir, module, arch)
ARCH_FLAG = {"sm_90a": "arch=compute_90a,code=sm_90a", "sm_80": "arch=compute_80,code=sm_80"}


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def source_digests(d):
    root = os.path.join(PAYLOAD, d, "csrc"); out = {}
    for r, _, fs in os.walk(root):
        if "__pycache__" in r:
            continue
        for f in fs:
            p = os.path.join(r, f); out[os.path.relpath(p, root)] = sha256(p)
    return dict(sorted(out.items()))


def check():
    rows, fails = [], 0
    for tag in sorted(os.listdir(PREBUILT)):
        d = os.path.join(PREBUILT, tag)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            ext = f[:-5]; rec = json.load(open(os.path.join(d, f))); so = os.path.join(d, ext + ".so")
            if ext not in EXTS:
                rows.append((tag, ext, "FAIL", "unknown extension name")); fails += 1; continue
            route, sdir, module, arch = EXTS[ext]
            probs, notes = [], []
            if not os.path.isfile(so):
                probs.append("no .so beside the record")
            else:
                s = sha256(so)
                if "so_sha256" in rec:
                    (notes if rec["so_sha256"] == s else probs).append("so_sha256 %s" % ("ok" if rec["so_sha256"] == s else "MISMATCH %s != %s" % (rec["so_sha256"][:12], s[:12])))
                elif "sha256" in rec:
                    (notes if rec["sha256"] == s else probs).append("sha256 %s" % ("ok" if rec["sha256"] == s else "MISMATCH"))
                else:
                    notes.append("no binary digest in the record (earlier generation; the byte gate covers it)")
                if rec.get("so_bytes") not in (None, os.path.getsize(so)):
                    probs.append("so_bytes %s != %d" % (rec.get("so_bytes"), os.path.getsize(so)))
            if "source_sha256" in rec:
                cur = source_digests(sdir)
                (notes if rec["source_sha256"] == cur else probs).append("source_sha256 %s" % ("ok (%d files)" % len(cur) if rec["source_sha256"] == cur else
                                                                          "MISMATCH: %s" % sorted(k for k in set(cur) | set(rec["source_sha256"]) if cur.get(k) != rec["source_sha256"].get(k))))
            if "module_sha256" in rec:
                cur = {module + ".py": sha256(os.path.join(PAYLOAD, sdir, module + ".py"))}
                (notes if rec["module_sha256"] == cur else probs).append("module_sha256 %s" % ("ok" if rec["module_sha256"] == cur else "MISMATCH"))
            a = rec.get("arch")
            if isinstance(a, list):
                (notes if a == [ARCH_FLAG[arch]] else probs).append("arch %s" % ("ok" if a == [ARCH_FLAG[arch]] else "MISMATCH %s" % a))
            elif isinstance(a, str):
                (notes if arch.replace("sm_", "") in a.replace(".", "").replace("sm_", "") or ARCH_FLAG[arch] in a else probs).append("arch '%s'" % a)
            if rec.get("stack_tag") not in (None, tag):
                probs.append("stack_tag %s != dir %s" % (rec.get("stack_tag"), tag))
            lc = rec.get("loadcheck")
            notes.append("loadcheck %s" % ("pending" if lc == "pending" else ("%d case(s)" % len(lc) if isinstance(lc, dict) else "absent")))
            rows.append((tag, ext, "FAIL" if probs else "ok", "; ".join(probs + notes)))
            fails += bool(probs)
    for tag, ext, st, msg in rows:
        print("%-4s %-52s %-22s %s" % (st, tag, ext, msg))
    print("check_prebuilt: %d record(s), %d with problems" % (len(rows), fails))
    return fails, rows


def write_index():
    idx_path = os.path.join(PREBUILT, "INDEX.json")
    idx = json.load(open(idx_path)) if os.path.isfile(idx_path) else {}
    stacks = {}
    for tag in sorted(os.listdir(PREBUILT)):
        d = os.path.join(PREBUILT, tag)
        if not os.path.isdir(d):
            continue
        exts = {}
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json") or f[:-5] not in EXTS:
                continue
            ext = f[:-5]; rec = json.load(open(os.path.join(d, f))); route, sdir, module, arch = EXTS[ext]
            so = os.path.join(d, ext + ".so")
            ptx = rec.get("ptxas"); ptx = ptx.get("version") if isinstance(ptx, dict) else ptx
            e = {"route": route, "dir": sdir, "arch": arch, "generation": "11" if "so_sha256" in rec else "10 (carried unchanged)", "module": module,
                 "torch": rec.get("torch"), "torch_cuda": rec.get("torch_cuda"), "python": rec.get("python"), "nvcc": rec.get("nvcc"), "ptxas": ptx,
                 "so_bytes": os.path.getsize(so) if os.path.isfile(so) else None, "built_utc": rec.get("built_utc"), "sha256": sha256(so) if os.path.isfile(so) else None,
                 "loadcheck": rec.get("loadcheck", "not recorded (earlier generation record; byte gate only)")}
            if rec.get("cutlass_tag"):
                e["cutlass_tag"] = rec["cutlass_tag"]
            exts[ext] = e
        old = (idx.get("stacks") or {}).get(tag) or {}
        stacks[tag] = {"for": old.get("for", "torch %s, CPython %s" % (next(iter(exts.values()))["torch"] if exts else "?", tag.split("cpython-")[-1][:3] if "cpython-" in tag else "?")), "extensions": exts}
    idx["stacks"] = stacks
    idx.setdefault("verified", {})
    for tag in stacks:
        idx["verified"].setdefault(tag, {})
    ns = idx.get("not_shipped") or {}
    idx["not_shipped"] = {k: v for k, v in ns.items() if k not in stacks}
    json.dump(idx, open(idx_path, "w"), indent=1); open(idx_path, "a").write("\n")
    print("INDEX.json: %d stack(s): %s" % (len(stacks), {t: sorted(s["extensions"]) for t, s in stacks.items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-index", action="store_true")
    a = ap.parse_args()
    fails, _ = check()
    if a.write_index:
        write_index()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
