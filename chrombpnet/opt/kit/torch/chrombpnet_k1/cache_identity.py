import os
import json
import hashlib
"""`python -m chrombpnet_k1.cache_identity <cache_dir>` — write JIT_IDENTITY.json beside a populated Triton cache (= build_cache_record).
The LAST populating step of a shipped cache runs this (selftest_k1.py does it itself on a --keep-triton-cache PASS; a later step that adds
kernels to the same dir ends with this command so the record covers every kernel present)."""
import sys, json
from .forward import build_cache_record
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] in ("-h", "--help"): print(__doc__); return 2
    rec = build_cache_record(argv[0]); print(json.dumps({k: rec[k] for k in ("triton", "torch", "cudnn", "device", "kernels_sha256_16")}, indent=1), "entries", len(rec.get("entries") or [])); return 0
if __name__ == "__main__": sys.exit(main())


# ---- pin equality: every shipped cache's JIT_IDENTITY.json must name the sha256 of the package's CURRENT kernels.py, else apply() cannot use that cache
# and every fresh process JIT-compiles; pin_gate() checks the whole cache list, kernels_jit_sha256_16() tells kernel edits from helper-only edits ----
def kernels_jit_sources(path):
    """The sources of the @triton.jit functions of kernels.py in definition order, [(name, source)]. Triton 3.0 hashes each kernel's source from
    the `def` line on (with the jit functions it calls, the Triton version and the kernel's starting line number) into its cache key, so this
    text is what a kernel edit changes; Python helpers outside the jit bodies enter a key only by shifting a kernel's line number. Information
    beside the whole-file pin (the sha256 of kernels.py, which is what apply() compares): it tells whether a pin mismatch comes from the kernels
    or only from the helpers."""
    import ast
    src = open(path).read(); t = ast.parse(src); out = []
    for n in t.body:
        if isinstance(n, ast.FunctionDef):
            for d in n.decorator_list:
                name = d.attr if isinstance(d, ast.Attribute) else (d.id if isinstance(d, ast.Name) else ((getattr(d.func, "attr", None) or getattr(d.func, "id", None)) if isinstance(d, ast.Call) else None))
                if name == "jit": out.append((n.name, ast.get_source_segment(src, n))); break
    return out

def kernels_jit_sha256_16(path):
    return hashlib.sha256("\n".join(f"{n}\n{s}" for n, s in kernels_jit_sources(path)).encode()).hexdigest()[:16]

def pin_gate(torch_dir, list_name="triton_cache_of_record.json"):
    """Pin equality over the kit's cache list: every shipped cache dir named in torch_dir/<list_name> must carry a JIT_IDENTITY.json whose
    kernels_sha256_16 equals the sha256[:16] of this package's own chrombpnet_k1/kernels.py — a stale pin means apply() never uses that cache
    and every fresh process JIT-compiles. Returns the per-dir report; raises RuntimeError naming every mismatching dir. test_cache_pin_cpu.py
    runs it against the shipped tree."""
    kp = os.path.join(torch_dir, "chrombpnet_k1", "kernels.py"); mine = hashlib.sha256(open(kp, "rb").read()).hexdigest()[:16]; mine_jit = kernels_jit_sha256_16(kp)
    lst = json.load(open(os.path.join(torch_dir, list_name))); report = []; bad = []
    for e in lst.get("entries", []):
        d = os.path.join(torch_dir, e["dir"]); rp = os.path.join(d, "JIT_IDENTITY.json")
        if not os.path.isfile(rp): bad.append(f"{e['dir']}: no JIT_IDENTITY.json"); continue
        r = json.load(open(rp)); theirs = r.get("kernels_sha256_16"); theirs_jit = r.get("kernels_jit_sha256_16")
        row = {"dir": e["dir"], "arch": e.get("arch"), "record_kernels_sha256_16": theirs, "package_kernels_sha256_16": mine, "record_kernels_jit_sha256_16": theirs_jit, "package_kernels_jit_sha256_16": mine_jit, "pin_equal": theirs == mine}
        report.append(row)
        if theirs != mine: bad.append(f"{e['dir']} ({e.get('arch')}): record kernels_sha256_16 {theirs} != the package's {mine} (jit sources {'EQUAL' if theirs_jit == mine_jit else 'differ' if theirs_jit else 'unknown'}: {theirs_jit} vs {mine_jit})")
    if bad: raise RuntimeError("PIN EQUALITY GATE REFUSED: " + "; ".join(bad) + " — re-capture the shipped cache from the package's current kernels.py (k1r4.25 rule)")
    return report
