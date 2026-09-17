"""TF imported first with an OLD typing_extensions (simulated: a stub without `deprecated` in sys.modules) -> the route
never exits 1: the kit swaps in its vendored typing_extensions >= 4.7 before importing torch; where the K1 stack still cannot import, resolve() reports
the TF route with 'K1 route: OFF — <reason>' (the entry script then refuses the mode by name) (the K1 path kept where it imports). No GPU (the class injected)."""
import os, sys, subprocess, textwrap, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "kit")   # the carried kit beside opt/kit_ho (opt/kit: torch/ + cache/)
CHILD = textwrap.dedent("""
import os, sys, types, json
stub = types.ModuleType("typing_extensions"); stub.__file__ = "/image/site-packages/typing_extensions.py"; stub.__version__ = "4.5.0"   # the image's old copy, imported by TF first
sys.modules["typing_extensions"] = stub
sys.path.insert(0, %r)
os.environ["CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE"] = "H100"
from chrombpnet_fastkit import fastdefault as F
r = F.resolve(kit_root=%r)
te = sys.modules.get("typing_extensions")
print(json.dumps({"forward": r["forward"], "reason": r["forward_reason"][:300], "present": r["torch_stack"].get("present"), "import_error": (r["torch_stack"].get("import_error") or "")[:200],
                  "swapped": r["torch_stack"].get("typing_extensions_swapped"), "te_has_deprecated": hasattr(te, "deprecated"), "te_file": getattr(te, "__file__", None)}))
""") % (HERE, ROOT)
def test_tf_first_old_typing_extensions_never_refuses():
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, (p.returncode, p.stderr[-800:])
    r = json.loads([l for l in p.stdout.split("\n") if l.startswith("{")][-1])
    assert r["forward"] in ("k1", "tf_function"), r
    if r["forward"] == "tf_function":
        assert "K1 route: OFF" in r["reason"] or "not importable" in r["reason"], r     # the fallback names the reason on the line
    else:
        assert r["present"] is True, r                                                   # the K1 path kept where it imports
    # the vendored typing_extensions (>= 4.7, carries `deprecated`) replaced the image's old copy for the torch import
    vend = os.path.join(ROOT, "torch", "chrombpnet_k1", "_vendor", "typing_extensions.py")   # the ONE vendored copy = the K1 package's
    if os.path.exists(vend) and r["present"]: assert r["te_has_deprecated"] and r["swapped"], r   # the swap happened on the way to the K1 import
SWAP = textwrap.dedent("""
import os, sys, types, json
stub = types.ModuleType("typing_extensions"); stub.__file__ = "/image/site-packages/typing_extensions.py"; stub.__version__ = "4.5.0"
sys.modules["typing_extensions"] = stub
sys.path.insert(0, %r)
from chrombpnet_fastkit import fastdefault as F
rec = F.swap_typing_extensions(kit_root=%r)
te = sys.modules.get("typing_extensions")
print(json.dumps({"rec": rec, "has_deprecated": hasattr(te, "deprecated"), "file": getattr(te, "__file__", None)}))
""") % (HERE, ROOT)
def test_vendored_typing_extensions_replaces_the_old_copy():
    vend = os.path.join(ROOT, "torch", "chrombpnet_k1", "_vendor", "typing_extensions.py")
    assert os.path.exists(vend), "the K1 package's vendored typing_extensions is missing"
    p = subprocess.run([sys.executable, "-c", SWAP], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-600:]
    r = json.loads([l for l in p.stdout.split("\n") if l.startswith("{")][-1])
    assert r["has_deprecated"] and r["rec"] and r["rec"].get("action") in ("replaced", "provided") and str(r["file"]).startswith(os.path.join(ROOT, "torch", "chrombpnet_k1", "_vendor")), r
if __name__ == "__main__":
    test_tf_first_old_typing_extensions_never_refuses(); test_vendored_typing_extensions_replaces_the_old_copy(); print("test_k1_import_fallback_cpu: PASS")
