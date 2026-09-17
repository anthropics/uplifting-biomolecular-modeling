"""The preimport hook — a failing module in the chain is RECORDED by the thread, never raised; start()/join() by path; no torch, no GPU."""
import os, sys, subprocess, textwrap, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "kit")   # the carried kit beside opt/kit_ho (opt/kit: torch/ + cache/)
PRE = os.path.join(ROOT, "torch", "chrombpnet_k1", "_preimport.py")
CHILD = textwrap.dedent("""
import json, importlib.util
spec = importlib.util.spec_from_file_location("_kit_preimport", %r); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.start(modules=("json", "this_module_does_not_exist_xyz", "os"))
st = m.join(timeout=30); print(json.dumps({"errors": list(st["errors"]), "n_ok": len(st["walls_s"]), "line": st["line"]}))
""") % PRE
def test_failing_import_is_recorded_never_raised():
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (r.returncode, r.stderr[-400:])
    st = json.loads([l for l in r.stdout.split("\n") if l.startswith("{")][-1])
    assert st["n_ok"] == 2 and any("this_module_does_not_exist_xyz" in str(e) for e in st["errors"]) and "import errors" in st["line"], st
if __name__ == "__main__":
    test_failing_import_is_recorded_never_raised(); print("test_preimport_hook_cpu: PASS")
