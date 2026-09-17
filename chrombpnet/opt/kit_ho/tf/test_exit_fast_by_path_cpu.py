"""The kit's exit-fast integration BY PATH (torch/chrombpnet_k1/_exit.py is a leaf module; no package import, no torch). A success child
(every output + the stamp on disk) exits 0 WITHOUT running the interpreter's atexit handlers; a child with a missing output leaves with rc != 0 and the
atexit handlers run (every failure path unchanged); K1_EXIT_TEARDOWN=1 keeps the teardown. No TF, no GPU."""
import os, sys, subprocess, tempfile, json, textwrap
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "kit")   # the carried kit beside opt/kit_ho (opt/kit: torch/ + cache/)
EXIT_PY = os.path.join(ROOT, "torch", "chrombpnet_k1", "_exit.py")
CHILD = textwrap.dedent("""
import os, sys, json, atexit, importlib.util
atexit.register(lambda: print("ATEXIT_RAN", flush=True))
d = %r; stamp = os.path.join(d, "pred_kit_stamp.json"); out = os.path.join(d, "pred_chrombpnet.bw")
json.dump({"ok": True}, open(stamp, "w"))
if %r: open(out, "wb").write(b"bigwig bytes")
spec = importlib.util.spec_from_file_location("_kit_exit", %r); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("BEFORE", flush=True)
m.exit_fast(stamp, outputs=[out])
print("AFTER", flush=True)
""")
def _run(write_output, env=None):
    d = tempfile.mkdtemp(); e = dict(os.environ); e.update(env or {})
    return subprocess.run([sys.executable, "-c", CHILD % (d, write_output, EXIT_PY)], capture_output=True, text=True, timeout=60, env=e)
def test_success_exits_0_past_atexit():
    r = _run(True); assert r.returncode == 0 and "BEFORE" in r.stdout and "AFTER" not in r.stdout and "ATEXIT_RAN" not in r.stdout, (r.returncode, r.stdout[-200:], r.stderr[-300:])
def test_missing_output_fails_loud_with_atexit():
    r = _run(False); assert r.returncode != 0 and "AFTER" not in r.stdout and "ATEXIT_RAN" in r.stdout, (r.returncode, r.stdout[-200:], r.stderr[-300:])
def test_debug_arm_keeps_teardown():
    r = _run(True, {"K1_EXIT_TEARDOWN": "1"}); assert r.returncode == 0 and "AFTER" in r.stdout and "ATEXIT_RAN" in r.stdout, (r.returncode, r.stdout[-200:], r.stderr[-300:])
if __name__ == "__main__":
    test_success_exits_0_past_atexit(); test_missing_output_fails_loud_with_atexit(); test_debug_arm_keeps_teardown(); print("test_exit_fast_by_path_cpu: PASS (3 cases)")
