"""CPU tests (no GPU, no TF): (a) resolve() on a stock-route class never imports/probes the K1 stack (k1_stack_import_and_probe 0.0, the probe
'skipped'); on a K1-default class the probe runs (here: torch absent -> the TF route with the missing list)."""
import os, sys, json, subprocess, textwrap
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); KIT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "kit")   # the carried kit (opt/kit: torch/ + cache/) = fastdefault.resolve's kit_root, as pred_bw_fast.py's _BASE_KIT
def _resolve_in_child(cls, extra_env=None):
    env = dict(os.environ); env.update(extra_env or {}); env["CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE"] = cls   # the stated test hook: the class with its own compute cap
    code = textwrap.dedent("""
    import sys, json; sys.path.insert(0, %r)
    from chrombpnet_fastkit import fastdefault as fd
    assert fd.detect_gpu()[0] == %r and fd.detect_gpu()[2] == fd._CLASS_CC[%r], fd.detect_gpu()
    r = fd.resolve(kit_root=%r)
    print("RES " + json.dumps({"forward": r["forward"], "k1_wanted": r["k1_wanted"], "probe": r["torch_stack"].get("probe"), "present": r["torch_stack"].get("present"), "fixed": r["fixed_cost_s"], "why": r["forward_reason"][:200]}))
    """) % (HERE, cls, cls, KIT)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=180)
    lines = [l for l in out.stdout.split("\n") if l.startswith("RES ")]; assert lines, out.stdout[-600:] + out.stderr[-900:]
    return json.loads(lines[-1][4:])
def test_no_probe_on_a_stock_route_class():
    r = _resolve_in_child("A100", {"TF_DETERMINISTIC_OPS": "1"}); assert r["forward"] == "tf_function" and r["k1_wanted"] is False, r   # A100 under the recipe: stock's own graph — K1 is not in that lever set, nothing to probe or refuse
    assert r["fixed"]["k1_stack_import_and_probe"] == 0.0 and "skipped" in str(r["probe"]), r
def test_a100_shipped_numerics_wants_k1():
    r = _resolve_in_child("A100"); assert r["k1_wanted"] is True and r["forward"] in ("k1", "tf_function"), r          # A100 at shipped numerics: K1 (tensor-core convs) is in the lever set — probed; forward=tf_function here means the entry script refuses by name
def test_probe_runs_on_a_k1_default_class():
    r = _resolve_in_child("H100"); assert "skipped" not in str(r["probe"]), r          # the decision needed the stack: probed (torch absent here -> the TF route)
    assert r["forward"] in ("k1", "tf_function") and r["k1_wanted"] is True, r        # K1 is in H100's lever set: forward=tf_function here means the entry script refuses the mode by name
if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("ok", name)
    print("V01266 CPU TESTS OK")
