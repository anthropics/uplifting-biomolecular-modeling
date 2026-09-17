"""CPU tests (no GPU, no TF): (a) the ComputeCache 'index' member rewritten -> HIT (named, never a rewrite); (c) resolve() under the DET recipe env
-> native dilation OFF with the reason (an explicit ON overridden, loudly); (e) a released writer that renames its partial onto the final name on the
plain sentinel leaves 0 files after the fail-fast (the abort sentinel + the release's unlink of the final name it created)."""
import os, sys, time, json, tempfile, hashlib, subprocess, textwrap
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import chrombpnet_fastkit as kit
from chrombpnet_fastkit import fastdefault as fd
def _mk(d, files):
    members = {}
    for rel, data in files.items():
        p = os.path.join(d, rel); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(data); members[rel] = hashlib.sha256(data).hexdigest()
    return members
def test_index_member_rewritten_is_not_a_rewrite():
    d = tempfile.mkdtemp(); members = _mk(d, {"0/0/aa11": b"kernel-1", "index": b"index-at-capture"}); open(os.path.join(d, "index"), "wb").write(b"index-rewritten-by-the-driver")
    w = kit.cache_content_witness(d, members); assert w["verdict"] == "HIT", w; assert w["index_rewritten"] == ["index"], w; assert w["rewritten"] == [], w
def test_native_dilation_off_under_the_det_recipe():
    env = dict(os.environ); env.update({"TF_USE_DEFAULT_CONV_ALGO": "1", "TF_DETERMINISTIC_OPS": "1"})
    code = textwrap.dedent("""
    import sys, json; sys.path.insert(0, %r)
    from chrombpnet_fastkit import fastdefault as fd
    fd.detect_gpu = lambda: ("H100", "NVIDIA H100 80GB HBM3 (the CPU test's stand-in)", "9.0")   # no GPU here: the class is what the DET detection is tested against
    r = fd.resolve(kit_root=%r); print(json.dumps({"nd": r["native_dilation"], "why": r["native_dilation_reason"]}))
    """) % (HERE, os.path.dirname(HERE))
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=120)
    lines = [l for l in out.stdout.split("\n") if l.startswith("{")]; assert len(lines) == 1, out.stdout[-800:] + out.stderr[-800:]
    a = json.loads(lines[0])
    assert a["nd"] is False and "DET recipe" in a["why"] and "requested ON (the class default) overridden" in a["why"], a
def test_released_writer_leaves_no_output():
    d = tempfile.mkdtemp(); final = os.path.join(d, "pred_chrombpnet.bw")
    child = textwrap.dedent("""
    import os, sys, time, multiprocessing as mp
    sys.path.insert(0, %r)
    import chrombpnet_fastkit as kit
    final = %r
    def writer(q, final):
        open(final + ".partial", "wb").write(b"partial bigwig bytes")
        while True:
            item = q.get()
            if item is None or item == kit.ABORT_SENTINEL: break
        if item == kit.ABORT_SENTINEL:
            os.unlink(final + ".partial"); return          # the abort path: close WITHOUT the rename
        os.replace(final + ".partial", final)              # the success path: the final name only after a successful close
    ctx = mp.get_context("fork"); q = ctx.Queue(maxsize=4); wp = ctx.Process(target=writer, args=(q, final)); wp.start()
    kit._LIVE_WRITERS.append((q, wp)); kit._register_partial(final); kit._install_fail_fast()
    time.sleep(0.3)
    raise RuntimeError("Random ops require a seed to be set when determinism is enabled (the stock's own load error)")
    """) % (HERE, final)
    r = subprocess.run([sys.executable, "-I", "-c", child], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, (r.returncode, r.stderr[-600:])
    left = sorted(os.listdir(d)); assert left == [], (left, r.stderr[-400:])
def test_released_writer_that_renamed_anyway_is_cleaned():
    d = tempfile.mkdtemp(); final = os.path.join(d, "pred_chrombpnet.bw")
    child = textwrap.dedent("""
    import os, sys, time, multiprocessing as mp
    sys.path.insert(0, %r)
    import chrombpnet_fastkit as kit
    final = %r
    def writer(q, final):
        open(final + ".partial", "wb").write(b"partial bigwig bytes")
        while True:
            item = q.get()
            if item is None or item == kit.ABORT_SENTINEL: break
        os.replace(final + ".partial", final)              # a v0.12.55-form writer: renames on ANY sentinel (the behaviour this test guards against)
    ctx = mp.get_context("fork"); q = ctx.Queue(maxsize=4); wp = ctx.Process(target=writer, args=(q, final)); wp.start()
    kit._LIVE_WRITERS.append((q, wp)); kit._register_partial(final); kit._install_fail_fast()
    time.sleep(0.3)
    raise RuntimeError("boom")
    """) % (HERE, final)
    r = subprocess.run([sys.executable, "-I", "-c", child], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, (r.returncode, r.stderr[-600:])
    left = sorted(os.listdir(d)); assert left == [], (left, r.stderr[-400:])
if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("ok", name)
    print("V01264 CPU TESTS OK")
