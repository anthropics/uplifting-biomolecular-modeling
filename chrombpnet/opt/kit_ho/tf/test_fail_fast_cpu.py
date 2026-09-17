"""The writer-aware fail-fast hook — a raise after the writer child's start must exit 1 within seconds with 0 live children (never a
hang), and leave NO partial output behind: the writer's bigWig is written under a temp name renamed on success; the
hook unlinks registered partials. No TF, no GPU."""
import os, sys, subprocess, time, textwrap, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
def _child(partial):
    return textwrap.dedent("""
import os, sys, time, multiprocessing as mp
sys.path.insert(0, %r)
import chrombpnet_fastkit as kit
def blocker(q, p):
    open(p, "wb").write(b"partial bigwig bytes")     # the writer child had opened the output (N52)
    while True:
        x = q.get()
        if x is None: return
ctx = mp.get_context("fork"); q = ctx.Queue(maxsize=4); wp = ctx.Process(target=blocker, args=(q, %r)); wp.start()
kit._LIVE_WRITERS.append((q, wp)); kit._PARTIAL_OUTPUTS.append(%r); kit._install_fail_fast()
time.sleep(0.5)
print("child pid", wp.pid, flush=True)
raise RuntimeError("Random ops require a seed to be set when determinism is enabled (the stock's own load error, reproduced)")
""") % (HERE, partial, partial)
def test_raise_after_writer_start_exits_1_fast_and_leaves_no_partial():
    d = tempfile.mkdtemp(); partial = os.path.join(d, "pred_chrombpnet.bw.partial")
    t = time.time(); p = subprocess.run([sys.executable, "-c", _child(partial)], capture_output=True, text=True, timeout=60); dt = time.time() - t
    assert p.returncode == 1, (p.returncode, p.stderr[-500:])
    assert dt < 20, dt
    assert "FAIL-FAST" in p.stderr and "1 writer child(ren) released" in p.stderr, p.stderr[-600:]
    pid = int([l for l in p.stdout.split("\n") if l.startswith("child pid")][0].split()[-1])
    time.sleep(0.5)
    alive = os.path.exists("/proc/%d" % pid) and open("/proc/%d/stat" % pid).read().split()[2] != "Z"
    assert not alive, "the writer child is still alive"
    assert not os.path.exists(partial), "a partial output was left behind (N52)"
    assert not os.path.exists(partial[:-len(".partial")]), "the final output name must not exist after a failure"
if __name__ == "__main__":
    test_raise_after_writer_start_exits_1_fast_and_leaves_no_partial(); print("test_fail_fast_cpu: PASS")
