#!/usr/bin/env python
"""CPU test of exit_fast(): (1) every output and the stamp are re-opened, non-empty and fsync'd before the exit; (2) a missing or empty output or
a truncated stamp raises before any exit (rc 1); (3) K1_EXIT_TEARDOWN=1 returns and the interpreter exits normally (atexit runs); (4) the fast
path exits rc 0 with stdout flushed and no atexit. Each case runs in a child interpreter."""
import os, sys, json, subprocess, tempfile, textwrap
here = os.path.dirname(os.path.abspath(__file__)); env0 = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=here + os.pathsep + os.environ.get("PYTHONPATH", ""))
def run(code, **env):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=120, env=dict(env0, **env))
d = tempfile.mkdtemp(); st = os.path.join(d, "stamp.json"); json.dump({"ok": True}, open(st, "w"))
outs = [os.path.join(d, n) for n in ("pred.bw", "pred.h5", "pred.bed", "pred_metrics.json")]
for p in outs: open(p, "wb").write(b"x" * 100)
code = f"""
import atexit, sys, os
atexit.register(lambda: print("ATEXIT_RAN", flush=True))
synced = []; _f = os.fsync; os.fsync = lambda fd: (synced.append(os.readlink(f"/proc/self/fd/{{fd}}")), _f(fd))
from chrombpnet_k1._exit import exit_fast
print("BEFORE", flush=False)
r = exit_fast({st!r}, outputs={outs!r}); print("AFTER", r, "SYNCED", sorted(os.path.basename(p) for p in synced), flush=True)
"""
# (4) + (1): the fast path — rc 0, stdout flushed, no atexit; the fsync calls are observed on the debug arm (the fast arm _exits before it can print them)
r = run(code); assert r.returncode == 0 and "BEFORE" in r.stdout and "ATEXIT_RAN" not in r.stdout and "AFTER" not in r.stdout, (r.returncode, r.stdout[-200:], r.stderr[-300:]); cases = 1
# (3) + (1): the debug arm returns; every output + the stamp fsync'd (5 files), atexit runs
r = run(code, K1_EXIT_TEARDOWN="1"); assert r.returncode == 0 and "AFTER teardown" in r.stdout and "ATEXIT_RAN" in r.stdout and "SYNCED ['pred.bed', 'pred.bw', 'pred.h5', 'pred_metrics.json', 'stamp.json']" in r.stdout, (r.returncode, r.stdout[-300:]); cases += 1
# (2) a missing output raises before any exit (rc 1 via the excepthook); an empty output raises; a truncated stamp raises
os.unlink(outs[1]); r = run(code); assert r.returncode != 0 and "AFTER" not in r.stdout and "missing or empty" in r.stderr, (r.returncode, r.stderr[-300:]); cases += 1
open(outs[1], "wb").close(); r = run(code); assert r.returncode != 0 and "missing or empty" in r.stderr, r.stderr[-300:]; cases += 1
open(outs[1], "wb").write(b"x"); open(st, "w").write('{"ok": tr'); r = run(code); assert r.returncode != 0 and "AFTER" not in r.stdout and ("JSONDecodeError" in r.stderr or "Expecting" in r.stderr), (r.returncode, r.stderr[-300:]); cases += 1
print(f"PASS test_exit_fast_cpu: {cases} cases (fast path rc 0/flushed/no atexit; debug arm returns with every output + the stamp fsync'd; a missing/empty output or a truncated stamp raises)")
