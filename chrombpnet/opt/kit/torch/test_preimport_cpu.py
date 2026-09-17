#!/usr/bin/env python
"""CPU test of the stage-2 pre-import thread: (1) start() imports the chain on a daemon thread while the main thread keeps running; the modules
that exist land in sys.modules with their walls recorded; (2) a module that fails to import is recorded, not raised; (3) join() at stage 2 records
'overlapped' and the summary line; (4) start() is idempotent. Runs in a fresh interpreter so the chain is not pre-imported."""
import os, sys, subprocess, textwrap, json
here = os.path.dirname(os.path.abspath(__file__)); env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=here + os.pathsep + os.environ.get("PYTHONPATH", ""))
code = """
import sys, time, json, threading
from chrombpnet_k1 import _preimport as P
chain = ("json", "scipy.stats", "matplotlib", "matplotlib.pyplot", "no_such_module_k1_test")
t0 = time.time(); st = P.start(chain); assert P.start(chain) is st                       # idempotent
ticks = 0
while st["done"] is None and time.time() - t0 < 60: ticks += 1; time.sleep(0.005)       # the main thread keeps running
st = P.join(timeout=60)
print(json.dumps({"ticks": ticks, "done": st["done"] is not None, "walls": st["walls_s"], "errors": st["errors"], "overlapped": st["overlapped"], "line": st["line"], "in_sys": {m: m in sys.modules for m in chain}, "main_alive": threading.main_thread().is_alive()}))
"""
r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=180, env=env)
out = [l for l in r.stdout.splitlines() if l.startswith("{")]; assert r.returncode == 0 and out, (r.returncode, r.stderr[-600:])
j = json.loads(out[-1]); cases = 0
assert j["done"] and j["in_sys"]["json"] and j["in_sys"]["matplotlib"], j; cases += 1
assert "no_such_module_k1_test" in j["errors"] and not j["in_sys"]["no_such_module_k1_test"] and "ModuleNotFoundError" in j["errors"]["no_such_module_k1_test"], j["errors"]; cases += 1
assert j["overlapped"] is True and "finished under stage 1 (overlapped)" in j["line"] and "import errors" in j["line"], j["line"]; cases += 1
assert j["ticks"] > 0 and all(m in j["walls"] for m in ("json", "matplotlib")) and all(v >= 0 for v in j["walls"].values()), j; cases += 1
print(f"PASS test_preimport_cpu: {cases} cases (the chain imports on a daemon thread while the main thread runs; a missing module is recorded, never raised; join records overlapped + the R8-close line; idempotent)")
