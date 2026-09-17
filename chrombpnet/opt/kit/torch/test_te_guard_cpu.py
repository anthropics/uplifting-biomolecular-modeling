#!/usr/bin/env python
"""CPU test of the typing_extensions guard: (1) an OLD typing_extensions without 'deprecated' already in sys.modules (as TensorFlow leaves it
when imported first) is replaced by the kit's vendored copy before torch is imported; (2) a typing_extensions that has the names is KEPT (nothing
replaced); (3) the guard is idempotent and recorded in TE_GUARD, and the vendored copy is sha-pinned beside VENDOR.json + its licence. Each case
runs in a fresh interpreter (torch must not be pre-imported)."""
import os, sys, subprocess, json, textwrap
here = os.path.dirname(os.path.abspath(__file__)); env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=here + os.pathsep + os.environ.get("PYTHONPATH", ""))
def run(code):
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=300, env=env); return r
# (1) an old typing_extensions simulated: a stub module WITHOUT 'deprecated' loaded first (as TF would leave it)
r = run(f'''
import sys, types, json
stub = types.ModuleType("typing_extensions"); stub.__file__ = "/simulated/image/site-packages/typing_extensions.py"; stub.Protocol = object; stub.Literal = None
sys.modules["typing_extensions"] = stub                      # what TensorFlow's import leaves behind on the image (an old copy, no 'deprecated')
import chrombpnet_k1
te = sys.modules["typing_extensions"]; import torch
print(json.dumps({{"action": chrombpnet_k1.TE_GUARD["action"], "has_deprecated": hasattr(te, "deprecated"), "file": te.__file__, "torch": torch.__version__, "detail": chrombpnet_k1.TE_GUARD["detail"][:160]}}))
''')
out = [l for l in r.stdout.splitlines() if l.startswith("{")]; assert r.returncode == 0 and out, r.stderr[-800:]
j = json.loads(out[-1]); assert j["action"] == "replaced" and j["has_deprecated"] and j["file"].endswith(os.path.join("chrombpnet_k1", "_vendor", "typing_extensions.py")) and "simulated/image" in j["detail"], j
assert "typing_extensions guard: replaced" in r.stdout, r.stdout[-300:]
cases = 1
# (2) a good typing_extensions already loaded -> kept, file unchanged
r = run('''
import sys, json, typing_extensions as te0
import chrombpnet_k1
te = sys.modules["typing_extensions"]; import torch
print(json.dumps({"action": chrombpnet_k1.TE_GUARD["action"], "same": te is te0, "has_deprecated": hasattr(te, "deprecated")}))
''')
out = [l for l in r.stdout.splitlines() if l.startswith("{")]; assert r.returncode == 0 and out, r.stderr[-800:]
j = json.loads(out[-1]); assert j["action"] == "kept" and j["same"] and j["has_deprecated"], j; cases += 1
# (3) idempotent + the apply stamp field exists in the source; the vendored copy is sha-pinned beside a VENDOR.json + LICENSE
r = run('''
import json, chrombpnet_k1
from chrombpnet_k1._te_guard import ensure_typing_extensions
a = ensure_typing_extensions(); b = ensure_typing_extensions(); print(json.dumps({"same": a is b, "action": a["action"]}))
''')
out = [l for l in r.stdout.splitlines() if l.startswith("{")]; assert r.returncode == 0 and out and json.loads(out[-1])["same"], r.stderr[-400:]
v = json.load(open(os.path.join(here, "chrombpnet_k1", "_vendor", "VENDOR.json")))["typing_extensions"]
assert v["version"] == "4.13.2" and os.path.exists(os.path.join(here, "chrombpnet_k1", "_vendor", "typing_extensions.py")) and os.path.exists(os.path.join(here, "chrombpnet_k1", "_vendor", "LICENSE.typing_extensions")); cases += 1
assert '"typing_extensions_guard": dict(TE_GUARD)' in open(os.path.join(here, "chrombpnet_k1", "forward.py")).read(); cases += 1
print(f"PASS test_te_guard_cpu: {cases} cases (an old typing_extensions loaded first is replaced by the kit's vendored 4.13.2 and torch imports; a good one is kept; idempotent; present + licensed; stamped at apply)")
