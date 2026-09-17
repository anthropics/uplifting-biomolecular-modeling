"""Minimal pytest-free runner for the design kit's GPU unit tests k/test_ef2_*.py (images without pytest/pip access).

    python -I run_tests_nopytest.py                      every k/test_ef2_*.py module, in name order
    python -I run_tests_nopytest.py test_ef2_trimul ...  the named modules only

Plain ``test_*`` functions; ``@pytest.mark.skipif`` / ``@pytest.mark.parametrize`` (stackable) and ``pytest.skip`` are stubbed here.
A module that does not import is one FAIL (never a silent skip). PASS = exit 0 and ``RESULT passed=N failed=0`` with N > 0.
test_ef2_loop.py and test_ef2_esmc_overlap.py drive the installed cookbook loop and need its pinned weights (HF_HOME); the others
build random-initialised modules of the installed fork and need only a CUDA device."""
import glob, itertools, os, sys, traceback, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# stub a tiny pytest module so the test files import
import importlib.machinery as _ilm
pt = types.ModuleType("pytest")
pt.__spec__ = _ilm.ModuleSpec("pytest", None)


class _Mark:
    def skipif(self, cond, reason=""):
        def deco(f):
            f._skip = bool(getattr(f, "_skip", False)) or bool(cond); return f
        return deco

    def parametrize(self, names, values, ids=None):
        def deco(f):
            f._params = [(names, values)] + list(getattr(f, "_params", [])); return f    # decorators apply bottom-up; keep declaration order
        return deco


pt.mark = _Mark()


def _skip(msg=""):
    raise RuntimeError("SKIP: " + msg)


pt.skip = _skip
pt.fixture = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
sys.modules["pytest"] = pt

wanted = [a[:-3] if a.endswith(".py") else a for a in sys.argv[1:]]
modules = wanted or sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(HERE, "test_ef2_*.py")))
n_pass = n_fail = 0
for modname in modules:
    print(f"==== {modname}", flush=True)
    try:
        T = __import__(modname)
    except Exception as e:  # noqa: BLE001
        n_fail += 1; print("FAIL", modname, "import", type(e).__name__, str(e)[:400], flush=True); traceback.print_exc(); continue
    for name, fn in sorted(vars(T).items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        if getattr(fn, "_skip", False):
            print("SKIP", name); continue
        combos = [dict()]
        for names, values in getattr(fn, "_params", []):
            keys = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
            rows = [dict(zip(keys, v if (isinstance(v, (tuple, list)) and len(keys) > 1) else (v,))) for v in values]
            combos = [dict(a, **b) for a, b in itertools.product(combos, rows)]
        for kw in combos:
            try:
                fn(**kw); n_pass += 1; print("PASS", name, kw, flush=True)
            except Exception as e:  # noqa: BLE001
                n_fail += 1; print("FAIL", name, kw, type(e).__name__, str(e)[:400], flush=True); traceback.print_exc()
print(f"RESULT passed={n_pass} failed={n_fail}")
sys.exit(1 if n_fail or not n_pass else 0)
