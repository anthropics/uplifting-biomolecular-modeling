"""Every carried kernel is routable TOP-LEVEL by ``opt_core.kernels.route(name)``; kits load the core copy either (a) by bare name with the kernels
directory on ``sys.path`` or (b) under a top-level ALIAS through ``importlib.util.spec_from_file_location(alias, <pkg>/__init__.py,
submodule_search_locations=[<pkg dir>])`` (the RouteFinder form).  Under either name a relative import that climbs beyond the package root raises
``ImportError: attempted relative import beyond top-level package`` -- at import or, worse, lazily inside a first-serve / probe / stamp path (the
0.5.200.0 regression).  Rules held here, on the CPU:
(1) STATIC: no routable kernel package carries a climbing import beyond its root; only the provider FACE packages (kits import those by their absolute
    ``opt_core`` path and never route them) may, and they are listed;
(2) DYNAMIC, both forms, every non-face routable name (packages and single-file modules), each in a FRESH interpreter: the import (and for packages every
    direct submodule) never raises a relative-import error -- environment gaps (no triton / jax / GPU here, a required export unset) are tolerated;
(3) fpf_trimul_v4's stamp facts / read / write run under both names."""
import ast
import json
import os
import subprocess
import sys
import textwrap

import pytest

from opt_core import kernels as KS

KERNELS_DIR = os.path.dirname(os.path.abspath(KS.__file__))
OPT_CORE_ROOT = os.path.dirname(os.path.dirname(KERNELS_DIR))                 # the directory holding the opt_core package
FACES = {"apb", "ln", "pallas", "transition", "triattn", "triattn_xla", "trimul", "trimul_xla"}   # provider faces: absolute opt_core path only (never routed by kits)
RELMSG = ("attempted relative import beyond top-level package", "attempted relative import with no known parent package")


def _routable():
    return [(n, KS.sums(n)["kind"]) for n in KS.names()]


def _climbing_imports(name):
    root = KS.carried_path(name)
    hits = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            depth = len(rel.split(os.sep))                                  # pkg/mod.py, pkg/__init__.py -> 1; pkg/sub/mod.py -> 2 ...
            try:
                tree = ast.parse(open(p, encoding="utf-8").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level and node.level > depth:
                    hits.append((rel, node.lineno, "." * node.level + (node.module or "")))
    return hits


def test_no_routable_kernel_package_climbs_beyond_its_root_with_a_relative_import():
    pk = [n for n, k in _routable() if k == "package"]
    assert {"fpf_trimul_v4", "fpf_transition_v2"} <= set(pk), pk
    offenders = {n: h for n in pk for h in [_climbing_imports(n)] if h}
    bad = {n: h for n, h in offenders.items() if n not in FACES}
    assert not bad, "kernel packages routable top-level must not climb beyond their root: %s" % bad
    assert not [h for h in offenders.get("trimul", []) if h[0].startswith("native" + os.sep)], offenders   # the stamp plumbing (trimul/native) holds the rule too; the faces themselves keep in-core relative imports (never routed)


CHILD = textwrap.dedent('''
    import ast, importlib, importlib.util, json, os, sys, traceback
    KERNELS_DIR, OPT_CORE_ROOT, NAME, KIND, FORM, EXTRA = json.loads(os.environ.pop("OPT_CORE_ROUTE_TEST"))
    RELMSG = ("attempted relative import beyond top-level package", "attempted relative import with no known parent package")
    sys.argv = ["route-test"]                            # some carried submodules are command-line tools that parse argv at import

    def is_script(path):
        """A submodule that acts at import (parses argv / runs a main outside a __main__ guard) is a tool, not importable library code: skipped."""
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except SyntaxError:
            return True
        for st in tree.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                continue
            if isinstance(st, ast.If) and "__name__" in ast.dump(st.test):
                continue
            src = ast.dump(st)
            if "parse_args" in src or "sys.argv" in src or "attr='exit'" in src:
                return True
        return False
    sys.path.insert(0, KERNELS_DIR)                      # form (a): `import NAME` is the top-level package / module (what kernels.route's finder yields)
    if OPT_CORE_ROOT not in sys.path:
        sys.path.append(OPT_CORE_ROOT)                   # the shared core stays importable by its absolute name, as in every kit image

    def bail(e, where):
        msg = "%s: %s" % (type(e).__name__, e)
        if isinstance(e, ImportError) and any(k in msg for k in RELMSG):
            print("RELATIVE-IMPORT at %s: %s" % (where, msg)); traceback.print_exc(); sys.exit(3)
        print("ENV at %s: %s" % (where, msg[:300])); sys.exit(0)            # no triton / jax / GPU / export here: environment, not the rule

    def load_alias(alias):
        """The RouteFinder form: the core copy executed under a top-level alias."""
        if KIND == "package":
            pdir = os.path.join(KERNELS_DIR, NAME)
            spec = importlib.util.spec_from_file_location(alias, os.path.join(pdir, "__init__.py"), submodule_search_locations=[pdir])
        else:
            spec = importlib.util.spec_from_file_location(alias, os.path.join(KERNELS_DIR, NAME + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[alias] = m
        spec.loader.exec_module(m)
        return m
    top = NAME if FORM == "bare" else "tp_" + NAME
    try:
        m = importlib.import_module(NAME) if FORM == "bare" else load_alias(top)
    except Exception as e:
        bail(e, "import %s (%s)" % (top, FORM))
    want = os.path.join(KERNELS_DIR, NAME) if KIND == "package" else os.path.join(KERNELS_DIR, NAME + ".py")
    assert m.__name__ == top and os.path.abspath(m.__file__).startswith(want), (m.__name__, m.__file__)
    subs = []
    if KIND == "package":                                # every direct submodule under the top-level name (lazy climbing imports live there)
        for fn in sorted(os.listdir(want)):
            if fn.endswith(".py") and fn != "__init__.py" and not fn.startswith("test_") and not is_script(os.path.join(want, fn)):
                sub = top + "." + fn[:-3]
                try:
                    importlib.import_module(sub); subs.append(fn[:-3])
                except Exception as e:
                    msg = "%s: %s" % (type(e).__name__, e)
                    if isinstance(e, ImportError) and any(k in msg for k in RELMSG):
                        print("RELATIVE-IMPORT at %s: %s" % (sub, msg)); sys.exit(3)
                    print("ENV at %s: %s" % (sub, msg[:160]))
    if EXTRA == "v4_stamps":
        G = importlib.import_module(top + ".generic")
        NV = G._stamp_api()
        assert NV is not None and NV.__name__ == "opt_core.kernels.trimul.native", NV
        facts = {"schema": G.PROBE_STAMP_SCHEMA, "package": "fpf_trimul_v4", "version": "x", "digest": G.package_digest(), "C": 128, "D": 128, "bias": False,
                 "in_f32": False, "sizes": [128, 512], "cell": "c", "bars": [G.SAME_CLASS_RMS, G.SAME_CLASS_MAX], "cc": "9.0", "device_name": "T",
                 "driver_version": None, "cuda_version": "13.0", "torch": "2", "triton": "3", "python": "3", "form": FORM}
        assert G.probe_stamp_facts("cpu", 128, 128, False, False, [128]) is None
        assert G.probe_stamp_read(facts) is None
        p = G.probe_stamp_write(facts, {"ok": True})
        assert p and os.path.isfile(p) and G.probe_stamp_read(facts)["_path"] == p, p
        assert NV.stamp_dirs(leaf=G.PROBE_STAMP_LEAF) == [("env", os.path.join(os.environ["OPT_CORE_VERDICT_DIR"], G.PROBE_STAMP_LEAF))]
        print("STAMPS-OK", FORM, p)
    print("OK", top, KIND, "submodules:", ",".join(subs))
''')


def _run_child(name, kind, form, extra="", env_extra=None):
    env = dict(os.environ); env.pop("PYTHONPATH", None); env["KMP_AFFINITY"] = "disabled"
    env.update(env_extra or {})
    env["OPT_CORE_ROUTE_TEST"] = json.dumps([KERNELS_DIR, OPT_CORE_ROOT, name, kind, form, extra])
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=600, env=env)
    return r.returncode, (r.stdout + r.stderr)


CASES = sorted((n, k, f) for n, k in _routable() if n not in FACES for f in ("bare", "alias"))


@pytest.mark.parametrize("name,kind,form", CASES, ids=["%s-%s" % (n, f) for n, k, f in CASES])
def test_routable_kernel_loads_top_level_without_a_relative_import_error(name, kind, form):
    rc, out = _run_child(name, kind, form)
    assert rc != 3 and not any(k in out for k in RELMSG), out[-2000:]
    assert rc == 0, out[-2000:]


@pytest.mark.parametrize("form", ["bare", "alias"])
def test_fpf_trimul_v4_stamp_paths_run_top_level_in_both_forms(form, tmp_path):
    rc, out = _run_child("fpf_trimul_v4", "package", form, extra="v4_stamps", env_extra={"OPT_CORE_VERDICT_DIR": str(tmp_path)})
    assert rc != 3 and not any(k in out for k in RELMSG), out[-2000:]
    if rc == 0 and "ENV at import" in out:
        pytest.skip("fpf_trimul_v4 does not import in this environment: %s" % out.strip()[-200:])
    assert rc == 0 and "STAMPS-OK %s" % form in out, out[-2000:]
