"""The instance counter (every creation route, gc seeding, the finder route), the kit build backend template (a fixture kit's regular and
editable wheels carry the .pth in RECORD), every adopting kit's backend copy and pin, the engine-free hygiene of the core."""
import ast
import copy
import glob
import importlib
import os
import pickle
import re
import subprocess
import sys
import textwrap
import zipfile

import pytest

import opt_core
from opt_core import gates, instances, kernels

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(CORE_DIR, "kit_template", "_build_backend.py")
TEMPLATE_GATE = os.path.join(CORE_DIR, "kit_template", "_core_gate.py")   # the pre-import core pin gate every kit carries inside its package

# ------------------------------------------------------------------------------------------------------------ instances


class Model:
    def __init__(self, width=1):
        self.width = width


class Sub(Model):
    pass


def test_counter_on_an_imported_class_seeds_from_gc_and_counts_every_route():
    existing = Model(1)
    first = instances.register_instance_counter(__name__, "Model")
    assert first["method"] == "counted" and first["n"] >= 1 and first["built"] == 0
    assert instances.register_instance_counter(__name__, "Model") == first                   # idempotent
    import inspect
    assert list(inspect.signature(Model).parameters) == ["width"]
    a = Model(2)
    b = copy.deepcopy(a)
    c = pickle.loads(pickle.dumps(a))
    d = Sub(3)
    now = instances.instance_check(__name__, "Model")
    assert now["built"] == 4 and now["n"] == first["n"] + 4 and now["method"] == "counted"
    assert (a.width, b.width, c.width, d.width) == (2, 2, 2, 3)
    del a, b, c, d
    import gc
    gc.collect()
    assert instances.instance_check(__name__, "Model")["n"] == first["n"] and existing.width == 1
    assert instances.counter_note(__name__, "Model") is None


def test_counter_through_the_finder_route(tmp_path, monkeypatch):
    mod = tmp_path / "acme_model.py"
    mod.write_text(textwrap.dedent("""
        class Net:
            def __init__(self, n=0):
                self.n = n
        BUILT_AT_IMPORT = Net(9)
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("acme_model", None)
    before = instances.register_instance_counter("acme_model", "Net")
    assert before == {"n": 0, "method": "none", "built": 0}
    finder = instances._entry("acme_model", "Net")["finder"]
    assert finder is sys.meta_path[0]
    import gc
    scans = []
    monkeypatch.setattr(gc, "get_objects", lambda *a, **k: scans.append(1) or [])       # the finder route never scans the heap at import
    m = importlib.import_module("acme_model")
    assert finder not in sys.meta_path and scans == []
    assert instances.instance_check("acme_model", "Net") == {"n": 0, "method": "counted", "built": 0}      # the kits' route: the instance the module body built is not seen
    kept = m.Net(1)
    assert instances.instance_check("acme_model", "Net") == {"n": 1, "method": "counted", "built": 1} and kept.n == 1
    sys.modules.pop("acme_model", None)
    assert instances.instance_check("acme_model", "Net") == {"n": 0, "method": "none", "built": 0}


class Slotted:
    __slots__ = ("x",)


def test_counter_without_weak_references_falls_back_to_gc():
    instances.register_instance_counter(__name__, "Slotted")
    s = Slotted()
    r = instances.instance_check(__name__, "Slotted")
    assert r["method"] == "gc" and r["n"] >= 1 and r["built"] == 1
    assert instances.counter_note(__name__, "Slotted") == "Slotted instances are counted by a gc scan: its instances take no weak references"
    del s


class Watched:
    pass


def test_a_dangling_weakref_proxy_in_the_gc_heap_is_not_an_instance_of_anything():
    """A tracing/compilation layer elsewhere in the process (dynamo's guards are the observed case) can leave a ``weakref.proxy``
    reachable via ``gc.get_objects()`` after its referent is already collected: ``isinstance()`` on that proxy raises
    ``ReferenceError``, not a clean False -- both the gc-seeding scan and the gc-fallback count must survive one in the heap
    without crashing or over/under-counting anything else."""
    import gc
    import weakref

    target = Watched()
    proxy = weakref.proxy(target)
    del target
    gc.collect()
    with pytest.raises(ReferenceError):
        isinstance(proxy, Watched)                                        # the underlying behaviour this test protects against

    live = Watched()                                                       # one real, live instance -- must still be seen correctly
    first = instances.register_instance_counter(__name__, "Watched")       # the gc-seeding scan walks a heap that includes `proxy`
    assert first["n"] >= 1 and first["method"] == "counted"
    assert instances.instance_check(__name__, "Watched")["n"] == first["n"]
    del live, proxy


# ------------------------------------------------------------------------------------------------------------ the build backend template

FIXTURE_PYPROJECT = """[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "_build_backend"
backend-path = ["."]

[project]
name = "acme_opt"
version = "0.0.1"

[tool.setuptools]
packages = ["acme_opt"]
"""


def _template():
    """The template module loaded from its file (it is not an importable package of the core)."""
    spec = importlib.util.spec_from_file_location("_kit_template_build_backend", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_fixture_kit(root):
    """A minimal kit: the package, the .pth, the backend copy — the shape every kit's opt/ has."""
    os.makedirs(os.path.join(root, "acme_opt"))
    open(os.path.join(root, "acme_opt", "__init__.py"), "w").write("")
    open(os.path.join(root, "acme_opt", "_autoload.py"), "w").write("import os\nMARK = os.environ.get('ACME_OPT')\n")
    open(os.path.join(root, "acme_opt_autoload.pth"), "w").write(_template().pth_text("acme_opt", "ACME_OPT", "acme-opt"))   # generated, never hand-written
    open(os.path.join(root, "pyproject.toml"), "w").write(FIXTURE_PYPROJECT)
    with open(TEMPLATE, "rb") as src, open(os.path.join(root, "_build_backend.py"), "wb") as dst:
        dst.write(src.read())
    return root



def _wheel_tooling_missing():
    """A named reason when this interpreter cannot build the fixture wheel (the template drives setuptools.build_meta): no setuptools, or
    neither the 'wheel' package nor a setuptools that vendors bdist_wheel (>= 70.1). None when the tooling is present."""
    if importlib.util.find_spec("setuptools") is None:
        return "setuptools needed to build the fixture wheel"
    if importlib.util.find_spec("wheel") is None:
        try:
            if importlib.util.find_spec("setuptools.command.bdist_wheel") is None:
                return "wheel-build tooling absent: no 'wheel' package and this setuptools has no vendored bdist_wheel"
        except (ImportError, ValueError):
            return "wheel-build tooling absent: no 'wheel' package and this setuptools has no vendored bdist_wheel"
    return None


WHEEL_TOOLING_MISSING = _wheel_tooling_missing()

def _record_of(wheel):
    with zipfile.ZipFile(wheel) as z:
        record = [n for n in z.namelist() if n.endswith(".dist-info/RECORD")][0]
        return z.read(record).decode().splitlines(), z.namelist()


@pytest.mark.skipif(WHEEL_TOOLING_MISSING is not None, reason=WHEEL_TOOLING_MISSING or "")
def test_template_builds_regular_and_editable_wheels_with_the_pth(tmp_path):
    kit = make_fixture_kit(os.path.abspath(str(tmp_path / "opt")))
    out = os.path.abspath(str(tmp_path / "wheels"))
    os.makedirs(out)
    driver = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build_driver.py")
    r = subprocess.run([sys.executable, driver, kit, out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    regular, editable = [ln[len("WHEEL:"):] for ln in r.stdout.splitlines() if ln.startswith("WHEEL:")]
    for name in (regular, editable):
        lines, names = _record_of(os.path.join(out, name))
        assert "acme_opt_autoload.pth" in names
        pth_lines = [ln for ln in lines if ln.startswith("acme_opt_autoload.pth,sha256=")]
        assert len(pth_lines) == 1 and pth_lines[0].endswith(",%d" % len(_template().pth_text("acme_opt", "ACME_OPT", "acme-opt")))
        assert lines[-1].endswith(".dist-info/RECORD,,")                               # RECORD's own entry stays last
    regular_names = _record_of(os.path.join(out, regular))[1]
    assert "acme_opt/_autoload.py" in regular_names and "acme_opt/__init__.py" in regular_names


@pytest.mark.skipif(WHEEL_TOOLING_MISSING is not None, reason=WHEEL_TOOLING_MISSING or "")
def test_template_builds_plain_wheels_for_a_kit_without_a_pth(tmp_path):
    """An argv-only kit ships no autoload .pth (a .pth would load a kit module in the stock child): the same template builds it plain."""
    kit = make_fixture_kit(os.path.abspath(str(tmp_path / "opt")))
    os.remove(os.path.join(kit, "acme_opt_autoload.pth"))
    out = os.path.abspath(str(tmp_path / "wheels"))
    os.makedirs(out)
    driver = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build_driver.py")
    r = subprocess.run([sys.executable, driver, kit, out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    for name in [ln[len("WHEEL:"):] for ln in r.stdout.splitlines() if ln.startswith("WHEEL:")]:
        lines, names = _record_of(os.path.join(out, name))
        assert not any(n.endswith("_autoload.pth") for n in names) and not any("_autoload.pth," in ln for ln in lines)   # setuptools' own __editable__ .pth stays
        assert lines[-1].endswith(".dist-info/RECORD,,")


@pytest.mark.skipif(importlib.util.find_spec("setuptools") is None, reason="setuptools needed: the template imports it before the .pth check")
def test_template_refuses_a_kit_with_two_pths(tmp_path):
    kit = make_fixture_kit(os.path.abspath(str(tmp_path / "opt")))
    with open(os.path.join(kit, "other_autoload.pth"), "w") as fh:
        fh.write(_template().pth_text("acme_opt", "ACME_OPT", "acme-opt"))
    driver = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build_driver.py")
    r = subprocess.run([sys.executable, driver, kit, str(tmp_path / "wheels")], capture_output=True, text=True)
    assert r.returncode != 0 and "expected at most one *_autoload.pth beside" in r.stderr


@pytest.mark.skipif(WHEEL_TOOLING_MISSING is not None, reason=WHEEL_TOOLING_MISSING or "")
def test_template_refuses_a_hand_written_pth(tmp_path):
    kit = make_fixture_kit(os.path.abspath(str(tmp_path / "opt")))
    open(os.path.join(kit, "acme_opt_autoload.pth"), "w").write("import acme_opt._autoload\n")
    driver = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build_driver.py")
    r = subprocess.run([sys.executable, driver, kit, str(tmp_path / "wheels")], capture_output=True, text=True)
    assert r.returncode != 0 and "does not start with the opt_core autoload guard header" in r.stderr


@pytest.mark.skipif(importlib.util.find_spec("setuptools") is None, reason="setuptools needed: the template imports it")
def test_pth_guard_in_a_real_site_dir(tmp_path):
    """The generated .pth, executed by site.py in a child: a set mode with the kit package absent ends the process with the NOT ACTIVE line and
    code 3 (site.py alone would swallow the ImportError and run stock); unset / off leave the process alone; an importable package runs its
    _autoload; a SystemExit from the kit's own refusal becomes the exit code."""
    t = _template()
    text = t.pth_text("acme_opt", "ACME_OPT", "acme-opt")
    assert t.pth_fields(text) == ("acme_opt", "ACME_OPT", "acme-opt", 3)
    assert text.splitlines()[0] == "# opt_core autoload guard: package=acme_opt env=ACME_OPT tag=acme-opt exit=3" and len(text.splitlines()) == 2
    with pytest.raises(ValueError):
        t.pth_fields("import acme_opt._autoload\n")
    with pytest.raises(ValueError):
        t.pth_text("acme-opt", "ACME_OPT", "acme-opt")                         # the package must be an identifier
    site_dir = tmp_path / "site"; site_dir.mkdir()
    (site_dir / "acme_opt_autoload.pth").write_text(text)

    def child(env_extra, pkg_dir=None):
        env = {k: v for k, v in os.environ.items() if k not in ("ACME_OPT", "PYTHONPATH")}
        env.update(env_extra)
        if pkg_dir:
            env["PYTHONPATH"] = str(pkg_dir)
        p = subprocess.run([sys.executable, "-S", "-c", f"import site; site.addsitedir({str(site_dir)!r}); print('alive')"],
                           env=env, capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout.strip(), p.stderr

    rc, out, err = child({"ACME_OPT": "fast"})
    assert (rc, out) == (3, "") and "[acme-opt] NOT ACTIVE: acme_opt._autoload is not importable (ModuleNotFoundError: No module named 'acme_opt') under ACME_OPT=fast (exit 3)" in err
    assert child({})[:2] == (0, "alive") and child({"ACME_OPT": "off"})[:2] == (0, "alive")     # stock untouched
    pkg = tmp_path / "pkgs"; (pkg / "acme_opt").mkdir(parents=True); (pkg / "acme_opt" / "__init__.py").write_text("")
    (pkg / "acme_opt" / "_autoload.py").write_text("import sys\nsys.stderr.write('autoload ran\\n')\n")
    rc, out, err = child({"ACME_OPT": "fast"}, pkg)
    assert (rc, out) == (0, "alive") and "autoload ran" in err
    (pkg / "acme_opt" / "_autoload.py").write_text("import sys\nsys.stderr.write('[acme-opt] NOT ACTIVE: refusing\\n'); sys.exit(3)\n")
    rc, out, err = child({"ACME_OPT": "fast"}, pkg)
    assert (rc, out) == (3, "") and "[acme-opt] NOT ACTIVE: refusing" in err


def test_template_mutates_no_sys_path():
    tree = ast.parse(open(TEMPLATE).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr in ("path", "meta_path", "modules", "path_hooks"):
            raise AssertionError(f"the build backend touches sys.{node.attr}")
    names = {n.names[0].name for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))} | {n.module for n in tree.body if isinstance(n, ast.ImportFrom)}
    assert "sys" not in names and "opt_core" not in names


# ------------------------------------------------------------------------------------------------------------ every adopting kit's copy and pin

RELEASE_TREE = os.path.dirname(os.path.dirname(CORE_DIR))          # <tree>/common/opt_core -> <tree>


def adopting_kits():
    """The kits beside common/ whose opt/pyproject.toml carries a [tool.opt_core] pin (an installed wheel has no tree: none)."""
    out = []
    for pp in sorted(glob.glob(os.path.join(RELEASE_TREE, "*", "opt", "pyproject.toml"))):
        if gates.read_pin_table(pp):
            out.append(pp)
    return out


def test_every_adopting_kit_carries_the_template_byte_for_byte_and_pins_this_core():
    kits = adopting_kits()
    if not kits:
        pytest.skip("no adopting kit beside common/ (installed wheel, or none adopted yet)")
    template_sha = gates.sha256_file(TEMPLATE)
    gate_sha = gates.sha256_file(TEMPLATE_GATE)
    bad = []
    for pp in kits:
        opt = os.path.dirname(pp)
        copy_path = os.path.join(opt, "_build_backend.py")
        if not os.path.isfile(copy_path) or gates.sha256_file(copy_path) != template_sha:
            bad.append(f"{copy_path}: not the template")
        gate_copies = sorted(glob.glob(os.path.join(opt, "*", "_core_gate.py")))          # ONE copy, inside the kit's package, byte-identical
        if len(gate_copies) != 1 or gates.sha256_file(gate_copies[0]) != gate_sha:
            bad.append(f"{opt}: _core_gate.py copies {gate_copies or 'none'} — exactly one byte-identical copy of kit_template/_core_gate.py inside the package is required")
        p = gates.core_pin(pp)
        if "package_tree_sha256" in gates.read_pin_table(pp):
            bad.append(f"{pp}: [tool.opt_core] carries package_tree_sha256 — the pin is path + minimum version only")
        if gates.version_tuple(p.get("version")) > gates.version_tuple(opt_core.__version__):
            bad.append(f"{pp}: pins opt_core >= v{p.get('version')}, this core is v{opt_core.__version__}")
        if os.path.normpath(p["abs_path"]) != os.path.normpath(CORE_DIR):
            bad.append(f"{pp}: path {p['path']} resolves to {p['abs_path']}, not {CORE_DIR}")
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------------------------------------------------ hygiene


def engine_tokens():
    """Engine names derived from the tree beside common/ (directory names, kit package names, kit tags) — never listed in the core."""
    tokens = set()
    for d in glob.glob(os.path.join(RELEASE_TREE, "*")):
        name = os.path.basename(d)
        if not os.path.isdir(d) or name == "common" or name.startswith("."):
            continue
        if not os.path.isdir(os.path.join(d, "opt")):
            continue
        tokens.add(name.lower())
        for pkg in glob.glob(os.path.join(d, "opt", "*_opt")):
            tokens.add(os.path.basename(pkg).lower())
            tokens.add(os.path.basename(pkg).lower().replace("_opt", "-opt"))
    return sorted(t for t in tokens if len(t) >= 4)



def mentions(token: str, text: str) -> bool:
    """``token`` occurs in ``text`` as a WORD — not inside another identifier or word (a kit named like a substring of an ordinary word is not a
    mention). The one matcher for every engine-name scan (this suite and the seam tests' engine-free checks)."""
    return re.search(r"(?<![a-z0-9_])" + re.escape(token.lower()) + r"(?![a-z0-9_])", text.lower()) is not None


SKIP_DIRS = ("__pycache__", ".pytest_cache", "build", "dist")   # build artefacts of an in-tree wheel build: never house-written source


def core_sources():
    """The house-written files of the core: everything but the carried kernel bytes and their carry manifests (which name their kit)."""
    from opt_core import kernels
    carried = {os.path.join(kernels.KERNELS_DIR, f) for f in kernels.carried_files()}
    for root, dirs, files in os.walk(CORE_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        if os.path.abspath(root) == os.path.abspath(kernels.META_DIR):
            continue                                             # a carry's metadata may need an engine word (one is an adapter named after an engine)
        for f in files:
            path = os.path.join(root, f)
            if f.endswith((".py", ".md", ".toml", ".json", ".txt")) and path not in carried:
                yield path


def test_core_names_no_engine():
    tokens = engine_tokens()
    if not tokens:
        pytest.skip("no engine tree beside common/")
    hits = []
    listed_debt = os.path.join(CORE_DIR, "tests", "test_selfcontained.py")   # its KNOWN_* tables name today's engine-named files by path (lists that only shrink)
    from tests.test_selfcontained import KNOWN_ENGINE_MENTIONS               # (file, engine) pairs known today — the list only shrinks
    stale = []
    for path in core_sources():
        if os.path.abspath(path) == os.path.abspath(listed_debt):
            continue
        rel = os.path.relpath(path, CORE_DIR)
        known = set(KNOWN_ENGINE_MENTIONS.get(rel, ()))
        text = open(path, encoding="utf-8").read().lower()
        seen = set()
        for t in tokens:
            if mentions(t, text):                                        # the engine name as a word
                seen.add(t)
                if t not in known:
                    hits.append(f"{rel}: {t}")
        stale += [f"{rel}: {t} (KNOWN_ENGINE_MENTIONS row without a mention: delete it)" for t in known - seen if t in tokens]
    assert not hits, "\n".join(hits)
    assert not stale, "\n".join(stale)


def test_core_modules_import_only_the_standard_library_at_module_level():
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:
        pytest.skip("sys.stdlib_module_names needs python 3.10")
    from tests.test_selfcontained import framework_row                       # the ONE table of framework files beyond the carried kernels
    carried = {"kernels/" + f for f in kernels.carried_files()}                  # carried kernel bytes: the framework's own code, exempt
    bad = []
    pkg = os.path.join(CORE_DIR, "opt_core")
    for path in sorted(glob.glob(os.path.join(pkg, "**", "*.py"), recursive=True)):
        rel = os.path.relpath(path, pkg).replace(os.sep, "/")
        if rel in carried or framework_row(rel.split("/")) is not None or "/tests/" in "/" + rel:
            continue
        for node in ast.parse(open(path).read()).body:
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                names = [node.module.split(".")[0]]
            else:
                continue
            for n in names:
                if n not in stdlib and n != "opt_core":
                    bad.append(f"{rel}: {n}")
    assert not bad, bad
