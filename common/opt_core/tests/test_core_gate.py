"""kit_template/_core_gate.py: the pre-import core pin gate every kit carries (absent / older / unversioned / same / newer core;
unreadable pin; standard-library-only hygiene). Each case runs in a fresh ``python -I -S`` child so neither the test interpreter's own opt_core nor an installed one is in reach."""
import ast
import os
import shutil
import subprocess
import sys

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_GATE = os.path.join(CORE_DIR, "kit_template", "_core_gate.py")


def _kit(tmp_path, version="9.9.9", pin_extra="", drop_key=None):
    opt = tmp_path / "kit" / "opt"
    pkg = opt / "acme_opt"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    shutil.copy(TEMPLATE_GATE, str(pkg / "_core_gate.py"))
    keys = {"path": "../../core", "version": version}
    if drop_key:
        keys.pop(drop_key)
    body = "\n".join('%s = "%s"' % kv for kv in keys.items())
    (opt / "pyproject.toml").write_text('[project]\nname = "acme-opt"\n\n[tool.opt_core]   # the pin\n%s\n%s\n' % (body, pin_extra))
    return opt, pkg


def _core(tmp_path, version="9.9.9"):
    root = tmp_path / "core"
    (root / "opt_core").mkdir(parents=True)
    (root / "opt_core" / "__init__.py").write_text(('__version__ = "%s"\n' % version) if version else "")
    return root


def _run(opt, pkg, core_root=None):
    paths = [str(opt)] + ([str(core_root)] if core_root else [])
    code = ("import sys; sys.path[:0] = %r\n"
            "from acme_opt._core_gate import gate\n"
            "facts = gate(%r)\n"
            "print('OK', facts['installed']['version'], facts['pinned']['version'], facts['tag'], sorted(facts['installed']))\n") % (paths, str(pkg / "__init__.py"))
    r = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True, timeout=60)   # -I -S: no PYTHONPATH, no user site, no site-packages — an installed opt_core (pip -e) is out of reach too
    return r.returncode, r.stdout.strip(), [l for l in r.stderr.splitlines() if "NOT ACTIVE" in l]


def test_absent_core_is_core_missing_exit_3(tmp_path):
    opt, pkg = _kit(tmp_path)
    rc, out, lines = _run(opt, pkg)
    assert rc == 3 and out == "" and len(lines) == 1, (rc, out, lines)
    assert lines[0] == "[acme-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v9.9.9 at ../../core; nothing importable as opt_core on sys.path)", lines[0]


def test_older_core_is_core_mismatch_naming_both_sides(tmp_path):
    opt, pkg = _kit(tmp_path, version="0.4.1")
    core = _core(tmp_path, version="0.2.5")
    rc, out, lines = _run(opt, pkg, core)
    assert rc == 3 and len(lines) == 1, (rc, out, lines)
    assert lines[0] == "[acme-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v0.4.1 at ../../core, installed v0.2.5 at %s" % core, lines[0]


def test_unversioned_core_is_a_mismatch_by_name(tmp_path):
    opt, pkg = _kit(tmp_path)
    core = _core(tmp_path, version=None)
    rc, out, lines = _run(opt, pkg, core)
    assert rc == 3 and len(lines) == 1, (rc, out, lines)
    assert "reason=core_mismatch: opt_core pinned >= v9.9.9 at ../../core, installed v? at %s" % core in lines[0], lines[0]


def test_same_version_passes_and_returns_the_facts(tmp_path):
    opt, pkg = _kit(tmp_path)
    core = _core(tmp_path)
    rc, out, lines = _run(opt, pkg, core)
    assert rc == 0 and lines == [] and out == "OK 9.9.9 9.9.9 acme-opt ['package_dir', 'root', 'version']", (rc, out, lines)


def test_newer_core_passes_the_pin_is_a_floor(tmp_path):
    opt, pkg = _kit(tmp_path, version="9.9.8")
    core = _core(tmp_path, version="9.10.0")                   # numeric, not lexical: 9.10.0 > 9.9.8
    rc, out, lines = _run(opt, pkg, core)
    assert rc == 0 and lines == [] and out.startswith("OK 9.10.0 9.9.8 acme-opt"), (rc, out, lines)


def test_older_by_one_patch_is_a_mismatch(tmp_path):
    opt, pkg = _kit(tmp_path, version="9.9.9.1")
    core = _core(tmp_path, version="9.9.9")
    rc, out, lines = _run(opt, pkg, core)
    assert rc == 3 and "reason=core_mismatch: opt_core pinned >= v9.9.9.1" in lines[0] and "installed v9.9.9 at" in lines[0]


def test_unreadable_pin_is_named(tmp_path):
    opt, pkg = _kit(tmp_path, drop_key="version")
    core = _core(tmp_path)
    rc, out, lines = _run(opt, pkg, core)
    assert rc == 3 and len(lines) == 1 and "reason=core_pin_unreadable:" in lines[0] and "lacks version" in lines[0], (rc, lines)


def test_this_core_passes_a_pin_of_its_own_version(tmp_path):
    import opt_core
    opt, pkg = _kit(tmp_path, version=opt_core.__version__)
    rc, out, lines = _run(opt, pkg, CORE_DIR)
    assert rc == 0 and out.startswith("OK %s %s " % (opt_core.__version__, opt_core.__version__)), (rc, out, lines)


def test_template_is_standard_library_only_and_never_imports_the_core():
    tree = ast.parse(open(TEMPLATE_GATE).read())
    stdlib = getattr(sys, "stdlib_module_names", None)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr in ("path", "meta_path", "modules", "path_hooks"):
                raise AssertionError("the gate touches sys.%s" % node.attr)
            continue
        assert "opt_core" not in names, names
        if stdlib is not None:
            assert all(n in stdlib for n in names), names


def test_table_reader_fallback_matches_tomllib(tmp_path, monkeypatch):
    opt, pkg = _kit(tmp_path, pin_extra='extra = "x"  # trailing comment')
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gate_under_test", TEMPLATE_GATE)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pp = str(opt / "pyproject.toml")
    full = mod.read_table(pp, "tool.opt_core")
    monkeypatch.setitem(sys.modules, "tomllib", None)            # the fallback reader (interpreters without tomllib)
    assert mod.read_table(pp, "tool.opt_core") == full == {"path": "../../core", "version": "9.9.9", "extra": "x"}
    assert mod.read_table(pp, "project") == {"name": "acme-opt"} and mod.find_pyproject(str(pkg / "__init__.py")) == pp
    assert mod.version_tuple("0.5.17.4") == (0, 5, 17, 4) and mod.version_tuple("0.5.x") == (0, 5, -1) and mod.version_tuple(None) == (-1,)
