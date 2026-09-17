"""The shared core (common/opt_core) under this kit: the pin, the template backend, the exit table — and the
kit's OWN autoload hook (opt_core.autoload is not used: core defect hold). The pin and the template are checked against the core's own tree when it stands beside the kit (../../common/opt_core); a
tree without it skips those two by name."""
from __future__ import annotations

import ast
import re as _re
import json
import os
import subprocess
import sys

import pytest

from opt_core import __version__ as CORE_VERSION
from opt_core import gates as cg
from opt_core import kernels as ck
from opt_core import report as cr

from af3_torch_opt import _autoload, cli, modes, registry, report, stack

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # af3_torch/
OPT = os.path.join(HOME, "opt")
CORE_BESIDE = os.path.join(HOME, "..", "common", "opt_core")


def _pin():
    return cg.core_pin(os.path.join(OPT, "pyproject.toml"))


def test_pin_names_the_imported_core():
    """The pin gate: the pin's version is at most the core this interpreter imports (whatever path it came from)."""
    pin = _pin(); core = cg.imported_core()
    assert core["version"] == CORE_VERSION and cg.version_tuple(core["version"]) >= cg.version_tuple(pin["version"]), (pin, core)
    import af3_torch_opt
    from af3_torch_opt._core_gate import gate                                  # THE pin gate (every entry's first statement): passes on this tree, names the same facts
    facts = gate(af3_torch_opt.__file__, tag="af3-torch-opt")
    assert facts["pinned"]["version"] == pin["version"] and facts["installed"]["version"] == CORE_VERSION, facts
    assert not [w for w in stack.gates(need_params=False) if "opt_core" in w]


def test_pin_path_is_the_tree_core():
    if not os.path.isdir(CORE_BESIDE):
        pytest.skip("no common/opt_core beside this kit: the pin's path is checked where the tree carries the core")
    pin = _pin()
    assert os.path.realpath(pin["abs_path"]) == os.path.realpath(CORE_BESIDE)


def test_build_backend_is_the_template():
    if not os.path.isdir(CORE_BESIDE):
        pytest.skip("no common/opt_core beside this kit: the template is read from the tree's core")
    tpl = os.path.join(CORE_BESIDE, "kit_template", "_build_backend.py")
    assert open(os.path.join(OPT, "_build_backend.py"), "rb").read() == open(tpl, "rb").read()


def test_pin_mismatch_is_the_first_gate(tmp_path):
    """A pin that demands a newer core than what is installed: the entry gate (_core_gate.gate, every route's first statement) refuses by
    name — ONE line `NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v… at …, installed v… at …`, SystemExit code 3 — before
    anything of the core is imported; stack.gates() judges the box (kernel routes, interpreters, parameters, kit dirs), never the pin."""
    import io
    from af3_torch_opt._core_gate import gate, CoreGateRefused
    d = tmp_path / "tree" / "opt"; d.mkdir(parents=True)
    src = open(os.path.join(OPT, "pyproject.toml"), encoding="utf-8").read().replace(
        'version = "%s"' % _pin()["version"], 'version = "99.0.0"')   # a floor higher than any installed core: FLOOR semantics refuse it
    (d / "pyproject.toml").write_text(src, encoding="utf-8")
    buf = io.StringIO()
    with pytest.raises(CoreGateRefused) as ei:
        gate(str(d), tag="af3-torch-opt", stream=buf)
    assert ei.value.code == 3 and ei.value.reason == "core_mismatch", (ei.value.code, ei.value.reason)
    line = buf.getvalue().strip()
    assert line.startswith("[af3-torch-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v99.0.0 at ") and " installed v" in line and "\n" not in line, line
    assert not [w for w in stack.gates(need_params=False) if "opt_core pinned" in w]


def test_exit_table_is_the_cores():
    assert (cli.EXIT_OK, cli.EXIT_FAILED, cli.EXIT_USAGE, cli.EXIT_NOT_ACTIVE) == (cr.EXIT_OK, cr.EXIT_FAIL, cr.EXIT_USAGE, cr.EXIT_NOT_ACTIVE) == (0, 1, 2, 3)
    assert report.PREFIX == cr.prefix(report.TAG) == "[af3-torch-opt]"


def _pth_run(tmp_path, extra_env, code):
    core_dir = os.path.dirname(os.path.dirname(os.path.abspath(cg.__file__)))
    env = {k: v for k, v in os.environ.items() if not k.startswith("AF3_TORCH_OPT")}
    env.update({"PYTHONPATH": os.pathsep.join([OPT, core_dir, str(tmp_path)]), "PYTHONDONTWRITEBYTECODE": "1", **extra_env})
    return subprocess.run([sys.executable, "-c", "import json; " + code], env=env, capture_output=True, text=True)


REFUSAL = ("[af3-torch-opt] NOT ACTIVE mode=fast reason=AF3_TORCH_OPT=fast cannot be applied by a direct import of the kit "
           "(the levers are build_model arguments the wrapper command passes: af3-torch-opt pred / run.sh pred)\n")


def test_autoload_is_the_kits_own_hook(tmp_path):
    """The .pth line in a fresh interpreter (the kit's own finder, not the core's): with AF3_TORCH_OPT unset or off nothing is armed and
    no line is printed; with a mode set, the first import of the kit's api (a stand-in ``xfold``) is refused: the NOT ACTIVE line, exit 3
    (this process cannot apply the mode — it would run stock under the variable)."""
    (tmp_path / "xfold.py").write_text("X = 1\n", encoding="utf-8")
    code = "import sys; import af3_torch_opt._autoload as A; armed = A._STATE['armed']; import xfold; print(json.dumps({'armed': armed, 'fired': A._STATE['fired'], 'x': xfold.X}))"
    r = _pth_run(tmp_path, {}, code)
    assert r.returncode == 0 and json.loads(r.stdout) == {"armed": False, "fired": False, "x": 1} and r.stderr == "", r
    r = _pth_run(tmp_path, {"AF3_TORCH_OPT": "off"}, code)
    assert r.returncode == 0 and json.loads(r.stdout) == {"armed": False, "fired": False, "x": 1} and r.stderr == "", r
    r = _pth_run(tmp_path, {"AF3_TORCH_OPT": "fast"}, code)
    assert r.returncode == cli.EXIT_NOT_ACTIVE == 3 and r.stdout == "" and r.stderr == REFUSAL, r


def test_autoload_fires_on_a_find_spec_probe_then_import(tmp_path):
    """A bare importlib.util.find_spec probe of a trigger counts as the first import: the refusal fires on the probe, before any real import."""
    (tmp_path / "xfold.py").write_text("X = 2\n", encoding="utf-8")
    code = "import sys, importlib.util; import af3_torch_opt._autoload as A; spec = importlib.util.find_spec('xfold'); import xfold; print('reached')"
    r = _pth_run(tmp_path, {"AF3_TORCH_OPT": "fast"}, code)
    assert r.returncode == 3 and "reached" not in r.stdout and r.stderr == REFUSAL, r


def test_autoload_refuses_an_unknown_selection(tmp_path):
    """A value that is not a mode of this package: the kit's own NOT ACTIVE line and exit 3 at interpreter start — never stock silently."""
    r = _pth_run(tmp_path, {"AF3_TORCH_OPT": "turbo"}, "import af3_torch_opt._autoload; print('reached')")
    assert r.returncode == cli.EXIT_NOT_ACTIVE == 3 and "reached" not in r.stdout, r
    assert r.stderr == "[af3-torch-opt] NOT ACTIVE mode=turbo reason=AF3_TORCH_OPT=turbo is not a mode of this package (modes: off, exact, fast, big)\n", r.stderr
    r = _pth_run(tmp_path, {"AF3_TORCH_OPT": "exact"}, "import af3_torch_opt._autoload; print('reached')")       # `exact`: a mode of this package like the others — the hook installs, nothing refused
    assert r.returncode == 0 and "reached" in r.stdout and "NOT ACTIVE" not in r.stderr, r


def test_refusals_hold_under_a_site_pth_line(tmp_path):
    """The installed form: ``af3_torch_opt_autoload.pth`` runs ``import af3_torch_opt._autoload`` inside site's .pth processing at interpreter
    start — a refusal there must still end the process with exit 3 and the line (a raised SystemExit would be reported by site as a startup
    error, exit 1). A site dir with that .pth line, added through PYTHONPATH's site processing (``site.addsitedir``)."""
    core_dir = os.path.dirname(os.path.dirname(os.path.abspath(cg.__file__)))
    site_dir = tmp_path / "site"; site_dir.mkdir(); (site_dir / "af3_torch_opt_autoload.pth").write_text("import af3_torch_opt._autoload\n", encoding="utf-8")
    (tmp_path / "xfold.py").write_text("X = 3\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("AF3_TORCH_OPT")}
    env.update({"PYTHONPATH": os.pathsep.join([OPT, core_dir, str(tmp_path)]), "PYTHONDONTWRITEBYTECODE": "1"})
    boot = f"import site; site.addsitedir({str(site_dir)!r}); "                       # the .pth line executes here, as site does at start
    r = subprocess.run([sys.executable, "-c", boot + "import xfold; print('reached', xfold.X)"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "reached 3" and r.stderr == "", r
    r = subprocess.run([sys.executable, "-c", boot + "import xfold; print('reached')"], env={**env, "AF3_TORCH_OPT": "fast"}, capture_output=True, text=True)
    assert r.returncode == 3 and "reached" not in r.stdout and r.stderr == REFUSAL, r
    r = subprocess.run([sys.executable, "-c", boot + "print('reached')"], env={**env, "AF3_TORCH_OPT": "bogus"}, capture_output=True, text=True)
    assert r.returncode == 3 and "reached" not in r.stdout and "Error processing line" not in r.stderr and r.stderr.startswith("[af3-torch-opt] NOT ACTIVE mode=bogus "), r


def test_refusals_hold_on_the_real_startup_route(tmp_path):
    """The real route: the .pth line processed by site at INTERPRETER START (a user site under PYTHONUSERBASE carrying
    af3_torch_opt_autoload.pth), `AF3_TORCH_OPT=bogus python -c pass` → rc 3, exactly the one line, no Traceback; likewise a known mode,
    whose refusal fires at the trigger import. Skipped by name where this interpreter disables the user site."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AF3_TORCH_OPT", "PYTHONPATH", "PYTHONUSERBASE", "PYTHONNOUSERSITE"))}
    probe = subprocess.run([sys.executable, "-c", "import site; print(site.ENABLE_USER_SITE)"], env=env, capture_output=True, text=True)
    if probe.stdout.strip() != "True":
        pytest.skip(f"this interpreter disables the user site ({probe.stdout.strip() or probe.stderr.strip()[:80]}): the PYTHONUSERBASE route cannot be exercised here")
    base = tmp_path / "userbase"
    usersite = base / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    usersite.mkdir(parents=True); (usersite / "af3_torch_opt_autoload.pth").write_text("import af3_torch_opt._autoload\n", encoding="utf-8")
    (usersite / "__af3_torch_opt_path.pth").write_text(OPT + "\n" + os.path.dirname(os.path.dirname(os.path.abspath(cg.__file__))) + "\n", encoding="utf-8")
    (tmp_path / "xfold.py").write_text("X = 4\n", encoding="utf-8")
    env.update({"PYTHONUSERBASE": str(base), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(tmp_path)})
    r = subprocess.run([sys.executable, "-c", "import site, af3_torch_opt._autoload as A; assert site.ENABLE_USER_SITE and str(A.__file__).startswith(%r); import xfold; print('ok', xfold.X)" % OPT], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok 4" and r.stderr == "", r          # the user site is live, the hook inert without the variable
    r = subprocess.run([sys.executable, "-c", "pass"], env={**env, "AF3_TORCH_OPT": "bogus"}, capture_output=True, text=True)
    assert r.returncode == 3 and r.stdout == "" and "Traceback" not in r.stderr and r.stderr.count("\n") == 1, r
    assert r.stderr == "[af3-torch-opt] NOT ACTIVE mode=bogus reason=AF3_TORCH_OPT=bogus is not a mode of this package (modes: off, exact, fast, big)\n", r.stderr
    r = subprocess.run([sys.executable, "-c", "import xfold; print('reached')"], env={**env, "AF3_TORCH_OPT": "fast"}, capture_output=True, text=True)
    assert r.returncode == 3 and "reached" not in r.stdout and "Traceback" not in r.stderr and r.stderr == REFUSAL, r
    # run.sh on the same route: its helper probes (configs/h100.env exports, `import af3_torch_opt`, check_pins) run WITHOUT the variable, so the
    # refusal is the wrapper's own start (rc 3, the one line) — not a false 'not importable' (rc 2) / 'pins not met' (rc 3) from a probe
    b = tmp_path / "bin"; b.mkdir(); (b / "python").symlink_to(sys.executable)
    stubs = {}
    for which in ("torch_python", "jax_python"):                              # two stub interpreters answering check_pins' version query with their pinned set
        stub = tmp_path / f"stub_{which}"; stubs[which] = str(stub)
        stub.write_text("#!/usr/bin/env python3\nimport json, os, sys\nP = json.load(open(os.path.join(%r, 'stock', 'PINS.json')))\n"
                        "print(json.dumps({'python': '3.12.0', **P['check_packages'][%r]}))\n" % (HOME, which), encoding="utf-8")
        stub.chmod(stub.stat().st_mode | 0o111)
    e2 = {**env, "PATH": f"{b}:{env.get('PATH', os.environ.get('PATH', ''))}", "AF3_TORCH_PY": stubs["torch_python"], "AF3_TORCH_JAX_PY": stubs["jax_python"], "AF3_TORCH_JAX_REPO": str(tmp_path), "AF3_TORCH_OPT": "bogus"}   # the config checks the three required variables by presence (the fork checkout: any dir here — the mode is refused before the gates read it)
    r = subprocess.run(["bash", os.path.join(HOME, "run.sh"), "check", "--config", "h100"], env=e2, capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and "not importable" not in r.stderr and "not installed" not in r.stderr and "pins not met" not in r.stderr and "Traceback" not in r.stderr, r
    assert r.stderr.strip().splitlines()[-1] == "[af3-torch-opt] NOT ACTIVE mode=bogus reason=AF3_TORCH_OPT=bogus is not a mode of this package (modes: off, exact, fast, big)", r.stderr


def test_wrapper_chain_is_unaffected_by_the_hook(box, tmp_path, capsys):
    """The wrapper imports no trigger and strips AF3_TORCH_OPT* from its model processes: `pred` under AF3_TORCH_OPT=fast runs (the hook is
    the .pth's concern, not the wrapper's), and the model processes see no AF3_TORCH_OPT* name (the STOCK proof)."""
    from .test_cli_manifest import _inputs
    os.environ["AF3_TORCH_OPT"] = "fast"
    try:
        out = tmp_path / "o"
        assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("x",))[0], "--output_dir", str(out)]) == 0
    finally:
        del os.environ["AF3_TORCH_OPT"]
    assert cli.last_run()["stock_proof"]["env_present"] == []
    capsys.readouterr()


def test_undeclared_package_variables_are_refused(box, monkeypatch, capsys):
    """A mistyped switch under the package prefix (AF3_TORCH_OPT_MODE=fast) is a refusal by name, never silently stripped."""
    assert stack.undeclared_env({"AF3_TORCH_OPT": "fast", "AF3_TORCH_OPT_HOME": "/t", "AF3_TORCH_OPT_KIT": "/k", "AF3_TORCH_OPT_DTK": "/d", "AF3_TORCH_PY": "/p"}) == []
    monkeypatch.setenv("AF3_TORCH_OPT_MODE", "fast"); monkeypatch.setenv("AF3_TORCH_OPTIONS", "1")
    why = stack.gates(need_params=False)
    assert [w for w in why if w.startswith("undeclared variable(s) under the package prefix: AF3_TORCH_OPTIONS, AF3_TORCH_OPT_MODE (declared: AF3_TORCH_OPT, AF3_TORCH_OPT_HOME, AF3_TORCH_OPT_KIT, AF3_TORCH_OPT_DTK)")], why
    assert cli.main(["check", "--mode", "fast"]) == 3 and "undeclared variable(s)" in capsys.readouterr().err
    monkeypatch.delenv("AF3_TORCH_OPT_MODE"); monkeypatch.delenv("AF3_TORCH_OPTIONS")
    assert not [w for w in stack.gates(need_params=False) if "undeclared" in w]
    monkeypatch.setenv("AF3_TORCH_BIG_GRAPH_DROP", "0")                       # big is one composition: no AF3_TORCH_BIG_* name is read, a caller-set one is refused by name
    assert stack.undeclared_env() == ["AF3_TORCH_BIG_GRAPH_DROP"]
    assert cli.main(["check", "--mode", "big"]) == 3 and "AF3_TORCH_BIG_GRAPH_DROP" in capsys.readouterr().err
    monkeypatch.delenv("AF3_TORCH_BIG_GRAPH_DROP")
    assert stack.undeclared_env() == []


def test_autoload_names_no_mode_itself():
    """_autoload restates no mode: the table is modes.MODES, read only under a set variable."""
    tree = ast.parse(open(_autoload.__file__, encoding="utf-8").read())
    assert not [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODES" for t in n.targets)]
    assert _autoload.TARGETS == ("af3_torch_api", "xfold") and _autoload.ENV_MODE == modes.ENV_MODE and _autoload.EXIT_NOT_ACTIVE == cli.EXIT_NOT_ACTIVE


# --- the routed kernels (registry.KERNEL_ROUTES): the core's carried copies stand in for the kit's byte-identical own copies ---

def test_routed_kernels_are_byte_identical_to_the_kits_copies():
    """The route is numerics-free by construction: for every routed kernel, every file the core carries (its META record's live ``files``
    list, opt_core.kernels.sums) is byte-identical, compared live, to the kit's carried copy — except a file registry.KERNEL_PARITY names,
    which must differ (live) and carry a non-empty proof that the served bytes still keep the class word; the core's own tests/ and its licence
    texts (LICENSE / NOTICE package data) are the core's alone; wherever the name resolves holds to the core copy (the core's own carried-file check); every export
    the route names is a kit file that exists. No digest is stored on either side: the bytes on disk are the only record."""
    import filecmp
    for name, r in registry.KERNEL_ROUTES.items():
        doc = ck.sums(name)
        assert isinstance(doc["files"], list) and doc["files"] == sorted(doc["files"]) and doc["files"], (name, doc["files"])
        assert ck.verify_carry(name) == [], (name, ck.verify_carry(name))
        for var, relf in r["exports"].items():                                 # the kit data the routed module reads: carried
            assert os.path.isfile(os.path.join(stack.forward_dir(), relf)), (name, var, relf)
        if r["kit_copy"] is None:                                              # the core's copy is the only one: nothing of the kit's to compare
            assert name not in registry.KERNEL_PARITY, name
            continue
        kit_copy = os.path.join(stack.forward_dir(), r["kit_copy"])
        assert os.path.exists(kit_copy), kit_copy
        core_copy = ck.carried_path(name)                                      # the core's package directory, or its module file
        core_root = core_copy if doc["kind"] == "package" else os.path.dirname(core_copy)
        parity = registry.KERNEL_PARITY.get(name, {})
        compared = 0
        for rel in doc["files"]:
            if os.path.isfile(kit_copy):                                     # a single-module route: the kit carries the module file itself
                if rel != os.path.basename(kit_copy):
                    assert rel.endswith(("LICENSE", ".NOTICE")) or rel.startswith("tests/"), (name, rel)
                    continue
                p = kit_copy
            else:
                p = os.path.join(kit_copy, rel)
                if not os.path.isfile(p):
                    assert rel.startswith("tests/") or rel.endswith(("LICENSE", "NOTICE")), (name, rel)   # the kit carries every executable file of the module
                    continue
            same = filecmp.cmp(p, os.path.join(core_root, rel), shallow=False); compared += 1
            if rel in parity:
                assert not same and parity[rel], (name, rel)                 # a proven exception: kit and core bytes differ, live; parity[rel] is the proof naming why that's still safe
            else:
                assert same, (name, rel, p, os.path.join(core_root, rel))    # DRY: the carried copy IS the core's bytes
        assert compared >= 1, (name, kit_copy)                               # a kit copy that compares nothing is not a carried copy
        assert set(parity) <= set(doc["files"]), (name, set(parity) - set(doc["files"]))
        required = {var for var, spec in doc.get("exports", {}).items() if spec.get("required")}
        assert required <= set(stack.kernel_exports()), (name, required)
        for param, rel in r.get("exports", {}).items():
            assert os.path.isfile(os.path.join(stack.forward_dir(), rel)), (name, param, rel)
    assert set(registry.KERNEL_PARITY) <= set(registry.KERNEL_ROUTES)
    for name, r in registry.KERNEL_ROUTES.items():                     # the levers a route serves are registry levers whose touches name the core path
        for lever in r["levers"]:
            assert any(t.startswith(registry.CORE) for t in registry.LEVERS[lever]["touches"]), (name, lever)


def test_route_gate_passes_in_a_fresh_model_process(tmp_path):
    """forward.py's own gate on a bare interpreter (-I) with the kit's sys.path recipe and the wrapper's exports: every routed name resolves to
    the core copy (not the kit's third_party copy behind it), holds to the sums file, and nothing imported it before the gate."""
    code = f"""
import json, os, sys
sys.path[:0] = {[os.path.join(OPT, 'af3_torch_opt')] + stack.kit_sys_path()!r}
import forward as F
recs, err = F.route_kernels({stack.kernel_routes()!r}, {stack.core_dir()!r})
print(json.dumps({{"recs": recs, "err": err, "from": F.imported_from({stack.kernel_routes()!r})}}))
"""
    env = {"PATH": os.environ.get("PATH", ""), **stack.kernel_exports()}
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["err"] is None, got
    for name in stack.kernel_routes():
        rec = got["recs"][name]
        assert rec["ok"] and rec["routed"] and not rec["already_imported"] and rec["differing"] == [] and rec["missing"] == [], (name, rec)
        assert rec["resolved"].startswith(os.path.realpath(stack.core_dir())) or rec["resolved"].startswith(stack.core_dir()), (name, rec["resolved"])
        assert got["from"][name] is None                                   # the gate imports nothing
    assert not stack.kernel_exports()                                       # no routed kernel of this kit reads a kit file: nothing to export (the TriMul provider carries its own cells)


def test_model_process_env_carries_the_kernel_exports(box):
    env = stack.model_process_env(jax=False)
    assert "FPF_TRIMUL_V4_CELLS" not in env and "FPF_TRIMUL_V4_CELLS" not in stack.model_process_env(jax=True)   # no kit cell table: the TriMul provider's own cells serve (nothing exported)
    assert not stack.kernel_route_gate()


def test_a_missing_core_is_named_not_active(tmp_path):
    """The shared core absent from the interpreter, under AF3_TORCH_OPT=fast: (a) the .pth hook (`import af3_torch_opt._autoload` — the package
    interface is lazy, so nothing of the core is imported before the hook's own guarded import of the mode table) prints `NOT ACTIVE mode=fast
    reason=core_missing:opt_core` and exits 3 at interpreter start; (b) the wrapper (`python -m af3_torch_opt pred`) prints the same line and exits 3 —
    never a traceback, never a stock run under the mode's name."""
    head = "[af3-torch-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v"                # the pin gate's line (_core_gate.gate, the core's kit template): ONE line, exit 3
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AF3_TORCH", "PYTHON"))}
    env.update(PYTHONPATH=OPT, AF3_TORCH_OPT="fast")                                   # the package importable, opt_core NOT on the path
    r = subprocess.run([sys.executable, "-S", "-c", "import af3_torch_opt._autoload; print('reached')"], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and r.stdout == "" and "Traceback" not in r.stderr and r.stderr.startswith(head) and len(r.stderr.strip().splitlines()) == 1, r   # the hook, at interpreter start
    r = subprocess.run([sys.executable, "-S", "-m", "af3_torch_opt", "pred", "--mode", "fast", "--json_path", "x.json", "--output_dir", str(tmp_path)], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 3 and r.stdout == "" and "Traceback" not in r.stderr, r
    assert r.stderr.startswith(head) and len(r.stderr.strip().splitlines()) == 1, r.stderr


def test_lever_names_are_canonical_strategy_ids():
    """Every LEVER line's name= is a canonical cross-engine strategy id (registry.STRATEGY, one per registry lever incl. big's) — held to the
    shared core's STRATEGIES.json when the core ships it — and every impl/origin slot is filled with origin in {core, kit}."""
    from af3_torch_opt import big
    assert set(registry.STRATEGY) == set(registry.IMPL) == set(registry.LEVERS) | set(big.LEVER_ORDER) | set(registry.N_GPU_LEVER)   # + the n_gpu axis's lever (rowpair)
    assert all(o in ("core", "kit") and i and " " not in i for i, o in registry.IMPL.values())
    assert all(_re.match(r"^(F[1-7]\.[a-z0-9_]+|LOCAL(\.af3_torch)?\.[a-z0-9_]+)$", s) for s in registry.STRATEGY.values()), registry.STRATEGY   # canonical ids, the core's LOCAL ids, or this kit's LOCAL.af3_torch.<name> ids (opt_core.report.lever_line's kit-local form)
