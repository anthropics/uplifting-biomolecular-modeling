"""The shared core (common/opt_core): the whole dependency, end to end. The pin in opt/pyproject.toml is a floor the installed core's
version must clear (the core pin gate, statement one of every entry, off included); the build backend and the gate module are the core's
templates byte for byte; the .pth is the backend's generated text; the run manifest and the stock proof carry the core's blocks; the JIT
cache key is the core's grammar over this package's probes; the deterministic recipe has the core's shape. At the real trigger --
`opt/genie3_opt_autoload.pth` under `site.addsitedir`, the mechanism behind `GENIE3_OPT=<mode> genie3 generate …` -- a kit mode named at
upstream's `genie3 generate` hands the process to the design verb (`python -m genie3_opt design --mode <m> …`, whose statement one is the
core pin gate: a stale core is refused there by name, exit 3), and a kit mode named at any other importer of genie3 is REFUSED by the env
route's own name and the process ends 3 before anything of upstream runs, whatever the core. `off` and unset stay inert whatever the core is. Every other entry point (`python -m genie3_opt`, the console script, `run.sh`, `configs/h100.env`,
in-process `enable`/`check`/`run_design`) calls the same `_core.core_gate()` as statement one -- structurally identical, not re-checked
per wrapper here.

The shared core's own manifest/gates return more than {version, package_dir} today (a separate branch narrows them) — so nothing here
asserts their exact key set; only version/package_dir/pin fields, which are stable across that landing."""

import os
import shutil
import subprocess
import sys

import pytest

from genie3_opt import _autoload, _core, _core_gate, det, manifest, stack, stock_cli
from genie3_opt.codes import TAG
from genie3_opt.tests import _stubs

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                       # opt/


# ===== the pin floor, the core's own templates, JIT cache-key grammar, deterministic recipe shape =====
def test_pin_matches_the_installed_core():
    f = _core.core_gate()
    assert f["tag"] == TAG == "genie3-opt"
    assert _core_gate.version_tuple(f["installed"]["version"]) >= _core_gate.version_tuple(f["pinned"]["version"]), f
    assert os.path.samefile(f["pinned"]["pyproject"], _core.PYPROJECT)
    import opt_core                                                    # the gate imported nothing of the core; the core it found is the one an import binds
    assert os.path.samefile(f["installed"]["package_dir"], os.path.dirname(opt_core.__file__)) and opt_core.__version__ == f["installed"]["version"]


def test_backend_gate_and_pth_are_the_templates():
    root = _core.core_gate()["installed"]["root"]
    for rel, mine in (("_build_backend.py", os.path.join(OPT, "_build_backend.py")), ("_core_gate.py", os.path.join(OPT, "genie3_opt", "_core_gate.py"))):
        assert open(mine, "rb").read() == open(os.path.join(root, "kit_template", rel), "rb").read(), rel
    sys.path.insert(0, OPT)
    try:
        import _build_backend
    finally:
        sys.path.remove(OPT)
    pth = open(os.path.join(OPT, "genie3_opt_autoload.pth")).read()
    assert pth == _build_backend.pth_text("genie3_opt", "GENIE3_OPT", TAG) and _build_backend.pth_fields(pth) == ("genie3_opt", "GENIE3_OPT", TAG, 3)


def test_activation_gates_on_the_core_first(tmp_path, monkeypatch, capsys):
    _stubs.box(str(tmp_path), monkeypatch)
    pinned_version = None
    for mode in ("exact", "off"):
        stack.reset_for_tests()
        rep = stack.activate(mode, dry_run=True)
        assert rep["opt_core"]["ok"], rep["opt_core"]
        assert _core_gate.version_tuple(rep["opt_core"]["version"]) >= _core_gate.version_tuple(rep["opt_core"]["pinned"]["version"]), rep["opt_core"]
        assert os.path.isdir(rep["opt_core"]["package_dir"]) and os.path.isdir(rep["opt_core"]["root"]), rep["opt_core"]
        pinned_version = rep["opt_core"]["pinned"]["version"]
    real = _core_gate.installed_core()
    for mode in ("exact", "off"):                                      # an OLDER core: the gate's core_mismatch line and SystemExit(3), never an inactive report, never a stock run
        stack.reset_for_tests()
        monkeypatch.setattr(_core_gate, "installed_core", lambda: dict(real, version="0.2.5"))
        with pytest.raises(SystemExit) as e:
            stack.activate(mode, dry_run=True)
        err = capsys.readouterr().err
        assert e.value.code == 3 and err.count("NOT ACTIVE") == 1
        assert f"[{TAG}] NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v{pinned_version} at" in err and "installed v0.2.5 at" in err, err
    monkeypatch.setattr(_core_gate, "installed_core", lambda: None)   # NO core importable: core_missing, on the strict route too (SystemExit is not an ActivationError: nothing of the kit runs)
    for strict in (False, True):
        stack.reset_for_tests()
        with pytest.raises(SystemExit) as e:
            stack.activate("exact", dry_run=not strict, strict=strict)
        assert e.value.code == 3 and f"NOT ACTIVE: reason=core_missing:opt_core (pinned >= v{pinned_version} at" in capsys.readouterr().err


def test_manifest_and_stock_proof_carry_the_core():
    block = manifest.core_block()
    assert "version" in block and block["version"], block               # the exact key set is opt_core's own to define; only version is this kit's contract
    env = {"PATH": "/bin", "GENIE3_ROOT": "/src/genie3", "GENIE3_WEIGHTS": "/w"}
    proof = stock_cli.env_proof(["GENIE3_OPT", "GENIE3_", "CUDA_MPS_", "NVIDIA_TF32_OVERRIDE"], [os.path.join(OPT, "forward")], environ=env,
                                modules={"os": None, "sys": None}, path=["/usr/lib/python3"], allowed=["GENIE3_ROOT", "GENIE3_WEIGHTS"])
    assert proof["ok"] and proof["core"]["ok"], proof
    assert proof["core"]["allowed_excluded"] == ["GENIE3_ROOT", "GENIE3_WEIGHTS"] and set(proof["core"]["proof"]) >= {"forbidden_present", "autoload_armed", "core_modules_loaded"}
    bad = stock_cli.env_proof(["GENIE3_OPT", "GENIE3_"], [os.path.join(OPT, "forward")], environ=dict(env, GENIE3_OPT="exact"), modules={"os": None}, path=[],
                              allowed=["GENIE3_ROOT", "GENIE3_WEIGHTS"])
    assert not bad["ok"] and not bad["core"]["ok"] and "GENIE3_OPT" in bad["core"]["violations"], bad["core"]


def test_jit_cache_key_is_the_core_grammar(tmp_path, monkeypatch):
    from opt_core.jit_cache import key
    _stubs.box(str(tmp_path), monkeypatch)
    assert stack.jit_cache_key({"cc": "9.0", "sm": "sm90"}) == key(version=(stack.dist_version("torch") or "unknown:PackageNotFoundError").partition("+")[0],
                                                                  cuda=stack.jit_cache_key({"cc": "9.0"}).split("-cu", 1)[1].split("-sm", 1)[0], cc="9.0")
    assert stack.jit_cache_key({"sm": "sm90"}).endswith("-sm90")
    assert stack.jit_cache_key({}).endswith("-smunknown:no-gpu")


def test_recipe_has_the_core_shape():
    from opt_core.det import PRODUCTION, describe, stock_exception
    r1, r0 = det.recipe(1), det.recipe(0)
    assert r0 is PRODUCTION and r1.level == 1 and not r1.env and not r1.unset and not r1.pythonpath
    assert stock_exception(r1) is None                       # nothing exported: the stock proof needs no carve-out
    d = det.describe(0, det_level=1)
    assert d["det"] == 1 and set(d) == {"det", "line", "seed", "seed_key", "seed_doc", "determinism", "tf32", "process_rule", "log_dir"} and d["line"] == describe(r1) and d["line"].startswith("det=1 env=none unset=none pythonpath=0")
    assert det.describe(None, det_level=0)["line"].startswith("det=0 ") and det.describe(None)["det"] == det.DEFAULT_LEVEL == 0


def test_jit_cache_key_on_the_pinned_stack_is_the_pinned_literal(tmp_path, monkeypatch):
    """On the pinned stack (torch 2.7.1+cu126, H100 cc 9.0) the exported key is byte-identical to the key every sealed run dir
    carries — `torch2.7.1-cu126-sm90` — so no mounted warm cache is orphaned; the PyPI-wheel spelling of the same stack gives the same key."""
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setattr(stack, "dist_version", lambda name: {"torch": "2.7.1+cu126"}.get(name))
    assert stack.jit_cache_key({"cc": "9.0", "sm": "sm90"}) == "torch2.7.1-cu126-sm90"
    monkeypatch.setattr(stack, "dist_version", lambda name: {"torch": "2.7.1", "nvidia-cuda-runtime-cu12": "12.6.77"}.get(name))
    assert stack.jit_cache_key({"cc": "9.0", "sm": "sm90"}) == "torch2.7.1-cu126-sm90"


# ===== the autoload route at the stock command line (a fake upstream on the path, no real .pth file) =====
PROG = "import genie3_opt._autoload as a; import genie3; print('imported', a.FINDER is not None)"


def test_install_rules(monkeypatch):
    _autoload.uninstall()
    monkeypatch.delenv("GENIE3_OPT", raising=False)
    assert _autoload.install() is False and _autoload.FINDER is None
    monkeypatch.setenv("GENIE3_OPT", "off")
    assert _autoload.install() is True and _autoload.FINDER is not None and _autoload.FINDER in sys.meta_path
    _autoload.uninstall()
    assert _autoload.FINDER is None
    monkeypatch.setenv("GENIE3_OPT_AUTOLOAD", "0")
    assert _autoload.install() is False


def _run(mode_env, tmp):
    """A fake upstream package `genie3` on the path (the finder wakes on its import), the .pth's import line run by hand."""
    os.makedirs(os.path.join(tmp, "fake", "genie3"), exist_ok=True)
    open(os.path.join(tmp, "fake", "genie3", "__init__.py"), "w").write("")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GENIE3_OPT")}
    env.update(mode_env)
    env["PYTHONPATH"] = os.path.join(tmp, "fake") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", PROG], env=env, capture_output=True, text=True)


def test_off_is_inert_and_a_kit_mode_refuses(tmp_path):
    tmp = str(tmp_path)
    r = _run({"GENIE3_OPT": "off"}, tmp)
    assert r.returncode == 0 and "imported" in r.stdout
    r = _run({"GENIE3_OPT": "exact"}, tmp)                                  # a kit mode at an importer of genie3 that is not `genie3 generate`: refused by name, the process ends 3 before anything of upstream runs
    assert r.returncode == 3 and "imported" not in r.stdout and "Traceback" not in r.stderr, (r.returncode, r.stderr[-600:])
    assert "[genie3-opt] NOT ACTIVE: env-route: GENIE3_OPT='exact' is set at `python -c …`" in r.stderr and "refused (exit 3), nothing of upstream ran" in r.stderr and "GENIE3_OPT=off for the stock path" in r.stderr and "GENIE3_OPT_AUTOLOAD=0" in r.stderr, r.stderr[-600:]
    assert r.stderr.count("[genie3-opt]") == 1 and "DECLINED" not in r.stderr and "-> stock" not in r.stderr
    r = _run({"GENIE3_OPT": "fast", "GENIE3_OPT_AUTOLOAD": "0"}, tmp)
    assert r.returncode == 0 and "imported" in r.stdout


def test_env_route_carries_genie3_generate_to_the_design_verb():
    """`<bin>/genie3 generate <args>` under a kit mode becomes `python -m genie3_opt design --mode <m> <args>` (os.execv of design_argv): upstream's
    generate flags are the design verb's own spellings; anything else (another genie3 command, `python -c`, a script importing the library) has
    no design argv and is refused by name."""
    gen = ["/venv/bin/genie3", "generate", "-c", "e.yaml", "--verbose", "--log-dir", "L", "--num-shards", "2", "--shard-id", "1"]
    want = ["-c", os.path.abspath("e.yaml"), "--verbose", "--log-dir", os.path.abspath("L"), "--num-shards", "2", "--shard-id", "1"]   # the request file and the log dir made absolute against the caller's cwd
    assert _autoload.design_argv("fast", gen, python="PY") == ["PY", "-m", "genie3_opt", "design", "--mode", "fast"] + want
    assert _autoload.design_argv("fast", ["genie3", "generate", "--config=e.yaml", "/abs/x"], python="PY")[6:] == ["--config=" + os.path.abspath("e.yaml"), "/abs/x"]
    assert _autoload.design_argv("exact", ["genie3", "generate"])[1:] == ["-m", "genie3_opt", "design", "--mode", "exact"]
    for other in (["/venv/bin/genie3", "evaluate", "-c", "e.yaml"], ["/venv/bin/genie3", "run", "-c", "e.yaml"], ["/venv/bin/genie3"], ["-c"], [], ["/x/generate.py", "generate"], ["/x/genie3.py", "generate"]):
        assert _autoload.design_argv("fast", other) is None, other
    line = _autoload.refused_line("fast", ["/venv/bin/genie3", "evaluate", "-c", "e.yaml"])
    assert line.startswith("[genie3-opt] NOT ACTIVE: env-route: GENIE3_OPT='fast' is set at `genie3 evaluate`") and "genie3-opt design --mode fast" in line and "exit 3" in line
    assert [_autoload.importer_words(a) for a in (["-c"], ["-m", "genie3.cli", "generate"], ["/x/tool.py", "run"], [], ["/venv/bin/genie3"])] == ["python -c …", "python -m genie3.cli", "python tool.py", "an embedded interpreter importing genie3", "genie3"]


def test_shard_slice_is_upstreams_own_slicing():
    """design.shard_slice (the pass's expected design names per shard) against upstream's OWN `_apply_shard_to_config` — the function the driver's
    build reaches through to_generation_config — executed from the pinned source (stock/src/src/genie3/config/loader.py; no torch needed), over a
    grid of n_sample < 40 and num_shards < 12: the same (offset, count) for every shard, and the shards tile 0..n exactly once."""
    import ast, math
    from typing import Any
    from genie3_opt import design
    src_path = os.path.join(os.path.dirname(OPT), "stock", "src", "src", "genie3", "config", "loader.py")
    tree = ast.parse(open(src_path, encoding="utf-8").read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_shard_to_config"]
    assert len(fn) == 1, "upstream's _apply_shard_to_config not found in " + src_path
    ns = {"math": math, "Any": Any}
    exec(compile(ast.Module(body=fn, type_ignores=[]), src_path, "exec"), ns)
    upstream = ns["_apply_shard_to_config"]
    for n in range(0, 40):
        for m in range(1, 12):
            covered = []
            for k in range(m):
                cfg = {"dataset": {"n_sample": n}, "inference": {}}
                if m > 1:
                    assert upstream(cfg, k, m) is False                        # non-beam: the caller's beam normalisation is a no-op
                    theirs = (cfg["dataset"]["sample_index_offset"], cfg["dataset"]["n_sample"])
                else:
                    theirs = (0, n)                                           # to_generation_config does not slice at num_shards 1
                lo, hi = design.shard_slice(n, k, m)
                assert (lo, hi - lo) == theirs, (n, m, k, (lo, hi), theirs)
                covered += list(range(lo, hi))
            assert covered == list(range(n)), (n, m, covered)


def test_real_pth_route_core_reached_through_the_tree(tmp_path):
    """Through a REAL .pth: the package's `opt/genie3_opt_autoload.pth` in a temporary site dir, processed by site.addsitedir in a `python -S`
    child whose path holds the package and a fake upstream and NOT the shared core — the GOOD-core case by the tree layout: at the trigger
    (`import genie3` from `python -c`, not upstream's `genie3 generate`) a kit mode is refused by the env route's own name and the process ends 3
    (`stock ran` never prints) — never site.py's swallowed error; `off` and unset stay inert and silent. The hook imports nothing of the core
    while site.py runs the line. (The core-ABSENT and core-STALE cases of this route: below, a kit copy with no core beside it.)"""
    tmp = str(tmp_path)
    site_dir, fake = os.path.join(tmp, "site"), os.path.join(tmp, "fake")
    os.makedirs(site_dir); os.makedirs(os.path.join(fake, "genie3"))
    open(os.path.join(fake, "genie3", "__init__.py"), "w").write("")
    opt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pth = open(os.path.join(opt, "genie3_opt_autoload.pth")).read()
    open(os.path.join(site_dir, "genie3_opt_autoload.pth"), "w").write(pth)
    prog = f"import sys, site; sys.path[:0] = [{opt!r}, {fake!r}]; site.addsitedir({site_dir!r}); import genie3; print('stock ran')"
    def child(mode_env):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GENIE3_OPT", "PYTHONPATH"))}
        env.update(mode_env)
        return subprocess.run([sys.executable, "-S", "-c", prog], env=env, capture_output=True, text=True)
    r = child({"GENIE3_OPT": "exact"})
    assert r.returncode == 3 and "[genie3-opt] NOT ACTIVE: env-route: GENIE3_OPT='exact' is set at `python -c …`" in r.stderr and "stock ran" not in r.stdout, (r.returncode, r.stderr[-600:])
    assert "Error processing line" not in r.stderr and "Traceback" not in r.stderr and r.stderr.count("[genie3-opt]") == 1, r.stderr[-600:]
    r = child({"GENIE3_OPT": "off"})
    assert r.returncode == 0 and "stock ran" in r.stdout and "[genie3-opt]" not in r.stderr, r.stderr[-600:]
    r = child({})
    assert r.returncode == 0 and "stock ran" in r.stdout and "[genie3-opt]" not in r.stderr, r.stderr[-600:]


# ===== the pin gate at the REAL .pth trigger: two broken cores (absent, older), reached exactly as production reaches it =====
HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))                       # genie3/
PIN = _core_gate.read_table(os.path.join(KIT, "opt", "pyproject.toml"), "tool." + _core_gate.PIN_TABLE)   # the kit's pin as the gate reads it: every refusal names it
PINNED_VERSION = PIN["version"]


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """<tmp>/genie3/{opt/{genie3_opt/,genie3_opt_autoload.pth}} and NO <tmp>/common/opt_core; a shadow older core; a fake upstream
    `genie3`; a site dir holding the real .pth."""
    tmp = str(tmp_path_factory.mktemp("gate"))
    kit = os.path.join(tmp, "genie3")
    os.makedirs(os.path.join(kit, "opt"))
    shutil.copytree(os.path.join(KIT, "opt", "genie3_opt"), os.path.join(kit, "opt", "genie3_opt"), ignore=shutil.ignore_patterns("tests", "__pycache__"))
    for rel in ("opt/pyproject.toml", "opt/genie3_opt_autoload.pth"):                # pyproject.toml: where the gate reads the pin from
        shutil.copy(os.path.join(KIT, *rel.split("/")), os.path.join(kit, *rel.split("/")))
    shadow = os.path.join(tmp, "shadow")
    os.makedirs(os.path.join(shadow, "opt_core"))
    open(os.path.join(shadow, "opt_core", "__init__.py"), "w").write('__version__ = "0.2.5"\n')   # installed_core() reads only this file's __version__ literal; no MANIFEST.json
    fake = os.path.join(tmp, "fake")
    os.makedirs(os.path.join(fake, "genie3"))
    open(os.path.join(fake, "genie3", "__init__.py"), "w").write("")
    site_dir = os.path.join(tmp, "site")
    os.makedirs(site_dir)
    shutil.copy(os.path.join(KIT, "opt", "genie3_opt_autoload.pth"), site_dir)
    assert not os.path.exists(os.path.join(tmp, "common"))
    return {"tmp": tmp, "kit": kit, "opt": os.path.join(kit, "opt"), "shadow": shadow, "fake": fake, "site": site_dir}


def _assert_refused(r, marker="stock ran"):
    """At the .pth trigger (an importer that is not `genie3 generate`) a kit mode is REFUSED by the env route's own name whatever the core: one
    `NOT ACTIVE: env-route: …` line, exit 3, the stock import never proceeds (`stock ran` absent), no traceback, no site.py swallowed error."""
    assert r.returncode == 3, (r.returncode, r.stderr[-800:], r.stdout[-300:])
    assert "[genie3-opt] NOT ACTIVE: env-route: " in r.stderr and r.stderr.count("[genie3-opt]") == 1 and "DECLINED" not in r.stderr, r.stderr[-800:]
    assert "Traceback" not in r.stderr and "Error processing line" not in r.stderr and marker not in r.stdout, (r.stderr[-800:], r.stdout[-300:])


CORES = ["absent", "stale"]


def test_package_import_and_status_stay_core_free(tree):
    """Importing the package, `status()` and the .pth module import nothing of the core and never refuse: the gate is at the trigger.
    `-S`: this process's own ambient site-packages must not be able to supply a real `opt_core` and mask the "absent" premise."""
    r = subprocess.run([sys.executable, "-S", "-c", "import genie3_opt as k, genie3_opt._autoload, sys; k.status(); assert 'opt_core' not in sys.modules; print('INERT')"],
                       env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": tree["opt"], "HOME": tree["tmp"], "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
                       capture_output=True, text=True, cwd=tree["tmp"])
    assert r.returncode == 0 and "INERT" in r.stdout and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stderr[-600:])


def test_interpreter_start_imports_the_shim_and_nothing_else(tree):
    """What the .pth line imports at interpreter start is `genie3_opt` + `genie3_opt._autoload` and NO other module of the package — whatever
    GENIE3_OPT says (unset, off, or a kit mode: installing the finder reads nothing); the line tag (`codes`) and the core gate load only when a
    named kit mode's finder wakes on `import genie3`."""
    probe = "import genie3_opt._autoload, sys; print(sorted(m for m in sys.modules if m == 'genie3_opt' or m.startswith('genie3_opt.')))"
    for env_mode in (None, "off", "fast", "exact"):
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": tree["opt"], "HOME": tree["tmp"], "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
        if env_mode is not None:
            env["GENIE3_OPT"] = env_mode
        r = subprocess.run([sys.executable, "-S", "-c", probe], env=env, capture_output=True, text=True, cwd=tree["tmp"])
        assert r.returncode == 0 and r.stdout.strip() == "['genie3_opt', 'genie3_opt._autoload']" and r.stderr == "", (env_mode, r.stdout, r.stderr[-400:])


def _pth_child(tree, core, mode_env):
    """The REAL generated .pth processed by site.addsitedir in a -S child: the hook installs the finder at start-up; `import genie3` is the trigger."""
    paths = ([tree["shadow"]] if core == "stale" else []) + [tree["opt"], tree["fake"]]
    prog = f"import sys, site; sys.path[:0] = {paths!r}; site.addsitedir({tree['site']!r}); import genie3; print('stock ran')"
    env = {"PATH": os.environ.get("PATH", ""), "HOME": tree["tmp"], "LANG": "C.UTF-8"}
    env.update(mode_env)
    return subprocess.run([sys.executable, "-S", "-c", prog], env=env, capture_output=True, text=True, cwd=tree["tmp"])


@pytest.mark.parametrize("core", CORES)
def test_real_pth_trigger_under_a_kit_mode(tree, core):
    for mode in ("exact", "fast"):
        _assert_refused(_pth_child(tree, core, {"GENIE3_OPT": mode}))


def test_real_pth_genie3_generate_becomes_the_design_verb_whose_core_gate_refuses_a_stale_core(tree):
    """Production's shape end to end: a console script named `genie3` at its `generate` command, the REAL .pth processed by site.addsitedir, a
    kit mode in GENIE3_OPT — the finder hands the process to `python -m genie3_opt design --mode fast <the same arguments>` (one NOTE line names
    the hand-over), and that process's statement one, the core pin gate, refuses the OLDER core on the path by name: exit 3, `stock ran` never
    prints, upstream never runs. (A good core would run the design verb proper: its activation report, refusals and outputs are design.py's,
    covered by the design tests.)"""
    script = os.path.join(tree["tmp"], "bin", "genie3")
    os.makedirs(os.path.dirname(script), exist_ok=True)
    open(script, "w").write(f"import sys, site; site.addsitedir({tree['site']!r}); import genie3; print('stock ran')\n")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": tree["tmp"], "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "GENIE3_OPT": "fast",
           "PYTHONPATH": os.pathsep.join([tree["shadow"], tree["opt"], tree["fake"]])}       # the exec'd design verb finds the package and the OLDER core through PYTHONPATH
    r = subprocess.run([sys.executable, "-S", script, "generate", "-c", os.path.join(tree["tmp"], "e.yaml"), "--num-shards", "2"], env=env, capture_output=True, text=True, cwd=tree["tmp"])
    assert r.returncode == 3 and "stock ran" not in r.stdout, (r.returncode, r.stderr[-900:], r.stdout[-300:])
    assert "[genie3-opt] NOTE env-route: GENIE3_OPT='fast' at `genie3 generate` — carried by the kit's design verb: -m genie3_opt design --mode fast -c " in r.stderr and "--num-shards 2" in r.stderr, r.stderr[-900:]
    assert "NOT ACTIVE: reason=core_mismatch:" in r.stderr and f"pinned >= v{PINNED_VERSION} at" in r.stderr and "installed v0.2.5 at" in r.stderr and "Traceback" not in r.stderr, r.stderr[-900:]


@pytest.mark.parametrize("core", CORES)
def test_real_pth_off_and_unset_stay_inert_whatever_the_core(tree, core):
    for mode_env in ({"GENIE3_OPT": "off"}, {}):
        r = _pth_child(tree, core, mode_env)
        assert r.returncode == 0 and "stock ran" in r.stdout and "NOT ACTIVE" not in r.stderr and "Error processing line" not in r.stderr, (mode_env, r.returncode, r.stderr[-600:])
