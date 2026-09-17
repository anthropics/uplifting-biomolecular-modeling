"""enable()/check()/status(): the gates, the refusal rules, idempotence and the late-activation rule — on the stub box (no GPU, no jax)."""
import os

import pytest

import af3_jax_opt
from af3_jax_opt import modes, report, stack
from .conftest import GPU_H100
import re
import subprocess
import sys
from af3_jax_opt import _autoload, report
from opt_core import report as core_report
from .conftest import PKG
from opt_core import gates, manifest as core_manifest
from .conftest import OPT
import importlib
from .conftest import CHILD_PYTHONPATH, OPT
import types
from af3_jax_opt import fpf_launch
import shutil
from af3_jax_opt import carry, modes, stack
from .conftest import FORWARD, FPF, KIT, OPT, PALLAS, TREE

EXACT_LEVERS = ["FIX1", "GLUT", "ATTNCFG", "L1", "WRITER"]
FAST_LEVERS = ["FIX1", "FPF_TRIMUL", "FPF_TRIATT", "FPF_HOIST", "TTR", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "TRIMUL_CD", "HOIST_LOGITS", "COND_SHARE", "ATOM_COND_HOIST", "LNP", "L1", "WRITER"]


def test_variant_required(box):
    rep = af3_jax_opt.enable("exact", None)
    assert rep["active"] is False and "a variant is required" in rep["reason"]
    assert af3_jax_opt.status()["reason"] == rep["reason"]


def test_unknown_mode_and_variant(box):
    with pytest.raises(ValueError, match="turbo"):
        af3_jax_opt.enable("turbo", "p2")
    rep = af3_jax_opt.enable("exact", "p3")
    assert not rep["active"] and "variant is required" in rep["reason"]


def test_gpu_gate(box, monkeypatch):
    """No visible GPU is the one hardware refusal. A device below the kit's reference line (24 GB, compute capability 8.0) ACTIVATES and is NAMED —
    one GPU NOTE per fact (stack.gpu_notes) — never refused: the modes run, a kernel that cannot run on the part names itself on its own line."""
    monkeypatch.setattr(stack, "gpu_info", lambda index=0: {"name": "NVIDIA A10", "memory_mib": 20000, "compute_cap": 8.6, "count": 1})
    rep = af3_jax_opt.check("off", "p2")
    assert rep["active"] and len(rep["gpu_notes"]) == 1 and "device memory" in rep["gpu_notes"][0] and "24 GB" in rep["gpu_notes"][0]
    monkeypatch.setattr(stack, "gpu_info", lambda index=0: {"name": "Tesla V100-SXM2-32GB", "memory_mib": 32768, "compute_cap": 7.0, "count": 1})
    stack._REPORT = None
    rep = af3_jax_opt.check("off", "p2")
    assert rep["active"] and len(rep["gpu_notes"]) == 1 and "compute capability 7.0 < 8.0" in rep["gpu_notes"][0]
    monkeypatch.setattr(stack, "gpu_info", lambda index=0: None)
    stack._REPORT = None
    rep = af3_jax_opt.check("off", "p2")
    assert not rep["active"] and "no NVIDIA GPU visible" in rep["reason"]
    assert stack.gate_gpu(GPU_H100) is None and stack.gpu_notes(GPU_H100) == []
    assert stack.gpu_notes({"name": "NVIDIA A10G", "memory_mib": 23028, "compute_cap": 8.6}) == []          # 24 GB nominal: at the line
    assert stack.gate_gpu(None) == "no NVIDIA GPU visible (nvidia-smi)"


def test_no_gpu(box, no_gpu):
    rep = af3_jax_opt.check("off", "p2")
    assert not rep["active"] and "no NVIDIA GPU" in rep["reason"] and rep["key"] is None
    assert report.activation_line(rep).startswith("[af3-jax-opt] NOT ACTIVE mode=off variant=p2 reason=")


def test_params_gate(box, monkeypatch):
    monkeypatch.setenv("AF3_JAX_PARAMS_ROOT", box.root + "/nowhere")
    rep = af3_jax_opt.check("off", "p2")
    assert not rep["active"] and "of3_ported_weights.bin.zst" in rep["reason"] and "run.sh install" in rep["reason"]


def test_off_is_active_without_a_cache(box):
    from af3_jax_opt import settings, stack
    rep = af3_jax_opt.enable("off", "p2")
    assert rep["active"] and rep["script"] == "run_alphafold.py" and rep["levers_applied"] == [] and rep["partial"] is False
    assert rep["key"] == "NVIDIA_H100_80GB_HBM3__jax0.10.2_jaxlib0.10.2" and rep["cache_dir"] == settings.CWD_JAX_CACHE   # off: ./jax under the pass's output dir (cli.pred resolves it) unless the caller names a --cache_dir
    assert stack.cache_dir(rep["key"], "exact") == box.cache_dir("off") and box.cache_dir("off").endswith(rep["key"]) and rep["stated"] == []   # exact's class <key>/: the one a stock --cache_dir shares
    assert rep["install"]["install"] == "present" and rep["launcher"] is None and rep["lever_env"] == {} and "carry" not in rep
    line = report.activation_line(rep)
    assert line.startswith("[af3-jax-opt] ACTIVE mode=off variant=p2 script=run_alphafold.py launcher=none") and "levers=" not in line and "install=present" in line


def test_a_cold_class_activates_named(box):
    """A cold cache class is a partial activation NAMED on the ACTIVE line (partial=cold_cache), never a refusal: this process builds the class it
    then belongs to; `warm` first is the deterministic recipe, not a precondition."""
    rep = af3_jax_opt.check("exact", "p2")
    assert rep["active"] and rep["partial"] is True and rep["cold_cache"] is True and rep["partial_conditions"] == ["cold_cache"]
    assert rep["cache_dir"] == box.cache_dir("exact") == box.cache_dir("off") and rep["levers_applied"] == EXACT_LEVERS and "cold_cache" in rep["note"]
    assert "partial=cold_cache" in report.activation_line(rep) and rep["reason"] is None
    box.warm_cache(mode="exact")
    stack._REPORT = None
    rep = af3_jax_opt.enable("exact", "p2")
    assert rep["active"] and rep["partial"] is False and rep["executables"] == 0 and rep["levers_applied"] == EXACT_LEVERS
    assert rep["launcher"] == ["levers_launch.py"] and rep["lever_env"] == modes.pallas_levers() and rep["carry"]["ok"] and rep["carry"]["files"] == 8
    line = report.activation_line(rep)
    assert "script=run_alphafold_fast.py launcher=levers_launch.py" in line and "levers=FIX1+GLUT+ATTNCFG" in line and "executables=0" in line
    assert "cold_cache" not in line and "install=present" in line


def test_fast_has_its_own_class(box):
    box.warm_cache(mode="exact")                                                # the shared class is warm ...
    rep = af3_jax_opt.check("fast", "p2")
    assert rep["active"] and rep["cold_cache"] is True and "partial=cold_cache" in report.activation_line(rep)   # ... fast's own is cold: named, active
    assert rep["cache_dir"] == box.cache_dir("fast") and rep["cache_dir"].endswith("__fast")
    box.warm_cache(mode="fast")                                                 # ... fast needs its own
    stack._REPORT = None
    rep = af3_jax_opt.enable("fast", "p2")
    assert rep["active"] and rep["levers_applied"] == FAST_LEVERS and rep["launcher"] == ["fpf_launch.py"]
    assert rep["lever_env"] == modes.lever_env("fast", FAST_LEVERS) and rep["lever_env"]["AF3_JAX_DATTN"] == "1" and rep["carry"]["files"] == 10 and rep["cache_class"] == "__fast"
    assert "launcher=fpf_launch.py" in report.activation_line(rep)


def test_idempotent_and_status(box):
    r1 = af3_jax_opt.enable("off", "p2"); r2 = af3_jax_opt.enable("off", "p2")
    assert r1 == r2 == af3_jax_opt.status()


def test_late_activation_rule(box):
    box.warm_cache(mode="exact")
    af3_jax_opt.enable("exact", "p2")
    stack.launched("exact", "p2")                                   # a model process ran in this interpreter
    assert af3_jax_opt.enable("exact", "p2")["active"]              # the same activation stays fine
    with pytest.raises(stack.ActivationError, match="late activation refused"):
        af3_jax_opt.enable("off", "p2")
    with pytest.raises(stack.ActivationError):
        af3_jax_opt.enable("fast", "p2")
    rep = af3_jax_opt.check("off", "p2")                            # the dry run reports instead of raising
    assert not rep["active"] and "late activation refused" in rep["reason"]


def test_install_must_be_the_pinned_stock(missing_stock_box):
    """A stock file absent from the install (here convert_of3_weights.py) refuses every mode by name: the pinned stock is the fork at
    the pin with stock/patches applied, every one of the nine files present (presence only — this check does not compare content)."""
    st = stack.install_state()
    assert st["install"] == "mixed" and st["files"]["convert_of3_weights.py"] == "missing" and st["counts"]["present"] == 8
    assert st["files"]["src/alphafold3/model/network/template_modules.py"] == "present"          # the box's site copy of the patched file is untouched
    for mode in modes.MODES:
        stack._REPORT = None
        rep = af3_jax_opt.check(mode, "p2")
        assert not rep["active"] and "does not carry the pinned stock" in rep["reason"] and "convert_of3_weights.py=missing" in rep["reason"], mode


def test_model_process_env_strips_and_sets(box, monkeypatch):
    monkeypatch.setenv("AF3_JAX_OPT", "exact"); monkeypatch.setenv("AF3_JAX_VARIANT", "p2")
    monkeypatch.setenv("AF3P_PAIR_CHUNK", "none"); monkeypatch.setenv("AF3_FLASHPAIRFORMER", "off"); monkeypatch.setenv("AF3_DIFFUSION_HOIST", "1")
    env = stack.model_process_env()
    assert not [k for k in env if k.startswith(("AF3_JAX_", "AF3P_", "AF3_FLASHPAIRFORMER", "AF3_DIFFUSION_HOIST"))]    # the caller's lever switches never reach the model process
    assert env["XLA_FLAGS"] == "--xla_gpu_enable_triton_gemm=false" and env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true" and env["XLA_CLIENT_MEM_FRACTION"] == "0.95"
    env = stack.model_process_env(mode_env=modes.mode_env("exact"))
    assert env["AF3P_GLU_T"] == "1" and env["AF3P_ATTN_CFG"] == "64,64,4,3" and "AF3P_PAIR_CHUNK" not in env               # the mode's own set, nothing else
    env = stack.model_process_env(mode_env=modes.mode_env("fast"))
    assert env["AF3_FLASHPAIRFORMER"] == "both" and env["AF3_DIFFUSION_HOIST"] == "1" and "AF3P_GLU_T" not in env
    assert "PYTHONPATH" not in {k for k in env if k not in os.environ}                          # mode off / no launcher: nothing of the tree on the path
    env = stack.model_process_env(mode_env=modes.mode_env("fast"), core_path=True)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == stack.core_dir()                            # lever modes: the pinned core's kernels by path
    monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.5")
    assert stack.model_process_env()["XLA_CLIENT_MEM_FRACTION"] == "0.5"                           # the caller's value is kept


# --- merged from test_core_missing.py (file consolidation, tests unchanged) ---

OPT = os.path.dirname(PKG)                                                 # af3_jax/opt: the package importable, the core NOT (python -S drops site-packages; PYTHONPATH names opt only)


def _run(args, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("AF3_JAX_", "PYTHON"))}
    e.update(PYTHONPATH=OPT, PYTHONDONTWRITEBYTECODE="1"); e.update(env)      # env may carry its own PYTHONPATH (a shadow core first)
    return subprocess.run([sys.executable, "-S"] + args, env=e, capture_output=True, text=True, timeout=60)


def test_constants_are_the_cores():
    assert _autoload.PREFIX == report.PREFIX == "[af3-jax-opt]" and _autoload.EXIT_NOT_ACTIVE == core_report.EXIT_NOT_ACTIVE == 3
    assert _autoload.TAG == report.TAG and _autoload.PREFIX == f"[{report.TAG}]"
    for rel in ("fpf_launch.py", "levers_launch.py", os.path.join("inprocess", "templates.py")):   # the model-process modules spell the prefix (stdlib only there): held equal here
        src = open(os.path.join(PKG, rel), encoding="utf-8").read()
        assert re.search(r'^PREFIX = "\[af3-jax-opt\]"', src, re.M), rel
    assert _autoload.core_missing(ImportError("x", name="opt_core.gates")) == "opt_core.gates" and _autoload.core_missing(ImportError("No module named 'opt_core'")) == "opt_core"
    assert _autoload.core_missing(ImportError("x", name="numpy")) == "" and _autoload.core_missing(ValueError("opt_core")) == ""


def _shadow(tmp_path, name, drop=None, version=None):
    """A copy of the imported core's package (opt_core/) first on a path: ``drop`` removes a file/dir under opt_core/ (the producer
    table must name it); ``version`` rewrites opt_core/__init__.py's __version__ literal — an OLDER version (the floor is a minimum:
    only an older core refuses this way, `core_mismatch`; a newer one passes)."""
    import re, shutil
    import opt_core
    root = os.path.dirname(os.path.dirname(opt_core.__file__))
    shadow = tmp_path / name; pkg = shadow / "opt_core"
    shutil.copytree(os.path.join(root, "opt_core"), str(pkg), ignore=shutil.ignore_patterns("__pycache__"))
    if drop:
        target = pkg.joinpath(*drop)
        shutil.rmtree(str(target)) if target.is_dir() else target.unlink()
    if version:
        init = pkg / "__init__.py"
        text = init.read_text(encoding="utf-8")
        text2 = re.sub(r'(?m)^__version__\s*=\s*"[^"]*"', f'__version__ = "{version}"', text, count=1)
        assert text2 != text, "could not rewrite opt_core/__init__.py's __version__ literal"
        init.write_text(text2, encoding="utf-8")
    return str(shadow)


GATE_MISSING = "[af3-jax-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v"       # _core_gate.py's words (the kit_template copy)
GATE_MISMATCH = "[af3-jax-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v"



def test_cli_without_the_core_is_not_active_exit_3():
    for verb in (["check", "--variant", "p2", "--mode", "exact"], ["check", "--variant", "p2", "--mode", "off"], ["pred", "--variant", "p2", "--mode", "fast", "--json_path", "x", "--output_dir", "y"]):
        r = _run(["-m", "af3_jax_opt"] + verb)
        assert r.returncode == 3, (verb, r.stderr)
        assert GATE_MISSING in r.stderr and "Traceback" not in r.stderr, r.stderr


def test_hook_without_the_core_is_not_active_exit_3():
    r = _run(["-c", "import af3_jax_opt._autoload, importlib; importlib.import_module('alphafold3')"], AF3_JAX_OPT="exact")
    assert r.returncode == 3 and r.stderr.startswith(GATE_MISSING) and "Traceback" not in r.stderr, r.stderr   # the gate's line, held for the model family's import
    r = _run(["-c", "import af3_jax_opt._autoload; print('unrelated process alive')"], AF3_JAX_OPT="exact")
    assert r.returncode == 0 and "alive" in r.stdout and r.stderr == "", (r.stdout, r.stderr)                   # an interpreter that never imports the model family is left alone
    r = _run(["-c", "import af3_jax_opt; print(af3_jax_opt.CORE_MISSING); af3_jax_opt.enable('exact', variant='p2')"])
    assert r.returncode == 3 and "opt_core" in r.stdout and GATE_MISSING in r.stderr and "Traceback" not in r.stderr, r.stderr   # the in-process route: the same gate, exit 3


def test_package_with_the_core_reports_none_missing():
    import af3_jax_opt
    assert af3_jax_opt.CORE_MISSING is None and af3_jax_opt.MODES == ("off", "exact", "fast", "big")
    from af3_jax_opt import modes
    assert modes.missing_producers() == [] and modes.missing_producers(("opt_core.mem.no_such_producer",)) == ["opt_core.mem.no_such_producer"]
    assert set(modes.BASE_PRODUCERS) <= set(modes.required_producers()) and "opt_core.mem.ngpu" in modes.required_producers()


def test_stale_core_is_core_mismatch_on_every_route(box, tmp_path):
    """A core PRESENT but OLDER than the pin (its __version__ literal is below the floor): `python -m` verbs, the in-process
    enable()/check(), the start-up hook at the model family's import, and `run.sh check --config h100` all print the pin gate's
    `core_mismatch: opt_core pinned >= v<want> at <path>, installed v<have> at <root>` and exit 3 — before any opt_core import."""
    stale = _shadow(tmp_path, "stale", version="0.2.5")
    pp = os.pathsep.join([stale, OPT])
    for verb in (["check", "--variant", "p2", "--mode", "exact"], ["warm", "--variant", "p2", "--mode", "fast", "--json_path", str(tmp_path)]):
        r = _run(["-m", "af3_jax_opt"] + verb, PYTHONPATH=pp)
        assert r.returncode == 3 and GATE_MISMATCH in r.stderr and "installed v0.2.5 at" in r.stderr and "Traceback" not in r.stderr, (verb, r.stderr)
    r = _run(["-c", "import af3_jax_opt as p; p.check('exact', variant='p2')"], PYTHONPATH=pp)
    assert r.returncode == 3 and GATE_MISMATCH in r.stderr and "Traceback" not in r.stderr, r.stderr
    r = _run(["-c", "import af3_jax_opt._autoload, importlib; importlib.import_module('alphafold3')"], PYTHONPATH=pp, AF3_JAX_OPT="fast")
    assert r.returncode == 3 and r.stderr.startswith(GATE_MISMATCH) and "Traceback" not in r.stderr, r.stderr
    from .conftest import make_venv, python_on_path
    py_s = make_venv(str(tmp_path / "venv_s"), hook=True, core_path=stale)
    bin_s = tmp_path / "bin_s"; bin_s.mkdir(); python_on_path(str(bin_s), py_s)
    rc, out, err = box.run_sh("check", "--variant", "p2", "--mode", "exact", env={"PATH": str(bin_s) + os.pathsep + box.bin + os.pathsep + os.environ.get("PATH", ""), "PYTHONPATH": ""})
    assert rc == 3 and GATE_MISMATCH in err and "Traceback" not in err, err


def test_pinned_core_lacking_a_producer_is_producer_missing(tmp_path):
    """A core that satisfies the pin (path + minimum version) but which lacks a module this package imports (opt_core.mem.ngpu removed): the pin gate passes
    and the producer table names it — `python -m` verbs and the in-process check()/enable() answer NOT ACTIVE reason=producer_missing:
    opt_core.mem.ngpu (opt_core >= 0.4.1), exit 3 / inactive report, never a ModuleNotFoundError; with opt_core.mem removed entirely the
    package import names core_missing:opt_core.mem… by its one ImportError guard."""
    lacking = _shadow(tmp_path, "lacking", drop=("mem", "ngpu.py"))
    pp = os.pathsep.join([lacking, OPT])
    for verb in (["check", "--variant", "p2", "--mode", "exact"], ["check", "--variant", "p2", "--mode", "off"]):
        r = _run(["-m", "af3_jax_opt"] + verb, PYTHONPATH=pp)
        assert r.returncode == 3 and "NOT ACTIVE" in r.stderr and "reason=producer_missing:opt_core.mem.ngpu" in r.stderr and "opt_core >= 0.4.1" in r.stderr and "Traceback" not in r.stderr, (verb, r.stderr)
    code = ("import json, af3_jax_opt as p\nout = {}\nfor verb in ('check', 'enable'):\n    try:\n        r = getattr(p, verb)('exact', variant='p2')\n        out[verb] = str(r.get('reason'))\n"
            "    except p.ActivationError as e:\n        out[verb] = str(e)\nprint(json.dumps(out))\n")
    r = _run(["-c", code], PYTHONPATH=pp)
    assert r.returncode == 0 and "Traceback" not in r.stderr and "ModuleNotFoundError" not in r.stderr, (r.stdout, r.stderr)
    got = __import__("json").loads(r.stdout.strip().splitlines()[-1])
    assert all("producer_missing:opt_core.mem.ngpu" in v for v in got.values()), got
    nomem = _shadow(tmp_path, "nomem", drop=("mem",))
    r = _run(["-m", "af3_jax_opt", "check", "--variant", "p2", "--mode", "exact"], PYTHONPATH=os.pathsep.join([nomem, OPT]))
    assert r.returncode == 3 and "NOT ACTIVE" in r.stderr and ("core_missing:opt_core.mem" in r.stderr or "producer_missing:opt_core.mem" in r.stderr) and "Traceback" not in r.stderr, r.stderr


def test_run_sh_without_the_core_is_not_active_exit_3(box, tmp_path):
    """The run.sh route with the shared core ABSENT (venv C: the package and the start-up hook in site, no core): `run.sh check` /
    `run.sh pred` / `--mode off` exit 3 with the pin gate's core_missing line from the config's probe — nothing filled, no traceback."""
    from .conftest import make_venv, python_on_path
    py_c = make_venv(str(tmp_path / "venv_c"), hook=True, core=False)
    bin_c = tmp_path / "bin_c"; bin_c.mkdir(); python_on_path(str(bin_c), py_c)
    env_c = {"PATH": str(bin_c) + os.pathsep + box.bin + os.pathsep + os.environ.get("PATH", ""), "PYTHONPATH": ""}
    for cmd in (("check", "--variant", "p2", "--mode", "exact"), ("pred", "--variant", "p2", "--mode", "fast", "--json_path", "x.json", "--output_dir", "y"), ("check", "--variant", "p2", "--mode", "off")):
        rc, out, err = box.run_sh(*cmd, env=env_c)
        assert rc == 3, (cmd, out, err)
        assert GATE_MISSING in err and "Traceback" not in err, err


# --- merged from test_core_pin.py (file consolidation, tests unchanged) ---

PYPROJECT = os.path.join(OPT, "pyproject.toml")
CORE_DIR = os.path.normpath(os.path.join(OPT, "..", "..", "common", "opt_core"))


def test_pin_names_the_sibling_core_by_path_and_minimum_version():
    pin = gates.core_pin(PYPROJECT)
    assert os.path.normpath(pin["abs_path"]) == CORE_DIR and os.path.isdir(os.path.join(CORE_DIR, "opt_core"))
    installed_version = open(os.path.join(CORE_DIR, "opt_core", "__init__.py"), encoding="utf-8").read().split('__version__ = "')[1].split('"')[0]
    assert gates.version_tuple(installed_version) >= gates.version_tuple(pin["version"]), f"pinned >= v{pin['version']}, the sibling core is v{installed_version}"


def test_imported_core_is_the_pinned_one():
    g = gates.core_pin_check(PYPROJECT)
    assert g.ok, g.reason


def test_build_backend_is_the_template():
    assert gates.sha256_file(os.path.join(OPT, "_build_backend.py")) == gates.sha256_file(os.path.join(CORE_DIR, "kit_template", "_build_backend.py"))


def test_activation_report_carries_the_core(box):
    """rep["core"] (built from the gate facts) carries version + package_dir + the pin, whatever core_block()'s own transitional shape
    is (the core track owns that function; this test reads only the stable field every shape has, "version")."""
    from af3_jax_opt import stack
    rep = stack.check("off", "p2")
    pin = gates.core_pin(PYPROJECT)
    assert rep["core"]["ok"] is True and rep["core"]["version"] and rep["core"]["package_dir"]
    assert rep["core"]["pinned"] == {"path": pin["path"], "version": pin["version"]}
    assert core_manifest.core_block()["version"] == rep["core"]["version"]


# --- merged from test_startup.py (file consolidation, tests unchanged) ---

def test_import_is_light():
    code = ("import sys; import af3_jax_opt; mods = sorted(m for m in sys.modules if m.split('.')[0] in ('jax', 'jaxlib', 'numpy', 'torch', 'alphafold3', 'haiku')); "
            "print(mods); print(af3_jax_opt.status())")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": CHILD_PYTHONPATH}, timeout=120)
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines()[0] == "[]"
    assert "'active': False" in p.stdout.splitlines()[1] and "no activation" in p.stdout.splitlines()[1]


def test_startup_hook_imports_nothing_but_itself_when_unset_or_off():
    """The .pth hook's import (`import af3_jax_opt._autoload`) with the mode unset or `off` leaves EXACTLY {af3_jax_opt, af3_jax_opt._autoload} of the
    kit and NOTHING of the shared core in sys.modules (the package's names resolve lazily, PEP 562); a kit mode arms the finder and may import more."""
    code = ("import sys; import af3_jax_opt._autoload; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('af3_jax_opt', 'opt_core', 'jax', 'jaxlib', 'numpy', 'torch', 'alphafold3', 'haiku')))")
    base = {k: v for k, v in os.environ.items() if not k.startswith("AF3_JAX_")}
    for extra in ({}, {"AF3_JAX_OPT": "off"}, {"AF3_JAX_OPT": "OFF"}):
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**base, "PYTHONPATH": CHILD_PYTHONPATH, **extra}, timeout=120)
        assert p.returncode == 0, (extra, p.stderr)
        assert p.stdout.strip() == "['af3_jax_opt', 'af3_jax_opt._autoload']", (extra, p.stdout)
    p = subprocess.run([sys.executable, "-c", "import af3_jax_opt; print(af3_jax_opt.MODES[0], af3_jax_opt.enable is af3_jax_opt.activate, af3_jax_opt.CORE_MISSING)"],
                       capture_output=True, text=True, env={**base, "PYTHONPATH": CHILD_PYTHONPATH}, timeout=120)
    assert p.returncode == 0 and p.stdout.split() == ["off", "True", "None"], (p.stdout, p.stderr)       # the lazy names resolve on first use


def test_version_flag():
    p = subprocess.run([sys.executable, "-m", "af3_jax_opt", "--version"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": CHILD_PYTHONPATH}, timeout=120)
    assert p.returncode == 0 and p.stdout.strip().startswith("af3_jax_opt ")


def test_pth_is_the_generated_guard():
    """The .pth is the build backend's generated text for this package (opt/_build_backend.py pth_text: a header + one guarded import of
    af3_jax_opt._autoload; the build refuses any other text)."""
    spec = importlib.util.spec_from_file_location("af3_jax_build_backend", os.path.join(OPT, "_build_backend.py"))
    backend = importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
    with open(os.path.join(OPT, "af3_jax_opt_autoload.pth"), encoding="utf-8") as f:
        text = f.read()
    assert text == backend.pth_text("af3_jax_opt", "AF3_JAX_OPT", "af3-jax-opt", 3) and "import af3_jax_opt._autoload" in text


def test_autoload_applies_nothing_and_says_so(monkeypatch):
    """Under a kit mode the hook arms the shared core's finder (opt_core.autoload.Finder, refusal path) with the kit's reason; off / unset arm nothing."""
    import af3_jax_opt._autoload as al
    from opt_core.autoload import Finder
    monkeypatch.setenv("AF3_JAX_OPT", "exact")
    al = importlib.reload(al)
    try:
        assert al._STATE["armed"] and isinstance(al.FINDER, Finder) and sys.meta_path[0] is al.FINDER
        assert al.FINDER.refuse == al._STATE["refusal"] == "[af3-jax-opt] NOT ACTIVE: AF3_JAX_OPT=exact is honoured by the wrapper command only (af3-jax-opt pred / run.sh pred); this process would run stock; exit 3"
        assert al.FINDER.spec.triggers == ("alphafold3",) and al.FINDER.spec.exit_not_active == 3 and al.FINDER.spec.package == "af3_jax_opt"
    finally:
        al.FINDER.remove()
    monkeypatch.setenv("AF3_JAX_OPT", "off")
    al = importlib.reload(al)
    assert not al._STATE["armed"] and al.FINDER is None
    monkeypatch.delenv("AF3_JAX_OPT")
    al = importlib.reload(al)
    assert not al._STATE["armed"] and al.FINDER is None


PROBE = "import af3_jax_opt._autoload, importlib.util; importlib.util.find_spec('alphafold3'); print('PROBED'); import alphafold3; print('REACHED')"
IMPORT = "import af3_jax_opt._autoload; import alphafold3; print('REACHED')"


def test_autoload_unknown_selection_exits_not_active(tmp_path):
    """AF3_JAX_OPT outside the mode table + an import of the model family → the kit's NOT ACTIVE line and exit 3 (never a silent stock run).
    A bare find_spec probe leaves the finder armed (the core's contract): the import that follows it is the one refused."""
    (tmp_path / "alphafold3").mkdir(); (tmp_path / "alphafold3" / "__init__.py").write_text("")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), CHILD_PYTHONPATH)), "AF3_JAX_OPT": "bogus"}
    p = subprocess.run([sys.executable, "-c", IMPORT], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 3 and "[af3-jax-opt] NOT ACTIVE: AF3_JAX_OPT='bogus' is not a mode (off|exact|fast|big); exit 3" in p.stderr and "REACHED" not in p.stdout
    p = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 3 and "PROBED" in p.stdout and "REACHED" not in p.stdout and "NOT ACTIVE" in p.stderr, "a probe leaves the finder armed; the import after it is refused"


def test_autoload_known_mode_under_direct_import_exits_not_active(tmp_path):
    """A known mode cannot be applied in the importing process (the levers are the wrapper's command line): NOT ACTIVE + exit 3 at the import (after a probe too)."""
    (tmp_path / "alphafold3").mkdir(); (tmp_path / "alphafold3" / "__init__.py").write_text("")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), CHILD_PYTHONPATH)), "AF3_JAX_OPT": "exact"}
    for code in (PROBE, IMPORT):
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
        assert p.returncode == 3 and "[af3-jax-opt] NOT ACTIVE: AF3_JAX_OPT=exact is honoured by the wrapper command only" in p.stderr and "REACHED" not in p.stdout, (code, p.stdout, p.stderr)
    env["AF3_JAX_OPT"] = "off"
    p = subprocess.run([sys.executable, "-c", "import af3_jax_opt._autoload as al; import alphafold3; print('armed', al._STATE['armed'])"], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0 and "armed False" in p.stdout and "NOT ACTIVE" not in p.stderr


LEVER_SWITCHES = ("AF3_JAX_DATTN", "AF3_JAX_TTR", "AF3_JAX_TRIATT_XLA", "AF3_JAX_SAMPLER_BF16", "AF3_JAX_ATOM_ATTN", "AF3_JAX_TRIMUL_CD", "AF3_JAX_LNP", "AF3_JAX_HOIST_LOGITS", "AF3_JAX_COND_SHARE", "AF3_JAX_ATOM_COND_HOIST")                            # the tree levers' model-process switches (inprocess/*.py ENV_SWITCH)
DECLARED = ("AF3_JAX_OPT", "AF3_JAX_VARIANT", "AF3_JAX_OPT_HOME", "AF3_JAX_REPO", "AF3_JAX_PY", "AF3_JAX_PARAMS_ROOT", "AF3_JAX_CACHE_ROOT", "AF3_JAX_N_GPU") + LEVER_SWITCHES


def test_declared_names_are_spelled_once_per_process_stage():
    """The kit reads exactly these AF3_JAX_* names (the package's deployment variables and axis + the tree levers' model-process switches); the
    hook (nothing imported at interpreter start) and stack.py spell the same set and the same prefix, and the prefix is the package's own in
    stock/PINS.json (stock_proof.prefix_owners)."""
    import af3_jax_opt._autoload as al
    from af3_jax_opt import stack
    assert stack.DECLARED_ENV == al.DECLARED_ENV == DECLARED and stack.ENV_PREFIX == al.ENV_PREFIX == "AF3_JAX_"
    assert all(n.startswith(stack.ENV_PREFIX) for n in DECLARED) and len(set(DECLARED)) == len(DECLARED)
    owners = stack.pins()["stock_proof"]["prefix_owners"]
    assert stack.ENV_PREFIX in owners and "af3_jax_opt" in owners[stack.ENV_PREFIX] and stack.ENV_PREFIX in stack.pins()["stock_proof"]["must_be_absent_prefixes"]
    assert set(DECLARED) == {stack.ENV_MODE, stack.ENV_VARIANT, stack.ENV_HOME, stack.ENV_REPO, stack.ENV_PY, stack.ENV_PARAMS_ROOT, stack.ENV_CACHE_ROOT, stack.ENV_N_GPU} | set(stack.LEVER_SWITCH_ENV)
    assert stack.LEVER_SWITCH_ENV == LEVER_SWITCHES


def test_every_af3_jax_name_the_kit_reads_is_declared():
    """Completeness, so an omission cannot recur: every AF3_JAX_* name spelled anywhere in the package's code (deployment variables, the axis,
    the in-process levers' switches — inprocess/*.py ENV_SWITCH and every variable modes.TREE_LEVER_ENV writes for the model process) is a
    member of DECLARED_ENV, or is one of the memory-lever switches the wrapper itself writes into a --n_gpu P > 1 model process (never read from
    the caller). A model process launched by the wrapper carries the lever switches; an interpreter with the kit's start-up hook installed must
    accept them, so they are declared."""
    import ast
    import glob
    import re
    from af3_jax_opt import modes, stack
    pkg = os.path.dirname(os.path.abspath(stack.__file__))
    seen = {}
    for path in glob.glob(os.path.join(pkg, "**", "*.py"), recursive=True):
        if os.sep + "tests" + os.sep in path:
            continue
        for m in re.finditer(r"AF3_JAX_[A-Z0-9_]+", open(path, encoding="utf-8").read()):
            seen.setdefault(m.group(0), set()).add(os.path.relpath(path, pkg))
    from af3_jax_opt import _autoload, big
    stray = {n: sorted(f) for n, f in seen.items() if n not in stack.DECLARED_ENV}
    assert not stray, f"AF3_JAX_* names in the package that DECLARED_ENV does not carry: {stray}"
    switches = set()
    for rel in set(modes.INPROCESS_MODULES.values()) | {os.path.join("inprocess", f) for f in os.listdir(os.path.join(pkg, "inprocess")) if f.endswith(".py")}:
        tree = ast.parse(open(os.path.join(pkg, rel), encoding="utf-8").read())   # read statically (executing a lever module here would register its levers twice)
        for node in tree.body:                                                     # every in-process lever module: its top-level ENV_SWITCH (when it has one) is declared
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ENV_SWITCH" for t in node.targets) and isinstance(node.value, ast.Constant):
                switches.add(node.value.value)
    for env in modes.TREE_LEVER_ENV.values():                              # and every variable the mode table writes for a tree lever
        switches |= set(env)
    assert switches and switches <= set(stack.DECLARED_ENV) and switches == set(stack.LEVER_SWITCH_ENV), (sorted(switches), stack.LEVER_SWITCH_ENV)


def test_lever_switches_are_accepted_by_the_hook_and_the_wrapper(tmp_path):
    """AF3_JAX_TTR / AF3_JAX_DATTN set (as in a model process the wrapper launched for a composition naming those levers): the
    start-up hook arms nothing and refuses nothing, a direct import of the model family proceeds, and the wrapper's own undeclared-name gate passes."""
    from af3_jax_opt import stack
    switches = {"AF3_JAX_TTR": "1", "AF3_JAX_DATTN": "1"}
    assert stack.undeclared_env(switches) == [] and stack.undeclared_env({**switches, "AF3_JAX_TTRX": "1"}) == ["AF3_JAX_TTRX"]
    (tmp_path / "alphafold3").mkdir(); (tmp_path / "alphafold3" / "__init__.py").write_text("")
    env = {k: v for k, v in os.environ.items() if not k.startswith("AF3_JAX_")}
    env.update(switches, PYTHONPATH=os.pathsep.join((str(tmp_path), CHILD_PYTHONPATH)))
    p = subprocess.run([sys.executable, "-c", "import af3_jax_opt._autoload as al; import alphafold3; print('armed', al._STATE['armed'], al._STATE['undeclared'])"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0 and "armed False []" in p.stdout and "NOT ACTIVE" not in p.stderr, p.stderr


def test_undeclared_package_variable_is_refused(tmp_path, box):
    """Any AF3_JAX_* name the package does not read (a mistyped switch, a foreign name under the prefix) is refused by the wrapper (rc 3,
    one line) and trips the hook under a direct import; every declared name passes."""
    from af3_jax_opt import cli, stack
    assert stack.undeclared_env({n: "x" for n in DECLARED}) == [] and stack.undeclared_env({"AF3_JAX_OPT": "exact", "AF3_JAX_OPT_HOME": "/x", "PATH": "/bin"}) == []
    assert stack.undeclared_env({"AF3_JAX_OPT_MODE": "exact", "AF3_JAX_OPTIONS": "1", "AF3_JAX_FOO": "1", "AF3P_GLU_T": "1"}) == ["AF3_JAX_FOO", "AF3_JAX_OPTIONS", "AF3_JAX_OPT_MODE"]
    os.environ["AF3_JAX_OPT_MODE"] = "exact"
    try:
        rc = cli.main(["check", "--variant", "p2", "--mode", "off"])
    finally:
        del os.environ["AF3_JAX_OPT_MODE"]
    assert rc == 3
    (tmp_path / "alphafold3").mkdir(); (tmp_path / "alphafold3" / "__init__.py").write_text("")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), CHILD_PYTHONPATH)), "AF3_JAX_OPT_MODE": "exact"}
    p = subprocess.run([sys.executable, "-c", "import af3_jax_opt._autoload; import alphafold3; print('REACHED')"], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 3 and f"NOT ACTIVE: undeclared variable(s) AF3_JAX_OPT_MODE (declared: {', '.join(DECLARED)}); exit 3" in p.stderr and "REACHED" not in p.stdout
    env = {**env, "AF3_JAX_FOO": "1"}; del env["AF3_JAX_OPT_MODE"]
    p = subprocess.run([sys.executable, "-c", "import af3_jax_opt._autoload; import alphafold3; print('REACHED')"], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 3 and "undeclared variable(s) AF3_JAX_FOO" in p.stderr and "REACHED" not in p.stdout
    env = {k: v for k, v in env.items() if k != "AF3_JAX_FOO"}
    p = subprocess.run([sys.executable, "-c", "import af3_jax_opt._autoload as al; import alphafold3; print('armed', al._STATE['armed'])"], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0 and "armed False" in p.stdout, p.stderr                                                     # the declared deployment names alone arm nothing


# --- merged from test_oom.py (file consolidation, tests unchanged) ---

class XlaRuntimeError(RuntimeError):
    """jaxlib's runtime error by class name — opt_core.oom recognises the name with RESOURCE_EXHAUSTED in the message (no jax import here)."""


def _lever_module(exc):
    def install():
        raise exc
    return types.SimpleNamespace(wanted=lambda environ=None: True, install=install, report=lambda: {"installed": False, "traced": 0, "sites": {}})


def test_oom_propagates_through_the_fast_launchers_lever_install(monkeypatch, capsys):
    """fpf_launch._install_tree_levers is the model process's first served step (DATTN / TTR before the add-on's import): a lever whose
    install runs out of device memory ends the process with that error; any other failure keeps its named route (printed, the lever absent,
    levers_short at exit)."""
    monkeypatch.setattr(fpf_launch, "_TREE", {name: None for name, _lever in fpf_launch.TREE_LEVERS})
    oom = XlaRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 68.51GiB.")
    monkeypatch.setattr(fpf_launch, "load_inprocess", lambda name: _lever_module(oom))
    with pytest.raises(XlaRuntimeError):
        fpf_launch._install_tree_levers()
    host_oom = MemoryError("featurisation buffers")
    monkeypatch.setattr(fpf_launch, "load_inprocess", lambda name: _lever_module(host_oom))
    with pytest.raises(MemoryError):
        fpf_launch._install_tree_levers()
    other = RuntimeError("no kernel image is available for execution on the device")
    monkeypatch.setattr(fpf_launch, "load_inprocess", lambda name: _lever_module(other))
    fpf_launch._install_tree_levers()                                            # the named route: no raise
    out = capsys.readouterr().out
    assert out.count("install failed: RuntimeError: no kernel image") == len(fpf_launch.TREE_LEVERS) and "RESOURCE_EXHAUSTED" not in out


# --- merged from test_carry.py (file consolidation, tests unchanged) ---

N_FILES = {"fast_inference": 5, "pallas": 3, "fpf": 5}
READ_AT_RUN_TIME = (os.path.join(KIT, "rows.sh"), os.path.join(KIT, "patches", "patched_files", "run_alphafold.py"),
                    os.path.join(KIT, "tests", "inputs"), os.path.join(PALLAS, "patches", "af3_pallas_levers.py"),
                    os.path.join(FPF, "run_alphafold_flashpairformer.py"),
                    os.path.join(FPF, "af3_flashpairformer", "__init__.py"))


def test_each_kit_is_carried_with_its_expected_file_count():
    res = carry.carry_check()
    assert res["ok"], res
    assert res["files"] == sum(N_FILES.values()) == 13
    for name, n in N_FILES.items():
        assert carry.carry_check([name])["files"] == n, name


def test_each_mode_gates_its_own_kits():
    assert set(modes.KIT_MODES) == set(modes.MODES)                          # every mode has a row (a loop over an empty table proves nothing)
    for mode, spec in modes.KIT_MODES.items():
        assert set(spec["kits"]) <= set(stack.KITS)
        assert carry.carry_check(spec["kits"])["files"] == sum(N_FILES[k] for k in spec["kits"])


def test_the_run_time_files_are_carried():
    for p in READ_AT_RUN_TIME:
        assert os.path.exists(p), p
    assert sorted(os.listdir(FORWARD)) == ["fast_inference", "flashpairformer", "pallas_addon"]


def _opt_copy(tmp_path):
    """A copy of opt/ the package can be pointed at (AF3_JAX_OPT_HOME) without touching the tree."""
    dst = os.path.join(str(tmp_path), "opt")

    def ignore(d, names):
        skip = {n for n in names if n == "__pycache__" or n.endswith(".pyc")}
        if os.path.basename(d) == "af3_jax_opt":
            skip.add("tests")                                                # the package's own tests, not the add-ons'
        return skip
    shutil.copytree(OPT, dst, ignore=ignore)
    return dst


def test_missing_kit_file_refuses_the_mode_by_name(box, monkeypatch, tmp_path):
    home = _opt_copy(tmp_path)
    monkeypatch.setenv("AF3_JAX_OPT_HOME", home); monkeypatch.setenv("MODEL_OPT", TREE)
    stack._CACHE.clear()
    box.warm_cache(mode="exact")                                        # warm the class's cache FIRST: warming needs the kit file present
    target = os.path.join(home, "forward", "pallas_addon", "patches", "af3_pallas_levers.py")   # the pallas add-on's identifying file (the add-on table in stack)
    os.remove(target)
    rep = stack.check("exact", "p2")
    assert not rep["active"] and "kit file forward/pallas_addon/patches/af3_pallas_levers.py is missing under opt/forward/" in rep["reason"]
    rep = stack.check("off", "p2")
    assert rep["active"]                                                    # off runs no kit file
    box.warm_cache(mode="fast")
    assert stack.check("fast", "p2")["active"]                              # fast does not run the Pallas kit


def test_missing_kit_file_fails_the_carry_check(monkeypatch, tmp_path):
    home = _opt_copy(tmp_path)
    monkeypatch.setenv("AF3_JAX_OPT_HOME", home); monkeypatch.setenv("MODEL_OPT", TREE)
    stack._CACHE.clear()
    os.remove(os.path.join(home, "forward", "flashpairformer", "af3_flashpairformer", "__init__.py"))   # the fpf kit's identifying file
    res = carry.carry_check(["fpf"])
    assert not res["ok"] and res["missing"] == [os.path.join("forward", "flashpairformer", "af3_flashpairformer", "__init__.py")]
    res = carry.carry_check(["fast_inference"])                          # an untouched kit is unaffected
    assert res["ok"] and not res["missing"]
