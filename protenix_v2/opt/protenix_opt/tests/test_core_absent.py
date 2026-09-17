"""An absent or older shared core: every entry route refuses by name with exit 3 before anything resolves (never a traceback, never a
run without the lever) — the CLI route (``python -m protenix_opt``), ``run.sh``'s verbs (the same entry) and the interpreter-start hook
(the ``.pth`` route). Older core = a shadow ``opt_core`` whose ``mem`` has none of the modules this tree imports; absent core = no
opt_core importable and no pinned copy beside the kit."""
import os
import re
import shutil
import subprocess
import sys

import pytest

from protenix_opt import _autoload, _core

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))            # <tree>/opt
TREE = os.path.dirname(OPT)                                                                   # <tree> (run.sh)
WORDS_OLDER = "[protenix-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned "          # the pre-import pin gate (_core_gate): a core that is not the pinned one
SHADOW_WORDS = "installed v0.0.0-shadow at "
WORDS_ABSENT = "[protenix-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v"


def _clean_env(pythonpath, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PROTENIX_OPT", "MODEL_OPT", "PTX_")) and k != "PYTHONPATH"}
    env.update(PYTHONPATH=os.pathsep.join(pythonpath), **extra)
    return env


@pytest.fixture
def shadow_core(tmp_path):
    fake = tmp_path / "shadow" / "opt_core"; (fake / "mem").mkdir(parents=True)
    (fake / "__init__.py").write_text('__version__ = "0.0.0-shadow"\n'); (fake / "mem" / "__init__.py").write_text("")
    return str(tmp_path / "shadow")


@pytest.fixture
def tree_without_core(tmp_path):
    """A copy of the kit tree (run.sh, configs, opt/ without the carried kits) with no common/opt_core beside it: the pin's relative path
    resolves to nothing and no opt_core is importable."""
    dst = tmp_path / "k" / "protenix_v2"
    shutil.copytree(TREE, dst, ignore=shutil.ignore_patterns("forward", "__pycache__", "*.egg-info", "stock", "tests"))
    os.symlink(os.path.join(OPT, "forward"), dst / "opt" / "forward")                          # the carried kits, by reference
    pp = dst / "opt" / "pyproject.toml"                                                          # the pin's path as released (relative): no core beside this copy
    pp.write_text(re.sub(r'(\[tool\.opt_core\][^\[]*?path = )"[^"]*"', r'\1"../../common/opt_core"', pp.read_text(), count=1, flags=re.S))
    shim = tmp_path / "bin"; shim.mkdir()                                                        # `python` = the suite's interpreter WITHOUT site dirs:
    (shim / "python").write_text(f"#!/bin/sh\nexec {sys.executable} -S \"$@\"\n"); (shim / "python").chmod(0o755)   # an installed core is invisible (absent)
    return str(dst)


def test_required_producers_are_importable_in_this_environment(tmp_path):
    assert _core.producer_refusal() is None                                    # the suite's own core has them
    import opt_core
    assert _autoload.missing_producers(os.path.dirname(opt_core.__file__)) == []
    (tmp_path / "opt_core" / "mem").mkdir(parents=True); (tmp_path / "opt_core" / "mem" / "ngpu.py").write_text("")
    assert _autoload.missing_producers(str(tmp_path / "opt_core")) == list(_autoload.REQUIRED_PRODUCERS[1:])        # the module-granular words' table
    assert _autoload.producer_refusal(str(tmp_path / "opt_core"), "0.0.1").startswith("reason=producer_missing:opt_core.mem.rowpair,")


@pytest.mark.parametrize("argv", [["check", "--mode", "exact"], ["pred", "--mode", "big", "--n_gpu", "2", "--input", "x"], ["pred", "--mode", "off", "--input", "x"]])
def test_cli_route_refuses_an_older_core_by_name(shadow_core, argv):
    r = subprocess.run([sys.executable, "-m", "protenix_opt", *argv], env=_clean_env([shadow_core, OPT]), capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and WORDS_OLDER in r.stderr and SHADOW_WORDS in r.stderr and "Traceback" not in r.stderr, (r.returncode, r.stderr[-700:])


def test_cli_route_refuses_an_absent_core_by_name(tree_without_core):
    r = subprocess.run([sys.executable, "-S", "-m", "protenix_opt", "check", "--mode", "fast"], env=_clean_env([os.path.join(tree_without_core, "opt")]),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and WORDS_ABSENT in r.stderr and "Traceback" not in r.stderr, (r.returncode, r.stderr[-700:])


@pytest.mark.parametrize("verb", [["check", "--mode", "exact"], ["pred", "--mode", "fast", "--input", "x", "--out_dir", "o"]])
def test_run_sh_verbs_refuse_an_absent_core_by_name(tree_without_core, verb):
    """run.sh's verbs are the real entry (`python -m protenix_opt <verb>`), never a bare `import protenix_opt` probe."""
    env = _clean_env([os.path.join(tree_without_core, "opt")])                                          # the package importable, no core anywhere:
    env["PATH"] = os.pathsep.join([os.path.join(os.path.dirname(os.path.dirname(tree_without_core)), "bin"), env.get("PATH", "")])   # `python -S` first on PATH
    r = subprocess.run(["bash", os.path.join(tree_without_core, "run.sh"), *verb], env=env, capture_output=True, text=True, timeout=120, cwd=tree_without_core)
    assert r.returncode == 3 and WORDS_ABSENT in r.stderr and "Traceback" not in r.stderr, (verb, r.returncode, r.stdout[-300:], r.stderr[-700:])


def test_run_sh_verbs_refuse_an_older_core_by_name(shadow_core, tmp_path):
    """The tree beside an older core: run.sh check with the shadow core first on PYTHONPATH."""
    env = _clean_env([shadow_core, OPT])                                                               # the older core first
    r = subprocess.run(["bash", os.path.join(TREE, "run.sh"), "check", "--mode", "exact"], env=env, capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert r.returncode == 3 and WORDS_OLDER in r.stderr and "Traceback" not in r.stderr, (r.returncode, r.stdout[-300:], r.stderr[-700:])


@pytest.mark.parametrize("selection", [{"PROTENIX_OPT": "exact"}, {"PROTENIX_OPT": "big", "PROTENIX_OPT_N_GPU": "2"}, {"PROTENIX_OPT_TP_ROUTE": "rowpair"}])
def test_pth_route_refuses_an_older_core_by_name_before_anything_resolves(shadow_core, selection):
    code = "import protenix_opt._autoload as a; a.install(); print('RESOLVED')"
    r = subprocess.run([sys.executable, "-c", code], env=_clean_env([shadow_core, OPT], **selection), capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and WORDS_OLDER in r.stderr and "RESOLVED" not in r.stdout and "Traceback" not in r.stderr, (selection, r.returncode, r.stderr[-700:])


def test_pth_route_refuses_an_absent_core_by_name(tree_without_core):
    code = "import protenix_opt._autoload as a; a.install(); print('RESOLVED')"
    r = subprocess.run([sys.executable, "-S", "-c", code], env=_clean_env([os.path.join(tree_without_core, "opt")], PROTENIX_OPT="fast"), capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and WORDS_ABSENT in r.stderr and "RESOLVED" not in r.stdout and "Traceback" not in r.stderr, (r.returncode, r.stderr[-700:])


def test_pth_route_of_a_stock_process_imports_nothing_of_the_core(shadow_core):
    """No selection: the hook returns without touching the core (a stock process stays stock even beside an older core)."""
    code = "import sys, protenix_opt._autoload as a; assert a.install() is None; assert 'opt_core' not in sys.modules, sorted(m for m in sys.modules if m.startswith('opt_core')); print('STOCK')"
    r = subprocess.run([sys.executable, "-c", code], env=_clean_env([shadow_core, OPT]), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "STOCK" in r.stdout, (r.returncode, r.stderr[-700:])


def test_config_env_probes_go_through_the_entry_and_refuse_an_older_core(shadow_core, tmp_path):
    """configs/h100.env keys the caches by its probe `python -m protenix_opt._stackkey`: beside an older core sourcing it returns 3 with the gate's line
    (no swallowed ImportError, no silently unset key)."""
    r = subprocess.run(["bash", "-c", f'source "{TREE}/configs/h100.env"; echo SOURCED rc=$?'], env=_clean_env([shadow_core, OPT]), capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert "SOURCED" not in r.stdout or "SOURCED rc=3" in r.stdout, r.stdout          # `source` returns 3 (a sourcing shell decides what to do with it)
    assert WORDS_OLDER in r.stderr and "Traceback" not in r.stderr, r.stderr[-700:]
    r = subprocess.run(["bash", "-c", f'source "{TREE}/configs/h100.env" || exit $?; echo CONTINUED'], env=_clean_env([shadow_core, OPT]), capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert r.returncode == 3 and "CONTINUED" not in r.stdout, (r.returncode, r.stdout, r.stderr[-400:])


def test_stack_key_probe_prints_a_key_or_the_word_unknown_and_is_not_a_command():
    r = subprocess.run([sys.executable, "-m", "protenix_opt._stackkey"], env=_clean_env([OPT] + [p for p in sys.path if p]), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and (r.stdout.strip() == "unknown" or r.stdout.strip().startswith("torch")), (r.returncode, r.stdout, r.stderr[-400:])
    r = subprocess.run([sys.executable, "-m", "protenix_opt", "jit-key"], env=_clean_env([OPT] + [p for p in sys.path if p]), capture_output=True, text=True, timeout=120)
    assert r.returncode == 2, "the CLI has three commands (pred, check, warm): anything else is a usage error"


@pytest.mark.parametrize("call", ["protenix_opt.enable('fast')", "protenix_opt.status()", "protenix_opt.MODES", "protenix_opt.stack"])
def test_in_process_route_refuses_an_older_core_by_name(shadow_core, call):
    """enable() / status() / the lazy module attributes: the pin gate runs before any module of the core is imported."""
    r = subprocess.run([sys.executable, "-c", f"import protenix_opt; {call}; print('RESOLVED')"], env=_clean_env([shadow_core, OPT]), capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and WORDS_OLDER in r.stderr and SHADOW_WORDS in r.stderr and "RESOLVED" not in r.stdout and "Traceback" not in r.stderr, (call, r.returncode, r.stderr[-700:])


def test_in_process_route_refuses_an_absent_core_by_name(tree_without_core):
    r = subprocess.run([sys.executable, "-S", "-c", "import protenix_opt; protenix_opt.enable('exact'); print('RESOLVED')"], env=_clean_env([os.path.join(tree_without_core, "opt")]),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and WORDS_ABSENT in r.stderr and "RESOLVED" not in r.stdout and "Traceback" not in r.stderr, (r.returncode, r.stderr[-700:])


def test_run_sh_with_config_refuses_an_older_core_by_name(shadow_core, tmp_path):
    """`run.sh check --config h100`: the config's cache-key probe is the gated entry; the file returns 3 and run.sh exits 3."""
    r = subprocess.run(["bash", os.path.join(TREE, "run.sh"), "check", "--config", "h100", "--mode", "exact"], env=_clean_env([shadow_core, OPT]), capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert r.returncode == 3 and WORDS_OLDER in r.stderr and "Traceback" not in r.stderr, (r.returncode, r.stdout[-300:], r.stderr[-700:])


def test_console_script_entry_is_the_gated_module():
    """`protenix-opt` (pyproject [project.scripts]) resolves to protenix_opt.__main__:main — the module whose statement one is the gate."""
    txt = open(os.path.join(OPT, "pyproject.toml"), encoding="utf-8").read()
    assert re.search(r'^protenix-opt\s*=\s*"protenix_opt\.__main__:main"', txt, re.M), "the console script must enter through the gated __main__"
