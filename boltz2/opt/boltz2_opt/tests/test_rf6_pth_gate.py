"""RF-6 (D77.2): the ENV route's hook-live gate — two venvs, no network. A: the kit's .pth in the venv's site (its lines: the tree's opt/ and
core on sys.path, then the hook import) → the hook is live at interpreter start → the gate passes and run.sh goes on to its next precondition;
B: the package importable via PYTHONPATH but NO .pth → `BOLTZ2_OPT=<mode> run.sh pred` (no --mode) refuses by name (exit 3, the one line,
diagnostic 'absent from the searched sites'); a .pth copied beside a PYTHONPATH entry → 'present … not processed'; `run.sh pred --mode <kit
mode>` is NOT gated (activation by construction); --mode off exempt. Removing the gate from run.sh makes B's env route pass it — this test
then fails."""
import os
import re
import shutil
import subprocess
import sys
import venv

from .. import stack

TREE = stack.tree_dir()
GATE = os.path.join(TREE, "opt", "boltz2_opt", "pth_gate.py")
RUN = os.path.join(TREE, "run.sh")
OPT = os.path.join(TREE, "opt")
CORE = os.path.normpath(os.path.join(TREE, "..", "common", "opt_core"))
PTH = "boltz2_opt_autoload.pth"


def _venv(path, with_pth):
    venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(path)
    py = os.path.join(path, "bin", "python")
    sp = subprocess.run([py, "-I", "-c", "import site; print(site.getsitepackages()[0])"], capture_output=True, text=True, check=True).stdout.strip()
    if with_pth:
        import opt_core                                        # the kit install puts the shared core beside the package (run.sh install): venv A has both
        core_parent = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))
        with open(os.path.join(sp, PTH), "w") as fh:
            fh.write(OPT + "\n" + core_parent + "\nimport boltz2_opt._autoload\n")   # the shipped .pth's import line, after the tree's package (and its core) as path lines (the hook imports nothing of the core with BOLTZ2_OPT unset)
    return py, sp


def _run_sh(py, args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("BOLTZ2_OPT", "PYTHONPATH")}
    env["PATH"] = os.path.dirname(py) + os.pathsep + env.get("PATH", ""); env.update(env_extra or {})
    return subprocess.run(["bash", RUN, *args], capture_output=True, text=True, env=env, timeout=120)


PRED = ["pred", "--input", "x.yaml", "--out_dir", "o"]


def test_gate_script_is_hook_live_and_names_the_diagnostic(tmp_path):
    py_a, _ = _venv(str(tmp_path / "A"), with_pth=True)
    py_b, sp_b = _venv(str(tmp_path / "B"), with_pth=False)
    a = subprocess.run([py_a, GATE], capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"})
    assert a.returncode == 0 and a.stderr == "", a.stderr
    b = subprocess.run([py_b, GATE], capture_output=True, text=True, env={**{k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}, "PYTHONPATH": OPT})   # importable via PYTHONPATH: not live
    assert b.returncode == 3 and b.stderr.startswith(f"run.sh: NOT ACTIVE (RF-6): the kit's hook boltz2_opt._autoload is not live in {py_b}") and "run.sh install" in b.stderr, b.stderr
    assert f"present at {os.path.join(OPT, PTH)} beside a sys.path/PYTHONPATH entry" in b.stderr and "processes no .pth file there" in b.stderr, b.stderr   # the tree ships opt/<pth> beside the package: the diagnostic names that copy
    pkg_copy = tmp_path / "pkgcopy"; shutil.copytree(os.path.join(OPT, "boltz2_opt"), pkg_copy / "boltz2_opt")                                   # the package alone on PYTHONPATH, no .pth anywhere: absent
    b2 = subprocess.run([py_b, GATE], capture_output=True, text=True, env={**{k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}, "PYTHONPATH": str(pkg_copy)})
    assert b2.returncode == 3 and "absent from the searched sites" in b2.stderr and sp_b in b2.stderr, b2.stderr
    beside = tmp_path / "pp"; beside.mkdir(); shutil.copy(os.path.join(OPT, PTH), beside / PTH)      # a copy where site.py processes nothing
    c = subprocess.run([py_b, GATE], capture_output=True, text=True, env={**{k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}, "PYTHONPATH": str(beside) + os.pathsep + OPT})
    assert c.returncode == 3 and f"present at {beside / PTH} beside a sys.path/PYTHONPATH entry" in c.stderr and "processes no .pth file there" in c.stderr, c.stderr


def test_run_sh_gates_the_env_route_only(tmp_path):
    py_a, _ = _venv(str(tmp_path / "A"), with_pth=True)
    py_b, _ = _venv(str(tmp_path / "B"), with_pth=False)
    r = _run_sh(py_b, PRED, {"BOLTZ2_OPT": "exact", "PYTHONPATH": OPT})                                   # the env route in B: refused by the gate, before the pin gate
    assert r.returncode == 3 and "run.sh: NOT ACTIVE (RF-6):" in r.stderr and "boltz is not installed at the pin" not in r.stderr, r.stderr
    r = _run_sh(py_b, ["check", "--mode", "exact"], {"PYTHONPATH": OPT})                                    # the --mode route in B: not this gate's (activation by construction) — the next precondition, the pin, refuses in an empty venv
    assert "RF-6" not in r.stderr and r.returncode == 3 and "boltz is not installed at the pin" in r.stderr, r.stderr
    r = _run_sh(py_b, PRED + ["--mode", "exact"], {"BOLTZ2_OPT": "exact", "PYTHONPATH": OPT})           # --mode given beside the variable: the command line names the mode — not the env route
    assert "RF-6" not in r.stderr and r.returncode == 3 and "boltz is not installed at the pin" in r.stderr, r.stderr
    r = _run_sh(py_a, PRED, {"BOLTZ2_OPT": "exact"})                                                    # the env route in A: the hook is live — the gate passes, the pin gate is next
    assert "RF-6" not in r.stderr and r.returncode == 3 and "boltz is not installed at the pin" in r.stderr, r.stderr
    r = _run_sh(py_b, PRED + ["--mode", "off"], {"PYTHONPATH": OPT})                                      # the stock route: exempt
    assert "RF-6" not in r.stderr and r.returncode == 3 and "boltz is not installed at the pin" in r.stderr, r.stderr
    r = _run_sh(py_b, PRED, {"BOLTZ2_OPT": "off", "PYTHONPATH": OPT})
    assert "RF-6" not in r.stderr and r.returncode == 3 and "boltz is not installed at the pin" in r.stderr, r.stderr


def _kit_free_python_with_boltz():
    """The interpreter the bare-interpreter form uses: boltz installed at the pin, the kit NOT installed (no .pth) — the base interpreter of a venv the
    tests run from (the box installs the kit into a venv over the system site); None where no such interpreter exists (a local run)."""
    if sys.prefix == sys.base_prefix:
        return None
    py = os.path.join(sys.base_prefix, "bin", "python3")
    if not os.path.isfile(py):
        return None
    pins = subprocess.run([py, "-I", os.path.join(TREE, "stock", "check_pins.py"), "--quiet"], capture_output=True, text=True)
    hook = subprocess.run([py, GATE], capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"})
    return py if pins.returncode == 0 and hook.returncode == 3 else None


def test_mode_route_passes_with_no_install_and_the_env_route_refuses(tmp_path):
    """(4) `PYTHONPATH=<kit> run.sh check --mode <kit mode>` with NO install → NOT refused: the package's own DRY-RUN line (activation by
    construction, the bare interpreter's form); the same interpreter's env route → refused with the diagnostic."""
    import pytest
    py = _kit_free_python_with_boltz()
    if py is None:
        pytest.skip("needs an interpreter with boltz at the pin and no kit install (the box form: the kit installed into a venv over the system site)")
    pp = OPT + os.pathsep + CORE
    r = _run_sh(py, ["check", "--config", "h100", "--mode", "exact"], {"PYTHONPATH": pp})
    assert "RF-6" not in r.stderr, r.stderr[-600:]                                   # the --mode route: never this gate's
    if r.returncode == 0:
        assert "[boltz2-opt] DRY-RUN mode=exact route=worker" in r.stdout, r.stdout[-600:]        # a GPU box: the plan renders
    else:                                                                                             # a CPU box: the plan's own GPU gate refuses by name, after the RF-6 point
        assert r.returncode == 3 and "no NVIDIA GPU (nvidia-smi found none)" in r.stdout + r.stderr, (r.returncode, r.stdout[-400:], r.stderr[-400:])
    r = _run_sh(py, ["check", "--config", "h100"], {"PYTHONPATH": pp, "BOLTZ2_OPT": "exact"})
    assert r.returncode == 3 and "run.sh: NOT ACTIVE (RF-6):" in r.stderr and f"present at {os.path.join(OPT, PTH)} beside a sys.path/PYTHONPATH entry" in r.stderr and "DRY-RUN" not in r.stdout, (r.stdout[-300:], r.stderr[-600:])


def test_user_site_install_is_live(tmp_path):
    """A `pip install --user` / PYTHONUSERBASE install of the kit: the route's interpreter (no -I) processes the user site's .pth, so the gate
    passes; the same layout with the user site disabled (-s) is refused with the user-site diagnostic."""
    import pytest
    py = os.path.join(sys.base_prefix, "bin", "python3")           # a non-venv interpreter (a venv disables the user site by construction)
    if not os.path.isfile(py):
        pytest.skip("no base interpreter at sys.base_prefix/bin/python3")
    ubase = tmp_path / "ubase"
    env = {**{k: v for k, v in os.environ.items() if k not in ("BOLTZ2_OPT", "PYTHONNOUSERSITE", "PYTHONPATH")}, "PYTHONUSERBASE": str(ubase)}
    usite = subprocess.run([py, "-c", "import site; print(site.getusersitepackages()); print(site.ENABLE_USER_SITE)"], capture_output=True, text=True, env=env, check=True).stdout.split()
    if usite[1] != "True":
        pytest.skip(f"the user site is disabled for {py}")
    sitepk = subprocess.run([py, "-s", "-c", "import site; print('\\n'.join(site.getsitepackages()))"], capture_output=True, text=True, env=env, check=True).stdout.split()
    if any(os.path.isfile(os.path.join(p, PTH)) for p in sitepk):
        pytest.skip(f"{PTH} is installed in the system site of {py}: -s cannot isolate the user-site install on this box")
    os.makedirs(usite[0], exist_ok=True)
    with open(os.path.join(usite[0], PTH), "w") as fh:
        fh.write(OPT + "\nimport boltz2_opt._autoload\n")
    live = subprocess.run([py, GATE], capture_output=True, text=True, env=env)
    assert live.returncode == 0 and live.stderr == "", live.stderr
    off = subprocess.run([py, "-s", GATE], capture_output=True, text=True, env=env)
    assert off.returncode == 3 and f"present at {os.path.join(usite[0], PTH)} (the user site) but not processed: the user site is disabled" in off.stderr, off.stderr


def test_config_probe_leaves_the_hooks_refusals_to_the_route(tmp_path):
    """configs/h100.env probes `import boltz2_opt` with BOLTZ2_OPT unset: with the hook live (venv A) a value that is not a mode is the ROUTE's to
    refuse (the config passes), an undeclared BOLTZ2_OPT* name is refused by the hook's own line through the config (exit 3) — never the
    config's 'not installed' text (exit 2), which is for an interpreter without the package (venv C)."""
    py_a, _ = _venv(str(tmp_path / "A"), with_pth=True)
    py_c, _ = _venv(str(tmp_path / "C"), with_pth=False)
    def source(py, extra):
        env = {**{k: v for k, v in os.environ.items() if k not in ("BOLTZ2_OPT", "PYTHONPATH") and not k.startswith("BOLTZ2_OPT_")}, "PATH": os.path.dirname(py) + os.pathsep + os.environ.get("PATH", ""), "MODEL_OPT": TREE}
        env.update(extra)
        return subprocess.run(["bash", "-euo", "pipefail", "-c", f'source "{TREE}/configs/h100.env" || exit $?; echo CONFIG_OK'], capture_output=True, text=True, env=env)   # run.sh's form: set -euo pipefail, source || exit
    r = source(py_a, {"BOLTZ2_OPT": "exact_k"})
    assert "CONFIG_OK" in r.stdout and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stderr[-400:])            # a value that is not a mode: the route's refusal, not the probe's
    r = subprocess.run([py_a, "-c", "import boltz2_opt"], capture_output=True, text=True, env={**{k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}, "BOLTZ2_OPT": "exact_k"})
    assert r.returncode == 3 and "[boltz2-opt] NOT ACTIVE: unknown BOLTZ2_OPT='exact_k'" in r.stderr, (r.returncode, r.stderr[-400:])   # …which the route's own import raises by name
    r = source(py_a, {"BOLTZ2_OPT_TYPO": "1"})
    assert r.returncode == 3 and "CONFIG_OK" not in r.stdout and "[boltz2-opt] NOT ACTIVE: undeclared BOLTZ2_OPT_TYPO" in r.stderr and "is not installed" not in r.stderr, (r.returncode, r.stderr[-400:])
    r = source(py_c, {})
    assert r.returncode == 2 and "[boltz2-opt] NOT ACTIVE: boltz2_opt is not installed on" in r.stderr, (r.returncode, r.stderr[-400:])


def test_run_sh_reaches_the_hooks_line_for_an_unknown_or_undeclared_env_mode(tmp_path):
    """End to end on an interpreter with boltz at the pin (the box form): `BOLTZ2_OPT=<value> run.sh check --config h100` — the config probe
    passes (the variable unset there) and the RF-6 gate passes (the hook is live in the kit venv); an undeclared BOLTZ2_OPT_TYPO name is refused
    by the hook's line at the config probe (exit 3); `BOLTZ2_OPT=bogus` is refused by the hook's own unknown-value line at the route's import
    (exit 3) — never the config's 'not installed' text."""
    import pytest
    if sys.prefix == sys.base_prefix or subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py"), "--quiet"], capture_output=True).returncode != 0:
        pytest.skip("needs the kit venv over a site with boltz at the pin (the box form)")
    if subprocess.run([sys.executable, GATE], capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}).returncode != 0:
        pytest.skip("the hook is not live in this interpreter")
    r = _run_sh(sys.executable, ["check", "--config", "h100"], {"BOLTZ2_OPT": "exact", "BOLTZ2_OPT_TYPO": "1"})
    assert r.returncode == 3 and "[boltz2-opt] NOT ACTIVE: undeclared BOLTZ2_OPT_TYPO" in r.stderr and "is not installed" not in r.stderr, (r.returncode, r.stderr[-500:])
    r = _run_sh(sys.executable, ["check", "--config", "h100"], {"BOLTZ2_OPT": "bogus"})
    assert r.returncode == 3 and "[boltz2-opt] NOT ACTIVE: unknown BOLTZ2_OPT='bogus'" in r.stderr and "is not installed" not in r.stderr, (r.returncode, r.stdout[-300:], r.stderr[-300:])   # the hook refuses the unknown value at the route's import (the core's finder, the kit's line)


def test_a13_probes_neutralise_the_mode_and_the_hook_names_its_refusal_at_interpreter_start(tmp_path):
    """With the kit's .pth INSTALLED (venv A) and BOLTZ2_OPT=<mode> exported, every interpreter start runs the hook's core pin gate: on a
    mismatched core a bare `python -c …` exits 3 — WITH the by-name NOT ACTIVE line, never silently — so run.sh / configs/*.env run their own
    import probes with the variable unset (`env -u BOLTZ2_OPT`: a probe answers importability only; the refusal is the real entry's to print).
    Locks: (1) every probe line of run.sh and configs/*.env carries `env -u BOLTZ2_OPT`; (2) in venv A with a shadow opt_core of another
    version first on PYTHONPATH: `BOLTZ2_OPT=exact python -c pass` → rc 3 + `NOT ACTIVE: reason=core_mismatch`; the neutralised package probe → rc 0."""
    for f in [RUN] + sorted(os.path.join(TREE, "configs", c) for c in os.listdir(os.path.join(TREE, "configs")) if c.endswith(".env")):
        code = [re.split(r"\s{2,}# ", l, maxsplit=1)[0] for l in open(f).read().splitlines() if not l.lstrip().startswith("#")]   # the command part, trailing comment dropped
        probes = [c for c in code if re.search(r"\bpython3? +(-I +)?(-c +|\S*(pth_gate|check_pins)\.py)", c)]
        assert probes and all(re.search(r"env -u BOLTZ2_OPT +python", c) for c in probes), (f, probes)
    py_a, _ = _venv(str(tmp_path / "A"), with_pth=True)
    shadow = tmp_path / "shadow"; (shadow / "opt_core").mkdir(parents=True)
    (shadow / "opt_core" / "__init__.py").write_text('__version__ = "0.2.5"\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith("BOLTZ2_OPT") and k != "PYTHONPATH"}
    r = subprocess.run([py_a, "-c", "pass"], env={**env, "BOLTZ2_OPT": "exact", "PYTHONPATH": str(shadow)}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and "[boltz2-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned " in r.stderr and "installed v0.2.5 at " in r.stderr, (r.returncode, r.stderr[-600:])
    probe = "import importlib.util as u, sys; sys.exit(0 if u.find_spec('boltz2_opt') else 5)"          # run.sh's package probe, as written there
    assert probe in open(RUN).read()
    r = subprocess.run(["env", "-u", "BOLTZ2_OPT", py_a, "-c", probe], env={**env, "BOLTZ2_OPT": "exact", "PYTHONPATH": str(shadow)}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (r.returncode, r.stderr[-400:])
    r = subprocess.run([py_a, "-m", "boltz2_opt", "check", "--mode", "exact"], env={**env, "BOLTZ2_OPT": "exact", "PYTHONPATH": str(shadow)}, capture_output=True, text=True, timeout=120)   # the real entry names it itself
    assert r.returncode == 3 and "NOT ACTIVE: reason=core_mismatch" in r.stderr and "Traceback" not in r.stderr, (r.returncode, r.stderr[-600:])
