"""A13 lock: the shell probes of run.sh / configs/h100.env under an EXPORTED activation variable, with the INSTALLED .pth processed at
interpreter start (a stdlib venv whose site-packages carry the generated `openfold3_ob0_opt_autoload.pth` byte for byte plus a path file for the
package and the core — the editable install's shape). The import probe must never re-diagnose the hook's own by-name refusal (exit 3 at
interpreter start: undeclared variable, the core pin gate, the producer probe) as `package_missing`: rc 3 = the hook's line
forwarded verbatim; only a genuine ImportError is `package_missing`."""
import os
import re
import shutil
import subprocess
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))                      # …/openfold3_ob0/opt
KIT = os.path.dirname(OPT)                                        # …/openfold3
PTH = os.path.join(OPT, "openfold3_ob0_opt_autoload.pth")             # the generated hook the install lays into site-packages


def _core_root():
    import opt_core
    return os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))      # <core root>, the opt_core package's parent directory


def _venv(tmp_path, core_root):
    """A stdlib venv (no pip) whose site-packages process the kit's INSTALLED .pth and see the package + `core_root`'s opt_core."""
    vdir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(str(vdir))
    py = str(vdir / "bin" / "python")
    site = subprocess.run([py, "-c", "import site, sys; print([p for p in site.getsitepackages() if p.startswith(sys.prefix)][0])"],
                          capture_output=True, text=True, check=True).stdout.strip()
    shutil.copyfile(PTH, os.path.join(site, os.path.basename(PTH)))                   # the hook, byte for byte
    with open(os.path.join(site, "_of3_editable_paths.pth"), "w") as fh:              # the editable installs' path lines
        fh.write(OPT + "\n" + core_root + "\n")
    return vdir, py


def _source_h100(py_dir, extra_env):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENFOLD3_OB0_OPT", "OF3", "PYTHONPATH", "MODEL_OPT"))}
    env.update(extra_env); env["PATH"] = str(py_dir) + os.pathsep + env.get("PATH", "")
    return subprocess.run(["bash", "-c", f"source {KIT}/configs/h100.env && echo SOURCED-OK"], capture_output=True, text=True, env=env, cwd=KIT)


def _mismatched_core(tmp_path):
    """A copy of the installed core whose __version__ is older than openfold3's pin: the pin gate's core_mismatch (floor semantics — a
    newer core passes, an older one refuses; the gate reads __version__ from opt_core/__init__.py directly, never a manifest file)."""
    src_root = _core_root()
    dst_root = tmp_path / "othercore"
    shutil.copytree(os.path.join(src_root, "opt_core"), dst_root / "opt_core", ignore=shutil.ignore_patterns("__pycache__"))
    init_py = dst_root / "opt_core" / "__init__.py"
    text, n = re.subn(r'__version__\s*=\s*"[^"]*"', '__version__ = "0.2.5"', init_py.read_text(encoding="utf-8"))
    assert n == 1, "opt_core/__init__.py __version__ literal not found to substitute"
    init_py.write_text(text, encoding="utf-8")
    return str(dst_root)


def test_probe_is_quiet_under_an_exported_mode_with_the_pinned_core(tmp_path):
    vdir, py = _venv(tmp_path, _core_root())
    r = _source_h100(vdir / "bin", {"OPENFOLD3_OB0_OPT": "fast"})
    assert r.returncode == 0 and "SOURCED-OK" in r.stdout and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stderr[-800:], r.stdout[-200:])


def test_probe_forwards_the_hooks_refusal_by_name_under_an_exported_mode(tmp_path):
    """OPENFOLD3_OB0_OPT=fast exported + a core that is not the pinned one: the .pth hook's pin gate refuses at interpreter start (exit 3); the
    probe forwards THAT line (`core_mismatch`), never `package_missing`."""
    vdir, py = _venv(tmp_path, _mismatched_core(tmp_path))
    r = _source_h100(vdir / "bin", {"OPENFOLD3_OB0_OPT": "fast"})
    assert r.returncode == 3, (r.returncode, r.stderr[-800:])
    assert "[openfold3_ob0-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned" in r.stderr and "package_missing" not in r.stderr and "SOURCED-OK" not in r.stdout, r.stderr[-800:]
    assert r.stderr.count("NOT ACTIVE") == 1, r.stderr[-800:]                          # one line: the hook's, forwarded once


def test_probe_forwards_an_undeclared_variable_refusal(tmp_path):
    vdir, py = _venv(tmp_path, _core_root())
    r = _source_h100(vdir / "bin", {"OPENFOLD3_OB0_OPTX": "1"})                             # a mistyped switch: the hook refuses it by name at start
    assert r.returncode == 3 and "NOT ACTIVE: undeclared variable(s) OPENFOLD3_OB0_OPTX" in r.stderr and "package_missing" not in r.stderr, (r.returncode, r.stderr[-600:])


def test_probe_names_a_genuinely_missing_package(tmp_path):
    vdir = tmp_path / "bare"; venv.EnvBuilder(with_pip=False, symlinks=True).create(str(vdir))        # no path file, no hook: the package is not importable here
    r = _source_h100(vdir / "bin", {"OPENFOLD3_OB0_OPT": "fast"})
    assert r.returncode == 3 and "NOT ACTIVE: reason=package_missing: openfold3_ob0_opt is not importable" in r.stderr and "ModuleNotFoundError" in r.stderr, (r.returncode, r.stderr[-600:])


def test_run_sh_and_every_config_env_carry_the_same_probe():
    """One probe text in every entry script (run.sh exits, a sourced configs/<card>.env returns; $HERE / $MODEL_OPT name the tree), and the
    per-card configs differ from h100.env only in their header prose and the MODEL_OPT_TARGET_GPU word (deployment parameters, no lever switch);
    `run.sh --config <card>` sources configs/<card>.env and refuses an unknown card by name."""
    a = [l for l in open(os.path.join(KIT, "run.sh")).read().splitlines() if l.startswith("_of3_err=$(python -c")]
    envs = sorted(f for f in os.listdir(os.path.join(KIT, "configs")) if f.endswith(".env"))
    assert envs == ["a100.env", "b200.env", "b300.env", "h100.env", "h200.env"], envs
    norm = lambda l: l.split("   #")[0].replace("$MODEL_OPT", "$HERE").replace("return 3 2>/dev/null || exit 3", "exit 3").replace('return "$_of3_rc" 2>/dev/null || exit "$_of3_rc"', 'exit "$_of3_rc"')   # noqa: E731
    body, card = {}, {"a100.env": "A100", "b200.env": "B200", "b300.env": "B300", "h100.env": "H100", "h200.env": "H200"}
    for f in envs:
        ls = open(os.path.join(KIT, "configs", f)).read().splitlines()
        b = [l for l in ls if l.startswith("_of3_err=$(python -c")]
        assert len(a) == 1 and len(b) == 1 and a[0].split("||")[0] == b[0].split("||")[0], f
        assert norm(a[0]) == norm(b[0]), f
        assert [l.split("   #")[0].strip() for l in ls if l.startswith("export MODEL_OPT_TARGET_GPU=")] == [f"export MODEL_OPT_TARGET_GPU=${{MODEL_OPT_TARGET_GPU:-{card[f]}}}"], f
        body[f] = [l for l in ls if not l.startswith("#") and not l.startswith("export MODEL_OPT_TARGET_GPU=")]
    for f in envs:
        assert body[f] == body["h100.env"], f                                                       # same statements; only the card words differ
    rs = open(os.path.join(KIT, "run.sh")).read()
    assert '[ -f "$HERE/configs/$CFG.env" ] || { echo "run.sh: no such config: $CFG (see $HERE/configs/)" >&2; exit 2; }' in rs and 'source "$HERE/configs/$CFG.env" || exit $?' in rs
    r = subprocess.run(["bash", os.path.join(KIT, "run.sh"), "check", "--config", "no_such_card"], capture_output=True, text=True)
    assert r.returncode == 2 and "no such config: no_such_card" in r.stderr, (r.returncode, r.stderr[-300:])
    assert "ModuleNotFoundError: No module named 'openfold3_ob0_opt'" in a[0] and 'exit "$_of3_rc"' in a[0]      # package_missing only for the package's own ModuleNotFoundError; everything else leaves with the interpreter's rc


def _shadow_venv(tmp_path):
    """A venv whose path file resolves `openfold3_ob0_opt` to a package that raises on import (a broken install), the hook installed as usual."""
    shadow = tmp_path / "shadow"; (shadow / "openfold3_ob0_opt").mkdir(parents=True)
    (shadow / "openfold3_ob0_opt" / "__init__.py").write_text('raise RuntimeError("simulated broken install")\n')
    vdir, py = _venv(tmp_path, _core_root())
    site = subprocess.run([py, "-c", "import site, sys; print([p for p in site.getsitepackages() if p.startswith(sys.prefix)][0])"], capture_output=True, text=True, check=True).stdout.strip()
    with open(os.path.join(site, "_of3_editable_paths.pth"), "w") as fh:               # the shadow ahead of nothing else: `openfold3_ob0_opt` is the broken package
        fh.write(str(shadow) + "\n" + _core_root() + "\n")
    return vdir


def test_probe_forwards_an_import_error_verbatim_with_its_own_rc(tmp_path):
    """An import of openfold3_ob0_opt that fails for any reason other than the package being absent leaves the probe with the interpreter's own
    exit code and its stderr verbatim — never `package_missing`, never rc 3 by the script's choice."""
    vdir = _shadow_venv(tmp_path)
    r = _source_h100(vdir / "bin", {})
    assert r.returncode == 1 and "RuntimeError: simulated broken install" in r.stderr and "package_missing" not in r.stderr and "SOURCED-OK" not in r.stdout, (r.returncode, r.stderr[-600:])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENFOLD3_OB0_OPT", "OF3", "PYTHONPATH", "MODEL_OPT"))}
    env["PATH"] = str(vdir / "bin") + os.pathsep + env.get("PATH", "")
    r = subprocess.run(["bash", os.path.join(KIT, "run.sh"), "check", "--config", "h100", "--mode", "fast"], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 1 and "RuntimeError: simulated broken install" in r.stderr and "package_missing" not in r.stderr and "not installed" not in r.stderr, (r.returncode, r.stderr[-600:])
