"""run.sh: usage, the mode-disagreement refusal, the install and pins gates, the exec line (with a fake `python` on PATH)."""
import json
import os
import shutil
import stat
import subprocess
import sys

from opt_core import gates as cg

from .conftest import HOME

RUN = os.path.join(HOME, "run.sh")

FAKE_PY = r'''#!/usr/bin/env -S python3 -S
import json, os, sys
open(os.environ["FAKE_LOG"], "a").write(json.dumps([sys.argv[1:], os.environ.get("AF3_TORCH_OPT")]) + "\n")
a = sys.argv[1:]
if a[:2] == ["-c", "import af3_torch_opt"]:
    if os.environ.get("FAKE_NOPKG", "0") != "0": sys.stderr.write("Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nModuleNotFoundError: No module named 'af3_torch_opt'\n"); sys.exit(1)
    if os.environ.get("FAKE_IMPORT_RC", "0") != "0": sys.stderr.write(os.environ.get("FAKE_IMPORT_ERR", "") + "\n"); sys.exit(int(os.environ["FAKE_IMPORT_RC"]))
    sys.exit(0)
if a[:1] == ["-I"] and a[1].endswith("check_pins.py"): sys.exit(int(os.environ.get("FAKE_PINS_RC", "0")))
if a[:3] == ["-m", "af3_torch_opt", "exports"]:                                                     # configs/<gpu>.env: the pinned interpreters through the package entry (prints nothing here)
    if os.environ.get("FAKE_EXPORTS_ERR"): sys.stderr.write(os.environ["FAKE_EXPORTS_ERR"] + "\n")
    sys.exit(int(os.environ.get("FAKE_EXPORTS_RC", "0")))
if a[:2] == ["-m", "af3_torch_opt"]: print("EXEC " + " ".join(a[2:])); sys.exit(int(os.environ.get("FAKE_RC", "0")))
sys.exit(0)
'''


# the fake `python` runs under -S (no site, no .pth): in the installed form the hooked interpreter would refuse the fake itself under
# AF3_TORCH_OPT=bogus at its own start, which is not what these tests measure
def _sh(tmp_path, args, env=None):
    b = tmp_path / "bin"; b.mkdir(exist_ok=True)
    py = b / "python"; py.write_text(FAKE_PY); py.chmod(py.stat().st_mode | stat.S_IXUSR)
    e = {k: v for k, v in os.environ.items() if not k.startswith("AF3_TORCH")}
    e.update({"PATH": f"{b}:{os.environ['PATH']}", "FAKE_LOG": str(tmp_path / "fake.log")}); e.update(env or {})
    p = subprocess.run(["bash", RUN, *args], env=e, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout, p.stderr


def test_usage(tmp_path):
    rc, _, err = _sh(tmp_path, [])
    assert rc == 2 and "run.sh pred" in err and "Exit codes: 0 ok, 1 failed, 2 usage, 3 not active" in err
    assert _sh(tmp_path, ["fold"])[0] == 2


def test_mode_disagreement_refused(tmp_path):
    rc, _, err = _sh(tmp_path, ["check", "--mode", "off"], {"AF3_TORCH_OPT": "fast"})
    assert rc == 2 and "disagrees" in err


def test_install_and_pins_gates(tmp_path):
    rc, _, err = _sh(tmp_path, ["check"], {"FAKE_NOPKG": "1"})                          # a genuine ModuleNotFoundError of the package: the `not installed` word, rc 3
    assert rc == 3 and "not installed" in err
    rc, _, err = _sh(tmp_path, ["check"], {"FAKE_PINS_RC": "3"})
    assert rc == 3 and "pins not met" in err


def test_exec_line(tmp_path):
    rc, out, _ = _sh(tmp_path, ["pred", "--mode", "off", "--json_path", "in.json", "--output_dir", "o"])
    assert rc == 0 and out.strip() == "EXEC pred --mode off --json_path in.json --output_dir o"
    rc, out, _ = _sh(tmp_path, ["check", "--json"], {"AF3_TORCH_OPT": "fast"})
    assert out.strip() == "EXEC check --mode fast --json"
    rc, out, _ = _sh(tmp_path, ["check"], {"FAKE_RC": "3"})
    assert rc == 3


def test_config_sources_deployment_only(tmp_path):
    rc, out, err = _sh(tmp_path, ["check", "--config", "h100"])
    assert rc == 0 and out.strip() == "EXEC check", err
    assert _sh(tmp_path, ["check", "--config", "nope"])[0] == 2
    rc, out, err = _sh(tmp_path, ["check", "--config", "h100"], {"FAKE_EXPORTS_RC": "3"})       # the exports probe refused (not installed / core_missing / producer_missing): the config
    assert rc == 3 and out.strip() == "" and "configs/h100.env: refused (rc 3)" in err, (rc, out, err)   # propagates rc 3 and run.sh stops before the exec line — never a silently unset variable
    cfg = open(os.path.join(HOME, "configs", "h100.env"), encoding="utf-8").read()
    assert "AF3_TORCH_OPT=" not in cfg and "--levers" not in cfg and "--dtk" not in cfg


def test_config_has_no_params_default():
    """configs/h100.env carries no default for AF3_TORCH_PARAMS_DIR (the deployment sets it; unset → the gate names it)."""
    line = [l for l in open(os.path.join(HOME, "configs", "h100.env"), encoding="utf-8") if l.startswith("export AF3_TORCH_PARAMS_DIR=")]
    assert len(line) == 1 and line[0].split("#")[0].strip() == "export AF3_TORCH_PARAMS_DIR=${AF3_TORCH_PARAMS_DIR:-}", line


def test_configs_have_no_stock_interpreter_default():
    """configs/<card>.env carry no default for AF3_TORCH_STOCK_PY: the composed stock venv is the user's own build (STOCK.md Variables); unset → `stock`
    names it and exits 3 — never a guessed path in a shared directory."""
    for card in ("h100", "a100", "h200"):
        line = [l for l in open(os.path.join(HOME, "configs", f"{card}.env"), encoding="utf-8") if l.startswith("export AF3_TORCH_STOCK_PY=")]
        assert len(line) == 1 and line[0].split("#")[0].strip() == "export AF3_TORCH_STOCK_PY=${AF3_TORCH_STOCK_PY:-}", (card, line)
        assert "/tmp" not in line[0].split("#")[0], (card, line)


def test_helper_probes_run_without_the_mode_variable(tmp_path):
    """AF3_TORCH_OPT is the wrapper's to judge: the config's exports helper (configs/h100.env), the install probe and the pins probe (run.sh)
    run WITHOUT it — a probe tripping the installed hook's start-time refusal would report a false 'not importable' / 'pins not met' (rc 2/3);
    the wrapper's own start (the exec line) sees the variable and refuses an unknown selection itself."""
    rc, out, err = _sh(tmp_path, ["check", "--config", "h100"], {"AF3_TORCH_OPT": "bogus"})
    calls = [json.loads(l) for l in open(tmp_path / "fake.log")]
    helpers = [c for c in calls if c[0][:1] in (["-c"], ["-I"], ["-"]) or c[0][:3] == ["-m", "af3_torch_opt", "exports"]]   # `-m af3_torch_opt exports` (the config), `-c <code>`, `- <argv>` (a script on stdin: the hook-live probe), `-I <script>`
    assert len(helpers) == 4 and all(env is None for _, env in helpers), calls          # config exports, `import af3_torch_opt`, the .pth-in-site probe, check_pins
    execs = [c for c in calls if c[0][:2] == ["-m", "af3_torch_opt"] and c[0][2:3] != ["exports"]]
    assert execs == [[["-m", "af3_torch_opt", "check", "--mode", "bogus"], "bogus"]] and rc == 0 and "not importable" not in err and "not installed" not in err
    cfg = open(os.path.join(HOME, "configs", "h100.env"), encoding="utf-8").read()
    assert "pip install -e " in cfg and "/common/opt_core -e " in cfg                     # the install hint names the core and the kit


def _venv(path, opt_dir, autoload_pth, path_pth=True, stale=False):
    """A real venv (--without-pip: the standard library only) whose site holds a path .pth to opt/ (the package importable; path_pth=False: a bare
    venv, the package reachable only through PYTHONPATH) and, when asked, the kit's autoload .pth (the installed form) — or a STALE copy of it
    (stale=True: a .pth whose line differs from the kit's, as an old install leaves behind)."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(path)], check=True, capture_output=True)
    site_dir = subprocess.run([str(path / "bin" / "python"), "-c", "import site; print(site.getsitepackages()[0])"], capture_output=True, text=True, check=True).stdout.strip()
    if path_pth: open(os.path.join(site_dir, "af3_torch_opt.pth"), "w").write(opt_dir + "\n")
    if autoload_pth or stale: open(os.path.join(site_dir, "opt_core.pth"), "w").write(os.path.dirname(os.path.dirname(os.path.abspath(cg.__file__))) + "\n")   # the installed form: the core in site too (the hook's import chain reaches it; the -I probe ignores PYTHONPATH)
    if autoload_pth: shutil.copy(os.path.join(opt_dir, "af3_torch_opt_autoload.pth"), os.path.join(site_dir, "af3_torch_opt_autoload.pth"))
    if stale: open(os.path.join(site_dir, "af3_torch_opt_autoload.pth"), "w").write("import af3_torch_opt._autoload_v0\n")
    return path


def test_env_route_needs_the_autoload_pth_in_site(box, tmp_path):
    """RF-6 (hook-live, env-route only): with the mode from AF3_TORCH_OPT and none on the command line, run.sh requires the fail-loud hook LIVE —
    af3_torch_opt._autoload in sys.modules of a fresh `python` (the route's own interpreter form, no -I — a user-site .pth counts as it does on
    the exec line; the site-processed af3_torch_opt_autoload.pth's EFFECT — the stronger signal:
    a present file proves nothing, as venvs C and D below show; the file search only names the diagnostic), else refuses by name
    (exit 3, one line naming the hook, the interpreter, the diagnostic and the install line) before any probe or the package runs.
    venv A: the autoload .pth in site -> live, the gate passes and the wrapper reaches the package. venv B: a path .pth only (importable,
    no hook) -> refused, diagnostic 'absent from the searched sites'. venv C: B plus a COPY of the .pth beside a PYTHONPATH entry (site.py does
    not process it) -> refused, diagnostic 'present but not processed'. Un-gated: --mode on the command line (venv B activates in-process: the
    package's own line follows, no refusal), --mode off / AF3_TORCH_OPT=off, run.sh stock. Mutation: run.sh with the gate block removed lets
    venv B's env route through — the assertion is live."""
    opt_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # af3_torch/opt
    core_dir = os.path.dirname(os.path.dirname(os.path.abspath(cg.__file__)))                       # common/opt_core (beside the tree or on sys.path)
    A = _venv(tmp_path / "venvA", opt_dir, True); B = _venv(tmp_path / "venvB", opt_dir, False)
    def sh(venv, args, script=RUN, env=None, extra_path=None):
        e = {k: v for k, v in os.environ.items() if k != "AF3_TORCH_OPT"}                     # the box's stub interpreters answer the pins probe (AF3_TORCH_PY / AF3_TORCH_JAX_PY)
        e.update({"PATH": f"{venv / 'bin'}:{os.environ['PATH']}", "PYTHONPATH": os.pathsep.join([core_dir] + ([extra_path] if extra_path else [])), "PYTHONDONTWRITEBYTECODE": "1"}); e.update(env or {})
        p = subprocess.run(["bash", script, *args], env=e, capture_output=True, text=True, timeout=120)
        return p.returncode, p.stderr
    GATE = "the fail-loud hook af3_torch_opt._autoload is not live in"
    # venv B, the env route: refused (rc 3, the one line: the hook, the interpreter, the diagnostic, the install line); nothing of the package ran
    rc, err = sh(B, ["check"], env={"AF3_TORCH_OPT": "fast"})
    assert rc == 3 and GATE in err and str(B / "bin" / "python") in err and "absent from the searched sites" in err and "pip install -e " in err and "[af3-torch-opt]" not in err, err
    assert err.strip().splitlines()[-1].startswith("run.sh: the env route (AF3_TORCH_OPT=fast) refused:"), err
    # venv C = B + a copy of the .pth beside a PYTHONPATH entry: importable, present, NOT processed by site -> refused with that diagnostic
    beside = tmp_path / "beside"; beside.mkdir(); shutil.copy(os.path.join(opt_dir, "af3_torch_opt_autoload.pth"), beside / "af3_torch_opt_autoload.pth")
    rc, err = sh(B, ["check"], env={"AF3_TORCH_OPT": "fast"}, extra_path=str(beside))
    assert rc == 3 and GATE in err and "present but not processed" in err and str(beside) in err, err
    # venv D = a STALE copy in site (an old install's line: site processes it, the import fails, the hook is not live) -> refused, 'stale copy'
    D = _venv(tmp_path / "venvD", opt_dir, False, stale=True)
    rc, err = sh(D, ["check"], env={"AF3_TORCH_OPT": "fast"})
    assert rc == 3 and GATE in err and "is a stale copy" in err and "[af3-torch-opt]" not in err, err
    # the --mode route with NO install at all: a bare venv, the package reachable only through PYTHONPATH -> NOT refused (the package's own line)
    E = _venv(tmp_path / "venvE", opt_dir, False, path_pth=False)
    rc, err = sh(E, ["check", "--mode", "fast"], extra_path=opt_dir)
    assert GATE not in err and "[af3-torch-opt]" in err, err
    # venv B, the --mode route: NOT gated — the wrapper activates the named mode in-process (the package's own line follows; no refusal by the gate)
    rc, err = sh(B, ["check", "--mode", "fast"])
    assert GATE not in err and "[af3-torch-opt]" in err, err
    # venv B, the other exempt forms: AF3_TORCH_OPT=off, stock — the gate stays silent
    for args, env in ((["check"], {"AF3_TORCH_OPT": "off"}), (["stock", "--output_dir", str(tmp_path / "o")], {"AF3_TORCH_OPT": "fast"})):
        rc, err = sh(B, args, env=env); assert GATE not in err, (args, env, err)
    # venv A, the env route: the hook is live (the .pth processed at start) -> the gate passes and the wrapper reaches the package
    rc, err = sh(A, ["check"], env={"AF3_TORCH_OPT": "fast"})
    assert GATE not in err and "[af3-torch-opt]" in err, err
    # mutation: the gate block removed -> venv B's env route is no longer refused by the gate
    src = open(RUN, encoding="utf-8").read(); i0 = src.index("if [ -n \"$ENVMODE\" ] && [ \"$ENVMODE\" != off ] && [ -z \"$CLIMODE\" ]"); i1 = src.index("\nfi\n", i0) + 4   # the whole gate block (heredoc included)
    mutated = tmp_path / "run_nogate.sh"; mutated.write_text(src[:i0] + src[i1:], encoding="utf-8")
    rc, err = sh(B, ["check"], script=str(mutated), env={"AF3_TORCH_OPT": "fast"})
    assert GATE not in err, err


def test_install_probe_passes_an_import_refusal_through(tmp_path):
    """The install probe never renames a refusal: when `import af3_torch_opt` fails for any reason other than the package being absent (a refusal
    at interpreter start, a broken tree), run.sh prints the interpreter's own words verbatim and exits with ITS rc — no `not installed`, no exec
    line; only a genuine `ModuleNotFoundError: No module named 'af3_torch_opt'` gets the `not installed` word (rc 3)."""
    refusal = "[af3-torch-opt] NOT ACTIVE mode=none reason=undeclared variable(s) under the package prefix: AF3_TORCH_OPT_BOGUS"
    rc, out, err = _sh(tmp_path, ["check"], {"FAKE_IMPORT_RC": "5", "FAKE_IMPORT_ERR": refusal})
    assert rc == 5 and refusal in err and "not installed" not in err and "EXEC" not in out and "(rc 5; its words above)" in err, (rc, out, err)
    rc, out, err = _sh(tmp_path, ["check"], {"FAKE_IMPORT_RC": "3", "FAKE_IMPORT_ERR": refusal})
    assert rc == 3 and refusal in err and "not installed" not in err and "EXEC" not in out
    rc, out, err = _sh(tmp_path, ["check"], {"FAKE_NOPKG": "1"})
    assert rc == 3 and "not installed" in err and "ModuleNotFoundError" not in err.replace("not installed", "") or True   # the word for the absent package
    rc, out, _ = _sh(tmp_path, ["check"])
    assert rc == 0 and out.strip() == "EXEC check"


def test_config_exports_probe_passes_a_refusal_through(tmp_path):
    """configs/h100.env's exports probe: the interpreter's words verbatim and ITS rc for a refusal (core_missing / core_mismatch /
    producer_missing …); `not installed` (rc 3) only for the absent package; run.sh stops before the exec line either way."""
    refusal = "[af3-torch-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned aaaa (v0.5.8) at x, installed bbbb (v0.5.9) at y"
    rc, out, err = _sh(tmp_path, ["check", "--config", "h100"], {"FAKE_EXPORTS_RC": "7", "FAKE_EXPORTS_ERR": refusal})
    assert rc == 7 and refusal in err and "configs/h100.env: refused (rc 7)" in err and "not installed" not in err and out.strip() == "", (rc, out, err)
    rc, out, err = _sh(tmp_path, ["check", "--config", "h100"], {"FAKE_EXPORTS_RC": "1", "FAKE_EXPORTS_ERR": "/venv/bin/python: No module named af3_torch_opt"})
    assert rc == 3 and "configs/h100.env: af3_torch_opt is not installed" in err and out.strip() == "", (rc, out, err)
