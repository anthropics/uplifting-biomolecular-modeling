"""run.sh on the stub box, driven as a caller would (bash, `python` = venv A of this interpreter, the package and its start-up hook in site, a fake nvidia-smi):
the refusal rows and exit codes of the wrapper, the pins gate with its diagnostic, the pass-through of --mode/--variant to the package."""
import os
import shutil
import subprocess
import sys

from .conftest import CHILD_PYTHONPATH, CORE, OPT, TREE, make_venv, python_on_path
from .conftest import PKG


def test_usage_rows(box):
    rc, _, err = box.run_sh("bogus")
    assert rc == 2 and "run.sh pred" in err                                                                   # no such verb: the usage text
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "turbo")
    assert rc == 2 and "unknown mode 'turbo'" in err and "run.sh:" not in err                               # the package's refusal, not the wrapper's
    rc, _, err = box.run_sh("check", "--variant", "p3", "--mode", "off")
    assert rc == 2 and "invalid choice: 'p3'" in err                                                         # argparse's choices = variants.VARIANTS
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "off", env={"AF3_JAX_OPT": "exact"})
    assert rc == 2 and "disagrees with AF3_JAX_OPT=exact" in err
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "off", env={"AF3_JAX_VARIANT": "p3"})
    assert rc == 2 and "disagrees with AF3_JAX_VARIANT=p3" in err
    rc, _, err = box.run_sh("check", "--config", "nosuch", "--variant", "p2", config=False)
    assert rc == 2 and "no such config: nosuch" in err
    rc, _, err = box.run_sh("install", "--variant", "p3")
    assert rc == 2 and "invalid choice: 'p3'" in err                                                         # the parameters step's refusal (the box names a parameters root)


def test_import_probe_passes_the_interpreters_refusal_through(box, tmp_path):
    """run.sh's (and the config's) import probe names 'not installed' ONLY for the package's own ModuleNotFoundError; any other failure of
    `import af3_jax_opt` — a refusal at start-up, a broken dependency — exits with the interpreter's own code and words, never masked."""
    fake = tmp_path / "fakesite" / "af3_jax_opt"; fake.mkdir(parents=True)
    (fake / "__init__.py").write_text('import sys\nsys.stderr.write("[af3-jax-opt] NOT ACTIVE mode=? reason=simulated refusal at import\\n")\nraise SystemExit(3)\n')
    env = {"PYTHONPATH": str(tmp_path / "fakesite")}                                                          # first on the path: `import af3_jax_opt` is this refusing package
    for config, tag in ((False, "run.sh: importing af3_jax_opt failed"), (True, "NOT ACTIVE: importing af3_jax_opt failed")):
        rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "off", env=env, config=config)
        assert rc == 3 and "reason=simulated refusal at import" in err and tag in err and "is not installed" not in err, (config, rc, err)
    (fake / "__init__.py").write_text('raise ModuleNotFoundError("No module named \'somedep\'", name="somedep")\n')   # a broken dependency: not the package absent
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "off", env=env, config=False)
    assert rc == 1 and "No module named 'somedep'" in err and "is not installed" not in err, (rc, err)


def test_pins_gate_prints_its_diagnostic(box, tmp_path):
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "off", config=False)                     # no config: the pinned XLA variables are unset in this shell — drift, NAMED, the verb runs
    assert rc == 0 and "[af3-jax-opt] PINS drift XLA_FLAGS=unset (pinned '--xla_gpu_enable_triton_gemm=false') — named, not refused" in err and "PINS NOT MET" not in err
    assert "run.sh: pins not met" not in err and "[af3-jax-opt] ACTIVE mode=off" in err
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "off", env={"AF3_JAX_PY": os.path.join(box.root, "nope")})
    assert rc == 3 and "interpreter=interpreter not found" in err and "run.sh: pins not met" in err
    bare = str(tmp_path / "bin_bare"); os.makedirs(bare); python_on_path(bare, make_venv(str(tmp_path / "venv_bare"), hook=False, package=False))
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "off", env={"PYTHONPATH": "/nonexistent", "PATH": bare + os.pathsep + box.bin + os.pathsep + os.environ.get("PATH", "")})
    assert rc == 2 and "af3_jax_opt is not installed" in err                                                 # a bare interpreter: the config refuses (rc 2) before the wrapper's own check (rc 3)
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "off", env={"PYTHONPATH": "/nonexistent", "PATH": bare + os.pathsep + box.bin + os.pathsep + os.environ.get("PATH", "")}, config=False)
    assert rc == 3 and "af3_jax_opt is not installed" in err                                                 # the wrapper's own line


def test_routes_pass_through_to_the_package(box):
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "off")
    assert rc == 0 and "[af3-jax-opt] ACTIVE mode=off variant=p2 script=run_alphafold.py" in err and "PINS" not in err   # --quiet: no pins line when met
    rc, _, err = box.run_sh("check", "--variant", "p2")
    assert rc == 0 and "ACTIVE mode=fast" in err and "partial=cold_cache" in err                             # --mode omitted = the package default (fast); its class is cold: named, active
    rc, _, err = box.run_sh("warm", "--variant", "p2")
    assert rc == 0 and "WARM PASS mode=fast variant=p2 key=" in err and "__fast" in err                     # --mode omitted = the package default (fast)
    rc, _, err = box.run_sh("check", "--variant", "p2")
    assert rc == 0 and "ACTIVE mode=fast variant=p2 script=run_alphafold_fast.py launcher=fpf_launch.py" in err
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "exact")
    assert rc == 0 and "ACTIVE mode=exact" in err and "partial=cold_cache" in err                            # fast's class does not warm exact's: exact's is cold, named
    rc, _, err = box.run_sh("warm", "--variant", "p2", "--mode", "exact")
    assert rc == 0 and "WARM PASS mode=exact variant=p2 key=" in err and "__fast" not in err
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "exact")
    assert rc == 0 and "ACTIVE mode=exact variant=p2 script=run_alphafold_fast.py launcher=levers_launch.py" in err and "partial=" not in err
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "exact", "--num_recycles=3")               # the stock flags pass through run.sh and the check verb too
    assert rc == 0 and "partial=" not in err                                                                # the class is warm: nothing partial
    inp = box.input_json("r", seeds=(1,))
    out = os.path.join(box.root, "out_sh")
    rc, _, err = box.run_sh("pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out)
    assert rc == 0 and "STOCK" in err and "DONE status=ok rc=0 predictions=5/5" in err
    rec = box.stub_record()
    assert rec["script"] == "run_alphafold.py" and rec["launcher"] is None and not [k for k in rec["env"] if k.startswith(("AF3_JAX_", "AF3P_", "AF3_FLASHPAIRFORMER"))]
    assert rec["env"]["XLA_FLAGS"] == "--xla_gpu_enable_triton_gemm=false"
    rc, _, err = box.run_sh("pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out, "--featurisation_workers=3")
    assert rc == 3 and "exact_only_flags=['--featurisation_workers=3'] proof=FAILED" in err, err        # a kit-script flag on the stock route: named on the STOCK line, the route is not the stock one (exit 3); stock's own flags are never refused
    rc, _, err = box.run_sh("pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out, env={"STUB_FAIL": "1"})
    assert rc == 1 and "DONE status=FAILED rc=7" in err
    rc, _, err = box.run_sh("pred", "--variant", "p2", "--mode", "off", "--output_dir", out)
    assert rc == 2 and "exactly one of --json_path / --input_dir" in err


def test_config_derives_the_image_values_from_pins(box):
    """configs/h100.env states no image path or XLA value of its own: they come from stock/PINS.json through the package."""
    import json
    import subprocess
    cfg = open(os.path.join(TREE, "configs", "h100.env"), encoding="utf-8").read()
    pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
    for v in (pins["image"]["repo_dir"], pins["image"]["python"], *pins["image"]["env"].values()):
        assert v not in cfg, v
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AF3_JAX_", "XLA_", "MODEL_OPT"))}
    env["PATH"] = box.bin + os.pathsep + env["PATH"]; env["PYTHONPATH"] = CHILD_PYTHONPATH
    p = subprocess.run(["bash", "-c", f"source {TREE}/configs/h100.env && env"], env=env, capture_output=True, text=True)
    got = dict(ln.split("=", 1) for ln in p.stdout.splitlines() if "=" in ln)
    assert p.returncode == 0, p.stderr
    assert got["AF3_JAX_REPO"] == pins["image"]["repo_dir"] and got["AF3_JAX_PY"] == pins["image"]["python"]
    assert {k: got.get(k) for k in pins["image"]["env"]} == pins["image"]["env"]
    env["XLA_FLAGS"] = "--custom"; env["AF3_JAX_PY"] = box.py
    p = subprocess.run(["bash", "-c", f"source {TREE}/configs/h100.env && env"], env=env, capture_output=True, text=True)
    got = dict(ln.split("=", 1) for ln in p.stdout.splitlines() if "=" in ln)
    assert got["XLA_FLAGS"] == "--custom" and got["AF3_JAX_PY"] == box.py                                    # pre-set values are kept


def test_a100_env_is_h100_env_for_compute_capability_80(box):
    """configs/a100.env exports what configs/h100.env exports (the pinned stack's values through the package); only MODEL_OPT_TARGET_GPU differs
    (A100), and its statements are h100.env's with the words a100/A100 for h100/H100."""
    import subprocess
    assert sorted(os.listdir(os.path.join(TREE, "configs"))) == ["a100.env", "h100.env", "h200.env"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AF3_JAX_", "XLA_", "MODEL_OPT"))}
    env["PATH"] = box.bin + os.pathsep + env["PATH"]; env["PYTHONPATH"] = CHILD_PYTHONPATH
    got = {}
    for cfg in ("h100", "a100"):
        p = subprocess.run(["bash", "-c", f"source {TREE}/configs/{cfg}.env && env"], env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        got[cfg] = {k: v for k, v in (ln.split("=", 1) for ln in p.stdout.splitlines() if "=" in ln) if k.startswith(("AF3_JAX_", "XLA_", "MODEL_OPT", "PYTHONDONT", "JAX_", "TF_"))}
    assert (got["h100"].pop("MODEL_OPT_TARGET_GPU"), got["a100"].pop("MODEL_OPT_TARGET_GPU")) == ("H100", "A100")
    assert got["h100"] == got["a100"]
    code = lambda cfg: [ln.split("#")[0].rstrip() for ln in open(os.path.join(TREE, "configs", cfg + ".env"), encoding="utf-8") if not ln.startswith("#")]
    assert [l.replace("a100", "h100").replace("A100", "H100") for l in code("a100")] == code("h100")


def test_hook_gate_two_venvs(box, tmp_path):
    """The env-route hook gate, hook-LIVE and env-route-only. The env route = AF3_JAX_OPT naming a kit mode with no --mode: there the
    start-up hook must be live — `af3_jax_opt._autoload` in sys.modules of a fresh `python -I` with the variable unset (the site-processed
    .pth's effect) — because importable alone, that variable with the fork's script would run stock silently. Venv A (the box's `python`:
    the package and af3_jax_opt_autoload.pth in its site) passes; venv B (no hook in site) is refused by name — exit 3, the one run.sh line
    naming the diagnostic, the package never runs; the diagnostic is three-way — absent from the searched sites / present but not processed
    (a copy beside a PYTHONPATH entry; a site copy whose import failed; the user site disabled) / a stale copy (another import line in site); a
    `pip install --user` form (PYTHONUSERBASE) is live when the user site is enabled — the probe is the route's own interpreter form, no -I. The --mode route is not
    gated (the wrapper activates by construction: the package's own line — on venv B, and with no install at all, PYTHONPATH=<kit> as the
    caller with no install runs it); --mode off and AF3_JAX_OPT=off are exempt. Mutation: with the gate removed the venv-B env-route call prints the
    package's line, not run.sh's."""
    rc, _, err = box.run_sh("check", "--variant", "p2", env={"AF3_JAX_OPT": "exact"})
    assert "not live" not in err and "[af3-jax-opt]" in err                                                     # venv A, the env route: the gate passes, the package answers (cold class: its own rc 3)
    py_b = make_venv(str(tmp_path / "venv_b"), hook=False)                                                     # the package in site (editable form), no hook .pth
    bin_b = tmp_path / "bin_b"; bin_b.mkdir(); python_on_path(str(bin_b), py_b)
    env_b = {"PATH": str(bin_b) + os.pathsep + box.bin + os.pathsep + os.environ.get("PATH", ""), "PYTHONPATH": "", "AF3_JAX_OPT": "exact"}
    for cmd in (("check", "--variant", "p2"), ("pred", "--variant", "p2", "--json_path", "x.json", "--output_dir", "y")):
        rc, out, err = box.run_sh(*cmd, env=env_b)
        lines = [l for l in err.splitlines() if l.strip()]
        assert rc == 3 and len(lines) == 1 and lines[0].startswith("run.sh: the env route (AF3_JAX_OPT=exact from configs/h100.env, no --mode) needs the start-up hook af3_jax_opt._autoload live in ") and "[af3-jax-opt]" not in err, (cmd, err)
        assert "af3_jax_opt_autoload.pth absent from the searched sites: " in lines[0] and "pip install -e" in lines[0] and "run stock silently" in lines[0]
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "exact", env=dict(env_b, AF3_JAX_OPT=""))
    assert "not live" not in err and "[af3-jax-opt]" in err                                                     # venv B, the --mode route: not gated — the package's own line
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "exact", env=env_b)
    assert "not live" not in err and "[af3-jax-opt]" in err                                                     # --mode given beside the agreeing variable: the --mode route
    rc, _, err = box.run_sh("check", "--variant", "p2", env=dict(env_b, AF3_JAX_OPT="off"))
    assert rc == 0 and "[af3-jax-opt] ACTIVE mode=off" in err and "not live" not in err                        # the stock route under the variable: exempt
    py_bare = make_venv(str(tmp_path / "venv_bare"), hook=False, package=False)                                # importable through PYTHONPATH alone: the .pth beside the package proves nothing
    bin_bare = tmp_path / "bin_bare"; bin_bare.mkdir(); python_on_path(str(bin_bare), py_bare)
    env_bare = dict(env_b, PATH=str(bin_bare) + os.pathsep + box.bin + os.pathsep + os.environ.get("PATH", ""), PYTHONPATH=CHILD_PYTHONPATH)
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "exact", env=dict(env_bare, AF3_JAX_OPT=""))
    assert "not live" not in err and "[af3-jax-opt]" in err                                                     # NO install, PYTHONPATH=<kit>, --mode: the no-install form, not refused
    rc, _, err = box.run_sh("check", "--variant", "p2", env=env_bare)
    lines = [l for l in err.splitlines() if l.strip()]
    assert rc == 3 and len(lines) == 1 and "af3_jax_opt_autoload.pth present but not processed: on PYTHONPATH at " + OPT in lines[0] and "[af3-jax-opt]" not in err
    py_stale = make_venv(str(tmp_path / "venv_stale"), hook=False, package=False, hook_text="import af3_jax_opt_v0._autoload\n")   # a stale copy in site
    bin_stale = tmp_path / "bin_stale"; bin_stale.mkdir(); python_on_path(str(bin_stale), py_stale)
    rc, _, err = box.run_sh("check", "--variant", "p2", env=dict(env_bare, PATH=str(bin_stale) + os.pathsep + box.bin + os.pathsep + os.environ.get("PATH", "")))
    assert rc == 3 and "af3_jax_opt_autoload.pth a stale copy: " in err and "imports 'af3_jax_opt_v0._autoload', the kit ships 'af3_jax_opt._autoload'" in err and "[af3-jax-opt]" not in err
    py_hook = make_venv(str(tmp_path / "venv_hook"), hook=True, package=False)                                # the kit's .pth in site, the package on PYTHONPATH: the route's python imports it the same way — live
    bin_hook = tmp_path / "bin_hook"; bin_hook.mkdir(); python_on_path(str(bin_hook), py_hook)
    env_hook = dict(env_bare, PATH=str(bin_hook) + os.pathsep + box.bin + os.pathsep + os.environ.get("PATH", ""))
    rc, _, err = box.run_sh("check", "--variant", "p2", env=env_hook)
    assert "not live" not in err and "[af3-jax-opt]" in err
    user_base = tmp_path / "userbase"; user_site = user_base / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"   # `pip install --user` / PYTHONUSERBASE: the user site
    user_site.mkdir(parents=True)
    with open(user_site / "__editable__af3_jax_opt.pth", "w") as f:
        f.write(OPT + "\n" + CORE + "\n")
    shutil.copyfile(os.path.join(OPT, "af3_jax_opt_autoload.pth"), str(user_site / "af3_jax_opt_autoload.pth"))
    env_user = dict(env_bare, PYTHONPATH="/nonexistent", PYTHONUSERBASE=str(user_base))
    rc, _, err = box.run_sh("check", "--variant", "p2", env=env_user)
    assert "not live" not in err and "[af3-jax-opt]" in err                                                     # the user-site install: processed by the route's python — passes the gate
    rc, _, err = box.run_sh("check", "--variant", "p2", env=dict(env_bare, PYTHONUSERBASE=str(user_base), PYTHONNOUSERSITE="1"))
    lines = [l for l in err.splitlines() if l.strip()]
    assert rc == 3 and len(lines) == 1 and "af3_jax_opt_autoload.pth present but not processed: in " + str(user_site) in lines[0] and "the user site is disabled" in lines[0] and "[af3-jax-opt]" not in err
    beside = tmp_path / "beside"; beside.mkdir(); shutil.copyfile(os.path.join(OPT, "af3_jax_opt_autoload.pth"), str(beside / "af3_jax_opt_autoload.pth"))
    rc, _, err = box.run_sh("check", "--variant", "p2", env=dict(env_bare, PYTHONPATH=str(beside) + os.pathsep + CHILD_PYTHONPATH))
    lines = [l for l in err.splitlines() if l.strip()]
    assert rc == 3 and len(lines) == 1 and "af3_jax_opt_autoload.pth present but not processed: on PYTHONPATH at " + str(beside) in lines[0] and "[af3-jax-opt]" not in err


# --- merged from test_levers_launch.py (file consolidation, tests unchanged) ---

LAUNCHER = os.path.join(PKG, "levers_launch.py")
FAKE_LEVERS = '''
import os
LEVERS = {}
def apply_levers():
    if os.environ.get("AF3P_GLU_T") == "1": LEVERS["L-GLUT"] = "glut"
    if os.environ.get("AF3P_ATTN_CFG"): LEVERS["L-ATTNCFG"] = "cfg"
    return LEVERS
'''
TARGET = '''
import sys, json
print("TARGET argv=" + json.dumps(sys.argv) + " name=" + __name__)
'''


def _launch(tmp_path, env, *args):
    """Runs the launcher the way the fork's own interpreter would: a plain venv, script directory on sys.path[0] by Python's ordinary
    default — PYTHONSAFEPATH (which some hosts' Python sets to suppress that) is stripped so the launcher's sibling import
    (fpf_launch, beside it) resolves the same way it does under the fork's interpreter, regardless of this one's own defaults."""
    levers = tmp_path / "levers"; levers.mkdir(exist_ok=True)
    (levers / "af3_pallas_levers.py").write_text(FAKE_LEVERS)
    script = tmp_path / "run_alphafold_fast.py"; script.write_text(TARGET)
    e = {k: v for k, v in os.environ.items() if not k.startswith("AF3P_") and k != "PYTHONSAFEPATH"}; e.update(env)
    p = subprocess.run([sys.executable, LAUNCHER, str(levers), str(script), *args], capture_output=True, text=True, env=e, timeout=60)
    out = p.stdout.splitlines(keepends=True)
    ck = [l for l in out if l.startswith("[af3-jax-opt] CACHEKEY accelerator=")]   # the persistent-cache key rebinding (inprocess/portable_cache_key.py), right after the LEVERS
    if p.returncode == 0:                                                    # line; this interpreter has no jax, so the line says accelerator=absent — asserted here once,
        assert len(ck) == 1 and ck[0].startswith("[af3-jax-opt] CACHEKEY accelerator=absent reason=jax_not_importable") and out.index(ck[0]) == 1, out[:3]
    out = [l for l in out if l not in ck]                                    # the callers read the launcher's other lines in their order
    return p.returncode, "".join(out), p.stderr


def test_levers_then_script(tmp_path):
    rc, out, err = _launch(tmp_path, {"AF3P_GLU_T": "1", "AF3P_ATTN_CFG": "64,64,4,3"}, "--json_path=/i.json", "--output_dir=/o")
    assert rc == 0, err
    lines = out.splitlines()
    assert lines[0].startswith("[af3-jax-opt] LEVERS active=L-ATTNCFG+L-GLUT af3_pallas_levers.py=") and "script=run_alphafold_fast.py sha256=" in lines[0]
    assert lines[1].startswith("[af3-jax-opt] TEMPLATES guard=failed:ModuleNotFoundError:")            # the template census guard, installed before the script runs: named when the fork is not importable (here), never silent
    assert lines[2] == f'TARGET argv=["{tmp_path}/run_alphafold_fast.py", "--json_path=/i.json", "--output_dir=/o"] name=__main__'


def test_no_lever_set_is_named(tmp_path):
    rc, out, _ = _launch(tmp_path, {})
    assert rc == 0 and out.splitlines()[0].startswith("[af3-jax-opt] LEVERS active=none")


def test_usage(tmp_path):
    p = subprocess.run([sys.executable, LAUNCHER], capture_output=True, text=True, timeout=60)
    assert p.returncode != 0 and "usage: levers_launch.py <levers dir> <script.py>" in p.stderr
    p = subprocess.run([sys.executable, LAUNCHER, str(tmp_path), str(tmp_path / "missing.py")], capture_output=True, text=True, timeout=60)
    assert p.returncode != 0 and "no such file" in p.stderr


def test_launcher_is_stdlib_only():
    src = open(LAUNCHER, encoding="utf-8").read()
    for word in ("af3_jax_opt", "import jax", "import numpy"):
        assert word not in src.replace("never imports af3_jax_opt", ""), word


def test_config_carries_no_box_path(box, monkeypatch):
    """configs/h100.env fills the pinned stack's values and never a box path: the parameters root and the cache root are the caller's —
    unset in, unset out; pre-set, passed through untouched. With AF3_JAX_PARAMS_ROOT unset every verb then refuses BY NAME (unless the caller's
    --model_dir names the weights); with AF3_JAX_CACHE_ROOT unset the cache root is the package default, never a mount path."""
    from af3_jax_opt import stack
    e = dict(os.environ); e["PATH"] = box.bin + os.pathsep + e.get("PATH", ""); e["PYTHONPATH"] = CHILD_PYTHONPATH
    for k in ("AF3_JAX_PARAMS_ROOT", "AF3_JAX_CACHE_ROOT", "MODEL_OPT"):
        e.pop(k, None)
    probe = 'source "$1" && printf "%s|%s|%s|%s" "${AF3_JAX_PARAMS_ROOT-<unset>}" "${AF3_JAX_CACHE_ROOT-<unset>}" "$AF3_JAX_REPO" "$AF3_JAX_PY"'
    p = subprocess.run(["bash", "-c", probe, "probe", os.path.join(TREE, "configs", "h100.env")], env=e, capture_output=True, text=True, timeout=120, cwd=box.root)
    assert p.returncode == 0 and p.stdout == f"<unset>|<unset>|{box.repo}|{box.py}", (p.returncode, p.stdout, p.stderr)   # nothing filled; the pre-set checkout/interpreter kept
    monkeypatch.delenv("AF3_JAX_CACHE_ROOT")
    assert stack.cache_root() == stack.DEFAULT_CACHE_ROOT and not stack.DEFAULT_CACHE_ROOT.startswith(("/jitcache", "/vol"))
    monkeypatch.setenv("AF3_JAX_CACHE_ROOT", box.cache_root)
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "off", unset=("AF3_JAX_PARAMS_ROOT",))
    assert rc == 3 and "NOT ACTIVE" in err and "AF3_JAX_PARAMS_ROOT is not set" in err and "--model_dir" in err, (rc, err)
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "exact", unset=("AF3_JAX_PARAMS_ROOT",))
    assert rc == 3 and "AF3_JAX_PARAMS_ROOT is not set" in err, (rc, err)
    rc, _, err = box.run_sh("check", "--variant", "p2", "--mode", "off", "--model_dir", os.path.join(box.params_root, "p2"), unset=("AF3_JAX_PARAMS_ROOT",))
    assert rc == 0 and "ACTIVE mode=off" in err, (rc, err)                                                   # the caller's --model_dir: no root needed
