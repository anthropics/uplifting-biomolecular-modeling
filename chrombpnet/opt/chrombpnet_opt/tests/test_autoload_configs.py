"""The environment route (the .pth finder), the deterministic recipe's composition against the kit's own hook, the configs, run.sh's
refusal paths, and the carried kit tree's route-critical files."""
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

from chrombpnet_opt import det
from . import _stubs

ARGS = ["-cm", "m.h5", "-r", "r.bed", "-g", "g.fa", "-c", "c.sizes", "-op", "out/p"]
# a program that (1) imports the .pth's module, as site does at interpreter start, (2) injects a child runner that records the launch (the
# parent's loaded modules, the child's argv and environment) and then runs the child for real, (3) sets argv to the stock console script's
# and imports the stock package the way the console script does (chrombpnet, then chrombpnet.CHROMBPNET); on the fast route the finder's
# action exits this process with the run's exit code (cli.run_fast) before MODULES/FINDER/main
DRIVER = """
import json, os, subprocess, sys
sys.argv = json.loads(os.environ["DRIVER_ARGV"])
import chrombpnet_opt._autoload as a
def rec(argv, env=None, **kw):
    heavy = sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "tensorflow", "triton", "bpnetlite", "_stub_probe_marker"))
    print("LAUNCH " + json.dumps({"path": argv[0], "argv": argv, "heavy_modules": heavy, "env": {k: v for k, v in env.items() if k.startswith(("CHROMBPNET", "PYTHONPATH", "TF_", "CUBLAS"))}}), flush=True)
    return subprocess.run(argv, env=env, **kw)
a.RUN = rec
import chrombpnet
from chrombpnet.CHROMBPNET import main
print("MODULES " + json.dumps(sorted(m for m in sys.modules if m.startswith("chrombpnet_opt"))), flush=True)
print("FINDER " + json.dumps({"installed": a.FINDER is not None, "fired": getattr(a.FINDER, "fired", None)}), flush=True)
main()
"""


def run_driver(kit, site, argv, extra=None, cwd=None):
    env = _stubs.base_env(kit, extra)
    env["PYTHONPATH"] = _stubs.OPT_DIR + os.pathsep + site
    env["DRIVER_ARGV"] = json.dumps(argv)
    return subprocess.run([sys.executable, "-c", DRIVER], env=env, cwd=cwd or tempfile.mkdtemp(prefix="cbp_opt_test_"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)   # never the tree


def launch(p):
    """The LAUNCH record the injected runner printed (the child's argv and environment, the parent's loaded modules at the launch)."""
    return json.loads([l for l in p.stdout.splitlines() if l.startswith("LAUNCH ")][0][len("LAUNCH "):])


def test_finder_runs_documented_line_and_owns_the_exit(tmp_path):
    """The environment route: the kit's documented line runs as a child of the console-script process, which then judges the kit's run record,
    writes the manifest beside the outputs (the record folded in) and exits with the run's exit code (cli.run_fast, the path `chrombpnet-opt pred_bw` takes)."""
    from chrombpnet_opt import manifest
    kit = _stubs.make_tree(str(tmp_path / "tree")); site = _stubs.make_stock_package(str(tmp_path / "site"))
    cfg = tmp_path / "nv_compute_cache_H100.tar"; cfg.write_text("staged\n")
    cwd = tmp_path / "cwd"; cwd.mkdir()
    p = run_driver(kit, site, ["/usr/local/bin/chrombpnet", "pred_bw"] + ARGS, {"CHROMBPNET_OPT": "fast", "STUB_GPU_CLASS": "H100", "CHROMBPNET_OPT_CACHE_TAR": str(cfg), "HOME_X": "1"}, cwd=str(cwd))
    assert p.returncode == 0, p.stdout + p.stderr
    lines = p.stdout.splitlines()
    assert lines[0] == "[chrombpnet-opt] ACTIVE mode=fast route=k1 kit=0.0.0-stub gpu=H100 det=0 precision=tf32"
    ex = launch(p)
    assert ex["path"] == sys.executable and ex["argv"] == [sys.executable, os.path.join(_stubs.fast_kit(kit), "tf", "pred_bw_fast.py")] + ARGS   # the fast mode's kit: opt/kit_ho
    assert not [k for k in ex["env"] if k.startswith("CHROMBPNET_OPT")] and [k for k in ex["env"] if k.startswith(("CHROMBPNET_FASTKIT_", "K1_"))] == ["CHROMBPNET_FASTKIT_RECORD"]
    assert ex["env"]["CHROMBPNET_JIT_CACHE_TAR"] == str(cfg)                    # the driver-cache deviation on the environment route too
    assert not ex["env"]["CHROMBPNET_FASTKIT_RECORD"].startswith(str(cwd))       # the run-record handshake the package composes: a temporary file, never beside the outputs
    assert ex["heavy_modules"] == []                                              # no torch/tensorflow/triton, and no kit probe, in the wrapper before the launch
    assert "[stub kit] ran" in lines and "[stub stock main] ran" not in p.stdout and "MODULES " not in p.stdout   # the child ran; the stock's main never did; the exit is the finder's
    man = manifest.read(str(cwd / "out"))
    assert man["partial"] == [] and man["levers_applied"] == ["forward_route", "jit_cache"] and man["exit_code"] == 0 and man["job_rc"] == 0 and "allow_partial" not in man
    assert man["argv"] == ["/usr/local/bin/chrombpnet", "pred_bw"] + ARGS and man["arm_argv"] == ex["argv"]
    assert "[chrombpnet-opt] EXIT " in p.stderr and "partial=none gated=" in p.stderr


def test_finder_partial_exit_rule(tmp_path):
    """The environment route owns the exit rule: a partial activation by the kit's run record -> the family line and exit 3;
    a failed job -> its own rc."""
    from chrombpnet_opt import manifest
    kit = _stubs.make_tree(str(tmp_path / "tree")); site = _stubs.make_stock_package(str(tmp_path / "site"))
    partial = {"CHROMBPNET_OPT": "fast", "STUB_GPU_CLASS": "H100", "STUB_STAMP_FORWARD": "tf_function", "STUB_STAMP_SKIPPED": "K1 route: OFF — the K1 stack failed to import (stub); TF route"}
    cwd = tmp_path / "c1"; cwd.mkdir()
    p = run_driver(kit, site, ["chrombpnet", "pred_bw"] + ARGS, partial, cwd=str(cwd))
    assert p.returncode == 3, p.stdout + p.stderr
    assert p.stdout.splitlines()[-1] == ("[chrombpnet-opt] NOT ACTIVE: partial activation — levers=forward_route: forward_route: p: the kit ran forward=tf_function, "
                                         "the tables' route is k1: K1 route: OFF — the K1 stack failed to import (stub); TF route; exit 3 (--mode off runs stock)")
    man = manifest.read(str(cwd / "out"))
    assert man["partial"] == ["forward_route"] and "allow_partial" not in man and man["exit_code"] == 3 and man["job_rc"] == 0 and "[stub stock main] ran" not in p.stdout
    cwd = tmp_path / "c3"; cwd.mkdir()
    p = run_driver(kit, site, ["chrombpnet", "pred_bw"] + ARGS, dict(partial, STUB_RC="5"), cwd=str(cwd))
    assert p.returncode == 5 and manifest.read(str(cwd / "out"))["job_rc"] == 5
    cwd = tmp_path / "c4"; cwd.mkdir()                                              # the kit refused by name: NOT ACTIVE, 3, nothing ran
    p = run_driver(kit, site, ["chrombpnet", "pred_bw"] + ARGS, {"CHROMBPNET_OPT": "fast", "STUB_GPU_CLASS": "H100", "STUB_REFUSED": "the K1 kernels did not start on this card (stub)"}, cwd=str(cwd))
    assert p.returncode == 3 and p.stdout.splitlines()[-1].startswith("[chrombpnet-opt] NOT ACTIVE mode=fast reason=the kit refused by name: the K1 kernels did not start on this card (stub)"), p.stdout + p.stderr
    assert manifest.read(str(cwd / "out"))["refused"] == "the K1 kernels did not start on this card (stub)" and "[stub stock main] ran" not in p.stdout


def test_wrapper_route_is_the_table_not_the_probe(tmp_path):
    """Activation takes the route from the kit's table functions (no probe); `check` runs the kit's full resolve() with the probe."""
    kit = _stubs.make_tree(str(tmp_path / "tree"))
    code = ("import sys, json\nfrom chrombpnet_opt import stack\n"
            "r = stack.activate('fast', route_label='cli', quiet=True)\nprobe_after_activate = '_stub_probe_marker' in sys.modules\n"
            "c = stack.activate('fast', dry_run=True, quiet=True)\nprobe_after_check = '_stub_probe_marker' in sys.modules\n"
            "print(json.dumps([r['route'], r['route_source'], r['kit_integrity']['scope'], probe_after_activate, c['route'], c['route_source'], c['kit_integrity']['scope'], probe_after_check,"
            " sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'tensorflow', 'triton'))]))\n")
    p = subprocess.run([sys.executable, "-c", code], env=_stubs.base_env(kit, {"STUB_GPU_CLASS": "H100"}), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.returncode == 0, p.stderr
    r_route, r_src, r_scope, probe1, c_route, c_src, c_scope, probe2, heavy = json.loads(p.stdout.strip().splitlines()[-1])
    assert r_route == "k1" and r_src == "class_route" and r_scope == "route" and probe1 is False
    assert c_route == "k1" and c_src.startswith("resolve") and c_scope == "full" and probe2 is True and heavy == []


def test_kit_integrity_scopes(tmp_path):
    from chrombpnet_opt import stack
    kit = _stubs.fast_kit(_stubs.make_tree(str(tmp_path / "tree"), with_cache_for=("H100",)))   # the fast kit (opt/kit_ho), its torch side carried at opt/kit beside it
    full = stack.kit_integrity(kit, scope="full"); route = stack.kit_integrity(kit, scope="route")
    assert full["scope"] == "full" and full["missing"] == [] and full["n_present"] > 0
    assert route["scope"] == "route" and route["missing"] == [] and 0 < route["n_present"] < full["n_present"]
    os.remove(os.path.join(kit, "tf", "det_subprocess", "sitecustomize.py"))       # a route member: both scopes require it
    assert "kit_ho/tf/det_subprocess/sitecustomize.py" in stack.kit_integrity(kit, scope="route")["missing"]
    assert "kit_ho/tf/det_subprocess/sitecustomize.py" in stack.kit_integrity(kit, scope="full")["missing"]
    with pytest.raises(ValueError):
        stack.kit_integrity(kit, scope="all")


def test_finder_det_env_route(tmp_path):
    kit = _stubs.make_tree(str(tmp_path / "tree")); site = _stubs.make_stock_package(str(tmp_path / "site"))
    p = run_driver(kit, site, ["chrombpnet", "pred_bw"] + ARGS, {"CHROMBPNET_OPT": "exact", "STUB_GPU_CLASS": "H100"})   # the mode carries the recipe: no CHROMBPNET_OPT_DET needed
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.splitlines()[0] == "[chrombpnet-opt] ACTIVE mode=exact route=k1 kit=0.0.0-stub gpu=H100 det=1 precision=fp32"
    env = launch(p)["env"]
    assert env["PYTHONPATH"].split(os.pathsep)[0] == os.path.join(_stubs.fast_kit(kit), "tf", "det_subprocess") and env["CHROMBPNET_DET_SUBPROCESS"] == "1" and env["TF_DETERMINISTIC_OPS"] == "1"


def test_finder_other_subcommand_stock_proceeds(tmp_path):
    kit = _stubs.make_tree(str(tmp_path / "tree")); site = _stubs.make_stock_package(str(tmp_path / "site"))
    p = run_driver(kit, site, ["/usr/local/bin/chrombpnet", "contribs_bw", "-m", "x"], {"CHROMBPNET_OPT": "fast", "STUB_GPU_CLASS": "H100"})
    assert p.returncode == 0, p.stderr
    lines = p.stdout.splitlines()
    assert lines[0] == "[chrombpnet-opt] NOT ACTIVE mode=fast reason=subcommand contribs_bw is not covered by the kit (stock runs)"
    assert "LAUNCH" not in p.stdout and lines[-1] == "[stub stock main] ran"
    assert json.loads([l for l in lines if l.startswith("FINDER ")][0][7:]) == {"installed": True, "fired": "chrombpnet"}
    p = run_driver(kit, site, ["/x/somescript.py", "pred_bw"] + ARGS, {"CHROMBPNET_OPT": "fast"})
    assert p.returncode == 0 and p.stdout.splitlines()[0] == "[chrombpnet-opt] NOT ACTIVE mode=fast reason=entry somescript.py is not the stock console script chrombpnet (stock runs)"


def test_finder_unset_imports_nothing(tmp_path):
    kit = _stubs.make_tree(str(tmp_path / "tree")); site = _stubs.make_stock_package(str(tmp_path / "site"))
    for extra in ({}, {"CHROMBPNET_OPT": "off"}):
        p = run_driver(kit, site, ["/usr/local/bin/chrombpnet", "pred_bw"] + ARGS, extra)
        assert p.returncode == 0, p.stderr
        assert not p.stdout.startswith("[chrombpnet-opt]")
        assert json.loads([l for l in p.stdout.splitlines() if l.startswith("MODULES ")][0][8:]) == ["chrombpnet_opt", "chrombpnet_opt._autoload"]
        assert json.loads([l for l in p.stdout.splitlines() if l.startswith("FINDER ")][0][7:]) == {"installed": False, "fired": None}
        assert p.stdout.splitlines()[-1] == "[stub stock main] ran"


def test_finder_refusals_exit_3(tmp_path):
    kit = _stubs.make_tree(str(tmp_path / "tree")); site = _stubs.make_stock_package(str(tmp_path / "site"))
    p = run_driver(str(tmp_path / "nokit" / "opt" / "kit"), site, ["chrombpnet", "pred_bw"] + ARGS, {"CHROMBPNET_OPT": "fast"})
    assert p.returncode == 3 and p.stdout.startswith("[chrombpnet-opt] NOT ACTIVE mode=fast reason=kit not found")
    for name in ("CHROMBPNET_FASTKIT_TAIL", "K1_PREIMPORT", "CHROMBPNET_JIT_CACHE_TAR", "CHROMBPNET_DET_SEED"):     # a kit-internal name set: removed for the run and named on the env route too, never refused
        cwd = tmp_path / ("ign_" + name); cwd.mkdir()
        p = run_driver(kit, site, ["chrombpnet", "pred_bw"] + ARGS, {"CHROMBPNET_OPT": "fast", "STUB_GPU_CLASS": "H100", name: "1"}, cwd=str(cwd))
        assert p.returncode == 0 and p.stdout.splitlines()[0].startswith("[chrombpnet-opt] ACTIVE mode=fast route=k1 "), (name, p.stdout + p.stderr)
        assert "[chrombpnet-opt] IGNORED names={} reason=kit-internal names set in the environment are removed for the run (the mode is the whole composition)".format(name) in p.stderr.splitlines(), (name, p.stderr)
        assert name not in launch(p)["env"], name                                  # the kit's line never saw it
    p = run_driver(kit, site, ["chrombpnet", "pred_bw"] + ARGS, {"CHROMBPNET_OPT": "exact", "CHROMBPNET_OPT_DET": "1", "CHROMBPNET_DET_SEED": "0", "STUB_GPU_CLASS": "H100"})   # exact: the recipe's names are the package's own (CHROMBPNET_OPT_DET=1 is redundant there, accepted)
    assert p.returncode == 0 and p.stdout.splitlines()[0].startswith("[chrombpnet-opt] ACTIVE mode=exact ") and p.stdout.splitlines()[0].endswith("det=1 precision=fp32"), p.stdout + p.stderr
    p = run_driver(kit, site, ["chrombpnet", "pred_bw"] + ARGS, {"CHROMBPNET_OPT": "fast", "CHROMBPNET_OPT_DET": "1"})   # `fast --det 1` does not exist: refused by name on the env route
    assert p.returncode == 3 and p.stdout.startswith("[chrombpnet-opt] NOT ACTIVE mode=fast reason=CHROMBPNET_OPT_DET=1 with mode fast"), p.stdout + p.stderr
    shaped = {"CHROMBPNET_OPT": "fast", "CHROMBPNET_KIT_HOME": kit, "KIT_HOME_VAR": "CHROMBPNET_KIT_HOME", "KIT_PYTHONPATH": "opt", "GPU_COUNT": "1", "ZOO_MANIFEST": "/x", "K1_BOOKKEEPING": "1"}
    p = run_driver(kit, site, ["chrombpnet", "pred_bw"] + ARGS, shaped)                   # a caller's bookkeeping sharing the kit's prefixes: not a switch, fast stays ACTIVE
    assert p.returncode == 0 and p.stdout.splitlines()[0].startswith("[chrombpnet-opt] ACTIVE mode=fast"), p.stdout + p.stderr
    for value in ("turbo", "EXACT2"):                                               # an unknown value: refused at the trigger, exit 3, the stock never runs
        p = run_driver(kit, site, ["chrombpnet", "pred_bw"] + ARGS, {"CHROMBPNET_OPT": value})
        assert p.returncode == 3 and p.stdout.splitlines() == ["[chrombpnet-opt] NOT ACTIVE mode={} reason=unknown CHROMBPNET_OPT={!r} (expected off|exact|fast)".format(value.lower(), value.lower())], p.stdout
        assert "[stub stock main] ran" not in p.stdout
    p = run_driver(kit, site, ["chrombpnet", "contribs_bw"], {"CHROMBPNET_OPT": "turbo"})                              # any subcommand: the value is wrong, exit 3
    assert p.returncode == 3 and "[stub stock main] ran" not in p.stdout


def test_pth_and_pyproject():
    opt = _stubs.OPT_DIR
    import importlib.util
    spec = importlib.util.spec_from_file_location("_kit_build_backend", os.path.join(opt, "_build_backend.py")); backend = importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
    text = open(os.path.join(opt, "chrombpnet_opt_autoload.pth")).read()
    assert text == backend.pth_text("chrombpnet_opt", "CHROMBPNET_OPT", "chrombpnet-opt", 3)          # the generated guard form (python _build_backend.py chrombpnet_opt CHROMBPNET_OPT chrombpnet-opt): the build refuses any other text
    assert backend.pth_fields(text) == ("chrombpnet_opt", "CHROMBPNET_OPT", "chrombpnet-opt", 3) and "import chrombpnet_opt._autoload" in text
    assert len(glob.glob(os.path.join(opt, "*_autoload.pth"))) == 1
    py = open(os.path.join(opt, "pyproject.toml")).read()
    assert 'name = "chrombpnet_opt"' in py and 'chrombpnet-opt = "chrombpnet_opt.__main__:main"' in py and 'build-backend = "_build_backend"' in py
    assert 'requires-python = ">=3.8"' in py and 'requires = ["setuptools==75.3.2"]' in py


def test_det_composition(tmp_path):
    kit = _stubs.make_tree(str(tmp_path / "tree"))
    env = det.env(_stubs.fast_kit(kit), {"PYTHONPATH": "/a" + os.pathsep + os.path.join(_stubs.fast_kit(kit), "tf", "det_subprocess"), "HOME": "/h"}, route="k1")
    assert env["PYTHONPATH"] == os.path.join(_stubs.fast_kit(kit), "tf", "det_subprocess") + os.pathsep + "/a"     # prepended once, never duplicated
    assert env["CHROMBPNET_DET_SUBPROCESS"] == "1" and env["CHROMBPNET_DET_SEED"] == "0" and env["HOME"] == "/h"
    assert all(env[k] == v for k, v in det.TF_ENV.items()) and env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "CUBLAS_WORKSPACE_CONFIG" not in det.env(_stubs.fast_kit(kit), {}, route="tf_function")
    assert det.requested({"CHROMBPNET_OPT_DET": "1"}) and not det.requested({"CHROMBPNET_OPT_DET": "0"}) and not det.requested({})
    with pytest.raises(FileNotFoundError):
        det.env(str(tmp_path / "nokit"), {})
    d = det.describe(env, True, route="k1")
    assert d["on"] and d["seed_hook"] == "tf/det_subprocess/sitecustomize.py" and set(d["env"]) >= {"PYTHONPATH", "CHROMBPNET_DET_SEED", "TF_DETERMINISTIC_OPS"}


def test_det_names_match_the_kit_hook():
    """The lock: the two names the package sets are the two the kit's sitecustomize.py reads, and the hook applies the seed itself."""
    kit = _stubs.real_kit("kit_ho")
    if not kit:
        pytest.skip("the kit is not in this tree (opt/kit_ho; carried separately)")
    reads = det.kit_hook_reads(kit)
    assert reads["gate"] == det.ENV_SUBPROCESS and reads["seed"] == det.ENV_SEED
    assert reads["applies_tf_seed"] and reads["enables_op_determinism"] and reads["tf32_off"]


def test_real_kit_resolves_on_this_box():
    """With the real kit: `check --mode fast` resolves through the kit's own fastdefault (no GPU here or not: the kit's word)."""
    kit = _stubs.real_kit()
    if not kit:
        pytest.skip("the kit is not in this tree (opt/kit; carried separately)")
    p = subprocess.run([sys.executable, "-m", "chrombpnet_opt", "check", "--mode", "fast", "--json"], env=_stubs.base_env(kit), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.returncode == 0, p.stderr
    first, rest = p.stdout.split("\n", 1)
    m = re.match(r"\[chrombpnet-opt\] ACTIVE mode=fast route=(\S+) kit=(\S+) gpu=(\S+) det=0 precision=tf32$", first)
    assert m and m.group(1) in _stubs.ROUTES, first
    rep = json.loads(rest)
    assert rep["kit_version"] == m.group(2) and rep["kit_integrity"]["missing"] == [] and rep["documented_form"]


def test_kit_tree_route_files_present():
    """Every route-critical carried kit file is present (byte identity is the checked-out git commit's job, not this package's)."""
    kit = _stubs.real_kit("kit_ho")
    if not kit or not _stubs.real_kit():
        pytest.skip("the kit is not in this tree (opt/kit_ho with opt/kit beside it; carried separately)")
    from chrombpnet_opt import stack
    integ = stack.kit_integrity(kit)
    assert integ["missing"] == [] and integ["n_present"] > 0 and stack.kit_tables_root(kit) == _stubs.real_kit()
    assert os.path.isfile(os.path.join(kit, "tf", "det_subprocess", "sitecustomize.py"))
    assert not os.path.isdir(os.path.join(_stubs.TREE_DIR, "opt", "kit", "cache")) or True     # the driver-cache tarball is not a tree member; its presence is not an error


def _config_exports(path):
    out = {}
    for line in open(path):
        m = re.match(r"export ([A-Z_]+)=\$\{\1:-(.*?)\}\s*(#.*)?$", line.rstrip("\n"))
        if m:
            out[m.group(1)] = m.group(2)
    return out


@pytest.mark.parametrize("name", ["a100", "h100", "h200"])
def test_configs(name):
    assert sorted(os.listdir(os.path.join(_stubs.TREE_DIR, "configs"))) == ["a100.env", "h100.env", "h200.env"]      # the supported classes, nothing else
    path = os.path.join(_stubs.TREE_DIR, "configs", name + ".env")
    ex = _config_exports(path)
    assert ex["MODEL_OPT_TARGET_GPU"] == name.upper() and "CHROMBPNET_OPT" not in ex                # a config never sets the mode
    assert "CHROMBPNET_OPT_WEIGHTS" not in ex and "CHROMBPNET_OPT_DATA" not in ex                    # the two roots: required, no default (README Variables)
    assert ex["CHROMBPNET_OPT_CACHE_TAR"] == "$CHROMBPNET_OPT_WEIGHTS/cache/nv_compute_cache_{}.tar".format(name.upper())
    assert ex["CHROMBPNET_OPT_MODEL"] == "$CHROMBPNET_OPT_WEIGHTS/GM12878_ATAC/fold_0/chrombpnet_recompiled.h5"
    for k in ("CHROMBPNET_OPT_GENOME", "CHROMBPNET_OPT_CHROM_SIZES", "CHROMBPNET_OPT_REGIONS", "CHROMBPNET_OPT_BIGWIG"):
        assert ex[k].startswith("$CHROMBPNET_OPT_DATA/")
    text = open(path).read()
    for forbidden in ("CHROMBPNET_FASTKIT_", "K1_", "CUDA_CACHE_PATH", "CUDA_CACHE_MAXSIZE", "TF_CPP_MIN_LOG_LEVEL", "PYTHONUNBUFFERED", "export CHROMBPNET_OPT=",
                      ":-/", "/weights/", "/inputs/"):
        assert forbidden not in text, forbidden                                   # no lever switch, no stock env block, no baked absolute path in a config
    import shutil
    bash = shutil.which("bash")
    roots = {"CHROMBPNET_OPT_WEIGHTS": "/w/chrombpnet", "CHROMBPNET_OPT_DATA": "/d/chrombpnet"}
    def source(env_extra, path_env=None):
        env = {k: v for k, v in os.environ.items() if k not in roots}; env.update(env_extra)
        if path_env is not None: env["PATH"] = path_env
        return subprocess.run([bash, "-c", "source {} ; echo rc=$? tar=$CHROMBPNET_OPT_CACHE_TAR genome=$CHROMBPNET_OPT_GENOME".format(path)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # a root unset: rc 2 and the NOT ACTIVE line naming the variable (never a default path)
    p = source({}); assert "rc=2" in p.stdout and "NOT ACTIVE" in p.stderr and "reason=CHROMBPNET_OPT_WEIGHTS is not set — see README Variables" in p.stderr, (p.stdout, p.stderr)
    p = source({"CHROMBPNET_OPT_WEIGHTS": "/w/chrombpnet"}); assert "rc=2" in p.stdout and "reason=CHROMBPNET_OPT_DATA is not set — see README Variables" in p.stderr, (p.stdout, p.stderr)
    # both roots set: the derived variables follow them
    p = source(roots); assert "tar=/w/chrombpnet/cache/nv_compute_cache_{}.tar genome=/d/chrombpnet/hg38.genome.fa".format(name.upper()) in p.stdout, (p.stdout, p.stderr)
    # sourcing the config in a shell whose python cannot import the package refuses with rc 2 and the install line
    p = source(roots, path_env="/nonexistent")
    assert "rc=2" in p.stdout and "pip install -e" in p.stderr


def _run_sh(args, env_extra=None, path_dirs=(), tree=None):
    env = dict(os.environ); env.update(env_extra or {})
    env["PATH"] = os.pathsep.join(list(path_dirs) + [env.get("PATH", "")])
    cwd = tempfile.mkdtemp(prefix="cbp_opt_test_")                                      # never the tree: a relative -op must not write into it
    return subprocess.run(["bash", os.path.join(tree or _stubs.TREE_DIR, "run.sh")] + args, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_run_sh_pins_gate(tmp_path):
    """The stock pin check is unconditional: no stock/check_pins.py -> rc 3; check_pins rc 3 (stock not the pinned stock) -> rc 3; rc 0 -> the package runs."""
    kit = _stubs.make_tree(str(tmp_path / "kit_tree")); base = _stubs.base_env(kit)
    nostock = _stubs.make_run_tree(str(tmp_path / "nostock")); shutil.rmtree(os.path.join(nostock, "stock"))   # a stand-in tree without stock/
    p = _run_sh(["check", "--mode", "fast"], base, tree=nostock)
    assert p.returncode == 3 and "does not carry stock/check_pins.py" in p.stderr
    tree = _stubs.make_run_tree(str(tmp_path / "tree"))
    p = _run_sh(["check", "--mode", "fast"], dict(base, STUB_PINS_RC="3"), tree=tree)
    assert p.returncode == 3 and "is not the pinned stock" in p.stderr
    p = _run_sh(["check", "--mode", "fast"], base, tree=tree)
    assert p.returncode == 0 and p.stdout.splitlines()[0].startswith("[chrombpnet-opt] ACTIVE mode=fast"), p.stderr


def test_run_sh_refusals(tmp_path):
    kit = _stubs.make_tree(str(tmp_path / "kit_tree"))
    base = {k: v for k, v in _stubs.base_env(kit).items()}
    tree = _stubs.make_run_tree(str(tmp_path / "tree"))
    def run(args, env_extra=None, path_dirs=()):                                       # the stand-in tree (run.sh + configs + a stub stock/check_pins.py)
        return _run_sh(args, env_extra, path_dirs, tree=tree)
    assert run(["bogus"], base).returncode == 2
    p = run(["check", "--mode", "off"], dict(base, CHROMBPNET_OPT="fast")); assert p.returncode == 2 and "disagrees" in p.stderr
    p = run(["check", "--mode", "turbo"], base); assert p.returncode == 2 and "is not a mode (off|exact|fast)" in p.stderr
    p = run(["warm", "--mode", "off"], base); assert p.returncode == 2
    p = run(["check", "--config", "nope"], base); assert p.returncode == 2
    # the package not importable on the python on PATH: rc 3 with the install line
    nopy = tmp_path / "nopy"; nopy.mkdir()                                            # a python that cannot import the package (everything else runs on the real one)
    (nopy / "python").write_text('#!/bin/bash\ncase "$*" in *chrombpnet_opt*) exit 1;; *) exec {} "$@";; esac\n'.format(sys.executable)); os.chmod(str(nopy / "python"), 0o755)
    p = run(["check", "--mode", "fast"], base, [str(nopy)]); assert p.returncode == 3 and "pip install -e" in p.stderr
    # the card note: a stub nvidia-smi reporting another card than MODEL_OPT_TARGET_GPU -> a NOTE on stderr, never a refusal (the package keys its route on the card); the stock route prints none
    smi = tmp_path / "smi"; smi.mkdir(); (smi / "nvidia-smi").write_text("#!/bin/sh\necho 'NVIDIA H200 STUB'\n"); os.chmod(str(smi / "nvidia-smi"), 0o755)
    p = run(["check", "--mode", "fast"], dict(base, MODEL_OPT_TARGET_GPU="H100"), [str(smi)]); assert p.returncode == 0 and "NOTE card_name" in p.stderr and "wrong_card" not in p.stderr, p.stderr
    p = run(["check", "--mode", "off"], dict(base, MODEL_OPT_TARGET_GPU="H100"), [str(smi)]); assert "NOTE card_name" not in p.stderr, p.stderr
    p = run(["check", "--mode", "fast"], dict(base, MODEL_OPT_TARGET_GPU="H200"), [str(smi)])
    assert p.returncode == 0 and p.stdout.splitlines()[0].startswith("[chrombpnet-opt] ACTIVE mode=fast route=keras_predict_fileorder kit=0.0.0-stub gpu=none")
    # --config h100 with a root unset: rc 2, the variable named; with both roots: sourced, the package resolves, the mode line prints
    p = run(["check", "--config", "h100"], {k: v for k, v in base.items() if k not in ("CHROMBPNET_OPT_WEIGHTS", "CHROMBPNET_OPT_DATA")})
    assert p.returncode == 2 and "reason=CHROMBPNET_OPT_WEIGHTS is not set — see README Variables" in p.stderr, p.stderr
    base = dict(base, CHROMBPNET_OPT_WEIGHTS=str(tmp_path / "weights"), CHROMBPNET_OPT_DATA=str(tmp_path / "data"))
    p = run(["check", "--config", "h100"], base)
    assert p.returncode == 0 and p.stdout.splitlines()[0].startswith("[chrombpnet-opt] ACTIVE mode=fast"), p.stderr
    p = run(["pred_bw", "--config", "h100", "--mode", "exact", "-op", str(tmp_path / "o" / "p"), "-cm", "m.h5"], dict(base, STUB_GPU_CLASS="H100"))
    assert p.returncode == 0 and p.stdout.splitlines()[0] == "[chrombpnet-opt] ACTIVE mode=exact route=k1 kit=0.0.0-stub gpu=H100 det=1 precision=fp32", p.stderr
    q = run(["pred_bw", "--config", "h100", "--mode", "fast", "--det", "1", "-op", str(tmp_path / "o" / "q"), "-cm", "m.h5"], dict(base, STUB_GPU_CLASS="H100"))
    assert q.returncode == 2 and "--det is only for --mode off" in q.stderr, q.stderr          # run.sh: the kit modes take no --det
    assert json.load(open(str(tmp_path / "o" / "p_kit_call.json")))["argv"] == ["-op", str(tmp_path / "o" / "p"), "-cm", "m.h5"]


def _probe_venv(base_dir, name):
    """A minimal, dedicated venv (--without-pip: this only needs its own site-packages dir and its own `python` on
    PATH, never an installer) -- its OWN site-packages, never the shared test interpreter's, hosts each sub-case's
    .pth fixture: real, isolated, and immune to this sandbox's own interpreter having a RELATIVE sys.prefix
    (./.venv/python), which would otherwise resolve site.getsitepackages() against whatever cwd a subprocess
    happens to run from (_run_sh gives every call a fresh tempdir cwd) rather than where the fixture was planted."""
    venv_dir = pathlib.Path(base_dir) / name
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv_dir)], check=True, capture_output=True)
    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    site_pkgs = next((venv_dir / "lib").glob("python*/site-packages"))
    return str(bin_dir), site_pkgs


def _plant_pkg(site_pkgs):
    """A real, importable chrombpnet_opt package (not a .pth trick) so `python -c "import chrombpnet_opt"`
    (run.sh's own install gate, no -I) succeeds regardless of which .pth fixture a sub-case plants beside it."""
    pkg = site_pkgs / "chrombpnet_opt"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "_autoload.py").write_text("")
    return pkg


def test_run_sh_hook_live_env_route_only(tmp_path):
    """run.sh's hook-live gate fires ONLY for the stock CLI under the variable (CHROMBPNET_OPT=<mode>, no --mode on
    the command line) -- this script's own invocation always passes --mode explicitly and cannot run stock silently,
    so it is never gated, hook live or not. The check itself is hook-live in an ISOLATED interpreter (`env -u
    CHROMBPNET_OPT python -I -`): -I keeps PYTHONPATH readable via os.environ (this diagnostic's own "beside a
    PYTHONPATH entry" branch reads it as data) while stopping it from actually landing on sys.path -- so this
    stand-in tree's own PYTHONPATH (which makes chrombpnet_opt importable for every other test here) can never be
    mistaken for the hook itself having fired. Every positive/negative site-dir case below runs against a DEDICATED
    venv's own `python` (put first on PATH, so run.sh's bare `python`/`command -v python` resolves to it) and that
    venv's own real site-packages -- never the shared test interpreter's, and never PYTHONUSERBASE (-I disables
    user-site outright)."""
    kit = _stubs.make_tree(str(tmp_path / "kit_tree")); base = _stubs.base_env(kit)
    tree = _stubs.make_run_tree(str(tmp_path / "tree"))
    def run(args, env_extra=None, path_dirs=()):
        return _run_sh(args, env_extra, path_dirs=path_dirs, tree=tree)

    # --mode fast given explicitly: never gated, hook state irrelevant (this tree's PYTHONPATH-only chrombpnet_opt
    # has no real .pth processed anywhere -- if the gate fired here every other test above would already be failing)
    p = run(["check", "--mode", "fast"], base); assert p.returncode == 0 and "is not live in" not in p.stderr, p.stderr

    # CHROMBPNET_OPT=fast alone, no --mode (the stock CLI under the variable): base's own PYTHONPATH=OPT_DIR (which
    # makes chrombpnet_opt importable for every test in this file) points AT the real opt/ dir, which DOES carry a
    # real chrombpnet_opt_autoload.pth -- so the diagnostic here is "beside a PYTHONPATH entry, not a site
    # dir", never file presence proving liveness; refused, names the diagnostic and the install line. -I keeps this
    # readable (os.environ survives -I; only its sys.path effect is suppressed) so the classification still fires.
    p = run(["check"], dict(base, CHROMBPNET_OPT="fast"))
    assert (p.returncode == 3 and "is not live in" in p.stderr
            and "present but not processed (a copy beside a PYTHONPATH entry, not a site dir)" in p.stderr
            and "pip install -e" in p.stderr), p.stderr

    # CHROMBPNET_OPT=off alone, no --mode: the stock route arms nothing, never gated
    p = run(["check"], dict(base, CHROMBPNET_OPT="off")); assert p.returncode == 0, p.stderr

    # $HERE/opt/chrombpnet_opt_autoload.pth (the diagnostic's "want" comparison file, run.sh:62's sys.argv[1]) is
    # never created by make_run_tree() -- without it "want" is None and the diagnostic's own stale check can never
    # fire (byte comparison is gated on "want is not None"); plant the real shipped bytes so it can.
    shipped_bytes = (pathlib.Path(_stubs.OPT_DIR) / "chrombpnet_opt_autoload.pth").read_bytes().strip()
    (pathlib.Path(tree) / "opt").mkdir(exist_ok=True)
    (pathlib.Path(tree) / "opt" / "chrombpnet_opt_autoload.pth").write_bytes(shipped_bytes + b"\n")

    # CASE live: a real .pth (byte-identical to the shipped one) in a dedicated venv's own REGULAR site-packages,
    # beside a real, importable chrombpnet_opt._autoload -- processed by a fresh interpreter at startup (not
    # imported by the diagnostic script itself) -- succeeds, no refusal.
    bin_live, site_live = _probe_venv(tmp_path, "venv_live")
    _plant_pkg(site_live)
    (site_live / "chrombpnet_opt_autoload.pth").write_bytes(shipped_bytes + b"\n")
    p = run(["check"], dict(base, CHROMBPNET_OPT="fast"), path_dirs=(bin_live,))
    assert p.returncode == 0 and "is not live in" not in p.stderr, p.stderr

    # CASE stale: same site dir shape, a .pth whose bytes differ from the shipped one -- named, not accepted
    # (byte comparison alone classifies this; the target module need not even exist).
    bin_stale, site_stale = _probe_venv(tmp_path, "venv_stale")
    _plant_pkg(site_stale)
    (site_stale / "chrombpnet_opt_autoload.pth").write_text("import chrombpnet_opt._stale_module_name\n")
    p = run(["check"], dict(base, CHROMBPNET_OPT="fast"), path_dirs=(bin_stale,))
    assert p.returncode == 3 and "is not live in" in p.stderr and "stale copy" in p.stderr and "pip install -e" in p.stderr, p.stderr

    # CASE present-but-not-processed: the .pth's bytes MATCH the shipped one exactly (so it is never "stale"), but
    # the package it names to import is genuinely broken in THIS venv (no chrombpnet_opt installed at all here) --
    # site.py's own addpackage() logs the ImportError and moves on, leaving the hook un-fired despite a
    # byte-correct file sitting right there. This is the "present in a real site dir, deliberately broken" case:
    # distinct from "stale" (wrong bytes) and from "beside a PYTHONPATH entry" (not a site dir at all).
    bin_broken, site_broken = _probe_venv(tmp_path, "venv_broken")
    (site_broken / "chrombpnet_opt_autoload.pth").write_bytes(shipped_bytes + b"\n")   # no _plant_pkg: the import inside it has nothing to import
    p = run(["check"], dict(base, CHROMBPNET_OPT="fast"), path_dirs=(bin_broken,))
    assert (p.returncode == 3 and "is not live in" in p.stderr
            and "present but not processed (in a site dir site.py did not process at start)" in p.stderr
            and "pip install -e" in p.stderr), p.stderr

    # CASE absent: no .pth anywhere the diagnostic looks -- a dedicated venv with the stub package installed (so
    # line 54's own `import chrombpnet_opt`, without -I, still succeeds via ITS site-packages) but no
    # chrombpnet_opt_autoload.pth in it, and base's own PYTHONPATH=OPT_DIR (which would otherwise "beside"-match
    # the real shipped .pth there) cleared for this one case.
    bin_empty, site_empty = _probe_venv(tmp_path, "venv_empty")
    _plant_pkg(site_empty)
    # _run_sh layers env_extra onto a COPY of the current os.environ -- an inherited PYTHONPATH would survive a
    # bare .pop(); an explicit empty string is what actually clears it for the subprocess
    no_pythonpath = dict(base, CHROMBPNET_OPT="fast", PYTHONPATH="")
    p = run(["check"], no_pythonpath, path_dirs=(bin_empty,))
    assert (p.returncode == 3 and "is not live in" in p.stderr
            and "absent from the searched sites" in p.stderr and "pip install -e" in p.stderr), p.stderr


def test_run_sh_hook_live_gate_is_load_bearing(tmp_path):
    """Mutation test (excise-and-rerun): delete the hook-live guard block from a copy of run.sh and rerun the fast
    scenario the guard exists to catch (CHROMBPNET_OPT=fast, no --mode, hook genuinely absent -- an empty venv,
    no chrombpnet_opt_autoload.pth anywhere). Without the guard, this script's own invocation still passes --mode
    fast to the package explicitly (line 84's MODEARG, built from $MODE regardless of the guard) -- so excising the
    guard does not merely stop the refusal text, it lets execution proceed all the way to the package's own line
    (`exec python -m chrombpnet_opt check --mode fast`), which SUCCEEDS: the failure mode this guard exists to
    prevent is exactly this -- a bare `chrombpnet pred_bw` run outside this script, relying on the same env var
    with no guard, would silently run stock. Asserts both halves: refusal text ABSENT, and execution genuinely
    reached the package (the real "[chrombpnet-opt]" mode line on stdout, rc == 0 -- not just "didn't crash")."""
    kit = _stubs.make_tree(str(tmp_path / "kit_tree")); base = _stubs.base_env(kit)
    tree = _stubs.make_run_tree(str(tmp_path / "excised_tree"))
    run_sh_path = pathlib.Path(tree) / "run.sh"
    original = run_sh_path.read_text()

    lines = original.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith('if [ -z "$MODE_FROM_CLI" ]'))
    end = next(i for i, l in enumerate(lines) if i > start and l.rstrip() == "fi")
    guard_block = "".join(lines[start:end + 1])
    assert "HOOKDIAG" in guard_block and "is not live in" in guard_block, "excised the wrong block -- guard markers not found in it"
    excised = "".join(lines[:start] + lines[end + 1:])
    assert "HOOKDIAG" not in excised and "is not live in" not in excised, "guard block still present after excision"
    run_sh_path.write_text(excised)

    bin_empty, _ = _probe_venv(tmp_path, "venv_excise_empty")
    p = _run_sh(["check"], dict(base, CHROMBPNET_OPT="fast"), path_dirs=(bin_empty,), tree=tree)
    assert "is not live in" not in p.stderr, ("the excised script still refuses -- the mutation didn't remove the "
                                               "guard, so this test isn't proving the guard is load-bearing: " + p.stderr)
    assert p.returncode == 0 and p.stdout.splitlines() and p.stdout.splitlines()[0].startswith("[chrombpnet-opt]"), (
        f"excising the guard should let execution reach the package's own line (line 86) and SUCCEED -- "
        f"rc={p.returncode} stdout={p.stdout!r} stderr={p.stderr!r}")
