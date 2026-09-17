"""The two arms through the CLI, with a stand-in kit, a stub `chrombpnet` package and the stock child launcher:
* off: the stock child (stock_pred_bw.py -> the stub's CHROMBPNET.main) runs in a subprocess whose environment has no kit or package
  variable and no PYTHONPATH entry under opt/kit or the tree — proved in the launching process from names and again inside the child
  (stock_env_proof.json beside the outputs, embedded in opt_manifest.json as env_proof.child); the off line is the first line on
  stdout, the stock's rc is the exit code; the stock arm strips kit switches, never refuses on them;
* fast: the kit's documented line runs with the same arguments and no flag added, the mode line first on stdout, CHROMBPNET_OPT*
  and every kit-internal name removed from the kit's environment (a caller-set one is named on the IGNORED line, never refused — on the
  verb, on `check`, and on the env route), the rc passed through; a kit that refuses the mode by name (a lever of the class's set cannot
  start) -> NOT ACTIVE, exit 3;
* --det on either arm: the kit's seed-hook directory first on PYTHONPATH, the two variables the hook reads, the TF block; the cuBLAS pin
  on the K1 route only; the recipe's two names are the package's to set (the caller's without --det are removed and named);
* the driver-cache deviation: CHROMBPNET_JIT_CACHE_TAR is set from CHROMBPNET_OPT_CACHE_TAR only when the kit's own cache tar for the
  class is absent (a caller-set CHROMBPNET_JIT_CACHE_TAR is a kit-internal name: removed and named)."""
import json
import os
import subprocess
import sys
import tempfile

import pytest

from chrombpnet_opt import det, manifest, stack, stock_pred_bw
from . import _stubs

ARGS = ["-cm", "m.h5", "-r", "r.bed", "-g", "g.fa", "-c", "c.sizes", "-os", "p.stats", "-bw", "obs.bw"]
from chrombpnet_opt import registry
FORBIDDEN_PREFIXES = ("CHROMBPNET_OPT",)                       # the package's own namespace, the one prefix rule
FORBIDDEN_NAMES = registry.KIT_SWITCH_NAMES                    # the kit's own switch names, by name
CALLER_BOOKKEEPING = {"CHROMBPNET_KIT_HOME": "<kit>", "KIT_HOME_VAR": "CHROMBPNET_KIT_HOME", "KIT_PYTHONPATH": "opt", "GPU_COUNT": "1", "ZOO_MANIFEST": "/x",
                  "K1_BOOKKEEPING": "1", "CHROMBPNET_FASTKIT": "x", "CHROMBPNET_K1_NOTE": "y"}   # a caller's own bookkeeping sharing the kit's prefixes: never a switch
DIRTY = {"CHROMBPNET_OPT_DET": "0", "CHROMBPNET_FASTKIT_TAIL": "padded", "CHROMBPNET_K1_DIR": "/k1", "CHROMBPNET_JIT_CACHE_TAR": "/x.tar",
         "CHROMBPNET_LP_DIR": "/lp", "CHROMBPNET_DET_SUBPROCESS": "1", "K1_PREIMPORT": "0", "CHROMBPNET_OPT_MODEL": "/w/m.h5"}
KIT_SWITCHES = ["CHROMBPNET_FASTKIT_TAIL", "CHROMBPNET_FASTKIT_TAIL_SOURCE", "CHROMBPNET_FASTKIT_RECORD",
                "CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE", "K1_PREIMPORT", "K1_EXIT_TEARDOWN", "CHROMBPNET_K1_DIR",
                "CHROMBPNET_JIT_CACHE_TAR", "CHROMBPNET_DET_SUBPROCESS", "CHROMBPNET_DET_SEED"]


def mk(tmp_path, name="a", **kw):
    """A stand-in tree <tmp>/<name>/tree with the kit at opt/kit and the stub stock package
    at <tmp>/<name>_site — outside the tree (the stock arm drops PYTHONPATH entries under the tree)."""
    kit = _stubs.make_tree(str(tmp_path / name / "tree"), **kw)
    site = _stubs.make_stock_package(str(tmp_path / (name + "_site")))
    return kit, site


def run_cli(kit, args, extra_env=None, site=None, pythonpath_extra=()):
    env = _stubs.base_env(kit, extra_env)
    env["PYTHONPATH"] = os.pathsep.join([_stubs.OPT_DIR] + ([site] if site else []) + list(pythonpath_extra))
    return subprocess.run([sys.executable, "-m", "chrombpnet_opt"] + args, env=env, cwd=tempfile.mkdtemp(prefix="cbp_opt_test_"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)   # never the tree


def test_off_env_proof(tmp_path):
    kit, site = mk(tmp_path)
    tree = os.path.dirname(os.path.dirname(kit))
    op = str(tmp_path / "out" / "p")
    dirty = dict(DIRTY, STUB_RC="7", CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE="L40S")
    under_kit, under_tree, elsewhere = os.path.join(kit, "torch"), os.path.join(tree, "opt"), str(tmp_path / "elsewhere")
    p = run_cli(kit, ["pred_bw", "--mode", "off", "-op", op] + ARGS, extra_env=dirty, site=site, pythonpath_extra=[under_kit, under_tree, elsewhere])
    assert p.returncode == 7, p.stderr
    assert p.stdout.splitlines()[0] == "[chrombpnet-opt] NOT ACTIVE mode=off (stock)"
    assert p.stdout.splitlines()[1] == "[stub stock main] ran"                    # the child prints nothing of its own before the stock
    call = json.load(open(op + "_stock_call.json"))
    assert call["argv"] == ["pred_bw", "-op", op] + ARGS                            # the console script's argv, as the stock's parser reads it
    assert [k for k in call["env"] if k.startswith(FORBIDDEN_PREFIXES) or k in FORBIDDEN_NAMES] == []
    assert all(m in stock_pred_bw.INERT_FINDER_MODULES for m in call["modules"]) and not [p for p in call["sys_path"] if p.startswith(kit)]   # no package module in the child beyond the installed .pth's inert finder (launched by file path); nothing of the kit on sys.path
    proof = json.load(open(os.path.join(os.path.dirname(op), stock_pred_bw.PROOF_FILENAME)))
    assert proof["ok"] and proof["forbidden_present"] == [] and proof["sys_path_under_kit"] == [] and proof["sys_path_under_tree"] == []
    assert proof["autoload_armed"] is False and proof["kit_modules_loaded"] == [] and proof["det"] is False and proof["det_hook_on_path"] is False
    assert proof["pythonpath"] == [_stubs.OPT_DIR, site, elsewhere] and proof["argv"][1:] == ["-op", op] + ARGS and proof["pid"] and proof["executable"] == sys.executable
    assert isinstance(proof["launcher_dir_on_path"], bool) and proof["package_modules_loaded"] == []   # the script dir is on sys.path unless the interpreter runs with safe-path (PYTHONSAFEPATH)
    man = manifest.read(os.path.dirname(op))
    assert man["schema"] == manifest.SCHEMA and man["mode"] == "off" and man["route"] == "stock_cli" and man["exit_code"] == 7
    assert man["env_proof"]["parent"]["ok"] and man["env_proof"]["parent"]["forbidden_present"] == [] and man["det"]["on"] is False
    assert man["env_proof"]["parent"]["pythonpath_dropped"] == [under_kit, under_tree]
    assert man["env_proof"]["child"]["ok"] and man["env_proof"]["child"]["pid"] == proof["pid"]
    assert "CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE" in man["env_stripped"] and "K1_PREIMPORT" in man["env_stripped"]   # off strips kit switches, never refuses on them


def test_off_with_det(tmp_path):
    kit, site = mk(tmp_path)
    op = str(tmp_path / "out" / "p")
    p = run_cli(kit, ["pred_bw", "--mode", "off", "--det", "1", "-op", op] + ARGS, site=site, pythonpath_extra=["/elsewhere"])
    assert p.returncode == 0, p.stderr
    env = json.load(open(op + "_stock_call.json"))["env"]
    hook = os.path.join(_stubs.fast_kit(kit), "tf", "det_subprocess")                # the fast mode's kit (opt/kit_ho) carries the seed hook it runs with
    assert env["PYTHONPATH"].split(os.pathsep) == [hook, _stubs.OPT_DIR, site, "/elsewhere"]
    assert env["CHROMBPNET_DET_SUBPROCESS"] == "1" and env["CHROMBPNET_DET_SEED"] == det.SEED
    for k, v in det.TF_ENV.items():
        assert env[k] == v
    assert "CUBLAS_WORKSPACE_CONFIG" not in env                                     # the K1 pin is the K1 route's only
    assert env.get("CHROMBPNET_DET_SUBPROCESS_APPLIED", "").startswith("stub seed=" + det.SEED)   # the hook ran at the child's start
    proof = json.load(open(os.path.join(os.path.dirname(op), stock_pred_bw.PROOF_FILENAME)))
    assert proof["ok"] and proof["det"] and proof["det_hook_on_path"] and proof["sys_path_under_kit"] == [] and proof["forbidden_present"] == []
    man = manifest.read(os.path.dirname(op))
    assert man["det"]["on"] and man["det"]["seed_hook"] == "tf/det_subprocess/sitecustomize.py" and man["env_proof"]["parent"]["ok"] and man["env_proof"]["child"]["ok"]


def test_off_needs_the_stock_package(tmp_path):
    kit, _ = mk(tmp_path)
    op = str(tmp_path / "out" / "p")
    p = run_cli(kit, ["pred_bw", "--mode", "off", "-op", op] + ARGS)                # no stub stock package on the path
    assert p.returncode == 3 and p.stdout.splitlines()[0] == "[chrombpnet-opt] NOT ACTIVE mode=off (stock)"
    assert "the stock package is not importable" in p.stderr
    assert json.load(open(os.path.join(os.path.dirname(op), stock_pred_bw.PROOF_FILENAME)))["ok"]


def test_child_proof_from_names():
    proof = stock_pred_bw.env_proof("/k", "/t", False, environ={"HOME": "/h", "PYTHONPATH": "/a"}, path=["/a", "/x"], modules={"os": None}, argv=["x"])
    assert proof["ok"] and proof["pythonpath"] == ["/a"] and proof["forbidden_present"] == []
    proof = stock_pred_bw.env_proof("/k", "/t", False, environ={"K1_PREIMPORT": "1", "CHROMBPNET_OPT": "fast", "K1_BOOKKEEPING": "1"}, path=["/k/torch", "/t/opt"], modules={}, argv=["x"])
    assert not proof["ok"] and proof["forbidden_present"] == ["CHROMBPNET_OPT", "K1_PREIMPORT"] and proof["sys_path_under_kit"] == ["/k/torch"] and proof["sys_path_under_tree"] == ["/t/opt"]
    proof = stock_pred_bw.env_proof("/k", "/t", True, environ={"CHROMBPNET_DET_SUBPROCESS": "1"}, path=["/k/tf/det_subprocess"], modules={}, argv=["x"])
    assert proof["ok"] and proof["det_hook_on_path"] and proof["sys_path_under_kit"] == []
    assert stock_pred_bw.FORBIDDEN_PREFIXES == stack.STOCK_FORBIDDEN_PREFIXES == ("CHROMBPNET_OPT",)
    assert tuple(sorted(stock_pred_bw.FORBIDDEN_NAMES)) == tuple(sorted(stack.STOCK_FORBIDDEN_NAMES)) == tuple(sorted(registry.KIT_SWITCH_NAMES)) and stock_pred_bw.DET_NAMES == registry.KIT_SWITCH_DET_NAMES


@pytest.mark.parametrize("gpu_class,route", [("H100", "k1"), (None, "keras_predict_fileorder")])
def test_fast_documented_line(tmp_path, gpu_class, route):
    kit, _ = mk(tmp_path)
    op = str(tmp_path / "out" / "p")
    extra = {"STUB_RC": "5", "CHROMBPNET_OPT_DET": "0", "CHROMBPNET_OPT_MODEL": "/w/m.h5", "HOME_X": "1"}
    if gpu_class:
        extra["STUB_GPU_CLASS"] = gpu_class
    p = run_cli(kit, ["pred_bw", "-op", op] + ARGS, extra_env=extra)                # no --mode: fast is the default
    assert p.returncode == 5, p.stderr
    gpu_word = gpu_class or "none"
    assert p.stdout.splitlines()[0] == "[chrombpnet-opt] ACTIVE mode=fast route={} kit=0.0.0-stub gpu={} det=0 precision=tf32".format(route, gpu_word)
    assert p.stdout.splitlines()[1] == "[stub kit] ran"
    call = json.load(open(op + "_kit_call.json"))
    assert call["argv"] == ["-op", op] + ARGS                                       # the same arguments, nothing added
    assert call["python"] == sys.executable
    assert not [k for k in call["env"] if k.startswith("CHROMBPNET_OPT")]          # the package's own switches never reach the kit process
    assert call["env"]["HOME_X"] == "1"                                             # the rest of the environment passes through
    man = manifest.read(os.path.dirname(op))
    assert man["mode"] == "fast" and man["route"] == route and man["exit_code"] == 5 and man["kit"]["version"] == "0.0.0-stub"
    assert man["arm_argv"] == [sys.executable, os.path.join(_stubs.fast_kit(kit), "tf", "pred_bw_fast.py"), "-op", op] + ARGS
    assert man["kit"]["integrity"]["missing"] == [] and man["kit"]["integrity"]["scope"] == "route"
    assert man["cache_tar"]["source"].startswith("none")


def test_fast_exit_rule_from_the_run_record(tmp_path):
    """The kit's run record judged after the run: the tables' forward -> 0 with the applied set; another forward -> 3 (PARTIAL, the
    manifest names it and the kit's reasons); the kit's declared rule (skipped, the route standing) -> gated, 0; no record -> 3; the shipped driver cache's fallback -> 3;
    outputs the kit listed absent -> incomplete, 1 (never partial); a failed job -> its own rc, the state still recorded; the record is folded
    into opt_manifest.json (`kit_record`) and no record file is left beside the outputs."""
    kit, _ = mk(tmp_path)
    cfg = tmp_path / "nv_compute_cache_H100.tar"; cfg.write_text("staged\n")
    n = [0]

    def pred(extra=(), **env):
        n[0] += 1
        op = str(tmp_path / "out{}".format(n[0]) / "p")
        e = {"STUB_GPU_CLASS": "H100", "CHROMBPNET_OPT_DET": "0"}; e.update(env)
        p = run_cli(kit, ["pred_bw", *extra, "-op", op] + ARGS, extra_env=e)
        return p, manifest.read(os.path.dirname(op)), op

    p, man, op = pred()
    assert p.returncode == 0, p.stderr
    assert man["route"] == "k1" and man["partial"] == [] and man["gated"] == [] and "allow_partial" not in man and man["incomplete"] is None
    assert man["levers_applied"] == ["forward_route"] and man["exit_code"] == 0 and man["job_rc"] == 0
    assert [it["prefix"] for it in man["kit_record"]["items"]] == [op] and man["kit_record"]["items"][0]["forward"] == "k1 (stub)"
    assert sorted(os.listdir(os.path.dirname(op))) == ["opt_manifest.json", "p_chrombpnet.bw", "p_chrombpnet_preds.bed", "p_kit_call.json"]   # one record file beside the outputs (p_kit_call.json is the stub's own trace)
    assert "partial=none gated=0 incomplete=none" in p.stderr and "PARTIAL" not in p.stderr and p.stdout.splitlines()[0].startswith("[chrombpnet-opt] ACTIVE mode=fast route=k1 ")
    assert not [l for l in p.stdout.splitlines() if "NOT ACTIVE" in l or "PARTIAL" in l]
    # the kit ran another forward than its tables give (a K1 HOLD/OFF at run time): partial -> 3, the outputs and the manifest stay
    p, man, op = pred(STUB_STAMP_FORWARD="tf_function", STUB_STAMP_SKIPPED="K1 route: OFF — the K1 stack failed to import (stub); TF route")
    assert p.returncode == 3, p.stderr
    assert p.stdout.splitlines()[-1] == ("[chrombpnet-opt] NOT ACTIVE: partial activation — levers=forward_route: forward_route: p: the kit ran forward=tf_function, "
                                         "the tables' route is k1: K1 route: OFF — the K1 stack failed to import (stub); TF route; exit 3 (--mode off runs stock)")
    assert "PARTIAL" not in p.stderr
    assert man["partial"] == ["forward_route"] and man["exit_code"] == 3 and man["job_rc"] == 0 and man["levers_applied"] == []
    assert os.path.isfile(op + "_chrombpnet.bw") and "partial=forward_route gated=0" in p.stderr
    assert run_cli(kit, ["pred_bw", "--allow" + "-partial", "-op", op] + ARGS, extra_env={"STUB_GPU_CLASS": "H100", "STUB_STAMP_FORWARD": "tf_function"}).returncode == 3   # no such switch: an unknown `--` word passes through to the kit's line as a stock argument and the partial run still exits 3
    # the kit refused the mode BY NAME (a lever of the class's set could not start: it ran nothing, wrote `refused` in its record, exit 3): NOT ACTIVE, 3
    p, man, op = pred(STUB_REFUSED="the K1 stack failed to import in the job process (ImportError: stub)")
    assert p.returncode == 3, p.stdout + p.stderr
    assert p.stdout.splitlines()[-1] == "[chrombpnet-opt] NOT ACTIVE mode=fast reason=the kit refused by name: the K1 stack failed to import in the job process (ImportError: stub) (a mode is all of its levers on a class or nothing; --mode off runs stock); exit 3"
    assert man["refused"] == "the K1 stack failed to import in the job process (ImportError: stub)" and man["exit_code"] == 3 and man["job_rc"] == 3 and "partial=refused" in p.stderr
    assert not os.path.isfile(op + "_chrombpnet.bw")                                    # it ran nothing
    # a kit-internal name in the caller's environment: removed for the run, named on the IGNORED line, the mode ACTIVE (never a refusal)
    p, man, op = pred(K1_NUM_WARPS="8", CHROMBPNET_FASTKIT_TAIL="stock")
    assert p.returncode == 0, p.stderr
    assert "[chrombpnet-opt] IGNORED names=CHROMBPNET_FASTKIT_TAIL,K1_NUM_WARPS reason=kit-internal names set in the environment are removed for the run (the mode is the whole composition)" in p.stderr.splitlines()
    assert man["ignored"] == ["CHROMBPNET_FASTKIT_TAIL", "K1_NUM_WARPS"] and {"CHROMBPNET_FASTKIT_TAIL", "K1_NUM_WARPS"} <= set(man["env_stripped"]) and man["partial"] == []
    call = json.load(open(op + "_kit_call.json"))
    assert "K1_NUM_WARPS" not in call["env"] and "CHROMBPNET_FASTKIT_TAIL" not in call["env"]      # the kit's line never saw them
    # the kit's declared rule with the route standing: gated, exit-neutral
    p, man, op = pred(STUB_STAMP_SKIPPED="native_dilation: class 'H100' uncertified")
    assert p.returncode == 0, p.stderr
    assert man["partial"] == [] and man["gated"] == ["forward_route: native_dilation: class 'H100' uncertified (the kit's declared rule; the tables' route stands)"]
    assert "[chrombpnet-opt] GATED forward_route: native_dilation" in p.stderr and "gated=1" in p.stderr
    # no record at all: the kit's record is absent -> partial
    p, man, op = pred(STUB_NO_RECORD="1")
    assert p.returncode == 3 and man["partial"] == ["forward_route"] and "NOT ACTIVE: partial activation — levers=forward_route: forward_route: p: no run record" in p.stdout and man["incomplete"] is None
    assert man["kit_record"] == {"kit": "chrombpnet_fastkit", "items": []}
    # the shipped driver cache fell back (the package composed the tarball): partial
    p, man, op = pred(STUB_STAMP_FORWARD="tf_function", STUB_JIT_CACHE_FALLBACK="identity mismatch (stub) -> the stock's own JIT", CHROMBPNET_OPT_CACHE_TAR=str(cfg))
    assert man["cache_tar"]["set"] is True and p.returncode == 3 and man["partial"] == ["forward_route", "jit_cache"] and "the shipped driver cache fell back" in p.stdout
    assert p.stdout.splitlines()[-1].startswith("[chrombpnet-opt] NOT ACTIVE: partial activation — levers=forward_route,jit_cache: ") and p.stdout.splitlines()[-1].endswith("; exit 3 (--mode off runs stock)")
    p, man, op = pred(CHROMBPNET_OPT_CACHE_TAR=str(cfg))
    assert p.returncode == 0 and man["levers_applied"] == ["forward_route", "jit_cache"] and man["partial"] == []
    # outputs the kit listed absent: incomplete -> 1, never partial
    p, man, op = pred(STUB_NO_OUTPUTS="1")
    assert p.returncode == 1, p.stderr
    assert man["incomplete"] == "0/1" and man["partial"] == [] and man["exit_code"] == 1 and man["job_rc"] == 0 and "incomplete=0/1" in p.stderr
    # a failed job: its own rc as is, the partial state recorded
    p, man, op = pred(STUB_RC="5", STUB_STAMP_FORWARD="tf_function")
    assert p.returncode == 5 and man["exit_code"] == 5 and man["job_rc"] == 5 and man["partial"] == ["forward_route"]
    # warm follows pred_bw: its 3 is warm's 3
    bed = tmp_path / "regions.bed"; bed.write_text("chr1\t100\t200\n" * 4)
    data = {"CHROMBPNET_OPT_MODEL": "m.h5", "CHROMBPNET_OPT_GENOME": "g.fa", "CHROMBPNET_OPT_CHROM_SIZES": "c.sizes", "CHROMBPNET_OPT_REGIONS": str(bed), "STUB_GPU_CLASS": "H100", "STUB_STAMP_FORWARD": "tf_function"}
    p = run_cli(kit, ["warm", "--mode", "fast", "--n", "2", "--out", str(tmp_path / "warm1")], extra_env=data)
    assert p.returncode == 3, p.stderr
    assert "WARM FAIL mode=fast n=2 rc=3" in p.stderr and "[chrombpnet-opt] NOT ACTIVE: partial activation — levers=forward_route: " in p.stdout   # the child's line, through the one formatter
    data_ok = dict(data); data_ok.pop("STUB_STAMP_FORWARD")
    p = run_cli(kit, ["warm", "--mode", "fast", "--n", "2", "--out", str(tmp_path / "warm2")], extra_env=data_ok)
    assert p.returncode == 0, p.stderr
    assert "WARM PASS mode=fast n=2 rc=0 wall=" in p.stderr and manifest.read(str(tmp_path / "warm2"))["partial"] == []


def test_items_exit_rule_from_the_run_record(tmp_path):
    """`pred_bw --items`: every item's record entry judged — one item's forward off the tables' route -> partial, 3, the line naming the item;
    the record short of the items -> incomplete, 1 (never partial); every entry on the route -> 0."""
    kit, _ = mk(tmp_path)
    bed = tmp_path / "a.bed"; bed.write_text("chr1\t1000\t3000\t.\t0\t.\t0\t0\t0\t1000\n")
    n = [0]

    def items(extra=(), **env):
        n[0] += 1
        d = tmp_path / "items{}".format(n[0]); d.mkdir()
        items_path = d / "items.tsv"
        items_path.write_text("# regions\toutput_prefix\tstats\n" + "".join("{}\t{}\t{}\n".format(bed, d / name, d / (name + ".stats")) for name in ("a", "b", "c")))
        e = {"STUB_GPU_CLASS": "H100", "CHROMBPNET_OPT_DET": "0"}; e.update(env)
        p = run_cli(kit, ["pred_bw", *extra, "--items", str(items_path), "-cm", "m.h5", "-g", "g.fa", "-c", "c.sizes"], extra_env=e)
        return p, manifest.read(str(d)), str(items_path)

    p, man, items_path = items()
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.splitlines()[0].endswith(" multi=3") and man["partial"] == [] and man["incomplete"] is None and man["levers_applied"] == ["forward_route"]
    assert [it["prefix"] for it in man["kit_record"]["items"]] == [os.path.join(os.path.dirname(items_path), name) for name in ("a", "b", "c")]
    assert not [f for f in os.listdir(os.path.dirname(items_path)) if f.endswith(".json") and f != "opt_manifest.json" and not f.endswith("_kit_call.json")]   # one record file
    p, man, items_path = items(STUB_STAMP_FORWARD="tf_function")
    assert p.returncode == 3, p.stdout + p.stderr
    line = p.stdout.splitlines()[-1]
    assert line.startswith("[chrombpnet-opt] NOT ACTIVE: partial activation — levers=forward_route: forward_route: a: the kit ran forward=tf_function, the tables' route is k1") \
        and "; forward_route: b: the kit ran forward=tf_function" in line and "; forward_route: c: the kit ran forward=tf_function" in line \
        and line.endswith("; exit 3 (--mode off runs stock)")
    assert man["partial"] == ["forward_route"] and man["exit_code"] == 3 and man["job_rc"] == 0 and len(man["partial_reasons"]) == 3
    p, man, items_path = items(STUB_RECORD_SHORT="1")
    assert p.returncode == 1, p.stdout + p.stderr
    assert man["incomplete"] == "2/3" and man["partial"] == [] and man["exit_code"] == 1 and man["job_rc"] == 0 and "incomplete=2/3" in p.stderr


def test_partial_line_family_grammar():
    """report.partial_line: the family grammar's literal parts (the one formatter every entry point prints through)."""
    from chrombpnet_opt import report
    assert report.partial_line("levers=forward_route: forward_route: p: why", 3) == \
        "[chrombpnet-opt] NOT ACTIVE: partial activation — levers=forward_route: forward_route: p: why; exit 3 (--mode off runs stock)"
    from chrombpnet_opt import cli
    assert cli.EXIT_INACTIVE == 3 and report.partial_line("d", cli.EXIT_INACTIVE).endswith("; exit 3 (--mode off runs stock)")
    assert report.ignored_line(["K1_PREIMPORT", "CHROMBPNET_K1_DIR"], "why") == "[chrombpnet-opt] IGNORED names=K1_PREIMPORT,CHROMBPNET_K1_DIR reason=why"


@pytest.mark.parametrize("name", KIT_SWITCHES)
def test_named_mode_ignores_kit_switches(tmp_path, name):
    """A mode is the whole composition: a kit-internal name set in the environment is removed from the kit's environment and named on the IGNORED
    line — fast stays ACTIVE, on the verb and on check; never a refusal."""
    kit, _ = mk(tmp_path)
    op = str(tmp_path / "out" / "p")
    line = "[chrombpnet-opt] IGNORED names={} reason=kit-internal names set in the environment are removed for the run (the mode is the whole composition)".format(name)
    p = run_cli(kit, ["pred_bw", "--mode", "fast", "-op", op] + ARGS, extra_env={name: "1", "STUB_GPU_CLASS": "H100"})
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines()[0] == "[chrombpnet-opt] ACTIVE mode=fast route=k1 kit=0.0.0-stub gpu=H100 det=0 precision=tf32" and line in p.stderr.splitlines(), p.stdout + p.stderr
    assert json.load(open(op + "_kit_call.json"))["env"].get(name) != "1"              # the kit's line never saw the caller's value (CHROMBPNET_FASTKIT_RECORD is then the package's own handshake)
    assert name in manifest.read(os.path.dirname(op))["env_stripped"]
    p = run_cli(kit, ["check", "--mode", "fast"], extra_env={name: "1", "STUB_GPU_CLASS": "H100"})
    assert p.returncode == 0 and p.stdout.splitlines()[0].startswith("[chrombpnet-opt] ACTIVE mode=fast route=k1 ") and line in p.stderr.splitlines(), p.stdout + p.stderr


def test_switch_names_are_the_kits_own(tmp_path):
    """The one-mode table keys on the kit's own switch NAMES read from its bytes, never on a prefix: a caller's bookkeeping variables that
    share the kit's prefixes (a caller's bookkeeping environment) leave fast ACTIVE, on the verb, on check and on the env route; every name of
    the list refuses; the list is the grep of the carried kit when the kit is in the tree."""
    import re
    kit, _ = mk(tmp_path)
    op = str(tmp_path / "out" / "p")
    shaped = dict(CALLER_BOOKKEEPING, CHROMBPNET_KIT_HOME=kit, STUB_GPU_CLASS="H100")
    p = run_cli(kit, ["pred_bw", "--mode", "fast", "-op", op] + ARGS, extra_env=shaped)
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines()[0] == "[chrombpnet-opt] ACTIVE mode=fast route=k1 kit=0.0.0-stub gpu=H100 det=0 precision=tf32"
    env = json.load(open(op + "_kit_call.json"))["env"]
    assert env["CHROMBPNET_KIT_HOME"] == kit and env["K1_BOOKKEEPING"] == "1"                 # the caller's bookkeeping passes through untouched
    p = run_cli(kit, ["check", "--mode", "fast"], extra_env=shaped)
    assert p.returncode == 0 and p.stdout.splitlines()[0].startswith("[chrombpnet-opt] ACTIVE mode=fast")
    assert stack.kit_switches_set(shaped) == [] and stack.kit_switches_set(dict(shaped, K1_NUM_WARPS="4")) == ["K1_NUM_WARPS"]
    assert len(registry.KIT_SWITCH_NAMES) == 17 and len(set(registry.KIT_SWITCH_NAMES)) == 17 and set(registry.KIT_SWITCH_DET_NAMES) <= set(registry.KIT_SWITCH_NAMES)
    for name in registry.KIT_SWITCH_NAMES:
        assert re.match(r"^(CHROMBPNET_(DET|FASTKIT|JIT|K1|LP)_|K1_)[A-Z0-9_]+$", name), name
    real, real_ho = _stubs.real_kit(), _stubs.real_kit("kit_ho")
    if real and real_ho:                                                             # the list against the kit's bytes: every name the kit's non-test code reads — the ONE tf/ tree (opt/kit_ho) plus the carried torch side (opt/kit)
        pat = re.compile(r"""(?:os\.environ(?:\.get)?\s*[\[(]|getenv\()\s*['"]([A-Z0-9_]+)['"]""")
        found = set()
        for root in (os.path.join(real_ho, "tf"), os.path.join(real, "torch")):
            for dp, dns, fns in os.walk(root):
                dns[:] = [d for d in dns if d != "vendor"]
                for fn in fns:
                    if fn.endswith(".py") and not fn.startswith("test_"):
                        found |= set(pat.findall(open(os.path.join(dp, fn), encoding="utf-8", errors="replace").read()))
        framework = {"TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "CUDA_CACHE_MAXSIZE", "TF_DETERMINISTIC_OPS", "TF_FORCE_GPU_ALLOW_GROWTH", "NPY_DISABLE_CPU_FEATURES", "PYTHONPATH"}
        expect = set(registry.KIT_SWITCH_NAMES) - {"K1_EXIT_TEARDOWN"}
        assert found - framework == expect, sorted((found - framework) ^ expect)
        assert "K1_EXIT_TEARDOWN" in open(os.path.join(real, "torch", "chrombpnet_k1", "_exit.py"), encoding="utf-8").read()   # read indirectly (the name as a parameter)
        assert not os.path.exists(os.path.join(real, "tf"))                          # one fast-kit tree: the carried kit holds no tf/ side


def test_det_names_are_the_packages_under_det(tmp_path):
    """The recipe's two names set by the caller WITH --det 1 are the package's own composition (kept); without it they are kit-internal names: removed and named."""
    kit, _ = mk(tmp_path)
    op = str(tmp_path / "out" / "p")
    extra = {"CHROMBPNET_DET_SUBPROCESS": "1", "CHROMBPNET_DET_SEED": "0", "STUB_GPU_CLASS": "H100"}
    p = run_cli(kit, ["pred_bw", "--mode", "exact", "-op", op] + ARGS, extra_env=extra)
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines()[0] == "[chrombpnet-opt] ACTIVE mode=exact route=k1 kit=0.0.0-stub gpu=H100 det=1 precision=fp32"
    p = run_cli(kit, ["pred_bw", "--mode", "fast", "-op", op] + ARGS, extra_env=extra)
    assert p.returncode == 0 and "[chrombpnet-opt] IGNORED names=CHROMBPNET_DET_SEED,CHROMBPNET_DET_SUBPROCESS reason=" in p.stderr and p.stdout.splitlines()[0].endswith("det=0 precision=tf32"), p.stdout + p.stderr
    env = json.load(open(op + "_kit_call.json"))["env"]
    assert "CHROMBPNET_DET_SEED" not in env and "CHROMBPNET_DET_SUBPROCESS" not in env          # removed for the run: the kit ran at shipped numerics, as the line says
    assert stack.kit_switches_set({"CHROMBPNET_DET_SUBPROCESS": "1", "CHROMBPNET_DET_SEED": "0", "HOME": "/h"}, det_on=True) == []
    assert stack.kit_switches_set({"CHROMBPNET_DET_SUBPROCESS": "1", "K1_PREIMPORT": "1", "K1_BOOKKEEPING": "1"}, det_on=False) == ["CHROMBPNET_DET_SUBPROCESS", "K1_PREIMPORT"]


def test_exact_composes_the_recipe_k1_pin(tmp_path):
    kit, _ = mk(tmp_path)
    op = str(tmp_path / "out" / "p")
    p = run_cli(kit, ["pred_bw", "--mode", "exact", "-op", op] + ARGS, extra_env={"STUB_GPU_CLASS": "H100"})
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines()[0] == "[chrombpnet-opt] ACTIVE mode=exact route=k1 kit=0.0.0-stub gpu=H100 det=1 precision=fp32"   # the stub kit's table: det -> k1 on H100
    env = json.load(open(op + "_kit_call.json"))["env"]
    assert env["PYTHONPATH"].split(os.pathsep)[0] == os.path.join(_stubs.fast_kit(kit), "tf", "det_subprocess") and env["CHROMBPNET_DET_SUBPROCESS"] == "1"
    assert env["CUBLAS_WORKSPACE_CONFIG"] == det.K1_ENV["CUBLAS_WORKSPACE_CONFIG"]        # the resolved route is k1: the K1 pin rides the recipe
    assert "CUBLAS_WORKSPACE_CONFIG" not in det.env(_stubs.fast_kit(kit), {}, route="tf_function")   # the pin is the K1 route's only


def test_exact_refused_by_name_without_a_bitwise_triton_route(tmp_path):
    """`exact` is the fp32 Triton kernels in stock's summation order, bit for bit. A class whose arch has no bitwise Triton route under the recipe
    (the stub's A100: det -> tf_function, as the kit's cuda-80 entry) has no exact mode: refused by name before anything runs, exit 3, the
    kit's line never launched; `check` says the same; `fast` on that class is the kit (route=k1)."""
    kit, _ = mk(tmp_path)
    op = str(tmp_path / "out" / "p")
    want = "[chrombpnet-opt] NOT ACTIVE mode=exact reason=the kit refused by name: no bitwise Triton route on cuda-80 (gpu=A100)"
    p = run_cli(kit, ["pred_bw", "--mode", "exact", "-op", op] + ARGS, extra_env={"STUB_GPU_CLASS": "A100"})
    assert p.returncode == 3 and p.stdout.splitlines()[0].startswith(want) and "--mode off --det 1" in p.stdout.splitlines()[0], p.stdout + p.stderr
    assert not os.path.exists(op + "_kit_call.json")                                                      # nothing ran
    p = run_cli(kit, ["check", "--mode", "exact"], extra_env={"STUB_GPU_CLASS": "A100"})
    assert p.returncode == 3 and p.stdout.splitlines() == [p.stdout.splitlines()[0]] and p.stdout.startswith(want), p.stdout + p.stderr
    p = run_cli(kit, ["pred_bw", "--mode", "fast", "-op", op] + ARGS, extra_env={"STUB_GPU_CLASS": "A100"})
    assert p.returncode == 0 and p.stdout.splitlines()[0] == "[chrombpnet-opt] ACTIVE mode=fast route=k1 kit=0.0.0-stub gpu=A100 det=0 precision=tf32", p.stdout + p.stderr
    p = run_cli(kit, ["check", "--mode", "exact"], extra_env={"STUB_GPU_CLASS": "H100"})                  # a class with the bitwise Triton route keeps it
    assert p.returncode == 0 and p.stdout.splitlines() == ["[chrombpnet-opt] ACTIVE mode=exact route=k1 kit=0.0.0-stub gpu=H100 det=1 precision=fp32"]


def test_det_is_the_stock_arms_setting(tmp_path):
    """`--det` exists on `--mode off` only (default 0): `off --det 1` = stock with TensorFlow's determinism settings, the reference `exact` equals.
    `fast --det N` / `exact --det N` are usage errors by name; CHROMBPNET_OPT_DET=1 with `fast` refuses the mode by name; with `exact` it is redundant."""
    kit, _ = mk(tmp_path)
    op = str(tmp_path / "out" / "p")
    for m in ("fast", "exact"):
        for d in ("0", "1"):
            p = run_cli(kit, ["pred_bw", "--mode", m, "--det", d, "-op", op] + ARGS, extra_env={"STUB_GPU_CLASS": "H100"})
            assert p.returncode == 2 and "--det is a stock-side setting" in p.stderr and "`{} --det {}` does not exist".format(m, d) in p.stderr, (m, d, p.stderr)
            p = run_cli(kit, ["check", "--mode", m, "--det", d], extra_env={"STUB_GPU_CLASS": "H100"})
            assert p.returncode == 2, (m, d, p.stderr)
    p = run_cli(kit, ["pred_bw", "--mode", "fast", "-op", op] + ARGS, extra_env={"CHROMBPNET_OPT_DET": "1", "STUB_GPU_CLASS": "H100"})
    assert p.returncode == 3 and p.stdout.startswith("[chrombpnet-opt] NOT ACTIVE mode=fast reason=CHROMBPNET_OPT_DET=1 with mode fast"), p.stdout + p.stderr
    p = run_cli(kit, ["pred_bw", "--mode", "exact", "-op", op] + ARGS, extra_env={"CHROMBPNET_OPT_DET": "1", "STUB_GPU_CLASS": "H100"})
    assert p.returncode == 0 and p.stdout.splitlines()[0] == "[chrombpnet-opt] ACTIVE mode=exact route=k1 kit=0.0.0-stub gpu=H100 det=1 precision=fp32", p.stdout + p.stderr
    p = run_cli(kit, ["pred_bw", "--mode", "off", "--det", "1", "-op", op] + ARGS)                  # off keeps --det 0|1: the recipe rides the stock child (the stub site has no stock package: the child's rc is its own)
    assert p.stdout.splitlines()[0] == "[chrombpnet-opt] NOT ACTIVE mode=off (stock)" and "det=1" in [l for l in p.stderr.splitlines() if " EXIT " in l][-1], p.stdout + p.stderr
    p = run_cli(kit, ["check", "--mode", "off", "--det", "0"])
    assert p.returncode == 0 and p.stdout.splitlines() == ["[chrombpnet-opt] NOT ACTIVE mode=off (stock)"], p.stdout + p.stderr


def test_mode_disagreement_is_usage(tmp_path):
    kit, _ = mk(tmp_path)
    p = run_cli(kit, ["check", "--mode", "off"], extra_env={"CHROMBPNET_OPT": "fast"})
    assert p.returncode == 2 and "disagrees" in p.stderr


def test_cache_tar_deviation(tmp_path):
    """CHROMBPNET_JIT_CACHE_TAR is set from the config only when the kit's own cache/nv_compute_cache_<class>.tar is absent."""
    cfg = tmp_path / "nv_compute_cache_H100.tar"; cfg.write_text("staged\n")
    kit_without, _ = mk(tmp_path, "without")
    kit_with, _ = mk(tmp_path, "with", with_cache_for=("H100",))
    base = {"CHROMBPNET_OPT_CACHE_TAR": str(cfg)}
    r = stack.cache_tar_env(base, kit_without, "H100")
    assert r["set"] and r["path"] == str(cfg) and r["source"].startswith("config")
    r = stack.cache_tar_env(base, kit_with, "H100")
    assert not r["set"] and r["path"] == os.path.join(kit_with, "cache", "nv_compute_cache_H100.tar") and r["source"] == "kit (HERE)"
    r = stack.cache_tar_env({"CHROMBPNET_OPT_CACHE_TAR": str(tmp_path / "missing.tar")}, kit_without, "H100")
    assert not r["set"] and r["source"].startswith("none (config path absent")
    assert not stack.cache_tar_env(base, kit_without, "unknown")["set"]
    for name, kit, expect in (("without", kit_without, str(cfg)), ("with", kit_with, None)):   # end to end: the kit process sees the config's tar only on the kit without its own copy
        op = str(tmp_path / name / "out" / "p")
        p = run_cli(kit, ["pred_bw", "-op", op] + ARGS, extra_env=dict(base, STUB_GPU_CLASS="H100"))
        assert p.returncode == 0, p.stderr
        assert json.load(open(op + "_kit_call.json"))["env"].get("CHROMBPNET_JIT_CACHE_TAR") == expect
        assert manifest.read(os.path.dirname(op))["cache_tar"]["set"] is (expect is not None)


def test_check_no_gpu_and_exit_codes(tmp_path):
    kit, _ = mk(tmp_path)
    p = run_cli(kit, ["check", "--mode", "fast", "--json"])
    assert p.returncode == 0, p.stderr
    lines = p.stdout.split("\n", 1)
    assert lines[0] == "[chrombpnet-opt] ACTIVE mode=fast route=keras_predict_fileorder kit=0.0.0-stub gpu=none det=0 precision=tf32"
    rep = json.loads(lines[1])
    assert rep["dry_run"] and rep["gpu"]["available"] is False and rep["applied"].startswith("none in this process") and rep["kit_integrity"]["scope"] == "full"
    p = run_cli(kit, ["check", "--mode", "exact"], extra_env={"STUB_GPU_CLASS": "H100"})
    assert p.stdout.splitlines() == ["[chrombpnet-opt] ACTIVE mode=exact route=k1 kit=0.0.0-stub gpu=H100 det=1 precision=fp32"]
    p = run_cli(kit, ["check", "--mode", "off"])
    assert p.returncode == 0 and p.stdout.splitlines() == ["[chrombpnet-opt] NOT ACTIVE mode=off (stock)"]
    p = run_cli(str(tmp_path / "nokit" / "kit"), ["check", "--mode", "fast"])
    assert p.returncode == 3 and p.stdout.startswith("[chrombpnet-opt] NOT ACTIVE mode=fast reason=kit not found")
    p = run_cli(kit, ["check", "--mode", "fast"], extra_env={"STUB_GPU_CLASS": "H100", "MODEL_OPT_TARGET_GPU": "H200"})
    assert p.returncode == 0                                                        # the target is a note in the report, run.sh gates the card


def test_kit_files_missing_gate(tmp_path):
    """A route-critical kit file absent refuses activation, on both the fast route scope and the full check scope; byte edits to a
    wildcard-matched file (e.g. fastdefault.py) are not gated — byte identity is the checked-out git commit's job, not this package's."""
    kit, _ = mk(tmp_path)
    ho = _stubs.fast_kit(kit)                                                        # the fast mode's kit directory is what an activation checks
    os.remove(os.path.join(ho, "tf", "det_subprocess", "sitecustomize.py"))
    p = run_cli(kit, ["pred_bw", "-op", str(tmp_path / "o" / "p")] + ARGS)            # a route member: the activation's route scope refuses
    assert p.returncode == 3 and "are missing: kit_ho/tf/det_subprocess/sitecustomize.py" in p.stdout
    p = run_cli(kit, ["check", "--mode", "fast"])                                    # the full scope catches the same required file
    assert p.returncode == 3 and "are missing: kit_ho/tf/det_subprocess/sitecustomize.py" in p.stdout


def test_enable_status_idempotent(tmp_path):
    kit, _ = mk(tmp_path)
    code = ("import chrombpnet_opt, json, sys\n"
            "r1 = chrombpnet_opt.enable('fast'); r2 = chrombpnet_opt.enable('fast'); r3 = chrombpnet_opt.enable('off')\n"
            "print(json.dumps([r1 is r2, r1['active'], r1['route'], r3['active'], r3['reason'], chrombpnet_opt.status() is r1]))\n")
    p = subprocess.run([sys.executable, "-c", code], env=_stubs.base_env(kit, {"STUB_GPU_CLASS": "H100"}), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.returncode == 0, p.stderr
    out = p.stdout.splitlines()
    assert out[0] == "[chrombpnet-opt] ACTIVE mode=fast route=k1 kit=0.0.0-stub gpu=H100 det=0 precision=tf32"
    same, active, route, active3, reason3, is_status = json.loads(out[-1])
    assert same and active and route == "k1" and not active3 and "already activated in mode 'fast'" in reason3 and is_status


def test_check_has_no_selftest(tmp_path):
    """`check` is the dry run only: the kit's self-test driver lives outside the release tree, so `--selftest` is no flag of this command."""
    kit, _ = mk(tmp_path)
    assert run_cli(kit, ["check", "--mode", "fast", "--self" + "test"]).returncode == 2


def test_child_proof_admits_the_package_root_only():
    """The package's OWN import root (<tree>/opt, the dir that makes chrombpnet_opt importable) on the stock child's sys.path is admitted —
    a launcher binding the package in every interpreter through a site .pth (`00_kit_bind.pth`: one `<tree>/opt` line) puts exactly that
    entry there; a kit dir and any other dir under the tree are still contamination (exit 3)."""
    here = os.path.dirname(os.path.abspath(stock_pred_bw.__file__)); root = os.path.dirname(here); tree = os.path.dirname(root)
    kit = os.path.join(root, "kit_ho")
    proof = stock_pred_bw.env_proof(kit, tree, False, environ={}, path=[root, here, "/x"], modules={}, argv=["x"])
    assert proof["ok"] and proof["sys_path_under_tree"] == [] and proof["sys_path_under_kit"] == [] and proof["package_root_on_path"] is True
    kit_dir, other = os.path.join(kit, "tf"), os.path.join(tree, "stock")
    proof = stock_pred_bw.env_proof(kit, tree, False, environ={}, path=[root, kit_dir, other], modules={}, argv=["x"])
    assert not proof["ok"] and proof["sys_path_under_kit"] == [kit_dir] and proof["sys_path_under_tree"] == [other]
    proof = stock_pred_bw.env_proof(kit, tree, False, environ={}, path=[root], modules={"chrombpnet_fastkit": None}, argv=["x"])
    assert not proof["ok"] and proof["kit_modules_loaded"] == ["chrombpnet_fastkit"]   # the admitted path never admits a loaded kit module


def test_stock_model_flags_pass_through(tmp_path):
    """-cmb / -bm are stock's own flags and reach the kit's process unchanged beside -cm (the route stays the -cm model's: K1 on the H100 stand-in);
    a line with only -cmb / -bm and no -cm runs stock's own graph, so the mode line names the TensorFlow route (K1 composes the -cm model only) and the
    run is not a partial activation."""
    kit, _ = mk(tmp_path)
    op = str(tmp_path / "out" / "p")
    both = ["-cm", "m.h5", "-cmb", "nb.h5", "-bm", "b.h5", "-r", "r.bed", "-g", "g.fa", "-c", "c.sizes"]
    p = run_cli(kit, ["pred_bw", "--mode", "fast", "-op", op] + both, extra_env={"STUB_GPU_CLASS": "H100", "CHROMBPNET_OPT_DET": "0"})
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.splitlines()[0] == "[chrombpnet-opt] ACTIVE mode=fast route=k1 kit=0.0.0-stub gpu=H100 det=0 precision=tf32"
    assert json.load(open(op + "_kit_call.json"))["argv"] == ["-op", op] + both                # the stock flags pass through, nothing added or dropped
    op2 = str(tmp_path / "out2" / "p")
    only = ["-bm", "b.h5", "-r", "r.bed", "-g", "g.fa", "-c", "c.sizes"]
    p = run_cli(kit, ["pred_bw", "--mode", "fast", "-op", op2] + only, extra_env={"STUB_GPU_CLASS": "H100", "CHROMBPNET_OPT_DET": "0", "STUB_STAMP_FORWARD": "tf_function"})
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.splitlines()[0] == "[chrombpnet-opt] ACTIVE mode=fast route=tf_function kit=0.0.0-stub gpu=H100 det=0 precision=tf32"
    man = manifest.read(os.path.dirname(op2))
    assert man["route"] == "tf_function" and man["partial"] == [] and man["levers_applied"] == ["forward_route"]
    assert stack.has_cm_model(["-cm", "x"]) and stack.has_cm_model(["--chrombpnet-model=x"]) and not stack.has_cm_model(["-cmb", "x", "--chrombpnet-model-nb", "y", "-bm", "z"])
