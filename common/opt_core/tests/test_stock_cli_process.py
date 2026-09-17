"""The stock proof (every violation class named, the det carve-out exact, the argv contract), the cli skeleton (usage errors only),
the watchdog runner, the det recipe shape, the home resolution, the jit cache key."""
import argparse
import os
import sys
import textwrap
import time
import types

import pytest

from opt_core import cli, det, home, jit_cache, modes, process, report, stock_proof

# ------------------------------------------------------------------------------------------------------------ stock proof


def _mod(name, file=None):
    m = types.ModuleType(name)
    if file:
        m.__file__ = file
    return m


def test_env_proof_names_every_violation(tmp_path):
    kit = str(tmp_path / "engine" / "opt")
    os.makedirs(os.path.join(kit, "acme_opt"))
    clean = stock_proof.env_proof(env_absent=["ACME_", "OPT_"], kit_dirs=[kit], module_prefixes=["acme_opt"], environ={"PATH": "/bin"},
                                  modules={"os": os, "opt_core": sys.modules["opt_core"], "opt_core.stock_proof": stock_proof}, path=["/usr/lib"], meta_path=[])
    assert clean["ok"] and clean["forbidden_present"] == [] and clean["core_modules_loaded"] == [] and clean["det_exception"] is None
    assert stock_proof.clean_sentence(clean) == f"absent=ACME_,OPT_ kit_modules=none kit_dirs=none no_user_site={bool(sys.flags.no_user_site)}"

    class Finder:
        armed = True
    Finder.__module__ = "acme_opt._autoload"
    dirty = stock_proof.env_proof(
        env_absent=["ACME_"], kit_dirs=[kit], module_prefixes=["acme_opt"], environ={"ACME_OPT": "fast"},
        modules={"acme_opt.stack": _mod("acme_opt.stack"), "other": _mod("other", os.path.join(kit, "x.py")), "sitecustomize": _mod("sitecustomize", os.path.join(kit, "sitecustomize.py")),
                 "torch": _mod("torch"), "opt_core.gates": _mod("opt_core.gates")},
        path=[os.path.join(kit, "acme_opt"), "/usr/lib"], meta_path=[Finder()])
    assert not dirty["ok"]
    assert dirty["forbidden_present"] == ["ACME_OPT"] and dirty["kit_modules_loaded"] == ["acme_opt.stack", "other", "sitecustomize"]
    assert dirty["kit_dirs_on_path"] == [os.path.join(kit, "acme_opt")] and dirty["autoload_armed"] == ["Finder"]
    assert dirty["kit_sitecustomize"] == os.path.join(kit, "sitecustomize.py") and dirty["torch_loaded_before_proof"] is True
    assert dirty["core_modules_loaded"] == ["opt_core.gates"]
    assert stock_proof.violations_sentence(dirty).startswith("forbidden env ['ACME_OPT'], kit modules ['acme_opt.stack', 'other', 'sitecustomize'], kit dirs [")


def test_det_carve_out_is_exact(tmp_path):
    kit = str(tmp_path / "opt")
    det_dir = os.path.join(kit, "det")
    os.makedirs(det_dir)
    exc = {"env": {"ACME_DET": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, "pythonpath": [det_dir]}
    good_env = {"ACME_DET": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    sc = _mod("sitecustomize", os.path.join(det_dir, "sitecustomize.py"))
    p = stock_proof.env_proof(env_absent=["ACME_"], kit_dirs=[kit], module_prefixes=["acme_opt"], environ=good_env,
                              modules={"sitecustomize": sc}, path=[det_dir, "/usr/lib"], meta_path=[], det_exception=exc)
    assert p["ok"] and p["det_exception"]["deviations"] == [] and p["forbidden_present"] == ["ACME_DET"]
    bad = stock_proof.env_proof(env_absent=["ACME_"], kit_dirs=[kit], module_prefixes=["acme_opt"], environ=dict(good_env, ACME_OPT="fast", CUBLAS_WORKSPACE_CONFIG=":16:8"),
                                modules={"sitecustomize": sc, "acme_opt.levers": _mod("acme_opt.levers")}, path=[det_dir, kit], meta_path=[], det_exception=exc)
    dev = bad["det_exception"]["deviations"]
    assert not bad["ok"] and len(dev) == 4
    assert dev[0] == "forbidden names present ['ACME_DET', 'ACME_OPT'] != the recipe's ['ACME_DET']"
    assert dev[1] == "CUBLAS_WORKSPACE_CONFIG=':16:8' != the recipe's ':4096:8'"
    assert dev[2].startswith("kit directories on sys.path") and dev[3] == "kit modules loaded ['acme_opt.levers', 'sitecustomize'] != ['sitecustomize']"
    assert stock_proof.violations_sentence(bad).startswith("det exception not met: forbidden names present")
    after = stock_proof.after_call_check([kit], ["acme_opt"], lever_modules=["acme_opt.levers"], det_exception=exc,
                                         modules={"sitecustomize": sc, "acme_opt.levers": _mod("acme_opt.levers")})
    assert after == {"kit_modules_loaded_after": ["acme_opt.levers", "sitecustomize"], "lever_modules_imported": ["acme_opt.levers"], "after_ok": False}


def test_strip_helpers():
    env, stripped = stock_proof.strip_env({"ACME_OPT": "1", "ACME_HOME": "/x", "PATH": "/bin", "PYTHONPATH": "/k"}, ["ACME_"], names=["PYTHONPATH"])
    assert env == {"PATH": "/bin"} and stripped == ["ACME_HOME", "ACME_OPT", "PYTHONPATH"]
    value, gone = stock_proof.strip_pythonpath("/kit/a:/usr/lib:/kit/b/c", ["/kit"])
    assert value == "/usr/lib" and gone == ["/kit/a", "/kit/b/c"]
    assert stock_proof.strip_pythonpath("/kit/a", ["/kit"]) == (None, ["/kit/a"])
    assert stock_proof.strip_pythonpath(None, ["/kit"]) == (None, [])


def test_stock_command_and_parse_round_trip(tmp_path):
    exc = {"env": {"ACME_DET": "1"}, "pythonpath": ["/kit/det"]}
    cmd = stock_proof.stock_command("/usr/bin/python", "acme_opt.stock_pred", proof_json="/tmp/p.json", env_absent=["ACME_", "OPT_"],
                                    kit_dirs=["/kit", "/kit2"], args=["--in", "x.fasta", "--", "tail"], module_prefixes=["acme_opt"], det=exc)
    assert cmd[:4] == ["/usr/bin/python", "-s", "-m", "acme_opt.stock_pred"] and cmd[-5:] == ["--", "--in", "x.fasta", "--", "tail"]
    a, stock = stock_proof.parse_stock_argv(cmd[4:])
    assert a.proof_json == "/tmp/p.json" and a.env_absent == ["ACME_", "OPT_"] and a.kit_dirs == ["/kit", "/kit2"] and a.module_prefixes == ["acme_opt"]
    assert a.det_exception == exc and stock == ["--in", "x.fasta", "--", "tail"]
    a2, stock2 = stock_proof.parse_stock_argv(["--proof-json", "p", "--env-absent", "A_"])
    assert a2.det_exception is None and a2.kit_dirs == [] and stock2 == []
    path = stock_proof.write_proof(str(tmp_path / "d" / "proof.json"), {"ok": True})
    assert open(path).read() == '{\n "ok": true\n}\n'


def test_core_modules_loaded_allows_only_the_proof_machinery():
    mods = {"opt_core": 1, "opt_core.stock_proof": 1, "opt_core.autoload": 1, "opt_core.gates": 1, "other": 1}
    assert stock_proof.core_modules_loaded(mods) == ["opt_core.autoload", "opt_core.gates"]


# ------------------------------------------------------------------------------------------------------------ cli

TABLE = modes.ModeTable(modes=("off", "exact", "fast"), default="fast")


def _verbs(record):
    def add(p):
        cli.add_mode_arguments(p, TABLE, env="ACME_OPT", variant_env="ACME_VARIANT", variants=("a", "b"), det_levels=(0, 1, 2), environ={})
        p.add_argument("--fail", default=None)

    def run(args):
        record.append((args.verb, cli.resolve_mode(args, environ={"ACME_OPT": "off"}), args.variant, args.allow_partial, args.det))
        if args.fail == "usage":
            raise cli.CliError("no input given")
        if args.fail == "crash":
            raise RuntimeError("model exploded")
        return 7

    return [cli.Verb("pred", "predict", add, run), cli.Verb("check", "check the box", add, run)]


def test_cli_dispatch_and_mode_precedence(capsys):
    rec = []
    rc = cli.main(["pred", "--mode", "Exact", "--variant", "b", "--allow-partial", "--det", "2"], prog="acme-opt", description="d", verbs=_verbs(rec), tag="acme-opt")
    assert rc == 7 and rec == [("pred", "exact", "b", True, 2)]
    rc = cli.main(["check"], prog="acme-opt", description="d", verbs=_verbs(rec), tag="acme-opt")
    assert rc == 7 and rec[-1] == ("check", "off", None, False, 0)                    # the env variable wins over the table default


def test_cli_usage_errors_exit_2_and_everything_else_propagates(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["pred", "--mode", "turbo"], prog="acme-opt", description="d", verbs=_verbs([]), tag="acme-opt")
    assert e.value.code == 2 and "unknown mode 'turbo'" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        cli.main([], prog="acme-opt", description="d", verbs=_verbs([]), tag="acme-opt")
    assert e.value.code == 2 and "required: pred|check" in capsys.readouterr().err
    rc = cli.main(["pred", "--fail", "usage"], prog="acme-opt", description="d", verbs=_verbs([]), tag="acme-opt")
    assert rc == report.EXIT_USAGE and capsys.readouterr().err == "usage: acme-opt pred ...\n[acme-opt] pred: no input given\n"
    with pytest.raises(RuntimeError, match="model exploded"):
        cli.main(["pred", "--fail", "crash"], prog="acme-opt", description="d", verbs=_verbs([]), tag="acme-opt")

    class VerifyError(Exception):
        pass

    def run(args):
        raise VerifyError("selftest failed: 3 of 4")
    rc = cli.main(["verify"], prog="acme-opt", description="d", verbs=[cli.Verb("verify", "v", lambda p: None, run)], tag="acme-opt", usage_errors=(VerifyError,))
    assert rc == 2 and capsys.readouterr().err.endswith("[acme-opt] verify: selftest failed: 3 of 4\n")


# ------------------------------------------------------------------------------------------------------------ process


def test_run_logged_kills_the_process_group_on_the_clock(tmp_path):
    script = tmp_path / "hang.py"
    script.write_text(textwrap.dedent("""
        import subprocess, sys, time
        print("started", flush=True)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open(sys.argv[1], "w").write(str(child.pid))
        time.sleep(60)
    """))
    pidfile = tmp_path / "child.pid"
    lines = []
    t0 = time.monotonic()
    r = process.run_logged([sys.executable, str(script), str(pidfile)], timeout_s=1.5, on_line=lines.append, log_path=str(tmp_path / "run.log"), what="selftest")
    assert r.timed_out and not r.ok and r.error == "SelftestTimeout" and r.rc != 0
    assert r.reason.startswith("selftest exceeded --timeout 1.5 s: process group killed at ") and time.monotonic() - t0 < 10
    assert lines == ["started\n"] and open(tmp_path / "run.log").read() == "started\n"
    child = int(pidfile.read_text())
    for _ in range(50):                                                              # the grandchild died with the group
        try:
            os.kill(child, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            break
    else:
        raise AssertionError("grandchild survived the group kill")


def test_run_logged_normal_completion_and_child_env(tmp_path):
    r = process.run_logged([sys.executable, "-c", "import sys; print('a'); print('b', file=sys.stderr); sys.exit(4)"], timeout_s=30)
    assert (r.rc, r.timed_out, r.error) == (4, False, None) and r.wall_s > 0
    env = process.child_env({"ACME_OPT": "fast", "ACME_HOME": "/x", "PATH": "/bin", "PYTHONPATH": "/p"}, strip_prefixes=["ACME_"], strip_names=["PYTHONPATH"], export={"N": 1})
    assert env == {"PATH": "/bin", "N": "1"}


# ------------------------------------------------------------------------------------------------------------ det


def test_det_recipe_apply_describe_exception():
    r = det.Recipe(level=2, env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "ACME_DET": 1}, unset=("ACME_FAST",), pythonpath=("/kit/det",), note="cuBLAS deterministic")
    env = {"ACME_FAST": "1", "PYTHONPATH": "/home/u:/kit/det"}
    before = det.apply_env(r, env)
    assert env == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "ACME_DET": "1", "PYTHONPATH": "/kit/det:/home/u"}
    assert before == {"ACME_FAST": "1", "CUBLAS_WORKSPACE_CONFIG": None, "ACME_DET": None, "PYTHONPATH": "/home/u:/kit/det"}
    assert det.describe(r) == "det=2 env=CUBLAS_WORKSPACE_CONFIG,ACME_DET unset=ACME_FAST pythonpath=1 (cuBLAS deterministic)"
    assert det.stock_exception(r) == {"env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "ACME_DET": "1"}, "pythonpath": ["/kit/det"]}
    assert det.stock_exception(det.PRODUCTION) is None and det.describe(det.PRODUCTION) == "det=0 env=none unset=none pythonpath=0 (production numerics)"


# ------------------------------------------------------------------------------------------------------------ home


def test_tree_home_precedence_and_refusal(tmp_path):
    pkg_file = tmp_path / "engine" / "opt" / "acme_opt" / "__init__.py"
    pkg_file.parent.mkdir(parents=True)
    pkg_file.write_text("")
    (tmp_path / "engine" / "stock").mkdir()
    assert home.tree_home(str(pkg_file), environ={}) == str(tmp_path / "engine")
    assert home.tree_home(str(pkg_file), environ={"MODEL_OPT": "/elsewhere"}) == "/elsewhere"
    assert home.tree_home(str(pkg_file), env_home="ACME_HOME", environ={"MODEL_OPT": "/elsewhere", "ACME_HOME": "/mine"}) == "/mine"
    assert home.opt_home(str(pkg_file), environ={}) == str(tmp_path / "engine" / "opt")
    assert home.tree_home(str(pkg_file), require=("opt", "stock"), environ={}) == str(tmp_path / "engine")
    with pytest.raises(home.HomeError) as e:
        home.tree_home(str(pkg_file), env_home="ACME_HOME", require=("opt", "stock", "configs"), environ={})
    assert str(e.value) == f"engine directory {tmp_path / 'engine'} (2 levels above the package) lacks configs: set ACME_HOME or MODEL_OPT to the engine directory"


def test_place_on_sys_path_is_idempotent():
    path = ["/site", "/usr/lib", "/kit/a"]
    assert home.place_on_sys_path(["/kit/a", "/kit/b"], path=path) == ["/kit/a", "/kit/b", "/site", "/usr/lib"]
    assert home.place_on_sys_path(["/kit/b"], after="/site", path=path) == ["/kit/a", "/site", "/kit/b", "/usr/lib"]


# ------------------------------------------------------------------------------------------------------------ jit cache


def test_jit_cache_key_and_dirs(tmp_path, monkeypatch):
    assert jit_cache.key("2.4.1+cu124", cc="9.0") == "torch2.4.1-cu124-sm90"
    assert jit_cache.key("2.4.1", cuda="12.4", cc="9.0") == "torch2.4.1-cu124-sm90"
    assert jit_cache.key("0.4.30", cuda="12", cc="9.0", dist="jaxlib") == "jaxlib0.4.30-cu12-sm90"
    monkeypatch.setenv("PATH", "")
    with pytest.raises(jit_cache.StackKeyUnknown):                 # no nvidia-smi on PATH: the cc part is unknown -> named, never silent
        jit_cache.key("2.4.1+cu124")
    assert jit_cache.key("2.4.1+cu124", strict=False) == "torch2.4.1-cu124-smunknown"
    d = jit_cache.cache_dirs("/cache", "torch2.4.1-cu124-sm90")
    assert d == {"TRITON_CACHE_DIR": "/cache/torch2.4.1-cu124-sm90/triton", "TORCH_EXTENSIONS_DIR": "/cache/torch2.4.1-cu124-sm90/torch_extensions", "TORCHINDUCTOR_CACHE_DIR": "/cache/torch2.4.1-cu124-sm90/inductor"}
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    assert jit_cache.keep_or_key(str(mounted / "triton"), "/cache/k/triton") == (str(mounted / "triton"), "kept")
    assert jit_cache.keep_or_key("/no/such/root/triton", "/cache/k/triton") == ("/cache/k/triton", "keyed")
    assert jit_cache.keep_or_key(None, "/cache/k/triton") == ("/cache/k/triton", "keyed")


def test_cli_usage_text_is_the_kits_through_on_usage_error(capsys):
    """A kit whose usage line differs from the core's two lines prints its own bytes and returns its own code."""
    seen = []

    def own(args, exc):
        seen.append((args.verb, str(exc)))
        print(f"[acme-opt] usage: pred needs an input ({exc})", file=sys.stderr)
        return 64
    rc = cli.main(["pred", "--fail", "usage"], prog="acme-opt", description="d", verbs=_verbs([]), tag="acme-opt", on_usage_error=own)
    assert rc == 64 and seen == [("pred", "no input given")]
    assert capsys.readouterr().err == "[acme-opt] usage: pred needs an input (no input given)\n"
    with pytest.raises(SystemExit) as e:                                             # a parse-time --mode error stays argparse's (exit 2)
        cli.main(["pred", "--mode", "turbo"], prog="acme-opt", description="d", verbs=_verbs([]), tag="acme-opt", on_usage_error=own)
    assert e.value.code == 2 and seen == [("pred", "no input given")]


def test_kill_group_falls_back_to_the_child_when_the_group_cannot_be_signalled(monkeypatch):
    import signal
    killed = []

    class P:
        pid = 4242

        def kill(self):
            killed.append("child")

    def killpg(pgid, sig):
        assert (pgid, sig) == (4242, signal.SIGKILL)
        raise PermissionError("not our group")
    monkeypatch.setattr(os, "killpg", killpg)
    process._kill_group(P())
    assert killed == ["child"]
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(ProcessLookupError()))
    killed.clear()
    process._kill_group(P())
    assert killed == []


def test_file_less_modules_and_non_directory_path_entries_are_never_the_kits(tmp_path, monkeypatch):
    """cwd inside a kit directory plus an editable-finder hook string on sys.path: builtins (no __file__) and hook keys are not kit evidence."""
    import types
    from opt_core import stock_proof
    kit = tmp_path / "acme" / "opt"; kit.mkdir(parents=True)
    monkeypatch.chdir(kit)
    mods = {"sys": sys, "builtins": __import__("builtins"), "nofile": types.ModuleType("nofile"),
            "inkit": types.ModuleType("inkit"), "elsewhere": types.ModuleType("elsewhere")}
    mods["inkit"].__file__ = str(kit / "inkit.py"); mods["elsewhere"].__file__ = str(tmp_path / "elsewhere.py"); mods["nofile"].__file__ = None
    assert stock_proof.kit_modules_loaded([str(kit)], (), modules=mods) == ["inkit"]
    path = ["", "__editable__.acme_opt-0.0.1.finder.__path_hook__", str(kit), str(tmp_path / "missing")]
    assert sorted(p for p in path if p and os.path.exists(p) and stock_proof._under(p, stock_proof._roots([str(kit)]))) == [str(kit)]


def test_kit_module_prefix_is_a_dotted_boundary_not_a_bare_substring():
    """A declared prefix ``bench`` names the module ``bench`` and any of its submodules (``bench.anything``) -- never an unrelated
    module that merely starts with the same characters (``bench_peak_meter``, the launcher's own instrumentation, was being
    counted as a kit's own the moment that kit's own module prefix happened to be ``bench``)."""
    mods = {"bench": _mod("bench"), "bench.kit": _mod("bench.kit"), "bench_peak_meter": _mod("bench_peak_meter"), "other": _mod("other")}
    assert stock_proof.kit_modules_loaded([], ["bench"], modules=mods) == ["bench", "bench.kit"]


def test_armed_finders_ignores_the_harness_instrumentation_by_marker_or_by_module():
    """A finder marked ``__harness__ = True`` on its class, or living in ``opt_core.stock_proof.HARNESS_MODULES`` (``bench_peak_meter``),
    is never reported as an armed autoload finder -- the launcher's own measurement instrumentation is present identically in every
    arm including stock, and is not a kit's, however its own class module happens to be named."""
    class HarnessTimedFinder:
        armed = True
    HarnessTimedFinder.__module__ = "bench_peak_meter"                    # matched by module name, no marker needed

    class HarnessPeakFinder:
        armed = True
        __harness__ = True
    HarnessPeakFinder.__module__ = "some.other.module"                    # matched by the explicit marker, regardless of module name

    class KitAutoloadFinder:
        armed = True
    KitAutoloadFinder.__module__ = "acme_opt._autoload"

    assert stock_proof.armed_finders([HarnessTimedFinder(), HarnessPeakFinder(), KitAutoloadFinder()]) == ["KitAutoloadFinder"]
    assert stock_proof.armed_finders([HarnessTimedFinder(), HarnessPeakFinder()]) == []           # both launcher finders, alone: no kit evidence at all
