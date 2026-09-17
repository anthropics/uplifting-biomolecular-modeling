"""The stock caller: on the cases form the override set of `off` equals the base driver's own CLI-arm composition plus the typed hydra
overrides (and the recipe's seed under `--det 1` only); on the hydra form it is the typed overrides verbatim (the weights directory appended
when untyped); the stock process's environment is stripped of every kit name and proves it (the package's checks and the shared core's);
the caller execs upstream's own script."""
import json
import os
import re
import subprocess
import sys
import types

from rfdiffusion1_opt import design, det, modes, stack, stock_cli
from rfdiffusion1_opt.registry import KIT_BASE
from rfdiffusion1_opt.tests._stubs import good_box

CASE = {"name": "pub_ins_195", "pdb": "/x/insulin_target.pdb", "contigs": "[A1-115/0 80-80]", "hotspots": "[A59,A83,A91]", "num_designs": 5, "startnum": 5}
HYDRA = ["inference.input_pdb=/x/insulin_target.pdb", "contigmap.contigs=[A1-115/0 80-80]", "ppi.hotspot_res=[A59,A83,A91]", "inference.output_prefix=/run/samples/binder",
         "inference.num_designs=5", "inference.design_startnum=5"]                     # one run_inference.py invocation's own arguments


def _kv(ov):
    return {o.split("=", 1)[0]: o.split("=", 1)[1] for o in ov}


def test_overrides_are_the_drivers_composition_plus_typed_knobs():
    src = open(os.path.join(stack.kit_dir(KIT_BASE), "drivers", "rfd_bench.py"), encoding="utf-8").read().splitlines()
    conf = "\n".join(src[99:103])                                             # rfd_bench.py:100-103, the driver's own per-case composition (build_conf)
    assert conf.lstrip().startswith("ov = [") and conf.rstrip().endswith('"inference.cautious=False"]'), conf
    keys = set(re.findall(r"(inference\.[a-z_]+|contigmap\.contigs|ppi\.hotspot_res)=", conf)) | {"inference.design_startnum"}   # the resident loop numbers the designs from the case's startnum itself; run_inference.py takes it as this key
    ov = design.stock_overrides(CASE, "/out", "/w", modes.resolve("off"))
    got = _kv(ov)
    composed = {"inference.write_trajectory", "inference.cautious", "inference.deterministic"}   # the driver composes these three keys explicitly and driver_run lays the launch's values over them (modes.DRIVER_FIXED); the stock arm types them only when the caller did
    assert composed <= keys and keys - composed == set(got), (keys - composed, set(got))
    got_c = set(_kv(design.stock_overrides(CASE, "/out", "/w", modes.resolve("off", overrides=["inference.write_trajectory=False", "inference.cautious=False"]), det_flag=True)))
    assert keys <= got_c
    assert "inference.deterministic" not in got and got["inference.design_startnum"] == "5" and got["inference.num_designs"] == "5"   # --det 0 (the default): upstream's unseeded default, nothing appended
    assert got["inference.output_prefix"] == "/out/pub_ins_195/des" and got["inference.model_directory_path"] == "/w"
    assert got["contigmap.contigs"] == "[A1-115/0 80-80]" and got["ppi.hotspot_res"] == "[A59,A83,A91]"
    assert "inference.write_trajectory" not in got and "inference.cautious" not in got     # nothing typed = upstream defaults
    assert ov == ["inference.input_pdb=/x/insulin_target.pdb", "inference.output_prefix=/out/pub_ins_195/des", "inference.model_directory_path=/w", "inference.num_designs=5",
                  "inference.design_startnum=5", "contigmap.contigs=[A1-115/0 80-80]", "ppi.hotspot_res=[A59,A83,A91]"]   # the whole list, in the driver arm's order
    ov2 = design.stock_overrides(CASE, "/out", "/w", modes.resolve("off", overrides=["inference.write_trajectory=False", "inference.cautious=False", "inference.final_step=5"]))
    assert ov2[len(ov):] == ["inference.write_trajectory=False", "inference.cautious=False", "inference.final_step=5"]   # the typed overrides verbatim, after the case's own, in the typed order (hydra: the last value of a key wins)
    ov4 = design.stock_overrides(dict(CASE, ckpt="Base_ckpt.pt"), "/out", "/w", modes.resolve("off"))
    assert ov4[-1] == "inference.ckpt_override_path=/w/Base_ckpt.pt"
    nohot = design.stock_overrides({k: v for k, v in CASE.items() if k != "hotspots"} | {"hotspots": design.NULL}, "/out", "/w", modes.resolve("off"))
    assert "ppi.hotspot_res=null" in nohot                                    # a case without hotspots composes upstream's own default word (base.yaml: hotspot_res: null)


def test_hydra_form_overrides_are_the_typed_ones_verbatim():
    """The hydra form (a case carrying its own `prefix`, design.case_from_overrides): the stock arm gets exactly what was typed, plus
    `inference.model_directory_path=<WEIGHTS>` when the caller did not type it, plus the seed under `--det 1` — nothing recomposed."""
    case = design.case_from_overrides(HYDRA, cwd="/run")
    assert case["prefix"] == "/run/samples/binder" and case["name"] == "binder" and (case["num_designs"], case["startnum"]) == (5, 5) and case["hotspots"] == "[A59,A83,A91]"
    res = modes.resolve("off", overrides=HYDRA)
    assert design.stock_overrides(case, "/run", "/w", res) == HYDRA + ["inference.model_directory_path=/w"]
    assert design.stock_overrides(case, "/run", "", res) == HYDRA                                             # no weights directory known: nothing to append (activation refuses that box before this)
    typed_w = HYDRA + ["inference.model_directory_path=/models/rfd"]
    assert design.stock_overrides(design.case_from_overrides(typed_w, cwd="/run"), "/run", "/w", modes.resolve("off", overrides=typed_w)) == typed_w   # typed: stands
    assert design.stock_overrides(case, "/run", "/w", res, det_flag=True) == HYDRA + ["inference.model_directory_path=/w", det.SEED_OVERRIDE]
    seeded = HYDRA + ["inference.deterministic=False"]
    assert design.stock_overrides(design.case_from_overrides(seeded, cwd="/run"), "/run", "/w", modes.resolve("off", overrides=seeded), det_flag=True) == seeded + ["inference.model_directory_path=/w"]   # a typed value of the seed key wins over --det
    odd = HYDRA + ["--config-name", "symmetry", "+inference.extra=1", "potentials.guide_scale=2"]           # Hydra flags, appending overrides, any key: upstream's to parse — passed as typed
    assert design.stock_overrides(design.case_from_overrides(odd, cwd="/run"), "/run", "/w", modes.resolve("off", overrides=odd))[:len(odd)] == odd
    rel = design.case_from_overrides(HYDRA[:3], cwd="/run/here")                                             # no prefix typed: upstream's default samples/design under the working directory, 10 designs from 0
    assert rel["prefix"] == "/run/here/samples/design" and rel["name"] == "design" and (rel["num_designs"], rel["startnum"]) == (10, 0)


def test_stock_environment_is_stripped(monkeypatch):
    good_box(monkeypatch)
    caller = {"PATH": "/usr/bin", "RFD_ROOT": "/opt/rfd", "WEIGHTS": "/w", "RFD_PREP": "0", "RFD_TRITON_LN": "1", "RFDIFFUSION1_OPT": "exact",
              "RFDIFFUSION1_OPT_HOME": "/tree", "NVIDIA_TF32_OVERRIDE": "1", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
              "ALLOW_ANY_GPU": "1", "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps", "MKL_CBWR": "AVX512", "OPENBLAS_CORETYPE": "Haswell", "ATEN_CPU_CAPABILITY": "avx2",
              "PYTHONPATH": stack.kit_dir("fast_inference") + os.pathsep + "/keep/me"}
    env, stripped, absent = design.stock_environment(environ=caller)
    assert stripped == sorted(["RFD_PREP", "RFD_TRITON_LN", "RFDIFFUSION1_OPT", "RFDIFFUSION1_OPT_HOME", "ALLOW_ANY_GPU"])   # the kit namespace only
    assert env["RFD_ROOT"] == "/opt/rfd" and env["WEIGHTS"] == "/w" and env["DGLBACKEND"] == "pytorch" and env["PYTHONPATH"] == "/keep/me"
    assert (env["NVIDIA_TF32_OVERRIDE"], env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"], env["CUDA_MPS_PIPE_DIRECTORY"]) == ("1", "1", "/tmp/mps")   # torch's, CUDA's and the caller's environment reach the stock process as set (the caller records them)
    assert (env["MKL_CBWR"], env["OPENBLAS_CORETYPE"], env["ATEN_CPU_CAPABILITY"]) == ("AVX512", "Haswell", "avx2")   # the caller's BLAS / ISA words are the caller's on both arms: this package pins none (det.py (3))
    assert absent == stack.pins()["stock_environment"]["must_be_absent_prefixes"]
    assert "recipe_variables" not in stack.pins()["stock_environment"] and set(stack.pins()["stock_environment"]) == {"reads", "must_be_absent_prefixes", "allowed_exceptions", "note"}
    import inspect
    assert list(inspect.signature(design.stock_environment).parameters) == ["environ"]      # no recipe switch: one environment for the stock arm
    env2, _, _ = design.stock_environment(environ={"PATH": "/usr/bin", "DGLBACKEND": "mxnet"})
    assert env2["DGLBACKEND"] == "mxnet"                                                    # setdefault: a caller's own backend word stands (upstream reads it; the package does not police it)
    env3, stripped3, _ = design.stock_environment(environ={"PYTHONPATH": stack.kit_dir("fast_inference") + "/drivers"})
    assert "PYTHONPATH" not in env3 and stripped3 == []                                     # a PYTHONPATH made only of kit directories is removed whole


def test_stock_command_shape(monkeypatch):
    good_box(monkeypatch)
    cmd = design.stock_command(CASE, "/out", "/opt/rfd", "/w", modes.resolve("off"), python="python")
    assert cmd[:4] == ["python", "-s", "-m", "rfdiffusion1_opt.stock_cli"]
    i = cmd.index("--")
    assert "--proof-json" in cmd and cmd[cmd.index("--proof-json") + 1] == "/out/pub_ins_195/stock_env_proof.json"
    assert cmd[cmd.index("--rfd-root") + 1] == "/opt/rfd" and cmd[i + 1].startswith("inference.input_pdb=")
    assert cmd[i + 1:] == design.stock_overrides(CASE, "/out", "/w", modes.resolve("off"))
    case = design.case_from_overrides(HYDRA, cwd="/run")                                    # the hydra form: the proof record goes to the run directory itself (no <out>/<case>/ directory upstream lacks)
    hcmd = design.stock_command(case, "/run/samples", "/opt/rfd", "/w", modes.resolve("off", overrides=HYDRA), python="python")
    assert hcmd[hcmd.index("--proof-json") + 1] == "/run/samples/stock_env_proof.json" and hcmd[hcmd.index("--") + 1:] == HYDRA + ["inference.model_directory_path=/w"]
    assert design.kit_dir_of(case, "/run/samples") == "/run/samples" and design.kit_dir_of(CASE, "/out") == "/out/pub_ins_195"


def test_det_default_0_is_upstreams_unseeded_line():
    """--det defaults to 0 (det_flag=False): the stock arm's override list carries no seed; --det 1 adds exactly the recipe's seed override,
    right after the case's own composition and before the typed overrides — nothing else moves."""
    res = modes.resolve("off")
    default = design.stock_overrides(CASE, "/out", "/w", res)
    explicit_0 = design.stock_overrides(CASE, "/out", "/w", res, det_flag=False)
    det1 = design.stock_overrides(CASE, "/out", "/w", res, det_flag=True)
    assert default == explicit_0 and det.SEED_OVERRIDE not in default
    assert [o for o in det1 if o not in default] == [det.SEED_OVERRIDE] and [o for o in default if o not in det1] == [] and len(det1) == len(default) + 1
    assert det1.index(det.SEED_OVERRIDE) == 7                                              # after the seven composed target strings
    typed = modes.resolve("off", overrides=["inference.deterministic=True"])
    assert design.stock_overrides(CASE, "/out", "/w", typed, det_flag=True).count(det.SEED_OVERRIDE) == 1   # typed and --det 1: one string, the typed one


def test_det_flag_threads_into_stock_command(monkeypatch):
    good_box(monkeypatch)
    cmd_default = design.stock_command(CASE, "/out", "/opt/rfd", "/w", modes.resolve("off"), python="python")
    cmd_det1 = design.stock_command(CASE, "/out", "/opt/rfd", "/w", modes.resolve("off"), python="python", det_flag=True)
    assert det.SEED_OVERRIDE not in cmd_default and det.SEED_OVERRIDE in cmd_det1
    assert [o for o in cmd_det1 if o not in cmd_default] == [det.SEED_OVERRIDE]


def test_det_stock_overrides_returns_the_seed_under_1_only():
    assert det.stock_overrides() == [] and det.stock_overrides(det=False) == []
    assert det.stock_overrides(det=True) == [det.SEED_OVERRIDE]
    assert det.stock_overrides(det=True, typed=["inference.deterministic=False"]) == []      # the caller typed the key: its value stands


def test_cli_det_flag_default_0_choices_0_1():
    from rfdiffusion1_opt import cli
    ap = cli.build_parser()
    subs = next(a for a in ap._actions if isinstance(a, __import__("argparse")._SubParsersAction)).choices
    det_action = next(a for a in subs["design"]._actions if a.option_strings == ["--det"])
    assert det_action.default == 0 == det.DEFAULT_LEVEL and set(det_action.choices) == {0, 1} == set(det.LEVELS)
    a = ap.parse_args(["design", "inference.input_pdb=/x.pdb"])
    assert a.det == 0
    a1 = ap.parse_args(["design", "inference.input_pdb=/x.pdb", "--det", "1"])
    assert a1.det == 1
    ah = ap.parse_args(["design", "--mode", "off", "inference.input_pdb=/x.pdb", "contigmap.contigs=[10-10]"])   # upstream's form: the overrides positional, no input flag of the kit's
    assert ah.det == 0 and not hasattr(ah, "input") and not hasattr(ah, "out_dir") and ah.overrides == ["inference.input_pdb=/x.pdb", "contigmap.contigs=[10-10]"]
    for verb in ("check", "warm"):
        assert not any(a.option_strings == ["--det"] for a in subs[verb]._actions), verb                          # a switch of the design pass only
    for gone in ("--repro", "--process-per-case", "--family", "--worker-gb", "--headroom-gb", "--composition"):
        assert not any(gone in a.option_strings for sp in subs.values() for a in sp._actions), gone
    assert "serve" not in subs and tuple(subs) == cli.VERBS == ("design", "check", "warm")


def test_env_proof_findings(tmp_path):
    kit = str(tmp_path / "kit")
    os.makedirs(kit)
    ok = stock_cli.env_proof(["RFD_", "RFDIFFUSION1_OPT"], [kit], environ={"PATH": "x", "WEIGHTS": "/w"}, modules={}, path=["/usr/lib"])
    assert ok["ok"] and ok["forbidden_env_present"] == [] and ok["kit_modules_loaded"] == []
    assert ok["core"]["ok"] is True and ok["core"]["violations"] is None and ok["core"]["proof"]["env_absent"] == ["RFD_", "RFDIFFUSION1_OPT"]   # the shared core's clean-process proof rides in the record (the core resolves in this process)
    bad = stock_cli.env_proof(["RFD_", "RFDIFFUSION1_OPT"], [kit], environ={"RFD_PREP": "1", "RFDIFFUSION1_OPT": "exact"}, modules={}, path=[])
    assert not bad["ok"] and bad["forbidden_env_present"] == ["RFDIFFUSION1_OPT", "RFD_PREP"]
    assert bad["core"]["ok"] is False and bad["core"]["violations"] and set(bad["core"]["proof"]["forbidden_present"]) >= {"RFD_PREP"}   # the core names the same variable
    m = types.ModuleType("helper")
    m.__file__ = os.path.join(kit, "drivers", "rfd_bench.py")
    bad2 = stock_cli.env_proof([], [kit], environ={}, modules={"helper": m, "rfd_fastpath": types.ModuleType("rfd_fastpath")}, path=[kit])
    assert bad2["kit_modules_loaded"] == ["helper", "rfd_fastpath"] and bad2["kit_dirs_on_sys_path"] == [kit] and not bad2["ok"]
    bad3 = stock_cli.env_proof([], [], environ={}, modules={"torch": types.ModuleType("torch")}, path=[])
    assert bad3["torch_imported"] and not bad3["ok"]
    allowed = stock_cli.env_proof(["RFD_", "WEIGHTS"], [], environ={"RFD_ROOT": "/r", "WEIGHTS": "/w"}, modules={}, path=[], allowed=["RFD_ROOT", "WEIGHTS"])
    assert allowed["ok"] and allowed["core"]["ok"]                                          # the two data paths are locations, not switches, for the core too
    only_core = stock_cli.env_proof([], [], environ={}, modules={"sitecustomize": types.ModuleType("sitecustomize")}, path=[])
    assert only_core["ok"] is True                                                          # a sitecustomize outside every kit directory is not a kit's


def test_core_proof_stands_alone_without_the_core(monkeypatch):
    """In a process where the shared core does not resolve (the stock caller runs `python -s` off the package directory alone), the core's proof is
    recorded as unavailable and the package's own checks decide."""
    import builtins
    real_import = builtins.__import__

    def no_core(name, *a, **k):
        if name == "opt_core" or name.startswith("opt_core."):
            raise ModuleNotFoundError("No module named 'opt_core'")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_core)
    rec = stock_cli.core_proof(["RFD_"], [], environ={"RFD_PREP": "1"}, modules={}, path=[])
    assert rec["ok"] is True and rec["unavailable"].startswith("ModuleNotFoundError(") and rec["proof"] is None
    full = stock_cli.env_proof(["RFD_"], [], environ={"RFD_PREP": "1"}, modules={}, path=[])
    assert full["ok"] is False and full["forbidden_env_present"] == ["RFD_PREP"] and full["core"]["ok"] is True   # the package's own finding still refuses


def _fake_root(tmp_path):
    root = tmp_path / "rfd"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "run_inference.py").write_text("import sys, os, json\nprint(json.dumps({'argv': sys.argv[1:], 'pid': os.getpid(), 'env_knob': os.environ.get('RFD_PREP'), 'dgl': os.environ.get('DGLBACKEND')}))\n")
    return str(root)


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RFD_", "RFDIFFUSION1"))}
    env["PYTHONPATH"] = os.path.join(stack.tree_root(), "opt")
    env.update(extra)
    return env


def test_stock_cli_process_proves_then_execs(tmp_path):
    root = _fake_root(tmp_path)
    proof = tmp_path / "p.json"
    cmd = [sys.executable, "-s", "-m", "rfdiffusion1_opt.stock_cli", "--proof-json", str(proof), "--env-absent", "RFD_,RFDIFFUSION1_OPT", "--kit-dirs", stack.kit_dir("fast_inference"),
           "--rfd-root", root, "--", "inference.deterministic=True", "contigmap.contigs=[A1-115/0 80-80]", "--config-name", "symmetry"]
    p = subprocess.run(cmd, env=_clean_env(), capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["argv"] == ["inference.deterministic=True", "contigmap.contigs=[A1-115/0 80-80]", "--config-name", "symmetry"] and out["env_knob"] is None   # every token after `--` reaches upstream verbatim (Hydra flags included)
    rec = json.loads(proof.read_text())
    assert rec["ok"] and rec["pid"] == out["pid"] and rec["argv"][1] == "-s" and rec["argv"][2].endswith("scripts/run_inference.py")
    assert rec["core"]["ok"] is True                                                          # off the package directory alone the core is unavailable, or clean when it resolves: ok either way
    assert "proof ok" in p.stderr
    hcmd = design.stock_command(design.case_from_overrides(HYDRA, cwd=str(tmp_path)), str(tmp_path / "rec"), root, "/w", modes.resolve("off", overrides=HYDRA), python=sys.executable)
    env, _, _ = design.stock_environment(environ=_clean_env(RFD_PREP="1", NVIDIA_TF32_OVERRIDE="1"))   # the package's own environment for the stock arm: the kit variable stripped, DGL's word set
    p = subprocess.run(hcmd, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["argv"] == HYDRA + ["inference.model_directory_path=/w"] and out["env_knob"] is None and out["dgl"] == "pytorch"
    assert json.loads((tmp_path / "rec" / "stock_env_proof.json").read_text())["ok"] is True


def test_stock_cli_process_refuses_a_kit_variable(tmp_path):
    root = _fake_root(tmp_path)
    proof = tmp_path / "p.json"
    cmd = [sys.executable, "-s", "-m", "rfdiffusion1_opt.stock_cli", "--proof-json", str(proof), "--env-absent", "RFD_", "--kit-dirs", "", "--rfd-root", root, "--", "x=1"]
    p = subprocess.run(cmd, env=_clean_env(RFD_PREP="1"), capture_output=True, text=True)
    assert p.returncode == 3 and "NOT STOCK" in p.stderr and p.stdout == ""
    assert json.loads(proof.read_text())["forbidden_env_present"] == ["RFD_PREP"]


def test_stock_cli_process_refuses_without_entry_point(tmp_path):
    proof = tmp_path / "p.json"
    p = subprocess.run([sys.executable, "-s", "-m", "rfdiffusion1_opt.stock_cli", "--proof-json", str(proof), "--rfd-root", str(tmp_path), "--"], env=_clean_env(), capture_output=True, text=True)
    assert p.returncode == 3 and "no stock entry point" in p.stderr
