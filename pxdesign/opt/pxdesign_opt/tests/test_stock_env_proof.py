"""The stock caller proves its environment against the one list (stock/PINS.json must_be_absent_prefixes) before importing torch."""
import json
import os
import subprocess
import sys
import types

import pytest

import pxdesign_opt
from pxdesign_opt import options, stock_infer

CORE = pxdesign_opt.core_gate()["installed"]["root"]            # the pinned core = the installed one (the core pin gate's facts): ../../common/opt_core from opt/ in the tree
KIT_MODULES = ("pxd_xattempt", "pxd_xattempt.hoist", "pxdesign_opt.stack", "opt_core.gates")   # modules a stock process never holds (the lever kit's, the activation module, a core module beyond the proof)
CLEAN = {"env": [], "modules": [], "sys_path": [], "autoload_finder": False, "torch_loaded": False, "kit_sitecustomize": None, "det_exception": []}


@pytest.fixture
def pins(tree):
    return options.read_pins(os.path.join(tree, "stock", "PINS.json"))


def _clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("PXD_", "PXDESIGN_", "TORCH_ALLOW_TF32", "CUDA_MPS_", "CUBLAS_WORKSPACE_CONFIG")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    for m in list(sys.modules):                                   # what a fresh stock interpreter never holds: the lever kit's modules, torch, the core beyond its proof machinery
        if m in KIT_MODULES or m == "torch" or m.startswith(("torch.", "pxd_xattempt")) or (m.startswith("opt_core.") and m != "opt_core.stock_proof"):
            monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if "opt/forward" not in p and "opt/serving" not in p])
    monkeypatch.setattr(sys, "meta_path", [f for f in sys.meta_path if type(f).__module__ != "opt_core.autoload"])


def test_one_list_is_pins(pins):
    assert pins["stock_environment"]["must_be_absent_prefixes"] == ["PXD_", "PXDESIGN_HOIST", "PXDESIGN_OPT", "CUDA_MPS_"]   # the kit's and its sibling kits' variables only: torch's own (TORCH_ALLOW_TF32_*, NVIDIA_TF32_OVERRIDE) pass through to stock
    assert pins["stock_environment"]["allowed_exceptions"] == ["CUBLAS_WORKSPACE_CONFIG"]


def test_clean_passes(monkeypatch, pins, tree):
    _clean(monkeypatch)
    import pxdesign_opt._autoload  # noqa: F401  (imported by the .pth in every process; its finder is what must be absent)
    p = stock_infer.env_proof(tree, pins, 0)
    assert p["clean"] and p["violations"] == CLEAN


@pytest.mark.parametrize("var", ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE"])
def test_torch_tf32_variables_pass_through_to_stock(monkeypatch, pins, tree, var):
    """torch's own TF32 variables are the caller's (a pass environment's) to set: the design verb hands them to the stock child unchanged
    (`_clean_env`) and the child's proof is clean with them present — recorded under `env`, reported by the KERNELS line, never refused."""
    from pxdesign_opt import cli
    _clean(monkeypatch)
    monkeypatch.setenv(var, "1")
    assert cli._clean_env(pins, det=0)[var] == "1"
    p = stock_infer.env_proof(tree, pins, 0)
    assert p["clean"] and p["violations"] == CLEAN and p["env"][var] == "1"


@pytest.mark.parametrize("var", ["PXD_HOIST", "PXD_HOIST_MODE", "PXDESIGN_HOIST", "PXDESIGN_OPT", "CUDA_MPS_PIPE_DIRECTORY"])
def test_forbidden_variable_refused(monkeypatch, pins, tree, var, tmp_path):
    _clean(monkeypatch)
    monkeypatch.setenv(var, "1")
    pj = str(tmp_path / "stock_env_proof.json")
    with pytest.raises(SystemExit) as e:
        stock_infer.env_proof(tree, pins, 0, proof_path=pj)
    assert e.value.code == 3
    p = json.load(open(pj))
    assert p["clean"] is False and p["exit_code"] == 3 and var in p["violations"]["env"]


def test_cublas_only_under_det(monkeypatch, pins, tree):
    _clean(monkeypatch)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(SystemExit):
        stock_infer.env_proof(tree, pins, 0)
    assert stock_infer.env_proof(tree, pins, 1)["violations"]["env"] == []
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")                  # under --det 1 the variable must hold the recipe's value (the core's det exception)
    with pytest.raises(SystemExit):
        stock_infer.env_proof(tree, pins, 1)


def test_kit_module_or_path_refused(monkeypatch, pins, tree):
    _clean(monkeypatch)
    monkeypatch.setitem(sys.modules, "pxd_xattempt", types.ModuleType("pxd_xattempt"))
    with pytest.raises(SystemExit):
        stock_infer.env_proof(tree, pins, 0)
    monkeypatch.delitem(sys.modules, "pxd_xattempt")
    monkeypatch.setattr(sys, "path", sys.path + [os.path.join(os.path.abspath(tree), "opt", "forward", "hoist")])
    with pytest.raises(SystemExit):
        stock_infer.env_proof(tree, pins, 0)


def test_layernorm_type_is_recorded_never_required(monkeypatch, pins, tree):
    """LAYERNORM_TYPE is stock's own variable: the proof records what the caller's environment holds (`layernorm_env`: the value, or `absent`
    for none / empty) and is clean either way — the child inherits it as is (configs/<gpu>.env exports fast_layernorm; absent or any other
    value selects OpenFold's LayerNorm exactly as for a user of the bare console script). No stock-caller flag names it."""
    _clean(monkeypatch)
    for value, recorded in (("fast_layernorm", "fast_layernorm"), ("torch", "torch"), ("", "absent"), (None, "absent")):
        if value is None:
            monkeypatch.delenv("LAYERNORM_TYPE", raising=False)
        else:
            monkeypatch.setenv("LAYERNORM_TYPE", value)
        p = stock_infer.env_proof(tree, pins, 0)
        assert p["clean"] and p["violations"] == CLEAN and p["layernorm_env"] == recorded and "layernorm_env_expected" not in p, (value, p.get("violations"))
    flags = {a.option_strings[0] for a in stock_infer._parser()._actions if a.option_strings}
    assert not any("layernorm" in f for f in flags) and not hasattr(stock_infer, "REQUIRED_ENV") and not hasattr(options, "LAYERNORM_ENVS")
    kind, cls, env = __import__("pxdesign_opt.stamps", fromlist=["x"]).layernorm_of_env({"LAYERNORM_TYPE": ""})
    assert (kind, env) == ("plain", None)                                                   # an empty value selects OpenFold's like an absent one and prints as absent


def test_stock_command_and_child_environment(pins):
    """The stock caller's command line and the console-script child's argv / environment: the default form is byte-identical to the one
    composition (no new word); the console-script child (--det 0) is the bare script with this process's environment."""
    from pxdesign_opt import cli, det, infer_loop
    base = ["/py", "-s", "-m", "pxdesign_opt.stock_infer", "--tree", "/t", "--tasks", "t.json", "--out_dir", "o", "--det", "0", "--seeds", "101,102"]
    opts = options.resolve(seeds="101,102", pins=pins)
    cmd = cli.stock_command("/t", "t.json", "o", opts=opts, det=0, ckpt_dir="/ckpt", extra=["--sample_diffusion_chunk_size", "5"], python="/py")
    assert cmd == base + ["--ckpt_dir", "/ckpt", "--extra", "--sample_diffusion_chunk_size", "5"]
    assert cli.stock_command("/t", "t.json", "o", opts=opts, det=1, python="/py") == base[:11] + ["1"] + base[12:]   # --det 1: the in-process route inside the child; the word is the only difference
    with pytest.raises(TypeError):
        cli.stock_command("/t", "t.json", "o", opts=opts, det=0, route="cli")                                 # no route / guard selectors: the route follows --det
    given = cli.stock_command("/t", "t.json", "o", opts=options.resolve(N_sample=2, num_workers=0, use_fast_ln="true", seeds="101,102", pins=pins), det=0, python="/py")
    assert given == base + ["--N_sample", "2", "--num_workers", "0", "--use_fast_ln", "true"]                   # --seeds, then the caller's given knob words verbatim in OPTION_ORDER
    assert cli.stock_command("/t", "t.json", "o", opts=options.resolve(pins=pins), det=0, python="/py") == base[:-2]   # no --seeds given: none passed (upstream derives one)
    # the design verb's child environment: the kit's variables stripped, everything else — LAYERNORM_TYPE included — as the caller has it
    environ = {"PATH": "/bin", "LAYERNORM_TYPE": "fast_layernorm", "PXDESIGN_OPT": "exact", "PXD_HOIST": "1", "HOME": "/h"}
    assert cli._clean_env(pins, det=0, environ=environ) == {"PATH": "/bin", "LAYERNORM_TYPE": "fast_layernorm", "HOME": "/h"}
    assert cli._clean_env(pins, det=0, environ={k: v for k, v in environ.items() if k != "LAYERNORM_TYPE"}) == {"PATH": "/bin", "HOME": "/h"}   # absent stays absent (the bare user's OpenFold LayerNorm)
    assert cli._clean_env(pins, det=0, environ=dict(environ, LAYERNORM_TYPE="")) == {"PATH": "/bin", "LAYERNORM_TYPE": "", "HOME": "/h"}          # a set-empty value passes as is
    # the console-script child (stock_infer): --det 0 = the bare console script and this process's environment, untouched
    argv = infer_loop.upstream_argv("t.json", "o", "/ckpt", opts.argv(), opts.seeds_arg(), ["--sample_diffusion_chunk_size", "5"])
    assert argv == ["-i", "t.json", "-o", "o", *options.resolve(pins=pins).argv(), "--seeds", "101,102", "--load_checkpoint_dir", "/ckpt", "--sample_diffusion_chunk_size", "5"]
    env0 = {"PATH": "/bin", "LAYERNORM_TYPE": "fast_layernorm"}
    assert stock_infer.cli_child_command(argv, exe="/usr/bin/pxdesign") == ["/usr/bin/pxdesign", "infer", *argv]
    assert stock_infer.cli_child_env(env0) == env0 and stock_infer.cli_child_env(env0) is not env0
    assert stock_infer.ROUTES == {0: "cli", 1: "inprocess"} and det.RECIPE_ENV == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}   # the route follows --det; the recipe's variable reaches the in-process route through det.apply_env
    assert not hasattr(stock_infer, "CLI_DET_PREAMBLE")                                                        # no program text is put in front of upstream's console script on any route


def test_det1_route_accepted_and_no_route_selector(monkeypatch, tree, tmp_path, capsys):
    """`--det 1` is the in-process route: the caller gets past argument handling and refuses at the step named for the box (preflight: the
    checkpoint absent), never with a usage error. The stock caller's words are exactly --tree/--tasks/--out_dir/--ckpt_dir/--det/--proof-json/
    --extra plus upstream's knobs and --seeds: no route, control-pass or environment selector."""
    _clean(monkeypatch)
    flags = sorted(a.option_strings[-1] for a in stock_infer._parser()._actions if a.option_strings and a.option_strings[-1] != "--help")
    assert flags == sorted(["--tree", "--tasks", "--out_dir", "--ckpt_dir", "--det", "--proof-json", "--extra", "--seeds"] + [f"--{k}" for k in options.OPTION_ORDER]), flags
    a = stock_infer.parse_args(["--tree", tree, "--tasks", str(tmp_path / "t.json"), "--out_dir", str(tmp_path / "o"), "--det", "1", "--ckpt_dir", str(tmp_path / "none")])
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        rc = stock_infer.run(a)
    except SystemExit as e:
        rc = e.code
    assert rc != 2 and "refused: the upstream CLI" not in capsys.readouterr().err


# The two checks below run the stock caller's own interpreter form — a fresh `python -s` in the design verb's stock environment — because
# the in-process tests above cannot see the defect class they lock: a module the stock caller imports pulling in a kit module at import
# time (the test session holds `pxdesign_opt.stack` already, and `_clean` drops it from sys.modules).

def _stock_process_env(tree, pins):
    from pxdesign_opt import cli
    env = cli._clean_env(pins)                                   # the design verb's stock subprocess environment (mode off)
    env["LAYERNORM_TYPE"] = "fast_layernorm"                     # the card's export, inherited by the stock child as is
    env.update(PYTHONPATH=os.pathsep.join([CORE, os.path.join(tree, "opt")]), PYTHONDONTWRITEBYTECODE="1")   # the box's two editable installs: the core and this kit
    return env


def test_stock_caller_process_is_clean(tree, pins):
    code = (
        "import json, sys\n"
        "import pxdesign_opt._autoload\n"                              # the .pth's import, in every process of an installed package
        "from pxdesign_opt import options, stock_infer\n"
        f"loaded = [m for m in {list(KIT_MODULES)!r} if m in sys.modules]\n"
f"proof = stock_infer.env_proof({tree!r}, options.read_pins({os.path.join(tree, 'stock', 'PINS.json')!r}), 0)\n"
        "print(json.dumps({'loaded': loaded, 'clean': proof['clean'], 'violations': proof['violations']}))\n"
    )
    r = subprocess.run([sys.executable, "-s", "-c", code], env=_stock_process_env(tree, pins), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["loaded"] == []
    assert out["clean"] is True and out["violations"] == CLEAN


def test_stock_caller_command_passes_the_proof(tree, pins, tmp_path):
    """The verb's command line (`cli.cmd_design`, mode off) in a box without weights: the process gets past the proof and refuses at the
    step named for the box — preflight (rc 1, the checkpoint absent; the stock pins are not consulted on this route) — never at the proof (rc 3)."""
    cmd = [sys.executable, "-s", "-m", "pxdesign_opt.stock_infer", "--tree", tree, "--tasks", str(tmp_path / "t.json"), "--out_dir", str(tmp_path / "o"),
           "--det", "0", "--ckpt_dir", str(tmp_path / "no_ckpt")]
    r = subprocess.run(cmd, env=_stock_process_env(tree, pins), capture_output=True, text=True, timeout=300)
    assert "environment not clean" not in r.stderr and r.returncode == 1, r.stderr
    assert "stock pins" not in r.stderr and "[pxdesign-opt] checkpoint missing" in r.stderr, (r.returncode, r.stderr)


