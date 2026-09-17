"""The command layer: mode/env agreement, exit codes, check on two stub interpreters, pred refusals before any GPU work."""
import json
import os
import subprocess
import sys

import pytest

from .. import cli, fold, modes, stack, stock_fold, tree
from . import _stubs


def test_mode_of_rules(monkeypatch):
    assert cli.mode_of(None) == "fast" == modes.DEFAULT_MODE                            # --mode omitted, no variable: the default is fast (the one selector, cli.mode_of)
    assert cli.mode_of("off") == "off" and cli.mode_of("exact") == "exact"
    monkeypatch.setenv("ROSETTAFOLD3_OPT", "exact")
    assert cli.mode_of(None) == "exact" and cli.mode_of("exact") == "exact"
    with pytest.raises(cli.CliError, match="disagrees"):
        cli.mode_of("off")
    monkeypatch.delenv("ROSETTAFOLD3_OPT")
    with pytest.raises(cli.CliError, match="unknown mode"):
        cli.mode_of("faster")
    monkeypatch.setenv("ROSETTAFOLD3_OPT", "exact_graphoff")                                   # a lever composition other than the four modes is not a selection
    with pytest.raises(cli.CliError, match="unknown mode"):
        cli.mode_of(None)


def test_usage_and_unknown_command(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    assert cli.main(["--help"]) == cli.EXIT_OK
    assert cli.main(["nope"]) == cli.EXIT_USAGE
    out = capsys.readouterr()
    assert "pred " in out.out and "install" in out.out


def test_check_on_two_stub_interpreters(tmp_path, monkeypatch, capsys):
    stock = _stubs.make_tree(str(tmp_path / "stock_sp"), "stock")
    opt = _stubs.make_tree(str(tmp_path / "opt_sp"), "patched")
    _stubs.make_dist(stock, route="vcs")
    _stubs.make_dist(opt, route="vcs")
    spy = _stubs.make_interpreter(str(tmp_path / "sbin"), stock)
    opy = _stubs.make_interpreter(str(tmp_path / "obin"), opt)
    monkeypatch.setenv(stack.ENV_STOCK_PYTHON, spy)
    monkeypatch.setenv(stack.ENV_PYTHON, opy)
    monkeypatch.setattr(stack, "gpu_info", lambda: None)
    # check --mode exact resolves and gates on THIS interpreter's tree and reports both stub interpreters: exit 0 when this
    # interpreter carries the patched bytes (a box), 3 when it carries no rf3 at all (a CPU test environment)
    import importlib.util
    rc = cli.main(["check", "--mode", "exact"])
    out = capsys.readouterr().out
    here = tree.classify(tree.this_site_packages(), stack.tree_digests()).state if importlib.util.find_spec("rf3") else None
    assert rc == (cli.EXIT_OK if here == "patched" else cli.EXIT_NOT_ACTIVE), (rc, here)
    assert "[rosettafold3-opt] DRY-RUN mode=exact row=RF3_CUDAGRAPH=1,RF3_HOIST=1" in out
    assert "interpreter stock:" in out and "tree=stock(5/5)" in out and "pins=vcs:ok" in out
    assert "interpreter opt:" in out and "tree=patched(5/5)" in out
    # check --mode off: the stock interpreter's state and pins decide
    rc = cli.main(["check", "--mode", "off"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK and "DRY-RUN mode=off" in out and "tree=stock(5/5)" in out and "pins=ok" in out and " cueq=" in out
    # an interpreter whose cuequivariance_torch does not import: said on a GPU-less machine, refused by check on a GPU machine (upstream would
    # silently run its plain triangle kernels)
    monkeypatch.setattr(stack, "cueq_probe", lambda py, timeout=180: {"ok": False, "version": None, "error": "ModuleNotFoundError: No module named 'cuequivariance_torch'"})
    rc = cli.main(["check", "--mode", "off"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK and "cueq=ABSENT(ModuleNotFoundError)" in out, out
    monkeypatch.setattr(stack, "gpu_info", lambda: {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm_90", "memory_mib": 81559})
    rc = cli.main(["check", "--mode", "off"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_NOT_ACTIVE and "check FAIL: cuequivariance_torch does not import on" in out, out
    monkeypatch.setattr(stack, "cueq_probe", lambda py, timeout=180: {"ok": True, "version": "0.11.1", "error": None})
    rc = cli.main(["check", "--mode", "off"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK and "cueq=0.11.1" in out, out


def test_check_off_refuses_a_patched_stock_interpreter(tmp_path, monkeypatch, capsys):
    opt = _stubs.make_tree(str(tmp_path / "opt_sp"), "patched")
    _stubs.make_dist(opt, route="vcs")
    py = _stubs.make_interpreter(str(tmp_path / "bin"), opt)
    monkeypatch.setenv(stack.ENV_STOCK_PYTHON, py)
    monkeypatch.setenv(stack.ENV_PYTHON, py)
    monkeypatch.setattr(stack, "gpu_info", lambda: None)
    assert cli.main(["check", "--mode", "off"]) == cli.EXIT_NOT_ACTIVE
    assert "tree=patched(5/5)" in capsys.readouterr().out


def test_pred_refuses_before_any_work(tmp_path, monkeypatch, capsys):
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps([{"name": "x", "components": [{"seq": "AAAA", "chain_id": "A"}]}]))
    monkeypatch.delenv(stack.ENV_CKPT, raising=False)
    rc = cli.main(["pred", "--input", str(inp), "--out_dir", str(tmp_path / "o"), "--mode", "off"])
    assert rc == cli.EXIT_FAIL and "no checkpoint" in capsys.readouterr().err
    rc = cli.main(["pred", "--input", str(inp), "--out_dir", str(tmp_path / "o"), "--mode", "off", "--ckpt", str(tmp_path / "none.ckpt")])
    assert rc == cli.EXIT_FAIL and "checkpoint not found" in capsys.readouterr().err
    rc = cli.main(["pred", "--input", str(tmp_path / "missing.json"), "--out_dir", str(tmp_path / "o")])
    assert rc == cli.EXIT_USAGE and "no such file or directory" in capsys.readouterr().err
    rc = cli.main(["pred", "--input", str(inp), "--out_dir", str(tmp_path / "o"), "--ckpt", str(inp), "--seeds", "5", "seed=5"])
    assert rc == cli.EXIT_USAGE and "together with --seeds" in capsys.readouterr().err     # seed= alone is rf3 fold's own override and passes; with --seeds it is given twice


def test_pred_off_with_a_stub_cli(tmp_path, monkeypatch, capsys):
    """The whole off route on CPU: the pristine stub interpreter runs a stub rf3.cli that writes one file; no side file beside the outputs."""
    stock = _stubs.make_tree(str(tmp_path / "stock_sp"), "stock")
    _stubs.make_dist(stock, route="vcs")
    (tmp_path / "stock_sp" / "rf3" / "cli.py").write_text(
        "import os, sys, json\n"
        "assert sys.argv[1] == 'fold'\n"
        "kv = dict(a.split('=', 1) for a in sys.argv[2:])\n"
        "assert 'RF3_HOIST' not in os.environ and 'ROSETTAFOLD3_OPT' not in os.environ\n"
        "os.makedirs(kv['out_dir'], exist_ok=True)\n"
        "json.dump(kv, open(os.path.join(kv['out_dir'], 'args.json'), 'w'))\n")
    spy = _stubs.make_interpreter(str(tmp_path / "sbin"), stock)
    monkeypatch.setenv(stack.ENV_STOCK_PYTHON, spy)
    monkeypatch.setenv("ROSETTAFOLD3_OPT_DIGEST_DIR", str(tmp_path / "digests"))                       # the weights digest memo of this test
    monkeypatch.setattr(stock_fold, "kit_dirs_on_path", lambda *a, **k: [])      # this test's "stock" interpreter is this one, which carries the package
    monkeypatch.setenv("RF3_HOIST", "1")
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps([{"name": "x", "components": [{"seq": "AAAA", "chain_id": "A"}]}]))
    ck = tmp_path / "w.ckpt"
    ck.write_bytes(b"0")
    out = tmp_path / "o"
    rc = cli.main(["pred", "--input", str(inp), "--out_dir", str(out), "--mode", "off", "--ckpt", str(ck), "--seeds", "42,43"])
    cap = capsys.readouterr()
    assert rc == cli.EXIT_OK, cap
    assert "WEIGHTS checkpoint=unknown" in cap.err and "proceeding" in cap.err                       # the stub checkpoint is not the pin: a named warning, rc 0
    args = json.load(open(out / "seed-43" / "args.json"))
    assert args == {"inputs": str(inp), "out_dir": str(out / "seed-43"), "ckpt_path": str(ck), "seed": "43"}   # the pass-through: inputs verbatim, outputs, weights, seed
    assert not os.path.exists(out / "input") and os.path.exists(out / "pred.log") and not os.path.exists(out / "opt_manifest.json")   # no side file beside the outputs


def test_pred_exact_route_with_a_stub_cli(tmp_path, monkeypatch, capsys):
    """The kit route on CPU: the patched stub interpreter runs a stub rf3.cli; the row and ROSETTAFOLD3_OPT reach the child."""
    opt = _stubs.make_tree(str(tmp_path / "opt_sp"), "patched")
    _stubs.make_dist(opt, route="vcs")
    (tmp_path / "opt_sp" / "rf3" / "cli.py").write_text(
        "import os, sys, json\n"
        "kv = dict(a.split('=', 1) for a in sys.argv[2:])\n"
        "os.makedirs(kv['out_dir'], exist_ok=True)\n"
        "json.dump({k: os.environ.get(k) for k in ('RF3_CUDAGRAPH', 'RF3_HOIST', 'ROSETTAFOLD3_OPT')}, open(os.path.join(kv['out_dir'], 'env.json'), 'w'))\n"
        "import rf3.graph_flags as gf\n"                     # the kit's own switch table (importable without torch): the watch and the tally read it
        "assert gf.describe()['RF3_HOIST'] is (os.environ['RF3_HOIST'] == '1')\n")
    opy = _stubs.make_interpreter(str(tmp_path / "obin"), opt, hook=True)   # the child's hook (what the .pth installs), without a pip install
    _stubs.fake_nvidia_smi(str(tmp_path / "obin"))                     # the child's own hook gates on a visible GPU
    monkeypatch.setenv(stack.ENV_PYTHON, opy)
    monkeypatch.setattr(stack, "gpu_info", lambda: dict(_stubs.FAKE_GPU))
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"name": "x", "components": [{"seq": "AAAA", "chain_id": "A"}]}))
    ck = tmp_path / "w.ckpt"
    ck.write_bytes(b"0")
    out = tmp_path / "o"
    rc = cli.main(["pred", "--input", str(inp), "--out_dir", str(out), "--mode", "exact", "--ckpt", str(ck), "--seeds", "7"])
    cap = capsys.readouterr()
    assert rc == cli.EXIT_FAIL, cap                              # a mode is a contract: the stub fold ran none of the row's kernels (no FPF tally, no xtr
    line = [l for l in cap.out.splitlines() if " pred FAIL row=exact " in l]   # tally), so the verdict is FAIL by name, exit 1, and the line names the escape
    assert line and "failures=seed 7: FPF fpf_rf3_adapter never imported" in line[0] and "seed 7: xtr not installed" in line[0] and fold.FAIL_ESCAPE in line[0], cap.out
    env = json.load(open(out / "seed-7" / "env.json"))
    assert env == {"RF3_CUDAGRAPH": "1", "RF3_HOIST": "1", "ROSETTAFOLD3_OPT": "exact"}
    log = open(out / "pred.log").read()
    assert "[rosettafold3-opt] ACTIVE mode=exact row=RF3_CUDAGRAPH=1,RF3_HOIST=1 fpf=tg+sapb+xatt+xmul.eager+xln+smsa@L1.warm tree=patched(5/5)" in log
    assert "[rosettafold3-opt] APPLIED rf3.graph_flags RF3_CUDAGRAPH=1 RF3_GRAPH_SAFE_OPS=True" in log and "RF3_HOIST=True" in log
    assert "[rosettafold3-opt] EXIT mode=exact rollouts=0" in log and "sharding=none" in log.split("EXIT mode=exact")[1].splitlines()[0]


def test_kit_env_is_the_row_plus_the_callers_environment(monkeypatch):
    """fold.kit_env alone (no hook, no child): exactly the row's switches, ROSETTAFOLD3_OPT=<row>, the tally file, and the caller's environment
    kept as given (PYTHONPATH included: a site attached from outside rides through); a contradicting switch in the environment is refused.
    The stock caller keeps the caller's environment too, minus the kits' names."""
    from .. import fold, modes
    res = modes.resolve("exact", stack.kit_home())
    env = fold.kit_env(res, environ={"PATH": "/usr/bin", "PYTHONPATH": "/x"}, tally_file="/t.json")
    assert env == {"PATH": "/usr/bin", "PYTHONPATH": "/x", "RF3_CUDAGRAPH": "1", "RF3_HOIST": "1", "ROSETTAFOLD3_OPT": "exact",
                   "PYTHONDONTWRITEBYTECODE": "1", stack.ENV_TALLY_FILE: "/t.json", stack.ENV_N_GPU: "1"}   # the resource axis rides to the fold process (absent flag = 1)
    env = fold.kit_env(res, environ={"PYTHONPATH": os.pathsep.join(["/attached_site", "/x"]), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    assert env["PYTHONPATH"] == os.pathsep.join(["/attached_site", "/x"]) and env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"   # the caller's variables ride through untouched
    with pytest.raises(RuntimeError, match="contradicts row exact"):
        fold.kit_env(res, environ={"RF3_HOIST": "0"})
    # the stock caller strips the kits' names and keeps everything else of the caller's environment
    pins = stack.pins()
    clean, stripped = stock_fold.clean_env(pins, environ={"RF3_HOIST": "1", "ROSETTAFOLD3_OPT": "exact", "HOME": "/h", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                                          "PYTHONPATH": os.pathsep.join(["/attached_site", "/x"])})
    assert stripped == ["RF3_HOIST", "ROSETTAFOLD3_OPT"]
    assert clean == {"HOME": "/h", "PYTHONDONTWRITEBYTECODE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": os.pathsep.join(["/attached_site", "/x"])}


def test_pred_exact_refuses_a_stock_patched_interpreter(tmp_path, monkeypatch, capsys):
    stock = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    py = _stubs.make_interpreter(str(tmp_path / "bin"), stock)
    monkeypatch.setenv(stack.ENV_PYTHON, py)
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"name": "x", "components": [{"seq": "AAAA", "chain_id": "A"}]}))
    ck = tmp_path / "w.ckpt"
    ck.write_bytes(b"0")
    rc = cli.main(["pred", "--input", str(inp), "--out_dir", str(tmp_path / "o"), "--mode", "exact", "--ckpt", str(ck)])
    assert rc == cli.EXIT_FAIL and "tree state is 'stock' (expected patched)" in capsys.readouterr().err


def test_console_entry_points():
    env = dict(os.environ, PYTHONPATH=stack.opt_root() + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-m", "rosettafold3_opt", "--help"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "usage: rosettafold3-opt" in r.stdout


def test_pred_hands_input_to_rf3_verbatim(tmp_path, monkeypatch, capsys):
    """--input is rf3 fold's inputs= verbatim (a directory here); a plain path that does not exist is a usage error by name."""
    seen = {}
    monkeypatch.setattr(fold, "run", lambda mode, **k: (seen.update(k), {"status": "PASS", "runs": [], "mode": mode})[1])
    monkeypatch.setattr(fold, "summary_line", lambda rec: "ok")
    d = tmp_path / "structs"; d.mkdir()
    assert cli.main(["pred", "--input", str(d), "--out_dir", str(tmp_path / "o"), "--mode", "off", "--seeds", "1"]) == cli.EXIT_OK
    assert seen["inputs"] == str(d) and not os.path.exists(tmp_path / "o" / "input")
    assert cli.main(["pred", "--input", str(tmp_path / "nope.cif"), "--out_dir", str(tmp_path / "o2"), "--mode", "off"]) == cli.EXIT_USAGE
    assert "no such file or directory" in capsys.readouterr().err


def test_pred_without_seeds_runs_one_unseeded_fold_into_out_dir(tmp_path, monkeypatch):
    """No --seeds → seeds=[None]: one `rf3 fold` with NO seed= token (upstream's `seed: null` default) whose out_dir is <out_dir>/ itself;
    --seeds S keeps <out_dir>/seed-<S>/ and seed=<S>."""
    assert cli._seeds(None) == [None] and cli._seeds("7,8") == [7, 8]
    assert fold.seed_dir("/o", None) == "/o" and fold.seed_dir("/o", 7) == "/o/seed-7" and fold.seed_label(None) == "unseeded"
    cmd = stock_fold.command("py", "/i.json", fold.seed_dir("/o", None), "/w.ckpt", ["diffusion_batch_size=1"], None)
    assert cmd[-3:] == ["out_dir=/o", "ckpt_path=/w.ckpt", "diffusion_batch_size=1"] and not any(t.startswith("seed=") for t in cmd)
    cmd7 = stock_fold.command("py", "/i.json", fold.seed_dir("/o", 7), "/w.ckpt", [], 7)
    assert "out_dir=/o/seed-7" in cmd7 and cmd7[-1] == "seed=7"
    seen = {}
    monkeypatch.setattr(fold, "run", lambda mode, **k: (seen.update(k), {"status": "PASS", "runs": [], "mode": mode})[1])
    monkeypatch.setattr(fold, "summary_line", lambda rec: "ok")
    inp = tmp_path / "in.json"; inp.write_text("[]")
    assert cli.main(["pred", "--input", str(inp), "--out_dir", str(tmp_path / "o"), "--mode", "off"]) == cli.EXIT_OK and seen["seeds"] == [None]


def test_the_unseeded_fold_has_no_seed_token_and_a_seed_override_is_read():
    """The unseeded fold has no seed= token (rf3 fold's default); a seed= override, the caller's or the launcher's drawn one, is what cmd_seed reads."""
    cmd = stock_fold.command("py", "/i.json", "/o", "/w.ckpt", [], None)
    assert fold.cmd_seed(cmd) is None and fold.cmd_seed(cmd + ["seed=5"]) == 5
    assert stock_fold.command("py", "/i.json", "/o", "/w.ckpt", ["seed=9"], None)[-1] == "seed=9"

