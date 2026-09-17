"""pred --mode off: the stock command (upstream's own options as given, nothing else), its clean environment and the proof (printed)."""
import json
import os
import re
import sys

import pytest

from .. import cli, det, settings, stack, stock_pred
from . import _stubs


KIT_ROW = ["--recycling_steps", "3", "--diffusion_samples", "1", "--output_format", "mmcif", "--write_full_pae", "--override", "--num_workers", "1", "--no_kernels"]   # the former `kit` settings row, as explicit boltz predict options


def test_the_former_presets_are_their_explicit_flags_byte_for_byte(tmp_path, monkeypatch):
    """No presets: the stock command carries exactly the `boltz predict` options the caller gave, in the caller's order. The three former stock
    presets render from explicit flags to the argv they used to produce: `upstream` = nothing (+ --seed S), `default` = nothing, `kit` = its
    seven tokens (+ --seed S); the former `--det 1` recipe = --num_workers 1 --no_kernels."""
    monkeypatch.setenv("BOLTZ_CACHE", "/cache")
    y = _stubs.write_yamls(str(tmp_path), ("a",))[0]; out = str(tmp_path / "o")
    stock = lambda argv: (lambda full: full[full.index("--") + 1:])(cli.stock_command(y, out, *argv)[0])
    base = ["predict", y, "--out_dir", out]
    assert stock(({},)) == stock((None,)) == base, "nothing given: upstream exactly as shipped (the former `default`, and `upstream` without a seed)"
    assert stock(({"seed": 5},)) == base + ["--seed", "5"], "the former `upstream` row: + --seed S"
    kit = {"recycling_steps": 3, "diffusion_samples": 1, "output_format": "mmcif", "write_full_pae": True, "override": True, "num_workers": 1, "no_kernels": True, "seed": 5}
    order = ["recycling_steps", "diffusion_samples", "output_format", "write_full_pae", "override", "num_workers", "no_kernels", "seed"]
    assert stock((kit, [], order)) == base + KIT_ROW + ["--seed", "5"], "the former `kit` row, byte for byte, from its explicit flags in their order"
    assert stock(({"seed": 5}, [], None, 1)) == base + ["--seed", "5", "--num_workers", "1", "--no_kernels"] == stock(({"seed": 5, "num_workers": 1, "no_kernels": True}, [], ["seed", "num_workers", "no_kernels"])), "--det 1 = the recipe's two switches, added when absent"
    assert stock((kit, [], order, 1)) == stock((kit, [], order)), "the former kit row already carries the recipe's switches"
    a = cli.build_parser().parse_args(["pred", "--mode", "off", "--input", y, "--out_dir", out] + KIT_ROW + ["--seed", "5"])
    a.knob_order = cli.knob_order(KIT_ROW + ["--seed", "5"])
    assert stock((cli.knob_flags(a), [], a.knob_order)) == base + KIT_ROW + ["--seed", "5"], "the command line's own order reaches boltz predict unchanged"
    assert settings.stock_argv({"write_full_pae": False, "recycling_steps": None}) == [] and settings.settings_word({}) == "defaults" and settings.settings_word({"seed": 1}) == "flags"
    for bad, msg in (({"sampling_steps": 0}, "--sampling_steps 0 is below 1"), ({"output_format": "cif"}, "--output_format 'cif': one of pdb|mmcif"), ({"bogus": 1}, "unknown boltz predict setting")):
        with pytest.raises(ValueError, match=re.escape(msg)):
            settings.given(bad)


def test_det_level_one_adds_only_what_is_absent():
    assert det.stock_args(1, ["--num_workers", "1"]) == ["--no_kernels"]
    assert det.stock_args(1, ["--no_kernels"]) == ["--num_workers", "1"]
    assert det.stock_args(0, []) == [] and det.stock_args(None, []) == []
    with pytest.raises(ValueError):
        det.stock_args(2, [])


def test_det_sees_the_remainder_and_never_doubles_its_switches(tmp_path, monkeypatch):
    """`--det 1` adds --num_workers 1 / --no_kernels only when absent — by name OR among the arguments after `--` — in the order: the caller's
    boltz predict options, the recipe's switches, then the REMAINDER verbatim."""
    monkeypatch.setenv("BOLTZ_CACHE", "/cache")
    y = _stubs.write_yamls(str(tmp_path), ("a",))[0]; out = str(tmp_path / "o")
    def stock(flags, extra, level):
        argv, _env = cli.stock_command(y, out, flags, extra, None, level)
        return argv, argv[argv.index("--") + 1:][4:]                       # after `predict <yaml> --out_dir <out>`
    argv, tail = stock({"seed": 3}, ["--", "--num_workers", "1", "--no_kernels"], 1)                       # (a) det 1 + the switches in the REMAINDER
    assert tail == ["--seed", "3", "--num_workers", "1", "--no_kernels"] and tail.count("--num_workers") == 1 and tail.count("--no_kernels") == 1
    argv_b, tail = stock({"seed": 3}, [], 1)                                                                 # (b) det 1, no REMAINDER: both added once
    assert tail == ["--seed", "3", "--num_workers", "1", "--no_kernels"]
    argv, tail = stock({"seed": 3}, ["--", "--num_workers", "1", "--no_kernels"], None)                    # (c) no det: the REMAINDER verbatim, nothing added
    assert tail == ["--seed", "3", "--num_workers", "1", "--no_kernels"]
    argv, tail = stock({"seed": 3}, ["--", "--max_msa_seqs", "64"], 1)                                       # ordering: options, recipe switches, then REMAINDER
    assert tail == ["--seed", "3", "--num_workers", "1", "--no_kernels", "--max_msa_seqs", "64"]
    assert argv_b[argv_b.index("--kernels-settings") + 1] == "flags+det1" and argv_b[argv_b.index("--kernels-route") + 1] == "stock"   # (d) the census words of (b) unchanged
    argv_d, _ = stock({}, [], 1)
    assert argv_d[argv_d.index("--kernels-settings") + 1] == "defaults+det1"


def test_stock_command_shape_and_clean_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BOLTZ_CACHE", "/cache"); monkeypatch.setenv("BOLTZ_LEVERS", "resid,mask2"); monkeypatch.setenv("BOLTZ2_OPT", "off"); monkeypatch.setenv("CUEQ_DISABLE_AOT_TUNING", "1")
    monkeypatch.setenv("PYTHONPATH", stack.kit_path("forward/trunk_levers") + os.pathsep + "/keep/me")
    y = _stubs.write_yamls(str(tmp_path), ("a",))[0]
    argv, env = cli.stock_command(y, str(tmp_path / "out"), {"seed": 3}, ["--", "--max_msa_seqs", "64"], None, 1)
    i = argv.index("--"); stock = argv[i + 1:]
    assert stock[:4] == ["predict", y, "--out_dir", str(tmp_path / "out")] and not {"--model", "--checkpoint", "--cache"} & set(stock), "upstream resolves the model, the checkpoint and the cache itself ($BOLTZ_CACHE)"
    assert stock.count("--no_kernels") == 1 and stock[stock.index("--seed") + 1] == "3" and stock[-2:] == ["--max_msa_seqs", "64"]
    assert "-s" in argv and "boltz2_opt.stock_pred" in argv
    assert "BOLTZ_LEVERS" not in env and "BOLTZ2_OPT" not in env and "CUEQ_DISABLE_AOT_TUNING" not in env and env["BOLTZ_CACHE"] == "/cache"
    assert env["PYTHONPATH"] == "/keep/me"
    assert "--proof-json" not in argv and not any("stock_env_proof" in x for x in argv), "the proof is the printed line; no proof file"


def test_the_worker_route_s_settings_are_upstream_s_defaults_with_the_given_options_over_them():
    """settings.worker_settings(mode, flags): nothing given = `boltz predict`'s defaults (stock/PINS.json cli_defaults = main.py's click defaults,
    step_scale 1.5 = predict()'s Boltz-2 value); the worker's predict_args / step_scale / writer format read the batch (no literal of its own);
    --no_kernels is refused by name on the worker route; --num_workers is the DataLoader's."""
    eff = settings.worker_settings("exact", None)
    assert {k: eff[k] for k in settings.PREDICT_FLAGS} == {"recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1, "max_parallel_samples": 5}
    assert eff["step_scale"] == 1.5 and eff["write_full_pae"] is False and eff["write_full_pde"] is False and eff["output_format"] == "mmcif" and eff["kernels"] == "on" and eff["given"] == []
    assert settings.worker_settings("fast", {}) == dict(eff) and settings.settings_tokens(eff) == ""
    main = open(os.path.join(stack.tree_dir(), "stock", "src", "boltz", "main.py")).read()
    d = settings.stock_defaults()
    for flag in ("recycling_steps", "sampling_steps", "diffusion_samples", "max_parallel_samples", "num_workers"):
        assert re.search(r'"--%s",(?:(?!@click).)*?default=%d,' % (flag, d[flag]), main, re.S), f"main.py's --{flag} default is {d[flag]}"
    assert re.search(r'"--output_format",(?:(?!@click).)*?default="mmcif"', main, re.S) and d["output_format"] == "mmcif"
    for flag in ("write_full_pae", "write_full_pde", "override", "no_kernels"):
        assert re.search(r'"--%s",(?:(?!@click).)*?is_flag=True' % flag, main, re.S) and d[flag] is False, flag
    assert "step_scale = 1.5 if step_scale is None else step_scale" in main and settings.BOLTZ2_STEP_SCALE == 1.5 and d["seed"] is None
    ten = settings.worker_settings("exact", {"recycling_steps": 10, "diffusion_samples": 5, "write_full_pae": True, "output_format": "pdb", "step_scale": 2.0})
    assert ten["recycling_steps"] == 10 and ten["diffusion_samples"] == 5 and ten["sampling_steps"] == 200 and ten["write_full_pae"] is True and ten["output_format"] == "pdb"
    assert settings.settings_tokens(ten) == "recycling_steps=10 diffusion_samples=5 step_scale=2.0 write_full_pae=1 output_format=pdb" and ten["given"] == ["recycling_steps", "diffusion_samples", "step_scale", "write_full_pae", "output_format"]
    with pytest.raises(ValueError, match=r"not served on the worker route \(--mode fast\): --no_kernels: "):
        settings.worker_settings("fast", {"no_kernels": True})
    assert eff["num_workers"] is None and settings.worker_settings("fast", {"num_workers": 3})["num_workers"] == 3, "--num_workers: the worker's DataLoader's (forwarded on the worker command); absent = the line's own 1"
    src = open(stack.kit_path("forward/trunk_levers/src/bz_worker_lev.py")).read()
    assert '"recycling_steps": PRED["recycling_steps"], "sampling_steps": PRED["sampling_steps"], "diffusion_samples": PRED["diffusion_samples"]' in src and 'PRED = B["predict"]' in src
    assert 'diffusion_params.step_scale = float(B["step_scale"])' in src and 'output_format=B["output_format"]' in src and '"write_full_pae": bool(B["write_full_pae"]), "write_full_pde": bool(B["write_full_pde"])' in src


def test_env_proof_passes_on_a_clean_process_with_the_pinned_tree(tmp_path):
    site = _stubs.unpack_wheel(str(tmp_path / "site")); sys.path.insert(0, site)
    try:
        import importlib; importlib.invalidate_caches()
        pins = stack.load_pins(); se = pins["stock_environment"]
        pr = stock_pred.env_proof({"BOLTZ_CACHE": "/c", "PATH": "/usr/bin"}, se["must_be_absent_prefixes"], se["kit_module_prefixes"], [stack.kit_path("forward")], pins,
                                  modules={"sys": sys, "os": os}, path=[site, "/usr/lib/python3"])
        assert pr["ok"], pr["problems"]
        assert pr["boltz_pinned"] is True and pr["boltz_tree"]["n_py_files"] == 107
    finally:
        sys.path.remove(site)


def test_env_proof_fails_on_forbidden_env_kit_module_or_kit_path(tmp_path):
    pins = stack.load_pins(); se = pins["stock_environment"]
    kd = [stack.kit_path("forward")]
    pr = stock_pred.env_proof({"BOLTZ_LEVERS": "resid,mask2", "BOLTZ_CACHE": "/c"}, se["must_be_absent_prefixes"], se["kit_module_prefixes"], kd, None, modules={}, path=[])
    assert not pr["ok"] and "forbidden env ['BOLTZ_LEVERS']" in pr["problems"][0] and pr["forbidden_present"] == ["BOLTZ_LEVERS"]
    import types
    fake = types.ModuleType("boltz_trunk_levers"); fake.__file__ = stack.kit_path("forward/trunk_levers/boltz_trunk_levers.py")
    pr = stock_pred.env_proof({}, se["must_be_absent_prefixes"], se["kit_module_prefixes"], kd, None, modules={"boltz_trunk_levers": fake}, path=[])
    assert pr["kit_modules_loaded"] == ["boltz_trunk_levers"] and any("kit modules ['boltz_trunk_levers']" in p for p in pr["problems"])
    pr = stock_pred.env_proof({}, se["must_be_absent_prefixes"], se["kit_module_prefixes"], kd, None, modules={}, path=[stack.kit_path("forward/dit_hoist/src")])
    assert pr["kit_dirs_on_path"] and any("kit dirs ['" in p for p in pr["problems"])
    pr = stock_pred.env_proof({}, se["must_be_absent_prefixes"], se["kit_module_prefixes"], kd, None, modules={"torch": types.ModuleType("torch")}, path=[])
    assert pr["torch_loaded_before_proof"] is True and any("torch loaded True" in p for p in pr["problems"])


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_env_proof_refuses_a_tree_that_is_not_the_pin(tmp_path):
    """The proof's pin check: the same pinned tree against tampered pins is 'boltz not at the pin' — in process and in the real subprocess (rc 3)."""
    import copy, subprocess
    site = _stubs.unpack_wheel(str(tmp_path / "site")); sys.path.insert(0, site)
    try:
        import importlib; importlib.invalidate_caches()
        pins = stack.load_pins(); se = pins["stock_environment"]
        bad = copy.deepcopy(pins); bad["upstream"]["installed_tree"]["sha256_all_py_concat"] = "0" * 64
        pr = stock_pred.env_proof({"BOLTZ_CACHE": "/c"}, se["must_be_absent_prefixes"], se["kit_module_prefixes"], [stack.kit_path("forward")], bad, modules={"sys": sys}, path=[site])
        assert pr["ok"] is False and pr["boltz_pinned"] is False and any(p.startswith("boltz not at the pin") for p in pr["problems"]), pr["problems"]
        bad2 = copy.deepcopy(pins); bad2["boltz_version"] = "2.2.0"
        pr = stock_pred.env_proof({}, se["must_be_absent_prefixes"], se["kit_module_prefixes"], [stack.kit_path("forward")], bad2, modules={"sys": sys}, path=[site])
        assert pr["ok"] is False and any("boltz not at the pin: version 2.2.1" in p for p in pr["problems"])
    finally:
        sys.path.remove(site)
    bad_pins = tmp_path / "bad_pins.json"; json.dump(bad, open(bad_pins, "w"))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BOLTZ", "FPFBZ_", "PF_", "CUEQ_", "BZ_"))}; env["PYTHONPATH"] = site; env["BOLTZ_CACHE"] = "/c"
    r = subprocess.run([sys.executable, "-s", "-m", "boltz2_opt.stock_pred", "--env-absent", ",".join(se["must_be_absent_prefixes"]),
                        "--kit-modules", ",".join(se["kit_module_prefixes"]), "--kit-dirs", stack.kit_path("forward"), "--pins", str(bad_pins), "--proof-only", "1", "--", "predict", "x.yaml"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "[boltz2-opt stock] NOT STOCK: boltz not at the pin" in r.stdout, (r.stdout, r.stderr)   # the proof's verdict is the printed line; nothing is written


def test_stock_caller_refuses_before_running_when_not_stock(tmp_path):
    import subprocess
    env = dict(os.environ, BOLTZ_LEVERS="resid,mask2"); env.pop("BOLTZ2_OPT", None)
    se = stack.load_pins()["stock_environment"]
    r = subprocess.run([sys.executable, "-s", "-m", "boltz2_opt.stock_pred", "--env-absent", ",".join(se["must_be_absent_prefixes"]),
                        "--kit-modules", ",".join(se["kit_module_prefixes"]), "--kit-dirs", stack.kit_path("forward"), "--", "predict", "x.yaml"], capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "[boltz2-opt stock] NOT STOCK: forbidden env ['BOLTZ_LEVERS']" in r.stdout


@pytest.mark.requires_core(reason="requires_core: exercises a subprocess/worker path that must find boltz2_opt+opt_core with no PYTHONPATH override (a real pip install, e.g. via run.sh install) -- this box carries them only on this run's dev PYTHONPATH")
def test_stock_caller_proves_a_clean_process_with_the_pinned_tree_and_the_packages_own_hook(tmp_path):
    """The real subprocess: the .pth hook's inert import (boltz2_opt._autoload) is this package's own and does not fail the proof."""
    import subprocess
    site = _stubs.unpack_wheel(str(tmp_path / "site"))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BOLTZ", "FPFBZ_", "PF_", "CUEQ_", "BZ_"))}; env["PYTHONPATH"] = site; env["BOLTZ_CACHE"] = "/c"
    pins = stack.load_pins(); se = pins["stock_environment"]
    r = subprocess.run([sys.executable, "-s", "-m", "boltz2_opt.stock_pred", "--env-absent", ",".join(se["must_be_absent_prefixes"]),
                        "--kit-modules", ",".join(se["kit_module_prefixes"]), "--kit-dirs", ",".join([stack.kit_path("forward")]), "--pins", stack.pins_path(),
                        "--proof-only", "1", "--", "predict", "x.yaml"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "[boltz2-opt stock] proven stock (proof only)" in r.stdout and "NOT STOCK" not in r.stdout, (r.stdout, r.stderr)
    env["BOLTZ2_OPT"] = "exact"                                           # the armed finder is the failure the hook check is for
    r = subprocess.run([sys.executable, "-s", "-m", "boltz2_opt.stock_pred", "--env-absent", ",".join(se["must_be_absent_prefixes"]),
                        "--kit-modules", ",".join(se["kit_module_prefixes"]), "--kit-dirs", stack.kit_path("forward"), "--proof-only", "1", "--", "predict", "x.yaml"], capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "BOLTZ2_OPT" in r.stdout


def test_off_route_refuses_before_the_stock_cli_could_download(tmp_path, monkeypatch, capsys):
    """Frozen weights: upstream's CLI downloads whatever its cache lacks (main.py:198-256) — `pred --mode off` refuses by name first, on every
    file of that download set, and never launches the subprocess."""
    cache = tmp_path / "cache"; cache.mkdir()
    for f in ("boltz2_conf.ckpt", "ccd.pkl", "mols.tar"):
        (cache / f).write_bytes(b"x")
    (cache / "mols").mkdir()
    monkeypatch.setenv("BOLTZ_CACHE", str(cache))
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("the stock CLI was launched")))
    y = _stubs.write_yamls(str(tmp_path), ("a",))[0]
    rc = cli.main(["pred", "--mode", "off", "--input", y, "--out_dir", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert rc == 3 and calls == [] and "[boltz2-opt] NOT ACTIVE: frozen weights: BOLTZ_CACHE=" in out and "lacks boltz2_aff.ckpt" in out, out
    monkeypatch.delenv("BOLTZ_CACHE")
    rc = cli.main(["pred", "--mode", "off", "--input", y, "--out_dir", str(tmp_path / "o")])
    assert rc == 3 and calls == [] and "BOLTZ_CACHE is not set" in capsys.readouterr().out
    assert stack.CACHE_FILES == ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "ccd.pkl", "mols", "mols.tar"), "the kits' inputs + upstream's download set (main.py:198-256)"
