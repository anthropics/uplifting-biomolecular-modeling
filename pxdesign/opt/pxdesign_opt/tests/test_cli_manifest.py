"""The command layer: check on a stub box, mode/env agreement, usage text, the manifest beside the outputs, the deployment gate (presence only, nothing digested)."""
import json
import os

import pytest

from . import _stubs
from pxdesign_opt import cli, manifest


def test_usage_lists_verbs_and_no_variant():
    for v in ("design", "check", "warm"):
        assert f"  {v} " in cli.USAGE
    assert "--variant" not in cli.USAGE and "  serve " not in cli.USAGE


def test_mode_env_disagreement_is_usage_error(monkeypatch, capsys):
    monkeypatch.setenv("PXDESIGN_OPT", "off")
    with pytest.raises(SystemExit) as e:
        cli.main(["check", "--mode", "exact"])
    assert e.value.code == 2 and "disagrees" in capsys.readouterr().err
    monkeypatch.setenv("PXDESIGN_OPT", "turbo")                      # an unknown name on the env route: usage error
    with pytest.raises(SystemExit) as e:
        cli.main(["check"])
    assert e.value.code == 2


def test_check_json_on_stub_box(fresh_stack, monkeypatch, capsys):
    stack = fresh_stack
    _stubs.install_torch(); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack)
    rc = cli.main(["check", "--mode", "exact", "--json"])
    out, err = capsys.readouterr()
    assert rc == 0 and "DRY-RUN mode=exact" in err
    rep = json.loads(out)
    assert rep["levers_planned"] == ["h1", "h2", "h3", "h4", "h5"] and [r["lever"] for r in rep["lever_table"]] == ["h1", "h2", "h3", "h4", "h5", "featdiet", "padmask", "rowpipe", "tf32", "sdedup"]   # the levers: the hoist families + the package levers (exact plans the hoist only)
    assert "package_levers_withheld" not in rep and rep["precision_policy"] == "stock"
    assert rep["package_levers"] == [] and rep["tier"] == "1"
    assert "not_wired" not in rep and rep["core_pin"]["ok"] is True                  # the report names wired levers only
    assert rep["env"] == {"PXD_HOIST": "1", "PXD_HOIST_MODE": "shape", "PXD_HOIST_MASK": "1"}
    stack.reset_for_tests()
    _stubs.fake_box(monkeypatch, stack, cuda=False)
    assert cli.main(["check", "--mode", "exact"]) == 3 == cli.EXIT_NOT_ACTIVE          # the mode would refuse on this box: the refusal's code
    assert cli.main(["check", "--mode", "off"]) == 0


def test_design_refused_activation_exits_3_with_the_not_active_line(fresh_stack, monkeypatch, capsys, tmp_path):
    """`design --mode exact` on a box where activation is refused returns exit status 3 (== EXIT_NOT_ACTIVE) after the NOT ACTIVE line:
    stock never runs silently under the mode."""
    stack = fresh_stack
    _stubs.install_torch(cuda=False); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack, cuda=False)
    tasks = tmp_path / "tasks.json"; tasks.write_text("[]")
    rc = cli.main(["design", "--mode", "exact", "--tasks", str(tasks), "--out_dir", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert rc == 3 and rc == cli.EXIT_NOT_ACTIVE
    assert "[pxdesign-opt] NOT ACTIVE:" in err
    assert "GUARD" not in err                                                       # no guard word on any route
    assert not (tmp_path / "out").exists()                                         # nothing ran


def test_default_mode_is_the_house_rule_and_the_cli_uses_it(clean_env, fresh_stack, monkeypatch, capsys):
    """DEFAULT_MODE names a tested line: `fast` (tier 2; `exact` is selected by name); the CLI resolves to it when neither --mode nor
    PXDESIGN_OPT is given, and `check` without --mode reports that mode."""
    import argparse
    from pxdesign_opt import modes
    assert modes.FAST_MODE == "fast" and modes.DEFAULT_MODE == modes.FAST_MODE
    assert cli._resolve_mode(argparse.Namespace(mode=None)) == modes.DEFAULT_MODE == "fast"
    stack = fresh_stack
    _stubs.install_torch(); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack)
    assert cli.main(["check"]) == 0
    assert "DRY-RUN mode=fast" in capsys.readouterr().err


def test_mode_takes_exactly_the_tables_names(fresh_stack, monkeypatch, capsys):
    """--mode takes exactly modes.MODES on every verb that has the flag: any other word (`rows`, a lever value, included) is a usage error (rc 2)
    carrying the table's one sentence naming the accepted modes, on every verb, before anything is resolved or gated; `--mode Exact` folds onto exact."""
    from pxdesign_opt import modes
    stack = fresh_stack
    _stubs.install_torch(); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack)
    sentence = "[pxdesign-opt] unknown mode 'rows'; expected one of off|exact|fast|big"
    assert sentence == f"[pxdesign-opt] {modes.unknown_message('rows')}"
    for argv in (["check", "--mode", "rows"], ["check", "--mode", "ROWS", "--json"], ["design", "--mode", "rows", "--tasks", "t.json", "--out_dir", "o"],
                 ["warm", "--mode=rows"]):
        with pytest.raises(SystemExit) as ex:
            cli.main(argv)
        out, err = capsys.readouterr()
        assert ex.value.code == cli.EXIT_USAGE and err.strip() == sentence and out == "", (argv, err, out)
    monkeypatch.setenv("PXDESIGN_OPT", "rows")
    with pytest.raises(SystemExit) as ex:
        cli.main(["check"])
    assert ex.value.code == cli.EXIT_USAGE and capsys.readouterr().err.strip() == sentence
    monkeypatch.delenv("PXDESIGN_OPT")
    with pytest.raises(SystemExit) as ex:
        cli.main(["check", "--mode", "turbo"])
    assert ex.value.code == cli.EXIT_USAGE and "unknown mode 'turbo'; expected one of off|exact|fast|big" in capsys.readouterr().err
    for verb in ("design", "check", "warm"):                                          # the flag's help names exactly the table (modes.MODES)
        with pytest.raises(SystemExit) as ex:
            cli.build_parser().parse_args([verb, "--help"])
        out = capsys.readouterr().out
        assert ex.value.code == 0 and "off|exact|fast|big (default fast, or PXDESIGN_OPT)" in out and "rows" not in out, (verb, out)
    assert "rows" not in cli.USAGE and "[--mode exact|fast|big|off]" in cli.USAGE
    assert cli.main(["check", "--mode", "Exact"]) == 0 and "DRY-RUN mode=exact" in capsys.readouterr().err


def test_the_verbs_are_design_check_warm():
    """The commands are exactly design, check and warm (argparse refuses any other word, `serve` included)."""
    sub = next(a for a in cli.build_parser()._actions if a.__class__.__name__ == "_SubParsersAction")
    assert sorted(sub.choices) == ["check", "design", "warm"]
    for argv in (["serve", "--mode", "exact", "--requests", "a:t.json:1:1", "--out_dir", "o"], ["bogus"]):
        with pytest.raises(SystemExit) as ex:
            cli.build_parser().parse_args(argv)
        assert ex.value.code == 2, argv

def test_timings_state_no_per_design_figure_without_the_load_time():
    """Speed statements: the amortised per-design cost of the sampling phase with the one-time load named beside it; when the load time
    is not separable (the console-script route) no per-design figure is stated."""
    summ = {"n_designs": 4, "n_tasks": 2, "n_seeds": 1}
    t = cli._timings(summ, wall=100.0, load_s=20.0)
    assert (t["load_s"], t["sampling_s"], t["s_per_design"], t["s_per_task"]) == (20.0, 80.0, 20.0, 40.0)
    t = cli._timings(summ, wall=100.0, load_s=None)
    assert t["wall_s"] == 100.0 and t["load_s"] is None and t["sampling_s"] is None and t["s_per_design"] is None and t["s_per_task"] is None
    assert "one-time" in t["note"]


def test_manifest_roundtrip(tmp_path):
    rep = {"active": True, "mode": "exact", "levers_planned": ["h1"], "levers_applied": ["h1"], "env": {"PXD_HOIST": "1"}, "gpu": {"name": "x"}, "upstream": {}}
    p = manifest.write(str(tmp_path), rep, mode="exact", command="design", argv=["design"], exit_code=0, options={"values": {"N_sample": "5"}, "given": {}},
                       outputs={"n_designs": 5, "complete": True, "cifs": ["a.cif"]}, timings={"wall_s": 1.0})
    m = manifest.read(str(tmp_path))
    assert p.endswith("opt_manifest.json") and m["mode"] == "exact" and m["outputs"]["n_designs"] == 5 and m["levers_applied"] == ["h1"]
    assert m["schema"] == "pxdesign_opt.opt_manifest.v3" and m["activation_report"]["active"]
    assert not {"pins_check", "checkpoint", "msa_census", "core_pin"} & set(m)             # monitoring only: no pin, digest or census blocks
    import inspect
    assert not {"pins", "msa_census"} & set(inspect.signature(manifest.build).parameters)


def test_preflight_refuses_before_upstream_would_download(tmp_path, tree, monkeypatch):
    """The tree never downloads: preflight refuses when the checkpoint, one of the three Protenix v0.5.0 checkpoints upstream requires
    beside it, or a CCD file is absent; with every file present it records their presence."""
    import json
    from pxdesign_opt import infer_loop
    pins = json.load(open(os.path.join(tree, "stock", "PINS.json")))
    ckpt = tmp_path / "ckpt"; ccd = tmp_path / "ccd"; ckpt.mkdir(); ccd.mkdir()
    monkeypatch.setenv("PROTENIX_DATA_ROOT_DIR", str(ccd))
    (ckpt / "pxdesign_v0.1.0.pt").write_bytes(b"x")
    required = list(pins["weights"]["required_in_dir"]["files"])
    assert required == ["protenix_base_default_v0.5.0.pt", "protenix_mini_default_v0.5.0.pt", "protenix_mini_tmpl_v0.5.0.pt"]
    with pytest.raises(SystemExit) as e:
        infer_loop.preflight(pins, str(ckpt), str(tmp_path / "out"))
    assert "protenix_base_default_v0.5.0.pt" in str(e.value) and "never downloaded" in str(e.value)
    for f in required:
        (ckpt / f).write_bytes(b"x")
    with pytest.raises(SystemExit) as e:
        infer_loop.preflight(pins, str(ckpt), str(tmp_path / "out"))
    assert "CCD cache file missing" in str(e.value)
    for f in pins["ccd_cache"]["files"]:
        (ccd / f).write_bytes(b"x")
    rec = infer_loop.preflight(pins, str(ckpt), str(tmp_path / "out"))
    assert all(rec["required_in_dir"].values()) and set(rec["required_in_dir"]) == set(required) and all(rec["ccd_files"].values())
    (ckpt / "pxdesign_v0.1.0.pt").unlink()
    with pytest.raises(SystemExit) as e:
        infer_loop.preflight(pins, str(ckpt), str(tmp_path / "out"))
    assert "checkpoint missing" in str(e.value)


X_SHA = _stubs.X_SHA                                                            # sha256(b"x"): the tests' stand-in checkpoint bytes


def test_preflight_digests_nothing(tmp_path, tree, monkeypatch, capsys):
    """The deployment gate checks presence only: no checkpoint digest is computed, printed or recorded on any route (the checkpoint is cited
    once, by name and sha256, in STOCK.md); a MISSING checkpoint stays a refusal by name."""
    from pxdesign_opt import infer_loop, outputs
    pins = json.load(open(os.path.join(tree, "stock", "PINS.json")))
    assert (pins["weights"]["checkpoint"]["file"], pins["weights"]["checkpoint"]["sha256"]) == ("pxdesign_v0.1.0.pt", "b075867bae942dc0c6487173736922b0e2913308c1ba542d227418b6e176478d")
    ckpt = _stubs.stage_weights(tmp_path, monkeypatch, pins)
    rec = infer_loop.preflight(pins, str(ckpt), str(tmp_path / "out"))
    assert set(rec) == {"checkpoint", "required_in_dir", "ccd_root", "ccd_files"} and rec["checkpoint"].endswith("pxdesign_v0.1.0.pt")
    assert capsys.readouterr().err == ""                                            # no weights line
    assert not hasattr(infer_loop, "weights_digest") and not hasattr(outputs, "sha256_file") and not hasattr(outputs, "cif_hashes")
    (ckpt / "pxdesign_v0.1.0.pt").unlink()
    with pytest.raises(SystemExit) as e:
        infer_loop.preflight(pins, str(ckpt), str(tmp_path / "out"))
    assert "checkpoint missing" in str(e.value)


def test_no_weights_opt_out_flag(capsys):
    """Warn-and-run needs no flag: `--skip-weights-check` is not an argument of design or the stock caller, and the stock caller's command
    line (cli.stock_command, the one place it is built) never carries it."""
    import inspect
    from pxdesign_opt import stock_infer, infer_loop
    for argv in (["design", "--tasks", "t.json", "--out_dir", "o", "--skip-weights-check"],):
        with pytest.raises(SystemExit) as e:
            cli.build_parser().parse_args(argv)
        assert e.value.code == 2 and "unrecognized arguments: --skip-weights-check" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        stock_infer.parse_args(["--tree", ".", "--tasks", "t.json", "--out_dir", "o", "--skip-weights-check"])
    assert "unrecognized arguments: --skip-weights-check" in capsys.readouterr().err
    assert "skip_weights_check" not in inspect.signature(cli.stock_command).parameters
    from pxdesign_opt import options
    cmd = cli.stock_command("/t", "t.json", "o", opts=options.resolve(seeds="317000", pins={"cli_defaults": {"N_step": 400, "N_sample": 5, "dtype": "bf16", "eta_type": "const", "eta_min": 2.5, "eta_max": 2.5, "num_workers": 16, "use_msa": True, "use_fast_ln": True}}), det=1, ckpt_dir="/ckpt")
    assert "--skip-weights-check" not in cmd and cmd[-2:] == ["--ckpt_dir", "/ckpt"]


def test_outputs_reused_dump_is_not_refused_and_yaml_counts_one_task(tmp_path):
    """A dump directory that already holds a completed (task, seed) is upstream's to handle (it skips it): nothing here refuses it, the count
    names what is found. A YAML input is one task (upstream converts it); its name comes from upstream's converted input_tasks.json once written."""
    from pxdesign_opt import outputs, infer_loop
    d = tmp_path / "task" / "seed_317000"
    d.mkdir(parents=True)
    (d / "SUCCESS_FILE").write_text("{}")
    assert not hasattr(outputs, "preflight_out_dir")
    pins = {"weights": {"checkpoint": {"file": "w.pt"}, "required_in_dir": {"files": {}}}, "ccd_cache": {"files": []}}
    ck = tmp_path / "ck"; ck.mkdir(); (ck / "w.pt").write_bytes(b"x")
    os.environ["PROTENIX_DATA_ROOT_DIR"] = str(ck)
    try:
        assert infer_loop.preflight(pins, str(ck), str(tmp_path))["checkpoint"].endswith("w.pt")   # the reused dump dir passes preflight
    finally:
        os.environ.pop("PROTENIX_DATA_ROOT_DIR", None)
    (d / "predictions").mkdir()
    (d / "predictions" / "task_sample_0.cif").write_text("data_x\n")
    c = outputs.count_designs(str(tmp_path), ["task"], [317000], 5)
    assert (c["n_designs"], c["expected"], c["complete"], c["scope"], c["n_success_files"]) == (1, 5, False, "job", 1)
    c2 = outputs.count_designs(str(tmp_path), ["task"], [], 5)                                           # seeds unknown: every design under the dir, one seed assumed
    assert (c2["n_designs"], c2["expected"], c2["scope"]) == (1, 5, "dir")
    (tmp_path / "task" / "seed_5" / "predictions").mkdir(parents=True); (tmp_path / "task" / "seed_5" / "predictions" / "task_sample_0.cif").write_text("x")
    assert outputs.count_designs(str(tmp_path), ["task"], [317000], 1)["complete"] is True          # another seed's designs in a reused dir do not count against this job
    assert outputs.count_designs(str(tmp_path), ["task"], [], 1)["n_designs"] == 2                   # scope dir sees both
    y = tmp_path / "in.yaml"; y.write_text("name: x\n")
    assert outputs.is_yaml(str(y)) and outputs.n_tasks_of(str(y)) == 1 and outputs.task_names(str(y)) == [] and outputs.task_names(str(y), str(tmp_path)) == []
    (tmp_path / "input_tasks.json").write_text(json.dumps([{"name": "task"}]))
    assert outputs.task_names(str(y), str(tmp_path)) == ["task"]
    s = outputs.summarize(str(tmp_path), str(y), [317000], 5)
    assert (s["tasks"], s["n_tasks"], s["n_seeds"], s["n_designs"], s["expected"], s["cifs"]) == (["task"], 1, 1, 1, 5, [os.path.join("task", "seed_317000", "predictions", "task_sample_0.cif")])
    j = tmp_path / "t.json"; j.write_text(json.dumps([{"name": "a"}, {"condition": {}}]))
    assert outputs.task_names(str(j)) == ["a", "sample_1"] and outputs.n_tasks_of(str(j)) == 2

