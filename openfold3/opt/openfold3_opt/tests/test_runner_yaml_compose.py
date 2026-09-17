"""A caller's `--runner-yaml` under a kit mode is composed under the line (runner_yaml.compose): the keys the mode requires are laid on top and
named; everything else runs as given (execution keys that differ from the member are named too); workload keys are never touched; the
composition is the caller's own path when the pins change nothing in it."""
import os

import yaml

from openfold3_opt import cli, modes, runner_yaml, tp

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EVAL = "model_update.custom.settings.memory.eval."


def test_deep_overlay_merges_mappings_replaces_leaves_and_names_both():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": "keep", "c": None}
    top = {"a": {"x": 2, "z": {"q": True}}, "c": {"d": 1}, "e": 5}
    merged, overrides, added = runner_yaml.deep_overlay(base, top)
    assert merged == {"a": {"x": 2, "y": [1, 2], "z": {"q": True}}, "b": "keep", "c": {"d": 1}, "e": 5}
    assert overrides == [("a.x", 1, 2)] and sorted(added) == [("a.z.q", True), ("c.d", 1), ("e", 5)]
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": "keep", "c": None}                     # the caller's document is not mutated
    m2, o2, a2 = runner_yaml.deep_overlay({"a": [1]}, {"a": {"k": 0}})                        # a list where the line needs a mapping: replaced whole, named
    assert m2 == {"a": {"k": 0}} and o2 == [("a", [1], {"k": 0})] and a2 == []


def test_line_pins_are_the_members_execution_keys_plus_the_graphed_lines_kernel_flags():
    """A mode pins every execution key its own runner yaml writes (pl_trainer_args + model_update of the member); the graphed line adds the
    four alternative attention kernel flags off; a member's workload sections (none today) are never pins."""
    M = lambda rel: os.path.join(HOME, rel)
    fast = runner_yaml.line_pins(modes.LINES[("fast", None)], M(modes.FAST_BF16_YAML))
    member = yaml.safe_load(open(M(modes.FAST_BF16_YAML)))
    assert fast["pl_trainer_args"] == member["pl_trainer_args"] == {"precision": "bf16-mixed"}
    ev = fast["model_update"]["custom"]["settings"]["memory"]["eval"]
    assert all(ev[f] is False for f in runner_yaml.KERNEL_EVAL_FLAGS)
    assert {k: v for k, v in ev.items() if k in member["model_update"]["custom"]["settings"]["memory"]["eval"]} == member["model_update"]["custom"]["settings"]["memory"]["eval"]
    for key, rel in ((("exact", "cueq"), modes.STOCK_DET_YAML), (("big", "resident"), modes.BIG_BF16_C16_YAML), (("big", "tp"), modes.BIG_BF16_C16_YAML)):
        doc = yaml.safe_load(open(M(rel)))
        assert runner_yaml.line_pins(modes.LINES[key], M(rel)) == {s: doc[s] for s in ("pl_trainer_args", "model_update") if s in doc}, key

def test_a_callers_yaml_under_fast_keeps_its_workload_keys_and_gets_the_lines_keys(tmp_path, capsys):
    mine = tmp_path / "mine.yml"
    yaml.safe_dump({"experiment_settings": {"seeds": [7, 8]}, "template_preprocessor_settings": {"structure_directory": "/data/templ", "fetch_missing_structures": False},
                    "pl_trainer_args": {"precision": "32-true", "devices": 1},
                    "model_update": {"presets": ["predict"], "custom": {"settings": {"memory": {"eval": {"use_cueq_triangle_kernels": True, "chunk_size": 4}}}}}},
                   open(mine, "w"), sort_keys=False)
    got = cli.row_yaml(HOME, modes.LINES[("fast", None)], modes.FAST_PRECISION, out_dir=str(tmp_path / "o"), runner_yaml=str(mine), mode="fast")
    err = capsys.readouterr().err
    doc = yaml.safe_load(open(got))
    assert got == str(tmp_path / "o" / ("runner_composed_" + mine.name))
    assert doc["experiment_settings"] == {"seeds": [7, 8]} and doc["template_preprocessor_settings"]["structure_directory"] == "/data/templ"   # workload keys untouched
    assert doc["pl_trainer_args"] == {"precision": "bf16-mixed", "devices": 1}                                                                 # the line's precision; the caller's other trainer keys kept
    ev = doc["model_update"]["custom"]["settings"]["memory"]["eval"]
    assert ev["use_cueq_triangle_kernels"] is False and ev["chunk_size"] == 4 and doc["model_update"]["presets"] == ["predict"]          # presets: the ordered union (equal here)
    assert "[openfold3-opt] note: runner yaml" in err and "composed under the fast configuration (opt/openfold3_opt/fast_bf16_predict.yml) -> " + got in err
    assert "fast overrides runner-yaml keys: pl_trainer_args.precision ('32-true' -> 'bf16-mixed'); " + EVAL + "use_cueq_triangle_kernels (True -> False)" in err, err
    assert "fast sets runner-yaml keys the caller's yaml leaves unset: " + EVAL + "use_deepspeed_evo_attention=False" in err
    assert "caller keys kept as written that differ from the member: " in err and EVAL + "chunk_size=4 (member unset)" in err and "pl_trainer_args.devices=1 (member unset)" in err
    rec = cli.COMPOSED[str(mine)]
    assert rec["composed"] == got and ["pl_trainer_args.precision", "32-true", "bf16-mixed"] in rec["overrides"] and [EVAL + "chunk_size", 4, "unset"] in rec["kept"]


def test_big_lays_its_precision_on_a_callers_fp32_yaml_and_a_member_copy_runs_as_is(tmp_path, capsys):
    """`big` is bf16: the fp32 kernels-off configuration (opt/forward/offload/config/stock_predict_c16.yml) under it gets the member's
    `precision: bf16-mixed` laid on and named; a caller's copy of the member itself (a seeded copy of it) already carries every key the
    mode writes — its own path comes back (byte-identical argv), nothing is written."""
    c16 = os.path.join(HOME, modes.OFFLOAD_C16_YAML)
    got = cli.row_yaml(HOME, modes.LINES[("big", "resident")], None, out_dir=str(tmp_path), runner_yaml=c16, mode="big")
    err = capsys.readouterr().err
    assert os.path.basename(got) == "runner_composed_stock_predict_c16.yml" and yaml.safe_load(open(got))["pl_trainer_args"] == {"precision": "bf16-mixed"}
    assert "composed under the big configuration (opt/openfold3_opt/big_bf16_c16_predict.yml) -> " in err and "big sets runner-yaml keys the caller's yaml leaves unset: pl_trainer_args.precision='bf16-mixed'" in err, err
    seeded = tmp_path / "runner_42.yml"; doc = yaml.safe_load(open(os.path.join(HOME, modes.BIG_BF16_C16_YAML))); doc["experiment_settings"] = {"seeds": [42]}
    yaml.safe_dump(doc, open(seeded, "w"), sort_keys=False)
    assert cli.row_yaml(HOME, modes.LINES[("big", "resident")], None, out_dir=str(tmp_path / "o"), runner_yaml=str(seeded), mode="big") == str(seeded)
    err = capsys.readouterr().err
    assert "every key the mode writes already as written, the file runs as is" in err and "overrides" not in err and not (tmp_path / "o").exists(), err

def test_off_composes_a_callers_yaml_under_the_stock_configuration_and_a_rank_takes_the_launchers(tmp_path, capsys):
    """A caller's yaml under off is an overlay on the stock configuration (stock's keys pinned on top, named) — never a replacement;
    a row-sharded rank takes the launcher's composition as given, silently."""
    mine = tmp_path / "mine.yml"; mine.write_text("pl_trainer_args:\n  precision: 32-true\n")
    got = cli.row_yaml(HOME, None, None, out_dir=str(tmp_path), runner_yaml=str(mine), mode="off")
    assert got == str(tmp_path / ("runner_composed_" + mine.name)) and yaml.safe_load(open(got)) == yaml.safe_load(open(os.path.join(HOME, modes.STOCK_YAML)))   # the one caller key is stock's own: pinned back
    assert cli.row_yaml(HOME, modes.LINES[("big", "tp")], None, out_dir=str(tmp_path), runner_yaml=str(mine), mode="big", as_given=True) == str(mine)
    err = capsys.readouterr().err                                                                 # off: composed, said once; a rank: silent (the launcher named it)
    assert err.count("\n") == 1 and "composed under the stock configuration (opt/openfold3_opt/stock_cueq_on_predict.yml) -> " + got in err, err
    assert "stock overrides runner-yaml keys: pl_trainer_args.precision ('32-true' -> 'bf16-mixed')" in err, err


def test_the_shipped_configuration_named_under_off_is_the_base_as_given(tmp_path, capsys):
    """The `default` tier (`pred --mode off --runner-yaml opt/openfold3_opt/shipped_predict.yml`): one of the tree's own configuration files
    named under off IS the configuration the stock caller runs — as given —; a further caller yaml is laid on IT."""
    shipped = os.path.join(HOME, modes.SHIPPED_YAML)
    assert cli.kit_base(HOME, shipped) and cli.kit_base(HOME, modes.SHIPPED_YAML) and not cli.kit_base(HOME, os.path.join(HOME, modes.STOCK_YAML))
    assert cli.row_yaml(HOME, None, None, out_dir=str(tmp_path), runner_yaml=shipped, mode="off") == shipped
    assert cli.row_yaml(HOME, None, None, out_dir=str(tmp_path), runner_yaml=[shipped], mode="off") == shipped
    err = capsys.readouterr().err
    assert err.count("\n") == 2 and "as given under off" in err and "composed" not in err, err
    rec = tmp_path / "recycles.yml"; rec.write_text("model_update:\n  custom:\n    architecture:\n      shared:\n        num_recycles: 10\n")
    got = cli.row_yaml(HOME, None, None, out_dir=str(tmp_path / "o"), runner_yaml=[shipped, str(rec)], mode="off")
    doc = yaml.safe_load(open(got))
    assert doc == {"model_update": {"presets": ["predict"], "custom": {"architecture": {"shared": {"num_recycles": 10}}}}}, doc   # shipped + the one key: DS4Sci at its shipped default, no cuEquivariance, no precision key


def test_the_fast_rows_kernels_off_base_composes_to_the_fast_members_configuration(tmp_path, capsys):
    """The kernels-off configuration under `fast` (the base the fast line's yaml is written from) composes to the fast member's configuration
    exactly: the model settings equal the member's but for the kernel flags the member leaves at upstream's off default, written out."""
    got = cli.row_yaml(HOME, modes.LINES[("fast", None)], modes.FAST_PRECISION, out_dir=str(tmp_path), runner_yaml=os.path.join(HOME, modes.KERNELS_OFF_YAML), mode="fast")
    doc, member = yaml.safe_load(open(got)), yaml.safe_load(open(os.path.join(HOME, modes.FAST_BF16_YAML)))
    assert doc["pl_trainer_args"] == member["pl_trainer_args"]
    ev = doc["model_update"]["custom"]["settings"]["memory"]["eval"]
    for f in ("use_cueq_triangle_kernels", "use_triton_triangle_kernels", "use_lma"):
        assert ev.pop(f) is False
    err = capsys.readouterr().err
    assert doc["model_update"] == member["model_update"] and "overrides runner-yaml keys" not in err and "fast sets runner-yaml keys the caller's yaml leaves unset: pl_trainer_args.precision='bf16-mixed'" in err, err


def test_the_row_sharded_launcher_lays_its_plan_on_a_callers_yaml_and_names_the_overrides(tmp_path, capsys):
    member = os.path.join(HOME, modes.BIG_BF16_C16_YAML)
    cal = tmp_path / "cal.yml"
    yaml.safe_dump({"experiment_settings": {"seeds": [42]}, "model_update": {"custom": {"settings": {"memory": {"eval": {"chunk_size": 4, "offload_inference": {"confidence_heads": True}, "per_sample_token_cutoff": 750}}}}}},
                   open(cal, "w"), sort_keys=False)
    pins = tp.pinned_yaml(str(cal), 32, str(tmp_path / "tp_predict.yml"), member=member, home=HOME)
    err = capsys.readouterr().err
    doc = yaml.safe_load(open(tmp_path / "tp_predict.yml")); ev = doc["model_update"]["custom"]["settings"]["memory"]["eval"]
    assert doc["experiment_settings"] == {"seeds": [42]} and ev["chunk_size"] == 32 and ev["offload_inference"] == {"confidence_heads": False, "msa_module": False} and ev["per_sample_token_cutoff"] is None
    assert all(ev[f] is False for f in tp.KERNEL_FLAGS_OFF) and doc["model_update"]["custom"]["architecture"]["pairformer"]["tune_chunk_size"] is False
    assert pins["chunk_size"] == 32 and pins["offload_inference"] is False and pins["base_yaml"] == str(cal)
    assert "note: runner yaml " in err and "composed under the tp line's required keys (chunk_size 32, chunk tuners off, fused pair kernels off, offload off, per-sample cutoffs null) and the big configuration (opt/openfold3_opt/big_bf16_c16_predict.yml) -> " in err, err
    assert doc["pl_trainer_args"] == {"precision": "bf16-mixed"} and "tp sets runner-yaml keys the caller's yaml leaves unset: pl_trainer_args.precision='bf16-mixed'" in err, err
    assert "tp overrides runner-yaml keys: " + EVAL + "chunk_size (4 -> 32); " + EVAL + "offload_inference.confidence_heads (True -> False); " + EVAL + "per_sample_token_cutoff (750 -> null)" in err, err
    tp.pinned_yaml(member, 16, str(tmp_path / "tp2.yml"), member=member, home=HOME)             # the member as the base (no caller yaml): the plan laid on silently
    assert capsys.readouterr().err == ""
    plan = yaml.safe_load(open(tmp_path / "tp2.yml"))["model_update"]["custom"]["settings"]["memory"]["eval"]
    assert plan["chunk_size"] == 16 and yaml.safe_load(open(member))["pl_trainer_args"]["precision"] == yaml.safe_load(open(tmp_path / "tp2.yml"))["pl_trainer_args"]["precision"] == "bf16-mixed"
