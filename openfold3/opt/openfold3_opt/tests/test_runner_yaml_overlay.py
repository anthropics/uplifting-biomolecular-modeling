"""A caller's `--runner-yaml` is an OVERLAY on every route (cli.row_yaml): laid on the configuration in force, the mode's own keys
pinned on top and named, never a replacement; the flag repeated lays several yamls in argv order (runner_yaml.merge_callers).
(a) no caller yaml: every route resolves to the member file itself, nothing written;
(b) a caller yaml holding ONLY model_update.custom.architecture.shared.num_recycles: exactly that key changes on every route — stock keeps
    bf16-mixed / cuEquivariance / the DS4Sci default, the det member its DS4Sci-off key, fast its kernels-off pins, big its own composition;
(c) a caller's per-seed yaml (a copy of the stock configuration + a seed list) AND a settings overlay both apply, in argv order;
plus the row-sharded launcher's base (tp: merge_callers then pinned_yaml) and the protocol line's recycles / trunk_passes."""
import os

import pytest
import yaml

from openfold3_opt import cli, modes, runner_yaml, tp

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RECYCLES = {"model_update": {"custom": {"architecture": {"shared": {"num_recycles": 10}}}}}
ROUTES = (  # (kwargs of cli.row_yaml, the member the route resolves without a caller yaml)
    (dict(mode="off"), modes.STOCK_YAML),
    (dict(mode="off", det=1), modes.STOCK_DET_YAML),
    (dict(line=modes.LINES[("exact", "cueq")], mode="exact"), modes.STOCK_YAML),
    (dict(line=modes.LINES[("exact", "cueq")], mode="exact", det=1), modes.STOCK_DET_YAML),
    (dict(line=modes.LINES[("fast", None)], precision=modes.FAST_PRECISION, mode="fast"), modes.FAST_BF16_YAML),
    (dict(line=modes.LINES[("big", "resident")], mode="big"), modes.BIG_BF16_C16_YAML),
    (dict(line=modes.LINES[("big", "tp")], mode="big"), modes.BIG_BF16_C16_YAML),
)


def _leaves(doc, path=()):
    out = {}
    for k, v in (doc or {}).items():
        if isinstance(v, dict):
            out.update(_leaves(v, path + (k,)))
        else:
            out[".".join(path + (k,))] = v
    return out


def _recycles_yaml(tmp_path, n=10, name="settings.yml"):
    p = tmp_path / name
    yaml.safe_dump({"model_update": {"custom": {"architecture": {"shared": {"num_recycles": n}}}}}, open(p, "w"), sort_keys=False)
    return str(p)


@pytest.mark.parametrize("kw,member", ROUTES, ids=[f"{k.get('mode')}-det{k.get('det', 0)}-{getattr(k.get('line'), 'name', None) or '-'}" for k, _ in ROUTES])
def test_a_no_caller_yaml_resolves_the_member_file_itself_on_every_route(tmp_path, capsys, kw, member):
    """(a): nothing written, nothing said — the argv every route builds names the member file itself."""
    for ry in (None, [], ()):
        assert cli.row_yaml(HOME, runner_yaml=ry, out_dir=str(tmp_path / "o"), **kw) == os.path.join(HOME, member)
    assert not (tmp_path / "o").exists() and capsys.readouterr().err == ""


@pytest.mark.parametrize("kw,member", ROUTES, ids=[f"{k.get('mode')}-det{k.get('det', 0)}-{getattr(k.get('line'), 'name', None) or '-'}" for k, _ in ROUTES])
def test_b_a_recycles_only_yaml_changes_exactly_that_key_on_every_route(tmp_path, capsys, kw, member):
    """(b): the composed document = the route's own configuration (the member's keys; on the graphed fast line its kernel-flag pins too) + the ONE key."""
    got = cli.row_yaml(HOME, runner_yaml=_recycles_yaml(tmp_path), out_dir=str(tmp_path / "o"), **kw)
    assert got == str(tmp_path / "o" / "runner_composed_settings.yml")
    want = _leaves(runner_yaml.line_pins(kw.get("line"), os.path.join(HOME, member)))          # the member's execution keys (+ the graphed line's kernel flags off)
    want["model_update.custom.architecture.shared.num_recycles"] = 10
    assert _leaves(yaml.safe_load(open(got))) == want
    member_leaves = _leaves(yaml.safe_load(open(os.path.join(HOME, member))))
    assert all(want[k] == v for k, v in member_leaves.items())                                   # every member key intact (stock: bf16-mixed, cuEquivariance on; the det member: DS4Sci off; big: chunk 16, tuner off)
    d = cli.protocol_dims(got)
    assert d["recycles"] == 10 and d["trunk_passes"] == 11 and d["diffusion_steps"] == 200        # the protocol line keys on the effective recycles
    err = capsys.readouterr().err
    word = "stock" if kw["mode"] == "off" else kw["mode"]
    assert f"composed under the {word} configuration ({member}) -> {got}" in err and "overrides" not in err, err
    assert "caller keys kept as written that differ from the member: model_update.custom.architecture.shared.num_recycles=10" in err, err


def test_b_stock_keeps_its_kernels_and_precision_against_a_caller_that_says_otherwise(tmp_path, capsys):
    """Under off the stock configuration's own keys are pinned over a caller's (named): a caller cannot turn the stock arm into another configuration by accident."""
    mine = tmp_path / "mine.yml"
    yaml.safe_dump({"pl_trainer_args": {"precision": "32-true"}, "model_update": {"custom": {"settings": {"memory": {"eval": {"use_cueq_triangle_kernels": False, "chunk_size": 64}}},
                                                                              "architecture": {"shared": {"num_recycles": 7}}}}}, open(mine, "w"), sort_keys=False)
    doc = yaml.safe_load(open(cli.row_yaml(HOME, runner_yaml=[str(mine)], out_dir=str(tmp_path / "o"), mode="off")))
    ev = doc["model_update"]["custom"]["settings"]["memory"]["eval"]
    assert doc["pl_trainer_args"]["precision"] == "bf16-mixed" and ev["use_cueq_triangle_kernels"] is True and "use_deepspeed_evo_attention" not in ev   # stock's keys; DS4Sci at its shipped default (absent = on)
    assert ev["chunk_size"] == 64 and doc["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 7                                        # the caller's other keys as written
    err = capsys.readouterr().err
    assert "stock overrides runner-yaml keys: pl_trainer_args.precision ('32-true' -> 'bf16-mixed'); model_update.custom.settings.memory.eval.use_cueq_triangle_kernels (False -> True)" in err, err


@pytest.mark.parametrize("kw,member", [ROUTES[1], ROUTES[3], ROUTES[4], ROUTES[5]], ids=["off-det1", "exact-det1", "fast", "big-resident"])
def test_c_a_seeded_yaml_and_a_settings_overlay_both_apply_in_argv_order(tmp_path, capsys, kw, member):
    """(c): a caller's per-seed runner yaml (the stock det configuration + experiment_settings.seeds [s]) and a settings overlay
    (num_recycles) are BOTH laid on, in argv order (a later yaml's keys on top of an earlier one's), then the mode's keys pinned."""
    seeded = tmp_path / "runner_5.yml"
    sdoc = yaml.safe_load(open(os.path.join(HOME, modes.STOCK_DET_YAML))); sdoc["experiment_settings"] = {"seeds": [5]}
    sdoc.setdefault("model_update", {}).setdefault("custom", {}).setdefault("architecture", {}).setdefault("shared", {})["num_recycles"] = 3   # an earlier yaml's key the later overlay overrides (argv order)
    yaml.safe_dump(sdoc, open(seeded, "w"), sort_keys=False)
    settings = _recycles_yaml(tmp_path)
    got = cli.row_yaml(HOME, runner_yaml=[str(seeded), settings], out_dir=str(tmp_path / "o"), **kw)
    doc = yaml.safe_load(open(got))
    assert doc["experiment_settings"] == {"seeds": [5]} and doc["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 10
    want = _leaves(runner_yaml.line_pins(kw.get("line"), os.path.join(HOME, member)))
    assert all(_leaves(doc)[k] == v for k, v in want.items())                                     # the route's own keys pinned on top of both
    err = capsys.readouterr().err
    assert "laid in argv order" in err and "runner_5.yml" in err and "settings.yml overrides: model_update.custom.architecture.shared.num_recycles (3 -> 10)" in err, err
    merged = tmp_path / "o" / "runner_callers_2_settings.yml"                                    # the two callers laid in argv order; composed under the route unless it already carries every key the mode writes (the stock det routes: the seeded yaml restates them — the file runs as is)
    assert merged.is_file() and got in (str(merged), str(tmp_path / "o" / "runner_composed_runner_callers_2_settings.yml"))
    assert (got == str(merged)) == (kw["mode"] in ("off", "exact")), (got, kw)
    # the reverse argv order: the seeded yaml's 3 wins
    got2 = cli.row_yaml(HOME, runner_yaml=[settings, str(seeded)], out_dir=str(tmp_path / "o2"), **kw)
    assert yaml.safe_load(open(got2))["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 3


def test_the_stock_family_named_beside_an_overlay_follows_the_det_level_and_takes_the_overlay(tmp_path):
    """Naming the stock configuration file itself (as a caller pinning the digest would) names the FAMILY — the det member as without the flag —; the overlay is laid on that member."""
    got = cli.row_yaml(HOME, runner_yaml=[os.path.join(HOME, modes.STOCK_YAML), _recycles_yaml(tmp_path)], out_dir=str(tmp_path / "o"), mode="off", det=1)
    doc = yaml.safe_load(open(got))
    assert doc["model_update"]["custom"]["settings"]["memory"]["eval"] == {"use_deepspeed_evo_attention": False, "use_cueq_triangle_kernels": True}
    assert doc["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 10


def test_the_row_sharded_launchers_base_is_the_callers_laid_in_argv_order_then_pinned(tmp_path, capsys):
    """tp: the launcher lays several caller yamls in argv order (merge_callers), composes that under the big configuration and the line's plan
    (pinned_yaml) and hands the ranks the ONE pinned file as given — num_recycles survives the plan; the plan's keys win."""
    seeded = tmp_path / "runner_42.yml"
    sdoc = yaml.safe_load(open(os.path.join(HOME, modes.BIG_BF16_C16_YAML))); sdoc["experiment_settings"] = {"seeds": [42]}
    yaml.safe_dump(sdoc, open(seeded, "w"), sort_keys=False)
    member = cli.row_yaml(HOME, modes.LINES[("big", "tp")], mode="big")
    base = runner_yaml.merge_callers(HOME, [str(seeded), _recycles_yaml(tmp_path)], str(tmp_path / "o"))
    out = tmp_path / "o" / tp.PINNED_YAML
    pins = tp.pinned_yaml(base, 64, str(out), member=member, home=HOME)
    doc = yaml.safe_load(open(out))
    ev = doc["model_update"]["custom"]["settings"]["memory"]["eval"]
    assert doc["experiment_settings"] == {"seeds": [42]} and doc["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 10
    assert ev["chunk_size"] == 64 and pins["chunk_size"] == 64 and all(ev[f] is False for f in tp.KERNEL_FLAGS_OFF)
    assert cli.row_yaml(HOME, modes.LINES[("big", "tp")], runner_yaml=[str(out)], mode="big", as_given=True) == str(out)   # a rank: the launcher's file, as given
    assert cli.protocol_dims(str(out))["trunk_passes"] == 11


def test_the_flag_repeats_and_resolves_in_argv_order(tmp_path):
    a = cli.build_parser().parse_args(["pred", "--mode", "off", "--query-json", "q.json", "--output-dir", "o", "--runner-yaml", "a.yml", "--runner-yaml", "b.yml"])
    assert a.runner_yaml == ["a.yml", "b.yml"]
    assert cli.build_parser().parse_args(["pred", "--mode", "off", "--query-json", "q.json", "--output-dir", "o"]).runner_yaml is None
    assert cli.build_parser().parse_args(["check", "--mode", "exact", "--runner-yaml", "a.yml"]).runner_yaml == ["a.yml"]
    (tmp_path / "a.yml").write_text("x: 1\n"); (tmp_path / "b.yml").write_text("y: 2\n")
    assert cli.resolve_runner_yaml([str(tmp_path / "a.yml"), str(tmp_path / "b.yml")], HOME) == [str(tmp_path / "a.yml"), str(tmp_path / "b.yml")]
    assert cli.resolve_runner_yaml([], HOME) is None and cli.resolve_runner_yaml(None, HOME) is None
    assert cli.resolve_runner_yaml(modes.SHIPPED_YAML, HOME) == os.path.join(HOME, modes.SHIPPED_YAML)           # one path (a str) still resolves as before
    with pytest.raises(ValueError):
        cli.resolve_runner_yaml([str(tmp_path / "missing.yml")], HOME)
    assert cli.caller_yamls(None) == [] and cli.caller_yamls("p") == ["p"] and cli.caller_yamls(["p", "q"]) == ["p", "q"]
    assert cli.composed_key("p") == "p" and cli.composed_key(["p"]) == "p" and cli.composed_key(["p", "q"]) == ("p", "q")
