"""A caller's `--runner-yaml` is COMPOSED UNDER the route's required configuration on every route (cli.compose_runner_yaml via cli.row_yaml):
the caller's user-level keys stay (template_preprocessor_settings, experiment_settings, …), the route's required keys win (the eval kernel triple
as upstream resolves the member file, the trainer precision, every key the member writes), ONE note line names the composition and the caller
keys it overrode — never silent, never a refusal — on every kit mode; under `off` a caller's yaml is laid OVER the stock member (an overlay: its keys win,
the member's other keys stay; one note line says so) — a caller yaml is never a replacement on any route. Named cases: either stock configuration
file under off / exact runs the member itself (the stock family), upstream's shipped preset under off is the BASE in place of the stock member (the
`default` arm), and a process the run started (as_given: the ×P ranks) takes the yaml its launcher composed. The flag repeats: every file an overlay,
merged in the order given."""
import os

import yaml

from openfold3_ob0_opt import cli, modes
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
TEMPL = {"structure_directory": "/pack/templates/structures", "structure_file_format": "cif", "fetch_missing_structures": False}


def _caller(tmp_path, **eval_flags):
    """A caller yaml as an outside caller renders it: template settings + a seed list + (optionally) eval kernel flags of its own."""
    doc = {"template_preprocessor_settings": dict(TEMPL), "experiment_settings": {"seeds": [1, 2, 3]}, "model_update": {"presets": ["predict"]}}
    if eval_flags:
        doc["model_update"]["custom"] = {"settings": {"memory": {"eval": dict(eval_flags)}}}
    p = tmp_path / "caller.yml"; p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return str(p)


def _flags(doc):
    node = doc
    for k in cli.KERNEL_FLAGS_PATH:
        node = (node or {}).get(k) or {}
    return {k: v for k, v in node.items() if k in cli.KERNEL_FLAG_DEFAULTS}


def test_the_exact_route_composes_the_caller_yaml_under_the_stock_member(tmp_path, capsys):
    own = _caller(tmp_path, use_deepspeed_evo_attention=True, use_cueq_triangle_kernels=False)      # a conflicting kernel key: the stock member wins, named
    for mode, det, member in (("exact", 0, modes.STOCK_YAML), ("exact", 1, modes.STOCK_DET_YAML)):
        line = modes.LINES[("exact", modes.EXACT_LINE)]
        out = cli.row_yaml(HOME, line, runner_yaml=own, mode=mode, det=det)
        assert not out.startswith(str(tmp_path)) and os.path.basename(out).startswith(cli.COMPOSED_PREFIX), out   # a temporary directory, never the output directory
        doc = yaml.safe_load(open(out))
        assert doc["template_preprocessor_settings"] == TEMPL and doc["experiment_settings"]["seeds"] == [1, 2, 3]     # the caller's user-level keys stay
        assert _flags(doc) == cli.yaml_kernel_flags(os.path.join(HOME, member))                                    # the member's kernel triple as upstream resolves it
        mdoc = yaml.safe_load(open(os.path.join(HOME, member))) or {}
        want_prec = (mdoc.get("pl_trainer_args") or {}).get("precision", "32-true")
        assert doc["pl_trainer_args"]["precision"] == want_prec                                                      # the member's trainer precision, explicit
        err = capsys.readouterr().err
        assert err.count("composed under") == 1 and f"({member})" in err and "use_deepspeed_evo_attention (True -> False)" in err, err   # ONE note line naming the overridden caller key


def test_nothing_overridden_is_named_none(tmp_path, capsys):
    own = _caller(tmp_path)                                                                          # template + seeds only: nothing to override
    ex = modes.LINES[("exact", modes.EXACT_LINE)]
    out = cli.row_yaml(HOME, ex, runner_yaml=own, mode="exact", det=1)
    doc = yaml.safe_load(open(out))
    assert doc["template_preprocessor_settings"] == TEMPL and _flags(doc) == cli.yaml_kernel_flags(os.path.join(HOME, modes.STOCK_DET_YAML))
    assert "overrides runner-yaml keys: none" in capsys.readouterr().err


def test_off_lays_a_callers_yaml_over_the_stock_member(tmp_path, capsys):
    """`off` = the stock configuration with the caller's runner yaml laid OVER it (an overlay, never a replacement): the caller's keys win — a caller
    turning DS4Sci on gets it, named on the one note line —, every stock key the caller does not set stays (the cuEquivariance / Triton flags, bf16-mixed,
    the pinned chunk plan); the composed file is written into a temporary directory; the caller's file is untouched."""
    own = _caller(tmp_path, use_deepspeed_evo_attention=True)
    stock = yaml.safe_load(open(os.path.join(HOME, modes.STOCK_YAML)))
    for det in (0, 1):
        got = cli.row_yaml(HOME, None, runner_yaml=own, mode="off", det=det)
        assert got != own and os.path.basename(got).startswith(cli.COMPOSED_PREFIX + "off_"), got
        doc = yaml.safe_load(open(got))
        assert doc["template_preprocessor_settings"] == TEMPL and doc["experiment_settings"]["seeds"] == [1, 2, 3]      # the caller's keys
        assert _flags(doc) == {**_flags(stock), "use_deepspeed_evo_attention": True}                                  # the caller's flag wins; the stock member's other flags stay
        assert doc["pl_trainer_args"] == stock["pl_trainer_args"]                                                     # bf16-mixed: the stock member's, kept
        err = capsys.readouterr().err
        assert err.count("\n") == 1 and f"note: runner yaml {own} laid over the base configuration ({modes.STOCK_YAML}) under off" in err and "use_deepspeed_evo_attention (False -> True)" in err and "composed under" not in err, err
    tree_rel = modes.KERNELS_OFF_YAML                                                                                # a tree-relative name resolves against the tree: the same composition as the absolute path
    assert yaml.safe_load(open(cli.row_yaml(HOME, None, runner_yaml=tree_rel, mode="off"))) == yaml.safe_load(open(cli.row_yaml(HOME, None, runner_yaml=os.path.join(HOME, tree_rel), mode="off")))
    capsys.readouterr()


def test_the_kit_lines_compose_under_their_own_member(tmp_path, capsys):
    own = _caller(tmp_path, use_deepspeed_evo_attention=True)
    fast = modes.LINES[("fast", None)]
    for member in (modes.STOCK_YAML,):
        assert os.path.isfile(os.path.join(HOME, member)), member
        out = cli.row_yaml(HOME, fast, runner_yaml=own, mode="fast")
        doc = yaml.safe_load(open(out))
        assert _flags(doc) == cli.yaml_kernel_flags(os.path.join(HOME, member)) and doc["template_preprocessor_settings"] == TEMPL
        assert cli.line_yaml_check(HOME, fast, out, "pred --mode fast") is None                    # the composed yaml carries the line's capture-safe triple: no refusal
    for bl in modes.BIG_LINES:
        ln = modes.LINES[("big", bl)]
        member = cli.line_member(ln, None, "big")
        assert member == ln.runner_yaml                                                       # every big line names its runner yaml
        assert os.path.isfile(os.path.join(HOME, member)), member
        out = cli.row_yaml(HOME, ln, runner_yaml=own, mode="big")
        doc = yaml.safe_load(open(out)); mdoc = yaml.safe_load(open(os.path.join(HOME, member))) or {}
        assert _flags(doc) == cli.yaml_kernel_flags(os.path.join(HOME, member)) and doc["template_preprocessor_settings"] == TEMPL
        mem = (((mdoc.get("model_update") or {}).get("custom") or {}).get("settings") or {}).get("memory") or {}
        got = (((doc.get("model_update") or {}).get("custom") or {}).get("settings") or {}).get("memory") or {}
        assert all(got.get("eval", {}).get(k) == v for k, v in (mem.get("eval") or {}).items())      # the member's chunk plan / offload keys win too
    capsys.readouterr()


def test_the_named_exceptions_run_a_file_as_given(tmp_path, capsys):
    ex = modes.LINES[("exact", modes.EXACT_LINE)]
    assert cli.row_yaml(HOME, ex, runner_yaml=os.path.join(HOME, modes.STOCK_YAML), mode="exact", det=1) == os.path.join(HOME, modes.STOCK_DET_YAML)   # the stock family: the member by det level
    own = _caller(tmp_path, use_deepspeed_evo_attention=True)
    assert cli.row_yaml(HOME, modes.LINES[("big", "tp")], runner_yaml=own, mode="big", as_given=True) == own                    # a rank: its launcher's file, untouched
    assert capsys.readouterr().err == ""                                                                                             # neither prints a composition note


def test_without_a_caller_yaml_the_route_runs_its_member():
    assert cli.row_yaml(HOME, None, mode="off") == os.path.join(HOME, modes.STOCK_YAML)
    assert cli.row_yaml(HOME, None, mode="off", det=1) == os.path.join(HOME, modes.STOCK_DET_YAML)
    assert cli.row_yaml(HOME, modes.LINES[("fast", None)], mode="fast") == os.path.join(HOME, modes.STOCK_YAML)
    for bl in modes.BIG_LINES:
        ln = modes.LINES[("big", bl)]
        assert cli.row_yaml(HOME, ln, mode="big") == os.path.join(HOME, ln.runner_yaml)


def test_deep_overlay_names_every_changed_leaf():
    merged, changed = cli.deep_overlay({"a": {"b": 1, "c": 2}, "d": [1], "e": 5}, {"a": {"b": 9}, "d": [2], "f": 0})
    assert merged == {"a": {"b": 9, "c": 2}, "d": [2], "e": 5, "f": 0} and changed == ["a.b (1 -> 9)", "d ([1] -> [2])"]


def test_the_shipped_yaml_is_the_base_under_off_only(tmp_path, capsys):
    """modes.SHIPPED_YAML (upstream's predict preset as shipped) under `off` is the BASE configuration in place of the stock member — by path or as a
    staged copy (digest) —, NAMED on the note line (the `default` arm: byte-equivalent to no runner yaml; nothing written when it is the only caller yaml,
    a further caller yaml is laid OVER it); under any kit line it is a caller yaml like any other, composed under the member."""
    import shutil
    shipped = os.path.join(HOME, modes.SHIPPED_YAML)
    for det in (0, 1):
        assert cli.row_yaml(HOME, None, runner_yaml=shipped, mode="off", det=det) == shipped
        assert f"under off = upstream's predict preset as shipped ({modes.SHIPPED_YAML}): the base configuration in place of the stock member" in capsys.readouterr().err
    copy = tmp_path / "elsewhere.yml"; shutil.copy(shipped, copy)
    assert cli.named_configuration(HOME, str(copy)) == modes.SHIPPED_YAML and cli.row_yaml(HOME, None, runner_yaml=str(copy), mode="off") == str(copy)   # a staged copy: named, the base from where it is
    r10 = tmp_path / "recycles.yml"; r10.write_text("model_update:\n  custom:\n    architecture:\n      shared:\n        num_recycles: 10\n")
    got = cli.row_yaml(HOME, None, runner_yaml=[shipped, str(r10)], mode="off")                                          # the `default` arm + a settings overlay: shipped + that key, nothing of the stock member's
    assert yaml.safe_load(open(got)) == {"model_update": {"presets": ["predict"], "custom": {"architecture": {"shared": {"num_recycles": 10}}}}}
    capsys.readouterr()
    ex = modes.LINES[("exact", modes.EXACT_LINE)]
    got = cli.row_yaml(HOME, ex, runner_yaml=shipped, mode="exact", det=1)
    assert os.path.basename(got).startswith(cli.COMPOSED_PREFIX) and cli.yaml_kernel_flags(got) == cli.yaml_kernel_flags(os.path.join(HOME, modes.STOCK_DET_YAML))   # under a kit line: composed
    assert cli.named_configuration(HOME, _caller(tmp_path)) is None


def test_presets_merge_as_an_ordered_union_everything_else_member_wins(tmp_path, capsys):
    """upstream's presets apply BEFORE explicit keys: a caller's extra preset (low_mem) survives composition on every route (member's presets first,
    the caller's extras after), while the member's explicit required keys still win — under off the stock member's kernel triple / precision, under
    the tp line its pinned overlay (offload of the msa / template modules forced off, named)."""
    import yaml
    from openfold3_ob0_opt import tp
    own = tmp_path / "lowmem.yml"
    own.write_text("model_update:\n  presets:\n    - predict\n    - low_mem\n  custom:\n    settings:\n      memory:\n        eval:\n          offload_inference:\n            msa_module: true\n")
    merged, changed = cli.deep_overlay({"model_update": {"presets": ["predict", "low_mem"]}}, {"model_update": {"presets": ["predict"]}})
    assert merged["model_update"]["presets"] == ["predict", "low_mem"] and changed == []                  # nothing of the caller's changed: not named
    merged, changed = cli.deep_overlay({"model_update": {"presets": ["low_mem"]}}, {"model_update": {"presets": ["predict"]}})
    assert merged["model_update"]["presets"] == ["predict", "low_mem"] and changed == ["model_update.presets (['low_mem'] -> ['predict', 'low_mem'])"]
    ex = modes.LINES[("exact", modes.EXACT_LINE)]
    out = cli.row_yaml(HOME, ex, runner_yaml=str(own), mode="exact")            # a kit mode: composed under the stock member
    doc = yaml.safe_load(open(out))
    assert doc["model_update"]["presets"] == ["predict", "low_mem"]
    assert _flags(doc) == cli.yaml_kernel_flags(os.path.join(HOME, modes.STOCK_YAML))                       # the stock member's triple still explicit
    assert doc["model_update"]["custom"]["settings"]["memory"]["eval"]["offload_inference"]["msa_module"] is True   # the stock member is silent on it: the caller's key stays
    pins = tp.pinned_yaml(out, 128, str(tmp_path / "tp.yml"))                                                # the tp line's pinned overlay on top (tp.launch's composition)
    tdoc = yaml.safe_load(open(tmp_path / "tp.yml"))
    assert tdoc["model_update"]["presets"] == ["predict", "low_mem"]
    ev = tdoc["model_update"]["custom"]["settings"]["memory"]["eval"]
    assert ev["offload_inference"]["msa_module"] is False and ev["offload_inference"]["template_module"] is False and ev["chunk_size"] == 128 and ev["tune_chunk_size"] is False
    assert any("offload_inference.msa_module (True -> False)" in o for o in (pins.get("overrides") or pins.get("overridden") or []))
