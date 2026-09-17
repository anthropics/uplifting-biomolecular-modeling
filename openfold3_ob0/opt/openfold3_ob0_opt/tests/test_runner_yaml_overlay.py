"""`--runner-yaml` is an OVERLAY on every route: no caller yaml = the route's member untouched (nothing composed, nothing written, the
same path as before); a caller yaml holding ONLY `model_update.custom.architecture.shared.num_recycles` changes exactly that key of the route's
effective configuration — under `off` (the stock member stays: bf16-mixed, cuEquivariance + Triton, DS4Sci off, chunk plan pinned), `exact`,
`fast`, `big` resident (its own kernels-off / chunk-16 composition intact) and the ×P line's member before the launcher pins its plan on it;
the flag repeats and the files merge in the order given (a per-seed copy of the stock yaml + a settings overlay both apply); the protocol line
reads the effective recycle count off the composed file (`recycles=10 trunk_passes=11`)."""
import os

import yaml

from openfold3_ob0_opt import cli, modes
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
R10 = {"model_update": {"custom": {"architecture": {"shared": {"num_recycles": 10}}}}}


def _routes():
    """(label, row_yaml kwargs, the member the route requires) for every route that builds a runner yaml."""
    ex, fast = modes.LINES[("exact", modes.EXACT_LINE)], modes.LINES[("fast", None)]
    out = [("off", dict(line=None, mode="off", det=0), modes.STOCK_YAML), ("off det1", dict(line=None, mode="off", det=1), modes.STOCK_DET_YAML),
           ("exact", dict(line=ex, mode="exact", det=0), modes.STOCK_YAML), ("exact det1", dict(line=ex, mode="exact", det=1), modes.STOCK_DET_YAML),
           ("fast", dict(line=fast, mode="fast", det=0), modes.STOCK_YAML)]
    for bl in modes.BIG_LINES:
        ln = modes.LINES[("big", bl)]
        out.append((f"big/{bl}", dict(line=ln, mode="big", det=0), ln.runner_yaml))
    return out


def _r10(tmp_path, name="recycles_10.yml"):
    p = tmp_path / name; p.write_text(yaml.safe_dump(R10, sort_keys=False)); return str(p)


def _effective(path):
    """A runner yaml's document with the route's eval kernel triple made explicit (as upstream resolves the file) — the comparison form."""
    doc = yaml.safe_load(open(path)) or {}
    node = doc
    for k in cli.KERNEL_FLAGS_PATH:
        node = node.setdefault(k, {}) if isinstance(node.get(k), dict) else node.setdefault(k, {})
    node.update(cli.yaml_kernel_flags(path))
    return doc


def test_no_caller_yaml_is_the_member_untouched_on_every_route(capsys):
    for label, kw, member in _routes():
        assert cli.row_yaml(HOME, **kw) == os.path.join(HOME, member), label                  # the member's own file: the same path as before the overlay rule
        assert cli.row_yaml(HOME, runner_yaml=[], **kw) == os.path.join(HOME, member), label
    assert capsys.readouterr().err == ""                                                       # nothing composed, no note


def test_a_recycles_only_overlay_changes_exactly_that_key_on_every_route(tmp_path, capsys):
    r10 = _r10(tmp_path)
    for label, kw, member in _routes():
        got = cli.row_yaml(HOME, runner_yaml=r10, **kw)
        assert os.path.basename(got).startswith(cli.COMPOSED_PREFIX), (label, got)
        want = _effective(os.path.join(HOME, member))
        want["model_update"].setdefault("custom", {}).setdefault("architecture", {}).setdefault("shared", {})["num_recycles"] = 10
        if kw["mode"] != "off":
            want.setdefault("pl_trainer_args", {}).setdefault("precision", "32-true")          # required_overlay states the trainer precision explicitly on the kit routes (the member's own here)
        assert _effective(got) == want, label                                                 # that key and nothing else: kernels, precision, chunk plan, presets all the member's
        assert cli.protocol_dims(got)["recycles"] == 10 and cli.protocol_dims(got)["trunk_passes"] == 11, label
        assert "recycles=10 trunk_passes=11 diffusion_steps=200 diffusion_samples=5" in cli.protocol_line(got, []), label
        err = capsys.readouterr().err
        assert err.count("\n") == 1 and ("laid over the base configuration" in err if kw["mode"] == "off" else "composed under" in err), (label, err)
    assert yaml.safe_load(open(r10)) == R10                                                    # the caller's file untouched


def test_the_stock_arm_keeps_its_configuration_under_a_recycles_overlay(tmp_path, capsys):
    """(b) spelled out for `off`: bf16-mixed, cuEquivariance + Triton on, DS4Sci off, the pinned chunk plan — all the stock member's — plus num_recycles 10."""
    got = yaml.safe_load(open(cli.row_yaml(HOME, None, runner_yaml=_r10(tmp_path), mode="off", det=1)))
    ev = got["model_update"]["custom"]["settings"]["memory"]["eval"]
    assert got["pl_trainer_args"]["precision"] == "bf16-mixed" and ev["use_cueq_triangle_kernels"] is True and ev["use_triton_triangle_kernels"] is True
    assert ev["use_deepspeed_evo_attention"] is False and ev["tune_chunk_size"] is False and ev["chunk_size"] == 1024
    assert got["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 10 and got["model_update"]["presets"] == ["predict"]
    capsys.readouterr()


def test_big_resident_keeps_its_composition_under_a_recycles_overlay(tmp_path, capsys):
    ln = modes.LINES[("big", "resident")]
    got = yaml.safe_load(open(cli.row_yaml(HOME, ln, runner_yaml=_r10(tmp_path), mode="big")))
    member = yaml.safe_load(open(os.path.join(HOME, ln.runner_yaml)))
    assert got["model_update"]["custom"]["settings"] == member["model_update"]["custom"]["settings"] and got["pl_trainer_args"] == member["pl_trainer_args"]
    assert got["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 10
    capsys.readouterr()


def test_two_overlays_apply_in_order_the_seeded_stock_copy_then_the_settings(tmp_path, capsys):
    """(c) a per-seed copy of the stock yaml (experiment_settings.seeds) AND a settings overlay (num_recycles): both apply on every route;
    under big the line's own kernels / chunk plan win over the stock copy's (named), the seeds and the recycle count pass through; a later
    overlay's key wins over an earlier one's."""
    stock = yaml.safe_load(open(os.path.join(HOME, modes.STOCK_YAML)))
    seeded = tmp_path / "runner_42.yml"; seeded.write_text(yaml.safe_dump({**stock, "experiment_settings": {"seeds": [42]}}, sort_keys=False))
    r10, r3 = _r10(tmp_path), tmp_path / "recycles_3.yml"
    r3.write_text(yaml.safe_dump({"model_update": {"custom": {"architecture": {"shared": {"num_recycles": 3}}}}}))
    for label, kw, member in _routes():
        got = yaml.safe_load(open(cli.row_yaml(HOME, runner_yaml=[str(seeded), r10], **kw)))
        assert got["experiment_settings"]["seeds"] == [42] and got["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 10, label
        mem = yaml.safe_load(open(os.path.join(HOME, member)))
        mev = mem["model_update"].get("custom", {}).get("settings", {}).get("memory", {}).get("eval", {})
        gev = got["model_update"]["custom"].get("settings", {}).get("memory", {}).get("eval", {})
        assert all(gev.get(k) == v for k, v in mev.items()), label                            # the route's own eval keys win over the stock copy's (big: kernels off, chunk 16)
        later = yaml.safe_load(open(cli.row_yaml(HOME, runner_yaml=[r10, str(r3)], **kw)))
        assert later["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 3, label   # in the order given: the later file's key wins
    capsys.readouterr()


def test_the_tp_launcher_pins_its_plan_on_the_composed_overlay(tmp_path, capsys):
    from openfold3_ob0_opt import tp
    ln = modes.LINES[("big", "tp")]
    composed = cli.row_yaml(HOME, ln, runner_yaml=[_r10(tmp_path)], mode="big")            # cmd_pred composes ONCE for the launcher; tp.launch pins the chunk plan on that file; the ranks take it as given
    tp.pinned_yaml(composed, 128, str(tmp_path / "tp.yml"))
    tdoc = yaml.safe_load(open(tmp_path / "tp.yml"))
    assert tdoc["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 10 and tdoc["model_update"]["custom"]["settings"]["memory"]["eval"]["chunk_size"] == 128
    assert cli.row_yaml(HOME, ln, runner_yaml=[str(tmp_path / "tp.yml")], mode="big", as_given=True) == str(tmp_path / "tp.yml")   # a rank: verbatim
    capsys.readouterr()


def test_the_flag_repeats_and_resolves_every_file(tmp_path):
    r10, r3 = _r10(tmp_path), _r10(tmp_path, "b.yml")
    a = cli.build_parser().parse_args(["pred", "--query-json", "q.json", "--output-dir", "o", "--runner-yaml", r10, "--runner-yaml", r3])
    assert a.runner_yaml == [r10, r3]
    assert cli.resolve_runner_yaml(a.runner_yaml, HOME) == [r10, r3] and cli.resolve_runner_yaml(None, HOME) is None
    assert os.path.samefile(cli.resolve_runner_yaml(modes.SHIPPED_YAML, HOME), os.path.join(HOME, modes.SHIPPED_YAML))   # one tree-relative name: as before (the working directory first, then the tree — one file either way)
    c = cli.build_parser().parse_args(["check", "--runner-yaml", r10])
    assert c.runner_yaml == [r10]
    assert cli.caller_yamls(None) == [] and cli.caller_yamls(r10) == [r10] and cli.caller_yamls([r10, "", r3]) == [r10, r3]
