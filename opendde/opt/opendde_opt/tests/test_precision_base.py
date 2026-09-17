"""The stock base: `--dtype bf16` + `LAYERNORM_TYPE=fast_layernorm` (`bf16_fastln`) on every route and mode — the base's `--dtype` passed unless the
caller states one, its environment exported unless already set, a caller's own value run as requested with a NOTE (never a refusal); the LayerNorm
census read from what upstream bound and failing loud by name; the frozen gate reading booleans as the stock CLI parses them."""
import json
import os
import re
import sys
import types

import pytest

from opendde_opt import cli, frozen, lncensus, settings, stack
from opendde_opt.tests import _stubs


def test_the_stock_base_is_one_pair_on_every_route():
    assert settings.STOCK_BASE == "bf16_fastln" and settings.BASE_DTYPE == "bf16" and settings.STOCK_ENV == {"LAYERNORM_TYPE": "fast_layernorm"}
    assert not hasattr(settings, "PRESETS") and not hasattr(settings, "cli_args")       # no settings table in this tree: upstream's flags pass through
    import argparse
    ap = argparse.ArgumentParser(); cli._common_pred(ap)
    a0 = ap.parse_args(["-i", "q", "-o", "o"]); a1 = ap.parse_args(["-i", "q", "-o", "o", "--dtype", "fp32"])
    assert settings.base_args(a0) == ["--dtype", "bf16"] and settings.effective(a0, [], _stubs.TREE)["dtype"] == "bf16"   # nothing stated: the base's flag is what the engine gets
    assert settings.base_args(a1) == [] and settings.stated_args(a1) == ["--dtype", "fp32"] and settings.effective(a1, [], _stubs.TREE)["dtype"] == "fp32"   # the caller's value wins, verbatim
    assert settings.base_args(a0, ["--dtype", "fp32"]) == [] and settings.effective(a0, ["--dtype", "fp32"], _stubs.TREE)["dtype"] == "fp32"          # also when passed after `--`
    d = settings.describe(settings.effective(a0, [], _stubs.TREE), {}, _stubs.TREE, environ={})
    assert d["base"] == "bf16_fastln" and d["stock_env"] == {"LAYERNORM_TYPE": "fast_layernorm"} and d["effective"]["dtype"] == "bf16" and d["stated"] == {}


def test_stock_env_is_exported_when_unset_and_a_callers_value_runs_with_a_note():
    env = {}
    assert settings.stock_env(env) == {"LAYERNORM_TYPE": "fast_layernorm"} and settings.stock_env_notes(env) == [] and settings.ln_requested(env) is False
    assert settings.apply_stock_env(env) == {"LAYERNORM_TYPE": "fast_layernorm"} and env == {"LAYERNORM_TYPE": "fast_layernorm"} and settings.ln_requested(env)
    assert settings.stock_env(env) == {} and settings.stock_env_notes(env) == []
    other = {"LAYERNORM_TYPE": "torch"}
    assert settings.stock_env(other) == {} and settings.apply_stock_env(other) == {} and other == {"LAYERNORM_TYPE": "torch"}   # the caller's value stays (runs as requested) ...
    notes = settings.stock_env_notes(other)
    assert len(notes) == 1 and notes[0].startswith("NOTE LAYERNORM_TYPE='torch' requested — running as requested;") and "LAYERNORM_TYPE=fast_layernorm" in notes[0]


def test_dtype_on_any_mode_runs_as_requested_with_a_note_naming_the_base():
    for mode in ("off", "exact", "fast", "big", "S1", None):
        assert settings.dtype_note("bf16", mode) is None and settings.dtype_note(None, mode) is None   # the base: nothing to say
        n = settings.dtype_note("fp32", mode)
        assert n.startswith("NOTE --dtype fp32 requested") and "running upstream's fp32 switch as requested" in n and "--dtype bf16 + LAYERNORM_TYPE=fast_layernorm" in n
    assert "requested on the stock arm" in settings.dtype_note("fp32", "off") and "requested under exact" in settings.dtype_note("fp32", "exact")
    assert settings.dtype_note("FP16", "fast").startswith("NOTE --dtype fp16 requested under fast")   # any value passes as stated (upstream parses it); the note only names the base


def _record_flags():
    """Upstream's defaults as the flags dict the frozen gate reads (settings.effective with nothing stated)."""
    import argparse
    ap = argparse.ArgumentParser(); cli._common_pred(ap)
    return settings.effective(ap.parse_args(["-i", "q", "-o", "o"]), [], _stubs.TREE)


def _query(tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"name": "t1", "sequences": [{"proteinChain": {"sequence": "MK", "unpairedMsaPath": "/m/u.a3m"}}], "modelSeeds": [101]}]))
    return str(q)


def test_pred_dtype_fp32_under_a_kit_mode_prints_the_note_and_proceeds(monkeypatch, tmp_path, capsys):
    """`pred --mode exact --dtype fp32` and `--mode fast --dtype fp32`: the NOTE line, then the run proceeds to the next gate (here the CPU
    box's own refusals) — never the usage exit and never a dtype refusal."""
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    for mode, words in (("exact", "NOTE --dtype fp32 requested under exact — running upstream's fp32 switch as requested"), ("fast", "NOTE --dtype fp32 requested under fast")):
        stack._REPORT = None
        rc = cli.main(["pred", "--mode", mode, "--dtype", "fp32", "-i", _query(tmp_path), "-o", str(tmp_path / ("o_" + mode))])
        out = capsys.readouterr().out
        assert rc != cli.EXIT_USAGE and words in out and "refused: --dtype" not in out, (mode, rc, out[-600:])


# ------------------------------------------------------------------------------------------------ the LayerNorm census (upstream's modules faked in-process)
def _fake_upstream(monkeypatch, *, use_fast, extension):
    layers = types.ModuleType(lncensus.LAYERS); layers._use_fast_layer_norm = use_fast
    loader = types.ModuleType(lncensus.LOADER); loader._load_fast_layer_norm_cuda_v2 = lambda: extension
    for name, mod in ((lncensus.LAYERS, layers), (lncensus.LOADER, loader)):
        monkeypatch.setitem(sys.modules, name, mod)
    for parent in ("opendde", "opendde.model", "opendde.model.triangular", "opendde.model.layer_norm"):
        monkeypatch.setitem(sys.modules, parent, sys.modules.get(parent) or types.ModuleType(parent))


def test_census_names_the_extension_upstream_bound(monkeypatch, capsys):
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    ext = types.SimpleNamespace(__file__="/jitcache/torch_ext/fast_layer_norm_cuda_v2/fast_layer_norm_cuda_v2.so")
    _fake_upstream(monkeypatch, use_fast=True, extension=ext)
    c = lncensus.census(strict=True, stream=sys.stdout)
    out = capsys.readouterr().out
    assert c["backend"] == "fast_layer_norm_cuda_v2" and c["reason"] is None
    assert re.fullmatch(r"\[opendde-opt\] LAYERNORM requested=fast_layernorm backend=fast_layer_norm_cuda_v2 module=/jitcache/torch_ext/fast_layer_norm_cuda_v2/fast_layer_norm_cuda_v2\.so jit_s=[0-9.]+", out.strip()), out   # jit_s: the load / JIT-build seconds, spent at the census (before any timed item) and named
    assert isinstance(c["jit_s"], float)


def test_census_refuses_by_name_when_fast_layernorm_was_requested_and_did_not_load(monkeypatch, capsys):
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    _fake_upstream(monkeypatch, use_fast=True, extension=None)                       # upstream's loader gave up (no ninja / nvcc): it would run torch LayerNorm after a warning
    with pytest.raises(lncensus.NotLoaded) as ei:
        lncensus.census(strict=True, stream=sys.stdout)
    out = capsys.readouterr().out
    assert out.startswith("[opendde-opt] LAYERNORM requested=fast_layernorm backend=torch reason=") and "fast_layer_norm_cuda_v2 unavailable" in out
    assert str(ei.value).startswith("LAYERNORM_TYPE=fast_layernorm requested but not loaded: extension fast_layer_norm_cuda_v2 unavailable") and "pip install ninja" in str(ei.value)
    _fake_upstream(monkeypatch, use_fast=False, extension=None)                      # FusedLayerNorm was not even selected at upstream import
    with pytest.raises(lncensus.NotLoaded):
        lncensus.census(strict=True, stream=sys.stdout)
    assert "backend=torch" in capsys.readouterr().out
    c = lncensus.census(strict=False, stream=sys.stdout)                              # the non-strict form only reports
    assert c["backend"] == "torch"


def test_census_is_unconfirmed_not_refused_on_an_upstream_without_the_selection_module(monkeypatch, capsys):
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    monkeypatch.setitem(sys.modules, lncensus.LAYERS, None)                          # `import` of a None entry raises ModuleNotFoundError: the module is absent
    c = lncensus.census(strict=True, stream=sys.stdout)
    assert c["backend"] == "unconfirmed" and c["reason"] == "absent:" + lncensus.LAYERS and "backend=unconfirmed" in capsys.readouterr().out


def test_census_does_not_refuse_what_was_not_requested(monkeypatch, capsys):
    monkeypatch.setenv("LAYERNORM_TYPE", "torch")                                     # a caller's own choice (runs as requested; the NOTE is settings')
    _fake_upstream(monkeypatch, use_fast=False, extension=None)
    c = lncensus.census(strict=True, stream=sys.stdout)
    assert c["backend"] == "torch" and c["reason"] == "not_requested" and "requested=torch backend=torch" in capsys.readouterr().out


def test_the_census_is_taken_on_every_model_route():
    """Source census: the kit route (cli.cmd_pred) and the stock process (stock_pred) each take it strict."""
    here = os.path.dirname(os.path.dirname(__file__))
    for f, words in (("cli.py", "_lncensus.census(strict=True"), ("stock_pred.py", "_lncensus.census(strict=True)")):
        assert words in open(os.path.join(here, f)).read(), f


# ------------------------------------------------------------------------------------------------ the frozen gate reads booleans as click does
def test_the_frozen_gate_reads_flag_booleans_as_the_stock_cli_parses_them(tmp_path):
    for w in ("1", "true", "True", "t", "yes", "Y", "on"):
        assert frozen._truthy(w), w
    for w in (None, "0", "false", "no", "off", "f", ""):
        assert not frozen._truthy(w), w
    root = tmp_path / "w"
    for rel in frozen.REQUIRED_FILES:
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"name": "t1", "sequences": [{"proteinChain": {"sequence": "MK", "unpairedMsaPath": "/m/u.a3m"}}], "modelSeeds": [101]}]))
    rec = _record_flags()
    first = frozen.problems(str(root), str(q), rec, ["--use_template", "true"])
    assert first and any("without templatesPath" in x for x in first), first                   # the flag is read: the template preconditions are named
    for spelling in (["--use_template", "1"], ["--use_template=YES"], ["--use_template", "true"]):
        assert frozen.problems(str(root), str(q), rec, spelling) == first, spelling            # the same reading whatever true-word the caller typed
    assert frozen.problems(str(root), str(q), rec, ["--use_template", "0"]) == []


def test_bigln_guard_is_inert_by_design_on_the_fused_layernorm_base(monkeypatch):
    """The offload unit's LayerNorm guard wraps torch.nn.functional.layer_norm; upstream's fused LayerNorm never calls it, so on the base the
    lever is inert by design (named), not a lever that never ran."""
    from opendde_opt import ran
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    ok, why = ran.engagement("bigln_guard", {"det": False, "n_gpu": 1})
    assert not ok and why.startswith("LAYERNORM_TYPE=fast_layernorm: upstream's LayerNorm is the fused extension")
    assert ran.inert_by_design(["bigln_guard", "diffz"], {"det": False, "n_gpu": 1}) == {"bigln_guard": f"lever_inert_by_design:bigln_guard({why})"}
    monkeypatch.setenv("LAYERNORM_TYPE", "torch")                                    # a caller's own torch LayerNorm: the guard engages as before
    assert ran.engagement("bigln_guard", {"det": False, "n_gpu": 1}) == (True, None)
    assert ran.engagement("bigln_guard", {"det": False, "n_gpu": 1, "layernorm": "fast_layernorm"})[0] is False
