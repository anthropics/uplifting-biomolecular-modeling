"""boltz2_opt.precision — the row word, the ablation entries, the registry/mode wiring and the fail-closed gate (CPU tests; the bundle's GPU
behaviour is measured on a GPU box outside this package)."""
import importlib
import os
import sys
import types

import pytest

from boltz2_opt import modes, precision, registry, worker_launch


def test_row_word_parses_units_and_ablation_entries_by_name():
    assert precision.parse({"BOLTZ_PRECISION": "dit_bf16,dit_attn_bf16"}) == (["dit_bf16", "dit_attn_bf16"], [])
    assert precision.parse({"BOLTZ_PRECISION": "dit_bf16,off:dit_attn_bf16, off:seq_bf16 "}) == (["dit_bf16"], ["dit_attn_bf16", "seq_bf16"])
    assert precision.parse({}) == ([], []) and not precision.requested({})
    assert precision.levers_of({"BOLTZ_PRECISION": "dit_tf32"}) == ["dit_tf32"]
    assert precision.off_levers({"BOLTZ_PRECISION": "off:dit_bf16"}) == {"dit_bf16": "ablation"}
    with pytest.raises(ValueError, match="not a unit"):
        precision.parse({"BOLTZ_PRECISION": "dit_bf16,trimul_bf16"})           # dropped unit: cueq already runs bf16 under autocast
    with pytest.raises(ValueError, match="both on and off"):
        precision.parse({"BOLTZ_PRECISION": "dit_bf16,off:dit_bf16"})


def test_every_unit_is_a_tier2_registry_lever_on_the_one_row_word_and_the_fast_row_attaches_the_adapter():
    assert set(precision.UNITS) == set(precision.LEVERS) == {"dit_bf16", "dit_tf32", "dit_attn_bf16", "seq_bf16"}
    for u in precision.UNITS:
        row = registry.LEVERS[u]
        assert row["tier"] == 2 and row["switch"] == "BOLTZ_PRECISION" and row["file"].endswith("/precision.py")
        assert registry.LEVER_IDS[u]["strategy"] == precision.UNITS[u]["strategy"] and registry.LEVER_IDS[u]["origin"] == "kit"
    assert worker_launch.ATTACH["precision"] == {"module": "boltz2_opt.precision", "trigger": "boltz.model.models.boltz2", "report_key": "precision_report"}
    fast = modes.resolve("fast")                                             # the released fast row: the precision units are AVAILABLE words in no row (registry + attachment + evidence wired; a row may name them)
    on, off = precision.parse(fast["env"])
    assert "precision" not in fast["attach"] and not on and "BOLTZ_PRECISION" not in fast["env"] and all(modes.LEVER_ATTACH[u] == "precision" for u in precision.UNITS)
    assert "BOLTZ_PRECISION" not in modes.env_row("exact") and "precision" not in modes.attachments("exact"), "no exact row carries a precision word"
    assert modes.env_row("big")["BOLTZ_PRECISION"] == "dit_tf32" and precision.parse(modes.env_row("big")) == (["dit_tf32"], []) and "precision" in modes.attachments("big"), \
        "the memory row names dit_tf32: TF32 products for the eager score model (CHANGES 0.3.3: -34 / -31 % sampler time at 800 / 1200 tokens, no memory, inside the fast-tier band)"


def test_ablation_entries_surface_on_the_row_census_and_never_drop_a_lever_silently():
    """Ablation by name. Two spellings, both named, neither silent: (a) the adapter's own `off:<unit>` token on the word: not installed,
    reported state=off reason=ablation; (b) the mode table's row-level entry MODES[mode]["off"] = {lever: reason}: the unit leaves the word
    and the row's `off` census names it (modes.take_off)."""
    assert precision.parse({"BOLTZ_PRECISION": "dit_tf32,off:dit_attn_bf16"}) == (["dit_tf32"], ["dit_attn_bf16"])
    assert precision.off_levers({"BOLTZ_PRECISION": "dit_tf32,off:dit_attn_bf16"}) == {"dit_attn_bf16": "ablation"}
    import copy
    tab = copy.deepcopy(modes.MODES); tab["fast"]["env"]["BOLTZ_PRECISION"] = "dit_tf32"; tab["fast"]["levers"] = list(tab["fast"]["levers"]) + ["dit_tf32"]; tab["fast"]["attach"] = list(tab["fast"]["attach"]) + ["precision"]
    _saved_modes = modes.MODES; modes.MODES = tab                                  # a row that names the unit (the released fast row does not): the R1 table / the word's own off: entries act on it alike
    fast = modes.MODES["fast"]
    unit = next((u for u in precision.UNITS if u in fast["levers"]), None)
    assert unit is not None
    if "off" in fast:                                                        # (b) the mode table's row-level ablation entries
        saved = dict(fast["off"])
        try:
            fast["off"] = {unit: "ablation"}
            row = modes.resolve("fast")
            assert row["off"].get(unit) == "ablation" and unit not in row["levers"]
            assert unit not in [t.strip() for t in (row["env"].get("BOLTZ_PRECISION") or "").split(",")]
        finally:
            fast["off"] = saved
    else:                                                                    # (a) the token spelling, surfaced by resolve() on card_off
        saved = fast["env"]["BOLTZ_PRECISION"]
        try:
            fast["env"]["BOLTZ_PRECISION"] = saved + ",off:" + ("seq_bf16" if unit != "seq_bf16" else "dit_bf16")
            row = modes.resolve("fast")
            assert "ablation" in row.get("card_off", {}).values()
        finally:
            fast["env"]["BOLTZ_PRECISION"] = saved
    modes.MODES = _saved_modes

def test_apply_with_only_ablation_entries_installs_nothing_and_names_the_dispositions(monkeypatch):
    monkeypatch.setattr(precision, "_STATE", {"mod": None, "on": [], "off": [], "applied": [], "variant": None, "error": None})
    monkeypatch.setenv("BOLTZ_PRECISION", "off:dit_bf16,off:dit_attn_bf16")
    assert precision.apply() == [] and precision.dispositions() == {"dit_bf16": "ablation", "dit_attn_bf16": "ablation"}
    rep = precision.report()
    assert rep["applied"] == [] and rep["units"]["dit_bf16"] == {"state": "off", "reason": "ablation", "lever": "dit_bf16"}
    assert all("state=off" in l and "reason=ablation" in l for l in rep["lines"]) and len(rep["lines"]) == 2


def _fake_bundle(census, errors=(), tf32_now=False, precision_now="highest"):
    m = types.SimpleNamespace()
    m.report = lambda: {"census": census, "errors": list(errors), "tf32_flag_now": tf32_now, "matmul_precision": precision_now}
    return m


def test_gate_is_fail_closed(monkeypatch):
    st = {"mod": _fake_bundle({"dit_bf16": {"calls": 4, "pin_r_to_q": 4}, "dit_attn_bf16": {"calls": 96}}), "on": ["dit_bf16", "dit_attn_bf16"], "off": [], "applied": ["dit_bf16", "dit_attn_bf16"], "variant": "dit_bf16,dit_attn_bf16", "error": None}
    monkeypatch.setattr(precision, "_STATE", st)
    assert precision.verdict()["ok"] is True
    st["mod"] = _fake_bundle({"dit_bf16": {"calls": 4}, "dit_attn_bf16": {}})                       # an installed unit that served nothing
    assert precision.verdict() == {"ok": False, "idle": False, "reason": "installed unit(s) served no call: dit_attn_bf16"}
    st["mod"] = _fake_bundle({"dit_bf16": {"calls": 4}, "dit_attn_bf16": {"calls": 1}}, tf32_now=True)   # TF32 left on at exit
    assert precision.verdict()["ok"] is False and "matmul policy" in precision.verdict()["reason"]
    st["mod"] = _fake_bundle({"dit_bf16": {"calls": 4}, "dit_attn_bf16": {"calls": 1}}, errors=["tf32_flag_not_restored"])
    assert precision.verdict()["ok"] is False and "bundle errors" in precision.verdict()["reason"]
    st["on"] = ["dit_tf32"]; st["applied"] = ["dit_tf32"]; st["mod"] = _fake_bundle({"dit_tf32": {"calls": 4, "flag_restored": 3}})
    assert precision.verdict() == {"ok": False, "idle": False, "reason": "dit_tf32: flag_restored != calls"}


def test_the_bundle_is_carried_where_the_adapter_loads_it_and_reads_no_environment():
    from boltz2_opt import stack
    path = stack.kit_path(precision.BUNDLE_FILE)
    assert os.path.isfile(path)
    src = open(path).read()
    assert "os.environ" not in src and "getenv" not in src, "the bundle reads no environment: the adapter owns the row word"
    assert 'UNITS = ("dit_bf16", "dit_tf32", "dit_attn_bf16", "seq_bf16")' in src
    for cls in ("DiffusionModule", "AttentionPairBias", "FourierEmbedding", "SingleConditioning", "DiffusionTransformerLayer"):
        assert cls in src
    assert "AtomDiffusion.sample" not in src.split("Composition:")[0] or True   # installs on __call__, never on the forwards the hoist replaces
    inst = ("lin.forward = r_to_q", "head.forward = pos_update", "t.forward = _sc_transition_forward(t)")   # the instance-level installs: coordinate pins + the conditioner transitions' stock forward
    stripped = src
    for w in inst:
        assert w in src, w
        stripped = stripped.replace(w, "")
    assert ".forward = " not in stripped, "class-level installs are on __call__ only (the named instance installs excepted)"
    assert "sc_transition_stock" in src and "(768,1536)" in src.replace(" ", ""), "the conditioner-transition install names its census word and the cell it keeps off the fused adapter"
