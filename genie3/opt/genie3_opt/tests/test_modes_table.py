"""The mode table: names (exactly off | exact | fast), tiers, levers, the batched capture line's argv (--batch-size = the request's
generation.dataset.batch_size, default upstream's 1), the default rule, unknown names refused by name; no lever selector outside the three words."""
import pytest

from genie3_opt import modes, registry


def test_mode_names_and_tiers():
    assert modes.MODE_NAMES == ("off", "exact", "fast")
    assert modes.MODES["off"].tier is None and modes.MODES["exact"].tier == 1 and modes.MODES["fast"].tier == 2


def test_exact_is_tier1_batched_capture_line():
    r = modes.resolve("exact")
    assert r.levers == ("L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19")
    assert r.flags == ("--batch-size", "1", "--cuda-graphs", "--hoist", "--reuse-graphs", "16", "--lean-pair", "--wide-capture", "--alloc", "expandable")   # g3batch.py: upstream's batch semantics at B (upstream's default 1; design --batch_size B), graph replay, hoist, graph reuse
    assert modes.batch_size(r) == 1 and modes.B_DEFAULT == "1" == registry.LEVERS["L11"].switch[2]
    assert r.attach == "driver" and r.driver == (("genie3_opt", "g3batch.py"),)
    assert r.line == "<genie3_opt>/g3batch.py --config <experiment.yaml> --batch-size 1 --cuda-graphs --hoist --reuse-graphs 16 --lean-pair --wide-capture --alloc expandable"
    for name in ("exact_b1", "exact_streams8", "exact_k1", "fast_l3", "fast_fp32", "fast_trimul", "tf32", "big"):   # no name outside the three words resolves: a lever lives inside a mode or in none
        with pytest.raises(modes.ModeError) as e:
            modes.resolve(name)
        assert f"unknown mode {name!r} (expected off|exact|fast)" in str(e.value), name
    for gone in ("KIT_COMPOSITIONS", "with_precision", "FAST_TRIMUL", "TRIMULS", "PRECISIONS"):   # no composition names, no precision selector, no per-lever selector: the mode set is the whole surface
        assert not hasattr(modes, gone), gone


def test_batch_is_a_run_parameter():
    r = modes.resolve("exact")
    assert modes.with_batch(r, None) is r
    r16 = modes.with_batch(r, 16)
    assert modes.batch_size(r16) == 16 and r16.flags == ("--batch-size", "16", "--cuda-graphs", "--hoist", "--reuse-graphs", "16", "--lean-pair", "--wide-capture", "--alloc", "expandable") and r16.levers == r.levers
    assert modes.with_batch(modes.resolve("fast"), 1).flags == ("--batch-size", "1", "--cuda-graphs", "--hoist", "--reuse-graphs", "16", "--lean-pair", "--wide-capture", "--alloc", "expandable", "--pt-chunk", "design", "--tf32", "--trimul", "fpf", "--compile")
    off = modes.resolve("off")
    assert modes.with_batch(off, 4) is off                              # the stock line carries no driver flag: upstream reads generation.dataset.batch_size itself
    with pytest.raises(modes.ModeError):
        modes.with_batch(r, 0)
    assert modes.request_batch({"generation": {"dataset": {"batch_size": 4}}}) == 4 and modes.request_batch({"generation": {"dataset": {}}}) is None and modes.request_batch({}) is None
    assert modes.effective("exact", 4).flags[:2] == ("--batch-size", "4") and modes.effective("exact", None) == modes.resolve("exact") and modes.effective("off", 4) == off
    assert modes.precision_of(modes.resolve("fast")) == "tf32" and modes.precision_of(r) == "fp32" and modes.precision_of(off) == "fp32"


def test_fast_is_the_exact_line_plus_ptchunk_tf32_trimul_and_the_compiled_core():
    r = modes.resolve("fast")
    assert r.levers == ("L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19", "L12", "L13", "L7", "L16")
    assert r.flags == ("--batch-size", "1", "--cuda-graphs", "--hoist", "--reuse-graphs", "16", "--lean-pair", "--wide-capture", "--alloc", "expandable", "--pt-chunk", "design", "--tf32", "--trimul", "fpf", "--compile")
    assert modes.batch_size(r) == 1 and r.driver == modes.resolve("exact").driver and modes.trimul_of(r) == "fpf"
    assert registry.LEVERS["L13"].tier == 3 and "L13" not in modes.MODES["exact"].levers          # TF32: inside fast only, never on the exact line
    assert registry.LEVERS["L12"].tier == 2 and "L12" not in modes.MODES["exact"].levers          # the per-design transition chunk: class 2, inside fast only
    assert registry.LEVERS["L16"].tier == 3 and "L16" not in modes.MODES["exact"].levers and registry.LEVERS["L16"].switch == ("flag", "--compile", None)   # the compiled core: fast only
    assert registry.LEVERS["L7"].tier == 3 and "L7" not in modes.MODES["exact"].levers and modes.trimul_of(modes.resolve("exact")) == "stock"   # the fused TriMul kernel: class 3, fast's provider, never on the exact line


def test_no_per_lever_switch_on_the_surface():
    """The mode set is the whole surface: no per-lever selector composes a line (modes has no with_trimul / TRIMUL_CHOICES; effective = the
    mode's line at the request's batch size); L7's driver flag pair is fast's own, never exact's."""
    assert not hasattr(modes, "with_trimul") and not hasattr(modes, "TRIMUL_CHOICES")
    assert modes.effective("fast", 4).flags == ("--batch-size", "4", "--cuda-graphs", "--hoist", "--reuse-graphs", "16", "--lean-pair", "--wide-capture", "--alloc", "expandable", "--pt-chunk", "design", "--tf32", "--trimul", "fpf", "--compile")
    assert modes.effective("exact").flags == modes.EXACT_FLAGS and modes.effective("fast", None) == modes.resolve("fast")
    assert modes.trimul_of(modes.resolve("fast")) == "fpf" and modes.trimul_of(modes.resolve("exact")) == "stock" and modes.trimul_of(modes.resolve("off")) == "stock"
    assert (modes.TRIMUL_FLAG, modes.TRIMUL_ON) == (registry.LEVERS["L7"].switch[1], registry.LEVERS["L7"].switch[2]) == ("--trimul", "fpf")
    with pytest.raises(TypeError):
        modes.effective("fast", 4, "stock")


def test_off_is_the_stock_cli():
    r = modes.resolve("off")
    assert r.attach == "stock-cli" and r.flags == () and r.levers == () and r.line == "stock CLI"
    assert modes.batch_size(r) is None


def test_default_mode_is_the_literal_fast():
    assert modes.DEFAULT_MODE == "fast"                                    # the package default: fast wherever a fast mode ships; exact is selected by name
    assert modes.resolve(None).mode == "fast" and modes.resolve("").mode == "fast" and modes.resolve("exact").mode == "exact"
    assert "fast" in modes.MODES and "exact" in modes.MODES
    src = open(modes.__file__, encoding="utf-8").read()                    # a literal, not a computed rule: guard against a "smart" default creeping back in
    assert "default_mode(" not in src and "DEFAULT_MODE_RULE" not in src


def test_no_lever_outside_the_table():
    """Every registry entry is a lever some mode plans (the registry carries nothing dormant), and the modes plan exactly fast's set: exact's levers + L12 L13 L7 L16."""
    in_modes = {lid for m in modes.MODES.values() for lid in m.levers}
    assert in_modes == set(modes.FAST_LEVERS) == set(registry.LEVERS) and "L7" in in_modes
    assert set(modes.EXACT_LEVERS) < set(modes.FAST_LEVERS)


def test_every_lever_of_every_mode_is_in_the_registry():
    for m in modes.MODES.values():
        for lid in m.levers:
            assert lid in registry.LEVERS
    for lid, lv in registry.LEVERS.items():                                  # the evidence grammar design.read_evidence reads: log | timings | judged
        assert lv.evidence is None or lv.evidence[0] in ("log", "timings", "judged"), (lid, lv.evidence)
        assert lv.switch[0] in ("flag", "default", "driver"), (lid, lv.switch)


def test_table_rows():
    rows = modes.table()
    assert [r["mode"] for r in rows] == ["off", "exact", "fast"]
    assert rows[1]["flags"] == ["--batch-size", "1", "--cuda-graphs", "--hoist", "--reuse-graphs", "16", "--lean-pair", "--wide-capture", "--alloc", "expandable"] and rows[2]["levers"] == list(modes.FAST_LEVERS) and rows[2]["flags"][-3:] == ["--trimul", "fpf", "--compile"]
