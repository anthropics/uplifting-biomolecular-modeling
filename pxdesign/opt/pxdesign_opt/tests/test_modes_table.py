"""The mode table is locked to the kit: one place, the kit's own switch values, the hoist levers tier 1, the `fast` line tier 2 and never the default, `rows` refused by name (not a mode), no variant; run.sh's pure-bash refusal carries the table's bytes."""
import os
import re

import pytest

from pxdesign_opt import modes, registry
from pxdesign_opt.modes import DEFAULT_MODE, FAST_MODE, KIT_MODES, MODES, kit_switch_defaults, lever_table, resolve


def test_modes_and_default():
    assert MODES == ("off", "exact", "fast", "big")  # every name --mode / PXDESIGN_OPT accepts (the house table off|exact|fast|big)
    assert FAST_MODE == "fast"                         # the tolerance-class speed line
    assert DEFAULT_MODE == FAST_MODE == "fast"        # the default names a tested line: the fast line (tier 2); exact by name; big is never the default
    assert set(KIT_MODES) == set(MODES) and tuple(modes.TABLE.modes) == MODES
    assert not any(hasattr(modes, n) for n in ("COMPOSITIONS", "ACCEPTED", "KIT_COMPOSITIONS", "SPECS"))   # no second table: a name is a mode or it is refused
    assert not hasattr(modes, "RETIRED")                # no alias table either: `rows` is a lever value `big` exports, refused like any other non-mode
    assert KIT_MODES["off"].levers == () and KIT_MODES["off"].env == {}
    assert KIT_MODES["exact"].tier == "1" and KIT_MODES["exact"].package_levers == ()
    big = KIT_MODES["big"]
    assert big.tier == "2" and big.levers == KIT_MODES["exact"].levers and big.env == dict(KIT_MODES["exact"].env, PXD_HOIST_MODE=modes.ROWS_VALUE) == dict(KIT_MODES["exact"].env, PXD_HOIST_MODE="rows")
    assert big.package_levers == ("featdiet", "padmask", "rowpipe", "tf32", "sdedup") == modes.BIG_LEVERS == modes.SIZE_LEVERS + ("tf32", "sdedup") and tuple(registry.PACKAGE_LEVERS) == ("featdiet", "padmask", "rowpipe", "tf32", "sdedup")
    assert all(lv.switch == "package_levers" for lv in registry.PACKAGE_LEVERS.values()) and [lv.tier for lv in registry.PACKAGE_LEVERS.values()] == ["1", "1", "2", "2", "2"]   # selected by the mode-table field; featdiet / padmask exact by construction (tier 1), rowpipe fp32-reassociation class, tf32 / sdedup tolerance class (tier 2)
    assert "2788" in big.promise and "5988" in big.promise and "never the default" in big.promise and "no fallback" in big.promise
    fast = KIT_MODES["fast"]
    assert fast.tier == "2" and fast.levers == KIT_MODES["exact"].levers and fast.env == big.env                     # exact's hoist levers, the rows switch value
    assert fast.package_levers == ("featdiet", "padmask", "tf32", "sdedup") == modes.FAST_LEVERS
    assert "tier 2" in fast.promise and "NOT byte-identical" in fast.promise and "no fallback" in fast.promise and "package default" in fast.promise and DEFAULT_MODE == "fast"


ROWS_SENTENCE = "unknown mode 'rows'; expected one of off|exact|fast|big"


def test_a_lever_value_is_not_a_mode():
    """`rows` is not a mode (mode doctrine: a lever value lives inside a mode; PXD_HOIST_MODE=rows is exported by `big`): every python route
    reaches modes.check_mode, which raises the table's one sentence naming the accepted set, as for any unknown name."""
    assert modes.unknown_message("rows") == ROWS_SENTENCE
    assert modes.unknown_message("turbo") == "unknown mode 'turbo'; expected one of off|exact|fast|big"
    for call in (modes.check_mode, resolve):
        with pytest.raises(ValueError) as e:
            call("rows")
        assert str(e.value) == ROWS_SENTENCE, (call, str(e.value))
    assert resolve("big").package_levers == ("featdiet", "padmask", "rowpipe", "tf32", "sdedup") and resolve("big").env["PXD_HOIST_MODE"] == "rows"
    assert resolve("fast").package_levers == ("featdiet", "padmask", "tf32", "sdedup") and resolve("fast").env["PXD_HOIST_MODE"] == "rows" and modes.check_mode("Fast") == "fast"
    with pytest.raises(ValueError) as e:
        modes.check_mode("Turbo")
    assert str(e.value) == "unknown mode 'Turbo'; expected one of off|exact|fast|big"
    with pytest.raises(ValueError) as e:
        modes.check_mode(None)                          # the CLI resolves the default before it asks; an absent value is refused here
    assert str(e.value) == "unknown mode None; expected one of off|exact|fast|big"
    assert [r["mode"] for r in modes.mode_table()] == list(KIT_MODES) and all("composition" not in r for r in modes.mode_table())


def test_run_sh_refusal_carries_the_tables_bytes(tree):
    """run.sh refuses a mode outside the table in pure bash (usage errors precede the gate): its case list is MODES — bash cannot import
    modes.py, so the spelling is locked here."""
    src = open(os.path.join(tree, "run.sh"), encoding="utf-8").read()
    assert f'case "$MODE" in {"|".join(MODES)}) ;;' in src
    assert f'*) echo "[pxdesign-opt] unknown --mode $MODE (modes: {"|".join(MODES)})" >&2; exit 2 ;;' in src


def test_exact_env_is_the_registry_defaults_one_copy():
    """The values exact exports are the registry's default column (no second copy of a switch value in the table)."""
    assert KIT_MODES["exact"].env == {k: registry.KIT_SWITCHES[k][0] for k in modes.EXACT_SWITCHES}
    assert KIT_MODES["exact"].env == {"PXD_HOIST": "1", "PXD_HOIST_MODE": "shape", "PXD_HOIST_MASK": "1"}


def test_jit_cache_key_shape():
    from pxdesign_opt.modes import jit_cache_key
    assert jit_cache_key("2.3.1+cu121", cc="9.0") == "torch2.3.1-cu121-sm90"
    assert jit_cache_key("2.3.1+cu121", cuda="121", cc="9.0") == "torch2.3.1-cu121-sm90"
    assert jit_cache_key("2.7.1+cu128", cc="9.0") == "torch2.7.1-cu128-sm90"
    assert jit_cache_key("2.3.1", cuda="121", cc="unknown") == "torch2.3.1-cu121-smunknown"


def test_exact_is_the_kit_default_line(tree):
    """Every switch value exact exports equals the default the kit's own hoist.py applies when the switch is unset."""
    defaults = kit_switch_defaults(os.path.join(tree, registry.KIT_RELPATH))
    assert defaults, "no os.environ.get literals found in hoist.py"
    for k, v in KIT_MODES["exact"].env.items():
        assert defaults.get(k) == v, (k, v, defaults.get(k))
    for k, (v, where, _) in registry.KIT_SWITCHES.items():
        assert defaults.get(k) == v, (k, v, defaults.get(k), where)
    assert set(KIT_MODES["exact"].env) == {"PXD_HOIST", "PXD_HOIST_MODE", "PXD_HOIST_MASK"}


def test_levers_tier1_forward_and_switches():
    assert tuple(registry.LEVERS) == ("h1", "h2", "h3", "h4", "h5")
    for lv in registry.LEVERS.values():
        assert lv.cls == "forward" and lv.tier == "1"
        assert lv.switch in registry.KIT_SWITCHES
        assert re.search(r"hoist\.py:\d+", lv.code)
    assert KIT_MODES["exact"].levers == tuple(registry.LEVERS)
    rows = lever_table()
    assert [r["lever"] for r in rows] == list(registry.LEVERS) + list(registry.PACKAGE_LEVERS)


def test_resolve_drops_caller_switches(monkeypatch):
    monkeypatch.setenv("PXD_HOIST_MODE", "rows")
    res = resolve("exact")
    assert res.env["PXD_HOIST_MODE"] == "shape"
    assert set(res.dropped) == {"PXD_HOIST_MODE"}
    assert res.levers_planned == ["h1", "h2", "h3", "h4", "h5"]
    off = resolve("off")
    assert off.is_stock and off.env == {} and off.levers_planned == []


def test_unknown_mode_and_variant_refused():
    import pytest
    with pytest.raises(ValueError):
        resolve("turbo")
    with pytest.raises(ValueError):
        modes.check_mode("hoist+graphs")
    with pytest.raises(ValueError):
        modes.check_mode("tf32")                        # a package lever of the fast line, not a mode
    with pytest.raises(ValueError):
        modes.check_mode("rows")                        # not a mode: the lever value `big` exports


def test_the_rows_switch_is_wired_by_big(tree):
    assert KIT_MODES["big"].env["PXD_HOIST_MODE"] == "rows" and KIT_MODES["fast"].env["PXD_HOIST_MODE"] == "rows"   # the lever value big and fast export
    assert "tf32" in registry.PACKAGE_LEVERS and not hasattr(registry, "NOT_WIRED")            # the registry names wired levers only: no candidate table ships


def test_table_lives_in_modes_only(tree):
    """No other package module or config carries a mode definition or a lever switch value."""
    pkg = os.path.join(tree, "opt", "pxdesign_opt")
    for fn in os.listdir(pkg):
        if fn.endswith(".py") and fn not in ("modes.py", "registry.py"):
            src = open(os.path.join(pkg, fn), encoding="utf-8").read()
            assert "KIT_MODES = " not in src and '"PXD_HOIST_MODE": "shape"' not in src, fn
    for fn in os.listdir(os.path.join(tree, "configs")):
        src = open(os.path.join(tree, "configs", fn), encoding="utf-8").read()
        for line in src.splitlines():
            if line.startswith("export"):
                assert not re.search(r"\bPXD_|\bPXDESIGN_OPT\b|PXDESIGN_HOIST", line.split("#")[0]), (fn, line)


