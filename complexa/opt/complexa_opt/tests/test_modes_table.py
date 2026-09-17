"""The one mode table: names, the default, the stock route, the three kit modes and their lever sets (levers.LEVERS names, every lever
of the registry in some mode: no dead lever), the identity class each mode is held to, the mode-argument precedence."""
import subprocess
import sys

import pytest

from complexa_opt import levers, modes


def test_table_names_default_routes():
    assert modes.MODES == ("off", "exact", "fast", "big")
    assert modes.STOCK_MODES == ("off",)
    assert modes.DEFAULT_MODE == "fast"
    assert modes.SERVED == modes.MODES
    assert set(modes.KIT_MODES) == {"exact", "fast", "big"}
    assert modes.ROUTES == {"off": "stock", "exact": "kit", "fast": "kit", "big": "kit"}
    assert set(modes.ROUTE_WORDS) == {"stock", "kit"}
    assert "default" not in modes.MODES                      # no fifth word: off | exact | fast | big
    assert modes.ENV_MODE == "COMPLEXA_OPT" and modes.ENV_RECORD == "COMPLEXA_OPT_RECORD" and modes.ENV_RECORD.startswith(modes.ENV_MODE)


def test_lever_sets_are_registry_names_and_every_lever_serves_a_mode():
    used = set()
    for mode, names in modes.KIT_MODES.items():
        assert names and len(set(names)) == len(names), mode
        assert all(n in levers.LEVERS for n in names), (mode, names)
        assert sum(n in levers.NEEDS_BANK for n in names) <= 1                       # rows / fused are alternative sources of the same bias
        used.update(names)
    assert used == set(levers.LEVERS), f"levers no mode uses (dead): {set(levers.LEVERS) - used}"
    assert modes.KIT_MODES["exact"] == modes.EXACT_SET
    assert set(modes.EXACT_SET) < set(modes.KIT_MODES["fast"]) and set(modes.EXACT_SET) < set(modes.KIT_MODES["big"])
    assert modes.levers_of("off") == () and modes.levers_of("big") == modes.KIT_MODES["big"]


def test_identity_class_per_mode_follows_the_levers_tiers():
    for mode, names in modes.KIT_MODES.items():
        assert modes.TIERS[mode] == levers.tier_of(names), mode
    assert modes.TIERS == {"exact": "exact", "fast": "tolerance", "big": "exact"}       # big's levers are all exact-class: byte-identical designs at a third of the memory
    assert [levers.LEVERS[n].tier for n in ("pair_bias_fused", "attn_sdpa")] == ["tolerance", "tolerance"]


def test_resolve_precedence():
    assert modes.resolve(None, {}) == "fast"                                            # no mode given: the default
    assert modes.resolve("  ", {"COMPLEXA_OPT": ""}) == "fast"
    assert modes.requested(None, {}) == "fast" and modes.requested(" Exact ", {}) == "exact" and modes.requested(None, {"COMPLEXA_OPT": "big"}) == "big"
    assert modes.resolve(None, {"COMPLEXA_OPT": "off"}) == "off"
    assert modes.resolve(" OFF ", {}) == "off" and modes.resolve("Exact", {}) == "exact" and modes.resolve(None, {"COMPLEXA_OPT": "big"}) == "big"
    assert modes.resolve("off", {"COMPLEXA_OPT": "off"}) == "off"
    with pytest.raises(modes.ModeConflict):
        modes.resolve("off", {"COMPLEXA_OPT": "exact"})
    with pytest.raises(modes.ModeError):
        modes.resolve("default", {})                                                    # not a mode of this table
    for m in modes.MODES:
        assert modes.route_of(m) == modes.ROUTES[m]
    with pytest.raises(ValueError):
        modes.route_of("turbo")


def test_unknown_mode_names_the_table():
    with pytest.raises(modes.ModeError) as e:
        modes.resolve("turbo", {})
    assert "off|exact|fast|big" in str(e.value)


def test_levers_registry_is_importable_without_torch_or_upstream():
    """levers.py is read by the command line and these tests on a CPU box: torch / proteinfoundation are imported inside functions only."""
    code = "import sys, complexa_opt.levers as L; assert 'torch' not in sys.modules and not any(m.startswith('proteinfoundation') for m in sys.modules); print(len(L.LEVERS))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert int(r.stdout.strip()) == len(levers.LEVERS)


def test_install_refuses_unknown_and_both_bias_sources():
    with pytest.raises(levers.LeverError) as e:
        levers.install(["nope"])
    assert "unknown lever" in str(e.value) and e.value.lever == "nope"
    with pytest.raises(levers.LeverError):
        levers.install(["pair_bias_rows", "pair_bias_fused"])
