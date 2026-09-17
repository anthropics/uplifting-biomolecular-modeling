"""The one mode table: the two words, default fast, off = the stock route (upstream's script), fast = the kit's one tier (batched sampling),
precedence and conflict of --mode / ESM_IF1_OPT, unknown words."""
import pytest

from esm_if1_opt import lines, modes


def test_table():
    assert modes.MODES == ("off", "fast")
    assert modes.DEFAULT_MODE == "fast"
    assert modes.KIT_MODES == {"fast": ("batched_sampling",)} and (modes.STOCK, modes.KIT) == ("stock", "kit") and lines.ROUTES == (modes.STOCK, modes.KIT)
    assert modes.route_of("off") == "stock" and not modes.is_kit("off") and modes.levers("off") == ()
    assert modes.route_of("fast") == "kit" and modes.is_kit("fast") and modes.levers("fast") == ("batched_sampling",)


def test_resolve_precedence_and_conflict():
    assert modes.resolve(None, {}) == "fast"
    assert modes.resolve("off", {}) == "off"
    assert modes.resolve(None, {"ESM_IF1_OPT": "off"}) == "off"
    assert modes.resolve(" FAST ", {}) == "fast"
    assert modes.resolve("off", {"ESM_IF1_OPT": "off"}) == "off"
    with pytest.raises(modes.ModeConflict):
        modes.resolve("off", {"ESM_IF1_OPT": "fast"})


def test_unknown_names_are_usage_errors():
    for bad in ("turbo", "default", "stock"):
        with pytest.raises(modes.ModeError) as e:
            modes.resolve(bad, {})
        assert "off|fast" in str(e.value)
        with pytest.raises(modes.ModeError):
            modes.resolve(None, {"ESM_IF1_OPT": bad})
