"""The mode table is the one source: its names and its lock against the kit's own constants (read from the kit's entry file by AST)."""
import pytest

from e1_opt import modes, stack
from e1_opt.tests import _stubs


def test_mode_table_names_and_default():
    assert modes.MODES == ("off", "exact") and tuple(modes.MODE_TABLE) == modes.MODES   # the stock word first; the table in the tuple's order
    assert modes.DEFAULT_MODE == "exact"
    assert modes.MODE_TABLE["exact"].active and modes.MODE_TABLE["exact"].kit_mode == modes.KIT_MODE == "eager"
    assert not modes.MODE_TABLE["off"].active and modes.MODE_TABLE["off"].kit_mode is None
    assert modes.KIT_MODES == ("exact",) and modes.kernels_expected("exact") == modes.kernels_expected("off") == modes.KERNELS_ALL
    assert not any(hasattr(modes, n) for n in ("MEMORY_LEVER", "memory_levers", "KIT_MODE_REQUESTED"))   # one kit mode, no memory mode


def test_check_mode_normalises_and_refuses_unknown_and_unshipped():
    assert modes.check_mode(None) == "exact" and modes.check_mode(" OFF ") == "off" and modes.check_mode("Exact") == "exact"
    for name in ("faster", "low-memory"):
        with pytest.raises(ValueError):
            modes.check_mode(name)
    for name in modes.NOT_SHIPPED:                                       # the family's other words: refused by name, saying they are not shipped
        with pytest.raises(ValueError) as ex:
            modes.check_mode(name)
        assert f"mode {name}: not shipped by this kit" in str(ex.value)
    assert modes.NOT_SHIPPED == ("fast", "big")


def test_kit_constants_read_from_the_kit_entry_and_locked():
    _stubs.tree_or_skip()
    kc = modes.kit_constants(stack.kit_dir())
    assert isinstance(kc["KIT"], str) and kc["KIT"].startswith("v") and kc["MODE"] == modes.KIT_MODE
    assert modes.lock_check(stack.kit_dir()) == []
    assert modes.kit_version(stack.kit_dir()) == kc["KIT"]


def test_mode_lock_fails_on_a_changed_kit_mode(tmp_path):
    d = tmp_path / "kit"
    d.mkdir()
    (d / "__init__.py").write_text('KIT = "vX"\nMODE = "fused"\n')
    bad = modes.lock_check(str(d))
    assert bad and "expects 'eager'" in bad[0]
    (d / "__init__.py").write_text('MODE = "eager"\n')
    assert modes.lock_check(str(d)) and "KIT not found" in modes.lock_check(str(d))[0]
