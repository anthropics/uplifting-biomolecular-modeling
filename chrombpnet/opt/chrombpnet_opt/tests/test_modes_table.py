"""The mode table: fast is the default, off is stock, exact is the deterministic set, any other name is a plain error on every route; the
registry is data (the levers the kit applies under fast, the kit's own switch names); nothing of the kit is imported by the package's import."""
import json
import os
import subprocess
import sys

import pytest

from chrombpnet_opt import modes, registry
from . import _stubs


def test_mode_table():
    assert modes.MODES == ("off", "exact", "fast")
    assert modes.DEFAULT_MODE == "fast"
    assert sorted(n for n in dir(modes) if n.isupper()) == ["DEFAULT_MODE", "DET_MODES", "ENV", "ENV_DET", "KIT_LINE_RELPATH", "MODES", "STOCK_CONSOLE_SCRIPT", "SUBCOMMAND"]   # the mode table's whole constant surface
    assert modes.DET_MODES == ("exact",)
    assert modes.check_mode("FAST") == "fast" and modes.check_mode("off") == "off"
    with pytest.raises(ValueError) as e:
        modes.check_mode("turbo")
    assert str(e.value) == "unknown mode 'turbo' (expected off|exact|fast)"


def test_documented_line_adds_no_flag(tmp_path):
    kit = _stubs.make_tree(str(tmp_path / "tree"))
    args = ["-cm", "m.h5", "-r", "r.bed", "-g", "g.fa", "-c", "c.sizes", "-op", "out/p"]
    line = modes.documented_line(kit, args, python="/usr/bin/python")
    assert line == ["/usr/bin/python", os.path.join(kit, "tf", "pred_bw_fast.py")] + args


def test_unknown_mode_refused_by_name_enable(tmp_path, capsys, monkeypatch):
    import chrombpnet_opt
    kit = _stubs.make_tree(str(tmp_path / "tree"))
    monkeypatch.setenv("CHROMBPNET_OPT_HOME", _stubs.tree_of(kit))
    rep = chrombpnet_opt.stack.activate("turbo", quiet=False, dry_run=True)
    assert rep["active"] is False and rep["reason"] == "unknown mode 'turbo' (expected off|exact|fast)"
    assert capsys.readouterr().out.splitlines()[0] == "[chrombpnet-opt] NOT ACTIVE mode=turbo reason=unknown mode 'turbo' (expected off|exact|fast)"
    with pytest.raises(chrombpnet_opt.ActivationError):
        chrombpnet_opt.stack.activate("turbo", strict=True, dry_run=True, quiet=True)


@pytest.mark.parametrize("verb", ["pred_bw", "check", "warm"])
def test_unknown_mode_is_a_plain_error(tmp_path, verb):
    """An unknown `--mode` name is an argparse usage error on every verb (exit 2, the reason on stderr): no mode line, nothing activated, nothing run."""
    kit = _stubs.make_tree(str(tmp_path / "tree"))
    op = str(tmp_path / "o" / "p")
    p = subprocess.run([sys.executable, "-m", "chrombpnet_opt", verb, "--mode", "turbo"] + (["-op", op] if verb == "pred_bw" else []),
                       env=_stubs.base_env(kit), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.returncode == 2, p.stderr
    assert p.stdout == "" and "argument --mode: unknown mode 'turbo' (expected off|exact|fast)" in p.stderr
    assert not os.path.exists(op + "_kit_call.json")
    p = subprocess.run([sys.executable, "-m", "chrombpnet_opt", verb, "--help"], env=_stubs.base_env(kit), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.returncode == 0 and "--mode off|exact|fast" in p.stdout


def test_env_route_unknown_mode_is_refused(tmp_path):
    """CHROMBPNET_OPT=<unknown name> (no argparse there) is refused at activation by name, exit 3 on the verbs."""
    kit = _stubs.make_tree(str(tmp_path / "tree"))
    p = subprocess.run([sys.executable, "-m", "chrombpnet_opt", "check"], env=_stubs.base_env(kit, {"CHROMBPNET_OPT": "turbo"}), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.returncode == 3 and p.stdout.splitlines()[0] == "[chrombpnet-opt] NOT ACTIVE mode=turbo reason=unknown mode 'turbo' (expected off|exact|fast)"


def test_registry_is_data():
    assert set(registry.ROUTES) == set(_stubs.ROUTES)
    for name, lever in registry.LEVERS.items():
        d = lever.as_dict()
        assert set(d) == {"name", "what", "kit_file"} and d["name"] == name and d["what"] and d["kit_file"]
        assert ":" in d["kit_file"] or d["kit_file"].endswith(".py") or "/" in d["kit_file"]
        assert "tier" not in d["what"].lower() and "not wired" not in d["what"].lower()
    assert sorted(n for n in dir(registry) if n.isupper()) == ["ARCH_TILES", "FASTDEFAULT", "KIT_CACHE_TAR_NAME", "KIT_RECORD_NAME", "KIT_SWITCH_DET_NAMES", "KIT_SWITCH_NAMES", "LEVERS", "PRED_BW_FAST", "ROUTES"]   # the registry's whole constant surface: the levers the kit applies, the kit's own switch names — no inventory of anything else
    assert sorted(registry.LEVERS) == sorted(registry.LEVERS) and "lp_model" not in registry.LEVERS
    assert "forward_route" in registry.LEVERS and "jit_cache" in registry.LEVERS
    assert "CHROMBPNET_FASTKIT_RECORD" in registry.KIT_SWITCH_NAMES and "CHROMBPNET_JIT_CACHE_TAR" in registry.KIT_SWITCH_NAMES and "CHROMBPNET_FASTKIT_FORWARD" not in registry.KIT_SWITCH_NAMES
    assert tuple(sorted(registry.KIT_SWITCH_NAMES)) == registry.KIT_SWITCH_NAMES     # sorted, no duplicates


def test_package_import_is_light(tmp_path):
    """`import chrombpnet_opt` (what the .pth does) loads the package module only; the finder is not installed without CHROMBPNET_OPT."""
    kit = _stubs.make_tree(str(tmp_path / "tree"))
    code = "import sys, json, chrombpnet_opt, chrombpnet_opt._autoload as a; print(json.dumps([sorted(m for m in sys.modules if m.startswith('chrombpnet_opt')), a.FINDER is None, any(isinstance(f, a.Finder) for f in sys.meta_path)]))"
    p = subprocess.run([sys.executable, "-c", code], env=_stubs.base_env(kit), stdout=subprocess.PIPE, text=True, check=True)
    mods, no_finder, in_meta = json.loads(p.stdout)
    assert mods == ["chrombpnet_opt", "chrombpnet_opt._autoload"] and no_finder and not in_meta
