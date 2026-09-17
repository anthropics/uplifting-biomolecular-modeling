"""A meta-path finder is a KIT hook iff the module defining its class lives under a hook directory of this tree — judged by file, never by
class name: an instrument attached from outside the tree whose finder class shares a kit's class name is not a kit hook (the stock proof and
the one-route gate ignore it); a finder defined under a hook directory is one."""
import importlib.util, os, sys

from openfold3_ob0_opt import env, hooks, modes
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
SRC = "class _Finder:\n    def find_spec(self, name, path=None, target=None):\n        return None\nFINDER = _Finder()\n"


def _load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[modname] = mod; spec.loader.exec_module(mod)
    return mod


def test_a_finder_defined_outside_the_tree_is_not_a_kit_hook(tmp_path):
    p = tmp_path / "outside_probe.py"; p.write_text(SRC)
    mod = _load(str(p), "outside_probe_for_test")
    sys.meta_path.insert(0, mod.FINDER)
    try:
        assert type(mod.FINDER).__name__ == "_Finder"                                   # the very class name a kit uses
        assert env.kit_hooks_installed(HOME) == [] and hooks.installed(HOME) == []
        assert env.finder_words(mod.FINDER) == f"_Finder@{os.path.abspath(str(p))}"
    finally:
        sys.meta_path.remove(mod.FINDER); sys.modules.pop("outside_probe_for_test", None)


def test_a_finder_defined_under_a_hook_directory_is_a_kit_hook(tmp_path, monkeypatch):
    home = tmp_path / "tree"
    hd = home / os.path.relpath(modes.hook_dir(str(home), "trunk_kernels"), str(home))     # the kit's hook DIRECTORY (modes.HOOK_DIR)
    hd.mkdir(parents=True)
    p = hd / modes.HOOK_FILE; p.write_text(SRC)
    mod = _load(str(p), "kit_probe_for_test")
    sys.meta_path.insert(0, mod.FINDER)
    try:
        got = env.kit_hooks_installed(home=str(home))
        assert got == [f"_Finder@{os.path.abspath(str(p))}"], got
        inst = hooks.installed(str(home))
        assert [h["kit"] for h in inst] == ["trunk_kernels"] and inst[0]["cls"] == "_Finder"
    finally:
        sys.meta_path.remove(mod.FINDER); sys.modules.pop("kit_probe_for_test", None)


def test_builtin_importers_and_the_interpreter_facts_words():
    words = env.interpreter_facts({})["meta_path"]
    assert all("@" in w for w in words) and any(w.endswith("@builtin") for w in words)   # the frozen/builtin importers have no file
