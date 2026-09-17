"""Fixtures for the CPU tests: a throwaway site-packages holding caliby/, chroma/ and protpardelle/ trees built from the kits' own
stock/ and fast/ copies of the touched files (so a fresh hash always matches the kit's own copy), with empty package __init__ files. Nothing
upstream is installed or imported; the fixtures exercise the package's tree-state, activation and stock-proof rules."""
import os
import shutil
import sys
import tempfile

from .. import stack

PKG_INITS = ("caliby/__init__.py", "caliby/data/__init__.py", "caliby/data/preprocessing/__init__.py", "caliby/data/preprocessing/atomworks/__init__.py", "caliby/model/__init__.py", "caliby/model/seq_denoiser/__init__.py", "caliby/model/seq_denoiser/denoisers/__init__.py",
             "caliby/model/seq_denoiser/denoisers/seq_design/__init__.py", "caliby/eval/__init__.py", "caliby/eval/eval_utils/__init__.py",
             "chroma/__init__.py", "chroma/layers/__init__.py", "protpardelle/__init__.py", "protpardelle/core/__init__.py", "protpardelle/data/__init__.py")


def core_dir():
    """The shared core's project directory as ``opt/pyproject.toml`` ``[tool.opt_core]`` pins it (the path a box installs with ``pip install -e``)."""
    from opt_core.gates import core_pin
    return core_pin(stack.pyproject_path())["abs_path"]


def pythonpath():
    """PYTHONPATH for a child interpreter of the tests: this package and the pinned core, nothing upstream."""
    return os.pathsep.join([stack.opt_home(), core_dir()])


ADDON_FILES = ("potts.py", "atom_mpnn_denoiser.py", "api.py", "seq_des_utils.py", "inference_dataloader.py", "complexity.py",
               "protpardelle_core_models.py", "protpardelle_data_pdb_io.py")          # the eight replaced modules (stack.TOUCHED minus NEVER_REPLACED)


def kit_sources():
    """(stock, addon): the add-on's own upstream copies and its lever files, per replaced module."""
    kx = stack.kit_dir(stack.KIT_ADDON)
    stock = {n: os.path.join(kx, "stock", n) for n in ADDON_FILES}
    addon = {n: os.path.join(kx, "fast", n) for n in ADDON_FILES}
    return stock, addon


class FakeSitePackages:
    """Builds the three package trees under a temp dir and puts it first on sys.path; ``set_state('stock'|'exact')`` rewrites
    the touched files from the kits' copies. Remove with ``close()``."""

    def __init__(self, with_protpardelle=True):
        self.root = tempfile.mkdtemp(prefix="caliby_opt_fake_sp_")
        self.with_pp = with_protpardelle
        for rel in PKG_INITS:
            if rel.startswith("protpardelle") and not with_protpardelle:
                continue
            p = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").close()
        sys.path.insert(0, self.root)
        self._purge_cache()
        self.set_state("stock")

    def _purge_cache(self):
        import importlib
        importlib.invalidate_caches()
        for m in list(sys.modules):
            if m == "caliby" or m.startswith("caliby.") or m == "chroma" or m.startswith("chroma.") or m == "protpardelle" or m.startswith("protpardelle."):
                del sys.modules[m]

    def path_of(self, name):
        pkg, rel = stack.TOUCHED[name]
        return os.path.join(self.root, pkg, rel)

    def set_state(self, state):
        stock, addon = kit_sources()
        for name, src in stock.items():
            if name in stack.PP_FILES and not self.with_pp:
                continue
            dst = self.path_of(name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        for name, rel in stack.STOCK_SRC.items():                      # tracked by the kit's tree tool, never replaced: the pin's bytes
            dst = self.path_of(name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(stack.tree_home(), rel), dst)
        if state == "exact":
            for name, src in addon.items():
                if name in stack.PP_FILES and not self.with_pp:
                    continue
                shutil.copyfile(src, self.path_of(name))
        self._purge_cache()

    def corrupt(self, name):
        with open(self.path_of(name), "a") as fh:
            fh.write("\n# a foreign byte\n")
        self._purge_cache()

    def close(self):
        try:
            sys.path.remove(self.root)
        except ValueError:
            pass
        self._purge_cache()
        shutil.rmtree(self.root, ignore_errors=True)
