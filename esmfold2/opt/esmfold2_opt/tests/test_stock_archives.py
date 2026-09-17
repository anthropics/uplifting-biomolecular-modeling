"""The source archives under stock/ (stock/PINS.json "upstream" + "archive_recipe"): each archive named there that is present holds one
`<name>-<commit8>/` tree with every path the recipe names, and carries every file its own setup.py opens at build time (an archive pip
cannot build is not an install of the pin). Byte sameness of an archive is git's job (the commit that added it), not a runtime hash
re-check. A tree without the archives (the online install route at the pinned commits, environment/requirements.lock, is the install)
skips these locks by name. Tree-present only; no pip, no network."""
import json
import os
import re
import tarfile
import unittest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # opt/esmfold2_opt
TREE = os.path.dirname(os.path.dirname(PKG))                                  # esmfold2/
STOCK = os.path.join(TREE, "stock")


def _tree_present():
    return os.path.isfile(os.path.join(STOCK, "PINS.json"))


def recipe_paths(recipe: str) -> dict:
    """`archive_recipe`'s per-fork path lists: the parenthetical `name: p1 p2 ...; name: ...` -> {name: [paths]}."""
    m = re.search(r"\(([^)]*)\)", recipe)
    assert m, recipe
    return {part.split(":", 1)[0].strip(): part.split(":", 1)[1].split() for part in m.group(1).split(";") if ":" in part}


@unittest.skipUnless(_tree_present(), "release tree not present around the package (installed copy): archive locks skipped")
class TestStockArchives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pins = json.load(open(os.path.join(STOCK, "PINS.json"), encoding="utf-8"))
        cls.paths = recipe_paths(cls.pins["archive_recipe"])

    def _archive(self, name):
        pin = self.pins["upstream"][name]
        path = os.path.join(TREE, pin["archive"])
        if not os.path.isfile(path):
            self.skipTest(f"{pin['archive']}: upstream source archive not in this tree (stock installs by the online route: {pin['install']})")
        return pin, path

    def test_one_prefixed_tree_with_the_recipe_paths(self):
        for name in self.pins["upstream"]:
            pin, path = self._archive(name)
            prefix = f"{name}-{pin['commit'][:8]}/"
            with tarfile.open(path, "r:gz") as tf:
                members = tf.getnames()
            outside = [m for m in members if not (m.startswith(prefix) or m == prefix.rstrip("/"))]          # the top directory itself has no slash
            self.assertTrue(members and not outside, f"{pin['archive']}: members outside {prefix}: {outside[:5]}")
            self.assertIn(name, self.paths, f"archive_recipe names no paths for {name}")
            rel = {m[len(prefix):] for m in members}
            for p in self.paths[name]:
                self.assertTrue(p in rel or any(r.startswith(p + "/") for r in rel), f"{pin['archive']}: recipe path {p} missing")

    def test_the_files_setup_py_opens_are_in_the_archive(self):
        """pip builds the archive with its setup.py: every file that setup.py opens (the fork's setup.py reads README.md) must be there."""
        for name in self.pins["upstream"]:
            pin, path = self._archive(name)
            prefix = f"{name}-{pin['commit'][:8]}/"
            with tarfile.open(path, "r:gz") as tf:
                rel = {m[len(prefix):] for m in tf.getnames()}
                if "setup.py" not in rel:
                    continue
                setup_py = tf.extractfile(prefix + "setup.py").read().decode("utf-8")
            opened = sorted(set(re.findall(r"""open\(\s*["']([^"']+)["']""", setup_py)))
            self.assertTrue(opened, f"{pin['archive']}: setup.py opens no file (the lock has nothing to check)")
            for f in opened:
                self.assertTrue(f in rel, f"{pin['archive']}: setup.py opens {f}, which the archive omits (pip cannot build it)")
                self.assertIn(f, self.paths.get(name, []), f"{pin['archive']}: setup.py opens {f}, which archive_recipe's path list omits")


if __name__ == "__main__":
    unittest.main()
