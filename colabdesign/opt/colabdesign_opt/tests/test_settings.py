"""The pinned settings are ONE file read at run time (settings.py): the loader returns BindCraft's own json after checking it exists at
the path stock/PINS.json declares (an absent file, or a pin naming another path, is refused by name without reading it), names the file
in its record, and reads the four phase counts from the file's own keys. No value of the file is typed in the package: this test compares
against the file, never against literals."""
import json
import os
import shutil
import tempfile
import unittest

from . import _stubs
from colabdesign_opt import names, settings


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestSettings(unittest.TestCase):
    def test_load_is_the_pinned_file(self):
        S = settings.load(_stubs.TREE)
        pins = _stubs.pins()
        self.assertEqual(S["name"], "default_4stage_multimer"); self.assertEqual(S["file"], settings.SETTINGS_RELPATH)
        self.assertEqual(S["file"], pins[settings.PIN_KEY]["file"])
        self.assertEqual(S["sha256"], names.sha256_file(os.path.join(_stubs.TREE, settings.SETTINGS_RELPATH)))   # observed, compared against nothing
        with open(settings.path(_stubs.TREE)) as fh:
            self.assertEqual(S["advanced"], json.load(fh))                                                        # the file's own dict, untouched
        self.assertEqual(settings.record(S), {"name": S["name"], "file": S["file"], "sha256": S["sha256"]})
        self.assertTrue(os.path.isfile(settings.filters_path(_stubs.TREE)))

    def test_iteration_counts_are_the_files_keys(self):
        S = settings.load(_stubs.TREE); a = S["advanced"]
        self.assertEqual(settings.iteration_counts(a), {"soft": a["soft_iterations"], "temp": a["temporary_iterations"], "hard": a["hard_iterations"], "greedy": a["greedy_iterations"]})
        with self.assertRaises(settings.SettingsError) as cm:
            settings.iteration_counts({k: v for k, v in a.items() if k != "hard_iterations"})
        self.assertIn("hard_iterations", str(cm.exception))

    def test_a_different_file_is_refused_not_read(self):
        """An absent file is refused by name; a file present at the declared path loads whatever its bytes (the check is presence + the
        declared path, not a byte re-check); a pin naming another path is refused by name too."""
        tmp = tempfile.mkdtemp(prefix="cd_opt_settings_")
        os.makedirs(os.path.join(tmp, "stock")); shutil.copy(_stubs.PINS, os.path.join(tmp, "stock", "PINS.json"))
        with self.assertRaises(settings.SettingsError) as cm:
            settings.load(tmp)
        self.assertIn("pinned settings file absent", str(cm.exception))
        dst = os.path.join(tmp, settings.SETTINGS_RELPATH); os.makedirs(os.path.dirname(dst))
        with open(settings.path(_stubs.TREE)) as fh:
            body = fh.read()
        with open(dst, "w") as gh:
            gh.write(body.replace("4stage", "3stage"))
        loaded = settings.load(tmp)                                                    # present at the declared path: loads, whatever its bytes
        self.assertEqual(loaded["sha256"], names.sha256_file(dst)); self.assertIn("3stage", json.dumps(loaded["advanced"]))
        pins = json.load(open(_stubs.PINS)); pins[settings.PIN_KEY]["file"] = "stock/src/bindcraft/settings_advanced/other.json"
        json.dump(pins, open(os.path.join(tmp, "stock", "PINS.json"), "w"))
        with self.assertRaises(settings.SettingsError):
            settings.load(tmp)                                                                            # the pin names another file: refused



class TestGivenFiles(unittest.TestCase):
    """bindcraft.py's `--advanced FILE` / `--filters FILE`: a given file is read as it stands (name = its stem, file = its absolute path, no pin
    check — the pin names the DEFAULT only); a file that does not exist is refused by name; absent = BindCraft's defaults in the vendored tree."""

    @unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
    def test_given_advanced_and_filters(self):
        tmp = tempfile.mkdtemp()
        adv = os.path.join(tmp, "my_4stage.json"); flt = os.path.join(tmp, "my_filters.json")
        shutil.copy(settings.path(_stubs.TREE), adv); shutil.copy(settings.filters_path(_stubs.TREE), flt)
        S = settings.load(_stubs.TREE, adv, flt)
        self.assertEqual((S["name"], S["file"], S["path"], S["filters"]), ("my_4stage", adv, adv, flt))
        self.assertEqual(S["sha256"], names.sha256_file(adv)); self.assertEqual(S["advanced"], settings.load(_stubs.TREE)["advanced"])
        D = settings.load(_stubs.TREE)                                                      # absent = the defaults: the pinned tree-relative label, the vendored filters
        self.assertEqual((D["name"], D["file"], D["filters"]), (settings.NAME, settings.SETTINGS_RELPATH, settings.filters_path(_stubs.TREE)))
        with self.assertRaises(settings.SettingsError) as cm:
            settings.load(_stubs.TREE, os.path.join(tmp, "absent.json"))
        self.assertIn("--advanced", str(cm.exception)); self.assertIn("no such file", str(cm.exception))
        with self.assertRaises(settings.SettingsError) as cm:
            settings.load(_stubs.TREE, None, os.path.join(tmp, "absent_filters.json"))
        self.assertIn("--filters", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
