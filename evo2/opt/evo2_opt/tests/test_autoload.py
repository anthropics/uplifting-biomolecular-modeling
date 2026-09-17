"""The autoload hook: nothing under EVO2_OPT unset/off; a finder under exact; a usage exit for any other value."""
import os
import sys
import unittest
from unittest import mock

from evo2_opt import _autoload


class TestAutoload(unittest.TestCase):
    def tearDown(self):
        sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _autoload._Finder)]

    def test_unset_and_off_install_nothing(self):
        for env in ({}, {"EVO2_OPT": ""}, {"EVO2_OPT": "off"}):
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertIsNone(_autoload.install())
        self.assertFalse(any(isinstance(f, _autoload._Finder) for f in sys.meta_path))

    def test_exact_installs_one_finder_once(self):
        with mock.patch.dict(os.environ, {"EVO2_OPT": "exact"}, clear=True):
            f = _autoload.install()
            self.assertIsInstance(f, _autoload._Finder)
            self.assertIsNone(_autoload.install())                     # idempotent
        self.assertIsNone(f.find_spec("numpy"))                        # only the evo2 package triggers it
        self.assertFalse(f.fired)

    def test_bad_value_exits_2_at_the_evo2_import(self):
        with mock.patch.dict(os.environ, {"EVO2_OPT": "faster"}, clear=True):
            f = _autoload.install()
        with self.assertRaises(SystemExit) as cm, mock.patch("sys.stderr"):
            f.find_spec("evo2.models")
        self.assertEqual(cm.exception.code, 2)
        self.assertNotIn(f, sys.meta_path)


if __name__ == "__main__":
    unittest.main()
