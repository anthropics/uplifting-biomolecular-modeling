"""The package declares its own EF2_* switch names; anything else in that namespace is named once at startup (cli.stray_switch_note)."""
import os
import unittest

from esmfold2_opt import cli, modes


class StraySwitches(unittest.TestCase):
    def test_declared_names_cover_the_mode_table_and_the_data_paths(self):
        d = cli.declared_switches()
        for name in (modes.ENV_GRAPH_BUDGET, modes.ENV_GRAPH_LRU_SAMPLER, modes.ENV_W4_IDPROBE, modes.ENV_MK, modes.ENV_MSA):
            self.assertIn(name, d)
        self.assertTrue(all(n.startswith("EF2_") for n in d))

    def test_undeclared_names_are_named_declared_ones_are_not(self):
        env = {modes.ENV_GRAPH_BUDGET: "1300", "EF2_W4_T9_CELLS": "x", "EF2_XL_FREE": "a", "EF2_HF_HOME": "/w", "HF_HOME": "/w", "PATH": "/bin"}
        self.assertEqual(cli.stray_switches(env), ["EF2_HF_HOME", "EF2_W4_T9_CELLS", "EF2_XL_FREE"])
        note = cli.stray_switch_note(env)
        self.assertTrue(note.startswith("[esmfold2-opt] NOTE ") and "EF2_HF_HOME, EF2_W4_T9_CELLS, EF2_XL_FREE" in note)
        self.assertIsNone(cli.stray_switch_note({modes.ENV_GRAPH_BUDGET: "0", "HOME": "/h"}))


if __name__ == "__main__":
    unittest.main()
