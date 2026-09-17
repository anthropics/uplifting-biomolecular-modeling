"""registry.LEVERS is metadata keyed by the kit's switch names: complete entries, every lever a forward (CLI-route) lever sourced to a
kit file, every mode names real levers, tiers per mode, marker levers have a marker. Values are never here (test_modes_match_env_sh.py
locks the names against env.sh)."""
import os
import re
import unittest

from protenix_opt import registry, stack
from protenix_opt.modes import ARM, MODES, MEM_LEVERS
from protenix_opt.registry import LEVERS

from protenix_opt.tests import _registry_views as views


class TestRegistry(unittest.TestCase):
    def test_every_lever_is_complete(self):
        for name, lv in LEVERS.items():
            self.assertEqual(lv.name, name)
            self.assertEqual(lv.cls, registry.FORWARD, name)
            self.assertIn(lv.tier, (registry.EXACT, registry.TOLERANCE), name)
            self.assertTrue(lv.description.strip() and lv.cells.strip() and lv.source.strip(), name)
            self.assertIn(lv.probe, ("env", "marker"), name)
            self.assertTrue(lv.env_keys or lv.conditional or name in MEM_LEVERS, f"{name}: no switch, no conditional (only the memory line's levers have neither)")
            for k in lv.env_keys + lv.conditional:
                self.assertRegex(k, r"^[A-Z][A-Z0-9_]+$", name)
            self.assertFalse(hasattr(lv, "worker_flags"), "a lever is a switch of the CLI route; there is no other route")

    def test_every_lever_is_sourced_to_a_kit_file(self):
        """Each entry names the env.sh line, the kit README or a kit document it was read from (never a value of its own)."""
        for name, lv in LEVERS.items():
            self.assertRegex(lv.source, r"env\.sh|kit README|README\.md|HAZARDS\.md|graphed\.py|^opt/protenix_opt/[a-z_]+\.py", name)   # a kit file, or the package's own lever module
            m = re.match(r"^(opt/protenix_opt/[a-z_]+\.py)", lv.source)
            if m:
                self.assertTrue(os.path.exists(os.path.join(os.path.dirname(stack.opt_home()), m.group(1))), (name, m.group(1)))   # opt_home() = <tree>/opt

    def test_marker_levers_have_a_marker(self):
        for name, lv in LEVERS.items():
            if lv.probe == "marker":
                self.assertIn(name, stack.MARKERS, name)

    def test_modes_name_real_levers(self):
        for mode, names in MODES.items():
            self.assertEqual(len(names), len(set(names)), mode)
            for n in names:
                self.assertIn(n, LEVERS, f"{mode}: {n}")
        self.assertEqual(MODES["off"], [])
        self.assertEqual(ARM, {"exact": "E", "fast": "T", "big": "T", "off": None})   # big: the default base's arm — fast (modes.big_base)
        self.assertEqual(set(LEVERS), set().union(*(set(v) for v in MODES.values())), "every registered lever is in a mode")

    def test_tiers_per_mode(self):
        for name in MODES["exact"]:
            self.assertEqual(LEVERS[name].tier, registry.EXACT, f"exact mode carries tolerance lever {name}")
        self.assertTrue(any(LEVERS[n].tier == registry.TOLERANCE for n in MODES["fast"]))
        self.assertIn("lazy_init", MODES["fast"], "the start-up lever is in fast too (contract)")

    def test_readme_row_levers(self):
        """pad8 is an E* switch (9.0 rows), glue v2 and MK-PF are E* and T* switches, TriMul v4 is T* only; all row-dependent, none exported by env.sh."""
        from protenix_opt import modes
        for n in ("pad8", "glue_v2", "mk_pf"):
            self.assertTrue(LEVERS[n].extra and LEVERS[n].row_dependent, n)
        self.assertIn("pad8", MODES["exact"]); self.assertNotIn("pad8", MODES["fast"]); self.assertNotIn("trimul_core", MODES["exact"])
        for n in ("glue_v2", "mk_pf"):
            self.assertIn(n, MODES["exact"]); self.assertIn(n, MODES["fast"])
        self.assertIn("hazard #41", LEVERS["mk_pf"].description)
        for key, row in modes.README_ROWS.items():
            for mode in ("exact", "fast"):
                listed = {n for n in MODES[mode] if LEVERS[n].row_dependent and registry.in_row(LEVERS[n], row[mode])}
                keys = set(row[mode]["pre"]) | set(row[mode]["post"])
                self.assertEqual({k for n in listed for k in LEVERS[n].env_keys} & keys, keys, f"{key} {mode}: every row switch belongs to a registered lever")
        for n in ("pad8", "glue_v2", "mk_pf"):
            self.assertFalse(registry.in_row(LEVERS[n], modes.OTHER_ROW["exact"]), n)
        self.assertTrue(all(lv.row_dependent is False for lv in LEVERS.values() if not lv.extra))

    def test_conditional_keys(self):
        """The probe-conditional switches: env.sh's cc 9.0 branch (FPF_TRIMUL_EXACT_NMAX), its prebuilt fast-LN selection
        (INFOPT_FASTLN_PREBUILT) and the PTX_T_TRIMUL=v4 knob keys; the fastln_prebuilt lever has no switch of its own."""
        self.assertEqual(LEVERS["fastln_prebuilt"].env_keys, ()); self.assertEqual(LEVERS["fastln_prebuilt"].conditional, ("INFOPT_FASTLN_PREBUILT",))
        self.assertIn("third_party/fastln_prebuilt*", LEVERS["fastln_prebuilt"].source)
        self.assertEqual(views.conditional_keys_for(MODES["exact"]), {"INFOPT_FASTLN_PREBUILT"})
        self.assertEqual(views.conditional_keys_for(MODES["fast"]), {"INFOPT_FASTLN_PREBUILT", *views.KNOB_KEYS, *views.ATT_KEYS}, "trimul_exact is an ARM=E lever")


if __name__ == "__main__":
    unittest.main()
