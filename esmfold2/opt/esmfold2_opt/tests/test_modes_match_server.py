"""Modes resolve to the kit server's own MODES table, by name — and the package's two kit modes are what the kit says they are:
``fast`` is the server's default mode, ``exact`` is the kit's bit-exact lever set — server mode ``opt7x``, string-equal to the kit
file's own definition, the same line on every variant, never an override switch. Also locks the premises the resolution rests on:
configure()'s parsing of a tuple, its reading of EF2_MK / EF2_MSA for any mode (and the tuple's own field when they are unset), and
the variant map against the driver's checkpoint table and stock/PINS.json."""
import ast
import os
import re
import unittest

from esmfold2_opt import modes, registry, stack
from esmfold2_opt.tests import _stubs


class TestModesMatchServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kit = _stubs.require_kit()
        cls.server = os.path.join(cls.kit, modes.SERVER_RELPATH)
        cls.table = modes.server_table(cls.server)
        cls.src = open(cls.server, encoding="utf-8").read()

    def test_every_kit_mode_names_a_server_mode(self):
        self.assertEqual(set(self.table), {km.server_mode for km in modes.KIT_MODES.values()})   # the table carries exactly the lever sets the package names: no entry without a mode
        for name, km in modes.KIT_MODES.items():
            self.assertIn(km.server_mode, self.table, f"{name} -> {km.server_mode} is not in the server table")
        for m in modes.MODES:
            self.assertTrue(m == "off" or m in modes.KIT_MODES)

    def test_fast_is_the_package_default(self):
        self.assertEqual(modes.DEFAULT_MODE, "fast"); self.assertEqual(modes.KIT_MODES["fast"].server_mode, "opt14_msa")
        self.assertEqual(modes.KIT_MODES["fast"].overrides, {})

    def test_exact_is_the_kit_opt7x_definition(self):
        """The resolver's exact line is the kit's own ``opt7x`` entry, string-equal to the literal in the kit file, for every variant;
        the line names no override switch, so the activation line is the bare server mode name."""
        m = re.search(r"['\"]opt7x['\"]\s*:\s*(\([^()]*\))", self.src)
        self.assertIsNotNone(m, "the kit server no longer defines opt7x")
        literal = m.group(1)
        self.assertEqual(modes.KIT_MODES["exact"].server_mode, "opt7x")
        self.assertEqual(modes.KIT_MODES["exact"].overrides, {})
        for v in modes.VARIANTS:
            res = modes.resolve("exact", v, self.kit)
            self.assertEqual(res.server_mode, "opt7x")
            self.assertEqual(res.entry, ast.literal_eval(literal))
            self.assertEqual(repr(res.entry), literal)                    # the kit's own spelling, character for character
            self.assertEqual(modes.describe_line(res), "opt7x")
            self.assertEqual(len(res.entry), 4, "opt7x carries its own W4 field and its own group field: EF2_W4 / EF2_MK / EF2_MSA are never read for it")
            groups = modes.parse_extra(res.entry[3])
            self.assertNotIn("mk", groups); self.assertNotIn("msa", groups)
            self.assertEqual(res.levers, ["fused"] + res.entry[2].split(",") + groups["atom"] + ["fz"] + res.entry[1].split("+") + groups["msa2"] + groups["pair"] + groups["hoist"] + groups["ln"] + groups["dit"])   # the kit's fields, in configure()'s order
            self.assertEqual(res.levers_for_variant + res.levers_not_for_variant if False else sorted(res.levers_for_variant + res.levers_not_for_variant), sorted(res.levers))
            self.assertEqual(res.levers_not_for_variant, [n for n in res.levers if not registry.acts_on(registry.LEVERS[n], v)])   # ONE set; the registry names the levers that act per variant (the MSA-module hoists on the Full model, rg on the Fast model)
            self.assertEqual(set(res.levers_not_for_variant), {"fast": {"mh", "trimul", "glue", "xtr"}, "full_msa": {"rg"}, "full_nomsa": {"rg", "mh"}}[v])   # xtr: the MSA module's PairTransition exists on the Full model only

    def test_resolution_reads_the_table(self):
        for mode in modes.KIT_MODES:
            if modes.KIT_MODES[mode].blocked:                                 # a BLOCKED mode refuses in resolve() by name (test_big.py)
                continue
            for v in modes.VARIANTS:
                res = modes.resolve(mode, v, self.kit)
                self.assertEqual(res.entry, self.table[res.server_mode])
                self.assertEqual(res.composition["base"], "fused")                 # every mode of the server table runs on the fused base (ef2_server.configure's own invariant)
                self.assertFalse(res.levers_unknown, f"{mode}/{v}: flags the registry does not describe: {res.levers_unknown}")
                for n in res.levers_for_variant + res.levers_not_for_variant:
                    self.assertIn(n, registry.LEVERS)
                if v == "fast":                                   # no msa_encoder: the package keeps the MSA-encoder levers off there (configure(off=)), named not_for_variant
                    self.assertTrue(set(res.levers_not_for_variant) >= {n for n in res.levers if n in registry.LEVERS and registry.LEVERS[n].variants == registry.FULL_MODEL})
                    self.assertNotIn("rg", res.levers_not_for_variant)
                else:                                             # the Full model (full_msa and full_nomsa): every lever of the line acts except rg (Fast model only) and, without an MSA, mh
                    self.assertEqual(set(res.levers_not_for_variant), {n for n in res.levers if not registry.acts_on(registry.LEVERS[n], v)})
                    self.assertTrue(set(res.levers_not_for_variant) <= {"rg", "mh"})

    def test_exact_never_sets_the_override_switches(self):
        """Regression: exact names no EF2_MK / EF2_MSA, on any variant — the kit's tuple decides (no ``mk``, no ``msa``), and every
        lever of the line is bit-exact against the kit line."""
        for v in modes.VARIANTS:
            res = modes.resolve("exact", v, self.kit)
            self.assertEqual(res.overrides, {})
            self.assertFalse(res.composition["mk"]); self.assertEqual(res.composition["msa"], [])
            self.assertNotIn("mk", res.levers); self.assertNotIn("msa", res.levers)
            for n in res.levers:
                self.assertFalse(registry.LEVERS[n].tier_vs_kit_line.startswith("T2"), f"{n} is a tolerance-tier lever but sits in the exact line")

    def test_configure_parsing_premises(self):
        """The tuple grammar composition() mirrors: field 2 split on '+', field 3 on ',', field 4 = ';'-separated '<group>:<flags>' groups
        (parse_extra), the group names one table in both files."""
        body = self.src[self.src.index("def configure("):]
        body = body[:body.index("\ndef ", 10)]
        pe = self.src[self.src.index("def parse_extra("):]
        pe = pe[:pe.index("\ndef ", 10)]
        self.assertRegex(body, r"\.split\(['\"]\+['\"]\)")          # opt flags
        self.assertRegex(body, r"\.split\(['\"],['\"]\)")           # w4 flags
        self.assertRegex(pe, r"\.split\(['\"];['\"]\)")             # the group field
        self.assertIn('partition(":")', pe)
        self.assertIn('"mk"', body); self.assertIn('groups.get("msa"', body)
        srv_groups = ast.literal_eval(next(n.value for n in ast.parse(self.src).body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "GROUPS" for t in n.targets)))
        self.assertEqual(tuple(srv_groups), modes.GROUPS)                # one vocabulary of groups in the server and the package
        for g in modes.GROUPS:
            self.assertIn(g, modes.GROUP_INSTALL_ORDER)

    def test_server_reads_the_override_switches_for_any_mode(self):
        """The premise of KIT_MODES.overrides: configure() consults EF2_MK / EF2_MSA from the environment regardless of the mode."""
        body = self.src[self.src.index("def configure("):]
        body = body[:body.index("\ndef ", 10)]
        self.assertRegex(body, r'os\.environ\.get\("EF2_MK"')
        self.assertRegex(body, r'os\.environ\.get\("EF2_MSA"')
        self.assertRegex(body, r'if env_mk is not None else')       # unset switches: the tuple's own field 4 decides (the premise of a mode with no overrides)
        self.assertRegex(body, r'if env_msa is not None else')
        for name, km in modes.KIT_MODES.items():
            entry = self.table[km.server_mode]
            if set(km.overrides) & {modes.ENV_MK, modes.ENV_MSA}:              # a lever override on the server's table (the memory modes' switches are the add-ons', not the server's)
                self.assertGreater(len(entry), 2, f"{name}: overrides on a mode without a W4 field would let EF2_W4 leak in")
            for k in km.overrides:
                self.assertIn(k, (modes.ENV_MK, modes.ENV_MSA, modes.ENV_GRAPH_BUDGET, modes.ENV_GRAPH_CAPTURE, modes.ENV_W4_IDPROBE))
        # the graph budget and the capture policy are ef2_opt's own import-time switches (CFG.graph_budget_tokens / CFG.graph_capture): a mode override is exported before the server import
        opt_src = open(os.path.join(os.path.dirname(self.server), "ef2_opt.py"), encoding="utf-8").read()
        self.assertRegex(opt_src, r'graph_budget_tokens = int\(os\.environ\.get\("' + modes.ENV_GRAPH_BUDGET + r'", "0"\)\)')
        self.assertRegex(opt_src, r'graph_capture = os\.environ\.get\("' + modes.ENV_GRAPH_CAPTURE + r'", "1"\)')
        self.assertRegex(opt_src, r"def _over_graph_budget\(")

    def test_variant_map_against_the_pins(self):
        self.assertEqual(set(modes.SERVER_VARIANT.values()), {"fast", "full"})           # configure()'s variant words (has_msa: 'full')
        pins = _stubs.pins_or_none()
        if pins is None:
            self.skipTest("stock/PINS.json not found: variant -> checkpoint check skipped")
        for v in modes.VARIANTS:
            self.assertEqual(pins["variants"][v]["kit_variant"], modes.SERVER_VARIANT[v], f"PINS.json variants.{v}.kit_variant != modes.SERVER_VARIANT")
            self.assertTrue(stack.variant_repo(v, pins).startswith("biohub/ESMFold2"), v)
            self.assertEqual(bool((pins["variants"][v]).get("msa")), modes.VARIANT_USES_MSA[v])

    def test_check_command_resolves_without_torch(self):
        from esmfold2_opt import cli
        rep = cli.activate("exact", "full_msa", dry_run=True)
        self.assertTrue(rep.get("dry_run"))
        self.assertEqual(rep["server_mode"], modes.KIT_MODES["exact"].server_mode)
        self.assertEqual(rep["server_line"], "opt7x")
        self.assertEqual(rep["overrides"], {})
        self.assertIn("fused", rep["levers_planned"])
        self.assertIn("kernel_key", rep)


if __name__ == "__main__":
    unittest.main()
