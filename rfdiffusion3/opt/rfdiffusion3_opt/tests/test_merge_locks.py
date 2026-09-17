"""Merge locks: one mode table (no copy in the autoload finder, none in run.sh), no run-parameter preset, one stock caller, no lever switch
outside modes.py / the pins, run.sh in step with the command list, no served-line or packing vocabulary in the house files, no
labels, no house file beside the kit."""
import os
import re
import unittest

from .. import _autoload, cli, manifest, modes, registry, report, stack

TREE = registry.tree_home()
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _code_only(path):
    """The file's code without docstrings (Python) and without comment lines: what a lock on literals should read."""
    text = open(path, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    if path.endswith(".py"):
        import ast
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
                for i in range(node.lineno - 1, node.end_lineno):
                    lines[i] = ""
    return "\n".join(l for l in lines if not l.lstrip().startswith("#"))


def _house_files():
    """Every house-authored file of the tree (the kit directory excluded)."""
    out = []
    for root, dirs, files in os.walk(TREE):
        if registry.KIT_RELPATH.replace("/", os.sep) in root or "__pycache__" in root or ".pytest_cache" in root:
            continue
        for f in files:
            if f.endswith((".py", ".sh", ".env", ".md", ".json", ".toml", ".txt", ".pth")):
                out.append(os.path.join(root, f))
    return out


class TestMergeLocks(unittest.TestCase):
    def test_one_mode_table(self):
        """`RFD3_HOIST` / `RFD3_FZT` as switch values live in modes.py (the table) and stock/PINS.json (the stock arm's forbidden list);
        every other house file names them only in prose or reads them through the table."""
        table = open(os.path.join(PKG, "modes.py"), encoding="utf-8").read()
        self.assertIn('KitMode("exact", {KIT_SWITCH: "1", KIT_FZT_SWITCH: "1", KIT_GRAPH_SWITCH: "1", KIT_INIT_CHUNK_SWITCH: "1"}', table)
        for switch in ("RFD3_HOIST", "RFD3_FZT"):
            pattern = re.compile(switch + r'''["']?\s*[:=]\s*["']?1\b''')
            hits = []
            for p in _house_files():
                if "/tests/" in p or not p.endswith((".py", ".sh", ".env")):
                    continue
                if pattern.search(_code_only(p)):
                    hits.append(p)
            rels = sorted(os.path.relpath(p, TREE) for p in hits)
            self.assertTrue(set(rels) <= {"opt/rfdiffusion3_opt/modes.py"}, (switch, rels))     # nowhere but the table

    def test_autoload_restates_the_table(self):
        """_autoload.py runs before the package imports anything: its copies of the mode names, the exit code and the
        declared switches equal their sources, and its top level imports nothing of the package or the core."""
        self.assertEqual(_autoload.MODES, modes.MODES)
        self.assertEqual(_autoload.EXIT_NOT_ACTIVE, report.EXIT_NOT_ACTIVE)
        self.assertEqual(_autoload.ENV, cli.ENV_MODE)
        self.assertIs(report.TAG, _autoload.TAG)                                # one spelling of the kit's tag: report takes it from the .pth module
        self.assertEqual(report.PREFIX, "[%s]" % _autoload.TAG)
        self.assertEqual(_autoload.TRIGGERS, ("rfd3",))
        from .. import det
        self.assertEqual(set(_autoload.ENV_NAMES), {cli.ENV_MODE, report.ENV_ALLOW_PARTIAL, manifest.CACHE_ENV, det.ENV_LEVEL})   # the declared package switches: the mode, the allowance, the cache dir, the det level
        self.assertEqual(tuple(stack.PACKAGE_ENV), _autoload.ENV_NAMES)
        code = _code_only(os.path.join(PKG, "_autoload.py"))
        top = [l for l in code.splitlines() if l.startswith(("import ", "from "))]
        self.assertEqual(top, ["import os", "import sys"])                      # the core and the package are imported inside functions / under a set variable only
        self.assertIsNone(_autoload.FINDER)                                     # this process has no RFDIFFUSION3_OPT
        s = _autoload.spec()
        self.assertEqual((s.env, s.package, s.tag, s.triggers, s.exit_not_active, s.on_unknown), ("RFDIFFUSION3_OPT", "rfdiffusion3_opt", "rfdiffusion3-opt", ("rfd3",), 3, "exit_now"))
        self.assertEqual(s.modes, modes.MODES)

    def test_no_lever_switch_in_configs(self):
        for f in os.listdir(os.path.join(TREE, "configs")):
            text = open(os.path.join(TREE, "configs", f), encoding="utf-8").read()
            for line in text.splitlines():
                code = line.split("#", 1)[0]
                for name in list(stack.lever_names()) + list(stack.upstream_switches()):
                    self.assertNotRegex(code, rf"\b{name}=", f"{f}: sets {name}")
            self.assertNotIn("PACK_K", text)
            self.assertNotIn("RFDIFFUSION3_OPT=", text.replace("RFDIFFUSION3_OPT=exact|fast|off", "").replace("RFDIFFUSION3_OPT=exact rfd3", ""))

    def test_run_sh_commands(self):
        text = open(os.path.join(TREE, "run.sh"), encoding="utf-8").read()
        m = re.search(r'case "\$CMD" in ([a-z|]+)\)', text)
        self.assertEqual(sorted(m.group(1).split("|")), sorted(cli.COMMANDS))
        self.assertEqual(set(cli.COMMANDS), {"design", "check", "warm"})
        self.assertNotIn("serve", m.group(1))
        self.assertIsNone(re.search(r'case "\$MODE" in', text))                # no second copy of the mode table: the package judges the mode
        self.assertIn("check_pins.py", text)
        self.assertIn("RFDIFFUSION3_STOCK_PYTHON", text)

    def test_one_stock_caller_and_no_preset_table(self):
        callers = [p for p in _house_files() if re.search(r'''["']rfd3\.cli["']''', open(p, encoding="utf-8", errors="replace").read()) and p.endswith(".py") and "/tests/" not in p]
        self.assertEqual(sorted(os.path.relpath(p, TREE) for p in callers), ["opt/rfdiffusion3_opt/stock_design.py"])
        presets = [p for p in _house_files() if re.search(r"inference_sampler\.step_scale\s*[=:]\s*[\"\']?\d", _code_only(p)) and p.endswith((".py", ".sh", ".env")) and "/tests/" not in p]
        self.assertEqual(presets, [])                                             # no file of the kit carries a sampler value: run parameters are upstream's, passed through (settings.py)
        from .. import settings
        self.assertFalse(hasattr(settings, "PRESETS"))

    def test_no_det_switch_in_the_package(self):
        for f in os.listdir(PKG):
            if f.endswith(".py"):
                text = open(os.path.join(PKG, f), encoding="utf-8").read()
                self.assertNotRegex(text, r'os\.environ\[["\']FOUNDRY_DET_SCATTER', f)
                self.assertNotRegex(text, r'(bash|subprocess|Popen|run)\([^\n]*install_det', f)     # the instrument is never invoked
        args = " ".join(a.option_strings[0] for p in cli.build_parser()._subparsers._group_actions[0].choices.values() for a in p._actions if a.option_strings)
        self.assertNotIn("--pack", args)
        self.assertNotIn("--variant", args)                                     # one variant: no axis to choose

    def test_kit_directory_holds_no_house_file(self):
        """The kit's MANIFEST.json is the only listing of what the kit carries; the tree keeps no sums file of its own beside it."""
        self.assertTrue(registry.check_tree()["ok"])
        self.assertFalse(os.path.exists(os.path.join(registry.kit_home(), "SHA256SUMS")))
        self.assertEqual(sorted(os.listdir(os.path.join(TREE, "opt", "forward"))), [os.path.basename(registry.kit_home())])


if __name__ == "__main__":
    unittest.main()
