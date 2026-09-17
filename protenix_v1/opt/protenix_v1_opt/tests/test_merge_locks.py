"""Merge locks: one copy of each concern in package CODE. Each test names the one place and fails when a second copy appears (the mode
table, the deterministic recipe, the stock caller, the tree rule) or the documented pair drifts
(_autoload.MODES == modes.MODES: the .pth path imports nothing at start); run.sh and the config stay in step with the package (commands,
modes, no lever switch). No torch, no GPU."""
import glob
import os
import re
import unittest

from protenix_v1_opt import _autoload, cli, modes

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # opt/protenix_v1_opt
OPT = os.path.dirname(PKG)                                                    # opt/
TREE = os.path.dirname(OPT)                                                   # protenix_v1/


def _modules():
    return {os.path.basename(f): open(f, encoding="utf-8").read() for f in glob.glob(os.path.join(PKG, "*.py"))}


def _doc(name):
    return open(os.path.join(TREE, name), encoding="utf-8").read()


class TestOneTableEach(unittest.TestCase):
    def test_one_mode_table(self):
        """modes.KIT_MODES is the only place a shipped arm string is spelled in package code."""
        arms = {m.arm for m in modes.KIT_MODES.values() if m.arm != "stock"}
        for f, s in _modules().items():
            if f == "modes.py":
                continue
            for arm in arms:
                self.assertNotIn(f'"{arm}"', s, f"{f} spells the arm {arm!r}: modes.KIT_MODES is the one table")
            names = "KIT_MODES|DEFAULT_MODE" if f == "_autoload.py" else "KIT_MODES|MODES|DEFAULT_MODE"   # _autoload.MODES: the documented pair
            self.assertNotRegex(s, re.compile(rf"^({names})\s*[:=]", re.M), f"{f} carries a second mode table")
        self.assertEqual(tuple(modes.MODES), ("exact", "fast", "big", "off"))
        self.assertEqual(modes.DEFAULT_MODE, "fast")                 # the package default wherever a fast mode ships (KIT_MODES ships one); the amortized warm steady-state is the timing that counts
        self.assertIn("fast", modes.KIT_MODES)

    def test_documented_pair(self):
        self.assertEqual(tuple(_autoload.MODES), tuple(modes.MODES))
        from protenix_v1_opt import report
        from opt_core import report as core_report
        self.assertEqual(_autoload.EXIT_NOT_ACTIVE, report.EXIT_NOT_ACTIVE)          # the .pth path imports nothing at start: its constant is the documented twin
        from protenix_v1_opt import stack
        self.assertEqual(tuple(_autoload.DECLARED), tuple(stack.PACKAGE_ENV))      # the declared switches: the same pair
        from protenix_v1_opt import big
        self.assertEqual(set(_autoload.BIG_DECLARED), set(big.GATE_SETTINGS))   # the memory line's two size statements: the same pair (the .pth path cannot import big at start)
        self.assertEqual((report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE),
                         (core_report.EXIT_OK, core_report.EXIT_FAIL, core_report.EXIT_USAGE, core_report.EXIT_NOT_ACTIVE))   # the exit table is the core's

    def test_one_recipe_one_stock_caller_one_tree_rule(self):
        mods = _modules()
        self.assertEqual([f for f, s in mods.items() if "use_deterministic_algorithms(" in s and "torch." in s.split("use_deterministic_algorithms(")[0][-8:]], ["det.py"], "det.py is the one deterministic recipe")
        self.assertEqual(sorted(f for f, s in mods.items() if "protenix_cli.main(" in s), ["cli.py", "stock_pred.py"], "the stock CLI is entered in-process by pred (a kit mode) and by the proven stock subprocess only")
        shared_text = {"digest_memo.py"}                               # the weights digest memo: one helper text in every kit, byte-identical (its own sha256_file; kit.checkpoint_digest passes opt_core.gates.sha256_file as the hasher)
        one_tree_rule = {"kit.py"}                                     # tree_files/tree_sha's own implementation: the core dropped its copy (it is identified by the git commit too), kit.py is the one place left
        self.assertEqual([f for f, s in mods.items() if ("os.walk(" in s or "hashlib." in s) and f not in shared_text | one_tree_rule], [], "no OTHER package module walks or hashes a tree: kit.py is the one tree rule (tree_files / tree_sha)")


class TestRunShAndConfig(unittest.TestCase):
    def setUp(self):
        self.sh = _doc("run.sh")
        self.cfg = _doc(os.path.join("configs", "h100.env"))

    def test_commands_in_step(self):
        m = re.search(r'case "\$CMD" in install\|probe\) ;; ([a-z|]+)\) ;;', self.sh)
        self.assertEqual(set(m.group(1).split("|")), set(cli.COMMANDS))              # the package's verbs; `install` (pip core then kit, the pin check, --weights) and `probe` are the script's own steps
        self.assertIn('pip install -e "$HERE/../common/opt_core" -e "$HERE/opt"', self.sh)

    def test_modes_in_step(self):
        self.assertIn('exec python -m protenix_v1_opt "$CMD"', self.sh)
        self.assertIn('[ "$MODE" = off ]', self.sh)
        self.assertNotRegex(self.sh, r"exact\+tg", "run.sh spells no arm: the package resolves modes")

    def test_config_carries_no_lever_switch(self):
        exported = re.findall(r"^export ([A-Z0-9_]+)=", self.cfg, re.M)
        self.assertEqual(sorted(exported), ["LAYERNORM_TYPE", "MODEL_OPT", "MODEL_OPT_TARGET_GPU", "PROTENIX_ROOT_DIR", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"])
        for name in ("PROTENIX_V1_OPT", "PROTENIX_V1_OPT_DET", "PROTENIX_DET_SCATTER", "CUBLAS_WORKSPACE_CONFIG", "PTX_", "FPF_"):
            self.assertNotRegex(self.cfg, rf"export {re.escape(name)}", name)
        self.assertIn('bash "$MODEL_OPT/run.sh" probe ||', self.cfg)                    # the package probe is run.sh's one definition
        self.assertIn("probe_package () {", self.sh)

    def test_configs_are_one_per_card_and_mirror_each_other(self):
        """configs/<card>.env, one per admitted card class; every card file exports exactly the variables h100.env exports (only the
        MODEL_OPT_TARGET_GPU default and comments differ), so a lever never depends on which card file was sourced."""
        names = sorted(os.listdir(os.path.join(TREE, "configs")))
        self.assertEqual(names, ["a100.env", "h100.env", "h200.env"])
        def exports(fn):
            return [re.match(r"export (\w+)=", l).group(1) for l in _doc(os.path.join("configs", fn)).splitlines() if l.startswith("export ")]
        self.assertEqual(exports("a100.env"), exports("h100.env"))
        self.assertIn("MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-A100}", _doc(os.path.join("configs", "a100.env")))


if __name__ == "__main__":
    unittest.main()
