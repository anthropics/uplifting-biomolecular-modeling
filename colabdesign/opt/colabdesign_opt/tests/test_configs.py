"""configs/<gpu>.env: one file per GPU class, the SAME statements — only the MODEL_OPT_TARGET_GPU default and the comments name the card. A twin that
drifts from configs/h100.env in any statement fails here."""
import glob
import os
import re
import unittest

from . import _stubs

CONFIGS = os.path.join(_stubs.TREE, "configs")
LABEL = re.compile(r"\$\{MODEL_OPT_TARGET_GPU:-[A-Za-z0-9_]+\}")


def statements(path: str) -> list:
    """The file's shell statements: comment lines dropped, trailing `  # ...` comments stripped, the card label default normalised."""
    out = []
    for line in open(path, encoding="utf-8").read().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        line = re.sub(r"\s{2,}# .*$", "", line).rstrip()
        out.append(LABEL.sub("${MODEL_OPT_TARGET_GPU:-<card>}", line))
    return out


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestConfigTwins(unittest.TestCase):
    def test_every_config_has_h100s_statements(self):
        ref = statements(os.path.join(CONFIGS, "h100.env"))
        self.assertTrue(any(s.startswith("export MODEL_OPT_TARGET_GPU=") for s in ref))
        others = sorted(p for p in glob.glob(os.path.join(CONFIGS, "*.env")) if os.path.basename(p) != "h100.env")
        self.assertIn(os.path.join(CONFIGS, "a100.env"), others)
        for p in others:
            self.assertEqual(statements(p), ref, os.path.basename(p))

    def test_labels(self):
        for name, card in (("h100.env", "H100"), ("a100.env", "A100")):
            text = open(os.path.join(CONFIGS, name), encoding="utf-8").read()
            self.assertIn(f"export MODEL_OPT_TARGET_GPU=${{MODEL_OPT_TARGET_GPU:-{card}}}", text, name)


if __name__ == "__main__":
    unittest.main()
