"""configs/<card>.env: the A100 file mirrors the H100 file line for line except the card it names; the target-GPU check is a note keyed on the
product name (stack.target_gpu_note), so neither card is refused."""
import os
import unittest

from . import _stubs   # noqa: F401  (sys.path for the package)
from mosaic_opt import stack

KIT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def _lines(card):
    with open(os.path.join(KIT, "configs", f"{card}.env"), encoding="utf-8") as fh:
        return fh.read().splitlines()


class TestCardConfigs(unittest.TestCase):
    DECLARED = (   # (h100.env text, a100.env text): the complete list of per-card substitutions
        ("# H100 deployment parameters (H100 80GB, compute capability 9.0; configs/a100.env is the A100 file).", "# A100 deployment parameters (A100 80GB or 40GB, compute capability 8.0; configs/h100.env is the H100 file)."),
        ("`run.sh design|check|warm --config h100 --mode <mode> ...`, or `source configs/h100.env`", "`run.sh design|check|warm --config a100 --mode <mode> ...`, or `source configs/a100.env`"),
        ("export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-H100}                                                            # the GPU this configuration targets (compute capability 9.0)",
         "export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-A100}                                                            # the GPU this configuration targets (compute capability 8.0)"),
        ("e.g. jax0.10.2-jaxlib0.10.2-cuda12plugin0.10.2-nvidia-h100-80gb-hbm3", "e.g. jax0.10.2-jaxlib0.10.2-cuda12plugin0.10.2-nvidia-a100-sxm4-80gb"),
    )

    def _read(self, card):
        return open(os.path.join(KIT, "configs", card + ".env"), encoding="utf-8").read()

    def test_a100_mirrors_h100_except_the_declared_card_lines(self):
        """configs/a100.env is configs/h100.env with exactly the declared per-card substitutions (the card name and compute capability in the
        header, the usage line's config name, MODEL_OPT_TARGET_GPU and its capability comment, the stack-key example) followed by the closing
        per-card-differences block — every other byte identical, the same variables exported in the same order."""
        h, a = self._read("h100"), self._read("a100")
        derived = h
        for old, new in self.DECLARED:
            self.assertEqual(derived.count(old), 1, old); derived = derived.replace(old, new)
        block_at = a.index("# Per-card differences from configs/h100.env")
        self.assertEqual(a[:block_at], derived.rstrip("\n") + "\n")                                   # byte-identical up to the closing block
        block = a[block_at:].splitlines()
        self.assertTrue(all(l.startswith("#") for l in block) and 3 <= len(block) <= 8, block)      # a comment block, nothing exported
        self.assertIn("NONE in the kit", a[block_at:])
        exports = lambda s: [l.split("=")[0].split("#")[0].strip() for l in s.splitlines() if l.startswith("export ")]
        self.assertEqual(exports(a), exports(h))

    def test_target_gpu_note_is_a_note_keyed_on_the_product_name(self):
        old = os.environ.get(stack.ENV_TARGET_GPU)
        try:
            os.environ[stack.ENV_TARGET_GPU] = "A100"
            for name in ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-40GB"):
                self.assertIsNone(stack.target_gpu_note({"name": name}))
            note = stack.target_gpu_note({"name": "NVIDIA H100 80GB HBM3"})
            self.assertIn("MODEL_OPT_TARGET_GPU=A100 but the GPU seen is NVIDIA H100 80GB HBM3", note)
            os.environ[stack.ENV_TARGET_GPU] = "H100"
            self.assertIsNone(stack.target_gpu_note({"name": "NVIDIA H100 80GB HBM3"}))
            self.assertIsNotNone(stack.target_gpu_note({"name": "NVIDIA A100-SXM4-80GB"}))
        finally:
            if old is None: os.environ.pop(stack.ENV_TARGET_GPU, None)
            else: os.environ[stack.ENV_TARGET_GPU] = old
