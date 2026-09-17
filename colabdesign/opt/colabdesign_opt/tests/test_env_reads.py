"""stock/PINS.json stock_environment vs the code: every environment variable read by the shared kernel the `pallas` lever executes (common/opt_core/opt_core/kernels/pallas_attn) is listed with its location, and every one that changes a design's
numerics has a must-be-absent prefix; the tolerated ones are named; no arm carries an exception."""
import os
import re
import unittest

from . import _stubs

RE_PY = re.compile(r"""os\.environ(?:\.get\(|\[)\s*["']([A-Z][A-Z0-9_]*)["']""")
RE_SH = re.compile(r"\$\{([A-Z][A-Z0-9_]*)[:-]")
CORE_KERNEL_DIR = os.path.join(_stubs.TREE, os.pardir, "common", "opt_core", "opt_core", "kernels", "pallas_attn")


def listed_names(se):
    out = set()
    for key in se["reads"]:
        out.update(n.strip() for n in key.split(","))
    return out


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestEnvReads(unittest.TestCase):
    def reads(self, roots):
        names = {}
        for root in roots:
            for r, _, fs in os.walk(root):
                for f in fs:
                    p = os.path.join(r, f)
                    if f.endswith(".py"):
                        for n in RE_PY.findall(open(p, encoding="utf-8").read()):
                            names.setdefault(n, set()).add(p)
                    elif f.endswith(".sh"):
                        for n in RE_SH.findall(open(p, encoding="utf-8").read()):
                            names.setdefault(n, set()).add(p)
        for local in ("HERE", "PWD"):
            names.pop(local, None)                                     # shell-local names of the self-test
        return names

    def test_every_read_is_listed(self):
        se = _stubs.pins()["stock_environment"]
        listed = listed_names(se)
        if os.path.isdir(CORE_KERNEL_DIR):                                     # the shared kernel's backward knobs (the tree beside the kit)
            kern = set(self.reads([CORE_KERNEL_DIR]))
            self.assertTrue(kern, "the shared kernel reads no variable — PINS.json lists its reads")
            self.assertEqual(kern - listed, set(), f"kernel reads not in PINS.json: {kern - listed}")
            for n in kern:
                self.assertTrue(n.startswith("AF_PALLAS_"), n)

    def test_lever_variables_have_a_must_be_absent_prefix(self):
        se = _stubs.pins()["stock_environment"]
        prefixes = tuple(se["must_be_absent_prefixes"])
        stock_only = tuple(se["stock_arm_absent_prefixes"])                    # jax's persistent-cache variables: lever compilecache sets them INTO the kit arm; absent from stock's
        tolerated = " ".join(se["tolerated"])
        for name in listed_names(se):
            if name.startswith(prefixes) or name.startswith(stock_only):
                continue
            self.assertIn(name, tolerated, f"{name} is listed, has no must-be-absent prefix and is not in the tolerated list")
        self.assertEqual(stock_only, ("JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_"))
        from colabdesign_opt import stack
        self.assertEqual(stack.stock_env_absent(_stubs.pins(), "stock"), list(prefixes) + list(stock_only)); self.assertEqual(stack.stock_env_absent(_stubs.pins(), "kit"), list(prefixes))
        self.assertIn("AF2M_", prefixes); self.assertIn("AF_PALLAS_", prefixes); self.assertIn("COLABDESIGN_OPT", prefixes)
        self.assertNotIn("fast_arm_exception", se)                            # no arm carries a kit or lever variable: a mode is its lever set


if __name__ == "__main__":
    unittest.main()
