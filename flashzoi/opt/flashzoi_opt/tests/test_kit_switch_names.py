"""The kit-switch refusal list is the kit's OWN switch names, read from its bytes — locked here against the os.environ reads of the kit's
closure modules (the kit dir kits/v1_25/*.py without its tests; engines/flashzoi/{predict_tracks_fast,result_pool}.py). Every name read
in the closure must be in stack.KIT_ENV_SWITCHES and every listed name must be read somewhere in the closure (a prefix or a stale name is
not a switch); the cache / data / host names (DATA_AND_HOST) are not switches — a new read anywhere trips this test."""
import ast
import os
import re
import unittest

from flashzoi_opt import stack

READ_RE = re.compile(r"""os\.environ(?:\.get)?\s*[\(\[]\s*["']([A-Z][A-Z0-9_]*)["']|os\.getenv\s*\(\s*["']([A-Z][A-Z0-9_]*)["']""")
CLOSURE = ("engines/flashzoi/kits/v1_25/__init__.py", "engines/flashzoi/kits/v1_25/_wrap.py", "engines/flashzoi/kits/v1_25/fz_exact.py",
           "engines/flashzoi/kits/v1_25/_pins.py", "engines/flashzoi/kits/v1_25/_oom.py", "engines/flashzoi/kits/v1_25/_class_records.py",
           "engines/flashzoi/predict_tracks_fast.py", "engines/flashzoi/result_pool.py")
DATA_AND_HOST = {"TRITON_CACHE_DIR", "CUBLAS_WORKSPACE_CONFIG"}   # cache / library names the kit may read that are not switches of the forward


def _reads(path):
    """{name: [(enclosing function or class name, or '<module>')]} for every environ read in the file."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    scopes = [(n.name, n.lineno, n.end_lineno, isinstance(n, ast.ClassDef)) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    out = {}
    for i, line in enumerate(src.splitlines(), 1):
        for m in READ_RE.finditer(line):
            name = m.group(1) or m.group(2)
            inner = sorted((s for s in scopes if s[1] <= i <= s[2]), key=lambda s: s[2] - s[1])
            out.setdefault(name, []).append((inner[0][0] if inner else "<module>", [s[0] for s in inner if s[3]]))
    return out


class KitSwitchNames(unittest.TestCase):
    def test_list_equals_the_wired_reads(self):
        root = stack.kit_root()
        listed = set(stack.KIT_ENV_SWITCHES)
        all_reads = set()
        for rel in CLOSURE:
            p = os.path.join(root, rel)
            self.assertTrue(os.path.isfile(p), f"closure module missing: {rel}")
            all_reads.update(_reads(p))
        wired = all_reads - DATA_AND_HOST
        self.assertTrue(all_reads & DATA_AND_HOST, "no environ reads found under the kit root — the regex or the root is wrong")   # the host reads (TRITON_CACHE_DIR) prove the scan works; the kit wires no switch of its own
        self.assertEqual(sorted(wired - listed), [], f"switch names read in the closure but not refused: {sorted(wired - listed)}")
        self.assertEqual(sorted(listed - all_reads), [], f"names in KIT_ENV_SWITCHES that no closure module reads: {sorted(listed - all_reads)}")
        for n in listed:
            self.assertFalse(n.endswith("_"), f"{n}: a prefix is not a switch name")

    def test_one_env_list(self):
        """stack.STRIPPED_ENV / FORBIDDEN_ENV are the package's one env list; the stock caller (opt/flashzoi_opt/stock_pred.py, which imports nothing of the
        package) carries its FORBIDDEN_ENV as a literal — read from its bytes here and locked to STRIPPED_ENV (the package's switch + the
        kit's); the TF32 overrides are the package gate's (refused before the stock subprocess is launched)."""
        from flashzoi_opt import modes
        # NOTE: stack.STRIPPED_ENV/FORBIDDEN_ENV are plain module-level tuple concatenations (PACKAGE_ENV + KIT_ENV_SWITCHES,
        # KIT_ENV_SWITCHES + TF32_OVERRIDE_ENV), assigned exactly once, never reassigned anywhere in the package -- restating
        # the same formula here would be an A==A tautology, not a check. What's real: the cross-file mirror against
        # stock_pred.py's independent literal (below) and the TF32 names actually matching the libraries' real switches.
        self.assertEqual(stack.TF32_OVERRIDE_ENV, ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"))
        pred = os.path.join(stack.tree_home(), "opt", "flashzoi_opt", "stock_pred.py")
        lit = modes._literal_assignments(pred, ("FORBIDDEN_ENV",))
        self.assertEqual(tuple(lit.get("FORBIDDEN_ENV") or ()), stack.STRIPPED_ENV, "stock_pred.py FORBIDDEN_ENV must mirror stack.STRIPPED_ENV")
        self.assertEqual(stack.env_hits(stack.TF32_OVERRIDE_ENV, {"NVIDIA_TF32_OVERRIDE": "0", "HOME": "/"}), ["NVIDIA_TF32_OVERRIDE"])
        self.assertEqual(stack.env_hits(stack.FORBIDDEN_ENV, {"FZ_KIT_JITX": "1"}), [])                                  # exact names, never a prefix


if __name__ == "__main__":
    unittest.main()
