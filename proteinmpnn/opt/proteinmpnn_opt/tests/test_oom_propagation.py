"""Out-of-memory propagates on the served paths; no fallback is applied (a served broad handler re-raises OOM first).

The package imports no torch and applies no lever in-process: ``design`` launches the kit executable (``kit/mpnn_worker2.py``) or
the stock command line as a subprocess (the package's own interpreter, where the pinned ``opt_core`` is installed) and maps its exit code, so a
worker that runs out of memory exits non-zero and ``design`` exits 1 with the worker's traceback on stderr — nothing reruns the stock
route, no lever is dropped by an exception handler (``test_cli_manifest.TestKitRoute.test_a_worker_out_of_memory_is_exit_1_and_nothing_reruns``).
This module locks the census behind that statement: the served modules (``opt/proteinmpnn_opt/*.py``, ``stock/check_pins.py``) carry no broad
exception handler; the carried tree's broad handlers (``opt/forward``) are exactly the named one, which wraps no lever, no kernel launch and
no allocating call. No torch, no GPU.
"""
import glob
import os
import re
import unittest

from opt_core.oom import is_oom

from proteinmpnn_opt import stack

TREE = stack.tree_home()
PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                    # opt/proteinmpnn_opt
BROAD = re.compile(r"^\s*except\s*(:|(Exception|BaseException)\b|\(.*\b(RuntimeError|Exception|BaseException)\b.*\))")

# The carried tree's broad handlers, by file (the file a mode launches): how many, and what the handler does.
CARRIED_BROAD = {
    "mpnn_exact_worker/addon/mpnn_worker2.py": 2,                     # launched: `git rev-parse HEAD` for the .fa header -> commit "unknown"; the fused-draw
                                                                     # start-up probe: a kernel that does not compile / launch is named PROBE FAIL (the job is
                                                                     # then refused by name) — it re-raises an out-of-memory first
}


def broad_handlers(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")
    return [(i, lines) for i, line in enumerate(lines) if BROAD.match(line)]


class TestOOMPropagation(unittest.TestCase):
    def test_is_oom_is_the_cores(self):
        class OutOfMemoryError(RuntimeError):                     # torch's class name (torch.OutOfMemoryError); no torch on this box
            pass
        self.assertTrue(is_oom(OutOfMemoryError("CUDA out of memory (mock)"))); self.assertTrue(is_oom(MemoryError()))
        self.assertFalse(is_oom(ValueError("shape mismatch")))
        self.assertEqual(is_oom.__module__, "opt_core.oom")

    def test_served_modules_carry_no_broad_handler(self):
        served = sorted(glob.glob(os.path.join(PACKAGE, "*.py"))) + [os.path.join(TREE, "stock", "check_pins.py")]
        self.assertGreaterEqual(len(served), 15, served)
        hits = [f"{os.path.relpath(p, TREE)}:{i + 1}: {lines[i].strip()}" for p in served for i, lines in broad_handlers(p)]
        self.assertEqual(hits, [], "a broad handler on a served path re-raises OOM first: `if is_oom(e): raise` (opt_core.oom)")

    def test_carried_broad_handlers_are_the_named_ones(self):
        kit_home = stack.kit_home()
        found = {}
        for p in glob.glob(os.path.join(kit_home, "**", "*.py"), recursive=True):
            n = len(broad_handlers(p))
            if n:
                found[os.path.relpath(p, kit_home)] = n
        self.assertEqual(found, CARRIED_BROAD)
        # the one a mode launches wraps no lever and no kernel launch
        (i, lines), (j, _) = broad_handlers(os.path.join(kit_home, "mpnn_exact_worker", "addon", "mpnn_worker2.py"))
        self.assertIn("git --git-dir", lines[i - 1]); self.assertEqual(lines[i + 1].strip(), 'self.commit = "unknown"')
        # the probe's handler: inside decide_fused_draw, OOM re-raised first, then the failure is a named verdict (no fallback: main() refuses the job)
        self.assertTrue(lines[j + 1].strip().startswith('if isinstance(e, MemoryError) or "out of memory" in str(e).lower(): raise'), lines[j + 1])
        self.assertIn('"PROBE FAIL (', lines[j + 2])
        k = max(n for n in range(j) if lines[n].lstrip().startswith("def "))
        self.assertIn("def decide_fused_draw(", lines[k])


if __name__ == "__main__":
    unittest.main()
