"""The carried driver's graph-budget token reader (driver/ef2_opt.py `_pair_tokens`, `plane_tokens`, `_over_graph_budget`): a square pair argument
decides as always; a call whose pair argument is a ROW BLOCK ([B, R, L, C], R != L — what a row-sharded caller passes) is still budget-gated on L,
never captured unbudgeted; a BATCHED pair [B, L, L, C] is budgeted by its plane (ceil(L * sqrt(B)) tokens) and, over the budget, returns the
generation's graph pools before it runs eagerly UNLESS the device has room for the region beside the generation (ef2_opt v4.3
_batched_region_keeps_generation: kept when free memory >= factor x the batched pair's bytes + margin; EF2_GRAPH_BATCHED_RELEASE=1 or an
unknown pair size = v4.2's unconditional release).
The assertions run in a fresh interpreter: these tests' stub rig swaps `torch` in sys.modules, and a real torch must not be imported into
the pytest process before it (a C extension does not survive re-import)."""
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(os.path.dirname(os.path.dirname(HERE)), "forward", "fast_inference", "driver")

PROBE = r"""
import importlib.util, os, sys
driver = sys.argv[1]
try:
    import torch
except Exception as e:                                   # noqa: BLE001
    print("NO_TORCH", type(e).__name__); sys.exit(80)
sys.path.insert(0, driver)
spec = importlib.util.spec_from_file_location("ef2_opt_under_test", os.path.join(driver, "ef2_opt.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
z = torch.zeros(1, 48, 48, 8); m = torch.zeros(1, 5, 48, 8); s = torch.zeros(1, 48, 16); rows = torch.zeros(1, 12, 48, 8)
checks = {
    "square_first": mod._pair_tokens((s, z), {}) == 48,
    "square_in_kwargs_wins": mod._pair_tokens((m,), {"pair": z}) == 48,
    "square_after_nonsquare_wins": mod._pair_tokens((m, z), {}) == 48,
    "no_4d_no_decision": mod._pair_tokens((s,), {"x": 3}) is None,
    "row_block_gated_on_L": mod._pair_tokens((rows,), {}) == 48,
    "either_middle_extent": mod._pair_tokens((), {"z": torch.zeros(1, 48, 12, 8)}) == 48,
    "largest_middle_extent_fail_safe": mod._pair_tokens((rows, torch.zeros(1, 64, 48, 8)), {}) == 64,
}
old = mod.CFG.graph_budget_tokens
mod.CFG.graph_budget_tokens = 1                              # the memory line's setting: every shape above 1 token runs eager
checks["budget_1_refuses_row_block"] = bool(mod._over_graph_budget("generic_lm_encoder", mod._pair_tokens((rows,), {})))
mod.CFG.graph_budget_tokens = 0                              # no budget: no decision either way
checks["budget_0_decides_nothing"] = not mod._over_graph_budget("generic_lm_encoder", mod._pair_tokens((rows,), {}))
# --- a batched pair is budgeted by its plane: [B, L, L, C] counts as ceil(L * sqrt(B)) tokens; over the budget with B > 1 the generation's
#     pools are returned (clear_graphs) before the region runs eagerly, and the budget line names batch / pair_tokens / pools
import io, math, re
from contextlib import redirect_stdout
checks["plane_b1_is_L"] = mod.plane_tokens(1147, 1) == 1147 and mod.plane_tokens(1147, 0) == 1147
checks["plane_b5"] = mod.plane_tokens(1147, 5) == math.ceil(1147 * math.sqrt(5)) == 2565
checks["plane_edge"] = mod.plane_tokens(581, 5) == 1300 and mod.plane_tokens(582, 5) == 1302
mod.CFG.graph_budget_tokens = 1300
cleared = []
real_clear = mod.clear_graphs
mod.clear_graphs = lambda model=None, reason="shape": cleared.append(reason)
mod._BUDGET_SEEN.clear()
out = io.StringIO()
with redirect_stdout(out):
    checks["b1_1147_captures"] = not mod._over_graph_budget("trunk_t", 1147)
    checks["b5_581_captures"] = not mod._over_graph_budget("trunk_t", 581, batch=5)
    mod._POOL["handle"] = None
    checks["b5_1147_eager_no_pool_no_clear"] = mod._over_graph_budget("trunk_c", 1147, batch=5) and cleared == []
    mod._POOL["handle"] = object()                              # a generation has captured
    mod._BUDGET_SEEN.clear()
    checks["b5_1147_eager_clears"] = mod._over_graph_budget("trunk_c", 1147, batch=5) and cleared == ["budget_trunk_c"]
    checks["b1_over_budget_never_clears"] = mod._over_graph_budget("trunk_t", 1400) and cleared == ["budget_trunk_c"]
    # v4.3: with the batched pair's size known and room on the device the generation is KEPT (no clear); without room, or under
    # EF2_GRAPH_BATCHED_RELEASE=1, it is released first as before. torch.cuda's memory readers are patched: the probe runs on the CPU.
    mod.torch.cuda.mem_get_info = lambda: (60 << 30, 80 << 30); mod.torch.cuda.memory_reserved = lambda: 8 << 30; mod.torch.cuda.memory_allocated = lambda: 6 << 30
    checks["b5_keep_when_room"] = mod._over_graph_budget("trunk_c", 1147, batch=5, plane_bytes=3 << 30) and cleared == ["budget_trunk_c"] and mod.STATS["graph_budget_batched_kept_trunk_c"] == 1
    checks["b5_release_when_tight"] = mod._over_graph_budget("trunk_c", 1147, batch=5, plane_bytes=20 << 30) and cleared == ["budget_trunk_c", "budget_trunk_c"]   # need 4 x 20 + 2 GiB > 62 GiB available
    mod.CFG.batched_release = True
    checks["b5_release_word_forces"] = mod._over_graph_budget("trunk_c", 1147, batch=5, plane_bytes=3 << 30) and cleared == ["budget_trunk_c"] * 3
    mod.CFG.batched_release = False
mod._POOL["handle"] = None
lines = [l for l in out.getvalue().splitlines() if l.startswith("[ef2_opt] graph budget:")]
b5 = [l for l in lines if " batch=5 " in l]
checks["b5_line_shape"] = len(b5) >= 3 and all(re.search(r"^\[ef2_opt\] graph budget: \S+ tokens=2565 > EF2_GRAPH_BUDGET_TOKENS=1300 -> eager ", l) for l in b5) \
    and any(l.endswith(" batch=5 pair_tokens=1147 pools=released(plane_bytes=unknown)") for l in b5) and any(l.endswith(" batch=5 pair_tokens=1147") for l in b5) \
    and any(re.search(r" batch=5 pair_tokens=1147 generation=kept\(avail=62\.0GiB,need=14\.0GiB,batch=5\)$", l) for l in b5) \
    and any(l.endswith(" pools=released(EF2_GRAPH_BATCHED_RELEASE=1)") for l in b5) and any(l.endswith(" pools=released(avail=62.0GiB,need=82.0GiB,batch=5)") for l in b5)   # one line per (site, tokens, decision kind)
b1 = [l for l in lines if "tokens=1400 " in l]
checks["b1_line_unchanged"] = len(b1) == 1 and " batch=" not in b1[0] and " pools=" not in b1[0]
checks["stats_count_sites"] = mod.STATS["graph_budget_eager_trunk_c"] == 5 and mod.STATS["graph_budget_eager_trunk_t"] == 1 and mod.STATS["graph_clears_budget_trunk_c"] == 0   # (clear_graphs was stubbed: the real counter never moved)
mod.clear_graphs = real_clear
mod.CFG.graph_budget_tokens = old
bad = [k for k, v in checks.items() if not v]
print("BAD", bad)
sys.exit(1 if bad else 0)
"""


class PairTokens(unittest.TestCase):
    def test_square_pair_decides_as_always_and_a_row_block_is_gated_on_L(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-c", PROBE, DRIVER], env=env, capture_output=True, text=True)
        if r.returncode == 80:
            self.skipTest("needs torch (CPU is enough): " + r.stdout.strip())
        self.assertEqual(r.returncode, 0, (r.stdout[-600:], r.stderr[-900:]))
        self.assertIn("BAD []", r.stdout)


if __name__ == "__main__":
    unittest.main()
