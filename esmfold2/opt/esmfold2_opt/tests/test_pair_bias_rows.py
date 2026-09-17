"""The carried driver's pair-bias launch bound (driver/ef2_opt.py `pair_bias_launch_rows`, `fused_pair_bias_rows`, `install_pair_bias_rows`):
the upstream fused pair-bias kernel forms 32-bit element offsets, so a pair plane past 2**31-1 offsets (c_z = 256, one sample: 2897 tokens
and more) is launched in row blocks that each stay inside the bound, the blocks' outputs assembled into one [B, H, Q, K] bias equal to the
single launch's; a plane inside the bound is ONE call, the caller's arguments untouched. The assertions run in a fresh interpreter with a
recording stand-in for the kernel: these tests' stub rig swaps `torch` in sys.modules, and a real torch must not be imported into the pytest
process before it (a C extension does not survive re-import)."""
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(os.path.dirname(os.path.dirname(HERE)), "forward", "fast_inference", "driver")

PROBE = r"""
import importlib.util, os, sys, types
driver = sys.argv[1]
try:
    import torch
except Exception as e:                                   # noqa: BLE001
    print("NO_TORCH", type(e).__name__); sys.exit(80)
sys.path.insert(0, driver)
spec = importlib.util.spec_from_file_location("ef2_opt_under_test", os.path.join(driver, "ef2_opt.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
INT32_MAX = 2**31 - 1
checks = {}
rows = mod.pair_bias_launch_rows
# --- the bound: c_z 256, 16 heads, one sample: 2896 tokens is one launch, 2897 is not; a 3036-token plane launches 2763-row blocks
checks["bound_is_int32_max"] = mod.PAIR_BIAS_OFFSET_MAX == INT32_MAX
checks["2896_one_launch"] = rows(1, 2896, 2896, 256, 16) == 2896
checks["2897_blocked"] = 0 < rows(1, 2897, 2897, 256, 16) < 2897
checks["3036_blocks_of_2763"] = rows(1, 3036, 3036, 256, 16) == 2763
def offsets_fit(B, r, K, D, H):                              # the largest offset of each pointer family one launch over [B, r, K, D] forms
    return max(B * r * K * D - 1, B * H * r * K - 1, B * r * K - 1, B * K - 1) <= INT32_MAX
for L in (2897, 3036, 4000, 6000, 10000):
    r = rows(1, L, L, 256, 16)
    checks[f"L{L}_block_fits"] = 0 < r < L and offsets_fit(1, r, L, 256, 16)
    checks[f"L{L}_block_is_maximal"] = not offsets_fit(1, r + 1, L, 256, 16)
r2 = rows(2, 2100, 2100, 256, 16)                            # two samples share the plane: 2*2100*2100*256 passes the bound, one sample's does not
checks["batch_blocked_and_fits"] = 0 < r2 < 2100 and offsets_fit(2, r2, 2100, 256, 16) and not offsets_fit(2, r2 + 1, 2100, 256, 16)
rh = rows(1, 6000, 6000, 8, 64)                              # the output's H*Q*K offsets bind when H > DIM_Z
checks["heads_over_dim_z_bind"] = 0 < rh < 6000 and offsets_fit(1, rh, 6000, 8, 64) and not offsets_fit(1, rh + 1, 6000, 8, 64)
checks["row_block_q_ne_k_one_launch"] = rows(1, 100, 3036, 256, 16) == 100          # a caller's own row block [B, R, L, C] inside the bound
checks["no_row_fits_is_zero"] = rows(1, 5, 9_000_000, 256, 16) == 0

# --- the launcher with a recording stand-in for upstream's fused_pair_bias(z, mask, w, ln_w, ln_b, *, num_heads, eps, inf) -> [B, H, Q, K]
CALLS = []
def fake(z, mask, w_proj_z, pair_norm_w=None, pair_norm_b=None, *, num_heads, eps=1e-5, inf=1e6):
    CALLS.append((tuple(z.shape), z.data_ptr(), z.is_contiguous(), None if mask is None else mask.clone(), eps, inf, pair_norm_w, pair_norm_b))
    B, Q, K, D = z.shape
    ln = torch.nn.functional.layer_norm(z.float(), (D,), None, None, eps)
    bias = (ln[:, None, :, :, :] * w_proj_z.float()[None, :, None, None, :]).sum(-1)      # [B, H, Q, K]: per-(q, k) arithmetic, whatever Q is
    if mask is not None:
        bias = bias + torch.where(mask[:, None, None, :], 0.0, -inf)
    return bias.to(z.dtype)
g = torch.Generator().manual_seed(3)
B, Q, K, D, H = 2, 10, 7, 4, 2
z = torch.randn((B, Q, K, D), generator=g); w = torch.randn((H, D), generator=g)
mask = torch.ones((B, K), dtype=torch.bool); mask[0, -2:] = False; mask[1, :1] = False     # the samples' key masks differ
single = mod.fused_pair_bias_rows(fake, z, mask, w, num_heads=H)                    # inside the bound: ONE call, the caller's z object itself
checks["fits_one_call"] = len(CALLS) == 1 and CALLS[0][0] == (B, Q, K, D) and CALLS[0][1] == z.data_ptr()
checks["fits_no_stat"] = mod.STATS["pair_bias_row_launches"] == 0
CALLS.clear()
bound = 3 * K * max(D, H)                                                            # a bound that admits 3 rows of one sample per launch
checks["test_bound_blocks_the_plane"] = B * Q * K * max(D, H) > bound and mod.pair_bias_launch_rows(1, Q, K, D, H, bound) == 3
blocked = mod.fused_pair_bias_rows(fake, z, mask, w, "LNW", "LNB", num_heads=H, bound=bound, eps=1e-3)
checks["blocked_per_sample_row_extents"] = [c[0] for c in CALLS] == [(1, r, K, D) for _b in range(B) for r in (3, 3, 3, 1)]
checks["blocked_blocks_are_contiguous_views_of_z"] = all(c[2] for c in CALLS) and [c[1] for c in CALLS] == [z[b:b + 1, s:s + 3].data_ptr() for b in range(B) for s in (0, 3, 6, 9)]
checks["blocked_each_sample_its_own_mask"] = all(torch.equal(c[3], mask[i // 4:i // 4 + 1]) for i, c in enumerate(CALLS))
checks["blocked_passes_arguments"] = all(c[4] == 1e-3 and c[5] == 1e6 and c[6] == "LNW" and c[7] == "LNB" for c in CALLS)
ref = fake(z, mask, w, num_heads=H, eps=1e-3)
checks["blocked_equals_single_launch"] = blocked.shape == (B, H, Q, K) and blocked.dtype == ref.dtype and torch.equal(blocked, ref)
checks["blocked_stat_counts_launches"] = mod.STATS["pair_bias_row_launches"] == 8
CALLS.clear()
nomask = mod.fused_pair_bias_rows(fake, z, None, w, num_heads=H, bound=bound)
checks["blocked_without_mask"] = all(c[3] is None for c in CALLS) and torch.equal(nomask, fake(z, None, w, num_heads=H))
try:
    mod.fused_pair_bias_rows(fake, z, mask, w, num_heads=H, bound=K * max(D, H) - 1)
    checks["no_row_fits_refuses_by_name"] = False
except RuntimeError as e:
    checks["no_row_fits_refuses_by_name"] = "32-bit element offsets" in str(e)

# --- install: the upstream module's _fused_pair_bias is routed through the launcher, idempotently; absent kernel -> False
ns = types.SimpleNamespace(_fused_pair_bias=fake)
mod._common = lambda: ns
checks["install_true"] = mod.install_pair_bias_rows() is True
routed = ns._fused_pair_bias
checks["install_routes"] = routed is not fake and getattr(routed, "_ef2opt_rows", False) and routed._ef2opt_inner is fake
checks["install_idempotent"] = mod.install_pair_bias_rows() is True and ns._fused_pair_bias is routed
CALLS.clear()
out = routed(z, mask, w, num_heads=H, pair_norm_w=None, pair_norm_b=None)         # the kit callers' keyword form
checks["routed_fits_one_call_same_bytes"] = len(CALLS) == 1 and torch.equal(out, fake(z, mask, w, num_heads=H))
ns2 = types.SimpleNamespace(_fused_pair_bias=None)
mod._common = lambda: ns2
checks["install_absent_kernel_false"] = mod.install_pair_bias_rows() is False and ns2._fused_pair_bias is None
bad = [k for k, v in checks.items() if not v]
print("BAD", bad)
sys.exit(1 if bad else 0)
"""


class PairBiasRows(unittest.TestCase):
    def test_planes_past_the_int32_offset_bound_launch_in_row_blocks_equal_to_one_launch(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-c", PROBE, DRIVER], env=env, capture_output=True, text=True)
        if r.returncode == 80:
            self.skipTest("needs torch (CPU is enough): " + r.stdout.strip())
        self.assertEqual(r.returncode, 0, (r.stdout[-600:], r.stderr[-900:]))
        self.assertIn("BAD []", r.stdout)


if __name__ == "__main__":
    unittest.main()
