"""The MSA kernels (driver/ef2_msa.py: t11 SiLU·mul rows, t12 outer-product-mean epilogue, t13 pair-weighted averaging, t14 LayerNorm rows)
address their operands with 64-bit row offsets.

Why this is locked: Triton types `tl.program_id`, `tl.arange` and int arguments below 2**31 as int32, so a row offset formed as a product of
two of them wraps past 2**31 elements — in the t12 epilogue the ws operand is [(L*32), (L*32)] and the offset (i*32 + c)*stride reaches
(L*32)**2, which passes 2**31 at L = 1449 tokens (an illegal address, or another row's bytes read silently); the row kernels' R*K products
(R = L x MSA depth) pass it at large L x depth. The promotion to int64 is address arithmetic only: every loaded value, every reduction order
and every store is the one the kernel computed before, so outputs below the extent are byte-identical (the kit's GPU record: big full_msa
--det 1 at 1038 tokens, cif + npz sha-identical across the port) and inputs past it address correctly.
"""
import os, re, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
MSA = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference", "driver", "ef2_msa.py"))


def _kernel(src: str, name: str) -> str:
    m = re.search(r"@triton\.jit\ndef %s\(.*?(?=\n(?:@triton\.jit|def |_[A-Z0-9_]+ = |# =====))" % name, src, re.S)
    assert m, name
    return m.group(0)


class MsaKernelOffsets(unittest.TestCase):
    def setUp(self):
        self.src = open(MSA).read()

    def test_every_row_index_is_promoted_to_int64_before_pointer_arithmetic(self):
        want = {"_ln_rows_kernel": "rows = rows.to(tl.int64)", "_silu_mul_kernel": "r = r.to(tl.int64)",
                "_opm_proj_kernel": "i = tl.program_id(0).to(tl.int64)", "_pwa_kernel": "offs_i = offs_i.to(tl.int64)"}
        for name, stmt in want.items():
            body = _kernel(self.src, name)
            self.assertIn(stmt, body, name)
            first_ptr = min(k for k in (body.find("tl.load("), body.find("tl.store(")) if k >= 0)
            self.assertLess(body.find(stmt), first_ptr, f"{name}: the promotion must precede the first pointer expression")
        self.assertIn("offs_j = offs_j.to(tl.int64)", _kernel(self.src, "_pwa_kernel"))

    def test_the_extents_the_promotion_covers(self):
        C = 32                                                             # the OPM hidden width (Wout: [256, 32*32])
        first_L = next(L for L in range(1, 4096) if (L * C) ** 2 >= 2 ** 31)
        self.assertEqual(first_L, 1449)                                   # t12 epilogue: (L*32)^2 elements
        first_L_out = next(L for L in range(1, 8192) if L * L * 256 >= 2 ** 31)
        self.assertEqual(first_L_out, 2897)                               # t12 out store: L*L*256 elements
        depth = 1024                                                       # msa_max_depth pinned
        first_L_silu = next(L for L in range(1, 8192) if L * depth * 2 * 512 >= 2 ** 31)
        self.assertEqual(first_L_silu, 2048)                              # t11 SiLU·mul rows: (L*M) x 2*512 elements


if __name__ == "__main__":
    unittest.main()
