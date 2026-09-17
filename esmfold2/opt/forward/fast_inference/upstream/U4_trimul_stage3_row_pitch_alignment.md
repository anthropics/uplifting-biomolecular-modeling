# PR: ESMFold2 fused TriMul — align the stage-3 einsum operands so cuBLAS uses its sm90 kernel (inference, 1.2-1.3x on the TriMul block at L % 8 != 0)

## Problem
`triangle_multiplicative_update_with_residual` (kernels/trimul_with_residual.py) runs its stage-3 triangle contraction in inference as
`torch.einsum("dbik,dbjk->dbij", a, b_t)` on bf16 operands of shape `(D, B, L, L)` where `L` is the token count. On H100 (torch 2.13,
CUDA 13) cuBLAS dispatches this batched GEMM on the 16-byte alignment of the row pitch (`L * 2` bytes). Profiled kernels for the
64-batch `L x L x L` contraction:

| L mod 8 | kernel | relative time |
|---|---|---|
| 0 | `nvjet_sm90_*` | 1.0 |
| 2/4/6 | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align2` | ~2.3 |
| odd | `cutlass_75_tensorop_bf16_s1688gemm_bf16_16x16_tn_align1` | ~3.2 |

Real complexes rarely have `L % 8 == 0`; at 705 tokens stage 3 is 1.07 s of a 5.07 s ESMFold2-Fast fold (10 loops, fused backend).

## Change
Inference-only (`torch.is_grad_enabled() == False`), guarded by `ESMFOLD2_TRIMUL_S3_ALIGN` (default 8; set 0 to disable): when
`L % 8 != 0`, copy `a` and `b_t` into zero-initialised buffers whose row pitch is the next multiple of 8, run `torch.matmul` on them and
slice the valid `L x L` block. `L % 8 == 0` inputs take the existing path unchanged. (The kit's `driver/ef2_w4.py` implements the
same idea without the two copy kernels by letting the stage-2 dual-GEMM kernel store at padded addresses and the stage-4 LN kernel read
them; this PR is the minimal, easily reviewable version: the copies cost ~8% of the gain.)

## Numerics
Multiplications by the zero padding and additions of +0.0 are exact, so the padded contraction computes the same sum; the only
difference is the fp32 accumulation order inside a different cuBLAS kernel ("reordered accumulation", the same class as choosing another
GEMM algorithm). Measured on the full fused TriMul (out+in) with real ESMFold2-Fast weights at L=705: max |diff| vs the current path 0.375
(bf16 pair units, values up to ~48), error vs an fp64 evaluation of the same math identical to the current path's own (max 0.792 vs 0.792,
rms 0.04391 vs 0.04391); deterministic run-to-run. Whole-model (deterministic config, 2 complexes x 2 model variants x 5 seeds):
|d ipTM| <= 0.005, superposed all-atom RMSD median 0.04-0.10 A vs a seed-to-seed spread of 1.4-1.6 A.

## Speed (H100 80GB HBM3, fused TriMul out+in, B=1, c_z=256)
| L | current ms | aligned ms | speed-up |
|---|---|---|---|
| 546 | 3.16 | 3.09 (L%8=2) | 1.02 |
| 691 | 6.53 | 5.48 | 1.19 |
| 705 | 6.92 | 5.39 | 1.28 |
| 780 | 8.53 | 7.73 (L%8=4) | 1.10 |
| 871 | 12.83 | 9.44 | 1.36 |
(numbers from a copy-free implementation of the same padding; this minimal patch pays for its two copies, ~1.15x at 705.)

## Testing
Bit-equality at L % 8 == 0 (the existing path is taken unchanged); at L % 8 != 0 compare max |diff| and the error against an fp64 evaluation
of the same contraction with the current path's own, as in "Numerics" above.
