# U5 — fused pair-bias kernel: 64-bit row offsets (`kernels/fused_attention_pair_bias.py`)

**Defect.** `_pair_bias_kernel` (and `_pair_bias_backward_kernel`) form every element offset in 32-bit integers: z `[B, Q, K, DIM_Z]` is read at
`pid_b*Q*K*DIM_Z + pid_q*K*DIM_Z + k*DIM_Z + c`, the bias `[B, H, Q, K]` written at `pid_b*H*Q*K + h*Q*K + pid_q*K + k`. Once
a TERM passes `2**31 - 1` it wraps negative (each term is added to the 64-bit pointer on its own, so the terms wrap one by one, not their sum):
the sample term `pid_b*Q*K*DIM_Z` — `B` is the fold's diffusion samples (`AttentionPairBias.forward` repeats z per sample) — for sample index
b once `b*L*L*256 > 2**31 - 1` (**the fifth sample from 1,449 tokens**, the fourth from 1,673, the third from 2,048), and the row term
`pid_q*K*DIM_Z` for rows `pid_q >= ceil(2**31 / (K*DIM_Z))` once **L >= 2,897 at any sample count**. A wrapped load reads 2**32 elements below
its address — a whole sample's bias (sample term) or a row band (row term): an illegal memory access when that address is unmapped
(`CUDA error: an illegal memory access was encountered`; under `CUDA_LAUNCH_BLOCKING=1` the launch of `_pair_bias_kernel` itself raises), else
those bias rows silently read from other memory — the fold then fails later (`torch.linalg.svd … failed to converge` in the sampler's
rigid alignment once the coordinates are non-finite) or completes with those rows wrong. It is an addressing fault, not memory pressure: the same
plane launched in row blocks runs in the same memory. `AttentionPairBias` (the diffusion token transformer) is the kernel's one caller, so stock's
`--backend fused` meets it at the first diffusion step of any fold past the bound — out of reach on an 80 GB card with one sample, within reach
with five.

**Patch.** Promote the row index to int64 before it is scaled: `(pid_b.to(tl.int64) * Q + pid_q) * K * DIM_Z` for z / d_z, the same for the
saved LN statistics, and `((pid_b.to(tl.int64) * NUM_HEADS + offs_h) * Q + pid_q) * K` for the bias / d_bias, in the forward and the backward
kernel alike. Per-element arithmetic is untouched (bit-identical outputs where the unpatched kernel ran at all); the in-tile offsets
(`k*DIM_Z + c`, at most `TILE_K*DIM_Z`) stay int32.

**In the kit (not this patch).** `driver/ef2_opt.py` `install_pair_bias_rows` routes the module's `_fused_pair_bias` through
`fused_pair_bias_rows`: a plane inside the bound is the one launch it always was; a larger plane is launched one sample at a time in row
blocks of z that each stay inside `2**31 - 1` offsets (`pair_bias_launch_rows`; 2,763 rows per launch at 3,036 tokens), every block a
contiguous view of z, the blocks' `[1, H, rows, K]` outputs written into one `[B, H, Q, K]` buffer. Same kernel, same per-element arithmetic:
the row-blocked bias equals one launch's element for element, and equals this patch's single launch where the unpatched kernel cannot run.
The EXIT line's `ef2_opt` stats count the blocked launches (`pair_bias_row_launches`).
