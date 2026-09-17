# kernels.fpf_trimul_rows — design note (row-block members of the triangle multiplicative update)

## What it computes and who calls it
The ROW-BLOCK statements of the triangle multiplicative update for the sharded row-block pair stack `opt_core.mem.rowpair.trimul_fused`
(tier word big with `--n_gpu P`, P > 1), at pair widths beside `fpf_trimul_v4`'s: c_z / c_hidden 64 and 384 (128 / 256 compile too).
Nothing here is a whole-plane TriangleMultiplication provider: no row of `kernels.trimul` (TRIMUL_CELLS.json) and no cell of
`fpf_trimul_v4/table.json` reads this package or its table, so every single-GPU line resolves exactly what it resolves without it. torch and
triton are imported by `kernels.py`, which the provider imports at its first served call -- never at this package's import (`__init__.py` is
pure: table, selection, words).

| piece | kernel | reads | writes |
|---|---|---|---|
| K1, ONE projection | `kernels._k1r` | z row block `[rows, N, C]` channel-last IN PLACE (any row / token strides), mask `[rows, N]` (any float dtype, any strides) | a-layout `dst [D, rows, N]` (the resident GEMM-A block), or b-layout = a `[D, rows, N]` view of `buf[D, N, w]` (the b sub-block in its GEMM layout: no transpose copy) |
| K3, tile epilogue | `kernels._k3r` | `T [CH, rows, w]` bf16 (channel-strided), the z column window `[rows, w, C]` IN PLACE (row stride `N*C`) | the same window in z's dtype, `+=` (residual) or `=` |

    K1:  x = LN_in(z);  dst = bf16( sigmoid(x W_g^T + b_g) * (x W_p^T + b_p) * mask )          one projection (a or b) per launch
    K3:  out_win[i, j, :] = sigmoid(LN_in(z) W_og^T + b_og) * (LN_out(T[:, i, j]) W_o^T + b_o)  (+ z_win[i, j, :])

The contraction between them and the block schedule belong to the provider (`mem.rowpair`), not to this package.

## Implementation
`_k1r` is `fpf_trimul_v4`'s K1 statement (`kernels._k1c`: LN_in with fp32 two-pass statistics on the register-resident tile -> the gate | proj
MMAs bf16 x bf16 -> fp32 -> (+ bias) -> sigmoid(g) * p * mask on the fp32 accumulators -> one bf16 rounding) reading z through two strides and
writing the planes through two strides + the plane stride: one kernel, two stride sets (`launch_k1`, layout word from `k1_layout(dst)`);
program = BM tokens of block row i (a-layout) or BM block rows at token k (b-layout, stored unit-stride along w). `_k3r` is `fpf_trimul_v4`'s K3
statement (`kernels._k3c`) on a column window of the shard. Widths 64 / 128 / 256 run as ONE channel chunk (then both kernels are the v4
statements after the address arithmetic); 384 runs as 3 x 128 resident chunks (`chunking(width) -> (CK, NCK)`), the chunking
`kernels.trimul_esm_shapes` uses for its whole-plane (384, 384) line. Fixed launch cells, no autotune at run time, no atomics.

Launch cells: `table_rows.json` (`TABLE_PATH`), keyed `"<cc M.m>|C<c_z>|H<c_hidden>"`; per key the K1 a-layout, K1 b-layout and K3 cells
(`{BM, BN, num_warps, num_stages}`; `tiles(entry)`), whether the b operand is written in its GEMM layout directly (`b_direct`), the size floor
`min_tokens` of the row-block lever at that (capability, width), `measured` + `status` words, and per capability the SAFE cells (`safe_cells(cc)`:
the `"<cc>"` entry else `"*"`). A key whose `kernels` word is `fpf_trimul_v4` carries no tiles: those widths keep v4's kernels and cells.

## Selection and refusals
`select(cc, c_z, c_h)` is the one statement: served iff the key exists, names this package's kernels and is measured (an unmeasured key serves
only under the engineering switch read by `allow_unmeasured_env()`, reason `served:unmeasured`); otherwise the reason word is `unmeasured` |
`no_row` | `v4` and the provider keeps its other path for that unit. A width / shape / layout / dtype the kernels do not serve raises
`RowsUnsupported` BEFORE any launch, by name; the provider declines that unit and counts it. `validate(table)` states a table's problems (key
grammar, kernels word, power-of-two integer tiles, measured/status agreement); `describe()` is the plain data a manifest prints.

## Numerics
Tolerance class, the rounding points of `fpf_trimul_v4`: bf16 tensor-core operands with fp32 accumulation, fp32 LayerNorm statistics, fp32
gating, one bf16 rounding of each plane element; K3 rounds gate * out to bf16 (`STOCK_ROUND`) before the fp32 residual add and stores in z's
dtype. `ref_k1` / `ref_k3` are the fp32 statements of the two pieces with the same rounding points, for tests and the provider's checks.

## Known limits
P > 1 only (single-GPU lines never reach it); forward only; widths and size floors exactly as the table states per capability; 384 requires
the 3-chunk path (K1's `[BM, 384]` LN tile and K3's LN over 384 channels are not tile-split). Kernels compile through Triton's JIT at the first
served call of a process (one configuration per cell).
