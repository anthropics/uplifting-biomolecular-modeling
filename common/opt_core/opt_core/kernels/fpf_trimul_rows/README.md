# fpf_trimul_rows (0.1.0) — row-block TriMul members beside fpf_trimul_v4 (c_z / c_hidden 64 and 384)

Reachable ONLY through `opt_core.mem.rowpair.trimul_fused` (the sharded row-block provider, `big --n_gpu P`, P > 1). No whole-plane
provider (`opt_core.kernels.trimul` / TRIMUL_CELLS.json, `fpf_trimul_v4` / table.json) imports this package or reads its table: every
single-GPU line resolves exactly what it resolved before this package existed.

| piece | kernel | reads | writes | copies it removes on the row-block path |
|---|---|---|---|---|
| K1 (one projection) | `kernels._k1r` | z row block `[rows, N, C]` channel-last IN PLACE (any row/token strides), mask `[rows, N]` (fp32/bf16/fp16, any strides) | a-layout `dst [D, rows, N]` (the resident A block) or b-layout `dst = buf[D, N, w].transpose(1, 2)` (the b sub-block in its GEMM layout) | torch path: `LN`, 4 linears, `permute(2,0,1).contiguous()` (A) / `permute(2,1,0).contiguous()` (B), the fp32 mask cast; fused v4 path at these widths: n/a (declined `c=64/64`, `c=384/384`) ; b-layout also removes `_b_slab`'s transpose copy |
| K3 (tile epilogue) | `kernels._k3r` | `T [CH, rows, w]` bf16 (channel-strided), the z column window `[rows, w, C]` IN PLACE (row stride `N*C_z`) | the same window (z's dtype), `+=` or `=` | torch path: `T.permute(1,2,0)` copy, LN_out, 2 linears, gate, the `zb += x` pass; v4-width fused path: the window staging copy in + copy out |

Widths: 64 / 128 / 256 = one channel chunk (then `_k1r` / `_k3r` are fpf_trimul_v4's `_k1c` / `_k3c` statement for statement after the
address arithmetic); 384 = 3 × 128 resident chunks (K1's `[BM, 384]` LN tile and K3's LN over 384 need no tile split: the same chunking
`trimul_esm_shapes` measured on H100 / A100 for the whole-plane (384, 384) line). Rounding points = fpf_trimul_v4's. bf16 or fp32 z.

`table_rows.json` (keys `<cc>|C<c_z>|H<c_h>`): `k1` / `k1b` / `k3` cells, `b_direct`, `min_tokens`, `measured` + `status`.
0.1.0 ships `9.0|C384|H384` (k1 64×128 w4 s1, k1b 64×128 w4 s1, k3 64×128 w8 s1, b_direct, min_tokens 1400) and `9.0|C64|H64`
(k1 128×32 w4 s2, k1b 64×32 w4 s2, k3 128×32 w4 s1, b_direct, min_tokens 2048) as **MEASURED_OP** (H100 tile sweep on
torch 2.10.0+cu128 / triton 3.6.0; ≤ 1 bf16 ulp vs the fp32 reference, C 64/128/256 bitwise equal to fpf_trimul_v4's kernels).
**cc 8.0**: `8.0|C384|H384` (k1 64×64 w4 s1, k1b 64×64 w4 s1, k3 64×128 w8 s1, b_direct, min_tokens 1400) and `8.0|C64|H64`
(k1/k1b/k3 128×32 w4 s1, b_direct, min_tokens 2048) MEASURED_OP, same correctness classes; `8.0|C256|H256` floor 0. Cards without a key keep declining `c=384/384` / `c=64/64` by name. A new
(cc, C) starts as a PLACEHOLDER (`measured: false`) which the provider declines by name unless `ROWPAIR_TRIMUL_ROWS_UNMEASURED=1`
(sweep / anchor use only). `9.0|C256|H256` is a floor-only entry (`kernels: fpf_trimul_v4`, `min_tokens: 0`): on H100 the (256, 256)
row blocks serve at every sharded size (floor rows measured 12–14× / 4× at (512..1024) × (1400, 2000)).

Safety net: a tuned cell that fails to BUILD falls to the table's `safe` cells by name (`[opt_core/pair_fused:trimul_rows] safe settings
served (build_failed:<Type>, cc …)`, census `settings=safe:…`); those failing too is the lever's refusal (`RowpairRefused`, names
`ROWPAIR_TRIMUL_KERNELS=torch`). A layout the kernels do not address raises `kernels.RowsUnsupported` before any launch and the provider
declines that unit by name (`layout`).
