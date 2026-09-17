# kernels.fpf_trimul — design note (rows `tmk3_exact`, `tmk3_fast`; a portable FlashPairformer member)

## Op and calling convention
TriangleMultiplication (outgoing | incoming) for modules that run the cuEquivariance fused op with `c_z == c_hidden == C`, one pair
tensor per call (`z [N, N, C]`, mask `[N, N]` or None), as five Triton / cuBLAS stages:

    S_in  x = LN_in(z)                                   rows of [N*N, C]
    A'    [a | b] = sigmoid(x Wg^T) * (x Wp^T) (* mask)   dual gated projection GEMM -> channel-major planes, row stride XS = plane_xs(N, pad)
    B     X_c = a_c b_c^T | a_c^T b_c                     cuBLAS on the plane buffers ('cublas') or the Triton kernel `_k_contract` ('triton', pad-64 planes)
    S_out y = LN_out(X)                                  over the channel axis of the channel-major result
    C'    out = sigmoid(x W_og^T) * (y W_o^T) (+ z)       output GEMM + gate (+ residual) epilogue -> [N, N, C] in z's dtype

Entries: `trimul.fn` (the stock module signature; serves `triangle_multiplicative == "cuequivariance"` with c_z == c_hidden, rank-3 z,
N > 100; everything else runs the stock function counted by word: `not-cueq`, `rank:<d>`, `n<=100`, an off class, `dtype:<cls>`),
`kernels.trimul_forward(z, direction, mask, w, cfg=..., residual=...)` and `kernels.pack_weights(...)` (the tensors exactly as the module hands
them to the library: `p_in` / `g_in` `[2C, C]` = a rows then b rows, output projection / gate, LN affine in their own dtype; GEMM weights in the
compute dtype). Inside opt_core the `kernels.trimul` face serves the rows `tmk3_exact` / `tmk3_fast` through `trimul_forward` with the
config its `tmk3_cfg(mode, dtype, N, C)` names.

## Modes and tables
`trimul.MODES`: **exact** = `pad=1, ln_mode="stock", contract="cublas"`; **fast** = `pad=16` (`FPF_TRIMUL_PAD`), `contract=FPF_TRIMUL_CONTRACT`
(`cublas` | `triton`), the package's own LayerNorm kernels. The process-wide mode is `FPF_TRIMUL_MODE` (default `exact`); `CONFIG_TABLE` is the
mode overlaid on the tile table. `TILES[(dtype class, C)]`: the A' and C' tile dicts (`BM, BN, BK, num_warps, num_stages`, kernel version `v`
selecting `_k_proj<v>` / `_k_out<v>`), the LN row tiles (`LN`, `LNO`); `TILES_DEFAULT[dtype class]` for widths without an entry.
`arch_tables.json`: per compute capability (`sm_<major><minor>`) replacements for the A' / C' tiles only (B and the LN stages are not
arch-tuned), admitted against the PROBED device's opt-in shared memory per block (`smem_bytes` estimate of the pipelined loop vs the device
property), `FPF_TRIMUL_ARCH_TABLE=0` = the default table everywhere. `cell_verdict(dtype class, C)` is the one per-process decision: a
recorded class is served on its tiles; a class recorded as off (`MEASURED_OFF`, e.g. the c 64 template stack under bf16 autocast) takes the
stock path BY NAME; an unrecorded bf16 / fp32 class is served on the default tiles and named once on stderr; a dtype the kernels do not compute
is `dtype:<cls>`. Fixed tiles, no autotune at run time; Triton compiles one configuration per stage at first use.

## Numerics and how the exact mode is exact
Compute dtype = the dtype the library's gated GEMMs run in: the autocast dtype under autocast (bf16 -> bf16 MMA), else the weight dtype
(fp32 -> TF32 with round-to-nearest converted operands when `torch.backends.cuda.matmul.allow_tf32`, else 3-pass TF32; `_cvt_tf32_rn`, `_dot`).
LayerNorm statistics fp32; sigmoid and gating fp32; one rounding per stored element; residual with the module's double rounding (update rounded
to the activation dtype, then added).

`tmk3_exact` reproduces the output of cuequivariance_torch's fused TriangleMultiplication bit for bit by construction, stage by stage:
1. S_in / S_out call the library's own `layer_norm_transpose` op at run time (`ln_mode = 'stock'`: `bijd->bijd` for the input, `dbij->bijd` on the
   XS = N planes for the output), so the LayerNorm outputs are the library's own bytes at every (rows, D). The package's LN kernels
   (`_k_ln_rows`, `_k_ln_cols_T`; the fast row) compute the same two-pass fp32 statistics in another summation order and are not bitwise equal to it.
2. A' and C' issue the products of the library's fused sigmoid-gated dual GEMM in its K order with fp32 accumulation, apply sigmoid, gate and
   mask at its rounding points, and round once.
3. B with `pad = 1`: the plane row stride is XS = N, so `torch.matmul` on the plane buffers is the library's own einsum problem (shapes, strides,
   dtype) and cuBLAS runs the same kernel with the same summation order. This is a property of the installed cuBLAS / library release, which is
   why the `kernels.trimul` face serves this row under the tier word `exact` only where `TRIMUL_CELLS.json` records the bitwise result on the
   caller's stack (else the stock row by name).
4. The construction is tile-invariant: A' / C' tiles change which program computes an element, not the order of its K loop, so the arch tables
   do not affect exactness.
`tmk3_fast` keeps the rounding points but pads the planes to 16 (another cuBLAS problem: tolerance class, summation order) and uses the
package's LayerNorm kernels (fp32 two-pass statistics in their own order).

## Known limits
CUDA only; one pair tensor per call (a leading batch is the stock path, `rank:<d>`); c_z == c_hidden; N > 100 (the library's own small-N path
below); mask float or bool `[N, N]`. Per call the planes `[2C, XS, XS-ish]` and the contraction result live for the call only (`key=None`);
`kernels.launch_grid_y(N, C, cfg)` states the largest grid axis-1 extent a call launches so a caller can refuse sizes past the CUDA limit before
launching. `exact_stages.py` is a stage-by-stage comparison script against the installed library (a development aid, not an entry).
