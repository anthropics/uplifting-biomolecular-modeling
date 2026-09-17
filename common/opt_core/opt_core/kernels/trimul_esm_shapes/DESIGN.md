# kernels.trimul_esm_shapes — design note (rows `esm_shapes`, `esm_shapes:f32in`; kernels behind `esm_k1ptr`)

## Op and calling convention
The `kernels.trimul` op boundary (pre-LayerNorm pair `z [N, N, c_z]` + `mask [N, N]` in; LN_in, four projections, two gates, the
contraction over c_hidden channels, LN_out and the gated output projection inside; `update` or `z + update` out, in z's dtype), B = 1
pairs, for the width pairs (c_z, c_hidden) in {(64,64), (64,128), (128,128), (256,256), (384,384)} and z bf16 (bf16 trunks) or fp32 (fp32 /
TF32 trunks: fp32 in and out, io word `f32`). Face (pure Python; torch / triton imported by `kernels` at the first served call):

    sel = select_cell(cc, io, c_z, c_hidden, n_tokens, direction)   # dict(key, cell, measured, size, note) | raises Unsupported(word)
    ok, why = supported(cc, io, c_z, c_hidden, n_tokens, batch=1)   # never raises
    out = triangle_multiplication(z, mask, direction=, weights=w10 | pack=, residual=, levers=None, cache=dict, pointer_k1=False)

Nothing reads the environment; a call outside the envelope raises `Unsupported(word)` before any launch and the caller keeps its own path
(the `kernels.trimul` face names `v4` / `cueq` as fallbacks). Weight packs are cached in the caller's dict under `pack_key` (data pointer,
shape, stride, dtype, device of the ten tensors).

## Implementation
K1 `kernels._k1s` (pointer loads) or `kernels_desc._k1s_tma` (tensor-descriptor reads of the z tile and the weight blocks; cc 9.0 cells,
needs `tl.make_tensor_descriptor`, imported only when a cell asks for it): LN_in + [a | b] projections + sigmoid gates + mask -> zero-padded
channel-major bf16 planes `ab[2*D, Np, Np]`. bmm: one cuBLAS strided-batched GEMM over the D channels (bf16 operands, fp32 accumulate; NT or
TN form, `plane_form`) -> `x[D, Np, Np]` bf16 (fp32 with lever `x32`). K3 `kernels._k3s`: LN_out(x^T) + output projection + the LN_in(z)
output gate recomputed on the z tile (+ residual, fp32 add) -> token-major output in z's dtype.
What the re-tiling adds, all compile-time: the pair width C (LN_in, the K1 projections' K, the K3 gate's K, the K3 output-channel loop) and
the hidden width CH (LN_out, the K3 out-projection's K) are walked as NCK x CK / NCH x CHK power-of-two chunks held resident (`chunk_plan`,
`chunking`: chunks <= 256, at most `MAX_CHUNKS`), so 384 = 3 x 128 runs without padding the reductions; with one chunk the kernels are the
derived-from line statement for statement (NOTICE). Levers a cell may set: `sigmoid` (tanh.approx gate), `stagger` (weight-block start offset per
CTA), `incnt` / `form` (contraction form NT | TN with the planes written in the orientation the form needs), `x32`, `stock_round`.
Launch cells: `TILE_TABLES.json`, keyed `<cc>|<io>|C<c_z>|H<c_hidden>|N<=<tokens>|<dir>` -> K1 / K3 tiles (BM, BN, num_warps, num_stages),
descriptor use, chunk widths, contraction form; per-cc `defaults` for widths inside the envelope at unmeasured sizes (`measured: false`).
`select_cell`: the smallest measured size >= n_tokens of the family, else its largest measured size, else the cc default. cc 9.0 cells read
through descriptors; cc 8.0 cells through pointer loads within sm_80 shared memory. Rows `esm_k1ptr(:f32in)` (`kernels/trimul/esm_k1ptr.py`)
serve the same cells with `pointer_k1=True`: the pointer-load K1 always (same planes byte for byte), for Triton builds without descriptors.

## Numerics
Tolerance class: bf16 tensor-core operands, fp32 accumulation, fp32 LayerNorm statistics (two-pass), fp32 gates / mask / residual, bf16
planes, one rounding per stored element; fp32 io = fp32 activations with bf16 MMAs (a tolerance-class substitute of an fp32 trunk's GEMMs,
gated in-model by the consumer). `kernels.reference_torch` is the op statement in torch at a chosen dtype (fp64 default).

## Known limits
B = 1; forward only; widths exactly the five pairs; CUDA cc >= 8.0 with a launch cell (or default) for the capability; the descriptor K1
needs the Triton descriptor API (else the pointer K1 through `esm_k1ptr`); transient memory per call = planes (2*D bf16) + x (D, bf16 or
fp32) at the padded extent (`kernels.workspace_bytes`). Kernels compile through Triton at the first call of a configuration.
