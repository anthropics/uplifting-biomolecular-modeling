# kernels.fpf_trimul_v4 — design note (the generic triangle-multiplication face; row `v4` of `kernels.trimul`)

## Op and calling convention
TriangleMultiplication, outgoing and incoming, forward, as three launches per call and no permute / transpose / copy kernels:

    x = LN_in(z);  a_ij = sigmoid(x_ij W_ag^T + b_ag) * (x_ij W_ap^T + b_ap) * mask_ij   (b_ij likewise; D channels each)
    X_ij = sum_k a_ik * b_jk (outgoing) | sum_k a_ki * b_kj (incoming)
    out_ij = sigmoid(x_ij W_og^T + b_og) * (LN_out(X_ij) W_o^T + b_o)  [+ z_ij if residual]

Entries: `generic.trimul(z, mask, direction=, weights=pack_weights(...), residual=)` / `generic.trimul_packed` (module-agnostic: tensors in,
optional biases, `supported()` answers without raising) and `trimul.fn` (the stock TriangleMultiplication forward signature: served cell =
CUDA, bf16 z under bf16 autocast, `c_z == c_hidden == 256`, `triangle_multiplicative == "cuequivariance"`; everything else takes the stock
forward for that call, counted by reason). Served by `generic`: CUDA z `[N,N,C]` or `[B,N,N,C]` contiguous, C = c_z in {128, 256}, D = c_hidden
in {128, 256}; bf16 z (fp32 z is accepted: pointer-load K1 with fp32 LayerNorm input, fp32 output); mask None / `[N,N]` / `[1,N,N]` / `[B,N,N]`
float or bool; N >= `N_MIN` (101 unless `FPF_TRIMUL_V4_NMIN`); a launch-cell row for the device. Outside that: `TrimulUnsupported(reason)`
before any launch. A lever that CANNOT run (its SAFE cell failed to build: `none:<why>`; the warm probe refused the shape class:
`probe-failed`; a kernel error) raises -- the caller's mode refuses by name; an out-of-memory propagates as torch's error and is never rerouted.
Residual: with the in-place flags of the module path a NEW tensor z + update is returned (z is not mutated); otherwise the update alone.

## Implementation
* **K1** (`kernels._k1c` pointer loads | `kernels._k1t` device-side tensor descriptors | `kdesc._k1d` host-built TMA descriptors, cell
  `impl='tma2'`): per program one (batch b, pair row i, token tile): LN_in with fp32 statistics on the register-resident tile, the [a|b] gated
  projections over the 2D output channels in weight blocks (double / triple buffered by `num_stages` in the descriptor kernels), mask, one bf16
  rounding -> channel-major planes `ab[2, B, D, Np, Np]`, Np = N rounded up to the cell's `pad` (16 on the module path), zero pad written by
  the kernel. Incoming with the host-descriptor cells: transposed planes, so the contraction is `a @ b^T` like outgoing.
* **bmm**: `a = ab[0]`, `b = ab[1]` as contiguous `[B*D, Np, Np]` operands of ONE cuBLAS strided-batched GEMM (bf16 operands, fp32 accumulate,
  bf16 out); all extents multiples of 16 keep cuBLAS on its aligned kernels. B = 1 is the `[2D, Np, Np]` layout byte for byte.
* **K3** (`kernels._k3c` | `kdesc._k3d`, cell `impl='tma'`): grid (token tiles, N rows, B); LN_out of the x tile (channels x tokens), output
  projection, output gate from LN_in(z) recomputed on the z tile, optional residual in fp32 -> `[B, N, N, C]` in z's dtype.
* **Launch cells**: `table.json`, ONE table keyed `'<cc M.m>|<triton M.m>'` (exact) then `'<cc M.m>|*'`, rows `{k1, k3, status, overrides}`
  with per-shape cells `k1_C<C>_D<D>[_bias]` / `k3_...` (`table.resolve_cfg`); a row serves when its status is admitted and the stack can build
  its cells (descriptor cells need Triton's tensor-descriptor API); at the parts listed under `kit_keys` a kit's own table (`FPF_TRIMUL_V4_CELLS`)
  serves; no admitted row -> the SAFE cell of `opt_core.kernels.safe_settings` BY NAME; else nothing serves (`no-cell`). Decided and printed once
  per device (`cells.cell_for`, `cells.cell_word`). Engineering switches: `FPF_TRIMUL_V4_CFG` (literal cells), `FPF_TRIMUL_V4_ALLOW_DEFAULT_CELLS`,
  `FPF_TRIMUL_V4_BATCH=loop` (B single-plane launch sets). Fixed tiles, no autotune at run time, no atomics, no split-K.
* **Warm probe** (`generic.probe`): once per process and shape class (C, D, bias, input dtype) the served kernels run on a random pair tensor at
  `PROBE_N` (and at `kdesc.N_MIN_DESC` when the cells name a host-descriptor kernel) against the fp64 statement streamed over output blocks; the
  class must meet the same-class bars (`SAME_CLASS_RMS`, `SAME_CLASS_MAX`). A passed verdict is stamped, keyed by package version + digest of the
  package's .py / .json files, cell word, device, driver, CUDA, torch, triton, python, so later processes skip it (`probe_all_classes` for an
  image bake).

## Numerics
Tolerance class, not bit-identical to the stock op: bf16 tensor-core GEMMs with fp32 accumulation, fp32 LayerNorm statistics (two-pass), fp32
gating, and the rounding points of the cuEquivariance op under bf16 autocast (LN_in out -> bf16 | gated projection -> bf16 AFTER gating (+ bias)
on the fp32 accumulators | contraction bf16 in / fp32 accumulate / bf16 out | LN_out out -> bf16 | gated output -> bf16 | residual add in fp32 ->
input dtype). Run-to-run reproducible and batch-invariant by construction (every batch element gets the single-plane instruction stream);
CUDA-graph replay equals eager. fp32 z: the GEMMs still run on bf16 operands -- a precision substitution a consuming kit gates in-model, not an
fp32 path. Variant `G` kernels (`_k1g` / `_k3g`, cfg mode `G`) add one bf16 rounding of the output gate and are engineering-only unless a table
row names mode G.

## Known limits and hazards
1. Memory per call: the planes `2 * B * D * Np^2` bf16 + the contraction result `B * D * Np^2` bf16 + the `[B, N, N, C]` output, all batch
   elements at once -- `generic.workspace_bytes(...)` = B x (3*D*Np^2*2 + N^2*C*elem_out) bytes, a documented formula and not a gate (about 4x a
   bf16 z at C = D).
2. No upper size limit in code. Every z / mask / plane / output offset in `_k1c`, `_k1t`, `_k3c` is formed in int64, so z, the planes and the
   output may each exceed 2^31 elements. The one int32 quantity is the TMA K1's descriptor row coordinate over `[B*N*N, C]`: limit
   `tma_rows>int32` (B*N*N <= 2^31 - 1 under a TMA cell; the pointer K1 has no such edge); the other named limit is `grid_batch>65535` (grid axis 2
   carries B); grid axis 1 carries Np (N <= 65535). `kernels.trimul_v4_forward` raises `kernels.BatchLimit(name)` BEFORE any allocation or
   launch; `generic.trimul_packed` then serves the batch as B single-plane launch sets, counted under the name; ONE pair plane beyond the row
   limit is the refusal `TrimulUnsupported('tma_rows>int32')`. The host-descriptor cells also need Np*Np <= 2^31 - 1 (`kdesc.serves`).
3. Byte-equality statements are per (stack, N, pad, direction, batch): `pad` changes the cuBLAS problem shape; a batched call is one GEMM at
   batch count B*D instead of D; the host-descriptor incoming cells contract `a @ b^T` (TN) instead of `a^T @ b` (NT). cuBLAS may pick another
   kernel for any of these on another stack or cuBLAS release, so equal bytes across them are an observed property of a stack
   (`tests/test_cells_eq.py`), never assumed downstream.
4. Below plane extent `kdesc.N_MIN_DESC` (512) and for fp32 z the host-descriptor cells hand the call to the replaced kernels (same bytes, their
   speed); which kernel served is tallied in `kernels.LAUNCHES`.
5. N below `N_MIN` -> the stock path (the stock library itself switches algorithm at N <= 100). mask: dense float / bool multiplied into a|b in
   fp32 before rounding; other mask shapes -> stock. Module path: bf16 autocast only (`not-bf16-autocast` otherwise).
6. The served math has LN_in inside; a module that normalises z outside the TriMul, or shares LayerNorms differently, must map its parameters --
   `generic.reference_torch` is the contract to test a new module layout against.
7. (128,256) / (256,128) resolve a row's (256,256) cells (valid, not tuned for those shapes); every row carries explicit (128,128) cells with and
   without bias under `overrides`.
