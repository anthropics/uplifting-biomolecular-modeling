# kernels.trimul — design note (the triangle-multiplication provider face)

FlashPairformer is the name of this release's pairformer-kernel family; for triangle multiplication its headline member is the CUDA-native
`trimul/native` package (`trimul_native`, arch-keyed cubins), `kernels.trimul_xla` is its XLA binding, and the `fpf_trimul*` Triton packages
(`fpf_trimul_v4`, `fpf_trimul`, `fpf_trimul_rows`) are its portable members; the face below serves all of them, and the other rows, by name.

## Op boundary
One provider over every carried implementation ("row") and the measured cell table `TRIMUL_CELLS.json`. Every row serves this boundary
unless its `rows` entry says otherwise; LN_in, the four projections, both gates, the contraction, LN_out and the gated output projection are
INSIDE it, and the weights are the ten canonical tensors of `opt_core.trimul_weights.WEIGHT_KEYS`:

    z [B,N,N,c_z] | [N,N,c_z] = PRE-LayerNorm pair tensor, mask [B,N,N] | [N,N] 0/1 (None = all ones)
    x = LN_in(z);  a = mask*sigmoid(x W_ag^T)*(x W_ap^T);  b = mask*sigmoid(x W_bg^T)*(x W_bp^T)
    X[i,j,:] = sum_k a[i,k,:] b[j,k,:] (outgoing) | sum_k a[k,i,:] b[k,j,:] (incoming)
    update = sigmoid(x W_og^T) * (LN_out(X) W_o^T);  returns update, or z + update when residual=True

`select(...)` is pure (json + words: capability, precision word `bf16 | f32z_bf16 | fp32 | tf32`, c_z, c_hidden, N, direction, residual,
batch extent, backward) and returns the row + facts; `triangle_multiplication(...)` serves through the selected row; the decision is recorded
once per call class in `opt_core.cell_census`. A row that cannot engage raises `Refusal(kind, row, fallback)` BY NAME -- never silently another
row's bytes. `compute_input` is the one rounding point a fused row applies to z under autocast (z cast ONCE to the autocast dtype).

## Rows
| row | implementation | envelope (refused by name outside) | numerics class | fallback |
|---|---|---|---|---|
| `v4` | `kernels.fpf_trimul_v4` generic face: Triton K1 / cuBLAS bmm / Triton K3, launch cells per (cc, triton) | cc >= 8.0, bf16, c in {128, 256} (+ (64,128) through the kernel face with the cell's launch cells) | tolerance (cueq-rounding class) | `cueq` |
| `tmk3_exact` / `tmk3_fast` | `kernels.fpf_trimul` kernel face with the EXACT-mode / FAST-mode config table (fast: padded planes, fused LN statistics) | c_hidden == c_z | exact vs the cuequivariance op / tolerance | `cueq` / `v4` |
| `tx_sm90a` / `tx_sm90a_exact` | `trimul/tx_sm90a` (design note: `DESIGN_tx_sm90a.md` beside this file): prebuilt sm_90a TMA + wgmma prologue / epilogue extension around one cuBLAS bmm; the exact assembly uses stock-order LayerNorms and unpadded planes | cc 9.0 + a prebuilt binary for the ABI key, bf16, c 256 | tolerance / exact vs the cuequivariance op | `v4` / `tmk3_exact` |
| `ef2_fused` | `trimul/ef2/ef2_trimul.py`: one autograd Function per TriMul, forward AND frozen-weight backward (d pair only); needs the ESM-family image's vendored GEMM kernels | residual required; import refusal elsewhere | tolerance | `cueq` |
| `ef2_cueq_tiles` | the same module's per-capability tile entries written into the stock library's in-process tuning cache; the op is the stock call (fwd and bwd) | per pin | exact (it IS the stock op) | `cueq` |
| `esm_v5_fwd` (`:f32in`) | `trimul/esm_v5`: Triton K1 with tensor-descriptor loads / cuBLAS bmm / Triton K3, levers incnt + formtab + sigmoid, plain-torch weight relayout (`route.py`) | cc >= 8.0, triton with tensor descriptors, bf16, c 128 / 256, forward; `:f32in` = z cast once to bf16 for fp32 / tf32 callers | tolerance (module-bf16 / fast-sigmoid class) | `v4` |
| `esm_v61` | `trimul/esm_v61`: the line above with K3 = a warp-specialized CuTe cubin (sm_90a, LN affine folded), driver launch, sealed payload | cc 9.0, bf16, c 256, forward | tolerance | `esm_v5_fwd` |
| `native` / `native_exact` / `native:f32in` | `trimul/native`: arch-keyed cubins (sm_90a, sm_80) K1 / K3 around cuBLAS bmm, sealed payload with byte gate | widths / forms per the payload's build manifest, B <= 64, N 16..4096, forward | tolerance / exact vs the cuequivariance op / tolerance vs the fp32 statement | `v4` / `tmk3_exact` / `v4` |
| `esm_shapes` / `esm_k1ptr` (`:f32in`) | `kernels.trimul_esm_shapes` re-tiled per width ((64,64) (64,128) (128,128) (256,256) (384,384)); `esm_k1ptr` forces its pointer-load K1 (same planes byte for byte as the descriptor K1) | cc >= 8.0 (k1ptr: triton >= 3.3), B 1 | tolerance | `v4` / `cueq` |
| `of3_form` | `trimul/of3_form.py`: the OpenFold-family module's INFERENCE statement issued whole-tensor (LayerNorm / Linear in the module's dtype rules, sigmoid-then-product gates, cuBLAS contraction in the module's 256-column blocks) | any c; forward | exact vs that MODULE (by measurement); tolerance vs the cuequivariance op | `torch_math` |
| `af3t_form` | `trimul/af3t_form.py`: the AF3-family torch module statement (fused row LayerNorm, ONE interleaved dual projection, mask-then-gate, one batched contraction, channels-last LayerNorm) whole-tensor | forward | exact vs that MODULE; tolerance vs the cuequivariance op | `torch_math` |
| `cueq` / `torch_math` | the stock library op / the module statements in torch ops | -- | reference | `torch_math` / none |

Sealed rows (`native`, `esm_v61`, and `tx_sm90a`'s prebuilt extension) load through their own faces: digests -> load -> load check -> byte
gate, once per process; see the DESIGN notes in those directories.

## Tier words and the exact contract
Kits bind TIER words; `word=<row>` serves exactly that row with its own launch cell. `fast`: the cell's measured winner on the caller's stack
(else the capability's reference stack), narrowed by `prefer=(row, ...)` to the rows a kit carries. `big`: the fast tier's rows ranked by
measured peak memory. `exact`: an exact-class row is a cell's exact value ONLY where the table records it bitwise-identical to the stock op on
the caller's stack (`exact_vouched`) and inside the size envelope and leading-batch extent that record covers (`exact_envelope`,
`exact_batch_vouched`); everywhere else the exact word resolves to the named stock row (`cueq`, or `torch_math`), by name. Module forms:
`select(word="exact", form="of3_module" | "af3t_module")` serves `of3_form` / `af3t_form` only at form cells where that row measured bitwise to
the engine MODULE the form names, else `Refusal("form:...")` and the kit keeps its module; form cells never enter the no-form resolution, and
fast / big ignore forms. Capabilities without cells read a measured capability's cells per family (`inherited_cc`), said in the census line.

## Members in this directory
* `esm_v5/` -- carried `ef2_trimul_v5.py` (Triton K1 `_k1x` / K3, levers) + its (cc|triton) launch-cell json + `route.py` (weight relayout,
  cell lookup, `check`, `forward`); imports torch and triton only.
* `ef2/` -- carried `ef2_trimul.py` (rows `ef2_fused`, `ef2_cueq_tiles`); imports the ESM-family image's model package at module level, so it
  is imported only where that image runs (the face turns the ImportError into a refusal).
* `of3_form.py`, `af3t_form.py` -- the two module-statement rows (`supported()` -> refusal word or None; `forward()`); `af3t_form` carries
  small Triton kernels for the engine-form row LayerNorm, the in-place mask/gate and the channels-last transpose.
* `esm_k1ptr.py` -- rows `esm_k1ptr(:f32in)`: words-only `refusal()` / `fallback()` and `serve()` through `kernels.trimul_esm_shapes` with
  `pointer_k1=True`.
* `native/`, `esm_v61/` -- sealed members with their own DESIGN notes; `tx_sm90a/` -- prebuilt member, its note is `DESIGN_tx_sm90a.md` here (the directory itself is digest-pinned); `TRIMUL_CELLS.json` -- the measured table;
  `_pystack.py` -- names kept for the large-frame import trampoline now in `opt_core._pystack`.
