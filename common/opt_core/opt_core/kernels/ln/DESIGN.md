# kernels/ln — design note (provider face and carried members)

## Op and face

`y = (x − mean(x)) · rsqrt(var(x) + eps) · gamma + beta` over the last dimension of a `[rows, C]` tensor.  Forms: `fp32` | `bf16` (bf16 x,
params, out) | `bf16w` (the autocast form: bf16 x, fp32 gamma/beta, fp32 arithmetic AND output) | `bf16o` (bf16 x, fp32 params and statistics,
bf16 OUTPUT — what a trunk's compiled LayerNorm extension returns under bf16 autocast; opt-in through `out="bf16"` / `out_dtype`).  Backward =
dx for FROZEN gamma/beta given (grad_y, x, mean, rstd), the residual-link gradient optionally folded in.  `lnlinear` = LN → a wide Linear.
`select(cc, dtype, cell, n_tokens, word=…)` is pure over `LN_CELLS.json` (`word` = a row name `row[:variant]` → exactly that row; a tier word →
the cell's measured winner; `capture=True` refuses capture-unsafe rows); `layer_norm(…)`, `layer_norm_backward_dx(…)`, `ln_linear(…)` import
torch inside the call; refusals are by name with the row's `.fallback`.  Rows: `exactln` (`:widen`, `:triton`), `fastln` (`:lp`), `ln_rows`,
`rfd`, `dtk_ln`, `ln_proj_ln_linear`, `ef2_ln_bwd_dx`; stock rows `aten`, `aten_autocast`, `fast_layernorm` (capture-unsafe, named),
`aten_autograd(_autocast)`, `esm_vendored_bwd`, `aten_bwd`, `torch_ln_linear1` — named, nothing carried.

## Members

* **`exactln`** (`exactln/`; EXACT vs ATen) — a bit-for-bit replica of torch's CUDA `layer_norm` forward, i.e. of ATen's
  `vectorized_layer_norm_kernel<T, float>` (torch 2.12.0), on a different schedule.  *How identity is achieved*: every output element goes through
  the arithmetic that kernel executes for its (32, 4)-thread CTA per row, restated with explicit round-to-nearest intrinsics so no compiler flag can
  change it (`exactln_fwd.cu` header carries the full specification): virtual lane t ∈ [0, 128) reads float4 vectors t, t+128, … and runs the ONLINE
  Welford update per element (new_count = count + 1; delta = val − mean; coef = rcp_rn(new_count); mean = fma(delta, coef, mean); sigma2 =
  fma(delta, val − mean_new, sigma2)); the warp tree `for offset in 16, 8, 4, 2, 1: combine(B = self, A = shfl_down(offset))` with ATen's
  `cuWelfordCombine` operation order; the shared-memory tree over the 4 warps (W0 = C(W0, W2); W1 = C(W1, W3); W0 = C(W0, W1)) INCLUDING the combines
  with empty partials (identities except in inf/NaN/−0.0 corners, so executed literally); var = div_rn(sigma2, C); rstd = rsqrtf(var + eps) with
  its non-FTZ expansion; y = fma(rstd·(x − mean), gamma, beta); bf16 storage converts exactly as ATen's `static_cast`.  Built WITHOUT fast-math /
  FTZ.  Schedule: one WARP per row (two rows per warp at C ≤ 64), many rows per CTA (256 threads, 8 CTAs/SM, grid-stride), rows gathered through
  the input's own leading strides (no `.contiguous()` copy of a transposed pair tensor), gamma/beta in registers — the schedule changes no bits.
  Served exactly where ATen takes its vectorized kernel: CUDA, x fp32 or bf16 with `stride(-1) == 1`, C % 4 == 0, 4 ≤ C ≤ 1024, weight/bias both
  or neither, row starts 16-byte aligned (8 for bf16); a contiguous-but-misaligned input (ATen's `RowwiseMoments` path, another arithmetic) is
  refused by name `aten_rowwise_path`; other refusals `width`, `dtype`, `affine_mismatch`, `import:cuda.bindings`.  Variants: `widen` (the bf16w
  form == `F.layer_norm(x.float(), …)`; with `out_dtype=bfloat16` the following cast is fused into the store, RNE — the bf16o form), `triton`
  (`triton_ln.py`: the same arithmetic as Triton block functions with inline-PTX `*.rn.f32` / `div.rn` / `rsqrt.approx` ops, C ∈ {64, 128}, so a
  fused Triton cell can normalise a tile in its prologue with ATen's bits — `exact_prologue()` on the face).  **Loading**: compiled per (C, affine,
  dtype) at first use with NVRTC through `cuda.bindings`, or through the same libraries bound with ctypes (`cubind.py`: `libcuda.so.1` +
  `libnvrtc`, the `(err, *values)` convention of cuda.bindings) where that package is absent; loaded and launched through the CUDA driver API on
  torch's current stream.  `prebuilt/exactln_fwd-sm80.cubin` / `-sm90.cubin` + `index.json` (source sha256, compile options, the lowered name of
  every (form, C, affine) instantiation the cells serve) are loaded as one module per arch, so a serving process compiles nothing where the bundle
  carries the kernel (`OPT_CORE_EXACTLN_PREBUILT=0` forces the compile path; `build_prebuilt.py` cross-compiles the bundles with NVRTC).
* **`fastln`** (`fastln/fastln.py`, carried module) — Triton row LayerNorm: one program per BLOCK_M rows, the whole row in registers, two-pass fp32
  statistics; tolerance class (tree-sum statistics, not Welford).  Variant `lp` (`fastln/lpout.py`, this tree's): the same kernel with the
  arithmetic pinned to fp32 whatever the storage dtypes and the OUTPUT dtype named by the caller — the bf16o form's row; int64 row offsets.
* **`ef2_ln_bwd_dx`** (`ef2/ef2_fused_ln.py`, carried; EXACT vs the vendored backward it replaces) — dx-only LayerNorm backward for frozen
  gamma/beta with the residual-link gradient folded in: x_hat = (x − mean)·rstd; wdy = dy·w; c1 = sum(wdy)/d; c2 = sum(wdy·x_hat)/d; dx = (wdy −
  (c1 + x_hat·c2))·rstd [+ link grad] — the same fp32 arithmetic in the same per-row reduction order as the ESM-family fork's `_ln_bwd_kernel`, on
  8 rows per 4-warp program and without the unused dgamma/dbeta partial sums, hence tensor-equal dx.  The channel-major operand form
  (`channel_major=True`, `_ln_bwd_cmajor_kernel`: 256-row runs × 32-column chunks) accumulates its two row sums per chunk and is tolerance class
  (fp32 reassociation), off by default.  Patch point `ef2_autograd_kernels._ln_bwd_dx_only`; install before any CUDA-graph capture of the trunk
  backward.
* **`ext_loader.py`** (row `fast_layernorm_ext`) — loader for a PREBUILT build of the trunks' fused LayerNorm extension keyed by the torch ABI key
  (every launch on torch's current stream, hence capture-safe; digest-checked; `available()` → `(False, "no_prebuilt:<key>")` where no binary for
  the key is present).  Exact class for an engine whose statement IS that extension; tolerance class against ATen.
* **`ln_rows`, `rfd`, `dtk_ln`, `ln_proj_ln_linear`** — the module kernels `kernels/ln_proj.py`, `kernels/rfd_layernorm.py`, `kernels/dtk_kernels.py`
  served in place (Triton row kernels, fp32 statistics; tolerance class, `rfd` bitwise on the bf16 cells the table marks).

## Known limits

`exactln` reproduces one ATen kernel of one torch line: a torch whose LayerNorm kernel changes arithmetic is a different reference (the cells
name the stacks where identity is recorded); it needs NVRTC (from `cuda.bindings` or the libraries torch ships) unless the prebuilt bundle covers
the instantiation.  No row produces dgamma/dbeta.  `fast_layernorm` (the stock extension) is capture-UNSAFE and is never a tier winner under
`capture=True`.  Carried member files are digest-pinned in `kernels/META/ln.json`.
