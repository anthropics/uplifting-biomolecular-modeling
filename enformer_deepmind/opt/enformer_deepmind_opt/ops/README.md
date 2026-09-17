# enformer_deepmind_opt.ops — the kit's TensorFlow GPU ops

TensorFlow custom ops, float32, GPU only, each computing one Enformer subgraph with **the same float32 operations, in the same
order, each rounded once, as TensorFlow 2.17.1's own GPU kernels** — so an op's output equals the subgraph's output bit for bit while reading and
writing the activation tensor once instead of once per TensorFlow op. The two elementwise building blocks first (their arithmetic is the whole
argument); `EdmPoolLogits`, `EdmBiasResidual`, `EdmRelShiftSoftmax`, `EdmLayerNorm` and the fused variants have their sections below.

| op | inputs → output | replaces (deepmind-research `enformer/enformer.py`) |
|---|---|---|
| `EdmScaleShiftGelu(x, scale, shift)` | `x (..., C)`, `scale (C)`, `shift (C)` → `y`, x's shape | `gelu(tf.nn.batch_normalization(x, …))` at inference: `v = x·scale + shift` with `scale = γ·rsqrt(σ²+ε)`, `shift = β − μ·scale` precomputed per channel, then `gelu(v) = sigmoid(1.702·v)·v` (`enformer.py` `gelu`) |
| `EdmSoftmaxPool2(x, logits)` | `x, logits (B, L, C)`, L even → `(B, L/2, C)`; or `(B, P, 2, C)` → `(B, P, C)` | `SoftmaxPooling1D` with `pool_size=2`, given the logits its `Linear` produced: `reduce_sum(x · softmax(logits, axis=pool), axis=pool)` |
| `EdmLayerNorm(x, gamma, beta; epsilon)` | `x (..., N)`, `gamma (N)`, `beta (N)` → `y`, x's shape; 1024 ≤ N < 4096 | `snt.LayerNorm(axis=-1)` at inference: `tf.nn.moments` over the channels then `tf.nn.batch_normalization` — `inv = rsqrt(σ²+ε)·γ`, `y = x·inv + (β − μ·inv)` with per-row `μ`, `σ²` |
| `EdmRelShiftSoftmax(content, rel)` | `content (B, H, L, L)`, `rel (B, H, L, 2L−1)`, L = 1536 → `probs`, content's shape | `attention_module.MultiheadAttention`: `tf.nn.softmax(content_logits + relative_shift(relative_logits))` — the attention weights |
| `EdmBiasAct(x, bias; act)` | `x (..., C)`, `bias (C)` → `y`, x's shape; `act` ∈ {`relu`, `softplus`} | a `snt.Linear` bias add followed by `tf.nn.relu` (the transformer MLPs' hidden layer, `enformer.py` `mlp()`) or by `tf.nn.softplus` (the two heads) |
| `EdmQScaleBias(q, bias_w, bias_r; scale)` | `q (B, H, T, K)`, `bias_w, bias_r (1, H, 1, K)` → `qw, qr`, q's shape | `attention_module.MultiheadAttention`: `q = q · key_size**-0.5`, then `q + r_w_bias` (feeding the content logits) and `q + r_r_bias` (feeding the relative-position logits) |

`EdmScaleShiftGelu` and `EdmSoftmaxPool2` take an int attr `arith_variant` (default 0). Variant 0 is the arithmetic TensorFlow 2.17.1 executes; the other values select the alternative
orderings documented below, kept in the binary for reference; the kit's graph rewrites always pass variant 0.

## Use

```python
from enformer_deepmind_opt.ops import load, check
edm = load()                                        # tf.load_op_library(edm_ops.so), once per process
y = edm.edm_scale_shift_gelu(x, scale, shift)       # arith_variant=0
p = edm.edm_softmax_pool2(x, logits)
z = edm.edm_layer_norm(x, gamma, beta, epsilon=1e-5)
info = check()                                      # {so_path, sha256, ops, tensorflow, archs, ptx, build}
```

`check()` raises `FileNotFoundError` / `RuntimeError` naming the cause when `edm_ops.so` or `BUILD.json` is missing, the library's sha256 differs from
the one `BUILD.json` records, a source file beside it differs from the one the library was built from, the running TensorFlow is not the version the
library was built against, or an op (or its GPU kernel) did not register. `python ops/__init__.py` prints `check()` as JSON. `load()` makes
`libcudart.so.12` resident first (from the `nvidia-cuda-runtime-cu12` package TensorFlow uses, else by soname), so the library loads in a process
that has not touched the GPU yet.

## The arithmetic

Notation: every line is one float32 operation rounded to nearest-even; `ftz` = subnormal inputs/results are zero (sign kept), the mode all of
TensorFlow 2.17.1's GPU float32 kernels run in. `expf` = CUDA libdevice `__nv_expf` (CUDA 12.3), the routine TensorFlow's kernels call; its own
steps are listed last. Instruction names are PTX; `edm_f32.cuh` spells each as inline assembly so no compiler setting can alter them.

**EdmScaleShiftGelu**, per element with channel `c`:

1. `v = x · scale[c]` — mul.rn.ftz (TensorFlow: Mul)
2. `v = v + shift[c]` — add.rn.ftz (AddV2)
3. `a = 1.702 · v` — mul.rn.ftz, `1.702` = float32 `0x3FD9DB23` (Mul)
4. `s = sigmoid(a)` — TensorFlow's GPU Sigmoid kernel for float32 (MLIR-generated `1/(1+exp(−a))`, flush-to-zero):
   1. `e = expf(−a)` (negation is a sign flip)
   2. `d = e + 1` — add.rn.ftz
   3. `s = 1 / d` — rcp.rn.ftz, the correctly rounded reciprocal
5. `y = s · v` — mul.rn.ftz (Mul)

Variant 1: step 4.2 fused with expf's final multiply (`d = fma.rn.ftz(p, 2ⁿ, 1)`, see expf below) — equal in value to variant 0 for every input.
Variant 2: step 4.3 as the special-function-unit reciprocal `rcp.approx.ftz(d)` (1 ulp; differs from TensorFlow on about a tenth of inputs).

**EdmSoftmaxPool2**, per output element, with the input pair `(x0, x1)` and logits `(l0, l1)` along the pool axis:

1. `m = (l1 > l0) ? l1 : l0` — TensorFlow's Softmax row maximum (cub::Max over the 2-wide row; a NaN in `l0` propagates, a NaN in `l1` does not)
2. `e0 = expf(l0 − m)`, `e1 = expf(l1 − m)` — sub.rn.ftz then expf
3. `s = e1 + e0` — add.rn (no flush; cub's float warp sum)
4. `w0 = e0 / s`, `w1 = e1 / s` — div.rn.ftz, IEEE division
5. `p0 = x0 · w0`, `p1 = x1 · w1` — mul.rn.ftz (Mul)
6. `y = p0 + p1` — add.rn.ftz (Sum over the pool axis)

Variant 1: expf written out instruction by instruction (below) instead of the libdevice call. Variant 2: step 4 as the approximate division
`div.full.ftz`.

Step 4's quotients are computed by `div_softmax` (`edm_f32.cuh`), which returns the division instruction's own result for every `(e, s)` a
softmax produces while keeping the pair's losing side off the instruction's slow path. The GPU has no division instruction: ptxas expands
`div.rn.f32` into a reciprocal-and-fma sequence behind an operand check, and a numerator that is zero, subnormal or within a few tens of binades
of the subnormal range takes a long divergent subroutine instead. In Enformer's pooling pairs that is the common case, not the exception: the two
logits of a pair typically differ by more than 87.3 — so the smaller side's `e` is `expf`'s flushed `+0` — in a large share of the pairs, by more than 70 in many of the rest, and hardly a warp is without one. `div_softmax(e, s)` (for `e` in
`{+0} ∪ [2⁻¹²⁶, 1]`, `s ≥ 1` finite; a NaN in either goes to the instruction): a zero or subnormal `e` gives `±0` with `e`'s sign (what the
flush-to-zero instruction returns for it); `0 < e < 2⁻⁶⁴` is divided as `q = RN(e·2⁶⁴ / s)` — the scaling is exact and the scaled numerator is
clear of the check — and returned as `q·2⁻⁶⁴`, exact and equal to `RN(e / s)` whenever `q > 2⁻⁶²` (round-to-nearest commutes with a power-of-two
scaling while no subnormal is involved); `q ≤ 2⁻⁶²` (a quotient at or below 2⁻¹²⁶, where rounding meets the flush) and every other `e` issue the
instruction on `(e, s)` unchanged.

**expf(a)** (libdevice 12.3 `__nv_expf`, flush-to-zero path), constants by bit pattern:

1. `t = fma.rn.ftz(a, 0x3BBB989D, 0.5)` (`0x3BBB989D` = log2(e)/252)
2. `t = min(max(t, 0), 1)` — cvt.ftz.sat
3. `j = fma.rm.ftz(t, 252, 0x4B400001)` (round toward −∞; `0x4B400001` = 12582913)
4. `n = j − 0x4B40007F` (`0x4B40007F` = 12583039; n is an integer in [−126, 126])
5. `r = fma.rn.ftz(a, 0x3FB8AA3B, −n)`; `r = fma.rn.ftz(a, 0x32A57060, r)` (log2(e) split high/low)
6. `p = ex2.approx.ftz(r)` — the GPU's single-instruction 2^x
7. `expf = p · 2ⁿ` — mul.ftz, `2ⁿ` assembled from j's low mantissa bits (`bits(j) << 23`)

## EdmPoolLogits(x [M, K], w [K, N], tail [T, N]) → y [M, N]   (float32, GPU)

Rows `0 … M−T−1` of `y` are `x[0:M−T] · w`; rows `M−T … M−1` are `tail`, copied. The product is computed in float32 with, per output element,
exactly this sequence of operations over k (the sequence the stock graph's strided-batched GEMM kernel performs for the pooling logits at these
shapes, K = N ∈ {768, 896, 1024, 1152, 1280, 1536}):

1. K is split in two halves, `[0, K/2)` and `[K/2, K)`.
2. Within a half, k runs in consecutive chunks of 256 (the last chunk of a half is shorter when K/2 is not a multiple of 256). A chunk's sum starts
   at +0 and takes one fused multiply-add per k in ascending order: `p = fma(x[m,k], w[k,n], p)`.
3. A half's sum is its chunk sums added left to right: `h = c0`, `h = h + c1`, `h = h + c2`.
4. `y[m,n] = h0 + h1`.

Every operation is a correctly rounded IEEE float32 operation (`__fmaf_rn`, `__fadd_rn`; this file is compiled with `--ftz=false --fmad=true`),
so the result is determined by this order alone. The rows of `tail` exist because the stock library computes a strided-batched GEMM of more than 65,535
two-row batches as launches over multiples of 65,535 batches plus a trailing launch of the remaining `nb mod 65,535`, and chooses each launch's
kernel by its batch count: a small trailing launch (or a small call as a whole) may run a kernel with another accumulation order, so the graph gives
exactly those batches to the stock `BatchMatMulV2` op and passes its result here as `tail`
(`enformer_deepmind_opt/_levers.py`, lever `poolgemm`). Execution structure: 64×128 output tiles computed by 128-thread blocks (up to three per
multiprocessor), each thread holding an 8×8 register tile; K in tiles of 16 through a three-stage shared-memory pipeline (`w` by asynchronous copies,
`x` staged through registers and stored k-major); the chunk accumulator lives in registers, a half's running chunk sum in per-thread shared-memory
slots, and the finished `h0` in `y` itself until `h1` is complete. The structure decides which thread computes an element and when — never the
per-element operation sequence above (the file header of `edm_pool_logits.cu.cc` has the details).

## EdmBiasResidual(y [..., C], bias [C], res [..., C]; assoc) → out [..., C]   (float32, GPU)

`assoc = 0`: `out = res + (y + bias)`; `assoc = 1`: `out = (res + y) + bias`; each `+` one `add.rn.ftz.f32` (TensorFlow's `BiasAdd`, `Add`,
`AddV2`). The graph as written adds the bias first; where an `Add` feeds the residual `AddV2` directly, TensorFlow's graph optimizer regroups
the add tree by shape and the stock graph executes `(res + y) + bias` — the lever passes `assoc = 1` exactly there (`_levers.rewrite_biasres`).

## Fused variants: EdmBiasScaleShiftGelu, EdmBiasGelu, EdmSoftmaxPool2Gelu, EdmBias2Residual   (float32, GPU)

Each performs, per output element, the operation sequence of two of the ops above (or of an op above and one TensorFlow kernel) back to back in
registers, so the intermediate tensor is neither written nor read back. No operation is added, removed or reordered, hence the output is the
composition's output bit for bit; only ordering 0 (TensorFlow's) is built. The TensorFlow kernel taken in is `BiasAdd` (channels last), which is
one `add.rn.ftz.f32` of the element and its channel's bias (`bias_op_gpu.cu.cc` `BiasNHWCKernel`) — the `add` of `edm_f32.cuh`.

| op | per element | in the Enformer graph |
|---|---|---|
| `EdmBiasScaleShiftGelu(x, bias, scale, shift)` → x's shape | `h = x + bias[c]`, then EdmScaleShiftGelu's five operations on `h` | the stem's and each conv-tower block's first convolution output → its bias → the pointwise block's BatchNorm → GELU (× 7) |
| `EdmBias2Residual(y, bias, res, res_bias)` → y's shape | `(res + res_bias[c]) + (y + bias[c])`: EdmBiasResidual's `assoc = 0` order with the residual input's own bias add recomputed | the same seven blocks' residual add, whose residual input is that biased convolution output — recomputing `h` costs one add and saves storing it |
| `EdmBiasGelu(x, bias)` → x's shape | `h = x + bias[c]`, `a = 1.702·h`, `s = sigmoid(a)`, `y = s·h` (the GELU's three kernels, as in EdmScaleShiftGelu) | the final pointwise convolution's bias and GELU (× 1). Not expressible as EdmScaleShiftGelu with `scale = 1, shift = 0`: `−0·1 + 0 = +0` would flip the sign of a `−0` input |
| `EdmSoftmaxPool2Gelu(x, logits, scale, shift)` → EdmSoftmaxPool2's output shape | EdmSoftmaxPool2's operations giving `p`, then EdmScaleShiftGelu's five operations on `p` | a pooling module whose output feeds only the next block's BatchNorm → GELU: the stem's and conv-tower blocks 0–4's (× 6) |

`EdmBiasScaleShiftGelu`, `EdmSoftmaxPool2Gelu`, `EdmBias2Residual` (and `EdmBiasResidual` above) each have two kernels with the same per-element
operation sequence: a row kernel (any `C`, any alignment: one thread block per row, as the ops above) and a `float4` kernel the launcher takes when
`C % 4 == 0` and every pointer is 16-byte aligned (always the case for whole TensorFlow tensors): one thread per four consecutive channels, 128-bit
loads and stores, the four elements' operation chains independent of each other — which is what lets these passes run at the memory bandwidth of
the card rather than at the latency of `expf` / `rcp.rn` / `div.rn`. `EdmBiasGelu` (one small site) has the row kernel only.

## EdmLayerNorm(x [..., N], gamma [N], beta [N]; epsilon) → y [..., N]   (float32, GPU, 1024 ≤ N < 4096)

Sonnet's `LayerNorm` over the last axis as its graph writes it — `tf.nn.moments` (`mean = Mean(x, -1)`, `var = Mean(SquaredDifference(x,
mean), -1)`) then `tf.nn.batch_normalization` with per-row statistics — one 256-thread block per row of N columns, every operation TensorFlow's:

1. `mean`: TensorFlow reduces a row of N ≥ 1024 columns with `cub::DeviceSegmentedReduce` (CUDA 12.3's CUB), one 256-thread block per row and
   its own `Sum<float>` functor (`a + b` in TensorFlow's flush-to-zero build: add.rn.ftz — the library's generic shuffle step calls the
   functor; its non-flushing `add.f32` shuffle specialisation serves `cub::Sum` only), in this order, which the kernel repeats step by step:
   1. thread `t` (0 … 255) sums its items `x[t], x[t+256], x[t+512], …` (those below N) left to right starting from the first: `a_t`;
   2. each warp `w` (threads 32w … 32w+31) runs five shuffle-down steps with offsets 1, 2, 4, 8, 16, a lane whose peer `lane + offset` exists
      replacing its value by `own + peer` — lane 0 ends with the balanced pairwise tree of the warp's 32 values: `W_w`;
   3. thread 0 adds the warp sums in turn: `S = ((((((W0 + W1) + W2) + W3) + W4) + W5) + W6) + W7`;
   4. the stored sum is `0 + S` (the reduction's initial value first), and `mean = (0 + S) / N` — an IEEE division by `float(N)`, div.rn.ftz.
2. `d = x − mean` (sub.rn.ftz), `sd = d · d` (mul.rn.ftz) — TensorFlow's `SquaredDifference` kernel — and `var` = step 1 over `sd`.
3. `a = var + epsilon` (add.rn.ftz; `epsilon` is the graph's constant, float32 `1e-5` in the released model); `rs = rsqrt(a)` as
   `rsqrt.approx.ftz.f32`, the single-instruction reciprocal square root TensorFlow's `Rsqrt` kernel executes (libdevice `__nv_rsqrtf`).
4. per column `c`: `inv = rs · gamma[c]`; `y = x · inv + (beta[c] − mean · inv)` — mul, mul, mul, sub, add, five roundings (the graph's `Mul`,
   `Mul`, `Mul`, `Sub`, `AddV2`, the first `Mul` being the broadcast of `rs` against `gamma` to the full row).

N = 4096 or more would take another path of the same library (full tiles, vectorized loads, another order) and is refused by name; N < 1024 takes
TensorFlow's warp-per-row kernel and is refused too. Replaces ten kernels per LayerNorm (two segmented reductions, the squared difference, the
epsilon add, the rsqrt and five full passes) by one that reads the row once and writes it once. The kernel is one template over the items a
thread holds: for Enformer's N = 1536 (six items) the launcher takes the instantiation with exactly six items and no per-item
bounds tests; any other N takes the 16-item form with item `k` present when `t + 256k < N` — the same operations in the same order either way.

## EdmRelShiftSoftmax(content [B, H, L, L], rel [B, H, L, 2L−1]) → probs [B, H, L, L]   (float32, GPU; L = 1536)

`probs[b,h,i,:] = softmax over j of ( content[b,h,i,j] + rel[b,h,i,(L−1)+j−i] )` — `attention_module.relative_shift` of the positional logits
(pad, reshape, slice, reshape, slice: the shift expressed as an index) added to the content logits with `add.rn.ftz` (the graph's `AddV2`), followed by
`tf.nn.softmax` over the last axis — TensorFlow's GPU `Softmax` op (tensorflow/core/kernels/softmax_op_gpu.cu.cc), which runs a row-maximum
reduction, a row sum-of-exponentials reduction and a normalization kernel. One block of 256 threads computes one row, with the logits held in
registers, performing per row exactly these operations:

1. `l_j = content_j + rel_(L−1+j−i)` — add.rn.ftz (AddV2)
2. `m = max_j l_j` with `max(a, b) = (b > a) ? b : a` (cub::Max), folded in the reduction order below starting from the row's own elements, then
   `m = max(−FLT_MAX, m)` (the reduction's initial value, `Eigen::NumTraits<float>::lowest()`, is combined last)
3. `e_j = expf(l_j − m)` — sub.rn.ftz, libdevice expf (SubtractAndExpFunctor)
4. `s = Σ_j e_j` in the same reduction order, then `s = 0 + s` (initial value 0)
5. `probs_j = expf(l_j − m) / s` — div.rn.ftz, IEEE division (GenerateNormalizedProb, not in log space; the exponential is the value of step 3)

The reduction order is the one TensorFlow's row reductions use for rows of 1,024 to 4,095 columns: `cub::DeviceSegmentedReduce` of CUB 2.2.0
(the CUB of TensorFlow 2.17.1's CUDA 12.3 toolchain), one 256-thread block per row holding a single partial tile —
(a) thread `t` folds columns `t, t+256, t+512, t+768, t+1024, t+1280` left to right: `a_t = op(…op(op(x_t, x_t+256), x_t+512)…, x_t+1280)`;
(b) each warp reduces its 32 values by shuffle-down steps of offset 1, 2, 4, 8, 16, lane `l` taking `op(own, lane l+offset)` while
`l + offset ≤ 31`, so lane 0 holds the balanced binary tree `((a_0+a_1)+(a_2+a_3))+… ` over lanes 0–31 (for the float sum CUB's step is its
inline `add.f32 peer, own`, a non-flushing add — the same value, every term being zero or a normal number);
(c) the 8 warp results are folded left to right, `g = op(…op(op(w_0, w_1), w_2)…, w_7)`; (d) `result = op(init, g)`.
`op` is add.rn.ftz for the sum and the compare-select of step 2 (setp.gt.ftz) for the maximum. Which thread performs step (c) is free (here
every thread does, instead of reading thread 0's value through shared memory); the operations and their order are not. Rows of another length
would take another TensorFlow reduction path, so the op accepts L = 1536 only (`kRelShiftSoftmaxCols`). `probs` may be written over `content`:
thread `t` of a row's block loads columns `t + 256k` into registers in step 1 and stores exactly those columns in step 5, so no thread reads an
address another thread writes. Replaces, per attention block, the relative shift's five kernels, the add and the three Softmax kernels — copies of and passes
over the (B, 8, 1536, 1536) logits — by one kernel that reads the two logit tensors once and writes the weights once.

## EdmBiasAct(x [..., C], bias [C]; act) → y [..., C]   (float32, GPU)

Per element with channel `c`: `v = x + bias[c]` — add.rn.ftz (the graph's `Add` / `AddV2`, or a channels-last `BiasAdd`: one add each), then

- `act = relu`: `y = max(v, +0)` — max.NaN.ftz. TensorFlow's GPU `Relu` for float32 is its MLIR-generated kernel: `tf.Relu` → `maximum(0, x)` →
  LLVM `llvm.maximum` → the NaN-propagating PTX maximum in which −0 orders below +0 (so `relu(−0) = +0` and `relu(NaN) = NaN`).
- `act = softplus`: TensorFlow's GPU `Softplus` for float32 is its MLIR-generated kernel, whose formula (`ConvertSoftplusOp`) is
  `t = log(2^−23) + 2`; `e = expf(v)`; `y = v > −t ? v : (v < t ? e : log1p(e))`, with `expf`, `logf`, `log1pf` = libdevice 12.3's
  `__nv_expf`, `__nv_logf`, `__nv_log1pf` in flush-to-zero mode (the routines this build links: `__nv_logf` and `__nv_log1pf` are sequences of
  correctly rounded `fma` / `add` / `mul` and integer operations without a division or a contractible multiply-add pair, so they execute alike
  under either compiler) and the two comparisons as written.

`y` may be written over `x` (each thread reads its element before writing it). Replaces the `Add` and the `Relu` of each transformer MLP's
hidden layer (tensors of (B, 1536, 3072)) and the `Add` and the `Softplus` of each head by one pass.

## EdmQScaleBias(q [B, H, T, K], bias_w [1, H, 1, K], bias_r [1, H, 1, K]; scale) → qw, qr [B, H, T, K]   (float32, GPU)

Per element with head `h` and key index `k`: `m = q · scale` — mul.rn.ftz (the graph's `Mul` by the float32 constant `key_size**-0.5`);
`qw = m + bias_w[h, k]`, `qr = m + bias_r[h, k]` — add.rn.ftz each (the graph's two `AddV2` with the broadcast `r_w_bias`, `r_r_bias`
variables). Three kernels and three passes per attention block become one kernel reading `q` once. Limits: `T·K ≤ 2^31 − 1`, `B·H ≤ 65535`.

## Build

`bash build.sh` — compiles `edm_ops.cu.cc`, `edm_layernorm.cu.cc`, `edm_pool_logits.cu.cc`, `edm_softmax.cu.cc` and `edm_biasact.cu.cc` with nvcc and `edm_ops.cc`, `edm_softmax.cc`, `edm_biasact.cc` with g++ against the
TensorFlow importable by `python3` (`PYTHON=` to choose), links `edm_ops.so` against `libtensorflow_framework.so.2` and `libcudart.so.12`, writes `BUILD.json`, then loads the library and lists the
registered ops. Requirements: tensorflow **2.17.1** (the build refuses another version unless `EDM_OPS_ALLOW_TF=<that version>` is set — the
arithmetic above is TensorFlow 2.17.1's), nvcc from CUDA **12.x** (12.3 is TensorFlow 2.17.1's CUDA; found as `$NVCC`, the `nvidia-cuda-nvcc-cu12`
package's `bin/nvcc` when it has one, `nvcc` on PATH, or `$CUDA_HOME/bin/nvcc`), the CUDA headers and `libcudart.so.12` (the
`nvidia-cuda-runtime-cu12` / `nvidia-cuda-nvcc-cu12` pip packages of the TensorFlow install, or `$CUDA_HOME`), g++ with C++17. No GPU is needed to
build. Flags: TensorFlow's own (`tf.sysconfig.get_compile_flags()` / `get_link_flags()`: include path, `_GLIBCXX_USE_CXX11_ABI`,
`EIGEN_MAX_ALIGN_BYTES`), `-O3 -std=c++17`, nvcc `--ftz=true --prec-div=true --prec-sqrt=true --fmad=true` (only `--ftz` reaches the arithmetic:
it selects libdevice's flush-to-zero `expf`, `logf`, `log1pf`; everything else is fixed by the inline assembly), device code
`-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90` (native SASS for compute
capability 8.0 and 9.0 — A100, H100 — which also serves 8.6/8.9 parts; compute_90 PTX for later architectures).

`BUILD.json` records: `tensorflow`, `python`, `nvcc`, `cxx` (versions), `cuda_include`, `archs` (`[[8,0],[9,0]]`), `ptx` (`[[9,0]]`), `build`
(`sm_80+sm_90`), `flags`, `ops`, `sources` (sha256 of every `.cc`, `.cu.cc`, `.h` and `.cuh` it compiled), `object` (file, sha256, bytes),
`built` (date). `check()` holds the tree to it.

## Files

| file | contents |
|---|---|
| `edm_ops.cc` | `REGISTER_OP` + shape functions + the GPU `OpKernel`s of every op but `EdmRelShiftSoftmax`, `EdmBiasAct`, `EdmQScaleBias` (TensorFlow headers; g++) |
| `edm_ops.cu.cc` | the elementwise kernels and their launchers on the op's stream (`edm_sigmoid<V>`, `gelu<V>`, `scale_shift_gelu<V>`, `softmax_pool2<V>`, the fused kernels; nvcc) |
| `edm_layernorm.cu.cc` | `EdmLayerNorm`'s kernel and launcher (the row reduction in TensorFlow's order, above; nvcc) |
| `edm_pool_logits.cu.cc` | the `EdmPoolLogits` GEMM kernel and its launcher (nvcc, `--ftz=false`) |
| `edm_softmax.cc` / `edm_softmax.cu.cc` | `EdmRelShiftSoftmax`: registration + GPU `OpKernel` (g++) / the kernel and its launcher (nvcc) |
| `edm_biasact.cc` / `edm_biasact.cu.cc` | `EdmBiasAct`, `EdmQScaleBias`: registration + GPU `OpKernel`s (g++) / `relu`, `softplus`, the two kernels and their launchers (nvcc) |
| `edm_ops.h` | the launcher signatures shared by the two sides |
| `edm_f32.cuh` | the float32 instruction vocabulary (inline PTX), `expf` both as the libdevice call and written out, `div_softmax` |
| `build.sh` | the build recipe above |
| `BUILD.json` | what was built, with what, from what |
| `edm_ops.so` | the built library (sm_80 + sm_90 SASS, compute_90 PTX) |
| `__init__.py` | `load()`, `check()`, `build_info()` |
