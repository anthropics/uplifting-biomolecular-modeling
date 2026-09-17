# Enformer (official TensorFlow release) kit — what changes vs stock

Stock = the TF-Hub SavedModel `deepmind/enformer/1` at the pin, called through `model.predict_on_batch` (STOCK.md). Each lever is a
rewrite of one node group of that function's graph: at a loaded model's first prediction (or at `apply(model)`) the kit edits the
`GraphDef` of the single concrete function behind `predict_on_batch` and builds a new concrete function from it, bound to the same
captured variables (no weight is copied; TensorFlow's own graph passes run on it as on the restored one). `off` applies nothing. The one
mode, `exact`, is all fourteen levers; the names are the ones printed as `levers=…` on the `ACTIVE` and `APPLIED` lines, in the order
they are applied. Everything outside the node groups below is the stock graph node for node — every convolution (cuDNN) and every
matrix product of the attention, MLP and head layers (cuBLAS, TF32 where TensorFlow uses it) runs as the stock op with its stock attributes.

Kernels: every `Edm*` op below is the kit's own TensorFlow op library `opt/enformer_deepmind_opt/ops/edm_ops.so` (CUDA C++; sm_80 and
sm_90 native code plus compute_90 PTX; built against TensorFlow 2.17.1 by `ops/build.sh`; each op's arithmetic written out in
`ops/README.md`). No shared-core (`common/opt_core`) kernel is bound. Numerics, every lever: bitwise — each op performs, per output
element, the same float32 operations in the same order and rounding as the TensorFlow 2.17.1 GPU kernels it replaces, and a folded
constant holds the value the stock ops produce on the same device. Steps aside: no lever steps aside alone — a prediction graph in which
any lever's node group is not found as the released model has it keeps its stock function whole (`STOCK model#k <reason>`).

## exact — outputs identical to stock

- `pool` — attention pooling (`SoftmaxPooling1D`, ×7): transpose → softmax over pairs → transpose → multiply → pair sum becomes one
  `EdmSoftmaxPool2` pass.
- `poolgemm` — the pooling logits (×7), a strided-batched float32 GEMM of B·L/2 two-row batches, become one large `EdmPoolLogits` GEMM with
  the stock kernel's per-element accumulation order; the trailing batches the stock library would give a separate small launch still go
  through the stock `BatchMatMulV2`, selected by int32 shape arithmetic that TensorFlow evaluates on the host.
- `bngelu` — inference `BatchNorm` (`x·scale + shift`) → GELU (`sigmoid(1.702·h)·h`), five kernels (×14), becomes one `EdmScaleShiftGelu` pass.
- `bnconst` — each `BatchNorm`'s per-channel `scale` / `shift`, which stock derives from the moving statistics on every call (×14), are
  evaluated once through the stock ops on the same device when the graph is rewritten and held as constants.
- `poscache` — each attention block's relative-position keys (positional basis functions over 3,071 distances and their `r_k_layer`
  projection: input-independent, recomputed by stock on every call, ×11) are evaluated once the same way and held as constants.
- `relsoftmax` — the attention weights (×11): `relative_shift` of the positional logits (pad, reshape, slice, reshape, slice), their add to
  the content logits and `tf.nn.softmax` over rows of 1,536 become one `EdmRelShiftSoftmax` kernel that reads both logit tensors once, with
  the row reductions in TensorFlow's reduction order for that row length.
- `biasres` — a bias add followed by a residual add (×29) becomes one `EdmBiasResidual` pass, in the association the stock graph executes.
- `biasgelu` — a convolution's `BiasAdd` whose output feeds a `BatchNorm` → GELU and a residual add (×7) is folded into both consumers
  (`EdmBiasScaleShiftGelu`, `EdmBias2Residual`: the biased tensor is never stored); the final pointwise convolution's `BiasAdd` + GELU (×1)
  becomes `EdmBiasGelu`.
- `poolgelu` — a pooling output that feeds only the next block's `BatchNorm` → GELU (×6) becomes one `EdmSoftmaxPool2Gelu` pass; the pooled
  tensor is never stored.
- `hostshape` — the int32 product B·T that Sonnet's `BatchApply` computes on the GPU for a reshape (a host → device → host round trip the
  host waits on, ×11) becomes the constant `-1`, which `Reshape` resolves to the same value on the host. Numerics: placement only.
- `layernorm` — the transformer's layer normalisations (`tf.nn.moments` + `tf.nn.batch_normalization`, ten kernels, ×22) become one
  `EdmLayerNorm` kernel per site that repeats the summation order of TensorFlow's row-reduction kernel.
- `biasact` — a `Linear` bias add followed by `relu` (the MLP hidden layers, ×11) or `softplus` (the two heads, ×2) becomes one `EdmBiasAct`
  pass with TensorFlow's activation formulas.
- `qbias` — the query scaling `q·key_size^-0.5` and its two bias adds `+ r_w_bias`, `+ r_r_bias` (×11) become one `EdmQScaleBias` pass
  writing both sums.
- `hostcrop` — stock's first op crops each 393,216-position window to the central 196,608 positions after the whole window has been
  uploaded; the kit declares the input with the cropped length and uploads only the central positions of each window (a contiguous view per
  window, no host copy), half the host-to-device transfer. Numerics: placement only — a crop performs no arithmetic.

The call itself: given a C-contiguous, writeable float32 NumPy array in a plain eager call, the kit hands the array to TensorFlow through
DLPack (no intermediate host copy; the function's own host-to-device copy completes before the call returns) and invokes the rebuilt
function without per-call signature binding. Any other argument converts as stock converts it. A call under a `tf.GradientTape` or while
TensorFlow traces a `tf.function` runs the stock function (`STOCK-CALL model#k <reason>`, once per model: the rewritten graph serves eager
forward calls and its ops define no gradients); so does an argument that is not a float32 `(B, 393216, 4)` batch. The constants of
`poscache` and `bnconst` hold values derived from the model's variables at rewrite time: a program that assigns new values to those
variables afterwards calls `disable()` then `enable()` again, or applies the kit after assigning.

## Every mode

- The stock exception (STOCK.md §Stock exceptions: the TensorFlow 2.17.1 stack) applies identically with and without the kit. No opt-in
  upstream fixes ship.
- Refusals, by name on stderr (`[enformer-deepmind-opt] NOT ACTIVE mode=exact reason=…`, nothing applied; exit 3 under the environment
  switch): TensorFlow not importable; no CUDA device visible to TensorFlow; the kit's files missing (it is installed editable from this
  directory); `edm_ops.so` missing, differing from `ops/BUILD.json` (the library or a source beside it), built against a TensorFlow other
  than the running one, an op not registered, or holding no code for the device's compute capability (below 8.0); the load hooks not
  installable. `ACTIVE … notes=…` = engaged with a named caveat: a device outside 8.0 / 9.0, or a stack component other than the pin.
  `APPLIED model#k … weights=…` names `deepmind/enformer/1` when the SavedModel's files match the pin by digest; a SavedModel with the same
  graph and other weights is served and named as such.

## Switches

- None beyond the mode: `ENFORMER_DEEPMIND_OPT=exact|off`, or `enable()` / `apply(model, batch_sizes=…)` / `disable()` / `status()` in
  code; `python -m enformer_deepmind_opt check` (`run.sh check`) is the dry run. Individual levers are not switchable.
