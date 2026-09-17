# Flashzoi kit — what changes vs stock

Stock = `borzoi-pytorch` 0.5.1 at the pin (STOCK.md). The kit has one optimized mode, `exact`; `off` loads nothing. Under `exact` the
kit attaches `kit.KitRunner(model)` (`engines.flashzoi.kits.v1_25`, PYTHONPATH root `opt/forward/kits_v1_25`) to each replicate once it is
on the GPU (`pred`) or at its first CUDA forward (`FLASHZOI_OPT` / `flashzoi_opt.enable`): the model object is patched in place, its
`forward` routes to the runner, and `kit.remove(runners)` restores it. A mode is all of its levers. Lever names are the ones printed as
`components=` on the run's ACTIVE line. The kernels are the kit's own Triton kernels (`kits/v1_25/fz_exact.py`, compiled per card at
first launch); they keep stock's fp16 activation dtype and reproduce each intermediate fp16 rounding of the stock op chain as an explicit
round-to-nearest conversion (inline PTX, floating-point contraction disabled on every launch), which is how every lever returns stock's
bytes. Attention stays upstream's FlashAttention-2 on every route.

## exact — outputs identical to stock

- `stage1` — the one-hot stem as one Triton kernel (`k_stage1_mma`): the DNA convolution as a tensor-core dot in cuDNN's reduction
  order with bias, max-pool, BatchNorm and GELU fused. Numerics: bitwise. Steps aside: never.
- `sites` — every pool→BatchNorm→GELU site as one fused Triton pass with the convolution bias folded in; the k=5 tower convolutions run
  channels-last on cuDNN's NHWC kernels with NHWC sites, the decoder stays NCHW. Numerics: bitwise. Steps aside: never.
- `crop` — the crop to the returned centre bins applied early, 8-aligned with a halo, so later layers work on the cropped length.
  Numerics: bitwise (same arithmetic on fewer positions). Steps aside: never.
- `head` — the human output head (a kernel-size-1 convolution in stock) as a cuBLAS `baddbmm` with the bias fused, fp32 with autocast
  off as in stock. Numerics: bitwise. Steps aside: a head of another size than the 7,611-track human head (e.g. after
  `set_track_subset`, or the mouse head) is served by upstream's own convolution + softplus on the kit's features.
- `precast` — trunk weights cast fp32→fp16 once at apply instead of by autocast on every call; the fp32 head and norms untouched.
  Numerics: bitwise (the same cast, done once). Steps aside: never.
- `graph` — CUDA-graph capture per batch shape at that shape's first call, replayed from then on; a graph is keyed by the batch and the
  human head it was captured with (a new track subset captures anew at its next call). Numerics: bitwise (scheduling only). Steps
  aside: a capture the device refuses is named on the kit's line and that shape runs the same kernels eagerly; out of memory propagates.
- `pinned` — the documented call's device→host path: each replicate's output copied non-blocking on a side stream into its slot of one
  page-locked result array from a small recycled pool (`engines/flashzoi/result_pool.py`), replicate k's copy under replicate k+1's
  forward; `pred` asks for C order (reordered on the device before the copy) and writes each window's file on a writer thread while the
  next window computes. Numerics: bitwise (placement only). Steps aside: when every pooled array is still held by the caller, pinned
  staging buffers and one host copy per replicate instead (same bytes).
- `ln_fused` — the transformer's LayerNorms swapped for a fused Triton LayerNorm (`k_layernorm_fused`) that reproduces ATen's
  vectorised LayerNorm, fp16 in and out. Numerics: bitwise. Steps aside: never.
- `transformer_fused` — each residual add fused into the following LayerNorm (`k_layernorm_fused_add`). Numerics: bitwise. Steps aside: never.
- `relu_epilogue` — the feed-forward block's first Linear with its bias + ReLU in the cuBLASLt GEMM epilogue. Numerics: bitwise. Steps aside: never.
- `decoder_fused` — the decoder's BatchNorm→GELU sites as one Triton pass (`k_bn_gelu`) instead of cuDNN BatchNorm + ATen GELU.
  Numerics: bitwise. Steps aside: never.
- `skip_fused` — the U-net skip path's bias add and NHWC→NCHW transpose in one Triton pass (`k_bias_nhwc_to_nchw`). Numerics: bitwise.
  Steps aside: never.
- `decoder_fused2` — the scale-2 upsample + skip add in one pass (`k_up2_add`, fp16 out), the horizontal convolutions' BatchNorm+GELU
  through `k_bn_gelu`, and the head GEMM followed by one fused bias + softplus kernel (`k_softplus_bias`). Numerics: bitwise. Steps aside: never.

Numerics class of the mode: `exact` constructs `kit.KitRunner(model)`; the runner's `numerics` knob has one accepted value, `"tf32"`,
its default — inside each call the kit sets
`torch.backends.cuda.matmul.allow_tf32` to the value `torch.backends.cudnn.allow_tf32` has at rest (the switch stock's cuDNN convolution
head obeys), so the head GEMM uses the same TF32 math as stock's head, and restores it on return; the process's switches at rest never
move. Bitwise equality to stock holds at PyTorch's defaults; a process whose switches at rest differ, or with a TF32 override variable
set, is served the same way and named as `drift` on the kit's line and the ACTIVE line, never refused.

Batch bound: one dispatch serves up to `KIT_MAX_BATCH` = 15 windows, upstream's own maximum at this window length (`torch.max_pool1d`
refuses a 16-window batch); the kernels index each window at int32 offsets from a 64-bit window base, and a larger batch through
`model(x)` goes through the same path in chunks of 15, concatenated (each chunk carries the bytes of a stock forward of that size).
`KitRunner.predict()` refuses a batch above the bound by name before any launch. The filters the kernels are written for (512, 608) are
asserted on the attached model at apply. `pred` runs one window per call, as stock does.

## Every mode

- Stock exceptions: none (STOCK.md). No upstream fix ships with this kit.
- Surfaces under `exact`: the attached model serves every documented upstream call — `forward(x, is_human, data_parallel_training,
  return_embeddings)` in every form and `get_embs_after_crop(x)` through the kit (the plain human-head call graph-dispatched, the other
  forms on the same kernels eagerly); `predict`, `predict_gene_count`, `set_track_subset` / `reset_track_subset` and `predict_tracks` are
  upstream's own code on top. `borzoi_pytorch.pytorch_borzoi_helpers.predict_tracks` over attached models runs
  `engines/flashzoi/predict_tracks_fast.py` (same forwards; the track slice applied on the device before the copy; result with stock's
  shape, strides, dtype and bytes). An input of another length than 524,288 bp goes through upstream's forward on the attached model
  (said once on the kit's line); `.to()` / `.cuda()` / `.cpu()` / `.half()` / `.float()` detach the kit (upstream's modules restored by
  reference, one line) and it re-attaches at the next CUDA forward; a model that is not a CUDA float32 model at its first call is served
  by upstream unchanged (`[flashzoi-opt] ASIDE model=<k>: <reason>`).
- Cards: one lever set on every card. H100 / H200 are pinned (`PINS['device_names']`, `PINS['device_classes']` by compute capability;
  `device class: pinned`); A100 is served by the class record `kits/v1_25/class_records/a100.json`, which names compute capability 8.0
  for this `fz_exact.py` (`device class: by record`, with a `memory is not a key` clause on a part of another memory size); any other
  CUDA GPU engages the same levers as `device class: unpinned` with `drift=[gpu …]` on the ACTIVE line. Nothing is sized from a card's
  memory. The mode refuses (`NOT ACTIVE`, exit 3) only when it cannot run: `borzoi-pytorch` off its pin, no CUDA device, the kit tree
  missing, or a kernel that does not compile or launch.

## Switches

- `--det` — the deterministic recipe on either route (STOCK.md, How stock is run); outputs under `exact` equal `off` with or without it.
- `--allow-partial` — a run whose lever evidence is incomplete after the run (`PARTIAL`) exits 0 instead of 3; outputs are kept either way.
- `FLASHZOI_OPT`, `MODEL_OPT_JIT_ROOT` / `TRITON_CACHE_DIR` — names only; values and defaults: STOCK.md Variables. The kit has no
  per-lever switch: `KitRunner(model, levers=…)` is the Python-level control, and the package always passes the full set.
