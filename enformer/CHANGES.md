# Enformer kit — what changes vs stock

Stock = `enformer-pytorch` 0.8.12 at the pin (STOCK.md); its files are never edited. The kit is the package `enformer_opt`
(`opt/enformer_opt/`) plus its lever modules and prebuilt extension objects under `opt/forward/` (`enformer_kit/engines/enformer/kits/v0_2`:
the levers and `attach`; `kits/v0`: the trunk-graph wrapper; `kits/class_pins.py`: build selection; `bin/`: the sm_90 objects;
`classes/a100_80gb/`: the sm_80 objects). It installs and imports nothing of the shared core (`common/opt_core`); the two objects are built
from the sources in the tree (`enformer_fastkit.py` `_CUDA_SRC` / `_CPP_SRC`, `enformer_xattn.cu`) as `kits/v0_2/BUILD.json` and
`classes/a100_80gb/CLASS_PINS.json` describe (nvcc 13.0.88, torch 2.13.0+cu130 headers, `load_inline -O3`, no fast-math). `off` loads
none of it. The one kit mode is all of its levers; lever names are the ones printed in `levers=` on the `ACTIVE` / `APPLIED` lines.

## exact — outputs identical to stock

`ENFORMER_OPT=exact` (a meta-path finder enables the kit right after `enformer_pytorch` is imported) or `enformer_opt.enable()` wraps
`enformer_pytorch.modeling_enformer.Enformer.forward` at class level; each instance is fitted at its first eval-mode forward on a CUDA
device, or up front by `enformer_opt.apply(model, batch_sizes=...)`. Every lever reproduces stock's fp32 operations in stock's order,
so `model(x)` returns the bytes stock returns at equal batch size, on the sm_90 and the sm_80 build alike; no torch numerics flag is set.

- `poscache` — stock recomputes the relative-position basis, copies the TF-gamma table host->device and re-projects it (`to_rel_k`)
  inside every `Attention.forward` on every call; the kit computes it once per (layer, length, device) with stock's own code and keeps
  it on the device (class-level `Attention.forward`). Numerics: bitwise (caching only). Steps aside: never.
- `xattn` — each layer's relative-position attention chain (positional logits, `relative_shift`, add, softmax) as one kernel sequence
  from `enformer_xattn_ext.so` (CUDA C++, sm_90 / sm_80): a band GEMM computing only the n x n band `relative_shift` keeps, accumulated
  over the 64-wide key dimension in cuBLAS's order, then a fused add + softmax in ATen's reduction order; the content GEMM, `attn @ v`
  and `to_out` remain stock's cuBLAS calls (instance forward on every `Attention`). Numerics: bitwise (same arithmetic in the same
  order). Steps aside: a window length other than 196,608 bp or a key width other than 64 runs stock's attention arithmetic on the
  cached basis inside the same forward (exact).
- `fused` — the trunk's elementwise passes (conv bias adds, BatchNorm(eval)+GELU, residual adds, the AttentionPool softmax and
  weighting — each a separate read and write of the activation in stock) as five kernels from `enformer_fastkit_fused.so` (CUDA C++,
  sm_90 / sm_80): bias+BN+GELU keeping the biased tensor for the residual, bias+residual, pool tail, pool tail + the next stage's
  BN+GELU, BN+GELU. Each ConvBlock's `(BatchNorm1d, GELU)` pair becomes one fused module, `AttentionPool.forward` the fused tail, and
  `stem` / `conv_tower` get instance forwards running each stage as cuDNN conv (bias withheld) -> bias+BN+GELU -> cuDNN pointwise conv
  (bias withheld) -> bias+residual -> cuDNN pool-logits conv -> pool tail (+ next BN+GELU). The convolutions are the identical cuDNN
  calls; the kernels use round-to-nearest intrinsics, libdevice `expf` and true division, float4 row kernels when the length is a
  multiple of 4 and scalar kernels otherwise. Numerics: bitwise (same arithmetic in the same order). Steps aside: a length the pool
  size does not divide takes stock's padded `AttentionPool.forward` (never the case at 196,608 bp).
- `graph` — `model(x, return_only_embeddings=True)` under the patches above, captured once per batch size into a CUDA graph over static
  buffers (two warm-up forwards on a side stream first) and replayed; the two output heads run eagerly on the replayed embedding exactly
  as stock (single-window head arithmetic on the 2-D view, batched on the 3-D view, kept distinct as in stock); every lazily allocated
  buffer a graph reads is recorded at capture and checked before each replay. Numerics: bitwise (the same kernels, replayed). Steps
  aside: a capture is attempted only when `CAPTURE_GIB_PER_WINDOW` x batch + `CAPTURE_GIB_FIXED` (`kits/v0`) fits free device memory —
  otherwise, or when the capture runs out of memory, that batch size runs the same patched modules without replay (`UNGRAPHED
  model#k batch=B: <why>`, same outputs); an out-of-memory beside a graph's pool releases that graph and the size stays ungraphed.

Mode-level conditions, each named on its line: `NOT ACTIVE mode=exact reason=...` with nothing hooked — no CUDA device, `enformer-pytorch`
missing or not 0.8.12, a kit file or object missing, no build for the device (capability below 8.0), an object lacking a fused entry
point, the class hook not installable; `build=sm_90 | class:a100_80gb | sm_90-ptx` by capability 9.0 / 8.x / above 9.0, with a `notes=`
clause for a later 8.x minor, a device above 9.0 or a torch runtime other than 2.13.0+cu130 (served, never refused); `STOCK model#k` —
a training-mode model or one off a CUDA device runs stock until its first eval-mode CUDA forward; `EAGER model#k` — keyword call forms
(`head=`, `return_embeddings=`, `target=` ...) and other window lengths run the stock forward code under the patches, no graph (exact);
`AUTOCAST` — under `torch.autocast` the kit's path computes in float32; `NUMERICS` — a numerics switch changed since capture (matmul
TF32 / precision, cuDNN TF32 / autotuner, deterministic algorithms) releases and recaptures the graphs; an input that is not float32 on
the model's device goes to the stock code, which raises as stock does. After application `model.train()`, `.to()` / `.cuda()` /
`.half()` and the other moves, and an input requiring grad under grad mode, are refused by name. `enformer_opt.disable()` closes every
handle (patches are refcounted per model; stock modules restored, graph pools released at once), drops the hooks and prints `REMOVED`.

## Every mode

No stock exception and no upstream fix ships: `off` is stock untouched and `exact` changes nothing but the four levers above.

## Switches

`ENFORMER_OPT=exact|off` (STOCK.md §Variables) and the package calls `enable(strict=False)`, `apply(model, batch_sizes=())`, `disable()`,
`status()`, `check()`; `python -m enformer_opt check [--json]` = `run.sh check` (console script `enformer-opt`). The lever set is fixed:
there is no per-lever switch, no other variable, flag, file format or side file.
