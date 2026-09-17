# E1 kit — what changes vs stock

Stock = Profluent E1 1.0.0 at the pin (STOCK.md). The kit never edits stock: under `exact` it applies one lever set to the loaded
model when an `E1Predictor` / `E1Scorer` is constructed (`e1_opt.apply(model)` is the same call for code that builds neither), and
every forward of that model then runs it — single-sequence or retrieval-augmented, any batch, any length; `off` loads none. A mode
is all of its levers: one that cannot run on the host is the mode's refusal by name, never a subset. Lever names are the ones printed
on the run's `[e1-opt] LEVER name=…` lines, in the order the kit applies them (`opt/forward/engines/e1/kits/eager`). "Stock's
kernels" below are upstream's own calls — flash-attn 2.8.3.post1, the Hugging Face hub Triton RMSNorm
(`kernels-community/triton-layer-norm`), torch 2.8.0 flex-attention: the kit changes when and how data reaches them and adds four
Triton kernels of its own (`kits/ew1`, compiled per card at first use) that reproduce stock's bf16 rounding sequence. Every lever's
numerics class is bitwise: `exact --det 1` equals `off --det 1` byte for byte.

## exact — outputs identical to stock

- `rmsnorm_autotune_pin` — the hub RMSNorm kernel picks `num_warps` by timing candidates at its first call, so two stock processes
  can pick differently; the kit fixes one of stock's own picks per (card class, model size) before any forward (`W=<n>` on the line;
  table `kits/pins.py` `TRITON_AUTOTUNE_PIN`). A card with no table entry takes its compute-capability class's default and the line
  says `source="default pin (no censused cell)"`. `--det 1` applies the same pin on the stock side.
- `P1_precast` — every `nn.Linear` weight and bias cast to bf16 once (the cast autocast otherwise repeats per call); norm weights and
  the MLM head's LayerNorm stay fp32 (`kits/v0`).
- `ew1:rope` — clamp(−8, 8) + rotary embedding on q/k (+ clamp on v) in one Triton kernel that writes the global layers'
  (B, H, L, hd) layout directly; one bf16 cos/sin table per rotary module (`kits/ew1`).
- `ew1:glu` — `silu(w1 x) * (w3 x)` in one kernel: fp32 `expf`, IEEE round-to-nearest division, one bf16 rounding, one bf16 multiply —
  stock's rounding points.
- `ew1:addnorm` — residual add (bf16) + RMSNorm with the hub kernel's reduction code at the pinned `num_warps`, in one kernel.
- `ew1:embed` — token + sequence-id embedding gather, fp32 add, one bf16 rounding, in one kernel.
- `attn:A1_varlen_shape_cu_seqlens` — within-sequence layers: `flash_attn_varlen_func` on the flat (B·L) views with `cu_seqlens`
  derived from the batch shape and cached per (B, L) — stock's kernel call without the unpad gathers, the pad scatter and their host
  syncs. Padded, multi-sequence, masked or KV-cached batches take stock's code path with the unpad data computed once per forward
  instead of once per layer (`kits/attn/adapter.py`).
- `attn:A2_cheap_blockmask` — global layers: `flex_attention` under an index-only all-true block mask with stock's block structure,
  built once per shape; no per-forward `create_block_mask`.
- `attn:A3_flex_kernel_options` — the global layers' flex kernel options fixed (`kits/attn` `FLEX_KERNEL_OPTIONS`) on card classes
  `h100` and `h200`. Steps aside: `state=off reason="OFF on <class>: …"` on the classes where those options are not bitwise with
  torch's own pick (`a100`, `b200`, `l40s`; table `kits/v1_2` `A3_BY_CLASS`) — torch then picks as under stock. A card of no listed
  class takes the setting of the class `MODEL_OPT_TARGET_GPU` names and the line says `untested class …`.
- `ew1_i64_offsets` — the four `ew1` kernels with int64 program offsets so index arithmetic cannot wrap on very large batches
  (`kits/ew1_i64`); an index guard counts calls beyond the int32 element range (`kits/eager/guard.py`).
- `anylen_flex` — one flex-attention callable serves every sequence length (`form=hybrid`): the kit's dynamic-shape compile with A3's
  tiling at ≥ 128 query tokens, stock's own compiled object below, so a shorter second input never re-enters torch's flex-decoding
  lowering (`kits/eager/anylen.py`).
- `multiseq:docmask` — rows holding several sequences (a retrieval-augmented prefill: context members + query) and padded rows:
  stock's block-causal document `BlockMask` built from per-block statistics of the sequence ids instead of evaluating the mask on the
  dense B×L×L grid — the same `BlockMask`, tensor for tensor, without the transient. Sequence ids that are not a sorted run with a
  padding suffix take stock's call (`kits/multiseq`).
- `multiseq:flex_opts` — the same rows' flex call runs the kit's pinned flex callable (A3's options where the class table has A3 on,
  torch's defaults where it is off): same `BLOCK_N`, same block visit order, same per-row reduction sequence.
- `kvcache:views` — retrieval-augmented scoring reuses a context's keys / values for every later batch (upstream's `KVCache`): the
  batch-1 cache is repeated as an expanded view instead of a per-layer copy, and selected back as a view (`kits/kvcache`).
- `kvcache:noappend_within` — within-sequence layers of a prefilled forward use the query's own K/V directly — what stock computes
  from `cat(cache, new)[:, -Lq:]` — and nothing is appended to the cache entry stock crops back afterwards.
- `kvcache:pack_global` — global layers of a prefilled forward: the packed [context ; query] K/V of every row written once into
  `flash_attn_varlen_func`'s layout (no repeat, concatenation or gather), `cu_seqlens` computed once per forward from the row layout,
  the kernel called as stock's `flash_attention_func` calls it; padded rows packed row by row.
- `tokenmemo` — host side: `E1BatchPreparer.prepare_multiseq` computes a context's token / position / sequence-id tensors and its
  cache key once per job with stock's own per-member calls, and each query once, instead of re-tokenising the whole context string
  per row — the same integer tensors, strings and exceptions (`kits/tokenmemo`).

`multiseq:*`, `kvcache:*` and `tokenmemo` act only where their mechanism applies (multi-sequence or padded rows, a prefilled context
cache, rows with a context); single-sequence jobs never reach them and run the levers above unchanged.

## Every mode

- No stock exceptions: both modes run E1 1.0.0 as released; the kit patches nothing in upstream.
- The `KERNELS` line (not a lever): each model process reads which accelerator objects it bound — flash-attn, the hub RMSNorm
  kernel, compiled flex-attention — and prints `[e1-opt stock|exact] KERNELS route=… flash_attn=… hub_layernorm=… flex_attention=…`
  with `engaged:<impl>@<version>`, `fallback:<what>(<why>)` or `absent(<why>)` per accelerator (`opt/e1_opt/accel.py`). Under `off` a
  fallback is stock's own path and the `EXIT` line lists it; under `exact`, whose levers run on those accelerators, it is the mode's
  refusal by name.

## Switches

- `--det 0|1` / `E1_OPT_DET` — the deterministic recipe on either mode (what it sets: STOCK.md); the default `0` touches nothing.
- `--variant` / `E1_VARIANT`, or `--model-name Profluent-Bio/E1-<size>` — the model size; the pinned checkpoints only.
- There is no per-lever switch: the set applies whole or the mode refuses by name.
