# GPN-Star kit — what changes vs stock

Stock = `gpn` 0.9.0 at the pin, run with its own speed flag `--tf32` (STOCK.md). The kit never edits stock: with the switch on, a hook on upstream's inference
wrappers (`MLMforVEPModel`, `MLMforLogitsModel`, `ModelCenterEmbedding`) attaches the levers to the wrapped `GPNStarForMaskedLM` (`.model`) when the wrapper is
built and the model is on the GPU, by replacing bound methods and buffers on that object; every forward then runs them (`gpnstar_opt.apply(model)` does the same to
a model you built). The switch unset loads none. The mode is all eight levers, always; the run's `[gpnstar-opt] LEVER name=… state=on origin=kit` lines name them in
apply order. Every lever changes where data lives and how many rows reach stock's own GEMMs, never the arithmetic; two of them carry small kernels of
their own for that data movement and re-summation: `colattn` two Triton kernels (a gather and one fma accumulation) and `fusedattn` two CUDA kernels
compiled at first use through torch's NVRTC bindings (no CUDA toolkit), each reproducing cuBLAS's own fp32 summation order and each guarded per batch
shape by whole-tensor equality against the stock ops on the run's data; weights, dtype (fp32) and every other operation are untouched. Calling contract, checked at engage and
per batch shape (else `NOT ACTIVE`, exit 3): the model on a CUDA device, one target row per window, `target_species` 0 — true of every `gpn star` path.

## exact — outputs identical to stock

Identical means bitwise, per identical batch (same windows, batch composition, card, stack and numerics flags): `torch.equal` logits, byte-identical
`scores.parquet`; compare the kit with stock at one batch size under the same flags.

- `devconst` — the phylogenetic-distance arrays live on the GPU as model buffers; stock holds them as fp64 host arrays and copies them to the device on
  every forward. Numerics: bitwise (data movement only). Steps aside: never.
- `constcache` — the per-batch-shape constants stock recomputes each forward (clade-mean distances, the clade attention mask, the FIRE time-bias MLP outputs,
  the sinusoidal positions) are produced once per shape by stock's own operations and reused. Numerics: bitwise (the stock ops' own outputs). Steps aside: never.
- `srcgather` — the source module indexes species with device tensors and gathers the singleton-clade embeddings in one call; stock indexes with Python
  lists, at a host-to-device copy per call. Numerics: bitwise (the same `attn_pool` calls on the same shapes; gathers are copies). Steps aside: never.
- `unifiedkv` — column cross-attention K/V: the possible singleton-clade rows and every multi-species clade row are projected in one GEMM per layer over
  that reduced row matrix, padded to a fixed row multiple, and gathered into place; stock projects all batch × length × clade rows. Numerics: bitwise (identical
  input rows give identical output rows in the GEMM at these row counts, under fp32 and under TF32; the layer-0 check below guards exactly that at run time). Steps aside: by route (below).
- `dedup` — on top of `unifiedkv`, multi-species clade rows are de-duplicated by their species-token tuple before the projection (a clade row's K/V
  depends only on the alignment column, not on the target), so one eager GEMM runs over the distinct rows; a representative stands for a row only where stock's
  pooling produced bit-identical values for both (the few rows of a large batch that round differently stay rows of their own). Numerics: bitwise.

- `colattn` — on top of `unifiedkv`/`dedup`, the column cross-attention itself. Stock hands per-head views of the clade K/V to
  `F.scaled_dot_product_attention` (math backend), inside which `at::matmul` cannot view those layouts and so writes, per layer, a scaled copy of Kᵀ, a
  contiguous copy of that and a contiguous copy of V — three passes of pure memory traffic over (batch·length·clades × 512) fp32 tensors — before cuBLAS reads
  each operand once. The lever runs the same op sequence (q·s, K·s with s = √scale applied as ATen applies it, matmul, in-place mask add, `_safe_softmax`,
  matmul) but builds the operands once, in their final layout, straight from the DISTINCT clade rows: the key rows are scaled before the gather (an
  elementwise multiply commutes with a gather bit for bit) and gathered directly into the contiguous (B, L, A, D, C) tensor stock's clone would be; matmul's
  reshape is then a view with the clone's strides, so cuBLAS receives stock's problem on stock's values. attn·V goes one step further where the card allows:
  cuBLAS's batched gemv sums each output element as one fp32 fma chain over the clades in increasing order (for every problem of a full 65 535-problem
  chunk; the trailing problems get their own launch and heuristic), so a Triton kernel reproduces that chain reading the distinct value rows from the
  (L2-resident) table — V is never materialised — and the trailing problems go to `torch.matmul` on rows gathered for them alone. That summation order is an
  observed property of the library, not a documented one, so it is decided per batch shape: the shape's first forward computes layer 0's context both ways
  and compares with `torch.equal`; a shape that does not compare equal keeps the by-construction operand + cuBLAS. Engages from 2 048 target tokens per
  forward (B·L; below that the forward is launch-bound and the `graph` lever serves it); steps aside with the K/V route (route=stock, memory, self-check
  reject), for `output_attentions`, and in training mode. Two small Triton kernels (a gather and one fma accumulation; torch fallbacks built in).
- `fusedattn` (accel/fused_col.py) — on top of `colattn`: the column cross-attention reads the reduced key/value rows THROUGH the gather index instead of
  gathering them to full size: per layer, one kernel computes the (B, L, heads, clades) scores and one the context, each output as the fp32 FMA chain the
  cuBLAS batched-GEMV kernel of stock's SDPA math backend runs at these shapes (increasing index from +0.0; the remainder of cuBLAS's 65,535-matrix
  chunking under that smaller call's own summation tree, decoded per shape from a fixed candidate list), with SDPA's q·√scale / k·√scale roundings and
  mask add folded in; `_safe_softmax` stays the stock ATen op. The full-size keys and values, their scaled/transposed copies and the two gathers are never
  made. Guard: the first forward of a batch shape on this route rebuilds, at one layer, the materialised stock ops on the real data and keeps the route
  only if fused scores, fused context and the whole attention output are `torch.equal` to them (the remainder tree is chosen by that equality); a shape
  that does not reproduce them runs the `colattn` route, warned once (`RuntimeWarning: gpnstar_exact: the fused column attention did not reproduce …`);
  a card or driver where the kernels do not compile disables the lever for the process, loudly, and the route stays exact. The summation order the
  kernels reproduce is the one cuBLAS runs on the H100 class (sm_90); on the A100 class (sm_80) cuBLAS sums these shapes differently and the guard has
  stepped aside on every shape measured (64 / 256 / 727 windows of 128 bp), so that class runs the `colattn` operands — named per shape, exact either way. Engages from 2 048 target
  tokens per forward like `colattn`; steps aside with the K/V route and per shape by its guard. Memory: steady state below `colattn` (no K-sized
  transient at all); the guard's one-layer reference costs up to stock's one-layer attention footprint once per shape.
- `graph` — small batches: a batch shape of up to 2 048 target tokens (B × L: batch 16 at 128 bp, 4 at 512 bp) is recorded ONCE into a whole-forward CUDA
  graph at its first forward after a clean eager one (the eager first forward of a shape does the self-check and the constant caches, outside any capture;
  the `colattn` Triton gathers are compiled and probed before it) and every later forward of that shape replays the graph: one launch, no Python between the
  ~1,100 kernels of a forward, which at these sizes is what the wall time was. Static input buffers per shape (inputs are copied in on the device),
  `target_species` checked against the captured constant per call, the output handed back as a fresh copy; one graph pool per model; a graphed shape runs
  the unified-K/V route (the de-dup's distinct-row count is data dependent, hence not capturable). Printed once per shape: `[gpnstar-opt] GRAPH
  state=captured shape=<B>x<L> cost_s=<s> pool_mib=<MiB> graphs=<n>` (capture 0.3–0.7 s; pool 0.1 GB at batch 1, 0.7 GB at batch 8, 1.2 GB at batch 16 of
  128 bp) or `GRAPH state=ungraphed shape=<B>x<L> reason=<why>` when the capture does not fit the free device memory by the footprint rule or fails — that
  shape then runs the same patched modules eagerly, exact either way. Larger shapes are untouched (with `colattn` the eager route is bandwidth-efficient and
  wins from batch 24 of 128 bp up). Not graphed, hence eager as before: grad mode, `train()`, `labels` / `output_probs` / `loss_weight` / attentions /
  hidden states requested, positional arguments, a caller that is itself capturing.

Routes. The projection path is decided per batch shape and printed after the shape's first forward as `[gpnstar-opt] KV route=dedup|unifiedkv|stock
pairs=<n> rejects=<k> shape=<B>x<L> first_ms=<ms> reason=kvcheck|small_batch|memory|none compile=on|off` (`first_ms`: that first forward's wall time,
the layer-0 check and cache builds inside; `shape=all` once at exit). Every route gives the exact output at exit 0, never a partial
activation: (1) small batches — below 512 target tokens per forward (B × L) the reduced GEMM costs more than it saves, so the shape takes the unified-K/V
route by rule (`reason=small_batch`); (2) memory — see below (`route=stock reason=memory`);
(3) the layer-0 K/V check (`reason=kvcheck`) — the first forward of every new (reduced rows, stock rows) pair computes the layer-0 clade K/V
projection twice with stock's weights (reduced rows plus gather; full stock-shaped matrix, in batch slices) and requires `torch.equal`; on a mismatch that
forward uses stock's projections in every layer and the shape tries the next row padding, then settles, with a `RuntimeWarning`, on the unified-K/V route
or, rejected again, on stock's projections (`pairs` = pairs checked so far, `rejects` = mismatches). The check is kit-against-kit at layer 0: it guards the
exactness argument's one assumption — the GEMM's row-position independence on this card, stack and precision — not a stock comparison.

Memory at large batches. The kit holds less device memory than stock at every batch from 64 up, so the de-dup route stays engaged wherever stock
itself fits: per layer the column attention holds one K-sized tensor at a time (the gathered Kᵀ, released before V is built; `colattn`) where stock's
SDPA-math path transiently holds four to five (K, V, the scaled key, its transposed copy, V's copy); in the source stage the singleton-embedding gather is
released before the clade pooling runs, the multi-clade embeddings are written in place (no list plus stacked copy), and the stage's one long-lived tensor
is allocated first so the caching allocator reuses the same blocks forward after forward. On one 80 GB H100 at 128 bp, both under PyTorch's default
allocator layout, stock `--tf32` runs up to B = 992 windows per forward and is out of memory at B = 1100; the kit runs `route=dedup` at every one of those
batches at 55-60 % of stock's peak and on to B = 1300, beyond which its reserved cache reaches the card. The reduced route is still decided per batch shape
before it allocates anything, from the shape's sizes, the route's own per-layer footprint (`colattn`: 1.25 K-sized tensors; the stock layout: 5) and the
memory obtainable once the source stage is done: a shape that does not fit runs stock's own K/V projections instead — stock speed and memory for that
shape, named on the KV line as `route=stock reason=memory` — and the output is the stock computation bit for bit either way. Should a forward run out of
memory regardless, the levers release their buffers and rerun it once on the stock projections; only a batch that does not fit the card for stock itself
fails, and says so (`[gpnstar-opt] ERROR: out of device memory at batch shape <B>x<L> on the STOCK projections …`).

## Every run

- No stock exceptions: the kit runs `gpn` 0.9.0 as released and patches nothing in upstream's files.
- Refused by name before any window is scored (`[gpnstar-opt] NOT ACTIVE: <reason>`, exit 3): no CUDA device; `gpn` absent or not the pinned stock (installed
  files compared with the archive in `stock/`); `transformers` off upstream's pin; the kit's pins disagreeing with its own model table; a checkpoint file off its
  digest; a model still on the CPU when the levers engage; a batch that breaks the calling contract; a switch value other than `exact` / `off`. Named on the
  ACTIVE line, not refused: torch, numpy, huggingface_hub, safetensors or accelerate off their pins (`stack=drift(…)`), a card outside the listed classes
  (`card=untested(…)`), an unpinned `--model-path` (`weights=unpinned(…)`).
- With upstream's `--torch-compile` the levers are already attached when the tool compiles its wrapper; the patched model's forward and the kit's hooks are
  excluded from dynamo (`torch.compiler.disable`) and run eager in one piece — no graph breaks inside, no recompiles, none of the model body's compilation
  time — while the rest of upstream's wrapper compiles around them; the KV line says `compile=on`, and one `[gpnstar-opt] COMPILE levers=eager(kit-disabled)
  rest=compiled|eager` line is printed the first time a forward is entered from compiled code (a kit-applied model wrapped in `torch.compile` by a library
  caller likewise runs eager, `rest=eager`). transformers turns TF32 on with `--torch-compile`; stock and kit are then both in TF32 matmuls, bitwise to each other.
- Untried, stated once: windows above 512 bp, cards outside the listed classes, other torch / cuBLAS builds (routes keep them exact); at batch 1 the gain depends on the host CPU.
- The accelerator package (`gpnstar_opt.accel`) also contains a whole-forward CUDA-graph runner and a static-capacity de-duplication variant for fixed-shape
  serving loops; the switch does not engage them and they are not part of the mode's contract.

## Switch

- `GPNSTAR_OPT=exact` per process in front of the unchanged command, or `gpnstar_opt.enable()` from code before the model is built, or `gpnstar_opt.apply(model)`
  on a model already on the GPU. `GPNSTAR_OPT` unset or `off`: stock, nothing of the kit loaded. Any other value: `NOT ACTIVE: unknown GPNSTAR_OPT=…`, exit 3 at
  the first `gpn.star.inference` import. `gpnstar_opt.disable()` withdraws the wrapper hooks and the kit's reporters (`[gpnstar-opt] REMOVED … levers=kept`): models
  built afterwards are stock; a model the levers were applied to keeps them.
- No kit variable, flag or file changes lever behaviour: there is one mode.
