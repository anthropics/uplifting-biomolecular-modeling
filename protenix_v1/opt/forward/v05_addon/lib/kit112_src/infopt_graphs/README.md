# infopt_graphs — CUDA-graph capture for Protenix inference

What the library provides to this kit: lever `sg` (the diffusion denoiser step captured once per static signature and replayed for the
remaining steps), lever `sampler_prep` (the graphed loop's host path), and the stream-correct rebuild of protenix's fast LayerNorm extension
that capture requires. `ptxfpf/levers_ptx1.py` installs it through `infopt_graphs.protenix.install(model, sampler=True, trunk=False, ...)`.

| module | what |
|---|---|
| `core.py` | `GraphedFunction` (signature-keyed capture cache, static input buffers allocated with the example's strides, shared memory pools per family, eager fallback by name), `StaticGraph` (caller-owned buffers for in-place state loops), `signature_of` / `static_like` / `copy_into`. The sync census and RNG guard used at warm-up are the shared core's (`opt_core.tools.graph_audit`). |
| `protenix/graphed.py` | `GraphedDenoiseLoop`: `sample_diffusion` with one graph per (batch shape, samples per chunk, N_atom, N_token, dtype, gamma pattern); `EmptyCacheGuard` (defers `torch.cuda.empty_cache()` while more than one sampler graph is live); `install` / `uninstall` / `summary`. Trunk capture (`trunk=True`: PairformerStack per recycle) is a library option no kit mode turns on. |
| `protenix/sampler_prep.py` | lever `sampler_prep`: pinned non-blocking rotation upload, shape-keyed cache with on-device comparison of integer / mask feature values, one warm-up pass, the poison self-test once per process, pool chaining between consecutive captures, vectorised step scalars. `PrepConfig`, `report()`. |
| `protenix/fastln_stream.py` | rebuilds protenix's `fast_layer_norm` extension with `at::cuda::getCurrentCUDAStream()` at every launch site (the stock extension launches on the legacy default stream, which a graph does not record), checks it bit-equal to the original, swaps the module. Build directory: `$TORCH_EXTENSIONS_DIR/fastln_stream` (else torch's extension cache). `lib/fastln_prebuilt.py` loads a prebuilt .so for the pinned stack first. |
| `tests/` | GPU tests of capture / replay / parity and of the cuEquivariance / fast-LN ops under side streams and capture. |

## Fidelity contract

A CUDA graph replays the recorded eager kernel sequence on static buffers, so the graphed step computes what the eager step computes when:
(1) no random number is consumed inside the captured region — every draw (rotation, translation, noise) is made OUTSIDE the graph, per
step, with the stock calls in the stock order, and copied into static buffers; the RNG guard refuses to capture a region that advanced any
generator; (2) no host synchronisation or host branch on device data happens inside — the sync census attributes every sync of the warm-up
passes, and a signature whose capture fails runs the same step body eagerly, recorded in `summary()` (never a silent change of results);
(3) shapes are static — one graph per exact signature, nothing padded; (4) the autocast weight-cast cache is disabled inside the captured
region so every cast is a kernel in the graph. Step scalars (t_hat, delta noise level, dt) are computed from the schedule with the stock
tensor ops; the gamma decision stays a host branch and its 0/1 pattern is part of the signature. Numerics: the eager kernels, replayed —
identical to the eager loop up to GPU run-to-run nondeterminism (byte-identical under the kit's deterministic recipe, `--det 1`).

## Knobs the code reads

`install(model, sampler=, trunk=, pool="private"|"shared", fastln_stream_fix=, sampler_prep=, ...)`; `GraphedDenoiseLoop.max_entries` /
`max_pool_bytes` / `max_tokens` (cache bounds by count and bytes; items above `max_tokens` run the stock sampler); `GraphedDenoiseLoop.biascache`
(the DiT hoist, `lib/kit112_src/dit_hoist.py`, bound by the kit when lever `hoist` is on — the graph cache is then clamped to one entry);
`TORCH_EXTENSIONS_DIR` (fast-LN build directory); `LAYERNORM_TYPE` (read by protenix; the rebuild is needed only for `fast_layernorm`, the
default). Memory: each cached graph holds its private pool plus static inputs (and the hoist's buffers when bound).
