# kernels/transition — design note (provider face and carried members)

## Op and face

Trunk transition of the AF3-family pair / single / template / MSA stacks and the ESM-family pair transition:
`y = W_o( silu(W_a n) * (W_b n) )`, `n = LayerNorm(x)` over the last dimension of a `[rows, c]` tensor, hidden = factor·c; optionally
`y *= mask[:, None]` (row mask) and `y = x + y` (residual), both folded into the kernel epilogue; backward = dX for FROZEN weights (no dW /
dgamma / dbeta) on the rows that carry one.  `select(cc, dtype, cell, n_tokens, word=…)` is pure (standard library + `TRANSITION_CELLS.json`,
whose `rows` block states each row's envelope, numerics class, capture safety and named fallback); the serving entries import torch inside the
call; a row that cannot serve is refused BY NAME with its `.fallback` (stock rows `torch_swiglu`, `engine_module`, `compile` are named, never
constructed here).  Row words: `v2`, `v1` (`v1:lnfused`, `v1:liger`), `pf` (`pf:fpf`, `pf:lnl`), `lnl`, `af3_fused`, `flash_sm90a`, `esm_t15`,
`esm_t16`, `esm_kd3` (`esm_kd3:lean`), `esm_t16_kd3`, `esm_t15_kd3`, `esm_fused_exact`, `rowpair` (the row-block schedule of `mem.rowpair` over
any forward row).

## Members

* **`v2`, `v1`, `pf`, `lnl`** — the module kernels `kernels/fpf_transition_v2`, `kernels/fpf_transition` (through `attn.pair_fused.transition`)
  and `kernels/lnl_fused`: the whole op in ONE Triton kernel per call (LayerNorm in the prologue or given, a|b GEMM chain, SiLU·b, W_o, mask and
  residual in the epilogue; int64 row offsets).  Numerics: the statement's bf16 rounding points with fp32 accumulation in the kernel's own order
  — tolerance class (`v1:liger` = the Liger-Kernel SiLU·b epilogue: silu of fp32 a times fp32 b, rounded once).
* **`af3_fused`** (`af3/af3_fused.py`, carried module) — LayerNorm + SwiGLU + W_o in one Triton kernel, non-power-of-two `c` handled by padding
  the register tile (c ∈ {64, 128, 256, 384}); tolerance class.
* **`flash_sm90a`** (`flash_sm90a/`, sealed unit `fpf_flash_transition`; cell c = 256, hidden = 1024, bf16, cc 9.0) — ONE sm_90a CUDA C++ kernel on
  CUTLASS/CuTe (`csrc/flash_transition_sm90.cu`).  Per 128-row tile the LayerNorm output stays in shared memory and the hidden dimension is
  processed in chunks of 32: GEMM1 (wgmma SS, m64n64k16 chain over K = 256) → SwiGLU in registers → GEMM2 (wgmma RS, m64n256k16; the h chunk is the
  register A operand after an accumulator→A-fragment relayout) accumulating `out[128, 256]` in fp32 registers; weight chunks stream through TMA
  rings ([Wa_j; Wb_j] 64×256 × 3 slots, Wo chunk 256×32 × 4 slots) multicast to a 2-CTA cluster; 1 producer warpgroup + 2 consumer warpgroups (64
  rows each); residual add + bf16 stores through a shared-memory-staged, 16-byte coalesced epilogue.  Variant LN=0 consumes the module's own
  LayerNorm output; variant LN=1 computes the LayerNorm in the prologue with the arithmetic of the engine's `fast_layernorm` extension kernel
  (per-lane sequential Welford, xor-butterfly merge, rcp / rsqrt approximations, flush-to-zero, bf16-cast weight / bias) — the same bits as that
  module (route `module-ln` when a module's LayerNorm is not that class).  **Numerics EXACT**: the statement's bf16-autocast chain bit for bit —
  bf16 × bf16 products accumulated in fp32 in ascending K (equal to the cuBLAS result at these shapes), a = bf16(acc), silu(a) → bf16,
  b·silu(a) → bf16, u = bf16(acc2), out = bf16(fp32(z) + fp32(u)); SiLU of a bf16 value = a·rcp(1 + 2^(−a·log2e)) with ex2.approx / rcp.approx,
  compared with torch's `F.silu` on all 65536 bf16 inputs at load (`silu_table()`).  **Loading**: `install()` picks
  `prebuilt/<torch>-cu<cuda>/flash_transition_sm90a.so`, checks `manifest.json` (torch, CUDA, python tag, sha256 of the `.so` and of
  `csrc/*.cu`), requires cc 9.0 and ≥ `smem_bytes()` opt-in shared memory, loads the extension, runs the load-time checks on the GPU (SiLU table;
  `torch.equal` against the statement on synthetic tiles of 8 / 333 / 65536 rows, both residual forms; LN variant: `ln_rows()` == the module),
  then rebinds `fpf_transition.transition.fn_residual / fn` under both import names (originals kept as `CORE_FN_*`); any failure raises
  `RuntimeError("flash_transition: …")` and the caller refuses by name.  Binaries for other interpreters live in `flash_prebuilt/<ABI key>/` with
  `flash_prebuilt/manifest.json`, built by `build_flash_prebuilt.py` with the sealed recipe (`csrc/build.py`: nvcc `sm_90a` flags, CUTLASS 4.2.0
  headers, a neutral staging path); the sealed directory is never written.  Envelope: CUDA bf16, `c_in == 256`, hidden 1024 (the kernel needs
  hidden % 64 == 0), eval mode, bias-free projections, sm_90; anything else takes route `core` (the function bound before `install()`), counted by
  reason.  Capture-safe (no host sync, packed weights cached on the module).
* **`esm_t15`** (`esm/ef2_pair_v2.py`, carried; imports the ESM-family `transformers` fork at module level → refused by name where that fork is
  absent) — ONE Triton kernel per call: in-kernel fp32 LayerNorm statistics (the x tile held as 4 bf16 [128, 64] chunks, x_hat rounded once to
  bf16), packed a|b GEMM (interleaved [Wa|Wb] columns → one N = 64 wgmma chain per hidden chunk, A read once), h = bf16(silu(a)·b), fp32
  accumulation of h @ W3ᵀ over ascending hidden chunks, out = bf16(fp32(x) + acc).  C = 256, hidden 1024.  Tolerance class.
* **`esm_t16`** (`esm/ef2_t16_transition.py` + loader `esm/ef2_t16_nvjit.py` + `esm/ef2_t16/` = kernel source `.cuh`, sm_90a cubin, manifest,
  PROVENANCE) — ONE persistent, warp-specialised sm_90a CUDA C++ / CuTe kernel: CTA = 384 threads = 3 warpgroups, one CTA per SM, persistent over
  128-row tiles.  WG0 producer: warp 0 streams the weight ring (W1_j, W2_j, W3_j per 64-unit hidden chunk: 32 KB TMA boxes into a 5-slot ring with
  full/empty mbarriers), warps 1/2 load the activation half-tiles (64 rows × 256, TMA, SW128) of WG1 / WG2.  WG1/WG2 consumers, 64 rows each
  (register rebalance producer↔consumers): LayerNorm in registers (fp32 two-pass statistics, one bf16 rounding of x_hat) written back IN PLACE
  over the x half-tile, then per chunk a = x_hat·W1_jᵀ, b = x_hat·W2_jᵀ (wgmma SS m64n64k16, two commit groups so silu(a) overlaps the b chain),
  h = bf16(silu(a)·b) built directly in the wgmma A-fragment layout (the m64n64 C fragment is the m64k64 A fragment), acc += h·W3_jᵀ (wgmma RS
  m64n256k16, left in flight into the next chunk); epilogue out = bf16(fp32(x) + acc) with the residual re-read from L2, direct 32-bit stores.
  d = 256, hidden 1024, bf16.  Numerics as `esm_t15` (fp32 statistics, bf16 operands, fp32 accumulation, h and out rounded once; sigmoid =
  1/(1 + 2^(−a·log2e)) with ex2.approx + rcp.approx; no TF32) — within one bf16 ulp of the K-D3 forward it replaces, tolerance class.
  **Loading**: the SHIPPED cubin only, through the CUDA driver API (`cuda-bindings`) on torch's current stream; served when the manifest's
  `source_key` (sha256 of the `.cuh` text + compile options, computed without a toolchain) equals this tree's, the file's sha256 equals
  `cubin_sha256`, the manifest records a spill-free build, the device reports zero local memory and the manifest's register count after
  `cuModuleLoadData`, and the install canary passes (back-to-back launches bytewise identical and within tolerance of the fp32 statement); otherwise
  `NvjitUnavailable` by name and the lever STEPS ASIDE by name (`esm_kd3`'s own forward serves).  No compiler at run time; cc 9.0 only.
* **`esm_kd3`** (`esm/ef2_autograd_kernels.py`, carried; same import rule) — the differentiable `TransitionRefround` (K-D3 fast | lean): forward
  AND dX backward for frozen weights; the backward recomputes the LayerNorm (+ its fp32 statistics) and the W12 GEMM from the block input, so an
  out-only forward kernel plugs in unchanged — rows `esm_t16_kd3` / `esm_t15_kd3` = lean with the `esm_t16` / `esm_t15` forward through the carried
  hook.  Reference numerics class (bf16 storage, fp32 statistics, fp32 GEMM accumulation).
* **`esm_fused_exact`** (`esm_fused/__init__.py`) — the ESM-family fork's FUSED INFERENCE statement in ONE Triton kernel, bit for bit: per-row
  statistics as the fork's `_ln_stats_kernel` stores them (bf16 tree mean, fp32 tree variance, both rounded to bf16), x_hat stepwise in bf16
  (four roundings), a|b = x_hat @ W12 with fp32 accumulation over K = 256 in four ascending 64-chunks, h = bf16(silu(a)·b), y = bf16(h @ W3ᵀ)
  accumulated in the chained K order cuBLAS's non-split kernels use at these shapes, out = bf16(x + y) ROUND-THEN-ADD.  One program per BM-row
  tile; the prologue's statistics tree and x_hat chain are the ESM-family kits' own Triton device functions carried byte for byte (their
  association is fixed by code, so the bits hold at any BM / warp count); the a|b columns use the column-interleaved [Wa|Wb] packing (each output
  element keeps its own K chain).  Envelope: C = 256, H % (2·BH) == 0, bf16 rows, LayerNorm with affine, the cc 9.0 launch row (227 KB
  shared-memory class); forward only; deterministic (no atomics, no split-K); `out` may alias `x`.

## Known limits

Forward rows produce no gradients (`esm_kd3` family excepted); every exact statement is exact against ITS reference chain on the stack the
cell names (another GEMM library or GPU class may produce different reference bytes — the cells record where identity holds); `flash_sm90a`
and `esm_t16` are sm_90a-only and refuse / step aside by name elsewhere; the ESM rows need the ESM-family fork importable.  All carried member
files are digest-pinned in `kernels/META/transition.json` — they are replaced together with their records, never edited in place.
