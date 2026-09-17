# kernels/apb — design note (attention with a pair bias: provider face and carried members)

## Op and face

Three boundaries (each row's `rows` entry in `APB_CELLS.json` says which it serves):
`core` — `o[s,n,h,:] = Σ_m softmax_m( scale·q[s,n,h,:]·k[s,m,h,:] + bias[h,n,m] (+ key mask) ) v[s,m,h,:]`, optionally `o *= sigmoid(gate)`; the
bias is SHARED by the S samples (diffusion samples, MSA rows) or given per sample; S = 1 is the pairformer's single attention.
`producer` — bias planes = `Linear_{c_z→H}(LayerNorm(z))` over the N² pair rows, written head-major `[H, N, ld]`.  `module` — the whole
AttentionPairBias module (LN(s), q/k/v/g projections, producer, core, gate, output projection).  `select(cc, dtype, cell, n_tokens, word=…)`
is pure over the table; serving imports torch inside the call; refusals by name with `.fallback` (stock rows `sdpa[:auto|cudnn|efficient|
flash|math]`, `sdpa_upcast`, `ds4sci` (capture-unsafe), `cueq_apb`, `torch_module`, `naive`, … — named, nothing carried).  Rows: `apb_attn`,
`fpf_apb` (`:fp16`), `fpf_atom`, `dit_exact`, `atom_exact`, `dit_fast`, `l3a`, `dtk_loop`, `sba` (`:tf32|ieee|tf32x3`), `ef2_pairbias`,
`composed`, `ln_proj`, `fpf_pf_bias`, `lnl_ln_linear`, `dtk_window`.

## Members

* **`fpf_apb` / `fpf_atom` / `fpf_pf_bias`** (`fpf_apb/`, carried package) — `apb_triton.apb_views` / `dit_apb`: flash attention with an additive
  pair bias shared across the sample batch (Triton).  q/k/v/g addressed as `[S, N, H, D]` through explicit strides (column blocks of one fused
  q|k|v|g GEMM output or per-projection views; no copies); bias `[H, N, N]` with any row pitch ≥ N (fp32 or bf16) read per (q-tile, k-tile) and
  shared by the S programs of that tile through L2 (grid x = sample index fastest, so the S CTAs of one (q-tile, head) are co-scheduled; on
  sm_90 with Triton ≥ 3.3 the bias tile comes through a host-side TMA tensor descriptor — a named cell variant, plain loads otherwise); head dim
  as D1 + D2 powers of two (48 = 32 + 16: no padded MMA columns); fp32 accumulation and softmax statistics, exp2 with scale·log2e folded; MMA
  operands = the 16-bit input dtype, or for fp32 inputs the word `tf32` | `tf32x3` | `ieee` (`opd='fp16'` = the precision cell: fp32 in, fp16
  operands).  `atom_triton.atom_apb`: the atom transformer's local attention — query block t = rows 32t…32t+31, keys [32t − 48, 32t + 80), one
  launch per call with the windows addressed in-kernel from N_atom (no padded / unfolded copies, no mask tensor), the bias `[H, n_blocks, 32, 128]`
  read once per (block, head) program, operands `tf32rn` (RNE to TF32, the cuBLAS TF32 class) | `tf32x3` | `tf32` | `bf16`.  `pf_triton.pf_bias`:
  the producer in ONE pass over z (fp32 statistics, normalized row cast to bf16 × bf16 weight with fp32 accumulation, bf16 `[H, N, N]` output with
  a row pitch rounded up to 8 elements so the core's TMA bias path applies at any N).  All tolerance class (reduction order differs from the
  cuBLAS / cutlass fp32 statements).  `loadcheck.py` + `VECTORS.json`: integer-hash-generated inputs, sha256 of output bytes per shipped cell
  keyed by (cc, Triton version).  D ∈ {24, 32, 48, 64}; cc ≥ 8.0 (cc 8.0 takes the non-TMA cells).
* **`dit_exact`** (`dit_exact/`; EXACT vs `sdpa_upcast`) — the diffusion transformer's pair-bias attention (fp32, head dim 48, scale 1.0, bias
  `[1|B, H, N, N]` broadcast over the samples) as a prebuilt sm_90a CUDA kernel whose output is bit-identical to torch's memory-efficient SDPA
  kernel for float32 (`AttentionKernel<float, Sm80, aligned, 64, 64, 64>`).  *How*: the per-element arithmetic follows that kernel — 3×TF32 operand
  split and mma pass order, tf32 tensor-core k-group chains with the same staged-accumulation grouping, fp32 bias add, FMUL by log2e, FADD +
  ex2.approx (no FTZ), the online-softmax rescale and row-sum orders, rcp.rn epilogue — while the work decomposition is its own: wgmma for both
  GEMMs (wgmma m64nNk8 tf32 gives the bits of mma.m16n8k8 tf32 on identical operands), the three accumulation chains issued as one asynchronous
  batch into three register tiles, register-resident P as the A operand of P·V (k-permutation inside every 8-key group, V rows stored to match),
  K / Vᵀ split to tf32 pairs once per CTA into the canonical core-matrix shared-memory layout, one producer warpgroup (cp.async + operand split +
  bias tiles) feeding two consumer warpgroups through a two-stage named-barrier pipeline, `setmaxnreg` rebalancing, persistent CTAs.  Envelope:
  q, k, v CUDA fp32 `[B, H, N, 48]` (same N), last-dim contiguous, row strides multiples of 4 floats, 16-byte aligned; bias fp32 last-dim stride 1
  (any row pitch).  **Loading**: `apply()` loads `prebuilt/<torch>-cu<cuda>-sm<cc>/dit_attn_exact.so`, checks `manifest.json` (torch, CUDA, arch,
  sha256 of the `.so` and of `csrc/`), runs the load-time check (three real-shaped cases incl. a partial key block and a padded-pitch bias view:
  `torch.equal` against torch's SDPA in this process + output digests against the manifest) and installs the route; a failure raises
  `RuntimeError("dit_attn_exact: <reason>")` and the caller refuses by name; calls outside the envelope take the original statement, counted by
  reason in `report()`.  `build_prebuilt.py`: nvcc `sm_90a` against the installed torch headers (PROVENANCE.md per key).
* **`atom_exact`** (`atom_exact/`; EXACT) — the windowed atom attention (32-query × 128-key windows at stride 32, zero padding, pad mask + pair bias,
  fp32) in ONE Triton launch reproducing the bits of the chunked math-SDPA statement: S = q·kᵀ as cuBLAS runs it — cutlass s1688 TF32 (operands
  RNE-rounded to tf32, sequential-K fp32 accumulation from +0) or, when cuBLAS's heuristic picks it for a remainder chunk, the FFMA fp32 kernel
  (acc = fma(q_d, k_d, acc), d ascending) —, S += (mask + bias), P = the warp softmax's order (row max; `expf(x − max)`; per-lane sums of the 4
  lane-strided elements from 0; xor butterfly 16, 8, 4, 2, 1; IEEE division), O = P·v (TF32, K = 128 sequential); output written directly in the
  `[.., N, H, 32]` layout the caller transposes to.  Which GEMM numerics cuBLAS uses for a given chunk batch count is determined per process
  before the first launch (`torch.matmul` on the exact shapes compared with both variants); a count matching neither, or a digest mismatch on
  the `vectors.json` load-check cases, refuses by name and the statement serves.
* **`dit_fast`** (`ditfast/kernels.py`, `atom_kernels.py`, carried) — the fused diffusion-transformer block's Triton ROW kernels: token stream
  `adaln`, `gate`, `swiglu`, `resgate`, `resgate_adaln`; atom stream 2-D row tiles `adaln2`, `resgate_adaln2`, `resgate`, `gate2d`, `swiglu2d`
  (fp32 statistics and math, activations out in the GEMM dtype, conditioning operands read with a row-modulo `Ns` for sample-deduplicated
  conditioning).  The block schedules that walk a module tree stay in the kits (`dit_block_rows()` hands them these kernels).
* **`l3a`** (`l3a/`, carried) — `fab_batched.flash_bias_attn_batched`: ONE launch for S samples sharing one bias — op-for-op the per-sample kernel
  `kernels/dtk_kernels.flash_bias_attn` with a sample grid axis (`stage_c=False`: the same bytes as the `dtk_loop` row; `stage_c=True` adds
  aligned-pitch / peeled-loop variants whose bits differ when a tail tile runs); `bias_layout.PairBiasLayout`: one padded `[L, H, N, Np]` bias
  buffer with aligned rows and a power-of-two stride skew.  Flag `KOPT_ATTN_L3A`.
* **`ef2_pairbias`** (`ef2/ef2_pairbias_attn.py`, carried; imports the ESM-family fork → refused by name elsewhere) — the structure module's
  pair-bias attention inside diffusion sampling: variant `hoist` (EXACT: the per-block pair bias `Linear(LayerNorm(z))` computed once per block per
  sampling call by the stock modules and reused, everything else verbatim) and `fused` (hoist + ONE Triton flash-style core per block: online
  softmax over key tiles, fp32 statistics and accumulation, split-mantissa 3×TF32 tensor-core products for fp32 operands, bias + key mask + gate
  fused, no N × N logits in memory, 64-bit addressing; tolerance class).  Named fallbacks `fallback_grad` / `fallback_shape` / `stock_cueq`.
* **`apb_attn`, `dtk_loop`, `dtk_window`, `sba`, `ln_proj`, `lnl_ln_linear`, `composed`** — module kernels served in place (`kernels/apb_attn.py`
  through `attn.apb_core`, `kernels/dtk_kernels.py`, `attn.shared_bias_attn`, `kernels/ln_proj.py`, `kernels/lnl_fused.py`); tolerance class.

## Known limits

The two exact members are exact against ONE reference kernel each (`dit_exact`: the memory-efficient SDPA float32 kernel; `atom_exact`: the
math-SDPA + cuBLAS TF32/FFMA statement of the process) and refuse by name where that reference is not what the stack runs; `dit_exact` is
sm_90a-only with prebuilts per (torch, CUDA) key.  Core rows serve forward only; `ds4sci` and the stock `fast` paths that synchronise are
capture-unsafe and named so.  Carried member files are digest-pinned in `kernels/META/apb.json` — replaced with their records, never edited.
