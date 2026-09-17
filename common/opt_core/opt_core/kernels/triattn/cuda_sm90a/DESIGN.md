# kernels/triattn/cuda_sm90a — design note

Triangle-attention forward (inference) for sm_90a as one hand-written CUDA kernel (`csrc/triattn_mw.cu`, PTX wrappers in `csrc/mw_ptx.h`;
no CUTLASS / CuTe), shipped as a prebuilt torch extension (`triattn_cuda_sm90`) and served by `attn()`; every refusal is `Refused` (a
`NotImplementedError`) naming the reason — the module never substitutes another kernel.

## Op and calling convention

`out = attn(q, k, v, bias, mask=None, scale=None)` — `softmax_t(scale·q·kᵀ + bias (−inf where mask == 0)) @ v` per pair row.

* `q, k, v`: `[B, N, H, S, D]` bf16 CUDA (rank < 5 → leading singleton dims); D = 32; `N` (pair rows) may differ from `S` (queries = keys).
  Read in place when `stride(-1) == 1`, the other strides are non-negative multiples of 8 elements and the base is 16-byte aligned
  (contiguous tensors, row slices `q[:, i0:i1]`, a fused prologue's strided outputs); anything else is copied once (`COUNTS["copy"]`).
* `bias`: `[B, 1, H, S, S]` fp32 | bf16 | fp16, any strides, shared by the `N` pair rows; staged once per call to an fp32
  `[B, H, ceil128(S), ceil64(S)]` copy pre-divided by `scale`.
* `mask`: `[B, N, 1, 1, S]` bool (True = attend) or None, any pattern; a pair row with no attended key yields the mean of `v` (the
  cuEquivariance convention).  `scale` defaults to `D ** -0.5`.  Returns a new contiguous `[B, N, H, S, D]` bf16 tensor.
* Envelope / refusals: compute capability 9.0 only; D = 32; bf16 operands; 1 ≤ S ≤ 65536; B·H ≤ 65535; `S_q == S_kv`; masks of shape
  `[B, N, 1, 1, S]` only (no per-head / per-query masks); CUDA tensors.

## Implementation

"Many-warp `mma.sync`" formulation.  One CTA = 12 warps = R = 3 consecutive pair rows × 4 warps × 32 queries — a 128-query tile of 3 pair rows
of one (b, h); grid = (q tiles, row groups, B·H) with the q tile fastest so the CTAs sharing K/V rows are co-resident (L2 reuse).

* Per 32-key half tile every warp computes in registers S = QKᵀ + bias/scale with `mma.sync.m16n8k16` (bf16 → fp32), the pre-scaled fp32
  bias tile being the mma C operand: the staged bias arrives as [128 q × 32 k] fp32 boxes laid out so one LDS.128 is exactly one accumulator
  fragment (`stage_bias_kernel`) — no per-logit bias instruction.  Then p = ex2(S·scale·log2e − m), packed to bf16 A fragments; O += P·V, with
  the row sums accumulated by the same mma from an all-ones B column.
* Max-free softmax: m is a per-row INTEGER offset kept SHIFT log2 units above the running log2-sum; it moves only by exact powers of two
  (applied to O and l) when the running sum leaves [2^-(SHIFT+20), 2^-(SHIFT−20)]; nothing per logit tracks a maximum.  Rows not yet seeded take
  m from the first half tile holding a finite logit.  CTA tiles that end non-finite or with a zero sum (logit swings beyond ~2^100 within a row)
  are appended to a per-call fix list and recomputed by the SAFE instantiation of the same kernel (max-tracking: m raised to ⌊max⌋ per row
  before exponentiating), so results are exact for any input; `COUNTS["fix_tiles"]`.
* Staging: two 4-deep shared-memory rings fed by TMA (`cp.async.bulk.tensor` + mbarrier): ring A stages hold {bias keys 0–31, K tiles of the 3
  rows} of a 64-key tile, ring B stages {bias keys 32–63, V tiles}; Q lives in registers as mma A fragments (plain 4-byte loads); after the
  prologue each stage is refilled by whichever warp releases it last — no dedicated producer warp.  The hot block is software-pipelined at
  8-key granularity: exponentials / bf16 pack / row-sum mma / P·V slice of the current half interleaved with the bias load + QKᵀ mma of the next
  half, so tensor pipe, FMA pipe and MUFU all have work in flight.
* Masks: keys attended by no pair row of the batch element (the OR over rows) and keys ≥ S become −inf columns of the staged bias (exact), so
  rows whose mask equals that OR — every row, for padding masks — run the mask-free instantiation.  Rows with their own pattern (interior
  holes, shorter prefixes, fully-masked rows) are listed by `stage_mask` (per-row tables in global memory, hence no S cap from shared memory)
  and recomputed by the general instantiation in a second pass over that list, per (row, 64-key tile) class skip / mixed / full / uniform;
  mixed tiles select −inf per key; `COUNTS["list_rowgroups"]`.
* Deterministic; launches on the current stream with no host synchronisation; capturable in a CUDA graph after the first call of a process
  (that call sizes the per-device int32 tile-list buffer, which is kept).

## Numerics

bf16 tensor-core products with fp32 accumulation; bias added in fp32 (÷ scale in fp32); softmax in fp32, base 2 (`ex2.approx`), with the
power-of-two row offset in place of a running maximum; P rounded to bf16 for P·V and the row sums taken from that same bf16 P; bf16 output.
The numerics class of cuequivariance_ops_torch's `triangle_attention` (bf16 tensor-core attention with fp32 softmax); not bitwise equal to it.

## Loading

`install()` (called by `attn()` on first use; kits call it at activation): stack key = `torch<version>-<SOABI>-sm<cc>`; loads
`prebuilt/<stack key>/triattn_cuda_sm90.so` with `importlib` after checking `prebuilt/<stack key>/manifest.json` against the running
process (sha256 of the binary, torch and CUDA versions, sha256 of every file under `csrc/`), then runs the load-time check: the loaded
kernel recomputes the small deterministic cases of `loadcheck_cases()` and must reproduce `prebuilt/<stack key>/loadcheck.pt` bit for bit
and stay within a fixed bound of an fp32 evaluation.  A device that is not cc 9.0, a missing prebuilt for the stack, a manifest mismatch,
a load error, a missing symbol or a load-check difference each raise `Refused` with the reason.  `install(allow_build=True)` /
`build_prebuilt.py` compile the extension for the running stack with nvcc through `torch.utils.cpp_extension` (`-arch=sm_90a`) and write
the manifest and `loadcheck.pt`; a serving process never builds.  `report()` returns the stack key, route, load time, load-check mode and
`COUNTS`.

## Known limits

Device memory per call besides the output (transient, from the caching allocator on the current stream): the staged bias
B·H·ceil128(S)·ceil64(S)·4 bytes; with a mask, B·N·S + 9·B·N·ceil(S/64) + B·ceil64(S) bytes of tables; the per-device int32 tile list
12·ceil(S/128)·ceil(N/3)·B·H bytes (at least 256 KiB, kept across calls).  No `[N, S, S]` logits are materialised.  cc 9.0 only (sm_90a
instructions: TMA, mbarrier transaction counts); other 9.x / 10.x parts are refused by name, not probed.
