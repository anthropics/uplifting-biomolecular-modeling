# FlashPairformer — triangle attention (triattn_native)

`kernels/triattn/triattn_native/` — the FlashPairformer triangle-attention forward (inference) for the AF3-family pairformer, the
CUDA-native member of the FlashPairformer kernel family, served as one routed package: a router, Gluon and Triton
kernels and four CUDA C++ extensions (three sm_90a, one sm_80) carried under `pkg/v11/triattn_pkg/` and reached only through the
face `opt_core.kernels.triattn.triattn_native` (`install()`, `warm()`, `triangle_attention()`, `route()`, `admits()`; every refusal is
`Unavailable`, whose `.kind` is the reason word).

## Op and calling convention

`out = triangle_attention(q, k, v, bias, mask=None, scale=None)` computes, per pair row `i`,
`out[b,i,h,s,:] = softmax_t(scale·q[b,i,h,s,:]·k[b,i,h,t,:] + bias[b,0,h,s,t] (−inf where mask[b,i,0,0,t] == 0)) @ v[b,i,h,t,:]`.

* `q, k, v`: `[B, N, H, S, D]` or `[N, H, S, D]`, bf16 or fp16, CUDA; contiguous, or strided views with `stride(-1) == 1` and the
  other strides multiples of 8 elements — an ending-node call is the same op on transposed views of the pair tensor, read in place.
  Any number of pair rows `N` (incl. `N ≠ S`), any `H`, `B ≥ 1`.
* `bias`: `[B, 1, H, S_q, S_kv]` (or 4-D), fp32 or the 16-bit dtype of `q` — ONE pair bias shared by all `N` pair rows.  Every member
  exploits this: a bias tile is staged or loaded once and reused by all the pair rows a CTA owns.
* `mask`: `[B, N, 1, 1, S_kv]` bool, True = attend (one padding length per batch element, per-row ragged lengths, interior holes), or
  `None`.  A pair row with no attendable key yields the uniform mean of `v` (the convention of cuequivariance_ops_torch's op).
* `scale`: default `D ** -0.5`.  Returns `[B, N, H, S_q, D]`, contiguous, in `q`'s dtype.
* Refused by name before any work (the payload's `Unsupported`, a `NotImplementedError`, surfaced as `Unavailable("unsupported:…")`):
  fp32 q/k/v, `D ∉ {16, 32, 64, 128}`, per-row biases, per-query masks, non-CUDA tensors, `S_q ≠ S_kv` outside the Triton member's
  reach, development knobs (`TRIATTN_M1_FLAGS`, `MW_GEOM`, `MW_TIMELINE`, `PTX_TRIATTN_CONFIG`) set in the environment, a Gluon route on a
  Triton without `triton.experimental.gluon` (`needs_triton_gluon`).  Nothing is substituted silently.

## Routing (`dispatch/candidate.py:best`, keyed by compute capability; `S` = `S_q` = `S_kv`)

| cc | operands | S | route → member (source dir, extension) |
|---|---|---|---|
| 9.0 | bf16, D 32 | S < 512, contiguous | `k13` → persistent warp-specialized Gluon kernel (`dispatch/kernels/k13.py`) |
| 9.0 | bf16, D 32 | S < 512, strided views | `cuda` → `triattn_cuda` (`cuda/`, `triattn_sm90_ext`) |
| 9.0 | bf16, D 32 | 512 ≤ S ≤ 3072, and S > 4096 | `cuda_b` → `triattn_m1` (`cuda_b/`, `triattn_m1_ext`) |
| 9.0 | bf16, D 32 | 3072 < S ≤ 4096 | `cuda_c` → `triattn_mw`, geometry 3x4 (`cuda_c/`, `triattn_mw_ext_g3x4`) |
| 9.0 | fp16, or D 16 | S ≤ 640 → `k13`; above → `tri` | `tri` = k12 (Gluon) here (`triton/`) |
| 9.0 | D ∈ {64, 128}, or S_q ≠ S_kv | any | `tri` = k10 (plain Triton) |
| 8.0 | bf16, D ∈ {16, 32, 64} | any, either layout | `cuda_80` → `triattn_sm80` (`cuda_80/`, `triattn_sm80_ext`) |
| 8.0 otherwise; any other cc | anything servable | any | `tri` (k10 / k12) |

Each CUDA member is built for exactly one architecture and is never routed to another device class.

## Members

* **`cuda`** (sm_90a; CUDA C++ on CUTLASS/CuTe; wgmma + TMA): warp-specialized, 1 TMA producer warp + 2 consumer warpgroups of 64 query
  rows in ping-pong; CTA = (b, h, 128-query tile, R consecutive pair rows); K/V tiles TMA-multicast over a cluster of CTAs sharing the
  same pair rows; the bias tile of a key block held in registers and reused by the R rows (bias staged to bf16 tiles); row sums on the
  tensor core (P × ones); per-row mask bit table in shared memory (capacity S ≤ 4096).  Grid x = q-tiles (cluster dim), y = row groups,
  z = B·H.
* **`cuda_b` = `triattn_m1`** (sm_90a; CUDA C++ on CUTLASS/CuTe; wgmma + TMA): CTA = (b, h, one 128-query tile, R = 3 consecutive pair
  rows), 512 threads = 1 producer warpgroup (one TMA thread) + 3 consumer warpgroups, warpgroup w owning pair row i0+w for all 128 queries
  (two m64 halves); K/V tiles of 128 keys stream through a 2-tile ring per row.  The 128 × 128 fp32 bias tile (÷ scale, shared by the 3
  rows) is staged once per call in MMA-fragment order, so a chunk's S accumulator is initialised from shared memory (4 × LDS.128 per
  thread) and the QK wgmma (m64n32k16) accumulates on top — no per-logit bias instruction.  Per warpgroup the keys form one stream of
  64 q × 32 k chunks: p = 2^(S·c + nm) with a FIXED per-row shift nm seeded from the first chunk column (max-free: no running max, no
  rescale), bf16 P, PV wgmma over [V | 1] (m64n40k16; the ones columns accumulate the row sum) issued one chunk late.  Masks: keys dead
  in every row of the batch element become −inf columns of the staged bias; the CTA streams the union of its rows' live 32-key column
  ranges; a row whose mask differs from the batch OR consumes only its own tiles and applies 32-bit mask words only on tiles that can
  hold a boundary of its range (a ragged row: on all) — a STEADY (mask-free) and a GENERAL body instantiation; a CTA tile whose rows are
  all fully masked skips the key loop and writes the uniform rows directly.  Tiles the max-free pass cannot finish (non-finite value or
  zero row sum — logit swings beyond ~2^100 within a row — or no finite logit in a row's first chunk) go on a per-call fix list and are
  recomputed by the SAFE (running-max) instantiation of the same routine; census `triattn_m1.FALLBACKS["fix_tiles"]`.
* **`cuda_c` = `triattn_mw`, geometry 3x4** (sm_90a; CUDA C++ with inline PTX; `mma.sync` + TMA): "many-warp" — CTA = 12 warps = 3 pair
  rows × 4 warps × 32 queries.  Per 32-key half tile every warp computes in registers S = QKᵀ + bias/scale (`mma.sync.m16n8k16`, the
  pre-scaled fp32 bias tile as the C operand; the bias is staged as [128 q × 32 k] fp32 boxes laid out so one LDS.128 = one accumulator
  fragment), p = ex2(S·scale·log2e − m), O += PV (bf16 P as the A operand; row sums from an all-ones B column).  Max-free softmax: m is a
  per-row integer offset moved only by exact powers of two when the running sum leaves [2^-20, 2^20].  Software-pipelined over half tiles;
  two 4-deep rings ({bias keys 0–31, K of the 3 rows}, {bias keys 32–63, V}), K/V through TMA descriptors, Q in registers, stages
  refilled by whichever warp releases them last (no dedicated producer).  Rows whose mask differs from the batch OR are listed by
  `stage_mask` and recomputed by the general instantiation in a second pass (per (row, 64-key tile) class skip / mixed / full / uniform);
  SAFE fix pass as above.  Refuses S > 4096.
* **`cuda_80` = `triattn_sm80`** (sm_80; CUDA C++, header-only device code, torch-free launch layer): the many-warp formulation on
  `mma.sync.m16n8k16` (bf16 → fp32) + `ldmatrix`, with a cooperative `cp.async.cg` 16-byte multistage ring in place of TMA / mbarrier.
  CTA = R pair rows × BM = 32·QG queries as WR × QG warps, each warp owning 32 queries of RW = R/WR consecutive rows; the R rows share one
  staged bias tile (BM × 64 keys, fp32 ÷ scale, m16n8 fragment order, loaded once per warp as the initial accumulator of its rows).
  Geometry per head dim: D 32 → 2×4 warps × 2 rows (R = 4, 128 q), 3 ring slots × 32 KB; D 16 → 2×4 warps × 1 row (R = 2, 128 q), 4 slots
  × 20 KB, two CTAs per SM; D 64 → 2×2 warps, R = 2 × 64 q, two CTAs per SM, plus a 1×4-warp 128-q form for transposed operands at large
  S.  Max-free softmax (renormalisation window [2^-84, 2^-44]); hot pass then an unconditional SAFE fix-list pass; masks handled inside
  the hot pass (regular / interval rows stream only their live key tiles and mask per key only on a tile holding an interval end, ragged
  rows apply 32-bit mask words per tile, fully-masked rows written in the epilogue); 64-bit addressing; optional per-row log2-sum-exp
  output (`return_lse=True`); census `triattn_sm80.FALLBACKS`.
* **`k13`** (Gluon, sm_90a, Triton ≥ 3.6): persistent warp-specialized kernel — TMA + mbarrier ring, async wgmma, one 8-warp compute
  partition, ROWS pair rows per work item sharing each 16-bit bias tile, one CTA per SM walking work items.  The key mask goes through
  the tensor core: S = QKᵀ + 1 ⊗ m as a second chained wgmma (K = 16), m = 0 for kept keys and −2^30 (bf16) / −2^15 (fp16) for masked
  keys, the m tile built by the producer warpgroup from the raw mask bytes; key tiles past the last kept key are skipped.
  **`tri`** = **k12** (the same design with the batch OR pattern folded into the staged bias and a per-key select only where a row
  differs, `triton/triattn/masking.py`) for square D ∈ {16, 32}, else **k10** (plain Triton, any CUDA GPU: one program = ROWS pair rows ×
  one BLOCK_M query tile, running-max online softmax in registers, source generated per ROWS).  Repeat launches reuse the compiled
  handle (`triton/triattn/launch.py`).

## Numerics

bf16/fp16 tensor-core products with fp32 accumulation; softmax in fp32 in the base-2 domain (scale·log2e folded into the QK scale or the
staged bias); P rounded to the 16-bit dtype for the PV product, and on the CUDA members the row sums come from that same 16-bit P.  Pair
bias: exact fp32 (accumulator initial value / C operand) on `cuda_b`, `cuda_c`, `cuda_80`; staged to 16-bit tiles on `k13`, `cuda` and k12
(exact for a 16-bit-representable bias, otherwise rounded once; k10 keeps fp32 when the conversion is lossy).  The max-free members are
exact for any input because the tiles their fixed-shift pass cannot represent are recomputed by the running-max instantiation.  No atomics
into outputs, fixed reduction order per launch geometry: deterministic run to run; output bytes identical across the prebuilt stacks.
Error class: that of cuequivariance_ops_torch 0.11.1 `triangle_attention` (bf16 output; not bitwise equal to it).

## Loading, byte gate, refusals

* The face imports the payload once under a private module name; the payload's router puts its kernel directories on `sys.path` under
  the top-level names `kernels, triattn, pins, candidate_tri, triattn_cuda, triattn_mw, triattn_m1, triattn_sm80` — a host process that
  already owns one of them is refused (`namespace_collision:<name>`).
* The CUDA members are torch C++ extensions keyed by `torch<torch.__version__>-<SOABI>`: `triattn_pkg/prebuilt/<key>/<ext>.so` +
  `<ext>.json` (build record: toolchain, `-gencode` flags, `so_sha256`, `source_sha256` over that member's `csrc/`, `loadcheck` output
  digests), loaded with `importlib.machinery.ExtensionFileLoader` after the digests are checked against the files (`digest:<ext>:<field>`).
  Inside opt_core `TRIATTN_PKG_PREBUILT=always`: never a JIT build in a serving process — a key or extension without a prebuilt is
  `no_prebuilt:<pkg>@<key>[:<ext>]`.  (Stand-alone, the payload JIT-builds a member with nvcc + ninja, plus CUTLASS headers at
  `$CUTLASS_PATH` for `cuda` / `cuda_b`.)
* Byte gate (`install()` → `loadcheck()`): the payload's own test vectors — inputs regenerated from (seed, shape) by `testvectors/cases.py`
  and checked against recorded input digests, the routed forward run on this device, output bytes compared bitwise with
  `testvectors/<case>.expected.pt` for the cases of the extensions this device executes (`LOADCHECK_CASES`); any difference is
  `loadcheck_failed:<case>`.  Where no extension route applies the gate is a determinism + finiteness self-check of the Triton member.
  A passed verdict is cached under `<MODEL_OPT_JIT_ROOT>[/<MODEL_OPT_STACK_KEY>]/triattn/` (else beside `TRITON_CACHE_DIR`) keyed by
  (payload version, ABI key, device, driver, extension digests, vector-manifest digest); the digests themselves are checked every
  process.  Kits call `warm(shapes)` at activation so the import, the gate and first-launch / Triton JIT costs land outside the first
  model call.
* Version words: `ACTIVE_PKG = PkgVersion("v11")`; `HONOURED` names the earlier payload words the active payload serves and on which device
  classes (`v10` on cc 9.0: the same routes and output bytes, so the word compares equal; refused by name on cc 8.0, where the member
  differs); `SERVED_CC` lists the device classes served at all.  `admits(dtype, head_dim, n, key=…, strided=…, triton_gluon=…, cc=…)`
  restates the routing table for admittance checks without importing the payload.

## Known limits

The sm_90a members and the Gluon kernels need cc 9.0; `cuda_80` is routed on cc 8.0 only (cc 8.6 / 8.9 stay on `tri`).  `cuda` and `cuda_c`
refuse S > 4096 (`cuda_b` serves any S); an fp32 bias below S = 512 on cc 9.0 is rounded to bf16 (routes `k13` / `cuda`).  Memory per call
≈ operands + output (≈ H·N²·(8·D + 4) bytes at N = S with an fp32 bias) plus the staged bias copy and a transient fix list; no kernel-side
size limit on `cuda_80` (64-bit offsets).  CUDA-graph capture is safe after one eager call of the same or larger size on the device (fix
buffers grow only in eager calls; `cuda_80` refuses growth under capture by name).  The first call per (dtype, D, mask kind) compiles the
Triton / Gluon member (seconds).
