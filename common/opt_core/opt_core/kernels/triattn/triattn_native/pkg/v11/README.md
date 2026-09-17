# triattn_pkg v11 — a triangle-attention forward for H100 and A100, packaged

One Python function that computes the pairformer's triangle attention (AlphaFold-3-family models: Boltz-2, OpenFold3, …) and routes each call to the
fastest measured kernel of this tree for its device, shape, dtype, mask and memory layout.  Everything a consumer needs is inside `pkg/v11/`; nothing
here imports the rest of the repository.  `../v10/` stays as published.

**v11 vs v10.**
* v11 = v10 + sm_80 member; unrelated to the unsealed fp16-P candidate of 2026-09-11.
* One kernel ADDED: `cuda_80` (`triattn_pkg/cuda_80/`, module `triattn_sm80`, extension `triattn_sm80_ext`) — the sm_80 (A100, cc 8.0) member of the
  family: `mma.sync.m16n8k16` (bf16 → fp32) + `ldmatrix` + a `cp.async` multistage pipeline within 163 KB of shared memory per block, two CTA geometries
  per head dim (64-query tiles, two CTAs per SM, for contiguous operands; 128-query tiles for transposed views at large S), with the family's shared
  algorithm and face — streaming max-free softmax in exp2 units, exact fp32 pair-bias staging in MMA-fragment order with bias-tile reuse across pair rows,
  key masks of every kind (none / one padding length per batch element / per-row ragged or interior masks), fully-masked rows = the uniform mean of `v`
  written without logit work, contiguous and transposed (strided-view, no copy) layouts, D 16 / 32 / 64, any H, any S and N incl. N ≠ S, B ≥ 1, 64-bit
  addressing, deterministic, CUDA-graph capturable, optional per-row log2-sum-exp output, SAFE fix pass with a visible census (`triattn_sm80.FALLBACKS`).
  Routed on cc 8.0 only.  Its D = 32 instantiations compile at the 255-register cap with an 8–120 B stack whose loads/stores all sit outside the
  pipelined key-loop bodies (SASS census; the general + lse small-tile kernel, used only for the log-sum-exp output, is the one exception with 3 stack
  loads per half step).
* The router (`dispatch/candidate.py`) is keyed by compute capability: on cc 9.0 it is generation 10's table, name for name; on cc 8.0 bf16 D ∈ {16, 32,
  64} with S_q = S_kv goes to `cuda_80` and fp16 / D = 128 / S_q ≠ S_kv to `tri`; on any other device every servable shape goes to `tri`.  A Gluon route
  (k13, or `tri` where it takes its k12 path) on an interpreter whose Triton has no `triton.experimental.gluon` (Triton < 3.6) now raises
  `Unsupported("needs_triton_gluon: …")` before any work instead of an untyped import error; nothing is rerouted silently.
* sm_90a members `cuda_b` (triattn_m1) and `cuda_c` (triattn_mw): dead-row early exit — a CTA tile whose pair rows are all fully masked skips the key
  loop and writes the uniform rows directly.  Outputs are bitwise unchanged on every vector and measured cell; on H100 80GB HBM3 (bf16 H4 D32, keypad with
  the last 14 % of pair rows and keys masked, the engines' padded form) triattn_m1 −11.9 % at N 1024, −12.7 % at 1536, −13.1 % at 2048 ms per call and
  triattn_mw −6.3 … −9.0 %; unpadded and key-padded cells within ±0.5 % (m1) / ±0.3 % (mw).  `cuda` (triattn_cuda), `triton/`, `dispatch/kernels/` are
  generation 10's sources (five docstrings reworded, no code change).
* Binaries: `prebuilt/` carries all four extensions for seven interpreter stacks (torch 2.7.1+cu126 / 2.7.1+cu128 / 2.10.0+cu128 / 2.12.0+cu130 /
  2.13.0+cu130 on CPython 3.11, torch 2.12.0+cu130 / 2.13.0+cu130 on CPython 3.12), each built by `tools/build_prebuilt.py` inside the stack's own image
  from exactly these sources, with digest records (`so_sha256`, `source_sha256`, `module_sha256`, `loadcheck`); `tools/check_prebuilt.py` re-verifies
  them.  The three sm_90a binaries of generation 10's one stack are replaced by this generation's builds (m1 / mw from the changed sources; sm90 from the
  unchanged `cuda/` sources — output-identical to generation 10's binary on every vector, not byte-identical as a file).
* Test vectors gain a `cc` field and 22 cc-8.0 cases (`a80_*`, made on A100-SXM4-80GB); the 17 cc-9.0 cases are generation 10's byte for byte;
  `test_pkg.py` runs the cases of the device's cc.  `CELLS.json` is keyed by cc: generation 10's 117 H100 cells unchanged plus 270 A100 cells.

```python
import sys; sys.path.insert(0, ".../pkg/v11")
import triattn_pkg
out = triattn_pkg.triangle_attention(q, k, v, bias, mask=None, scale=None)
```

## Contract

| argument | |
|---|---|
| `q, k, v` | `[B, N, H, S, D]` (or `[N, H, S, D]`), **bf16 or fp16**, CUDA. Contiguous, or strided views with `stride(-1) == 1` — an *ending-node* call is the same op on the transposed pair tensor and is passed as transposed views (no copies). Any number of pair rows `N`, including `N != S`; any `H`; `B ≥ 1`. |
| `bias` | `[B, 1, H, S_q, S_kv]` (or 4-D), fp32 or the 16-bit dtype of `q`: ONE pair bias shared by all `N` rows (the structural fact every kernel here exploits). |
| `mask` | `[B, N, 1, 1, S_kv]` bool, True = attend — a key mask per pair row (one padding length per batch element, per-row ragged lengths, interior zeros) — or `None`. |
| `scale` | softmax scale, default `D ** -0.5`. |
| returns | `[B, N, H, S_q, D]` (or 4-D) in `q`'s dtype: `softmax_k(scale·q·k + bias (+ −inf where mask == 0)) @ v` per pair row. Fully-masked pair rows yield the uniform average of `v` (the convention of cuequivariance_ops_torch's triangle_attention). |
| refusals | `triattn_pkg.Unsupported` (a `NotImplementedError` naming the reason), raised **before any work**: fp32 q/k/v, head dims outside {16, 32, 64, 128}, per-row biases, per-query masks, non-CUDA tensors, S_q ≠ S_kv outside the `tri` route's reach, a missing binary with no toolkit to build it, a Gluon route (k13, or tri's k12 path) on an interpreter whose Triton lacks `triton.experimental.gluon` (`needs_triton_gluon`), `TRIATTN_M1_FLAGS` / `MW_GEOM` set in the environment. Nothing is substituted silently. |

`triattn_pkg.route(q, k, v, bias, mask)` names the kernel a call takes; `triattn_pkg.prebuilt_status()` shows which binaries this process resolved;
`triattn_pkg.PINS` / `BEST_COMMIT` name the sealed commits.  Deterministic run-to-run; capturable in CUDA graphs (the cells are graph-replay timings).

## Routes and their numerics class

| cc | operands | S (keys = queries) | route | kernel | bias handling | source dir |
|---|---|---|---|---|---|---|
| 9.0 | bf16, D = 32 | S < 512, contiguous | `k13` | persistent warp-specialized Gluon kernel (Triton ≥ 3.6) | staged to bf16 tiles: exact for bf16-representable bias (a trunk's bias is), an arbitrary fp32 bias is rounded to bf16 first | `dispatch/kernels/k13.py` |
| 9.0 | bf16, D = 32 | S < 512, strided views | `cuda` | warp-specialized wgmma / TMA kernel (CUDA C++, CUTLASS) | staged to bf16 (same class as k13) | `cuda/` |
| 9.0 | bf16, D = 32 | 512 ≤ S ≤ 3072, and S > 4096 | `cuda_b` | `triattn_m1`: three consumer warpgroups, max-free streaming softmax, per-row dead key-tile skip, SAFE fix pass (CUDA C++, CUTLASS) | fp32 staging, exact | `cuda_b/` |
| 9.0 | bf16, D = 32 | 3072 < S ≤ 4096 | `cuda_c` | `triattn_mw`, geometry 3x4: TMA-multicast variant (CUDA C++, inline PTX) | fp32 staging, exact | `cuda_c/` |
| 9.0 | fp16, or D = 16 | S ≤ 640 → `k13`; above → `tri` | `k13` / `tri` | Gluon k13 / k12 | 16-bit tiles (exact for 16-bit-representable bias) | `triton/` |
| 9.0 | D ∈ {64, 128}, or S_q ≠ S_kv | any | `tri` | k10 (plain Triton) | 16-bit tiles when lossless, else fp32 | `triton/` |
| 8.0 | bf16, D ∈ {16, 32, 64}, S_q = S_kv | any S, contiguous or strided views | `cuda_80` | `triattn_sm80`: mma.sync / ldmatrix / cp.async member of the family (CUDA C++) | fp32 staging, exact | `cuda_80/` |
| 8.0 | fp16, D = 128, or S_q ≠ S_kv | any | `tri` | k10 | as above; unmeasured on this device | `triton/` |
| other | anything servable | any | `tri` | k10 / k12 | unmeasured | `triton/` |

Error class on the measured cc-9.0 cells (bf16 output vs an fp64 reference): max |err| 2-5·10⁻³ at |out| ≲ 4, i.e. ≤ 3.5 ulp above the
bf16 rounding floor (`ulpe_max` in CELLS.json) — the same class as cuequivariance_ops_torch 0.11.1 on the same cases.  The cc-8.0 member is held to the same class (its cells carry the same error fields).  The routing table itself is
the docstring of `triattn_pkg/dispatch/candidate.py`.

## Measured cells (CELLS.json)

`CELLS.json` (schema `triattn_pkg_cells/v2`) holds one record per measured case, keyed by (cc, dtype, H, D, B, N, S, node, mask kind, bias
class, grid): per contender the CUDA-graph-replay median / min ms and the eager median, the error vs fp64, the route, and the source file + case id;
`by_cc` holds the hardware, contender versions, timing protocol and clock statistics per cc and grid.  Timing protocol (both cc): CUDA-graph replay,
sustained and interleaved across contenders in 1 s blocks, 4 rounds, ≥ 20 replays, median; the error columns are the sampled-row error of the bf16
output against an fp64 reference (`ulpe_max` = max error in units of the bf16 spacing at the reference value, above the rounding floor).

**cc 9.0** (H100 80GB HBM3, torch 2.12.0+cu130, triton 3.7.0): generation 10's 117 cells (scoreboard 36, fp32bias 18, strided 18, prefixvar 5,
realdata 40), carried unchanged — generation 11's router takes the same route on every one of them and its binaries produce the same bytes.

cc 9.0 scoreboard, bf16 H4 D32 B = 1, starting node, unmasked (× = contender ms / package ms; contenders: opt_core's fpf_triatt_k2b and flash_triattn v9):

| S (= N) | route | package ms | k2b ms (×) | v9 ms (×) | package max err (ulpe_max) |
| --- | --- | --- | --- | --- | --- |
| 256 | k13 | 0.058 | 0.061 (×1.07) | 0.068 (×1.18) | 3.8e-03 (2.3) |
| 512 | cuda_b | 0.310 | 0.379 (×1.22) | 0.418 (×1.35) | 4.9e-03 (2.9) |
| 1024 | cuda_b | 2.063 | 2.796 (×1.36) | 3.125 (×1.51) | 3.4e-03 (3.5) |
| 2048 | cuda_b | 15.161 | 21.614 (×1.43) | 24.225 (×1.60) | 2.6e-03 (2.6) |
| 3072 | cuda_b | 53.850 | 71.803 (×1.33) | 80.655 (×1.50) | 2.1e-03 (2.3) |
| 4096 | cuda_c | 129.954 | 178.612 (×1.37) | 201.565 (×1.55) | 2.0e-03 (3.3) |


**cc 8.0** (A100-SXM4-80GB, 400 W, SM clock median 1395 MHz under load; torch 2.12.0+cu130, triton 3.7.0): 270 cells in 9 grids — four engine-shape
grids of 30 (bf16 B = 1; H4 D32, H8 D32, H4 D64, H4 D16; N = S ∈ {400, 800, 1200, 1536, 2048}; contiguous operands = starting node, transposed views =
ending node without a copy; forms bias_only = no mask, keypad = one padding length per batch element (7 % of keys), mask_bias = per-row ragged
lengths (7 % mean)), a scoreboard (H4 D32, N 256 … 4096, starting and ending node, no mask / keypad), a strided grid (the same on transposed views), an
fp32bias grid (the scoreboard's starting-node cells with a generic fp32 bias), a realdata grid (40 cells replaying operands, pair bias and masks recorded
from Boltz-2 and OpenFold3 trunk passes at N 917 … 1945, both nodes) and a secondary grid (fp32 generic bias; D 16 / 64, H 8, B 2, N ≠ S, N ∈ {255,
257, 1000, 1537}, interior mask holes, fp16).  Contenders timed in the same
runs: the Triton kernel fpf_triatt_k2b of opt_core 0.5.76.0 at its cc-8.0 launch cell ("k2b cell") and at tuned launch words ("k2b tuned":
BLOCK_M / BLOCK_N / rows / warps / stages / register cap as named) and this package's `tri` (k10).  The package column is the `cuda_80` kernel's own timing plug-in (the router adds a
capability lookup and shape checks per call); on the fp16 cells, which the router sends to `tri`, it is the `tri` plug-in.  × = contender ms / package ms.  The cc-8.0 cells were timed on the `cuda_80` build preceding the final
barrier-form change (outputs bitwise identical); an independent eager A/B of the shipped build on the general-kernel cells measured +0.0 … +0.4 % (one
cell, D32 H4 N 1536 mask_bias, +1.6 %), inside the stated margins.

On the 120 engine-shape cells (bf16, B = 1; H4 D32, H8 D32, H4 D64, H4 D16; N = S ∈ {400, 800, 1200, 1536, 2048}; contiguous and transposed-view operands; forms bias_only / keypad 7 % / mask_bias 7 %) the package is faster than the k2b launch cell on every cell (k2b ms / package ms: min ×1.08, geometric mean ×1.51, max ×2.01) and faster than the best of the tuned k2b launch words measured beside it on 119 of 120 (min ×0.98 = H8 D32 N 800 transposed bias_only, where the m128n32 word is 2 % faster; geometric mean ×1.42).
Outside the engine shapes: at N = S ≤ 257 the package ties the k2b cell (scoreboard N 256: ×1.00–1.07; transposed N 256: ×0.96; fp32-bias mask_bias
N 256: ×0.95); on the
fp16 cells (route `tri` = k10) k2b (×0.89–0.95) is faster than the package's Triton path.  Every other cell:
package fastest — including all 40 recorded-operand cells (k2b ×1.31–1.47) and all 18 fp32-bias scoreboard cells (k2b ×1.14–2.20).  Package error over the 270 cells: ulpe_max ≤ 3.4 with a bf16-exact bias, ≤ 4.3 with a generic fp32 bias (the contenders that stage
the bias in bf16 are not more accurate there), ≤ 2.7 on the recorded engine operands; deterministic on every cell.

Memory: the kernel allocates the output and a fix list (≤ 64 MB transient at N 2048 H4, 256 MB at N 4096); operands + output of one call are
≈ H·N²·(8·D + 4) bytes at N = S (bf16 q, k, v, out + fp32 bias), e.g. 4.4 GB at N 2048 H4 D32 — on a 40 GB card that bounds N at roughly 6000
(H4 D32), 4200 (H8 D32 or H4 D64), 8400 (H4 D16) for the tensors alone; there is no kernel-side size limit (64-bit addressing throughout).

#### cc 8.0 engine grid, bf16 H4 D32

| N (= S) | layout | form | package ms | k2b cell ms (×) | k2b tuned ms (×) [m128n32r2w4s3reg255, m64n64r2w4s3reg255] | k10 (tri) ms (×) | package err: max abs (ulpe_max) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | contiguous | bias_only | 0.366 | 0.444 (×1.22) | 0.457 (×1.25) m64n64r2w4s3 | 0.485 (×1.33) | 6.9e-03 (2.9) |
| 400 | contiguous | keypad 7 % | 0.369 | 0.570 (×1.55) | 0.635 (×1.72) m64n64r2w4s3 | 0.437 (×1.19) | 6.0e-03 (3.1) |
| 400 | contiguous | mask_bias 7 % | 0.489 | 0.570 (×1.17) | 0.636 (×1.30) m64n64r2w4s3 | 0.709 (×1.45) | 5.4e-03 (2.8) |
| 400 | transposed view | bias_only | 0.385 | 0.464 (×1.20) | 0.488 (×1.27) m64n64r2w4s3 | 0.519 (×1.35) | 5.4e-03 (2.7) |
| 400 | transposed view | keypad 7 % | 0.390 | 0.604 (×1.55) | 0.670 (×1.72) m64n64r2w4s3 | 0.462 (×1.18) | 6.0e-03 (2.9) |
| 400 | transposed view | mask_bias 7 % | 0.504 | 0.604 (×1.20) | 0.671 (×1.33) m64n64r2w4s3 | 0.741 (×1.47) | 4.9e-03 (2.8) |
| 800 | contiguous | bias_only | 2.229 | 2.872 (×1.29) | 2.796 (×1.25) m64n64r2w4s3 | 2.954 (×1.33) | 3.6e-03 (2.5) |
| 800 | contiguous | keypad 7 % | 2.161 | 3.653 (×1.69) | 3.805 (×1.76) m64n64r2w4s3 | 2.801 (×1.30) | 5.0e-03 (2.6) |
| 800 | contiguous | mask_bias 7 % | 2.662 | 3.652 (×1.37) | 3.805 (×1.43) m64n64r2w4s3 | 4.278 (×1.61) | 4.2e-03 (2.7) |
| 800 | transposed view | bias_only | 2.477 | 3.012 (×1.22) | 3.002 (×1.21) m128n32r2w4s3 | 3.254 (×1.31) | 4.5e-03 (2.6) |
| 800 | transposed view | keypad 7 % | 2.433 | 3.817 (×1.57) | 4.111 (×1.69) m64n64r2w4s3 | 3.127 (×1.29) | 4.0e-03 (2.8) |
| 800 | transposed view | mask_bias 7 % | 2.853 | 3.818 (×1.34) | 4.109 (×1.44) m64n64r2w4s3 | 4.584 (×1.61) | 3.9e-03 (3.0) |
| 1200 | contiguous | bias_only | 7.005 | 9.406 (×1.34) | 8.640 (×1.23) m64n64r2w4s3 | 9.160 (×1.31) | 3.5e-03 (2.8) |
| 1200 | contiguous | keypad 7 % | 6.676 | 12.076 (×1.81) | 11.909 (×1.78) m64n64r2w4s3 | 8.917 (×1.34) | 2.9e-03 (3.0) |
| 1200 | contiguous | mask_bias 7 % | 8.132 | 12.011 (×1.48) | 11.909 (×1.46) m64n64r2w4s3 | 13.101 (×1.61) | 4.3e-03 (2.7) |
| 1200 | transposed view | bias_only | 7.805 | 10.382 (×1.33) | 9.701 (×1.24) m128n32r2w4s3 | 10.925 (×1.40) | 3.0e-03 (2.6) |
| 1200 | transposed view | keypad 7 % | 7.539 | 12.765 (×1.69) | 13.078 (×1.73) m64n64r2w4s3 | 10.713 (×1.42) | 3.0e-03 (3.1) |
| 1200 | transposed view | mask_bias 7 % | 7.983 | 12.763 (×1.60) | 13.078 (×1.64) m64n64r2w4s3 | 14.440 (×1.81) | 3.2e-03 (2.7) |
| 1536 | contiguous | bias_only | 14.217 | 18.738 (×1.32) | 16.883 (×1.19) m64n64r2w4s3 | 18.125 (×1.27) | 2.8e-03 (3.1) |
| 1536 | contiguous | keypad 7 % | 13.671 | 24.070 (×1.76) | 23.018 (×1.68) m64n64r2w4s3 | 18.120 (×1.33) | 2.3e-03 (2.9) |
| 1536 | contiguous | mask_bias 7 % | 14.531 | 24.070 (×1.66) | 23.023 (×1.58) m64n64r2w4s3 | 26.477 (×1.82) | 2.3e-03 (3.0) |
| 1536 | transposed view | bias_only | 14.926 | 22.209 (×1.49) | 18.323 (×1.23) m128n32r2w4s3 | 23.202 (×1.55) | 2.5e-03 (2.8) |
| 1536 | transposed view | keypad 7 % | 14.572 | 26.087 (×1.79) | 25.456 (×1.75) m128n32r2w4s3 | 22.896 (×1.57) | 2.3e-03 (2.9) |
| 1536 | transposed view | mask_bias 7 % | 15.142 | 26.082 (×1.72) | 25.455 (×1.68) m128n32r2w4s3 | 29.494 (×1.95) | 2.5e-03 (2.9) |
| 2048 | contiguous | bias_only | 32.979 | 44.045 (×1.34) | 39.440 (×1.20) m64n64r2w4s3 | 41.938 (×1.27) | 1.9e-03 (2.8) |
| 2048 | contiguous | keypad 7 % | 31.019 | 56.303 (×1.82) | 54.106 (×1.74) m64n64r2w4s3 | 40.888 (×1.32) | 2.6e-03 (2.4) |
| 2048 | contiguous | mask_bias 7 % | 33.026 | 56.292 (×1.70) | 53.902 (×1.63) m64n64r2w4s3 | 60.643 (×1.84) | 3.5e-03 (2.5) |
| 2048 | transposed view | bias_only | 34.803 | 56.571 (×1.63) | 42.339 (×1.22) m128n32r2w4s3 | 57.948 (×1.67) | 2.5e-03 (2.5) |
| 2048 | transposed view | keypad 7 % | 32.981 | 62.332 (×1.89) | 59.385 (×1.80) m128n32r2w4s3 | 55.102 (×1.67) | 2.8e-03 (2.4) |
| 2048 | transposed view | mask_bias 7 % | 34.684 | 62.325 (×1.80) | 59.378 (×1.71) m128n32r2w4s3 | 70.455 (×2.03) | 2.9e-03 (2.7) |

#### cc 8.0 engine grid, bf16 H8 D32

| N (= S) | layout | form | package ms | k2b cell ms (×) | k2b tuned ms (×) [m128n32r2w4s3reg255, m64n64r2w4s3reg255] | k10 (tri) ms (×) | package err: max abs (ulpe_max) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | contiguous | bias_only | 0.710 | 0.870 (×1.23) | 0.901 (×1.27) m64n64r2w4s3 | 0.946 (×1.33) | 5.0e-03 (2.7) |
| 400 | contiguous | keypad 7 % | 0.696 | 1.124 (×1.61) | 1.245 (×1.79) m64n64r2w4s3 | 0.840 (×1.21) | 5.3e-03 (3.2) |
| 400 | contiguous | mask_bias 7 % | 0.903 | 1.124 (×1.24) | 1.244 (×1.38) m64n64r2w4s3 | 1.315 (×1.46) | 7.5e-03 (3.0) |
| 400 | transposed view | bias_only | 0.781 | 0.916 (×1.17) | 0.996 (×1.28) m64n64r2w4s3 | 1.044 (×1.34) | 6.2e-03 (2.8) |
| 400 | transposed view | keypad 7 % | 0.782 | 1.192 (×1.52) | 1.339 (×1.71) m64n64r2w4s3 | 0.934 (×1.19) | 5.1e-03 (3.2) |
| 400 | transposed view | mask_bias 7 % | 0.955 | 1.191 (×1.25) | 1.339 (×1.40) m64n64r2w4s3 | 1.405 (×1.47) | 6.7e-03 (3.2) |
| 800 | contiguous | bias_only | 4.411 | 5.682 (×1.29) | 5.565 (×1.26) m64n64r2w4s3 | 5.863 (×1.33) | 5.4e-03 (2.5) |
| 800 | contiguous | keypad 7 % | 4.255 | 7.266 (×1.71) | 7.577 (×1.78) m64n64r2w4s3 | 5.547 (×1.30) | 5.7e-03 (2.7) |
| 800 | contiguous | mask_bias 7 % | 5.255 | 7.267 (×1.38) | 7.578 (×1.44) m64n64r2w4s3 | 8.462 (×1.61) | 4.6e-03 (3.0) |
| 800 | transposed view | bias_only | 6.154 | 6.664 (×1.08) | 6.022 (×0.98) m128n32r2w4s3 | 7.164 (×1.16) | 5.0e-03 (3.5) |
| 800 | transposed view | keypad 7 % | 6.027 | 7.855 (×1.30) | 8.270 (×1.37) m128n32r2w4s3 | 6.904 (×1.15) | 6.0e-03 (3.5) |
| 800 | transposed view | mask_bias 7 % | 6.465 | 7.855 (×1.22) | 8.269 (×1.28) m128n32r2w4s3 | 9.376 (×1.45) | 5.9e-03 (3.4) |
| 1200 | contiguous | bias_only | 13.952 | 18.613 (×1.33) | 17.087 (×1.22) m64n64r2w4s3 | 18.182 (×1.30) | 2.8e-03 (3.1) |
| 1200 | contiguous | keypad 7 % | 13.254 | 23.987 (×1.81) | 23.535 (×1.78) m64n64r2w4s3 | 17.581 (×1.33) | 2.9e-03 (3.0) |
| 1200 | contiguous | mask_bias 7 % | 16.020 | 23.984 (×1.50) | 23.530 (×1.47) m64n64r2w4s3 | 26.112 (×1.63) | 3.3e-03 (3.3) |
| 1200 | transposed view | bias_only | 15.600 | 22.665 (×1.45) | 19.265 (×1.23) m128n32r2w4s3 | 23.552 (×1.51) | 2.6e-03 (3.5) |
| 1200 | transposed view | keypad 7 % | 15.185 | 26.336 (×1.73) | 26.841 (×1.77) m128n32r2w4s3 | 22.868 (×1.51) | 2.6e-03 (3.2) |
| 1200 | transposed view | mask_bias 7 % | 16.197 | 26.124 (×1.61) | 26.833 (×1.66) m128n32r2w4s3 | 29.613 (×1.83) | 2.5e-03 (3.6) |
| 1536 | contiguous | bias_only | 28.104 | 37.441 (×1.33) | 33.742 (×1.20) m64n64r2w4s3 | 35.995 (×1.28) | 4.6e-03 (3.2) |
| 1536 | contiguous | keypad 7 % | 27.208 | 48.079 (×1.77) | 46.208 (×1.70) m64n64r2w4s3 | 36.237 (×1.33) | 4.3e-03 (2.8) |
| 1536 | contiguous | mask_bias 7 % | 28.583 | 48.079 (×1.68) | 46.009 (×1.61) m64n64r2w4s3 | 53.110 (×1.86) | 3.4e-03 (3.1) |
| 1536 | transposed view | bias_only | 30.201 | 47.756 (×1.58) | 36.753 (×1.22) m128n32r2w4s3 | 49.237 (×1.63) | 2.6e-03 (2.8) |
| 1536 | transposed view | keypad 7 % | 29.182 | 53.310 (×1.83) | 50.898 (×1.74) m128n32r2w4s3 | 48.044 (×1.65) | 3.1e-03 (3.0) |
| 1536 | transposed view | mask_bias 7 % | 30.855 | 53.302 (×1.73) | 50.901 (×1.65) m128n32r2w4s3 | 60.602 (×1.96) | 3.3e-03 (3.3) |
| 2048 | contiguous | bias_only | 66.031 | 88.051 (×1.33) | 78.647 (×1.19) m64n64r2w4s3 | 83.690 (×1.27) | 2.3e-03 (3.2) |
| 2048 | contiguous | keypad 7 % | 62.113 | 112.570 (×1.81) | 108.007 (×1.74) m64n64r2w4s3 | 81.800 (×1.32) | 3.2e-03 (2.8) |
| 2048 | contiguous | mask_bias 7 % | 66.893 | 112.600 (×1.68) | 107.859 (×1.61) m64n64r2w4s3 | 120.698 (×1.80) | 2.8e-03 (3.1) |
| 2048 | transposed view | bias_only | 69.746 | 113.220 (×1.62) | 85.090 (×1.22) m128n32r2w4s3 | 115.989 (×1.66) | 3.7e-03 (2.7) |
| 2048 | transposed view | keypad 7 % | 66.183 | 124.694 (×1.88) | 118.685 (×1.79) m128n32r2w4s3 | 110.310 (×1.67) | 2.9e-03 (3.0) |
| 2048 | transposed view | mask_bias 7 % | 70.182 | 124.670 (×1.78) | 118.660 (×1.69) m128n32r2w4s3 | 140.891 (×2.01) | 2.1e-03 (3.0) |

#### cc 8.0 engine grid, bf16 H4 D64

| N (= S) | layout | form | package ms | k2b cell ms (×) | k2b tuned ms (×) [m128n64r1w4s2reg255] | k10 (tri) ms (×) | package err: max abs (ulpe_max) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | contiguous | bias_only | 0.608 | 0.743 (×1.22) | 0.839 (×1.38) m128n64r1w4s2 | 0.794 (×1.30) | 5.3e-03 (2.6) |
| 400 | contiguous | keypad 7 % | 0.608 | 1.020 (×1.68) | 1.048 (×1.72) m128n64r1w4s2 | 0.675 (×1.11) | 5.0e-03 (3.0) |
| 400 | contiguous | mask_bias 7 % | 0.654 | 1.020 (×1.56) | 1.039 (×1.59) m128n64r1w4s2 | 0.970 (×1.48) | 4.9e-03 (2.8) |
| 400 | transposed view | bias_only | 0.620 | 0.751 (×1.21) | 0.855 (×1.38) m128n64r1w4s2 | 0.798 (×1.29) | 5.9e-03 (2.7) |
| 400 | transposed view | keypad 7 % | 0.614 | 1.038 (×1.69) | 1.073 (×1.75) m128n64r1w4s2 | 0.682 (×1.11) | 5.4e-03 (2.7) |
| 400 | transposed view | mask_bias 7 % | 0.665 | 1.040 (×1.57) | 1.064 (×1.60) m128n64r1w4s2 | 0.990 (×1.49) | 5.9e-03 (2.7) |
| 800 | contiguous | bias_only | 3.798 | 4.649 (×1.22) | 4.660 (×1.23) m128n64r1w4s2 | 4.862 (×1.28) | 4.4e-03 (2.5) |
| 800 | contiguous | keypad 7 % | 3.801 | 6.195 (×1.63) | 5.967 (×1.57) m128n64r1w4s2 | 4.342 (×1.14) | 4.1e-03 (2.5) |
| 800 | contiguous | mask_bias 7 % | 3.949 | 6.198 (×1.57) | 5.936 (×1.50) m128n64r1w4s2 | 6.050 (×1.53) | 5.5e-03 (2.9) |
| 800 | transposed view | bias_only | 3.884 | 4.726 (×1.22) | 4.737 (×1.22) m128n64r1w4s2 | 4.946 (×1.27) | 4.4e-03 (2.6) |
| 800 | transposed view | keypad 7 % | 3.881 | 6.324 (×1.63) | 6.096 (×1.57) m128n64r1w4s2 | 4.471 (×1.15) | 5.0e-03 (2.7) |
| 800 | transposed view | mask_bias 7 % | 4.036 | 6.323 (×1.57) | 6.068 (×1.50) m128n64r1w4s2 | 6.191 (×1.53) | 4.6e-03 (2.5) |
| 1200 | contiguous | bias_only | 11.916 | 14.636 (×1.23) | 14.065 (×1.18) m128n64r1w4s2 | 15.173 (×1.27) | 2.3e-03 (3.3) |
| 1200 | contiguous | keypad 7 % | 11.730 | 19.395 (×1.65) | 18.004 (×1.53) m128n64r1w4s2 | 13.739 (×1.17) | 3.2e-03 (3.2) |
| 1200 | contiguous | mask_bias 7 % | 12.090 | 19.380 (×1.60) | 17.953 (×1.48) m128n64r1w4s2 | 18.692 (×1.55) | 2.6e-03 (3.2) |
| 1200 | transposed view | bias_only | 12.464 | 14.758 (×1.18) | 14.203 (×1.14) m128n64r1w4s2 | 15.332 (×1.23) | 2.4e-03 (2.5) |
| 1200 | transposed view | keypad 7 % | 12.135 | 19.679 (×1.62) | 18.294 (×1.51) m128n64r1w4s2 | 14.318 (×1.18) | 2.8e-03 (3.0) |
| 1200 | transposed view | mask_bias 7 % | 12.406 | 19.684 (×1.59) | 18.251 (×1.47) m128n64r1w4s2 | 19.093 (×1.54) | 2.2e-03 (2.7) |
| 1536 | contiguous | bias_only | 23.720 | 28.675 (×1.21) | 26.419 (×1.11) m128n64r1w4s2 | 30.415 (×1.28) | 2.9e-03 (3.1) |
| 1536 | contiguous | keypad 7 % | 23.554 | 38.305 (×1.63) | 34.115 (×1.45) m128n64r1w4s2 | 27.654 (×1.17) | 2.5e-03 (2.7) |
| 1536 | contiguous | mask_bias 7 % | 24.215 | 38.205 (×1.58) | 33.744 (×1.39) m128n64r1w4s2 | 37.824 (×1.56) | 3.2e-03 (2.7) |
| 1536 | transposed view | bias_only | 24.532 | 29.185 (×1.19) | 26.758 (×1.09) m128n64r1w4s2 | 30.222 (×1.23) | 3.3e-03 (2.6) |
| 1536 | transposed view | keypad 7 % | 23.910 | 39.229 (×1.64) | 34.664 (×1.45) m128n64r1w4s2 | 29.290 (×1.22) | 3.4e-03 (2.8) |
| 1536 | transposed view | mask_bias 7 % | 23.526 | 39.188 (×1.67) | 34.443 (×1.46) m128n64r1w4s2 | 38.585 (×1.64) | 3.2e-03 (3.2) |
| 2048 | contiguous | bias_only | 55.540 | 67.886 (×1.22) | 60.926 (×1.10) m128n64r1w4s2 | 71.867 (×1.29) | 1.9e-03 (2.4) |
| 2048 | contiguous | keypad 7 % | 53.958 | 88.333 (×1.64) | 78.482 (×1.45) m128n64r1w4s2 | 63.222 (×1.17) | 1.8e-03 (2.6) |
| 2048 | contiguous | mask_bias 7 % | 56.665 | 88.331 (×1.56) | 78.286 (×1.38) m128n64r1w4s2 | 86.878 (×1.53) | 2.1e-03 (2.5) |
| 2048 | transposed view | bias_only | 56.784 | 68.951 (×1.21) | 61.657 (×1.09) m128n64r1w4s2 | 71.487 (×1.26) | 1.9e-03 (2.4) |
| 2048 | transposed view | keypad 7 % | 54.045 | 90.804 (×1.68) | 79.360 (×1.47) m128n64r1w4s2 | 67.936 (×1.26) | 2.7e-03 (2.4) |
| 2048 | transposed view | mask_bias 7 % | 54.104 | 90.830 (×1.68) | 79.229 (×1.46) m128n64r1w4s2 | 88.999 (×1.64) | 2.1e-03 (2.8) |

#### cc 8.0 engine grid, bf16 H4 D16

| N (= S) | layout | form | package ms | k2b cell ms (×) | k2b tuned ms (×) [m128n32r2w4s3regnone, m64n64r2w4s2reg128] | k10 (tri) ms (×) | package err: max abs (ulpe_max) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | contiguous | bias_only | 0.293 | 0.325 (×1.11) | 0.340 (×1.16) m64n64r2w4s2 | 0.364 (×1.24) | 4.8e-03 (3.1) |
| 400 | contiguous | keypad 7 % | 0.302 | 0.474 (×1.57) | 0.469 (×1.55) m64n64r2w4s2 | 0.317 (×1.05) | 6.3e-03 (3.3) |
| 400 | contiguous | mask_bias 7 % | 0.357 | 0.474 (×1.33) | 0.469 (×1.31) m64n64r2w4s2 | 0.515 (×1.44) | 6.5e-03 (4.1) |
| 400 | transposed view | bias_only | 0.308 | 0.368 (×1.19) | 0.371 (×1.20) m64n64r2w4s2 | 0.400 (×1.30) | 5.3e-03 (2.7) |
| 400 | transposed view | keypad 7 % | 0.324 | 0.536 (×1.66) | 0.502 (×1.55) m64n64r2w4s2 | 0.356 (×1.10) | 4.6e-03 (2.9) |
| 400 | transposed view | mask_bias 7 % | 0.376 | 0.534 (×1.42) | 0.502 (×1.34) m64n64r2w4s2 | 0.553 (×1.47) | 5.1e-03 (3.1) |
| 800 | contiguous | bias_only | 1.704 | 2.035 (×1.19) | 2.058 (×1.21) m128n32r2w4s3 | 2.180 (×1.28) | 5.3e-03 (3.0) |
| 800 | contiguous | keypad 7 % | 1.676 | 2.918 (×1.74) | 2.822 (×1.68) m64n64r2w4s2 | 1.994 (×1.19) | 5.4e-03 (3.0) |
| 800 | contiguous | mask_bias 7 % | 1.894 | 2.917 (×1.54) | 2.810 (×1.48) m64n64r2w4s2 | 3.169 (×1.67) | 4.6e-03 (3.0) |
| 800 | transposed view | bias_only | 1.785 | 2.223 (×1.25) | 2.167 (×1.21) m128n32r2w4s3 | 2.428 (×1.36) | 5.4e-03 (3.1) |
| 800 | transposed view | keypad 7 % | 1.798 | 3.267 (×1.82) | 3.017 (×1.68) m64n64r2w4s2 | 2.273 (×1.26) | 4.8e-03 (3.1) |
| 800 | transposed view | mask_bias 7 % | 1.997 | 3.267 (×1.64) | 3.018 (×1.51) m64n64r2w4s2 | 3.446 (×1.73) | 4.4e-03 (3.2) |
| 1200 | contiguous | bias_only | 5.329 | 6.755 (×1.27) | 6.416 (×1.20) m64n64r2w4s2 | 6.662 (×1.25) | 2.9e-03 (3.4) |
| 1200 | contiguous | keypad 7 % | 5.028 | 9.471 (×1.88) | 8.812 (×1.75) m64n64r2w4s2 | 6.263 (×1.25) | 2.8e-03 (3.1) |
| 1200 | contiguous | mask_bias 7 % | 5.618 | 9.469 (×1.69) | 8.816 (×1.57) m64n64r2w4s2 | 9.758 (×1.74) | 3.0e-03 (2.9) |
| 1200 | transposed view | bias_only | 5.646 | 7.540 (×1.34) | 7.129 (×1.26) m128n32r2w4s3 | 7.833 (×1.39) | 2.9e-03 (2.5) |
| 1200 | transposed view | keypad 7 % | 5.492 | 10.567 (×1.92) | 9.575 (×1.74) m64n64r2w4s2 | 7.502 (×1.37) | 2.8e-03 (2.8) |
| 1200 | transposed view | mask_bias 7 % | 6.028 | 10.612 (×1.76) | 9.576 (×1.59) m64n64r2w4s2 | 10.799 (×1.79) | 2.6e-03 (2.6) |
| 1536 | contiguous | bias_only | 10.575 | 13.452 (×1.27) | 12.469 (×1.18) m128n32r2w4s3 | 13.190 (×1.25) | 2.4e-03 (2.9) |
| 1536 | contiguous | keypad 7 % | 9.958 | 18.993 (×1.91) | 17.241 (×1.73) m64n64r2w4s2 | 12.865 (×1.29) | 2.2e-03 (3.0) |
| 1536 | contiguous | mask_bias 7 % | 11.184 | 18.982 (×1.70) | 17.230 (×1.54) m64n64r2w4s2 | 20.141 (×1.80) | 2.2e-03 (2.7) |
| 1536 | transposed view | bias_only | 11.475 | 15.154 (×1.32) | 13.148 (×1.15) m128n32r2w4s3 | 16.180 (×1.41) | 3.6e-03 (2.6) |
| 1536 | transposed view | keypad 7 % | 11.170 | 21.121 (×1.89) | 19.134 (×1.71) m64n64r2w4s2 | 15.787 (×1.41) | 3.1e-03 (2.8) |
| 1536 | transposed view | mask_bias 7 % | 12.216 | 21.123 (×1.73) | 19.141 (×1.57) m64n64r2w4s2 | 22.303 (×1.83) | 3.4e-03 (3.1) |
| 2048 | contiguous | bias_only | 24.698 | 31.750 (×1.29) | 29.526 (×1.20) m128n32r2w4s3 | 30.386 (×1.23) | 4.0e-03 (3.4) |
| 2048 | contiguous | keypad 7 % | 22.475 | 43.993 (×1.96) | 40.613 (×1.81) m64n64r2w4s2 | 28.884 (×1.29) | 3.3e-03 (2.3) |
| 2048 | contiguous | mask_bias 7 % | 25.485 | 43.985 (×1.73) | 40.605 (×1.59) m64n64r2w4s2 | 45.447 (×1.78) | 2.9e-03 (3.1) |
| 2048 | transposed view | bias_only | 28.470 | 46.026 (×1.62) | 30.926 (×1.09) m128n32r2w4s3 | 50.841 (×1.79) | 2.1e-03 (2.8) |
| 2048 | transposed view | keypad 7 % | 26.138 | 52.580 (×2.01) | 49.844 (×1.91) m128n32r2w4s3 | 48.178 (×1.84) | 3.5e-03 (2.9) |
| 2048 | transposed view | mask_bias 7 % | 26.518 | 52.603 (×1.98) | 49.845 (×1.88) m128n32r2w4s3 | 58.752 (×2.22) | 1.8e-03 (2.8) |

#### cc 8.0 scoreboard, bf16 H4 D32 (starting node = contiguous; ending node = the transposed operands are made contiguous before the call)

| N (= S) | layout | form | package ms | k2b cell ms (×) | k2b tuned ms (×) | k10 (tri) ms (×) | package err: max abs (ulpe_max) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | contiguous | bias_only | 0.114 | 0.114 (×1.00) | — | 0.121 (×1.06) | 6.1e-03 (2.9) |
| 256 | contiguous | keypad 7 % | 0.130 | 0.147 (×1.13) | — | 0.136 (×1.05) | 5.7e-03 (2.9) |
| 256 | ending node, operands made contiguous | bias_only | 0.114 | 0.114 (×1.00) | — | 0.121 (×1.06) | 5.1e-03 (2.9) |
| 256 | ending node, operands made contiguous | keypad 7 % | 0.130 | 0.147 (×1.13) | — | 0.136 (×1.05) | 5.4e-03 (2.5) |
| 384 | contiguous | bias_only | 0.295 | 0.340 (×1.15) | — | 0.339 (×1.15) | 4.9e-03 (3.8) |
| 384 | contiguous | keypad 7 % | 0.312 | 0.436 (×1.40) | — | 0.369 (×1.18) | 5.3e-03 (3.6) |
| 384 | ending node, operands made contiguous | bias_only | 0.295 | 0.340 (×1.15) | — | 0.339 (×1.15) | 4.5e-03 (3.2) |
| 384 | ending node, operands made contiguous | keypad 7 % | 0.313 | 0.436 (×1.39) | — | 0.370 (×1.18) | 5.2e-03 (2.9) |
| 512 | contiguous | bias_only | 0.618 | 0.755 (×1.22) | — | 0.744 (×1.21) | 4.8e-03 (4.2) |
| 512 | contiguous | keypad 7 % | 0.641 | 0.977 (×1.52) | — | 0.789 (×1.23) | 5.1e-03 (3.6) |
| 512 | ending node, operands made contiguous | bias_only | 0.621 | 0.755 (×1.22) | — | 0.744 (×1.20) | 4.5e-03 (3.2) |
| 512 | ending node, operands made contiguous | keypad 7 % | 0.640 | 0.977 (×1.53) | — | 0.789 (×1.23) | 5.9e-03 (3.0) |
| 768 | contiguous | bias_only | 1.904 | 2.431 (×1.28) | — | 2.374 (×1.25) | 5.0e-03 (3.1) |
| 768 | contiguous | keypad 7 % | 1.924 | 3.131 (×1.63) | — | 2.512 (×1.31) | 4.8e-03 (3.0) |
| 768 | ending node, operands made contiguous | bias_only | 1.912 | 2.431 (×1.27) | — | 2.374 (×1.24) | 4.8e-03 (2.6) |
| 768 | ending node, operands made contiguous | keypad 7 % | 1.925 | 3.131 (×1.63) | — | 2.515 (×1.31) | 3.7e-03 (2.8) |
| 1024 | contiguous | bias_only | 4.318 | 5.646 (×1.31) | — | 5.458 (×1.26) | 5.0e-03 (3.9) |
| 1024 | contiguous | keypad 7 % | 4.083 | 7.237 (×1.77) | — | 5.358 (×1.31) | 5.1e-03 (3.8) |
| 1024 | ending node, operands made contiguous | bias_only | 4.320 | 5.646 (×1.31) | — | 5.447 (×1.26) | 4.9e-03 (3.5) |
| 1024 | ending node, operands made contiguous | keypad 7 % | 4.086 | 7.236 (×1.77) | — | 5.361 (×1.31) | 5.0e-03 (4.5) |
| 1536 | contiguous | bias_only | 14.088 | 18.736 (×1.33) | — | 17.995 (×1.28) | 2.8e-03 (3.1) |
| 1536 | contiguous | keypad 7 % | 13.674 | 24.070 (×1.76) | — | 18.119 (×1.33) | 2.3e-03 (2.9) |
| 1536 | ending node, operands made contiguous | bias_only | 14.085 | 18.737 (×1.33) | — | 18.010 (×1.28) | 2.5e-03 (2.8) |
| 1536 | ending node, operands made contiguous | keypad 7 % | 13.664 | 24.070 (×1.76) | — | 18.136 (×1.33) | 2.3e-03 (2.9) |
| 2048 | contiguous | bias_only | 32.976 | 44.034 (×1.34) | — | 41.891 (×1.27) | 1.9e-03 (2.8) |
| 2048 | contiguous | keypad 7 % | 31.019 | 56.293 (×1.81) | — | 40.904 (×1.32) | 2.6e-03 (2.4) |
| 2048 | ending node, operands made contiguous | bias_only | 32.973 | 44.042 (×1.34) | — | 41.885 (×1.27) | 2.5e-03 (2.5) |
| 2048 | ending node, operands made contiguous | keypad 7 % | 31.022 | 56.295 (×1.81) | — | 40.873 (×1.32) | 2.8e-03 (2.4) |
| 3072 | contiguous | bias_only | 109.452 | 147.898 (×1.35) | — | 141.884 (×1.30) | 2.1e-03 (2.8) |
| 3072 | contiguous | keypad 7 % | 103.005 | 193.324 (×1.88) | — | 140.672 (×1.37) | 2.7e-03 (2.7) |
| 3072 | ending node, operands made contiguous | bias_only | 109.480 | 148.233 (×1.35) | — | 141.770 (×1.29) | 1.8e-03 (2.7) |
| 3072 | ending node, operands made contiguous | keypad 7 % | 102.906 | 193.331 (×1.88) | — | 140.919 (×1.37) | 3.1e-03 (2.4) |
| 4096 | contiguous | bias_only | 258.386 | 368.190 (×1.42) | — | 340.683 (×1.32) | 1.6e-03 (3.1) |
| 4096 | contiguous | keypad 7 % | 240.852 | 477.134 (×1.98) | — | 334.606 (×1.39) | 1.7e-03 (3.0) |
| 4096 | ending node, operands made contiguous | bias_only | 258.215 | 368.116 (×1.43) | — | 340.535 (×1.32) | 1.7e-03 (3.1) |
| 4096 | ending node, operands made contiguous | keypad 7 % | 240.774 | 477.698 (×1.98) | — | 334.882 (×1.39) | 1.7e-03 (3.4) |

#### cc 8.0 strided grid, bf16 H4 D32 (ending node as transposed views, no copy)

| N (= S) | layout | form | package ms | k2b cell ms (×) | k2b tuned ms (×) | k10 (tri) ms (×) | package err: max abs (ulpe_max) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | transposed view | bias_only | 0.121 | 0.116 (×0.96) | — | 0.127 (×1.05) | 5.1e-03 (2.9) |
| 256 | transposed view | keypad 7 % | 0.137 | 0.157 (×1.14) | — | 0.143 (×1.05) | 5.4e-03 (2.5) |
| 384 | transposed view | bias_only | 0.308 | 0.347 (×1.13) | — | 0.359 (×1.17) | 4.5e-03 (3.2) |
| 384 | transposed view | keypad 7 % | 0.334 | 0.452 (×1.35) | — | 0.390 (×1.17) | 5.2e-03 (2.9) |
| 512 | transposed view | bias_only | 0.648 | 0.777 (×1.20) | — | 0.795 (×1.23) | 4.5e-03 (3.2) |
| 512 | transposed view | keypad 7 % | 0.670 | 1.008 (×1.50) | — | 0.855 (×1.28) | 5.9e-03 (3.0) |
| 768 | transposed view | bias_only | 2.018 | 2.511 (×1.24) | — | 2.580 (×1.28) | 4.8e-03 (2.6) |
| 768 | transposed view | keypad 7 % | 2.051 | 3.220 (×1.57) | — | 2.742 (×1.34) | 3.7e-03 (2.8) |
| 1024 | transposed view | bias_only | 4.610 | 6.335 (×1.37) | — | 6.832 (×1.48) | 4.9e-03 (3.5) |
| 1024 | transposed view | keypad 7 % | 4.412 | 7.730 (×1.75) | — | 6.633 (×1.50) | 5.0e-03 (4.5) |
| 1536 | transposed view | bias_only | 14.806 | 22.040 (×1.49) | — | 23.258 (×1.57) | 2.5e-03 (2.8) |
| 1536 | transposed view | keypad 7 % | 14.461 | 25.783 (×1.78) | — | 22.954 (×1.59) | 2.3e-03 (2.9) |
| 2048 | transposed view | bias_only | 34.517 | 56.728 (×1.64) | — | 58.108 (×1.68) | 2.5e-03 (2.5) |
| 2048 | transposed view | keypad 7 % | 32.845 | 61.881 (×1.88) | — | 55.254 (×1.68) | 2.8e-03 (2.4) |
| 3072 | transposed view | bias_only | 115.750 | 191.935 (×1.66) | — | 195.538 (×1.69) | 1.8e-03 (2.7) |
| 3072 | transposed view | keypad 7 % | 109.855 | 210.702 (×1.92) | — | 185.685 (×1.69) | 3.1e-03 (2.4) |
| 4096 | transposed view | bias_only | 270.427 | 464.011 (×1.72) | — | 460.985 (×1.70) | 1.7e-03 (3.1) |
| 4096 | transposed view | keypad 7 % | 254.521 | 509.740 (×2.00) | — | 436.513 (×1.72) | 1.7e-03 (3.4) |

#### cc 8.0 fp32bias grid, bf16 H4 D32, generic fp32 bias, starting node

| N (= S) | layout | form | package ms | k2b cell ms (×) | k2b tuned ms (×) | k10 (tri) ms (×) | package err: max abs (ulpe_max) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | contiguous | bias_only | 0.114 | 0.130 (×1.14) | — | 0.139 (×1.23) | 3.7e-03 (3.5) |
| 256 | contiguous | keypad 7 % | 0.130 | 0.164 (×1.26) | — | 0.158 (×1.21) | 4.7e-03 (2.8) |
| 384 | contiguous | bias_only | 0.296 | 0.383 (×1.29) | — | 0.404 (×1.36) | 5.5e-03 (3.1) |
| 384 | contiguous | keypad 7 % | 0.312 | 0.478 (×1.53) | — | 0.435 (×1.39) | 5.7e-03 (2.9) |
| 512 | contiguous | bias_only | 0.623 | 0.845 (×1.36) | — | 0.874 (×1.40) | 4.3e-03 (3.4) |
| 512 | contiguous | keypad 7 % | 0.640 | 1.061 (×1.66) | — | 0.940 (×1.47) | 4.4e-03 (3.0) |
| 768 | contiguous | bias_only | 1.910 | 2.724 (×1.43) | — | 2.836 (×1.48) | 5.0e-03 (2.7) |
| 768 | contiguous | keypad 7 % | 1.924 | 3.393 (×1.76) | — | 3.013 (×1.57) | 4.0e-03 (3.0) |
| 1024 | contiguous | bias_only | 4.319 | 6.236 (×1.44) | — | 6.508 (×1.51) | 5.2e-03 (4.0) |
| 1024 | contiguous | keypad 7 % | 4.101 | 7.799 (×1.90) | — | 6.429 (×1.57) | 3.8e-03 (3.7) |
| 1536 | contiguous | bias_only | 14.087 | 20.730 (×1.47) | — | 21.487 (×1.53) | 2.9e-03 (2.7) |
| 1536 | contiguous | keypad 7 % | 13.669 | 25.992 (×1.90) | — | 21.744 (×1.59) | 2.9e-03 (2.9) |
| 2048 | contiguous | bias_only | 32.977 | 48.182 (×1.46) | — | 49.529 (×1.50) | 2.9e-03 (2.3) |
| 2048 | contiguous | keypad 7 % | 31.016 | 61.647 (×1.99) | — | 48.784 (×1.57) | 3.9e-03 (2.7) |
| 3072 | contiguous | bias_only | 109.504 | 181.556 (×1.66) | — | 177.835 (×1.62) | 1.7e-03 (2.3) |
| 3072 | contiguous | keypad 7 % | 102.871 | 220.453 (×2.14) | — | 174.113 (×1.69) | 1.4e-03 (2.7) |
| 4096 | contiguous | bias_only | 258.303 | 428.803 (×1.66) | — | 437.190 (×1.69) | 1.6e-03 (3.6) |
| 4096 | contiguous | keypad 7 % | 241.449 | 530.312 (×2.20) | — | 427.232 (×1.77) | 1.8e-03 (3.2) |

#### cc 8.0 realdata grid, bf16 H4 D32: operands, bias and masks recorded from engine trunk passes (mask % = masked keys)

| dtype | H | D | B | N | S | layout | form | bias | route | package ms | k2b cell ms (×) | k10 (tri) ms (×) | package err: max abs (ulpe_max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | 4 | 32 | 1 | 917 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 14.724 | 21.297 (×1.45) | 19.140 (×1.30) | 3.2e-02 (1.8) |
| bf16 | 4 | 32 | 1 | 917 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 14.766 | 21.328 (×1.44) | 19.123 (×1.30) | 3.2e-02 (2.1) |
| bf16 | 4 | 32 | 1 | 917 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 14.774 | 21.289 (×1.44) | 19.117 (×1.29) | 4.2e-02 (1.1) |
| bf16 | 4 | 32 | 1 | 917 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 14.787 | 21.291 (×1.44) | 19.119 (×1.29) | 6.2e-02 (1.1) |
| bf16 | 4 | 32 | 1 | 917 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 14.663 | 19.491 (×1.33) | 19.083 (×1.30) | 3.3e-02 (1.8) |
| bf16 | 4 | 32 | 1 | 917 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 14.661 | 19.474 (×1.33) | 19.076 (×1.30) | 3.7e-02 (2.0) |
| bf16 | 4 | 32 | 1 | 917 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 14.836 | 19.514 (×1.32) | 19.104 (×1.29) | 6.0e-02 (1.1) |
| bf16 | 4 | 32 | 1 | 917 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 14.849 | 19.601 (×1.32) | 19.107 (×1.29) | 6.2e-02 (1.2) |
| bf16 | 4 | 32 | 1 | 992 | 992 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 4.192 | 6.160 (×1.47) | 5.679 (×1.35) | 1.9e-02 (2.2) |
| bf16 | 4 | 32 | 1 | 992 | 992 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 4.216 | 6.186 (×1.47) | 5.680 (×1.35) | 1.8e-02 (2.4) |
| bf16 | 4 | 32 | 1 | 992 | 992 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 4.235 | 6.189 (×1.46) | 5.680 (×1.34) | 1.8e-02 (1.3) |
| bf16 | 4 | 32 | 1 | 992 | 992 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 4.234 | 6.188 (×1.46) | 5.680 (×1.34) | 2.0e-02 (1.2) |
| bf16 | 4 | 32 | 1 | 992 | 992 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 4.260 | 6.201 (×1.46) | 5.705 (×1.34) | 3.8e-02 (2.7) |
| bf16 | 4 | 32 | 1 | 992 | 992 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 4.253 | 6.191 (×1.46) | 5.687 (×1.34) | 3.4e-02 (2.2) |
| bf16 | 4 | 32 | 1 | 992 | 992 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 4.230 | 6.190 (×1.46) | 5.677 (×1.34) | 6.3e-02 (1.0) |
| bf16 | 4 | 32 | 1 | 992 | 992 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 4.237 | 6.189 (×1.46) | 5.679 (×1.34) | 6.3e-02 (1.1) |
| bf16 | 4 | 32 | 1 | 992 | 992 | contiguous | recorded engine mask 7 % | real | cuda_80 | 4.199 | 5.530 (×1.32) | 5.667 (×1.35) | 2.0e-02 (2.5) |
| bf16 | 4 | 32 | 1 | 992 | 992 | contiguous | recorded engine mask 7 % | real | cuda_80 | 4.215 | 5.531 (×1.31) | 5.667 (×1.34) | 2.0e-02 (2.5) |
| bf16 | 4 | 32 | 1 | 992 | 992 | contiguous | recorded engine mask 7 % | real | cuda_80 | 4.226 | 5.569 (×1.32) | 5.666 (×1.34) | 3.1e-02 (1.7) |
| bf16 | 4 | 32 | 1 | 992 | 992 | contiguous | recorded engine mask 7 % | real | cuda_80 | 4.220 | 5.569 (×1.32) | 5.666 (×1.34) | 3.1e-02 (1.4) |
| bf16 | 4 | 32 | 1 | 992 | 992 | contiguous | recorded engine mask 7 % | real | cuda_80 | 4.213 | 5.530 (×1.31) | 5.665 (×1.34) | 4.2e-02 (2.4) |
| bf16 | 4 | 32 | 1 | 992 | 992 | contiguous | recorded engine mask 7 % | real | cuda_80 | 4.213 | 5.531 (×1.31) | 5.666 (×1.34) | 4.1e-02 (2.4) |
| bf16 | 4 | 32 | 1 | 992 | 992 | contiguous | recorded engine mask 7 % | real | cuda_80 | 4.231 | 5.529 (×1.31) | 5.664 (×1.34) | 6.3e-02 (1.4) |
| bf16 | 4 | 32 | 1 | 992 | 992 | contiguous | recorded engine mask 7 % | real | cuda_80 | 4.241 | 5.566 (×1.31) | 5.666 (×1.34) | 6.3e-02 (1.7) |
| bf16 | 4 | 32 | 1 | 1028 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 16.386 | 23.647 (×1.44) | 21.339 (×1.30) | 3.2e-02 (2.0) |
| bf16 | 4 | 32 | 1 | 1028 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 16.405 | 23.645 (×1.44) | 21.332 (×1.30) | 2.6e-02 (1.7) |
| bf16 | 4 | 32 | 1 | 1028 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 16.384 | 23.667 (×1.44) | 21.333 (×1.30) | 4.0e-02 (1.0) |
| bf16 | 4 | 32 | 1 | 1028 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 16.425 | 23.662 (×1.44) | 21.348 (×1.30) | 6.2e-02 (1.0) |
| bf16 | 4 | 32 | 1 | 1028 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 16.320 | 21.866 (×1.34) | 21.323 (×1.31) | 4.2e-02 (1.6) |
| bf16 | 4 | 32 | 1 | 1028 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 16.320 | 21.788 (×1.34) | 21.305 (×1.31) | 3.9e-02 (2.4) |
| bf16 | 4 | 32 | 1 | 1028 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 16.383 | 21.825 (×1.33) | 21.346 (×1.30) | 5.9e-02 (1.0) |
| bf16 | 4 | 32 | 1 | 1028 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 16.345 | 21.852 (×1.34) | 21.333 (×1.31) | 6.2e-02 (1.0) |
| bf16 | 4 | 32 | 1 | 1945 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 30.777 | 44.615 (×1.45) | 40.131 (×1.30) | 1.6e-02 (1.7) |
| bf16 | 4 | 32 | 1 | 1945 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 30.806 | 44.638 (×1.45) | 40.121 (×1.30) | 1.7e-02 (1.8) |
| bf16 | 4 | 32 | 1 | 1945 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 30.982 | 44.724 (×1.44) | 40.160 (×1.30) | 3.0e-02 (0.9) |
| bf16 | 4 | 32 | 1 | 1945 | 1945 | ending node, operands made contiguous | recorded engine mask 7 % | real | cuda_80 | 30.937 | 44.639 (×1.44) | 40.135 (×1.30) | 3.0e-02 (1.2) |
| bf16 | 4 | 32 | 1 | 1945 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 30.706 | 41.091 (×1.34) | 40.090 (×1.31) | 1.7e-02 (2.2) |
| bf16 | 4 | 32 | 1 | 1945 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 30.731 | 40.972 (×1.33) | 40.094 (×1.30) | 1.7e-02 (2.0) |
| bf16 | 4 | 32 | 1 | 1945 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 30.729 | 41.246 (×1.34) | 40.117 (×1.31) | 2.9e-02 (1.1) |
| bf16 | 4 | 32 | 1 | 1945 | 1945 | contiguous | recorded engine mask 7 % | real | cuda_80 | 30.722 | 41.028 (×1.34) | 40.104 (×1.31) | 2.0e-02 (1.2) |

#### cc 8.0 secondary grid (generic fp32 bias; other head dims / H / B / N ≠ S / interior holes / fp16)

| dtype | H | D | B | N | S | layout | form | bias | route | package ms | k2b cell ms (×) | k10 (tri) ms (×) | package err: max abs (ulpe_max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | 4 | 16 | 1 | 256 | 256 | contiguous | bias_only | fp32 | cuda_80 | 0.092 | 0.098 (×1.07) | 0.112 (×1.21) | 6.4e-03 (2.6) |
| bf16 | 4 | 16 | 1 | 512 | 512 | contiguous | bias_only | fp32 | cuda_80 | 0.480 | 0.609 (×1.27) | 0.691 (×1.44) | 4.7e-03 (3.0) |
| bf16 | 4 | 16 | 1 | 1024 | 1024 | contiguous | bias_only | fp32 | cuda_80 | 3.293 | 4.460 (×1.35) | 5.011 (×1.52) | 4.0e-03 (3.4) |
| bf16 | 4 | 16 | 1 | 2048 | 2048 | contiguous | bias_only | fp32 | cuda_80 | 24.695 | 35.106 (×1.42) | 39.814 (×1.61) | 3.4e-03 (2.7) |
| bf16 | 4 | 32 | 1 | 255 | 255 | contiguous | bias_only | fp32 | cuda_80 | 0.122 | 0.140 (×1.15) | 0.166 (×1.36) | 5.1e-03 (3.1) |
| bf16 | 4 | 32 | 1 | 256 | 256 | contiguous | mask_bias (interior holes) 7 % | fp32 | cuda_80 | 0.131 | 0.164 (×1.25) | 0.158 (×1.20) | 5.8e-03 (2.7) |
| bf16 | 4 | 32 | 1 | 256 | 256 | contiguous | mask_bias 7 % | fp32 | cuda_80 | 0.173 | 0.164 (×0.95) | 0.214 (×1.24) | 4.8e-03 (2.7) |
| bf16 | 4 | 32 | 1 | 257 | 257 | contiguous | bias_only | fp32 | cuda_80 | 0.156 | 0.266 (×1.71) | 0.247 (×1.59) | 5.5e-03 (2.9) |
| bf16 | 4 | 32 | 1 | 257 | 257 | contiguous | keypad 7 % | fp32 | cuda_80 | 0.162 | 0.301 (×1.86) | 0.220 (×1.36) | 4.5e-03 (3.1) |
| bf16 | 4 | 32 | 1 | 512 | 512 | contiguous | mask_bias (interior holes) 7 % | fp32 | cuda_80 | 0.636 | 1.046 (×1.65) | 0.943 (×1.48) | 4.8e-03 (2.9) |
| bf16 | 4 | 32 | 1 | 512 | 512 | contiguous | mask_bias 7 % | fp32 | cuda_80 | 0.799 | 1.044 (×1.31) | 1.266 (×1.58) | 4.8e-03 (3.0) |
| bf16 | 4 | 32 | 1 | 1000 | 1000 | contiguous | bias_only | fp32 | cuda_80 | 4.139 | 6.156 (×1.49) | 6.740 (×1.63) | 5.1e-03 (4.3) |
| bf16 | 4 | 32 | 1 | 1000 | 1000 | contiguous | keypad 7 % | fp32 | cuda_80 | 3.933 | 8.069 (×2.05) | 6.366 (×1.62) | 5.3e-03 (3.9) |
| bf16 | 4 | 32 | 1 | 1024 | 1024 | contiguous | mask_bias (interior holes) 7 % | fp32 | cuda_80 | 4.257 | 7.751 (×1.82) | 6.855 (×1.61) | 6.9e-03 (3.7) |
| bf16 | 4 | 32 | 1 | 1024 | 1024 | contiguous | mask_bias 7 % | fp32 | cuda_80 | 5.074 | 7.745 (×1.53) | 8.859 (×1.75) | 5.5e-03 (4.1) |
| bf16 | 4 | 32 | 1 | 1537 | 1537 | contiguous | bias_only | fp32 | cuda_80 | 15.574 | 23.077 (×1.48) | 24.995 (×1.60) | 2.7e-03 (2.8) |
| bf16 | 4 | 32 | 1 | 2048 | 2048 | ending node, operands made contiguous | mask_bias (interior holes) 7 % | fp32 | cuda_80 | 32.712 | 60.989 (×1.86) | 52.036 (×1.59) | 2.5e-03 (3.2) |
| bf16 | 4 | 32 | 1 | 2048 | 2048 | contiguous | mask_bias (interior holes) 7 % | fp32 | cuda_80 | 32.744 | 61.157 (×1.87) | 52.029 (×1.59) | 3.2e-03 (3.0) |
| bf16 | 4 | 32 | 1 | 2048 | 2048 | contiguous | mask_bias 7 % | fp32 | cuda_80 | 32.772 | 60.815 (×1.86) | 68.638 (×2.09) | 4.3e-03 (3.2) |
| bf16 | 4 | 32 | 2 | 256 | 256 | contiguous | bias_only | fp32 | cuda_80 | 0.201 | 0.244 (×1.21) | 0.262 (×1.31) | 5.2e-03 (3.3) |
| bf16 | 4 | 32 | 2 | 512 | 512 | contiguous | bias_only | fp32 | cuda_80 | 1.206 | 1.645 (×1.36) | 1.725 (×1.43) | 4.9e-03 (3.0) |
| bf16 | 4 | 32 | 2 | 1024 | 1024 | contiguous | bias_only | fp32 | cuda_80 | 8.519 | 12.347 (×1.45) | 13.038 (×1.53) | 4.0e-03 (4.1) |
| bf16 | 8 | 32 | 1 | 256 | 256 | contiguous | bias_only | fp32 | cuda_80 | 0.201 | 0.243 (×1.21) | 0.261 (×1.30) | 5.3e-03 (3.3) |
| bf16 | 8 | 32 | 1 | 512 | 512 | contiguous | bias_only | fp32 | cuda_80 | 1.202 | 1.643 (×1.37) | 1.723 (×1.43) | 5.2e-03 (3.6) |
| bf16 | 8 | 32 | 1 | 1024 | 1024 | contiguous | bias_only | fp32 | cuda_80 | 8.523 | 12.313 (×1.44) | 13.059 (×1.53) | 6.6e-03 (4.1) |
| bf16 | 8 | 32 | 1 | 2048 | 2048 | contiguous | bias_only | fp32 | cuda_80 | 65.611 | 95.580 (×1.46) | 98.652 (×1.50) | 2.9e-03 (3.4) |
| bf16 | 4 | 64 | 1 | 256 | 256 | contiguous | bias_only | fp32 | cuda_80 | 0.177 | 0.213 (×1.20) | 0.230 (×1.30) | 5.5e-03 (3.2) |
| bf16 | 4 | 64 | 1 | 512 | 512 | contiguous | bias_only | fp32 | cuda_80 | 1.060 | 1.335 (×1.26) | 1.500 (×1.42) | 5.4e-03 (3.5) |
| bf16 | 4 | 64 | 1 | 1024 | 1024 | contiguous | bias_only | fp32 | cuda_80 | 7.366 | 10.463 (×1.42) | 11.423 (×1.55) | 3.8e-03 (3.3) |
| bf16 | 4 | 64 | 1 | 2048 | 2048 | contiguous | bias_only | fp32 | cuda_80 | 56.028 | 77.939 (×1.39) | 85.735 (×1.53) | 1.9e-03 (2.3) |
| fp16 | 4 | 32 | 1 | 256 | 256 | contiguous | bias_only | fp32 | tri | 0.141 | 0.130 (×0.92) | (= package) | 5.0e-04 (2.0) |
| fp16 | 4 | 32 | 1 | 256 | 256 | contiguous | keypad 7 % | fp32 | tri | 0.155 | 0.164 (×1.06) | (= package) | 5.0e-04 (1.9) |
| fp16 | 4 | 32 | 1 | 512 | 512 | contiguous | bias_only | fp32 | tri | 0.896 | 0.848 (×0.95) | (= package) | 5.0e-04 (2.0) |
| fp16 | 4 | 32 | 1 | 512 | 512 | contiguous | keypad 7 % | fp32 | tri | 0.930 | 1.055 (×1.13) | (= package) | 5.0e-04 (2.3) |
| fp16 | 4 | 32 | 1 | 1024 | 1024 | contiguous | bias_only | fp32 | tri | 6.955 | 6.203 (×0.89) | (= package) | 5.0e-04 (2.7) |
| fp16 | 4 | 32 | 1 | 1024 | 1024 | contiguous | keypad 7 % | fp32 | tri | 6.353 | 7.814 (×1.23) | (= package) | 5.0e-04 (2.6) |
| fp16 | 4 | 32 | 1 | 2048 | 2048 | contiguous | bias_only | fp32 | tri | 50.931 | 48.468 (×0.95) | (= package) | 2.0e-04 (1.8) |
| fp16 | 4 | 32 | 1 | 2048 | 2048 | contiguous | keypad 7 % | fp32 | tri | 48.062 | 61.583 (×1.28) | (= package) | 2.0e-04 (1.8) |

`tools/make_cells.py show` prints the unmasked starting-node rows per cc.

## Binaries (prebuilt/)

The CUDA routes are torch C++ extensions, specific to (torch version + CUDA major, CPython minor).  `triattn_pkg/prebuilt/<torch version>-<SOABI>/`
ships all four per stack; `prebuilt/INDEX.json` lists them with versions, digests and what was verified; each `<ext>.json` beside an `.so` is its
build record (kernel directory, torch / CUDA / python / nvcc, the ptxas that assembled it, CUTLASS tag where used, `-gencode` flags, `so_sha256`,
`source_sha256` over the directory's `csrc/`, `module_sha256`, and `loadcheck` = {test-vector id: sha256/16 of the output bytes on a device of the
extension's architecture}).  Output bytes are identical across stacks (equal `loadcheck` digests).

| stack tag | sm_90a: `triattn_sm90_ext`, `triattn_mw_ext_g3x4`, `triattn_m1_ext` | sm_80: `triattn_sm80_ext` |
|---|---|---|
| `torch2.12.0+cu130-cpython-312-x86_64-linux-gnu` (the stack of the cells and vectors) | nvcc 13.0.88 (m1: ptxas 13.4, CUTLASS 4.7.1); H100 80GB HBM3: test_pkg 17/17 bitwise, == JIT builds == the live router | nvcc 13.0.88; A100-SXM4-80GB: test_pkg 22/22 bitwise, == JIT build == the live router |
| `torch2.12.0+cu130-cpython-311-x86_64-linux-gnu` | nvcc 13.0.88; H100: 17/17 | nvcc 13.0.88; A100 80GB: 22/22 |
| `torch2.13.0+cu130-cpython-312-x86_64-linux-gnu` | nvcc 13.0.88; H100: 17/17 | nvcc 13.0.88; A100 80GB: 22/22 |
| `torch2.13.0+cu130-cpython-311-x86_64-linux-gnu` | nvcc 13.0.88; H100: 17/17 | nvcc 13.0.88; A100 80GB: 22/22 |
| `torch2.10.0+cu128-cpython-311-x86_64-linux-gnu` | nvcc 12.8; H100: 17/17 | nvcc 12.8; A100 80GB: 22/22 |
| `torch2.7.1+cu128-cpython-311-x86_64-linux-gnu` | nvcc 12.8; H100: 13/13 servable (the 4 S ≤ 256 vectors route to k13 = Gluon: `needs_triton_gluon` on this stack's Triton 3.3) | nvcc 12.8; A100 80GB: 22/22 |
| `torch2.7.1+cu126-cpython-311-x86_64-linux-gnu` | nvcc 12.6; H100: 13/13 servable (as above) | nvcc 12.6; A100 80GB: 22/22 |

Resolution at the first call of a CUDA route: the prebuilt for this process's stack tag if present (default), else a JIT build through the kernel
directory's own `_build()` when `nvcc` + `ninja` (+ CUTLASS ≥ 4 headers at `$CUTLASS_PATH` for `cuda` / `cuda_b`; `cuda_80` and `cuda_c` need no
third-party headers) are present (minutes, cached under `TORCH_EXTENSIONS_DIR`), else `Unsupported` naming the missing binary.
`TRIATTN_PKG_PREBUILT=never` forces the JIT build, `=always` forbids it.  `cuda` / `cuda_c` / `cuda_b` and the Gluon kernels are **sm_90a** code;
`cuda_80` is **sm_80** code, built with `-gencode arch=compute_80,code=sm_80` and routed on cc 8.0 only (cc 8.6 / 8.9 devices can execute sm_80 code
but are not measured, so the router keeps them on `tri`).  Other stacks: `python tools/build_prebuilt.py --exts all` inside the target interpreter
writes `prebuilt/<its tag>/` with the same records (`TORCH_CUDA_ARCH_LIST` is scoped per extension during the build).

## Test vectors (testvectors/, test_pkg.py)

39 cases in `testvectors/cases.py`, each naming the compute capabilities it is a vector for.  cc 9.0: the 17 cases of generation 10 (same ids, seeds,
input digests, expected bytes, reference rows and recorded errors; made on H100 80GB HBM3): bf16 H4 D32 unless noted, S ∈ {256, 384, 512, 1024, 2048,
3584} with a few pair rows each plus one square N = S = 256 case, masks none / prefix / prefixvar, bf16-representable fp32 bias, generic fp32 bias,
bias handed as bf16, ending-node strided views at S = 384 and 1024, one fp16 and one D = 64 case — every cc-9.0 route is exercised.  cc 8.0: 22
`a80_*` cases on the `cuda_80` route (made on A100-SXM4-80GB with the shipped `triattn_sm80_ext`, torch 2.12.0+cu130): bf16 D32 H4 S ∈ {256, 400,
512, 1024, 2048} none / prefix / prefixvar, a square N = S = 400 case, generic fp32 bias, bias as bf16, strided views at S = 384 and 1024, D = 16 (S
512), D = 64 (S 512 contiguous, S 1024 strided), H = 8 (S 800, ragged masks), N = 6 rows of S = 1000, B = 2 (S 400).  Inputs are regenerated from a
CPU seed and checked against recorded SHA-256 digests; stored per case are the package's output bytes on the case's device class
(`<id>.expected.pt`), pair row 0 of the fp64 reference (`<id>.ref_rows.pt`), and in `manifest.json` the route, the output digest and the full-output
error vs fp64 (`made` records the generating interpreter and device per cc).

```
python test_pkg.py                                   # this device's cc: bitwise == expected, error gate vs fp64 (<= 1.5x recorded), route == recorded; exit 0 = pass
TRIATTN_PKG_PREBUILT=never python test_pkg.py        # same with the CUDA extensions JIT-built here
python test_pkg.py --impl dispatch:<tree>/dispatch   # another implementation of the contract against the same vectors
python test_pkg.py --list [--cc 8.0]                 # the case table and what is recorded
python test_pkg.py --generate --note "..."           # maintainers, on a device of the cc: rewrite THAT cc's vectors from the package's outputs (other cc untouched)
python test_pkg.py --record-loadcheck                # maintainers, on a device of the cc: fill pending loadcheck digests of this stack's build records
python tools/check_prebuilt.py [--write-index]       # digests of every prebuilt record vs the files and sources here; no GPU
```

## Requirements and hazards

* NVIDIA H100 (cc 9.0) or A100 (cc 8.0) for the measured routes; torch ≥ 2.7 with CUDA 12.6 / 12.8 / 13.x (prebuilts for the seven stacks above);
  Triton ≥ 3.6 with `triton.experimental.gluon` for the cc-9.0 Gluon routes (k13 at S < 512 contiguous bf16 D32 and S ≤ 640 fp16 / D16; k12 above) —
  on older Triton those calls are refused by name (`needs_triton_gluon`), every other route is unaffected.  Importing `triattn_pkg.Unsupported` or
  making the first call loads torch + triton.
* **Module namespace.** The router puts `triattn_pkg/{dispatch,triton,cuda,cuda_c,cuda_b,cuda_80}` on `sys.path` and imports the kernel directories
  under their own top-level names (`triattn`, `kernels`, `pins`, `candidate_tri`, `triattn_cuda`, `triattn_mw`, `triattn_m1`, `triattn_sm80`, plus the
  extension modules `triattn_sm90_ext`, `triattn_mw_ext_g3x4`, `triattn_m1_ext`, `triattn_sm80_ext`).  A host process with its own top-level
  `kernels` or `triattn` module collides; a consumer that needs isolation loads the package under a private importer.
* Leave `MW_GEOM`, `MW_TIMELINE`, `TRIATTN_M1_FLAGS`, `PTX_TRIATTN_CONFIG` unset (`TRIATTN_PTXAS=image` is the one supported knob: cuda_b's JIT
  build then uses the toolkit ptxas): they select development variants and the router refuses them by name.
* First-call latency: Triton compiles k13 / tri per new (dtype, D, mask-kind) class (seconds); a CUDA route without a prebuilt compiles for minutes.
* cc 9.0: S > 4096 routes to `cuda_b` and is served but lies outside the measured cells; B > 1, H ≠ 4 are served but the cells are B = 1, H = 4.
  fp32 pair bias below S = 512 (routes k13 / cuda) is rounded to bf16 before use (≤ 2⁻⁹ relative on the logits); from S = 512 up it is exact.
* cc 8.0: transposed views are addressed with 64-bit offsets at every size; fp16 and D = 128 are served by `tri` (k10), where the k2b Triton kernel of
  opt_core is faster (cells above); the memory bound of a call is the operands themselves (see Measured cells).

## Rebuilding

```
python tools/build_prebuilt.py --exts all                                  # inside the target stack: build the four extensions, write digest records, load-check on a device of each arch if present
python test_pkg.py --record-loadcheck                                      # on an H100 and on an A100: fill the pending loadcheck digests
python test_pkg.py --generate --note "..."                                 # test vectors of this device's cc
python tools/make_cells.py append --cc 8.0 --runs <timing result files>   # cells (result-file schema triattn_bench/v2)
python tools/check_prebuilt.py --write-index                               # check records, regenerate prebuilt/INDEX.json
```

## Pins

`triattn_pkg.PINS`: each kernel directory is the tree's directory at the commit that sealed it, unmodified (runtime files only); `BEST_COMMIT` =
`29f5831ac67` is the router's.

| route | directory | commit |
|---|---|---|
| `tri` | `triattn_pkg/triton/` | `HEAD` |
| `k13` | `triattn_pkg/dispatch/kernels/` | `HEAD` |
| `cuda_b` | `triattn_pkg/cuda_b/` | `6fb95c4b646` |
| `cuda_c` | `triattn_pkg/cuda_c/` | `d424d883727` |
| `cuda` | `triattn_pkg/cuda/` | `010f83b752a` |
| `cuda_80` | `triattn_pkg/cuda_80/` | `20bbc5dc491` |

## Files

| path | what |
|---|---|
| `triattn_pkg/__init__.py`, `_face.py` | the face: `triangle_attention`, `route`, `Unsupported`, `prebuilt_status`, `stack_tag`, `PINS`, `BEST_COMMIT`, `CUDA_SEATS`; loads the router by path, resolves binaries |
| `triattn_pkg/dispatch/` | `candidate.py` = the router `best` (verbatim from the tree's `dispatch/`), `pins.py`, `kernels/k13.py` |
| `triattn_pkg/triton/` | `candidate.py` = `tri`; `triattn/` k10, k11, k12, launch, masking, errors (`Unsupported`) |
| `triattn_pkg/cuda/`, `cuda_c/`, `cuda_b/` | the sm_90a kernel directories: python entry (`triattn_cuda.py` / `triattn_mw.py` / `triattn_m1.py`) + `csrc/` |
| `triattn_pkg/cuda_80/` | the sm_80 member: `triattn_sm80.py` + `csrc/` |
| `triattn_pkg/prebuilt/<stack tag>/` | `<ext>.so` + `<ext>.json` build record, four per stack, seven stacks; `INDEX.json` |
| `testvectors/` | `cases.py` (case table with `cc`, input generator, fp64 reference), `manifest.json`, `<case>.expected.pt`, `<case>.ref_rows.pt` |
| `test_pkg.py` | verify / cross-implementation compare / list / regenerate per cc / record load-check digests |
| `CELLS.json`, `tools/make_cells.py` | measured cells keyed by cc and their generator |
| `tools/build_prebuilt.py`, `tools/check_prebuilt.py` | the in-stack extension builder; the record verifier + INDEX writer |
