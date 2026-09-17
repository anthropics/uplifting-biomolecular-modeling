# FlashPairformer — triangle multiplication (trimul_native)

## Op and calling convention
The triangle multiplicative update, outgoing and incoming, forward only, served from a sealed payload (`pkg/v5/`, package
`trimul_native`) of arch-keyed CUDA cubins behind the face in `__init__.py` (`admits` / `install` / `pack` / `forward`).

    z [N,N,c_z] | [B,N,N,c_z] = PRE-LayerNorm pair tensor, mask [N,N] | [B,N,N] 0/1 or None
    x = LN_in(z);  a = mask*sigmoid(x W_ag^T)*(x W_ap^T);  b = mask*sigmoid(x W_bg^T)*(x W_bp^T)
    X[i,j,:] = sum_k a[i,k,:] b[j,k,:] (outgoing) | sum_k a[k,i,:] b[k,j,:] (incoming)
    update = sigmoid(x W_og^T) * (LN_out(X) W_o^T);  forward() returns update, or z + update with residual=True

Weights are the ten canonical tensors (`ln_in_w ln_in_b w_ag w_ap w_bg w_bp ln_out_w ln_out_b w_o w_og`), packed once per weight set
(`pack`, cached by tensor identity). Forms: bf16 z (row `native`); fp32-resident z computed in bf16 (K1 reads the fp32 values; row
`native:f32in` is the word for fp32 / tf32 callers outside autocast and refuses bf16 callers by name; under bf16 autocast the bf16 rows apply);
the exact variant (row `native_exact`, bf16 z, c_z == c_hidden, batch 1). Channel widths c_z, c_hidden in {64, 128, 256, 384}; the (arch, c_z,
c_hidden, form, variant) combinations served are exactly the `serves` records of `pkg/v5/build/manifest.json`; N 16..4096 (ragged sizes
included), B <= 64 (B > 1 = one launch set per pair plane). Anything else is refused BY NAME before any launch, never substituted:
`digest:<relpath>` `payload:` `import:` `driver_unavailable:<why>` `cc_unsupported:<cc>` `no_cubin:<arch>` `manifest:<what>` `load:<driver error>`
`loadcheck_failed:<unit.kernel|case>` `vectors:<what>` `shape:` `dtype:` `direction:` `weights:` `variant:<what>` `no_cell:<key>`
`exact_small_n:<N>` `exact_no_reference:<c_z>x<c_hidden>`.

## Implementation
Three stages per pair plane: **K1** prologue kernel (LN_in + four gated projections + mask -> channel-major bf16 planes `[2*c_hidden, Np, Np]`,
Np = ceil16(N), zero pads written by K1, no memset) | **one strided-batched cuBLAS GEMM** on the planes through `torch.bmm` (bf16 operands, fp32
accumulate, bf16 out) | **K3** epilogue kernel (LN_in recomputed for the output gate + LN_out of the contraction planes + W_o projection x
sigmoid(gate) [+ residual] -> `[N, N, c_z]` in z's dtype; an update-only specialisation `_u` has the residual compiled out). Incoming: K1 reads
z with the token axes swapped and writes transposed planes, so both directions run the same NT GEMM (`kernel.INCOMING_MODE = "kt"`).
One kernel family (`csrc/tmn_kernels.cuh`); widths, z dtype, tile shape, weight-ring depth, LayerNorm summation mode and mask on/off are
template parameters, and every instantiation is an `extern "C"` symbol whose name encodes its configuration
(`tmn_k1_z<CZ>_h<CH>_<b|f>_t<BI>x<BJ>_s<NSLOT>k<SKCH>_m<0|1>_l<LNM>_v<0|1>`, `tmn_k3[w]_z<CZ>_h<CH>_<b|f|p|c>_t<BI>x<BJ>_s<NSLOT>a<NACC>_l<LNM>`);
`kernel.py` selects by name from its tile table (`TILE_TABLE`: per (arch, c_z, c_hidden, form) default K1 (BI, BJ, NSLOT, SKCH) and K3
(BI, BJ, NSLOT, NACC) configurations with per-N buckets; the payload's `CELLS.json` may name the configuration per measured cell).

* **sm_90a member** (units `tmn90_z<c_z>_h<c_hidden>`, one cubin per width pair; wide units `tmn90w_*`): persistent CTAs of one producer
  warpgroup (TMA bulk-tensor loads into 128B-swizzled shared memory, mbarrier rings of NSLOT weight slots) and BI*BJ/64 consumer warpgroups
  (ldmatrix -> LayerNorm on the A fragments in registers -> `wgmma` m64n64k16 bf16 -> fp32 with A from registers and the [64 n x 64 k] weight chunk
  from shared memory -> fused epilogue, software-pipelined one weight block ahead so the epilogue of block b overlaps the MMAs of block b+1).
  Tiles read through the generic proxy are released to the producer after `fence.proxy.async`. Wide-K K3 (`csrc/tmn_k3_wide.cuh`, pairs whose two
  MMA operands cannot both stay register-resident: c_z/16 + c_hidden/16 > 24 k-steps, i.e. the pairs containing 384): the X sub-tile is
  LayerNorm'd in registers once and written back token-major into the same shared tile in the canonical K-major `wgmma` layout, the projection
  MMAs read it through a matrix descriptor (SS) while the gate operand stays register-sourced (RS); ring slots alternate projection and gate blocks.
* **sm_80 member** (`trimul_k1_sm80`, `trimul_k3_sm80`; loads on cc 8.0 / 8.6 / 8.7 / 8.9): grid (token tiles, plane row, batch), BM tokens per
  CTA, BM/WM warps; the LayerNormed A rows are register-resident in `mma.sync` m16n8k16 fragment order straight from global memory (16-byte loads +
  a quad transpose), weight chunks of BN output channels stream through a `cp.async` ring of STAGES slots, B fragments by `ldmatrix` from the same
  128B-swizzled chunk layout, CTA-wide epilogue staging -> 16-byte channel-major stores. K3 stages the X tile with `cp.async`, reads it with
  `ldmatrix.trans` into A fragments (LN_out on the fragments) and re-uses the tile's bytes for the LN_in(z) rows. Tile points: `sm80_ops.TILES` /
  `TILES_EXACT`.
* Shared by both members: the numerics statement (`csrc/common/tmn_math.cuh`), the LayerNorm fragment math, the epilogue arithmetic and the POD
  parameter blocks (`K1Params` / `K3Params`: tensor maps first, then pointers, then scalars; `tmn_info` reports sizeof/offsetof, which the
  Python packer checks at load).

## Numerics
Fast variant (`native`, `native:f32in`): the rounding sequence of cuequivariance_torch 0.11.1 `triangle_multiplicative_update` up to reduction
order. LayerNorm statistics fp32 per row (two-pass, centred variance, `rsqrt.approx`), affine in fp32, output rounded ONCE to bf16; bf16 x bf16 ->
fp32 MMAs whose accumulators are never rounded before the gate; sigmoid(g) = rcp.approx(1 + ex2.approx(-g log2 e)); gate*proj(*mask) rounded once
to bf16 (the planes; the update); residual: bf16 z -> bf16(z + update) rounded once, fp32 z -> the fp32 sum. Every operation of the statement is an
explicit round-to-nearest fp32 instruction or a named approximate one (nothing for the compiler to contract or reorder); `--use_fast_math` is off.
fp32-resident z: statistics and normalisation read the fp32 values (never pre-rounded); `residual=False` returns the bf16 update, `residual=True`
the fp32 sum; at (256, 256) the fp32 form casts z to bf16 once, runs the bf16 K1, and K3 mode `c` normalises the cast tile and adds the fp32
residual from global memory.

Exact variant (`native_exact`): bitwise identity with cuequivariance_torch 0.11.1 comes from three choices, all inside K1 / K3 and the op
assembly (`ops.trimul_plane_exact`, `sm80_ops._serve_exact`): (1) both LayerNorms are evaluated in the reference library's fp32 summation trees
(name field `l2` / `l3`; the transposing LN_out has two trees, selected by N % 4); (2) K1 writes UNPADDED planes `[2*c_hidden, N, N]` in the (i, k)
orientation for both directions and the contraction is issued as the reference `einsum` expression on `[c_hidden, 1, N, N]` chunk views, hence
the same cuBLAS problem, kernel and bits (N % 8 != 0: planes are written 16-token-padded and packed by one strided copy, and the result is pad-copied
once for K3's 16-byte rows -- both copies are bitwise-neutral); (3) projection / gating / residual arithmetic in the reference order, with the
module's separate residual add. Identity holds by construction at c 256 and is a measured property at the other square widths; the provider
serves the word `exact` through this row only where `TRIMUL_CELLS.json` records the bitwise result for the caller's stack, refuses it below 101
tokens (the reference library takes its own torch path there) and at c_hidden != c_z (no reference result), and names `tmk3_exact` as fallback.

## Loading, gates, cells
`DIGESTS` in `__init__.py` pins the sha256 of the payload's `SHA256SUMS`, `VERSION`, `build/manifest.json` and `testvectors/manifest.json`; every
`SHA256SUMS` line is re-hashed and no unlisted file may sit under `python/ csrc/ build/ testvectors/ tests/` (else `digest:<relpath>`). The
payload's Python is imported by path under a private module name, never as `trimul_native` from `sys.path`. `install()` runs once per (process,
device): driver binding (the `cuda.bindings` wheel when importable with the tensor-map API, else ctypes over `libcuda.so.1`) -> capability ->
`cuModuleLoadData` of the architecture's cubins into the framework's primary context (cubin digests from the build manifest) -> load check
(registers / local bytes of every kernel == the ptxas record) -> byte gate (closed-form input tensors regenerated in row slabs, outputs bitwise ==
the digests recorded for the device class `sm90` | `sm80`). A passed gate is stamped under `OPT_CORE_VERDICT_DIR`, keyed by the payload digest and
the stack facts, so later processes skip the replay; the outcome is printed as `[opt_core] NATIVE_STAMP ...`. Launches: `cuLaunchKernel` on
`torch.cuda.current_stream()`, by-value parameter blocks patched per call, `CUtensorMap` descriptors encoded host-side and keyed by address,
workspaces from torch's caching allocator. Cells: the payload's `CELLS.json` keys (cc, form, class, c_z, c_hidden, direction, residual, N bucket)
to a configuration; the nearest bucket in log N within x1.5 serves silently, a farther one serves with the notice `UNCOVERED_CELL:...` once per key,
no cell of that class is `no_cell:<key>` -- never inherited across width, class, card or call form.

## Known limits
Memory per call at batch 1 = z + output + planes `[2*c_hidden, Np, Np]` + contraction result `[c_hidden, Np, Np]` (about 10*c*N^2 bytes for
bf16 z and 14*c*N^2 for fp32-resident z at c_z = c_hidden = c); the planes + result of one (unit, Np) stay in the caller's cache only while they
total <= 256 MiB (`ops.WORKSPACE_PAIR_KEEP_BYTES`), larger ones are transient; weight packs are cached per weight identity with a bounded count
(`OPT_CORE_TRIMUL_PACK_CACHE`, one resident pack under the tier word big). No host synchronisation in a serve; CUDA-graph capture is possible
once the modules are loaded (run `install()` or one eager call first). Outputs are run-to-run reproducible on a device class (the byte gate
replays recorded bytes). Forward only; sm_90a and sm_80 SASS only (other capabilities: `cc_unsupported`); the sm_80 exact word only at the square
pairs compiled into `TILES_EXACT`.
