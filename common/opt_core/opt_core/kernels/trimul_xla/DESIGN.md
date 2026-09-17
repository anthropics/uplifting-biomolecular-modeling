# kernels.trimul_xla — design note (row `native_xla`: FlashPairformer triangle multiplication for JAX / XLA programs)

## Op and calling convention
The bias-free TriangleMultiplication (the `kernels.trimul.native` op boundary: pre-LayerNorm `z [N,N,c_z]` or `[B,N,N,c_z]`, bf16 or fp32
(`f32z` form = fp32-resident z under bf16 compute), mask `[.., N, N]` 0/1 or None; `update` or `z + update` in z's dtype) for jax programs,
on the SEALED `trimul_native` sm_90a cubins carried under `pkg/v1/` (`build/sm_90a/tmn90_z64_h64.cubin`, `tmn90_z128_h128.cubin`: c_z =
c_hidden = 64 and 128), launched through the XLA-FFI launcher of `kernels/triattn_xla` (generic target `xla_cubin_call`).

    out = triangle_multiplication(z, mask, weights=w10 | pack, direction="outgoing" | "incoming", residual=False)
    w10 = weights_from_math_layout(trimul_params)      # kernels.pallas' math layout -> the ten canonical tensors (torch Linear layout [out, in])

Refused BY NAME (`Refused(kind)`, fallback `kernels.pallas` row `cd_trimul` named): modules with projection / gate / output biases (`biases`),
capabilities other than 9.0 (`arch`: the sm_80 units are not carried in this version), sizes outside 16 <= N <= 4096, a jax without `jax.ffi`
or no launcher build, a digest mismatch, `MODEL_OPT_LEVERS_OFF` words `trimul_xla` / `trimul_xla:native_xla`. FORWARD ONLY: jax.grad / vjp / jvp
through the entry raise `Refused` naming the differentiable fallback; vmap = a per-sample rule (`jax.custom_batching`) where the running jax
has it, else the ffi calls' sequential rule (`report()["vmap"]`).

## Implementation
K1 (LN_in + the four gated projections + mask -> channel-major bf16 planes `ab[2 c_h, Np, Np]`, Np = ceil16(N)) | the contraction as ONE
strided-batched cuBLAS GEMM on the two halves of the plane buffer through the launcher target `xla_cublas_bgemm_nt` (bf16 operands, fp32
compute, bf16 out -- the call the torch package issues through `torch.bmm`; XLA's own dot when that target is unavailable, decided by
`selfcheck()` in an eager context) | K3 (LN_out + output projection x sigmoid(gate) [+ residual]). The kernels, tile configurations and
shared-memory formulas are the package's: `tile_cfg(unit, form, N)` = the unit's defaults overridden by the smallest covering N bucket of
`CELLS.json["units"]` (restated from the package's tile table), `k1_name` / `k3_name` spell the `extern "C"` symbol, `k1_smem` / `k3_smem`
equal the csrc `K1Cfg::SMEM` / `K3Cfg::SMEM`. `k1_launch` / `k3_launch` produce the launcher's token lists (buffers, by-value parameter block
layout, tensor-map descriptors, grid = persistent CTAs, block, dynamic shared memory); `pack_weights` builds the kernels' operand forms in jnp,
traced with the program (w1 `[4 c_h, c_z]` bf16 = 32-row blocks interleaving [w_ag; w_bg] and [w_ap; w_bp], ...).

## Numerics
The cubins' class (see `kernels/trimul/native/DESIGN.md`, fast variant): fp32 LayerNorm statistics, LN output rounded once to bf16, bf16 x
bf16 -> fp32 MMAs, approximate sigmoid, gate*proj rounded once, residual per z dtype. K1's planes and K3's output equal the torch package's
launch of the same cubins bit for bit on identical inputs; with the cuBLAS target the contraction is the same GEMM call, so the end-to-end
output equals the torch row's (`vectors.json` records the cases; another cuBLAS release may order that GEMM's sums differently -- then equal up
to bf16 rounding of the same fp32 sums). `reference(...)` is the op statement in jnp, fp32 throughout; `forward_with_intermediates` and
`k3_only` expose the stages for stage-wise comparison.

## Loading and gates
`pkg/v1/` is sealed: `SHA256SUMS` lists every carried file and `cubin_entry(unit)` checks the cubin's digest against it once per process before
the launcher registers it; `manifest()` / `package_version()` read `build/manifest.json` / `VERSION`. `load()` imports the launcher from
`kernels/triattn_xla` (its own digest-checked binaries), registers this package's root, and caches the handle; failures are `Refused` by name,
never a rebuild (nothing is compiled at install or run time). `report()` states version, units + digests, launcher state, counts and the vmap
rule without a GPU call. Selection data: `CELLS.json` ("units" = tile table, "measured" = this row per (cc, jax line, dtype, width, direction, N)).

## Known limits
cc 9.0 only; c_z = c_hidden in {64, 128}; 16 <= N <= 4096; bias-free modules only; forward only; the contraction target needs the launcher's
cuBLAS entry (else XLA's dot: same class, not the same bits). Memory per call as the native package at batch 1 (z + output + planes
`[2 c_h, Np, Np]` + contraction `[c_h, Np, Np]`).
