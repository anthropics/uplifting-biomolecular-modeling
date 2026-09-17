# kernels.trimul.esm_v61 — design note (row `esm_v61`; sealed payload `pkg/v6.1/`, package `esmfold2_trimul`)

## Op and calling convention
TriangleMultiplication forward for bf16 pair tensors, c_z = c_hidden = D = 256, compute capability 9.0 only:

    zn = LN_in(z);  a|b = (zn Wp^T) * sigmoid(zn Wg_in^T) [* mask]     (Wp, Wg_in [512, 256]: rows [0,256) -> a, [256,512) -> b)
    x_ij = sum_k a_ik b_jk (outgoing) | sum_k a_ki b_kj (incoming)       per channel d
    out  = z + sigmoid(zn Wg_out^T) * (LN_out(x) Wz^T)

Face (`__init__.py`): `install()` -> `pack(weights)` (from the ten canonical tensors: [value | gate] input projections stacked a-then-b, output
projection / gate, LN_in affine; LN_out affine folded, below) -> `forward(z, outgoing, mask, wp, residual=True|False)` with z `[N,N,256]` or
`[B,N,N,256]` bf16 CUDA, mask `[N,N]` / `[B,N,N]` 0/1 (float or bool) or None. `residual=True` is the kernel's epilogue (z + update);
`residual=False` is served by the SAME cubin with the epilogue's residual tensor map laid over a zero tile (every residual box is then either the
zero tile or out of bounds of the map, which the tensor-map unit zero-fills): out = update + 0.0 exactly, no extra pass over z
(`ZERO_TILE_RESIDUAL = False` turns this into the refusal `needs_residual`). `check(z, wp)` raises the envelope refusals before any launch.
Refusals are `Unavailable(.kind)`: `digest:<file>` | `driver:<reason>` | `prebuilt:<reason>` | `gate:<n>` | `needs_residual` | envelope words;
the face's fallback row is `esm_v5_fwd`.

## Implementation
K1 | bmm | K3 with only K3 in CUDA C++:
* **K1** = the Triton kernel `_k1x` of `src/ef2_trimul_v5.py` (tensor-descriptor loads; needs a Triton with `tl` tensor descriptors): LN_in +
  gated dual projection + mask -> channel-major bf16 planes `ab[2D, Np, Np]`, Np = ceil16(N); lever `incnt` writes the incoming direction's
  planes transposed so both directions run the NT GEMM; lever `sigmoid` = tanh.approx gate sigmoid; launch cell from
  `src/ef2_w4_fpf_trimul_v4_cells.json` by `<cc>|<triton M.m>` -> `<cc>|*` -> `*|*`.
* **bmm** = one cuBLAS strided-batched GEMM over the D channel planes (bf16 operands, fp32 accumulate).
* **K3** = `k3v6`, one warp-specialized CUDA C++ / CuTe kernel (`src/k3v6.cu`, identical to `K3_SRC` in `src/ef2_trimul_v6.py`) shipped as
  sm_90a cubins `bin/k3v6.1_fastsig<0|1>_lnfold1.sm_90a.cubin`: out[i, j, :] = z[i, j, :] + sigmoid(LN_in(z[i,j]) Wg^T + cg) * (LN_out(x[:, i, j])
  Wo^T + cp). One persistent CTA per SM, 384 threads, ~221 KB dynamic shared memory. Producer warpgroup: lane 0 of warp 0 issues the TMA loads
  of the x^T tile `[256 ch x 64 tok]` (x2) and the z tile `[128 tok x 256]` (SW128; 3-D tensor maps so partial tiles at a pair-row end are
  hardware-clipped), then the Wo / Wg `[32 x 256]` weight blocks through a 4-slot mbarrier ring; lane 0 of warp 1 streams the residual
  `z[i, j-tile, 32-channel block]` tiles through a 2-slot ring per consumer warpgroup. Two consumer warpgroups (128 threads, 64 tokens each, raised
  register budget) run the LayerNorms in registers and the `wgmma` projections against the ring. Release ordering: the consumer-side `empty_stage`
  (x^T / z slot) and `empty_res` (residual slot) mbarrier arrivals are issued only after every `ldmatrix` that read the slot has returned its data
  (a data dependency on one destination register per `ldmatrix`) and after `fence.proxy.async`, so the producer's next TMA fill cannot overtake
  the reads (`src/PATCHES/k3v6_release_ordering_fix.diff` is that rule as a diff against the unpatched kernel).
* **LN folding** (lever `lnfold`, the only shipped K3 configuration): the LN affine transforms are folded into the projection weights on the host,
  W' = bf16(W * gamma), bias c = W @ beta in fp32 (the fold runs with autocast disabled; the kernel reads `const float*` biases); LN_out statistics
  stay fp32 inside the kernel. Gate sigmoid: `fastsig1` = tanh.approx form, `fastsig0` = ex2 / rcp form (one cubin each).

## Numerics
Tolerance class (never an exact-tier value): bf16 operands, fp32 accumulation, fp32 LayerNorm statistics; K1 / bmm rounding points as row
`esm_v5_fwd`; K3 folds the affine into bf16 weights (W * gamma rounded once to bf16, beta through an fp32 bias) and rounds z + gate * projection
once to bf16. The payload's `tests/vectors/` state max |d| against the package's fp32 statement per sealed case.

## Loading and gates
`DIGESTS` in `__init__.py` pins the sha256 of every carried file of `pkg/v6.1/`; the files named in `NOT_CARRIED` must be absent; a differing byte
is `digest:<file>` before anything loads. `install()` once per process: digests -> driver binding through `cudrv` (the `cuda.bindings` wheel when
present, else a ctypes binding of the same entry points over `libcuda.so.1`; the package's single driver accessor is pointed at it, nothing in
`sys.modules` is replaced; the NVRTC half refuses by name, so a serving process never compiles) -> device capability 9.0 -> `cuModuleLoadData` of
the cubins whose digests match `bin/manifest.json`, injected into the package's kernel table -> load check (registers / local bytes read back
off the loaded functions == the producer's ptxas record) -> byte gate = the package's own `tests/test_vectors.py` over its sealed vectors (inputs
sha256-checked; every output bitwise == the recorded bytes for the sm90 device class, else within the stated ulp bound; any FAIL is `gate:<n>`).
While the carried modules load, gate and launch, a module an engine registered under the package's names is parked and restored afterwards
(`carried_binding`); such a foreign copy is admitted in the process only when its sealed bytes equal ours (`foreign_check`).

## Known limits
sm_90a SASS only (cc 9.0; `a`-suffixed features are not forward-compatible); c 256; bf16; forward only; K1 needs Triton's tensor-descriptor
API (a stack without it is refused by name and the face names `esm_v5_fwd`, the same K1 | bmm line with a Triton K3). One persistent CTA per
SM with ~221 KB shared memory: the kernel owns the SM while it runs.
