# protenix_fpf_trimul_tx 1.2 — integration note (from the package authors, for the kit maintainers)

**What it is.** TriangleMultiplication providers for Protenix v2 on cc 9.0 (H100/H200): CUDA sm_90a prologue/epilogue kernels (TMA + wgmma,
one producer + two consumer warpgroups, persistent grid) around one cuBLAS strided-batched bf16 contraction. Two levers from one package / one binary:

- `trimul_tx` (FAST tier) = `protenix_fpf_trimul_tx.trimul:fn`: same op contract, rounding points and dispatch cell as the current provider
  `fpf_trimul_v4.trimul:fn` (lever `trimul_v4`), which it replaces on cc 9.0. Numerics class TOLERANCE (LayerNorm statistics fp32 in the kernels'
  own summation order; sigmoid via ex2.approx; everything else at the cuEquivariance rounding points).
- `trimul_tx_exact` (EXACT tier) = `protenix_fpf_trimul_tx.trimul_exact:fn`: the stock module's statement order with results bitwise identical to
  the stock module's, entirely in this package's kernels — both LayerNorms inside K1 / K3 as variants whose bf16 output equals the installed
  library LayerNorm's (csrc `ln_cueq`; established against the installed function's outputs only, see the authors' numerics note), K1 writing
  unpadded planes, the contraction through the same
  `torch.einsum` expression (hence the stock cuBLAS problem), K3 with the residual folded in (update rounded to bf16, added in fp32, rounded once).
  Output == the stock module's output bit for bit (numerics class EXACT-BY-CONSTRUCTION, proved by torch.equal on recorded activations and by
  end-to-end byte identity under the deterministic recipe). Replaces `fpf_trimul_exact` on cc 9.0. (1.1 called the stock LayerNorm kernels around
  the GEMM kernels; 1.2 removes those two calls and their intermediate tensors: ~1.45x faster op at N % 8 == 0.)

No TF32 anywhere (bf16 MMAs with fp32 accumulate; fp32 LayerNorm arithmetic).

**Files (kit-relative).**
```
opt/forward/flashpairformer/third_party/protenix_fpf_trimul_tx/
  __init__.py  ops.py  trimul.py  trimul_exact.py  INTEGRATION.md  NOTICE.md  CELLS.json  TESTVECTORS.json
  csrc/trimul_tx.cu  csrc/tx_ptx.h                      # the source the binary is built from (digests in the manifest)
  prebuilt/manifest.json                                # binaries{abi tag -> file, module, sha256, toolchain, flags}, sources{sha256}, loadcheck{k1,k3,bmm,k1s,k3s2,k3s3 digests}
  prebuilt/torch213_cu130_sm90/protenix_trimul_tx_sm90.so  prebuilt/torch213_cu130_sm90/PROVENANCE.json
```
No new Python dependency. Build-time only: nvcc 13.x + the torch headers of the stack (tool: the package authors' `build_prebuilt.py --pkg <this dir>`
(workbench, outside the kit) on a machine of the target stack; it copies csrc to a scratch dir, builds with `-ffile-prefix-map` and without line info, writes the
.so + manifest incl. the load-time fingerprint). One binary per ABI tag `torch<maj><min>_cu<cuda>_sm90`; a stack without a matching entry cannot
run the lever (refused by name, below) until its binary is added to the manifest by the same script.

**Env lines / registry rows (proposal; the kit maintainers own registry.py and env.sh).**
- FAST: env.sh, in the `PTX_T_TRIMUL` case: `tx) export FPF_SMALLN_TRIMUL_FAST_FN=protenix_fpf_trimul_tx.trimul:fn ;;`  (v4 stays the `v4)` branch;
  the KIT_SPEC trimul word becomes `...+tx_above_gate`). registry.py:
  `_lever("trimul_tx", FORWARD, TOLERANCE, ("PTX_T_TRIMUL",), "Above-gate TriMul served by protenix_fpf_trimul_tx (cc 9.0 CUDA kernels; PTX_T_TRIMUL=tx is read BY env.sh).")`
  as the cc-9.0 member of the fast / big modes in place of `trimul_v4`; other GPU classes keep `trimul_v4` (this package serves cc 9.0 only and
  says so: a cc != 9.0 call takes the stock forward, counted `cc!=9.0`). Ablation `MODEL_OPT_LEVERS_OFF=trimul_tx` -> the `v4)`/partner branch.
- EXACT: env.sh line 52 equivalent for ARM=E: `export FPF_OPS="trimul_out=protenix_fpf_trimul_tx.trimul_exact:fn,trimul_in=protenix_fpf_trimul_tx.trimul_exact:fn"`
  when the exact lever word is `tx` (the kit names the switch; today `FPF_TRIMUL_EXACT` selects fpf_trimul_exact). Optionally also
  `FPF_SMALLN_TRIMUL_EXACT_FN=protenix_fpf_trimul_tx.trimul_exact:fn` (the fast tier's below-gate TriMul). registry.py:
  `_lever("trimul_tx_exact", FORWARD, EXACT, (<switch>,), "TriMul served by protenix_fpf_trimul_tx's exact path (cc 9.0; bitwise == stock).")`
  as the cc-9.0 member of the exact mode in place of `trimul_exact`; ablation -> fpf_trimul_exact (or stock) exactly as today's switch.

**Dispatch cell and refusals — FAST (trimul.py).** Served: `triangle_multiplicative == "cuequivariance"`, `c_z == c_hidden == 256`, CUDA bf16 `z` under
bf16 autocast, `[N,N,C]` or `[B,N,N,C]` (B > 1 served plane by plane), mask `None | [N,N] | [1|B,N,N]`, `N >= 101`, capability 9.0. Outside the
cell -> the stock forward for that call, counted by reason and named once on stderr (identical policy to fpf_trimul_v4). Cannot-run conditions ->
`ops.TrimulTxUnavailable` raised BY NAME at the first served-cell call (the mode refuses; nothing continues under this name): manifest missing, no
binary for the ABI tag, binary sha256 != manifest, shipped csrc digests != manifest (binary/source provenance broken), load-time fingerprint of K1 or
K3 != manifest (binary does not reproduce the recorded kernels' output). The cuBLAS contraction digest is informational (named if it differs).
Kernel runtime errors are raised (named), never rerouted. Residual: fused iff `inplace_safe is True and _add_with_inplace` (PairformerBlock / MSA
pair stack); otherwise the update is returned and the caller adds — as v4.

**Dispatch cell and refusals — EXACT (trimul_exact.py).** Served: as above but ONE pair plane only (`[N,N,C]` or `[1,N,N,C]`; a leading batch
> 1 takes the stock forward, counted `shape/batch`), `N > 100` (at or below it the stock forward serves, by name), mask values 0/1. No upper N bound in
code; bitwise identity is established on recorded activations at 400/800/1400 tokens (both directions), on module-generated activations at
130/389/478/1001 with and without masks, and end to end under the deterministic recipe (see the authors' numerics note). Cannot-run adds the load-time
LIVE check — before the first served call per process and direction the provider runs this exact path against the INSTALLED stock forward
(`fpf.original`, i.e. the live library call in that process) on the closed-form N = 160 and N = 162 tiles
(both LayerNorm variants of K3, both plane layouts) with the caller's weights, a mask with zeros and the caller's
residual arguments; `torch.equal` -> marker `livecheck=eq` on stderr and in COUNTS, else `TrimulTxUnavailable` by name (a cuEquivariance upgrade
or tile-configuration change that moves the stock bits is refused in-process). Ragged N: N % 8 != 0 planes are written 16-token-padded and packed
by one strided copy, and the contraction result is pad-copied once for K3's 16-byte plane rows (both bitwise-neutral copies; an IOU below);
N % 4 != 0 selects K3's second LayerNorm variant (template LNM = 3).

**Lines it prints.** `[protenix_fpf_trimul_tx] loaded {...tag, file, sha256...} loadcheck=ok` once (`[protenix_fpf_trimul_tx.exact] ... loadcheck=ok` + `livecheck=eq (out|in, ...)` for the exact provider); `FIRST CALL served: {...}` once;
`stock forward for this call (reason: ...)` once per reason; `COUNTS {...calls, served, fallback{reason: n}, ...}` at exit (also copied into
`ptx_trunk2_levers._STATS["protenix_fpf_trimul_tx"]` / `["protenix_fpf_trimul_tx.exact"]` for the kit's tally).

**CUDA-graph capture.** Capture-safe (no host syncs, no allocator calls outside torch's caching allocator; per call: one [512,Np,Np] bf16 plane
buffer + the cuBLAS output + the result). Runs under fpf_stackgraph capture/replay in the fast tier.

**Memory.** FAST: same footprint class as v4 (planes padded to Np = ceil16(N)). EXACT: unpadded planes (+ one padded copy of the planes / of the contraction result at ragged N); no LayerNorm intermediates ([N,N,256] bf16
each), the same intermediates the stock pipeline allocates.

**Synchronization note (1.1).** Tiles read through the generic proxy (ldmatrix / ld.shared) are released to the TMA producer only after a
`fence.proxy.async`; warps reconverge (`__syncwarp`) before every named barrier; ragged-row plane stores are predicated inside one asm block.
1.0's binary lacked the proxy fences (a write-after-read window between a tile's reads and its asynchronous refill; never observed on the 1.0
paths, observed on the exact path during bring-up); 1.1's fast-path output equals 1.0's bit for bit on all recorded activations.


## IOUs (1.2)
- Ragged N (N % 8 != 0) in the exact provider pays two strided torch copies (planes pack after K1, contraction-result pad before K3): e.g. K1+pack
  0.38 ms @389 / 2.2 ms @1001 vs 0.16 / 1.0 ms for the aligned kernel alone. A dedicated pack kernel (or 4/8-byte plane stores keyed on row
  alignment) would recover most of it; not started.
- Bitwise equality of the exact tier's LayerNorm variants with the installed cuEquivariance build (0.11.1 here) is an observed property of that
  build; the live check refuses by name if a future build returns different bytes, and the exact provider then stays off until its LayerNorm
  variants are re-established against the new build's outputs.
