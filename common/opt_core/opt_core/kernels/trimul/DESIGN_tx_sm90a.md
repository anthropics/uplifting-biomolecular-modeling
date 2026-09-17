# kernels.trimul.tx_sm90a — design note (rows `tx_sm90a`, `tx_sm90a_exact`; package `protenix_fpf_trimul_tx`)

## Op and calling convention
TriangleMultiplication (outgoing | incoming), forward, for c_z = c_hidden = 256, bf16 pair tensors under bf16 autocast, compute capability 9.0.
Two providers with the stock module signature `fn(module, z, mask=None, inplace_safe=False, _add_with_inplace=False, ...)`:
`trimul.py:fn` (fast tier) and `trimul_exact.py:fn` (exact tier); `ops.trimul` / `ops.trimul_exact` are the tensor-level entries the
`kernels.trimul` face calls for the rows `tx_sm90a` / `tx_sm90a_exact` (fallback rows `v4` / `tmk3_exact`).

    x = LN_in(z);  a|b = mask * sigmoid(x Wg^T) * (x Wp^T);  X = sum_k a_ik b_jk | sum_k a_ki b_kj;  out = [z +] sigmoid(x W_og^T) * (LN_out(X) W_o^T)

Served cell: CUDA bf16 z `[N,N,256]` or `[B,N,N,256]` (fast: B > 1 plane by plane; exact: ONE plane, `[N,N,C]` or `[1,N,N,C]`), mask `None |
[N,N] | [1|B,N,N]` (exact: values 0/1), N >= 101 (`trimul.N_MIN`; the exact provider N > 100, below which cuEquivariance takes its own small-N
function), `triangle_multiplicative == "cuequivariance"` on the module path. A module-path call outside the cell takes the stock forward for
that call, counted by reason and named once. Cannot-run is `ops.TrimulTxUnavailable` raised BY NAME (no binary for this ABI key, manifest
mismatch, load-time fingerprint mismatch, exact live check unequal); nothing continues under the row's name on another path. Residual: fused iff
`inplace_safe and _add_with_inplace` (a NEW tensor z + update is returned, z is not mutated), else the update alone.

## Implementation
K1 (`k1_kernel`) and K3 (`k3_kernel`) in `csrc/trimul_tx.cu`, with `csrc/tx_ptx.h` = thin inline-asm wrappers over mbarrier,
`cp.async.bulk.tensor`, ldmatrix / stmatrix, wgmma, setmaxnreg and the driver's `cuTensorMapEncodeTiled`; no third-party kernel source.
K1: LN_in + gated dual projection + mask -> channel-major bf16 planes `[512, Np, Np]`, Np = ceil16(N), zero pad. Contraction: one cuBLAS
strided-batched GEMM on the planes (`torch.bmm`; bf16 operands, fp32 accumulate, bf16 out). K3: LN_in recomputed for the output gate + LN_out +
W_o projection x sigmoid(gate) [+ residual] -> `[N, N, 256]` bf16.
Both kernels: persistent CTAs of three warpgroups -- WG0 producer (TMA bulk-tensor loads into 128B-swizzled shared memory, mbarrier rings),
WG1 / WG2 consumers (ldmatrix -> LayerNorm in registers on the m64 x k256 A fragment, statistics fp32 -> `wgmma` m64n64k16 bf16 -> fp32 with A from
registers and the weight block from shared memory -> fused epilogue, software-pipelined one weight block ahead). Tiles read through the generic
proxy are released to the producer only after `fence.proxy.async`; warps reconverge before every named barrier; ragged-row plane stores are
predicated inside one asm block.

Exact assembly (`ops.trimul_exact`): K1 with the LayerNorm variant whose bf16 output is bitwise identical to the stock LayerNorm's (`ln_cueq` in
the source) writing UNPADDED
planes `[512, N, N]` (`ops.k1_stock_order`: N % 8 == 0 written directly with 16-byte rows, otherwise 16-token-padded and packed by one strided
copy); the contraction as the stock `einsum` expression on `[256, 1, N, N]` chunk views (`ops.contract_stock`), hence the stock cuBLAS problem,
kernel and bits; K3 with the transposing-LayerNorm variant bitwise identical to the stock one (two arithmetic variants; N % 4 != 0 selects the
second, template `LNM = 3`) and the module's residual arithmetic (update rounded to bf16, added in fp32, rounded once). Bitwise equality with the
installed library (cuequivariance_torch 0.11.1 on the pinned stack) was established against that function's outputs only (see `NOTICE.md`) and
is guarded per process by the live check (`trimul_exact._livecheck`): before the first served call per direction this path runs against the
INSTALLED stock forward on closed-form N = 160 and N = 162 tiles (both variants, both plane layouts) with the caller's weights, a mask with zeros
and the caller's residual arguments -- `torch.equal`, else `TrimulTxUnavailable` by name.

## Numerics
Fast tier: tolerance class = the cuEquivariance rounding points (LN output -> bf16; gate*proj(*mask) on fp32 accumulators -> bf16; bf16 GEMM,
fp32 accumulate, bf16 out; residual folded into K3), LayerNorm statistics fp32 (two-pass, centred) in the kernels' own summation order, sigmoid
via `ex2.approx`; no TF32 anywhere. Exact tier: output == the stock module's output bit for bit by the construction above.

## Loading
One prebuilt torch extension per ABI key under `prebuilt/<key>/protenix_trimul_tx_sm90.so`, key = `torch<torch.__version__>-<sysconfig
SOABI>-sm90` (`ops.abi_tag`); `prebuilt/manifest.json` records per key the binary sha256, module name, nvcc flags and toolchain, the sha256 of
the two `csrc/` files, and the load-time fingerprint (`loadcheck`: digests of the K1 planes and of the K3 output on closed-form N = 160 inputs,
for the fast entry points and the stock-order ones). `ops.load()` once per process: binary and source digests == manifest -> import through
`ExtensionFileLoader` -> fingerprint == manifest (the cuBLAS digest is informational). A key without a binary is refused by name; `build_prebuilt.py`
builds the extension for the running process's key from the shipped `csrc/` with the manifest's flags and accepts it only if a fresh interpreter
reproduces the recorded fingerprint.

## Known limits
cc 9.0 only (sm_90a SASS); c_z = c_hidden = 256 only; bf16 autocast only; forward only. Capture-safe: no host synchronisation, allocations only
through torch's caching allocator (per call: the plane buffer, the cuBLAS output, the result). Exact tier: batch 1; ragged N (N % 8 != 0) pays two
strided copies (plane pack after K1, result pad before K3); a cuEquivariance release or tile-configuration change that moves the stock bits is
refused in-process by the live check rather than served. Kernel runtime errors are raised, never rerouted.
