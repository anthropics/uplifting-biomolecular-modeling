# kernels/triattn_xla — design note

Triangle attention (pair bias + key mask, forward; a differentiable variant by word) for JAX / XLA programs, served by PRE-COMPILED GPU
kernels through one XLA-FFI launcher library: the XLA binding of the FlashPairformer triangle-attention kernels (the CUDA-native members
carried by `kernels/triattn/triattn_native` and `kernels/triattn/cuda_sm90a`, and the portable Triton member `kernels/fpf_triatt_k2b` compiled
ahead of time).  No compiler runs at import or call time for the forward rows: the package ships the binaries
under `bin/` (every file listed with its sha256 in `manifest.json`, a `.json` build record beside each) and refuses BY NAME whatever they do
not cover (`Refused`, a `NotImplementedError`, always naming the fallback `stock: xla` = the program's own einsum/softmax).  `__version__ = "1.6.1"`.

## Op and calling convention

`out = triangle_attention(q, k, v, bias, mask=None, scale=None, *, impl="auto", layout="BNHSD", vjp=None)`

* `q, k, v`: `[B, N, H, S, D]` (`"BNHSD"`, the cuEquivariance convention) or `[B, N, S, H, D]` (`"BNSHD"`, the AF3-family module layout); rank 4 =
  no batch; dense row-major device arrays; N = pair rows, S = queries = keys.
* `bias`: `[B, H, S, S]` | `[B, 1, H, S, S]` | `[H, S, S]`, fp32 / bf16 / fp16, shared by the N rows of a batch element.
* `mask`: `[B, N, S]` (or `[B, N, 1, 1, S]` / `[N, S]`) bool or uint8, 1 = attend, or None; a row with no attended key yields the mean of `v`.
* `scale`: default `D ** -0.5`.  Returns an array of `q`'s shape, dtype and layout.
* `select(cc, dtype, head_dim, S, …)` answers "which row, or why not" without arrays; `report()` lists what loaded, what was served or refused,
  and the vmap mode; `reference()` is the plain-XLA statement of the op.

## Rows (`ROWS = ("triattn_native", "cuda_sm90a", "cuda_80", "k2b_aot")`; envelopes are the `rows` block of `CELLS.json`)

| row | binary | serves |
|---|---|---|
| `triattn_native` | `bin/cuda/sm_90a/libtriattn_m1_xla.so`, FFI target `triattn_xla_m1_fwd` | cc 9.0, bf16, D 32, bias f32/bf16, mask or none: the `cuda_b` (`triattn_m1`) kernel of `kernels/triattn/triattn_native` — kernel header compiled unmodified, host side restated over raw pointers in `csrc/triattn_m1_xla.cu`, plain C entry |
| `cuda_sm90a` | `bin/cuda/sm_90a/libtriattn_mw_cuda.so`, target `triattn_xla_cuda_fwd` | cc 9.0, bf16, D 32, bias f32/bf16/f16: the kernel of `kernels/triattn/cuda_sm90a/csrc/` (device section unmodified, host unit `csrc/triattn_mw_cuda.cu`, static CUDA runtime) |
| `cuda_80` | `bin/cuda/sm_80/libtriattn_sm80_xla.so`, target `triattn_xla_sm80_fwd` | cc 8.0 only, bf16, D ∈ {16, 32, 64}, bias f32/bf16: the `cuda_80` (`triattn_sm80`) member of `kernels/triattn/triattn_native`, host side `csrc/triattn_sm80_xla.cu` |
| `k2b_aot` | `bin/k2b/sm_90/*.cubin`, `bin/k2b/sm_80/*.cubin`, target `triattn_xla_run` | cc 9.0 (sm_90 cubins) and 8.x (sm_80 cubins run on 8.0 / 8.6 / 8.9), bf16 and fp32, D ∈ {16, 32}: the Triton kernel of `kernels/fpf_triatt_k2b` (`_bias_prep` + `_flash_triattn_fwd`) compiled ahead of time |

All rows: `S_q == S_kv`, layouts BNHSD and BNSHD, forward only.  Selection (`impl="auto"`): `CELLS.json by_cc[<cc>].order`, first row that
serves the call, with `by_cc[<cc>].rules` expressing size preferences among rows that can serve; a cc without an entry follows the same-major
rule for 8.x.  Each row is individually switchable off through `MODEL_OPT_LEVERS_OFF` (`triattn_xla` = all, `triattn_xla:<row>`, `triattn_xla:vjp`).
Every decision is recorded in the opt_core cell census.

## Launcher and launches

`bin/launcher/ffi-<major>.<minor>/libtriattn_xla_launch.so` — one build of `csrc/cubin_launch.cc` per XLA-FFI API version of the jaxlib
headers (the version a jaxlib carries is read from its own include directory at load; a jax without `jax.ffi`, or an API version with no
build here, is refused by name).  Loaded with ctypes; its handlers are registered with `jax.ffi.register_ffi_target` (typed FFI,
`api_version=1`, platform CUDA): `triattn_xla_run` (cubin launch through the driver API), `triattn_xla_cuda_fwd`, `triattn_xla_m1_fwd`,
`triattn_xla_sm80_fwd` (calls into the CUDA rows' C entries).  A cubin launch is SELF-DESCRIBING: the cubin key + digest, kernel name, grid,
shared memory and the parameter recipe travel as attributes of the custom call and the launcher reads the cubin from the package on first
use in a process — so an executable restored from a persistent compilation cache runs in a process that never traced it.
`k2b_aot` restates the torch launcher's host arithmetic in `_k2b.py`: the bias preparation pass (fp32 copy with a 16-element row pitch +
an optional lossless 16-bit copy and flags), the tile grid, the int32 offset bound, the constexpr set of the launch cell; the cubin is
chosen by (arch, dtype, D, mask, BIAS16, divisibility class) — file names `k2b_{prep,fwd}_<dtype>_d<D>_mask<0|1>_b16<0|1>_<any|s16>`.
The CUDA rows' scratch buffers (staged bias, mask tables, fix list) are extra outputs of the FFI call that the caller discards, so XLA owns
every byte and nothing persists between calls.

## Under `jax.vmap` (`_batching.py`)

Every FFI call is built with `vmap_method="sequential"` where the running jax accepts it, and every launch is additionally wrapped in
`jax.custom_batching.custom_vmap` (where available) with the rule: fold the vmapped axis into the kernel batch axis B (operands that are not
vmapped are broadcast) and launch ONCE after re-checking the envelope for the folded shape; a folded call outside a row's envelope is
launched per sample instead.  Either way each sample gets the bytes an unbatched call gives; `report()["vmap"]` says which layer applies.

## The differentiable row (`vjp="auto"` | `"attbwd"` | `"flash"` | `"flash_xla"`; `_vjp.py`, `jax.custom_vjp`)

Forward per cc (`CELLS.json vjp.fwd_by_cc`): on cc 9.0 `cuda_lse` = the `cuda_sm90a` kernel built WITH a per-row log2-sum-exp store
(`csrc/cuda_lse.patch`: an `lse` pointer in the launch parameters and one fp32 store per query row in the epilogue →
`bin/cuda/sm_90a/libtriattn_mw_cuda_lse.so`; `out` bit-identical to row `cuda_sm90a`'s), elsewhere `k2bl` = the K2B kernel compiled with the
store (`csrc/k2b_lse_kernel.py` → `bin/k2b/sm_*/k2bl_fwd_*`; `out` bit-identical to `k2b_aot`'s); `cuda_80` serves it through its own lse
output (word `cuda80_lse`).  Residuals: (q, k, v, bias, mask, out, lse) — B·N·H·S·(D·itemsize + 4) bytes on top of the inputs, no logits
kept, no forward re-run.  Backward: Pallas kernels lowered by the running jax, chosen per (cc, dtype) from `CELLS.json vjp.by_cc` — `attbwd`
(`kernels/pallas_triatt`: dK/dV over key blocks, dQ + row-group-summed d(bias) partials over query blocks, partials reduced by XLA; no
atomics) or `flash` / `flash_xla` (`kernels/pallas_attn` backward with d(bias) summed in-kernel / per-row partials reduced by XLA;
`flash_xla` is refused by name above a partial-memory bound).  d(bias) = the sum over the N rows, returned in the bias' dtype and shape; the
mask takes no gradient; a fully-masked row gets dV from the uniform weights and no dQ / dK / d(bias).  Serves layout BNHSD, B == 1, D = 32,
bf16 (log2-domain statistics) and fp32 (TF32 products), cc 9.0 and 8.x; anything else — or a jax on which the backward's Pallas modules do
not import or lower — is refused by name with the fallback (XLA autodiff of the program's own attention).  With `vjp=None` the forward rows
define no VJP/JVP and differentiation raises `Refused` naming the row.

## Numerics

CUDA rows: bf16 tensor-core products, fp32 accumulation, fp32 bias and softmax (the max-free base-2 scheme with an exact SAFE pass of their
torch counterparts — the same device code), bf16 P for P·V; deterministic.  `k2b_aot`: identical arithmetic to the torch-launched K2B kernel
(same compiler output class per arch, same launch cells, same constexpr set): bf16 with base-2 softmax, fp32 with TF32 products in the
natural domain.  `_vectors.py` + `vectors.json`: numpy-generated, bit-defined inputs (`vector_cases()`, `vector_inputs()`) and the recorded
sha256 of each row's output bytes per case (one K2B digest per cubin arch), for byte-level checks of a row on a device.

## Known limits

Forward rows: no gradient, `S_q == S_kv`, one pair bias per batch element, dense inputs only; per-row envelopes as tabled (cc 8.6 / 8.9 reach
only `k2b_aot`).  The launcher must match the jaxlib's XLA-FFI API version (builds for `ffi-0.1` and `ffi-0.3`).  The differentiable row
compiles its Pallas backward through the running jax/Triton at first use; the forward rows never compile.
