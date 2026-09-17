# csrc interface: what an architecture member implements


> **Status (R1 candidate).** The sm_80 member that landed (`csrc/sm80/`: `trimul_k1_sm80.cu`, `trimul_k3_sm80.cu`, `math_sm80.cuh`) is a separate
> pair of kernels written against this document's OP CONTRACT (sections 1-3: plane layout `[2 c_h][Np][Np]` bf16 with masked zero padding, the NT
> contraction, the K3 inputs/outputs, the weight pack) and the shared NUMERICS STATEMENT (`csrc/common/tmn_math.cuh` semantics; the sm_80 files
> currently carry their own copy in `math_sm80.cuh` pending the include switch) -- it does NOT reuse `tmn_kernels.cuh`'s `k1_body` / `k3_body` with
> a swapped staging layer, which is what section 4 below sketches as the eventual single-source form.  Read section 4 as design intent, not as a
> description of `csrc/sm80/`.  The sm_90a kernels' name grammar, tile table and launch parameter blocks are owned by `python/trimul_native/kernel.py`
> (`unit_name`, `k1_name`, `k3_name`, `TILE_TABLE`, `lookup`, `serve_names`); the sm_80 member's by `python/trimul_native/sm80_ops.py`.
The kernel family is split into architecture-neutral math (namespace `tmn`, `tmn_kernels.cuh`) and one mainloop/staging member per architecture
(`namespace tmn::sm90` today; `tmn::sm80` slots in beside it).  A member provides, for K1 and K3, a persistent-CTA body

    template <class Cfg, bool HAS_MASK, int LNM, bool SAVE> __device__ void k1_body(const K1Params&);
    template <class Cfg, int LNM>                           __device__ void k3_body(const K3Params&);

instantiated by `tmn_<arch>.cu` as extern "C" kernels whose names encode the configuration (kernel.py selects by name; the parameter blocks are the
same POD structs for every member; members that do not use tensor maps ignore those fields and read the raw pointers/strides instead).

What the member owns (and nothing else):
1. **Tile staging**: bring the z tile [BM tokens x C_Z] (bf16 or fp32), the X tile [C_H x 64 tokens per warpgroup-equivalent] and the weight
   chunks ([64 n x 64 k] bf16, 8 KB, k contiguous 128-B rows) into shared memory in the canonical layouts below, with whatever copy engine the
   architecture has (sm90: TMA + mbarrier rings, 128B swizzle; sm80: cp.async 16-B, same swizzle function `swz128`).
2. **MMA issue**: D[64 x 64 fp32] += A[64 x 16 bf16 from registers] x B[16 x 64 from smem] chains over K (sm90: wgmma m64n64k16 RS; sm80:
   mma.sync m16n8k16 with the identical A-fragment register layout — the m16n8k16 A fragment of rows (g, g+8), k (2q, 2q+1, 2q+8, 2q+9) is exactly
   the per-lane layout `ln_fragment` produces, so the LayerNorm code is shared unchanged; B fragments come from ldmatrix on the same swizzled chunk).
3. **Accumulator -> epilogue adapter**: hand the shared epilogue functors the accumulator values with their (row, column) meaning.  The shared code
   assumes the sm90 m64nNk16 C layout per warp: lane holds rows (lane/4, +8) of its 16-row slice, columns 8q' + 2(lane%4) + {0,1}; the sm80 member
   arranges its m16n8 C fragments in the same order (it is the same layout by construction of mma.sync m16n8k16).

Canonical shared-memory layouts (both members): every operand tile is stored as 128-byte rows with the 16-byte granule index XOR (row & 7)
(`swz128(row, byte_col)`), chunked by 64 bf16 (or 32 fp32) channels; chunk c of a [rows x C] tile lives at base + c * (rows * 128).
Shared math the member calls: `ln_fragment` / `ln_rows_f32` / `ln_stock` (LayerNorm on A fragments), `load_frag_bf16`, `sigmoidf_`, `pack_bf16`,
`stg_ragged` (predicated ragged stores), the K1 plane-store epilogue and the K3 gate*proj(+residual) epilogue arithmetic.
Parameter blocks: `K1Params` / `K3Params` (POD; tensor maps first, then pointers, ints, float); `tmn_info` reports sizeof/offsetof for the packer.
