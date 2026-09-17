// Triangle attention forward, sm_90a, consumer design "M1": max-free chunk streaming with narrow chunks and three consumer warpgroups.
//   CTA = (batch b, head h, one q-tile of 128 queries, R = 3 consecutive pair rows); 512 threads = 1 producer warpgroup (one TMA thread)
//   + 3 consumer warpgroups, consumer warpgroup w owns pair row i0 + w for all 128 queries (two m64 halves h = 0, 1).
//   K/V tiles of 128 keys stream through a 2-tile ring per row (stage = (tile & 1) * 3 + w); the pair-bias tile (128 q x 128 keys,
//   shared by the 3 rows) arrives as four 16 KB fp32 slots, one per 32-key chunk column c, holding bias / scale in MMA-FRAGMENT ORDER
//   (written once per call by stage_bias_m1) so a chunk's S accumulator is initialised with 4 LDS.128 per thread and the QK wgmma
//   accumulates on top: no per-logit bias instruction, no bias MMA.
//   Per consumer warpgroup the keys of its row form ONE stream of chunks k = 8*tile + 2*c + h (64 q x 32 keys): S(k) = bias/scale +
//   Q_h K_c^T (m64n32k16 x2, issued two chunks ahead), p = 2^(S*c_l2 + nm[h]) with a FIXED per-row shift nm seeded from the first
//   chunk column (max-free: no running max, no rescale), bf16 pack + PV wgmma over [V | 1] (m64n40k16 x2, the ones columns
//   accumulate the row sum) one chunk late and committed LAST in each chunk body (so ptxas cannot hoist the next wgmma wait above the
//   exponentials); every wgmma retired once per 16-chunk period (= the 2-tile ring, so every smem stage index is compile-time).
//   Key masks: keys dead in every row of the batch element are -inf columns of the staged bias; the CTA's tile stream covers the union of
//   its rows' live 32-key column ranges; a row whose mask differs from the batch OR ("irregular") consumes only the tiles of its own range
//   (its K/V ring carries only those; for the other tiles of the CTA stream its warpgroup just takes part in the shared bias ring's
//   protocol) and applies its mask words only on the tiles that can hold a boundary. The period body exists in two instantiations: the
//   STEADY body (no mask-word code: the body an unmasked call runs everywhere) and the GENERAL body (mask words on the flagged tiles of
//   the period), which a warpgroup with an irregular row runs for its first period and its last one or two (a ragged row: for all).
//   Bias half-slot releases are exact in both (no arrival for the chunks the schedule runs past a row's last tile).
//   Fully-masked rows get the uniform mean of v from uniform_rows_kernel (m1_binding.cu), whatever this kernel stores for them: a CTA tile
//   whose rows are ALL fully masked (sequence padding) therefore streams nothing -- producer and consumers leave right after the row kinds
//   are read (the tile's Q load, issued ahead of them, is awaited first); a fully-masked row that shares its tile with a live row is
//   computed like a regular one and not validated.
//
//   out[b,i,h,q,:] = softmax_k( scale * q[b,i,h,q,:].k[b,i,h,k,:] + bias[b,h,q,k] (-inf where mask[b,i,k] == 0) ) @ v[b,i,h,k,:]
//
// Grid x = q-tiles, y = row triples, z = b*H + h.
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cuda_bf16.h>
#include <climits>

#include "../fa3_utils.h"

namespace triattn_m1 {

using namespace cute;

// Ring of smem stages filled by TMA and drained by consumers. full[s]: transaction barrier (1 producer arrival + the stage's bytes);
// empty[s]: `empty_arrivals` consumer arrivals per use. Use u of a stage has phase u & 1.
template <int Stages>
struct Pipe {
    struct SharedStorage {
        cutlass::arch::ClusterTransactionBarrier full[Stages];
        cutlass::arch::ClusterBarrier empty[Stages];
    };
    SharedStorage& st;
    CUTLASS_DEVICE Pipe(SharedStorage& s) : st(s) {}
    CUTLASS_DEVICE static void init(SharedStorage& s, int empty_arrivals, int full_arrivals = 1) {
        for (int i = 0; i < Stages; ++i) { s.full[i].init(full_arrivals); s.empty[i].init(empty_arrivals); }
    }
    // NOTE: cutlass::arch::ClusterBarrier::wait() carries no "memory" clobber, so the compiler may hoist ordinary shared-memory loads
    // (the bias-fragment init) ABOVE a wait -- reading a stage before its TMA bytes land. Every wait here is followed by a compiler barrier.
    CUTLASS_DEVICE void producer_wait_empty(int stage, uint32_t phase) { st.empty[stage].wait(phase ^ 1); asm volatile("" ::: "memory"); }
    CUTLASS_DEVICE void producer_expect(int stage, uint32_t bytes) { st.full[stage].arrive_and_expect_tx(bytes); }
    CUTLASS_DEVICE uint64_t* full_barrier(int stage) { return reinterpret_cast<uint64_t*>(&st.full[stage]); }
    CUTLASS_DEVICE void wait_full(int stage, uint32_t phase) { st.full[stage].wait(phase); asm volatile("" ::: "memory"); }
    CUTLASS_DEVICE bool test_full(int stage, uint32_t phase) { return st.full[stage].test_wait(phase); }   // non-blocking probe: consume the result later
    CUTLASS_DEVICE void release(int stage, bool elected) { if (elected) { st.empty[stage].arrive(); } }
    // arrive whose ADDRESS depends on `dep` (a value derived from registers loaded out of the stage): the arrive cannot issue before those
    // loads have RETURNED. A plain arrive after an ld.shared is not enough: the load may still sit in the MIO queue when the arrive
    // (and the producer's refill) take effect.
    CUTLASS_DEVICE void release_dep(int stage, bool elected, uint32_t dep, int rt_zero) {   // rt_zero: a kernel parameter that is 0 (opaque to ptxas)
        uint32_t const addr = cute::cast_smem_ptr_to_uint(&st.empty[stage]) + dep * uint32_t(rt_zero);   // IMAD on the loaded registers: the arrive cannot ISSUE before the loads returned
        if (elected) { asm volatile("{\n\t.reg .b64 state;\n\tmbarrier.arrive.shared::cta.b64 state, [%0];\n\t}" :: "r"(addr) : "memory"); }
    }
    CUTLASS_DEVICE void release2(int stage, bool elected, uint32_t peer) { if (elected) { st.empty[stage].arrive(); st.empty[stage].arrive(peer, 1u); } }   // + the cluster peer's barrier
};

template <int kFlags_ = 0>
struct Traits {
    // kFlags bit 4 (16): clock64 trace stamps of CTA (1,1,0); bit 0 (1): ABLATION zero bias init (timing); bit 1 (2): ABLATION replay
    // (after the first ring fill no barrier or TMA traffic at all, stale tiles recomputed)
    static constexpr int kFlags = kFlags_;
    // bit 10 (1024): SAFE instantiation -- exact running max with per-chunk rescale (every wgmma drained per chunk), looping over the
    // fix list written by the hot instantiation; same tile routine otherwise (one source of truth for the recompute)
    static constexpr bool kSafe = (kFlags_ & 1024) != 0;
    static constexpr bool kList = kSafe;                         // persistent over a list of CTA tiles (else: one grid tile per CTA)
    // bit 8 (256): thread-block cluster of 2 CTAs along y (row triples 2m, 2m+1 of the same q-tile) sharing every bias slot by TMA
    // multicast (each CTA issues half a slot; halves the pair-bias L2->SMEM traffic per row)
    static constexpr int CR = (kFlags_ & 256) ? 2 : 1;
    using ClusterShape = Shape<_1, Int<CR>, _1>;
    using Element = cutlass::bfloat16_t;
    static constexpr int kHeadDim = 32, kBlockM = 128, kBlockN = 128, CW = 32, R = 3, kRingKV = 2, kSlotsB = kBlockN / CW;
    static constexpr int kStagesKV = kRingKV * R;
    static constexpr int kNumMmaWG = R;                          // consumer warpgroup w <-> pair row i0 + w
    static constexpr int kNumMmaThreads = kNumMmaWG * 128;
    static constexpr int kNumThreads = kNumMmaThreads + 128;
    static constexpr int kChunksPerTile = 2 * (kBlockN / CW);    // per warpgroup: (c, h) pairs
    static_assert(kChunksPerTile == 8 && kSlotsB == 4);

    using AtomLayout = Layout<Shape<_1, _1, _1>>;                // every tiled MMA here is ONE warpgroup
    using TiledMmaQK = decltype(make_tiled_mma(GMMA::ss_op_selector<Element, Element, float, Shape<_64, Int<CW>, Int<kHeadDim>>>(), AtomLayout{}));
    // flag 1048576: Q as the REGISTER operand of the QK wgmma (loaded once per tile by LDSM; 16 registers): the tensor core reads only K from shared memory
    static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0;
    // flag 4194304: schedule W = two S buffers (chunk parity), QK issued one chunk ahead at the top of the body, E then pack + PV of the
    // SAME chunk (PV the last commit), S(k+2) initialised at the end of the body into the buffer E just freed: 32 fewer live registers
    static constexpr bool kW = (kFlags_ & 4194304) != 0;
    using TiledMmaQKr = decltype(make_tiled_mma(GMMA::rs_op_selector<Element, Element, float, Shape<_64, Int<CW>, Int<kHeadDim>>>(), AtomLayout{}));
    using TiledMmaPV = decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, AtomLayout{}));   // O += P V (B = V^T view, MN-major)
    using TiledMmaL  = decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}, AtomLayout{}));     // l += P 1 (B = all-ones 8x32 tile)

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockM>, Int<kHeadDim>>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<R>{})));          // (M, D, R)
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kStagesKV>{})));   // (BN, D, st)
    static constexpr int kStageElemsK = kBlockN * kHeadDim;
    static_assert(cosize(SmemLayoutK{}) == kStageElemsK * kStagesKV);
    // V stage = (BN, D) exactly like a K stage (D-contiguous rows, 64B swizzle, TMA box rows = 64 B = 2 full sectors); the PV B operand reads it
    // through the transposed (D, BN) view = MN-major swizzled descriptors (as FlashAttention-3 does)
    using SmemLayoutV = SmemLayoutK;                                                                                                  // (BN, D, st)
    using SmemLayoutVt = decltype(cute::composition(SmemLayoutV{}, make_ordered_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}, Int<kStagesKV>{}), Step<_2, _1, _3>{})));  // (D, BN, st)
    static constexpr int kStageElemsV = kStageElemsK;
    // all-ones B tile (8 x 32 bf16) for the row-sum MMA l += P 1: K-major 8x8 core matrices (values are layout-invariant anyway)
    using SmemLayoutOnes = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<Element>{}, make_shape(_8{}, Int<CW>{})));            // (8, CW)
    // bias slot c (keys 32c..32c+31 of the tile, all 128 q): 4096 fp32 = bias/scale in fragment order [h(2)][u(4)][thread(128)][4]:
    // thread t's u-th float4 = accumulator elements 4u..4u+3 of its m64n32 S chunk (h) = rows 16*(t/32) + (t%32)/4 (+8), keys 8u + 2*(t%4) (+1).
    // The pipeline unit is the HALF slot m = 2c + h (2048 fp32 = one chunk's bias): its own full/empty barrier pair, released by each
    // warp right after its init loads, so the producer refills it while the other half and the other columns are still in use.
    static constexpr int kSlotElems = 2 * 4 * 128 * 4, kHalfElems = kSlotElems / 2, kHalves = 2 * kSlotsB;
    using SmemLayoutBiasHalf = Layout<Shape<_256, _8>, Stride<_1, _256>>;                                 // one half slot as the TMA box sees it
    using ShapeB  = Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;                                     // (256, 8, 8*nk, nq, B*H): block 8*tile + 2c + h
    using StrideB = Stride<_1, _256, int64_t, int64_t, int64_t>;
    using StrideQK = Stride<int64_t, _1, int64_t, int64_t, int64_t>;      // (S, D, H, N, B)
    using StrideV  = Stride<_1, int64_t, int64_t, int64_t, int64_t>;      // (D, S, H, N, B)
    using ShapeQK  = Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;

    using TMA_Q = decltype(make_tma_copy(SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQK{}, StrideQK{}),
                                        take<0, 2>(SmemLayoutQ{}), make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), _1{}));
    using TMA_K = decltype(make_tma_copy(SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQK{}, StrideQK{}),
                                        take<0, 2>(SmemLayoutK{}), make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), _1{}));
    using TMA_V = TMA_K;                                         // V is loaded exactly like K
    using GmemTiledCopyB = std::conditional_t<(CR > 1), SM90_TMA_LOAD_MULTICAST, SM90_TMA_LOAD>;
    using TMA_B = decltype(make_tma_copy(GmemTiledCopyB{}, make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), ShapeB{}, StrideB{}),
                                        SmemLayoutBiasHalf{}, make_shape(_256{}, _8{}), Int<CR>{}));
    // ABLATION (timing only, flag 512): the h = 1 halves are fetched with a 1 KB box (garbage bias for those chunks): halves the bias L2 stream
    static constexpr bool kThinH1 = (kFlags_ & 512) != 0;
    using SmemLayoutBiasThin = Layout<Shape<_256, _1>, Stride<_1, _256>>;
    using TMA_BT = decltype(make_tma_copy(SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), ShapeB{}, StrideB{}),
                                        SmemLayoutBiasThin{}, make_shape(_256{}, _1{}), _1{}));

    static constexpr uint32_t kBytesQ = kBlockM * kHeadDim * sizeof(Element);          // per row
    static constexpr uint32_t kBytesK = kBlockN * kHeadDim * sizeof(Element), kBytesV = kBytesK;   // K, V of one stage (separate transactions)
    static constexpr uint32_t kBytesHalf = kHalfElems * sizeof(float);

    using PipeKV = Pipe<kStagesKV>;                              // full: K and V transactions (2 arrivals); empty = V free (after the tile's last PV)
    using PipeK = Pipe<kStagesKV>;                               // empty only: K free (after the tile's last QK retired); its full barriers are unused
    using PipeB = Pipe<kHalves>;                                 // empty[m]: half slot m = 2c + h free; full[c] (c < 4 used): both halves of slot c landed (2 transactions)
    static constexpr int kArrivalsKV = 4;                        // one per WARP of the owning consumer warpgroup, each after its own wgmma wait: wgmma completion
                                                                 // (wgmma.wait_group) is tracked per warp, so one warp's wait says nothing about a lagging sibling's operand reads
    static constexpr int kArrivalsB = kNumMmaWG * 4 * CR;        // one per consumer WARP of every CTA of the cluster (lane 0 after __syncwarp, once its lanes' init loads have completed)

    struct SharedStorage {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, 1024> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, 1024> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, 1024> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutOnes>, 128> smem_ones;
        cute::array_aligned<float, kSlotElems * kSlotsB, 1024> smem_bias;
        typename PipeKV::SharedStorage pipe_kv;
        typename PipeK::SharedStorage pipe_k;
        typename PipeB::SharedStorage pipe_b;
        int bad;                                                     // some row of this CTA tile failed validation (hot pass)
        cutlass::arch::ClusterTransactionBarrier barrier_q;
    };

    struct Params {
        TMA_Q tma_q; TMA_K tma_k; TMA_V tma_v; TMA_B tma_b; TMA_BT tma_bt;
        ShapeQK shape_qk; ShapeB shape_b;
        Element* out; int64_t so_b, so_n, so_h, so_s;                 // out [B,N,H,S,D] element strides (d stride 1)
        int S, N, H, n_qtiles, n_ktiles;
        float scale;
        unsigned long long* trace;                                     // kFlags & 16: clock64 stamps of CTA (1,1,0), [wg][chunk<32][8]
        int zero;                                                      // always 0: multiplies the period counter into operand bases (defeats LICM, stays uniform)
        int* fix;                                                      // fix list: fix[0] = count (atomic), then (qtile, rowgroup, bh) triples; hot pass appends, SAFE pass consumes
        int* fix_total;                                                // device census: the SAFE pass adds this call's count (FALLBACKS['fix_tiles'])
        int force_fix;                                                 // debug: every CTA tile goes on the fix list (everything recomputed by the SAFE pass)
        // key masks (all null when the call has no mask): see stage_mask in m1_binding.cu
        uint32_t const* maskw;                                         // [B, N, 4 * n_ktiles] attended-key bits per row (bit k%32 of word k/32), zero-padded
        uint8_t const* rowkind;                                        // [B, N]: 0 = row mask == batch OR (folded into -inf bias columns), 1 = irregular INTERVAL (one contiguous run of
                                                                       // attended keys: mask words only on its boundary tiles), 3 = irregular RAGGED (words on every tile), 2 = fully masked (uniform mean of v)
        int const* kcend;                                              // [B]: 32-key columns up to the batch's last attended key (dead trailing columns skipped)
        int const* kcstart;                                            // [B]: first 32-key column holding an attended key
        int wpr;                                                       // mask words per row = 4 * (n_ktiles + 1)
        int const* rowkc0;                                             // [B, N]: the row's first 32-key column holding an attended key
        int const* rowkc1;                                             // [B, N]: one past the row's last such column (0, 0 for a fully-masked row)
    };
};

__device__ __forceinline__ float ex2_approx(float x) { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

// ---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void __launch_bounds__(T::kNumThreads, 1) triattn_m1_kernel(CUTE_GRID_CONSTANT typename T::Params const params) {
    using Element = typename T::Element;
    constexpr int R = T::R, kBlockM = T::kBlockM, kBlockN = T::kBlockN, kHeadDim = T::kHeadDim, CW = T::CW;
    constexpr bool kCheckB = (T::kFlags & 65536) != 0;           // DEBUG: re-read each bias fragment just before its half-slot release and count mismatches vs the init registers in fix_total[1]
    constexpr int kStagger = (T::kFlags & 8388608) ? 200 : ((T::kFlags & 16777216) ? 400 : 0);   // A/B: one-time de-phasing of the 3 consumer warpgroups after the start sync (WG w spins w*kStagger cycles), never re-aligned
    constexpr bool kQK1 = (T::kFlags & 2097152) != 0;            // flag 2097152: QK issued ONE chunk ahead (wait<1>) instead of two: -1..-2 % sustained at S <= 2048, +2.4..+3.7 % at 4096 (power-bound) -> not the default
    constexpr bool kNoProbe = (T::kFlags & 524288) == 0;         // default: blocking waits at the point of use; flag 524288 = the older early test_wait probes + ready tokens (+8 instr/chunk, measured 0.7 % slower)
    constexpr bool kRelAlways = (T::kFlags & 262144) == 0;       // default: bias half-slot release site unconditional, its electing predicate exact (no arrival for a chunk past the range); flag = the release wrapped in a per-chunk branch (ablation)
    constexpr bool kReplayOps = (T::kFlags & 128) != 0;          // ABLATION (timing only): replay (no traffic after the first fill) but consumers still execute every barrier probe/arrive (on completed phases)
    constexpr bool kSafe = T::kSafe;                             // the exact running-max instantiation that recomputes the CTA tiles listed by the hot pass (fix list)
    constexpr bool kList = T::kList;
    constexpr bool kTrace = (T::kFlags & 16) != 0, kNoBias = (T::kFlags & 1) != 0, kReplay = (T::kFlags & 2) != 0 || kReplayOps;
    constexpr bool kNoWait = (T::kFlags & 4) != 0;              // ABLATION (timing only): consumers never wait on full barriers inside the loop (races; producer keeps streaming)
    constexpr bool kPrmtPack = (T::kFlags & 8) != 0;             // P -> bf16 by truncation (PRMT, ALU pipe) instead of round-to-nearest (F2FP, XU pipe)
    constexpr bool kSyncPeriod = (T::kFlags & 64) != 0;          // re-align the 3 consumer warpgroups at every period drain (bounds their skew to keep the shared bias ring's slack usable)
    constexpr bool kRing = (T::kFlags & 32) != 0;                // the 3 consumer warpgroups take turns on the exponential phase (named-barrier token ring 0 -> 1 -> 2 -> 0):
                                                                 // one warp per SM sub-partition owns the MUFU pipe at a time instead of three sharing it at 1/3 rate each
    using SharedStorage = typename T::SharedStorage;
    extern __shared__ char smem_buf[];
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const lane_predicate = cute::elect_one_sync();
    int const wg_idx = cutlass::canonical_warp_group_idx();
    int const tid = threadIdx.x;
    int const lane = tid % 32;

    constexpr int CR = T::CR;
    uint32_t const self_rank = (CR > 1) ? cute::block_rank_in_cluster() : 0u;   // cluster (1, CR, 1): rank = y parity
    uint32_t const peer_rank = self_rank ^ 1u;
    uint16_t const b_mask = (CR > 1) ? uint16_t(3) : uint16_t(1);
    int const S = params.S;
    // hot pass: this CTA's tile straight from the grid, decoded here (before the producer's register budget shrinks)
    int const g_qtile = int(blockIdx.x), g_rg = int(blockIdx.y), g_bh = int(blockIdx.z);
    int const g_b = g_bh / params.H, g_h = g_bh % params.H;

    auto init_barriers = [&]() {                                 // (one elected thread) fresh pipeline state for one CTA tile
        shared.barrier_q.init(1);
        T::PipeKV::init(shared.pipe_kv, T::kArrivalsKV, 2);
        T::PipeK::init(shared.pipe_k, T::kArrivalsKV);
        T::PipeB::init(shared.pipe_b, T::kArrivalsB, 2);
        shared.bad = 0;
        cutlass::arch::fence_barrier_init();
    };
    if (warp_idx == 0 && lane_predicate) {
        cute::prefetch_tma_descriptor(params.tma_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_v.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_b.get_tma_descriptor());
        if constexpr (T::kThinH1) { cute::prefetch_tma_descriptor(params.tma_bt.get_tma_descriptor()); }
        init_barriers();
    }
    typename T::PipeKV pipe_kv(shared.pipe_kv);
    typename T::PipeK pipe_k(shared.pipe_k);
    typename T::PipeB pipe_b(shared.pipe_b);

    if (wg_idx != 0) {
        // the all-ones tile of the row-sum MMA
        for (int idx = tid - 128; idx < int(cute::cosize_v<typename T::SmemLayoutOnes>); idx += T::kNumMmaThreads) { shared.smem_ones[idx] = Element(1.f); }
        cutlass::arch::fence_view_async_shared();                  // generic-proxy smem writes -> visible to wgmma (async proxy)
    }
    if constexpr (CR > 1) { cute::cluster_arrive(); cute::cluster_wait(); } else { __syncthreads(); }   // barrier inits visible cluster-wide before any multicast / remote arrive

    // ================================================= CTA TILES: (q-tile, row triple rg, b*H + h) ===================================
    // The hot pass runs exactly one tile (its grid position); the SAFE pass is persistent over the fix list the hot pass wrote. Producer and consumer warpgroups each run their OWN copy of the tile loop inside their role branch (the two
    // branches never reconverge, so the register reallocation holds in every pass); between list entries all 512 threads meet at a named
    // barrier, one thread re-initialises the pipeline barriers, and they meet again.
    constexpr uint32_t kBarReinit = uint32_t(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier) + 4;
    int const* const list = params.fix;
    int const idx0 = kList ? int(blockIdx.x) : 0;
    auto decode = [&](int idx, int& qtile, int& rg, int& bh, int& b, int& h) __attribute__((always_inline)) -> bool {
        if constexpr (kList) {
            int const count = *reinterpret_cast<int const volatile*>(list);
            if (idx >= count) { return false; }
            qtile = list[1 + 3 * idx]; rg = list[2 + 3 * idx]; bh = list[3 + 3 * idx];
            b = bh / params.H; h = bh % params.H;
            if (idx != idx0) {                                       // fresh pipeline state for this entry
                cutlass::arch::NamedBarrier::sync(T::kNumThreads, kBarReinit);
                if (warp_idx == 0 && lane_predicate) { init_barriers(); }
                cutlass::arch::NamedBarrier::sync(T::kNumThreads, kBarReinit);
            }
            return true;
        } else {
            qtile = g_qtile; rg = g_rg; bh = g_bh; b = g_b; h = g_h;
            return idx == 0;
        }
    };
    // per-tile facts every thread derives identically: first row i0, the 3 row kinds (2 bits each), the CTA's tile stream (key origin
    // key0 = 32 * kc0, n_tiles tiles), each row's tile range [jb[r], je[r]) inside it and its first live key column f[r].
    // A row's live key-column range [f, e): a regular row (kind 0) or a fully-masked one (kind 2) takes the batch's attended range (a
    // fully-masked batch: 1 column of -inf), an irregular row (kinds 1, 3) its own. The CTA stream covers the union over the rows sharing
    // its bias ring (its R rows, and the peer CTA's R rows when the bias is multicast across a row-group pair); row r consumes only tiles
    // [jb[r], je[r]) of it (jb even, so tile parity inside the row's own stream equals the CTA stream's): the producer loads row r's K/V for
    // those tiles only and warpgroup r runs the chunk loop over them, taking part in the shared bias ring's protocol for the tiles before
    // and after (follow_tile). A regular / uniform row's range is the whole stream. No mask: every range is the whole key axis.
    // lockstep: the E-token ring / per-period sync ablations and schedule W need the three warpgroups on the same tiles (every row takes the whole stream)
    constexpr bool kLockstep = (T::kFlags & (32 | 64)) != 0 || T::kW;
    // dead: every row sharing the tile's bias stream (its R rows, clamped at N - 1, and the cluster peer's) is fully masked (kind 2). Their
    // output is written by uniform_rows_kernel whatever this kernel stores, so such a tile streams and computes nothing: both roles leave it
    // right after derive (CTA-uniform; with CR > 1 both CTAs of the pair see the same 2R kinds and leave together).
    auto derive = [&](int rg, int b, int& i0, int& kind3, int& kc0, int& n_tiles, int (&jb)[R], int (&je)[R], int (&f)[R], bool& dead) __attribute__((always_inline)) {
        i0 = rg * R;                                             // rows >= N (incl. cluster padding): loaded clamped, never stored
        kind3 = 0;
        dead = false;
        int const kcmax = 4 * params.n_ktiles - 1;
        kc0 = 0; int kc1 = kcmax + 1;                            // no mask: every range is the whole key axis
        bool anyirr = false;                                     // some row sharing the stream is irregular (else every range is the whole stream: the short path below)
        int e[R];
        if (params.rowkind != nullptr) {                         // masked call: every load issued together (one round trip ahead of the first TMA / the prologue)
            constexpr int RR = R * CR;                           // this CTA's rows, then the bias-pair peer's (they share the stream)
            int kd[RR], r0[RR], r1[RR];
            #pragma unroll
            for (int r = 0; r < RR; ++r) {
                int const i = (r < R) ? i0 + r : (rg ^ 1) * R + (r - R);
                int64_t const row = (int64_t)b * params.N + min(i, params.N - 1);
                kd[r] = int(params.rowkind[row]); r0[r] = params.rowkc0[row]; r1[r] = params.rowkc1[row];
            }
            kc0 = min(params.kcstart[b], kcmax); kc1 = max(params.kcend[b], kc0 + 1);   // the batch's attended range (a fully-masked batch: 1 column of -inf)
            int kinds = 0;
            #pragma unroll
            for (int r = 0; r < RR; ++r) { kinds |= kd[r] << (2 * r); }
            kind3 = kinds & ((1 << (2 * R)) - 1);
            dead = kinds == (0xAAA & ((1 << (2 * RR)) - 1));    // every 2-bit kind == 2
            anyirr = (kinds & 0x555) != 0 && !kLockstep;
            if (anyirr) {                                        // an irregular row takes its own attended range, a regular / fully-masked one the batch's; the stream is their union
                int const bk0 = kc0, bk1 = kc1;
                kc0 = INT_MAX; kc1 = 0;
                #pragma unroll
                for (int r = 0; r < RR; ++r) {
                    bool const own = (kd[r] & 1) != 0;
                    int const fr = own ? min(r0[r], kcmax) : bk0, er = own ? max(r1[r], fr + 1) : bk1;
                    kc0 = min(kc0, fr); kc1 = max(kc1, er);
                    if (r < R) { f[r] = fr; e[r] = er; }
                }
            }
        }
        n_tiles = (kc1 - kc0 + 3) / 4;
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (anyirr) { jb[r] = ((f[r] - kc0) / 4) & ~1; je[r] = min(max((e[r] - kc0 + 3) / 4, jb[r] + 1), n_tiles); }
            else { f[r] = kc0; jb[r] = 0; je[r] = n_tiles; }
        }
    };
    // irregular rows (their own mask words differ from the batch OR) are masked per chunk inside the loop (+0.8 % time on every call, A/B);
    // flag 4096 = the lean loop without it, which sends a CTA tile with an irregular row to the fix list instead
    constexpr bool kMaskInKernelC = kSafe || (T::kFlags & 4096) == 0;

    if (wg_idx == 0) {
        // =============================================== PRODUCER =====================================================
        cutlass::arch::warpgroup_reg_dealloc<32>();              // 128*32 + 384*160 = 65536
        for (int idx = idx0; ; idx += int(gridDim.x)) {
        int qtile, rg, bh, b, h, i0, kind3, kc0, n_tiles; int jb[R], je[R], fr[R]; bool dead;
        if (!decode(idx, qtile, rg, bh, b, h)) { break; }
        int const warp_idx_in_wg = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // Q of the R rows (independent of the mask facts): issued FIRST, so its latency covers the round trip of derive's loads that the
        // K/V and bias coordinates wait for (the lean loop issues it after the fix-list decision: a tile it hands to the SAFE pass loads nothing)
        auto issue_q = [&]() __attribute__((always_inline)) {
            if (warp_idx_in_wg == 1 && lane_predicate) {
                Tensor sQ = make_tensor(make_smem_ptr(shared.smem_q.data()), typename T::SmemLayoutQ{});
                Tensor mQ = params.tma_q.get_tma_tensor(params.shape_qk)(_, _, h, _, b);                 // (S, D, N)
                Tensor gQ = local_tile(mQ, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(qtile, _0{}, _));      // (M, D, N)
                auto block_tma_q = params.tma_q.get_slice(_0{});
                Tensor tQgQ = group_modes<0, 3>(block_tma_q.partition_S(gQ));      // (TMA, N)
                Tensor tQsQ = group_modes<0, 3>(block_tma_q.partition_D(sQ));      // (TMA, R)
                shared.barrier_q.arrive_and_expect_tx(T::kBytesQ * R);
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    int const i = min(rg * R + r, params.N - 1);
                    copy(params.tma_q.with(reinterpret_cast<uint64_t&>(shared.barrier_q), 0), tQgQ(_, i), tQsQ(_, r));
                }
            }
        };
        if constexpr (kMaskInKernelC) { issue_q(); }
        derive(rg, b, i0, kind3, kc0, n_tiles, jb, je, fr, dead); (void)fr;
        if constexpr (kSafe) { if (idx == idx0 && idx == 0 && tid == 0 && params.fix_total != nullptr) { atomicAdd(params.fix_total, *reinterpret_cast<int const volatile*>(list)); } }   // census, once per call
        if (dead) {                                              // every row fully masked (uniform_rows_kernel writes them): no K/V/bias stream; the consumers await the Q load and leave too
            if constexpr (!kSafe && (T::kFlags & 2048) == 0) {   // force_fix keeps its meaning (every tile listed; the SAFE pass leaves the tile the same way)
                if (params.force_fix != 0 && tid == 0) { int const slot = atomicAdd(params.fix, 1); params.fix[1 + 3 * slot] = qtile; params.fix[2 + 3 * slot] = rg; params.fix[3 + 3 * slot] = bh; }
            }
            if constexpr (!kList) { break; } else { continue; }
        }
        if constexpr (!kMaskInKernelC) {                         // lean loop: a CTA tile with an irregular row goes straight to the fix list (SAFE pass)
            if ((kind3 & 0x15) != 0) {
                if (tid == 0) { int const slot = atomicAdd(params.fix, 1); params.fix[1 + 3 * slot] = qtile; params.fix[2 + 3 * slot] = rg; params.fix[3 + 3 * slot] = bh; }
                break;
            }
        }
        int const key0 = 32 * kc0;
        if constexpr (!kMaskInKernelC) { issue_q(); }
        if (warp_idx_in_wg == 1 && lane_predicate) {
            // ---- the K/V stream: tile j, row w -> stage (j & 1) * R + w (use = the tile's index among row w's own tiles >> 1); K waits for the K-free barrier, V for V-free
            Tensor sK = make_tensor(make_smem_ptr(shared.smem_k.data()), typename T::SmemLayoutK{});
            Tensor sV = make_tensor(make_smem_ptr(shared.smem_v.data()), typename T::SmemLayoutV{});
            Tensor mK = params.tma_k.get_tma_tensor(params.shape_qk)(_, _, h, _, b);                 // (S, D, N)
            Tensor mV = params.tma_v.get_tma_tensor(params.shape_qk)(_, _, h, _, b);                 // (S, D, N)
            // the key stream starts at key0 (a multiple of 32): tile j covers keys key0 + 128 j .. +127 (TMA zero-fills keys >= S)
            Tensor gK = local_tile(domain_offset(make_coord(key0, _0{}, _0{}), mK), make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}, _));   // (BN, D, ktile, N)
            Tensor gV = local_tile(domain_offset(make_coord(key0, _0{}, _0{}), mV), make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}, _));   // (BN, D, ktile, N)
            auto block_tma_k = params.tma_k.get_slice(_0{});
            Tensor tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));      // (TMA, ktile, N)
            Tensor tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));      // (TMA, st)
            auto block_tma_v = params.tma_v.get_slice(_0{});
            Tensor tVgV = group_modes<0, 3>(block_tma_v.partition_S(gV));      // (TMA, ktile, N)
            Tensor tVsV = group_modes<0, 3>(block_tma_v.partition_D(sV));      // (TMA, st)

            for (int j = 0; j < n_tiles; ++j) {
                if (kReplay && j >= 2) { break; }                                  // ABLATION (timing only): first ring fill, no traffic afterwards
                int const set = (j & 1) * R;
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (j < jb[r] || j >= je[r]) { continue; }                      // row r does not consume this tile (outside its live range)
                    uint32_t const kph = ((j - jb[r]) >> 1) & 1;                    // use parity of stage (set, r): counted over row r's own tiles
                    int const i = min(i0 + r, params.N - 1);
                    int const st = set + r;
                    pipe_k.producer_wait_empty(st, kph);
                    pipe_kv.producer_expect(st, T::kBytesK);
                    copy(params.tma_k.with(*pipe_kv.full_barrier(st), 0), tKgK(_, j, i), tKsK(_, st));
                }
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (j < jb[r] || j >= je[r]) { continue; }
                    uint32_t const kph = ((j - jb[r]) >> 1) & 1;
                    int const i = min(i0 + r, params.N - 1);
                    int const st = set + r;
                    pipe_kv.producer_wait_empty(st, kph);
                    pipe_kv.producer_expect(st, T::kBytesV);
                    copy(params.tma_v.with(*pipe_kv.full_barrier(st), 0), tVgV(_, j, i), tVsV(_, st));
                }
            }
        } else if (warp_idx_in_wg == 0 && lane_predicate) {
            // ---- the bias stream: tile j, half slot m = 2c + h <- staged block 2*kc + h = 2*kc0 + 8j + m (use j: phase j & 1)
            int const blk0 = 2 * kc0;
            auto sB_half = [&](int mm) { return make_tensor(make_smem_ptr(shared.smem_bias.data() + mm * T::kHalfElems), typename T::SmemLayoutBiasHalf{}); };
            Tensor gB = params.tma_b.get_tma_tensor(params.shape_b)(_, _, _, qtile, bh);             // (256, 8, 8*nk)
            auto block_tma_b = params.tma_b.get_slice(int(self_rank));      // CR > 1: this CTA's half of every box, multicast to both
            Tensor tBgB = group_modes<0, 3>(block_tma_b.partition_S(gB));      // (TMA, 8*nk)
            auto tBsB = [&](int mm) { return group_modes<0, 3>(block_tma_b.partition_D(sB_half(mm))); };
            Tensor gBt = params.tma_bt.get_tma_tensor(params.shape_b)(_, _, _, qtile, bh);
            auto block_tma_bt = params.tma_bt.get_slice(_0{});
            Tensor tBgBt = group_modes<0, 3>(block_tma_bt.partition_S(gBt));
            auto tBsBt = [&](int mm) { return group_modes<0, 3>(block_tma_bt.partition_D(make_tensor(make_smem_ptr(shared.smem_bias.data() + mm * T::kHalfElems), typename T::SmemLayoutBiasThin{}))); };
            for (int j = 0; j < n_tiles; ++j) {
                if (kReplay && j >= 1) { break; }
                uint32_t const bph = j & 1;
                #pragma unroll
                for (int mm = 0; mm < T::kHalves; ++mm) {
                    pipe_b.producer_wait_empty(mm, bph);
                    if (T::kThinH1 && (mm & 1)) {
                        pipe_b.producer_expect(mm >> 1, 1024u);
                        copy(params.tma_bt.with(*pipe_b.full_barrier(mm >> 1), 0), tBgBt(_, blk0 + 8 * j + mm), tBsBt(mm));
                    } else {
                        pipe_b.producer_expect(mm >> 1, T::kBytesHalf);
                        copy(params.tma_b.with(*pipe_b.full_barrier(mm >> 1), b_mask), tBgB(_, blk0 + 8 * j + mm), tBsB(mm));
                    }
                }
            }
        }
        if constexpr (!kList) { break; }
        }   // producer tile loop
        if constexpr (CR > 1) { cute::cluster_arrive(); cute::cluster_wait(); }   // smem (barriers, multicast destinations) outlives the peer's traffic
        return;
    }
    {
    // ================================================= CONSUMERS ======================================================
    cutlass::arch::warpgroup_reg_alloc<160>();
    for (int idx = idx0; ; idx += int(gridDim.x)) {
    int qtile, rg, bh, b, h, i0, kind3, kc0, n_tiles; int jb_[R], je_[R], fr_[R]; bool dead;
    if (!decode(idx, qtile, rg, bh, b, h)) { break; }
    derive(rg, b, i0, kind3, kc0, n_tiles, jb_, je_, fr_, dead);
    if (dead) {                                                  // every row fully masked: nothing to compute (uniform_rows_kernel writes these rows; the producer streams nothing)
        if constexpr (kMaskInKernelC) { shared.barrier_q.wait(0); asm volatile("" ::: "memory"); }   // the tile's Q load was issued ahead of the row kinds: it must land before the CTA retires / the barriers are re-initialised
        if constexpr (!kList) { break; } else { continue; }
    }
    if constexpr (!kMaskInKernelC) { if ((kind3 & 0x15) != 0) { break; } }   // -> fix list (the producer thread appended the tile)
    int const thread_idx = tid - 128;                            // 0..383
    int const cwg = int(__reduce_max_sync(0xffffffffu, unsigned(thread_idx) / 128u));   // uniform register: this warpgroup's row
    int const t128 = thread_idx % 128;
    bool const wg_leader = t128 == 0;
    bool const warp_leader = lane == 0;
    // this warpgroup's row consumes tiles [jb_w, jb_w + n_w) of the CTA stream (derive), n_trail tiles follow them; its first live key column is f_w (absolute)
    int const jb_w = (cwg == 0) ? jb_[0] : ((cwg == 1) ? jb_[1] : jb_[2]);
    int const n_w = ((cwg == 0) ? je_[0] : ((cwg == 1) ? je_[1] : je_[2])) - jb_w;
    int const n_trail = n_tiles - jb_w - n_w;                   // 0 for a regular / uniform row (range = the whole stream)
    int const f_w = (cwg == 0) ? fr_[0] : ((cwg == 1) ? fr_[1] : fr_[2]);
    bool const tracer = kTrace && wg_leader && params.trace != nullptr && qtile == 1 && rg == 1 && bh == 0;
    auto stamp = [&](int it, int k) __attribute__((always_inline)) {
        if constexpr (kTrace) { if (tracer && it >= 0 && it < 32) { params.trace[(cwg * 32 + it) * 8 + k] = clock64(); } }
    };
    float const scale = params.scale;
    constexpr float kLog2e = 1.4426950408889634f;
    float const c_l2 = scale * kLog2e;                           // x (log2 units) = S * c_l2 + nm, S = q.k + bias/scale

    typename T::TiledMmaQK tiled_mma_qk;
    typename T::TiledMmaQKr tiled_mma_qkr;
    constexpr bool kQinRegs = T::kQinRegs;
    typename T::TiledMmaPV tiled_mma_pv;
    typename T::TiledMmaL tiled_mma_l;
    auto wg_mma_qk = tiled_mma_qk.get_slice(0);
    auto wg_mma_pv = tiled_mma_pv.get_slice(0);
    auto wg_mma_l = tiled_mma_l.get_slice(0);
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(t128);
    auto thr_mma_l = tiled_mma_l.get_thread_slice(t128);
    Tensor sOnes = make_tensor(make_smem_ptr(shared.smem_ones.data()), typename T::SmemLayoutOnes{});
    Tensor tOnes = wg_mma_l.partition_fragment_B(sOnes);          // (frag, 1, 2 (k-block))
    int const kv0 = cwg;                                         // this warpgroup's K/V stage of set 0 (set s -> stage kv0 + s*R)

    Tensor sQ = make_tensor(make_smem_ptr(shared.smem_q.data()), typename T::SmemLayoutQ{});
    Tensor tQ = wg_mma_qk.partition_fragment_A(local_tile(sQ, make_shape(_64{}, Int<kHeadDim>{}), make_coord(_, _0{}, _)));   // (frag, 1, 2 (k-block), 2 (h), R) smem descriptors
    static_assert(decltype(size<2>(tQ))::value == 2 && decltype(size<3>(tQ))::value == 2 && decltype(size<4>(tQ))::value == R);
    // register copies of this warpgroup's two 64 x 32 Q halves (kQinRegs): A fragments of the RS wgmma, filled by LDSM after the Q barrier
    auto thr_mma_qkr = tiled_mma_qkr.get_thread_slice(t128);
    Tensor tQr0 = thr_mma_qkr.partition_fragment_A(local_tile(sQ(_, _, 0), make_shape(_64{}, Int<kHeadDim>{}), make_coord(0, 0)));   // (frag, 1, 2 (k-block))
    Tensor tQr1 = make_fragment_like(tQr0);
    auto load_q_regs = [&]() __attribute__((always_inline)) {
        if constexpr (kQinRegs) {
            auto smem_tiled_copy_q = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma_qkr);
            auto smem_thr_copy_q = smem_tiled_copy_q.get_thread_slice(t128);
            #pragma unroll
            for (int hq = 0; hq < 2; ++hq) {
                Tensor sQh = local_tile(sQ(_, _, cwg), make_shape(_64{}, Int<kHeadDim>{}), make_coord(hq, 0));
                Tensor tQsQ = smem_thr_copy_q.partition_S(sQh);
                Tensor tQrQ = smem_thr_copy_q.retile_D(hq == 0 ? tQr0 : tQr1);
                cute::copy(smem_tiled_copy_q, tQsQ, tQrQ);
            }
        }
    };
    // K / V operand descriptor tensors over (k-block, chunk column c, stage of set 0|1), based at this warpgroup's stage of set 0 and
    // offset by params.zero * p per period (== 0; keeps the per-period descriptors out of loop-invariant hoisting, on the uniform path)
    auto mk_K = [&](int zoff) {
        Tensor sKz = make_tensor(make_smem_ptr(shared.smem_k.data() + cwg * T::kStageElemsK + zoff), typename T::SmemLayoutK{});
        if constexpr (kQinRegs) { return tiled_mma_qkr.get_slice(0).partition_fragment_B(local_tile(sKz, make_shape(Int<CW>{}, Int<kHeadDim>{}), make_coord(_, _0{}))); }
        else { return wg_mma_qk.partition_fragment_B(local_tile(sKz, make_shape(Int<CW>{}, Int<kHeadDim>{}), make_coord(_, _0{}))); }        // (frag, 1, 2, c, st)
    };
    auto mk_V = [&](int zoff) {
        Tensor sVz = make_tensor(make_smem_ptr(shared.smem_v.data() + cwg * T::kStageElemsV + zoff), typename T::SmemLayoutVt{});    // (D, BN, st) view
        return wg_mma_pv.partition_fragment_B(local_tile(sVz, make_shape(Int<kHeadDim>{}, Int<CW>{}), make_coord(_0{}, _)));       // (frag, 1, 2, c, st)
    };
    using OpK = decltype(mk_K(0)); using OpV = decltype(mk_V(0));

    Tensor cO = make_identity_tensor(make_shape(_64{}, Int<kHeadDim>{}));
    Tensor tOcO = thr_mma_pv.partition_C(cO);
    Tensor tOcO_rc = make_tensor(tOcO.data(), flash::convert_layout_acc_rowcol(tOcO.layout()));
    using AccC = decltype(partition_fragment_C(tiled_mma_qk, make_shape(_64{}, Int<CW>{})));                  // 16 fp32
    using AccO = decltype(partition_fragment_C(tiled_mma_pv, make_shape(_64{}, Int<kHeadDim>{})));            // 16 fp32
    using AccL = decltype(partition_fragment_C(tiled_mma_l, make_shape(_64{}, _8{})));                         // 4 fp32: rows (mi) x 2 equal columns
    static_assert(decltype(size(AccC{}))::value == 16 && decltype(size(AccO{}))::value == 16 && decltype(size(AccL{}))::value == 4);
    constexpr int kNRows = 2, kNC = CW / 4, kNKB = CW / 16;      // accumulator rows / rowcol columns per thread per chunk, PV k-blocks
    constexpr int kNColsO = kHeadDim / 4;
    AccO acc_o[2];                                               // per q half h
    AccL acc_l[2];                                               // row sums per half
    AccC accC[4];                                                // S chunk k in accC[k & 3] (QK issued two chunks ahead)
    auto p_proto = make_tensor_like<Element>(make_tensor(accC[0].data(), flash::convert_layout_acc_Aregs<typename T::TiledMmaPV>(accC[0].layout())));
    decltype(p_proto) PCb[2];                                    // bf16 P of chunk k in PCb[k & 1]
    static_assert(decltype(size<2>(p_proto))::value == kNKB);
    float nm[2][kNRows];                                         // per (h, accumulator row) offset, log2 units (see seeding below)
    int const kind = (kind3 >> (2 * cwg)) & 3;                   // this warpgroup's row: 0 regular, 1 / 3 irregular (apply its mask words), 2 fully masked (output = mean of v, written by uniform_rows_kernel; next to a live row it is computed here as a regular row and not validated -- a tile of only such rows was left above)
    // irregular rows (mask != the batch OR): their mask words are applied per chunk inside the loop (kMaskInKernel, the default; the lean
    // flag-4096 build sends their CTA tile to the fix list instead)
    constexpr bool kMaskInKernel = kMaskInKernelC;
    bool const irregular = ((kind & 1) != 0) && kMaskInKernel, ragged = (kind == 3) || kLockstep, uniform = kind == 2;   // kind 1: one contiguous run (mask words only on its boundary tiles), 3: ragged (words on every tile; lockstep builds treat every irregular row so)
    bool bad = (((kind & 1) != 0) && !kMaskInKernel) || (!kSafe && params.force_fix != 0);   // this thread has a row the hot pass cannot finish: the CTA tile goes on the fix list
    uint32_t const* maskrow = irregular ? params.maskw + ((int64_t)b * params.N + min(i0 + cwg, params.N - 1)) * params.wpr + kc0 + 4 * jb_w : nullptr;   // word of this warpgroup's chunk column x (relative to its first tile) at maskrow[x]
    // Which of this warpgroup's tiles apply the row's mask words: an interval row needs them only in the tile holding its first live
    // column (tile jf01 in {0, 1} of its range; tile 0 before it is dead and takes the words too) and in its last tile; a ragged row in all.
    // Evaluated per GENERAL period into 3 bits (tiles 2p, 2p+1, 2p+2: word_tiles), tested per chunk with a compile-time tile index
    // (the SAFE pass applies an irregular row's words on every tile instead: nothing of this stays live there).
    int const jf01 = ragged ? (1 << 28) : (f_w - kc0) / 4 - jb_w;
    auto word_tiles = [&](int p2) __attribute__((always_inline)) -> uint32_t {   // p2 = 2p
        return uint32_t(p2 <= jf01 || p2 == n_w - 1) | (uint32_t(p2 + 1 <= jf01 || p2 + 1 == n_w - 1) << 1) | (uint32_t(p2 + 2 <= jf01 || p2 + 2 == n_w - 1) << 2);
    };
    uint32_t wt = word_tiles(0);
    // Bias half-slot releases are EXACT: one arrival per half slot per tile of this warpgroup's range, none for the chunks the schedule runs
    // past its last tile (their arrival would count into an open phase of a slot the longer rows still stream through). The electing
    // predicate of a chunk's release is picked by its tile offset in the period (compile-time): lead0 for tile 2p (always inside), lead1 /
    // lead2 for tiles 2p+1 / 2p+2 = the warp leader AND that tile being inside the range, evaluated once per period.
    bool const lead0 = warp_leader; bool lead1 = warp_leader, lead2 = warp_leader;
    // Period kinds of this warpgroup: a regular / uniform row (range = the whole CTA stream) runs the STEADY body everywhere; an interval row
    // runs the GENERAL body in period 0 (first-live-tile words) and from period g_from on (the periods touching its last tile: last-tile
    // words); a ragged row and the SAFE pass run it everywhere.
    int const g_from = (kSafe || ragged) ? 0 : max((n_w - 2) >> 1, 0);
    clear(acc_o[0]); clear(acc_o[1]); clear(acc_l[0]); clear(acc_l[1]);
    warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(acc_l[0]); warpgroup_fence_operand(acc_l[1]);   // pin the zeroing here

    // ---- per-chunk primitives (h, c, st, slot compile-time) ----
    float const* bias_thread = shared.smem_bias.data() + t128 * 4;           // + slot*4096 + (h*4+u)*512 floats
    auto check_chunk = [&](AccC const& acc, int c, int hh) __attribute__((always_inline)) {   // DEBUG (kCheckB): does the slot still hold what init loaded?
        int bad_ = 0;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            float const volatile* pv_ = reinterpret_cast<float const volatile*>(bias_thread + c * T::kSlotElems + (hh * 4 + u) * 512);
            #pragma unroll
            for (int x = 0; x < 4; ++x) { bad_ |= (__float_as_uint(pv_[x]) != __float_as_uint(acc(4 * u + x))); }
        }
        if (bad_) { atomicAdd(params.fix_total + 1, 1); }
    };
    auto init_chunk = [&](AccC& acc, auto cc, auto hc) __attribute__((always_inline)) {     // acc <- bias/scale fragment (4 x LDS.128)
        constexpr int c = decltype(cc)::value, hh = decltype(hc)::value;
        if constexpr (kNoBias) {
            #pragma unroll
            for (int v = 0; v < 16; ++v) { acc(v) = 0.f; }
        } else {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                float4 const f4 = *reinterpret_cast<float4 const*>(bias_thread + c * T::kSlotElems + (hh * 4 + u) * 512);
                acc(4 * u + 0) = f4.x; acc(4 * u + 1) = f4.y; acc(4 * u + 2) = f4.z; acc(4 * u + 3) = f4.w;
            }
        }
    };
    auto issue_qk = [&](AccC& acc, OpK const& tK, auto hc, auto cc, auto stc) __attribute__((always_inline)) {   // S += Q_h K_c^T (accumulates onto the bias init)
        constexpr int hh = decltype(hc)::value, c = decltype(cc)::value, st = decltype(stc)::value;
        warpgroup_fence_operand(acc);
        warpgroup_arrive();
        if constexpr (kQinRegs) {
            tiled_mma_qkr.accumulate_ = GMMA::ScaleOut::One;
            auto& tQr = *((hh == 0) ? &tQr0 : &tQr1);
            #pragma unroll
            for (int kb = 0; kb < 2; ++kb) { cute::gemm(tiled_mma_qkr, tQr(_, _, kb), tK(_, _, kb, c, st), acc); }
        } else {
            tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
            #pragma unroll
            for (int kb = 0; kb < 2; ++kb) { cute::gemm(tiled_mma_qk, tQ(_, _, kb, hh, cwg), tK(_, _, kb, c, st), acc); }
        }
        warpgroup_commit_batch();                                    // no fence here: the accumulator is in flight until the next wait
    };
    auto issue_pv = [&](decltype(p_proto)& tP, OpV const& tV, auto hc, auto cc, auto stc) __attribute__((always_inline)) {   // no fence on acc_o (chained PVs)
        constexpr int hh = decltype(hc)::value, c = decltype(cc)::value, st = decltype(stc)::value;
        warpgroup_fence_operand(tP);
        warpgroup_arrive();
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One; tiled_mma_l.accumulate_ = GMMA::ScaleOut::One;
        #pragma unroll
        for (int kb = 0; kb < kNKB; ++kb) { cute::gemm(tiled_mma_pv, tP(_, _, kb), tV(_, _, kb, c, st), acc_o[hh]); }
        #pragma unroll
        for (int kb = 0; kb < kNKB; ++kb) { cute::gemm(tiled_mma_l, tP(_, _, kb), tOnes(_, _, kb), acc_l[hh]); }     // same commit group: l over the same bf16 P
        warpgroup_commit_batch();
    };
    auto pack_chunk = [&](AccC& acc, decltype(p_proto)& tP) __attribute__((always_inline)) {
        auto dst32 = recast<uint32_t>(tP);
        #pragma unroll
        for (int pr = 0; pr < CW / 4; ++pr) {
            if constexpr (kPrmtPack) {   // high halves of the two floats: bf16 truncation; the ones-column row sum sees the same values
                uint32_t u; asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(u) : "r"(__float_as_uint(acc(2 * pr))), "r"(__float_as_uint(acc(2 * pr + 1))));
                dst32(pr) = u;
            } else {
                __nv_bfloat162 const h2 = __floats2bfloat162_rn(acc(2 * pr), acc(2 * pr + 1));
                dst32(pr) = reinterpret_cast<uint32_t const&>(h2);
            }
        }
    };
    auto chunk_rowmax = [&](AccC& a, int mi) __attribute__((always_inline)) {               // max over this chunk's 32 keys of accumulator row mi (quad shuffles)
        Tensor s0 = make_tensor(a.data(), flash::convert_layout_acc_rowcol(a.layout()));
        float mx = s0(mi, 0);
        #pragma unroll
        for (int ni = 1; ni < kNC; ++ni) { mx = max(mx, s0(mi, ni)); }
        mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        return max(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    };
    // ---- per-chunk numerics ahead of the exponentials: irregular rows put -inf on their masked keys (mask word of the chunk, this
    //      thread's 8 key columns 8u + 2(t%4) + e); rows not yet seeded take their offset from the first chunk holding a finite logit
    //      (nm = -(max * c_l2) - kShift, ONCE per row; unseeded rows have seen only -inf so far, p = 0 with nm = 0); uniform rows skip both.
    // Operating point (hot pass): the offset sits kShift = 64 log2 units ABOVE the row max seen at seeding, so the seed key has
    // p = 2^-64 exactly (a bf16 value: the dominant p is unrounded) and a later logit may exceed the seed by ~190 log2 units before
    // ex2 overflows; keys > ~62 units below flush to 0 (invisible in fp32). The SAFE pass runs at shift 0 with a running max.
    constexpr float kShift = kSafe ? 0.f : 64.f;
    uint32_t seeded = 0;                                         // bit 2*hh + mi
    bool need_seed = true;                                       // warp-uniform: some row of this warp is unseeded
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh) { nm[hh][0] = 0.f; nm[hh][1] = 0.f; }
    auto prep_chunk = [&](AccC& acc, int hh, int kc, bool words, auto seedc) __attribute__((always_inline)) {   // words (warpgroup-uniform): this chunk's tile applies the row's mask words
        constexpr bool kSeedHere = decltype(seedc)::value;      // hot pass: only the prologue chunks (this warpgroup's first key column) seed; a row still unseeded afterwards -> fix list. SAFE: every chunk
        Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
        if (words) {                                             // (interval row: its first-live and last tiles; ragged row: all; never in a STEADY period)
            uint32_t const word = maskrow[kc] >> (2 * (t128 % 4));
            #pragma unroll
            for (int ni = 0; ni < kNC; ++ni) {
                bool const keep = (word >> (8 * (ni >> 1) + (ni & 1))) & 1u;
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) { s_rc(mi, ni) = keep ? s_rc(mi, ni) : -INFINITY; }
            }
        }
        if constexpr (kSeedHere) {
        if (need_seed) {
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                float const mx = chunk_rowmax(acc, mi);
                uint32_t const bit = 1u << (2 * hh + mi);
                if (!(seeded & bit) && mx > -INFINITY) { nm[hh][mi] = -(mx * c_l2) - kShift; seeded |= bit; }
            }
            need_seed = __any_sync(0xffffffffu, seeded != 0xfu);
        }
        }
    };
    auto exp_chunk = [&](AccC& acc, int hh, int kc, bool words, auto seedc) __attribute__((always_inline)) {   // p = 2^(S * c_l2 + nm) in place (fp32)
        Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
        prep_chunk(acc, hh, kc, words, seedc);
        if constexpr (kSafe) {
            // exact running max (every wgmma of this warpgroup is retired here, so O and l may be rescaled): the offset follows the
            // row max; chunk k-1 (the other half) is unaffected, chunk k-2 (this half) is already inside O
            Tensor o_rc = make_tensor(acc_o[hh].data(), flash::convert_layout_acc_rowcol(acc_o[hh].layout()));
            Tensor l_rc = make_tensor(acc_l[hh].data(), flash::convert_layout_acc_rowcol(acc_l[hh].layout()));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                float const cand = -(chunk_rowmax(acc, mi) * c_l2);
                if (cand < nm[hh][mi]) {                                                 // the max grew: rescale O, l by 2^(cand - nm) < 1
                    float const f = ex2_approx(cand - nm[hh][mi]);
                    #pragma unroll
                    for (int ni = 0; ni < kNColsO; ++ni) { o_rc(mi, ni) *= f; }
                    l_rc(mi, 0) *= f; l_rc(mi, 1) *= f;
                    nm[hh][mi] = cand;
                }
            }
        }
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi])); }
        }
        warpgroup_fence_operand(acc);
    };
    // E-phase token ring: named barrier kRingBar0 + w has 256 participants = warpgroup w (sync = wait for the token) + warpgroup w-1
    // (arrive = pass the token after its own E phase). Warpgroup 2 pre-arrives on barrier 0 once so warpgroup 0's first E can start.
    constexpr uint32_t kRingBar0 = uint32_t(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    auto ring_wait = [&]() __attribute__((always_inline)) { if constexpr (kRing) { cutlass::arch::NamedBarrier::sync(256, kRingBar0 + cwg); } };
    auto ring_pass = [&]() __attribute__((always_inline)) { if constexpr (kRing) { cutlass::arch::NamedBarrier::arrive(256, kRingBar0 + (cwg == R - 1 ? 0 : cwg + 1)); } };
    if constexpr (kRing) { if (cwg == R - 1) { cutlass::arch::NamedBarrier::arrive(256, kRingBar0); } }
    // drained (hot pass): keep every row's sum inside [2^-84, 2^-44] by an INTEGER offset change k = floor(log2 l) + 64 (one unsigned
    // compare per row; the rescale 2^-k is exact); `pend` = the p values of the one chunk (half h_pend) that is exp'd but not yet packed.
    // Rows whose l is 0 / inf / NaN take an arbitrary k here and fail validation at the end (fix list).
    auto l_check = [&](AccC& pend, int h_pend) __attribute__((always_inline)) {
        if constexpr (!kSafe) {
        constexpr uint32_t kLo = uint32_t(127 - 84) << 23, kWidth = (uint32_t(127 - 44) << 23) - kLo;
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            warpgroup_fence_operand(acc_o[hh]); warpgroup_fence_operand(acc_l[hh]);
            Tensor o_rc = make_tensor(acc_o[hh].data(), flash::convert_layout_acc_rowcol(acc_o[hh].layout()));
            Tensor l_rc = make_tensor(acc_l[hh].data(), flash::convert_layout_acc_rowcol(acc_l[hh].layout()));
            bool out = false;
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) { out |= (__float_as_uint(l_rc(mi, 0)) - kLo) > kWidth; }
            if (__any_sync(0xffffffffu, out)) {
                Tensor p_rc = make_tensor(pend.data(), flash::convert_layout_acc_rowcol(pend.layout()));
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) {
                    uint32_t const lb = __float_as_uint(l_rc(mi, 0));
                    bool const o_ = (lb - kLo) > kWidth;
                    int k = int((lb >> 23) & 0xffu) - 127 + 64;                             // floor(log2 l) + 64
                    k = o_ ? min(max(k, -126), 126) : 0;
                    float const f = __uint_as_float(uint32_t(127 - k) << 23);               // 2^-k, exact
                    nm[hh][mi] -= float(k);
                    #pragma unroll
                    for (int ni = 0; ni < kNColsO; ++ni) { o_rc(mi, ni) *= f; }
                    l_rc(mi, 0) *= f; l_rc(mi, 1) *= f;
                    if (hh == h_pend) {
                        #pragma unroll
                        for (int ni = 0; ni < kNC; ++ni) { p_rc(mi, ni) *= f; }
                    }
                }
            }
            warpgroup_fence_operand(acc_o[hh]); warpgroup_fence_operand(acc_l[hh]);
        }
        }
    };
    // dep: OR of one register per LDS.128 of the init that read this half slot (all lanes: the shuffle-free __syncwarp + lane 0's own
    // dependency covers lane 0; the other lanes' loads are covered by making dep warp-wide via __reduce_or_sync)
    // The half slot may be re-filled by TMA as soon as the 12th warp arrives. An mbarrier arrive does NOT wait for this warp's earlier
    // ld.shared of the slot to have RETURNED (the loads' only consumers are the QK wgmma much later; a compiler-level operand fence emits no
    // instruction), so a plain arrive placed after the init loads races with the refill. The arrive's address is therefore made
    // data-dependent on one register of each of the 4 LDS.128 (x params.zero == 0), OR-reduced over the warp: no lane's arrive can issue
    // before every lane's loads have returned.
    // (an LDS.128 is one warp instruction: its scoreboard clears for all lanes at once, so lane 0's own registers carry the dependency
    // for the whole warp; __syncwarp keeps the lanes converged at the arrive)
    constexpr bool kLateRel = (T::kFlags & 8192) == 0;           // default: release after the consuming QK issue (the HGMMA issue itself waits for the loads): plain arrive.
                                                                 // 8192: release one QK earlier (top of the body) with the data-dependent arrive below
    auto bias_release_by = [&](int c, bool leader, uint32_t dep) __attribute__((always_inline)) {   // leader: this warp's electing lane, false in every lane = no arrival (a chunk past the range)
        asm volatile("" ::: "memory"); __syncwarp();
        if constexpr (CR > 1) { pipe_b.release2(c, leader, peer_rank); (void)dep; }
        else if constexpr (kLateRel) { pipe_b.release(c, leader); (void)dep; }
        else { pipe_b.release_dep(c, leader, dep, params.zero); }
    };
    auto bias_release = [&](int c, uint32_t dep) __attribute__((always_inline)) { bias_release_by(c, warp_leader, dep); };
    auto dep_of = [&](AccC const& a) __attribute__((always_inline)) {
        return __float_as_uint(a(0)) | __float_as_uint(a(4)) | __float_as_uint(a(8)) | __float_as_uint(a(12));
    };   // this warp's lanes have read slot c (values fenced into registers by the preceding issue_qk)

    // ---- body for chunk dd (0..15) of period p: chunk k = 16p + 2 + dd, pair index e = dd + 2: h = e&1, c = (e>>1)&3, tile 2p + (e>>3)
    //      (stage set (e>>3)&1). Body 0 follows a full drain (prologue or period end) and issues no wait; the others:
    //      wait<2> -> QK(k), PV(k-3) retired (QK(k+1), PV(k-2) may pend) | K/V release of chunk k-3's tile if that was its last chunk |
    //      chunk k+2: K/V wait at a tile start, QK issue (its S buffer was bias-initialised by body k-1) | E(k) | pack(k-1), PV(k-1)
    //      = the LAST wgmma commit | chunk k+3: bias-slot wait (h = 0 chunks), S init by LDS into the buffer pack(k-1) just freed
    //      (+ slot release after an h = 1 init): the loads have a whole body to land before QK(k+3) is issued.
    //      Two instantiations: STEADY (kGen false: no tile of the period holds a boundary of the warpgroup's live range -> no mask-word
    //      code) and GENERAL (kGen true: mask words on the tiles flagged in wt). Both guard the waits by the range end n_w and elect the
    //      bias-slot releases exactly (lead0/1/2), as the last period of any row needs.
    bool kv_tok = true;                                          // K/V stage of the next body's QK chunk known-ready (probed one body ahead); false -> that body waits
    auto body = [&](auto ddc, auto genc, int p, uint32_t pp, OpK const& tK, OpV const& tV) __attribute__((always_inline)) {
        constexpr bool kGen = decltype(genc)::value;
        constexpr int dd = decltype(ddc)::value, e = dd + 2;
        constexpr int hh = e & 1;
        constexpr int e2 = e + (kQK1 ? 1 : 2), h2 = e2 & 1, c2 = (e2 >> 1) & 3, t2 = e2 >> 3;               // the chunk whose QK this body issues: k+2 (k+1 under kQK1; t2 == 2: tile 2p+2)
        constexpr int e3 = e + 3, h3 = e3 & 1, c3 = (e3 >> 1) & 3, t3 = e3 >> 3;                             // chunk k+3
        constexpr int ep = e - 1, hp = ep & 1, cp = (ep >> 1) & 3, tp = ep >> 3;                            // chunk k-1
        constexpr int bq = e & 3, bn = e2 & 3, bp = (e - 1) & 3, pb = (e - 1) & 1;                           // bp == (e + 3) & 3: chunk k+3's buffer
        int const k = 16 * p + e;
        stamp(k, 0);
        if constexpr (kSafe) { warpgroup_wait<0>(); warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(acc_l[0]); warpgroup_fence_operand(acc_l[1]); }
        else if constexpr (dd != 0) { warpgroup_wait<kQK1 ? 1 : 2>(); }
        // Early release (default): fence chunk k+2's S buffer here and release its bias half slot at once = shortest slot hold time;
        // the init loads issued at the end of the previous body must have completed here (some long-scoreboard stalls). Flag 8192
        // (A/B, measured 1-3 % slower): let the loads complete under the barrier work below and release right after issue_qk.
        constexpr bool kEarlyRel = (T::kFlags & 8192) != 0;
        warpgroup_fence_operand(accC[bq]); warpgroup_fence_operand(PCb[pb]);
        if constexpr (kEarlyRel) { warpgroup_fence_operand(accC[bn]); }
        int const j2 = 2 * p + t2, j3 = 2 * p + t3;                                                         // tiles of chunks k+2, k+3 (this warpgroup's own tile index)
        bool const rel2 = j2 < n_w && (kRelAlways || !kReplay || kReplayOps || j2 < 1);                     // chunk k+2's tile is inside the range (debug check; the replay ablation's guarded form)
        bool const leadk2 = (t2 == 0) ? lead0 : ((t2 == 1) ? lead1 : lead2);                              // its release's electing predicate (exact: no arrival past the range)
        auto release2 = [&](uint32_t dep) __attribute__((always_inline)) {                              // chunk k+2's half slot may be refilled
            if constexpr (kRelAlways) { bias_release_by(e2 & 7, leadk2, dep); } else { if (rel2) { bias_release(e2 & 7, dep); } }
        };
        uint32_t const dep2 = dep_of(accC[bn]);                                                             // chunk k+2's init registers (its QK is issued below)
        if constexpr (kCheckB) { if (rel2) { check_chunk(accC[bn], c2, h2); } }
        if constexpr (kEarlyRel) { release2(dep2); }
        // early non-blocking probes (their latency overlaps the work below): bias slot of chunk k+3 (needed by the init at the end of
        // this body) and the K/V stage of chunk k+3's tile (needed by QK(k+3) at the top of the next body)
        bool const need_b3 = (h3 == 0) && !kNoWait && j3 < n_w && (!kReplay || kReplayOps || j3 < 1);
        bool const need_kv3 = ((e3 & 7) == 0) && !kNoWait && j3 < n_w && (!kReplay || kReplayOps || j3 < 2);
        uint32_t const bph3 = kReplayOps ? 0u : uint32_t(t3 & 1);                                          // replay-ops: probe the completed first-fill phase
        uint32_t const kvph3 = kReplayOps ? 0u : ((t3 == 2) ? (pp ^ 1u) : pp);
        bool b3_ready = !kNoProbe, kv3_ready = !kNoProbe;                                                  // kNoProbe: 'not known ready' -> blocking waits below
        if constexpr (h3 == 0 && !kNoProbe) { if (need_b3) { b3_ready = pipe_b.test_full(c3, bph3); } }
        if constexpr ((e3 & 7) == 0 && !kNoProbe) { if (need_kv3) { kv3_ready = pipe_kv.test_full(kv0 + (t3 & 1) * R, kvph3); } }
        stamp(k, 1);
        if (!kReplay || kReplayOps) {
            if constexpr (dd == 0) { if (p > 0) { pipe_kv.release(kv0 + R, warp_leader); } }                    // V: chunk k-3 = 16p-1 = last chunk of tile 2p-1 (set 1), its PV retired
            if constexpr (dd == 8) { pipe_kv.release(kv0, warp_leader); }                                     // V: chunk k-3 = 16p+7 = last chunk of tile 2p (set 0)
            if constexpr (dd == 5) { pipe_k.release(kv0, warp_leader); }                                      // K: QK(k = 16p+7) = tile 2p's last QK retired at this wait
            if constexpr (dd == 13) { pipe_k.release(kv0 + R, warp_leader); }                                 // K: tile 2p+1's last QK retired
        }
        if constexpr ((e2 & 7) == 0) { if (!kv_tok && !kNoWait && j2 < n_w && (!kReplay || kReplayOps || j2 < 2)) { pipe_kv.wait_full(kv0 + (t2 & 1) * R, kReplayOps ? 0u : ((t2 == 2) ? (pp ^ 1u) : pp)); } }   // first chunk of tile j2 (probed by the previous body)
        stamp(k, 2);
        issue_qk(accC[bn], tK, Int<h2>{}, Int<c2>{}, Int<(t2 & 1) * R>{});                                 // (its operand fence completes chunk k+2's init loads) phantom past the end: stale smem, result unused
        if constexpr (!kEarlyRel) { release2(dep2); }                                                       // half slot of chunk k+2 may be refilled
        ring_wait();
        stamp(k, 3);
        exp_chunk(accC[bq], hh, 4 * (2 * p + (e >> 3)) + ((e >> 1) & 3), kGen && irregular && (kSafe || ((wt >> (e >> 3)) & 1u)), cute::bool_constant<kSafe>{});   // (SAFE: words on every tile of an irregular row, no word-tile state)
        ring_pass();
        stamp(k, 4);
        pack_chunk(accC[bp], PCb[pb]);
        issue_pv(PCb[pb], tV, Int<hp>{}, Int<cp>{}, Int<(tp & 1) * R>{});
        stamp(k, 5);
        if constexpr (h3 == 0) { if (need_b3 && !b3_ready) { pipe_b.wait_full(c3, bph3); } }               // slot c3 (both halves) holds tile j3's column: probed above, block only if it was not ready
        init_chunk(accC[bp], Int<c3>{}, Int<h3>{});                                                         // S(k+3) <- bias (buffer of chunk k-1, packed above); released at the top of the next body
        if constexpr ((e3 & 7) == 0 && !kNoProbe) { if (need_kv3 && !kv3_ready) { kv3_ready = pipe_kv.test_full(kv0 + (t3 & 1) * R, kvph3); } }
        if constexpr (!kNoProbe) { kv_tok = kv3_ready; }                                                    // -> next body's QK(k+3); kNoProbe: kv_tok stays false = always wait there
        stamp(k, 6);
    };
    // drained state: every wgmma retired; the exp'd, unpacked chunk (always an h = 1 chunk) sits in accC[buf]
    auto drain = [&](auto bufc) __attribute__((always_inline)) {
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(acc_l[0]); warpgroup_fence_operand(acc_l[1]);
        warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]); warpgroup_fence_operand(accC[2]); warpgroup_fence_operand(accC[3]);
        warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]);
        l_check(accC[decltype(bufc)::value], 1);
        if constexpr (kSyncPeriod) { cutlass::arch::NamedBarrier::sync(T::kNumMmaThreads, kRingBar0 + 3); }
    };

    if constexpr (T::kW) {
    OpK const tK0 = mk_K(0); OpV const tV0 = mk_V(0);
    // one wgmma issue group per chunk body: QK(k+1) [+ PV(k-1) and its ones-MMA]: a single warpgroup arrive (which waits for every
    // outstanding register write of the warpgroup, so it is placed where no MUFU/F2FP result is in flight) and a single commit
    auto issue_group = [&](auto with_pv, AccC& accq, OpK const& tK, auto h1c, auto c1c, auto st1c, decltype(p_proto)& tP, OpV const& tV, auto hpc, auto cpc, auto stpc) __attribute__((always_inline)) {
        constexpr bool kPV = decltype(with_pv)::value;
        constexpr int h1 = decltype(h1c)::value, c1 = decltype(c1c)::value, st1 = decltype(st1c)::value;
        constexpr int hp = decltype(hpc)::value, cp = decltype(cpc)::value, stp = decltype(stpc)::value;
        warpgroup_fence_operand(accq);
        if constexpr (kPV) { warpgroup_fence_operand(tP); }
        warpgroup_arrive();
        if constexpr (kQinRegs) {
            tiled_mma_qkr.accumulate_ = GMMA::ScaleOut::One;
            auto& tQr = *((h1 == 0) ? &tQr0 : &tQr1);
            #pragma unroll
            for (int kb = 0; kb < 2; ++kb) { cute::gemm(tiled_mma_qkr, tQr(_, _, kb), tK(_, _, kb, c1, st1), accq); }
        } else {
            tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
            #pragma unroll
            for (int kb = 0; kb < 2; ++kb) { cute::gemm(tiled_mma_qk, tQ(_, _, kb, h1, cwg), tK(_, _, kb, c1, st1), accq); }
        }
        if constexpr (kPV) {
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One; tiled_mma_l.accumulate_ = GMMA::ScaleOut::One;
            #pragma unroll
            for (int kb = 0; kb < kNKB; ++kb) { cute::gemm(tiled_mma_pv, tP(_, _, kb), tV(_, _, kb, cp, stp), acc_o[hp]); }
            #pragma unroll
            for (int kb = 0; kb < kNKB; ++kb) { cute::gemm(tiled_mma_l, tP(_, _, kb), tOnes(_, _, kb), acc_l[hp]); }
        }
        warpgroup_commit_batch();
    };
    // =========================== schedule W: chunk k = 16p + e, body dd <-> e = dd + 2 (2..17; e >= 16: tile 2p+2) ===========================
    //   top: wait<0> retires the previous body's group = QK(k) and PV(k-2) (body 0 of a period follows the drain and issues no wait) |
    //   K/V releases | K/V stage wait when chunk k+1 opens a tile | ONE issue group: QK(k+1) into S buffer (k+1)&1 (initialised at the
    //   end of body k-1) + PV(k-1) with its ones-MMA from P buffer (k-1)&1 (packed at the end of body k-1) | chunk k+1's half-slot
    //   release | mask + E(k) in S buffer k&1 | pack(k) -> P buffer k&1 | bias-slot wait (h == 0 chunks) + init S(k+2) into S buffer k&1.
    auto bodyW = [&](auto ddc, int p, uint32_t pp, OpK const& tK, OpV const& tV) __attribute__((always_inline)) {
        constexpr int dd = decltype(ddc)::value, e = dd + 2, hh = e & 1, c = (e >> 1) & 3, t = e >> 3;
        constexpr int e1 = e + 1, h1 = e1 & 1, c1 = (e1 >> 1) & 3, t1 = e1 >> 3;                             // chunk k+1: its QK is issued here
        constexpr int e2 = e + 2, h2 = e2 & 1, c2 = (e2 >> 1) & 3, t2 = e2 >> 3;                             // chunk k+2: its S buffer is initialised here
        constexpr int ep = e - 1, hp = ep & 1, cp = (ep >> 1) & 3, tp = ep >> 3;                            // chunk k-1: its PV is issued here (dd > 0)
        constexpr int bq = e & 1, bn = e1 & 1;
        int const k = 16 * p + e;
        stamp(k, 0);
        if constexpr (dd != 0) { warpgroup_wait<0>(); }
        warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(acc_l[0]); warpgroup_fence_operand(acc_l[1]);
        warpgroup_fence_operand(accC[bq]); warpgroup_fence_operand(accC[bn]); warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]);
        if (!kReplay || kReplayOps) {                              // per-warp arrivals, each after this warp's own wgmma wait
            if constexpr (e == 9) { pipe_kv.release(kv0, warp_leader); }                       // V set 0 (tile 2p): PV(16p+7), issued by body e=8, retired at this wait
            if constexpr (e == 17) { pipe_kv.release(kv0 + R, warp_leader); }                  // V set 1 (tile 2p+1): PV(16p+15) retired
            if constexpr (e == 7) { pipe_k.release(kv0, warp_leader); }                        // K set 0: QK(16p+7), issued by body e=6, retired at this wait
            if constexpr (e == 15) { pipe_k.release(kv0 + R, warp_leader); }                   // K set 1: QK(16p+15) retired
        }
        int const j1 = 2 * p + t1, j2 = 2 * p + t2;
        if constexpr ((e1 & 7) == 0) { if (!kNoWait && j1 < n_w && (!kReplay || kReplayOps || j1 < 2)) { pipe_kv.wait_full(kv0 + (t1 & 1) * R, kReplayOps ? 0u : ((t1 == 2) ? (pp ^ 1u) : pp)); } }
        stamp(k, 1);
        uint32_t const dep1 = dep_of(accC[bn]);
        issue_group(cute::bool_constant<dd != 0>{}, accC[bn], tK, Int<h1>{}, Int<c1>{}, Int<(t1 & 1) * R>{}, PCb[hp], tV, Int<hp>{}, Int<cp>{}, Int<(tp & 1) * R>{});
        if constexpr (kRelAlways) { bias_release_by(e1 & 7, warp_leader && j1 < n_w, dep1); } else { if (j1 < n_w) { bias_release(e1 & 7, dep1); } }   // chunk k+1's half slot (exact: no arrival past the last tile): its init loads feed the HGMMA just issued
        warpgroup_fence_operand(accC[bq]);                                                                  // E(k) stays below the issue group
        stamp(k, 2);
        ring_wait();
        exp_chunk(accC[bq], hh, 4 * (2 * p + t) + c, irregular, cute::bool_constant<kSafe>{});           // lockstep schedule: an irregular row applies its words on every tile
        ring_pass();
        stamp(k, 3);
        pack_chunk(accC[bq], PCb[bq]);                                                                      // P(k): consumed by the next body's group
        stamp(k, 4);
        if constexpr (h2 == 0) { if (!kNoWait && j2 < n_w && (!kReplay || kReplayOps || j2 < 1)) { pipe_b.wait_full(c2, kReplayOps ? 0u : uint32_t(j2 & 1)); } }
        init_chunk(accC[bq], Int<c2>{}, Int<h2>{});                                                         // S(k+2) <- bias; QK(k+2) in the next body's group
        stamp(k, 5);
    };
    // drained state after the last processed chunk kk (always an h = 1 chunk, P in PCb[1]): its PV issued and every wgmma retired
    auto drainW = [&](auto cc, auto stc) __attribute__((always_inline)) {
        constexpr int cl = decltype(cc)::value, stl = decltype(stc)::value;
        warpgroup_fence_operand(PCb[1]);
        warpgroup_arrive();
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One; tiled_mma_l.accumulate_ = GMMA::ScaleOut::One;
        #pragma unroll
        for (int kb = 0; kb < kNKB; ++kb) { cute::gemm(tiled_mma_pv, PCb[1](_, _, kb), tV0(_, _, kb, cl, stl), acc_o[1]); }
        #pragma unroll
        for (int kb = 0; kb < kNKB; ++kb) { cute::gemm(tiled_mma_l, PCb[1](_, _, kb), tOnes(_, _, kb), acc_l[1]); }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(acc_l[0]); warpgroup_fence_operand(acc_l[1]);
        warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]); warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]);
        l_check(accC[0], -1);                                    // no exp'd-unpacked chunk exists at a drain in this schedule
        if constexpr (kSyncPeriod) { cutlass::arch::NamedBarrier::sync(T::kNumMmaThreads, kRingBar0 + 3); }
    };
    // ---- prologue = chunks 0, 1 (column 0 of tile 0, both halves, with the seed) + PV(0), PV(1), QK(2) issued + S(3) initialised, ending DRAINED
    shared.barrier_q.wait(0); asm volatile("" ::: "memory");
    load_q_regs();
    if constexpr (kQinRegs) { warpgroup_fence_operand(tQr0); warpgroup_fence_operand(tQr1); }
    pipe_kv.wait_full(kv0, 0);
    pipe_b.wait_full(0, 0);                                     // slot 0 (both halves)
    init_chunk(accC[0], _0{}, _0{}); init_chunk(accC[1], _0{}, _1{});
    uint32_t const dep0 = dep_of(accC[0]), dep1p = dep_of(accC[1]);
    issue_qk(accC[0], tK0, _0{}, _0{}, _0{}); issue_qk(accC[1], tK0, _1{}, _0{}, _0{});
    bias_release(0, dep0); bias_release(1, dep1p);
    warpgroup_wait<0>();
    warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]);
    stamp(0, 2);
    ring_wait(); exp_chunk(accC[0], 0, 0, irregular, cute::true_type{}); ring_pass();
    pack_chunk(accC[0], PCb[0]);
    issue_pv(PCb[0], tV0, _0{}, _0{}, _0{});
    pipe_b.wait_full(1, 0);                                     // slot 1 (column 1 of tile 0)
    init_chunk(accC[0], _1{}, _0{});                             // S(2)
    ring_wait(); exp_chunk(accC[1], 1, 0, irregular, cute::true_type{}); ring_pass();
    pack_chunk(accC[1], PCb[1]);
    issue_pv(PCb[1], tV0, _1{}, _0{}, _0{});
    uint32_t const dep2p = dep_of(accC[0]);
    issue_qk(accC[0], tK0, _0{}, _1{}, _0{});                    // QK(2)
    bias_release(2, dep2p);
    init_chunk(accC[1], _1{}, _1{});                             // S(3)
    stamp(0, 3);
    warpgroup_wait<0>();
    if constexpr (!kSafe) { bad |= need_seed && (seeded != 0xfu); }   // a row with no finite logit in key column 0 of the live window: exact recompute by the SAFE pass
    cutlass::arch::NamedBarrier::sync(T::kNumMmaThreads, kRingBar0 + 3);   // the 3 warpgroups start the stream aligned
    if constexpr (kStagger > 0) { long long const t0_ = clock64(); while (clock64() - t0_ < (long long)cwg * kStagger) { } }
    warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]); warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]);
    warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(acc_l[0]); warpgroup_fence_operand(acc_l[1]);

    // ---- the loop: period p = chunks 16p+2 .. 16p+17; exits drained after the last chunk K-1 = 8 n_w - 1 = body 5 (n_w odd) or 13 (even)
    int p = 0; uint32_t pp = 0;
    #pragma unroll 1
    for (;; ++p, pp ^= 1u) {
        int const zo = params.zero * p;
        OpK const tK = mk_K(zo); OpV const tV = mk_V(zo);
        bodyW(Int<0>{}, p, pp, tK, tV);  bodyW(Int<1>{}, p, pp, tK, tV);  bodyW(Int<2>{}, p, pp, tK, tV);
        bodyW(Int<3>{}, p, pp, tK, tV);  bodyW(Int<4>{}, p, pp, tK, tV);  bodyW(Int<5>{}, p, pp, tK, tV);
        if (2 * p + 1 == n_w) { drainW(_3{}, _0{}); break; }
        bodyW(Int<6>{}, p, pp, tK, tV);  bodyW(Int<7>{}, p, pp, tK, tV);  bodyW(Int<8>{}, p, pp, tK, tV);  bodyW(Int<9>{}, p, pp, tK, tV);
        bodyW(Int<10>{}, p, pp, tK, tV); bodyW(Int<11>{}, p, pp, tK, tV); bodyW(Int<12>{}, p, pp, tK, tV); bodyW(Int<13>{}, p, pp, tK, tV);
        if (2 * p + 2 == n_w) { drainW(_3{}, Int<R>{}); break; }
        bodyW(Int<14>{}, p, pp, tK, tV); bodyW(Int<15>{}, p, pp, tK, tV);
        drainW(_0{}, _0{});
    }
    if (!kReplay) { pipe_kv.release(kv0 + ((n_w - 1) & 1) * R, warp_leader); }   // V stage of the last tile (its K stage was released in-loop)
    } else {
    // ---- prologue = chunks 0, 1 (tile 0, chunk column 0, both halves) with the seed, ending DRAINED in the period-end state:
    //      S(2), S(3) landed in accC[2], accC[3]; chunk 1 exp'd (unpacked) in accC[1]; PV(0) retired.
    OpK const tK0 = mk_K(0); OpV const tV0 = mk_V(0);
    shared.barrier_q.wait(0); asm volatile("" ::: "memory");
    load_q_regs();
    if constexpr (kQinRegs) { warpgroup_fence_operand(tQr0); warpgroup_fence_operand(tQr1); }
    // a tile of the CTA stream this warpgroup does not consume: observe each bias slot's fill (an empty-barrier arrival may never precede the
    // completion of the fill it answers to, or the producer's next expect_tx would land in an unfinished phase; observing full(j) also bounds
    // the follower to one use ahead, so its arrivals count into the phase they are meant for), then release both its halves
    auto follow_tile = [&](uint32_t ph) __attribute__((always_inline)) {   // ph = the tile's index in the CTA stream & 1 (= its bias use parity)
        #pragma unroll 1
        for (int c = 0; c < T::kSlotsB; ++c) {
            pipe_b.wait_full(c, ph); asm volatile("" ::: "memory");
            bias_release(2 * c, 0u); bias_release(2 * c + 1, 0u);
        }
    };
    bool const lead_follow = (jb_[0] | jb_[1] | jb_[2]) != 0;        // some warpgroup's range starts later than the CTA stream (uniform over the CTA)
    if (lead_follow) {                                           // then the warpgroups align BEFORE the stream (a follower needs the others' progress inside it); else after the prologue
        cutlass::arch::NamedBarrier::sync(T::kNumMmaThreads, kRingBar0 + 3);
        #pragma unroll 1
        for (int j = 0; j < jb_w; ++j) { follow_tile(uint32_t(j & 1)); }   // leading tiles of the CTA stream outside this row's range
    }
    pipe_kv.wait_full(kv0, 0);
    pipe_b.wait_full(0, 0);                                     // slot 0 (both halves)
    init_chunk(accC[0], _0{}, _0{}); init_chunk(accC[1], _0{}, _1{});
    uint32_t const dep0 = dep_of(accC[0]), dep1 = dep_of(accC[1]);
    issue_qk(accC[0], tK0, _0{}, _0{}, _0{}); issue_qk(accC[1], tK0, _1{}, _0{}, _0{});
    bias_release(0, dep0); bias_release(1, dep1);
    warpgroup_wait<0>();
    warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]);
    pipe_b.wait_full(1, 0);                                     // slot 1
    init_chunk(accC[2], _1{}, _0{}); init_chunk(accC[3], _1{}, _1{});
    uint32_t const dep2p = dep_of(accC[2]), dep3p = dep_of(accC[3]);
    issue_qk(accC[2], tK0, _0{}, _1{}, _0{});
    if constexpr (!kQK1) { issue_qk(accC[3], tK0, _1{}, _1{}, _0{}); bias_release(2, dep2p); bias_release(3, dep3p); }
    else { bias_release(2, dep2p); (void)dep3p; }                // kQK1: QK(3) (and half slot 3's release) belong to body 0
    stamp(0, 2);
    ring_wait(); exp_chunk(accC[0], 0, 0, irregular, cute::true_type{}); ring_pass();   // tile 0 of an irregular row always applies its words (columns of the stream before its first live one)
    pack_chunk(accC[0], PCb[0]);
    issue_pv(PCb[0], tV0, _0{}, _0{}, _0{});
    ring_wait(); exp_chunk(accC[1], 1, 0, irregular, cute::true_type{}); ring_pass();
    pipe_b.wait_full(2, 0);                                     // slot 2
    init_chunk(accC[0], _2{}, _0{});                             // S(4) <- bias (c = 2, h = 0; half slot 4): body 0 issues QK(4)
    kv_tok = false;                                              // body 0's QK(4) is in tile 0 (no wait needed); generic: unknown -> wait path
    stamp(0, 3);
    warpgroup_wait<0>();                                        // (half slot 4 is released by body 0)
    if constexpr (!kSafe) { bad |= need_seed && (seeded != 0xfu); }   // a row with no finite logit in this warpgroup's first key column (leading masked run): exact recompute by the SAFE pass
    if (!lead_follow) { cutlass::arch::NamedBarrier::sync(T::kNumMmaThreads, kRingBar0 + 3); }   // the 3 warpgroups start the stream aligned (row 0's K/V lands first; the shared bias ring couples them, skew eats its slack)
    if constexpr (kStagger > 0) { long long const t0_ = clock64(); while (clock64() - t0_ < (long long)cwg * kStagger) { } }
    warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]); warpgroup_fence_operand(accC[2]); warpgroup_fence_operand(accC[3]);
    warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]); warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(acc_l[0]); warpgroup_fence_operand(acc_l[1]);

    // ---- the loop: period p = chunks 16p+2 .. 16p+17 (tiles 2p, 2p+1 and the first two chunks of 2p+2 of this warpgroup's range); exits
    //      drained after the last chunk K-1 = 8 n_w - 1, which is body 5 (n_w odd) or body 13 (even) of the last period. One period
    //      routine in two instantiations (STEADY / GENERAL, see body); per-period state (the exact-release predicates, GENERAL's word tiles) is hoisted to its head.
    auto period = [&](auto genc, int p, uint32_t pp) __attribute__((always_inline)) -> bool {   // -> the stream ended in this period
        constexpr bool kGen = decltype(genc)::value;
        int const zo = params.zero * p;
        OpK const tK = mk_K(zo); OpV const tV = mk_V(zo);
        lead1 = warp_leader && (2 * p + 1 < n_w); lead2 = warp_leader && (2 * p + 2 < n_w);            // exact releases: tiles 2p+1, 2p+2 inside the range
        if constexpr (kGen && !kSafe) { wt = word_tiles(2 * p); }
        body(Int<0>{}, genc, p, pp, tK, tV);  body(Int<1>{}, genc, p, pp, tK, tV);  body(Int<2>{}, genc, p, pp, tK, tV);
        body(Int<3>{}, genc, p, pp, tK, tV);  body(Int<4>{}, genc, p, pp, tK, tV);  body(Int<5>{}, genc, p, pp, tK, tV);
        if (2 * p + 1 == n_w) { drain(_3{}); return true; }
        body(Int<6>{}, genc, p, pp, tK, tV);  body(Int<7>{}, genc, p, pp, tK, tV);  body(Int<8>{}, genc, p, pp, tK, tV);  body(Int<9>{}, genc, p, pp, tK, tV);
        body(Int<10>{}, genc, p, pp, tK, tV); body(Int<11>{}, genc, p, pp, tK, tV); body(Int<12>{}, genc, p, pp, tK, tV); body(Int<13>{}, genc, p, pp, tK, tV);
        if (2 * p + 2 == n_w) { drain(_3{}); return true; }
        body(Int<14>{}, genc, p, pp, tK, tV); body(Int<15>{}, genc, p, pp, tK, tV);
        drain(_1{});
        return false;
    };
    constexpr bool kHasSteady = !kSafe, kHasGeneral = kMaskInKernel;   // the SAFE pass runs GENERAL periods only; the lean flag-4096 hot pass has no irregular rows (fix list) and STEADY periods only
    int p = 0; uint32_t pp = 0;                                  // pp = p & 1 = K/V use parity of tile 2p's stages
    #pragma unroll 1
    for (;; ++p, pp ^= 1u) {
        bool ended;
        if constexpr (kHasSteady && kHasGeneral) {
            if (irregular && (p == 0 || p >= g_from)) { ended = period(cute::true_type{}, p, pp); } else { ended = period(cute::false_type{}, p, pp); }
        } else if constexpr (kHasGeneral) { ended = period(cute::true_type{}, p, pp); }
        else { ended = period(cute::false_type{}, p, pp); }
        if (ended) { break; }
    }
    {   // last chunk K-1 (h = 1, c = 3, tile n_w-1, exp'd in accC[3]): pack + PV, then release that tile's K/V stage
        auto finish = [&](auto setc) __attribute__((always_inline)) {
            constexpr int set = decltype(setc)::value;
            pack_chunk(accC[3], PCb[1]);
            issue_pv(PCb[1], tV0, _1{}, _3{}, Int<set * R>{});
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(acc_l[0]); warpgroup_fence_operand(acc_l[1]);
            if (!kReplay) { pipe_kv.release(kv0 + set * R, warp_leader); }
        };
        if ((n_w - 1) & 1) { finish(_1{}); } else { finish(_0{}); }
    }
    #pragma unroll 1
    for (int t = 0; t < n_trail; ++t) { follow_tile(uint32_t((n_w + t) & 1)); }   // the shared bias ring streams on for the longer rows: keep this warpgroup's arrivals coming (tile jb_w + n_w + t, jb_w even)
    }   // !kW

    // ---- epilogue: O / l -> bf16 -> global; rows that fail validation put the CTA tile on the fix list -------------------
    int const i = i0 + cwg;
    if (i < params.N) {
        Element* obase = params.out + (int64_t)b * params.so_b + (int64_t)i * params.so_n + (int64_t)h * params.so_h;
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            Tensor o_rc = make_tensor(acc_o[hh].data(), flash::convert_layout_acc_rowcol(acc_o[hh].layout()));
            Tensor l_rc = make_tensor(acc_l[hh].data(), flash::convert_layout_acc_rowcol(acc_l[hh].layout()));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                float const l = l_rc(mi, 0);
                if constexpr (!kSafe && (T::kFlags & 2048) == 0) {                          // validation: l finite and > 0, O finite; else the CTA tile goes on the fix list
                    float chk = 0.f;
                    #pragma unroll
                    for (int ni = 0; ni < kNColsO; ++ni) { chk += o_rc(mi, ni); }
                    chk = l + chk * 0.f;                                                     // NaN if any O is inf/NaN
                    int const qv = qtile * kBlockM + 64 * hh + get<0>(tOcO_rc(mi, _0{}));
                    if (qv < S && !uniform && !(chk > 0.f && chk < INFINITY)) { bad = true; }
                }
                float const inv = l > 0.f ? 1.f / l : 0.f;
                int const q = qtile * kBlockM + 64 * hh + get<0>(tOcO_rc(mi, _0{}));
                if (q < S) {
                    Element* orow = obase + (int64_t)q * params.so_s;
                    #pragma unroll
                    for (int ni = 0; ni < kNColsO; ni += 2) {
                        int const d = get<1>(tOcO_rc(mi, ni));
                        __nv_bfloat162 v2 = __floats2bfloat162_rn(o_rc(mi, ni) * inv, o_rc(mi, ni + 1) * inv);
                        *reinterpret_cast<__nv_bfloat162*>(orow + d) = v2;
                    }
                }
            }
        }
    }
    if constexpr (!kSafe && (T::kFlags & 2048) == 0) {
        if (__any_sync(0xffffffffu, bad)) { if (lane == 0) { atomicOr(&shared.bad, 1); } }
        cutlass::arch::NamedBarrier::sync(T::kNumMmaThreads, kRingBar0 + 3);
        if (thread_idx == 0 && shared.bad) {
            int const slot = atomicAdd(params.fix, 1);
            params.fix[1 + 3 * slot] = qtile; params.fix[2 + 3 * slot] = rg; params.fix[3 + 3 * slot] = bh;
        }
    }
    if constexpr (!kList) { break; }
    }   // consumer tile loop
    }   // consumers
    if constexpr (CR > 1) { cute::cluster_arrive(); cute::cluster_wait(); }
}

}  // namespace triattn_m1
