// Triangle attention forward (inference), sm_90a, the "R=2 x BN=128 x cluster" corner:
//   one CTA = (batch b, head h, one q-tile of 128 queries, R=2 consecutive pair rows), 384 threads = 1 TMA producer warp (+3 warps
//   that build the mask tables, then exit) and 2 consumer warpgroups (q rows 0-63 / 64-127 of the tile);
//   thread-block cluster (CQ, CR): the CQ CTAs along x own adjacent q-tiles of the SAME rows and receive each K/V tile by TMA
//   multicast (each issues a 1/CQ slice); the CR CTAs along y own the same q-tile of DIFFERENT rows and receive each [128 x 128]
//   bf16 pair-bias tile by multicast (1/CR slice each) -- the GEMM A-along-N / B-along-M multicast pattern;
//   per k-tile of 128 keys: S = Q K^T as one m64n128k32 wgmma chain per warpgroup whose fp32 accumulator is PRE-LOADED with
//   bias/scale (so no per-logit bias FFMA and no bias registers live during the softmax), softmax with a lazily updated running
//   max, P -> bf16, O += P [V | 1] (m64n40k128, the 8 extra columns accumulate the row sum on the tensor core);
//   software pipeline over the two rows: row b's QK GEMM is in flight during row a's softmax and row a's next QK during row b's.
//
//   out[b,i,h,q,:] = softmax_k( scale * q[b,i,h,q,:].k[b,i,h,k,:] + bias[b,h,q,k]  (masked keys excluded) ) @ v[b,i,h,k,:]
//
// Grid x = q-tiles of this launch (cluster x), y = row groups padded to CR (cluster y), z = b*H + h.
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

#include "fa3_utils.h"

namespace triattn_b {

using namespace cute;

enum TileClass : uint8_t { kClean = 0, kMixed = 1, kSkip = 2 };

// Ring of smem stages filled by TMA (possibly multicast from a peer CTA) and drained by the consumer warpgroups of every CTA that
// receives the stage. full[s]: transaction barrier local to each CTA (1 producer arrival + the whole stage's bytes, whichever CTA's
// TMA delivers them); empty[s]: `empty_arrivals` arrivals per use = every releasing agent of every CTA of the multicast group
// (peers arrive remotely), so no producer of the group overwrites a stage some peer still reads. Use u of a stage has phase u & 1.
template <int Stages>
struct Pipe {
    struct SharedStorage {
        cutlass::arch::ClusterTransactionBarrier full[Stages];
        cutlass::arch::ClusterBarrier empty[Stages];
    };
    SharedStorage& st;
    CUTLASS_DEVICE Pipe(SharedStorage& s) : st(s) {}
    CUTLASS_DEVICE static void init(SharedStorage& s, int empty_arrivals) {
        for (int i = 0; i < Stages; ++i) { s.full[i].init(1); s.empty[i].init(empty_arrivals); }
    }
    CUTLASS_DEVICE void producer_wait_empty(int stage, uint32_t phase) { st.empty[stage].wait(phase ^ 1); }
    CUTLASS_DEVICE void producer_expect(int stage, uint32_t bytes) { st.full[stage].arrive_and_expect_tx(bytes); }
    CUTLASS_DEVICE void producer_skip(int stage) { st.full[stage].arrive(); }
    CUTLASS_DEVICE uint64_t* full_barrier(int stage) { return reinterpret_cast<uint64_t*>(&st.full[stage]); }
    CUTLASS_DEVICE void wait_full(int stage, uint32_t phase) { st.full[stage].wait(phase); }
    template <int NP>
    CUTLASS_DEVICE void release(int stage, bool elected, uint32_t const (&ranks)[NP], uint32_t self_rank) {
        if (elected) {
            #pragma unroll
            for (int p = 0; p < NP; ++p) {
                if (ranks[p] == self_rank) { st.empty[stage].arrive(); } else { st.empty[stage].arrive(ranks[p], 1u); }
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------
template <int CQ_, int CR_, int kRingKV_, int kRingB_, typename Element_, int kFlags_ = 0, int kBlockN_ = 128>
struct Traits {
    using Element = Element_;
    // kFlags bit 0: the two consumer warpgroups take turns on the exponential section (named-barrier ping-pong)
    //        bit 1: rescale the output accumulator on every row step (no lazy-max vote/branch)
    static constexpr int kFlags = kFlags_;
    static constexpr bool kPingPong = (kFlags_ & 1) != 0;
    static constexpr int kHeadDim = 32, kBlockM = 128, kBlockN = kBlockN_, R = 2, CQ = CQ_, CR = CR_;
    static_assert(kBlockN % 32 == 0 && kBlockN >= 64 && kBlockN <= 256);
    static constexpr int kRingKV = kRingKV_, kRingB = kRingB_;
    static constexpr int kStagesKV = kRingKV * R;
    static constexpr int kBlockMwg = 64;
    static constexpr int kNumMmaWG = 2;
    static constexpr int kNumMmaThreads = kNumMmaWG * 128;
    static constexpr int kNumThreads = kNumMmaThreads + 128;
    static constexpr int kMaxS = 8192;                       // capacity of the per-row key-mask bit tables
    static constexpr int kMaxTiles = (kMaxS + kBlockN - 1) / kBlockN;
    static constexpr int kClusterRows = R * CR;              // rows whose masks decide the (cluster-uniform) active k-tile list
    static_assert(kRingKV >= 2 && kRingB >= 2);
    static_assert((CQ == 1 || CQ == 2) && (CR == 1 || CR == 2));

    using ClusterShape = Shape<Int<CQ>, Int<CR>, _1>;
    static constexpr int kHeadDimV = kHeadDim + 8;              // PV runs on [V | 1]: accumulator columns D..D+7 hold the row sum l
    using TileShapeQK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using AtomLayout = Layout<Shape<Int<kNumMmaWG>, _1, _1>>;
    using TiledMmaQK = decltype(make_tiled_mma(GMMA::ss_op_selector<Element, Element, float, TileShapeQK>(), AtomLayout{}));
    using TileShapeQKC = Shape<Int<kBlockM>, Int<kBlockN_ / 2>, Int<kHeadDim>>;
    using TiledMmaQKC = decltype(make_tiled_mma(GMMA::ss_op_selector<Element, Element, float, TileShapeQKC>(), AtomLayout{}));
    static_assert(std::is_same_v<Element, cutlass::bfloat16_t>, "PV atom picked by hand for D=32 bf16");
    using TiledMmaPV = decltype(make_tiled_mma(SM90_64x40x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, AtomLayout{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockM>, Int<kHeadDim>>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<R>{})));          // (M, D, R)
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kStagesKV>{})));   // (BN, D, st)
    // V^T stage = (D+8, BN) MN-major, unswizzled 8x8 core matrices; rows 0..D-1 are the TMA destination (one contiguous D*BN block
    // per stage because BN is tiled first), rows D..D+7 hold 1.0 (written once at kernel start, never touched by TMA)
    using SmemLayoutAtomVt = GMMA::Layout_MN_INTER_Atom<Element>;
    using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomVt{}, make_shape(Int<kHeadDimV>{}, Int<kBlockN>{}, Int<kStagesKV>{}), Step<_2, _1, _3>{}));  // (D+8, BN, st)
    using SmemLayoutVload1 = decltype(tile_to_shape(SmemLayoutAtomVt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), Step<_2, _1>{}));                      // (D, BN)
    static constexpr int kStageElemsV = kHeadDimV * kBlockN;
    static_assert(size(SmemLayoutVload1{}) == kHeadDim * kBlockN && cosize(SmemLayoutVload1{}) == kHeadDim * kBlockN && cosize(SmemLayoutVt{}) == kStageElemsV * kStagesKV);
    // pair-bias stage, two modes:
    //  kBiasMMA (stream loop): the ORIGINAL bf16 bias tile (128 q x BN keys, TMA straight from the caller's tensor, 128B swizzle) is
    //    added to S on the TENSOR CORE: S = Q K^T + (beta I) Bias, i.e. 4 extra k16-blocks of the QK wgmma chain with A = beta * I_64
    //    (smem, bf16, beta = bf16(1/scale)) and B = this warpgroup's 64 bias rows viewed MN-major -> no per-logit bias instruction at all;
    //  otherwise: fp32 = bias / scale in MMA-fragment order (see stage_bias in prep.cu), copied into the S accumulators by LDS.128.
    static constexpr bool kBiasMMA = (kFlags_ & 512) != 0;
    static constexpr int kChunkW = kBlockN / 2;                  // S-chunk width of the streaming loop (2 chunks per k-tile)
    static_assert(kChunkW % 16 == 0);
    static_assert(!kBiasMMA || kChunkW == 64, "bias-MMA B operand uses the 64-wide MN-major 128B-swizzle atom: BN must be 128");
    using BiasElement = std::conditional_t<kBiasMMA, Element, float>;
    static constexpr int kBiasTileElems = kBlockM * kBlockN;
    // fp32 fragment-order mode
    using SmemLayoutBias1 = Layout<Shape<_256, Int<kBlockN / 2>>, Stride<_1, _256>>;                      // one stage as the TMA box sees it
    using ShapeBF  = Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;                                    // (256, BN/2, nk, nq, B*H)
    using StrideBF = Stride<_1, _256, int64_t, int64_t, int64_t>;
    // bf16 tile mode
    using SmemLayoutAtomBh = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockM>, Int<kBlockN>>());
    using SmemLayoutBiasH = decltype(tile_to_shape(SmemLayoutAtomBh{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{})));          // (M, BN) one stage
    using ShapeBH  = Shape<int32_t, int32_t, int32_t, int32_t>;                                             // (Sq, Sk, H, B)
    using StrideBH = Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeB  = std::conditional_t<kBiasMMA, ShapeBH, ShapeBF>;
    using StrideB = std::conditional_t<kBiasMMA, StrideBH, StrideBF>;
    // identity operand and the bias MMA (one warpgroup: M = 64 rows, N = kChunkW keys, K = 64 = this warpgroup's q rows)
    using SmemLayoutI = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{}, make_shape(_64{}, _64{})));              // (m, k)
    using SmemLayoutBc = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{}, make_shape(_64{}, _64{})));            // (n = key, k = q row)
    using TiledMmaBias = decltype(make_tiled_mma(GMMA::ss_op_selector<Element, Element, float, Shape<_64, Int<kChunkW>, _64>, GMMA::Major::K, GMMA::Major::MN>(),
                                                Layout<Shape<_1, _1, _1>>{}));

    using GmemTiledCopyKV = std::conditional_t<(CQ > 1), SM90_TMA_LOAD_MULTICAST, SM90_TMA_LOAD>;
    using GmemTiledCopyB = std::conditional_t<(CR > 1), SM90_TMA_LOAD_MULTICAST, SM90_TMA_LOAD>;
    using StrideQK = Stride<int64_t, _1, int64_t, int64_t, int64_t>;      // (S, D, H, N, B)
    using StrideV  = Stride<_1, int64_t, int64_t, int64_t, int64_t>;      // (D, S, H, N, B)
    using ShapeQK  = Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;

    using TMA_Q = decltype(make_tma_copy(SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQK{}, StrideQK{}),
                                        take<0, 2>(SmemLayoutQ{}), make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), _1{}));
    using TMA_K = decltype(make_tma_copy(GmemTiledCopyKV{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQK{}, StrideQK{}),
                                        take<0, 2>(SmemLayoutK{}), make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), Int<CQ>{}));
    using TMA_V = decltype(make_tma_copy(GmemTiledCopyKV{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQK{}, StrideV{}),
                                        SmemLayoutVload1{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), Int<CQ>{}));
    using TMA_BF = decltype(make_tma_copy(GmemTiledCopyB{}, make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), ShapeBF{}, StrideBF{}),
                                        SmemLayoutBias1{}, make_shape(_256{}, Int<kBlockN / 2>{}), Int<CR>{}));
    using TMA_BH = decltype(make_tma_copy(GmemTiledCopyB{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeBH{}, StrideBH{}),
                                        SmemLayoutBiasH{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{}), Int<CR>{}));
    using TMA_B = std::conditional_t<kBiasMMA, TMA_BH, TMA_BF>;

    static constexpr uint32_t kBytesQ = kBlockM * kHeadDim * sizeof(Element);          // per row r
    static constexpr uint32_t kBytesKV = 2u * kBlockN * kHeadDim * sizeof(Element);    // K + V of one stage (whole tile, all slices)
    static constexpr uint32_t kBytesBias = kBlockM * kBlockN * sizeof(BiasElement);   // whole bias tile (either mode)

    using PipeKV = Pipe<kStagesKV>;
    using PipeB = Pipe<kRingB>;
    static constexpr int kArrivalsKV = kNumMmaWG * CQ;          // one elected thread per consumer warpgroup per CTA of the q group
    static constexpr int kArrivalsB = kNumMmaWG * 4 * CR;       // one per consumer warp per CTA of the row group

    struct SharedStorage {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, 1024> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, 1024> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, 1024> smem_v;
        cute::array_aligned<BiasElement, kBiasTileElems * kRingB, 1024> smem_bias;
        cute::array_aligned<Element, (kBiasMMA ? 64 * 64 : 8), 1024> smem_ident;          // beta * I_64 (bias-MMA A operand)
        uint32_t mask_bits[kClusterRows][kMaxS / 32];           // key k of cluster row rr attendable (in range and not masked)
        uint8_t tile_class[kClusterRows][kMaxTiles];
        uint16_t active[kMaxTiles];                              // k-tiles with work for some row of the cluster, ascending
        int n_active;
        int uniform[kClusterRows];                               // 1 = row has no attendable key: attends uniformly to its S keys
        typename PipeKV::SharedStorage pipe_kv;
        typename PipeB::SharedStorage pipe_b;
        cutlass::arch::ClusterTransactionBarrier barrier_q;
    };

    struct Params {
        TMA_Q tma_q; TMA_K tma_k; TMA_V tma_v; TMA_B tma_b;
        ShapeQK shape_qk; ShapeB shape_b;
        Element* out; int64_t so_b, so_n, so_h, so_s;                 // out [B,N,H,S,D] element strides (d stride 1)
        uint8_t const* mask; int64_t sm_b, sm_n, sm_s;                // mask [B,N,S] (byte, nonzero = keep); nullptr = none
        int S, N, H;
        int n_qtiles, n_ktiles, qtile_base;                            // this launch covers q-tiles [qtile_base, qtile_base + gridDim.x)
        float scale;
        float beta;                                                    // bias-MMA: bf16-representable ~1/scale used on the identity
        unsigned long long* trace;
        int zero;                                                      // always 0; multiplies loop counters into operand bases so per-period descriptors are not loop-invariant                                     // kFlags & 16: clock64 stamps of CTA (1,1,0), [wg][iter<32][16]
    };
};

enum class NamedBarriers { WarpSchedulerWG1 = 1, WarpSchedulerWG2 = 2, ProducerWG = 3 };

CUTLASS_DEVICE int warp_group_idx_nosync() { return threadIdx.x / cutlass::NumThreadsPerWarpGroup; }

CUTLASS_DEVICE void warp_scheduler_barrier_sync() {
    cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup,
        static_cast<uint32_t>(NamedBarriers::WarpSchedulerWG1) - 1 + warp_group_idx_nosync() + static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier));
}
CUTLASS_DEVICE void warp_scheduler_barrier_arrive() {
    int const cur = warp_group_idx_nosync() - 1;   // 0 or 1
    cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup,
        static_cast<uint32_t>(NamedBarriers::WarpSchedulerWG1) + (1 - cur) + static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier));
}

// volatile: keeps the exponentials of a chunk behind the wgmma issue that precedes them in program order (ptxas would otherwise
// sink the HGMMAs into the MUFU sequence and expose their latency at the next wait)
__device__ __forceinline__ float ex2_approx(float x) { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

// ---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void __launch_bounds__(T::kNumThreads, 1) triattn_fwd_kernel(CUTE_GRID_CONSTANT typename T::Params const params) {
    using Element = typename T::Element;
    constexpr int R = T::R, kBlockM = T::kBlockM, kBlockN = T::kBlockN, kHeadDim = T::kHeadDim, CQ = T::CQ, CR = T::CR;
    using SharedStorage = typename T::SharedStorage;
    extern __shared__ char smem_buf[];
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const lane_predicate = cute::elect_one_sync();
    int const wg_idx = cutlass::canonical_warp_group_idx();
    int const tid = threadIdx.x;
    int const lane = tid % 32;

    // ---- cluster / block coordinates ---------------------------------------------------------------------------------
    uint32_t const self_rank = (CQ * CR > 1) ? cute::block_rank_in_cluster() : 0u;
    int const cx = (CQ > 1) ? int(self_rank % CQ) : 0;          // rank = x + y * CQ
    int const cy = (CR > 1) ? int(self_rank / CQ) : 0;
    uint32_t kv_ranks[CQ], b_ranks[CR];
    uint16_t kv_mask = 0, b_mask = 0;
    #pragma unroll
    for (int x = 0; x < CQ; ++x) { kv_ranks[x] = uint32_t(x + cy * CQ); kv_mask |= uint16_t(1u << kv_ranks[x]); }
    #pragma unroll
    for (int y = 0; y < CR; ++y) { b_ranks[y] = uint32_t(cx + y * CQ); b_mask |= uint16_t(1u << b_ranks[y]); }

    int const qtile = params.qtile_base + int(blockIdx.x);
    int const rgroup = blockIdx.y;                   // may address rows >= N when padded to CR: rows are clamped, nothing stored
    int const bh = blockIdx.z;
    int const b = bh / params.H, h = bh % params.H;
    int const i0 = rgroup * R;                       // first pair row of this CTA
    int const ic0 = (rgroup - cy) * R;               // first pair row of the cluster
    int const S = params.S;
    int const n_ktiles = params.n_ktiles;

    if (warp_idx == 0 && lane_predicate) {
        cute::prefetch_tma_descriptor(params.tma_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_v.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_b.get_tma_descriptor());
        shared.barrier_q.init(1);
        T::PipeKV::init(shared.pipe_kv, T::kArrivalsKV);
        T::PipeB::init(shared.pipe_b, T::kArrivalsB);
        cutlass::arch::fence_barrier_init();
    }
    typename T::PipeKV pipe_kv(shared.pipe_kv);
    typename T::PipeB pipe_b(shared.pipe_b);

    // ---- mask bit rows + k-tile classes for the cluster's rows (producer warpgroup, 128 threads) + ones block ---------
    constexpr uint32_t kBarMask = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier) + static_cast<uint32_t>(NamedBarriers::ProducerWG);
    if (wg_idx == 0) {
        int const w4 = tid / 32;      // 4 warps
        int const n_words = n_ktiles * (kBlockN / 32);
        if (w4 == 0) { for (int rr = lane; rr < T::kClusterRows; rr += 32) { shared.uniform[rr] = 0; } }
        cutlass::arch::NamedBarrier::sync(128, kBarMask);
        for (int rr = 0; rr < T::kClusterRows; ++rr) {
            int const i = min(ic0 + rr, params.N - 1);
            uint8_t const* mrow = params.mask ? params.mask + (int64_t)b * params.sm_b + (int64_t)i * params.sm_n : nullptr;
            int cnt = 0;
            for (int w = w4; w < n_words; w += 4) {
                int const k = w * 32 + lane;
                bool keep = k < S;
                if (keep && mrow) { keep = mrow[(int64_t)k * params.sm_s] != 0; }
                uint32_t const word = __ballot_sync(0xffffffffu, keep);
                if (lane == 0) { shared.mask_bits[rr][w] = word; }
                cnt += __popc(word);
            }
            if (lane == 0) { atomicAdd(&shared.uniform[rr], cnt); }     // attendable-key count of row rr
        }
        cutlass::arch::NamedBarrier::sync(128, kBarMask);
        if (w4 == 0) { for (int rr = lane; rr < T::kClusterRows; rr += 32) { shared.uniform[rr] = (params.mask != nullptr && shared.uniform[rr] == 0) ? 1 : 0; } }
        cutlass::arch::NamedBarrier::sync(128, kBarMask);
        constexpr int kWordsPerTile = kBlockN / 32;
        for (int rr = 0; rr < T::kClusterRows; ++rr) {
            for (int j = tid; j < n_ktiles; j += 128) {
                bool all1 = true, all0 = true;
                #pragma unroll
                for (int w = 0; w < kWordsPerTile; ++w) {
                    uint32_t const word = shared.mask_bits[rr][j * kWordsPerTile + w];
                    all1 &= (word == 0xffffffffu); all0 &= (word == 0u);
                }
                // a row with no attendable key attends uniformly to all its S keys: none of its in-range tiles may be skipped
                shared.tile_class[rr][j] = all1 ? kClean : ((all0 && !shared.uniform[rr]) ? kSkip : kMixed);
            }
        }
        cutlass::arch::NamedBarrier::sync(128, kBarMask);
        if (tid == 0) {
            int n = 0;
            for (int j = 0; j < n_ktiles; ++j) {
                bool any_work = false;
                for (int rr = 0; rr < T::kClusterRows; ++rr) { any_work |= (shared.tile_class[rr][j] != kSkip); }
                if (any_work) { shared.active[n++] = uint16_t(j); }
            }
            if (n == 0) { shared.active[n++] = 0; }              // keep the schedule non-empty (all-skip cannot happen: see uniform)
            shared.n_active = n;
        }
    } else {
        // rows D..D+7 of every V^T stage = 1.0: the PV GEMM then yields the softmax row sums in accumulator columns D..D+7
        Tensor sVt = make_tensor(make_smem_ptr(shared.smem_v.data()), typename T::SmemLayoutVt{});
        constexpr int kOnes = 8 * kBlockN * T::kStagesKV;
        for (int idx = tid - 128; idx < kOnes; idx += T::kNumMmaThreads) {
            int const st = idx / (8 * kBlockN), rem = idx % (8 * kBlockN);
            sVt(kHeadDim + rem / kBlockN, rem % kBlockN, st) = Element(1.f);
        }
        if constexpr (T::kBiasMMA) {   // beta * I_64: A operand of the bias part of the QK wgmma chain
            Tensor sI = make_tensor(make_smem_ptr(shared.smem_ident.data()), typename T::SmemLayoutI{});
            for (int idx = tid - 128; idx < 64 * 64; idx += T::kNumMmaThreads) { sI(idx / 64, idx % 64) = Element(idx / 64 == idx % 64 ? params.beta : 0.f); }
        }
        cutlass::arch::fence_view_async_shared();                  // generic-proxy smem writes -> visible to wgmma (async proxy)
    }
    // barrier inits + tables visible cluster-wide before any TMA / remote arrival
    if constexpr (CQ * CR > 1) { cute::cluster_arrive_relaxed(); cute::cluster_wait(); } else { __syncthreads(); }

    int const n_active = int(__reduce_max_sync(0xffffffffu, unsigned(shared.n_active)));   // uniform register (loop bounds, tile refs)

    if (wg_idx == 0) {
        // =============================================== PRODUCER =====================================================
        cutlass::arch::warpgroup_reg_dealloc<24>();                 // 128*24 + 256*240 = 384*168 = the launch allocation: the inc must fit what the dec frees
        int const warp_idx_in_wg = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_wg == 0 && lane_predicate) {
            Tensor sQ = make_tensor(make_smem_ptr(shared.smem_q.data()), typename T::SmemLayoutQ{});
            Tensor sK = make_tensor(make_smem_ptr(shared.smem_k.data()), typename T::SmemLayoutK{});
            auto sV_stage = [&](int st) { return make_tensor(make_smem_ptr(shared.smem_v.data() + st * T::kStageElemsV), typename T::SmemLayoutVload1{}); };
            auto sB_stage = [&](int st) {
                if constexpr (T::kBiasMMA) { return make_tensor(make_smem_ptr(shared.smem_bias.data() + st * T::kBiasTileElems), typename T::SmemLayoutBiasH{}); }
                else                       { return make_tensor(make_smem_ptr(shared.smem_bias.data() + st * T::kBiasTileElems), typename T::SmemLayoutBias1{}); }
            };

            Tensor mQ = params.tma_q.get_tma_tensor(params.shape_qk)(_, _, h, _, b);                 // (S, D, N)
            Tensor mK = params.tma_k.get_tma_tensor(params.shape_qk)(_, _, h, _, b);                 // (S, D, N)
            auto shape_v = make_shape(get<1>(params.shape_qk), get<0>(params.shape_qk), get<2>(params.shape_qk), get<3>(params.shape_qk), get<4>(params.shape_qk));
            Tensor mVt = params.tma_v.get_tma_tensor(shape_v)(_, _, h, _, b);                        // (D, S, N)
            auto gB = [&]() {
                if constexpr (T::kBiasMMA) {   // (M, BN, ktile) tiles of the caller's bf16 [Sq, Sk] bias of head (b, h)
                    return local_tile(params.tma_b.get_tma_tensor(params.shape_b)(_, _, h, b), make_shape(Int<kBlockM>{}, Int<kBlockN>{}), make_coord(qtile, _));
                } else {                       // (256, BN/2, ktile): fragment-order fp32 tiles
                    return params.tma_b.get_tma_tensor(params.shape_b)(_, _, _, qtile, bh);
                }
            }();

            Tensor gQ = local_tile(mQ, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(qtile, _0{}, _));      // (M, D, N)
            Tensor gK = local_tile(mK, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}, _));          // (BN, D, ktile, N)
            Tensor gVt = local_tile(mVt, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_0{}, _, _));        // (D, BN, ktile, N)

            auto block_tma_q = params.tma_q.get_slice(_0{});
            Tensor tQgQ = group_modes<0, 3>(block_tma_q.partition_S(gQ));      // (TMA, N)
            Tensor tQsQ = group_modes<0, 3>(block_tma_q.partition_D(sQ));      // (TMA, R)
            auto block_tma_k = params.tma_k.get_slice(cx);
            Tensor tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));      // (TMA, ktile, N)
            Tensor tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));      // (TMA, st)
            auto block_tma_v = params.tma_v.get_slice(cx);
            Tensor tVgV = group_modes<0, 3>(block_tma_v.partition_S(gVt));     // (TMA, ktile, N)
            auto tVsV = [&](int st) { return group_modes<0, 3>(block_tma_v.partition_D(sV_stage(st))); };   // (TMA)
            auto block_tma_b = params.tma_b.get_slice(cy);
            Tensor tBgB = group_modes<0, 3>(block_tma_b.partition_S(gB));      // (TMA, ktile)
            auto tBsB = [&](int st) { return group_modes<0, 3>(block_tma_b.partition_D(sB_stage(st))); };   // (TMA)

            shared.barrier_q.arrive_and_expect_tx(T::kBytesQ * R);
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                int const i = min(i0 + r, params.N - 1);
                copy(params.tma_q.with(reinterpret_cast<uint64_t&>(shared.barrier_q), 0), tQgQ(_, i), tQsQ(_, r));
            }
            // Active tile jj: bias(j) -> bias stage jj % kRingB (always loaded: the row peers of the cluster need it even when this
            // CTA's rows have nothing to attend in tile j); K,V(j, r) -> KV stage (jj % kRingKV) * R + r.
            for (int jj = 0; jj < n_active; ++jj) {
                int const j = shared.active[jj];
                int const bslot = jj % T::kRingB; uint32_t const bph = (jj / T::kRingB) & 1;
                // kFlags & 32768: ABLATION (timing only) -- after each ring is filled once, later stages are declared full without
                // any TMA traffic (consumers recompute on stale tiles): separates the compute structure from L2/TMA supply.
                constexpr bool kNoRefill = (T::kFlags & 32768) != 0;
                if ((T::kFlags & 65536) != 0 && jj >= T::kRingKV) { break; }          // replay ablation: first ring fill only (bias ring < K/V ring)
                if ((T::kFlags & 65536) != 0 && jj >= T::kRingB) { /* replay: consumers never wait for bias slots after the first fill */ }
                else {
                  pipe_b.producer_wait_empty(bslot, bph);
                  if (kNoRefill && jj >= T::kRingB) { pipe_b.producer_skip(bslot); }
                  else {
                    pipe_b.producer_expect(bslot, T::kBytesBias);
                    copy(params.tma_b.with(*pipe_b.full_barrier(bslot), b_mask), tBgB(_, j), tBsB(bslot));
                  }
                }
                int const kslot = jj % T::kRingKV; uint32_t const kph = (jj / T::kRingKV) & 1;
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    int const i = min(i0 + r, params.N - 1);
                    int const st = R * kslot + r;
                    pipe_kv.producer_wait_empty(st, kph);
                    if (kNoRefill && jj >= T::kRingKV) { pipe_kv.producer_skip(st); continue; }
                    pipe_kv.producer_expect(st, T::kBytesKV);
                    copy(params.tma_k.with(*pipe_kv.full_barrier(st), kv_mask), tKgK(_, j, i), tKsK(_, st));
                    copy(params.tma_v.with(*pipe_kv.full_barrier(st), kv_mask), tVgV(_, j, i), tVsV(st));
                }
            }
        }
        // stay resident until the whole cluster is done (peers arrive remotely on this CTA's barriers)
        if constexpr (CQ * CR > 1) { cute::cluster_arrive_relaxed(); cute::cluster_wait(); }
        return;
    }

    // ================================================= CONSUMERS ======================================================
    cutlass::arch::warpgroup_reg_alloc<240>();
    int const thread_idx = tid - 128;                       // 0..255
    // consumer warpgroup 0 / 1 -> q rows [cwg*64, cwg*64+64). REDUX puts the (warp-uniform) value in a uniform register, so the smem
    // operand descriptors derived from it stay on the uniform datapath (no per-HGMMA R2UR chains).
    int const cwg = int(__reduce_max_sync(0xffffffffu, unsigned(thread_idx) / 128u));
    bool const wg_leader = (thread_idx % 128) == 0;
    constexpr bool kTrace = (T::kFlags & 16) != 0;
    bool const tracer = kTrace && wg_leader && params.trace != nullptr && blockIdx.x == 1 && blockIdx.y == 1 && blockIdx.z == 0;
    auto stamp = [&](int it, int k) __attribute__((always_inline)) {
        if constexpr (kTrace) { if (tracer && it < 32) { params.trace[(cwg * 32 + it) * 16 + k] = clock64(); } }
    };

    typename T::TiledMmaQK tiled_mma_qk;
    typename T::TiledMmaPV tiled_mma_pv;
    auto wg_layout = make_layout(make_shape(Int<T::kNumMmaWG>{}), make_stride(Int<128>{}));
    auto wg_mma_qk = tiled_mma_qk.get_slice(wg_layout(cwg));
    auto wg_mma_pv = tiled_mma_pv.get_slice(wg_layout(cwg));
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

    Tensor sQ = make_tensor(make_smem_ptr(shared.smem_q.data()), typename T::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared.smem_k.data()), typename T::SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(shared.smem_v.data()), typename T::SmemLayoutVt{});
    Tensor tSrQ = wg_mma_qk.partition_fragment_A(sQ);        // (frag, MMA_M, MMA_K, R)
    Tensor tSrK = wg_mma_qk.partition_fragment_B(sK);        // (frag, MMA_N, MMA_K, st)
    Tensor tOrV = wg_mma_pv.partition_fragment_B(sVt);       // (frag, MMA_N, MMA_K, st)   B operand = [V | 1]^T, N = D+8

    // accumulator-layout coordinate maps
    Tensor cS = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
    Tensor tScS = thr_mma_qk.partition_C(cS);
    Tensor tScS_rc = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol(tScS.layout()));   // (nrow, ncol) -> (m, n) in tile
    Tensor cO = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<T::kHeadDimV>{}));
    Tensor tOcO = thr_mma_pv.partition_C(cO);
    Tensor tOcO_rc = make_tensor(tOcO.data(), flash::convert_layout_acc_rowcol(tOcO.layout()));

    using AccS = decltype(partition_fragment_C(tiled_mma_qk, make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using AccO = decltype(partition_fragment_C(tiled_mma_pv, make_shape(Int<kBlockM>{}, Int<T::kHeadDimV>{})));   // O (D cols) and l (8 cols)
    AccO acc_o[R];
    constexpr int kNRows = 2;                                   // accumulator rows per thread
    constexpr int kNColsO = decltype(size<1>(make_tensor(acc_o[0].data(), flash::convert_layout_acc_rowcol(acc_o[0].layout()))))::value;   // (D+8)/4
    constexpr int kColL = kNColsO - 2;                          // this thread's first accumulator column >= D (holds l)
    static_assert(kNColsO == (T::kHeadDimV) / 4);
    float row_m[R][kNRows];                                     // running max per row, in S units (S = q.k + bias/scale)
    #pragma unroll
    for (int r = 0; r < R; ++r) { clear(acc_o[r]); row_m[r][0] = -INFINITY; row_m[r][1] = -INFINITY; }
    // Pin the zeroing here: left free, ptxas sinks each accumulator's zero-fill to just before its first wgmma, i.e. inside an
    // earlier wgmma's fence..wait window, and then serializes every wgmma of the kernel (ptxas diagnostic C7515).
    #pragma unroll
    for (int r = 0; r < R; ++r) { warpgroup_fence_operand(acc_o[r]); }

    AccS acc_s[R];                                              // S accumulator of row a / row b (pre-loaded with bias/scale)
    Tensor s_rc0 = make_tensor(acc_s[0].data(), flash::convert_layout_acc_rowcol(acc_s[0].layout()));
    constexpr int kNCols = decltype(size<1>(s_rc0))::value;     // = kBlockN / 4
    static_assert(decltype(size<0>(s_rc0))::value == kNRows && kNCols == kBlockN / 4);
    // P (bf16, PV A-operand registers) lives IN PLACE in the first half of each S accumulator's registers: in the A-register
    // element order t (= the accumulator's own element order), bf16 element t sits in 32-bit register t/2, so packing pair
    // (2p, 2p+1) -> register p in ascending p only overwrites registers whose S values were already consumed.
    static constexpr bool kPInPlace = (T::kFlags & 128) == 0;   // 128: P in its own registers (A-operand tensor separate from S)
    auto p_proto = make_tensor_like<Element>(make_tensor(acc_s[0].data(), flash::convert_layout_acc_Aregs<typename T::TiledMmaPV>(acc_s[0].layout())));
    auto p_layout = p_proto.layout();
    auto tOrP_a_sep = make_tensor_like<Element>(p_proto);
    auto tOrP_b_sep = make_tensor_like<Element>(p_proto);
    auto tOrP_a_in = make_tensor(make_rmem_ptr(reinterpret_cast<Element*>(raw_pointer_cast(acc_s[0].data()))), p_layout);
    auto tOrP_b_in = make_tensor(make_rmem_ptr(reinterpret_cast<Element*>(raw_pointer_cast(acc_s[1].data()))), p_layout);
    auto& tOrP_a = *([&]() { if constexpr (kPInPlace) return &tOrP_a_in; else return &tOrP_a_sep; }());
    auto& tOrP_b = *([&]() { if constexpr (kPInPlace) return &tOrP_b_in; else return &tOrP_b_sep; }());
    static_assert(decltype(size(p_layout))::value == kBlockN / 2 && decltype(cosize(p_layout))::value == kBlockN / 2);
    // pack S elements [8*KB0, 8*KB1) (fp32 p values, accumulator element order) into bf16 pairs: in place (register p <- elements
    // 2p, 2p+1) or into the separate P tensor (same element order)
    auto pack_inplace = [&](AccS& acc, auto kb0c, auto kb1c) __attribute__((always_inline)) {
        constexpr int KB0 = decltype(kb0c)::value, KB1 = decltype(kb1c)::value;
        if constexpr (kPInPlace) {
            #pragma unroll
            for (int pr = 4 * KB0; pr < 4 * KB1; ++pr) {
                __nv_bfloat162 const h2 = __floats2bfloat162_rn(acc(2 * pr), acc(2 * pr + 1));   // .x (low half) = element 2p
                acc(pr) = __uint_as_float(reinterpret_cast<uint32_t const&>(h2));
            }
        } else {
            auto& dstP = (&acc == &acc_s[0]) ? tOrP_a_sep : tOrP_b_sep;
            auto dst32 = recast<uint32_t>(dstP);                                                 // pair p <- elements 2p, 2p+1
            #pragma unroll
            for (int pr = 4 * KB0; pr < 4 * KB1; ++pr) {
                __nv_bfloat162 const h2 = __floats2bfloat162_rn(acc(2 * pr), acc(2 * pr + 1));
                dst32(pr) = reinterpret_cast<uint32_t const&>(h2);
            }
        }
    };
    constexpr bool kStream = (T::kFlags & 512) != 0;            // 512: max-free chunk-streaming loop (loop3) with the bias added by MMA
    if constexpr (!kStream) { warpgroup_fence_operand(acc_s[0]); warpgroup_fence_operand(acc_s[1]); }

    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kLazyNat = 8.0f;                            // running max may lag the true max by < 8 natural units (p <= e^8)
    float const scale = params.scale;
    float const inv_scale = 1.0f / scale;
    float const c_l2 = scale * kLog2e;                          // x (log2 units) = S * c_l2 - m * c_l2
    float const lazyS = kLazyNat * inv_scale;

    // Bias stage (fragment order): this thread's 4 floats for (row half mi, column pairs 2u, 2u+1 of chunk c) sit at
    //   stage + ((((c * 2 + cwg) * 4 + warp) * (kChunkW / 8) + mi * (kChunkW / 16) + u) * 32 + lane) * 16 bytes.
    uint32_t const sB_base = cute::cast_smem_ptr_to_uint(shared.smem_bias.data());
    constexpr uint32_t kBiasStageBytes = T::kBiasTileElems * sizeof(float);
    constexpr int kChunkW = T::kChunkW;
    uint32_t const bias_thr_off = uint32_t(((cwg * 4 + (thread_idx % 128) / 32) * (kChunkW / 8)) * 32 + (thread_idx % 32)) * 16u;
    auto bias_ld4 = [&](int slot, int c, int mi, int u, float& f0, float& f1, float& f2, float& f3) __attribute__((always_inline)) {
        uint32_t const a = sB_base + uint32_t(slot) * kBiasStageBytes + uint32_t(c) * uint32_t(2 * 4 * (kChunkW / 8) * 32 * 16) + bias_thr_off
                         + uint32_t(mi * (kChunkW / 16) + u) * (32u * 16u);
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(f0), "=f"(f1), "=f"(f2), "=f"(f3) : "r"(a));
    };
    // full-tile S accumulator (m64 x BN per WG) <- bias stage: rowcol columns ni = 4*nj' .. of pair nj = c*(CW/8) + 2u (+1)
    auto init_acc = [&](AccS& acc, int slot) __attribute__((always_inline)) {
        Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                #pragma unroll
                for (int u = 0; u < kChunkW / 16; ++u) {
                    int const nj = c * (kChunkW / 8) + 2 * u;
                    bias_ld4(slot, c, mi, u, s_rc(mi, 2 * nj), s_rc(mi, 2 * nj + 1), s_rc(mi, 2 * nj + 2), s_rc(mi, 2 * nj + 3));
                }
            }
        }
    };
    auto issue_qk = [&](AccS& acc, int r, int st) __attribute__((always_inline)) {
        warpgroup_fence_operand(acc);
        warpgroup_arrive();
        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
        auto tQ = tSrQ(_, _, _, r); auto tK = tSrK(_, _, _, st);
        #pragma unroll
        for (int kb = 0; kb < size<2>(tQ); ++kb) { cute::gemm(tiled_mma_qk, tQ(_, _, kb), tK(_, _, kb), acc); }
        warpgroup_commit_batch();
        warpgroup_fence_operand(acc);
    };
    auto issue_pv = [&](auto& tOrP, int r, int st) __attribute__((always_inline)) {
        warpgroup_fence_operand(tOrP); warpgroup_fence_operand(acc_o[r]);
        warpgroup_arrive();
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        auto tV = tOrV(_, _, _, st);
        #pragma unroll
        for (int kb = 0; kb < size<2>(tOrP); ++kb) { cute::gemm(tiled_mma_pv, tOrP(_, _, kb), tV(_, _, kb), acc_o[r]); }
        warpgroup_commit_batch();
        warpgroup_fence_operand(acc_o[r]);
    };

    // softmax of one row step, in place: acc (fp32 S of row r, k-tile j) -> P (bf16, tOrP); running max / O rescale bookkeeping
    int cur_it = 0;
    auto softmax_step = [&](AccS& acc, auto& tOrP, int r, int j) __attribute__((always_inline)) {
        Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
        int const rr = cy * R + r;                              // cluster row index of my row r
        int const cls = shared.tile_class[rr][j];
        if (cls != kClean) {
            if (shared.uniform[rr]) {
                // no attendable key in this row: uniform attention over its S keys (documented semantics), i.e. equal logits
                int const n_inrange = S - j * kBlockN;
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) {
                    #pragma unroll
                    for (int ni = 0; ni < kNCols; ++ni) { s_rc(mi, ni) = get<1>(tScS_rc(mi, ni)) < n_inrange ? 0.f : -INFINITY; }
                }
            } else {
                // this thread's columns: n(ni) = 8*(ni/2) + c2 + ni%2 with c2 = 2*(lane%4), so ni/8 selects the 32-key word (compile
                // time) and the bit within it is 8*((ni/2)%4) + ni%2 + c2
                int const c2 = get<1>(tScS_rc(0, 0));
                #pragma unroll
                for (int wq = 0; wq < kBlockN / 32; ++wq) {
                    uint32_t const w = shared.mask_bits[rr][j * (kBlockN / 32) + wq] >> c2;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        int const ni = wq * 8 + e;
                        bool const keep = (w >> (8 * (e >> 1) + (e & 1))) & 1u;
                        #pragma unroll
                        for (int mi = 0; mi < kNRows; ++mi) { s_rc(mi, ni) = keep ? s_rc(mi, ni) : -INFINITY; }
                    }
                }
            }
        }
        float nm[kNRows];
        if constexpr ((T::kFlags & 8) != 0) {
            // ABLATION (timing only): no running max at all, shift 0
            nm[0] = 0.f; nm[1] = 0.f;
        } else {
        float mx[kNRows];
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            float a0 = max(s_rc(mi, 0), s_rc(mi, 1)), a1 = max(s_rc(mi, 2), s_rc(mi, 3)), a2 = max(s_rc(mi, 4), s_rc(mi, 5)), a3 = max(s_rc(mi, 6), s_rc(mi, 7));
            #pragma unroll
            for (int ni = 8; ni < kNCols; ni += 8) {
                a0 = max(a0, max(s_rc(mi, ni), s_rc(mi, ni + 1))); a1 = max(a1, max(s_rc(mi, ni + 2), s_rc(mi, ni + 3)));
                a2 = max(a2, max(s_rc(mi, ni + 4), s_rc(mi, ni + 5))); a3 = max(a3, max(s_rc(mi, ni + 6), s_rc(mi, ni + 7)));
            }
            mx[mi] = max(max(a0, a1), max(a2, a3));
        }
        bool any_up = false;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) { any_up |= (mx[mi] > row_m[r][mi] + lazyS); }
        bool resc;
        if constexpr ((T::kFlags & 2) != 0) { resc = true; } else { resc = __any_sync(0xffffffffu, any_up); }   // rare after a row's first tile
        if (resc) {
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                flash::MaxOp<float> op;
                float const gmx = flash::Allreduce<4>::run(mx[mi], op);
                float const m_new = gmx > row_m[r][mi] + lazyS ? gmx : row_m[r][mi];
                float const alpha = ex2_approx((row_m[r][mi] - m_new) * c_l2);          // row_m = -inf -> 0 (acc_o is 0 then)
                row_m[r][mi] = m_new;
                Tensor o_rc = make_tensor(acc_o[r].data(), flash::convert_layout_acc_rowcol(acc_o[r].layout()));
                #pragma unroll
                for (int ni = 0; ni < kNColsO; ++ni) { o_rc(mi, ni) *= alpha; }
            }
        }
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) { nm[mi] = row_m[r][mi] == -INFINITY ? 0.f : -row_m[r][mi] * c_l2; }
        }
        if constexpr (T::kPingPong) { warp_scheduler_barrier_sync(); }
        if (r == 0) stamp(cur_it, 14);
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < kNCols; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[mi])); }
        }
        if (r == 0) stamp(cur_it, 15);
        pack_inplace(acc, Int<0>{}, Int<kBlockN / 16>{});
    };

    auto kv_stage = [&](int jj, int r) __attribute__((always_inline)) { return (jj % T::kRingKV) * R + r; };
    auto kv_phase = [&](int jj) __attribute__((always_inline)) { return uint32_t((jj / T::kRingKV) & 1); };
    auto b_slot = [&](int jj) __attribute__((always_inline)) { return jj % T::kRingB; };
    auto b_phase = [&](int jj) __attribute__((always_inline)) { return uint32_t((jj / T::kRingB) & 1); };

    shared.barrier_q.wait(0);
    if constexpr (T::kPingPong) {                               // consumer warpgroup 0 starts holding the turn
        if (cwg == 1) {
            cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup,
                static_cast<uint32_t>(NamedBarriers::WarpSchedulerWG1) + static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier));
        }
    }

    // =========================== phases of the interleaved (fast) loop ===========================================
    // M phase: [mask] -> tile row max -> exact running max, O/l rescale (unconditional: branch-free), exponent offset
    auto m_phase = [&](AccS& acc, int r, int j, bool masked) __attribute__((always_inline)) {
        Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
        int const rr = cy * R + r;
        if (masked) {                                           // warp-uniform; no wgmma inside the branch
            int const c2 = get<1>(tScS_rc(0, 0));
            #pragma unroll
            for (int wq = 0; wq < kBlockN / 32; ++wq) {
                uint32_t const w = shared.mask_bits[rr][j * (kBlockN / 32) + wq] >> c2;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    int const ni = wq * 8 + e;
                    bool const keep = (w >> (8 * (e >> 1) + (e & 1))) & 1u;
                    #pragma unroll
                    for (int mi = 0; mi < kNRows; ++mi) { s_rc(mi, ni) = keep ? s_rc(mi, ni) : -INFINITY; }
                }
            }
        }
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            float a0 = max(s_rc(mi, 0), s_rc(mi, 1)), a1 = max(s_rc(mi, 2), s_rc(mi, 3)), a2 = max(s_rc(mi, 4), s_rc(mi, 5)), a3 = max(s_rc(mi, 6), s_rc(mi, 7));
            #pragma unroll
            for (int ni = 8; ni < kNCols; ni += 8) {
                a0 = max(a0, max(s_rc(mi, ni), s_rc(mi, ni + 1))); a1 = max(a1, max(s_rc(mi, ni + 2), s_rc(mi, ni + 3)));
                a2 = max(a2, max(s_rc(mi, ni + 4), s_rc(mi, ni + 5))); a3 = max(a3, max(s_rc(mi, ni + 6), s_rc(mi, ni + 7)));
            }
            flash::MaxOp<float> op;
            float const gmx = flash::Allreduce<4>::run(max(max(a0, a1), max(a2, a3)), op);
            float const m_old = row_m[r][mi];
            float const m_new = max(m_old, gmx);
            float const alpha = (m_new == m_old) ? 1.f : ex2_approx((m_old - m_new) * c_l2);   // m_old = -inf -> 0 (O is 0 then)
            row_m[r][mi] = m_new;
            Tensor o_rc = make_tensor(acc_o[r].data(), flash::convert_layout_acc_rowcol(acc_o[r].layout()));
            #pragma unroll
            for (int ni = 0; ni < kNColsO; ++ni) { o_rc(mi, ni) *= alpha; }
        }
    };
    // E phase over P k-blocks [KB0, KB1) (k-block = 16 keys = rowcol columns ni 4*kb .. 4*kb+3): p = 2^(S*c - m*c) -> bf16 P regs
    auto e_phase = [&](AccS& acc, auto& tOrP, int r, auto kb0c, auto kb1c) __attribute__((always_inline)) {
        constexpr int KB0 = decltype(kb0c)::value, KB1 = decltype(kb1c)::value;
        Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
        float nm[kNRows];
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) { nm[mi] = (row_m[r][mi] == -INFINITY) ? 0.f : -row_m[r][mi] * c_l2; }
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 4 * KB0; ni < 4 * KB1; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[mi])); }
        }
        pack_inplace(acc, kb0c, kb1c);
    };
    // loop2: per iteration both S tiles have landed at the start; the M phase (mask, max, rescale) of one row is placed in the
    // same straight-line block as the E phase (exponentials) of the other row so ptxas can interleave the latency-bound max
    // chain with the MUFU-bound exponentials; QK GEMMs of the next tile are issued as soon as a row's S registers are free
    // (P lives in separate registers), each half-iteration ends with every wgmma retired (FA3-style anchor).
    static constexpr int kNKB = kBlockN / 16;                  // P k-blocks per tile
    using KB0 = Int<0>; using KBH = Int<kNKB / 2>; using KBN = Int<kNKB>;
    auto blockA = [&](int jj, int j, bool mb, auto last) __attribute__((always_inline)) {
        constexpr bool kLast = decltype(last)::value;
        stamp(jj, 0);
        m_phase(acc_s[1], 1, j, mb);                                                             // M_b(jj)   } one block
        e_phase(acc_s[0], tOrP_a, 0, KB0{}, KBN{});                                              // E_a(jj)   }
        stamp(jj, 1);
        issue_pv(tOrP_a, 0, kv_stage(jj, 0));                                                   // PV_a(jj)
        stamp(jj, 2);
        if constexpr (!kLast) {
            pipe_b.wait_full(b_slot(jj + 1), b_phase(jj + 1));
            stamp(jj, 3);
            init_acc(acc_s[0], b_slot(jj + 1));
            stamp(jj, 4);
            pipe_kv.wait_full(kv_stage(jj + 1, 0), kv_phase(jj + 1));
            issue_qk(acc_s[0], 0, kv_stage(jj + 1, 0));                                        // QK_a(jj+1)
        }
        stamp(jj, 5);
    };
    auto blockB = [&](int jj, int jn, bool ma, auto last) __attribute__((always_inline)) {
        constexpr bool kLast = decltype(last)::value;
        e_phase(acc_s[1], tOrP_b, 1, KB0{}, KBH{});                                              // E_b(jj) first half
        stamp(jj, 6);
        warpgroup_wait<0>();                                                                     // PV_a(jj) retired, S_a(jj+1) landed
        warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_s[0]);
        stamp(jj, 7);
        if constexpr (!kLast) { m_phase(acc_s[0], 0, jn, ma); }                                 // M_a(jj+1) } one block
        e_phase(acc_s[1], tOrP_b, 1, KBH{}, KBN{});                                              // E_b h2    }
        stamp(jj, 8);
        issue_pv(tOrP_b, 1, kv_stage(jj, 1));                                                   // PV_b(jj)
        stamp(jj, 9);
        if constexpr (!kLast) {
            init_acc(acc_s[1], b_slot(jj + 1));
            __syncwarp();
            pipe_b.release(b_slot(jj + 1), lane == 0, b_ranks, self_rank);                    // bias(jj+1) read by both inits
            stamp(jj, 10);
            pipe_kv.wait_full(kv_stage(jj + 1, 1), kv_phase(jj + 1));
            issue_qk(acc_s[1], 1, kv_stage(jj + 1, 1));                                        // QK_b(jj+1)
        }
        stamp(jj, 11);
        warpgroup_wait<0>();                                                                     // PV_b(jj) retired, S_b(jj+1) landed
        warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(acc_s[1]);
        stamp(jj, 12);
        pipe_kv.release(kv_stage(jj, 0), wg_leader, kv_ranks, self_rank);
        pipe_kv.release(kv_stage(jj, 1), wg_leader, kv_ranks, self_rank);
    };
    auto tile_masked = [&](int r, int j) __attribute__((always_inline)) { return shared.tile_class[cy * R + r][j] != kClean; };

    if constexpr (kStream) {
    // =====================================================================================================================
    // loop3: MAX-FREE CHUNK STREAMING. The k-tiles of both rows are consumed as one stream of S-chunks (64 q x CW keys per WG),
    // chunk k = (tile jj, row r, half c): k = 4*jj + 2*r + c. Per chunk: S(k) = Q K^T + (beta I) Bias (one wgmma group, issued one
    // chunk ahead); x = S*c_l2 + nm[r] with a FIXED per-row shift nm (seeded from the exact row max of tile 0, so every later
    // p = 2^x can only overflow (-> Inf -> fixup pass) and never silently flush); p = ex2(x) in place; bf16 pack and the PV wgmma
    // over [V|1] run one chunk late; 3 S-chunk buffers, 2 P buffers; all wgmma retired at the end of every PERIOD of 12 chunks
    // (3 tiles = the K/V ring, so every stage index is compile-time) so ptxas can account async groups statically.
    // Tile 0 is the prologue; period p >= 0 covers tiles jj = 1 + 3p + t (t = 0..2), chunks k = 4 + 12p + dd (dd = 0..11).
    // Rows without any attendable key ("uniform") get logits 0 on their in-range keys.
    // =====================================================================================================================
        static_assert(R == 2 && T::kRingKV == 3 && T::kRingB == 2, "stream loop: R = 2, K/V ring 3 (= period), bias ring 2");
        static_assert(T::kBiasMMA, "the stream loop adds the bias on the tensor core");
        constexpr int CW = kChunkW;
        constexpr int kNC = CW / 4;                             // rowcol columns per thread per chunk
        constexpr int kNKBc = CW / 16;                          // PV k-blocks per chunk
        typename T::TiledMmaQKC tiled_mma_qkc;
        auto wg_mma_qkc = tiled_mma_qkc.get_slice(wg_layout(cwg));
        auto thr_mma_qkc = tiled_mma_qkc.get_thread_slice(thread_idx);
        using AccC = decltype(partition_fragment_C(tiled_mma_qkc, make_shape(Int<kBlockM>{}, Int<CW>{})));
        Tensor cSc = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<CW>{}));
        Tensor tCc = thr_mma_qkc.partition_C(cSc);
        Tensor tCc_rc = make_tensor(tCc.data(), flash::convert_layout_acc_rowcol(tCc.layout()));
        typename T::TiledMmaBias tiled_mma_bias;
        auto thr_mma_bias = tiled_mma_bias.get_thread_slice(thread_idx % 128);
        // operand descriptor tensors with (k-block, chunk, stage) modes. K/V/bias ones are rebuilt per loop period from a base offset by
        // params.zero * k0 (== 0): the ~90 distinct descriptors of an unrolled period are then not loop-invariant, so ptxas derives each
        // at its use from a uniform base (UIADD3) instead of hoisting all of them into registers (spills) -- and, unlike an asm launder,
        // the values stay on the uniform datapath (no R2UR chains in front of the HGMMAs).
        Tensor tSrQc = wg_mma_qkc.partition_fragment_A(sQ);                                                            // (frag, 1, 2, R)
        auto mk_K = [&](int zoff) {
            Tensor sKz = make_tensor(make_smem_ptr(shared.smem_k.data() + zoff), typename T::SmemLayoutK{});
            return wg_mma_qkc.partition_fragment_B(local_tile(sKz, make_shape(Int<CW>{}, Int<kHeadDim>{}), make_coord(_, _0{})));   // (frag, N, 2, c, st)
        };
        auto mk_V = [&](int zoff) {
            Tensor sVz = make_tensor(make_smem_ptr(shared.smem_v.data() + zoff), typename T::SmemLayoutVt{});
            return wg_mma_pv.partition_fragment_B(local_tile(sVz, make_shape(Int<T::kHeadDimV>{}, Int<CW>{}), make_coord(_0{}, _)));  // (frag, N, 4, c, st)
        };
        Tensor sI = make_tensor(make_smem_ptr(shared.smem_ident.data()), typename T::SmemLayoutI{});
        Tensor tIrI = thr_mma_bias.partition_fragment_A(sI);                                                            // (frag, 1, 4)
        // bias stages as the MN-major B operand (n = key, k = q row): 8 KB blocks, block index = warpgroup + 2*chunk + 4*slot
        auto sBall_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{}, Shape<_64, _64, Int<4 * T::kRingB>>{});
        static_assert(decltype(cosize(sBall_layout))::value == T::kBiasTileElems * T::kRingB && decltype(sBall_layout(_0{}, _0{}, _1{}))::value == 64 * 64);
        auto mk_B = [&](int zoff) {
            Tensor sBz = make_tensor(make_smem_ptr(shared.smem_bias.data() + zoff), sBall_layout);
            return thr_mma_bias.partition_fragment_B(sBz);                                                              // (frag, N, 4, blk)
        };
        using OpK = decltype(mk_K(0)); using OpV = decltype(mk_V(0)); using OpB = decltype(mk_B(0));
        using AccBiasLayout = decltype(partition_fragment_C(tiled_mma_bias, make_shape(_64{}, Int<CW>{})).layout());
        static_assert(decltype(size(AccBiasLayout{}))::value == decltype(size(AccC{}.layout()))::value);
        AccC accC[4];                                           // S chunk buffers, chunk k lives in accC[k % 4]; QK runs two chunks ahead
        auto pc_proto = make_tensor_like<Element>(make_tensor(accC[0].data(), flash::convert_layout_acc_Aregs<typename T::TiledMmaPV>(accC[0].layout())));
        decltype(pc_proto) PCb[2];                              // bf16 P of chunk k in PCb[k & 1] (PV(k-1) may still be reading its buffer while chunk k is packed)
        static_assert(decltype(size<2>(pc_proto))::value == kNKBc);
        float nm[R][kNRows];
        int const K_total = n_active * 4;

        // ---- per-chunk primitives: K/V stage st, chunk half c, row r compile-time; bias slot runtime ----
        auto issue_qk = [&](AccC& acc, OpK const& tK, OpB const& tB, auto rc, auto cc, auto stc, int slot) __attribute__((always_inline)) {
            constexpr int r = decltype(rc)::value, c = decltype(cc)::value, st = decltype(stc)::value;
            Tensor accb = make_tensor(acc.data(), AccBiasLayout{});
            int const blk = cwg + 2 * c + 4 * slot;
            warpgroup_fence_operand(acc);
            warpgroup_arrive();
            tiled_mma_qkc.accumulate_ = GMMA::ScaleOut::Zero;
            #pragma unroll
            for (int kb = 0; kb < 2; ++kb) { cute::gemm(tiled_mma_qkc, tSrQc(_, _, kb, r), tK(_, _, kb, c, st), acc); tiled_mma_qkc.accumulate_ = GMMA::ScaleOut::One; }
            tiled_mma_bias.accumulate_ = GMMA::ScaleOut::One;
            if constexpr ((T::kFlags & 2048) == 0) {                     // 2048: ABLATION (timing only) -- no bias MMA
                #pragma unroll
                for (int kb = 0; kb < 4; ++kb) { cute::gemm(tiled_mma_bias, tIrI(_, _, kb), tB(_, _, kb, blk), accb); }
            }
            warpgroup_commit_batch();                                    // no fence here: the accumulator is in flight until the next wait
        };
        auto issue_pvc = [&](decltype(pc_proto)& tP, OpV const& tV, auto rc, auto cc, auto stc) __attribute__((always_inline)) {   // no fence on acc_o (chained PVs)
            constexpr int r = decltype(rc)::value, c = decltype(cc)::value, st = decltype(stc)::value;
            warpgroup_fence_operand(tP);
            warpgroup_arrive();
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
            #pragma unroll
            for (int kb = 0; kb < kNKBc; ++kb) { cute::gemm(tiled_mma_pv, tP(_, _, kb), tV(_, _, kb, c, st), acc_o[r]); }
            warpgroup_commit_batch();
        };
        auto pack_chunk = [&](AccC& acc, decltype(pc_proto)& tP) __attribute__((always_inline)) {
            auto dst32 = recast<uint32_t>(tP);
            #pragma unroll
            for (int pr = 0; pr < CW / 4; ++pr) {
                __nv_bfloat162 const h2 = __floats2bfloat162_rn(acc(2 * pr), acc(2 * pr + 1));
                dst32(pr) = reinterpret_cast<uint32_t const&>(h2);
            }
        };
        // masked keys of chunk (tile j, row r, half c) -> -inf; rows without attendable keys ("uniform"): 0 on in-range keys
        auto mask_chunk = [&](AccC& acc, int j, int r, int c) __attribute__((always_inline)) {
            int const rr = cy * R + r;
            Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
            int const key0 = j * kBlockN + c * CW + get<1>(tCc_rc(0, 0));                     // key of this thread's column 0
            if (shared.uniform[rr]) {
                #pragma unroll
                for (int nj = 0; nj < kNC / 2; ++nj) {
                    #pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        bool const keep = key0 + 8 * nj + e < S;
                        #pragma unroll
                        for (int mi = 0; mi < kNRows; ++mi) { s_rc(mi, 2 * nj + e) = keep ? 0.f : -INFINITY; }
                    }
                }
            } else {
                #pragma unroll
                for (int nj = 0; nj < kNC / 2; ++nj) {
                    int const kk = key0 + 8 * nj;
                    uint32_t const w = shared.mask_bits[rr][kk >> 5] >> (kk & 31);             // kk even, kk+1 in the same word
                    #pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        bool const keep = (w >> e) & 1u;
                        #pragma unroll
                        for (int mi = 0; mi < kNRows; ++mi) { s_rc(mi, 2 * nj + e) = keep ? s_rc(mi, 2 * nj + e) : -INFINITY; }
                    }
                }
            }
        };
        // kPingPong: the two consumer warpgroups take TURNS on the exponential section (named-barrier hand-off, FA3-style), so the
        // warp pair of an SM sub-partition is anti-phased: one warp's 32 MUFU.EX2 (8 clk each on the shared XU) run while the other
        // warp does its pack / wgmma issue / barrier work, instead of both halving each other's MUFU rate and then idling the XU.
        auto pp_sync = [&]() __attribute__((always_inline)) { if constexpr (T::kPingPong) { warp_scheduler_barrier_sync(); } };
        auto pp_arrive = [&]() __attribute__((always_inline)) { if constexpr (T::kPingPong) { warp_scheduler_barrier_arrive(); } };
        auto exp_core = [&](AccC& acc, int r) __attribute__((always_inline)) {
            Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[r][mi])); }
            }
            warpgroup_fence_operand(acc);        // the exponentials stay BEFORE the next body's wgmma wait (else ptxas sinks them behind it: no E/MMA overlap)
        };
        auto exp_chunk = [&](AccC& acc, int r) __attribute__((always_inline)) { pp_sync(); exp_core(acc, r); pp_arrive(); };   // prologue use
        auto seed_row = [&](AccC& a0, AccC& a1, int r) __attribute__((always_inline)) {      // nm[r] <- -max(tile 0 of row r) * c_l2
            Tensor s0 = make_tensor(a0.data(), flash::convert_layout_acc_rowcol(a0.layout()));
            Tensor s1 = make_tensor(a1.data(), flash::convert_layout_acc_rowcol(a1.layout()));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                float mx = -INFINITY;
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { mx = max(mx, max(s0(mi, ni), s1(mi, ni))); }
                mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                nm[r][mi] = (mx == -INFINITY) ? 0.f : -mx * c_l2;                            // all -inf cannot happen (uniform rows are 0)
            }
        };
        // drained: rare power-of-two renormalisation of (O, l, nm) of a row whose sum got large; `pend` holds the p values of the one
        // chunk (row r_pend) that is exp'd but not yet packed, they are rescaled with the row.
        auto l_check = [&](AccC& pend, int r_pend) __attribute__((always_inline)) {
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                warpgroup_fence_operand(acc_o[r]);
                Tensor o_rc = make_tensor(acc_o[r].data(), flash::convert_layout_acc_rowcol(acc_o[r].layout()));
                bool big = false;
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) { big |= (o_rc(mi, kColL) > 0x1p+60f); }
                if (__any_sync(0xffffffffu, big)) {
                    Tensor p_rc = make_tensor(pend.data(), flash::convert_layout_acc_rowcol(pend.layout()));
                    #pragma unroll
                    for (int mi = 0; mi < kNRows; ++mi) {
                        bool const b_ = o_rc(mi, kColL) > 0x1p+60f;
                        float const f = b_ ? 0x1p-64f : 1.f;
                        nm[r][mi] += b_ ? -64.f : 0.f;
                        #pragma unroll
                        for (int ni = 0; ni < kNColsO; ++ni) { o_rc(mi, ni) *= f; }
                        if (r == r_pend) {
                            #pragma unroll
                            for (int ni = 0; ni < kNC; ++ni) { p_rc(mi, ni) *= f; }
                        }
                    }
                }
                warpgroup_fence_operand(acc_o[r]);
            }
        };
        // ---- period structure: PERIOD = 3 tiles = 12 chunks (= the K/V ring, so every K/V STAGE index is compile-time); tile 0 is the
        //      prologue, period p >= 0 holds tiles jj = 1 + 3p + t (t = 0..2), chunks k = 4 + 12p + dd (dd = 0..11, k % 3 == (1+dd) % 3).
        //      K/V stage of (t, r) = 2*((1+t)%3) + r; runtime per tile from jj: K/V phase (jj/3)&1, bias slot jj&1, bias phase (jj>>1)&1.
        //      Tile 0 == "t = 2 of period -1" (stages 0/1, phases 0, slot 0). A partial last period leaves the loop after body 3 or 7,
        //      so the loop body is the only copy of the chunk code (instruction-cache footprint).
        constexpr bool kReplay = (T::kFlags & 65536) != 0;     // ABLATION (timing only): after the first ring fill no barrier traffic at all (stale tiles recomputed)
        auto st_of  = [](auto tc, auto rc) { return Int<2 * ((1 + decltype(tc)::value) % 3) + decltype(rc)::value>{}; };
        // per-tile runtime facts, loaded at the tile's first chunk: live k-tile index j and the two rows' mask-class bits
        int jcur = 0; uint32_t mskcur = 0;
        auto load_tile = [&](int jj) __attribute__((always_inline)) {
            jcur = int(shared.active[jj]);
            mskcur = uint32_t(shared.tile_class[cy * R + 0][jcur] != kClean) | (uint32_t(shared.tile_class[cy * R + 1][jcur] != kClean) << 1);
        };

        // ---- body for chunk dd (0..11) of a period, k = 4 + 12p + dd (k % 4 == dd % 4):
        //      wait<1>: everything but PV(k-1)... i.e. QK(k+1) (issued a whole E phase ago) and PV(k-2) retired | releases | waits for
        //      chunk k+2's tile, QK(k+2) -> accC[(k+2)%4] | mask + E(k) | pack(k-1) -> PCb[(k-1)&1], PV(k-1) LAST (ptxas can hoist the
        //      next wait only up to this commit, so E(k) stays ahead of it). QK two chunks ahead: its HGMMAs, wherever ptxas spreads
        //      them inside E(k), have a full chunk to land. Body 0 follows a full drain and issues no wait (a wait with nothing pending
        //      makes ptxas serialize every wgmma in the function, C7514/C7515).
        auto body = [&](auto ddc, int jbase, OpK const& tK, OpV const& tV, OpB const& tB) __attribute__((always_inline)) {
            constexpr int dd = decltype(ddc)::value;
            constexpr int bq = dd & 3, bn = (dd + 2) & 3, bp = (dd + 3) & 3;     // S(k), S(k+2), S(k-1) buffers
            constexpr int pb = (dd + 1) & 1;                                     // P buffer of chunk k-1
            constexpr int t = dd / 4, r = (dd >> 1) & 1, c = dd & 1;
            constexpr int dd2 = dd + 2, t2 = dd2 / 4, r2 = (dd2 >> 1) & 1, c2 = dd2 & 1;            // chunk k+2 (t2 == 3: next period's tile 0)
            constexpr int ddp = (dd + 11) % 12, tp = ddp / 4, rp = (ddp >> 1) & 1, cpv = ddp & 1;    // chunk k-1 (dd == 0: previous period's tile 2)
            constexpr int ddo = (dd + 9) % 12, to = ddo / 4, ro = (ddo >> 1) & 1, co = ddo & 1;      // chunk k-3 (its PV has retired at this body's wait)
            int const kk = 4 + (jbase - 1) * 4 + dd;                             // chunk index (trace only)
            stamp(kk, 0);
            if constexpr (c == 0 && r == 0) { load_tile(jbase + t); }            // first chunk of the tile: live index + mask-class bits
            if constexpr (dd != 0) { warpgroup_wait<1>(); }
            warpgroup_fence_operand(accC[bq]);                                   // S(k) reads below stay after the wait
            warpgroup_fence_operand(PCb[pb]);                                    // pack's P writes stay after the wait (PV(k-3) read this buffer)
            stamp(kk, 1);
            if (!kReplay) {
                if constexpr (co == 1) { pipe_kv.release(decltype(st_of(Int<to>{}, Int<ro>{}))::value, wg_leader, kv_ranks, self_rank); }  // chunk k-3 = half 1 of its stage
                if constexpr ((dd & 3) == 2) { __syncwarp(); pipe_b.release((jbase + t) & 1, lane == 0, b_ranks, self_rank); }         // this tile's 4 QKs (chunks 4t..4t+3 = k-2..k+1) retired
            }
            stamp(kk, 2);
            int const j2 = jbase + t2;                                           // tile of chunk k+2
            if (j2 < n_active && (!kReplay || j2 < 3)) {
                if constexpr ((dd2 & 3) == 0) { if (!kReplay || j2 < 2) { pipe_b.wait_full(j2 & 1, uint32_t((j2 >> 1) & 1)); } }
                if constexpr (c2 == 0) { pipe_kv.wait_full(decltype(st_of(Int<t2 % 3>{}, Int<r2>{}))::value, uint32_t((j2 / 3) & 1)); }
            }
            stamp(kk, 3);
            issue_qk(accC[bn], tK, tB, Int<r2>{}, Int<c2>{}, st_of(Int<t2 % 3>{}, Int<r2>{}), j2 & 1);   // phantom past the end: stale smem, result unused
            stamp(kk, 4);
            if constexpr ((T::kFlags & 8192) == 0) { if ((mskcur >> r) & 1u) { mask_chunk(accC[bq], jcur, r, c); } }   // 8192: ABLATION no mask step
            stamp(kk, 5);
            pp_sync();
            stamp(kk, 9);
            exp_core(accC[bq], r);
            stamp(kk, 10);
            pp_arrive();
            stamp(kk, 6);
            pack_chunk(accC[bp], PCb[pb]);                                       // previous chunk's p values (no MUFU->F2FP adjacency)
            stamp(kk, 7);
            issue_pvc(PCb[pb], tV, Int<rp>{}, Int<cpv>{}, st_of(Int<tp>{}, Int<rp>{}));
            stamp(kk, 8);
        };
        auto drain = [&]() __attribute__((always_inline)) {                    // after body 3/7/11: the exp'd, unpacked chunk (row b) sits in accC[3]
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(accC[3]);
            warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]);
            l_check(accC[3], 1);
        };

        // ---- prologue: tile 0 (chunks 0..3 in accC[0..3]; stages 0/1; bias slot 0; all phases 0) with the exact per-row max as the
        //      shift; ends DRAINED in the state body 0 expects: S(4), S(5) landed in accC[0], accC[1]; chunk 3 exp'd in accC[3], unpacked.
        OpK const tK0 = mk_K(params.zero); OpV const tV0 = mk_V(params.zero); OpB const tB0 = mk_B(params.zero);
        load_tile(0);
        pipe_b.wait_full(0, 0); pipe_kv.wait_full(0, 0);
        issue_qk(accC[0], tK0, tB0, _0{}, _0{}, _0{}, 0); issue_qk(accC[1], tK0, tB0, _0{}, _1{}, _0{}, 0);
        warpgroup_wait<0>();
        warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]);
        if (mskcur & 1u) { mask_chunk(accC[0], jcur, 0, 0); mask_chunk(accC[1], jcur, 0, 1); }
        seed_row(accC[0], accC[1], 0);
        pipe_kv.wait_full(1, 0);
        issue_qk(accC[2], tK0, tB0, _1{}, _0{}, _1{}, 0); issue_qk(accC[3], tK0, tB0, _1{}, _1{}, _1{}, 0);   // QK(2), QK(3) under E(0), E(1)
        exp_chunk(accC[0], 0); pack_chunk(accC[0], PCb[0]); issue_pvc(PCb[0], tV0, _0{}, _0{}, _0{});
        exp_chunk(accC[1], 0); pack_chunk(accC[1], PCb[1]); issue_pvc(PCb[1], tV0, _0{}, _1{}, _0{});
        warpgroup_wait<0>();
        warpgroup_fence_operand(accC[2]); warpgroup_fence_operand(accC[3]); warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]); warpgroup_fence_operand(acc_o[0]);
        if (!kReplay) { __syncwarp(); pipe_b.release(0, lane == 0, b_ranks, self_rank); }     // tile 0's bias stage: its 4 QKs retired
        // (K/V stage 0 = row a of tile 0 is released by body 0, whose chunk k-3 is chunk 1; stage 1 by body 2 or `finish`)
        if (mskcur & 2u) { mask_chunk(accC[2], jcur, 1, 0); mask_chunk(accC[3], jcur, 1, 1); }
        seed_row(accC[2], accC[3], 1);
        if (n_active > 1) { pipe_b.wait_full(1, 0); pipe_kv.wait_full(2, 0); }                 // tile 1 = t 0 of period 0: slot 1 phase 0, stage 2 phase 0
        issue_qk(accC[0], tK0, tB0, _0{}, _0{}, _2{}, 1); issue_qk(accC[1], tK0, tB0, _0{}, _1{}, _2{}, 1);   // QK(4), QK(5) (phantom when n_active == 1)
        exp_chunk(accC[2], 1); pack_chunk(accC[2], PCb[0]); issue_pvc(PCb[0], tV0, _1{}, _0{}, _1{});
        exp_chunk(accC[3], 1);                                                                  // chunk 3: packed + PV'd by body 0
        warpgroup_wait<0>();                                                                    // loop entry == the drained state body 0 assumes
        warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]); warpgroup_fence_operand(accC[2]); warpgroup_fence_operand(accC[3]);
        warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]); warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]);
        // stage 1 (row b of tile 0, chunks 2/3): released by body 2 (its k-3 = 3) or by `finish`; bias slot 1 (tile 1): body 1.

        // ---- the loop: full periods and the partial last one (4 or 8 chunks) share this single copy of the bodies; ends drained
        #pragma unroll 1
        for (int k0 = 4, jbase = 1; k0 < K_total; k0 += 12, jbase += 3) {
            int const left = K_total - k0;                                       // 4, 8 or >= 12
            int const zoff = params.zero * k0;                                   // == 0, opaque per period
            OpK const tK = mk_K(zoff); OpV const tV = mk_V(zoff); OpB const tB = mk_B(zoff);
            body(Int<0>{}, jbase, tK, tV, tB); body(Int<1>{}, jbase, tK, tV, tB); body(Int<2>{}, jbase, tK, tV, tB);  body(Int<3>{}, jbase, tK, tV, tB);
            if (left == 4) { drain(); break; }
            body(Int<4>{}, jbase, tK, tV, tB); body(Int<5>{}, jbase, tK, tV, tB); body(Int<6>{}, jbase, tK, tV, tB);  body(Int<7>{}, jbase, tK, tV, tB);
            if (left == 8) { drain(); break; }
            body(Int<8>{}, jbase, tK, tV, tB); body(Int<9>{}, jbase, tK, tV, tB); body(Int<10>{}, jbase, tK, tV, tB); body(Int<11>{}, jbase, tK, tV, tB);
            drain();
        }
        {   // last chunk K-1 = 4 jl + 3 (row b, half 1 of tile jl = n_active-1), exp'd in accC[3]; its K/V stages are 2*(jl%3) + r:
            // three cases keyed by jl % 3. State: drained.
            int const m3 = (n_active - 1) % 3;
            auto finish = [&](auto m3c) __attribute__((always_inline)) {
                constexpr int st0 = 2 * decltype(m3c)::value;
                pack_chunk(accC[3], PCb[1]);
                issue_pvc(PCb[1], tV0, _1{}, _1{}, Int<st0 + 1>{});
                warpgroup_wait<0>();
                warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]);
                if (!kReplay) {                                                  // the last tile's two stages (chunks K-3, K-1: no later body released them)
                    pipe_kv.release(st0, wg_leader, kv_ranks, self_rank);
                    pipe_kv.release(st0 + 1, wg_leader, kv_ranks, self_rank);
                }
            };
            if (m3 == 0) { finish(_0{}); } else if (m3 == 1) { finish(_1{}); } else { finish(_2{}); }
        }
    } else {
    bool any_uniform = false;
    #pragma unroll
    for (int r = 0; r < R; ++r) { any_uniform |= (shared.uniform[cy * R + r] != 0); }
    bool const use_fast = ((T::kFlags & 256) != 0) || (((T::kFlags & 64) != 0) && !any_uniform);   // 64: loop2 unless a row is fully masked; 256: loop2 only (experiment)

    if (use_fast) {
        static_assert(!kPInPlace || (T::kFlags & 64) == 0, "loop2 needs P in its own registers (flag 128)");
        // ---- prologue: both S tiles of active tile 0 initialised, QK'd, landed; M_a(0) done -------------------------------
        int const j0 = shared.active[0];
        pipe_b.wait_full(b_slot(0), b_phase(0));
        init_acc(acc_s[0], b_slot(0));
        init_acc(acc_s[1], b_slot(0));
        __syncwarp();
        pipe_b.release(b_slot(0), lane == 0, b_ranks, self_rank);
        pipe_kv.wait_full(kv_stage(0, 0), kv_phase(0));
        issue_qk(acc_s[0], 0, kv_stage(0, 0));                 // QK_a(0)
        pipe_kv.wait_full(kv_stage(0, 1), kv_phase(0));
        issue_qk(acc_s[1], 1, kv_stage(0, 1));                 // QK_b(0)
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_s[0]); warpgroup_fence_operand(acc_s[1]);
        m_phase(acc_s[0], 0, j0, tile_masked(0, j0));
        // ---- steady state (no divergent code around wgmma: mask handling is a branch inside the M phase only) -----------
        for (int jj = 0; jj + 1 < n_active; ++jj) {
            int const j = shared.active[jj];
            int const jn = shared.active[jj + 1];
            blockA(jj, j, tile_masked(1, j), cute::false_type{});
            blockB(jj, jn, tile_masked(0, jn), cute::false_type{});
        }
        {   // last tile
            int const jj = n_active - 1;
            int const j = shared.active[jj];
            blockA(jj, j, tile_masked(1, j), cute::true_type{});
            blockB(jj, j, false, cute::true_type{});
        }
    } else if constexpr ((T::kFlags & 256) == 0) {
        // ---- prologue: both S accumulators of tile 0 <- bias/scale; S of row a landed ---------------------------------------
        pipe_b.wait_full(b_slot(0), b_phase(0));
        init_acc(acc_s[0], b_slot(0));
        init_acc(acc_s[1], b_slot(0));
        __syncwarp();
        pipe_b.release(b_slot(0), lane == 0, b_ranks, self_rank);
        pipe_kv.wait_full(kv_stage(0, 0), kv_phase(0));
        issue_qk(acc_s[0], 0, kv_stage(0, 0));                     // group: QK_a(0)
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_s[0]);

        // ---- main loop over active k-tiles ------------------------------------------------------------------------------
        // Iteration jj starts with no wgmma in flight and S_a(jj) landed in acc_s[0].
        // Schedule S (default): QK_b(jj) | softmax_a | PV_a(jj) | init_a, QK_a(jj+1) | wait<1> (QK_b, PV_a retired) | softmax_b |
        //   PV_b(jj) | init_b | wait<0>.  Both rows' QK GEMMs run under the other row's softmax; P_a is retired before softmax_b so
        //   the live set there is S_a(jj+1) + S_b/P_b + O.
        // Schedule L (kFlags & 4): QK_a(jj+1) is issued after softmax_b instead (init_a, QK_a, init_b at the end): smaller live set,
        //   QK_a's latency only covered by init_b.
        constexpr bool kSchedL = (T::kFlags & 4) != 0;
        for (int jj = 0; jj < n_active; ++jj) {
            int const j = shared.active[jj];
            bool const has_next = jj + 1 < n_active;
            // A1: row b's QK in flight during row a's softmax
            cur_it = jj; stamp(jj, 0);
            pipe_kv.wait_full(kv_stage(jj, 1), kv_phase(jj));
            stamp(jj, 1);
            issue_qk(acc_s[1], 1, kv_stage(jj, 1));                 // group: QK_b(jj)
            stamp(jj, 2);
            // A3
            softmax_step(acc_s[0], tOrP_a, 0, j);
            stamp(jj, 3);
            issue_pv(tOrP_a, 0, kv_stage(jj, 0));                   // group: PV_a(jj)
            stamp(jj, 4);
            if constexpr (T::kPingPong) { warp_scheduler_barrier_arrive(); }
            if constexpr (!kSchedL) {
                // A4: next tile's row-a QK in flight during row b's softmax (acc_s[0] is free: S_a(jj) became P_a)
                if (has_next) {
                    pipe_b.wait_full(b_slot(jj + 1), b_phase(jj + 1));
                    stamp(jj, 5);
                    init_acc(acc_s[0], b_slot(jj + 1));
                    stamp(jj, 6);
                    pipe_kv.wait_full(kv_stage(jj + 1, 0), kv_phase(jj + 1));
                    stamp(jj, 7);
                    issue_qk(acc_s[0], 0, kv_stage(jj + 1, 0));    // group: QK_a(jj+1)
                } else {
                    warpgroup_commit_batch();                       // empty group in place of QK_a(jj+1)
                }
            }
            stamp(jj, 8);
            // B2: S_b(jj) landed and PV_a(jj) retired (pending: QK_a(jj+1)|empty under S, nothing else under L)
            warpgroup_wait<kSchedL ? 0 : 1>();
            warpgroup_fence_operand(acc_s[1]); warpgroup_fence_operand(acc_o[0]);      // S_b readable; O_a settled (P_a's registers are free)
            stamp(jj, 9);
            // B3
            softmax_step(acc_s[1], tOrP_b, 1, j);
            stamp(jj, 10);
            issue_pv(tOrP_b, 1, kv_stage(jj, 1));                   // group: PV_b(jj)
            stamp(jj, 11);
            if constexpr (T::kPingPong) { warp_scheduler_barrier_arrive(); }
            // B4: next tile's S accumulators <- bias/scale (overlaps PV_b); the bias stage has then been read by both rows
            if (has_next) {
                if constexpr (kSchedL) {
                    pipe_b.wait_full(b_slot(jj + 1), b_phase(jj + 1));
                    init_acc(acc_s[0], b_slot(jj + 1));
                    pipe_kv.wait_full(kv_stage(jj + 1, 0), kv_phase(jj + 1));
                    issue_qk(acc_s[0], 0, kv_stage(jj + 1, 0));    // group: QK_a(jj+1)
                }
                init_acc(acc_s[1], b_slot(jj + 1));
                __syncwarp();
                pipe_b.release(b_slot(jj + 1), lane == 0, b_ranks, self_rank);
            }
            stamp(jj, 12);
            // END: retire everything (QK_a(jj+1) has long landed; PV_b(jj)'s tail is what is waited for); tile jj's KV stages are consumed
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_s[0]); warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]);
            stamp(jj, 13);
            pipe_kv.release(kv_stage(jj, 0), wg_leader, kv_ranks, self_rank);
            pipe_kv.release(kv_stage(jj, 1), wg_leader, kv_ranks, self_rank);
        }
    }

    }   // !kStream

    // ---- epilogue: O / l -> bf16 -> global ---------------------------------------------------------------------------
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        int const i = i0 + r;
        if (i >= params.N || qtile >= params.n_qtiles) { continue; }
        Tensor o_rc = make_tensor(acc_o[r].data(), flash::convert_layout_acc_rowcol(acc_o[r].layout()));
        Element* obase = params.out + (int64_t)b * params.so_b + (int64_t)i * params.so_n + (int64_t)h * params.so_h;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            float const l = o_rc(mi, kColL);
            float const inv = l > 0.f ? 1.f / l : 0.f;
            int const q = qtile * kBlockM + get<0>(tOcO_rc(mi, _0{}));
            if (q < S) {
                Element* orow = obase + (int64_t)q * params.so_s;
                #pragma unroll
                for (int ni = 0; ni < kColL; ni += 2) {
                    int const d = get<1>(tOcO_rc(mi, ni));
                    __nv_bfloat162 v2 = __floats2bfloat162_rn(o_rc(mi, ni) * inv, o_rc(mi, ni + 1) * inv);
                    *reinterpret_cast<__nv_bfloat162*>(orow + d) = v2;
                }
            }
        }
    }
    // a CTA's shared memory (pipeline barriers) must outlive the remote arrivals of its cluster peers
    if constexpr (CQ * CR > 1) { cute::cluster_arrive_relaxed(); cute::cluster_wait(); }
}

}  // namespace triattn_b
