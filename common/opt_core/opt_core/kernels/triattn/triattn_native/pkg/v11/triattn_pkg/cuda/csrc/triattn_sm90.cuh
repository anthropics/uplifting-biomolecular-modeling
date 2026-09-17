// Triangle attention forward (inference) for sm_90a: warp-specialized (1 TMA producer warp + 2 consumer warpgroups in
// ping-pong), K/V tiles TMA-multicast over a cluster of CTAs that share the same pair rows, the [S,S] pair bias tile of a
// k-block held in registers and reused by the R pair rows a CTA owns, row sums on the tensor core (P x ones).
//
//   out[b,i,h,q,:] = softmax_k( scale * q[b,i,h,q,:].k[b,i,h,k,:] + bias[b,h,q,k]  (masked keys excluded) ) @ v[b,i,h,k,:]
//
// One CTA = (batch b, head h, q-tile of kBlockM=128 queries, R consecutive pair rows i0..i0+R-1). Grid x = q-tiles
// (cluster dim), y = row groups, z = b*H + h. See DESIGN.md next to this file for the budget arithmetic.
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

namespace triattn {

using namespace cute;

enum TileClass : uint8_t { kClean = 0, kMixed = 1, kSkip = 2 };

// ---------------------------------------------------------------------------------------------------------------------
// CTA-local TMA pipeline (usable inside a cluster launch, unlike PipelineTmaAsync built with a 1x1x1 cluster shape, whose
// arrivals target cluster rank 0): full[s] = transaction barrier (1 producer arrive + TMA bytes), empty[s] = NumConsumerWG
// local arrivals (one elected thread per consumer warpgroup, after that warpgroup has finished reading the stage).
// ---------------------------------------------------------------------------------------------------------------------
template <int Stages, int NumConsumerWG>
struct LocalPipe {
    using State = cutlass::PipelineState<Stages>;
    struct SharedStorage {
        cutlass::arch::ClusterTransactionBarrier full[Stages];
        cutlass::arch::ClusterBarrier empty[Stages];
    };
    SharedStorage& st;
    uint32_t tx_bytes;
    CUTLASS_DEVICE LocalPipe(SharedStorage& s, uint32_t tx) : st(s), tx_bytes(tx) {}
    CUTLASS_DEVICE static void init(SharedStorage& s, int extra_release = 0) {      // empty arrivals per phase: NumConsumerWG + extra_release
        for (int i = 0; i < Stages; ++i) { s.full[i].init(1); s.empty[i].init(NumConsumerWG + extra_release); }
    }
    CUTLASS_DEVICE void producer_acquire(State const& state) {
        st.empty[state.index()].wait(state.phase());
        st.full[state.index()].arrive_and_expect_tx(tx_bytes);
    }
    CUTLASS_DEVICE uint64_t* producer_get_barrier(State const& state) { return reinterpret_cast<uint64_t*>(&st.full[state.index()]); }
    CUTLASS_DEVICE void producer_tail(State state) {
        for (int i = 0; i < Stages; ++i) { st.empty[state.index()].wait(state.phase()); ++state; }
    }
    CUTLASS_DEVICE void consumer_wait(State const& state) { st.full[state.index()].wait(state.phase()); }
    CUTLASS_DEVICE void consumer_release(State const& state, bool elected) { if (elected) { st.empty[state.index()].arrive(); } }
    CUTLASS_DEVICE static State producer_start() { return State{0, 1, 0}; }
    // explicit form: use number u of a stage has phase = u & 1
    CUTLASS_DEVICE void producer_wait_empty(int stage, uint32_t phase) { st.empty[stage].wait(phase ^ 1); }
    CUTLASS_DEVICE void producer_expect(int stage) { st.full[stage].arrive_and_expect_tx(tx_bytes); }
    CUTLASS_DEVICE void producer_skip(int stage) { st.full[stage].arrive(); }             // completes the phase with no data
    CUTLASS_DEVICE uint64_t* full_barrier(int stage) { return reinterpret_cast<uint64_t*>(&st.full[stage]); }
    CUTLASS_DEVICE void wait_full(int stage, uint32_t phase) { st.full[stage].wait(phase); }
    CUTLASS_DEVICE void spin_full(int stage, uint32_t phase) {                             // polling wait (no thread suspension)
        uint32_t const addr = cute::cast_smem_ptr_to_uint(reinterpret_cast<uint64_t const*>(&st.full[stage]));
        uint32_t done = 0;
        do {
            asm volatile("{\n .reg .pred P1;\n mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n selp.u32 %0, 1, 0, P1;\n}" : "=r"(done) : "r"(addr), "r"(phase) : "memory");
        } while (!done);
    }
    CUTLASS_DEVICE void release(int stage, bool elected) { if (elected) { st.empty[stage].arrive(); } }
};

// Cluster-wide K/V pipeline: cutlass::PipelineTmaAsync (multicast-aware empty-barrier arrivals) when ClusterQ > 1.
template <int Stages, int NumConsumerWG, int ClusterQ>
struct ClusterPipe {
    using Impl = cutlass::PipelineTmaAsync<Stages>;
    using State = cutlass::PipelineState<Stages>;
    using SharedStorage = typename Impl::SharedStorage;
    Impl impl;
    CUTLASS_DEVICE static typename Impl::Params make_params(uint32_t tx, bool producer_wg, bool leader) {
        typename Impl::Params p;
        p.transaction_bytes = tx;
        p.role = producer_wg ? Impl::ThreadCategory::Producer : Impl::ThreadCategory::Consumer;
        p.is_leader = leader;
        p.num_consumers = NumConsumerWG * cutlass::NumThreadsPerWarpGroup;
        p.num_producers = 1;
        return p;
    }
    CUTLASS_DEVICE ClusterPipe(SharedStorage& s, uint32_t tx, bool producer_wg, bool leader)
        : impl(s, make_params(tx, producer_wg, leader), Shape<Int<ClusterQ>, _1, _1>{}) {}
    CUTLASS_DEVICE void producer_acquire(State const& state) { impl.producer_acquire(state); }
    CUTLASS_DEVICE uint64_t* producer_get_barrier(State const& state) { return reinterpret_cast<uint64_t*>(impl.producer_get_barrier(state)); }
    CUTLASS_DEVICE void producer_tail(State state) { impl.producer_tail(state); }
    CUTLASS_DEVICE void consumer_wait(State const& state) { impl.consumer_wait(state); }
    CUTLASS_DEVICE void consumer_release(State const& state, bool) { impl.consumer_release(state); }
    CUTLASS_DEVICE static State producer_start() { return State{0, 1, 0}; }
};

// ---------------------------------------------------------------------------------------------------------------------
template <int kHeadDim_, int kBlockN_, int kRows_, int kClusterQ_, int kStages_, typename Element_, bool kPingPong_ = false, int kPolyEvery_ = 0, int kDebug_ = 0, int kNumMmaWG_ = 2, int kRowsPerStep_ = (kNumMmaWG_ == 3 ? 1 : 2)>
struct Traits {
    using Element = Element_;
    static constexpr int kDebug = kDebug_;                     // bit 1: rescale unconditionally (no branch); bit 8: clock64 trace stamps; bit 2/32: drop mask / rescale code (diagnosis only: WRONG results)
    static constexpr bool kPingPong = kPingPong_;
    static constexpr int kPolyEvery = kPolyEvery_;             // 0: all exps on MUFU; n: every n-th column's exp via a degree-3 polynomial on the FMA pipe
    static constexpr int kHeadDim = kHeadDim_, kBlockN = kBlockN_, R = kRows_, ClusterQ = kClusterQ_, kStages = kStages_;
    static constexpr int kNumMmaWG = kNumMmaWG_;               // consumer warpgroups (64 query rows each); + 1 producer warpgroup
    static_assert(kNumMmaWG == 2 || kNumMmaWG == 3);
    static constexpr bool kSingleRow = kRowsPerStep_ == 1;      // one row per step (S/P of one row in registers) or row pairs
    static_assert(kRowsPerStep_ == 1 || kRowsPerStep_ == 2);
    static_assert(!(kNumMmaWG == 3 && kRowsPerStep_ == 2), "pair steps do not fit the 160-register budget of 3 consumer warpgroups");
    static constexpr int kBlockMwg = 64;
    static constexpr int kBlockM = kBlockMwg * kNumMmaWG;
    static constexpr int kNumMmaThreads = kNumMmaWG * 128;
    static constexpr int kNumThreads = kNumMmaThreads + 128;
    // register split after setmaxnreg: launch_bounds(kNumThreads, 1) gives 65536/kNumThreads at entry; the producer keeps kRegsProducer
    // The CTA's register allocation is fixed at launch by __launch_bounds__(kNumThreads, 1) = 168/thread (64512 for 384 threads);
    // setmaxnreg only redistributes it: 128*24 + 256*240 = 64512 exactly (a larger producer share makes the consumers' inc block forever).
    static constexpr int kRegsProducer = kNumMmaWG == 2 ? 24 : 32;
    static constexpr int kRegsConsumer = kNumMmaWG == 2 ? 240 : 160;   // 3 WGs: (168*512 - 128*32) / 384 = 213 -> 208 usable; 160 kept
    // MAX-FREE exponent: the running max m of a row is seeded exactly from its first live k-tile and then frozen; a later row-step
    // takes the exact (lazy running max) path only when the bound  scale*sqrt(D)*|q_row|*max|k_tile| + max|bias_tile| - m  says an
    // exponent could exceed 2^100. The per-stage max|k| and per-bias-stage max|bias| come from the 3 otherwise idle producer warps.
    static constexpr bool kMaxFree = (kDebug_ & 0x30000) != 0;
    static constexpr bool kMaxFreeNoBound = (kDebug_ & 0x20000) != 0;   // diagnosis only: no helper warps, no bound (unsafe for logit swings > 69)
    static constexpr int kHelperWarps = (kMaxFree && !kMaxFreeNoBound) ? 3 : 0;
    static_assert(128 * kRegsProducer + kNumMmaThreads * kRegsConsumer <= 65536);
    static constexpr int kStagesBias = (kStages_ / kRows_) > 3 ? 3 : (kStages_ / kRows_);   // bias ring (= kRingB); the K/V ring may be deeper
    static constexpr int kMaxS = 4096;                       // capacity of the per-row mask bit table in smem (S <= kMaxS)
    static_assert((kBlockN == 32 || kBlockN == 64) && kBlockN % ClusterQ == 0 && (kBlockN / ClusterQ) % 8 == 0);
    static_assert(kStages % R == 0 && kStages / R >= 2, "k-tile j uses KV stages [R*(j%ring), +R) and bias stage j%ring, ring = kStages/R tiles in flight");
    static constexpr int kRing = kStages / R;                   // K/V ring: k-tiles in flight
    static constexpr int kRingB = kStagesBias;                  // bias ring

    using ClusterShape = Shape<Int<ClusterQ>, _1, _1>;
    static constexpr int kHeadDimV = kHeadDim + 8;              // PV runs on [V | 1]: the 8 extra accumulator columns hold the row sum l
    using TileShapeQK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShapePV = Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
    using AtomLayout = Layout<Shape<Int<kNumMmaWG>, _1, _1>>;
    using TiledMmaQK = decltype(make_tiled_mma(GMMA::ss_op_selector<Element, Element, float, TileShapeQK>(), AtomLayout{}));
    // chunk-streaming consumer (kStream): S is produced in chunks of kChunkW keys; the bias tile is staged in MMA-fragment order
    static constexpr bool kStream = (kDebug_ & 0x1000000) != 0;
    static constexpr bool kStream64 = (kDebug_ & 0x8000000) != 0;  // stream consumer with 64-key chunks (one chunk per (tile, row)), 3 S buffers, QK one chunk ahead, period 6
    static constexpr bool kBiasReload = (kDebug_ & 0x20000000) != 0;   // stream: bias half re-loaded in every body (8 live bias registers instead of 16; for the 160-register 3-WG budget)
    static constexpr bool kBiasF32 = (kDebug_ & 0x10000000) != 0;  // stream: bias staged as fp32 bias/scale in ACCUMULATOR order; 4 LDS.128 load it straight into S(k+2) at the top of body k
    static constexpr bool kInitAcc = kBiasF32 || (kDebug_ & 0x4000000) != 0;   // stream: S accumulator starts as bias/scale (QK accumulates onto it); E = 1 FFMA + MUFU per logit
    static constexpr int kChunkW = 32;
    using TileShapeQKC = Shape<Int<kBlockM>, Int<kChunkW>, Int<kHeadDim>>;
    using TiledMmaQK64 = decltype(make_tiled_mma(GMMA::ss_op_selector<Element, Element, float, Shape<Int<kBlockM>, Int<64>, Int<kHeadDim>>>(), AtomLayout{}));
    using TiledMmaQKC = decltype(make_tiled_mma(GMMA::ss_op_selector<Element, Element, float, TileShapeQKC>(), AtomLayout{}));
    static_assert(kHeadDim == 32 && std::is_same_v<Element, cutlass::bfloat16_t>, "PV atom picked by hand for D=32 bf16");
    using TiledMmaPV = decltype(make_tiled_mma(SM90_64x40x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, AtomLayout{}));
    // P staged through shared memory (stmatrix) and read by an SS wgmma: no register-sourced (RS) wgmma issue in the hot loop
    static constexpr bool kPss = (kDebug_ & 0x100000) != 0;
    using TiledMmaPVss = decltype(make_tiled_mma(SM90_64x40x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{}, AtomLayout{}));
    using TiledMmaPV32 = decltype(make_tiled_mma(GMMA::rs_op_selector<Element, Element, float, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>, GMMA::Major::K, GMMA::Major::MN>(), AtomLayout{}));   // F4 microkernel only

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockM>, Int<kHeadDim>>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<R>{})));          // (M, D, R)
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kStages>{})));     // (BN, D, st)
    // V^T stage = (D+8, BN) MN-major, unswizzled 8x8 core matrices; rows 0..D-1 are the TMA destination (one contiguous D*BN block per
    // stage because BN is tiled first), rows D..D+7 hold 1.0 (written once at kernel start, never touched by TMA)
    using SmemLayoutAtomVt = GMMA::Layout_MN_INTER_Atom<Element>;
    using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomVt{}, make_shape(Int<kHeadDimV>{}, Int<kBlockN>{}, Int<kStages>{}), Step<_2, _1, _3>{}));  // (D+8, BN, st)
    using SmemLayoutVload1 = decltype(tile_to_shape(SmemLayoutAtomVt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), Step<_2, _1>{}));                    // (D, BN)
    static constexpr int kStageElemsV = kHeadDimV * kBlockN;
    static_assert(size(SmemLayoutVload1{}) == kHeadDim * kBlockN && cosize(SmemLayoutVload1{}) == kHeadDim * kBlockN && cosize(SmemLayoutVt{}) == kStageElemsV * kStages);
    using SmemLayoutP = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<(kRowsPerStep_ == 1 ? 1 : 2)>{})));   // (M, BN, buf)
    using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockM>, Int<kBlockN>>());
    using SmemLayoutBias = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kStagesBias>{})));  // (M, BN, st)

    using GmemTiledCopyKV = std::conditional_t<(ClusterQ > 1), SM90_TMA_LOAD_MULTICAST, SM90_TMA_LOAD>;
    using StrideQK = Stride<int64_t, _1, int64_t, int64_t, int64_t>;      // (S, D, H, N, B)
    using StrideV  = Stride<_1, int64_t, int64_t, int64_t, int64_t>;      // (D, S, H, N, B)
    using StrideB  = Stride<int64_t, _1, int64_t, int64_t>;               // (Sq, Sk, H, B)
    using ShapeQK  = Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;
    using ShapeB   = Shape<int32_t, int32_t, int32_t, int32_t>;

    using TMA_Q = decltype(make_tma_copy(SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQK{}, StrideQK{}),
                                        take<0, 2>(SmemLayoutQ{}), make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), _1{}));
    using TMA_K = decltype(make_tma_copy(GmemTiledCopyKV{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQK{}, StrideQK{}),
                                        take<0, 2>(SmemLayoutK{}), make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), Int<ClusterQ>{}));
    using TMA_V = decltype(make_tma_copy(GmemTiledCopyKV{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQK{}, StrideV{}),
                                        SmemLayoutVload1{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), Int<ClusterQ>{}));
    using TMA_B = decltype(make_tma_copy(SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeB{}, StrideB{}),
                                        take<0, 2>(SmemLayoutBias{}), make_shape(Int<kBlockM>{}, Int<kBlockN>{}), _1{}));

    static constexpr uint32_t kBytesQ = kBlockM * kHeadDim * sizeof(Element);          // per row r
    static constexpr uint32_t kBytesKV = 2u * kBlockN * kHeadDim * sizeof(Element);    // K + V of one stage
    static constexpr uint32_t kBytesBias = kBlockM * kBlockN * (kBiasF32 ? 4 : sizeof(Element));   // one bias stage

    using PipeKV = std::conditional_t<(ClusterQ > 1), ClusterPipe<kStages, kNumMmaWG, ClusterQ>, LocalPipe<kStages, kNumMmaWG>>;
    using PipeBias = LocalPipe<kStagesBias, kNumMmaWG * 4>;     // released per consumer warp

    struct SharedStorage {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, 1024> smem_q;
        cute::array_aligned<Element, kPss ? cute::cosize_v<SmemLayoutP> : 64, 1024> smem_p;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, 1024> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, 1024> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutBias> * (kBiasF32 ? 2 : 1), 1024> smem_bias;
        uint32_t mask_bits[R][kMaxS / 32];
        uint8_t tile_class[R][kMaxS / kBlockN];
        uint16_t live[kMaxS / kBlockN];                        // the k-tiles that are computed (tile 0 always; others unless all R rows are fully masked there), ascending
        int n_live;
        int uniform[R];                                        // 1 = row r has no attendable key
        float kmax_part[kMaxFree ? kStages : 1][3];            // (max-free) max|k| over the K tile in stage st, one partial per helper warp
        float bmax_part[kMaxFree ? kStagesBias : 1][3];        // (max-free) max|bias| over the bias tile in slot, per helper warp
        cutlass::arch::ClusterBarrier bar_bound[kMaxFree ? kStages : 1];   // (max-free) kmax_part[st] (and bmax_part of its slot) are written
        typename PipeKV::SharedStorage pipe_kv;
        typename PipeBias::SharedStorage pipe_bias;
        cutlass::arch::ClusterTransactionBarrier barrier_q;
    };

    struct Params {
        TMA_Q tma_q; TMA_K tma_k; TMA_V tma_v; TMA_B tma_b;
        ShapeQK shape_qk; ShapeB shape_b;
        Element* out; int64_t so_b, so_n, so_h, so_s;                 // out [B,N,H,S,D] element strides (d stride 1)
        uint8_t const* mask; int64_t sm_b, sm_n, sm_s;                // mask [B,N,S] (byte, nonzero = keep); nullptr = none
        int S, N, H;
        int n_qtiles, n_ktiles;
        float scale_log2e;
        unsigned long long* trace;                                     // debug: per-step clock64 stamps of 2 threads of one CTA (nullptr = off)
        unsigned long long* counters;                                  // [0] += row-steps that took the exact-max path after the seed (nullptr = off)
        char const* bias_frag;                                         // (kStream) bias staged in fragment order: [B*H][n_qtiles][n_ktiles] blocks of kBytesBias
        int zero;                                                      // always 0 (an opaque zero for per-period operand bases)
    };
};


template <class F, int... Is>
CUTLASS_DEVICE void static_for_impl(F&& f, std::integer_sequence<int, Is...>) { (f(cute::Int<Is>{}), ...); }
template <int N, class F>
CUTLASS_DEVICE void static_for(F&& f) { static_for_impl(static_cast<F&&>(f), std::make_integer_sequence<int, N>{}); }

CUTLASS_DEVICE int warp_group_idx_nosync() { return threadIdx.x / cutlass::NumThreadsPerWarpGroup; }

// Token ring over the consumer warpgroups (MUFU section): consumer warpgroup c waits on named barrier FirstUser+c (256 threads = its own
// 128 + the 128 of the warpgroup passing the token) and passes the token by arriving on FirstUser+(c+1)%n.
template <class T>
CUTLASS_DEVICE void warp_scheduler_barrier_sync() {
    int const cur = warp_group_idx_nosync() - 1;
    cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, uint32_t(cur) + static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier));
}
template <class T>
CUTLASS_DEVICE void warp_scheduler_barrier_arrive() {
    int const cur = warp_group_idx_nosync() - 1;
    int const nxt = cur + 1 == T::kNumMmaWG ? 0 : cur + 1;
    cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, uint32_t(nxt) + static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier));
}

__device__ __forceinline__ float ex2_approx(float x) { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// 1-D bulk async copy global -> shared, completion signalled on an mbarrier (transaction bytes)
__device__ __forceinline__ void bulk_g2s(uint32_t dst_smem, void const* src, uint32_t bytes, uint32_t mbar_smem) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst_smem), "l"(src), "r"(bytes), "r"(mbar_smem) : "memory");
}

// 2^x on the FMA pipe: Cody-Waite split + degree-3 minimax on [-0.5, 0.5] (rel err 8e-4 < bf16 rounding); x <= 0 expected, clamped at -126
__device__ __forceinline__ float ex2_poly3(float x) {
    x = fmaxf(x, -126.f);
    float const t = x + 12582912.f;                  // 1.5 * 2^23: round to nearest integer in the low mantissa bits
    float const f = x - (t - 12582912.f);
    float p = fmaf(0.0555054f, f, 0.2402265f);
    p = fmaf(p, f, 0.6931472f);
    p = fmaf(p, f, 1.0f);
    return __int_as_float(__float_as_int(p) + (__float_as_int(t) << 23));
}

// ---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void __launch_bounds__(T::kNumThreads, 1) triattn_fwd_kernel(CUTE_GRID_CONSTANT typename T::Params const params) {
    using Element = typename T::Element;
    constexpr int R = T::R, kBlockM = T::kBlockM, kBlockN = T::kBlockN, kHeadDim = T::kHeadDim, ClusterQ = T::ClusterQ;
    using SharedStorage = typename T::SharedStorage;
    extern __shared__ char smem_buf[];
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const lane_predicate = cute::elect_one_sync();
    int const wg_idx = cutlass::canonical_warp_group_idx();
    int const tid = threadIdx.x;

    // ---- block coordinates -------------------------------------------------------------------------------------------
    int const qtile = blockIdx.x;                    // may be >= n_qtiles when padded to the cluster size: such CTAs still run
    int const rgroup = blockIdx.y;                   // the full protocol (they multicast K/V slices) but store nothing.
    int const bh = blockIdx.z;
    int const b = bh / params.H, h = bh % params.H;
    int const i0 = rgroup * R;
    int const S = params.S;
    int const n_ktiles = params.n_ktiles;

    if (warp_idx == 0 && lane_predicate) {
        cute::prefetch_tma_descriptor(params.tma_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_v.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_b.get_tma_descriptor());
        shared.barrier_q.init(1);
        T::PipeBias::init(shared.pipe_bias, T::kHelperWarps);          // helper warps also release the stages they scan
        if constexpr (ClusterQ == 1) { T::PipeKV::init(shared.pipe_kv, T::kHelperWarps); }
        if constexpr (T::kMaxFree && !T::kMaxFreeNoBound) { for (int i = 0; i < T::kStages; ++i) { shared.bar_bound[i].init(T::kHelperWarps); } }
    }
    // Cluster KV pipeline constructs (and inits) its own barriers; must be constructed by all threads.
    typename T::PipeKV pipe_kv = [&] {
        if constexpr (ClusterQ > 1) {
            return typename T::PipeKV(shared.pipe_kv, T::kBytesKV, wg_idx == 0, (tid % 128) == 0);
        } else {
            return typename T::PipeKV(shared.pipe_kv, T::kBytesKV);
        }
    }();
    typename T::PipeBias pipe_bias(shared.pipe_bias, T::kBytesBias);

    // ---- (consumer warpgroups, before their main loop) ones rows of [V|1], mask bit rows, k-tile classes, live-tile list -----
    // Runs while the producer already loads Q and k-tile 0; the producer waits for the live list (barrier kBarLive) before tile 1.
    constexpr uint32_t kBarMask = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier) + 4;   // ids +0..+2: token ring
    constexpr uint32_t kBarLive = kBarMask + 1;
    auto consumer_prologue = [&]() {
        int const ctid = tid - 128;                          // 0..kNumMmaThreads-1
        int const lane = tid % 32;
        constexpr int NT = T::kNumMmaThreads;
        // rows D..D+7 of every V^T stage = 1.0: the PV GEMM then yields the softmax row sums in accumulator columns D..D+7.
        // In the (D+8, BN) MN-major INTER stage layout those 8 rows are the last contiguous 8*BN*2-byte band of the stage.
        {
            static_assert(sizeof(Element) == 2);
            constexpr int kBandBytes = 8 * kBlockN * 2, kVecPerStage = kBandBytes / 16, kStageBytes = T::kStageElemsV * 2;
            uint32_t const one2 = (uint32_t(0x3f80) << 16) | 0x3f80u;             // bf16 1.0 pair  (fp16 would be 0x3c00)
            uint4 const ones4 = make_uint4(one2, one2, one2, one2);
            char* vbase = reinterpret_cast<char*>(shared.smem_v.data());
            for (int idx = ctid; idx < kVecPerStage * T::kStages; idx += NT) {
                int const st = idx / kVecPerStage, c = idx % kVecPerStage;
                *reinterpret_cast<uint4*>(vbase + st * kStageBytes + T::kHeadDim * kBlockN * 2 + 16 * c) = ones4;
            }
            {   // layout self-check (once): element (D, 0, 0) must be the first element of the band
                Tensor sVt = make_tensor(make_smem_ptr(shared.smem_v.data()), typename T::SmemLayoutVt{});
                constexpr int kD = T::kHeadDim;
                if (ctid == 0 && reinterpret_cast<char*>(&sVt(kD + 0, 0, 0)) != vbase + kD * kBlockN * 2) { asm volatile("trap;"); }
                if (ctid == 0 && reinterpret_cast<char*>(&sVt(kD + 7, kBlockN - 1, 1)) != vbase + kStageBytes + kD * kBlockN * 2 + kBandBytes - 2) { asm volatile("trap;"); }
            }
            cutlass::arch::fence_view_async_shared();            // generic-proxy smem writes -> visible to the tensor core (async proxy)
        }
        int const n_words = (n_ktiles * kBlockN) / 32;
        if (ctid < R) { shared.uniform[ctid] = 0; }
        int cnt[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) { cnt[r] = 0; }
        bool const vec16 = params.mask != nullptr && params.sm_s == 1 && (S % 16) == 0 && (params.sm_n % 16) == 0 && (params.sm_b % 16) == 0
                           && (reinterpret_cast<uintptr_t>(params.mask) % 16) == 0;
        if (params.mask == nullptr) {
            // no mask: every key < S attends; only the ragged last tile (if any) is not all-keep and needs its bit words
            int const tail_words = n_words - (S / kBlockN) * (kBlockN / 32);   // words of the partial tile (0 if S % kBlockN == 0)
            for (int w = ctid; w < R * tail_words; w += NT) {
                int const r = w / tail_words, wi = (S / kBlockN) * (kBlockN / 32) + w % tail_words, k0 = wi * 32;
                shared.mask_bits[r][wi] = k0 + 32 <= S ? 0xffffffffu : (k0 >= S ? 0u : ((1u << (S - k0)) - 1u));
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) { cnt[r] = (ctid == 0) ? S : 0; }
            cutlass::arch::NamedBarrier::sync(NT, kBarMask);     // (uniform[] init visible before the atomics below)
        } else if (vec16) {
            // 16 mask bytes per load, all of a thread's loads in flight together; words covering keys >= S are zeroed first
            for (int w = ctid; w < R * n_words; w += NT) { int const wi = w % n_words; if (wi * 32 + 32 > S) { shared.mask_bits[w / n_words][wi] = 0u; } }
            cutlass::arch::NamedBarrier::sync(NT, kBarMask);
            int const n16 = S / 16, total = R * n16;
            constexpr int kUnroll = 4;
            for (int base = ctid; base < total; base += NT * kUnroll) {
                uint4 v[kUnroll];
                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    int const idx = base + u * NT;
                    if (idx < total) {
                        int const r = idx / n16, c = idx % n16;
                        int const i = min(i0 + r, params.N - 1);
                        v[u] = *reinterpret_cast<uint4 const*>(params.mask + (int64_t)b * params.sm_b + (int64_t)i * params.sm_n + 16 * c);
                    }
                }
                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    int const idx = base + u * NT;
                    if (idx < total) {
                        int const r = idx / n16, c = idx % n16;
                        uint32_t bits = 0;
                        uint32_t const wv[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
                        #pragma unroll
                        for (int q4 = 0; q4 < 4; ++q4) {
                            #pragma unroll
                            for (int by = 0; by < 4; ++by) { bits |= (((wv[q4] >> (8 * by)) & 0xffu) != 0u ? 1u : 0u) << (4 * q4 + by); }
                        }
                        reinterpret_cast<uint16_t*>(&shared.mask_bits[r][0])[c] = uint16_t(bits);
                        #pragma unroll
                        for (int rr = 0; rr < R; ++rr) { if (rr == r) cnt[rr] += __popc(bits); }
                    }
                }
            }
        } else {
            // generic strides / odd S: one key per thread, ballot per 32 keys (writes every word, incl. keys >= S as 0)
            cutlass::arch::NamedBarrier::sync(NT, kBarMask);
            int const cw = ctid / 32, n_cwarps = NT / 32;
            for (int r = 0; r < R; ++r) {
                int const i = min(i0 + r, params.N - 1);
                uint8_t const* mrow = params.mask + (int64_t)b * params.sm_b + (int64_t)i * params.sm_n;
                for (int w = cw; w < n_words; w += n_cwarps) {
                    int const k = w * 32 + lane;
                    bool const keep = k < S && mrow[(int64_t)min(k, S - 1) * params.sm_s] != 0;
                    uint32_t const word = __ballot_sync(0xffffffffu, keep);
                    if (lane == 0) { shared.mask_bits[r][w] = word; }
                    #pragma unroll
                    for (int rr = 0; rr < R; ++rr) { if (rr == r && lane == 0) cnt[rr] += __popc(word); }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            int const c32 = __reduce_add_sync(0xffffffffu, cnt[r]);
            if (lane == 0 && c32) { atomicAdd(&shared.uniform[r], c32); }
        }
        cutlass::arch::NamedBarrier::sync(NT, kBarMask);
        // A row with no attendable key: masked keys carry the additive -1e9 of cuEquivariance's semantics, so such a row attends
        // (uniformly) to all its S keys -> none of its tiles may be skipped and all are 'mixed'.
        constexpr int kWordsPerTile = kBlockN / 32;
        for (int idx = ctid; idx < R * n_ktiles; idx += NT) {
            int const r = idx / n_ktiles, j = idx % n_ktiles;
            uint8_t cls;
            if (params.mask == nullptr) { cls = (j + 1) * kBlockN <= S ? kClean : kMixed; }
            else {
                bool const uniform_row = shared.uniform[r] == 0;  // (count of attendable keys == 0)
                bool all1 = true, all0 = true;
                #pragma unroll
                for (int w = 0; w < kWordsPerTile; ++w) {
                    uint32_t const word = shared.mask_bits[r][j * kWordsPerTile + w];
                    all1 &= (word == 0xffffffffu); all0 &= (word == 0u);
                }
                cls = all1 ? kClean : ((all0 && !uniform_row) ? kSkip : kMixed);
            }
            shared.tile_class[r][j] = cls;
        }
        cutlass::arch::NamedBarrier::sync(NT, kBarMask);
        if (ctid < R) { shared.uniform[ctid] = shared.uniform[ctid] == 0 ? 1 : 0; }     // now: 1 = row has no attendable key
        if (ctid < 32) {                                         // live list: one warp, ballot scan over <= kMaxS/kBlockN tiles
            int n = 0;
            for (int j0 = 0; j0 < n_ktiles; j0 += 32) {
                int const j = j0 + lane;
                bool live = false;
                if (j < n_ktiles) {
                    bool all_skip = true;
                    #pragma unroll
                    for (int r = 0; r < R; ++r) { all_skip &= (shared.tile_class[r][j] == kSkip); }
                    live = (j == 0) || !all_skip;              // tile 0 is always computed (the producer loads it before the list exists)
                }
                uint32_t const bal = __ballot_sync(0xffffffffu, live);
                if (live) { shared.live[n + __popc(bal & ((1u << lane) - 1u))] = uint16_t(j); }
                n += __popc(bal);
            }
            if (lane == 0) { shared.n_live = n; }
        }
    };

    if constexpr (ClusterQ > 1) { cute::cluster_arrive_relaxed(); cute::cluster_wait(); } else { __syncthreads(); }

    if (wg_idx == 0) {
        // =============================================== PRODUCER =====================================================
        cutlass::arch::warpgroup_reg_dealloc<T::kRegsProducer>();
        int const warp_idx_in_wg = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_wg != 0) {
            if constexpr (T::kHelperWarps == 0) { return; }
            else {
                // ---- helper warps 1..3: per live tile, max|bias| of the bias stage and max|k| of each K stage (bf16 magnitude bits
                // compared as 16-bit integers), published per warp in smem; bar_bound[st] tells the consumers, and the warps count as
                // extra releasers of every stage they read (empty barriers were initialised with +kHelperWarps).
                int const lane = tid % 32, hw = warp_idx_in_wg - 1, ht = hw * 32 + lane;          // helper thread 0..95
                cutlass::arch::NamedBarrier::sync(T::kNumThreads, kBarLive);                       // live list ready
                int const n_live = shared.n_live;
                uint32_t const kbase = cute::cast_smem_ptr_to_uint(shared.smem_k.data());
                uint32_t const bbase = cute::cast_smem_ptr_to_uint(shared.smem_bias.data());
                constexpr int kKStageBytes = kBlockN * T::kHeadDim * 2, kBStageBytes = int(T::kBytesBias);
                auto absmax_words = [&](uint32_t base, int nbytes) {                              // max over |bf16| in [base, base+nbytes)
                    uint32_t m2 = 0;
                    for (int off = ht * 16; off < nbytes; off += 96 * 16) {
                        uint32_t w0, w1, w2, w3;
                        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3) : "r"(base + off));
                        m2 = __vmaxu2(m2, __vmaxu2(__vmaxu2(w0 & 0x7fff7fffu, w1 & 0x7fff7fffu), __vmaxu2(w2 & 0x7fff7fffu, w3 & 0x7fff7fffu)));
                    }
                    uint32_t m1 = max(m2 & 0xffffu, m2 >> 16);
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) { m1 = max(m1, __shfl_xor_sync(0xffffffffu, m1, o)); }
                    return __uint_as_float(m1 << 16);                                              // bf16 bits -> fp32 value
                };
                for (int jl = 0; jl < n_live; ++jl) {
                    int const slot = jl % T::kRing; uint32_t const ph = (jl / T::kRing) & 1;
                    int const bslot = jl % T::kRingB; uint32_t const bph = (jl / T::kRingB) & 1;
                    pipe_bias.wait_full(bslot, bph);
                    float const bm = absmax_words(bbase + bslot * kBStageBytes, kBStageBytes);
                    if (lane == 0) { shared.bmax_part[bslot][hw] = bm; }
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        int const st = R * slot + r;
                        pipe_kv.wait_full(st, ph);
                        float const km = absmax_words(kbase + st * kKStageBytes, kKStageBytes);
                        if (lane == 0) { shared.kmax_part[st][hw] = km; }
                        __syncwarp(); __threadfence_block();
                        if (lane == 0) { shared.bar_bound[st].arrive(); pipe_kv.release(st, true); }
                    }
                    if (lane == 0) { pipe_bias.release(bslot, true); }
                }
                return;
            }
        }

        uint32_t block_rank_in_cluster = ClusterQ > 1 ? cute::block_rank_in_cluster() : 0;
        uint16_t mcast_mask = 0;
        if constexpr (ClusterQ > 1) { for (int c = 0; c < ClusterQ; ++c) { mcast_mask |= (uint16_t(1) << c); } }

        Tensor sQ = make_tensor(make_smem_ptr(shared.smem_q.data()), typename T::SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared.smem_k.data()), typename T::SmemLayoutK{});
        auto sV_stage = [&](int st) { return make_tensor(make_smem_ptr(shared.smem_v.data() + st * T::kStageElemsV), typename T::SmemLayoutVload1{}); };   // rows 0..D-1 of stage st
        Tensor sB = make_tensor(make_smem_ptr(shared.smem_bias.data()), typename T::SmemLayoutBias{});

        Tensor mQ = params.tma_q.get_tma_tensor(params.shape_qk)(_, _, h, _, b);                 // (S, D, N)
        Tensor mK = params.tma_k.get_tma_tensor(params.shape_qk)(_, _, h, _, b);                 // (S, D, N)
        auto shape_v = make_shape(get<1>(params.shape_qk), get<0>(params.shape_qk), get<2>(params.shape_qk), get<3>(params.shape_qk), get<4>(params.shape_qk));
        Tensor mVt = params.tma_v.get_tma_tensor(shape_v)(_, _, h, _, b);                        // (D, S, N)
        Tensor mB = params.tma_b.get_tma_tensor(params.shape_b)(_, _, h, b);                     // (Sq, Sk)

        Tensor gQ = local_tile(mQ, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(qtile, _0{}, _));      // (M, D, N)
        Tensor gK = local_tile(mK, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}, _));          // (BN, D, ktile, N)
        Tensor gVt = local_tile(mVt, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_0{}, _, _));        // (D, BN, ktile, N)
        Tensor gB = local_tile(mB, make_shape(Int<kBlockM>{}, Int<kBlockN>{}), make_coord(qtile, _));               // (M, BN, ktile)

        auto block_tma_q = params.tma_q.get_slice(_0{});
        Tensor tQgQ = group_modes<0, 3>(block_tma_q.partition_S(gQ));      // (TMA, N)
        Tensor tQsQ = group_modes<0, 3>(block_tma_q.partition_D(sQ));      // (TMA, R)
        auto block_tma_k = params.tma_k.get_slice(block_rank_in_cluster);
        Tensor tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));      // (TMA, ktile, N)
        Tensor tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));      // (TMA, st)
        auto block_tma_v = params.tma_v.get_slice(block_rank_in_cluster);
        Tensor tVgV = group_modes<0, 3>(block_tma_v.partition_S(gVt));     // (TMA, ktile, N)
        auto tVsV = [&](int st) { return group_modes<0, 3>(block_tma_v.partition_D(sV_stage(st))); };   // (TMA)
        auto block_tma_b = params.tma_b.get_slice(_0{});
        Tensor tBgB = group_modes<0, 3>(block_tma_b.partition_S(gB));      // (TMA, ktile)
        Tensor tBsB = group_modes<0, 3>(block_tma_b.partition_D(sB));      // (TMA, st)

        // live tile jl (k-tile j = live[jl]) uses bias stage jl%ring and KV stages R*(jl%ring)+r with phase (jl/ring)&1; the
        // producer runs up to ring-1 live tiles ahead of the consumers. Tile 0 is loaded before the live list exists.
        unsigned long long* ptrace = (params.trace != nullptr && blockIdx.x == 1 && blockIdx.y == 37 && blockIdx.z == 1) ? params.trace + 3 * 4096 : nullptr;   // slot of (unused) warp 3
        int ptrace_n = 0;
        auto pstamp = [&](int tag) { if constexpr ((T::kDebug & 8) != 0) { if (ptrace != nullptr && ptrace_n < 4090) { ptrace[ptrace_n++] = (clock64() << 8) | tag; } } };
        auto load_tile = [&](int jl, int j) {
            pstamp(13);
            int const slot = jl % T::kRing; uint32_t const ph = (jl / T::kRing) & 1;
            int const bslot = jl % T::kRingB; uint32_t const bph = (jl / T::kRingB) & 1;
            pipe_bias.producer_wait_empty(bslot, bph);
            pipe_bias.producer_expect(bslot);
            if constexpr (T::kStream || T::kStream64) {      // fragment-ordered bias block of (q-tile, k-tile j): kBytesBias contiguous bytes
                char const* src = params.bias_frag + (((int64_t)(b * params.H + h) * params.n_qtiles + qtile) * params.n_ktiles + ((T::kDebug & 4096) ? 0 : j)) * (int64_t)T::kBytesBias;   // dbg 4096: every tile load re-reads this CTA's tile 0 (L2-traffic isolation; WRONG results)
                bulk_g2s(cute::cast_smem_ptr_to_uint(shared.smem_bias.data()) + uint32_t(bslot) * T::kBytesBias, src, T::kBytesBias, cute::cast_smem_ptr_to_uint(pipe_bias.full_barrier(bslot)));
            } else {
                copy(params.tma_b.with(*pipe_bias.full_barrier(bslot), 0), tBgB(_, (T::kDebug & 4096) ? 0 : j), tBsB(_, bslot));
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                int const i = min(i0 + r, params.N - 1);
                int const st = R * slot + r;
                pipe_kv.producer_wait_empty(st, ph);
                pipe_kv.producer_expect(st);
                copy(params.tma_k.with(*pipe_kv.full_barrier(st), mcast_mask), tKgK(_, (T::kDebug & 4096) ? 0 : j, i), tKsK(_, st));
                copy(params.tma_v.with(*pipe_kv.full_barrier(st), mcast_mask), tVgV(_, (T::kDebug & 4096) ? 0 : j, i), tVsV(st));
            }
            pstamp(14);
        };
        static_assert(ClusterQ == 1, "static-schedule producer implemented for the CTA-local pipeline");
        if (lane_predicate) {
            shared.barrier_q.arrive_and_expect_tx(T::kBytesQ * R);
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                int const i = min(i0 + r, params.N - 1);
                copy(params.tma_q.with(reinterpret_cast<uint64_t&>(shared.barrier_q), 0 /*mcast*/), tQgQ(_, i), tQsQ(_, r));
            }
            if constexpr ((T::kDebug & 32768) == 0) { load_tile(0, 0); }
        }
        __syncwarp();
        cutlass::arch::NamedBarrier::sync(T::kHelperWarps ? T::kNumThreads : T::kNumThreads - 96, kBarLive);   // producer warp (+ helpers) + consumers: live list ready
        if (lane_predicate) {
            int const n_live = shared.n_live;
            if constexpr ((T::kDebug & 32768) != 0) { load_tile(0, 0); }
            for (int jl = 1; jl < n_live; ++jl) { load_tile(jl, shared.live[jl]); }
        }
        return;
    }

    // ================================================= CONSUMERS ======================================================
    cutlass::arch::warpgroup_reg_alloc<T::kRegsConsumer>();
    int const thread_idx = tid - 128;                       // 0..kNumMmaThreads-1
    int const cwg = int(__reduce_max_sync(0xffffffffu, unsigned(thread_idx / 128)));   // uniform register: every operand partition derived from it stays on the uniform datapath                       // consumer warpgroup c -> q rows [c*64, c*64+64) of the tile
    consumer_prologue();
    cutlass::arch::NamedBarrier::sync(T::kHelperWarps ? T::kNumThreads : T::kNumThreads - 96, kBarLive);   // + the producer warp(s): live list / classes / ones ready
    int const n_live = int(__reduce_max_sync(0xffffffffu, unsigned(shared.n_live)));   // uniform register (loop bounds, tile refs)
    bool const wg_leader = (thread_idx % 128) == 0;
    // trace: every lane of a traced warp executes the (same-address) store, so the stamp adds no divergent branch (a divergent branch
    // next to wgmma code makes ptxas serialize every wgmma of the kernel, diagnostic C7518)
    unsigned long long* trace = (params.trace != nullptr && blockIdx.x == 1 && blockIdx.y == 37 && blockIdx.z == 1)
                                ? params.trace + (thread_idx / 32) * 4096 : nullptr;     // lane 0 of each consumer warp
    int trace_n = 0;
    auto stamp = [&](int tag) __attribute__((always_inline)) { if constexpr ((T::kDebug & 8) != 0) { if (trace != nullptr && trace_n < 4090) { trace[trace_n++] = (clock64() << 8) | tag; } } };

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
    Tensor sB = make_tensor(make_smem_ptr(shared.smem_bias.data()), typename T::SmemLayoutBias{});

    Tensor tSrQ = wg_mma_qk.partition_fragment_A(sQ);        // (frag, MMA_M, MMA_K, R)
    Tensor tSrK = wg_mma_qk.partition_fragment_B(sK);        // (frag, MMA_N, MMA_K, st)
    Tensor tOrV = wg_mma_pv.partition_fragment_B(sVt);       // (frag, MMA_N, MMA_K, st)
    // P via smem: stmatrix copy partition (accumulator layout -> swizzled K-major tile) and the SS-wgmma A descriptors
    typename T::TiledMmaPVss tiled_mma_pvss;
    auto wg_mma_pvss = tiled_mma_pvss.get_slice(wg_layout(cwg));
    Tensor sP = make_tensor(make_smem_ptr(shared.smem_p.data()), typename T::SmemLayoutP{});
    Tensor tOsP = wg_mma_pvss.partition_fragment_A(sP);      // (frag, MMA_M, MMA_K, buf)
    Tensor tOrVss = wg_mma_pvss.partition_fragment_B(sVt);   // (frag, MMA_N, MMA_K, st)
    auto r2s_p = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, tiled_mma_qk);
    auto thr_r2s_p = r2s_p.get_thread_slice(thread_idx);
    Tensor tPsP = thr_r2s_p.partition_D(sP);                 // (CPY, CPY_M, CPY_N, buf)

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
    float row_m[R][kNRows];
    #pragma unroll
    for (int r = 0; r < R; ++r) { clear(acc_o[r]); row_m[r][0] = -INFINITY; row_m[r][1] = -INFINITY; }
    // Pin the zeroing here: left free, ptxas sinks each accumulator's zero-fill to just before its first wgmma, i.e. inside an
    // earlier wgmma's fence..wait window, and then serializes every wgmma of the kernel (ptxas diagnostic C7515).
    #pragma unroll
    for (int r = 0; r < R; ++r) { warpgroup_fence_operand(acc_o[r]); }

    AccS bias_f_unused;                                         // (layout carrier only)
    Tensor bias_rc_layout = make_tensor(bias_f_unused.data(), flash::convert_layout_acc_rowcol(bias_f_unused.layout()));
    constexpr int kNCols = decltype(size<1>(bias_rc_layout))::value;   // = kBlockN / 4
    // the bias of this thread's accumulator elements for the current k-tile, kept PACKED (bf16x2 per column pair: 16 registers) and
    // widened inside part1 (fp32 view of a bf16 = its 16 bits in the high half)
    uint32_t bias_w[2][kNCols / 2];
    static_assert(decltype(size<0>(bias_rc_layout))::value == kNRows);
    constexpr float kLog2e = 1.4426950408889634f;
constexpr float kMaskedLogit = -1.0e30f;      // logit of an excluded key: finite (no inf-inf NaN paths), exp underflows to exactly 0
    constexpr float kLazy = 5.5f;                               // natural units: running max may lag the true max by < e^5.5 = 245x
    float const scale = params.scale_log2e / kLog2e;

    // This thread's bias elements in a stage: rows r0 (mi=0) and r0+8 (mi=1), columns 2q+8c (+1), c = 0..7, q = lane%4. The stage is
    // the TMA 128B-swizzled K-major tile: byte(row, col) = row*128 + ((col*2) ^ ((row%8)*16)); rows r0 and r0+8 share the phase.
    // Address of pair c = row_base + ((16c) ^ x16) with x16 = (r0%8)*16: one LOP3 per load, no per-column address registers.
    static_assert(kBlockN * sizeof(Element) == 128, "bias smem rows are one 128-byte swizzle line (kBlockN = 64 for bf16)");
    constexpr uint32_t kRow8Bytes = 8 * kBlockN * sizeof(Element);
    constexpr uint32_t kBiasStageBytes = T::kBytesBias;
    int const lane = tid % 32;
    // accumulator-fragment coordinates of this thread (wgmma m64nN layout): rows acc_row0 + 8*mi, columns acc_col0 + 8*(ni/2) + ni%2
    int const acc_row0 = cwg * 64 + ((thread_idx % 128) / 32) * 16 + lane / 4;
    int const acc_col0 = 2 * (lane % 4);
    if (acc_row0 != get<0>(tScS_rc(0, 0)) || acc_col0 + 9 != get<1>(tScS_rc(1, 3)) || acc_row0 + 8 != get<0>(tOcO_rc(1, 0)) || acc_col0 + 8 != get<1>(tOcO_rc(0, 2))) { asm volatile("trap;"); }
    uint32_t const bias_r0 = uint32_t(acc_row0);                                       // this thread's first accumulator row in the tile
    uint32_t const bias_rowbase = cute::cast_smem_ptr_to_uint(shared.smem_bias.data()) + bias_r0 * 128u + uint32_t(lane % 4) * 4u;
    uint32_t const bias_x16 = (bias_r0 & 7u) << 4;
    {   // layout self-check against CuTe's swizzled layout (compile-time shape, runtime once; traps on mismatch)
        Tensor sB0 = sB(_, _, 0);
        uint32_t const ref = uint32_t(reinterpret_cast<char const*>(&sB0(get<0>(tScS_rc(0, 2)), get<1>(tScS_rc(0, 2)))) - reinterpret_cast<char const*>(&sB0(_0{}, _0{})));
        uint32_t const mine = bias_r0 * 128u + uint32_t(lane % 4) * 4u + (16u ^ bias_x16);
        if (ref != mine) { asm volatile("trap;"); }
    }

    static_assert(R % 2 == 0, "rows are processed in pairs");
    AccS acc_s[2];                                              // S / P (fp32) of the row pair in flight
    auto tOrP_a = make_tensor_like<Element>(make_tensor(acc_s[0].data(), flash::convert_layout_acc_Aregs<typename T::TiledMmaPV>(acc_s[0].layout())));   // packed bf16 P
    auto tOrP_b = make_tensor_like<Element>(tOrP_a);

    stamp(11);
    shared.barrier_q.wait(0);
    stamp(12);
    // (max-free) qn[r][mi] = scale*sqrt(D)*log2e*|q row|: the 4 threads of a quad own the same rows and split the D=32 elements
    float qn[R][kNRows];
    if constexpr (T::kMaxFree) {
        uint32_t const qbase = cute::cast_smem_ptr_to_uint(shared.smem_q.data());
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                // Q smem stage r: (kBlockM, D) K-major 64-byte rows, swizzled within the row -> a row's 32 elements are its 64 bytes
                uint32_t const a = qbase + uint32_t(r) * T::kBytesQ + uint32_t(acc_row0 + 8 * mi) * (T::kHeadDim * 2) + uint32_t(lane % 4) * 16;
                uint32_t w0, w1, w2, w3;
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3) : "r"(a));
                float ss = 0.f;
                #pragma unroll
                for (uint32_t w : {w0, w1, w2, w3}) { float const lo = __uint_as_float(w << 16), hi = __uint_as_float(w & 0xffff0000u); ss = fmaf(lo, lo, fmaf(hi, hi, ss)); }
                ss += __shfl_xor_sync(0xffffffffu, ss, 1); ss += __shfl_xor_sync(0xffffffffu, ss, 2);
                qn[r][mi] = sqrtf(ss) * params.scale_log2e * sqrtf(float(T::kHeadDim));
            }
        }
    } else {
        #pragma unroll
        for (int r = 0; r < R; ++r) { qn[r][0] = 0.f; qn[r][1] = 0.f; }
    }
    unsigned n_exact_steps = 0;                                 // (max-free) row-steps after the seed that needed the exact path
    if constexpr (T::kPingPong) {                               // consumer warpgroup 0 starts holding the token: the last warpgroup pre-arrives on its barrier
        if (cwg == T::kNumMmaWG - 1) {
            cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier));
        }
    }

    auto mma_kloop = [&](auto& mma, auto const& tA, auto const& tB, auto& tC, bool zero) __attribute__((always_inline)) {
        mma.accumulate_ = zero ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
        #pragma unroll
        for (int kb = 0; kb < size<2>(tA); ++kb) { cute::gemm(mma, tA(_, _, kb), tB(_, _, kb), tC); mma.accumulate_ = GMMA::ScaleOut::One; }
    };
    auto load_bias = [&](int slot, uint32_t ph) __attribute__((always_inline)) {
        if constexpr (T::kDebug & 2048) { pipe_bias.spin_full(slot, ph); } else { pipe_bias.wait_full(slot, ph); }
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int c = 0; c < kNCols / 2; ++c) {
                uint32_t const a = bias_rowbase + ((uint32_t(16 * c) ^ bias_x16) + uint32_t(slot * kBiasStageBytes + mi * kRow8Bytes));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(bias_w[mi][c]) : "r"(a));
            }
        }
        __syncwarp();
        pipe_bias.release(slot, lane == 0);                        // 8 consumer warps arrive
    };
    // softmax part 1 on one S fragment of row r: u = scale*s + bias (+ mask), thread-local row maxes. kMasked: per-element select
    // from the row's key bits (masked key: u - 1e9, cuEquivariance's additive mask, so fully-masked rows need no special case;
    // key >= S: excluded); instantiated only for tiles where some row's tile is not all-keep.
    auto part1 = [&](AccS& acc, int r, int j, float (&mx)[kNRows], auto masked_c, auto tree_c) __attribute__((always_inline)) {
        constexpr bool kMasked = decltype(masked_c)::value, kTree = decltype(tree_c)::value;
        Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int c = 0; c < kNCols / 2; ++c) {
                s_rc(mi, 2 * c) = fmaf(s_rc(mi, 2 * c), scale, __uint_as_float(bias_w[mi][c] << 16));                 // even column = low half
                s_rc(mi, 2 * c + 1) = fmaf(s_rc(mi, 2 * c + 1), scale, __uint_as_float(bias_w[mi][c] & 0xffff0000u)); // odd column = high half
            }
        }
        if constexpr (kMasked) {
            uint32_t const w0 = shared.mask_bits[r][j * (kBlockN / 32)], w1 = kBlockN > 32 ? shared.mask_bits[r][j * (kBlockN / 32) + 1] : 0u;
            uint64_t const bits = (uint64_t(w1) << 32) | w0;
            if (((bits + 1) & bits) == 0 && !shared.uniform[r]) {
                // keep-pattern = a prefix of the tile (left-aligned padding masks, and the ragged last tile: keys >= S have 0 bits):
                // dropped keys get a large finite negative logit (exp -> 0 exactly); one compare + select per element
                int const thr = __popcll(bits) - acc_col0;
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) {
                    #pragma unroll
                    for (int ni = 0; ni < kNCols; ++ni) { s_rc(mi, ni) = (8 * (ni / 2) + (ni % 2)) < thr ? s_rc(mi, ni) : kMaskedLogit; }
                }
            } else {
                // general pattern (interior zeros): dropped keys get the finite sentinel (exp -> 0 exactly). A row with NO attendable
                // key attends uniformly to its S keys (= cuEquivariance's additive-mask result, ruling H8): every in-range logit is
                // replaced by the constant 0, so P = 1 exactly and O = mean(V) at any logit magnitude; keys >= S stay excluded.
                int const n_inrange = S - j * kBlockN;
                bool const urow = shared.uniform[r] != 0;
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) {
                    #pragma unroll
                    for (int ni = 0; ni < kNCols; ++ni) {
                        int const n = acc_col0 + 8 * (ni / 2) + (ni % 2);
                        bool const keep = (bits >> n) & 1u;
                        float const fill = (urow && n < n_inrange) ? 0.f : kMaskedLogit;
                        s_rc(mi, ni) = keep ? s_rc(mi, ni) : fill;
                    }
                }
            }
        }
        if constexpr (kTree) {
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                float a0 = max(s_rc(mi, 0), s_rc(mi, 1)), a1 = max(s_rc(mi, 2), s_rc(mi, 3));
                #pragma unroll
                for (int ni = 4; ni < kNCols; ni += 4) { a0 = max(a0, max(s_rc(mi, ni), s_rc(mi, ni + 1))); a1 = max(a1, max(s_rc(mi, ni + 2), s_rc(mi, ni + 3))); }
                mx[mi] = max(a0, a1);
            }
        } else { (void)mx; }
    };
    auto update_max = [&](int r, float (&mx)[kNRows], float (&alpha)[kNRows]) __attribute__((always_inline)) {
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            flash::MaxOp<float> op;
            float const gmx = flash::Allreduce<4>::run(mx[mi], op);
            // a tile whose keys are all excluded (max = kMaskedLogit) must not move the running max: with m == kMaskedLogit the
            // exponent s*log2e - m*log2e of an excluded key would be the rounding error of two huge terms instead of 0 or -huge
            float const m_new = (gmx > row_m[r][mi] + kLazy && gmx > 0.5f * kMaskedLogit) ? gmx : row_m[r][mi];
            alpha[mi] = m_new == row_m[r][mi] ? 1.f : ex2_approx((row_m[r][mi] - m_new) * kLog2e);   // (-inf) - (-inf) would be NaN
            row_m[r][mi] = m_new;
        }
    };
    // exponent arguments x = s*log2e - m*log2e (in place): FFMA work, done BEFORE the warpgroup acquires the MUFU token
    auto exp_args = [&](AccS& acc, float const (&neg_mL)[kNRows]) __attribute__((always_inline)) {
        Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < kNCols; ++ni) { s_rc(mi, ni) = fmaf(s_rc(mi, ni), kLog2e, neg_mL[mi]); }
        }
        warpgroup_fence_operand(acc);                                           // keep the FFMAs on this side of the token barrier
    };
    // p = 2^x (in place): the MUFU stream, inside the token section
    auto exp_run = [&](AccS& acc) __attribute__((always_inline)) {
        Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < kNCols; ++ni) { s_rc(mi, ni) = ex2_approx(s_rc(mi, ni)); }
        }
    };
    auto exponentiate = [&](AccS& acc, int r, float const (&neg_mL)[kNRows]) __attribute__((always_inline)) {
        (void)r;
        Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
        // all 32 exponent arguments first (in place), then the ex2 stream: every MUFU op finds its input ready
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < kNCols; ++ni) { s_rc(mi, ni) = fmaf(s_rc(mi, ni), kLog2e, neg_mL[mi]); }
        }
        if constexpr (T::kDebug & 16384) { warpgroup_fence_operand(acc); }     // optional scheduling fence between the two passes
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < kNCols; ++ni) {
                float const x = s_rc(mi, ni);
                if constexpr (T::kPolyEvery > 0) { s_rc(mi, ni) = ((ni + mi) % T::kPolyEvery == T::kPolyEvery - 1) ? ex2_poly3(x) : ex2_approx(x); }
                else { s_rc(mi, ni) = ex2_approx(x); }
            }
        }
    };
    auto rescale_acc = [&](int r, float const (&alpha)[kNRows]) __attribute__((always_inline)) {
        Tensor o_rc = make_tensor(acc_o[r].data(), flash::convert_layout_acc_rowcol(acc_o[r].layout()));
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < kNColsO; ++ni) { o_rc(mi, ni) *= alpha[mi]; }
        }
    };
    auto pack_p = [&](AccS& acc, auto& tOrP) __attribute__((always_inline)) {
        Tensor p_acc = make_tensor(acc.data(), flash::convert_layout_acc_Aregs<typename T::TiledMmaPV>(acc.layout()));
        flash::convert_type_out(p_acc, tOrP);
    };
    // P via smem: pack to bf16 (accumulator layout), stmatrix into sP[buf], make the writes visible to the async proxy
    auto store_p = [&](AccS& acc, int buf) __attribute__((always_inline)) {
        Tensor rP = make_tensor_like<Element>(acc);
        flash::convert_type_out(acc, rP);
        Tensor taccrP = thr_r2s_p.retile_S(rP);
        cute::copy(r2s_p, taccrP, tPsP(_, _, _, buf));
        cutlass::arch::fence_view_async_shared();
    };

    // One pair-step = rows (2rp, 2rp+1) of k-tile j: S GEMMs of both rows (waited), softmax part 1 of both, then per row:
    // exponentials -> bf16 P -> PV GEMM issued (one register-sourced wgmma chain per fence group; the row-b exponentials run
    // under row a's GEMM). The pair's PV GEMMs stay in flight until the next pair-step's wait (after its S GEMMs).
    // S GEMMs of row pair rp for the k-tile in ring slot `slot`: waits for its K/V stages, issues both rows into acc_s[0], acc_s[1]
    // (wait: false only past the last tile, where the early S targets a stage that will never be refilled; its result is unused)
    auto issue_s = [&](int slot, uint32_t ph, auto rp_c, bool wait) __attribute__((always_inline)) {
        constexpr int ra = 2 * decltype(rp_c)::value, rb = ra + 1;
        int const sta = R * slot + ra, stb = sta + 1;
        if (wait) { pipe_kv.wait_full(sta, ph); pipe_kv.wait_full(stb, ph); }
        stamp(3);
        warpgroup_fence_operand(acc_s[0]); warpgroup_fence_operand(acc_s[1]);
        warpgroup_arrive();
        { auto tQ = tSrQ(_, _, _, Int<ra>{}); auto tK = tSrK(_, _, _, sta); mma_kloop(tiled_mma_qk, tQ, tK, acc_s[0], true); }
        { auto tQ = tSrQ(_, _, _, Int<rb>{}); auto tK = tSrK(_, _, _, stb); mma_kloop(tiled_mma_qk, tQ, tK, acc_s[1], true); }
        warpgroup_commit_batch();
        warpgroup_fence_operand(acc_s[0]); warpgroup_fence_operand(acc_s[1]);
    };
    // One pair-step, rows (2rp, 2rp+1) of k-tile j. Its S GEMMs were issued at the end of the previous pair-step (or by the cold start),
    // so the step begins with the wait; it ends by issuing the NEXT pair-step's S GEMMs (slot nslot, phase nph, row pair nrp) into the
    // S registers freed by the packs, so that they execute under this step's tail and the other warpgroup's work.
    auto pair_step = [&](int j, uint32_t ph, int slot, auto rp_c, bool pending, auto masked_c, int nslot, uint32_t nph, auto nrp_c, bool nwait, bool seed, int bslot) __attribute__((always_inline)) {
        constexpr int rp = decltype(rp_c)::value;
        constexpr int ra = 2 * rp, rb = ra + 1;
        int const sta = R * slot + ra, stb = sta + 1;
        int const psta = sta >= 2 ? sta - 2 : sta - 2 + T::kStages, pstb = psta + 1;   // V stages of the pending pair
        constexpr int pra = (ra + R - 2) % R, prb = pra + 1;                            // rows of the pending pair
        stamp(1);
        if constexpr (T::kDebug & 8192) { issue_s(slot, ph, rp_c, true); }      // variant without the early S issue
        warpgroup_wait<0>();                                                    // S done (and the pending PV GEMMs, issued a phase earlier)
        warpgroup_fence_operand(acc_s[0]); warpgroup_fence_operand(acc_s[1]);
        warpgroup_fence_operand(acc_o[pra]); warpgroup_fence_operand(acc_o[prb]);
        warpgroup_fence_operand(tOrP_a); warpgroup_fence_operand(tOrP_b);
        if (pending) { pipe_kv.release(psta, wg_leader); pipe_kv.release(pstb, wg_leader); }
        stamp(4);
        float mx_a[kNRows], mx_b[kNRows], al_a[kNRows], al_b[kNRows], nm_a[kNRows], nm_b[kNRows];
        // exact (running-max) path: always without max-free; with max-free: the seed step of each row (its first live tile), and any
        // later step whose exponent bound could exceed 2^100 (or whose row has no max yet)
        bool exact = true;
        if constexpr (T::kMaxFreeNoBound) {
            if (!seed) {
                bool risky = false;
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) { risky |= (row_m[ra][mi] == -INFINITY) | (row_m[rb][mi] == -INFINITY); }
                exact = __any_sync(0xffffffffu, risky);
            }
        } else if constexpr (T::kMaxFree) {
            if (!seed) {
                shared.bar_bound[sta].wait(ph); shared.bar_bound[stb].wait(ph);
                float const kma = max(max(shared.kmax_part[sta][0], shared.kmax_part[sta][1]), shared.kmax_part[sta][2]);
                float const kmb = max(max(shared.kmax_part[stb][0], shared.kmax_part[stb][1]), shared.kmax_part[stb][2]);
                float const bmL = max(max(shared.bmax_part[bslot][0], shared.bmax_part[bslot][1]), shared.bmax_part[bslot][2]) * kLog2e;
                bool risky = false;
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) {
                    risky |= !(fmaf(qn[ra][mi], kma, bmL - row_m[ra][mi] * kLog2e) < 100.f);   // also true for m = -inf / NaN
                    risky |= !(fmaf(qn[rb][mi], kmb, bmL - row_m[rb][mi] * kLog2e) < 100.f);
                }
                exact = __any_sync(0xffffffffu, risky);
                n_exact_steps += (exact && lane == 0) ? 1u : 0u;
            }
        } else { (void)seed; }
        if (exact) {
            part1(acc_s[0], ra, j, mx_a, masked_c, cute::true_type{});
            part1(acc_s[1], rb, j, mx_b, masked_c, cute::true_type{});
            bool any_up = false;
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) { any_up |= ((mx_a[mi] > row_m[ra][mi] + kLazy) & (mx_a[mi] > 0.5f * kMaskedLogit)) | ((mx_b[mi] > row_m[rb][mi] + kLazy) & (mx_b[mi] > 0.5f * kMaskedLogit)); }
            bool const resc = __any_sync(0xffffffffu, any_up);                 // rare after the first tile of a row
            if constexpr (T::kDebug & 32) { (void)resc; }
            else if constexpr (T::kDebug & 1) { update_max(ra, mx_a, al_a); update_max(rb, mx_b, al_b); rescale_acc(ra, al_a); rescale_acc(rb, al_b); (void)resc; }
            else { if (resc) { update_max(ra, mx_a, al_a); update_max(rb, mx_b, al_b); rescale_acc(ra, al_a); rescale_acc(rb, al_b); } }
        } else {
            part1(acc_s[0], ra, j, mx_a, masked_c, cute::false_type{});    // scale + bias (+ mask) only: no max, shuffle, vote or rescale
            part1(acc_s[1], rb, j, mx_b, masked_c, cute::false_type{});
        }
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            nm_a[mi] = row_m[ra][mi] == -INFINITY ? 0.f : -row_m[ra][mi] * kLog2e;
            nm_b[mi] = row_m[rb][mi] == -INFINITY ? 0.f : -row_m[rb][mi] * kLog2e;
        }
        // The kernel is MUFU-bound, so the two consumer warpgroups take turns on the EXPONENTIAL section (not on the GEMMs as in
        // FlashAttention-3): one warpgroup's ex2 stream runs while the other does its GEMM waits + softmax part 1, keeping the SM's
        // MUFU pipes busy back-to-back instead of contended-then-idle.
        if constexpr (T::kPss) {
            // P through shared memory: token section = MUFU stream a, pack+stmatrix a, PV a (SS wgmma), MUFU stream b, pack+stmatrix b;
            // PV b is issued after the token is released. sP buffers of this warpgroup are free: the previous PV GEMMs retired above.
            exp_args(acc_s[0], nm_a);
            exp_args(acc_s[1], nm_b);
            if constexpr (T::kPingPong) { warp_scheduler_barrier_sync<T>(); }
            stamp(5);
            exp_run(acc_s[0]);
            store_p(acc_s[0], 0);
            warpgroup_fence_operand(acc_o[ra]);
            warpgroup_arrive();
            { auto tA = tOsP(_, _, _, 0); auto tV = tOrVss(_, _, _, sta); mma_kloop(tiled_mma_pvss, tA, tV, acc_o[ra], false); }
            warpgroup_commit_batch();
            warpgroup_fence_operand(acc_o[ra]);
            stamp(6);
            exp_run(acc_s[1]);
            store_p(acc_s[1], 1);
            if constexpr (T::kPingPong) { warp_scheduler_barrier_arrive<T>(); }
            warpgroup_fence_operand(acc_o[rb]);
            warpgroup_arrive();
            { auto tA = tOsP(_, _, _, 1); auto tV = tOrVss(_, _, _, stb); mma_kloop(tiled_mma_pvss, tA, tV, acc_o[rb], false); }
            warpgroup_commit_batch();
            warpgroup_fence_operand(acc_o[rb]);
            stamp(7);
            if constexpr ((T::kDebug & 8192) == 0) { issue_s(nslot, nph, nrp_c, nwait); }
            stamp(2);
            return;
        }
        if constexpr (T::kDebug & 0x40000) {
            // exponent FFMAs of both rows before the token; token section = MUFU stream a, pack a, PV a, MUFU stream b, pack b, PV b
            exp_args(acc_s[0], nm_a);
            exp_args(acc_s[1], nm_b);
            if constexpr (T::kPingPong) { warp_scheduler_barrier_sync<T>(); }
            stamp(5);
            if constexpr (T::kDebug & 0x80000) {                           // DIAGNOSIS (wrong results): packs do not depend on the MUFU outputs
                pack_p(acc_s[0], tOrP_a); exp_run(acc_s[0]); warpgroup_fence_operand(acc_s[0]);
                Tensor s_rc0 = make_tensor(acc_s[0].data(), flash::convert_layout_acc_rowcol(acc_s[0].layout()));
                float sink = 0.f;
                #pragma unroll
                for (int ni = 0; ni < kNCols; ni += 4) { sink += s_rc0(0, ni) + s_rc0(1, ni + 1); }
                if (sink == 12345.f) { tOrP_a(0) = Element(sink); }             // keep the MUFUs alive
            } else {
            exp_run(acc_s[0]);
            pack_p(acc_s[0], tOrP_a);
            }
            warpgroup_fence_operand(tOrP_a); warpgroup_fence_operand(acc_o[ra]);
            warpgroup_arrive();
            { auto tV = tOrV(_, _, _, sta); mma_kloop(tiled_mma_pv, tOrP_a, tV, acc_o[ra], false); }
            warpgroup_commit_batch();
            warpgroup_fence_operand(tOrP_a); warpgroup_fence_operand(acc_o[ra]);
            stamp(6);
            if constexpr (T::kDebug & 0x80000) {
                pack_p(acc_s[1], tOrP_b); exp_run(acc_s[1]); warpgroup_fence_operand(acc_s[1]);
                Tensor s_rc1 = make_tensor(acc_s[1].data(), flash::convert_layout_acc_rowcol(acc_s[1].layout()));
                float sink = 0.f;
                #pragma unroll
                for (int ni = 0; ni < kNCols; ni += 4) { sink += s_rc1(0, ni) + s_rc1(1, ni + 1); }
                if (sink == 12345.f) { tOrP_b(0) = Element(sink); }
            } else {
            exp_run(acc_s[1]);
            pack_p(acc_s[1], tOrP_b);
            }
            warpgroup_fence_operand(tOrP_b); warpgroup_fence_operand(acc_o[rb]);
            warpgroup_arrive();
            { auto tV = tOrV(_, _, _, stb); mma_kloop(tiled_mma_pv, tOrP_b, tV, acc_o[rb], false); }
            warpgroup_commit_batch();
            if constexpr (T::kPingPong) { warp_scheduler_barrier_arrive<T>(); }
            warpgroup_fence_operand(tOrP_b); warpgroup_fence_operand(acc_o[rb]);
            stamp(7);
            if constexpr ((T::kDebug & 8192) == 0) { issue_s(nslot, nph, nrp_c, nwait); }
            stamp(2);
            return;
        }
        if constexpr (T::kPingPong) { warp_scheduler_barrier_sync<T>(); }
        stamp(5);
        if constexpr (T::kDebug & 128) {
            // token section = the two rows' exponentials and packs only; both PV GEMMs are issued after the token is released, so the
            // partner warpgroup's exponentials start as soon as ours end (wgmma issue can block on a full tensor-core queue)
            exponentiate(acc_s[0], ra, nm_a);
            exponentiate(acc_s[1], rb, nm_b);
            pack_p(acc_s[0], tOrP_a);
            pack_p(acc_s[1], tOrP_b);
            if constexpr (T::kPingPong) { warp_scheduler_barrier_arrive<T>(); }
            stamp(6);
            warpgroup_fence_operand(tOrP_a); warpgroup_fence_operand(acc_o[ra]);
            warpgroup_arrive();
            { auto tV = tOrV(_, _, _, sta); mma_kloop(tiled_mma_pv, tOrP_a, tV, acc_o[ra], false); }
            warpgroup_commit_batch();
            warpgroup_fence_operand(tOrP_a); warpgroup_fence_operand(acc_o[ra]);
            warpgroup_fence_operand(tOrP_b); warpgroup_fence_operand(acc_o[rb]);
            warpgroup_arrive();
            { auto tV = tOrV(_, _, _, stb); mma_kloop(tiled_mma_pv, tOrP_b, tV, acc_o[rb], false); }
            warpgroup_commit_batch();
            warpgroup_fence_operand(tOrP_b); warpgroup_fence_operand(acc_o[rb]);
            stamp(7);
            if constexpr ((T::kDebug & 8192) == 0) { issue_s(nslot, nph, nrp_c, nwait); }
            stamp(2);
            return;
        }
        if constexpr (T::kDebug & 64) {
            // both rows' 64 exponentials are issued before any bf16 pack consumes one: the MUFU stream runs at pipe rate instead of
            // stalling the in-order issue on each pack's operand latency
            exponentiate(acc_s[0], ra, nm_a);
            exponentiate(acc_s[1], rb, nm_b);
            asm volatile("" ::: "memory");
            pack_p(acc_s[0], tOrP_a);
        } else {
            exponentiate(acc_s[0], ra, nm_a);
            if constexpr (T::kDebug & 0x200000) { warpgroup_fence_operand(acc_s[0]); stamp(15); }
            pack_p(acc_s[0], tOrP_a);
        }
        warpgroup_fence_operand(tOrP_a); warpgroup_fence_operand(acc_o[ra]);
        if constexpr (T::kDebug & 0x200000) { stamp(16); }
        warpgroup_arrive();
        { auto tV = tOrV(_, _, _, sta); mma_kloop(tiled_mma_pv, tOrP_a, tV, acc_o[ra], false); }
        warpgroup_commit_batch();                                               // PV of row a runs under row b's exponentials
        warpgroup_fence_operand(tOrP_a); warpgroup_fence_operand(acc_o[ra]);
        stamp(6);
        if constexpr ((T::kDebug & 64) == 0) { exponentiate(acc_s[1], rb, nm_b); }
        if constexpr (T::kDebug & 0x200000) { warpgroup_fence_operand(acc_s[1]); stamp(17); }
        pack_p(acc_s[1], tOrP_b);
        warpgroup_fence_operand(tOrP_b); warpgroup_fence_operand(acc_o[rb]);
        if constexpr (T::kDebug & 0x200000) { stamp(18); }
        warpgroup_arrive();
        { auto tV = tOrV(_, _, _, stb); mma_kloop(tiled_mma_pv, tOrP_b, tV, acc_o[rb], false); }
        warpgroup_commit_batch();
        if constexpr (T::kPingPong) { warp_scheduler_barrier_arrive<T>(); }
        warpgroup_fence_operand(tOrP_b); warpgroup_fence_operand(acc_o[rb]);
        stamp(7);
        if constexpr ((T::kDebug & 8192) == 0) { issue_s(nslot, nph, nrp_c, nwait); }   // next pair-step's S GEMMs
        else { (void)nslot; (void)nph; (void)nwait; }
        stamp(2);
    };
    // ---- single-row steps (3 consumer warpgroups) -------------------------------------------------------------------
    auto issue_s1 = [&](int slot, uint32_t ph, auto r_c, bool wait) __attribute__((always_inline)) {
        constexpr int r = decltype(r_c)::value;
        int const st = R * slot + r;
        if (wait) { pipe_kv.wait_full(st, ph); }
        stamp(3);
        warpgroup_fence_operand(acc_s[0]);
        warpgroup_arrive();
        { auto tQ = tSrQ(_, _, _, Int<r>{}); auto tK = tSrK(_, _, _, st); mma_kloop(tiled_mma_qk, tQ, tK, acc_s[0], true); }
        warpgroup_commit_batch();
        warpgroup_fence_operand(acc_s[0]);
    };
    // One row-step (row r of k-tile j): its S GEMM was issued at the end of the previous step (or by the cold start); it ends with
    // PV(this) then S(next). npend = number of earlier steps whose PV is unretired (0 or 1 here).
    auto row_step = [&](int j, uint32_t ph, int slot, auto r_c, int npend, auto masked_c, int nslot, uint32_t nph, auto nr_c, bool nwait) __attribute__((always_inline)) {
        constexpr int r = decltype(r_c)::value;
        constexpr int pr = (r + R - 1) % R;                                     // row of the previous step
        int const st = R * slot + r;
        int const pst = st >= 1 ? st - 1 : T::kStages - 1;                      // V stage of the previous step
        stamp(1);
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_s[0]); warpgroup_fence_operand(acc_o[pr]); warpgroup_fence_operand(tOrP_a);
        if (npend >= 1) { pipe_kv.release(pst, wg_leader); }
        stamp(4);
        float mx[kNRows], al[kNRows], nm[kNRows];
        part1(acc_s[0], r, j, mx, masked_c, cute::true_type{});
        bool any_up = false;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) { any_up |= mx[mi] > row_m[r][mi] + kLazy && mx[mi] > 0.5f * kMaskedLogit; }
        bool const resc = __any_sync(0xffffffffu, any_up);
        if constexpr (T::kDebug & 1) { update_max(r, mx, al); rescale_acc(r, al); (void)resc; }
        else { if (resc) { update_max(r, mx, al); rescale_acc(r, al); } }
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) { nm[mi] = row_m[r][mi] == -INFINITY ? 0.f : -row_m[r][mi] * kLog2e; }
        if constexpr (T::kPss) {
            exp_args(acc_s[0], nm);
            if constexpr (T::kPingPong) { warp_scheduler_barrier_sync<T>(); }
            stamp(5);
            exp_run(acc_s[0]);
            store_p(acc_s[0], 0);                                               // the previous PV (reader of sP) retired at the wait above
            if constexpr (T::kPingPong) { warp_scheduler_barrier_arrive<T>(); }
            warpgroup_fence_operand(acc_o[r]);
            warpgroup_arrive();
            { auto tA = tOsP(_, _, _, 0); auto tV = tOrVss(_, _, _, st); mma_kloop(tiled_mma_pvss, tA, tV, acc_o[r], false); }
            warpgroup_commit_batch();
            warpgroup_fence_operand(acc_o[r]);
        } else {
        if constexpr (T::kPingPong) { warp_scheduler_barrier_sync<T>(); }      // token section: this warpgroup's turn on the MUFU pipes
        stamp(5);
        exponentiate(acc_s[0], r, nm);
        stamp(13);
        pack_p(acc_s[0], tOrP_a);
        stamp(16);
        if constexpr (T::kPingPong) { warp_scheduler_barrier_arrive<T>(); }
        warpgroup_fence_operand(tOrP_a); warpgroup_fence_operand(acc_o[r]);
        warpgroup_arrive();
        { auto tV = tOrV(_, _, _, st); mma_kloop(tiled_mma_pv, tOrP_a, tV, acc_o[r], false); }
        warpgroup_commit_batch();
        warpgroup_fence_operand(tOrP_a); warpgroup_fence_operand(acc_o[r]);
        }
        stamp(7);
        issue_s1(nslot, nph, nr_c, nwait);                                      // next row-step's S GEMM
        stamp(2);
    };
    // retire the pending step(s): rows R-2, R-1 (pair mode) or row R-1 (single-row mode) of the tile in ring slot pslot
    auto drain = [&](int pslot, int npend) __attribute__((always_inline)) {
        int const plast = R * pslot + R - 1;                                    // stage of the last processed step (row R-1 of tile pslot)
        warpgroup_wait<0>();
        #pragma unroll
        for (int r = 0; r < R; ++r) { warpgroup_fence_operand(acc_o[r]); }
        warpgroup_fence_operand(tOrP_a); warpgroup_fence_operand(tOrP_b); warpgroup_fence_operand(acc_s[0]);
        if constexpr (!T::kSingleRow) { if (npend >= 1) { pipe_kv.release(plast - 1, wg_leader); } }
        if (npend >= 1) { pipe_kv.release(plast, wg_leader); }
    };
    // a skipped k-tile: cycle its stages without computing (keeps the schedule static)
    // npend: steps whose PV GEMMs are unretired / whose V stages are unreleased (0 or 1)
    auto tile = [&](int jl, int& npend) __attribute__((always_inline, flatten)) {
        int const j = shared.live[jl];
        int const slot = jl % T::kRing; uint32_t const ph = (jl / T::kRing) & 1;
        int const nslot = slot + 1 == T::kRing ? 0 : slot + 1;
        int const bslot = T::kRingB == T::kRing ? slot : jl % T::kRingB; uint32_t const bph = T::kRingB == T::kRing ? ph : (jl / T::kRingB) & 1;
        stamp(8);
        if (jl == 0) {                                                          // cold start
            if constexpr (T::kSingleRow) { issue_s1(slot, ph, Int<0>{}, true); } else if constexpr ((T::kDebug & 8192) == 0) { issue_s(slot, ph, Int<0>{}, true); }
        }
        load_bias(bslot, bph);
        stamp(9);
        // early S for the next live tile (slot nslot): past the last one that stage is never refilled, so the wait is skipped and
        // the (unused) GEMM reads stale smem; the drain retires it
        bool const nwait = jl + 1 < n_live;
        uint32_t const nph = nslot == 0 ? ph ^ 1 : ph;
        bool all_clean = true;
        if constexpr ((T::kDebug & 2) == 0) {
            #pragma unroll
            for (int r = 0; r < R; ++r) { all_clean &= shared.tile_class[r][j] == kClean; }
        }
        static_assert(R == 2 || R == 4 || (T::kSingleRow && R % 2 == 0));
        auto steps = [&](auto masked_c) __attribute__((always_inline)) {
            if constexpr (T::kSingleRow) {
                row_step(j, ph, slot, Int<0>{}, npend, masked_c, slot, ph, Int<1>{}, true);
                static_for<R - 2>([&](auto q_c) { row_step(j, ph, slot, Int<decltype(q_c)::value + 1>{}, 1, masked_c, slot, ph, Int<decltype(q_c)::value + 2>{}, true); });
                row_step(j, ph, slot, Int<R - 1>{}, 1, masked_c, nslot, nph, Int<0>{}, nwait);
            } else if constexpr (R == 2) {
                pair_step(j, ph, slot, Int<0>{}, npend > 0, masked_c, nslot, nph, Int<0>{}, nwait, jl == 0, bslot);
            } else {
                pair_step(j, ph, slot, Int<0>{}, npend > 0, masked_c, slot, ph, Int<1>{}, true, jl == 0, bslot);
                pair_step(j, ph, slot, Int<1>{}, true, masked_c, nslot, nph, Int<0>{}, nwait, jl == 0, bslot);
            }
            npend = 1;
        };
        if (all_clean) { steps(cute::false_type{}); } else { steps(cute::true_type{}); }
    };

    if constexpr (T::kStream64) {
    // =================================================================================================================
    // CHUNK-STREAMING MAX-FREE consumer, 64-key chunks: chunk k = 2*jl + r covers the whole live tile jl of pair row r (64 q rows per
    // warpgroup x 64 keys = 32 exponentials per thread per chunk, so the per-chunk fixed costs -- wgmma wait, barrier waits, HGMMA
    // issue, pack -- are amortised over twice the MUFU work of a 32-key chunk). S(k) = Q_r K^T issued ONE chunk ahead into accC[k % 3];
    // x = S*c_l2 + bias*log2e + nm[r] (per-row shift fixed after the exact seed on tile 0; renormalised by 2^-64 at a period boundary
    // if a row sum passes 2^60); p = 2^x in place; pack + PV over [V | 1] one chunk late from PCb[k % 2], PV the last commit of a body.
    // PERIOD = 6 chunks = 3 tiles = the K/V and bias rings (compile-time stages); partial last period leaves after body 1 or 3; body 0
    // follows a drain and issues no wait. Bias: bf16 fragment order, 4 x 16 B shared loads per thread per tile, widened inside E.
    // =================================================================================================================
        static_assert(R == 2 && T::kRing == 3 && T::kRingB == 3 && kBlockN == 64 && T::kNumMmaWG == 2, "stream64 loop geometry");
        constexpr int CW = 64, kNC = CW / 4, kNKBc = CW / 16;
        typename T::TiledMmaQK64 tiled_mma_qkc;
        auto wg_mma_qkc = tiled_mma_qkc.get_slice(wg_layout(cwg));
        using AccC = decltype(partition_fragment_C(tiled_mma_qkc, make_shape(Int<kBlockM>{}, Int<CW>{})));
        Tensor tSrQc = wg_mma_qkc.partition_fragment_A(sQ);                                             // (frag, 1, 2, R)
        auto mk_K = [&](int zoff) {
            Tensor sKz = make_tensor(make_smem_ptr(shared.smem_k.data() + zoff), typename T::SmemLayoutK{});
            return wg_mma_qkc.partition_fragment_B(sKz);                                                 // (frag, N, 2, st)
        };
        auto mk_V = [&](int zoff) {
            Tensor sVz = make_tensor(make_smem_ptr(shared.smem_v.data() + zoff), typename T::SmemLayoutVt{});
            return wg_mma_pv.partition_fragment_B(sVz);                                                  // (frag, N, 4, st)
        };
        using OpK = decltype(mk_K(0)); using OpV = decltype(mk_V(0));
        AccC accC[3];                                           // S chunk buffers: chunk k in accC[k % 3]
        auto pc_proto = make_tensor_like<Element>(make_tensor(accC[0].data(), flash::convert_layout_acc_Aregs<typename T::TiledMmaPV>(accC[0].layout())));
        decltype(pc_proto) PCb[2];                              // bf16 P of chunk k in PCb[k % 2]
        static_assert(decltype(size<2>(pc_proto))::value == kNKBc);
        float nm[R][kNRows];
        float const c_l2 = params.scale_log2e;
        int const K_total = 2 * n_live;
        constexpr uint32_t kFragHalf = T::kBytesBias / 2;
        uint32_t const frag_base = cute::cast_smem_ptr_to_uint(shared.smem_bias.data()) + uint32_t((cwg * 4 + (thread_idx % 128) / 32) * 2 * 32 + lane) * 16u;
        uint32_t bw[2][kNRows][4];                              // [key half][row mi][4 words]: this thread's bias fragment of the current tile
        auto load_bias_tile = [&](int bslot) __attribute__((always_inline)) {
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) {
                    uint32_t const a = frag_base + uint32_t(bslot) * T::kBytesBias + uint32_t(c) * kFragHalf + uint32_t(mi) * 512u;
                    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(bw[c][mi][0]), "=r"(bw[c][mi][1]), "=r"(bw[c][mi][2]), "=r"(bw[c][mi][3]) : "r"(a));
                }
            }
        };
        auto bias_f = [&](int mi, int ni) __attribute__((always_inline)) {       // rowcol column ni (0..15) of a 64-key chunk: half ni/8, word (ni%8)/2
            uint32_t const w = bw[ni / 8][mi][(ni % 8) / 2];
            return __uint_as_float((ni % 2) ? (w & 0xffff0000u) : (w << 16));
        };
        auto issue_qk = [&](AccC& acc, OpK const& tK, auto rc, auto stc) __attribute__((always_inline)) {
            constexpr int r = decltype(rc)::value, st = decltype(stc)::value;
            warpgroup_fence_operand(acc);
            warpgroup_arrive();
            tiled_mma_qkc.accumulate_ = GMMA::ScaleOut::Zero;
            #pragma unroll
            for (int kb = 0; kb < 2; ++kb) { cute::gemm(tiled_mma_qkc, tSrQc(_, _, kb, Int<r>{}), tK(_, _, kb, Int<st>{}), acc); tiled_mma_qkc.accumulate_ = GMMA::ScaleOut::One; }
            warpgroup_commit_batch();
        };
        auto issue_pvc = [&](decltype(pc_proto)& tP, OpV const& tV, auto rc, auto stc) __attribute__((always_inline)) {
            constexpr int r = decltype(rc)::value, st = decltype(stc)::value;
            warpgroup_fence_operand(tP);
            warpgroup_arrive();
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
            #pragma unroll
            for (int kb = 0; kb < kNKBc; ++kb) { cute::gemm(tiled_mma_pv, tP(_, _, kb), tV(_, _, kb, Int<st>{}), acc_o[r]); }
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
        auto pp_sync = [&]() __attribute__((always_inline)) { if constexpr (T::kPingPong) { warp_scheduler_barrier_sync<T>(); } };
        auto pp_arrive = [&]() __attribute__((always_inline)) { if constexpr (T::kPingPong) { warp_scheduler_barrier_arrive<T>(); } };
        auto exp_core = [&](AccC& acc, auto rc) __attribute__((always_inline)) {             // p = 2^(S*c_l2 + bias*log2e + nm[r]) in place
            constexpr int r = decltype(rc)::value;
            Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = fmaf(s_rc(mi, ni), c_l2, fmaf(bias_f(mi, ni), kLog2e, nm[r][mi])); }
            }
            if constexpr (T::kDebug & 0x4000000) { warpgroup_fence_operand(acc); pp_sync(); }   // token around the MUFU pass only
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(s_rc(mi, ni)); }
            }
            warpgroup_fence_operand(acc);
            if constexpr (T::kDebug & 0x4000000) { pp_arrive(); }
        };
        auto affine = [&](AccC& acc) __attribute__((always_inline)) {
            Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = fmaf(s_rc(mi, ni), c_l2, bias_f(mi, ni) * kLog2e); }
            }
        };
        auto seed_row = [&](AccC& a0, int r) __attribute__((always_inline)) {                    // nm[r] <- -(max over tile 0 of row r's logits)
            Tensor s0 = make_tensor(a0.data(), flash::convert_layout_acc_rowcol(a0.layout()));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                float mx = -INFINITY;
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { mx = max(mx, s0(mi, ni)); }
                mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                nm[r][mi] = (mx == -INFINITY) ? 0.f : -mx;
            }
        };
        auto exp_seeded = [&](AccC& acc, int r) __attribute__((always_inline)) {
            Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
            pp_sync();
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(s_rc(mi, ni) + nm[r][mi]); }
            }
            warpgroup_fence_operand(acc);
            pp_arrive();
        };
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
        auto st_of = [](auto tc, auto rc) { return Int<2 * ((1 + decltype(tc)::value) % 3) + decltype(rc)::value>{}; };

        // ---- body for chunk dd (0..5) of a period, k = 2 + 6p + dd = 2*jl + r (tile t = dd/2 of the period, row r = dd&1):
        //      [tile's bias fragments at r == 0] | wait<1> (all but PV(k-2): QK(k) retired) | release stage of chunk k-3, bias of tile t-1 |
        //      wait for chunk k+1's stage (and its tile's bias) | QK(k+1) -> accC[(k+1)%3] | E(k) | pack(k-1) -> PCb[(k-1)%2] | PV(k-1) LAST
        auto body = [&](auto ddc, int jbase, OpK const& tK, OpV const& tV) __attribute__((always_inline)) {
            constexpr int dd = decltype(ddc)::value;
            constexpr int bq = (dd + 2) % 3, bn = (dd + 3) % 3, bp = (dd + 1) % 3;                 // S(k), S(k+1), S(k-1) buffers ((2 + dd) % 3 ...)
            constexpr int pb = (dd + 1) & 1;                                                       // P buffer of chunk k-1 ((k-1) % 2, k = 2 + 6p + dd)
            constexpr int t = dd / 2, r = dd & 1;
            constexpr int dd1 = dd + 1, t1 = dd1 / 2, r1 = dd1 & 1;                                 // chunk k+1 (t1 == 3: next period's tile 0)
            constexpr int ddp = (dd + 5) % 6, tp = ddp / 2, rp = ddp & 1;                           // chunk k-1
            constexpr int ddo = (dd + 3) % 6, to = ddo / 2, ro = ddo & 1;                           // chunk k-3 (its PV retired at this body's wait)
            stamp(1);
            if constexpr (r == 0) { load_bias_tile((1 + t) % 3); }
            if constexpr (dd != 0) { warpgroup_wait<1>(); }
            warpgroup_fence_operand(accC[bq]);
            warpgroup_fence_operand(PCb[pb]);
            if constexpr (dd == 0) {                                                               // chunk k-3 does not exist in period 0 (k = 2)
                if (jbase != 1) { pipe_kv.release(decltype(st_of(Int<to>{}, Int<ro>{}))::value, wg_leader); }
            } else { pipe_kv.release(decltype(st_of(Int<to>{}, Int<ro>{}))::value, wg_leader); }  // stage of chunk k-3 (its PV retired at the wait)
            if constexpr (r == 0) { __syncwarp(); pipe_bias.release(t % 3, lane == 0); }           // previous tile's bias slot (consumed by E(k-2), E(k-1))
            int const j1 = jbase + t1;
            if (j1 < n_live) {
                if constexpr (r1 == 0) { pipe_bias.wait_full((1 + t1) % 3, uint32_t((j1 / 3) & 1)); }
                pipe_kv.wait_full(decltype(st_of(Int<t1 % 3>{}, Int<r1>{}))::value, uint32_t((j1 / 3) & 1));
            }
            issue_qk(accC[bn], tK, Int<r1>{}, st_of(Int<t1 % 3>{}, Int<r1>{}));                    // phantom past the end
            stamp(5);
            if constexpr (!(T::kDebug & 0x4000000)) { pp_sync(); }
            stamp(7);
            exp_core(accC[bq], Int<r>{});
            if constexpr (!(T::kDebug & 0x4000000)) { pp_arrive(); }
            stamp(6);
            pack_chunk(accC[bp], PCb[pb]);
            issue_pvc(PCb[pb], tV, Int<rp>{}, st_of(Int<tp>{}, Int<rp>{}));
            stamp(2);
        };
        // after body 1 / 3 / 5 (k odd = row 1): the exp'd unpacked chunk k is in accC[(k)%3]... k = 2+6p+dd with dd odd -> (dd+2)%3
        auto drain = [&](auto ddlast) __attribute__((always_inline)) {
            constexpr int bq_last = (decltype(ddlast)::value + 2) % 3;
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]);
            warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]); warpgroup_fence_operand(accC[2]);
            warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]);
            l_check(accC[bq_last], 1);
        };

        // ---- prologue: tile 0 = chunks 0 (row 0, stage 0) and 1 (row 1, stage 1), bias slot 0, phases 0; exact per-row max as the
        //      shift; ends DRAINED in the state body 0 expects: S(2) landed in accC[2]; chunk 1 exp'd in accC[1], unpacked.
        OpK const tK0 = mk_K(params.zero); OpV const tV0 = mk_V(params.zero);
        pipe_bias.wait_full(0, 0); pipe_kv.wait_full(0, 0);
        load_bias_tile(0);
        issue_qk(accC[0], tK0, _0{}, _0{});
        pipe_kv.wait_full(1, 0);
        issue_qk(accC[1], tK0, _1{}, _1{});
        warpgroup_wait<0>();
        warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]);
        affine(accC[0]); seed_row(accC[0], 0);
        affine(accC[1]); seed_row(accC[1], 1);
        if (n_live > 1) { pipe_bias.wait_full(1, 0); pipe_kv.wait_full(2, 0); }                    // tile 1 = t 0 of period 0: slot 1, stage 2 (row 0)
        issue_qk(accC[2], tK0, _0{}, _2{});                                                        // QK(2) (phantom when n_live == 1)
        exp_seeded(accC[0], 0); pack_chunk(accC[0], PCb[0]); issue_pvc(PCb[0], tV0, _0{}, _0{});
        exp_seeded(accC[1], 1);                                                                    // chunk 1 stays unpacked: body 0 (or the finish) packs it
        warpgroup_wait<0>();
        warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]); warpgroup_fence_operand(accC[2]);
        warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]); warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]);
        // (stage 0 is released by body 1 (chunk k-3 = 0), stage 1 by body 2 or the finish, bias slot 0 by body 0)

        #pragma unroll 1
        for (int k0 = 2, jbase = 1; k0 < K_total; k0 += 6, jbase += 3) {
            int const left = K_total - k0;                                                         // 2, 4 or >= 6
            int const zoff = params.zero * k0;
            OpK const tK = mk_K(zoff); OpV const tV = mk_V(zoff);
            body(Int<0>{}, jbase, tK, tV); body(Int<1>{}, jbase, tK, tV);
            if (left == 2) { drain(Int<1>{}); break; }
            body(Int<2>{}, jbase, tK, tV); body(Int<3>{}, jbase, tK, tV);
            if (left == 4) { drain(Int<3>{}); break; }
            body(Int<4>{}, jbase, tK, tV); body(Int<5>{}, jbase, tK, tV);
            drain(Int<5>{});
        }
        {   // the last chunk K-1 (row 1 of tile n_live-1, slot m3 = (n_live-1) % 3, stage 2*m3+1) is exp'd, unpacked, in accC[(K_total-1) % 3]
            int const m3 = (n_live - 1) % 3;
            auto finish = [&](auto m3c, AccC& last) __attribute__((always_inline)) {
                constexpr int st0 = 2 * decltype(m3c)::value;
                pack_chunk(last, PCb[1]);
                issue_pvc(PCb[1], tV0, _1{}, Int<st0 + 1>{});
                warpgroup_wait<0>();
                warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]);
                pipe_kv.release(st0 + 1, wg_leader);                                                   // stage of the last chunk (st0 = row 0 was chunk K-2: released above unless K == 2)
                if (n_live == 1) { pipe_kv.release(st0, wg_leader); }
                __syncwarp(); pipe_bias.release(decltype(m3c)::value, lane == 0);
            };
            // K_total - 1 = 2*n_live - 1: (K_total - 1) % 3 depends on n_live % 3: n_live%3==1 -> 1, ==2 -> 0, ==0 -> 2
            if (m3 == 0) { finish(_0{}, accC[1]); } else if (m3 == 1) { finish(_1{}, accC[0]); } else { finish(_2{}, accC[2]); }
        }
    } else if constexpr (T::kStream) {
    // =================================================================================================================
    // CHUNK-STREAMING MAX-FREE consumer. The live k-tiles of both rows form one stream of S-chunks (64 q rows per warpgroup x
    // kChunkW = 32 keys): chunk k = 4*jl + 2*r + c for live tile jl, pair row r, key half c. Per chunk: S(k) = Q_r K^T, one wgmma
    // group issued TWO chunks ahead into accC[k % 4]; x = S*c_l2 + bias*log2e + nm[r] with a per-row shift nm that is fixed after
    // an exact seed on tile 0 (no running max, no rescale; a row whose sum passes 2^60 is renormalised by 2^-64 at a period boundary);
    // p = 2^x in place (MUFU); the bf16 pack and the PV wgmma over [V | 1] (row sums from the same bf16 P) run one chunk late from
    // PCb[k % 2], and PV is the LAST commit of a chunk body (ptxas hoists the next body's wait only up to it, so the exponentials of
    // chunk k overlap the tensor work of chunks k-1 / k+2). Every wgmma group retires at the end of a PERIOD of 12 chunks = 3 tiles
    // = the K/V and bias rings, so all stage indices are compile-time; the 12 chunk bodies exist once; a partial last period leaves
    // after body 3 or 7. Body 0 follows a full drain and issues no wait (a wait with nothing pending makes ptxas serialize every
    // wgmma of the function). Bias: bf16, staged per (q-tile, k-tile) in MMA-fragment order (one 16-byte shared load per thread,
    // accumulator row and key half; conflict-free), widened in the exponent FFMA chain. M1 scope: no per-row masks (keys >= S are
    // excluded through -inf bias columns written by the staging pass).
    // =================================================================================================================
        static_assert(R == 2 && T::kRing == 3 && T::kRingB == 3 && kBlockN == 64 && (T::kNumMmaWG == 2 || T::kNumMmaWG == 3), "stream loop geometry");
        constexpr int CW = T::kChunkW, kNC = CW / 4, kNKBc = CW / 16;
        typename T::TiledMmaQKC tiled_mma_qkc;
        auto wg_mma_qkc = tiled_mma_qkc.get_slice(wg_layout(cwg));
        using AccC = decltype(partition_fragment_C(tiled_mma_qkc, make_shape(Int<kBlockM>{}, Int<CW>{})));
        Tensor tSrQc = wg_mma_qkc.partition_fragment_A(sQ);                                             // (frag, 1, 2, R)
        // K / [V|1] operand tensors with (k-block, chunk half, stage) modes, rebuilt per period from a base offset by params.zero * k0
        // (== 0, opaque): the period's distinct descriptors are then derived at their use from a uniform base instead of hoisted.
        auto mk_K = [&](int zoff) {
            Tensor sKz = make_tensor(make_smem_ptr(shared.smem_k.data() + zoff), typename T::SmemLayoutK{});
            return wg_mma_qkc.partition_fragment_B(local_tile(sKz, make_shape(Int<CW>{}, Int<kHeadDim>{}), make_coord(_, _0{})));   // (frag, N, 2, c, st)
        };
        auto mk_V = [&](int zoff) {
            Tensor sVz = make_tensor(make_smem_ptr(shared.smem_v.data() + zoff), typename T::SmemLayoutVt{});
            return wg_mma_pv.partition_fragment_B(local_tile(sVz, make_shape(Int<T::kHeadDimV>{}, Int<CW>{}), make_coord(_0{}, _)));  // (frag, N, 2, c, st)
        };
        using OpK = decltype(mk_K(0)); using OpV = decltype(mk_V(0));
        AccC accC[4];                                           // S chunk buffers: chunk k in accC[k % 4]
        auto pc_proto = make_tensor_like<Element>(make_tensor(accC[0].data(), flash::convert_layout_acc_Aregs<typename T::TiledMmaPV>(accC[0].layout())));
        decltype(pc_proto) PCb[2];                              // bf16 P of chunk k in PCb[k % 2]
        static_assert(decltype(size<2>(pc_proto))::value == kNKBc);
        float nm[R][kNRows];                                    // per-row exponent shift (log2 units), fixed after the seed
        float const c_l2 = params.scale_log2e;
        int const K_total = 4 * n_live;
        // this thread's bias fragments: 16 bytes (8 bf16: rows mi, columns 2*nj + e of half c) at frag_base + c*kFragHalf + mi*512 in a stage
        constexpr uint32_t kFragHalf = T::kBytesBias / 2;
        uint32_t const frag_base = cute::cast_smem_ptr_to_uint(shared.smem_bias.data()) + uint32_t((cwg * 4 + (thread_idx % 128) / 32) * 2 * 32 + lane) * 16u;
        uint32_t bw[2][kNRows][4];                              // [half c][row mi][4 words]: bias of this thread's fragment of the current tile
        auto load_bias_half = [&](auto cc, int bslot) __attribute__((always_inline)) {
            constexpr int c = decltype(cc)::value;
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                uint32_t const a = frag_base + uint32_t(bslot) * T::kBytesBias + uint32_t(c) * kFragHalf + uint32_t(mi) * 512u;
                asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(bw[c][mi][0]), "=r"(bw[c][mi][1]), "=r"(bw[c][mi][2]), "=r"(bw[c][mi][3]) : "r"(a));
            }
        };
        auto bias_f = [&](int c, int mi, int ni) __attribute__((always_inline)) {       // fp32 value of bf16 fragment element ni of row mi
            uint32_t const w = bw[c][mi][ni / 2];
            return __uint_as_float((ni % 2) ? (w & 0xffff0000u) : (w << 16));
        };
        float const inv_scale = kLog2e / c_l2;                  // 1/scale (bias enters the accumulator as bias/scale; x = (qk + bias/scale) * c_l2 + nm)
        // (kInitAcc) S accumulator of a chunk <- bias/scale of its (tile slot, half c) fragment, before its QK is issued
        auto init_acc = [&](AccC& acc, auto cc, int bslot) __attribute__((always_inline)) {
            constexpr int c = decltype(cc)::value;
            Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                uint32_t w4[4];
                uint32_t const a = frag_base + uint32_t(bslot) * T::kBytesBias + uint32_t(c) * kFragHalf + uint32_t(mi) * 512u;
                asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(w4[0]), "=r"(w4[1]), "=r"(w4[2]), "=r"(w4[3]) : "r"(a));
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = __uint_as_float((ni % 2) ? (w4[ni / 2] & 0xffff0000u) : (w4[ni / 2] << 16)) * inv_scale; }
            }
        };
        // (kBiasF32) S accumulator of a chunk <- fp32 bias/scale, staged in accumulator order: 64 contiguous bytes per (thread, half): 4 LDS.128
        uint32_t const frag_base32 = cute::cast_smem_ptr_to_uint(shared.smem_bias.data()) + uint32_t((cwg * 4 + (thread_idx % 128) / 32) * 32 + lane) * 64u;
        auto init_acc_f32 = [&](AccC& acc, auto cc, int bslot) __attribute__((always_inline)) {
            constexpr int c = decltype(cc)::value;
            static_assert(decltype(size(acc))::value == 16);
            uint32_t const a = frag_base32 + uint32_t(bslot) * T::kBytesBias + uint32_t(c) * kFragHalf;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(acc(4 * i)), "=f"(acc(4 * i + 1)), "=f"(acc(4 * i + 2)), "=f"(acc(4 * i + 3)) : "r"(a + 16u * i) : "memory");
            }
        };
        auto issue_qk = [&](AccC& acc, OpK const& tK, auto rc, auto cc, auto stc) __attribute__((always_inline)) {
            constexpr int r = decltype(rc)::value, c = decltype(cc)::value, st = decltype(stc)::value;
            warpgroup_fence_operand(acc);
            warpgroup_arrive();
            tiled_mma_qkc.accumulate_ = T::kInitAcc ? GMMA::ScaleOut::One : GMMA::ScaleOut::Zero;
            #pragma unroll
            for (int kb = 0; kb < 2; ++kb) { cute::gemm(tiled_mma_qkc, tSrQc(_, _, kb, Int<r>{}), tK(_, _, kb, Int<c>{}, Int<st>{}), acc); tiled_mma_qkc.accumulate_ = GMMA::ScaleOut::One; }
            warpgroup_commit_batch();                                   // no fence: the accumulator is in flight until a later wait
        };
        auto issue_pvc = [&](decltype(pc_proto)& tP, OpV const& tV, auto rc, auto cc, auto stc) __attribute__((always_inline)) {   // no fence on acc_o: consecutive PVs chain
            constexpr int r = decltype(rc)::value, c = decltype(cc)::value, st = decltype(stc)::value;
            warpgroup_fence_operand(tP);
            warpgroup_arrive();
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
            #pragma unroll
            for (int kb = 0; kb < kNKBc; ++kb) { cute::gemm(tiled_mma_pv, tP(_, _, kb), tV(_, _, kb, Int<c>{}, Int<st>{}), acc_o[r]); }
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
        auto pp_sync = [&]() __attribute__((always_inline)) { if constexpr (T::kPingPong) { warp_scheduler_barrier_sync<T>(); } };
        auto pp_arrive = [&]() __attribute__((always_inline)) { if constexpr (T::kPingPong) { warp_scheduler_barrier_arrive<T>(); } };
        auto exp_core = [&](AccC& acc, auto rc, auto cc) __attribute__((always_inline)) {   // p = 2^(S*c_l2 + bias*log2e + nm[r]) in place
            constexpr int r = decltype(rc)::value, c = decltype(cc)::value;
            Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {           // all exponent arguments first (FFMA pipe), then the MUFU stream: no ex2 waits on its own FFMA
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) {
                    if constexpr (T::kInitAcc || (T::kDebug & 0x2000000)) { s_rc(mi, ni) = fmaf(s_rc(mi, ni), c_l2, nm[r][mi]); }   // bias already in S (or diagnosis: no bias)
                    else { s_rc(mi, ni) = fmaf(s_rc(mi, ni), c_l2, fmaf(bias_f(c, mi, ni), kLog2e, nm[r][mi])); }
                }
            }
            if constexpr (T::kInitAcc) { warpgroup_fence_operand(acc); pp_sync(); }   // token section = the MUFU stream only
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(s_rc(mi, ni)); }
            }
            warpgroup_fence_operand(acc);        // the exponentials stay before the next body's wgmma wait
            if constexpr (T::kInitAcc) { pp_arrive(); }
        };
        // prologue-only pieces: logits of a landed chunk in log2 units (in place), the exact row max of tile 0 -> shift, seeded exponentials
        auto affine = [&](AccC& acc, auto cc) __attribute__((always_inline)) {
            constexpr int c = decltype(cc)::value;
            Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) {
                    if constexpr (T::kInitAcc) { s_rc(mi, ni) = s_rc(mi, ni) * c_l2; }
                    else { s_rc(mi, ni) = fmaf(s_rc(mi, ni), c_l2, bias_f(c, mi, ni) * kLog2e); }
                }
            }
        };
        auto seed_row = [&](AccC& a0, AccC& a1, int r) __attribute__((always_inline)) {          // nm[r] <- -(max over tile 0 of row r's logits)
            Tensor s0 = make_tensor(a0.data(), flash::convert_layout_acc_rowcol(a0.layout()));
            Tensor s1 = make_tensor(a1.data(), flash::convert_layout_acc_rowcol(a1.layout()));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                float mx = -INFINITY;
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { mx = max(mx, max(s0(mi, ni), s1(mi, ni))); }
                mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                nm[r][mi] = (mx == -INFINITY) ? 0.f : -mx;
            }
        };
        auto exp_seeded = [&](AccC& acc, int r) __attribute__((always_inline)) {                  // p = 2^(x + nm) of a chunk holding logits x
            Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
            pp_sync();
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(s_rc(mi, ni) + nm[r][mi]); }
            }
            warpgroup_fence_operand(acc);
            pp_arrive();
        };
        // at a drained period boundary: rare power-of-two renormalisation of (O, l, nm) of a row whose sum got large; `pend` holds the
        // exponentials of the one chunk (row r_pend) that is not yet packed, rescaled with its row
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
        // K/V stage of (tile t of a period, row r): the live tile jl = 1 + 3p + t sits in ring slot (1 + t) % 3; tile 0 == t = 2 of period -1
        auto st_of = [](auto tc, auto rc) { return Int<2 * ((1 + decltype(tc)::value) % 3) + decltype(rc)::value>{}; };

        // ---- body for chunk dd (0..11) of a period, k = 4 + 12p + dd:
        //      [bias fragments of (tile, half) at the tile's first row] | wait<1> (all but PV(k-1): QK(k+1), PV(k-2) retired) | releases |
        //      waits for chunk k+2's tile data | QK(k+2) -> accC[(k+2)%4] | E(k) | pack(k-1) -> PCb[(k-1)%2] | PV(k-1) LAST
        auto body = [&](auto ddc, int jbase, OpK const& tK, OpV const& tV) __attribute__((always_inline)) {
            constexpr int dd = decltype(ddc)::value;
            constexpr int bq = dd & 3, bn = (dd + 2) & 3, bp = (dd + 3) & 3;                       // S(k), S(k+2), S(k-1) buffers
            constexpr int pb = (dd + 1) & 1;                                                       // P buffer of chunk k-1
            constexpr int t = dd / 4, r = (dd >> 1) & 1, c = dd & 1;
            constexpr int dd2 = dd + 2, t2 = dd2 / 4, r2 = (dd2 >> 1) & 1, c2 = dd2 & 1;            // chunk k+2 (t2 == 3: next period's tile 0)
            constexpr int ddp = (dd + 11) % 12, tp = ddp / 4, rp = (ddp >> 1) & 1, cpv = ddp & 1;    // chunk k-1 (dd == 0: previous period's tile 2)
            constexpr int ddo = (dd + 9) % 12, to = ddo / 4, ro = (ddo >> 1) & 1, co = ddo & 1;      // chunk k-3 (its PV has retired at this body's wait)
            stamp(1);
            if constexpr (!T::kInitAcc && (r == 0 || T::kBiasReload)) { load_bias_half(Int<c>{}, (1 + t) % 3); }   // this tile's half-c fragments (rows 0 and 1 share them unless kBiasReload)
            int const j2 = jbase + t2;                                                             // live-tile ordinal of chunk k+2
            if constexpr (T::kBiasF32) {                                                           // S(k+2) <- bias/scale (its buffer held S(k-2): packed in body k-1)
                if constexpr ((dd2 & 3) == 0) { if (j2 < n_live) { pipe_bias.wait_full((1 + t2) % 3, uint32_t((j2 / 3) & 1)); } }
                init_acc_f32(accC[bn], Int<c2>{}, (1 + t2) % 3);
            }
            if constexpr (dd != 0) { warpgroup_wait<1>(); }
            warpgroup_fence_operand(accC[bq]);                                                     // S(k) reads stay after the wait
            warpgroup_fence_operand(PCb[pb]);                                                      // the pack's writes stay after the wait (PV(k-3) read this buffer)
            if constexpr (co == 1) { pipe_kv.release(decltype(st_of(Int<to>{}, Int<ro>{}))::value, wg_leader); }   // chunk k-3 = half 1 of its stage: both its PVs retired
            if constexpr ((dd & 3) == 0) { __syncwarp(); pipe_bias.release(t % 3, lane == 0); }    // the previous tile's bias stage (slot t % 3): its fragments are consumed
            if (j2 < n_live) {
                if constexpr ((dd2 & 3) == 0 && !T::kBiasF32) { pipe_bias.wait_full((1 + t2) % 3, uint32_t((j2 / 3) & 1)); }
                if constexpr (c2 == 0) { pipe_kv.wait_full(decltype(st_of(Int<t2 % 3>{}, Int<r2>{}))::value, uint32_t((j2 / 3) & 1)); }
            }
            if constexpr (T::kInitAcc && !T::kBiasF32) { init_acc(accC[bn], Int<c2>{}, (1 + t2) % 3); }   // S(k+2) <- bias/scale; the QK accumulates onto it
            issue_qk(accC[bn], tK, Int<r2>{}, Int<c2>{}, st_of(Int<t2 % 3>{}, Int<r2>{}));      // phantom past the end: stale smem, result unused
            stamp(5);
            if constexpr (!T::kInitAcc) { pp_sync(); }
            stamp(7);
            exp_core(accC[bq], Int<r>{}, Int<c>{});
            if constexpr (!T::kInitAcc) { pp_arrive(); }
            stamp(6);
            pack_chunk(accC[bp], PCb[pb]);                                                         // previous chunk's exponentials (no MUFU->F2FP adjacency)
            issue_pvc(PCb[pb], tV, Int<rp>{}, Int<cpv>{}, st_of(Int<tp>{}, Int<rp>{}));
            stamp(2);
        };
        auto drain = [&]() __attribute__((always_inline)) {           // after body 3 / 7 / 11: the exponentiated, unpacked chunk (row 1) is in accC[3]
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(accC[3]);
            warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]);
            l_check(accC[3], 1);
        };

        // ---- prologue: tile 0 (chunks 0..3 in accC[0..3]; K/V stages 0 / 1; bias slot 0; all phases 0), exact per-row max as the
        //      shift; ends DRAINED in the state body 0 expects: S(4), S(5) landed in accC[0], accC[1]; chunk 3 exp'd in accC[3], unpacked.
        OpK const tK0 = mk_K(params.zero); OpV const tV0 = mk_V(params.zero);
        pipe_bias.wait_full(0, 0); pipe_kv.wait_full(0, 0);
        if constexpr (T::kBiasF32) { init_acc_f32(accC[0], _0{}, 0); init_acc_f32(accC[1], _1{}, 0); } else if constexpr (T::kInitAcc) { init_acc(accC[0], _0{}, 0); init_acc(accC[1], _1{}, 0); } else { load_bias_half(_0{}, 0); load_bias_half(_1{}, 0); }
        issue_qk(accC[0], tK0, _0{}, _0{}, _0{}); issue_qk(accC[1], tK0, _0{}, _1{}, _0{});
        warpgroup_wait<0>();
        warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]);
        affine(accC[0], _0{}); affine(accC[1], _1{});
        seed_row(accC[0], accC[1], 0);
        pipe_kv.wait_full(1, 0);
        if constexpr (T::kBiasF32) { init_acc_f32(accC[2], _0{}, 0); init_acc_f32(accC[3], _1{}, 0); } else if constexpr (T::kInitAcc) { init_acc(accC[2], _0{}, 0); init_acc(accC[3], _1{}, 0); }
        issue_qk(accC[2], tK0, _1{}, _0{}, _1{}); issue_qk(accC[3], tK0, _1{}, _1{}, _1{});      // QK(2), QK(3) under E(0), E(1)
        exp_seeded(accC[0], 0); pack_chunk(accC[0], PCb[0]); issue_pvc(PCb[0], tV0, _0{}, _0{}, _0{});
        exp_seeded(accC[1], 0); pack_chunk(accC[1], PCb[1]); issue_pvc(PCb[1], tV0, _0{}, _1{}, _0{});
        warpgroup_wait<0>();
        warpgroup_fence_operand(accC[2]); warpgroup_fence_operand(accC[3]); warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]); warpgroup_fence_operand(acc_o[0]);
        affine(accC[2], _0{}); affine(accC[3], _1{});
        seed_row(accC[2], accC[3], 1);
        if (n_live > 1) { pipe_bias.wait_full(1, 0); pipe_kv.wait_full(2, 0); }                    // tile 1 = t 0 of period 0: bias slot 1, stage 2, phase 0
        if constexpr (T::kBiasF32) { init_acc_f32(accC[0], _0{}, 1); init_acc_f32(accC[1], _1{}, 1); } else if constexpr (T::kInitAcc) { init_acc(accC[0], _0{}, 1); init_acc(accC[1], _1{}, 1); }
        issue_qk(accC[0], tK0, _0{}, _0{}, _2{}); issue_qk(accC[1], tK0, _0{}, _1{}, _2{});      // QK(4), QK(5) (phantom when n_live == 1)
        exp_seeded(accC[2], 1); pack_chunk(accC[2], PCb[0]); issue_pvc(PCb[0], tV0, _1{}, _0{}, _1{});
        exp_seeded(accC[3], 1);                                                                    // chunk 3 stays unpacked: body 0 (or the finish) packs it
        warpgroup_wait<0>();                                                                       // loop entry == the drained state body 0 assumes
        warpgroup_fence_operand(accC[0]); warpgroup_fence_operand(accC[1]); warpgroup_fence_operand(accC[2]); warpgroup_fence_operand(accC[3]);
        warpgroup_fence_operand(PCb[0]); warpgroup_fence_operand(PCb[1]); warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]);
        // (K/V stage 0 = row 0 of tile 0 is released by body 0, whose chunk k-3 is chunk 1; stage 1 by body 2 or the finish; bias slot 0 by body 0)

        // ---- the loop: full periods and the partial last one (4 or 8 chunks) share this single copy of the bodies; ends drained
        #pragma unroll 1
        for (int k0 = 4, jbase = 1; k0 < K_total; k0 += 12, jbase += 3) {
            int const left = K_total - k0;                                                         // 4, 8 or >= 12
            int const zoff = params.zero * k0;                                                     // == 0, opaque per period
            OpK const tK = mk_K(zoff); OpV const tV = mk_V(zoff);
            body(Int<0>{}, jbase, tK, tV); body(Int<1>{}, jbase, tK, tV); body(Int<2>{}, jbase, tK, tV);  body(Int<3>{}, jbase, tK, tV);
            if (left == 4) { drain(); break; }
            body(Int<4>{}, jbase, tK, tV); body(Int<5>{}, jbase, tK, tV); body(Int<6>{}, jbase, tK, tV);  body(Int<7>{}, jbase, tK, tV);
            if (left == 8) { drain(); break; }
            body(Int<8>{}, jbase, tK, tV); body(Int<9>{}, jbase, tK, tV); body(Int<10>{}, jbase, tK, tV); body(Int<11>{}, jbase, tK, tV);
            drain();
        }
        {   // the last chunk K-1 (row 1, half 1 of tile n_live-1, ring slot m3 = (n_live-1) % 3) is exp'd in accC[3], unpacked; state: drained
            int const m3 = (n_live - 1) % 3;
            auto finish = [&](auto m3c) __attribute__((always_inline)) {
                constexpr int st0 = 2 * decltype(m3c)::value;
                pack_chunk(accC[3], PCb[1]);
                issue_pvc(PCb[1], tV0, _1{}, _1{}, Int<st0 + 1>{});
                warpgroup_wait<0>();
                warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]);
                pipe_kv.release(st0, wg_leader); pipe_kv.release(st0 + 1, wg_leader);               // the last tile's stages and bias slot (no later body released them)
                __syncwarp(); pipe_bias.release(decltype(m3c)::value, lane == 0);
            };
            if (m3 == 0) { finish(_0{}); } else if (m3 == 1) { finish(_1{}); } else { finish(_2{}); }
        }
    } else {
    // (compile-time ring slots via a kRing-way unrolled loop were tried: ptxas then serializes the wgmmas (C7512) and spills)
    int npend = 0;
    for (int jl = 0; jl < n_live; ++jl) { tile(jl, npend); }
    drain((n_live - 1) % T::kRing, npend);
    }

    // ---- epilogue: O / l -> bf16 -> global ---------------------------------------------------------------------------
    stamp(19);
    if constexpr (T::kMaxFree) {
        if (params.counters != nullptr && lane == 0 && n_exact_steps != 0) { atomicAdd(params.counters, (unsigned long long)n_exact_steps); }
    }
    // (block coordinates re-read from blockIdx here so they need not stay live through the main loop)
    int const e_bh = blockIdx.z, e_b = e_bh / params.H, e_h = e_bh % params.H, e_qtile = blockIdx.x, e_i0 = blockIdx.y * R;
    int const e_lane = threadIdx.x % 32, e_tidx = threadIdx.x - 128;
    int const e_row0 = (e_tidx / 128) * 64 + ((e_tidx % 128) / 32) * 16 + e_lane / 4, e_col0 = 2 * (e_lane % 4);
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        int const i = e_i0 + r;
        if (i >= params.N || e_qtile >= params.n_qtiles) { continue; }
        Tensor o_rc = make_tensor(acc_o[r].data(), flash::convert_layout_acc_rowcol(acc_o[r].layout()));
        Element* obase = params.out + (int64_t)e_b * params.so_b + (int64_t)i * params.so_n + (int64_t)e_h * params.so_h;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            float const l = o_rc(mi, kColL);
            float const inv = l > 0.f ? 1.f / l : 0.f;
            int const q = e_qtile * kBlockM + e_row0 + 8 * mi;
            if (q < params.S) {
                Element* orow = obase + (int64_t)q * params.so_s;
                #pragma unroll
                for (int ni = 0; ni < kColL; ni += 2) {
                    int const d = e_col0 + 8 * (ni / 2);
                    __nv_bfloat162 v2 = __floats2bfloat162_rn(o_rc(mi, ni) * inv, o_rc(mi, ni + 1) * inv);
                    *reinterpret_cast<__nv_bfloat162*>(orow + d) = v2;
                }
            }
        }
    }
    stamp(20);
}


#ifdef TRIATTN_MICROBENCH
// ------------------------------------------------------------------------------------------------------------------
// F4 microkernel: raw wgmma throughput of this kernel's GEMM shapes with 2 consumer warpgroups per SM issuing concurrently.
// kind 0: S GEMMs (m64n64k16 SS x2 per row, 2 rows per group); 1: PV GEMMs (m64n40k16 RS x4 per row, 2 rows per group);
// 2: one pair-step's mix (S x4 then PV x8); 3: PV with N=32 (m64n32k16 RS x4 per row x2). Each WG: iters groups, commit+wait<0> per
// group (kPipe groups kept in flight). out[blockIdx.x*2+wg] = clocks for the loop.
template <class T, int kKind, int kPipe>
__global__ void __launch_bounds__(256, 1) wgmma_bench_kernel(int iters, unsigned long long* out) {
    using namespace cute;
    using Element = typename T::Element;
    extern __shared__ char smem_raw[];
    typename T::SharedStorage& shared = *reinterpret_cast<typename T::SharedStorage*>(smem_raw);
    int const tid = threadIdx.x, cwg = tid / 128, thread_idx = tid;
    for (int i = tid; i < (int)sizeof(typename T::SharedStorage) / 4; i += 256) { reinterpret_cast<uint32_t*>(smem_raw)[i] = 0x3c003c00u; }
    __syncthreads();
    typename T::TiledMmaQK tiled_mma_qk;
    typename T::TiledMmaPV tiled_mma_pv;
    auto wg_layout = make_layout(make_shape(Int<T::kNumMmaWG>{}), make_stride(Int<128>{}));
    auto wg_mma_qk = tiled_mma_qk.get_slice(wg_layout(cwg));
    auto wg_mma_pv = tiled_mma_pv.get_slice(wg_layout(cwg));
    Tensor sQ = make_tensor(make_smem_ptr(shared.smem_q.data()), typename T::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared.smem_k.data()), typename T::SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(shared.smem_v.data()), typename T::SmemLayoutVt{});
    Tensor tSrQ = wg_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = wg_mma_qk.partition_fragment_B(sK);
    Tensor tOrV = wg_mma_pv.partition_fragment_B(sVt);
    using AccS = decltype(partition_fragment_C(tiled_mma_qk, make_shape(Int<T::kBlockM>{}, Int<T::kBlockN>{})));
    using AccO = decltype(partition_fragment_C(tiled_mma_pv, make_shape(Int<T::kBlockM>{}, Int<T::kHeadDimV>{})));
    using TiledMma32 = typename T::TiledMmaPV32;
    TiledMma32 tiled_mma_32; auto wg_mma_32 = tiled_mma_32.get_slice(wg_layout(cwg));
    Tensor sV32 = make_tensor(make_smem_ptr(shared.smem_v.data()), typename T::SmemLayoutVload1{});
    Tensor tOrV32 = wg_mma_32.partition_fragment_B(sV32);
    using AccO32 = decltype(partition_fragment_C(tiled_mma_32, make_shape(Int<T::kBlockM>{}, Int<T::kHeadDim>{})));
    AccS acc_s[2]; AccO acc_o[2]; AccO32 acc_32[2];
    auto tOrP_a = make_tensor_like<Element>(make_tensor(acc_s[0].data(), flash::convert_layout_acc_Aregs<typename T::TiledMmaPV>(acc_s[0].layout())));
    auto tOrP_b = make_tensor_like<Element>(tOrP_a);
    clear(acc_o[0]); clear(acc_o[1]); clear(acc_32[0]); clear(acc_32[1]); clear(tOrP_a); clear(tOrP_b);
    warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(acc_32[0]); warpgroup_fence_operand(acc_32[1]);
    auto kloop = [&](auto& mma, auto const& tA, auto const& tB, auto& tC, bool zero) {
        mma.accumulate_ = zero ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
        CUTE_UNROLL
        for (int kb = 0; kb < size<2>(tA); ++kb) { cute::gemm(mma, tA(_, _, kb), tB(_, _, kb), tC); mma.accumulate_ = GMMA::ScaleOut::One; }
    };
    auto group = [&](int it) {
        warpgroup_fence_operand(acc_s[0]); warpgroup_fence_operand(acc_s[1]);
        warpgroup_arrive();
        if constexpr (kKind == 0 || kKind == 2) {
            { auto tQ = tSrQ(_, _, _, _0{}); auto tK = tSrK(_, _, _, 0); kloop(tiled_mma_qk, tQ, tK, acc_s[0], true); }
            { auto tQ = tSrQ(_, _, _, _1{}); auto tK = tSrK(_, _, _, 1); kloop(tiled_mma_qk, tQ, tK, acc_s[1], true); }
        }
        if constexpr (kKind == 1 || kKind == 2) {
            { auto tV = tOrV(_, _, _, 0); kloop(tiled_mma_pv, tOrP_a, tV, acc_o[0], false); }
            { auto tV = tOrV(_, _, _, 1); kloop(tiled_mma_pv, tOrP_b, tV, acc_o[1], false); }
        }
        if constexpr (kKind == 3) {
            kloop(tiled_mma_32, tOrP_a, tOrV32, acc_32[0], false);
            kloop(tiled_mma_32, tOrP_b, tOrV32, acc_32[1], false);
        }
        warpgroup_commit_batch();
        warpgroup_fence_operand(acc_s[0]); warpgroup_fence_operand(acc_s[1]);
        warpgroup_fence_operand(acc_o[0]); warpgroup_fence_operand(acc_o[1]); warpgroup_fence_operand(acc_32[0]); warpgroup_fence_operand(acc_32[1]);
    };
    __syncthreads();
    unsigned long long t0 = clock64();
    for (int g = 0; g < kPipe; ++g) { group(g); }
    for (int it = kPipe; it < iters; ++it) { warpgroup_wait<kPipe - 1>(); group(it); }
    warpgroup_wait<0>();
    unsigned long long t1 = clock64();
    if (tid % 128 == 0) { out[blockIdx.x * 2 + cwg] = t1 - t0; }
    if (iters < 0) {   // keep results alive
        float acc = 0.f;
        for (int i = 0; i < size(acc_o[0]); ++i) acc += acc_o[0](i) + acc_o[1](i);
        for (int i = 0; i < size(acc_s[0]); ++i) acc += acc_s[0](i) + acc_s[1](i);
        for (int i = 0; i < size(acc_32[0]); ++i) acc += acc_32[0](i) + acc_32[1](i);
        out[0] = (unsigned long long)acc;
    }
}

#endif  // TRIATTN_MICROBENCH

}  // namespace triattn
