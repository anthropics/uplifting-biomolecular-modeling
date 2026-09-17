// PyTorch binding for the cuda_b triangle-attention kernel: configuration dispatch + the odd-q-tile tail launch.
// The per-configuration kernels are explicit instantiations in generated translation units (see triattn_cuda_b.py, which writes
// inst/cfg_*.cu and inst/table.inc from its CONFIGS list); this file only sees their host entry points.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <tuple>

#include "launch.cuh"

namespace triattn_b {

torch::Tensor stage_bias(torch::Tensor const& bias, double scale, int64_t block_n);   // prep.cu

struct CfgEntry { RunFn run; int64_t smem; };
using CfgKey = std::tuple<int, int, int, int, int, int>;   // cq, cr, ring_kv, ring_b, flags, block_n

#include "table.inc"   // generated: declarations of run_cfg_* / smem_cfg_* and `static std::map<CfgKey, CfgEntry> make_table()`

static std::map<CfgKey, CfgEntry> const& table() { static std::map<CfgKey, CfgEntry> t = make_table(); return t; }

static CfgEntry const& lookup(int cq, int cr, int rkv, int rb, int flags, int bn) {
    auto it = table().find(CfgKey{cq, cr, rkv, rb, flags, bn});
    TORCH_CHECK(it != table().end(), "no kernel instantiated for cq=", cq, " cr=", cr, " ring_kv=", rkv, " ring_b=", rb, " flags=", flags, " block_n=", bn);
    return it->second;
}

int64_t smem_bytes(int64_t cq, int64_t cr, int64_t rkv, int64_t rb, int64_t flags, int64_t bn) { return lookup(cq, cr, rkv, rb, flags, bn).smem; }

void fwd(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& bias,
         c10::optional<torch::Tensor> const& mask, double scale, torch::Tensor& out,
         int64_t cq, int64_t cr, int64_t rkv, int64_t rb, int64_t flags, int64_t bn, c10::optional<torch::Tensor> const& trace) {
    c10::cuda::CUDAGuard guard(q.device());
    unsigned long long* tr = trace.has_value() ? reinterpret_cast<unsigned long long*>(trace->data_ptr<int64_t>()) : nullptr;
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && (bias.scalar_type() == torch::kFloat32 || bias.scalar_type() == torch::kBFloat16), "q/k/v bf16; bias = staged fp32 (stage_bias) or raw bf16 (stream kernels)");
    int const S = q.size(3);
    int const n_qtiles = (S + 127) / 128;
    if (cq == 2) {
        // clusters pair adjacent q-tiles: the even part runs with cluster x = 2, a last odd q-tile as its own cluster-x = 1 launch
        int const n_even = n_qtiles & ~1;
        if (n_even > 0) { FwdArgs a{q, k, v, bias, mask, scale, out, 0, n_even, tr}; lookup(2, cr, rkv, rb, flags, bn).run(a); }
        if (n_qtiles & 1) { FwdArgs a{q, k, v, bias, mask, scale, out, n_even, 1, tr}; lookup(1, cr, rkv, rb, flags, bn).run(a); }
    } else {
        FwdArgs a{q, k, v, bias, mask, scale, out, 0, n_qtiles, tr}; lookup(1, cr, rkv, rb, flags, bn).run(a);
    }
}

}  // namespace triattn_b

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &triattn_b::fwd, "triangle attention forward, cuda_b corner (sm_90a)");
    m.def("smem_bytes", &triattn_b::smem_bytes);
    m.def("stage_bias", &triattn_b::stage_bias, "pair bias -> fragment-order fp32 staging (bias / scale)");
}
