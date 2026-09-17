// triattn_xla launcher: an XLA-FFI handler library that (a) launches PRE-COMPILED GPU kernels (cubins produced ahead of time by the
// Triton compiler, shipped under ../bin/k2b/) through the CUDA driver API, and (b) forwards one typed call to the companion CUDA
// library (../bin/cuda/, the sm_90a triangle-attention kernel, plain C ABI) -- both on XLA's own stream, with XLA-owned buffers.
//
// Host-only C++: g++ + cuda.h (driver API) + the XLA FFI API headers that ship with jaxlib (xla/ffi/api/*.h, header-only).  The library
// links libcuda only; it does not link jaxlib, libcudart or Python.  One build per XLA-FFI API version (the version is a property of the
// jaxlib headers; see ../manifest.json "launcher" for the jaxlib lines each build serves).
//
// FFI targets (registered from Python with jax.ffi.register_ffi_target, api_version=1):
//   triattn_xla_run       RemainingArgs / RemainingRets / string+int attributes naming the cubin (key + sha256 prefix), kernel, smem, warps, grid and the
//                         parameter recipe: SELF-DESCRIBING -- the cubin is read from the package on first use in any process, nothing is registered per shape,
//                         so executables restored from a persistent compilation cache run in a fresh process.
//   triattn_xla_m1_fwd    RemainingArgs (q, k, v, bias[, mask]) / RemainingRets (out + scratch) / Attr<float>("scale"), Attr<int64>("flags"): the M1 library's forward
//                         (row triattn_native), entry installed through txla_set_m1_entry().
//   triattn_xla_sm80_fwd  (launcher 1.6) row cuda_80: the kernel family's sm_80 member (bin/cuda/sm_80/libtriattn_sm80_xla.so, C ABI triattn_sm80_abi.h).
//   xla_cubin_call        (launcher 1.5) generic: a kernel from a cubin under a NAMED root, by-value struct parameters assembled from a field recipe with
//                         CUtensorMap descriptors encoded at call time, grid optionally clamped to the multiprocessor count -- used by kernels/trimul_xla.
//   xla_cublas_bgemm_nt   (launcher 1.5) X[c] = A[c].B[c]^T on bf16 planes at element offsets inside the call's buffers through the jaxlib-shipped cuBLAS
//                         (strided-batched, fp32 compute, default tensor-op algorithm, caller workspace) on XLA's stream -- kernels/trimul_xla's contraction.
//   triattn_xla_launch    (previous interface, kept for one version) Attr<int64>("spec"): launch a spec registered in THIS process through txla_register().
//   triattn_xla_cuda_fwd  RemainingArgs (q, k, v, bias[, mask]) / RemainingRets (out + scratch [+ lse: flags bit 1]) / Attr<float>("scale"), Attr<int64>("flags"):
//                         the CUDA library's forward, entry point installed through txla_set_cuda_entry().
//
// Derived from the cubin-launch prototype written in this repository for the JAX triangle-attention transfer study (same authorship,
// same licence); changes: per-context module cache (several devices per process), the CUDA-library forward, launch-error text carrying
// the spec label.
#include <cuda.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "xla/ffi/api/ffi.h"
#include "triattn_cuda_abi.h"
#include "triattn_m1_abi.h"
#include "triattn_sm80_abi.h"

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------------------------- portability of the built library
// Built with a recent g++ but loaded by the libstdc++ of older images: the two symbols a GCC >= 11 / >= 13 build would otherwise import
// from libstdc++ (GLIBCXX_3.4.29 / 3.4.32) are defined here, so the library asks libstdc++ for nothing newer than GLIBCXX_3.4.21 (GCC 5).
#include <ios>
#include <cstdlib>
namespace std {
void ios_base_library_init() { static std::ios_base::Init keep; (void)keep; }
void __throw_bad_array_new_length() { std::abort(); }
}  // namespace std
// The FFI headers' diagnostics default-construct a std::stringstream when decoding string attributes; GCC >= 9 headers call a constructor
// symbol that only libstdc++ >= GLIBCXX_3.4.26 exports.  Instantiating it here keeps the floor at GLIBCXX_3.4.21.
#include <sstream>
template std::basic_stringstream<char>::basic_stringstream();

namespace {
enum Kind : int { IN_BUF = 0, OUT_BUF = 1, I32 = 2, I64 = 3, F32 = 4, NULLPTR = 5 };
struct Param { int kind; long long ival; double fval; };
struct CtxFn { CUcontext ctx; CUmodule mod; CUfunction fn; };
struct Spec {
  std::string cubin, name, label;
  int shared = 0, num_warps = 4;
  unsigned grid[3] = {1, 1, 1};
  std::vector<Param> params;
  int n_trailing_null = 0;         // Triton >= 3.4 appends global_scratch (+ profile_scratch) pointer parameters; NULL when the kernel uses none
  std::vector<CtxFn> loaded;       // one module per CUDA context (device) that launched this spec
  unsigned long long launches = 0;
};
std::mutex g_mu;
std::vector<Spec*> g_specs;
triattn_cuda_fwd_fn g_cuda_fwd = nullptr;
triattn_m1_fwd_fn g_m1_fwd = nullptr;
triattn_sm80_fwd_fn g_sm80_fwd = nullptr;            // libtriattn_sm80_xla.so entry (row cuda_80), installed at registration                // libtriattn_m1_xla.so entry (row triattn_native), installed at registration
std::string g_root;                               // the package directory (…/kernels/triattn_xla): set once per process by txla_set_root() when the FFI targets are registered
struct RunSpec {                                  // one parsed `triattn_xla_run` attribute set (self-describing launch: everything travels in the custom call)
  std::string key, kname, label, sha;
  int shared = 0, warps = 4, nnull = 0;
  std::vector<Param> params;
  std::string cubin;                              // file bytes, read on first use in this process
  std::vector<CtxFn> loaded;
  unsigned long long launches = 0;
};
std::map<std::string, RunSpec*> g_run;            // parse cache: attribute tuple -> spec
triattn_cuda_fwd_fn g_cuda_fwd_lse = nullptr;     // the lse library's entry (flags bit 1: rets carry an extra fp32 [B,N,H,S] log-sum-exp buffer)

std::string cu_err(const char* what, CUresult r) {
  const char* s = nullptr; cuGetErrorString(r, &s);
  return std::string(what) + ": " + (s ? s : "?") + " (" + std::to_string(static_cast<int>(r)) + ")";
}
}  // namespace

extern "C" {
// Register one launch spec; returns its id (>= 0) or -1.
int txla_register(const char* cubin, size_t cubin_len, const char* name, const char* label, int shared, int num_warps, int gx, int gy, int gz,
                  int nparams, const int* kinds, const long long* ivals, const double* fvals, int n_trailing_null) {
  if (!cubin || !name || nparams < 0) return -1;
  auto* s = new Spec();
  s->cubin.assign(cubin, cubin_len); s->name = name; s->label = label ? label : ""; s->shared = shared; s->num_warps = num_warps;
  s->grid[0] = gx; s->grid[1] = gy; s->grid[2] = gz;
  for (int i = 0; i < nparams; ++i) s->params.push_back(Param{kinds[i], ivals[i], fvals[i]});
  s->n_trailing_null = n_trailing_null;
  std::lock_guard<std::mutex> lk(g_mu);
  g_specs.push_back(s);
  return static_cast<int>(g_specs.size()) - 1;
}
unsigned long long txla_launch_count(int id) {
  std::lock_guard<std::mutex> lk(g_mu);
  return (id >= 0 && id < static_cast<int>(g_specs.size())) ? g_specs[id]->launches : 0ULL;
}
int txla_num_specs() { std::lock_guard<std::mutex> lk(g_mu); return static_cast<int>(g_specs.size()); }
void txla_set_cuda_entry(void* fn) { std::lock_guard<std::mutex> lk(g_mu); g_cuda_fwd = reinterpret_cast<triattn_cuda_fwd_fn>(fn); }
void txla_set_cuda_lse_entry(void* fn) { std::lock_guard<std::mutex> lk(g_mu); g_cuda_fwd_lse = reinterpret_cast<triattn_cuda_fwd_fn>(fn); }
// The package directory: cubins are read from <root>/bin/k2b/<key>.cubin and the CUDA libraries from <root>/bin/cuda/sm_90a/ on first use in a
// process -- the launch needs no per-process registration beyond the FFI targets themselves, so an XLA executable that was serialized by a
// persistent compilation cache runs when it is loaded in a fresh process.
void txla_set_sm80_entry(void* fn) { std::lock_guard<std::mutex> lk(g_mu); g_sm80_fwd = reinterpret_cast<triattn_sm80_fwd_fn>(fn); }
void txla_set_m1_entry(void* fn) { std::lock_guard<std::mutex> lk(g_mu); g_m1_fwd = reinterpret_cast<triattn_m1_fwd_fn>(fn); }
std::map<std::string, std::string> g_roots;            // named package roots for the generic cubin target (xla_cubin_call): root name -> directory
void txla_set_cublas(void* create, void* set_stream, void* set_ws, void* bgemm, const char* origin);   // the cuBLAS entry points, resolved by the Python side (ctypes) from the library jaxlib ships
const char* txla_cublas_state(void);
void txla_add_root(const char* name, const char* dir) { std::lock_guard<std::mutex> lk(g_mu); if (name && dir) g_roots[name] = dir; }
void txla_set_root(const char* root) { std::lock_guard<std::mutex> lk(g_mu); g_root = root ? root : ""; }
const char* txla_root() { return g_root.c_str(); }
unsigned long long txla_run_launches(const char* key) {          // launches of every run spec whose cubin key matches (status / tests)
  std::lock_guard<std::mutex> lk(g_mu); unsigned long long n = 0;
  for (auto& kv : g_run) if (kv.second->key == key) n += kv.second->launches;
  return n;
}
int txla_abi_version() { return TRIATTN_CUDA_ABI_VERSION; }
const char* txla_ffi_api_version() {
#if defined(XLA_FFI_API_MAJOR) && defined(XLA_FFI_API_MINOR)
#define TXLA_STR2(x) #x
#define TXLA_STR(x) TXLA_STR2(x)
  return TXLA_STR(XLA_FFI_API_MAJOR) "." TXLA_STR(XLA_FFI_API_MINOR);
#else
  return "unknown";
#endif
}
}  // extern "C"

// ------------------------------------------------------------------------------------------------------- cubin launch handler
static ffi::Error LaunchImpl(CUstream stream, ffi::RemainingArgs args, ffi::RemainingRets rets, int64_t spec_id) {
  Spec* s = nullptr;
  CUfunction fn = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (spec_id < 0 || spec_id >= static_cast<int64_t>(g_specs.size()))
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_launch: unknown spec id " + std::to_string(spec_id));
    s = g_specs[spec_id];
    CUcontext ctx = nullptr;
    CUresult rc = cuCtxGetCurrent(&ctx);
    if (rc != CUDA_SUCCESS || ctx == nullptr) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuCtxGetCurrent", rc) + " (no current CUDA context) spec=" + s->label);
    for (const CtxFn& cf : s->loaded) if (cf.ctx == ctx) { fn = cf.fn; break; }
    if (fn == nullptr) {          // first launch in this context: load the cubin
      CtxFn cf; cf.ctx = ctx; cf.mod = nullptr; cf.fn = nullptr;
      rc = cuModuleLoadData(&cf.mod, s->cubin.data());
      if (rc != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuModuleLoadData", rc) + " kernel=" + s->name + " spec=" + s->label);
      rc = cuModuleGetFunction(&cf.fn, cf.mod, s->name.c_str());
      if (rc != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuModuleGetFunction", rc) + " kernel=" + s->name + " spec=" + s->label);
      if (s->shared > 49152) {
        rc = cuFuncSetAttribute(cf.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, s->shared);
        if (rc != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuFuncSetAttribute(max dynamic smem)", rc) + " shared=" + std::to_string(s->shared) + " spec=" + s->label);
        cuFuncSetCacheConfig(cf.fn, CU_FUNC_CACHE_PREFER_SHARED);
      }
      s->loaded.push_back(cf);
      fn = cf.fn;
    }
    s->launches++;
  }
  const size_t n = s->params.size() + s->n_trailing_null;
  std::vector<void*> ptrs; ptrs.reserve(n + 1);
  std::vector<int32_t> i32s; i32s.reserve(n + 1);
  std::vector<int64_t> i64s; i64s.reserve(n + 1);
  std::vector<float> f32s; f32s.reserve(n + 1);
  std::vector<void*> kparams; kparams.reserve(n + 1);
  for (const Param& p : s->params) {
    switch (p.kind) {
      case IN_BUF: {
        auto b = args.get<ffi::AnyBuffer>(static_cast<size_t>(p.ival));
        if (b.has_error()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_launch: input buffer index " + std::to_string(p.ival) + " out of range, spec=" + s->label);
        ptrs.push_back(b.value().untyped_data()); kparams.push_back(&ptrs.back()); break;
      }
      case OUT_BUF: {
        auto b = rets.get<ffi::AnyBuffer>(static_cast<size_t>(p.ival));
        if (b.has_error()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_launch: output buffer index " + std::to_string(p.ival) + " out of range, spec=" + s->label);
        ptrs.push_back(b.value()->untyped_data()); kparams.push_back(&ptrs.back()); break;
      }
      case I32: i32s.push_back(static_cast<int32_t>(p.ival)); kparams.push_back(&i32s.back()); break;
      case I64: i64s.push_back(static_cast<int64_t>(p.ival)); kparams.push_back(&i64s.back()); break;
      case F32: f32s.push_back(static_cast<float>(p.fval)); kparams.push_back(&f32s.back()); break;
      case NULLPTR: ptrs.push_back(nullptr); kparams.push_back(&ptrs.back()); break;
      default: return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_launch: bad parameter kind in spec " + s->label);
    }
  }
  for (int i = 0; i < s->n_trailing_null; ++i) { ptrs.push_back(nullptr); kparams.push_back(&ptrs.back()); }
  CUresult r = cuLaunchKernel(fn, s->grid[0], s->grid[1], s->grid[2], 32u * static_cast<unsigned>(s->num_warps), 1u, 1u,
                              static_cast<unsigned>(s->shared), stream, kparams.data(), nullptr);
  if (r != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuLaunchKernel", r) + " kernel=" + s->name + " spec=" + s->label);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TriattnXlaLaunch, LaunchImpl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<CUstream>>().RemainingArgs().RemainingRets().Attr<int64_t>("spec"));

// ------------------------------------------------------------------------------------------- spec-free launch handler (triattn_xla_run)
// Attributes (all in the custom call, so the serialized HLO describes the launch completely):
//   key    "sm_90/k2b_fwd_bf16_d32_mask1_b161_s16"  -> <root>/bin/k2b/<key>.cubin        sha    first 16 hex digits of that file's sha256 at trace time
//   kname  the kernel's entry name                      shared/warps/nnull  dynamic smem bytes / warps per block / trailing null pointer params
//   gx gy gz  the grid                                  params  the kernel parameter recipe: comma-separated  bN (input N) | oN (output N) | i<int32>
//   label  text for errors                                      | l<int64> | f<C99 hex float> | n (nullptr)
namespace {
std::string hex_sha256_prefix16(const std::string& bytes);
bool parse_int(std::string_view t, long long* v) {           // [-]digits
  if (t.empty()) return false;
  size_t i = 0; bool neg = false; if (t[0] == '-') { neg = true; i = 1; } else if (t[0] == '+') i = 1;
  if (i >= t.size()) return false;
  long long x = 0; for (; i < t.size(); ++i) { if (t[i] < '0' || t[i] > '9') return false; x = x * 10 + (t[i] - '0'); }
  *v = neg ? -x : x; return true;
}
int hexval(char c) { return (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1; }
bool parse_hexfloat(std::string_view t, double* v) {      // Python float.hex(): [-]0x1.hhhhp[+-]d | [-]0x0.0p+0 | inf | -inf | nan  (exact)
  if (t == "inf") { *v = HUGE_VAL; return true; } if (t == "-inf") { *v = -HUGE_VAL; return true; } if (t == "nan") { *v = NAN; return true; }
  size_t i = 0; bool neg = false; if (!t.empty() && t[0] == '-') { neg = true; i = 1; }
  if (t.size() < i + 3 || t[i] != '0' || (t[i + 1] != 'x' && t[i + 1] != 'X')) return false;
  i += 2; double m = 0.0; int fracdig = 0; bool dot = false, any = false;
  for (; i < t.size() && t[i] != 'p' && t[i] != 'P'; ++i) {
    if (t[i] == '.') { if (dot) return false; dot = true; continue; }
    const int h = hexval(t[i]); if (h < 0) return false; m = m * 16.0 + h; if (dot) fracdig++; any = true;
  }
  if (!any || i >= t.size()) return false;
  ++i; long long e = 0; if (!parse_int(t.substr(i), &e)) return false;
  *v = std::ldexp(m, static_cast<int>(e) - 4 * fracdig); if (neg) *v = -*v; return true;
}
bool parse_params(std::string_view rec, std::vector<Param>* out, std::string* err) {
  size_t i = 0;
  while (i < rec.size()) {
    size_t j = rec.find(',', i); if (j == std::string_view::npos) j = rec.size();
    std::string_view tok = rec.substr(i, j - i); i = j + 1;
    if (tok.empty()) continue;
    Param p{NULLPTR, 0, 0.0};
    const char c = tok[0]; const std::string_view body = tok.substr(1);
    bool ok = true;
    if (c == 'b') { p.kind = IN_BUF; ok = parse_int(body, &p.ival); }
    else if (c == 'o') { p.kind = OUT_BUF; ok = parse_int(body, &p.ival); }
    else if (c == 'i') { p.kind = I32; ok = parse_int(body, &p.ival); }
    else if (c == 'l') { p.kind = I64; ok = parse_int(body, &p.ival); }
    else if (c == 'f') { p.kind = F32; ok = parse_hexfloat(body, &p.fval); }
    else if (c == 'n') { p.kind = NULLPTR; }
    else ok = false;
    if (!ok) { *err = "bad parameter token '" + std::string(tok) + "'"; return false; }
    out->push_back(p);
  }
  return true;
}
// Minimal SHA-256 (public-domain algorithm restated here; used once per cubin per process to compare with the trace-time digest).
struct Sha256 { uint32_t h[8]; uint64_t len = 0; unsigned char buf[64]; size_t nbuf = 0; };
inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
void sha256_block(Sha256& s, const unsigned char* p) {
  static const uint32_t K[64] = {0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 | (uint32_t)p[4*i+2] << 8 | (uint32_t)p[4*i+3];
  for (int i = 16; i < 64; ++i) { uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3), s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10); w[i] = w[i-16] + s0 + w[i-7] + s1; }
  uint32_t a=s.h[0],b=s.h[1],c=s.h[2],d=s.h[3],e=s.h[4],f=s.h[5],g=s.h[6],hh=s.h[7];
  for (int i = 0; i < 64; ++i) { uint32_t S1 = rotr(e,6)^rotr(e,11)^rotr(e,25), ch = (e&f)^(~e&g), t1 = hh+S1+ch+K[i]+w[i], S0 = rotr(a,2)^rotr(a,13)^rotr(a,22), mj = (a&b)^(a&c)^(b&c), t2 = S0+mj;
    hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2; }
  s.h[0]+=a; s.h[1]+=b; s.h[2]+=c; s.h[3]+=d; s.h[4]+=e; s.h[5]+=f; s.h[6]+=g; s.h[7]+=hh;
}
std::string hex_sha256_prefix16(const std::string& bytes) {
  Sha256 s; const uint32_t iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}; std::memcpy(s.h, iv, sizeof(iv));
  const unsigned char* p = reinterpret_cast<const unsigned char*>(bytes.data()); size_t n = bytes.size(); s.len = n;
  while (n >= 64) { sha256_block(s, p); p += 64; n -= 64; }
  unsigned char tail[128]; std::memset(tail, 0, sizeof(tail)); std::memcpy(tail, p, n); tail[n] = 0x80;
  size_t tl = (n < 56) ? 64 : 128; uint64_t bits = s.len * 8; for (int i = 0; i < 8; ++i) tail[tl - 1 - i] = (unsigned char)(bits >> (8 * i));
  sha256_block(s, tail); if (tl == 128) sha256_block(s, tail + 64);
  char out[17]; std::snprintf(out, sizeof(out), "%08x%08x", s.h[0], s.h[1]); return std::string(out);
}
}  // namespace

static ffi::Error RunImpl(CUstream stream, ffi::RemainingArgs args, ffi::RemainingRets rets, std::string_view key, std::string_view sha, std::string_view kname,
                         int64_t shared, int64_t warps, int64_t nnull, int64_t gx, int64_t gy, int64_t gz, std::string_view params, std::string_view label) {
  RunSpec* s = nullptr;
  CUfunction fn = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    std::string ck; ck.reserve(key.size() + params.size() + kname.size() + 64);
    ck.append(key).append("|").append(sha).append("|").append(kname).append("|").append(std::to_string(shared)).append("|").append(std::to_string(warps)).append("|").append(std::to_string(nnull)).append("|").append(params);
    auto it = g_run.find(ck);
    if (it == g_run.end()) {                       // first use of this launch in the process: parse the recipe, read + check the cubin
      if (g_root.empty()) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, "triattn_xla_run: package root not set (the FFI targets are registered by importing opt_core.kernels.triattn_xla, which sets it)");
      auto* rs = new RunSpec(); rs->key.assign(key); rs->kname.assign(kname); rs->label.assign(label); rs->sha.assign(sha);
      rs->shared = static_cast<int>(shared); rs->warps = static_cast<int>(warps); rs->nnull = static_cast<int>(nnull);
      std::string err;
      if (!parse_params(params, &rs->params, &err)) { delete rs; return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_run: " + err + " spec=" + std::string(label)); }
      const std::string path = g_root + "/bin/k2b/" + rs->key + ".cubin";
      std::FILE* f = std::fopen(path.c_str(), "rb");
      if (f == nullptr) { delete rs; return ffi::Error(ffi::ErrorCode::kNotFound, "triattn_xla_run: cubin not found: " + path + " (the program was traced against a package that carried it) spec=" + std::string(label)); }
      std::fseek(f, 0, SEEK_END); const long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
      rs->cubin.resize(sz > 0 ? static_cast<size_t>(sz) : 0);
      const size_t got = sz > 0 ? std::fread(&rs->cubin[0], 1, static_cast<size_t>(sz), f) : 0; std::fclose(f);
      if (got != rs->cubin.size()) { delete rs; return ffi::Error(ffi::ErrorCode::kInternal, "triattn_xla_run: short read of " + path); }
      rs->cubin.push_back('\0');
      const std::string have = hex_sha256_prefix16(rs->cubin.substr(0, rs->cubin.size() - 1));
      if (!rs->sha.empty() && have != rs->sha) { std::string m = "triattn_xla_run: " + path + " has sha256 " + have + "…, the program was traced against " + rs->sha + "… -- re-trace the program against this package; spec=" + std::string(label); delete rs; return ffi::Error(ffi::ErrorCode::kFailedPrecondition, m); }
      it = g_run.emplace(ck, rs).first;
    }
    s = it->second;
    CUcontext ctx = nullptr;
    CUresult rc = cuCtxGetCurrent(&ctx);
    if (rc != CUDA_SUCCESS || ctx == nullptr) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuCtxGetCurrent", rc) + " (no current CUDA context) spec=" + s->label);
    for (const CtxFn& cf : s->loaded) if (cf.ctx == ctx) { fn = cf.fn; break; }
    if (fn == nullptr) {
      CtxFn cf; cf.ctx = ctx; cf.mod = nullptr; cf.fn = nullptr;
      rc = cuModuleLoadData(&cf.mod, s->cubin.data());
      if (rc != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuModuleLoadData", rc) + " kernel=" + s->kname + " spec=" + s->label);
      rc = cuModuleGetFunction(&cf.fn, cf.mod, s->kname.c_str());
      if (rc != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuModuleGetFunction", rc) + " kernel=" + s->kname + " spec=" + s->label);
      if (s->shared > 49152) {
        rc = cuFuncSetAttribute(cf.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, s->shared);
        if (rc != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuFuncSetAttribute(max dynamic smem)", rc) + " shared=" + std::to_string(s->shared) + " spec=" + s->label);
        cuFuncSetCacheConfig(cf.fn, CU_FUNC_CACHE_PREFER_SHARED);
      }
      s->loaded.push_back(cf); fn = cf.fn;
    }
    s->launches++;
  }
  const size_t n = s->params.size() + s->nnull;
  std::vector<void*> ptrs; ptrs.reserve(n + 1);
  std::vector<int32_t> i32s; i32s.reserve(n + 1);
  std::vector<int64_t> i64s; i64s.reserve(n + 1);
  std::vector<float> f32s; f32s.reserve(n + 1);
  std::vector<void*> kparams; kparams.reserve(n + 1);
  for (const Param& p : s->params) {
    switch (p.kind) {
      case IN_BUF: { auto b = args.get<ffi::AnyBuffer>(static_cast<size_t>(p.ival));
        if (b.has_error()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_run: input buffer index " + std::to_string(p.ival) + " out of range, spec=" + s->label);
        ptrs.push_back(b.value().untyped_data()); kparams.push_back(&ptrs.back()); break; }
      case OUT_BUF: { auto b = rets.get<ffi::AnyBuffer>(static_cast<size_t>(p.ival));
        if (b.has_error()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_run: output buffer index " + std::to_string(p.ival) + " out of range, spec=" + s->label);
        ptrs.push_back(b.value()->untyped_data()); kparams.push_back(&ptrs.back()); break; }
      case I32: i32s.push_back(static_cast<int32_t>(p.ival)); kparams.push_back(&i32s.back()); break;
      case I64: i64s.push_back(static_cast<int64_t>(p.ival)); kparams.push_back(&i64s.back()); break;
      case F32: f32s.push_back(static_cast<float>(p.fval)); kparams.push_back(&f32s.back()); break;
      default: ptrs.push_back(nullptr); kparams.push_back(&ptrs.back()); break;
    }
  }
  for (int i = 0; i < s->nnull; ++i) { ptrs.push_back(nullptr); kparams.push_back(&ptrs.back()); }
  CUresult r = cuLaunchKernel(fn, static_cast<unsigned>(gx), static_cast<unsigned>(gy), static_cast<unsigned>(gz), 32u * static_cast<unsigned>(s->warps), 1u, 1u,
                              static_cast<unsigned>(s->shared), stream, kparams.data(), nullptr);
  if (r != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuLaunchKernel", r) + " kernel=" + s->kname + " spec=" + s->label);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TriattnXlaRun, RunImpl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<CUstream>>().RemainingArgs().RemainingRets()
                                  .Attr<std::string_view>("key").Attr<std::string_view>("sha").Attr<std::string_view>("kname")
                                  .Attr<int64_t>("shared").Attr<int64_t>("warps").Attr<int64_t>("nnull").Attr<int64_t>("gx").Attr<int64_t>("gy").Attr<int64_t>("gz")
                                  .Attr<std::string_view>("params").Attr<std::string_view>("label"));

// ------------------------------------------------------------------------------------------------------- CUDA library forward
// args: 0 q [B,N,H,S,D] | 1 k | 2 v (bf16; layout code in flags) | 3 bias [Bb,H,S,S] (f32 / bf16 / f16) | 4 mask [B,N,S] u8 (only when flags&1)
// rets: 0 out (bf16, q's shape) | 1 bias_staged f32 [Bb,H,S128,S64] | 2 keyany u8 [B,S64] | 3 rowkind u8 [B,N] | 4 irr i32 [1+B*YG] | 5 rgflag i32 [B*YG]
//       6 ktend i32 [B] | 7 mtab i32 [B,N,nkt,2] | 8 ctab u8 [B,N,nkt] | 9 ktendr i32 [B,N] | 10 fix i32 [>= fix_elems]
// flags: bit0 has_mask; bits 4..7 layout (0: [B,N,H,S,D], 1: [B,N,S,H,D]); bits 8.. dbg (0 in service)
static ffi::Error CudaFwdImpl(CUstream stream, ffi::RemainingArgs args, ffi::RemainingRets rets, float scale, int64_t flags) {
  const bool has_mask = (flags & 1) != 0;
  const bool want_lse = (flags & 2) != 0;                       // the differentiable row's forward: ret 11 = fp32 [B,N,H,S] log-sum-exp, the lse library's entry
  const int layout = static_cast<int>((flags >> 4) & 0xF);
  const int dbg = static_cast<int>(flags >> 8);
  triattn_cuda_fwd_fn fwd = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    fwd = want_lse ? g_cuda_fwd_lse : g_cuda_fwd;
  }
  if (fwd == nullptr) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, want_lse ? "triattn_xla_cuda_fwd: the lse CUDA library entry is not installed (importing opt_core.kernels.triattn_xla installs it when the FFI targets are registered)"
                                                                                 : "triattn_xla_cuda_fwd: the CUDA library entry is not installed (importing opt_core.kernels.triattn_xla installs it when the FFI targets are registered)");
  const size_t n_rets = want_lse ? 12u : 11u;
  if (args.size() < (has_mask ? 5u : 4u) || rets.size() < n_rets)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_cuda_fwd: expected 4(+mask) args and " + std::to_string(n_rets) + " rets, got " + std::to_string(args.size()) + "/" + std::to_string(rets.size()));
  auto gq = args.get<ffi::AnyBuffer>(0), gk = args.get<ffi::AnyBuffer>(1), gv = args.get<ffi::AnyBuffer>(2), gb = args.get<ffi::AnyBuffer>(3);
  if (gq.has_error() || gk.has_error() || gv.has_error() || gb.has_error()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_cuda_fwd: q/k/v/bias buffers");
  ffi::AnyBuffer q = gq.value(), k = gk.value(), v = gv.value(), bias = gb.value();
  auto qd = q.dimensions();
  if (qd.size() != 5) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_cuda_fwd: q must be rank 5");
  TriattnCudaCall c; std::memset(&c, 0, sizeof(c));
  c.abi_version = want_lse ? TRIATTN_CUDA_ABI_LSE_VERSION : TRIATTN_CUDA_ABI_VERSION;
  c.stream = reinterpret_cast<void*>(stream);
  c.q = q.untyped_data(); c.k = k.untyped_data(); c.v = v.untyped_data(); c.bias = bias.untyped_data();
  c.B = qd[0]; c.N = qd[1];
  if (layout == 0) { c.H = qd[2]; c.S = qd[3]; } else { c.S = qd[2]; c.H = qd[3]; }
  c.D = qd[4]; c.layout = layout;
  auto bd = bias.dimensions();
  if (bd.size() != 4) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_cuda_fwd: bias must be rank 4 [Bb,H,S,S]");
  c.bias_B = bd[0];
  switch (bias.element_type()) {
    case ffi::DataType::F32: c.bias_dtype = 0; break;
    case ffi::DataType::BF16: c.bias_dtype = 1; break;
    case ffi::DataType::F16: c.bias_dtype = 2; break;
    default: return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_cuda_fwd: bias dtype must be f32 / bf16 / f16");
  }
  if (has_mask) {
    auto gm = args.get<ffi::AnyBuffer>(4);
    if (gm.has_error()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_cuda_fwd: mask buffer");
    c.mask = gm.value().untyped_data();
  }
  void* outs[12]; outs[11] = nullptr;
  for (size_t i = 0; i < n_rets; ++i) {
    auto r = rets.get<ffi::AnyBuffer>(i);
    if (r.has_error()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_cuda_fwd: ret " + std::to_string(i));
    outs[i] = r.value()->untyped_data();
    if (i == 10) c.fix_elems = static_cast<long long>(r.value()->element_count());
  }
  c.out = outs[0]; c.bias_staged = outs[1]; c.keyany = outs[2]; c.rowkind = outs[3]; c.irr = outs[4]; c.rgflag = outs[5];
  c.ktend = outs[6]; c.mtab = outs[7]; c.ctab = outs[8]; c.ktendr = outs[9]; c.fix = outs[10];
  c.lse = outs[11];                                             // nullptr unless want_lse
  c.scale = scale; c.dbg = dbg;
  const int rc = fwd(&c);
  if (rc != 0) return ffi::Error(ffi::ErrorCode::kInternal, std::string("triattn_xla_cuda_fwd: ") + (c.err[0] ? c.err : "error") + " (rc " + std::to_string(rc) + ")");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TriattnXlaCudaFwd, CudaFwdImpl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<CUstream>>().RemainingArgs().RemainingRets().Attr<float>("scale").Attr<int64_t>("flags"));


// ------------------------------------------------------------------------------------------- row triattn_native (triattn_xla_m1_fwd)
// args: q, k, v [B,N,H,S,32] (layout 0) or [B,N,S,H,32] (layout 1) bf16; bias [B,H,S,S] fp32|bf16; [mask u8 [B,N,S]]
// rets: out (q's shape/dtype), bias_staged f32, fix i32, fix_total i32[2], then with a mask: words, keyany, rowkind, kcend, kcstart, rowkc0, rowkc1, counts
// attrs: scale (f32), flags (i64: bit 0 = mask present, bit 1 = layout 1)
static ffi::Error M1FwdImpl(CUstream stream, ffi::RemainingArgs args, ffi::RemainingRets rets, float scale, int64_t flags) {
  triattn_m1_fwd_fn fwd = nullptr;
  { std::lock_guard<std::mutex> lk(g_mu); fwd = g_m1_fwd; }
  if (fwd == nullptr) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, "triattn_xla_m1_fwd: the M1 library entry is not installed (importing opt_core.kernels.triattn_xla installs it when the FFI targets are registered)");
  const bool has_mask = (flags & 1) != 0; const int layout = (flags & 2) ? 1 : 0;
  const size_t n_args = has_mask ? 5 : 4, n_rets = has_mask ? 12 : 4;
  if (args.size() != n_args || rets.size() != n_rets) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_m1_fwd: expected " + std::to_string(n_args) + " args / " + std::to_string(n_rets) + " rets, got " + std::to_string(args.size()) + " / " + std::to_string(rets.size()));
  auto q = args.get<ffi::AnyBuffer>(0).value(); auto k = args.get<ffi::AnyBuffer>(1).value(); auto v = args.get<ffi::AnyBuffer>(2).value(); auto bias = args.get<ffi::AnyBuffer>(3).value();
  auto qd = q.dimensions();
  if (qd.size() != 5 || qd[4] != 32) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_m1_fwd: q must be rank 5 with head_dim 32");
  TriattnM1Call c; std::memset(&c, 0, sizeof(c));
  c.abi_version = TRIATTN_M1_ABI_VERSION;
  const int64_t B = qd[0], N = qd[1], H = layout ? qd[3] : qd[2], S = layout ? qd[2] : qd[3], D = 32;
  c.B = (int32_t)B; c.N = (int32_t)N; c.H = (int32_t)H; c.S = (int32_t)S;
  int64_t st[4];                                                  // element strides of (B, N, H, S) for a contiguous array in the given layout
  if (layout == 0) { st[3] = D; st[2] = S * D; st[1] = H * S * D; st[0] = N * H * S * D; }
  else             { st[2] = D; st[3] = H * D; st[1] = S * H * D; st[0] = N * S * H * D; }
  for (int i = 0; i < 4; ++i) { c.sq[i] = c.sk[i] = c.sv[i] = c.so[i] = st[i]; }
  c.bias_dtype = (bias.element_type() == ffi::DataType::F32) ? 0 : (bias.element_type() == ffi::DataType::BF16 ? 1 : -1);
  c.has_mask = has_mask ? 1 : 0; c.scale = scale;
  c.q = q.untyped_data(); c.k = k.untyped_data(); c.v = v.untyped_data(); c.bias = bias.untyped_data();
  c.mask = has_mask ? args.get<ffi::AnyBuffer>(4).value().untyped_data() : nullptr;
  void* outs[12]; for (size_t i = 0; i < 12; ++i) outs[i] = nullptr;
  for (size_t i = 0; i < n_rets; ++i) { auto r = rets.get<ffi::AnyBuffer>(i); if (r.has_error()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_m1_fwd: ret " + std::to_string(i) + " missing"); outs[i] = r.value()->untyped_data(); }
  c.out = outs[0]; c.bias_staged = outs[1]; c.fix = outs[2]; c.fix_total = outs[3];
  c.words = outs[4]; c.keyany = outs[5]; c.rowkind = outs[6]; c.kcend = outs[7]; c.kcstart = outs[8]; c.rowkc0 = outs[9]; c.rowkc1 = outs[10]; c.counts = outs[11];
  c.stream = reinterpret_cast<void*>(stream);
  const int rc = fwd(&c);
  if (rc != 0) return ffi::Error(ffi::ErrorCode::kInternal, std::string("triattn_xla_m1_fwd: ") + (c.err[0] ? c.err : "error") + " (rc " + std::to_string(rc) + ")");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TriattnXlaM1Fwd, M1FwdImpl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<CUstream>>().RemainingArgs().RemainingRets().Attr<float>("scale").Attr<int64_t>("flags"));


// ------------------------------------------------------------------------------------------- generic cubin call (xla_cubin_call), launcher 1.5
// Self-describing launch of ONE kernel from a cubin under a NAMED package root (txla_add_root): every launch parameter travels as an attribute,
// nothing is registered per shape.  Beyond the scalar / pointer parameters of triattn_xla_run this target assembles BY-VALUE STRUCT parameters
// from a field recipe and encodes CUtensorMap descriptors inside them at call time (cuTensorMapEncodeTiled over the call's own buffers), and can
// clamp the grid to the device's multiprocessor count (persistent kernels).  Attributes:
//   root, path, sha  : cubin = <dir of root>/<path>; sha = 16-hex-digit sha256 prefix checked on first use ("" = unchecked)
//   kname, shared, bx: kernel name, dynamic shared bytes, threads per block (x)
//   gx, gy, gz, grule: grid; grule 1 = gx := min(gx, multiprocessors of the stream's device)
//   params           : ','-separated top-level kernel parameters: b<i> in-buffer ptr | o<i> out-buffer ptr | i<v> i32 | l<v> i64 | f<hex> f32 | n null ptr |
//                      S<size>:<fields> a by-value struct of exactly <size> bytes whose fields (';'-separated, appended in order, the tracer inserts
//                      the padding) are: b<i> / o<i> (8-byte ptr) | n (8 zero bytes) | i<v> (i32) | q<v> (i64) | f<hex> (f32) | z<k> (k zero bytes) |
//                      t<b|o><i>:<m> (the 128-byte CUtensorMap of tensor-map spec m over in/out buffer i)
//   tmaps            : '|'-separated tensor-map specs: dt=<u8|u16|u32|i32|u64|i64|f16|f32|f64|bf16>,dims=d0/d1[/d2..] (innermost first),str=s1[/s2..] (bytes,
//                      dims 1..), box=b0/b1[/..],swz=<0|32|64|128>,l2=<0|64|128|256>,oob=<0|1>   (interleave none, element strides 1)
//   label            : text for error messages
struct TmapSpec { CUtensorMapDataType dt; int rank; cuuint64_t dims[5]; cuuint64_t strides[4]; cuuint32_t box[5]; CUtensorMapSwizzle swz; CUtensorMapL2promotion l2; CUtensorMapFloatOOBfill oob; };
struct SField { int kind; long long ival; double fval; int buf; int tmap; };     // kind: 0 inptr 1 outptr 2 null 3 i32 4 i64 5 f32 6 zeros(ival) 7 tmap-in 8 tmap-out
struct SParam { int size; std::vector<SField> fields; };
struct GParam { int kind; long long ival; double fval; int sidx; };               // kind as Param (IN_BUF..NULLPTR) or 100 = struct #sidx
struct GenSpec {
  std::string root, path, sha, kname, label, cubin;
  int shared = 0, bx = 128, grule = 0;
  std::vector<GParam> params; std::vector<SParam> structs; std::vector<TmapSpec> tmaps;
  std::vector<CtxFn> loaded; unsigned long long launches = 0;
};
std::map<std::string, GenSpec*> g_gen;
std::map<CUdevice, int> g_sms;

static bool split_next(std::string_view s, char sep, size_t* pos, std::string_view* tok) {
  if (*pos > s.size()) return false;
  size_t j = s.find(sep, *pos); if (j == std::string_view::npos) j = s.size();
  *tok = s.substr(*pos, j - *pos); *pos = j + 1; return true;
}
static bool parse_u64_list(std::string_view body, cuuint64_t* out, int maxn, int* n) {
  size_t pos = 0; std::string_view tok; *n = 0;
  while (pos <= body.size() && split_next(body, '/', &pos, &tok)) { if (tok.empty()) continue; long long v; if (*n >= maxn || !parse_int(tok, &v) || v < 0) return false; out[(*n)++] = static_cast<cuuint64_t>(v); }
  return true;
}
static bool parse_tmaps(std::string_view rec, std::vector<TmapSpec>* out, std::string* err) {
  size_t pos = 0; std::string_view one;
  while (pos <= rec.size() && split_next(rec, '|', &pos, &one)) {
    if (one.empty()) continue;
    TmapSpec t; std::memset(&t, 0, sizeof(t)); t.swz = CU_TENSOR_MAP_SWIZZLE_NONE; t.l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE; t.oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    int ndims = 0, nstr = 0, nbox = 0; bool have_dt = false;
    size_t p2 = 0; std::string_view kv;
    while (p2 <= one.size() && split_next(one, ',', &p2, &kv)) {
      if (kv.empty()) continue;
      size_t eq = kv.find('='); if (eq == std::string_view::npos) { *err = "tmaps: bad field '" + std::string(kv) + "'"; return false; }
      std::string_view k = kv.substr(0, eq), v = kv.substr(eq + 1);
      if (k == "dt") {
        static const struct { const char* n; CUtensorMapDataType d; } T[] = {{"u8", CU_TENSOR_MAP_DATA_TYPE_UINT8}, {"u16", CU_TENSOR_MAP_DATA_TYPE_UINT16}, {"u32", CU_TENSOR_MAP_DATA_TYPE_UINT32},
          {"i32", CU_TENSOR_MAP_DATA_TYPE_INT32}, {"u64", CU_TENSOR_MAP_DATA_TYPE_UINT64}, {"i64", CU_TENSOR_MAP_DATA_TYPE_INT64}, {"f16", CU_TENSOR_MAP_DATA_TYPE_FLOAT16},
          {"f32", CU_TENSOR_MAP_DATA_TYPE_FLOAT32}, {"f64", CU_TENSOR_MAP_DATA_TYPE_FLOAT64}, {"bf16", CU_TENSOR_MAP_DATA_TYPE_BFLOAT16}};
        bool ok = false; for (const auto& e : T) if (v == e.n) { t.dt = e.d; ok = true; break; }
        if (!ok) { *err = "tmaps: data type '" + std::string(v) + "'"; return false; } have_dt = true;
      } else if (k == "dims") { if (!parse_u64_list(v, t.dims, 5, &ndims)) { *err = "tmaps: dims"; return false; } }
      else if (k == "str") { cuuint64_t tmp[4]; if (!parse_u64_list(v, tmp, 4, &nstr)) { *err = "tmaps: str"; return false; } for (int i = 0; i < nstr; ++i) t.strides[i] = tmp[i]; }
      else if (k == "box") { cuuint64_t tmp[5]; if (!parse_u64_list(v, tmp, 5, &nbox)) { *err = "tmaps: box"; return false; } for (int i = 0; i < nbox; ++i) t.box[i] = static_cast<cuuint32_t>(tmp[i]); }
      else if (k == "swz") { long long x; if (!parse_int(v, &x)) { *err = "tmaps: swz"; return false; } t.swz = x == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : x == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : x == 32 ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE; }
      else if (k == "l2") { long long x; if (!parse_int(v, &x)) { *err = "tmaps: l2"; return false; } t.l2 = x == 256 ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B : x == 128 ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B : x == 64 ? CU_TENSOR_MAP_L2_PROMOTION_L2_64B : CU_TENSOR_MAP_L2_PROMOTION_NONE; }
      else if (k == "oob") { long long x; if (!parse_int(v, &x)) { *err = "tmaps: oob"; return false; } t.oob = x ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE; }
      else { *err = "tmaps: unknown field '" + std::string(k) + "'"; return false; }
    }
    if (!have_dt || ndims < 1 || nstr != ndims - 1 || nbox != ndims) { *err = "tmaps: spec '" + std::string(one) + "' needs dt, dims (rank r), str (r-1), box (r)"; return false; }
    t.rank = ndims; out->push_back(t);
  }
  return true;
}
static bool parse_sfields(std::string_view rec, SParam* sp, int ntmaps, std::string* err) {
  size_t pos = 0; std::string_view tok; int bytes = 0;
  while (pos <= rec.size() && split_next(rec, ';', &pos, &tok)) {
    if (tok.empty()) continue;
    SField f; f.kind = -1; f.ival = 0; f.fval = 0; f.buf = -1; f.tmap = -1;
    const char c = tok[0]; std::string_view body = tok.substr(1); bool ok = true;
    if (c == 'b') { f.kind = 0; ok = parse_int(body, &f.ival); f.buf = static_cast<int>(f.ival); bytes += 8; }
    else if (c == 'o') { f.kind = 1; ok = parse_int(body, &f.ival); f.buf = static_cast<int>(f.ival); bytes += 8; }
    else if (c == 'n') { f.kind = 2; bytes += 8; }
    else if (c == 'i') { f.kind = 3; ok = parse_int(body, &f.ival); bytes += 4; }
    else if (c == 'q') { f.kind = 4; ok = parse_int(body, &f.ival); bytes += 8; }
    else if (c == 'f') { f.kind = 5; ok = parse_hexfloat(body, &f.fval); bytes += 4; }
    else if (c == 'z') { f.kind = 6; ok = parse_int(body, &f.ival) && f.ival >= 0; bytes += static_cast<int>(f.ival); }
    else if (c == 't' && body.size() >= 4 && (body[0] == 'b' || body[0] == 'o')) {
      size_t colon = body.find(':'); long long bi = -1, mi = -1;
      ok = colon != std::string_view::npos && parse_int(body.substr(1, colon - 1), &bi) && parse_int(body.substr(colon + 1), &mi) && mi >= 0 && mi < ntmaps;
      f.kind = body[0] == 'b' ? 7 : 8; f.buf = static_cast<int>(bi); f.tmap = static_cast<int>(mi); bytes += 128;
    } else ok = false;
    if (!ok) { *err = "struct field '" + std::string(tok) + "'"; return false; }
    sp->fields.push_back(f);
  }
  if (bytes != sp->size) { *err = "struct fields add up to " + std::to_string(bytes) + " bytes, declared " + std::to_string(sp->size); return false; }
  return true;
}
static bool parse_gparams(std::string_view rec, GenSpec* gs, std::string* err) {
  // top-level tokens separated by ','; a struct token is S<size>:<fields> where <fields> uses ';' (no ',' inside)
  size_t pos = 0; std::string_view tok;
  while (pos <= rec.size() && split_next(rec, ',', &pos, &tok)) {
    if (tok.empty()) continue;
    GParam p{NULLPTR, 0, 0.0, -1}; const char c = tok[0]; std::string_view body = tok.substr(1); bool ok = true;
    if (c == 'b') { p.kind = IN_BUF; ok = parse_int(body, &p.ival); }
    else if (c == 'o') { p.kind = OUT_BUF; ok = parse_int(body, &p.ival); }
    else if (c == 'i') { p.kind = I32; ok = parse_int(body, &p.ival); }
    else if (c == 'l') { p.kind = I64; ok = parse_int(body, &p.ival); }
    else if (c == 'f') { p.kind = F32; ok = parse_hexfloat(body, &p.fval); }
    else if (c == 'n') { p.kind = NULLPTR; }
    else if (c == 'S') {
      size_t colon = body.find(':'); long long sz = 0;
      ok = colon != std::string_view::npos && parse_int(body.substr(0, colon), &sz) && sz > 0 && sz <= 4096;
      if (ok) { SParam sp; sp.size = static_cast<int>(sz); ok = parse_sfields(body.substr(colon + 1), &sp, static_cast<int>(gs->tmaps.size()), err); if (ok) { p.kind = 100; p.sidx = static_cast<int>(gs->structs.size()); gs->structs.push_back(sp); } else return false; }
    } else ok = false;
    if (!ok) { *err = "bad parameter token '" + std::string(tok) + "'"; return false; }
    gs->params.push_back(p);
  }
  return true;
}

static ffi::Error GenImpl(CUstream stream, ffi::RemainingArgs args, ffi::RemainingRets rets, std::string_view root, std::string_view path, std::string_view sha, std::string_view kname,
                         int64_t shared, int64_t bx, int64_t gx, int64_t gy, int64_t gz, int64_t grule, std::string_view params, std::string_view tmaps, std::string_view label) {
  GenSpec* s = nullptr; CUfunction fn = nullptr; int sms = 0;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    std::string ck; ck.reserve(root.size() + path.size() + params.size() + tmaps.size() + kname.size() + 64);
    ck.append(root).append("|").append(path).append("|").append(sha).append("|").append(kname).append("|").append(std::to_string(shared)).append("|").append(std::to_string(bx)).append("|").append(params).append("|").append(tmaps);
    auto it = g_gen.find(ck);
    if (it == g_gen.end()) {
      auto rt = g_roots.find(std::string(root));
      if (rt == g_roots.end()) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, "xla_cubin_call: package root '" + std::string(root) + "' is not registered in this process (importing the package that owns the program registers it) spec=" + std::string(label));
      auto* gs = new GenSpec(); gs->root.assign(root); gs->path.assign(path); gs->sha.assign(sha); gs->kname.assign(kname); gs->label.assign(label);
      gs->shared = static_cast<int>(shared); gs->bx = static_cast<int>(bx); gs->grule = static_cast<int>(grule);
      std::string err;
      if (!parse_tmaps(tmaps, &gs->tmaps, &err) || !parse_gparams(params, gs, &err)) { delete gs; return ffi::Error(ffi::ErrorCode::kInvalidArgument, "xla_cubin_call: " + err + " spec=" + std::string(label)); }
      const std::string full = rt->second + "/" + gs->path;
      std::FILE* f = std::fopen(full.c_str(), "rb");
      if (f == nullptr) { delete gs; return ffi::Error(ffi::ErrorCode::kNotFound, "xla_cubin_call: cubin not found: " + full + " (the program was traced against a package that carried it) spec=" + std::string(label)); }
      std::fseek(f, 0, SEEK_END); const long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
      gs->cubin.resize(sz > 0 ? static_cast<size_t>(sz) : 0);
      const size_t got = sz > 0 ? std::fread(&gs->cubin[0], 1, static_cast<size_t>(sz), f) : 0; std::fclose(f);
      if (got != gs->cubin.size()) { delete gs; return ffi::Error(ffi::ErrorCode::kInternal, "xla_cubin_call: short read of " + full); }
      gs->cubin.push_back('\0');
      const std::string have = hex_sha256_prefix16(gs->cubin.substr(0, gs->cubin.size() - 1));
      if (!gs->sha.empty() && have != gs->sha) { std::string m = "xla_cubin_call: " + full + " has sha256 " + have + "..., the program was traced against " + gs->sha + "... -- re-trace the program against this package; spec=" + std::string(label); delete gs; return ffi::Error(ffi::ErrorCode::kFailedPrecondition, m); }
      it = g_gen.emplace(ck, gs).first;
    }
    s = it->second;
    CUcontext ctx = nullptr;
    CUresult rc = cuCtxGetCurrent(&ctx);
    if (rc != CUDA_SUCCESS || ctx == nullptr) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuCtxGetCurrent", rc) + " (no current CUDA context) spec=" + s->label);
    for (const CtxFn& cf : s->loaded) if (cf.ctx == ctx) { fn = cf.fn; break; }
    if (fn == nullptr) {
      CtxFn cf; cf.ctx = ctx; cf.mod = nullptr; cf.fn = nullptr;
      rc = cuModuleLoadData(&cf.mod, s->cubin.data());
      if (rc != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuModuleLoadData", rc) + " cubin=" + s->path + " spec=" + s->label);
      rc = cuModuleGetFunction(&cf.fn, cf.mod, s->kname.c_str());
      if (rc != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuModuleGetFunction", rc) + " kernel=" + s->kname + " spec=" + s->label);
      if (s->shared > 49152) {
        rc = cuFuncSetAttribute(cf.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, s->shared);
        if (rc != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuFuncSetAttribute(max dynamic smem)", rc) + " shared=" + std::to_string(s->shared) + " kernel=" + s->kname + " spec=" + s->label);
      }
      s->loaded.push_back(cf); fn = cf.fn;
    }
    if (s->grule == 1) {
      CUdevice dev; rc = cuCtxGetDevice(&dev);
      if (rc != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuCtxGetDevice", rc));
      auto is = g_sms.find(dev);
      if (is == g_sms.end()) { int v = 0; rc = cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev); if (rc != CUDA_SUCCESS || v <= 0) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuDeviceGetAttribute(multiprocessors)", rc)); is = g_sms.emplace(dev, v).first; }
      sms = is->second;
    }
    s->launches++;
  }
  // resolve buffers
  auto in_ptr = [&](int i, void** out) -> bool { auto b = args.get<ffi::AnyBuffer>(static_cast<size_t>(i)); if (b.has_error()) return false; *out = b.value().untyped_data(); return true; };
  auto out_ptr = [&](int i, void** out) -> bool { auto b = rets.get<ffi::AnyBuffer>(static_cast<size_t>(i)); if (b.has_error()) return false; *out = b.value()->untyped_data(); return true; };
  const size_t n = s->params.size();
  std::vector<void*> ptrs; ptrs.reserve(n + 1); std::vector<int32_t> i32s; i32s.reserve(n + 1); std::vector<int64_t> i64s; i64s.reserve(n + 1); std::vector<float> f32s; f32s.reserve(n + 1);
  std::vector<std::vector<unsigned char>> blobs; blobs.reserve(s->structs.size() + 1);
  std::vector<void*> kparams; kparams.reserve(n + 1);
  for (const GParam& p : s->params) {
    switch (p.kind) {
      case IN_BUF: { void* q; if (!in_ptr(static_cast<int>(p.ival), &q)) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "xla_cubin_call: input buffer " + std::to_string(p.ival) + " out of range, spec=" + s->label); ptrs.push_back(q); kparams.push_back(&ptrs.back()); break; }
      case OUT_BUF: { void* q; if (!out_ptr(static_cast<int>(p.ival), &q)) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "xla_cubin_call: output buffer " + std::to_string(p.ival) + " out of range, spec=" + s->label); ptrs.push_back(q); kparams.push_back(&ptrs.back()); break; }
      case I32: i32s.push_back(static_cast<int32_t>(p.ival)); kparams.push_back(&i32s.back()); break;
      case I64: i64s.push_back(static_cast<int64_t>(p.ival)); kparams.push_back(&i64s.back()); break;
      case F32: f32s.push_back(static_cast<float>(p.fval)); kparams.push_back(&f32s.back()); break;
      case 100: {
        const SParam& sp = s->structs[static_cast<size_t>(p.sidx)];
        blobs.emplace_back(static_cast<size_t>(sp.size) + 64, 0);
        unsigned char* base = blobs.back().data(); base += (64 - (reinterpret_cast<uintptr_t>(base) & 63)) & 63;   // 64-byte aligned start
        size_t off = 0;
        for (const SField& f : sp.fields) {
          switch (f.kind) {
            case 0: { void* q; if (!in_ptr(f.buf, &q)) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "xla_cubin_call: struct field: input buffer " + std::to_string(f.buf) + " out of range, spec=" + s->label); std::memcpy(base + off, &q, 8); off += 8; break; }
            case 1: { void* q; if (!out_ptr(f.buf, &q)) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "xla_cubin_call: struct field: output buffer " + std::to_string(f.buf) + " out of range, spec=" + s->label); std::memcpy(base + off, &q, 8); off += 8; break; }
            case 2: off += 8; break;
            case 3: { int32_t v = static_cast<int32_t>(f.ival); std::memcpy(base + off, &v, 4); off += 4; break; }
            case 4: { int64_t v = static_cast<int64_t>(f.ival); std::memcpy(base + off, &v, 8); off += 8; break; }
            case 5: { float v = static_cast<float>(f.fval); std::memcpy(base + off, &v, 4); off += 4; break; }
            case 6: off += static_cast<size_t>(f.ival); break;
            case 7: case 8: {
              void* q; if (!(f.kind == 7 ? in_ptr(f.buf, &q) : out_ptr(f.buf, &q))) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "xla_cubin_call: tensor map over buffer " + std::to_string(f.buf) + " out of range, spec=" + s->label);
              if ((off & 63) != 0) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "xla_cubin_call: tensor-map field at offset " + std::to_string(off) + " is not 64-byte aligned, spec=" + s->label);
              const TmapSpec& t = s->tmaps[static_cast<size_t>(f.tmap)];
              cuuint32_t es[5] = {1, 1, 1, 1, 1};
              CUtensorMap tm;
              CUresult r = cuTensorMapEncodeTiled(&tm, t.dt, static_cast<cuuint32_t>(t.rank), q, t.dims, t.strides, t.box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, t.swz, t.l2, t.oob);
              if (r != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInvalidArgument, cu_err("cuTensorMapEncodeTiled", r) + " (map " + std::to_string(f.tmap) + ", buffer " + std::to_string(f.buf) + ", address % 16 = " + std::to_string(reinterpret_cast<uintptr_t>(q) & 15) + ") spec=" + s->label);
              std::memcpy(base + off, &tm, 128); off += 128; break; }
            default: break;
          }
        }
        kparams.push_back(base); break; }
      default: ptrs.push_back(nullptr); kparams.push_back(&ptrs.back()); break;
    }
  }
  long long gxx = gx;
  if (s->grule == 1 && sms > 0 && gxx > sms) gxx = sms;
  if (gxx < 1) gxx = 1;
  CUresult r = cuLaunchKernel(fn, static_cast<unsigned>(gxx), static_cast<unsigned>(gy), static_cast<unsigned>(gz), static_cast<unsigned>(s->bx), 1u, 1u,
                              static_cast<unsigned>(s->shared), stream, kparams.data(), nullptr);
  if (r != CUDA_SUCCESS) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuLaunchKernel", r) + " kernel=" + s->kname + " spec=" + s->label);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(XlaCubinCall, GenImpl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<CUstream>>().RemainingArgs().RemainingRets()
                                  .Attr<std::string_view>("root").Attr<std::string_view>("path").Attr<std::string_view>("sha").Attr<std::string_view>("kname")
                                  .Attr<int64_t>("shared").Attr<int64_t>("bx").Attr<int64_t>("gx").Attr<int64_t>("gy").Attr<int64_t>("gz").Attr<int64_t>("grule")
                                  .Attr<std::string_view>("params").Attr<std::string_view>("tmaps").Attr<std::string_view>("label"));


// ------------------------------------------------------------------------------------------- cuBLAS strided-batched GEMM (xla_cublas_bgemm_nt), launcher 1.5
// X[c] = A[c] . B[c]^T for row-major bf16 planes A [batch, m, k], B [batch, n, k] living INSIDE the call's input buffers at element offsets (so the two
// halves of ONE plane buffer are contracted without XLA materialising slices); fp32 compute, bf16 out, the default tensor-op algorithm, a caller-provided
// workspace (ret 1) -- i.e. the call torch.bmm(a, b.transpose(1, 2)) issues (cublasGemmStridedBatchedEx opT/opN), on XLA's stream.  The cuBLAS entry points are
// the running jaxlib's own library, resolved by the Python side (ctypes) and installed through txla_set_cublas(); a per-context handle is created on first use.
// attrs: a_buf a_off b_buf b_off (input buffer indices + element offsets), batch, m, n, k, ws (workspace bytes = size of ret 1; 0 = none), label
typedef struct cublasContext* cublasHandle_t;
typedef int (*cublasCreate_t)(cublasHandle_t*);
typedef int (*cublasSetStream_t)(cublasHandle_t, CUstream);
typedef int (*cublasSetWorkspace_t)(cublasHandle_t, void*, size_t);
typedef int (*cublasGemmStridedBatchedEx_t)(cublasHandle_t, int, int, int, int, int, const void*, const void*, int, int, long long, const void*, int, int, long long,
                                            const void*, void*, int, int, long long, int, int, int);
struct Cublas { cublasCreate_t create = nullptr; cublasSetStream_t set_stream = nullptr; cublasSetWorkspace_t set_ws = nullptr; cublasGemmStridedBatchedEx_t bgemm = nullptr; std::string state = "not installed (the Python side installs the cuBLAS entry points at registration)"; };
Cublas g_cublas;
std::map<CUcontext, cublasHandle_t> g_cublas_handles;
enum { kCUBLAS_OP_N = 0, kCUBLAS_OP_T = 1, kCUDA_R_16BF = 14, kCUBLAS_COMPUTE_32F = 68, kCUBLAS_GEMM_DEFAULT_TENSOR_OP = 99 };

extern "C" void txla_set_cublas(void* create, void* set_stream, void* set_ws, void* bgemm, const char* origin) {
  std::lock_guard<std::mutex> lk(g_mu);
  Cublas cb; cb.create = reinterpret_cast<cublasCreate_t>(create); cb.set_stream = reinterpret_cast<cublasSetStream_t>(set_stream); cb.set_ws = reinterpret_cast<cublasSetWorkspace_t>(set_ws);
  cb.bgemm = reinterpret_cast<cublasGemmStridedBatchedEx_t>(bgemm);
  cb.state = (cb.create && cb.set_stream && cb.bgemm) ? std::string("installed: ") + (origin ? origin : "?") : std::string("incomplete entry points from ") + (origin ? origin : "?");
  g_cublas = cb;
}
extern "C" const char* txla_cublas_state(void) { std::lock_guard<std::mutex> lk(g_mu); static std::string st; st = g_cublas.state; return st.c_str(); }
static bool cublas_ready(std::string* err) { if (g_cublas.create && g_cublas.set_stream && g_cublas.bgemm) return true; *err = g_cublas.state; return false; }

static ffi::Error BgemmImpl(CUstream stream, ffi::RemainingArgs args, ffi::RemainingRets rets, int64_t a_buf, int64_t a_off, int64_t b_buf, int64_t b_off, int64_t batch, int64_t m, int64_t n, int64_t k, int64_t ws,
                           std::string_view label) {
  cublasHandle_t h = nullptr; Cublas cb;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    std::string err;
    if (!cublas_ready(&err)) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, "xla_cublas_bgemm_nt: " + err + " spec=" + std::string(label));
    cb = g_cublas;
    CUcontext ctx = nullptr; CUresult rc = cuCtxGetCurrent(&ctx);
    if (rc != CUDA_SUCCESS || ctx == nullptr) return ffi::Error(ffi::ErrorCode::kInternal, cu_err("cuCtxGetCurrent", rc));
    auto it = g_cublas_handles.find(ctx);
    if (it == g_cublas_handles.end()) {
      cublasHandle_t nh = nullptr; const int st = cb.create(&nh);
      if (st != 0 || nh == nullptr) return ffi::Error(ffi::ErrorCode::kInternal, "xla_cublas_bgemm_nt: cublasCreate failed (" + std::to_string(st) + ")");
      it = g_cublas_handles.emplace(ctx, nh).first;
    }
    h = it->second;
  }
  auto ab = args.get<ffi::AnyBuffer>(static_cast<size_t>(a_buf)); auto bb = args.get<ffi::AnyBuffer>(static_cast<size_t>(b_buf)); auto xb = rets.get<ffi::AnyBuffer>(0);
  if (ab.has_error() || bb.has_error() || xb.has_error()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "xla_cublas_bgemm_nt: buffer index out of range spec=" + std::string(label));
  const char* A = static_cast<const char*>(ab.value().untyped_data()) + 2 * a_off; const char* B = static_cast<const char*>(bb.value().untyped_data()) + 2 * b_off;
  void* X = xb.value()->untyped_data();
  void* wsp = nullptr;
  if (ws > 0) { auto w = rets.get<ffi::AnyBuffer>(1); if (w.has_error()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "xla_cublas_bgemm_nt: workspace ret missing"); wsp = w.value()->untyped_data(); }
  static std::mutex call_mu;                     // the handle's stream / workspace are handle state: serialise the (cheap, asynchronous) enqueue
  std::lock_guard<std::mutex> lk2(call_mu);
  int st = cb.set_stream(h, stream); if (st != 0) return ffi::Error(ffi::ErrorCode::kInternal, "xla_cublas_bgemm_nt: cublasSetStream failed (" + std::to_string(st) + ")");
  if (cb.set_ws != nullptr && wsp != nullptr) { st = cb.set_ws(h, wsp, static_cast<size_t>(ws)); if (st != 0) return ffi::Error(ffi::ErrorCode::kInternal, "xla_cublas_bgemm_nt: cublasSetWorkspace failed (" + std::to_string(st) + ")"); }
  const float alpha = 1.f, beta = 0.f;
  // row-major X[m x n] = A[m x k] B[n x k]^T  ==  column-major X^T = B_rm . A_rm^T : op(T) on B's memory (k x n col-major), op(N) on A's memory (k x m col-major)
  st = cb.bgemm(h, kCUBLAS_OP_T, kCUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha, B, kCUDA_R_16BF, static_cast<int>(k), static_cast<long long>(n) * k,
                A, kCUDA_R_16BF, static_cast<int>(k), static_cast<long long>(m) * k, &beta, X, kCUDA_R_16BF, static_cast<int>(n), static_cast<long long>(m) * n, static_cast<int>(batch),
                kCUBLAS_COMPUTE_32F, kCUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (st != 0) return ffi::Error(ffi::ErrorCode::kInternal, "xla_cublas_bgemm_nt: cublasGemmStridedBatchedEx failed (" + std::to_string(st) + ") m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k) + " batch=" + std::to_string(batch) + " spec=" + std::string(label));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(XlaCublasBgemm, BgemmImpl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<CUstream>>().RemainingArgs().RemainingRets()
                                  .Attr<int64_t>("a_buf").Attr<int64_t>("a_off").Attr<int64_t>("b_buf").Attr<int64_t>("b_off").Attr<int64_t>("batch").Attr<int64_t>("m").Attr<int64_t>("n").Attr<int64_t>("k")
                                  .Attr<int64_t>("ws").Attr<std::string_view>("label"));


// ------------------------------------------------------------------------------------------- row cuda_80 (triattn_xla_sm80_fwd)
// args: q, k, v [B,N,H,S,D] (layout 0) or [B,N,S,H,D] (layout 1) bf16, D in {16, 32, 64}; bias [B,H,S,S] fp32|bf16; [mask u8 [B,N,S]]
// rets: out (q's shape/dtype); [lse f32 [B,N,H,S] when flags bit 2]; bias_staged f32; fix i32; census i32[4]; then with a mask: keyany u8, rows i32, maskw i32, rgflag i32
// attrs: scale (f32), flags (i64: bit 0 = mask present, bit 1 = layout 1, bit 2 = lse output), fix_elems (i64: the fix ret's length)
static ffi::Error Sm80FwdImpl(CUstream stream, ffi::RemainingArgs args, ffi::RemainingRets rets, float scale, int64_t flags, int64_t fix_elems) {
  triattn_sm80_fwd_fn fwd = nullptr;
  { std::lock_guard<std::mutex> lk(g_mu); fwd = g_sm80_fwd; }
  if (fwd == nullptr) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, "triattn_xla_sm80_fwd: the sm_80 library entry is not installed (importing opt_core.kernels.triattn_xla installs it when the FFI targets are registered)");
  const bool has_mask = (flags & 1) != 0; const int layout = (flags & 2) ? 1 : 0; const bool want_lse = (flags & 4) != 0;
  const size_t n_args = has_mask ? 5 : 4, n_rets = 4 + (want_lse ? 1 : 0) + (has_mask ? 4 : 0);
  if (args.size() != n_args || rets.size() != n_rets) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_sm80_fwd: expected " + std::to_string(n_args) + " args / " + std::to_string(n_rets) + " rets, got " + std::to_string(args.size()) + " / " + std::to_string(rets.size()));
  auto q = args.get<ffi::AnyBuffer>(0).value(); auto k = args.get<ffi::AnyBuffer>(1).value(); auto v = args.get<ffi::AnyBuffer>(2).value(); auto bias = args.get<ffi::AnyBuffer>(3).value();
  auto qd = q.dimensions();
  if (qd.size() != 5 || (qd[4] != 16 && qd[4] != 32 && qd[4] != 64)) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_sm80_fwd: q must be rank 5 with head_dim 16, 32 or 64");
  TriattnSm80Call c; std::memset(&c, 0, sizeof(c));
  c.abi_version = TRIATTN_SM80_ABI_VERSION;
  const int64_t D = qd[4], B = qd[0], N = qd[1], H = layout ? qd[3] : qd[2], S = layout ? qd[2] : qd[3];
  c.B = (int32_t)B; c.N = (int32_t)N; c.H = (int32_t)H; c.S = (int32_t)S; c.D = (int32_t)D;
  int64_t st[4];                                                  // element strides of (B, N, H, S) for a contiguous array in the given layout
  if (layout == 0) { st[3] = D; st[2] = S * D; st[1] = H * S * D; st[0] = N * H * S * D; }
  else             { st[2] = D; st[3] = H * D; st[1] = S * H * D; st[0] = N * S * H * D; }
  for (int i = 0; i < 4; ++i) { c.sq[i] = c.sk[i] = c.sv[i] = c.so[i] = st[i]; }
  c.bias_dtype = (bias.element_type() == ffi::DataType::F32) ? 0 : (bias.element_type() == ffi::DataType::BF16 ? 1 : -1);
  c.has_mask = has_mask ? 1 : 0; c.want_lse = want_lse ? 1 : 0; c.scale = scale; c.fix_elems = fix_elems;
  c.q = q.untyped_data(); c.k = k.untyped_data(); c.v = v.untyped_data(); c.bias = bias.untyped_data();
  c.mask = has_mask ? args.get<ffi::AnyBuffer>(4).value().untyped_data() : nullptr;
  void* outs[9]; for (size_t i = 0; i < 9; ++i) outs[i] = nullptr;
  for (size_t i = 0; i < n_rets; ++i) { auto r = rets.get<ffi::AnyBuffer>(i); if (r.has_error()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "triattn_xla_sm80_fwd: ret " + std::to_string(i) + " missing"); outs[i] = r.value()->untyped_data(); }
  size_t p = 0;
  c.out = outs[p++]; if (want_lse) c.lse = outs[p++];
  c.bias_staged = outs[p++]; c.fix = outs[p++]; c.census = outs[p++];
  if (has_mask) { c.keyany = outs[p++]; c.rows = outs[p++]; c.maskw = outs[p++]; c.rgflag = outs[p++]; }
  c.stream = reinterpret_cast<void*>(stream);
  const int rc = fwd(&c);
  if (rc != 0) return ffi::Error(ffi::ErrorCode::kInternal, std::string("triattn_xla_sm80_fwd: ") + (c.err[0] ? c.err : "error") + " (rc " + std::to_string(rc) + ")");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TriattnXlaSm80Fwd, Sm80FwdImpl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<CUstream>>().RemainingArgs().RemainingRets().Attr<float>("scale").Attr<int64_t>("flags").Attr<int64_t>("fix_elems"));
