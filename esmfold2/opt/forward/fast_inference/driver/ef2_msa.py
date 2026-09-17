"""ef2_msa — MSA-module inference-execution levers for the ESMFold2 FULL model (MSAEncoder: OuterProductMean, MSAPairWeightedAveraging,
msa_transition), layered on ef2_opt + ef2_w4 and written in the ef2_w4
conventions (levers T11-T14).  Nothing here changes model math, weights, inputs, MSAs, loops or RNG draws; every
lever replaces the EXECUTION of one op of Biohub transformers@ef32577f (modeling_esmfold2.py / modeling_esmfold2_common.py).
Facts the design rests on (measured on H100, 418/705 tok, M=2048): the whole MSA module runs under torch.autocast(bf16):
Linear = cuBLAS bf16 GEMM, LayerNorm = ATen fp32 (bf16->fp32 copy + fp32 LN + fp32->bf16 copy = ~1/3 of each op's time), einsum = bf16 cuBLAS
GEMM (+ permute copies), softmax fp32.  All three stock ops are deterministic and CUDA-graph replayable (r2r bitwise, replay == eager).

 T11  msa_transition  MSAEncoderBlock.msa_transition = PairTransition(128, x4): LN -> w12 -> silu(x1)*x2 -> w3 (residual added by the block).
                      T11b (default, own): T14 bf16-in LN with fp32 statistics -> stock w12 GEMM -> ONE fused silu(x1)*x2 kernel with the stock rounding
                      points (bf16 silu, bf16 product) -> stock w3 GEMM.  TIER-2 (LN statistics reduction order; exp implementation inside silu).
                      T11a (t11_impl="biohub", record only): Biohub's FusedLNLinearSwiGLU unmodified — measured to FAIL the Tier-2 bar on this op
                      (it stores the LN mean/rstd in bf16: err vs fp64 1.43x max / 1.89x rms of stock's at 705 tok), so it is NOT the default.
 T12  opm             OuterProductMean.forward (chunk None): stock LN + W (bitwise operands; a/b built in bf16 == stock's fp32-promote + autocast
                      recast, checked), contraction through torch.matmul on the SAME cuBLAS problem the stock einsum lowers to ([(i c), M] x [M, (j d)],
                      measured bitwise at 418/705), then ONE Triton epilogue: projection K=1024 -> 256 as a single ascending fp32 dot chain, + bias,
                      -> bf16 (the stock Linear's rounding point), / n_valid with IEEE div.rn -> bf16 (stock: bf16 / bf16 in fp32 opmath -> bf16),
                      stored straight into [L, L, 256].  Removes the [L,L,32,32] -> [L,L,1024] re-layout copy, the bias/div passes.
                      EXACT iff the in-process probe says bitwise (needs the Triton K=1024 fp32 chain to reproduce cuBLAS's accumulation; MEASURED,
                      never assumed: first call per shape is always probed against the stock forward); else TIER-2 'same rounding points, fp32
                      accumulation order of the K=1024 projection only'.  describe()/stats() print which.
 T13  pwa             MSAPairWeightedAveraging.forward: stock LN(m) / LN(pair)->bias->masked softmax (fp32) / Wv / Wgate / sigmoid / Wout (cuBLAS,
                      bitwise operands; the fp32->bf16 cast of msa_normed done ONCE instead of twice), the 3-operand einsum replaced by ONE Triton kernel:
                      acc = sum_j attn[i,j,h] v[j,m,h,:] (fp32) -> bf16 (stock's (attn.v) rounding point, checked: stock einsum == bf16(attn.v)*gate)
                      * gate -> bf16, written [L, M, 128] contiguous for Wout (no permute copies).  TIER-2 (fp32 accumulation order over j vs cuBLAS).
 T14  ln_bf16         the two big LayerNorm(128) calls on m inside T12/T13 (OPM.norm, PWA.norm_single) -> one Triton row-LN kernel reading bf16,
                      fp32 statistics, writing bf16 (= the value the stock Linear consumes after autocast's cast).  TIER-2 (fp32 reduction order of the
                      LN statistics; ~1 bf16 ulp flips); turns T12 into TIER-2 when on.  Only active inside T12/T13 (needs opm/pwa enabled).
 probe                recompute the STOCK op next to every patched op for the first probe_n (default 2) calls per (op, shape); record (max|diff|, bitwise).

Ownership / composition contract:
  * per-INSTANCE patches only (instance.forward = MethodType) on blk.outer_product_mean / blk.msa_pair_weighted_averaging / blk.msa_transition of
    model.msa_encoder.blocks.  NEVER touches C.Transition.forward, C.PairTransition.forward, MOD.PairTransition.forward (W4 saves/restores those),
    C._fused_trimul_with_residual (ef2_w4's TriMul binding), any pair_transition / TriMul instance (ef2_w4, the kit's M1), or the kit's MSAEncoderBlock.forward (M1).
  * order: ef2_msa.enable(model) -> ef2_w4.enable(...) -> ef2_opt.install(...) (graph capture LAST).  enable()/disable() REFUSE (AssertionError)
    when ef2_opt already holds captured msa_encoder graphs for the model (levers would silently be absent from / frozen into the replayed graph).
  * capture-safe: no .item()/D2H, no data-dependent host control flow, static shapes per (L, M); the in-process probe is skipped while capturing.
Usage:  import ef2_msa; ef2_msa.enable(model, opm=True, pwa=True, msa_transition=True, ln_bf16=True)  |  EF2_MSA="t11,t12,t13,t14" ef2_msa.enable_from_env(model)
        ef2_msa.disable(model); ef2_msa.describe(); ef2_msa.stats()
Credits: Biohub (FusedLNLinearSwiGLU + model code); the FlashPairformer MSA kernels fpf_msa_chunk 0.1.0 (T12 construction) and fpf_msa 0.2.3
(single-chain projection epilogue idea); ef2_opt / ef2_w4 for the framework conventions.
"""
import os, types, collections
import torch
import triton
import triton.language as tl
from transformers.models.esmfold2.kernels.fused_lnlin_swiglu import FusedLNLinearSwiGLU

VERSION = "msa.0.1.1"
STATS = collections.Counter()
PROBES = []                       # (lever, shape, max_abs_diff, bitwise, ref_absmax, max_rel_diff)
_STATE = dict(enabled=False, opm=False, pwa=False, msa_transition=False, ln_bf16=False, probe=False, probe_n=2, models=[], t12_bitwise=None, t11_impl="own")
_BF16 = torch.bfloat16


def _cfg(name, default):
    """tile config override: EF2_MSA_<NAME>_CFG='k=v,k=v'"""
    s = os.environ.get(f"EF2_MSA_{name}_CFG", "")
    d = dict(default)
    for kv in [x for x in s.split(",") if x]:
        k, v = kv.split("="); d[k] = int(v)
    return d


# =====================================================================================================================
# T14: bf16-in / bf16-out row LayerNorm (fp32 statistics) for LayerNorm(128) on the MSA representation
# =====================================================================================================================
@triton.jit
def _ln_rows_kernel(X_ptr, W_ptr, B_ptr, Y_ptr, R, stride_x, stride_y, eps, K: tl.constexpr, BR: tl.constexpr, HAS_BIAS: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, K)
    rm = rows < R
    rows = rows.to(tl.int64)                                             # 64-bit row offsets: R = L x M rows of K, R*K passes 2^31 at large L x depth
    x = tl.load(X_ptr + rows[:, None] * stride_x + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / K
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / K
    rstd = 1.0 / tl.sqrt_rn(var + eps)
    w = tl.load(W_ptr + cols).to(tl.float32)
    y = xc * rstd[:, None] * w[None, :]
    if HAS_BIAS:
        y = y + tl.load(B_ptr + cols).to(tl.float32)[None, :]
    tl.store(Y_ptr + rows[:, None] * stride_y + cols[None, :], y.to(Y_ptr.dtype.element_ty), mask=rm[:, None])


def ln_bf16(x, norm):
    """LayerNorm(norm) applied to bf16 x [..., K] -> bf16 (fp32 math).  K must be a power of two <= 1024."""
    K = x.shape[-1]
    x2 = x.contiguous().view(-1, K)
    y = torch.empty_like(x2)
    BR = 32 if K <= 128 else 8
    _ln_rows_kernel[(triton.cdiv(x2.shape[0], BR),)](x2, norm.weight, norm.bias if norm.bias is not None else norm.weight, y, x2.shape[0], x2.stride(0), y.stride(0), norm.eps,
                                                      K=K, BR=BR, HAS_BIAS=norm.bias is not None, num_warps=4)
    STATS["t14_calls"] += 1
    return y.view(x.shape)


def _ln_ok(x, norm):
    K = x.shape[-1]
    return _STATE["ln_bf16"] and x.dtype == _BF16 and (K & (K - 1)) == 0 and K <= 1024 and norm.weight.shape[0] == K


# =====================================================================================================================
# T11: msa_transition through Biohub's FusedLNLinearSwiGLU (unmodified) + stock w3
# =====================================================================================================================
def _t11_build(tr):
    d_model = tr.norm.normalized_shape[0]; d_inner = tr.ffn.hidden_features
    fused = FusedLNLinearSwiGLU(d_model=d_model, d_inner=d_inner, has_ln_bias=tr.norm.bias is not None, device=tr.ffn.w12.weight.device, dtype=tr.ffn.w12.weight.dtype)
    with torch.no_grad():
        fused.LN_W.copy_(tr.norm.weight)
        if tr.norm.bias is not None: fused.LN_B.copy_(tr.norm.bias)
        fused.W12.copy_(tr.ffn.w12.weight.t().contiguous())        # same construction as C.Transition.set_kernel_backend('fused')
    tr._ef2msa_fused = fused.eval().requires_grad_(False)
    tr._ef2msa_wkey = (tr.ffn.w12.weight.data_ptr(), tr.ffn.w12.weight._version, tr.norm.weight.data_ptr(), tr.norm.weight._version)
    return tr._ef2msa_fused


def _msa_transition_forward_t11(self, x):
    """PairTransition.forward replacement (instance level).  Returns ffn(norm(x)) like stock; the residual is added by MSAEncoderBlock."""
    if torch.is_grad_enabled() or (not x.is_cuda) or x.dtype != _BF16 or self.ffn.w12.bias is not None:
        STATS["t11_fallthrough"] += 1
        return self._ef2msa_orig_forward(x)
    fused = getattr(self, "_ef2msa_fused", None)
    if fused is None or self._ef2msa_wkey != (self.ffn.w12.weight.data_ptr(), self.ffn.w12.weight._version, self.norm.weight.data_ptr(), self.norm.weight._version):
        fused = _t11_build(self)
    hidden = fused(x)                    # bf16 [.., hidden] = silu(x1)*x2 from fp32 accumulators of LN(x) @ W12 (Biohub kernel, custom_fwd bf16)
    out = self.ffn.w3(hidden)            # the stock w3 Linear (cuBLAS bf16 under autocast)
    STATS["t11_calls"] += 1
    _verify("t11", self, (x,), out)
    return out


@triton.jit
def _silu_mul_kernel(X12_ptr, OUT_ptr, R, HID: tl.constexpr, BR: tl.constexpr, BH: tl.constexpr):
    """out[r, c] = bf16( f32(silu_bf16(x1)) * f32(x2) ),  x1 = x12[r, c], x2 = x12[r, HID + c]  — reproduces F.silu(x1) * x2 on bf16 tensors:
    ATen silu: bf16( x / (1 + exp(-x)) in fp32 opmath ), then mul: bf16( f32 * f32 )."""
    r = tl.program_id(0) * BR + tl.arange(0, BR); c = tl.program_id(1) * BH + tl.arange(0, BH)
    m = (r < R)[:, None] & (c < HID)[None, :]
    r = r.to(tl.int64)                                                   # 64-bit row offsets: R = L x M rows of 2*HID, R*2*HID passes 2^31 at L x M >= 2^21
    x1 = tl.load(X12_ptr + r[:, None] * (2 * HID) + c[None, :], mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(X12_ptr + r[:, None] * (2 * HID) + (HID + c)[None, :], mask=m, other=0.0).to(tl.float32)
    s = (x1 / (1.0 + tl.exp(-x1))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT_ptr + r[:, None] * HID + c[None, :], (s * x2).to(tl.bfloat16), mask=m)


def silu_mul(x12):
    """x12 [..., 2*HID] bf16 contiguous -> [..., HID] bf16 = F.silu(x1) * x2 (chunk order: first half is the silu input)."""
    HID = x12.shape[-1] // 2
    x2d = x12.contiguous().view(-1, 2 * HID)
    out = torch.empty((x2d.shape[0], HID), device=x12.device, dtype=x12.dtype)
    BR, BH = 8, 256 if HID % 256 == 0 else 128
    _silu_mul_kernel[(triton.cdiv(x2d.shape[0], BR), triton.cdiv(HID, BH))](x2d, out, x2d.shape[0], HID=HID, BR=BR, BH=BH, num_warps=4)
    return out.view(*x12.shape[:-1], HID)


def _msa_transition_forward_t11b(self, x):
    """T11b (own construction, replaces T11): T14 bf16 LN (fp32 statistics) -> stock w12 GEMM -> fused silu(x1)*x2 (stock rounding points) -> stock w3.
    TIER-2: only the LN statistics reduction order and the exp implementation inside silu differ from stock (no bf16 storage of LN statistics)."""
    if torch.is_grad_enabled() or (not x.is_cuda) or x.dtype != _BF16 or self.ffn.w12.bias is not None or not _ln_ok_t11(x, self.norm) or (self._chunk_size is not None and x.shape[1] > self._chunk_size):
        STATS["t11_fallthrough"] += 1
        return self._ef2msa_orig_forward(x)
    xn = ln_bf16(x, self.norm)             # bf16 [.., 128]
    x12 = self.ffn.w12(xn)                 # stock GEMM (cuBLAS bf16) [.., 1024]
    hidden = silu_mul(x12)                 # bf16 [.., 512]
    out = self.ffn.w3(hidden)              # stock GEMM
    STATS["t11_calls"] += 1
    _verify("t11b", self, (x,), out)
    return out


def _ln_ok_t11(x, norm):
    K = x.shape[-1]
    return x.dtype == _BF16 and (K & (K - 1)) == 0 and K <= 1024 and norm.weight.shape[0] == K


# =====================================================================================================================
# T12: OuterProductMean — cuBLAS contraction on the stock problem + fused projection/bias/normalise epilogue (Triton)
# =====================================================================================================================
@triton.jit
def _opm_proj_kernel(WS_ptr, WT_ptr, BIAS_ptr, NV_ptr, OUT_ptr, L, stride_ws_row, stride_out_i, stride_out_j, stride_nv_i,
                     C_HID: tl.constexpr, D_OUT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """For one i (pid 0), BM j's (pid 1), BN n's (pid 2):  acc[j, n] = sum_k ws[i, c(k), j, d(k)] * Wt[k, n],  k = c*C + d (= stock flatten(-2) order),
    single ascending fp32 chain over k in BK chunks; y = bf16(acc + bias[n]); out[i, j, n] = bf16( div_rn(f32(y), f32(nv[i, j])) ).
    ws = cuBLAS output of [(i c), M] x [M, (j d)] -> element (i,c,j,d) at (i*C + c)*stride_ws_row + j*C + d (each (i,c) row: j*C+d contiguous)."""
    i = tl.program_id(0).to(tl.int64); jb = tl.program_id(1); nb = tl.program_id(2)   # i in 64 bits: the ws row offset (i*C + c)*stride_ws_row reaches (L*C)^2,
    offs_j = jb * BM + tl.arange(0, BM); offs_n = nb * BN + tl.arange(0, BN)         # past 2^31 from L = 1449 at C = 32, and the out row offset i*stride_out_i
    jmask = offs_j < L                                                               # reaches L*L*D_OUT
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    KTOT: tl.constexpr = C_HID * C_HID
    for k0 in tl.range(0, KTOT, BK):
        offs_k = k0 + tl.arange(0, BK)
        cc = offs_k // C_HID; dd = offs_k % C_HID
        a = tl.load(WS_ptr + (i * C_HID + cc)[None, :] * stride_ws_row + offs_j[:, None] * C_HID + dd[None, :], mask=jmask[:, None], other=0.0)
        b = tl.load(WT_ptr + offs_k[:, None] * D_OUT + offs_n[None, :])
        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float32)
    bias = tl.load(BIAS_ptr + offs_n).to(tl.float32)
    y = (acc + bias[None, :]).to(tl.bfloat16).to(tl.float32)
    nv = tl.load(NV_ptr + i * stride_nv_i + offs_j, mask=jmask, other=1.0).to(tl.float32)
    nvb = tl.broadcast_to(nv[:, None], (BM, BN))
    o = tl.div_rn(y, nvb)
    tl.store(OUT_ptr + i * stride_out_i + offs_j[:, None] * stride_out_j + offs_n[None, :], o.to(tl.bfloat16), mask=jmask[:, None])


_T12_DEFAULT = dict(BM=128, BN=256, BK=64, num_warps=8, num_stages=3)     # optest2 sweep (EXACT under every variant tried)


def _opm_weights(opm):
    w, b = opm.Wout.weight, opm.Wout.bias
    key = (w.data_ptr(), w._version, b.data_ptr(), b._version)
    ent = getattr(opm, "_ef2msa_wc", None)
    if ent is None or ent[0] != key:
        ent = (key, w.detach().to(_BF16).t().contiguous(), b.detach().to(_BF16).contiguous())     # Wt [1024, 256] bf16 (== autocast's cast of Wout.weight), bias bf16
        opm._ef2msa_wc = ent
    return ent[1], ent[2]


def opm_core_t12(self, m, msa_attention_mask, cfg=None):
    """m [1, L, M, 128] bf16, msa_attention_mask [1, L, M] (float, {0,1}).  Returns bf16 [1, L, L, 256] like the stock forward under bf16 autocast."""
    cfg = cfg or _cfg("T12", _T12_DEFAULT)
    B, L, M, _ = m.shape
    C_ = self.d_hidden
    if _ln_ok(m, self.norm):
        xw = self.W(ln_bf16(m, self.norm))                                # T14: bf16 LN -> the same W GEMM without the cast pass
    else:
        xw = self.W(self.norm(m))                                         # stock: fp32 LN, autocast casts to bf16 inside the Linear
    ab = xw * msa_attention_mask.unsqueeze(-1).to(xw.dtype)               # bf16; bitwise == bf16(fp32(xw) * mask) for mask in {0,1}
    a = ab[0, :, :, :C_]; b = ab[0, :, :, C_:]
    A2 = a.permute(0, 2, 1).reshape(L * C_, M)                            # [(i c), M]  (the same operand copies einsum makes)
    B2 = b.permute(0, 2, 1).reshape(L * C_, M)                            # [(j d), M]
    ws = torch.matmul(A2, B2.t())                                         # [(i c), (j d)] bf16 = stock einsum('bimc,bjmd->bijcd') storage (P1: bitwise)
    mask_f = msa_attention_mask.to(torch.float32)                         # stock: mask_f = mask.to(a.dtype) with a fp32
    n_valid = (mask_f @ mask_f.transpose(-1, -2)).clamp(min=1.0)          # under autocast -> bf16 [1, L, L] exactly as stock (values in {1..M} where exact)
    Wt, bias = _opm_weights(self)
    D = Wt.shape[1]
    out = torch.empty((B, L, L, D), device=m.device, dtype=_BF16)
    grid = (L, triton.cdiv(L, cfg["BM"]), D // cfg["BN"])
    _opm_proj_kernel[grid](ws, Wt, bias, n_valid[0], out[0], L, ws.stride(0), out.stride(1), out.stride(2), n_valid.stride(1),
                           C_HID=C_, D_OUT=D, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return out


def _opm_forward_t12(self, m, msa_attention_mask):
    cfg = _cfg("T12", _T12_DEFAULT)
    ok = (not torch.is_grad_enabled() and m.is_cuda and m.dtype == _BF16 and self._chunk_size is None and not self.divide_outer_before_proj and m.dim() == 4
          and m.shape[0] == 1 and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == _BF16
          and self.Wout.out_features % cfg["BN"] == 0 and (self.d_hidden * self.d_hidden) % cfg["BK"] == 0 and msa_attention_mask is not None and msa_attention_mask.dim() == 3)
    if not ok:
        STATS["t12_fallthrough"] += 1
        return self._ef2msa_orig_forward(m, msa_attention_mask)
    key = ("t12", tuple(m.shape))
    if key in self._ef2msa_bad:
        STATS["t12_routed_to_stock"] += 1
        return self._ef2msa_orig_forward(m, msa_attention_mask)
    out = opm_core_t12(self, m, msa_attention_mask, cfg)
    STATS["t12_calls"] += 1
    # first call per shape ALWAYS probed (cuBLAS-kernel / chain-order insurance); more calls when probe=True
    nmax = _STATE["probe_n"] if _STATE["probe"] else 1
    if self._ef2msa_seen[key] < nmax and not torch.cuda.is_current_stream_capturing():
        self._ef2msa_seen[key] += 1
        ref = self._ef2msa_orig_forward(m, msa_attention_mask)
        bit = bool(torch.equal(ref, out)); dmax = float((ref.float() - out.float()).abs().max()); scale = float(ref.float().abs().max())
        pure = not _ln_ok(m, self.norm)                                      # T14 off -> this probe is evidence about the EXACT lever T12 itself
        PROBES.append(("t12" if pure else "t12+t14", key[1], dmax, bit, scale, dmax / max(scale, 1e-30))); STATS["t12_probed" if pure else "t12+t14_probed"] += 1
        if pure:
            if _STATE["t12_bitwise"] is None or (_STATE["t12_bitwise"] and not bit):
                _STATE["t12_bitwise"] = bit
            if not bit: STATS["t12_verify_not_bitwise"] += 1
        if not bit:
            if dmax > 0.05 * scale + 1e-2:                                  # gross => a real bug or a wrong-layout cuBLAS result: route this shape to stock, loudly
                self._ef2msa_bad.add(key); STATS["t12_MISMATCH_routed_to_stock"] += 1
                print(f"[ef2_msa] T12 MISMATCH at shape {key[1]}: max|d| {dmax:.3e} (scale {scale:.3e}) -> stock for this shape", flush=True)
                return ref
    return out


# =====================================================================================================================
# T13: MSAPairWeightedAveraging — fused (attn . v) * gate (Triton); stock LN / bias / softmax / projections
# =====================================================================================================================
@triton.jit
def _pwa_kernel(ATT_ptr, V_ptr, G_ptr, OUT_ptr, L, M, s_att_i, s_att_j, s_att_h, s_row,
                H: tl.constexpr, DH: tl.constexpr, BI: tl.constexpr, BMM: tl.constexpr, BJ: tl.constexpr):
    """out[i, m, h*DH+d] = bf16( f32(bf16( sum_j att[i,j,h] * v[j,m,h*DH+d] )) * f32(g[i,m,h*DH+d]) )   (fp32 accumulate over j)
    att [L, L, H] bf16; v, g, out: [L, M, H*DH] bf16 rows of stride s_row (element (r, m, e) at r*s_row + m*H*DH + e).
    program = (i-block [pid0, fastest -> consecutive CTAs share the v tile in L2], m-block, head)."""
    ib = tl.program_id(0); mb = tl.program_id(1); h = tl.program_id(2)
    offs_i = ib * BI + tl.arange(0, BI)
    idx = tl.arange(0, BMM * DH)
    mm_ = mb * BMM + idx // DH
    col = mm_ * (H * DH) + h * DH + (idx % DH)                       # [BMM*DH] column offsets inside one row
    cmask = mm_ < M
    imask = offs_i < L
    offs_i = offs_i.to(tl.int64)                                         # 64-bit row offsets: i*s_att_i reaches L*L*H, i*s_row and j*s_row reach L*M*H*DH
    acc = tl.zeros((BI, BMM * DH), dtype=tl.float32)
    for j0 in tl.range(0, L, BJ):
        offs_j = j0 + tl.arange(0, BJ); jm = offs_j < L
        offs_j = offs_j.to(tl.int64)
        att = tl.load(ATT_ptr + offs_i[:, None] * s_att_i + offs_j[None, :] * s_att_j + h * s_att_h, mask=imask[:, None] & jm[None, :], other=0.0)
        vt = tl.load(V_ptr + offs_j[:, None] * s_row + col[None, :], mask=jm[:, None] & cmask[None, :], other=0.0)
        acc = tl.dot(att, vt, acc, input_precision="ieee", out_dtype=tl.float32)
    av = acc.to(tl.bfloat16).to(tl.float32)
    g = tl.load(G_ptr + offs_i[:, None] * s_row + col[None, :], mask=imask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    tl.store(OUT_ptr + offs_i[:, None] * s_row + col[None, :], (av * g).to(tl.bfloat16), mask=imask[:, None] & cmask[None, :])


_T13_DEFAULT = dict(BI=128, BMM=16, BJ=32, num_warps=8, num_stages=3)    # optest2 sweep best @705 (x1.15 alone; the lever is worth shipping only together with T14)


def pwa_core_t13(self, msa_repr, pair_repr, pair_attention_mask, cfg=None):
    cfg = cfg or _cfg("T13", _T13_DEFAULT)
    B, L, M, _ = msa_repr.shape
    h, dh = self.n_heads, self.head_width
    if _ln_ok(msa_repr, self.norm_single):
        msa_normed = ln_bf16(msa_repr, self.norm_single)                  # T14
    else:
        msa_normed = self.norm_single(msa_repr).to(_BF16)                  # stock fp32 LN; the bf16 cast autocast would do inside Wv AND Wgate, done once
    bias = self.compute_bias(pair_repr)                                    # stock [1, L, L, h] (bf16)
    bias = bias.masked_fill(~pair_attention_mask.unsqueeze(-1).bool(), -1e5)
    attn = torch.softmax(bias, dim=-2)                                     # fp32 under autocast (softmax over j), as stock
    attn_b = attn.to(_BF16)                                                # == autocast's cast of the einsum operand
    v = self.Wv(msa_normed)                                                # bf16 [1, L, M, h*dh]
    gate = torch.sigmoid(self.Wgate(msa_normed))                           # bf16
    out = torch.empty((B, L, M, h * dh), device=msa_repr.device, dtype=_BF16)
    a0, v0, g0, o0 = attn_b[0], v[0], gate[0], out[0]
    assert v0.stride(2) == 1 and v0.stride(1) == h * dh and g0.stride() == v0.stride() and o0.stride() == v0.stride()
    grid = (triton.cdiv(L, cfg["BI"]), triton.cdiv(M, cfg["BMM"]), h)
    _pwa_kernel[grid](a0, v0, g0, o0, L, M, a0.stride(0), a0.stride(1), a0.stride(2), v0.stride(0),
                      H=h, DH=dh, BI=cfg["BI"], BMM=cfg["BMM"], BJ=cfg["BJ"], num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return self.Wout(out)                                                  # stock projection on a contiguous [1, L, M, 128]


def _pwa_forward_t13(self, msa_repr, pair_repr, pair_attention_mask):
    dh = self.head_width
    ok = (not torch.is_grad_enabled() and msa_repr.is_cuda and msa_repr.dtype == _BF16 and msa_repr.dim() == 4 and msa_repr.shape[0] == 1
          and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == _BF16 and (dh & (dh - 1)) == 0 and 8 <= dh <= 64 and pair_attention_mask is not None)
    if not ok:
        STATS["t13_fallthrough"] += 1
        return self._ef2msa_orig_forward(msa_repr, pair_repr, pair_attention_mask)
    res = pwa_core_t13(self, msa_repr, pair_repr, pair_attention_mask)
    STATS["t13_calls"] += 1
    _verify("t13" + ("+t14" if _ln_ok(msa_repr, self.norm_single) else ""), self, (msa_repr, pair_repr, pair_attention_mask), res)
    return res


# =====================================================================================================================
# plumbing
# =====================================================================================================================
def _verify(lever, inst, args, out):
    if not _STATE["probe"] or torch.cuda.is_current_stream_capturing():
        return
    key = (lever, tuple(args[0].shape))
    if inst._ef2msa_seen[key] >= _STATE["verify_n"]:
        return
    inst._ef2msa_seen[key] += 1
    ref = inst._ef2msa_orig_forward(*args)
    dmax = float((ref.float() - out.float()).abs().max()); scale = float(ref.float().abs().max())
    PROBES.append((lever, key[1], dmax, bool(torch.equal(ref, out)), scale, dmax / max(scale, 1e-30)))     # (lever, shape, max|d|, bitwise, ref absmax, max|d|/absmax)
    STATS[lever + "_probed"] += 1


def _msa_blocks(model):
    enc = getattr(model, "msa_encoder", None)
    return [] if enc is None else list(enc.blocks)


def _assert_not_captured(model, what):
    enc = getattr(model, "msa_encoder", None)
    graphs = getattr(enc, "_ef2opt_graphs", None) if enc is not None else None
    assert not graphs, (f"ef2_msa.{what}() called while ef2_opt holds {len(graphs)} captured msa_encoder CUDA graph(s): the change would not reach the replayed graph. "
                        "Order: ef2_msa.enable -> ef2_w4.enable -> ef2_opt.install  (or ef2_opt.clear_graphs(model) first).")


def _patch(inst, fn):
    if getattr(inst, "_ef2msa_orig_forward", None) is None:
        inst._ef2msa_orig_forward = inst.forward                # bound stock method (or a pre-existing instance-level override, which we then wrap)
        inst._ef2msa_had_inst_fwd = "forward" in inst.__dict__
    inst._ef2msa_seen = collections.Counter(); inst._ef2msa_bad = set()
    inst.forward = types.MethodType(fn, inst)


def _unpatch(inst):
    orig = getattr(inst, "_ef2msa_orig_forward", None)
    if orig is None:
        return
    if inst.__dict__.get("_ef2msa_had_inst_fwd"):
        inst.forward = orig
    elif "forward" in inst.__dict__:
        del inst.__dict__["forward"]
    inst._ef2msa_orig_forward = None


def enable(model, opm=False, pwa=False, msa_transition=False, ln_bf16=False, probe=False, probe_n=None, t11_impl=None):
    """Install the requested levers on model.msa_encoder.blocks[*] (per instance).  Call BEFORE ef2_opt.install(..., encoder_graphs=True).
    msa_transition: T11b (own: bf16 LN fp32-stats -> stock w12 -> fused silu*mul -> stock w3).  t11_impl="biohub" selects the Biohub FusedLNLinearSwiGLU
    variant (T11a) instead — it FAILED the Tier-2 bar at op level (bf16-stored LN statistics; err-vs-fp64 1.4x/1.9x stock's) and is kept for the record only."""
    _assert_not_captured(model, "enable")
    _STATE["t11_impl"] = (t11_impl or os.environ.get("EF2_MSA_T11_IMPL", "own")).lower()
    blocks = _msa_blocks(model)
    _STATE.update(enabled=bool(blocks) and bool(opm or pwa or msa_transition), opm=bool(opm), pwa=bool(pwa), msa_transition=bool(msa_transition), ln_bf16=bool(ln_bf16),
                  probe=bool(probe), probe_n=int(probe_n if probe_n is not None else 2))
    if model not in _STATE["models"]: _STATE["models"].append(model)
    n = collections.Counter()
    for blk in blocks:
        if opm: _patch(blk.outer_product_mean, _opm_forward_t12); n["opm"] += 1
        else: _unpatch(blk.outer_product_mean)
        if not blk.is_final_block:
            if pwa: _patch(blk.msa_pair_weighted_averaging, _pwa_forward_t13); n["pwa"] += 1
            else: _unpatch(blk.msa_pair_weighted_averaging)
            if msa_transition: _patch(blk.msa_transition, _msa_transition_forward_t11 if _STATE["t11_impl"] == "biohub" else _msa_transition_forward_t11b); n["msa_transition"] += 1
            else: _unpatch(blk.msa_transition)
    d = describe(); d["patched"] = dict(n)
    return d


def disable(model=None):
    for mdl in ([model] if model is not None else list(_STATE["models"])):
        _assert_not_captured(mdl, "disable")
        for blk in _msa_blocks(mdl):
            _unpatch(blk.outer_product_mean)
            if not blk.is_final_block:
                _unpatch(blk.msa_pair_weighted_averaging); _unpatch(blk.msa_transition)
    _STATE.update(enabled=False, opm=False, pwa=False, msa_transition=False, ln_bf16=False, probe=False)
    return describe()


def labels():
    t12 = _STATE["t12_bitwise"]
    l12 = ("TIER-2 (T14 bf16 LN statistics order + K=1024 projection order; rounding points kept)" if _STATE["ln_bf16"] else
           "EXACT (bitwise vs stock, probed in-process on every shape seen)" if t12 is True else
           "TIER-2 (same rounding points; fp32 accumulation order of the K=1024 projection) — in-process probe NOT bitwise" if t12 is False else
           "EXACT-if-probed (no call probed yet)")
    l11 = ("TIER-2 (T11b own: LN statistics reduction order + exp impl in silu; stock rounding points, fp32 statistics)" if _STATE["t11_impl"] != "biohub" else
           "T11a Biohub FusedLNLinearSwiGLU — FAILS the Tier-2 bar (bf16-stored LN mean/rstd): record only")
    return dict(t11=l11, t12=l12,
                t13="TIER-2 (fp32 accumulation order over j; rounding points kept)" + (" + T14 LN" if _STATE["ln_bf16"] else ""), t14="TIER-2 (LN statistics reduction order)")


def describe():
    return dict(version=VERSION, enabled=_STATE["enabled"], opm_t12=_STATE["opm"], pwa_t13=_STATE["pwa"], msa_transition_t11=_STATE["msa_transition"], t11_impl=_STATE["t11_impl"], ln_bf16_t14=_STATE["ln_bf16"],
                probe=_STATE["probe"], t12_bitwise_in_process=_STATE["t12_bitwise"], labels=labels(),
                tiles=dict(t12=_cfg("T12", _T12_DEFAULT), t13=_cfg("T13", _T13_DEFAULT)))


def stats():
    d = dict(STATS)
    d["probe"] = [list(v) for v in PROBES]
    levers = sorted({v[0] for v in PROBES})
    d["probe_bitwise_by_lever"] = {k: f"{sum(1 for v in PROBES if v[0] == k and v[3])}/{sum(1 for v in PROBES if v[0] == k)}" for k in levers}
    d["probe_max_abs_diff_by_lever"] = {k: max(v[2] for v in PROBES if v[0] == k) for k in levers}
    d["probe_max_rel_diff_by_lever"] = {k: max(v[5] for v in PROBES if v[0] == k) for k in levers}          # max|d| / max|ref| per call, worst call (bf16 ulp = 2^-8 rel ~ 3.9e-3)
    d["t12_bitwise_in_process"] = _STATE["t12_bitwise"]
    return d


def parse_flags(spec):
    fl = set(x for x in spec.strip().lower().replace("+", ",").split(",") if x)
    if "all" in fl: fl |= {"t11", "t12", "t13", "t14"}
    d = dict(opm=bool(fl & {"t12", "opm"}), pwa=bool(fl & {"t13", "pwa"}), msa_transition=bool(fl & {"t11", "t11a", "t11b", "msa_transition"}), ln_bf16=bool(fl & {"t14", "ln_bf16", "ln"}))
    if "t11a" in fl: d["t11_impl"] = "biohub"
    return d


def enable_from_env(model, default=""):
    spec = os.environ.get("EF2_MSA", default)
    if not spec or spec.strip().lower() in ("0", "off", "none"):
        return None
    return enable(model, **parse_flags(spec))
