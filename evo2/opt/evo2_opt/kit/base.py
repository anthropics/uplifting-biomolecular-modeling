"""The Hyena-side levers (the Triton kernels, caches and op chains shared by every route): the cached modal filter (L1b), the Triton 3-tap featurizer FIR (E7), the fused RMSNorm tail (E11), the bf16 depthwise FIR / fused short-FIR chains (E8, E7-hcs7, E9a/E9b), the persistent padded-FFT and spectrum buffers (E10, L1c), the direct cuFFT plans (E59/E60), the interleave fold (L2) and the fused bias+residual out-projection epilogue (E50/E56). Every chain reproduces the stock operation order per element; the stock originals are captured at import in _ORIGINALS."""
import collections
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import vortex.model.engine as _eng
from vortex.model.engine import HyenaInferenceEngine, fftconv_func, adjust_filter_shape_for_broadcast as _adj
from vortex.model.model import HyenaCascade, ParallelGatedConvBlock, AttentionBlock, StripedHyena
from vortex.model.layers import RMSNorm, HAS_TE

CTR = collections.Counter()
HAVE_TRITON = True
_orig_compute_filter = HyenaCascade.compute_filter
_orig_parallel_iir = HyenaInferenceEngine.parallel_iir
_orig_parallel_fir = HyenaInferenceEngine.parallel_fir
_orig_rms_fwd = RMSNorm.forward
from evo2_opt.kit.lkeyed import OneLength as _OneLength                        # noqa: E402
# the per-length caches below hold ONE length at a time: a write for a new L releases the previous length's filters and spectra (lkeyed.hold)
_filter_cache, _H_cache, _kf_cache = _OneLength("v6_hcl_filters"), _OneLength("v6_hcl_spectra"), _OneLength("v6_hcm_spectra")
_pad_bufs = {}
KCFG = {"fir3": (64, 64, 4), "hcs7": (32, 128, 4), "rms": 8}      # launch geometry only; per-element arithmetic unchanged

def _cached_compute_filter(self, L, device):
    key = (id(self), int(L), str(device))
    if key not in _filter_cache:
        CTR["hcl_filter_cache_miss"] += 1
        _filter_cache[key] = _compute_filter_tiled(self, L, device)
    else:
        CTR["hcl_filter_cache_hit"] += 1; self.update_time(L, device)   # a hit sets self.t as stock's compute_filter does on every call (vortex model.py:387-401): a generation prefill reads t (engine.prefill_via_modal_fft, reached under inference_params only — scoring never reads it); on this line so that no jit kernel below moves (a kernel's starting line is part of its cache key)
    return _filter_cache[key]

def _parallel_fir_bf16(self, fir_fn, u, weight, bias, L, dims, groups=None, gated_bias=False, column_split_hyena=False, dim_last=True,
                       fir_length=3, gate=False, inference_params=None, prefill_mode=None, padding_mask=None):
    if fir_length >= 128 or fir_fn != torch.nn.functional.conv1d or inference_params is not None:
        if fir_length < 128: CTR["fir_bf16_fallback"] += 1
        return _orig_parallel_fir(self, fir_fn, u, weight, bias, L, dims, groups=groups, gated_bias=gated_bias, column_split_hyena=column_split_hyena,
                                  dim_last=dim_last, fir_length=fir_length, gate=gate, inference_params=inference_params, prefill_mode=prefill_mode, padding_mask=padding_mask)
    CTR["fir_bf16_conv1d"] += 1
    L = u.shape[1] if dim_last else u.shape[2]
    if gate:
        hidden_size = dims[0]
        x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
        if self.hyena_flip_x1x2: x1, x2 = x2, x1
        u = x1 * v
    if dim_last: u = u.permute(0, 2, 1)
    z = fir_fn(u, weight.to(u.dtype), bias=None, stride=1, padding=fir_length - 1, groups=u.shape[1])[..., :L]
    if bias is not None:
        z = z + bias[None, :, None] * u if gated_bias else z + bias[None, :, None]
    if type(padding_mask) == torch.Tensor: z = z * padding_mask[:, None]
    if gate: z = x2 * z
    return z, None

@triton.jit
def _fir3_kernel(x_ptr, w_ptr, o_ptr, B, D3, L, sxb, sxl, sxd, sob, sod, sol, ORDER: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    # x: (B, L, D3) any strides; w: (D3, 3) contiguous; out: (B, D3, L). correlation: o[t] = sum_k w[k] * x[t-2+k]
    pb = tl.program_id(0); pd = tl.program_id(1); pl = tl.program_id(2)
    od = pd * BLOCK_D + tl.arange(0, BLOCK_D); ol = pl * BLOCK_L + tl.arange(0, BLOCK_L)
    md = od < D3; ml = ol < L
    w0 = tl.load(w_ptr + od * 3 + 0, mask=md, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + od * 3 + 1, mask=md, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + od * 3 + 2, mask=md, other=0.0).to(tl.float32)
    base = x_ptr + pb * sxb + od[None, :] * sxd
    t0 = ol[:, None] - 2; t1 = ol[:, None] - 1; t2 = ol[:, None]
    x0 = tl.load(base + t0 * sxl, mask=(t0 >= 0) & ml[:, None] & md[None, :], other=0.0).to(tl.float32)
    x1 = tl.load(base + t1 * sxl, mask=(t1 >= 0) & ml[:, None] & md[None, :], other=0.0).to(tl.float32)
    x2 = tl.load(base + t2 * sxl, mask=ml[:, None] & md[None, :], other=0.0).to(tl.float32)
    m0 = (t0 >= 0) & ml[:, None] & md[None, :]; m1 = (t1 >= 0) & ml[:, None] & md[None, :]; m2 = ml[:, None] & md[None, :]
    p0 = tl.where(m0, w0[None, :] * x0, 0.0); p1 = tl.where(m1, w1[None, :] * x1, 0.0); p2 = tl.where(m2, w2[None, :] * x2, 0.0)   # invalid taps = exact +0.0 (ATen skips them)
    if ORDER == 0:
        acc = ((0.0 + p0) + p1) + p2      # ascending kW order from +0.0 = ATen conv_depthwise2d_forward_kernel_generic
    elif ORDER == 1:
        acc = (p2 + p1) + p0
    elif ORDER == 2:
        acc = (p0 + p2) + p1
    else:
        acc = p0 + (p1 + p2)
    out = acc.to(tl.bfloat16)
    tl.store(o_ptr + pb * sob + od[None, :] * sod + ol[:, None] * sol, out, mask=ml[:, None] & md[None, :])

def fir3_triton(x_bld, w, order, BLOCK_D=None, BLOCK_L=None):
    """x_bld: (B, L, D3) bf16 (the projection output as produced, no permute/cast); w: (D3,1,3) bf16 -> (B, D3, L) bf16"""
    bd, bl, nw = KCFG["fir3"]; BLOCK_D = BLOCK_D or bd; BLOCK_L = BLOCK_L or bl
    Bn, Ln, D3 = _i32_fir3(x_bld)                                            # the shape, once it fits the kernel's 32-bit element addressing (refused before the launch otherwise: _i32_fir3, i32.py)
    out = torch.empty((Bn, D3, Ln), dtype=torch.bfloat16, device=x_bld.device)
    w2 = w.reshape(D3, 3).contiguous()
    grid = (Bn, triton.cdiv(D3, BLOCK_D), triton.cdiv(Ln, BLOCK_L))
    _fir3_kernel[grid](x_bld, w2, out, Bn, D3, Ln, x_bld.stride(0), x_bld.stride(1), x_bld.stride(2), out.stride(0), out.stride(1), out.stride(2),
                       ORDER=order, BLOCK_D=BLOCK_D, BLOCK_L=BLOCK_L, num_warps=nw)
    return out

def _pad_buf(shape, dtype, device):
    key = (tuple(shape), str(dtype), str(device))
    if key not in _pad_bufs:
        _pad_bufs[key] = torch.zeros(*shape, dtype=dtype, device=device)
    return _pad_bufs[key]

def _fftconv_cached(u, k, D, dropout_mask, gelu=True, k_rev=None, bidirectional=False, layer_idx=None, **kwargs):
    assert not bidirectional and k_rev is None
    seqlen = u.shape[-1]; fft_size = 2 * seqlen
    key = (layer_idx, seqlen, str(u.device))
    if key not in _kf_cache:
        k_f = torch.fft.rfft(k, n=fft_size) / fft_size
        _kf_cache[key] = _adj(u, k_f)
    k_f = _kf_cache[key]; CTR["hcm_fftconv_cached"] += 1
    buf = _pad_buf((u.shape[0], u.shape[1], fft_size), torch.float32, u.device)
    buf[..., :seqlen].copy_(u)                       # tail stays zero; same values the stock's constant_pad_nd produces
    u_f = torch.fft.rfft(buf)                        # n == buf length: no pad; same onesided R2C plan shape as the stock's padded call
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)

def _parallel_iir_Hcache_padbuf(self, z_pre, h, D, L, poles, residues, t, dims, layer_idx, inference_params=None, prefill_style="fft",
                                fftconv_fn=None, padding_mask=None, use_flashfft=False, column_split_hyena=False, long_fir_threshold=None):
    assert inference_params is None and not use_flashfft and long_fir_threshold is None and not column_split_hyena
    fft_size = 2 * L; hidden_size = dims[0]
    x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)
    if self.hyena_flip_x1x2: x1, x2 = x2, x1
    x1v = x1 * v
    key = (layer_idx, int(L), str(h.device))
    if key not in _H_cache:
        _H_cache[key] = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
    H = _H_cache[key]; CTR["hcl_cached_iir"] += 1
    buf = _pad_buf((x1v.shape[0], x1v.shape[1], fft_size), torch.float32, x1v.device)
    buf[..., :L].copy_(x1v)                          # fused cast+copy into the first half; tail zero
    X_s = torch.fft.fft(buf)                         # real input -> ATen fft_r2c(onesided=False), n == length: no pad
    X = X_s[..., : H.shape[-1]]
    y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]
    y = y.to(dtype=x1v.dtype)
    y = (y + x1v * D.unsqueeze(-1)) * x2
    return y.permute(0, 2, 1)

@triton.jit
def _rms_tail_kernel(x_ptr, n_ptr, s_ptr, o_ptr, H, inv_sqrt_h, eps, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H); m = offs < H
    n = tl.load(n_ptr + row).to(tl.float32)
    d = (n * inv_sqrt_h).to(tl.bfloat16).to(tl.float32)       # bf16(norm * H^-0.5)
    d = (d + eps).to(tl.bfloat16).to(tl.float32)              # bf16(. + eps)
    x = tl.load(x_ptr + row * H + offs, mask=m, other=0.0).to(tl.float32)
    y = tl.math.div_rn(x, d).to(tl.bfloat16).to(tl.float32)   # bf16(x / d), IEEE round-to-nearest division as torch's
    s = tl.load(s_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * H + offs, (s * y).to(tl.bfloat16), mask=m)

def _rms_fused_forward(self, x):
    _i32_rows("E11_fused_rmsnorm_tail", "the RMSNorm input", x)
    xc = x.contiguous(); Hd = xc.shape[-1]
    if _RN.served(xc, self.scale):                             # one launch: the norm reduction in ATen's order + the tail (kit/rmsnorm.py)
        CTR["rms_fused_norm"] += 1
        return _RN.rmsnorm_forward(xc, self.scale, self.eps)
    CTR["rms_fused_tail"] += 1; _rms_two_launch_named(xc)
    n = x.norm(2, dim=-1, keepdim=True)                        # torch's own reduction (bf16 out), unchanged
    out = torch.empty_like(xc)
    rows = xc.numel() // Hd
    _rms_tail_kernel[(rows,)](xc.view(rows, Hd), n.contiguous().view(rows), self.scale, out.view(rows, Hd), Hd, float(Hd ** (-1.0 / 2)), float(self.eps),
                              BLOCK_H=triton.next_power_of_2(Hd), num_warps=KCFG["rms"])
    return out

@triton.jit
def _hcs7_kernel_s(x1_ptr, v_ptr, x2_ptr, w_ptr, d_ptr, o_ptr, B, D, L, sb, sd, sl, ob, od_, ol_, ORDER: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    pb = tl.program_id(0); pd = tl.program_id(1); pl = tl.program_id(2)
    od = pd * BLOCK_D + tl.arange(0, BLOCK_D); ol = pl * BLOCK_L + tl.arange(0, BLOCK_L)
    md = od < D; ml = ol < L
    base = pb * sb + od[:, None] * sd
    acc = tl.zeros((BLOCK_D, BLOCK_L), dtype=tl.float32)
    for k in tl.static_range(7):
        t = ol[None, :] - 6 + k
        m = (t >= 0) & ml[None, :] & md[:, None]
        u = (tl.load(x1_ptr + base + t * sl, mask=m, other=0.0).to(tl.float32) * tl.load(v_ptr + base + t * sl, mask=m, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        wk = tl.load(w_ptr + od * 7 + k, mask=md, other=0.0).to(tl.float32)
        acc = acc + tl.where(m, wk[:, None] * u, 0.0)   # ascending taps from +0.0; invalid taps add exact +0.0
    z = acc.to(tl.bfloat16).to(tl.float32)
    if HAS_BIAS:
        dk = tl.load(d_ptr + od, mask=md, other=0.0).to(tl.float32)
        z = (z + dk[:, None]).to(tl.bfloat16).to(tl.float32)
    m2 = ml[None, :] & md[:, None]
    x2 = tl.load(x2_ptr + base + ol[None, :] * sl, mask=m2, other=0.0).to(tl.float32)
    tl.store(o_ptr + pb * ob + od[:, None] * od_ + ol[None, :] * ol_, (x2 * z).to(tl.bfloat16), mask=m2)

def _parallel_fir_fused(self, fir_fn, u, weight, bias, L, dims, groups=None, gated_bias=False, column_split_hyena=False, dim_last=True,
                        fir_length=3, gate=False, inference_params=None, prefill_mode=None, padding_mask=None):
    if inference_params is not None or fir_fn != torch.nn.functional.conv1d or padding_mask is not None or fir_length >= 128:
        return _parallel_fir_bf16(self, fir_fn, u, weight, bias, L, dims, groups=groups, gated_bias=gated_bias, column_split_hyena=column_split_hyena,
                                  dim_last=dim_last, fir_length=fir_length, gate=gate, inference_params=inference_params, prefill_mode=prefill_mode, padding_mask=padding_mask)
    if not gate and dim_last and fir_length == 3 and bias is None and u.dim() == 3 and weight.shape[-1] == 3:
        CTR["fir3_triton_fwd"] += 1
        return fir3_triton(u, weight, 0), None                                   # (B, 3D, L) bf16, ascending tap order
    if gate and not dim_last and fir_length == 7 and weight.shape[-1] == 7 and not gated_bias:
        hidden_size = dims[0]
        x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
        if self.hyena_flip_x1x2: x1, x2 = x2, x1
        Bn, Dn, Ln = x1.shape
        assert x1.stride() == v.stride() == x2.stride()
        out = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=x1.device)
        w2 = weight.reshape(Dn, 7).contiguous()
        grid = (Bn, triton.cdiv(Dn, KCFG["hcs7"][0]), triton.cdiv(Ln, KCFG["hcs7"][1]))
        _hcs7_kernel_s[grid](x1, v, x2, w2, bias if bias is not None else w2, out, Bn, Dn, Ln, x1.stride(0), x1.stride(1), x1.stride(2),
                             out.stride(0), out.stride(1), out.stride(2), ORDER=0, HAS_BIAS=bias is not None, BLOCK_D=KCFG["hcs7"][0], BLOCK_L=KCFG["hcs7"][1], num_warps=KCFG["hcs7"][2])
        CTR["hcs7_triton_fwd"] += 1
        return out, None
    if fir_length < 128: CTR["fir_fused_fallback"] += 1
    return _parallel_fir_bf16(self, fir_fn, u, weight, bias, L, dims, groups=groups, gated_bias=gated_bias, column_split_hyena=column_split_hyena,
                              dim_last=dim_last, fir_length=fir_length, gate=gate, inference_params=inference_params, prefill_mode=prefill_mode, padding_mask=padding_mask)

def _bdl_grid(Bn, Dn, Ln, BD=32, BL=128):
    return (Bn, triton.cdiv(Dn, BD), triton.cdiv(Ln, BL))

@triton.jit
def _gate_padbuf_kernel(x1_ptr, v_ptr, buf_ptr, B, D, L, sb, sd, sl, bb, bd, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    # buf[b, d, t] = fp32(bf16(x1 * v)) for t < L (the stock's `x1v = x1 * v` then `.to(float32)`); tail untouched (zero)
    pb = tl.program_id(0); pd = tl.program_id(1); pl = tl.program_id(2)
    od = pd * BLOCK_D + tl.arange(0, BLOCK_D); ol = pl * BLOCK_L + tl.arange(0, BLOCK_L)
    m = (od < D)[:, None] & (ol < L)[None, :]
    offs = pb * sb + od[:, None] * sd + ol[None, :] * sl
    x1v = (tl.load(x1_ptr + offs, mask=m, other=0.0).to(tl.float32) * tl.load(v_ptr + offs, mask=m, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(buf_ptr + pb * bb + od[:, None] * bd + ol[None, :], x1v, mask=m)

@triton.jit
def _hcm_epilogue_kernel(y_ptr, x1_ptr, v_ptr, x2_ptr, d_ptr, o_ptr, B, D, L, yb, yd, sb, sd, sl, ob, od_, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    # stock (fp32 math under autocast): z = bf16(y + u*D) with u = fp32(bf16(x1*v)); then the bf16 gate x2 * z
    pb = tl.program_id(0); pd = tl.program_id(1); pl = tl.program_id(2)
    od = pd * BLOCK_D + tl.arange(0, BLOCK_D); ol = pl * BLOCK_L + tl.arange(0, BLOCK_L)
    md = od < D; m = md[:, None] & (ol < L)[None, :]
    offs = pb * sb + od[:, None] * sd + ol[None, :] * sl
    y = tl.load(y_ptr + pb * yb + od[:, None] * yd + ol[None, :], mask=m, other=0.0).to(tl.float32)
    u = (tl.load(x1_ptr + offs, mask=m, other=0.0).to(tl.float32) * tl.load(v_ptr + offs, mask=m, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    dk = tl.load(d_ptr + od, mask=md, other=0.0).to(tl.float32)
    z = (y + u * dk[:, None]).to(tl.bfloat16).to(tl.float32)
    x2 = tl.load(x2_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + pb * ob + od[:, None] * od_ + ol[None, :], (x2 * z).to(tl.bfloat16), mask=m)

@triton.jit
def _rne_bf16(x):
    # round an fp32 value to the nearest bf16 (ties to even) THROUGH INTEGER BITS: the compiler cannot fold this pair of casts
    # or contract the surrounding mul+add into an FMA (E9a debug: `.to(bf16).to(f32)` between a mul and an add was folded -> FMA)
    b = x.to(tl.int32, bitcast=True)
    b = b + 0x7FFF + ((b >> 16) & 1)
    b = b & -65536
    return b.to(tl.float32, bitcast=True)

@triton.jit
def _hcl_epilogue_kernel(y_ptr, x1_ptr, v_ptr, x2_ptr, d_ptr, o_ptr, B, D, L, yb, yd, sb, sd, sl, ob, od_, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    # stock: y.to(bf16); (y + bf16(x1v*D)) * x2 with bf16 rounding after every op (integer-RNE so no cast pair can be folded)
    pb = tl.program_id(0); pd = tl.program_id(1); pl = tl.program_id(2)
    od = pd * BLOCK_D + tl.arange(0, BLOCK_D); ol = pl * BLOCK_L + tl.arange(0, BLOCK_L)
    md = od < D; m = md[:, None] & (ol < L)[None, :]
    offs = pb * sb + od[:, None] * sd + ol[None, :] * sl
    y = _rne_bf16(tl.load(y_ptr + pb * yb + od[:, None] * yd + ol[None, :], mask=m, other=0.0))
    x1v = _rne_bf16(tl.load(x1_ptr + offs, mask=m, other=0.0).to(tl.float32) * tl.load(v_ptr + offs, mask=m, other=0.0).to(tl.float32))
    dk = tl.load(d_ptr + od, mask=md, other=0.0).to(tl.float32)
    t = _rne_bf16(x1v * dk[:, None])
    s = _rne_bf16(y + t)
    x2 = tl.load(x2_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + pb * ob + od[:, None] * od_ + ol[None, :], (s * x2).to(tl.bfloat16), mask=m)

def _parallel_fir_v2(self, fir_fn, u, weight, bias, L, dims, groups=None, gated_bias=False, column_split_hyena=False, dim_last=True,
                     fir_length=3, gate=False, inference_params=None, prefill_mode=None, padding_mask=None):
    if fir_length >= 128 and gate and not dim_last and inference_params is None and padding_mask is None and fir_fn == torch.nn.functional.conv1d and bias is not None:
        hidden_size = dims[0]
        x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
        if self.hyena_flip_x1x2: x1, x2 = x2, x1
        assert x1.stride() == v.stride() == x2.stride()
        Bn, Dn, Ln = x1.shape; fft_size = 2 * Ln
        key = (self.layer_idx, Ln, str(u.device))
        if key not in _kf_cache:
            k = weight[:, :, :Ln].to(torch.float32)
            k_f = torch.fft.rfft(k, n=fft_size) / fft_size
            _kf_cache[key] = _adj(torch.empty((Bn, Dn, Ln), device=u.device), k_f)
        k_f = _kf_cache[key]
        buf = _pad_buf((Bn, Dn, fft_size), torch.float32, u.device)
        _gate_padbuf_kernel[_bdl_grid(Bn, Dn, Ln)](x1, v, buf, Bn, Dn, Ln, x1.stride(0), x1.stride(1), x1.stride(2), buf.stride(0), buf.stride(1), BLOCK_D=32, BLOCK_L=128)
        u_f = torch.fft.rfft(buf)
        y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")       # (B, D, 2L) fp32
        out = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=u.device)
        _hcm_epilogue_kernel[_bdl_grid(Bn, Dn, Ln)](y, x1, v, x2, bias, out, Bn, Dn, Ln, y.stride(0), y.stride(1), x1.stride(0), x1.stride(1), x1.stride(2),
                                                    out.stride(0), out.stride(1), BLOCK_D=32, BLOCK_L=128)
        CTR["hcm_fused_v2"] += 1
        return out, None
    return _parallel_fir_fused(self, fir_fn, u, weight, bias, L, dims, groups=groups, gated_bias=gated_bias, column_split_hyena=column_split_hyena,
                               dim_last=dim_last, fir_length=fir_length, gate=gate, inference_params=inference_params, prefill_mode=prefill_mode, padding_mask=padding_mask)

def _parallel_iir_v2(self, z_pre, h, D, L, poles, residues, t, dims, layer_idx, inference_params=None, prefill_style="fft",
                     fftconv_fn=None, padding_mask=None, use_flashfft=False, column_split_hyena=False, long_fir_threshold=None):
    assert inference_params is None and not use_flashfft and long_fir_threshold is None and not column_split_hyena and padding_mask is None
    fft_size = 2 * L; hidden_size = dims[0]
    x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)
    if self.hyena_flip_x1x2: x1, x2 = x2, x1
    assert x1.stride() == v.stride() == x2.stride()
    key = (layer_idx, int(L), str(h.device))
    if key not in _H_cache:
        _H_cache[key] = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
    H = _H_cache[key]
    Bn, Dn, Ln = x1.shape
    buf = _pad_buf((Bn, Dn, fft_size), torch.float32, x1.device)
    _gate_padbuf_kernel[_bdl_grid(Bn, Dn, Ln)](x1, v, buf, Bn, Dn, Ln, x1.stride(0), x1.stride(1), x1.stride(2), buf.stride(0), buf.stride(1), BLOCK_D=32, BLOCK_L=128)
    X_s = torch.fft.fft(buf)
    X = X_s[..., : H.shape[-1]]
    y = torch.fft.irfft(X * H, n=fft_size, norm="forward")           # (B, D, 2L) fp32; the stock slices [..., :L] (a view)
    out = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=x1.device)
    _hcl_epilogue_kernel[_bdl_grid(Bn, Dn, Ln)](y, x1, v, x2, D, out, Bn, Dn, Ln, y.stride(0), y.stride(1), x1.stride(0), x1.stride(1), x1.stride(2),
                                                out.stride(0), out.stride(1), BLOCK_D=32, BLOCK_L=128)
    CTR["hcl_fused_v2"] += 1
    return out.permute(0, 2, 1)


# ----------------------------------------------------------------------------------------------------------------- fold (L2)
_fold_state = {}
def _interleave_perm(n3):
    idx = torch.arange(n3)
    return torch.cat([idx[0::3], idx[1::3], idx[2::3]])          # the permutation vortex.model.utils.interleave applies to dim 1 of z_pre

def fold_interleave(model):
    """Permute the projection weight rows and the 3-tap featurizer taps so the interleave op becomes a no-op. Bitwise (a row permutation)."""
    cfg = model.config
    assert cfg.interleave is True, "fold requires config.interleave True (already folded?)"
    for b in [blk for blk in model.blocks if isinstance(blk, ParallelGatedConvBlock)]:
        W = b.projections.weight
        assert W.dim() == 2 and W.dtype == torch.bfloat16 and W.is_contiguous(), (type(b.projections).__name__, W.shape, W.dtype, W.is_contiguous())
        perm = _interleave_perm(W.shape[0]).to(W.device)
        _fold_state[id(b)] = (W.data.clone(), b.filter.short_filter_weight.data.clone(),
                              None if b.filter.short_filter_bias is None else b.filter.short_filter_bias.data.clone(),
                              None if b.projections.bias is None else b.projections.bias.data.clone())
        W.data = W.data[perm].contiguous()
        if b.projections.bias is not None: b.projections.bias.data = b.projections.bias.data[perm].contiguous()
        b.filter.short_filter_weight.data = b.filter.short_filter_weight.data[perm].contiguous()
        if b.filter.short_filter_bias is not None: b.filter.short_filter_bias.data = b.filter.short_filter_bias.data[perm].contiguous()
        assert W.is_contiguous() and b.filter.short_filter_weight.is_contiguous()
    cfg.interleave = False

def unfold_interleave(model):
    for b in [blk for blk in model.blocks if isinstance(blk, ParallelGatedConvBlock)]:
        if id(b) not in _fold_state: continue
        W0, sw0, sb0, pb0 = _fold_state.pop(id(b))
        b.projections.weight.data = W0; b.filter.short_filter_weight.data = sw0
        if sb0 is not None: b.filter.short_filter_bias.data = sb0
        if pb0 is not None: b.projections.bias.data = pb0
    model.config.interleave = True

# ----------------------------------------------------------------------------------------------------------------- block forwards (E50 / E56 / E65), direct cuFFT (E59 / E60)
# ---- E50: out_proj GEMM on the TRANSPOSED operand. The filter returns (B, D, L) permuted to a (B, L, D) view; F.linear's 2-D reshape copies it
#      at B >= 2; at::linear's own torch.matmul on the view folds a B == 1 operand to a transposed 2-D view (one GEMM, no copy) and keeps the
#      stock's copy + flat GEMM at B >= 2 — the stock's GEMM descriptor at every (B, L); the bias add is E56's.
_orig_hyena_forward = ParallelGatedConvBlock.forward
def _e50_hyena_forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
    assert padding_mask is None and not self.print_activations, "E50 is built for the unpadded, non-logging forward only"
    u, z = _pair_proj_norm(self, u)
    z, inference_params = self.filter(z, inference_params=inference_params, padding_mask=padding_mask)
    W = self.out_filter_dense.weight
    if z.is_contiguous():
        zo = self.out_filter_dense(z); CTR["e50_contiguous_fallback"] += 1
    else:
        zo = _LT.matmul_wt(z, W); CTR["e50_bmm_transposed"] += 1                   # at::linear's own call on the (B, L, D) view: one transposed GEMM at B == 1 (T9-served)
        if self.out_filter_dense.bias is not None: zo = zo + self.out_filter_dense.bias
    z_in = zo + u
    y = self.res_mlp_norm(z_in)
    return y, inference_params

# ---- E56: fused bias-add + residual-add after the E50 bmm. The stock's at::linear on a 3-D input = matmul, then a SEPARATE bias add (bf16 out), then + u:
#      out = bf16( bf16(zo + b) + u ) with both roundings replicated in one Triton kernel (two ATen adds -> one kernel): exact by construction (the same two roundings, in the same order).
@triton.jit
def _bias_resid_kernel(z_ptr, b_ptr, u_ptr, o_ptr, n, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid * BLOCK + tl.arange(0, BLOCK); m = offs < n
    z = tl.load(z_ptr + offs, mask=m, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + (offs % D), mask=m, other=0.0).to(tl.float32)
    t = (z + b).to(tl.bfloat16).to(tl.float32)                                  # bf16(zo + bias)  (ATen add, bf16 out)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs, (t + u).to(tl.bfloat16), mask=m)                      # bf16(t + u)
def _bias_resid(zo, b, u):
    _i32_rows("E56_fused_bias_residual", "the block output", zo); zo = zo.contiguous(); u = u.contiguous(); out = torch.empty_like(zo); n = zo.numel()
    _bias_resid_kernel[(triton.cdiv(n, 4096),)](zo, b, u, out, n, zo.shape[-1], BLOCK=4096, num_warps=8)
    return out
from evo2_opt.kit import ltsel as _LT                                              # noqa: E402 — T9 serves E50's transposed GEMM; bound here, below the kernels

def _e56_hyena_forward(self, u, inference_params=None, padding_mask=None, *args, pair_out=False, **kwargs):
    assert padding_mask is None and not self.print_activations, "E50/E56 are built for the unpadded, non-logging forward only"
    u, z = _pair_proj_norm(self, u)                                                  # u may arrive as the previous block's pair (E66)
    z, inference_params = self.filter(z, inference_params=inference_params, padding_mask=padding_mask)
    W = self.out_filter_dense.weight; b = self.out_filter_dense.bias
    if z.is_contiguous() or b is None:
        zo = self.out_filter_dense(z); CTR["e56_fallback"] += 1; z_in = zo + u
    else:
        zo = _LT.matmul_wt(z, W); CTR["e50_bmm_transposed"] += 1                   # at::linear's own call on the (B, L, D) view (linear.cpp: matmul, then the
                                                                                    # bias add_): one transposed GEMM at B == 1, the stock's copy + flat GEMM at B >= 2
        uc = u.contiguous()
        if _post_norm_plain(self) and _RN.served_add(zo, uc, self.post_norm.scale, b):   # bias + residual + post_norm in ONE launch (kit/rmsnorm.py add_rmsnorm);
            z_in, y = _RN.add_rmsnorm(zo, b, uc, self.post_norm.scale, self.post_norm.eps); CTR["e56_bias_resid_norm_fused"] += 1
            return _pair_or_sum(self.mlp(y), z_in, pair_out), inference_params      # res_mlp_norm: mlp(post_norm(z_in)) + z_in (the add here, or in the next pre_norm)
        z_in = _bias_resid(zo, b, u); CTR["e56_bias_resid_fused"] += 1
    y = self.res_mlp_norm(z_in)
    return y, inference_params


def _post_norm_plain(block) -> bool:
    """The block's post_norm is vortex's RMSNorm on its tensor-op path (the arithmetic kit/rmsnorm.py reproduces)."""
    n = getattr(block, "post_norm", None)
    return isinstance(n, RMSNorm) and not n.use_flash_rmsnorm


# ---- E65: the attention block's residual add + post_norm in one launch (kit/rmsnorm.py add_rmsnorm): s = bf16(mha(pre_norm(u)) + u), y = post_norm(s),
#      then mlp(y) + s as the stock's forward; a padding-mask tensor, activation logging or a generation call take the stock's forward whole.
_orig_attn_forward = AttentionBlock.forward
_orig_stateless_forward = StripedHyena.stateless_forward
def _attn_forward(self, u, inference_params=None, padding_mask=None, *args, pair_out=False, **kwargs):
    if type(padding_mask) == torch.Tensor or self.print_activations or inference_params is not None or not _post_norm_plain(self):
        CTR["e65_stock_forward"] += 1
        return _orig_attn_forward(self, _added(u), inference_params, padding_mask, *args, **kwargs)
    u, x = _pair_pre_norm(self, u)                                                    # u may arrive as the previous block's pair (E66)
    _i32_rows("E65_attention_residual_norm", "the attention block input", u)
    h = self.inner_mha_cls(x, inference_params=inference_params)
    uc, hc = u.contiguous(), h.contiguous()
    if not _RN.served_add(hc, uc, self.post_norm.scale):
        CTR["e65_two_launch"] += 1
        s = h + u
        return _pair_or_sum(self.mlp(self.post_norm(s)), s, pair_out), None
    s, y = _RN.add_rmsnorm(hc, None, uc, self.post_norm.scale, self.post_norm.eps); CTR["e65_attn_resid_norm_fused"] += 1
    return _pair_or_sum(self.mlp(y), s, pair_out), None

# ---- E59/E60: direct cuFFT with torch's OWN plan parameters (torch v2.7.1 SpectralOps.cpp / CuFFTPlanCache.h, read from bytes).
#      E60: the stock HCL `torch.fft.fft(real)` = _fft_r2c(onesided=False): R2C planned with onembed [n], odist n (the full complex row), followed by a
#           conjugate-symmetry fill of the half the stock then discards. The same plan into a persistent full-row buffer, no fill.
#      E59: `irfft(Y, n, norm='forward')` = _fft_c2r: a defensive clone of Y, then C2R (inembed [n], idist n/2+1; onembed [n], odist n).
#           The same plan on Y itself (a fresh product never reused): the same cuFFT plan and arguments as torch's own call. The plan
#           parameters are torch-version-specific: apply() pins torch 2.7.1 (re-derive from SpectralOps.cpp for any other torch).
#      SOURCE for the plan parameters: pytorch tag v2.7.1 = commit e2d141dbde55c2a4370fac5165b0561b6af4798b (GitHub tag ref),
#           aten/src/ATen/native/cuda/SpectralOps.cpp  sha256 a647acc48074b2371595c54577f1bbf341e8fd0bffac3884b3f273762b4a4403
#           aten/src/ATen/native/cuda/CuFFTPlanCache.h sha256 6e3f37122ee32a7fbc47a16fca0421653f669c3a0ec78581e8106a2ad7c152fa
#           (_exec_fft: batch collapse + out.resize_ contiguous -> CuFFTParams(in_strides, out_strides, signal_size); CuFFTConfig: as_cufft_embed ->
#            simple_layout false for both calls -> cufftXtMakePlanMany(rank 1, n, inembed, istride, idist, itype, onembed, ostride, odist, otype, batch, exec C_32F);
#            _fft_r2c_cufft onesided=False -> out_sizes = full sizes, then _fft_fill_with_conjugate_symmetry_; _fft_c2r_cufft -> self.clone(Contiguous) then C2R).
import ctypes
class _CuFFT:
    CUDA_R_32F, CUDA_C_32F, FORWARD, INVERSE = 0, 4, -1, 1
    def __init__(self):
        self.lib = None; self.plans = {}
        for name in ("libcufft.so.11", "libcufft.so"):
            try: self.lib = ctypes.CDLL(name); break
            except OSError: continue
        assert self.lib is not None, "libcufft not loadable"
        L = self.lib; ll = ctypes.c_longlong
        L.cufftCreate.argtypes = [ctypes.POINTER(ctypes.c_int)]; L.cufftSetAutoAllocation.argtypes = [ctypes.c_int, ctypes.c_int]
        L.cufftXtMakePlanMany.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ll), ctypes.POINTER(ll), ll, ll, ctypes.c_int,
                                          ctypes.POINTER(ll), ll, ll, ctypes.c_int, ll, ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        L.cufftSetWorkArea.argtypes = [ctypes.c_int, ctypes.c_void_p]; L.cufftSetStream.argtypes = [ctypes.c_int, ctypes.c_void_p]
        L.cufftXtExec.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    def plan(self, kind, batch, n):
        key = (kind, int(batch), int(n))
        if key in self.plans: return self.plans[key]
        L = self.lib; ll = ctypes.c_longlong; h = ctypes.c_int(); assert L.cufftCreate(ctypes.byref(h)) == 0
        assert L.cufftSetAutoAllocation(h.value, 0) == 0
        nn = (ll * 1)(n); emb = (ll * 1)(n); ws = ctypes.c_size_t()
        if kind == "r2c_full":        # torch _fft_r2c onesided=False: out strides (n,1) complex -> onembed [n], ostride 1, odist n
            rc = L.cufftXtMakePlanMany(h.value, 1, nn, emb, 1, n, self.CUDA_R_32F, emb, 1, n, self.CUDA_C_32F, batch, ctypes.byref(ws), self.CUDA_C_32F)
        elif kind == "c2r":           # torch _fft_c2r: in strides (n/2+1,1) complex -> inembed [n], istride 1, idist n/2+1; out (n,1) real -> onembed [n], odist n
            rc = L.cufftXtMakePlanMany(h.value, 1, nn, emb, 1, n // 2 + 1, self.CUDA_C_32F, emb, 1, n, self.CUDA_R_32F, batch, ctypes.byref(ws), self.CUDA_C_32F)
        else: raise ValueError(kind)
        assert rc == 0, f"cufftXtMakePlanMany({kind}) rc={rc}"
        work = torch.empty(int(ws.value), dtype=torch.uint8, device="cuda") if ws.value else None
        if work is not None: assert L.cufftSetWorkArea(h.value, work.data_ptr()) == 0
        self.plans[key] = (h.value, work, int(ws.value)); return self.plans[key]
    def r2c_full(self, buf, out_full):
        # buf: (B, D, n) fp32 contiguous; out_full: (B, D, n) complex64 contiguous (persistent); writes the first n/2+1 of every row
        Bn, Dn, n = buf.shape; h, work, _ = self.plan("r2c_full", Bn * Dn, n)
        assert self.lib.cufftSetStream(h, torch.cuda.current_stream().cuda_stream) == 0
        assert self.lib.cufftXtExec(h, buf.data_ptr(), out_full.data_ptr(), self.FORWARD) == 0; CTR["e60_r2c_nofill"] += 1
        return out_full[..., : n // 2 + 1]
    def c2r(self, Y, n, out):
        # Y: (B, D, n/2+1) complex64 contiguous (consumed/overwritten); out: (B, D, n) fp32 contiguous
        Bn, Dn, h2 = Y.shape; assert h2 == n // 2 + 1 and Y.is_contiguous(); h, work, _ = self.plan("c2r", Bn * Dn, n)
        assert self.lib.cufftSetStream(h, torch.cuda.current_stream().cuda_stream) == 0
        assert self.lib.cufftXtExec(h, Y.data_ptr(), out.data_ptr(), self.INVERSE) == 0; CTR["e59_c2r_noclone"] += 1
        return out
_CF = None
def _cf():
    global _CF
    if _CF is None: _CF = _CuFFT()
    return _CF
_spec_bufs = {}
def _spec_buf(shape, dtype, device):
    key = (tuple(shape), str(dtype), str(device))
    if key not in _spec_bufs: _spec_bufs[key] = torch.empty(*shape, dtype=dtype, device=device)
    return _spec_bufs[key]

def _parallel_iir_v5(self, z_pre, h, D, L, poles, residues, t, dims, layer_idx, inference_params=None, prefill_style="fft",
                           fftconv_fn=None, padding_mask=None, use_flashfft=False, column_split_hyena=False, long_fir_threshold=None):
    assert inference_params is None and not use_flashfft and long_fir_threshold is None and not column_split_hyena and padding_mask is None
    fft_size = 2 * L; hidden_size = dims[0]
    x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)
    if self.hyena_flip_x1x2: x1, x2 = x2, x1
    key = (layer_idx, int(L), str(h.device))
    if key not in _H_cache:
        _H_cache[key] = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
    H = _H_cache[key]
    Bn, Dn, Ln = x1.shape
    buf = _pad_buf((Bn, Dn, fft_size), torch.float32, x1.device)
    _gate_padbuf_kernel[_bdl_grid(Bn, Dn, Ln)](x1, v, buf, Bn, Dn, Ln, x1.stride(0), x1.stride(1), x1.stride(2), buf.stride(0), buf.stride(1), BLOCK_D=32, BLOCK_L=128)
    X = _cf().r2c_full(buf, _spec_buf((Bn, Dn, fft_size), torch.complex64, x1.device))      # E60: the stock's X_s[..., :n/2+1], no conjugate fill
    Y = X * H                                                                                   # the stock's ATen complex multiply (contiguous output)
    y = _cf().c2r(Y, fft_size, _spec_buf((Bn, Dn, fft_size), torch.float32, x1.device))       # E59: the stock's irfft without the clone
    out = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=x1.device)
    _hcl_epilogue_kernel[_bdl_grid(Bn, Dn, Ln)](y, x1, v, x2, D, out, Bn, Dn, Ln, y.stride(0), y.stride(1), x1.stride(0), x1.stride(1), x1.stride(2),
                                                out.stride(0), out.stride(1), BLOCK_D=32, BLOCK_L=128)
    CTR["hcl_fused_v2"] += 1
    return out.permute(0, 2, 1)

def _parallel_fir_v5(self, fir_fn, u, weight, bias, L, dims, groups=None, gated_bias=False, column_split_hyena=False, dim_last=True,
                           fir_length=3, gate=False, inference_params=None, prefill_mode=None, padding_mask=None):
    if fir_length >= 128 and gate and not dim_last and inference_params is None and padding_mask is None and fir_fn == torch.nn.functional.conv1d and bias is not None:
        hidden_size = dims[0]
        x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
        if self.hyena_flip_x1x2: x1, x2 = x2, x1
        Bn, Dn, Ln = x1.shape; fft_size = 2 * Ln
        key = (self.layer_idx, Ln, str(u.device))
        if key not in _kf_cache:
            k = weight[:, :, :Ln].to(torch.float32)
            k_f = torch.fft.rfft(k, n=fft_size) / fft_size
            _kf_cache[key] = _adj(torch.empty((Bn, Dn, Ln), device=u.device), k_f)
        k_f = _kf_cache[key]
        buf = _pad_buf((Bn, Dn, fft_size), torch.float32, u.device)
        _gate_padbuf_kernel[_bdl_grid(Bn, Dn, Ln)](x1, v, buf, Bn, Dn, Ln, x1.stride(0), x1.stride(1), x1.stride(2), buf.stride(0), buf.stride(1), BLOCK_D=32, BLOCK_L=128)
        u_f = torch.fft.rfft(buf)
        Y = u_f * k_f
        y = _cf().c2r(Y, fft_size, _spec_buf((Bn, Dn, fft_size), torch.float32, u.device))       # E59
        out = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=u.device)
        _hcm_epilogue_kernel[_bdl_grid(Bn, Dn, Ln)](y, x1, v, x2, bias, out, Bn, Dn, Ln, y.stride(0), y.stride(1), x1.stride(0), x1.stride(1), x1.stride(2),
                                                    out.stride(0), out.stride(1), BLOCK_D=32, BLOCK_L=128)
        CTR["hcm_fused_v2"] += 1
        return out, None
    return _parallel_fir_v2(self, fir_fn, u, weight, bias, L, dims, groups=groups, gated_bias=gated_bias, column_split_hyena=column_split_hyena,
                            dim_last=dim_last, fir_length=fir_length, gate=gate, inference_params=inference_params, prefill_mode=prefill_mode, padding_mask=padding_mask)

VERSIONS = {
    "v0": ["L1b", "E10", "L1c", "L2", "E8", "E11"],
    "v1": ["L1b", "E10", "L1c", "L2", "E8", "E11", "E7"],
    "v2": ["L1b", "E10", "L1c", "L2", "E8", "E11", "E7", "E9b"],
    "v2p": ["L1b", "E10", "L1c", "L2", "E8", "E11", "E7", "E9b", "E9a"],
    "v3": ["L1b", "E10", "L1c", "L2", "E8", "E11", "E7", "E9b", "E9a", "E50"],
    "v4": ["L1b", "E10", "L1c", "L2", "E8", "E11", "E7", "E9b", "E9a", "E50", "E56"],
    "v5": ["L1b", "E10", "L1c", "L2", "E8", "E11", "E7", "E9b", "E9a", "E50", "E56", "E59E60"],
}
_applied = {"version": None}

def expected_counts(version):
    """per single forward of the 7B (27 hyena blocks = 9 HCS + 9 HCM + 9 HCL; 5 attention; 65 RMSNorms)"""
    lv = set(VERSIONS[version]); c = {}
    if "E10" in lv: c["hcl_cached_iir"] = 9
    if "E9a" in lv: c["hcl_fused_v2"] = 9; c.pop("hcl_cached_iir", None)
    if "L1c" in lv: c["hcm_fftconv_cached"] = 9
    if "E9b" in lv: c["hcm_fused_v2"] = 9; c.pop("hcm_fftconv_cached", None)
    if "E7" in lv: c["fir3_triton_fwd"] = 27; c["hcs7_triton_fwd"] = 9
    elif "E8" in lv: c["fir_bf16_conv1d"] = 36
    if "E11" in lv: c["rms_fused_tail"] = 65
    if "E50" in lv: c["e50_bmm_transposed"] = 27
    if "E56" in lv: c["e56_bias_resid_fused"] = 27
    if "E59E60" in lv: c["e60_r2c_nofill"] = 9; c["e59_c2r_noclone"] = 18
    return c

def check_conditions(model):
    import vortex, evo2
    cfg = model.config
    assert cfg.get("use_fp8_input_projections", False) is False and not HAS_TE, "kit built for the bf16 (no-TE) route only"
    hb = [blk for blk in model.blocks if isinstance(blk, ParallelGatedConvBlock)]
    ab = [blk for blk in model.blocks if isinstance(blk, AttentionBlock)]
    assert hb and len(hb) + len(ab) == len(model.blocks), (len(hb), len(ab), len(model.blocks))          # every block is a Hyena or an attention block (7b: 27 + 5; 40b: 42 + 8)
    hidden = int(cfg.hidden_size)
    for b in hb:
        assert type(b.projections).__name__ == "TELinear" and isinstance(b.projections.weight, torch.Tensor) and b.projections.weight.dtype == torch.bfloat16, \
            f"projection module {type(b.projections).__name__}: the fold/E8 signatures are validated for vortex's TELinear fallback only"
        assert b.filter.short_filter_bias is None and b.filter.fir_fn is torch.nn.functional.conv1d
    # NUMERICS PIN of the stock: Wqkv.weight layout (3·hidden, hidden) bf16, strides (1, 3·hidden) — produced by custom_load_state_dict's
    # column-split permutation (model.py:1021-1031, preserve_format); F.linear on this layout is the stock numerics; a row-major copy changes the logits.
    assert all(b.inner_mha_cls.Wqkv.bias is None for b in ab), "qkv_proj_bias False (evo2-7b-1m.yml / evo2-40b-1m.yml); a bias would take the loader's silent try/except path"
    wq = [b.inner_mha_cls.Wqkv.weight for b in ab]
    assert [tuple(w.stride()) for w in wq] == [(1, 3 * hidden)] * len(ab), f"Wqkv layout changed: {[tuple(w.stride()) for w in wq]}"
    noncontig = {n for n, p in model.named_parameters() if not p.is_contiguous()}
    assert noncontig == {n for n, p in model.named_parameters() if any(p is w for w in wq)}, f"unexpected non-contiguous parameters: {sorted(noncontig)}"
    return dict(vtx=getattr(vortex, "__version__", "?"), torch=torch.__version__, triton=triton.__version__, projections=type(hb[0].projections).__name__)

_ORIGINALS = dict(compute_filter=_orig_compute_filter, parallel_iir=_orig_parallel_iir, parallel_fir=_orig_parallel_fir, rms_fwd=_orig_rms_fwd,
                  hyena_forward=_orig_hyena_forward, attn_fwd=_orig_attn_forward, fftconv_func=fftconv_func, stateless_forward=_orig_stateless_forward)
PROCESS_WIDE_PATCHES = ("HyenaCascade.compute_filter", "HyenaInferenceEngine.parallel_iir", "HyenaInferenceEngine.parallel_fir",
                        "vortex.model.engine.fftconv_func", "RMSNorm.forward", "ParallelGatedConvBlock.forward", "AttentionBlock.forward", "StripedHyena.stateless_forward",
                        "in-place weight fold (L2) of projections/short_filter")

def is_unpatched():
    """True iff every class/module attribute this kit patches is the original object (compared with `is`) and no fold is live."""
    return (HyenaCascade.compute_filter is _ORIGINALS["compute_filter"] and HyenaInferenceEngine.parallel_iir is _ORIGINALS["parallel_iir"]
            and HyenaInferenceEngine.parallel_fir is _ORIGINALS["parallel_fir"] and RMSNorm.forward is _ORIGINALS["rms_fwd"]
            and ParallelGatedConvBlock.forward is _ORIGINALS["hyena_forward"] and AttentionBlock.forward is _ORIGINALS["attn_fwd"]
            and StripedHyena.stateless_forward is _ORIGINALS["stateless_forward"] and _eng.fftconv_func is _ORIGINALS["fftconv_func"] and not _fold_state)

# ---------------------------------------------------------------------------
# 32-bit element addressing of the Triton kernels above (kit/i32.py): tl.program_id and every integer argument below 2^31 are int32,
# so a launch whose widest tensor spans more than 2^31 elements is refused BEFORE the launch, by name — one `[evo2-kit] SHAPE CEILING <lever> …`
# line and a ShapeCeiling RuntimeError; never launched to fault, never rerouted. E7's featurizer input (B, L, 3·hidden) is the tightest: the
# stock's F.conv1d featurizer stops at that element count by torch's canUse32BitIndexMath rule, so a shape refused here is one the stock refuses.
# Defined after the kernels on purpose: nothing above moves (a jit function's starting line is part of its cache key).
from evo2_opt.kit import i32 as _I32  # noqa: E402


def _i32_fir3(x_bld):
    """(B, L, D3) of the featurizer input once its extent (read through its strides) and the (B, D3, L) output's fit; else refused by name."""
    Bn, Ln, D3 = x_bld.shape
    one = _I32.strided_extent((Ln, D3), (x_bld.stride(1), x_bld.stride(2)))
    _I32.check("E7_triton_fir3_featurizer", f"the featurizer input (B={Bn}, L={Ln}, 3D={D3})",
               (_I32.strided_extent(x_bld.shape, x_bld.stride()), Bn * D3 * Ln),
               f"batch <= {min(_I32.max_leading(x_bld.stride(0), one), _I32.EXTENT // max(1, D3 * Ln))} at L={Ln}")
    return Bn, Ln, D3


def _i32_rows(lever, what, x):
    """E11 / E56: the kernel reads a contiguous (rows, width) copy of x — rows·width must fit."""
    width = int(x.shape[-1])
    _I32.check(lever, f"{what} {tuple(x.shape)}", (x.numel(),), f"batch x length <= {_I32.EXTENT // max(1, width):,} rows at width {width}")


from evo2_opt.kit import rmsnorm as _RN  # noqa: E402  (after the kernels: nothing above moves)
_RMS_NAMED = set()


def _rms_two_launch_named(xc):
    """One stderr line per input width the fused RMSNorm kernel does not serve (ATen's norm + the tail kernel run for those calls)."""
    width = int(xc.shape[-1])
    if width in _RMS_NAMED:
        return
    _RMS_NAMED.add(width)
    import sys
    print(f"[evo2-kit] E11: RMSNorm inputs like {tuple(xc.shape)} (width {width}) are outside the fused norm kernel's rule (kit/rmsnorm.py served) — "
          f"ATen's norm + the tail kernel serve them", file=sys.stderr, flush=True)


FILTER_TILE = 512                                                  # channels per tile of the torch filter build (kshapes.filter_tiles)

def _compute_filter_tiled(self, L, device):
    """vortex HyenaCascade.compute_filter (model.py:387-403) with the torch expression h = (residues[..., None] * (log_poles * t).exp()).sum(1)
    evaluated per tile of FILTER_TILE channels into one (D, L) tensor: the same elementwise terms and the same per-channel sum over the state
    dimension, the (tile, state, L) intermediates in place of (D, state, L). The vortex-kernels branch (use_hcl_kernel) is the stock's call."""
    from vortex.model import model as _vm
    if self.engine.use_hcl_kernel and getattr(_vm, "_hcl_compute_filter", None) is not None:
        return _orig_compute_filter(self, L, device)
    self.update_time(L, device)
    filter_dtype = torch.float32
    residues, log_poles = self.residues.to(filter_dtype), self.log_poles.to(filter_dtype)
    D = residues.shape[0]
    h = None
    from evo2_opt.kit import kshapes as _KS
    for d0, d1 in _KS.filter_tiles(D, FILTER_TILE):
        tile = (residues[d0:d1][..., None] * (log_poles[d0:d1] * self.t).exp()).sum(1)
        if h is None:
            h = torch.empty((D,) + tuple(tile.shape[1:]), dtype=tile.dtype, device=tile.device)
        h[d0:d1] = tile
    CTR["hcl_filter_tiled"] += 1
    return h[None], filter_dtype, log_poles, residues


# ---- E66: a block's closing residual add (mlp(y) + s: one bf16 read-read-write pass) carried as the pair (mlp(y), s) into the NEXT block, whose
#      pre_norm launch adds it, stores the sum (that block's residual input u) and normalizes it (kit/rmsnorm.py add_rmsnorm / add_rmsnorm_e4m3):
#      bf16(m + s) is torch's add, the norm reads the stored sum. A block on another device receives the added tensor; the last block's pair is
#      added before the loop returns; a padding-mask tensor or activation logging take the stock loop.
FOLDC = {"pair_proj_norm": False}                                  # True once the route installs a proj_norm that takes the pair (E63's, kit/fp8emit.py)

def _added(u):
    """The carried pair (m, s) added with torch's bf16 add — the previous block's closing add — or the tensor itself."""
    if isinstance(u, tuple):
        CTR["e66_added_apart"] += 1
        return u[0] + u[1]
    return u

def _pair_or_sum(m, s, pair_out: bool):
    """A block's closing residual: the pair for a caller that folds the add into the next pre_norm, the sum otherwise."""
    return (m, s) if pair_out else m + s

def _pair_proj_norm(self, u):
    """-> (u, proj_norm(u)) of a Hyena block; a carried pair is added inside the pre_norm launch: by this route's proj_norm when it takes pairs
    (E63's, with the e4m3 cast), else by add_rmsnorm ahead of vortex's pad + projections when the rows are served."""
    if isinstance(u, tuple):
        if FOLDC["pair_proj_norm"]:
            return self.proj_norm(u)
        n = self.pre_norm
        m, s = u[0].contiguous(), u[1].contiguous()
        if isinstance(n, RMSNorm) and not n.use_flash_rmsnorm and not self.print_activations and _RN.served_add(m, s, n.scale):
            CTR["e66_residual_pre_norm_fused"] += 1
            u, y = _RN.add_rmsnorm(m, None, s, n.scale, n.eps)
            y = self.pad_to_multiple(y)
            with torch.cuda.device(y.device):
                projected = self.projections(y)
            return u, (projected[0] if isinstance(projected, tuple) else projected)
    u = _added(u)
    return u, self.proj_norm(u)

def _pair_pre_norm(self, u):
    """-> (u, pre_norm(u)) of an attention block; a carried pair is added inside the norm launch (add_rmsnorm) when the rows are served."""
    if isinstance(u, tuple):
        n = self.pre_norm
        m, s = u[0].contiguous(), u[1].contiguous()
        if isinstance(n, RMSNorm) and not n.use_flash_rmsnorm and _RN.served_add(m, s, n.scale):
            CTR["e66_residual_pre_norm_fused"] += 1
            return _RN.add_rmsnorm(m, None, s, n.scale, n.eps)
    u = _added(u)
    return u, self.pre_norm(u)

def _stateless_forward_pairs(self, x, padding_mask=None):
    """vortex StripedHyena.stateless_forward (model.py:800) with every block asked for its closing pair instead of the sum (pair_out=True)."""
    if type(padding_mask) == torch.Tensor or self.print_activations:
        CTR["e66_stock_loop"] += 1
        return _orig_stateless_forward(self, x, padding_mask=padding_mask)
    for block_idx, block in enumerate(self.blocks):
        if isinstance(x, tuple) and self.block_idx_to_device[max(block_idx - 1, 0)] != self.block_idx_to_device[block_idx]:
            x = x[0] + x[1]; CTR["e66_added_at_device_change"] += 1                  # the device boundary carries one tensor, as the stock's
        x = self.cross_device_transfer(x, block_idx)
        x, _ = block(x, inference_params=None, padding_mask=padding_mask, pair_out=True)
    if isinstance(x, tuple):
        x = x[0] + x[1]; CTR["e66_last_pair_added"] += 1                             # the last block's closing add, before the final norm
    return x, None
