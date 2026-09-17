"""The vortex-kernels route's two FFT long convolutions (vortex.ops.hcl_interface.hcl_fft_conv, hcm_interface.hcm_fft_conv) restructured around what they recompute and copy on every call. Filter spectra are cached per (layer, L, device) — the HCM spectrum kept once per filter group (its D rows are the groups' rows repeated) — for ONE length at a time (kit/lkeyed.py) and within a budget of 1/7 of the device's total memory (kit/kshapes.py SPECTRA_BUDGET): a layer whose spectrum does not fit builds and transforms its filter on every call as the stock does, named once per length; the HCL time-domain filter is never stored (the stock's compute_filter builds it on a layer's first call at a length, the spectrum is filled from it). The blocks' featurizer FIR is deferred into the chains (routes: the featurizer call returns its raw (B, 3D, L) input view with the taps attached; the pad and epilogue launches apply the 3-tap FIR to the x1 / v / x2 rows inline with base._fir3_kernel's element, so z_pre is never materialised for these blocks). The gated activation is cast and zero-padded into a persistent buffer by one Triton launch and transformed by a persistent one-sided cuFFT plan; the spectral product is vortex's complex multiply (its fp32 expressions, the filter row read per group) into a persistent product buffer; the inverse is a persistent C2R plan into the storage the spectrum occupied; the epilogues read the inverse transform at its own strides with the vortex kernels' fp32 expressions and roundings. cuFFT is called with torch 2.7.1's own plan parameters for these calls (compact.r2c_half_plan_args / c2r_plan_args), so every transform is the transform the stock's torch.fft call executes."""
from __future__ import annotations

import ctypes
import sys

import torch
import triton
import triton.language as tl

from evo2_opt.kit import base as K7
from evo2_opt.kit.base import _rne_bf16, _gate_padbuf_kernel
from evo2_opt.kit import compact as C
from evo2_opt.kit import kshapes as KS
from evo2_opt.kit import lkeyed as LK
from evo2_opt.kit import multidev as M40
from evo2_opt.kit import i32 as _I32

CTR = K7.CTR
PREFIX = "[evo2-kit]"
GEOM = (8, 256, 4)                         # (BLOCK_D, BLOCK_L, num_warps) of the pad / epilogue launches: geometry only
MUL_BLOCK, MUL_WARPS = 512, 8              # the complex multiply's 1-D tile: geometry only
_H = LK.OneLength("k_hcl_spectra")         # (layer_idx, L, device) -> (rfft(h, 2L) (1, D, L+1) complex64, rows per group = 1)
_KF = LK.OneLength("k_hcm_spectra")        # (layer_idx, L, device) -> (rfft(k, 2L) one row per filter group (G, L+1) complex64, rows per group)
_BUF: dict = {}                            # (kind, shape, device) -> persistent pad / spectrum-then-output / product buffers (shape-bound)
_PC = LK.OneLength("k_per_call")           # (layer_idx, L, device) -> kind: the layers whose spectrum is past the budget at the resident length
_PH: dict = {}                             # device -> the (1, 0, 0) placeholder filter
_NAMED: set = set()                        # (kind, L, device) whose per-call line was printed
TAPS = "_evo2_kit_fir3_taps"               # the attribute a deferred featurizer output carries: its (3D, 1, 3) taps
DEFERRED: frozenset = frozenset()          # layer_idx of the blocks whose featurizer FIR the chains apply inline (the HCL and HCM blocks; set at apply)


# ----------------------------------------------------------------------------------------------------------------- buffers
def _buf(kind, shape, dtype, device, zero=False):
    key = (kind, tuple(shape), str(device))
    b = _BUF.get(key)
    if b is None:
        b = _BUF[key] = (torch.zeros if zero else torch.empty)(*shape, dtype=dtype, device=device)
        CTR[f"k_alloc_{kind}"] += 1
    return b


def clear_buffers():
    _BUF.clear()


def clear_spectra():
    LK.clear_all()


def _nbytes(store, dev=None) -> int:
    return sum(t.numel() * t.element_size() for (t, _r) in store.values() if dev is None or str(t.device) == dev)


def resident_bytes() -> dict:
    """Bytes held, by store (what a memory report reads)."""
    return {"hcl_spectra": _nbytes(_H), "hcm_spectra": _nbytes(_KF), "shape_buffers": sum(t.numel() * t.element_size() for t in _BUF.values()),
            "cufft_work": 0 if _CF is None else _CF.work_bytes_total()}


def _fits(dev: torch.device, new_bytes: int) -> bool:
    return KS.fits_budget(_nbytes(_H, str(dev)) + _nbytes(_KF, str(dev)), new_bytes, torch.cuda.get_device_properties(dev).total_memory)


def _name_per_call(kind: str, L: int, dev: torch.device, layer_idx: int):
    key = (kind, int(L), str(dev))
    if key in _NAMED:
        return
    _NAMED.add(key)
    num, den = KS.SPECTRA_BUDGET
    budget = KS.spectra_budget_bytes(torch.cuda.get_device_properties(dev).total_memory) / 2 ** 30
    print(f"{PREFIX} {kind}_spectra=per-call above {num}/{den} of device memory ({budget:.1f} GiB) at L={int(L):,} on {dev}: {kind.upper()} layer {layer_idx} and "
          f"the later ones build and transform their filter on every call as the stock does; the earlier layers' spectra stay cached", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------------------------- cuFFT: one-sided R2C + C2R, per device
class _Plans:
    """libcufft through base._CuFFT's bindings; plans keyed (kind, batch, n) with torch's parameters for rfft (one-sided) and irfft; ONE work
    area per device shared by its plans (they execute in sequence on the current stream)."""

    def __init__(self):
        self.base = K7._CuFFT()
        self.lib = self.base.lib
        self.lib.cufftDestroy.argtypes = [ctypes.c_int]
        self.plans = {}                    # (kind, batch, n) -> handle
        self.work = None; self.work_bytes = 0

    def _plan(self, kind, batch, n):
        key = (kind, int(batch), int(n))
        h = self.plans.get(key)
        if h is not None:
            return h
        L, ll = self.lib, ctypes.c_longlong
        a = C.r2c_half_plan_args(n, batch) if kind == "r2c_half" else C.c2r_plan_args(n, batch)
        T = {"CUDA_R_32F": K7._CuFFT.CUDA_R_32F, "CUDA_C_32F": K7._CuFFT.CUDA_C_32F}
        hd = ctypes.c_int(); assert L.cufftCreate(ctypes.byref(hd)) == 0
        assert L.cufftSetAutoAllocation(hd.value, 0) == 0
        ws = ctypes.c_size_t()
        nn = (ll * 1)(*a["n"]); ei = (ll * 1)(*a["inembed"]); eo = (ll * 1)(*a["onembed"])
        rc = L.cufftXtMakePlanMany(hd.value, a["rank"], nn, ei, a["istride"], a["idist"], T[a["itype"]], eo, a["ostride"], a["odist"], T[a["otype"]], a["batch"], ctypes.byref(ws), T[a["exec"]])
        assert rc == 0, f"cufftXtMakePlanMany({kind}, batch={batch}, n={n}) rc={rc}"
        need = int(ws.value)
        if need > self.work_bytes:
            self.work = torch.empty(need, dtype=torch.uint8, device="cuda"); self.work_bytes = need
            for hh in self.plans.values():
                assert L.cufftSetWorkArea(int(hh), self.work.data_ptr()) == 0
        if need:
            assert L.cufftSetWorkArea(hd.value, self.work.data_ptr()) == 0
        self.plans[key] = hd.value; CTR["k_plan_created"] += 1
        return hd.value

    def r2c_half(self, buf, out):
        Bn, Dn, n = buf.shape; assert out.shape == (Bn, Dn, n // 2 + 1) and buf.is_contiguous() and out.is_contiguous()
        h = self._plan("r2c_half", Bn * Dn, n)
        assert self.lib.cufftSetStream(h, torch.cuda.current_stream().cuda_stream) == 0
        assert self.lib.cufftXtExec(h, buf.data_ptr(), out.data_ptr(), K7._CuFFT.FORWARD) == 0
        return out

    def c2r(self, Y, out):
        Bn, Dn, n = out.shape; assert Y.shape == (Bn, Dn, n // 2 + 1) and Y.is_contiguous() and out.is_contiguous()
        h = self._plan("c2r", Bn * Dn, n)
        assert self.lib.cufftSetStream(h, torch.cuda.current_stream().cuda_stream) == 0
        assert self.lib.cufftXtExec(h, Y.data_ptr(), out.data_ptr(), K7._CuFFT.INVERSE) == 0
        return out

    def destroy(self):
        for h in list(self.plans.values()):
            self.lib.cufftDestroy(int(h))
        self.plans.clear(); self.work = None; self.work_bytes = 0


class _PlansPerDevice:
    def __init__(self):
        self.per_device = {}

    def on(self, device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
        p = self.per_device.get(idx)
        if p is None:
            with torch.cuda.device(idx):
                p = self.per_device[idx] = _Plans()
        return p

    def destroy(self):
        for p in self.per_device.values():
            p.destroy()
        self.per_device.clear()

    def work_bytes_total(self):
        return sum(p.work_bytes for p in self.per_device.values())


_CF: _PlansPerDevice | None = None


def _cf():
    global _CF
    if _CF is None:
        _CF = _PlansPerDevice()
    return _CF


def destroy_plans():
    if _CF is not None:
        _CF.destroy()


def stores() -> dict:
    """Shape-bound state (released by the shape manager when (batch, length) changes)."""
    return {"k_fft_buffers": clear_buffers, "k_cufft_plans": destroy_plans}


def model_stores() -> dict:
    """Model-bound state (released before a stock-path call): the filter spectra (one length resident at a time on their own)."""
    return {"k_filter_spectra": clear_spectra}


# ----------------------------------------------------------------------------------------------------------------- kernels
@triton.jit
def _fir3_acc(row_ptr, offs_l, sl, L, w0, w1, w2, mrow):
    # base._fir3_kernel's element (E7, = ATen's depthwise conv): o[t] = sum_k w[k] * x[t-2+k], invalid taps exactly +0.0, ascending tap order from +0.0;
    # row_ptr: (BLOCK_D, 1) pointers to x[d, 0] of rows strided sl along L; returns the fp32 accumulator tile (BLOCK_D, BLOCK_L)
    t0 = offs_l[None, :] - 2; t1 = offs_l[None, :] - 1; t2 = offs_l[None, :]
    ml = t2 < L
    m0 = (t0 >= 0) & ml & mrow; m1 = (t1 >= 0) & ml & mrow; m2 = ml & mrow
    x0 = tl.load(row_ptr + t0 * sl, mask=m0, other=0.0).to(tl.float32)
    x1 = tl.load(row_ptr + t1 * sl, mask=m1, other=0.0).to(tl.float32)
    x2 = tl.load(row_ptr + t2 * sl, mask=m2, other=0.0).to(tl.float32)
    p0 = tl.where(m0, w0[:, None] * x0, 0.0); p1 = tl.where(m1, w1[:, None] * x1, 0.0); p2 = tl.where(m2, w2[:, None] * x2, 0.0)
    return ((0.0 + p0) + p1) + p2


@triton.jit
def _taps3(w_ptr, od, md):
    # the three taps of channels od from a (D, 3) table
    return (tl.load(w_ptr + od * 3 + 0, mask=md, other=0.0).to(tl.float32), tl.load(w_ptr + od * 3 + 1, mask=md, other=0.0).to(tl.float32),
            tl.load(w_ptr + od * 3 + 2, mask=md, other=0.0).to(tl.float32))


@triton.jit
def _fir_pad_kernel(x1_ptr, v_ptr, w1_ptr, wv_ptr, buf_ptr, D, L, sb, sd, sl, bb, bd, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    # buf[b, d, t] = fp32(bf16(x1[t] * v[t])) for t < L with x1, v = bf16(fir3) of the raw projection rows (the featurizer's outputs, E7's element,
    # then base._gate_padbuf_kernel's product and cast); the tail untouched (zero)
    pb = tl.program_id(0); pd = tl.program_id(1); pl = tl.program_id(2)
    od = pd * BLOCK_D + tl.arange(0, BLOCK_D); ol = pl * BLOCK_L + tl.arange(0, BLOCK_L)
    md = od < D; m = md[:, None] & (ol < L)[None, :]
    a0, a1, a2 = _taps3(w1_ptr, od, md)
    b0, b1, b2 = _taps3(wv_ptr, od, md)
    fx1 = _rne_bf16(_fir3_acc(x1_ptr + pb * sb + od[:, None] * sd, ol, sl, L, a0, a1, a2, md[:, None]))
    fv = _rne_bf16(_fir3_acc(v_ptr + pb * sb + od[:, None] * sd, ol, sl, L, b0, b1, b2, md[:, None]))
    x1v = (fx1 * fv).to(tl.bfloat16).to(tl.float32)
    tl.store(buf_ptr + pb * bb + od[:, None] * bd + ol[None, :], x1v, mask=m)


@triton.jit
def _mul_kernel(u_ptr, k_ptr, y_ptr, DF, F, R, inv_fft_size, stride_batch, BLOCK: tl.constexpr):
    # y[b] = u_f[b] * k_f * inv_fft_size over the flattened (D, F) plane, interleaved (re, im): hcm_interface._hcm_complex_mul_kernel's loads and fp32
    # expressions; the filter row of channel d is row d // R of k (R consecutive channels share a filter group's spectrum; R = 1: one row per channel)
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < DF
    koffs = (offs // F // R) * F + offs % F
    u_base = u_ptr + pid_b * stride_batch
    y_base = y_ptr + pid_b * stride_batch
    u_re = tl.load(u_base + 2 * offs, mask=mask, other=0.0)
    u_im = tl.load(u_base + 2 * offs + 1, mask=mask, other=0.0)
    k_re = tl.load(k_ptr + 2 * koffs, mask=mask, other=0.0)
    k_im = tl.load(k_ptr + 2 * koffs + 1, mask=mask, other=0.0)
    y_re = (u_re * k_re - u_im * k_im) * inv_fft_size
    y_im = (u_re * k_im + u_im * k_re) * inv_fft_size
    tl.store(y_base + 2 * offs, y_re, mask=mask)
    tl.store(y_base + 2 * offs + 1, y_im, mask=mask)


@triton.jit
def _gate_x2(x2_ptr, w2_ptr, pb, od, ol, md, m, sb, sd, sl, L, FIR: tl.constexpr):
    # the gate stream x2 as fp32: loaded (a materialised featurizer output) or, FIR set, bf16(fir3) of its raw projection rows
    if FIR:
        c0, c1, c2 = _taps3(w2_ptr, od, md)
        x2 = _rne_bf16(_fir3_acc(x2_ptr + pb * sb + od[:, None] * sd, ol, sl, L, c0, c1, c2, md[:, None]))
    else:
        x2 = tl.load(x2_ptr + pb * sb + od[:, None] * sd + ol[None, :] * sl, mask=m, other=0.0).to(tl.float32)
    return x2


@triton.jit
def _hcl_gate_kernel(y_ptr, u_ptr, x2_ptr, w2_ptr, d_ptr, o_ptr, D, L, yb, yd, sb, sd, sl, ob, od_, FIR: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    # out = bf16((y + x1v * D) * x2): hcl_interface._hcl_bias_residual_gate_kernel's fp32 expression and its single rounding at the store;
    # y is read at the inverse transform's strides and x1v (fp32 of the bf16 product) from the padded transform input, both (B, D, 2L)
    pb = tl.program_id(0); pd = tl.program_id(1); pl = tl.program_id(2)
    od = pd * BLOCK_D + tl.arange(0, BLOCK_D); ol = pl * BLOCK_L + tl.arange(0, BLOCK_L)
    md = od < D; m = md[:, None] & (ol < L)[None, :]
    yo = pb * yb + od[:, None] * yd + ol[None, :]
    y = tl.load(y_ptr + yo, mask=m, other=0.0).to(tl.float32)
    x1v = tl.load(u_ptr + yo, mask=m, other=0.0).to(tl.float32)
    x2 = _gate_x2(x2_ptr, w2_ptr, pb, od, ol, md, m, sb, sd, sl, L, FIR)
    bias = tl.load(d_ptr + od, mask=md, other=0.0).to(tl.float32)
    acc = (y + x1v * bias[:, None]) * x2
    tl.store(o_ptr + pb * ob + od[:, None] * od_ + ol[None, :], acc.to(tl.bfloat16), mask=m)


@triton.jit
def _hcm_gate_kernel(y_ptr, u_ptr, x2_ptr, w2_ptr, d_ptr, o_ptr, D, L, yb, yd, sb, sd, sl, ob, od_, FIR: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    # z = fp32(y + u * bias) as hcm_interface._hcm_bias_residual_kernel stores it; parallel_fir then rounds z to bf16 and gates: out = bf16(x2 * z)
    pb = tl.program_id(0); pd = tl.program_id(1); pl = tl.program_id(2)
    od = pd * BLOCK_D + tl.arange(0, BLOCK_D); ol = pl * BLOCK_L + tl.arange(0, BLOCK_L)
    md = od < D; m = md[:, None] & (ol < L)[None, :]
    yo = pb * yb + od[:, None] * yd + ol[None, :]
    y = tl.load(y_ptr + yo, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + yo, mask=m, other=0.0).to(tl.float32)
    bias = tl.load(d_ptr + od, mask=md, other=0.0).to(tl.float32)
    acc = y + u * bias[:, None]
    z = _rne_bf16(acc)
    x2 = _gate_x2(x2_ptr, w2_ptr, pb, od, ol, md, m, sb, sd, sl, L, FIR)
    tl.store(o_ptr + pb * ob + od[:, None] * od_ + ol[None, :], (x2 * z).to(tl.bfloat16), mask=m)


# ----------------------------------------------------------------------------------------------------------------- the chains
def defer_featurizer(u, weight):
    """The featurizer call of a DEFERRED block: its (B, L, 3D) input returned as the (B, 3D, L) rows view (no copy, any strides) carrying the taps;
    the chains below apply the 3-tap FIR to the x2 / x1 / v rows inline, so z_pre is never materialised for these blocks."""
    x = u.permute(0, 2, 1)
    setattr(x, TAPS, weight)
    CTR["k_fir3_deferred"] += 1
    return x


def _split(self, z, hidden_size):
    """x2, x1, v (B, D, L) views + their (D, 3) tap tables when z is a deferred featurizer input ((None, None, None) for a materialised z_pre)."""
    taps = getattr(z, TAPS, None)
    x2, x1, v = z.split([hidden_size, hidden_size, hidden_size], dim=1)
    w = (None, None, None)
    if taps is not None:
        w3 = taps.reshape(3 * hidden_size, 3)
        w = (w3[:hidden_size], w3[hidden_size:2 * hidden_size], w3[2 * hidden_size:])
    if self.hyena_flip_x1x2:
        x1, x2 = x2, x1
        w = (w[1], w[0], w[2])
    assert x1.stride() == v.stride() == x2.stride()
    return x2, x1, v, w, _I32.strided_extent(z.shape, z.stride())


def _group_spectrum(k_f: torch.Tensor, R: int):
    """(one row per group, R) when the D rows of k_f (1, D, F) are R-fold repeats — checked on the values; else (k_f, 1)."""
    _, Dn, Fn = k_f.shape
    if R > 1:
        g = k_f.view(Dn // R, R, Fn)
        if torch.equal(g, g[:, :1].expand(Dn // R, R, Fn)):
            return g[:, 0].contiguous(), R
    return k_f.contiguous(), 1


def _conv(x1, v, x2, w, zext, filt_f, R, bias, epilogue, tag):
    """(fir3 +) pad+cast -> R2C -> X*F/n -> C2R -> fused epilogue; returns (B, D, L) bf16. w = (x2, x1, v) tap tables or Nones; zext = the input's element extent."""
    Bn, Dn, Ln = x1.shape; n, Fn = KS.fft_sizes(Ln); dev = x1.device
    _I32.check(f"K_{tag}_fft_chain", f"the spectrum (B={Bn}, D={Dn}, L+1={Fn}) as (re, im) pairs / the block input", (KS.shared_floats(Bn, Dn, Ln), zext),
               f"batch <= {_I32.EXTENT // max(1, 3 * Dn * max(Ln, 1), 2 * Dn * Fn)} at L={Ln}")
    BD, BL, NW = GEOM; grid = (Bn, triton.cdiv(Dn, BD), triton.cdiv(Ln, BL))
    buf = _buf("pad", (Bn, Dn, n), torch.float32, dev, zero=True)                 # tail [L, 2L) zero by allocation; nothing writes it
    if w[1] is None:
        _gate_padbuf_kernel[grid](x1, v, buf, Bn, Dn, Ln, x1.stride(0), x1.stride(1), x1.stride(2), buf.stride(0), buf.stride(1), BLOCK_D=BD, BLOCK_L=BL, num_warps=NW)
    else:
        CTR["k_fir3_inline"] += 1
        _fir_pad_kernel[grid](x1, v, w[1], w[2], buf, Dn, Ln, x1.stride(0), x1.stride(1), x1.stride(2), buf.stride(0), buf.stride(1), BLOCK_D=BD, BLOCK_L=BL, num_warps=NW)
    xo = _buf("spectrum_then_output", (KS.shared_floats(Bn, Dn, Ln),), torch.float32, dev)   # the spectrum X (B, D, L+1) complex, then (X consumed by the product) the C2R output (B, D, 2L) fp32 in the same storage
    X = _cf().on(dev).r2c_half(buf, torch.view_as_complex(xo.view(Bn, Dn, Fn, 2)))
    Y = _buf("product", (Bn, Dn, Fn), torch.complex64, dev)
    Xr, Fr, Yr = torch.view_as_real(X), torch.view_as_real(filt_f), torch.view_as_real(Y)
    DF = Dn * Fn
    _mul_kernel[(triton.cdiv(DF, MUL_BLOCK), Bn)](Xr, Fr, Yr, DF, Fn, R, 1.0 / n, Xr.stride(0), BLOCK=MUL_BLOCK, num_warps=MUL_WARPS)
    y = _cf().on(dev).c2r(Y, xo[: Bn * Dn * n].view(Bn, Dn, n))
    out = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=dev)
    epilogue[grid](y, buf, x2, x2 if w[0] is None else w[0], bias, out, Dn, Ln, y.stride(0), y.stride(1), x2.stride(0), x2.stride(1), x2.stride(2), out.stride(0), out.stride(1),
                   FIR=w[0] is not None, BLOCK_D=BD, BLOCK_L=BL, num_warps=NW)
    return out


def parallel_iir_k(self, z_pre, h, D, L, poles, residues, t, dims, layer_idx, inference_params=None, prefill_style="fft",
                   fftconv_fn=None, padding_mask=None, use_flashfft=False, column_split_hyena=False, long_fir_threshold=None):
    """HCL (hcl_fft_conv): H = rfft(h, 2L) by the stock's call — cached from the h compute_filter built on this layer's first call at L, or
    transformed per call for a layer past the budget; then the chain."""
    assert inference_params is None and not use_flashfft and long_fir_threshold is None and not column_split_hyena and padding_mask is None
    x2, x1, v, w, zext = _split(self, z_pre, dims[0])
    dev = x1.device; key = (layer_idx, int(L), str(dev))
    LK.hold(int(L))
    ent = _H.get(key)
    if ent is None:
        assert h.numel() > 0, "HCL spectrum cache miss with the placeholder filter"
        ent = (torch.fft.rfft(h.to(dtype=torch.float32), n=2 * L).contiguous(), 1)
        if key in _PC:
            CTR["k_hcl_spectrum_per_call"] += 1
        else:
            _H[key] = ent; CTR["k_hcl_spectrum_fill"] += 1
    CTR["k_hcl_chain"] += 1
    return _conv(x1, v, x2, w, zext, ent[0], ent[1], D, _hcl_gate_kernel, "hcl").permute(0, 2, 1)


def serves_fir(self, fir_fn, weight, bias, dim_last, fir_length, gate, inference_params, padding_mask, column_split_hyena) -> bool:
    """The parallel_fir call hcm_fft_conv serves on this route (engine.py: gate, fir_length >= 128, the hcm kernel bound and enabled)."""
    return KS.is_hcm_fft_call(fir_length=int(fir_length), gate=bool(gate), dim_last=bool(dim_last), has_inference_params=inference_params is not None,
                              has_padding_mask=padding_mask is not None, has_bias=bias is not None, fir_is_conv1d=fir_fn is torch.nn.functional.conv1d,
                              column_split_hyena=bool(column_split_hyena), use_hcm_kernel=bool(self.use_hcm_kernel), hcm_bound=K7._eng.hcm_fft_conv is not None)


def parallel_fir_k(self, fir_fn, u, weight, bias, L, dims, groups=None, gated_bias=False, column_split_hyena=False, dim_last=True,
                   fir_length=3, gate=False, inference_params=None, prefill_mode=None, padding_mask=None):
    """HCM (hcm_fft_conv under parallel_fir's autocast): k_f = rfft(fp32(weight[:, :, :L]), 2L) by the stock's call, kept one row per filter group
    within the budget or transformed per call past it; then the chain, whose epilogue carries parallel_fir's bf16 cast and x2 gate."""
    x2, x1, v, w, zext = _split(self, u, dims[0])
    Ln = x1.shape[-1]; dev = u.device; key = (self.layer_idx, int(Ln), str(dev))
    LK.hold(int(Ln))
    ent = _KF.get(key)
    if ent is None:
        with torch.autocast("cuda"):
            k_f = torch.fft.rfft(weight[:, :, :Ln].to(torch.float32), n=2 * Ln).squeeze(1).unsqueeze(0)
        ent = _group_spectrum(k_f, KS.group_rows(weight.shape[0], groups))
        if key not in _PC and not _fits(dev, ent[0].numel() * ent[0].element_size()):
            _PC[key] = "hcm"; _name_per_call("hcm", Ln, dev, self.layer_idx)
        if key in _PC:
            CTR["k_hcm_spectrum_per_call"] += 1
        else:
            _KF[key] = ent; CTR["k_hcm_spectrum_fill"] += 1
    CTR["k_hcm_chain"] += 1
    return _conv(x1, v, x2, w, zext, ent[0], ent[1], bias, _hcm_gate_kernel, "hcm"), None


def compute_filter_k(self, L, device):
    """Once this layer's spectrum is cached at L: a (1, 0, 0) placeholder (parallel_forward hands h only to parallel_iir, which reads the spectrum) and
    self.t following L as the stock's does. Otherwise the stock's compute_filter builds h — on the layer's first call at L (parallel_iir fills the
    spectrum from it; h is not kept), or on every call for a layer whose spectrum would take the resident spectra past the budget."""
    dev = str(device); key = (self.layer_idx, int(L), dev)
    LK.hold(int(L))
    if key in _H:
        CTR["k_h_placeholder"] += 1; self.update_time(L, device)
        ph = _PH.get(dev)
        if ph is None:
            ph = _PH[dev] = torch.empty((1, 0, 0), dtype=torch.float32, device=device)
        return ph, torch.float32, None, None
    tdev = torch.device(device) if not isinstance(device, torch.device) else device
    if key not in _PC and not _fits(tdev, KS.spectrum_bytes(self.D.numel(), L)):
        _PC[key] = "hcl"; _name_per_call("hcl", L, tdev, self.layer_idx)
    CTR["k_h_built"] += 1
    return K7._orig_compute_filter(self, L, device)


parallel_iir_k_guarded = M40.on_device(parallel_iir_k)


def compute_filter_k_guarded(self, L, device):
    with M40._guard(torch.device(device) if not isinstance(device, torch.device) else device):
        return compute_filter_k(self, L, device)


def is_unpatched() -> bool:
    return not _H and not _KF and not _PC and not _BUF and _CF is None
