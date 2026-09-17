"""C_SPEC: the compact half-spectrum long-convolution chains (one-sided r2c into a shared spectrum, the product in place, c2r into the same buffer) and their cuFFT plan arguments."""
from __future__ import annotations

import torch

from evo2_opt.kit import kshapes as KS


class Deps:
    def __init__(self, pad_kernel, hcl_epilogue, hcm_epilogue, grid, H_cache, kf_cache, filter_cache, adj, cached_compute_filter, cf, parallel_fir_v2, CTR,
                 launch=None, full_factory=None):
        self.pad_kernel, self.hcl_epilogue, self.hcm_epilogue, self.grid = pad_kernel, hcl_epilogue, hcm_epilogue, grid
        self.H_cache, self.kf_cache, self.filter_cache, self.adj, self.cached_compute_filter = H_cache, kf_cache, filter_cache, adj, cached_compute_filter
        self.cf, self.parallel_fir_v2, self.CTR = cf, parallel_fir_v2, CTR
        self.launch = launch or (lambda kernel, grid: kernel[grid])     # Triton: kernel[grid](...); tests: a plain callable
        self.full_factory = full_factory                                 # kit.base's persistent full buffer (module) | None (tests)
        self.spec_half: dict = {}
        self.full_spec: dict = {}
        self.io_buf: dict = {}
        self.placeholder: dict = {}
        self.h_keys: dict = {}                                           # (layer_idx, L, device) -> the L1b filter-cache key of that layer's h

    # stores
    def spec(self, Bn, Dn, n, device):
        key = ((Bn, Dn, n // 2 + 1), str(device))
        if key not in self.spec_half:
            self.spec_half[key] = torch.empty(Bn, Dn, n // 2 + 1, dtype=torch.complex64, device=device)
        return self.spec_half[key]

    def full(self, Bn, Dn, n, device):
        """The HCL full-row complex buffer (B, D, n) of the E60 full plan (``full_factory``: kit.base's _spec_buf in the module; a dict here)."""
        key = ((Bn, Dn, n), str(device))
        if key not in self.full_spec:
            self.full_spec[key] = self.full_factory(Bn, Dn, n, device) if self.full_factory else torch.empty(Bn, Dn, n, dtype=torch.complex64, device=device)
        return self.full_spec[key]

    def out(self, Bn, Dn, n, device):
        """The inverse transform's (B, D, n) fp32 output: the first half of the full-row spectrum buffer's storage. Within a layer's call the
        full-row spectrum is read into the half spectrum by the product before the c2r writes here (HCL), or never written (HCM); every
        layer call of a device runs on that device's one block stream, so no other call's spectrum is live in it."""
        full = self.full(Bn, Dn, n, device)
        return torch.view_as_real(full).reshape(-1)[: Bn * Dn * n].view(Bn, Dn, n)

    def io(self, Bn, Dn, n, Ln, device):
        key = ((Bn, Dn, n), str(device))
        buf = self.io_buf.get(key)
        if buf is None:
            buf = self.io_buf[key] = torch.zeros(Bn, Dn, n, dtype=torch.float32, device=device)
            self.CTR["c_io_alloc"] += 1                                      # the tail is zero by allocation (first call at this shape per device: a process's
        else:                                                                # first forward, or the first forward after a shape release)
            buf[..., Ln:].zero_(); self.CTR["c_tail_memset"] += 1           # (b): the c2r wrote all n columns last time
        self.CTR["c_tail_zero"] += 1                                         # the invariant the r2c needs: the tail is zero (by allocation or by the memset)
        return buf

    def clear_stores(self):
        self.spec_half.clear(); self.io_buf.clear(); self.full_spec.clear(); self.h_keys.clear()


def hcm_spectrum(d: Deps, weight, groups, X, Bn, Dn, Ln, fft_size, device):
    """The kf_cache entry (k, R): the stock's k_f = rfft(k, 2L) / 2L of the (D, 1, W) FIR weight, stored ONE ROW PER FILTER GROUP — k (G, 1, F),
    R = D // G rows per group — when the weight is the groups' rows repeated R times: checked on the spectrum's rows and on this call's own
    product (the group form broadcast over the group's rows against the full rows, bit for bit); otherwise the full rows (k (1, D, F), R = 1)."""
    k = weight[:, :, :Ln].to(torch.float32)
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size                                   # (D, 1, F)
    k_full = d.adj(torch.empty((Bn, Dn, Ln), device=device), k_f)
    R = KS.group_rows(Dn, groups)
    if R > 1:
        F = k_f.shape[-1]
        rows = k_f.reshape(Dn, F).view(Dn // R, R, F)
        k_g = rows[:, :1].contiguous()                                                # (G, 1, F)
        if torch.equal(torch.view_as_real(rows[:, 1:]), torch.view_as_real(rows[:, :1].expand(Dn // R, R - 1, F))) and torch.equal(
                torch.view_as_real(torch.mul(X, k_full)), torch.view_as_real(torch.mul(X.view(Bn, Dn // R, R, F), k_g).view(Bn, Dn, F))):
            d.CTR["c_kf_group_rows"] += 1
            return k_g, R
    d.CTR["c_kf_full_rows"] += 1
    return k_full, 1


def mul_spectrum_(X, entry, Bn, Dn):
    """X *= the cached filter spectrum, in place: the group form over X viewed (B, G, R, F), the full form over X as is."""
    k, R = entry
    if R > 1:
        Xg = X.view(Bn, Dn // R, R, X.shape[-1]); torch.mul(Xg, k, out=Xg)
    else:
        torch.mul(X, k, out=X)


def drop_h(d: Deps, hkey) -> None:
    """Once a layer's spectrum H is cached under ``hkey`` = (layer_idx, L, device), its time-domain filter leaves the L1b filter cache: the (e)
    placeholder serves the layer's later compute_filter calls."""
    fkey = d.h_keys.pop(hkey, None)
    if fkey is not None and fkey in d.filter_cache:
        del d.filter_cache[fkey]; d.CTR["c_h_dropped"] += 1


def parallel_iir_c(d: Deps, self, z_pre, h, D, L, poles, residues, t, dims, layer_idx, inference_params=None, prefill_style="fft",
                   fftconv_fn=None, padding_mask=None, use_flashfft=False, column_split_hyena=False, long_fir_threshold=None):
    """HCL: pad -> onesided r2c into the shared spectrum -> Y = X*H in place -> c2r into the SAME fp32 buffer -> the fused epilogue."""
    assert inference_params is None and not use_flashfft and long_fir_threshold is None and not column_split_hyena and padding_mask is None
    fft_size = 2 * L; hidden_size = dims[0]
    x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)
    if self.hyena_flip_x1x2:
        x1, x2 = x2, x1
    key = (layer_idx, int(L), str(x1.device))
    if key not in d.H_cache:
        assert h.numel() > 0, "H cache miss with the (e) placeholder h: compute_filter dropped h before H was cached"
        d.H_cache[key] = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
        drop_h(d, key)
    H = d.H_cache[key]
    Bn, Dn, Ln = x1.shape
    buf = d.io(Bn, Dn, fft_size, Ln, x1.device)
    d.launch(d.pad_kernel, d.grid(Bn, Dn, Ln))(x1, v, buf, Bn, Dn, Ln, x1.stride(0), x1.stride(1), x1.stride(2), buf.stride(0), buf.stride(1), BLOCK_D=32, BLOCK_L=128)
    X = d.cf.r2c_full(buf, d.full(Bn, Dn, fft_size, x1.device))                  # the FULL r2c plan (E60 == the stock HCL plan); X = the first n/2+1 bins (a strided view)
    Y = d.spec(Bn, Dn, fft_size, x1.device)
    torch.mul(X, H, out=Y); d.CTR["c_mul_packed"] += 1                            # (d'): X*H written into the packed half buffer (no transient)
    y = d.cf.c2r(Y, fft_size, buf)                                                 # (b)
    out = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=x1.device)
    d.launch(d.hcl_epilogue, d.grid(Bn, Dn, Ln))(y, x1, v, x2, D, out, Bn, Dn, Ln, y.stride(0), y.stride(1), x1.stride(0), x1.stride(1), x1.stride(2),
                                                 out.stride(0), out.stride(1), BLOCK_D=32, BLOCK_L=128)
    d.CTR["hcl_compact"] += 1
    return out.permute(0, 2, 1)


def parallel_fir_c(d: Deps, self, fir_fn, u, weight, bias, L, dims, groups=None, gated_bias=False, column_split_hyena=False, dim_last=True,
                   fir_length=3, gate=False, inference_params=None, prefill_mode=None, padding_mask=None):
    """HCM (fir_length >= 128): the same chain with the cached filter spectrum; every other FIR = the E7 path (parallel_fir_v2)."""
    if fir_length >= 128 and gate and not dim_last and inference_params is None and padding_mask is None and fir_fn == torch.nn.functional.conv1d and bias is not None:
        hidden_size = dims[0]
        x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1
        Bn, Dn, Ln = x1.shape; fft_size = 2 * Ln
        key = (self.layer_idx, Ln, str(u.device))
        buf = d.io(Bn, Dn, fft_size, Ln, u.device)
        d.launch(d.pad_kernel, d.grid(Bn, Dn, Ln))(x1, v, buf, Bn, Dn, Ln, x1.stride(0), x1.stride(1), x1.stride(2), buf.stride(0), buf.stride(1), BLOCK_D=32, BLOCK_L=128)
        X = d.cf.r2c_half(buf, d.spec(Bn, Dn, fft_size, u.device))                # (c): the onesided plan == torch.fft.rfft's _fft_r2c layout
        if key not in d.kf_cache:
            d.kf_cache[key] = hcm_spectrum(d, weight, groups, X, Bn, Dn, Ln, fft_size, u.device)
        mul_spectrum_(X, d.kf_cache[key], Bn, Dn); d.CTR["c_mul_inplace"] += 1   # (d)
        y = d.cf.c2r(X, fft_size, buf)                                             # (b)
        out = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=u.device)
        d.launch(d.hcm_epilogue, d.grid(Bn, Dn, Ln))(y, x1, v, x2, bias, out, Bn, Dn, Ln, y.stride(0), y.stride(1), x1.stride(0), x1.stride(1), x1.stride(2),
                                                     out.stride(0), out.stride(1), BLOCK_D=32, BLOCK_L=128)
        d.CTR["hcm_compact"] += 1
        return out, None
    return d.parallel_fir_v2(self, fir_fn, u, weight, bias, L, dims, groups=groups, gated_bias=gated_bias, column_split_hyena=column_split_hyena,
                             dim_last=dim_last, fir_length=fir_length, gate=gate, inference_params=inference_params, prefill_mode=prefill_mode, padding_mask=padding_mask)


def compute_filter_c(d: Deps, self, L, device):
    """(e): kit.base's cached compute_filter until this layer's spectrum H is cached; from then on a 0-element placeholder is returned (the stock
    forward only hands h to parallel_iir, which uses H; the chain dropped h from the filter cache when it cached H)."""
    key = (id(self), int(L), str(device))
    hkey = (self.layer_idx, int(L), str(device))
    if hkey in d.H_cache:
        if key in d.filter_cache:
            del d.filter_cache[key]; d.CTR["c_h_dropped"] += 1
        d.CTR["c_h_placeholder"] += 1; d.CTR["hcl_filter_cache_hit"] += 1        # counted as the warm path of kit.base's cached compute_filter (the per-forward tables read it)
        ph = d.placeholder.get(str(device))
        if ph is None:
            ph = d.placeholder[str(device)] = torch.empty((1, 0, 0), dtype=torch.float32, device=device)
        return ph, torch.float32, None, None
    d.h_keys[hkey] = key
    return d.cached_compute_filter(self, L, device)


# the onesided r2c plan parameters (torch.fft.rfft's _fft_r2c onesided=True layout) — ONE home; the cuFFT class (compact_apply._CuFFTHalf) reads these
def r2c_half_plan_args(n: int, batch: int):
    """cufftXtMakePlanMany(plan, rank=1, n=[n], inembed=[n], istride=1, idist=n, CUDA_R_32F, onembed=[n/2+1], ostride=1, odist=n/2+1, CUDA_C_32F, batch, ws, CUDA_C_32F)."""
    return {"rank": 1, "n": [n], "inembed": [n], "istride": 1, "idist": n, "itype": "CUDA_R_32F", "onembed": [n // 2 + 1], "ostride": 1, "odist": n // 2 + 1, "otype": "CUDA_C_32F", "batch": batch, "exec": "CUDA_C_32F"}


def r2c_full_plan_args(n: int, batch: int):
    """The E60 full plan (torch _fft_r2c onesided=False, the stock HCL plan): inembed=[n], istride=1, idist=n, CUDA_R_32F -> onembed=[n], ostride=1, odist=n, CUDA_C_32F."""
    return {"rank": 1, "n": [n], "inembed": [n], "istride": 1, "idist": n, "itype": "CUDA_R_32F", "onembed": [n], "ostride": 1, "odist": n, "otype": "CUDA_C_32F", "batch": batch, "exec": "CUDA_C_32F"}


def c2r_plan_args(n: int, batch: int):
    """The E59 c2r plan (torch _fft_c2r): inembed=[n], istride=1, idist=n/2+1, CUDA_C_32F -> onembed=[n], ostride=1, odist=n, CUDA_R_32F."""
    return {"rank": 1, "n": [n], "inembed": [n], "istride": 1, "idist": n // 2 + 1, "itype": "CUDA_C_32F", "onembed": [n], "ostride": 1, "odist": n, "otype": "CUDA_R_32F", "batch": batch, "exec": "CUDA_C_32F"}


# ------------------------------------------------------------------------------------------------------ the per-forward tables (one home)
REPLACED = ("hcl_fused_v2", "hcm_fused_v2", "e60_r2c_nofill", "e59_c2r_noclone")     # the base chains' counters (kit.base _parallel_iir_v5 / _parallel_fir_v5) the compact chains replace


def compact_expected(base: dict, n_hcl: int, n_hcm: int) -> dict:
    """The per-forward table AFTER the fill forward: ``base`` (kit.stores' table) with the base chains' counters replaced by the
    compact chains' (the cuFFT class's c_r2c_half / c_c2r_alias, the chains' c_mul_inplace / c_tail_zero / hcl_compact / hcm_compact,
    the (e) placeholder). c_plan_created, c_h_dropped and c_kf_group_rows / c_kf_full_rows are one-time (the first forward) and never in a
    per-forward table."""
    c = dict(base)
    for k in REPLACED:
        assert k in c, k; del c[k]
    c.update({"hcl_compact": n_hcl, "hcm_compact": n_hcm, "c_r2c_full": n_hcl, "c_r2c_half": n_hcm, "c_c2r_alias": n_hcl + n_hcm, "c_mul_packed": n_hcl,
              "c_mul_inplace": n_hcm, "c_tail_zero": n_hcl + n_hcm, "c_h_placeholder": n_hcl})
    return c


def compact_fill(base_fill: dict, n_hcl: int, n_hcm: int) -> dict:
    """The FIRST forward: the io buffers are allocated on their first use per device (c_tail_zero depends on the device count) and h is
    computed, not yet dropped (no placeholder) — those two are not asserted on the fill forward."""
    c = dict(base_fill)
    for k in REPLACED:
        assert k in c, k; del c[k]
    c.update({"hcl_compact": n_hcl, "hcm_compact": n_hcm, "c_r2c_full": n_hcl, "c_r2c_half": n_hcm, "c_c2r_alias": n_hcl + n_hcm, "c_mul_packed": n_hcl, "c_mul_inplace": n_hcm})
    return c
