"""C2R_OUT: the compact chains with the inverse transform written outside the io buffer (no tail memset per call): into the first half of the full-row spectrum buffer's storage, which no transform reads by then."""
from __future__ import annotations

import torch

from evo2_opt.kit import compact as C

N_C2R_PER_FORWARD = 28                      # 14 HCL + 14 HCM blocks (evo2-40b-1m.yml; kit.multidev's N_HCL, N_HCM defaults)


class Deps8:
    """A view over kit.compact's Deps (the same stores, caches, kernels, cuFFT object and counters) with the c2r OUTPUT outside the
    io buffer: ``out`` (compact.Deps.out) is the full-row spectrum buffer's storage, one per (B, D, n) per device. ``io`` never re-zeroes: the
    io buffer's tail is zero by allocation and nothing writes it (the c2r targets ``out``)."""
    def __init__(self, base):
        self.base = base
        self.pad_cfg = (32, 128)                        # the launch geometry of kit.compact's chains (BLOCK_D, BLOCK_L); not a knob

    def __getattr__(self, name):                        # everything else is the base Deps' (kernels, caches, cf, CTR, launch, grid, adj, out, ...)
        return getattr(self.base, name)

    def io(self, Bn, Dn, n, Ln, device):
        key = ((Bn, Dn, n), str(device))
        buf = self.base.io_buf.get(key)
        if buf is None:
            buf = self.base.io_buf[key] = torch.zeros(Bn, Dn, n, dtype=torch.float32, device=device)
            self.CTR["c_io_alloc"] += 1
        self.CTR["c_tail_zero"] += 1                    # the invariant the r2c needs, by allocation alone: no memset, ever
        return buf

    def clear_stores(self):
        self.base.clear_stores()

    def tails_are_zero(self) -> dict:
        """The invariant check (after forwards): every io buffer's tail beyond its largest L is zero; the out storage (the full-row spectrum
        buffer's) is distinct from the io buffers."""
        rep = {}
        for key, buf in self.base.io_buf.items():
            (Bn, Dn, n), dev = key; Ln = n // 2
            tail_nonzero = int(torch.count_nonzero(buf[..., Ln:]).item())
            ob = self.base.full_spec.get(key)
            rep[str(key)] = {"tail_nonzero": tail_nonzero, "out_present": ob is not None, "distinct_storage": (ob is not None and ob.data_ptr() != buf.data_ptr())}
        return rep


def parallel_iir_c8(d: Deps8, self, z_pre, h, D, L, poles, residues, t, dims, layer_idx, inference_params=None, prefill_style="fft",
                    fftconv_fn=None, padding_mask=None, use_flashfft=False, column_split_hyena=False, long_fir_threshold=None):
    """HCL: the compact chain (compact.parallel_iir_c) with the c2r into the out storage instead of the io buffer (the arithmetic
    of every kernel unchanged; the launch geometry the base's)."""
    assert inference_params is None and not use_flashfft and long_fir_threshold is None and not column_split_hyena and padding_mask is None
    fft_size = 2 * L; hidden_size = dims[0]
    x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)
    if self.hyena_flip_x1x2:
        x1, x2 = x2, x1
    key = (layer_idx, int(L), str(x1.device))
    if key not in d.H_cache:
        assert h.numel() > 0, "H cache miss with the (e) placeholder h: compute_filter dropped h before H was cached"
        d.H_cache[key] = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
        C.drop_h(d, key)
    H = d.H_cache[key]
    Bn, Dn, Ln = x1.shape
    BD, BL = d.pad_cfg
    buf = d.io(Bn, Dn, fft_size, Ln, x1.device)
    d.launch(d.pad_kernel, d.grid(Bn, Dn, Ln, BD, BL))(x1, v, buf, Bn, Dn, Ln, x1.stride(0), x1.stride(1), x1.stride(2), buf.stride(0), buf.stride(1), BLOCK_D=BD, BLOCK_L=BL)
    X = d.cf.r2c_full(buf, d.full(Bn, Dn, fft_size, x1.device))
    Y = d.spec(Bn, Dn, fft_size, x1.device)
    torch.mul(X, H, out=Y); d.CTR["c_mul_packed"] += 1
    y = d.cf.c2r(Y, fft_size, d.out(Bn, Dn, fft_size, x1.device))                  # into the full-row buffer's storage (X is consumed); the io tail stays zero
    out = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=x1.device)
    d.launch(d.hcl_epilogue, d.grid(Bn, Dn, Ln, BD, BL))(y, x1, v, x2, D, out, Bn, Dn, Ln, y.stride(0), y.stride(1), x1.stride(0), x1.stride(1), x1.stride(2),
                                                         out.stride(0), out.stride(1), BLOCK_D=BD, BLOCK_L=BL)
    d.CTR["hcl_compact"] += 1; d.CTR["c2r_dedicated_out"] += 1
    return out.permute(0, 2, 1)


def parallel_fir_c8(d: Deps8, self, fir_fn, u, weight, bias, L, dims, groups=None, gated_bias=False, column_split_hyena=False, dim_last=True,
                    fir_length=3, gate=False, inference_params=None, prefill_mode=None, padding_mask=None):
    """HCM: the compact chain (compact.parallel_fir_c) with the c2r into the out storage; every other FIR = the E7 path."""
    if fir_length >= 128 and gate and not dim_last and inference_params is None and padding_mask is None and fir_fn == torch.nn.functional.conv1d and bias is not None:
        hidden_size = dims[0]
        x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1
        Bn, Dn, Ln = x1.shape; fft_size = 2 * Ln
        key = (self.layer_idx, Ln, str(u.device))
        BD, BL = d.pad_cfg
        buf = d.io(Bn, Dn, fft_size, Ln, u.device)
        d.launch(d.pad_kernel, d.grid(Bn, Dn, Ln, BD, BL))(x1, v, buf, Bn, Dn, Ln, x1.stride(0), x1.stride(1), x1.stride(2), buf.stride(0), buf.stride(1), BLOCK_D=BD, BLOCK_L=BL)
        X = d.cf.r2c_half(buf, d.spec(Bn, Dn, fft_size, u.device))
        if key not in d.kf_cache:
            d.kf_cache[key] = C.hcm_spectrum(d, weight, groups, X, Bn, Dn, Ln, fft_size, u.device)
        C.mul_spectrum_(X, d.kf_cache[key], Bn, Dn); d.CTR["c_mul_inplace"] += 1
        y = d.cf.c2r(X, fft_size, d.out(Bn, Dn, fft_size, u.device))                # into the full-row buffer's storage (the HCM chain never writes it otherwise)
        out = torch.empty((Bn, Dn, Ln), dtype=torch.bfloat16, device=u.device)
        d.launch(d.hcm_epilogue, d.grid(Bn, Dn, Ln, BD, BL))(y, x1, v, x2, bias, out, Bn, Dn, Ln, y.stride(0), y.stride(1), x1.stride(0), x1.stride(1), x1.stride(2),
                                                             out.stride(0), out.stride(1), BLOCK_D=BD, BLOCK_L=BL)
        d.CTR["hcm_compact"] += 1; d.CTR["c2r_dedicated_out"] += 1
        return out, None                                                              # the HCM chain's (out, None), as compact.parallel_fir_c's
    return d.parallel_fir_v2(self, fir_fn, u, weight, bias, L, dims, groups=groups, gated_bias=gated_bias, column_split_hyena=column_split_hyena,
                             dim_last=dim_last, fir_length=fir_length, gate=gate, inference_params=inference_params, prefill_mode=prefill_mode, padding_mask=padding_mask)


def expected8(base: dict, n_hcl: int, n_hcm: int) -> dict:
    """The per-forward table after the fill forward: ``base`` (kit.hooks_apply's) + the chain counter (the tail memset counter never appears)."""
    c = dict(base); c["c2r_dedicated_out"] = n_hcl + n_hcm
    assert "c_tail_memset" not in c
    return c


