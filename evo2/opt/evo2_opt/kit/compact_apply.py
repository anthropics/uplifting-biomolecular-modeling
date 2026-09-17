"""Installs the compact spectrum chains on the torch-conv chain (one half-spectrum cuFFT object per device)."""
from __future__ import annotations

import ctypes
import torch

from evo2_opt.kit import stores as M2
from evo2_opt.kit import gate as GT
from evo2_opt.kit import multidev as M40
from evo2_opt.kit import compact as C

MF, LG, K7 = M2.MF, M2.LG, M2.K7
HyenaInferenceEngine, HyenaCascade, RMSNorm = K7.HyenaInferenceEngine, K7.HyenaCascade, K7.RMSNorm
CTR = M2.CTR
VERSIONS = {"v0": list(M2.VERSIONS["v0"]) + ["C_SPEC_C"]}      # C_SPEC_C = the compact-spectrum form (b)(c)(d)(d')(e) — HCL on the full plan; tail-zero counted on allocation too
BASE_KIT = "v40_full_2"
GATE = M2.GATE
N_HCL, N_HCM = M40.N_HCL, M40.N_HCM
_applied = {"version": None}
_installed = M2._installed                                   # the gate's wrappers table (shared: remove() walks it)
_vmm, _ORIG_MODEL_FORWARD = M2._vmm, M2._ORIG_MODEL_FORWARD   # the gated model forward's home + its stock original (the precondition reads them)
SHAPES: GT.ShapeManager | None = None                         # = M2.SHAPES after apply (the shape manager with the compact stores)


# ------------------------------------------------------------------------------------------------------- (a) onesided plans, one work area per device
class _CuFFTHalf:
    """Per device: kit.base's ``_CuFFT`` (libcufft via ctypes, torch-2.7.1 argtypes) with an ONESIDED r2c plan (torch.fft.rfft's layout) and the
    same c2r plan; every plan of the device shares ONE work area (sized to the largest request; plans run sequentially on the stream)."""
    CUDA_R_32F, CUDA_C_32F, FORWARD, INVERSE = K7._CuFFT.CUDA_R_32F, K7._CuFFT.CUDA_C_32F, K7._CuFFT.FORWARD, K7._CuFFT.INVERSE

    def __init__(self):
        self.base = K7._CuFFT()                                 # loads libcufft + sets the argtypes; its own plan table stays empty
        self.lib = self.base.lib
        self.lib.cufftDestroy.argtypes = [ctypes.c_int]
        self.plans = {}                                         # (kind, batch, n) -> (handle, ws_bytes)
        self.work = None                                        # the shared work area (torch uint8 tensor on this device)
        self.work_bytes = 0

    def _plan(self, kind, batch, n):
        key = (kind, int(batch), int(n))
        if key in self.plans:
            return self.plans[key]
        L, ll = self.lib, ctypes.c_longlong
        h = ctypes.c_int(); assert L.cufftCreate(ctypes.byref(h)) == 0
        assert L.cufftSetAutoAllocation(h.value, 0) == 0
        a = C.r2c_half_plan_args(n, batch) if kind == "r2c_half" else C.r2c_full_plan_args(n, batch) if kind == "r2c_full" else C.c2r_plan_args(n, batch) if kind == "c2r" else None
        if a is None:
            raise ValueError(kind)
        ws = ctypes.c_size_t(); T = {"CUDA_R_32F": self.CUDA_R_32F, "CUDA_C_32F": self.CUDA_C_32F}
        nn = (ll * len(a["n"]))(*a["n"]); emb_in = (ll * len(a["inembed"]))(*a["inembed"]); emb_out = (ll * len(a["onembed"]))(*a["onembed"])
        rc = L.cufftXtMakePlanMany(h.value, a["rank"], nn, emb_in, a["istride"], a["idist"], T[a["itype"]], emb_out, a["ostride"], a["odist"], T[a["otype"]], a["batch"], ctypes.byref(ws), T[a["exec"]])
        assert rc == 0, f"cufftXtMakePlanMany({kind}) rc={rc}"
        need = int(ws.value)
        if need > self.work_bytes:                              # grow the SHARED work area; re-point every existing plan at it
            self.work = torch.empty(need, dtype=torch.uint8, device="cuda"); self.work_bytes = need
            for (hh, _) in self.plans.values():
                assert L.cufftSetWorkArea(int(hh), self.work.data_ptr()) == 0
        if need:
            assert L.cufftSetWorkArea(h.value, self.work.data_ptr()) == 0
        self.plans[key] = (h.value, need); CTR["c_plan_created"] += 1
        return self.plans[key]

    def r2c_half(self, buf, out_half):
        """buf (B, D, n) fp32 contiguous -> out_half (B, D, n/2+1) complex64 contiguous (the persistent spectrum buffer)."""
        Bn, Dn, n = buf.shape; assert out_half.shape == (Bn, Dn, n // 2 + 1) and buf.is_contiguous() and out_half.is_contiguous()
        h, _ = self._plan("r2c_half", Bn * Dn, n)
        assert self.lib.cufftSetStream(h, torch.cuda.current_stream().cuda_stream) == 0
        assert self.lib.cufftXtExec(h, buf.data_ptr(), out_half.data_ptr(), self.FORWARD) == 0; CTR["c_r2c_half"] += 1
        return out_half

    def r2c_full(self, buf, out_full):
        """The E60 full plan: buf (B, D, n) fp32 contiguous -> out_full (B, D, n) complex64 contiguous (persistent; the first n/2+1 of every row written); returns
        the (B, D, n/2+1) strided view (the bins the stock slices)."""
        Bn, Dn, n = buf.shape; assert out_full.shape == (Bn, Dn, n) and buf.is_contiguous() and out_full.is_contiguous()
        h, _ = self._plan("r2c_full", Bn * Dn, n)
        assert self.lib.cufftSetStream(h, torch.cuda.current_stream().cuda_stream) == 0
        assert self.lib.cufftXtExec(h, buf.data_ptr(), out_full.data_ptr(), self.FORWARD) == 0; CTR["c_r2c_full"] += 1
        return out_full[..., : n // 2 + 1]

    def c2r(self, Y, n, out):
        """Y (B, D, n/2+1) complex64 contiguous (consumed) -> out (B, D, n) fp32 contiguous (the SAME buffer the r2c read: (b))."""
        Bn, Dn, h2 = Y.shape; assert h2 == n // 2 + 1 and Y.is_contiguous() and out.shape == (Bn, Dn, n) and out.is_contiguous()
        h, _ = self._plan("c2r", Bn * Dn, n)
        assert self.lib.cufftSetStream(h, torch.cuda.current_stream().cuda_stream) == 0
        assert self.lib.cufftXtExec(h, Y.data_ptr(), out.data_ptr(), self.INVERSE) == 0; CTR["c_c2r_alias"] += 1
        return out

    def destroy(self):
        for (h, _) in list(self.plans.values()):
            try:
                self.lib.cufftDestroy(int(h))
            except Exception:
                pass
        self.plans.clear(); self.work = None; self.work_bytes = 0


class CuFFTHalfPerDevice:
    def __init__(self):
        self.per_device = {}

    def _for(self, device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if idx not in self.per_device:
            with torch.cuda.device(idx):
                self.per_device[idx] = _CuFFTHalf()
        return self.per_device[idx]

    def r2c_half(self, buf, out_half):
        assert buf.device == out_half.device
        with torch.cuda.device(buf.device):
            return self._for(buf.device).r2c_half(buf, out_half)

    def r2c_full(self, buf, out_full):
        assert buf.device == out_full.device
        with torch.cuda.device(buf.device):
            return self._for(buf.device).r2c_full(buf, out_full)

    def c2r(self, Y, n, out):
        assert Y.device == out.device
        with torch.cuda.device(Y.device):
            return self._for(Y.device).c2r(Y, n, out)

    def destroy(self):
        for cf in self.per_device.values():
            cf.destroy()
        self.per_device.clear()

    @property
    def plans(self):
        return {idx: dict(cf.plans) for idx, cf in self.per_device.items()}

    @property
    def work_bytes(self):
        return {idx: cf.work_bytes for idx, cf in self.per_device.items()}


_CFH: CuFFTHalfPerDevice | None = None
_DEPS: C.Deps | None = None


def _deps():
    assert _DEPS is not None, "compact chains before apply"
    return _DEPS


def _parallel_iir_c(self, *args, **kwargs):
    return C.parallel_iir_c(_deps(), self, *args, **kwargs)


def _parallel_fir_c(self, *args, **kwargs):
    return C.parallel_fir_c(_deps(), self, *args, **kwargs)


def _compute_filter_c(self, L, device):
    return C.compute_filter_c(_deps(), self, L, device)


_parallel_iir_c_guarded = M40.on_device(_parallel_iir_c)
_parallel_fir_c_guarded = M40.on_device(_parallel_fir_c)


def _compute_filter_c_guarded(self, L, device):
    with M40._guard(torch.device(device) if not isinstance(device, torch.device) else device):
        return _compute_filter_c(self, L, device)


# ------------------------------------------------------------------------------------------------------- tables
def expected_counts(version):
    """Per forward after the fill forward: kit.stores' table with the base chains' counters replaced by the compact chains' (compact.compact_expected)."""
    assert version in VERSIONS, version
    return C.compact_expected(M2.expected_counts("v0"), N_HCL, N_HCM)


def fill_counts(version):
    """The first forward (compact.compact_fill on kit.stores' fill table)."""
    assert version in VERSIONS, version
    return C.compact_fill(M2.fill_counts("v0"), N_HCL, N_HCM)


counters = M2.counters


def _stores():
    return {**M2._stores(), "c_spec_half_and_io_buf": (lambda: _DEPS.clear_stores() if _DEPS is not None else None), "c_cufft_plans": (lambda: _CFH.destroy() if _CFH is not None else None)}


def _model_stores(model):
    return M2._model_stores(model)


# ------------------------------------------------------------------------------------------------------- apply / ensure_shape
def apply(model, version="v0"):
    """kit.stores' apply (the composed chain + the gate), then the two compact op chains + the (e) compute_filter REPLACE the kit branch of
    the corresponding gate wrappers, and the shape manager gets the compact stores (the gate, passthrough and route table untouched)."""
    global _CFH, SHAPES, _DEPS
    assert _applied["version"] is None, f"kit already applied ({_applied['version']})"
    assert version in VERSIONS, version
    info = M2.apply(model, "v0")
    _CFH = CuFFTHalfPerDevice()
    _DEPS = C.Deps(pad_kernel=K7._gate_padbuf_kernel, hcl_epilogue=K7._hcl_epilogue_kernel, hcm_epilogue=K7._hcm_epilogue_kernel, grid=K7._bdl_grid,
                   H_cache=K7._H_cache, kf_cache=K7._kf_cache, filter_cache=K7._filter_cache, adj=K7._adj, cached_compute_filter=K7._cached_compute_filter,
                   cf=_CFH, parallel_fir_v2=K7._parallel_fir_v2, CTR=CTR,
                   full_factory=lambda Bn, Dn, n, device: K7._spec_buf((Bn, Dn, n), torch.complex64, device))     # kit.base's E60 full buffer (one per shape per device)
    for name, kit_fn in (("HyenaInferenceEngine.parallel_iir", _parallel_iir_c_guarded), ("HyenaInferenceEngine.parallel_fir", _parallel_fir_c_guarded),
                         ("HyenaCascade.compute_filter", _compute_filter_c_guarded)):
        owner, attr, old_kit, w = _installed[name]
        w2 = GT.gated(GATE, kit_fn, w.__wrapped_stock__ if hasattr(w, "__wrapped_stock__") else _stock_of(w), name)
        setattr(owner, attr, w2)
        _installed[name] = (owner, attr, kit_fn, w2)
    M2.SHAPES = GT.ShapeManager(_stores(), CTR, devices=M2._devices(model), model_stores=_model_stores(model))
    M2._vmm.StripedHyena.forward = GT.forward_gate(M2._ORIG_MODEL_FORWARD, GATE, M2.SHAPES)
    SHAPES = M2.SHAPES
    _applied["version"] = version
    info["compact_spectrum"] = {"levers": ["b_io_alias_tail_memset", "c_hcm_onesided_plan_eq_torch_rfft", "d_hcm_mul_inplace", "dprime_hcl_mul_into_packed_half", "e_h_dropped", "shared_work_area_per_device", "a_onesided_plan_for_hcl_DEAD_probe_cufft_layout_2"], "base_kit": BASE_KIT}
    info["gate"]["shape_stores"] = sorted(_stores())
    return info


def _stock_of(w):
    st = getattr(w, "__stock_fn__", None)
    assert st is not None, "gate wrapper without a stock function"
    return st


def ensure_shape(B, L):
    return M2.ensure_shape(B, L)


PROCESS_WIDE_PATCHES = M2.PROCESS_WIDE_PATCHES


def is_unpatched(model=None):
    return M2.is_unpatched(model) and _applied["version"] is None and _CFH is None and _DEPS is None


