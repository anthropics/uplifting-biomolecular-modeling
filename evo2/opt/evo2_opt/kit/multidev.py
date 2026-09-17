"""The base levers under a current-device guard (the 40b's two-device layer split: every kernel launches on the device of its input) and one cuFFT wrapper per device; the torch-conv chain's launch conditions read from the constructed model."""
import contextlib
import functools
import hashlib
import importlib.metadata

import torch
import vortex.model.engine as _eng
from vortex.model.engine import HyenaInferenceEngine
from vortex.model.model import HyenaCascade, ParallelGatedConvBlock, AttentionBlock, StripedHyena
from vortex.model.layers import RMSNorm, HAS_TE

from evo2_opt.kit import base as K7          # kit.base (kernels, caches, buffers, op chains)

CTR = K7.CTR
HIDDEN = 8192
N_HCS, N_HCM, N_HCL, N_ATTN = 14, 14, 14, 8
N_BLOCKS = N_HCS + N_HCM + N_HCL + N_ATTN
N_RMS = 2 * N_BLOCKS + 1
WQKV_STRIDES = (1, 3 * HIDDEN)
PINS = {"torch": "2.7.1", "triton": "3.3.1", "vtx": "1.1.0", "evo2": "0.6.0", "transformer_engine": "2.3.0"}


# ------------------------------------------------------------------------------------------------------- current-device guard
def _guard(device):
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


def _first_tensor(args, kwargs):
    for a in args:
        if isinstance(a, torch.Tensor):
            return a
        if isinstance(a, tuple) and a and isinstance(a[0], torch.Tensor):                      # a block's carried pair (E66)
            return a[0]
    for a in kwargs.values():
        if isinstance(a, torch.Tensor):
            return a
    raise TypeError("no tensor argument to bind the current device to")


def on_device(fn):
    """``fn(self, *args)`` run with the current CUDA device = the device of its first tensor argument (a no-op when they agree)."""
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        with _guard(_first_tensor(args, kwargs).device):
            return fn(self, *args, **kwargs)
    wrapped.__wrapped_kit_fn__ = fn
    return wrapped


GUARDED = ("_parallel_iir_Hcache_padbuf", "_parallel_iir_v2", "_parallel_iir_v5",
           "_parallel_fir_bf16", "_parallel_fir_fused", "_parallel_fir_v2", "_parallel_fir_v5", "_rms_fused_forward", "_attn_forward")
G = {name: on_device(getattr(K7, name)) for name in GUARDED}


# ------------------------------------------------------------------------------------------------------- cuFFT per device
class CuFFTPerDevice:
    """One kit.base ``_CuFFT`` (plans keyed (kind, batch, n); the torch-2.7.1 plan parameters) per CUDA device. A plan is created and
    executed with ITS device current, on that device's current stream; the work area lives on that device."""

    def __init__(self):
        self.per_device = {}

    def _for(self, device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if idx not in self.per_device:
            with torch.cuda.device(idx):
                self.per_device[idx] = K7._CuFFT()
        return self.per_device[idx]

    def r2c_full(self, buf, out_full):
        assert buf.device == out_full.device, (buf.device, out_full.device)
        with torch.cuda.device(buf.device):
            return self._for(buf.device).r2c_full(buf, out_full)

    def c2r(self, Y, n, out):
        assert Y.device == out.device, (Y.device, out.device)
        with torch.cuda.device(Y.device):
            return self._for(Y.device).c2r(Y, n, out)

    @property
    def plans(self):
        return {idx: dict(cf.plans) for idx, cf in self.per_device.items()}


# ------------------------------------------------------------------------------------------------------- versions / counts
VERSIONS = {
    "v0": ["L1b", "E10", "L1c"],
    "v1": ["L1b", "E10", "L1c", "E8"],
    "v2": ["L1b", "E10", "L1c", "E8", "E7"],
    "v3": ["L1b", "E10", "L1c", "E8", "E7", "E11"],
    "v4": ["L1b", "E10", "L1c", "E8", "E7", "E11", "E9b", "E9a"],
    "v5": ["L1b", "E10", "L1c", "E8", "E7", "E11", "E9b", "E9a", "E59E60"],
}
_applied = {"version": None}


def expected_counts(version):
    """Per single forward, from this module's census constants (the 40B's by default: 14 HCS + 14 HCM + 14 HCL + 8 attention, 101 RMSNorm calls;
    routes.configure sets them from the constructed model)."""
    lv = set(VERSIONS[version]); c = {}
    if "E10" in lv: c["hcl_cached_iir"] = N_HCL
    if "E9a" in lv: c["hcl_fused_v2"] = N_HCL; c.pop("hcl_cached_iir", None)
    if "L1c" in lv: c["hcm_fftconv_cached"] = N_HCM
    if "E9b" in lv: c["hcm_fused_v2"] = N_HCM; c.pop("hcm_fftconv_cached", None)
    if "E7" in lv: c["fir3_triton_fwd"] = N_HCS + N_HCM + N_HCL; c["hcs7_triton_fwd"] = N_HCS
    elif "E8" in lv: c["fir_bf16_conv1d"] = N_HCS + N_HCM + N_HCL + N_HCS
    if "E11" in lv: c["rms_fused_tail"] = N_RMS
    if "E59E60" in lv: c["e60_r2c_nofill"] = N_HCL; c["e59_c2r_noclone"] = N_HCL + N_HCM
    return c


def _dist_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_conditions(model):
    """The torch-conv stock route (TE fp8 input projections, kernel flags off) at this module's census constants: refuses anything else by
    assertion; returns the resolved facts."""
    import triton
    import transformer_engine
    cfg = model.config
    assert HAS_TE and cfg.get("use_fp8_input_projections", False) is True, "kit v40 is built for the 40B TE route (fp8 input projections) only"
    assert int(cfg.hidden_size) == HIDDEN and int(cfg.num_layers) == N_BLOCKS, (cfg.hidden_size, cfg.num_layers)
    assert cfg.interleave is True, "the stock interleave runs in this kit (no fold)"
    assert cfg.get("column_split_hyena", True) is False and cfg.get("use_flashfft", False) is False and cfg.get("prefill_style") == "fft"
    assert not any(cfg.get(k, False) for k in ("use_hcs_kernel", "use_hcm_kernel", "use_hcl_kernel", "use_flash_rmsnorm", "use_flash_depthwise", "print_activations"))
    hb = [blk for blk in model.blocks if isinstance(blk, ParallelGatedConvBlock)]
    ab = [blk for blk in model.blocks if isinstance(blk, AttentionBlock)]
    assert len(hb) == N_HCS + N_HCM + N_HCL and len(ab) == N_ATTN and len(model.blocks) == N_BLOCKS, (len(hb), len(ab), len(model.blocks))
    kinds = {"hcs": 0, "hcm": 0, "hcl": 0}
    for b in hb:
        p = b.projections
        assert type(p).__name__ == "TELinear" and type(p).__module__ == "vortex.model.layers" and getattr(p, "use_fp8_input_projections", None) is True, \
            f"projection module {type(p).__module__}.{type(p).__name__}: the 40B route is TELinear under fp8 autocast"
        assert type(b.out_filter_dense) is torch.nn.Linear and type(b.pre_norm) is RMSNorm and type(b.post_norm) is RMSNorm
        assert not b.pre_norm.use_flash_rmsnorm and not b.post_norm.use_flash_rmsnorm and not b.print_activations
        f = b.filter
        assert f.short_filter_bias is None and f.fir_fn is torch.nn.functional.conv1d and f.short_filter_length == 3 and not f.column_split_hyena
        assert not f.hyena_flip_x1x2 and f.long_fir_threshold is None and not f.use_flashfft and f.fftconv_fn is None
        assert type(f.engine) is HyenaInferenceEngine and not (f.engine.use_hcs_kernel or f.engine.use_hcm_kernel or f.engine.use_hcl_kernel)
        if f.fir_inner_filter_length is None:
            kinds["hcl"] += 1; assert f.hyena_filter_groups == HIDDEN and f.h is None and f.D is not None
        elif f.fir_inner_filter_length >= 128:
            kinds["hcm"] += 1; assert f.fir_inner_filter_length == 128 and f.fir_inner_fn is torch.nn.functional.conv1d and f.D is not None
        else:
            kinds["hcs"] += 1; assert f.fir_inner_filter_length == 7 and f.fir_inner_fn is torch.nn.functional.conv1d and f.D is None
    assert kinds == {"hcs": N_HCS, "hcm": N_HCM, "hcl": N_HCL}, kinds
    assert type(model.norm) is RMSNorm and not model.norm.use_flash_rmsnorm
    for b in ab:
        assert type(b.pre_norm) is RMSNorm and type(b.post_norm) is RMSNorm and b.inner_mha_cls.Wqkv.bias is None
    wq = [b.inner_mha_cls.Wqkv.weight for b in ab]
    assert [tuple(w.stride()) for w in wq] == [WQKV_STRIDES] * N_ATTN, f"Wqkv layout: {[tuple(w.stride()) for w in wq]}"
    noncontig = {n for n, p in model.named_parameters() if not p.is_contiguous()}
    assert noncontig == {n for n, p in model.named_parameters() if any(p is w for w in wq)}, f"unexpected non-contiguous parameters: {sorted(noncontig)}"
    got = {"torch": torch.__version__.split("+")[0], "triton": triton.__version__, "vtx": _dist_version("vtx"), "evo2": _dist_version("evo2"),
           "transformer_engine": getattr(transformer_engine, "__version__", None)}
    devices = sorted({str(d) for d in model.block_idx_to_device.values()})
    return {"stack": got, "projections": "vortex.model.layers.TELinear", "blocks": {**kinds, "attn": len(ab)},
            "devices": devices, "layers_per_device": {d: sum(1 for v in model.block_idx_to_device.values() if str(v) == d) for d in devices}}


# ------------------------------------------------------------------------------------------------------- apply / remove
def apply(model, version="v5"):
    assert _applied["version"] is None, f"kit already applied ({_applied['version']})"
    assert K7._applied["version"] is None and K7.is_unpatched(), "the 7B kit module is patched in this process"
    info = check_conditions(model); lv = VERSIONS[version]
    HyenaCascade.compute_filter = K7._cached_compute_filter                                     # L1b
    HyenaInferenceEngine.parallel_iir = G["_parallel_iir_v2"] if "E9a" in lv else G["_parallel_iir_Hcache_padbuf"]   # E10 (+E9a)
    _eng.fftconv_func = K7._fftconv_cached                                                      # L1c
    if "E9b" in lv: HyenaInferenceEngine.parallel_fir = G["_parallel_fir_v2"]
    elif "E7" in lv: HyenaInferenceEngine.parallel_fir = G["_parallel_fir_fused"]
    elif "E8" in lv: HyenaInferenceEngine.parallel_fir = G["_parallel_fir_bf16"]
    if "E11" in lv:                                                                             # E65 and E66 ride with the norm kernel
        RMSNorm.forward = G["_rms_fused_forward"]; AttentionBlock.forward = G["_attn_forward"]; StripedHyena.stateless_forward = K7._stateless_forward_pairs
    if "E59E60" in lv:
        K7._CF = CuFFTPerDevice()                                                               # kit.base's chains resolve _cf() at call time
        HyenaInferenceEngine.parallel_iir = G["_parallel_iir_v5"]; HyenaInferenceEngine.parallel_fir = G["_parallel_fir_v5"]
    _applied["version"] = version; CTR.clear()
    info["version"] = version; info["levers"] = list(lv)
    return info


PROCESS_WIDE_PATCHES = ("HyenaCascade.compute_filter", "HyenaInferenceEngine.parallel_iir", "HyenaInferenceEngine.parallel_fir",
                        "vortex.model.engine.fftconv_func", "RMSNorm.forward")


def is_unpatched():
    return K7.is_unpatched() and _applied["version"] is None and K7._CF is None


