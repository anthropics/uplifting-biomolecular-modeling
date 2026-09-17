"""fpf_trimul.engines — per-engine entry points for the fpf_engines adapter.
One callable per (engine, op) with EXACTLY the stock signature; returns the stock return value; raises fpf_engines.FPFFallback(reason)
on unsupported cases so the stock path runs (counted, never silent).

  FPF_ENGINE=boltz2  FPF_OPS=trimul_out,trimul_in FPF_IMPL=trimul_out=fpf_trimul.engines:boltz2_trimul_out,trimul_in=fpf_trimul.engines:boltz2_trimul_in
  FPF_ENGINE=opendde FPF_OPS=trimul_out,trimul_in FPF_IMPL=trimul_out=fpf_trimul.engines:opendde_trimul_out,trimul_in=fpf_trimul.engines:opendde_trimul_in

engine_stock_class reproduced:
  boltz2  : "Boltz-2 2.2.1 cuEq TriMul {outgoing|incoming} (kernels-ON)" = cuEq.triangle_multiplicative_update(x, dir, mask, norm_in, p_in[2D,D], g_in, norm_out, p_out, g_out, eps=1e-5)
            under bf16-mixed autocast (TF32 off); D = token_z = 128; caller adds the residual.  use_kernels=False (torch path) -> FPFFallback("torch_path_class").
  opendde : "OpenDDE 1.0.0 cuEq TriMul {outgoing|incoming} (tuned-cache path)" = z_in = z.clone(); y = cuEq(...)(z[None])[0]; return y + z_in
            fp32 activations and weights, TF32 ON -> our GEMMs run tf32 (RN-converted operands) exactly as cuEq's gated GEMM does under allow_tf32;
            contraction operands fp32 with tf32 MMA (cuBLAS-in-class).  triangle_multiplicative != 'cuequivariance' -> FPFFallback("torch_path_class").
Precision class is templated on the operand dtype (timing run rows print the resolved PREC per call): no bf16 shortcut is ever taken for OpenDDE.
"""
import os, torch
from . import kernels as K
from .trimul import _select_cfg, CONTRACT, config_table_sha

try:
    from fpf_engines import FPFFallback
except Exception:  # adapter not on the path (op-level timing run): a plain exception with the same shape
    class FPFFallback(Exception):
        def __init__(self, reason): super().__init__(reason); self.reason = reason

STATS = {"calls": 0}
from .trimul import FPF_META as _META
FPF_META = {"boltz2_trimul_out": dict(_META["trimul_out"], engine="Boltz-2 2.2.1 kernels-ON (cuEq 0.10.0 class; bf16 GEMMs under autocast, z fp32 in, bf16 update out)"),
            "boltz2_trimul_in": dict(_META["trimul_in"], engine="Boltz-2 2.2.1 kernels-ON"),
            "opendde_trimul_out": dict(_META["trimul_out"], engine="OpenDDE 1.0.0 cuEq tuned-cache class (fp32 z/weights, TF32 MMA with RN-converted operands)", inplace="returns z + update as a new tensor (not written into z)"),
            "opendde_trimul_in": dict(_META["trimul_in"], engine="OpenDDE 1.0.0 cuEq tuned-cache class", inplace="returns z + update as a new tensor (not written into z)")}


def _cache(module, wdt, names):
    cache = getattr(module, "_fpf_cache", None)
    if cache is None:
        cache = module._fpf_cache = {}
    key = ("w", wdt)
    if key not in cache:
        ln_in, p_in, g_in, ln_out, p_out, g_out = names
        def c(t): return t.detach().to(wdt).contiguous()     # GEMM weights: cast as cuEq does under autocast (maybe_to(autocast_dtype)); no-op for fp32 engines
        def n(t): return t.detach().contiguous()             # LN affine params native (cuEq LN kernel loads them as fp32)
        cache[key] = K.pack_weights(n(ln_in.weight), n(ln_in.bias), c(p_in), c(g_in), n(ln_out.weight), n(ln_out.bias), c(p_out.weight), c(g_out.weight))
    return cache[key]


def _compute_dtype(weight_dtype):
    return torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else weight_dtype


def _run(module, x, mask, direction, w, cdt, residual):
    """x: [B,N,N,D] or [N,N,D]; mask: [B,N,N] / [N,N] / None.  Returns the stock-shaped output.  Cell decision = trimul.cell_verdict: a MEASURED_OFF class or a dtype the
    kernels do not compute is the engine's stock TriMul BY NAME (FPFFallback 'cell:<cls>_C<C>+off(not-measured)' / 'dtype:<cls>'); an UNKNOWN bf16 / fp32 class is served
    on the default tiles, named once ('unverified(<cls>_C<C>)')."""
    from .trimul import cell_verdict, _note_unverified
    kind, word = cell_verdict(cdt, x.shape[-1])
    if kind in ("off", "unsupported"): raise FPFFallback(word)
    if kind == "unverified": _note_unverified(word)
    squeeze = x.dim() == 3
    xs = x if not squeeze else x[None]
    ms = None if mask is None else (mask if mask.dim() == 3 else mask[None])
    outs = []
    for b in range(xs.shape[0]):
        xb = xs[b]
        if not xb.is_contiguous(): xb = xb.contiguous()
        N = xb.shape[0]
        if N <= 100: raise FPFFallback("N<=100_cueq_torch_fallback_regime")
        mb = None if ms is None else ms[b]
        outs.append(K.trimul_forward(xb, direction, mb, w, eps=1e-5, residual=residual, cfg=_select_cfg(cdt, N, xb.shape[-1]), contract=CONTRACT, cdt=cdt))
    out = outs[0] if squeeze else torch.stack(outs, 0)
    STATS["calls"] += 1
    return out


# ------------------------------------------------------------------------------------------------------------ Boltz-2 2.2.1
def _boltz2(module, x, mask, use_kernels, direction):
    if not use_kernels: raise FPFFallback("torch_path_class")      # a different function class (bf16 einsum in fp32): not ours
    if not x.is_cuda or x.dim() != 4: raise FPFFallback("not_cuda_or_rank")
    if x.shape[-1] % 64 != 0: raise FPFFallback(f"D_{x.shape[-1]}_not_multiple_of_64")
    cdt = _compute_dtype(module.p_in.weight.dtype)
    if cdt not in (torch.bfloat16, torch.float16, torch.float32): raise FPFFallback(f"dtype_{cdt}")
    w = _cache(module, cdt, (module.norm_in, module.p_in.weight, module.g_in.weight, module.norm_out, module.p_out, module.g_out))
    return _run(module, x, mask, direction, w, cdt, residual=False)


def boltz2_trimul_out(module, x, mask, use_kernels=False):
    return _boltz2(module, x, mask, use_kernels, "outgoing")


def boltz2_trimul_in(module, x, mask, use_kernels=False):
    return _boltz2(module, x, mask, use_kernels, "incoming")


# ------------------------------------------------------------------------------------------------------------ OpenDDE 1.0.0 / Protenix-style
def _ptxlike(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative, direction):
    if triangle_multiplicative != "cuequivariance" or module.c_z != module.c_hidden: raise FPFFallback("torch_path_class")
    if not z.is_cuda or z.dim() not in (3, 4): raise FPFFallback("not_cuda_or_rank")
    if z.shape[-1] % 64 != 0: raise FPFFallback(f"C_{z.shape[-1]}_not_multiple_of_64")
    cdt = _compute_dtype(module.linear_a_p.weight.dtype)
    if cdt not in (torch.bfloat16, torch.float16, torch.float32): raise FPFFallback(f"dtype_{cdt}")
    w = _cache(module, cdt, (module.layer_norm_in, torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], 0),
                             torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], 0), module.layer_norm_out, module.linear_z, module.linear_g))
    add = bool(inplace_safe is True and _add_with_inplace)
    return _run(module, z, mask, direction, w, cdt, residual=add)


# ENGINE DEFAULTS: Protenix = EXACT (x1.12-1.17 vs the tuned-cache stock, bit-exact under the tuned cache), Boltz-2 = EXACT (bit-exact in-engine, never slower),
# OpenDDE = EXACT: the default rule ('a default must not be
# slower than the stock path and must be inside the in-model floor') is met by the measured rows — op level bit-exact 24/24 + 12/12 at
# x1.03-1.07 (never slower at 185/436/705), in-model 20/20 inside the same-host drift control (max |d ipSAE_min| 2.6e-4 < the run-to-run band 5.8e-4), design wall
# -2.1 % @705 / -2.5 % @703.  FAST (TIER2) stays opt-in (FPF_TRIMUL_OPENDDE=fast); FPF_TRIMUL_OPENDDE=stock keeps the engine's stock TriMul (counted FPFFallback).
from .trimul import _MODE
OPENDDE_SERVE = os.environ.get("FPF_TRIMUL_OPENDDE", "exact")
ENGINE_DEFAULTS = {"protenix": "exact", "boltz2": "exact", "opendde": "exact (v3.2 default; opt-in FPF_TRIMUL_OPENDDE=fast|stock)"}


def opendde_trimul_out(module, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch"):
    if OPENDDE_SERVE not in ("exact", "fast"): raise FPFFallback(f"opendde_serve_{OPENDDE_SERVE} (FPF_TRIMUL_OPENDDE=stock|other -> stock path; v3.2 default is exact)")
    if OPENDDE_SERVE != _MODE: raise FPFFallback(f"opendde_opt_in_mode_{OPENDDE_SERVE}_but_package_mode_{_MODE} (set FPF_TRIMUL_MODE={OPENDDE_SERVE})")
    return _ptxlike(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative, "outgoing")


def opendde_trimul_in(module, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch"):
    if OPENDDE_SERVE not in ("exact", "fast"): raise FPFFallback(f"opendde_serve_{OPENDDE_SERVE} (FPF_TRIMUL_OPENDDE=stock|other -> stock path; v3.2 default is exact)")
    if OPENDDE_SERVE != _MODE: raise FPFFallback(f"opendde_opt_in_mode_{OPENDDE_SERVE}_but_package_mode_{_MODE} (set FPF_TRIMUL_MODE={OPENDDE_SERVE})")
    return _ptxlike(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative, "incoming")


# Protenix v2 (FPF_SPEC_v0 registry, same class as OpenDDE's forward): re-exported for symmetry
protenix_trimul_out = opendde_trimul_out
protenix_trimul_in = opendde_trimul_in
