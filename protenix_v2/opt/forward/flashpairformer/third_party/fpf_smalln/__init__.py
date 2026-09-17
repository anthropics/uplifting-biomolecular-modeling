# Copyright 2026 Anthropic, PBC
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 . Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""fpf_smalln — small-N routing for FlashPairformer ARM T (Tier-2) on Protenix v2 (lives at third_party/fpf_smalln/ under the FlashPairformer home).

What it does (default OFF; switched on by FPF_SMALLN=1 in the ARM T environment):
  * TriMul (registry ops trimul_out / trimul_in, FPF_OPS=...=fpf_smalln:trimul_c256):
        N_token <  FPF_SMALLN_TRIMUL_EXACT_BELOW (default 100000 = always)  -> partner TriMul EXACT mode (vendored fpf_trimul v3 pkg as fpf_smalln._tmx:
                                                                               stock cuEq LN kernels + dual gated GEMMs + unpadded cuBLAS bmm; bitwise == stock module on real dumps)
        N_token >= threshold                                                 -> bundle FAST path (ptx_trunk2_levers.trimul_partner_c256 = third_party/fpf_trimul v1, FPF_TRIMUL_CONTRACT=triton)
        c_z != 256 / unsupported                                             -> stock forward (unchanged guard of the bundle wrapper)
  * Tri-attention inside the BLK2 block core (PTX_BLK_ATT=k2b):
        N_token <  FPF_SMALLN_K2B_MIN_TOKENS (default 0 = never gate)         -> stock cuEq triangle attention (= ARM E block core, EXACT)
        N_token >= threshold                                                 -> K2B flash tri-attention (unchanged, Tier-2)
    The gate is installed by wrapping ptx_trunk2_levers._BLK_ATT['fn'] AFTER apply_from_env() selected K2B (install() is idempotent; the levers'
    sitecustomize calls fpf.enable_from_env() after apply_from_env(), and importing this module from FPF_OPS runs install()).
Checked dispatch cell (probe at first use on the live device, printed once): GPU cc 9.0 (H100 class), bf16 autocast, c_z == c_hidden == 256, z.dim()==3.
Anything else -> the pre-existing path for that call (never a served-but-unchecked cell).  Counters printed at exit as one line
'[fpf_smalln] COUNTS {...}' and merged into the levers' PTX_LEVER_REPORT line under key 'fpf_smalln'.
"""
import os, sys, atexit, json

__version__ = "0.1.1"
ENABLED = os.environ.get("FPF_SMALLN", "0") not in ("", "0")
TRIMUL_EXACT_BELOW = int(os.environ.get("FPF_SMALLN_TRIMUL_EXACT_BELOW", "100000"))   # N <  this -> EXACT partner TriMul ; N >= -> FAST triton (bundle v1)
# ---- integrator v0.3.1: ARCH-KEYED TriMul crossover (NX-Blackwell t2opB200: on sm_100 the triton TriMul is x0.94-0.97 of stock cuEq at 356 tok and only x1.07-1.11 at 546,
#      while K2B is x2.5 over cuDNN at all sizes) -> the ARM T gate is per-OP per-ARCH: the K2B threshold stays FPF_SMALLN_K2B_MIN_TOKENS; the TriMul exact->fast crossover
#      resolves lazily per arch: env FPF_SMALLN_TRIMUL_EXACT_BELOW_<ARCH> (e.g. _SM100=576) > CELLS.json "smalln_trimul_exact_below" {"sm100": 576} > FPF_SMALLN_TRIMUL_EXACT_BELOW.
_TRIMUL_THR_CACHE = {}
def _trimul_exact_below(device):
    key = str(device)
    if key in _TRIMUL_THR_CACHE:
        return _TRIMUL_THR_CACHE[key]
    thr, src, arch = TRIMUL_EXACT_BELOW, "FPF_SMALLN_TRIMUL_EXACT_BELOW", "?"
    try:
        import ptx_trunk2_levers as LEV
        arch = LEV.gpu_arch(device)
        env_k = "FPF_SMALLN_TRIMUL_EXACT_BELOW_" + arch.upper()
        cells = (getattr(LEV, "_CELLS", None) or {}).get("smalln_trimul_exact_below") or {}
        if os.environ.get(env_k, ""):
            thr, src = int(os.environ[env_k]), env_k
        elif arch in cells:
            thr, src = int(cells[arch]), "CELLS.json smalln_trimul_exact_below[%s]" % arch
    except Exception as e:
        src += " (arch lookup failed: %r)" % (e,)
    _TRIMUL_THR_CACHE[key] = thr
    COUNTS["trimul_exact_below_effective"] = {"arch": arch, "thr": thr, "source": src}
    _log("TriMul crossover on %s (gpu_arch=%s): EXACT provider below %d tok, FAST triton at/above  [%s]" % (key, arch, thr, src))
    return thr
K2B_MIN_TOKENS = int(os.environ.get("FPF_SMALLN_K2B_MIN_TOKENS", "0"))                  # N <  this -> cuEq attention (exact) ; N >= -> K2B
EXACT_FN_SPEC = os.environ.get("FPF_SMALLN_TRIMUL_EXACT_FN", "fpf_smalln._tmx.trimul:fn")   # provider of the EXACT TriMul (module:attr, stock signature); default = vendored partner v3 exact pkg
FAST_FN_SPEC = os.environ.get("FPF_SMALLN_TRIMUL_FAST_FN", "ptx_trunk2_levers:trimul_partner_c256")   # provider at/above the threshold (bundle FAST wrapper)
COUNTS = {"version": __version__, "enabled": ENABLED, "trimul_exact_below": TRIMUL_EXACT_BELOW, "k2b_min_tokens": K2B_MIN_TOKENS, "exact_fn": EXACT_FN_SPEC, "fast_fn": FAST_FN_SPEC,
          "trimul_exact_calls": 0, "trimul_fast_calls": 0, "trimul_stock_calls": 0, "att_cueq_calls": 0, "att_k2b_calls": 0,
          "probe": None, "probes": 0, "probe_fail": 0, "gate_installed": False, "cell_refusals": 0}
_PROBED = {}          # (direction, N) -> bool : live bitwise probe result per distinct token count (cuBLAS picks kernels per N; census 231b2be2 §3)
_INSTALLED = False


def _log(msg):
    sys.stderr.write(f"[fpf_smalln] {msg}\n"); sys.stderr.flush()


_FN_CACHE = {}
def _resolve(spec):
    """'module.sub:attr' -> callable (cached)."""
    if spec not in _FN_CACHE:
        import importlib
        modpath, attr = spec.split(":")
        _FN_CACHE[spec] = getattr(importlib.import_module(modpath.strip()), attr.strip())
    return _FN_CACHE[spec]


def _cell_ok(module, z):
    """checked dispatch cell: H100-class (sm90), bf16 z under bf16 autocast, c_z == c_hidden == 256, single [N,N,C] plane."""
    import torch
    try:
        if not z.is_cuda or z.dtype != torch.bfloat16 or z.dim() != 3 or z.shape[-1] != 256: return False
        if int(getattr(module, "c_z", 0)) != 256 or int(getattr(module, "c_hidden", 0)) != 256: return False
        if torch.cuda.get_device_capability(z.device)[0] != 9: return False
        if not torch.is_autocast_enabled() or torch.get_autocast_dtype("cuda") != torch.bfloat16: return False
        return True
    except Exception:
        return False


def _probe_trimul(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative):
    """One-time on-device bitwise probe of the EXACT TriMul against the stock forward on the live call (same module, same z). Result cached; on mismatch
    the EXACT route is disabled for the process (falls back to the stock forward below the threshold — still exact, just not faster)."""
    import torch, fpf
    _exact_fn = _resolve(EXACT_FN_SPEC)
    stock = fpf.original("trimul_out" if module._outgoing else "trimul_in")
    with torch.no_grad():
        ref = stock(module, z.clone(), mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)
        out = _exact_fn(module, z.clone(), mask=mask, inplace_safe=inplace_safe, _add_with_inplace=_add_with_inplace, _inplace_chunk_size=_inplace_chunk_size, triangle_multiplicative=triangle_multiplicative)
    ok = bool(torch.equal(out, ref))
    rec = {"op": "trimul_out" if module._outgoing else "trimul_in", "N": int(z.shape[-2]), "bitwise": ok,
           "max_abs_diff": float((out.float() - ref.float()).abs().max()), "device": torch.cuda.get_device_name(z.device)}
    COUNTS["probes"] += 1; COUNTS["probe_fail"] += (not ok)
    if COUNTS["probe"] is None or not ok: COUNTS["probe"] = rec          # first probe (or the first failing one) is kept in the exit line
    if COUNTS["probes"] <= 2 or not ok or os.environ.get("FPF_SMALLN_VERBOSE"):
        _log(f"PROBE exact-trimul vs stock on live call: {rec}")
    return ok, ref


def trimul_c256(module, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch"):
    """FPF_OPS entry for trimul_out / trimul_in (stock signature). Size-routed: EXACT partner TriMul below FPF_SMALLN_TRIMUL_EXACT_BELOW tokens, bundle FAST
    triton path at/above it; anything outside the checked cell -> the bundle wrapper (which itself falls back to stock outside c_z==256)."""
    import fpf
    N = int(z.shape[-2]) if z.dim() >= 3 else 0
    if not ENABLED or triangle_multiplicative != "cuequivariance" or N >= (_trimul_exact_below(z.device) if z.is_cuda else TRIMUL_EXACT_BELOW):   # v0.3.1: arch-keyed crossover
        COUNTS["trimul_fast_calls"] += 1
        return _resolve(FAST_FN_SPEC)(module, z, mask=mask, inplace_safe=inplace_safe, _add_with_inplace=_add_with_inplace, _inplace_chunk_size=_inplace_chunk_size, triangle_multiplicative=triangle_multiplicative)
    if not _cell_ok(module, z) or N <= 100:
        COUNTS["cell_refusals"] += 1; COUNTS["trimul_stock_calls"] += 1
        return fpf.original("trimul_out" if module._outgoing else "trimul_in")(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)
    pk = (bool(module._outgoing), N)
    if pk not in _PROBED:                    # first call of this (direction, N) in the process: bitwise probe on the live activation
        ok, ref = _probe_trimul(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)
        _PROBED[pk] = ok
        COUNTS["trimul_exact_calls" if ok else "trimul_stock_calls"] += 1
        return ref                           # the probe computed the stock result on this very call; return it (bitwise equal to the exact output when ok)
    if not _PROBED[pk]:
        COUNTS["trimul_stock_calls"] += 1
        return fpf.original("trimul_out" if module._outgoing else "trimul_in")(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)
    COUNTS["trimul_exact_calls"] += 1
    return _resolve(EXACT_FN_SPEC)(module, z, mask=mask, inplace_safe=inplace_safe, _add_with_inplace=_add_with_inplace, _inplace_chunk_size=_inplace_chunk_size, triangle_multiplicative=triangle_multiplicative)


def _gate_attention():
    """Wrap the BLK2 block-core attention selector: below K2B_MIN_TOKENS use cuEq (exact arm), else the kernel apply_from_env() selected (K2B)."""
    try:
        import ptx_trunk2_levers as LEV
        import protenix.model.triangular.layers as TL
    except Exception as e:
        _log(f"attention gate NOT installed ({e!r})"); return False
    inner = LEV._BLK_ATT.get("fn")
    if inner is None:
        _log("attention gate: PTX_BLK_ATT not active (cuEq already) -> nothing to gate"); return True
    if getattr(inner, "_fpf_smalln_gate", False):
        return True
    thr = K2B_MIN_TOKENS
    def gated(q5, k5, v5, bias, mask=None, scale=None):
        # layout (cuEq triangle_attention): q5 [B, N, H, S, D]; token count = S (= N for the pairformer)
        n_tok = int(q5.shape[-2])
        if n_tok < thr:
            COUNTS["att_cueq_calls"] += 1
            q, k, v = (t[0] if t.dim() == 5 and t.shape[0] == 1 else t for t in (q5, k5, v5))
            b = bias[0] if bias.dim() == 5 and bias.shape[0] == 1 else bias
            # identical call to the ARM E branch of _triatt_block_pro_epi: TL.cuequivariance_triangular_attn(q, k, v, bias.unsqueeze(0), None, scale) with q [I,H,J,D]? ->
            # E branch passes q,k,v exactly as the prologue returned them (4-D) and bias.unsqueeze(0); the T branch unsqueezes q,k,v to 5-D and bias to 4-D if 3-D.
            # We therefore undo the T-branch reshapes and issue the E-branch call.
            _msk = None
            try:                                                                    # v0.3.2 (integrator): shared sm100f exactness guard from ptx_trunk2_levers
                import ptx_trunk2_levers as _LEVg
                if _LEVg._cueq_sm100f_exposed(k.shape[-2]): _msk = _LEVg._stock_true_mask(q, q.shape[-4], k.shape[-2])
            except Exception:
                _msk = None
            return TL.cuequivariance_triangular_attn(q, k, v, b if b.dim() == 4 else b.unsqueeze(0), _msk, scale)
        COUNTS["att_k2b_calls"] += 1
        return inner(q5, k5, v5, bias, mask=mask, scale=scale)
    gated._fpf_smalln_gate = True
    LEV._BLK_ATT["fn"] = gated
    COUNTS["gate_installed"] = True
    _log(f"attention gate installed: cuEq (exact) below {thr} tokens, {getattr(inner, '__module__', '?')}.{getattr(inner, '__name__', '?')} at/above")
    return True


def install():
    global _INSTALLED
    if _INSTALLED: return COUNTS
    _INSTALLED = True
    if not ENABLED:
        _log("FPF_SMALLN not set -> inactive (trimul_c256 forwards to the bundle FAST wrapper; no attention gate)"); return COUNTS
    if K2B_MIN_TOKENS > 0:
        _gate_attention()
    _log(f"ACTIVE v{__version__}: TriMul EXACT below {TRIMUL_EXACT_BELOW} tok (FAST triton at/above); K2B gated to cuEq below {K2B_MIN_TOKENS} tok")
    def _report():
        try:
            import ptx_trunk2_levers as LEV
            LEV._STATS["fpf_smalln"] = dict(COUNTS)
            m = sys.modules.get("fpf_smalln._tmx.trimul")
            if m is not None: COUNTS["tmx_stats"] = dict(getattr(m, "STATS", {}))
        except Exception:
            pass
        _log("COUNTS " + json.dumps(COUNTS, default=str))
    atexit.register(_report)
    return COUNTS


install()
