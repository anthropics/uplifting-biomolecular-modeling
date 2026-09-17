# SPDX-License-Identifier: Apache-2.0
"""EXACT-tier provider (numerics class EXACT-BY-CONSTRUCTION, proved bitwise): stock module signature, for the FPF ops registry
    FPF_OPS=trimul_out=protenix_fpf_trimul_tx.trimul_exact:fn,trimul_in=protenix_fpf_trimul_tx.trimul_exact:fn
or as fpf_smalln's EXACT provider.  Served cell: triangle_multiplicative == "cuequivariance", c_z == c_hidden == 256, CUDA bf16 z under bf16
autocast, ONE square pair plane ([N, N, C] or [1, N, N, C]; a leading batch > 1 takes the stock forward, counted), mask None | [N, N] | [1, N, N]
holding 0/1 values, N > 100 (cuEquivariance itself takes another function at N <= 100), device capability 9.0.  Outside the cell -> the stock forward
for that call (counted by reason, named once).  Cannot-run -> ops.TrimulTxUnavailable raised by name; that includes the load-time live check
(_livecheck: this path vs the installed stock forward on closed-form N = 160 and N = 162 tiles, once per process and direction, torch.equal or refuse).
Output == the stock module's output bit for bit: both LayerNorms run inside K1 / K3 with a summation order chosen so results are bitwise identical to the stock op for
supported shapes, the contraction is the
stock expression on unpadded planes (the stock cuBLAS problem), projection / gating / residual arithmetic in the stock order (1.2; 1.1 called the
stock LayerNorm kernels instead)."""
import atexit, json, sys
import torch
from . import ops

N_MIN_EXCLUSIVE = 100
LIVECHECK_N = (160, 162)      # both summation trees of the stock transposing LayerNorm (N % 4 == 0 | != 0) and both plane layouts (direct | packed)
COUNTS = {"provider": "protenix_fpf_trimul_tx.exact", "calls": 0, "served": 0, "fallback": {}, "first_call": None, "loaded": None, "livecheck": {}, "cannot_run": None, "kernel_error": None}
_SAID = set()


def _log(msg):
    sys.stderr.write("[protenix_fpf_trimul_tx.exact] %s\n" % msg); sys.stderr.flush()


def _fallback(reason, module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative):
    import fpf
    COUNTS["fallback"][reason] = COUNTS["fallback"].get(reason, 0) + 1
    if reason not in _SAID and len(_SAID) < 12:
        _SAID.add(reason); _log("stock forward for this call (reason: %s; N=%s c=%s)" % (reason, z.shape[-2] if z.dim() >= 2 else "?", z.shape[-1]))
    return fpf.original("trimul_out" if module._outgoing else "trimul_in")(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)


def _stock_forward(module):
    try:
        import fpf
        return fpf.original("trimul_out" if module._outgoing else "trimul_in")
    except ImportError:
        return type(module).forward


def _livecheck(module, w, inplace_safe, _add_with_inplace, _inplace_chunk_size):
    """Once per process and direction, before the first served call: this exact path vs the INSTALLED stock forward (the live cuEquivariance
    kernels and their tile configuration in this very process) on the closed-form N = 160 tile with the caller's own weights, mask with zeros,
    the caller's residual arguments.  torch.equal or the mode refuses by name (a cuEquivariance change that moves the stock bits is caught here,
    in-process, not downstream)."""
    key = "out" if module._outgoing else "in"
    if key in COUNTS["livecheck"]:
        return
    dev = next(module.parameters()).device
    residual = bool(inplace_safe is True and _add_with_inplace)
    for n in LIVECHECK_N:
        z, mask, _ = ops._closed_form_inputs(n, dev)
        with torch.no_grad():
            ref = _stock_forward(module)(module, z.reshape(1, n, n, ops.C).clone(), mask.reshape(1, n, n).clone(), inplace_safe, _add_with_inplace, _inplace_chunk_size, "cuequivariance")
            out = ops.trimul_exact(z, bool(module._outgoing), mask, w, residual=residual)
        torch.cuda.synchronize(dev)
        ref = ref.reshape(n, n, ops.C)
        if not torch.equal(out, ref):
            d = (out.float() - ref.float()).abs()
            COUNTS["livecheck"][key] = "NE@%d" % n
            reason = "livecheck: exact path != the installed cuEquivariance TriMul on the N=%d closed-form tile (%s: %d of %d elements differ, max |d| %.3g) — a cuEquivariance version / tile-configuration change moved the stock bits" % (
                n, key, int((out != ref).sum()), out.numel(), d.max().item())
            COUNTS["cannot_run"] = reason; _log("CANNOT RUN (raised; the mode refuses by this name): %s" % reason)
            raise ops.TrimulTxUnavailable(reason)
    COUNTS["livecheck"][key] = "eq"
    _log("livecheck=eq (%s, N=%s closed-form tiles, this process's installed cuEquivariance TriMul, residual=%s)" % (key, "/".join(str(n) for n in LIVECHECK_N), residual))


def _weights(module):
    cache = getattr(module, "_fpf_cache", None)
    if cache is None:
        cache = module._fpf_cache = {}
    key = ("trimul_tx", torch.bfloat16)
    if key not in cache:
        cache[key] = ops.pack_weights(module)
    return cache[key]


def fn(module, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch"):
    COUNTS["calls"] += 1
    fb = lambda why: _fallback(why, module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)
    if triangle_multiplicative != "cuequivariance":
        return fb("backend!=cuequivariance")
    if int(getattr(module, "c_z", 0)) != ops.C or int(getattr(module, "c_hidden", 0)) != ops.C or z.shape[-1] != ops.C:
        return fb("c!=256")
    if not z.is_cuda or z.dim() not in (3, 4) or z.shape[-2] != z.shape[-3] or (z.dim() == 4 and z.shape[0] != 1):
        return fb("shape/batch")
    N = int(z.shape[-2])
    if N <= N_MIN_EXCLUSIVE:
        return fb("N<=%d" % N_MIN_EXCLUSIVE)
    if not (torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.bfloat16) or z.dtype != torch.bfloat16:
        return fb("not-bf16-autocast")
    if mask is not None and (tuple(mask.shape[-2:]) != (N, N) or mask.numel() != N * N):
        return fb("mask-shape")
    if torch.cuda.get_device_capability(z.device) != (9, 0):
        return fb("cc!=9.0")
    if COUNTS["cannot_run"]:
        raise ops.TrimulTxUnavailable(COUNTS["cannot_run"])
    try:
        ops.load()
    except ops.TrimulTxUnavailable as e:
        COUNTS["cannot_run"] = e.reason; _log("CANNOT RUN (raised; the mode refuses by this name): %s" % e.reason); raise
    if COUNTS["loaded"] is None:
        COUNTS["loaded"] = ops._STATE["loaded"]; _log("loaded %s loadcheck=%s" % (json.dumps(ops._STATE["loaded"]), "ok" if ops._STATE["loadcheck"] else "-"))
    w = _weights(module)
    _livecheck(module, w, inplace_safe, _add_with_inplace, _inplace_chunk_size)
    residual = bool(inplace_safe is True and _add_with_inplace)
    zz = z.reshape(N, N, ops.C)
    if not zz.is_contiguous():
        zz = zz.contiguous()
    mm = None
    if mask is not None:
        mm = mask.reshape(N, N)
        if mm.dtype != torch.float32:
            mm = mm.to(torch.float32)
        mm = mm.contiguous()
    try:
        out = ops.trimul_exact(zz, bool(module._outgoing), mm, w, residual=residual)
    except ops.TrimulTxUnavailable:
        raise
    except Exception as e:
        from opt_core.oom import is_oom
        if is_oom(e):
            raise
        COUNTS["kernel_error"] = repr(e)[:300]; _log("KERNEL ERROR (raised): %r" % (e,)); raise
    out = out.reshape(z.shape)
    COUNTS["served"] += 1
    if COUNTS["first_call"] is None:
        COUNTS["first_call"] = {"N": N, "dir": "out" if module._outgoing else "in", "residual": residual, "mask": mask is not None, "device": torch.cuda.get_device_name(z.device)}
        _log("FIRST CALL served: %s %s" % (json.dumps(COUNTS["first_call"]), ops.describe()))
    return out


def _report():
    _log("COUNTS " + json.dumps(COUNTS, default=str))
    try:
        import ptx_trunk2_levers as LEV
        LEV._STATS["protenix_fpf_trimul_tx.exact"] = dict(COUNTS)
    except Exception:
        pass
atexit.register(_report)
