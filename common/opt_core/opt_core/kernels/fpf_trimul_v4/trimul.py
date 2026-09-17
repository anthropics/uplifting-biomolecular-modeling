"""fpf_trimul_v4.trimul:fn — FPF_SPEC_v0 registry entry / fpf_smalln FAST_FN provider (the stock TriangleMultiplication forward signature).
v4.1.0: thin wrapper over fpf_trimul_v4.generic.trimul_packed (module-agnostic entry); outputs byte-identical to v4.0.4r1 (TEST RECORD §5).
Served cell: CUDA, z [N,N,256] (or [B,N,N,256]) bf16 under bf16 autocast, module.c_z == module.c_hidden == 256, triangle_multiplicative == 'cuequivariance',
N >= N_MIN (no upper size limit), a compute capability the package's cells table or its SAFE cell serves (cells.py). Everything else -> fpf.original('trimul_out'|'trimul_in') (the stock forward),
counted by reason (`no-cell`: neither a row nor a SAFE cell for this capability). A lever that CANNOT RUN — its SAFE cell failed to build (`none:<why>`), the warm numerics probe refused the
shape class (`probe-failed`), a kernel error — RAISES (TrimulUnsupported with .cannot_run / the kernel's own exception): the caller's mode refuses by name; nothing continues under its name on the stock forward;
an out-of-memory is never rerouted — it propagates as torch's OutOfMemoryError (the stock TriMul needs more memory than this cell, not less).
In-place contract: with inplace_safe=True and _add_with_inplace=True the stock cuEq branch returns a NEW tensor z_new = update + z_in (it does not mutate z); we do the same
(the residual is fused in the epilogue). Without the flags the update alone is returned (bf16), like stock."""
import os, sys, json, atexit, torch
from . import __version__
from . import kernels as K
from . import cells as CELLS
from . import generic as G

__all__ = ["fn", "COUNTS", "cells_sha"]
N_MIN = int(os.environ.get("FPF_TRIMUL_V4_NMIN", "101"))          # cuEq itself takes its torch path at N <= 100 (different function) -> stock there
_CFG_OVERRIDE = CELLS._CFG_OVERRIDE               # engineering only (printed, marks COUNTS['engineering']=True)
_STOCK_ROUND = os.environ.get("FPF_TRIMUL_V4_STOCK_ROUND", "1") != "0"
COUNTS = {"version": __version__, "calls": 0, "served": 0, "fallback": {}, "cell": None, "first_call": None, "engineering": bool(_CFG_OVERRIDE), "stock_round": _STOCK_ROUND}
_SAID = set()


def _log(msg):
    sys.stderr.write("[fpf_trimul_v4] %s\n" % msg); sys.stderr.flush()


def cells_sha():
    return CELLS.cells_sha()


def _cell_for(device):
    cfg = CELLS.cell_for(device)
    COUNTS["cell"] = CELLS.INFO.get(str(device))
    return cfg


def _fallback(reason, module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative):
    import fpf
    COUNTS["fallback"][reason] = COUNTS["fallback"].get(reason, 0) + 1
    if reason not in _SAID and len(_SAID) < 12:
        _SAID.add(reason); _log("fallback -> stock forward (reason: %s; N=%s c=%s)" % (reason, z.shape[-2] if z.dim() >= 2 else "?", z.shape[-1]))
    return fpf.original("trimul_out" if module._outgoing else "trimul_in")(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)


def _weights(module, cdt):
    cache = getattr(module, "_fpf_cache", None)
    if cache is None:
        cache = module._fpf_cache = {}
    key = ("trimul_v4", cdt)
    if key not in cache:
        cache[key] = K.pack_weights(module, cdt)
    return cache[key]


def fn(module, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch"):
    COUNTS["calls"] += 1
    fb = lambda why: _fallback(why, module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)
    if triangle_multiplicative != "cuequivariance":
        return fb("backend!=cuequivariance")
    if int(getattr(module, "c_z", 0)) != 256 or int(getattr(module, "c_hidden", 0)) != 256 or z.shape[-1] != 256:
        return fb("c!=256")
    if not z.is_cuda or z.dim() not in (3, 4) or z.shape[-2] != z.shape[-3]:
        return fb("shape/device")
    N = int(z.shape[-2])
    if N < N_MIN:
        return fb("N<%d" % N_MIN)
    if not (torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.bfloat16) or z.dtype != torch.bfloat16:
        return fb("not-bf16-autocast")
    if mask is not None and tuple(mask.shape[-2:]) != (N, N):
        return fb("mask-shape")
    cfg = _cell_for(z.device)
    if cfg is None:
        if CELLS.off_word():                                 # the lever went off by name earlier in this process: it cannot run — raised, never a stock route
            COUNTS["cannot_run"] = CELLS.off_word()
            raise G.TrimulUnsupported(CELLS.off_word(), CELLS.INFO.get(str(z.device), {}).get("source", ""))
        return fb("no-cell")                                # neither a row nor a SAFE cell for this capability
    w = _weights(module, torch.bfloat16)
    residual = bool(inplace_safe is True and _add_with_inplace)
    if mask is not None and mask.dim() == 3 and z.dim() == 3 and mask.shape[0] != 1:
        return fb("mask-shape")
    try:                                                     # the module-signature provider is a thin wrapper over the module-agnostic entry (same kernels, same flags)
        out = G.trimul_packed(z, mask, outgoing=bool(module._outgoing), weights=w, residual=residual, eps=1e-5, stock_round=_STOCK_ROUND)
    except G.TrimulUnsupported as e:
        if e.cannot_run:                                     # the lever cannot run in this process (none:<why> / probe-failed): the mode refuses by that name — no stock route here
            COUNTS["cannot_run"] = e.reason
            raise
        return fb("generic:" + e.reason)                    # a structural refusal (an exotic mask layout): the stock forward for this call, counted
    except Exception as e:                                   # a kernel error: named once and raised — the mode refuses; nothing continues under its name on the stock forward
        from opt_core.oom import is_oom
        if is_oom(e):
            raise
        COUNTS["kernel_error"] = repr(e)[:300]
        _log("KERNEL ERROR (raised; the caller's mode refuses by name): %r" % (e,))
        raise
    COUNTS["served"] += 1
    if COUNTS["first_call"] is None:
        COUNTS["first_call"] = {"N": N, "dir": "out" if module._outgoing else "in", "residual": residual, "mask": mask is not None, "device": torch.cuda.get_device_name(z.device)}
        _log("FIRST CALL served: %s %s cfg=%s cells_sha=%s stock_round=%s" % (COUNTS["first_call"], CELLS.cell_word(z.device), json.dumps(_cell_for(z.device)), cells_sha(), _STOCK_ROUND))
    return out


def _report():
    _log("COUNTS " + json.dumps(COUNTS, default=str))
    try:
        import ptx_trunk2_levers as LEV
        LEV._STATS["fpf_trimul_v4"] = dict(COUNTS)
    except Exception:
        pass
atexit.register(_report)
