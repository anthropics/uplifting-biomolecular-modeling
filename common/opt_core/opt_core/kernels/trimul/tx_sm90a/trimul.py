# SPDX-License-Identifier: Apache-2.0
"""Provider with the stock module signature (FPF ops registry / fpf_smalln FAST_FN):
    fn(module, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch")
Served cell: triangle_multiplicative == "cuequivariance", c_z == c_hidden == 256, CUDA bf16 z under bf16 autocast, square pair tensor [N, N, C] or
[B, N, N, C] (B > 1 served plane by plane), mask None | [N, N] | [1|B, N, N], N >= N_MIN, device capability 9.0.  A call outside the cell takes the
stock forward for that call (counted by reason, named once) — the same dispatch contract as the current provider it replaces.  A process where the
binary cannot load raises TrimulTxUnavailable by name at the first served-cell call (the mode refuses).  Residual: fused iff inplace_safe is True and
_add_with_inplace (how PairformerBlock and the MSA pair stack call the module); otherwise the update alone is returned and the caller adds."""
import atexit, json, os, sys
import torch
from . import ops

N_MIN = 101                      # cuEquivariance itself takes a different (torch) function at N <= 100: never served here
COUNTS = {"provider": "protenix_fpf_trimul_tx", "calls": 0, "served": 0, "fallback": {}, "first_call": None, "loaded": None, "cannot_run": None, "kernel_error": None}
_SAID = set()


def _log(msg):
    sys.stderr.write("[protenix_fpf_trimul_tx] %s\n" % msg); sys.stderr.flush()


def _fallback(reason, module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative):
    import fpf
    COUNTS["fallback"][reason] = COUNTS["fallback"].get(reason, 0) + 1
    if reason not in _SAID and len(_SAID) < 12:
        _SAID.add(reason); _log("stock forward for this call (reason: %s; N=%s c=%s)" % (reason, z.shape[-2] if z.dim() >= 2 else "?", z.shape[-1]))
    return fpf.original("trimul_out" if module._outgoing else "trimul_in")(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)


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
    if not z.is_cuda or z.dim() not in (3, 4) or z.shape[-2] != z.shape[-3]:
        return fb("shape/device")
    N = int(z.shape[-2])
    if N < N_MIN:
        return fb("N<%d" % N_MIN)
    if not (torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.bfloat16) or z.dtype != torch.bfloat16:
        return fb("not-bf16-autocast")
    B = z.shape[0] if z.dim() == 4 else 1
    if mask is not None and (tuple(mask.shape[-2:]) != (N, N) or mask.numel() not in (N * N, B * N * N)):
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
    residual = bool(inplace_safe is True and _add_with_inplace)
    z4 = z if z.dim() == 4 else z.unsqueeze(0)
    m3 = None
    if mask is not None:
        m3 = mask.reshape(-1, N, N)
        if m3.dtype != torch.float32:
            m3 = m3.to(torch.float32)
    try:
        outs = []
        for b in range(B):                                       # B > 1: one launch set per pair plane (b enters only as a pointer offset)
            zb = z4[b].contiguous()
            mb = None if m3 is None else m3[b if m3.shape[0] == B else 0].contiguous()
            outs.append(ops.trimul(zb, bool(module._outgoing), mb, w, residual=residual))
        out = outs[0].unsqueeze(0) if B == 1 else torch.stack(outs, 0)
    except ops.TrimulTxUnavailable:
        raise
    except Exception as e:                                       # a kernel error is named and raised: the mode refuses; no other path continues under this name
        from opt_core.oom import is_oom
        if is_oom(e):
            raise
        COUNTS["kernel_error"] = repr(e)[:300]; _log("KERNEL ERROR (raised): %r" % (e,)); raise
    out = out if z.dim() == 4 else out[0]
    COUNTS["served"] += 1
    if COUNTS["first_call"] is None:
        COUNTS["first_call"] = {"N": N, "B": B, "dir": "out" if module._outgoing else "in", "residual": residual, "mask": mask is not None, "device": torch.cuda.get_device_name(z.device)}
        _log("FIRST CALL served: %s %s" % (json.dumps(COUNTS["first_call"]), ops.describe()))
    return out


def _report():
    _log("COUNTS " + json.dumps(COUNTS, default=str))
    try:
        import ptx_trunk2_levers as LEV
        LEV._STATS["protenix_fpf_trimul_tx"] = dict(COUNTS)
    except Exception:
        pass
atexit.register(_report)
