"""Autocast weight-cast memo (lever ``castcache``): each fp32 ``nn.Linear``'s bf16 weight (and bias) cast once per process, not once per call.

Stock under ``exact`` (the ``off`` tree's numerics: fp32 parameters, ``af3_torch_api.inference()`` = inference_mode + bf16 autocast):
every ``nn.Linear`` call is ``F.linear(input, self.weight, self.bias)`` (torch's own forward, pinned by tests/test_castcache.py), and
autocast casts the fp32 ``weight`` / ``bias`` to bf16 on EVERY call — a ``_to_copy`` kernel per parameter per call: in the diffusion
sampler's step graphs alone several hundred per step, ~700 k cast kernels and ~1.4–2.0 s of device time per item at 400–800 tokens
(the trunk's share is small: its big Linears are few per pass). autocast's own cast cache does not apply (it caches only leaf tensors
that require grad, outside inference mode).

Here (``forward`` below): the module keeps the bf16 copies autocast would make — ``weight.to(torch.bfloat16)``, the same cast op, made
once at ``install`` (before any graph capture: plain tensors at fixed addresses, so the step graphs read them like the parameters) —
and under an enabled bf16 CUDA autocast calls ``F.linear(input, <bf16 weight>, <bf16 bias>)``: autocast passes bf16 tensors through
untouched and casts ``input`` exactly as before, so the GEMM receives the same three operands bit for bit (bitwise by construction).
Outside autocast (or under another autocast dtype, or on another device) the stock statement runs on the fp32 parameters. The
parameters themselves stay fp32 and in place: the fastnn kernels that read ``.weight`` directly (gated_linear_unit reads
``transition1.weight.T``) see what they saw. Under ``fast`` / ``big`` the kit lever ``bf16w`` stores every Linear weight in bf16 —
autocast casts nothing there, so this lever is ``exact``'s alone (``install`` on such a model caches nothing and says so).
The census (``COUNTS``) rides forward.json (``castcache``) and the LEVER line: linears cached, MiB held, calls served from the memo /
run on the stock statement (Python-level calls: a step graph's replays do not count again).
"""
from __future__ import annotations

import types

COUNTS = {"served": 0, "stock": 0}
_STATE = {"linears": 0, "bytes": 0}


def take() -> dict:
    out = dict(COUNTS)
    for k in COUNTS:
        COUNTS[k] = 0
    return out


def _autocast_bf16_cuda(torch) -> bool:
    """True inside an enabled CUDA autocast whose dtype is bf16 (the context af3_torch_api.inference() opens)."""
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:                                   # torch < 2.4 spelling
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return False
    try:
        return torch.get_autocast_dtype("cuda") == torch.bfloat16
    except (AttributeError, TypeError):
        return torch.get_autocast_gpu_dtype() == torch.bfloat16


def forward(self, input):
    """nn.Linear.forward — ``return F.linear(input, self.weight, self.bias)`` — with the memoised bf16 casts standing in for the fp32
    parameters exactly where autocast would cast them (an enabled bf16 CUDA autocast, a CUDA input); the stock statement otherwise."""
    import torch
    import torch.nn.functional as F
    w = getattr(self, "_castcache_weight", None)
    if w is not None and input.is_cuda and _autocast_bf16_cuda(torch):
        COUNTS["served"] += 1
        return F.linear(input, w, self._castcache_bias)
    COUNTS["stock"] += 1
    return F.linear(input, self.weight, self.bias)


def eligible(m, torch) -> bool:
    """An ``nn.Linear`` proper (not a subclass with its own forward) whose parameters are fp32 CUDA tensors: what autocast casts per call."""
    return (type(m) is torch.nn.Linear and m.weight is not None and m.weight.dtype == torch.float32 and m.weight.is_cuda
            and (m.bias is None or (m.bias.dtype == torch.float32 and m.bias.is_cuda)))


def install(model) -> dict:
    """Memoise the bf16 casts of every eligible Linear of ``model`` and bind ``forward`` on those instances. Idempotent.
    {'installed', 'already', 'linears', 'mib', 'reason'} — ``reason`` names why nothing was cached (no eligible Linear: the weights
    are not fp32, e.g. under the kit lever bf16w)."""
    import torch
    if getattr(model, "_castcache", None):
        return {"installed": True, "already": True, "linears": _STATE["linears"], "mib": round(_STATE["bytes"] / 2**20, 1), "reason": None}
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("castcache: install during CUDA graph capture (the memo must exist before any capture)")
    n = nbytes = 0
    with torch.no_grad():
        for m in model.modules():
            if not eligible(m, torch):
                continue
            m._castcache_weight = m.weight.detach().to(torch.bfloat16)            # the cast autocast makes per call (aten::_to_copy), made once
            m._castcache_bias = None if m.bias is None else m.bias.detach().to(torch.bfloat16)
            m.forward = types.MethodType(forward, m)
            n += 1
            nbytes += m._castcache_weight.numel() * 2 + (0 if m._castcache_bias is None else m._castcache_bias.numel() * 2)
    _STATE["linears"], _STATE["bytes"] = n, nbytes
    model._castcache = True
    return {"installed": True, "already": False, "linears": n, "mib": round(nbytes / 2**20, 1),
            "reason": None if n else "no_fp32_cuda_linear(weights_not_fp32)"}
