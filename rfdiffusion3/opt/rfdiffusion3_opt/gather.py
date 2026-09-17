"""The `gather_attn` lever [RFD3_GATHER_ATTN]: the atom-level index-set attention of the diffusion module (`LocalAttentionPairBias` under
the atom encoder / decoder: H=4 heads of 32, k=128 listed keys per atom query, the pair bias `to_b(P_LL)` shared across the diffusion batch)
served by the shared core's fused gather kernel — `opt_core.kernels.gather_attn`, strategy `F5.gather_attn` — instead of upstream's dense
path (a `(D, H, L, L)` bf16 bias copy with -inf outside each query's key set, then `F.scaled_dot_product_attention`; attention.py:484-557)
or its sparse path (gathered K / V copies, attention.py:560-638). The kernel reads the k listed keys and the k bias scalars per query
directly: nothing of size `L x L` per sample is allocated, so the dense path's memory rule (`2·D·H·L²·2 < 0.35 × free`, attention.py:473-480)
has nothing to decide — under the lever upstream's decision rule answers GATHER for every atom call on CUDA at inference (upstream's
force-sparse switches RFD3_DENSE_SDPA_ATTENTION=0 / RFD3_LOW_MEMORY_MODE=1 keep their meaning and are named in the census as before).
Token calls (the 18-block diffusion transformer, `full=True`: per-sample bias, k=32) are not routed here: they stay on upstream's einsum path
or, under RFD3_TOKEN_SDPA, the dense SDPA function (the kernel supports that cell too; cuDNN's SDPA is the faster path at that shape).

Class `fast` only (tolerance): the kernel keeps the dense path's rounding points (q, k -> bf16 before the product, P -> bf16, fp32 accumulation)
and changes the summation order over keys; run to run it is bitwise reproducible. Refusals are BY NAME and counted, never silent: a call the
kernel does not serve (`opt_core.kernels.gather_attn.supported`: an unsupported (dh, k, dtype) cell, fp32 v, a non-CUDA tensor, live autograd …)
runs upstream's dense function and is counted under `stock_by[<reason>]`; the census word then carries `stock=<n>` and, when nothing was
served, the KERNELS guard refuses the pass (census.gather_word). The lever is installed by the kernels observer at the engine's initialize
(census.Observer.after_initialize -> install), on the attention module upstream's call sites look their functions up on; it needs the kit's
patched attention module (its dense function takes `dedup=`, the atom/token discriminator: attention.py:357-366) and reuses the kit's
per-denoiser-call memo (rfd3.model.hoist.call_get, part `valid`) for the one int32 sorted copy of the step's index sets all nine atom calls share.

  STATS (process tally, read by stack.report_gather / report.gather_stats / census.gather_word):
    RFD3_GATHER_ATTN  the switch as read at install      installs  modules wrapped (0 | 1)      unavailable  None | the reason the kernel cannot load
    served / stock    atom calls served by the kernel / run on upstream's dense function, stock_by = {reason: n}
    cells_served      {"dh32/k128/bfloat16": n, …}       idx_builds  int32 sorted index copies built (one per denoiser call under the memo)
    kernel            the core module's KERNEL_VERSION   strategy  F5.gather_attn
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

SWITCH = "RFD3_GATHER_ATTN"
STRATEGY = "F5.gather_attn"
CORE_MODULE = "opt_core.kernels.gather_attn"
ATTENTION_MODULE = "rfd3.model.layers.attention"
HOIST_MODULE = "rfd3.model.hoist"
DENSE_FUNCTION = "dense_sdpa_pairbias_attention"
RULE_FUNCTION = "_use_dense_sdpa_pairbias_stock"            # the kit's patched file keeps upstream's decision rule under this name (census.DECISION_RULE_KIT)
REPORT_FUNCTION = "_report_attention_path"
PATH_WORD = "GATHER"                                        # upstream's once-per-process path record reads `Atom attention path: GATHER (<reason>) [D=… L=… k=… H=…]`
PATH_REASON = f"{SWITCH}=1: index-set kernel {STRATEGY}, no dense bias"
DENSE_ENV = "RFD3_DENSE_SDPA_ATTENTION"
LOWMEM_ENV = "RFD3_LOW_MEMORY_MODE"
MEMO_PART = "valid"                                         # the hoist part the per-call index memo belongs to (hoist.call_get part_name): the per-call index-derived tensors

STATS: Dict[str, object] = {SWITCH: False, "installs": 0, "unavailable": None, "served": 0, "stock": 0, "stock_by": {}, "cells_served": {},
                           "idx_builds": 0, "kernel": None, "strategy": STRATEGY}
_TRUE = ("1", "true", "on")


def wanted(environ=None) -> bool:
    """The switch is on in this process's environment (the fast delta exports it)."""
    environ = os.environ if environ is None else environ
    return str(environ.get(SWITCH, "0")).strip().lower() in _TRUE


def active() -> bool:
    """The lever is installed on the attention module of this process (its rule answers GATHER, its dense function dispatches to the kernel)."""
    return bool(STATS["installs"]) and STATS["unavailable"] is None


def _load_kernel():
    """The core kernel module, or raise ImportError with the reason (no core, no torch, …). Triton absence is not an import error: the module
    loads and refuses every call by name (`no-triton`), which is reported as `unavailable` here so the census names it before any call."""
    import importlib
    GA = importlib.import_module(CORE_MODULE)
    if not getattr(GA, "HAVE_TRITON", False):
        raise ImportError("no-triton")
    return GA


def install(module=None, environ=None) -> bool:
    """Wrap the attention module's dense function and decision rule (idempotent). Returns True when the lever is live. Never raises for an
    unavailable kernel: STATS['unavailable'] names the reason and nothing is wrapped (the census word then reads fallback:gather-unavailable(…)
    and the KERNELS guard refuses a fast pass by name)."""
    STATS[SWITCH] = wanted(environ)
    if not STATS[SWITCH]:
        return False
    module = module if module is not None else sys.modules.get(ATTENTION_MODULE)
    if module is None:
        STATS["unavailable"] = f"{ATTENTION_MODULE}-not-imported"
        return False
    dense = getattr(module, DENSE_FUNCTION, None)
    rule = getattr(module, RULE_FUNCTION, None)
    if dense is None or rule is None:
        STATS["unavailable"] = f"patched-attention-module-required({DENSE_FUNCTION}+{RULE_FUNCTION})"
        return False
    if getattr(dense, "_gather_installed", False):
        return active()
    try:
        GA = _load_kernel()
    except ImportError as e:
        STATS["unavailable"] = f"kernel-import({_token(str(e) or type(e).__name__)})"
        return False
    STATS["kernel"] = getattr(GA, "KERNEL_VERSION", "unknown")
    hoist = sys.modules.get(HOIST_MODULE)
    wrapped_dense = _dispatcher(dense, GA, hoist)
    wrapped_rule = _rule(rule, getattr(module, REPORT_FUNCTION, None))
    if hoist is not None and getattr(hoist, "COMPILE", False):
        # [RFD3_COMPILE] both wrappers are dynamo boundaries (opt_core.capture.compile.nodynamo): the kernel launch, the memo and the tally run in
        # the interpreter between inductor's graphs, exactly like the kit's own cache accessors; CUDA-graph capture records the kernel launch.
        from opt_core.capture.compile import nodynamo
        wrapped_dense = _carry(nodynamo(wrapped_dense, active=True), wrapped_dense)
        wrapped_rule = _carry(nodynamo(wrapped_rule, active=True), wrapped_rule)
    setattr(module, DENSE_FUNCTION, wrapped_dense)
    setattr(module, RULE_FUNCTION, wrapped_rule)
    STATS["installs"] = 1
    return True


def _carry(marked, src):
    for attr in ("_gather_installed", "__wrapped__"):
        if hasattr(src, attr):
            setattr(marked, attr, getattr(src, attr))
    return marked


def _token(text: str) -> str:
    import re
    return re.sub(r"\s+", "-", str(text).strip())


def _rule(rule, report):
    """Upstream's decision rule with the lever's answer first: an ATOM call (full falsy) on CUDA without autograd, with neither force-sparse switch
    set, is served by the kernel -> True (the dense branch, whose function is the dispatcher below) and the path record says GATHER. Everything
    else is upstream's rule unchanged (token calls, training, RFD3_DENSE_SDPA_ATTENTION=0, low-memory mode, CPU)."""
    import functools

    @functools.wraps(rule)
    def _use_dense_sdpa_pairbias_stock(Q, indices, full, H):
        if not full and getattr(Q, "is_cuda", False) and not _grad_enabled() and os.environ.get(DENSE_ENV, "1") == "1" and os.environ.get(LOWMEM_ENV, "0") != "1":
            if callable(report):
                D, L = int(Q.shape[0]), int(Q.shape[1])
                report(PATH_WORD, PATH_REASON, f"D={D} L={L} k={int(indices.shape[-1])} H={int(H)}")
            return True
        return rule(Q=Q, indices=indices, full=full, H=H)

    _use_dense_sdpa_pairbias_stock._gather_installed = True
    _use_dense_sdpa_pairbias_stock.__wrapped__ = rule
    return _use_dense_sdpa_pairbias_stock


def _grad_enabled() -> bool:
    torch = sys.modules.get("torch")
    return bool(torch is not None and torch.is_grad_enabled())


def _dispatcher(dense, GA, hoist):
    """The attention module's dense function with the kernel first for ATOM calls (`dedup` truthy — the patched call site passes dedup=not full):
    bias in its native layout ((L, L, H) shared -> a (1, L, L, H) view; (D, L, L, H) per sample as is), the step's index sets as one sorted int32
    copy per denoiser call (hoist.call_get memo, shared by the nine atom calls), the output gate fused. A call the kernel refuses runs `dense`
    and is counted by reason; token calls (`dedup=False`) run `dense` uncounted (they are not this lever's)."""
    import functools
    torch = sys.modules.get("torch") or __import__("torch")

    def _idx32(indices):
        def build():
            STATS["idx_builds"] = int(STATS["idx_builds"]) + 1
            return torch.sort(indices.to(torch.int32), dim=-1).values.contiguous()
        if hoist is not None and callable(getattr(hoist, "call_get", None)):
            D_i, L_i, k_i = (int(s) for s in indices.shape[-3:]) if indices.ndim >= 3 else (1, int(indices.shape[0]), int(indices.shape[1]))
            return hoist.call_get(("gather_idx32", D_i, L_i, k_i), indices, build, part_name=MEMO_PART)
        return build()

    @functools.wraps(dense)
    def dense_sdpa_pairbias_attention(Q, K, V, B, indices, H, G=None, dedup=True, **kw):
        if not dedup:
            return dense(Q, K, V, B, indices, H, G=G, dedup=dedup, **kw)
        bias = B.unsqueeze(0) if B.ndim == 3 else B
        idx = indices.unsqueeze(0) if indices.ndim == 2 else indices
        idx32 = _idx32(idx)
        why = GA.supported(Q, K, V, bias, idx32, int(H), gate=G)
        if why is not None:
            STATS["stock"] = int(STATS["stock"]) + 1
            by = STATS["stock_by"]
            by[why] = by.get(why, 0) + 1
            return dense(Q, K, V, B, indices, H, G=G, dedup=dedup, **kw)
        out = GA.gather_attn(Q, K, V, bias, idx32, int(H), gate=G, ensure_sorted=False)
        STATS["served"] = int(STATS["served"]) + 1
        cell = f"dh{int(Q.shape[-1]) // int(H)}/k{int(idx32.shape[-1])}/{str(V.dtype).replace('torch.', '')}"
        cs = STATS["cells_served"]
        cs[cell] = cs.get(cell, 0) + 1
        return out

    dense_sdpa_pairbias_attention._gather_installed = True
    dense_sdpa_pairbias_attention.__wrapped__ = dense
    return dense_sdpa_pairbias_attention


def describe() -> dict:
    """A copy of STATS with plain values (the APPLIED line, opt_manifest.json `gather_stats`, the census detail)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in STATS.items()}
    out["active"] = active()
    return out


def census_suffix() -> str:
    """`[served=<n>,stock=<m>]` for the KERNELS atom_attn word."""
    return f"[served={int(STATS['served'])},stock={int(STATS['stock'])}]"


def final_word(kernel_word: str, calls_seen: bool = True) -> str:
    """The atom_attn final word at the pass's end, from this process's tally: `fallback:gather-unavailable(<reason>)` (the kernel could not be
    installed) | `fallback:gather-never-served(stock=<m>)` (installed, atom calls ran, none served) | `partial:gather_attn(stock=<m>,stock_by=<reason:n+…>)[served=<n>]`
    (served some, refused others by name — a DISTINCT kind: a route expecting atom_attn engaged is refused, exit 5) | `<kernel_word>[served=<n>,stock=0]`
    (kind engaged: every atom call served) — `kernel_word` is census.gather_word(), `engaged:gather_attn(<kernel>;<strategy>)`. `calls_seen` False
    (no atom call ran in the pass: nothing to serve) keeps the engaged word with its zero census."""
    bad = unavailable_word()
    if bad is not None:
        return bad
    served, stock = int(STATS["served"] or 0), int(STATS["stock"] or 0)
    if served == 0 and calls_seen:
        return f"fallback:gather-never-served(stock={stock})"
    if stock > 0:
        by = "+".join(f"{_token(k)}:{v}" for k, v in sorted((STATS.get("stock_by") or {}).items())) or "unnamed"
        return f"partial:gather_attn(stock={stock},stock_by={by})[served={served}]"
    return f"{kernel_word}{census_suffix()}"


def unavailable_word() -> Optional[str]:
    """The atom_attn word when the switch is on but the lever could not install: `fallback:gather-unavailable(<reason>)` (kind fallback: a fast
    pass expecting atom_attn engaged is refused by name); None when not wanted or live."""
    if STATS[SWITCH] and STATS["unavailable"] is not None:
        return f"fallback:gather-unavailable({_token(str(STATS['unavailable']))})"
    return None
