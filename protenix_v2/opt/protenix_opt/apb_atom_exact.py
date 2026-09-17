"""protenix_opt.apb_atom_exact — lever ``atom_attn_exact`` (EXACT-BITWISE): the atom transformers' 32-query x 128-key local attention bound to
the shared core's pair-bias attention provider BY THE TIER WORD ``exact`` (``opt_core.kernels.apb``: ``select(word="exact")`` names, per call
class, either the carried row ``atom_exact`` — a prebuilt kernel the core carries, vouched
per stack / card by the provider's cells — or the stock statement (``sdpa_gather``), which the site then runs BY NAME, counted; never a guess).

The runner-seam contract of ``protenix_opt.sampler_levers`` (``install(model)`` / ``report()`` / ``kernel``): install determines the classes the
exact word vouches here (pure table reads), loads and checks the carried kernel once through the provider (``atom_exact_ready``: the cuBLAS route
table of this process + the carried load-check vectors' output digests; a failure refuses BY NAME -> ``ATOMATTNEXACT:unavailable``, the mode
refuses), then replaces ``protenix.model.modules.primitives._local_attention``; per call the exact word decides kernel | statement.  The
dtype gate (``sampler_levers.gate_atom_attn_exact``) still wraps the bound site: a per-call ``kernel.Refused`` (an unvouched chunk batch count)
answers with the stock statement, counted.  The site belongs to ``atom_attn`` / ``atom_fused`` in the fast tiers: then install refuses by name.
"""
import atexit
import json
import os
import sys
import time
from typing import Any, Dict, Optional

from . import apb_core as C

NAME = "atom_attn_exact"
PREFIX = "[protenix-opt]"
REPORT: Dict[str, Any] = {"installed": False}
STATE: Dict[str, Any] = {"calls": 0, "routes": {}, "classes": {}, "stack": None, "row": None}
kernel = None                                   # the provider's carried kernel module once loaded (sampler_levers' dtype gate reads kernel.Refused / kernel.STATE)
try:
    __version__ = C.core_version()
except Exception:  # noqa: BLE001
    __version__ = "?"


class LeverRefused(RuntimeError):
    pass


def _probe(A, stack: Optional[str]) -> Dict[str, str]:
    """What the exact word resolves to on THIS stack for the atom classes the model produces: {"<atoms>/<S>/<eager|graph>": arm}; table reads only."""
    out: Dict[str, str] = {}
    cc = C._device_cc_tuple()
    for n_tok in (256, 400, 800, 1200, 1536, 2048):
        for samples in (1, 5):
            for capture in (False, True):
                key = f"{n_tok * 8}/S{samples}/{'graph' if capture else 'eager'}"
                try:
                    s = A.select(cc, "fp32", C.CELLS["atom_attn"], n_tok, word="exact", samples=samples, capture=capture,
                                 heads=C.ATOM_HEADS, head_dim=C.ATOM_HEAD_DIM, stack=stack)
                    out[key] = A.arm_word(s.row, s.variant)
                except A.Refusal as e:
                    out[key] = "refused:" + str(e.kind)
    return out


def install(model=None) -> Dict[str, Any]:
    global kernel
    import torch
    import protenix.model.modules.primitives as PR
    if os.environ.get("PTX_ATOM_ATTN_EXACT", "0") != "1":
        raise LeverRefused(f"{NAME}: install called with PTX_ATOM_ATTN_EXACT != 1")
    for other, env in (("atom_attn", "PTX_ATOM_ATTN"), ("atom_fused", "PTX_ATOM_FAST")):
        if os.environ.get(env, "0") == "1":
            raise LeverRefused(f"{NAME}: the atom local-attention site is owned by lever {other} ({env}=1) in this mode")
    if getattr(PR._local_attention, "_atom_attn_exact", False):
        return REPORT
    if not torch.cuda.is_available():
        raise LeverRefused(f"{NAME}: CUDA not available")
    t0 = time.perf_counter()
    try:
        A = C.face()
    except Exception as e:  # noqa: BLE001
        raise LeverRefused(f"{NAME}: {e}") from e
    stack = A.stack_word()
    probe = _probe(A, stack)
    classes = sorted(k for k, v in probe.items() if v == "atom_exact")
    orig = PR._local_attention
    cc = torch.cuda.get_device_capability()
    routes, bad, n_lc = "none", [], 0
    if classes:                                                   # the provider vouches the carried kernel here for some class: load + check it now (a failure refuses by name)
        try:
            K = A.atom_exact_ready(torch.device("cuda", torch.cuda.current_device()))   # route table + carried load-check digests, once per process
        except A.Refusal as e:
            raise LeverRefused(f"{NAME}: the provider refuses row atom_exact here ({e})") from e
        kernel = K
        routes = K.route_summary()
        n_lc = len((json.load(open(os.path.join(os.path.dirname(os.path.abspath(K.__file__)), "vectors.json"))) or {}).get("cases", []))
        fused = K.make_local_attention(orig)                      # the kernel's own envelope: misses take the original statement, counted in K.STATE
        K.STATE["calls"] = {"kernel": 0, "original": 0}; K.STATE["why_original"] = {}

        def _local_attention(q, k, v, n_queries, n_keys, *a, **kw):
            STATE["calls"] += 1
            try:
                S = int(q.shape[0]) if q.dim() >= 4 else 1
                sel = C._select(A, "atom_attn", q, N=int(q.shape[-2]), S=S, heads=C.ATOM_HEADS, head_dim=C.ATOM_HEAD_DIM, word="exact",
                                capture=C._capturing(q), stack=stack, cell=C.CELLS["atom_attn"])
                why = None if sel.row == "atom_exact" else "exact_word:" + A.arm_word(sel.row, sel.variant)
            except A.Refusal as e:
                why = "exact_word:refused:" + str(e.kind)
            if why is None:
                C._count(STATE["routes"], "kernel")
                return fused(q, k, v, n_queries, n_keys, *a, **kw)
            C._count(STATE["routes"], why)
            return orig(q, k, v, n_queries, n_keys, *a, **kw)

        _local_attention._atom_attn_exact = True
        PR._local_attention = _local_attention
        STATE["row"] = "atom_exact"
    else:
        STATE["row"] = "statement"                                # the exact word names the stock statement for every class here: nothing is bound, said on the marker
    STATE.update(stack=stack, classes=probe)
    REPORT.update(installed=True, routes=routes, unreproducible_batches=bad, loadcheck=[f"c{i + 1}" for i in range(n_lc)], cc=f"sm_{cc[0]}{cc[1]}",
                  tf32=bool(torch.backends.cuda.matmul.allow_tf32), seconds=round(time.perf_counter() - t0, 2), row=STATE["row"],
                  classes=f"{len(classes)}/{len(probe)}", stack=stack, provider=f"{C.FACE} {C.core_version()}")
    if classes:
        floor = sorted({v for v in probe.values() if v != "atom_exact"})
        served = (f"word=exact provider={C.FACE} {C.core_version()} row=atom_exact serves {len(classes)}/{len(probe)} atom classes vouched on {stack}"
                  + (f"; the rest keep the statement by name ({','.join(floor)})" if floor else ""))
        sys.stderr.write(f"{PREFIX} ATOMATTNEXACT:on(routes={routes} unreproducible={bad} loadcheck={n_lc}/{n_lc} cc=sm_{cc[0]}{cc[1]} tf32={REPORT['tf32']} "
                         f"t={REPORT['seconds']}s; {served}; EXACT-BITWISE vs the chunked SDPA statement fp32 D={C.ATOM_HEAD_DIM})\n")
    else:
        sys.stderr.write(f"{PREFIX} ATOMATTNEXACT:on(word=exact provider={C.FACE} {C.core_version()} -> {','.join(sorted(set(probe.values()))) or 'none'} on {stack} "
                         f"(no class vouched for atom_exact here): every call keeps the statement BY NAME; cc=sm_{cc[0]}{cc[1]})\n")
    sys.stderr.flush()
    atexit.register(_exit_line)
    return REPORT


def report() -> Dict[str, Any]:
    calls = dict(getattr(kernel, "STATE", {}).get("calls") or {"kernel": 0, "original": 0}) if kernel is not None else {"kernel": 0, "original": 0}
    why = dict(getattr(kernel, "STATE", {}).get("why_original") or {}) if kernel is not None else {}
    return {"installed": REPORT.get("installed"), "routes": REPORT.get("routes"), "calls": calls, "original_path_reasons": why,
            "row": STATE.get("row"), "word_routes": dict(STATE["routes"]), "site_calls": STATE["calls"], "stack": STATE.get("stack")}


def _exit_line() -> None:
    try:
        sys.stderr.write(f"{PREFIX} EXIT atom_attn_exact " + json.dumps(report(), default=str) + "\n"); sys.stderr.flush()
        p = os.environ.get("PTX_LEVER_REPORT", "")
        if p:
            with open(p, "a") as fh:
                fh.write(json.dumps({"apb_atom_exact": report(), "pid": os.getpid()}, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass
