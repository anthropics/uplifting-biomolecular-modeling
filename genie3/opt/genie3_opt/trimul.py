"""Lever L7 — Genie 3's pair-transform TriangleMultiplication (Jumper et al. 2021, suppl. Alg. 11/12 form, module
``genie3.generation.model.module.triangular_multiplicative_update.TriangleMultiplicativeUpdate``: ``tri_mul_out`` / ``tri_mul_in`` of each of the
five ``LatentTransformerBlock``s, latent/transformer.py:118-119 — 10 calls per denoiser call, on the pair tensor ``[B, N, N, 128]`` fp32 with
N = n_token + 10 global tokens) served by the shared core's fused TriMul kernel through the core's one TriMul adapter: ``opt_core.trimul.Lever``
(the decision ladder, the census, the fail-closed gate, the activation-evidence line) with provider ``opt_core.trimul.fpf_v4`` = the carried kernel
``fpf_trimul_v4`` (its module-agnostic ``generic`` entry), strategy word ``F2.fpf_trimul_fast``. The kernel is the core's copy, reached BY NAME:
``route()`` exports this kit's cell table ``fpf_cells.json`` as ``FPF_TRIMUL_V4_CELLS`` before the first import, routes ``import fpf_trimul_v4`` to
the core copy and holds its bytes (``opt_core.kernels.exports`` / ``route`` / ``route_check``); nothing of the kernel lives in this tree.

Numerics: NOT value-identical — bf16 tensor-core GEMM operands with fp32 accumulation, fp32 LayerNorm statistics / gating / biases, fp32 in and
out (the kernel's fp32-z path); class 3 (tolerance tier): mode ``fast``'s TriangleMultiplication provider (modes.FAST_FLAGS carry ``--trimul fpf``),
never on the exact line. Math served = the module's own: ``a = (linear_a_p(LN_in z) + b) · sigmoid(linear_a_g(LN_in z) + b) · mask``, ``b`` likewise,
``out = (linear_z(LN_out(a ∘ b)) + b) · sigmoid(linear_g(LN_in z) + b)``, no residual (triangular_multiplicative_update.py:135-146; the six Linear
biases are passed to the kernel), ``_outgoing`` → direction outgoing | incoming.

Serve rule (the core ladder, one decision per call, never silent): a CUDA fp32/bf16 pair tensor with c_z = 128 and c_hidden = 128, N ≥ 101
(``MIN_TOKENS`` = the kernel's own small-N floor ``fpf_trimul_v4.generic.N_MIN``), on a device whose (compute capability | triton) has a served
row in ``fpf_cells.json`` (H100 / H200, cc 9.0; A100, cc 8.0 — its row carries C128/D128 tile overrides) is served; there is no upper size limit in the kernel. Anything else runs the module's own forward
COUNTED BY REASON on the lever's census: ``below_min_tokens`` (N < 101 — binder / motif requests whose padded pair extent is under 101 tokens; the
one fallback mode ``fast`` EXPECTS; a request whose EVERY call is under the floor served nothing: the lever DECLINED by name — word
``declined:below_min_tokens[…]``, LEVER line ``state=skipped`` — and the pass ran the module's own forward throughout, no refusal), ``C:…`` / ``D:…`` (another checkpoint's widths), ``dtype:…``, ``mask:…``, ``no_cell:…`` (a GPU without a
served row: L4, L40S, …), ``device:cpu``, ``rank:…``; a kernel exception is ``error:<Type>`` (counted, the first two printed, the call served
by the module's forward). A fallback reason the mode does not expect, or any error, REFUSES the lever's gate: the KERNELS word reads
``fallback:…`` and lever L7 has no evidence — the kernel could not serve this box, so mode fast REFUSES BY NAME after the pass (design.run: exit 3,
``levers_missing``; a mode is all of its levers — ``OPT_CORE_TRIMUL_STRICT=1`` re-raises instead). CUDA out-of-memory propagates uncaught.

Capture safety (L4): ``enable()`` patches the class BEFORE the batched driver's graph capture; the capture's two eager warm-up forwards
(g3fast.py GraphedDenoiser) do the host work — weight packing (cached on the module by the core adapter), the cell lookup, the kernel's first-call
line — so only kernel launches are captured; graph replay == eager bit for bit. The census
counts EAGER calls (warm-ups + capture: 3 × 10 per captured batch shape); the replayed steps execute the captured launches without passing through
Python.

Evidence: ``word()`` is the ``trimul=`` word of the KERNELS census line (kernels.py) the batched driver prints at exit —
``engaged:fpf_trimul_v4@<version>-<origin>[served=<n>,fallback=0]`` when every call was served, ``partial:…[served=<n>]`` when some calls ran the
module's own forward for an expected, counted reason (the gate holds), ``declined:below_min_tokens[…]`` when every
call was under the kernel's floor (nothing to serve: registry L7's ``declined`` rule), ``fallback:<gate reason>[…]`` when the gate refuses,
``off-by-route:stock`` when the lever is off — and registry L7's evidence rule matches ``engaged`` | ``partial`` with ``served`` ≥ 1;
``evidence()`` is the record for the timings JSON (``trimul``): mode, the core census, the gate, the route details, the kernel's cell row.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from . import registry
from .codes import TAG

KERNEL = "fpf_trimul_v4"
STRATEGY = "F2.fpf_trimul_fast"                                           # opt_core STRATEGIES.json canonical id of the provider this lever routes
STOCK_MODULE, STOCK_CLASS = "genie3.generation.model.module.triangular_multiplicative_update", "TriangleMultiplicativeUpdate"
CELLS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fpf_cells.json")
SWITCH, _ON = registry.LEVERS["L7"].switch[1], registry.LEVERS["L7"].switch[2]      # "--trimul", "fpf": the DRIVER's flag (registry L7's switch, the one spelling); the mode table passes it on fast (modes.FAST_FLAGS)
CHOICES, DEFAULT = ("stock", _ON), "stock"                                            # the driver's own default is the module's forward; `stock` is also `design --trimul stock`, the opt-out
MIN_TOKENS = 101                                                          # fpf_trimul_v4.generic.N_MIN: below it the module's own forward runs, counted `below_min_tokens`
EXPECTED_FALLBACKS = ("below_min_tokens",)                                # the fallback reasons mode `fast` expects (opt_core.trimul.Lever.expected); anything else refuses the gate
STATE = {"mode": DEFAULT, "route": None, "lever": None}
_ORIG = {}


def cell_row(cc: Optional[str]) -> Optional[str]:
    """The fpf_cells.json row key that serves compute capability ``cc`` (``"<cc>|*"`` or ``"<cc>|<triton>"``), or None — on a card without a
    row every call is a counted ``no_cell:…`` fallback to the module's forward, the lever's gate refuses and a fast pass is partial."""
    if not cc:
        return None
    rows = json.load(open(CELLS_JSON, encoding="utf-8"))                              # the kit's own file: absent or unreadable is an error, not a silent "no row"
    for key in rows:
        if not str(key).startswith("_") and str(key).partition("|")[0] == str(cc):
            return key
    return None


def route() -> dict:
    """Export the cell table, route ``fpf_trimul_v4`` to the core copy. A refusal raises with the gate's reason: nothing is patched
    over a copy that is not the carried one."""
    from . import _core
    _core.ensure_importable()
    import opt_core.kernels as kernels
    os.environ.update(kernels.exports(KERNEL, cells=CELLS_JSON))          # before the first import: cells.py reads it at import
    kernels.route(KERNEL)
    gate = kernels.route_check(KERNEL)
    if not gate.ok:
        raise RuntimeError(f"{KERNEL}: {gate.reason}")
    STATE["route"] = dict(gate.details)                                     # runtime_imports is {}: fpf_trimul_v4 imports no other carried kernel at run time
    return STATE["route"]


def weights_of(m) -> dict:
    """The module's parameters under the core adapter's canonical TriMul vocabulary (opt_core.trimul.WEIGHT_KEYS + BIAS_KEYS); the core packs
    them once per module and caches the pack on the module."""
    return dict(ln_in_w=m.layer_norm_in.weight, ln_in_b=m.layer_norm_in.bias,
                w_ag=m.linear_a_g.weight, b_ag=m.linear_a_g.bias, w_ap=m.linear_a_p.weight, b_ap=m.linear_a_p.bias,
                w_bg=m.linear_b_g.weight, b_bg=m.linear_b_g.bias, w_bp=m.linear_b_p.weight, b_bp=m.linear_b_p.bias,
                ln_out_w=m.layer_norm_out.weight, ln_out_b=m.layer_norm_out.bias,
                w_o=m.linear_z.weight, b_o=m.linear_z.bias, w_og=m.linear_g.weight, b_og=m.linear_g.bias)


def lever():
    """The process's one ``opt_core.trimul.Lever`` for this kit's L7 (built on first use; mode ``fast`` routes provider ``fpf_v4``)."""
    if STATE["lever"] is None:
        from . import _core
        _core.ensure_importable()
        from opt_core import trimul as T
        STATE["lever"] = T.Lever(TAG, "fast", provider=T.fpf_v4(weights_of, cache_key="genie3"), min_tokens=MIN_TOKENS, expected=EXPECTED_FALLBACKS)
    return STATE["lever"]


def _forward(self, z, mask=None):
    from opt_core.trimul import Call
    return STATE["lever"].serve(Call(self, z, mask, "outgoing" if self._outgoing else "incoming", residual=False, orig=lambda: _ORIG["forward"](self, z, mask)))


def enable() -> dict:
    """Route the kernel, build the lever and patch the stock class in this process (idempotent). Returns the route record."""
    import importlib
    rec = route()
    lever()
    cls = getattr(importlib.import_module(STOCK_MODULE), STOCK_CLASS)
    if "forward" not in _ORIG:
        _ORIG["forward"] = cls.forward
        cls.forward = _forward
    STATE["mode"] = "fpf"
    return rec


def census() -> dict:
    """The core lever's census (``{strategy, mode, provider, kernel, origin, served, fallback{reason:n}, errors{type:n}, by_direction, first, …}``),
    or the stock record when the lever is off."""
    if STATE["mode"] != "fpf" or STATE["lever"] is None:
        return {"mode": "stock", "served": 0, "fallback": {}, "errors": {}}
    return dict(STATE["lever"].census(), mode="fpf")


def gate() -> dict:
    """``{ok, reason}`` of the core lever's fail-closed gate (unexpected fallback reasons / kernel errors refuse it); ok with reason None when off."""
    if STATE["mode"] != "fpf" or STATE["lever"] is None:
        return {"ok": True, "reason": None}
    g = STATE["lever"].gate()
    return {"ok": bool(g.ok), "reason": g.reason}


def declined(ev: Optional[dict] = None) -> Optional[str]:
    """The reason the lever DECLINED this request, or None: the kernel was routed, no call was served, no kernel error, and every module-forward
    call was counted under an EXPECTED reason (``below_min_tokens``: every pair extent of the request under the kernel's floor) — nothing the
    kernel could serve arrived. (The core gate reads `served 0 of n` as refused; for this kit that census is a named decline, not a failure.)"""
    if ev is None:
        ev = {"mode": STATE["mode"], "census": census(), "gate": gate()}
    if ev.get("mode") != "fpf":
        return None
    c = ev.get("census") or {}
    fb = dict(c.get("fallback") or {})
    if int(c.get("served") or 0) != 0 or not fb or c.get("errors"):
        return None
    if all(r in EXPECTED_FALLBACKS for r in fb):                                    # exactly the expected reason(s), nothing else
        return "+".join(sorted(fb))
    return None


def evidence() -> dict:
    """The record for the timings JSON (``trimul``): mode, census, gate, route, the kernel's cell row (INFO without the launch cfg), the word,
    ``declined`` (the decline reason or None)."""
    out = {"mode": STATE["mode"], "census": census(), "gate": gate(), "route": STATE["route"], "strategy": STRATEGY, "word": word(), "declined": declined()}
    if STATE["mode"] == "fpf":
        import sys
        c = sys.modules.get("fpf_trimul_v4.cells")
        out["cells"] = {k: {kk: vv for kk, vv in v.items() if kk != "cfg"} for k, v in (getattr(c, "INFO", {}) or {}).items()}
    return out


def _reasons(d: dict) -> str:
    return "+".join(f"{k}:{v}".replace(" ", "") for k, v in sorted((d or {}).items()))   # no blanks inside a census word


def word(ev: Optional[dict] = None) -> str:
    """The ``trimul=`` word of the KERNELS census line: ``off-by-route:stock`` | ``engaged:<kernel>@<version>-<origin>[served=<n>,fallback=0]`` (every
    call served) | ``partial:<kernel>@<version>-<origin>(stock=<m>,stock_by=<reason>:<m>+…)[served=<n>]`` (m calls ran the module's own forward for an
    EXPECTED, counted reason — e.g. below_min_tokens on a short binder input; the gate holds) | ``declined:<reason>[served=0,fallback=<n>(…)]`` (every
    call ran the module's own forward for an EXPECTED reason — nothing the kernel could serve arrived: the lever declined this request, by name) |
    ``fallback:<gate reason>[served=<n>,fallback=<n>(…),errors=<n>(…)]`` (the gate refused: an unexpected fallback reason or a kernel error)."""
    if ev is None:
        ev = {"mode": STATE["mode"], "census": census(), "gate": gate()}
    if ev.get("mode") != "fpf":
        return "off-by-route:stock"
    c, g = ev.get("census") or {}, ev.get("gate") or {}
    kernel = f"{c.get('kernel') or KERNEL + '@?'}-{c.get('origin') or '?'}"
    served, stock = int(c.get("served") or 0), sum((c.get("fallback") or {}).values())
    tail = f"served={served},fallback={stock}"
    if stock:
        tail += f"({_reasons(c['fallback'])})"
    if c.get("errors"):
        tail += f",errors={sum(c['errors'].values())}({_reasons(c['errors'])})"
    why = declined(ev)
    if why is not None:
        return f"declined:{why}[{tail}]"                                             # every call under the kernel's floor: the lever declined this request by name (the module's own forward ran)
    if g.get("ok") and served >= 1 and not stock:
        return f"engaged:{kernel}[{tail}]"                                             # every call served by the kernel
    if g.get("ok") and served >= 1:
        return f"partial:{kernel}(stock={stock},stock_by={_reasons(c['fallback'])})[served={served}]"   # some calls ran the module's own forward for an EXPECTED reason (counted)
    reason = g.get("reason") or ("served_none" if not served else "unknown")
    return f"fallback:{str(reason).replace(' ', '_')[:60]}[{tail}]"


def lever_line() -> str:
    """The core adapter's activation-evidence line for this lever (``[genie3-opt] LEVER name=F2.trimul state=… impl=… served=… gate=…``); a declined
    request prints the core's ``state=skipped reason=<why>`` form (its ``gate=`` word stays the core's reading of the empty census)."""
    if STATE["lever"] is None:
        return f"[{TAG}] LEVER name=F2.trimul state=off mode=stock"
    why = declined()
    if why is not None:                                                              # the core's own grammar for "selected, not applied": state=skipped reason=<why> (its gate word stays the core's)
        return STATE["lever"].line(state="skipped", reason=why)
    return STATE["lever"].line()
