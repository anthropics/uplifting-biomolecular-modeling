"""boltz2_opt.trimul — the engine adapter of the triangle-multiplication levers (registry: ``fpf_trimul_exact`` = the exact row's TriMul,
strategy id F2.fpf_trimul_exact; ``fpf_trimul`` = the fast / big rows' TriMul, strategy id F2.fpf_trimul_fast), applied in the worker process
by ``boltz2_opt.worker_launch --attach trimul`` right after boltz's own ``boltz.model.layers.triangular_mult`` has executed. It composes the
core's TriMul adapter (``opt_core.trimul``: the decision ladder, the census, the fail-closed gate, the one activation-evidence line) over boltz
2.2.1's two TriMul classes and binds, per call, the shared core's ONE triangle-multiplication provider (``opt_core.kernels.trimul`` through
``opt_core.trimul.by_word``) BY TIER WORD: the provider's measured cell table names, for this process's card and stack and the call's own facts
(precision — Boltz-2's pair stacks hand the TriMul an fp32-resident z under bf16 autocast, the table's ``f32z_bf16`` segment —, c_z — 128 on
the trunk / confidence / MSA-module pair stacks, 64 on the template Pairformer —, N bucket, direction), the row that serves. This tree carries
no TriMul kernel, no TriMul cell table and no token gate of its own: every (card, shape) decision is the provider's.

Switch ``BOLTZ_FPF_TRIMUL`` (the mode table sets it) names the numerics class: ``exact`` = the provider's ``exact`` TIER word — per call an
exact-class KERNEL row the provider vouches byte-identical to upstream's fused TriMul on this stack (``native_exact``, ``tmk3_exact`` …)
serves through the provider's face; where the provider names the library op itself as the cell's exact row (its stock rows ``cueq`` /
``torch_math``) the engine's OWN call of that op serves, BY NAME (counted ``library_op:<row>`` on the line; the bytes are upstream's either
way); ``1`` = a tolerance-class tier word (``fast`` on the fast row, ``big`` on the big row: ``BOLTZ_FPF_TRIMUL_PROVIDER``) — the provider's
measured winner of the cell serves through the face, its stock rows included. ``BOLTZ_FPF_TRIMUL_PROVIDER=<word>`` is the row's provider word:
a TIER word of the variant's class (``exact`` | ``fast`` / ``big``; unset = ``exact`` on the exact variant, ``fast`` on ``1``) or one provider
ROW by name, class-true (exact-class rows on the exact variant, tolerance-class rows on ``1``): a named row this card / stack cannot serve is
refused at install BY NAME (``<row>:<kind>``, e.g. ``tx_sm90a:cc:8.0!=9.0(…)`` on an A100; the lever is then a skipped lever of the row), a
named row that cannot serve one call refuses that call by name (counted; the gate refuses) — nothing is substituted; any other word is refused
at apply by name. ``BOLTZ_FPF_TRIMUL_MAX_TOKENS=<n>``, when a row names it, is a ceiling: a call above it takes the engine's own TriMul BY NAME
(counted ``above_max_tokens``); unset = no ceiling. Under the fast / big rows' PAIRFUSE layer driver the C=128 stacks' TriMul is the driver's
own site (its ``trimul=core.<word>`` pick, the same provider by the same tier words); this class-level lever then serves the stacks the driver
hands back (the template Pairformer's C=64 TriMul, inputs below the driver's floor) and prints ``superseded_by=pairfuse@c128``.

What is engine-specific and therefore lives here: the class patched (``TriangleMultiplicationOutgoing`` / ``…Incoming``; stock
``boltz/model/layers/triangular_mult.py:39,127``), the parameter map (``weights_of``: boltz packs the a|b projections as the two halves of
``p_in`` / ``g_in`` along the OUTPUT features, ``triangular_mult.py:110-116``), the switch, and the fallbacks each row EXPECTS (``EXPECTED_OF``:
the exact row's library-op cells, a ceiling's word — everything else the ladder counts refuses the kit's exit gate).

Contract (read by worker_launch and stack.evidence, the same shape as boltz2_opt.big): ``LEVERS``; ``apply(spec=None)`` returns the levers
applied ([] when the row does not request the lever: the attach refuses by name); ``report()`` returns ``{applied, disabled: {}, line, census,
gate, tier}`` — ``line`` is the lever's activation-evidence line (also printed on stderr at worker exit), ``gate`` the core ladder's fail-closed
verdict, ``tier`` the provider word's routes in this process (word / rows served through the face / library-op cells / last row and cell).
"""
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

TAG = "boltz2-opt"
SWITCH = "BOLTZ_FPF_TRIMUL"
MAX_TOKENS_ENV, PROVIDER_ENV = "BOLTZ_FPF_TRIMUL_MAX_TOKENS", "BOLTZ_FPF_TRIMUL_PROVIDER"   # companion words of the switch (leave the row with it: modes.take_off)
VARIANTS = ("exact", "1")
TIER_OF = {"exact": "exact", "1": "fast"}                              # the provider TIER word a variant binds when the row names none
TIER_WORDS_OF = {"exact": ("exact",), "1": ("fast", "big")}          # the provider TIER words a variant may name (class-true: the exact word never resolves to a tolerance-class row)
ROW_WORDS_OF = {"exact": ("tmk3_exact", "native_exact", "tx_sm90a_exact"),   # the provider ROWS a variant may name by name (class-true: exact-class kernel rows on the exact lever,
                "1": ("v4", "tmk3_fast", "tx_sm90a", "native", "esm_v5_fwd", "esm_shapes", "esm_k1ptr")}   # tolerance-class rows on the fast lever); a row word is an ablation entry, no mode row names one
PROVIDER_WORDS_OF = {v: TIER_WORDS_OF[v] + ROW_WORDS_OF[v] for v in VARIANTS}
LIBRARY_OP = "library_op"                                            # census word (line: fallback_by=library_op:<row>:<n>): under the exact word the provider named the library op itself
                                                                     # (its stock row <row>) as the cell's exact row — the engine's own call of it serves that call, by name
ABOVE_MAX = "above_max_tokens"                                       # the fallback word of a call above the row's ceiling (declared: EXPECTED when a ceiling is set)
LEVERS_OF = {"exact": ("fpf_trimul_exact",), "1": ("fpf_trimul",)}
LEVERS = ("fpf_trimul", "fpf_trimul_exact")
KERNEL_OF = {"exact": "fpf_trimul", "1": None}                       # the carried core kernel a variant's row routes by name (modes.ROUTED_KERNELS; the provider reaches every kernel by its core name too)
EXPECTED_OF = {                                                     # the rows' declared paths to the engine's own TriMul (counted, printed in the line's fallback_by=; the gate stays open):
    "1": (),                                                        # the tolerance-class words serve every cell through the face (the provider's stock rows included) — none;
    "exact": (LIBRARY_OP,),                                         # the exact word: the cells whose exact row is the library op (the provider's stock rows) — the engine calls it
}
EXPECTED = EXPECTED_OF["1"]
IDLE = "idle: every TriMul call of the run took a declared path to the engine's own TriMul (the provider names the library op for every cell of this input on this card) — the lever is installed and served nothing"
_STATE: Dict[str, Any] = {"lever": None, "applied": [], "patched": [], "variant": None, "max_tokens": None, "provider_word": None, "disabled": {}, "provider": None}


def weights_of(m) -> Dict[str, Any]:
    """boltz TriangleMultiplication{Outgoing,Incoming} -> the core's ten canonical TriMul tensors (opt_core.trimul.WEIGHT_KEYS)."""
    d = m.p_in.weight.shape[0] // 2
    return dict(ln_in_w=m.norm_in.weight, ln_in_b=m.norm_in.bias, w_ap=m.p_in.weight[:d], w_ag=m.g_in.weight[:d], w_bp=m.p_in.weight[d:], w_bg=m.g_in.weight[d:],
                ln_out_w=m.norm_out.weight, ln_out_b=m.norm_out.bias, w_o=m.p_out.weight, w_og=m.g_out.weight)


def variant(environ=None) -> Optional[str]:
    env = os.environ if environ is None else environ
    v = env.get(SWITCH, "").strip().lower()
    return v if v in VARIANTS else None


def requested(environ=None) -> bool:
    return variant(environ) is not None


def max_tokens(environ=None) -> Optional[int]:
    """The row's ceiling (BOLTZ_FPF_TRIMUL_MAX_TOKENS) or None (no ceiling: today's rows)."""
    env = os.environ if environ is None else environ
    v = env.get(MAX_TOKENS_ENV, "").strip()
    return int(v) if v.isdigit() else None


def provider_word(environ=None) -> Optional[str]:
    """The row's provider word (BOLTZ_FPF_TRIMUL_PROVIDER) or None (the variant's tier word, TIER_OF)."""
    env = os.environ if environ is None else environ
    v = env.get(PROVIDER_ENV, "").strip().lower()
    return v or None


def tier_word(v: str, word: Optional[str]) -> str:
    """The provider word a variant binds: the row's word, else the variant's tier word."""
    return word if word else TIER_OF[v]


def _exact_kernel_rows(KT) -> Tuple[str, ...]:
    """The provider's exact-class KERNEL rows: its table's rows of class 'exact' that are not its stock rows (``cueq`` / ``torch_math``: the words
    with which the face says 'the library op is the exact row of this cell') nor its module-form rows — the rows the exact word serves through the face."""
    rows = (KT.table() or {}).get("rows") or {}
    aside = set(getattr(KT, "STOCK_ROWS", ("cueq", "torch_math"))) | set(getattr(KT, "MODULE_EXACT_ROWS", ()))   # module-form rows state another engine's module arithmetic, not the library op
    return tuple(sorted(r for r, rec in rows.items() if isinstance(rec, dict) and rec.get("class") == "exact" and r not in aside))


def _provider(T, v: str, word: Optional[str]):
    """The variant's provider: the core provider by the row's word (``opt_core.trimul.by_word``) — a TIER word of the variant's class or one of its
    rows by name (class-true; any other word raises ValueError, the attach refuses by name).  The returned provider tallies its routes
    (``.routes``: word / rows served through the face / library-op cells handed to the engine / the last row and cell) for the line and report().

    exact word: after the face's own eligibility (``opt_core.kernels.trimul.select`` + ``admits`` on the call's facts; the Selection rides on
    ``call.extra``) a cell whose row is one of the provider's STOCK rows is handed to the engine's own TriMul BY NAME (``Refused('library_op:<row>')``,
    counted, EXPECTED): upstream's kernel stays the engine's call — the stock arm's bytes by construction; an exact-class kernel row serves
    through the face.  Tolerance-class words: the face serves every cell it names."""
    word = tier_word(v, word)
    if word not in PROVIDER_WORDS_OF[v]:
        raise ValueError(f"{PROVIDER_ENV}={word!r}: not a word the {('exact' if v == 'exact' else 'fast')} variant accepts — its tier word(s) "
                         f"{', '.join(TIER_WORDS_OF[v])} or one of the provider's {('exact' if v == 'exact' else 'tolerance')}-class rows by name "
                         f"({', '.join(ROW_WORDS_OF[v])})")
    prov = T.by_word(weights_of, word, name=f"trimul:{word}")
    routes: Dict[str, Any] = {"word": word, "face": {}, "aside": {}, "last_face": None,
                              "kernel_rows": sorted(_exact_kernel_rows(__import__("opt_core.kernels.trimul", fromlist=["trimul"]))) if v == "exact" else []}
    stock_rows = set(getattr(__import__("opt_core.kernels.trimul", fromlist=["trimul"]), "STOCK_ROWS", ("cueq", "torch_math")))
    inner_eligible, inner_fn = prov.eligible, prov.fn

    def eligible(call):
        inner_eligible(call)                                               # the face: select + admits on the call's own facts, or Refused by name
        sel = call.extra.get("trimul_selection")
        row = getattr(sel, "row", None)
        if v == "exact" and row in stock_rows:                             # the library op IS this cell's exact row: the engine's own call of it serves, by name
            call.extra.pop("trimul_selection", None)
            why = f"{LIBRARY_OP}:{row}"
            routes["aside"][why] = routes["aside"].get(why, 0) + 1
            raise T.Refused(why)
        return None

    def fn(call):
        sel = call.extra.get("trimul_selection")
        row, cell = getattr(sel, "row", None), getattr(sel, "cell", None)
        out = inner_fn(call)
        key = str(row or "?")
        routes["face"][key] = routes["face"].get(key, 0) + 1
        routes["last_face"] = (key, cell)
        return out

    prov.eligible, prov.fn, prov.routes = eligible, fn, routes
    return prov


def _card_refusal(word: str) -> Optional[str]:
    """None when the core provider's ROW `word` can serve Boltz-2's C=128 bf16 pair stack on this process's card / stack; else the provider's refusal
    `<row>:<kind>` (its table, no launch: e.g. `tx_sm90a:cc:8.0!=9.0(…)` on an A100, `tx_sm90a:no_prebuilt:<abi>` on an H100 stack without the binary).
    Asked for a ROW word only (a tier word resolves per call and always names a row that serves)."""
    try:
        import torch
        from opt_core.kernels import trimul as KT
        if not torch.cuda.is_available():
            return None                                              # no device: nothing to ask (the lever's own eligibility names a CPU call)
        cc = tuple(torch.cuda.get_device_capability(0))
        stack = KT.stack_word(torch.empty(0, device="cuda")) if hasattr(KT, "stack_word") else None
        for d in ("outgoing", "incoming"):
            sel = KT.select(cc, "bf16", 128, 128, 512, d, word=word, stack=stack)
            KT.admits(sel.row, cc, "bf16", 128, 128, 512, backward=False, residual=False, batch=1,
                      abi=(KT.tx_abi_tag() if str(sel.row).startswith("tx_sm90a") and hasattr(KT, "tx_abi_tag") else None))
    except Exception as e:  # noqa: BLE001 — the provider's Refusal (by name) or an import problem
        row, kind = getattr(e, "row", None), getattr(e, "kind", None)
        return (f"{row}:{kind}" if kind else f"{type(e).__name__}:{e}").replace(" ", "_")[:160]
    return None


def _tier_facts() -> Optional[Dict[str, Any]]:
    """report()['tier']: the provider word's routes in this process — {word, face: {row: n}, aside: {library_op:<row>: n}, row, cell, kernel_rows}
    (row / cell: the face's row and cell that served last; None when nothing served through the face) — None before apply."""
    r = getattr(_STATE.get("provider"), "routes", None)
    if not r:
        return None
    last = r.get("last_face")
    return {"word": r["word"], "face": dict(r["face"]), "aside": dict(r["aside"]), "row": last[0] if last else None,
            "cell": last[1] if last else None, "kernel_rows": list(r["kernel_rows"])}


def _with_ceiling(T, prov, ceiling: Optional[int]):
    """Above `ceiling` tokens the provider refuses BY NAME (`above_max_tokens`): the engine's own TriMul serves that call, counted."""
    if ceiling is None:
        return prov
    inner = prov.eligible

    def eligible(call, _inner=inner, _c=int(ceiling)):
        if call.n_tokens > _c:
            raise T.Refused(ABOVE_MAX)
        return _inner(call)

    prov.eligible = eligible
    return prov


def apply(spec: Optional[str] = None) -> List[str]:
    """Patch both TriMul classes with the core ladder (idempotent). ``spec`` overrides the env switch ("1" applies)."""
    if _STATE["lever"] is not None:
        return list(_STATE["applied"])
    v = variant() if spec is None else (str(spec).strip().lower() if str(spec).strip().lower() in VARIANTS else None)
    if v is None:
        return []
    from opt_core import trimul as T
    import boltz.model.layers.triangular_mult as TM

    pw, mx = provider_word(), max_tokens()
    word = tier_word(v, pw)
    if word not in TIER_WORDS_OF[v] and word in ROW_WORDS_OF[v]:       # a ROW by name this card / stack cannot serve steps aside BY NAME at install (the lever is then a skipped lever of the row)
        why = _card_refusal(word)
        if why is not None:
            _STATE["disabled"] = {LEVERS_OF[v][0]: why}
            _STATE["provider_word"] = word
            sys.stderr.write(f"[{TAG}] {PROVIDER_ENV}={word}: REFUSED at install — {why}: the engine's TriMul serves\n")
            return []
    expected = tuple(EXPECTED_OF[v]) + ((ABOVE_MAX,) if mx is not None else ())
    prov = _with_ceiling(T, _provider(T, v, pw), mx)
    lever = T.Lever(TAG, "exact" if v == "exact" else "fast", provider=prov, min_tokens=0, expected=expected)
    _STATE["max_tokens"], _STATE["provider_word"] = mx, word
    orig_out, orig_in = TM.TriangleMultiplicationOutgoing.forward, TM.TriangleMultiplicationIncoming.forward

    def forward_outgoing(self, x, mask, use_kernels: bool = False):
        return lever.serve(T.Call(self, x, mask, "outgoing", residual=False, orig=lambda: orig_out(self, x, mask, use_kernels)))

    def forward_incoming(self, x, mask, use_kernels: bool = False):
        return lever.serve(T.Call(self, x, mask, "incoming", residual=False, orig=lambda: orig_in(self, x, mask, use_kernels)))

    TM.TriangleMultiplicationOutgoing.forward = forward_outgoing
    TM.TriangleMultiplicationIncoming.forward = forward_incoming
    lever.register_exit_line()
    _STATE.update(lever=lever, provider=prov, applied=list(LEVERS_OF[v]), variant=v, patched=["TriangleMultiplicationOutgoing.forward", "TriangleMultiplicationIncoming.forward"])
    sys.stderr.write(lever.line() + "\n")
    return list(LEVERS_OF[v])


UNEXPECTED_FALLBACK = "unexpected fallback"   # opt_core.trimul.Lever.gate's reason word for a fallback outside the kit's ``expected`` list
KERNEL_ERRORS = "kernel errors"               # … and for a counted kernel error; its third and last refusal is 'served 0 of N calls'


def verdict(lever) -> Dict[str, Any]:
    """The core ladder's fail-closed gate (``Lever.gate()``: refused on kernel errors, then on a fallback outside ``expected``, then on a routed
    kernel that served none of the calls that arrived), with one engine rule: a run whose EVERY call took a declared path to the engine's own
    TriMul (census: served 0, no errors, and the gate's refusal is neither the kernel-error nor the unexpected-fallback word — an input whose
    every cell names the library op as its exact row on this card) is the installed lever idling, ``{ok: True, idle: True}``, the same standing
    as the flash lever below its gate (the core passes that census itself as a named aside — ``aside`` carries its word); a kernel error or an undeclared
    fallback is the core's refusal unchanged. Only the gate's public verdict and census are read."""
    g = lever.gate(); c = lever.census()
    reason = str(g.reason or "")
    details = getattr(g, "details", None)
    aside = details.get("aside") if isinstance(details, dict) else None            # a zero-served census whose every call was ANSWERED with the cell's stock
    for w in (getattr(g, "words", None) or ()):                                    #  row by name is the core's own named aside (gate ok, words 'aside:<word>') — read it back so the
        if aside is None and str(w).startswith("aside:"):                          #  report's idle word stays truthful on either core
            aside = str(w)[len("aside:"):]
    declared_idle = (c["served"] == 0 and not c["errors"] and bool(c["fallback"]) and sum(int(n) for n in c["fallback"].values()) > 0
                     and not reason.startswith(UNEXPECTED_FALLBACK) and not reason.startswith(KERNEL_ERRORS))
    if declared_idle:                                                              # a core that REFUSES that census ('mode <m> routed … served 0 of N'): the engine rule opens it;
        out = {"ok": True, "idle": True, "reason": IDLE, "core_reason": g.reason}   #  newer cores pass it as a named aside: idle either way, the core's words carried
        if aside is not None:
            out["aside"] = aside
        return out
    return {"ok": bool(g.ok), "idle": False, "reason": g.reason}


def report() -> Dict[str, Any]:
    lever = _STATE["lever"]
    if lever is None:
        return {"applied": [], "disabled": dict(_STATE.get("disabled") or {}), "line": None, "census": None, "gate": None, "patched": [],
                "provider_word": _STATE.get("provider_word")}
    return {"applied": list(_STATE["applied"]), "disabled": {}, "line": lever.line(), "census": lever.census(), "gate": verdict(lever),
            "patched": list(_STATE["patched"]), "min_tokens": 0, "max_tokens": _STATE.get("max_tokens"), "provider_word": _STATE.get("provider_word"),
            "expected": list(lever.expected), "variant": _STATE["variant"], "tier": _tier_facts()}
