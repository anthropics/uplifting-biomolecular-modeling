"""ptx_trimul_routes — the pair-stack triangle multiplication (c_z 256: the 48 pairformer blocks, the MSA module's pair stack, the confidence
head's pairformer) bound to the shared core's ONE TriMul provider (``opt_core.kernels.trimul``: every carried row + the measured cell table)
BY TIER WORD.  The kit carries no TriMul kernel of its own for these sites: the tier word serves
the row the provider's cell table names for the call's cell on this card (under exact: a row byte-vouched on this stack, else the library
op by name) — whatever the table returns IS the binding; no row
is pinned here.

  lever               switch (env.sh)      class      binding
  trimul_core_exact   PTX_TRIMUL=exact     EXACT      ARM E: ``opt_core.trimul.by_word(weights_of, 'exact')`` — the cell's exact tier row
  trimul_core         PTX_TRIMUL=tier      TOLERANCE  ARM T: ``by_word(weights_of, <tier>)`` — ``big`` under the kit mode big (PROTENIX_OPT), else ``fast``

An absent word installs nothing (``MODEL_OPT_LEVERS_OFF=trimul_core`` / ``trimul_core_exact`` removes the switch after env.sh): every call is
the arm's stock forward, counted ``off``.  An unknown word steps aside BY NAME (``aside:unknown_word:<w>``), never silently.

Sites: every ``TriangleMultiplication{Outgoing,Incoming}`` call the ``fpf`` registry routes here (FPF_OPS ``trimul_out`` / ``trimul_in``) whose
z has 256 channels on the library branch (``triangle_multiplicative == 'cuequivariance'``) goes through ONE ``opt_core.trimul.Lever`` (min_tokens
101: the library routes N <= 100 to its torch branch, those calls keep the stock forward, counted ``below_min_tokens``).  The template stack
(c = 64) is ``ptx_c64_routes``' (composed AFTER this module in FPF_OPS: it serves c 64 and hands every other call to this module).  A call this
module does not target (another channel count, the torch branch, an unknown call form) is the stock forward, counted ``passthrough``.
Under ARM T ``fpf_smalln`` keeps its by-number size gate (PTX_T_MIN_TOKENS, 300): below it the TriMul callee is ``trimul_exact_c256`` (the provider's
``exact`` word — the small-N route of fast / big is the exact tier) and at / above it ``trimul_c256`` (the kit mode's tier word); both are
Levers of this module over the same provider, so the kit carries no TriMul kernel and pins no row on either side of the gate.

Lines (stdout; '[ptx_trimul_routes] …'):
    APPLIED word=<exact|tier|off|aside:why> bind=<tier:exact|tier:fast|tier:big|none> min_tokens=101 opt_core=<version>@<file>
    FIRST trimul <outgoing|incoming> N=.. C=256 dtype=.. residual=.. mask=.. selection=[<kernels.trimul describe(selection)>]
    at exit, the Lever's census line + this module's words:
    LEVER name=F2.trimul state=on impl=trimul:<row>@<word>/<cell> origin=core served=<n> fallback=<n> fallback_by={..} min_tokens=101 shapes=.. mode=<exact|fast> errors={} gate=ok word=<w> bind=tier:<w> lever=<registry lever> passthrough=<n>
    (+ under ARM T a second line for the below-gate exact Lever: '[ptx_trimul_routes.smalln_exact] LEVER name=F2.trimul … bind=tier:exact lever=trimul_core smalln_exact_calls=<n>')
``report()`` returns the same census as plain data (ptx_trunk2_levers._STATS['trimul_routes'] carries it into the kit's lever report).
"""
import os
import sys
import threading

TAG = "ptx_trimul_routes"
WORD_ENV = "PTX_TRIMUL"                               # env.sh: exact under ARM E, tier under ARM T
MODE_ENV = "PROTENIX_OPT"                             # the kit mode word of this process (protenix_opt.stack exports it at activation)
WORDS = {"exact": ("exact", "trimul_core_exact"),     # switch word -> (opt_core.trimul.Lever class word, registry lever)
         "tier": ("fast", "trimul_core")}
ARM_T_TIERS = ("fast", "big")
CHANNELS = 256
MIN_TOKENS = 101
STATE = {"word": "off", "bind": "none", "lever": None, "opt_core": None, "first": {}, "passthrough": 0, "off_calls": 0, "lever_calls": 0, "smalln_exact_calls": 0}
_LEVER = None                 # the word's Lever: exact under PTX_TRIMUL=exact, the kit mode's tier word under PTX_TRIMUL=tier
_ATEXIT = False
_EXACT_LEVER = None           # under PTX_TRIMUL=tier only: the provider's exact word for fpf_smalln's below-gate callee (trimul_exact_c256); under =exact it IS _LEVER
_LOCK = threading.Lock()


def _say(msg):
    print(f"[{TAG}] {msg}", flush=True)


def _tok(x) -> str:
    return "_".join(str(x).split()) or "-"


def tier_word(environ=None) -> str:
    """The core tier word ARM T binds: the kit mode's own word when it is fast | big (PROTENIX_OPT), else fast."""
    m = ((environ if environ is not None else os.environ).get(MODE_ENV, "") or "").strip().lower()
    return m if m in ARM_T_TIERS else "fast"


def weights_of(m):
    """The ten canonical tensors (opt_core.trimul.WEIGHT_KEYS) of a protenix TriangleMultiplication{Outgoing,Incoming} (bias-free linear layers)."""
    return dict(ln_in_w=m.layer_norm_in.weight, ln_in_b=m.layer_norm_in.bias, w_ag=m.linear_a_g.weight, w_ap=m.linear_a_p.weight,
                w_bg=m.linear_b_g.weight, w_bp=m.linear_b_p.weight, ln_out_w=m.layer_norm_out.weight, ln_out_b=m.layer_norm_out.bias,
                w_o=m.linear_z.weight, w_og=m.linear_g.weight)


def _capturing() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _stock(op: str):
    import fpf
    return fpf.original(op)


def _trimul_args(z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch", **rest):
    return z, mask, inplace_safe, _add_with_inplace, triangle_multiplicative, rest


def _direction(module, op=None) -> str:
    if op == "trimul_out":
        return "outgoing"
    if op == "trimul_in":
        return "incoming"
    return "outgoing" if bool(getattr(module, "_outgoing", True)) else "incoming"


def _serve(op, module, a, kw, which="word"):
    """One TriMul call in the stock signature: the Lever for a targeted call, the stock forward otherwise (counted).  ``which``: 'word' = the
    switch word's Lever; 'exact' = the provider's exact word (fpf_smalln's below-gate callee under ARM T; the same Lever under ARM E)."""
    direction = _direction(module, op)
    stock_op = op or ("trimul_out" if direction == "outgoing" else "trimul_in")

    def orig():
        return _stock(stock_op)(module, *a, **kw)
    lever = (_EXACT_LEVER or _LEVER) if which == "exact" else _LEVER
    if lever is None:
        STATE["off_calls"] += 1
        return orig()
    if which == "exact":
        STATE["smalln_exact_calls"] += 1
    try:
        z, mask, inplace_safe, add_inplace, tm, rest = _trimul_args(*a, **kw)
    except TypeError:                                       # a call form this module does not know: the stock forward's to handle
        STATE["passthrough"] += 1
        return orig()
    if rest or tm != "cuequivariance" or getattr(z, "dim", lambda: 0)() < 3 or int(z.shape[-1]) != CHANNELS:
        STATE["passthrough"] += 1
        return orig()
    from opt_core import trimul as T
    residual = bool(inplace_safe is True and add_inplace)   # the library branch returns z + update exactly then, else the bare update: the row fuses the same add
    call = T.Call(module, z, mask, direction, residual=residual, orig=orig)
    STATE["lever_calls"] += 1
    out = lever.serve(call)
    if direction not in STATE["first"] and not _capturing():
        with _LOCK:
            if direction not in STATE["first"]:
                try:
                    from opt_core.kernels import trimul as KT
                    sel = call.extra.get("trimul_selection")
                    desc = KT.describe(sel) if sel is not None else "none(the stock forward served: see fallback_by)"
                    STATE["first"][direction] = f = {"N": int(z.shape[-2]), "C": int(z.shape[-1]), "dtype": str(z.dtype).replace("torch.", ""), "residual": residual,
                                                     "mask": None if mask is None else list(mask.shape), "row": getattr(sel, "row", None), "cell": getattr(sel, "cell", None), "selection": desc}
                    _say(f"FIRST trimul {direction} N={f['N']} C={f['C']} dtype={f['dtype']} residual={residual} mask={f['mask']} selection=[{desc}]")
                except Exception as e:  # noqa: BLE001 — a report line, never a failure of the call
                    STATE["first"][direction] = {"error": repr(e)}
                    _say(f"FIRST trimul {direction} facts unavailable: {e!r}")
    return out


def _make(op, which="word"):
    def provider(module, *a, **kw):
        return _serve(op, module, a, kw, which)
    provider.__name__ = provider.__qualname__ = op or ("trimul_exact_c256" if which == "exact" else "trimul_c256")
    provider._ptx_trimul_routes = True
    return provider


trimul_out = _make("trimul_out")        # FPF_OPS entries (ARM E): trimul_out=ptx_trimul_routes:trimul_out,trimul_in=ptx_trimul_routes:trimul_in
trimul_in = _make("trimul_in")
trimul_c256 = _make(None)               # fpf_smalln's at/above-gate callee under ARM T (FPF_SMALLN_TRIMUL_FAST_FN): the kit mode's tier word; direction from the module
trimul_exact_c256 = _make(None, "exact")   # fpf_smalln's below-gate callee under ARM T (FPF_SMALLN_TRIMUL_EXACT_FN): the provider's exact word (bitwise == the stock forward where vouched; the library op by name elsewhere)
fn = trimul_c256


def _line() -> str:
    from opt_core import report as R
    lever_name = STATE.get("lever") or "trimul_core"
    if _LEVER is None:
        why = STATE["word"] if str(STATE["word"]).startswith("aside:") else f"word:{STATE['word']}"
        return R.lever_line(TAG, "F2.trimul", "skipped", reason=_tok(why), impl="none", origin="core", bind=STATE.get("bind", "none"), lever=lever_name,
                            passthrough=STATE["passthrough"], off_calls=STATE["off_calls"])
    return _LEVER.line() + " " + R.kv(("word", STATE["word"]), ("bind", STATE.get("bind", "none")), ("lever", lever_name), ("passthrough", STATE["passthrough"]))


def _exit_line():
    try:
        ln = _line()
        _say(ln.split("] ", 1)[-1] if ln.startswith("[") else ln)
        if _EXACT_LEVER is not None:                     # ARM T: the below-gate exact Lever's own census line
            from opt_core import report as R
            lx = _EXACT_LEVER.line() + " " + R.kv(("word", "exact"), ("bind", "tier:exact"), ("lever", STATE.get("lever") or "trimul_core"), ("smalln_exact_calls", STATE["smalln_exact_calls"]))
            _say(lx.split("] ", 1)[-1] if lx.startswith("[") else lx)
    except Exception as e:  # noqa: BLE001
        _say(f"LEVER name=F2.trimul state=unknown reason=line_failed:{_tok(repr(e))[:120]}")


def apply(environ=None) -> dict:
    """Build the Lever from the word (idempotent per process); prints the APPLIED line."""
    global _LEVER, _EXACT_LEVER, _ATEXIT
    environ = os.environ if environ is None else environ
    _LEVER = _EXACT_LEVER = None
    STATE.update(word="off", bind="none", lever=None)
    w = (environ.get(WORD_ENV, "") or "").strip().lower()
    try:
        import opt_core
        core_v, core_f = getattr(opt_core, "__version__", None), getattr(opt_core, "__file__", None)
    except Exception as e:  # noqa: BLE001
        core_v, core_f = None, f"import-failed:{e!r}"
    STATE["opt_core"] = {"version": core_v, "file": core_f}
    if not w:
        STATE["word"] = "off"
    elif w not in WORDS:
        STATE["word"] = f"aside:unknown_word:{_tok(w)}"
    elif core_v is None:
        STATE["word"] = f"aside:opt_core_unavailable:{_tok(core_f)[:80]}"
    else:
        try:
            from opt_core import trimul as T
            mode_w, lever_name = WORDS[w]
            word = tier_word(environ) if w == "tier" else "exact"
            prov = T.by_word(weights_of, word, name=f"trimul:{word}")
            _LEVER = T.Lever(TAG, mode_w, provider=prov, min_tokens=MIN_TOKENS, expected=("mode_stock", "below_min_tokens"))
            if w == "tier":                                  # fpf_smalln's below-gate callee: the provider's exact word, its own Lever and census
                _EXACT_LEVER = T.Lever(TAG + ".smalln_exact", "exact", provider=T.by_word(weights_of, "exact", name="trimul:exact"), min_tokens=MIN_TOKENS, expected=("mode_stock", "below_min_tokens"))
            STATE["word"], STATE["bind"], STATE["lever"] = w, f"tier:{word}", lever_name
        except Exception as e:  # noqa: BLE001 — a core that cannot build the lever: aside by name, the stock forward serves
            STATE["word"] = f"aside:{type(e).__name__}:{_tok(e)[:80]}"
    if not _ATEXIT:
        try:
            import atexit
            atexit.register(_exit_line)
            _ATEXIT = True
        except Exception:  # noqa: BLE001
            pass
    try:                                                    # the kit's lever report reads the trunk levers' tally: carry this census there by name
        LEV = sys.modules.get("ptx_trunk2_levers")
        if LEV is not None and isinstance(getattr(LEV, "_STATS", None), dict):
            LEV._STATS["trimul_routes"] = STATE
    except Exception:  # noqa: BLE001
        pass
    _say(f"APPLIED word={STATE['word']} bind={STATE['bind']} min_tokens={MIN_TOKENS} opt_core={core_v}@{core_f}")
    return STATE


def report() -> dict:
    """Plain-data census for a manifest: the word, the binding, the core, the Lever's counters."""
    out = {k: v for k, v in STATE.items()}
    out["census"] = _LEVER.census() if _LEVER is not None else None
    out["smalln_exact_census"] = _EXACT_LEVER.census() if _EXACT_LEVER is not None else None
    return out


if os.environ.get("PTX_TRIMUL_ROUTES_NO_AUTOAPPLY", "") != "1":      # the kit process imports this module through FPF_OPS / fpf_smalln: apply at import (tests import it with the guard set)
    apply()
