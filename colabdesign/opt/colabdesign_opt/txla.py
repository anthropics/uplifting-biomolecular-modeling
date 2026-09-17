"""Lever `txla` — opt_core's FORWARD-ONLY triangle-attention bridge `opt_core.kernels.triattn_xla` (TX: an XLA-FFI custom call over the
core's triangle-attention kernels — CUDA kernels and Triton ahead-of-time cubins per compute capability (9.0, 8.x); 16 or 32 per head) on the
design step's FORWARD-ONLY attention calls, composed on lever `triatt`. The row of every call is THE FACE'S OWN MEASURED TABLE: this module
hands the face `impl='auto'` on every call and names no row and no size (the face's CELLS.json holds the rows' size / head-dim / card rules);
`rows=` on the line is the face's own census of what it bound.

Mechanism. The op the serve layer (`opt_core.kernels.pallas_attn_serve`, the tree's ONE haiku `Attention` interception) calls for every
served `Attention` call of the design model becomes a `jax.custom_vjp` function whose

* PRIMAL   calls `TX.triangle_attention(bf16(q*scale), k, v, pair_bias[H,Sq,Sk], key_mask, 1.0, impl='auto', layout='BNHSD')` — q, k, v arrive
             heads-major `[B rows, H, S, D]` exactly as the served call (and lever `proj`) lay them, which IS TX's `BNHSD` layout (no transpose);
             q is scaled once and rounded to its dtype exactly where `triatt` (and stock) round it; the key mask goes through
             `lax.cond(all(key_mask), <mask-free call>, <u8 key-row call>)` (the design model's masks are all ones except the extra-MSA row's,
             so the mask-free kernel runs where it can and a masked call is still honoured);
* FORWARD  rule is `jax.vjp` of the `triatt` op (its output, its residuals) and whose
* BACKWARD rule is that vjp — i.e. under `jax.grad` (ColabDesign's `grad_fn`, the graded design step, hk.remat included) JAX takes the fwd/bwd
             rules = lever `triatt`'s forward and backward kernels exactly as `triatt` runs them, and with no differentiation (ColabDesign's `fn`:
             recycle 0, `backprop=False`; BindCraft's forward-only `predict` calls) the primal = the bridge serves.

So `grad_fn` is unchanged bit for bit against `fast` without this lever, and `fn`
changes precision class only (a different forward kernel of the same mathematics: online softmax, bf16 products, fp32 statistics). Calls with
fewer than 16 queries or keys (the MSA column attention over 2 sequences) or a head dim below 16 that reaches the op unpadded go to the `triatt`
op unchanged (its own plain-XLA small path / zero padding), counted `small_call` (declared); a call TX refuses BY NAME at trace time
(`TX.Refused`: a shape / dtype / card no row serves, a row switched off by `MODEL_OPT_LEVERS_OFF=triattn_xla[:row]`, a binary that does not
load) goes to the `triatt` op unchanged, counted `refused` with TX's own reason text kept (evidence()) — the op contract op(q, k, v,
pair_bias, key_mask, scale) = "applies exactly `scale`" holds on every path.

Binding. Installed AFTER `triatt` (and after `proj`, which composes on the serve layer's record at trace time): the modules package is re-served
through the same seam with triatt's own ledger, scope and rules and op = the bridged op (`pallas_attn_serve.disable` + `enable(op=…)`), and
`proj`, when it was stacked on the class re-served, is stacked again (its install is idempotent by marker). Without `triatt` serving the package
(`fast-no-triatt`: pallas restored; or no attention-kernel lever) install() raises its named refusal `NeedsTriatt` (and `TxUnavailable` when the
bridge cannot be imported): levers.install prints the lever's ONE line `state=skipped reason=cannot_run detail=NeedsTriatt:… source=install` and the
mode runs the rest of its set (the kit's step-aside protocol, REFUSALS) — never silent, never a refusal of the mode. Ablation: the mode word `fast-no-txla`.

Numerics class `precision` (never bitwise vs stock or vs `fast-no-txla`: a different forward kernel in recycle 0 feeds a different `prev` into
the graded step; run-to-run bitwise). Evidence: ONE LEVER line at process exit through the package's exit printer (kernels.register_exit_line;
an atexit printer while the registry does not name the lever), registered once install engaged:

    [colabdesign-opt] LEVER name=txla state=on impl=triattn_xla@<core version> origin=core served=<n> fallback=<n> fallback_by=<small_call:n,refused:n|none>
                      shapes=<BxHxSxD:n,…> rows=<row:n,…|none> served_fn=<n> served_grad=0 vjp_traced=<n> vmap=<n> refused=<shape:n,…|none>
                      layout=BNHSD impl_word=auto requires=triatt grad=triatt proj_restacked=<0|1> numerics=precision precision=bf16 cc=<cc> tx=<TX version> source=exit pid=<pid>

`served` / `served_fn` count TRACED primal calls the bridge took (8 per forward-only program of the design model: triangle start/end ×2 in the
Evoformer and ×2 in the extra-MSA stack, ×2 in the template pair stack, the extra-MSA row attention padded to 16, the MSA row attention);
`rows` is TX's own census of kernel binds (each served site binds both mask branches); `served_grad=0` by construction (the differentiated
sites, `vjp_traced`, run triatt); `vmap` counts primal traces under jax.vmap / hk.vmap (the XLA-FFI call has no batching rule of its own: a
`jax.custom_batching.custom_vmap` face serves a mapped caller slice by slice through lax.map — the design model maps none today). Lever protocol (registry.py): NUMERICS, REFUSALS, install() -> dict, installed(), uninstall(), off_line(reason), evidence().
"""
from __future__ import annotations

import atexit
import functools
import os
from typing import Callable, Dict, Optional, Tuple

from opt_core import report as _core_report
from opt_core.counters import Ledger
from opt_core.kernels import pallas_attn_serve as F1

from .names import TAG

LEVER = "txla"                                     # registry id = the LEVER line's name=
NUMERICS = "precision"                             # a different forward kernel of the same mathematics on the forward-only calls; grad_fn unchanged
REQUIRES = "triatt"                                # composes on lever triatt's op (its fwd/bwd rules ARE triatt); steps aside by name without it
IMPL_WORD = "auto"                                 # the face's routing word on EVERY call: its own measured table (CELLS.json by_cc order + rules: rows per compute
                                                   # capability / dtype / head dim / queries / mask form) names the row — this module names no row and no size; `rows=` on
                                                   # the line is the face's own census of what it bound
LAYOUT = "BNHSD"                                   # heads-major [rows, H, S, D]: what the served call and lever proj hand an op — no transpose
PRECISION = "bf16"
MIN_SEQ = 16                                       # = kernels/triatt_attn.MIN_SEQ: fewer queries or keys -> the triatt op's own plain-XLA small path (`small_call`)
MIN_HEAD_DIM = 16                                  # = kernels/triatt_attn.MIN_HEAD_DIM: an unpadded head dim below it -> the triatt op's own zero padding (`small_call`)
PAD_HEAD_DIM_BELOW_MIN = True                      # = triatt_lever.install's serve-layer rule (head dims below 16 zero-padded by the seam before the op)
SMALL_CALL = "small_call"                          # fallback reason: S_q / S_k < MIN_SEQ or D < MIN_HEAD_DIM at the op -> triatt op unchanged (declared, by design)
REFUSED = "refused"                                # fallback reason: TX.Refused at trace time -> triatt op unchanged (by name; TX's text kept in evidence())
NEEDS_TRIATT = "needs_triatt"                      # info()/evidence() word when install() raised NeedsTriatt (the printed line is levers.install's: reason=cannot_run detail=NeedsTriatt:…)
TX_UNAVAILABLE = "tx_unavailable"                  # … when install() raised TxUnavailable (opt_core.kernels.triattn_xla not importable in this process)


class NeedsTriatt(RuntimeError):
    """Lever triatt does not serve colabdesign's modules package in this run (fast-no-triatt restores pallas; no attention-kernel lever): txla composes on triatt's op only."""


class TxUnavailable(RuntimeError):
    """`opt_core.kernels.triattn_xla` cannot be imported in this process (a core below the kit's floor, a broken install)."""

EXPECTED_FALLBACKS = (SMALL_CALL, REFUSED)         # both hand the call to triatt BY NAME; neither is a defect of this lever
REFUSALS = (NeedsTriatt, TxUnavailable)            # cannot-run HERE, by name: levers.install prints `state=skipped reason=cannot_run detail=<Class>:<text> source=install` and the mode runs the rest of its set
ATTENTION_FWD_KERNELS = ()                         # not an attention-kernel lever of its own (levers.ATTENTION_KERNEL_LEVERS reads this attribute): under jax.grad the forward is triatt's kernels
MARKER = "_colabdesign_opt_txla"                   # attribute on the bridged op object (installed() reads the serve layer's record for it)
LEDGER: Optional[Ledger] = None
_STATE: Dict[str, object] = {"skipped": None, "installed_on": None, "op": None, "tri_op": None, "proj_restacked": 0, "cc": None,
                             "vjp_traced": 0, "vmap_traced": 0, "refusals": {}, "refused_shapes": {}, "rows_seen": {}, "exit_registered": False, "atexit": False}


def core_version() -> str:
    try:
        import opt_core  # noqa: WPS433
        return str(getattr(opt_core, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


def impl_word() -> str:
    """The line's impl=: the kernel lives in opt_core (origin=core rule: impl=<mechanism>@<core version>)."""
    return f"triattn_xla@{core_version()}"


def tx_version() -> str:
    try:
        from opt_core.kernels import triattn_xla as TX  # noqa: WPS433
        return str(getattr(TX, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unavailable"


def _ledger() -> Ledger:
    global LEDGER
    if LEDGER is None:
        LEDGER = Ledger(LEVER, impl=impl_word(), origin="core", expected=EXPECTED_FALLBACKS)      # no size rule of this module: the face's table holds the rows' size rules
    return LEDGER


def route(n_queries: int, n_keys: int, head_dim_q: int, head_dim_v: int) -> str:
    """The per-call rule of the op as ONE function: `small_call` (-> the triatt op unchanged) or `bridge` (-> the custom_vjp: TX primal by the
    face's table, triatt rules)."""
    if int(n_queries) < MIN_SEQ or int(n_keys) < MIN_SEQ or int(head_dim_q) < MIN_HEAD_DIM or int(head_dim_v) < MIN_HEAD_DIM:
        return SMALL_CALL
    return "bridge"


def _cc() -> str:
    if _STATE["cc"] is None:
        from opt_core.kernels import triattn_xla as TX  # noqa: WPS433
        _STATE["cc"] = TX.compute_capability()
    return str(_STATE["cc"])


def select_row(TX, cc: str, dtype, D: int, Sq: int, Sk: int, B: int, H: int, bias_dtype, has_mask: bool, impl: str = IMPL_WORD) -> str:
    """The row the face will run for one arm (mask-free | masked) of a call under `impl` (IMPL_WORD `auto`: the face's own table). Raises
    TX.Refused when NO row serves the arm (the caller hands the call to triatt by name)."""
    r, _cell = TX.select(cc, dtype, D, Sq, N=B, H=H, B=1, has_mask=has_mask, impl=impl, bias_dtype=bias_dtype, SK=Sk)
    return r


def make_op(tri_op: Callable, impl: str = IMPL_WORD) -> Callable:
    """Build the served op `op(q, k, v, pair_bias, key_mask, scale) -> [B, H, S_q, D_v]` (the serve layer's / proj's op contract: q, k, v
    heads-major `[B, H, S, D]`, pair_bias `[H, S_q, S_k]` in q's dtype, key_mask `[B, S_k]` bool, scale float applied exactly once):
    a jax.custom_vjp whose primal is TX (row per call = the face's table under `impl`, default IMPL_WORD `auto`; a row word given here is
    handed to the face for every call — tests) and whose fwd / bwd rules are `tri_op`'s forward (+ residuals) and backward."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    from opt_core.kernels import triattn_xla as TX  # noqa: WPS433

    led = _ledger()

    faces: Dict[Tuple[str, str], Callable] = {}

    def _face(w0: str, w1: str) -> Callable:
        """The bridge call proper for one (mask-free word, masked word) pair: the mask-free kernel when the key mask is all ones, the u8
        key-row call otherwise — each arm with the row the face's table named for it."""
        if (w0, w1) in faces:
            return faces[(w0, w1)]

        def _face_plain(qq, k, v, bias, key_mask):
            free = lambda: TX.triangle_attention(qq, k, v, bias, None, 1.0, impl=w0, layout=LAYOUT)  # noqa: E731
            masked = lambda: TX.triangle_attention(qq, k, v, bias, key_mask.astype(jnp.uint8), 1.0, impl=w1, layout=LAYOUT)  # noqa: E731
            return jax.lax.cond(jnp.all(key_mask), free, masked)

        face = jax.custom_batching.custom_vmap(_face_plain)                          # the XLA-FFI call beneath: a caller under jax.vmap / hk.vmap is served slice by slice

        @face.def_vmap                                                                # (lax.map over the mapped axis, the same kernel per slice), counted vmap=
        def _face_vmap(axis_size, in_batched, qq, k, v, bias, key_mask):            # (the design model maps no attention call today: every program traces without this rule)
            _STATE["vmap_traced"] = int(_STATE["vmap_traced"]) + 1
            args = (qq, k, v, bias, key_mask)
            idx = [i for i, b in enumerate(in_batched) if b]

            def body(ms):
                full = list(args)
                for i, m in zip(idx, ms):
                    full[i] = m
                return _face_plain(*full)
            return jax.lax.map(body, [args[i] for i in idx]), True

        faces[(w0, w1)] = face
        return face

    def _primal(q, k, v, bias, key_mask, scale):
        B, H, Sq, D = (int(x) for x in q.shape)
        Sk = int(k.shape[2])
        key = F1.shape_key(q)
        cc = _cc()
        try:
            r0 = select_row(TX, cc, q.dtype, D, Sq, Sk, B, H, bias.dtype, False, impl)    # the face's row per arm under its own table (impl=auto)
            r1 = select_row(TX, cc, q.dtype, D, Sq, Sk, B, H, bias.dtype, True, impl)
        except TX.Refused as e:                                                    # BY NAME: no row serves this call here — triatt's op takes it, unchanged
            led.fallback(REFUSED)
            _STATE["refused_shapes"][key] = int(_STATE["refused_shapes"].get(key, 0)) + 1
            _STATE["refusals"][key] = str(e)[:400]
            return None
        led.serve(key)
        for r in {r0, r1}:
            _STATE["rows_seen"][r] = int(_STATE["rows_seen"].get(r, 0)) + 1
        qq = (q.astype(jnp.float32) * float(scale)).astype(q.dtype)                # stock's / triatt's rounding point: dtype(q * key_dim**-0.5) once; TX gets scale 1.0
        return _face(r0, r1)(qq, k, v, bias, key_mask)

    @functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
    def bridged(q, k, v, bias, key_mask, scale):
        out = _primal(q, k, v, bias, key_mask, scale)
        return tri_op(q, k, v, bias, key_mask, scale) if out is None else out

    def _fwd(q, k, v, bias, key_mask, scale):                                      # under jax.grad: triatt's forward, its residuals = its own vjp closure
        _STATE["vjp_traced"] = int(_STATE["vjp_traced"]) + 1
        out, vjp_fn = jax.vjp(lambda q_, k_, v_, b_: tri_op(q_, k_, v_, b_, key_mask, scale), q, k, v, bias)
        return out, vjp_fn

    def _bwd(scale, vjp_fn, do):                                                   # triatt's backward (attbwd_dkdv + dq), exactly as lever triatt runs it
        dq, dk, dv, db = vjp_fn(do)
        return dq, dk, dv, db, None

    bridged.defvjp(_fwd, _bwd)

    def op(q, k, v, bias, key_mask, scale):
        """q, k, v [B, H, S, D]; bias [H, S_q, S_k]; key_mask [B, S_k] bool; scale float -> o [B, H, S_q, D_v] in q's dtype."""
        why = route(q.shape[2], k.shape[2], q.shape[-1], v.shape[-1])
        if why != "bridge":                                                        # small_call: triatt's op, unchanged
            led.fallback(why)
            return tri_op(q, k, v, bias, key_mask, scale)
        return bridged(q, k, v, bias, key_mask, float(scale))

    op.__name__ = "txla_bridged_attention"
    op.__qualname__ = "txla_bridged_attention"
    setattr(op, MARKER, True)
    op.tri_op = tri_op
    op.knobs = dict(getattr(tri_op, "knobs", {}) or {})                            # triatt_attn.describe_op reads .knobs: the graded step's kernels are triatt's, so are the knobs
    op.impl = impl
    return op


def _modules_pkg():
    from colabdesign.af.alphafold.model import modules as CDM  # noqa: WPS433
    return CDM


def triatt_record(modules_pkg=None) -> Optional[dict]:
    """The serve layer's record for colabdesign's `modules` package IF lever triatt serves it ({"module", "stock_cls", "ledger", "op"}), else None."""
    from .kernels import triatt_lever as TL  # noqa: WPS433
    m = modules_pkg if modules_pkg is not None else _modules_pkg()
    rec = F1._ENABLED.get(id(m))
    if rec is None or TL.LEDGER is None or rec.get("ledger") is not TL.LEDGER:
        return None
    A = getattr(m, "Attention", None)
    if A is None or not getattr(A, TL.MARKER, False):
        return None
    return rec


def _register_exit() -> None:
    """The package's one exit printer prints this lever's line (registry order) when the registry names the lever; otherwise an atexit printer does."""
    if _STATE["exit_registered"] or _STATE["atexit"]:
        return
    try:
        from . import registry  # noqa: WPS433
        if LEVER in registry.LEVERS:
            from .kernels import register_exit_line  # noqa: WPS433
            register_exit_line(LEVER, exit_line)
            _STATE["exit_registered"] = True
            return
    except (ImportError, ValueError):
        pass
    atexit.register(lambda: _core_report.emit(exit_line()))
    _STATE["atexit"] = True


def install() -> dict:
    """Re-serve `colabdesign.af.alphafold.model.modules.Attention` through the serve layer with lever triatt's ledger, scope and rules and
    op = the bridged op; re-stack lever proj when it was stacked on the class re-served; register the exit line. Raises NeedsTriatt /
    TxUnavailable (REFUSALS: levers.install steps the lever aside by name, `state=skipped reason=cannot_run detail=…`) when triatt does not
    serve the package or TX cannot be imported. Idempotent. Call before any model function is traced. Returns the install facts."""
    _ledger()
    if installed():
        return info()
    CDM = _modules_pkg()
    rec = triatt_record(CDM)
    tri_op = rec.get("op") if rec is not None else None                            # triatt always enables the seam with its own op; a record without one is not triatt's
    if tri_op is None:
        _STATE["skipped"] = NEEDS_TRIATT
        raise NeedsTriatt(f"lever triatt does not serve {CDM.__name__} in this run (txla composes on triatt's op; fast-no-triatt restores pallas)")
    try:
        from opt_core.kernels import triattn_xla as TX  # noqa: WPS433, F401
    except Exception as e:  # noqa: BLE001 — a core without the bridge / a broken install: named, the mode runs the rest of its set
        _STATE["skipped"] = TX_UNAVAILABLE
        raise TxUnavailable(f"{type(e).__name__}: {e}") from e
    from .kernels import proj_attn as PJ, triatt_lever as TL  # noqa: WPS433
    _register_exit()                                                               # the exit line: only for a process where the lever engaged (a step-aside's ONE line is levers.install's)
    op = make_op(tri_op, IMPL_WORD)
    proj_on = bool(PJ.installed())
    F1.disable([CDM])
    F1.enable([CDM], ledger=TL.LEDGER, all_calls=(TL.SCOPE == "all_calls"), op=op, pad_head_dim_below_min=PAD_HEAD_DIM_BELOW_MIN, min_keys=TL.MIN_KEYS)
    if proj_on:
        PJ.install()                                                               # proj composes on the serve layer's record at trace time; its class sat on the one we replaced
    _STATE.update({"skipped": None, "installed_on": CDM, "op": op, "tri_op": tri_op, "proj_restacked": int(proj_on)})
    return info()


def uninstall() -> bool:
    """Put lever triatt's own op back on the serve layer (and proj back on top when it was there); re-trace afterwards. True when something was restored."""
    import sys
    CDM = sys.modules.get("colabdesign.af.alphafold.model.modules")
    if CDM is None or not installed():
        _STATE["installed_on"] = None
        return False
    from .kernels import proj_attn as PJ, triatt_lever as TL  # noqa: WPS433
    proj_on = bool(PJ.installed())
    F1.disable([CDM])
    F1.enable([CDM], ledger=TL.LEDGER, all_calls=(TL.SCOPE == "all_calls"), op=_STATE.get("tri_op"), pad_head_dim_below_min=PAD_HEAD_DIM_BELOW_MIN, min_keys=TL.MIN_KEYS)
    if proj_on:
        PJ.install()
    _STATE["installed_on"] = None
    return True


def installed() -> bool:
    """THIS lever's op is the one the serve layer calls for colabdesign's modules package (triatt's ledger behind it)."""
    import sys
    CDM = sys.modules.get("colabdesign.af.alphafold.model.modules")
    if CDM is None:
        return False
    try:
        rec = triatt_record(CDM)
    except Exception:  # noqa: BLE001
        return False
    return bool(rec is not None and getattr(rec.get("op"), MARKER, False))


def tx_rows() -> Dict[str, int]:
    """TX's own census of kernel binds by row in this process ({} when TX is not importable); `refused` is counted on this lever's ledger instead."""
    try:
        from opt_core.kernels import triattn_xla as TX  # noqa: WPS433
        counts = dict((TX.report() or {}).get("counts") or {})
    except Exception:  # noqa: BLE001
        try:
            from opt_core.kernels import triattn_xla as TX  # noqa: WPS433
            counts = dict(getattr(TX, "COUNTS", {}) or {})
        except Exception:  # noqa: BLE001
            return {}
    return {k: int(v) for k, v in counts.items() if k != "refused" and int(v) > 0}


def _token(text: str) -> str:
    """A LEVER line value carries no blank (opt_core.report.lever_line): whitespace runs as `_`, `=` as `:` (levers.detail_token's rule)."""
    import re  # noqa: WPS433
    return re.sub(r"\s+", "_", str(text).strip()).replace("=", ":") or "none"


def _csv(d: Dict[str, int]) -> str:
    return ",".join(f"{k}:{int(v)}" for k, v in sorted(d.items())) if d else "none"


def _words() -> dict:
    led = _ledger()
    return {"rows": _csv(tx_rows()), "served_fn": int(led.served), "served_grad": 0,
            "vjp_traced": int(_STATE["vjp_traced"]), "vmap": int(_STATE["vmap_traced"]), "refused": _csv(_STATE["refused_shapes"]), "layout": LAYOUT, "impl_word": IMPL_WORD,
            "requires": REQUIRES, "grad": REQUIRES, "proj_restacked": int(_STATE["proj_restacked"]), "numerics": NUMERICS, "precision": PRECISION,
            "cc": str(_STATE["cc"]) if _STATE["cc"] is not None else "unread", "tx": tx_version(), "source": "exit", "pid": os.getpid()}


def off_line(reason: str) -> str:
    """The lever's line in a kit-route mode that does not select it: state=off, zero counters (nothing of TX is imported)."""
    return _core_report.lever_line(TAG, LEVER, "off", reason=reason, impl="triattn_xla", origin="core", requires=REQUIRES, numerics=NUMERICS)


def exit_line() -> str:
    """The lever's ONE evidence line of this process: skipped with its reason, or the ledger's census + TX's row census + the numerics words."""
    if LEDGER is None:
        return off_line("not_installed")
    if _STATE["skipped"]:                                                          # install() raised (a direct caller; under levers.install the line is levers'): the protocol's words
        return LEDGER.line(TAG, "skipped", "cannot_run", detail=_token(f"{_STATE['skipped']}"), **_words())
    if not installed() and _STATE["installed_on"] is None:
        return LEDGER.line(TAG, "skipped", "not_installed", **_words())
    return LEDGER.line(TAG, None, None, **_words())                            # on when served; skipped all_fallback:<reason> / no_calls otherwise (the ledger's words)


def lever_line(state: Optional[str] = None, reason: Optional[str] = None) -> str:
    return _ledger().line(TAG, state if state is not None else ("on" if installed() else "off"), reason, **_words())


def info() -> dict:
    return {"lever": LEVER, "impl": impl_word(), "origin": "core", "numerics": NUMERICS, "requires": REQUIRES, "layout": LAYOUT, "impl_word": IMPL_WORD,
            "installed": installed(), "skipped": _STATE["skipped"], "proj_restacked": int(_STATE["proj_restacked"]),
            "expected_fallbacks": list(EXPECTED_FALLBACKS), "tx": tx_version()}


def evidence() -> dict:
    """The ledger's fields ({} before install), TX's row census, the refusal texts by shape, the numerics words and the gate verdict."""
    if LEDGER is None:
        return {}
    g = LEDGER.gate(require_served=False)
    return {**LEDGER.fields(), **_words(), "skipped": _STATE["skipped"], "refusal_texts": dict(_STATE["refusals"]), "rows_selected": dict(_STATE["rows_seen"]),
            "gate_ok": bool(g.ok), "gate_reason": g.reason}


def reset_for_tests() -> None:
    global LEDGER
    LEDGER = None
    _STATE.update({"skipped": None, "installed_on": None, "op": None, "tri_op": None, "proj_restacked": 0, "cc": None, "vjp_traced": 0, "vmap_traced": 0,
                   "refusals": {}, "refused_shapes": {}, "rows_seen": {}})
