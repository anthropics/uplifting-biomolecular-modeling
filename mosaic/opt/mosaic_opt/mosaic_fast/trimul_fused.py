"""F8 `trimul_fused` — the triangle multiplication served by the shared core's kernel provider BY TIER WORD, ON TOP of F6's channel-major layout.

What it does. F6 (`mosaic.fast.trimul_layout`) owns the rebinding of `joltz.TriangleMultiplication{Outgoing,Incoming}.__call__` and serves the
block in one channel-major layout through XLA. This lever gives F6's served call another BODY through F6's one extension point
(`trimul_layout.set_body`): `OP_MODULE.trimul(module, x, mask, direction)` — the kit's binding of the shared core's JAX-family kernel provider
(`mosaic.fast.trimul_provider` → `opt_core.kernels.pallas.serve.triangle_multiplication`), with EXACTLY `trimul_layout.trimul_cmajor`'s contract
(same math to the served row's numerics class, shapes, dtypes, leading dims, mask multiplied as given). The spec IS the provider's TIER WORD
(WORDS): `fast` | `big` — the word the mode of the same name carries. The kit names no kernel, no row, no tile and no
size threshold: for every traced call the provider's cell table for (the running jax version, this GPU, the activation dtype, the module
family, the token count, the call kind) decides — a fused Pallas / XLA-FFI kernel row where the table names one, F6's OWN body BY NAME
(`xla(cell)`, counted) where it names XLA or no cell covers the call. Both activation dtypes are served here: float32 calls and the bfloat16 calls P7 `halfpair` hands over
(P7 installs no dtype route while this lever holds the body). The differentiated call (the design step) and the un-differentiated call (the
refold, inference) are asked separately, so each runs the row selected for its kind (the op's docstring: one custom_vjp per call class).
Nothing of joltz is rebound here, so the install order, P5's `sub` wrapping and the off state stay F6's. Numerics class `fast`: the served rows'
class (bf16 tensor-core products with f32 accumulation and f32 LayerNorm statistics on bfloat16 operands; tf32-class products on
float32 operands), F6's arithmetic where XLA serves — not bitwise with stock.

Refusals (by name, before anything is traced; no silent fallback anywhere): `unknown_spec` (a word outside WORDS); `not_installed`; `needs_F6`
(F6 not installed or not configured on: this lever has no layout of its own); `op_missing` (the op module is not in the installed mosaic);
`probe_failed:<reason>` (the op's probe: the provider is missing from the installed opt_core, or cannot answer on this host — `backend_not_gpu`);
`op_refused:<kind>` (the op's configure refuses the word). A call the provider refuses at trace time runs F6's body and is COUNTED by the
refusal's name — this lever declares no fallback words (`EXPECTED_FALLBACKS = ()`), so a census with any fallback record fails the gate, and a
traced call outside the op's two named bodies (`kernel`, `xla(<why>)`) fails it too (`op_census_open`).

    from mosaic.fast import trimul_layout, trimul_fused
    trimul_layout.install(); trimul_layout.configure(None)      # F6 first
    trimul_fused.install(); trimul_fused.configure("fast")      # mode fast's word ("big" in mode big; None = SETTING; 'stock' = off: F6's XLA body back)
    trimul_fused.describe()                                     # flat single-token record: on, spec, word, kernel_calls, xla_calls, rows, cells, served, shapes, …
    trimul_fused.emit_line(tag); trimul_fused.gate()            # the census line; the fail-closed gate

Uniform lever API (mosaic_opt.levers): install / uninstall / configure(None = SETTING) / describe / gate / emit_line / ENV_REQUIRED.
Registry entry F8 (route install; modes fast and big, after F6).
"""
from typing import Any, Dict

LEVER = "F8"
NAME = "trimul_fused"
MODULE = "mosaic.fast.trimul_fused"
KLASS = "fast"
LAYOUT_MODULE = "mosaic.fast.trimul_layout"                     # F6: the owner of the joltz rebinding and of the extension point this lever uses
OP_MODULE = "mosaic.fast.trimul_provider"                       # the op: trimul / configure(word) / reset / settings / census / probe / VERSION / WORDS
SETTING = "fast"                                                 # configure(None) applies it
WORDS = ("fast", "big")                                        # the provider's tier words (the spec IS the mode's tier word)
SPEC_WORDS = "'stock' | 'off' (F6's XLA body) | " + " | ".join(repr(w) for w in WORDS)
ENV_REQUIRED: Dict[str, str] = {}                                # nothing must be exported before the interpreter starts
EXPECTED_FALLBACKS = ()                                          # no fallback is declared: any fallback record fails the gate

STATE: Dict[str, Any] = {"installed": False, "spec": None, "on": False, "ledger": None, "probe": "unprobed", "op_version": "none", "word": "none",
                         "probe_facts": "none"}


class Refusal(RuntimeError):
    """A named refusal: `.reason` is the single word the arm fails by."""
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{LEVER} {NAME}: {reason}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------------------------------------------------------- helpers
def _import(name: str):
    import importlib
    return importlib.import_module(name)


def _layout():
    """F6's module (the same object the installer's kit_module door holds: sys.modules by name)."""
    try:
        return _import(LAYOUT_MODULE)
    except ImportError as e:
        raise Refusal("needs_F6", f"{LAYOUT_MODULE} not importable: {e}") from None


def _op():
    try:
        return _import(OP_MODULE)
    except ImportError as e:
        raise Refusal("op_missing", f"{OP_MODULE} is not in the installed mosaic ({type(e).__name__}: {e})") from None


def _ledger():
    from opt_core.counters import Ledger
    return Ledger(f"{LEVER}.{NAME}", impl=MODULE, origin="kit", expected=tuple(EXPECTED_FALLBACKS))


def parse(spec):
    """`None` → SETTING; 'stock' | 'off' → off; one of WORDS → on with that tier word. Anything else refuses `unknown_spec` (by name,
    before F6 or the op is touched)."""
    word = SETTING if spec is None else str(spec).strip().lower()
    if word in ("stock", "off"):
        return {"on": False, "spec": "stock", "word": None}
    if word in WORDS:
        return {"on": True, "spec": word, "word": word}
    raise Refusal("unknown_spec", f"{spec!r}; words: {SPEC_WORDS}")


def _token(v) -> str:
    """One non-blank token: containers joined (`k:v+k:v`, `a+b`), empties → `none`, whitespace → `_` (the LEVER grammar splits fields on whitespace)."""
    if v is None:
        return "none"
    if isinstance(v, dict):
        return "+".join(f"{_token(k)}:{_token(x)}" for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))) or "none"
    if isinstance(v, (list, tuple, set, frozenset)):
        return "+".join(_token(x) for x in (sorted(v, key=str) if isinstance(v, (set, frozenset)) else v)) or "none"
    s = "_".join(str(v).split())
    return s if s else "none"


# --------------------------------------------------------------------------------------------------------------------------- the served body
def _body(module, x, mask, direction):
    """F6's served call lands here when this lever is on: census, then the op (its refusals propagate — nothing falls back silently)."""
    L = STATE["ledger"]
    if L is not None:
        lead = x.shape[:-3]
        batch = 1
        for d in lead:
            batch *= int(d)
        L.serve(f"{direction}:B{batch}xN{x.shape[-2]}xC{x.shape[-1]}")
    return STATE["op"].trimul(module, x, mask, direction)


_body.__serves_dtypes__ = ("float32", "bfloat16")               # read by P7 halfpair's route: both dtypes are the op's — P7 installs no dtype route


# --------------------------------------------------------------------------------------------------------------------------- the lever API
def install():
    """Idempotent; rebinds nothing (F6 owns the joltz rebinding). configure() is where the body is switched."""
    STATE["installed"] = True


def installed() -> bool:
    return bool(STATE["installed"])


def uninstall():
    if STATE["on"]:
        try:
            _layout().set_body(None)
        except Exception:                                        # noqa: BLE001  F6 already uninstalled: its own uninstall reset the body
            pass
    op = STATE.pop("op", None)
    if op is not None and callable(getattr(op, "reset", None)):
        op.reset()
    STATE.update(installed=False, spec=None, on=False, ledger=None, probe="unprobed", op_version="none", word="none", probe_facts="none")


def configure(spec=None) -> Dict[str, Any]:
    """Switch F6's served body: on → the op under the tier word (after `needs_F6`, `op_missing`, `probe_failed:<reason>`, `op_refused:<kind>`
    are excluded, in that order, BEFORE any trace); off → F6's XLA body. Starts a fresh census; returns describe()."""
    cfg = parse(spec)
    if not installed():
        raise Refusal("not_installed", f"configure({spec!r}) before install()")
    T = _layout()
    if cfg["on"]:
        if not (T.installed() and T.STATE.get("on")):
            raise Refusal("needs_F6", "F6 trimul_cmajor must be installed AND configured on before F8: the provider's rows serve F6's channel-major call")
        op = _op()
        probe = getattr(op, "probe", None)
        try:
            verdict = probe() if callable(probe) else {"ok": True, "kind": "no_probe"}
        except Exception as e:  # noqa: BLE001 — a probe that raises names its reason
            verdict = {"ok": False, "kind": _token(getattr(e, "kind", None) or getattr(e, "reason", None) or type(e).__name__)}
        ok = bool(verdict.get("ok")) if isinstance(verdict, dict) else bool(verdict)
        reason = (_token(verdict.get("kind") or verdict.get("reason") or ("ok" if ok else "unknown")) if isinstance(verdict, dict)
                  else ("ok" if ok else "unknown"))
        STATE["probe"] = "ok" if ok else reason
        STATE["probe_facts"] = _token({k: v for k, v in verdict.items() if k not in ("ok", "kind", "reason", "version")}) if isinstance(verdict, dict) else "none"
        if not ok:
            raise Refusal(f"probe_failed:{reason}", f"{OP_MODULE}.probe() = {verdict!r}")
        if callable(getattr(op, "reset", None)):                         # start from the op's clean state (census, cached selections)
            op.reset()
        try:
            op.configure(word=cfg["word"])
        except Exception as e:  # noqa: BLE001  the op's named refusal
            raise Refusal(f"op_refused:{_token(getattr(e, 'kind', None) or getattr(e, 'reason', None) or type(e).__name__)}", str(e)) from None
        STATE["op"] = op
        STATE["op_version"] = _token(getattr(op, "VERSION", "unversioned"))
        T.set_body(_body, cfg["word"])                                   # F6's census names the body by this lever's word
        STATE.update(spec=cfg["spec"], on=True, word=cfg["word"], ledger=_ledger())
    else:
        if T.installed() and T.STATE.get("body") is _body:
            T.set_body(None)
        op = STATE.pop("op", None)
        if op is not None and callable(getattr(op, "reset", None)):
            op.reset()
        STATE.update(spec=cfg["spec"], on=False, ledger=None, word="none")
    return describe()


def serves_dtypes():
    """The activation dtypes F6's served body takes while this lever holds it: ("float32", "bfloat16") when on, () when off."""
    return tuple(getattr(_body, "__serves_dtypes__", ())) if STATE["on"] else ()


def _op_loaded():
    """The op module if this process imported it (a census read never imports)."""
    import sys
    return sys.modules.get(OP_MODULE)


def _op_words() -> Dict[str, Any]:
    """The op's census as LEVER words: kernel_calls / xla_calls = traced calls per body (a provider row | F6's body by cell, refusal or
    layout, counted by name); op_calls = every traced op call; rows = `<dtype>:<un-differentiated arm>/<differentiated arm>` per call class as
    the provider answered; cells = the provider cell keys the answers came from; op_bodies = the op's census; op_traces = which rule bodies
    were traced and what served them."""
    op = _op_loaded()
    empty = {"kernel_calls": 0, "xla_calls": 0, "op_calls": 0, "rows": "none", "cells": "none", "op_bodies": "none", "op_traces": "none"}
    if op is None or not callable(getattr(op, "census", None)):
        return empty
    try:
        cen = dict(op.census() or {})
    except Exception:  # noqa: BLE001 — a census read never breaks the LEVER line
        return dict(empty, op_bodies="unread")
    return {"kernel_calls": int(cen.get("kernel_calls", 0)), "xla_calls": int(cen.get("xla_calls", 0)),
            "op_calls": int(cen.get("calls", sum((cen.get("words") or {}).values()))), "rows": _token(cen.get("rows") or None),
            "cells": _token(cen.get("cells") or None), "op_bodies": _token(cen.get("words") or None), "op_traces": _token(cen.get("traces") or None)}


def describe() -> Dict[str, Any]:
    """FLAT, single-token values in every state: lever, module, klass, on, spec, word, installed, op, op_version, probe, body, the census so far."""
    out: Dict[str, Any] = {"lever": LEVER, "lever_name": NAME, "module": MODULE, "klass": KLASS, "on": int(bool(STATE["on"])),
                           "spec": STATE["spec"] or "stock", "word": _token(STATE["word"]) if STATE["on"] else "none", "installed": int(installed()),
                           "impl": MODULE, "origin": "kit", "op": OP_MODULE, "op_version": _token(STATE["op_version"]), "probe": _token(STATE["probe"]),
                           "probe_facts": _token(STATE["probe_facts"]), "body": _token(STATE["word"]) if STATE["on"] else "xla", "layout_lever": "F6", **_op_words()}
    L = STATE["ledger"]
    if L is not None:
        for k, v in L.fields().items():
            if k in ("name", "state", "tag", "reason", "impl", "origin"):      # the LEVER line's own slots are the installer's; impl/origin said above
                continue
            out[k] = v if isinstance(v, (int, float)) and not isinstance(v, bool) else (int(v) if isinstance(v, bool) else _token(v))
    return out


def emit_line(tag: str = "") -> str:
    from opt_core.report import emit, lever_line
    L = STATE["ledger"]
    if L is None or not STATE["on"]:
        return emit(lever_line(tag, f"{LEVER}.{NAME}", "off", reason="not_configured" if not installed() else "mode_stock", impl=MODULE, origin="kit"))
    w = _op_words()
    return emit(L.line(tag, body=_token(STATE["word"]), word=_token(STATE["word"]), op_version=_token(STATE["op_version"]),
                       kernel_calls=w["kernel_calls"], xla_calls=w["xla_calls"], rows=w["rows"], cells=w["cells"]))


def gate():
    """Fail closed: installed but never configured on (a run under the lever's name on F6's XLA body); a census error or ANY fallback of this
    lever; nothing served; the op's bodies not accounting for its calls (every traced op call is `kernel` or `xla(<why>)` — a third, unnamed
    path is a defect)."""
    L = STATE["ledger"]
    if L is None or not STATE["on"]:
        if installed():
            raise Refusal("not_configured", f"{LEVER} is installed but configure() never turned it on: the run would carry the lever's name on F6's XLA body")
        return
    g = L.gate(f"{LEVER}.{NAME}")
    if not g.ok:
        raise Refusal("lever_gate", str(g.reason))
    if L.served == 0:
        raise Refusal("nothing_served", f"{LEVER} configured on and no triangle multiplication was traced through the op")
    w = _op_words()
    if w["op_calls"] and w["kernel_calls"] + w["xla_calls"] != w["op_calls"]:
        raise Refusal("op_census_open", f"op calls {w['op_calls']} != kernel {w['kernel_calls']} + xla {w['xla_calls']}: an op body outside the named two")
