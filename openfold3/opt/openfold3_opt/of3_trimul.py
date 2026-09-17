"""of3_trimul — this kit's binding of the core's ONE triangle-multiplication provider for the pair cells' TriMul: lever `trimul_provider`
(tolerance class; rides on `trimul_v4`).

The provider is :mod:`opt_core.kernels.trimul` (every carried TriMul implementation a named ROW — `v4` (fpf_trimul_v4), `tmk3_fast` / `tmk3_exact`
(fpf_trimul), the prebuilt sm_90a extension, the Triton row families, `native` / `native_exact`, `cueq`, `torch_math` — and the measured
cell table `TRIMUL_CELLS.json` per (compute capability, precision, c_z, c_hidden, token bucket, direction)). This kit asks it ONE word for every
TriMul call class: the LINE'S TIER WORD (`fast` on the fast line, `big` on the big lines: ``OPENFOLD3_OPT_TRIMUL_TIER``), or the caller's
word (``OPENFOLD3_OPT_TRIMUL_WORD=<row | fast | exact | big>``, an ablation knob). The core's cell table names the row per class and card and
the provider serves it; the fused v4 kernel is reachable only as the provider's row `v4` (its launch cells are the core's own table,
opt_core/kernels/fpf_trimul_v4/table.json). A class the tier word's rows all refuse is asked for the row `v4` by name, then runs the line's own
TriMul — every step named on the census. The provider's stock rows (`cueq`, `torch_math`) name the line's own statement: a cell that resolves to
one of them runs the line's TriMul for that class (`line=cell:<row>`).

Numerics: tolerance class. Every row's first statement per (class, row) in a process is checked against a reference of the class — the v4 face's
statement where its envelope admits the class, else the provider's fp32 `torch_math` — non-finite, or max-abs error above WITNESS_GROSS_FRAC of the
reference's scale -> that row is refused for the class BY NAME for the rest of the process (`witness=<class>:<row>=gross`) and the next candidate
serves; inside a CUDA-graph capture the witness is skipped by name (`witness=…=in_capture`).

Census (stderr, at exit, printed by cells/pairfused.py):
  `[openfold3-opt/pairfused] LEVER name=trimul_provider state=on word=<w> tier=<fast|big> rows=<row>:<n>,… kit=v4:0 line=<reason>:<n>,… skip=<class>:<row>(<why>),… witness=… cells=<cell>><row>,…`
  `… state=off reason=<not requested | trimul_v4 cell off | opt_core.kernels.trimul: ImportError …>`
"""
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

TAG = "openfold3-opt/trimul"
ENV = "OPENFOLD3_OPT_TRIMUL_PROVIDER"          # =1: the trimul_v4 cell asks this module which row serves each call class (the fast line exports it)
ENV_WORD = "OPENFOLD3_OPT_TRIMUL_WORD"          # the caller's knob: <row name> | fast | exact | big for every class (default: the line's tier word, ENV_TIER)
ENV_TIER = "OPENFOLD3_OPT_TRIMUL_TIER"          # the line's tier word (`fast` on the fast line, `big` on a big line that carries the cell); set by modes.py per line
KIT_ROW = "v4"                                  # the provider's name of the fused v4 kernel (opt_core.kernels fpf_trimul_v4): asked by name only for a class the tier word's rows refuse
STOCK_ROWS = ("cueq", "torch_math")             # the provider's stock rows: in this kit they name the LINE's own TriMul (the yaml's / the trunk-kernels add-on's route)
TIER_WORDS = ("fast", "exact", "big")
WITNESS_MAX_RELRMS = 0.05                       # bf16 fused rows sit at ~1e-3..1e-2 relative RMS against one another; beyond this (or non-finite) = a broken statement, refused by name


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    return (environ.get(ENV) or "").strip() == "1"


def tier_word(environ=None) -> str:
    """The line's tier word for the provider: ``OPENFOLD3_OPT_TRIMUL_TIER`` (fast | big; the resident and tp lines export
    big), else fast."""
    environ = os.environ if environ is None else environ
    w = (environ.get(ENV_TIER) or "").strip().lower()
    return w if w in ("fast", "big") else "fast"


def spec(environ=None) -> str:
    """The word asked for every class: the caller's ``OPENFOLD3_OPT_TRIMUL_WORD`` (a provider row name or a tier word) when set, else the
    line's tier word."""
    environ = os.environ if environ is None else environ
    w = (environ.get(ENV_WORD) or "").strip()
    return w if w else tier_word(environ)


def known_words() -> Tuple[str, ...]:
    """Every word the knob takes: the provider's row names + tier words (read from the core without importing torch)."""
    from opt_core.kernels import trimul as KT
    return tuple(KT.ROW_NAMES) + tuple(w for w in TIER_WORDS if w not in KT.ROW_NAMES)


def check_word(word: str) -> str:
    w = (word or "").strip() or tier_word()
    if w not in known_words():
        raise ValueError(f"{ENV_WORD}={w!r} is not a word of the core's triangle-multiplication provider (rows {', '.join(known_words())})")
    return w


def _log(msg: str) -> None:
    sys.stderr.write(f"[{TAG}] {msg}\n")


def _tok(s, n: int = 60) -> str:
    """A census token: blank-free (the LEVER line grammar is `k=v` words), capped."""
    return re.sub(r"[\s,|=]+", "_", str(s))[:n]


class Plan:
    """One class decision: kind 'kit' (the cell's own v4 statement) | 'row' (a provider row: prov + selection) | 'line' (the line's TriMul, reason)."""
    __slots__ = ("kind", "row", "prov", "sel", "reason", "key")

    def __init__(self, kind: str, row: Optional[str] = None, prov: Any = None, sel: Any = None, reason: Optional[str] = None, key: Optional[str] = None):
        self.kind, self.row, self.prov, self.sel, self.reason, self.key = kind, row, prov, sel, reason, key


class Router:
    """Per process: the word, the per-class plans (memoised), the providers (one per word, built lazily over opt_core.trimul.by_word with the kit's
    ONE weight-name map cells.pairfused.trimul_weights), the census counters and the witness records.  Thread-safe counting."""

    def __init__(self, word: Optional[str] = None, weights_of=None):
        self.word = (word if word is not None else spec()).strip()   # the ONE word asked for every class: the line's tier word, or the caller's
        self.tier = tier_word()                                  # the line's tier word (fast | big)
        self.by_tier = self.word == self.tier                    # the word IS the line's tier word (census: cells=<cell>><row>)
        self._weights_of = weights_of
        self._provs: Dict[str, Any] = {}
        self._memo: Dict[Tuple, Plan] = {}
        self._lock = threading.Lock()
        self.rows: Dict[str, int] = {}          # row -> block-direction calls served by a provider row
        self.kit = 0                            # kept at 0: the v4 kernel serves only as the provider's row (census field kit=v4:0)
        self.line: Dict[str, int] = {}          # reason -> calls (directions) the line served
        self.skips: Dict[str, int] = {}         # '<row>(<kind>)' -> times asked and refused
        self.cells: Dict[str, str] = {}         # class key -> row decided
        self.witness: Dict[str, str] = {}       # '<key>:<row>' -> 'maxabs=…/relrms=…(vs …)' | 'in_capture' | 'bad_numerics(…)'
        self.refused: Dict[Tuple[str, str], str] = {}   # (key, row) -> reason (bad numerics / serve-time refusal): never asked again in the process
        self._stack = None

    # ---------------------------------------------------------------------------------------------------------------- providers
    def weights_of(self):
        if self._weights_of is None:
            from .cells.pairfused import trimul_weights          # the ONE weight-name map of this kit (WEIGHT_KEYS + BIAS_KEYS vocabulary)
            self._weights_of = trimul_weights
        return self._weights_of

    def provider(self, word: str):
        p = self._provs.get(word)
        if p is None:
            from opt_core import trimul as T
            p = T.by_word(self.weights_of(), word, name=f"of3_trimul:{word}", cache_key=f"of3_trimul.{word}")
            self._provs[word] = p
        return p

    # ------------------------------------------------------------------------------------------------------------------ classes
    @staticmethod
    def facts(z4, mod) -> Tuple[str, str, int, int, int]:
        """(cc, dtype word, c_z, c_hidden, N) of one call in the provider's vocabulary."""
        import torch
        from opt_core.kernels import trimul as KT
        cc = KT.cc_word(KT._device_cc(z4))                       # '9.0' | '8.0' | ... (the table's capability word)
        prec, _ = KT.call_precision(z4)
        dt = prec.split("_")[-1] if prec.startswith("f32z_") else ("fp32" if prec == "tf32" else prec)
        C = int(z4.shape[-1]); N = int(z4.shape[-2])
        D = int(mod.linear_a_p.weight.shape[0])
        return str(cc), dt, C, D, N

    def class_key(self, cc, dt, C, D, N, direction) -> str:
        from opt_core.kernels import trimul as KT
        try:
            key = KT.cell_key(cc, dt, C, D, N, direction)
            key = key[0] if isinstance(key, tuple) else key
        except Exception:                                        # noqa: BLE001 — a class the table's grammar has no key for: named by its facts
            key = None
        return key or f"{cc}|{dt}|C{C}|H{D}|N{N}|{direction[:3]}"

    def plan(self, mod, z4, m3, direction: str, kit_why: Optional[str]) -> Plan:
        """The decision for this call's class (memoised per class key + kit admissibility): which row serves it."""
        cc, dt, C, D, N = self.facts(z4, mod)
        key = self.class_key(cc, dt, C, D, N, direction)
        mk = (key, kit_why is None)
        hit = self._memo.get(mk)
        if hit is not None and not (hit.kind == "row" and (key, hit.row) in self.refused):
            return hit
        words = (self.word,) + ((KIT_ROW,) if self.word in TIER_WORDS else ())   # the word for every class; a class its rows all refuse asks the v4 row by name
        plan = None
        from opt_core import trimul as T
        for w in words:
            if (key, w) in self.refused:
                continue
            prov = self.provider(w)
            call = T.Call(mod, z4, m3, direction, True, None)
            try:
                prov.eligible(call)
            except T.Refused as e:
                self._bump(self.skips, f"{w}({_tok(e)})"); continue
            sel = call.extra.get("trimul_selection")
            row = getattr(sel, "row", w)
            if row in STOCK_ROWS:                                  # the table names the line's own op for this class (a tier word's answer, or the word itself)
                plan = Plan("line", row, reason=f"cell:{row}", key=key); break
            if (key, row) in self.refused:
                self._bump(self.skips, f"{w}(refused:{row})"); continue
            plan = Plan("row", row, prov=prov, sel=sel, key=key)   # served THROUGH the provider: the word's row for this class (a tier word's row is the core table's, movable there)
            if w in TIER_WORDS:
                plan.reason = f"{w}>{row}"
            break
        if plan is None:                                           # no word served: the line's own TriMul for the class, the refusals named (skip=)
            plan = Plan("line", None, reason=kit_why or "refused", key=key)
        with self._lock:
            self._memo[mk] = plan
            self.cells[key] = (plan.reason if (plan.kind == "row" and plan.reason) else plan.row) if plan.kind != "line" else f"line({_tok(plan.reason)})"
        return plan

    # ------------------------------------------------------------------------------------------------------------------ serving
    def serve_row(self, plan: Plan, mod, z4, m3, direction: str, ref_fn=None):
        """Serve one call through the plan's provider row: returns z + update, or None when the row refused at serve time / failed its witness
        (the (class, row) is then refused for the process and the caller re-plans).  ``ref_fn()``: the reference statement (z + update) the
        class has without this lever, for the first-call witness; None = the provider's fp32 torch_math."""
        import torch
        from opt_core import trimul as T
        call = T.Call(mod, z4, m3, direction, True, None)
        call.extra["trimul_selection"] = plan.sel
        wkey = f"{plan.key}:{plan.row}"
        need_witness = wkey not in self.witness
        capturing = bool(torch.cuda.is_current_stream_capturing()) if z4.is_cuda else False
        try:
            out = plan.prov.fn(call)
        except T.Refused as e:                                     # the kernel's own check with the tensors in hand: this row leaves the class for the process, named
            with self._lock:
                self.refused[(plan.key, plan.row)] = f"serve:{_tok(e)}"
                self._bump(self.skips, f"{plan.row}(serve:{_tok(e, 50)})")
                self._memo.pop((plan.key, True), None); self._memo.pop((plan.key, False), None)
            return None
        if need_witness:
            if capturing:
                with self._lock:
                    self.witness.setdefault(wkey, "in_capture")
            elif self._witness_skipped(plan, wkey, z4):                # the core's witness policy (opt_core.kernels.trimul.witness_policy): under word=big above its
                pass                                                    #  WITNESS_BIG_MAX_N the N^2 reference statement is NOT made at the memory line's ceiling -- named, the row serves
            else:
                verdict, ok = self._witness(out, call, ref_fn)
                with self._lock:
                    self.witness[wkey] = verdict
                    if not ok:
                        self.refused[(plan.key, plan.row)] = f"bad_numerics({verdict})"
                        self._bump(self.skips, f"{plan.row}(bad_numerics)")
                        self._memo.pop((plan.key, True), None); self._memo.pop((plan.key, False), None)
                if not ok:
                    _log(f"trimul_provider: row {plan.row} class {plan.key}: first statement {verdict} — REFUSED for the process by name (bad_numerics); the next row serves")
                    return None
                _log(f"trimul_provider: row {plan.row} serves class {plan.key} (cell {getattr(plan.sel, 'cell', None) or 'no-cell'}; first statement {verdict})")
        with self._lock:
            self._bump(self.rows, plan.row)
        return out

    def _witness_skipped(self, plan: Plan, wkey: str, z4) -> bool:
        """Ask the core's witness policy before the first-call witness: ('skip', token) under the tier word big above the core's
        WITNESS_BIG_MAX_N tokens (the reference statement is an N^2-class allocation the memory line must not make at its ceiling) -> the
        token is this (class, row)'s witness record and census word, no statement is issued; ('run', '') -> False, the witness runs."""
        from opt_core.kernels import trimul as KT
        act, token = KT.witness_policy(self.word, int(z4.shape[-2]))
        if act != "skip":
            return False
        with self._lock:
            self.witness[wkey] = token
        _log(f"trimul_provider: row {plan.row} serves class {plan.key} (cell {getattr(plan.sel, 'cell', None) or 'no-cell'}; first statement {token})")
        return True

    def _witness(self, out, call, ref_fn) -> Tuple[str, bool]:
        import torch
        try:
            if ref_fn is not None:
                ref = ref_fn(); vs = KIT_ROW
            else:
                from opt_core.kernels import trimul as KT
                w = dict(self.weights_of()(call.module))
                ref = KT.torch_math(call.z.float(), None if call.mask is None else call.mask.float(), direction=call.direction, weights=w, residual=True); vs = "torch_math"
            o = out.float(); r = ref.float()
            if o.shape != r.shape:
                return _tok(f"shape:{tuple(o.shape)}!={tuple(r.shape)}(vs:{vs})"), False
            finite = bool(torch.isfinite(o).all().item())
            d = (o - r)
            maxabs = float(d.abs().max().item()) if finite else float("nan")
            relrms = float((d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-12)).item()) if finite else float("nan")
            verdict = f"maxabs:{maxabs:.3g}/relrms:{relrms:.3g}(vs:{vs})"
            return verdict, bool(finite and relrms <= WITNESS_MAX_RELRMS)
        except Exception as e:                                     # noqa: BLE001 — a witness that cannot be computed never blocks serving; named
            return f"unavailable({type(e).__name__}:{_tok(e, 40)})", True

    def count_kit(self, n: int = 1):
        with self._lock:
            self.kit += n

    def count_line(self, reason: str, n: int = 2):
        with self._lock:
            self._bump(self.line, reason, n)

    @staticmethod
    def _bump(d: Dict[str, int], k: str, n: int = 1):
        d[k] = d.get(k, 0) + n

    # ------------------------------------------------------------------------------------------------------------------- census
    def fields(self) -> str:
        j = lambda d: ",".join(f"{k}:{v}" for k, v in sorted(d.items())) or "-"
        wit = "|".join(f"{k}:{v}" for k, v in sorted(self.witness.items())) or "-"
        cells = "|".join(f"{k}:{v}" for k, v in sorted(self.cells.items())) or "-"
        return (f"word={self.word} tier={self.tier} rows={j(self.rows)} kit={KIT_ROW}:{self.kit} line={j(self.line)} skip={j(self.skips)} "
                f"witness={wit} cells={cells}")
