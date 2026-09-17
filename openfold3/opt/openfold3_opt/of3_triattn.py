"""of3_triattn — the OpenFold3 code family's binding of its triangle-attention statements onto the core's ONE provider
(`opt_core.kernels.triattn`: every carried implementation is a named row — k2b / k2 / flash / cuda_sm90a / exact_headsplit,
plus the named stock rows cueq / ds4sci / sdpa — and `TRIATTN_CELLS.json` states, per (compute capability, dtype, head dim, heads, token
bucket), which row serves a tier word).  Two statements bind here, each through a `Router`:

    pair core      the `fast` line's fused triangle-attention block (cells/pairfused.py `triatt_block` over opt_core.attn.pair_fused): the
                   attention core between the fused prologue and epilogue.  pair_fused accepts a callable core; `pair_core(spec)` returns one
                   that asks the provider per call shape with the line's TIER word (`fast`; or an explicit provider word) and the call's form
                   (`keypad` when the call carries a key mask, else `bias_only`) and serves the row the cell table names — no kit-side row order.  Tolerance class (the rows' rel-RMS is the bf16 rounding class of the flash core it replaces).
    exact          the `exact` line's stock call `openfold3.core.model.primitives.attention.triangle_attention` (the cuEquivariance library
                   function `_cueq_triangle_attn` looks up as a module global): rebound to a router on word `exact` whose stock op is that very
                   function.  Exact class: word `exact` is answered with the stock library call itself on every card — where the cell table says so
                   the same library kernel is issued one head at a time (`exact_headsplit`: identical bits, less transient memory), elsewhere the
                   library call as stock makes it, by name; a kernel row offered on word exact would serve a call shape only after its first
                   eager call proved `torch.equal` against the library on the call's own operands (`proven=`; `bits_differ:S<n>` refuses it for
                   the process, `unproven_in_capture` inside a CUDA-graph capture).  Either way the bytes are the line's own.

Refusals are the provider's names (`Refusal.kind`: no_prebuilt:<stack>, strides, smem, mask_shape, dtype_*, head_dim_*, unknown_word:…) and
are counted per kind; the row the cell names instead (`Refusal.fallback`) serves, counted as `fallback:<kind>=<n>`; a fallback that cannot
serve either takes the binding's terminal (pair core: pair_fused's flash core = the block's core before this lever; exact: the stock op).
Nothing is silent: every row that served is on the census (`rows=<row>:<n>,…`).

Census (one line per bound statement, written by the cell that owns the switch):
    LEVER name=<lever> state=on word=<word> rows=<row>:<n>,… fallback=<n> [fallback:<kind>=<n> …] cells=<cell>|… [proven=S<n>,…] [refused=…]

The kit adapter passes its env names / prefix (`configure`)."""
import os
import sys
import threading
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

PREFIX = "[openfold3-opt/triattn]"
CONFIGURABLE = ("PREFIX",)

TIER_PAIR = "fast"                                                        # the pair core's tier word for the bare `provider` spelling (the fast line; the big lines spell `provider:big`)
FORM_KEYPAD, FORM_BIAS = "keypad", "bias_only"                            # the provider's call-form words: a key-padding mask per row block | bias only
TERMINAL_PAIR = "flash_triattn"                                           # pair_fused's own core word: what the block ran before this lever


def configure(**kw) -> None:
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise TypeError(f"of3_triattn.configure: unknown setting {k!r} (configurable: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def stock_cueq() -> Optional[Callable]:
    """The cuEquivariance library op (the provider's `cueq` row and the exact rows' reference), None when not importable."""
    try:
        from cuequivariance_torch.primitives.triangle import triangle_attention as fn
    except Exception:
        return None
    return fn


def _cc_word(cc) -> str:
    return f"{int(cc[0])}.{int(cc[1])}"


class Router:
    """One engine statement bound to the provider.  `word`: a provider word (a tier word 'fast' | 'exact', a row name, or a setting word) — the
    row per call class is the provider's cell table's for that word and the call's form (no kit-side row order); `stock`: the
    cuequivariance-signature callable the stock rows and the exact rows need;
    `terminal`: (name, callable(q, k, v, bias, mask, scale)) served when neither the selected row nor its named fallback can serve;
    `prove_exact`: run-time bit proof per call class before an exact row replaces the stock op (the exact line's rule)."""

    def __init__(self, tag: str, word: str, *, stock: Optional[Callable] = None,
                 terminal: Optional[Tuple[str, Callable]] = None, prove_exact: bool = False):
        self.tag, self.word, self.stock, self.terminal, self.prove_exact = tag, word, stock, terminal, prove_exact
        self.served: Dict[str, int] = {}
        self.refused: Dict[str, int] = {}
        self.fallback_calls = 0
        self.cells: Dict[str, str] = {}                 # cell key -> row (first selection per cell)
        self.proven: Dict[tuple, bool] = {}            # exact bit proof per call class
        self.errors: list = []
        self._memo: Dict[tuple, Any] = {}
        self._lock = threading.Lock()
        self._stack_key = None

    # -- facts / selection ------------------------------------------------------------------------------------------------------------
    def _facts(self, q, k, mask=None):
        import torch
        dt = q.dtype
        if torch.is_autocast_enabled():
            dt = torch.get_autocast_dtype("cuda")
        dtype = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}.get(dt, str(dt))
        cc = torch.cuda.get_device_capability(q.device) if q.is_cuda else (0, 0)
        heads = int(q.shape[-3]) if q.dim() >= 3 else 1
        grad = bool(torch.is_grad_enabled() and (q.requires_grad or k.requires_grad))
        return (tuple(cc), dtype, int(q.shape[-1]), heads, int(k.shape[-2]), grad, FORM_KEYPAD if mask is not None else FORM_BIAS)

    def _stack(self, cc):
        if tuple(cc) != (9, 0):
            return None
        if self._stack_key is None:
            from opt_core.kernels.triattn import cuda_sm90a as C          # loads nothing: the key is computed from torch's own version facts
            self._stack_key = C.stack_key()
        return self._stack_key

    def _select(self, facts):
        """-> ("row", Selection) | ("fallback", kind, fallback_row).  Memoised per facts: the table walk and a refusal happen once per shape."""
        hit = self._memo.get(facts)
        if hit is not None:
            return hit
        from opt_core.kernels import triattn as T
        cc, dtype, D, H, S, grad, form = facts
        try:
            sel = T.select(cc, dtype, D, H, S, T.FWDBWD if grad else T.FWD, word=self.word, stack=self._stack(cc), form=form)
            hit = ("row", sel)
            if sel.cell and sel.cell not in self.cells:
                self.cells[sel.cell] = sel.row
        except T.Refusal as r:
            hit = ("fallback", r.kind, r.fallback)
            _log(f"{self.tag}: word={self.word} refused by name for {_cc_word(cc)}|{dtype}|D{D}|H{H}|S{S}: {r.kind} -> {r.fallback or 'terminal'}")
        self._memo[facts] = hit
        return hit

    # -- serving -----------------------------------------------------------------------------------------------------------------------
    def _bump(self, d: Dict[str, int], key: str) -> None:
        d[key] = d.get(key, 0) + 1

    def _serve_named(self, row: Optional[str], q, k, v, bias, mask, scale):
        """Serve the cell's named fallback row (k2b | cueq | sdpa | None) or the binding's terminal."""
        from opt_core.kernels import triattn as T
        if row in ("cueq", "ds4sci") and self.stock is not None:
            self._bump(self.served, row); return self.stock(q, k, v, bias, mask=mask, scale=scale)
        if row in ("k2b", "k2", "flash", "sdpa"):
            try:
                out = T.triangle_attention(q, k, v, bias, mask, scale, word=row, stock=self.stock)
                self._bump(self.served, row); return out
            except T.Refusal as r2:
                self._bump(self.refused, r2.kind)
        if self.terminal is not None:
            name, fn = self.terminal
            self._bump(self.served, name); return fn(q, k, v, bias, mask, scale)
        if self.stock is not None:
            self._bump(self.served, "cueq"); return self.stock(q, k, v, bias, mask=mask, scale=scale)
        raise RuntimeError(f"{PREFIX} {self.tag}: no row can serve and the binding has no terminal")

    def _prove(self, sel, facts, q, k, v, bias, mask, scale, out):
        """The exact tier's rule for a row that claims another kernel's bits: the first eager call of each class runs the stock op on the same
        operands and compares; True -> the class is served for the process; False -> refused by name for the process."""
        import torch
        cls = (facts, tuple(q.shape[:-3]), mask is not None)
        ok = self.proven.get(cls)
        if ok is not None:
            return ok
        if torch.cuda.is_current_stream_capturing():
            self._bump(self.refused, "unproven_in_capture")
            return None                                                   # not decided: this call takes the stock op, the class stays unproven
        ref = self.stock(q, k, v, bias, mask=mask, scale=scale)
        ok = bool(out.shape == ref.shape and out.dtype == ref.dtype and torch.equal(out, ref))
        self.proven[cls] = ok
        if not ok:
            _log(f"{self.tag}: row {sel.row} REFUSED for S{facts[4]} (bits differ from {sel.exact_vs or 'the stock op'} on this call's operands) — the stock op serves this class")
        return ok

    def __call__(self, q, k, v, bias, mask=None, scale=None):
        from opt_core.kernels import triattn as T
        if q.dim() == 4:                                                      # a batchless call ([N,H,S,D], bias [1,H,S,S], mask [N,1,1,S]: the engine's pair stacks
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)         #  without a batch axis): the rows' contract is 5-D — batch axis 1 as views; the output keeps
            if bias.dim() == 4:                                               #  it ([1,N,H,S,D]), which is what the library itself returns for a 4-D call (the engine squeezes it)
                bias = bias.unsqueeze(0)
            if mask is not None and mask.dim() == 4:
                mask = mask.unsqueeze(0)
        facts = self._facts(q, k, mask)
        hit = self._select(facts)
        if hit[0] == "fallback":
            _, kind, fb = hit
            self._bump(self.refused, kind); self.fallback_calls += 1
            return self._serve_named(fb, q, k, v, bias, mask, scale)
        sel = hit[1]
        if sel.cls == "stock":                                                # the cell names the stock op for this shape (by name, counted)
            return self._serve_named(sel.row, q, k, v, bias, mask, scale)
        if self.prove_exact and sel.cls == "exact" and self.stock is not None:
            cls = (facts, tuple(q.shape[:-3]), mask is not None)
            if self.proven.get(cls) is False:
                self.fallback_calls += 1
                return self._serve_named(sel.exact_vs or "cueq", q, k, v, bias, mask, scale)
        try:
            out = T.triangle_attention(q, k, v, bias, mask, scale, word=self.word, stock=self.stock, selection=sel, form=facts[6])
        except T.Refusal as r:                                                # a call-time refusal of the row (strides / smem / mask_shape / grad …)
            self._bump(self.refused, r.kind); self.fallback_calls += 1
            return self._serve_named(r.fallback, q, k, v, bias, mask, scale)
        if self.prove_exact and sel.cls == "exact" and self.stock is not None:
            ok = self._prove(sel, facts, q, k, v, bias, mask, scale, out)
            if ok is not True:                                                # differ / undecided in capture: the stock op's own output for this call
                self.fallback_calls += 1
                return self._serve_named(sel.exact_vs or "cueq", q, k, v, bias, mask, scale)
        self._bump(self.served, sel.row)
        return out

    # -- census ------------------------------------------------------------------------------------------------------------------------
    def fields(self) -> str:
        rows = ",".join(f"{r}:{n}" for r, n in sorted(self.served.items())) or "none"
        cells = "|".join(f"{c}:{r}" for c, r in sorted(self.cells.items())) or "none"
        refused = " ".join(f"fallback:{k}={n}" for k, n in sorted(self.refused.items()))
        out = f"word={self.word} rows={rows} fallback={self.fallback_calls}" + (f" {refused}" if refused else "") + f" cells={cells}"
        if self.prove_exact:
            proven = ",".join(sorted({f"S{c[0][4]}" for c, ok in self.proven.items() if ok})) or "none"
            differ = ",".join(sorted({f"S{c[0][4]}" for c, ok in self.proven.items() if ok is False})) or "none"
            out += f" proven={proven} bits_differ={differ}"
        return out


# ------------------------------------------------------------------------------------------------------------------ pair core (fast line)
PAIR: Dict[str, Any] = {"router": None, "spec": None}


def parse_pair_spec(spec: str) -> str:
    """`provider` -> the tier word `fast` (the fast line); `provider:<word>` -> <word>, any provider word: a tier (`big` — the big lines'
    spelling, `provider:big`; `exact`), a row name, or a setting word such as k2b@m128r2."""
    spec = (spec or "").strip()
    if spec == "provider":
        return TIER_PAIR
    if spec.startswith("provider:") and spec[len("provider:"):].strip():
        return spec[len("provider:"):].strip()
    raise ValueError(f"of3_triattn: pair core spec {spec!r} is not 'provider' or 'provider:<word>'")


def pair_core(spec: str, tag: str = "triatt_provider") -> Router:
    """The callable core for pair_fused.tri_attn_block(core=…): one Router per process (the census is the process's).  Kernel rows only: the
    library rows (cueq / ds4sci / exact_headsplit) are not bound inside the fused block — a refusal whose named fallback is the library, or an
    explicit library word, takes the block's flash core (TERMINAL_PAIR), counted under that name."""
    if PAIR["router"] is not None and PAIR["spec"] == spec:
        return PAIR["router"]
    word = parse_pair_spec(spec)

    def _terminal(q, k, v, bias, mask, scale):
        from opt_core.attn import pair_fused as PF
        return PF.core_attention(q, k, v, bias, mask, core=TERMINAL_PAIR, scale=scale)
    # no library rows inside the fused block: a refusal whose named fallback is the library (cueq) takes the block's own flash core instead
    PAIR["router"] = Router(tag, word, stock=None, terminal=(TERMINAL_PAIR, _terminal))
    PAIR["spec"] = spec
    return PAIR["router"]


# ------------------------------------------------------------------------------------------------------------------ exact line binding
EXACT: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "router": None, "module": None, "orig": None}


def install_exact(module_name: str, word: str = "exact", tag: str = "triatt_exact") -> Dict[str, Any]:
    """Rebind `<module_name>.triangle_attention` (the library function the engine's `_cueq_triangle_attn` calls) to a Router on `word` whose
    stock op is the original.  Refused by name (state=refused, nothing patched) when the module has no such global (the library is not
    installed: the engine never takes the cuEquivariance path then) or CUDA is absent.  Idempotent."""
    if EXACT["installed"]:
        return EXACT
    import importlib
    M = importlib.import_module(module_name)
    orig = getattr(M, "triangle_attention", None)
    if orig is None or not callable(orig):
        EXACT.update(installed=True, state="refused", reason="no_cueq:triangle_attention", module=module_name)
        _log(f"{tag}: REFUSED — {module_name} has no cuEquivariance `triangle_attention` (library not installed): the engine's own paths run")
        return EXACT
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if not has_cuda:
        EXACT.update(installed=True, state="refused", reason="no_cuda_device", module=module_name)
        _log(f"{tag}: REFUSED — no CUDA device")
        return EXACT
    router = Router(tag, word, stock=orig, terminal=("cueq", lambda q, k, v, bias, mask, scale: orig(q, k, v, bias, mask=mask, scale=scale)),
                    prove_exact=True)

    def triangle_attention(q, k, v, bias, mask=None, scale=None):
        return router(q, k, v, bias, mask, scale)
    triangle_attention._of3opt_triattn = True
    triangle_attention.__wrapped__ = orig
    M.triangle_attention = triangle_attention
    EXACT.update(installed=True, state="on", reason="", router=router, module=module_name, orig=orig)
    _log(f"{tag}: installed — {module_name}.triangle_attention -> opt_core.kernels.triattn word={word} (the cell's exact row after a per-class "
         f"bit proof against the library on the call's own operands; the library where the cell names it)")
    return EXACT
