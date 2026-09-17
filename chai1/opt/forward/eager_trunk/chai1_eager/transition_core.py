"""chai1_eager.transition_core — the trunk's Transition statement bound BY TIER WORD to the shared core's transition provider
(``opt_core.kernels.transition``), plugged via ``chai1_eager.trunk.CFG['transition_impl']`` (kit lever ``transition``, chai1_opt.pairtrack).

The statement (``Transition.forward_statement``, the export re-expressed): per row chunk (``ceil(numel * 2h/c / 2^30)`` chunks on dim -2) LayerNorm in fp32
with affine -> bf16, ONE merged ``[c -> 2h]`` bf16 projection ``a|b`` (rows a = the silu branch, then b), ``silu(a) * b`` in bf16, the ``[h -> c]`` bf16
projection.  The trunk has four classes of it: the pairformer's pair transition (256 -> 512, pair rows), the MSA module's pair transition (256 -> 1024,
pair rows), its MSA transition (64 -> 256, MSA rows) and the single transition (384 -> 768, token rows).

The binding names NO row: every call asks the provider for the MODE'S TIER WORD (``exact`` | ``fast`` | ``big`` = ``TIER[mode]``) at the call's own
cell (channels, expansion, row family read off the operand's rank, the crop as ``n_tokens``, this card, this stack) and serves what the tier resolves:
  * a carried row -> the provider's forward for that row, called per row chunk exactly as the statement chunks (same transient sizes).  On ``exact``
    the row is handed THE STATEMENT'S OWN LayerNorm output (``x_ln``: an exact-class row reproduces the statement's bytes only from the statement's own
    normalisation; a tier winner that cannot take it refuses by name) and the first call of every ``(class, rows per chunk, row)`` is bit-compared
    against the statement in this process (``torch.equal``; a class that differs is served by the statement from then on: ``fallback:class_differs``,
    declared and counted — exact bytes never change); on ``fast`` / ``big`` the row carries its own fused normalisation and the first call per class is
    checked finite and inside rel-RMS 5e-2 of the statement (a wild class -> ``fallback:wild``, undeclared: the exit gate refuses the run by name);
  * a stock row (the tier resolves to the library op at that cell) -> the statement itself serves, booked ``fallback:stock`` (declared: the
    tier word chose it);
  * a ``Refusal`` by name -> the statement serves, booked ``fallback:no_cell`` when the provider's table has no cell for the class, or
    ``fallback:refused`` otherwise (no exact-class row listed for this stack, below the row's token floor, a prebuilt absent, ...); the
    refusal kind is named once per class on stderr and on the LEVER line.
Selections are resolved once per ``(class, family, crop, rows)`` and reused; every call books ``served:<row>`` or one fallback word on the lever's Ledger."""
import sys

__all__ = ["TransitionBinding", "new_ledger", "TIER", "tier_for", "family_of", "NAME", "STRATEGY", "PROVIDER", "EXPECTED_FALLBACKS"]
__version__ = "2"
NAME = "transition"
STRATEGY = "LOCAL.chai1.transition"
TAG = "[chai1-opt]"
PROVIDER = "opt_core.kernels.transition"
EXPECTED_FALLBACKS = ("stock", "no_cell", "refused", "class_differs")
TIER = {"exact": "exact", "fast": "fast", "big": "big"}          # mode -> the provider's tier word (big asks its own word)
REL_RMS_MAX = 5e-2                                                   # fast / big first-call sanity bound on rel-RMS vs the statement


def tier_for(mode: str) -> str:
    """The provider tier word a mode asks (KeyError for a mode this lever has no word for)."""
    return TIER[str(mode)]


def family_of(x, c: int) -> str:
    """The provider's row-family word of a call, read off the operand: width-64 rows = the MSA transition ('rows'); a rank >= 4 operand [.., N, N, c] =
    a pair transition ('pair'); a rank-3 operand [B, N, c] = the single transition ('single')."""
    if int(c) == 64:
        return "rows"
    return "pair" if x.dim() >= 4 else "single"


def new_ledger(expected=EXPECTED_FALLBACKS):
    """This lever's ``opt_core.counters.Ledger``; its gate holds on a run whose every call was served or a declared step-aside (``stock`` /
    ``no_cell`` / ``refused`` / ``class_differs``); ``wild`` or an error refuses it."""
    from opt_core.counters import Ledger

    class TransitionLedger(Ledger):
        def gate(self, name=None, *, require_served=False):
            return Ledger.gate(self, name, require_served=require_served)

    return TransitionLedger(STRATEGY, impl=PROVIDER, origin="core", expected=tuple(expected))


class TransitionBinding:
    """``CFG['transition_impl']`` callable for one mode.  ``TR`` = the installed ``chai1_eager.trunk`` module (its ln_bf16 / n_chunks_for / bfw are
    called, never restated)."""

    chai1_opt_lever = NAME

    def __init__(self, TR, ledger, mode: str, selftest: bool = True):
        from opt_core.kernels import transition as T
        self.T, self.TR, self.ledger, self.mode = T, TR, ledger, str(mode)
        self.tier = tier_for(self.mode)
        self.exact = self.tier == "exact"
        self.selftest = bool(selftest)
        try:
            import torch
            mj, mn = torch.cuda.get_device_capability()
            self.cc = f"{mj}.{mn}"
        except Exception:                                   # no device (CPU tests): the provider resolves nothing here; every class stays on the statement
            self.cc = "none"
        try:
            self.stack = T.stack_word() if self.cc != "none" else "none"
        except Exception:
            self.stack = "none"
        self.packs = {}                     # id(module) -> (module, Weights)
        self.decisions = {}                 # (c, h, family, n_tokens, rows) -> ("row", word) | ("stock", row) | ("no_cell", kind) | ("refused", kind)
        self.classes = {}                   # (c, h, rows, row) -> "pass" | "differs" | "checked:relrms=.." | "wild"
        self.refused = {}                   # "c..h..:<family>@N" -> refusal kind (named once)
        self.stock = {}                     # "c..h..:<family>@N" -> the stock row the tier resolved
        self.served_rows = {}               # row word served -> calls
        self.facts = {"provider": PROVIDER, "core": getattr(__import__("opt_core"), "__version__", "?"), "mode": self.mode, "tier": self.tier}

    # ------------------------------------------------------------------------------------------------------------------------- facts
    def describe(self) -> dict:
        cl = {f"c{k[0]}h{k[1]}r{k[2]}:{k[3]}": v for k, v in self.classes.items()}
        return dict(self.facts, cc=self.cc, stack=self.stack, rows=dict(self.served_rows) or "none", classes=cl or "none",
                    refused=dict(self.refused) or "none", stock=dict(self.stock) or "none")

    def _weights(self, mod):
        ent = self.packs.get(id(mod))
        if ent is not None and ent[0] is mod:
            return ent[1]
        W = self.T.pack(w_ab=mod.linear_no_bias_ab.weight, w_o=mod.linear_out.weight, ln_w=mod.layer_norm.weight, ln_b=mod.layer_norm.bias,
                        eps=float(mod.layer_norm.eps), device=mod.linear_out.weight.device)
        self.packs[id(mod)] = (mod, W)
        return W

    # ------------------------------------------------------------------------------------------------------------------------- selection
    def _decide(self, c, h, fam, n_tokens, rows, device):
        """Resolve the tier word once per (class, family, crop, rows): a carried row to serve, a stock row (the statement serves), or a refusal by name."""
        key = (c, h, fam, n_tokens, rows)
        d = self.decisions.get(key)
        if d is not None:
            return d
        T = self.T
        name = f"c{c}h{h}:{fam}@{n_tokens}"
        try:
            sel = T.select(self.tier, c=c, hidden=h, n_tokens=n_tokens, dtype="bf16", direction="fwd", timing="eager", family=fam,
                           device=device, rows_count=rows, ln_given=self.exact)
        except T.Refusal as e:
            kind = str(e.kind)
            d = ("no_cell", kind) if kind.startswith("no_cell") else ("refused", kind)
            self.refused[name] = kind
            sys.stderr.write(f"{TAG} TRANSITION class c={c} h={h} {fam} N={n_tokens}: tier {self.tier!r} refused by name ({kind}; fallback {e.fallback})"
                             f" — the statement serves this class\n"); sys.stderr.flush()
        else:
            if sel.row in T.STOCK_ROWS:
                d = ("stock", sel.row)
                self.stock[name] = sel.row
                sys.stderr.write(f"{TAG} TRANSITION class c={c} h={h} {fam} N={n_tokens}: tier {self.tier!r} -> {sel.row} (the stock op: the statement serves)"
                                 f" cell={sel.cell_key}\n"); sys.stderr.flush()
            else:
                d = ("row", sel.row + ((":" + sel.variant) if sel.variant else ""))
        self.decisions[key] = d
        return d

    # ------------------------------------------------------------------------------------------------------------------------- the plug
    def __call__(self, mod, x):
        import torch
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            self.ledger.fallback("refused"); return NotImplemented
        c = int(mod.c); h = int(mod.linear_out.weight.shape[1])
        fam = family_of(x, c)
        n_tokens = self._n_tokens(x, fam)
        n = self.TR.n_chunks_for(x, mod.linear_no_bias_ab.weight)
        W = None
        outs = []
        for ch in torch.chunk(x, n, -2):
            rows = ch.numel() // c
            kind, what = self._decide(c, h, fam, n_tokens, rows, ch.device)
            if kind != "row":
                outs.append(self._statement_chunk(mod, ch, c)); self.ledger.fallback(kind); continue
            xln = self.TR.ln_bf16(ch, c, mod.layer_norm.weight, mod.layer_norm.bias) if self.exact else None
            state = self.classes.get((c, h, rows, what))
            if state == "differs":
                outs.append(self._statement_chunk_from_ln(mod, xln)); self.ledger.fallback("class_differs"); continue
            if W is None:
                W = self._weights(mod)
            try:
                y, sel = self.T.transition(ch, W, word=self.tier, x_ln=xln, n_tokens=n_tokens, family=fam)
            except self.T.Refusal as e:                       # a serving-time refusal (prebuilt absent on this stack, envelope): named once, the class stays on the statement
                self.decisions[(c, h, fam, n_tokens, rows)] = ("refused", str(e.kind))
                self.refused[f"c{c}h{h}:{fam}@{n_tokens}"] = str(e.kind)
                sys.stderr.write(f"{TAG} TRANSITION class c={c} h={h} {fam} N={n_tokens}: tier {self.tier!r} refused at the call ({e.kind}; fallback {e.fallback})"
                                 f" — the statement serves this class\n"); sys.stderr.flush()
                outs.append(self._statement_chunk_from_ln(mod, xln) if xln is not None else self._statement_chunk(mod, ch, c))
                self.ledger.fallback("refused"); continue
            row = sel.row + ((":" + sel.variant) if sel.variant else "")
            ckey = (c, h, rows, row)
            if self.classes.get(ckey) is None:
                verdict = self._first_call_check(mod, ch, c, xln, y, ckey, sel)
                if verdict == "differs":
                    outs.append(self._statement_chunk_from_ln(mod, xln) if xln is not None else self._statement_chunk(mod, ch, c))
                    self.ledger.fallback("class_differs"); continue
                if verdict == "wild":
                    outs.append(self._statement_chunk(mod, ch, c)); self.ledger.fallback("wild"); continue
            self.served_rows[row] = self.served_rows.get(row, 0) + 1
            self.ledger.serve(f"c{c}h{h}", impl=f"transition:{row}")
            outs.append(y)
        return outs[0] if len(outs) == 1 else torch.concatenate(outs, -2)

    @staticmethod
    def _n_tokens(x, fam):
        if fam == "pair" and x.dim() >= 3:
            return int(x.shape[-2])
        if fam == "rows" and x.dim() >= 2:
            return int(x.shape[-2])
        if fam == "single" and x.dim() >= 2:
            return int(x.shape[-2])
        return None

    def _statement_chunk(self, mod, ch, c):
        xn = self.TR.ln_bf16(ch, c, mod.layer_norm.weight, mod.layer_norm.bias)
        return self._statement_chunk_from_ln(mod, xn)

    def _statement_chunk_from_ln(self, mod, xn):
        import torch
        import torch.nn.functional as F
        ab = F.linear(xn, self.TR.bfw(mod.linear_no_bias_ab.weight))
        a, b = torch.chunk(ab, 2, -1)
        prod = F.silu(a).mul_(b)
        del a, b, ab
        return F.linear(prod, self.TR.bfw(mod.linear_out.weight))

    def _first_call_check(self, mod, ch, c, xln, y, ckey, sel):
        """exact: torch.equal(row, statement) on this chunk -> pass | differs; fast / big: finite and rel-RMS < REL_RMS_MAX -> checked | wild."""
        import torch
        ref = self._statement_chunk_from_ln(mod, xln) if xln is not None else self._statement_chunk(mod, ch, c)
        roww = ckey[3]
        if self.exact and self.selftest:
            ok = bool(torch.equal(y, ref))
            self.classes[ckey] = "pass" if ok else "differs"
            sys.stderr.write(f"{TAG} TRANSITION class c={ckey[0]} h={ckey[1]} rows={ckey[2]} tier={self.tier!r} -> {roww} cell={sel.cell_key}: bit-compare vs the"
                             f" statement = {'BITWISE' if ok else 'DIFFERS (the statement keeps this class)'}\n"); sys.stderr.flush()
            del ref
            return "pass" if ok else "differs"
        d = (y.float() - ref.float())
        rr = float(d.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt().clamp_min(1e-30))
        fin = bool(torch.isfinite(y).all())
        verdict = "checked" if (fin and rr < REL_RMS_MAX) else "wild"
        self.classes[ckey] = verdict if verdict == "wild" else f"checked:relrms={rr:.1e}"
        sys.stderr.write(f"{TAG} TRANSITION class c={ckey[0]} h={ckey[1]} rows={ckey[2]} tier={self.tier!r} -> {roww} cell={sel.cell_key}: rel-RMS vs the statement"
                         f" {rr:.2e} finite={fin} -> {verdict}\n"); sys.stderr.flush()
        del ref, d
        return verdict
