"""Lever dit_apb — ``atlasfold.model.network.attention:Attention.forward`` (attention.py L42-100), the diffusion transformer's rank-4
``[B, N, L, c]`` calls (N = the diffusion samples of one chunk),
through the shared core's pair-bias attention provider ``opt_core.kernels.apb`` BY THE MODE'S TIER WORD: ``--mode fast`` -> ``fast``,
``--mode big`` -> ``big``, ``--mode exact`` -> ``exact``.  The provider's cell table (``APB_CELLS``: per compute capability,
activation dtype, head geometry ``dit_h16d48``, sample count, token bucket and timing form ``eager`` | ``graph``) names the row that serves
each call class; this module carries no row table, no card table and no size floor — the provider does the choosing, the kit reports what
it chose (``plan=`` / ``rows=`` on the LEVER line, the core's cell census at exit).

Statement (per batch element ``b``; ``H = num_heads``, ``D = head_dim``; ``bias`` = the stock two statements ``(~mask) * -inf`` then
``+ pair_bias.to(q.dtype)`` — the form ``sampler_hoist`` folds once per roll-out — shared by the N samples (``[1, H, Lq, Lk]``))::

    o[n, :, l, :] = softmax_m( q[n, :, l, :] . k[n, :, m, :] / sqrt(D) + bias[:, l, m] ) v[n, :, m, :]      o -> '... h lq d -> ... lq (h d)'

q / k / v are the module's own linears viewed ``[B, N, H, L, D]`` (no copy); one provider launch per batch element serves its N samples
against the ONE bias they share (``layout='shnd'``).

Tiers.  ``fast`` / ``big`` (tolerance class): the selected row serves — a fused kernel's summation order on the operands the row's
autocast hands the DiT (bf16 under ``diffusion_bf16``, the provider's fp32 cells when that lever is ablated); deterministic; the timing
form follows the call: inside ``denoiser_graph``'s warm-up / capture (a side stream, :func:`graph_context`) the ``graph`` cells with
``capture=True`` (capture-unsafe rows refused by the provider by name), on the default stream the ``eager`` cells — so the row the warm-up
compiles is the row the capture records.  A row that refuses at launch (arch / abi / shape) is followed ONCE to the fallback row the
provider names for it (``refused=<row>:<kind>-><fallback>`` on the line); a call class whose rows all refuse takes the statement below by
name (``refused:<row>``).  ``exact`` (byte class): the provider's exact word serves a row here only where (a) it is an exact-class row
(``EXACT_ROWS``: a replica by construction), (b) the table vouches it bitwise ON THIS STACK (``stack_word``) and (c) its reference statement
is THIS engine's statement — torch SDPA's MATH backend, which the rank-5 stock call reaches (``STATEMENT_ARM``).  The provider's exact class
replicates the fp32 memory-efficient SDPA statement (``sdpa_upcast``), not the MATH one, and carries no vouch on this kit's stack: its exact
word resolves to a stock-family arm ("exact vouch not recorded"), so every exact call class takes the module's own statement BY NAME
(``stock_row:<arm>``, decided by the provider's pure selection — no launch — and recorded in the core's cell census as
``NAMED_FALLBACK … word=exact refused=<row>:<reason>``).  Bytes never change in the exact row.

Steps aside BY NAME to the statement below this wrapper (``atom_sdpa``'s in fast / big, stock in exact; installed after it on the same
attribute): ``disabled`` (``AFO_DIT_APB_WORD=0|off``: installed, inert), ``rank`` (rank-3 Pairformer / rank-5 atom calls: not this
statement), ``high_precision``, ``kv_lead`` (cross-attention leading dims), ``bias_form`` (no per-row / per-head pair bias, or a per-sample
mask: not the DiT statement the sampler runs), ``cpu``, ``no_core``, ``stock_row:<arm>`` (exact tier, above), ``refused:<row>``.
``AFO_DIT_APB_WORD=<row[:variant]>`` is a developer override naming ONE provider row for every call (``l3a``, ``apb_attn``, ``fpf_apb``,
``sdpa:cudnn``, …; served as asked or refused by name) in fast / big; under ``--mode exact`` only an exact tier word is a word (anything
else installs skipped: ``not_an_exact_word:<w>``)."""
import os
from typing import Dict, Optional, Tuple

from opt_core.counters import Ledger

from . import Installed, graph_context, rebind

LEVER = "dit_apb"
NAME = "LOCAL.atlasfold.dit_apb"
ENV = "AFO_DIT_APB_WORD"                              # developer override: one provider row word (or tier word) for every call; 0 | off -> disabled
CELL = "dit_h16d48"                                  # the provider's cell word for the DiT head geometry (16 heads x 48)
TIER_OF_MODE = {"exact": "exact", "fast": "fast", "big": "big"}   # the mode word IS the provider's tier word
EXACT_TIER_WORDS = ("exact",)                        # the words an exact-mode run may carry (byte class)
STATEMENT_ARM = "sdpa:math"                          # the provider's arm word for THIS engine's stock statement (torch SDPA, MATH backend — the rank-5 call's route)
TARGET = "atlasfold.model.network.attention"
EXPECTED = ("disabled", "rank", "high_precision", "kv_lead", "bias_form", "cpu", "no_core", "stock_row:", "refused:")   # prefixes of the parametric words
_STATE: Dict[str, Optional[str]] = {"override": None, "mode": None}   # override: None = read AFO_DIT_APB_WORD; str = the scratch A/B switch


def tier_word(mode: Optional[str]) -> str:
    """The provider tier word of a kit mode (``fast`` for an unknown / unset mode)."""
    return TIER_OF_MODE.get(str(mode or ""), "fast")


def word(mode: Optional[str] = None) -> str:
    """The word this process hands the provider: the scratch switch, else ``AFO_DIT_APB_WORD``, else the mode's tier word; '' = disabled."""
    mode = mode if mode is not None else _STATE["mode"]
    o = _STATE["override"]
    if o is not None:
        w = tier_word(mode) if o.strip().upper() == "B" else o
    else:
        w = os.environ.get(ENV)
        w = tier_word(mode) if w is None or not w.strip() else w
    w = (w or "").strip()
    return "" if w.lower() in ("0", "off", "none", "a") else w


def bench_arm(arm=None):
    """Scratch A/B switch for developer timing scripts (one process, both arms): 'A' -> the statement below, 'B' -> the mode's tier word, any other
    string -> that provider word, None -> back to ``AFO_DIT_APB_WORD`` / the tier word.  Takes effect on the next call."""
    _STATE["override"] = None if arm is None else str(arm)
    return word()


def expected(reason: str, words=EXPECTED) -> bool:
    """A counted fallback reason is declared when it equals or starts with a listed word (``stock_row:<arm>``, ``refused:<row>`` are parametric)."""
    return any(reason == w or reason.startswith(w) for w in words)


def gate_for(ledger, words=EXPECTED):
    """The exit gate: kernel errors refuse (the core's sentence), a fallback reason outside ``words`` refuses, an all-fallback run for declared
    reasons (a CPU run, an exact run, a disabled run) is a legitimate run of the statement below."""
    def gate():
        from opt_core.gates import Gate
        f = ledger.fields()
        if f["errors"]:
            return ledger.gate(require_served=False)
        unexp = {r: n for r, n in f["fallback_by"].items() if not expected(r, words)}
        if unexp:
            return Gate(name=ledger.name, ok=False, reason="unexpected fallback: " + ",".join(f"{k}:{v}" for k, v in sorted(unexp.items())),
                        details=f, words=tuple(f["words"]))
        return Gate(name=ledger.name, ok=True, details=f, words=tuple(f["words"]))
    return gate


class Binding:
    """The provider binding of one process: the word, the per-call-class decision (cached: the provider's Selection to serve, or the named
    reason the statement below serves), the plan / rows / refused facts on the LEVER line."""

    def __init__(self, APB, ledger, w: str, tier: str):
        self.APB, self.ledger, self.word, self.tier = APB, ledger, w, tier
        self.exact = tier == "exact"
        self.decided: Dict[Tuple, Tuple] = {}         # (dtype, Lq, S, graph, H, D) -> ("serve", Selection, serve_word) | ("below", reason)
        self.plan: Dict[str, str] = {}
        self.rows: Dict[str, int] = {}
        self.refused: Dict[str, str] = {}
        self.heads: Dict[Tuple, str] = {}             # call class -> the first row that refused it at launch
        self._cc = self._stack = self._abi = None

    def _device_facts(self, dev):
        if self._cc is None:
            import torch
            self._cc = tuple(torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else (0, 0)
            if self.exact:                            # the exact word is judged on THIS stack (the table's vouch is stack-specific)
                try:
                    self._stack = self.APB.stack_word(dev)
                except Exception:  # noqa: BLE001
                    self._stack = None
                try:
                    self._abi = self.APB.dit_exact_abi()
                except Exception:  # noqa: BLE001
                    self._abi = None
        return self._cc

    def arm(self, sel) -> str:
        return f"{sel.row}{(':' + sel.variant) if getattr(sel, 'variant', None) else ''}"

    def _publish(self):
        L = self.ledger
        L.set("plan", ",".join(f"{k}:{v}" for k, v in sorted(self.plan.items())) or "none")
        L.set("rows", ",".join(f"{k}:{v}" for k, v in sorted(self.rows.items())) or "none")
        L.set("refused", ",".join(f"{k}:{v}" for k, v in sorted(self.refused.items())) or "none")

    def decide(self, dtype: str, device, Lq: int, S: int, g: bool, H: int, D: int):
        """The decision for this call class: the provider's Selection under the word (tier word: the cell's measured row for (cc, dtype,
        dit_h16d48, S, N-bucket, eager|graph); row word: that row by the provider's default rule), or the named reason the statement below
        serves.  Pure (no launch, no tensor); cached per class; the plan fact is updated when a new class is seen."""
        APB = self.APB
        key = (dtype, Lq, S, g, H, D)
        got = self.decided.get(key)
        if got is not None:
            return got
        cc = self._device_facts(device)
        tag = f"{Lq}{'g' if g else 'e'}{'' if dtype == 'bf16' else ':' + dtype}"
        try:
            sel = APB.select(cc, dtype, CELL, Lq, word=self.word, samples=S, capture=g, head_dim=D, heads=H,
                             stack=self._stack if self.exact else None, abi=self._abi if self.exact else None)
        except APB.Refusal as e:                      # the word cannot be selected here (unknown word, a row word this arch / abi refuses): below, by that name
            row = str(getattr(e, "row", None) or self.word).replace(" ", "_")
            self.refused[row] = str(getattr(e, "kind", e)).replace(" ", "_").replace("=", "~")[:60]
            got = ("below", "refused:" + row.split(":")[0])
            self.plan[tag] = "refused/" + row
        else:
            arm = self.arm(sel)
            if self.exact and not (sel.row in getattr(APB, "EXACT_ROWS", ()) and getattr(sel, "exact_vs", None) == STATEMENT_ARM):
                got = ("below", "stock_row:" + arm)   # no exact-class row vouched bitwise on this stack against THIS engine's statement: the module's own statement, by name
                self.plan[tag] = "stock_row/" + arm
                if "vouch not recorded" in str(getattr(sel, "reason", "")):
                    self.refused.setdefault("exact_word", f"vouch_not_recorded_on:{self._stack}")
                elif sel.row in getattr(APB, "EXACT_ROWS", ()):
                    self.refused.setdefault("exact_word", f"{arm}_replicates:{getattr(sel, 'exact_vs', None)}")
            else:
                got = ("serve", sel, self.word)
                self.plan[tag] = arm
        self.decided[key] = got
        self._publish()
        return got

    def follow(self, key, e, tag):
        """A row refused AT LAUNCH (``APB.Refusal``): record it; if the provider names a fallback row, the class serves that row from now on
        (returned), else the class takes the statement below (None)."""
        APB = self.APB
        row = str(getattr(e, "row", None) or self.word)
        fb = getattr(e, "fallback", None)
        kind = str(getattr(e, "kind", e)).replace(" ", "_").replace("=", "~")[:48]
        self.heads.setdefault(key, row)
        prev = self.decided.get(key)
        already_followed = prev is not None and prev[0] == "serve" and prev[2] != self.word
        if fb and not already_followed:
            try:
                cc = self._cc if self._cc is not None else (0, 0)
                dtype, Lq, S, g, H, D = key
                sel = APB.select(cc, dtype, CELL, Lq, word=fb, samples=S, capture=g, head_dim=D, heads=H)
            except APB.Refusal:
                sel = None
            if sel is not None:
                self.refused[row] = f"{kind}->{self.arm(sel)}"
                self.decided[key] = ("serve", sel, fb)
                self.plan[tag] = self.arm(sel)
                self._publish()
                return self.decided[key]
        self.refused[row] = kind
        head = self.heads.get(key, row)               # the class steps aside named after the FIRST row that refused it (the word's row), not the last link followed
        self.decided[key] = ("below", "refused:" + head.split(":")[0])
        self.plan[tag] = "refused/" + head
        self._publish()
        return None

    def served(self, sel, n: int = 1):
        a = self.arm(sel)
        if not self.rows and self.ledger.get("row") is None:   # the first served call's row / measured class (the recorded LEVER tokens row= cls=)
            self.ledger.set("row", a); self.ledger.set("cls", getattr(sel, "cls", None))
        self.rows[a] = self.rows.get(a, 0) + n
        self.ledger.set("rows", ",".join(f"{k}:{v}" for k, v in sorted(self.rows.items())))


def install(mode, tag, ctx):
    import torch
    A = __import__(TARGET, fromlist=["Attention"])
    cls = A.Attention
    below = cls.forward                                                       # atom_sdpa's wrapper when that lever is in the row (fast / big), else stock
    _STATE["mode"] = str(mode) if mode is not None else None
    tier = tier_word(mode)
    w0 = word(mode)
    if str(mode) == "exact" and w0 and w0 not in EXACT_TIER_WORDS:            # a tolerance row / tier word is not a word of the byte-class row: skipped by name
        return Installed(LEVER, False, reason=f"not_an_exact_word:{w0}")
    try:
        from opt_core.kernels import apb as APB
        if not hasattr(APB, "pair_bias_attention") or not hasattr(APB, "select"):
            raise ImportError("opt_core.kernels.apb without pair_bias_attention/select")
        no_core = None
    except Exception as e:  # noqa: BLE001
        APB, no_core = None, f"{type(e).__name__}:{str(e)[:40]}".replace(" ", "_")
    ledger = Ledger(NAME, impl=(f"opt_core.kernels.apb:{w0 or 'off'}" if APB is not None else "aside(no_core)"), origin="kit", expected=EXPECTED)
    ledger.set("word", w0 or "off")
    if APB is None:
        ledger.word("no_core", no_core)
    bindings: Dict[str, Binding] = {}

    def binding_of(w: str) -> Binding:
        b = bindings.get(w)
        if b is None:
            b = bindings[w] = Binding(APB, ledger, w, tier)
            b.exact = tier == "exact" or w in EXACT_TIER_WORDS            # an exact word under any mode is judged as the byte class; the exact MODE judges every word so
        return b
    binding = binding_of(w0) if (APB is not None and w0) else None
    caches: Dict[Tuple, dict] = {}

    def forward(self, a_q, a_k, mask, pair_bias):
        w = word()
        if not w:
            ledger.fallback("disabled")
            return below(self, a_q, a_k, mask, pair_bias)
        if APB is None:
            ledger.fallback("no_core")
            return below(self, a_q, a_k, mask, pair_bias)
        if self.use_high_precision:
            ledger.fallback("high_precision")
            return below(self, a_q, a_k, mask, pair_bias)
        if a_q.dim() != 4 or a_k.dim() != 4:                                    # rank 3 (Pairformer single attention) / rank 5 (windowed atom attention): not this statement
            ledger.fallback("rank")
            return below(self, a_q, a_k, mask, pair_bias)
        if a_k.shape[:2] != a_q.shape[:2]:
            ledger.fallback("kv_lead")
            return below(self, a_q, a_k, mask, pair_bias)
        if not a_q.is_cuda and not getattr(APB, "SERVES_CPU", False):           # the provider's rows are CUDA kernels (a test double may declare SERVES_CPU)
            ledger.fallback("cpu")
            return below(self, a_q, a_k, mask, pair_bias)
        if not isinstance(mask, torch.Tensor) or mask.dim() != 4 or mask.shape[1] != 1:   # a per-sample mask ([B, N, 1, L]) is not the sampler's DiT statement (one token mask per batch element)
            ledger.fallback("bias_form")
            return below(self, a_q, a_k, mask, pair_bias)
        b = binding_of(w)                                                        # the scratch switch may name another word mid-process: its own decisions
        B, N, Lq, _ = a_q.shape
        Lk = a_k.shape[-2]
        H, D = self.num_heads, self.head_dim
        g = bool(graph_context(a_q))                                              # denoiser_graph warm-up (side stream) or capture: the graph cells; CPU / default stream: eager
        wd = torch.get_autocast_dtype("cuda") if (a_q.is_cuda and torch.is_autocast_enabled("cuda")) else a_q.dtype   # the dtype the linears will hand the row (decided before any op: the exact tier launches nothing)
        got = b.decide(APB.dtype_word(wd), a_q.device, int(Lq), int(N), g, int(H), int(D))
        if got[0] == "below":
            ledger.fallback(got[1])
            return below(self, a_q, a_k, mask, pair_bias)
        q = self.linear_q(a_q)                                                   # [B, N, Lq, H*D] under the caller's autocast state
        k = self.linear_k(a_k); v = self.linear_v(a_k)
        if APB.dtype_word(q.dtype) != APB.dtype_word(wd):                        # an autocast state the prediction missed: decide again for the real dtype (pure)
            got = b.decide(APB.dtype_word(q.dtype), q.device, int(Lq), int(N), g, int(H), int(D))
            if got[0] == "below":
                ledger.fallback(got[1])
                return below(self, a_q, a_k, mask, pair_bias)
        q = q.view(B, N, Lq, H, D).permute(0, 1, 3, 2, 4)                        # [B, N, H, Lq, D] views over the linears' outputs
        k = k.view(B, N, Lk, H, D).permute(0, 1, 3, 2, 4)
        v = v.view(B, N, Lk, H, D).permute(0, 1, 3, 2, 4)
        attn_bias = ((~mask).to(q.dtype) * (-self.inf)).unsqueeze(-3)          # the stock two statements: [B, 1, 1, 1|Lq, Lk]
        if pair_bias is not None:
            attn_bias = attn_bias + pair_bias.to(q.dtype)                       # [B, 1, H, Lq, Lk] (sampler_hoist: folded once per roll-out, the held tensor afterwards)
        if not isinstance(attn_bias, torch.Tensor) or attn_bias.dim() != 5 or attn_bias.shape[0] != B or attn_bias.shape[1] != 1 \
                or attn_bias.shape[2] not in (1, H) or attn_bias.shape[3] not in (1, Lq) or attn_bias.shape[4] != Lk:
            ledger.fallback("bias_form")
            return below(self, a_q, a_k, mask, pair_bias)
        bias5 = attn_bias.expand(B, 1, H, Lq, Lk)                               # heads / query rows broadcast as stride-0 views (no copy); ONE bias per batch element, shared by its N samples
        if bias5.stride(-2) == 0 or (H > 1 and bias5.stride(-3) == 0):           # a bias without per-row / per-head values (no pair bias): not the DiT statement this lever serves
            ledger.fallback("bias_form")
            return below(self, a_q, a_k, mask, pair_bias)
        key = (APB.dtype_word(q.dtype), int(Lq), int(N), g, int(H), int(D))
        tagk = f"{Lq}{'g' if g else 'e'}{'' if key[0] == 'bf16' else ':' + key[0]}"
        while True:
            _, sel, sw = got
            cache = caches.setdefault((sw, key), {})
            try:
                outs = []
                for bi in range(B):                                              # one launch per batch element: its N samples share bias5[bi] ([1, H, Lq, Lk])
                    o, _s = APB.pair_bias_attention(q[bi], k[bi], v[bi], bias5[bi], None, None, word=sw, selection=sel, layout="shnd", cell=CELL,
                                                    capture=g, cache=cache)
                    outs.append(o)
                out = outs[0].unsqueeze(0) if B == 1 else torch.stack(outs)      # [B, N, H, Lq, D]
                break
            except APB.Refusal as e:                                             # the row declines AT LAUNCH (arch / abi / shape): follow the provider's named fallback row once, else below by name
                got = b.follow(key, e, tagk)
                if got is None:
                    ledger.fallback(b.decided[key][1])
                    return below(self, a_q, a_k, mask, pair_bias)
        b.served(sel)
        ledger.serve(f"L{Lq}xN{N}xB{B}{'g' if g else 'e'}")
        return out.permute(0, 1, 3, 2, 4).reshape(B, N, Lq, H * D)              # == stock's rearrange('... h lq d -> ... lq (h d)')
    forward.__qualname__ = "Attention.forward[atlasfold_opt:dit_apb]"
    rebind(cls, "forward", forward, below)
    return Installed(LEVER, True, lines=[lambda: ledger.line(tag)], gates=[gate_for(ledger)],
                     facts={"impl": getattr(ledger, "impl", None), "word": w0 or "off", "tier": tier, "ledger": ledger, "binding": binding, "bindings": bindings})
