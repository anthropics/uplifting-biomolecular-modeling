"""boltz2_opt.pairblock — the engine adapter of the core's fused triangle-attention block (``opt_core.attn.pair_fused``: LayerNorm -> one
q|k|v|g|pair-bias prologue kernel -> attention core -> one sigmoid-gate * o @ W_o epilogue kernel) onto Boltz-2's ``TriangleAttention``
(``boltz.model.layers.triangular_attention.attention``; the starting- and ending-node classes of every Pairformer stack: trunk, MSA-module
pair stack, template module, confidence module).  Attached by ``boltz2_opt.worker_launch --attach pairblock`` right after that module imports.

Switch ``BOLTZ_PAIRBLOCK`` (the mode table sets it, modes.py): ``cueq`` = the block around the STOCK attention core (cuEquivariance's
``triangle_attention``, the kernel the shipped stock runs) — the exact-class construction: the module's own LayerNorm (fp32 under autocast, as
stock), the bf16 cast autocast applies at every projection applied once, the prologue's projections and bf16-rounded pair bias, the stock core on
the same q/k/v/bias/mask layouts and dtypes stock hands it, the epilogue's gate and output projection, the update returned in bf16 for the
Pairformer's own fp32 residual add (under ``BOLTZ_TRIATTN_EXACT=1``, the exact row, that core is asked of the core's triangle-attention
provider by its ``exact`` word with the library call as its stock op: boltz2_opt.triattn_exact); ``flash`` = the same block with the core's flash triangle attention (``opt_core.kernels.flash_triattn``) as
the core at every token count (no kit token gate: the named core's own measured table decides per card and N) inside the
same fused surround) — Tier 2;
``k2b`` = the same with the core's K2B flash kernel (``fpf_triatt_k2b``) as the attention core — Tier 2;
``core.<word>`` = the same block with the attention core served through the core's ONE triangle-attention provider (``opt_core.kernels.triattn``:
every carried row + the measured cell table per compute capability) BY WORD — a row name (``k2b``, ``k2``, ``flash``,
``cuda_sm90a``, ``exact_headsplit``, ``cueq`` …) or a tier word (``fast`` | ``exact``: the cell's measured winner, opt-in); a row that
cannot serve this box or shape refuses BY NAME (``core:<row>:<kind>``, counted; the gate refuses) and the call takes the original forward — nothing is
substituted.  The class of a ``core.<word>`` variant is the word's (exact rows are bitwise to the library kernel; the others Tier 2).
``default`` = the block's own keyed attention core (``core="default"``: the core's DEFAULT_CORE_TABLE — the triattn provider's
``fast`` tier word at and above its keys threshold per (cc, dtype, head_dim, heads), the flash_triattn cell below it; Tier 2), so a threshold
the core re-measures reaches this adapter with no kit change; what served is counted as ``pf_cores=`` on the line.
Calls the block does not serve take the module's original forward, counted by reason (``report()``): kernels
off (``--no_kernels``: the torch triangle path, the kits' kernels-off line), autocast off or not bf16, and the core's own refusals
(``opt_core.attn.pair_fused.Unsupported``: a shape without a served cell, ...).  ``EXPECTED`` lists the reasons a healthy run may show; any
other reason, or a kernel error, refuses the fail-closed gate (``verdict()``), which the run's evidence check turns into a failed run.
On a card with a served row rule for ``cueq`` (``CARD_ROWS``, boltz2_opt.rowfloor; compute capability 8.0: [5,857, 3,145,632) rows of the
flattened call, and from 1,048,544 rows only when the trailing piece the card works the call in clears 5,857 rows — outside which the stock
projections' own GEMMs sum in another order) the calls outside it take the original forward, counted ``below_min_rows`` / ``above_max_rows`` /
``trailing_piece``; ``flash`` / ``k2b`` have no rule.
"""
import math
import os
import sys
from typing import Any, Dict, List, Optional


from opt_core.oom import is_oom          # the core's one out-of-memory predicate: an out-of-memory error propagates, no fallback applied

from . import rowfloor as RF                                          # the per-card served row range of the bitwise variant: its words, the card probe, the table lookup, the row count
from . import exactln as XLN                                         # the bitwise LayerNorm replica's fused-cast entry (ln_to_bf16: the stock statement `module(x).to(bfloat16)` when that lever is off) [EXACTLN]
from . import triattn_exact as TX                                    # the exact mode's binding of the `cueq` core to the core provider's `exact` word (core_for: the library call itself unless BOLTZ_TRIATTN_EXACT=1)

TAG = "boltz2-opt"
NAME = "F1.pairblock"
SWITCH = "BOLTZ_PAIRBLOCK"
VARIANTS = ("cueq", "flash", "k2b", "default")                        # `default`: the block's OWN keyed attention core (opt_core.attn.pair_fused core="default",
                                                                       # its DEFAULT_CORE_TABLE: the triattn provider's `fast` tier word at and above the table's keys threshold per
                                                                       # (cc, dtype, head_dim, heads), its flash_triattn cell below it and for unlisted classes; Tier 2) + `core.<word>` (CORE_PREFIX): the core's triangle-attention provider by word (core_words())
CORE_PREFIX = "core."
EXACT_CORE_WORDS = ("exact_headsplit", "cueq", "exact")   # the provider's exact-class words (bitwise to the library kernel): a `core.<word>` of these is the exact-class block (lever pairblock alone)
LEVERS_OF = {"cueq": ("pairblock",), "flash": ("pairblock", "flash_triattn"), "k2b": ("pairblock", "flash_triattn"), "default": ("pairblock", "flash_triattn")}
LN_OF = {"cueq": "stock", "flash": "stock", "k2b": "stock", "default": "stock"}          # every variant feeds the module's own LayerNorm output (bf16) to the prologue: the core table's served rows; the LayerNorm-in-kernel form on the fp32 pair tensor is a candidate row of that table


def core_word(v) -> Optional[str]:
    """'core.k2' -> 'k2' (a word of the core's triangle-attention face); anything else -> None."""
    if not (isinstance(v, str) and v.startswith(CORE_PREFIX) and len(v) > len(CORE_PREFIX)):
        return None
    w = v[len(CORE_PREFIX):]
    return w if w.split("@")[0] in core_words() else None


def core_words() -> tuple:
    """The words the core's triangle-attention face accepts (rows + tier words; the face is standard-library-only at import)."""
    from opt_core.kernels import triattn as KA
    return tuple(KA.ROW_NAMES) + tuple(KA.TIER_WORDS)


def levers_of(v: str) -> tuple:
    """The registry levers a variant installs: the block; + the fast attention core it hosts (flash_triattn) for the Tier-2 cores."""
    if v in LEVERS_OF:
        return LEVERS_OF[v]
    w = core_word(v)
    if w is None:
        return ()
    return ("pairblock",) if w.split("@")[0] in EXACT_CORE_WORDS else ("pairblock", "flash_triattn")


class CoreRefused(Exception):
    """The core's provider refused a call with the tensors in hand (its Refusal, by name): `row`, `kind`."""
    def __init__(self, row, kind):
        Exception.__init__(self, f"core:{row}:{kind}")
        self.row, self.kind = row, kind


_CORE_CALLABLES: Dict[str, Any] = {}
_CORE_ADMIT: Dict[tuple, Optional[str]] = {}
_CORE_ROUTES: Dict[str, str] = {}


def _layout(t) -> str:
    """'contig' | 'strided' — whether the operand's position axis is a transposed / strided view (position stride != head_dim for [.., S, D])."""
    try:
        return "contig" if int(t.stride(-2)) == int(t.shape[-1]) else "strided"
    except Exception:  # noqa: BLE001
        return "?"


def _note_route(q, k, v, bias, mask5) -> None:
    key = f"{int(q.shape[-2])}:{call_form(mask5)}:q_{_layout(q)}:bias_{'contig' if bias.is_contiguous() else 'strided'}"
    if key in _CORE_ROUTES:
        return
    try:
        from opt_core.kernels.triattn import triattn_native as CCm
        r = str(CCm.route(q, k, v, bias, mask5))
    except Exception as e:  # noqa: BLE001 — the package's typed refusal names itself; the call that follows raises it by name
        r = f"REFUSED:{getattr(e, 'kind', None) or type(e).__name__}:{str(e)[:80]}"
    _CORE_ROUTES[key] = r.replace(" ", "_")
    sys.stderr.write(f"[{TAG}] {NAME} CORE-ROUTE {key} -> {_CORE_ROUTES[key]}\n")


def core_routes() -> Dict[str, str]:
    return dict(_CORE_ROUTES)


def cueq_stock(q, k, v, bias, mask=None, scale=None):
    """The stock attention core in the library's own signature (the provider's `stock=` for its cueq / exact_headsplit rows)."""
    return _cueq_core(q, k, v, bias, mask, scale)


def call_form(mask5) -> str:
    """The provider's call-form word for what this kit hands the attention core: `bias_only` when no key mask reaches the core (the driver drops an
    all-ones pair mask — every single-input Boltz-2 pass: `mask_real=0` on the census), else `mask_bias` (pair_fused's per-row key-mask layout
    [B, I, 1, 1, J] built from the engine's pair mask, itself the outer product of the token-padding vector)."""
    return "bias_only" if mask5 is None else "mask_bias"


_FORM_HINT = {"ok": None}     # None = not yet asked; True = the face takes form=; False = a face without that parameter (called without the hint)


def core_callable(word: str):
    """pair_fused's `core=` callable serving the attention core through the core's triangle-attention face BY WORD (memoised per word).  A refusal with
    the tensors in hand raises CoreRefused (the caller books it by name and takes the original forward; a resident driver fails closed)."""
    fn = _CORE_CALLABLES.get(word)
    if fn is None:
        from opt_core.kernels import triattn as KA

        def fn(q, k, v, bias, mask5, scale, _word=word):
            if _word.split("@")[0] == "triattn_native":                  # the package's served route per (tokens, operand layout), once each: the census that the transposed
                _note_route(q, k, v, bias, mask5)                     # (ending-node) operands are served by the strided extension and not handed back
            try:
                if _FORM_HINT["ok"] is not False:                       # form=<bias_only|mask_bias>: the provider's call-form hint (a cell's per-form order; a hint, never a refusal)
                    try:
                        out = KA.triangle_attention(q, k, v, bias, mask5, scale, word=_word, stock=cueq_stock, form=call_form(mask5))
                        _FORM_HINT["ok"] = True
                        return out
                    except TypeError:
                        if _FORM_HINT["ok"]:
                            raise
                        _FORM_HINT["ok"] = False                        # an older face without form=: the same call without the hint
                return KA.triangle_attention(q, k, v, bias, mask5, scale, word=_word, stock=cueq_stock)
            except KA.Refusal as e:
                raise CoreRefused(e.row, str(e.kind).replace(" ", "_")[:120]) from None
        fn.__name__ = f"triattn_core_{word}"
        _CORE_CALLABLES[word] = fn
    return fn


def core_admit(word: str, m, x, n: int) -> Optional[str]:
    """Ask the core's face (its table; no launch) whether `word` serves this (card, bf16, head_dim, heads, N) — once per (word, N, device).
    None = admitted (the selection is recorded for the census); else the refusal `core:<row>:<kind>`."""
    key = (word, int(n), str(x.device))
    if key in _CORE_ADMIT:
        return _CORE_ADMIT[key]
    try:
        import torch
        from opt_core.kernels import triattn as KA
        cc = torch.cuda.get_device_capability(x.device)
        stack = None
        if tuple(cc) == (9, 0) and word.split("@")[0] in ("cuda_sm90a", "triattn_native", "fast"):
            from opt_core.kernels.triattn import cuda_sm90a as _C
            stack = _C.stack_key()                                   # the CUDA rows' per-ABI prebuilt: refused here by name when absent
        sel = KA.select(cc, "bf16", int(m.mha.c_hidden), int(m.mha.no_heads), int(n), word=word, stack=stack)
        _STATE["core_cells"][f"{int(n)}"] = KA.describe(sel)
        sys.stderr.write(f"[{TAG}] {NAME} CORE-CELL n={int(n)} {KA.describe(sel)}\n")
        out = None
    except Exception as e:  # noqa: BLE001 — the face's Refusal (by name) or an import problem
        row, kind = getattr(e, "row", None), getattr(e, "kind", None)
        out = ("core:" + (f"{row}:{kind}" if kind else f"{type(e).__name__}")).replace(" ", "_")[:160]
    _CORE_ADMIT[key] = out
    return out
IMPL_ENV, IMPLS = "BOLTZ_PAIRBLOCK_IMPL", ("fpf", "lnl", "torch")     # the core implementation family behind the cell (opt_core.attn.pair_fused IMPLS); the mode rows leave it at fpf — a development A/B switch, not a lever
IMPL = os.environ.get(IMPL_ENV, "fpf") if os.environ.get(IMPL_ENV, "fpf") in IMPLS else "fpf"
LEVERS = ("pairblock", "flash_triattn", "pairblock_c64")            # the registry levers this adapter installs (worker_launch reads it): the block, and the flash core it hosts in the flash variant, and the template C=64 stack service of the Tier-2 variants
C64_ENV = "BOLTZ_PAIRBLOCK_C64"                                       # companion word (registry lever pairblock_c64): `1` = the Tier-2 variants (flash | k2b) also serve the template Pairformer's
                                                                      # C=64 stack — key (64,4,32): opt_core's qualified prologue / epilogue cells r11 / r21 (9.0) and r91 / r92 (8.0), opt_core >=
                                                                      # TOLERANCE class (fp64-error ratio 1.05 under ln='stock', max |d| <= 0.0078 bf16) — with cuEquivariance's core below
                                                                      # the variant's core at every token count; absent = that stack takes the module's original forward by name
                                                                      # (kept_out:64x4x32). `cueq` (the exact-class construction) never serves it: the cells are not bitwise, and a templated input
                                                                      # carries their numerics to the output (an untemplated input does not — the template module's update is masked to zero
                                                                      # there, which is what lever templ_skip elides).
KEYS_BASE = ((128, 4, 32),)                                           # (C, H, D) keys every variant serves: the trunk / MSA-module / confidence pair stacks (fused prologue / epilogue proven BITWISE to
                                                                      # the stock projections: core rows r07 / r16 on real activation dumps; every identity item of this kit's ladders and bank)
KEY_C64 = (64, 4, 32)                                                 # the template Pairformer's tri-attention (template_dim 64, 4 heads x 32)
EXPECTED = ("kernels_off", "kept_out:64x4x32", "no-cell:prologue:64x4x32") + RF.WORDS     # declared stock paths of a healthy run: the kernels-off route (the torch triangle path), the template
                                                          # module's C=64 stack where the variant does not serve it (kept_out: `cueq` always, flash | k2b without BOLTZ_PAIRBLOCK_C64=1) or the card has no cell for it,
                                                          # and the calls outside the card's served row rule (`cueq`, CARD_ROWS; no rule on 9.0: the words never show there)
# The per-card served row rule of the bitwise variant `cueq` (the rule and its words: boltz2_opt.rowfloor), from the stock block's own projection
# GEMMs on the card (q|k|v|g|o: 128->128; the pair bias: 128->4; cuBLAS under the pinned torch). 8.0: they sum in another order than at the models'
# row counts from 1,665 to 3,264 and from 5,631 to 5,856 rows (128->128; the bias below 17 rows), and a call of 1,048,544 rows (2**20 - 32) or more is
# worked in pieces of that many rows whose trailing piece (rows mod 1,048,544) meets those bands again when it is under the floor; the block equals
# stock's bits at every other row count measured on A100-80GB (every count to 65,536 and from 1,046,576 up, every N·N to 1,023²) and — end to end,
# every output file of `--mode exact` byte-identical — at every pieced call it serves that an input can reach below the top (N·N rows: N = 1,027 /
# 1,200 / 1,400, two pieces, trailing 6,185 / 391,456 / 911,456; N = 1,451, three pieces, trailing 8,313), and NOT with the block serving a trailing
# piece of 32, 2,081 or 2,513 rows (N = 1,024 / 1,025 / 1,449; 32 is outside every single-call band and differs all the same: a piece is not in
# every respect a call of its own size, so the rule is a FLOOR, never a band list); trailing pieces of 4,132 / 5,412 rows (N = 1,026 / 1,450, between
# the bands) were byte-identical with the block serving them and stay under the floor: 5,857 is the lowest row count from which every larger
# count, single or trailing, is measured equal. So `cueq` serves [5,857, 3,145,632) rows — inputs of 77 to 1,773 tokens — but a pieced call whose
# trailing piece is under 5,857 rows (1,024..1,026 / 1,449 / 1,450 tokens), and the module's original forward serves the rest, by name; the top is
# the measured top (three pieces), not a limit of the rule. flash / k2b have no rule (Tier 2: a summation order is inside their tolerance).
CARD_ROWS: Dict[str, Dict[str, RF.Range]] = {"8.0": {"cueq": (5857, 3 * RF.PIECE_ROWS_SM80, RF.PIECE_ROWS_SM80)}}
_STATE: Dict[str, Any] = {"ledger": None, "applied": [], "patched": [], "variant": None, "errors": {}, "served_by": {}, "rows": None, "core_cells": {}, "keys": None}


def variant(environ=None) -> Optional[str]:
    env = os.environ if environ is None else environ
    v = env.get(SWITCH, "").strip().lower()
    if v in VARIANTS:
        return v
    return v if core_word(v) is not None else None


def requested(environ=None) -> bool:
    return variant(environ) is not None


def c64(environ=None) -> bool:
    """True when the companion word BOLTZ_PAIRBLOCK_C64=1 is set (the Tier-2 variants serve the template C=64 stack; lever pairblock_c64)."""
    env = os.environ if environ is None else environ
    return env.get(C64_ENV, "").strip() == "1"


TIER2_VARIANTS = ("flash", "k2b", "default")                             # the named Tier-2 variants (a `core.<word>` is Tier 2 by its word's class, tier2())
PF_DEFAULT_CORE = "default"                                            # == opt_core.attn.pair_fused.DEFAULT_CORE (the block resolves it per call; tests pin the equality)


def tier2(v: Optional[str]) -> bool:
    """True for the Tier-2 variants: flash | k2b | default | a `core.<word>` whose word is not of the provider's exact class."""
    if v in TIER2_VARIANTS:
        return True
    w = core_word(v) if v else None
    return w is not None and w.split("@")[0] not in EXACT_CORE_WORDS


def keys_for(v: Optional[str], environ=None, cc: Optional[str] = None):
    """The (C, H, D) keys variant `v` hands to the core's block: KEYS_BASE; + the template's (64,4,32) in the Tier-2 variants under BOLTZ_PAIRBLOCK_C64=1
    (`cueq` never: the c=64 cells are tolerance-class). `cc` is accepted for the callers' symmetry with rowfloor (the cell exists on 9.0 and 8.0;
    a card without it answers no-cell:prologue:64x4x32 at the core, by name)."""
    return KEYS_BASE + ((KEY_C64,) if (tier2(v) and c64(environ)) else ())


def c64_word(v: Optional[str], environ=None, cc: Optional[str] = None) -> str:
    """The census word of a template-key call the variant does not serve."""
    return "kept_out:64x4x32"


def levers_for(v: Optional[str], environ=None, cc: Optional[str] = None):
    """The registry levers the variant installs (levers_of) — plus pairblock_c64 when the template key is among the served keys."""
    base = list(levers_of(v)) if v else []
    if not base:
        return []
    return base + (["pairblock_c64"] if KEY_C64 in keys_for(v, environ, cc) else [])


def weights_of(m):
    """boltz TriangleAttention -> the core's canonical triangle-attention tensors (opt_core.attn.pair_fused.TRIATTN_KEYS)."""
    a = m.mha
    for lin in (m.linear, a.linear_q, a.linear_k, a.linear_v, a.linear_g):        # every parameter of the module the pack is not given must not exist (never a silently dropped bias)
        assert lin.bias is None, f"{type(m).__name__}: a projection bias the fused block is not given ({lin})"
    return dict(ln_w=m.layer_norm.weight, ln_b=m.layer_norm.bias, w_q=a.linear_q.weight, w_k=a.linear_k.weight, w_v=a.linear_v.weight,
                w_g=a.linear_g.weight, w_b=m.linear.weight, w_o=a.linear_o.weight, b_o=a.linear_o.bias, n_heads=a.no_heads, head_dim=a.c_hidden,
                eps=getattr(m.layer_norm, "eps", 1e-5))


def _reason(r: str) -> str:
    """The core's refusal token, booked without the box's (cc|triton) suffix: 'no-cell:fpf:prologue:64x4x32:9.0|3.7' -> 'no-cell:prologue:64x4x32'."""
    if r.startswith("no-cell:"):
        parts = r.split(":")
        return "no-cell:" + ":".join(parts[2:4]) if len(parts) >= 4 else r
    return r


def _cueq_core(q, k, v, bias, mask5, scale):
    """The stock attention core exactly as boltz hands it (primitives.kernel_triangular_attn): q/k/v [B,I,H,J,D] bf16, the pair bias in the
    projections' dtype (stock's bias is the bf16 output of its autocast Linear; the prologue's fp32 copy holds the same values), mask [B,I,1,1,J] bool."""
    from cuequivariance_torch.primitives.triangle import triangle_attention
    b = bias if bias.dtype == q.dtype else bias.to(q.dtype)
    m = mask5 if mask5 is not None else None
    return triangle_attention(q, k, v, b, mask=m, scale=scale)


def apply(spec: Optional[str] = None) -> List[str]:
    """Patch TriangleAttention.forward class-wide (idempotent). ``spec`` overrides the env switch."""
    if _STATE["ledger"] is not None:
        return list(_STATE["applied"])
    v = variant() if spec is None else variant({SWITCH: str(spec)})
    if v is None:
        return []
    import torch
    from opt_core.attn import pair_fused as PF
    import boltz.model.layers.triangular_attention.attention as A

    ln = LN_OF.get(v, "stock")
    cw = core_word(v)                                                # the core face's word for a `core.<word>` variant (None for cueq / flash / k2b)
    keys = keys_for(v)                                               # the (C, H, D) keys this variant serves; every other key takes the original forward by name (kept_out:CxHxD)
    kept_c64 = c64_word(v)
    L = PF.ledger(NAME, expected=EXPECTED, min_tokens=0)          # no kit token gate: the named core serves every token count (its own table decides per card and N)
    rule_v = "cueq" if (cw is not None and cw.split("@")[0] in EXACT_CORE_WORDS) else v   # an exact-class core word is the `cueq` construction around another bitwise core: the card's `cueq` row rule is its rule
    rng = RF.range_for(CARD_ROWS, rule_v, RF.card_cc())             # this card's served row rule for the variant (None = no rule: nothing below changes, the line is unchanged)
    _STATE["rows"] = rng
    for k_, n_ in RF.facts(rng).items():
        L.set(k_, n_)                                                # named on the lever's line (min_rows=<n> max_rows=<n> piece_rows=<n>) and in the report, only on a card that has one
    orig = A.TriangleAttention.forward

    def _weights(m, device):
        W = getattr(m, "_opt_pairblock_W", None)
        if W is None:
            w = weights_of(m)
            W = PF.pack_triattn_weights(ln_w=w["ln_w"], ln_b=w["ln_b"], w_q=w["w_q"], w_k=w["w_k"], w_v=w["w_v"], w_g=w["w_g"], w_b=w["w_b"],
                                       w_o=w["w_o"], b_o=w["b_o"], n_heads=w["n_heads"], head_dim=w["head_dim"], eps=w["eps"], device=device)
            m._opt_pairblock_W = W
        return W

    def forward(self, x, mask=None, chunk_size=None, use_kernels: bool = False):
        if not use_kernels:
            L.fallback("kernels_off"); return orig(self, x, mask, chunk_size, use_kernels)
        if not (x.is_cuda and torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.bfloat16 and x.dtype in (torch.float32, torch.bfloat16)):
            L.fallback("autocast_off"); return orig(self, x, mask, chunk_size, use_kernels)
        key = (int(x.shape[-1]), int(self.mha.no_heads), int(self.mha.c_hidden))
        if key not in keys:                                          # a key this variant does not serve (the template's (64,4,32) in `cueq`, or in flash | k2b without BOLTZ_PAIRBLOCK_C64=1): the original forward, by name
            L.fallback(kept_c64 if key == KEY_C64 else "kept_out:%dx%dx%d" % key); return orig(self, x, mask, chunk_size, use_kernels)
        rows = RF.rows_of(x)
        for k_, n_ in RF.call_facts(rows, rng).items():
            L.set(k_, n_)                                            # a call the card works in pieces: its arithmetic on the line (pieces=<n> trailing=<rows>, the last such call's)
        word = RF.outside(rows, rng)
        if word is not None:                                         # outside the card's served row rule: the module's original forward, by name (its own GEMMs' bits are the contract there)
            L.fallback(word); return orig(self, x, mask, chunk_size, use_kernels)
        n = int(x.shape[-2])
        ending = not self.starting
        core = v                                                       # flash | k2b | default | core.<word>: the named core at every token count; cueq: the exact-class construction
        refused = None                                               # the refusal WORD; the stock path runs outside the except block (a bound exception keeps the frame's tensors alive)
        try:
            W = _weights(self, x.device)
            if cw is not None and core != "cueq":
                refused = core_admit(cw, self, x, n)                 # the provider's table first (by name, no launch): a row that cannot serve this box / shape -> the original forward
                core_arg = core_callable(cw)
            else:
                core_arg = {"flash": "flash_triattn", "k2b": "k2b", "default": PF_DEFAULT_CORE}.get(core, TX.core_for(_cueq_core))   # default: the block's own keyed core; cueq: the library kernel call stock makes — under BOLTZ_TRIATTN_EXACT=1 served through the core provider's `exact` word with that call as its stock op (boltz2_opt.triattn_exact)
            ok, reason = (False, refused) if refused is not None else PF.supported_triattn(x, W, mask, impl=IMPL, core=core_arg, ending=ending, ln=ln, residual=False)
            if not ok:
                refused = _reason(reason)
            else:
                x_ln = None
                if ln == "stock":                                        # exact: the module's own LayerNorm (fp32 under autocast, as stock), then the bf16 cast autocast applies at each of stock's projections, applied once
                    x_ln = XLN.ln_to_bf16(self.layer_norm, x.transpose(-2, -3) if ending else x)   # = self.layer_norm(.).to(torch.bfloat16): the stock statement unless the exactln lever is on, which fuses the cast into its bitwise LayerNorm replica's store [EXACTLN]
                    if not x_ln.is_contiguous():
                        x_ln = x_ln.contiguous()
                u = PF.tri_attn_block(x, W, mask=mask, ending=ending, residual=False, impl=IMPL, core=core_arg, ln=ln, x_ln=x_ln, scale=1.0 / math.sqrt(self.mha.c_hidden))
        except PF.Unsupported as e:
            refused = _reason(e.reason)
        except CoreRefused as e:                                     # the provider refused with the tensors in hand (its row's own supported()): by name, this call takes the original forward
            refused = f"core:{e.row}:{e.kind}"
        except Exception as e:  # noqa: BLE001 — a kernel error: counted (the gate refuses), this call served by the original forward —
            if is_oom(e): raise     # except a GPU out-of-memory error, which propagates: no fallback on out-of-memory
            k = type(e).__name__
            _STATE["errors"][k] = _STATE["errors"].get(k, 0) + 1; L.error(e)
            if _STATE["errors"][k] <= 2:
                print(f"[{TAG}] {NAME} kernel error {k}: {str(e)[:300]} — this call takes the stock forward", file=sys.stderr, flush=True)
            refused = "error:" + k
        if refused is not None:
            if not refused.startswith("error:"):
                L.fallback(refused)
            return orig(self, x, mask, chunk_size, use_kernels)
        kind = core
        L.serve(f"{n}x{int(x.shape[-1])}/{'end' if ending else 'start'}", impl=kind)
        _STATE["served_by"][kind] = _STATE["served_by"].get(kind, 0) + 1
        return u

    A.TriangleAttention.forward = forward
    _STATE.update(ledger=L, applied=levers_for(v), patched=["TriangleAttention.forward"], variant=v, keys=list(keys))
    try:
        from opt_core import report as _report
        _report.register_exit_tally(TAG + "/" + NAME, line)
    except Exception:  # noqa: BLE001
        pass
    sys.stderr.write(line() + "\n")
    return levers_for(v)


def line() -> Optional[str]:
    L = _STATE["ledger"]
    if L is None:
        return None
    from opt_core.attn import pair_fused as PF
    return PF.emit_line(L, TAG, variant=_STATE["variant"], impl=IMPL, ln=LN_OF.get(_STATE["variant"] or "", "stock"), pf_cores=pf_cores_word(), kit_served=",".join(f"{k}:{n}" for k, n in sorted(_STATE["served_by"].items())) or "-",   # kit_served=: this adapter's per-stack tally — `cores=` is the CORE's own head token (opt_core >= 0.5.98: the block's served attention cores), never passed by the kit
                        core_cells=";".join(f"{k}={str(t).replace(' ', '_')}" for k, t in sorted(_STATE["core_cells"].items())) or "-", c64=int(KEY_C64 in (_STATE.get("keys") or [])))


IDLE = "idle: every triangle-attention call of the run took a declared stock path (kernels off) — the block is installed and served nothing"


def verdict() -> Dict[str, Any]:
    """Fail-closed (the ledger's gate): refused on any kernel error, on any fallback reason outside EXPECTED, or when calls arrived and none
    was served — except the idle case (every call took a DECLARED stock path: the lever is installed, the input never reached it)."""
    L = _STATE["ledger"]
    if L is None:
        return {"ok": False, "idle": False, "reason": "not applied"}
    if L.calls > 0 and L.served == 0 and not L.errors and not L.unexpected():
        return {"ok": True, "idle": True, "reason": IDLE}
    g = L.gate(NAME)
    return {"ok": bool(g.ok), "idle": False, "reason": getattr(g, "reason", None)}


def pf_cores() -> Dict[str, int]:
    """The attention cores the core block served in this process through its OWN dispatch (opt_core.attn.pair_fused SERVED_CORES:
    `default=flash_triattn`, `default=tier:fast=<row>`, `tier:<word>=<row>`, `...=flash_triattn(refused:<kind>)`) — the census of the
    `default` variant (and of the pair-track driver's `default` pick, which calls the same block); a by-word callable or a named cell
    (k2b / flash) is counted on the kit's own ledgers, not here."""
    try:
        from opt_core.attn import pair_fused as PF
        return dict(getattr(PF, "SERVED_CORES", {}) or {})
    except Exception:  # noqa: BLE001
        return {}


def pf_cores_word() -> str:
    c = pf_cores()
    return ",".join(f"{k}:{n}" for k, n in sorted(c.items())) or "-"


def census() -> Dict[str, Any]:
    L = _STATE["ledger"]
    return None if L is None else {"served": dict(_STATE["served_by"]), "served_total": L.served, "fallback": L.fallbacks, "errors": L.errors, "shapes": L.shapes, "calls": L.calls}


def report() -> Dict[str, Any]:
    L = _STATE["ledger"]
    if L is None:
        return {"applied": [], "disabled": {}, "line": None, "census": None, "gate": None, "patched": []}
    return {"applied": list(_STATE["applied"]), "disabled": {}, "line": line(), "census": census(), "gate": verdict(), "patched": list(_STATE["patched"]),
            "variant": _STATE["variant"], "impl": IMPL, "min_tokens": 0, "expected": list(EXPECTED), "pf_cores": pf_cores(), "core_cells": dict(_STATE["core_cells"]), "core_routes": dict(_CORE_ROUTES), "form_hint": _FORM_HINT["ok"],
            "keys": ["%dx%dx%d" % k for k in (_STATE.get("keys") or [])], "c64": int(KEY_C64 in (_STATE.get("keys") or [])), **RF.facts(_STATE.get("rows")),
            **{k_: L.get(k_) for k_ in ("pieces", "trailing") if L.get(k_) is not None}}   # + the last pieced call's arithmetic, on a card that pieces (rowfloor.call_facts)
