"""Pair-track lever ``triattn``: Chai-1's triangle attention (the 48 pairformer blocks', the 4 MSA-module blocks' and the template pair
stack's ``TriangleAttention``: per direction ``SDPA(q, k, v, attn_mask = pair2b(zn) with masked_fill(~mask, -10000)) * sigmoid(g)``) evaluated
by the shared core's ONE triangle-attention provider ``opt_core.kernels.triattn`` BY THE MODE'S TIER WORD at every crop, head dim and card —
no kit cell table, no row pinned by name or number:

    fast  -> select(cc, bf16, dh, H, N, word="fast",  form="bias_only", position_stride=H*4*dh, stack=<this process's ABI key>)
    big -> select(..., word="big"): the provider's big cells (the memory-lean rows; the LEVER line names tier=big word=big; no row of
             the family holds an S x S buffer, so its fast rows are memory-lean too)
    exact -> select(..., word="exact"): served only when the provider names an exact-class row whose bits are vouched against THIS engine's
             reference arithmetic (SDPA) on this stack; the provider's exact rows reproduce the cuEquivariance op's bits (exact_vs=cueq), so on
             this engine the exact tier keeps the statement BY NAME (the kit's exact mode does not compose the lever; ``decide(tier="exact")``
             answers ``stock_statement``)

The provider decides the row per (cc, dtype, head dim, heads, tokens, call form, operand strides): the row its table names for the cell, with the cell's
launch setting (row names are the provider's; a table change applies here with no kit edit).  Where
the provider names a STOCK row for the cell (``cueq`` / ``sdpa`` / ``stock``: "the kit's own stock op") the statement serves BY NAME — that is
the provider's decision, booked ``fallback:stock_statement`` (declared) and announced once with the row word.  A row that refuses the call
(``Refusal``: e.g. ``int32_offset``, ``install_failed``, an unknown word) is booked ``fallback:refused:<kind>`` — NOT declared: the exit gate
refuses by name — announced once, and the statement serves.  A row that raises at its first build / launch on this card (a row inherited on
an architecture outside the provider's table, meeting a toolchain gap) prints its traceback once, is booked ``fallback:row_error`` (declared) with LEVER fact
``error=<ExcType>``, and the module's own statement serves the rest of the process; never a crash, never silent.  The provider's own coverage census
(``[opt_core] CELLS ...`` at exit) records every decided call class (cell hit / named fallback / UNCOVERED_CELL).

Binding = the statement's own tensors, no copies: q / k / v / g are strided ``[B, N, H, S, D]`` views of the statement's ``[B, N, N, (H, 4, dh)]``
projection buffer (the ending direction projects the transposed LayerNorm output so its rows have unit-stride positions: the Triton rows keep
32-bit in-row offsets and the provider refuses ``int32_offset`` BY NAME past them — ``position_stride=`` answers that at selection); the
statement's bf16 pair bias with the pair mask folded exactly as the statement folds it, widened once to fp32 (the provider's bias dtype), no key
mask (the statement has none: call form ``bias_only``); the provider's ``[B, N, H, S, D]`` output gated straight into the statement's cat
buffer.  A batch above one (the template pair stack: one row per template) is served one batch row per call (the rows' in-row offsets never
carry a batch stride).  Same arithmetic class as the cuDNN / flash SDPA backends the statement runs on (bf16 products, fp32 accumulation and
softmax): Tier 2 (fast-class numerics, CHANGES.md), never bitwise.

LEVER line facts: ``tier= word= row= core_cell= class= core_measured= launch= form=`` of the first served cell; every decided (dh, H, N) is
announced once on stderr: ``chai1-opt TRIATTN tier=.. word=.. form=bias_only row=.. core_cell=.. class=.. launch=.. N=.. H=.. D=..``.
The tier1 line sets no plug of its own: a step-aside runs the module's own
statement; big's ``trunk_chunk`` owns the plug at its gate and above (unchanged).  Strategy id F1.flash_triatt.
"""
import os
import sys
from typing import Any, Dict, Optional, Tuple

LEVER = "triattn"
TAG = "chai1-opt"                                         # the kit's stderr note prefix (pairtrack's TRIATTN notes print the same)
STRATEGY = "F1.flash_triatt"
PROVIDER = "opt_core.kernels.triattn"
CORE_FLOOR = "0.5.37.2"                                   # the first core whose provider takes the call form (form=), the position-stride hint and the stack word this binding passes
DTYPE_WORD = "bf16"                                       # the statement's q/k/v dtype (the trunk's interface dtype)
FORM = "bias_only"                                        # the statement's call form in the provider's words: one additive pair bias per (batch, head) broadcast over rows, the pair
TIERS = ("exact", "fast", "big")                        #  mask folded INTO it (masked_fill -10000), no separate key mask
REFERENCE = "sdpa"                                        # this engine's reference arithmetic for the exact tier: torch SDPA (cuDNN | flash), not the cuEquivariance op
EXPECTED_FALLBACKS = ("stock_statement", "no_cell", "row_error")   # declared step-asides: the provider names the stock op for a cell (its decision); it carries no cell / no row
                                                          #  for this card, dtype or head dim; or the row it serves fails to build / launch on this card at its first call (an
                                                          #  inherited row on an architecture outside its table: traceback printed once, LEVER fact error=<ExcType>) — the statement serves
_ANNOUNCED: Dict[str, bool] = {}


# ----------------------------------------------------------------------------------------------------------------- words (pure Python, CPU-testable)
def cc_word(cc) -> str:
    """'9.0' for (9, 0) / '9.0' / 9.0."""
    if isinstance(cc, (tuple, list)):
        return f"{int(cc[0])}.{int(cc[1])}"
    return str(cc)


def word_for(tier: str, provider=None) -> str:
    """The provider word a kit tier asks is the tier itself: exact -> 'exact', fast -> 'fast', big -> 'big' (every provider
    face lists the three tier words; the big cells encode the family's peak-memory rule)."""
    if tier not in TIERS:
        raise LookupError(f"triattn: unknown tier {tier!r}; tiers: {TIERS}")
    return tier


def position_stride(dh: int, heads: int) -> int:
    """The q / k / v position stride (elements) of the views this binding hands the provider: one token step inside the ``[B, N, N, (H, 4, dh)]``
    projection buffer = ``heads * 4 * dh`` (1024 for the pairformer) — the ``position_stride=`` hint of the provider's ``select`` (its Triton rows
    keep 32-bit in-row offsets and refuse ``int32_offset`` BY NAME past ``(N-1) * stride + dh >= 2**31``; this stride stays under it to 2**21 tokens)."""
    return int(heads) * 4 * int(dh)


def launch_word(config: Optional[Dict[str, Any]]) -> str:
    """'m128n32R1w4s3r168' for a provider launch setting (the LEVER line's ``launch=`` fact); '' for the row's own cell."""
    if not config:
        return ""
    c = dict(config)
    w = f"m{c.get('BLOCK_M', '?')}n{c.get('BLOCK_N', '?')}R{c.get('ROWS', '?')}w{c.get('num_warps', '?')}s{c.get('num_stages', '?')}"
    if c.get("ORDER"):
        w += f"o{c['ORDER']}"
    if c.get("MAXNREG"):
        w += f"r{c['MAXNREG']}"
    return w


def decide(cc, dh: int, heads: int, n_tokens: int, tier: str = "fast", select=None, stack: Optional[str] = None):
    """The per-call-class decision (pure Python; the provider's ``select`` is stdlib-only): ``(selection | None, fallback_word | None, word)``.
    ``select(cc, bf16, dh, heads, N, word=word_for(tier), position_stride=heads*4*dh, stack=stack, form=FORM)`` — the provider's row
    for this call form and these strides with its cell's launch setting.  ``stack`` = the provider's ABI key of this process (None on a CPU:
    static admission), so a row this stack cannot load is passed over at selection, by name.  None with ``fallback_word``:
    ``stock_statement`` (the provider names the stock op for the cell — on the exact tier also when its exact row is not vouched against this
    engine's reference arithmetic; declared) | ``no_cell`` (the provider's ``no_cell:*`` / ``no_row:*`` refusal: its table carries nothing for
    this card / dtype / head dim; declared) | ``refused:<kind>`` (any other refusal — NOT declared: the exit gate refuses by name)."""
    from opt_core.kernels import triattn as T
    sel_fn = T.select if select is None else select
    word = word_for(tier, T)
    try:
        sel = sel_fn(cc_word(cc), DTYPE_WORD, int(dh), int(heads), int(n_tokens), word=word, position_stride=position_stride(dh, heads), stack=stack, form=FORM)
    except T.Refusal as e:
        kind = str(getattr(e, "kind", e))
        if kind.startswith(("no_cell", "no_row")):                                   # the provider's table carries nothing for this card / dtype / head dim: a NAMED step-aside (declared)
            return None, "no_cell", word
        return None, f"refused:{kind}", word
    row = getattr(sel, "row", None)
    if row in tuple(getattr(T, "STOCK_ROWS", ())) or row is None:                 # the provider names the stock op for this cell: on this engine that op IS the statement
        return None, "stock_statement", word
    if tier == "exact" and (getattr(sel, "cls", None) != "exact" or str(getattr(sel, "exact_vs", "cueq")) != REFERENCE):
        return None, "stock_statement", word                                        # an exact-class row vouched against another reference op is not exact FOR THIS ENGINE
    return sel, None, word


def coverage(cc, dh: int = 64, heads: int = 4, tier: str = "fast", sizes=(256, 384, 512, 768, 1024, 1536, 2048), select=None, stack: Optional[str] = None) -> Dict[int, str]:
    """{tokens: what serves} for a card, head dim and tier — the notes' and the tests' view of the binding ('k2b m128n32R1w4s3r168 (core <cell>)' |
    'stock_statement (<word>)' | 'refused:<kind> (<word>)')."""
    out = {}
    for n in sizes:
        sel, fb, word = decide(cc, dh, heads, n, tier, select=select, stack=stack)
        if sel is None:
            out[n] = f"{fb} ({word})"
        else:
            lw = launch_word(getattr(sel, "config", None))
            out[n] = f"{word} -> {sel.row}{(' ' + lw) if lw else ''} (core {sel.cell})"
    return out


def new_ledger(expected=EXPECTED_FALLBACKS):
    """This lever's opt_core.counters.Ledger.  Its gate holds on a run whose every call was a DECLARED step-aside (the provider naming the stock
    op for every cell of a fold books calls and serves none BY ITS DECISION: ``state=skipped reason=all_fallback:stock_statement`` on the LEVER
    line); an undeclared word (``refused:<kind>``) or a kernel error refuses it as for every trunk lever."""
    from opt_core.counters import Ledger

    class TriattnLedger(Ledger):
        def gate(self, name=None, *, require_served=False):
            return Ledger.gate(self, name, require_served=require_served)

    return TriattnLedger(STRATEGY, impl=PROVIDER, origin="core", expected=tuple(expected))


# ----------------------------------------------------------------------------------------------------------------- the binding (torch at call time)
def _announce(key: str, text: str) -> None:
    if not _ANNOUNCED.get(key):
        _ANNOUNCED[key] = True
        sys.stderr.write(f"{TAG} TRIATTN {text}\n")
        sys.stderr.flush()


def _stack_key() -> Optional[str]:
    """The provider's stack key for its prebuilt CUDA rows in THIS process (torch already imported by the caller), or None when the provider
    carries no such helper (the rows then meet the stack at their first call and refuse there by name)."""
    try:
        from opt_core.kernels.triattn import cuda_sm90a as _C
        return str(_C.stack_key())
    except Exception:                                                         # pragma: no cover - a core without the helper / a CPU-only interpreter
        return None


def make_impl(TR, ledger, state: Dict[str, Any]):
    """The ``CFG['triattn_impl']`` callable.  ``TR`` = the trunk module (``chai1_eager.trunk``: ``bfw``); ``ledger`` = this lever's
    opt_core.counters.Ledger; ``state['tier']`` = the kit mode's tier; ``state['prev']`` = the plug the line had set when this lever installed
    (the tier1 statement) — what a step-aside calls (None -> NotImplemented: the module's own statement); ``state['cc']`` (optional) = the
    compute capability to decide by (read off the tensor's device once when absent)."""
    sel_cache: Dict[Tuple, Any] = {}
    tier = state.get("tier", "fast")

    def step_aside(mod, zn, mask):
        prev = state.get("prev")
        if prev is None:
            return NotImplemented
        return prev(mod, zn, mask)

    def triattn_impl(mod, zn, mask):
        if state.get("row_error"):                                            # a row failed to build / launch earlier in this process: the statement serves from there on, counted
            ledger.fallback("row_error")
            return step_aside(mod, zn, mask)
        B, N = int(mask.shape[0]), int(mask.shape[1])
        H, dh = int(mod.H), int(mod.dh)
        ck = (dh, H, N)
        ent = sel_cache.get(ck)
        if ent is None:
            cc = state.get("cc")
            if cc is None:
                import torch
                cc = state["cc"] = tuple(torch.cuda.get_device_capability(zn.device)) if zn.device.type == "cuda" else (0, 0)
            if "stack" not in state:                                          # the provider's ABI key of this process (its prebuilt CUDA rows are keyed by it): rows this stack
                state["stack"] = _stack_key()                                 #  cannot load are passed over at selection by name, not met at the first call
            ent = sel_cache[ck] = list(decide(cc, dh, H, N, tier, stack=state["stack"]))
            sel, why, word = ent
            if why == "stock_statement":
                _announce(f"stock:{cc}:{dh}:{H}:{N}", f"tier={tier} word={word} form={FORM} statement=line reason=stock_statement cc={cc_word(cc)} D={dh} H={H} N={N} "
                          f"(the provider names the stock op for this cell: the line's own statement serves, counted fallback:stock_statement)")
            elif why == "no_cell":
                _announce(f"no_cell:{cc}:{dh}:{H}", f"tier={tier} word={word} form={FORM} statement=line reason=no_cell cc={cc_word(cc)} D={dh} H={H} "
                          f"(the provider carries no cell / row for this card and head dim: the line's own statement serves, counted fallback:no_cell)")
            elif why is not None and why.startswith("refused:"):
                _announce(f"refused:{cc}:{dh}:{H}:{N}", f"tier={tier} word={word} form={FORM} REFUSED {why[8:]} cc={cc_word(cc)} D={dh} H={H} N={N} - the line's own statement serves; the exit gate refuses by name")
            if sel is not None:
                lw = launch_word(getattr(sel, "config", None))
                if ledger.get("row") is None:                                     # the first serving cell's words ride the LEVER line
                    ledger.set("tier", tier); ledger.set("word", word); ledger.set("row", sel.row)
                    ledger.set("core_cell", str(sel.cell)); ledger.set("class", str(sel.cls))
                    ledger.set("core_measured", "yes" if getattr(sel, "measured", False) else "no")
                    ledger.set("launch", lw if lw else "row_cell"); ledger.set("form", FORM)
                _announce(f"cell:{cc}:{dh}:{H}:{N}", f"tier={tier} word={word} form={FORM} row={sel.row} core_cell={sel.cell} class={sel.cls} launch={lw if lw else 'row_cell'} N={N} H={H} D={dh}")
        sel, why, word = ent
        if sel is None:
            ledger.fallback(why)
            return step_aside(mod, zn, mask)
        # ---- the statement, on the provider's row -------------------------------------------------------------------------------------
        import torch
        import torch.nn.functional as F
        from opt_core.kernels import triattn as T
        BF = torch.bfloat16
        bfw = TR.bfw
        zb = zn if zn.dtype == BF else zn.to(BF)
        b = F.linear(zb, bfw(mod.pair2b.weight))                                            # [B, N, N, 2H] bf16: pair2b(zn)
        b = b.masked_fill(torch.bitwise_not(mask.to(torch.bool)).unsqueeze(-1), -10000)      # the statement's fold of the pair mask at (query, key), every (d, h)
        bias = b.permute(0, 3, 1, 2).to(dtype=torch.float32, memory_format=torch.contiguous_format)   # [B, 2H, S, S] fp32, unit key stride (one copy, both directions)
        del b
        out = torch.empty(B, N, N, 2 * H * dh, dtype=BF, device=zn.device)
        for d, W in enumerate((mod.pair2qkvg1.weight, mod.pair2qkvg2.weight)):
            zin = zb if d == 0 else zb.transpose(1, 2)                                       # the ending direction projects the TRANSPOSED LN output (one bf16 copy inside the GEMM's
            x = F.linear(zin, bfw(W)).view(B, N, N, H, 4, dh)                                #  reshape): row r of x is then pair row r of that direction with unit-stride positions —
            del zin                                                                          #  the rows' 32-bit in-row tile offsets cannot carry the transposed read as an N*H*4*dh stride
            tgt = out[..., d * H * dh:(d + 1) * H * dh].view(B, N, N, H, dh)                  # out[b, r, s, (h, e)]: the ending direction stays in its transposed orientation, as traced
            for bi in range(B):                                                              # one batch row per call (the template pair stack: one row per template; the trunk: B == 1)
                xb = x[bi:bi + 1]
                q, k, v, g = (xb[:, :, :, :, i, :].permute(0, 1, 3, 2, 4) for i in range(4)) # [1, r, H, s, D] strided views (position stride H*4*dh, unit last stride): no copies
                try:
                    o = T.triangle_attention(q, k, v, bias[bi:bi + 1, d * H:(d + 1) * H].unsqueeze(1), None, None, word=word, selection=sel, form=FORM)   # [1, r, H, s, D] contiguous
                except T.Refusal as e:                                                       # the row refused THESE tensors before launching (e.g. int32_offset): booked (undeclared:
                    ent[0], ent[1] = None, f"refused:{e.kind}"                               #  the exit gate refuses by name), announced once, and the line's statement serves from here on
                    _announce(f"refused:{state.get('cc')}:{dh}:{H}:{N}", f"tier={tier} word={word} row={sel.row} REFUSED at call time {e.kind} D={dh} H={H} N={N} - the line's own statement serves this process; the exit gate refuses by name")
                    ledger.fallback(ent[1])
                    del x, xb, q, k, v, g, out, bias
                    return step_aside(mod, zn, mask)
                except Exception as e:                                                       # the row failed to build / launch on this card (an inherited row on an architecture outside the provider's table,
                    state["row_error"] = type(e).__name__                                    #  a toolchain gap): the traceback ONCE, booked fallback:row_error (declared: the fold proceeds on
                    if not _ANNOUNCED.get("row_error:traceback"):                            #  the statement, named), LEVER fact error=<ExcType>; the statement serves the rest of the process
                        _ANNOUNCED["row_error:traceback"] = True
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                    _announce("row_error", f"tier={tier} word={word} row={sel.row} core_cell={sel.cell} ERROR {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''} "
                                           f"D={dh} H={H} N={N} - stepped aside: the module's own statement serves this process from here (counted fallback:row_error, LEVER fact error={type(e).__name__})")
                    ledger.set("error", type(e).__name__)
                    ledger.fallback("row_error")
                    del x, xb, q, k, v, g, out, bias
                    return step_aside(mod, zn, mask)
                torch.mul(o.permute(0, 1, 3, 2, 4), torch.sigmoid(g.permute(0, 1, 3, 2, 4)), out=tgt[bi:bi + 1])
                del xb, q, k, v, g, o
            del x
        ledger.serve(f"N{N}xH{H}xD{dh}")
        return out

    triattn_impl.chai1_opt_lever = LEVER
    return triattn_impl


def install(tw, TR, ledger, mode: str = "fast", cc=None) -> str:
    """Wrap the trunk wrapper's ``cfg_fn`` (the stack's per-call fixer of ``chai1_eager.trunk.CFG``, already extended by the levers installed
    before this one): after it runs, ``CFG['triattn_impl']`` becomes this binding and what it had set is kept as the step-aside target.
    ``mode`` = the kit mode (its tier picks the provider word); ``cc`` pins the compute capability to decide by (None = the tensor's device at
    the first call).  Returns the fact string for the activation report.  Raises LookupError when the wrapper or the trunk module lacks the
    plug points or the mode is not a tier."""
    if not isinstance(getattr(TR, "CFG", None), dict) or "triattn_impl" not in TR.CFG:
        raise LookupError("triattn: the eager trunk module carries no CFG['triattn_impl'] plug point")
    if not hasattr(tw, "cfg_fn"):
        raise LookupError("triattn: the installed trunk wrapper carries no cfg_fn (the stack's per-call CFG fixer)")
    if mode not in TIERS:
        raise LookupError(f"triattn: mode {mode!r} is not a tier ({TIERS})")
    state: Dict[str, Any] = {"prev": None, "cc": (tuple(cc) if cc is not None else None), "tier": mode}
    impl = make_impl(TR, ledger, state)
    impl.chai1_opt_state = state                                                  # read by the tests and the notes' probes (prev plug, cc, tier)
    base_cfg = tw.cfg_fn
    if base_cfg is not None:
        base_cfg()
    cur = TR.CFG.get("triattn_impl")                                              # the plug the line sets (the tier1 statement): the step-aside target, pinned
    state["prev"] = cur if cur is not impl else None                              # now — a wrapper installed after this lever (big's chunker) is never adopted (no plug loop)

    def cfg_fn():
        if base_cfg is not None:
            base_cfg()
        TR.CFG["triattn_impl"] = impl
    cfg_fn.chai1_opt_lever = getattr(base_cfg, "chai1_opt_lever", "") + "+" + LEVER
    tw.cfg_fn = cfg_fn
    cfg_fn()
    word = word_for(mode)
    ledger.set("tier", mode); ledger.set("word", word)
    return f"CFG[triattn_impl]={PROVIDER} by tier word (tier={mode} word={word} form={FORM}; step-aside target: {getattr(state['prev'], '__name__', state['prev'])})"
