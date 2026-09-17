"""Lever triattn_exact — atlasfold's stock cuEquivariance triangle-attention call (`primitives/triangle_update.py: cueq_tri_attn(q, k, v, bias,
mask, scale)`, the cuequiv branch of TriangleAttentionStartingNode / EndingNode.forward) served through the core's triangle-attention provider
`opt_core.kernels.triattn` under the EXACT tier word (AFO_TRIATTN_EXACT_WORD, default `exact`): the provider's cell decides per
(cc, dtype, head_dim, heads, N) and per software stack, and for the word `exact` its answer is the stock op itself, by name, or — on cc 8.0 at
large N and on a few fp32 cells — `exact_headsplit`, the same library kernel called one head at a time (bit-identical to the stock call, lower
peak memory); so the lever's outputs are the stock outputs byte for byte (class exact); no kit size floor: the provider's cells and row
envelopes are the only verdict.
Steps aside BY NAME per call, the stock call (whatever was bound beneath this lever) serving it: `stock_row:<row>` (the cell's exact row IS a
stock row: cueq), `dtype_fp32` / `arch:sm_XY` / `head_dim_<D>` / … (the provider's `Refusal.kind`), `rank` (q is not the [.., N, H, N, D]
pair-stack call); a kernel that raises refuses the gate at exit (fail-loud) after the stock call served that call.
Rows: exact AND fast / big (exact ⊂ fast ⊂ big).  In the fast rows it is installed after `flash_triattn` and before `triatt_block`: the fused
block consumes the pair stacks' calls at N >= 512 (this lever then reads `no_calls`); below the block's floor the SAME call is already bound
beneath it by the fast tier word (hooks/triatt.py) — the tier that subsumes this one (the fast order holds the exact rows too and serves the
measured fastest) — so under a fast / big MODE this lever hands every call to that binding BY NAME (`tier_beneath:<word>`); with the flash lever
ablated (MODEL_OPT_LEVERS_OFF=flash_triattn) it serves as in the exact row.
Individually switchable: MODEL_OPT_LEVERS_OFF=triattn_exact; AFO_TRIATTN_EXACT_WORD=<row word> picks an exact-class row by name
(`exact_headsplit`, `cueq`).
LEVER line: name=LOCAL.atlasfold.triattn_exact impl=opt_core.kernels.triattn@<core>:<word> origin=core served=<calls> fallback=<n>
fallback_by=<word:n> shapes=<BxN:n> exact_stack=<cc|torch|cueq> plan=<N:row> rows=<row:n> word=<word> errors=<n>."""
import os
from typing import Dict

from . import Installed, rebind
from . import pair_cells as PC

LEVER = "triattn_exact"
NAME = "LOCAL.atlasfold.triattn_exact"
TARGET = "atlasfold.model.network.primitives.triangle_update"
WORD_ENV = "AFO_TRIATTN_EXACT_WORD"
DEFAULT_WORD = "exact"
EXACT = "exact"
EXPECTED = ("stock_row:", "dtype_", "arch:", "rank", "no_calls", "head_dim_", "single_head", "no_candidate_admitted", "tier_beneath:")


def word() -> str:
    w = (os.environ.get(WORD_ENV) or "").strip()
    return w or DEFAULT_WORD


def unwrap(fn):
    """The stock callable beneath every kit wrapper bound at the site (rebind records `__wrapped_stock__`)."""
    seen = 0
    while getattr(fn, "__wrapped_stock__", None) is not None and seen < 8:
        fn = fn.__wrapped_stock__; seen += 1
    return fn


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    try:
        tu = importlib.import_module(TARGET)
        import torch
        import opt_core
        from opt_core.counters import Ledger
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"import:{type(e).__name__}:{str(e)[:80]}".replace(" ", "_"))
    try:
        from opt_core.kernels import triattn as T
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"core:triattn_not_in_opt_core@{opt_core.__version__}:{type(e).__name__}")
    w = word()
    base = w.split("@")[0]
    if base not in tuple(T.ROW_NAMES) + tuple(T.TIER_WORDS):
        return Installed(LEVER, False, reason=f"unknown_word:{w[:40]}".replace(" ", "_"))
    if base == "fast" or (base in T.ROW_NAMES and (T.rows().get(base) or {}).get("class") == "fast"):
        return Installed(LEVER, False, reason=f"not_an_exact_word:{w[:40]}")          # this lever's class is exact: a fast row word belongs to triattn_core
    inner = tu.cueq_tri_attn                                                          # what serves a call this lever steps aside from (the stock op, or the fast rows' tier binding beneath)
    stock_raw = unwrap(inner)                                                         # the stock cuEquivariance op the provider's stock rows call
    beneath = getattr(inner, "__afo_tier_word__", None) if mode != EXACT else None    # a fast / big MODE with the fast tier's binding of this call beneath (hooks/triatt.py): that tier subsumes this one
    cc = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (0, 0)
    try:
        exact_stack = T.exact_stack_key(cc)                                           # '<cc>|torch<v>|cueq<v>' of THIS process: the provider serves an exact-class row only on a stack it vouches byte identity on
    except Exception:  # noqa: BLE001
        exact_stack = None
    from ..registry import LEVERS
    words = tuple(LEVERS[LEVER]["expected"])
    ledger = Ledger(NAME, impl=f"opt_core.kernels.triattn@{opt_core.__version__}:{w}", origin="core", expected=words, max_shapes=16)
    rows: Dict[str, int] = {}; plan: Dict[int, str] = {}; sels: Dict[tuple, object] = {}

    def stock_kw(q, k, v, bias, mask=None, scale=None):
        return stock_raw(q, k, v, bias, mask, scale)

    def select(q5):
        dtype = T.norm_dtype(q5.dtype); D = int(q5.shape[-1]); H = int(q5.shape[-3]); S = int(q5.shape[-2])
        key = (dtype, D, H, S)
        if key not in sels:
            sels[key] = T.select(cc, dtype, D, H, S, "fwd", word=w, form="mask_bias", exact_stack=exact_stack)   # raises T.Refusal by name (dtype_fp32, arch:…, …); an exact-class row only where vouched on this stack
            plan[S] = sels[key].row
        return sels[key]

    def cueq_tri_attn(q, k, v, bias, mask, scale):
        if beneath is not None:                                                       # a fast / big mode: the fast tier's binding beneath serves the call (its own census), by name
            ledger.fallback(f"tier_beneath:{beneath}"); return inner(q, k, v, bias, mask, scale)
        if q.dim() < 5 or mask is None:
            ledger.fallback("rank"); return inner(q, k, v, bias, mask, scale)
        shp = q.shape
        q5, k5, v5 = q.reshape(-1, *q.shape[-4:]), k.reshape(-1, *k.shape[-4:]), v.reshape(-1, *v.shape[-4:])
        b5, m5 = bias.reshape(-1, *bias.shape[-4:]), mask.reshape(-1, *mask.shape[-4:])
        try:
            sel = select(q5)
            if sel.cls == "stock":                                                    # the cell's exact row is the stock op: step aside by name, the bound call serves
                ledger.fallback(f"stock_row:{sel.row}"); return inner(q, k, v, bias, mask, scale)
            o = T.triangle_attention(q5, k5, v5, b5, m5, scale, word=w, stock=stock_kw, selection=sel)
        except T.Refusal as e:
            ledger.fallback(str(e.kind).replace(" ", "_")[:60]); return inner(q, k, v, bias, mask, scale)
        except Exception as e:  # noqa: BLE001 — a kernel that raised: the bound call serves THIS call, the gate refuses at exit (fail-loud)
            ledger.error(type(e).__name__); return inner(q, k, v, bias, mask, scale)
        rows[sel.row] = rows.get(sel.row, 0) + 1
        ledger.serve(f"{int(q5.shape[0])}x{int(q5.shape[-2])}")
        return o.view(*shp)
    cueq_tri_attn.__qualname__ = "cueq_tri_attn[atlasfold_opt:triattn_exact]"
    rebind(tu, "cueq_tri_attn", cueq_tri_attn, inner)
    ledger.set("word", w); ledger.set("exact_stack", str(exact_stack or "none").replace(" ", "_"))

    def line():
        ledger.set("rows", ",".join(f"{r}:{n}" for r, n in rows.items()) or "none")
        ledger.set("plan", ",".join(f"{n}:{r}" for n, r in sorted(plan.items())) or "none")
        return ledger.line(tag)

    return Installed(LEVER, True, lines=[line], gates=[PC.gate_for(ledger, words)],       # all-fallback for listed reasons (an fp32-only / cc 8.0 run) is a legitimate stock run; errors / other words refuse
                     facts={"impl": ledger.impl, "ledger": ledger, "word": w, "rows": rows, "plan": plan, "exact_stack": exact_stack, "beneath": beneath})
