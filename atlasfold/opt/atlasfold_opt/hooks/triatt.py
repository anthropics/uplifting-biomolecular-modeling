"""Lever flash_triattn — atlasfold's cuEquivariance triangle-attention wrapper ``cueq_tri_attn(q, k, v, bias, mask, scale)``
(triangle_update.py L94-116), called by TriangleAttentionStartingNode/EndingNode.forward when kernel_backend == "cuequiv" (L483-491),
bound to the core's triangle-attention provider `opt_core.kernels.triattn` BY THE MODE'S TIER WORD (``fast`` under --mode fast, ``big``
under --mode big): the provider serves the measured cell's row for that word at each call class (cc, dtype, head_dim, heads, keys, mask form)
— its own table decides the row per cell and card — so no triangle-attention call of the fast rows bypasses the
provider.  Shapes at the call: q/k/v [B, L, H=4, L, 32] (bf16 in the trunk; fp32 under autocast in the confidence stacks — the provider reads the
autocast dtype as the compute dtype), bias [B, 1, H, L, L], mask [B, L, 1, 1, L] bool, scale = 32**-0.5.
In the fast / big rows the fused block (`triatt_block`) consumes the pair stacks' calls from its floor (512 tokens) up, so this binding serves the
calls below it (and reads `no_calls` on larger inputs).  Steps aside BY NAME per call, the stock cuEquivariance op serving that call: the provider's
`Refusal.kind` (`no_cell:…`, `unknown_word:…`, `dtype_…`, …), `stock_row:<row>` (the cell's word IS a stock row), `rank` (not the
[.., N, H, N, D] pair-stack call); a kernel that raises refuses the gate at exit (fail-loud) after the stock op served that call.
Individually switchable: MODEL_OPT_LEVERS_OFF=flash_triattn.
LEVER line: name=F1.flash_triattn impl=opt_core.kernels.triattn@<core>:<word> origin=core served=<calls> fallback=<n> fallback_by=<word:n>
shapes=<BxN:n> plan=<N:row,…> rows=<row:n> word=<word>."""
from typing import Dict

from . import Installed, rebind

LEVER = "flash_triattn"
NAME = "F1.flash_triattn"
TARGET = "atlasfold.model.network.primitives.triangle_update"
TIER_OF_MODE = {"fast": "fast", "big": "big"}
FORM = "mask_bias"                                      # the call form: per-row key mask + pair bias (kernels.triattn's `forms` cell key)
QUALNAME = "cueq_tri_attn[atlasfold_opt:flash_triattn]"  # hooks/triattn_exact reads it: the fast tier's binding of the same call beneath the exact lever


def word(mode: str = "fast") -> str:
    """The MODE's tier word (this lever is a fast-row lever: `fast` | `big`)."""
    return TIER_OF_MODE.get(mode, "fast")


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
        from opt_core.kernels import triattn as T
        from opt_core.attn import pair_fused as PF
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"import:{type(e).__name__}:{str(e)[:80]}".replace(" ", "_"))
    w = word(mode)
    inner = tu.cueq_tri_attn                                                          # what serves a call this lever steps aside from
    stock_raw = unwrap(inner)                                                         # the stock cuEquivariance op the provider's stock rows call
    cc = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (0, 0)
    from ..registry import LEVERS
    words = tuple(LEVERS[LEVER]["expected"])
    ledger = Ledger(NAME, impl=f"opt_core.kernels.triattn@{opt_core.__version__}:{w}", origin="core", expected=words, max_shapes=16)
    rows: Dict[str, int] = {}; plan: Dict[int, str] = {}; sels: Dict[tuple, object] = {}

    def stock_kw(q, k, v, bias, mask=None, scale=None):
        return stock_raw(q, k, v, bias, mask, scale)

    def select(q5):
        """The provider's Selection for the word at this call class, resolved as attn.pair_fused resolves its tier core (the running process's
        prebuilt key for the CUDA rows; the autocast dtype is the compute dtype).  Raises T.Refusal by name."""
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else q5.dtype
        key = (T.norm_dtype(dt), int(q5.shape[-1]), int(q5.shape[-3]), int(q5.shape[-2]))
        if key not in sels:
            sels[key] = PF.resolve_tier_core(w, cc, key[0], key[1], key[2], key[3], form=FORM)
            plan[key[3]] = sels[key].row
        return sels[key]

    def cueq_tri_attn(q, k, v, bias, mask, scale):
        if q.dim() < 5 or mask is None:
            ledger.fallback("rank"); return inner(q, k, v, bias, mask, scale)
        shp = q.shape
        q5, k5, v5 = q.reshape(-1, *q.shape[-4:]), k.reshape(-1, *k.shape[-4:]), v.reshape(-1, *v.shape[-4:])
        b5, m5 = bias.reshape(-1, *bias.shape[-4:]).float(), mask.reshape(-1, *mask.shape[-4:])
        try:
            sel = select(q5)
            if sel.cls == "stock":                                                    # the cell's word IS a stock row: step aside by name, the bound call serves
                ledger.fallback(f"stock_row:{sel.row}"); return inner(q, k, v, bias, mask, scale)
            o = T.triangle_attention(q5, k5, v5, b5, m5, scale, word=w, stock=stock_kw, selection=sel, form=FORM)
        except T.Refusal as e:
            ledger.fallback(str(e.kind).replace(" ", "_")[:60]); return inner(q, k, v, bias, mask, scale)
        except Exception as e:  # noqa: BLE001 — a kernel that raised: the bound call serves THIS call, the gate refuses at exit (fail-loud)
            ledger.error(type(e).__name__); return inner(q, k, v, bias, mask, scale)
        rows[sel.row] = rows.get(sel.row, 0) + 1
        ledger.serve(f"{int(q5.shape[0])}x{int(q5.shape[-2])}")
        return o.view(*shp)
    cueq_tri_attn.__qualname__ = QUALNAME
    cueq_tri_attn.__afo_tier_word__ = w
    rebind(tu, "cueq_tri_attn", cueq_tri_attn, inner)
    ledger.set("word", w)

    def line():
        ledger.set("rows", ",".join(f"{r}:{n}" for r, n in sorted(rows.items())) or "none")
        ledger.set("plan", ",".join(f"{n}:{r}" for n, r in sorted(plan.items())) or "none")
        return ledger.line(tag)

    from . import pair_cells as PC
    return Installed(LEVER, True, lines=[line], gates=[PC.gate_for(ledger, words)],
                     facts={"impl": ledger.impl, "ledger": ledger, "word": w, "rows": rows, "plan": plan, "cc": cc})
