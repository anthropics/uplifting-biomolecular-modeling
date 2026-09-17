"""Lever transition_exact — atlasfold's SwiGLU Transition (stock primitives/transition.py: ``Transition(channel, n)`` = nn.Sequential(LayerNorm(C),
SwiGLU(LinearNoBias C -> 2*n*C; chunk a | b, silu(a) * b), LinearNoBias n*C -> C); ``forward(self, x)`` returns the UPDATE the caller adds, block.py
L281 ``z = z + transition_z(z)``) bound BY TIER WORD to the core's transition provider ``opt_core.kernels.transition``, fed the module's OWN
LayerNorm output — the exact-class construction.

Word: ``AFO_TRANSITION_EXACT_WORD`` (default the tier word ``exact``).  The provider resolves the word per call class (cc, dtype, C, hidden,
n_tokens, the running torch / triton stack): a carried row serves the exact word only where the provider's table records it BITWISE against the
statement on THIS stack at or below the call's size (its vouch records and exact floor decide — no kit-side size floor, ceiling or row pin);
everywhere else the provider names the stock statement and the module's OWN forward runs, BY NAME: `stock_row:<cell>` (the cell's exact winner is
the statement: e.g. the single track's C = 384 / 768 transitions, `stock_row:c384` / `stock_row:c768`; a pair cell whose exact winner is the
statement, `stock_row:n<size>`), `refused:<kind>` (ANY provider Refusal: no cell for the shape, no vouch recorded on this stack, below the
stack's vouched floor, ...).  A ROW word (``v1``, ``flash_sm90a``, ...) is the operator's explicit choice, served wherever the provider admits it
(printed ``pinned=1``); the tier words ``fast`` / ``big`` are not exact-class words: the lever installs SKIPPED by name.

Per call, BY NAME (LEVER census): `training`, `dtype:<t>` (bf16 rows only: the fp32 diffusion transitions run the statement), `rank`, `cpu`,
`stock_row:<cell>`, `refused:<kind>`, `error:<Type>` (a row that raised: the statement serves THIS call, the gate refuses at exit).
Fields: word= rows=<row:n> plan=<C/N:row/class | stock | refused:kind> pinned= copies=.
ORDER: after conf_transition_chunk / pair_transition_chunk (their row-block wrapper is this lever's stock callable) and BEFORE the fast lever
pair_transition in the fast / big rows (exact ⊂ fast: there the fast binding is outermost and this lever serves what it declines).
Individually switchable: MODEL_OPT_LEVERS_OFF=transition_exact.  Class: exact."""
import os
from typing import Optional

from . import Installed, rebind
from . import transition_chunk as TC
from . import pair_cells as PC

LEVER = "transition_exact"
NAME = "LOCAL.fused_transition"
TARGET = "atlasfold.model.network.primitives.transition"
WORD_ENV = "AFO_TRANSITION_EXACT_WORD"
WORD_DEFAULT = "exact"
EXACT_TIER_WORDS = ("exact", "faithful")                              # the provider's tier words an exact-class lever may carry (fast / big resolve tolerance rows)
FAMILY_OF_C = {128: "pair", 384: "single", 768: "single"}             # atlasfold widths -> the provider's cell family (pair c_z 128; single c_s 384; the LM stack's 768)


def word() -> str:
    return (os.environ.get(WORD_ENV) or "").strip() or WORD_DEFAULT


def word_ok(w: str, KT) -> Optional[str]:
    """None when `w` is a word this exact-class lever may carry, else the install-skip reason."""
    if w in EXACT_TIER_WORDS:
        return None
    if w in KT.TIER_WORDS:
        return f"word_not_exact_class:{w}"
    row = KT.split_word(w)[0]
    if row not in KT.ROW_NAMES:
        return f"unknown_word:{w}"
    if row in KT.STOCK_ROWS or row in KT.SCHEDULE_ROWS:
        return f"word_names_no_kernel:{w}"
    return None


def geometry(module):
    """(C, factor) of a stock Transition: C = the out projection's rows, factor = the SwiGLU linear's rows / 2C."""
    C = int(module[2].weight.shape[0])
    return C, int(module[1].linear.weight.shape[0]) // (2 * C)


def refusal(module, x) -> Optional[str]:
    """The structural reason this call runs the stock statement before the provider is asked, or None."""
    import torch
    if module.training:
        return "training"
    if not torch.is_tensor(x):
        return "rank"
    if x.dtype != torch.bfloat16:
        return "dtype:" + str(x.dtype).replace("torch.", "")
    if x.dim() < 3:
        return "rank"
    if not x.is_cuda:
        return "cpu"
    return None


def weights(module, KT):
    """The module's parameters packed ONCE for the provider (opt_core.kernels.transition.pack: w_ab rows a-then-b = the stock chunk order, w_o,
    the LayerNorm affine + eps) and cached on the module; a linear carrying a bias is refused `bias:<name>`, never dropped."""
    W = getattr(module, "_afo_kt_transition_w", None)
    if W is None:
        for name, lin in (("swiglu", module[1].linear), ("out", module[2])):
            if getattr(lin, "bias", None) is not None:
                raise KT.Refusal("bias:" + name, None, "torch_swiglu")
        ln = module[0]
        W = KT.pack(w_o=module[2].weight, w_ab=module[1].linear.weight, ln_w=ln.weight, ln_b=ln.bias, eps=float(ln.eps), device=module[2].weight.device)
        module._afo_kt_transition_w = W
    return W


def stock_word(sel, C, N) -> str:
    """`stock_row:<cell>` for a Selection naming the statement: the single track by width (`c384` / `c768`), a pair cell by its size word
    `N<=<n>` spelled `n<n>` (LEVER fields carry no `=` / `<`)."""
    if C != 128:
        return f"stock_row:c{C}"
    cellw = (sel.cell_key or "").split("|")
    size = ("n" + cellw[3].split("=")[-1]) if len(cellw) > 3 and cellw[3].startswith("N") else f"n{N}"
    return "stock_row:" + size


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    try:
        TR = importlib.import_module(TARGET)
        import torch  # noqa: F401
        import opt_core
        from opt_core.kernels import transition as KT
        from opt_core.counters import Ledger
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"import:{type(e).__name__}:{str(e)[:80]}")
    w = word()
    bad = word_ok(w, KT)
    if bad is not None:
        return Installed(LEVER, False, reason=bad)
    from ..registry import LEVERS
    words = tuple(LEVERS[LEVER]["expected"])
    pinned = w not in KT.TIER_WORDS
    ledger = Ledger(NAME, impl=f"opt_core.kernels.transition@{opt_core.__version__}:{w}", origin="core", expected=words)
    ledger.set("copies", 0); ledger.set("word", w); ledger.set("pinned", int(pinned)); ledger.set("rows", "none"); ledger.set("plan", "none")
    plan, rows = {}, {}                                # "C/N" -> row/class | stock | refused:kind ; row -> calls
    cache = {}                                        # (C, hidden, N) -> ("serve", row_word, cls) | ("stock", word) | ("refused", word)
    stock = TR.Transition.forward                    # the transition_chunk wrapper when that lever is in the row (installed before this one), else nn.Sequential.forward

    def resolve(C, hidden, N):
        key = (C, hidden, N)
        got = cache.get(key)
        if got is None:
            fam = FAMILY_OF_C.get(C, "pair")
            try:
                sel = KT.select(w, c=C, hidden=hidden, n_tokens=N, dtype="bf16", direction="fwd", timing="eager", family=fam, ln_given=True)
                if sel.row in KT.STOCK_ROWS:              # the provider names the statement for this cell: the module's OWN forward runs, by name
                    got = ("stock", stock_word(sel, C, N))
                else:
                    got = ("serve", sel.row + (f":{sel.variant}" if sel.variant else ""), str(sel.cls))
            except KT.Refusal as e:
                got = ("refused", "refused:" + str(e.kind).split(":")[0])
            cache[key] = got
            plan[f"{C}/{N}"] = (got[1] + (f"/{got[2]}" if got[0] == "serve" else "")) if got[0] != "stock" else "stock"
            ledger.set("plan", ",".join(f"{k}:{v}" for k, v in sorted(plan.items())))
        return got

    def forward(self, x):
        why = refusal(self, x)
        if why is not None:
            ledger.fallback(why)
            return stock(self, x)
        C, factor = geometry(self)
        N = int(x.shape[-2])
        got = resolve(C, factor * C, N)
        if got[0] != "serve":
            ledger.fallback(got[1])
            return stock(self, x)
        row_word = got[1]
        xin = x
        if not xin.is_contiguous():                   # the rows are flattened for the kernel: a strided x would copy silently there — copy here, counted
            xin = xin.contiguous(); ledger.count("copies")
        err = ref = None
        try:
            W = weights(self, KT)
            x_ln = self[0](xin)                       # the module's OWN LayerNorm, as installed (stock statement, ln_bf16's or exactln's): bf16 in -> bf16 out
            if not x_ln.is_contiguous():
                x_ln = x_ln.contiguous(); ledger.count("copies")
            u, _sel = KT.transition(xin, W, word=row_word, residual=False, x_ln=x_ln, n_tokens=N, family=FAMILY_OF_C.get(C, "pair"), timing="eager")
        except KT.Refusal as e:                       # the provider's named refusal at serve time: keep the WORD (head) only
            ref = "refused:" + str(e.kind).split(":")[0]
        except Exception as e:  # noqa: BLE001 — a kernel that raised: this call runs the stock statement, the gate refuses at exit
            err = type(e).__name__
        if ref is not None or err is not None:
            if ref is not None:
                ledger.fallback(ref)
            else:
                ledger.error(err)
            return stock(self, x)
        rows[row_word] = rows.get(row_word, 0) + 1
        ledger.set("rows", ",".join(f"{k}:{v}" for k, v in sorted(rows.items())))
        ledger.serve(f"{'x'.join(str(int(d)) for d in x.shape[:-2])}x{N}")
        TC.note_upstream(x, "upstream_transition_exact")                    # the row-block wrapper below never sees this call: its ledger names who served it
        return u
    forward.__qualname__ = f"Transition.forward[atlasfold_opt:{LEVER}]"
    rebind(TR.Transition, "forward", forward, stock)
    return Installed(LEVER, True, lines=[lambda: ledger.line(tag)], gates=[PC.gate_for(ledger, words)],   # kernel errors / unexpected words refuse; the registry's words match by prefix; an all-fallback run (every call the statement by name) is a legitimate stock run
                     facts={"impl": ledger.impl, "ledger": ledger, "word": w, "pinned": pinned})
