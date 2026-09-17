"""ptx_transition_core.py — every trunk Transition call (protenix.model.modules.primitives.Transition: the pair transitions of the Pairformer / MSA-module /
confidence-head / template pair stacks, the single transition, the MSA transition) through the shared core's ONE transition provider,
opt_core.kernels.transition, by the mode's TIER WORD (levers transition_core_exact | transition_core; switch PTX_TRANSITION=exact|fast|big in env.sh;
a big mode pre-sets big).  No row name, no kit cell table: the provider's select() names the row per cell (cc | stack | dtype | shape | size |
timing) from TRANSITION_CELLS.json and serves it; the exact word serves only rows the table records BITWISE against the statement on this stack from the
call's size up (the module's own LayerNorm output is handed to the row: the exact-candidate construction) and otherwise names the stock row, the
fast | big words serve the cell's measured winner inside the engine band (the row's own fused LayerNorm where it carries one).

What runs the MODULE STATEMENT by name (counted per reason on the LEVER line, one '[FPF] TRCORE ...' line per call class): a call the provider
resolves to a stock row (the statement is the cell's winner, or the exact word below the stack's vouch floor names it), a call it refuses by name
(no cell for the family on this card, rows outside a row's envelope), a call whose (c, hidden, family) the provider's table has no cell WORD for (the
diffusion conditioning's n=2 transitions: c 256 x 512, c 384 x 768), and every call outside the bf16 autocast eval path (training, fp32, CPU).
The block statement's in-block pair transition (blk2_block_path) asks the same face with the residual folded (y = z + t(z)).

Timing column: the provider keys its cells by timing (eager | graph) and under a TIER word the two columns may name
different rows for one call class.  A call the trunk stack graph (fpf_stackgraph) is PLANNING to capture -- its eager reference call and side-stream warm-up of a
first-sighted signature -- is selected from the GRAPH column exactly like the call inside the capture (`planned`), so the graph's bitwise oracle and the graph serve
the same rows; `capture=` stays the literal capture state (the provider's capture-unsafe refusals and its priming outside a capture key on it: priming a row with
process-level initialisation during the planned eager reference is what makes it capture-ready right after).  Under the EXACT word nothing changes: timing follows
the literal capture state (every admissible exact row equals the module statement bitwise by vouch, so the oracle holds whichever column answers).
"""
from opt_core.oom import is_oom   # a broad handler that reroutes around a lever re-raises device out-of-memory first
import math
import os
import sys

import torch

WORDS = ("exact", "fast", "big")            # the provider's tier words this kit binds (opt_core.kernels.transition.TIER_WORDS carries them and 'faithful')
MARK = "TRCORE:"                              # protenix_opt.stack.MARKERS: 'TRCORE:on(word=<w> ...' = installed; ':unavailable(' / ':refused(' carry BAD
PAIR_C = (128, 256)                           # pair widths (c_z of the shipped configurations); c 64 rows are the table's `rows` cells (MSA / template stacks), other widths `single`
_ST = {"word": None, "asked": None, "installed": False, "opt_core": None, "calls": 0, "served": 0, "module": 0, "errors": 0,
       "rows": {}, "module_reasons": {}, "selections": {}}
_SEL = {}                                     # (c, hidden, family, n_tokens, residual, timing) -> ("serve", "<row arm>") | ("module", "<reason>")
_ONCE = set()
_ORIG = {}


def _T():
    from opt_core.kernels import transition as T   # the shared core's transition provider: one face over every carried row + TRANSITION_CELLS.json
    return T


def _stackgraph_planning():
    """True while the trunk stack graph (src/fpf_stackgraph) runs the eager reference / warm-up / capture of a signature it is about to capture (its plan-for-capture
    window); False when that module is not loaded in this process (big holds no stack graph; a CPU test) or predates the window."""
    for name in ("fpf_stackgraph.stackgraph", "fpf_stackgraph"):
        f = getattr(sys.modules.get(name), "planning", None)
        if f is not None:
            try:
                return bool(f())
            except Exception:                                       # noqa: BLE001 -- a foreign / partial module: no window
                return False
    return False


def timing_word(word, capturing):
    """The provider timing column of a call: 'graph' inside a CUDA-graph capture; under a TIER word (fast | big) also 'graph' for a call the trunk stack graph is
    planning to capture (its eager reference and warm-up), so the oracle and the graph serve one row; 'eager' otherwise.  The exact word keys on the literal
    capture state alone (unchanged: its rows equal the statement bitwise by vouch whichever column answers)."""
    planned = bool(capturing) or (word != "exact" and _stackgraph_planning())
    return "graph" if planned else "eager"


def _say_once(key, line):
    if key not in _ONCE:
        _ONCE.add(key)
        print(line, file=sys.stderr, flush=True)


def _count(d, k, n=1):
    d[k] = d.get(k, 0) + n


def classify(x, c):
    """(family, n_tokens, rows) of a transition input x [..., c]: square [.., N, N, c] inputs are pair-like keyed by N (c 64 -> the `rows` cells, keyed
    by the N whose N^2 equals the row count); other inputs of width 64 are MSA rows [S, N, 64] keyed by the equivalent N = sqrt(rows); the rest are
    `single` rows keyed by their row count."""
    rows = x.numel() // c if c else 0
    square = x.dim() >= 3 and int(x.shape[-2]) == int(x.shape[-3])
    if c == 64:
        fam = "rows"
    elif c in PAIR_C:
        fam = "pair"
    else:
        fam = "single"
    if fam == "single":
        n = rows
    else:
        n = int(x.shape[-2]) if square else int(round(math.sqrt(rows)))
    return fam, max(int(n), 1), rows


def weights(module, T):
    W = getattr(module, "_ptx_trcore_W", None)
    if W is None:
        ln = module.layernorm1
        W = T.pack(w_o=module.linear_no_bias.weight, w_a=module.linear_no_bias_a.weight, w_b=module.linear_no_bias_b.weight,
                   ln_w=getattr(ln, "weight", None), ln_b=getattr(ln, "bias", None), eps=float(getattr(ln, "eps", 1e-5)),
                   device=module.linear_no_bias.weight.device)
        try:
            module._ptx_trcore_W = W
        except Exception:                                           # noqa: BLE001 -- a module that refuses attributes packs per call (never seen on protenix)
            pass
    return W


def _eligible(module, x):
    if module.training:
        return "training"
    if not x.is_cuda:
        return "cpu"
    if x.dtype != torch.bfloat16 or not torch.is_autocast_enabled():
        return "not_bf16_autocast"
    return None


def serve(module, x, residual=False):
    """One Transition call through the provider by the installed tier word -> y ([..., c]: t(x), or x + t(x) with ``residual``), or None when the module
    statement answers this call by name (counted under its reason).  Never raises a provider refusal; device out-of-memory propagates."""
    st = _ST
    if not st["installed"]:
        return None
    st["calls"] += 1
    c = int(x.shape[-1])
    why = _eligible(module, x)
    if why is not None:
        st["module"] += 1; _count(st["module_reasons"], why)
        return None
    word = st["word"]
    T = _T()
    hidden = int(module.linear_no_bias_a.weight.shape[0])
    fam, n, rows = classify(x, c)
    if T.cell_word(c, hidden, fam) is None:                        # the table defines no cell word for this (c, hidden, family): the statement, by name
        r = "no_cell_word:c%d_h%d_%s" % (c, hidden, fam)
        st["module"] += 1; _count(st["module_reasons"], r)
        _say_once(("ncw", c, hidden, fam), "[FPF] TRCORE word=%s c=%d hidden=%d family=%s -> module statement by name (the provider's table has no cell word for the shape)" % (word, c, hidden, fam))
        return None
    capturing = bool(torch.cuda.is_current_stream_capturing())
    timing = timing_word(word, capturing)                          # graph inside a capture AND for a call the stack graph plans to capture (tier words); capture= below stays the literal state
    key = (c, hidden, fam, n, bool(residual), timing)
    d = _SEL.get(key)
    if d is None:                                                   # one selection per call class decides serve | statement (the exact word never lets the provider run its own torch statement: the MODULE's statement is the stock op by name)
        try:
            sel = T.select(word, c=c, hidden=hidden, n_tokens=n, dtype="bf16", family=fam, residual=bool(residual), rows_count=rows,
                           ln_given=(word == "exact"), timing=timing, capture=capturing, device=x.device)
            arm = sel.row + ((":" + sel.variant) if sel.variant else "") + (("@" + T.cfg_word(sel.cfg)) if (sel.cfg and sel.row == "v2") else "")
            if sel.row in T.STOCK_ROWS:
                d = ("module", "stock_cell" if sel.cell_key else "stock_word")
            else:
                d = ("serve", arm)
            st["selections"]["%s:c%dx%d:%s" % (fam, c, hidden, "res" if residual else "nores")] = sel.line()
            _say_once(("sel", c, hidden, fam, d), "[FPF] TRCORE word=%s c=%d hidden=%d family=%s n_tokens=%d -> %s (%s)" % (word, c, hidden, fam, n, ("row " + arm) if d[0] == "serve" else "module statement by name", sel.line()))
        except T.Refusal as r:
            d = ("module", "refused:%s" % r.kind)
            _say_once(("ref", c, hidden, fam, r.kind), "[FPF] TRCORE word=%s c=%d hidden=%d family=%s n_tokens=%d -> module statement by name (provider refusal %s, fallback row %s)" % (word, c, hidden, fam, n, r.kind, r.fallback))
        _SEL[key] = d
    if d[0] == "module":
        st["module"] += 1; _count(st["module_reasons"], d[1])
        return None
    W = weights(module, T)
    x_ln = module.layernorm1(x) if word == "exact" else None         # the exact-candidate construction: the module's own LayerNorm output, the row projects it
    try:
        y, sel = T.transition(x, W, word=word, residual=bool(residual), x_ln=x_ln, n_tokens=n, family=fam, timing=timing, capture=capturing)
    except T.Refusal as r:                                         # a per-call refusal by name (operand strides, a launch that cannot build here): the statement answers, counted
        st["module"] += 1; _count(st["module_reasons"], "refused:%s" % r.kind)
        _say_once(("ref2", c, hidden, fam, r.kind), "[FPF] TRCORE word=%s c=%d hidden=%d family=%s -> module statement by name for such calls (provider refusal %s)" % (word, c, hidden, fam, r.kind))
        return None
    except Exception as e:                                          # noqa: BLE001
        if is_oom(e):
            raise
        st["errors"] += 1; st["module"] += 1; _count(st["module_reasons"], "error:%s" % type(e).__name__)
        if st["errors"] <= 2:
            print("[FPF] TRCORE word=%s c=%d hidden=%d family=%s: provider error %r -> module statement for this call" % (word, c, hidden, fam, e), file=sys.stderr, flush=True)
        return None
    arm = sel.row + ((":" + sel.variant) if sel.variant else "") + (("@" + T.cfg_word(sel.cfg)) if (sel.cfg and sel.row == "v2") else "")
    st["served"] += 1; _count(st["rows"], "%s@%s" % (arm, sel.cell_key or "none"))
    return y


def transition_forward(self, x):
    """primitives.Transition.forward under the lever: the provider serves the eval call, else the module's own statement (stock forward) by name."""
    orig = _ORIG["forward"]
    if self.training:
        return orig(self, x)
    y = serve(self, x, residual=False)
    return y if y is not None else orig(self, x)


def module_forward(module, x):
    """The module's own statement (the stock forward this lever displaced, or the live one when the lever is not installed)."""
    f = _ORIG.get("forward")
    return f(module, x) if f is not None else module(x)


def block_pair_transition(module, z):
    """The block statement  z += pair_transition(z)  : through the provider with the residual folded (a new tensor y = z + t(z)) when the lever is
    installed and the provider serves the call, else the stock statement in place on the module's own forward."""
    if _ST["installed"]:
        y = serve(module, z, residual=True)
        if y is not None:
            return y
    z += module_forward(module, z)
    return z


def apply(word):
    """Install: bind primitives.Transition.forward to the provider under tier word ``word`` -> the `applied` marker ('TRCORE:on(...)' or a BAD
    'TRCORE:refused(...)' / 'TRCORE:unavailable(...)' the mode reads as a fallback by name)."""
    _ST["asked"] = word
    if word not in WORDS:
        return MARK + "refused(word=%s not one of %s)" % (word, "|".join(WORDS))
    try:
        T = _T()
        import opt_core
        core_v = getattr(opt_core, "__version__", "?")
    except Exception as e:                                          # noqa: BLE001 -- an older core without the provider: the lever is a fallback by name, the statement runs
        return MARK + "unavailable(%s: %s)" % (type(e).__name__, str(e)[:120])
    if word not in getattr(T, "TIER_WORDS", ()):                    # an older core whose transition face lacks the word (big: opt_core >= 0.5.113.0): the lever is a fallback by name
        return MARK + "refused(word=%s not a tier word of opt_core %s kernels.transition)" % (word, core_v)
    import protenix.model.modules.primitives as PR
    if "forward" not in _ORIG:
        _ORIG["forward"] = PR.Transition.forward
    PR.Transition.forward = transition_forward
    _ST.update(word=word, installed=True, opt_core=core_v)
    return MARK + "on(word=%s bind=tier:%s provider=opt_core.kernels.transition opt_core=%s sites=Transition.forward+block_pair_transition)" % (word, word, core_v)


def report():
    """Live counters for the LEVER line / the kit report: word, calls, served (provider rows) with the per-row census, module (statement by name) with
    the per-reason census, errors, the core version."""
    st = _ST
    return {"word": st["word"], "asked": st["asked"], "installed": st["installed"], "opt_core": st["opt_core"], "calls": st["calls"],
            "served": st["served"], "module": st["module"], "errors": st["errors"], "rows": dict(st["rows"]),
            "module_reasons": dict(st["module_reasons"]), "selections": dict(st["selections"])}
