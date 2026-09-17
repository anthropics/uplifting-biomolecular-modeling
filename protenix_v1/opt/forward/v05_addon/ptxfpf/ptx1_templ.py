"""ptx1_templ — the kit's levers on the TEMPLATE EMBEDDER of the pip-installed protenix 1.1.0 (protenix/model/modules/pairformer.py
TemplateEmbedder, AF3 Algorithm 16: per template t, v_t = linear_z(LN(z)) + linear_a(features_t) through a 2-block c = 64 Pairformer,
u = sum_t LN_v(v_t); u / (1e-7 + T) -> linear_u(relu(u)) added to z once per recycle).

Why it runs at the CLI defaults (`protenix pred`: use_template False): the featuriser assembles the template block whatever --use_template says
(data/template/template_featurizer.py TemplateFeatureAssemblyLine(max_templates=4): a chain without a template hit carries the
`empty_template_features` slot — ONE template of restype `-` = 31, all-zero positions and masks — padded to 4 slots with zeros), the checkpoints'
model config sets template_embedder.n_blocks = 2, and TemplateEmbedder.forward returns 0 early ONLY when `template_aatype` is absent or
n_blocks < 1. So on every item of every mode the c = 64 stack runs T = 4 templates x 2 blocks x N_cycle recycles on features that carry no
structural information (slot 0 = restype 31 everywhere, slots 1..3 = restype 0 everywhere, every distogram / unit-vector / mask plane zero;
describe() restates the grouping per item). Its contribution to z is NOT input-independent — v_t depends on z at every recycle — so no exact
`skip` exists; but slots with identical features give identical v_t, which is what `template_dedupe` uses.

  template_dedupe   EXACT class, every mode. Distinct-template evaluation: the template slots are grouped by elementwise equality of the five
                    template-indexed features single_template_forward reads (TEMPLATE_KEYS; torch.equal on device, once per feature dict = once
                    per item), each group's representative is evaluated ONCE and the stock accumulation runs verbatim over the slots in the stock
                    order with the representative's tensor: u = (((0 + v_0) + v_1) + v_1) + v_1 at the CLI defaults (2 distinct of 4). Bitwise:
                    the stock loop computes single_template_forward on identical input values with identical statements for the duplicate slots;
                    the additions are the same additions on the same values in the same order. A feature dict without the five keys, or one whose
                    slots are all distinct, runs the stock number of evaluations through the same statements.
  tmpl_triatt       TOLERANCE class (fast, big). The template Pairformer's triangle attention (c_in 64, 4 heads x 32, starting + ending node)
                    through the SAME fused block the trunk's `gflash` uses (opt_core.attn.pair_fused, impl lnl: the LN+bias kernel at (64,) and the
                    gate kernel at (128,), one cuBLAS q|k|v|g GEMM, the trunk lever's attention core — under gblock the provider's `exact` tier
                    word when `triexact` rides it, keyed at the template call's own row length; no template twin of the word). The routing decision is made ONCE per
                    (lever, c, heads, dim) from the core's cell table (pair_fused.preflight) — served cells: the block answers the template calls
                    (levers_ptx1's census counts them with the trunk's); no servable cell: the stock statement BY NAME
                    (`stock:tmpl_triatt:<the core's word>`), never a refused call.
  tmpl_trimul       TOLERANCE class (fast, big) and
  tmpl_trimul_exact EXACT class (exact). The template Pairformer's triangle multiplication (outgoing + incoming, c_z 64, c_hidden 128 — the stock
                    takes its TORCH path there because c_z != c_hidden) through the core's triangle-multiplication provider (opt_core.trimul.by_word
                    over opt_core.kernels.trimul) by the MODE's TIER word (TRIMUL_WORDS): `fast` under --mode fast and `big` under --mode big
                    serve the provider's row for the call's cell (cc, precision, c_z, c_hidden, N bucket, direction); `exact` (tmpl_trimul_exact,
                    the exact arm) serves an exact-class kernel row the provider vouches bitwise on this stack and names the stock op where it has
                    none. A selection that IS the stock's class of computation (STOCK_ROWS, or class stock) is nothing to launch: the lever idles
                    BY NAME (`stock:tmpl_trimul:<row>:<why>`, exit 0); a call the provider refuses is answered by the stock statement BY NAME
                    (`stock:tmpl_trimul:<row>:<kind>`); no kit-side size floor or cell table — the provider's cells decide.
  tmpl_xtr          EXACT class (exact, fast, big). The template Pairformer's pair Transition (c 64, n 2) — first through the core's transition
                    PROVIDER by the mode's tier word (opt_core.kernels.transition, XTR_WORDS: exact | fast | big; family `rows` keyed by the
                    item's tokens; a Refusal by name -> the construction below), else through opt_core.attn.pair_fused's transition in the trunk
                    lever `xtr`'s exact construction (impl fpf, the stock LayerNorm's output handed in: ln stock) — the statement
                    levers_ptx1._transition_forward serves at c 128, at the template width. A shape the table does not serve is answered by the
                    stock statement BY NAME (`stock:tmpl_xtr:<word>`), a refusal at call time by `fallback:tmpl_xtr:<reason>` (the stock statement
                    for that call). On a card with a served-row rule for the exact transition (opt_core.attn.pair_fused exact_rows: a cuBLAS
                    piece-remainder effect makes the fused transition's bits differ from the stock's outside that range) the template transition
                    rides the SAME rule (`stock:tmpl_xtr:above_max_rows` / `...:below_min_rows`), exactly where `xtr` steps aside; a card without
                    an entry has no range (every call served).
  tmpl_pairfused    TOLERANCE class (fast, big). The template triangle-attention block's prologue / epilogue in impl fpf WITH the LayerNorm fused
                    (pair_fused cells fpf prologue / epilogue (64, 4, 32)) instead of tmpl_triatt's impl lnl — the core is the trunk lever's; an
                    impl choice for the calls tmpl_triatt routes (nothing without tmpl_triatt); the cells triatt_gate checks are then the fpf ones.

Grammar: the words join levers_ptx1's arm string (`<trimul>[+lever...]`); `LEVER_NAMES` here is the tuple levers_ptx1 extends its own with.
Census: `COUNTS` (per-call words, levers_ptx1's convention) and `describe()` (the account the package's report reads).
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import math
import torch

LEVER_NAMES = ("template_dedupe", "tmpl_triatt", "tmpl_trimul", "tmpl_xtr", "tmpl_pairfused", "tmpl_trimul_exact")
CFG = {"template_dedupe": False, "tmpl_triatt": False, "tmpl_trimul": False, "tmpl_xtr": False, "tmpl_pairfused": False, "tmpl_trimul_exact": False}
CLASS = {"template_dedupe": "exact", "tmpl_triatt": "tolerance", "tmpl_trimul": "tolerance", "tmpl_xtr": "exact", "tmpl_pairfused": "tolerance", "tmpl_trimul_exact": "exact"}
TEMPLATE_KEYS = ("template_aatype", "template_distogram", "template_pseudo_beta_mask", "template_unit_vector", "template_backbone_frame_mask")   # the template-indexed features single_template_forward reads (pairformer.py:1056-1084)
TE_MODULE, TE_CLASS = "protenix.model.modules.pairformer", "TemplateEmbedder"
COUNTS: Dict[str, Dict[str, int]] = {"dedupe": {}, "triatt": {}, "trimul": {}, "xtr": {}, "pairfused": {}}
TRIMUL_WORDS = {"exact": "exact", "fast": "fast", "big": "big"}     # opt_core.kernels.trimul TIER word per kit mode for the template TriMul (c_z 64, c_hidden 128):
                                                                    # `tmpl_trimul` asks `fast` under --mode fast and `big` under --mode big (the provider's fast-class
                                                                    # row for the call's cell on this card), `tmpl_trimul_exact` asks `exact` (an exact-class KERNEL row the
                                                                    # provider vouches bitwise on this stack, else it names the stock op -> idle by name). No size floor here:
                                                                    # the provider's cell table decides per (cc, dtype, C, H, N bucket, direction).
TRIMUL_WORD = None                                                  # a fixed provider word for every call (None = the mode's tier word above; no kit mode sets it)
STOCK_ROWS = ("cueq", "torch_math")                                 # provider rows that ARE the stock statement's class of computation: a selection naming one = nothing to serve, by name
_TM: Dict[str, object] = {"provider": None, "error": None, "first": None}
_ORIG: Dict[str, object] = {}
_STATE: Dict[str, object] = {"installed": False, "error": None, "group_key": None, "groups": None, "items": 0, "last_groups": "-", "slots_max": 0}
_DECISIONS: Dict[Tuple, Tuple[bool, str, str]] = {}                 # (lever, c, H, D, device) -> (served, word, cells)


def _c(kind: str, word: str, n: int = 1) -> None:
    d = COUNTS.setdefault(kind, {})
    d[word] = d.get(word, 0) + n


# =====================================================================================  template_dedupe
def template_groups(feats) -> Optional[List[int]]:
    """[representative slot per template slot] by elementwise equality of TEMPLATE_KEYS along the template axis (slot t's representative is
    the FIRST slot r <= t whose five features equal slot t's); None when the dict does not carry the five keys as tensors of one template
    count (the stock statements then run as they are)."""
    if not isinstance(feats, dict):
        return None
    vals = [feats.get(k) for k in TEMPLATE_KEYS]
    if any(v is None or not torch.is_tensor(v) or v.dim() < 1 for v in vals):
        return None
    T = int(vals[0].shape[0])
    if any(int(v.shape[0]) != T for v in vals):
        return None
    reps: List[int] = []
    for t in range(T):
        rep = t
        for r in sorted(set(reps)):
            if all(torch.equal(v[t], v[r]) for v in vals):       # aatype first: the cheap [N] row decides the CLI-default case before any [N,N,·] plane is read
                rep = r
                break
        reps.append(rep)
    return reps


def _group_key(feats) -> Tuple:
    return (id(feats),) + tuple((feats[k].data_ptr(), tuple(feats[k].shape)) for k in TEMPLATE_KEYS)


def _groups_for(feats) -> Optional[List[int]]:
    """template_groups(feats), computed once per feature dict (the recycles of one item share the dict: one grouping per item)."""
    try:
        key = _group_key(feats)
    except Exception:
        return template_groups(feats)
    if _STATE["group_key"] != key:
        g = template_groups(feats)
        _STATE["group_key"], _STATE["groups"] = key, g
        _STATE["items"] = int(_STATE["items"]) + 1
        if g is not None:
            _STATE["last_groups"] = groups_word(g)
            _STATE["slots_max"] = max(int(_STATE["slots_max"]), len(g))
    return _STATE["groups"]


def groups_word(groups: Sequence[int]) -> str:
    """'0|1,2,3' — the slots of each distinct template, groups separated by |."""
    by: Dict[int, List[int]] = {}
    for t, r in enumerate(groups):
        by.setdefault(r, []).append(t)
    return "|".join(",".join(str(t) for t in ts) for _, ts in sorted(by.items())) or "-"


def _dedupe_forward(self, input_feature_dict, z, pair_mask=None, triangle_attention="torch", triangle_multiplicative="torch", inplace_safe=False, chunk_size=None):
    """TemplateEmbedder.forward (pairformer.py:982-1039) with each DISTINCT template evaluated once; every other statement verbatim."""
    stock = _ORIG["forward"]
    _c("dedupe", "calls")
    if "template_aatype" not in input_feature_dict or self.n_blocks < 1:        # the stock's own early return (pairformer.py:1009-1011): no template stack runs; nothing to dedupe
        _c("dedupe", "stock:untemplated")
        return stock(self, input_feature_dict, z, pair_mask=pair_mask, triangle_attention=triangle_attention, triangle_multiplicative=triangle_multiplicative, inplace_safe=inplace_safe, chunk_size=chunk_size)
    try:
        groups = _groups_for(input_feature_dict)
    except Exception as e:                                                      # a feature form the grouping cannot read: the stock statements, by name
        groups, _STATE["error"] = None, "groups:%s" % type(e).__name__
    if groups is None:
        _c("dedupe", "stock:unkeyed")
        return stock(self, input_feature_dict, z, pair_mask=pair_mask, triangle_attention=triangle_attention, triangle_multiplicative=triangle_multiplicative, inplace_safe=inplace_safe, chunk_size=chunk_size)
    # ---- the stock preamble, verbatim (pairformer.py:1012-1023)
    asym_id = input_feature_dict["asym_id"]
    multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(z.dtype)
    num_residues = z.shape[0]
    num_templates = input_feature_dict["template_aatype"].shape[0]
    query_num_channels = z.shape[-1]
    if pair_mask is None:
        pair_mask = z.new_ones(z.shape[:-1])
    z = self.layernorm_z(z)
    # ---- the stock accumulation over the slots in the stock order; a duplicate slot adds its representative's tensor (pairformer.py:1024-1035)
    last = {}
    for t, r in enumerate(groups):
        last[r] = t
    kept = {}
    u = 0
    for template_id in range(num_templates):
        r = groups[template_id]
        if r == template_id:
            v = self.single_template_forward(template_id=template_id, input_feature_dict=input_feature_dict, z=z, pair_mask=pair_mask, multichain_mask=multichain_mask,
                                             triangle_attention=triangle_attention, triangle_multiplicative=triangle_multiplicative, inplace_safe=inplace_safe, chunk_size=chunk_size)
            _c("dedupe", "evaluated")
        else:
            v = kept[r]
            _c("dedupe", "reused")
        u = u + v
        if last[r] > template_id:
            kept[r] = v
        else:
            kept.pop(r, None)
        del v
    _c("dedupe", "slots", int(num_templates))
    # ---- the stock tail, verbatim (pairformer.py:1036-1039)
    u = u / (1e-7 + num_templates)
    u = self.linear_no_bias_u(self.relu(u))
    assert u.shape == (num_residues, num_residues, query_num_channels)
    return u


def _te_class():
    import importlib
    return getattr(importlib.import_module(TE_MODULE), TE_CLASS)


def set_dedupe(on: bool) -> bool:
    """Install (on) / remove (off) the distinct-template forward on the stock TemplateEmbedder class. -> installed."""
    on = bool(on)
    CFG["template_dedupe"] = on
    TE = _te_class()
    if "forward" not in _ORIG:
        _ORIG["forward"] = TE.forward
    if on and TE.forward is not _dedupe_forward:
        TE.forward = _dedupe_forward
    elif not on and TE.forward is _dedupe_forward:
        TE.forward = _ORIG["forward"]
    _STATE["installed"] = TE.forward is _dedupe_forward
    return bool(_STATE["installed"])


def describe_dedupe() -> dict:
    """The account the package's report reads: installed; calls (TemplateEmbedder.forward calls through the lever); evaluated / reused (template
    evaluations run / replaced by a representative's tensor); slots (template slots seen); items (feature dicts grouped); groups (the last
    item's grouping, '0|1,2,3' at the CLI defaults); stock:<why> (calls the stock statements answered: untemplated = the stock's own
    early return, unkeyed = a feature form without the five template-indexed keys); error (the last grouping error's word, if any)."""
    c = COUNTS.get("dedupe") or {}
    return {"installed": bool(_STATE["installed"]), "class": CLASS["template_dedupe"], "calls": int(c.get("calls", 0)), "evaluated": int(c.get("evaluated", 0)),
            "reused": int(c.get("reused", 0)), "slots": int(c.get("slots", 0)), "items": int(_STATE["items"]), "groups": str(_STATE["last_groups"]),
            "slots_per_call": int(_STATE["slots_max"]), "stock": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("stock:")}, "error": _STATE["error"]}


# =====================================================================================  tmpl_triatt
def triatt_cells(lever: str, c: int, heads: int, dim: int) -> List[Tuple[str, str, Tuple[int, ...]]]:
    """The core's (impl, piece, key) cells the fused block needs at this shape for `lever` (pair_fused's plan: gflash = impl lnl -> ln_linear (C,)
    + gate_transpose (H*D,); gblock = impl fpf -> prologue / epilogue (C, H, D))."""
    if lever == "gflash" and CFG.get("tmpl_pairfused"):                       # tmpl_pairfused: the fpf prologue / epilogue (LayerNorm fused) serve the block instead of lnl's pieces
        return [("fpf", "prologue", (int(c), int(heads), int(dim))), ("fpf", "epilogue", (int(c), int(heads), int(dim)))]
    if lever == "gflash":
        return [("lnl", "ln_linear", (int(c),)), ("lnl", "gate_transpose", (int(heads) * int(dim),))]
    return [("fpf", "prologue", (int(c), int(heads), int(dim))), ("fpf", "epilogue", (int(c), int(heads), int(dim)))]


def _preflight(wanted, device):
    from opt_core.attn import pair_fused as PF
    return PF.preflight(wanted, device)


def triatt_decision(lever: str, c: int, heads: int, dim: int, device=None) -> Tuple[bool, str, str]:
    """(served, word, cells) for the template block at (c, heads, dim) under `lever` on this device — decided once per key from the core's cell
    table (nothing launches): served = every needed cell resolves; word = the first unserved cell's word (CellDecision.word(), e.g.
    off:not-measured:64x4x32) or 'served'; cells = 'impl:piece(key)+…' for the LEVER line."""
    key = (lever, int(c), int(heads), int(dim), str(device))
    if key not in _DECISIONS:
        wanted = triatt_cells(lever, c, heads, dim)
        cells = "+".join("%s:%s(%s)" % (i, p, "x".join(str(x) for x in k)) for i, p, k in wanted)
        try:
            ds = _preflight(wanted, device)
            bad = [d for d in ds if not d.served]
            _DECISIONS[key] = (not bad, "served" if not bad else str(bad[0].word() or bad[0].reason or "no-cell"), cells)
        except Exception as e:                                                  # an older core without preflight, or a table it cannot read: unserved by name, never a guess
            _DECISIONS[key] = (False, "preflight:%s" % type(e).__name__, cells)
    return _DECISIONS[key]


def triatt_gate(lever: str, module, x) -> Optional[str]:
    """levers_ptx1._tri_forward's question for a call whose pair width is not the trunk's 128: None = answer it through the fused block (tmpl_triatt
    on, the cells serve on this card — the caller's own served path and census follow), else the census word under which the stock statement
    answers the call: `stock:c=<c>` (the lever is off) or `stock:tmpl_triatt:<the core's word>` (on, no servable cell)."""
    c = int(x.shape[-1])
    if not CFG["tmpl_triatt"]:
        return "stock:c=%d" % c
    mha = getattr(module, "mha", None)
    heads, dim = getattr(mha, "no_heads", None), getattr(mha, "c_hidden", None)
    if heads is None or dim is None:
        _c("triatt", "stock:tmpl_triatt:shape")
        return "stock:tmpl_triatt:shape"
    served, word, _cells = triatt_decision(lever, c, int(heads), int(dim), x.device)
    if served:
        _c("triatt", "routed:%s:c=%d" % (lever, c))
        return None
    w = "stock:tmpl_triatt:%s" % word
    _c("triatt", w)
    return w


def set_triatt(on: bool) -> None:
    CFG["tmpl_triatt"] = bool(on)


def describe_triatt() -> dict:
    """on; routed (template calls sent through the block, per lever:c); stock (calls answered by the stock statement, per word); decisions (per
    (lever, c, H, D): served, word, cells)."""
    c = COUNTS.get("triatt") or {}
    return {"on": bool(CFG["tmpl_triatt"]), "class": CLASS["tmpl_triatt"], "routed": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("routed:")},
            "routed_total": sum(v for k, v in c.items() if k.startswith("routed:")), "stock": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("stock:")},
            "decisions": {"%s:c%dh%dd%d" % k[:4]: {"served": v[0], "word": v[1], "cells": v[2]} for k, v in _DECISIONS.items()}}


def stock_forward(orig, module, x, mask=None, chunk_size=None, triangle_attention="torch", inplace_safe=False):
    """The stock TriangleAttention statement for a template-stack call levers_ptx1._tri_forward hands back by name."""
    return orig(module, x, mask, chunk_size, triangle_attention, inplace_safe)


# =====================================================================================  tmpl_pairfused (impl choice for tmpl_triatt's calls)
def triatt_impl(lever: str, impl: str, ln: str, x_ln, c: int):
    """levers_ptx1._tri_forward's (impl, ln, x_ln) for a template call it is about to send through the block: with tmpl_pairfused on under the
    trunk's gflash the fpf prologue / epilogue with the LayerNorm fused (the cells triatt_gate already found served for this shape); otherwise
    the trunk lever's own choice, untouched."""
    if int(c) == 128 or not CFG.get("tmpl_pairfused") or lever != "gflash":
        return impl, ln, x_ln
    _c("pairfused", "fpf")
    return "fpf", "fused", None


def set_pairfused(on: bool) -> bool:
    CFG["tmpl_pairfused"] = bool(on)
    return CFG["tmpl_pairfused"]


def describe_pairfused() -> dict:
    c = COUNTS.get("pairfused") or {}
    return {"on": bool(CFG["tmpl_pairfused"]), "class": CLASS["tmpl_pairfused"], "calls": int(c.get("fpf", 0)), "counts": dict(c)}


# =====================================================================================  tmpl_xtr (the template pair Transition, exact construction)
XTR_CELL = ("fpf", "transition", (64, 128))
_XTR: Dict[str, object] = {"decision": None}


XTR_WORDS = {"exact": "exact", "fast": "fast", "big": "big"}        # opt_core.kernels.transition TIER word per kit mode for the template pair transition (c 64, hidden 128,
XTR_WORD = None                                                     # family `rows`, keyed by the item's token count): the provider's row when it serves the call; a Refusal by
XTR_PROVIDER_TIERS = ("fast", "big")                              # name -> the pair_fused exact construction below (xtr's statement at c 64). XTR_WORD: a fixed word (None =
                                                                    # the mode's); XTR_PROVIDER_TIERS: the modes whose tier word asks the provider (a word absent here keeps
                                                                    # the pair_fused construction for that mode by name).


def xtr_word() -> Optional[str]:
    """The transition provider's tier word for this process (XTR_WORD, else the kit mode's word when the mode is in XTR_PROVIDER_TIERS; the bare module: `fast`),
    or None when the mode keeps the pair_fused construction."""
    if XTR_WORD:
        return str(XTR_WORD)
    mode = _kit_mode() or "fast"
    return XTR_WORDS.get(mode) if mode in XTR_PROVIDER_TIERS else None


def _transition_provider():
    from opt_core.kernels import transition as T                            # the shared core's transition provider: one face over every carried transition row
    return T


def _xtr_provider_weights(m, T):
    c = m.__dict__.setdefault("_ptx1_templ", {})
    if "xtr_provider_w" not in c:
        ln = m.layernorm1
        c["xtr_provider_w"] = T.pack(w_o=m.linear_no_bias.weight, w_a=m.linear_no_bias_a.weight, w_b=m.linear_no_bias_b.weight, ln_w=ln.weight,
                                     ln_b=getattr(ln, "bias", None), eps=float(getattr(ln, "eps", 1e-5)), device=m.linear_no_bias.weight.device)
    return c["xtr_provider_w"]


def _xtr_provider_route(module, x):
    """The template pair transition through opt_core.kernels.transition by the mode's tier word -> y, or None when the provider is absent, the mode keeps the
    pair_fused construction, or the provider refuses the call BY NAME (`provider:refused:<kind>`, counted; the caller takes the pair_fused construction)."""
    word = xtr_word()
    if word is None:
        _c("xtr", "provider:not_in_mode"); return None
    try:
        T = _transition_provider()
    except Exception as e:                                                  # an older core without the provider: named once, the pair_fused construction
        _c("xtr", "provider:none:%s" % type(e).__name__); return None
    xs = x if x.stride(-1) == 1 else x.contiguous()
    n_eq = int(xs.shape[-2]) if (xs.dim() >= 3 and xs.shape[-2] == xs.shape[-3]) else int(round(math.sqrt(max(1, xs.numel() // int(xs.shape[-1])))))
    try:
        y, sel = T.transition(xs, _xtr_provider_weights(module, T), word=word, residual=False, n_tokens=n_eq, family="rows", timing="eager")
    except Exception as e:
        R = getattr(T, "Refusal", None)
        if R is not None and isinstance(e, R):                              # the provider's word for a call its rows do not take on this card / stack: by that name
            _c("xtr", "provider:refused:%s" % str(getattr(e, "kind", None) or "refusal").replace(" ", "_")[:64]); return None
        if type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower():
            raise
        w = "provider:error:%s" % type(e).__name__
        _c("xtr", w)
        if COUNTS["xtr"].get(w, 0) <= 2:
            print("[ptx1_templ] tmpl_xtr: transition provider error %r -> the pair_fused construction for this call" % (e,), flush=True)
        return None
    row = str(getattr(sel, "row", "") or word) + ((":" + str(sel.variant)) if getattr(sel, "variant", None) else "")
    if _XTR.get("provider_first") is None:
        _XTR["provider_first"] = {"word": word, "row": row, "cell": str(getattr(sel, "cell_key", "") or ""), "N": n_eq}
    _c("xtr", "routed:provider:%s" % row)
    return y


def _xtr_weights(m, PF):
    c = m.__dict__.setdefault("_ptx1_templ", {})
    if "xtr_w" not in c:
        ln = m.layernorm1
        c["xtr_w"] = PF.pack_transition_weights(ln_w=ln.weight, ln_b=ln.bias, w_a=m.linear_no_bias_a.weight, w_b=m.linear_no_bias_b.weight,
                                               w_out=m.linear_no_bias.weight, eps=float(getattr(ln, "eps", 1e-5)))
    return c["xtr_w"]


def xtr_decision(device=None) -> Tuple[bool, str]:
    """(served, word) for the template transition cell on this device, decided once from the core's cell table (nothing launches)."""
    if _XTR["decision"] is None:
        try:
            ds = _preflight([XTR_CELL], device)
            bad = [d for d in ds if not d.served]
            _XTR["decision"] = (not bad, "served" if not bad else str(bad[0].word() or bad[0].reason or "no-cell"))
        except Exception as e:
            _XTR["decision"] = (False, "preflight:%s" % type(e).__name__)
    return _XTR["decision"]


def _device_cc(dev) -> str:
    """'M.m' for a CUDA device (levers_ptx1.device_cc's cached probe when importable), else the device type ('cpu')."""
    try:
        import levers_ptx1 as L                                             # lazy: levers_ptx1 imports this module at its own import
        return str(L.device_cc(dev))
    except Exception:
        return ("%d.%d" % torch.cuda.get_device_capability(dev)) if getattr(dev, "type", "") == "cuda" else str(getattr(dev, "type", dev))


def xtr_card_word(rows: int, cc: str) -> Optional[str]:
    """The exact transition's served-row rule of this card, as the shared core states it (opt_core.attn.pair_fused "exact_rows", piece
    `transition`: the cuBLAS floor / piece-remainder rule per compute capability — the refusal PF.transition raises by name at plan time, read
    before the call): the census word (`stock:above_max_rows` | `stock:below_min_rows`) when a call of `rows` rows lies outside the served rows on
    compute capability `cc`, None when served, when the card has no entry, or on a core without the rule."""
    try:
        from opt_core.attn import pair_fused as PF
        w = PF.exact_rows_word("transition", cc, int(rows))
    except Exception:
        return None
    return None if w is None else ("stock:below_min_rows" if "below" in str(w) else "stock:above_max_rows")


def transition_route(module, x):
    """levers_ptx1._transition_forward's route for the template Pairformer's Transition (c 64, hidden 128; the caller has checked cuda / bf16 autocast /
    the row gate): the exact construction's output when tmpl_xtr is on and the cell serves; None when the stock statement answers (off: nothing counted;
    unserved shape / refusal: counted by name)."""
    if not CFG["tmpl_xtr"]:
        return None
    served, word = xtr_decision(x.device)
    if not served:
        _c("xtr", "stock:tmpl_xtr:%s" % word); return None
    y = _xtr_provider_route(module, x)                                            # the core's transition PROVIDER by the mode's tier word first (opt_core.kernels.transition):
    if y is not None:                                                             # served -> done; a refusal by name -> the pair_fused construction below
        return y
    card = xtr_card_word(x.numel() // int(x.shape[-1]), _device_cc(x.device))   # the card's served-row rule for the exact transition (cc 8.0): outside it the
    if card is not None:                                                          # stock statement BY NAME, where `xtr` steps aside too — bits == stock by construction
        _c("xtr", "stock:tmpl_xtr:%s" % card.split(":", 1)[-1]); return None
    from opt_core.attn import pair_fused as PF
    T = _xtr_weights(module, PF)
    xs = x if x.stride(-1) == 1 else x.contiguous()
    try:
        y = PF.transition(xs, T, residual=False, ln="stock", x_ln=module.layernorm1(xs), impl="fpf")
    except PF.Unsupported as e:                                             # the word, never the exception: the stock statement for this call
        w = "fallback:tmpl_xtr:%s" % str(getattr(e, "reason", e))[:80]
        _c("xtr", w)
        if COUNTS["xtr"].get(w, 0) <= 2:
            print("[ptx1_templ] tmpl_xtr refused by the core (%s) -> the stock statement for this call" % (getattr(e, "reason", e),), flush=True)
        return None
    _c("xtr", "routed:fpf")
    return y


def set_xtr(on: bool) -> bool:
    CFG["tmpl_xtr"] = bool(on); _XTR["decision"] = None
    return CFG["tmpl_xtr"]


def describe_xtr() -> dict:
    c = COUNTS.get("xtr") or {}
    d = _XTR["decision"]
    return {"on": bool(CFG["tmpl_xtr"]), "class": CLASS["tmpl_xtr"], "cell": "%s:%s(%s)" % (XTR_CELL[0], XTR_CELL[1], "x".join(str(v) for v in XTR_CELL[2])),
            "decision": None if d is None else {"served": d[0], "word": d[1]}, "routed_total": sum(v for k, v in c.items() if k.startswith("routed:")),
            "word": xtr_word() if True else None, "provider_first": _XTR.get("provider_first"), "provider": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("provider:")},
            "stock": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("stock:")}, "fallback": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("fallback:")}}


# =====================================================================================  tmpl_trimul
def trimul_weights(m) -> dict:
    """The ten opt_core.trimul_weights.WEIGHT_KEYS tensors of a protenix TriangleMultiplicativeUpdate (its projections carry no biases)."""
    c = m.__dict__.setdefault("_ptx1_templ", {})
    if "trimul_w" not in c:
        d = lambda t: t.detach()
        c["trimul_w"] = {"ln_in_w": d(m.layer_norm_in.weight), "ln_in_b": d(m.layer_norm_in.bias), "w_ag": d(m.linear_a_g.weight), "w_ap": d(m.linear_a_p.weight),
                         "w_bg": d(m.linear_b_g.weight), "w_bp": d(m.linear_b_p.weight), "ln_out_w": d(m.layer_norm_out.weight), "ln_out_b": d(m.layer_norm_out.bias),
                         "w_o": d(m.linear_z.weight), "w_og": d(m.linear_g.weight)}
    return c["trimul_w"]


def _kit_mode() -> Optional[str]:
    """The kit's active mode word (exact | fast | big) as levers_ptx1.kit_mode reads it from the package, else None (tests, the bare module)."""
    try:
        import levers_ptx1 as L                                             # lazy: levers_ptx1 imports this module at its own import
        return L.kit_mode()
    except Exception:
        return None


def trimul_word() -> str:
    """The provider word of this process's template TriMul calls: TRIMUL_WORD when set; else `exact` when only tmpl_trimul_exact is on or the kit runs
    --mode exact, `big` under --mode big, `fast` otherwise (TRIMUL_WORDS)."""
    if TRIMUL_WORD:
        return str(TRIMUL_WORD)
    mode = _kit_mode()
    if mode == "exact" or (CFG.get("tmpl_trimul_exact") and not CFG.get("tmpl_trimul")):
        return TRIMUL_WORDS["exact"]
    return TRIMUL_WORDS.get(mode or "fast", "fast")


def _trimul_provider(word=None):
    word = word or trimul_word()
    prov = _TM.setdefault("providers", {})
    if prov.get(word) is None:
        from opt_core import trimul as OT                                    # the shared core's by_word face over kernels.trimul
        prov[word] = OT.by_word(trimul_weights, word, name="tmpl_trimul")
    _TM["provider"] = prov[word]
    return prov[word]


def trimul_route(module, z, mask=None, inplace_safe=False, add_with_inplace=False):
    """levers_ptx1._trimul_forward's route for a TriangleMultiplicativeUpdate whose c_z != c_hidden (the template Pairformer's; the stock statement takes
    its torch path there): the provider's output (the update, or z + update in the stock's in-place-add form) when tmpl_trimul is on and the row serves
    the call; None when the caller's stock statement answers it (lever off: nothing counted here, the caller's `stock:path`; a refusal / a small item /
    an import failure: counted `stock:tmpl_trimul:<why>`)."""
    if not (CFG["tmpl_trimul"] or CFG.get("tmpl_trimul_exact")):
        return None
    N = int(z.shape[-2])
    word = trimul_word()
    try:
        prov = _trimul_provider(word)
        from opt_core.trimul import Call, Refused
    except Exception as e:                                                  # an older core without the provider: the stock statement by name (the LEVER line reads the error)
        _TM["error"] = "import:%s" % type(e).__name__
        _c("trimul", "stock:tmpl_trimul:import"); return None
    direction = "outgoing" if getattr(module, "_outgoing", True) else "incoming"
    residual = bool(inplace_safe is True and add_with_inplace)
    m2 = mask
    call = Call(module, z, m2, direction, residual, orig=lambda: None)
    try:
        prov.eligible(call)
        sel = call.extra.get("trimul_selection")
        row = str(getattr(sel, "row", "") or "")
        if sel is not None and (row in STOCK_ROWS or str(getattr(sel, "cls", "") or "").split("(")[0].strip() == "stock"):   # (the core spells classes `stock(<row>)`)   # the word names the stock for this cell (no cell, or the cell's row is the stock op): the stock
            why = "no_cell" if "no_cell" in str(getattr(sel, "reason", "") or "") else "stock_row"   # statement itself answers — idle BY NAME, nothing launched
            _c("trimul", "stock:tmpl_trimul:%s:%s" % (row or "stock", why)); return None
        out = prov.fn(call)
    except Refused as e:                                                    # the table's word for a call this row does not take (`<row>:<kind>`): the stock statement, by that name
        _c("trimul", "stock:tmpl_trimul:%s" % str(e).replace(" ", "_")[:96]); return None
    except Exception as e:
        if type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower():
            raise
        w = "error:%s" % type(e).__name__
        _c("trimul", "stock:tmpl_trimul:%s" % w)
        if COUNTS["trimul"].get("stock:tmpl_trimul:%s" % w, 0) <= 2:
            print("[ptx1_templ] tmpl_trimul kernel error -> the stock statement for this call: %r" % (e,), flush=True)
        return None
    row = str(getattr(prov, "kernel", "") or word).split(":")[-1]
    if _TM["first"] is None:
        _TM["first"] = {"row": row, "word": word, "version": getattr(prov, "version", None), "N": N, "C": int(z.shape[-1]), "dtype": str(z.dtype).replace("torch.", ""), "direction": direction, "residual": residual}
    _c("trimul", "routed:%s" % row)
    return out


def set_trimul(on: bool) -> bool:
    CFG["tmpl_trimul"] = bool(on)
    return CFG["tmpl_trimul"]


def set_trimul_exact(on: bool) -> bool:
    CFG["tmpl_trimul_exact"] = bool(on)
    return CFG["tmpl_trimul_exact"]


def describe_trimul(name: str = "tmpl_trimul") -> dict:
    """One account for both words (they share the route and the census): `on` is the asked word's switch, `word` the provider word of this process,
    `class` = exact for tmpl_trimul_exact (the provider's exact tier word names bitwise-vouched rows only) and tolerance for tmpl_trimul."""
    c = COUNTS.get("trimul") or {}
    w = None
    try:
        w = trimul_word()
    except Exception:
        pass
    return {"on": bool(CFG.get(name)), "class": CLASS[name], "word": w, "error": _TM["error"], "first": _TM["first"],
            "routed": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("routed:")}, "routed_total": sum(v for k, v in c.items() if k.startswith("routed:")),
            "stock": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("stock:")}}


# =====================================================================================  the arm's entry points (levers_ptx1.apply calls these)
def apply(cfg: dict) -> dict:
    """Set the levers from an arm's CFG (keys LEVER_NAMES, absent = off). -> {'template_dedupe': installed, 'tmpl_triatt': on, 'tmpl_trimul': on, 'tmpl_xtr': on, 'tmpl_pairfused': on}."""
    out = {"template_dedupe": set_dedupe(bool(cfg.get("template_dedupe"))), "tmpl_triatt": bool(cfg.get("tmpl_triatt")),
           "tmpl_trimul": set_trimul(bool(cfg.get("tmpl_trimul"))), "tmpl_trimul_exact": set_trimul_exact(bool(cfg.get("tmpl_trimul_exact"))), "tmpl_xtr": set_xtr(bool(cfg.get("tmpl_xtr"))), "tmpl_pairfused": set_pairfused(bool(cfg.get("tmpl_pairfused")))}
    set_triatt(out["tmpl_triatt"])
    return out


def describe() -> dict:
    return {"cfg": dict(CFG), "template_dedupe": describe_dedupe(), "tmpl_triatt": describe_triatt(), "tmpl_trimul": describe_trimul(),
            "tmpl_trimul_exact": describe_trimul("tmpl_trimul_exact"),
            "tmpl_xtr": describe_xtr(), "tmpl_pairfused": describe_pairfused()}


def lever_facts(name: str) -> dict:
    """The LEVER-line facts of one word (the package's report.lever_line kwargs): template_dedupe -> served (evaluations run through the lever) /
    reused / slots / distinct (the last item's grouping) / class; tmpl_triatt -> routed / class / cells + the stock words counted; tmpl_trimul -> routed calls / class / row word + the stock words."""
    if name == "template_dedupe":
        d = describe_dedupe()
        return {"served": d["evaluated"], "reused": d["reused"], "slots": d["slots"], "groups": d["groups"], "cls": d["class"], **({"stock_" + k: v for k, v in d["stock"].items()})}
    if name == "tmpl_xtr":
        d = describe_xtr()
        return {"served": d["routed_total"], "cls": d["class"], "cell": d["cell"], **{"stock_" + k.replace(":", "_"): v for k, v in d["stock"].items()}}
    if name == "tmpl_pairfused":
        d = describe_pairfused()
        return {"served": d["calls"], "cls": d["class"]}
    if name in ("tmpl_trimul", "tmpl_trimul_exact"):
        d = describe_trimul(name)
        return {"served": d["routed_total"], "cls": d["class"], "word": d["word"], **{"stock_" + k.replace(":", "_"): v for k, v in d["stock"].items()}}
    d = describe_triatt()
    facts = {"served": d["routed_total"], "cls": d["class"]}
    for k, v in d["decisions"].items():
        facts["cells"] = v["cells"] + ("" if v["served"] else "!" + v["word"])
    facts.update({"stock_" + k.replace(":", "_"): v for k, v in d["stock"].items()})
    return facts
