"""ef2_transition_cute — lever ``t16``: every d=256 / hidden=1024 pair Transition of the fast / big sets (folding trunk, lm_encoder, parcae coda,
confidence trunk: ``x + T(x)``) and the MSA encoder blocks' PairTransition (``T(x)``, the block adds the residual) served through the shared
core's transition provider BY TIER WORD (opt_core.kernels.transition, ``TIER_WORDS``): the mode's own word — ``fast`` in fast, ``big`` in
big — and the provider's measured cell for this card, stack and size decides the row (on compute capability 9.0 the ESM-family CuTe kernel
esm_t16, on 8.0 the Triton kernel esm_t15, the fused statement kernel where the cell names it, ...).  This module holds NO kernel: the class
patches, the eligibility test, the weight packs and the engagement record only.  A call the provider refuses BY NAME (a size below the cell
table, a shape outside the row's envelope, a stack without the row) keeps the previous statement (the fused backend's own forward) — counted in
STATS and named once on stderr, never silent, never an error on a completed fold.

    install(model)        patches C.Transition.forward and MOD.PairTransition.forward process-wide, resolves the word, pre-packs the weights of
                          every servable module (nothing is packed inside a graph capture); idempotent.  ef2_server.configure calls it in the pair slot.
    uninstall(model)      restores the previous forwards.
    describe() / levers_on() / kernel_evidence() / STATS    engagement record for the package's LEVER / EXIT lines.
    EF2_T16_WORD=<word>   an A/B override: any tier word or row word of the provider (default: the mode's tier word).
    EF2_T16_MEMO=0|1      the per-class resolution memo: live under the `big` word ONLY (the graph-free
                          memory mode is the line that pays the glue's host work per call; `fast` / `exact` / a row word resolve every call
                          through select + transition and print no memo token on their LEVER line); "1" = live under any word (A/B), "0" = off everywhere.  When live: in EAGER use the
                          FIRST call of a class (word, family, form, residual, n_tokens, rows, device) resolves through the provider (select +
                          transition: the core's census records the class once, its rows are primed) and the resolved Selection is kept; the
                          class's later calls dispatch that same row directly (the provider's serve-a-Selection entry) -- the same row, the same
                          launch, the same bytes, ~160 us less host work per call (610 calls per 400-token Fast-line fold).  Calls made while a
                          CUDA graph is capturing always take the full path (the provider's capture-time substitutions stand).
Inference only: grad enabled / CPU / non-bf16 / d != 256 / hidden != 1024 / biased linears fall through to the previous forward (counted in STATS).
"""
import collections
import math
import os
import sys

import torch

VERSION = "transition_cute.2.2"
WORD_ENV = "EF2_T16_WORD"             # A/B override of the bound word (a tier word or a provider row word); unset = the mode's tier word
MEMO_ENV = "EF2_T16_MEMO"             # "0" = no per-class resolution memo anywhere; "1" = the memo under any word (A/B); unset = the memo under MEMO_WORDS only
MEMO_WORDS = ("big",)               # the bound words whose eager calls take the per-class memo by default: the memory mode's tier word (fast / exact / row words
                                      # resolve every call through select + transition with no memo, and print no memo token on their LEVER line)
MODE_ENV = "ESMFOLD2_OPT"             # the package exports the mode before configure; fast -> `fast`, big -> `big` (exact never carries this lever)
TIER_OF_MODE = {"fast": "fast", "big": "big", "exact": "exact"}
DEFAULT_WORD = "fast"
FAMILY, FORM = "esmpair", "esmfused"   # the provider's cell family of this engine's pair Transition (residual folded) under the fork's fused inference statement
PAIR_FAMILY = "pair"                  # the MSA blocks' PairTransition: T(x) without the residual -> the provider's plain pair family
STATS = collections.Counter()
_BF16 = torch.bfloat16
_STATE = dict(installed=False, orig_transition_forward=None, orig_pair_transition_forward=None, trunk=False, msa=False, word=None, core=None, face=None,
              rows=collections.Counter(), refused=collections.Counter(), printed=set(), memo={}, memo_on=None)


def _C():
    import transformers.models.esmfold2.modeling_esmfold2_common as C
    return C


def _MOD():
    import transformers.models.esmfold2.modeling_esmfold2 as MOD
    return MOD


def bound_word():
    """The word this process binds: EF2_T16_WORD if set, else the mode's tier word (ESMFOLD2_OPT), else `fast`."""
    if _STATE["word"] is None:
        w = (os.environ.get(WORD_ENV) or "").strip() or TIER_OF_MODE.get((os.environ.get(MODE_ENV) or "").strip().lower(), DEFAULT_WORD)
        _STATE["word"] = w
    return _STATE["word"]


def _face():
    """opt_core.kernels.transition (imported once); ImportError propagates BY NAME (the package's core gate runs before any lever: a core without the provider never reaches here)."""
    if _STATE["face"] is None:
        import opt_core
        from opt_core.kernels import transition as TR
        _STATE["core"] = getattr(opt_core, "__version__", "?")
        _STATE["face"] = TR
    return _STATE["face"]


def _pkey(*params):
    return tuple((p.data_ptr(), p._version, p.dtype, tuple(p.shape)) for p in params if p is not None)


def _face_pack_for(mod):
    """opt_core.kernels.transition Weights for a servable module (w12 fused: rows [0,H) = the silu branch), cached on the module by parameter identity + version."""
    cache = mod.__dict__.setdefault("_transition_cute_face_pack", {})
    key = _pkey(mod.norm.weight, mod.norm.bias, mod.ffn.w12.weight, mod.ffn.w3.weight)
    if cache.get("key") != key:
        TR = _face()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            W = TR.pack(w_ab=mod.ffn.w12.weight.detach(), w_o=mod.ffn.w3.weight.detach(), ln_w=mod.norm.weight.detach(),
                        ln_b=(mod.norm.bias.detach() if mod.norm.bias is not None else torch.zeros_like(mod.norm.weight)), eps=mod.norm.eps, device=mod.ffn.w12.weight.device)
        cache.clear(); cache["key"] = key; cache["pack"] = W; STATS["face_packs"] += 1
    return cache["pack"]


def _servable(mod):
    """module shape the provider's ESM-family rows serve: LayerNorm(256, affine) + SwiGLUMLP(256 -> 2x1024 -> 256, no bias)."""
    ffn, norm = getattr(mod, "ffn", None), getattr(mod, "norm", None)
    return (ffn is not None and isinstance(norm, torch.nn.LayerNorm) and tuple(norm.normalized_shape) == (256,) and norm.weight is not None
            and getattr(ffn, "hidden_features", None) == 1024 and tuple(ffn.w12.weight.shape) == (2048, 256) and tuple(ffn.w3.weight.shape) == (256, 1024)
            and ffn.w12.bias is None and ffn.w3.bias is None)


def _eligible(mod, x):
    return (_STATE["installed"] and x.is_cuda and x.dtype == _BF16 and x.shape[-1] == 256 and x.numel() > 0 and not torch.is_grad_enabled()
            and mod.ffn.w12.weight.is_cuda and _servable(mod))


def _n_tokens(x, rows):
    """The token count the provider's cell table is keyed by: a pair tensor [..., N, N, 256] names N itself; any other layout the square root of its rows."""
    if x.dim() >= 3 and x.shape[-2] == x.shape[-3]:
        return int(x.shape[-2])
    return max(1, int(math.isqrt(max(1, rows))))


def memo_on():
    """The per-class resolution memo is live for this process's bound word: EF2_T16_MEMO "0" -> off, "1" -> on under any word (A/B), unset -> on
    under the `big` word only (MEMO_WORDS).  Decided once per (env, bound word); ``uninstall`` resets it."""
    word = bound_word()
    hit = _STATE["memo_on"]
    if hit is None or hit[0] != word:
        v = (os.environ.get(MEMO_ENV) or "").strip()
        live = False if v == "0" else (True if v == "1" else word in MEMO_WORDS)
        hit = _STATE["memo_on"] = (word, live)
    return hit[1]


def _serve_memoised(TR, x2d, W, ent, residual, form):
    """Dispatch the class's memoised Selection through the provider's serve entry for a resolved Selection (``_dispatch``: what ``transition``
    runs after its own ``select``); a core without that entry serves through ``transition(<row word>)`` (one select more, the same row)."""
    sel, roww = ent
    serve = getattr(TR, "_dispatch", None)
    if serve is not None:
        y, _s = serve(x2d, W, sel, roww, residual, None, None, None, None, form)
        return y
    y, _s = TR.transition(x2d, W, word=roww, residual=residual, n_tokens=getattr(sel, "n_tokens", None), family=(FAMILY if residual else PAIR_FAMILY), form=form,
                          timing="eager", capture=False)
    return y


def _name_once(key, text):
    if key not in _STATE["printed"]:
        _STATE["printed"].add(key)
        print(f"[esmfold2-opt] NOTE t16: {text}", file=sys.stderr, flush=True)


def _run(mod, x, residual, kind):
    """Serve one call through the provider by the bound word -> tensor, or None when the provider refuses BY NAME (the caller keeps the previous statement)."""
    x2d = x.contiguous().view(-1, 256)
    rows = x2d.shape[0]
    TR = _face()
    capturing = bool(x.is_cuda and torch.cuda.is_current_stream_capturing())
    fam, form = (FAMILY, FORM) if residual else (PAIR_FAMILY, "swiglu")
    timing = "graph" if capturing else "eager"
    n_tok = _n_tokens(x, rows)
    mkey = (bound_word(), fam, form, bool(residual), int(n_tok), int(rows), str(x.device)) if (memo_on() and not capturing) else None   # EAGER calls of a class
    ent = _STATE["memo"].get(mkey) if mkey is not None else None                                                                       # already resolved once dispatch the kept row
    if ent is not None:
        if ent[0] == "stock":                                            # the class resolves to the statement itself (decided once through select): the module's own call, by name
            STATS["face_refused"] += 1; STATS["memo_hits"] += 1; _STATE["refused"][f"{kind}:cell_names_{ent[1]}"] += 1
            return None
        try:
            with torch.amp.autocast("cuda", enabled=False):
                y = _serve_memoised(TR, x2d, _face_pack_for(mod), ent[1:], residual, form)
        except TR.Refusal:                                               # the kept row refuses this call with the tensors in hand: the class resolves afresh through the full path below
            _STATE["memo"].pop(mkey, None); STATS["memo_dropped"] += 1
        else:
            STATS["face_calls"] += 1; STATS["memo_hits"] += 1
            _STATE["rows"][f"{kind}:{ent[2]}"] += 1
            return y.view(x.shape)
    try:
        sel = TR.select(bound_word(), c=256, hidden=1024, n_tokens=n_tok, dtype="bf16", direction="fwd", timing=timing, family=fam, residual=residual,
                        device=x.device, capture=capturing, rows_count=rows, form=form)
        if sel.row in TR.STOCK_ROWS:                                     # the cell names the statement itself for this class / size: the module's own call, by name
            STATS["face_refused"] += 1; _STATE["refused"][f"{kind}:cell_names_{sel.row}"] += 1
            _name_once(("stock", kind, sel.row), f"word={bound_word()} resolves to the statement ({sel.row}) for {kind} at this size / class -> the module's own statement serves (by name)")
            if mkey is not None:
                _STATE["memo"][mkey] = ("stock", sel.row); STATS["memo_classes"] += 1
            return None
        roww = sel.row + ((":" + sel.variant) if getattr(sel, "variant", None) else "")
        with torch.amp.autocast("cuda", enabled=False):
            y, _sel = TR.transition(x2d, _face_pack_for(mod), word=roww, residual=residual, n_tokens=n_tok, family=fam, form=form, timing=timing, capture=capturing)
    except TR.Refusal as r:                                              # named: the previous statement serves this call (counted; printed once per reason)
        kindw = str(getattr(r, "kind", "refusal"))[:72]
        STATS["face_refused"] += 1; _STATE["refused"][f"{kind}:{kindw}"] += 1
        _name_once(("refused", kind, kindw), f"the transition provider refuses word={bound_word()} for {kind} (rows={rows}): {kindw} -> the module's own statement serves these calls (by name)")
        return None
    if mkey is not None and _sel is not None and getattr(_sel, "row", None) == sel.row:   # the class is resolved: keep the Selection the provider SERVED (same row as the word's answer;
        _STATE["memo"][mkey] = ("serve", _sel, roww); STATS["memo_classes"] += 1          # a substituted / stepped-aside serve is not kept -- such a class keeps resolving through the full path)
    STATS["face_calls"] += 1
    _STATE["rows"][f"{kind}:{roww}"] += 1
    return y.view(x.shape)


def _transition_forward_cute(self, x):
    """C.Transition.forward = x + ffn(norm(x)): the provider's row when eligible (bf16 CUDA pair, fused backend active, no grad), else the previous forward."""
    if not (_eligible(self, x) and self._can_use_fused_path(x)):
        STATS["fallthrough_transition"] += 1
        return _STATE["orig_transition_forward"](self, x)
    y = _run(self, x, residual=True, kind="transition")
    if y is None:
        STATS["fallthrough_transition"] += 1
        return _STATE["orig_transition_forward"](self, x)
    STATS["transition_calls"] += 1
    return y


def _pair_transition_forward_cute(self, x):
    """MOD.PairTransition.forward = ffn(norm(x)) (the MSA encoder blocks' pair transition; the block adds the residual): the provider's row when eligible."""
    if not _eligible(self, x):
        STATS["fallthrough_pair_transition"] += 1
        return _STATE["orig_pair_transition_forward"](self, x)
    y = _run(self, x, residual=False, kind="pair_transition")
    if y is None:
        STATS["fallthrough_pair_transition"] += 1
        return _STATE["orig_pair_transition_forward"](self, x)
    STATS["pair_transition_calls"] += 1
    return y


def install(model=None, trunk=True, msa=True, prepack=True):
    """Engage the lever: import the provider (by name if absent), patch C.Transition.forward (trunk transitions of every FoldingTrunk) and
    MOD.PairTransition.forward (MSA encoder blocks) process-wide, resolve the word, and pre-pack the weights of every servable module of `model`
    (so nothing is packed inside a graph capture).  Idempotent; returns a record for the LEVER line."""
    import ef2_srcguard                                                # the class forwards are re-issued (x + ffn(norm(x)) with the fused-path test; T(x) for PairTransition):
    ef2_srcguard.check("t16")                                          # refuse by name on another upstream source
    TR = _face()
    word = bound_word()
    if word not in tuple(TR.TIER_WORDS) + tuple(TR.ROW_NAMES) and word.split(":")[0].split("@")[0] not in TR.ROW_NAMES:
        raise ValueError(f"ef2_transition_cute: {WORD_ENV}={word!r} is neither a tier word {TR.TIER_WORDS} nor a row word of opt_core.kernels.transition")
    C, MOD = _C(), _MOD()
    if trunk and _STATE["orig_transition_forward"] is None:
        _STATE["orig_transition_forward"] = C.Transition.forward
        C.Transition.forward = _transition_forward_cute
    if msa and _STATE["orig_pair_transition_forward"] is None:
        _STATE["orig_pair_transition_forward"] = MOD.PairTransition.forward
        MOD.PairTransition.forward = _pair_transition_forward_cute
    _STATE["installed"] = True
    n_pack = dict(transition=0, pair_transition=0, skipped=0)
    if prepack and model is not None:
        with torch.no_grad():
            for m in model.modules():
                kind = "transition" if (trunk and isinstance(m, C.Transition)) else ("pair_transition" if (msa and isinstance(m, MOD.PairTransition)) else None)
                if kind is None:
                    continue
                if _servable(m) and m.ffn.w12.weight.is_cuda:
                    _face_pack_for(m); n_pack[kind] += 1
                else:
                    n_pack["skipped"] += 1
    _STATE["trunk"], _STATE["msa"] = bool(trunk) or _STATE.get("trunk", False), bool(msa) or _STATE.get("msa", False)
    STATS["packed_transition"], STATS["packed_pair_transition"], STATS["packed_skipped"] = n_pack["transition"], n_pack["pair_transition"], n_pack["skipped"]
    return dict(version=VERSION, transition=bool(trunk), pair_transition=bool(msa), packed=n_pack, word=word, core=_STATE["core"], provider="opt_core.kernels.transition")


def uninstall(model=None):
    C, MOD = _C(), _MOD()
    if _STATE["orig_transition_forward"] is not None:
        C.Transition.forward = _STATE["orig_transition_forward"]; _STATE["orig_transition_forward"] = None
    if _STATE["orig_pair_transition_forward"] is not None:
        MOD.PairTransition.forward = _STATE["orig_pair_transition_forward"]; _STATE["orig_pair_transition_forward"] = None
    if model is not None:
        for m in model.modules():
            m.__dict__.pop("_transition_cute_face_pack", None)
    _STATE["installed"] = False
    _STATE["word"] = None; _STATE["rows"].clear(); _STATE["refused"].clear(); _STATE["printed"].clear(); _STATE["memo"].clear(); _STATE["memo_on"] = None
    _STATE["trunk"] = _STATE["msa"] = False
    return dict(version=VERSION, installed=False)


def levers_on():
    """{"t16": bool} — the class patches are live (the package's probe for the LEVER line)."""
    C, MOD = _C(), _MOD()
    live = (_STATE["installed"]
            and (not _STATE.get("trunk") or C.Transition.forward is _transition_forward_cute)
            and (not _STATE.get("msa") or MOD.PairTransition.forward is _pair_transition_forward_cute))
    return {"t16": bool(live)}


def kernel_evidence():
    """Blank-free k=v words for the t16 LEVER line: word=<bound word> provider= core= family= (+ memo=per_class under the big word; + rows=<kind:row=n,...>
    refused=<kind:reason=n,...> once calls ran)."""
    ev = {"word": str(bound_word()), "provider": "opt_core.kernels.transition", "core": str(_STATE.get("core") or "none"), "family": f"{FAMILY}+{FORM}"}
    if memo_on():                                                        # `memo=per_class` under the big word (or EF2_T16_MEMO=1); fast / exact / row words: no token (the 2.0 census)
        ev["memo"] = "per_class"
    if _STATE["rows"]:
        ev["rows"] = ",".join(f"{k}={v}" for k, v in sorted(_STATE["rows"].items()))
    if _STATE["refused"]:
        ev["refused"] = ",".join(f"{k}={v}" for k, v in sorted(_STATE["refused"].items()))
    return ev


def describe():
    d = dict(version=VERSION, installed=_STATE["installed"], levers=levers_on(), stats=dict(STATS), word=bound_word(), core=_STATE.get("core"),
             rows=dict(_STATE["rows"]), refused=dict(_STATE["refused"]))
    if memo_on():                                                        # the memo's record only where the memo is live (big); the 2.0 record otherwise
        d.update(memo="per_class", memo_classes=len(_STATE["memo"]))
    return d


def stats():
    return dict(STATS)
