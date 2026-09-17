"""ef2_pair_v2 — the pair Transition levers of classes other than 9.0 and the MSA module's PairTransition, ALL served through the shared core's
transition provider BY TIER WORD (opt_core.kernels.transition ``TIER_WORDS``: the mode's own word — ``exact`` in exact, ``fast`` in fast,
``big`` in big); the provider's measured cell for the card, stack and size decides the row.  This module holds NO kernel: class patches,
instance wrappers, weight packs and the engagement record only.  Where the tier word resolves to the stock statement (a class / size without a
served row: e.g. the exact word on compute capability 8.0, or below the exact vouch floor) or the provider refuses BY NAME, the module's own
statement runs — counted in STATS and named once on stderr, never silent.

    t15      C.Transition.forward (every FoldingTrunk's d=256 / hidden=1024 transitions: x + T(x), the residual folded by the row) [fast / big sets
             of the classes t16 does not serve]
    t15msa   the MSA encoder blocks' `pair = pair + PairTransition(pair)` through the same call (residual folded) [fast / big, Full model]
    xtr      MOD.PairTransition.forward (pair_transition c=256 / msa_transition c=128) left on the module's own LayerNorm statement, the SwiGLU
             projection served by the tier word with the LayerNorm output GIVEN (the exact-class construction: bitwise where the provider's cell
             vouches the row on this stack; the module's statement by name elsewhere) [exact set, Full model]
    EF2_PAIR_WORD=<word>   an A/B override of the bound word (any tier word or row word of the provider); unset = the mode's tier word.

install(model, trunk=, msa=, xtr=) after ef2_opt.install (M1 rebinds the MSA-block forwards; graphs are captured at the first fold, after this).
"""
import collections
import math
import os
import sys
import types

import torch

import transformers.models.esmfold2.modeling_esmfold2_common as C
import transformers.models.esmfold2.modeling_esmfold2 as MOD

VERSION = "pair_v2.2.0"
STATS = collections.Counter()
_BF16 = torch.bfloat16
WORD_ENV = "EF2_PAIR_WORD"
MODE_ENV = "ESMFOLD2_OPT"
TIER_OF_MODE = {"fast": "fast", "big": "big", "exact": "exact"}
DEFAULT_WORD = "fast"
_STATE = dict(installed=False, orig_transition_forward=None, orig_pair_transition_forward=None, msa_blocks=[], t15=False, t15msa=False, xtr=False, xtr_modules=0,
              word=None, core=None, face=None, rows=collections.Counter(), refused=collections.Counter(), printed=set())


def bound_word():
    """The word this process binds: EF2_PAIR_WORD if set, else the mode's tier word (ESMFOLD2_OPT), else `fast`."""
    if _STATE["word"] is None:
        _STATE["word"] = (os.environ.get(WORD_ENV) or "").strip() or TIER_OF_MODE.get((os.environ.get(MODE_ENV) or "").strip().lower(), DEFAULT_WORD)
    return _STATE["word"]


def _face():
    if _STATE["face"] is None:
        import opt_core
        from opt_core.kernels import transition as TR
        _STATE["core"] = getattr(opt_core, "__version__", "?")
        _STATE["face"] = TR
    return _STATE["face"]


def _pkey(*params):
    return tuple((p.data_ptr(), p._version, p.dtype, tuple(p.shape)) for p in params if p is not None)


def _pack(mod, cache_name="_pair_v2_face_pack"):
    """opt_core.kernels.transition Weights for a Transition / PairTransition module (w12 rows [0,H) = the silu branch a, [H,2H) = b), cached on the module."""
    TR = _face()
    cache = mod.__dict__.setdefault(cache_name, {})
    key = _pkey(mod.norm.weight, mod.norm.bias, mod.ffn.w12.weight, mod.ffn.w3.weight)
    if cache.get("key") != key:
        H = mod.ffn.hidden_features
        w12 = mod.ffn.w12.weight.detach()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            W = TR.pack(w_a=w12[:H], w_b=w12[H:], w_o=mod.ffn.w3.weight.detach(), ln_w=mod.norm.weight.detach(),
                        ln_b=(mod.norm.bias.detach() if mod.norm.bias is not None else torch.zeros_like(mod.norm.weight)), eps=mod.norm.eps, device=w12.device)
        cache.clear(); cache["key"] = key; cache["pack"] = W; STATS["packs"] += 1
    return cache["pack"]


def _n_tokens(x, rows):
    """The token count the provider's cell table is keyed by: a pair tensor [..., N, N, c] names N itself; any other layout the square root of its rows."""
    if x.dim() >= 3 and x.shape[-2] == x.shape[-3]:
        return int(x.shape[-2])
    return max(1, int(math.isqrt(max(1, rows))))


def _name_once(key, text):
    if key not in _STATE["printed"]:
        _STATE["printed"].add(key)
        print(f"[esmfold2-opt] NOTE pair: {text}", file=sys.stderr, flush=True)


def serve(mod, x, *, residual, kind, x_ln=None):
    """ONE provider call for module `mod` on x [..., c]: the tier word is resolved for this call class (select), a resolution to a stock row or a
    Refusal keeps the module's own statement BY NAME (returns None; counted, printed once); else the resolved row serves (returns the tensor)."""
    c = int(x.shape[-1])
    x2d = x.contiguous().view(-1, c)
    rows = x2d.shape[0]
    TR = _face()
    capturing = bool(torch.cuda.is_current_stream_capturing())
    fam, form = ("esmpair", "esmfused") if residual else ("pair", "swiglu")
    kw = dict(c=c, hidden=int(mod.ffn.hidden_features), n_tokens=_n_tokens(x, rows), dtype="bf16", direction="fwd", timing=("graph" if capturing else "eager"),
              family=fam, residual=residual, device=x.device, capture=capturing, rows_count=rows, ln_given=(x_ln is not None), form=form)
    try:
        sel = TR.select(bound_word(), **kw)
        if sel.row in TR.STOCK_ROWS:                                       # the cell names the statement itself for this class / size: the module's own call, by name
            STATS[f"{kind}_stock_by_name"] += 1; _STATE["refused"][f"{kind}:cell_names_{sel.row}"] += 1
            _name_once(("stock", kind, sel.row), f"word={bound_word()} resolves to the statement ({sel.row}) for {kind} at this size / class -> the module's own statement serves (by name)")
            return None
        arm = sel.row + ((":" + sel.variant) if sel.variant else "")
        with torch.amp.autocast("cuda", enabled=False):
            y, _ = TR.transition(x2d if x_ln is None else x2d, _pack(mod), word=arm, residual=residual, x_ln=(None if x_ln is None else x_ln.contiguous().view(-1, c)),
                                 n_tokens=kw["n_tokens"], family=fam, timing=kw["timing"], capture=capturing, form=form)
    except TR.Refusal as r:
        kindw = str(getattr(r, "kind", "refusal"))[:72]
        STATS[f"{kind}_refused"] += 1; _STATE["refused"][f"{kind}:{kindw}"] += 1
        _name_once(("refused", kind, kindw), f"the transition provider refuses word={bound_word()} for {kind} (rows={rows}, c={c}): {kindw} -> the module's own statement serves these calls (by name)")
        return None
    _STATE["rows"][f"{kind}:{arm}"] += 1
    return y.view(x.shape)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# patched forwards
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _eligible(x):
    return (_STATE["installed"] and x.is_cuda and x.dtype == _BF16 and x.shape[-1] == 256 and x.numel() > 0 and not torch.is_grad_enabled())


def _servable(mod):
    ffn, norm = getattr(mod, "ffn", None), getattr(mod, "norm", None)
    return (ffn is not None and isinstance(norm, torch.nn.LayerNorm) and tuple(norm.normalized_shape) == (256,) and norm.weight is not None
            and getattr(ffn, "hidden_features", None) == 1024 and getattr(ffn.w12, "bias", None) is None and getattr(ffn.w3, "bias", None) is None)


def _transition_forward_v2(self, x):
    """C.Transition.forward (lever t15): x + ffn(norm(x)) through the provider's row when eligible (bf16 CUDA pair, fused backend), else the previous forward."""
    if not (_eligible(x) and _servable(self) and self._can_use_fused_path(x)):
        STATS["t15_fallthrough"] += 1
        return _STATE["orig_transition_forward"](self, x)
    y = serve(self, x, residual=True, kind="t15")
    if y is None:
        return _STATE["orig_transition_forward"](self, x)
    STATS["t15_calls"] += 1
    return y


def _pair_transition_residual_v2(pt, pair):
    """pair + PairTransition(pair) for MOD.PairTransition(256) (MSA encoder block) through the provider's row (residual folded); else the stock statements."""
    if not (_eligible(pair) and isinstance(pt, MOD.PairTransition) and _servable(pt)):
        return pair + pt(pair)
    y = serve(pt, pair, residual=True, kind="t15msa")
    if y is None:
        return pair + pt(pair)
    STATS["t15_msa_pair_transition_calls"] += 1
    return y


def _msa_block_forward_v2(self, m, pair, msa_attention_mask, pair_attention_mask):
    """MSAEncoderBlock.forward (lever t15msa): the forward bound before install() (ef2_opt's M1 forward, or the class forward) runs with
    pair_transition swapped for a zero-returning shim, so its final `pair = pair + self.pair_transition(pair)` is a no-op add; the fused
    transition + residual is then applied to its result.  Every other statement of the block (OPM, PWA, msa_transition, TriMul out/in) is
    executed by that previous forward unchanged."""
    prev = self._pair_v2_prev_forward
    pt = self.pair_transition
    self.pair_transition = _ZERO_SHIM
    try:
        m, pair = prev(m, pair, msa_attention_mask, pair_attention_mask)
    finally:
        self.pair_transition = pt
    pair = _pair_transition_residual_v2(pt, pair)
    return m, pair


def _pair_transition_forward_xtr(self, x):
    """MOD.PairTransition.forward = ffn(norm(x)) (lever xtr): the module's own LayerNorm statement, then the provider's row for the tier word with
    the LayerNorm output GIVEN (residual=False: the MSA block adds it, unchanged); the previous forward when not eligible (grad / CPU / non-bf16 /
    chunked / biased linears) or when the word resolves to the statement / is refused for this class and size — counted by name."""
    prev = _STATE["orig_pair_transition_forward"]
    c = int(x.shape[-1])
    if not (_STATE["xtr"] and x.is_cuda and x.dtype == _BF16 and x.numel() > 0 and not torch.is_grad_enabled() and c in (256, 128)
            and getattr(self.ffn.w12, "bias", None) is None and getattr(self.ffn.w3, "bias", None) is None) \
            or (self._chunk_size is not None and x.shape[1] > self._chunk_size):
        STATS["xtr_fallthrough"] += 1
        return prev(self, x)
    n = self.norm(x)                                                  # the stock statement (fp32 LayerNorm under the caller's autocast), unchanged
    if n.dtype != _BF16:
        n = n.to(_BF16)                                               # == the cast autocast applies at ffn.w12's input
    y = serve(self, x, residual=False, kind="xtr", x_ln=n)
    if y is None:
        STATS["xtr_fallthrough"] += 1
        return prev(self, x)
    STATS["xtr_calls"] += 1; STATS["xtr_c%d_calls" % c] += 1
    return y


class _ZeroShim(torch.nn.Module):
    """pair_transition stand-in: returns an additive zero so `pair + self.pair_transition(pair)` leaves pair unchanged (one cheap broadcast add)."""
    def forward(self, x):
        STATS["t15_msa_shim_calls"] += 1
        return torch.zeros((), dtype=x.dtype, device=x.device)


_ZERO_SHIM = _ZeroShim()


def install(model=None, trunk=True, msa=True, xtr=False, **legacy):
    """Levers (independent): trunk (t15) -> C.Transition class patch; msa (t15msa) -> model.msa_encoder's blocks' PairTransition + residual (instance
    wrappers; a model without msa_encoder has nothing to wrap: msa_blocks=0, the lever is not for that variant); xtr -> MOD.PairTransition class
    patch (LayerNorm given).  Weight packs are built here (eager, before any graph capture).  Idempotent; returns the LEVER-line record.
    ``legacy``: keyword names accepted for call compatibility and ignored (t6s= / t6i=)."""
    import ef2_srcguard                                          # the class forwards / the MSA shim re-issue C.Transition.forward / PairTransition: refuse by name on another upstream source
    ef2_srcguard.check_many({"t15": trunk, "t15msa": msa, "xtr": xtr})
    if xtr and msa:
        raise ValueError("ef2_pair_v2.install: xtr with t15msa — one owner of the MSA blocks' PairTransition per process (xtr is the exact line's, t15msa / t16 the fast line's)")
    TR = _face()
    word = bound_word()
    if word not in tuple(TR.TIER_WORDS) and word.split(":")[0].split("@")[0] not in TR.ROW_NAMES:
        raise ValueError(f"ef2_pair_v2: {WORD_ENV}={word!r} is neither a tier word {TR.TIER_WORDS} nor a row word of opt_core.kernels.transition")
    if trunk:
        if _STATE["orig_transition_forward"] is None:
            _STATE["orig_transition_forward"] = C.Transition.forward
        C.Transition.forward = _transition_forward_v2
    _STATE["installed"] = True
    n_msa = 0
    enc = getattr(model, "msa_encoder", None) if model is not None else None
    if msa and enc is not None:
        for blk in enc.blocks:
            if getattr(blk, "_pair_v2_prev_forward", None) is None:
                blk._pair_v2_prev_forward = blk.forward          # bound: ef2_opt's M1 forward if installed, else the class forward
                blk.forward = types.MethodType(_msa_block_forward_v2, blk)
                _STATE["msa_blocks"].append(blk)
                n_msa += 1
    _STATE["t15"] = bool(trunk); _STATE["t15msa"] = bool(msa) and n_msa > 0
    n_pack = 0
    if (trunk or _STATE["t15msa"]) and model is not None:            # packs built now (eager, before any graph capture) for every servable module
        with torch.no_grad():
            for mod in model.modules():
                if isinstance(mod, (C.Transition, MOD.PairTransition)) and _servable(mod) and mod.ffn.w12.weight.is_cuda:
                    _pack(mod); n_pack += 1
    n_xtr = 0
    if xtr and enc is not None:                                       # lever xtr: MOD.PairTransition class patch; packs built now
        with torch.no_grad():
            for mod in enc.modules():
                if isinstance(mod, MOD.PairTransition) and int(mod.norm.normalized_shape[0]) in (256, 128) and getattr(mod.ffn.w12, "bias", None) is None and mod.ffn.w12.weight.is_cuda:
                    _pack(mod); n_xtr += 1
        if n_xtr and _STATE["orig_pair_transition_forward"] is None:
            _STATE["orig_pair_transition_forward"] = MOD.PairTransition.forward
            MOD.PairTransition.forward = _pair_transition_forward_xtr
    _STATE["xtr"] = bool(xtr) and n_xtr > 0; _STATE["xtr_modules"] = n_xtr
    return dict(version=VERSION, word=word, core=_STATE["core"], provider="opt_core.kernels.transition", transition=("C.Transition.forward" if trunk else None),
                msa_blocks=n_msa, packed=n_pack, xtr=_STATE["xtr"], xtr_modules=n_xtr)


def uninstall(model=None):
    if _STATE["orig_transition_forward"] is not None:
        C.Transition.forward = _STATE["orig_transition_forward"]
        _STATE["orig_transition_forward"] = None
    if _STATE["orig_pair_transition_forward"] is not None:
        MOD.PairTransition.forward = _STATE["orig_pair_transition_forward"]
        _STATE["orig_pair_transition_forward"] = None
    _STATE["t15"] = _STATE["t15msa"] = _STATE["xtr"] = False
    _STATE["xtr_modules"] = 0
    _STATE["word"] = None; _STATE["rows"].clear(); _STATE["refused"].clear(); _STATE["printed"].clear()
    for blk in _STATE["msa_blocks"]:
        prev = blk.__dict__.pop("_pair_v2_prev_forward", None)
        if prev is not None:
            blk.forward = prev
    _STATE["msa_blocks"].clear()
    _STATE["installed"] = False
    return dict(version=VERSION, installed=False)


def stats():
    d = dict(STATS)
    if _STATE["rows"]:
        d["rows"] = dict(_STATE["rows"])
    if _STATE["refused"]:
        d["refused"] = dict(_STATE["refused"])
    return d


def bind_words():
    """Blank-free k=v words for the t15 / t15msa / xtr LEVER lines: word=<bound word> provider= core= (+ rows= refused= once calls ran)."""
    ev = {"word": str(bound_word()), "provider": "opt_core.kernels.transition", "core": str(_STATE.get("core") or "none")}
    if _STATE["rows"]:
        ev["rows"] = ",".join(f"{k}={v}" for k, v in sorted(_STATE["rows"].items()))
    if _STATE["refused"]:
        ev["refused"] = ",".join(f"{k}={v}" for k, v in sorted(_STATE["refused"].items()))
    return ev


t15_bind_words = bind_words                                           # the report's name for the t15 / t15msa evidence
xtr_words = bind_words                                                # ... and for xtr's


def levers_on():
    """{lever name: bool} for the three levers as they stand in this process."""
    return {k: bool(_STATE.get(k)) and bool(_STATE["installed"]) for k in ("t15", "t15msa", "xtr")}


def describe():
    return dict(version=VERSION, levers=dict(t15="fast / big: C.Transition through the transition provider by tier word (residual folded)",
                                              t15msa="fast / big: MSA-block PairTransition + residual through the same call",
                                              xtr="exact: MSA-module PairTransition through the transition provider by tier word (LayerNorm given)"),
                installed=_STATE["installed"], on=levers_on(), word=bound_word(), core=_STATE.get("core"), xtr=_STATE.get("xtr"), xtr_modules=_STATE.get("xtr_modules"),
                msa_blocks=len(_STATE["msa_blocks"]), rows=dict(_STATE["rows"]), refused=dict(_STATE["refused"]))
