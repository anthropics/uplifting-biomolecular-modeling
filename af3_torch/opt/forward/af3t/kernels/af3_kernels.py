# af3_kernels.py -- fused-kernel adapters for xfold (github.com/Shenggan/xfold @ 22bdeed, PyTorch AlphaFold3), H100, INFERENCE ONLY.
#
#   import af3_kernels as K
#   K.enable(["trimul", "triattn", "transition", "apb"])   # any subset; composable
#   ... run xfold as usual (torch.inference_mode + torch.autocast("cuda", torch.bfloat16), fastnn config = 'torch') ...
#   K.census()      # dict of served / fallback counters per lever (also printed once at interpreter exit)
#   K.disable()     # restores every patched class attribute
#
# All patches are CLASS-level monkey-patches of xfold classes (no weight edits, no instance surgery). Packed / recast weights are cached per module
# on `module._af3k` on first use (pack ONCE). Every lever fails safe: outside its served cell it calls the stock forward and counts the reason
# (printed once per distinct reason). Kernel exceptions disable that lever for the rest of the process and fall back to stock; an out-of-memory error propagates.
# Numerics classes: trimul / triattn / transition / apb = "same class as stock-under-bf16-autocast" (bf16 tensor-core operands,
# fp32 accumulation, fp32 LN/softmax statistics, stock rounding points; not byte-equal to stock; byte-equal run-to-run).
import os, sys, json, atexit
import torch
import torch.nn.functional as F
from opt_core.oom import is_oom          # the shared core's one OOM recogniser: every kernel-error route below re-raises an OOM before anything else

LEVERS = ("trimul", "triattn", "transition", "apb", "resid_fold", "attn_epi", "tmpl_trimul", "pwa_lnl", "pwa_msa", "trimul_exact", "glu_proj")   # trimul_exact: the exact-class TriMul binding (the shared core provider's `exact` word in this engine's module form); resid_fold modifies trimul+transition, attn_epi triattn: the block's residual adds folded into those kernels' epilogues
_ON = set()
_TRIATTN_PROV = {"mod": None, "stack": None, "sel": {}, "refused": None}   # lever triattn's binding to the shared core's triangle-attention provider (opt_core.kernels.triattn): "mod" = the provider
                                           #   module once bound (_apply_arch_cells), "stack" = this process's torch/python/sm key the provider's prebuilt rows are keyed by, "sel" = the
                                           #   provider's Selection (None: the stock statement serves that shape, by name) per (N, H, D, word) -- the word is the MODE's tier word
                                           #   (_TIER), the row per cell is the provider's; "refused" = the last "<row>:<kind>" refusal by name
_MASK5 = {"src": None, "N": 0, "val": None}    # lever triattn's bool key mask [1,N,1,1,N], memoised on the pair-mask tensor it was built from (identity): 1 launch per Evoformer pass instead of 1 per call
_TRIMUL_PROV = {"mod": None, "stack": None, "sel": {}, "refused": None, "cache": {}, "has_cueq": None, "why": None}   # the binding of levers trimul / tmpl_trimul / trimul_exact to the
                                           #   shared core's TriMul provider (opt_core.kernels.trimul): "mod" = the provider module once bound (None: absent from this interpreter, "why" =
                                           #   the reason; every triangle multiplication then keeps the module statement, by name), "stack" = this process's stack word (the provider's
                                           #   cells and vouch records are per stack), "sel" = the provider's Selection (or None: refused by name) per (word, N, C, direction, precision),
                                           #   "refused" = the last "<row>:<kind>" refusal. No row word and no launch cell is chosen in this tree: the provider's measured cells name the row.
_TIER = {"word": "fast"}                   # the provider TIER WORD this process's mode names (opt_core.kernels.trimul TIER_WORDS: exact | fast | big): set by the model
                                           #   process before build_model (af3_torch_api.set_provider_tier: the mode's own word -- exact in exact, big when the pred composes big's
                                           #   memory levers, fast otherwise); levers trimul / tmpl_trimul ask THIS word, lever trimul_exact asks `exact` (its row is the exact word's);
                                           #   levers transition and apb ask the shared core's transition / pair-bias attention providers for the same word (_TRANS_PROV, _APB_PROV)
_EXACT_FORM = "af3t_module"                # the provider's module-form word for this engine's TriangleMultiplication statement (opt_core.kernels.trimul FORMS): word `exact` under this
                                           #   form = row af3t_form (the module statement issued whole-tensor) where the provider's table vouches it byte-identical to the module for the
                                           #   running stack, width and size; a shape or stack without a vouch record is refused BY NAME (the module's own statement runs)
_APB_PROV = {"mod": None, "tried": False, "sel": {}, "refused": None, "graph_cells": True}   # lever apb's binding to the shared core's pair-bias attention provider (opt_core.kernels.apb):
                                           #   the word asked is ALWAYS the mode's tier word (_TIER: fast | big) -- the provider's measured cell per (cc, dtype, geometry, samples,
                                           #   N bucket, eager|graph) names the row (its own Triton kernel, a carried package, or its stock SDPA row); "mod" = the face once imported
                                           #   (None after "tried": not importable here -> the kit's own SDPA statement and fused LayerNorm + projection serve, by name); "sel" =
                                           #   the provider's Selection per call class (op, cell, samples, N, heads, head dim, timing) or "<row>:<kind>" once that class was refused
                                           #   by name (the kit statement serves it, counted refused:<op>:<row>:<kind>); "refused" = the last refusal; "graph_cells" = the diffusion
                                           #   transformer's attention consults the provider's graph-replay cells (True: the sampler captures whole steps -- stepgraph -- so the
                                           #   eager warm-up steps launch the same row the capture will replay; False: hoist only, eager cells; af3_torch_api.build_model sets it)
_TRANS_PROV = {"mod": None, "cc": None, "stack": None, "refused": None, "sel": {}}   # lever transition's binding to the shared core's transition provider (opt_core.kernels.transition),
                                           #   asked by the MODE's tier word (_TIER: fast | big) per call class: "mod" = the provider module once bound (_apply_arch_cells; None =
                                           #   the package did not import here: the stock statement serves, counted fallback:no-provider); "cc" / "stack" = this device's capability and
                                           #   software-stack words; "sel" = per (tier word, c, hidden, dtype, tokens, residual, capture) the provider's Selection (a kernel row the
                                           #   face serves, or a stock row: the engine's own module measured fastest for that cell) | the kind word of a refusal by name (no cell for
                                           #   the shape on this card, an unmeasured stack, a row that cannot build here: the stock statement serves that class, counted
                                           #   fallback:c=<c>,<kind>); "refused" = the last "<tier>:<kind>" refusal.  No launch tile is chosen in this tree: every row's launch is
                                           #   the provider's measured cell word for the card
_TRIATTN_EPI = {}                          # (c_out, H, D) -> the shared core's fpf_triatt_epi launch cell on this device (the core's pair-fused cell table, opt_core.attn.pair_fused
                                           # lookup_cell, resolved once: _apply_arch_cells); a key present = lever triattn's gate * o -> output projection -> block residual add run as
                                           # ONE kernel (fpf_triatt_epi.triatt_epilogue, block mode); absent = lnl_fused.gate_transpose + F.linear + the block's own add
_ORIG = {}
_DEAD = {}                       # lever -> repr(exception) once a kernel error disabled it
# A kernel that raises disables its lever for the rest of the process and the stock path serves (_kernel_error); an OOM propagates first (is_oom).
COUNTS = {k: {} for k in LEVERS}
_PRINTED = set()
_GLU_MIN_ROWS = 16384                                                                   # lever glu_proj: the gated-linear-unit + projection kernel serves a transition statement from this many rows (below it the two stock launches are microseconds; nothing to fuse for)
_GLU_CHECKED = {}                                                                       # lever glu_proj: (rows, C, HID, dtype, residual) -> True once the restatement was checked byte-equal to the stock statement at that shape in THIS process (False: the cuBLAS statement differs there -> stock serves that shape, by name)
_GLU_CARD_OFF = {"8.0": (64,)}                                                        # lever glu_proj: per card (cc), the channel widths served by the stock statement BY NAME -- cc 8.0, c=64: the restatement's
                                                                                        #   bytes differ from the cuBLAS bf16 GEMM torch selects there at rows >= 16384 (byte-equal at c=128, differs at c=64 on this
                                                                                        #   stack); the first-call check would hand every such class to stock anyway -- naming it keeps
                                                                                        #   the census inside the declared coverage, fallback:c=64,card-off)
_GLU_CARD_OFF_C = set()                                                                 # lever glu_proj: the channel widths THIS card serves by the stock statement, by name (_GLU_CARD_OFF[cc], filled per process in _apply_arch_cells)


def _log(msg):
    sys.stderr.write("[af3_kernels] %s\n" % msg); sys.stderr.flush()


def _count(lever, key):
    d = COUNTS[lever]; d[key] = d.get(key, 0) + 1


def _fallback(lever, reason):
    _count(lever, "fallback:" + reason)
    if (lever, reason) not in _PRINTED:
        _PRINTED.add((lever, reason)); _log("%s: fallback -> stock (%s)" % (lever, reason))


def _capturing() -> bool:
    """The current CUDA stream is being captured into a graph (the whole-step graph of DiffusionHead.forward_graphed): an exception raised
    under capture is the capture's event — an invalidated or unsupported capture surfaces at whatever call allocates next — not this kernel's,
    so the served sites re-raise it to the capture's owner (which steps the step graph aside by name and runs the step eagerly, these levers
    still serving) instead of disabling their lever for the process."""
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _kernel_error(lever, e):
    """A kernel of `lever` raised `e` outside a capture: the lever is disabled for the rest of the process and the stock path serves this
    call and every later one (the caller re-raised an OOM, and any error under capture, before routing here: is_oom, _capturing)."""
    _DEAD[lever] = repr(e)[:300]
    _count(lever, "kernel_error")
    _log("%s: KERNEL ERROR -> lever disabled for the rest of the process, stock path serves: %s" % (lever, repr(e)[:300]))


def _cache(m):
    c = getattr(m, "_af3k", None)
    if c is None:
        c = {}
        object.__setattr__(m, "_af3k", c)          # plain attribute, not a Parameter/buffer/submodule
    return c


def _autocast_bf16():
    return torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.bfloat16


def _of3():
    """True when running the af3t-patched xfold with the OpenFold3 weight layout flag set (xfold/of3.py: of3.OF3)."""
    try:
        from xfold import of3
        return bool(of3.OF3)
    except Exception:
        return False


def census():
    out = {k: dict(v) for k, v in COUNTS.items()}
    out["on"] = sorted(_ON); out["dead"] = dict(_DEAD)
    out["mask_terms"] = len(_MTERM)
    out["arch"] = dict(_ARCH)
    if "apb" in _ON or _APB_PROV["tried"]:
        out["arch"]["apb_provider"] = provider_binding()                    # lever apb: the tier word asked of the shared core's provider and the arm it served per call class
    return out


def _print_census():
    if any(COUNTS[k] for k in LEVERS) or _ON:
        _log("CENSUS %s" % (census(),))


atexit.register(_print_census)


# =============================================================================================================================================
# levers 'trimul' / 'tmpl_trimul' / 'trimul_exact' -- xfold.nn.triangle_multiplication.TriangleMultiplication served by the shared core's TriMul provider
# (opt_core.kernels.trimul) by TIER WORD: the model process names its mode's word before build_model (set_tier: exact | fast | big) and the provider's measured
# cell table names the row per (cc, precision, c_z, c_hidden, size bucket, direction) on the running stack -- no row word and no launch cell is chosen in this tree.
#   trimul        c = 128 (trunk / MSA-module / confidence pair stacks; 256 where a model carries it): the mode's tier word (fast | big).
#   tmpl_trimul   beside trimul: the template pair stack's c = 64 blocks (c_z 64, c_hidden 64) under the same tier word; off = those rows keep the module statement (fallback c=64).
#   trimul_exact  the provider's `exact` word in this engine's module form (_EXACT_FORM -> row af3t_form: byte-identical to the module where the provider's table vouches it
#                 for the running stack and size; anywhere else the provider refuses BY NAME and the module's own statement runs, counted fallback:unvouched). `exact` carries it
#                 alone; under fast / big lever trimul owns the class and trimul_exact steps aside per call by name (fallback:superseded:trimul).
# Every row is served through the provider face with the module's ten tensors. A tier word whose cell names a stock row (the provider measured a library / torch statement
# fastest there) is served by the module's OWN statement, by name (fallback stock_row:<row>): a pair stack is never routed to another statement of the op silently. A row that
# refuses with the tensors in hand steps aside inside the face to the cell's next measured row (said once on stdout, counted stepaside:<from>-><to>); a refusal nothing else of
# the cell serves = the module statement, by name (fallback refused).
# Weight mapping (xfold mirrors AlphaFold 3's JAX reference): projection/gate Linear(c, 2c, no bias) -> channel-major [2c,N,N] -> reshape(c, 2, N, N): a = rows 0::2, b = rows 1::2 (INTERLEAVED).
#   outgoing  'cik,cjk->cij':  X_ij = sum_k a_ik b_jk  == the provider's 'outgoing' with (A, B) = (a, b)
#   incoming  'ckj,cki->cij':  X_ij = sum_k a_kj b_ki  == the provider's 'incoming' (sum_k A_ki B_kj) with (A, B) = (b, a)   <- a/b SWAPPED for incoming
# mask: xfold multiplies all 2c projected channels at (i,j) by pair_mask[i,j] == the provider's op (a and b masked). LN eps 1e-5 both norms. No biases anywhere.
# =============================================================================================================================================
def _plus(pair, update, residual):
    """The block's residual statement when the add was NOT folded into a kernel epilogue: `pair += update` in place (returned) under residual=True —
    exactly the stock statement of xfold/nn/pairformer.py — else the bare update (the stock module contract: forward returns the update)."""
    if residual:
        pair += update
        return pair
    return update


def _trimul_wdict(m):
    """The ten TriMul tensors by the provider's names (opt_core.kernels.trimul WEIGHT_KEYS), for rows served through the provider face."""
    c = _cache(m)
    if "trimul_w10" not in c:
        P = m.projection.weight.detach(); Gw = m.gate.weight.detach()
        a_p, b_p, a_g, b_g = P[0::2].contiguous(), P[1::2].contiguous(), Gw[0::2].contiguous(), Gw[1::2].contiguous()
        if m.equation == 'cik,cjk->cij':
            w_ap, w_bp, w_ag, w_bg = a_p, b_p, a_g, b_g
        else:
            w_ap, w_bp, w_ag, w_bg = b_p, a_p, b_g, a_g
        c["trimul_w10"] = {"ln_in_w": m.left_norm_input.weight.detach(), "ln_in_b": m.left_norm_input.bias.detach(), "w_ag": w_ag, "w_ap": w_ap, "w_bg": w_bg, "w_bp": w_bp,
                           "ln_out_w": m.center_norm.weight.detach(), "ln_out_b": m.center_norm.bias.detach(), "w_o": m.output_projection.weight.detach(), "w_og": m.gating_linear.weight.detach()}
        c["trimul_face"] = {}
    return c["trimul_w10"], c["trimul_face"]


def set_tier(word):
    """Name the provider TIER WORD this process's mode asks the shared core's TriMul provider for (exact | fast | big; af3_torch_api.set_provider_tier, called by the
    model process before build_model). Levers trimul / tmpl_trimul ask this word; lever trimul_exact asks `exact` whatever it is. Naming the word drops memoised selections."""
    if word not in ("exact", "fast", "big"):
        raise ValueError("provider tier word must be exact | fast | big (got %r)" % (word,))
    _TIER["word"] = word
    _TRIMUL_PROV["sel"].clear()
    return word


def provider_tier():
    """The tier word this process asks the shared core's providers for (set_tier; fast until a mode names big)."""
    return _TIER["word"]


def _trimul_select(lever, word, N, C, direction, prec="bf16"):
    """The provider's Selection for `word` (a tier word; `exact` is asked under this engine's module form, _EXACT_FORM) at (cc, prec, c_z=C, c_hidden=C, N, direction),
    asked of opt_core.kernels.trimul once per word, shape and precision (prec "bf16": the trunk / MSA / template pair streams; "f32z_bf16": an fp32 pair inside the bf16
    autocast region, the confidence head's). None when no row serves the call: the provider is absent from the interpreter, or the word is refused by name -- under `exact`,
    a shape or stack the provider's table does not vouch the module-form row for. A refusal is counted refused:<row>:<reason> on `lever`'s census, once per shape."""
    TPm = _TRIMUL_PROV["mod"]
    if TPm is None:
        return None
    key = (word, N, C, direction, prec)
    if key in _TRIMUL_PROV["sel"]:
        return _TRIMUL_PROV["sel"][key]
    cc = "%d.%d" % torch.cuda.get_device_capability()
    sel = None
    try:
        sel = TPm.select(cc, "bf16", C, C, N, direction, word=word, residency=("fp32" if prec == "f32z_bf16" else None), stack=_TRIMUL_PROV["stack"],
                         has_cueq=_TRIMUL_PROV["has_cueq"], form=(_EXACT_FORM if word == "exact" else None))
    except TPm.Refusal as r:
        _TRIMUL_PROV["refused"] = "%s:%s" % (r.row or word, str(r.kind).split("(")[0])
        _count(lever, "refused:%s:%s" % (r.row or word, str(r.kind).split("(")[0]))
    _TRIMUL_PROV["sel"][key] = sel
    return sel


def _trimul_serve(lever, word, self, pair, mask, residual=False):
    """TriangleMultiplication.forward through the provider under `word`, the census on `lever` (at the template stack's c = 64 under lever trimul: on tmpl_trimul).
    residual=True (the kit's PairformerBlock statement under lever resid_fold, _pairformer_forward): returns pair + update with the add folded into the serving row's
    epilogue (the update rounded to bf16 as stock returns it, + pair, one bf16 store = the bytes of torch's `pair += update`; a new tensor, the caller rebinds); every
    by-name step-aside does the stock in-place add itself (_plus). residual=False (every other caller: exact's blocks, the MSA stack's EvoformerBlock, the template stack):
    the module contract, the update alone."""
    # _ORIG["trimul"]: the module's own statement (xfold's forward; package lever tri_layout's restatement of it where that lever is on) -- read per call, by name
    if lever in _DEAD:
        return _plus(pair, _ORIG["trimul"](self, pair, mask), residual)
    TPm = _TRIMUL_PROV["mod"]
    if TPm is None:
        _fallback(lever, "no-provider"); return _plus(pair, _ORIG["trimul"](self, pair, mask), residual)
    if not pair.is_cuda or pair.dim() != 3:
        _fallback(lever, "cpu-or-batched"); return _plus(pair, _ORIG["trimul"](self, pair, mask), residual)
    N, C = int(pair.shape[-2]), int(pair.shape[-1])
    widths = (64, 128, 256) if (lever == "trimul_exact" or "tmpl_trimul" in _ON) else (128, 256)   # c 64 = the template pair stack: lever tmpl_trimul beside trimul, or the exact word (vouched at both widths)
    if C != self.c_pair or C not in widths:
        _fallback(lever, "c=%d" % C); return _plus(pair, _ORIG["trimul"](self, pair, mask), residual)
    if not (_autocast_bf16() or pair.dtype == torch.bfloat16):
        _fallback(lever, "no-bf16-autocast"); return _plus(pair, _ORIG["trimul"](self, pair, mask), residual)
    if abs(self.left_norm_input.eps - self.center_norm.eps) > 0:
        _fallback(lever, "eps-mismatch"); return _plus(pair, _ORIG["trimul"](self, pair, mask), residual)
    if mask is not None and (mask.dim() != 2 or tuple(mask.shape) != (N, N)):
        _fallback(lever, "mask-shape"); return _plus(pair, _ORIG["trimul"](self, pair, mask), residual)
    lever_c = "tmpl_trimul" if (C == 64 and lever == "trimul") else lever      # the census name at this width
    try:
        z = pair if pair.is_contiguous() else pair.contiguous()
        if residual and z.dtype != torch.bfloat16:                             # the folded add is stock's bytes on a bf16 pair (what the trunk carries under autocast); anything else keeps torch's add
            _count(lever_c, "residual_unfused:%s" % z.dtype)
            return _plus(pair, _trimul_serve(lever, word, self, pair, mask, False), True)
        direction = "outgoing" if self.equation == 'cik,cjk->cij' else "incoming"
        prec = "bf16" if z.dtype == torch.bfloat16 else "f32z_bf16"          # an fp32 pair here is inside the bf16 autocast region (the confidence head's pair stream): the provider's f32z cells
        sel = _trimul_select(lever_c, word, N, C, direction, prec)             # the provider's row for (word, N, C, direction, prec) on this stack; None: refused by name
        if sel is None:
            _fallback(lever_c, "unvouched" if word == "exact" else "refused"); return _plus(pair, _ORIG["trimul"](self, pair, mask), residual)
        if sel.row in TPm.STOCK_ROWS:                                          # the cell names a library / torch statement of the op: this module's OWN statement serves, by name
            _fallback(lever_c, "stock_row:%s" % sel.row); return _plus(pair, _ORIG["trimul"](self, pair, mask), residual)
        w10, fcache = _trimul_wdict(self)
        try:
            out = TPm.triangle_multiplication(z, mask, direction=direction, weights=w10, selection=sel, residual=bool(residual), cache=fcache, eps=self.left_norm_input.eps)
        except TPm.Refusal as r:                                               # refused with the tensors in hand and nothing else of the cell serves (a module-form row never steps aside): the module, BY NAME
            fcache.pop("_z_cast", None)                                        # (the face's cast memo is call-scoped here, as below)
            _TRIMUL_PROV["refused"] = "%s:%s" % (r.row or sel.row, str(r.kind).split("(")[0]); _TRIMUL_PROV["sel"][(word, N, C, direction, prec)] = None
            _count(lever_c, "refused:%s:%s" % (r.row or sel.row, str(r.kind).split("(")[0]))
            _fallback(lever_c, "unvouched" if word == "exact" else "refused"); return _plus(pair, _ORIG["trimul"](self, pair, mask), residual)
        if fcache.pop("_z_cast", None) is not None:                            # the face's per-call cast memo (compute_input: an fp32-resident pair's bf16 copy, made once per call for rows that read bf16)
            _count(lever_c, "z_cast_released")                                 # is CALL-scoped here: released now, so the confidence head's 8 modules never hold 8 x [N,N,c] bf16 copies (and their fp32
                                                                               # sources) across samples and items (2.82 GB resident at 1216 tokens, 26 GB at 3584, measured); numerics unchanged
        last = fcache.get("_last")
        served = getattr(last, "row", None) or sel.row                         # the row that served this call (the selection's, or the one the tier word stepped aside to inside the face)
        if served != sel.row:
            _count(lever_c, "stepaside:%s->%s" % (sel.row, served))
            if last is not None:
                _TRIMUL_PROV["sel"][(word, N, C, direction, prec)] = last      # later calls of this shape go straight to the row that served (a stock row: the module's own statement, by name)
        _count(lever_c, "served:%s:%s" % (served, direction))
        if residual: _count("resid_fold", "served:trimul_%s" % direction)
        if torch.is_autocast_enabled() and out.dtype != torch.get_autocast_dtype("cuda"):
            out = out.to(torch.get_autocast_dtype("cuda"))                    # stock output dtype under autocast
        _count(lever_c, "served:%s" % direction)
        return out
    except Exception as e:
        if is_oom(e) or _capturing(): raise                                    # OOM propagates; no fallback applied (opt_core.oom.is_oom); under a CUDA-graph capture the error is the capture's (_capturing)
        _kernel_error(lever, e); return _plus(pair, _ORIG["trimul"](self, pair, mask), residual)


def _trimul_forward(self, pair, mask, residual=False):
    """Lever trimul (and tmpl_trimul at the template pair stack's c = 64): the MODE's provider tier word."""
    return _trimul_serve("trimul", _TIER["word"], self, pair, mask, residual)


def _trimul_exact_forward(self, pair, mask, residual=False):
    """Lever trimul_exact with lever trimul off (`exact`; or trimul ablated): the provider's `exact` word in this engine's module form."""
    return _trimul_serve("trimul_exact", "exact", self, pair, mask, residual)


def _trimul_forward_exact_aside(self, pair, mask, residual=False):
    """Both levers selected (fast / big): lever trimul's tier word serves the class; lever trimul_exact steps aside per call BY NAME."""
    _count("trimul_exact", "fallback:superseded:trimul")
    return _trimul_serve("trimul", _TIER["word"], self, pair, mask, residual)


# =============================================================================================================================================
# lever 'triattn' -- xfold.nn.attention.GridSelfAttention -> [lnl_fused.ln_linear: LN + bf16 cast (+ row transpose for transpose=True) + pair-bias projection in one pass]
#                    -> one concatenated q|k|v|gate GEMM -> the attention core: the shared core's triangle-attention provider (opt_core.kernels.triattn) asked for the
#                    MODE's tier word (_TIER: fast | big) at every (card, bf16, head dim, heads, size) cell -- the trunk / MSA-module / confidence pair stacks'
#                    head-dim-32 attention and the template pair stack's head-dim-16 attention alike; the provider names the row per cell (its measured row for the
#                    card: a prebuilt CUDA row, its sealed triangle-attention package, K2B, flash ...) and the kit carries no row word, tile or size table of its own
#                    -> lnl_fused.gate_transpose (sigmoid(gate)*o, + transpose back) -> output_projection.  Stock computes the bias from the UN-transposed LN(pair)
#                    and applies mask as pair_mask[row, key] WITHOUT transposing it -- mirrored exactly.
# =============================================================================================================================================
def _triattn_weights(m):
    c = _cache(m)
    if "triattn" not in c:
        dt = torch.bfloat16
        Wcat = torch.cat([m.q_projection.weight, m.k_projection.weight, m.v_projection.weight, m.gating_query.weight], 0).detach().to(dt).contiguous()
        H = m.num_head; C = m.c_pair
        NOUT = 16 if H <= 16 else 1 << (H - 1).bit_length()
        Wb = torch.zeros((NOUT, C), device=Wcat.device, dtype=dt); Wb[:H] = m.pair_bias_projection.weight.detach().to(dt)
        for lin in (m.q_projection, m.k_projection, m.v_projection, m.gating_query, m.pair_bias_projection, m.output_projection):
            assert lin.bias is None
        c["triattn"] = dict(Wcat=Wcat, Wb=Wb, lnw=m.act_norm.weight.detach().float().contiguous(), lnb=m.act_norm.bias.detach().float().contiguous(), eps=m.act_norm.eps)
        wo16 = m.output_projection.weight.detach().to(torch.bfloat16).contiguous()          # [C, C] (K contiguous): the epilogue's out-projection operand (= autocast's per-call cast of the weight)
        c["triattn"]["Wo16"] = wo16; c["triattn"]["WoT16"] = wo16.t().contiguous()          # + its transposed copy, built once (fpf_triatt_epi keys its own woT16 cache on the version counter, which inference tensors lack)
    return c["triattn"]


def _triattn_word():
    """The word lever triattn hands the provider: the MODE's tier word (_TIER: fast | big; af3_torch_api.set_provider_tier) -- fast in fast, big in big."""
    return _TIER["word"]


def _triattn_select(N, H, D):
    """The provider's Selection for lever triattn at (this card, bf16, head dim D, H heads, N tokens), memoised per (N, H, D, word): the mode's tier word
    (_triattn_word: fast | big) asked of opt_core.kernels.triattn with the kit's call form (form="keypad": head-plane pair bias + key-padding mask) -- the provider resolves
    it to its measured row for the card's cell; the kit names no row.  None = no kernel row serves this shape in this process (the provider refused the word by
    name -- no cell for this card / head dim, no prebuilt for this stack with no kernel fallback -- or resolved it to a stock row): those calls keep the stock
    statement, by name (counted refused:<row>:<kind> once per shape, fallback:refused:<row>:<kind> per call)."""
    TP = _TRIATTN_PROV["mod"]
    if TP is None:
        return None
    word = _triattn_word()
    key = (N, H, D, word)
    memo = _TRIATTN_PROV["sel"]
    if key in memo:
        return memo[key]
    return _triattn_bind(key, word)


def _triattn_bind(key, word):
    """Ask the provider for `word` at shape `key` = (N, H, D, tier word) and memoise the answer under key: the Selection of a kernel row, or None (refused by
    name, or a stock row named: the kit's stock statement IS its stock op).  A refusal's named fallback row is followed (the provider's own instruction:
    a kit binds .fallback), at most two hops."""
    TP = _TRIATTN_PROV["mod"]
    N, H, D, _tier = key
    cc = "%d.%d" % torch.cuda.get_device_capability()
    sel = None
    for _ in range(3):
        try:
            sel = TP.select(cc, "bf16", D, H, N, "fwd", word=word, stack=_TRIATTN_PROV["stack"], form="keypad")
            break
        except TP.Refusal as r:
            _TRIATTN_PROV["refused"] = "%s:%s" % (r.row or word, r.kind)
            _count("triattn", "refused:%s:%s" % (r.row or word, r.kind))
            if not r.fallback or r.fallback == word or r.fallback in _TRIATTN_STOCK_ROWS(TP):
                sel = None; break
            word = r.fallback                                               # the provider names the row that serves instead: bound by ITS name, counted above
    if sel is not None and sel.row in _TRIATTN_STOCK_ROWS(TP):               # the word resolved to a stock row: this kit's stock op is the stock GridSelfAttention statement itself, served whole, by name
        _TRIATTN_PROV["refused"] = "%s:stock_row" % sel.row
        _count("triattn", "refused:%s:stock_row" % sel.row)
        sel = None
    _TRIATTN_PROV["sel"][key] = sel
    return sel


def _TRIATTN_STOCK_ROWS(TP):
    """The provider's rows that ARE a stock op (served by the kit's own stock callable through the face): never a kernel this kit routes the trunk to."""
    return set(getattr(TP, "STOCK_ROWS", ())) | set(getattr(TP, "NEEDS_STOCK", ()))


def _triattn_forward(self, pair, mask, residual=False):
    """residual=True (the kit's PairformerBlock statement, _residual_call): returns pair + update — with a tuned epilogue cell on this device the gate,
    output projection and the add run as ONE kernel writing the bf16 pair stream in place (fpf_triatt_epi block mode: sigmoid(g)*o rounded to bf16,
    @ Wo^T accumulated in fp32 in ascending k, + pair, one bf16 store — bitwise the gate_transpose + F.linear + `pair +=` statements it replaces, ending
    node scattered into the untransposed frame); without a cell, or on an fp32 pair stream, the update is formed as before and added by torch (_plus).
    residual=False: the stock contract, the update alone."""
    lever = "triattn"
    if lever in _DEAD:
        return _plus(pair, _ORIG["triattn"](self, pair, mask), residual)
    if not pair.is_cuda or pair.dim() != 3:
        _fallback(lever, "cpu-or-batched"); return _plus(pair, _ORIG["triattn"](self, pair, mask), residual)
    N, C = pair.shape[-2], pair.shape[-1]
    H = self.num_head; D = C // H
    if C not in (64, 128) or D not in (16, 32, 64, 128):
        _fallback(lever, "c=%d,d=%d" % (C, D)); return _plus(pair, _ORIG["triattn"](self, pair, mask), residual)
    if not _autocast_bf16():
        _fallback(lever, "no-bf16-autocast"); return _plus(pair, _ORIG["triattn"](self, pair, mask), residual)
    if mask is not None and not (mask.dim() == 2 and tuple(mask.shape) == (N, N)) and not (mask.dim() == 1 and mask.shape[0] == N):
        _fallback(lever, "mask-shape"); return _plus(pair, _ORIG["triattn"](self, pair, mask), residual)
    if N < 16:
        _fallback(lever, "N<16"); return _plus(pair, _ORIG["triattn"](self, pair, mask), residual)
    sel = _triattn_select(N, H, D)                                                 # the shared core's triangle-attention provider row for this (card, bf16, D, H, N) under the mode's tier word
    if sel is None:                                                                 # no kernel row serves this shape here (refused by name / a stock row named): the stock statement, by name
        _fallback(lever, "refused:%s" % _TRIATTN_PROV["refused"]); return _plus(pair, _ORIG["triattn"](self, pair, mask), residual)
    TP = _TRIATTN_PROV["mod"]
    try:
        import lnl_fused as RFU
        W = _triattn_weights(self)
        x = pair if pair.is_contiguous() else pair.contiguous()
        y16, b16 = RFU.ln_linear(x, W["lnw"], W["lnb"], W["Wb"], eps=W["eps"], write_y=True, transpose=bool(self.transpose), planes=True)   # head-plane-major [NOUT, N, N]
        bias5 = b16[:H][None, None]                                        # [1,1,H,N,N] (query i, key j) from untransposed LN(pair) -- as stock; head planes contiguous (the core's bias pass reads unit strides)
        if self.transpose and _of3():
            bias5 = bias5.transpose(-1, -2)                                # OF3 weight layout (af3t xfold/of3.py): column attention uses the transposed bias
        qkvg = F.linear(y16, W["Wcat"])                                    # [N',N',4C] bf16
        def heads(t):                                                      # 'b n (h d) -> 1 b h n d' as a strided view
            return t.unflatten(-1, (H, D)).permute(0, 2, 1, 3)[None]
        q5, k5, v5 = heads(qkvg[..., 0:C]), heads(qkvg[..., C:2 * C]), heads(qkvg[..., 2 * C:3 * C])
        mask5 = None
        if mask is not None:
            if _MASK5["src"] is mask and _MASK5["N"] == N:                    # one trunk pass hands the SAME pair-mask tensor to every block: its bool key mask is built once per pass, not per call                    # one Evoformer pass hands the SAME pair-mask tensor to every block: its bool key mask is built once per pass, not per call
                mask5 = _MASK5["val"]
            else:
                mb = mask if mask.dtype == torch.bool else (mask != 0)
                mask5 = mb[None, :, None, None, :] if mb.dim() == 2 else mb[None, None, None, None, :].expand(1, N, 1, 1, N)
                _MASK5.update(src=mask, N=N, val=mask5)
        try:
            o = TP.triangle_attention(q5, k5, v5, bias5, mask5, D ** -0.5, word=sel.word, selection=sel)      # [1, N', H, N', D] bf16 contiguous: the provider's row for the cell
        except TP.Refusal as r:                                            # refused by name AT the call (a prebuilt that does not load on this stack, an operand bound of the row): the provider's
            if _capturing(): raise                                         #   named fallback row serves this shape from here on, this call included; none named = the stock statement, by name
            nxt, key = r.fallback, (N, H, D, _triattn_word())
            _TRIATTN_PROV["refused"] = "%s:%s" % (r.row or sel.row, r.kind)
            _count(lever, "refused:%s:%s" % (r.row or sel.row, r.kind))
            if nxt and nxt != sel.row and nxt not in _TRIATTN_STOCK_ROWS(TP):
                sel = _triattn_bind(key, nxt)                              # the fallback row the provider names, bound by ITS name (memoised for the shape)
            else:
                sel = _TRIATTN_PROV["sel"][key] = None
            if sel is None:
                _fallback(lever, "refused:%s" % _TRIATTN_PROV["refused"]); return _plus(pair, _ORIG["triattn"](self, pair, mask), residual)
            o = TP.triangle_attention(q5, k5, v5, bias5, mask5, D ** -0.5, word=sel.word, selection=sel)
        _count(lever, "row:%s:d%d" % (sel.row, D))                                # the provider row that attended, per head dim (32: trunk / MSA / confidence pair stacks; 16: the template pair stack)
        epi = _TRIATTN_EPI.get((C, H, D)) if residual and x.dtype == torch.bfloat16 else None  # the block statement on a bf16 pair stream with a tuned cell: one kernel
        if residual and epi is None: _count("attn_epi", "unfused:%s" % ("no_cell" if x.dtype == torch.bfloat16 else "fp32_pair"))   # lever attn_epi on without a cell for this (card, shape) or on an fp32 pair stream: the separate statements, counted by name (not a kernel fallback: the op kernels served)
        if epi is not None:
            from fpf_triatt_epi.epilogue import triatt_epilogue            # the routed core package (registry.KERNEL_ROUTES)
            triatt_epilogue(o, qkvg[None][..., 3 * C:], W["Wo16"], z=x[None], ending=bool(self.transpose), residual=True, cfg=epi, woT16=W["WoT16"])   # x[.., :] += (sigmoid(g)*o) @ Wo^T, in place
            _count("attn_epi", "served:%s" % ("end" if self.transpose else "start"))
            _count(lever, "served:%s+epi" % ("end" if self.transpose else "start"))
            return x                                                       # = pair when pair was contiguous (updated in place, as `pair +=` is); else its updated copy (the caller rebinds)
        y = RFU.gate_transpose(o, qkvg[None], 3 * C, transpose=bool(self.transpose))[0]         # [N, N, C] in the caller's (untransposed) layout
        _count(lever, "served:%s" % ("end" if self.transpose else "start"))
        return _plus(pair, F.linear(y, self.output_projection.weight), residual)              # autocast -> bf16; residual: the block's own in-place add
    except Exception as e:
        if is_oom(e) or _capturing(): raise                                # OOM propagates; no fallback applied (opt_core.oom.is_oom); under a CUDA-graph capture the error is the capture's (_capturing)
        _kernel_error(lever, e); return _plus(pair, _ORIG["triattn"](self, pair, mask), residual)


# =============================================================================================================================================
# lever 'transition' -- xfold.nn.primitives.Transition (LayerNorm -> [W1|W2] gated linear unit -> W3) served by the shared core's transition provider
# (opt_core.kernels.transition) asked by the MODE's tier word (_TIER: fast | big).  Per (cc, dtype, c, hidden, tokens bucket) cell the provider serves the row it
# measured fastest in class on this card (rows v2 / v1 / af3_fused / lnl / ... : ONE carried copy each in the core, launched with the cell's measured launch word --
# no candidate tile is timed in the model process and this tree carries no copy and no launch table of any of them), names the engine's own module where THAT
# measured fastest (stock rows: the stock statement serves, counted fallback:c=<c>,stock-row), or refuses by NAME (no cell for the shape on this card, an unmeasured
# software stack, a row that cannot build here) -> the class keeps the stock statement, counted fallback:c=<c>,<kind>; nothing is substituted silently.
# Shapes in this model: pair c=128 x4 (trunk / MSA-module / confidence-head pair, N^2 rows: cells pair_c128_n4), MSA c=64 x4 (S*N rows: rows_c64_n4), template pair
# stack c=64 x2 (N^2 rows per template: rows_c64_n2), single c=384 x4 (N rows: single_c384_n4).
# GLU convention (xfold/fastnn/gated_linear_unit.py::gated_linear_unit_torch): y = x @ transition1.weight.T ; a, b = chunk(y, 2, -1); out = silu(a) * b
#   -> the provider's pack: w_ab = transition1.weight ([Wa; Wb], HID = factor * C rows each), w_o = transition2.weight [C, HID], the input LayerNorm's affine + eps.
# Numerics: the fast / big rows are this family's class (bf16 operands, fp32 accumulation, the stock graph's rounding points; accumulation order differs from
# cuBLAS's) = tier 2, inside the identity band -- not bitwise; the exact tier never enables this lever (exact's transition statement is xfold's own kernels + lever
# glu_proj, bitwise: the provider's exact word is vouched against the cuBLAS SwiGLU statement, not against xfold's Triton gated-linear-unit kernel).
# Masks: none (Transition takes no mask).  CUDA-graph note: capture-safe (the provider refuses its capture-unsafe rows by name under capture).
# =============================================================================================================================================
def _transition_pack(m, TP):
    """The provider's canonical weight pack for module m (opt_core.kernels.transition.pack: w_a | w_b = transition1's rows a-then-b -- the GLU convention above --,
    w_o = transition2, the input LayerNorm's affine and eps; bf16 operand copies made once), cached on the module."""
    c = _cache(m)
    if "transition_prov" not in c:
        c["transition_prov"] = TP.pack(w_o=m.transition2.weight.detach(), w_ab=m.transition1.weight.detach(), ln_w=m.input_layer_norm.weight.detach(),
                                       ln_b=m.input_layer_norm.bias.detach(), eps=m.input_layer_norm.eps)
    return c["transition_prov"]


_PROV_DTYPE = {torch.bfloat16: "bf16", torch.float32: "fp32"}      # the provider's dtype words (any other x dtype: the stock statement, counted fallback:dtype=<dtype>)
_TRANS_FAMILY = {384: "single", 64: "rows"}                         # the provider's cell family per channel width in this model: c=384 = the single transition (N rows), c=64 = the MSA /
                                                                    # template pair-stack transitions (rows cells, keyed by hidden); every other width = pair (N^2 rows)


def _transition_decide(TP, C, HID, dt, N, rows, residual, capture):
    """The provider's decision for the mode's tier word (_TIER) at one call class (c, hidden, dtype, tokens, residual, capture) on this card, asked of
    opt_core.kernels.transition ONCE per class in the process: its Selection (a kernel row the face serves, or a stock row = the engine's own module measured fastest for
    that cell), or the kind word of a refusal by name (the stock statement serves the class).  Counted once per class: word:<tier>=<row>@<cell key> | refused:<tier>:<kind>."""
    word = _TIER["word"]
    key = (word, C, HID, dt, int(N), bool(residual), bool(capture))
    state = _TRANS_PROV["sel"].get(key)
    if state is None:
        try:
            state = TP.select(word, c=C, hidden=HID, n_tokens=int(N), dtype=dt, family=_TRANS_FAMILY.get(C, "pair"), residual=bool(residual), cc=_TRANS_PROV["cc"],
                              stack=_TRANS_PROV["stack"], capture=bool(capture), rows_count=rows)
            _count("transition", "word:%s=%s@%s" % (word, state.row, getattr(state, "cell_key", None) or "opt_in"))
        except TP.Refusal as r:
            state = str(r.kind)
            _TRANS_PROV["refused"] = "%s:%s" % (word, state)
            _count("transition", "refused:%s:%s" % (word, state))
            _log("transition: tier word %s REFUSED by name at (c=%d, hidden=%d, %s, tokens=%d, residual=%d): %s -> the stock statement serves that class"
                 % (word, C, HID, dt, int(N), int(bool(residual)), state))
        _TRANS_PROV["sel"][key] = state
    return state


def _transition_refused(TP, C, HID, dt, N, residual, capture, r):
    """A row refused at SERVE time (it cannot build on this device, ...): recorded for the class like a selection-time refusal -- the stock statement serves the
    class from here, by name."""
    state = str(getattr(r, "kind", r))
    _TRANS_PROV["sel"][(_TIER["word"], C, HID, dt, int(N), bool(residual), bool(capture))] = state
    _TRANS_PROV["refused"] = "%s:%s" % (_TIER["word"], state)
    _count("transition", "refused:%s:%s" % (_TIER["word"], state))
    _log("transition: tier word %s REFUSED by name at serve time (c=%d, hidden=%d, %s, tokens=%d): %s -> the stock statement serves that class" % (_TIER["word"], C, HID, dt, int(N), state))
    return state


def _transition_forward(self, x, residual=False):
    """residual=True (the kit's PairformerBlock statement, lever resid_fold): returns x + update, the add folded into the served row's epilogue (the update rounded
    to bf16, + x in fp32, one bf16 store = torch's `x += update` bytes on a bf16 x); by-name fallbacks do the stock in-place add (_plus).  residual=False: the stock
    contract, the update alone (every other caller).  The row is the shared core's transition provider's for the MODE's tier word at this call's cell
    (_transition_decide); a stock row or a refusal by name keeps the stock statement for the class, counted."""
    lever = "transition"
    if lever in _DEAD:
        return _plus(x, _ORIG["transition"](self, x), residual)
    C = int(x.shape[-1])
    if not x.is_cuda:
        _fallback(lever, "cpu"); return _plus(x, _ORIG["transition"](self, x), residual)
    if not _autocast_bf16():
        _fallback(lever, "no-bf16-autocast"); return _plus(x, _ORIG["transition"](self, x), residual)
    TP = _TRANS_PROV["mod"]
    if TP is None:                                                            # the provider did not import in this process (printed at start-up): the stock statement, by name
        _fallback(lever, "no-provider"); return _plus(x, _ORIG["transition"](self, x), residual)
    dt = _PROV_DTYPE.get(x.dtype)
    if dt is None:
        _fallback(lever, "dtype=%s" % x.dtype); return _plus(x, _ORIG["transition"](self, x), residual)
    if x.dim() < 2:
        _fallback(lever, "rank<2"); return _plus(x, _ORIG["transition"](self, x), residual)
    HID = int(self.transition2.weight.shape[1]); N = int(x.shape[-2]); rows = x.numel() // C; capture = _capturing()
    try:
        if residual and x.dtype != torch.bfloat16:                          # the folded add is stock's bytes on a bf16 x; anything else keeps torch's add
            _count(lever, "residual_unfused:%s" % x.dtype)
            return _plus(x, _transition_forward(self, x), True)
        sel = _transition_decide(TP, C, HID, dt, N, rows, residual, capture)
        if isinstance(sel, str):                                             # refused by name for this class
            _fallback(lever, "c=%d,%s" % (C, sel.split(":")[0])); return _plus(x, _ORIG["transition"](self, x), residual)
        if sel.row in TP.STOCK_ROWS:                                         # the provider measured the engine's own module fastest for this cell: the stock statement serves, by the table
            _fallback(lever, "c=%d,stock-row" % C); return _plus(x, _ORIG["transition"](self, x), residual)
        try:
            out = TP.transition(x, _transition_pack(self, TP), word=_TIER["word"], residual=bool(residual), n_tokens=N, family=_TRANS_FAMILY.get(C, "pair"),
                                capture=capture, stack=_TRANS_PROV["stack"])[0]
        except TP.Refusal as r:                                              # refused at serve time (a row that cannot build here, ...): named; the class keeps the stock statement
            if capture: raise
            kind = _transition_refused(TP, C, HID, dt, N, residual, capture, r)
            _fallback(lever, "c=%d,%s" % (C, kind.split(":")[0])); return _plus(x, _ORIG["transition"](self, x), residual)
        if residual: _count("resid_fold", "served:transition_c=%d" % C)
        _count(lever, "served:c=%d" % C)
        return out
    except Exception as e:
        if is_oom(e) or _capturing(): raise                                # OOM propagates; no fallback applied (opt_core.oom.is_oom); under a CUDA-graph capture the error is the capture's (_capturing)
        _kernel_error(lever, e); return _plus(x, _ORIG["transition"](self, x), residual)


# =============================================================================================================================================
# lever 'glu_proj' -- xfold.nn.primitives.Transition: the fastnn gated-linear-unit kernel + transition2 (+ the block's `pair += update`) as ONE kernel
# (af3t_glu_proj.glu_proj), the module's own layer norm serving y. The restatement keeps the stock kernels' rounding points AND accumulation order, so its
# bytes are the stock statement's — a fact this adapter CHECKS once per (rows, C, HID, dtype, residual) in the process (torch.equal against the stock
# statement's own output on the first call of a shape; a shape whose cuBLAS statement differs is served by stock from then on, BY NAME: fallback:differs).
# Served: CUDA, bf16 autocast (transition2's statement is the bf16 GEMM), xfold's fastnn gated-linear-unit kernel selected (the statement restated; the eager
# port's torch GLU rounds a | b first — another statement: fallback glu=torch), c = 64 | 128 (pair / MSA / template transitions; the c = 384 single transition is
# cuBLAS's: fallback c=384), rows >= _GLU_MIN_ROWS. With lever transition on (fast / big) the transition provider's row serves these modules and this lever is superseded by name.
# =============================================================================================================================================
def _transition_forward_glu_aside(self, x, residual=False):
    """Both levers selected (fast / big): lever transition's fused layer-norm kernel serves the module; lever glu_proj steps aside per call BY NAME."""
    _count("glu_proj", "fallback:superseded:transition")
    return _transition_forward(self, x, residual)


def _glu_proj_weights(m, dtype):
    c = _cache(m)
    key = "glu_proj_%s" % dtype
    if key not in c:
        HID = m.transition2.weight.shape[1]
        W = m.transition1.weight.detach()
        assert W.shape[0] == 2 * HID
        c[key] = dict(W1=W[:HID].to(dtype).contiguous(), W2=W[HID:].to(dtype).contiguous(), W3=m.transition2.weight.detach().to(torch.bfloat16).contiguous(), HID=HID)   # the casts the stock statement makes per call (weight.to(x.dtype); autocast's bf16 weight), made once
    return c[key]


def _glu_proj_stock(self, y, x):
    """The stock statement from the layer norm's output on: xfold's fastnn gated-linear-unit kernel with transition1's weight, transition2 under the ambient autocast,
    then (x given) the block's in-place add on x itself — the bytes the stock module and block produce, y computed once."""
    from xfold.fastnn.gated_linear_unit import gated_linear_unit_triton
    upd = self.transition2(gated_linear_unit_triton(y, self.transition1.weight.T))
    return _plus(x, upd, x is not None)


def _glu_proj_forward(self, x, residual=False):
    """Transition.forward under lever glu_proj (lever transition off): residual=True = the PairformerBlock's `pair + Transition(pair)` with the add in the kernel's
    epilogue (a new tensor; the caller rebinds), False = the module contract (the update). Every step-aside is by name and runs the stock statement."""
    lever = "glu_proj"
    if lever in _DEAD:
        return _plus(x, _ORIG["transition"](self, x), residual)
    C = x.shape[-1]
    if not x.is_cuda:
        _fallback(lever, "cpu"); return _plus(x, _ORIG["transition"](self, x), residual)
    if not _autocast_bf16():
        _fallback(lever, "no-bf16-autocast"); return _plus(x, _ORIG["transition"](self, x), residual)
    from xfold.fastnn import config as _fastnn_config
    if _fastnn_config.gated_linear_unit_implementation != "triton":           # the statement restated is the fastnn kernel's (one rounding after silu(a) * b in fp32); the torch GLU is another statement
        _fallback(lever, "glu=%s" % _fastnn_config.gated_linear_unit_implementation); return _plus(x, _ORIG["transition"](self, x), residual)
    import af3t_glu_proj as GP
    if C not in GP.SUPPORTED_C:
        _fallback(lever, "c=%d" % C); return _plus(x, _ORIG["transition"](self, x), residual)
    if x.dtype not in GP.SUPPORTED_DTYPES:
        _fallback(lever, "dtype=%s" % x.dtype); return _plus(x, _ORIG["transition"](self, x), residual)
    if C in _GLU_CARD_OFF_C:                                                  # this card names the width off (_GLU_CARD_OFF: the stock GEMM's bytes differ there): the stock statement, by name
        _fallback(lever, "c=%d,card-off" % C); return _plus(x, _ORIG["transition"](self, x), residual)
    rows = x.numel() // C
    if rows < _GLU_MIN_ROWS:
        _fallback(lever, "c=%d,rows<%d" % (C, _GLU_MIN_ROWS)); return _plus(x, _ORIG["transition"](self, x), residual)
    try:
        W = _glu_proj_weights(self, x.dtype)
        if W["HID"] % 64 != 0:
            _fallback(lever, "hid=%d" % W["HID"]); return _plus(x, _ORIG["transition"](self, x), residual)
        y = self.input_layer_norm(x)                                          # the module's own layer norm, whatever serves it
        key = (rows, C, W["HID"], x.dtype, bool(residual))
        ok = _GLU_CHECKED.get(key)
        if ok is None:                                                        # first call of this shape in the process: the restatement's bytes against the stock statement's, once
            ref = _glu_proj_stock(self, y, x.clone() if residual else None)
            out = GP.glu_proj(y, W["W1"], W["W2"], W["W3"], residual=x if residual else None)
            ok = bool(torch.equal(out, ref)); _GLU_CHECKED[key] = ok
            tag = "rows=%d,c=%d,hid=%d%s%s" % (rows, C, W["HID"], "" if x.dtype == torch.bfloat16 else ",%s" % str(x.dtype).split(".")[-1], ",res" if residual else "")
            _count(lever, ("equal:" if ok else "differs:") + tag)
            if not ok:
                _log("glu_proj: the stock statement's bytes differ at %s (the cuBLAS GEMM torch selects there is not a one-pass kernel) -> stock serves this shape, by name" % tag)
                del out
                if residual:
                    x.copy_(ref); return x                                    # the block's own in-place statement on ITS tensor
                return ref
            del ref
        elif not ok:
            _fallback(lever, "differs:c=%d" % C); return _glu_proj_stock(self, y, x if residual else None)
        else:
            out = GP.glu_proj(y, W["W1"], W["W2"], W["W3"], residual=x if residual else None)
        _count(lever, "served:c=%d%s" % (C, "" if x.dtype == torch.bfloat16 else ",%s" % str(x.dtype).split(".")[-1]))
        if residual: _count(lever, "served:residual_c=%d" % C)
        return out
    except Exception as e:
        if is_oom(e) or _capturing(): raise
        _kernel_error(lever, e); return _plus(x, _ORIG["transition"](self, x), residual)


# =============================================================================================================================================
# lever 'apb' -- attention-with-pair-bias on the single track, served through the shared core's pair-bias attention PROVIDER by the mode's TIER WORD
# (opt_core.kernels.apb: one face over every carried implementation and its measured cell table; the word is _TIER's, fast | big -- big asks big):
#   (a) xfold.nn.diffusion_transformer.SelfAttention (PairformerBlock.single_attention_ [16 heads x 24, c 384: cell pf_h16d24] AND, when the fused diffusion
#       transformer is off, the 24 diffusion-transformer blocks [16 x 48, c 768: cell dit_h16d48]): stock ops for adaptive-LN / q,k,v / gate / AdaLNZero; the
#       attention core (stock: bf16 logits matmul + bias + masked_fill(-1e9) + fp32 softmax + matmul, then the sigmoid gate) is the provider's CORE op
#       (pair_bias_attention: q / k / v / gate as [1, N, H, Dh] views of the projection rows, the [1, H, N, N] bf16 bias with the key mask folded in as -1e9,
#       the gate fused where the row fuses it); the row per (card, dtype, cell, N bucket, eager | graph replay) is the provider's measured winner for the word --
#       its own Triton kernel (apb_attn), a carried package (fpf_apb, l3a, ...) or its stock SDPA row (sdpa:auto) where SDPA measured fastest. A process without
#       the provider, or a call class the word's row refuses by name, is served by the kit's own SDPA statement (torch.scaled_dot_product_attention with the
#       additive attn_mask = pair logits + -1e9 on masked keys) -- counted refused:selfattn:<row>:<kind> once per class, never silent.
#   (b) xfold.nn.pairformer.PairformerBlock: pair_logits = single_pair_logits_projection(single_pair_logits_norm(pair)) [N,N,128]->[N,N,16], head-major: the
#       provider's PRODUCER op (pair_bias_planes: LayerNorm + projection over the N^2 pair rows written head-plane-major in bf16; rows ln_proj / lnl_ln_linear /
#       fpf_pf_bias per cell bias_c128h16); refused by name or no provider -> the kit's fused LayerNorm + projection kernel (lnl_fused.ln_linear: one pass over
#       the bf16 pair tensor where stock materialises LN(pair) in fp32 and re-reads it); a kernel error -> the two stock statements.
#   The same arithmetic class as the stock statements under bf16 autocast (bf16 operands, fp32 statistics / accumulation / softmax), not the same bits:
#   fast / big only. exact and off run xfold's own statements by name: the provider's exact word names its stock SDPA floor on this stack (no row is
#   byte-vouched against the eager-softmax statement), so no exact-tier binding exists for this lever.
# =============================================================================================================================================
def set_graph_cells(flag):
    """The diffusion transformer's attention (cell dit_h16d48) consults the provider's graph-replay cells (True: the sampler captures the whole step, so the
    eager warm-up steps already launch the row the capture replays -- no row meets its first launch under capture) or its eager cells (False: hoist only, no
    capture). af3_torch_api.build_model sets it from the lever set (stepgraph); the Pairformer's cells (pf_h16d24) follow the stream's actual state."""
    _APB_PROV["graph_cells"] = bool(flag)
    _APB_PROV["sel"].clear()
    return _APB_PROV["graph_cells"]


def _apb_face():
    """The shared core's pair-bias attention provider (opt_core.kernels.apb), imported once; None when this process cannot import it (said once on stderr:
    the kit's own statements serve lever apb, by name)."""
    P = _APB_PROV
    if not P["tried"]:
        P["tried"] = True
        try:
            from opt_core.kernels import apb as KA                             # noqa: N811 -- standard library only at import; torch / triton inside its serving calls
            P["mod"] = KA
        except Exception as e:                                                 # noqa: BLE001 -- a core copy without the face: named, the kit statements serve
            P["mod"] = None; P["refused"] = "import:%s" % type(e).__name__
            _log("apb: the shared core's pair-bias attention provider is not importable (%r): the kit's SDPA statement and fused pair-logits kernel serve" % (e,))
    return P["mod"]


def provider_binding():
    """Lever apb's provider record (census 'arch': apb_provider): the tier word asked, the arm the provider served (or '<row>:<kind>' = refused by name, the
    kit statement serving) per call class 'op:cell:S<samples>:N<tokens>:<word>:<eager|graph>', the last refusal, and whether the face imported."""
    KA = _APB_PROV["mod"]
    rows = {}
    for key, sel in _APB_PROV["sel"].items():
        rows["%s:%s:S%d:N%d:%s:%s" % (key[0], key[1], key[2], key[3], key[-2], key[-1])] = sel if isinstance(sel, str) else (KA.arm_word(sel.row, sel.variant) if KA is not None else "?")
    return {"word": _TIER["word"], "face": (KA is not None) if _APB_PROV["tried"] else None, "graph_cells": _APB_PROV["graph_cells"], "rows": rows, "refused": _APB_PROV["refused"]}


def _apb_refused(op, key, word, r):
    """Record a by-name refusal of the provider for one call class: the kit statement serves that class for the rest of the process."""
    why = "%s:%s" % (getattr(r, "row", None) or word, str(getattr(r, "kind", r)).split(" ")[0][:60])
    _APB_PROV["sel"][key] = why; _APB_PROV["refused"] = "%s:%s" % (op, why)
    _count("apb", "refused:%s:%s" % (op, why))
    _log("apb: the provider's %s row refused call class %s (%s): the kit statement serves it" % (word, ":".join(str(k) for k in key), r))


def _apb_attention(q, k, v, g, am, N, H, Dh):
    """Lever apb's attention CORE through the provider by the mode's tier word. q / k / v / g: the [N, H*Dh] bf16 projection rows; am: the [1, H, N, N] bf16 bias
    with the key mask folded (-1e9). Returns the gated output rows [N, H*Dh] (sigmoid(g) * softmax(Dh^-0.5 q k^T + am) v in the served row's arithmetic), or None
    when the kit's SDPA statement serves the call: no provider in this process, no bias to carry, or the word's row refused this call class by name (counted
    once; the class stays with the kit statement). The provider's Selection is resolved once per call class (op, cell, samples, N, H, Dh, timing) and reused --
    no table walk per call; the first decision of each class is also the provider's own census record."""
    KA = _apb_face()
    if KA is None or am is None:
        return None
    cell = KA.cell_word("pf", H, Dh) or KA.cell_word("dit", H, Dh)             # pf_h16d24 | dit_h16d48 | None (a geometry no cell family lists: the provider names its stock row)
    cap = _capturing() or (cell == "dit_h16d48" and _APB_PROV["graph_cells"])  # the diffusion transformer's calls are replayed from the whole-step graph when the sampler captures
    word = _TIER["word"]
    key = ("selfattn", cell, 1, N, H, Dh, word, "graph" if cap else "eager")   # the call class (the word rides in it: a process that renames its tier re-resolves)
    sel = _APB_PROV["sel"].get(key)
    if isinstance(sel, str):
        return None
    q4, k4, v4, g4 = (t.unflatten(-1, (H, Dh))[None] for t in (q, k, v, g))   # [1, N, H, Dh] views of the rows: the provider's 'snhd' layout, no copies
    try:
        o, sel2 = KA.pair_bias_attention(q4, k4, v4, am, None, gate=g4, word=word, selection=sel, scale=Dh ** -0.5, layout="snhd", cell=cell, capture=cap)
    except KA.Refusal as r:
        _apb_refused("selfattn", key, word, r)
        return None
    if sel is None:
        _APB_PROV["sel"][key] = sel2
    _count("apb", "served:selfattn:c=%d:%s" % (H * Dh, KA.arm_word(sel2.row, sel2.variant)))
    return o.reshape(N, H * Dh)


def _selfattn_forward(self, x, mask, pair_logits=None, single_cond=None):
    lever = "apb"
    if lever in _DEAD or not x.is_cuda or not _autocast_bf16() or x.dim() != 2:
        if lever not in _DEAD:
            _fallback(lever, "selfattn:cpu/no-autocast/batched")
        return _ORIG["selfattn"](self, x, mask, pair_logits, single_cond)
    try:
        assert (single_cond is None) == (self.use_single_cond is False)
        H = self.num_head; Dh = self.qkv_dim; N = x.shape[0]
        x = self.adaptive_layernorm(x, single_cond)
        q = self.q_projection(x); k = self.k_projection(x); v = self.v_projection(x)             # [N, C] bf16
        g = self.gating_query(x)                                                                # [N, C] bf16 gate logits (the sigmoid is the attention row's epilogue, or applied below)
        am = None
        if pair_logits is not None:
            am = pair_logits[None] if pair_logits.dim() == 3 else pair_logits                        # [1, H, N, N]
            if am.dtype != torch.bfloat16:
                am = am.to(torch.bfloat16)
        if mask is not None:
            mterm = _mask_term(mask, H, N)
            am = mterm.expand(1, H, N, N) if am is None else am + mterm                             # one fused add -> contiguous [1,H,N,N]
        if am is not None and not am.is_contiguous():
            am = am.contiguous()
        o = _apb_attention(q, k, v, g, am, N, H, Dh)                                            # [N, C] gated, through the provider's row for the tier word; None: the kit statement below
        if o is None:                                                                           # the kit's own SDPA statement (no provider / no bias / the row refused this class by name)
            q, k, v = (t.unflatten(-1, (H, Dh)).transpose(0, 1)[None] for t in (q, k, v))       # [1, H, N, Dh] views
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=am, scale=Dh ** -0.5)          # [1, H, N, Dh]
            o = o[0].transpose(0, 1).reshape(N, H * Dh)
            o = o * torch.sigmoid(g)
            _count(lever, "served:selfattn:c=%d" % (H * Dh))
        return self.adaptive_zero_init(o, single_cond)
    except Exception as e:
        if is_oom(e) or _capturing(): raise                                # OOM propagates; no fallback applied (opt_core.oom.is_oom); under a CUDA-graph capture the error is the capture's (_capturing)
        _kernel_error(lever, e); return _ORIG["selfattn"](self, x, mask, pair_logits, single_cond)


def _apb_planes(block, c, pair, C, H):
    """The Pairformer's single-track pair logits through the provider's PRODUCER op by the tier word: [H, N, N] bf16 (LayerNorm + projection of the pair rows,
    head-major), or None when the kit's fused kernel serves (no provider, or the word's producer row refused this call class by name -- counted once)."""
    KA = _apb_face()
    if KA is None:
        return None
    N = pair.shape[0]; word = _TIER["word"]
    key = ("pairlogits", KA.cell_word("bias", heads=H, c_z=C), 1, N, H, C, word, "eager")
    sel = _APB_PROV["sel"].get(key)
    if isinstance(sel, str):
        return None
    if "apb_face" not in c:                                                    # the block's LayerNorm + projection parameters once, fp32 contiguous (every producer row packs / casts from these)
        assert block.single_pair_logits_projection.bias is None
        c["apb_face"] = {"lnw": block.single_pair_logits_norm.weight.detach().float().contiguous(), "lnb": block.single_pair_logits_norm.bias.detach().float().contiguous(),
                         "W": block.single_pair_logits_projection.weight.detach().float().contiguous(), "eps": block.single_pair_logits_norm.eps, "pack": {}}
    Wf = c["apb_face"]
    try:
        planes, sel2 = KA.pair_bias_planes(pair if pair.is_contiguous() else pair.contiguous(), Wf["lnw"], Wf["lnb"], Wf["W"], word=word, selection=sel, eps=Wf["eps"],
                                           out_layout="hij", out_dtype=torch.bfloat16, cache=Wf["pack"])
    except KA.Refusal as r:
        _apb_refused("pairlogits", key, word, r)
        return None
    if sel is None:
        _APB_PROV["sel"][key] = sel2
    if planes.dim() == 4:                                                      # a leading batch axis of one (rows that keep z's optional batch dim)
        planes = planes[0]
    if planes.shape[-1] != N:                                                  # rows that write [H, N, ld] planes (ld = N rounded up to 8): the N columns
        planes = planes[..., :N]
    if planes.dtype != torch.bfloat16:
        planes = planes.to(torch.bfloat16)
    _count("apb", "served:pairlogits:%s" % KA.arm_word(sel2.row, sel2.variant))
    return planes[:H]


def _pair_logits_fused(block, pair):
    """PairformerBlock single-track pair logits: the provider's producer row for the tier word, else the kit's fused LN + projection kernel. Returns
    [n_heads, N, N] bf16 (head planes; the attention adds the key-mask term into a fresh contiguous bias) or None (the caller runs the two stock statements)."""
    if "apb" in _DEAD or not pair.is_cuda or pair.dim() != 3 or not _autocast_bf16():
        return None
    C = pair.shape[-1]; H = block.single_pair_logits_projection.weight.shape[0]
    if C not in (64, 128) or H > 32:
        _fallback("apb", "pairlogits:c=%d,h=%d" % (C, H)); return None
    try:
        c = _cache(block)
        b16 = _apb_planes(block, c, pair, C, H)
        if b16 is not None:
            return b16
        import lnl_fused as RFU
        if "apb" not in c:
            NOUT = 16 if H <= 16 else 32
            Wb = torch.zeros((NOUT, C), device=pair.device, dtype=torch.bfloat16); Wb[:H] = block.single_pair_logits_projection.weight.detach().to(torch.bfloat16)
            assert block.single_pair_logits_projection.bias is None
            c["apb"] = dict(Wb=Wb, lnw=block.single_pair_logits_norm.weight.detach().float().contiguous(), lnb=block.single_pair_logits_norm.bias.detach().float().contiguous(),
                            eps=block.single_pair_logits_norm.eps, H=H)
        W = c["apb"]
        _, b16 = RFU.ln_linear(pair if pair.is_contiguous() else pair.contiguous(), W["lnw"], W["lnb"], W["Wb"], eps=W["eps"], write_y=False, planes=True)   # head-plane-major [NOUT, N, N]
        _count("apb", "served:pairlogits")
        return b16[:H]                                                     # [H, N, N], head planes contiguous (the attention's logits add reads unit strides)
    except Exception as e:
        if is_oom(e) or _capturing(): raise                                # OOM propagates; no fallback applied (opt_core.oom.is_oom); under a CUDA-graph capture the error is the capture's (_capturing)
        _kernel_error("apb", e); return None


def _pwa_msa_weights(m):
    c = _cache(m)
    if "pwa_msa" not in c:
        Wvg = torch.cat([m.v_projection.weight.detach(), m.gating_query.weight.detach()], 0).to(torch.bfloat16).contiguous()   # [H*value_dim + c_msa, c_msa] = [128, 64]
        c["pwa_msa"] = {"lnw": m.act_norm.weight.detach().float().contiguous(), "lnb": m.act_norm.bias.detach().float().contiguous(), "Wvg": Wvg, "eps": m.act_norm.eps, "NV": m.v_projection.weight.shape[0]}
    return c["pwa_msa"]


def _pwa_weights(m):
    c = _cache(m)
    if "pwa" not in c:
        H, C = m.num_head, m.c_pair
        Wb = torch.zeros(16, C, device=m.pair_logits.weight.device, dtype=torch.bfloat16)      # the logits projection [H, C] zero-padded to the kernel's 16 output planes
        Wb[:H] = m.pair_logits.weight.detach().to(torch.bfloat16)
        c["pwa"] = {"lnw": m.pair_norm.weight.detach().float().contiguous(), "lnb": m.pair_norm.bias.detach().float().contiguous(), "Wb": Wb, "eps": m.pair_norm.eps, "H": H}
    return c["pwa"]


# lever 'pwa_lnl' -- xfold.nn.attention.MSAAttention (the MSA module's pair-weighted averaging, 4 blocks per trunk pass): its pair statements
#   `pair = self.pair_norm(pair); logits = self.pair_logits(pair); logits = logits.permute(2, 0, 1)` ([N,N,128] LayerNorm -> [N,N,8] projection -> [8,N,N])
#   run as ONE kernel, the kit's fused LayerNorm + projection (kernels/third_party/lnl_fused.py ln_linear(planes=True)): LN in fp32, the normalised row rounded to
#   bf16, bf16 x bf16 products accumulated in fp32, one bf16 store per logit, written head-plane-major so the [8,N,N] logits are contiguous (no permuted view).
#   The same arithmetic CLASS as the two stock statements under bf16 autocast (fp32 LayerNorm, bf16 GEMM with fp32 accumulation) but not the same bits
#   (LayerNorm's reduction order, the GEMM's summation order): fast / big only. Every other statement of the module verbatim.
def _msaattn_forward(self, msa, msa_mask, pair):
    """xfold.nn.attention.MSAAttention.forward (the MSA module's pair-weighted averaging) with two independently switchable fused prologues; every other
    statement verbatim. lever pwa_lnl: `pair_norm(pair) -> pair_logits -> permute` as ONE LayerNorm+projection kernel writing the [H, N, N] logits head-plane-
    major. lever pwa_msa: `act_norm(msa) -> v_projection` and `-> gating_query` as ONE LayerNorm+projection kernel over the msa rows ([S, N, 64] -> [S, N, 128]:
    the value projection in columns 0..63, the gate logits in 64..127; LN in fp32, the normalised row rounded to bf16, bf16 x bf16 products accumulated in fp32,
    one bf16 store) -- the same arithmetic class as the stock statements under bf16 autocast, not the same bits (fast / big only). A lever that cannot serve
    (cpu, no bf16 autocast, another width) takes its stock statements by name (fallback:<reason>); a kernel error likewise (kernel_error), OOM propagates."""
    logits = None
    if "pwa_lnl" in _ON and "pwa_lnl" not in _DEAD:
        lever = "pwa_lnl"; C = pair.shape[-1]
        if not pair.is_cuda:
            _fallback(lever, "cpu")
        elif not _autocast_bf16():
            _fallback(lever, "no-bf16-autocast")
        elif C != 128 or self.num_head > 16 or pair.dim() != 3:
            _fallback(lever, "c=%d,h=%d,dim=%d" % (C, self.num_head, pair.dim()))
        else:
            try:
                import lnl_fused as RFU
                W = _pwa_weights(self)
                x = pair if pair.is_contiguous() else pair.contiguous()
                _, b16 = RFU.ln_linear(x, W["lnw"], W["lnb"], W["Wb"], eps=W["eps"], write_y=False, planes=True)   # [16, N, N] bf16 head planes
                logits = b16[:W["H"]]                                      # == pair_logits(pair_norm(pair)).permute(2, 0, 1) of the stock statements, contiguous
                _count(lever, "served:h=%d" % W["H"])
            except Exception as e:
                if is_oom(e) or _capturing(): raise                        # OOM propagates; no fallback applied (opt_core.oom.is_oom); under a CUDA-graph capture the error is the capture's (_capturing)
                _kernel_error(lever, e); logits = None
    v = gate_values = None
    if "pwa_msa" in _ON and "pwa_msa" not in _DEAD:
        lever = "pwa_msa"; Cm = msa.shape[-1]
        if not msa.is_cuda:
            _fallback(lever, "cpu")
        elif not _autocast_bf16():
            _fallback(lever, "no-bf16-autocast")
        elif Cm != 64 or msa.dim() != 3 or msa.dtype not in (torch.bfloat16, torch.float32):
            _fallback(lever, "c=%d,dim=%d,%s" % (Cm, msa.dim(), msa.dtype))
        else:
            try:
                import lnl_fused as RFU
                Wm = _pwa_msa_weights(self)
                xm = msa if msa.is_contiguous() else msa.contiguous()
                _, vg = RFU.ln_linear(xm, Wm["lnw"], Wm["lnb"], Wm["Wvg"], eps=Wm["eps"], write_y=False)   # [S, N, 128] bf16: v_projection(act_norm(msa)) | gating_query(act_norm(msa))
                v, gate_values = vg[..., :Wm["NV"]], vg[..., Wm["NV"]:]
                _count(lever, "served:rows=%d" % (xm.shape[0] * xm.shape[1]))
            except Exception as e:
                if is_oom(e) or _capturing(): raise
                _kernel_error(lever, e); v = gate_values = None
    if logits is None:                                                     # the stock statements, verbatim (xfold/nn/attention.py MSAAttention.forward)
        pair = self.pair_norm(pair)
        logits = self.pair_logits(pair)
        logits = logits.permute(2, 0, 1)
    if v is None:
        msa = self.act_norm(msa)
    logits += 1e9 * (torch.max(msa_mask, dim=0).values - 1.0)
    weights = torch.softmax(logits, dim=-1)
    if v is None:
        v = self.v_projection(msa)
    v = v.unflatten(-1, (self.num_head, self.value_dim))                   # 'b k (h c) -> b k h c'
    v_avg = torch.einsum('hqk, bkhc -> bqhc', weights, v)
    v_avg = torch.reshape(v_avg, v_avg.shape[:-2] + (-1,))
    if gate_values is None:
        gate_values = self.gating_query(msa)
    v_avg *= torch.sigmoid(gate_values)
    return self.output_projection(v_avg)


def _residual_call(lever, module, pair, *args):
    """One residual pair statement of the block, `pair + module(pair, ...)`: with the op's lever and lever resid_fold on, the add is folded into that lever's kernel epilogue (the patched
    forward takes residual=True and returns pair + update as a new tensor, or does the stock in-place add itself on a by-name fallback); with the lever off, the
    stock statement verbatim (`pair += module(pair, ...)`, the stock forward untouched). The caller rebinds pair (Evoformer / ConfidenceHead / template loops do)."""
    fold = "attn_epi" if lever == "triattn" else "resid_fold"        # the lever that owns this statement's fold: attn_epi (triattn's gate/projection/add epilogue) | resid_fold (trimul, transition)
    if lever in _ON and fold in _ON:                                # off -> the stock statement and its separate add, as before the fold existed
        return module(pair, *args, residual=True)
    if lever == "transition" and "glu_proj" in _ON and "transition" not in _ON:   # lever glu_proj restates GLU + projection + this add as one kernel (its own epilogue; by-name step-asides do the stock add)
        return module(pair, *args, residual=True)
    pair += module(pair, *args)
    return pair


def _pairformer_forward(self, pair, pair_mask, single=None, seq_mask=None):
    pair = _residual_call("trimul", self.triangle_multiplication_outgoing, pair, pair_mask)      # pair + TriMul_out(pair): the add in the kernel's K3 epilogue (lever trimul)
    pair = _residual_call("trimul", self.triangle_multiplication_incoming, pair, pair_mask)
    pair = _residual_call("triattn", self.pair_attention1, pair, pair_mask)                    # pair + TriAttn_start(pair): gate, out-projection and the add in one epilogue kernel (lever triattn)
    pair = _residual_call("triattn", self.pair_attention2, pair, pair_mask)                    # pair + TriAttn_end(pair): same, scattered into the untransposed frame
    pair = _residual_call("transition", self.pair_transition, pair)                            # pair + Transition(pair): the add in the fused kernel's epilogue (lever transition)
    if self.with_single is True:
        pair_logits = _pair_logits_fused(self, pair) if "apb" in _ON else None
        if pair_logits is None:
            pair_logits = self.single_pair_logits_projection(self.single_pair_logits_norm(pair))
            pair_logits = pair_logits.permute(2, 0, 1)
        single += self.single_attention_(single, seq_mask, pair_logits=pair_logits)
        single += self.single_transition(single)
        return pair, single
    return pair


# =============================================================================================================================================
# Per-process caches of the 'apb' lever: the additive key-mask term (-1e9 on masked keys), memoised per mask tensor (the same seq mask serves every
# block and every diffusion step). Entries hold a reference to their source tensor; clear_caches() drops them (the model process calls it before
# every item, so no item inherits another's cached state).
# =============================================================================================================================================
def _tkey(t):
    """Memo key of a source tensor for the mask-term cache (the tensor object itself, not its address). Cache entries also HOLD a reference to the tensor, so its storage cannot be freed and
    re-used at the same address while cached. In-place modification is detected through the version counter when the tensor has one (inference-mode
    tensors do not -> the cache assumes the conditioning tensors are not mutated in place during a trajectory, which holds for xfold's sampler;
    call clear_caches() otherwise)."""
    try:
        ver = t._version
    except Exception:
        ver = -1
    return (t.data_ptr(), ver, tuple(t.shape), t.dtype, str(t.device))


def clear_caches():
    """Drop the apb mask-term cache (called before every item and by disable())."""
    _MTERM.clear()


def _device_bytes(obj, seen):
    """Bytes of the distinct CUDA storages reachable from ``obj`` (tensors in dicts / lists / tuples), each storage counted once."""
    if torch.is_tensor(obj):
        if not obj.is_cuda:
            return 0
        st = obj.untyped_storage(); p = st.data_ptr()
        if p in seen:
            return 0
        seen.add(p); return int(st.nbytes())
    if isinstance(obj, dict):
        return sum(_device_bytes(v, seen) for v in list(obj.values()))
    if isinstance(obj, (list, tuple)):
        return sum(_device_bytes(v, seen) for v in obj)
    return 0


def release_face_workspaces(model):
    """big's stage boundary (lever diff_free, after the trunk): the TriMul serve state the trunk left is released -- (1) the native row's
    shared per-geometry payload caches on this device (opt_core.kernels.trimul.native.payload_cache: the a|b and contraction planes, cast rows,
    tensor maps and the LRU weight pack of every live (device, N, c_z, c_hidden) geometry; kept by the payload's policy while small: 4 x
    [N, N, 128]-class bf16 = 215 MiB at 448 tokens, transient above ~700), through the package's public surface only; (2) every TriMul module's
    provider-face dict (descriptors, the call memo).  None of it is read by the sampler or the heads; the next served call re-creates what it
    needs (first-call cost only).  Returns (MiB of device storage released, geometries emptied, modules emptied)."""
    seen = set(); freed = 0; n_geo = 0; n_mod = 0
    try:
        from opt_core.kernels.trimul import native as _TN                  # the package's __all__: shared_geometries(), payload_cache()
        dev = torch.cuda.current_device() if torch.cuda.is_available() else None
        for key in list(_TN.shared_geometries()):                          # [(device, N, c_z, c_hidden, workspace_bytes)]
            if dev is not None and key and key[0] != dev:
                continue
            d = _TN.payload_cache(key[0], key[1], key[2], key[3])
            if d:
                freed += _device_bytes(d, seen); d.clear(); n_geo += 1
    except (ImportError, AttributeError, TypeError, RuntimeError):          # no native package / another surface: nothing shared to release, by name in the record (geometries 0)
        pass
    for m in model.modules():
        c = getattr(m, "__dict__", {}).get("_af3k")                       # _cache(m)'s attribute, read without creating one
        f = c.get("trimul_face") if isinstance(c, dict) else None
        if f:
            freed += _device_bytes(f, seen); f.clear(); n_mod += 1
    return freed / float(1 << 20), n_geo, n_mod


_MTERM = {}


def _mask_term(mask, H, N):
    """additive key-mask term (-1e9 on masked keys, bf16 [1,1,1,N] or [B,1,1,N]); cached per mask tensor (same seq mask serves every block / diffusion step)."""
    key = _tkey(mask)
    ent = _MTERM.get(key)
    if ent is not None and ent[0] is mask:
        return ent[1]
    mterm = torch.zeros(mask.shape, device=mask.device, dtype=torch.bfloat16).masked_fill_(~(mask.to(torch.bool)), -1e9)
    mterm = mterm[None, None, None, :] if mask.dim() == 1 else mterm[:, None, None, :]
    if len(_MTERM) >= 8:
        _MTERM.pop(next(iter(_MTERM)))
    _MTERM[key] = (mask, mterm)
    return mterm

# =============================================================================================================================================
def _install():
    from xfold.nn.triangle_multiplication import TriangleMultiplication
    from xfold.nn.attention import GridSelfAttention, MSAAttention
    from xfold.nn.primitives import Transition
    from xfold.nn.diffusion_transformer import SelfAttention
    from xfold.nn.pairformer import PairformerBlock
    if not _ORIG:
        _ORIG["trimul"] = TriangleMultiplication.forward
        _ORIG["triattn"] = GridSelfAttention.forward
        _ORIG["msaattn"] = MSAAttention.forward
        _ORIG["transition"] = Transition.forward
        _ORIG["selfattn"] = SelfAttention.forward
        _ORIG["pairformer"] = PairformerBlock.forward
    TriangleMultiplication.forward = ((_trimul_forward_exact_aside if "trimul_exact" in _ON else _trimul_forward) if "trimul" in _ON
                                      else (_trimul_exact_forward if "trimul_exact" in _ON else _ORIG["trimul"]))   # lever trimul's tier word serves the class where both are on; trimul_exact steps aside by name per call
    GridSelfAttention.forward = _triattn_forward if "triattn" in _ON else _ORIG["triattn"]
    MSAAttention.forward = _msaattn_forward if _ON & {"pwa_lnl", "pwa_msa"} else _ORIG["msaattn"]
    Transition.forward = ((_transition_forward_glu_aside if "glu_proj" in _ON else _transition_forward) if "transition" in _ON
                          else (_glu_proj_forward if "glu_proj" in _ON else _ORIG["transition"]))   # lever transition (the provider's row, LN inside) serves where both are on; glu_proj steps aside by name per call
    SelfAttention.forward = _selfattn_forward if "apb" in _ON else _ORIG["selfattn"]
    PairformerBlock.forward = _pairformer_forward if _ON & {"apb", "resid_fold", "attn_epi", "glu_proj"} else _ORIG["pairformer"]   # apb: fused pair logits; resid_fold: the trimul / transition residual adds folded; attn_epi: triattn's gate/projection/add epilogue (_residual_call)


_ARCH = {"cc": None, "notices": []}


def _apply_arch_cells():
    """Per-architecture bindings, keyed by torch.cuda.get_device_capability(): the shared core's triangle-attention / TriMul / transition / pair-bias attention providers (the mode's tier word per cell); no cell table of the kit's.
    Printed once per process. ln_linear / glu_proj launch ONE fixed tile each -- no candidate timed."""
    if not torch.cuda.is_available():
        return
    cc = "%d.%d" % torch.cuda.get_device_capability()
    sel = (cc, "triattn" in _ON, "attn_epi" in _ON, "transition" in _ON)   # the selection depends on the card and on levers triattn / attn_epi / transition (a provider binds only with its lever on)
    if _ARCH["cc"] == cc and _ARCH.get("sel") == sel:
        return
    _ARCH["cc"] = cc; _ARCH["sel"] = sel; notes = []
    # lever triattn: bind the shared core's triangle-attention provider; the MODE's tier word names the row per cell (no kit row word, tile or size table)
    _TRIATTN_PROV.update(mod=None, stack=None, refused=None); _TRIATTN_PROV["sel"] = {}
    if "triattn" in _ON:
        try:
            from opt_core.kernels import triattn as TP                           # the provider face: pure selection + one serving call; its Triton rows come through the core's carried-kernel route
            _TRIATTN_PROV["mod"] = TP
            dev_cc = tuple(torch.cuda.get_device_capability())
            if dev_cc == (9, 0):                                                 # the stack key the provider's prebuilt rows are keyed by on this card (the provider's own modules name it; torch is imported: nothing loads)
                from opt_core.kernels.triattn import cuda_sm90a as _TPC
                _TRIATTN_PROV["stack"] = _TPC.stack_key()
            elif dev_cc == (8, 0):
                from opt_core.kernels.triattn import triattn_native as _TPK
                _TRIATTN_PROV["stack"] = _TPK.stack_key()
            w = _triattn_word()
            parts = []
            for D, N in ((32, 448), (32, 832), (16, 448)):                       # the activation notice: the rows the word resolves to at the pair stacks' head dim 32 and the template stack's 16
                s = _triattn_select(N, 4, D)
                parts.append("d%d@%d=%s" % (D, N, s.row if s is not None else "stock(%s)" % _TRIATTN_PROV["refused"]))
            notes.append("triattn: opt_core.kernels.triattn word=%s form=keypad for cc %s: %s [stack %s]" % (w, cc, " ".join(parts), _TRIATTN_PROV["stack"]))
        except Exception as e:                                                   # no provider in this core copy: lever triattn keeps the stock statement, by name (counted per call)
            if is_oom(e): raise
            _TRIATTN_PROV.update(mod=None, refused="provider:%s" % type(e).__name__)
            notes.append("triattn: opt_core.kernels.triattn unavailable (%s: %s) -> the stock statement serves, by name" % (type(e).__name__, e))
    _ARCH["triattn_word"] = _triattn_word() if _TRIATTN_PROV["mod"] is not None else None; _ARCH["triattn_refused"] = _TRIATTN_PROV["refused"]; _ARCH["triattn_stack"] = _TRIATTN_PROV["stack"]
    notes.append("apb pair-logits / pwa: ln_linear launches its one fixed tile (lnl_fused.ln_linear) -- no candidate tile timed")
    _ARCH["notices"] = notes
    _TRIMUL_PROV.update(mod=None, stack=None, refused=None, has_cueq=None, why=None); _TRIMUL_PROV["sel"].clear(); _TRIMUL_PROV["cache"].clear()
    if _ON & {"trimul", "tmpl_trimul", "trimul_exact"}:
        try:
            from opt_core.kernels import trimul as TPm                        # the shared core's TriMul provider: ONE face over every carried row, the measured cell per
            _TRIMUL_PROV["mod"] = TPm; _TRIMUL_PROV["stack"] = TPm.stack_word()   #   (cc, precision, c_z, c_hidden, size, direction) on this process's stack; tier words and the exact word's module form
            _TRIMUL_PROV["has_cueq"] = not str(_TRIMUL_PROV["stack"]).endswith("nocueq")
            for lever_, word_, C_ in (("trimul", _TIER["word"], 128), ("tmpl_trimul", _TIER["word"], 64), ("trimul_exact", "exact", 128), ("trimul_exact", "exact", 64)):
                if lever_ not in _ON or (lever_ == "tmpl_trimul" and "trimul" not in _ON):
                    continue
                rows = []
                for n in (448, 832, 1216):                                     # the binding at the padded ladder sizes, said once per process (the census counts what each call served)
                    try:
                        s = TPm.select(cc, "bf16", C_, C_, n, "outgoing", word=word_, stack=_TRIMUL_PROV["stack"], has_cueq=_TRIMUL_PROV["has_cueq"], form=(_EXACT_FORM if word_ == "exact" else None))
                        rows.append("%d:%s" % (n, s.row))
                    except TPm.Refusal as r:
                        _TRIMUL_PROV["refused"] = "%s:%s" % (r.row or word_, str(r.kind).split("(")[0])
                        rows.append("%d:refused(%s)" % (n, _TRIMUL_PROV["refused"]))
                notes.append("%s: c%d word %s for cc %s = opt_core.kernels.trimul rows by size %s [stack %s] -- every row through the provider face; a stock row or a refusal = the module statement, by name"
                             % (lever_, C_, word_ + ("+" + _EXACT_FORM if word_ == "exact" else ""), cc, ", ".join(rows), _TRIMUL_PROV["stack"]))
        except Exception as e:
            _TRIMUL_PROV["mod"] = None; _TRIMUL_PROV["why"] = repr(e)[:160]
            notes.append("trimul: opt_core.kernels.trimul unavailable (%r) -> every triangle multiplication keeps the module statement, by name (fallback no-provider)" % (e,))
    _ARCH["trimul_word"] = _TIER["word"]; _ARCH["trimul_refused"] = _TRIMUL_PROV["refused"]; _ARCH["trimul_stack"] = _TRIMUL_PROV["stack"]
    _GLU_CARD_OFF_C.clear(); _GLU_CARD_OFF_C.update(_GLU_CARD_OFF.get(cc, ()))          # lever glu_proj: widths this card serves by the stock statement, by name
    if _GLU_CARD_OFF_C and "glu_proj" in _ON:
        notes.append("glu_proj: c in %s served by the stock statement on cc %s by name (the restatement's bytes differ from the stock GEMM's there)" % (sorted(_GLU_CARD_OFF_C), cc))
    _TRANS_PROV.update(mod=None, cc=cc, stack=None, refused=None); _TRANS_PROV["sel"].clear()
    if "transition" in _ON:                                                  # lever transition: the shared core's transition provider asked by the MODE's tier word (_TIER) per call class; the
        try:                                                                 #   ladder's classes resolved here once for the start-up line (pair c=128, MSA c=64x4, template c=64x2, single c=384 at
            from opt_core.kernels import transition as TPt                   #   448 / 832 / 1216 tokens): a kernel row, a stock row (the engine's own module measured fastest), or a refusal by name
            assert callable(TPt.transition) and callable(TPt.pack) and callable(TPt.select) and TPt.STOCK_ROWS
            _TRANS_PROV["mod"] = TPt
            try:
                _TRANS_PROV["stack"] = TPt.stack_word()
            except Exception:                                                # noqa: BLE001 -- the stack word is advisory (the face resolves it itself when None)
                _TRANS_PROV["stack"] = None
            word_t = _TIER["word"]; cells_t = []
            for (c_, h_, fam_) in ((128, 512, "pair"), (64, 256, "rows"), (64, 128, "rows"), (384, 1536, "single")):
                per = []
                for n_ in (448, 832, 1216):
                    try:
                        ts = TPt.select(word_t, c=c_, hidden=h_, n_tokens=n_, dtype="bf16", family=fam_, residual=("resid_fold" in _ON and c_ != 384), cc=cc, stack=_TRANS_PROV["stack"])
                        per.append("%d:%s" % (n_, ("stock(%s)" % ts.row) if ts.row in TPt.STOCK_ROWS else ts.row))
                    except TPt.Refusal as r:
                        per.append("%d:REFUSED(%s)" % (n_, str(r.kind).split(":")[0])); _TRANS_PROV["refused"] = "%s:%s" % (word_t, r.kind)
                cells_t.append("c%dx%d %s" % (c_, h_ // c_, " ".join(per)))
            notes.append("transition: tier word %s for cc %s = opt_core.kernels.transition rows per cell [%s] (stack %s) -- every row through the provider face with its measured launch word; "
                         "stock(...) = the engine's own module measured fastest there; REFUSED(...) = no cell by name -> the stock statement, counted" % (word_t, cc, "; ".join(cells_t), _TRANS_PROV["stack"]))
        except Exception as e:                                               # noqa: BLE001
            _TRANS_PROV["mod"] = None
            notes.append("transition: opt_core.kernels.transition unavailable (%r) -> the stock statement serves, counted fallback:no-provider" % (e,))
    _ARCH["transition_word"] = _TIER["word"] if "transition" in _ON else None; _ARCH["transition_row"] = ("provider" if _TRANS_PROV["mod"] is not None else "stock") if "transition" in _ON else None
    _ARCH["transition_refused"] = _TRANS_PROV["refused"]; _ARCH["transition_stack"] = _TRANS_PROV["stack"]
    _TRIATTN_EPI.clear()                                                     # the fused epilogue cell: the shared core's pair-fused table row for (c_out, H, D) on this device (opt_core.attn.pair_fused lookup_cell)
    if "attn_epi" not in _ON:
        pass                                                                 # lever attn_epi off: gate_transpose + linear + the block's own add (no cell resolved)
    else:
        try:
            from fpf_triatt_epi.epilogue import triatt_epilogue              # routed (registry.KERNEL_ROUTES); its cells live in the core's pair_fused table, not in the package
            from opt_core.attn import pair_fused as PF
            assert callable(triatt_epilogue)
            d = PF.lookup_cell("fpf", "epilogue", (128, 4, 32), torch.device("cuda"), variant="v2")   # the trunk / MSA-module / confidence pair stacks' cell; the template stack (64, 4, 16) keeps gate_transpose + linear
            if d.row is not None and d.served_by not in ("safe", "default"):           # a tuned row of the table serves (the core's own admission rule: pair_fused.lookup_cell)
                _TRIATTN_EPI[(128, 4, 32)] = dict(d.row["cfg"])
                notes.append("triattn: epilogue for cc %s = fpf_triatt_epi %s row %s %s [%s]" % (cc, d.row.get("variant"), d.row.get("id"), d.row["cfg"], str(d.row.get("evidence", ""))[:80]))
            else:
                notes.append("triattn: no tuned fpf_triatt_epi cell for (128, 4, 32) on cc %s (%s) -> gate_transpose + linear + add" % (cc, d.reason if d.row is None else d.word()))
        except Exception as e:
            notes.append("triattn: fpf_triatt_epi unavailable (%r) -> gate_transpose + linear + add -- FALLBACK EPILOGUE" % (e,))
    for n in notes:
        _log("arch %s: %s" % (cc, n))


def arch_info():
    return dict(_ARCH)


def enable(levers=LEVERS):
    """Enable the given levers (list of names or 'all'); replaces the current set. Returns the active set."""
    if isinstance(levers, str):
        levers = LEVERS if levers == "all" else [p for p in levers.replace(",", "+").split("+") if p]
    bad = [l for l in levers if l not in LEVERS]
    if bad:
        raise ValueError("unknown levers %s; known: %s" % (bad, LEVERS))
    _ON.clear(); _ON.update(levers)
    _install()
    try:
        _apply_arch_cells()
    except Exception as e:
        _log("arch cell selection failed (%r); built-in defaults in use" % (e,))
    return sorted(_ON)


def disable():
    _ON.clear()
    if _ORIG:
        _install()
    clear_caches()
    return []


def active():
    return sorted(_ON)
