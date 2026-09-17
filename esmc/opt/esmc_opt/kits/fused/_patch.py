"""fused / _patch.py — the fused EAGER forward patch for ESM C: NO new CUDA. The same kernels in the same order with fewer launches, so the
outputs are bitwise the stock's by construction. Serves the flash-attn-constructed model and the SDPA-constructed model (use_flash_attn=False)
on EVERY call through the same seams on the [b*L, d] rows (pads included, as the stock computes them) with the stock's own attention dispatch and
key-mask rule — no padding: views + cu_seqlens on device; padding: the stock's own unpad_input / pad_input ONCE per forward; collected hidden
states written once into a preallocated fp32 slab (copies unpadded; ONE zero fill + a scatter per state on the padded route); the optional merged
seams (out_proj_residual, the deferred residual+LN) when a composed item supplies them.

What the stock SDK route does per forward at batch 1 (esm at the pinned commit, the pinned stack: TE LayerNormLinear / LayerNormMLP,
flash_attn varlen, Triton rotary, the wrapper's bf16 autocast), and what this patch changes — every item read from the package's bytes:
  (a) VARLEN METADATA PER LAYER: `seqlens = seq_id.sum(-1)`, `cu_seqlens = F.pad(cumsum(seqlens))`, `max_seqlen = int(seqlens.max().item())`
      = 4 launches + ONE HOST SYNC in every one of the 80 layers (layers.py EsmcFlashMultiHeadAttention.forward).  Here: computed ONCE per
      forward; at batch 1 without padding the values are known from the shape (cu = [0, L], max = L): no launch, no sync.
  (b) AUTOCAST CASTS: the wrapper's bf16 autocast re-casts q_ln/k_ln's INPUT and WEIGHT to fp32 on every call (grad is off, so
      autocast's weight-cast cache is not used) and the LN output back to bf16 = 4 launches per LN, 8 per layer; the final norm 3.
      Here: the fast path runs with autocast DISABLED and makes the dtype decisions explicit: the fp32 weight copies are made ONCE at
      engage (an exact upcast — the same values autocast produces per call); the input upcast + the fp32 LayerNorm + the bf16
      round-trip are the stock's own ops in the stock's order (3 launches per LN).
  (c) THE STACK: `torch.stack([q, k, v], dim=1).view(T, 3, H, D)` copies the whole QKV tensor once per layer.  Here: the LN'd q and k are
      written back into their own slots of the QKV GEMM output (`qkv.view(T, 3, D)[:, 0/1].copy_(...)` = the stock's `.to(bf16)` round
      and the stack's copy as ONE copy; RNE both), and `qkv.view(T, 3, H, D)` IS the packed layout — no stack.
  (d) UNPAD / PAD AT BATCH 1: `unpad_input` (sum, nonzero + sync, cumsum, pad, max + sync, gather) and `pad_input` (zeros + index_put) on
      the last hidden state, the prenorm state and EVERY collected hidden state (81 in the full class) are equality gathers when no
      token is padding.  Here: views.  One host read per forward decides it (`(input_ids != pad).all().item()` — the stock's own mask
      rule; the stock itself syncs at the same point inside unpad_input).
  (e) THE WRAPPER: `ESMC.logits` always runs `output_hidden_states=True` and builds the full hidden-states stack; here hidden states are computed
      on demand (`config.return_hidden_states`) — the same LogitsOutput fields, types and dtypes.
Everything else (TE modules, the Triton rotary, flash_attn varlen, the residual div/add, the LM head under the caller's autocast) is the
stock's code path, untouched. Batch > 1, padding, sequence_id, output_attentions, SAE, grad enabled or training mode -> the STOCK
forward (with the per-forward metadata hoist of (a), bitwise by the same argument).

Contract (what esmc_opt.kits.fused calls): engage(model) -> dict(engaged=[...], pins, engage_s, patch_sha256, esm_commit);
disengage(model) -> a full restore of every class attribute the patch replaced (idempotent both ways).
"""
from __future__ import annotations

import hashlib; from esmc_opt._oom import is_oom
import inspect
import os
import time

PATCH = "fused/_patch"
ESM_COMMIT = "43ccece2ad485f27db46afdb67da2a9601e8f106"
# sha256 of the installed module files this patch was written against (Biohub/esm at ESM_COMMIT); engage refuses on drift.
FILE_PINS = {
    "esm.models.esmc.layers": "5a850fa105a40de514122d46a00cf8a5ec84094baeba012b0283bebb3ccb0a92",
    "esm.models.esmc.model": "3d3a15281d84ee491c94503855abadc168afa0a526b0f9b4ca909c03354ab8a5",
    "esm.models.esmc.compatibility": "82107ae077e5c58fd00d919341848376c7d92bfd1b46470fec25b37380849dd6",
}
COUNTERS = {"model_forwards": 0, "hs_slab_forwards": 0, "s_route_attn": 0, "s_route_padded_to_stock": 0, "s_route_padded_fast": 0, "s_route_thin_off_calls": 0, "fast_path": 0, "stock_path": 0, "attn_forwards": 0, "cache_hits": 0, "cache_misses": 0,
            "stack_forwards": 0, "logits_calls": 0, "hidden_requested": 0, "weight_recast": 0, "rotary_single_launch": 0, "fast_unpadded": 0, "fast_padded": 0}
_TRITON_ROTARY_FORWARD = None
# THE SEAMS (one call site per kernel class on the fast path; a composed item may swap an entry for a thinner launch of the SAME kernel —
# e.g. thin_launch.py's TE functional calls / flash direct binding / cached Triton launch — without touching the path's structure).
# Every entry takes the module (or None) + the tensors and returns what the stock's call returns; defaults = the stock's own calls.
_OPS = {
    "ln_qkv":   lambda mod, x: mod(x),                                                                    # TE LayerNormLinear (or the pure fallback)
    "qk_ln":    None,                                                                                     # set at engage: (ln_module, x_bf16_view) -> fp32 LN of x.float() with the cached fp32 weights
    "rotary":   None,                                                                                     # set at engage: (rot_module, qkv_packed, cu, max) -> in place
    "attn":     None,                                                                                     # set at engage: (qkv_packed, cu, max, scale) -> context
    "out_proj": lambda mod, ctx: mod(ctx),                                                                # TE Linear (or nn.Linear)
    "ffn":      lambda mod, x: mod(x),                                                                    # TE LayerNormMLP (or the pure fallback): [T, D] -> [T, D]
    "residual": lambda x, y, sf: x + y / sf,                                                              # the block's `x + y / scaling_factor` (bf16 div then add: two launches)
    # OPTIONAL merged seams (absent by default): when present they replace a PAIR/SEQUENCE of the entries above —
    #   "out_proj_residual": (mod_out_proj, ctx2d, x_res, sf) -> x_new  = residual(x_res, out_proj(mod, ctx2d), sf) in one item call (the item
    #        runs the stock pair itself below its engagement rows); the block's attention-side residual is then skipped.
    #   "residual_ln_qkv": (x, y, sf, mod) -> (x_new, qkv)  = residual(x, y, sf) followed by ln_qkv(mod, x_new) for the FFN-side residual of
    #        block i feeding block i+1's QKV LayerNorm (the deferred form below; seq_only forwards only — with hidden states collected every
    #        block resolves its residual eagerly); the attention-side residual stays "residual".
    #   "qk_ln_rotary": (rot, ln_q, ln_k, qkv_packed, cu_seqlens, max_seqlen) -> None = qk_ln x2 + the two copy_ into the QKV slots + rotary,
    #        IN PLACE on qkv_packed (slots 0 and 1).
    # No cache keyed on data_ptr/version may be shared ACROSS two seams (refused by name).
}
OPTIONAL_SEAMS = ("residual_ln_qkv", "qk_ln_rotary", "out_proj_residual")
HS_SLAB_MIN_ROWS = 2048   # (below 2048 rows the per-layer copies cost more CPU than the stack on launch-bound calls -> the stock's stack form there)
                          # with hidden states collected on the fast path, every collected state is written ONCE into a preallocated fp32
                          # [n, T, D] slab (unpadded: bf16 -> fp32 copies, exact) or, on the padded route, SCATTERED with the forward's own `indices` into
                          # a zero-filled fp32 [n, B*L, D] slab (one fill + one scatter per state = the stock's pad_input + stack, exact) — the slab IS
                          # the returned hidden_states tensor. No stack, no per-state pad_input. Always on (the stack already cost 81 copies).


_state = {"engaged": False, "orig": {}, "ln32": {}, "arange2": {}, "cu": {}, "pins": None, "reason": None, "last": None, "last_block_ids": set(), "norms": []}   # last_block_ids / norms: one entry per model engaged (every model built under the mode gets the instance patch and its last block registered)
_CACHE = {"mask": None, "cu": None, "max": None, "fast": False, "qkv": None, "pending": None, "defer_ok": False}


class PatchRefused(RuntimeError):
    pass


def counters_snapshot() -> dict:
    return dict(COUNTERS)


def counters_delta(before: dict) -> dict:
    return {k: COUNTERS[k] - before.get(k, 0) for k in COUNTERS}


def _sha_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def pins() -> dict:
    """File shas of the three esm modules + source shas of the four functions the patch replaces (read, never trusted labels)."""
    import esm.models.esmc.compatibility as C
    import esm.models.esmc.layers as L
    import esm.models.esmc.model as M
    out = {"files": {}, "functions": {}}
    for name, mod in (("esm.models.esmc.layers", L), ("esm.models.esmc.model", M), ("esm.models.esmc.compatibility", C)):
        out["files"][name] = {"path": mod.__file__, "sha256": _sha_file(mod.__file__), "pinned": FILE_PINS[name]}
    for label, fn in (("EsmcModel.forward", M.EsmcModel.forward), ("EsmcFlashMultiHeadAttention.forward", L.EsmcFlashMultiHeadAttention.forward),
                      ("EsmcTransformerStack.forward", L.EsmcTransformerStack.forward), ("EsmcUnifiedTransformerBlock.forward", L.EsmcUnifiedTransformerBlock.forward),
                      ("ESMC.logits", C.ESMC.logits)):
        try:
            out["functions"][label] = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
        except Exception as ex:  # noqa: BLE001 — a patched function has no file source: stated
            out["functions"][label] = f"unavailable: {type(ex).__name__}"
    out["drift"] = [n for n, v in out["files"].items() if v["sha256"] != v["pinned"]]
    return out


# ----------------------------------------------------------------------------------------------------------------------------------
# the fast path helpers
# ----------------------------------------------------------------------------------------------------------------------------------
def _cu_seqlens(L: int, device, b: int = 1):
    """[0, L, 2L, ..., bL] int32 on device = the stock's F.pad(cumsum(seqlens)) for b equal-length sequences without padding (b = 1: [0, L]);
    built on device (no host copy), cached per (device, b, L)."""
    import torch
    key = (str(device), int(b), int(L))
    t = _state["cu"].get(key)
    if t is None:
        a = _state["arange2"].get((str(device), int(b)))
        if a is None:
            a = torch.arange(int(b) + 1, dtype=torch.int32, device=device)
            _state["arange2"][(str(device), int(b))] = a
        t = a * int(L)                                                             # int32 multiply on device: exact
        _state["cu"][key] = t
    return t


class _LN32:
    """fp32 copies of a LayerNorm's weight (and bias) made once; re-made when the parameter's version counter or storage moves (the guard)."""
    __slots__ = ("w", "b", "wv", "bv", "wp", "bp")

    def __init__(self, weight, bias):
        self.refresh(weight, bias)

    def refresh(self, weight, bias):
        self.w = weight.detach().float()
        self.wv, self.wp = weight._version, weight.data_ptr()
        if bias is not None:
            self.b = bias.detach().float()
            self.bv, self.bp = bias._version, bias.data_ptr()
        else:
            self.b, self.bv, self.bp = None, None, None

    def get(self, weight, bias):
        if weight._version != self.wv or weight.data_ptr() != self.wp or (bias is not None and (bias._version != self.bv or bias.data_ptr() != self.bp)):
            COUNTERS["weight_recast"] += 1
            self.refresh(weight, bias)
        return self.w, self.b


def _ln32(module):
    st = _state["ln32"].get(id(module))
    if st is None:
        st = _LN32(module.weight, module.bias)
        _state["ln32"][id(module)] = st
    return st.get(module.weight, module.bias)


def _install_default_ops(flash_fn, rotary_fn):
    """The default entries for the seams = the stock's own calls, minus the launches this patch removes (see the module docstring)."""
    import torch
    import torch.nn.functional as F

    def qk_ln(ln, x_view):
        # the stock under autocast: x -> fp32 (a contiguous copy), weight -> fp32 (per call; here cached), LayerNorm(fp32) -> fp32; the caller's
        # copy_ into the QKV slot is the stock's `.to(bf16)` (RNE) and the stack's copy as ONE copy.
        w, b = _ln32(ln)
        return F.layer_norm(x_view.float(), (x_view.shape[-1],), w, b, ln.eps)

    def rotary(rot, qkv_packed, cu_seqlens, max_seqlen):
        if rotary_fn is not None and type(rot).forward is _TRITON_ROTARY_FORWARD:
            # the stock's EsmcTritonRotaryEmbedding.forward = the SAME Triton kernel launched twice (qkv[:, 0], then qkv[:, 1]);
            # the q and k slots are adjacent in the packed layout, so ONE launch over the [T, 2*H, D] view rotates both — per
            # (token, head) the kernel's arithmetic is unchanged (the rotation depends on the position only): bitwise, one launch.
            T, _, H, D = qkv_packed.shape
            rot._update_cos_sin_cache(max_seqlen, device=qkv_packed.device, dtype=qkv_packed.dtype)
            rotary_fn(qkv_packed[:, :2].view(T, 2 * H, D), rot._cos_cached, rot._sin_cached, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, inplace=True)
            COUNTERS["rotary_single_launch"] += 1
        else:
            rot(qkv_packed, cu_seqlens, max_seqlen)

    def attn(qkv_packed, cu_seqlens, max_seqlen, scale):
        return flash_fn(qkv_packed, cu_seqlens, max_seqlen, softmax_scale=scale)

    _OPS.update({"qk_ln": qk_ln, "rotary": rotary, "attn": attn, "ln_qkv": lambda mod, x: mod(x), "out_proj": lambda mod, ctx: mod(ctx),
                 "ffn": lambda mod, x: mod(x), "residual": lambda x, y, sf: x + y / sf})


# ----------------------------------------------------------------------------------------------------------------------------------
# the patched forwards
# ----------------------------------------------------------------------------------------------------------------------------------
def _make_attn_forward(orig, flash_fn, rotary_fn=None, sdpa_class=False):
    import torch
    import torch.nn.functional as F

    def forward(self, x, seq_id, output_attentions=False):
        COUNTERS["attn_forwards"] += 1
        if sdpa_class and not (_CACHE["fast"] and _CACHE.get("s_route") and seq_id is _CACHE["mask"]):
            return orig(self, x, seq_id, output_attentions)                          # the SDPA class outside the S fast path = its own stock forward (3-D x, no varlen)
        if output_attentions:
            raise ValueError("output_attentions=True is not supported with attn_implementation='flash_attention_2'. "
                             "Re-load the model with attn_implementation='sdpa' (or 'eager').")
        assert seq_id is not None and seq_id.dtype == torch.bool
        hit = seq_id is _CACHE["mask"]
        if hit:
            cu_seqlens, max_seqlen = _CACHE["cu"], _CACHE["max"]
            COUNTERS["cache_hits"] += 1
        else:                                                                       # a caller outside the stack: the stock computation
            seqlens = seq_id.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
            max_seqlen = int(seqlens.max().item())
            COUNTERS["cache_misses"] += 1
        if hit and _CACHE["fast"] and isinstance(self.q_ln, torch.nn.LayerNorm):
            qkv = _CACHE["qkv"]
            if qkv is None:
                qkv = _OPS["ln_qkv"](self.layernorm_qkv, x)
            else:                                                                   # precomputed by the merged residual_ln_qkv seam (block i-1)
                _CACHE["qkv"] = None
            T = qkv.shape[0]
            qkv_packed = qkv.view(T, 3, self.n_heads, self.d_head)
            merged = _OPS.get("qk_ln_rotary")
            if merged is not None:
                merged(self.rotary, self.q_ln, self.k_ln, qkv_packed, cu_seqlens, max_seqlen)
            else:
                # autocast is OFF here; the stock's q_ln/k_ln chain under autocast = input->fp32, weight->fp32, LayerNorm(fp32),
                # ->bf16; then stack copies q, k, v into the packed buffer. Here: the same fp32 LayerNorm on the same fp32 inputs with the
                # SAME fp32 weight values (cast once at engage), rounded to bf16 by ONE copy_ into the q / k slot of the QKV GEMM output.
                qkv3 = qkv.view(T, 3, self.d_model)
                qkv3[:, 0].copy_(_OPS["qk_ln"](self.q_ln, qkv3[:, 0]))
                qkv3[:, 1].copy_(_OPS["qk_ln"](self.k_ln, qkv3[:, 1]))
                _OPS["rotary"](self.rotary, qkv_packed, cu_seqlens, max_seqlen)
            if _CACHE.get("s_route"):
                # the SDPA-constructed model (use_flash_attn=False) on an UNPADDED call: the stock's own attention dispatch
                # (esmc_scaled_dot_product_attention: xformers -> flash_attn_func -> F.sdpa, seq_id None) on the stock's [b, s, d] views
                from esm.models.esmc.layers import esmc_scaled_dot_product_attention as _sdpa
                b_, L_ = _CACHE["pad_shape"]
                q3 = qkv_packed[:, 0].reshape(b_, L_, self.n_heads * self.d_head)
                k3 = qkv_packed[:, 1].reshape(b_, L_, self.n_heads * self.d_head)
                v3 = qkv_packed[:, 2].reshape(b_, L_, self.n_heads * self.d_head)
                context = _sdpa(q3, k3, v3, n_heads=self.n_heads, d_head=self.d_head, seq_id=_CACHE.get("s_mask")).reshape(T, self.n_heads, self.d_head)   # the stock's own key mask on padded calls
                COUNTERS["s_route_attn"] += 1
            else:
                context = _OPS["attn"](qkv_packed, cu_seqlens, max_seqlen, self.d_head**-0.5)
            n_out, h_out, d_out = context.shape
            merged_out = _OPS.get("out_proj_residual")
            if merged_out is not None and _CACHE.get("resid_in") is not None:      # the OPTIONAL merged seam: out_proj + residual in one item call
                x_res, sf = _CACHE["resid_in"]
                _CACHE["resid_in"] = None
                _CACHE["resid_consumed"] = True
                return (merged_out(self.out_proj, context.reshape(n_out, h_out * d_out), x_res, sf), None)
            return (_OPS["out_proj"](self.out_proj, context.reshape(n_out, h_out * d_out)), None)
        # the stock lines (a cache miss or a non-fast forward)
        qkv = self.layernorm_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = self.q_ln(q).to(q.dtype)
        k = self.k_ln(k).to(q.dtype)
        T = q.shape[0]
        qkv_packed = torch.stack([q, k, v], dim=1).view(T, 3, self.n_heads, self.d_head)
        qkv_packed = self.rotary(qkv_packed, cu_seqlens, max_seqlen)
        context = flash_fn(qkv_packed, cu_seqlens, max_seqlen, softmax_scale=self.d_head**-0.5)
        n_out, h_out, d_out = context.shape
        return (self.out_proj(context.reshape(n_out, h_out * d_out)), None)
    forward.__wrapped_stock__ = orig
    return forward


def _make_block_forward(orig):
    """EsmcUnifiedTransformerBlock.forward: on the fast path the same two residual updates through the seams; otherwise the stock."""

    def forward(self, x, sequence_id, output_attentions=False):
        if not (_CACHE["fast"] and sequence_id is _CACHE["mask"]):
            return orig(self, x, sequence_id, output_attentions)
        pend = _CACHE["pending"]
        if pend is not None:                                                        # block i-1 deferred its FFN residual: resolve it fused with THIS block's QKV LayerNorm
            _CACHE["pending"] = None
            x, _CACHE["qkv"] = _OPS["residual_ln_qkv"](x, pend[0], pend[1], self.attn.layernorm_qkv)
        row = _CACHE.get("collect_row")
        if row is not None:                                                         # this block's INPUT (after the merge) is a collected hidden state
            _CACHE["collect_row"] = None
            _slab_write(row, x)
        _CACHE["resid_in"] = (x, self.scaling_factor) if "out_proj_residual" in _OPS else None
        _CACHE["resid_consumed"] = False
        attn_out, attn_weights = self.attn(x, sequence_id, output_attentions=output_attentions)
        if _CACHE["resid_consumed"]:                                                # the merged seam returned x + out_proj(ctx) / sf
            x = attn_out
        else:
            x = _OPS["residual"](x, attn_out, self.scaling_factor)
        _CACHE["resid_in"] = None
        y = _OPS["ffn"](self.ffn, x)
        if _CACHE["defer_ok"] and id(self) not in _state["last_block_ids"]:
            _CACHE["pending"] = (y, self.scaling_factor)                             # the next block's merged seam applies x + y / sf
            return x, attn_weights
        return _OPS["residual"](x, y, self.scaling_factor), attn_weights
    forward.__wrapped_stock__ = orig
    return forward


def _slab_write(row, h):
    """write one collected state into the preallocated fp32 slab row: unpadded -> copy_ (bf16 -> fp32 exact); padded -> a scatter with the
    forward's own unpad `indices` into the zero-filled [B*L, D] row (= the stock's pad_input, then the stack's promotion: exact)."""
    slab, idx = _CACHE["slab"], _CACHE["indices"]
    if idx is None:
        slab[row].copy_(h)
    else:
        from . import _hs_write
        _hs_write.scatter_cast(slab[row], idx, h.contiguous())                      # one scatter-with-cast launch: the bytes of pad_input(h).float()


def _make_stack_forward(orig):
    import torch
    import torch.nn.functional as F

    def forward(self, x, sequence_id=None, layers_to_collect=None, output_attentions=False):
        COUNTERS["stack_forwards"] += 1
        mine = False
        if _CACHE["mask"] is None and sequence_id is not None and sequence_id.dtype == torch.bool and sequence_id.ndim == 2:
            # the stock path's hoist (for batch > 1 / padded calls): the same three ops on the same mask, ONCE per forward
            seqlens = sequence_id.sum(dim=-1, dtype=torch.int32)
            _CACHE["cu"] = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
            _CACHE["max"] = int(seqlens.max().item())
            _CACHE["mask"] = sequence_id
            _CACHE["fast"] = False
            mine = True
        try:
            if (_CACHE["fast"] and layers_to_collect and not output_attentions and x.dim() == 2 and int(x.shape[0]) >= HS_SLAB_MIN_ROWS
                    and sequence_id is _CACHE["mask"]):
                # the stock's loop with the collected states written once into a preallocated fp32 slab (the stack's promotion, per layer)
                want = sorted(set(int(i) for i in layers_to_collect))
                pos = {i: j for j, i in enumerate(want)}
                idx = _CACHE["indices"]
                if idx is None:
                    slab = torch.empty((len(want), int(x.shape[0]), int(x.shape[1])), dtype=torch.float32, device=x.device)
                else:                                                                 # the padded route — ONE zero fill, then one scatter per state
                    bL = int(_CACHE["pad_shape"][0]) * int(_CACHE["pad_shape"][1])
                    slab = torch.zeros((len(want), bL, int(x.shape[1])), dtype=torch.float32, device=x.device)
                _CACHE["slab"] = slab
                try:
                    for layer_idx, block in enumerate(self.blocks):
                        _CACHE["collect_row"] = pos.get(layer_idx)                    # the block writes its (post-merge) input into that row
                        x, _ = block(x, sequence_id, output_attentions=False)
                    _CACHE["collect_row"] = None
                    norm_x = self.norm(x)
                    j = pos.get(len(self.blocks))
                    if j is not None:
                        _slab_write(j, norm_x)                                        # fp32 -> fp32: a copy / scatter
                finally:
                    _CACHE["slab"] = None; _CACHE["collect_row"] = None
                COUNTERS["hs_slab_forwards"] += 1
                return norm_x, x, slab, None
            return orig(self, x, sequence_id, layers_to_collect, output_attentions)
        finally:
            if mine:
                _CACHE.update({"mask": None, "cu": None, "max": None, "fast": False, "qkv": None, "pending": None, "defer_ok": False, "resid_in": None, "resid_consumed": False, "s_mask": None, "indices": None, "slab": None, "collect_row": None, "s_route": False})
    forward.__wrapped_stock__ = orig
    return forward


class _thin_off:
    """Run a STOCK-PATH call on the SDPA-constructed model with the thin item's lazy instance forwards lifted — thin's templates are
    built from 2-D packed calls and replay on any input with the same dtype/device/last dim (a padded 3-D S call included, where replaying
    the 2-D templates does not reproduce the stock's fp32 hidden states bit for bit) — so the S model's stock path runs the pure stock forward: the instance
    attribute `forward` thin set on each module is popped for the call and put back after (the class forward applies meanwhile)."""

    def __init__(self, model):
        self.model, self.saved = model, []

    def __enter__(self):
        for m in self.model.modules():
            if hasattr(m, "_thin_launch_orig_forward") and "forward" in m.__dict__:
                self.saved.append((m, m.__dict__.pop("forward")))
        COUNTERS["s_route_thin_off_calls"] += 1
        return self

    def __exit__(self, *exc):
        for m, f in self.saved:
            m.forward = f
        self.saved = []
        return False


def _s_attn_mask(mask, padded):
    """the stock's S-route attention mask (model.py: `trans_seq_id = None` when bool_mask.all() and not output_attentions, else
    `bool_mask[:, None, None, :]`): None on an all-valid batch (the fused unmasked dispatch), the KEY mask [b, 1, 1, L] otherwise."""
    return None if not padded else mask[:, None, None, :]


def _make_model_forward(orig):
    import torch
    from esm.models.esmc.model import EsmcOutput

    def forward(self, input_ids=None, attention_mask=None, sequence_id=None, output_hidden_states=None, output_attentions=None,
                return_dict=None, compute_sae=True, normalize_sae=False):
        COUNTERS["model_forwards"] += 1
        oa = output_attentions if output_attentions is not None else self.config.output_attentions
        reason = None
        s_route = not self._use_flash_attn                                          # the SDPA-constructed model
        if s_route and not _state.get("s_route_ok"):
            reason = "sdpa model (attention class not the stock's EsmcMultiHeadAttention)"
        elif input_ids is None or sequence_id is not None:
            reason = "sequence_id supplied (or no input_ids)"
        elif attention_mask is not None and (attention_mask.shape != input_ids.shape or not attention_mask.is_cuda):
            reason = "attention_mask shape/device"
        elif input_ids.dim() != 2:
            reason = "input_ids not 2-D"
        elif oa:
            reason = "output_attentions"
        elif compute_sae and len(self._sae_models) > 0:
            reason = "sae"
        elif torch.is_grad_enabled():
            reason = "grad enabled"
        elif not input_ids.is_cuda:
            reason = "not cuda"
        padded = False
        if reason is None:
            # the stock's own mask rule: a supplied attention_mask is used as-is (bool()); else input_ids != pad_token_id
            mask = attention_mask.bool() if attention_mask is not None else (input_ids != self.config.pad_token_id)
            padded = not bool(mask.all().item())                                     # ONE host read per forward (the stock syncs here too, inside unpad_input)
        if reason is not None:
            COUNTERS["stock_path"] += 1
            _state["reason"] = reason
            _state.setdefault("reasons", {})[reason] = _state.setdefault("reasons", {}).get(reason, 0) + 1
            if s_route and _state.get("s_route_ok"):
                with _thin_off(self):
                    return orig(self, input_ids, attention_mask, sequence_id, output_hidden_states, output_attentions, return_dict, compute_sae, normalize_sae)
            return orig(self, input_ids, attention_mask, sequence_id, output_hidden_states, output_attentions, return_dict, compute_sae, normalize_sae)
        COUNTERS["fast_path"] += 1
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        layers_to_collect = list(range(self.config.num_hidden_layers + 1)) if output_hidden_states else []
        b, L = input_ids.shape
        _ac_on = torch.is_autocast_enabled("cuda") if hasattr(torch, "is_autocast_enabled") else False   # the caller's autocast state, restored for the padded-S stock-path call made inside this context
        _ac_dtype = torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype") else torch.bfloat16
        with torch.autocast(device_type="cuda", enabled=False):
            x = self.embed(input_ids)                                                 # bf16 (autocast never touched the embedding)
            d = x.shape[-1]
            if not padded:
                x = x.view(b * L, d)                                                  # = unpad_input's gather without padding (any b): a view, no copy
                indices = None
                cu, mx = _cu_seqlens(L, x.device, b), L                              # = the stock's F.pad(cumsum(mask.sum(-1))) / seqlens.max() for b x L valid tokens
                COUNTERS["fast_unpadded"] += 1
            elif s_route:                                                             # a PADDED call on the S model -> the S FAST PATH on the [b*L, d] rows, pads included
                x = x.view(b * L, d)                                                  # (the stock never unpads on the S route: every row is computed; its attention takes the KEY mask)
                indices = None
                cu, mx = _cu_seqlens(L, x.device, b), L                              # rotary positions 0..L-1 per sequence = the stock's on [b, L] rows
                COUNTERS["s_route_padded_fast"] += 1
            else:                                                                     # the STOCK's own unpad ONCE per forward (flash_attn.bert_padding via esm.models.esmc.model)
                from esm.models.esmc.model import unpad_input as _unpad
                res = _unpad(x, mask)
                x, indices = res[0], res[1]
                seqlens = mask.sum(dim=-1, dtype=torch.int32)                        # the stock's per-layer metadata ops, in the stock's order, once
                cu = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
                mx = int(seqlens.max().item())
                COUNTERS["fast_padded"] += 1
            _CACHE.update({"mask": mask, "cu": cu, "max": mx, "fast": True, "qkv": None, "pending": None, "resid_in": None, "resid_consumed": False,
                           "indices": indices, "pad_shape": (b, L), "slab": None, "collect_row": None, "s_route": bool(s_route),
                           "s_mask": _s_attn_mask(mask, padded) if s_route else None,
                           "defer_ok": ("residual_ln_qkv" in _OPS) and id(self.transformer.blocks[-1]) in _state["last_block_ids"]
                                       and (not layers_to_collect or int(x.shape[0]) >= HS_SLAB_MIN_ROWS)})   # on the full route too, iff the slab loop collects
            try:
                last_hidden_state, prenorm, collected, attentions = self.transformer(x, sequence_id=mask, layers_to_collect=layers_to_collect, output_attentions=False)
            finally:
                _CACHE.update({"mask": None, "cu": None, "max": None, "fast": False, "qkv": None, "pending": None, "defer_ok": False, "resid_in": None, "resid_consumed": False, "s_mask": None, "indices": None, "slab": None, "collect_row": None, "s_route": False})
            slab = collected if torch.is_tensor(collected) else None
            if indices is None:
                last_hidden_state = last_hidden_state.view(b, L, d)                  # = pad_input without padding: a view
                prenorm = prenorm.view(b, L, d)
                collected = (slab.view(slab.shape[0], b, L, d),) if slab is not None else tuple(h.view(b, L, d) for h in collected)
            else:
                from esm.models.esmc.model import pad_input as _pad                   # the stock's own scatter, in the stock's order
                last_hidden_state = _pad(last_hidden_state, indices, b, L)
                prenorm = _pad(prenorm, indices, b, L)
                collected = (slab.view(slab.shape[0], b, L, d),) if slab is not None else tuple(_pad(h, indices, b, L) for h in collected)
            if slab is not None:
                collected_tensor = collected[0]                                       # the slab IS the stacked tensor (fp32 [n, b, L, d], contiguous)
            else:
                collected_tensor = torch.stack(collected, dim=0) if collected else None   # the stock's stack (bf16 inputs + the fp32 norm -> fp32)
        hidden_states_tensor = collected_tensor if output_hidden_states else None
        if not return_dict:
            return tuple(v for v in [last_hidden_state, hidden_states_tensor, None, attentions] if v is not None)
        return EsmcOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states_tensor, sae_outputs=None, attentions=attentions,
                          last_hidden_state_prenorm=prenorm if output_hidden_states else None)
    forward.__wrapped_stock__ = orig
    return forward


def _make_final_norm_forward(orig):
    """The stack's final nn.LayerNorm(bias=False) under the wrapper's autocast = input->fp32, weight->fp32 (per call), LayerNorm -> fp32.
    On the fast path (autocast off) the same fp32 LayerNorm on the same fp32 input with the weight cast once."""
    import torch
    import torch.nn.functional as F

    def forward(self, input):
        if _CACHE["fast"] and not torch.is_autocast_enabled():
            w, b = _ln32(self)
            return F.layer_norm(input.float(), self.normalized_shape, w, b, self.eps)
        return orig(self, input)
    forward.__wrapped_stock__ = orig
    return forward


def _make_logits(orig):
    """ESMC.logits with hidden states ON DEMAND: identical to the wrapper's except the native forward
    runs with output_hidden_states=config.return_hidden_states and the full hidden-states stack is built only then; the context is the stock's."""
    import torch

    def logits(self, input, config=None):
        import esm.models.esmc.compatibility as C
        from esm.sdk.api import LogitsConfig
        COUNTERS["logits_calls"] += 1
        if config is None:
            config = LogitsConfig()
        if not isinstance(input, C._BatchedESMProteinTensor):
            input = C._BatchedESMProteinTensor.from_protein_tensor(input)
        device = torch.device(input.device)
        want_hidden = bool(config.return_hidden_states)
        COUNTERS["hidden_requested"] += int(want_hidden)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = self.model(input_ids=input.sequence, attention_mask=None, sequence_id=None, output_hidden_states=want_hidden,
                             output_attentions=False, return_dict=True)
        hidden = C._legacy_hidden_states(out.hidden_states, out.last_hidden_state_prenorm) if want_hidden else None
        if hidden is not None and config.ith_hidden_layer != -1:
            layer = config.ith_hidden_layer
            hidden = hidden[layer: layer + 1]
        return C.LogitsOutput(logits=C.ForwardTrackData(sequence=out.logits if config.sequence else None),
                              embeddings=out.last_hidden_state if config.return_embeddings else None,
                              hidden_states=hidden if want_hidden else None)
    logits.__wrapped_stock__ = orig
    return logits


# ----------------------------------------------------------------------------------------------------------------------------------
# engage / disengage
# ----------------------------------------------------------------------------------------------------------------------------------
def _engage_model(model, engaged=None) -> None:
    """The per-MODEL part of engage: the final norm's INSTANCE patch (nn.LayerNorm is shared with q_ln/k_ln and the LM head; only this
    one), the model's last block registered (the deferred FFN residual resolves there), the fp32 weight copies warmed (exact upcasts) so the
    first call pays no extra launches. Runs once per model; every model built under the mode goes through it."""
    import torch
    esmc = model.esmc if hasattr(model, "esmc") else model
    last = id(esmc.transformer.blocks[-1])
    if last in _state["last_block_ids"]:
        return
    norm = esmc.transformer.norm
    norm.forward = _make_final_norm_forward(type(norm).forward).__get__(norm, type(norm))
    _state["norms"].append(norm)
    _state["last_block_ids"].add(last)
    if engaged is not None:
        engaged.append("final norm: weight cast once (instance patch)")
    for blk in esmc.transformer.blocks:
        if isinstance(blk.attn.q_ln, torch.nn.LayerNorm):
            _ln32(blk.attn.q_ln); _ln32(blk.attn.k_ln)
    _ln32(norm)
    if next(model.parameters()).is_cuda:
        _cu_seqlens(2, next(model.parameters()).device)


def engage(model=None) -> dict:
    """Pins + three class-attribute patches (once per process) + the per-model part (_engage_model: the final norm's instance patch, the
    last block registered) + the wrapper's logits. Idempotent per process and per model: a second model gets its per-model part only."""
    if _state["engaged"]:
        if model is not None:
            _engage_model(model)
        return _state["last"]
    import esm.models.esmc.layers as L
    import esm.models.esmc.model as M
    p = pins()
    if p["drift"]:
        raise PatchRefused(f"{PATCH}: module drift vs esm@{ESM_COMMIT[:12]}: {p['drift']} — {[(n, p['files'][n]['sha256'][:12]) for n in p['drift']]}")
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func
    except Exception as ex:  # noqa: BLE001
        raise PatchRefused(f"{PATCH}: flash_attn is not importable ({type(ex).__name__}: {ex}); r1 targets the flash-attn (SDK) route") from ex
    t0 = time.perf_counter()
    global _TRITON_ROTARY_FORWARD
    from esm.models.esmc.kernels import apply_triton_rotary as rotary_fn
    _TRITON_ROTARY_FORWARD = L.EsmcTritonRotaryEmbedding.forward
    _install_default_ops(flash_attn_varlen_qkvpacked_func, rotary_fn)
    _state["orig"] = {"attn": L.EsmcFlashMultiHeadAttention.forward, "stack": L.EsmcTransformerStack.forward, "model": M.EsmcModel.forward,
                      "block": L.EsmcUnifiedTransformerBlock.forward}
    L.EsmcFlashMultiHeadAttention.forward = _make_attn_forward(_state["orig"]["attn"], flash_attn_varlen_qkvpacked_func, rotary_fn)
    _state["orig"]["attn_sdpa"] = L.EsmcMultiHeadAttention.__dict__.get("forward")   # the SDPA class's own forward (the S model's modules)
    if _state["orig"]["attn_sdpa"] is not None:
        L.EsmcMultiHeadAttention.forward = _make_attn_forward(_state["orig"]["attn_sdpa"], flash_attn_varlen_qkvpacked_func, rotary_fn, sdpa_class=True)
        _state["s_route_ok"] = True
    L.EsmcUnifiedTransformerBlock.forward = _make_block_forward(_state["orig"]["block"])
    L.EsmcTransformerStack.forward = _make_stack_forward(_state["orig"]["stack"])
    M.EsmcModel.forward = _make_model_forward(_state["orig"]["model"])
    engaged = ["varlen metadata once per forward (no per-layer sync)", "q_ln/k_ln: no per-call weight casts, LN'd q/k written into the QKV slots (no stack)",
               "rotary: one Triton launch for q+k" if rotary_fn is not None else "rotary: stock (Triton kernel absent)",
               "unpad/pad elided at batch 1 (views)", "fast path with autocast off (explicit dtypes)"]
    if model is not None:
        _engage_model(model, engaged)
    try:
        import esm.models.esmc.compatibility as C
        _state["orig"]["logits"] = C.ESMC.logits
        C.ESMC.logits = _make_logits(_state["orig"]["logits"])
        engaged.append("SDK logits: hidden states on demand")
    except Exception as ex:  # noqa: BLE001 — no SDK wrapper on this image: stated
        if is_oom(ex): raise                                  # an out-of-memory propagates: no fallback applied
        _state["orig"]["logits"] = None; engaged.append(f"SDK logits: not patched ({type(ex).__name__})")
    _state["engaged"] = True
    _state["pins"] = p
    _state["last"] = {"engaged": engaged, "pins": p, "engage_s": round(time.perf_counter() - t0, 4),
                      "patch_sha256": _sha_file(os.path.abspath(__file__)), "esm_commit": ESM_COMMIT}
    return _state["last"]


def disengage(model=None) -> None:
    """Restore every class attribute and the instance patch; clear the caches. Idempotent."""
    if not _state["engaged"]:
        return
    import esm.models.esmc.layers as L
    import esm.models.esmc.model as M
    o = _state["orig"]
    L.EsmcFlashMultiHeadAttention.forward = o["attn"]
    if o.get("attn_sdpa") is not None:
        L.EsmcMultiHeadAttention.forward = o["attn_sdpa"]
    _state["s_route_ok"] = False
    L.EsmcUnifiedTransformerBlock.forward = o["block"]
    L.EsmcTransformerStack.forward = o["stack"]
    M.EsmcModel.forward = o["model"]
    for norm in _state["norms"]:                      # every engaged model's final-norm instance patch
        if "forward" in norm.__dict__:
            del norm.__dict__["forward"]
    if o.get("logits") is not None:
        import esm.models.esmc.compatibility as C
        C.ESMC.logits = o["logits"]
    _state.update({"engaged": False, "orig": {}, "ln32": {}, "arange2": {}, "cu": {}, "reason": None, "last": None, "last_block_ids": set(), "norms": []})
    _CACHE.update({"mask": None, "cu": None, "max": None, "fast": False, "qkv": None, "pending": None, "defer_ok": False})
    for k in OPTIONAL_SEAMS:
        _OPS.pop(k, None)


