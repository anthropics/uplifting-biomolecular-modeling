"""
boltz_trunk_levers.py -- runtime monkeypatches for the Boltz-2 2.2.1 trunk (inference only; the boltz wheel is untouched).
Default (BOLTZ_LEVERS unset/empty): NOTHING is patched -> stock bytes. The modes' row is ``BOLTZ_LEVERS=resid,mask2``
(boltz2_opt.modes); the worker imports this module and calls ``apply()`` once, right after its own boltz imports
(forward/trunk_levers/src/bz_worker_lev.py), and records ``report()`` in its log at exit.

Levers (read ONCE at apply(); nothing is re-read per call). Both are EXACT: the same arithmetic per output element as
stock, only redundant work removed; they act beside upstream's fused triangle kernels (use_kernels=True), which they
never replace:
    resid   eval-mode residuals `z = z + dropout_mask * f(z)` with dropout_mask == 1.0 (fp32 ones, shape [B,N,1,1]) ->
            `z = z + f(z)` with the same fp32 promotion (the torch.rand call of get_dropout_mask is KEPT so the CUDA RNG
            stream -- and therefore every diffusion noise draw -- is unchanged).  PairformerLayer / PairformerNoSeqLayer.
            Its residual statements are the hook of the kit's residual/LayerNorm fuser (``RESID_FUSER``, set by
            boltz2_opt.exactln when the row carries lever exactln_resid): the fp32 add `z + f(z)` and the NEXT block's
            LayerNorm of the sum in one pass over z, the LayerNorm parked for that block's own call -- offered at the
            three sites whose consumer makes a LayerNorm call (tri_mul_in -> tri_att_start, tri_att_start -> tri_att_end
            [transposed read], tri_att_end -> transition_z) and, in PairformerLayer, transition_z -> the sequence
            attention's pair-bias LayerNorm (fp32, autocast off); NOT after tri_mul_out (tri_mul_in normalises inside the
            triangle-multiplication kernels: no call to hand a parked result to) nor after a PairformerNoSeqLayer's
            transition (the consumer is the caller's).  No fuser registered, or the fuser hands the call back (None):
            the statement below runs as written.
    mask2   pairformer sequence attention (AttentionPairBias, fp32, autocast off): skip `attn + (1 - mask) * -inf` (adds
            -0.0 to every logit) when the pair mask is all ones (a single unpadded input => always).  The trivial-mask flag
            is computed ONCE per PairformerModule / PairformerNoSeqModule / MSAModule call (1 host sync) and consulted only
            inside that scope; outside it (the diffusion token transformer) the stock path runs with no per-call check.  A
            mask the scope cannot classify counts ``mask_scope_error`` and that call runs the stock mask arithmetic (the
            caller's evidence reads the count: boltz2_opt.stack.partial_activation).

Switches read: ``BOLTZ_LEVERS`` (the row; an unknown name raises by name), ``BOLTZ_LEVERS_VERBOSE`` (1: one
``[boltz_trunk_levers] APPLIED levers=...`` line at apply()).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

import torch
from opt_core.oom import is_oom          # the core's one out-of-memory predicate: an out-of-memory error propagates, no silent stock route

LEVERS = ("resid", "mask2")              # every lever this module implements, in the row's order
RESID_FUSER = None                       # the residual/LayerNorm fuser (boltz2_opt.exactln.resid_fuser when the row carries exactln_resid, else None):
                                         # RESID_FUSER(z, u, next_ln, transpose_ln, ln_autocast, site) -> z + u (torch's promote-add bits) with
                                         # next_ln's LayerNorm of the sum parked for that module's coming call, or None = this call not fused

_STATE: Dict[str, Any] = {"applied": False, "levers": (), "orig": {}, "t_apply": None}
STATS: Dict[str, int] = {}
_CTX = threading.local()          # .mask_trivial : Optional[bool] for the module call in flight
_LOCK = threading.Lock()


def _cnt(k, n=1):
    STATS[k] = STATS.get(k, 0) + n


def parse_levers(spec: Optional[str]) -> tuple:
    """``BOLTZ_LEVERS`` -> the tuple of lever names in the order given (duplicates dropped); an unknown name raises by name."""
    spec = (spec or "").strip().lower()
    if not spec:
        return ()
    seen = []
    for tok in spec.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in LEVERS:
            raise ValueError(f"BOLTZ_LEVERS: unknown lever '{tok}' (known: {LEVERS})")
        if tok not in seen:
            seen.append(tok)
    return tuple(seen)


def on(name: str) -> bool:
    return name in _STATE["levers"]


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------
def _mask_is_trivial(mask: torch.Tensor) -> bool:
    """True iff every element of the pair mask equals 1 (one device->host sync)."""
    return bool((mask == 1).all().item())


def _resid(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """stock: z + ones_fp32[B,N,1,1] * x   (result dtype = promote(z, fp32, x))."""
    if z.dtype == torch.float32 or x.dtype == torch.float32:
        return z + x if (z.dtype == torch.float32 and x.dtype in (torch.float32, torch.bfloat16, torch.float16)) or \
                        (x.dtype == torch.float32 and z.dtype in (torch.float32, torch.bfloat16, torch.float16)) else z.float() + x.float()
    # both low precision (first residual after the outer-product-mean in MSALayer): promote exactly like stock
    return z + x.float()


def _fused(z: torch.Tensor, x: torch.Tensor, next_ln, site: str, transpose_ln: bool = False, ln_autocast: bool = True) -> Optional[torch.Tensor]:
    """The residual `z + x` through RESID_FUSER -- when a fuser is registered, the consumer's LayerNorm module is known and the statement
    is the fp32 promote-add of one shape (z fp32 CUDA, x bf16 or fp32: what `_resid` / `z + x` compute as ONE torch.add there); None
    otherwise (no fuser, another operand form, or the fuser hands the call back by its own census word) -> the caller runs the statement."""
    f = RESID_FUSER
    if f is None or next_ln is None or z.dtype != torch.float32 or x.dtype not in (torch.float32, torch.bfloat16) or x.shape != z.shape or not z.is_cuda:
        return None
    r = f(z, x, next_ln, transpose_ln, ln_autocast, site)
    _cnt("resid_fused" if r is not None else "resid_unfused")
    return r


def _resid_ln(z: torch.Tensor, x: torch.Tensor, next_ln, site: str, transpose_ln: bool = False) -> torch.Tensor:
    """`_resid(z, x)` with the sum's coming LayerNorm (`next_ln`, under the trunk's bf16 autocast) offered to the fuser first."""
    zn = _fused(z, x, next_ln, site, transpose_ln=transpose_ln, ln_autocast=True)
    return zn if zn is not None else _resid(z, x)


def _pair_bias_ln(layer) -> Any:
    """The sequence attention's pair-bias LayerNorm (AttentionPairBias.proj_z[0]: LayerNorm(c_z) before the Linear to the heads), or None."""
    pz = getattr(getattr(layer, "attention", None), "proj_z", None)
    try:
        m = pz[0]
    except (TypeError, IndexError, KeyError):
        return None
    return m if (hasattr(m, "normalized_shape") or hasattr(m, "c_in")) else None      # nn.LayerNorm / boltz's primitives.LayerNorm (the fuser reads weight / bias / eps itself)


def _keep_rng(dropout: float, z: torch.Tensor, columnwise: bool = False) -> None:
    """Consume the CUDA generator exactly like boltz.model.layers.dropout.get_dropout_mask (one fp32 torch.rand of B*N)."""
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    torch.rand(v.shape, dtype=torch.float32, device=v.device)
    _cnt("rng_keep")


# ----------------------------------------------------------------------------------------------------------------------
# pairformer layers (residual lever) and module-level trivial-mask scoping
# ----------------------------------------------------------------------------------------------------------------------
def _pf_layer_forward(self, s, z, mask, pair_mask, chunk_size_tri_attn=None, use_kernels=False, use_cuequiv_mul=False, use_cuequiv_attn=False):
    if self.training or not on("resid"):
        return _STATE["orig"]["pfl_fwd"](self, s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels, use_cuequiv_mul, use_cuequiv_attn)
    _keep_rng(self.dropout, z)
    z = _resid(z, self.tri_mul_out(z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels))   # not offered to the fuser: tri_mul_in normalises inside the TriMul kernels
    _keep_rng(self.dropout, z)
    z = _resid_ln(z, self.tri_mul_in(z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels), getattr(self.tri_att_start, "layer_norm", None), "tri_att_start")
    _keep_rng(self.dropout, z)
    z = _resid_ln(z, self.tri_att_start(z, mask=pair_mask, chunk_size=chunk_size_tri_attn, use_kernels=use_cuequiv_attn or use_kernels), getattr(self.tri_att_end, "layer_norm", None), "tri_att_end", transpose_ln=True)
    _keep_rng(self.dropout, z, columnwise=True)
    z = _resid_ln(z, self.tri_att_end(z, mask=pair_mask, chunk_size=chunk_size_tri_attn, use_kernels=use_cuequiv_attn or use_kernels), getattr(self.transition_z, "norm", None), "transition_z")
    t = self.transition_z(z)
    zn = _fused(z, t, _pair_bias_ln(self), "proj_z", ln_autocast=False)                                # the sum's next LayerNorm: the sequence attention's pair bias reads z in fp32, autocast off
    z = zn if zn is not None else z + t
    with torch.autocast("cuda", enabled=False):
        s_normed = self.pre_norm_s(s.float())
        s = s.float() + self.attention(s=s_normed, z=z.float(), mask=mask.float(), k_in=s_normed)
        s = s + self.transition_s(s)
        s = self.s_post_norm(s)
    _cnt("pf_layer_resid")
    return s, z


def _pf_noseq_layer_forward(self, z, pair_mask, chunk_size_tri_attn=None, use_kernels=False, use_cuequiv_mul=False, use_cuequiv_attn=False):
    if self.training or not on("resid"):
        return _STATE["orig"]["pfnl_fwd"](self, z, pair_mask, chunk_size_tri_attn, use_kernels, use_cuequiv_mul, use_cuequiv_attn)
    _keep_rng(self.dropout, z)
    z = _resid(z, self.tri_mul_out(z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels))   # not offered to the fuser: tri_mul_in normalises inside the TriMul kernels
    _keep_rng(self.dropout, z)
    z = _resid_ln(z, self.tri_mul_in(z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels), getattr(self.tri_att_start, "layer_norm", None), "tri_att_start")
    _keep_rng(self.dropout, z)
    z = _resid_ln(z, self.tri_att_start(z, mask=pair_mask, chunk_size=chunk_size_tri_attn, use_kernels=use_cuequiv_attn or use_kernels), getattr(self.tri_att_end, "layer_norm", None), "tri_att_end", transpose_ln=True)
    _keep_rng(self.dropout, z, columnwise=True)
    z = _resid_ln(z, self.tri_att_end(z, mask=pair_mask, chunk_size=chunk_size_tri_attn, use_kernels=use_cuequiv_attn or use_kernels), getattr(self.transition_z, "norm", None), "transition_z")
    z = z + self.transition_z(z)                                                                        # not offered: the sum's next LayerNorm is the CALLER's next layer / module
    _cnt("pf_noseq_layer_resid")
    return z


def _scoped(orig_key: str, mask_from):
    """Wrap a module forward so patched inner layers see a precomputed trivial-mask flag (1 sync per call)."""
    def w(self, *a, **k):
        prev = getattr(_CTX, "mask_trivial", None)
        flag = None
        if on("mask2"):
            try:
                m = mask_from(self, a, k)
                flag = _mask_is_trivial(m) if m is not None else None
                _cnt("mask_scope_trivial" if flag else "mask_scope_nontrivial")
            except Exception as e:
                if is_oom(e): raise                                   # out-of-memory propagates (no silent stock-mask route)
                flag = None
                _cnt("mask_scope_error")
        _CTX.mask_trivial = flag
        try:
            return _STATE["orig"][orig_key](self, *a, **k)
        finally:
            _CTX.mask_trivial = prev
    return w


def _pfm_mask(self, a, k):      # PairformerModule.forward(self, s, z, mask, pair_mask, use_kernels=False)
    return k.get("pair_mask", a[3] if len(a) > 3 else None)


def _pfnm_mask(self, a, k):     # PairformerNoSeqModule.forward(self, z, pair_mask, use_kernels=False)
    return k.get("pair_mask", a[1] if len(a) > 1 else None)


def _msam_mask(self, a, k):     # MSAModule.forward(self, z, emb, feats, use_kernels=False): token_mask outer product
    feats = k.get("feats", a[2] if len(a) > 2 else None)
    tm = feats["token_pad_mask"].float()
    return tm[:, :, None] * tm[:, None, :]


# ----------------------------------------------------------------------------------------------------------------------
# mask2: trivial-mask elimination in the pairformer sequence attention
# ----------------------------------------------------------------------------------------------------------------------
def _apb_forward(self, s, z, mask, k_in, multiplicity=1):
    """AttentionPairBias.forward (pairformer seq attention): skip `attn + (1-mask)*-inf` (adds -0.0) when the pair mask is all ones."""
    # Only inside a patched trunk module scope (PairformerModule): the flag was computed once per module call.
    # Outside that scope (e.g. the diffusion token transformer, ~5000 calls per prediction) take the stock path
    # WITHOUT a per-call (mask == 1).all() host sync.
    if self.training or not on("mask2") or multiplicity != 1 or getattr(_CTX, "mask_trivial", None) is not True:
        return _STATE["orig"]["apb_fwd"](self, s, z, mask, k_in, multiplicity)
    B = s.shape[0]
    q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
    k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
    v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
    bias = self.proj_z(z)
    bias = bias.repeat_interleave(multiplicity, 0)
    g = self.proj_g(s).sigmoid()
    with torch.autocast("cuda", enabled=False):
        attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        attn = attn / (self.head_dim**0.5) + bias.float()
        attn = attn.softmax(dim=-1)
        o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
    o = o.reshape(B, -1, self.c_s)
    o = self.proj_o(g * o)
    _cnt("apb_mask_skipped")
    return o


def apply(spec: Optional[str] = None) -> tuple:
    """Patch according to `spec` (default: env BOLTZ_LEVERS). Returns the tuple of active levers (empty -> nothing patched)."""
    levers = parse_levers(os.environ.get("BOLTZ_LEVERS") if spec is None else spec)
    with _LOCK:
        if _STATE["applied"]:
            return _STATE["levers"]
        _STATE["levers"] = levers
        if not levers:
            return levers
        from boltz.model.layers import pairformer as PF
        from boltz.model.layers import attentionv2 as AV2
        from boltz.model.modules import trunkv2 as TR
        O = _STATE["orig"]
        O["pfl_fwd"] = PF.PairformerLayer.forward
        O["pfnl_fwd"] = PF.PairformerNoSeqLayer.forward
        O["pfm_fwd"] = PF.PairformerModule.forward
        O["pfnm_fwd"] = PF.PairformerNoSeqModule.forward
        O["msam_fwd"] = TR.MSAModule.forward
        O["apb_fwd"] = AV2.AttentionPairBias.forward
        if on("mask2"):
            AV2.AttentionPairBias.forward = _apb_forward
        if on("resid"):
            PF.PairformerLayer.forward = _pf_layer_forward
            PF.PairformerNoSeqLayer.forward = _pf_noseq_layer_forward
        # module-level scoping of the trivial-mask flag
        PF.PairformerModule.forward = _scoped("pfm_fwd", _pfm_mask)
        PF.PairformerNoSeqModule.forward = _scoped("pfnm_fwd", _pfnm_mask)
        TR.MSAModule.forward = _scoped("msam_fwd", _msam_mask)
        _STATE["applied"] = True
        _STATE["t_apply"] = time.time()
    if os.environ.get("BOLTZ_LEVERS_VERBOSE", "1") == "1":
        print(f"[boltz_trunk_levers] APPLIED levers={','.join(levers)}", flush=True)
    return levers


def remove() -> None:
    with _LOCK:
        if not _STATE["applied"]:
            return
        from boltz.model.layers import pairformer as PF
        from boltz.model.layers import attentionv2 as AV2
        from boltz.model.modules import trunkv2 as TR
        O = _STATE["orig"]
        PF.PairformerLayer.forward = O["pfl_fwd"]
        PF.PairformerNoSeqLayer.forward = O["pfnl_fwd"]
        PF.PairformerModule.forward = O["pfm_fwd"]
        PF.PairformerNoSeqModule.forward = O["pfnm_fwd"]
        TR.MSAModule.forward = O["msam_fwd"]
        AV2.AttentionPairBias.forward = O["apb_fwd"]
        _STATE["applied"] = False
        _STATE["levers"] = ()


def report() -> Dict[str, Any]:
    return {"applied": _STATE["applied"], "levers": list(_STATE["levers"]), "stats": dict(STATS), "torch": torch.__version__,
            "resid_fuser": None if RESID_FUSER is None else f"{getattr(RESID_FUSER, '__module__', '?')}.{getattr(RESID_FUSER, '__qualname__', '?')}"}
