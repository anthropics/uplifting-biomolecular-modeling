"""cond_levers.py — levers on boltz 2.2.1's DiffusionConditioning (stock/src/boltz/model/modules/diffusion_conditioning.py), the
once-per-prediction conditioning of the diffusion module (Boltz2.forward, boltz2.py:516-531; bf16 autocast on).

  condproj  Tier 2 (fast). Stock projects the conditioned pair tensor z [B,N,N,128] into the 24 token-transformer layers' attention biases with 24
            separate ``Sequential(LayerNorm(128), Linear(128 -> 16, bias=False))`` (diffusion_conditioning.py:74-81, :111-114) and the atom pair
            tensor p [B,K,32,128,16] into the 3+3 atom encoder/decoder layers' biases the same way (:56-72, :101-109) — 30 LayerNorm passes and 30
            thin GEMMs over pair tensors, then two/three ``torch.cat``. The LayerNorms of one list read the SAME input, so their statistics are
            the same numbers; only (gamma_l, beta_l) differ:  Linear_l(LN_l(x)) = W_l (gamma_l * xhat + beta_l) = (W_l diag gamma_l) xhat + W_l beta_l
            with xhat = (x - mean) * rstd. The lever computes xhat ONCE per list (one affine-free layer_norm pass) and ONE GEMM against the
            concatenated folded weight [L*H, C] with the folded bias [L*H] — the concatenation order l*H + h is torch.cat(dim=-1)'s. Exact in
            real arithmetic, not bitwise (gamma folded into W before the bf16 rounding, beta*W added as a bias, xhat rounded to bf16 instead of
            gamma*xhat+beta): a declared Tier-2 (fast) lever. dtypes as stock under bf16 autocast: the pair biases come
            out bf16 (stock's Linear outputs are bf16 there); z's statistics are taken in fp32 inside the LayerNorm kernel from the bf16 z the
            pairwise conditioner produces (stock upcasts the same bf16 z to fp32 for its LN — same statistic arithmetic); p is fp32 (AtomEncoder
            runs with autocast off) and is normalised in fp32 as stock does, then cast to bf16 for the GEMM as stock's autocast Linear does.
            Refuses by name at install when the module is not the pinned structure (Sequential(LayerNorm affine, Linear bias-free), one eps and
            width per list).

STATS: calls, token_lists_fused, atom_lists_fused, layers_folded.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

LEVERS = ("condproj",)
STATS: Dict[str, int] = {"calls": 0, "token_lists_fused": 0, "atom_lists_fused": 0, "layers_folded": 0}
_STATE: Dict[str, Any] = {"installed": (), "orig_forward": None}
_FOLD_ATTR = "_conf_condproj_folded"


def _cnt(k: str, n: int = 1) -> None:
    STATS[k] = STATS.get(k, 0) + n


def check_list(layers) -> str | None:
    """None when `layers` is a ModuleList of Sequential(LayerNorm(C, affine), Linear(C, H, bias=False)) sharing C, H and eps; else the reason."""
    if len(layers) == 0:
        return "empty ModuleList"
    C = H = eps = None
    for i, seq in enumerate(layers):
        if not isinstance(seq, torch.nn.Sequential) or len(seq) != 2:
            return f"layer {i}: not Sequential(LayerNorm, Linear)"
        ln, lin = seq[0], seq[1]
        if not isinstance(ln, torch.nn.LayerNorm) or not isinstance(lin, torch.nn.Linear):
            return f"layer {i}: {type(ln).__name__}, {type(lin).__name__} (want LayerNorm, Linear)"
        if not ln.elementwise_affine or ln.weight is None or ln.bias is None or len(ln.normalized_shape) != 1:
            return f"layer {i}: LayerNorm must be affine over the last dim"
        if lin.bias is not None:
            return f"layer {i}: Linear has a bias (stock: bias=False)"
        c, h = lin.in_features, lin.out_features
        if ln.normalized_shape[0] != c:
            return f"layer {i}: LayerNorm width {ln.normalized_shape[0]} != Linear in_features {c}"
        if C is None:
            C, H, eps = c, h, ln.eps
        elif (c, h, ln.eps) != (C, H, eps):
            return f"layer {i}: (C,H,eps)=({c},{h},{ln.eps}) != ({C},{H},{eps})"
    return None


def fold(layers) -> dict:
    """The folded weight [L*H, C] and bias [L*H] of a checked list, fp32 masters + bf16 copies (computed once: inference weights are frozen)."""
    with torch.no_grad():
        Ws, bs = [], []
        for seq in layers:
            ln, lin = seq[0], seq[1]
            W = lin.weight.detach().float()                       # [H, C]
            Ws.append(W * ln.weight.detach().float()[None, :])   # W diag(gamma)
            bs.append(W @ ln.bias.detach().float())              # W beta
        Wf = torch.cat(Ws, dim=0).contiguous(); bf = torch.cat(bs, dim=0).contiguous()
    _cnt("layers_folded", len(layers))
    return {"W32": Wf, "b32": bf, ("W", torch.bfloat16): Wf.to(torch.bfloat16), ("b", torch.bfloat16): bf.to(torch.bfloat16), "eps": layers[0][0].eps,
            "C": layers[0][1].in_features, "L": len(layers), "H": layers[0][1].out_features}


def _folded(self, name: str) -> dict:
    cache = self.__dict__.setdefault(_FOLD_ATTR, {})
    f = cache.get(name)
    layers = getattr(self, name)
    dev = layers[0][1].weight.device
    if f is None or f["W32"].device != dev:
        f = fold(layers); cache[name] = f
    return f


def fused_projection(x: torch.Tensor, f: dict) -> torch.Tensor:
    """cat_l Linear_l(LayerNorm_l(x)) over the last dim, as one affine-free normalisation + one GEMM with the folded weight and bias.
    GPU under bf16 autocast (the model's inference context): bf16 in/out GEMM as stock's autocast Linear; otherwise (CPU tests) fp32 throughout."""
    C = f["C"]
    lowp = None
    if x.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") in (torch.bfloat16, torch.float16):
        lowp = torch.get_autocast_dtype("cuda")                                      # the dtype stock's autocast Linear computes and returns in
    with torch.autocast(device_type="cuda" if x.is_cuda else "cpu", enabled=False):
        if lowp is not None:
            W, b = f.get(("W", lowp)), f.get(("b", lowp))
            if W is None:
                W = f[("W", lowp)] = f["W32"].to(lowp); b = f[("b", lowp)] = f["b32"].to(lowp)
            if x.dtype == lowp:
                xhat = F.layer_norm(x, (C,), None, None, f["eps"])                  # bf16 in -> fp32 statistics in-kernel -> bf16 out (one pass)
            else:
                xhat = F.layer_norm(x.float(), (C,), None, None, f["eps"]).to(lowp)  # fp32 input (p): normalised in fp32 as stock, then cast as stock's autocast Linear casts
            return F.linear(xhat, W, b)
        xhat = F.layer_norm(x.float(), (C,), None, None, f["eps"])
        return F.linear(xhat, f["W32"], f["b32"]).to(x.dtype if x.is_floating_point() else torch.float32)


def forward_condproj(self, s_trunk, z_trunk, relative_position_encoding, feats):
    """DiffusionConditioning.forward (diffusion_conditioning.py:83-116) with the three projection lists fused (condproj)."""
    if self.training or "condproj" not in _STATE["installed"]:
        return _STATE["orig_forward"](self, s_trunk, z_trunk, relative_position_encoding, feats)
    _cnt("calls")
    z = self.pairwise_conditioner(z_trunk, relative_position_encoding)
    q, c, p, to_keys = self.atom_encoder(feats=feats, s_trunk=s_trunk, z=z)
    atom_enc_bias = fused_projection(p, _folded(self, "atom_enc_proj_z")); _cnt("atom_lists_fused")
    atom_dec_bias = fused_projection(p, _folded(self, "atom_dec_proj_z")); _cnt("atom_lists_fused")
    token_trans_bias = fused_projection(z, _folded(self, "token_trans_proj_z")); _cnt("token_lists_fused")
    return q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias


def install(levers, model_hook: bool = True) -> tuple:
    """Install condproj as a class-level replacement of DiffusionConditioning.forward (idempotent). The pinned structure lives on instances
    (the three ModuleLists), so it is checked per instance at its first call (_checked_forward) and a violation raises by name there —
    never a silent stock fallback. ``model_hook`` is accepted for interface symmetry with template_levers (unused)."""
    want = tuple(l for l in LEVERS if l in set(levers))
    if not want:
        return _STATE["installed"]
    from boltz.model.modules import diffusion_conditioning as DC
    if _STATE["orig_forward"] is None:
        _STATE["orig_forward"] = DC.DiffusionConditioning.forward
        DC.DiffusionConditioning.forward = _checked_forward
    _STATE["installed"] = tuple(l for l in LEVERS if l in set(_STATE["installed"]) | set(want))
    return _STATE["installed"]


def _checked_forward(self, s_trunk, z_trunk, relative_position_encoding, feats):
    """First call per instance: check the pinned structure (refuse by name otherwise), then serve through forward_condproj."""
    if not self.__dict__.get("_conf_condproj_checked"):
        for name in ("atom_enc_proj_z", "atom_dec_proj_z", "token_trans_proj_z"):
            why = check_list(getattr(self, name, ()))
            if why is not None:
                raise RuntimeError(f"[boltz2-opt] conf.condproj REFUSED: DiffusionConditioning.{name}: {why}")
        self.__dict__["_conf_condproj_checked"] = True
    return forward_condproj(self, s_trunk, z_trunk, relative_position_encoding, feats)


def uninstall() -> None:
    from boltz.model.modules import diffusion_conditioning as DC
    if _STATE["orig_forward"] is not None:
        DC.DiffusionConditioning.forward = _STATE["orig_forward"]; _STATE["orig_forward"] = None
    _STATE["installed"] = ()


def report() -> Dict[str, Any]:
    return {"installed": list(_STATE["installed"]), "stats": dict(STATS)}
