"""template_levers.py — levers on boltz 2.2.1's TemplateV2Module (stock/src/boltz/model/modules/trunkv2.py:361-517), the
template stack Boltz2.forward calls once per trunk pass (recycling_steps + 1 = 4 calls per prediction at the CLI defaults) on EVERY input —
an input that declares no template runs it on the featurizer's one dummy slot (featurizerv2.py:2318-2321, load_dummy_templates_features tdim=1).

Levers (class-level replacements of TemplateV2Module.forward, installed by boltz2_opt.conf; each reads nothing from the environment):

  tfeat   Tier 1 (exact). The recycle-invariant part of the forward — every tensor computed from ``feats`` and ``pair_mask`` alone: the fp32
          featurization block (trunkv2.py:441-474: cdist -> distogram one_hot, frame unit vectors, the cb / frame pair masks, the visibility
          pair mask, the res_type expands, ``a_proj``), ``template_mask`` / ``num_templates`` (:426-428) and the expanded pair mask
          (:477-478) — is computed by the FIRST template call of a ``Boltz2.forward`` and reused by that forward's later calls (the same
          ``feats`` dict and the same ``pair_mask`` tensor reach all of them: boltz2.py:439-467). Exactness: the reused tensors are the first
          call's own; stock's later calls re-issue the same kernels on the same operands, and those kernels (elementwise, cdist, one_hot, a
          fixed-shape fp32 cuBLAS GEMM on one stream) reproduce run to run — the premise of the exact tier's file identity everywhere. The cache
          is keyed by the forward's generation (bumped at every Boltz2.forward entry) and dropped at its exit: nothing survives into the next item.
  tdummy  Tier 1 by algebra, INPUT-CONDITIONAL (selected per mode row). When no template slot of
          the batch is live by the model's own gate (``template_mask = feats['template_mask'].any(dim=2)``, :424 — all zero <=> every slot is the
          featurizer's dummy or maps no token), the module's output ``u`` is +0.0 in every element: v is finite, ``v * template_mask`` is ±0.0,
          ``.sum(dim=1)`` accumulates from +0.0 (=> +0.0) or copies (=> ±0.0), ``/ 1.0`` keeps it, ``relu`` keeps ±0.0 (ATen clamp: -0.0 < 0 is
          false), ``u_proj`` = Linear(64 -> 128, bias=False) whose MMA accumulates ±0.0 products onto +0.0 => +0.0.
          So stock's ``z = z + u`` is ``z + (+0.0)``: z with -0.0 normalised to +0.0. The lever replaces the call by (a) the eight
          ``torch.rand`` draws its 2-layer pairformer's get_dropout_mask calls issue per call (dropout.py:30: ``torch.rand(v.shape, float32)``
          with v = z[:, :, 0:1, 0:1] for tri_mul_out / tri_mul_in / tri_att_start and z[:, 0:1, :, 0:1] for tri_att_end, pairformer.py:172-196;
          the kit's ``resid`` lever keeps the same draws, boltz_trunk_levers.py:86-88) — same shapes, dtype, device and order, so the CUDA
          Philox offset advances exactly as in stock and every later draw (trunk masks, diffusion noise) is unchanged — and (b) a zeros tensor of
          u's shape and dtype (the autocast dtype when autocast is on, as ``u_proj`` returns), so the caller's add is the same ``z + 0.0`` pass.
          The liveness predicate is read once per forward (one host sync at the first template call, before the trunk's heavy work).

STATS (module-level counters the adapter reports): calls, featurize_computed / featurize_reused (tfeat), elided / live (tdummy).
The featurization statements below are boltz 2.2.1's own (MIT), transcribed statement for statement.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch.nn.functional import one_hot

LEVERS = ("tfeat", "tdummy")
STATS: Dict[str, int] = {"calls": 0, "featurize_computed": 0, "featurize_reused": 0, "elided": 0, "live": 0, "forwards": 0}
_STATE: Dict[str, Any] = {"installed": (), "orig_forward": None, "orig_model_forward": None, "gen": 0}
_CACHE_ATTR = "_conf_template_cache"      # per-instance, per-forward cache (a plain dict in the module's __dict__; never a Parameter / buffer)


def _cnt(k: str, n: int = 1) -> None:
    STATS[k] = STATS.get(k, 0) + n


# ------------------------------------------------------------------------------------------------------------------------------------------
# the recycle-invariant featurization: trunkv2.py:415-478 statement for statement (only `z` is absent — nothing here reads it)
# ------------------------------------------------------------------------------------------------------------------------------------------
def featurize(self, feats: dict, pair_mask: torch.Tensor) -> dict:
    res_type = feats["template_restype"]
    frame_rot = feats["template_frame_rot"]
    frame_t = feats["template_frame_t"]
    frame_mask = feats["template_mask_frame"]
    cb_coords = feats["template_cb"]
    ca_coords = feats["template_ca"]
    cb_mask = feats["template_mask_cb"]
    visibility_ids = feats["visibility_ids"]
    template_mask = feats["template_mask"].any(dim=2).float()
    num_templates = template_mask.sum(dim=1)
    num_templates = num_templates.clamp(min=1)

    # Compute pairwise masks
    b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]
    b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :]

    b_cb_mask = b_cb_mask[..., None]
    b_frame_mask = b_frame_mask[..., None]

    # Compute asym mask, template features only attend within the same chain
    B, T = res_type.shape[:2]  # noqa: N806
    tmlp_pair_mask = (
        visibility_ids[:, :, :, None] == visibility_ids[:, :, None, :]
    ).float()

    # Compute template features
    with torch.autocast(device_type="cuda", enabled=False):
        # Compute distogram
        cb_dists = torch.cdist(cb_coords, cb_coords)
        boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1)
        boundaries = boundaries.to(cb_dists.device)
        distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
        distogram = one_hot(distogram, num_classes=self.num_bins)

        # Compute unit vector in each frame
        frame_rot = frame_rot.unsqueeze(2).transpose(-1, -2)
        frame_t = frame_t.unsqueeze(2).unsqueeze(-1)
        ca_coords = ca_coords.unsqueeze(3).unsqueeze(-1)
        vector = torch.matmul(frame_rot, (ca_coords - frame_t))
        norm = torch.norm(vector, dim=-1, keepdim=True)
        unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
        unit_vector = unit_vector.squeeze(-1)

        # Concatenate input features
        a_tij = [distogram, b_cb_mask, unit_vector, b_frame_mask]
        a_tij = torch.cat(a_tij, dim=-1)
        a_tij = a_tij * tmlp_pair_mask.unsqueeze(-1)

        res_type_i = res_type[:, :, :, None]
        res_type_j = res_type[:, :, None, :]
        res_type_i = res_type_i.expand(-1, -1, -1, res_type.size(2), -1)
        res_type_j = res_type_j.expand(-1, -1, res_type.size(2), -1, -1)
        a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
        a_tij = self.a_proj(a_tij)

    # Expand mask
    pair_mask = pair_mask[:, None].expand(-1, T, -1, -1)
    pair_mask = pair_mask.reshape(B * T, *pair_mask.shape[2:])
    return {"a_tij": a_tij, "pair_mask": pair_mask, "template_mask": template_mask, "num_templates": num_templates, "B": B, "T": T}


def _tail(self, z, c: dict, use_kernels: bool):
    """trunkv2.py:480-494 on the (cached or fresh) featurization `c`."""
    B, T = c["B"], c["T"]
    # Compute input projections
    v = self.z_proj(self.z_norm(z[:, None])) + c["a_tij"]
    v = v.view(B * T, *v.shape[2:])
    v = v + self.pairformer(v, c["pair_mask"], use_kernels=use_kernels)
    v = self.v_norm(v)
    v = v.view(B, T, *v.shape[1:])

    # Aggregate templates
    template_mask = c["template_mask"][:, :, None, None, None]
    num_templates = c["num_templates"][:, None, None, None]
    u = (v * template_mask).sum(dim=1) / num_templates.to(v)

    # Compute output projection
    u = self.u_proj(self.relu(u))
    return u


def _cached(self, feats, pair_mask, want_live: bool) -> dict:
    """The forward's featurization cache: computed at the first template call of the current Boltz2.forward generation, reused after."""
    key = (_STATE["gen"], id(feats), pair_mask.data_ptr(), tuple(pair_mask.shape))
    c = self.__dict__.get(_CACHE_ATTR)
    if c is not None and c["key"] == key:
        _cnt("featurize_reused")
        return c
    c = {"key": key}
    if want_live:                                   # tdummy's predicate: one host sync per forward, at its first template call
        c["live"] = bool(feats["template_mask"].any().item())
    self.__dict__[_CACHE_ATTR] = c
    return c


def _replay_dropout_draws(self, BT: int, N: int, device) -> None:
    """The torch.rand calls the skipped 2-layer template pairformer would have issued (pairformer.py:172-196 via dropout.py:30), in order."""
    for _layer in self.pairformer.layers:
        for columnwise in (False, False, False, True):      # tri_mul_out, tri_mul_in, tri_att_start (rowwise); tri_att_end (columnwise)
            shape = (BT, 1, N, 1) if columnwise else (BT, N, 1, 1)
            torch.rand(shape, dtype=torch.float32, device=device)
    _cnt("rng_draws_replayed", 4 * len(self.pairformer.layers))


def _u_dtype(self) -> torch.dtype:
    """The dtype stock's `u_proj` (a Linear) returns: the autocast dtype under autocast, else the weight's."""
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return self.u_proj.weight.dtype


def forward_conf(self, z, feats, pair_mask, use_kernels: bool = False):
    """TemplateV2Module.forward under the installed CONF levers (tfeat and/or tdummy); stock statement order otherwise."""
    levers = _STATE["installed"]
    if self.training or not levers:
        return _STATE["orig_forward"](self, z, feats, pair_mask, use_kernels)
    _cnt("calls")
    tdummy = "tdummy" in levers; tfeat = "tfeat" in levers
    c = _cached(self, feats, pair_mask, want_live=tdummy)
    if tdummy and not c["live"]:
        B, T = feats["template_restype"].shape[:2]; N = z.shape[1]
        _replay_dropout_draws(self, B * T, N, z.device)
        _cnt("elided")
        return torch.zeros((B, N, z.shape[2], self.u_proj.out_features), dtype=_u_dtype(self), device=z.device)
    if tdummy:
        _cnt("live")
    if tfeat:
        if "a_tij" not in c:
            c.update(featurize(self, feats, pair_mask)); _cnt("featurize_computed")
        return _tail(self, z, c, use_kernels)
    fresh = featurize(self, feats, pair_mask); _cnt("featurize_computed")
    return _tail(self, z, fresh, use_kernels)


def begin_forward() -> int:
    """A new cache generation (called at every Boltz2.forward entry by the installed wrapper; tests call it directly)."""
    _STATE["gen"] += 1; _cnt("forwards")
    return _STATE["gen"]


def end_forward(model) -> None:
    """Drop the template cache of `model` (called at Boltz2.forward exit: nothing outlives the item)."""
    tm = getattr(model, "template_module", None)
    if tm is not None:
        getattr(tm, "_orig_mod", tm).__dict__.pop(_CACHE_ATTR, None)


def _model_forward(self, feats, *args, **kwargs):
    """Boltz2.forward wrapper: a new cache generation per forward; the template cache dropped at exit."""
    begin_forward()
    try:
        return _STATE["orig_model_forward"](self, feats, *args, **kwargs)
    finally:
        end_forward(self)


def install(levers, model_hook: bool = True) -> tuple:
    """Install the named template levers (subset of LEVERS) as class-level replacements; idempotent, additive. ``model_hook`` wraps
    Boltz2.forward with the cache-generation bump (the worker always; a CPU unit test without pytorch_lightning passes False and calls
    begin_forward / end_forward itself). Returns the installed set."""
    want = tuple(l for l in LEVERS if l in set(levers))
    if not want:
        return _STATE["installed"]
    from boltz.model.modules import trunkv2 as T2
    if _STATE["orig_forward"] is None:
        _STATE["orig_forward"] = T2.TemplateV2Module.forward
        T2.TemplateV2Module.forward = forward_conf
    if model_hook and _STATE["orig_model_forward"] is None:
        from boltz.model.models import boltz2 as B2
        _STATE["orig_model_forward"] = B2.Boltz2.forward
        B2.Boltz2.forward = _model_forward
    _STATE["installed"] = tuple(l for l in LEVERS if l in set(_STATE["installed"]) | set(want))
    return _STATE["installed"]


def uninstall() -> None:
    from boltz.model.modules import trunkv2 as T2
    if _STATE["orig_forward"] is not None:
        T2.TemplateV2Module.forward = _STATE["orig_forward"]; _STATE["orig_forward"] = None
    if _STATE["orig_model_forward"] is not None:
        from boltz.model.models import boltz2 as B2
        B2.Boltz2.forward = _STATE["orig_model_forward"]; _STATE["orig_model_forward"] = None
    _STATE["installed"] = ()


def report() -> Dict[str, Any]:
    return {"installed": list(_STATE["installed"]), "stats": dict(STATS)}
