"""boltz2_opt.templskip — the dummy-template elision (registry lever ``templ_skip``; exact class).

Boltz-2 runs its template module on EVERY input: an input that declares no template (or whose declared templates all became dummy slots) gets
dummy template features (``load_dummy_templates_features``: every ``template_mask`` slot all zero), and ``TemplateV2Module.forward`` still
projects and embeds them, runs its two C=64 Pairformer blocks over the N x N pair rows (every recycling pass) and only then aggregates under the
mask — ``u = (u * template_mask).sum(dim=1) / num_templates.clamp(min=1)`` is the zero tensor [B, N, N, template_dim] whenever no slot is set,
whatever the Pairformer and ``v_norm`` computed per slot — and applies its tail ``u = u_proj(relu(u))``: the output projection of zeros (its
bias per channel), which the trunk adds (``z = z + template_module(...)``).  This lever serves that update by running ONLY the tail on the zero
tensor the aggregation produces (same dtype, same [B, N, N, template_dim] shape, the module's own ``relu`` / ``u_proj``: the same kernels on the
same values — ``_elided_update``), replays the random draws the Pairformer's dropout masks issue on the CUDA generator (four ``torch.rand``
per layer, in eval mode too — the diffusion sampler's noise downstream depends on that generator state: ``_replay_dropout_draws``), and skips
the input projections, the Pairformer and ``v_norm``.  A pass with any template slot set
runs the module exactly as stock; so does a pass whose ``feats`` carry no ``template_mask`` (counted ``no_mask``) and a call issued inside a
CUDA-graph capture (``torch.cuda.is_current_stream_capturing()``: no host read of the mask there; counted ``capturing`` — boltz2_opt.graph does
not capture the template module today, the branch is defensive).

Per call ONE decision, counted: ``skipped`` (no slot set: the tail-only update served), ``live`` (a slot set: the module ran), ``capturing``,
``no_mask``, ``errors`` (the decision or the tail raised: the module ran; the gate refuses).  The decision reads B x T x N mask values on the
host once per trunk pass (recycling_steps + 1 per item).

Tier 1 (bitwise): the elided passes contribute the tensor stock's own tail statements compute on the same zero input; templated passes are
stock's statements.  Attached by ``boltz2_opt.worker_launch --attach templskip`` at the worker's ``boltz.model.models.boltz2`` import, AFTER
``graph`` (the trunk CUDA-graph lever's `templ` unit restates ``TemplateV2Module.forward`` with its distogram boundaries hoisted and pins the
STOCK source at its apply — this wrapper therefore goes on top of that body, never under it): a class-level patch of
``TemplateV2Module.forward`` and, when present, ``TemplateModule.forward`` (idempotent); the wrapped body runs for every pass this lever does
not elide.  Under ``--n_gpu P`` the row-sharded trunk builds each rank's template rows with its own statement (rowpair._template_rows) and never
calls the class forward: the lever leaves that line by name (modes.TP_DROPS).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

TAG = "boltz2-opt"
NAME = "templ_skip"
SWITCH = "BOLTZ_TEMPL_SKIP"
VARIANTS = ("1",)
LEVERS = ("templ_skip",)                                              # the registry lever this adapter installs (worker_launch / stack.attachment_problems read it)
CLASSES = ("TemplateV2Module", "TemplateModule")                      # boltz.model.modules.trunkv2 classes whose forward(z, feats, pair_mask, use_kernels) returns the template update u
_STATE: Dict[str, Any] = {"applied": [], "patched": [], "variant": None, "orig": {},
                          "census": {"calls": 0, "skipped": 0, "live": 0, "capturing": 0, "no_mask": 0, "errors": 0}, "shapes": {}}


def variant(environ=None) -> Optional[str]:
    env = os.environ if environ is None else environ
    v = env.get(SWITCH, "").strip().lower()
    return v if v in VARIANTS else None


def requested(environ=None) -> bool:
    return variant(environ) is not None


def dummy_pass(feats) -> Optional[bool]:
    """True when no template slot of the pass is set (``feats["template_mask"]`` all zero: the module's aggregation is the zero tensor and its
    update the output projection of zeros, _elided_update), False when any slot is set (the module runs), None without a template mask (the module runs)."""
    tm = feats.get("template_mask") if isinstance(feats, dict) else None
    if tm is None:
        return None
    return not bool(tm.any())


DRAWS_PER_LAYER = ((0, "row"), (1, "row"), (2, "row"), (3, "col"))       # boltz PairformerNoSeqLayer.forward (eval mode too): get_dropout_mask before tri_mul_out, tri_mul_in,
                                                                      # tri_att_start (row view [B*T, N, 1, 1]) and tri_att_end (columnwise view [B*T, 1, N, 1]) — each one
                                                                      # torch.rand(view.shape, dtype=float32, device) on the CUDA generator


def _replay_dropout_draws(module, bt: int, n: int, device) -> int:
    """Issue the random draws the skipped Pairformer would have issued, in its order: boltz's get_dropout_mask draws
    ``torch.rand(v.shape, dtype=torch.float32, device=v.device)`` four times per PairformerNoSeqLayer (also in eval mode: the mask is all
    ones then, but the generator advances), v = z[:, :, 0:1, 0:1] (three times) and z[:, 0:1, :, 0:1] (once) of the template stack's
    z [B*T, N, N, template_dim].  The same calls advance the CUDA Philox offset by the same amount, so every later consumer (the diffusion
    sampler's noise first of all) sees the generator state stock leaves.  Returns the number of draws issued."""
    import torch
    layers = getattr(getattr(module, "pairformer", None), "layers", None)
    n_layers = len(layers) if layers is not None else 0
    k = 0
    for _ in range(n_layers):
        for _i, kind in DRAWS_PER_LAYER:
            torch.rand((bt, n, 1, 1) if kind == "row" else (bt, 1, n, 1), dtype=torch.float32, device=device)
            k += 1
    return k


def _elided_update(module, z, template_mask):
    """The template module's update for a pass whose template mask has no slot set, computed by the module's OWN output projection on the
    tensor its aggregation produces there — without the input projections, the Pairformer and ``v_norm`` — after replaying the Pairformer's
    random draws (_replay_dropout_draws).  Stock (boltz 2.2.1 trunkv2.py TemplateV2Module / TemplateModule.forward):
    ``v = v_norm(v + pairformer(v))`` per template slot, then ``u = (v * template_mask).sum(dim=1) / num_templates.to(v)`` — the zero tensor
    [B, N, N, template_dim] when no slot is set (``v`` is finite) — then ``u = u_proj(relu(u))``: the projection of zeros, i.e. its bias per
    channel (or zeros without one).  Served here: that zero tensor, in the dtype stock's aggregation has (``v_norm``'s output — fp32 under
    the trunk's autocast, the parameters' dtype otherwise — promoted with the mask's), through the module's own ``relu`` / ``u_proj`` on the
    same [B, N, N, template_dim] shape: the same kernels on the same values, bit for bit, for one N^2 x template_dim x token_z projection
    instead of the whole template stack."""
    import torch
    B, N = int(z.shape[0]), int(z.shape[1])
    T = int(template_mask.shape[1]) if (torch.is_tensor(template_mask) and template_mask.dim() >= 2) else 1
    _replay_dropout_draws(module, B * T, N, z.device)
    dt_v = torch.float32 if (z.is_cuda and torch.is_autocast_enabled()) else module.u_proj.weight.dtype   # v_norm's output dtype (autocast runs layer_norm in fp32)
    dt_u = torch.promote_types(dt_v, template_mask.dtype) if torch.is_tensor(template_mask) else dt_v
    u = torch.zeros((B, N, int(z.shape[2]), int(module.u_proj.in_features)), dtype=dt_u, device=z.device)
    return module.u_proj(module.relu(u))


def _make_forward(orig):
    import torch
    C = _STATE["census"]

    def forward(self, z, feats, pair_mask, use_kernels: bool = False, *a, **kw):
        C["calls"] += 1
        try:
            if z.is_cuda and torch.cuda.is_current_stream_capturing():   # inside a CUDA-graph capture (boltz2_opt.graph's templ unit): no host read; the module runs, captured
                C["capturing"] += 1
                return orig(self, z, feats, pair_mask, use_kernels, *a, **kw)
            d = dummy_pass(feats)
        except Exception as e:  # noqa: BLE001 — the decision itself failed: counted (the gate refuses), the module runs
            C["errors"] += 1
            if C["errors"] <= 2:
                sys.stderr.write(f"[{TAG}] {NAME}: decision error {type(e).__name__}: {e} — the template module runs\n")
            return orig(self, z, feats, pair_mask, use_kernels, *a, **kw)
        if d is None:
            C["no_mask"] += 1
            return orig(self, z, feats, pair_mask, use_kernels, *a, **kw)
        if not d:
            C["live"] += 1                                            # a templated pass: the module runs exactly as stock
            return orig(self, z, feats, pair_mask, use_kernels, *a, **kw)
        C["skipped"] += 1
        key = "x".join(str(int(n)) for n in z.shape[1:])
        _STATE["shapes"][key] = _STATE["shapes"].get(key, 0) + 1
        try:
            return _elided_update(self, z, feats["template_mask"])         # the module's update for an all-dummy pass, by the module's own tail statements (_elided_update)
        except Exception as e:  # noqa: BLE001 — a module without the stock tail attributes: counted (the gate refuses), the module runs
            C["errors"] += 1; C["skipped"] -= 1
            if C["errors"] <= 2:
                sys.stderr.write(f"[{TAG}] {NAME}: elision error {type(e).__name__}: {e} — the template module runs\n")
            return orig(self, z, feats, pair_mask, use_kernels, *a, **kw)

    return forward



def apply(spec: Optional[str] = None) -> List[str]:
    """Patch the template module classes' forward class-wide (idempotent). ``spec`` overrides the env switch."""
    if _STATE["applied"]:
        return list(_STATE["applied"])
    v = variant() if spec is None else (str(spec).strip().lower() if str(spec).strip().lower() in VARIANTS else None)
    if v is None:
        return []
    import boltz.model.modules.trunkv2 as T2
    patched = []
    for cname in CLASSES:
        cls = getattr(T2, cname, None)
        if cls is None or not hasattr(cls, "forward"):
            continue
        orig = cls.forward
        _STATE["orig"][cname] = orig
        cls.forward = _make_forward(orig)
        patched.append(f"{cname}.forward")
    if not patched:
        sys.stderr.write(f"[{TAG}] {NAME}: no template module class among {CLASSES} in boltz.model.modules.trunkv2 — nothing installed\n")
        return []
    _STATE.update(applied=list(LEVERS), patched=patched, variant=v)
    sys.stderr.write(f"[{TAG}] {NAME} APPLIED variant={v} patched={','.join(patched)}\n")
    import atexit
    atexit.register(lambda: sys.stderr.write((line() or "") + "\n"))
    return list(LEVERS)


def census() -> Dict[str, Any]:
    C = dict(_STATE["census"])
    return {**C, "served": C["skipped"], "templated_passes": C["live"], "shapes": dict(_STATE["shapes"])}   # served = the untemplated passes elided (stack.evidence reads `served`); templated_passes = the passes that ran the module (a slot set)


def verdict() -> Dict[str, Any]:
    """Fail-closed: refused on any decision error, or when installed and no call arrived over a pass that ran the trunk; idle = every call
    live / capturing / without a mask (installed, the input never met an all-dummy eager pass)."""
    if not _STATE["applied"]:
        return {"ok": False, "idle": False, "reason": "not applied"}
    C = _STATE["census"]
    if C["errors"]:
        return {"ok": False, "idle": False, "reason": f"errors={C['errors']}"}
    if C["calls"] > 0 and C["skipped"] == 0:
        return {"ok": True, "idle": True, "reason": "every call live, capturing or without a template mask"}
    return {"ok": True, "idle": False, "reason": None}


def line() -> Optional[str]:
    if not _STATE["applied"]:
        return None
    C = _STATE["census"]; g = verdict()
    return (f"[{TAG}] LEVER name={NAME} state=on variant={_STATE['variant']} calls={C['calls']} served={C['skipped']} templated_passes={C['live']} "
            f"skipped={C['skipped']} capturing={C['capturing']} no_mask={C['no_mask']} errors={C['errors']} idle={g['idle']} gate={'ok' if g['ok'] else 'refused:' + str(g['reason'])}")   # served = the untemplated passes elided; templated_passes = the passes with a template slot set (the module ran, as stock)


def report() -> Dict[str, Any]:
    return {"applied": list(_STATE["applied"]), "disabled": {}, "variant": _STATE["variant"], "line": line(), "census": census(),
            "gate": verdict() if _STATE["applied"] else None, "patched": list(_STATE["patched"])}
