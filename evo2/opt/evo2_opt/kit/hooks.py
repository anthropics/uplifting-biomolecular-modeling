"""The forward-hook rule of the gate: a model carrying user forward hooks (Evo2.forward(return_embeddings=True)) takes the stock path by name for that call; a hook on a folded projection is refused by name (its output layout is the folded one)."""
from __future__ import annotations

import weakref

from evo2_opt.kit import gate as GT


class KitRefusedHooks(RuntimeError):
    pass


_hook_dicts: dict = {}            # id(model) -> (weakref to the model, [every submodule's forward / forward-pre hook dict]): the module tree is fixed after
                                  # construction and torch registers hooks INTO these dicts, so one truth test per dict finds any later hook


def hooked_modules(model) -> list:
    """[(name, module)] of every submodule with a registered forward or forward-pre hook (torch.nn.Module's own hook dicts)."""
    ent = _hook_dicts.get(id(model))
    if ent is None or ent[0]() is not model:
        ent = _hook_dicts[id(model)] = (weakref.ref(model), [d for _, m in model.named_modules()
                                                              for d in (getattr(m, "_forward_hooks", None), getattr(m, "_forward_pre_hooks", None)) if d is not None])
    if not any(ent[1]):
        return []
    out = []
    for name, m in model.named_modules():
        if getattr(m, "_forward_hooks", None) or getattr(m, "_forward_pre_hooks", None):
            out.append((name, m))
    return out


def folded_projection_hooks(hooked: list) -> list:
    """The hooked modules that are folded projections (W1 marks them ``_v40_w1``; the L2 fold changed their output layout)."""
    return [name for name, m in hooked if getattr(m, "_v40_w1", False) or name.endswith(".projections")]


def decide_hooks(model) -> tuple:
    """('kit', None) when no hook is registered; ('stock', 'hooks|…') for a passthrough; raises KitRefusedHooks for a hook on a folded projection."""
    hooked = hooked_modules(model)
    if not hooked:
        return "kit", None
    bad = folded_projection_hooks(hooked)
    if bad:
        raise KitRefusedHooks(f"stock-only (kit not applicable): a forward hook on a folded projection {bad} — the L2 fold changes the projection's output layout; "
                              f"run the package without the kit for these layer names (return_embeddings)")
    names = [n for n, _ in hooked]
    return "stock", f"hooks|{len(hooked)} forward hooks (return_embeddings): {names[:4]}"


def decide(model, B, L, inference_params_dict, padding_mask) -> tuple:
    route, reason = decide_hooks(model)
    if route == "kit":
        route, reason = GT.decide_route(B, L, inference_params_dict, padding_mask)
    return route, reason


def forward_gate(orig_forward, gate: GT.Gate, manager: GT.ShapeManager):
    """The model forward's gate with the hook decision FIRST, then the route decision."""
    f = GT.route_forward(orig_forward, gate, manager, decide)
    f.__hooks_gate__ = True
    return f
