"""The Evo 2 exact kit: ``apply(evo2_model)`` on a constructed ``evo2.Evo2`` installs the levers of the model's route — read from the object,
never from a name: ``vortex-kernels`` (``Evo2(..., use_kernels=True)``), ``torch-conv`` (the constructor's defaults), or, on a stack without
Transformer Engine, ``bf16-projections`` (upstream's own bf16 fallback) — and, on a model split over two devices, the pipelined scoring
surface. Nothing runs at apply but the patching (the fold of the projection weights in place, class attributes wrapped); the caches and
buffers fill on the first forward of each (batch, length) exactly as the stock computes them, and are reused after. Every scoring forward
then takes the kit path (bitwise the stock's logits); cached generation calls and hooked forwards take the stock path per call, by name.
One line at apply (``[evo2-kit] APPLIED …``), one at exit (``[evo2-kit] EXIT …``)."""
from __future__ import annotations

import atexit
import sys
import time

from evo2_opt.kit.errors import KitRefused   # noqa: F401 — re-exported

PREFIX = "[evo2-kit]"
_state = {"applied": False, "record": None, "forwards": {"scoring": 0, "generation": 0}, "exit_registered": False}


def _te_importable() -> bool:
    import importlib.util
    return importlib.util.find_spec("transformer_engine") is not None


def _count_forwards() -> None:
    """Wrap vortex's StripedHyena.forward (the class attribute, outermost: over the gate when one is installed) to count scoring / generation
    calls for the EXIT line. Not a torch hook: a registered hook is exactly what sends a forward to the stock path (hooks.py), and Evo2.forward
    calls model.forward directly, past torch's hooks."""
    import functools
    import vortex.model.model as _vmm
    inner = _vmm.StripedHyena.forward
    if getattr(inner, "__evo2_kit_counter__", False):
        return

    @functools.wraps(inner)
    def forward(self, x, inference_params_dict=None, padding_mask=None, *args, **kwargs):
        _state["forwards"]["generation" if inference_params_dict is not None else "scoring"] += 1
        return inner(self, x, inference_params_dict, padding_mask, *args, **kwargs)
    forward.__evo2_kit_counter__ = True
    forward.__wrapped_stock__ = getattr(inner, "__wrapped_stock__", inner)
    _vmm.StripedHyena.forward = forward


def apply(evo2_model, *, model_name: str | None = None, log=None) -> dict:
    """Install the route's levers on ``evo2_model`` (an ``evo2.Evo2``); returns the record {route, levers, not_applicable, pipeline, apply_s, …}.
    Raises KitRefused (nothing installed) when a launch condition does not hold — the caller names it and the model runs as constructed."""
    log = log or sys.stderr
    if _state["applied"]:
        raise KitRefused("the kit is already applied in this process (one Evo2 model per process)")
    import torch
    model = evo2_model.model
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        raise KitRefused("the kit's levers are CUDA kernels: the model is not on a CUDA device")
    t0 = time.perf_counter()
    if _te_importable():
        from evo2_opt.kit import routes as R
        info = R.apply(model)
        route, levers, na = info["route"], list(info["levers"]), dict(info.get("not_applicable") or {})
        from evo2_opt.kit import pipelined
        P = pipelined.install(evo2_model, R.ensure_shape)
        pipeline_words = pipelined.words()
        counters = R.counters
    else:
        from evo2_opt.kit import bf16
        info = bf16.apply(evo2_model)
        route, levers, na = f"{info['route']}({info['conv_path']})", list(info["levers"]), dict(info["not_applicable"])
        P = {"ok": False, "why": "the bf16 route holds one device"}
        pipeline_words = "off(one device)"
        counters = bf16.counters
    from evo2_opt.kit import batch as _batch
    levers.append(_batch.install())                   # S1: evo2.scoring.prepare_batch's token rows as numpy arrays (every route)
    _count_forwards()
    n_dev = len({str(p.device) for p in model.parameters()})
    name = model_name or info.get("facts", {}).get("model_key") or f"hidden{int(model.config.hidden_size)}x{len(list(model.blocks))}blocks"
    rec = {"applied": True, "model": name, "route": route, "levers": levers, "not_applicable": na, "pipeline": bool(P.get("ok")), "pipeline_words": pipeline_words,
           "devices": n_dev, "gpu": torch.cuda.get_device_name(dev), "apply_s": round(time.perf_counter() - t0, 3), "counters": counters}
    _state.update(applied=True, record=rec)
    na_words = ("; not applicable on this route: " + ", ".join(sorted(na))) if na else ""
    print(f"{PREFIX} APPLIED model={name} route={route} levers={len(levers)} ({', '.join(levers)}){na_words}; pipeline={pipeline_words}; "
          f"gpu={rec['gpu']}x{n_dev}; apply {rec['apply_s']:.2f}s — every scoring forward on the kit path (caches fill on the first call of a shape); "
          f"generation and hooked calls on the stock path per call, named", file=log, flush=True)
    if not _state["exit_registered"]:
        atexit.register(exit_line, log)
        _state["exit_registered"] = True
    return rec


def record() -> dict | None:
    return _state["record"]


def tally() -> dict:
    """The EXIT line's numbers: forwards by kind, the gate's kit / stock-path counts by reason, the pipelined windows."""
    rec = _state["record"]
    if rec is None:
        return {"applied": False}
    c = {k: int(v) for k, v in (rec["counters"]() or {}).items()}
    forwards = dict(_state["forwards"])
    stock_reasons = {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("passthrough:")}
    stock_path = c.get("gate_passthrough", sum(stock_reasons.values()))
    kit_path = c["gate_kit"] if "gate_kit" in c else forwards["scoring"]           # the bf16 route has no gate: every scoring forward is the kit path (its generation calls are counted as passthrough:generation_call)
    out = {"applied": True, "model": rec["model"], "route": rec["route"], "forwards": forwards,
           "kit_path": kit_path, "stock_path": stock_path, "stock_path_reasons": stock_reasons}
    if rec["pipeline"]:
        from evo2_opt.kit import pipelined
        out["pipelined"] = pipelined.tally()
    return out


def exit_line(log=None) -> str:
    t = tally()
    if not t.get("applied"):
        return ""
    f = t["forwards"]
    reasons = ",".join(f"{k}:{v}" for k, v in sorted(t["stock_path_reasons"].items())) or "-"
    kit_path = t["kit_path"] if t["kit_path"] is not None else f["scoring"]          # the bf16 route has no gate: every scoring forward is the kit path
    pipe = f" pipelined_windows={t['pipelined']['pipelined_windows']}" if "pipelined" in t else ""
    line = (f"{PREFIX} EXIT model={t['model']} route={t['route']} forwards={f['scoring'] + f['generation']} (scoring={f['scoring']} generation={f['generation']}) "
            f"kit_path={kit_path} stock_path={t['stock_path']} ({reasons}){pipe}")
    print(line, file=log or sys.stderr, flush=True)
    return line
