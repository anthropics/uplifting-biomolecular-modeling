"""kits/eager — the E1 kit: ONE lever set, applied to EVERY forward of the model, eagerly. `apply(model, size=...)` on the stock model
(fp32 weights, before its first forward) puts in force, in order:

    rmsnorm_autotune_pin      the hub RMSNorm Triton kernel pinned to ONE num_warps before the first forward (kits/pins.py: the size's cell
                              on the card's class; a card of no censused cell takes its compute-capability class's default with one named line)
    ew1:*                     fused Triton elementwise kernels at the stock's own rounding points (clamp+RoPE, silu*gate, add+RMSNorm, embedding)
    attn:A1, A2, A3           flash-attn varlen with shape-derived cu_seqlens (packing computed once per forward), the index-only block mask,
                              the global layers' fixed flex kernel options (A3: per GPU class — kits/v1_2 — off by name where not bit-exact)
    P1_precast                every Linear's weight / bias cast to bf16 ONCE (what autocast would produce per call): the model is CONSUMED
    ew1_i64_offsets           ew1's kernels with int64 offsets (kits/ew1_i64) + the index-bound guard (guard.py): no silent wrong element
                              above 2^31 elements of the GLU intermediate
    anylen_flex               the global layers' flex call served at every sequence length (anylen.py: one dynamic compile with A3's tiling
                              at >= 128 query tokens, the stock's own compiled object below)
    multiseq:docmask, flex_opts  rows of several sequences / padded rows (kits/multiseq): the block-causal document mask built from block
                              statistics (the stock's BlockMask, tensor for tensor, without the B x L x L evaluation) and the pinned flex
                              callable on the adapter's fallback route (the same tiling as the dense route: bitwise on document masks)
    kvcache:views, noappend_within, pack_global  forwards over a prefilled DynamicCache (kits/kvcache; upstream's context reuse in
                              retrieval-augmented scoring): the cache repeated as a view, within-sequence layers on the query's own K/V, the
                              global layers' packed [context ; query] K/V written once into flash_attn_varlen_func's layout with the row
                              layout computed once per forward — the stock's kernel calls on the stock's argument values, without the copies
    tokenmemo                 host side (kits/tokenmemo): E1BatchPreparer.prepare_multiseq tokenises a context once per job, not once per row

— the composition kits/v1 (through kits/v1_2's class table) plus the routing pieces and the three multi-sequence / cache / host
components, nothing else. Every engaged lever reproduces the
stock's bits (A3 is keyed by GPU class for that reason: kits/v1_2); the kit adds no arithmetic of its own. A lever that cannot run here (a compile / launch failure, a
component missing, weights not fp32) is the KIT's refusal by name (KitRefused, everything already applied rolled back) — never a subset.
Printed at apply, one fact per line: `[e1-opt] KIT v2.0 mode=eager …` then one `[e1-opt] LEVER name=… state=on|off …` per lever.
"""
from __future__ import annotations

import time

from .. import ew1_i64, kvcache, multiseq, pins, tokenmemo, v1, v1_2
from . import anylen, guard

KIT = "v2.0"
MODE = "eager"
LINE_PREFIX = "[e1-opt]"
A3_LEVER = "attn:A3_flex_kernel_options"
MULTISEQ_LEVERS = tuple(f"multiseq:{n}" for n in multiseq.LEVERS)      # non-dense forwards (multi-sequence / padded rows): document block mask from block statistics + the pinned flex callable on the fallback route
KVCACHE_LEVERS = tuple(f"kvcache:{n}" for n in kvcache.LEVERS)         # prefilled-cache forwards (retrieval-augmented scoring): cache views, no re-append, packed global attention written once
TOKENMEMO_LEVERS = ("tokenmemo",)                                      # host side: a context is tokenised once per job, not once per row
LEVERS = ("rmsnorm_autotune_pin",) + tuple(v1.LEVERS) + ("ew1_i64_offsets", "anylen_flex") + MULTISEQ_LEVERS + KVCACHE_LEVERS + TOKENMEMO_LEVERS
TESTED_SHAPES = v1_2.TESTED_SHAPES
KitRefused = v1.KitRefused

_BLANK = {"applied": False, "record": None, "model_id": None}
_state = dict(_BLANK)


def kit_files() -> dict:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    files = {n: os.path.isfile(os.path.join(here, n)) for n in ("__init__.py", "anylen.py", "guard.py")}
    for comp in ("multiseq", "kvcache", "tokenmemo"):
        files[comp + "/__init__.py"] = os.path.isfile(os.path.join(os.path.dirname(here), comp, "__init__.py"))
    return files


def counters_snapshot() -> dict:
    return {"v1": v1_2.counters_snapshot(), "guard": guard.counters(), "anylen": anylen.counters(), "multiseq": multiseq.counters(), "kvcache": kvcache.counters(),
            "tokenmemo": tokenmemo.counters()}


def counters_delta(prev: dict) -> dict:
    return {"v1": v1_2.counters_delta(prev["v1"]), "guard": guard.counters(), "anylen": anylen.counters(), "multiseq": multiseq.counters(), "kvcache": kvcache.counters(),
            "tokenmemo": tokenmemo.counters()}


def expected_per_forward(model, B: int, L: int) -> dict:
    return v1_2.expected_per_forward(model, B, L)


def _q(text) -> str:
    """A free-text field on a key=value line: double-quoted, inner quotes normalised."""
    return '"' + str(text).replace('"', "'") + '"'


def kit_lines(rec: dict) -> list:
    """The kit's report at apply: the KIT line, then one LEVER line per lever (`state=on`, or `state=off reason="…"` for the one lever a
    class table leaves out — A3 where not bit-exact; a GPU of no tested class engages it and its line NAMES that: `note="untested class …"`)."""
    out = [f"{LINE_PREFIX} KIT {KIT} mode={MODE} size={rec['size']} card={rec.get('gpu_class')} gpu={_q(rec.get('gpu'))} pin=W{rec.get('num_warps')} levers={rec['n_on']}/{len(LEVERS)}"]
    for name in LEVERS:
        st = rec["levers"][name]
        fields = f"name={name}"
        if st.get("note"):
            fields += f" note={_q(st['note'])}"
        if name == "rmsnorm_autotune_pin":
            fields += f" W={rec.get('num_warps')} source={_q(st.get('source', '?'))}"
        if name == "anylen_flex" and st.get("form"):
            fields += f" form={st['form']}"
        out.append(f"{LINE_PREFIX} LEVER {fields} state={'on' if st['on'] else 'off'}" + (f" reason={_q(st['reason'])}" if not st["on"] else ""))
    return out


def apply(model, *, size: str, det: bool = False, require_gpu: bool = True, verbose: bool = True, **_) -> dict:
    """The whole step, once per process (one model per process): kits/v1_2 (pin -> ew1 -> attn -> P1) -> ew1_i64 + the index guard ->
    the any-length flex route -> multiseq -> kvcache -> tokenmemo. Returns the record (also `applied()`); prints the KIT / LEVER lines unless verbose=False. A failure after
    the first component is in force rolls everything back and raises KitRefused naming it."""
    import torch
    if _state["applied"]:
        raise KitRefused(f"kit {KIT}: already applied in this process (one model per process)")
    if size not in pins.WEIGHTS:
        raise KitRefused(f"kit {KIT}: unknown size {size!r} (known: {', '.join(pins.WEIGHTS)})")
    t0 = time.perf_counter()
    pin = pins.apply_autotune_pin(size)                                               # FIRST, before any forward: the size's pin (PinDrift by name on a host that cannot pin; an uncensused card is served its cc class's default with one named line)
    rec12 = v1_2.apply(model, size=size, num_warps=int(pin["num_warps"]), require_gpu=require_gpu, det=det)   # KitRefused / PinDrift propagate by name (nothing in force yet, or v1 rolled itself back)
    try:
        from ..ew1 import patches as _ew1p
        cfg = model.config
        i64 = ew1_i64.install()
        grec = guard.install(model, _ew1p._state, int(cfg.hidden_size), int(cfg.intermediate_size), bound=guard.INT64_BOUND)
        frec = anylen.install("hybrid")
        mrec = multiseq.install()                                                     # over the adapter: document masks from block statistics, the pinned flex callable on the fallback route
        krec = kvcache.install(model)                                                 # over the adapter's routed _flash_attn: the prefilled-cache forward without the cache copies
        trec = tokenmemo.install()                                                    # E1BatchPreparer.prepare_multiseq: a context tokenised once
    except Exception as e:                                                            # a lever of the set cannot be put in force: ALL of it comes off, the kit refuses by name
        for undo in (tokenmemo.uninstall, kvcache.uninstall, multiseq.uninstall, anylen.uninstall, guard.uninstall, ew1_i64.uninstall, v1_2.unapply):
            try:
                undo()
            except Exception:
                pass
        raise KitRefused(f"kit {KIT}: the lever set cannot run on this host — {type(e).__name__}: {e}") from e
    levers = {name: {"on": True} for name in LEVERS}
    levers["rmsnorm_autotune_pin"]["source"] = "default pin (no censused cell)" if pin.get("uncensused") else "size pin"
    if not rec12["A3"]:
        levers[A3_LEVER] = {"on": False, "reason": f"OFF on {rec12['gpu_class']}: {rec12['A3_cite']}"}
    elif rec12.get("A3_untested"):
        levers[A3_LEVER]["note"] = rec12["A3_cite"]
    levers["anylen_flex"]["form"] = frec["form"]
    if not rec12["A3"]:
        levers["multiseq:flex_opts"]["note"] = f"torch default kernel options on {rec12['gpu_class']} (A3 off): the stock's config through the kit's callable"
    rec = {"kit": KIT, "mode": MODE, "size": size, "gpu": rec12.get("gpu"), "gpu_class": rec12.get("gpu_class"), "num_warps": rec12.get("num_warps"),
           "det": bool(det), "pin": {k: pin[k] for k in pin if k in ("num_warps", "uncensused", "gpu", "kernel")}, "levers": levers, "n_on": sum(1 for v in levers.values() if v["on"]), "v1_2": rec12, "ew1_i64": i64, "guard": grec, "anylen": frec, "multiseq": mrec, "kvcache": krec, "tokenmemo": trec,
           "apply_s": round(time.perf_counter() - t0, 3), "torch": torch.__version__}
    rec["lines"] = kit_lines(rec)
    _state.update({"applied": True, "record": rec, "model_id": id(model)})
    if verbose:
        for ln in rec["lines"]:
            print(ln, flush=True)
    return rec


def unapply() -> None:
    """Release everything in LIFO order (tokenmemo, kvcache, multiseq, flex route, guard, ew1_i64, then attn / ew1 through v1_2). P1's weight cast is not reversible:
    reload the checkpoint for a stock model."""
    if not _state["applied"]:
        return
    for undo in (tokenmemo.uninstall, kvcache.uninstall, multiseq.uninstall, anylen.uninstall, guard.uninstall, ew1_i64.uninstall, v1_2.unapply):
        try:
            undo()
        except Exception:
            pass
    _state.clear(); _state.update(_BLANK)


def applied() -> dict | None:
    """The apply record when the kit is in force in this process, else None."""
    return _state["record"] if _state["applied"] else None


def stamp() -> dict:
    if not _state["applied"]:
        raise KitRefused(f"stamp: kit {KIT} is not in force in this process")
    return {"kit": KIT, "mode": MODE, "kit_files": kit_files(), "levers": LEVERS, "v1_2": v1_2.stamp(), "guard": guard.counters(),
            "anylen": {"in_force": anylen.in_force(), "counters": anylen.counters()}, "ew1_i64": ew1_i64.in_force(),
            "multiseq": {"in_force": multiseq.in_force(), "counters": multiseq.counters()}, "kvcache": {"in_force": kvcache.in_force(), "counters": kvcache.counters()},
            "tokenmemo": {"in_force": tokenmemo.in_force(), "counters": tokenmemo.counters()}}
