"""ESMC kit ``fused`` — the fused EAGER forward for the SDK (flash-attn) route; no new CUDA in this module (the CUDA / Triton kernels
are the composed items').

ONE line: ``from esmc_opt.kits import fused as kit; kit.apply(client.model)`` — then use the SDK or the bare forward
exactly as before. ``kit.remove(client.model)`` restores the stock (every class attribute and the one instance patch; no residue).
No knobs. The stock this lever equals is the pinned stack's (TE + flash_attn + the Triton rotary under the wrapper's bf16 autocast);
an install without those accelerators (the SDPA classes) is a different stock whose logits differ from the pinned stack's.

What the patch does (``_patch``, whose docstring has the detail): the per-layer varlen metadata + host sync hoisted out of the layer
loop; the autocast weight re-casts removed (the fp32 copies made once); the LN'd q/k written into the QKV slots (no per-layer stack);
unpad / pad elided at batch 1 (views); the wrapper's hidden states on demand. Same kernels, same order, fewer launches = bitwise by
construction. The composed items (COMPOSED_ITEMS: residual_ln, qk_rotary, thin) add their seams per hidden size (composition_for).
"""
from __future__ import annotations

from esmc_opt._oom import is_oom
import os
import time

from . import _patch

KIT = "fused"
REVISION = "r1"
ARM = "fused-eager"
PIECES = ("_patch",)
KIT_ENV = "ESMC_KIT"
HERE = os.path.dirname(os.path.abspath(__file__))
_state = {"applied": False, "record": None, "model_id": None, "models": {}, "model_refs": [], "item_modules": []}   # models: id(model) -> its apply record (every model built under the mode); record / model_id = the latest


class KitRefused(RuntimeError):
    pass


def counters_snapshot() -> dict:
    return _patch.counters_snapshot()


def counters_delta(before: dict) -> dict:
    return _patch.counters_delta(before)


def pins() -> dict:
    return _patch.pins()


REQUIRED_FILES = ("__init__.py", "_hs_write.py", "_patch.py")


def install_proof() -> dict:
    """Every file this kit reads from its own dir is present beside it (refuse on any missing file)."""
    missing = [name for name in REQUIRED_FILES if not os.path.isfile(os.path.join(HERE, name))]
    if missing:
        raise KitRefused(f"{KIT}: install proof FAILED — missing {missing}")
    return {"n_files": len(REQUIRED_FILES), "files": list(REQUIRED_FILES)}


def _unwrap(model):
    if hasattr(model, "esmc"):
        return model
    if hasattr(model, "model") and hasattr(model.model, "esmc"):
        return model.model
    raise KitRefused(f"{KIT}: apply() wants the native EsmcForMaskedLM (client.model) or the ESMC wrapper; got {type(model).__name__}")


def r8_line(model=None) -> str:
    rec = _state["record"] or {}
    gpu = rec.get("gpu", "none/cpu")
    eng = rec.get("engage", {}).get("engaged", [])
    comp = rec.get("composed") or {}
    outs = "; ".join(f"{n} ({why})" for n, why in (comp.get("items_out") or [])) or "none"
    return (f"[{KIT}{'+ITEMS' if comp else ''}] GPU {gpu} | size {comp.get('size')} | levers ON: {'; '.join(eng)} | seams out by shape: {outs} | EAGER ONLY (by design): no CUDA graphs, no fixed-shape capture, no bucketing — every call is an eager per-call forward | knobs: none | "
            f"PAID at apply {rec.get('apply_s', 0.0) * 1e3:.1f} ms (install proof {rec.get('install_proof_s', 0.0) * 1e3:.1f} + pins/patch {rec.get('engage', {}).get('engage_s', 0.0) * 1e3:.1f} ms; "
            f"fp32 LN weight copies made once; qk_ln_rope compile classes warmed {len(rec.get('qk_ladder', {}).get('warmed', []))} (BLOCK_B {rec.get('qk_ladder', {}).get('block_b_classes', [])} x n_seqs/seqlen_ro specializations): "
            f"cache HITS {rec.get('qk_ladder', {}).get('hits', 0)} / COMPILES {rec.get('qk_ladder', {}).get('misses', 0)} in {rec.get('qk_ladder', {}).get('s', 0.0):.2f} s{(' [' + rec['qk_ladder']['error'] + ']') if rec.get('qk_ladder', {}).get('error') else ''}; "
            f"classes: {'; '.join(rec.get('qk_ladder', {}).get('classes', []))}; "
            f"thin templates primed by one {PRIME_TOKENS}-token forward: {(rec.get('prime') or {}).get('primed')} in {float((rec.get('prime') or {}).get('s') or 0.0) * 1e3:.0f} ms"
            f"{(' [' + str(rec['prime'].get('note')) + ']') if (rec.get('prime') or {}).get('note') else ''}) | DEFERRED: nothing (no capture, no build; the one priming forward is paid here)")


# THE SERVED MODEL CONFIGS: the three ESM C sizes as the model prints them (head_dim 64 on every size). Any other model REFUSES BY NAME at
# apply (the refusal line) and leaves the stock forward untouched. The base path (_patch: the varlen metadata hoisted, the q/k
# LayerNorm weight casts made once, unpad/pad as views at batch 1 — the stock's kernels in the stock's order) serves every size; each
# composed ITEM serves the shapes its kernels are bitwise on (ITEM_SERVES, from the items' own records) and is composed only there —
# the composition of a size is this table, fixed per shape, named on the kit's line (never a silent subset: composition_for).
SERVED_CONFIGS = {
    "300m": {"hidden_size": 960, "num_hidden_layers": 30, "num_attention_heads": 15},
    "600m": {"hidden_size": 1152, "num_hidden_layers": 36, "num_attention_heads": 18},
    "6b": {"hidden_size": 2560, "num_hidden_layers": 80, "num_attention_heads": 40},
}
TESTED_CONFIG = SERVED_CONFIGS["6b"]                                             # the 6B config: the shape the composed items' kernels were written against
# Per item, per seam: the hidden sizes the seam's kernel is bitwise on, and why (the item's own record). A seam absent for a size is named
# on the kit's line with this reason; the item's other seams still compose.
ITEM_SERVES = {
    "esmc_opt.kits.residual_ln": {
        "residual": {"hidden_size": (8, 8192), "why": "the residual kernel is elementwise (ATen's div+add pair per 8-wide vector): any hidden size % 8 == 0"},
        "residual_ln_qkv": {"hidden_size": (513, 8192), "why": "residual_ln_te mirrors TE's general LayerNorm configs <1024|2048,4,1,16> (512 < N <= 2048: 960, 1152) and <8192,1,4,16> (2048 < N <= 8192: 2560)"},
    },
    "esmc_opt.kits.qk_rotary": {
        "qk_ln_rotary": {"hidden_size": (8, 8192), "multiple_of": 4, "why": "qk_ln_rope mirrors ATen's NT-128 float4 Welford order — N % 4 == 0 (ATen's vectorized path), the ragged last pass masked as ATen's strided loop; 960 / 1152 / 2560 proven bitwise"},
    },
    "esmc_opt.kits.thin": {
        "attn": {"hidden_size": (8, 8192), "why": "recorded lazily per module from its first real call: shape-generic"},
        "ffn": {"hidden_size": (8, 8192), "why": "recorded lazily per module from its first real call: shape-generic"},
        "ln_qkv": {"hidden_size": (8, 8192), "why": "recorded lazily per module from its first real call: shape-generic"},
        "out_proj": {"hidden_size": (8, 8192), "why": "recorded lazily per module from its first real call: shape-generic"},
        "rotary": {"hidden_size": (8, 8192), "why": "the cached Triton rotary launcher with the stock's own arguments: shape-generic"},
    },
}
# THE COMPOSITION: the seam modules composed onto the base patch, in apply order — (module, the seams its ops(model) returns
# at the 6B shape); composition_for(hidden) selects per size. No knob.
COMPOSED_ITEMS = (
    ("esmc_opt.kits.residual_ln", ("residual", "residual_ln_qkv")),
    ("esmc_opt.kits.qk_rotary", ("qk_ln_rotary",)),
    ("esmc_opt.kits.thin", ("attn", "ffn", "ln_qkv", "out_proj", "rotary")),
)


def seam_serves(module: str, seam: str, hidden: int) -> bool:
    rule = (ITEM_SERVES.get(module) or {}).get(seam) or {}
    lo, hi = rule.get("hidden_size", (None, None))
    if lo is not None and not (lo <= int(hidden) <= hi):
        return False
    m = rule.get("multiple_of")
    return not (m and int(hidden) % int(m))


def composition_for(hidden: int):
    """([(module, seams composed at this hidden size)], [(item.seam, why not)]) — COMPOSED_ITEMS filtered by ITEM_SERVES; an item with no
    seam left at this size is not imported at all."""
    on, out = [], []
    for mod, seams in COMPOSED_ITEMS:
        keep = tuple(x for x in seams if seam_serves(mod, x, hidden))
        for x in seams:
            if x not in keep:
                out.append((f"{mod.rsplit('.', 1)[-1]}.{x}", ITEM_SERVES[mod][x]["why"]))
        if keep:
            on.append((mod, keep))
    return on, out


def size_of(got: dict):
    """the served size name whose config equals the model's (hidden, layers, heads), else None."""
    for name, cfg in SERVED_CONFIGS.items():
        if all(got.get(k) == v for k, v in cfg.items()):
            return name
    return None


def model_gate(m) -> dict:
    """the config read from the model (no forward, no patch); refused = True with the reasons BY NAME when it is not a served ESM C shape.
    ``composition`` / ``items_out`` = composition_for(hidden) for a served shape."""
    cfg = getattr(m, "config", None)
    got = {k: getattr(cfg, k, None) for k in TESTED_CONFIG}
    hd = None
    try:
        hd = int(got["hidden_size"]) // int(got["num_attention_heads"])
    except Exception:  # noqa: BLE001
        pass
    size = size_of(got)
    reasons = []
    if size is None:
        reasons.append(f"{KIT} serves the ESM C shapes " + "; ".join(f"{n}=" + ",".join(f"{k}={v}" for k, v in c.items()) for n, c in SERVED_CONFIGS.items())
                       + " — got " + ", ".join(f"{k}={v}" for k, v in got.items()))
    elif hd != 64:
        reasons.append(f"head_dim {hd}: every served shape has head_dim 64")
    comp, out = composition_for(int(got["hidden_size"])) if not reasons else ([], [])
    return {"refused": bool(reasons), "model": got, "head_dim": hd, "size": size, "reasons": reasons, "composition": comp, "items_out": out}


def r8_refusal_line(gate) -> str:
    return (f"[{KIT}] REFUSED at apply — nothing patched, the stock forward untouched | model {gate['model']} head_dim {gate['head_dim']} | "
            f"refused by: {' | '.join(gate['reasons'])} | knobs: none")


QK_LADDER_N_SEQS = (1, 3, 7, 15, 31, 63, 127, 255, 16, 32, 64, 128)   # the qk_ln_rope kernel is compiled per (BLOCK_B = next_power_of_2(n_seqs + 1),
QK_LADDER_SEQLEN_RO = (16, 18)                               # Triton's int specializations of n_seqs (== 1, % 16) and of the rotary cache length seqlen_ro (% 16))
                                                             # -> 20 classes; apply() runs the seam once per class (tiny scratch tensors, block 0's modules),
                                                             # so no forward compiles; each class's seconds classify it as a cache HIT (< 0.25 s) or a compile.


def _prewarm_qk_ladder(m) -> dict:
    """run the merged q/k-LN+rotary seam once per compile class on a tiny packed dummy (block 0's own modules; L = 8 rows per sequence);
    pure function calls on scratch tensors — the only model state touched is the rotary cos/sin cache, grown to 16 then 18 rows (its
    values are position-wise the same at any size; the first real forward grows it further exactly as the stock does). A class Triton
    has compiled before on this machine is a hit in its own cache (fast); a new class compiles here, once, instead of inside the caller's
    first forward. Returns the classes warmed, per-class seconds, and the hit/compile counts."""
    import torch
    merged = _patch._OPS.get("qk_ln_rotary")
    if merged is None:
        return {"warmed": [], "seconds": [], "hits": 0, "misses": 0, "s": 0.0, "note": "no qk_ln_rotary seam composed"}
    blk = m.esmc.transformer.blocks[0].attn
    dev = next(m.parameters()).device
    H, Dh = int(blk.n_heads), int(blk.d_head)
    warmed, secs, labels, t0 = [], [], [], time.perf_counter()
    _run_ladder(m, blk, dev, H, Dh, merged, warmed, secs, labels, torch)
    hits = sum(1 for x in secs if x < 0.25)
    return {"warmed": warmed, "block_b_classes": sorted({max(2, 1 << (n + 1 - 1).bit_length()) for n in QK_LADDER_N_SEQS}), "seconds": secs,
            "hits": hits, "misses": len(secs) - hits, "s": round(time.perf_counter() - t0, 3), "classes": labels}


def _run_ladder(m, blk, dev, H, Dh, merged, warmed, secs, labels, torch):
    with torch.no_grad():
        for ro in QK_LADDER_SEQLEN_RO:
            blk.rotary._update_cos_sin_cache(ro, device=dev, dtype=torch.bfloat16)
            for n in QK_LADDER_N_SEQS:
                L = 8
                qkv = torch.zeros((n * L, 3, H, Dh), dtype=torch.bfloat16, device=dev)
                cu = torch.arange(n + 1, device=dev, dtype=torch.int32) * L
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                merged(blk.rotary, blk.q_ln, blk.k_ln, qkv, cu, L)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                secs.append(round(time.perf_counter() - t1, 3)); warmed.append((ro, n))
                labels.append(f"(n_seqs {n} [{'==1' if n == 1 else ('%16' if n % 16 == 0 else 'generic')}], BLOCK_B {max(2, 1 << (n + 1 - 1).bit_length())}, seqlen_ro {ro} [{'%16' if ro % 16 == 0 else 'generic'}]): "
                              + ("COMPILED" if secs[-1] >= 0.25 else "cached"))


PRIME_TOKENS = 61                      # the priming forward's row count: a prime that equals no constant dimension of any served shape (thin's template classifier keys the row count)


def _prime_templates(m) -> dict:
    """ONE tiny forward (1 x PRIME_TOKENS tokens) through the patched model right after the items are composed, so the thin item's lazy
    per-module templates (recorded from a module's first real call, which runs the stock forward under the recorder) are recorded HERE, at
    apply, and the caller's first call already replays. Pure function of scratch inputs under inference_mode: no model state changes besides
    the templates and the libraries' own workspaces (which the first real call would allocate identically). A failure is named, never fatal:
    the first real call would record as before."""
    import torch
    t0 = time.perf_counter()
    try:
        dev = next(m.parameters()).device
        if dev.type != "cuda":
            return {"primed": False, "s": 0.0, "note": "model not on CUDA"}
        ids = torch.full((1, PRIME_TOKENS), 5, dtype=torch.long, device=dev); ids[0, 0] = 0; ids[0, -1] = 2
        mask = torch.ones((1, PRIME_TOKENS), dtype=torch.bool, device=dev)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):   # the SDK's own context (ESMC.logits / the cookbook's batched call)
            m(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize(dev)
        return {"primed": True, "s": round(time.perf_counter() - t0, 4), "tokens": PRIME_TOKENS}
    except Exception as ex:  # noqa: BLE001 — priming is an optimisation of WHEN the templates record, not a gate
        if is_oom(ex): raise
        return {"primed": False, "s": round(time.perf_counter() - t0, 4), "note": f"{type(ex).__name__}: {str(ex)[:160]}"}


def apply(model, **opts) -> dict:
    if opts:
        raise KitRefused(f"{KIT}: no knobs — apply(model) takes no options; got {sorted(opts)}")
    import torch, weakref
    m = _unwrap(model)
    if id(m) in _state["models"]:
        raise KitRefused(f"{KIT}: already applied to this model; remove() first")
    gate = model_gate(m)                                                          # refuse BY NAME before anything is patched
    if gate["refused"]:
        _state["last_refusal"] = gate
        print(r8_refusal_line(gate), flush=True)
        raise KitRefused(f"{KIT}: REFUSED (nothing patched; the stock forward untouched): " + "; ".join(gate["reasons"]))
    t0 = time.perf_counter()
    proof = install_proof()
    t_proof = time.perf_counter() - t0
    engage = _patch.engage(m)
    # COMPOSITION: COMPOSED_ITEMS (the seam modules, in apply order); each item's ops(model) must return EXACTLY the
    # listed seams, and the entries replace the base patch's defaults in _patch._OPS.
    import importlib
    composed = {"items": [], "size": gate["size"], "items_out": gate["items_out"]}
    for module, seams in gate["composition"]:                                     # the composition of THIS size (composition_for): each item's seams bitwise at this hidden size
        mod = importlib.import_module(module)
        d = os.path.dirname(os.path.abspath(mod.__file__))
        ops = mod.ops(m)
        if set(ops) != set(seams):
            raise KitRefused(f"{KIT}: item {module} ops() returned {sorted(ops)} != the listed seams {sorted(seams)}")
        unknown = sorted(set(ops) - set(_patch._OPS) - set(getattr(_patch, "OPTIONAL_SEAMS", ("residual_ln_qkv", "qk_ln_rotary", "out_proj_residual"))))
        if unknown:
            raise KitRefused(f"{KIT}: item {module} names unknown seams {unknown}")
        _patch._OPS.update(ops)
        composed["items"].append({"module": module, "dir": d, "seams": sorted(ops),
                                  "ext_load": getattr(mod, "ext_load", lambda: None)()})
        if mod not in _state["item_modules"]:
            _state["item_modules"].append(mod)
        engage["engaged"] = list(engage["engaged"]) + [f"{module.rsplit('.', 1)[-1]}: {', '.join(sorted(ops))}"]
    prime = _prime_templates(m)                                                     # the thin launchers' templates recorded NOW (one tiny forward through the patched path) — the caller's first call replays, never records
    try:
        ladder = _prewarm_qk_ladder(m)                                              # the BLOCK_B ladder warmed at apply, once per process
    except Exception as ex:  # noqa: BLE001  (a warm-up failure is not a refusal: a forward would JIT as before)
        if is_oom(ex): raise                                  # an out-of-memory propagates: no fallback applied
        ladder = {"warmed": [], "block_b_classes": [], "seconds": [], "hits": 0, "misses": 0, "s": 0.0, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    rec = {"kit": KIT, "revision": REVISION, "arm": ARM, "engage": engage, "composed": composed, "install_proof": proof, "install_proof_s": t_proof, "qk_ladder": ladder, "prime": prime,
           "gpu": torch.cuda.get_device_name(next(m.parameters()).device) if next(m.parameters()).is_cuda else "none/cpu",
           "model_dtype": str(next(m.parameters()).dtype), "applied_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rec["apply_s"] = time.perf_counter() - t0; rec["model_index"] = len(_state["models"]) + 1
    _state["models"][id(m)] = rec; _state["model_refs"].append(weakref.ref(m))
    _state.update({"applied": True, "record": rec, "model_id": id(m)})
    os.environ[KIT_ENV] = KIT
    print(r8_line(m), flush=True)
    return rec


def remove(model=None) -> None:
    """Process-wide: every model the kit was applied to gets its instance patches removed and the class patches are restored."""
    if not _state["applied"]:
        return
    models = [r() for r in _state["model_refs"]]
    if model is not None:
        models.append(_unwrap(model))
    models = list({id(x): x for x in models if x is not None}.values())
    for mod in reversed(_state["item_modules"]):              # items that replaced symbols / instance forwards restore them (thin launchers), per model
        if hasattr(mod, "disengage"):
            for m in (models or [None]):
                try:
                    mod.disengage(m)
                except Exception as ex:  # noqa: BLE001 — a restore that fails is reported, never hidden
                    print(f"[{KIT}] item {mod.__name__} disengage: {type(ex).__name__}: {ex}", flush=True)
    _patch.disengage(model)
    os.environ.pop(KIT_ENV, None)
    _state.update({"applied": False, "record": None, "model_id": None, "models": {}, "model_refs": [], "item_modules": []})


def restore() -> None:
    remove()


def is_applied() -> bool:
    return bool(_state["applied"])
