"""enformer_opt._runtime — the whole runtime: resolve the kit on this machine, arm the class-level hook on ``Enformer.forward``, apply the kit to
each instance at its first forward (or eagerly through ``apply``), print the lines, undo on ``disable``.

Lines (stderr, one each, ``[enformer-opt]`` first):
  ACTIVE    mode=exact gpu=<name>(smNN) kit=v0.2 levers=poscache,fused,graph,xattn build=<sm_90|class:<slug>|sm_90-ptx> applied=deferred [notes=…]
  DRY       the same fields from ``check`` — nothing applied
  NOT ACTIVE mode=exact reason=<why>   — nothing applied, stock untouched
  APPLIED   model#<k> levers=… batch=<B> cost=<s>s (extensions + module patches + one CUDA-graph capture for batch <B>)
  CAPTURED  model#<k> batch=<B> cost=<s>s (one CUDA-graph capture at this batch size's first call)
  UNGRAPHED model#<k> batch=<B>: <why> — no CUDA graph at this batch size (its capture does not fit the free device memory by the kit's footprint
            rule, or ran out of memory): that size runs the patched modules without replay, same outputs — said once per size
  EAGER     model#<k> call form <kwargs> … (said once per model) | window length <n> bp … (said once per length): the stock forward code under
            the kit's module patches, no graph (exact)
  STOCK     model#<k> <why> — a training-mode model, or one not on a CUDA device, runs the stock forward (said once per model)
  AUTOCAST  model#<k> … — under the caller's autocast the kit's path computes in float32 (said once per model)
  NUMERICS  model#<k> … — a numerics switch changed since the graphs were captured: they are recaptured under it (said once per change)
  REMOVED   model#<k>
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
import time
import weakref

PREFIX = "[enformer-opt]"
MODE = "exact"
UPSTREAM_DIST, UPSTREAM_VERSION = "enformer-pytorch", "0.8.12"          # the stock the levers reproduce (stock/PINS.json upstream)
TESTED_TORCH = "2.13.0+cu130"                                       # the torch build of the pinned stack (a different one is named on the line, not refused)
MODEL_MODULE, MODEL_CLASS = "enformer_pytorch.modeling_enformer", "Enformer"
KIT_PACKAGE = "engines.enformer.kits.v0_2"
OPT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))               # enformer/opt (the package is installed editable from it)
KIT_HOME = os.path.join(OPT_HOME, "forward", "enformer_kit")                          # the vendored kit closure (package `engines`)
BIN_DIR = os.path.join(OPT_HOME, "forward", "bin")                                    # the kit's own extension objects (sm_90 cubin + compute_90 PTX)
TESTED_CARDS = {(9, 0): "H100 / H200", (8, 0): "A100"}                                   # the cards of the pinned builds (sm_90 objects, sm_80 class build); any other capability a build serves is named on the line


class ActivationError(RuntimeError):
    """enable(strict=True) / the ENFORMER_OPT route: the kit cannot engage on this machine (the NOT ACTIVE line carries the reason)."""


_LOCK = threading.RLock()
_REPORT: dict | None = None            # the activation report of this process (None until enable / check ran)
_HOOK = {"installed": False, "cls": None, "forward": None}
_MODELS: dict = {}                      # id(model) -> {"index", "ref", "handle", "orig_forward", "said_eager", "batch_sizes"}
_COUNTER = {"n": 0}
_ENABLING = False                       # True while enable() imports upstream itself (the autoload finder stands down)


# ----------------------------------------------------------------------------------------------------------------- lines
def log(line: str) -> str:
    print(line, file=sys.stderr, flush=True)
    return line


def _gpu_label(rep: dict) -> str:
    g = rep.get("gpu") or {}
    return f"{g['name']}(sm{g['sm'][0]}{g['sm'][1]})" if g.get("name") else "none"


def line_of(rep: dict) -> str:
    if not rep.get("active"):
        return f"{PREFIX} NOT ACTIVE mode={MODE} reason={rep.get('reason') or 'unknown'}"
    core = f"mode={MODE} gpu={_gpu_label(rep)} kit={rep['kit_version']} levers={','.join(rep['levers'])} build={rep['build']}"
    head = "DRY" if rep.get("dry_run") else "ACTIVE"
    tail = "" if rep.get("dry_run") else " applied=deferred"
    notes = f" notes={'; '.join(rep['notes'])}" if rep.get("notes") else ""
    return f"{PREFIX} {head} {core}{tail}{notes}"


# ----------------------------------------------------------------------------------------------------------------- resolution
def _device() -> dict:
    """Device 0 by torch: name, capability, memory; the reason when there is none."""
    try:
        import torch
    except Exception as e:                                             # noqa: BLE001
        return {"name": None, "reason": f"torch is not importable ({type(e).__name__}: {e})"}
    if not torch.cuda.is_available():
        return {"name": None, "reason": "no CUDA device visible (torch.cuda.is_available() is False)"}
    p = torch.cuda.get_device_properties(0)
    return {"name": torch.cuda.get_device_name(0), "sm": (p.major, p.minor), "memory_mib": int(p.total_memory // (1 << 20)),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def _upstream_version() -> str | None:
    try:
        from importlib import metadata
        return metadata.version(UPSTREAM_DIST)
    except Exception:                                                  # noqa: BLE001
        return None


def _kit():
    """The kit module (engines.enformer.kits.v0_2), importable from the vendored closure."""
    if not os.path.isdir(os.path.join(KIT_HOME, "engines", "enformer", "kits", "v0_2")):
        raise FileNotFoundError(f"the kit closure is not at {KIT_HOME} (the package must be installed editable from enformer/opt: bash run.sh install)")
    if KIT_HOME not in sys.path:
        sys.path.insert(0, KIT_HOME)
    return importlib.import_module(KIT_PACKAGE)


def _so_paths(kit) -> dict:
    return {key: os.path.join(BIN_DIR, kit.KIT_PINS[key]) for key in kit.BINARIES}


def resolve(dry_run: bool) -> dict:
    """Everything enable() decides before touching a model: device, upstream pin, kit files, the build that runs here. Never raises."""
    rep = {"mode": MODE, "active": False, "dry_run": dry_run, "reason": None, "notes": [], "gpu": None, "kit_version": None, "levers": [],
           "build": None, "binaries": {}, "upstream": None, "package": OPT_HOME}
    dev = _device()
    rep["gpu"] = dev
    if not dev.get("name"):
        rep["reason"] = dev["reason"]
        return rep
    up = _upstream_version()
    rep["upstream"] = up
    if up != UPSTREAM_VERSION:
        rep["reason"] = (f"{UPSTREAM_DIST} {up} is installed, the kit reproduces {UPSTREAM_VERSION} (bash run.sh install)" if up
                         else f"{UPSTREAM_DIST} is not installed (bash run.sh install)")
        return rep
    try:
        kit = _kit()
        stamp = kit.check_files(_so_paths(kit))                       # the sources + the two objects that load on THIS device (class dir / own objects / refusal by name)
    except Exception as e:                                             # noqa: BLE001
        rep["reason"] = f"{type(e).__name__}: {e}"
        return rep
    cp = stamp.get("class_pins")
    sm = tuple(dev["sm"])
    if cp:
        rep["build"] = f"class:{cp['slug']}"
        if cp.get("serves"):
            rep["notes"].append(f"uncertified architecture — {cp['serves']}")
    elif sm == tuple(kit.KIT_PINS["sm"]):
        rep["build"] = "sm_90"
    else:
        rep["build"] = "sm_90-ptx"
        rep["notes"].append(f"uncertified architecture — sm_{sm[0]}{sm[1]} runs the kit's sm_90 objects through their compute_90 PTX (compiled by the driver at load)")
    if sm not in TESTED_CARDS and not any("uncertified" in n for n in rep["notes"]):
        rep["notes"].append(f"uncertified device — capability {sm[0]}.{sm[1]} is outside the tested set ({', '.join(f'{a}.{b} {n}' for (a, b), n in TESTED_CARDS.items())})")
    if dev.get("torch") != TESTED_TORCH:
        rep["notes"].append(f"torch {dev.get('torch')} differs from the tested {TESTED_TORCH} (the levers engage; byte-identity to stock was established on that build)")
    rep.update(active=True, kit_version=kit.KIT_PINS["version"], levers=list(kit.KIT_PINS["levers"]),
               binaries={k: {"path": stamp.get(f"{k}_path"), "sha256": stamp.get(f"{k}_sha256")} for k in kit.BINARIES})
    return rep


# ----------------------------------------------------------------------------------------------------------------- public API
def check() -> dict:
    """The dry run: resolve and print the DRY / NOT ACTIVE line; apply nothing."""
    rep = resolve(dry_run=True)
    log(line_of(rep))
    return rep


def status() -> dict:
    """The activation report of this process plus the per-model records (index, batch sizes captured, eager calls)."""
    rep = dict(_REPORT or {"mode": MODE, "active": False, "reason": "enable() has not been called"})
    rep["models"] = [{"index": r["index"], "applied": r["handle"] is not None, "batch_sizes": sorted(getattr(r["handle"], "graphs", {}) or {}),
                      "eager_calls": r["eager_calls"], "other_lengths": list(r["other_lengths"])} for r in _MODELS.values()]
    return rep


def enable(*, strict: bool = False, trigger: str | None = None) -> dict:
    """Engage the kit for this process: every ``Enformer`` instance (existing or future) runs through the levers from its next forward on.
    Idempotent. Returns the activation report; with ``strict`` a refusal raises ActivationError after the NOT ACTIVE line."""
    global _REPORT, _ENABLING
    with _LOCK:
        if _REPORT is not None and _REPORT.get("active") and not _REPORT.get("dry_run"):
            return _REPORT
        rep = resolve(dry_run=False)
        rep["trigger"] = trigger or "enable()"
        if rep["active"]:
            _ENABLING = True
            try:
                _install_hook()
            except Exception as e:                                     # noqa: BLE001
                rep.update(active=False, reason=f"cannot hook {MODEL_MODULE}.{MODEL_CLASS}.forward ({type(e).__name__}: {e})")
            finally:
                _ENABLING = False
        _REPORT = rep
        log(line_of(rep))
        if not rep["active"] and strict:
            raise ActivationError(rep["reason"])
        return rep


def apply(model, batch_sizes=()) -> dict:
    """Apply the kit to ``model`` now (extensions, module patches) and capture one trunk graph per size in ``batch_sizes`` up front, instead of
    at the model's first forward. Requires enable(). Returns the model's record."""
    if _REPORT is None or not _REPORT.get("active") or _REPORT.get("dry_run"):
        raise ActivationError("enformer_opt.apply(): call enable() first (the kit is not active in this process)")
    with _LOCK:
        rec = _record(model)
        if rec["handle"] is None:
            _apply(model, rec, tuple(int(b) for b in batch_sizes))
        return {k: rec[k] for k in ("index", "levers", "apply_cost_s", "eager_calls")} | {"batch_sizes": sorted(rec["handle"].graphs)}


def disable() -> dict:
    """Undo everything: close each model's kit handle (the stock modules are restored), drop the instance forwards and the class hook."""
    global _REPORT
    with _LOCK:
        removed = []
        for rec in list(_MODELS.values()):
            m = rec["ref"]()
            if rec["handle"] is not None:
                try:
                    rec["handle"].close()
                except Exception as e:                                 # noqa: BLE001
                    log(f"{PREFIX} REMOVED model#{rec['index']} close raised {type(e).__name__}: {e}")
                rec["handle"] = None
                if m is not None and "forward" in m.__dict__:
                    del m.__dict__["forward"]
                removed.append(rec["index"])
                log(f"{PREFIX} REMOVED model#{rec['index']}")
        _MODELS.clear()
        if _HOOK["installed"]:
            _HOOK["cls"].forward = _HOOK["forward"]
            _HOOK.update(installed=False, cls=None, forward=None)
        _REPORT = None
        return {"removed": removed}


# ----------------------------------------------------------------------------------------------------------------- hook + application
def _install_hook() -> None:
    if _HOOK["installed"]:
        return
    cls = getattr(importlib.import_module(MODEL_MODULE), MODEL_CLASS)
    orig = cls.forward

    def forward(self, x, *a, **kw):
        return _hooked_forward(self, x, *a, **kw)

    forward.__wrapped__ = orig
    forward.__enformer_opt__ = True
    cls.forward = forward
    _HOOK.update(installed=True, cls=cls, forward=orig)


def _record(model) -> dict:
    rec = _MODELS.get(id(model))
    if rec is None or rec["ref"]() is None:
        _COUNTER["n"] += 1
        rec = {"index": _COUNTER["n"], "ref": weakref.ref(model), "handle": None, "levers": [], "apply_cost_s": None, "eager_calls": 0, "other_lengths": [], "said_autocast": False, "numerics": None,
               "said_eager": False, "said_stock": False, "busy": False, "ungraphed_said": set()}
        _MODELS[id(model)] = rec
    return rec


def _hooked_forward(model, x, *a, **kw):
    rec = _record(model)
    if rec["busy"]:                                                    # the kit's own trunk captures call the model: the stock forward code (under whatever patches are in place)
        return _HOOK["forward"](model, x, *a, **kw)
    if rec["handle"] is None:
        dev = _device_of(model)
        stock_reason = ("is in training mode: the stock forward runs (the kit applies to eval-mode models; call model.eval() first)" if model.training
                        else f"is on {dev}: the stock forward runs (the kit applies to a model on a CUDA device; model.cuda() first)" if dev is None or dev.type != "cuda"
                        else None)
        if stock_reason:                                               # said once per model, never silent; the instance is applied at its first eval-mode forward on a CUDA device
            if not rec["said_stock"]:
                rec["said_stock"] = True
                log(f"{PREFIX} STOCK model#{rec['index']} {stock_reason}")
            return _HOOK["forward"](model, x, *a, **kw)
        with _LOCK:
            if rec["handle"] is None:
                b = _batch_of(x)
                _apply(model, rec, (b,) if b else ())
    return model.forward(x, *a, **kw)                                  # the instance forward _apply installed


def _batch_of(x) -> int | None:
    """The number of windows a forward input carries (a graph is captured for it at apply): (B, L, 4) one-hot or (B, L) indices -> B;
    one window (L, 4) / (L,) / a string -> 1; a list of strings -> its length."""
    import torch
    if isinstance(x, torch.Tensor):
        if x.ndim == 3 or (x.ndim == 2 and x.dtype == torch.long):
            return int(x.shape[0])
        return 1
    if isinstance(x, str):
        return 1
    if isinstance(x, (list, tuple)):
        return max(1, len(x))
    return None


def _device_of(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _apply(model, rec: dict, batch_sizes: tuple) -> None:
    import torch
    kit = _kit()
    dev = _device_of(model)
    if dev is None or dev.type != "cuda":
        raise ActivationError(f"enformer_opt: model#{rec['index']} is on {dev}, the kit applies to a model on a CUDA device (model.cuda() first)")
    t0 = time.perf_counter()
    rec["busy"] = True
    try:
        handle, stamp = kit.attach(model, _so_paths(kit), batch_shapes=batch_sizes, device=str(dev))
    finally:
        rec["busy"] = False
    torch.cuda.synchronize(dev)
    rec.update(handle=handle, levers=list(stamp["levers"]), apply_cost_s=round(time.perf_counter() - t0, 2), stamp=stamp, numerics=_numerics())   # the switches the graphs are captured under
    rec["orig_forward"] = _HOOK["forward"] if _HOOK["installed"] else type(model).forward
    model.forward = _instance_forward(model, rec)
    what = "extensions + module patches" + (f" + one CUDA-graph capture for batch {', '.join(map(str, sorted(handle.graphs)))}" if handle.graphs else "")
    log(f"{PREFIX} APPLIED model#{rec['index']} levers={','.join(rec['levers'])} batch={','.join(map(str, sorted(handle.graphs))) or 'none'} cost={rec['apply_cost_s']}s ({what})")
    for b in sorted(handle.ungraphed):
        rec["ungraphed_said"].add(b)
        log(_ungraphed_line(rec, b, handle))


def _numerics() -> tuple:
    """The caller's numerics switches the stock forward follows (matmul TF32 / precision, cuDNN TF32 / benchmark, deterministic algorithms)."""
    import torch
    return (bool(torch.backends.cuda.matmul.allow_tf32), torch.get_float32_matmul_precision(), bool(torch.backends.cudnn.allow_tf32),
            bool(torch.backends.cudnn.benchmark), bool(torch.are_deterministic_algorithms_enabled()))


def _instance_forward(model, rec: dict):
    """``model(x)``: a one-hot window (L, 4) / batch (B, L, 4), an index tensor, a sequence string or a list of strings -> the kit's path (trunk
    graph replay + the stock heads, the stock's own single-window / batched head arithmetic); any other call form (head=, return_embeddings=,
    target=, ...) -> the stock forward code under the kit's module patches, eager (exact), counted and said once."""
    handle, orig = rec["handle"], rec["orig_forward"]

    def forward(x, *a, **kw):
        import torch
        if model.training:
            raise RuntimeError(f"{PREFIX} model#{rec['index']}: model.train() after the kit was applied — the kit is inference-only; call enformer_opt.disable() before training")
        if rec["busy"]:                                                # a trunk capture in progress (return_only_embeddings=True from TrunkGraph): the stock code under the patches
            with torch.no_grad():
                return orig(model, x, *a, **kw)
        if torch.is_grad_enabled() and isinstance(x, torch.Tensor) and x.requires_grad:   # gradients are asked of an inference-only path: refused by name, never silently wrong
            raise RuntimeError(f"{PREFIX} model#{rec['index']}: the input requires grad — the kit is inference-only (its kernels carry no autograd); "
                               "call enformer_opt.disable() before gradient computations")
        if a or kw:
            rec["eager_calls"] += 1
            if not rec["said_eager"]:
                rec["said_eager"] = True
                log(f"{PREFIX} EAGER model#{rec['index']} call form {tuple(kw) or 'positional args'} runs the stock forward code under the kit's module patches (exact; no graph)")
            with torch.no_grad():
                return orig(model, x, *a, **kw)
        if isinstance(x, str) or (isinstance(x, (list, tuple)) and x and isinstance(x[0], str)):
            from enformer_pytorch import str_to_one_hot
            x = str_to_one_hot(x if isinstance(x, str) else (list(x) if len(x) > 1 else x[0]))
        elif isinstance(x, torch.Tensor) and x.dtype == torch.long:
            from enformer_pytorch import seq_indices_to_one_hot
            x = seq_indices_to_one_hot(x)
        if x.ndim in (2, 3) and int(x.shape[-2]) != handle.seq_len:          # a window length other than the graphs' (196,608 bp): the stock forward code under the module patches, eager (exact; said once per length)
            n = int(x.shape[-2])
            if n not in rec["other_lengths"]:
                rec["other_lengths"].append(n)
                log(f"{PREFIX} EAGER model#{rec['index']} window length {n:,} bp is not the graphs' {handle.seq_len:,}: calls of this length run the patched modules without graph replay (exact)")
            rec["eager_calls"] += 1
            with torch.no_grad():
                return orig(model, x, *a, **kw)
        if x.device != _device_of(model) or x.dtype != torch.float32:      # not a float32 tensor on the model's device: the stock code decides (it raises for a CPU input on a CUDA model; the kit does not paper over it)
            with torch.no_grad():
                return orig(model, x, *a, **kw)
        now = _numerics()
        if now != rec["numerics"]:                                     # the caller changed a numerics switch since the graphs were captured: they are released and recaptured under it (said once per change)
            log(f"{PREFIX} NUMERICS model#{rec['index']} switches changed since capture {rec['numerics']} -> {now} (matmul TF32, matmul precision, cuDNN TF32, cuDNN benchmark, deterministic): "
                "the CUDA graphs are recaptured under the new switches; byte-identity to stock was established at torch's defaults")
            handle.release_graphs()
            rec["numerics"] = now
        if torch.is_autocast_enabled():                                # the caller's autocast: the kit's path computes in float32 — the stock's default numerics — said once
            if not rec["said_autocast"]:
                rec["said_autocast"] = True
                log(f"{PREFIX} AUTOCAST model#{rec['index']} autocast is active: the kit computes in float32 (outputs equal the stock's float32 forward, not the stock under autocast); "
                    "enformer_opt.disable() first for a mixed-precision stock forward")
            with torch.autocast(device_type="cuda", enabled=False):
                return _dispatch(model, rec, handle, x)
        return _dispatch(model, rec, handle, x)

    forward.__wrapped__ = orig
    forward.__enformer_opt__ = True
    return forward


def _dispatch(model, rec: dict, handle, x):
    """The kit's path for a float32 (L, 4) / (B, L, 4) input on the model's device at the graphs' window length."""
    import torch
    if x.ndim == 2:                                                    # one window, rank 2: the stock's single-window path (heads on the 2-D embedding)
        return {h: t[0] for h, t in _timed(handle, rec, 1, handle.forward_device, x.unsqueeze(0)).items()}
    b = int(x.shape[0])
    if b == 1:                                                         # one window, rank 3: the stock's BATCHED head arithmetic on the (1, bins, d) embedding
        emb = _timed(handle, rec, 1, handle.embed, x)
        with torch.no_grad():
            return {h: head(emb) for h, head in model._heads.items()}
    return _timed(handle, rec, b, handle.forward_device, x)


def _timed(handle, rec: dict, b: int, fn, x):
    """fn(x) through the kit; when this batch size has no graph yet the call includes its capture — timed and printed once (CAPTURED); a size
    whose capture (or the first call beside its pool) ran out of device memory runs the same patched modules without replay — UNGRAPHED, said
    once per size."""
    import torch
    new = b not in handle.graphs and b not in handle.ungraphed
    t0 = time.perf_counter() if new else None
    rec["busy"] = True                                                 # the kit's own inner calls (a capture's warm-up, an ungraphed size) re-enter forward(): the stock code under the patches, silently
    try:
        out = fn(x)
    finally:
        rec["busy"] = False
    if new and b in handle.graphs:
        torch.cuda.synchronize()
        log(f"{PREFIX} CAPTURED model#{rec['index']} batch={b} cost={round(time.perf_counter() - t0, 2)}s (one CUDA-graph capture at this batch size's first call)")
    if b in handle.ungraphed and b not in rec["ungraphed_said"]:
        rec["ungraphed_said"].add(b)
        log(_ungraphed_line(rec, b, handle))
    return out


def _ungraphed_line(rec: dict, b: int, handle=None) -> str:
    why = (getattr(handle, "ungraphed", None) or {}).get(b) if handle is not None else None
    return (f"{PREFIX} UNGRAPHED model#{rec['index']} batch={b}: " + (why or "no CUDA graph at this batch size")
            + " — calls of this size run the same patched modules without graph replay (same outputs; the forward is GPU-bound at this size, so replay would save nothing measurable)")
