"""The `post_release` lever (exact class): the item-boundary memory release of the graphed lines at large token counts.

The graphed sampler keeps its captured step resident between items (of3_graphs: the graph, the shared pool, the static copies of every step
input, the gather tables; of3t_paircache: the static per-rollout pair tensors — zij 10.7 GiB at 3000 tokens x 5 samples; trunk_graph: the
trunk stack's graphs) so the next item of the same shapes replays without a capture. At 3000 tokens that residency is what the CONFIDENCE head
then lacks: its per-sample pair batch ([samples, N, N, c_z] bf16 = 10.7 GiB at 3000 x 5) is the forward's largest late allocation and the item
fails there with the sampler's state still resident (measured: 72.6 GiB in use, 0 structures written). This lever releases — (1) right after
the sampler returns, BEFORE the confidence head, and (2) again after the item's forward — everything the graph machinery keeps between items
(of3_graphs.release(): generations + gather tables + pool; of3t_paircache.release(): static pair buffers + rollout cache; trunk_graph.release();
apb_hoist's memo; cuBLAS workspaces) and returns the allocator's cached-free segments to the driver, WHEN headroom is low: the item's token count
>= <KIT>_POST_RELEASE_NTOK (default 2400) or free device memory (total - this process's reserved bytes) < <KIT>_POST_RELEASE_MIN_FREE_GB (16).
Below the gate nothing happens (the resident graph keeps serving the next item). After a release the next rollout start flushes the allocator
once more (the trunk that ran in between leaves its dead intermediates as cached-free segments of the default stream, which the sampler add-on's
side-stream warm-up cannot reuse) and of3_graphs flushes before its next capture. Above the gate every item re-captures by design (~0.3 s at 3000
tokens against a ~115 s forward). Outputs untouched by construction: exact (bitwise with the gate forced on every item).

Order and dependence: of3_graphs first; if its generation cannot be dropped (no release API, or its release raised) the sources a resident
graph READS — of3t_paircache's static buffers, apb_hoist's memo — are NOT released, named (`<source>:skipped:graphs_resident`; under
OF3_GRAPHS_STRICT=1 a failed release raises instead): never a resident graph over freed addresses. trunk_graph's graphs read the stack's own
parameters and static inputs only and are released independently. NOT released, by design: the sampler's per-rollout memo store (atom_hoist's
address-stable buffers, capped by its own switch, refreshed in place at the next rollout) and castcache's bf16 weight copies (~0.3 GB, keyed on
the weights) — bounded, item-independent, and re-deriving them costs every item.

Census: which point fired or was kept per item, MiB freed per source (driver free-memory delta after each source's drop + flush) and MiB
dropped per source (live-tensor delta: what the source itself let go of), named fallbacks when a source cannot be released
(`<module>:no_release_api`, `<source>:release_error:<Exc>`, `<source>:skipped:graphs_resident`) or the token count cannot be read
(`n_tok_unreadable:<why>`: the free-memory criterion still decides). Point (2) sits inside the kit's per-item forward timer when the timer wraps
this lever's wrapper (both are installed at the model module's import): <= 0.5 s at 3000 tokens, nothing below the gate.

Switch (the kit adapter's): <KIT>_POST_RELEASE=1; knobs <KIT>_POST_RELEASE_NTOK, <KIT>_POST_RELEASE_MIN_FREE_GB. Exit line
`<PREFIX> LEVER name=post_release state=on ntok=<n> min_free_gb=<g> pre_confidence=<n> post_forward=<n> kept=<n> flushes=<n> freed_mib=<per source>
dropped_mib=<per source> fallback=<..>`.
"""
from __future__ import annotations

import atexit
import gc
import os
import sys
from typing import Any, Dict, Optional

PREFIX = "[opt_core/of3_sampler.post_release]"
ENV: Optional[str] = None                                # <KIT>_POST_RELEASE=1
ENV_NTOK: Optional[str] = None                           # <KIT>_POST_RELEASE_NTOK
ENV_MIN_FREE_GB: Optional[str] = None                    # <KIT>_POST_RELEASE_MIN_FREE_GB
VALUES = ("1",)
NTOK_DEFAULT = 2400
MIN_FREE_GB_DEFAULT = 16.0
M_DIFFUSION: Optional[str] = None                        # ….core.model.structure.diffusion_module (SampleDiffusion: release point 1 wraps its forward)
SAMPLER_CLASS = "SampleDiffusion"
M_MODEL: Optional[str] = None                            # ….projects.of3_all_atom.model (release point 2 wraps MODEL_CLASS.forward)
MODEL_CLASS: Optional[str] = None
M_GRAPHS = "of3_graphs"                                  # the fast-inference add-on's graphed step (release())
M_PAIRCACHE = "of3t_paircache"                           # the trunk-kernels add-on's pair cache (release())
CONFIGURABLE = ("PREFIX", "ENV", "ENV_NTOK", "ENV_MIN_FREE_GB", "M_DIFFUSION", "SAMPLER_CLASS", "M_MODEL", "MODEL_CLASS", "M_GRAPHS", "M_PAIRCACHE")
SD_PARAMS = ("batch", "si_input", "si_trunk", "zij_trunk", "noise_schedule", "no_rollout_samples", "use_conditioning", "chunk_size",
             "use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels", "use_lma", "use_high_precision_attention", "_mask_trans")


def configure(**kw) -> None:
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "ntok": None, "min_free_gb": None, "pre_confidence": 0, "post_forward": 0,
                         "kept": 0, "flushes": 0, "freed_mib": {}, "dropped_mib": {}, "fallback": {}, "released_pending_flush": False, "total": None, "patched": []}


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _count(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    if not ENV:
        return False
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def _knob(environ, name, default, conv):
    raw = (environ.get(name) or "").strip() if name else ""
    if not raw:
        return default
    try:
        v = conv(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number") from None
    if v < 0:
        raise ValueError(f"{name}={raw!r} must be >= 0")
    return v


def serving() -> bool:
    return STATE["state"] == "on"


def _empty_cache(torch):
    """the allocator flush of the defining module (immune to anything placed over torch.cuda.empty_cache)"""
    f = getattr(getattr(torch.cuda, "memory", None), "empty_cache", None)
    (f if callable(f) else torch.cuda.empty_cache)()


def _free_bytes(torch) -> int:
    """free device memory for the gate from the allocator's bookkeeping (total - this process's reserved bytes; no driver round-trip)"""
    if STATE["total"] is None:
        STATE["total"] = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    return STATE["total"] - torch.cuda.memory_reserved()


def _n_tok(batch) -> Optional[int]:
    """the item's token count from the sampler's batch (None, counted `n_tok_unreadable:<why>`, when the batch has no readable token mask: the
    free-memory criterion of the gate still decides)"""
    try:
        tm = batch.get("token_mask") if isinstance(batch, dict) else None
        if tm is None:
            _count(STATE["fallback"], "n_tok_unreadable:no_token_mask"); return None
        return int(tm.shape[-1])
    except Exception as e:  # noqa: BLE001
        from ..oom import is_oom
        if is_oom(e):
            raise
        _count(STATE["fallback"], f"n_tok_unreadable:{type(e).__name__}"); return None


def _gate(torch, n_tok) -> Optional[str]:
    """the reason to release now, or None (keep)"""
    if n_tok is not None and n_tok >= STATE["ntok"]:
        return f"n_tok_{n_tok}_ge_{STATE['ntok']}"
    free = _free_bytes(torch)
    if free < STATE["min_free_gb"] * 2 ** 30:
        return "free_%.1fGB_lt_%.0fGB" % (free / 2 ** 30, STATE["min_free_gb"])
    return None


def release(torch, why: str, point: str) -> None:
    """Drop every between-items resident of the graph machinery, source by source, flushing the allocator after each and recording the MiB of
    device memory each returned (driver free-memory delta)."""
    dev = torch.cuda.current_device()
    torch.cuda.synchronize(dev)
    free0 = torch.cuda.mem_get_info(dev)[0]; last = free0
    alloc0 = torch.cuda.memory_allocated(dev); last_alloc = alloc0

    def measured(name, fn) -> bool:
        nonlocal last, last_alloc
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            from ..oom import is_oom
            if is_oom(e):
                raise
            _count(STATE["fallback"], f"{name}:release_error:{type(e).__name__}")
            _log(f"release of {name} FAILED ({type(e).__name__}: {e}) — left as is")
            if os.environ.get("OF3_GRAPHS_STRICT") == "1":               # a strict graphed line: a half-released item boundary is an error, not a census word
                raise
            return False
        gc.collect(); _empty_cache(torch)
        free = torch.cuda.mem_get_info(dev)[0]; alloc = torch.cuda.memory_allocated(dev)
        _count(STATE["freed_mib"], name, int((free - last) / 2 ** 20)); last = free                 # device memory returned to the driver after this source
        _count(STATE["dropped_mib"], name, int((last_alloc - alloc) / 2 ** 20)); last_alloc = alloc   # live tensor bytes this source let go of
        return True

    graphs_resident = False                                               # the graphed step's generation could not be dropped: everything a resident graph reads
    gm = sys.modules.get(M_GRAPHS)                                        # (the pair cache's static buffers, the key-mask memo) must stay — skipped BY NAME below
    if gm is not None:
        fn = getattr(gm, "release", None)
        if not callable(fn):
            _count(STATE["fallback"], f"{M_GRAPHS}:no_release_api"); graphs_resident = True
        elif not measured(M_GRAPHS, fn):
            graphs_resident = True
    pm = sys.modules.get(M_PAIRCACHE)
    if pm is not None:
        fn = getattr(pm, "release", None)
        if not callable(fn):
            _count(STATE["fallback"], f"{M_PAIRCACHE}:no_release_api")
        elif graphs_resident:
            _count(STATE["fallback"], f"{M_PAIRCACHE}:skipped:graphs_resident")
        else:
            measured(M_PAIRCACHE, fn)
    tg = sys.modules.get("opt_core.of3_trunk.trunk_graph")
    if tg is not None:
        measured("trunk_graph", tg.release)                               # its graphs read the stack's own parameters and static inputs only: independent of the above
    ah = sys.modules.get("opt_core.of3_trunk.apb_hoist")
    if ah is not None:
        if graphs_resident:
            _count(STATE["fallback"], "apb_hoist:skipped:graphs_resident")
        else:
            measured("apb_hoist", ah.release)
    clear_ws = getattr(getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None)
    if callable(clear_ws):
        measured("cublas_workspaces", clear_ws)
    else:
        _count(STATE["fallback"], "cublas_workspaces:no_api")
    measured("allocator_cache", lambda: None)                              # whatever else was cached-free
    STATE["released_pending_flush"] = True
    _count(STATE, point)
    _log("release at %s (%s): %.1f GB free -> %.1f GB free; live tensors %.1f GB -> %.1f GB; dropped MiB %s" % (
        point, why, free0 / 2 ** 30, last / 2 ** 30, alloc0 / 2 ** 30, last_alloc / 2 ** 30, dict(STATE["dropped_mib"])))


def _make_sampler_forward(orig):
    import torch

    def forward(self, *a, **k):
        if serving() and STATE["released_pending_flush"]:                 # a release since the last rollout: the trunk that just ran left its dead intermediates as
            STATE["released_pending_flush"] = False                       # cached-free default-stream segments; return them to the driver BEFORE the pair cache's static
            torch.cuda.synchronize(); _empty_cache(torch); STATE["flushes"] += 1   # buffers pin them and the add-on's side-stream warm-up needs fresh device memory
        out = orig(self, *a, **k)
        if serving() and not self.training:
            args = dict(zip(SD_PARAMS, a)); args.update(k)
            why = _gate(torch, _n_tok(args.get("batch")))
            if why:
                release(torch, why, "pre_confidence")
            else:
                STATE["kept"] += 1
        return out
    forward.__wrapped__ = orig; forward._of3opt_post_release = True
    for att in dir(orig):                                                 # foreign markers of the callable beneath stay visible (other add-ons' idempotence checks)
        if att.startswith("_of3") and not hasattr(forward, att):
            try:
                setattr(forward, att, getattr(orig, att))
            except Exception as e:  # noqa: BLE001
                from ..oom import is_oom
                if is_oom(e):
                    raise
                pass
    return forward


def _make_model_forward(orig):
    import torch

    def forward(self, batch, *a, **k):
        try:
            return orig(self, batch, *a, **k)
        finally:
            if serving() and not self.training and torch.cuda.is_available():
                why = _gate(torch, _n_tok(batch))
                if why:
                    release(torch, why, "post_forward")
    forward.__wrapped__ = orig; forward._of3opt_post_release = True
    for att in dir(orig):
        if att.startswith("_of3") and not hasattr(forward, att):
            try:
                setattr(forward, att, getattr(orig, att))
            except Exception as e:  # noqa: BLE001
                from ..oom import is_oom
                if is_oom(e):
                    raise
                pass
    return forward


def census_line() -> str:
    freed = ",".join("%s:%d" % kv for kv in sorted(STATE["freed_mib"].items())) or "none"
    dropped = ",".join("%s:%d" % kv for kv in sorted(STATE["dropped_mib"].items())) or "none"
    return (f"{PREFIX} LEVER name=post_release state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" ntok={STATE['ntok']} min_free_gb={STATE['min_free_gb']} pre_confidence={STATE['pre_confidence']} post_forward={STATE['post_forward']}"
            f" kept={STATE['kept']} flushes={STATE['flushes']} freed_mib={freed} dropped_mib={dropped}"
            f" fallback={','.join('%s:%d' % kv for kv in sorted(STATE['fallback'].items())) or 'none'}")


def install(environ=None) -> dict:
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not (M_DIFFUSION and M_MODEL and MODEL_CLASS):
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_DIFFUSION=, M_MODEL=, MODEL_CLASS=) before install")
    STATE["ntok"] = _knob(environ, ENV_NTOK, NTOK_DEFAULT, int)
    STATE["min_free_gb"] = _knob(environ, ENV_MIN_FREE_GB, MIN_FREE_GB_DEFAULT, float)
    import importlib
    D = importlib.import_module(M_DIFFUSION); M = importlib.import_module(M_MODEL)
    S = getattr(D, SAMPLER_CLASS); C = getattr(M, MODEL_CLASS)
    if not getattr(S.forward, "_of3opt_post_release", False):
        S.forward = _make_sampler_forward(S.forward); STATE["patched"].append(f"{SAMPLER_CLASS}.forward")
    if not getattr(C.forward, "_of3opt_post_release", False):
        C.forward = _make_model_forward(C.forward); STATE["patched"].append(f"{MODEL_CLASS}.forward")
    STATE.update(installed=True, state="on")
    _log(f"installed: release points after {SAMPLER_CLASS}.forward (pre_confidence) and after {MODEL_CLASS}.forward (post_forward); gate n_tok >= {STATE['ntok']} "
         f"or free < {STATE['min_free_gb']:g} GB (allocator bookkeeping); sources {M_GRAPHS}, {M_PAIRCACHE}, trunk_graph, apb_hoist, cuBLAS workspaces, allocator cache")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
