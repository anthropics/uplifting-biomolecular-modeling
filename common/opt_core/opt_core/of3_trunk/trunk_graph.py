"""The `trunk_graph` lever (exact class): CUDA-graph capture of `PairFormerStack.forward` — the 48-block trunk stack (one call per recycle pass) and
the confidence head's 4-block stack (one call per confidence call): both are launch-bound below ~1000 tokens (fast line at 400 tokens: 0.85 s of
host time per item for ~0.5 s of GPU work). Per KEY = (stack instance, the four tensor arguments' shape/stride/dtype/device, every scalar flag, the
numerics mode: autocast state, deterministic algorithms, fp32 matmul precision, TF32) the FIRST call runs eager (chunk-size tuning, Triton
autotuning and lazy compilation settle there), the SECOND call is captured (static input copies, one process-wide graph pool, a side stream,
`capture_error_mode="thread_local"`, the CUDA RNG offset checked unchanged, the output form checked) and replayed for that call, every later call
copies its inputs into the static buffers, replays and returns CLONES of the static outputs (the next replay rewrites them). Same kernels in the
same order on the same operands: bitwise. One token count resident: a new token count drops every graph of the previous one (like the sampler
add-on's one-generation rule).

Refused BY NAME (the stack's own forward runs; counted `fallback:<reason>`): token count above <KIT>_TRUNK_GRAPH_NMAX (default 1024 — the pool
keeps one stack call's activations resident), grad / training, CPU tensors, a call already under capture, a stack whose attention would run
DeepSpeed's DS4Sci kernel (it launches on the legacy default stream: uncapturable — `use_deepspeed_evo_attention` without cuEquivariance serving,
i.e. at or below cuEquivariance's fallback threshold), low-memory attention, a non-tensor argument, a capture that failed (that key runs eager from
then on), a capture that consumed CUDA RNG, an output that is not the (s, z) pair, and — checked at every replay — a stack whose parameter /
buffer storage moved since capture (a weight reload or a .cpu()/.cuda() round trip in a long-lived process: `storage_moved`, the graph is dropped,
that call runs eager and the next one captures again; never a replay on stale addresses).

The engine's chunk-size tuner (one per stack instance) tunes inside a call whose argument record differs from its cache; a capture is taken
only when it will NOT (`capture_deferred:tuner_unsettled`: that call runs eager and settles the tuner, the next one captures — tuning runs
timed trial evaluations a capture would bake in untimed; a cached record of a form this module does not know counts as unsettled, named
`tuner_format_unknown`: fail-closed). Structurally different records (the confidence stack's batched vs per-sample call forms) are the
`tuner_guard` lever's concern (of3_trunk/tuner_guard.py): with it a process serves items of any mix of sizes.

Census note: no Python runs inside a replay — the per-call censuses of any lever INSIDE the stack (cells serving TriMul / triangle attention /
transition / AttentionPairBias) tick on the eager and capture calls only; `replayed_block_calls` (replays x blocks) is the reconciliation term.

Switch (the kit adapter's, `configure(ENV=…, ENV_NMAX=…)`): <KIT>_TRUNK_GRAPH=1, <KIT>_TRUNK_GRAPH_NMAX=<int>. Exit line
`<PREFIX> LEVER name=trunk_graph state=on nmax=<n> eager_first=<n> captures=<n> replays=<n> replayed_block_calls=<n>
per_call_censuses_exclude_replayed_calls=true drops=<n> fallback=<..>`.
"""
from __future__ import annotations

import atexit
import os
import sys
import time
from typing import Any, Dict, Optional

PREFIX = "[opt_core/of3_trunk.trunk_graph]"
ENV: Optional[str] = None                                # <KIT>_TRUNK_GRAPH=1
ENV_NMAX: Optional[str] = None                           # <KIT>_TRUNK_GRAPH_NMAX (default NMAX_DEFAULT)
VALUES = ("1",)
NMAX_DEFAULT = 1024
M_PF: Optional[str] = None                               # ….core.model.latent.pairformer (class PairFormerStack)
CONFIGURABLE = ("PREFIX", "ENV", "ENV_NMAX", "M_PF")
PF_PARAMS = ("s", "z", "single_mask", "pair_mask", "chunk_size", "use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels",
             "use_lma", "inplace_safe", "_mask_trans")       # PairFormerStack.forward's parameter order (positional calls map by it; the trunk passes keywords)
TENSOR_ARGS = ("s", "z", "single_mask", "pair_mask")


def configure(**kw) -> None:
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "nmax": None, "eager_first": 0, "captures": 0, "replays": 0,
                         "replayed_block_calls": 0, "drops": 0, "fallback": {}, "capture_ms": [], "patched": []}
_G: Dict[str, Any] = {"pool": None, "stream": None, "n_tok": None, "entries": {}}   # entries: key -> {"state": "seen"|"uncapturable"} | {"graph", "static_in", "static_out"}


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


def nmax(environ=None) -> int:
    environ = os.environ if environ is None else environ
    raw = (environ.get(ENV_NMAX) or "").strip() if ENV_NMAX else ""
    if not raw:
        return NMAX_DEFAULT
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{ENV_NMAX}={raw!r} is not an integer token count") from None
    if v < 0:
        raise ValueError(f"{ENV_NMAX}={raw!r} must be >= 0")
    return v


def serving() -> bool:
    return STATE["state"] == "on"


def release() -> int:
    """Drop every captured graph and the pool (an item-boundary release under memory pressure; the next calls run eager then re-capture).
    Returns the number of graphs dropped."""
    n = sum(1 for e in _G["entries"].values() if isinstance(e, dict) and "graph" in e)
    _G["entries"].clear(); _G["n_tok"] = None; _G["pool"] = None
    if n:
        STATE["drops"] += n
    return n


def _cueq_min() -> int:
    try:
        from cuequivariance_ops_torch.triangle_attention import CUEQ_TRIATTN_FALLBACK_THRESHOLD   # noqa: N811
        return int(CUEQ_TRIATTN_FALLBACK_THRESHOLD)
    except Exception as e:  # noqa: BLE001
        from ..oom import is_oom
        if is_oom(e):
            raise
        return 100


def _record_now(cached_entry, t):
    """This call's tuner record entry for tensor `t`, in the form of the engine's cached entry: `torch.Size` (the 3.x engine records `a.shape`)
    or `(torch.Size, itemsize)` (the 0.5.x fork records `(a.shape, a.dtype.itemsize)`); None for a form this guard does not know."""
    import torch
    if isinstance(cached_entry, torch.Size):
        return t.shape
    if isinstance(cached_entry, (tuple, list)) and len(cached_entry) == 2 and isinstance(cached_entry[0], torch.Size) and isinstance(cached_entry[1], int):
        return type(cached_entry)((t.shape, t.dtype.itemsize))
    return None


def _tuner_records(stack, s_, z_, chunk_size):
    """(cached, now) records of upstream's per-stack ChunkSizeTuner for this call, or None when the tuner is not consulted (no tuner, no
    chunking, nothing cached yet); now is None when the cached record has a form this guard does not know (arity != 2 or unknown entries)."""
    tuner = getattr(stack, "chunk_size_tuner", None)
    if tuner is None or chunk_size is None:
        return None
    cached = getattr(tuner, "cached_arg_data", None)
    if cached is None:
        return None
    if not isinstance(cached, (tuple, list)) or len(cached) != 2:                              # upstream records args=(s.clone(), z.clone())
        return cached, None
    now = tuple(_record_now(c, t) for c, t in zip(cached, (s_, z_)))
    return cached, (None if any(n is None for n in now) else now)


def _tuner_settled(stack, s_, z_, chunk_size) -> bool:
    """True when upstream's tuner will NOT tune inside this call (no tuner / no chunking / its cached record equals this call's): a capture
    may only record a settled call — tuning runs timed trial evaluations, which a capture would bake into the graph untimed. A cached record
    of unknown form counts as unsettled (the call runs eager, named): fail-closed."""
    tuner = getattr(stack, "chunk_size_tuner", None)
    if tuner is None or chunk_size is None:
        return True
    recs = _tuner_records(stack, s_, z_, chunk_size)
    if recs is None:                                                      # nothing cached yet: upstream tunes on this call
        return False
    cached, now = recs
    if now is None:
        _count(STATE["fallback"], "tuner_format_unknown")
    return now is not None and tuple(cached) == tuple(now)


def _storage_signature(module) -> tuple:
    """The device addresses the captured graph baked in: every parameter and buffer of the stack (a few hundred data_ptr reads per call)."""
    return tuple(t.data_ptr() for t in module.parameters()) + tuple(t.data_ptr() for t in module.buffers())


def _make_forward(orig):
    import torch
    NMAX = STATE["nmax"]
    CUEQ_MIN = _cueq_min()

    def _sig(t):
        return (tuple(t.shape), tuple(t.stride()), t.dtype, t.device.index)

    def _mode():
        try:
            on = torch.is_autocast_enabled("cuda"); dt = torch.get_autocast_dtype("cuda") if on else None
        except TypeError:
            on = torch.is_autocast_enabled(); dt = torch.get_autocast_gpu_dtype() if on else None
        return (on, dt, torch.are_deterministic_algorithms_enabled(), torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic)

    def _fb(why, self, a, k):
        _count(STATE["fallback"], why)
        return orig(self, *a, **k)

    def forward(self, *a, **k):
        if not serving():
            return orig(self, *a, **k)
        args = dict(zip(PF_PARAMS, a)); args.update(k)
        ts = [args.get(n) for n in TENSOR_ARGS]
        if any(not isinstance(t, torch.Tensor) for t in ts):
            return _fb("non_tensor_arg", self, a, k)
        z_ = ts[1]
        n_tok = int(z_.shape[-2])
        if self.training or (torch.is_grad_enabled() and any(t.requires_grad for t in ts)):
            return _fb("grad", self, a, k)
        if not z_.is_cuda:
            return _fb("cpu", self, a, k)
        if torch.cuda.is_current_stream_capturing():
            return _fb("already_capturing", self, a, k)
        if n_tok > NMAX:
            return _fb("n_tok_gt_nmax", self, a, k)
        if args.get("use_deepspeed_evo_attention") and (not args.get("use_cueq_triangle_kernels") or n_tok <= CUEQ_MIN):
            return _fb("ds4sci_attention", self, a, k)                   # DS4Sci launches on the legacy default stream: uncapturable
        if args.get("use_lma"):
            return _fb("lma", self, a, k)
        consts = tuple((n, v) for n, v in sorted(args.items()) if n not in TENSOR_ARGS and isinstance(v, (bool, int, float, str, type(None))))
        key = (id(self), tuple(_sig(t) for t in ts), consts, _mode())
        if _G["n_tok"] != n_tok:                                          # one token count resident
            dropped = release()
            if dropped:
                _log(f"token count {n_tok} != the resident graphs' -> {dropped} graph(s) dropped")
            _G["n_tok"] = n_tok
        e = _G["entries"].get(key)
        if e is None:                                                     # 1st call of this key: eager
            _G["entries"][key] = {"state": "seen"}
            STATE["eager_first"] += 1
            return orig(self, *a, **k)
        dev = z_.device
        if "graph" not in e:
            if e.get("state") == "uncapturable":
                return _fb("uncapturable_key", self, a, k)
            if not _tuner_settled(self, ts[0], z_, args.get("chunk_size")):    # upstream would tune INSIDE this call: run it eager (it settles the tuner), capture the next one
                STATE["eager_first"] += 1; _count(STATE["fallback"], "capture_deferred:tuner_unsettled")
                return orig(self, *a, **k)
            t0 = time.perf_counter()                                      # 2nd call: capture
            if _G["pool"] is None:
                _G["pool"] = torch.cuda.graph_pool_handle()
            if _G["stream"] is None:
                _G["stream"] = torch.cuda.Stream(device=dev)
            static_in = [t.clone() for t in ts]
            cargs = dict(args)
            for n, t in zip(TENSOR_ARGS, static_in):
                cargs[n] = t
            st = _G["stream"]
            torch.cuda.synchronize(dev)
            st.wait_stream(torch.cuda.current_stream(dev))
            rng_before = torch.cuda.get_rng_state(dev)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, pool=_G["pool"], stream=st, capture_error_mode="thread_local"):
                    static_out = orig(self, **cargs)
            except Exception as ex:  # noqa: BLE001
                from opt_core.oom import is_oom
                if is_oom(ex):
                    raise
                torch.cuda.synchronize(dev)
                _G["entries"][key] = {"state": "uncapturable"}
                _log(f"capture FAILED for n_tok={n_tok} ({type(ex).__name__}: {str(ex)[:200]}) — this key runs eager from now on")
                return _fb(f"capture_failed:{type(ex).__name__}", self, a, k)
            torch.cuda.current_stream(dev).wait_stream(st)
            torch.cuda.synchronize(dev)
            if not torch.equal(rng_before, torch.cuda.get_rng_state(dev)):
                torch.cuda.set_rng_state(rng_before, dev)
                _G["entries"][key] = {"state": "uncapturable"}
                return _fb("consumes_rng", self, a, k)
            if not (isinstance(static_out, tuple) and len(static_out) == 2 and all(isinstance(o, torch.Tensor) for o in static_out)):
                _G["entries"][key] = {"state": "uncapturable"}
                return _fb("output_form", self, a, k)
            e.update(graph=graph, static_in=static_in, static_out=static_out, storage_sig=_storage_signature(self))
            STATE["captures"] += 1
            ms = (time.perf_counter() - t0) * 1e3
            STATE["capture_ms"].append(round(ms))
            _log(f"captured PairFormerStack n_blocks={len(getattr(self, 'blocks', []))} n_tok={n_tok} key#{len(_G['entries'])} in {ms:.0f} ms "
                 "(capture is not execution: replaying now for this call)")
        if e["storage_sig"] != _storage_signature(self):                  # the graph baked the stack's parameter / buffer addresses (and castcache's bf16 copies keyed on
            del _G["entries"][key]                                        # them): a weight reload / .cpu()-.cuda() round trip since capture must never replay — drop the
            return _fb("storage_moved", self, a, k)                       # graph, run this call eager (named); the next call of the signature captures again
        for dst, src in zip(e["static_in"], ts):                          # replay: inputs in, replay, clones out
            dst.copy_(src)
        e["graph"].replay()
        STATE["replays"] += 1; STATE["replayed_block_calls"] += len(getattr(self, "blocks", []))
        return tuple(o.clone() for o in e["static_out"])
    forward.__wrapped__ = orig; forward._of3opt_trunk_graph = True
    return forward


def census_line() -> str:
    return (f"{PREFIX} LEVER name=trunk_graph state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" nmax={STATE['nmax']} eager_first={STATE['eager_first']} captures={STATE['captures']} replays={STATE['replays']}"
            f" replayed_block_calls={STATE['replayed_block_calls']} per_call_censuses_exclude_replayed_calls=true drops={STATE['drops']}"
            f" fallback={','.join('%s:%d' % kv for kv in sorted(STATE['fallback'].items())) or 'none'}"
            + (f" capture_ms={','.join(map(str, STATE['capture_ms'][:8]))}" if STATE["capture_ms"] else ""))


def install(environ=None) -> dict:
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not M_PF:
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_PF=) before install")
    STATE["nmax"] = nmax(environ)
    import importlib
    PF = importlib.import_module(M_PF)
    orig = PF.PairFormerStack.forward
    if not getattr(orig, "_of3opt_trunk_graph", False):
        PF.PairFormerStack.forward = _make_forward(orig); STATE["patched"].append("PairFormerStack.forward")
    STATE.update(installed=True, state="on")
    _log(f"installed: PairFormerStack.forward captured per (instance, input signature, flags, numerics mode) at its 2nd call and replayed after "
         f"(nmax={STATE['nmax']}; 1st call eager; refusals by name run the stack's own forward)")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
