"""Lever ``prefetch``: the persistent featurizer — the NEXT item's Transform pipeline runs in one side process while the current
item's forward runs; the fold process never featurizes on its critical path after the first item.

What it replaces. ``RF3InferenceEngine.run`` (``rf3/inference_engines/rf3.py``) is fully serial per item: ``self.pipeline(
input_spec.to_pipeline_input())`` (atomworks parsing / MSA loading and filtering — a ``multiprocessing.Pool`` forked per item from the
process that holds the CUDA context — templates, atomization: seconds per item, growing with token count) → ``to_device`` →
the forward → the writers. Nothing overlaps; the GPU idles while the CPU featurizes and vice versa.

What it does instead (``sidecar.py`` is the process plumbing):

* at ``BaseInferenceEngine._construct_pipeline`` (the pipeline exists, the trainer / model / CUDA context do not) ONE helper process is
  forked holding the engine's own pipeline object; it lowers its priority and waits for work; the fold process goes on to build its
  model. ``self.pipeline`` becomes a :class:`Proxy` around the real pipeline.
* ``run()``'s ``DataLoader`` (the module's name, subclassed) hands the proxy the item ORDER the stock sampler yields (the
  ``InferenceInput`` objects, already in memory; the stock iterator is created at the same statement, so its one ``base_seed`` draw
  from torch's CPU generator happens where stock draws it) and the helper the same list.
* item 0 of the process is featurized TWICE from the same RNG state — by the stock statement in the fold process (that output is the
  one used) and by the helper — and the two outputs are compared leaf by leaf (dtype, shape, stride, bytes; atom arrays by their
  annotations, bonds and coordinates): ``PREFETCH first_input=same`` or ``first_input=DIFFERENT:<path>``, in which case the lever steps
  aside for the rest of the process BY NAME (every later item featurized by the stock statement; ``source=parent:first_input``).
* every later item k: the proxy captures the fold process's RNG states (``random``, ``numpy.random``, torch CPU) at the pipeline call —
  the point where stock would start drawing — and asks the helper for item k with that state. The helper has been featurizing item k
  SPECULATIVELY since it answered k−1, from its own post-(k−1) state; if that state equals the one the parent sends (it does once the
  helper has learnt the forward's own CPU-generator draws — the sampler's per-step rotations / translations, ``shadow=TxD`` — and
  advanced its state by them; the writers draw nothing) the speculative output is
  the answer at once (``speculated=1 wait_s≈0``); if not, the helper discards it and featurizes k again from the parent's state
  (``respec=1``: correct, merely not overlapped). Either way the parent then SETS its RNG states to the helper's post-featurization
  states, so the forward starts from exactly the state stock's in-process featurization would have left. Same statements, same
  objects, same RNG stream → the same features; the model never sees a different byte (the first-item comparison is the run-time
  witness, the state check the per-item one).
* named step-asides (``source=parent:<reason>``, counted): ``n_items=1`` (nothing to overlap), ``n_gpu>1`` (the row-sharded line: not
  armed there), ``order`` (the loader yielded an item the proxy was not told about), ``helper_error`` / ``helper_gone`` (the helper's
  traceback is printed once; the parent featurizes from the state it captured), ``first_input``.

Cost model: featurization leaves the item period entirely when it is shorter than forward + post-processing; the helper's CPU work now
competes with the fold process's launch thread (``nice`` + the machine's core count decide how much). Host memory: one extra process holding the pipeline and one item's features; no device memory.
"""
from __future__ import annotations

import hashlib
import pickle
import random
import sys
import time
import traceback
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import sidecar as _sc

NAME = "prefetch"
ENGINE_MODULE = "rf3.inference_engines.rf3"                  # RF3InferenceEngine (run's loop, its DataLoader name) — the trigger
ENGINE_CLASS = "RF3InferenceEngine"
CONSTRUCT = "_construct_pipeline"                           # BaseInferenceEngine._construct_pipeline: sets self.pipeline; the fork point
TEMPL_MODULE, TEMPL_FUNCTION = "rf3.utils.inference", "apply_template_selection"   # templ.py's census wrapper: the helper runs the stock function (the census is the parent's)
PREFIX = "[rosettafold3-opt]"
FIRST_TIMEOUT_S = 1800.0                                    # the helper's answer for one item (a featurization is seconds to a minute; a hung helper is named, not waited on forever)

STATE: Dict[str, Any] = {"armed": False, "installed": False, "on": False, "reason": None, "off_reason": None, "conflict": None, "n_gpu": 1,
                         "helper_pid": None, "forked_before_cuda": None, "threads_at_fork": None, "nice": _sc.NICE,
                         "first_input": None, "first_input_path": None, "split": None, "trace_top": None, "trace_rng": None}
CENSUS: Counter = Counter()
ITEMS: List[dict] = []                                      # per-item records (last 64)
_ORIG: Dict[str, Any] = {}
_PROXIES: List["Proxy"] = []


class PrefetchRefused(RuntimeError):
    """The lever cannot be installed (upstream reshaped) — named on the record, the fold runs stock."""


# ---------------------------------------------------------------- RNG states (python, numpy legacy, torch CPU)
def _torch():
    t = sys.modules.get("torch")
    if t is not None:
        return t
    try:
        import torch
        return torch
    except Exception:                                       # noqa: BLE001 — the CPU tests run without torch: its state is None on both sides
        return None


def rng_get() -> tuple:
    """(python ``random`` state, numpy legacy global state, torch default CPU generator state) — the generators an in-process
    featurization can draw from (CUDA generators are the model's: the helper never holds a CUDA context)."""
    import numpy as np
    t = _torch()
    return (random.getstate(), np.random.get_state(), t.random.get_rng_state() if t is not None else None)


def rng_set(s: tuple) -> None:
    import numpy as np
    random.setstate(s[0])
    np.random.set_state(s[1])
    t = _torch()
    if t is not None and s[2] is not None:
        t.random.set_rng_state(s[2])


def rng_eq(a: tuple, b: tuple) -> bool:
    return all(rng_eq_parts(a, b))


def rng_digest(s: tuple) -> str:
    import numpy as np
    h = hashlib.sha256()
    h.update(pickle.dumps(s[0], protocol=4))
    h.update(str(s[1][0]).encode()); h.update(np.asarray(s[1][1]).tobytes()); h.update(repr(tuple(s[1][2:])).encode())
    if s[2] is not None:
        h.update(bytes(s[2].numpy().tobytes()))
    return h.hexdigest()[:10]


GENS = ("py", "np", "torch")                                # the three CPU generators, in rng_get() order


def rng_eq_parts(a: tuple, b: tuple) -> Tuple[bool, bool, bool]:
    """Per-generator equality of two rng_get() triples (python ``random``, numpy global, torch CPU)."""
    import numpy as np
    try:
        py = a[0] == b[0]
    except Exception:                                       # noqa: BLE001
        py = False
    try:
        na, nb = a[1], b[1]
        npq = bool(na[0] == nb[0] and np.array_equal(np.asarray(na[1]), np.asarray(nb[1])) and tuple(na[2:]) == tuple(nb[2:]))
    except Exception:                                       # noqa: BLE001
        npq = False
    if (a[2] is None) or (b[2] is None):
        tq = (a[2] is None) and (b[2] is None)
    else:
        try:
            tq = bool(_torch().equal(a[2], b[2]))
        except Exception:                                   # noqa: BLE001
            tq = False
    return (py, npq, tq)


def rng_used(pre: tuple, post: tuple) -> Tuple[bool, bool, bool]:
    """Which generators a featurization ADVANCED (state before != after): the ones its output can depend on. A generator it left untouched
    was not drawn from (these PRNGs cannot be read without advancing), so the parent's value of it is irrelevant to the output."""
    return tuple(not e for e in rng_eq_parts(pre, post))


def rng_merge(base: tuple, other: tuple, mask: Tuple[bool, bool, bool]) -> tuple:
    """``other``'s state for the generators in ``mask``, ``base``'s for the rest."""
    return tuple(other[i] if mask[i] else base[i] for i in range(3))


def used_word(mask) -> str:
    return "+".join(g for g, m in zip(GENS, mask) if m) or "-"


def rng_digest3(s: tuple) -> str:
    """py:/np:/t: digests (6 hex each) of one rng_get() triple, for the report line."""
    import numpy as np
    hp = hashlib.sha256(pickle.dumps(s[0], protocol=4)).hexdigest()[:6]
    hn = hashlib.sha256(str(s[1][0]).encode() + np.asarray(s[1][1]).tobytes() + repr(tuple(s[1][2:])).encode()).hexdigest()[:6]
    ht = hashlib.sha256(bytes(s[2].numpy().tobytes())).hexdigest()[:6] if s[2] is not None else "-"
    return f"py:{hp}/np:{hn}/t:{ht}"


WORDS_LIMIT = 400_000                                       # the forward's CPU-generator draws searched up to this many 32-bit words (~2 s once)


def torch_words_between(a, b, limit: int = WORDS_LIMIT) -> Optional[int]:
    """How many 32-bit outputs of torch's CPU Mersenne twister take state ``a`` to state ``b`` (None when ``b`` is not on ``a``'s stream
    within ``limit`` words, or torch is absent). Used to learn, from the parent's own states, how far one item's forward advances the CPU
    generator (RF3's sampler draws its per-step random rotations / translations on the CPU: a count fixed by the sampling settings, not by
    the item) so the helper can start the next speculation from the state the parent WILL have. A wrong guess costs nothing but the
    overlap (the per-generator check at the hand-off then re-featurizes: respec)."""
    t = _torch()
    if t is None or a is None or b is None:
        return None
    try:
        g = t.Generator()
        g.set_state(a)
        one = t.empty(1, dtype=t.int64)
        for w in range(int(limit) + 1):
            if t.equal(g.get_state(), b):
                return w
            one.random_(0, 65536, generator=g)             # a range below 2**32: exactly one 32-bit word per element on the CPU generator
    except Exception:                                       # noqa: BLE001
        return None
    return None


def torch_advance(words: int) -> None:
    """Advance torch's DEFAULT CPU generator by ``words`` 32-bit outputs (the helper, before it speculates the next item)."""
    t = _torch()
    if t is None or not words or words <= 0:
        return
    t.empty(int(words), dtype=t.int64).random_(0, 65536)


# ---------------------------------------------------------------- leaf-by-leaf comparison of two pipeline outputs
def compare(a, b, path: str = "") -> Optional[str]:
    """None when ``a`` and ``b`` are the same tree (containers by structure; tensors / arrays by dtype, shape, stride and bytes; atom
    arrays by annotations, bonds, box and coordinates; other objects by their attributes or pickled bytes); else the first differing path."""
    import numpy as np
    torch = sys.modules.get("torch")
    if type(a) is not type(b):
        return f"{path}:type({type(a).__name__}!={type(b).__name__})"
    if torch is not None and isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            return f"{path}:tensor({a.dtype},{tuple(a.shape)}!={b.dtype},{tuple(b.shape)})"
        if a.stride() != b.stride():
            return f"{path}:stride({a.stride()}!={b.stride()})"
        if a.device.type != "cpu" or b.device.type != "cpu":
            return None if torch.equal(a.cpu(), b.cpu()) else f"{path}:values"
        ab = a.detach().contiguous().view(-1).view(torch.uint8) if a.numel() else a.detach().reshape(0)
        bb = b.detach().contiguous().view(-1).view(torch.uint8) if b.numel() else b.detach().reshape(0)
        return None if torch.equal(ab, bb) else f"{path}:bytes"
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return f"{path}:array({a.dtype},{a.shape}!={b.dtype},{b.shape})"
        if a.dtype == object:
            for i, (x, y) in enumerate(zip(a.ravel().tolist(), b.ravel().tolist())):
                d = compare(x, y, f"{path}[{i}]")
                if d:
                    return d
            return None
        return None if np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes() else f"{path}:bytes"
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            return f"{path}:keys({list(a.keys())}!={list(b.keys())})"
        for k in a:
            d = compare(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}:len({len(a)}!={len(b)})"
        for i, (x, y) in enumerate(zip(a, b)):
            d = compare(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, (str, bytes, int, float, bool, type(None), complex)):
        if isinstance(a, float) and a != a and b != b:
            return None
        return None if a == b else f"{path}:value({a!r}!={b!r})"
    if isinstance(a, (set, frozenset)):
        return None if a == b else f"{path}:set"
    mod = type(a).__module__ or ""
    if mod.startswith("biotite") and hasattr(a, "get_annotation_categories"):       # AtomArray / AtomArrayStack
        ca, cb = sorted(a.get_annotation_categories()), sorted(b.get_annotation_categories())
        if ca != cb:
            return f"{path}:annotations({ca}!={cb})"
        for c in ca:
            d = compare(a.get_annotation(c), b.get_annotation(c), f"{path}.{c}")
            if d:
                return d
        d = compare(getattr(a, "coord", None), getattr(b, "coord", None), f"{path}.coord")
        if d:
            return d
        d = compare(getattr(a, "box", None), getattr(b, "box", None), f"{path}.box")
        if d:
            return d
        ba, bb2 = getattr(a, "bonds", None), getattr(b, "bonds", None)
        if (ba is None) != (bb2 is None):
            return f"{path}.bonds:presence"
        if ba is not None:
            d = compare(ba.as_array(), bb2.as_array(), f"{path}.bonds")
            if d:
                return d
        return None
    da, db = getattr(a, "__dict__", None), getattr(b, "__dict__", None)
    if isinstance(da, dict) and isinstance(db, dict) and da:
        return compare(da, db, f"{path}({type(a).__name__})")
    try:
        if a == b:
            return None
    except Exception:                                       # noqa: BLE001
        pass
    try:
        return None if pickle.dumps(a, protocol=4) == pickle.dumps(b, protocol=4) else f"{path}:pickle({type(a).__name__})"
    except Exception:                                       # noqa: BLE001
        return f"{path}:uncomparable({type(a).__name__})"


def payload_mib(obj) -> float:
    """Approximate size of a pipeline output (tensor / array bytes), MiB — a report word, not a measurement of the pipe."""
    import numpy as np
    torch = sys.modules.get("torch")
    n = 0
    stack = [obj]
    seen = set()
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if torch is not None and isinstance(x, torch.Tensor):
            n += x.numel() * x.element_size()
        elif isinstance(x, np.ndarray):
            n += x.nbytes
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        elif hasattr(x, "get_annotation_categories"):
            try:
                stack.extend(x.get_annotation(c) for c in x.get_annotation_categories())
                stack.append(getattr(x, "coord", None))
            except Exception:                               # noqa: BLE001
                pass
    return n / 2 ** 20


# ---------------------------------------------------------------- the helper (child) side
def _unwrap_templ_census() -> None:
    """In the helper: run the stock ``apply_template_selection`` (templ.py's census belongs to the fold process, which runs the wrapped
    statement itself for every item; the helper's speculative ``to_pipeline_input`` would otherwise print each TEMPLATE line twice)."""
    m = sys.modules.get(TEMPL_MODULE)
    fn = getattr(m, TEMPL_FUNCTION, None) if m is not None else None
    if fn is not None and getattr(fn, "_rosettafold3_opt_gate", False) and getattr(fn, "__wrapped__", None) is not None:
        setattr(m, TEMPL_FUNCTION, fn.__wrapped__)


def transforms_of(pipeline) -> Optional[list]:
    """The pipeline's transform list when it is an atomworks ``Compose`` (``.transforms`` + ``_stop_before`` in its call), else None."""
    ts = getattr(pipeline, "transforms", None)
    try:
        return list(ts) if ts is not None and len(ts) > 0 else None
    except Exception:                                       # noqa: BLE001
        return None


def run_tail(pipeline, data, start: int):
    """Transforms ``start..`` of the Compose applied one by one — the statements ``Compose.forward`` executes for them (each transform's
    own ``__call__``: history, validation, timing as in the full call)."""
    for t in transforms_of(pipeline)[start:]:
        data = t(data)
    return data


def split_index(trace: list, gens_mask) -> Optional[int]:
    """The first transform (trace order) that advanced any generator in ``gens_mask``; None when none did."""
    for idx, name, secs, used in trace:
        if any(u and g for u, g in zip(used, gens_mask)):
            return idx
    return None


SHADOW_MAX_T, SHADOW_MAX_D = 512, 16


def shadow_step(D: int, generator=None) -> None:
    """One diffusion step's CPU-generator draws, the statements RF3's sampler executes per step (stock
    ``foundry.utils.rotation_augmentation.get_random_augmentation`` and the kit's ``_predraw`` alike): ``uniform_random_rotation((D,))`` =
    three ``torch.rand((D,))``, then ``torch.normal(mean=0, std=1, size=(D, 1, 3))`` — same calls, same sizes, hence the same generator
    advance whatever torch does inside them."""
    t = _torch()
    if generator is None:
        t.rand((D,)); t.rand((D,)); t.rand((D,))
        t.normal(mean=0.0, std=1.0, size=(D, 1, 3))
    else:
        t.rand((D,), generator=generator); t.rand((D,), generator=generator); t.rand((D,), generator=generator)
        t.normal(mean=0.0, std=1.0, size=(D, 1, 3), generator=generator)


def shadow_learn(a, b, d_hint: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """(T, D) such that T shadow steps of batch D take torch CPU state ``a`` to ``b`` — the forward's whole CPU draw between two
    featurizations — or None. ``d_hint`` (the engine's diffusion_batch_size) is tried first."""
    t = _torch()
    if t is None or a is None or b is None:
        return None
    order = ([int(d_hint)] if d_hint else []) + [d for d in range(1, SHADOW_MAX_D + 1) if d != d_hint]
    try:
        g = t.Generator()
        for D in order:
            g.set_state(a)
            for T in range(1, SHADOW_MAX_T + 1):
                shadow_step(D, g)
                if t.equal(g.get_state(), b):
                    return (T, D)
    except Exception:                                       # noqa: BLE001
        return None
    return None


def shadow_apply(TD) -> None:
    """Advance torch's default CPU generator as one item's forward will (T steps of batch D)."""
    if TD:
        for _ in range(int(TD[0])):
            shadow_step(int(TD[1]))


FWD_GENS_DEFAULT = (False, False, True)                      # RF3's forward draws the torch CPU generator (the sampler's per-step rotations); also observed per item (Proxy.fwd)


def _serve(rx, tx, pipeline, d_hint: Optional[int] = None) -> None:
    """The helper's loop. Messages: ``("order", [InferenceInput…])``; ``("featurize", seq, pipeline_input, parent_rng, fwd_gens)``;
    ``("nosplit",)``. Replies (one per featurize): ``(kind, seq, payload, post_rng, seconds, speculated, respec, used_word, extra)`` with
    kind ``ok`` (payload = the pipeline output), ``head`` (payload = the data after transforms[:split]; the parent applies the rest;
    extra = {"split": i, "check": (data_mid_item0, post_head_rng, used_head) once}) or ``error`` (payload = traceback). Item 0 is
    featurized transform by transform (the per-transform trace: seconds and generators advanced), later items speculatively one ahead —
    the head only (transforms before the first one that draws a generator the forward also draws) once the split is known."""
    _unwrap_templ_census()
    ts = transforms_of(pipeline)
    S: Dict[str, Any] = {"order": [], "next": None, "spec": {}, "trace": None, "split": None, "nosplit": ts is None, "check": None,
                         "fwd": FWD_GENS_DEFAULT, "p0": None,
                         "shadow": None, "shadow_tries": 0, "shadow_word": "unlearnt", "after": None, "mid": None}   # the forward's CPU draw pattern (T, D), once learnt

    def featurize(pin, head: Optional[int] = None, traced: bool = False):
        """(kind, pre, post, payload, secs, head, trace) — full pipeline (head None), its first ``head`` transforms, or (traced) the full
        pipeline transform by transform with a per-transform (idx, name, secs, used) trace."""
        pre = rng_get()
        t0 = time.perf_counter()
        try:
            if traced and ts is not None:
                trace = []
                data = pin
                for idx, t in enumerate(ts):
                    r0, q0 = rng_get(), time.perf_counter()
                    data = t(data)
                    trace.append((idx, type(t).__name__, round(time.perf_counter() - q0, 3), rng_used(r0, rng_get())))
                return ("ok", pre, rng_get(), data, time.perf_counter() - t0, None, trace)
            if head is not None:
                out = pipeline(pin, _stop_before=head)
                return ("head", pre, rng_get(), out, time.perf_counter() - t0, head, None)
            out = pipeline(pin)
            return ("ok", pre, rng_get(), out, time.perf_counter() - t0, None, None)
        except Exception:                                   # noqa: BLE001 — the parent re-runs the statement and prints this
            return ("error", pre, rng_get(), traceback.format_exc(), time.perf_counter() - t0, head, None)

    def head_now() -> Optional[int]:
        """The split for the NEXT speculation: none (the whole pipeline) once the forward's draws are predicted (shadow) or when the
        pipeline does not split; else the first forward-coupled transform."""
        if S["shadow"] is not None:
            return None
        return None if (S["nosplit"] or S["split"] is None) else S["split"]

    def continue_as_parent(reply_kind, post) -> None:
        """Put this process where the parent stands after ITS featurization of the item just answered (a head answer: the tail's draws
        replayed here on the retained head output), then where it WILL stand after the item's forward (the shadow); note the former."""
        rng_set(post)
        if reply_kind == "head" and S["mid"] is not None:
            try:
                run_tail(pipeline, S["mid"][0], S["mid"][1])   # the tail's generator draws, as the parent makes them (output discarded)
            except Exception:                               # noqa: BLE001
                pass
        S["mid"] = None
        S["after"] = rng_get()
        if S["shadow"] is not None:
            shadow_apply(S["shadow"])

    def idle() -> bool:
        k = S["next"]
        if k is None or k >= len(S["order"]) or k in S["spec"]:
            return False
        try:
            pin = S["order"][k].to_pipeline_input()
        except Exception:                                   # noqa: BLE001
            S["spec"][k] = ("error", rng_get(), rng_get(), traceback.format_exc(), 0.0, None, None)
            S["next"] = None
            return True
        S["spec"][k] = featurize(pin, head=head_now())
        S["next"] = None                                    # one item ahead, no further (memory; a deeper queue buys nothing when featurize < forward)
        return True

    def learn_shadow(pstate) -> None:
        """From the state this process mirrored after the previous item (S['after']) to the parent's state now: the forward's draws."""
        if S["shadow"] is not None or S["shadow_tries"] >= 2 or S["after"] is None or not S["fwd"][2]:
            return
        S["shadow_tries"] += 1
        t0 = time.perf_counter()
        td = shadow_learn(S["after"][2], pstate[2], d_hint)
        S["shadow_word"] = (f"{td[0]}x{td[1]}" if td else "not_found") + f"@try{S['shadow_tries']}:{time.perf_counter() - t0:.2f}s"
        if td:
            S["shadow"] = td                                # the pending head speculation (if any) stays valid; the shadow serves the next one

    def first(seq, pin, pstate):
        """Item 0: the traced full run (this output is compared with the parent's leaf by leaf), then — when the pipeline splits — the
        head once more from the same state for the parent's split check, then speculation of item 1 starts."""
        S["p0"] = pstate
        rng_set(pstate)
        r = featurize(pin, traced=True)
        S["trace"] = r[6]
        if r[0] == "ok" and S["trace"] and not S["nosplit"]:
            S["split"] = split_index(S["trace"], S["fwd"])
        extra = {"trace": S["trace"], "split": S["split"], "n_transforms": len(ts) if ts else None}
        reply = (r[0], seq, r[3], r[2], r[4], False, False, used_word(rng_used(r[1], r[2])), extra)
        _sc.send(tx, reply)                                 # answer first: the parent is waiting to compare
        post_full = r[2]
        if r[0] == "ok" and head_now() is not None:
            try:
                rng_set(pstate)
                h = featurize(S["order"][seq].to_pipeline_input() if seq < len(S["order"]) else pin, head=S["split"])
                if h[0] == "head":
                    S["check"] = (h[3], h[2], rng_used(h[1], h[2]), S["split"])
            except Exception:                               # noqa: BLE001 — no check: the parent keeps the split off by name
                S["check"] = None
        rng_set(post_full)                                  # continue from where the parent continues (its own full run's post-state)
        S["after"] = rng_get()
        S["next"] = seq + 1
        return None                                         # (already sent)

    def handle(msg):
        kind = msg[0]
        if kind == "order":
            S["order"] = list(msg[1]); S["spec"].clear(); S["next"] = None
            return None
        if kind == "nosplit":
            S["nosplit"] = True; S["spec"].clear()
            return None
        if kind == "featurize":
            _, seq, pin, pstate, fwd = msg
            if S["trace"] is None and seq == 0 or S["p0"] is None:
                return first(seq, pin, pstate)
            acc = tuple(bool(a or b) for a, b in zip(S["fwd"], fwd)) if fwd is not None else S["fwd"]
            if acc != tuple(S["fwd"]):                      # the parent saw the forward draw a generator not yet accounted: the split moves earlier
                S["fwd"] = acc
                if S["trace"]:
                    new_split = split_index(S["trace"], S["fwd"])
                    if new_split != S["split"]:
                        S["split"] = new_split
                        S["spec"].clear()
            learn_shadow(pstate)
            r = S["spec"].pop(seq, None)
            for stale in [k for k in S["spec"] if k < seq]:
                del S["spec"][stale]
            valid = False
            if r is not None and (r[0] == "ok" or (r[0] == "head" and not S["nosplit"])):   # a head made before the shadow was learnt still stands
                used = rng_used(r[1], r[2])                 # the speculative run stands iff the parent's state equals its start state on every
                same = rng_eq_parts(r[1], pstate)           # generator it advanced (the others cannot have shaped its output)
                valid = all(s or not u for s, u in zip(same, used))
            if r is not None and not valid and S["shadow"] is not None and r[0] == "ok":   # the predicted state missed: the split again from here
                S["shadow"], S["shadow_word"] = None, S["shadow_word"] + f",miss@{seq}"
            extra = {"split": (r[5] if (r is not None and valid) else None), "check": None, "shadow": S["shadow_word"]}
            if S["check"] is not None:                      # the item-0 head for the parent's one split check rides the first answer after it exists
                extra["check"] = S["check"]; S["check"] = None
            if valid:
                post = rng_merge(pstate, r[2], used)        # the parent continues: advanced generators from the featurizer, the rest its own
                reply = (r[0], seq, r[3], post, r[4], True, False, used_word(used), extra)
                S["mid"] = (r[3], r[5]) if r[0] == "head" else None
            else:
                respec = r is not None
                rng_set(pstate)
                rr = featurize(pin)                         # from the parent's own state: the full pipeline (exact whatever the split)
                post = rr[2]
                reply = (rr[0], seq, rr[3], post, rr[4], False, respec, used_word(rng_used(rr[1], rr[2])), extra)
                S["mid"] = None
            _sc.send(tx, reply)                             # answer first, then mirror the parent and speculate
            continue_as_parent(reply[0], post)
            S["next"] = seq + 1
            return None
        if kind == "speculate":                             # ("speculate", seq): start item seq now
            S["next"] = int(msg[1])
            return None
        return ("error", -1, f"unknown message {kind!r}", None, 0.0, False, False, "-", {})

    _sc.serve_loop(rx, tx, handle, idle)


# ---------------------------------------------------------------- the fold process side
def _say(line: str) -> None:
    print(f"{PREFIX} PREFETCH {line}", file=sys.stderr, flush=True)


def _record(rec: dict) -> None:
    ITEMS.append(rec)
    del ITEMS[:-64]


class Proxy:
    """``engine.pipeline``'s stand-in: ``proxy(pipeline_input)`` returns what ``pipeline(pipeline_input)`` returns — computed by the
    helper one item ahead when it can (the head; the tail transforms — from the first one that draws a generator the forward also draws —
    run here, from this process's own RNG state), by the wrapped pipeline (the stock statement) when it cannot (named)."""

    def __init__(self, real, car: Optional[_sc.Sidecar]):
        self.real = real
        self.car = car
        self.order: Optional[list] = None
        self.seq = 0
        self.p0 = None                                      # the RNG triple item 0 was featurized from (the split check replays the tail from it)
        self.out0 = None                                    # item 0's pipeline output, kept until the split check ran
        self.last_post = None                               # the RNG triple this process continued from after the previous item's featurization
        self.fwd = (False, False, False)                    # generators the forward was SEEN to advance (previous post-state vs this call's state)
        self.split_state = "unknown"                        # unknown | on:<i>/<n>:<name> | off:<reason>
        _PROXIES.append(self)

    def __getattr__(self, name):                            # anything else the engine reads off its pipeline (repr, transforms) is the real object's
        return getattr(self.real, name)

    def set_order(self, items: list) -> None:
        self.order = list(items)
        self.seq = 0
        if self.car is not None and self.car.alive() and not STATE["off_reason"] and len(self.order) > 1:
            try:
                self.car.send(("order", self.order))
            except _sc.SidecarError as e:
                STATE["off_reason"] = f"helper_gone:{e}"

    def _parent(self, pipeline_input, reason: str, seq: int, ex):
        CENSUS[f"served:parent:{reason}"] += 1
        t0 = time.perf_counter()
        out = self.real(pipeline_input)
        dt = time.perf_counter() - t0
        _record({"seq": seq, "item": ex, "source": f"parent:{reason}", "featurize_s": round(dt, 3), "wait_s": round(dt, 3)})
        _say(f"item={ex} seq={seq}/{len(self.order or [])} source=parent:{reason} featurize_s={dt:.3f}")
        return out

    def __call__(self, pipeline_input, *args, **kwargs):
        if args or kwargs:                                  # a call with Compose's own keywords (rng_state_dict / _stop_before): the real object's
            return self.real(pipeline_input, *args, **kwargs)
        seq = self.seq
        self.seq += 1
        ex = pipeline_input.get("example_id") if isinstance(pipeline_input, dict) else None
        CENSUS["items"] += 1
        n = len(self.order or [])
        if self.car is None or not self.car.alive() or STATE["off_reason"]:
            return self._parent(pipeline_input, (STATE["off_reason"] or "no_helper").split(":")[0], seq, ex)
        if self.order is None or seq >= n or getattr(self.order[seq], "example_id", None) != ex:
            return self._parent(pipeline_input, "order", seq, ex)
        if n <= 1:
            return self._parent(pipeline_input, "n_items=1", seq, ex)
        pstate = rng_get()
        if STATE["first_input"] is None:                    # the process's first served item: stock statement here AND the helper, compared
            return self._first(pipeline_input, pstate, seq, ex, n)
        if self.last_post is not None:                      # which generators did the forward (everything since the last featurization) advance?
            seen = rng_used(self.last_post, pstate)
            self.fwd = tuple(bool(a or b) for a, b in zip(self.fwd, seen))
        t0 = time.perf_counter()
        try:
            self.car.send(("featurize", seq, pipeline_input, pstate, self.fwd))
            reply, nbytes = self.car.recv(timeout=FIRST_TIMEOUT_S)
        except _sc.SidecarError as e:
            STATE["off_reason"] = f"helper_gone:{e}"
            _say(f"helper gone: {e}")
            return self._parent(pipeline_input, "helper_gone", seq, ex)
        wait = time.perf_counter() - t0
        kind, rseq, payload, post, secs, speculated, respec, usedw, extra = reply
        if kind not in ("ok", "head") or rseq != seq:
            CENSUS["helper_errors"] += 1
            _say(f"item={ex} seq={seq}/{n} helper error (the stock statement runs here):\n{payload if kind == 'error' else 'seq mismatch %r' % rseq}")
            rng_set(pstate)
            return self._parent(pipeline_input, "helper_error", seq, ex)
        check_word = ""
        if extra.get("check") is not None and self.out0 is not None:
            check_word = " " + self._split_check(extra["check"])
        rng_set(post)                                        # continue from the featurizer's post-state, as stock's in-process featurization leaves it
        tail_s = 0.0
        if kind == "head":                                   # the tail transforms here, from this process's own state of the generators the forward drew
            q0 = time.perf_counter()
            try:
                out = run_tail(self.real, payload, int(extra["split"]))
            except Exception:                                # noqa: BLE001 — the tail raised here: the whole statement here from the captured state, named
                CENSUS["helper_errors"] += 1
                _say(f"item={ex} seq={seq}/{n} tail error (the stock statement runs here):\n{traceback.format_exc()}")
                rng_set(pstate)
                return self._parent(pipeline_input, "tail_error", seq, ex)
            tail_s = time.perf_counter() - q0
            CENSUS["served:helper:head"] += 1
        else:
            out = payload
        self.last_post = rng_get()
        CENSUS["served:helper"] += 1
        CENSUS["speculated"] += int(bool(speculated))
        CENSUS["respec"] += int(bool(respec))
        CENSUS["rng_used:" + usedw] += 1
        CENSUS["shadow:" + str(extra.get("shadow")).split("@")[0]] += 1
        STATE["shadow"] = extra.get("shadow")
        rec = {"seq": seq, "item": ex, "source": "helper" + (":head" if kind == "head" else ""), "speculated": int(bool(speculated)), "respec": int(bool(respec)),
               "wait_s": round(wait, 3), "featurize_s": round(float(secs), 3), "tail_s": round(tail_s, 3), "recv_mib": round(nbytes / 2 ** 20, 2),
               "rng_used": usedw, "fwd_gens": used_word(self.fwd), "rng": f"{rng_digest3(pstate)}->{rng_digest3(self.last_post)}"}
        _record(rec)
        _say(f"item={ex} seq={seq}/{n} source={rec['source']} speculated={rec['speculated']} respec={rec['respec']} wait_s={wait:.3f} "
             f"featurize_s={float(secs):.3f} tail_s={tail_s:.3f} recv_mib={rec['recv_mib']} feats_mib={payload_mib(out):.1f} rng_used={usedw} "
             f"fwd_gens={rec['fwd_gens']} split={self.split_state} shadow={extra.get('shadow')}{check_word} rng={rec['rng']}")
        return out

    def _split_check(self, check) -> str:
        """Item 0 once more, the split way: the helper's head output for item 0 + the tail HERE from the state stock had at that point,
        compared leaf by leaf with item 0's in-process output. DIFFERENT → the split is switched off by name (whole-pipeline speculation)."""
        data_mid, post_head, used_head, split = check
        cur = rng_get()
        try:
            rng_set(rng_merge(self.p0, post_head, used_head))
            q0 = time.perf_counter()
            out = run_tail(self.real, data_mid, int(split))
            dt = time.perf_counter() - q0
            where = compare(self.out0, out, "output")
        except Exception as e:                              # noqa: BLE001
            where, dt = f"tail_raised:{type(e).__name__}", 0.0
        finally:
            rng_set(cur)
        self.out0 = None
        ts = transforms_of(self.real) or []
        name = type(ts[int(split)]).__name__ if split is not None and int(split) < len(ts) else "?"
        if where is None:
            self.split_state = f"on:{split}/{len(ts)}:{name}"
            STATE["split"] = self.split_state
            CENSUS["split_check:same"] += 1
            return f"split_check=same check_tail_s={dt:.3f}"
        self.split_state = f"off:check_different"
        STATE["split"] = f"off:DIFFERENT:{where}"
        CENSUS["split_check:different"] += 1
        try:
            self.car.send(("nosplit",))
        except _sc.SidecarError:
            pass
        return f"split_check=DIFFERENT:{where} (split off: whole-pipeline speculation from here)"

    def _first(self, pipeline_input, pstate, seq, ex, n):
        t0 = time.perf_counter()
        sent = True
        try:
            self.car.send(("featurize", seq, pipeline_input, pstate, None))   # the helper featurizes item seq from the same state, concurrently
        except _sc.SidecarError as e:
            sent = False
            STATE["off_reason"] = f"helper_gone:{e}"
        out = self.real(pipeline_input)                       # the stock statement, in this process: this is the output used
        mine = time.perf_counter() - t0
        post_parent = rng_get()
        if not sent:
            STATE["first_input"] = "not_compared"
            return self._parent_done(out, seq, ex, n, mine, "helper_gone")
        try:
            reply, nbytes = self.car.recv(timeout=FIRST_TIMEOUT_S)
        except _sc.SidecarError as e:
            STATE["off_reason"] = f"helper_gone:{e}"
            STATE["first_input"] = "not_compared"
            _say(f"helper gone during the first item: {e}")
            return self._parent_done(out, seq, ex, n, mine, "helper_gone")
        total = time.perf_counter() - t0
        kind, rseq, hout, post, secs, speculated, respec, usedw, extra = reply
        if kind != "ok":
            STATE["off_reason"] = "first_input:helper_error"
            STATE["first_input"] = "helper_error"
            CENSUS["helper_errors"] += 1
            _say(f"first_input=HELPER_ERROR (the lever steps aside for this process; every item featurized here):\n{hout}")
            return self._parent_done(out, seq, ex, n, mine, "first_input")
        where = compare(out, hout, "output")
        if where is None and not rng_eq(post, post_parent):
            where = f"rng_post({rng_digest3(post_parent)}!={rng_digest3(post)})"
        STATE["first_input"] = "same" if where is None else "DIFFERENT"
        STATE["first_input_path"] = where
        CENSUS["first_input:same" if where is None else "first_input:different"] += 1
        trace = extra.get("trace") or []
        split = extra.get("split")
        top = sorted(trace, key=lambda r: -r[2])[:6]
        STATE["trace_top"] = [(i, nm, s, used_word(u)) for i, nm, s, u in top]
        STATE["trace_rng"] = [(i, nm, used_word(u)) for i, nm, s, u in trace if any(u)]
        _say(f"first_input={'same' if where is None else 'DIFFERENT:' + where} item={ex} parent_s={mine:.3f} helper_s={float(secs):.3f} "
             f"total_s={total:.3f} recv_mib={nbytes / 2 ** 20:.2f} feats_mib={payload_mib(out):.1f} rng_used={usedw} rng={rng_digest3(pstate)}->{rng_digest3(post_parent)}")
        if trace:
            _say(f"pipeline transforms={extra.get('n_transforms')} split_before={split}"
                 f"{(':' + trace[split][1]) if split is not None and split < len(trace) else ''} "
                 f"top=" + ",".join(f"{i}:{nm}:{s:.2f}s:{used_word(u)}" for i, nm, s, u in top)
                 + " rng_transforms=" + (",".join(f"{i}:{nm}:{used_word(u)}" for i, nm, s, u in trace if any(u)) or "-"))
        if where is not None:
            STATE["off_reason"] = f"first_input:{where}"
            return self._parent_done(out, seq, ex, n, mine, "first_input")
        self.p0, self.out0 = pstate, (out if split is not None else None)
        self.split_state = "pending_check" if split is not None else "off:no_rng_transform" if trace else "off:not_a_compose"
        self.last_post = post_parent
        CENSUS["served:parent:first"] += 1
        _record({"seq": seq, "item": ex, "source": "parent:first", "featurize_s": round(mine, 3), "wait_s": round(total, 3),
                 "helper_s": round(float(secs), 3), "first_input": STATE["first_input"], "split_before": split})
        return out                                            # the helper is already speculating item seq+1

    def _parent_done(self, out, seq, ex, n, mine, reason):
        CENSUS[f"served:parent:{reason}"] += 1
        _record({"seq": seq, "item": ex, "source": f"parent:{reason}", "featurize_s": round(mine, 3), "wait_s": round(mine, 3)})
        _say(f"item={ex} seq={seq}/{n} source=parent:{reason} featurize_s={mine:.3f}")
        return out


def tap_loader(base):
    """``rf3.inference_engines.rf3``'s ``DataLoader`` name → a subclass whose iterator materializes the sampler's order (the dataset
    holds ``InferenceInput`` objects already in memory; the stock iterator — and its one base-seed draw — is created at the same
    statement) and hands it to the proxy before yielding the items unchanged."""
    if getattr(base, "_rosettafold3_opt_prefetch", False):
        return base

    class PrefetchOrderLoader(base):                        # type: ignore[misc, valid-type]
        _rosettafold3_opt_prefetch = True

        def __iter__(self):
            it = super().__iter__()
            items = list(it)
            specs = [x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x for x in items]   # batch_size=1, collate identity: the item itself
            for p in _PROXIES:
                p.set_order(specs)
            return iter(items)

    PrefetchOrderLoader.__name__ = "DataLoader"
    PrefetchOrderLoader.__qualname__ = "DataLoader"
    return PrefetchOrderLoader


def _wrap_construct(orig):
    def _construct_pipeline(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            start_helper(self)
        except Exception as e:                               # noqa: BLE001 — the fold runs stock featurization; the reason is on the record
            STATE.update({"on": False, "off_reason": f"start_failed:{type(e).__name__}", "reason": f"{type(e).__name__}: {e}"})
            _say(f"helper not started ({type(e).__name__}: {e}); every item featurized in the fold process")
        return out
    _construct_pipeline.__wrapped__ = orig
    _construct_pipeline.__name__ = "_construct_pipeline_prefetch"
    return _construct_pipeline


def start_helper(engine) -> Optional[_sc.Sidecar]:
    """Fork the featurizer for this engine (once per engine) and put the proxy in ``engine.pipeline``'s place."""
    real = getattr(engine, "pipeline", None)
    if real is None or isinstance(real, Proxy):
        return None
    if int(STATE.get("n_gpu") or 1) > 1:
        STATE["off_reason"] = "n_gpu>1"
        engine.pipeline = Proxy(real, None)
        _say("not forked: n_gpu>1 (the row-sharded line featurizes in each rank process as stock)")
        return None
    d_hint = (getattr(engine, "transform_overrides", None) or {}).get("diffusion_batch_size") if isinstance(getattr(engine, "transform_overrides", None), dict) else None
    try:
        car = _sc.Sidecar(NAME, _serve, (real, d_hint), nice=int(STATE.get("nice") or _sc.NICE)).start()
    except Exception as e:                                  # noqa: BLE001 — the platform refuses a side process (fork / pipes): every item featurized in-process, named
        STATE["off_reason"] = f"start_refused:{type(e).__name__}"
        engine.pipeline = Proxy(real, None)
        _say(f"not forked: start_refused ({type(e).__name__}: {e}) — every item featurized in-process as stock")
        return None
    STATE.update({"on": True, "helper_pid": car.pid, "forked_before_cuda": car.forked_before_cuda, "threads_at_fork": car.threads_at_fork})
    engine.pipeline = Proxy(real, car)
    _say(f"helper pid={car.pid} forked_before_cuda={car.forked_before_cuda} threads_at_fork={car.threads_at_fork} nice={car.nice}")
    return car


def enable(engine_module=None) -> dict:
    """Install on ``rf3.inference_engines.rf3`` (given or imported): the engine class's ``_construct_pipeline`` and the module's
    ``DataLoader`` name. Idempotent; a reshaped upstream is :class:`PrefetchRefused`."""
    m = engine_module if engine_module is not None else __import__(ENGINE_MODULE, fromlist=["_"])
    cls = getattr(m, ENGINE_CLASS, None)
    base_loader = getattr(m, "DataLoader", None)
    missing = [w for w, ok in ((f"{ENGINE_CLASS}.{CONSTRUCT}", hasattr(cls, CONSTRUCT)), (f"{ENGINE_CLASS}.run", hasattr(cls, "run")),
                               ("DataLoader", base_loader is not None)) if not ok]
    if missing:
        STATE.update({"installed": False, "on": False, "reason": "upstream reshaped: " + ",".join(missing)})
        raise PrefetchRefused(STATE["reason"])
    cur = getattr(cls, CONSTRUCT)
    if getattr(cur, "__name__", "") != "_construct_pipeline_prefetch":
        _ORIG[CONSTRUCT] = cur
        setattr(cls, CONSTRUCT, _wrap_construct(cur))
    if not getattr(base_loader, "_rosettafold3_opt_prefetch", False):
        _ORIG["DataLoader"] = base_loader
        m.DataLoader = tap_loader(base_loader)
    STATE.update({"installed": True, "reason": None})
    return describe()


def disable(engine_module=None) -> None:
    """Restore upstream's names (tests)."""
    m = engine_module if engine_module is not None else sys.modules.get(ENGINE_MODULE)
    if m is None:
        return
    cls = getattr(m, ENGINE_CLASS, None)
    if CONSTRUCT in _ORIG and cls is not None:
        setattr(cls, CONSTRUCT, _ORIG.pop(CONSTRUCT))
    if "DataLoader" in _ORIG:
        m.DataLoader = _ORIG.pop("DataLoader")
    for p in list(_PROXIES):
        if p.car is not None:
            p.car.stop()
    _PROXIES.clear()
    STATE.update({"installed": False, "on": False})


CONFLICT_N_GPU = ("the row-sharded line (n_gpu > 1: rowpair) runs the next item's featurization in every rank process as stock; the lever is a single-GPU "
                  "lever and steps aside by name pending a measured n_gpu=2 run (composition #6)")


def decline(kind: str, why: str) -> dict:
    """Off BY NAME for this process (``conflict:<kind>`` on the LEVER line; the tally block carries the sentence): nothing is forked or installed."""
    STATE.update(on=False, installed=False, conflict=kind, reason=why, off_reason=kind)
    return describe()


def arm(rep: dict, install_watch: Callable) -> dict:
    """Activation-time hook: when the row names ``prefetch``, install on ``rf3.inference_engines.rf3`` as soon as it executes (a module
    imported already is patched at once). Records ``rep["prefetch"]``; a reshaped upstream is a named refusal on the record."""
    if NAME not in (rep.get("levers") or []) or STATE.get("armed"):
        return rep
    STATE["armed"] = True
    STATE["n_gpu"] = int(rep.get("n_gpu") or 1)
    if STATE["n_gpu"] > 1:                                     # the row-sharded line: off by name in every rank, like confhoist / confln (rowpair owns the item loop there)
        rep[NAME] = decline("n_gpu", CONFLICT_N_GPU)
        return rep

    def on_engine(module):
        try:
            rep[NAME] = enable(module)
        except PrefetchRefused as e:
            rep[NAME] = {"on": False, "installed": False, "reason": str(e), "census": None}
            return
        rep["levers_applied"] = list(dict.fromkeys(list(rep.get("levers_applied") or []) + [NAME]))

    if ENGINE_MODULE in sys.modules:
        on_engine(sys.modules[ENGINE_MODULE])
    else:
        install_watch(ENGINE_MODULE, on_engine, rep)
    return rep


def census() -> dict:
    c = dict(CENSUS)
    parent = {k[len("served:parent:"):]: v for k, v in c.items() if k.startswith("served:parent:")}
    return {"items": c.get("items", 0), "helper": c.get("served:helper", 0), "parent": parent, "speculated": c.get("speculated", 0),
            "respec": c.get("respec", 0), "helper_errors": c.get("helper_errors", 0), "first_input": STATE.get("first_input"),
            "first_input_path": STATE.get("first_input_path"), "by_key": c}


def describe() -> dict:
    return {"on": bool(STATE["on"]), "installed": bool(STATE["installed"]), "reason": STATE["reason"], "off_reason": STATE["off_reason"], "conflict": STATE["conflict"],
            "helper_pid": STATE["helper_pid"], "forked_before_cuda": STATE["forked_before_cuda"], "threads_at_fork": STATE["threads_at_fork"],
            "nice": STATE["nice"], "rule": "next item featurized by one pre-CUDA-forked helper from the parent's RNG state; item 0 compared; parent by name otherwise",
            "census": census(), "items": list(ITEMS)}


def lever_tokens(desc: Optional[dict] = None) -> List[Tuple[str, Any]]:
    """(key, value) tokens for the LEVER line: helper=<items served by the helper> parent=<items featurized in-process>
    speculated / respec counts, first_input=<same|DIFFERENT|…>, forked_before_cuda."""
    d = desc or describe()
    c = d.get("census") or census()
    out: List[Tuple[str, Any]] = [("helper", c["helper"]), ("parent", sum(c["parent"].values())), ("speculated", c["speculated"]), ("respec", c["respec"])]
    if c["parent"]:
        out.append(("parent_by", "|".join(f"{k}:{v}" for k, v in sorted(c["parent"].items()))))
    if c.get("first_input"):
        out.append(("first_input", c["first_input"]))
    if d.get("forked_before_cuda") is not None:
        out.append(("forked_before_cuda", d["forked_before_cuda"]))
    return out
