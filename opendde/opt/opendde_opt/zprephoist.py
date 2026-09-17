"""zprep_hoist — the tree's own lever on the denoiser's per-step pair preparation (exact class: the same tensor, made once).

Stock OpenDDE 1.1.1 `DiffusionModule.f_forward` (opendde/model/modules/diffusion.py:1511-1512, the efficient-fusion path every kit line runs)
prepares the token transformer's pair input on EVERY denoiser call of the 200-step roll-out:

    z = self.normalize(z_pair.to(dtype=torch.float32))
    z = permute_final_dims(z, [2, 0, 1]).contiguous()        # (N, N, c) -> (c, N, N): one full fp32 pair copy per step

`z_pair` is the diffusion pair conditioning — a function of the trunk outputs and the features only, i.e. STEP-INVARIANT inside one
`sample_diffusion()` call. With the DiT conditioning hoist engaged (`dit_hoist`, every kit line below its size gate) `self.normalize` already
answers from its recorded buffer from the second step on (the SAME tensor object every step), and the token transformer is handed the hoist's
recorded copy of the permuted tensor — but the stock statement in between still performs the permute + `.contiguous()` copy on every step and
its result is discarded: 2.0-2.3 s per item at 800 tokens, ~5 s at 1200 (10 ms per copy).

The lever binds the module-global name `permute_final_dims` of `opendde.model.modules.diffusion` (its only use in that module is this
statement) to a memo of ONE entry per source tensor: the first call on a tensor computes the stock permutation AND its contiguous copy (the
copy stock makes on the next line — `.contiguous()` on an already-contiguous tensor is the identity, so nothing is done twice) and records it
keyed by the source's identity (object, storage pointer, version counter, shape, dtype, strides); a later call on the same, unmodified source
returns the recorded copy — no kernel at all. Bitwise by construction: the recorded copy IS the tensor stock computed from the same values
(the hoist's normalize buffer is only ever rewritten by `copy_`, which bumps its version counter, so a new item / sampler call re-records).
A source that is a fresh tensor every step (the conditioning hoist not engaged: `no_dit_hoist` at and above its size gate, or a line without
it) simply records every call — exactly stock's one copy per step (`records` = `calls`, `hits` 0); the lever then saves nothing and its row says so.

Discipline. The memo holds ONE entry (`MAX_ENTRIES`), dropped BEFORE a new source's copy is made — peak memory is stock's transient copy made
persistent for the sampler call (one (c,N,N) fp32 pair tensor: 0.28 GiB at 768 tokens), never two; the entry references its source weakly (an id
reused by a new tensor cannot alias: pointer + version + strides are part of the key and the weak reference must resolve to the very object).
Under `stepgraph` the capture step is a memo hit (recorded on the eager head step), so the graph holds NO copy kernel and reads the recorded
tensor by address as a constant — the entry outlives the graph (a sampler call's graph is dropped at the call's end; the entry is replaced
only when the hoist re-records for the next call, before that call's capture). Calls with autograd enabled, non-CUDA tensors, or another
permutation than [2, 0, 1] are stock (`orig`), counted as `passthrough`.

Binding: `opt_core.autoload.patch_attr_at_import` on the diffusion module (now, or at its import); idempotent; `state=on` when bound on the
imported module and the site was reached (`calls`), inert by name when the module is never imported in the process. LEVER row evidence:
`calls= hits= records= passthrough= entries=`.
"""
from __future__ import annotations

import json
import threading
import weakref

LEVER = "zprep_hoist"
TAG = "opendde-opt"
MODULE, ATTR = "opendde.model.modules.diffusion", "permute_final_dims"       # the name f_forward resolves (imported there from opendde.model.utils)
PERM = (2, 0, 1)
MAX_ENTRIES = 1                                                               # ONE recorded copy alive at any time: a new source drops the old copy BEFORE its own is made

_STATS0 = {"installed": False, "patches": 0, "calls": 0, "hits": 0, "records": 0, "passthrough": 0, "entries": 0,
           "failures": 0, "failure": None}
STATS = json.loads(json.dumps(_STATS0))
_PATCH = {"perm": None}
_LOCK = threading.Lock()
_MEMO: "dict[int, _Entry]" = {}


class ActivationError(RuntimeError):
    pass


class _Entry:
    __slots__ = ("ref", "key", "out", "__weakref__")

    def __init__(self, src, key, out):
        self.ref = weakref.ref(src)
        self.key = key
        self.out = out


def _key(t):
    return (t.data_ptr(), int(t._version), tuple(t.shape), t.dtype, tuple(t.stride()), str(t.device))


def _evict_dead() -> None:
    for k in [k for k, e in _MEMO.items() if e.ref() is None]:
        _MEMO.pop(k, None)


def lookup(src):
    """The recorded contiguous permutation of ``src`` (same object, same storage, same version) or None."""
    ent = _MEMO.get(id(src))
    if ent is None:
        return None
    if ent.ref() is not src or ent.key != _key(src):
        _MEMO.pop(id(src), None)                                             # the id names another (or a rewritten) tensor now: forget it
        return None
    return ent.out


def record(src, out) -> None:
    _evict_dead()
    while len(_MEMO) >= MAX_ENTRIES:                                          # oldest first (dict order)
        _MEMO.pop(next(iter(_MEMO)))
    _MEMO[id(src)] = _Entry(src, _key(src), out)
    STATS["entries"] = len(_MEMO)


def clear() -> None:
    _MEMO.clear(); STATS["entries"] = 0


def make_perm_wrapper(orig):
    """`permute_final_dims(tensor, inds)` served from the memo for inds == [2, 0, 1] on CUDA tensors without autograd; stock otherwise."""
    import torch

    def permute_final_dims(tensor, inds):
        try:
            eligible = (tuple(int(i) for i in inds) == PERM and isinstance(tensor, torch.Tensor) and tensor.is_cuda
                        and tensor.dim() >= 3 and not (torch.is_grad_enabled() and tensor.requires_grad))
        except Exception:  # noqa: BLE001
            eligible = False
        if not eligible:
            STATS["passthrough"] += 1
            return orig(tensor, inds)
        with _LOCK:
            STATS["calls"] += 1
            hit = lookup(tensor)
            if hit is not None:
                STATS["hits"] += 1
                return hit                                                    # already contiguous: the caller's .contiguous() is the identity
            clear()                                                           # a new (or rewritten) source: the previous copy is dropped first, so the memo never holds two
            out = orig(tensor, inds).contiguous()                             # stock's permutation + the copy its next statement makes
            record(tensor, out)                                               # (a fresh source every call — the conditioning hoist not engaged — records every call: stock's cost, hits stay 0)
            STATS["records"] += 1
            return out

    permute_final_dims.__wrapped__ = orig
    return permute_final_dims


# ---------------------------------------------------------------------------------------------------------------- binding + accounting
def _sync() -> None:
    p = _PATCH["perm"]
    STATS["patches"] = 1 if (p is not None and getattr(p, "state", None) == "installed") else 0
    STATS["installed"] = p is not None and STATS["patches"] == 1
    STATS["entries"] = len(_MEMO)


def install() -> None:
    """Patch the diffusion module's `permute_final_dims` name (now, or at the module's import). Idempotent."""
    from opt_core import autoload
    if _PATCH["perm"] is None:
        try:
            _PATCH["perm"] = autoload.patch_attr_at_import(MODULE, ATTR, make_perm_wrapper, tag=TAG, name=LEVER)
        except autoload.PatchError as e:
            raise ActivationError(str(e)) from None
    _sync()


def armed() -> bool:
    return _PATCH["perm"] is not None


def _reset() -> None:
    """Test support: counters and memo back to import time (the patch stays: one AttrPatch per site per process)."""
    clear()
    STATS.clear(); STATS.update(json.loads(json.dumps(_STATS0)))


def served() -> int:
    return int(STATS["calls"])


def fallbacks(planned) -> list:
    if LEVER not in planned or not STATS["failure"]:
        return []
    return [f"{LEVER}:{STATS['failure']}"]


def kit_stats() -> dict:
    _sync()
    return json.loads(json.dumps(STATS, default=str))
