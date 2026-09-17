"""DEVICE_RESIDENT — the host↔device transfer lever (strategy F6.device_resident: device-resident parameters and
host outputs fetched once, written for this model's recycle loop).

What stock does (alphafold-colabfold 2.3.13, ``alphafold/model/model.py:164-192``): the recycle loop runs in Python; every iteration calls
``self.apply(self.params, key, {**feat, "prev": prev})`` with host NumPy parameters, features and recycling inputs — jax uploads every
leaf to the device at every call — then ``_jnp_to_np`` (``:165-172``) pulls every output leaf back to the host and casts it to float16 with
NumPy (``np.asarray(v, np.float16)``), the recycling inputs of the next iteration included, which are uploaded again.

What the lever does, without editing stock: ``enable()`` rebinds ``alphafold.model.model.RunModel`` to the subclass below before colabfold
builds its runners (``colabfold/alphafold/models.py:169``: ``model.RunModel(...)`` is looked up at the call), and the subclass wraps the
instance's jitted ``apply``:
  1. host NumPy leaves of ``params`` and of the batch dict are uploaded once and found again by identity at the next call (the parameter
     tree once per model swap, ``colabfold/batch.py:426``; the features once per input; the recycling inputs never — see 3); under `big`
     ``--n_gpu P > 1`` the parameters are placed replicated on the mesh and the batch's host leaves enter through the program's input
     shardings at each call (the recipe's selection, ``opt_core.mem.rowpair_jax.alphafold.jit_sharded``; the fetch in 2 assembles the
     row-sharded outputs on the host — the writer boundary of the N² outputs);
  2. the output tree is cast to float16 ON THE DEVICE (one jitted convert per output structure); per recycle only its 0-d leaves — what
     the host loop reads: ``ranking_confidence`` / ``tol`` for the stop rules (``model.py:203-206``) and the means colabfold's per-recycle
     callback prints (``colabfold/batch.py:469-476``: ``mean_plddt``, ``ptm``, ``iptm``, ``tol``) — are fetched, in ONE ``jax.device_get``;
     every other leaf stays on the device behind a host placeholder of its shape and dtype inside a deferred result dict (``_Deferred``:
     ``items`` / ``pop`` / assignment — all stock's ``_jnp_to_np`` and ``result.pop('prev')`` do — never fetch; ``result[key]`` fetches that
     entry once, so a callback that READS arrays (``--save-recycles``) gets the call's true values); ``RunModel.predict`` (overridden to
     wrap stock's own loop, which runs unchanged) fetches what is still on the device ONCE after the last recycle and returns plain dicts of
     host float16 arrays — stock's ``np.asarray(v, np.float16)`` of the same values, byte for byte, inside the same timed call;
  3. the float16 device arrays behind the ``prev`` leaves are kept one step, so the next call's ``prev`` resolves to them (no upload), and the
     device leaves a finished recycle deferred are released when the next call starts (the device holds one output tree, as stock's route).
  Fetch policy (``fetch=`` on the LEVER line): ``last`` (the above) unless the command line asks colabfold for per-recycle artefacts
  (``--save-recycles`` / ``--save-all``: a PDB / pickle written INSIDE the loop from the whole tree), then ``each`` — every call fetched whole,
  so those files stay stock's bytes; a caller of ``colabfold.batch.run(save_recycles=True)`` without that command line
  still receives true values (the deferred dict fetches on access) and only a per-recycle pickle of it names a plain-dict reduce.
Numerics: placement only. The casts are IEEE round-to-nearest-even on both routes (NumPy's float32→float16, XLA's convert) and the jitted
network receives arguments of the same shapes and dtypes, hence the same executable: the outputs are expected byte-identical to stock's
(``exact`` = this lever alone). Memory: the device holds what stock holds during a call (one
parameter tree, one feature set) plus the float16 ``prev`` of the last step.

Evidence: ``_STATE`` (below) — ``calls`` (wrapped apply calls), ``uploads`` (host leaves uploaded), ``hits`` (host leaves found on the
device by identity), ``fetched_bytes`` (float16 output bytes fetched, every route), ``fetch`` (``last`` | ``each``: the policy of the last
predict), ``deferred`` (leaves left on the device at a call, summed), ``deferred_fetches`` (batched fetches of deferred leaves: one per predict when nothing reads an array inside the loop); the EXIT line prints it at
interpreter exit (report.LEVER_EXIT_FMT) and the manifest records it (``lever_states_exit.DEVICE_RESIDENT``). Marker: ``RunModel._device_resident``.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Tuple

NAME = "DEVICE_RESIDENT"
TARGET_MODULE = "alphafold.model.model"
TARGET_CLASS = "RunModel"
MARKER = "_device_resident"

_STATE: Dict[str, Any] = {"enabled": False, "calls": 0, "uploads": 0, "hits": 0, "fetched_bytes": 0, "fetch": "last", "deferred": 0, "deferred_fetches": 0}
FETCH_LAST, FETCH_EACH = "last", "each"
EACH_FLAGS = ("--save-recycles", "--save-all")   # colabfold_batch flags that write per-recycle artefacts from the whole result tree inside the loop (batch.py:478-490)


def fetch_policy(argv=None) -> str:
    """``each`` when the process's command line asks colabfold for per-recycle artefacts (EACH_FLAGS), else ``last``."""
    args = list(sys.argv[1:] if argv is None else argv)
    return FETCH_EACH if any(a in EACH_FLAGS for a in args) else FETCH_LAST


class _Deferred(dict):
    """A result (sub)tree of one apply call whose array leaves are still on the device. Plain-dict behaviour for everything stock's loop does
    (``items`` / ``values`` / ``pop`` of a sub-dict / assignment: the placeholders — host float16 arrays of the true shape, never read); ``d[key]``
    (and ``get``, ``pop`` of an array entry, pickling) fetches that entry's leaves once and stores the host values in place. ``materialize(tree)``
    turns a whole tree into plain dicts of host arrays with one ``device_get``."""

    __slots__ = ("_dev", "_jax", "_twin")

    def __init__(self, *args, jax=None, twin=None, **kwargs):
        dict.__init__(self, *args, **kwargs)     # dict's own construction forms stay open: dm-tree rebuilds a mapping as type(instance)(pairs)
        self._dev: Dict[Any, Any] = {}           # (alphafold's `tree.map_structure(lambda x: x.shape, result)` log line) and must get a filled dict back
        self._jax = jax
        self._twin = twin                        # the resident table of the recycling inputs (the ``prev`` subtree only): a fetched leaf keeps its device twin

    def _fetch(self, key):
        """A read of a deferred entry: everything still deferred in this dict and below it comes over in ONE ``device_get`` (the first array
        read after the loop — alphafold's own ``tree.map_structure(lambda x: x.shape, result)`` log line — is the whole remaining tree at once;
        a read of ``prev`` brings ``prev`` only), then the entry's host value."""
        _fetch_pending([self])
        return dict.__getitem__(self, key)

    def __getitem__(self, key):
        if key in self._dev:
            return self._fetch(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        return self[key] if dict.__contains__(self, key) else default

    def pop(self, key, *default):
        if key in self._dev and not isinstance(dict.get(self, key), dict):   # an array entry leaves with its values; a sub-dict leaves deferred (stock's result.pop('prev'))
            self._fetch(key)
        self._dev.pop(key, None)
        return dict.pop(self, key, *default)

    def __setitem__(self, key, value):
        if key in self._dev and value is not dict.get(self, key):        # stock's x[k] = np.asarray(v, np.float16) hands the placeholder back: still deferred
            self._dev.pop(key, None)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        self._dev.pop(key, None)
        dict.__delitem__(self, key)

    def release(self) -> None:
        """Drop the device leaves still deferred here and below (the placeholders stay, unread): called when the next recycle starts."""
        self._dev.clear()
        for v in dict.values(self):
            if isinstance(v, _Deferred):
                v.release()

    def __reduce_ex__(self, protocol):
        return (dict, (materialize(self),))


def _host_f16(jax, tree):
    import numpy as np
    return jax.tree_util.tree_map(lambda v: np.asarray(v, np.float16), tree)


def _nbytes(jax, tree) -> int:
    return sum(int(getattr(v, "nbytes", 0)) for v in jax.tree_util.tree_leaves(tree))


def _fetch_pending(roots) -> int:
    """Everything still deferred in these dicts and below them, fetched in ONE ``device_get`` and stored in place (a ``prev`` leaf keeps its
    device twin in the resident table). Returns the number of leaves fetched."""
    pending, slots = [], []
    for r in roots:
        _collect_pending(r, pending, slots)
    if not pending:
        return 0
    jax = next(d._jax for d, _ in slots)
    host = _host_f16(jax, jax.device_get(pending))
    _STATE["fetched_bytes"] += _nbytes(jax, host); _STATE["deferred_fetches"] += 1
    for (d, k), h, dev in zip(slots, host, pending):
        if d._twin is not None and _is_host_array(h):
            d._twin[id(h)] = (h, dev)
        dict.__setitem__(d, k, h)
    return len(pending)


def _collect_pending(d, pending, slots) -> None:
    if isinstance(d, _Deferred):
        for k in list(d._dev):
            pending.append(d._dev.pop(k)); slots.append((d, k))
    for k in list(dict.keys(d)):
        v = dict.__getitem__(d, k)
        if isinstance(v, dict):
            _collect_pending(v, pending, slots)


def _plain(x):
    """A dict tree rebuilt as plain dicts (dict-level reads: nothing deferred is fetched); leaves as they are."""
    if isinstance(x, dict):
        return {k: _plain(dict.__getitem__(x, k)) for k in dict.keys(x)}
    return x


def materialize(tree):
    """Plain dicts of host float16 arrays from a (possibly deferred) result tree: what is still on the device fetched in ONE ``device_get``."""
    if not isinstance(tree, dict):
        return tree
    _fetch_pending([tree])
    return _plain(tree)
_MESH: Dict[str, Any] = {"rmesh": None}      # big --n_gpu P>1 (big.apply → set_mesh): the jitted apply is bound to the mesh by the recipe
                                              # (opt_core.mem.rowpair_jax.alphafold.jit_sharded: the pair-shaped leaves of the batch and of the outputs —
                                              # the recycled pair, the distogram / aligned-error logits, PAE — row-sharded, everything else replicated; the
                                              # recipe is the one producer of that selection). Parameters are uploaded once, replicated (shard.put); the
                                              # batch's host leaves enter through the program's input shardings at each call; the returned ``prev`` device
                                              # arrays (row-sharded pair, replicated rest) are kept one step and fed back as they are.


def set_mesh(rmesh) -> None:
    """The RowMesh the runners built AFTER this call bind their apply to (None: single-device placement, the P=1 bytes)."""
    _MESH["rmesh"] = rmesh


def _device_put(jax, v, replicated: bool):
    """One host leaf onto the device(s). P = 1: ``jax.device_put``. P > 1: a leaf known replicated (a parameter) is placed replicated on the mesh
    (``shard.put``) and cached; any other host leaf (the batch) is handed to the program as it is — the program's input shardings, which the
    recipe produced, place it (a row-sharded leaf is never first uploaded whole and then re-laid)."""
    rmesh = _MESH["rmesh"]
    if rmesh is None:
        return jax.device_put(v)
    if not replicated:
        return v
    from opt_core.mem.rowpair_jax import shard as _shard  # noqa: PLC0415 — P>1 only
    return _shard.put(v, rmesh)


def _is_host_array(v) -> bool:
    import numpy as np
    return isinstance(v, np.ndarray)


class _Resident:
    """Per-runner bookkeeping: host leaf id -> (the host array itself, its device copy). Holding the host array pins its id; the table is
    rebuilt at every call from that call's leaves, so it never outlives the arrays stock itself holds."""

    def __init__(self):
        self.table: Dict[int, Tuple[Any, Any]] = {}
        self.param_ids: set = set()              # the entries placed as the model's parameters (replicated=True): the only ones kept across predicts

    def purge(self) -> None:
        """Drop every cached device copy that is not a parameter (the previous predict's features and recycling-input twins): called when a
        predict has handed its outputs on, so the device holds between predicts what stock's route holds — nothing of the finished input."""
        self.table = {k: v for k, v in self.table.items() if k in self.param_ids}

    def to_device(self, tree, jax, keep: Dict[int, Tuple[Any, Any]], replicated: bool = True):
        """``tree`` with every host NumPy leaf replaced by a device array: the cached one when this very array was uploaded (or fetched)
        before, else a fresh placement (:func:`_device_put`; ``replicated``: the tree is known replicated on a mesh — the parameters).
        ``keep`` collects the entries this call used; a leaf the program places itself (P > 1, the batch) is counted as an upload and not
        cached (it has no device twin here)."""
        def leaf(v):
            if not _is_host_array(v):
                return v
            hit = self.table.get(id(v))
            if hit is not None and hit[0] is v:
                keep[id(v)] = hit
                _STATE["hits"] += 1
                if replicated:
                    self.param_ids.add(id(v))
                return hit[1]
            d = _device_put(jax, v, replicated)
            _STATE["uploads"] += 1
            if d is not v:
                keep[id(v)] = (v, d)
                if replicated:
                    self.param_ids.add(id(v))
            return d
        return jax.tree_util.tree_map(leaf, tree)


def _cast_tree_f16(jax, jnp):
    """The float16 convert of an output tree (dicts of arrays), ONE LEAF AT A TIME on the device, each float32 source leaf released the
    moment its float16 copy exists (``jax.Array.delete``): the device then holds at most one leaf twice — a whole-tree convert allocated
    every float16 leaf while every float32 leaf was still alive (the float32 tree + its float16 copy at once), which at large token counts
    asked the allocator for one more region than stock's route ever does (stock converts on the host). Same convert op per leaf → the same
    float16 values; the tree structure is rebuilt as plain dicts / the container types ``tree_map`` gives."""
    f16 = jnp.float16

    def cast_leaf(v):
        w = v.astype(f16)
        if w is not v and hasattr(v, "delete"):
            try:
                w.block_until_ready()                    # the copy exists before its source goes
                v.delete()
            except Exception:                            # noqa: BLE001 — a leaf that cannot be released early (aliased / already deleted) is left to scope exit
                pass
        return w

    def cast(tree):
        return jax.tree_util.tree_map(cast_leaf, tree)
    return cast


def _build_deferred(jax, node, resident, keep, scal_dev, scal_slot):
    """One level of ``_defer_tree`` (a plain module-level recursion: no self-referencing closure — such a cycle would keep a finished recycle's
    device leaves alive until Python's cyclic collector ran, which raises the device peak)."""
    import numpy as np
    if not isinstance(node, dict):
        return node
    d = _Deferred(jax=jax, twin=keep if resident else None)
    for k, v in node.items():
        if isinstance(v, dict):
            dict.__setitem__(d, k, _build_deferred(jax, v, resident or k == "prev", keep, scal_dev, scal_slot))
        elif getattr(v, "ndim", None) == 0:
            scal_dev.append(v); scal_slot.append((d, k)); dict.__setitem__(d, k, None)
        else:
            ph = np.empty(tuple(v.shape), np.float16)
            if resident:                                                  # the recycling inputs: the placeholder handed back next call resolves to this device array
                keep[id(ph)] = (ph, v)
            dict.__setitem__(d, k, ph); d._dev[k] = v
            _STATE["deferred"] += 1
    return d


def _defer_tree(jax, out16, keep):
    """The host view of one call's float16 device tree: 0-d leaves fetched now (ONE ``device_get`` for all of them), every other leaf a placeholder
    (``np.empty`` of its shape, float16 — untouched memory, never read) whose device twin is (a) registered in ``keep`` by identity, so a ``prev``
    leaf handed back as the next call's input resolves to the device array, and (b) deferred in the enclosing ``_Deferred`` under its key."""
    scal_dev, scal_slot = [], []
    host = _build_deferred(jax, out16, False, keep, scal_dev, scal_slot)
    if scal_dev:
        vals = _host_f16(jax, jax.device_get(scal_dev))
        _STATE["fetched_bytes"] += _nbytes(jax, vals)
        for (d, k), hv in zip(scal_slot, vals):
            dict.__setitem__(d, k, hv)
    return host


def _wrap_apply(apply, jax, jnp, runner=None):
    """The instance's jitted ``apply(params, key, batch)`` with resident inputs and device-side float16 outputs (module docstring 1-3)."""
    import numpy as np
    res = _Resident()
    cast = _cast_tree_f16(jax, jnp)
    last: Dict[str, Any] = {"tree": None}

    def is_concrete(v) -> bool:
        return isinstance(v, jax.Array) and not isinstance(v, jax.core.Tracer)

    def wrapped(params, key, batch):
        if last["tree"] is not None:                                      # the previous recycle's deferred leaves: unreachable for stock from here on
            last["tree"].release(); last["tree"] = None
        batch = _plain(batch)                                             # a deferred ``prev`` handed back enters the program as the plain dict stock hands (same executable), its placeholders resolved by ``res``
        keep: Dict[int, Tuple[Any, Any]] = {}
        params_d = res.to_device(params, jax, keep, replicated=True)
        batch_d = res.to_device(batch, jax, keep, replicated=False)
        out = apply(params_d, key, batch_d)
        _STATE["calls"] += 1
        leaves = jax.tree_util.tree_leaves(out)
        if not leaves or not all(is_concrete(v) for v in leaves):       # traced (eval_shape) or empty: nothing to place
            res.table = keep
            return out
        out16 = cast(out)                                                 # float16 on the device: the values stock's np.asarray(v, float16) yields
        policy = getattr(runner, "_fetch_policy", None) or _STATE["fetch"]
        if policy == FETCH_EACH or not isinstance(out16, dict):          # per-recycle artefacts asked for: the whole tree now, plain dicts
            host = _host_f16(jax, jax.device_get(out16))                  # 0-d leaves come back as NumPy scalars: arrays, as stock's asarray gives
            _STATE["fetched_bytes"] += _nbytes(jax, host)
            prev_h = host.get("prev") if isinstance(host, dict) else None   # the next call's recycling inputs: their device twins kept one step
            prev_d = out16.get("prev") if isinstance(out16, dict) else None
            if isinstance(prev_h, dict) and isinstance(prev_d, dict):
                for k, hv in prev_h.items():
                    dv = prev_d.get(k)
                    if _is_host_array(hv) and dv is not None:
                        keep[id(hv)] = (hv, dv)
            res.table = keep
            return host
        host = _defer_tree(jax, out16, keep)                              # scalars now, arrays when read or at the end of predict
        res.table = keep
        last["tree"] = host
        return host

    def purge() -> None:
        """After a predict: release the last recycle's deferred device leaves and every non-parameter device copy (see _Resident.purge)."""
        if last["tree"] is not None:
            last["tree"].release(); last["tree"] = None
        res.purge()

    wrapped._device_resident_inner = apply      # type: ignore[attr-defined]
    wrapped._device_resident_purge = purge      # type: ignore[attr-defined]
    return wrapped


def _subclass(base):
    class RunModel(base):                        # noqa: D401 — the stock name, so repr / logging read the same
        __doc__ = base.__doc__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            import jax                           # the model's own dependency, imported where the model is built (never at enable())
            import jax.numpy as jnp
            apply = self.apply
            if _MESH["rmesh"] is not None:                                  # P>1: the program runs on the mesh with the recipe's leaf shardings (pair-shaped leaves by rows in and out)
                from opt_core.mem.rowpair_jax import alphafold as _recipe  # noqa: PLC0415
                apply = _recipe.jit_sharded(apply, _MESH["rmesh"], n_args=3)
            self.apply = _wrap_apply(apply, jax, jnp, runner=self)
            self._fetch_policy = FETCH_LAST

        def predict(self, *args, **kwargs):
            """Stock's recycle loop, unchanged (``super().predict``), under the fetch policy of this command line; what the loop left on the
            device is fetched once here and the result handed on as plain dicts of host float16 arrays (stock's own types and values)."""
            self._fetch_policy = _STATE["fetch"] = fetch_policy()
            result = super().predict(*args, **kwargs)
            try:
                if isinstance(result, tuple):                               # (result, recycles) — alphafold-colabfold's signature
                    return (materialize(result[0]),) + tuple(result[1:])
                return materialize(result)
            finally:                                                        # the outputs are host dicts now: nothing of this input stays on the device (stock's route holds nothing either)
                getattr(self.apply, "_device_resident_purge", lambda: None)()

    setattr(RunModel, MARKER, True)
    RunModel.__qualname__ = base.__qualname__
    RunModel.__module__ = base.__module__
    return RunModel


def enable(module=None) -> bool:
    """Rebind ``alphafold.model.model.RunModel`` to the resident subclass (idempotent). ``module``: the target module (tests pass a stub);
    None imports ``alphafold.model.model``. Returns True when the class is (already) rebound."""
    import importlib
    m = module if module is not None else importlib.import_module(TARGET_MODULE)
    base = getattr(m, TARGET_CLASS)
    if not getattr(base, MARKER, False):
        setattr(m, TARGET_CLASS, _subclass(base))
    _STATE["enabled"] = True
    return True


def reset_for_tests() -> None:
    _STATE.update(enabled=False, calls=0, uploads=0, hits=0, fetched_bytes=0, fetch=FETCH_LAST, deferred=0, deferred_fetches=0)
    _MESH["rmesh"] = None


def marker_present(module=None) -> Optional[bool]:
    """The marker on the class the target module currently exposes; None when the module is not imported."""
    m = module if module is not None else sys.modules.get(TARGET_MODULE)
    if m is None:
        return None
    return bool(getattr(getattr(m, TARGET_CLASS, None), MARKER, False))
