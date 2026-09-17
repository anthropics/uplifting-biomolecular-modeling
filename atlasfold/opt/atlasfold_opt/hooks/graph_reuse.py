"""Lever graph_reuse (class fast; requires denoiser_graph) — the denoiser's captured CUDA graph(s) KEPT ACROSS ITEMS instead of re-captured
by every item of the same shape.

denoiser_graph captures DiffusionModule.forward once per DiffusionHead.sample roll-out (= once per item and seed) and drops the graph when the
roll-out returns: a run over many records of one length bucket pays the capture (torch.cuda.graph: synchronize + gc.collect + empty_cache +
stream capture + instantiate + a fresh private pool — a visible share of a short item's forward) at every item for the identical
kernels.  This lever keeps the graph(s) of the LAST item alive — graph, private pool, static r_noisy / single_cond / output
buffers AND every tensor the captured kernels read by reference — and answers the next roll-out of the same structure by REPLAY:

  * structure = the key denoiser_graph already computes minus the addresses: for every tensor reachable from the call's ``batch`` dict
    (nested containers, relpos_lazy's LazyFeat producer closures) and ``pair_bias``: (size, stride, dtype, device, storage group, offset);
    r_noisy / single_cond by layout; the autocast state; the grad mode (``args_sig``).  Before the trunk of an item runs, a kept set whose
    (B, L) differs from the item's — or an item above denoiser_graph's token gate — is dropped (``AtlasFold(_Multimer).inference`` is
    wrapped for that one comparison: the trunk of a differently-shaped item never runs beside a kept pool, exactly as without this lever).
  * by-reference state.  The captured kernels read, by address: the batch tensors the statement indexes, pair_bias (PairConditioning's
    output; under sampler_hoist the HoistedPairBias stand-in's tensor with the DiT mask bias folded in place at the first call), and every
    tensor sampler_hoist's roll-out state holds (atom_c / atom_pair / atom_win / atom_cond / atom_bias / scond leaves and the memo the
    atom_kdedup / atom_rows leaves join) — all created at the eager warm-up call and, without this lever, freed when sample() returns.  The
    kept set HOLDS the capturing item's copies of all of them (``RefTable``: one ordered walk over ``batch``, ``pair_bias`` and
    sampler_hoist's open ``Roll``, taken right after the capture; nn.Modules are leaves — parameters are process-resident —, scalars under the Roll count by type only).
    A later item of the same structure runs the eager warm-up call denoiser_graph runs before every capture (its levers' memos are built
    there, for THIS item), walks the same roots, requires the identical structural signature, copies every storage of its walk into the
    kept storage at the same position (``UntypedStorage.copy_``, one per distinct storage: views, slices and overlapping unfold views ride
    on their storage; aliasing is part of the signature), copies r_noisy / single_cond into the static inputs and REPLAYS.  The first replay
    of an adopted graph is compared with the warm-up call's own output (``torch.equal``, once per adoption — the two computations the
    capture protocol runs anyway): unequal -> the kept set is dropped and the item captures its own graph (counted ``probe``, gate refused:
    a tensor the walk does not reach would show here, loudly, never as a silently reused address).
  * one shape alive.  Any capture first drops a kept set of another key (graph reset -> its pool freeable, then torch.cuda.graph's own
    empty_cache frees it before the new pool exists): at most ONE item's graphs (1 per distinct sample-chunk shape, as in denoiser_graph)
    and ONE private pool per graph are alive at any time; a mixed-length run (512 -> 896 -> 512 -> ...) behaves item for item as without
    the lever.  Memory: between two items of one structure the kept set stays allocated through the second item's trunk — the pool
    (reserved, as the dropped-but-not-yet-freed pool is without the lever) plus the held by-reference tensors (allocated: pair_bias
    [12,B,L,L,16] + the sampler_hoist leaves + the batch dict, ``kept_gib`` on the LEVER line).
  * RNG, noise, the EDM update, single conditioning stay outside the captured call exactly as in denoiser_graph; nothing captured is
    re-stated.  Both stock runners hand ``inference`` a fresh device feature dict per model_run (runner.py L310-312, runner_multimer.py
    L338-340), so a kept batch tensor of a finished item is never read again by its producer.

Words (graph requests = denoiser_graph key misses while a roll-out is at or below the gate): served ``L<tokens>xN<chunk>`` = adopted (no capture);
fallbacks ``first`` (nothing kept for this module: the run's first item, or the first after a drop), ``key`` (a set of another structure was kept:
dropped, captured), ``disabled`` (AFO_GRAPH_REUSE=0: installed, inert — denoiser_graph as without this lever) are expected; ``table`` (the
post-warm-up walk differs from the kept one), ``probe`` (first replay != warm-up output), ``warmup:<Exc>`` are not (gate refused; the item still
finishes on its own capture).  Facts: ``kept`` (adoptions), ``drops`` / ``drops_by`` (sets dropped, by cause: key / prekey / module / table / probe / trip),
``live`` / ``max_live`` (kept sets now / ever at once: 0|1), ``kept_gib`` / ``kept_gib_max`` (bytes of the kept set's distinct storages),
``tensors`` / ``storages`` (the kept walk), ``adopt_s`` (host seconds spent adopting: warm-up + walk + copies + probe), ``prekey_drops``.
MODEL_OPT_LEVERS_OFF=graph_reuse removes the lever (denoiser_graph unchanged); MODEL_OPT_LEVERS_OFF=denoiser_graph removes both (requires)."""
from __future__ import annotations

import functools
import os
import time
import types
import weakref
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

from opt_core.counters import Ledger

from . import Installed, rebind, size_gated
from . import denoiser_graph as DG
from .relpos import LazyFeat
from ..registry import LEVERS

NAME = "LOCAL.atlasfold.graph_reuse"
IMPL = "keeper"
ENV_SWITCH = "AFO_GRAPH_REUSE"                                      # =0: installed, inert (every graph request `disabled`)
EXPECTED = tuple(LEVERS["graph_reuse"]["expected"])                 # ("first", "key", "disabled")
MODEL_SITES = (("atlasfold.model.model", "AtlasFold"), ("atlasfold.model.model_multimer", "AtlasFold_Multimer"))
FIRST, KEY, DISABLED, TABLE, PROBE, WARMUP = "first", "key", "disabled", "table", "probe", "warmup"
MAX_DEPTH = 12
_SCALARS = (bool, int, float, str, bytes, torch.dtype, torch.device, type(None))
_OPAQUE = (torch.nn.Module, Ledger, torch.cuda.Stream if hasattr(torch.cuda, "Stream") else type(None), weakref.ReferenceType)


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV_SWITCH, "1")).strip().lower() not in ("0", "off", "false", "no")


# ----------------------------------------------------------------------------------------------------------------- the by-reference walk
class RefTable:
    """Every tensor reachable from the walk roots, in walk order, with the structural signature of the walk (no address, no id) and the
    distinct storages behind those tensors (first-seen order; ``nbytes`` = their sum = what holding the table keeps allocated)."""
    __slots__ = ("tensors", "storages", "sig", "nbytes", "unordered")

    def __init__(self):
        self.tensors: List[torch.Tensor] = []
        self.storages: List[torch.Tensor] = []                   # one representative tensor per distinct storage (keeps it addressable)
        self.sig: List[tuple] = []
        self.nbytes = 0
        self.unordered = False                                      # a set holding tensors was met: no deterministic order -> never adopted

    def key(self) -> tuple:
        return tuple(self.sig)

    def copy_from(self, other: "RefTable") -> int:
        """Copy every storage of ``other`` into the storage at the same position here (same signature required); returns the count copied
        (storages shared by both tables — process-resident tensors — are skipped)."""
        n = 0
        for dst, src in zip(self.storages, other.storages):
            ds, ss = dst.untyped_storage(), src.untyped_storage()
            if ds.data_ptr() == ss.data_ptr():
                continue
            ds.copy_(ss)
            n += 1
        return n


def _key_sig(k) -> tuple:
    if isinstance(k, str):
        return ("s", k)
    if isinstance(k, bool):
        return ("b", k)
    if isinstance(k, int):
        return ("i",)                                               # ids / indices as dict keys: structure, not value
    if isinstance(k, tuple):
        return ("u",) + tuple(_key_sig(x) for x in k)
    return ("o", type(k).__name__)


def _cells(fn) -> list:
    held = []
    for c in tuple(getattr(fn, "__closure__", None) or ()):
        try:
            held.append(c.cell_contents)
        except ValueError:
            continue
    held += list(getattr(fn, "args", None) or ()) + list((getattr(fn, "keywords", None) or {}).values())
    return held


def _walk(obj, t: RefTable, seen: Dict[int, int], groups: Dict[int, int], depth: int, valued: bool) -> None:
    sig = t.sig
    if isinstance(obj, torch.Tensor):
        oid = id(obj)
        if oid in seen:
            sig.append(("R", seen[oid])); return
        seen[oid] = len(seen)
        st = obj.untyped_storage()
        sp = st.data_ptr()
        g = groups.get(sp)
        if g is None:
            g = groups[sp] = len(groups)
            t.storages.append(obj)
            t.nbytes += int(st.nbytes())
            sig.append(("N", g, int(st.nbytes())))
        sig.append(("T", tuple(obj.size()), tuple(obj.stride()), obj.dtype, str(obj.device), g, int(obj.storage_offset())))
        t.tensors.append(obj)
        return
    if isinstance(obj, _SCALARS):
        sig.append(("V", repr(obj)) if valued else ("V", type(obj).__name__)); return
    if depth > MAX_DEPTH or isinstance(obj, _OPAQUE) or isinstance(obj, type):
        sig.append(("X", type(obj).__name__)); return
    oid = id(obj)
    if oid in seen:
        sig.append(("R", seen[oid])); return
    seen[oid] = len(seen)
    if isinstance(obj, dict):
        sig.append(("D", len(obj)))
        for k, v in list(obj.items()):                              # insertion order (memo keys carry ids: their order is the call order, their value is not structure)
            sig.append(("K",) + _key_sig(k))
            _walk(v, t, seen, groups, depth + 1, valued)
        return
    if isinstance(obj, (list, tuple)):
        sig.append(("L" if isinstance(obj, list) else "U", len(obj)))
        for v in list(obj):
            _walk(v, t, seen, groups, depth + 1, valued)
        return
    if isinstance(obj, (set, frozenset)):
        if any(isinstance(v, torch.Tensor) for v in obj):
            t.unordered = True
        sig.append(("S", len(obj))); return
    if isinstance(obj, LazyFeat):
        sig.append(("F", repr(getattr(obj, "ac", None))))
        for v in _cells(obj.fn):
            _walk(v, t, seen, groups, depth + 1, valued)
        return
    if isinstance(obj, (types.FunctionType, functools.partial)):
        sig.append(("C",))
        for v in _cells(obj):
            _walk(v, t, seen, groups, depth + 1, valued)
        return
    if isinstance(obj, (types.MethodType, types.BuiltinFunctionType)) or (callable(obj) and not hasattr(obj, "__dict__") and not hasattr(type(obj), "__slots__")):
        sig.append(("X", type(obj).__name__)); return
    names = []
    for kls in type(obj).__mro__:
        names += [n for n in getattr(kls, "__slots__", ()) if isinstance(n, str)]
    if hasattr(obj, "__dict__"):
        names += list(vars(obj).keys())
    names = sorted(set(n for n in names if n not in ("__dict__", "__weakref__")))
    sig.append(("O", type(obj).__name__, tuple(names)))
    for n in names:
        try:
            v = getattr(obj, n)
        except AttributeError:                                      # an unset slot
            sig.append(("-", n)); continue
        _walk(v, t, seen, groups, depth + 1, False)                   # inside an object (stand-in, namespace, Roll): scalars are counters / flags -> by type; tensors by layout


def walk_refs(roots: List[Tuple[Any, bool]]) -> RefTable:
    """RefTable of ``roots`` = [(object, valued), ...] walked in order with ONE seen-set (an object reachable from two roots is recorded once)."""
    t = RefTable(); seen: Dict[int, int] = {}; groups: Dict[int, int] = {}
    for obj, valued in roots:
        t.sig.append(("ROOT",))
        _walk(obj, t, seen, groups, 0, valued)
    return t


def keeper_roots() -> List[Tuple[Any, bool]]:
    """The roll-out state of the levers whose tensors the captured denoiser reads by reference besides its arguments: sampler_hoist's open
    Roll of this thread (its held leaves and memo; scalars there are counters -> by type).  Empty when that lever is absent or idle."""
    roots: List[Tuple[Any, bool]] = []
    try:
        from . import sampler_hoist as SH
        ro = SH._cur()
    except Exception:  # noqa: BLE001
        ro = None
    if ro is not None:
        roots.append((ro, False))
    return roots


def _flat_sig(obj: Any, _depth: int = 0) -> tuple:
    """denoiser_graph.flatten_key without addresses or ids: tensors by (size, stride, dtype, device); dicts (sorted by key), lists, tuples and
    LazyFeat producers by structure; scalars by value; anything else by type NAME only (a stand-in's internals evolve during a roll-out and
    are not part of a call's structure — the post-warm-up RefTable records what they hold)."""
    if isinstance(obj, torch.Tensor):
        return DG._tensor_sig(obj, False)
    if _depth > 8:
        return ("O", type(obj).__name__)
    if isinstance(obj, dict):
        keys = sorted(obj.keys(), key=repr)
        return ("D", tuple((repr(k), _flat_sig(obj[k], _depth + 1)) for k in keys))
    if isinstance(obj, (list, tuple)):
        return ("L" if isinstance(obj, list) else "U", tuple(_flat_sig(v, _depth + 1) for v in obj))
    if isinstance(obj, LazyFeat):
        inner = tuple(_flat_sig(v, _depth + 1) for v in _cells(obj.fn) if isinstance(v, (torch.Tensor, dict, list, tuple, LazyFeat)))
        return ("F", repr(getattr(obj, "ac", None)), inner)
    if isinstance(obj, _SCALARS):
        return ("V", repr(obj))
    return ("O", type(obj).__name__)


def args_sig(batch, r_noisy, single_cond, pair_bias) -> tuple:
    """The structure of one call's arguments = denoiser_graph's call key with every address / id left out: the lookup key of a kept entry
    (equal for two items of one bucket, batch size, sample chunk, dtype and autocast state; computed before any warm-up)."""
    ac = DG.autocast_state(r_noisy.device.type if r_noisy.device.type in ("cuda", "cpu") else "cuda")
    return (_flat_sig(batch), DG._tensor_sig(r_noisy, False), DG._tensor_sig(single_cond, False), _flat_sig(pair_bias),
            ("A",) + tuple(repr(x) for x in ac), ("G", torch.is_grad_enabled(), torch.is_inference_mode_enabled()))


def full_table(batch, pair_bias) -> RefTable:
    """The by-reference table of a call AFTER its warm-up: arguments first (valued), then the keeper roots."""
    return walk_refs([(batch, True), (pair_bias, True)] + keeper_roots())


# ----------------------------------------------------------------------------------------------------------------- the kept set + keeper
class KeptSet:
    """The graphs ONE item captured, kept for the items after it: args_sig -> denoiser_graph entry (each with its RefTable); ``prekey`` =
    (B, L) an item is compared on before its trunk; ``building`` = the roll-out still capturing into it (None once that roll-out closed);
    ``module`` = a weak reference to the DiffusionModule the graphs belong to (another instance — a second model in one process — never
    adopts them: its parameters live at other addresses)."""

    def __init__(self, module, prekey, building):
        self.entries: Dict[tuple, Any] = {}
        self.prekey = prekey
        self.building = building
        self.hits = 0
        try:
            self.module = weakref.ref(module)
        except TypeError:                                           # (a test double without weakref support)
            self.module = lambda m=module: m

    def of(self, module) -> bool:
        return self.module() is module

    def nbytes(self) -> int:
        ptrs: Dict[int, int] = {}
        for e in self.entries.values():
            if e.table is not None:
                for s in e.table.storages:
                    st = s.untyped_storage(); ptrs[st.data_ptr()] = int(st.nbytes())
        return sum(ptrs.values())

    def release(self) -> None:
        for e in list(self.entries.values()):
            e.release()
        self.entries.clear()
        self.building = None


class Keeper:
    """Process state of the lever: THE kept set (one slot: one item's graphs alive in the process, whatever module they belong to), the
    ledger, the switch, denoiser_graph's token gate."""

    def __init__(self, ledger: Ledger, on: bool, limit: int):
        self.ledger, self.on, self.limit = ledger, bool(on), int(limit)
        self.cur: Optional[KeptSet] = None
        self.max_live = 0
        self.kept_gib_max = 0.0
        self.adopt_s = 0.0
        self.drops_by: Dict[str, int] = {}

    # -- facts
    def _sync_facts(self) -> None:
        live = 1 if self.cur is not None else 0
        self.max_live = max(self.max_live, live)
        gib = (self.cur.nbytes() if self.cur is not None else 0) / float(2 ** 30)
        self.kept_gib_max = max(self.kept_gib_max, gib)
        led = self.ledger
        led.set("live", live); led.set("max_live", self.max_live)
        led.set("kept_gib", f"{gib:.3f}"); led.set("kept_gib_max", f"{self.kept_gib_max:.3f}")
        led.set("adopt_s", f"{self.adopt_s:.3f}")
        led.set("drops_by", ",".join(f"{k}:{v}" for k, v in self.drops_by.items()) or "none")
        ent = next((e for e in (self.cur.entries.values() if self.cur is not None else ()) if e.table is not None), None)
        led.set("tensors", len(ent.table.tensors) if ent is not None else 0)
        led.set("storages", len(ent.table.storages) if ent is not None else 0)

    # -- the slot
    @property
    def sets(self) -> Dict[int, KeptSet]:
        """{0: the kept set} or {} (a mapping view for callers that count)."""
        return {0: self.cur} if self.cur is not None else {}

    def live(self, module) -> Optional[KeptSet]:
        """The kept set when it belongs to ``module``, else None (a set of another module stays in the slot until a drop names it)."""
        ks = self.cur
        return ks if ks is not None and ks.of(module) else None

    def new_set(self, module, prekey, building) -> KeptSet:
        if self.cur is not None:
            self.drop("module" if not self.cur.of(module) else "key")
        ks = KeptSet(module, prekey, building)
        self.cur = ks
        try:                                                        # the module dies (its model dropped): its graphs go with it, at once
            weakref.finalize(module, self._gone, ks)
        except TypeError:
            pass
        self._sync_facts()
        return ks

    def _gone(self, ks: KeptSet) -> None:
        if self.cur is ks:
            self.drop("module")

    def drop(self, cause: Optional[str]) -> Optional[KeptSet]:
        ks, self.cur = self.cur, None
        if ks is None:
            return None
        ks.release()
        if cause:
            self.drops_by[cause] = self.drops_by.get(cause, 0) + 1
            self.ledger.count("drops")
            if cause == "prekey":
                self.ledger.count("prekey_drops")
        self._sync_facts()
        return ks

    def drop_all(self, cause: str) -> None:
        self.drop(cause)


KEEPER: Optional[Keeper] = None                                     # set by install(); the RollOut subclass and the inference wrapper read it


# ----------------------------------------------------------------------------------------------------------------- the roll-out that keeps
class KeepingRollOut(DG.RollOut):
    """denoiser_graph's roll-out with the keeper's hook points: adopt a kept graph on a key miss, drop other keys before a capture, record a
    capture into the kept set, never release a kept entry itself."""

    def __init__(self, ledger: Ledger, limit: int, graphs=None, keeper: Optional[Keeper] = None):
        super().__init__(ledger, limit, graphs=graphs)
        self.keeper = keeper if keeper is not None else KEEPER
        self.module = None

    # -- hook points
    def owns(self, ent) -> bool:
        return getattr(ent, "owner", None) is None

    def adopt(self, key, refs, shape, stock_forward, module, batch, r_noisy, single_cond, pair_bias):
        kp = self.keeper
        self.module = module
        if kp is None:
            return None
        led = kp.ledger
        if not kp.on:
            led.fallback(DISABLED); return None
        ks = kp.live(module)
        if ks is None:                                              # nothing kept, or a set of another module (dropped before this roll-out's capture)
            led.fallback(FIRST); return None
        if ks.building is self:                                     # this roll-out's own set (a second chunk shape): a capture, not an adoption
            led.fallback(FIRST); return None
        skey = args_sig(batch, r_noisy, single_cond, pair_bias)
        cand = ks.entries.get(skey) if ks.building is None else None
        if cand is None or cand.graph is None or cand.table is None:
            self._drop_set(kp, "key"); led.fallback(KEY); return None
        t0 = time.perf_counter()
        try:
            out_w = self.backend.warmup(lambda: stock_forward(module, batch, r_noisy, single_cond, pair_bias), r_noisy.device)
        except Exception as e:  # noqa: BLE001 — the eager call raised: nothing adopted; denoiser_graph's own warm-up meets the same error and names it
            self._drop_set(kp, "warmup"); led.fallback(f"{WARMUP}:{type(e).__name__}"); return None
        table = full_table(batch, pair_bias)
        if table.unordered or table.key() != cand.table.key():
            self._drop_set(kp, "table"); led.fallback(TABLE); return None
        with torch.no_grad():
            cand.table.copy_from(table)
            cand.r.copy_(r_noisy); cand.sc.copy_(single_cond)
            cand.graph.replay()
            same = out_w is not None and cand.out.shape == out_w.shape and bool(torch.equal(cand.out, out_w))
        del out_w
        if not same:
            self._drop_set(kp, "probe"); led.fallback(PROBE); return None
        kp.adopt_s += time.perf_counter() - t0
        ks.hits += 1
        cand.refs = refs                                            # the objects THIS roll-out's key pins (its own batch / pair_bias), held for the roll-out as denoiser_graph does
        led.serve("L%dxN%d" % (int(r_noisy.shape[2]), int(r_noisy.shape[1])))
        led.count("kept")
        kp._sync_facts()
        return cand

    def before_capture(self, module, shape) -> None:
        kp = self.keeper
        self.module = module
        self._table = None
        if kp is None:
            return
        ks = kp.cur
        if ks is not None and ks.building is not self:              # one alive: whatever is kept (another key, or another module's) goes BEFORE this capture allocates its pool
            self._drop_set(kp, ("key" if ks.of(module) else "module") if kp.on else None)

    def after_capture(self, module, key, ent, batch, pair_bias) -> None:
        kp = self.keeper
        if kp is None or not kp.on:
            return
        table = full_table(batch, pair_bias)                        # the by-reference state as captured (the roll-out leaves exist since the warm-up call; an adopting item takes its own after ITS warm-up call)
        if table.unordered:                                         # no deterministic walk: this item's graph is not kept (a plain denoiser_graph entry)
            return
        ks = kp.live(module)
        if ks is None or ks.building is not self:
            B, L = int(ent.shape[0][0]), int(ent.shape[0][2])
            ks = kp.new_set(module, (B, L), self)
        ent.table = table
        ent.owner = ks
        ks.entries[args_sig(batch, ent.r, ent.sc, pair_bias)] = ent
        kp._sync_facts()

    # -- lifetime
    def _drop_set(self, kp: Keeper, cause) -> None:
        ks = kp.drop(cause)
        if ks is not None:                                          # entries this roll-out adopted from it are gone with it
            self.graphs = {k: e for k, e in self.graphs.items() if getattr(e, "owner", None) is not ks}

    def trip(self, reason, stock):
        kp = self.keeper
        if kp is not None and kp.cur is not None and (kp.cur.building is self or any(e.owner is kp.cur for e in self.graphs.values())):
            self._drop_set(kp, "trip")
        return super().trip(reason, stock)

    def close(self) -> None:
        kp = self.keeper
        if kp is not None:
            ks = kp.cur
            if ks is not None and ks.building is self:
                if self.dead is not None or not ks.entries:
                    self._drop_set(kp, "trip")
                else:
                    ks.building = None                              # complete: the next roll-out of this structure may adopt it
            for ent in self.graphs.values():                        # a kept entry lets go of THIS roll-out's key objects; its table holds what its kernels read
                if getattr(ent, "owner", None) is not None:
                    ent.refs = None
        super().close()


def make_factory(keeper: Keeper):
    def factory(ledger, limit, graphs=None):
        return KeepingRollOut(ledger, limit, graphs=graphs, keeper=keeper)
    factory.__qualname__ = "RollOut[atlasfold_opt:graph_reuse]"
    return factory


def prekey_of(batch) -> Tuple[Optional[int], Optional[int]]:
    for k in ("aatype_int", "seq_mask", "res_idx"):
        v = batch.get(k) if isinstance(batch, dict) else None
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            return int(v.shape[0]), int(v.shape[1])
    return None, None


def make_inference(inner, keeper: Keeper):
    """``AtlasFold(_Multimer).inference``: before the item's trunk, a kept set of another (B, L) — or any kept set when this item is above
    denoiser_graph's token gate — is dropped (its pool freeable before the trunk allocates, as without this lever); then the stock call."""
    @functools.wraps(inner)
    def inference(self, batch, *a, **k):
        ks = keeper.cur
        if keeper.on and ks is not None and ks.building is None:
            B, L = prekey_of(batch)
            sm = getattr(getattr(self, "diffusion_head", None), "score_model", None)
            if sm is not None and not ks.of(sm):
                keeper.drop("module")                               # kept for another model instance of this process: never this one's
            elif L is None or ks.prekey != (B, L) or L > keeper.limit:
                keeper.drop("prekey")
        return inner(self, batch, *a, **k)
    inference.__qualname__ = f"{getattr(inner, '__qualname__', 'inference').split('[')[0]}[atlasfold_opt:graph_reuse]"
    return inference


def _line(ledger: Ledger, tag: str):
    def line():
        return ledger.line(tag)
    return line


def install(mode: str, tag: str, ctx: dict) -> Installed:
    global KEEPER
    import importlib
    st = (ctx or {}).get("denoiser_graph") or DG.INSTALLED
    if not st or st.get("ledger") is None:
        return Installed("graph_reuse", False, reason="denoiser_graph_absent")
    on = enabled()
    ledger = Ledger(NAME, impl=IMPL, origin="kit", expected=EXPECTED)
    for k in ("kept", "drops", "prekey_drops"):
        ledger.set(k, 0)
    ledger.set("switch", "on" if on else "off")
    keeper = Keeper(ledger, on, int(st.get("limit", DG.max_tokens())))
    keeper._sync_facts()
    KEEPER = keeper
    DG.ROLLOUT_FACTORY = make_factory(keeper)
    sites = []
    for mod_name, cls_name in MODEL_SITES:
        try:
            cls = getattr(importlib.import_module(mod_name), cls_name)
        except Exception:  # noqa: BLE001 — a model class this stock tree lacks (monomer-only / multimer-only installs)
            continue
        inner = cls.__dict__.get("inference") or getattr(cls, "inference", None)
        if inner is None:
            continue
        rebind(cls, "inference", make_inference(inner, keeper), inner)
        sites.append(f"{mod_name}.{cls_name}.inference")
    ledger.set("sites", len(sites))
    return Installed("graph_reuse", True, lines=[_line(ledger, tag)], gates=[size_gated(ledger)],
                     facts={"impl": IMPL, "switch": on, "keeper": keeper, "ledger": ledger, "sites": sites})
