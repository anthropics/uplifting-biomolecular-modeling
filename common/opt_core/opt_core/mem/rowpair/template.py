"""The template embedder under row sharding (per template slot t, ``v_t = linear(LN(z)) + embed(template
pair features of t)``; a small pair stack per slot; ``z += linear(relu(mean_t LN(v_t)))``). The pair track ``z`` stays SHARDED through it:
every ``[T, N, N, *]`` template tensor of the dense statement exists here only as this rank's ROWS ``[T, R, N, *]`` (``R = layout.R``), built
or sliced per row block; nothing ``N x N`` is materialised whole on a rank. Engine sub-modules enter as CALLABLES (the featuriser linears, the
per-slot pair stack, the closing LayerNorm / linear / mean) exactly as the other L2 statements of this package take ``fn`` + tensors + a
:class:`~opt_core.mem.rowpair.dist.Layout`; this module holds the row-block schedule, the slot bookkeeping and the two cross-rank agreements.

Statements (row dim of every pair-shaped tensor is dim ``-3``: ``z_shard[..., R, N, C]`` — a leading batch dim of 1 is carried through):

    slice_template_inputs_to_rows(feats, layout, pair_keys)
                                        the ENTRY seam for an engine whose featuriser emits dense template pair features: each named
                                        ``[..., T, N, N, F]`` key becomes this rank's rows ``[..., T, R, N, F]`` (a view) ONCE, at trunk entry;
                                        per-token keys are left alone (replicated by design). Equality at P == 1 (an L1 seam).
    slot_census(*token_masks)           which template slots are REAL (some per-token mask entry non-zero) and which are DUMMIES (all-zero
                                        masks: their pair features are exactly +0.0 by the featurisers' own statements) -> :class:`SlotCensus`.
    note_all_dummy(census, templated=, refuse=)
                                        the FAIL-LOUD hook of a templated run whose slots are all dummies: a NAMED event (schedule field
                                        ``templ_event=all_slots_dummy`` + the NOTE text returned for the kit to print) or a refusal by name
                                        (``refuse=True``) — the kit decides which; this module never proceeds silently as if templated.
    TemplatePairRows(...)               the ONE row accessor of template pair features, three sources: ``rows=`` (row-sliced dense features),
                                        ``lazy_dummy=census`` (zero rows synthesised per block — refused by name when a slot is real), ``computed=``
                                        (per-key callables ``fn(slots, g0, g1)`` building rows from O(T·N) precursors, e.g. :func:`distogram_rows`).
    distogram_rows / unit_vector_rows / same_chain_rows / same_or_group_rows / pairs_unmasked
                                        per-row-block feature statements from per-token precursors (coordinates, backbone frames, masks, chain
                                        ids; the inter-chain template mask ``(asym_i == asym_j) | (grp_t(i) == grp_t(j) >= 0)``); the per-slot
                                        pair mask rows are :func:`~opt_core.mem.rowpair.trunk.pair_mask_rows` ``(tok_mask[..., slots, :], g0, g1)``.
    real_template_rows(layout, n_slots, pb_coords=, pb_mask=, frames=, frame_mask=, asym_id=, edges=, grp=)
                                        the COMPUTED :class:`TemplatePairRows` of the template pair products (``template_distogram``,
                                        ``template_unit_vector``) from O(T·N) precursors: feature rows x pair-mask rows x chain-mask rows, on the
                                        featurizer's device class (``ROWPAIR_TEMPL_FEAT_DEVICE=cpu``, default) or the pair device (``cuda``);
                                        ``[T, N, N, F]`` never exists. Bit-exact the dense statement sliced.
    slot_census(..., asym_id=) / SlotCensus.line()
                                        per-slot token and chain counts and the consumption line ``[templ] REAL templates: slots k/T real (t0:
                                        tokens m/N chains c/C; …) dummy=[…]``.
    parse_interchain_spec / load_interchain_spec / group_feature / interchain_words / interchain_scope
                                        the inter-chain template group lever (``ROWPAIR_TEMPL_INTERCHAIN=<spec path>``,
                                        ``ROWPAIR_TEMPL_INTERCHAIN_SCOPE=all|features``): a spec (``prep`` groups file or ``slots`` list) →
                                        :class:`InterchainSpec` (consistency rule: a group masked BY NAME unless every member chain carries it
                                        on exactly one, identical row) → the per-slot per-token group feature ``grp[T, N]`` (-1 = none) that
                                        :func:`same_or_group_rows` reads → the census words ``templ_interchain=groups:n,pairs:m,scope:s``.
    template_slot_groups(keys, n_slots, layout)
                                        de-duplication of bit-identical slots (``v_t`` dedup): slots whose key tensors are equal share ONE
                                        pair-stack pass. The verdict is AGREED across ranks (every rank's local verdict AND-ed: gather to rank 0 +
                                        broadcast), so row-sharded keys give the dense verdict and every rank runs the same number of pair stacks.
    template_embed_rows(z_shard, layout, n_templ=, unit_rows_fn=, pair_stack_fn=, finish_fn=, ...)
                                        the driver: per unique slot group ``u = unit_rows_fn(z rows, slot, (g0, g1))`` in row blocks ->
                                        ``pair_stack_fn(u, mask_loc)`` -> per row block the slots are stacked in their ORIGINAL order
                                        (``[..., T, rows, N, c_t]``, so the engine's T-term mean sees the dense operand order) -> ``finish_fn`` ->
                                        ``z rows += result`` in place (or into ``out``). Optional host parking of ``z_shard`` while the pair stacks
                                        run and of finished ``u`` slabs (:class:`ParkedStorage`; levers ``ROWPAIR_TEMPL_PARK_Z`` / ``ROWPAIR_TEMPL_PARK_U``).
    template_embed_dense(z, ...)        the dense statement with the same callables (one device, all rows, no dedup) — the tests' reference.

Replicated BY DESIGN (named here and in the schedule census as ``templ_replicated=token_feats``): every per-token template feature
``[T, N, *]`` (residue types, pseudo-beta / backbone-frame masks, coordinates), the chain ids, and the pair-stack MASK rows the caller passes.

Numerics: ``unit_rows_fn`` / ``finish_fn`` are row-local (per-element arithmetic of the dense statement; launch M = local rows); the pair
stack's classes are those of the installs inside ``pair_stack_fn`` (trimul / triatt / transition of this package); slot de-duplication feeds
the closing mean the same T operands in the same order (identical slots have identical ``u``), so it changes no value. The row block size is
an argument or the ``ROWPAIR_ROWBLK_MB`` MiB target (:func:`templ_block_rows` = :func:`~opt_core.mem.rowpair.shard.choose_block_rows`, THE
chooser; P-invariant by construction), is walked with :func:`~opt_core.mem.rowpair.shard.produce_rows_` (block starts on the layout's chunk grid) and is
recorded (``templ_rows``, ``templ_rows_source``); the slot facts land as ``templ_slots``, ``templ_groups``, ``templ_mode``, the parking outcome as ``templ_park_z|u``.

At P == 1 (no group / a P == 1 layout) ``template_embed_rows`` refuses by name like every L2 statement (the engine's own template embedder runs
at ``--n_gpu 1``); the L1-style helpers (slicing, census, feature rows, the dense reference) are plain statements usable anywhere.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from . import RowpairRefused
from ._torch import torch
from .dist import Layout, allreduce_checksum, broadcast_obj, chunk_align, gather_cat_to_rank0, is_dist, require_sharded, world
from .evidence import record_schedule
from .shard import choose_block_rows, iter_row_blocks, owns_whole_storage, produce_rows_, regrow_storage_, release_storage_, shard_rows, storage_resizable
from .trunk import DevicePeak, host_alloc, load_rows, pair_mask_rows, park_host, same_rows, store_rows

__all__ = ["slice_template_inputs_to_rows", "SlotCensus", "slot_census", "note_all_dummy", "ALL_DUMMY_EVENT", "zero_template_rows",
           "TemplatePairRows", "distogram_rows", "unit_vector_rows", "real_template_rows", "feat_device", "same_chain_rows", "same_or_group_rows",
           "same_or_group_dense", "pairs_unmasked", "InterchainSpec", "parse_interchain_spec", "load_interchain_spec", "interchain_scope",
           "group_feature", "interchain_words", "template_slot_groups", "groups_word", "templ_block_rows", "ParkedStorage", "template_embed_rows",
           "template_embed_dense", "ENV_TEMPL_NODEDUPE", "ENV_TEMPL_PARK_Z", "ENV_TEMPL_PARK_U", "ENV_TEMPL_FEAT_DEVICE", "ENV_TEMPL_INTERCHAIN",
           "ENV_TEMPL_INTERCHAIN_SCOPE", "REAL_PAIR_KEYS", "COMPUTED_TRANSIENT_CHANNELS",
           "ENV_TEMPL_ORDER", "TEMPL_ORDERS", "templ_order", "template_embed_rows_zparked"]

ENV_TEMPL_NODEDUPE = "ROWPAIR_TEMPL_NODEDUPE"  # 1: every slot runs its own pair stack (no de-duplication of identical slots)
ENV_TEMPL_PARK_Z = "ROWPAIR_TEMPL_PARK_Z"      # 1: park z_shard on the host while the per-slot pair stacks run (device storage released)
ENV_TEMPL_ORDER = "ROWPAIR_TEMPL_ORDER"        # classic (default: today's statement order) | zparked (z parked BEFORE the slabs are built; template_embed_rows_zparked)
TEMPL_ORDERS = ("classic", "zparked")
ENV_TEMPL_PARK_U = "ROWPAIR_TEMPL_PARK_U"      # 1: with > 1 slot group, park each FINISHED u slab on the host; stream its rows back at the close
ENV_TEMPL_FEAT_DEVICE = "ROWPAIR_TEMPL_FEAT_DEVICE"            # cpu (default) | cuda: where real_template_rows runs the featurizer statements
ENV_TEMPL_INTERCHAIN = "ROWPAIR_TEMPL_INTERCHAIN"              # path of an inter-chain template group spec (unset = lever off)
ENV_TEMPL_INTERCHAIN_SCOPE = "ROWPAIR_TEMPL_INTERCHAIN_SCOPE"  # all (default) | features: which chain masks the group rule un-masks
ALL_DUMMY_EVENT = "all_slots_dummy"            # the schedule-census word of a templated run whose slots are all dummies
ROW_DIM = -3                                   # the row dim of every pair-shaped tensor here ([..., rows, N, C])
REAL_PAIR_KEYS = ("template_distogram", "template_unit_vector")   # the template pair products real_template_rows computes per row block
COMPUTED_TRANSIENT_CHANNELS = 42 + 2 * 46      # per-row-element transients of the computed features on the pair device, in fp32-element units


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip() == "1"


# ================================================================================================================ entry seam: slicing
def slice_template_inputs_to_rows(feats: Mapping[str, object], layout: Layout, pair_keys: Iterable[str], *, missing: str = "refuse"
                                  ) -> Dict[str, object]:
    """A copy of ``feats`` in which every key of ``pair_keys`` (dense template pair features ``[..., T, N, N, F]``) is replaced by this rank's
    ROWS ``[..., T, R, N, F]`` (:func:`~opt_core.mem.rowpair.shard.shard_rows` on dim -3: a view — the caller drops the dense tensor to free
    it). Other keys are carried unchanged (per-token features are replicated by design). A listed key that is absent is refused by name
    (``missing="skip"`` tolerates it); a listed key whose dims -3 / -2 are not ``N x N`` is refused by name (e.g. lazy placeholders: do not list
    them). Equality at P == 1 / a replicated layout. Records ``templ_pair_keys`` and ``templ_pair_rows``."""
    if missing not in ("refuse", "skip"):
        raise RowpairRefused(f"slice_template_inputs_to_rows: missing={missing!r} (refuse|skip)")
    out = dict(feats)
    done = []
    for k in pair_keys:
        if k not in feats:
            if missing == "refuse":
                raise RowpairRefused(f"slice_template_inputs_to_rows: template pair key {k!r} is not in the features (keys listed as pair-shaped "
                                     f"must be present, or pass missing='skip')")
            continue
        t = feats[k]
        if not hasattr(t, "dim") or t.dim() < 3 or int(t.shape[-3]) != layout.N or int(t.shape[-2]) != layout.N:
            raise RowpairRefused(f"slice_template_inputs_to_rows: {k!r} has shape {tuple(getattr(t, 'shape', ()))}; a template pair feature is "
                                 f"[..., T, N, N, F] with N={layout.N} on dims -3 and -2")
        out[k] = shard_rows(t, layout, dim=ROW_DIM)
        done.append(str(k))
    record_schedule(templ_pair_keys=",".join(done) if done else "none", templ_pair_rows=layout.R)
    return out


# ================================================================================================================ slot census + the event
class SlotCensus(object):
    """Which of the ``n_slots`` template slots are real / dummies. ``real`` / ``dummy``: tuples of slot indices; ``all_dummy``; ``tokens``: per
    slot, the tokens where ANY given mask is non-zero; ``n_tokens`` (N); ``chains``: per slot, the distinct ``asym_id`` values over those tokens
    (empty when no ``asym_id`` was given); ``n_chains`` (distinct non-zero asym ids); ``words()`` -> the schedule fields ``templ_slots templ_real
    templ_dummy`` (+ ``templ_tokens templ_chains templ_n_chains`` when ``asym_id`` was given); ``line(tag)`` -> the consumption line a templated
    run prints."""

    __slots__ = ("n_slots", "real", "dummy", "tokens", "n_tokens", "chains", "n_chains")

    def __init__(self, n_slots: int, real: Sequence[int], tokens: Sequence[int] = (), n_tokens: int = 0, chains: Sequence[int] = (),
                 n_chains: int = 0):
        self.n_slots = int(n_slots)
        self.real = tuple(sorted(int(t) for t in real))
        self.dummy = tuple(t for t in range(self.n_slots) if t not in self.real)
        self.tokens = tuple(int(x) for x in tokens)
        self.n_tokens = int(n_tokens)
        self.chains = tuple(int(x) for x in chains)
        self.n_chains = int(n_chains)

    @property
    def all_dummy(self) -> bool:
        return self.n_slots > 0 and not self.real

    def is_dummy(self, t: int) -> bool:
        return int(t) not in self.real

    def words(self) -> Dict[str, object]:
        w: Dict[str, object] = {"templ_slots": self.n_slots, "templ_real": len(self.real), "templ_dummy": len(self.dummy)}
        if self.chains:
            w["templ_tokens"] = "+".join(str(self.tokens[t]) for t in self.real) or "none"
            w["templ_chains"] = "+".join(str(self.chains[t]) for t in self.real) or "none"
            w["templ_n_chains"] = self.n_chains
        return w

    def line(self, tag: str = "") -> str:
        """``[templ] REAL templates: slots <k>/<T> real (t<i>: tokens <m>/<N> chains <c>/<C>; …) dummy=[…]`` or ``[templ] templates: all <T>
        slots dummy`` — the words a templated run prints on rank 0 at every template call."""
        head = "[templ]" + (f" {tag}" if tag else "")
        if self.all_dummy or self.n_slots == 0:
            return f"{head} templates: all {self.n_slots} slots dummy"
        per = []
        for t in self.real:
            tok = f"tokens {self.tokens[t]}/{self.n_tokens}" if self.tokens else "tokens ?"
            ch = f" chains {self.chains[t]}/{self.n_chains}" if self.chains else ""
            per.append(f"t{t}: {tok}{ch}")
        return f"{head} REAL templates: slots {len(self.real)}/{self.n_slots} real ({'; '.join(per)}) dummy={list(self.dummy)}"

    def __repr__(self) -> str:
        return f"SlotCensus(n_slots={self.n_slots}, real={list(self.real)}, dummy={list(self.dummy)})"


def slot_census(*token_masks, slot_dim: int = -2, asym_id=None) -> SlotCensus:
    """The census from per-token template masks ``[..., T, N]`` (slot dim ``slot_dim``, token dim last): slot t is REAL iff any mask has a
    non-zero entry for t; per slot the token count where any mask is non-zero and, with ``asym_id`` ``[..., N]``, the count of distinct chains
    over those tokens. With no mask given every slot counts as real (nothing proves a dummy). Records ``templ_slots templ_real templ_dummy``
    (+ ``templ_tokens templ_chains templ_n_chains`` with ``asym_id``)."""
    if not token_masks:
        raise RowpairRefused("slot_census: at least one per-token template mask [..., T, N] is required")
    T = int(token_masks[0].shape[slot_dim])
    N = int(token_masks[0].shape[-1])
    hit = None
    for m in token_masks:
        if int(m.shape[slot_dim]) != T:
            raise RowpairRefused(f"slot_census: masks disagree on the slot count ({tuple(m.shape)} vs T={T} on dim {slot_dim})")
        mm = (m.movedim(slot_dim, -2) != 0)                                           # [..., T, N]
        mm = mm.reshape(-1, T, int(mm.shape[-1])).any(dim=0)                          # [T, N] over any leading dims
        if int(mm.shape[-1]) != N:
            raise RowpairRefused(f"slot_census: masks disagree on the token count ({tuple(m.shape)} vs N={N})")
        hit = mm if hit is None else (hit | mm)
    real = [t for t in range(T) if bool(hit[t].any())]
    tokens = [int(hit[t].sum()) for t in range(T)]
    chains, n_chains = (), 0
    if asym_id is not None:
        aid = asym_id.reshape(-1, int(asym_id.shape[-1]))[0].to(hit.device)
        if int(aid.shape[-1]) != N:
            raise RowpairRefused(f"slot_census: asym_id {tuple(asym_id.shape)} vs N={N}")
        chains = tuple(int(torch.unique(aid[hit[t]]).numel()) for t in range(T))
        nz = aid[aid != 0]
        n_chains = int(torch.unique(nz).numel()) if nz.numel() else int(torch.unique(aid).numel())
    c = SlotCensus(T, real, tokens, N, chains, n_chains)
    record_schedule(**c.words())
    return c


def note_all_dummy(census: SlotCensus, *, templated: bool, refuse: bool = False, tag: str = "template embedder",
                   lever: str = "rowpair") -> Optional[str]:
    """The named event of a TEMPLATED run (``templated=True``: the request declared templates) whose slots are ALL dummies: records
    ``templ_event=all_slots_dummy`` and either raises :class:`RowpairRefused` (``refuse=True``) or returns the NOTE text the kit prints.
    Returns None when the run is untemplated or some slot is real (nothing to note)."""
    if not templated or not census.all_dummy:
        return None
    record_schedule(templ_event=ALL_DUMMY_EVENT)
    text = (f"{tag}: templates were requested but all {census.n_slots} template slots featurised as dummies "
            f"(event={ALL_DUMMY_EVENT}; the template embedder adds the dummy-slot term only)")
    if refuse:
        raise RowpairRefused("refused: " + text, lever)
    return "NOTE " + text


# ================================================================================================================ row accessor
def zero_template_rows(lead: Sequence[int], n_slots: int, rows: int, N: int, F: int, *, dtype, device):
    """The lazy-dummy statement: the pair-feature rows of dummy slots are exactly +0.0 -> ``zeros([*lead, S, rows, N, F])``."""
    return torch.zeros(tuple(int(x) for x in lead) + (int(n_slots), int(rows), int(N), int(F)), dtype=dtype, device=device)


class TemplatePairRows(object):
    """Row accessor of the template pair features: ``rows(key, slots, i0, i1) -> [..., len(slots), i1-i0, N, F]`` for LOCAL rows ``i0:i1``
    (global ``r0+i0 : r0+i1``). Exactly one source:

      * ``rows={key: [..., T, R, N, F]}``   row-sliced dense features (:func:`slice_template_inputs_to_rows`) — sliced;
      * ``lazy_dummy=census``               every slot is a dummy: zero rows synthesised per call (:func:`zero_template_rows`); ``feature_dims=
                                            {key: F}``, ``dtype``, ``device`` required. REFUSED BY NAME when the census has a real slot (a real
                                            template's pair rows must be featurised: ``rows=`` or ``computed=``);
      * ``computed={key: fn}``              ``fn(slots, g0, g1) -> [..., len(slots), g1-g0, N, F]`` builds the rows of GLOBAL rows ``g0:g1`` from
                                            per-token precursors (e.g. :func:`distogram_rows` over coordinates, the engine's unit-vector statement).

    ``mode`` in {rows, lazy_dummy, computed}; recorded as ``templ_mode``."""

    def __init__(self, layout: Layout, n_slots: int, *, rows: Optional[Mapping[str, object]] = None, lazy_dummy: Optional[SlotCensus] = None,
                 computed: Optional[Mapping[str, Callable]] = None, feature_dims: Optional[Mapping[str, int]] = None, lead: Sequence[int] = (),
                 dtype=None, device=None):
        given = [name for name, v in (("rows", rows), ("lazy_dummy", lazy_dummy), ("computed", computed)) if v is not None]
        if len(given) != 1:
            raise RowpairRefused(f"TemplatePairRows: exactly one source of rows= / lazy_dummy= / computed= is required (got {given or 'none'})")
        self.layout, self.n_slots, self.mode = layout, int(n_slots), given[0]
        self.N, self.R, self.r0 = layout.N, layout.R, layout.r0
        self.lead = tuple(int(x) for x in lead)
        self._rows, self._fns, self._dims, self.dtype, self.device = None, None, None, dtype, device
        self._transient_channels = None
        self.feat_device = None
        if rows is not None:
            self._rows = dict(rows)
            for k, t in self._rows.items():
                if t.dim() < 4 or int(t.shape[-4]) != self.n_slots or int(t.shape[-3]) != self.R or int(t.shape[-2]) != self.N:
                    raise RowpairRefused(f"TemplatePairRows(rows=): {k!r} has shape {tuple(t.shape)}; this rank's rows are [..., T={self.n_slots}, "
                                         f"R={self.R}, N={self.N}, F] (slice the dense features with slice_template_inputs_to_rows first)")
        elif lazy_dummy is not None:
            if not lazy_dummy.all_dummy:
                raise RowpairRefused(f"TemplatePairRows(lazy_dummy=): slots {list(lazy_dummy.real)} are REAL templates — zero rows may only stand in "
                                     f"for dummy slots; featurise the pair rows of real slots (rows= or computed=)")
            if lazy_dummy.n_slots != self.n_slots or not feature_dims or dtype is None or device is None:
                raise RowpairRefused("TemplatePairRows(lazy_dummy=): feature_dims={key: F}, dtype, device and a census of n_slots slots are required")
            self._dims = {str(k): int(v) for k, v in feature_dims.items()}
        else:
            self._fns = dict(computed)
        record_schedule(templ_mode=self.mode)

    def keys(self) -> List[str]:
        src = self._rows if self._rows is not None else (self._dims if self._dims is not None else self._fns)
        return list(src.keys())

    def rows(self, key: str, slots: Sequence[int], i0: int, i1: int):
        i0, i1 = int(i0), int(i1)
        if not (0 <= i0 <= i1 <= self.R):
            raise RowpairRefused(f"TemplatePairRows.rows: local rows {i0}:{i1} outside 0:{self.R}")
        slots = [int(s) for s in slots]
        if self._rows is not None:
            t = self._rows[key]
            return t[..., slots, i0:i1, :, :]
        if self._dims is not None:
            return zero_template_rows(self.lead, len(slots), i1 - i0, self.N, self._dims[key], dtype=self.dtype, device=self.device)
        y = self._fns[key](slots, self.r0 + i0, self.r0 + i1)
        if int(y.shape[-4]) != len(slots) or int(y.shape[-3]) != i1 - i0 or int(y.shape[-2]) != self.N:
            raise RowpairRefused(f"TemplatePairRows(computed=): {key!r} returned {tuple(y.shape)} for slots {slots} rows {i0}:{i1}; expected "
                                 f"[..., {len(slots)}, {i1 - i0}, {self.N}, F]")
        return y

    @property
    def transient_channels(self) -> int:
        """The per-row-element transient channel count the row-block budget accounts for (:func:`templ_block_rows` ``feat_dims``): the total
        feature width for ``rows`` / ``lazy_dummy``; for ``computed`` what its producer declared (:func:`real_template_rows`:
        :data:`COMPUTED_TRANSIENT_CHANNELS` when the featurizer statements run on the pair device, else the feature width), 64 otherwise."""
        if self._transient_channels is not None:
            return int(self._transient_channels)
        if self._rows is not None:
            return sum(int(t.shape[-1]) for t in self._rows.values())
        if self._dims is not None:
            return sum(self._dims.values())
        return 64

    @transient_channels.setter
    def transient_channels(self, v: int) -> None:
        self._transient_channels = int(v)

    def describe(self) -> str:
        extra = ""
        if self.mode == "computed" and self.feat_device is not None:
            extra = f" feat_device={self.feat_device} transient_channels={self.transient_channels} (per row block: rows x N x {self.transient_channels} x 4 B)"
        return f"[templ] slots={self.n_slots} mode={self.mode} rows={self.r0}:{self.r0 + self.R} of N={self.N}{extra}"


# ================================================================================================================ per-row-block feature statements
def same_chain_rows(asym_id, g0: int, g1: int):
    """bool ``[..., 1, g1-g0, N, 1]``: ``asym_id[i] == asym_id[j]`` for GLOBAL rows ``g0:g1`` (``asym_id`` ``[..., N]``) — the same-chain template
    mask (:func:`~opt_core.mem.rowpair.trunk.same_rows`) shaped to broadcast over the slot dim and the feature dim."""
    return same_rows(asym_id, g0, g1)[..., None, :, :, None]


def same_or_group_rows(asym_id, grp, slots: Sequence[int], g0: int, g1: int):
    """bool ``[..., S, g1-g0, N, 1]``: ``(asym_i == asym_j) | (grp[t, i] == grp[t, j] >= 0)`` for t in ``slots``, GLOBAL rows ``g0:g1`` — the
    inter-chain template mask (``grp`` ``[..., T, N]`` integer chain-group ids per slot, -1 = no group; ``asym_id`` ``[..., N]``)."""
    same = same_rows(asym_id, g0, g1)[..., None, :, :]                                          # [..., 1, r, N]
    g = grp[..., list(slots), :]
    gi = g[..., g0:g1, None]
    gj = g[..., None, :]
    grpm = (gi == gj) & (gi >= 0)                                                                # [..., S, r, N]
    return (same | grpm)[..., None]


def same_or_group_dense(asym_id, grp):
    """bool ``[..., T, N, N, 1]``: :func:`same_or_group_rows` over all rows and slots (the dense reference / small-N path)."""
    return same_or_group_rows(asym_id, grp, list(range(int(grp.shape[-2]))), 0, int(asym_id.shape[-1]))


def pairs_unmasked(asym_id, grp) -> List[int]:
    """Per slot: the number of ORDERED token pairs (i, j) with ``asym_i != asym_j`` that the group rule un-masks — closed form from per-group
    per-chain token counts (``n_g**2 - sum_c n_gc**2``); no ``N x N`` tensor. ``asym_id`` ``[..., N]``, ``grp`` ``[..., T, N]`` (leading dims: the
    first entry is read)."""
    aid = asym_id.reshape(-1, asym_id.shape[-1])[0].long().cpu()
    g = grp.reshape(-1, *grp.shape[-2:])[0].long().cpu()
    out = []
    for t in range(int(g.shape[0])):
        total = 0
        for gid in torch.unique(g[t]).tolist():
            if gid < 0:
                continue
            sel = g[t] == gid
            n_g = int(sel.sum())
            _chains, counts = torch.unique(aid[sel], return_counts=True)
            total += n_g * n_g - int((counts * counts).sum())
        out.append(total)
    return out


# ================================================================================================================ inter-chain template groups
class InterchainSpec(NamedTuple):
    """An inter-chain template group spec, normalised: ``row_group[chain_id][t]`` = the group name template ROW t of that chain belongs to (None =
    none); ``group_ids`` = the group names in id order (the int id of a group = its index; -1 = none, the convention of :func:`same_or_group_rows`);
    ``invalid`` = groups the consistency rule masked (name -> reason); ``members`` = declared member chains per group; ``names`` = slot names
    (``slots`` format)."""
    fmt: str
    source: str
    item: Optional[str]
    variant: Optional[str]
    group_ids: Tuple[str, ...]
    row_group: Dict[str, Tuple[Optional[str], ...]]
    members: Dict[str, Tuple[str, ...]]
    invalid: Dict[str, str]
    names: Dict[str, str]


def _gname(g) -> Optional[str]:
    return None if g in (None, "", "none") else str(g)


def parse_interchain_spec(obj, *, source: str = "<inline>") -> InterchainSpec:
    """Normalise a spec object. Format ``prep`` (a HIER_TEMPL_INPUTS_v1 groups file: ``per_chain_templates`` {chain: [{group: name}|name|None per
    template ROW]}, ``groups`` [{group_id, query_to_template_chain | query_chains_ring_order}]) or format ``slots`` (``{"slots": [{"name",
    "groups": [[chain ids], ...]}, ...]}`` or a bare list of slots). CONSISTENCY RULE (prep): a group is valid iff every member chain carries the
    group on EXACTLY ONE row and that row index is identical across the members; a violating group is MASKED (``invalid[g] = reason``, its rows
    set to None) — named, never silently kept. Malformed input is refused by name (``interchain spec <source>: <what>``), never an empty spec."""
    what = f"interchain spec {source}"
    if isinstance(obj, dict) and "per_chain_templates" in obj:
        fmt = "prep"
        groups = obj.get("groups", []) or []
        if not isinstance(groups, list) or not isinstance(obj.get("per_chain_templates"), dict):
            raise RowpairRefused(f"{what}: 'groups' must be a list and 'per_chain_templates' a mapping chain -> rows")
        gids: List[str] = []
        members: Dict[str, List[str]] = {}
        for g in groups:
            if not isinstance(g, dict) or "group_id" not in g:
                raise RowpairRefused(f"{what}: every entry of 'groups' needs a 'group_id' (got {g!r})")
            gid = str(g["group_id"])
            gids.append(gid)
            mem = [str(c) for c in (g.get("query_to_template_chain") or {}).keys()] or [str(c) for c in (g.get("query_chains_ring_order") or [])]
            members[gid] = mem
        row_group: Dict[str, List[Optional[str]]] = {}
        for c, rows in obj["per_chain_templates"].items():
            if not isinstance(rows, (list, tuple)):
                raise RowpairRefused(f"{what}: per_chain_templates[{c!r}] must be a list of rows (got {type(rows).__name__})")
            lst: List[Optional[str]] = []
            for r in rows:
                g = _gname(r.get("group") if isinstance(r, dict) else r)
                if g is not None and g not in gids:
                    gids.append(g)
                    members.setdefault(g, [])
                lst.append(g)
            row_group[str(c)] = lst
        invalid: Dict[str, str] = {}
        for g in gids:
            mem = members.get(g) or [c for c, lst in row_group.items() if g in lst]
            rows_of = {c: [t for t, x in enumerate(row_group.get(c, [])) if x == g] for c in mem}
            if not mem:
                invalid[g] = "no member chains"
            elif any(len(v) != 1 for v in rows_of.values()):
                invalid[g] = "a member chain carries several rows or none: {" + ", ".join(f"{c}:{v}" for c, v in rows_of.items() if len(v) != 1) + "}"
            elif len({v[0] for v in rows_of.values()}) != 1:
                invalid[g] = "rows differ across chains: {" + ", ".join(f"{c}:{v[0]}" for c, v in rows_of.items()) + "}"
        for g in invalid:
            for c, lst in row_group.items():
                row_group[c] = [None if x == g else x for x in lst]
        return InterchainSpec(fmt, str(source), (str(obj["item"]) if "item" in obj else None), (str(obj["variant"]) if "variant" in obj else None),
                              tuple(gids), {c: tuple(v) for c, v in row_group.items()}, {g: tuple(v) for g, v in members.items()}, invalid, {})
    slots = obj.get("slots") if isinstance(obj, dict) else obj
    if not isinstance(slots, (list, tuple)):
        raise RowpairRefused(f"{what}: neither a 'per_chain_templates' mapping (prep format) nor a 'slots' list (slots format)")
    gids, row_group, names, members = [], {}, {}, {}
    n_slots = len(slots)
    for k, sl in enumerate(slots):
        name = str(sl.get("name", f"slot{k}")) if isinstance(sl, dict) else f"slot{k}"
        names[str(k)] = name
        grs = (sl.get("groups", []) if isinstance(sl, dict) else sl) or []
        if not isinstance(grs, (list, tuple)):
            raise RowpairRefused(f"{what}: slot {k} 'groups' must be a list of chain-id lists")
        for gi, grp in enumerate(grs):
            if not isinstance(grp, (list, tuple)):
                raise RowpairRefused(f"{what}: slot {k} group {gi} must be a list of chain ids (got {grp!r})")
            gid = f"{name}#{gi}"
            gids.append(gid)
            members[gid] = tuple(str(c) for c in grp)
            for c in grp:
                c = str(c)
                lst = row_group.setdefault(c, [None] * n_slots)
                if lst[k] is not None:
                    raise RowpairRefused(f"{what}: chain {c!r} appears in two groups of slot {k} ({name!r})")
                lst[k] = gid
    return InterchainSpec("slots", str(source), (str(obj["item"]) if isinstance(obj, dict) and "item" in obj else None),
                          (str(obj["variant"]) if isinstance(obj, dict) and "variant" in obj else None), tuple(gids),
                          {c: tuple(v) for c, v in row_group.items()}, members, {}, names)


_SPEC_CACHE: Dict[Tuple[str, int], InterchainSpec] = {}


def load_interchain_spec(path: Optional[str] = None) -> Optional[InterchainSpec]:
    """The spec at ``path`` (None: ``ROWPAIR_TEMPL_INTERCHAIN``; unset / blank -> None = the lever is off), parsed once per (path, mtime)."""
    path = (os.environ.get(ENV_TEMPL_INTERCHAIN, "") if path is None else str(path)).strip()
    if not path:
        return None
    try:
        key = (path, os.stat(path).st_mtime_ns)
    except OSError as e:
        raise RowpairRefused(f"interchain spec {path}: unreadable ({e})") from None
    if key not in _SPEC_CACHE:
        import json
        try:
            with open(path) as f:
                obj = json.load(f)
        except (OSError, ValueError) as e:
            raise RowpairRefused(f"interchain spec {path}: unreadable ({type(e).__name__}: {e})") from None
        _SPEC_CACHE.clear()
        _SPEC_CACHE[key] = parse_interchain_spec(obj, source=path)
    return _SPEC_CACHE[key]


def interchain_scope(value: Optional[str] = None) -> str:
    """``ROWPAIR_TEMPL_INTERCHAIN_SCOPE`` (or ``value``): ``all`` (default: the featurizer's AND the model-side chain masks follow the group rule) |
    ``features`` (the featurizer products only); anything else refused by name."""
    v = (os.environ.get(ENV_TEMPL_INTERCHAIN_SCOPE, "") if value is None else str(value)).strip().lower() or "all"
    if v not in ("all", "features"):
        raise RowpairRefused(f"{ENV_TEMPL_INTERCHAIN_SCOPE}={v!r}: all | features")
    return v


def group_feature(spec: Optional[InterchainSpec], chain_ids_token: Sequence[str], n_slots: int, n_tokens: int):
    """``(grp int32 [n_slots, n_tokens], events)``: ``grp[t, i]`` = the id (index into ``spec.group_ids``) of the group template ROW t of token i's
    chain belongs to, -1 = none (rows beyond the chain's list, chains absent from the spec, padding tokens beyond ``len(chain_ids_token)``, masked
    groups). ``events``: ``templ_interchain_masked=<g>:<reason>`` per masked group and ``templ_interchain_absent_chains=<ids>`` for spec chains
    the query lacks — the kit prints them; ``templ_interchain_masked`` is recorded. ``spec`` None -> all -1, no events."""
    import numpy as np
    T, N = int(n_slots), int(n_tokens)
    out = np.full((T, N), -1, dtype=np.int32)
    events: List[str] = []
    ids = [str(c) for c in chain_ids_token]
    if len(ids) > N:
        raise RowpairRefused(f"group_feature: {len(ids)} chain ids for {N} tokens")
    if spec is None:
        return torch.from_numpy(out), events
    gindex = {g: k for k, g in enumerate(spec.group_ids)}
    for g, why in spec.invalid.items():
        events.append(f"templ_interchain_masked={g}:{why}")
    absent = sorted(set(spec.row_group) - set(ids))
    if absent:
        events.append(f"templ_interchain_absent_chains={'+'.join(absent)}")
    if spec.invalid:
        record_schedule(templ_interchain_masked="/".join(f"{g}:{why}" for g, why in spec.invalid.items()))
    if ids:
        uniq, inv = np.unique(np.asarray(ids, dtype=object).astype(str), return_inverse=True)
        table = np.full((T, len(uniq)), -1, dtype=np.int32)
        for u, c in enumerate(uniq.tolist()):
            rows = spec.row_group.get(c, ())
            for t in range(min(T, len(rows))):
                if rows[t] is not None and rows[t] not in spec.invalid:
                    table[t, u] = gindex[rows[t]]
        out[:, :len(ids)] = table[:, inv]
    return torch.from_numpy(out), events


def interchain_words(asym_id, grp, spec: Optional[InterchainSpec] = None, scope: Optional[str] = None) -> str:
    """``groups=<n distinct un-masked groups> pairs=<pairs_unmasked total> per_slot=<p0>+<p1>+… scope=<scope>`` (+ `` INTERCHAIN_UNMASK`` when
    any pair is un-masked); records ``templ_interchain=groups:<n>,pairs:<m>,scope:<s>``."""
    per = pairs_unmasked(asym_id, grp)
    g = grp.reshape(-1, *grp.shape[-2:])[0]
    n_groups = int((torch.unique(g[g >= 0])).numel())
    sc = interchain_scope(scope)
    total = int(sum(per))
    record_schedule(templ_interchain=f"groups:{n_groups},pairs:{total},scope:{sc}")
    return f"groups={n_groups} pairs={total} per_slot={'+'.join(str(x) for x in per) or 'none'} scope={sc}" + (" INTERCHAIN_UNMASK" if total > 0 else "")


# ================================================================================================================ per-row-block feature statements (pair)
def distogram_rows(x, lower, upper, slots: Sequence[int], g0: int, g1: int, *, out_dtype=None):
    """One-hot distogram rows: ``d2[..., s, i, j] = sum_k (x[s, i, k] - x[s, j, k])**2`` summed as ``((d0 + d1) + d2)`` (numpy's pairwise order
    for 3 terms), ``onehot = ((d2 > lower) * (d2 < upper))`` -> ``[..., S, g1-g0, N, n_bins]`` for slots ``slots``, GLOBAL rows ``g0:g1``.
    ``x`` ``[..., T, N, 3]`` coordinates (in the dtype of the engine's featuriser, e.g. float64 for a numpy-side one; NaN = missing atom gives an
    all-zero row, as the dense statement does); ``lower`` / ``upper`` ``[n_bins]`` SQUARED bin edges in x's dtype (the engine's edge statement,
    e.g. ``linspace(min, max, n)**2`` and its shift with ``inf`` appended); the one-hot is returned in ``out_dtype`` (default x's dtype).
    Masks (pseudo-beta pair mask, chain mask) are the caller's products (:func:`~opt_core.mem.rowpair.trunk.pair_mask_rows`, :func:`same_chain_rows`)."""
    xs = x[..., list(slots), :, :]
    diff = xs[..., g0:g1, None, :] - xs[..., None, :, :]                                        # [..., S, r, N, 3]
    sq = diff ** 2
    d2 = ((sq[..., 0] + sq[..., 1]) + sq[..., 2])[..., None]
    onehot = ((d2 > lower) * (d2 < upper)).to(d2.dtype)
    return onehot if out_dtype is None else onehot.to(out_dtype)


def _norm3(x, y, z, eps: float):
    """``Vec3Array.normalized(eps)`` of the engine's geometry module: ``n = sqrt(clamp(x*x + y*y + z*z, min=eps**2))``; ``(x/n, y/n, z/n)``."""
    n2 = x * x + y * y + z * z
    n2 = torch.clamp(n2, min=eps ** 2)
    n = torch.sqrt(n2)
    return x / n, y / n, z / n


def unit_vector_rows(frames, slots: Sequence[int], g0: int, g1: int, *, eps: float = 1e-6, out_dtype=None):
    """The template unit-vector rows ``[..., S, g1-g0, N, 3]`` for slots ``slots``, GLOBAL rows ``g0:g1``: ``frames`` ``[..., T, N, 3, 3]``
    backbone coordinates (atoms N, CA, C on dim -2; xyz on dim -1; FINITE fp32 — the featurizer's ``nan_to_num``, which :func:`real_template_rows`
    applies for its callers). The statement is the engine's own vector geometry,
    component by component in its operation order (bit-exact the dense statement sliced): per token the frame of
    ``make_transform_from_reference(N, CA, C)`` — ``e0 = normalize(C - CA)``, ``c = (N - CA)·e0`` (operand order ``e1·e0``),
    ``e1 = normalize((N - CA) - c e0)``, ``e2 = e0 x e1``, rotation columns ``(e0, e1, e2)``, translation ``CA`` — its inverse
    (``R^T``, ``t_inv = R^T (t * -1)``), then for row i and every j ``u_ij = R_i^T t_j + t_inv_i`` (rotate THEN add) normalized with
    ``sqrt(clamp(|u|^2, min=eps**2))``. Pure statement (records nothing); masks are the caller's products."""
    if frames.dim() < 4 or int(frames.shape[-1]) != 3 or int(frames.shape[-2]) != 3:
        raise RowpairRefused(f"unit_vector_rows: frames {tuple(frames.shape)} must be [..., T, N, 3 (N, CA, C), 3 (xyz)]")
    T, N = int(frames.shape[-4]), int(frames.shape[-3])
    slots = [int(t) for t in slots]
    if any(not (0 <= t < T) for t in slots):
        raise RowpairRefused(f"unit_vector_rows: slots {slots} outside [0, {T})")
    g0, g1 = int(g0), int(g1)
    if not (0 <= g0 <= g1 <= N):
        raise RowpairRefused(f"unit_vector_rows: rows [{g0}, {g1}) outside [0, {N}]")
    f = frames[..., slots, :, :, :]                                           # [..., S, N, 3, 3]
    a, b, c = f[..., 0, :], f[..., 1, :], f[..., 2, :]                        # N, CA, C  [..., S, N, 3]
    e0x, e0y, e0z = _norm3(c[..., 0] - b[..., 0], c[..., 1] - b[..., 1], c[..., 2] - b[..., 2], eps)
    vx, vy, vz = a[..., 0] - b[..., 0], a[..., 1] - b[..., 1], a[..., 2] - b[..., 2]
    cc = vx * e0x + vy * e0y + vz * e0z                                       # e1.dot(e0)
    e1x, e1y, e1z = _norm3(vx - cc * e0x, vy - cc * e0y, vz - cc * e0z, eps)
    e2x = e0y * e1z - e0z * e1y                                               # e0.cross(e1)
    e2y = e0z * e1x - e0x * e1z
    e2z = e0x * e1y - e0y * e1x
    tx, ty, tz = b[..., 0], b[..., 1], b[..., 2]
    ntx, nty, ntz = tx * -1, ty * -1, tz * -1                                 # -translation is translation * -1
    # R has columns (e0, e1, e2); R^T rows are e0, e1, e2: (R^T p).x = e0x p.x + e0y p.y + e0z p.z, .y with e1, .z with e2
    tix = e0x * ntx + e0y * nty + e0z * ntz
    tiy = e1x * ntx + e1y * nty + e1z * ntz
    tiz = e2x * ntx + e2y * nty + e2z * ntz
    r = slice(g0, g1)

    def rows_of(q):                                                           # [..., S, r, 1]
        return q[..., r, None]

    pjx, pjy, pjz = tx[..., None, :], ty[..., None, :], tz[..., None, :]     # [..., S, 1, N]
    ux = rows_of(e0x) * pjx + rows_of(e0y) * pjy + rows_of(e0z) * pjz + rows_of(tix)
    uy = rows_of(e1x) * pjx + rows_of(e1y) * pjy + rows_of(e1z) * pjz + rows_of(tiy)
    uz = rows_of(e2x) * pjx + rows_of(e2y) * pjy + rows_of(e2z) * pjz + rows_of(tiz)
    ux, uy, uz = _norm3(ux, uy, uz, eps)
    out = torch.stack([ux, uy, uz], dim=-1)                                   # [..., S, r, N, 3]
    return out if out_dtype is None else out.to(out_dtype)


def feat_device(value: Optional[str] = None) -> str:
    """``ROWPAIR_TEMPL_FEAT_DEVICE`` (or ``value``): ``cpu`` (default — the stock featurizer's device class) | ``cuda`` (the pair device);
    anything else refused by name."""
    v = (os.environ.get(ENV_TEMPL_FEAT_DEVICE, "") if value is None else str(value)).strip().lower() or "cpu"
    if v not in ("cpu", "cuda"):
        raise RowpairRefused(f"{ENV_TEMPL_FEAT_DEVICE}={v!r}: cpu | cuda")
    return v


def real_template_rows(layout: Layout, n_slots: int, *, pb_coords, pb_mask, frames, frame_mask, asym_id, edges, grp=None, feature_dtype=None,
                       device=None, out_device=None, lead: Sequence[int] = (), keys: Sequence[str] = REAL_PAIR_KEYS,
                       unit_vector_fn: Optional[Callable] = None, distogram_fn: Optional[Callable] = None) -> "TemplatePairRows":
    """The COMPUTED row accessor of the template pair products from per-token precursors — the ``[T, N, N, F]`` features never exist:
    ``template_distogram`` rows = :func:`distogram_rows` (``pb_coords`` ``[..., T, N, 3]`` FLOAT64, NaN = absent; ``edges`` ``[2, n_bins]``
    float64 = the featurizer's squared (lower, upper) bin edges) ``x`` :func:`~opt_core.mem.rowpair.trunk.pair_mask_rows` of ``pb_mask``
    ``[..., T, N]`` ``x`` the chain mask; ``template_unit_vector`` rows = :func:`unit_vector_rows` (``frames`` ``[..., T, N, 3, 3]`` fp32 backbone
    coordinates, NaN = absent: ``nan_to_num(nan=0.0)`` is applied here as the featurizer's first statement does; ``frame_mask`` ``[..., T, N]``)
    ``x`` pair mask ``x`` chain mask — feature x pair mask, then x chain mask (the dense statement's order).
    Chain mask = :func:`same_chain_rows` of ``asym_id`` ``[..., N]`` (the FEATURIZER's ids), or :func:`same_or_group_rows` with ``grp``
    ``[..., T, N]`` (inter-chain groups, -1 = none: :func:`group_feature`). ``device``: where the statements run (None: :func:`feat_device`,
    ``cuda`` = ``out_device``); the precursors move there ONCE; rows are returned on ``out_device`` (None: ``pb_mask``'s device) in
    ``feature_dtype`` (default float32). ``unit_vector_fn`` / ``distogram_fn``: an engine's own statement with the same signature (the named
    engine-specific escape). Refused by name: ``pb_coords`` not float64 (precursor dtype drift), shape disagreements. Records ``templ_mode=computed
    templ_feat_device templ_feat_keys templ_interchain_grp``; the accessor's ``transient_channels`` is :data:`COMPUTED_TRANSIENT_CHANNELS` when the
    statements run on the pair device, else the feature width."""
    if pb_coords.dtype != torch.float64:
        raise RowpairRefused(f"real_template_rows: template pseudo-beta coordinates arrived as {pb_coords.dtype}; the featurizer's float64 is the "
                             "statement's operand (precursor dtype drift)")
    T, N = int(n_slots), int(layout.N)
    for name, t, tail in (("pb_coords", pb_coords, (T, N, 3)), ("pb_mask", pb_mask, (T, N)), ("frames", frames, (T, N, 3, 3)),
                          ("frame_mask", frame_mask, (T, N)), ("asym_id", asym_id, (N,))):
        if tuple(int(v) for v in t.shape[-len(tail):]) != tail:
            raise RowpairRefused(f"real_template_rows: {name} {tuple(t.shape)} must end in {tail} (T={T} slots, N={N} tokens)")
    if grp is not None and tuple(int(v) for v in grp.shape[-2:]) != (T, N):
        raise RowpairRefused(f"real_template_rows: grp {tuple(grp.shape)} must end in {(T, N)}")
    if edges.dim() != 2 or int(edges.shape[0]) != 2:
        raise RowpairRefused(f"real_template_rows: edges {tuple(edges.shape)} must be [2, n_bins] (lower, upper squared bin edges)")
    out_dev = torch.device(out_device) if out_device is not None else pb_mask.device
    where = feat_device(device)
    dev = out_dev if where == "cuda" else torch.device("cpu")
    if where == "cuda" and out_dev.type != "cuda":
        raise RowpairRefused(f"real_template_rows: {ENV_TEMPL_FEAT_DEVICE}=cuda but the pair device is {out_dev}")
    fd = torch.float32 if feature_dtype is None else feature_dtype
    pbD, pbmD, frD, frmD, aidD = (x.to(dev) for x in (pb_coords, pb_mask, frames, frame_mask, asym_id))
    frD = torch.nan_to_num(frD, nan=0.0)                                       # the featurizer's first statement: absent (masked) residues' NaN coordinates -> 0.0
    grpD = grp.to(dev) if grp is not None else None
    lower, upper = edges[0].to(device=dev, dtype=torch.float64), edges[1].to(device=dev, dtype=torch.float64)
    uv_fn = unit_vector_rows if unit_vector_fn is None else unit_vector_fn
    dg_fn = distogram_rows if distogram_fn is None else distogram_fn

    def chain_rows(slots, g0, g1):
        if grpD is not None:
            return same_or_group_rows(aidD, grpD, slots, g0, g1)                # [..., S, r, N, 1]
        return same_chain_rows(aidD, g0, g1)                                   # [..., 1, r, N, 1] broadcast over S

    def distogram(slots, g0, g1):
        slots = [int(t) for t in slots]
        feat = dg_fn(pbD, lower, upper, slots, g0, g1, out_dtype=fd)
        pm = pair_mask_rows(pbmD[..., slots, :], g0, g1)[..., None].to(fd)
        return (feat * pm * chain_rows(slots, g0, g1).to(fd)).to(out_dev)

    def unit_vector(slots, g0, g1):
        slots = [int(t) for t in slots]
        feat = uv_fn(frD, slots, g0, g1, out_dtype=fd)
        pm = pair_mask_rows(frmD[..., slots, :], g0, g1)[..., None].to(fd)
        return (feat * pm * chain_rows(slots, g0, g1).to(fd)).to(out_dev)

    fns = {"template_distogram": distogram, "template_unit_vector": unit_vector}
    unknown = [k for k in keys if k not in fns]
    if unknown:
        raise RowpairRefused(f"real_template_rows: keys {unknown} have no computed statement (known: {sorted(fns)})")
    acc = TemplatePairRows(layout, T, computed={k: fns[k] for k in keys}, lead=lead, dtype=fd, device=out_dev)
    acc.feat_device = where
    acc.transient_channels = COMPUTED_TRANSIENT_CHANNELS if dev == out_dev else sum((int(edges.shape[1]) if k == "template_distogram" else 3) for k in keys)
    record_schedule(templ_feat_device=where, templ_feat_keys="+".join(keys), templ_interchain_grp=int(grp is not None))
    return acc


# ================================================================================================================ slot de-duplication (agreed)
def _nan_equal(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        return bool(torch.equal(torch.isnan(a), torch.isnan(b))) and bool(torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)))
    return bool(torch.equal(a, b))


def _agree_all(flags: Sequence[bool], layout: Optional[Layout], device) -> List[bool]:
    """AND of every rank's ``flags`` (a short list of booleans) — gather to rank 0 (``gather_cat_to_rank0``) + broadcast of the reduced list
    (``broadcast_obj``): two tiny collectives every rank issues. The local list without a group / at P == 1."""
    if layout is None or layout.P == 1 or not is_dist():
        return [bool(f) for f in flags]
    P, _rank = world()
    piece = torch.tensor([[1 if f else 0 for f in flags]], dtype=torch.int32, device=device)     # [1, K]
    allv = gather_cat_to_rank0(piece, [1] * P)                                                  # [P, K] on rank 0
    agreed = None
    if allv is not None:
        agreed = [bool(v) for v in (allv.min(dim=0).values > 0).tolist()]
    return list(broadcast_obj(agreed, src=0))


def groups_word(groups: Sequence[Tuple[int, Sequence[int]]]) -> str:
    """``[(0, [0, 2]), (1, [1, 3])] -> '0+2/1+3'`` (the ``templ_groups`` schedule field)."""
    return "/".join("+".join(str(int(m)) for m in members) for _rep, members in groups) or "none"


def template_slot_groups(keys: Sequence[object], n_slots: int, layout: Optional[Layout] = None, *, dedupe: Optional[bool] = None,
                         device=None) -> List[Tuple[int, List[int]]]:
    """Group template slots whose key tensors are bit-identical (NaN-aware) -> ``[(rep, [rep, members...]), ...]`` in first-appearance
    order. Every key tensor carries the SLOT on dim 0 (pass ``t.movedim(d, 0)`` views: per-token features ``[T, ...]``, row-sliced pair features
    ``[T, R, N, F]``, coordinates ...). The verdict is AGREED across the ranks of ``layout`` (each rank's slot-pair equalities AND-ed via
    :func:`_agree_all`): keys that are this rank's ROWS therefore yield the dense verdict, replicated keys the common one, and every rank runs the
    same number of pair-stack passes (a rank-divergent group list would desynchronise the pair stack's collectives). ``dedupe=False`` or
    ``ROWPAIR_TEMPL_NODEDUPE=1``: every slot is its own group. ``device``: where the agreement's two tiny collectives run (default: the first
    key's device; under NCCL pass the pair tensor's device). Records ``templ_slots``, ``templ_groups``, ``templ_dedupe``."""
    T = int(n_slots)
    if dedupe is None:
        dedupe = not _flag(ENV_TEMPL_NODEDUPE)
    for k in keys:
        if int(k.shape[0]) != T:
            raise RowpairRefused(f"template_slot_groups: a key has shape {tuple(k.shape)}; the slot dim (size T={T}) must be dim 0")
    pairs = [(a, b) for a in range(T) for b in range(a + 1, T)]
    if dedupe and pairs:
        local = [all(_nan_equal(k[a], k[b]) for k in keys) for (a, b) in pairs]
    else:
        local = [False] * len(pairs)
    if device is None:                                                                           # the agreement's tensors live where the group's
        device = keys[0].device if keys else torch.device("cpu")                                 # collectives take them (NCCL: the pair tensor's device)
    eq = dict(zip(pairs, _agree_all(local, layout, device)))                                    # agreed even when trivially all-False: same call count on every rank
    groups: List[Tuple[int, List[int]]] = []
    for t in range(T):
        for rep, members in groups:
            if eq.get((rep, t), False):
                members.append(t)
                break
        else:
            groups.append((t, [t]))
    record_schedule(templ_slots=T, templ_groups=groups_word(groups), templ_dedupe="on" if dedupe else "off")
    return groups


def _guard_groups(groups: Sequence[Tuple[int, Sequence[int]]], n_templ: int, layout: Layout, device) -> None:
    """Prove the slot-group list is identical on every rank (one all-gather of a checksum); a mismatch is refused by name (it would otherwise
    desynchronise the per-group pair-stack collectives)."""
    covered = sorted(int(m) for _rep, members in groups for m in members)
    if covered != list(range(int(n_templ))):
        raise RowpairRefused(f"template_embed_rows: slot_groups {groups_word(groups)} do not cover slots 0..{int(n_templ) - 1} exactly once")
    enc = []
    for rep, members in groups:
        enc += [int(rep), -1] + [int(m) for m in members] + [-2]
    allreduce_checksum(torch.tensor(enc, dtype=torch.int64, device=device), name=f"template slot_groups {groups_word(groups)}", raise_on_mismatch=True)


# ================================================================================================================ row block
def templ_block_rows(N: int, c_t: int, n_templ: int, C_z: int, *, rows: Optional[int] = None, elem_bytes: int = 4, feat_dims: int = 64,
                     n_max: Optional[int] = None) -> Tuple[int, str]:
    """``(rows, source)`` of the template row block = :func:`~opt_core.mem.rowpair.shard.choose_block_rows` (THE chooser) over this module's
    per-row transient channel count ``c_t * (2 + n_templ) + C_z + feat_dims`` (the projected z rows and one slot's ``u`` rows, the closing
    ``[T, rows, N, c_t]`` stack, the ``finish_fn`` output, the feature rows): ``rows`` given -> ``given``; else the ``ROWPAIR_ROWBLK_MB`` MiB
    target (``env:ROWPAIR_ROWBLK_MB`` when set, ``default`` = 512 MiB otherwise), int32-capped, clamped to ``n_max`` (the longest shard).
    P-invariant by construction (no rank-local reading enters). Records ``templ_rows`` / ``templ_rows_source``."""
    channels = int(c_t) * (2 + int(n_templ)) + int(C_z) + int(feat_dims)
    n, source = choose_block_rows(int(N), channels, int(elem_bytes), rows=rows, n_max=n_max)
    record_schedule(templ_rows=n, templ_rows_source=source)
    return n, source


# ================================================================================================================ host parking
class ParkedStorage(object):
    """Park a tensor that OWNS its whole storage on the host and release its device storage in place (``untyped_storage().resize_(0)``);
    :meth:`restore` re-grows the SAME storage and copies the data back, so every reference the callers hold (the trunk's ``z_shard``, views)
    stays valid; :meth:`block` (alias :meth:`zrows`) serves row slabs ``[..., i0:i1, N, C]`` from the host copy while the device storage is
    released; :meth:`drop` releases the host copy WITHOUT restoring (the device storage stays released: the tensor is consumed). No arithmetic
    touches the data (bit copy out, bit copy back). Host buffer: from ``pool`` when given (a :class:`opt_core.mem.torch_hostpair.PinPool` —
    :func:`.trunk.park_pool` builds the family's private ones: page-locked for a CUDA tensor, a refused page-lock answered with a pageable buffer
    whose kind is NAMED, or refused by name under ``ROWPAIR_PARK_STRICT=1``), else such a private pool (``ROWPAIR_PARK_PIN_MAX_GB`` /
    ``ROWPAIR_PARK_STRICT``); the copy is :func:`.trunk.host_alloc`'s exact-size chunked buffer (``ROWPAIR_PARK_CHUNK_GIB``; ``chunks`` / ``pinned_gib``
    say what it occupies). ``where`` names the placement in the census vocabulary of :class:`.trunk.ShardPark`: ``device`` (not parked) | ``host_pinned`` |
    ``host_pageable:<kind>`` (``kind`` = ``pin_alloc_failed`` | ``pin_budget_exceeded`` | ``host_ram_insufficient``) | ``host`` (a CPU tensor's plain
    host copy); ``fallback`` is ``<kind>`` or None. ``shape`` / ``dtype`` / ``device`` / ``dim()`` are the parked tensor's, so a park is accepted where a
    row-served shard is (:func:`.heads.embed_rows`). A tensor that does not own a contiguous whole storage, or whose storage is not resizable (foreign
    memory: :func:`.shard.storage_resizable`), is NOT parked (``ok=False``, ``why`` = ``shared_storage`` | ``storage_not_resizable``; :meth:`restore` /
    :meth:`block` then read the live tensor). ``pool=None`` under ``ROWPAIR_HOST_SLAB=lease``: the host copy is this rank's LEASED slab
    (:func:`.trunk.park_host` -> :class:`.trunk.HostSlabLease`; a :class:`.trunk.LeasedHost` in ``host``): :meth:`restore` / :meth:`drop` hand the slab
    back to the lease; a park taken while another holder has the slab gets a NAMED private buffer (census ``host_slab_second``)."""

    STATS = {"parked": 0, "restored": 0, "bytes": 0}

    def __init__(self, t, name: str = "z_shard", log: Optional[Callable[[str], None]] = None, *, pool=None):
        self.t, self.name, self.ok, self.host, self.how, self._log = t, str(name), False, None, "not parked", log
        self.where, self.fallback, self.pinned, self.dropped, self.why, self._pool = "device", None, False, False, None, pool
        st = t.untyped_storage()
        self.nbytes = int(t.numel() * t.element_size())
        if not owns_whole_storage(t):
            self.why = "shared_storage"
            self.how = f"not parked ({self.name} does not own a contiguous whole storage: {int(st.nbytes())} B vs {self.nbytes} B)"
            self._say(self.how)
            return
        if not storage_resizable(t):
            self.why = "storage_not_resizable"
            self.how = f"not parked ({self.name}: storage not resizable — foreign memory, e.g. a DLPack view of another framework's buffer)"
            self._say(self.how)
            return
        t0 = time.time()
        if pool is None:
            pool = park_host(t)                                                    # ROWPAIR_HOST_SLAB unset: park_pool(pin=) — the family's one pool constructor
            self._pool = pool                                                      # (PIN_MAX_GB / STRICT / CHUNK_GIB); =lease: this rank's HostSlabLease
        self.host, self.where, self.fallback, pinned = host_alloc(pool, t, f"ParkedStorage({self.name})", self._say)
        self.pinned = pinned
        self.host.store(t)
        release_storage_(t)                                                    # synchronize (CUDA), then storage.resize_(0)
        if t.is_cuda:
            torch.cuda.empty_cache()
        self.ok = True
        self.how = f"parked {'pinned' if pinned else 'pageable'}"
        ParkedStorage.STATS["parked"] += 1
        ParkedStorage.STATS["bytes"] += self.nbytes
        self._say(f"[park] {self.name} {tuple(t.shape)} {self.nbytes / 2**30:.2f} GiB -> {self.how} host in {time.time() - t0:.1f}s; storage released "
                  f"chunks={self.host.n_chunks} pinned_gib={self.host.pinned_bytes / 2**30:.2f}")

    def _say(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    # --- the parked tensor's facts (a park stands in for the shard wherever rows are served: heads.embed_rows / _rows_dim_checks)
    @property
    def shape(self):
        return self.t.shape

    @property
    def dtype(self):
        return self.t.dtype

    @property
    def device(self):
        return self.t.device

    def dim(self) -> int:
        return int(self.t.dim())

    def _release_host(self) -> None:
        if self.host is not None:
            self.host.release()
        self.host = None

    @property
    def chunks(self) -> int:
        """Host chunks the park holds (0 when not parked / restored / dropped)."""
        return int(self.host.n_chunks) if self.host is not None else 0

    @property
    def pinned_gib(self) -> float:
        """What the page-locked chunks occupy in torch's caching host allocator (each rounded to the next power of two), GiB."""
        return round(self.host.pinned_bytes / 2 ** 30, 3) if self.host is not None else 0.0

    def block(self, i0: int, i1: int):
        """Rows ``i0:i1`` (dim -3) on the tensor's device: a staged copy of the host slab while parked, else a view of the live tensor."""
        if self.dropped:
            raise RowpairRefused(f"ParkedStorage({self.name}).block: the host copy was dropped (the tensor is consumed)")
        if not self.ok or self.host is None:
            return self.t[..., i0:i1, :, :]
        stage = torch.empty(tuple(self.t.shape[:-3]) + (int(i1) - int(i0),) + tuple(self.t.shape[-2:]), dtype=self.t.dtype, device=self.t.device)
        return load_rows(self.host, tuple(self.t.shape), stage.element_size(), int(i0), int(i1), stage, non_blocking=False)

    zrows = block
    read_block = block

    def write_block(self, i0: int, i1: int, src) -> None:
        """The inverse of :meth:`block`: write ``src`` ``[..., i1 - i0, N, C]`` (the tensor's dtype; any device) into rows ``i0:i1``
        of the PARKED host copy — byte copies (:func:`.trunk.store_rows`), returned after the bytes landed so ``src`` may be freed at once. While
        NOT parked (``ok=False``: the live tensor is the data) the rows are written into the live tensor (``t[..., i0:i1, :, :].copy_(src)``).
        Refused by name after :meth:`drop` / :meth:`restore` released the host copy, or for a ``src`` of another dtype / row shape."""
        if self.dropped:
            raise RowpairRefused(f"ParkedStorage({self.name}).write_block: the host copy was dropped (the tensor is consumed)")
        i0, i1 = int(i0), int(i1)
        R = int(self.t.shape[-3])
        if i0 < 0 or i1 > R or i1 < i0:
            raise RowpairRefused(f"ParkedStorage({self.name}).write_block: rows {i0}:{i1} outside [0, {R})")
        want = tuple(int(v) for v in self.t.shape[:-3]) + (i1 - i0,) + tuple(int(v) for v in self.t.shape[-2:])
        if tuple(int(v) for v in src.shape) != want or src.dtype != self.t.dtype:
            raise RowpairRefused(f"ParkedStorage({self.name}).write_block: src {tuple(src.shape)} {src.dtype} for rows {i0}:{i1}; expected {want} {self.t.dtype}")
        if not self.ok:
            self.t[..., i0:i1, :, :].copy_(src)
            return
        if self.host is None:
            raise RowpairRefused(f"ParkedStorage({self.name}).write_block: the park was restored (write the live tensor's rows instead)")
        store_rows(self.host, tuple(self.t.shape), int(self.t.element_size()), i0, i1, src.contiguous(), non_blocking=False)

    def restore(self):
        """Re-grow the storage of the SAME tensor object and copy the parked bytes back; returns the tensor. No-op when not parked."""
        if self.dropped:
            raise RowpairRefused(f"ParkedStorage({self.name}).restore: the host copy was dropped (the tensor is consumed)")
        if not self.ok or self.host is None:
            return self.t
        t0 = time.time()
        regrow_storage_(self.t, self.nbytes)
        self.host.load(self.t)
        if self.t.is_cuda:
            torch.cuda.synchronize()
        self._release_host()
        self.where = "device"
        ParkedStorage.STATS["restored"] += 1
        self._say(f"[park] {self.name} restored to {self.t.device} in {time.time() - t0:.1f}s")
        return self.t

    def drop(self) -> None:
        """Release the host copy WITHOUT restoring: the device storage stays released (0 bytes) and the tensor holds no data — every later
        :meth:`block` / :meth:`restore` is refused by name. No-op when not parked; idempotent."""
        if not self.ok or self.dropped:
            return
        self._release_host()
        self.dropped = True
        self.where = "dropped"
        self._say(f"[park] {self.name} host copy dropped; device storage stays released")


# ================================================================================================================ the driver
def template_embed_rows(z_shard, layout: Layout, *, n_templ: int, c_t: int, unit_rows_fn: Callable[[object, int, Tuple[int, int]], object],
                        pair_stack_fn: Callable[[object, object], object], finish_fn: Callable[[object], object], mask_loc=None,
                        slot_groups: Optional[Sequence[Tuple[int, Sequence[int]]]] = None, rows: Optional[int] = None, add: bool = True, out=None,
                        park_z: Optional[bool] = None, park_u: Optional[bool] = None, log: Optional[Callable[[str], None]] = None,
                        feat_channels: Optional[int] = None, order: Optional[str] = None):
    """The template embedder on this rank's rows. ``z_shard`` ``[..., R, N, C_z]`` (row dim -3); returns ``z_shard`` with the template term
    ADDED IN PLACE per row block (``add=True``), else ``out`` (allocated when None) holding ``z rows + term``. ``c_t``: the template channel
    count (sizes the ``u`` slabs and the budgeted row block).

    Callables (engine statements; every tensor they return is this rank's rows):
      ``unit_rows_fn(z_rows, slot, (g0, g1)) -> [..., 1, g1-g0, N, c_t]``   ``v_t`` rows: the z projection of ``z_rows`` (``= z_shard[..., i0:i1, :, :]``,
                                                                           GLOBAL rows ``g0:g1``) plus the embedded pair features of slot ``slot``
      ``pair_stack_fn(u, mask_loc) -> u``                                  the per-slot template pair stack (its final LayerNorm included) on the
                                                                           row shard ``u [..., 1, R, N, c_t]`` — the row-sharded pair-block installs
      ``finish_fn(t) -> [..., rows, N, C_z]``                              the closing statement on ``t [..., T, rows, N, c_t]`` (slots in their
                                                                           ORIGINAL order): mean over slots, activation, output linear

    ``slot_groups``: the AGREED groups of identical slots (:func:`template_slot_groups`); None = every slot its own group. The list is proven
    identical across ranks (checksum) before use. ``rows``: the row block of the u-construction and closing loops (None: :func:`templ_block_rows`);
    recorded as ``templ_rows`` / ``templ_rows_source``; the blocks are :func:`~opt_core.mem.rowpair.shard.iter_row_blocks` (starts on the layout's chunk grid). ``park_z`` / ``park_u`` (None:
    ``ROWPAIR_TEMPL_PARK_Z`` / ``ROWPAIR_TEMPL_PARK_U``, default off): :class:`ParkedStorage` of ``z_shard`` while the pair stacks run (z is not
    read again until the close; restored into the same tensor object before it) and of each finished ``u`` slab when there is more than one group
    (rows streamed back from the host at the close); the outcome is recorded as ``templ_park_z`` (off|parked|not_parked) / ``templ_park_u``
    (off|<k>of<n>) — parking is a memory lever only (no value changes, nothing replicated or gathered). ``feat_channels``: the per-row-element
    transient width of the feature rows ``unit_rows_fn`` reads (the accessor's ``transient_channels``: computed features carry their featurizer
    transients inside the row block; None = 64), entering the row-block budget (:func:`templ_block_rows` ``feat_dims``; recorded
    ``templ_feat_channels``). P == 1 / replicated / no group: refused by name. ``order`` (None: ``ROWPAIR_TEMPL_ORDER``, default ``classic`` =
    this body): ``zparked`` runs :func:`template_embed_rows_zparked` on the same arguments (the same statements with z parked
    FIRST; bitwise this order's result at a lower device peak)."""
    if templ_order(order) == "zparked":
        return template_embed_rows_zparked(z_shard, layout, n_templ=n_templ, c_t=c_t, unit_rows_fn=unit_rows_fn, pair_stack_fn=pair_stack_fn,
                                           finish_fn=finish_fn, mask_loc=mask_loc, slot_groups=slot_groups, rows=rows, add=add, out=out,
                                           park_u=park_u, log=log, feat_channels=feat_channels)
    require_sharded(layout, "template_embed_rows")
    R, N = layout.R, layout.N
    if z_shard.dim() < 3 or int(z_shard.shape[-3]) != R or int(z_shard.shape[-2]) != N:
        raise RowpairRefused(f"template_embed_rows: z shard {tuple(z_shard.shape)} vs layout rows R={R} N={N} (row dim is -3)")
    n_templ, c_t = int(n_templ), int(c_t)
    if n_templ < 1 or c_t < 1:
        raise RowpairRefused(f"template_embed_rows: n_templ={n_templ} c_t={c_t}: at least one template slot and one channel are required "
                             "(an untemplated engine skips the embedder)")
    if out is not None and (tuple(out.shape) != tuple(z_shard.shape) or out.dtype != z_shard.dtype):
        raise RowpairRefused(f"template_embed_rows: out {tuple(out.shape)} {out.dtype} must match the z shard {tuple(z_shard.shape)} {z_shard.dtype}")
    lead = tuple(int(v) for v in z_shard.shape[:-3])
    C_z = int(z_shard.shape[-1])
    dev = z_shard.device
    groups = [(t, [t]) for t in range(n_templ)] if slot_groups is None else [(int(rep), [int(m) for m in members]) for rep, members in slot_groups]
    _guard_groups(groups, n_templ, layout, dev)
    fch = 64 if feat_channels is None else int(feat_channels)
    step, _source = templ_block_rows(N, c_t, n_templ, C_z, rows=rows, elem_bytes=int(z_shard.element_size()), feat_dims=fch, n_max=layout.Rmax)
    record_schedule(templ_slots=n_templ, templ_groups=groups_word(groups), templ_replicated="token_feats", templ_feat_channels=fch)
    park_z = _flag(ENV_TEMPL_PARK_Z) if park_z is None else bool(park_z)
    park_u = (_flag(ENV_TEMPL_PARK_U) if park_u is None else bool(park_u)) and len(groups) > 1
    r0 = layout.r0
    # ---- per unique slot: u = unit_rows_fn(z rows, rep, rows) produced in row blocks -> [..., 1, R, N, c_t]
    us: List[object] = []
    for rep, _members in groups:
        u = z_shard.new_empty(lead + (1, R, N, c_t))

        def unit_block(g0, g1, rep=rep):
            y = unit_rows_fn(z_shard[..., g0 - r0:g1 - r0, :, :], rep, (g0, g1))
            if y.dim() != len(lead) + 4 or int(y.shape[-4]) != 1:
                raise RowpairRefused(f"template_embed_rows: unit_rows_fn returned {tuple(y.shape)} for slot {rep} rows {g0}:{g1}; expected "
                                     f"[..., 1, {g1 - g0}, {N}, {c_t}]")
            return y

        us.append(produce_rows_(u, layout, unit_block, op="set", block_rows=step))
        del u
    # ---- z is not read again until the close: optionally park it on the host while the pair stacks run
    parked = ParkedStorage(z_shard, name="z_shard (template pair stack)", log=log) if park_z else None
    # ---- the per-slot pair stack on the row shard, one pass per unique slot
    for gi in range(len(us)):
        y = pair_stack_fn(us[gi], mask_loc)
        if tuple(int(v) for v in y.shape) != tuple(int(v) for v in us[gi].shape):
            raise RowpairRefused(f"template_embed_rows: pair_stack_fn returned {tuple(y.shape)} for a slab {tuple(us[gi].shape)} (rows must stay this rank's)")
        us[gi] = y
        del y
        if park_u:
            us[gi] = ParkedStorage(us[gi].contiguous(), name=f"template slab u (slot group {gi})", log=log)
    slot_to_group = {}
    for gi, (_rep, members) in enumerate(groups):
        for m in members:
            slot_to_group[int(m)] = gi
    n_u_parked = sum(1 for ug in us if isinstance(ug, ParkedStorage) and ug.ok)
    record_schedule(templ_park_z=("off" if parked is None else ("parked" if parked.ok else "not_parked")),
                    templ_park_u=("off" if not park_u else f"{n_u_parked}of{len(us)}"))
    if parked is not None and parked.ok:
        record_schedule(templ_park_z_where=parked.where, templ_park_z_gib=round(parked.nbytes / 2 ** 30, 3), templ_park_z_chunks=parked.chunks,
                        templ_park_z_pinned_gib=parked.pinned_gib)
    if parked is not None:
        parked.restore()
        parked = None

    # ---- closing statements per row block: t [..., T, rows, N, c_t] in the ORIGINAL slot order -> finish_fn
    def close_block(g0, g1):
        i0, i1 = g0 - r0, g1 - r0
        blk = {}
        for gi in set(slot_to_group.values()):
            ug = us[gi]
            blk[gi] = ug.block(i0, i1)[..., 0, :, :, :] if isinstance(ug, ParkedStorage) else ug[..., 0, i0:i1, :, :]
        t = torch.stack([blk[slot_to_group[ti]] for ti in range(n_templ)], dim=-4)
        del blk
        y = finish_fn(t)
        del t
        if y.dim() != len(lead) + 3 or int(y.shape[-1]) != C_z:
            raise RowpairRefused(f"template_embed_rows: finish_fn returned {tuple(y.shape)} for rows {g0}:{g1}; expected [..., {g1 - g0}, {N}, {C_z}]")
        return y if add else z_shard[..., i0:i1, :, :] + y

    if add:
        result = produce_rows_(z_shard, layout, close_block, op="add", block_rows=step)
    else:
        result = produce_rows_(out if out is not None else torch.empty_like(z_shard), layout, close_block, op="set", block_rows=step)
    return result


def _like_rows_view(stage, z_shard):
    """Give a staged z row block ``[..., rows, N, C]`` the STRIDES the classic order's view ``z_shard[..., i0:i1, :, :]`` has (the shard's own
    strides) when every leading dim is of size 1: a size-1 dim's stride addresses nothing, but ``torch.matmul`` reads it when it decides between
    folding the leading dims into one GEMM and a batched GEMM — two launch shapes, two roundings. With identical sizes AND strides every statement
    downstream (LayerNorm, Linear, ``empty_like`` / ``clone`` preserve_format) sees byte-identical metadata in both orders. A 3-dim shard (no lead)
    or a lead with an extent > 1 is returned as staged (contiguous; a lead > 1 cannot alias the shard's row pitch without its bytes)."""
    lead = tuple(int(v) for v in stage.shape[:-3])
    if not lead or any(d != 1 for d in lead) or stage.dim() != z_shard.dim():
        return stage
    want = tuple(int(v) for v in z_shard.stride())
    if tuple(int(v) for v in stage.stride()[-3:]) != want[-3:]:                 # the shard's row / token / channel pitch must be the contiguous one
        return stage                                                             # (it is: a parkable shard owns a contiguous whole storage)
    return stage.as_strided(tuple(int(v) for v in stage.shape), want)


def templ_order(order: Optional[str] = None) -> str:
    """The template-stage statement ORDER word: ``order`` given wins, else ``ROWPAIR_TEMPL_ORDER`` (unset / empty -> ``classic``). One of
    :data:`TEMPL_ORDERS`; any other word is refused by name."""
    w = (os.environ.get(ENV_TEMPL_ORDER, "") if order is None else str(order)).strip().lower()
    if w == "":
        return "classic"
    if w not in TEMPL_ORDERS:
        raise RowpairRefused(f"{ENV_TEMPL_ORDER if order is None else 'template order'}={w!r}: one of {TEMPL_ORDERS} is required")
    return w


def template_embed_rows_zparked(z_shard, layout: Layout, *, n_templ: int, c_t: int, unit_rows_fn: Callable[[object, int, Tuple[int, int]], object],
                                pair_stack_fn: Callable[[object, object], object], finish_fn: Callable[[object], object], mask_loc=None,
                                slot_groups: Optional[Sequence[Tuple[int, Sequence[int]]]] = None, rows: Optional[int] = None, add: bool = True,
                                out=None, park_u: Optional[bool] = None, log: Optional[Callable[[str], None]] = None,
                                feat_channels: Optional[int] = None):
    """:func:`template_embed_rows` in the Z-PARKED order (``ROWPAIR_TEMPL_ORDER=zparked``). The SAME statements on the same row
    blocks — ``unit_rows_fn`` / ``pair_stack_fn`` / ``finish_fn`` see byte-identical operands of identical shapes in the same order per slot group, so
    the result is BITWISE the classic order's — with a different residency:

      (1) ``z_shard`` is parked FIRST (:class:`ParkedStorage`: the leased slab under ``ROWPAIR_HOST_SLAB=lease``, else a private pool) — the classic
          order builds every slot group's ``u`` slab from DEVICE z rows and parks z only afterwards, so its stage peak is ``Z + G*u_slab + block``;
      (2) per slot group g (in group order): the slab ``u_g [..., 1, R, N, c_t]`` is built by streaming z row blocks BACK from the park
          (:meth:`ParkedStorage.block`: a staged device copy of rows ``i0:i1``, bit-identical to ``z_shard[..., i0:i1, :, :]``) through
          ``unit_rows_fn`` on :func:`~opt_core.mem.rowpair.shard.produce_rows_`'s grid; its pair stack runs at once (device: the finished slabs of
          groups < g unless parked + this slab + the stack's transients; z is off-device); a finished slab of a NON-last group is parked when
          ``park_u`` (None: ``ROWPAIR_TEMPL_PARK_U``, whose DEFAULT is ON in this order — ``0`` keeps finished slabs on device);
      (3) the close runs WITH z STILL PARKED: per row block (the grid the classic close's ``produce_rows_`` walks: same chooser, same arguments)
          ``t = stack(slab rows)`` -> ``y = finish_fn(t)`` -> the z row block is streamed in (``block``), ``+= y`` (``add``; else ``out`` rows =
          ``z rows + y``) and written back into the park (:meth:`ParkedStorage.write_block`);
      (4) the slabs are dropped, then z is restored into the SAME tensor object (:meth:`ParkedStorage.restore`).

    Device peak ≈ ``max(u_slab + stack transients, G_live*u_slab + 2 blocks)`` instead of ``Z + G*u_slab`` (Z = 2 u_slab at c_t = C_z/2); traffic
    = one D2H + one H2D of z for the park (as ``ROWPAIR_TEMPL_PARK_Z=1``) plus one z read per slot group (the builds) and one z read + write (the close),
    at PCIe rate. A z shard that cannot be parked (shared / foreign storage: ``ParkedStorage.ok`` False) runs the CLASSIC order by name (census
    ``templ_order=zparked:not_parked(<why>)``). Census: ``templ_order=zparked``, ``templ_park_z*`` (the park's placement words), ``templ_park_u``,
    ``templ_zparked_h2d_gib`` / ``templ_zparked_d2h_gib`` (bytes moved), ``templ_zparked_peak_gib`` / ``_at`` (sampled device residency at the
    stage's block boundaries: a lower bound of the true high-water, which :func:`.trunk.run_trunk_sharded` brackets exactly as ``templ_stage_peak_gib``
    when the stage sets the process high-water). P == 1 / replicated: refused by name (as the classic order)."""
    require_sharded(layout, "template_embed_rows_zparked")
    R, N = layout.R, layout.N
    if z_shard.dim() < 3 or int(z_shard.shape[-3]) != R or int(z_shard.shape[-2]) != N:
        raise RowpairRefused(f"template_embed_rows_zparked: z shard {tuple(z_shard.shape)} vs layout rows R={R} N={N} (row dim is -3)")
    n_templ, c_t = int(n_templ), int(c_t)
    if n_templ < 1 or c_t < 1:
        raise RowpairRefused(f"template_embed_rows_zparked: n_templ={n_templ} c_t={c_t}: at least one template slot and one channel are required "
                             "(an untemplated engine skips the embedder)")
    if out is not None and (tuple(out.shape) != tuple(z_shard.shape) or out.dtype != z_shard.dtype):
        raise RowpairRefused(f"template_embed_rows_zparked: out {tuple(out.shape)} {out.dtype} must match the z shard {tuple(z_shard.shape)} {z_shard.dtype}")
    say = log if log is not None else (lambda _m: None)
    lead = tuple(int(v) for v in z_shard.shape[:-3])
    C_z = int(z_shard.shape[-1])
    dev = z_shard.device
    groups = [(t, [t]) for t in range(n_templ)] if slot_groups is None else [(int(rep), [int(m) for m in members]) for rep, members in slot_groups]
    _guard_groups(groups, n_templ, layout, dev)
    fch = 64 if feat_channels is None else int(feat_channels)
    step, _source = templ_block_rows(N, c_t, n_templ, C_z, rows=rows, elem_bytes=int(z_shard.element_size()), feat_dims=fch, n_max=layout.Rmax)
    record_schedule(templ_slots=n_templ, templ_groups=groups_word(groups), templ_replicated="token_feats", templ_feat_channels=fch)
    if park_u is None:                                                             # this order's default: park finished NON-last slabs (the classic
        pu_env = os.environ.get(ENV_TEMPL_PARK_U, "").strip()                      # order's default is off); ROWPAIR_TEMPL_PARK_U=0 keeps them on device
        park_u = True if pu_env == "" else _flag(ENV_TEMPL_PARK_U)
    park_u = bool(park_u) and len(groups) > 1
    r0 = layout.r0
    t_stage = time.time()
    peak = DevicePeak(dev)
    gib = float(2 ** 30)
    moved = {"h2d": 0, "d2h": 0}
    # ---- (1) park z FIRST: nothing of the stage is on the device yet
    peak.phase("park")
    zpark = ParkedStorage(z_shard, name="z_shard (template pair stack)", log=log)
    if not zpark.ok:
        record_schedule(templ_order=f"zparked:not_parked({zpark.why})")
        say(f"[templ] order=zparked: z shard not parked ({zpark.why}) — running the classic order (z stays on device)")
        return template_embed_rows(z_shard, layout, n_templ=n_templ, c_t=c_t, unit_rows_fn=unit_rows_fn, pair_stack_fn=pair_stack_fn, finish_fn=finish_fn,
                                   mask_loc=mask_loc, slot_groups=slot_groups, rows=rows, add=add, out=out, park_z=False, park_u=park_u, log=log,
                                   feat_channels=feat_channels, order="classic")
    moved["d2h"] += zpark.nbytes
    record_schedule(templ_order="zparked", templ_park_z="parked", templ_park_z_where=zpark.where, templ_park_z_gib=round(zpark.nbytes / gib, 3),
                    templ_park_z_chunks=zpark.chunks, templ_park_z_pinned_gib=zpark.pinned_gib)
    peak.mark("z_parked")
    # ---- (2) per unique slot group: build u_g from PARKED z rows, run its pair stack, park it when another group follows
    us: List[object] = []
    for gi, (rep, _members) in enumerate(groups):
        peak.phase(f"build{gi}")
        u = z_shard.new_empty(lead + (1, R, N, c_t))

        def unit_block(g0, g1, rep=rep):
            zr = _like_rows_view(zpark.block(g0 - r0, g1 - r0), z_shard)              # staged device copy of z rows g0:g1 (bit-identical to the shard's rows,
            moved["h2d"] += int(zr.numel()) * int(zr.element_size())                   # carrying the strides the classic order's view carries)
            y = unit_rows_fn(zr, rep, (g0, g1))
            peak.mark(f"build{gi}", 0)
            del zr
            if y.dim() != len(lead) + 4 or int(y.shape[-4]) != 1:
                raise RowpairRefused(f"template_embed_rows_zparked: unit_rows_fn returned {tuple(y.shape)} for slot {rep} rows {g0}:{g1}; expected "
                                     f"[..., 1, {g1 - g0}, {N}, {c_t}]")
            return y

        u = produce_rows_(u, layout, unit_block, op="set", block_rows=step)
        peak.mark(f"built{gi}")
        peak.phase(f"stack{gi}")
        y = pair_stack_fn(u, mask_loc)
        if tuple(int(v) for v in y.shape) != tuple(int(v) for v in u.shape):
            raise RowpairRefused(f"template_embed_rows_zparked: pair_stack_fn returned {tuple(y.shape)} for a slab {tuple(u.shape)} (rows must stay this rank's)")
        del u
        peak.mark(f"stack{gi}")
        if park_u and gi < len(groups) - 1:
            y = ParkedStorage(y.contiguous(), name=f"template slab u (slot group {gi})", log=log)
            if y.ok:
                moved["d2h"] += y.nbytes
        us.append(y)
        del y
    slot_to_group = {}
    for gi, (_rep, members) in enumerate(groups):
        for m in members:
            slot_to_group[int(m)] = gi
    n_u_parked = sum(1 for ug in us if isinstance(ug, ParkedStorage) and ug.ok)
    record_schedule(templ_park_u=("off" if not park_u else f"{n_u_parked}of{len(us)}"))
    # ---- (3) the close WITH z STILL PARKED, on the grid produce_rows_ walks for the classic close (same chooser, same arguments)
    peak.phase("close")
    n_loc = int(layout.n_loc)
    rd = z_shard.dim() - 3
    tgt = None
    if not add:
        tgt = out if out is not None else torch.empty(tuple(int(v) for v in z_shard.shape), dtype=z_shard.dtype, device=dev)
    if n_loc > 0:
        unit = int(layout.align or chunk_align())
        per_row = max(1, int(z_shard.numel()) // max(1, n_loc))
        block_rows, source = choose_block_rows(elems_per_row=per_row, elem_bytes=int(z_shard.element_size()), rows=step, align=unit, n_max=layout.n_max)
        blocks = iter_row_blocks(layout, block_rows, unit)
        record_schedule(produce_block_rows=block_rows, produce_block_source=source)
        for b0, b1, g0, g1 in blocks:
            blk = {}
            for gi in set(slot_to_group.values()):
                ug = us[gi]
                blk[gi] = ug.block(b0, b1)[..., 0, :, :, :] if isinstance(ug, ParkedStorage) else ug[..., 0, b0:b1, :, :]
            t = torch.stack([blk[slot_to_group[ti]] for ti in range(n_templ)], dim=-4)
            del blk
            y = finish_fn(t)
            del t
            if y.dim() != len(lead) + 3 or int(y.shape[-1]) != C_z:
                raise RowpairRefused(f"template_embed_rows_zparked: finish_fn returned {tuple(y.shape)} for rows {g0}:{g1}; expected [..., {g1 - g0}, {N}, {C_z}]")
            zr = _like_rows_view(zpark.block(b0, b1), z_shard)                        # the z row block, staged from the park (the classic view's strides)
            moved["h2d"] += int(zr.numel()) * int(zr.element_size())
            peak.mark("close", 0)
            if add:
                if tuple(y.shape) != tuple(zr.shape):
                    raise RowpairRefused(f"template_embed_rows_zparked: fn returned {tuple(y.shape)} for global rows {g0}:{g1}; expected {tuple(zr.shape)}")
                zr.add_(y)                                                             # == produce_rows_(op='add'): dst.add_(y) on the same bytes
                zpark.write_block(b0, b1, zr)                                          # the closed rows go back into the park (byte copy)
                moved["d2h"] += int(zr.numel()) * int(zr.element_size())
            else:
                val = zr + y                                                            # == the classic close_block's `z rows + y`, then produce_rows_(op='set')
                dst = tgt.narrow(rd, b0, b1 - b0)
                if tuple(val.shape) != tuple(dst.shape):
                    raise RowpairRefused(f"template_embed_rows_zparked: fn returned {tuple(val.shape)} for global rows {g0}:{g1}; expected {tuple(dst.shape)}")
                dst.copy_(val)
                del val
            del y, zr
    # ---- (4) drop the slabs, restore z into the same tensor object
    for ug in us:
        if isinstance(ug, ParkedStorage):
            ug.drop()
    ug = None                                                                      # the loop name must not keep the last slab alive through the restore
    del us, ug
    peak.phase("restore")
    peak.mark("before_restore")
    zpark.restore()
    moved["h2d"] += zpark.nbytes
    pk = peak.close("restored")
    phases_word = ",".join(f"{k}:{v:.2f}" for k, v in (pk.get("phases") or {}).items())
    record_schedule(templ_zparked_phases_gib=phases_word)
    record_schedule(templ_zparked_h2d_gib=round(moved["h2d"] / gib, 3), templ_zparked_d2h_gib=round(moved["d2h"] / gib, 3),
                    templ_zparked_peak_gib=pk.get("sampled_peak_gib", pk["peak_gib"]),
                    templ_zparked_peak_at=pk.get("sampled_at", pk["at"]), templ_zparked_s=round(time.time() - t_stage, 3))
    say(f"[templ] order=zparked: {len(groups)} slot group(s), row block {step}; z parked first ({zpark.nbytes / gib:.2f} GiB {zpark.where}), "
        f"slabs parked {n_u_parked if park_u else 'off'}; moved h2d {moved['h2d'] / gib:.2f} GiB d2h {moved['d2h'] / gib:.2f} GiB; "
        f"sampled device residency max {pk.get('sampled_peak_gib', pk['peak_gib']):.2f} GiB at {pk.get('sampled_at', pk['at'])}; per phase [{phases_word}] GiB "
        f"(entry {pk['entry_gib']:.2f}, exit {pk['exit_gib']:.2f} GiB); {time.time() - t_stage:.1f}s")
    return z_shard if add else tgt


def template_embed_dense(z, *, n_templ: int, unit_fn: Callable[[object, int, Tuple[int, int]], object],
                         pair_stack_fn: Callable[[object, object], object], finish_fn: Callable[[object], object], mask=None):
    """The dense statement with the same callables on ONE device: ``u_t = pair_stack_fn(unit_fn(z, t, (0, N)), mask)`` for every slot (no
    de-duplication), ``t = cat_t u_t`` ``[..., T, N, N, c_t]``, ``z + finish_fn(t)`` (a new tensor). The tests' reference; usable at P == 1."""
    N = int(z.shape[-2])
    us = [pair_stack_fn(unit_fn(z, t, (0, N)), mask) for t in range(int(n_templ))]
    t = torch.cat(us, dim=-4)
    del us
    return z + finish_fn(t)
