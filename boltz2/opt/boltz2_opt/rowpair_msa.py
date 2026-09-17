"""boltz2_opt.rowpair_msa — seam 4 of the tensor-parallel line (``boltz2_opt.rowpair``, ``--n_gpu P > 1``): the MSA module's INPUT
statements (``MSAModule.forward``, trunkv2.py:614-634 — the one-hot / concatenation of the raw MSA features, the per-cycle subsample draw,
the row selection) under the two PLACEMENTS of the raw MSA features the line serves. The statements are boltz's, verbatim and in boltz's
order; the placement mechanism is the core's (``opt_core.mem.rowpair.msa_host``: host copies, zero-row placeholders, the chunked device
broadcast, the census); this module holds the feature keys and the plumbing only.

    device (``ROWPAIR_MSA_HOST`` unset / ``0``)   the stock placement: ``feats["msa" | "has_deletion" | "deletion_value" | "msa_paired"]``
                                                ``[1, S, N]`` reach the device with the batch; each cycle one-hots ALL S rows
                                                (``int64 [1, S, N, 33]``: 264·S·N bytes), concatenates (``fp32 [1, S, N, 36]``: 144·S·N bytes),
                                                then draws and selects (``m[:, idx]``) when the module subsamples
    host (``=all`` | ``=rank0``)                 ``HOST_KEYS`` never reach the device whole: ``Boltz2InferenceDataModule.transfer_batch_to_device``
                                                (the worker's and Lightning's H2D of the batch) keeps them on the host — pinned host copies
                                                (``all``: every rank; ``rank0``: rank 0 only, ranks > 0 hold ZERO-ROW placeholders, no host RAM)
                                                — and every cycle the draw (the same statement, the CPU generator, ``S`` read from
                                                ``feats["msa_mask"]``, which stays on the device as stock's mask rows and as the depth every
                                                rank reads) selects rows ON THE HOST; only the selected raw rows reach the device
                                                (``rank0``: rank 0 moves them, the others receive them by the core's chunked broadcast), where
                                                the one-hot / concatenation run on those rows. Values identical to the device placement (row
                                                selection commutes with the per-row one-hot / cat: placement only, the core's EXACT class).
                                                With the subsample OFF (the worker line's ``MSAModuleArgs(subsample_msa=False)``) ALL S rows are
                                                staged to the device each cycle: the saving is then the raw features' residency outside the MSA
                                                module (≈25·S·N bytes per rank through the Pairformer, the sampler and the confidence stage),
                                                the cost one H2D (+ broadcast) of them per cycle — both named in the census.

The MSA representation's LAYOUT across the ranks (``ROWPAIR_MSA_M_LAYOUT``, :func:`m_layout`): ``token_sharded`` (the default) — rank q
embeds and updates only its token columns ``m[:, :, r0_q:r1_q, :]`` (the columns are the pair shard's rows; :func:`msa_input` one-hots and
concatenates only those columns: ``264·S·R`` + ``144·S·R`` bytes), the module's statements all-gather only what pairs a local token with every
token (``rowpair._pwa_update`` / ``_opm_add_``: one S-chunk of values at a time, the OPM's ``b`` operand) — or ``replicated`` (``m [1, S, N,
c_m]`` whole on every rank: ``1024·S·N`` bytes live in the pair-weighted averaging at ``S = 8192``). The layout changes the M (row count) of
per-(row, token) launches only; every reduction stays whole on one rank (the core's ``opt_core.mem.rowpair.msa`` class).

Census (the core's schedule facts, printed on every rank's SCHEDULE line and carried in ``tp_report.schedule``): ``msa_m=token_sharded|
replicated`` ``msa_cols=<c0>:<c1>/<N>|all`` (this seam's), ``msa_host_mode`` /
``msa_host_pinned_gib`` / ``msa_host_pageable_gib`` (the core's, at parking), ``msa_host_rows`` (the core's, per staged tensor), and this
seam's summary per cycle: ``msa_host=rank0|all|off`` ``msa_subsample=1|0`` ``msa_rows=<rows used>/<S>`` ``msa_host_cycle_gib=<GiB staged
host->device on this rank this cycle>`` ``msa_host_cycles=<n>``. Refused BY NAME: a host word set while the raw features stand on the
device at the MSA module (the batch did not pass through the placement hook), an ``msa_mask`` that differs across ranks (proven once per batch
before the first staging: ``trunk.guard_replicated``), an unknown word (the core's ``host_mode``).

The batch source at ``n_gpu > 1`` (:func:`_predict_dataloader` over ``Boltz2InferenceDataModule.predict_dataloader``, ``data_form=rank0_bcast``):
rank 0 alone runs boltz's featurizer — its stock ``DataLoader``, iterated as stock iterates it — and every batch it yields goes to every other
rank (the core's ``rankdata.broadcast_features``: the ranks > 0 wait at the store while rank 0 featurizes, then receive the tensors by broadcast
and the plain entries pickled; a featurizer exception on rank 0 is ``refused: feats_rank0_failed: …`` on EVERY rank); a rank > 0 builds no
``DataLoader``, featurizes nothing and yields the received batches (:class:`_ReceivedBatches`). The placement hook then digests the batch on
every rank (``rankdata.digest_features``: sha256 over dtype, shape and bytes of every feature but ``NON_FEATURE_KEYS``) and the ranks' digests must
agree: one ``<tag> [feats] rank <r> digest <hex16> feats_ranks_equal=yes ranks=<P> digests=<hex16>,…`` line per batch on every rank
(:func:`feats_census`), ``refused: feats_ranks_differ: …`` on every rank otherwise; ``data_form=rank0_bcast entry_inputs_equal=<equal>/<checked>``
and the broadcast's facts (``feats_bcast_gib`` …) on the LEVER line.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

DATAMODULE = "boltz.data.module.inferencev2"          # the module whose Boltz2InferenceDataModule.transfer_batch_to_device moves the batch (inferencev2.py:398-433)
DATAMODULE_CLASS = "Boltz2InferenceDataModule"
HOST_KEYS: Tuple[str, ...] = ("msa", "has_deletion", "deletion_value", "msa_paired")   # the raw MSA features [1, S, N] (featurizerv2.py:1653-1657); row dim 1 = the MSA row the subsample indexes
ROW_DIM = 1
DEPTH_KEY = "msa_mask"                                 # [1, S, N], stays on the device: the mask rows the module uses (trunkv2.py:620, 634) and the depth S every rank reads for the draw
M_LAYOUT_ENV = "ROWPAIR_MSA_M_LAYOUT"                  # the MSA representation's layout across the ranks (modes.TP_EXPORTS): token_sharded (default, also when unset) | replicated
M_LAYOUTS: Tuple[str, ...] = ("token_sharded", "replicated")     # the core's words (opt_core.mem.rowpair.msa.M_LAYOUTS), the default first; m_layout checks the two agree
COL_DIM = 2                                            # the token dim of the raw MSA features [1, S, N]: the columns a token-sharded rank embeds
FEATS_WHAT = "feats"                                    # the `what` of the core's cross-rank digest census and feature broadcast (opt_core.mem.rowpair.rankdata): `[feats] rank <r> digest <hex16> feats_ranks_equal=yes|no ranks=<P> digests=…`; store keys feats/<n>
FEATS_FORM = "rank0_bcast"                              # the LEVER line's data_form= (the core's DATA_FORMS word): rank 0 runs boltz's featurizer, every other rank receives its batch
NON_FEATURE_KEYS: Tuple[str, ...] = ("record",)         # batch entries outside the digest, by name: the manifest records (the writer's bookkeeping, not model input)
SOURCE_ATTR = "predict_dataloader"                      # the data module's batch source (inferencev2.py:369-396): rebound at n_gpu > 1 so rank 0 alone featurizes (install_batch_transfer binds it with the H2D)
SOURCE_ONCE = "feats source iterated twice (the rank0_bcast batch source serves the worker's in-process predict loop, --pipeline 1)"   # refused: a second iter() of one source object
SOURCE_VERB = "rowpair feats source:"                    # the per-batch transport line on every rank: `<tag> rowpair feats source: role=featurize|receive rank=<r> data_form=rank0_bcast <the core's FeatureBroadcast words> msa_keys_gib=<gib>`

_STATE: Dict[str, Any] = {"orig": None, "orig_source": None, "owner": None, "armed": False, "parked": None, "cycles": 0, "guarded": None, "feats": None,
                          "source": None, "batches": 0}      # source: the facts of the last batch this rank featurized-and-sent / received (data_form rank0_bcast); batches: numbers the store key feats/<n>


class Refused(RuntimeError):
    """A placement this seam refuses by name."""


def _core():
    from opt_core.mem.rowpair import msa_host as MH, evidence as EV, trunk as TRK
    return MH, EV, TRK


def host_mode() -> Optional[str]:
    """The placement word in force in this process: None (device) | ``"all"`` | ``"rank0"`` (the core's ``msa_host.host_mode``)."""
    return _core()[0].host_mode()


def m_layout() -> str:
    """``ROWPAIR_MSA_M_LAYOUT``: the MSA representation's layout under n_gpu > 1 — ``token_sharded`` (the default, also when the word is unset:
    rank q embeds and updates ``m[:, :, r0_q:r1_q, :]``) | ``replicated`` (``m [1, S, N, c_m]`` whole on every rank). Any other word is
    refused by name."""
    from opt_core.mem.rowpair.msa import M_LAYOUTS as CORE_M_LAYOUTS
    if set(CORE_M_LAYOUTS) != set(M_LAYOUTS):
        raise Refused(f"refused: {M_LAYOUT_ENV}: the kit's layouts {M_LAYOUTS} and the core's {tuple(CORE_M_LAYOUTS)} disagree")
    v = (os.environ.get(M_LAYOUT_ENV) or "token_sharded").strip()
    if v not in M_LAYOUTS:
        raise Refused(f"refused: {M_LAYOUT_ENV}={v!r}: one of {'|'.join(M_LAYOUTS)}")
    return v


# ----------------------------------------------------------------------------------------------------------------- the input-feature census (n_gpu > 1)
def feats_census(batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """At n_gpu > 1 (a live rank group) and only there: the digest of THIS rank's model-input features — the batch as the H2D receives it
    (rank 0: from its featurizer; ranks > 0: from rank 0), every entry but ``NON_FEATURE_KEYS``, before any placement of this line (nothing in
    it is per-rank yet: the raw MSA keys become host copies / zero-row placeholders in the hook below, the token-pair planes row slabs and z
    row shards in the trunk) — compared with every rank's: the core's census line ``<tag> [feats] rank <r> digest <hex16>
    feats_ranks_equal=yes|no ranks=<P> digests=<hex16>,…`` on every rank (``rankdata.assert_ranks_agree(mode="census")``), then digests that
    differ are REFUSED on every rank (``rankdata.differ_refusal``: ``refused: feats_ranks_differ: …`` — every rank holds the census answer, so
    every rank raises; none is left in a collective). Feeds the LEVER line's ``entry_inputs_equal=<equal>/<checked>`` (``rowpair`` guards
    ``inputs_checked`` / ``inputs_equal``) and ``report()["feats"]``. Returns the facts; None without a group (n_gpu = 1: nothing hashed or printed)."""
    from opt_core.mem.rowpair import dist as D, rankdata as RD
    if not D.is_dist():
        return None
    from . import rowpair as RP                                             # the line's guards and its rank-transcript writer (rowpair imports this module at load: imported here)
    fd = RD.digest_features(batch, exclude=NON_FEATURE_KEYS)               # tensors by dtype / shape / bytes, plain values canonically, anything else excluded BY NAME (fd.unhashed)
    equal = RD.assert_ranks_agree(fd.digest, what=FEATS_WHAT, mode="census", log=lambda text: RP._say(f"{RP.TAG} {text}"))
    RP._STATE["guards"]["inputs_checked"] += 1
    RP._STATE["guards"]["inputs_equal"] += int(bool(equal))
    facts = {"form": FEATS_FORM, "rank": int(D.comm().rank), "digest": fd.digest, "equal": bool(equal), "excluded": list(NON_FEATURE_KEYS),
             "tensor_leaves": int(fd.tensor_leaves), "nontensor_leaves": list(fd.nontensor_leaves), "unhashed": list(fd.unhashed),
             "batches": int(RP._STATE["guards"]["inputs_checked"])}
    _STATE["feats"] = facts
    if not equal:
        raise RD.differ_refusal(FEATS_WHAT, int(D.comm().rank), fd.digest)
    return facts


# ----------------------------------------------------------------------------------------------------------------- the batch source (n_gpu > 1): rank 0 featurizes, every other rank receives
def _feats_key() -> str:
    """The store key of this process's next feature broadcast (``feats/<n>``): one per batch, the same n on every rank."""
    _STATE["batches"] += 1
    return f"{FEATS_WHAT}/{_STATE['batches']}"


def source_bytes(batch: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, Any]:
    """``{key: {"dtype", "shape", "bytes"}}`` of the ``keys`` present in ``batch`` as tensors — the sizes a broadcast of those entries moves."""
    import torch
    return {k: {"dtype": str(v.dtype).replace("torch.", ""), "shape": list(v.shape), "bytes": int(v.numel()) * int(v.element_size())}
            for k, v in batch.items() if k in keys and isinstance(v, torch.Tensor)}


def _source_facts(role: str, fb, batch) -> Dict[str, Any]:
    """The transport facts of one batch on this rank (``report()["source"]``) and its line: ``<tag> rowpair feats source: role=featurize|receive
    rank=<r> data_form=rank0_bcast feats_status=ok tensors=<n> gib=<g> skipped=… host_keys=… rng=carried|off|none wait_s=<s> bcast_s=<s>
    msa_keys_gib=<g>`` (the core's ``FeatureBroadcast.words`` between the kit's)."""
    from opt_core.mem.rowpair import dist as D, rankdata as RD
    from opt_core.report import kv
    from . import rowpair as RP
    msa_keys = source_bytes(batch, tuple(HOST_KEYS) + (DEPTH_KEY,)) if isinstance(batch, dict) else {}
    facts = {"form": FEATS_FORM, "role": role, "rank": int(D.comm().rank), "batches": int(_STATE["batches"]), "status": str(fb.status).split()[0],
             "tensors": int(fb.n_tensors), "gib": round(fb.nbytes / 2 ** 30, 3), "bytes": int(fb.nbytes), "skipped": sorted(fb.skipped), "host_keys": list(fb.host),
             "rng": fb.rng, "wait_s": round(float(fb.wait_s), 1), "bcast_s": round(float(fb.bcast_s), 1), "words": fb.words(),
             "msa_keys": msa_keys, "msa_keys_bytes": int(sum(v["bytes"] for v in msa_keys.values()))}
    _STATE["source"] = facts
    RP._say(f"{RP.TAG} {SOURCE_VERB} " + kv(role=role, rank=facts["rank"]) + f" {RD.data_form_word(FEATS_FORM)} {fb.words()} " + kv(msa_keys_gib=round(facts["msa_keys_bytes"] / 2 ** 30, 3)))
    return facts


def _iterate_once(source) -> None:
    """A source object serves ONE pass (the worker's in-process predict loop iterates it once per input): a second ``iter()`` — a caller that
    re-iterates the loader, whose second pass would draw a second ``base_seed`` on rank 0 only — is refused by name (``SOURCE_ONCE``)."""
    from opt_core.mem.rowpair import RowpairRefused
    if source.iterated:
        raise RowpairRefused(f"refused: {SOURCE_ONCE}", lever="n_gpu")
    source.iterated = True


# ----------------------------------------------------------------------------------------------------------------- the batch's diet (rank 0, before the broadcast)
FEATS_DIET_ENV = "ROWPAIR_FEATS_DIET"     # =0: the batch travels as boltz's featurizer built it; unset/1 (default at n_gpu > 1): FEATS_DIET below
FEATS_DIET_DROP = ("disto_target", "r_set_to_rep_atom", "token_to_center_atom")
                                          # keys NO inference statement reads (audited over boltz 2.2.1 model/ + data/write + this kit: disto_target [B,N,N,1,64] f32 = 256·N² B is the
                                          # distogram LOSS target; r_set_to_rep_atom / token_to_center_atom [B,N,A] i64 one-hot = 8·A·N B each feed losses only) — dropped from the
                                          # batch on rank 0, so they are neither broadcast nor moved to any rank's device (a reader would fail LOUDLY by KeyError, never silently)
FEATS_DIET_UINT8 = ("atom_to_token", "token_to_rep_atom")
                                          # {0,1} one-hot int64 maps [B,A,N] / [B,N,A] (8·A·N B each, REPLICATED on every rank for the whole item: 2 x 1.52 GB at N=5060) whose every
                                          # reader casts first (`.float()`: encodersv2 / diffusionv2 / confidencev2 / confidence_utils / this kit's heads and atom lever) or reduces
                                          # (`.sum(1)`: confidencev2:366 — torch sums uint8 into int64): carried as uint8, the same values -> the same float copies, 8x fewer bytes


def feats_diet_(batch) -> Dict[str, Any]:
    """Shrink one featurized batch IN PLACE on rank 0 before it is broadcast (``_Rank0Iter``): drop :data:`FEATS_DIET_DROP`, carry
    :data:`FEATS_DIET_UINT8` as uint8 when the tensor is an integer/bool tensor whose values are all 0 or 1 (anything else stays as it is, named).
    Every rank receives the dieted batch, so the census is rank 0's: schedule words ``feats_diet=on|off feats_dropped=<keys> feats_uint8=<keys>
    feats_diet_saved_gib=<g>`` (bytes that no longer travel / land per rank). Returns the facts dict."""
    from opt_core.mem.rowpair import evidence as EV
    facts: Dict[str, Any] = {"on": False, "dropped": [], "uint8": [], "kept": [], "saved_bytes": 0}
    word = (os.environ.get(FEATS_DIET_ENV, "") or "1").strip().lower()
    if word in ("0", "off") or not isinstance(batch, dict):
        EV.record_schedule(feats_diet="off")
        _STATE["diet"] = facts
        return facts
    import torch
    facts["on"] = True
    for k in FEATS_DIET_DROP:
        t = batch.get(k)
        if isinstance(t, torch.Tensor):
            facts["saved_bytes"] += int(t.numel()) * int(t.element_size())
            del batch[k]
            facts["dropped"].append(k)
    for k in FEATS_DIET_UINT8:
        t = batch.get(k)
        if not isinstance(t, torch.Tensor) or t.dtype == torch.uint8:
            continue
        ok = (not t.dtype.is_floating_point) and (t.numel() == 0 or (int(t.min()) >= 0 and int(t.max()) <= 1))
        if not ok:                                                          # not a {0,1} integer map: travels as it is, named
            facts["kept"].append(f"{k}:{str(t.dtype).replace('torch.', '')}")
            continue
        u = t.to(torch.uint8)
        facts["saved_bytes"] += int(t.numel()) * (int(t.element_size()) - 1)
        batch[k] = u.pin_memory() if (t.is_pinned() if hasattr(t, "is_pinned") else False) else u
        facts["uint8"].append(k)
    EV.record_schedule(feats_diet="on", feats_dropped=",".join(facts["dropped"]) or "-", feats_uint8=",".join(facts["uint8"]) or "-",
                       feats_diet_saved_gib=round(facts["saved_bytes"] / 2 ** 30, 3), **({"feats_diet_kept": ",".join(facts["kept"])} if facts["kept"] else {}))
    _STATE["diet"] = facts
    return facts


class _Rank0Batches:
    """Rank 0's batch source: stock's ``DataLoader``, iterated exactly as the caller iterates it (``iter()`` is the DataLoader's own — its
    ``base_seed`` draw included), every batch it yields sent to the other ranks before the caller sees it (``rankdata.broadcast_features``; a
    featurizer exception is sent as the ``failed:`` status first, so every rank raises ``refused: feats_rank0_failed`` — this rank's original
    exception chained under it — and none waits)."""

    def __init__(self, loader):
        self.loader = loader
        self.iterated = False

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        _iterate_once(self)
        return _Rank0Iter(iter(self.loader))


class _Rank0Iter:
    def __init__(self, it):
        self.it = it

    def __iter__(self):
        return self

    def __next__(self):
        from opt_core.mem.rowpair import rankdata as RD
        try:
            batch = next(self.it)
        except StopIteration:
            raise                                                            # the end of rank 0's loader: every rank's source ends after the manifest's records, no message needed
        except BaseException as e:                                           # the featurizer raised on rank 0: every rank learns it at the store and raises (the core's word), this one too
            RD.broadcast_features(None, src=0, key=_feats_key(), status=RD.status_word(e), what=FEATS_WHAT)   # raises refused: feats_rank0_failed here as on every rank, `e` chained under it
            raise                                                            # not reached (the line above raises); kept so no path returns without a batch
        feats_diet_(batch)                                                    # the loss target and the unread / one-hot atom maps leave or shrink BEFORE they travel (FEATS_DIET)
        fb = RD.broadcast_features(batch, src=0, key=_feats_key(), what=FEATS_WHAT)
        _source_facts("featurize", fb, batch)
        return batch


class _ReceivedBatches:
    """A rank > 0's batch source: no ``DataLoader``, no featurizer, no loader worker processes — ``len(records)`` batches received from rank 0
    (``rankdata.broadcast_features``: this rank waits at the store while rank 0 featurizes, then receives; rank 0's host RNG state after its
    featurization comes along and is set here). ``iter()`` makes the one CPU-generator draw ``iter(DataLoader)`` makes on rank 0 (its
    ``base_seed``), so the ranks' host RNG streams stay in step statement for statement."""

    def __init__(self, n_batches: int):
        self.n = int(n_batches)
        self.iterated = False

    def __len__(self):
        return self.n

    def __iter__(self):
        _iterate_once(self)
        import torch
        torch.empty((), dtype=torch.int64).random_()                      # torch's _BaseDataLoaderIter: base_seed = one int64 draw from the default CPU generator
        return _ReceivedIter(self.n)


class _ReceivedIter:
    def __init__(self, n: int):
        self.left = int(n)

    def __iter__(self):
        return self

    def __next__(self):
        if self.left <= 0:
            raise StopIteration
        self.left -= 1
        from opt_core.mem.rowpair import rankdata as RD
        fb = RD.broadcast_features(None, src=0, key=_feats_key(), what=FEATS_WHAT)
        _source_facts("receive", fb, fb.feats)
        return fb.feats


def _predict_dataloader(self):
    """``Boltz2InferenceDataModule.predict_dataloader`` at n_gpu > 1: on rank 0 stock's ``DataLoader`` wrapped so every batch also goes to the
    other ranks (:class:`_Rank0Batches`); on a rank > 0 a source of ``len(manifest.records)`` received batches (:class:`_ReceivedBatches`) —
    that rank featurizes nothing. The source serves the worker's in-process predict loop (``--pipeline 1``): ONE ``iter()`` per source object,
    ``next()`` until it ends; a second ``iter()`` is refused by name. Stock's own method at n_gpu = 1; a rank of a ×P launch without its rank
    group is refused by name (never a per-rank featurization in silence)."""
    from opt_core.mem.rowpair import RowpairRefused, dist as D, launch
    orig = _STATE["orig_source"]
    if not D.is_dist():
        if int(launch.world_size()) > 1:
            raise RowpairRefused(f"refused: rank {launch.rank()} of {launch.world_size()} has no rank group for the {FEATS_FORM} batch source (the ×P line attaches before the data module is built)", lever="n_gpu")
        return orig(self)
    if int(D.comm().rank) == 0:
        return _Rank0Batches(orig(self))
    return _ReceivedBatches(len(self.manifest.records))


# ----------------------------------------------------------------------------------------------------------------- the batch's H2D
def _transfer_batch_to_device(self, batch, device, dataloader_idx):
    """``Boltz2InferenceDataModule.transfer_batch_to_device`` with ``HOST_KEYS`` kept on the host under a host word (the core's
    ``park_features``: pinned copies / zero-row placeholders); every other key moves as stock moves it. A pass-through without a host word.
    At n_gpu > 1 the input-feature census (:func:`feats_census`) reads the batch first, as it arrives — under either placement."""
    MH, EV, _ = _core()
    md = MH.host_mode()
    orig = _STATE["orig"]
    feats_census(batch)                                                     # n_gpu > 1: this rank's feature digest against every rank's (the core's census line, the entry_inputs_equal counters); nothing at n_gpu = 1
    if md is None:
        return orig(self, batch, device, dataloader_idx)
    held = {k: batch.pop(k) for k in HOST_KEYS if k in batch}
    batch = orig(self, batch, device, dataloader_idx)
    facts = MH.park_features(held, HOST_KEYS, mode=md, row_dims=ROW_DIM)
    batch.update(held)
    _STATE["parked"] = {"mode": facts["mode"], "parked": list(facts["parked"]), "placeholders": list(facts["placeholders"]),
                        "where": dict(facts["where"]), "rows": dict(facts["rows"]),      # the core's words per key (host_pinned | host_pageable:<kind> | host | placeholder) and parked row counts
                        "gib": round(sum(int(v.numel()) * int(v.element_size()) for v in held.values() if hasattr(v, "numel")) / 2 ** 30, 3)}
    EV.record_schedule(msa_host=md, msa_host_keys=",".join(k for k in HOST_KEYS if k in held))
    return batch


def install_batch_transfer() -> bool:
    """Rebind ``Boltz2InferenceDataModule.transfer_batch_to_device`` (the H2D: placement + census) and ``.predict_dataloader`` (the batch
    source: rank 0 featurizes, the other ranks receive) — idempotent; the originals kept for ``reset_for_tests``. The data module is imported
    before the model module on the worker route (bz_worker_lev.py:57-59), so the class exists when ``rowpair`` installs; when it does not (a
    process that never imports it), the rebind rides its import. Returns True when bound now, False when armed."""
    mod = sys.modules.get(DATAMODULE)
    if mod is not None:
        _bind(mod)
        return True
    if not _STATE["armed"]:
        from boltz2_opt.worker_launch import _AfterImport
        sys.meta_path.insert(0, _AfterImport(DATAMODULE, _bind))
        _STATE["armed"] = True
    return False


def _bind(mod) -> None:
    cls = getattr(mod, DATAMODULE_CLASS, None)
    if cls is None:
        raise Refused(f"refused: {DATAMODULE} carries no {DATAMODULE_CLASS} (boltz {DATAMODULE} changed; this seam binds boltz 2.2.1's)")
    for attr in ("transfer_batch_to_device", SOURCE_ATTR):
        if not hasattr(cls, attr):
            raise Refused(f"refused: {DATAMODULE}.{DATAMODULE_CLASS} carries no {attr} (boltz {DATAMODULE} changed; this seam binds boltz 2.2.1's)")
    if _STATE["orig"] is not None:
        return
    from opt_core.mem.rowpair.rankdata import check_data_form
    check_data_form(FEATS_FORM)                                             # the kit's word is one of the core's DATA_FORMS (refused by name otherwise)
    _STATE["orig"], _STATE["orig_source"], _STATE["owner"] = cls.transfer_batch_to_device, getattr(cls, SOURCE_ATTR), cls
    cls.transfer_batch_to_device = _transfer_batch_to_device
    setattr(cls, SOURCE_ATTR, _predict_dataloader)


def reset_for_tests() -> None:
    if _STATE["orig"] is not None and _STATE["owner"] is not None:
        _STATE["owner"].transfer_batch_to_device = _STATE["orig"]
        setattr(_STATE["owner"], SOURCE_ATTR, _STATE["orig_source"])
    _STATE.update(orig=None, orig_source=None, owner=None, parked=None, cycles=0, guarded=None, feats=None, source=None, batches=0)


# ----------------------------------------------------------------------------------------------------------------- seam 4's input statements
def msa_input(msa_module, feats: Dict[str, Any], device, guards: Optional[dict] = None, cols: Optional[Tuple[int, int]] = None):
    """trunkv2.py:614-634 — ``(m, msa_mask)``: the MSA module's fp32 input rows ``cat([one_hot(msa), has_deletion, deletion_value(, msa_paired)])``
    AFTER the per-cycle selection, and the mask rows. The draw is boltz's statement (``torch.randperm(S)[:num_subsampled_msa]``, the CPU
    default generator, :632) proven identical on every rank by the core (``trunk.guard_replicated``); ``S`` = ``msa.shape[1]`` on the device
    placement (stock's operand) = ``feats["msa_mask"].shape[1]`` on the host placement (the same number: featurizerv2.py:1620
    ``msa_mask = ones_like(msa)``). Device placement: stock's order (one-hot and cat over all rows, then select). Host placement: select on the
    host copies (the core's ``HostTensor.rows_to`` inside ``rows_to_device``), then one-hot and cat over the selected rows on the device.
    ``cols = (c0, c1)`` (the token-sharded layout, :func:`m_layout`): only token columns ``[c0, c1)`` of the raw features are one-hot /
    concatenated — ``m`` is this rank's column block ``[1, rows, c1-c0, 36]`` (column selection commutes with the per-(row, token) one-hot / cat
    and with the row selection: the values are the stock statement's); ``msa_mask`` is returned WHOLE (``[1, rows, N]``: the outer-product
    mean's pair count reads every column)."""
    import torch
    from boltz.data import const
    MH, EV, TRK = _core()
    md = MH.host_mode()
    msa_mask = feats[DEPTH_KEY]
    N = int(msa_mask.shape[COL_DIM])
    c0, c1 = (0, N) if cols is None else (int(cols[0]), int(cols[1]))
    if not (0 <= c0 < c1 <= N):
        raise Refused(f"refused: msa_input cols={cols}: not a column block of [0, {N})")
    col = (lambda x: x) if cols is None else (lambda x: x.narrow(COL_DIM, c0, c1 - c0))   # noqa: E731 — the token columns this rank embeds
    cols_word = "all" if cols is None else f"{c0}:{c1}/{N}"
    if md is None:                                                          # ---- the stock placement, stock order (rowpair's reference statement)
        msa = col(feats["msa"])
        msa = torch.nn.functional.one_hot(msa, num_classes=const.num_tokens)
        has_deletion = col(feats["has_deletion"]).unsqueeze(-1)
        deletion_value = col(feats["deletion_value"]).unsqueeze(-1)
        is_paired = col(feats["msa_paired"]).unsqueeze(-1)
        if msa_module.use_paired_feature:
            m = torch.cat([msa, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m = torch.cat([msa, has_deletion, deletion_value], dim=-1)
        S = int(msa.shape[1])
        idx = _draw(msa_module, S, device, guards, TRK)
        if idx is not None:
            m = m[:, idx]
            msa_mask = msa_mask[:, idx]
        EV.record_schedule(msa_host="off", msa_subsample=int(idx is not None), msa_rows=f"{int(m.shape[1])}/{S}", msa_cols=cols_word)
        return m, msa_mask
    # ---- host placement: the raw features are host copies (or zero-row placeholders on ranks > 0 under rank0)
    for k in HOST_KEYS:
        v = feats.get(k)
        if torch.is_tensor(v) and v.is_cuda:
            raise Refused(f"refused: ROWPAIR_MSA_HOST={md} but feats[{k!r}] stands on {v.device} at the MSA module — the batch did not pass through "
                          f"{DATAMODULE_CLASS}.transfer_batch_to_device (the placement hook of this line); unset the word or route the batch through it")
    S = int(msa_mask.shape[1])                                              # the depth, replicated on the device on every rank —
    B = int(msa_mask.shape[0])
    if _STATE.get("guarded") is not msa_mask:                               # PROVEN once per batch before the first staging: every rank declares (B, rows, N) of ITS mask to the
        TRK.guard_replicated(msa_mask, "msa_mask")                          # broadcast, so a featurisation that differs across ranks is refused by name here, never a size fault / hang there
        _STATE["guarded"] = msa_mask
        if guards is not None:
            guards["replicated_checked"] = int(guards.get("replicated_checked", 0)) + 1
    idx = _draw(msa_module, S, device, guards, TRK)
    n = S if idx is None else int(idx.numel())
    keys = HOST_KEYS if msa_module.use_paired_feature else HOST_KEYS[:3]
    staged = 0
    rows = {}
    for k in keys:
        def build(k=k):
            ht = MH.HostTensor(feats[k], name=k, row_dim=ROW_DIM)
            return ht.to(device).contiguous() if idx is None else ht.rows_to(idx, device)   # all rows (no subsample) | the selected rows, gathered on the host
        rows[k] = MH.rows_to_device(build, shape=(B, n, N), dtype=feats[k].dtype, device=device, mode=md, name=k)
        staged += int(rows[k].numel()) * int(rows[k].element_size())
    msa = torch.nn.functional.one_hot(col(rows["msa"]), num_classes=const.num_tokens)   # :616 on the selected rows (this rank's columns under the token-sharded layout)
    parts = [msa, col(rows["has_deletion"]).unsqueeze(-1), col(rows["deletion_value"]).unsqueeze(-1)]
    if msa_module.use_paired_feature:
        parts.append(col(rows["msa_paired"]).unsqueeze(-1))
    m = torch.cat(parts, dim=-1)                                            # :624-628
    del rows, parts, msa
    if idx is not None:
        msa_mask = msa_mask[:, idx]                                         # :634 (device rows of the replicated mask)
    _STATE["cycles"] += 1
    EV.record_schedule(msa_host=md, msa_subsample=int(idx is not None), msa_rows=f"{n}/{S}", msa_host_cycle_gib=round(staged / 2 ** 30, 3),
                       msa_host_cycles=int(_STATE["cycles"]), msa_cols=cols_word)
    return m, msa_mask


def _draw(msa_module, S: int, device, guards: Optional[dict], TRK):
    """trunkv2.py:631-632: the subsample indices (CPU generator; every rank's generator state is rank 0's — rowpair._generators_from_rank0),
    proven identical across ranks; None when the module does not subsample."""
    import torch
    if not msa_module.subsample_msa:
        return None
    idx = torch.randperm(S)[: msa_module.num_subsampled_msa]
    TRK.guard_replicated(idx.to(device), "msa_indices")                    # the draw itself: proven, never assumed (one all-reduce of checksums; mismatch refused by name)
    if guards is not None:
        guards["replicated_checked"] = int(guards.get("replicated_checked", 0)) + 1
    return idx


def report() -> Dict[str, Any]:
    """This seam's facts for ``rowpair.report()``: the word in force, whether the batch hook is bound, what the last batch parked, cycles served,
    the last batch's input-feature census (``feats``: form / rank / digest / equal / what entered the digest, None at n_gpu = 1) and source
    (``source``: role featurize|receive, the broadcast's tensors / gib / wait_s / bcast_s / rng word, the MSA keys' sizes; None at n_gpu = 1)."""
    return {"mode": host_mode() or "off", "hook_bound": _STATE["orig"] is not None, "hook_armed": bool(_STATE["armed"]), "parked": _STATE["parked"],
            "cycles": int(_STATE["cycles"]), "host_keys": list(HOST_KEYS), "depth_key": DEPTH_KEY, "feats": _STATE["feats"], "source": _STATE["source"]}
