"""The model-input features across the ranks of the row-sharded line (``pred --mode big --n_gpu P``, ``P > 1``; :mod:`.rowpair`).

The P ranks are separate interpreter processes folding the same item, and the feature dict the model receives must be byte-identical on
every rank — the pair rows, the replicated MSA representation and every collective of a fold assume one input. Three things make and state that:

* ONE hash seed for every rank. :func:`.rowpair.launch` hands the launching process's environment to ``opt_core.mem.rowpair.launch.run_rank_processes``,
  which starts every rank interpreter with one ``PYTHONHASHSEED`` (the launcher's own value when it names one, ``source=inherited``; else ``0``,
  ``source=default``: ``opt_core.mem.rowpair.rankdata``) and prints it once per launch on stderr:

      [esmfold2-opt] RANKENV hashseed=<v> source=default|inherited [parent='<value>'] ranks=<P>

  (``parent=`` appears only when the launching process held a ``PYTHONHASHSEED`` that is not an integer, e.g. ``random``: the value it replaced).

* RANK 0 FEATURISES, EVERY OTHER RANK RECEIVES (data form ``rank0_bcast``). :func:`.rowpair.install` binds :func:`prepare_input_rank0` as the
  builder instance's ``prepare_input`` — outermost: over the kit's feature cache and SMILES seed guard, which wrap upstream's
  ``ESMFold2InputBuilder.prepare_input`` — so on every ``prepare_input`` call of a run (upstream's ``fold`` makes one per fold, the kit's fold loop
  one more per fold for the output metadata: on rank 0 a feature-cache hit, handed over again):

  - rank 0 runs the featurisation (host work: tokenisation, CCD / SMILES conformers, MSA parsing; its str-hash-ordered and seeded steps run in ONE
    process) and hands the feature dict plus the chain records to every rank through ``rankdata.broadcast_features`` — the ranks meet at the
    family's store rendezvous first (no collective is pending while rank 0 featurises), rank 0's status word travels, then the keys / dtypes /
    shapes, then the tensors by broadcast onto each rank's device (the chain records pickled with the structure), and rank 0's RNG state after
    featurisation (python / numpy / torch CPU and CUDA generators) is set on the receivers. If rank 0's featurisation raises, its status word
    says so and every OTHER rank raises the family's ``refused: feats_rank0_failed: <Type>: <message>`` — no rank waits behind a failed rank 0 —
    while rank 0 raises the same refusal chained to its exception, except that a GPU out-of-memory error (``opt_core.oom.is_oom``) and an
    interpreter-exit signal are re-raised on rank 0 as themselves, after the other ranks were told;
  - ranks > 0 never featurise: they receive, and the family's receipt check refuses a tree that lacks an announced key or holds a tensor of another
    dtype / shape / placement (``refused: feats_bcast_malformed: <key> (…)``) — raised on the receiving rank; the other ranks are ended by the
    launcher (``rank_failed``), not by a refusal of that name;
  - every rank then takes ONE sha256 digest over the features it holds (``rankdata.digest_features``: every tensor by dtype, shape and bytes,
    every other leaf by its canonical form, keys sorted; a leaf without a canonical form is named ``nontensor_unhashed=``, never dropped) and
    ``rankdata.assert_ranks_agree(mode="refuse")`` refuses on EVERY rank when one rank's digest differs from rank 0's
    (``refused: feats_ranks_differ: rank <r> digest <hex16> …`` -> the rank's NOT ACTIVE exit -> the launcher's ``rank_failed``);
  - every rank prints, per call,

      [esmfold2-opt] rowpair feats: data_form=rank0_bcast src=0 rank=<r> call=<n> N=<tokens> keys=<k> digest=<hex16> feats_ranks_equal=yes feats_status=ok tensors=<t> gib=<g> skipped=none host_keys=none rng=carried wait_s=<s> bcast_s=<s> feat_s=<s|none> excluded=none none=none tensor_leaves=<t> nontensor_leaves=0 nontensor_unhashed=none

    (``wait_s``: this rank's seconds at the rendezvous — a receiver's is rank 0's featurisation time as seen there; ``feat_s``: rank 0's
    featurisation seconds, ``none`` on receivers); the install line carries ``data_form=rank0_bcast``, the fold line ``data_form= feats_digest=
    feats_ranks_equal= feats_excluded= feats_none= feats_tensor_leaves= feats_nontensor_leaves= feats_unhashed=``.

* The per-rank census (data form ``per_rank``): a process holding the group WITHOUT the builder binding (an ``install()`` given no builder) keeps
  every rank featurising, and at the rank forward's entry :func:`census_per_rank` gathers each rank's digest with the others'
  (``rankdata.assert_ranks_agree(mode="census")``: ``[esmfold2-opt] [feats] rank <r> digest <hex16> feats_ranks_equal=yes|no ranks=<P>
  digests=…``), naming a disagreement without refusing it; the fold line adds what the digest covered and left aside (``feats_excluded=
  feats_none= feats_tensor_leaves= feats_nontensor_leaves= feats_unhashed=``).

THE DIGEST'S VIEW (:func:`model_inputs`, both forms): the call's entries minus the fold SETTINGS, excluded by name (:data:`CALL_SETTINGS`:
``num_loops``, ``num_sampling_steps``, … — printed as ``excluded=``), and minus the optional inputs that are ``None`` (not given; printed as
``none=``); everything else enters — so the ``per_rank`` form's digest at the forward's entry and the ``rank0_bcast`` form's digest of the
handed-over features are the same number for the same input.

What differs across ranks BY DESIGN is created later, inside the fold, and is not part of any digest here: each rank's ``Layout`` rows, the
pair-RNG rows of the pair-state init (``opt_core.mem.rowpair.rng``), and the MSA module's per-loop subsample / mask draws (:mod:`.rowpair_msa`).
What stays per rank and host-side under ``rank0_bcast``: the input file's parse into a ``StructurePredictionInput`` (``inputs.build_spi``, A3M
text included) and the CCD dictionary load — cheap, deterministic reads; the featurisation built from them is rank 0's alone.

Every digest, gather, broadcast, census word and refusal is ``opt_core.mem.rowpair.rankdata``'s; this module holds the kit's call sites and
record fields.
"""
from __future__ import annotations

import functools
import sys
import time
from typing import Callable, Dict, Optional, Tuple

from opt_core.mem.rowpair import RowpairRefused, rankdata as RK
from opt_core.report import kv

from . import report as _report

__all__ = ["WHAT", "SRC", "CHAINS_KEY", "CALL_SETTINGS", "FORM_PER_RANK", "FORM_BCAST", "WORD_FORM", "WORD_EQUAL", "STATE", "model_inputs",
           "digest_inputs", "feature_digest", "prepare_input_rank0", "featurise_rank0", "census_per_rank", "fold_fields", "reset"]

WHAT = "feats"                                          # the census's subject word: `[feats] …`, `feats_ranks_equal=`, `feats_ranks_differ`, `feats_rank0_failed`
SRC = 0                                                 # the featurising rank of the rank0_bcast form
CHAINS_KEY = "chain_infos"                              # the chain records ride in the broadcast tree under this key (pickled with the structure; not a model input, not digested)
CALL_SETTINGS = ("num_loops", "num_sampling_steps", "num_diffusion_samples", "early_exit", "msa_max_depth", "msa_column_mask_rate",   # the fold call's SETTINGS — excluded from the
                 "msa_subsample_at_inference", "lm_mask_pct", "noise_scale", "step_scale", "max_inference_sigma")               # digest BY NAME (upstream fold()'s keywords to the model)
FORM_PER_RANK = RK.check_data_form("per_rank")          # data form: every rank featurises its own copy of the input (the digests are gathered and named)
FORM_BCAST = RK.check_data_form("rank0_bcast")          # data form: rank SRC featurises, every other rank receives (the digests must agree)
WORD_FORM = RK.data_form_word(FORM_PER_RANK).split("=")[0]   # `data_form` — the family's word for who featurises (install line, fold line, report)
WORD_EQUAL = RK.agree_word(WHAT, None).split("=")[0]    # `feats_ranks_equal` — the family's census key, also the fold line's field
LINE_HEAD = "rowpair feats:"                            # `[esmfold2-opt] rowpair feats: k=v …` — the rank0_bcast form's line, one per prepare_input call on every rank
N_KEY = "token_attention_mask"                          # the feature whose last extent is the token count N (the forward reads N off the same tensor)
STATE: Dict[str, object] = {"form": FORM_PER_RANK, "calls": 0, "last": None}   # this rank's feats record: the data form, prepare_input calls / censuses taken, the last record


def reset() -> None:
    STATE.update({"form": FORM_PER_RANK, "calls": 0, "last": None})


def model_inputs(d) -> Tuple[dict, Tuple[str, ...], Tuple[str, ...]]:
    """``(view, excluded, none)``: the entries of a feature / call dict the digest covers — everything but the fold SETTINGS present
    (:data:`CALL_SETTINGS`, returned as ``excluded``) and the optional inputs given as ``None`` (returned as ``none``); nothing else is left out."""
    d = {str(k): v for k, v in dict(d).items()}
    excluded = tuple(sorted(k for k in d if k in CALL_SETTINGS))
    none = tuple(sorted(k for k, v in d.items() if v is None and k not in CALL_SETTINGS))
    return {k: v for k, v in d.items() if k not in excluded and k not in none}, excluded, none


def digest_inputs(d):
    """``(FeatureDigest, excluded, none)`` of a feature / call dict: ``rankdata.digest_features`` over :func:`model_inputs`' view — every tensor
    by key path, dtype, shape and bytes (a CUDA tensor's bytes through one host copy), every other leaf by its canonical form, keys sorted; a
    leaf with no canonical form is named in ``FeatureDigest.unhashed``, never dropped."""
    view, excluded, none = model_inputs(d)
    return RK.digest_features(view), excluded, none


def feature_digest(feats) -> str:
    """sha256 hex (64 digits) of :func:`digest_inputs`."""
    return digest_inputs(feats)[0].digest


def _aside_words(fd, excluded, none) -> str:
    """``excluded=<settings|none> none=<keys|none> <FeatureDigest.words()>`` — what the digest covered and left aside, by name."""
    return kv(("excluded", _names(excluded)), ("none", _names(none))) + " " + fd.words()


def _emit(line: str) -> None:
    """A census line on this rank's stderr behind the kit's prefix."""
    print(f"{_report.PREFIX} {line}", file=sys.stderr, flush=True)


def _secs(x) -> Optional[float]:
    return None if x is None else round(float(x), 3)


# ------------------------------------------------------------------------------------------ rank 0 featurises, every other rank receives
def prepare_input_rank0(inner: Callable) -> Callable:
    """The builder's ``prepare_input`` under the line (:func:`.rowpair.install` binds it on the builder INSTANCE, outermost): ``inner`` is what
    the instance held — the kit's feature cache and SMILES seed guard over upstream's ``prepare_input`` — and runs on rank :data:`SRC` only
    (:func:`featurise_rank0`). Creating the binding makes this process's data form ``rank0_bcast``."""
    STATE["form"] = FORM_BCAST

    @functools.wraps(inner)
    def prepare_input(input, seed=None, device=None):
        return featurise_rank0(inner, input, seed=seed, device=device)
    prepare_input._esmfold2_opt_data_form = FORM_BCAST                          # the binding is recognisable on the builder
    return prepare_input


def _split(tree) -> Tuple[dict, list]:
    """``(features, chain_infos)`` out of the broadcast tree ``{<feature>: tensor, …, CHAINS_KEY: [ChainInfo, …]}``."""
    return {k: v for k, v in tree.items() if k != CHAINS_KEY}, list(tree[CHAINS_KEY])


def featurise_rank0(inner: Callable, input, seed=None, device=None):
    """ONE ``prepare_input(input, seed=, device=)`` call under the line, entered by every rank in step: rank :data:`SRC` featurises through
    ``inner`` and hands ``{**features, CHAINS_KEY: chain_infos}`` to every rank (``rankdata.broadcast_features``, one store key per call); an
    exception in rank SRC's featurisation becomes the family's ``refused: feats_rank0_failed: …`` on every rank (raised from the original on
    rank SRC; a GPU out-of-memory error or an interpreter-exit signal on rank SRC is re-raised as itself after the other ranks were told); ranks > 0
    never call ``inner``. Every
    rank then digests the feature tensors it holds and ``rankdata.assert_ranks_agree(mode="refuse")`` refuses on every rank unless all agree
    with rank SRC; one ``rowpair feats: data_form=rank0_bcast …`` line; returns ``(dict(features), chain_infos)`` as upstream's ``prepare_input`` does."""
    from opt_core.mem.rowpair import dist as RD
    P, rank = RD.world() if RD.is_dist() else (1, 0)
    call = int(STATE["calls"]) + 1                                              # the same count on every rank: the store key of this call
    STATE["calls"] = call
    key = f"{WHAT}/{call}"
    t0 = time.perf_counter()
    feat_s = None
    if rank == SRC:
        try:
            features, chain_infos = inner(input, seed=seed, device=device)
            payload = {**features, CHAINS_KEY: list(chain_infos)}             # everything rank SRC does before the hand-over is inside this try: a receiver never waits behind a silent failure
            feat_s = time.perf_counter() - t0
        except BaseException as exc:                                            # the other ranks learn it at the rendezvous (each raises the family's refusal), then rank SRC raises
            try:
                RK.broadcast_features(None, src=SRC, key=key, status=RK.status_word(exc), device=device, what=WHAT)
            except RowpairRefused as refused:
                from opt_core.oom import is_oom
                if isinstance(exc, Exception) and not is_oom(exc):
                    raise refused from exc
            raise                                                               # KeyboardInterrupt / SystemExit / GPU out-of-memory on rank SRC: as raised, after the peers were told
        fb = RK.broadcast_features(payload, src=SRC, key=key, device=device, what=WHAT)
    else:
        fb = RK.broadcast_features(None, src=SRC, key=key, device=device, what=WHAT)
    features, chain_infos = _split(fb.feats)
    fd, excluded, none = digest_inputs(features)
    identical = RK.assert_ranks_agree(fd.digest, what=WHAT, mode="refuse")     # `refused: feats_ranks_differ: …` on EVERY rank when one differs; True (None without a group: `n/a`)
    n_tok = int(features[N_KEY].shape[-1]) if hasattr(features.get(N_KEY), "shape") else None
    rec = {"form": FORM_BCAST, "src": SRC, "rank": int(rank), "P": int(P), "call": call, "N": n_tok, "keys": len(features), "digest": fd.digest,
           "equal": identical, **_aside(fd, excluded, none), "status": fb.status,
           "tensors": fb.n_tensors, "bytes": fb.nbytes, "rng": fb.rng, "wait_s": fb.wait_s, "bcast_s": fb.bcast_s, "feat_s": feat_s}
    STATE["last"] = rec
    _emit(f"{LINE_HEAD} " + kv((WORD_FORM, FORM_BCAST), ("src", SRC), ("rank", rec["rank"]), ("call", call), ("N", n_tok), ("keys", rec["keys"]),
                              ("digest", fd.digest[:RK.DIGEST_HEX])) + f" {RK.agree_word(WHAT, rec['equal'])} {fb.words()} "
          + kv(("feat_s", _secs(feat_s))) + " " + _aside_words(fd, excluded, none))
    return dict(features), chain_infos


# ----------------------------------------------------------------------------------------------------------- the per-rank census
def census_per_rank(call) -> Optional[dict]:
    """At the rank forward's entry of a process WITHOUT the builder binding (every rank, once per fold): the digest of THIS rank's model-input
    tensors gathered with every other rank's (``rankdata.assert_ranks_agree(mode="census")``: the family's ``[feats] rank <r> digest <hex16>
    feats_ranks_equal=yes|no ranks=<P> digests=…`` line, never a refusal); returns the record (also ``STATE["last"]``). No multi-rank group in
    this process: no census (None)."""
    from opt_core.mem.rowpair import dist as RD
    if not RD.is_dist():
        return None
    P, rank = RD.world()
    fd, excluded, none = digest_inputs(call)
    identical = RK.assert_ranks_agree(fd.digest, what=WHAT, mode="census", log=_emit)
    if identical is None:                                                        # a one-rank group: nothing to compare, nothing printed
        return None
    rec = {"form": FORM_PER_RANK, "rank": int(rank), "P": int(P), "call": int(STATE["calls"]) + 1, "keys": len(model_inputs(call)[0]),
           "digest": fd.digest, "equal": bool(identical), **_aside(fd, excluded, none)}
    STATE["calls"] = rec["call"]
    STATE["last"] = rec
    return rec


def fold_fields() -> dict:
    """The fold record's feats fields (the fold line prints them): the data form, the last record's digest / agreement, and what that digest
    covered and left aside by name — ``feats_excluded`` (the settings), ``feats_none`` (optional inputs not given), ``feats_tensor_leaves`` /
    ``feats_nontensor_leaves`` (what entered), ``feats_unhashed`` (entries without a canonical form); ``None`` before any record."""
    rec = STATE["last"] if isinstance(STATE["last"], dict) else None
    return {WORD_FORM: STATE["form"], "feats_digest": (rec["digest"][:RK.DIGEST_HEX] if rec else None),
            WORD_EQUAL: (RK.agree_word(WHAT, rec["equal"]).split("=", 1)[1] if rec else None),
            "feats_excluded": (_names(rec["excluded"]) if rec else None), "feats_none": (_names(rec["none"]) if rec else None),
            "feats_tensor_leaves": (rec["tensor_leaves"] if rec else None), "feats_nontensor_leaves": (_names(rec["nontensor_leaves"]) if rec else None),
            "feats_unhashed": (_names(rec["unhashed"]) if rec else None)}


def _names(xs) -> str:
    return ",".join(xs) or "none"


def _aside(fd, excluded, none) -> dict:
    """The record's account of the digest's view: the settings excluded by name, the optional inputs given as None, and the family's leaf
    census (tensor leaves that entered, non-tensor leaves that entered by path, entries without a canonical form by ``path:type``)."""
    return {"excluded": tuple(excluded), "none": tuple(none), "tensor_leaves": int(fd.tensor_leaves), "nontensor_leaves": tuple(fd.nontensor_leaves),
            "unhashed": tuple(fd.unhashed)}
