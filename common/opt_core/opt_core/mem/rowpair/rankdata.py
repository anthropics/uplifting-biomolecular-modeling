"""The rank processes' input contract of a row-sharded run: every rank of one launch hands the model BYTE-IDENTICAL features. Two halves:

  launcher side (standard library only — the process that SPAWNS the P rank interpreters; nothing here imports torch):
    hash_seed(parent_env=None) -> (value, source)      the ONE ``PYTHONHASHSEED`` every rank interpreter of the launch starts with: the
                                                       launcher's own value when it names a seed (``source=inherited``), else
                                                       :data:`HASHSEED_DEFAULT` = ``"0"`` (``source=default``; an unset, empty or ``random``
                                                       value names no seed the ranks could share)
    hashseed_word(parent_env, n_ranks) -> str          the launch line's census word ``hashseed=<v> source=default|inherited [parent=<repr>] ranks=<P>``
                                                       (``parent=`` names a launcher value that was present but invalid — ``random``, garbage —
                                                       so the discarded value is visible; never a refusal)
    hashseed_fields(parent_env=None) -> dict           the run record's fields ``{"hashseed": <v>, "hashseed_source": <source>}``
    ranks_env(parent_env, n_ranks) -> (env, word)      ``dict(parent_env)`` with that seed exported, and the census word — the BASE
                                                       environment every rank's own environment is layered on (:func:`launch.rank_env`, hence
                                                       :func:`launch.run_rank_processes`, layers on it; :func:`launch.run_sharded` exports the
                                                       seed into the launching process's environment before its ``spawn`` workers start)
  rank side (torch, reached lazily; the comm surface :func:`dist.comm`):
    digest_features(feats, exclude=()) -> FeatureDigest   the identity of what a rank hands the model: sha256 over the feature dict minus
                                                       the TOP-LEVEL keys named in ``exclude`` (entries that differ across ranks BY DESIGN —
                                                       host-parked rows one rank holds; the caller prints what it excluded), keys sorted at
                                                       every level. TENSORS ARE THE CONTRACT: every torch tensor / numpy array leaf enters as
                                                       ``path|dtype|shape|`` + its bytes. Non-tensor leaves never enter by ``repr()``: an atom
                                                       array (biotite ``AtomArray``-like: ``get_annotation_categories`` / ``get_annotation`` /
                                                       ``coord``) enters as its CANONICAL summary — the sorted annotation category names, each
                                                       annotation array's dtype / shape / bytes in that order, the coordinates' and the box's
                                                       dtype / shape / bytes; bonds are not part of the summary — so the order in which
                                                       annotations were attached is not identity;
                                                       ``str / int / float / bool / None`` leaves (and numpy scalars) enter as canonical JSON;
                                                       ``bytes`` by length + content; any OTHER object is EXCLUDED BY NAME (``unhashed``:
                                                       ``<path>:<type>``), never silently. ``FeatureDigest`` = ``(digest, tensor_leaves,
                                                       nontensor_leaves, unhashed)`` + ``.words()`` = ``tensor_leaves=<n> nontensor_leaves=<k>:<paths>
                                                       nontensor_unhashed=<path:type,…|none>`` (the gate line's census words)
    feature_digest(feats, exclude=()) -> str           ``digest_features(...).digest`` (64 hex digits)
    ranks_agree(digest, comm=None) -> Optional[bool]   every rank's digest (its first :data:`DIGEST_HEX` hex digits) equals rank 0's: one
                                                       broadcast + one max all-reduce, so the answer is the SAME bool on every rank; ``None``
                                                       without a multi-rank group (P == 1 / no comm) — nothing to compare
    gather_digests(digest, comm=None) -> Optional[list]  every rank's ``digest[:DIGEST_HEX]``, index = rank (one object all-gather); ``None``
                                                       without a group — the census form's listing
    agree_word(what, identical) -> str                 ``<what>_ranks_equal=yes|no|n/a`` (the census word)
    differ_refusal(what, rank, digest) -> RowpairRefused   the one refusal for differing ranks, ``refused: <what>_ranks_differ: rank <r> digest <d16> — …``
    assert_ranks_agree(digest, what="feats", mode="refuse"|"census", comm=None, log=None) -> Optional[bool]
                                                       ``refuse``: :func:`ranks_agree`, raising :func:`differ_refusal` on ``False`` on every
                                                       rank; ``census``: :func:`gather_digests`, one line ``[<what>] rank <r> digest <d16>
                                                       <what>_ranks_equal=yes|no ranks=<P> digests=<d16,…>`` through ``log`` (default the
                                                       comm's log), never a refusal; both return the bool (``None`` without a group)
  the launch line (launcher side): rankenv_line(tag, parent_env, n_ranks) -> ``[<tag>] RANKENV hashseed=<v> source=… ranks=<P>`` — printed
                                                       once per P > 1 launch by :func:`launch.run_rank_processes` / :func:`launch.run_sharded`
                                                       (tag = ``ROWPAIR_TAG``, default ``rowpair``); a kit with its own launcher prints
                                                       :func:`hashseed_word` on its own launch line

  rank-0 featurisation (the ``rank0_bcast`` data form; the other form, every rank featurising, is ``per_rank``):
    broadcast_features(feats=None, *, comm=None, src=0, key="feats/0", status=None, skip_keys=(), host_keys=(), device=None,
                       carry_rng=True, wait_timeout_s=None, what="feats", pin_host=True) -> FeatureBroadcast
        ONE rank featurises; every rank returns holding the same feature tree. Protocol, in order — (i) RENDEZVOUS OUTSIDE ANY COLLECTIVE:
        rank ``src`` stores ``status`` (``status_word``: ``ok[ <extra>]`` | ``failed:<Type>:<msg>`` | any caller control word) under
        ``key`` in the group's store (``Comm.host_signal``); the other ranks ``Comm.host_wait(key, wait_timeout_s)`` (default
        :data:`FEATS_WAIT_S` = 6 h, ``ROWPAIR_FEATS_WAIT_S``) — no collective is pending while ``src`` featurises for minutes, so no
        collective timeout applies; (ii) ``failed:…`` raises ``refused: <what>_rank0_failed: <Type>: <msg>`` (RowpairRefused) on EVERY
        rank, ``src`` included — never a hang; a control word (first token not ``ok``) is returned to every rank and nothing is broadcast;
        (iii) META (one object broadcast): the top-level key set, the dtype / shape / device of every ``skip_keys`` entry (those tensors do
        NOT travel — receivers build their by-design placeholders from the meta, e.g. :func:`msa_host.place_skipped`), the ``host_keys``
        list, and — ``carry_rng`` — rank ``src``'s RNG state after featurising (:func:`ckpt.rng_state`: python, numpy, torch CPU, torch
        CUDA when initialised), SET on the receivers (:func:`ckpt.set_rng_state`) so later replicated draws agree; (iv) TENSORS + non-tensor
        leaves through :func:`bcast.broadcast_tensordict` (small tensors coalesced, large ones one collective each; non-tensor leaves —
        strings, lists, atom arrays — pickled with the structure); receivers land tensors on ``device`` (default the comm device) and
        ``host_keys`` on (pinned, ``pin_host``) host memory; (v) RECEIPT CHECK: a missing key or a tensor on the wrong device raises
        ``refused: <what>_bcast_malformed: <key>``. ``FeatureBroadcast`` = ``(feats, status, skipped, host, n_tensors, nbytes, wait_s,
        bcast_s, rng)`` + ``.words()``; facts recorded (``feats_src feats_bcast_keys … feats_bcast_s``). Without a multi-rank group the call
        returns ``feats`` untouched (P == 1: the engine's own path).
    status_word(exc=None, extra="") -> str             ``ok[ <extra>]`` or, for an exception, ``failed:<Type>:<msg>`` (one line, <= 200 chars)
    data_form_word(form) -> str                        ``data_form=rank0_bcast|per_rank`` (:data:`DATA_FORMS`; the launch line's word)

Why one seed: CPython salts ``str`` / ``bytes`` hashing per interpreter unless ``PYTHONHASHSEED`` is set, so the iteration order of a set
(or of anything ordered by ``hash()``) over strings differs between two processes started without one. An engine input pipeline that orders
any feature by such an iteration — chain or entity sets, template hit tables, annotation categories — featurises one query to different
BYTES in rank processes carrying different seeds, and the digest gate refuses the run by name (``feats_ranks_differ``). One seed for every
rank of a launch removes the process dependence; the digest gate stays the check (it is computed and compared on every multi-rank
run, never skipped). A P == 1 run spawns nothing and is untouched.
"""
from __future__ import annotations

import os
import hashlib
import json
import sys
from typing import Callable, Iterable, List, Mapping, NamedTuple, Optional, Tuple

__all__ = ["HASHSEED_ENV", "HASHSEED_DEFAULT", "HASHSEED_MAX", "SOURCES", "DIGEST_HEX", "MODES", "hash_seed", "invalid_parent", "hashseed_word", "hashseed_fields", "ranks_env",
           "rankenv_line", "FeatureDigest", "digest_features", "feature_digest", "is_atom_array", "ranks_agree", "gather_digests", "agree_word", "differ_refusal",
           "assert_ranks_agree", "DATA_FORMS", "FEATS_WAIT_S", "ENV_FEATS_WAIT", "FeatureBroadcast", "status_word", "data_form_word", "check_data_form",
           "broadcast_features"]

HASHSEED_ENV = "PYTHONHASHSEED"
HASHSEED_DEFAULT = "0"                     # the seed exported when the launcher names none (CPython: 0 disables str-hash randomisation; any fixed value serves — one value for all ranks is the point)
HASHSEED_MAX = 4294967295                  # CPython accepts an integer in [0, 4294967295]; anything else ("random", "", out of range) is not a seed the ranks could share
SOURCES = ("default", "inherited")
DIGEST_HEX = 16                            # the digits of the sha256 hex digest compared across ranks and printed on refusal (64 bits)
MODES = ("refuse", "census")               # assert_ranks_agree: differing digests are a refusal on every rank | a printed census word, never a refusal
DATA_FORMS = ("rank0_bcast", "per_rank")   # who featurises: rank 0 alone, the tree broadcast (broadcast_features) | every rank (the digest gate compares them)
FEATS_WAIT_S = 6 * 3600.0                  # how long a receiving rank waits at the rendezvous for rank 0's status word (no collective pending meanwhile)
ENV_FEATS_WAIT = "ROWPAIR_FEATS_WAIT_S"
STATUS_MAX = 200                           # a failure status word carries at most this many characters of the message (one line)


# ---------------------------------------------------------------- launcher side (stdlib only)
def _parent_raw(parent_env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if parent_env is None else parent_env
    return str(env.get(HASHSEED_ENV, "") or "").strip()


def hash_seed(parent_env: Optional[Mapping[str, str]] = None) -> Tuple[str, str]:
    """``(value, source)``: the launcher's own ``PYTHONHASHSEED`` when it is a decimal integer in CPython's accepted range ``[0, HASHSEED_MAX]``
    (``inherited``; canonicalised to its decimal spelling), else :data:`HASHSEED_DEFAULT` (``default`` — unset, empty, ``random`` or any other
    value: never a refusal)."""
    raw = _parent_raw(parent_env)
    if raw.isascii() and raw.isdigit() and int(raw) <= HASHSEED_MAX:
        return str(int(raw)), "inherited"
    return HASHSEED_DEFAULT, "default"


def invalid_parent(parent_env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The launcher's ``PYTHONHASHSEED`` value when it is PRESENT (non-empty) but not a seed the ranks could share (``random``, garbage, out of
    range) — shown on the census word so a discarded value is visible; ``None`` otherwise (unset / empty / a valid seed)."""
    raw = _parent_raw(parent_env)
    return raw if raw and hash_seed(parent_env)[1] == "default" else None


def hashseed_word(parent_env: Optional[Mapping[str, str]], n_ranks: int) -> str:
    """The launch line's census word: ``hashseed=<v> source=default|inherited [parent=<repr>] ranks=<P>`` (``parent=`` only when the launcher's
    value was present but invalid: its first 16 characters, quoted)."""
    seed, source = hash_seed(parent_env)
    bad = invalid_parent(parent_env)
    return f"hashseed={seed} source={source}" + (f" parent={bad[:16]!r}" if bad is not None else "") + f" ranks={int(n_ranks)}"


def hashseed_fields(parent_env: Optional[Mapping[str, str]] = None) -> dict:
    """The run record's fields: ``{"hashseed": <v>, "hashseed_source": default|inherited}`` (+ ``"hashseed_parent": <repr>`` when the
    launcher's value was present but invalid)."""
    seed, source = hash_seed(parent_env)
    out = {"hashseed": seed, "hashseed_source": source}
    bad = invalid_parent(parent_env)
    if bad is not None:
        out["hashseed_parent"] = bad[:16]
    return out


def ranks_env(parent_env: Optional[Mapping[str, str]], n_ranks: int) -> Tuple[dict, str]:
    """``(env, census_word)``: ``env`` = ``dict(parent_env)`` (default ``os.environ``) with the launch's ONE hash seed exported under
    ``PYTHONHASHSEED`` — the base environment of every rank interpreter of an ``n_ranks``-rank launch (a spawned interpreter reads the
    variable at start-up, so it must be in the environment the rank is started WITH); ``census_word`` = :func:`hashseed_word`."""
    base = os.environ if parent_env is None else parent_env
    seed, _source = hash_seed(base)
    env = dict(base)
    env[HASHSEED_ENV] = seed
    return env, hashseed_word(base, n_ranks)


def rankenv_line(tag: str, parent_env: Optional[Mapping[str, str]], n_ranks: int) -> str:
    """The launcher's one line per P > 1 launch: ``[<tag>] RANKENV hashseed=<v> source=default|inherited ranks=<P>``."""
    return f"[{tag}] RANKENV {hashseed_word(parent_env, n_ranks)}"


# ---------------------------------------------------------------- rank side (torch through the comm surface, imported lazily)
class FeatureDigest(NamedTuple):
    """The digest of a feature dict and its census: which leaves entered it and which were excluded by name."""
    digest: str                              # sha256 hex, 64 digits
    tensor_leaves: int                       # torch tensors + numpy arrays hashed by dtype / shape / bytes
    nontensor_leaves: Tuple[str, ...]        # paths of the non-tensor leaves that entered (canonical JSON scalars, bytes, atom-array summaries `<path>:atom_array`)
    unhashed: Tuple[str, ...]                # `<path>:<type>` of every leaf EXCLUDED from the digest (an object with no canonical form here) — named, never silent

    def words(self) -> str:
        """The gate line's census words: ``tensor_leaves=<n> nontensor_leaves=<k>[:<paths>] nontensor_unhashed=<path:type,…|none>``."""
        named = f"{len(self.nontensor_leaves)}" + (":" + ",".join(self.nontensor_leaves) if self.nontensor_leaves else "")
        return f"tensor_leaves={self.tensor_leaves} nontensor_leaves={named} nontensor_unhashed={','.join(self.unhashed) or 'none'}"


def is_atom_array(o) -> bool:
    """A biotite ``AtomArray`` / ``AtomArrayStack``-like object (duck-typed: no biotite import here)."""
    return all(hasattr(o, a) for a in ("get_annotation_categories", "get_annotation", "coord")) and not isinstance(o, type)


def _np():
    try:
        import numpy as np                                                # noqa: PLC0415
        return np
    except ModuleNotFoundError:
        return None


JSON_SCALARS = (str, bool, int, float, type(None))


def _json(o) -> Optional[str]:
    """Canonical JSON of a scalar (or a list of scalars); ``None`` when ``o`` has no such form (the caller excludes the leaf by name)."""
    try:
        return json.dumps(o, sort_keys=True)
    except (TypeError, ValueError, OverflowError):
        return None


def _hash_ndarray(h, path: str, arr) -> Optional[str]:
    """``path|dtype|shape|`` + the array's bytes (C order; nothing after the header for an empty array). An object-dtype array of scalars enters
    as their canonical JSON. Returns ``None`` when hashed, else the type word of a leaf that has no byte form here (excluded by the caller)."""
    np = _np()
    a = np.asarray(arr)
    if a.dtype == object:
        flat = a.reshape(-1).tolist()
        js = _json(flat) if all(isinstance(x, JSON_SCALARS) for x in flat) else None
        if js is None:
            return "ndarray[object]"
        h.update(f"{path}|{a.dtype}|{tuple(a.shape)}|".encode())
        h.update(js.encode())
        return None
    try:
        mv = memoryview(np.ascontiguousarray(a).reshape(-1)).cast("B") if a.size else None
    except (TypeError, ValueError):
        return f"ndarray[{a.dtype}]"
    h.update(f"{path}|{a.dtype}|{tuple(a.shape)}|".encode())
    if mv is not None:
        h.update(mv)
    return None


def _hash_atom_array(h, path: str, a, unhashed: List[str]) -> None:
    """The canonical, attachment-order-independent summary of an atom array: sorted category names, each annotation's dtype/shape/bytes in
    that order, then ``coord`` and ``box`` (when present) by dtype/shape/bytes. Bonds are not part of the summary. An annotation with no byte
    form is excluded by name (``<path>.<category>:<type>``)."""
    cats = sorted(str(c) for c in a.get_annotation_categories())
    h.update(f"{path}|atom_array|categories={','.join(cats)}|".encode())
    for c in cats:
        bad = _hash_ndarray(h, f"{path}.{c}", a.get_annotation(c))
        if bad:
            unhashed.append(f"{path}.{c}:{bad}")
    for name in ("coord", "box"):
        arr = getattr(a, name, None)
        if arr is not None:
            bad = _hash_ndarray(h, f"{path}.{name}", arr)
            if bad:
                unhashed.append(f"{path}.{name}:{bad}")


def digest_features(feats, exclude: Iterable[str] = ()) -> FeatureDigest:
    """The :class:`FeatureDigest` of ``feats`` minus the top-level keys in ``exclude`` (module contract above). A mapping is filtered by key;
    any other object is digested whole (``exclude`` must then be empty)."""
    from ._torch import torch
    from .ckpt import hash_tensor_leaf
    drop = set(exclude)
    if isinstance(feats, Mapping):
        view = {k: v for k, v in feats.items() if k not in drop}
    elif drop:
        raise TypeError(f"digest_features: exclude={sorted(drop)} given for a {type(feats).__name__} (only a mapping's top-level keys can be excluded)")
    else:
        view = feats
    np = _np()
    h = hashlib.sha256()
    n_tensor = 0
    nontensor: List[str] = []
    unhashed: List[str] = []

    def rec(o, path: str) -> None:
        nonlocal n_tensor
        if isinstance(o, torch.Tensor):
            hash_tensor_leaf(h, path, o)                                       # the ONE tensor-leaf rule (ckpt.hash_tensor_leaf): header + bytes
            n_tensor += 1
        elif np is not None and isinstance(o, np.ndarray):
            bad = _hash_ndarray(h, path, o)
            if bad:
                unhashed.append(f"{path}:{bad}")
            else:
                n_tensor += 1
        elif isinstance(o, Mapping):
            if path:
                h.update(f"{path}|map|{len(o)}|".encode())                     # a nested mapping's size is identity (the root's is not: `exclude` filters it)
            for k in sorted(o.keys(), key=str):
                rec(o[k], f"{path}/{k}")
        elif isinstance(o, (list, tuple)):
            h.update(f"{path}|seq|{len(o)}|".encode())
            for i, v in enumerate(o):
                rec(v, f"{path}/{i}")
        elif is_atom_array(o):
            _hash_atom_array(h, path, o, unhashed)
            nontensor.append(f"{path}:atom_array")
        elif isinstance(o, (bytes, bytearray)):
            h.update(f"{path}|bytes|{len(o)}|".encode())
            h.update(bytes(o))
            nontensor.append(path)
        elif isinstance(o, JSON_SCALARS) or (np is not None and isinstance(o, np.generic)):
            val = o.item() if (np is not None and isinstance(o, np.generic)) else o
            js = _json(val) if isinstance(val, JSON_SCALARS) else None
            if js is None:
                unhashed.append(f"{path}:{type(o).__name__}")                # a scalar with no canonical JSON (complex, datetime, …): excluded BY NAME
            else:
                tag = f"json:{o.dtype}" if (np is not None and isinstance(o, np.generic)) else "json"
                h.update(f"{path}|{tag}|{js}".encode())
                nontensor.append(path)
        else:
            unhashed.append(f"{path}:{type(o).__name__}")                     # excluded from the digest BY NAME (the gate line prints it)

    rec(view, "")
    return FeatureDigest(h.hexdigest(), n_tensor, tuple(nontensor), tuple(unhashed))


def feature_digest(feats, exclude: Iterable[str] = ()) -> str:
    """``digest_features(feats, exclude).digest`` — sha256 hex (64 digits)."""
    return digest_features(feats, exclude).digest


def _comm(comm=None):
    if comm is not None:
        return comm
    from .dist import comm as _current
    return _current()


def ranks_agree(digest: str, comm=None) -> Optional[bool]:
    """``True`` when every rank's ``digest[:DIGEST_HEX]`` equals rank 0's, ``False`` otherwise — the same answer on EVERY rank (rank 0's
    digits are broadcast, each rank compares, one max all-reduce of the per-rank differ bit); ``None`` when there is no multi-rank group
    (``comm`` inactive or P == 1). Every rank of the group must call it (two collectives)."""
    c = _comm(comm)
    if not getattr(c, "active", False) or int(getattr(c, "P", 1)) <= 1:
        return None
    from ._torch import torch
    head = str(digest)[:DIGEST_HEX]
    if len(head) != DIGEST_HEX:
        raise ValueError(f"ranks_agree: digest {digest!r} has fewer than {DIGEST_HEX} hex digits")
    mine = torch.tensor(list(bytes.fromhex(head)), dtype=torch.int64, device=c.device)
    ref = mine.clone()
    c.bcast_(ref, 0)                                                          # rank 0's digits on every rank
    differ = (mine != ref).any().to(torch.int64).reshape(1)
    c.allreduce_(differ, "max")                                               # any rank differing = every rank answers False
    return int(differ.item()) == 0


def gather_digests(digest: str, comm=None) -> Optional[List[str]]:
    """Every rank's ``digest[:DIGEST_HEX]`` (index = rank) on every rank — one object all-gather; ``None`` without a multi-rank group."""
    c = _comm(comm)
    if not getattr(c, "active", False) or int(getattr(c, "P", 1)) <= 1:
        return None
    return [str(d) for d in c.allgather_obj(str(digest)[:DIGEST_HEX])]


def agree_word(what: str, identical: Optional[bool]) -> str:
    """The census word: ``<what>_ranks_equal=yes|no|n/a``."""
    return f"{what}_ranks_equal={'n/a' if identical is None else ('yes' if identical else 'no')}"


def differ_refusal(what: str, rank: int, digest: str):
    """The refusal raised when the ranks' digests differ (``<what>_ranks_differ``; ``what`` = ``feats`` for the model-input features)."""
    from . import RowpairRefused
    return RowpairRefused(f"refused: {what}_ranks_differ: rank {int(rank)} digest {str(digest)[:DIGEST_HEX]} — the ranks featurised the query differently "
                          f"(upstream's data pipeline ran once per rank); every rank's `[{what}]` line names its digest", lever="n_gpu")


def assert_ranks_agree(digest: str, what: str = "feats", mode: str = "refuse", comm=None,
                       log: Optional[Callable[[str], None]] = None) -> Optional[bool]:
    """``mode="refuse"``: :func:`ranks_agree`; ``False`` raises :func:`differ_refusal` on EVERY rank (all ranks learn the same answer, so all
    refuse — no rank is left waiting in a collective); returns ``True`` or ``None`` (no group). ``mode="census"``: :func:`gather_digests`; one
    line ``[<what>] rank <r> digest <d16> <what>_ranks_equal=yes|no ranks=<P> digests=<d16,d16,…>`` through ``log`` (default: the comm's own
    ``log``, else stderr) and the bool returned — never a refusal (an engine whose ranks legitimately hold different inputs records the fact)."""
    if mode not in MODES:
        raise ValueError(f"assert_ranks_agree: mode={mode!r} is not one of {MODES}")
    c = _comm(comm)
    rank = int(getattr(c, "rank", 0))
    if mode == "refuse":
        identical = ranks_agree(digest, c)
        if identical is False:
            raise differ_refusal(what, rank, digest)
        return identical
    heads = gather_digests(digest, c)
    identical = None if heads is None else len(set(heads)) == 1
    line = f"[{what}] rank {rank} digest {str(digest)[:DIGEST_HEX]} {agree_word(what, identical)} ranks={len(heads) if heads else 1}" + (f" digests={','.join(heads)}" if heads else "")
    emit = log if log is not None else getattr(c, "log", None)
    if callable(emit):
        emit(line)
    else:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    return identical


# ---------------------------------------------------------------- rank-0 featurisation: status word, meta, tensors (the rank0_bcast data form)
class FeatureBroadcast(NamedTuple):
    """What :func:`broadcast_features` hands every rank."""
    feats: Optional[object]                  # the feature tree on this rank (rank src: its own object, untouched; receivers: the received tree; None under a control word)
    status: str                              # the status word every rank saw (`ok …` | a caller control word; `failed:…` raises instead)
    skipped: dict                            # {key: {"shape": tuple, "dtype": str, "dev": str}} — skip_keys present on src: NOT broadcast; receivers build placeholders from this
    host: Tuple[str, ...]                    # host_keys present in the tree (landed on host memory on receivers)
    n_tensors: int                           # tensor leaves broadcast
    nbytes: int                              # their total bytes
    wait_s: float                            # seconds this rank spent at the rendezvous (receivers: rank src's featurisation time as seen here)
    bcast_s: float                           # seconds in the meta + tensor broadcasts
    rng: str                                 # carried | off | none (carry_rng but src had no state to send)

    def words(self) -> str:
        gib = self.nbytes / 2 ** 30
        return (f"feats_status={self.status.split()[0]} tensors={self.n_tensors} gib={gib:.2f} skipped={','.join(sorted(self.skipped)) or 'none'} "
                f"host_keys={','.join(self.host) or 'none'} rng={self.rng} wait_s={self.wait_s:.1f} bcast_s={self.bcast_s:.1f}")


def status_word(exc: Optional[BaseException] = None, extra: str = "") -> str:
    """``ok`` (+ `` <extra>``) when ``exc`` is None, else ``failed:<Type>:<msg>`` — one line, the message cut to :data:`STATUS_MAX` characters."""
    if exc is None:
        return "ok" + (f" {extra}" if extra else "")
    msg = " ".join(str(exc).split())[:STATUS_MAX]
    return f"failed:{type(exc).__name__}:{msg}"


def data_form_word(form: str) -> str:
    """The launch line's word ``data_form=rank0_bcast|per_rank``."""
    return f"data_form={check_data_form(form)}"


def check_data_form(form) -> str:
    """``form`` when it is one of :data:`DATA_FORMS`; anything else is refused by name (RowpairRefused)."""
    f = str(form or "").strip()
    if f not in DATA_FORMS:
        from . import RowpairRefused
        raise RowpairRefused(f"refused: data_form={f!r} is not one of {'|'.join(DATA_FORMS)}", lever="n_gpu")
    return f


def _wait_timeout(wait_timeout_s: Optional[float]) -> float:
    if wait_timeout_s is not None:
        return float(wait_timeout_s)
    raw = (os.environ.get(ENV_FEATS_WAIT) or "").strip()
    return float(raw) if raw else FEATS_WAIT_S


def _rank0_failed(what: str, word: str):
    from . import RowpairRefused
    _tag, _, rest = word.partition(":")
    etype, _, msg = rest.partition(":")
    return RowpairRefused(f"refused: {what}_rank0_failed: {etype or 'Error'}: {msg}", lever="n_gpu")


def _malformed(what: str, key: str, why: str):
    from . import RowpairRefused
    return RowpairRefused(f"refused: {what}_bcast_malformed: {key} ({why})", lever="n_gpu")


def _tensor_meta(t) -> dict:
    return {"shape": tuple(int(x) for x in t.shape), "dtype": str(t.dtype).replace("torch.", ""), "dev": t.device.type}


def check_received(feats, meta: dict, device=None, what: str = "feats") -> None:
    """The receipt check of a received tree against the source rank's meta: every announced top-level key present (``skip`` excepted); every
    announced top-level tensor with the announced dtype and shape; ``host`` keys on the host; every other tensor where the source held it — on
    ``device``'s type when the source held it on an accelerator, on the host when the source held it there. Raises
    ``refused: <what>_bcast_malformed: <key> (…)`` (RowpairRefused)."""
    from ._torch import torch
    if not isinstance(feats, Mapping):
        raise _malformed(what, "<tree>", f"a mapping with keys {list(meta.get('keys', []))[:6]} was announced, a {type(feats).__name__} arrived")
    skip, host, tens = meta.get("skip", {}), set(meta.get("host", ())), meta.get("tensors", {})
    dev_type = torch.device(device).type if device is not None else None
    for k in meta.get("keys", []):
        if k in skip:
            continue
        if k not in feats:
            raise _malformed(what, str(k), "announced by the source rank, absent on this rank")
        if k in tens:
            v, m = feats[k], tens[k]
            if not isinstance(v, torch.Tensor):
                raise _malformed(what, str(k), f"a {m['dtype']}{list(m['shape'])} tensor was announced, a {type(v).__name__} arrived")
            if str(v.dtype).replace("torch.", "") != m["dtype"] or tuple(v.shape) != tuple(m["shape"]):
                raise _malformed(what, str(k), f"announced {m['dtype']}{list(m['shape'])}, arrived {str(v.dtype).replace('torch.', '')}{list(v.shape)}")
            want = "cpu" if (k in host or m["dev"] == "cpu") else (dev_type or v.device.type)
            if v.device.type != want:
                raise _malformed(what, str(k), f"expected on {want}, arrived on {v.device.type}")


def broadcast_features(feats=None, *, comm=None, src: int = 0, key: str = "feats/0", status: Optional[str] = None,
                       skip_keys: Iterable[str] = (), host_keys: Iterable[str] = (), device=None, carry_rng: bool = True,
                       wait_timeout_s: Optional[float] = None, what: str = "feats", pin_host: bool = True) -> FeatureBroadcast:
    """Rank ``src`` featurised ``feats`` (a mapping; the other ranks pass None): every rank returns a :class:`FeatureBroadcast` holding the same
    tree (module contract: rendezvous outside any collective -> status word -> meta -> tensors -> receipt check). A source argument that is
    not a mapping is the status word ``failed:TypeError:…`` every rank refuses on (checked before the signal); with no group (P <= 1) ``feats``
    is returned as given, unchecked. ``key`` must be unique per call within a run (e.g. ``feats/<batch index>``): the store keeps every key."""
    import time
    c = _comm(comm)
    P = int(getattr(c, "P", 1))
    rank = int(getattr(c, "rank", 0))
    word = status if status is not None else ("ok" if (feats is not None or rank != src) else status_word(RuntimeError("rank src produced no features")))
    if not getattr(c, "active", False) or P <= 1:
        if word.startswith("failed:"):
            raise _rank0_failed(what, word)
        return FeatureBroadcast(feats, word, {}, (), 0, 0, 0.0, 0.0, "off")
    from ._torch import torch
    from . import bcast as B, ckpt as CK
    from .evidence import record_schedule
    skip = [str(k) for k in skip_keys]
    hostk = [str(k) for k in host_keys]
    t0 = time.time()
    if rank == src:                                                          # (i) rendezvous through the store: no collective is pending while src featurised
        if (word.split() or [""])[0] == "ok" and not isinstance(feats, Mapping):   # the source's ARGUMENT is checked BEFORE the signal: a bad one is the
            word = status_word(TypeError(f"rank {src} must pass a mapping of features (got {type(feats).__name__})"))   # word every rank refuses on, so it
        c.host_signal(key, word)                                             # never leaves the receivers waiting in the meta broadcast
    else:
        word = str(c.host_wait(key, timeout_s=_wait_timeout(wait_timeout_s)))
    wait_s = time.time() - t0
    if word.startswith("failed:"):                                            # (ii) the same refusal on every rank
        raise _rank0_failed(what, word)
    if (word.split() or [""])[0] != "ok":                                     # a caller control word: every rank has it; nothing travels
        return FeatureBroadcast(feats if rank == src else None, word, {}, (), 0, 0, wait_s, 0.0, "off")
    t1 = time.time()
    dev = torch.device(device) if device is not None else c.device
    if rank == src:                                                          # (iii) meta (feats is a Mapping: checked before the rendezvous)
        skipped = {k: _tensor_meta(feats[k]) for k in skip if k in feats and isinstance(feats[k], torch.Tensor)}
        present_host = tuple(k for k in hostk if k in feats)
        rng = CK.rng_state() if carry_rng else None
        meta = {"keys": sorted((str(k) for k in feats.keys()), key=str), "skip": skipped, "host": list(present_host),
                "tensors": {str(k): _tensor_meta(v) for k, v in feats.items() if isinstance(v, torch.Tensor) and k not in skipped}, "rng": rng is not None}
        c.broadcast_obj({"meta": meta, "rng": rng}, src)
        payload = {k: v for k, v in feats.items() if k not in skipped}
    else:
        box = c.broadcast_obj(None, src)
        meta, rng = box["meta"], box["rng"]
        skipped, present_host = dict(meta["skip"]), tuple(meta["host"])
        payload = None
    got = B.broadcast_tensordict(payload, src=src, device=dev, keep_device=False)   # (iv) tensors (+ pickled non-tensor leaves): source placement reproduced (device / cpu)
    n_t, n_b = 0, 0
    for _p, v in _tensor_leaves(got):
        n_t += 1
        n_b += int(v.numel()) * int(v.element_size())
    rng_word = "off" if not carry_rng else ("carried" if rng is not None else "none")
    if rank != src:
        for k in present_host:                                               # host keys: (pinned) host memory, as the source holds them
            v = got.get(k) if isinstance(got, Mapping) else None
            if isinstance(v, torch.Tensor):
                if v.device.type != "cpu":
                    v = v.to("cpu")
                if pin_host and torch.cuda.is_available() and not v.is_pinned():
                    try:
                        v = v.pin_memory()
                    except RuntimeError:                                     # pinning is an optimisation; pageable host memory is msa_host's named fallback too
                        pass
                got[k] = v
        check_received(got, meta, dev, what)                                 # (v) the receipt check: keys, dtypes, shapes, placement
        if carry_rng and rng is not None:
            CK.set_rng_state(rng)
        out = got
    else:
        out = feats
    bcast_s = time.time() - t1
    fb = FeatureBroadcast(out, word, skipped, present_host, n_t, n_b, wait_s, bcast_s, rng_word)
    record_schedule(feats_src=src, feats_status=word.split()[0], feats_bcast_tensors=n_t, feats_bcast_gib=round(n_b / 2 ** 30, 3),
                    feats_skipped=",".join(sorted(skipped)) or "none", feats_host_keys=",".join(present_host) or "none", feats_rng=rng_word,
                    feats_wait_s=round(wait_s, 1), feats_bcast_s=round(bcast_s, 1))
    return fb


def _tensor_leaves(o, path=""):
    from ._torch import torch
    if isinstance(o, torch.Tensor):
        yield path, o
    elif isinstance(o, Mapping):
        for k in o:
            yield from _tensor_leaves(o[k], f"{path}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _tensor_leaves(v, f"{path}/{i}")
