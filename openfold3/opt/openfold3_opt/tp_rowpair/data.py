"""The data-pipeline side of the tp line: OpenFold3's template featurizer
``featurize_template_structures_of3`` (``core/data/pipelines/featurization/template.py``) builds the dense template pair features
``template_distogram [T, N, N, 39]`` and ``template_unit_vector [T, N, N, 3]`` on the host for every query (T = 4 slots, all dummies under
``--use-templates false``: ``672 · N²`` bytes that no rank may hold). ``install()`` rebinds it (and the name the inference dataset module
imported) to ``featurize_template_structures_lazy``: the token-level template features are OpenFold3's statements unchanged; the two pair keys
become ``[T, 1, 1, F]`` zero placeholders plus the flag ``_lazy_template_pair``, and the template binding (``template.template_rows``) builds
each rank's ROWS of them per row block:

* DUMMY slots (every predict-time slot of an untemplated chain): zero rows synthesised per block (the dummies' pair features are exactly +0.0
  by the featurizer's own statements);
* REAL slots — a templated chain whose top-k templates the sampler returned, which happens only under a template-consuming upstream fix
  (``--upstream-fix OF3-001`` with ``--use-templates true``; ``templ_guard.policy()`` = ``consume`` in the rank, inherited by the DataLoader
  workers it forks): every rank COMPUTES its rows of the two pair features per block from the featurizer's O(T·N) per-token PRECURSORS
  (``opt_core.mem.rowpair.template.real_template_rows``: on the host — the line's constant placement — bitwise the dense features sliced to rows).
  Under the consume policy EVERY item's batch carries the precursor keys (``REAL_KEYS``: the float64 pseudo-beta coordinates and the squared
  distogram bin edges as int64 bit patterns, the finite fp32 backbone frame coordinates, the featurizer's per-token chain ids) and the flag
  ``_lazy_template_real`` whose VALUE says whether a slot is real — one batch schema per run on every rank (the rank-0 batch sync pairs keys by
  name), whatever a rank's own featurization met. One census line per templated item: ``[rowpair data rank<r>] REAL templates: slots k/T real (…)``.

Refused by name (``DataRefused``): real template hits WITHOUT the consume policy in the featurizing process — the lazy path never runs a real
template as a dummy.
"""
from __future__ import annotations

import os
import sys

import torch

from . import core as C

LAZY_FLAG = "_lazy_template_pair"
REAL_FLAG = "_lazy_template_real"                        # [1]: 1.0 = a template slot of this item is real (template.template_rows -> real_template_rows over the REAL_KEYS precursors), 0.0 = all dummies; present, with REAL_KEYS, on every item under the consume policy
PB_COORDS_KEY = "template_pseudo_beta_coords_f64bits"   # [T, N, 3] int64: the featurizer's float64 pseudo-beta coordinates as bit patterns (NaN = atom absent) — integers pass the rank-0
                                                         # batch sync and the cross-rank feature digest untouched by any float cast (model.forward_synced / feature_digest)
FRAMES_KEY = "template_frame_atom_coords"                # [T, N, 3, 3] fp32: backbone N/CA/C coordinates, NaN -> 0.0 (create_template_unit_vector's own first statement; absence is
                                                         # template_backbone_frame_mask == 0, computed from the NaNs before)
ASYM_KEY = "template_asym_id"                            # [N] int32: the featurizer's per-token chain renumbering (featurization/template.py:96-105), the operand of its inter-chain pair mask
EDGES_KEY = "template_dgram_edges_f64bits"               # [2, n_bins] int64: create_template_distogram's squared (lower, upper) bin edges, float64 bit patterns (primitives/featurization/template.py:240-243)
REAL_KEYS = (PB_COORDS_KEY, FRAMES_KEY, ASYM_KEY, EDGES_KEY)
DGRAM_INF = 1e8                                          # create_template_distogram's inf_value default (the last upper edge; featurize_template_structures_of3 never overrides it)
DATASET_MODULES = ("openfold3.core.data.framework.single_datasets.inference", "openfold3.core.data.framework.single_datasets.base_of3")   # the modules that import the featurizer's name


class DataRefused(RuntimeError):
    pass


def log_data(msg: str) -> None:
    """``[rowpair data rank<r>] <msg>`` on stderr. The featurizer runs inside upstream's DataLoader worker processes (forked from the rank): a plain
    write, no communicator or device query (``core.comm()`` belongs to the rank process)."""
    rank = (os.environ.get("OF3TP_RANK") or "0").strip()
    sys.stderr.write(f"[rowpair data rank{rank}] {msg}\n")
    sys.stderr.flush()


def consume_policy():
    """``(policy, fix)`` of this process's template guard (``templ_guard.policy()``: ``("consume", "OF3-001")`` once ``cli.apply_upstream_fix`` applied
    a template-consuming upstream fix in this rank, else ``("ignore", "upstream_0.4.1")``) — the one key under which the lazy path serves REAL slots."""
    from .. import templ_guard
    return templ_guard.policy()


def real_template_precursors(atom_array, pre, n_tokens, min_bin, max_bin, n_bins, real: bool) -> dict:
    """The O(T·N) operands of the two dense pair statements, for the batch (``REAL_KEYS``): ``pre`` is OpenFold3's ``TemplateFeaturePrecursor``
    (``pseudo_beta_atom_coords [T, N, 3]`` / ``frame_atom_coords [T, N, 3, 3]``, float64, NaN = absent). The per-token chain ids and the bin edges
    are featurization/template.py:96-105 and primitives/featurization/template.py:240-243 verbatim; nothing ``N x N`` is formed. ``real`` is the
    value of ``REAL_FLAG``."""
    import numpy as np
    from openfold3.core.data.primitives.featurization.padding import pad_token_dim
    from openfold3.core.data.primitives.featurization.structure import extract_starts_entities
    if pre.pseudo_beta_atom_coords.dtype != np.float64:
        raise DataRefused(f"refused: the template precursor's pseudo-beta coordinates are {pre.pseudo_beta_atom_coords.dtype}; expected float64 (openfold3 0.4.1's "
                          "TemplateFeaturePrecursor: the distogram statement runs in float64 — an operand of another width is refused, never cast)")
    pb = np.ascontiguousarray(pre.pseudo_beta_atom_coords)
    if tuple(pb.shape[1:]) != (int(n_tokens), 3) or tuple(pre.frame_atom_coords.shape[1:]) != (int(n_tokens), 3, 3):
        raise DataRefused(f"refused: template precursor shapes {pb.shape} / {pre.frame_atom_coords.shape} do not match n_tokens={n_tokens} "
                          "(openfold3's TemplateFeaturePrecursor changed shape; the lazy real-template path targets 0.4.1)")
    token_starts_with_stop, _ = extract_starts_entities(atom_array)                                  # featurization/template.py:96-105: the featurizer's asym ids
    token_starts = token_starts_with_stop[:-1]
    chain_ids_token = atom_array.chain_id[token_starts]
    _, renum_ids = np.unique(chain_ids_token, return_inverse=True)
    asym_id = pad_token_dim({"asym_id": torch.tensor(np.asarray(renum_ids).reshape(-1) + 1, dtype=torch.int32)}, n_tokens)["asym_id"]
    lower = np.linspace(min_bin, max_bin, n_bins) ** 2                                                # primitives/featurization/template.py:240-243: squared bin edges
    upper = np.concatenate([lower[1:], np.array([DGRAM_INF], dtype=lower.dtype)], axis=-1)
    return {PB_COORDS_KEY: torch.from_numpy(pb.copy()).view(torch.int64),
            FRAMES_KEY: torch.nan_to_num(torch.tensor(pre.frame_atom_coords, dtype=torch.float32), nan=0.0),   # create_template_unit_vector's first statement
            ASYM_KEY: asym_id,
            EDGES_KEY: torch.from_numpy(np.ascontiguousarray(np.stack([lower, upper]), dtype=np.float64)).view(torch.int64),
            REAL_FLAG: torch.ones(1) if real else torch.zeros(1)}


def real_slot_words(features) -> str:
    """``slots k/T real (t<i>: tokens <m>/<N> ...)`` for the featurizer's census line (per real slot: tokens with a pseudo-beta atom). The grammar is the
    core's ``slot_census`` one, written here from the token masks because this line prints in the DataLoader worker at featurize time (no layout,
    no communicator) under its own ``[rowpair data rank<r>]`` prefix."""
    pbm = features["template_pseudo_beta_mask"]
    T, N = int(pbm.shape[0]), int(pbm.shape[1])
    real = [t for t in range(T) if bool((pbm[t] != 0).any()) or bool((features["template_backbone_frame_mask"][t] != 0).any())]
    per = "; ".join(f"t{t}: tokens {int((pbm[t] != 0).sum())}/{N}" for t in real)
    return f"slots {len(real)}/{T} real ({per})"


def featurize_template_structures_lazy(atom_array, template_slice_collection, n_templates, n_tokens, min_bin, max_bin, n_bins):
    """``featurize_template_structures_of3`` with the dense ``[T, N, N, F]`` pair features replaced by ``[T, 1, 1, F]`` placeholders + ``LAZY_FLAG``;
    the token-level features (``template_pseudo_beta_mask``, ``template_backbone_frame_mask``, ``template_restype``) are OpenFold3's statements.
    under the consume policy every item adds the ``REAL_KEYS`` precursors + ``REAL_FLAG`` (1.0 when a slot is real); REAL slots (some mask entry
    non-zero) without that policy are refused by name."""
    import numpy as np
    from openfold3.core.data.primitives.featurization.template import create_template_feature_precursor_of3, create_template_restype
    pre = create_template_feature_precursor_of3(template_slice_collection, n_templates, n_tokens)
    features = {}
    features["template_pseudo_beta_mask"] = torch.tensor(~np.isnan(pre.pseudo_beta_atom_coords).any(axis=-1), dtype=torch.float)
    features["template_backbone_frame_mask"] = torch.tensor(~np.isnan(pre.frame_atom_coords).any(axis=(-2, -1)), dtype=torch.float)
    features["template_restype"] = create_template_restype(pre.res_names, features["template_pseudo_beta_mask"])
    T = int(features["template_pseudo_beta_mask"].shape[0])
    real = bool((features["template_pseudo_beta_mask"] != 0).any()) or bool((features["template_backbone_frame_mask"] != 0).any())
    policy, fix = consume_policy()
    if real and policy != "consume":
        raise DataRefused("refused: real template hits in the query on the tp line with no template-consuming upstream fix applied in this process "
                          f"(templ_policy={policy}({fix})): the line's lazy template path serves dummy slots, and real slots only under "
                          "`--upstream-fix OF3-001` with `--use-templates true` (each rank then computes its rows of the template pair features from the "
                          "featurizer's per-token precursors) — it never runs a real template as a dummy; without the fix run with --use-templates false")
    if policy == "consume":                                                  # one batch schema per run policy on every rank: the precursor keys travel for every item, REAL_FLAG's value marks a real slot
        features.update(real_template_precursors(atom_array, pre, n_tokens, min_bin, max_bin, n_bins, real=real))
        if real:
            log_data(f"REAL templates: {real_slot_words(features)}; consumed under --upstream-fix {fix}: per-token precursors in the batch "
                     f"({', '.join(REAL_KEYS)}), each rank computes its rows of template_distogram / template_unit_vector per row block")
    features["template_distogram"] = torch.zeros((T, 1, 1, int(n_bins)), dtype=torch.float32)
    features["template_unit_vector"] = torch.zeros((T, 1, 1, 3), dtype=torch.float32)
    features[LAZY_FLAG] = torch.ones(1)
    return features


def install(patches) -> bool:
    """Rebind ``featurize_template_structures_of3`` in its module and in the dataset modules that imported the name, through the line's
    ``PatchSet`` (``patches.replace``: named in ``patches.names()``, restored by ``patches.restore()``). Idempotent (a second call finds the
    name rebound); returns whether it installed."""
    from openfold3.core.data.pipelines.featurization import template as _T
    stock = _T.featurize_template_structures_of3
    if stock is featurize_template_structures_lazy:
        return False
    featurize_template_structures_lazy.__wrapped__ = stock
    patches.replace(_T, "featurize_template_structures_of3", featurize_template_structures_lazy)
    rebound = []
    for modname in DATASET_MODULES:
        try:
            mod = __import__(modname, fromlist=["_"])
        except ImportError:                     # the training dataset module's optional dependencies may be absent; inference is the one predict uses
            continue
        if getattr(mod, "featurize_template_structures_of3", None) is stock:
            patches.replace(mod, "featurize_template_structures_of3", featurize_template_structures_lazy)
            rebound.append(modname.rsplit(".", 1)[-1])
    C.comm().log(f"[data] lazy template pair features installed (placeholders [T, 1, 1, F] + {LAZY_FLAG}; under a template-consuming upstream fix "
                 f"(templ_policy=consume at featurize time) per-token precursors + {REAL_FLAG} on every item, real slots computed per row block; a query with real template hits and no such fix "
                 f"stops by name (DataRefused); dataset modules rebound: {', '.join(rebound) or 'none'})")
    return True


# ----------------------------------------------------------------------------------------------- the rank0_bcast data form: ranks > 0 do not featurise
RANK_STUB_FLAG = "_rank0_bcast_stub"
"""Present ([1] int tensor) in the light record a rank > 0's dataset returns under the ``rank0_bcast`` data form (model.transfer_rank0_bcast
replaces the record by rank 0's featurised batch)."""

REGISTRY_NAME = "InferenceDataset"
_REGISTRY_ORIG = []                                          # [(registry dict, original class)] while the stub class is registered


def rank_stub_record(dataset, index: int) -> dict:
    """The light record a rank > 0 hands the DataLoader instead of featurising: upstream ``InferenceDataset.__getitem__``'s own fields
    (``query_id``, ``seed``, ``repeated_sample``, ``valid_sample`` — single_datasets/inference.py, the featurised branch minus
    ``create_all_features``) plus :data:`RANK_STUB_FLAG`. Rank 0 featurises the query; this rank receives rank 0's batch in the transfer hook."""
    dp = dataset.datapoint_cache.iloc[index]
    return {"query_id": dp["query_id"], "seed": torch.tensor([dp["seed"]]), "repeated_sample": torch.tensor([bool(dp["repeated_sample"])], dtype=torch.bool),
            "valid_sample": torch.tensor([True], dtype=torch.bool), RANK_STUB_FLAG: torch.tensor([1])}


_STUB_CLASS = None


def rank_stub_class():
    """``RankStubInferenceDataset``: upstream's ``InferenceDataset`` with ``__getitem__`` = :func:`rank_stub_record`. Built on first use and
    reachable as this module's attribute of that name (module ``__getattr__``), so a DataLoader worker of either start method — fork inherits
    it, forkserver / spawn unpickle the dataset by importing this module — resolves the class."""
    global _STUB_CLASS
    if _STUB_CLASS is None:
        from openfold3.core.data.framework.single_datasets.inference import InferenceDataset

        class RankStubInferenceDataset(InferenceDataset):          # upstream's dataset; items are the light record
            __getitem__ = rank_stub_record

        RankStubInferenceDataset.__module__ = __name__
        RankStubInferenceDataset.__qualname__ = "RankStubInferenceDataset"
        _STUB_CLASS = RankStubInferenceDataset
    return _STUB_CLASS


def __getattr__(name):                                       # PEP 562: pickle resolves `<this module>.RankStubInferenceDataset` in a worker that never called rank_stub_class()
    if name == "RankStubInferenceDataset":
        return rank_stub_class()
    raise AttributeError(name)


def install_rank_stub() -> bool:
    """Register the stub dataset class under upstream's registry name (``DATASET_REGISTRY['InferenceDataset']``, the name the data module
    constructs the predict dataset by) in THIS process — a rank > 0's main process under the ``rank0_bcast`` data form, before the data module
    builds its datasets. Idempotent; returns whether it installed. :func:`uninstall_rank_stub` restores the registry."""
    from openfold3.core.data.framework.single_datasets.abstract_single import DATASET_REGISTRY
    import openfold3.core.data.framework.single_datasets.inference  # noqa: F401 — registers upstream's class first
    cls = rank_stub_class()
    if DATASET_REGISTRY.get(REGISTRY_NAME) is cls:
        return False
    _REGISTRY_ORIG.append((DATASET_REGISTRY, DATASET_REGISTRY[REGISTRY_NAME]))
    DATASET_REGISTRY[REGISTRY_NAME] = cls
    log_data(f"rank0_bcast: this rank does not featurise — {REGISTRY_NAME} items are the light record (query_id, seed, repeated_sample, valid_sample); "
             f"rank 0's featurised batch arrives in the transfer hook")
    return True


def uninstall_rank_stub() -> bool:
    done = False
    while _REGISTRY_ORIG:
        reg, orig = _REGISTRY_ORIG.pop()
        reg[REGISTRY_NAME] = orig
        done = True
    return done
