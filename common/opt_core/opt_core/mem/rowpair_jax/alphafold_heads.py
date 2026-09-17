"""The AlphaFold-2 heads that READ THE PAIR, on this device's pair ROWS — ONE producer for both pinned AlphaFold-2 Haiku trees (the 2.3.x
multimer-era library and the dl_binder_design cafa3853 monomer library): ``DistogramHead``, ``PredictedAlignedErrorHead`` and, for a tree whose
``AlphaFold.__call__`` computes the confidence metrics INSIDE the Haiku program (``alphafold.common.confidence`` with ``use_jnp=True``), the two
confidence functions that read the aligned-error logits per residue pair (``compute_predicted_aligned_error``, ``predicted_tm_score``).

This module owns only the head BODIES. The region they run in (a ``shard_map`` after the trunk in which ``representations['pair']`` is this device's
row block ``[N/P, N, c_z]``), the trace flag that says so, the mesh and the placement belong to the caller (:mod:`alphafold`, the AF2 recipe), which
hands everything in through :func:`install_heads` — this module imports nothing from it. Every rebound body DISPATCHES TO THE STOCK BODY unless
``rows_active()`` is true, so replicated call sites, ``conf=replicated`` and host-side calls keep the stock program bit for bit.

The plan under ``rows_active()`` (rows = dim 0 of the pair block; ``P = n_gpu``; every collective is :mod:`shard`'s / :mod:`transition`'s):

| site | on the row block | collective | numerics class | leaves the region as |
|---|---|---|---|---|
| ``DistogramHead.__call__`` | ``half_logits = Linear(pair rows)`` ``[N/P, N, bins]``; ``logits = transition.symmetrize(half_logits)`` = the stock ``half_logits + swapaxes(half_logits, -2, -3)`` whose transposed rows arrive by ONE all_to_all; ``bin_edges`` as stock | ``transition.symmetrize`` (all_to_all) | row_local (the Linear) + moves_bytes (the symmetrisation) | ``logits`` = the ROW BLOCK — no device gathers ``[N, N, bins]``; the host's ``np.asarray`` of the program output assembles it (the writer boundary) |
| ``PredictedAlignedErrorHead.__call__`` | the STOCK body: ``Linear`` per ``(i, j)`` — on the row block it yields exactly the row block of the stock logits; the rebound body adds the row-extent gate only | none | row_local | ``logits`` = the row block |
| ``confidence.compute_predicted_aligned_error`` (in-program trees) | the STOCK body: softmax + expectation per ``(i, j)``; gated | none | row_local | ``aligned_confidence_probs`` ``[N/P, N, bins]`` and ``predicted_aligned_error`` ``[N/P, N]`` = row blocks; ``max_predicted_aligned_error`` a replicated scalar |
| ``confidence.predicted_tm_score`` (in-program trees; pTM and, with ``asym_id``, ipTM) | the per-``(i, j)`` TM term and the per-row normalised sum over ``j`` (complete per row) on my rows — ``residue_weights`` / ``asym_id`` sliced to my rows where they index ``i``, whole where they index ``j``; ``per_alignment`` ``[N/P]`` all-gathered to ``[N]``; the stock ``max`` over residues | ``shard.gather`` of an ``[N]``-vector | row_local + moves_bytes | a replicated scalar |
| ``MaskedMsaHead`` / ``PredictedLDDTHead`` / ``ExperimentallyResolvedHead``, ``confidence.compute_plddt`` | untouched: they read the MSA representation ``[S, N, c_m]`` / the structure module's single activations ``[N, c_s]`` / the pLDDT logits ``[N, bins]`` — not pair-shaped, REPLICATED by design (the AF2 recipe's stated floor) | — | stock | replicated |

Where each pinned tree computes the confidence metrics (:data:`CONFIDENCE_IN_GRAPH`, keyed by the sha256 of ``modules.py``): the 2.3.x tree's
``modules.AlphaFold.__call__`` (modules.py:466-470; ``AlphaFold_noE`` :205-210; ``modules_multimer.AlphaFold.__call__`` modules_multimer.py:464-470)
calls ``confidence.get_confidence_metrics(..., use_jnp=True)`` INSIDE the jitted Haiku program (model.py:63 ``jax.jit(hk.transform(_forward_fn).apply)``),
so pLDDT / PAE / pTM / ipTM / ranking are program outputs and the two pair-reading functions above are rebound ON the ``alphafold.common.confidence``
module object (:func:`confidence_module`; the ``confidence`` NAME inside ``modules`` is the caller's to replace). The dl_binder_design tree's program returns
the distogram and aligned-error LOGITS only (its ``alphafold/common/confidence.py`` is NumPy / SciPy, called on the host from the returned arrays:
model.py:31-46 ``get_confidence_metrics``, ``predict_pdb.py`` after ``jax.device_get``) — nothing confidence-shaped runs in its program, nothing is rebound,
and the host code receives the ``[N, N, bins]`` logits assembled by the transfer.

PIN. :func:`install_heads` hashes ``modules.__file__`` (and the confidence module's file for an in-program tree) and REFUSES BY NAME unless the sha256 is a
key of :data:`HEADS_PIN` — the bodies transcribe those bytes (per-site source lines in :data:`HEADS_BODIES`). FAIL-LOUD: under ``rows_active()`` a body handed
a pair block that is not ``[N/P, N, …]`` for the installed ``n_gpu`` raises :class:`opt_core.mem.MemLeverRefused` at trace (:func:`rows_gate`) — a whole pair
under the rows flag would otherwise be computed replicated, silently. Hazards named in :data:`HEADS_HAZARDS`. Standard library at import; jax inside the
bodies (:mod:`_lazy`); Python 3.8 syntax; no engine name.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Tuple

from .. import MemLeverRefused
from . import LEVER as FAMILY_LEVER
from . import _lazy
from . import haiku as _hk
from . import shard as _shard
from . import transition as _tr
from .alphafold import PIN_FILES, PIN_TREE, TREE_CF, TREE_DL, file_sha   # the family's ONE pin table and pin unit (alphafold imports this module lazily: no cycle)

__all__ = ["install_heads", "confidence_module", "file_sha", "rows_gate", "describe_heads", "HEADS_PIN", "PIN_MODULES", "PIN_CONFIDENCE",
           "CONFIDENCE_IN_GRAPH", "HEADS_BODIES", "HEADS_HAZARDS", "REQUIRED", "REQUIRED_CONFIDENCE", "OUTPUTS_ROWS", "SITES"]

TREE_A, TREE_B = TREE_CF, TREE_DL                                        # the 2.3.x multimer-era library / the dl_binder_design monomer library
PIN_MODULES: Dict[str, str] = PIN_FILES["modules"]                       # sha256 of alphafold/model/modules.py → the tree it is (a view of alphafold.PIN)
PIN_CONFIDENCE: Dict[str, str] = PIN_FILES["confidence"]                 # sha256 of alphafold/common/confidence.py for the trees that run it IN the program
HEADS_PIN: Dict[str, str] = {**PIN_MODULES, **PIN_CONFIDENCE}           # every stock file whose statements the bodies transcribe → its label
CONFIDENCE_IN_GRAPH: Dict[str, bool] = {sha: PIN_TREE[sha] == TREE_CF for sha in PIN_MODULES}   # per pinned modules.py: the 2.3.x tree's AlphaFold.__call__ computes the
                                                                         # confidence metrics IN the program (modules.py:466-470, modules_multimer.py:464-470); the dl_binder_design
                                                                         # program returns logits and confidence.py runs on the host (model.py:31-46; the kit's predict_pdb.py)
REQUIRED = ("DistogramHead", "PredictedAlignedErrorHead", "MaskedMsaHead", "PredictedLDDTHead", "ExperimentallyResolvedHead", "common_modules", "utils")
REQUIRED_CONFIDENCE = ("get_confidence_metrics", "compute_predicted_aligned_error", "predicted_tm_score", "compute_plddt", "_calculate_bin_centers")
MODULES_SUFFIX = ".model.modules"                                        # modules.__name__ ends so; the confidence module is <package>.common.confidence
SITES = ("DistogramHead.__call__", "PredictedAlignedErrorHead.__call__", "confidence.compute_predicted_aligned_error", "confidence.predicted_tm_score")
HEADS_BODIES: Dict[str, Tuple[str, str]] = {                             # site → (numerics class vs the stock body on the same rows, the stock statements transcribed)
    "DistogramHead.__call__": ("row_local+moves_bytes(symmetrize:all_to_all)",
                               "2.3.x modules.py:1526-1549 = dl_binder_design modules.py:1367-1391 (the same statements): half_logits Linear on the row block; "
                               "logits = transition.symmetrize(half_logits) = the stock `half_logits + jnp.swapaxes(half_logits, -2, -3)` (:1545 / :1387); no gather"),
    "PredictedAlignedErrorHead.__call__": ("row_local(stock_body,rows_gate)",
                                           "2.3.x modules.py:1225-1250 = dl_binder_design modules.py:1138-1163: the STOCK body (Linear per (i, j)) on the row block; no gather"),
    "confidence.compute_predicted_aligned_error": ("row_local(stock_body,rows_gate)",
                                                   "2.3.x confidence.py:89-112: the STOCK body (softmax + expectation per (i, j)) on the logits rows; in-program trees only"),
    "confidence.predicted_tm_score": ("row_local+moves_bytes(gather:[N])",
                                      "2.3.x confidence.py:114-170: per-(i, j) TM term and the per-row normalised sum on my rows (residue_weights / asym_id sliced to my rows "
                                      "where they index i, whole where they index j); per_alignment [N/P] all-gathered to [N]; the stock max over residues; in-program trees only"),
    "MaskedMsaHead/PredictedLDDTHead/ExperimentallyResolvedHead/confidence.compute_plddt": ("replicated_by_design(stock,untouched)",
                                                                                             "read msa [S,N,c_m] / the structure module's single act [N,c_s] / pLDDT logits "
                                                                                             "[N,bins] — not pair-shaped: the stated floor of the AF2 recipe"),
}
HEADS_HAZARDS = ("dispatch_on_rows_flag_only(caller_raises_it_inside_its_region;never_under_hk_init)",
                 "rows_gate(n_loc*P==N_at_trace;whole_pair_under_flag_refused)",
                 "ptm_default_residue_weights_over_keys(N=logits.shape[1])",
                 "host_confidence_untouched(use_jnp=False_is_stock)",
                 "confidence_rebound_on_module_object(modules.confidence_name_is_the_callers)",
                 "closure_state_per_install(patches.restore()_is_the_uninstall)")
OUTPUTS_ROWS: Dict[str, Tuple[str, ...]] = {                             # the program-output leaves that LEAVE ROW-SHARDED per confidence location ('/'-joined result keys)
    "in_graph": ("distogram/logits", "aligned_confidence_probs", "predicted_aligned_error", "pae_matrix_with_logits/logits"),
    "host": ("distogram/logits", "predicted_aligned_error/logits"),
}
_MARK = "_rowpair_af2_heads_body"                                       # attribute set on every rebound body: a second install onto patched classes is refused by name
_LAST: Dict[str, Any] = {"sites": [], "confidence": "not_installed", "library": {}, "n_gpu": 1, "axis": None}


def confidence_module(modules: Any, lever: str = FAMILY_LEVER) -> Any:
    """The ``<package>.common.confidence`` module of the package ``modules`` (``<package>.model.modules``) belongs to — the module OBJECT whose functions an
    in-program tree's ``AlphaFold.__call__`` runs (``get_confidence_metrics`` resolves ``predicted_tm_score`` / ``compute_predicted_aligned_error`` in that
    module's namespace at call time, so rebinding them there reaches the program). Imported by name, not read off ``modules.confidence`` (the caller may have
    replaced that name); refused by name when ``modules`` is not a ``….model.modules`` or the import fails."""
    name = str(getattr(modules, "__name__", ""))
    if not name.endswith(MODULES_SUFFIX):
        raise MemLeverRefused(lever, f"install_heads: {name or modules!r} is not a '<package>{MODULES_SUFFIX}' module — cannot locate its common.confidence")
    target = name[: -len(MODULES_SUFFIX)] + ".common.confidence"
    try:
        mod = importlib.import_module(target)
    except ImportError as e:
        raise MemLeverRefused(lever, f"install_heads: {target} not importable ({e})") from None
    bound = vars(modules).get("confidence")
    if bound is not None and str(getattr(bound, "__name__", target)) != target:
        raise MemLeverRefused(lever, f"install_heads: {name}.confidence names {getattr(bound, '__name__', bound)!s}, not {target} — the tree differs from the pin's")
    return mod


def rows_gate(block: Any, n_gpu: int, lever: str, site: str) -> int:
    """``N/P`` of a pair-rows block ``[N/P, N, …]`` under the installed ``n_gpu``; anything else under the rows flag is refused by name AT TRACE (a whole
    ``[N, N, …]`` pair handed to a rows body would be computed replicated on every device — the silent-fallback class)."""
    shp = tuple(int(s) for s in getattr(block, "shape", ()))
    if len(shp) < 2 or shp[0] * int(n_gpu) != shp[1]:
        raise MemLeverRefused(lever, f"{site}: rows are active but the pair block {shp} is not [N/P, N, ...] for n_gpu={int(n_gpu)} "
                                     f"(a whole pair under the rows flag would run replicated — refused)")
    return shp[0]


def describe_heads() -> Dict[str, Any]:
    """The last install's evidence words for the recipe's lever line: ``conf_sites`` (the rebound sites, ``,``-joined), ``confidence`` (``in_graph`` | ``host`` |
    ``not_installed`` — where the tree computes pLDDT / PAE / pTM), ``outputs_rows`` (the result leaves that leave the program row-sharded under conf=sharded,
    :data:`OUTPUTS_ROWS`), ``heads_library`` (``file:sha8`` of the pinned files found)."""
    where = _LAST["confidence"]
    return {"conf_sites": ",".join(_LAST["sites"]) or "none", "confidence": where,
            "outputs_rows": ",".join(OUTPUTS_ROWS.get(where, ())) or "none",
            "heads_library": ",".join(f"{k}:{v[:8]}" for k, v in sorted(_LAST["library"].items())) or "none"}


def install_heads(patches: Any, modules: Any, *, lever: str, rows_active: Callable[[], bool], axis: str, n_gpu: int) -> List[str]:
    """Rebind, into ``patches`` (the caller's :class:`opt_core.mem.patchset.PatchSet`; ``patches.restore()`` is the uninstall), the pair-reading heads of
    ``modules`` (``<package>.model.modules``, pinned) to bodies that run on this device's pair ROW BLOCK whenever ``rows_active()`` is true and call the STOCK
    body otherwise: ``DistogramHead.__call__`` and ``PredictedAlignedErrorHead.__call__`` through :func:`haiku.rebind` (Haiku's own method wrapper: the
    parameter tree is the stock's), and — only for a tree that computes the confidence metrics in the program (:data:`CONFIDENCE_IN_GRAPH`) —
    ``compute_predicted_aligned_error`` / ``predicted_tm_score`` ON the ``<package>.common.confidence`` module object (``patches.replace``). ``axis``: the mesh
    axis the caller's region maps; ``n_gpu``: P (the row-extent gate); ``lever``: the name on refusals. Returns the rebound site names (:data:`SITES` order).
    Refusals by name: ``modules`` lacking a class of :data:`REQUIRED`; bytes that are not a key of :data:`PIN_MODULES` (or, in-program, :data:`PIN_CONFIDENCE`);
    a tree whose ``confidence`` import disagrees with the pin table; a second install onto already-rebound classes; ``n_gpu < 1``; a non-callable flag."""
    lv = str(lever or FAMILY_LEVER)
    n = int(n_gpu)
    ax = str(axis)
    if n < 1:
        raise MemLeverRefused(lv, f"install_heads: n_gpu={n} — must be >= 1")
    if not callable(rows_active):
        raise MemLeverRefused(lv, "install_heads: rows_active must be a callable returning the caller's rows flag")
    mname = str(getattr(modules, "__name__", modules))
    missing = [nm for nm in REQUIRED if not hasattr(modules, nm)]
    if missing:
        raise MemLeverRefused(lv, f"install_heads: {mname} lacks {','.join(missing)}")
    sha = file_sha(modules, lv)
    if sha not in PIN_MODULES:
        raise MemLeverRefused(lv, f"install_heads: {mname} sha256 {sha[:12]} is not a transcribed tree ({'; '.join(sorted(set(PIN_MODULES.values())))})")
    if getattr(_hk.stock_body(modules.DistogramHead, "__call__"), _MARK, False):
        raise MemLeverRefused(lv, f"install_heads: {mname} heads are already rebound to the row bodies (restore the PatchSet that installed them first)")
    in_graph = bool(CONFIDENCE_IN_GRAPH[sha])
    imports_confidence = "confidence" in vars(modules)
    if imports_confidence != in_graph:
        raise MemLeverRefused(lv, f"install_heads: {mname} ({sha[:12]}) {'binds' if imports_confidence else 'does not bind'} the name 'confidence' but the pin table says "
                                  f"confidence is computed {'in the program' if in_graph else 'on the host'} — the tree differs from the pin's")
    library = {"modules": sha}
    conf_mod = None
    if in_graph:
        conf_mod = confidence_module(modules, lv)
        csha = file_sha(conf_mod, lv)
        if csha not in PIN_CONFIDENCE:
            raise MemLeverRefused(lv, f"install_heads: {conf_mod.__name__} sha256 {csha[:12]} is not a transcribed tree ({'; '.join(sorted(set(PIN_CONFIDENCE.values())))})")
        cmissing = [nm for nm in REQUIRED_CONFIDENCE if not hasattr(conf_mod, nm)]
        if cmissing:
            raise MemLeverRefused(lv, f"install_heads: {conf_mod.__name__} lacks {','.join(cmissing)}")
        library["confidence"] = csha
    M = modules
    stock: Dict[str, Any] = {}
    sites: List[str] = []

    # ---- DistogramHead.__call__ (2.3.x modules.py:1526-1549 = dl_binder_design modules.py:1367-1391) on this device's pair rows
    def distogram_call(self, representations, batch, is_training):
        """``representations['pair']``: ``[N/P, N, c_z]`` (my rows) under the flag → ``logits`` ``[N/P, N, bins]`` = my rows of the stock's symmetric logits;
        ``bin_edges`` as stock. Outside the flag: the stock body."""
        if not rows_active():
            return stock["DistogramHead"](self, representations, batch, is_training)
        jnp = _lazy.jnp(lv)
        pair = representations["pair"]
        rows_gate(pair, n, lv, "DistogramHead.__call__")
        half_logits = M.common_modules.Linear(self.config.num_bins, initializer=M.utils.final_init(self.global_config), name="half_logits")(pair)   # :1540-1543 / :1381-1385
        logits = _tr.symmetrize(half_logits, ax)                            # :1545 / :1387 `half_logits + jnp.swapaxes(half_logits, -2, -3)`: my rows of the transpose by one all_to_all
        breaks = jnp.linspace(self.config.first_break, self.config.last_break, self.config.num_bins - 1)                                    # :1546-1547 / :1388-1389
        return dict(logits=logits, bin_edges=breaks)
    setattr(distogram_call, _MARK, True)

    # ---- PredictedAlignedErrorHead.__call__ (2.3.x modules.py:1225-1250 = dl_binder_design modules.py:1138-1163): the stock body IS row-local
    def aligned_error_call(self, representations, batch, is_training):
        """The STOCK body (``Linear`` per ``(i, j)`` + ``breaks``): on the row block it returns the row block of the stock logits. Under the flag the row extent
        is gated first; nothing else differs."""
        if rows_active():
            rows_gate(representations["pair"], n, lv, "PredictedAlignedErrorHead.__call__")
        return stock["PredictedAlignedErrorHead"](self, representations, batch, is_training)
    setattr(aligned_error_call, _MARK, True)

    try:
        stock["DistogramHead"] = _hk.rebind(patches, M.DistogramHead, "__call__", distogram_call, lv)
        sites.append("DistogramHead.__call__")
        stock["PredictedAlignedErrorHead"] = _hk.rebind(patches, M.PredictedAlignedErrorHead, "__call__", aligned_error_call, lv)
        sites.append("PredictedAlignedErrorHead.__call__")
        if conf_mod is not None:
            C = conf_mod
            stock_cpae = C.compute_predicted_aligned_error
            stock_ptm = C.predicted_tm_score

            # ---- confidence.compute_predicted_aligned_error (2.3.x confidence.py:89-112): softmax + expectation per (i, j) — the stock body on the logits rows
            def compute_predicted_aligned_error(logits, breaks, use_jnp=False):
                """``logits`` ``[N/P, N, bins]`` under the flag → ``aligned_confidence_probs`` ``[N/P, N, bins]``, ``predicted_aligned_error`` ``[N/P, N]`` (my rows),
                ``max_predicted_aligned_error`` (scalar): the STOCK function; the row extent is gated first. Host calls (``use_jnp=False``) are never row calls."""
                if use_jnp and rows_active():
                    rows_gate(logits, n, lv, "confidence.compute_predicted_aligned_error")
                return stock_cpae(logits, breaks, use_jnp=use_jnp)
            setattr(compute_predicted_aligned_error, _MARK, True)

            # ---- confidence.predicted_tm_score (2.3.x confidence.py:114-170) on the logits rows: pTM (asym_id None) and ipTM (asym_id given)
            def predicted_tm_score(logits, breaks, residue_weights=None, asym_id=None, use_jnp=False):
                """``logits`` ``[N/P, N, bins]`` (my rows ``i``, every key ``j``); ``residue_weights`` ``[N]`` / ``asym_id`` ``[N]`` whole (replicated). The per-``(i, j)``
                TM term and the per-row normalisation (a sum over ``j``, complete on my rows) are the stock statements with ``residue_weights[:, None]`` /
                ``asym_id[:, None]`` (the ``i`` side) sliced to my rows; ``per_alignment`` ``[N/P]`` is all-gathered to ``[N]`` and the stock ``max`` follows. Returns the
                replicated scalar. Outside the flag or with ``use_jnp=False``: the stock function."""
                if not (use_jnp and rows_active()):
                    return stock_ptm(logits, breaks, residue_weights=residue_weights, asym_id=asym_id, use_jnp=use_jnp)
                jax = _lazy.jax(lv)
                jnp = _lazy.jnp(lv)
                n_loc = rows_gate(logits, n, lv, "confidence.predicted_tm_score")
                num_keys = int(logits.shape[1])                              # = N: the stock's logits.shape[0] on the whole map
                if residue_weights is None:
                    residue_weights = jnp.ones(num_keys)                     # :137-138 (the stock default spans every residue)
                if int(residue_weights.shape[0]) != num_keys:
                    raise MemLeverRefused(lv, f"confidence.predicted_tm_score: residue_weights has {int(residue_weights.shape[0])} entries for N={num_keys} keys")
                mine = _shard.local_block(residue_weights, ax, n_loc, dim=0)  # residue_weights[i] for my rows i
                bin_centers = C._calculate_bin_centers(breaks, use_jnp=True)   # :140 (the stock helper)
                clipped_num_res = jnp.maximum(residue_weights.sum(), 19)     # :144
                d0 = 1.24 * (clipped_num_res - 15) ** (1. / 3) - 1.8         # :149
                probs = jax.nn.softmax(logits, axis=-1)                      # :152
                tm_per_bin = 1. / (1 + jnp.square(bin_centers) / jnp.square(d0))   # :155
                predicted_tm_term = (probs * tm_per_bin).sum(-1)             # :157 — [N/P, N]
                if asym_id is None:
                    pair_mask = jnp.full((n_loc, num_keys), True)            # :160 (my rows of the all-True mask)
                else:
                    if int(asym_id.shape[0]) != num_keys:
                        raise MemLeverRefused(lv, f"confidence.predicted_tm_score: asym_id has {int(asym_id.shape[0])} entries for N={num_keys} keys")
                    pair_mask = _shard.local_block(asym_id, ax, n_loc, dim=0)[:, None] != asym_id[None, :]   # :162 with the i side sliced to my rows
                predicted_tm_term *= pair_mask                               # :164
                pair_residue_weights = pair_mask * (residue_weights[None, :] * mine[:, None])          # :166
                normed_residue_mask = pair_residue_weights / (1e-8 + pair_residue_weights.sum(-1, keepdims=True))   # :167 — the sum over j is complete per row
                per_alignment = (predicted_tm_term * normed_residue_mask).sum(-1)                      # :168 — [N/P]
                per_alignment = _shard.gather(per_alignment, ax, dim=0)      # [N] on every device (moves_bytes)
                return (per_alignment * residue_weights).max()              # :170
            setattr(predicted_tm_score, _MARK, True)

            patches.replace(C, "compute_predicted_aligned_error", compute_predicted_aligned_error)
            sites.append("confidence.compute_predicted_aligned_error")
            patches.replace(C, "predicted_tm_score", predicted_tm_score)
            sites.append("confidence.predicted_tm_score")
    except Exception:
        patches.restore()
        raise
    _LAST.update(sites=list(sites), confidence="in_graph" if in_graph else "host", library=library, n_gpu=n, axis=ax)
    return sites
