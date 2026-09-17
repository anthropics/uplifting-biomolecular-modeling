"""The template census: a declared template selection is a named event, folded as upstream folds it.

RF3 has no template slots: its template track is ground-truth distogram conditioning of the INPUT structure's own selection
(``template_selection`` on the input / the CLI → ``rf3.utils.inference.apply_template_selection`` marks the selected atoms in the
annotation ``is_input_file_templated`` → the featurizer ``rf3.data.ground_truth_template`` turns the marked tokens into
``has_distogram_condition`` / ``distogram_condition`` → ``RF3TemplateEmbedder``). Upstream, a declared selection that matches no atom is
an INFO log line ("Selected 0 atoms …") and the fold proceeds with ``has_distogram_condition`` all-False. This module wraps that one
function when its module is imported (no stock byte is edited; the stock call runs first and its result is returned unchanged): every
declared selection is a NAMED event on stderr with its census (``TEMPLATE selection=<k syntaxes> atoms_selected=<n>/<N>``), so a selection
that selected nothing is visible as ``atoms_selected=0/<N>``. An item that declares no selection is untouched (the stock no-op path). The
census of the process is the exit tally's ``template`` block.

Under ``--n_gpu P`` (``rowpair`` installed) the template FEATURE is born as rows: upstream's featurizer
(``featurize_noised_ground_truth_as_template_distogram``: one dense ``[N, N, 64]`` fp32 one-hot ``distogram_condition`` + ``[N, N]``
``has_distogram_condition`` per query, templated or not) is rebound to :func:`distogram_condition_precursors` — its statements up to the pair
allocations, the three random draws in stock order — which hands the model per-token precursors instead (the noised centre coordinates,
the fill mask, the molecule ids, the bin edges; ``distogram_condition_noise_scale`` as stock); the row-sharded template embedder builds
GLOBAL pair rows ``[g0, g1)`` of the two dense tensors from them with :func:`distogram_condition_rows`, bitwise equal to the dense
statement's rows (tests/test_templ_rows.py). No rank forms an ``[N, N]`` template tensor; the one-GPU line keeps upstream's featurizer.
"""
from __future__ import annotations

import sys
from typing import Optional

from . import report as _report

MODULE = "rf3.utils.inference"                  # the upstream module whose function marks the selected atoms
FUNCTION = "apply_template_selection"           # (atom_array, template_selection) -> atom_array; annotation ANNOTATION
ANNOTATION = "is_input_file_templated"
STATE: dict = {"installed": False, "declared": 0, "items": []}
FEATURIZER_MODULE = "rf3.data.ground_truth_template"   # upstream's noised-template featurizer (the Transform of the inference pipeline calls it by this module-level name)
FEATURIZER = "featurize_noised_ground_truth_as_template_distogram"
DENSE = ("distogram_condition", "has_distogram_condition")                      # its two [N, N, ...] outputs
PRECURSORS = ("distogram_condition_centers", "distogram_condition_fill",        # their per-token precursors under --n_gpu > 1 (rowpair): noised centre
              "distogram_condition_molecule", "distogram_condition_edges")     # coordinates [N, 3], fill mask [N], molecule ids [N], bin edges [63] — the two
                                                                                # float ones as the integer BIT PATTERNS of their values (the engine casts the
                                                                                # float features of the batch to its AMP dtype; an integer tensor passes as is)
_BITS_OF = {"torch.float32": "int32", "torch.float64": "int64"}                 # float dtype -> the integer dtype of the same width
_FLOAT_OF = {"torch.int32": "float32", "torch.int64": "float64"}


def _as_list(sel) -> list:
    if sel is None:
        return []
    if isinstance(sel, str):
        return [sel] if sel.strip() else []
    return [s for s in list(sel) if s]


def census(atom_array, template_selection) -> Optional[dict]:
    """``{syntaxes, atoms_selected, atoms}`` for a declared selection on the returned atom array; None when nothing was declared."""
    sel = _as_list(template_selection)
    if not sel:
        return None
    try:
        ann = atom_array.get_annotation(ANNOTATION)
        n, total = int(ann.astype(bool).sum()), int(len(ann))
    except Exception:                                # noqa: BLE001 — an atom array without the annotation after the stock call: 0 selected, named below
        n, total = 0, int(len(atom_array)) if hasattr(atom_array, "__len__") else 0
    return {"syntaxes": len(sel), "selection": list(sel), "atoms_selected": n, "atoms": total}


def line(c: dict) -> str:
    return f"{_report.PREFIX} TEMPLATE selection={c['syntaxes']} atoms_selected={c['atoms_selected']}/{c['atoms']}"


def gate(orig):
    """The wrapped ``apply_template_selection``: the stock call, then the census line; the stock result is returned unchanged."""
    def apply_template_selection(atom_array, template_selection):
        out = orig(atom_array, template_selection)
        c = census(out, template_selection)
        if c is None:
            return out
        STATE["declared"] += 1
        STATE["items"].append({k: c[k] for k in ("syntaxes", "atoms_selected", "atoms")})
        del STATE["items"][:-32]
        print(line(c), file=sys.stderr, flush=True)
        return out
    apply_template_selection.__wrapped__ = orig
    apply_template_selection._rosettafold3_opt_gate = True
    return apply_template_selection


def wrap(module) -> bool:
    """Install the gate on the imported module; False when the function is absent (named in the activation report) or already wrapped."""
    fn = getattr(module, FUNCTION, None)
    if fn is None or getattr(fn, "_rosettafold3_opt_gate", False):
        return False
    setattr(module, FUNCTION, gate(fn))
    STATE["installed"] = True
    return True


def state() -> dict:
    """The exit tally's ``template`` block: installed, items that declared a selection, the last censuses."""
    return {"installed": STATE["installed"], "declared": STATE["declared"], "items": list(STATE["items"])}


def distogram_condition_precursors(stock, atom_array, *, noise_scale, distogram_bins, allowed_chain_types, is_unconditional: bool = True,
                                   p_condition_per_token: float = 0.0, p_provide_inter_molecule_distances: float = 0.0,
                                   existing_annotation_to_check: str = "is_input_file_templated") -> dict:
    """Upstream's ``featurize_noised_ground_truth_as_template_distogram`` (``stock``, rebound to this under ``--n_gpu > 1``) statement for
    statement up to its ``[N, N]`` allocations — the per-token noise draw (``torch.normal``), the noised centre coordinates, the fill mask
    (supported chain type & resolved centre atom & finite noise; the per-token ``np.random.rand(N)`` draw unless unconditional; the forced
    annotation), the one inter-molecule ``np.random.rand()`` draw — and returns per-token PRECURSORS of its two pair tensors in their place
    (``PRECURSORS``; ``distogram_condition_noise_scale`` exactly as stock). The three draws happen in stock's order, so every later draw of
    the process is the one stock makes. The model's row-sharded template embedder builds the pair ROWS from these
    (:func:`distogram_condition_rows`); the helpers (token starts, centre masks, ``np``/``torch``, ``assert_no_nans``) are the stock module's own."""
    M = sys.modules[stock.__module__]
    np, torch = M.np, M.torch
    _a_token_starts = M.get_token_starts(atom_array)                       # [n_token] (int)
    _n_token = len(_a_token_starts)
    noise_scale_t = M.cast(M.Tensor, noise_scale)
    noise = torch.normal(mean=0.0, std=1.0, size=(_n_token, 3)) * noise_scale_t.unsqueeze(-1)
    center_token_mask = M.get_af3_token_center_masks(atom_array)           # [n_atom] (bool)
    noisy_center_coords = torch.from_numpy(atom_array.coord[center_token_mask]) + noise   # [n_token, 3]
    tokens_with_supported_chain_types_mask = np.isin(atom_array.chain_type[center_token_mask], allowed_chain_types or [])
    resolved_tokens_mask = atom_array.occupancy[center_token_mask] > 0
    token_to_fill_mask = tokens_with_supported_chain_types_mask & resolved_tokens_mask & torch.isfinite(noise).all(dim=-1).numpy()
    if existing_annotation_to_check and existing_annotation_to_check in atom_array.get_annotation_categories():
        forced_template_mask = np.asarray(atom_array.get_annotation(existing_annotation_to_check)[center_token_mask], dtype=bool)
    else:
        forced_template_mask = np.full(_n_token, False, dtype=bool)
    if is_unconditional:
        token_to_fill_mask = np.full_like(token_to_fill_mask, False)
    else:
        _should_apply_condition = np.random.rand(_n_token) < p_condition_per_token
        token_to_fill_mask = (token_to_fill_mask & _should_apply_condition) | forced_template_mask
    mask_inter_molecule = np.random.rand() > p_provide_inter_molecule_distances           # stock's one draw (after its distances, before its binning: no draw between)
    if mask_inter_molecule:                                                                 # pairs of different molecules masked: the ids decide, row by row
        molecule = np.asarray(atom_array.molecule_id[center_token_mask]).astype(np.int64)
    else:                                                                                   # inter-molecule distances provided: one id for all (no pair differs)
        molecule = np.zeros(_n_token, dtype=np.int64)
    expanded_noise_scale = noise_scale.expand(_n_token) if isinstance(noise_scale, M.Tensor) else torch.full_like(noise, fill_value=noise_scale)
    expanded_noise_scale[~token_to_fill_mask] = 0.0
    out = {
        "distogram_condition_noise_scale": expanded_noise_scale,                             # (n_token,)  — stock's
        "distogram_condition_centers": float_bits(noisy_center_coords),                     # (n_token, 3) float bits
        "distogram_condition_fill": torch.as_tensor(token_to_fill_mask, dtype=torch.bool),  # (n_token,)
        "distogram_condition_molecule": torch.as_tensor(molecule),                          # (n_token,) int64
        "distogram_condition_edges": float_bits(distogram_bins),                            # (n_bins - 1,) float bits
    }
    M.assert_no_nans({k: v for k, v in out.items() if k != "distogram_condition_centers"}, msg="Conditioning features contain NaNs!")   # stock's check on stock's outputs' precursors; a centre is NaN exactly where stock's noisy coordinate is (an unresolved token, never filled)
    return out


def float_bits(t):
    """A float tensor's values as the integer tensor of the same width holding their bit patterns (a dtype cast of the feature dict —
    the engine's AMP cast of its float features — cannot alter them; :func:`bits_float` is the inverse, bit for bit)."""
    torch = sys.modules["torch"]
    return t.contiguous().view(getattr(torch, _BITS_OF[str(t.dtype)]))


def bits_float(t):
    """The float tensor whose bit patterns an integer tensor holds (:func:`float_bits`); a float tensor is returned as it is."""
    torch = sys.modules["torch"]
    return t.contiguous().view(getattr(torch, _FLOAT_OF[str(t.dtype)])) if str(t.dtype) in _FLOAT_OF else t


def precursors_of(f: dict) -> Optional[dict]:
    """The precursor tensors of a feature dict (``PRECURSORS`` keys → centers, fill, molecule, edges; the float ones decoded from their bit
    patterns), or None when any is absent."""
    vals = [f.get(k) for k in PRECURSORS]
    if any(v is None for v in vals):
        return None
    centers, fill, molecule, edges = vals
    return {"centers": bits_float(centers), "fill": fill, "molecule": molecule, "edges": bits_float(edges)}


def distogram_condition_rows(centers, fill, molecule, edges, g0: int, g1: int):
    """GLOBAL pair rows ``[g0, g1)`` of upstream's ``distogram_condition`` (``[rows, N, n_bins]`` fp32 one-hot) and ``has_distogram_condition``
    (``[rows, N]`` bool) from the precursors, by the dense statement's own steps on the row slab: distances ``nan`` everywhere, the filled
    rows × filled columns block from ``torch.cdist(…, compute_mode="donot_use_mm_for_euclid_dist")`` (each entry is the same reduction as the
    dense call's), the pairs of different molecule ids masked (``has`` False, distance ``nan``), ``torch.bucketize`` against the edges (``nan``
    → the last bin), one-hot, fp32. CPU tensors in, CPU tensors out (the dense statement runs on the host; the caller moves the rows)."""
    torch = sys.modules["torch"]
    g0, g1 = int(g0), int(g1)
    n = int(centers.shape[0])
    dist = torch.full((g1 - g0, n), float("nan"))
    rows_fill = fill[g0:g1]
    ri = rows_fill.nonzero()[:, 0]
    ci = fill.nonzero()[:, 0]
    if len(ri) and len(ci):
        dist[ri[:, None], ci[None, :]] = torch.cdist(centers[g0:g1][rows_fill], centers[fill], compute_mode="donot_use_mm_for_euclid_dist")
    has = rows_fill[:, None] & fill[None, :]
    inter = molecule[g0:g1][:, None] != molecule[None, :]
    has[inter] = False
    dist[inter] = float("nan")
    binned = torch.bucketize(dist, boundaries=edges)
    return torch.nn.functional.one_hot(binned, num_classes=int(edges.shape[0]) + 1).to(torch.float32), has
