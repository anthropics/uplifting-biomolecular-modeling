"""The template feature under --n_gpu > 1: (1) templ.distogram_condition_rows over any row blocks is BITWISE the dense statement's rows
(upstream's featurize_noised_ground_truth_as_template_distogram from its noised centres on: nan matrix, cdist block of the filled tokens,
inter-molecule mask, bucketize, one-hot — transcribed below from rf3/data/ground_truth_template.py at the pin); (2) templ.distogram_condition_precursors
makes upstream's three random draws in upstream's order (the process's later draws are unchanged) and its precursors reproduce upstream's dense
outputs bitwise; (3) a dense feature reaching the row-sharded template embedder is refused by name."""
import sys
import types
import typing

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from .. import templ  # noqa: E402

EDGES = torch.concat((torch.arange(1.0, 4.0, 0.1), torch.arange(4.0, 20.5, 0.5)))      # rf3.data.ground_truth_template.DEFAULT_DISTOGRAM_BINS (63 edges, 64 bins)


def dense_from_precursors(centers, fill, molecule, edges):
    """Upstream's statements after its noised centres, verbatim in form, on the precursors (the reference the rows must equal)."""
    MASK_VALUE = float("nan")
    _n_token = int(centers.shape[0])
    token_to_fill_mask = fill.numpy()
    template_distogram = torch.full((_n_token, _n_token), fill_value=MASK_VALUE)
    token_idxs_to_fill = np.where(token_to_fill_mask)[0]
    ix1, ix2 = np.ix_(token_idxs_to_fill, token_idxs_to_fill)
    template_distogram[ix1.astype(int), ix2.astype(int)] = torch.cdist(centers[token_to_fill_mask], centers[token_to_fill_mask],
                                                                        compute_mode="donot_use_mm_for_euclid_dist")
    token_to_fill_mask_II = token_to_fill_mask[:, None] & token_to_fill_mask[None, :]
    mol = molecule.numpy()
    is_inter_molecule = mol[:, None] != mol                                             # ids all equal = upstream's draw said "provide inter-molecule distances"
    token_to_fill_mask_II[is_inter_molecule] = False
    template_distogram[is_inter_molecule] = MASK_VALUE
    template_distogram_binned = torch.bucketize(template_distogram, boundaries=edges)
    n_bins = len(edges) + 1
    onehot = torch.nn.functional.one_hot(template_distogram_binned, num_classes=n_bins).to(torch.float32)
    return onehot, torch.as_tensor(token_to_fill_mask_II, dtype=torch.bool)


def _blocks(n, P):
    """Row blocks covering [0, n): P even-ish shards, each cut again into two uneven pieces (empty pieces included)."""
    bounds = [round(i * n / P) for i in range(P + 1)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = a + (b - a) // 3
        out += [(a, m), (m, m), (m, b)]
    return out


@pytest.mark.parametrize("n,P,p_fill,n_mol,nan_unfilled", [(1, 2, 1.0, 1, False), (7, 3, 0.6, 2, True), (50, 2, 0.0, 1, False), (50, 4, 1.0, 3, False),
                                                            (64, 8, 0.5, 1, True), (333, 3, 0.8, 4, True), (129, 2, 0.3, 2, False)])
def test_rows_are_the_dense_statements_rows_bitwise(n, P, p_fill, n_mol, nan_unfilled):
    g = torch.Generator().manual_seed(1000 * n + P)
    centers = torch.randn(n, 3, generator=g) * 12.0
    centers[: max(1, n // 9)] = centers[0]                                               # coincident tokens: zero distances (below the first edge)
    fill = torch.rand(n, generator=g) < p_fill
    molecule = (torch.arange(n) * n_mol // max(n, 1)).to(torch.int64)
    if nan_unfilled:
        centers[~fill] = float("nan")                                                    # an unresolved token's coordinate: never filled, never read
        if int((~fill).sum()) == 0:
            centers[-1] = float("nan"); fill[-1] = False
    dense, has = dense_from_precursors(centers, fill, molecule, EDGES)
    assert dense.shape == (n, n, 64) and dense.dtype == torch.float32 and has.shape == (n, n) and has.dtype == torch.bool
    rows, hrows = [], []
    for g0, g1 in _blocks(n, P):
        c, h = templ.distogram_condition_rows(centers, fill, molecule, EDGES, g0, g1)
        assert c.shape == (g1 - g0, n, 64) and c.dtype == torch.float32 and h.dtype == torch.bool
        assert torch.equal(c, dense[g0:g1]) and torch.equal(h, has[g0:g1]), (n, P, g0, g1)
        rows.append(c); hrows.append(h)
    assert torch.equal(torch.cat(rows), dense) and torch.equal(torch.cat(hrows), has)
    if p_fill == 0.0:                                                                    # an untemplated query: every pair in the last (mask) bin, no condition
        assert bool((dense[..., -1] == 1).all()) and not bool(has.any())


# ---- upstream's featurizer, transcribed statement for statement (rf3/data/ground_truth_template.py @ 4010e3e, lines 265-384), over stub helpers
def _stock_featurize(atom_array, *, noise_scale, distogram_bins, allowed_chain_types, is_unconditional=True, p_condition_per_token=0.0,
                     p_provide_inter_molecule_distances=0.0, existing_annotation_to_check="is_input_file_templated"):
    M = sys.modules[_stock_featurize.__module__]
    MASK_VALUE = float("nan")
    _a_token_starts = M.get_token_starts(atom_array)
    _n_token = len(_a_token_starts)
    template_distogram = torch.full((_n_token, _n_token), fill_value=MASK_VALUE)
    noise_scale_t = typing.cast(torch.Tensor, noise_scale)
    noise = torch.normal(mean=0.0, std=1.0, size=(_n_token, 3)) * noise_scale_t.unsqueeze(-1)
    center_token_mask = M.get_af3_token_center_masks(atom_array)
    noisy_center_coords = torch.from_numpy(atom_array.coord[center_token_mask]) + noise
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
    token_idxs_to_fill = np.where(token_to_fill_mask)[0]
    ix1, ix2 = np.ix_(token_idxs_to_fill, token_idxs_to_fill)
    template_distogram[ix1.astype(int), ix2.astype(int)] = torch.cdist(noisy_center_coords[token_to_fill_mask], noisy_center_coords[token_to_fill_mask],
                                                                        compute_mode="donot_use_mm_for_euclid_dist")
    token_to_fill_mask_II = token_to_fill_mask[:, None] & token_to_fill_mask[None, :]
    if np.random.rand() > p_provide_inter_molecule_distances:
        is_inter_molecule = atom_array.molecule_id[center_token_mask][:, None] != atom_array.molecule_id[center_token_mask]
        token_to_fill_mask_II[is_inter_molecule] = False
        template_distogram[is_inter_molecule] = MASK_VALUE
    template_distogram_binned = torch.bucketize(template_distogram, boundaries=distogram_bins)
    n_bins = len(distogram_bins) + 1
    template_distogram_onehot = torch.nn.functional.one_hot(template_distogram_binned, num_classes=n_bins).to(torch.float32)
    expanded_noise_scale = noise_scale.expand(_n_token) if isinstance(noise_scale, torch.Tensor) else torch.full_like(noise, fill_value=noise_scale)
    expanded_noise_scale[~token_to_fill_mask] = 0.0
    out = {"distogram_condition_noise_scale": expanded_noise_scale,
           "has_distogram_condition": torch.as_tensor(token_to_fill_mask_II, dtype=torch.bool),
           "distogram_condition": template_distogram_onehot}
    M.assert_no_nans(out, msg="Conditioning features contain NaNs!")
    return out


class _Atoms:
    """A stand-in atom array: APT atoms per token, the token's centre = its first atom; annotations as upstream reads them."""
    APT = 3

    def __init__(self, n_token, g, chain_types, n_mol, unresolved=(), forced=None):
        n_atom = n_token * self.APT
        self.coord = (torch.randn(n_atom, 3, generator=g) * 9.0).numpy().astype(np.float32)
        self.chain_type = np.repeat(np.asarray(chain_types), -(-n_atom // len(chain_types)))[:n_atom]
        self.occupancy = np.ones(n_atom, dtype=np.float32)
        for t in unresolved:
            self.occupancy[t * self.APT] = 0.0
        self.molecule_id = (np.arange(n_atom) // self.APT) * n_mol // max(n_token, 1)
        self._ann = {} if forced is None else {"is_input_file_templated": np.repeat(np.asarray(forced, dtype=bool), self.APT)}

    def __len__(self):
        return len(self.coord)

    def get_annotation_categories(self):
        return list(self._ann)

    def get_annotation(self, name):
        return self._ann[name]


def _assert_no_nans(d, msg=""):
    for k, v in d.items():
        assert not bool(torch.isnan(v.to(torch.float64)).any()), (msg, k)


STUB = types.ModuleType("stub_ground_truth_template")
STUB.np, STUB.torch, STUB.Tensor, STUB.cast, STUB.assert_no_nans = np, torch, torch.Tensor, typing.cast, _assert_no_nans
STUB.get_token_starts = lambda a: np.arange(0, len(a), _Atoms.APT)
STUB.get_af3_token_center_masks = lambda a: (np.arange(len(a)) % _Atoms.APT) == 0
_stock_featurize.__module__ = STUB.__name__


def _rng_state():
    return torch.get_rng_state().clone(), np.random.get_state()[1].copy(), int(np.random.get_state()[2])


@pytest.mark.parametrize("case", [
    dict(n=40, uncond=False, p_cond=0.0, p_inter=0.0, forced=True, n_mol=2, unresolved=(3, 17)),     # the inference case: forced selection, inter-molecule pairs masked
    dict(n=40, uncond=False, p_cond=0.0, p_inter=1.0, forced=True, n_mol=3, unresolved=()),          # inter-molecule distances provided (the draw says so)
    dict(n=25, uncond=True, p_cond=0.0, p_inter=0.0, forced=True, n_mol=1, unresolved=(0,)),         # unconditional: nothing filled
    dict(n=61, uncond=False, p_cond=0.5, p_inter=0.5, forced=False, n_mol=2, unresolved=(5,)),       # the training-style draws (per-token coin)
    dict(n=1, uncond=False, p_cond=0.0, p_inter=0.0, forced=True, n_mol=1, unresolved=()),
])
def test_precursors_make_stocks_draws_in_stocks_order_and_reproduce_its_dense_outputs(case, monkeypatch):
    monkeypatch.setitem(sys.modules, STUB.__name__, STUB)
    n = case["n"]
    g = torch.Generator().manual_seed(7 + n)
    forced = ((torch.rand(n, generator=g) < 0.5).numpy() if case["forced"] else None)
    atoms = _Atoms(n, g, chain_types=[3, 6, 9], n_mol=case["n_mol"], unresolved=case["unresolved"], forced=forced)   # chain type 9 is not an allowed type
    kw = dict(distogram_bins=EDGES, allowed_chain_types=[3, 6], is_unconditional=case["uncond"], p_condition_per_token=case["p_cond"],
              p_provide_inter_molecule_distances=case["p_inter"])
    scale = torch.rand(n, generator=g) * 3.0 + 0.5                                      # the TokenGroupNoiseScaleSampler form: one scale per token

    torch.manual_seed(11); np.random.seed(11)
    ref = _stock_featurize(atoms, noise_scale=scale.clone(), **kw)
    ref_state = _rng_state()
    ref_next = (torch.rand(3), np.random.rand(3))

    torch.manual_seed(11); np.random.seed(11)
    pre = templ.distogram_condition_precursors(_stock_featurize, atoms, noise_scale=scale.clone(), **kw)
    state = _rng_state()
    nxt = (torch.rand(3), np.random.rand(3))

    assert set(pre) == {"distogram_condition_noise_scale", *templ.PRECURSORS} and not any(k in pre for k in templ.DENSE)
    assert torch.equal(ref_state[0], state[0]) and np.array_equal(ref_state[1], state[1]) and ref_state[2] == state[2]   # the generators are where stock leaves them
    assert torch.equal(ref_next[0], nxt[0]) and np.array_equal(ref_next[1], nxt[1])                                        # so the process's next draws are stock's
    assert torch.equal(pre["distogram_condition_noise_scale"], ref["distogram_condition_noise_scale"])
    p = templ.precursors_of(pre)
    assert p is not None and p["centers"].shape == (n, 3) and p["fill"].dtype == torch.bool and p["molecule"].dtype == torch.int64
    dense, has = dense_from_precursors(p["centers"], p["fill"], p["molecule"], p["edges"])
    assert torch.equal(dense, ref["distogram_condition"]) and torch.equal(has, ref["has_distogram_condition"])
    whole, whole_has = templ.distogram_condition_rows(p["centers"], p["fill"], p["molecule"], p["edges"], 0, n)
    assert torch.equal(whole, ref["distogram_condition"]) and torch.equal(whole_has, ref["has_distogram_condition"])
    if not case["uncond"] and case["forced"] and case["p_cond"] == 0.0:
        assert int(p["fill"].sum()) == int(np.asarray(forced, dtype=bool).sum())         # p_condition 0: exactly the forced (selected) tokens are filled


def test_precursors_of_is_none_without_every_key():
    f = {k: torch.zeros(2) for k in templ.PRECURSORS[:-1]}
    assert templ.precursors_of(f) is None
    f[templ.PRECURSORS[-1]] = EDGES
    assert set(templ.precursors_of(f)) == {"centers", "fill", "molecule", "edges"}


def test_a_dense_template_feature_at_the_row_sharded_embedder_is_refused_by_name():
    rowpair = pytest.importorskip("rosettafold3_opt.rowpair")
    try:
        RP = rowpair._rp()
    except Exception as e:                                                               # noqa: BLE001 — no core beside this tree: the refusal type is the core's
        pytest.skip(f"opt_core rowpair not importable here: {e}")
    f = {"distogram_condition": torch.zeros(4, 4, 64), "has_distogram_condition": torch.zeros(4, 4, dtype=torch.bool),
         "distogram_condition_noise_scale": torch.zeros(4)}
    with pytest.raises(RP.RowpairRefused, match="arrived dense"):
        rowpair._template_rows(None, f, torch.zeros(2, 4, 3), None)
    with pytest.raises(RP.RowpairRefused, match="neither the template distogram precursors"):
        rowpair._template_rows(None, {"distogram_condition_noise_scale": torch.zeros(4)}, torch.zeros(2, 4, 3), None)


def test_the_float_precursors_ride_the_batch_as_bit_patterns_an_amp_cast_cannot_alter(monkeypatch):
    """The engine casts the float features of the batch to its AMP dtype (bf16): stock's one-hot survives that exactly (0/1), a coordinate or a
    bin edge would not — so the two float precursors are integer tensors of the values' bit patterns; a bf16 cast of every FLOAT entry of the
    dict leaves the decoded precursors, hence the rows, bitwise what they were."""
    monkeypatch.setitem(sys.modules, STUB.__name__, STUB)
    n = 30
    g = torch.Generator().manual_seed(5)
    atoms = _Atoms(n, g, chain_types=[3], n_mol=2, forced=(torch.rand(n, generator=g) < 0.7).numpy())
    torch.manual_seed(3); np.random.seed(3)
    pre = templ.distogram_condition_precursors(_stock_featurize, atoms, noise_scale=torch.full((n,), 1.5), distogram_bins=EDGES, allowed_chain_types=[3],
                                               is_unconditional=False)
    assert pre["distogram_condition_centers"].dtype == torch.int32 and pre["distogram_condition_edges"].dtype == torch.int32
    cast = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in pre.items()}     # what an AMP cast of the dict does
    a, b = templ.precursors_of(pre), templ.precursors_of(cast)
    assert a["centers"].dtype == torch.float32 and torch.equal(a["centers"], b["centers"]) and torch.equal(a["edges"], EDGES) and torch.equal(b["edges"], EDGES)
    ra = templ.distogram_condition_rows(a["centers"], a["fill"], a["molecule"], a["edges"], 0, n)
    rb = templ.distogram_condition_rows(b["centers"], b["fill"], b["molecule"], b["edges"], 7, 19)
    assert torch.equal(ra[0][7:19], rb[0]) and torch.equal(ra[1][7:19], rb[1])
    f64 = torch.randn(4, 3, dtype=torch.float64)
    assert templ.float_bits(f64).dtype == torch.int64 and torch.equal(templ.bits_float(templ.float_bits(f64)), f64)   # a float64 coordinate array keeps its width
