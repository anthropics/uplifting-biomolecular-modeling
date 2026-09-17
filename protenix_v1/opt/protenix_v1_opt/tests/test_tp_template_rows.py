"""Templated inputs under `--n_gpu P`: the template pair features are BORN AS ROWS per rank (tp.template_pair_rows) from the featuriser's
per-token precursors — never formed whole [T, N, N(, F)] on any rank — and those rows equal, bit for bit, the stock featuriser's dense
features (`Templates.as_protenix_dict`, protenix 1.1.0) sliced to the same rows: the conformance test of the row form. Also: install's
featuriser site emits precursors + the per-token census masks only; the entry seam (tp.host_side_inputs) bears this rank's rows from them and
the slot census of the item is unchanged and carries `form=row_born`."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
try:
    from protenix.data.template.template_featurizer import Templates       # noqa: E402
except Exception as _e:                                                  # named skip (a box without the stock wheel)
    pytest.skip(f"protenix not importable here: {_e!r}", allow_module_level=True)
pytest.importorskip("opt_core.testing")

from opt_core import testing as _T                                      # noqa: E402
from opt_core.mem.rowpair import dist as D                              # noqa: E402
from protenix_v1_opt import tp, templates                               # noqa: E402

PAIR_KEYS = ("template_distogram", "template_unit_vector", "template_pseudo_beta_mask", "template_backbone_frame_mask")


def _templates(T=4, N=203, seed=0):
    """A synthetic template stack exercising every statement: a chain-like CA trace (distances span the bins), ~8% atoms missing, GLY
    residues (pseudo-beta = CA), a gapped stretch, one all-dummy slot (gap restype, empty mask), two coincident residues (d2 = 0)."""
    rng = np.random.default_rng(seed)
    aatype = rng.integers(0, 21, size=(T, N)).astype(np.int64)
    aatype[:, ::17] = 7
    pos = (rng.normal(size=(T, N, 24, 3)) * 8.0).astype(np.float32)
    ca = np.cumsum(rng.normal(size=(T, N, 3)) * 2.2, axis=1).astype(np.float32)
    pos[:, :, 1, :] = ca
    pos[:, :, 0, :] = ca + rng.normal(size=(T, N, 3)).astype(np.float32) * 1.5
    pos[:, :, 2, :] = ca - rng.normal(size=(T, N, 3)).astype(np.float32) * 1.4
    pos[:, :, 4, :] = ca + rng.normal(size=(T, N, 3)).astype(np.float32) * 2.4
    mask = rng.random(size=(T, N, 24)) > 0.08
    mask[1, 50:80, :] = False
    mask[T - 1] = False
    aatype[T - 1] = 31
    pos[2, 10] = pos[2, 11]
    return Templates(aatype=aatype, atom_positions=pos, atom_mask=mask)


def _rank_bounds(N, P, chunk=4):
    """The kit's own partition of N rows over P ranks (tp.layout_for: the aligned policy at choose_align(N, P, chunk))."""
    align = tp.choose_align(N, P, chunk)
    return [D.Layout(N, P, q, B=align, align=align).bounds[q] for q in range(P)]


@pytest.mark.parametrize("N", [203, 640])
def test_template_pair_rows_equal_the_stock_dense_features_sliced(N):
    """C3 conformance: for every 64-grid block, the exact rank partitions of P in {2, 4, 8} and two off-grid blocks, the born rows are
    array_equal AND dtype-equal (float32) to `Templates.as_protenix_dict()` sliced to those rows, for all four keys."""
    tm = _templates(N=N, seed=N)
    dense = tm.as_protenix_dict()
    prec = tp.template_token_precursors(tm.aatype, tm.atom_positions, tm.atom_mask)
    blocks = [(0, N), (5, 131), (N - 3, N)] + [(r0, min(N, r0 + 64)) for r0 in range(0, N, 64)]
    for P in (2, 4, 8):
        blocks += _rank_bounds(N, P)
    checked = 0
    for g0, g1 in blocks:
        if g1 <= g0:
            continue
        rows = tp.template_pair_rows(prec, g0, g1)
        assert tuple(sorted(rows)) == tuple(sorted(PAIR_KEYS))
        for k in PAIR_KEYS:
            ref = dense[k][:, g0:g1]
            assert rows[k].dtype == ref.dtype == np.float32, (k, rows[k].dtype, ref.dtype)
            assert rows[k].shape == ref.shape, (k, rows[k].shape, ref.shape)
            assert np.array_equal(rows[k], ref), (k, g0, g1, float(np.abs(rows[k] - ref).max()))
        checked += 1
    assert checked >= 8
    assert 0.0 < float(dense["template_distogram"].sum()) and 0.0 < float(np.abs(dense["template_unit_vector"]).sum())   # the stack exercises the statements (not all-masked)
    masks = tp.template_token_masks(prec)                                  # the per-token census masks: their outer products are the dense 2-D masks
    for k in ("template_pseudo_beta_mask", "template_backbone_frame_mask"):
        assert masks[k].shape == (tm.aatype.shape[0], N) and masks[k].dtype == np.float32
        assert np.array_equal(masks[k][:, :, None] * masks[k][:, None, :], dense[k])


def test_the_featuriser_site_emits_precursors_and_token_masks_never_the_dense_pair_features():
    tm = _templates(N=131, seed=3)
    dense = Templates.as_protenix_dict(tm)
    site = tp.templates_as_precursors(Templates.as_protenix_dict)
    assert site.__wrapped__ is Templates.as_protenix_dict
    out = site(tm)
    assert set(out) == {"template_aatype", "template_atom_positions", "template_atom_mask", "template_pseudo_beta_mask", "template_backbone_frame_mask"}
    assert all(out[k].ndim <= 2 or k in ("template_atom_positions", "template_atom_mask") for k in out)     # nothing [T, N, N(, F)]
    assert out["template_atom_positions"].shape == (4, 131, 24, 3)
    c_dense = templates.census({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in dense.items()})
    c_prec = templates.census({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()})
    for f in ("key", "slots", "real", "dummy", "per_template"):           # the slot census of the item is the same either way
        assert c_dense[f] == c_prec[f], (f, c_dense[f], c_prec[f])
    assert c_prec["slots"] == 4 and c_prec["real"] == 3 and c_prec["dummy"] == 1


def _entry(rank, P, feats):
    """One rank's item entry: host_side_inputs bears this rank's template rows from the precursors; returns (bounds, rows, remaining keys)."""
    f = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats.items()}
    tp.host_side_inputs(f, None)
    host = tp.host_inputs()
    lay = tp.layout_for(int(f["token_index"].shape[-1]))
    return (lay.r0, lay.r1), {k: v.clone() for k, v in host["template_rows"].items()}, sorted(k for k in f if k.startswith("template_"))


@pytest.mark.parametrize("P", [2, 4])
def test_the_entry_seam_bears_this_ranks_rows_from_precursors_equal_to_dense_sliced(P):
    N = 203
    tm = _templates(N=N, seed=11)
    dense = tm.as_protenix_dict()
    prec_feats = tp.templates_as_precursors(Templates.as_protenix_dict)(tm)
    feats = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in prec_feats.items()}
    feats["token_index"] = torch.arange(N)
    feats["asym_id"] = torch.zeros(N, dtype=torch.long)
    c0 = tp.COUNTS["template_rows"]
    tp.STATE["chunk"] = 4
    outs = _T.run_ranks(P, _entry, feats, timeout_s=90.0)
    assert tp.COUNTS["template_rows"] - c0 == P
    seen = []
    for (g0, g1), rows, left in outs:
        assert left == ["template_aatype", "template_backbone_frame_mask", "template_pseudo_beta_mask"], left     # the atom coordinates left the item; aatype + per-token masks stay
        assert set(rows) == set(PAIR_KEYS)
        for k in PAIR_KEYS:
            ref = torch.from_numpy(np.ascontiguousarray(dense[k][:, g0:g1]))
            assert rows[k].dtype == torch.float32 and rows[k].shape == ref.shape, (k, rows[k].shape, ref.shape)
            assert torch.equal(rows[k].cpu(), ref), (k, g0, g1)
        seen.append((g0, g1))
    seen = sorted(seen)
    assert seen == sorted(_rank_bounds(N, P))                             # the ranks' rows are the layout's partition: they tile [0, N) exactly
    assert seen[0][0] == 0 and seen[-1][1] == N and all(a[1] == b[0] for a, b in zip(seen, seen[1:]))


def test_the_census_line_names_the_row_form_only_when_the_item_carries_precursors(capsys, monkeypatch):
    tm = _templates(N=131, seed=5)
    lines = []
    monkeypatch.setattr(templates.R, "log", lambda line: lines.append(line))
    cfg = {"use_template": True}
    prec = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in tp.templates_as_precursors(Templates.as_protenix_dict)(tm).items()}
    dense = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in Templates.as_protenix_dict(tm).items()}
    for feats in (prec, dense):
        feats["asym_id"] = torch.zeros(131, dtype=torch.long)
    templates.check_item(cfg, {"input_feature_dict": prec, "sample_name": "q_rows"})
    templates.check_item(cfg, {"input_feature_dict": dense, "sample_name": "q_dense"})
    rows_line = next(l for l in lines if "event=census" in l and "item=q_rows" in l)
    dense_line = next(l for l in lines if "event=census" in l and "item=q_dense" in l)
    assert " form=row_born " in rows_line and "slots=4 real=3 dummy=1 form=row_born per_chain=0:3" in rows_line, rows_line
    assert "form=" not in dense_line and "slots=4 real=3 dummy=1 per_chain=0:3" in dense_line, dense_line   # n_gpu 1 (the stock's dense features): the line is unchanged
