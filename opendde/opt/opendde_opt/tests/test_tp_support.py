"""Row-born input features under `--n_gpu P>1` (opendde_opt/tp_feats.py): the featurizer's template pair features for THIS RANK'S ROWS equal
the stock's dense features sliced to those rows — element for element (the stock `Templates.as_opendde_dict` is the reference) — at P in {2, 3}
including ragged N; the rows of all ranks tile N; the marker is the trunk's layout (one function); no rank allocates an [*, N, N(, *)] array;
the dense dummy-template path is refused by name.

Input-feature sharding under `--n_gpu P`: the [T, N, N(, *)] template pair features never reach a rank's card whole (host-resident under
the runner's to_device; this rank's rows move at trunk entry), and the kit's ran-or-refuse registry names the levers the row-sharded line
serves itself (diffz -> replaced_by=rowpair_tp) instead of reading them as never-ran.

`tp._summary_rank0` (the `--n_gpu P>1` confidence finish: summary / full_data assembled from the core's row-block reducer statistics) equals
the stock `sample_confidence._compute_full_data_and_summary` on the same DENSE logits, key by key — the stock function is the reference and is
CALLED here (it needs whole `[N, N, bins]` logits, which no rank holds under `--n_gpu P>1`; hence the mirror). Runs wherever the stock
`opendde` package and torch import (the kit's CPU lane with the stock wheel, every GPU box); skipped BY NAME otherwise."""
import tracemalloc
import types

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from opendde_opt import tp, tp_feats

try:
    import opendde.model.sample_confidence as _SC_MODULE       # captured now (collection time, before any test body runs) so a later test's
except ImportError:                                            # module-level stub install (test_tp_seams_cpu.py's _tp_stub_engine, which
    _SC_MODULE = None                                           # only patches sys.modules from inside a test/fixture body) cannot shadow it


def _TF():
    """The stock featurizer (rdkit, biotite) is the dense reference: a per-test guard, not a module-wide one — the other tests in this file
    (tp_inputs', tp_summary_mirror's) do not all share this specific stock dependency."""
    return pytest.importorskip("opendde.data.template.template_featurizer", reason="the stock featurizer (rdkit, biotite) is the dense reference")


def _templates(N: int, T: int = 3, seed: int = 0):
    """Random template precursors in token space: two 'real' slots (random coordinates / partial atom masks, residue types incl. non-standard)
    and one dummy slot (the stock's empty template: gap type 31, zero masks)."""
    TF = _TF()
    rng = np.random.default_rng(seed)
    aatype = rng.integers(0, 21, size=(T, N)).astype(np.int64)
    pos = (rng.normal(size=(T, N, 24, 3)) * 8.0).astype(np.float32)
    mask = (rng.random((T, N, 24)) > 0.15).astype(np.float32)
    aatype[-1, :] = 31; pos[-1] = 0.0; mask[-1] = 0.0                            # dummy slot as the stock assembles it
    return TF.Templates(aatype=aatype, atom_positions=pos, atom_mask=mask)


def _rank_env(monkeypatch, P, r):
    from opt_core.mem.rowpair import launch
    monkeypatch.setenv(launch.ENV_WORLD, str(P)); monkeypatch.setenv(launch.ENV_RANK, str(r))


@pytest.mark.parametrize("geom", [(37, 2), (100, 3), (64, 2), (90, 3)], ids=["N37_P2_ragged", "N100_P3_R48_48_4", "N64_P2", "N90_P3_ragged"])   # (61, 3) has no valid row grid: the core refuses it by name for the whole run
def test_row_born_template_pair_features_equal_the_stock_dense_features(monkeypatch, geom):
    N, P = geom
    templ = _templates(N)
    dense = templ.as_opendde_dict()                                              # the stock statement (this process is no rank): [T, N, N(, F)]
    assert dense["template_distogram"].shape == (3, N, N, 39)
    rows_fn = tp_feats._make_as_opendde_dict_rows(_TF().Templates.as_opendde_dict)
    cover = np.zeros(N, dtype=int)
    for r in range(P):
        _rank_env(monkeypatch, P, r)
        assert tp.in_rank_process() and tp.rank() == r and tp.rank_world() == P
        lay = tp.layout_of(N, P, r)
        out = rows_fn(templ)
        r0, R, n = (int(x) for x in out[tp_feats.ROWS_KEY])
        assert (r0, R, n) == (lay.r0, lay.R, N)                                   # the marker IS the trunk's layout
        cover[r0:r0 + R] += 1
        for k in tp_feats.PAIR_KEYS:
            got, ref = out[k], dense[k][:, r0:r0 + R]
            assert got.shape == ref.shape and got.dtype == ref.dtype, (k, got.shape, ref.shape, got.dtype, ref.dtype)
            md = float(np.abs(got.astype(np.float64) - ref.astype(np.float64)).max()) if got.size else 0.0
            print(f"FEATS N={N} P={P} rank={r} rows=[{r0},{r0 + R}) {k} max|diff|={md} equal={np.array_equal(got, ref)}")
            assert np.array_equal(got, ref), (k, md)                              # element for element (max|diff| reported: 0.0)
        for k in ("template_aatype", "template_atom_positions", "template_atom_mask"):
            assert np.array_equal(out[k], dense[k])                              # the O(T·N) keys are the stock's
        assert bool((out["template_distogram"] != 0).any()) and bool((out["template_unit_vector"] != 0).any())   # real slots exercised
        assert not bool((out["template_distogram"][-1] != 0).any())              # the dummy slot's rows are zeros, as the stock's
    assert (cover == 1).all(), cover                                             # the ranks' rows tile N exactly once (ragged last rank included)


def test_no_rank_allocates_an_N_by_N_input_array(monkeypatch):
    """Allocation guard: the row-born featurizer's peak traced allocation is linear in R·N (≤ 2.5 × the born rows' own bytes) and below HALF of
    ONE dense pair key, and no returned array has two axes of length N (the dense statement allocates T·N²·39·4 bytes for the distogram alone)."""
    N, P, r = 256, 8, 3
    templ = _templates(N)
    _rank_env(monkeypatch, P, r)
    rows_fn = tp_feats._make_as_opendde_dict_rows(_TF().Templates.as_opendde_dict)
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = rows_fn(templ)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    lay = tp.layout_of(N, P, r)
    dense_dgram_bytes = 3 * N * N * 39 * 4
    rows_bytes = tp_feats.rows_pair_bytes(3, lay.R, N)
    print(f"ALLOC peak={peak} rows_bytes={rows_bytes} dense_distogram_bytes={dense_dgram_bytes} R={lay.R} N={N}")
    assert peak <= 2.5 * rows_bytes, (peak, rows_bytes)                            # linear in the rows: the slabs born + row-sized temporaries
    assert peak < dense_dgram_bytes // 2, (peak, dense_dgram_bytes)             # a dense [T, N, N, 39] never existed
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            assert sum(1 for d in v.shape if d == N) <= 1, (k, v.shape)          # at most one N-length axis: rows, never [*, N, N(, *)]
    tracemalloc.start(); tracemalloc.reset_peak()
    templ.as_opendde_dict.__func__(templ) if hasattr(templ.as_opendde_dict, "__func__") else _TF().Templates.as_opendde_dict(templ)
    _, peak_dense = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"ALLOC dense peak={peak_dense}")
    assert peak_dense >= dense_dgram_bytes                                       # the guard sees the dense statement's allocation (the instrument works)


def test_outside_a_rank_process_the_featurizer_is_the_stock_statement(monkeypatch):
    from opt_core.mem.rowpair import launch
    monkeypatch.delenv(launch.ENV_WORLD, raising=False); monkeypatch.delenv(launch.ENV_RANK, raising=False)
    templ = _templates(20)
    out = tp_feats._make_as_opendde_dict_rows(_TF().Templates.as_opendde_dict)(templ)
    assert tp_feats.ROWS_KEY not in out and out["template_distogram"].shape == (3, 20, 20, 39)


def test_dense_template_dummy_path_is_refused_by_name_in_a_rank_process(monkeypatch):
    from opt_core.mem.rowpair import RowpairRefused
    U = pytest.importorskip("opendde.data.utils")
    _rank_env(monkeypatch, 2, 0)
    wrapped = tp_feats._make_dummy_feature_refusing(U.make_dummy_feature)
    with pytest.raises(RowpairRefused, match="make_dummy_feature"):
        wrapped(features_dict={}, dummy_feats=["template"])


from opendde_opt import ran, registry, tp, tp_feats


def _feats(T=2, N=48, S=5):
    g = torch.Generator().manual_seed(0)
    return {"template_distogram": torch.rand(T, N, N, 39, generator=g), "template_unit_vector": torch.rand(T, N, N, 3, generator=g),
            "template_pseudo_beta_mask": torch.ones(T, N, N), "template_backbone_frame_mask": torch.ones(T, N, N),
            "template_aatype": torch.zeros(T, N, dtype=torch.long), "msa": torch.zeros(S, N, dtype=torch.long), "token_bonds": torch.zeros(N, N)}


def _stock_to_device(obj, device, non_blocking=False):                 # == opendde.utils.torch_utils.to_device (a copy container, tensors moved)
    if isinstance(obj, dict):
        return {k: _stock_to_device(v, device, non_blocking) if isinstance(v, (dict, torch.Tensor)) else v for k, v in obj.items()}
    return obj.to(device=device, non_blocking=non_blocking)


def _lay():
    from opt_core.mem.rowpair.dist import Layout
    return Layout.checked(48, 2, 1, B=16, lever="test")                     # rank 1 of 2 over 48 tokens: rows [32, 48) (grid B=16: 2 blocks + 1)


def _born(lay, T=2, N=48, S=5):
    """The feature dict as the row-born featurizer (tp_feats) hands it on: pair keys [T, R, N(, F)] + the marker [r0, R, N]."""
    fd = _feats(T, N, S)
    for k in tp.TEMPLATE_PAIR_FEATS:
        fd[k] = fd[k][:, lay.r0:lay.r0 + lay.R].contiguous()
    fd[tp_feats.ROWS_KEY] = torch.tensor([lay.r0, lay.R, N])
    return fd


@pytest.mark.parametrize("nested", [True, False])
def test_row_born_template_feats_stay_on_the_host_and_move_as_rows(monkeypatch, nested):
    lay = _lay()
    monkeypatch.setattr(tp, "in_rank_process", lambda: True)
    monkeypatch.setattr(tp, "_layout", lambda N: lay)
    for k in ("feat_host_keys", "template_rows_born", "unsharded"):
        monkeypatch.setitem(tp.STATS, k, 0)
    fd = _born(lay)
    data = {"sample_name": "x", "input_feature_dict": fd} if nested else fd
    wrapped = tp._make_to_device_host_feats(_stock_to_device)
    out = wrapped(data, "cpu")
    ofd = out["input_feature_dict"] if nested else out
    R, r0 = lay.R, lay.r0
    assert ofd["template_distogram"].shape == (2, R, 48, 39) and ofd["template_pseudo_beta_mask"].shape == (2, R, 48)
    assert ofd[tp.HOST_FEATS_KEY] == ("rows", r0, R) and tp_feats.ROWS_KEY not in ofd and ofd["msa"] is not None and ofd["template_aatype"].shape == (2, 48)
    assert tp.STATS["feat_host_keys"] == 4 and tp.STATS["template_rows_born"] == 4 and tp.STATS["unsharded"] == 0
    assert tp.STATS["inputs_full_gb"] == round(tp_feats.full_pair_bytes(2, 48) / 1e9, 6) and tp.STATS["inputs_born_gb"] == round(tp_feats.rows_pair_bytes(2, R, 48) / 1e9, 6)
    again = wrapped(out, "cpu")                                                 # idempotent: held rows pass through
    afd = again["input_feature_dict"] if nested else again
    assert afd[tp.HOST_FEATS_KEY] == ("rows", r0, R) and afd["template_distogram"].shape == (2, R, 48, 39) and tp.STATS["feat_host_keys"] == 4
    feats = dict(afd)
    tp._slice_template_feats(feats, lay, device="cpu")                         # trunk entry: host rows -> the card as they are
    assert afd[tp.HOST_FEATS_KEY] == ("rows", r0, R)
    assert feats[tp.TEMPLATE_ROWS_KEY] is True and tp.HOST_FEATS_KEY not in feats and feats["template_distogram"].shape == (2, R, 48, 39)


def test_whole_template_feats_in_a_rank_process_are_refused_by_name(monkeypatch):
    """A [T, N, N, *] tensor built whole on a rank's host = the featurizer patch did not run: refused by name (never silently sliced);
    ROWPAIR_ALLOW_UNSHARDED=1 opts in to cutting it at the move — a named unsharded event."""
    from opt_core.mem.rowpair import RowpairRefused
    lay = _lay()
    monkeypatch.setattr(tp, "in_rank_process", lambda: True)
    monkeypatch.setattr(tp, "_layout", lambda N: lay)
    for k in ("feat_host_keys", "template_rows_born", "unsharded"):
        monkeypatch.setitem(tp.STATS, k, 0)
    monkeypatch.delenv(tp.ENV_ALLOW_UNSHARDED, raising=False)
    with pytest.raises(RowpairRefused, match="built WHOLE"):
        tp._make_to_device_host_feats(_stock_to_device)(_feats(), "cpu")
    monkeypatch.setenv(tp.ENV_ALLOW_UNSHARDED, "1")
    out = tp._make_to_device_host_feats(_stock_to_device)(_feats(), "cpu")
    assert out[tp.HOST_FEATS_KEY] == ("rows", lay.r0, lay.R) and out["template_distogram"].shape == (2, lay.R, 48, 39)
    assert tp.STATS["unsharded"] == 4 and tp.STATS["template_rows_born"] == 0                     # one named event per key


def test_row_born_feats_of_another_layout_are_refused_by_name(monkeypatch):
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair.dist import Layout
    lay, other = _lay(), Layout.checked(48, 2, 0, B=16, lever="test")
    monkeypatch.setattr(tp, "in_rank_process", lambda: True)
    monkeypatch.setattr(tp, "_layout", lambda N: lay)
    with pytest.raises(RowpairRefused, match="born for rows"):
        tp._make_to_device_host_feats(_stock_to_device)(_born(other), "cpu")


def test_host_rows_of_another_layout_are_refused_by_name(monkeypatch):
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair.dist import Layout
    lay = _lay()
    monkeypatch.setattr(tp, "in_rank_process", lambda: True)
    monkeypatch.setattr(tp, "_layout", lambda N: lay)
    out = tp._make_to_device_host_feats(_stock_to_device)(_born(lay), "cpu")
    other = Layout.checked(48, 2, 0, B=16, lever="test")                     # rank 0's rows [0, 32)
    with pytest.raises(RowpairRefused, match="host template rows"):
        tp._slice_template_feats(out, other, device="cpu")


def test_to_device_is_the_stock_statement_outside_a_rank_process(monkeypatch):
    monkeypatch.setattr(tp, "in_rank_process", lambda: False)
    fd = _feats()
    out = tp._make_to_device_host_feats(_stock_to_device)({"input_feature_dict": fd}, "cpu")
    assert tp.HOST_FEATS_KEY not in out["input_feature_dict"] and out["input_feature_dict"]["template_distogram"].shape == (2, 48, 48, 39)


def test_trunk_entry_cuts_whole_template_feats_to_this_ranks_rows(monkeypatch):
    """A caller that bypassed the runner (whole [T, N, N, *] tensors in the dict): the trunk entry cuts them to the rows."""
    lay = _lay()
    fd = _feats()
    monkeypatch.setitem(tp.STATS, "template_feats_rows", 0)
    tp._slice_template_feats(fd, lay, device="cpu")
    R, r0 = lay.R, lay.r0
    assert fd["template_distogram"].shape == (2, R, 48, 39) and fd["template_backbone_frame_mask"].shape == (2, R, 48)
    assert torch.equal(fd["template_unit_vector"], _feats()["template_unit_vector"][:, r0:r0 + R])
    assert fd[tp.TEMPLATE_ROWS_KEY] is True and fd["template_aatype"].shape == (2, 48)
    tp._slice_template_feats(fd, lay, device="cpu")                       # idempotent
    assert fd["template_distogram"].shape == (2, R, 48, 39) and tp.STATS["template_feats_rows"] == 4


def test_diffz_is_replaced_by_rowpair_under_n_gpu_gt_1_not_never_ran():
    for lever in ("diffz", "bigln_guard"):
        engaged, why = ran.engagement(lever, {"n_gpu": 4})
        assert engaged is False and why.startswith("replaced_by=rowpair_tp:"), (lever, why)
        assert ran.engagement(lever, {"n_gpu": 1}) == (True, None)          # one card: the unit's own path, counted as before
        assert ran.engagement(lever, {}) == (True, None)                    # unknown facts never make a lever inert
    assert ran.engagement("rowpair_tp", {"n_gpu": 1})[0] is False and ran.engagement("rowpair_tp", {"n_gpu": 2}) == (True, None)
    assert isinstance(registry.ENGAGEMENT["diffz"].single_gpu, str)


def test_rowpair_predicate_is_the_conjunction_of_its_statements(monkeypatch):
    import sys, types
    from opendde_opt import tp_diffusion
    monkeypatch.setitem(tp.STATS, "presharded_calls", 7)
    monkeypatch.setitem(tp_diffusion.STATS, "zcond_rows", 3)
    monkeypatch.setitem(tp_diffusion.STATS, "diff_denoise_calls", 0)
    assert ran.count("rowpair_tp") == 0                                   # the roll-out never ran the sharded denoiser: never-ran, by name
    monkeypatch.setitem(tp_diffusion.STATS, "diff_denoise_calls", 200)
    assert ran.count("rowpair_tp") == 3


N, APT, NS, CBINS, PLBINS = 28, 3, 1, 64, 50


def _configs(need_full: bool):
    conf = types.SimpleNamespace(pae={"min_bin": 0.0, "max_bin": 32.0, "no_bins": CBINS}, pde={"min_bin": 0.0, "max_bin": 32.0, "no_bins": CBINS},
                                 plddt={"min_bin": 0.0, "max_bin": 1.0, "no_bins": PLBINS})
    metrics = types.SimpleNamespace(clash=types.SimpleNamespace(af3_clash_threshold=1.1, vdw_clash_threshold=0.5))
    return types.SimpleNamespace(confidence=conf, metrics=metrics, need_atom_confidence=need_full)


def _case(seed=0):
    g = torch.Generator().manual_seed(seed)
    n_atom = N * APT
    asym = (torch.arange(N) >= N // 3).long() + (torch.arange(N) >= 2 * N // 3).long()          # 3 chains
    has_frame = torch.rand(N, generator=g) > 0.15
    is_lig_atom = torch.zeros(n_atom, dtype=torch.long)
    is_lig_atom[(2 * N // 3) * APT:] = 1                                                        # chain 2 = ligand atoms
    return dict(pae=torch.randn(NS, N, N, CBINS, generator=g) * 2, pde=torch.randn(NS, N, N, CBINS, generator=g) * 2,
                plddt=torch.randn(NS, n_atom, PLBINS, generator=g), contact=torch.rand(N, N, generator=g),
                asym=asym, has_frame=has_frame, coords=torch.randn(NS, n_atom, 3, generator=g) * 6.0,
                a2t=torch.arange(N).repeat_interleave(APT), is_lig_atom=is_lig_atom)


@pytest.mark.parametrize("need_full", [False, True], ids=["summary", "summary+full_data"])
def test_summary_from_reducer_stats_equals_stock_compute_full_data_and_summary(need_full):
    if _SC_MODULE is None:
        pytest.skip("the stock opendde package is not importable here: the mirror test runs where the stock wheel is installed")
    SC = _SC_MODULE                                             # the module-level binding (collection time), immune to a later test's sys.modules stub-install
    if not hasattr(SC, "_compute_full_data_and_summary"):
        pytest.skip("opendde.model.sample_confidence resolved to something else entirely (neither the real stock package nor its usual shape)")
    pytest.importorskip("opt_core.mem.rowpair.confidence", reason="opt_core rowpair.confidence required")
    from opt_core.mem.rowpair import confidence as C
    from opendde_opt import tp
    cfg, c = _configs(need_full), _case(1)
    ref_s, ref_f = SC._compute_full_data_and_summary(configs=cfg, pae_logits=c["pae"], plddt_logits=c["plddt"], pde_logits=c["pde"], contact_probs=c["contact"],
                                                     token_asym_id=c["asym"], token_has_frame=c["has_frame"], atom_coordinate=c["coords"], atom_to_token_idx=c["a2t"],
                                                     atom_is_polymer=1 - c["is_lig_atom"], N_recycle=4, interested_atom_mask=None, mol_id=None, return_full_data=need_full)
    tok_is_lig = torch.zeros(N, dtype=torch.long).scatter_add(0, c["a2t"], c["is_lig_atom"]) > 0
    chains = C.ChainIndex(c["asym"], c["has_frame"], tok_is_lig)
    red = C.RowBlockReducer(chains, 0, N, c["pae"].device, pae_bins=(0.0, 32.0, CBINS), pde_bins=(0.0, 32.0, CBINS), finish="exact", allow_unsharded=True)
    for i0 in range(0, N, 8):                                                                   # row blocks, as the head produces them
        i1 = min(N, i0 + 8)
        red.consume(i0, i1, c["pae"][0, i0:i1], c["pde"][0, i0:i1], c["contact"][i0:i1])
    stats = red.finalize([(0, N)], collect_full=need_full)
    got_s, got_f = tp._summary_rank0(cfg, stats, red, plddt_logits=c["plddt"], token_asym_id=c["asym"], token_has_frame=c["has_frame"], atom_coordinate=c["coords"],
                                      atom_to_token_idx=c["a2t"], atom_is_polymer=1 - c["is_lig_atom"], N_recycle=4, return_full_data=need_full)
    assert len(ref_s) == len(got_s) == NS
    for rs, gs in zip(ref_s, got_s):
        assert sorted(rs) == sorted(gs), (sorted(set(rs) ^ set(gs)),)
        for k in rs:
            a, b = rs[k], gs[k]
            if torch.is_tensor(a):
                assert torch.is_tensor(b) and tuple(a.shape) == tuple(b.shape), (k, getattr(a, "shape", None), getattr(b, "shape", None))
                af, bf = a.float(), b.float()
                nan = torch.isnan(af)
                assert torch.equal(nan, torch.isnan(bf)), k                                    # chain pairs without frames are NaN in both
                assert torch.allclose(af[~nan], bf[~nan], rtol=1e-5, atol=1e-5), (k, float((af[~nan] - bf[~nan]).abs().max()))
            else:
                assert a == b, (k, a, b)
    if need_full:
        assert len(ref_f) == len(got_f) == NS
        for rf, gf in zip(ref_f, got_f):
            assert sorted(rf) == sorted(gf), (sorted(set(rf) ^ set(gf)),)
            for k in rf:
                a, b = rf[k].float(), gf[k].float()
                tol = 3e-2 if k in ("token_pair_pae", "token_pair_pde", "contact_probs") else 1e-5    # the reducer collects the [N, N] matrices in fp16 on the host (named precision)
                assert tuple(a.shape) == tuple(b.shape) and torch.allclose(a, b, rtol=0, atol=tol), (k, tuple(a.shape), float((a - b).abs().max()))
    else:
        assert got_f == [{}] and ref_f == [{}]


# ------------------------------------------------------- the cross-rank feature census (tp._feats_census): every rank's digest of its model inputs, compared by the core
def test_the_per_rank_feature_keys_are_named_once():
    """The keys left out of the cross-rank digest are exactly this rank's rows of the four template pair features plus the three row markers;
    the featurizer's (numpy-side) marker literal and the trunk's are one word; the LEVER field name is the core's census word."""
    from opt_core.mem.rowpair import rankdata
    assert tp.INPUT_ROWS_KEY == tp_feats.ROWS_KEY and tp_feats.PAIR_KEYS == tp.TEMPLATE_PAIR_FEATS
    assert tp.PER_RANK_FEATS == tp.TEMPLATE_PAIR_FEATS + (tp.INPUT_ROWS_KEY, tp.HOST_FEATS_KEY, tp.TEMPLATE_ROWS_KEY)
    assert (tp.FEATS_WHAT, tp.FEATS_EQUAL, tp.FEATS_DIGEST) == ("feats", "feats_ranks_equal", "feats_digest")
    assert rankdata.agree_word(tp.FEATS_WHAT, True) == f"{tp.FEATS_EQUAL}=yes"                     # the kit's LEVER field is the core's word
    assert {"feats_batches", "feats_unequal", tp.FEATS_EQUAL, tp.FEATS_DIGEST} <= set(tp.STATS)


def _replicated_feats(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"msa": torch.randint(0, 32, (4, 12), generator=g), "token_index": torch.arange(12), "ref_pos": torch.randn(30, 3, generator=g),
            "is_protein": torch.ones(30, dtype=torch.bool), "token_bonds": torch.zeros(12, 12)}


def test_feats_digest_is_the_cores_digest_of_the_replicated_keys_only():
    """feats_digest = the core's feature_digest over the dict minus PER_RANK_FEATS: a rank's own template rows / markers never move it, one
    element of a replicated feature does; dtype and shape count, not just bytes."""
    from opt_core.mem.rowpair import rankdata
    fd = _replicated_feats()
    d0 = tp.feats_digest(fd)
    assert d0 == rankdata.feature_digest(fd) == rankdata.feature_digest(fd, exclude=tp.PER_RANK_FEATS) and len(d0) == 64
    assert tp.feats_digest(_replicated_feats()) == d0                                                               # deterministic
    mine = dict(fd, template_distogram=torch.full((1, 6, 12, 39), 3.0), **{tp.INPUT_ROWS_KEY: torch.tensor([6, 6, 12]), tp.HOST_FEATS_KEY: ("rows", 6, 6), tp.TEMPLATE_ROWS_KEY: True})
    assert tp.feats_digest(mine) == d0                                                                              # this rank's rows and the markers are left out by name
    flipped = dict(fd, msa=fd["msa"].clone()); flipped["msa"][3, 11] += 1
    assert tp.feats_digest(flipped) != d0                                                                           # one element of a replicated feature
    assert tp.feats_digest(dict(fd, token_index=fd["token_index"].to(torch.int32))) != d0                           # same values, other dtype


def test_the_move_takes_the_census_once_per_fresh_batch_and_never_without_a_rank_world(monkeypatch, capsys):
    """The runner's patched to_device hands every FRESH feature dict to the census once (a dict it already moved passes uncounted); the census
    itself compares nothing and prints nothing in a process whose environment names no rank world."""
    seen = []
    monkeypatch.setattr(tp, "_feats_census", lambda fd: seen.append(dict(fd)))
    monkeypatch.setattr(tp, "in_rank_process", lambda: True)
    to_dev = tp._make_to_device_host_feats(lambda obj, device, non_blocking=False: dict(obj))
    fd = _replicated_feats()
    moved = to_dev({"input_feature_dict": fd, "N_token": torch.tensor([12])}, "cpu")
    assert len(seen) == 1 and set(seen[0]) == set(fd)                                          # the input_feature_dict, as the featurizer handed it on
    again = dict(fd, **{tp.HOST_FEATS_KEY: ("rows", 0, 6)})
    to_dev(again, "cpu")
    assert len(seen) == 1                                                                          # a dict this wrapper already marked: not a new batch
    monkeypatch.undo()
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    before = {k: tp.STATS[k] for k in ("feats_batches", "feats_unequal", tp.FEATS_EQUAL, tp.FEATS_DIGEST, "group")}
    tp._feats_census(fd)
    assert {k: tp.STATS[k] for k in before} == before and capsys.readouterr().err == ""          # rank_world() == 1: nothing compared, nothing printed, no group touched


def _census_rank_entry(flip: bool):
    """Rank body (gloo, CPU): the same replicated features on both ranks (+ this rank's own rows and marker), one element flipped on rank 1 when
    asked; the core's census line captured on this rank's stderr; this rank asserts its own verdict and returns the facts (rank 0's reach the test)."""
    import contextlib
    import io
    import re
    from opendde_opt import tp as _tp
    from opt_core.mem.rowpair import launch
    r = launch.rank()
    fd = _replicated_feats()
    fd["template_unit_vector"] = torch.full((1, 6, 12, 3), float(r))                            # this rank's rows: differ BY DESIGN, left out
    fd[_tp.INPUT_ROWS_KEY] = torch.tensor([6 * r, 6, 12])
    if flip and r == 1:
        fd["ref_pos"] = fd["ref_pos"].clone(); fd["ref_pos"][29, 2] += 1.0                       # one element of one replicated feature on rank 1 only
    from opt_core.mem.rowpair import RowpairRefused
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        if flip:
            with pytest.raises(RowpairRefused, match=r"refused: feats_ranks_differ: ") as ei:   # the census line first, then the same refusal on EVERY rank
                _tp._feats_census(fd)
            assert f"rank {r} digest " in str(ei.value), str(ei.value)          # the core names THIS rank and its digest in the refusal
        else:
            _tp._feats_census(fd)                                                                 # equal digests: the census line, no refusal
    lines = [ln for ln in buf.getvalue().splitlines() if "[feats]" in ln]
    assert len(lines) == 1, buf.getvalue()                                                        # ONE census line per batch per rank
    line = lines[0]
    expect = "no" if flip else "yes"
    m = re.match(r"^\[opendde-opt\] \[feats\] rank (\d) digest ([0-9a-f]{16}) feats_ranks_equal=(yes|no) ranks=2 digests=([0-9a-f]{16}),([0-9a-f]{16})$", line)
    assert m, line
    assert int(m.group(1)) == r and m.group(3) == expect and m.group(2) == (m.group(4), m.group(5))[r], line
    assert (m.group(4) == m.group(5)) is (not flip), line
    assert _tp.STATS[_tp.FEATS_EQUAL] == expect and _tp.STATS[_tp.FEATS_DIGEST] == m.group(2) and _tp.STATS["feats_batches"] == 1 and _tp.STATS["feats_unequal"] == int(flip)
    ev = dict(_tp.evidence_pairs())
    assert ev[_tp.FEATS_EQUAL] == expect and ev[_tp.FEATS_DIGEST] == m.group(2) and ev["feats_batches"] == 1 and ev["feats_excluded"] == ";".join(_tp.PER_RANK_FEATS)   # the LEVER census fields
    return {"rank": r, "equal": m.group(3), "digests": [m.group(4), m.group(5)], "line": line}


@pytest.mark.parametrize("flip", [False, True], ids=["replicated", "one_element_differs_on_rank1"])
def test_feature_census_across_two_ranks_names_agreement_and_refuses_disagreement_on_every_rank(flip):
    """Two CPU ranks (gloo): identical replicated features -> `feats_ranks_equal=yes` on both with one digest twice, no refusal; one element changed
    on rank 1 -> the census line reads `feats_ranks_equal=no` with two digests on BOTH ranks and BOTH raise the core's `refused: feats_ranks_differ`
    (rank 1 asserts its side in its own process)."""
    pytest.importorskip("opt_core.mem.rowpair.rankdata", reason="the pinned core is older than the rank-data surface these ranks use")
    from opendde_opt.tests.conftest import run_sharded_or_skip
    out = run_sharded_or_skip(2, _census_rank_entry, flip, mode="big", backend="gloo", cpu_ok=True, run_timeout_s=300)
    assert out["rank"] == 0 and out["equal"] == ("no" if flip else "yes") and (out["digests"][0] == out["digests"][1]) is (not flip), out
