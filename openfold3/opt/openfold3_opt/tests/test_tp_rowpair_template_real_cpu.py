"""REAL template slots on the tp line (a templated chain consumed under ``--upstream-fix OF3-001`` at ``--n_gpu P``), CPU threaded ranks:
the lazy featurizer (``tp_rowpair.data.featurize_template_structures_lazy``) against the STOCK openfold3 0.4.1 featurizer
``featurize_template_structures_of3`` on one synthetic templated item (two 24-residue chains; chain A carries two distinct template hits, chain B
one partial hit -> slots 0 and 1 real, 2 and 3 the predict-time dummies):
 * without a template-consuming upstream fix in the process the real hits are refused by name (``DataRefused``), never served as dummies;
 * under the consume policy the token-level features are the stock ones and the per-token precursors (``data.REAL_KEYS``) restate the stock
   featurizer's operands (float64 pseudo-beta coordinates and squared bin edges as int64 bit patterns, finite fp32 frames, its chain ids);
 * the COMPUTED row accessor (``template.template_rows`` -> ``opt_core...real_template_rows``) is ``torch.equal`` the stock DENSE
   ``template_distogram`` / ``template_unit_vector`` sliced to rows, for several slot sets and row ranges on every rank of P in ``P_CASES``;
 * the sharded template embedder on the computed rows vs the stock ``TemplateEmbedderAllAtom`` on the dense batch: fp32 ``max|diff| <= 1e-5``,
   schedule census ``templ_mode=computed``, slot groups ``0/1/2+3`` (the two dummies embedded once);
 * an UNTEMPLATED item featurizes to exactly the placeholder key set without the policy (untemplated tp runs unchanged) and, under the policy, to
   the same values plus the precursor keys with a ZERO real flag — one batch schema per run policy on every rank — served as dummy rows;
 * the launcher's rank command carries ``--upstream-fix`` and ``--use-templates`` to every rank (``tp.launch``), and the data seams reach the
   featurizing DataLoader workers by fork inheritance (no upstream 0.4.1 source names a multiprocessing context or start method; fork is this
   interpreter's effective start method) — the two facts the consume policy travels on.
Helpers come from ``test_tp_rowpair_msa_cpu`` / ``test_tp_rowpair_template_cpu`` (one definition)."""

import math
import types

import pytest

from openfold3_opt.tests.test_tp_rowpair_msa_cpu import (CHUNK, KW, P_CASES, PAIRSTACK_MODES, TOL32, HAVE_TORCH, PairGuard, diff, dims, layout, run,
                                                          use_pairstack)
from openfold3_opt.tests.test_tp_rowpair_template_cpu import build_template_embedder, needs_stack

if HAVE_TORCH:
    import torch

T_SLOTS, MIN_BIN, MAX_BIN, N_BINS = 4, 3.25, 50.75, 39        # OpenFold3's predict-time template settings: 4 slots, distogram of 39 bins over 3.25..50.75 A
N_RES = 24                                                    # residues per query chain (two chains -> N = 48 tokens)
FIX = "OF3-001"


# ------------------------------------------------------------------------------------------------------------- the synthetic item ----
def query_atom_array(n_res_a=N_RES, n_res_b=N_RES):
    """Two peptide chains of glycine-like residues (N, CA, C, O) with the token-level annotations openfold3's featurizers read."""
    import numpy as np
    import biotite.structure as struc
    atoms, res_id = [], 0
    for chain, nres in (("A", n_res_a), ("B", n_res_b)):
        for r in range(nres):
            res_id += 1
            for k, name in enumerate(("N", "CA", "C", "O")):
                atoms.append(struc.Atom([float(res_id), float(k), 0.0], chain_id=chain, res_id=r + 1, res_name="GLY", atom_name=name, element=name[0]))
    arr = struc.array(atoms)
    n_tok = n_res_a + n_res_b
    arr.set_annotation("token_id", np.repeat(np.arange(n_tok), 4).astype(np.int32))
    for name, val in (("is_atomized", False), ("hetero", False)):
        arr.set_annotation(name, np.full(arr.array_length(), val))
    arr.set_annotation("entity_id", np.where(arr.chain_id == "A", 1, 2).astype(np.int32))
    arr.set_annotation("mol_type", np.full(arr.array_length(), "protein"))
    return arr, n_tok


def helix_template(n_res=N_RES, radius=2.3, rise=1.5):
    """A synthetic ALA alpha-helix (N, CA, C, O, CB per residue: non-degenerate backbone frames, distinct pseudo-beta positions)."""
    import numpy as np
    import biotite.structure as struc
    atoms = []
    for r in range(n_res):
        phi = math.radians(100.0 * r)
        ca = np.array([radius * math.cos(phi), radius * math.sin(phi), rise * r])
        radial, tang, up = np.array([math.cos(phi), math.sin(phi), 0.0]), np.array([-math.sin(phi), math.cos(phi), 0.0]), np.array([0.0, 0.0, 1.0])
        pos = {"N": ca - 0.9 * tang - 0.8 * up + 0.3 * radial, "CA": ca, "C": ca + 1.0 * tang + 0.7 * up + 0.2 * radial,
               "O": ca + 1.3 * tang + 1.7 * up, "CB": ca + 1.2 * radial - 0.5 * up + 0.6 * tang}
        for name in ("N", "CA", "C", "O", "CB"):
            atoms.append(struc.Atom(pos[name].tolist(), chain_id="T", res_id=r + 1, res_name="ALA", atom_name=name, element=name[0]))
    return struc.array(atoms)


def templated_item(templated=True):
    """(atom_array, N, TemplateSliceCollection): chain A aligned over its full length to TWO distinct helices (slots 0, 1), chain B's residues
    3..20 to one (slot 0; NaN rows inside a real slot); ``templated=False``: no slices (every slot a dummy — the untemplated predict case)."""
    import numpy as np
    from openfold3.core.data.primitives.structure.template import TemplateSliceCollection
    arr, n_tok = query_atom_array()
    if not templated:
        return arr, n_tok, TemplateSliceCollection(template_slices={})
    h1, h2 = helix_template(), helix_template(radius=2.6, rise=1.45)
    sl = lambda t, q0, n: types.SimpleNamespace(atom_array=t, template_residue_repeats=np.ones(n, dtype=int), query_token_positions=np.arange(q0, q0 + n))
    sub = h1[(h1.res_id >= 4) & (h1.res_id <= 21)]
    tsc = TemplateSliceCollection(template_slices={"A": [sl(h1, 0, N_RES), sl(h2, 0, N_RES)], "B": [sl(sub, N_RES + 3, 18)]})
    return arr, n_tok, tsc


def stock_featurizer():
    """OpenFold3's own ``featurize_template_structures_of3`` (the statement under any rebinding this process may carry)."""
    from openfold3.core.data.pipelines.featurization import template as FT
    f = FT.featurize_template_structures_of3
    return getattr(f, "__wrapped__", f)


def featurize_both(monkeypatch, templated=True):
    """(atom_array, N, dense stock features, lazy features under templ_policy=consume(OF3-001))."""
    from openfold3_opt import templ_guard
    from openfold3_opt.tp_rowpair import data as DATA
    monkeypatch.setenv("ROWPAIR_VERBOSE", "0")
    arr, N, tsc = templated_item(templated)
    dense = stock_featurizer()(arr, tsc, T_SLOTS, N, MIN_BIN, MAX_BIN, N_BINS)
    monkeypatch.setattr(templ_guard, "_CONSUMED_BY", FIX)                     # what cli.apply_upstream_fix -> templ_guard.consumed_by records in a rank
    lazy = DATA.featurize_template_structures_lazy(arr, tsc, T_SLOTS, N, MIN_BIN, MAX_BIN, N_BINS)
    return arr, N, dense, lazy


def model_batch(feats, N):
    """The features as the model sees them (leading batch dim; token_mask; the model's asym_id, here the featurizer's partition)."""
    b = {k: v.unsqueeze(0) for k, v in feats.items()}
    b["token_mask"] = torch.ones(1, N)
    b["asym_id"] = torch.tensor([1.0] * N_RES + [2.0] * (N - N_RES))[None]
    return b


# ------------------------------------------------------------------------------------------------------------------ the featurizer ----
@needs_stack
def test_real_hits_without_the_fix_are_refused_by_name(monkeypatch):
    from openfold3_opt import templ_guard
    from openfold3_opt.tp_rowpair import data as DATA
    monkeypatch.setattr(templ_guard, "_CONSUMED_BY", None)                    # every default run: templ_policy=ignore(upstream_0.4.1)
    monkeypatch.setenv("ROWPAIR_VERBOSE", "0")
    arr, N, tsc = templated_item()
    with pytest.raises(DATA.DataRefused) as ei:
        DATA.featurize_template_structures_lazy(arr, tsc, T_SLOTS, N, MIN_BIN, MAX_BIN, N_BINS)
    msg = str(ei.value)
    assert "real template hits" in msg and "no template-consuming upstream fix" in msg and "--upstream-fix OF3-001" in msg and "never runs a real template as a dummy" in msg, msg
    arr, N, empty = templated_item(templated=False)                            # the untemplated case is served, policy or not
    out = DATA.featurize_template_structures_lazy(arr, empty, T_SLOTS, N, MIN_BIN, MAX_BIN, N_BINS)
    assert DATA.REAL_FLAG not in out and DATA.LAZY_FLAG in out


@needs_stack
def test_precursors_restate_the_stock_featurizers_operands(monkeypatch):
    import numpy as np
    from openfold3_opt.tp_rowpair import data as DATA
    arr, N, dense, lazy = featurize_both(monkeypatch)
    for k in ("template_pseudo_beta_mask", "template_backbone_frame_mask", "template_restype"):
        assert torch.equal(lazy[k], dense[k]), k
    assert tuple(dense["template_distogram"].shape) == (T_SLOTS, N, N, N_BINS) and tuple(lazy["template_distogram"].shape) == (T_SLOTS, 1, 1, N_BINS)
    assert tuple(lazy["template_unit_vector"].shape) == (T_SLOTS, 1, 1, 3) and bool(lazy[DATA.LAZY_FLAG].any()) and bool(lazy[DATA.REAL_FLAG].any())
    assert set(lazy) == set(dense) | {DATA.LAZY_FLAG, DATA.REAL_FLAG, *DATA.REAL_KEYS}, sorted(lazy)
    pb = lazy[DATA.PB_COORDS_KEY]
    assert pb.dtype == torch.int64 and tuple(pb.shape) == (T_SLOTS, N, 3)
    present = ~torch.isnan(pb.view(torch.float64)).any(dim=-1)
    assert torch.equal(present.float(), dense["template_pseudo_beta_mask"])              # NaN pattern == the stock pseudo-beta mask
    assert int(present[0].sum()) == N_RES + 18 and int(present[1].sum()) == N_RES and int(present[2:].sum()) == 0, present.sum(dim=-1)
    fr = lazy[DATA.FRAMES_KEY]
    assert fr.dtype == torch.float32 and tuple(fr.shape) == (T_SLOTS, N, 3, 3) and bool(torch.isfinite(fr).all())
    assert torch.equal((fr.abs().sum(dim=(-2, -1)) != 0).float(), dense["template_backbone_frame_mask"])
    asym = lazy[DATA.ASYM_KEY]
    assert asym.dtype == torch.int32 and asym.tolist() == [1] * N_RES + [2] * (N - N_RES)
    e = lazy[DATA.EDGES_KEY]
    assert e.dtype == torch.int64 and tuple(e.shape) == (2, N_BINS)
    lower = np.linspace(MIN_BIN, MAX_BIN, N_BINS) ** 2
    upper = np.concatenate([lower[1:], np.array([1e8])])
    assert np.array_equal(e.view(torch.float64).numpy(), np.stack([lower, upper]))          # bit-exact: the same numpy statements
    words = DATA.real_slot_words(lazy)
    assert words == f"slots 2/{T_SLOTS} real (t0: tokens {N_RES + 18}/{N}; t1: tokens {N_RES}/{N})", words


@needs_stack
def test_untemplated_item_featurizes_as_before(monkeypatch):
    """No real slot. Without the policy: exactly the placeholder key set. Under the consume policy: the same values on those keys plus the precursor
    keys with a ZERO real flag (one batch schema per run policy on every rank, whatever a rank's own featurization met), and the row accessor
    serves DUMMY rows (``templ_mode=lazy_dummy``), never the computed ones."""
    from openfold3_opt import templ_guard
    from openfold3_opt.tp_rowpair import data as DATA, template as TEMPL
    arr, N, dense, lazy = featurize_both(monkeypatch, templated=False)
    monkeypatch.setattr(templ_guard, "_CONSUMED_BY", None)
    plain = DATA.featurize_template_structures_lazy(arr, templated_item(templated=False)[2], T_SLOTS, N, MIN_BIN, MAX_BIN, N_BINS)   # policy-free featurization of the same item
    placeholder_keys = {"template_pseudo_beta_mask", "template_backbone_frame_mask", "template_restype", "template_distogram", "template_unit_vector", DATA.LAZY_FLAG}
    assert set(plain) == placeholder_keys, sorted(plain)
    assert set(lazy) == placeholder_keys | {DATA.REAL_FLAG, *DATA.REAL_KEYS}, sorted(lazy)
    for k in plain:
        assert torch.equal(lazy[k], plain[k]) and lazy[k].dtype == plain[k].dtype, k
    assert not bool(lazy[DATA.REAL_FLAG].any()) and not bool(dense["template_pseudo_beta_mask"].any())
    b = model_batch(lazy, N)
    assert not TEMPL.has_real_precursors(b)
    acc = TEMPL.template_rows(b, layout(N, 2, 0))
    assert acc.mode == "lazy_dummy", acc.mode


# --------------------------------------------------------------------------------------------------------------- the row accessor ----
@needs_stack
@pytest.mark.parametrize("P", P_CASES)
def test_computed_rows_equal_the_dense_features_sliced(P, monkeypatch):
    from openfold3_opt.tp_rowpair import template as TEMPL
    arr, N, dense, lazy = featurize_both(monkeypatch)
    bl, bd = model_batch(lazy, N), model_batch(dense, N)

    def rank_fn(rank, P_):
        from opt_core.mem.rowpair.evidence import schedule
        lay = layout(N, P_, rank)
        acc = TEMPL.template_rows(bl, lay)
        worst, equal, checked = 0.0, True, 0
        for slots in ([0], [1], [3], [0, 1, 2, 3], [2, 0]):
            for g0, g1 in ((lay.r0, lay.r1), (lay.r0, min(lay.r0 + 1, lay.r1)), (max(lay.r0, lay.r1 - 3), lay.r1)):
                if g1 <= g0:
                    continue
                for key in ("template_distogram", "template_unit_vector"):
                    got = acc.rows(key, slots, g0 - lay.r0, g1 - lay.r0)
                    ref = bd[key][:, slots, g0:g1]
                    equal = equal and got.dtype == ref.dtype and tuple(got.shape) == tuple(ref.shape) and torch.equal(got.cpu(), ref)
                    worst = max(worst, float((got.cpu().double() - ref.double()).abs().max())) if got.numel() else worst
                    checked += 1
        nnz = int((acc.rows("template_distogram", [0, 1], 0, lay.R) != 0).sum())
        return {"rank": rank, "mode": acc.mode, "equal": equal, "worst": worst, "checked": checked, "nnz01": nnz, "templ_mode": schedule().get("templ_mode")}

    total_nnz = 0
    for r in run(P, rank_fn):
        print(P, r)
        assert r["mode"] == "computed" and r["templ_mode"] == "computed", r
        assert r["equal"] and r["worst"] == 0.0 and r["checked"] > 0, r
        total_nnz += r["nnz01"]
    assert total_nnz == int((dense["template_distogram"][[0, 1]] != 0).sum()) > 0, total_nnz     # the ranks' rows tile the dense real slots


@needs_stack
def test_real_flag_without_precursors_is_refused_by_name(monkeypatch):
    from openfold3_opt.tp_rowpair import data as DATA, template as TEMPL
    arr, N, dense, lazy = featurize_both(monkeypatch)
    bl = model_batch(lazy, N)
    del bl[DATA.FRAMES_KEY]
    with pytest.raises(TEMPL.TemplateRefused) as ei:
        TEMPL.template_rows(bl, layout(N, 2, 0))
    assert "lacks their precursors" in str(ei.value) and DATA.FRAMES_KEY in str(ei.value), str(ei.value)


# --------------------------------------------------------------------------------------------------------- the template embedder ----
@needs_stack
@pytest.mark.parametrize("pairstack", PAIRSTACK_MODES)
@pytest.mark.parametrize("P", P_CASES)
def test_template_embedder_on_computed_rows_vs_stock(P, pairstack, monkeypatch):
    from openfold3_opt.tp_rowpair import template as TEMPL
    arr, N, dense, lazy = featurize_both(monkeypatch)
    use_pairstack(monkeypatch, pairstack)
    te = build_template_embedder()
    bl, bd = model_batch(lazy, N), model_batch(dense, N)
    g = torch.Generator().manual_seed(5)
    z0 = torch.randn(1, N, N, dims()[1], generator=g) * 0.5
    pair_mask = bd["token_mask"][..., None] * bd["token_mask"][..., None, :]
    with torch.no_grad():
        t_ref = z0 + te(batch=bd, z=z0.clone(), pair_mask=pair_mask, chunk_size=CHUNK, _mask_trans=True, inplace_safe=True, **KW)
        t_none = z0 + te(batch=bd_untemplated(N), z=z0.clone(), pair_mask=pair_mask, chunk_size=CHUNK, _mask_trans=True, inplace_safe=True, **KW)
    assert float((t_ref - t_none).abs().max()) > 1e-3                          # the real templates move the stock embedder's output (the test item is not degenerate)

    def rank_fn(rank, P_):
        from opt_core.mem.rowpair.evidence import schedule
        lay = layout(N, P_, rank)
        z_loc = z0[:, lay.r0:lay.r1].clone().contiguous()
        pm_loc = pair_mask[:, lay.r0:lay.r1].contiguous()
        with torch.no_grad(), PairGuard(N, int(z0.shape[-1])) as guard:
            out = TEMPL.template_embedder_add_rows_(te, bl, z_loc, pm_loc, lay, census_tag="cycle0", chunk_size=CHUNK, _mask_trans=True, inplace_safe=True, **KW)
        sched = schedule()
        return {"rank": rank, "z": diff(out, t_ref[:, lay.r0:lay.r1]), "same_object": out is z_loc, "pair_guard_hits": guard.hits if pairstack == "sharded" else [],
                "templ_groups": sched.get("templ_groups"), "templ_mode": sched.get("templ_mode"), "templ_event": sched.get("templ_event")}

    for r in run(P, rank_fn):
        print(P, pairstack, r["rank"], r["z"], "groups", r["templ_groups"], "mode", r["templ_mode"])
        assert r["z"]["max_abs_diff"] <= TOL32 and r["same_object"], (P, pairstack, r)
        assert not r["pair_guard_hits"], (r["rank"], r["pair_guard_hits"][:3])
        assert r["templ_mode"] == "computed" and r["templ_groups"] == "0/1/2+3" and r["templ_event"] is None, r


def bd_untemplated(N):
    """The stock dense batch of the UNTEMPLATED item (four dummy slots)."""
    arr, n_tok, empty = templated_item(templated=False)
    return model_batch(stock_featurizer()(arr, empty, T_SLOTS, N, MIN_BIN, MAX_BIN, N_BINS), N)


# --------------------------------------------------------------------------------------------------------------------- the launcher ----
def test_launcher_hands_the_fix_and_the_template_word_to_every_rank():
    """OF3-001 applies per PROCESS and ``--use-templates`` is upstream's per-process flag: ``tp.launch`` composes every rank's command line with
    both words (each rank's ``cli.cmd_pred`` then applies the fix and sets the consume policy before its DataLoader forks)."""
    import inspect
    from openfold3_opt import cli, tp
    src = inspect.getsource(tp.launch)
    assert '("--use-templates", a.use_templates)' in src, "tp.launch: the rank command no longer forwards --use-templates"
    assert 'sub += ["--upstream-fix", a.upstream_fix]' in src, "tp.launch: the rank command no longer forwards --upstream-fix"
    assert cli.UPSTREAM_FIX_FLAG == "--upstream-fix"


def test_data_seams_reach_the_featurizing_workers_by_fork():
    """The lazy featurizer and the consume policy are installed / set in the RANK process and reach upstream's DataLoader workers by fork
    inheritance: no openfold3 0.4.1 source (``stock/src/openfold3``, its tests aside) names a DataLoader ``multiprocessing_context``, a
    ``set_start_method`` / ``get_context`` call or the ``forkserver`` / ``spawn`` start methods, and fork is this interpreter's EFFECTIVE start method
    (``multiprocessing.get_start_method()``: what a DataLoader with no context uses). A stock or interpreter change of either premise fails here by
    name (a worker started as a fresh interpreter would featurize with the stock statements: dense ``[T, N, N, F]`` features, real hits not consumed)."""
    import multiprocessing
    import os
    import re
    from openfold3_opt.tests import _stubs
    src = os.path.join(_stubs.tree_home(), "stock", "src", "openfold3")
    if not os.path.isdir(src):
        pytest.skip(f"stock source not in this tree: {src}")
    pat = re.compile(r"multiprocessing_context|set_start_method|get_context\(|forkserver|\bspawn\b")
    hits = []
    for root, _dirs, files in os.walk(src):
        if os.sep + "tests" in root[len(src):]:
            continue
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                for i, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
                    if pat.search(line):
                        hits.append(f"{os.path.relpath(path, src)}:{i}: {line.strip()[:120]}")
    assert not hits, "openfold3 sources name a worker start method / multiprocessing context — the tp line's data seams no longer reach the DataLoader workers by fork: " + "; ".join(hits[:10])
    method = multiprocessing.get_start_method(allow_none=True) or multiprocessing.get_context().get_start_method()
    assert method == "fork", f"effective start method {method!r}: DataLoader workers would be fresh interpreters (the rank's data seams and consume policy would not reach them)"
