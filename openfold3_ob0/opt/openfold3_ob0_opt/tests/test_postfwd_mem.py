"""The `postfwd_mem` cell (exact class): the runner's confidence scoring after the forward per (sample, row block). Block planning and the
switch words without an engine; the lean statement groups bit-for-bit against the engine's own functions on CPU tensors when the engine
and torch are importable (a GPU environment with the stack installed), skipped elsewhere."""
import os
import sys
import types

import pytest

from openfold3_ob0_opt import of3_postfwd as P
from openfold3_ob0_opt.cells import postfwd_mem as cell


def test_the_cell_binds_this_kits_switch_names_and_engine_modules():
    assert cell.ENV == "OPENFOLD3_OB0_OPT_POSTFWD_MEM" and cell.ENV_MIB == "OPENFOLD3_OB0_OPT_POSTFWD_MEM_MIB" and cell.STATE is P.STATE
    assert (P.M_ACR, P.M_SR, P.M_CONF, P.M_ATOMIZE) == ("openfold3.core.metrics.aggregate_confidence_ranking", "openfold3.core.metrics.sample_ranking",
                                                       "openfold3.core.metrics.confidence", "openfold3.core.utils.atomize_utils")
    assert P.PREFIX == "[openfold3_ob0-opt/postfwd_mem]" and P.BLOCK_MIB == 256 and P.MIN_ROWS == 16


def test_blocks_cover_the_rows_with_the_floor():
    for n, r in ((1200, 41), (1200, 16), (100, 100), (10, 16), (33, 16), (9672, 155)):
        bl = P._blocks(n, r)
        assert bl[0][0] == 0 and bl[-1][1] == n and all(b[1] == c[0] for b, c in zip(bl, bl[1:]))       # a partition of range(n), in order
        assert all((b1 - b0) >= min(n, P.MIN_ROWS) for b0, b1 in bl)                                       # no launch below the floor (a short tail is merged)
        assert all((b1 - b0) <= r + P.MIN_ROWS for b0, b1 in bl)


def test_rows_follow_the_budget_and_the_floor():
    P.STATE["block_mib"] = 256
    assert P._rows(1200, 1200 * 64 * 4 * 3) == (256 << 20) // (1200 * 64 * 12)                              # 291 rows of a [rows, 1200, 64] fp32 group
    assert P._rows(9672, 9672 * 52) == max(P.MIN_ROWS, (256 << 20) // (9672 * 52))
    P.STATE["block_mib"] = 1
    assert P._rows(1200, 1200 * 64 * 12) == P.MIN_ROWS                                                      # a tiny budget: the floor
    P.STATE["block_mib"] = None


def test_switch_words():
    assert P.requested({}) is False and P.requested({"OPENFOLD3_OB0_OPT_POSTFWD_MEM": "1"}) is True
    with pytest.raises(ValueError):
        P.requested({"OPENFOLD3_OB0_OPT_POSTFWD_MEM": "on"})
    assert P.block_mib({}) == 256 and P.block_mib({"OPENFOLD3_OB0_OPT_POSTFWD_MEM_MIB": "0"}) == 0 and P.block_mib({"OPENFOLD3_OB0_OPT_POSTFWD_MEM_MIB": "64"}) == 64
    for bad in ("-1", "x", "1.5"):
        with pytest.raises(ValueError):
            P.block_mib({"OPENFOLD3_OB0_OPT_POSTFWD_MEM_MIB": bad})


def test_an_engine_without_the_scoring_surface_is_refused_by_name(monkeypatch):
    prev = {k: getattr(P, k) for k in P.CONFIGURABLE}; saved = dict(P.STATE)
    for name in ("of3opt_t_acr", "of3opt_t_sr", "of3opt_t_conf", "of3opt_t_atomize"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    P.configure(M_ACR="of3opt_t_acr", M_SR="of3opt_t_sr", M_CONF="of3opt_t_conf", M_ATOMIZE="of3opt_t_atomize")
    P.STATE.update(installed=False, state="off", reason="")
    try:
        st = P.install({"OPENFOLD3_OB0_OPT_POSTFWD_MEM": "1"})
        assert st["state"] == "refused" and st["reason"].startswith("engine_surface:") and not P.ORIG
        assert "state=refused reason=engine_surface:" in P.census_line()
    finally:
        P.configure(**prev); P.STATE.clear(); P.STATE.update(saved)


def test_a_source_digest_off_the_pin_is_refused_by_name(monkeypatch):
    torch = pytest.importorskip("torch")
    eng = _engine_or_skip()
    prev = dict(P.DIGESTS); saved = dict(P.STATE)
    P.STATE.update(installed=False, state="off", reason="")
    P.configure(DIGESTS={"compute_ptm": ("0000000000000000",)})
    try:
        st = P.install({"OPENFOLD3_OB0_OPT_POSTFWD_MEM": "1"})
        assert st["state"] == "refused" and st["reason"].startswith("engine_source:compute_ptm:") and not P.ORIG
        assert eng["conf"].compute_ptm is eng["orig_ptm"]                                                     # nothing patched
    finally:
        P.configure(DIGESTS=prev); P.STATE.clear(); P.STATE.update(saved)


def _engine_or_skip():
    """The engine's scoring modules (the installed package, else the kit's stock/src tree) or skip."""
    pytest.importorskip("torch")
    try:
        import openfold3  # noqa: F401
    except ImportError:
        from openfold3_ob0_opt.tests import _stubs
        src = os.path.join(_stubs.tree_home(), "stock", "src")
        if src not in sys.path:
            sys.path.insert(0, src)
    try:
        import importlib
        acr = importlib.import_module(P.M_ACR); sr = importlib.import_module(P.M_SR); conf = importlib.import_module(P.M_CONF); A = importlib.import_module(P.M_ATOMIZE)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the engine's scoring modules are not importable here ({type(e).__name__}: {str(e)[:60]})")
    if getattr(acr._get_confidence_scores, "_of3opt_postfwd_mem", False):
        P.uninstall()
    return {"acr": acr, "sr": sr, "conf": conf, "A": A, "orig_ptm": conf.compute_ptm}


def _synthetic(torch, S=2, n_tok=80, atoms_per=5, seed=0):
    """A protein-like item: n_tok tokens x atoms_per atoms, two chains, one atomized token; logits random fp32 (CPU)."""
    g = torch.Generator().manual_seed(seed)
    n_atom = n_tok * atoms_per
    batch = {
        "token_mask": torch.ones(n_tok), "asym_id": torch.tensor([1] * (n_tok // 2) + [2] * (n_tok - n_tok // 2), dtype=torch.int32),
        "num_atoms_per_token": torch.full((n_tok,), atoms_per, dtype=torch.int32), "atom_mask": torch.ones(n_atom),
        "start_atom_index": torch.arange(0, n_atom, atoms_per, dtype=torch.int32), "is_atomized": torch.zeros(n_tok, dtype=torch.int32),
        "is_protein": torch.ones(n_tok, dtype=torch.int32), "is_dna": torch.zeros(n_tok, dtype=torch.int32), "is_rna": torch.zeros(n_tok, dtype=torch.int32),
        "restype": torch.nn.functional.one_hot(torch.randint(0, 20, (n_tok,), generator=g), 32).to(torch.int32),
        "atom_to_token_index": torch.arange(n_tok).repeat_interleave(atoms_per),
    }
    batch["is_atomized"][3] = 1
    batch["atom_mask"][7] = 0.0
    outputs = {"pae_logits": torch.randn(S, n_tok, n_tok, 64, generator=g), "pde_logits": torch.randn(S, n_tok, n_tok, 64, generator=g),
               "plddt_logits": torch.randn(S, n_atom, 50, generator=g), "distogram_logits": torch.randn(1, n_tok, n_tok, 64, generator=g),
               "atom_positions_predicted": torch.randn(S, n_atom, 3, generator=g) * 8.0}
    return batch, outputs


def test_lean_statement_groups_equal_the_engines_bit_for_bit_on_cpu():
    torch = pytest.importorskip("torch")
    eng = _engine_or_skip()
    conf, A = eng["conf"], eng["A"]
    torch.set_num_threads(1)
    batch, outputs = _synthetic(torch)
    saved = dict(P.STATE); P.STATE["block_mib"] = 1                                                          # a 1 MiB budget: many row blocks at these sizes
    try:
        pde_cfg = dict(bin_min=0, bin_max=32, no_bins=64)
        ref = conf.probs_to_expected_error(torch.softmax(outputs["pde_logits"], dim=-1), **pde_cfg)
        assert torch.equal(P.expected_error_lean(outputs["pde_logits"], conf.probs_to_expected_error, pde_cfg), ref)
        g_ref, c_ref = conf.compute_global_predicted_distance_error(pde=ref, logits=outputs["distogram_logits"], bin_min=2.3125, bin_max=21.6875, no_bins=64)
        g_lean, c_lean = P.gpde_lean(pde=ref, logits=outputs["distogram_logits"], bin_min=2.3125, bin_max=21.6875, no_bins=64)
        assert torch.equal(g_lean, g_ref) and torch.equal(c_lean, c_ref)
        ptm = P.make_compute_ptm(conf.get_bin_centers, conf.compute_ptm)
        has_frame = torch.ones(outputs["pae_logits"].shape[0], batch["token_mask"].shape[0], dtype=torch.bool)
        full = torch.ones_like(batch["token_mask"]).bool(); part = full.clone(); part[5:9] = False       # all-true mask (views) and a subset (the fused gather)
        for mask in (full, part):
            for interface in (False, True):
                kw = dict(bin_min=0, bin_max=32, no_bins=64, mask_i=mask, asym_id=batch["asym_id"], interface=interface)
                assert torch.equal(ptm(outputs["pae_logits"], has_frame, **kw), conf.compute_ptm(outputs["pae_logits"], has_frame, **kw)), (interface, int(mask.sum()))
        frames = P.make_token_frame_atoms(A, A.get_token_frame_atoms)
        phi_l, valid_l = frames(batch=batch, x=outputs["atom_positions_predicted"], atom_mask=batch["atom_mask"])
        phi_e, valid_e = A.get_token_frame_atoms(batch=batch, x=outputs["atom_positions_predicted"], atom_mask=batch["atom_mask"])
        assert torch.equal(valid_l, valid_e) and all(torch.equal(a, b) for a, b in zip(phi_l, phi_e))
        assert P.STATE["lean"].get("ptm", 0) >= 4 and P.STATE["lean"].get("frames", 0) >= 1
    finally:
        P.STATE.clear(); P.STATE.update(saved); P.STATE.setdefault("lean", {}); P.STATE.setdefault("fallback", {})


def test_below_the_size_floors_the_engines_statements_run_counted():
    torch = pytest.importorskip("torch")
    eng = _engine_or_skip()
    conf, A = eng["conf"], eng["A"]
    batch, outputs = _synthetic(torch, S=1, n_tok=12, atoms_per=4)                                          # 12 tokens, 48 atoms: below MIN_TOKENS / MIN_ATOMS
    saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in P.STATE.items()}; P.STATE["block_mib"] = 256
    try:
        ptm = P.make_compute_ptm(conf.get_bin_centers, conf.compute_ptm)
        has_frame = torch.ones(1, 12, dtype=torch.bool); mask = torch.ones(12).bool()
        kw = dict(bin_min=0, bin_max=32, no_bins=64, mask_i=mask, asym_id=batch["asym_id"], interface=True)
        assert torch.equal(ptm(outputs["pae_logits"], has_frame, **kw), conf.compute_ptm(outputs["pae_logits"], has_frame, **kw))
        frames = P.make_token_frame_atoms(A, A.get_token_frame_atoms)
        _, v1 = frames(batch=batch, x=outputs["atom_positions_predicted"], atom_mask=batch["atom_mask"])
        _, v2 = A.get_token_frame_atoms(batch=batch, x=outputs["atom_positions_predicted"], atom_mask=batch["atom_mask"])
        assert torch.equal(v1, v2) and P.STATE["fallback"].get("ptm_lt_min_tokens") and P.STATE["fallback"].get("frames_lt_min_atoms")
    finally:
        P.STATE.clear(); P.STATE.update(saved)
