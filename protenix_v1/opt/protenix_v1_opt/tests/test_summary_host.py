"""Lever summary_hostidx (lib/ptx1_summary_host.py) against the pinned stock statement (stock/protenix-1.1.0-py3-none-any.whl on sys.path, CPU
torch): on synthetic multi-chain inputs — an asym-id gap, a frameless chain, a ligand chain, a forced clash, two samples — every summary tensor and
every full_data tensor the lever returns is torch.equal to the stock function's; the install / step-aside / signature-drift rules by name.
Skipped without torch or the wheel (the kit's other CPU tests need neither)."""
import glob
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ml_collections")
from ml_collections.config_dict import ConfigDict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LIB = os.path.join(KIT, "opt", "forward", "v05_addon", "lib")
WHEELS = glob.glob(os.path.join(KIT, "stock", "protenix-*.whl"))


@pytest.fixture(scope="module")
def SC():
    """The stock module (an installed protenix wins; else the wheel's pure-Python modules on sys.path for this module only) and the lever.
    Teardown restores sys.path and drops every module this fixture brought in from the wheel, so the rest of the suite sees the interpreter
    it started with (tests that skip without an installed protenix keep skipping)."""
    if not WHEELS:
        pytest.skip("stock wheel not found under stock/")
    path0, mods0 = list(sys.path), set(sys.modules)
    try:
        try:
            import protenix.model.sample_confidence as sc
        except Exception:
            sys.path.insert(0, WHEELS[0])
            try:
                import protenix.model.sample_confidence as sc
            except Exception as e:                              # a dependency of the stock module missing in this interpreter
                pytest.skip(f"stock sample_confidence not importable here: {e!r}")
        if LIB not in sys.path:
            sys.path.insert(0, LIB)
        import ptx1_summary_host as SH
        yield sc, SH
        SH.uninstall()
    finally:
        sys.path[:] = path0
        for name in [m for m in sys.modules if m not in mods0 and (m == "protenix" or m.startswith(("protenix.", "runner", "ptx1_summary_host")))]:
            f = getattr(sys.modules[name], "__file__", "") or ""
            if WHEELS[0] in f or name == "ptx1_summary_host":
                del sys.modules[name]


def _configs():
    return ConfigDict({"loss": {"pae": {"min_bin": 0, "max_bin": 32, "no_bins": 64}, "pde": {"min_bin": 0, "max_bin": 32, "no_bins": 64},
                                "plddt": {"min_bin": 0, "max_bin": 1.0, "no_bins": 50}},
                       "metrics": {"clash": {"af3_clash_threshold": 1.1, "vdw_clash_threshold": 0.75}}})


def _inputs(seed=0, n_sample=2):
    g = torch.Generator().manual_seed(seed)
    # chains: asym 0 (12 tok, protein), 1 (10 tok, protein), 3 (6 tok, frameless polymer), 4 (5 tok, ligand: atoms not polymer) — id 2 absent
    asym = torch.tensor([0] * 12 + [1] * 10 + [3] * 6 + [4] * 5)
    n_tok = asym.numel()
    has_frame = torch.ones(n_tok, dtype=torch.long); has_frame[22:28] = 0; has_frame[3] = 0
    atoms_per_tok = torch.tensor([3] * 12 + [2] * 10 + [4] * 6 + [1] * 5)
    atom_to_token = torch.repeat_interleave(torch.arange(n_tok), atoms_per_tok)
    n_atom = atom_to_token.numel()
    atom_is_polymer = (asym[atom_to_token] != 4).long()
    coord = torch.randn(n_sample, n_atom, 3, generator=g) * 8.0
    coord[1, : 3 * 12] = coord[1, 36:37] + 0.3 * torch.randn(36, 3, generator=g)      # sample 1: chain 0's atoms piled onto a chain-1 atom -> af3 clash flag
    return dict(pae_logits=torch.randn(n_sample, n_tok, n_tok, 64, generator=g), plddt_logits=torch.randn(n_sample, n_atom, 50, generator=g),
                pde_logits=torch.randn(n_sample, n_tok, n_tok, 64, generator=g), contact_probs=torch.rand(n_tok, n_tok, generator=g),
                token_asym_id=asym, token_has_frame=has_frame, atom_coordinate=coord, atom_to_token_idx=atom_to_token,
                atom_is_polymer=atom_is_polymer, N_recycle=4)


def _equal(a, b, path="", diffs=None):
    diffs = [] if diffs is None else diffs
    if isinstance(a, torch.Tensor):
        if not (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)):
            diffs.append(path)
    elif isinstance(a, dict):
        if set(a) != set(b):
            diffs.append(path + ":keys"); return diffs
        for k in a:
            _equal(a[k], b[k], f"{path}.{k}", diffs)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append(path + ":len"); return diffs
        for i, (x, y) in enumerate(zip(a, b)):
            _equal(x, y, f"{path}[{i}]", diffs)
    elif a != b:
        diffs.append(path)
    return diffs


@pytest.mark.parametrize("seed", [0, 1])
def test_summary_and_full_data_are_bitwise_the_stock_values(SC, seed):
    sc, SH = SC
    SH.uninstall()
    stock = sc.compute_full_data_and_summary
    assert getattr(stock, "__module__", "") == SH.TARGET and not getattr(stock, "_ptx_summary_host", False)
    cfg, kw = _configs(), _inputs(seed)
    ref_s, ref_f = stock(cfg, return_full_data=True, **kw)
    assert SH.install().startswith("on(") and sc.compute_full_data_and_summary is SH.compute_full_data_and_summary
    got_s, got_f = sc.compute_full_data_and_summary(cfg, return_full_data=True, **kw)
    assert len(ref_s) == len(got_s) == 2
    assert _equal(ref_s, got_s, "summary") == [] and _equal(ref_f, got_f, "full") == []
    assert bool(ref_s[1]["has_clash"]) and not bool(ref_s[0]["has_clash"])            # the forced clash is seen (by both)
    assert ref_s[0]["chain_ptm"].shape[-1] == 4 and torch.equal(got_s[0]["chain_pair_iptm"], ref_s[0]["chain_pair_iptm"])
    r = SH.report()
    assert r["installed"] and r["samples"] >= 2 and r["calls"] >= 1 and r["delegated"] == 0 and r["errors"] == 0
    SH.uninstall()
    assert sc.compute_full_data_and_summary is stock


def test_install_rules_by_name(SC):
    sc, SH = SC
    SH.uninstall(); stock = sc.compute_full_data_and_summary
    tp_like = types.SimpleNamespace()                                                   # the row-sharded line's replacement: another module's callable -> step aside
    def compute_full_data_and_summary(*a, **k): return stock(*a, **k)
    compute_full_data_and_summary.__module__ = "protenix_v1_opt.tp"; compute_full_data_and_summary.__wrapped__ = stock
    sc.compute_full_data_and_summary = compute_full_data_and_summary
    try:
        mark = SH.install()
        assert mark.startswith("stepped_aside(protenix_v1_opt.tp.") and not SH.report()["installed"] and "row-sharded" in SH.report()["stepped_aside"]
        assert sc.compute_full_data_and_summary is compute_full_data_and_summary           # nothing replaced
    finally:
        sc.compute_full_data_and_summary = stock; SH._ST["stepped_aside"] = None
    def drifted(configs, pae_logits, extra=None): return None                           # a stock surface that is not the restated one -> refused by name
    drifted.__module__ = SH.TARGET
    sc.compute_full_data_and_summary = drifted
    try:
        with pytest.raises(RuntimeError, match="summary_hostidx: protenix.model.sample_confidence.compute_full_data_and_summary signature"):
            SH.install()
    finally:
        sc.compute_full_data_and_summary = stock
    assert SH.install().startswith("on(") and SH.install() == "on(already)"
    SH.uninstall(); assert sc.compute_full_data_and_summary is stock
