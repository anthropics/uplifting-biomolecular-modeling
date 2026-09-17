"""Lever IO1 (pdbio.py). Two coverage angles in one file: the CPU-fixture tests below need no torch — pdbio.text_single / text_multi (the
code the driver-facing writers run) reproduce, byte for byte, what upstream ``rfdiffusion.util.writepdb`` / ``writepdb_multi`` of the
pinned checkout wrote for the same inputs (the fixture ``fixtures/pdbio_fixture.npz`` + ``.json``: inputs, upstream's output bytes,
upstream's ``aa2long`` / ``num2aa`` tables; regenerated on the tested image by ``fixtures/make_pdbio_fixture.py``). The GPU-live tests
further down need torch and the RFdiffusion checkout importable (``RFD_ROOT`` on ``sys.path``; the GPU tested image) and are skipped on
this torch-less CPU test image: the package's numpy PDB writers produce upstream's bytes for every branch of ``writepdb`` /
``writepdb_multi``, and the in-process proof keeps upstream's file and counts the event when they would not."""
import json
import os
import sys

import numpy as np
import pytest

from rfdiffusion1_opt import pdbio

FX = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture():
    meta = json.load(open(os.path.join(FX, "pdbio_fixture.json"), encoding="utf-8"))
    arrays = np.load(os.path.join(FX, "pdbio_fixture.npz"))
    aa2long = [tuple(x) for x in meta["aa2long"]]
    return meta, arrays, aa2long, meta["num2aa"]


def _clamp01(x):
    return np.clip(x, 0, 1).tolist()


def test_fixture_is_upstreams_and_covers_the_branches():
    meta, arrays, aa2long, num2aa = _fixture()
    names = [c["name"] for c in meta["cases"]]
    assert len(names) == len(set(names)) == 29 and meta["N_BACKBONE_ATOMS"] == 3 and meta["N_HEAVY"] == 14
    assert {n.split("_")[0] for n in names} == {"ca", "n3", "n4", "n14", "n27", "m14", "m27"}
    assert len(aa2long) == 22 and all(len(a) == 27 for a in aa2long) and num2aa[8] == "HIS" and any(c.get("his_d") and any(c["his_d"]) for c in meta["cases"])
    assert "rfdiffusion.util.writepdb" in meta["source"]


def test_single_structure_text_is_upstreams_bytes():
    meta, arrays, aa2long, num2aa = _fixture()
    seq, bf, idx = arrays["seq"].tolist(), arrays["bf"], arrays["idx"].tolist()
    n = 0
    for c in (c for c in meta["cases"] if c["kind"] == "single"):
        a = np.squeeze(arrays[c["name"]])
        kw = c["kwargs"]
        Bf = _clamp01(bf) if kw.get("bfacts") is not None else [0.0] * a.shape[0]
        ridx = kw["idx_pdb"] if kw.get("idx_pdb") is not None else list(range(1, a.shape[0] + 1))
        natoms = a.shape[1] if a.ndim > 1 else 0
        text = pdbio.text_single(a.tolist(), a.ndim, natoms, seq, [int(v) for v in ridx], Bf, kw.get("binderlen"), kw.get("chain_idx"), c["his_d"], aa2long, num2aa)
        want = arrays["text_" + c["name"]].tobytes()
        assert text.encode() == want, c["name"]
        n += 1
    assert n == 15


def test_trajectory_text_is_upstreams_bytes():
    meta, arrays, aa2long, num2aa = _fixture()
    Bf = _clamp01(arrays["bf"])
    n = 0
    for c in (c for c in meta["cases"] if c["kind"] == "multi"):
        stack = arrays[c["name"]]
        seqx = arrays[c["seq"]]
        seq_rows = (np.tile(seqx, (stack.shape[0], 1)) if seqx.ndim != 2 else seqx).tolist()
        kw = c["kwargs"]
        stop = meta["N_BACKBONE_ATOMS"] if kw.get("backbone_only") else (meta["N_HEAVY"] if not kw.get("use_hydrogens", True) else None)
        models = [(m.tolist(), np.isnan(m).all(axis=-1).tolist()) for m in stack[:len(seq_rows)]]
        text = pdbio.text_multi(models, seq_rows, Bf, kw.get("chain_ids"), stop, aa2long, num2aa)
        want = arrays["text_" + c["name"]].tobytes()
        assert text.encode() == want and text.count("ENDMDL\n") == stack.shape[0], c["name"]
        n += 1
    assert n == 14


def test_a_wrong_byte_is_seen(monkeypatch):
    """The comparison is byte-exact: one altered format digit and every case differs."""
    meta, arrays, aa2long, num2aa = _fixture()
    monkeypatch.setattr(pdbio, "FMT", pdbio.FMT.replace("%8.3f%8.3f%8.3f", "%8.3f%8.3f%8.4f"))
    c = next(c for c in meta["cases"] if c["name"] == "n4_chains")
    a = np.squeeze(arrays[c["name"]]); kw = c["kwargs"]
    text = pdbio.text_single(a.tolist(), a.ndim, a.shape[1], arrays["seq"].tolist(), [int(v) for v in kw["idx_pdb"]], _clamp01(arrays["bf"]), None, kw["chain_idx"], c["his_d"], aa2long, num2aa)
    assert text.encode() != arrays["text_" + c["name"]].tobytes()


# ----------------------------------------------------------------------------------------------------------------- the GPU-live tests (need torch + the RFdiffusion checkout; skipped on the CPU image)
try:
    import torch
    _rfd = os.environ.get("RFD_ROOT", "/opt/rfd")
    if os.path.isdir(_rfd) and _rfd not in sys.path:
        sys.path.insert(0, _rfd)
    import rfdiffusion.util as U
    _GPU_STACK_ERROR = None
except ImportError as _e:
    torch = U = None
    _GPU_STACK_ERROR = str(_e)

needs_gpu_stack = pytest.mark.skipif(_GPU_STACK_ERROR is not None, reason=f"needs torch + the RFdiffusion checkout (GPU tested image): {_GPU_STACK_ERROR}")


def _coords(gen, *shape):
    x = torch.randn(*shape, generator=gen) * 30.0
    x.view(-1)[::17] = 0.0                                    # exact zeros, "-0.000" after tiny negatives, large magnitudes: printf corner cases
    x.view(-1)[5::29] = -0.0004
    x.view(-1)[7::31] *= 33.0
    return x


def _same(tmp_path, name, ours, theirs, *args, **kwargs):
    a, b = str(tmp_path / f"{name}_ours.pdb"), str(tmp_path / f"{name}_upstream.pdb")
    ours(a, *args, **kwargs)
    theirs(b, *args, **kwargs)
    da, db = open(a, "rb").read(), open(b, "rb").read()
    assert len(db) > 0 and da == db, name
    return da


@needs_gpu_stack
def test_writepdb_branches_are_byte_identical(tmp_path):
    g = torch.Generator().manual_seed(0)
    L = 37
    seq = torch.randint(0, 21, (L,), generator=g); seq[3] = 8; seq[9] = 8          # histidines for the protonation hack of the full-atom branch
    bf = torch.rand(L, generator=g) * 1.6 - 0.3                                    # clamped to [0, 1] by both
    idx = [int(v) for v in (torch.arange(L) + 5)]
    chains = ["A"] * 20 + ["B"] * 17
    for tag, atoms in (("ca", _coords(g, L, 3)), ("n3", _coords(g, L, 3, 3)), ("n4", _coords(g, L, 4, 3)), ("n14", _coords(g, L, 14, 3)), ("n27", _coords(g, L, 27, 3))):
        if tag == "n14":
            atoms[3, 9] = atoms[3, 5] + 0.5                                            # HIS 3: |CE1? - CG| < 1.7 -> the his_d atom names; HIS 9 stays far
        _same(tmp_path, f"{tag}_plain", pdbio.writepdb, U.writepdb, atoms, seq)
        _same(tmp_path, f"{tag}_binder", pdbio.writepdb, U.writepdb, atoms, seq, 20, idx, bf)
        _same(tmp_path, f"{tag}_chains", pdbio.writepdb, U.writepdb, atoms, seq, binderlen=None, idx_pdb=torch.tensor(idx), bfacts=bf, chain_idx=chains)
    with pytest.raises(TypeError):                                                  # L == 1 squeezes seq to 0-d: upstream cannot iterate it; neither can this
        pdbio.writepdb(str(tmp_path / "l1.pdb"), _coords(g, 1, 4, 3), torch.tensor([3]))
    with pytest.raises(TypeError):
        U.writepdb(str(tmp_path / "l1u.pdb"), _coords(g, 1, 4, 3), torch.tensor([3]))


@needs_gpu_stack
def test_writepdb_multi_is_byte_identical(tmp_path):
    g = torch.Generator().manual_seed(1)
    T, L = 4, 29
    seq1 = torch.randint(0, 21, (L,), generator=g)
    seq2 = torch.randint(0, 21, (T, L), generator=g)
    bf = torch.rand(L, generator=g)
    chains = ["A"] * 12 + ["B"] * 17
    for natoms in (14, 27):
        atoms = _coords(g, T, L, natoms, 3)
        atoms[:, :, 4:, :][torch.rand(T, L, natoms - 4, generator=g) < 0.5] = float("nan")   # absent side-chain atoms, as the sampler's stacks carry them
        atoms[1, 6, 2, 0] = float("nan")                                                      # one NaN coordinate of a present atom: written as "nan" by both
        for tag, seq in (("seq1", seq1), ("seq2", seq2)):
            kws = [dict(use_hydrogens=False), dict(backbone_only=True), dict(chain_ids=chains, use_hydrogens=False, backbone_only=False)]   # the drivers' call: use_hydrogens=False, backbone_only=False, chain_ids
            if natoms == 27:
                kws.append(dict())                                                              # hydrogens written: needs the 27-atom axis
            for n, kw in enumerate(kws):
                data = _same(tmp_path, f"multi{natoms}_{tag}_{n}", pdbio.writepdb_multi, U.writepdb_multi, atoms, bf, seq, **kw)
                assert data.count(b"ENDMDL\n") == T
        if natoms == 14:                                                                        # hydrogens asked of a 14-atom stack: upstream indexes past the atom axis; so does this
            with pytest.raises(IndexError):
                U.writepdb_multi(str(tmp_path / "u14h.pdb"), atoms, bf, seq1)
            with pytest.raises(IndexError):
                pdbio.writepdb_multi(str(tmp_path / "o14h.pdb"), atoms, bf, seq1)


@needs_gpu_stack
def test_the_in_process_proof_keeps_upstreams_bytes_on_a_mismatch(tmp_path, monkeypatch, capsys):
    g = torch.Generator().manual_seed(2)
    L = 11
    atoms, seq = _coords(g, L, 4, 3), torch.randint(0, 21, (L,), generator=g)
    ref = str(tmp_path / "ref.pdb"); U.writepdb(ref, atoms, seq)
    monkeypatch.setattr(pdbio, "_state", dict(armed=False, n_calls=0, n_verified=0, n_mismatch=0, verified_keys=[], mismatches=[], seconds=0.0))
    monkeypatch.setattr(pdbio, "_orig", {"writepdb": U.writepdb})
    good = pdbio._verified("writepdb", pdbio.writepdb)
    out = str(tmp_path / "a.pdb"); good(out, atoms, seq); good(out, atoms, seq)
    assert open(out, "rb").read() == open(ref, "rb").read() and pdbio.stats()["n_verified"] == 1 and pdbio.stats()["n_calls"] == 2 and not os.path.exists(out + ".upstream~")
    assert "pdb writer lever IO1: confirmed writepdb(" in capsys.readouterr().out
    def wrong(fn, *a, **k):
        with open(fn, "w") as f:
            f.write("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00\n")
    bad = pdbio._verified("writepdb", wrong)
    out2 = str(tmp_path / "b.pdb"); bad(out2, atoms, seq, 4)                          # a new signature (binderlen given): compared again, differs
    refb = str(tmp_path / "refb.pdb"); U.writepdb(refb, atoms, seq, 4)
    st = pdbio.stats()
    assert open(out2, "rb").read() == open(refb, "rb").read()                          # the output is upstream's bytes
    assert st["n_mismatch"] == 1 and len(st["mismatches"]) == 1 and "MISMATCH" in capsys.readouterr().out and not os.path.exists(out2 + ".upstream~")


@needs_gpu_stack
def test_arm_is_keyed_to_the_switch_value(monkeypatch):
    assert pdbio.arm(None) is None and pdbio.arm("0") is None and pdbio.arm("") is None
    monkeypatch.setattr(pdbio, "_state", dict(armed=False, n_calls=0, n_verified=0, n_mismatch=0, verified_keys=[], mismatches=[], seconds=0.0))
    orig = (U.writepdb, U.writepdb_multi)
    try:
        st = pdbio.arm("1")
        assert st["armed"] and U.writepdb is not orig[0] and U.writepdb_multi is not orig[1] and getattr(U.writepdb, "__wrapped_upstream__", False)
        assert pdbio.final_line().startswith("PDBIO_FINAL {")
    finally:
        U.writepdb, U.writepdb_multi = orig
