"""aligncap (opt/forward/waste/aligncap.py): U,S,Vh bitwise torch.linalg.svd(cov, driver='gesvd') incl. output strides, for Bm in {1,2,5}, on random,
covariance-like, rank-deficient, repeated-singular-value and zero matrices; weighted_rigid_align_nosync == stock weighted_rigid_align (torch.equal) on
random weighted point clouds incl. masked atoms; flags() reads devInfo once. GPU test (CUDA + boltz required); pytest or plain python3."""
import importlib.util, os, sys
try:
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required", allow_module_level=True)
    if importlib.util.find_spec("boltz") is None:
        pytest.skip("boltz not importable", allow_module_level=True)
except ModuleNotFoundError:
    pytest = None
    import torch
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import aligncap as AC  # noqa: E402

dev = "cuda"


def _mats():
    torch.manual_seed(0)
    ms = [torch.randn(3, 3, device=dev) * s for s in (1e-3, 1.0, 1e3) for _ in range(50)]
    for npts in (4, 40, 4000):
        for _ in range(30):
            X = torch.randn(npts, 3, device=dev); Y = X @ torch.linalg.qr(torch.randn(3, 3, device=dev))[0] + 0.05 * torch.randn(npts, 3, device=dev)
            ms.append((X.T @ Y).contiguous())
    q = torch.linalg.qr(torch.randn(3, 3, device=dev))[0]
    ms += [torch.zeros(3, 3, device=dev), torch.eye(3, device=dev), torch.ones(3, 3, device=dev), torch.diag(torch.tensor([5.0, 5.0, 1e-20], device=dev)),
           (q @ torch.diag(torch.tensor([3.0, 3.0, 3.0], device=dev)) @ q.T).contiguous()]
    return ms


def test_svd_bitwise_and_strides():
    ms = _mats()
    for Bm in (1, 2, 5):
        AC.init(Bm, dev)
        U, S, Vh = AC.new_U(Bm, dev), AC.new_S(Bm, dev), AC.new_Vh(Bm, dev)
        for k in range(0, len(ms) - Bm, Bm):
            cov = torch.stack(ms[k:k + Bm]).contiguous()
            Ut, St, Vt = torch.linalg.svd(cov, driver="gesvd")
            AC.aligncap(cov, U, S, Vh)
            assert U.stride() == Ut.stride() == (9, 1, 3) and Vh.stride() == Vt.stride() and S.stride() == St.stride()
            assert torch.equal(U, Ut) and torch.equal(S, St) and torch.equal(Vh, Vt), (Bm, k)
    assert AC.flags()["gesvd_info_nonzero"] is False and AC.flags()["calls"] > 0
    AC.reset_flags(); assert AC.flags()["calls"] == 0


def test_bad_buffers_refused():
    AC.init(1, dev)
    cov = torch.randn(1, 3, 3, device=dev)
    for U in (torch.empty(1, 3, 3, device=dev),):                       # C-contig U is the wrong layout
        try:
            AC.aligncap(cov, U, AC.new_S(1), AC.new_Vh(1)); raise AssertionError("accepted a C-contiguous U_out")
        except ValueError:
            pass
    try:
        AC.aligncap(cov.mT, AC.new_U(1), AC.new_S(1), AC.new_Vh(1)); raise AssertionError("accepted a non-contiguous cov")
    except ValueError:
        pass


def test_weighted_rigid_align_nosync_equals_stock():
    from boltz.model.loss.diffusionv2 import weighted_rigid_align as stock
    torch.manual_seed(1)
    g = torch.zeros(2, dtype=torch.bool, device=dev)
    for n_atoms in (37, 480, 5003):
        for _ in range(10):
            true = torch.randn(1, n_atoms, 3, device=dev) * 7; pred = true @ torch.linalg.qr(torch.randn(3, 3, device=dev))[0] + torch.randn(1, n_atoms, 3, device=dev)
            mask = (torch.rand(1, n_atoms, device=dev) > 0.1).float(); w = torch.rand(1, n_atoms, device=dev) + 0.5
            ref = stock(true, pred, w, mask)
            out = AC.weighted_rigid_align_nosync(true, pred, w, mask, guards=g)
            assert torch.equal(ref, out)
    assert g.tolist() == [False, False]


if __name__ == "__main__":
    for name in sorted(k for k in dir(sys.modules[__name__]) if k.startswith("test_")):
        getattr(sys.modules[__name__], name)(); print("ok ", name, flush=True)
    print("ALL PASSED")
