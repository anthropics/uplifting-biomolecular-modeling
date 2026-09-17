"""zprep_hoist: the denoiser's per-step (N,N,c)->(c,N,N) pair copy answered from a memo on the (hoisted) source tensor — bitwise."""
import pytest

torch = pytest.importorskip("torch")
from opendde_opt import modes, registry, ran, zprephoist


def _perm(t, inds):                                    # upstream's permute_final_dims (opendde/model/utils.py) verbatim semantics
    zero_index = -1 * len(inds)
    first_inds = list(range(len(t.shape[:zero_index])))
    return t.permute(first_inds + [zero_index + i for i in inds])


def test_composition_rows():
    assert "zprep_hoist" in registry.LEVERS and registry.LEVERS["zprep_hoist"].tier == "exact"
    for ln in ("S1", "LSTAR2A"):
        assert "zprep_hoist" in modes.LINES[ln].levers, ln
    assert "zprep_hoist" not in modes.LINES["BIG_F"].levers and "zprep_hoist" in modes.BIG_DROP             # off every big line by name: a measured peak cost (0.2.47)
    assert "zprep_hoist" not in modes.LINES["BIG_TP"].levers and "zprep_hoist" in modes.BIG_TP_DROP
    assert modes.LEVER_SWITCHES["zprep_hoist"] == () and "zprep_hoist" in ran.COUNTERS
    out = modes.line_without(modes.LINES["S1"], ("zprep_hoist",))
    assert "zprep_hoist" not in out.levers and len(out.levers) == len(modes.LINES["S1"].levers) - 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_memo_is_bitwise_and_tracks_the_source(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    served = zprephoist.make_perm_wrapper(_perm)
    g = torch.Generator().manual_seed(0)
    src = torch.randn(2, 24, 24, 8, generator=g).to(device)             # the hoist's normalize buffer: the SAME object every step
    ref = _perm(src, [2, 0, 1]).contiguous()
    a = served(src, [2, 0, 1]).contiguous()
    if device == "cpu":                                                   # CPU tensors are stock (passthrough): the memo serves CUDA only
        assert torch.equal(a, ref) and zprephoist.STATS["passthrough"] == 1 and zprephoist.STATS["calls"] == 0
        return
    b = served(src, [2, 0, 1]).contiguous()
    assert torch.equal(a, ref) and b is a and a.is_contiguous() and tuple(a.shape) == (2, 8, 24, 24)
    assert zprephoist.STATS["calls"] == 2 and zprephoist.STATS["hits"] == 1 and zprephoist.STATS["records"] == 1
    src.copy_(torch.randn(2, 24, 24, 8, generator=g).to(device))          # the hoist re-records for the next sampler call: version bumps -> re-record
    c = served(src, [2, 0, 1])
    assert c is not a and torch.equal(c, _perm(src, [2, 0, 1]).contiguous()) and zprephoist.STATS["records"] == 2
    d = served(src.clone(), [2, 0, 1])                                    # a fresh tensor with the same values: a new record replaces the entry (no aliasing by id; ONE entry)
    assert torch.equal(d, c) and zprephoist.STATS["records"] == 3 and zprephoist.STATS["entries"] == 1
    e = served(src, [0, 2, 1])                                            # another permutation: stock, a view, uncounted as a call
    assert zprephoist.STATS["passthrough"] == 1 and not e.is_contiguous()
    assert zprephoist.STATS["entries"] <= zprephoist.MAX_ENTRIES


def test_autograd_sources_are_stock():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    served = zprephoist.make_perm_wrapper(_perm)
    src = torch.randn(4, 4, 3, device="cuda", requires_grad=True)
    out = served(src, [2, 0, 1])
    assert out.requires_grad and zprephoist.STATS["passthrough"] == 1 and zprephoist.STATS["calls"] == 0


def test_real_upstream_f_forward_site(monkeypatch):
    """The bound name is the one f_forward resolves: opendde.model.modules.diffusion.permute_final_dims (imported there from model.utils)."""
    diffusion = pytest.importorskip("opendde.model.modules.diffusion")
    import inspect
    src = inspect.getsource(diffusion.DiffusionModule.f_forward)
    assert "permute_final_dims(z, [2, 0, 1]).contiguous()" in src           # the statement the lever serves (diffusion.py:1512)
    assert src.count("permute_final_dims(") == 1                           # ... and the only use of the name in f_forward
    zprephoist.install()
    assert zprephoist.STATS["installed"] and getattr(diffusion.permute_final_dims, "__wrapped__", None) is not None
