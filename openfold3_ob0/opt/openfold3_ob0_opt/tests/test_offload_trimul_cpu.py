"""The offload port's triangle multiplication with the residual snapshot on the host (`of3_offload.trimul_inference_forward`, the O1
`trimul_hostsnap` lever installed as `TriangleMultiplicativeUpdate._inference_forward` under the unit switch OF3O_TRIMUL) against upstream
0.5.0's own `_inference_forward` on a tiny pair tensor (CPU, float64; skipped without torch or the openfold3 wheel): outgoing and incoming,
with and without the in-place residual add, N not a multiple of the chunk. Identical numbers (rtol 0, atol 1e-12: the same statements at the
same chunk boundaries; the outgoing update reads the original rows back from a host buffer — pageable here, OF3O_PIN_BUDGET_GB=0, since a CPU
box pins nothing). Sensitivity: the un-chunked path with the wrong direction is a different function of z."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openfold3", reason="the openfold3 wheel is not installed: the host-snapshot TriMul is checked against upstream's module")

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))      # the tree home (opt/openfold3_ob0_opt/tests -> .)
OF3O = os.path.join(HOME, "opt", "forward", "offload", "of3o")


@pytest.fixture(scope="module")
def O():
    sys.path.insert(0, OF3O)
    try:
        import of3_offload
        yield of3_offload
    finally:
        sys.path.remove(OF3O)


def _randomise(module, gen):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype))


def _tiny(outgoing, N=11, c=8, seed=0):
    from openfold3.core.model.layers.triangular_multiplicative_update import (TriangleMultiplicationIncoming,
                                                                              TriangleMultiplicationOutgoing)
    gen = torch.Generator().manual_seed(seed)
    cls = TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    tm = cls(c_z=c, c_hidden=c).double().eval()
    _randomise(tm, gen)                                        # upstream's 'final' / 'gating' inits zero or saturate projections: random weights make the check meaningful
    z = torch.randn(1, N, N, c, generator=gen, dtype=torch.float64)
    mask = (torch.rand(1, N, N, generator=gen) > 0.15).double()
    return tm, z, mask


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("with_add", [True, False])
@pytest.mark.parametrize("chunk", [4, 256])
def test_trimul_hostsnap_equals_upstream_inference_forward(O, outgoing, with_add, chunk, monkeypatch):
    monkeypatch.setenv("OF3O_PIN_BUDGET_GB", "0")               # no pinned memory on a CPU box: the snapshot buffer is pageable (counted, same numbers)
    monkeypatch.setenv("OF3O_PIN_POLICY", "census")
    O.free_pinned("trimul_snap")
    tm, z, mask = _tiny(outgoing)
    with torch.no_grad():
        ref = tm._inference_forward(z.clone(), mask, inplace_chunk_size=chunk, with_add=with_add)
        out = O.trimul_inference_forward(tm, z.clone(), mask, inplace_chunk_size=chunk, with_add=with_add)
    O.free_pinned("trimul_snap")
    assert out.shape == ref.shape == z.shape
    assert torch.allclose(out, ref, rtol=0, atol=1e-12), float((out - ref).abs().max())
    # sensitivity: the update alone (no residual) differs from the update added in place, and outgoing != incoming on the same z
    other, _, _ = _tiny(not outgoing)
    with torch.no_grad():
        alt = other._inference_forward(z.clone(), mask, inplace_chunk_size=chunk, with_add=with_add)
    assert not torch.allclose(alt, ref, rtol=0, atol=1e-6)


def test_the_unit_is_installed_under_its_switch(O):
    """apply_core installs the port's function as TriangleMultiplicativeUpdate._inference_forward when OF3O_TRIMUL is on (default) — read
    statically here (the hook's install lines), the live binding being the GPU box's `LEVER name=trimul_hostsnap state=on` census."""
    src = open(os.path.join(OF3O, "of3_offload.py"), encoding="utf-8").read()
    assert 'if env_int("OF3O_TRIMUL", 1):\n            TM.TriangleMultiplicativeUpdate._inference_forward = trimul_inference_forward' in src
    assert '_ORIG["TMU._inference_forward"] = TM.TriangleMultiplicativeUpdate._inference_forward' in src
    assert 'free_pinned("trimul_snap")' in src
