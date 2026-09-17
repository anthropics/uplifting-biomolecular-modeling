"""The triangle-multiplication contraction on STRUCTURED operands — the standing operand class for every contraction seam: distinct iid
randn a != b (a symmetric product would hide an operand-role error), module-produced operands (LayerNorm -> linear -> sigmoid gate of a
smooth pair tensor: small, correlated values), the same plus noise, zero-padded rows / columns (identical tiles) and rank-2 operands.
``trimul.trimul_outgoing`` / ``trimul_incoming`` (projected-operand primitives) at P in {2, 3, 4} on rank-threads must equal
``trimul.trimul_dense`` to the fp32 GEMM class (max|diff| <= TOL x max|dense|: every tile holds the full k range in ONE matmul, so the
only difference from the dense statement is the GEMM's M — bit-exact where the CPU / GPU GEMM is M-invariant, which is REPORTED, not
asserted: MKL selects kernels by M on some CPUs), and ``trimul_dense`` equals the einsum statement to 1e-5; one 2-process gloo run covers
the production CPU transport. Operands are built ONCE in the launching process and handed to every rank (rank-threads share
the process RNG: drawing inside a rank would hand the ranks different operands)."""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from opt_core.mem.rowpair import dist as D, trimul as TM  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402

N, C = 40, 16
TOL = 1e-5                      # x max|dense|: the fp32 GEMM class (M differs; k complete per tile)
KINDS = ("randn", "projected", "projected_noise", "padded", "lowrank")


def operands(kind: str):
    g = torch.Generator().manual_seed(1)
    if kind == "randn":
        a, b = torch.randn(N, N, C, generator=g), torch.randn(N, N, C, generator=g)
    elif kind in ("projected", "projected_noise"):
        u = torch.linspace(0, 3, N)[:, None] * torch.randn(1, 8, generator=g)
        v = torch.cos(torch.linspace(0, 2, N))[:, None] * torch.randn(1, 8, generator=g)
        z = torch.nn.functional.layer_norm(u[:, None, :] + v[None, :, :], (8,))                 # a smooth pair tensor
        W, G = torch.randn(8, 2 * C, generator=g) * 0.5, torch.randn(8, 2 * C, generator=g) * 0.5
        p = (z @ W) * torch.sigmoid(z @ G)                                                       # the engine's gated dual projection
        a, b = (t.contiguous() for t in p.chunk(2, dim=-1))
        if kind == "projected_noise":
            a, b = a + 0.1 * torch.randn(N, N, C, generator=g), b + 0.1 * torch.randn(N, N, C, generator=g)
    elif kind == "padded":
        a, b = torch.randn(N, N, C, generator=g), torch.randn(N, N, C, generator=g)
        a[30:], b[30:] = 0, 0
        a[:, 30:], b[:, 30:] = 0, 0
    elif kind == "lowrank":
        x, y, w = torch.randn(N, 2, generator=g), torch.randn(N, 2, generator=g), torch.randn(2, 2, C, generator=g)
        a = torch.einsum("ip,jq,pqc->ijc", x, y, w).contiguous()
        b = torch.einsum("ip,jq,pqc->ijc", y, x, w).contiguous()
    else:
        raise ValueError(kind)
    assert not torch.equal(a, b)
    return a.float(), b.float()


def test_dense_statement_equals_einsum():
    for kind in KINDS:
        a, b = operands(kind)
        for outgoing in (True, False):
            ref = torch.einsum("ikc,jkc->ijc", a, b) if outgoing else torch.einsum("kic,kjc->ijc", a, b)
            got = TM.trimul_dense(a, b, outgoing)
            assert float((got - ref).abs().max()) <= 1e-5 * float(ref.abs().max()), (kind, outgoing)


def _rank(rank, P, a, b, outgoing, B, RA, RB):
    lay = D.Layout(N, P, rank, B)
    fn = TM.trimul_outgoing if outgoing else TM.trimul_incoming
    out, _st = fn(a[lay.r0:lay.r1].contiguous(), b[lay.r0:lay.r1].contiguous(), lay, RA=RA, RB=RB)
    return lay.r0, out


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("P,B,RA,RB", [(2, 20, 20, 20), (2, 20, None, None), (3, 8, 4, 8), (4, 10, None, None)])
@pytest.mark.parametrize("outgoing", [True, False], ids=["outgoing", "incoming"])
def test_legacy_wrappers_bitwise_on_structured_operands(kind, P, B, RA, RB, outgoing):
    a, b = operands(kind)
    dense = TM.trimul_dense(a, b, outgoing)
    res = run_ranks(P, _rank, a, b, outgoing, B, RA, RB)
    full = torch.cat([o for _, o in sorted(res, key=lambda t: t[0])], dim=0)
    d = float((full - dense).abs().max())
    print(f"{kind} P={P} B={B} RA={RA} RB={RB} {'out' if outgoing else 'in'}: max|diff|={d:.3e} bitwise={torch.equal(full, dense)}")
    assert d <= TOL * max(1.0, float(dense.abs().max())), (d, kind, P, B, RA, RB, outgoing)


def _mp_entry():
    P, r = D.world()
    res = {}
    for kind in KINDS:
        a, b = operands(kind)                                                  # seeded generator: identical bytes in every process
        D.allreduce_checksum(a, "a")                                           # the replication proof an adapter owes its operands
        D.allreduce_checksum(b, "b")
        lay = D.Layout(N, P, r, 20)
        for outgoing in (True, False):
            fn = TM.trimul_outgoing if outgoing else TM.trimul_incoming
            out, _ = fn(a[lay.r0:lay.r1].contiguous(), b[lay.r0:lay.r1].contiguous(), lay)
            dense = TM.trimul_dense(a, b, outgoing)[lay.r0:lay.r1]
            res[f"{kind}.{'out' if outgoing else 'in'}"] = float((out - dense).abs().max()) <= TOL * max(1.0, float(dense.abs().max()))
    D.barrier()
    return {"rank": r, "within_tol": res}


def test_mp_legacy_wrappers_gloo_processes():
    from opt_core.mem.rowpair import launch
    got = launch.run_sharded(2, _mp_entry, cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert got["rank"] == 0 and all(got["within_tol"].values()), got
