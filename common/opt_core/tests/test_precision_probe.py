"""CPU tests of opt_core.precision.probe.IdentityProbe: cells pass/fail/error, the sticky verdict under both policies, serve() accounting
(served / reference / unprobed / undecided), the LEVER line grammar, the gate and the manifest record. Needs torch (CPU); skips by name without it."""
from __future__ import annotations

import pytest

from opt_core.precision.probe import IdentityProbe, ProbeClosed

torch = pytest.importorskip("torch", reason="probe tests need torch (CPU)")


def _ops(B=4, n=8, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randint(-3, 4, (B, n, k), generator=g).float()          # integer-valued: bmm == loop of mm bit for bit on any backend
    X = torch.randint(-3, 4, (B, k, n), generator=g).float()
    return A, X


def test_pass_all_policy_serves_every_probed_key_and_counts():
    P = IdentityProbe("hybrid_gemm")
    for B in (2, 4):
        A, X = _ops(B=B)
        assert P.cell(("B", B), lambda: torch.bmm(A, X), lambda: torch.stack([A[i] @ X[i] for i in range(B)])) is True
    assert P.serve(("B", 2)) is False and P.stats()["undecided"] == 1          # before the verdict: reference form, counted
    assert P.decide("all") is True
    assert P.serve(("B", 2)) and P.serve(("B", 4)) and not P.serve(("B", 8))    # unprobed key -> reference form, counted
    s = P.stats()
    assert (s["probe"], s["cells"], s["bad"], s["served"], s["reference"], s["unprobed"]) == ("pass", 2, 0, 2, 2, 1)
    line = P.line("decoder-opt", strategy=None)
    assert line.startswith("[decoder-opt] LEVER name=hybrid_gemm state=on impl=identity_probe origin=core probe=pass cells=2 bad=0 maxabs=0 policy=all served=2 reference=2 unprobed=1 undecided=1")
    assert P.gate(require_pass=True) == [] and P.gate(max_unprobed=0) == ["hybrid_gemm: unprobed=1 calls above the allowed 0 (shapes served by the reference form without a cell)"]
    with pytest.raises(ProbeClosed):
        P.cell(("B", 8), lambda: 0, lambda: 0)
    r = P.record()
    assert r["probe"] == "pass" and [c["key"] for c in r["cells"]] == ["Bx2", "Bx4"]


def test_fail_under_all_policy_serves_nothing_and_names_the_cells():
    P = IdentityProbe("hybrid_gemm")
    A, X = _ops(B=2)
    P.cell("good", lambda: torch.bmm(A, X), lambda: torch.stack([A[i] @ X[i] for i in range(2)]))
    P.cell("off", lambda: torch.bmm(A, X) + 1e-6, lambda: torch.stack([A[i] @ X[i] for i in range(2)]))
    P.cell("boom", lambda: (_ for _ in ()).throw(RuntimeError("no kernel")), lambda: A)
    assert P.decide("all") is False
    assert not P.serve("good")
    s = P.stats()
    assert s["probe"] == "fail" and s["bad"] == 2 and s["bad_keys"] == ["off", "boom"] and s["errors"] == ["RuntimeError"] and 0 < s["maxabs"] < 1e-5
    line = P.line("kit")
    assert " state=skipped reason=probe_fail impl=identity_probe origin=core probe=fail cells=3 bad=2 " in line
    g = P.gate(require_pass=True)
    assert len(g) == 1 and g[0].startswith("hybrid_gemm: probe fail (2/3 cells not bit-identical, keys off,boom")


def test_per_cell_policy_serves_exactly_the_agreeing_keys():
    P = IdentityProbe("x")
    A, X = _ops(B=2)
    P.cell("good", lambda: torch.bmm(A, X), lambda: torch.stack([A[i] @ X[i] for i in range(2)]))
    P.cell("off", lambda: torch.bmm(A, X) * 1.0000001, lambda: torch.stack([A[i] @ X[i] for i in range(2)]))
    assert P.decide("per_cell") is True
    assert P.serve("good") and not P.serve("off")
    assert P.stats()["probe"] == "partial" and " state=on impl=identity_probe origin=core probe=partial " in P.line("kit")
    assert P.gate(require_pass=True) == [] and P.gate(require_pass=True, allow_partial=False)
    with pytest.raises(ValueError):
        IdentityProbe("y").decide("most")


def test_undecided_and_empty_probe_words():
    P = IdentityProbe("z")
    assert P.stats()["probe"] == "undecided" and " state=skipped reason=undecided " in P.line("kit")
    assert P.decide("all") is False and P.stats()["probe"] == "no_cells" and P.gate(require_pass=True)
