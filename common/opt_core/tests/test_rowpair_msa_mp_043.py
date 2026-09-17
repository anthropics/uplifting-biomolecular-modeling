"""``m_layout='token_sharded'`` on gloo PROCESSES (the production CPU transport; the kit launcher is the launcher): every rank's OPM output rows
(local ``a``, gathered ``b``), pair-weighted-averaging m-update of ITS tokens and MSA transition on the token shard equal the dense slices
(fp32, <= 1e-5; bit-exact reported), the census names ``msa_m=token_sharded``, and the layout-word / operand mismatches refuse by name — the
same rank body as the threaded case of ``tests/test_rowpair_msa_043.py``, run in P in {2, 3} processes.

Run: ``python -m pytest tests/test_rowpair_msa_mp_043.py -q -rfE`` (torch + a host that permits gloo TCP; refused sockets -> the launcher's
``RankFailed`` names it).
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

torch = pytest.importorskip("torch")

TOL32 = 1e-5


def _mp_token_sharded_entry():
    """Runs in each gloo process: this rank's token-sharded statements vs the dense references (raised on mismatch -> RankFailed)."""
    import torch as _t
    import test_rowpair_msa_043 as T
    from opt_core.mem.rowpair import dist as DD
    _t.set_num_threads(1)
    P, r = DD.world()
    assert DD.comm().backend == "gloo", DD.comm().backend
    N, unit, s_chunk = 46, 4, 3
    m, z, msa_mask, pair_mask = T._inputs(N)                                  # seeded: identical in every process
    o = T._rank_token_sharded(r, P, N, unit, m, z, msa_mask, pair_mask, s_chunk)
    opm, _C = T._make_opm()
    dense_opm = opm.forward(m, msa_mask, unit)
    dense_pwa = T._make_pwa().forward(m, z, pair_mask, s_chunk)
    dense_tr = T._make_transition()(m, 0, N, msa_mask)
    r0, r1 = o["r0"], o["r1"]
    pairs = {"opm": (o["rows"], dense_opm[r0:r1]), "pwa": (o["upd_loc"], dense_pwa[:, r0:r1]), "transition": (o["tr_loc"], dense_tr[:, r0:r1])}
    diffs = {k: float((a - b).abs().max()) for k, (a, b) in pairs.items()}
    bitwise = {k: bool(_t.equal(a, b)) for k, (a, b) in pairs.items()}
    print(f"mp token_sharded m P={P} rank={r} tokens[{r0}:{r1}] " + " ".join(f"{k}={v:.3e}(eq={bitwise[k]})" for k, v in diffs.items()), flush=True)
    assert max(diffs.values()) <= TOL32, (r, diffs)
    assert o["msa_m_opm"] == "token_sharded" and o["msa_m_pwa"] == "token_sharded", o
    assert all(v is not None for v in o["refusals"].values()), o["refusals"]
    DD.barrier()
    return {"rank": r, "P": P, "diffs": diffs, "bitwise": bitwise, "tokens": (r0, r1)}


@pytest.mark.parametrize("P", [2, 3])
def test_mp_token_sharded_m_gloo(P):
    from opt_core.mem.rowpair import launch
    res = launch.run_sharded(P, _mp_token_sharded_entry, cpu_ok=True, backend="gloo", run_timeout_s=600)
    assert res["rank"] == 0 and res["P"] == P and max(res["diffs"].values()) <= TOL32, res
