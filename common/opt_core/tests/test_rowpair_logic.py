"""opt_core.mem.rowpair — logic tests: pure-Python layout / refusal / evidence checks in-process, and P in {2, 3} multi-process
checks under gloo on CPU (the launcher itself is the launcher: ``launch.run_sharded(P, entry, cpu_ok=True, backend="gloo")``).
Data movement is asserted BIT-EXACT (torch.equal); fp64 CPU statements are asserted within 1e-9 and their bit-exact flag is REPORTED.

Run: ``python -m pytest tests/test_rowpair_logic.py -q`` (torch required for the ``mp_*`` / tensor tests; the pure tests need nothing) or
``python tests/test_rowpair_logic.py`` (prints one RESULT line per case and a SUMMARY line; rc != 0 on any failure).
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # the opt_core checkout under test

from opt_core.mem.rowpair import (REFUSE_MODE, NGpuRefused, RowpairRefused, refuse_unless_big, refuse_unless_visible)  # noqa: E402
from opt_core.mem import ngpu  # noqa: E402
from opt_core.mem.rowpair import evidence, launch  # noqa: E402
from opt_core.mem.rowpair.dist import Layout  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False

TOL64 = 1e-9


# ================================================================================================================ pure (no torch)
def test_layout_grid():
    for N in (1, 15, 16, 127, 128, 129, 300, 383, 1000, 2500):
        for P in (1, 2, 3, 4, 8):
            for B in (16, 64, 128):
                lays = [Layout(N, P, q, B) for q in range(P)]
                b = lays[0].bounds
                assert all(l.bounds == b for l in lays)
                if N < P * B:
                    assert all(l.replicated for l in lays) and all(x == (0, N) for x in b)
                    continue
                assert b[0][0] == 0 and b[-1][1] == N
                for q in range(P - 1):
                    assert b[q][1] == b[q + 1][0]
                for q0, q1 in b:
                    assert q0 % B == 0 or q0 == N
                    assert q1 % B == 0 or q1 == N
                assert sum(q1 - q0 for q0, q1 in b) == N
                # chunks(): a partition of each rank's rows on the GLOBAL chunk grid
                for l in lays:
                    for chunk in (16, 32, B):
                        cs = list(l.chunks(chunk))
                        assert sum(c1 - c0 for c0, c1 in cs) == l.R
                        assert all(l.r0 <= c0 < c1 <= l.r1 for c0, c1 in cs)
                        if B % chunk == 0:
                            assert all(c0 % chunk == 0 for c0, c1 in cs)
                # owner()
                for i in range(0, N, max(1, N // 17)):
                    q = lays[0].owner(i)
                    assert b[q][0] <= i < b[q][1], (N, P, B, i, q, b)


LADDER_BINS = (1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000)      # the memory ladder's token bins


def test_layout_auto_has_a_plan_for_every_ladder_bin():
    """F8-05: every bin x P in {2,4,8} gets a valid, non-empty, P-consistent plan from Layout.auto (B from 128/64/32/16)."""
    table = {}
    for N in LADDER_BINS:
        for P in (2, 4, 8):
            lays = [Layout.auto(N, P, q) for q in range(P)]
            B = lays[0].B
            assert all(l.B == B and l.bounds == lays[0].bounds for l in lays)
            assert not lays[0].replicated and all(l.R > 0 for l in lays), (N, P, B)
            assert lays[0].bounds[0][0] == 0 and lays[0].bounds[-1][1] == N
            table[(N, P)] = B
    assert table[(1500, 8)] == 64 and table[(1000, 8)] == 64 and table[(2000, 8)] == 128 and table[(16000, 2)] == 128, table
    # chunk-constrained candidates and the refusal words
    assert Layout.auto(1500, 8, 0, chunk=32).B == 64
    try:
        Layout.auto(100, 8, 0)
        raise AssertionError("expected refusal")
    except RowpairRefused as e:
        assert "no valid row grid among B=128:replicated,B=64:replicated,B=32:replicated,B=16:" in e.reason, e.reason
    assert Layout.auto(777, 1, 0).P == 1


def test_layout_checked_refusals():
    Layout.checked(1000, 1, 0)                                    # P == 1 always passes
    Layout.checked(1000, 4, 0, 128)
    for (N, P, B) in ((100, 2, 128), (255, 2, 128)):              # replicates
        try:
            Layout.checked(N, P, 0, B)
            raise AssertionError("expected refusal")
        except RowpairRefused as e:
            assert "cannot shard" in e.reason
    try:                                                          # 5 blocks over 4 ranks at bpr=2 -> rank 3 empty
        Layout.checked(5 * 128, 4, 0, 128)
        raise AssertionError("expected refusal")
    except RowpairRefused as e:
        assert "zero rows" in e.reason


def test_refusal_words():
    assert refuse_unless_big(1, "exact") == 1 and refuse_unless_big("1", "fast") == 1 and refuse_unless_big(4, "big") == 4
    for mode in ("exact", "fast", "", None, "BIGX"):
        try:
            refuse_unless_big(2, mode)
            raise AssertionError("expected refusal")
        except NGpuRefused as e:
            assert e.reason == REFUSE_MODE == "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)"
    try:
        refuse_unless_visible(4, visible=1)
        raise AssertionError("expected refusal")
    except NGpuRefused as e:
        assert e.reason == "refused: n_gpu=4 visible=1"
    assert ngpu.active_fields(1) == "n_gpu=1 sharding=none" and ngpu.active_fields(4) == "n_gpu=4 sharding=rowpair"
    assert ngpu.active_fields(2, "foldcp2d") == "n_gpu=2 sharding=foldcp2d"
    assert issubclass(RowpairRefused, NGpuRefused) and issubclass(NGpuRefused, __import__("opt_core.mem", fromlist=["MemLeverRefused"]).MemLeverRefused)
    assert refuse_unless_visible(2, visible=2) == 2 and refuse_unless_visible(1, visible=0) == 1
    for bad in (0, -1, "x", None):
        try:
            refuse_unless_big(bad, "big")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_evidence_tokens():
    assert evidence.fields_text(1) == "n_gpu=1 sharding=none"
    assert evidence.fields_text(2) == "n_gpu=2 sharding=rowpair"
    assert evidence.fields_text(8, B=128) == "n_gpu=8 sharding=rowpair B=128"
    import opt_core.mem.rowpair as rp
    evidence.reset_schedule()                                                                   # the census is process-wide: start from this test's facts
    evidence.record_schedule(trimul_RA=256, trimul_RB=128, tiles_source="fixed")
    ln = evidence.lever_line("KIT-OPT", "on", 4, layout=Layout(1024, 4, 1), peaks_gib={0: 10.5, 1: 10.25})
    assert ln.startswith(f"[KIT-OPT] LEVER name={rp.LEVER} state=on impl=opt_core.mem.rowpair@"), ln
    assert f" origin=core strategy={rp.LEVER} n_gpu=4 sharding=rowpair " in ln, ln          # keyed for the strategy matrix (canonical id)
    assert (" N=1024 P=4 B=128 rank=1 rows=256:512 R=256 Rmax=256 peak_alloc_gib_max=10.5 peak_alloc_gib_ranks=r0:10.50,r1:10.25"
            " trimul_RA=256 trimul_RB=128 tiles_source=fixed") in ln, ln                     # the numerics-relevant schedule is printed
    assert " trimul_RA=" not in evidence.lever_line("KIT-OPT", "on", 4, with_schedule=False)
    off = evidence.lever_line("KIT-OPT", "skipped", 2, reason="refused:n_gpu=2_visible=1")
    assert "state=skipped reason=refused:n_gpu=2_visible=1" in off
    e = launch.RankFailed("rank_failed", 1, 3, "boom", "/tmp/x/rank1.log")
    assert evidence.rank_failed_line("KIT-OPT", e) == "[KIT-OPT] ROWPAIR event=rank_failed rank=1 exitcode=3 log=/tmp/x/rank1.log", evidence.rank_failed_line("KIT-OPT", e)


def test_arch_declarations():
    from opt_core import arch
    import opt_core.mem.rowpair as rp
    from opt_core import strategies
    for lever in (rp.LEVER,) + tuple(rp.SUB_LEVERS.values()):
        for sm in ("sm80", "sm90", "sm100", "sm103"):                  # declared, no floor, no exclusion; test records: none yet
            want = "supported" if sm in rp.CERTIFIED_SM else "uncertified"
            assert arch.supports(lever, sm).word == want, (lever, sm, arch.supports(lever, sm).word)
    st = arch.lever_state(rp.LEVER, "sm100")
    assert st["state"] == "on" and st["evidence"]["card_support"] == "uncertified:sm100", st
    rp._declare_arch()                                               # idempotent (same content)
    assert rp.LEVER == "F7.tensor_parallel"
    for lever in (rp.LEVER,) + tuple(rp.SUB_LEVERS.values()):        # the TP lever and every sub-lever id are catalogue ids (the LEVER lines are keyed for the strategy matrix)
        assert strategies.check(lever) == lever, lever
    assert rp.sub_strategy("trimul_ring") == rp.SUB_LEVERS["trimul_ring"] == "F7.tp_trimul_ring_contract"


def test_clean_venv_import_without_torch():
    """`import opt_core`, `opt_core.mem`, `opt_core.mem.rowpair` and every rowpair submodule succeed with NO torch importable (a fresh
    interpreter with torch masked); a primitive called there refuses BY NAME (RowpairRefused naming torch), never an ImportError at import.
    rowpair holds no model-keyed recipe: every engine mapping is the kit's adapter."""
    import subprocess
    code = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('masked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"sys.path.insert(0, {os.path.dirname(HERE)!r})\n"
        "import opt_core, opt_core.mem, opt_core.mem.ngpu, opt_core.mem.rowpair as rp\n"
        "from opt_core.mem.rowpair import launch, dist, shard, trimul, triatt, transition, confidence, bcast, ckpt, evidence\n"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m.startswith('torch'))\n"
        "assert evidence.fields_text(2) == 'n_gpu=2 sharding=rowpair'\n"
        "assert launch.run_sharded(1, (lambda x: x + 1), 1, mode='fast') == 2\n"
        "try:\n"
        "    dist.Layout(8, 1, 0); trimul.trimul_dense(None, None, True)\n"
        "    raise SystemExit('expected a refusal')\n"
        "except rp.RowpairRefused as e:\n"
        "    assert 'torch is not importable' in e.reason, e.reason\n"
        "print('CLEAN-VENV-OK')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "CLEAN-VENV-OK" in r.stdout, (r.returncode, r.stdout[-500:], r.stderr[-2000:])


def test_p1_passthrough_is_inprocess():
    env0 = dict(os.environ)
    calls = []

    def entry(a, k=None):
        calls.append((os.getpid(), a, k))
        return {"pid": os.getpid()}

    out = launch.run_sharded(1, entry, 7, k="v", mode="exact")     # P == 1 passes under ANY mode
    assert out["pid"] == os.getpid() and calls == [(os.getpid(), 7, "v")]
    assert dict(os.environ) == env0                                  # no environment change
    assert "torch.distributed" not in sys.modules or not __import__("torch").distributed.is_initialized()
    try:
        launch.run_sharded(2, entry, 7, mode="fast")
        raise AssertionError("expected refusal")
    except NGpuRefused as e:
        assert e.reason == REFUSE_MODE


# ================================================================================================================ multi-process (gloo, CPU)
def _seeded(N, C, seed, dtype=None, device="cpu"):
    """Seeded on CPU (identical on every rank), then moved: replicated by construction."""
    import torch
    g = torch.Generator().manual_seed(seed)
    return torch.randn((N, N, C), generator=g, dtype=torch.float64).to(dtype=dtype or torch.float64).to(device)


def _dev():
    """cuda:<current> when the group is NCCL / CUDA is bound by the launcher, else cpu."""
    import torch
    if torch.cuda.is_available() and os.environ.get("ROWPAIR_TEST_DEVICE", "auto") != "cpu":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _tol(dtype, ref_scale=1.0):
    """Sharded-vs-dense max-abs class per dtype: fp64 1e-9 abs; fp32 1e-4*scale; bf16/fp16 2e-2*scale (rounding class of the dtype)."""
    import torch
    return {torch.float64: TOL64, torch.float32: 1e-4 * max(1.0, ref_scale), torch.bfloat16: 2e-2 * max(1.0, ref_scale),
            torch.float16: 2e-2 * max(1.0, ref_scale)}[dtype]


def _rtol_atol(dtype):
    """Elementwise class (torch.testing's defaults per dtype): fp64 (1e-7, 1e-7); fp32 (1.3e-6, 1e-5); bf16 (1.6e-2, 1e-5); fp16 (1e-3, 1e-5)."""
    import torch
    return {torch.float64: (1e-7, 1e-7), torch.float32: (1.3e-6, 1e-5), torch.bfloat16: (1.6e-2, 1e-5), torch.float16: (1e-3, 1e-5)}[dtype]


def _cmp(got, dense, ref64=None):
    """Sharded vs dense (same dtype) and, when given, vs the fp64 reference: ``{bitwise_vs_dense, frac_diff, maxabs_vs_dense, allclose,
    [maxabs_sharded_vs_fp64, maxabs_dense_vs_fp64, ratio_ok]}``. ``ratio_ok`` = sharded-vs-fp64 error <= 2 x dense-vs-fp64 error + 1e-6 (the
    P-independent correctness test: sharding may not make the statement less accurate than the dense statement is); ``allclose`` =
    elementwise |got-dense| <= atol + rtol*|dense| with the dtype's class, EXCEPT that two roundings of the same fp64 value may differ by one
    ulp of the dtype, so the elementwise test is taken against the fp64 reference when one is given."""
    import torch
    rtol, atol = _rtol_atol(got.dtype)
    m = {"bitwise_vs_dense": bool(torch.equal(got, dense)), "frac_diff": float((got != dense).float().mean().item()),
         "maxabs_vs_dense": (got.double() - dense.double()).abs().max().item()}
    if ref64 is not None:
        e_sh = (got.double() - ref64).abs().max().item()
        e_de = (dense.double() - ref64).abs().max().item()
        m.update(maxabs_sharded_vs_fp64=e_sh, maxabs_dense_vs_fp64=e_de, ratio_ok=bool(e_sh <= 2.0 * e_de + 1e-6), ref_maxabs=ref64.abs().max().item(),
                 allclose=bool(torch.allclose(got.double(), ref64, rtol=max(rtol, 4 * e_de / max(1e-30, ref64.abs().max().item())), atol=atol + 2 * e_de)))
    else:
        m.update(allclose=bool(torch.allclose(got.double(), dense.double(), rtol=rtol, atol=atol)))
    return m


def _ok(m, tol):
    """A case passes when max-abs is inside the dtype class AND the elementwise test holds AND (when present) the fp64 ratio test holds."""
    return m["maxabs_vs_dense"] <= tol and m["allclose"] and m.get("ratio_ok", True)


def _entry_dist(case: str, N: int, B: int):
    """Runs on every rank; returns rank 0's result dict."""
    import torch
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair.shard import shard_rows, unshard_rows, unshard_rows_to_rank0, local_rows, local_rows_
    P, r = D.world()
    lay = D.Layout.checked(N, P, r, B)
    dev = _dev()
    full = _seeded(N, 4, 11, device=dev)                             # identical on every rank (seeded)
    mine = shard_rows(full, lay).contiguous()
    res = {"case": case, "P": P, "N": N, "B": B, "device": str(dev), "checks": {}}
    ck = res["checks"]
    if case == "gathers":
        ck["all_gather_rows"] = bool(torch.equal(unshard_rows(mine, lay), full))
        g0 = unshard_rows_to_rank0(mine, lay)
        ck["gather_rows_to_rank0"] = bool(torch.equal(g0, full)) if r == 0 else (g0 is None)
        ck["unshard_dim1"] = bool(torch.equal(unshard_rows(shard_rows(full, lay, dim=1).contiguous(), lay, dim=1), full))
        counts = [lay.nrows(q) for q in range(P)]
        cat = D.gather_cat_to_rank0(mine, counts)
        ck["gather_cat_to_rank0"] = bool(torch.equal(cat, full)) if r == 0 else (cat is None)
        ck["broadcast_obj"] = D.broadcast_obj({"x": 5} if r == 0 else None) == {"x": 5}
        D.allreduce_checksum(full, "full")                            # must pass
        bad = full.clone()
        if r == 1:
            bad[0, 0, 0] += 1.0
        try:
            D.allreduce_checksum(bad, "bad")
            ck["checksum_mismatch_refused"] = False
        except RowpairRefused:
            ck["checksum_mismatch_refused"] = True
        y = local_rows(lambda x: x * 2.0 + 1.0, mine, rows=7)
        ck["local_rows_blocked"] = bool(torch.equal(y, mine * 2.0 + 1.0))
        z = mine.clone()
        local_rows_(lambda x: x * 3.0, z, rows=5, add=True)
        ck["local_rows_inplace_add"] = bool(torch.equal(z, mine + mine * 3.0))
    elif case == "transpose":
        fullT = full.transpose(0, 1)
        for chunks in (1, 3):
            zt = D.transpose_shards(mine, lay, chunks=chunks)
            ck[f"transpose_shards_chunks{chunks}"] = bool(torch.equal(zt, fullT[lay.r0:lay.r1]))
        zt2 = D.transpose_shards(mine, lay, block_rows=B)
        ck["transpose_shards_blockrows"] = bool(torch.equal(zt2, fullT[lay.r0:lay.r1]))
        ok = True
        seen = 0
        for i0, i1, blk in D.transpose_blocks(mine, lay, step=max(1, B // 2)):
            ok = ok and bool(torch.equal(blk, fullT[lay.r0 + i0: lay.r0 + i1]))
            seen += i1 - i0
        ck["transpose_blocks"] = ok and seen == lay.R
        for lo in ("rows", "gemm_a", "gemm_b"):
            w = D.alltoall_window(lambda c0, c1: mine[:, c0:c1, :], lay, 0, None, C=4, dtype=full.dtype, device=full.device, out_layout=lo, chunks=2)
            ref = fullT[lay.r0:lay.r1]                                # [R, N, C]: ref[w, j, c] = X[j, r0+w, c]
            exp = {"rows": ref, "gemm_a": ref.permute(2, 0, 1), "gemm_b": ref.permute(2, 1, 0)}[lo]
            ck[f"alltoall_window_{lo}"] = bool(torch.equal(w, exp.contiguous()))
        # windowed: off/width inside each rank's rows
        w = D.alltoall_window(lambda c0, c1: mine[:, c0:c1, :], lay, 3, 20, C=4, dtype=full.dtype, device=full.device, out_layout="rows", chunks=1)
        a, b = min(lay.r1, lay.r0 + 3), min(lay.r1, lay.r0 + 23)
        ck["alltoall_window_offset"] = bool(torch.equal(w, fullT[a:b]))
    elif case == "ring":
        got = {}
        for q, blk in D.ring_blocks(mine, lay):
            got[q] = blk.clone()
        ck["ring_every_rank_once"] = sorted(got) == list(range(P))
        ck["ring_content"] = all(bool(torch.equal(got[q], full[lay.bounds[q][0]:lay.bounds[q][1]])) for q in got)
        RB = max(1, B // 2)
        j0 = RB                                                        # second sub-slab of every rank
        sub = mine[j0:j0 + RB]
        got2 = {}
        for q, blk in D.ring_blocks(sub, lay, rows=(j0, RB)):
            got2[q] = blk.clone()
        exp2 = {q: full[lay.bounds[q][0] + j0: min(lay.bounds[q][1], lay.bounds[q][0] + j0 + RB)] for q in range(P)}
        ck["ring_subslab"] = sorted(got2) == list(range(P)) and all(bool(torch.equal(got2[q], exp2[q])) for q in got2)
    res["ok"] = all(ck.values())
    return res


def _entry_trimul(N: int, C: int, B: int, RA: int, RB: int, dtype_name: str):
    import torch
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair.shard import shard_rows, unshard_rows
    from opt_core.mem.rowpair.trimul import trimul_dense, trimul_incoming, trimul_outgoing
    dtype = getattr(torch, dtype_name)
    P, r = D.world()
    lay = D.Layout.checked(N, P, r, B)
    dev = _dev()
    a, b = _seeded(N, C, 3, dtype, dev), _seeded(N, C, 4, dtype, dev)
    res = {"case": "trimul", "P": P, "N": N, "C": C, "B": B, "RA": RA, "RB": RB, "dtype": dtype_name, "device": str(dev),
           "tf32": bool(torch.backends.cuda.matmul.allow_tf32), "checks": {}, "metrics": {}}
    for outgoing in (True, False):
        dense = trimul_dense(a, b, outgoing)
        ref64 = trimul_dense(a.double(), b.double(), outgoing)
        fn = trimul_outgoing if outgoing else trimul_incoming
        out, st = fn(shard_rows(a, lay).contiguous(), shard_rows(b, lay).contiguous(), lay, RA=RA, RB=RB)
        full = unshard_rows(out, lay)
        tag = "outgoing" if outgoing else "incoming"
        m = _cmp(full, dense, ref64)
        m["stats"] = st.facts()
        res["metrics"][tag] = m
        tol = _tol(dtype, ref64.abs().max().item())
        res["checks"][f"{tag}_within_tol"] = _ok(m, tol)
        res["checks"][f"{tag}_tiles"] = st.tiles > 0
        # epilogue form: accumulate into a preexisting shard (residual add)
        acc = shard_rows(a, lay).contiguous().clone()

        def epi(T, rows, cols, acc=acc):
            acc[rows[0]:rows[1], cols[0]:cols[1], :] += T.permute(1, 2, 0)

        fn(shard_rows(a, lay).contiguous(), shard_rows(b, lay).contiguous(), lay, epi, RA=RA, RB=RB)
        res["checks"][f"{tag}_epilogue_add"] = (unshard_rows(acc, lay).double() - (a.double() + dense.double())).abs().max().item() <= max(tol, 1e-12)
    res["ok"] = all(res["checks"].values())
    return res


def _attn_rows_factory(Wq, Wk, Wv, Wg, H):
    """The reference starting-node statement for rows g0:g1 given the whole bias: softmax_k(q_ij.k_ik/sqrt(d) + tb[j,k,h]) v_ik, gated."""
    import torch

    def attn_rows(z_rows, tb_full, span):
        rows, N, C = z_rows.shape
        d = C // H
        q = (z_rows @ Wq).view(rows, N, H, d)
        k = (z_rows @ Wk).view(rows, N, H, d)
        v = (z_rows @ Wv).view(rows, N, H, d)
        logits = torch.einsum("ijhd,ikhd->ihjk", q, k) / (d ** 0.5) + tb_full.permute(2, 0, 1).unsqueeze(0)   # [rows, H, N(j), N(k)]
        w = torch.softmax(logits, dim=-1)
        o = torch.einsum("ihjk,ikhd->ijhd", w, v).reshape(rows, N, C)
        return o * torch.sigmoid(z_rows @ Wg)

    return attn_rows


def _entry_triatt(N: int, C: int, H: int, B: int, q_rows: int, dtype_name: str = "float64"):
    import torch
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair.shard import shard_rows, unshard_rows
    from opt_core.mem.rowpair.triatt import gather_triangle_bias, triatt_dense, triatt_ending, triatt_starting
    dtype = getattr(torch, dtype_name)
    P, r = D.world()
    lay = D.Layout.checked(N, P, r, B)
    dev = _dev()
    g = torch.Generator().manual_seed(9)
    Wq, Wk, Wv, Wg = ((torch.randn((C, C), generator=g, dtype=torch.float64) / C ** 0.5).to(dtype).to(dev) for _ in range(4))
    Wb = (torch.randn((C, H), generator=g, dtype=torch.float64) / C ** 0.5).to(dtype).to(dev)
    attn_rows = _attn_rows_factory(Wq, Wk, Wv, Wg, H)
    attn_rows64 = _attn_rows_factory(Wq.double(), Wk.double(), Wv.double(), Wg.double(), H)
    tb_of = lambda rows: rows @ Wb  # noqa: E731
    tb_of64 = lambda rows: rows @ Wb.double()  # noqa: E731
    z = _seeded(N, C, 5, dtype, dev)
    res = {"case": "triatt", "P": P, "N": N, "C": C, "H": H, "B": B, "q_rows": q_rows, "dtype": dtype_name, "device": str(dev),
           "checks": {}, "metrics": {}}
    # starting
    dense = triatt_dense(attn_rows, z, tb_of(z))
    ref64 = triatt_dense(attn_rows64, z.double(), tb_of64(z.double()))
    tol = _tol(dtype, ref64.abs().max().item())
    mine = shard_rows(z, lay).contiguous()
    tb_full = gather_triangle_bias(tb_of(mine), lay)
    res["checks"]["bias_gather_bitwise"] = bool(torch.equal(tb_full, tb_of(z)))
    out = triatt_starting(attn_rows, mine, tb_full, lay, q_rows=q_rows)
    full = unshard_rows(out, lay)
    res["metrics"]["starting"] = _cmp(full, dense, ref64)
    res["checks"]["starting_within_tol"] = _ok(res["metrics"]["starting"], tol)
    # ending = the starting statement on z^T, transposed back
    zT = z.transpose(0, 1).contiguous()
    denseE = triatt_dense(attn_rows, zT, tb_of(zT)).transpose(0, 1).contiguous()
    outE = triatt_ending(attn_rows, mine, tb_of, lay, q_rows=q_rows)
    fullE = unshard_rows(outE.contiguous(), lay)
    res["metrics"]["ending"] = _cmp(fullE, denseE)
    res["checks"]["ending_within_tol"] = _ok(res["metrics"]["ending"], tol)
    # in-place residual form
    zz = mine.clone()
    triatt_starting(attn_rows, zz, tb_full, lay, q_rows=q_rows, inplace_add=True)
    res["checks"]["starting_inplace_add"] = (unshard_rows(zz, lay).double() - (z + dense).double()).abs().max().item() <= tol
    res["ok"] = all(res["checks"].values())
    return res


def _entry_transition(N: int, C: int, B: int, dtype_name: str = "float64"):
    import torch
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair.shard import shard_rows, unshard_rows
    from opt_core.mem.rowpair.transition import apb_local_queries, opm_rows, transition_rows
    dtype = getattr(torch, dtype_name)
    P, r = D.world()
    lay = D.Layout.checked(N, P, r, B)
    dev = _dev()
    g = torch.Generator().manual_seed(21)
    rnd = lambda *shape: torch.randn(shape, generator=g, dtype=torch.float64)  # noqa: E731  (CPU seeded -> identical on every rank)
    W1 = (rnd(C, 4 * C) / C ** 0.5).to(dtype).to(dev)
    W2 = (rnd(4 * C, C) / C ** 0.5).to(dtype).to(dev)
    ln = torch.nn.LayerNorm(C, dtype=dtype, device=dev)
    with torch.no_grad():
        ln.weight.copy_(rnd(C).to(dtype))
        ln.bias.copy_(rnd(C).to(dtype))
    fn = lambda x: torch.relu(ln(x) @ W1) @ W2  # noqa: E731
    z = _seeded(N, C, 6, dtype, dev)
    tol = _tol(dtype, float(z.abs().max()))
    res = {"case": "transition", "P": P, "N": N, "C": C, "B": B, "dtype": dtype_name, "device": str(dev), "checks": {}, "metrics": {}}
    with torch.no_grad():
        dense = z + fn(z)
        mine = shard_rows(z, lay).contiguous().clone()
        transition_rows(fn, mine, lay, rows=13, add=True)
        full = unshard_rows(mine, lay)
        res["metrics"]["transition"] = _cmp(full, dense)
        res["checks"]["transition_within_tol"] = _ok(res["metrics"]["transition"], tol)
        # OPM output rows: a,b [S, N, c] -> out[i,j,:] = flatten(mean_s a_si (x) b_sj) @ Wo
        S, c, Cz = 6, 3, 5
        a = rnd(S, N, c).to(dtype).to(dev)
        b = rnd(S, N, c).to(dtype).to(dev)
        Wo = rnd(c * c, Cz).to(dtype).to(dev)
        outer = lambda ar, bb: (torch.einsum("sic,sjd->ijcd", ar, bb) / S).reshape(ar.shape[1], bb.shape[1], c * c) @ Wo  # noqa: E731
        dense_opm = outer(a, b)
        mine_opm = opm_rows(a, b, lay, outer, rows=11)
        full_opm = unshard_rows(mine_opm, lay)
        ref_opm = None if dtype == torch.float64 else (torch.einsum("sic,sjd->ijcd", a.double(), b.double()) / S).reshape(N, N, c * c) @ Wo.double()
        res["metrics"]["opm_rows"] = _cmp(full_opm, dense_opm, ref_opm)
        res["checks"]["opm_within_tol"] = _ok(res["metrics"]["opm_rows"], tol)
        # APB: queries local rows, keys all, bias = z rows (mean over channels as a 1-head bias)
        cs = 8
        s = rnd(N, cs).to(dtype).to(dev)
        Wq2, Wk2, Wv2 = ((rnd(cs, cs) / cs ** 0.5).to(dtype).to(dev) for _ in range(3))

        def attn_fn(q_rows, s_all, bias_rows):
            logits = (q_rows @ Wq2) @ (s_all @ Wk2).T / cs ** 0.5 + bias_rows.mean(-1)
            return torch.softmax(logits, -1) @ (s_all @ Wv2)

        dense_apb = attn_fn(s, s, z)
        got = apb_local_queries(attn_fn, s, shard_rows(z, lay), lay, gather=True)
        ref_apb = None
        if dtype != torch.float64:
            s64, z64, W64 = s.double(), z.double(), [w.double() for w in (Wq2, Wk2, Wv2)]
            lg = (s64 @ W64[0]) @ (s64 @ W64[1]).T / cs ** 0.5 + z64.mean(-1)
            ref_apb = torch.softmax(lg, -1) @ (s64 @ W64[2])
        res["metrics"]["apb_local_queries"] = _cmp(got, dense_apb, ref_apb)
        res["checks"]["apb_within_tol"] = _ok(res["metrics"]["apb_local_queries"], tol)
    res["ok"] = all(res["checks"].values())
    return res


def _entry_rank(x):
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair import evidence as EV
    P, r = D.world()
    cen = dict(EV.rows_census(D.Layout(512, P, r)))
    return {"rank": r, "P": P, "x": x, "env_rank": os.environ.get("ROWPAIR_RANK"), "is_output_rank": launch.is_output_rank(),
            "torchrun_names_leaked": any(n in os.environ for n in ("RANK", "WORLD_SIZE", "MASTER_PORT")), "rows_tiled": cen["rows_tiled"],
            "ranks": cen["ranks"]}


def _entry_fail_rank1():
    from opt_core.mem.rowpair import dist as D
    P, r = D.world()
    if r == 1:
        raise RuntimeError("rank one fails by design")
    time.sleep(60)                                                   # rank 0 would hang here; the launcher must tear it down
    return "unreachable"


def _mp(P, entry, *args, **kw):
    """P ranks through the launcher itself. Device / backend: CUDA + NCCL when at least P devices are visible (and ROWPAIR_TEST_DEVICE is not
    'cpu'), else CPU + gloo (gloo takes CPU tensors only) — the choice is inherited by the workers through ROWPAIR_TEST_DEVICE."""
    ncuda = torch.cuda.device_count() if HAVE_TORCH and torch.cuda.is_available() else 0
    use_cuda = ncuda >= P and os.environ.get("ROWPAIR_TEST_DEVICE", "auto") != "cpu"
    prev = os.environ.get("ROWPAIR_TEST_DEVICE")
    os.environ["ROWPAIR_TEST_DEVICE"] = "auto" if use_cuda else "cpu"
    try:
        return launch.run_sharded(P, entry, *args, mode="big", backend="nccl" if use_cuda else "gloo", cpu_ok=not use_cuda,
                                  nccl_timeout_s=120, run_timeout_s=600, **kw)
    finally:
        if prev is None:
            os.environ.pop("ROWPAIR_TEST_DEVICE", None)
        else:
            os.environ["ROWPAIR_TEST_DEVICE"] = prev


def _report(res):
    print("RESULT " + json.dumps(res, sort_keys=True, default=str))
    assert res["ok"], res


import pytest  # noqa: E402  (only the decorators below need it)

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch not importable")


@needs_torch
@pytest.mark.parametrize("P", [2, 3])
def test_mp_launcher_result_and_ranks(P):
    for n in launch.TORCHRUN_NAMES:
        os.environ.pop(n, None)
    out = _mp(P, _entry_rank, 5)
    assert out == {"rank": 0, "P": P, "x": 5, "env_rank": "0", "is_output_rank": True, "torchrun_names_leaked": False, "rows_tiled": True,
                   "ranks": P}, out


@needs_torch
def test_mp_launcher_rank_failure_tears_down():
    import multiprocessing
    t0 = time.monotonic()
    try:
        _mp(2, _entry_fail_rank1)
        raise AssertionError("expected RankFailed")
    except launch.RankFailed as e:
        assert e.event == "rank_failed" and e.rank == 1, (e.event, e.rank, e.exitcode)
        assert "rank one fails by design" in e.detail or "rank one fails by design" in open(e.log).read()
    assert time.monotonic() - t0 < 45, "teardown must not wait for the hung rank"
    assert not multiprocessing.active_children(), multiprocessing.active_children()


@needs_torch
def test_mp_run_rank_processes():
    import tempfile
    d = tempfile.mkdtemp(prefix="rowpair_test_")
    lines = []
    recs = launch.run_rank_processes(2, [sys.executable, "-c", "import os,sys; assert 'RANK' not in os.environ and 'MASTER_PORT' not in os.environ, 'torchrun names leaked'; print('RANKLINE', os.environ['ROWPAIR_RANK'], os.environ['ROWPAIR_WORLD'], os.environ['ROWPAIR_LOCAL_RANK']); sys.exit(0)"],
                                    mode="big", log_dir=d, cpu_ok=True, on_line=lines.append, run_timeout_s=60, env={k: v for k, v in os.environ.items() if k not in launch.TORCHRUN_NAMES})
    assert [r["rc"] for r in recs] == [0, 0] and any("RANKLINE 0 2 0" in ln for ln in lines)
    assert "RANKLINE 1 2 1" in open(os.path.join(d, "rank1.log")).read()
    try:
        launch.run_rank_processes(2, argv_of=lambda r: [sys.executable, "-c", f"import sys,time; time.sleep(0 if {r}==1 else 30); sys.exit(7 if {r}==1 else 0)"],
                                  mode="big", log_dir=d, cpu_ok=True, run_timeout_s=60, fail_grace_s=2.0)   # rank 1 EXITS 7: rank 0 gets the failure grace, then is torn down
        raise AssertionError("expected RankFailed")
    except launch.RankFailed as e:
        assert e.rank == 1 and e.exitcode == 7 and e.event == "rank_failed", (e.rank, e.exitcode, e.event)
    try:
        launch.run_rank_processes(2, [sys.executable, "-c", "pass"], mode="exact", cpu_ok=True)
        raise AssertionError("expected refusal")
    except NGpuRefused as e:
        assert e.reason == REFUSE_MODE


@needs_torch
@pytest.mark.parametrize("P,N,B", [(2, 256, 128), (2, 300, 64), (3, 384, 128), (3, 200, 16), (4, 512, 64)])
@pytest.mark.parametrize("case", ["gathers", "transpose", "ring"])
def test_mp_dist(case, P, N, B):
    _report(_mp(P, _entry_dist, case, N, B))


@needs_torch
@pytest.mark.parametrize("P,N,B,RA,RB", [(2, 256, 128, 64, 64), (2, 320, 64, 48, 40), (3, 384, 128, 128, 32)])
@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
def test_mp_trimul(P, N, B, RA, RB, dtype_name):
    _report(_mp(P, _entry_trimul, N, 8, B, RA, RB, dtype_name))


@needs_torch
def test_p1_l2_entry_points_refuse_by_name():
    """The structural n_gpu=1 rule: every L2 statement refuses a P == 1 / no-group layout BY NAME (the adapter installs nothing at n_gpu=1)."""
    import torch
    from opt_core.mem.rowpair import dist as D, transition as TR, triatt as TA, trimul as TM
    from opt_core.mem.rowpair.confidence import ChainIndex, RowBlockReducer
    lay = D.Layout(64, 1, 0)
    a = torch.zeros((64, 64, 4))
    cases = {
        "trimul_outgoing": lambda: TM.trimul_outgoing(a, a, lay, RA=32, RB=32),
        "trimul_incoming": lambda: TM.trimul_incoming(a, a, lay, RA=32, RB=32),
        "gather_triangle_bias": lambda: TA.gather_triangle_bias(torch.zeros((64, 64, 2)), lay),
        "triatt_starting": lambda: TA.triatt_starting(lambda z, tb, g: z, a, None, lay),
        "transition_rows": lambda: TR.transition_rows(lambda x: x, a, lay),
        "opm_rows": lambda: TR.opm_rows(torch.zeros((2, 64, 3)), torch.zeros((2, 64, 3)), lay, lambda x, y: x),
        "apb_local_queries": lambda: TR.apb_local_queries(lambda q, s, z: q, torch.zeros((64, 8)), a, lay),
        "RowBlockReducer": lambda: RowBlockReducer(ChainIndex(torch.zeros(64, dtype=torch.long), torch.ones(64, dtype=torch.bool)), 0, 64, "cpu"),
    }
    for name, call in cases.items():
        try:
            call()
            raise AssertionError(f"{name}: expected a refusal at P=1")
        except RowpairRefused as e:
            assert "n_gpu=1" in e.reason and "installs nothing" in e.reason, (name, e.reason)
    # L1 seams stay the equality at P=1 (true claims): shard/unshard/all_gather_rows/transpose
    from opt_core.mem.rowpair.shard import shard_rows, unshard_rows
    z = torch.arange(64 * 64 * 2.).reshape(64, 64, 2)
    assert shard_rows(z, lay) is z and unshard_rows(z, lay) is z and D.all_gather_rows(z, lay) is z
    assert torch.equal(D.transpose_shards(z, lay), z.transpose(0, 1))


@needs_torch
def test_p1_confidence_single_device_optin_is_dense():
    """A kit MAY ship the reducer as its own single-device memory lever (allow_unsharded=True, named in its line): then it reproduces the
    dense head (exact finish within the stated class; bit-exact reported)."""
    for finish in ("exact", "rowsum"):
        _report(launch.run_sharded(1, _entry_confidence, 200, 3, 128, finish, 40, True, mode="big"))


@needs_torch
@pytest.mark.parametrize("P,N,B,q_rows", [(2, 256, 128, 0), (2, 256, 128, 40), (3, 192, 64, 17)])
def test_mp_triatt(P, N, B, q_rows):
    _report(_mp(P, _entry_triatt, N, 16, 4, B, q_rows))


@needs_torch
@pytest.mark.parametrize("P,N,B", [(2, 256, 128), (3, 240, 16)])
def test_mp_transition(P, N, B):
    _report(_mp(P, _entry_transition, N, 16, B))


def _dense_confidence_reference(pae, pde, contact, asym, has_frame, is_ligand, eps=1e-8):
    """Independent single-device statements of the AF3-style summaries on FULL matrices (the reducer never builds these)."""
    import torch
    from opt_core.mem.rowpair.confidence import tm_bin_centers, tm_bin_weight
    dev = pae.device
    N = pae.shape[0]
    cen = tm_bin_centers(0.0, 32.0, 64)
    p = torch.softmax(pae.unsqueeze(0), dim=-1)                                   # [1,N,N,b]
    uniq = torch.unique(asym)
    remap = {int(u): i for i, u in enumerate(uniq)}
    a_c = torch.tensor([remap[int(x)] for x in asym], device=dev)
    C = len(uniq)
    masks = [a_c == a for a in range(C)]
    hf = has_frame.bool()

    def T_of(mask, n):
        sub = p if mask is None else p[:, mask][:, :, mask]
        return (sub * tm_bin_weight(n, cen, dev)).sum(-1)                           # [1,n,n]

    def ptm_of(T, hfm):
        if int(hfm.sum()) == 0:
            return torch.zeros(T.shape[:-2], device=dev)
        return T.mean(-1)[..., hfm].max(-1).values

    def iptm_of(T, hfm, asub):
        if int(hfm.sum()) == 0:
            return torch.zeros(T.shape[:-2], device=dev)
        diff = asub[None, :] != asub[:, None]
        v = (T * diff).sum(-1) / (eps + diff.sum(-1))
        return v[..., hfm].max(-1).values

    out = {}
    Tf = T_of(None, N)
    out["ptm"], out["iptm"] = ptm_of(Tf, hf), iptm_of(Tf, hf, asym.long())
    chain_ptm = torch.zeros((1, C), device=dev)
    cpi = torch.zeros((1, C, C), device=dev)
    for a in range(C):
        chain_ptm[:, a] = ptm_of(T_of(masks[a], int(masks[a].sum())), hf[masks[a]])
        for b in range(a + 1, C):
            m = masks[a] | masks[b]
            cpi[:, a, b] = iptm_of(T_of(m, int(m.sum())), hf[m], a_c[m])
            cpi[:, b, a] = cpi[:, a, b]
    out["chain_ptm"], out["chain_pair_iptm"] = chain_ptm, cpi
    chain_has_frame = [bool((masks[a] & hf).any()) for a in range(C)]
    chain_iptm = torch.zeros((1, C), device=dev)
    for aid in range(C):
        vals = [cpi[:, i, j] for i in range(C) for j in range(C) if (i == aid or j == aid) and i != j and chain_has_frame[i]]
        if vals:
            chain_iptm[:, aid] = torch.stack(vals, -1).mean(-1)
    out["chain_iptm"] = chain_iptm
    cen_pde = tm_bin_centers(0.0, 32.0, 64).to(dev)
    E = torch.softmax(pde.unsqueeze(0), -1) @ cen_pde                              # [1,N,N]
    out["gpde"] = (E * contact).sum(dim=[-1, -2]) / contact.sum(dim=[-1, -2])
    cg = torch.zeros((1, C), device=dev)
    cpg = torch.zeros((1, C, C), device=dev)
    for a in range(C):
        for b in range(C):
            mc = contact[masks[a], :][:, masks[b]]
            mp = E[..., masks[a], :][..., masks[b]]
            v = (mp * mc).sum(dim=(-1, -2)) / (mc.sum(dim=(-1, -2)) + eps)
            if a == b:
                cg[:, a] = v
            else:
                cpg[:, a, b] = v if a < b else cpg[:, b, a]
    out["chain_gpde"], out["chain_pair_gpde"] = cg, cpg
    out["token_pair_pde_f32"] = E
    return out


def _entry_confidence(N: int, nchains: int, B: int, finish: str, step: int = 40, allow_unsharded: bool = False):
    import torch
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair.confidence import ChainIndex, RowBlockReducer
    P, r = D.world()
    lay = D.Layout.checked(N, P, r, B)
    dev = _dev()
    g = torch.Generator().manual_seed(31)
    pae = (torch.randn((N, N, 64), generator=g) * 2.0).to(dev)
    pde = (torch.randn((N, N, 64), generator=g) * 2.0).to(dev)
    contact = torch.sigmoid(torch.randn((N, N), generator=g)).to(dev)
    sizes = [N // nchains + (1 if i < N % nchains else 0) for i in range(nchains)]
    asym = torch.cat([torch.full((s,), 3 + 2 * i) for i, s in enumerate(sizes)]).to(dev)     # non-contiguous raw ids (remap path)
    has_frame = (torch.rand((N,), generator=g) > 0.3).to(dev)
    is_ligand = (asym == asym.max()) if nchains > 2 else torch.zeros((N,), dtype=torch.bool, device=dev)
    res = {"case": "confidence", "P": P, "N": N, "B": B, "chains": nchains, "finish": finish, "device": str(dev), "checks": {}, "metrics": {}}
    chains = ChainIndex(asym, has_frame, is_ligand)
    red = RowBlockReducer(chains, lay.r0, lay.r1, dev, finish=finish, allow_unsharded=allow_unsharded)
    c = lay.r0
    while c < lay.r1:
        c1 = min(lay.r1, c + step)
        red.consume(c, c1, pae[c:c1], pde[c:c1], contact[c:c1])
        c = c1
    out = red.finalize(lay.bounds)
    if r != 0:
        res["ok"] = out is None
        return res
    ref = _dense_confidence_reference(pae, pde, contact, asym, has_frame, is_ligand)
    # Tolerance classes (stated): the TM family (ptm/iptm/chain_*) of the ``exact`` finish applies the dense statements on same-shape
    # tensors -> asserted within 1e-6 (bit-exact REPORTED; observed bit-exact on CPU fp32 and H100 fp32); the gpde family and the expected-PDE
    # rows go through ``prob @ centers`` per ROW BLOCK (GEMV M = rows*N differs from the dense M = N*N: kernel choice by M on GPU) ->
    # asserted within 1e-6 RELATIVE to the value scale; ``rowsum`` reorders sums -> 2e-5 relative.
    rel = 1e-6 if finish == "exact" else 2e-5
    for k in ("ptm", "iptm", "chain_ptm", "chain_pair_iptm", "chain_iptm", "gpde", "chain_gpde", "chain_pair_gpde"):
        got, exp = out[k].float(), ref[k].float()
        d = (got - exp).abs().max().item()
        scale = max(1.0, float(exp.abs().max())) if exp.numel() else 1.0
        res["metrics"][k] = {"bitwise": bool(torch.equal(got, exp)), "maxabs": d, "rel": d / scale, "value": [round(float(x), 6) for x in exp.flatten()[:4]]}
        res["checks"][k] = d <= rel * scale and got.shape == exp.shape
    if finish == "exact":
        E_ref = ref["token_pair_pde_f32"]
        d = (out["token_pair_pde_f32"] - E_ref).abs().max().item()
        res["metrics"]["token_pair_pde_f32"] = {"bitwise": bool(torch.equal(out["token_pair_pde_f32"], E_ref)), "maxabs": d, "rel": d / float(E_ref.abs().max())}
        res["checks"]["token_pair_pde_f32"] = d <= 1e-6 * float(E_ref.abs().max())
    res["checks"]["f16_full_shapes"] = tuple(out["token_pair_pae_f16"].shape) == (N, N) and tuple(out["contact_probs_f16"].shape) == (N, N)
    res["ok"] = all(res["checks"].values())
    return res


@needs_torch
@pytest.mark.parametrize("P,N,B,nchains", [(2, 256, 128, 1), (2, 300, 64, 3), (3, 384, 128, 4), (4, 512, 64, 5)])
@pytest.mark.parametrize("finish", ["exact", "rowsum"])
def test_mp_confidence(P, N, B, nchains, finish):
    _report(_mp(P, _entry_confidence, N, nchains, B, finish))




def _entry_bcast():
    """Rank 0 holds a nested feature dict (CPU tensors of several dtypes incl. bool + metadata); the others receive it; every leaf then
    checksums equal across ranks (assert_replicated) and a deliberately corrupted leaf is refused by name."""
    import torch
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair.bcast import assert_replicated, broadcast_tensordict, tensordict_checksum
    P, r = D.world()
    dev = _dev()
    g = torch.Generator().manual_seed(5)
    feats = None
    if r == 0:
        feats = {"msa": torch.randint(0, 21, (64, 300), generator=g).to(torch.int8), "coords": torch.randn((300, 37, 3), generator=g),
                 "mask": torch.rand((300,), generator=g) > 0.5, "big": torch.randn((1200, 1200), generator=g),   # > coalesce threshold below
                 "meta": {"name": "case", "n": 300, "tags": ("a", "b")}, "nested": [torch.arange(10), {"e": torch.zeros(0)}]}
    got = broadcast_tensordict(feats, src=0, coalesce_bytes=1 << 20, keep_device=False)
    res = {"case": "bcast", "P": P, "device": str(dev), "checks": {}}
    cs = assert_replicated(got, "features")
    res["checks"]["replicated"] = len(cs) == 6
    res["checks"]["structure"] = isinstance(got["meta"]["tags"], tuple) and got["meta"]["n"] == 300 and got["nested"][1]["e"].numel() == 0
    res["checks"]["dtypes"] = str(got["msa"].dtype) == "torch.int8" and str(got["mask"].dtype) == "torch.bool"
    bad = dict(got)
    bad["coords"] = got["coords"] + (1.0 if r == P - 1 else 0.0)
    try:
        assert_replicated(bad, "features")
        res["checks"]["mismatch_refused"] = False
    except RowpairRefused as e:
        res["checks"]["mismatch_refused"] = "coords" in e.reason
    res["ok"] = all(res["checks"].values())
    return res


@needs_torch
@pytest.mark.parametrize("P", [2, 3])
def test_mp_bcast(P):
    _report(_mp(P, _entry_bcast))


if __name__ == "__main__":
    import inspect
    fails, ran = [], 0
    mod = sys.modules[__name__]
    t_all = time.monotonic()
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if not name.startswith("test_"):
            continue
        marks = getattr(fn, "pytestmark", [])
        params = [m for m in marks if m.name == "parametrize"]
        if any(m.name == "skipif" and m.args and m.args[0] for m in marks):
            print(f"SKIP {name}")
            continue
        combos = [dict()]
        for m in params:
            keys = [k.strip() for k in m.args[0].split(",")]
            combos = [dict(c, **dict(zip(keys, (v if isinstance(v, (tuple, list)) else (v,))))) for c in combos for v in m.args[1]]
        for kw in combos:
            ran += 1
            t0 = time.monotonic()
            try:
                fn(**kw)
                print(f"PASS {name} {kw} {time.monotonic() - t0:.1f}s")
            except BaseException as e:  # noqa: BLE001
                fails.append((name, kw, repr(e)[:400]))
                print(f"FAIL {name} {kw} {e!r}"[:600])
    print(f"SUMMARY ran={ran} failed={len(fails)} wall_s={time.monotonic() - t_all:.0f}")
    for f in fails:
        print("FAILED", f)
    sys.exit(1 if fails else 0)
