"""GPU box (cc >= 8.0): ``ROWPAIR_TRIATT_STAGE=once`` is BITWISE the per-call path for the same kernel -- the flash word (prepared plane) and a
tier word kernels.triattn resolves to triattn_native (the sealed package's member staging hoisted: m1 ``cuda_b`` for 512<=S<=3072 / S>4096 and the
high band ``cuda_c`` for 3072<S<=4096 on cc 9.0, the sm_80 member ``cuda_80`` on cc 8.0) -- on row windows of one plane with padded and dead
rows, both orientations' views, grad mode and ``torch.inference_mode``, a re-stage on a mask change; ``ROWPAIR_TRIATT_BIG=native`` above the
(patched-small) int32 bound serves the tier's triattn_native row on the whole window (per_call vs once bitwise, inside the fast class of the default
q-block flash path); and a per-window wall report per_call vs once (``test_zz_stage_once_gain_per_window``, numbers printed, nothing asserted)."""
from __future__ import annotations

import contextlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
    pytest.skip("needs a cc >= 8.0 CUDA device", allow_module_level=True)

from opt_core.mem.rowpair import triatt as RA                             # noqa: E402
from opt_core.mem.rowpair.evidence import schedule, reset_schedule        # noqa: E402

H, D = 4, 32
DEV = torch.device("cuda:0")
CC = torch.cuda.get_device_capability(0)


def _same(a, b) -> bool:
    return tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and bool(((a == b) | (a.isnan() & b.isnan())).all())


def _plane(g, S):
    tb_cl = (torch.randn(S, S, H, generator=g) * 0.7).to(DEV, torch.bfloat16)          # the gathered bias, channel-last [S, S, H] bf16 (the tp line's tensor)
    return tb_cl.movedim(-1, 0)[None, None], tb_cl.movedim(-1, 0).transpose(-1, -2)[None, None]   # starting / ending views [1, 1, H, S, S] (no copies)


def _windows(g, S, rows, n=3):
    out = []
    for w in range(n):
        q, k, v = ((torch.randn(1, rows, H, S, D, generator=g) * 0.8).to(DEV, torch.bfloat16) for _ in range(3))
        keep = torch.rand(1, rows, 1, 1, S, generator=g) > 0.15
        keep[..., S - 37 * (1 if w < n - 1 else 2):] = False                            # padded keys: the same words for every window but the last (reuse, then ONE re-stage)
        keep[:, -1] = False                                                             # a dead row
        mb = torch.zeros(1, rows, 1, 1, S).masked_fill(~keep, float("-inf")).to(DEV, torch.bfloat16)
        out.append((q, k, v, mb))
    return out


def _ref32(q, k, v, mb, tb, rows=4):
    """fp32 evaluation of the first ``rows`` pair rows (the whole window's logits do not fit at the larger S)."""
    q, k, v, mb = q[:, :rows], k[:, :rows], v[:, :rows], mb[:, :rows]
    a = torch.einsum("bnhqd,bnhkd->bnhqk", q.float() * D ** -0.5, k.float()) + tb.float() + mb.float()
    dead = torch.isinf(mb.float()).all(dim=-1, keepdim=True)
    a = torch.where(dead.expand_as(a), torch.zeros_like(a), a)
    return torch.einsum("bnhqk,bnhkd->bnhqd", torch.softmax(a, -1), v.float())


def _err32(outs, wins, tb, rows=4):
    return max(float((o[:, :rows].float() - _ref32(q, k, v, mb, tb, rows)).abs().max()) for o, (q, k, v, mb) in zip(outs, wins))


def _stock(q, k, v, biases):
    a = torch.einsum("...qd,...kd->...qk", q.float() * D ** -0.5, k.float())
    for b in biases:
        a = a + b.float()
    return torch.einsum("...qk,...kd->...qd", torch.softmax(a, -1), v.float()).to(q.dtype)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for e in (RA.ENV_TRIATT_STAGE, RA.ENV_TRIATT_BIG, RA.ENV_TRIATT_INT32_GUARD, RA.ENV_TRIATT_FLASH_QBLOCK, RA.ENV_TRIATT_CORE):
        monkeypatch.delenv(e, raising=False)
    RA.STAGE.release(); reset_schedule()
    yield
    RA.STAGE.release()


CASES = [("flash_triattn", False, 1024, 8), ("tier:big", False, 1024, 8), ("flash_triattn", True, 1024, 8), ("tier:big", True, 1024, 8),   # m1 band (cc 9.0) / sm_80
         ("tier:big", False, 2048, 16),
         ("tier:big", False, 3762, 16), ("tier:big", True, 3762, 16), ("tier:big", False, 3762, 128),                                # the cuda_c band on cc 9.0
         ("tier:big", False, 4096, 16), ("tier:big", True, 4096, 128),
         ("tier:big", False, 6072, 16), ("tier:big", False, 8192, 16)]                                                                  # m1 again above 4096 / sm_80 at every S


@pytest.mark.parametrize("kernel,inference_mode,S,rows", CASES)
def test_stage_once_is_bitwise_the_per_call_path(kernel, inference_mode, S, rows):
    with (torch.inference_mode() if inference_mode else contextlib.nullcontext()):   # the kits predict under inference_mode (no version counters)
        _once_body(kernel, S, rows)


def _once_body(kernel, S, rows):
    g = torch.Generator().manual_seed(11 + S + rows)
    tb, tbT = _plane(g, S)
    wins = _windows(g, S, rows)
    L = RA.core_ledger(); L.clear()
    core = RA.attention_core(_stock, kernel=kernel)
    ref = {id(p): [core(q, k, v, [mb, p]) for (q, k, v, mb) in wins] for p in (tb, tbT)}
    served_pc = L.served
    assert served_pc == 6, L.fields()
    with RA.STAGE.plane("once"):
        for p in (tb, tbT):
            outs = [core(q, k, v, [mb, p]) for (q, k, v, mb) in wins]
            assert all(_same(a, b) for a, b in zip(outs, ref[id(p)])), [float((a.float() - b.float()).abs().max()) for a, b in zip(outs, ref[id(p)])]
        sch = schedule()
        row = sch.get("triatt_tier_row") if kernel.startswith("tier:") else "flash"
        staged_kind = {"triattn_native": "triattn_native", "flash": ("flash_triattn" if not kernel.startswith("tier:") else None)}.get(row)
        member = RA.native_stage_member(CC, "bf16", D, S) if staged_kind == "triattn_native" else None
        if staged_kind == "triattn_native" and member is None:                             # a native route without a staged member (never at these S on 8.0 / 9.0): aside by name
            assert str(sch.get("triatt_stage_aside", "")).startswith("native_route:"), sch
        elif staged_kind is not None:
            sb = RA.STAGE.staged
            assert sb is not None and sb.kernel == staged_kind and sb.n_served == 3 and RA.STAGE.planes == 2, (getattr(sb, "kernel", None), getattr(sb, "n_served", None), RA.STAGE.planes, sch)
            want = 2 if staged_kind == "triattn_native" else 1                             # native members fold the attended-key set: the last window's other key words = ONE re-stage per plane
            assert sb.n_staged == want and RA.STAGE.restaged >= 1, (sb.n_staged, RA.STAGE.restaged, sch)   # (+ the second plane inside one armed span, counted)
            assert str(sch.get("triatt_stage", "")).startswith("once:"), sch
            assert sch.get("triatt_stage_gib", 0) > 0
            assert "triatt_stage_aside" not in sch, sch
            if staged_kind == "triattn_native":
                assert sch.get("triatt_stage_member") == member, (sch.get("triatt_stage_member"), member)
        else:
            assert str(sch.get("triatt_stage_aside", "")).startswith("unstaged_row"), sch
    assert L.served == 12, L.fields()
    assert RA.STAGE.staged is None
    err = _err32(ref[id(tb)], wins, tb)
    assert err <= 3e-2, err
    print("KERNEL %s S=%d rows=%d cc=%d.%d row=%s member=%s stage_once bitwise OK; maxerr_vs_fp32=%.3e census=%s" % (
        kernel, S, rows, CC[0], CC[1], row, sch.get("triatt_stage_member"), err, {k_: v_ for k_, v_ in sch.items() if str(k_).startswith("triatt_stage")}))


def test_big_native_serves_the_whole_window_above_the_bound(monkeypatch):
    """ROWPAIR_TRIATT_BIG=native with a tier word above the int32 bias bound (patched to just below H*S*S): the tier's triattn_native row on the
    whole window (per call), stage=once bitwise the same, and both inside the fast class of the default q-block flash path."""
    S, rows = 1024, 8
    monkeypatch.setattr(RA, "INT32_MAX", H * S * S - 1)
    monkeypatch.setenv(RA.ENV_TRIATT_FLASH_QBLOCK, "256")
    g = torch.Generator().manual_seed(12)
    tb, tbT = _plane(g, S)
    wins = _windows(g, S, rows, n=2)
    # default word: q-block flash
    L = RA.core_ledger(); L.clear()
    qb = [RA.attention_core(_stock, kernel="tier:big")(q, k, v, [mb, tbT]) for (q, k, v, mb) in wins]
    sch = schedule()
    assert sch.get("triatt_big", "").startswith("qblocks:") and sch.get("triatt_core_tier_big") == "flash_triattn:bias_elems>int32", sch
    # native: whole window through the provider door
    monkeypatch.setenv(RA.ENV_TRIATT_BIG, "native"); reset_schedule()
    core = RA.attention_core(_stock, kernel="tier:big")
    cr = [core(q, k, v, [mb, tbT]) for (q, k, v, mb) in wins]
    sch = schedule()
    if not str(sch.get("triatt_big", "")).startswith("native:"):
        pytest.skip("the tier word does not resolve to triattn_native on this box: %s" % sch)
    assert sch.get("triatt_core_tier_big") == "triattn_native:whole_window", sch
    with RA.STAGE.plane("once"):
        cr1 = [core(q, k, v, [mb, tbT]) for (q, k, v, mb) in wins]
        sb = RA.STAGE.staged
        assert sb is not None and sb.kernel == "triattn_native" and sb.n_served == 2, schedule()
        gib = schedule().get("triatt_stage_gib")
    assert all(_same(a, b) for a, b in zip(cr, cr1))
    e_qb, e_cr = _err32(qb, wins, tbT, rows), _err32(cr, wins, tbT, rows)
    e_x = max(float((a.float() - b.float()).abs().max()) for a, b in zip(cr, qb))
    assert e_qb <= 3e-2 and e_cr <= 3e-2 and e_x <= 3e-2, (e_qb, e_cr, e_x)
    print("BIG native vs qblocks: maxerr_vs_fp32 qblocks=%.3e native=%.3e |native-qblocks|=%.3e staged_gib=%s bytes_formula=%d" % (e_qb, e_cr, e_x, gib, RA.native_staged_bytes(H, S)))


def test_zz_stage_once_gain_per_window():
    """Per-window wall of the tier word's triattn_native row, per_call vs stage once, on the ENDING orientation's view (the transposed plane):
    ``BENCH cc S rows member per_call_ms once_ms gain%`` lines (ROWPAIR_STAGE_BENCH_S / _ROWS / _N override the sizes). Report only."""
    Ss = [int(x) for x in os.environ.get("ROWPAIR_STAGE_BENCH_S", "2048,3762,6072,8192" if CC[0] == 8 else "2048,3762,4096,6072,8192").split(",")]
    rows_l = [int(x) for x in os.environ.get("ROWPAIR_STAGE_BENCH_ROWS", "16,128").split(",")]
    n = int(os.environ.get("ROWPAIR_STAGE_BENCH_N", "6"))
    for S in Ss:
        for rows in rows_l:
            g = torch.Generator().manual_seed(7)
            _tb, tbT = _plane(g, S)
            (q, k, v, mb), = _windows(g, S, rows, n=1)
            core = RA.attention_core(_stock, kernel="tier:big")
            reset_schedule()
            core(q, k, v, [mb, tbT]); core(q, k, v, [mb, tbT])                         # warm (JIT / member resolution / allocator)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(n):
                o_pc = core(q, k, v, [mb, tbT])
            e1.record(); torch.cuda.synchronize()
            ms_pc = e0.elapsed_time(e1) / n
            with RA.STAGE.plane("once"):
                o_once = core(q, k, v, [mb, tbT])                                       # the first window stages
                torch.cuda.synchronize()
                e2, e3 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e2.record()
                for _ in range(n):
                    o_once = core(q, k, v, [mb, tbT])
                e3.record(); torch.cuda.synchronize()
                ms_once = e2.elapsed_time(e3) / n
                sch = schedule()
                member, gib, aside = sch.get("triatt_stage_member"), sch.get("triatt_stage_gib"), sch.get("triatt_stage_aside")
            same = _same(o_pc, o_once)
            print("BENCH cc=%d.%d S=%d rows=%d row=%s member=%s per_call_ms=%.2f once_ms=%.2f gain=%.1f%% staged_gib=%s bitwise=%s%s" % (
                CC[0], CC[1], S, rows, sch.get("triatt_tier_row"), member, ms_pc, ms_once, 100.0 * (ms_pc - ms_once) / ms_pc, gib, same,
                "" if aside is None else " ASIDE=" + str(aside)), flush=True)
            del q, k, v, mb, _tb, tbT, o_pc, o_once
            torch.cuda.empty_cache()


def test_zy_host_issue_time_once_vs_per_call():
    """0.5.220.4: per-window HOST issue time (perf_counter around the python call, no sync) and GPU time of a tier word's triattn_native row,
    per_call vs stage once armed the triatt_update_ way (plane mask + rows, window(w) marks): once must not pay a per-window host readback.
    Sizes ROWPAIR_STAGE_HOST_S (default 8192,16304) x ROWPAIR_STAGE_HOST_ROWS (16,128); the plane mask has a dead tail window; the dead-tail
    window and window 0 are compared per_call vs once (torch.equal). Report lines ``HOST cc S rows ...``."""
    import statistics
    import time
    Ss = [int(x) for x in os.environ.get("ROWPAIR_STAGE_HOST_S", "8192,16304").split(",")]
    rows_l = [int(x) for x in os.environ.get("ROWPAIR_STAGE_HOST_ROWS", "16,128").split(",")]
    for S in Ss:
        g = torch.Generator().manual_seed(5)
        tb_cl = (torch.randn(S, S, H, generator=g) * 0.7).to(DEV, torch.bfloat16)
        tbT = tb_cl.movedim(-1, 0).transpose(-1, -2)[None, None]                          # the ending orientation's view
        keep_plane = (torch.rand(S, S, generator=g) > 0.1)
        keep_plane[:, S - 61:] = False                                                     # padded keys
        for rows in rows_l:
            n_win = int(os.environ.get("ROWPAIR_STAGE_HOST_N", "128" if rows <= 16 else "40"))
            n_win = min(n_win, S // rows)
            R = n_win * rows
            keep = keep_plane[:R].clone()
            keep[R - rows:, :] = False                                                     # the LAST window: all-dead rows (re-stage by name, once)
            keep_d = keep.to(DEV)
            q, k, v = ((torch.randn(1, rows, H, S, D, generator=g) * 0.8).to(DEV, torch.bfloat16) for _ in range(3))
            mbs = [torch.zeros(1, rows, 1, 1, S, device=DEV, dtype=torch.bfloat16).masked_fill(~keep_d[w * rows:(w + 1) * rows][None, :, None, None, :], float("-inf")) for w in range(n_win)]
            core = RA.attention_core(_stock, kernel="tier:big")
            reset_schedule()
            for w in (0, 1, n_win - 1):
                core(q, k, v, [mbs[w], tbT])                                                # warm
            torch.cuda.synchronize()
            # per_call: host issue per window + GPU total
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            issue_pc = []; outs_pc = {}
            e0.record()
            for w in range(n_win):
                t0 = time.perf_counter(); o = core(q, k, v, [mbs[w], tbT]); issue_pc.append((time.perf_counter() - t0) * 1e3)
                if w in (0, n_win - 1):
                    outs_pc[w] = o
            e1.record(); torch.cuda.synchronize()
            gpu_pc = e0.elapsed_time(e1) / n_win
            # once (triatt_update_'s arming: plane mask + rows, window marks)
            issue_once = []; outs_once = {}
            with RA.STAGE.plane("once", mask=keep_d, rows=rows) as slot:
                slot.window(0); core(q, k, v, [mbs[0], tbT]); torch.cuda.synchronize()     # the plane's one staging (+ the key plan's one readback), outside the timed loop
                e2, e3 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e2.record()
                for w in range(n_win):
                    slot.window(w)
                    t0 = time.perf_counter(); o = core(q, k, v, [mbs[w], tbT]); issue_once.append((time.perf_counter() - t0) * 1e3)
                    if w in (0, n_win - 1):
                        outs_once[w] = o
                e3.record(); torch.cuda.synchronize()
                gpu_once = e3 and e2.elapsed_time(e3) / n_win
                sch = schedule()
                ks, kp, rs, member = slot.keysyncs, sch.get("triatt_stage_keyplan"), slot.restaged, sch.get("triatt_stage_member")
            same = all(_same(outs_pc[w], outs_once[w]) for w in outs_pc)
            med = statistics.median
            print("HOST cc=%d.%d S=%d rows=%d n=%d member=%s issue_ms per_call median=%.3f mean=%.3f | once median=%.3f mean=%.3f | gpu_ms per_call=%.2f once=%.2f | keysyncs=%s keyplan=%s restaged=%s bitwise(win0,dead_tail)=%s" % (
                CC[0], CC[1], S, rows, n_win, member, med(issue_pc), sum(issue_pc) / n_win, med(issue_once), sum(issue_once) / n_win, gpu_pc, gpu_once, ks, kp, rs, same), flush=True)
            assert same
            del q, k, v, mbs, outs_pc, outs_once, keep_d
            torch.cuda.empty_cache()
        del tb_cl, tbT
        torch.cuda.empty_cache()
