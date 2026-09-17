"""opt_core.mem.rowpair — GPU numerics + NCCL launcher tests (a 2-GPU box; H100:2 for numbers that are quoted). Reuses the entries of
``test_rowpair_logic`` on CUDA with realistic channel counts and the engines' dtypes: fp32 (TF32 off), fp32 (TF32 on), bf16.

Assertion classes (stated; implemented in ``test_rowpair_logic._cmp`` / ``_ok``): data movement BIT-EXACT; statements: (a) max |sharded -
dense| within the dtype's rounding class (fp32: 1e-4 x ref scale; bf16: 2e-2 x ref scale), (b) elementwise |sharded - ref| <= atol +
rtol |ref| with torch.testing's per-dtype (rtol, atol) against the fp64 reference (two roundings of one fp64 value may differ by an ulp, so
the elementwise test is not taken against the dense dtype result), (c) the RATIO test: sharded-vs-fp64 error <= 2 x dense-vs-fp64 error +
1e-6 (sharding may not make a statement less accurate than its dense form). The bit-exact flag and the fraction of differing elements vs
dense are REPORTED per case (the qualification's tolerance-table input; cuBLAS may pick kernels by M, so bit-exact is not asserted).

Run: ``python tests/test_rowpair_numerics_gpu.py`` on a box with >= 2 CUDA devices (rc != 0 on failure; RESULT lines are the record).
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import test_rowpair_logic as L  # noqa: E402
from opt_core.mem.rowpair import NGpuRefused, launch, refuse_unless_visible, visible_gpus  # noqa: E402


def _gpu(P, entry, *args, **kw):
    return launch.run_sharded(P, entry, *args, mode="big", nccl_timeout_s=300, run_timeout_s=1200, **kw)


def _entry_binding():
    import torch
    from opt_core.mem.rowpair import dist as D
    P, r = D.world()
    x = torch.full((3,), float(r), device="cuda")
    lay = D.Layout(3 * P, P, r, B=3)
    g = D.all_gather_rows(x, lay)
    return {"rank": r, "P": P, "current_device": torch.cuda.current_device(), "backend": torch.distributed.get_backend(),
            "gather_ok": bool(torch.equal(g.cpu(), torch.arange(P).repeat_interleave(3).float()))}


def _entry_nccl_fail():
    import torch
    from opt_core.mem.rowpair import dist as D
    P, r = D.world()
    if r == 1:
        raise RuntimeError("rank one fails before the collective")
    lay = D.Layout(256, P, r)
    D.all_gather_rows(torch.zeros((lay.R, 4), device="cuda"), lay)     # rank 0 blocks here until torn down / NCCL timeout
    return "unreachable"


def _with_tf32(flag: bool, entry, *args):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = bool(flag)
    torch.backends.cudnn.allow_tf32 = bool(flag)
    return entry(*args)


def _b_for(N, P, B):
    """The case's row-block size, lowered for higher P so the grid shards (Layout.auto's candidates <= B); refused shapes stay refused."""
    from opt_core.mem.rowpair.dist import Layout
    try:
        return Layout.auto(N, P, 0, candidates=tuple(b for b in (128, 64, 32, 16) if b <= B)).B
    except Exception:  # noqa: BLE001
        return B


def main():
    import torch
    results, fails = [], []
    K = visible_gpus()
    print(f"visible_gpus={K} names={[torch.cuda.get_device_name(i) for i in range(K)]} torch={torch.__version__}")
    assert K >= 2, "needs >= 2 CUDA devices"
    # refusal with the REAL device count
    try:
        refuse_unless_visible(K + 1)
        fails.append("visible refusal did not fire")
    except NGpuRefused as e:
        assert e.reason == f"refused: n_gpu={K + 1} visible={K}", e.reason
        print("PASS refusal", e.reason)
    # binding + NCCL
    b = _gpu(2, _entry_binding)
    assert b["backend"] == "nccl" and b["current_device"] == 0 and b["gather_ok"], b
    print("PASS nccl binding", b)
    # failure teardown under NCCL
    t0 = time.monotonic()
    try:
        _gpu(2, _entry_nccl_fail)
        fails.append("nccl failure not raised")
    except launch.RankFailed as e:
        assert e.rank == 1 and e.event == "rank_failed", (e.rank, e.event)
        print(f"PASS nccl rank failure teardown in {time.monotonic() - t0:.1f}s")
    assert time.monotonic() - t0 < 120
    import multiprocessing
    assert not multiprocessing.active_children()

    def run(name, P, entry, *args):
        t1 = time.monotonic()
        try:
            res = _gpu(P, entry, *args)
            res["wall_s"] = round(time.monotonic() - t1, 1)
            res["name"] = name
            print("RESULT " + json.dumps(res, sort_keys=True, default=str))
            results.append(res)
            if not res.get("ok"):
                fails.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name} {e!r}"[:2000])
            fails.append(name)

    for P in [p for p in (2, 4, 8) if p <= K]:                     # every power-of-two rank count the box can hold
        for case in ("gathers", "transpose", "ring"):
            for (N, B) in ((512, 128), (1000, 64)):
                run(f"dist/{case}/P{P}/N{N}/B{B}", P, L._entry_dist, case, N, B)
        for tf32 in (False, True):
            for dt in ("float32", "bfloat16"):
                if dt == "bfloat16" and tf32:
                    continue
                for (N, C, B, RA, RB) in ((256, 128, 128, 128, 128), (512, 128, 128, 128, 64), (1024, 64, 128, 256, 128)):
                    run(f"trimul/P{P}/N{N}/C{C}/RA{RA}/RB{RB}/{dt}/tf32={tf32}", P, _with_tf32, tf32, L._entry_trimul, N, C, _b_for(N, P, B), min(RA, N // P), RB, dt)
        for dt in ("float32", "bfloat16"):
            for (N, C, H, B, q) in ((256, 128, 4, 128, 0), (256, 128, 4, 128, 64), (512, 64, 4, 128, 128)):
                run(f"triatt/P{P}/N{N}/C{C}/H{H}/q{q}/{dt}", P, _with_tf32, False, L._entry_triatt, N, C, H, _b_for(N, P, B), q, dt)
            for (N, C, B) in ((512, 128, 128), (1000, 128, 64)):
                run(f"transition/P{P}/N{N}/C{C}/{dt}", P, _with_tf32, False, L._entry_transition, N, C, _b_for(N, P, B), dt)
        for finish in ("exact", "rowsum"):
            for (N, nch, B) in ((512, 3, 128), (1000, 5, 64)):
                run(f"confidence/P{P}/N{N}/chains{nch}/{finish}", P, _with_tf32, False, L._entry_confidence, N, nch, _b_for(N, P, B), finish)
        run(f"bcast/P{P}", P, _with_tf32, False, L._entry_bcast)
    # tolerance table
    print("TOLTABLE " + json.dumps([{k: r.get(k) for k in ("name", "ok", "metrics")} for r in results], default=str))
    print(f"SUMMARY ran={len(results) + 3} failed={len(fails)}")
    for f in fails:
        print("FAILED", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())


def test_gpu_numerics_is_a_script_for_a_multi_gpu_box():      # pytest collects this file; the numerics run is the script above, not a unit test
    import pytest
    pytest.skip("script: `python tests/test_rowpair_numerics_gpu.py` on an H100!:P box (P>=2); see opt_core/mem/rowpair/API.md")
