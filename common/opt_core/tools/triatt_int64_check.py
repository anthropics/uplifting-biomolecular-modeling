#!/usr/bin/env python
"""triatt_int64_check.py — one-GPU check of the row-pair triangle-attention core above / below the int32 bias-element bound (opt_core 0.5.18.8, KT18).

Builds synthetic q/k/v ``[1, rows, H, S, D]`` (``--dtype``), an additive key-mask bias ``[1, rows, 1, 1, S]`` (``--mask-frac`` of the keys dropped) and a
triangle bias handed the way the tp_rowpair kits hand it (a NON-contiguous ``[1, 1, H, S, S]`` view of a channel-last ``[S, S, H]`` tensor), then runs

  flash            ``opt_core.mem.rowpair.triatt.attention_core(stock, kernel="flash_triattn")`` as shipped — above the bound (H*S*S > 2**31-1; H=4: S > 23,170)
                   the kernel is launched per query block of ``ROWPAIR_TRIATT_FLASH_QBLOCK`` (``--qblock``); ``ROWPAIR_TRIATT_INT32_GUARD=1`` in the
                   environment restores the <= 0.5.18.7 refusal (the run then reports ``served=0 fallback_by=unsupported:bias_elems>int32`` and flash == stock)
  flash_unblocked  the same adapter with the bound patched away: ONE launch over the whole plane (what 0.5.18.7 does below the bound; possible while
                   S*ceil16(S) < 2**31, i.e. S <= ~46,300) — ``flash`` must equal it BITWISE (query blocking is an M-only change)
  flash_blocked    the same adapter with the bound patched BELOW this plane: the query-blocked launches forced at any S — below the bound it must equal
                   ``flash`` (= one launch) BITWISE
  stock            the engine statement (the engine's ``_attention``: matmul, += biases, softmax, matmul in the input dtype) over ``--stock-qblock`` query blocks
  ref64            the materialised float64 reference (query-blocked to bound memory)

and prints one JSON line per measurement (``KT18_ROW {...}``): ms per call (CUDA events, ``--repeat`` after one warm-up), max|d| / mean|d| of flash vs
stock (fp32 of the dtype outputs), rel-RMS of flash and of stock vs ref64 and their ratio (the kit's Tier-2 criterion: ratio <= 1), bitwise flags,
the core ledger's served / fallback census. ``--save`` writes the flash output (+ args) to a .pt; ``--compare`` checks bitwise equality against one
(pre/post-change runs: point ``--core`` at another opt_core checkout).

  python tools/triatt_int64_check.py --S 16384 --rows 64                       # below the bound: flash == flash_unblocked == 0.5.18.7 bitwise
  python tools/triatt_int64_check.py --S 24576 --rows 64 --qblock 2048         # above: served per query block; vs stock within Tier 2; == unblocked bitwise
  ROWPAIR_TRIATT_INT32_GUARD=1 python tools/triatt_int64_check.py --S 24576    # the opt-out: refused by name, stock serves
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--S", type=int, default=24576)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--qblock", type=int, default=None, help="ROWPAIR_TRIATT_FLASH_QBLOCK for this run (default: env / 2048)")
    ap.add_argument("--stock-qblock", type=int, default=256, help="query block of the stock statement and of ref64 (bounds the logits transient)")
    ap.add_argument("--mask-frac", type=float, default=0.05, help="fraction of keys dropped by the key-mask bias (production single-query inference: 0 = all ones, still passed)")
    ap.add_argument("--prescaled", type=int, default=1, help="1 (production, tp_rowpair): q arrives PRE-SCALED by D**-0.5 in its dtype and the core gets scale=1.0; 0: scale inside")
    ap.add_argument("--bias", default="view", choices=["view", "view_fp32", "transposed", "hnn_fp32"],
                    help="view (the row-pair TP kits): the NON-contiguous [1,1,H,S,S] view of the channel-last [S,S,H] bias in z's dtype; "
                         "view_fp32: that view of an fp32 channel-last tensor (the TP line's gathered bias is fp32: linear_z runs in fp32); "
                         "transposed: a TP kit's ENDING node (transpose_bias=True) = that view with its two token dims swapped; "
                         "hnn_fp32: opt_core bias_hnn(tb, float32) = a contiguous fp32 [1,1,H,S,S] (4*H*S*S bytes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--modes", default="flash,flash_unblocked,flash_blocked,stock,ref64")
    ap.add_argument("--core", default=None, help="path of the opt_core checkout to import (default: the one this script lives in)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    core_dir = os.path.abspath(args.core or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, core_dir)
    if args.qblock is not None:
        os.environ["ROWPAIR_TRIATT_FLASH_QBLOCK"] = str(int(args.qblock))
    import torch
    import opt_core
    from opt_core.mem.rowpair import triatt as RA

    dev = torch.device("cuda")
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    S, R, H, D = args.S, args.rows, args.heads, args.D
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    g = torch.Generator(device=dev).manual_seed(args.seed)
    q = torch.randn(1, R, H, S, D, device=dev, dtype=dt, generator=g)
    k = torch.randn(1, R, H, S, D, device=dev, dtype=dt, generator=g)
    v = torch.randn(1, R, H, S, D, device=dev, dtype=dt, generator=g)
    tb_dt = torch.float32 if args.bias in ("view_fp32", "hnn_fp32") else dt
    tb_cl = torch.empty(S, S, H, device=dev, dtype=tb_dt)                                        # channel-last [S, S, H], the gathered bias of the tp line (filled in row
    for i0 in range(0, S, 4096):                                                                  #  chunks: no whole-plane fp32 transient at S = 60,120)
        i1 = min(S, i0 + 4096)
        tb_cl[i0:i1] = (0.5 * torch.randn(i1 - i0, S, H, device=dev, dtype=torch.float32, generator=g)).to(dt).to(tb_dt)   # bf16-representable values in either dtype
    if args.bias in ("view", "view_fp32"):
        tb = tb_cl.movedim(-1, 0).unsqueeze(0).unsqueeze(0)                                      # [1, 1, H, S, S] NON-contiguous view (the TP kits' tp_rowpair/pairstack.attend)
    elif args.bias == "transposed":
        tb = tb_cl.movedim(-1, 0).transpose(-1, -2).unsqueeze(0).unsqueeze(0)                    # a TP kit's ending node: permute_final_dims(., (2, 1, 0)) — token dims swapped
    else:
        tb = RA.bias_hnn(tb_cl, torch.float32)                                                    # [1, 1, H, S, S] contiguous fp32 (opt_core bias_hnn)
    keep = torch.rand(1, R, 1, 1, S, device=dev, generator=g) >= args.mask_frac
    mask_bias = torch.zeros(1, R, 1, 1, S, device=dev, dtype=dt).masked_fill(~keep, float("-inf"))   # additive key-mask bias inf*(mask-1): 0 keep / -inf drop
    biases = [mask_bias, tb]
    if args.prescaled:                                                                            # tp_rowpair: mha._prep_qkv scales q in its dtype; the core gets scale=1.0
        q = q * (float(D) ** -0.5)
        scale, core_scale = 1.0, 1.0
    else:
        scale, core_scale = float(D) ** -0.5, None
    big = H * S * S > 2 ** 31 - 1
    qblock_eff = getattr(RA, "flash_qblock", lambda: None)()             # None on a core that predates 0.5.18.8 (--core at the release head)
    meta = dict(tag=args.tag, S=S, rows=R, H=H, D=D, dtype=args.dtype, big=bool(big), qblock=qblock_eff, stock_qblock=args.stock_qblock,
                int32_guard_env=os.environ.get("ROWPAIR_TRIATT_INT32_GUARD", ""), opt_core=opt_core.__version__, core_dir=core_dir,
                torch=str(torch.__version__), device=str(torch.cuda.get_device_name(0)), seed=args.seed, mask_frac=args.mask_frac,
                prescaled=int(args.prescaled), bias_form=args.bias)
    try:
        import triton
        meta["triton"] = str(triton.__version__)
    except Exception:  # noqa: BLE001
        meta["triton"] = None

    def stock(q_, k_, v_, bs):                       # the engine's primitives.attention._attention (use_high_precision False): the dtype's matmul, `a += b` IN PLACE
        a = torch.matmul(q_ * scale, k_.transpose(-1, -2))   #  (an fp32 bias is added into the dtype's logits without promoting them, as the engine's statement does), softmax, matmul
        for b in bs:
            a += b
        a = torch.softmax(a, dim=-1)
        return torch.matmul(a, v_)

    def ref64_fn(q_, k_, v_, bs):
        a = torch.matmul(q_.double() * scale, k_.double().transpose(-1, -2))
        for b in bs:
            a = a + b.double()
        return torch.matmul(torch.softmax(a, dim=-1), v_.double())

    def timed(fn, repeat):
        fn()                                         # warm-up (JIT compile, allocator)
        torch.cuda.synchronize()
        ts = []
        out = None
        for _ in range(max(1, repeat)):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            out = fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return out, sorted(ts)[len(ts) // 2], ts

    def emit(**row):
        d = dict(meta)
        d.update(row)
        print("KT18_ROW " + json.dumps(d, sort_keys=True), flush=True)

    outs = {}
    L = RA.core_ledger()
    if "flash" in modes:
        L.clear()
        core = RA.attention_core(stock, kernel="flash_triattn", stock_qblock=args.stock_qblock, scale=core_scale)
        torch.cuda.reset_peak_memory_stats()
        o, ms, ts = timed(lambda: core(q, k, v, biases), args.repeat)
        outs["flash"] = o
        ncalls = 1 + max(1, args.repeat)                                                          # warm-up + timed calls through the adapter
        emit(mode="flash", ms=ms, ms_all=ts, calls=ncalls, served=L.served, fallbacks=dict(L.fallbacks), ledger_calls=L.calls,
             census_exact=bool(L.served + sum(L.fallbacks.values()) == ncalls), shapes=dict(L.shapes),
             peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30, finite=bool(torch.isfinite(o.float()).all()))
    if "flash_unblocked" in modes:
        SKp = -(-S // 16) * 16
        free_b = torch.cuda.mem_get_info()[0]
        need_b = (4 + 4 + 2 + (4 if tb.dtype != torch.float32 or not tb.is_contiguous() else 0)) * H * S * SKp
        if S * SKp >= 2 ** 31:
            emit(mode="flash_unblocked", skipped="S*ceil16(S) >= 2**31: the kernel's own int32 row-pitch bound (one launch impossible)")
        elif need_b > 0.92 * free_b:
            emit(mode="flash_unblocked", skipped="whole-plane copies (%.1f GB) exceed free memory (%.1f GB) on this card" % (need_b / 2 ** 30, free_b / 2 ** 30))
        else:
            L.clear()
            old = RA.INT32_MAX
            RA.INT32_MAX = 2 ** 62                   # the adapter's bound patched away: one launch over the whole plane (0.5.18.7's path below the bound)
            try:
                core_u = RA.attention_core(stock, kernel="flash_triattn", stock_qblock=args.stock_qblock, scale=core_scale)
                torch.cuda.reset_peak_memory_stats()
                o, ms, ts = timed(lambda: core_u(q, k, v, biases), args.repeat)
            finally:
                RA.INT32_MAX = old
            outs["flash_unblocked"] = o
            emit(mode="flash_unblocked", ms=ms, ms_all=ts, served=L.served, fallbacks=dict(L.fallbacks), peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                 finite=bool(torch.isfinite(o.float()).all()))
    if "flash_blocked" in modes and hasattr(RA, "flash_qblock"):              # the query-blocked launch FORCED at any S (the bound patched below this plane): must equal `flash` bitwise below the bound
        L.clear()
        old = RA.INT32_MAX
        RA.INT32_MAX = min(old, H * S * S - 1)
        try:
            core_b = RA.attention_core(stock, kernel="flash_triattn", stock_qblock=args.stock_qblock, scale=core_scale)
            torch.cuda.reset_peak_memory_stats()
            o, ms, ts = timed(lambda: core_b(q, k, v, biases), args.repeat)
        finally:
            RA.INT32_MAX = old
        outs["flash_blocked"] = o
        emit(mode="flash_blocked", ms=ms, ms_all=ts, served=L.served, fallbacks=dict(L.fallbacks), peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
             finite=bool(torch.isfinite(o.float()).all()))
    if "stock" in modes:
        torch.cuda.reset_peak_memory_stats()
        o, ms, ts = timed(lambda: RA.attend_query_blocks(stock, q, k, v, biases, args.stock_qblock), args.repeat)
        outs["stock"] = o
        emit(mode="stock", ms=ms, ms_all=ts, peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30, finite=bool(torch.isfinite(o.float()).all()))
    if "ref64" in modes:
        rq = max(16, min(args.stock_qblock, int(6e8 // max(1, R * H * S)) // 16 * 16))
        t0 = time.time()
        outs["ref64"] = RA.attend_query_blocks(ref64_fn, q, k, v, biases, rq)
        torch.cuda.synchronize()
        emit(mode="ref64", ms=(time.time() - t0) * 1e3, ref_qblock=rq)

    def dstats(a, b):
        d = (a.float() - b.float()).abs()
        return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), bitwise=bool(torch.equal(a, b)))

    def relrms(a, ref):
        a, ref = a.double(), ref.double()
        return float((a - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())

    cmp = {}
    if "flash" in outs and "stock" in outs:
        cmp["flash_vs_stock"] = dstats(outs["flash"], outs["stock"])
    if "flash" in outs and "flash_unblocked" in outs:
        cmp["flash_vs_unblocked"] = dstats(outs["flash"], outs["flash_unblocked"])
    if "flash" in outs and "flash_blocked" in outs:
        cmp["flash_vs_blocked"] = dstats(outs["flash"], outs["flash_blocked"])
    if "ref64" in outs:
        for m in ("flash", "flash_unblocked", "stock"):
            if m in outs:
                cmp["relrms64_" + m] = relrms(outs[m], outs["ref64"])
                cmp["maxabs64_" + m] = float((outs[m].double() - outs["ref64"]).abs().max())
        if "flash" in outs and "stock" in outs:
            cmp["tier2_ratio_relrms"] = cmp["relrms64_flash"] / max(cmp["relrms64_stock"], 1e-30)
            cmp["tier2_ratio_maxabs"] = cmp["maxabs64_flash"] / max(cmp["maxabs64_stock"], 1e-30)
    if args.compare and "flash" in outs:
        ref = torch.load(args.compare, map_location="cpu", weights_only=False)      # our own file (tensor + plain meta)
        other = ref["flash"].to(dev)
        cmp["compare_file"] = args.compare
        cmp["compare_meta"] = {k_: ref["meta"].get(k_) for k_ in ("opt_core", "core_dir", "tag", "qblock")}
        cmp["flash_vs_file"] = dstats(outs["flash"], other)
    emit(mode="compare", **cmp)
    if args.save and "flash" in outs:
        torch.save({"flash": outs["flash"].cpu(), "meta": meta}, args.save)
        emit(mode="saved", path=args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
