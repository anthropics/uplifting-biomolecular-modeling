#!/usr/bin/env python3
"""Memory behaviour of face.serve under the workspace policy (transient workspaces above 256 MiB per (unit, Np) pair):
(1) allocated-after-call delta (torch.cuda.memory_allocated before the call vs after the output is freed) at (64,64) N=2000 and (128,128)
    N=1536 (both above the threshold): the first call may leave weight-derived state (<= output bytes + 1 MiB), a repeated call leaves <= 1 MiB;
(2) an N sweep 512..2048 at one width retains nothing per call (inputs and the freed output excluded);
(3) a multi-N process (5 distinct N x 2 widths x outgoing/incoming) shows no accumulation;
(4) a small N under the threshold may keep its workspace pair cached: the retained bytes stay <= 256 MiB.
python tests/t_face_memory.py [--out /tmp/t_face_memory.json]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "python"))
MiB = float(1 << 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/t_face_memory.json")
    a = ap.parse_args()
    import torch
    from trimul_native import face as F, vectors as V
    dev = torch.device("cuda", torch.cuda.current_device())
    rec = {"version": F.VERSION, "device": torch.cuda.get_device_name(dev), "cc": "%d.%d" % torch.cuda.get_device_capability(dev), "legs": {}}
    F.check(dev)                                                             # load + gate once, outside the measured region
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    ok = True

    def one(cz, ch, n, direction, cache, weights=None, residual=False):
        case = {"c_z": cz, "c_hidden": ch, "n": n, "batch": 1, "direction": direction, "form": "bf16", "residual": residual, "mask": "tail3", "variant": "fast"}
        z, m, w = V.closed_form_inputs(case, "cuda")
        if weights is not None:
            w = weights
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(dev)
        with torch.no_grad():
            out = F.serve(z, m, direction=direction, weights=w, residual=residual, cache=cache, config={"cells": False})
        torch.cuda.synchronize()
        out_bytes = out.numel() * out.element_size()
        with_out = torch.cuda.memory_allocated(dev)
        del out
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated(dev)
        del z, m
        return {"before_mib": before / MiB, "after_call_mib": after / MiB, "delta_mib": (after - before) / MiB, "with_output_mib": (with_out - before) / MiB, "output_mib": out_bytes / MiB}, w

    # (1) large-N single calls above the threshold: nothing but weight-derived state stays behind
    leg1 = {}
    for cz, ch, n in ((64, 64, 2000), (128, 128, 1536)):
        cache = {}
        r0, w = one(cz, ch, n, "outgoing", cache)                            # first call: weight pack + plan (small, persistent by design)
        r1, _ = one(cz, ch, n, "outgoing", cache, weights=w)                 # second call, same key: the delta is the workspace retention alone
        r1["first_call_delta_mib"] = r0["delta_mib"]
        r1["ok"] = bool(r1["delta_mib"] <= 1.0 and r0["delta_mib"] <= r0["output_mib"] + 1.0)
        leg1["z%d_h%d_N%d" % (cz, ch, n)] = r1
        ok &= r1["ok"]
    rec["legs"]["large_n_delta"] = leg1
    # (2) N sweep at one width: allocated-after-call must not grow with N
    cache, w, sweep = {}, None, []
    for n in (512, 768, 1024, 1536, 2048):
        r, w = one(128, 128, n, "outgoing", cache, weights=w)
        sweep.append({"n": n, "after_call_mib": round(r["after_call_mib"], 2), "delta_mib": round(r["delta_mib"], 3)})   # after_call includes this call's inputs (they scale with N)
    keep_mib = 256.0                                                        # the policy: a (unit, Np) pair under this total may stay cached; above it nothing is retained
    for pt in sweep:
        np_ = -(-pt["n"] // 16) * 16
        pt["pair_mib"] = round(6 * 128 * np_ * np_ / MiB, 1)
        pt["bound_mib"] = (keep_mib if pt["pair_mib"] <= keep_mib else 1.0) + (8.0 if pt is sweep[0] else 0.0)   # first call: weight-derived state
        pt["ok"] = bool(pt["delta_mib"] <= pt["bound_mib"])
    growth = max(pt["delta_mib"] for pt in sweep if pt["pair_mib"] > keep_mib)   # retained per call above the threshold (inputs and the freed output excluded)
    rec["legs"]["n_sweep_z128"] = {"points": sweep, "growth_mib": round(growth, 3), "ok": bool(all(pt["ok"] for pt in sweep) and growth <= 1.0)}
    ok &= rec["legs"]["n_sweep_z128"]["ok"]
    # (3) multi-N process: 5 N x 2 widths x 2 directions through one cache per width -> no accumulation
    torch.cuda.synchronize(); start = torch.cuda.memory_allocated(dev)
    caches, ws, trace = {64: {}, 128: {}}, {64: None, 128: None}, []
    for n in (896, 1152, 1408, 1664, 1920):                                  # all above the keep threshold for both widths
        for c in (64, 128):
            for d in ("outgoing", "incoming"):
                r, ws[c] = one(c, c, n, d, caches[c], weights=ws[c])
                trace.append(round(r["after_call_mib"], 2))
    torch.cuda.synchronize(); end = torch.cuda.memory_allocated(dev)
    rec["legs"]["multi_n_process"] = {"calls": len(trace), "allocated_start_mib": round(start / MiB, 2), "allocated_end_mib": round(end / MiB, 2),
                                     "accumulated_mib": round((end - start) / MiB, 3), "after_call_trace_mib": trace, "ok": bool((end - start) / MiB <= 8.0)}
    ok &= rec["legs"]["multi_n_process"]["ok"]
    # (4) small N under the threshold: the pair may stay cached, bounded by 256 MiB
    cache = {}
    r0, w = one(128, 128, 256, "outgoing", cache)
    r1, _ = one(128, 128, 256, "outgoing", cache, weights=w)
    kept = r0["delta_mib"]
    rec["legs"]["small_n_cached"] = {"z128_h128_N256_first_call_delta_mib": round(kept, 3), "second_call_delta_mib": round(r1["delta_mib"], 3), "ok": bool(kept <= 256.0 and r1["delta_mib"] <= 1.0)}
    ok &= rec["legs"]["small_n_cached"]["ok"]
    rec["ok"] = bool(ok)
    json.dump(rec, open(a.out, "w"), indent=1)
    l1 = rec["legs"]["large_n_delta"]
    print("[t_face_memory] %s: z64_h64 N2000 delta %.3f MiB (output %.0f MiB, first call %.3f); z128_h128 N1536 delta %.3f MiB (output %.0f MiB); sweep retained max %.3f MiB; multi-N accumulated %.3f MiB over %d calls; small-N kept %.1f MiB -> %s" % (
        "OK" if ok else "FAIL", l1["z64_h64_N2000"]["delta_mib"], l1["z64_h64_N2000"]["output_mib"], l1["z64_h64_N2000"]["first_call_delta_mib"], l1["z128_h128_N1536"]["delta_mib"], l1["z128_h128_N1536"]["output_mib"],
        rec["legs"]["n_sweep_z128"]["growth_mib"], rec["legs"]["multi_n_process"]["accumulated_mib"], rec["legs"]["multi_n_process"]["calls"], kept, a.out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
