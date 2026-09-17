#!/usr/bin/env python3
"""Cold-serve cost through the face: the load check + byte gate (face.check) must run ONCE per (process, device); every later call-key miss
(new weights object, new N, new mask geometry) may only pay plan / descriptor creation.
3 closed-form weight sets x 3 N x 2 mask geometries (none, tail3) through one cache: asserts the check body ran exactly once, reports the
cold-serve milliseconds of every new call key (host wall clock around a synchronised serve) next to the warm per-call cost, and fails when a
cold serve after the first exceeds --ceiling-ms (default 50).

    python tests/t_face_coldserve.py [--c 128] [--ceiling-ms 50] [--out /tmp/t_face_coldserve.json]
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "python"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=128)
    ap.add_argument("--ceiling-ms", type=float, default=50.0)
    ap.add_argument("--out", default="/tmp/t_face_coldserve.json")
    a = ap.parse_args()
    import torch
    from trimul_native import face as F, vectors as V
    dev = torch.device("cuda", torch.cuda.current_device())
    rec = {"version": F.VERSION, "device": torch.cuda.get_device_name(dev), "c": a.c, "serves": []}
    cache = {}
    t0 = time.time(); F.check(dev); rec["explicit_check_s"] = round(time.time() - t0, 2)
    runs_after_check = F._STATE.get("check_runs")
    ws = {}
    for salt in (0, 1, 2):
        case = {"c_z": a.c, "c_hidden": a.c, "n": 256, "batch": 1, "direction": "outgoing", "form": "bf16", "residual": True, "mask": "tail3", "variant": "fast"}
        _, _, w = V.closed_form_inputs(case, "cuda", weight_salt=salt)
        ws[salt] = {k: v.to(torch.bfloat16) for k, v in w.items()}
    for n in (256, 320, 403):
      for mask in ("tail3", "none"):
        case = {"c_z": a.c, "c_hidden": a.c, "n": n, "batch": 1, "direction": "outgoing", "form": "bf16", "residual": True, "mask": mask, "variant": "fast"}
        z, m, _ = V.closed_form_inputs(case, "cuda")
        z = z.to(torch.bfloat16)
        for salt in (0, 1, 2):
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                out = F.serve(z, m, direction="outgoing", weights=ws[salt], residual=True, cache=cache, config={"cells": False})
            torch.cuda.synchronize(); cold_ms = (time.perf_counter() - t) * 1e3
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                for _ in range(10):
                    F.serve(z, m, direction="outgoing", weights=ws[salt], residual=True, cache=cache, config={"cells": False})
            torch.cuda.synchronize(); warm_ms = (time.perf_counter() - t) * 1e3 / 10
            rec["serves"].append({"n": n, "mask": mask, "weights": salt, "first_call_ms": round(cold_ms, 3), "warm_call_ms": round(warm_ms, 3), "check_runs": F._STATE.get("check_runs")})
    rec["check_runs_total"] = F._STATE.get("check_runs")
    firsts = [s["first_call_ms"] for s in rec["serves"][1:]]                   # after the very first serve (which builds the op assembly's module cache)
    rec["first_serve_ms"] = rec["serves"][0]["first_call_ms"] if rec["serves"] else None
    rec["cold_serve_ms_new_key_max"] = max(firsts) if firsts else None
    rec["cold_serve_ms_new_key_median"] = sorted(firsts)[len(firsts) // 2] if firsts else None
    rec["cold_serve_ms_new_n"] = max([s["first_call_ms"] for s in rec["serves"][1:] if s["weights"] == 0] or [0.0])          # first call at a new (N, mask) with a known weights object
    rec["cold_serve_ms_new_weights"] = max([s["first_call_ms"] for s in rec["serves"][1:] if s["weights"] != 0] or [0.0])     # first call of a new weights object at a known geometry
    rec["warm_call_ms_median"] = sorted(s["warm_call_ms"] for s in rec["serves"])[len(rec["serves"]) // 2] if rec["serves"] else None
    rec["ceiling_ms"] = a.ceiling_ms
    rec["ok"] = bool(rec["check_runs_total"] == 1 == runs_after_check) and bool(firsts) and max(firsts) < a.ceiling_ms
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("version", "device", "explicit_check_s", "check_runs_total", "ok", "first_serve_ms", "cold_serve_ms_new_key_max", "cold_serve_ms_new_key_median", "cold_serve_ms_new_n", "cold_serve_ms_new_weights", "warm_call_ms_median", "ceiling_ms")}))
    for s_ in rec["serves"]:
        print("  n=%d mask=%s weights=%d first=%.3f ms warm=%.3f ms check_runs=%s" % (s_["n"], s_["mask"], s_["weights"], s_["first_call_ms"], s_["warm_call_ms"], s_["check_runs"]))
    print("[t_face_coldserve] %s -> %s" % ("OK" if rec["ok"] else "FAIL", a.out))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
